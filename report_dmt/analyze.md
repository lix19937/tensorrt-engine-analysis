# dmt — TensorRT 性能诊断（backend: tofp16）

## Summary

- **模型判定（高可信）**：这是一个**多模态 BEV 多任务感知模型**（自动驾驶域），而非纯 Transformer。证据来自层名前缀与输入：相机分支 `img_backbone`/`img_neck`、LiDAR 分支 `lidar_backbone_od`/`lidar_backbone_ld` + `ld_VFEScatter`/`od_VFEScatter` 体素散射插件、`neck_lss/view_transformer`（LSS 视图变换 + deformable attention）、以及 `map_head`/`fusion_object_head`/`lane_mask_head`/`qm_map_head` 等多个任务头。输入张量含 `voxels/voxel_coords/voxel_num_points`（点云）、`img`(7×3×544×960，多相机)、`sd_map`/`osm_mask`（地图先验）。脚本自动填入的 `category=Transformer`、`hidden_size=256`、`intermediate_size=100000` 属误判，不应采信。
- **验证通过、指标可用**：`layer_profile` 模式，层与性能日志按 `Name` 逐条配对（1309↔1309），延迟/占比指标可解读。
- **计算图歧义（关键注意）**：`graph.ambiguous=true`（18 个张量存在多生产者，多来自 `bev_decoder/ScatterND` 与 `__myln_k_arg__` 融合中间量）。因此**数据流、输入形状推断、引擎划分、图级融合线索均为低可信度，需人工复核**。
- **无单点热层**（高可信）：最大单层仅 3.7%，性能呈“长尾均摊”形态，优化重心应放在**层类别聚集**与**融合/reshape 开销**，而非单个算子。
- **最高优先级方向**：`kgen`（reshape/cast/transpose/move 等访存类）占比最高，且集中在 deformable-attention 与 BEV decoder 的形状搬运上——这是最值得投入的融合/布局优化点。

## Details

### 1. 后端与耗时结构
单后端 `tofp16`，平均单次推理约 **42.1 ms**。三大类别占据绝大部分耗时：`kgen` 35.8%、`correlation`（卷积/implicit-gemm）31.4%、`gemm` 24.4%；`kgen_mha`（融合注意力）仅 2.0%。按标签聚合，`matmul/gemm` 与 `attention` 相关计算已高度落在 Tensor Core 路径上（见下），说明**主要计算算子本身效率尚可，瓶颈更多在访存与算子编排**。

### 2. 高耗时观察（客观事实）
- 最大单层是 `map_head` deformable decoder 的 `value_proj` MatMul（3.67%），tactic 为 `cutlass3x_sm100_tensorop_...f16`（Blackwell/SM100 Tensor Core，FP16）——已是理想内核，无直接优化空间。
- 排名靠前的 `kgen` 层几乎全是形状搬运：`__myl_ReshMove`、`__myl_Resh`、`__myl_Cast`、`__myl_GridCastReshTran`、`__myl_MulMinMaxRounCast`（对应 `img_QuantizeLinear`）。其 `Metadata` 指向 `neck_lss/.../deformable_attention/Reshape*` 与 `map_head/bev_decoders.0` 的 Transpose/Reshape 链。
- 相机主干首层 `conv_act_pool`（2.20%，tactic `sm100_conv_act_pool_v5 ... s8 ... NDHWC`，INT8）与 LiDAR 主干 `conv1`（1.76%，SM100 implicit-gemm FP16）是各自 backbone 的入口大层，属预期热点。

### 3. 计算精度与 tactic（客观事实，修正脚本提示）
脚本 `issues` 报了“95 个 GEMM 的 tactic 不像 Tensor Core”。**逐条核对原始 tactic 后未获证实**：全部 449 个 `gemm` 层均为 `cutlass3x_sm100_tensorop`（Tensor Core）；卷积类中仅 9 个走旧式 `sm80_xmma_fprop_implicit_gemm`（集中在 `lane_mask_head/sdmap_reduce`、`neck_lss/raster_encoder` 等小分支），单层占比均 <0.08%，合计影响可忽略。**结论：GEMM tactic 无需作为优化重点**，脚本该提示可视为误报。

### 4. 引擎划分（低可信，须复核）
脚本据 StreamId 推断 6 个引擎。实测层分布为 stream0=1283、其余 stream 各 3–9 层，且这些非 0 流层为 `signal`/`wait`/plugin/散射等同步与插件算子。因此**更可能是单主引擎 + 少量多流同步**，而非 6 个真正的引擎切分。叠加计算图歧义，**“引擎划分”结论不可作为优化依据**，如需确认应核对相邻 `.engine`/`_subgraph` 产物与构建日志。

### 5. 优化猜想（每条附证据 + 验证实验，按优先级）

1. **deformable-attention / BEV-decoder 的 reshape-transpose 访存融合**（优先级最高）
   - 证据：`kgen` 类 35.8% 为最大类别；`__myl_Resh_myl2_12/13`、`__myl_ReshMove_myl8_821`、`__myl_GridCastReshTran` 等单层各占 1.2–2.0%，`Metadata` 均落在 `deformable_attention/Reshape*` 与 `bev_decoders.0` 的 Transpose 链。
   - 验证：对比 ONNX 图中 deformable attention 前后的 Reshape/Transpose，尝试在导出侧消除冗余转置或改用 channels-last 布局；重建 engine 后观察 `kgen` 聚合占比与上述层是否下降。

2. **量化边界的 Cast/Quantize 开销**（中优先级）
   - 证据：`__myl_MulMinMaxRounCast_myl2_14`（`img_QuantizeLinear`，1.88%）与多个 `__myl_Cast`（各 ~1.5%）位于相机分支 INT8 量化入口，是 FP↔INT8 转换的显式代价。
   - 验证：检查相机 backbone 是否可将 Q/DQ 折叠进上游算子、或扩大 INT8 覆盖范围以摊薄转换成本；对比开启前后该组 `kgen` 层耗时。

3. **VFEScatter 体素散射插件**（中低优先级，观察项）
   - 证据：`ld_VFEScatter`(1.19%)+`od_VFEScatter`(0.90%) 为 PluginV2，占比不高但为自定义内核，属潜在长尾。
   - 验证：确认插件是否已用最新实现/是否存在与前后算子的 host 同步；如占比稳定可暂不投入。

4. **长尾小核聚集**（框架性）
   - 证据：`positives` 显示无单层 >5%，但 31 个零耗时层 + 大量 <0.5% 小核；kernel launch/编排开销可能被稀释在众多小 `kgen`/`misc` 中。
   - 验证：优先推进方向 1 的融合以减少 kernel 数量，再用 Nsight 观察 launch 间隙是否成为次要瓶颈。

> 说明：本报告的延迟/占比/ tactic 均来自 `analyze-data.json` 逐层配对结果；计算图相关推断（数据流、输入形状、引擎划分、图级融合）因 `graph.ambiguous=true` 标注为低可信度，建议结合 ONNX 原图与构建日志人工复核。
