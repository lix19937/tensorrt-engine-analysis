---
name: tensorrt-engine-analysis
description: 用来校验和分析 TensorRT 的性能数据。它接收成对的「算子/层信息 json」和「性能/耗时 json」，验证两者有效且来自同一模型，推断模型基本信息，找出可能的算子融合（fusion）或延迟优化机会，最终产出一份 Markdown 性能报告或结构化 json.
---

# Tensorrt engine Analysis

## scripts 目录说明    
```
./scripts/                   # 只读
├── run.sh                   # 只读，Unix-like系统运行包装脚本，自动发现 Python 3.8+
├── analyze_trt_perf.py      # 只读，分析脚本，校验和分析engine对应的 `layers_*.json` 和 `profile_*.json` 两个文件，会生成 analysis.json
├── package_report.py        # 只读，打包脚本，会生成 analyze-data.json
├── trt_perf/                # 只读              
|
├── validate_analyze_data.py # 只读
└── visualize_layer_info.py  # 只读
```

> 运行 Python 脚本时，Unix-like 系统使用 `scripts/run.sh`作为运行器（下文记为 `<runner>`）。包装脚本会尽力自动发现 Python 3.8+；可通过设置环境变量 `SKILL_PYTHON` 指定 Python 可执行文件以覆盖自动发现。占位符请替换为对应平台的原生路径。

## 工作流     
```
analysis.json     ---|   
                     |  ----->  analyze.md
analyze-data.json ---|  
```

1. 选择输入   
优先咨询用户，等待用户输入目录      
使用**直接存放** `layers_*.json` 或 `profile_*.json` 文件的文件夹作为输入。不要对仅包含子模块文件夹、而不直接包含这些 json 的父目录调用打包脚本。针对由独立编码器、Transformer、解码器、VAE 等模块组成的模型套件，需分别处理每个模块文件夹。

2. 推断模型身份与组成      
优先使用用户明确给出的模型名称，通过 `--model-name` 参数传入； 
若无明确名称，则基于配置文件、输入目录名、层/性能文件名称自动推断。推断可信度较低时留空。输出的结构化 json 会在顶层节点、每个校验通过后端的 `model` 对象中记录推断出的模型信息。

3. 执行确定性分析   
调用 `scripts/analyze_trt_perf.py`, 进行**数据完整性验证**，然后分析。该脚本仅依赖 Python 内置库，提取结构化性能数据并序列化为 `analysis.json` 文件。该文件为权威输出，供校验、打包、AI 诊断使用。 

4. 解读前先检查验证结果       
+ 当分析脚本能够输出结构化验证数据时，分析脚本会以0退出，即使一个或多个后端验证失败。  
  + 当分析脚本执行完返回值为`0`情况下，读取生成 json 中的 `validation` 对象，筛选可用的后端性能报告。 
  + 若分析脚本执行完返回值为非`0`，说明无法生成结构化数据，终止流程；  

+ 若格式验证步骤非0退出，代表生成的 json 契约不符合规范，终止流程。  
+ 当后端验证状态为 `passed`、分析模式为 `layer_profile`、层名称与性能日志名称一一匹配时，才可解读该后端的延迟与性能指标（热点算子、层类型耗时占比、GEMM tactic 等）。这些性能指标按 `Name` 将层与性能日志逐条配对得出，不依赖计算图结构。
+ 若 `graph.ambiguous` 为 `true` 或 `graph.is_dag` 为 `false`（即存在张量多生产者、自环或非 DAG，`validation.messages` 中会有对应 `warning`），仍可解读上述延迟/性能指标，但**依赖计算图结构的结论必须降级**：数据流与生产者→消费者依赖、输入形状推断（`model.input_shapes`）、引擎划分推断、以及算子融合线索均不可靠，须在报告中标注为低可信度并建议人工复核，可引用 `graph.duplicate_producers` 说明歧义来源。
+ `layer_only` 模式后端仅支持图结构、算子信息查看，无法解读延迟与性能指标。

5. 理解模型整体信息后，再撰写 AI 诊断      
以生成的`analysis.json` 文件作为事实依据，撰写结论前，清晰梳理模型/组件范围、不同后端差异、计算图结构、耗时分布、高耗时算子/热点算子、算子融合线索与注意事项。算子/层信息json 中的每一个层都有`Metadata`字段，如果其内容（对应onnx layer）如果不为空，可在分析热点算子与算子融合的时候一并加入到`analyze.md`，网页端展示的「Summary」面板由 `analyze-data.json` 自动生成，禁止将该标准化摘要复制到 `analyze.md`。      

+ 最终 AI 诊断报告分为两个固定章节：    
  1. `## Summary`（摘要）：简洁罗列高可信度客观结论、优先级最高的优化方向。
  2. `## Details`（详细分析）：性能解读、多后端横向对比、优化猜想、优先级排序的验证实验、人工分析判断，不可单纯机械复述分析脚本输出数据。     

+ 输出最终报告前，需为每一个新建/更新的报告文件夹生成 `analyze.md` 诊断文件，文件内容严格匹配当前报告范围。单模型多后端对比场景，可仅生成一份 `analyze.md` 完成多后端横向分析；多模型模块、多份独立报告场景，需为每个模块单独生成诊断文件，禁止将套件级总分析内容复制到每一份模块报告。   

+ `analyze.md` 中不得重复写入校验清单、完整耗时表格、高耗时算子排行榜、文件路径打印等可由前端页面从 `analyze-data.json` 自动渲染的内容。仅在支撑新结论、提出后续验证实验时，简短引用关键指标或算子名称。   

6. 打包分析报告   
+ 调用打包脚本 `scripts/package_report.py`。单次调用仅能基于一个输入目录（存放了 `layers_*.json`/`profile_*.json` ）生成一个报告文件夹。  
打包脚本会将相关内容写入 `analyze-data.json`、校验数据格式、复制网页前端模板并保留生成的 json。输出根目录从用户当前工作区读取，而非当前skill目录子路径；若仅能获取skill目录路径，则省略 `--output-parent` 参数，脚本自动回落至系统临时目录保存报告。   

+ 单次需求生成多份报告时，建议显式指定 `--report-dir` 参数，使报告文件夹名称携带模型/模块标识，避免使用自动递增后缀区分文件。若需要套件整体总结，在输出根目录单独生成一份总览 Markdown 文件，不要将总览写入每一份模块报告的 `analyze.md`。

7. 仅预览打包完成后的最终报告文件夹    
用户需要网页预览时，提供打包后的报告文件夹，不要直接提供技能目录。`visualize_layer_info.py` 脚本仅用于独立调试算子信息 HTML 页面，不属于标准报告打包流程，前端页面不依赖该脚本。

### 通用脚本/命令    
`<runner>` 在 Unix-like 系统为 `<skill-dir>/scripts/run.sh`，在 Windows 为 `<skill-dir>\scripts\run.cmd`。
```bash
# Validate or inspect inputs. Folder mode discovers matching backends; repeat --data for explicit backends.
<runner> <skill-dir>/scripts/analyze_trt_perf.py <model-folder> --output <analysis.json>

<runner> <skill-dir>/scripts/analyze_trt_perf.py --data <layers-a.json> <profile-a.json> --data <layers-b.json> --output <analysis.json>

# Create a report. Use --report-dir for component-specific reports and --analyze-md after writing final AI analysis.
<runner> <skill-dir>/scripts/package_report.py <model-folder> --output-parent <workspace-folder> [--model-name <model>] [--analyze-md <final-analysis.md>]

<runner> <skill-dir>/scripts/package_report.py --analyze-data <report-folder>/analyze-data.json [--analyze-md <final-analysis.md>]

# web preview.
<runner> -m http.server 8765 --bind 127.0.0.1 --directory <report-folder>
```
然后打开 `http://127.0.0.1:8765/`.

## 输入  
每个推理引擎/性能报告对应一组文件：一份算子/层信息 json + 一份性能延迟 json，两份 json来自同一个 TensorRT 引擎文件。文件夹可存放多组配对文件，示例如下：
  `layers_torch-trt-aot.json` 匹配 `profile_torch-trt-aot.json`
  `layers_onnx-tensorrt.json` 匹配 `profile_onnx-tensorrt.json`

+ 算子/层信息json 文件每条记录应包含字段：`Name`、`LayerType`、`Inputs`、`Outputs`、`Metadata`      
+ 性能日志每条记录应包含字段：`name`、`timeMs`、`averageMs`、`medianMs`、`percentage`；文件首行允许存在 `{ "count": 20 }`    

为报告推导算子类别`Type`。默认直接使用原始 `LayerType`；若算子名称、算法策略、元数据可明确识别特殊算子，则做分类覆盖。已定义特殊分类规则如下：
- `kgen_mha`: a raw `LayerType` of `kgen` that is actually an MHA/FMHA-style attention layer.
- `misc`: raw `LayerType` values `reshape`, `shape_call`, `signal`, and `wait` are grouped into this category.

## 模型类别  
Use this starter category list and revise as evidence warrants:
- Transformer encoder / sentence embedding
- Transformer decoder / LLM
- Vision transformer
- CNN / convolutional vision model
- Diffusion / U-Net style model
- Recommender / ranking model
- Classical ML or feature pipeline
- Unknown DL model

Report category only when the evidence is strong. Good evidence includes Hugging Face config fields, names like `encoder.layer`, attention layers, input names such as `input_ids`, or repeated convolution/GEMM patterns. Say "unknown" instead of over-claiming.

## Analysis Heuristics
Prioritize issues by likely runtime impact and confidence:
- Hot layers: individual layers with high `percentage` or `averageMs`.
- Layer-type concentration: high aggregate time in `kgen`, shape, cast, gather, reshape, or pointwise/reduction layers may indicate fusion or dynamic-shape overhead.
- Fusion clues: for transformer encoders, check whether Q/K/V matmuls are fused per block, whether attention is an obvious fused SDPA/FMHA layer, and whether layer-norm/residual/GELU patterns are fused into larger kernels.
- Engine partitioning: one engine is usually ideal. Use adjacent `.trt`/`.engine` files, `_subgraph`, and `StreamId` as hints; state when this is only inferred.
- GEMM tactics: GEMM layers without Tensor Core or xMMA-like tactic names may deserve inspection.
- Many tiny kernels: if no single layer dominates but many small kernels add up, rank launch/fusion overhead above isolated micro-hotspots.

报告中严格区分客观事实与优化猜想。每一条优化方向均需引用 JSON 中的对应指标作为证据，并给出后续验证步骤或对比实验方案。
