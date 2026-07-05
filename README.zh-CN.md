# HMS Shadow

[English](README.md) | [中文](README.zh-CN.md)

HMS Shadow 是一个可复现的长期记忆问答实验框架。它用于测试：
在 memory retrieval 之后加入 answer-time evidence organization，是否能让
语言模型更稳定地基于检索到的记忆完成推理。

项目聚焦 LongMemEval 场景。在这类问题中，答案可能需要同时利用多个
session、多个时间点、抽取后的 memory facts，以及原始 source snippets。

## 实验设计

整个实验被拆成四个阶段：

```text
Question
  -> retrieve existing memories
  -> organize retrieved evidence
  -> generate a grounded answer
  -> evaluate the answer
```

核心思想是：不要直接把松散的 retrieved facts 交给回答模型，而是在生成前
构造一个中间证据结构，把时间、来源、事件状态和数值信号显式组织出来。

这套设置适合研究以下问题：

- 模型能否跨 session 连接证据
- 模型能否区分旧状态和当前状态
- 模型能否把相对日期定位到具体 memories
- 模型能否避免重复计数
- 当数值比较缺少一侧证据时，模型是否能保守处理

## 可视化 Demo

项目包含一个不依赖数据库的可视化 demo，方便外部读者先理解整体架构。它展示了
原始 retrieved sessions 如何在 answer generation 前被组织成 evidence
ledger。

![Memory evidence organization demo](docs/assets/memory_pipeline_demo.svg)

可以直接在浏览器中打开静态页面：

```text
docs/memory_pipeline_demo.html
```

这个页面不需要模型 key、数据库访问或 benchmark 运行产物。

## 动态单题回放

仓库还包含一个具体的 benchmark-style 单题回放。它展示一个 multi-session
问题，并动态呈现散落的 session snippets 如何经过 retrieval、evidence
ledger construction、deduplication，最后进入 grounded answer generation。

![Dynamic benchmark case replay](docs/assets/benchmark_case_replay.svg)

可以直接打开自动播放的回放页面：

```text
docs/benchmark_case_replay.html
```

回放页面会自动推进同一个案例的 raw session snippets、recall candidates、ledger
rows、duplicate-control rule、answer packet，以及最终 grounded response。

## 实验管线

benchmark 脚本暴露两种 pipeline mode。

### Ledger Pipeline

Ledger pipeline 保持 memory retrieval 不变，在 answer generation 前加入
结构化证据账本。

对于高风险问题类型，ledger 会组织：

- event time
- mention time
- source session 或 document
- fact type
- compact evidence text
- numeric、date、update signals
- raw source snippets，用于 grounding

当你想复现主线 evidence organization 实验时，使用这个模式。

### Self-Evolution Pipeline

Self-evolution pipeline 保留 ledger pipeline，并额外加入一个轻量的
answer-time controller。controller 来自错误模式诊断，主要覆盖：

- count 和 total 去重
- relative-date lookup grounding
- amount 和 difference calibration
- current 与 previous state arbitration

这个模式用于研究：在 retrieval 之后加入有针对性的控制指令，是否会改变或
改善长期记忆推理行为。

## 目录结构

```text
.
├── .aaaSCRIPT/
│   └── run_benchmark.sh
├── core/
│   ├── dataplane/
│   ├── daemon/
│   └── local-suite/
├── deploy/
├── docs/
│   ├── assets/
│   │   ├── benchmark_case_replay.svg
│   │   └── memory_pipeline_demo.svg
│   ├── benchmark_case_replay.html
│   └── memory_pipeline_demo.html
├── interface/
├── lab/
│   └── evaluation/
│       └── benchmarks/
│           ├── common/
│           │   └── benchmark_runner.py
│           └── longmemeval/
│               └── longmemeval_benchmark.py
├── tooling/
├── .env.example
├── README.md
└── README.zh-CN.md
```

关键文件：

- `.aaaSCRIPT/run_benchmark.sh`：统一实验入口脚本
- `docs/benchmark_case_replay.html`：自动播放的单题过程回放页面
- `docs/assets/benchmark_case_replay.svg`：README 中直接展示的动态单题回放
- `docs/memory_pipeline_demo.html`：静态 before/after 可视化页面
- `docs/assets/memory_pipeline_demo.svg`：README 中直接展示的架构示意图
- `lab/evaluation/benchmarks/longmemeval/longmemeval_benchmark.py`：LongMemEval pipeline 实现
- `lab/evaluation/benchmarks/common/benchmark_runner.py`：共享 evaluation runner
- `.env.example`：本地配置模板

## 环境配置

创建本地环境文件：

```bash
cp .env.example .env
```

然后填入自己的数据库和模型服务配置。

框架会从 `.env` 加载配置。不要把凭证硬编码到源码里。

## 复现逻辑

benchmark 脚本默认使用 retrieval-only 模式：

```text
HMS_RETRIEVAL_ONLY=1
```

在这个模式下，memory extraction 和 ingestion 会被跳过。实验直接使用配置
数据库中已经存在的 memory units。这样可以保证 retrieval 和 answer-time
evidence organization 的实验更一致，也更快。

推荐复现流程：

```text
1. 在 .env 中准备数据库和模型配置
2. 确认 memory database 中已经有需要的 memory units
3. 选择 pipeline mode
4. 运行 LongMemEval 脚本
5. 查看 ignored runtime directories 下生成的本地运行产物
```

## 运行 Ledger Pipeline

```bash
export HMS_RETRIEVAL_ONLY=1
export HMS_PIPELINE=ledger
export HMS_MAX_INSTANCES=500
export HMS_SESSION_EXPANSION_WEIGHT=0.5

bash .aaaSCRIPT/run_benchmark.sh \
  --parallel 8 \
  --max-concurrent-questions 8 \
  --eval-semaphore-size 8 \
  --quiet
```

## 运行 Self-Evolution Pipeline

```bash
export HMS_RETRIEVAL_ONLY=1
export HMS_PIPELINE=self_evolution
export HMS_MAX_INSTANCES=500
export HMS_SESSION_EXPANSION_WEIGHT=0.5

bash .aaaSCRIPT/run_benchmark.sh \
  --parallel 8 \
  --max-concurrent-questions 8 \
  --eval-semaphore-size 8 \
  --quiet
```

## 常用运行参数

常用环境变量：

- `HMS_PIPELINE`：`ledger` 或 `self_evolution`
- `HMS_RETRIEVAL_ONLY`：设为 `1` 时跳过 ingestion，复用已有 memories
- `HMS_MAX_INSTANCES`：限制评测问题数量
- `HMS_MAX_QUESTIONS`：在筛选后继续限制问题数量
- `HMS_DATASET_PATH`：指定本地 LongMemEval 数据集路径
- `HMS_SESSION_EXPANSION_WEIGHT`：覆盖 session expansion weight
- `HMS_PYTHON_BIN`：指定 Python 解释器

常用命令行参数：

- `--parallel`：并发处理的 instance 数量
- `--max-concurrent-questions`：question-level 最大并发数
- `--eval-semaphore-size`：evaluator 并发限制
- `--category`：只运行指定 LongMemEval 类别
- `--question-id`：运行一个或多个指定 question IDs
- `--skip-ingestion`：跳过 ingestion，使用数据库中的已有 memories
- `--quiet`：减少控制台输出

## 本地运行产物

实验运行时，本地运行产物会写入 ignored directories：

```text
.aaaLOG/
.aaaRESULT/
```

这些目录只用于本地复现，不应提交。
