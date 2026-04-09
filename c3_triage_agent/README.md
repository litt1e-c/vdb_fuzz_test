# C3 Triage Agent

这个目录现在只保留了 `autonomous` 模式需要的最小文件集。

它的目标很单纯：

> 输入一段 `fuzzer mismatch`，让 agent 自己搜索历史 issue、运行脚本、判断是否重复根因。

## 三分钟上手

最常用的流程只有 3 步。

### 1. 配 `.env`

如果你当前用的是 iFlow：

```env
C3_API_KEY=your_iflow_key
C3_BASE_URL=https://apis.iflow.cn/v1
C3_WIRE_API=chat_completions
C3_DISABLE_RESPONSE_STORAGE=true
C3_REASONING_EFFORT=xhigh
C3_VERBOSITY=high
```

### 2. 先测接口

```bash
python c3_triage_agent/provider_probe.py \
  --model qwen3-coder-plus \
  --endpoint chat \
  --payload-mode both
```

如果这里不通，就先修 API，不要先改 agent。

### 3. 再跑 agent

```bash
python c3_triage_agent/autonomous_triage_runner.py \
  --report-file c3_triage_agent/incidents/qdrant_test71_mismatch.txt \
  --provider openai \
  --model qwen3-coder-plus \
  --max-steps 6
```

## 你平时真正会用到的命令

正常 triage：

```bash
python c3_triage_agent/autonomous_triage_runner.py \
  --report-file c3_triage_agent/incidents/qdrant_test71_mismatch.txt \
  --provider openai \
  --model qwen3-coder-plus \
  --max-steps 6
```

严格测“自主发现新问题”的能力：

```bash
python c3_triage_agent/autonomous_triage_runner.py \
  --report-file c3_triage_agent/incidents/qdrant_test71_mismatch.txt \
  --provider openai \
  --model qwen3-coder-plus \
  --run-mode discovery \
  --max-steps 6
```

盲测“独立发现能力”：

```bash
python c3_triage_agent/autonomous_triage_runner.py \
  --report-file c3_triage_agent/incidents/qdrant_test71_mismatch.txt \
  --provider openai \
  --model qwen3-coder-plus \
  --history-prompt-mode hidden \
  --max-steps 6
```

如果你既想隐藏摘要，又想彻底禁止它访问历史 issue，优先用：

```bash
--run-mode discovery
```

只看 prompt，不真正调用模型：

```bash
python c3_triage_agent/autonomous_triage_runner.py \
  --report-file c3_triage_agent/incidents/qdrant_test71_mismatch.txt \
  --provider openai \
  --model qwen3-coder-plus \
  --dry-run
```

## 我该怎么换自己的 case

最简单的方式就是新建一个文本文件，里面放一段 mismatch，再把 `--report-file` 指过去。

例如：

```bash
python c3_triage_agent/autonomous_triage_runner.py \
  --report-file /path/to/your_mismatch.txt \
  --provider openai \
  --model qwen3-coder-plus \
  --history-prompt-mode hidden \
  --max-steps 6
```

你不需要先写 JSON manifest。

## 当前保留的文件

- [autonomous_triage_runner.py](/home/caihao/compare_test/c3_triage_agent/autonomous_triage_runner.py)
  主入口。负责给模型发消息、执行模型要求的命令、保存全过程记录。

- [AUTO_AGENT_PROMPT.md](/home/caihao/compare_test/c3_triage_agent/AUTO_AGENT_PROMPT.md)
  自主 triage 用的系统提示词。

- [providers.py](/home/caihao/compare_test/c3_triage_agent/providers.py)
  provider 适配层。现在主要支持 OpenAI-compatible `chat/completions`，也保留 `responses` 和 Anthropic 兼容逻辑。

- [provider_probe.py](/home/caihao/compare_test/c3_triage_agent/provider_probe.py)
  接口探针。先测 API/provider 是否正常，再测 agent，避免把接口问题误认为 agent 问题。

- [history_issue_catalog.json](/home/caihao/compare_test/c3_triage_agent/history_issue_catalog.json)
  历史 issue 的简短索引。可选择直接放进 prompt，也可隐藏后让 agent 自己搜。

- [incidents/qdrant_test71_mismatch.txt](/home/caihao/compare_test/c3_triage_agent/incidents/qdrant_test71_mismatch.txt)
  一个现成的测试输入，用来评估 agent 的自主发现能力。

- [auto_cases](/home/caihao/compare_test/c3_triage_agent/auto_cases)
  agent 如果要自动生成 case 文件，会写到这里。
  这是运行时输出目录，可以随时清空。

- [repros](/home/caihao/compare_test/c3_triage_agent/repros)
  agent 如果要自动生成临时复现脚本，会写到这里。
  这是运行时输出目录，可以随时清空。

- [runs](/home/caihao/compare_test/c3_triage_agent/runs)
  每次运行的完整记录目录。
  这是运行时输出目录，可以随时清空。

- [.env.example](/home/caihao/compare_test/c3_triage_agent/.env.example)
  环境变量模板。

- [.env](/home/caihao/compare_test/c3_triage_agent/.env)
  你自己的真实配置。

## `hidden` 和 `discovery` 的区别

`--history-prompt-mode hidden` 的意思是：

- prompt 里不会直接给历史 issue 摘要
- 只会告诉它 `history_bug_root` 路径
- 它需要自己用 `rg`、`ls`、`python` 去搜索和验证

`--run-mode discovery` 的意思更严格：

- 不给历史摘要
- 不给 `history_bug_root`
- 不允许命令访问 `history_find_bug`
- 工作目录限制在 `c3_triage_agent`
- 它只能自己写 `repros/` 里的脚本，再自己运行和分析

如果你要评估“它能不能自己从报错出发构造复现并定位问题”，更推荐 `discovery`。

如果你只是想隐藏某个特定历史 issue，也可以继续用：

```bash
--exclude-history 8617
```

## agent 现在到底能不能自己执行代码

可以。

当前 agent 的工作方式是：

1. 模型先返回 JSON，里面带 `actions`
2. runner 执行这些 `actions`
3. 把脚本输出再喂回模型
4. 模型继续分析，直到给出最终结论

当前支持的 action 类型只有两种：

- `run`
- `write_file`

当前允许的命令前缀是：

- `python`
- `python3`
- `rg`
- `sed`
- `cat`
- `head`
- `tail`
- `ls`
- `find`

在 `--run-mode discovery` 下会更严格：

- 工作目录变成 `c3_triage_agent`
- 只能运行 `python repros/...`、`python auto_cases/...`、`python incidents/...`
- 只能对 `repros/`、`auto_cases/`、`incidents/` 做 `ls/find/cat/head/tail`
- 禁止访问 `history_find_bug`

所以它现在已经具备：

- 自己搜索历史 issue
- 自己运行 POC
- 自己读取脚本内容
- 根据输出继续分析

## 运行后看哪里

每轮运行都会在 `runs/<timestamp>/` 下保存：

- `runner_config.json`
  这轮运行的 provider、model、history 模式等配置

- `report_snapshot.txt`
  原始 mismatch 输入

- `system_prompt.md`
  发给 agent 的系统提示词

- `prompt_step_0_user.md`
  第一次发给 agent 的用户消息

- `prompt_step_N_user.md`
  后续每一步的跟进消息

- `provider_request_step_N.json`
  每一步真实发给模型接口的请求快照

- `response_step_N.json`
  每一步模型返回的结构化 JSON

- `transcript.jsonl`
  system / user / assistant / action_results 的完整往返记录

- `analysis_trace.md`
  把每一步的 `summary / analysis / actions` 汇总成易读文本

- `commands/`
  agent 要求执行的每条命令及输出

- `final_verdict.json`
  最终判断结果

如果你只想快速看“它有没有真的执行脚本并读输出”，优先看：

- `commands/*.json`
- `analysis_trace.md`
- `transcript.jsonl`

这三个通常就够了。

## 当前设计结论

这套目录现在更适合做两件事：

1. 测 provider 是否稳定
2. 测 agent 是否能独立发现“这是重复问题还是新问题”
