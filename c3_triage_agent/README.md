# C3 Triage Agent

这个目录现在只保留了 `autonomous` 模式需要的最小文件集。

它的目标很单纯：

> 输入一段 `fuzzer mismatch`，让 agent 自己搜索历史 issue、运行脚本、判断是否重复根因。

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

- [repros](/home/caihao/compare_test/c3_triage_agent/repros)
  agent 如果要自动生成临时复现脚本，会写到这里。

- [runs](/home/caihao/compare_test/c3_triage_agent/runs)
  每次运行的完整记录目录。

- [.env.example](/home/caihao/compare_test/c3_triage_agent/.env.example)
  环境变量模板。

- [.env](/home/caihao/compare_test/c3_triage_agent/.env)
  你自己的真实配置。

## 推荐的 `.env`

如果你当前用的是 iFlow：

```env
C3_API_KEY=your_iflow_key
C3_BASE_URL=https://apis.iflow.cn/v1
C3_WIRE_API=chat_completions
C3_DISABLE_RESPONSE_STORAGE=true
C3_REASONING_EFFORT=xhigh
C3_VERBOSITY=high
```

## 先测接口

先不要急着跑 agent，先测 provider：

```bash
python c3_triage_agent/provider_probe.py \
  --model qwen3-coder-plus \
  --endpoint chat \
  --payload-mode both
```

如果这一步不通，优先看 API 和模型，不要先怀疑 agent。

## 再测 agent

正常测试：

```bash
python c3_triage_agent/autonomous_triage_runner.py \
  --report-file c3_triage_agent/incidents/qdrant_test71_mismatch.txt \
  --provider openai \
  --model qwen3-coder-plus \
  --max-steps 6
```

## 测“独立发现能力”

如果你不想在 prompt 里提前告诉它历史 bug 摘要，而是希望它自己去搜：

```bash
python c3_triage_agent/autonomous_triage_runner.py \
  --report-file c3_triage_agent/incidents/qdrant_test71_mismatch.txt \
  --provider openai \
  --model qwen3-coder-plus \
  --history-prompt-mode hidden \
  --max-steps 6
```

这个模式下：

- prompt 里不会直接给历史 issue 摘要
- 只会告诉它 `history_bug_root` 路径
- 它需要自己用 `rg`、`ls`、`python` 去搜索和验证

如果你只是想隐藏某个特定历史 issue，也可以继续用：

```bash
--exclude-history 8617
```

## 运行后会记录什么

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

## 当前设计结论

这套目录现在更适合做两件事：

1. 测 provider 是否稳定
2. 测 agent 是否能独立发现“这是重复问题还是新问题”

