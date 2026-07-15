# Self Agent

一个本地优先、可长期运行的个人智能体。它不是只会回复的聊天机器人，而是围绕 **长期记忆、主动交互、可信任控制** 三件事做深：记得住你说过的重要信息，能在合适时机主动跟进，也允许你随时查看、修改和删除它记住的东西。

当前形态：Telegram 对话入口 + 本地 SQLite 记忆库 + 精确提醒 + 主动心跳 + 工具/MCP/Skill 扩展 + FastAPI 控制台。

## 特点

- **长期记忆不是简单存聊天记录**  
  对话先进入 `pending_intake`，再经过时间归一化、画像 resolve、叙事去重、情绪抽取、承诺分类，最后投影成可注入 prompt 的 `MEMORY.md`。

- **主动消息有生命周期和边界**  
  开放回路按 `open -> sent -> done` 流转，主动心跳会综合活跃时段、冷却、每日预算、情绪状态和重复消息 guard，尽量做到“有用但不烦”。

- **精确提醒走确定性调度**  
  用户明确说“几点提醒我”时进入 `reminder_job`，不交给 LLM 再判断；支持启动恢复、misfire 补偿、失败重试和 delivery log。

- **工具调用可扩展、可追踪、可确认**  
  内置工具、MCP、Skill 都统一注册到工具表；危险工具先进入确认门，用户确认后才执行，并留下审计记录。

- **记忆可见、可改、可忘**  
  本地控制台可以查看画像、叙事、情绪、提醒、trace、工具调用和矛盾队列。长期 agent 不应该是黑盒。

## 功能概览

| 模块 | 能力 |
|---|---|
| 对话入口 | Telegram bot，支持用户白名单；CLI 可本地调试 |
| 记忆系统 | 画像、偏好、事件、情绪、开放回路、向量召回、MemoryGate 注入门控 |
| 主动引擎 | 到期承诺跟进、低频开放回路、情绪调制、频率预算、trace |
| 精确提醒 | SQLite 持久化任务、APScheduler 调度、恢复与补偿 |
| 工具系统 | builtin tools、MCP 热更新、Skill 渐进披露、危险工具确认门 |
| 控制台 | 查看/编辑/遗忘记忆，处理矛盾，查看 turn/tool/tick trace |
| 部署 | Docker Compose，适合 VPS / NAS / 家用小主机长期运行 |

## 快速开始

推荐先用 Docker 跑长期版本。

```bash
git clone https://github.com/Yikinge/self-agent.git
cd self-agent

cp .env.example .env
```

编辑 `.env`，至少填写：

```env
TELEGRAM_BOT_TOKEN=你的 Telegram Bot Token
TELEGRAM_ALLOW_FROM=你的 Telegram 用户名
DEEPSEEK_API_KEY=你的模型 API Key
```

启动：

```bash
docker compose up -d --build
docker compose logs -f agent
```

控制台默认只绑定本机：

```text
http://127.0.0.1:8787
```

如果部署在服务器上，建议通过 SSH 隧道访问：

```bash
ssh -L 8787:localhost:8787 user@your-server
```

## 本地 CLI

CLI 不需要 Telegram，适合测试记忆、工具和主动心跳。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp config.example.toml config.toml
python -m agent.cli
```

常用命令：

```bash
python -m agent.cli --script
python -m agent.cli --facts
python -m agent.cli --recall "健身"
python -m agent.cli --consolidate
python -m agent.cli --memory-md
python -m agent.cli --tick --force
python -m agent.cli --tools
python -m agent.cli --console
```

没有配置模型 API key 时，网关会使用离线兜底逻辑，能跑通基本流程；配置真实模型后，记忆抽取和回复质量会明显更好。

## 架构

```text
Telegram / CLI
      │
      ▼
Orchestrator.run_turn
      │
      ├── MemoryService / Consolidator
      │     ├── pending_intake
      │     ├── profile_fact / narrative_note / mood_log / commitment
      │     └── MEMORY.md + vector recall + MemoryGate
      │
      ├── ToolRegistry
      │     ├── builtin tools
      │     ├── MCP tools
      │     └── Skills
      │
      ├── ConfirmGate
      ├── ReminderRuntime
      └── ProactiveEngine
            │
            ▼
     LLMRouter + SQLite
```

项目结构：

```text
agent/
├── channels/        # Telegram 等入口
├── console/         # FastAPI 本地控制台
├── eval/            # 记忆质量评测
├── gateway/         # LLM / embedding 网关
├── memory/          # 记忆存储、抽取、召回、巩固、journal
├── orchestration/   # 单轮对话编排
├── proactive/       # 主动心跳、候选筛选、频率策略
├── reminders/       # 精确提醒运行时
├── tools/           # builtin / MCP / Skill 工具系统
├── trust/           # 确认门
├── cli.py
└── main.py
```

## 测试

```bash
python -m pytest
```

当前有 `126` 个确定性测试，覆盖：

- 时间归一化与跨天锚点
- 记忆路由、合并、遗忘、门控
- 主动承诺生命周期
- 精确提醒创建、恢复、补偿
- 工具循环、懒加载、危险工具确认
- 评分策略与每日记忆投影
LLM 质量评测单独运行，避免把非确定性模型输出放进 CI：

```bash
python -m agent.eval.memory_eval
```

## 隐私边界

这是本地优先项目：运行时数据默认放在 `data/`，并被 `.gitignore` 排除。

不要提交：

- `.env`
- `config.toml`
- `data/agent.db`
- `data/MEMORY.md`
- `data/agent_db.md`
- `data/journal/*`
- `data/mcp.json`

这些文件可能包含聊天内容、个人画像、情绪、提醒、API key 或 MCP token。控制台默认只适合本机或 SSH 隧道访问，不建议直接暴露到公网。

## 配置文件

- `.env.example`：Docker / Telegram / API key 示例。
- `config.example.toml`：本地 CLI 示例配置。
- `config.docker.toml`：Docker 默认配置，敏感信息从环境变量读取。
- `SOUL.md`：默认人格文件，可以改成自己的智能体人格。
- `data/mcp.json`：运行时 MCP 配置，可能包含 token，不要提交。
