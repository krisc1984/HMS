# 通用 Agent 接入 HMS 记忆的契约

> 面向所有"通用编码/对话 agent"（opencode、claude-code、codex、cursor、cline、continue、windsurf、gemini-cli 等）的**唯一**接入说明。
> HMS 通过 **MCP** 暴露记忆能力；agent 侧只需注册一个 MCP server，再加三段生命周期 hook 即完成"无缝挂载 HMS 记忆"。

---

## 1. TL;DR

- **数据面/工具面 = MCP**。工具的权威定义在 `core/dataplane/hms_api/mcp_tools.py`（`_ALL_TOOLS`），HTTP 传输由 `core/dataplane/hms_api/api/mcp.py` 提供，stdio 传输由 `core/dataplane/hms_api/mcp_local.py` 提供。
- **控制面 = agent 自己的 hooks**（session 开始 / 结束 / 压缩），负责在合适时机调用 MCP 工具 `recall` 与 `retain`。
- **认证 = 单一 Bearer token**（`HMS_API_TOKEN` / `HMS_API_TENANT_API_KEY`）。
- **多项目/多会话隔离 = `bank_id` 动态命名**（`agent::project[::session[::user]]`，分隔符 `::`）。

新接一个通用 agent 的**最小步骤**只有三步：

1. 在 agent 的配置文件里注册一个 MCP server（HTTP 或 stdio 二选一）。
2. 在 agent 的 hooks 中，会话开始时调用 `recall`，会话结束/空闲时调用 `retain`。
3. 用 `Authorization: Bearer <token>` 与 `X-Bank-Id: <bank>` 把请求打到 HMS。

现有 `interface/adapters/opencode` 与 `interface/adapters/claude-code` 就是这套契约的两个参考实现，可直接抄用。

---

## 2. 接入形态选择

| 形态 | 适用场景 | 特点 |
|---|---|---|
| **Streamable HTTP MCP** | 远程/团队/多机共享；生产部署 | 单端点、可跨机；需要 HTTP 可达；使用 Bearer 鉴权 |
| **stdio MCP** | 单机本地/离线；开发调试；agent 与 HMS 同机 | 直接由 agent 拉起子进程；无网络依赖；免鉴权（本地信任） |

**决策树**：

- 团队多人共用同一 HMS 实例 → **HTTP**
- 单机开发、demo、离线环境 → **stdio**
- 不确定 → 先用 HTTP，本地跑 HMS 也走 HTTP（`http://localhost:18080/mcp`）

---

## 3. Endpoint / 传输

### 3.1 HTTP MCP

- **挂载路径**：`/mcp`（来自 `core/dataplane/hms_api/main.py` 的 `mcp_mount_path="/mcp"`）
- **完整 URL 示例**：
  - 本地：`http://localhost:18080/mcp`
  - 生产（走 gateway/nginx）：`https://{PUBLIC_BASE_URL}/mcp`
- **协议**：Streamable HTTP（FastMCP HTTP transport）

### 3.2 stdio MCP

- **推荐启动方式**：
  - `uvx hms-mcp`（若已发布 wheel）
  - 或 `python -m hms_api.mcp_local`（源码模式，需在 HMS repo 中 `uv sync`）
- 参考 `core/dataplane/hms_api/mcp_local.py`。

---

## 4. 认证

### 4.1 环境变量契约

| 变量名 | 位置 | 用途 |
|---|---|---|
| `HMS_API_TENANT_API_KEY` | HMS server 侧 | 由 `ApiKeyTenantExtension` 校验的合法 key（默认见 `.env.example`） |
| `HMS_API_TOKEN` / `HMS_MEMORY_API_KEY` | Client / agent 侧 | 发到 HMS 的 Bearer token（值应等于服务端某个合法 key） |
| `HMS_API_MCP_AUTH_TOKEN` | 两侧 | **可选**的旁路 token；若设置，MCP 传输层会先用它做短路鉴权，命中则跳过 tenant 校验（老适配包在用，新接入不推荐） |

### 4.2 HTTP 请求头

```
Authorization: Bearer <HMS_API_TOKEN>
X-Bank-Id: <bank_id>          # 可选；也可通过工具入参 bank_id 传
Content-Type: application/json
```

服务端解析逻辑见 `core/dataplane/hms_api/api/mcp.py` 中的 `_get_header("Authorization")` 与 tenant extension；同时支持 `Bearer <token>` 与裸 token 两种写法。

### 4.3 stdio 场景

stdio 模式默认信任本地进程，通常不需要 Bearer。若要求鉴权，通过环境变量 `HMS_API_TOKEN` 传入子进程即可，MCP local 会读取。

---

## 5. Bank 作用域约定

一个 `bank_id` 对应一个记忆命名空间。**同一个 agent 的不同项目/不同用户，一定要用不同的 `bank_id`，否则记忆会互相污染**。

### 5.1 静态 bank_id

由 agent 配置固定：`HMS_BANK_ID=my-project`。适合单人、单项目场景。

### 5.2 动态 bank_id（推荐）

以 `::` 为分隔符拼接多维度字段。参考 `interface/adapters/opencode/src/bank.ts` 与 `interface/adapters/claude-code`。

**合法字段**：

| 字段 | 说明 |
|---|---|
| `agent` | agent 名称（`opencode` / `claude-code` / …） |
| `project` | 工作目录 basename |
| `gitProject` | 若在 git 仓库内，取主 worktree 根目录 basename（保证所有 linked worktree 共享同一 bank） |
| `session` | agent 侧 session id |
| `channel` | 环境变量 `HMS_CHANNEL_ID` |
| `user` | 环境变量 `HMS_USER_ID` |

**推荐粒度**：

- 单人本地：`agent::project`（如 `opencode::HMS`）
- 团队协作：`agent::gitProject::user`
- 多会话相互隔离：`agent::project::session`

**前缀**：`HMS_BANK_PREFIX` 可给所有 bank 加统一前缀（如租户名）。

### 5.3 Gateway 命名空间

若 HMS 通过 vendor gateway 暴露（`HMS_GATEWAY_SCOPE_BANK_IDS=true`），gateway 会透明地把外部传入的 `bank_id` 用调用方 API key 打上前缀，避免不同 vendor 之间的 bank 名撞车。这一层对 agent 侧完全透明，不需要感知。

---

## 6. 工具面（权威列表）

**取真列表以 MCP `list_tools` 为准**；下面仅作为语义速查，schema 由 agent 从 MCP 拉。

### 6.1 核心记忆

| 工具 | 用途 | 关键入参 |
|---|---|---|
| `retain` | 异步入库一段对话/事实。返回 `operation_id`。 | `messages`（`[{role, content}]`），`bank_id`，`tags`，`metadata` |
| `sync_retain` | 同步入库，阻塞至完成。 | 同上 |
| `recall` | 向量+关键词检索。返回若干 memory hits。 | `query`，`bank_id`，`top_k`，`budget`，`types` |
| `reflect` | 基于 recall 结果生成综合回答。 | `query`，`bank_id` |

### 6.2 Bank 管理

`list_banks` / `create_bank` / `get_bank` / `get_bank_stats` / `update_bank` / `delete_bank` / `clear_memories`

### 6.3 心智模型

`list_mental_models` / `get_mental_model` / `create_mental_model` / `update_mental_model` / `refresh_mental_model` / `delete_mental_model`

### 6.4 指令 / 记忆 / 文档 / 操作 / 标签

- 指令：`list_directives` / `create_directive` / `delete_directive`
- 记忆条目：`list_memories` / `get_memory`
- 文档：`list_documents` / `get_document` / `delete_document`
- 异步操作：`list_operations` / `get_operation` / `cancel_operation`
- 标签：`list_tags`

**Single-bank 模式**下，`*_bank` 与 `clear_memories` 会被隐藏；工具入参也不再需要 `bank_id`。由 server 侧的 `MCPToolsConfig.bank_id_resolver` 控制。

---

## 7. 推荐的 hook 时机（控制面契约）

以下时机名沿用现有两个参考适配包的语义。任意通用 agent 只要能对上其中一档即可接入。

| 时机 | 建议动作 | 参考 |
|---|---|---|
| **session_start / SessionStart** | 以首条 user message（或项目摘要）为 query，调 `recall(top_k=?, budget=mid)`；把 hits 拼成 system message 注入上下文 | claude-code `hooks/session_start.py`；opencode `session.created` |
| **assistant_message_end / session.idle / Stop** | 累计最近 N 轮（默认 10 轮，overlap 2 轮）后异步 `retain`；不要每轮都写 | claude-code `hooks/stop.py`；opencode `session.idle` |
| **context_compacting / experimental.session.compacting** | 把最近一次 recall 结果标记为"不可压缩"，压缩后重新注入 system prompt | opencode `experimental.session.compacting` |
| **session_end / SessionEnd** | 最后一次 flush `retain`，用 `get_operation` 轮询直到完成，避免会话结束后丢消息 | claude-code `hooks/session_end.py` |

**默认 knob（可覆盖）**：

- `retainEveryNTurns=10`，`retainOverlapTurns=2`
- `recallMaxTokens=1024`，`recallBudget=mid`，`recallContextTurns=1`
- `recallRoles=["user","assistant"]`，`retainRoles=["user","assistant"]`
- 参考 `interface/adapters/claude-code/settings.json`

---

## 8. 各家 agent 配置片段

### 8.1 opencode

已有官方适配包 `@hms-memory/opencode-hms`（`interface/adapters/opencode`），可直接 `bun add` 或 npm 安装。若手动接入，`opencode.jsonc`：

```jsonc
{
  "mcpServers": {
    "hms": {
      "type": "http",
      "url": "http://localhost:18080/mcp",
      "headers": {
        "Authorization": "Bearer ${HMS_API_TOKEN}",
        "X-Bank-Id": "opencode::HMS"
      }
    }
  }
}
```

hooks 由插件自动挂载。若不使用官方插件，可自行订阅 `session.created` / `session.idle` / `experimental.session.compacting` 走 §7 逻辑。

### 8.2 claude-code

已有官方适配包（`interface/adapters/claude-code`，通过 `.claude-plugin/marketplace.json` 分发的 `hms-memory` 插件）。若手动接入，`~/.claude/settings.json`：

```json
{
  "mcpServers": {
    "hms": {
      "type": "http",
      "url": "http://localhost:18080/mcp",
      "headers": {
        "Authorization": "Bearer ${HMS_API_TOKEN}",
        "X-Bank-Id": "claude-code::HMS"
      }
    }
  },
  "hooks": {
    "SessionStart":     [{ "command": "python -m hms_client.hooks.session_start" }],
    "UserPromptSubmit": [{ "command": "python -m hms_client.hooks.prompt_submit" }],
    "Stop":             [{ "command": "python -m hms_client.hooks.stop" }],
    "SessionEnd":       [{ "command": "python -m hms_client.hooks.session_end" }]
  }
}
```

stdio 版：

```json
{
  "mcpServers": {
    "hms": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "hms_api.mcp_local"],
      "env": { "HMS_MCP_BANK_ID": "claude-code::HMS" }
    }
  }
}
```

### 8.3 codex（OpenAI）

`~/.codex/config.toml`：

```toml
[mcp_servers.hms]
type    = "http"
url     = "http://localhost:18080/mcp"
headers = { Authorization = "Bearer ${HMS_API_TOKEN}", X-Bank-Id = "codex::HMS" }
```

参考 `interface/adapters/codex`。

### 8.4 cursor

`.cursor/mcp.json`（项目级）或 `~/.cursor/mcp.json`（全局）：

```json
{
  "mcpServers": {
    "hms": {
      "url": "http://localhost:18080/mcp",
      "headers": {
        "Authorization": "Bearer ${HMS_API_TOKEN}",
        "X-Bank-Id": "cursor::HMS"
      }
    }
  }
}
```

Cursor 目前无稳定的 session hook，只能靠工具面手工触发，或在 rules 里写"每轮结束后请调用 `retain`"作为兜底。

### 8.5 cline / continue.dev / windsurf / gemini-cli

三家均支持 MCP 且配置格式基本一致（`mcpServers` 键 + `url` + `headers` 或 `command`/`args`）。照抄 §8.4 的写法即可，`bank_id` 换成各自的 `agent` 段名。gemini-cli 请以 `~/.gemini/settings.json` 为准。

### 8.6 无 MCP 支持的 agent

对于目前不支持 MCP 的 agent（如某些老式框架），退化路径是直接调 REST：

- Retain: `POST /v1/default/banks/{bank_id}/memories`
- Recall: `POST /v1/default/banks/{bank_id}/recall`

契约见 `core/dataplane/hms_api/api/http.py`。此路径**不推荐**用于新接入，除非该 agent 明确不打算支持 MCP。

---

## 9. 异步操作与流控

- `retain` 默认异步，返回 `{ "operation_id": "op_..." }`。
- 客户端可用 `get_operation(op_id)` 轮询状态；`list_operations` 查看当前所有操作；`cancel_operation` 中止。
- 会话结束时建议对最后一次 `retain` 做**同步等待**（`sync_retain` 或轮询直到 `status in {succeeded, failed}`）以避免丢数据。
- 推荐超时：`retain` 60s、`recall` 10s、`reflect` 30s。失败按指数退避重试 2 次；仍失败降级为记录一条 tag 为 `hms.retain.failed` 的本地日志，避免阻塞主流程。

---

## 10. 观测与调试

| 手段 | 用途 |
|---|---|
| `GET /health` | 服务存活检查 |
| `GET /version` | HMS 版本号 |
| `GET /metrics` | Prometheus 指标（可开可关，见 `HMS_API_METRICS_ENABLED`） |
| MCP `list_tools` | 自检：确认 agent 侧成功连上并能看到全部工具 |
| MCP `list_banks` | 自检：确认 bank scope 正确 |
| `list_operations` | 追查最近的 retain/recall 执行状态 |

**常见错误码**：

| HTTP / MCP | 现象 | 排查 |
|---|---|---|
| 401 | Authorization header required / invalid | `HMS_API_TOKEN` 与服务端 `HMS_API_TENANT_API_KEY` 不一致；或 header 拼错 |
| 404 bank not found | 未创建对应 bank | 先调 `create_bank`，或使用会自动创建的适配包 |
| 5xx on retain | embedding daemon 未起 | 见 `core/daemon/README.md`，本地一般由 adapter 自启；docker 部署由 compose 保证 |
| MCP 连不上 | 端口/mount path 不对 | 确认 `mcp_enabled=true`、访问路径是 `/mcp` 而非 `/v1/.../mcp` |

---

## 11. 兼容性矩阵

| Agent | HTTP MCP | stdio MCP | Session Hooks | 现有适配包 | 推荐形态 |
|---|---|---|---|---|---|
| opencode | ✅ | ✅ | ✅ | `interface/adapters/opencode` | HTTP + 官方插件 |
| claude-code | ✅ | ✅ | ✅ | `interface/adapters/claude-code` | HTTP + 官方插件 |
| codex | ✅ | ✅ | 部分 | `interface/adapters/codex` | HTTP |
| cursor | ✅ | ✅ | ❌ | — | HTTP，工具触发兜底 |
| cline | ✅ | ✅ | ❌ | — | HTTP |
| continue.dev | ✅ | ✅ | 部分 | — | HTTP |
| windsurf | ✅ | ✅ | ❌ | — | HTTP |
| gemini-cli | ✅ | ✅ | ❌ | — | HTTP |
| ag2 / autogen / crewai / langgraph / llamaindex / smolagents / strands / pydantic-ai / agno / agentcore / ai-sdk / pipecat / dify / n8n / litellm / openai-agents | — | — | — | 各自 `interface/adapters/*` | 走各自框架的 tool 抽象包装 MCP，或直连 REST |

> **说明**：无 hooks 支持的 agent 请用 §8.4 中的"工具触发兜底"方案，即依赖模型主动调用 `retain` / `recall`，效果不如带 hooks 的自动化，但仍可用。

---

## 12. 相关文件索引

**Server 端（工具/传输的权威源）**：

- `core/dataplane/hms_api/mcp_tools.py`：所有 MCP 工具的实现与 `_ALL_TOOLS` 列表
- `core/dataplane/hms_api/api/mcp.py`：HTTP MCP 传输、鉴权、bank/tenant context
- `core/dataplane/hms_api/mcp_local.py`：stdio MCP 入口
- `core/dataplane/hms_api/api/http.py`：REST 路由（fallback）
- `core/dataplane/hms_api/main.py`：`mcp_mount_path="/mcp"` 与启动装配

**参考适配包（照抄用）**：

- `interface/adapters/opencode/`：TS 插件，含 hooks / bank 派生 / mission 管理
- `interface/adapters/claude-code/`：hooks + stdio MCP + `.claude-plugin` marketplace 分发
- `interface/adapters/codex/`：Codex 配置示例

**契约参考**：

- `.env.example`：完整环境变量清单
- `vendor_sdk/DESIGN.md`、`vendor_sdk/GATEWAY.md`：对外 vendor gateway（与本 doc 不重叠，用于第三方 vendor 直连）

---

## 13. 变更规则

- **本文档是 HMS 对通用 agent 生态的公共契约**。修改 `bank_id` 分隔符、`Authorization` 头名、MCP mount path、工具语义等**破坏性**变更前，先更新本文件并列入 release note。
- 新增工具时无需改本 doc（agent 走 `list_tools` 自发现），但**移除或重命名**工具要在此处标注 deprecated 至少一个 release。
- 各家 adapter 的 `settings.json` / `opencode.jsonc` 默认值调整不必登记到本 doc；仅当新 agent 首次接入时补一节到 §8。
