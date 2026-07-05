# costrict-agent 融入 chat-rag 的最小改动方案

## 结论

最小改动方案是：

```text
costrict-agent 增加一个很薄的 HTTP serve 层；
chat-rag 不改核心代码，只通过现有 GenericTool 配置调用 costrict-agent。
```

目标链路：

```text
用户请求
-> chat-rag
-> 模型输出 costrict_agent XML 工具调用
-> chat-rag GenericToolExecutor 通过 HTTP 调 costrict-agent
-> costrict-agent 执行本地 Agent Loop
-> costrict-agent 返回结果
-> chat-rag 把工具结果回填模型
-> 模型生成最终回答
```

这个方案避免把本地文件读写、bash、workspace 权限、session 等复杂逻辑塞进 `chat-rag`，同时复用 `chat-rag` 已有 generic tool 调用机制。

## 为什么这是最小改动

`chat-rag` 已经具备：

- 工具配置结构：`ToolConfig` / `GenericToolConfig`
- XML 工具调用检测：`DetectTools`
- XML 参数解析：`ExtractParametersWithContext`
- 参数校验：`ValidateParameters`
- HTTP 工具客户端：`GenericToolClient`
- ready check：`CheckToolReady`
- 工具描述注入 prompt：`XmlToolAdapter`
- 工具结果回填模型：`handleToolExecution`
- Nacos `tools_prompt` 热更新：`nacos_config_manager.go`

因此，`chat-rag` 侧不需要新建 Agent Runtime，也不需要理解 `read_file`、`grep`、`edit_file`、`bash` 等细粒度本地工具。

`costrict-agent` 已经具备：

- CLI Agent Loop
- 本地工具注册表
- 文件读写、搜索、bash、todo 工具
- workspace 路径限制
- `--allow-write` / `--allow-bash`
- session 落盘
- 工具 schema
- 工具错误恢复 hint

因此，`costrict-agent` 侧只需要把已有 CLI 能力包一层 HTTP API。

## 改动范围

### 必改

```text
middles/costrict-agent/
```

新增 HTTP 服务层。

```text
deploy/compose/chat-rag/chat-api.yaml
```

或 Nacos 的 `tools_prompt` 配置中新增 `costrict_agent` generic tool。

### 视验证方式决定是否改

```text
deploy/compose/docker-compose.yml.tpl
```

如果要通过 Compose 做 `chat-rag -> costrict-agent` 端到端验证，需要增加 `costrict-agent` 常驻服务。

### 暂不改

```text
services/chat-rag/internal/logic/chat.go
services/chat-rag/internal/functions/tool_executor.go
services/chat-rag/internal/client/tool_client_factory.go
services/chat-rag/internal/promptflow/processor/xml_tool_adapter.go
```

最小方案应尽量不改 `chat-rag` 核心代码。

## 现有 chat-rag 调用机制

关键链路：

```text
XmlToolAdapter
-> 把 GenericTools 工具描述插入 prompt
-> LLM 输出 XML 工具标签
-> GenericToolExecutor.DetectTools 检测工具名
-> GenericParameterParser 解析 XML 参数
-> getGenericParameters 合并上下文
-> GenericToolClient HTTP 调用 endpoints.search
-> 工具结果返回 chat-rag
-> chat-rag 把结果回填 LLM
```

关键文件：

```text
services/chat-rag/internal/config/config.go
services/chat-rag/internal/functions/tool_executor.go
services/chat-rag/internal/client/tool_client_factory.go
services/chat-rag/internal/client/http_client.go
services/chat-rag/internal/promptflow/processor/xml_tool_adapter.go
services/chat-rag/internal/bootstrap/service_context.go
services/chat-rag/internal/bootstrap/nacos_config_manager.go
```

`chat-rag` 会自动合并这些上下文参数：

```text
clientId
codebasePath
clientVersion
authorization
```

因此 `costrict-agent` 的 HTTP API 必须能接收这些字段。

## 新增 costrict-agent serve

### CLI

新增命令：

```bash
costrict-agent serve --listen :8080 --workspace /workspace
```

建议支持：

```bash
costrict-agent serve \
  --listen :8080 \
  --workspace /workspace \
  --allow-write=false \
  --allow-bash=false
```

服务启动权限是上限，单次请求权限不能超过服务启动权限。

实际权限：

```text
effectiveAllowWrite = serverAllowWrite && requestAllowWrite
effectiveAllowBash  = serverAllowBash  && requestAllowBash
```

### API

MVP 只需要两个 endpoint。

```text
GET /healthz
```

用途：

- 给 `chat-rag` ready check 使用。
- 给 Compose healthcheck 使用。

响应：

```json
{
  "ok": true
}
```

```text
POST /v1/runs
```

用途：

- 给 `chat-rag` GenericToolClient 调用。
- 执行一次本地 Agent Loop。

请求体：

```json
{
  "clientId": "local-dev",
  "codebasePath": ".",
  "clientVersion": "dev",
  "task": "分析 services/chat-rag 的工具调用流程",
  "allowWrite": false,
  "allowBash": false,
  "mock": false,
  "maxSteps": 12
}
```

字段说明：

| 字段 | 来源 | 是否必需 | 说明 |
|---|---|---:|---|
| `task` | LLM | 是 | 传给本地 Agent Loop 的任务 |
| `clientId` | chat-rag context | 否 | 审计字段，MVP 不做鉴权 |
| `codebasePath` | chat-rag context | 否 | 项目路径，MVP 只允许 workspace 内相对路径 |
| `clientVersion` | chat-rag context/header | 否 | 审计字段 |
| `allowWrite` | tool config/manual | 否 | 单次请求是否请求写权限 |
| `allowBash` | tool config/manual | 否 | 单次请求是否请求 bash 权限 |
| `mock` | 测试 | 否 | 本地验证时使用 mock model |
| `maxSteps` | manual | 否 | Agent Loop 最大步数 |

成功响应：

```json
{
  "ok": true,
  "content": "分析结果或最终回答",
  "session": ".costrict/sessions/20260705T000000Z.json"
}
```

失败响应：

```json
{
  "ok": false,
  "error": "错误原因",
  "hint": "恢复建议"
}
```

## chat-rag 工具配置

新增 generic tool：

```yaml
GenericTools:
  - name: costrict_agent
    description: "Use CoStrict local agent to inspect a mounted code workspace and produce code-task results."
    capability: "Can inspect repository files, search code, optionally edit files or run commands when explicitly allowed, and return a session trace."
    endpoints:
      search: "http://costrict-agent:8080/v1/runs"
      ready: "http://costrict-agent:8080/healthz"
    method: "POST"
    parameters:
      - name: task
        type: string
        description: "The coding task for the local agent."
        required: true
        source: llm
      - name: allowWrite
        type: boolean
        description: "Whether this run may write files. Defaults to false."
        required: false
        default: false
        source: manual
      - name: allowBash
        type: boolean
        description: "Whether this run may execute bash commands. Defaults to false."
        required: false
        default: false
        source: manual
      - name: maxSteps
        type: integer
        description: "Maximum number of agent loop tool steps."
        required: false
        default: 12
        source: manual
    rule: "Use this tool for repository-level code investigation or patch-oriented tasks. Prefer read-only investigation unless the user explicitly allows file writes or command execution."
```

注意：

- 发布包 v0.0.7 中 `chat-rag` 配置大量迁移到 Nacos。
- 实际部署时可能需要更新 Nacos 的 `tools_prompt`，而不是只改本地 `chat-api.yaml`。
- 开发环境可先用本地 YAML 或测试配置验证。

## 细颗粒度实施步骤

### Step 1：抽取 run 复用点

目标：

让 CLI `run` 和 HTTP `/v1/runs` 复用同一段执行逻辑。

建议新增或调整：

```text
middles/costrict-agent/internal/agent
middles/costrict-agent/internal/cli
```

不要把 HTTP handler 直接复制 CLI `runCommand` 的全部逻辑。

验收：

- 现有 `costrict-agent run --mock ...` 行为不变。

### Step 2：新增 server package

新增：

```text
middles/costrict-agent/internal/server/server.go
```

职责：

- 注册 `/healthz`
- 注册 `/v1/runs`
- 解析 JSON 请求
- 调用 Agent Loop
- 返回 JSON 响应

验收：

```bash
go run . serve --listen :8080 --workspace .
curl http://127.0.0.1:8080/healthz
```

### Step 3：新增 serve CLI

修改：

```text
middles/costrict-agent/internal/cli/cli.go
```

新增：

```text
costrict-agent serve
```

参数：

```text
--listen
--workspace
--allow-write
--allow-bash
```

验收：

```bash
go run . serve --listen :8080 --workspace .
```

### Step 4：实现 /v1/runs mock 验证

先用 `mock: true` 验证，不依赖外部模型。

```bash
curl -X POST http://127.0.0.1:8080/v1/runs \
  -H 'Content-Type: application/json' \
  -d '{"task":"读取 go.mod 并告诉我模块名","mock":true}'
```

验收：

- HTTP 200。
- `ok=true`。
- 返回 `content`。
- 生成 session。

### Step 5：兼容 chat-rag 请求体

用接近 `GenericRequestBuilder` 的请求体测试：

```bash
curl -X POST http://127.0.0.1:8080/v1/runs \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer test' \
  -H 'Client-Version: dev' \
  -d '{
    "clientId": "local-dev",
    "codebasePath": ".",
    "clientVersion": "dev",
    "task": "读取 go.mod 并告诉我模块名",
    "allowWrite": false,
    "allowBash": false,
    "mock": true,
    "maxSteps": 12
  }'
```

验收：

- 不因多余上下文字段失败。
- 不把 Authorization 明文写入响应或 session。
- `codebasePath` 越界时拒绝。

### Step 6：增加 costrict_agent 工具配置

开发环境先加入 `chat-rag` 工具配置。

位置视环境而定：

```text
deploy/compose/chat-rag/chat-api.yaml
```

或 Nacos：

```text
tools_prompt
```

验收：

- `chat-rag` 启动时 `GenericToolExecutor` 能加载 `costrict_agent`。
- `CheckToolReady` 能访问 `/healthz`。
- prompt 中能注入 `costrict_agent` 描述。

### Step 7：端到端验证

目标 XML：

```xml
<costrict_agent>
  <task>读取 go.mod 并告诉我模块名</task>
</costrict_agent>
```

验收：

- `chat-rag` 检测到工具调用。
- `chat-rag` POST 到 `/v1/runs`。
- `costrict-agent` 执行任务。
- `chat-rag` 收到工具结果。
- 模型继续生成最终回答。

## Compose 仅作为运行承载

如果用 Compose 做端到端验证，需要增加常驻服务：

```yaml
  costrict-agent:
    build:
      context: ../../middles/costrict-agent
    image: zgsm/costrict-agent:dev
    command: ["serve", "--listen", ":8080", "--workspace", "/workspace"]
    environment:
      TZ: "Asia/Shanghai"
      COSTRICT_AGENT_BASE_URL: ${COSTRICT_AGENT_BASE_URL:-}
      COSTRICT_AGENT_API_KEY: ${COSTRICT_AGENT_API_KEY:-}
      COSTRICT_AGENT_MODEL: ${COSTRICT_AGENT_MODEL:-}
      DEEPSEEK_API_KEY: ${DEEPSEEK_API_KEY:-}
    volumes:
      - ${COSTRICT_AGENT_WORKSPACE:-../..}:/workspace
    networks:
      - shenma
```

注意：

- 这不是 CLI profile 容器方案。
- 这是给 `chat-rag` 调用的常驻 HTTP 服务。
- 是否使用 profile 取决于工具配置是否默认启用。

如果 `chat-rag` 默认配置了 `costrict_agent`，但 `costrict-agent` 服务没启动，ready check 会失败，工具可能不会进入可用状态。

## 安全边界

### workspace

MVP 中：

- `serve --workspace` 是服务可访问的根目录。
- 请求里的 `codebasePath` 只能是相对路径。
- 解析后必须仍在 workspace 内。
- 不允许请求传绝对路径动态切换到任意宿主目录。

### 写权限

默认：

```text
serverAllowWrite = false
requestAllowWrite = false
```

模型不能通过 XML 自行打开写权限。

### bash 权限

默认：

```text
serverAllowBash = false
requestAllowBash = false
```

模型不能通过 XML 自行打开 bash 权限。

### token

MVP 中：

- 可接收 Authorization。
- 不做完整鉴权。
- 不把 token 写入 session 或响应。
- 不通过 APISIX 对外暴露 `costrict-agent`。

## 不做事项

最小改动方案不做：

- 不把 `costrict-agent` 代码合入 `chat-rag`。
- 不让 `chat-rag` 直接管理 `read_file`、`grep`、`edit_file`、`bash`。
- 不重写 `chat-rag` 工具执行器。
- 不新增 APISIX 外部路由。
- 不新增任务队列。
- 不做多租户动态 workspace 挂载。
- 不做完整审计系统。
- 不接 Prometheus 指标。
- 不接 SWE-bench adapter。
- 不深度接入 `codebase-indexer`。

## 主要风险

### chat-rag 工具配置实际来源

源码环境可改 YAML。

发布包 v0.0.7 中，`chat-rag` 依赖 Nacos 配置中心，工具配置可能来自 `tools_prompt`。

实施前必须确认目标验证环境到底读哪个配置。

### GenericResponseHandler 返回字符串

`chat-rag` 当前 generic tool handler 最终给模型的是字符串。

因此 `/v1/runs` 第一版响应应保持模型可读，即使是 JSON，也要简洁清楚。

### costrict-agent 内部又调用模型

链路会变成：

```text
chat-rag 模型
-> costrict-agent 工具
-> costrict-agent 自己再调用模型
```

这会增加延迟和成本。

MVP 可以先用 `mock: true` 验证链路，再接真实模型。

后续可考虑让 `costrict-agent` 复用 `model-proxy` 或 `chat-rag` 的模型配置。

### 长任务超时

`chat-rag` generic client 当前 HTTP timeout 较短，`costrict-agent` 多轮 Agent Loop 可能超过默认 timeout。

如果端到端验证超时，可能需要最小改动之一：

- 提高 generic tool client timeout。
- 或让 `/v1/runs` 支持异步任务。

为了保持最小改动，MVP 先控制任务简单、步数较少。

## 验收标准

最小方案完成的判断标准：

```text
chat-rag 能通过 costrict_agent generic tool 调用 costrict-agent /v1/runs，
costrict-agent 能执行一次 mock 或真实 Agent Loop，
chat-rag 能拿到结果并继续生成最终回答。
```

最低验收命令/现象：

1. `costrict-agent serve` 可启动。
2. `/healthz` 返回 200。
3. `/v1/runs` 支持 mock run。
4. `chat-rag` 工具配置能加载 `costrict_agent`。
5. `costrict_agent` ready check 通过。
6. `chat-rag` 触发 XML 工具调用后，能收到 agent 结果。

## 后续演进

最小方案跑通后，再考虑：

- `costrict-agent` 复用 CoStrict `model-proxy`。
- 服务模式 session 查询 API。
- 异步任务和任务状态。
- 更强权限审批。
- APISIX 路由。
- Prometheus 指标。
- 与 `codebase-indexer` 的高级代码理解工具集成。
- SWE-bench adapter。
