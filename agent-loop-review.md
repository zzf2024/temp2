# CoStrict Agent Loop 评审报告

> 评审日期：2026-07-10
> 评审范围：`/workspace/costrict-for-vscode/`（基于 Roo Code/Cline fork，v2.8.15）
> 评审依据：`handoff.md` 交接文档 + agent loop 核心源码审查
> 参照对象：Claude Code、Codex CLI、Cursor Agent、Devin 等先进编程 agent

---

## 结论：agent loop 存在多处结构性不足

CoStrict 的 agent loop 在"能跑通"层面是完整的，但与先进编程 agent 相比，在**并行性、自主性、验证闭环、上下文管理、架构清晰度**上有明显差距。问题分两大类：handoff.md 已记录的运维问题（4 个），以及与先进 agent loop 相比的结构性差距（10 项）。

---

## 核心文件清单

| 文件 | 行数 | 职责 |
|------|------|------|
| `src/core/task/Task.ts` | 5656 | agent loop 主体，最核心 |
| `src/core/assistant-message/presentAssistantMessage.ts` | ~1080 | 消息呈现 + 工具执行调度 |
| `src/api/providers/costrict.ts` | ~1262 | CoStrict provider，API 请求 + 流式响应 |
| `src/core/context-management/index.ts` | 418 | 上下文管理（condense + 滑窗截断） |
| `src/core/condense/index.ts` | 703 | 对话摘要 |
| `src/core/auto-approval/AutoApprovalHandler.ts` | ~120 | 自动审批限制检查 |
| `src/core/task/SmartMistakeDetector.ts` | 350 | 智能错误检测 |
| `src/core/task/ModelFallbackManager.ts` | 371 | 模型故障切换（部分被注释禁用） |
| `src/core/tools/AttemptCompletionTool.ts` | ~180 | 任务完成工具 |
| `src/core/tools/NewTaskTool.ts` | ~160 | 子任务委派工具 |
| `src/core/prompts/sections/*.ts` | - | system prompt 各段生成 |

---

## 一、handoff.md 已记录的运维问题（4 个）

这些问题已被交接文档识别，属运维层面，确认属实：

### 问题 1：checkpoint 报错（非致命）

- **位置**：`src/core/checkpoints/index.ts:426` `updateCospecMetadata()`
- **现象**：
  ```
  [Task#updateCospecMetadataForCheckpoint] caught unexpected error, disabling checkpoints
  TypeError: Cannot read properties of undefined (reading 'text')
  ```
- **原因**：`task.clineMessages.filter(v => v.say === "checkpoint_saved")[0].text` — 没有 `checkpoint_saved` 消息时 `[0]` 为 undefined，访问 `.text` 崩溃
- **影响**：CLI 下 `enableCheckpoints: false`，但 `presentAssistantMessage` 仍调用 `updateCospecMetadata`，每次写文件都报错
- **修法**：加 `?.` 可选链 + 空值检查
- **难度**：极低（1 行）

### 问题 2：token 不自动刷新

- **位置**：`src/api/providers/costrict.ts:177`
- **现象**：`CostrictAuthService.getInstance()` 在 CLI 下未初始化（需要 `ClineProvider` 实例），会 throw，被 catch 后用构造时传入的旧 token
- **影响**：token 过期（约 1 小时）后请求 401，需重启脚本
- **修法方向**：CLI 模式下跳过 `CostrictAuthService`，直接用 `this.options.costrictAccessToken`
- **难度**：低

### 问题 3：OutputChannel 日志全部丢弃

- **位置**：`apps/cli/src/commands/cli/run.ts:111`
- **现象**：把 shim logger 设为空函数，插件运行时详细日志（API 请求头、工具执行细节、模式切换等）全部丢失
- **修法**：把 `setLogger` 改成写文件或写 stderr
- **难度**：低

### 问题 4：strict 等特色模式未验证

- CoStrict 的核心卖点是 `strict` 模式（标准化 AI 代码生成流程：需求→设计→任务→测试）
- CLI 下只测了 `ask` 和 `code` 模式
- `strict` 模式会触发 `fromWorkflow = true`，`prompt_mode` 从 `"vibe"` 变为模式名
- 需验证后端 chat-rag 是否正确处理 workflow 模式

---

## 二、与先进 agent loop 相比的结构性差距（10 项）

### 差距 1：伪并行工具执行（最关键）

**现状**：

- `buildNativeToolsArrayWithRestrictions` 向 API 传 `parallel_tool_calls: true`（`Task.ts:4425` 等），告诉模型可以并行调用多个工具
- 但执行层 `presentAssistantMessage` 用 `didAlreadyUseTool` 标志**强制串行**：
  - `presentAssistantMessage.ts:342` — 第一个工具执行后设 `didAlreadyUseTool = true`
  - `Task.ts:3485` — 检测到后中断流：`"Only one tool may be used at a time and should be placed at the end of the message."`
- 即模型返回了 3 个并行 tool_call，实际只执行第 1 个，其余被中断丢弃

**先进做法**：

- Claude Code、Codex **真正并行执行**无依赖的工具（如同时读多个文件、并行 grep），延迟可降低 50%+
- 对有依赖的工具（如先读后改）才串行

**问题加剧**：prompt 鼓励并行（`tool-use-guidelines.ts:6` "Multiple tools may be called in one message"），执行却惩罚并行，行为矛盾，会误导模型。

**改进方向**：
- 分析 tool_call 间的依赖关系（读操作可并行，写操作串行）
- 用 `Promise.all` 批量执行无依赖工具
- 移除或弱化 `didAlreadyUseTool` 的强制中断

**难度**：中

### 差距 2：缺少显式验证闭环（verification loop）

**现状**：

- `objective.ts` 和 `rules.ts` 只说 "Wait for user confirmation after each tool use to verify success"
- **没有**内置的 test/lint/build 验证步骤
- 模型改完代码后直接 `attempt_completion`，是否跑测试完全靠模型自觉
- `AttemptCompletionTool.ts:45` 只检查 `didToolFailInCurrentTurn`，不检查是否有验证步骤

**先进做法**：

- Codex 有明确的 "validate your work" 指令，修改后主动运行测试
- Claude Code 有测试运行闭环，修改代码后验证是否通过

**改进方向**：
- 在 system prompt 中加入验证指令（改完代码后运行相关 test/lint）
- `attempt_completion` 前检查是否执行过验证步骤
- 提供 `run_tests` 工具或在 `execute_command` 中引导验证

**难度**：中

### 差距 3：上下文管理较粗放

**现状**：

`context-management/index.ts` 的 `manageContext`：
- 只有"超阈值时 summarize + 滑窗截断"两种手段，**无主动 compaction**
- 滑窗截断固定丢 50%（`fracToRemove: 0.5`），无基于重要性的保留策略
- `summarizeConversation`（`condense/index.ts:256`）把 tool_use/tool_result 全转文本再摘要，**丢失结构化信息**，摘要质量受限
- subtask 完成后回传 parent 的只有一行 `completionResultSummary`（`AttemptCompletionTool.ts` 的 `delegateToParent`），**无结构化 context handoff**

**先进做法**：

- 先进 agent 在每 N 轮主动压缩，而非被动等爆
- 基于重要性保留（最近的 tool_result 完整保留，早期的摘要）
- subtask 回传关键文件列表、决策记录、失败教训

**改进方向**：
- 增加主动 compaction（每 N 轮检查并压缩）
- 改进截断策略，保留关键 tool_result 完整内容
- subtask 完成时回传结构化 context（修改的文件、遇到的问题、关键决策）

**难度**：中

### 差距 4：重试机制有栈溢出风险

**现状**：

`attemptApiRequest`（`Task.ts:4531`）失败后用**递归 generator** 重试：
```ts
yield* this.attemptApiRequest(retryAttempt + 1)  // Task.ts:4969, 4990, 5006
```

- 持续失败时调用栈无限增长
- `backoffAndAnnounce` 有限速（指数退避，上限 `MAX_EXPONENTIAL_BACKOFF_SECONDS = 600`），但**无硬性最大重试次数**
- `MAX_CONTEXT_WINDOW_RETRIES = 3` 只管 context window 错误，不管其他错误

**先进做法**：

- 迭代 + 有界重试 + circuit breaker
- 区分可重试错误（429、5xx）和不可重试错误（401、400），后者直接停止

**改进方向**：
- 将递归 generator 改为迭代循环
- 加全局最大重试次数（如 10 次）
- 对 401/400 等不可重试错误直接 `shouldStop = true`（部分已实现，见 `convertErrorMessage` 的 `pauseHandler`）

**难度**：中

### 差距 5：终止条件依赖模型自觉

**现状**：

- `initiateTaskLoop`（`Task.ts:2781`）靠模型调用 `attempt_completion` 结束
- 模型不调工具也不 `attempt_completion` 时，注入 `noToolsUsed()` 警告再循环
- **无硬性 turn 上限**：`MAX_REQUESTS_PER_TASK` 只在注释里提到（`Task.ts:2801`），实际靠 `AutoApprovalHandler` 的 `allowedMaxRequests`，默认 `Infinity`（`AutoApprovalHandler.ts:51`）
- `noToolsUsed()` 的提示词（`responses.ts:42`）是生硬的 `[ERROR] You did not use a tool!`，引导性差

**先进做法**：

- 硬性 turn 上限 + 超限后优雅降级（总结当前进度，交还用户）
- 建设性的继续引导（而非纯错误提示）

**改进方向**：
- 设默认 `allowedMaxRequests` 为合理值（如 50-100）
- `noToolsUsed()` 提示词改为建设性引导
- 超限后总结进度并 `attempt_completion`

**难度**：中

### 差距 6：单体 Task.ts（5656 行）职责过重

**现状**：

一个文件包含：主循环、流式解析、工具调度、重试、backoff、上下文管理、checkpoint、subtask、错误转换、消息保存、token 统计……

**影响**：

- 测试困难（`Task.ts` 的 `__tests__` 很难覆盖所有路径）
- 修改一处易引发连锁问题
- handoff.md 提到的多个 bug 都源于这个文件的耦合

**先进做法**：

- 拆分为独立模块：`AgentLoop` / `ToolExecutor` / `ContextManager` / `RetryPolicy` / `StreamingHandler`
- 各模块可独立测试

**改进方向**：
- 将流式解析逻辑抽到 `StreamingHandler`
- 将工具执行调度抽到 `ToolExecutor`
- 将重试/backoff 抽到 `RetryPolicy`
- 将上下文管理抽到 `ContextManager`

**难度**：高

### 差距 7：架构债——VSCode 扩展包袱

**现状**：

整个 loop 深度耦合 VSCode provider 模型：
- 每轮调 `providerRef.deref()?.getState()` 多次（`Task.ts` 里出现 30+ 次）
- 工具执行插入了 `askApproval` 人工确认环节（`presentAssistantMessage.ts` 每个 tool 都过 `askApproval`）
- CLI 模式下这些 provider/ask 逻辑全是 mock，增加复杂度但无实际价值

**先进做法**：

- 先进 CLI agent（Codex CLI）从底层就是 CLI-first，无 UI 耦合
- 核心逻辑与 UI 层解耦

**改进方向**：
- 抽象出 `TaskContext` 接口，屏蔽 VSCode provider 细节
- CLI 和 VSCode 各自实现该接口
- 工具审批改为可配置策略（CLI 下自动通过或配置化）

**难度**：极高

### 差距 8：codebase_search 被注释禁用

**现状**：

- `presentAssistantMessage.ts:808-810` 和 `native-tools/index.ts:88` 都注释掉了 `codebase_search`
- 但 `CodebaseSearchTool.ts` 和整个 `services/code-index/`（embedder、vector store、cache manager）代码都在
- 模型无法做语义代码搜索，只能 `search_files`（正则）或 `list_files`

**影响**：

- 大型 codebase 中检索效率远低于有语义索引的 agent（如 Cursor）

**改进方向**：
- 调查禁用原因（可能是 CLI 下 code-index 依赖未初始化）
- 修复后取消注释并启用

**难度**：中

### 差距 9：每轮重建 environment_details 开销大

**现状**：

`getEnvironmentDetails`（325 行）每轮都收集：可见编辑器、tab、终端状态、git status、文件树。
- `shouldAttachFileTree` 有间隔控制，但基础信息每轮都跑

**先进做法**：

- 增量更新和缓存
- 只在发生变化时更新

**改进方向**：
- 缓存 environment_details，基于文件变更事件增量更新
- 减少 `providerRef.deref()?.getState()` 调用次数（每轮缓存一次）

**难度**：中

### 差距 10：防御机制部分失效

**现状**：

- `SmartMistakeDetector` 和 `ModelFallbackManager` 代码存在
- 但 `attemptApiRequest` 里 fallback 的关键调用**被注释掉**（`Task.ts:4952-4959`）：
  ```ts
  // if (this.apiConfiguration.apiProvider === "costrict") {
  //   const switched = this.modelFallbackManager?.recordFailure(error, "server_error")
  ```
- `backoffAndAnnounce` 里的 `say("auto_switch_model", statusMsg)` 也被注释（`Task.ts:5084-5086`）
- 即 fallback 机制**实际未启用**

**改进方向**：
- 调查注释原因，修复后启用
- 确认 fallback 切换时通知用户

**难度**：低（取消注释 + 验证）

---

## 三、改进优先级建议

| 优先级 | 改进项 | 影响 | 难度 | 对应问题 |
|--------|--------|------|------|----------|
| P0 | 修 checkpoint 空值 | 消除噪音，1 行改动 | 极低 | 运维问题 1 |
| P0 | 修 token 刷新 | 1 小时后不崩 | 低 | 运维问题 2 |
| P0 | 恢复日志 | 可调试 | 低 | 运维问题 3 |
| P1 | 真正并行执行无依赖工具 | 延迟降 50%+ | 中 | 差距 1 |
| P1 | 加硬性 max turns + 有界重试 | 防无限循环/栈溢出 | 中 | 差距 4、5 |
| P1 | 加验证闭环（改后跑 test/lint） | 提升正确率 | 中 | 差距 2 |
| P2 | 启用 ModelFallbackManager | 容错 | 低 | 差距 10 |
| P2 | 主动上下文 compaction | 长任务不爆 | 中 | 差距 3 |
| P2 | 拆分 Task.ts 为独立模块 | 可维护性 | 高 | 差距 6 |
| P3 | 启用 codebase_search | 大库检索 | 中 | 差距 8 |
| P3 | subtask 结构化 context handoff | 多任务协作 | 高 | 差距 3 |
| P3 | 去 VSCode provider 耦合（CLI-first 重构） | 架构清晰 | 极高 | 差距 7 |
| P3 | environment_details 增量更新 | 性能 | 中 | 差距 9 |

---

## 四、核心差距总结

最关键的三点决定了与先进 agent 在效率和可靠性上的核心差距：

1. **伪并行工具执行**（差距 1）— prompt 鼓励并行但执行强制串行，既慢又矛盾
2. **缺验证闭环**（差距 2）— 改完代码不验证，正确率无保障
3. **单体 Task.ts**（差距 6）— 5656 行耦合一切，难以测试和维护

handoff.md 已识别的 4 个运维问题（P0）应先修，作为后续改进的基础。
