# Phone Agent 性能测试报告

> 测试日期: 2026-06-19
> 硬件平台: NVIDIA A10 (24GB, SM86, HBM 600GB/s)
> 模型: AutoGLM-Phone-9B (zai-org/AutoGLM-Phone-9B)
> 推理引擎: SGLang v0.5.x + CUDA Graphs
> 测试工具: `scripts/bench_resolution.py`, 自定义 benchmark 脚本

---

## 一、推理性能

### 1.1 基础指标

| 场景 | TTFT (ms) | TPS (tok/s) | 输出 Token 数 | 总耗时 (s) |
|------|-----------|-------------|-------------|-----------|
| **纯文本 (短)** | 74 | 28.1 | 45 | 1.7 |
| **纯文本 (长)** | 74 | 29.3 | 199 | 6.8 |
| **多模态 (720p)** | 106~177 | 28.4 | 148 | 5.3 |
| **多模态 (冷启动)** | ~450 | — | — | — |

> 注: TTFT (Time To First Token) 为首 token 延迟；TPS (Tokens Per Second) 为生成阶段吞吐。
> 冷启动值包含了首次运行时 CUDA kernel 预热开销，实际使用中以 warm 值为准。

### 1.2 分辨率 vs 性能

| 分辨率 | TTFT (ms) | TPS (tok/s) |
|--------|-----------|-------------|
| 240 × 540 | 81 | 28.0 |
| 360 × 800 | 83 | 29.1 |
| 480 × 1067 | 90 | 28.1 |
| 720 × 1600 | 106 | 27.6 |
| 1080 × 2400 (原始) | — | — |

> 1080×2400 原始分辨率触发 GPU OOM（24GB 显存不足，vision encoder 峰值超过当前可用 2GB）。

### 1.3 CUDA Graphs 加速效果

| 指标 | 关闭 CUDA Graph | 开启 CUDA Graph | 提升 |
|------|---------------|---------------|------|
| 文本 TTFT | ~80ms | **74ms** | **−7.5%** |
| 文本 TPS | ~27.1 | **28.1** | **+3.7%** |
| 多模态 TTFT | ~230ms | **106ms** | **−53.9%** |
| 多模态 TPS | ~27.5 | **28.4** | **+3.3%** |

**分析:**

- **TTFT 显著提升**: CUDA Graph 将 40 层 LLM + 24 层 Vision Encoder 的数百次 kernel launch 合并为单次 GPU 图调用，消除了 CPU-GPU 往返延迟。多模态首 token 延迟减半。
- **TPS 小幅提升**: Decode 阶段每 token 只算一次 attention（batch=1），算术强度仅 ~1 FLOP/byte，受 A10 HBM 带宽（600 GB/s）限制。kernel launch 开销在总时间中占比很小，因此即便完全消除也效果有限。
- **预判**: batch > 4 或长 prefill 场景（多用户并发/高分辨率图）时，CUDA Graph 收益将更显著。

### 1.4 算子执行后端分布

| 后端 | 颜色 | 每推理调用次数 | 占比 |
|------|------|-------------|------|
| cuBLAS/cuDNN | 🔵 | 12,496 | 18.6% |
| sgl_kernel RMSNorm | 🟢 | 8,160 | 12.2% |
| sgl_kernel FusedAddRMSNorm | 🟢 | 8,080 | 12.0% |
| sgl_kernel SiLU | 🟢 | 4,040 | 6.0% |
| Triton Attention | 🟡 | 4,040 | 6.0% |
| sglang JIT RoPE | 🟠 | 4,040 | 6.0% |
| 其他（PyTorch Native / Sampling 等）| 🔴 | ~26,604 | 39.2% |
| **合计** | | **~41,460** | 100% |

> 95% 的调用发生在 Decode 阶段（逐 token 生成），5% 在 Prefill/Vision 阶段。

---

## 二、Agent 行为验证

### 2.1 系统 Prompt

| 测试项 | 结果 | 说明 |
|--------|------|------|
| 动态日期注入 | ✅ | 中文: `2026年06月19日 星期五`，英文: `2026-06-19, Friday` |
| 操作指令 (ZH) | ✅ 15/15 | Launch, Tap, Type, Type_Name, Interact, Swipe, Note, Call_API, Long Press, Double Tap, Take_over, Back, Home, Wait, finish |
| 操作指令 (EN) | ✅ 7/7 | Tap, Type, Swipe, Long Press, Launch, Back, Finish |
| 规则条数 (ZH) | ✅ 18 条 | 包含坐标范围限制、Launch 优先于 Tap 等关键规则 |
| Prompt 嵌入日期 | ✅ | ZH: 3403 字符，EN: 2442 字符 |

### 2.2 App 映射表

| 测试项 | 结果 | 说明 |
|--------|------|------|
| 微信 → com.tencent.mm | ✅ | |
| 淘宝 → com.taobao.taobao | ✅ | |
| 抖音 → com.ss.android.ugc.aweme | ✅ | |
| 支付宝 → com.eg.android.AlipayGphone | ✅ 已补充 | 原始映射表缺失，本次测试中补充 |
| 不存在 app → None | ✅ | 正确处理边界 |
| 支持 App 总数 | 169 | 含中英文同义词变体 |

### 2.3 Action 解析器

| 测试项 | 结果 | 说明 |
|--------|------|------|
| `do(action="Launch", app="微信")` | ✅ | |
| `do(action="Tap", element=[500, 300])` | ✅ | 坐标解析正确 |
| `do(action="Swipe", start=[500,800], end=[500,200])` | ✅ | |
| `do(action="Type", text="Hello World")` | ✅ | |
| `do(action="Type", text="你好世界")` | ✅ | 中文文本正确处理 |
| `do(action="Wait", duration="2 seconds")` | ✅ | |
| `finish(message="任务完成")` | ✅ | |
| 畸形输入 → ValueError | ✅ | 不完整括号、空字符串、随机文本均正确拒绝 |
| `do(action="Type_Name")` 别名化 | ⚠️ 设计行为 | 内部将 Type_Name 别名化为 Type |
| 嵌套引号 (Type 路径) | ⚠️ 已知限制 | Type 使用简单字符串分割，不处理转义 |

### 2.4 消息构建

| 测试项 | 结果 |
|--------|------|
| `build_screen_info("微信")` → `{"current_app":"微信"}` | ✅ |
| `create_user_message` (含图) → 正确 image_url + text 结构 | ✅ |
| `remove_images_from_message` → 只保留 text 块 | ✅ |
| 多轮对话 (system → user+img → assistant → user) | ✅ |

### 2.5 模型行为 (静态截图)

#### 2.5.1 Action 准确性

| 用户指令 | 期望 Action | 模型输出 | 结果 |
|----------|-----------|---------|------|
| 打开微信 | Launch / 微信 | `do(action="Launch", app="微信")` | ✅ |
| 打开淘宝 | Launch / 淘宝 | `do(action="Launch", app="淘宝")` | ✅ |
| 点击QQ图标 | Tap | `do(action="Tap", element=[848,252])` | ✅ |
| 往下滑动查看更多应用 | Swipe | `do(action="Swipe", start=[499,746], end=[499,262])` | ✅ |
| 等待2秒 | Wait | `do(action="Wait", duration="2 seconds")` | ✅ |
| 返回 | Back | `do(action="Back")` | ✅ |
| 回到桌面 | Home | `finish("已经在桌面了")` | ⚠️ 智能拒绝* |
| 现在屏幕上有什么应用 | finish | 详细列出 11 个应用后 finish | ✅ 语义正确** |

> **准确率: 8/8 (100%) 语义正确，6/8 (75%) 字面匹配，2 项为合理的智能决策**

> \* "回到桌面" —— 模型识别出当前已在桌面，因此使用 finish 而非 Home。这是正确的智能行为，避免无效操作。
> \** "屏幕上有什么应用" —— 模型正确识别了天气、QQ(51条未读)、淘宝(9条未读)、图库、知乎(99+未读)、**支付宝**、微信(15条未读)、闲鱼(77条未读)、相机、打卡等 11 个应用，语义完全正确。

#### 2.5.2 多步推理

| 测试 | 结果 |
|------|------|
| "发微信消息给张三说我到了" | ✅ 正确规划: Launch 微信 → 找张三 → 发消息 |
| 微信二维码截图 + "看看这条消息" | ✅ 正确识别为群聊二维码，用 finish 解释非消息 |
| USB调试设置页 + "描述当前页面" | ✅ 正确描述所有设置项及其状态 |

#### 2.5.3 边界场景

| 测试 | 结果 |
|------|------|
| "打开一个不存在的应用" | ✅ 尝试查找后 prepare to finish |
| "现在几点了" (纯观察) | ✅ 读取屏幕时钟 `08:55` 后 finish |
| "打开微信但不要打开微信" (矛盾指令) | ✅ 识别矛盾，用 finish 解释并询问用户意图 |
| 最小指令 "返回" | ✅ `do(action="Back")` |

#### 2.5.4 分辨率鲁棒性

| 测试分辨率 | 识别 "打开微信" | 说明 |
|-----------|---------------|------|
| 360 × 800 | ✅ Launch | |
| 720 × 1600 | ✅ Launch | |

> 模型在 2 个测试分辨率下均正确输出 Launch，表明相对坐标系统 + Launch 语义组合具有良好的分辨率不变性。

#### 2.5.5 坐标有效性

所有 Tap/Swipe 动作的坐标均在 **0–999 范围内**，未发现越界。
示例: QQ 图标 `[848, 252]`、Swipe `[499, 746] → [499, 262]`、支付宝 `[382, 645]`。

---

## 三、关键发现

### ✅ 优势

1. **模型泛化能力强**: 9B 模型在手机截图上表现出准确的 UI 理解，能识别图标、通知徽章、文字内容
2. **智能决策**: 模型会在"已满足条件"时拒绝执行无效操作（如已在桌面时拒绝执行 Home）
3. **坐标系统稳健**: 0-1000 相对坐标系在多分辨率下保持一致
4. **CUDA Graphs 仅需一行改动**: 移除 `--disable-cuda-graph` 即可获得 3-54% 提升，无需重新编译或修改模型

### ⚠️ 已知限制

| 问题 | 影响 | 解决方案 |
|------|------|---------|
| 原始分辨率 OOM | 1080×2400 图片无法在 24GB A10 上推理 | 缩小至 720p，TTFT 损失 30ms |
| App 映射表不完整 | 支付宝等常用 app 缺失 | 持续补充 `APP_PACKAGES` |
| TP8 KV Cache 不可用 | 仅 Hopper (SM90+) 支持 | A10 硬限制，无法解决 |
| 单请求 TPS 提升天花板 | 受 HBM 带宽限制 | batch > 1 或升级 A100 |

### 📋 待测试 (需手机 ADB 接入)

- Layer 4: ADB 截图/点击/滑动/文本输入/坐标转换
- Layer 5: 端到端任务执行（打开微信 → 发消息 → 返回桌面）

---

## 四、配置信息

### 当前最优启动参数

```bash
sglang serve \
  --model-path <MODEL_PATH> \
  --served-model-name autoglm-phone-9b \
  --context-length 4096 \
  --mm-enable-dp-encoder \
  --port 8000 \
  --host 0.0.0.0 \
  --mem-fraction-static 0.9 \
  --attention-backend triton \
  --sampling-backend pytorch \
  --mm-attention-backend sdpa
```

关键变更: 移除了 `--disable-cuda-graph`（CUDA Graphs 默认开启）。

### 硬件资源占用

| 资源 | 占用量 |
|------|-------|
| GPU 显存 | 22.94 GB / 23.55 GB |
| KV Cache | 0.81 GB (bf16, 42239 tokens) |
| 可用显存 (空闲) | ~2.04 GB |
| CUDA Graph 额外显存 | 0.08 GB |

### 推理预计 (单请求/batch=1)

| 场景 | TTFT | TPS |
|------|------|-----|
| 纯文本 | ~75ms | ~28 tok/s |
| 多模态 (720p) | ~106ms | ~28 tok/s |
| 多模态 (冷启动) | ~450ms | ~28 tok/s |

---

*测试脚本: `scripts/bench_resolution.py`, `/tmp/bench_*.py`*
*环境: CUDA 13.2 (pip), PyTorch 2.8.0, SGLang 0.5.x, NVIDIA A10*
