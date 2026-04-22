# Notion2API

> 将 Notion AI 封装为 OpenAI 兼容 API 的本地代理服务

Notion2API 通过逆向 Notion 内置 AI 接口，将其包装为标准的 OpenAI 兼容 API，让你无需付费订阅任何 AI 服务，即可在 Cherry Studio、Zotero、Claude Code 等主流客户端中直接使用 Claude、GPT、Gemini 等顶级模型。

---

## 功能特性

- **OpenAI 兼容** — 标准 `/v1/chat/completions` 接口，无缝接入主流客户端
- **Anthropic 兼容** — 额外支持 `/v1/messages`，可供 Claude Code 等原生 SDK 直接使用
- **流式响应（SSE）** — 实时逐字输出，体验流畅
- **Thinking 面板** — 支持所有模型展示推理过程
- **网络搜索** — 支持展示 AI 的 Web 搜索来源
- **多账号池** — 多账号轮询负载均衡，自动故障转移
- **三种运行模式** — Lite / Standard / Heavy，按需选择

---

## 三种模式对比

| 特性 | Lite | Standard | Heavy |
|------|------|----------|-------|
| **记忆管理** | ❌ 无 | ✅ 客户端管理 | ✅ 服务端 SQLite |
| **数据库** | ❌ 不需要 | ❌ 不需要 | ✅ SQLite |
| **Thinking 面板** | ❌ | ✅ | ✅ |
| **搜索结果** | ❌ | ✅ | ✅ |
| **速率限制** | 30次/分 | 25次/分 | 20次/分 |
| **适用场景** | 简单问答 | 中短期对话 | 长期记忆对话 |

> 修改 `.env` 中的 `APP_MODE` 变量即可切换模式，**推荐日常使用 `standard`**。

---

## 快速开始

### 第一步：获取 Notion 凭据

> 你需要一个已订阅 Notion AI 的账号（或处于 AI 试用期的账号）。

1. 打开浏览器，访问 [https://www.notion.so/ai](https://www.notion.so/ai) 并登录
2. 按 `F12` 打开开发者工具

**获取 `token_v2`：**
- 切换到 **Application**（应用程序）标签
- 左侧展开 **Storage → Cookies → https://www.notion.so**
- 找到名为 `token_v2` 的条目，复制其 **Value（值）**

**获取其他字段（space_id、user_id 等）：**
- 切换到 **Console**（控制台）标签
- 将 `scripts/extract_notion_info.js` 文件中的代码完整粘贴进去，回车运行
- 脚本会自动输出一段 JSON，将其中的 `YOUR_TOKEN_V2_HERE` 替换为上一步复制的 `token_v2` 值
- 复制完整输出，备用

---

### 第二步：配置 `.env` 文件

打开项目根目录的 `.env` 文件，填入你的凭据：

```dotenv
# 将上一步获取的 JSON 粘贴到这里（保留外侧的单引号）
NOTION_ACCOUNTS='[{"token_v2":"你的token","space_id":"你的space_id","user_id":"你的user_id","space_view_id":"你的space_view_id","user_name":"你的名字","user_email":"你的邮箱"}]'

# 运行模式，推荐 standard
APP_MODE=standard
```

**如果使用 Heavy 模式**，还需额外配置（用于对早期对话进行摘要压缩）：
1. 前往 [https://siliconflow.cn](https://siliconflow.cn) 免费注册，获取 API Key
2. 填入 `.env`：
   ```dotenv
   SILICONFLOW_API_KEY=你的SiliconFlow_API_Key
   APP_MODE=heavy
   ```

---

### 第三步：安装依赖并启动

```bash
# 1. 安装 Python 依赖（首次运行时执行）
pip install -r requirements.txt

# 2. 启动服务
uvicorn app.server:app --host 0.0.0.0 --port 8000 --reload
```

服务启动后访问：

| 地址 | 说明 |
|------|------|
| `http://localhost:8000/health` | 服务健康状态检查 |
| `http://localhost:8000/v1/models` | 查看可用模型列表 |

---

## 支持的模型

| 模型名称 | 说明 |
|----------|------|
| `claude-sonnet4.6` | **最推荐** — 速度与质量最佳平衡，优化最完善 |
| `claude-opus4.6` | 推理能力更强，但响应较慢，不建议高频使用 |
| `gemini-3.1pro` | Google 模型，可正常访问，但不支持网络搜索 |
| `gpt-5.2` | OpenAI 最新模型，效果不错 |
| `gpt-5.4` | OpenAI 最新模型，效果不错 |

完整模型列表：`GET http://localhost:8000/v1/models`

---

## 在客户端中使用

本项目兼容所有支持自定义 API 地址的 OpenAI 客户端，配置方式通用：

| 配置项 | 值 |
|--------|----|
| **API Base URL** | `http://localhost:8000/v1` |
| **API Key** | 若 `.env` 中 `API_KEY` 为空，填任意字符即可；否则填写你设定的值 |
| **模型** | `claude-sonnet4.6`（推荐）或其他支持模型 |

### Python 调用示例

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="any_key"  # API_KEY 未设置时随意填写
)

response = client.chat.completions.create(
    model="claude-sonnet4.6",
    messages=[{"role": "user", "content": "你好，介绍一下你自己"}],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

### Claude Code / Anthropic SDK 调用示例

本项目同时兼容 Anthropic 原生 SDK（`/v1/messages` 端点）：

```python
import anthropic

client = anthropic.Anthropic(
    base_url="http://localhost:8000",
    api_key="any_key"
)

message = client.messages.create(
    model="claude-sonnet4.6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "你好"}]
)
print(message.content[0].text)
```

---

## 环境变量说明

> 所有配置均在项目根目录的 `.env` 文件中修改。

| 变量 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `NOTION_ACCOUNTS` | ✅ 必填 | — | Notion 账号凭据，JSON 数组格式 |
| `APP_MODE` | — | `heavy` | 运行模式：`lite` / `standard` / `heavy` |
| `API_KEY` | — | 空（不鉴权） | 客户端请求时需携带的 Bearer Token，留空则不验证 |
| `HOST` | — | `0.0.0.0` | 服务绑定 IP |
| `PORT` | — | `8000` | 服务监听端口 |
| `DB_PATH` | — | `./data/conversations.db` | Heavy 模式 SQLite 数据库路径 |
| `SILICONFLOW_API_KEY` | — | 空 | Heavy 模式使用，用于压缩早期对话摘要 |
| `LOG_LEVEL` | — | `INFO` | 日志级别：`DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `TZ` | — | `Asia/Shanghai` | 时区设置 |

---

## 多账号配置

多账号可以分散请求，提升稳定性，避免单账号触发 Notion 速率限制。

在 `.env` 中，`NOTION_ACCOUNTS` 支持配置多个账号（JSON 数组）：

```dotenv
NOTION_ACCOUNTS='[
  {"token_v2":"token1","space_id":"space1","user_id":"uid1","space_view_id":"view1","user_name":"账号1","user_email":"email1@example.com"},
  {"token_v2":"token2","space_id":"space2","user_id":"uid2","space_view_id":"view2","user_name":"账号2","user_email":"email2@example.com"}
]'
```

服务会自动对多账号进行 **Round-Robin 轮询**，失败账号进入冷却期后自动恢复。

---

## 常见问题

**Q：Thinking 面板不显示？**
确保 `APP_MODE` 设置为 `standard` 或 `heavy`，Lite 模式不支持 Thinking 展示。

**Q：响应延迟较高（约 3 秒）？**
这是 Notion 官方 AI 接口本身的限制，无法规避。因此不推荐在对延迟敏感的场景（如沉浸式翻译）中使用。

**Q：token 失效怎么办？**
重新登录 Notion，按照「第一步」重新获取 `token_v2` 并更新 `.env` 文件，然后重启服务即可。

**Q：如何设置 API 访问鉴权？**
在 `.env` 中设置 `API_KEY=你的密钥`，客户端请求时在 Header 中携带 `Authorization: Bearer 你的密钥`。

**Q：Heavy 模式记忆是如何工作的？**
- **滑动窗口**：默认保留最近 8 轮对话（16 条消息）
- **摘要压缩**：超出窗口的历史由 SiliconFlow API 自动压缩为简短摘要注入上下文
- **完整归档**：所有历史永久存储在本地 SQLite 数据库

---

## 兼容性测试

> 注意：由于 Notion AI 接口特性，请求通常有约 **3 秒**的初始延迟，不适合对响应速度要求极高的场景。

| 客户端 | 状态 | 备注 |
|--------|------|------|
| Cherry Studio | ✅ 完美支持 | 推荐，体验最佳 |
| Zotero 翻译 | ✅ 完美支持 | 速度略慢，但 sonnet 模型翻译准确 |
| Claude Code | ✅ 完美支持 | 使用 `/v1/messages` 端点 |
| 沉浸式翻译 | ⚠️ 不推荐 | 延迟过高，影响使用体验 |

---

## 许可证

MIT License — 开源免费，欢迎二次开发。

---

如有问题或建议，欢迎提交 Issue 反馈！
