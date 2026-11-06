# LLMClient - 工程化的大模型调用客户端

一个遵循 Python 最佳实践的、工程化的 LLM 调用客户端，支持多供应商、完整日志追踪、成本计算和结构化数据提取。

## 豆包计量与独立计价

Python 可以直接 `from LLMClient import TokenUsageTracker, ModelPricingConfig`。Node/TypeScript
不能直接 import Python 包；可保存供应商原始 usage，交给 Python 账户服务或离线脚本计算，
不需要复制价格公式或迁移模型请求。计价函数本身不访问网络。

```python
from LLMClient import TokenUsageTracker, ModelPricingConfig

pricing = ModelPricingConfig()
usage = TokenUsageTracker.extract_usage_from_response(response_json)
cost = pricing.calculate_cost(
    usage,
    model_name=response_json["model"],
    provider="volcengine",  # 必须明确官方供应商，不能混用第三方/默认价
    input_modality="audio",  # 纯文本用 text；音频输入要求 audio_tokens 实际存在
    service_tier=response_json.get("service_tier", "default"),
    cache_mode="implicit",
)
speech_cost = pricing.calculate_speech_cost({"text_words": 18}, "seed-tts-2.0")
```

新增 `audio_tokens`、`audio_cached_tokens` 和 `raw_usage`；缺失与零分别保存。
音频和缓存是总输入的子集，分别计价，不能叠加重复收费。
当前已核价版本为 `doubao-seed-2-1-lite-260915` 常规在线、0–1024K 输入档位，币种 CNY；
未知官方模型、非 default tier、不完整或矛盾的计量会抛 `ValueError`，不回退默认美元价格。
官方精确计价要求原始 `prompt_tokens_details` 为对象且实际包含合法的 `cached_tokens`；
缺明细、空对象或字段缺失不能当作零缓存命中。纯文本请求缺少 `audio_tokens` 可按无音频计价。
官方隐式缓存只在命中音频缓存时返回 `audio_cached_tokens`，缺省按未命中处理；
显式音频缓存缺少细分时拒绝精确计价。显式缓存存储费不包含在单请求推理成本中。
会话音频计数从零累计；某次记录真实缺失的音频字段仍保留 None，并使对应累计字段为未知，
直到 `reset_session()` 开始新会话，不把缺失上报当作零用量。

TTS 2.0 按供应商 `usage.text_words` 计字符，先加请求头
`X-Control-Require-Usage-Tokens-Return: *`；流中累计统计不重复相加。
`price_version=volc-public-cn-2026-11-04` 是查价快照日期；`primary_cost` 是优惠/免费额度抵扣前的
公开价成本，不是供应商现金账单。当前 `get_session_summary()` 是旧会话汇总接口，
其 currency 默认值不能用于跨币种财务汇总；本次集成使用每条 cost 的币种与金额。

上述官方豆包 Chat 与 TTS 计价结果统一新增 `primary_cost_exact`、`standard_cost_exact`，
均从现有 Decimal 计算结果直接导出为不使用科学计数法的十进制字符串，单位沿用 `currency`
（当前为 CNY 元），不预先舍入或固定小数位数。原 `primary_cost`、`standard_cost` 保持 float。
账户侧应将精确字符串转为 Decimal，按操作累计成本后乘积分系数，再向上取整一次；
识别和翻译合计取整，朗读操作单独取整。不要先转 float，也不要逐次调用先取整后相加：

```python
from decimal import Decimal, ROUND_CEILING

operation_costs = [cost]  # 同一操作的成功调用成本；加入识别/翻译的各次 cost
total = sum((Decimal(item["primary_cost_exact"]) for item in operation_costs), Decimal(0))
points = (total * Decimal("300")).to_integral_value(rounding=ROUND_CEILING)
# 300 仅为产品可配置的本地初值；计价函数本身不决定积分或执行扣费。
```

本次集成复用计量/计价函数，音频和文字请求仍由粤来粤懂现有 Node 函数发送；
未改变 `llm_apis.yaml`、默认供应商或已有调用重试策略。

正式回归（不访问外网）：

```sh
python -m pytest test_volcengine_pricing.py _test_opencode_go_pricing.py _test_runtime_config.py _test_llm_fallback.py -q
```

覆盖模态/缓存拆分、原始 usage 保存、Chat/TTS 精确金额与旧 float 兼容、累计成本取整临界值、
TTS 计价、缺失与非法计量拒绝，以及旧 DeepSeek 与
配置/fallback 行为。真实调用对照入口见 `../cantonese-translator/scripts/probe_doubao_usage.ts`
和 `probe_price_usage.py`；它们是有费用、需明确运行的探针，不加入默认测试。

## ✨ 核心特性

### 1. **基础 LLM 调用 (`LLMClient`)**
- ✅ 异步调用（基于 `AsyncOpenAI`）
- ✅ 自动重试机制（指数退避）
- ✅ 空响应检测
- ✅ 完整的元数据日志（每次调用生成唯一 ID，支持追溯）
- ✅ Token 使用量统计
- ✅ 多供应商成本计算（云雾专用公式 + 标准计费）

### 2. **结构化数据提取 (`StructuredLLMClient`)**
- ✅ 基于 Pydantic 的数据验证
- ✅ 自动生成 JSON Schema Prompt
- ✅ 验证失败时自动修复（二次调用 LLM）
- ✅ 批量处理支持

### 3. **批量任务管理 (`TaskManager`)**
- ✅ 大规模批量处理支持（10万+ 任务）
- ✅ 状态追踪（pending / success / failed）
- ✅ 断点续传（程序崩溃后恢复）
- ✅ 多轮重试支持
- ✅ 完整的统计和报告

### 4. **多供应商支持**
- ✅ 云雾 AI（Gemini、GPT、Deepseek）
- ✅ Deepseek 官方
- ✅ WD 渠道
- ✅ 配置化设计，易于扩展

### 5. **阶段路由**
- ✅ 为不同任务配置不同的 API/模型/温度
- ✅ 支持快速切换

## 📦 安装依赖

```bash
# 推荐使用uv
uv add openai pydantic json5 json-repair

# 或使用pip
pip install openai pydantic json5 json-repair
```

**依赖说明**：
- `openai`: 异步LLM API调用
- `pydantic`: 数据验证和Schema生成
- `json5`: 宽容的JSON解析（支持注释、尾部逗号等）
- `json-repair`: JSON自动修复（处理格式错误）

## 🚀 快速开始

### 基础文本调用

```python
import asyncio
from LLMClient import LLMClient

async def main():
    # 创建客户端
    client = LLMClient(
        api_name="yunwu_gemini",  # 使用云雾的 Gemini
        log_file="logs/my_app.jsonl"  # 日志文件路径
    )
    
    # 调用 LLM
    response = await client.get_completion(
        prompt="请用一句话介绍 Python",
        temperature=0.7,
        metadata={"user_id": "12345", "session": "test"}  # 可选的额外元数据
    )
    
    if response.success:
        print(f"回复: {response.content}")
        print(f"Token: {response.usage.total_tokens}")
        print(f"成本: ${response.cost['primary_cost']:.6f}")
        print(f"调用ID: {response.call_id}")
    else:
        print(f"调用失败: {response.error}")

asyncio.run(main())
```

### 结构化数据提取

```python
import asyncio
from LLMClient import StructuredLLMClient
from LLMClient.models import ProcurementInfo

async def main():
    # 创建结构化客户端
    client = StructuredLLMClient(api_name="yunwu_gemini")
    
    # 待提取的文本
    text = """
    某市人民医院发布采购公告，采购医疗设备一批。
    经过公开招标，某医疗设备有限公司中标，
    采购数量为10台，中标金额为150000元。
    """
    
    # 提取结构化数据
    result = await client.get_structured_completion(
        text_input=text,
        response_model=ProcurementInfo,  # Pydantic 模型
        enable_repair=True  # 启用自动修复
    )
    
    if result:
        print(result.model_dump_json(indent=2, ensure_ascii=False))
        # 输出:
        # {
        #   "purchaser": "某市人民医院",
        #   "winner": "某医疗设备有限公司",
        #   "item_name": "医疗设备",
        #   "quantity": 10,
        #   "amount": 150000.0
        # }

asyncio.run(main())
```

### 使用阶段路由

```python
# 为特定阶段创建客户端（自动从 llm_config.py 读取配置）
client = LLMClient.from_stage("metadata_extraction")

response = await client.get_completion(
    prompt="...",
    stage="metadata_extraction"  # 会覆盖 model 和 temperature
)
```

### 自定义 Pydantic 模型

```python
from pydantic import BaseModel, Field
from typing import Optional

class MyDataModel(BaseModel):
    """你的自定义数据模型"""
    name: str = Field(..., description="名称")
    value: Optional[float] = Field(None, description="数值")

# 使用
result = await client.get_structured_completion(
    text_input="...",
    response_model=MyDataModel
)
```

## 📊 日志和追踪

### 调用日志格式

每次调用都会生成一条 JSON Lines 格式的日志：

```json
{
  "call_id": "call_a1b2c3d4",
  "timestamp_start": "2025-10-10T10:30:01.123Z",
  "timestamp_end": "2025-10-10T10:30:03.456Z",
  "duration_ms": 2333,
  "status": "success",
  "api_name": "yunwu_gemini",
  "provider": "yunwu",
  "model": "gemini-2.5-pro",
  "temperature": 0.7,
  "prompt": "...",
  "response": "...",
  "usage": {
    "prompt_tokens": 512,
    "completion_tokens": 128,
    "total_tokens": 640
  },
  "cost": {
    "yunwu_formula_cost": 0.0015,
    "standard_cost": 0.0019,
    "primary_cost": 0.0015,
    "currency": "USD"
  },
  "error": null,
  "metadata": {"user_id": "12345"}
}
```

### 查看会话统计

```python
# 获取当前会话的统计摘要
summary = client.get_session_summary()
print(summary)
# {
#   "total_calls": 10,
#   "successful_calls": 9,
#   "failed_calls": 1,
#   "success_rate": 0.9,
#   "total_usage": {"prompt_tokens": 5000, "completion_tokens": 1200, ...},
#   "total_cost": 0.0234,
#   "average_cost_per_call": 0.00234
# }

# 导出会话记录
client.export_session("session_report.json")

# 重置统计
client.reset_session()
```

## ⚙️ Profile 配置与命令行接入

`LLMClient_v2` 现在以 **profile** 作为可调用端点的最小单位。`provider`、`family`、`protocol`、`model` 分开记录，因此同一个模型可以通过不同供应商或不同协议调用。

默认配置文件：`LLMClient_v2/llm_apis.yaml`。密钥从环境变量、`LLMClient/.env` 或项目根目录 `.env` 读取；已有环境变量优先，且加载行为不依赖启动目录。推荐显式区分供应商/模型族，例如：

```env
YUNWU_GPT_API_KEY=...
YUNWU_GEMINI_API_KEY=...
DEEPSEEK_API_KEY=...
OPENCODE_GO_API_KEY=...
```

不会从 `YUNWU_GPT_API_KEY` / `YUNWU_GEMINI_API_KEY` 回退到通用 `YUNWU_API_KEY`。

### YAML profile 示例

```yaml
apis:
  yunwu_gpt_4o_chat:
    provider: yunwu
    family: gpt
    protocol: openai_chat
    base_url: https://yunwu.ai/v1
    api_key_env: YUNWU_GPT_API_KEY
    default_model: gpt-4o-2024-08-06
    temperature: 0.7

  yunwu_gemini_25_pro_chat:
    provider: yunwu
    family: gemini
    protocol: openai_chat
    base_url: https://yunwu.ai/v1
    api_key_env: YUNWU_GEMINI_API_KEY
    default_model: gemini-2.5-pro

  opencode_go_deepseek_pro:
    provider: opencode_go
    family: deepseek
    protocol: openai_chat
    base_url: https://opencode.ai/zen/go/v1
    api_key_env: OPENCODE_GO_API_KEY
    default_model: deepseek-v4-pro

  opencode_go_deepseek_flash:
    provider: opencode_go
    family: deepseek
    protocol: openai_chat
    base_url: https://opencode.ai/zen/go/v1
    api_key_env: OPENCODE_GO_API_KEY
    default_model: deepseek-v4-flash
```

### 思考模式透传（DeepSeek）

profile 支持两个可选字段，会原样透传到 chat completion 请求（仅 `openai_chat` 协议）：

```yaml
  deepseek_official_chat:
    reasoning_effort: high      # low/high/max；思考模式下 temperature 被服务端忽略
    extra_body:
      thinking:
        type: enabled           # enabled/disabled 思考模式开关
```

也可在 `LLMClient(...)` 构造函数或单次 `get_completion(..., extra_body=..., reasoning_effort=...)` 上覆盖（调用级 > 构造级 > profile）。思考 token 计入 `completion_tokens`，token/成本/缓存统计无需改动。

### OpenCode Go 错误分类与官方 fallback

`llm_error_classifier.py` 会把兼容 OpenAI 的异常统一标记为 `api_key_error`、
`connection_error`、`timeout`、`quota_exhausted`、`rate_limited`、`server_error`
等类型。401 认证失败（以及明确指向 API key/credential 的 403）和带有
`GoUsageLimitError`、`FreeUsageLimitError` 等明确信号的额度耗尽会停止当前
供应商重试，并按模型切换到官方 DeepSeek：

```text
opencode_go_deepseek_pro   → deepseek_official_chat
opencode_go_deepseek_flash → deepseek_official_flash
```

连接错误可以识别（普通/视觉调用按 retryable 重试，流式调用记录后结束），但不会因网络抖动切换供应商；普通限流、
区域/权限限制和服务端错误也不会触发供应商切换。`LLMResponse`
及 JSONL 调用日志会记录 `error_classification`、`fallback_used`、
`fallback_from` 和 `fallback_reason`，错误正文会脱敏 API key/Bearer token。

### 在业务脚本中一行接入

```python
import argparse
from LLMClient_v2 import add_llm_cli_args, create_client_from_cli

parser = argparse.ArgumentParser()
add_llm_cli_args(parser)
args = parser.parse_args()

client = create_client_from_cli(args)
```

常用参数：

```bash
--api yunwu_gpt_4o_chat
--llm-config config/llm_apis.yaml
--env-file .env
--model gpt-4o-2024-08-06
--temperature 0.1
--max-retries 3
--base-url https://yunwu.ai/v1
--api-key-env YUNWU_GPT_API_KEY
--log-file logs/run.jsonl
--max-concurrent 20
--protocol openai_chat
```

优先级：

```text
命令行/显式参数 > 环境变量/.env > YAML profile > legacy llm_config.py
```

### Smoke test

```bash
python -m LLMClient_v2.cli list-apis
python -m LLMClient_v2.cli smoke --api deepseek_official_chat --api-key dummy --prompt "你好" --dry-run
python -m LLMClient_v2.cli smoke --api deepseek_official_chat --prompt "你好"
python -m LLMClient_v2.cli smoke --api opencode_go_deepseek_pro --prompt "你好"
python -m LLMClient_v2.cli smoke --api opencode_go_deepseek_flash --prompt "你好"
python -m LLMClient_v2.cli smoke --api yunwu_gemini_25_pro_chat --prompt-file prompts/test.txt
```

### 配置阶段路由

```python
LLM_STAGE_ROUTING = {
    "quick_task": "deepseek",  # 简写：只指定 API
    "complex_task": {  # 扩展：指定 API、模型和温度
        "api": "yunwu_gemini",
        "model": "gemini-2.5-pro",
        "temperature": 0.7
    },
}
```

## 🏗️ 架构设计

```
LLMClient/
├── llm_config.py              # 配置管理（API、路由）
├── token_usage_tracker.py     # Token 统计和成本计算
├── prompt_utils.py            # Prompt 工具（填充、JSON 提取）
├── llm_client.py              # 基础 LLM 客户端（核心）
├── structured_llm_client.py   # 结构化客户端（继承 LLMClient）
├── models.py                  # Pydantic 模型示例
└── __init__.py                # 包导出
```

### 类继承关系

```
LLMClient (基类)
    ↓ 继承
StructuredLLMClient (增强类)
```

### 核心流程

1. **LLMClient.get_completion()**
   - 生成 call_id 和元数据
   - 调用 AsyncOpenAI
   - 提取 Token 使用量
   - 计算成本（根据 provider）
   - 记录日志到 TokenUsageTracker
   - 返回 LLMResponse

2. **StructuredLLMClient.get_structured_completion()**
   - 生成包含 JSON Schema 的 Prompt
   - 调用 `super().get_completion()`
   - 验证响应（Pydantic）
   - 如果失败且启用修复：生成修复 Prompt → 再次调用 → 再次验证
   - 返回验证后的 Pydantic 对象

## 💡 最佳实践

### 1. 为不同任务创建不同的客户端实例

```python
# 快速任务使用 Deepseek
quick_client = LLMClient(api_name="deepseek")

# 复杂任务使用 GPT-5
complex_client = LLMClient(api_name="yunwu_gpt5")
```

### 2. 使用较低温度进行结构化提取

```python
result = await client.get_structured_completion(
    text_input=text,
    response_model=MyModel,
    temperature=0.1  # 低温度提高稳定性
)
```

### 3. 利用元数据追溯问题

```python
response = await client.get_completion(
    prompt="...",
    metadata={
        "user_id": "user123",
        "document_id": "doc456",
        "operation": "extract_metadata"
    }
)

# 后续可以通过 call_id 或 metadata 在日志中快速定位
```

### 4. 定期导出和分析日志

```python
# 处理完一批任务后
client.export_session("reports/batch_001.json")
client.reset_session()
```

## 🔧 扩展

### 添加新的成本计算逻辑

在 `token_usage_tracker.py` 的 `ModelPricingConfig.calculate_cost()` 中：

```python
def calculate_cost(self, usage, model_name, provider, ...):
    # 标准计算
    standard_cost = ...
    
    # 云雾专用计算
    if provider == "yunwu":
        yunwu_cost = ...
    
    # 添加你的供应商
    if provider == "my_provider":
        my_cost = ...
        result["my_formula_cost"] = my_cost
        result["primary_cost"] = my_cost
    
    return result
```

## 📝 License

MIT

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！
# Seed ASR2 时长计价（生产门禁关闭）

`asr_duration_pricing.py` 为独立增量：固定 `volc.seedasr.sauc.duration`、1 CNY/小时、版本 `volc-seedasr2-duration-cn-2026-11-06`，区分最大时长预算 cap、音频 metadata 和已确认可计费秒数。官方逐请求舍入及账单 usage 来源未确认，因此正式 `calculate_asr_duration_cost` 输入当前均拒绝；只有显式 local mock 来源可计算合成费用，不允许用布尔标志启生产。原 Chat/TTS 方法不变。

离线测试入口：`python -m pytest <本包>/test_asr_duration_pricing.py <本包>/test_volcengine_pricing.py -q`；不访问供应商，不消耗额度。使用方是相邻 cantonese-translator 的独立账户钱包，产品门禁与操作说明由其 `STREAMING-ASR.md` 维护。
