[根目录](../CLAUDE.md) > **LLMClient_v2**

# LLMClient_v2 - 工程化的大模型调用客户端

> 版本: 1.0.0 | 最后更新: 2026-02-08

## 模块职责

提供工程化的大模型调用客户端封装，支持：

- **多供应商支持**: 云雾 AI、DeepSeek、WD、Anthropic 等
- **异步调用**: 基于 AsyncOpenAI 的异步 API 调用
- **Token追踪**: 自动统计 Token 使用量和计算成本
- **结构化输出**: 基于 Pydantic 的数据验证和 JSON Schema 生成
- **批量任务管理**: 支持大规模任务的断点续传和多轮重试

## 入口与启动

### 导入方式

```python
from LLMClient_v2 import LLMClient, StructuredLLMClient
from LLMClient_v2.llm_runtime_config import resolve_runtime_config, list_runtime_profiles
```

### 初始化客户端

```python
import asyncio
from LLMClient_v2 import LLMClient

async def main():
    client = LLMClient(
        api_name="deepseek_official",  # 使用配置的 API 名称
        log_file="logs/llm_activity.jsonl"  # 日志文件
    )
```

## 对外接口

### 基础 LLM 调用 (`LLMClient`)

```python
response = await client.get_completion(
    prompt="你的提示词",
    temperature=0.7,           # 温度参数
    max_tokens=1024,           # 最大输出 token
    stage="metadata_extraction",  # 阶段路由
    metadata={"user_id": "123"}   # 元数据
)

# 响应属性
response.success       # bool: 是否成功
response.content       # str: 模型输出
response.usage         # TokenUsage: Token 统计
response.cost          # Dict: 成本信息
response.call_id       # str: 唯一调用 ID
response.duration_ms   # int: 耗时(ms)
```

### 结构化数据提取 (`StructuredLLMClient`)

```python
from LLMClient_v2 import StructuredLLMClient
from pydantic import BaseModel

class MyModel(BaseModel):
    name: str
    value: float

result = await client.get_structured_completion(
    text_input="输入文本",
    response_model=MyModel,
    enable_repair=True  # 启用自动修复
)
```

### Anthropic SDK 适配 (`AnthropicLLMClient`)

```python
from LLMClient_v2 import AnthropicLLMClient

client = AnthropicLLMClient(
    api_name="anthropic_antigravity_local",
    log_file="logs/anthropic_activity.jsonl"
)
```

### 任务管理 (`TaskManager`)

```python
from LLMClient_v2 import TaskManager, BatchTaskExecutor

manager = TaskManager("tasks.jsonl")
executor = BatchTaskExecutor(
    task_manager=manager,
    llm_client=client,
    concurrency=5
)
results = await executor.run_all()
```

### Prompt 工具函数

```python
from LLMClient_v2.prompt_utils import (
    load_prompt_template,
    render_prompt_template,
    fill_prompt_with_variables,
    extract_json_from_response,
    extract_and_repair_json,
)
```

### Token 使用追踪

```python
from LLMClient_v2 import TokenUsageTracker

tracker = TokenUsageTracker("usage.jsonl")
tracker.log_call_record({
    "call_id": "xxx",
    "usage": {"prompt_tokens": 100, "completion_tokens": 50}
})

summary = tracker.get_session_summary()
```

## 关键依赖与配置

### 核心依赖

```txt
openai>=1.0.0      # 异步 LLM 调用
pydantic>=2.0.0    # 数据验证
anthropic>=0.0.0   # Anthropic SDK (可选)
json5>=0.9.0       # 宽容 JSON 解析
json-repair>=0.0.0 # JSON 自动修复
```

### 配置文件

| 文件 | 作用 |
|------|------|
| `llm_apis.yaml` | Profile 配置：provider/family/protocol/model/env key |
| `llm_runtime_config.py` | 加载 `.env`、YAML、环境变量与显式覆盖 |
| `cli_args.py` | 可复用 argparse helper |
| `cli.py` | `python -m LLMClient_v2.cli` smoke test |
| `token_usage_tracker.py` | Token 统计与成本计算 |

### Profile 配置示例

```yaml
apis:
  yunwu_gpt_4o_chat:
    provider: yunwu
    family: gpt
    protocol: openai_chat
    base_url: https://yunwu.ai/v1
    api_key_env: YUNWU_GPT_API_KEY
    default_model: gpt-4o-2024-08-06

  yunwu_gemini_25_pro_chat:
    provider: yunwu
    family: gemini
    protocol: openai_chat
    base_url: https://yunwu.ai/v1
    api_key_env: YUNWU_GEMINI_API_KEY
    default_model: gemini-2.5-pro
```

`--api` 选择的是 profile，不是 provider。密钥必须使用 profile 明确指定的 `api_key_env`，例如 `YUNWU_GPT_API_KEY` 或 `YUNWU_GEMINI_API_KEY`，不会回退到通用 `YUNWU_API_KEY`。

## 数据模型

### 核心数据类

```python
@dataclass
class LLMResponse:
    success: bool
    content: str
    usage: Optional[TokenUsage]
    cost: Optional[Dict[str, float]]
    error: Optional[str]
    call_id: Optional[str]
    duration_ms: Optional[int]

@dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
```

## 测试与质量

### 测试文件

- `_test_anthropic_model_list.py` - Anthropic 模型列表测试

### 日志格式

每次调用生成 JSON Lines 格式日志：

```json
{
  "call_id": "call_a1b2c3d4",
  "timestamp_start": "2025-10-10T10:30:01.123Z",
  "duration_ms": 2333,
  "status": "success",
  "api_name": "deepseek_official",
  "provider": "deepseek",
  "model": "deepseek-chat",
  "prompt": "...",
  "response": "...",
  "usage": {
    "prompt_tokens": 512,
    "completion_tokens": 128,
    "total_tokens": 640
  },
  "cost": {
    "primary_cost": 0.000128,
    "currency": "USD"
  }
}
```

## 常见问题 (FAQ)

**Q: 如何切换不同的 LLM 供应商?**
A: 在 `llm_apis.yaml` 中配置对应 profile，初始化时指定 `api_name` 参数，或通过 CLI `--api` 选项指定。

**Q: 如何添加新的 API?**
A: 在 `llm_apis.yaml` 的 `apis` 节点下添加新 profile，指定 provider、base_url、api_key_env、default_model 等。

**Q: Token 追踪如何计算成本?**
A: `TokenUsageTracker.pricing_config.calculate_cost()` 根据 provider 类型使用不同计算公式。

## 相关文件清单

| 文件 | 说明 |
|------|------|
| `__init__.py` | 模块导出入口 |
| `llm_client.py` | 基础 LLM 客户端 |
| `structured_llm_client.py` | 结构化输出客户端 |
| `anthropic_client.py` | Anthropic SDK 适配 |
| `task_manager.py` | 批量任务管理 |
| `token_usage_tracker.py` | Token 统计 |
| `prompt_utils.py` | Prompt 工具 |
| `llm_runtime_config.py` | 运行时配置解析 |
| `models.py` | Pydantic 模型 |
| `README.md` | 详细文档 |
| `ARCHITECTURE.md` | 架构设计 |

## 变更记录

| 日期 | 变更内容 |
|------|----------|
| 2026-06-01 | 移除对已废弃 llm_config.py 的所有引用，FAQ 更新为 llm_apis.yaml 配置方式 |
| 2026-02-08 | 初始化模块文档，添加 Anthropic 适配器 |
