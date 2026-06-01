"""
LLMClient - 工程化的大模型调用客户端

主要功能：
- 异步LLM调用（基于AsyncOpenAI）
- 多供应商支持（云雾、Deepseek、WD等）
- 完整的调用日志记录（每次调用生成唯一ID）
- Token使用量和成本追踪
- 基于Pydantic的结构化数据提取
- 自动验证和修复机制

使用示例：
    from LLMClient_v2 import LLMClient, StructuredLLMClient
    from LLMClient_v2.models import ProcurementInfo

    # 基础文本调用
    client = LLMClient(api_name="yunwu_gpt_4o_chat")
    response = await client.get_completion(prompt="你好")

    # 结构化数据提取
    structured_client = StructuredLLMClient(api_name="yunwu_gpt_4o_chat")
    result = await structured_client.get_structured_completion(
        text_input="某市医院采购设备...",
        response_model=ProcurementInfo
    )
"""

__version__ = "1.1.0"

from .llm_runtime_config import (
    LLMRuntimeConfig,
    resolve_runtime_config,
    list_runtime_profiles,
    get_default_api,
)

from .cli_args import (
    add_llm_cli_args,
    resolve_llm_cli_config,
    create_client_from_cli,
)

from .token_usage_tracker import (
    TokenUsageTracker,
    TokenUsage,
    ModelPricing,
    ModelPricingConfig,
)

from .llm_client import (
    LLMClient,
    LLMResponse,
)

from .anthropic_client import (
    AnthropicLLMClient,
    AnthropicLLMResponse,
)

from .structured_llm_client import (
    StructuredLLMClient,
)

from .prompt_utils import (
    fill_prompt_with_document,
    fill_prompt_with_variables,
    set_prompt_root,
    get_prompt_template_path,
    load_prompt_template,
    render_prompt_template,
    extract_json_from_response,
    extract_and_repair_json,
    clean_json_response,
)

from .task_manager import (
    TaskManager,
    BatchTaskManager,
    BatchTaskExecutor,
    BatchExecutionContext,
    TaskExecutionResult,
    LLMActivityLog,
)

from .models import (
    ProcurementInfo,
)

__all__ = [
    # 配置
    "LLMRuntimeConfig",
    "resolve_runtime_config",
    "list_runtime_profiles",
    "get_default_api",
    "add_llm_cli_args",
    "resolve_llm_cli_config",
    "create_client_from_cli",

    # 追踪器
    "TokenUsageTracker",
    "TokenUsage",
    "ModelPricing",
    "ModelPricingConfig",

    # 客户端
    "LLMClient",
    "LLMResponse",
    "AnthropicLLMClient",
    "AnthropicLLMResponse",
    "StructuredLLMClient",

    # 任务管理
    "TaskManager",
    "BatchTaskManager",
    "BatchTaskExecutor",
    "BatchExecutionContext",
    "TaskExecutionResult",
    "LLMActivityLog",

    # 工具函数
    "fill_prompt_with_document",
    "fill_prompt_with_variables",
    "set_prompt_root",
    "get_prompt_template_path",
    "load_prompt_template",
    "render_prompt_template",
    "extract_json_from_response",
    "extract_and_repair_json",
    "clean_json_response",

    # 示例模型
    "ProcurementInfo",
]
