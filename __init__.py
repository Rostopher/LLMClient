"""
LLMClient - 工程化的大模型调用客户端

主要功能：
- 异步LLM调用（基于AsyncOpenAI）
- 多供应商支持（云雾、Deepseek、WD等）
- 完整的调用日志记录（每次调用生成唯一ID）
- Token使用量和成本追踪
- 基于Pydantic的结构化数据提取
- 自动验证和修复机制
- Stage Routing（按处理阶段自动路由到不同LLM配置）
- 流式输出支持（SSE streaming）

使用示例：
    from LLMClient import LLMClient, StructuredLLMClient, create_llm_client

    # 基础文本调用
    client = LLMClient(api_name="yunwu_gpt_4o_chat")
    response = await client.get_completion(prompt="你好")

    # 通过 stage 自动路由创建客户端
    client = create_llm_client(stage="metadata_extraction")
    result = await client.get_json_completion(prompt="...")

    # 结构化数据提取
    structured_client = StructuredLLMClient(api_name="yunwu_gpt_4o_chat")
    result = await structured_client.get_structured_completion(
        text_input="某市医院采购设备...",
        response_model=ProcurementInfo
    )
"""

__version__ = "1.2.0"

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
    set_global_task_token_file,
    get_global_task_token_file,
    clear_global_task_token_file,
    set_global_task_context,
    get_global_task_context,
    clear_global_task_context,
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

from .stage_routing import (
    StageConfig,
    LLMPipeline,
    get_route_for_stage,
    get_api_for_stage,
    get_stage_config,
    set_stage_route,
    list_stage_routing,
    load_stage_routing,
    reload_stage_routing,
)

from .models import (
    ProcurementInfo,
)


def create_llm_client(
    api_name: str | None = None,
    stage: str | None = None,
    log_file: str | None = None,
    enable_tracking: bool = True,
    **kwargs,
) -> LLMClient:
    """
    LLMClient 工厂函数

    - 若传 stage，自动查 stage routing 获取 api_name/model/temperature
    - 若 log_file=None，自动使用全局 task token file（如已设置）
    - 兼容旧 create_enhanced_llm_client() 的使用模式

    Args:
        api_name: API profile 名称（优先于 stage 路由）
        stage: 处理阶段名称，自动路由到对应 API/model/temperature
        log_file: token 日志文件路径
        enable_tracking: 是否启用 token 追踪
        **kwargs: 传递给 LLMClient 构造函数的其他参数
    """
    effective_api = api_name
    route_model = kwargs.pop("model", None)
    route_temp = kwargs.pop("temperature", None)

    if stage and not api_name:
        route = get_route_for_stage(stage)
        effective_api = route.get("api_name")
        if route.get("model") and route_model is None:
            route_model = route["model"]
        if route.get("temperature") is not None and route_temp is None:
            route_temp = route["temperature"]

    if log_file is None:
        log_file = get_global_task_token_file()

    return LLMClient(
        api_name=effective_api,
        log_file=log_file,
        enable_tracking=enable_tracking,
        model=route_model,
        temperature=route_temp,
        **kwargs,
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

    # 全局 Token 文件路由
    "set_global_task_token_file",
    "get_global_task_token_file",
    "clear_global_task_token_file",
    "set_global_task_context",
    "get_global_task_context",
    "clear_global_task_context",

    # 客户端
    "LLMClient",
    "LLMResponse",
    "AnthropicLLMClient",
    "AnthropicLLMResponse",
    "StructuredLLMClient",

    # 工厂函数
    "create_llm_client",

    # Stage Routing & Pipeline
    "StageConfig",
    "LLMPipeline",
    "get_route_for_stage",
    "get_api_for_stage",
    "get_stage_config",
    "set_stage_route",
    "list_stage_routing",
    "load_stage_routing",
    "reload_stage_routing",

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
