#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Token使用量统计和成本计算模块

基于不同供应商的返回格式统计token消耗，计算实际成本
支持不同模型的定价配置和成本分析，以及完整的调用日志记录
"""

import os
import json
import time
import contextvars
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class TokenUsage:
    """Token使用量数据类"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_cache_hit_tokens: int = 0
    prompt_cache_miss_tokens: int = 0

    def __add__(self, other: 'TokenUsage') -> 'TokenUsage':
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            prompt_cache_hit_tokens=self.prompt_cache_hit_tokens + other.prompt_cache_hit_tokens,
            prompt_cache_miss_tokens=self.prompt_cache_miss_tokens + other.prompt_cache_miss_tokens,
        )


@dataclass
class ModelPricing:
    """模型定价配置"""
    model_name: str
    input_price_per_million: float
    output_price_per_million: float
    multiplier: float = 1.0
    completion_multiplier: float = 1.0
    cache_hit_input_price_per_million: float = 0.0
    currency: str = "USD"


class ModelPricingConfig:
    """模型定价配置管理器"""
    
    def __init__(self):
        """初始化默认定价配置"""
        self.base_rate = 0.002  # 基准1倍率 = $0.002
        self.default_group_discount = 1.0  # 默认令牌分组折扣
        
        # 预设模型定价（基于云雾平台实际配置和市场价格）
        self.model_configs = {
            "gemini-2.5-flash": ModelPricing(
                model_name="gemini-2.5-flash",
                input_price_per_million=0.3,
                output_price_per_million=2.5,
                multiplier=0.15,
                completion_multiplier=8.333333333333334
            ),
            "gemini-2.5-pro": ModelPricing(
                model_name="gemini-2.5-pro", 
                input_price_per_million=1.25,
                output_price_per_million=10.0,
                multiplier=0.625,
                completion_multiplier=8.0
            ),
            "gpt-4o": ModelPricing(
                model_name="gpt-4o",
                input_price_per_million=5.0,
                output_price_per_million=15.0,
                multiplier=2.5,
                completion_multiplier=3.0
            ),
            "gpt-4o-2024-08-06": ModelPricing(
                model_name="gpt-4o-2024-08-06",
                input_price_per_million=5.0,
                output_price_per_million=15.0,
                multiplier=2.5,
                completion_multiplier=3.0
            ),
            "gpt-5-2025-08-07": ModelPricing(
                model_name="gpt-5-2025-08-07",
                input_price_per_million=10.0,
                output_price_per_million=30.0,
                multiplier=5.0,
                completion_multiplier=3.0
            ),
            "deepseek-chat": ModelPricing(
                model_name="deepseek-chat",
                input_price_per_million=0.14,
                output_price_per_million=0.28,
                multiplier=0.07,
                completion_multiplier=2.0
            ),
            "deepseek-reasoner": ModelPricing(
                model_name="deepseek-reasoner",
                input_price_per_million=0.55,
                output_price_per_million=2.19,
                multiplier=0.275,
                completion_multiplier=3.98
            ),
            "deepseek-r1": ModelPricing(
                model_name="deepseek-r1",
                input_price_per_million=0.55,
                output_price_per_million=2.19,
                multiplier=0.275,
                completion_multiplier=3.98
            ),
            "deepseek-v4-flash": ModelPricing(
                model_name="deepseek-v4-flash",
                input_price_per_million=1.0,
                output_price_per_million=2.0,
                cache_hit_input_price_per_million=0.02,
                currency="CNY",
            ),
            "deepseek-v4-pro": ModelPricing(
                model_name="deepseek-v4-pro",
                input_price_per_million=3.0,
                output_price_per_million=6.0,
                cache_hit_input_price_per_million=0.025,
                currency="CNY",
            ),
            "default": ModelPricing(
                model_name="default",
                input_price_per_million=1.0,
                output_price_per_million=3.0,
                multiplier=1.0,
                completion_multiplier=3.0
            )
        }

        # OpenCode Go 暂不使用供应商专属价格，复用上面的 DeepSeek 官方模型价格。
        # 保留历史配置作为注释，避免误把旧费率当成当前计费口径。
        self.provider_model_configs = {
            # ("opencode_go", "deepseek-v4-pro"): ModelPricing(
            #     model_name="deepseek-v4-pro",
            #     input_price_per_million=0.435,
            #     output_price_per_million=0.87,
            #     cache_hit_input_price_per_million=0.003625,
            #     currency="USD",
            # ),
            # ("opencode_go", "deepseek-v4-flash"): ModelPricing(
            #     model_name="deepseek-v4-flash",
            #     input_price_per_million=0.14,
            #     output_price_per_million=0.28,
            #     cache_hit_input_price_per_million=0.0028,
            #     currency="USD",
            # ),
        }
    
    def get_model_config(self, model_name: str, provider: Optional[str] = None) -> ModelPricing:
        """优先获取供应商级模型配置，否则回退到通用模型配置。"""
        provider_key = str(provider or "").lower()
        provider_config = self.provider_model_configs.get((provider_key, model_name))
        if provider_config is not None:
            return provider_config
        return self.model_configs.get(model_name, self.model_configs["default"])
    
    def add_model_config(self, config: ModelPricing):
        """添加新的模型配置"""
        self.model_configs[config.model_name] = config
    
    def calculate_cost(self,
                      usage: TokenUsage,
                      model_name: str,
                      provider: str = "unknown",
                      group_discount: float = 1.0) -> Dict[str, float]:
        """
        计算成本（支持多供应商）

        云雾计费公式：按量计费费用 = 令牌分组折扣 × 模型倍率 × （提示token数 + 补全token数 × 补全倍率）/ 500000
        标准计费公式：成本 = (输入tokens / 1M × 输入单价) + (输出tokens / 1M × 输出单价)
        DeepSeek缓存计费：成本 = (缓存命中tokens / 1M × 缓存命中单价) + (缓存未命中tokens / 1M × 输入单价) + (输出tokens / 1M × 输出单价)
        """
        config = self.get_model_config(model_name, provider=provider)

        has_cache_info = (usage.prompt_cache_hit_tokens > 0 or usage.prompt_cache_miss_tokens > 0)

        if has_cache_info and config.cache_hit_input_price_per_million > 0:
            cache_hit_cost = (usage.prompt_cache_hit_tokens * config.cache_hit_input_price_per_million) / 1_000_000
            cache_miss_cost = (usage.prompt_cache_miss_tokens * config.input_price_per_million) / 1_000_000
            input_cost = cache_hit_cost + cache_miss_cost
            output_cost = (usage.completion_tokens * config.output_price_per_million) / 1_000_000
            standard_cost = input_cost + output_cost
        else:
            input_cost = (usage.prompt_tokens * config.input_price_per_million) / 1_000_000
            output_cost = (usage.completion_tokens * config.output_price_per_million) / 1_000_000
            standard_cost = input_cost + output_cost
            cache_hit_cost = 0.0
            cache_miss_cost = 0.0

        yunwu_cost = None
        if provider.lower() == "yunwu":
            yunwu_cost = (group_discount * config.multiplier *
                         (usage.prompt_tokens + usage.completion_tokens * config.completion_multiplier)) / 500000

        result = {
            "standard_cost": standard_cost,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "model_multiplier": config.multiplier,
            "completion_multiplier": config.completion_multiplier,
            "group_discount": group_discount,
            "provider": provider,
            "currency": config.currency,
        }

        if has_cache_info:
            result["cache_hit_cost"] = cache_hit_cost
            result["cache_miss_cost"] = cache_miss_cost
            result["cache_hit_tokens"] = usage.prompt_cache_hit_tokens
            result["cache_miss_tokens"] = usage.prompt_cache_miss_tokens

        if yunwu_cost is not None:
            result["yunwu_formula_cost"] = yunwu_cost
            result["primary_cost"] = yunwu_cost
        else:
            result["primary_cost"] = standard_cost

        return result


class TokenUsageTracker:
    """Token使用量追踪器和调用日志记录器"""
    
    def __init__(self, log_file: Optional[str] = None):
        """
        初始化追踪器
        
        Args:
            log_file: 日志文件路径（.jsonl格式），如果为None则不记录到文件
        """
        self.log_file = log_file
        self.pricing_config = ModelPricingConfig()
        
        # 会话统计
        self.session_usage = TokenUsage()
        self.session_cost = 0.0
        self.session_records = []
        self.call_count = 0
        self.success_count = 0
        self.failure_count = 0
        
        # 确保日志目录存在
        if self.log_file:
            os.makedirs(os.path.dirname(os.path.abspath(self.log_file)), exist_ok=True)
    
    @staticmethod
    def extract_usage_from_response(response: Any) -> TokenUsage:
        """
        从API响应中提取Token使用量
        
        Args:
            response: API响应对象或字典
            
        Returns:
            TokenUsage对象
        """
        usage_dict = {}
        
        # 处理不同的响应格式
        if hasattr(response, 'usage'):
            usage_obj = response.usage
            if hasattr(usage_obj, 'model_dump'):
                usage_dict = usage_obj.model_dump()
            elif hasattr(usage_obj, 'dict'):
                usage_dict = usage_obj.dict()
            elif isinstance(usage_obj, dict):
                usage_dict = usage_obj
        elif isinstance(response, dict) and 'usage' in response:
            usage_dict = response['usage']
        
        # OpenAI 兼容字段
        prompt_tokens = usage_dict.get("prompt_tokens", 0)
        completion_tokens = usage_dict.get("completion_tokens", 0)
        total_tokens = usage_dict.get("total_tokens", 0)

        # DeepSeek 缓存字段
        prompt_cache_hit_tokens = usage_dict.get("prompt_cache_hit_tokens", 0) or 0
        prompt_cache_miss_tokens = usage_dict.get("prompt_cache_miss_tokens", 0) or 0

        # OpenAI 兼容的标准缓存字段：prompt_tokens_details.cached_tokens
        prompt_details = usage_dict.get("prompt_tokens_details") or {}
        if not prompt_cache_hit_tokens:
            if isinstance(prompt_details, dict):
                prompt_cache_hit_tokens = prompt_details.get("cached_tokens", 0) or 0
            else:
                prompt_cache_hit_tokens = getattr(prompt_details, "cached_tokens", 0) or 0
        if not prompt_cache_miss_tokens and prompt_cache_hit_tokens:
            prompt_cache_miss_tokens = max(
                0, int(prompt_tokens or 0) - int(prompt_cache_hit_tokens)
            )

        # Anthropic 常见字段：input_tokens / output_tokens
        if (not prompt_tokens) and ("input_tokens" in usage_dict or "output_tokens" in usage_dict):
            prompt_tokens = usage_dict.get("input_tokens", 0) or 0
            completion_tokens = usage_dict.get("output_tokens", 0) or 0
            total_tokens = int(prompt_tokens) + int(completion_tokens)
            # Anthropic 缓存字段
            if not prompt_cache_hit_tokens:
                prompt_cache_hit_tokens = usage_dict.get("cache_read_input_tokens", 0) or 0
            if not prompt_cache_miss_tokens and prompt_cache_hit_tokens:
                prompt_cache_miss_tokens = max(0, int(prompt_tokens) - int(prompt_cache_hit_tokens))

        return TokenUsage(
            prompt_tokens=int(prompt_tokens or 0),
            completion_tokens=int(completion_tokens or 0),
            total_tokens=int(total_tokens or 0),
            prompt_cache_hit_tokens=int(prompt_cache_hit_tokens),
            prompt_cache_miss_tokens=int(prompt_cache_miss_tokens),
        )
    
    def log_call_record(self, record: Dict[str, Any]):
        """
        记录一次完整的LLM调用
        
        Args:
            record: 包含调用完整信息的字典，应包含：
                - call_id: 调用ID
                - timestamp_start: 开始时间
                - timestamp_end: 结束时间
                - duration_ms: 耗时（毫秒）
                - status: 状态（success/failure）
                - api_name: API名称
                - provider: 供应商
                - model: 模型名称
                - temperature: 温度参数
                - prompt: 提示词
                - response: 响应内容
                - usage: Token使用量
                - cost: 成本信息
                - error: 错误信息（如果有）
        """
        self.call_count += 1
        
        # 更新成功/失败计数
        if record.get('status') == 'success':
            self.success_count += 1
        else:
            self.failure_count += 1
        
        # 更新会话统计
        if 'usage' in record and record['usage']:
            usage = TokenUsage(**record['usage']) if isinstance(record['usage'], dict) else record['usage']
            self.session_usage += usage
        
        if 'cost' in record and record['cost'] and 'primary_cost' in record['cost']:
            self.session_cost += record['cost']['primary_cost']
        
        # 保存到会话记录
        self.session_records.append(record)
        
        # 写入日志文件
        if self.log_file:
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            except Exception as e:
                print(f"警告: 无法写入日志文件 {self.log_file}: {e}")
    
    def get_session_summary(self) -> Dict[str, Any]:
        """
        获取会话统计摘要
        
        Returns:
            包含会话统计信息的字典
        """
        return {
            "total_calls": self.call_count,
            "successful_calls": self.success_count,
            "failed_calls": self.failure_count,
            "success_rate": self.success_count / self.call_count if self.call_count > 0 else 0,
            "total_usage": asdict(self.session_usage),
            "total_cost": self.session_cost,
            "average_cost_per_call": self.session_cost / self.call_count if self.call_count > 0 else 0,
            "currency": "USD"
        }
    
    def reset_session(self):
        """重置会话统计"""
        self.session_usage = TokenUsage()
        self.session_cost = 0.0
        self.session_records = []
        self.call_count = 0
        self.success_count = 0
        self.failure_count = 0
    
    def export_session_to_json(self, output_file: str):
        """
        将会话记录导出到JSON文件
        
        Args:
            output_file: 输出文件路径
        """
        export_data = {
            "summary": self.get_session_summary(),
            "records": self.session_records
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 会话记录已导出到: {output_file}")


# ---------------------------------------------------------------------------
# 全局 Task-Level Token 文件路由（基于 contextvars）
# ---------------------------------------------------------------------------

_global_token_file: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "_global_token_file", default=None
)
_global_task_context: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar(
    "_global_task_context", default=None
)


def set_global_task_token_file(file_path: str) -> None:
    """设置当前异步上下文的全局 token 日志文件路径"""
    _global_token_file.set(file_path)


def get_global_task_token_file() -> Optional[str]:
    """获取当前异步上下文的全局 token 日志文件路径"""
    return _global_token_file.get()


def clear_global_task_token_file() -> None:
    """清除当前异步上下文的全局 token 日志文件路径"""
    _global_token_file.set(None)


def set_global_task_context(**kwargs: Any) -> None:
    """设置当前异步上下文的全局任务上下文（如 task_id, user_id）"""
    _global_task_context.set(dict(kwargs))


def get_global_task_context() -> Optional[Dict[str, Any]]:
    """获取当前异步上下文的全局任务上下文"""
    return _global_task_context.get()


def clear_global_task_context() -> None:
    """清除当前异步上下文的全局任务上下文"""
    _global_task_context.set(None)


if __name__ == "__main__":
    # 测试代码
    print("=== Token Usage Tracker 测试 ===\n")
    
    # 创建追踪器
    tracker = TokenUsageTracker(log_file="test_logs/llm_activity.jsonl")
    
    # 模拟一次调用记录
    test_record = {
        "call_id": "call_test123",
        "timestamp_start": "2025-10-10T10:30:01.123Z",
        "timestamp_end": "2025-10-10T10:30:03.456Z",
        "duration_ms": 2333,
        "status": "success",
        "api_name": "yunwu_gemini",
        "provider": "yunwu",
        "model": "gemini-2.5-pro",
        "temperature": 0.7,
        "prompt": "测试提示词",
        "response": "测试响应",
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150
        },
        "cost": tracker.pricing_config.calculate_cost(
            TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            "gemini-2.5-pro",
            "yunwu"
        ),
        "error": None
    }
    
    # 记录调用
    tracker.log_call_record(test_record)
    
    # 打印摘要
    print("会话摘要:")
    print(json.dumps(tracker.get_session_summary(), ensure_ascii=False, indent=2))
