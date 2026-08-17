#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLM客户端基类

提供异步LLM调用、自动重试、元数据日志记录和Token使用量追踪功能
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, Union, List, AsyncGenerator
from dataclasses import dataclass

# 尝试导入OpenAI库
try:
    from openai import AsyncOpenAI, APIConnectionError, APIError
    openai_available = True
except ImportError:
    print("警告: 未安装openai库，请使用 'pip install openai' 安装")
    openai_available = False

from .llm_runtime_config import LLMRuntimeConfig, overlay_runtime_config, resolve_runtime_config, get_default_api
from .llm_error_classifier import LLMErrorClassification, classify_llm_error
from .token_usage_tracker import TokenUsageTracker, TokenUsage, get_global_task_token_file


@dataclass
class LLMResponse:
    """LLM响应数据类"""
    success: bool
    content: str
    usage: Optional[TokenUsage] = None
    cost: Optional[Dict[str, float]] = None
    error: Optional[str] = None
    call_id: Optional[str] = None
    duration_ms: Optional[int] = None
    error_classification: Optional[Dict[str, Any]] = None
    fallback_used: bool = False
    fallback_from: Optional[str] = None
    fallback_reason: Optional[str] = None


class LLMClient:
    """
    LLM客户端基类
    
    特性：
    - 异步调用大模型API
    - 自动重试机制（指数退避）
    - 完整的元数据日志记录（每次调用都有唯一ID）
    - Token使用量和成本追踪
    - 空响应检测
    - 支持多供应商配置
    """

    # OpenCode Go 的 API-key/auth 错误和明确额度耗尽切换到官方 DeepSeek；
    # 普通限流、连接和服务端错误不会静默改走另一家供应商。
    _API_KEY_FALLBACK_PROFILES = {
        "opencode_go_deepseek_pro": "deepseek_official_chat",
        "opencode_go_deepseek_flash": "deepseek_official_flash",
    }
    
    def __init__(
        self, 
        api_name: Optional[str] = None,
        log_file: Optional[str] = None,
        enable_tracking: bool = True,
        runtime_config: Optional[LLMRuntimeConfig] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_retries: Optional[int] = None,
        retry_base_delay: Optional[float] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        api_key_env: Optional[str] = None,
        config_file: Optional[str] = None,
        env_file: Optional[str] = None,
        protocol: Optional[str] = None,
        provider: Optional[str] = None,
        family: Optional[str] = None,
        max_concurrent: Optional[int] = None,
    ):
        """
        初始化LLM客户端
        
        Args:
            api_name: API名称，如果为None则使用默认API（在llm_config.py中配置）
            log_file: 调用日志文件路径（.jsonl格式），如果为None则使用默认路径
            enable_tracking: 是否启用Token使用量追踪和日志记录
        """
        if not openai_available:
            raise ImportError("未安装openai库，无法初始化客户端")

        if runtime_config is not None and (api_name is None or api_name == runtime_config.api_name):
            runtime_config = overlay_runtime_config(
                runtime_config,
                default_model=model,
                temperature=temperature,
                max_retries=max_retries,
                retry_base_delay=retry_base_delay,
                api_url=base_url,
                api_key=api_key,
                log_file=log_file,
                protocol=protocol,
                provider=provider,
                family=family,
                max_concurrent=max_concurrent,
            )
        else:
            runtime_config = resolve_runtime_config(
                api_name=api_name or get_default_api(),
                config_file=config_file,
                env_file=env_file,
                model=model,
                temperature=temperature,
                max_retries=max_retries,
                retry_base_delay=retry_base_delay,
                base_url=base_url,
                api_key=api_key,
                api_key_env=api_key_env,
                log_file=log_file,
                max_concurrent=max_concurrent,
                protocol=protocol,
                provider=provider,
                family=family,
            )

        if runtime_config.protocol not in {"openai_chat", "openai_responses"}:
            raise ValueError(
                f"LLMClient 只支持 openai_chat/openai_responses，"
                f"profile '{runtime_config.api_name}' 使用的是 {runtime_config.protocol}"
            )

        self.runtime_config = runtime_config
        self.api_name = runtime_config.api_name
        self.config = runtime_config.to_legacy_dict()
        
        # 提取配置信息
        self.provider = runtime_config.provider
        self.family = runtime_config.family
        self.protocol = runtime_config.protocol
        self.default_model = runtime_config.default_model
        self.default_temperature = runtime_config.temperature
        self.max_retries = runtime_config.max_retries
        self.retry_base_delay = runtime_config.retry_base_delay
        self.max_concurrent = runtime_config.max_concurrent
        
        # 获取API密钥
        resolved_api_key = runtime_config.api_key
        if not resolved_api_key:
            raise ValueError(f"缺少API密钥，请为 profile {self.api_name} 配置 api_key")
        
        # 初始化异步客户端
        self.client = AsyncOpenAI(
            api_key=resolved_api_key,
            base_url=runtime_config.api_url
        )
        
        # 初始化追踪器
        self.enable_tracking = enable_tracking
        if self.enable_tracking:
            global_token_file = get_global_task_token_file()
            effective_log_file = (
                runtime_config.log_file
                or global_token_file
                or f"logs/llm_activity_{self.api_name}.jsonl"
            )
            self.tracker = TokenUsageTracker(log_file=effective_log_file)
        else:
            self.tracker = None
        
        print(f"✅ LLMClient 初始化成功:")
        print(f"   - Profile: {self.api_name}")
        print(f"   - Provider: {self.provider}")
        print(f"   - Protocol: {self.protocol}")
        print(f"   - Default Model: {self.default_model}")
        print(f"   - Tracking: {'Enabled' if enable_tracking else 'Disabled'}")
    
    @classmethod
    def from_profile(cls, profile_name: str, **kwargs) -> "LLMClient":
        """Create a client from a runtime profile name."""
        return cls(api_name=profile_name, **kwargs)

    @classmethod
    def from_stage(cls, stage_name: str, fallback_api: Optional[str] = None, **kwargs) -> "LLMClient":
        """根据 stage routing 配置创建客户端"""
        from .stage_routing import get_route_for_stage
        route = get_route_for_stage(stage_name, default=fallback_api)
        api_name = route.get("api_name") or fallback_api
        model = route.get("model")
        temperature = route.get("temperature")
        return cls(api_name=api_name, model=model, temperature=temperature, **kwargs)
    
    def _generate_call_id(self) -> str:
        """生成唯一的调用ID（16位哈希）

        格式保持为 `call_<16 hex>`，以兼容现有日志前缀 `call_`。
        如果需要兼容历史的8位ID，旧的日志仍然保留在 logs/ 中，不受本方法影响。
        """
        return f"call_{uuid.uuid4().hex[:16]}"
    
    def _get_current_timestamp(self) -> str:
        """获取当前时间戳（ISO 8601格式）"""
        return datetime.utcnow().isoformat() + 'Z'

    def _fallback_profile_for_error(self, classification: LLMErrorClassification) -> Optional[str]:
        """返回当前 profile 对应的官方 fallback；非认证错误返回 None。"""
        if not classification.fallback_eligible:
            return None
        return self._API_KEY_FALLBACK_PROFILES.get(self.api_name)

    def _fallback_log_file(self) -> Optional[str]:
        if self.tracker:
            return self.tracker.log_file
        return self.runtime_config.log_file

    def _build_fallback_client(self, profile_name: str) -> "LLMClient":
        """创建官方 profile 客户端，不继承 OpenCode 的 endpoint/key。"""
        return LLMClient.from_profile(
            profile_name,
            log_file=self._fallback_log_file(),
            enable_tracking=self.enable_tracking,
            max_retries=self.max_retries,
            retry_base_delay=self.retry_base_delay,
        )

    def _record_fallback_event(
        self,
        call_record: Dict[str, Any],
        classification: LLMErrorClassification,
        fallback_profile: str,
        retries: int,
        start_time: float,
    ) -> None:
        """记录原供应商失败及 fallback 决策，避免 fallback 变成不可见的旁路。"""
        duration_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)
        call_record.update({
            "timestamp_end": self._get_current_timestamp(),
            "duration_ms": duration_ms,
            "status": "fallback",
            "response": None,
            "response_length": 0,
            "usage": None,
            "cost": None,
            "error": f"{classification.kind}: {classification.message}",
            "error_classification": classification.to_dict(),
            "retry_count": retries,
            "fallback_used": True,
            "fallback_to": fallback_profile,
            "fallback_reason": classification.kind,
        })
        if self.tracker:
            self.tracker.log_call_record(call_record)

    async def _run_completion_fallback(
        self,
        classification: LLMErrorClassification,
        *,
        prompt: str,
        model: Optional[str],
        temperature: Optional[float],
        stage: Optional[str],
        metadata: Optional[Dict[str, Any]],
        max_tokens: Optional[int],
        system_prompt: Optional[str] = None,
    ) -> Optional[LLMResponse]:
        fallback_profile = self._fallback_profile_for_error(classification)
        if not fallback_profile:
            return None
        print(
            f"[{self.api_name}] 检测到 {classification.kind}，"
            f"切换官方 DeepSeek profile: {fallback_profile}"
        )
        try:
            fallback_client = self._build_fallback_client(fallback_profile)
            response = await fallback_client.get_completion(
                prompt=prompt,
                model=model,
                temperature=temperature,
                stage=stage,
                metadata=metadata,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
            )
        except Exception as fallback_error:
            fallback_classification = classify_llm_error(fallback_error)
            return LLMResponse(
                success=False,
                content="",
                error=(
                    f"原供应商 {classification.kind}，fallback {fallback_profile} 初始化/调用失败: "
                    f"{fallback_classification.kind}: {fallback_classification.message}"
                ),
                error_classification=classification.to_dict(),
                fallback_used=True,
                fallback_from=self.api_name,
                fallback_reason=classification.kind,
            )

        response.fallback_used = True
        response.fallback_from = self.api_name
        response.fallback_reason = classification.kind
        response.error_classification = classification.to_dict()
        return response

    async def _run_vision_fallback(
        self,
        classification: LLMErrorClassification,
        *,
        prompt: str,
        image_base64: str,
        model: Optional[str],
        temperature: Optional[float],
        stage: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> Optional[LLMResponse]:
        fallback_profile = self._fallback_profile_for_error(classification)
        if not fallback_profile:
            return None
        print(
            f"[{self.api_name}] 检测到 {classification.kind}，"
            f"切换官方 DeepSeek profile: {fallback_profile}"
        )
        try:
            fallback_client = self._build_fallback_client(fallback_profile)
            response = await fallback_client.get_vision_completion(
                prompt=prompt,
                image_base64=image_base64,
                model=model,
                temperature=temperature,
                stage=stage,
                metadata=metadata,
            )
        except Exception as fallback_error:
            fallback_classification = classify_llm_error(fallback_error)
            return LLMResponse(
                success=False,
                content="",
                error=(
                    f"原供应商 {classification.kind}，fallback {fallback_profile} 初始化/调用失败: "
                    f"{fallback_classification.kind}: {fallback_classification.message}"
                ),
                error_classification=classification.to_dict(),
                fallback_used=True,
                fallback_from=self.api_name,
                fallback_reason=classification.kind,
            )
        response.fallback_used = True
        response.fallback_from = self.api_name
        response.fallback_reason = classification.kind
        response.error_classification = classification.to_dict()
        return response

    def _annotate_stream_fallback_record(
        self,
        call_record: Dict[str, Any],
        classification: LLMErrorClassification,
        fallback_profile: str,
        start_time: float,
    ) -> None:
        self._record_fallback_event(
            call_record,
            classification,
            fallback_profile,
            retries=0,
            start_time=start_time,
        )
    
    async def get_vision_completion(
        self,
        prompt: str,
        image_base64: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        stage: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> LLMResponse:
        """
        获取包含图片的LLM完成结果（vision API）
        
        Args:
            prompt: 提示词
            image_base64: Base64编码的图片数据
            model: 模型名称，如果为None则使用默认模型
            temperature: 温度参数，如果为None则使用默认温度
            stage: 阶段名称，用于路由覆盖（可选）
            metadata: 额外的元数据，会被记录到日志中（可选）
            
        Returns:
            LLMResponse对象，包含成功状态、内容、用量、成本等信息
        """
        # 生成调用ID和开始时间
        call_id = self._generate_call_id()
        timestamp_start = self._get_current_timestamp()
        start_time = asyncio.get_event_loop().time()
        
        # 确定最终使用的参数
        model_name = model or self.default_model
        temperature_effective = (
            temperature
            if temperature is not None
            else self.default_temperature
        )
        
        # 构建vision消息
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                }
            ]
        }]
        
        # 构建初始元数据记录
        call_record = {
            "call_id": call_id,
            "timestamp_start": timestamp_start,
            "api_name": self.api_name,
            "provider": self.provider,
            "model": model_name,
            "temperature": temperature_effective,
            "prompt": prompt,
            "prompt_length": len(prompt),
            "has_image": True,
            "image_size_bytes": len(image_base64),
            "stage": stage,
        }
        
        # 添加用户提供的额外元数据
        if metadata:
            call_record["metadata"] = metadata
        
        # 执行调用（带重试机制）
        retries = 0
        last_error = None
        last_classification: Optional[LLMErrorClassification] = None
        
        while retries <= self.max_retries:
            try:
                print(f"[{call_id}] 调用Vision API: {self.api_name}, 模型: {model_name}, 重试: {retries}/{self.max_retries}")
                
                response = await self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature_effective,
                )
                
                # 提取响应内容
                if hasattr(response, 'choices') and response.choices:
                    content = response.choices[0].message.content or ""
                else:
                    raise ValueError("API返回了意外的响应格式")
                
                # 检测空响应
                if not content.strip():
                    raise ValueError("Empty response from LLM")
                
                # 提取Token使用量
                usage = (
                    TokenUsageTracker.extract_usage_from_response(response)
                    if hasattr(response, "usage") and response.usage
                    else None
                )
                
                # 计算成本
                cost = None
                if usage and self.tracker:
                    cost = self.tracker.pricing_config.calculate_cost(
                        usage=usage,
                        model_name=model_name,
                        provider=self.provider
                    )
                
                # 计算耗时
                end_time = asyncio.get_event_loop().time()
                duration_ms = int((end_time - start_time) * 1000)
                timestamp_end = self._get_current_timestamp()
                
                # 补完调用记录
                call_record.update({
                    "timestamp_end": timestamp_end,
                    "duration_ms": duration_ms,
                    "status": "success",
                    "response": content,
                    "response_length": len(content),
                    "usage": usage.__dict__ if usage else None,
                    "cost": cost,
                    "error": None,
                    "retry_count": retries
                })
                
                # 记录到追踪器
                if self.tracker:
                    self.tracker.log_call_record(call_record)
                
                print(f"[{call_id}] ✅ Vision调用成功 ({duration_ms}ms)")
                if usage:
                    cache_info = ""
                    if usage.prompt_cache_hit_tokens > 0:
                        cache_info = f" (cache_hit={usage.prompt_cache_hit_tokens}, miss={usage.prompt_cache_miss_tokens})"
                    print(f"[{call_id}] Token: {usage.prompt_tokens}+{usage.completion_tokens}={usage.total_tokens}{cache_info}")
                if cost:
                    currency = cost.get('currency', 'USD')
                    symbol = '¥' if currency == 'CNY' else '$'
                    print(f"[{call_id}] Cost: {symbol}{cost.get('primary_cost', 0):.6f} {currency}")
                
                # 返回成功响应
                return LLMResponse(
                    success=True,
                    content=content,
                    usage=usage,
                    cost=cost,
                    error=None,
                    call_id=call_id,
                    duration_ms=duration_ms
                )
                
            except ValueError as e:
                # 空响应或格式错误
                if "Empty response from LLM" in str(e):
                    last_error = f"空响应: {str(e)}"
                    print(f"[{call_id}] ⚠️ 收到空响应，重试中...")
                else:
                    last_error = f"值错误: {str(e)}"
                    print(f"[{call_id}] ❌ 值错误: {e}")
                    break  # 格式错误不重试
                
            except Exception as e:
                classification = classify_llm_error(e)
                last_classification = classification
                last_error = f"{classification.kind}: {classification.message}"

                fallback_profile = self._fallback_profile_for_error(classification)
                if fallback_profile:
                    fallback_response = await self._run_vision_fallback(
                        classification,
                        prompt=prompt,
                        image_base64=image_base64,
                        model=model,
                        temperature=temperature,
                        stage=stage,
                        metadata=metadata,
                    )
                    self._record_fallback_event(
                        call_record, classification, fallback_profile, retries, start_time
                    )
                    return fallback_response

                level = "⚠️" if classification.retryable else "❌"
                print(f"[{call_id}] {level} {last_error}")
                if not classification.retryable:
                    break
            
            # 重试逻辑
            retries += 1
            if retries <= self.max_retries:
                delay = self.retry_base_delay * (2 ** (retries - 1))
                print(f"[{call_id}] 等待 {delay} 秒后重试...")
                await asyncio.sleep(delay)
        
        # 所有重试都失败，记录失败
        end_time = asyncio.get_event_loop().time()
        duration_ms = int((end_time - start_time) * 1000)
        timestamp_end = self._get_current_timestamp()
        
        call_record.update({
            "timestamp_end": timestamp_end,
            "duration_ms": duration_ms,
            "status": "failure",
            "response": None,
            "response_length": 0,
            "usage": None,
            "cost": None,
            "error": last_error,
            "error_classification": last_classification.to_dict() if last_classification else None,
            "retry_count": retries
        })
        
        if self.tracker:
            self.tracker.log_call_record(call_record)
        
        print(f"[{call_id}] ❌ Vision调用失败: {last_error}")
        
        return LLMResponse(
            success=False,
            content="",
            usage=None,
            cost=None,
            error=last_error,
            call_id=call_id,
            duration_ms=duration_ms,
            error_classification=last_classification.to_dict() if last_classification else None,
        )
    
    async def get_completion(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        stage: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """
        获取LLM完成结果（核心方法）
        
        Args:
            prompt: 提示词
            model: 模型名称，如果为None则使用默认模型
            temperature: 温度参数，如果为None则使用默认温度
            stage: 阶段名称，用于路由覆盖（可选）
            metadata: 额外的元数据，会被记录到日志中（可选）
            
        Returns:
            LLMResponse对象，包含成功状态、内容、用量、成本等信息
        """
        # 生成调用ID和开始时间
        call_id = self._generate_call_id()
        timestamp_start = self._get_current_timestamp()
        start_time = asyncio.get_event_loop().time()
        
        # 确定最终使用的参数
        model_name = model or self.default_model
        temperature_effective = (
            temperature
            if temperature is not None
            else self.default_temperature
        )
        
        # 构建初始元数据记录
        call_record = {
            "call_id": call_id,
            "timestamp_start": timestamp_start,
            "api_name": self.api_name,
            "provider": self.provider,
            "family": self.family,
            "protocol": self.protocol,
            "model": model_name,
            "temperature": temperature_effective,
            "prompt": prompt,
            "prompt_length": len(prompt),
            "stage": stage,
        }
        
        # 添加用户提供的额外元数据
        if metadata:
            call_record["metadata"] = metadata
        
        # 执行调用（带重试机制）
        retries = 0
        last_error = None
        last_classification: Optional[LLMErrorClassification] = None
        
        while retries <= self.max_retries:
            try:
                print(f"[{call_id}] 调用API: {self.api_name}, 模型: {model_name}, 重试: {retries}/{self.max_retries}")

                if self.protocol == "openai_responses":
                    create_kwargs: Dict[str, Any] = {
                        "model": model_name,
                        "input": prompt,
                        "temperature": temperature_effective,
                    }
                    if max_tokens is not None:
                        create_kwargs["max_output_tokens"] = max_tokens
                    response = await self.client.responses.create(**create_kwargs)
                    content = getattr(response, "output_text", "") or ""
                else:
                    messages = []
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})
                    messages.append({"role": "user", "content": prompt})
                    create_kwargs = {
                        "model": model_name,
                        "messages": messages,
                        "temperature": temperature_effective,
                    }
                    if max_tokens is not None:
                        create_kwargs["max_tokens"] = max_tokens
                    response = await self.client.chat.completions.create(**create_kwargs)

                    # 提取响应内容
                    if hasattr(response, 'choices') and response.choices:
                        content = response.choices[0].message.content or ""
                    else:
                        raise ValueError("API返回了意外的响应格式")
                
                # 检测空响应
                if not content.strip():
                    raise ValueError("Empty response from LLM")
                
                # 提取Token使用量
                usage = (
                    TokenUsageTracker.extract_usage_from_response(response)
                    if hasattr(response, "usage") and response.usage
                    else None
                )
                
                # 计算成本
                cost = None
                if usage and self.tracker:
                    cost = self.tracker.pricing_config.calculate_cost(
                        usage=usage,
                        model_name=model_name,
                        provider=self.provider
                    )
                
                # 计算耗时
                end_time = asyncio.get_event_loop().time()
                duration_ms = int((end_time - start_time) * 1000)
                timestamp_end = self._get_current_timestamp()
                
                # 补完调用记录
                call_record.update({
                    "timestamp_end": timestamp_end,
                    "duration_ms": duration_ms,
                    "status": "success",
                    "response": content,
                    "response_length": len(content),
                    "usage": usage.__dict__ if usage else None,
                    "cost": cost,
                    "error": None,
                    "retry_count": retries
                })
                
                # 记录到追踪器
                if self.tracker:
                    self.tracker.log_call_record(call_record)
                
                print(f"[{call_id}] ✅ 调用成功 ({duration_ms}ms)")
                if usage:
                    cache_info = ""
                    if usage.prompt_cache_hit_tokens > 0:
                        cache_info = f" (cache_hit={usage.prompt_cache_hit_tokens}, miss={usage.prompt_cache_miss_tokens})"
                    print(f"[{call_id}] Token: {usage.prompt_tokens}+{usage.completion_tokens}={usage.total_tokens}{cache_info}")
                if cost:
                    currency = cost.get('currency', 'USD')
                    symbol = '¥' if currency == 'CNY' else '$'
                    print(f"[{call_id}] Cost: {symbol}{cost.get('primary_cost', 0):.6f} {currency}")
                
                # 返回成功响应
                return LLMResponse(
                    success=True,
                    content=content,
                    usage=usage,
                    cost=cost,
                    error=None,
                    call_id=call_id,
                    duration_ms=duration_ms
                )
                
            except ValueError as e:
                # 空响应或格式错误
                if "Empty response from LLM" in str(e):
                    last_error = f"空响应: {str(e)}"
                    print(f"[{call_id}] ⚠️ 收到空响应，重试中...")
                else:
                    last_error = f"值错误: {str(e)}"
                    print(f"[{call_id}] ❌ 值错误: {e}")
                    break  # 格式错误不重试
                
            except Exception as e:
                classification = classify_llm_error(e)
                last_classification = classification
                last_error = f"{classification.kind}: {classification.message}"

                fallback_profile = self._fallback_profile_for_error(classification)
                if fallback_profile:
                    fallback_response = await self._run_completion_fallback(
                        classification,
                        prompt=prompt,
                        model=model,
                        temperature=temperature,
                        stage=stage,
                        metadata=metadata,
                        max_tokens=max_tokens,
                        system_prompt=system_prompt,
                    )
                    self._record_fallback_event(
                        call_record, classification, fallback_profile, retries, start_time
                    )
                    return fallback_response

                level = "⚠️" if classification.retryable else "❌"
                print(f"[{call_id}] {level} {last_error}")
                if not classification.retryable:
                    break
            
            # 重试逻辑
            retries += 1
            if retries <= self.max_retries:
                delay = self.retry_base_delay * (2 ** (retries - 1))
                print(f"[{call_id}] 等待 {delay} 秒后重试...")
                await asyncio.sleep(delay)
        
        # 所有重试都失败，记录失败
        end_time = asyncio.get_event_loop().time()
        duration_ms = int((end_time - start_time) * 1000)
        timestamp_end = self._get_current_timestamp()
        
        call_record.update({
            "timestamp_end": timestamp_end,
            "duration_ms": duration_ms,
            "status": "failure",
            "response": None,
            "response_length": 0,
            "usage": None,
            "cost": None,
            "error": last_error,
            "error_classification": last_classification.to_dict() if last_classification else None,
            "retry_count": retries
        })
        
        if self.tracker:
            self.tracker.log_call_record(call_record)
        
        print(f"[{call_id}] ❌ 调用失败: {last_error}")
        
        return LLMResponse(
            success=False,
            content="",
            usage=None,
            cost=None,
            error=last_error,
            call_id=call_id,
            duration_ms=duration_ms,
            error_classification=last_classification.to_dict() if last_classification else None,
        )
    
    async def get_json_completion(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        stage: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
    ) -> Optional[Union[Dict[str, Any], List[Any]]]:
        """
        获取JSON格式的LLM完成结果

        内部调用 get_completion()，然后使用 extract_and_repair_json() 解析响应。
        三级解析策略：直接解析 → json5宽容解析 → json_repair修复。

        Args:
            prompt: 提示词
            model: 模型名称
            temperature: 温度参数
            stage: 阶段名称（用于元数据标注）
            metadata: 额外元数据
            max_tokens: 最大输出 token 数

        Returns:
            解析后的JSON对象（Dict或List），解析失败返回None
        """
        from .prompt_utils import extract_and_repair_json

        response = await self.get_completion(
            prompt=prompt,
            model=model,
            temperature=temperature,
            stage=stage,
            metadata=metadata,
            max_tokens=max_tokens,
        )

        if not response.success:
            print(f"[{response.call_id}] get_json_completion: LLM调用失败 - {response.error}")
            return None

        try:
            return extract_and_repair_json(response.content)
        except (ValueError, json.JSONDecodeError) as e:
            print(f"[{response.call_id}] get_json_completion: JSON解析失败 - {e}")
            return None

    async def get_streaming_completion(
        self,
        prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        stage: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        流式获取LLM完成结果

        逐 chunk yield delta content。流结束后自动记录 token usage 和成本。

        Args:
            prompt: 提示词（与 messages 二选一）
            model: 模型名称
            temperature: 温度参数
            stage: 阶段名称
            metadata: 额外元数据
            max_tokens: 最大输出 token 数
            messages: OpenAI messages 数组（优先于 prompt）

        Yields:
            每个 chunk 的文本片段
        """
        if self.protocol != "openai_chat":
            raise ValueError(f"Streaming 仅支持 openai_chat 协议，当前: {self.protocol}")

        if messages is None and prompt is None:
            raise ValueError("prompt 和 messages 至少提供一个")

        call_id = self._generate_call_id()
        timestamp_start = self._get_current_timestamp()
        start_time = asyncio.get_event_loop().time()

        model_name = model or self.default_model
        temperature_effective = (
            temperature if temperature is not None else self.default_temperature
        )

        prompt_for_log = prompt or (messages[-1]["content"][:200] if messages else "")

        call_record: Dict[str, Any] = {
            "call_id": call_id,
            "timestamp_start": timestamp_start,
            "api_name": self.api_name,
            "provider": self.provider,
            "family": self.family,
            "protocol": self.protocol,
            "model": model_name,
            "temperature": temperature_effective,
            "prompt": prompt_for_log,
            "prompt_length": len(prompt_for_log),
            "stage": stage,
            "streaming": True,
        }
        if metadata:
            call_record["metadata"] = metadata

        final_messages = messages if messages else [{"role": "user", "content": prompt}]

        create_kwargs: Dict[str, Any] = {
            "model": model_name,
            "messages": final_messages,
            "temperature": temperature_effective,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if max_tokens is not None:
            create_kwargs["max_tokens"] = max_tokens

        full_content = []
        usage = None
        error_msg = None
        error_classification: Optional[LLMErrorClassification] = None

        try:
            stream = await self.client.chat.completions.create(**create_kwargs)

            async for chunk in stream:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        full_content.append(delta.content)
                        yield delta.content

                if hasattr(chunk, 'usage') and chunk.usage:
                    usage = TokenUsageTracker.extract_usage_from_response(chunk)

        except Exception as e:
            classification = classify_llm_error(e)
            error_classification = classification
            fallback_profile = self._fallback_profile_for_error(classification)
            if fallback_profile:
                self._annotate_stream_fallback_record(
                    call_record, classification, fallback_profile, start_time
                )
                print(
                    f"[{call_id}] 检测到 {classification.kind}，"
                    f"切换官方 DeepSeek profile: {fallback_profile}"
                )
                try:
                    fallback_client = self._build_fallback_client(fallback_profile)
                    async for delta in fallback_client.get_streaming_completion(
                        prompt=prompt,
                        model=model,
                        temperature=temperature,
                        stage=stage,
                        metadata=metadata,
                        max_tokens=max_tokens,
                        messages=messages,
                    ):
                        yield delta
                except Exception as fallback_error:
                    fallback_classification = classify_llm_error(fallback_error)
                    print(
                        f"[{call_id}] fallback {fallback_profile} 失败: "
                        f"{fallback_classification.kind}: {fallback_classification.message}"
                    )
                return

            error_msg = f"{classification.kind}: {classification.message}"
            print(f"[{call_id}] {error_msg}")

        end_time = asyncio.get_event_loop().time()
        duration_ms = int((end_time - start_time) * 1000)
        content_str = "".join(full_content)

        cost = None
        if usage and self.tracker:
            cost = self.tracker.pricing_config.calculate_cost(
                usage=usage, model_name=model_name, provider=self.provider
            )

        call_record.update({
            "timestamp_end": self._get_current_timestamp(),
            "duration_ms": duration_ms,
            "status": "success" if not error_msg else "failure",
            "response": content_str,
            "response_length": len(content_str),
            "usage": usage.__dict__ if usage else None,
            "cost": cost,
            "error": error_msg,
            "error_classification": error_classification.to_dict() if error_classification else None,
            "retry_count": 0,
        })

        if self.tracker:
            self.tracker.log_call_record(call_record)

        if not error_msg:
            print(f"[{call_id}] Streaming完成 ({duration_ms}ms, {len(content_str)} chars)")

    def get_session_summary(self) -> Dict[str, Any]:
        """
        获取当前会话的统计摘要
        
        Returns:
            包含调用次数、成功率、总成本等信息的字典
        """
        if not self.tracker:
            return {"error": "Tracking is disabled"}
        return self.tracker.get_session_summary()
    
    def reset_session(self):
        """重置会话统计"""
        if self.tracker:
            self.tracker.reset_session()
            print("✅ 会话统计已重置")
    
    def export_session(self, output_file: str):
        """
        导出会话记录到JSON文件
        
        Args:
            output_file: 输出文件路径
        """
        if self.tracker:
            self.tracker.export_session_to_json(output_file)
    
    async def process_batch_with_manager(
        self,
        task_manager: 'TaskManager',
        input_data: Dict[str, str],
        prompt_template: Optional[str] = None,
        max_concurrent: int = 5,
        **llm_kwargs
    ) -> Dict[str, Any]:
        """
        使用 TaskManager 批量处理任务
        
        Args:
            task_manager: TaskManager 实例
            input_data: {unique_id: text} 的映射字典
            prompt_template: Prompt 模板（可选，用 {text} 占位符），如果为None则直接使用text
            max_concurrent: 最大并发数
            **llm_kwargs: 传递给 get_completion 的其他参数（如 temperature, model 等）
            
        Returns:
            处理摘要统计字典
            
        Example:
            task_mgr = TaskManager("tasks/batch.jsonl")
            input_data = {"doc_001": "文本1", "doc_002": "文本2"}
            
            summary = await client.process_batch_with_manager(
                task_manager=task_mgr,
                input_data=input_data,
                prompt_template="请处理：{text}",
                max_concurrent=10,
                temperature=0.7
            )
        """
        # 获取待处理任务
        pending_ids = task_manager.get_pending_tasks()
        
        if not pending_ids:
            print("✅ 没有待处理的任务")
            return task_manager.get_statistics()
        
        print(f"📋 开始处理 {len(pending_ids)} 个待处理任务...")
        print(f"⚙️  并发数: {max_concurrent}")
        
        # 创建信号量控制并发
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # 处理单个任务的函数
        async def process_one(unique_id: str, index: int):
            async with semaphore:
                # 获取文本
                text = input_data.get(unique_id)
                if text is None:
                    error_msg = f"在 input_data 中找不到 unique_id: {unique_id}"
                    print(f"[{index}/{len(pending_ids)}] ❌ {unique_id}: {error_msg}")
                    task_manager.update_task_failure(unique_id, "N/A", error_msg)
                    return
                
                # 构建 prompt
                if prompt_template:
                    prompt = prompt_template.replace("{text}", text)
                else:
                    prompt = text
                
                # 调用 LLM
                print(f"[{index}/{len(pending_ids)}] 🔄 处理中: {unique_id}")
                
                response = await self.get_completion(
                    prompt=prompt,
                    metadata={"unique_id": unique_id},
                    **llm_kwargs
                )
                
                # 更新任务状态
                if response.success:
                    task_manager.update_task_success(unique_id, response.call_id)
                    print(f"[{index}/{len(pending_ids)}] ✅ 成功: {unique_id} (call_id: {response.call_id})")
                else:
                    task_manager.update_task_failure(unique_id, response.call_id or "N/A", response.error or "Unknown error")
                    print(f"[{index}/{len(pending_ids)}] ❌ 失败: {unique_id} - {response.error}")
        
        # 并发处理所有任务
        tasks = [
            process_one(uid, i+1) 
            for i, uid in enumerate(pending_ids)
        ]
        
        await asyncio.gather(*tasks)
        
        # 返回统计信息
        stats = task_manager.get_statistics()
        
        print("\n" + "="*60)
        print("批量处理完成")
        print("="*60)
        print(f"总任务数:   {stats['total']}")
        print(f"成功:       {stats['success']}")
        print(f"失败:       {stats['failed']}")
        print(f"待处理:     {stats['pending']}")
        print(f"成功率:     {stats['success_rate']*100:.1f}%")
        print("="*60 + "\n")
        
        return stats


# 测试代码
if __name__ == "__main__":
    import asyncio
    
    async def test_client():
        print("=== LLMClient 测试 ===\n")
        
        # 创建客户端
        client = LLMClient(
            api_name="yunwu_gemini",
            log_file="test_logs/test_llm_activity.jsonl"
        )
        
        # 测试调用
        response = await client.get_completion(
            prompt="你好，请用一句话介绍自己。",
            temperature=0.7,
            metadata={"test": True, "purpose": "greeting"}
        )
        
        print(f"\n响应:")
        print(f"- 成功: {response.success}")
        print(f"- 内容: {response.content[:100]}...")
        print(f"- Call ID: {response.call_id}")
        
        if response.usage:
            print(f"- Token使用: {response.usage.total_tokens}")
        
        if response.cost:
            print(f"- 成本: ${response.cost.get('primary_cost', 0):.6f}")
        
        # 打印会话摘要
        print("\n会话摘要:")
        import json
        print(json.dumps(client.get_session_summary(), ensure_ascii=False, indent=2))
    
    # 运行测试
    asyncio.run(test_client())
