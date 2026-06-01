#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Anthropic SDK 适配的 LLM 客户端（方案A）

- 使用 anthropic / AsyncAnthropic 的 messages.create 接口
- 复用现有 TokenUsageTracker 做 token 统计与成本计算、日志落盘
- 与 LLMClient.get_completion 尽量保持一致的调用与日志结构
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from .llm_runtime_config import LLMRuntimeConfig, overlay_runtime_config, resolve_runtime_config, get_default_api
from .token_usage_tracker import TokenUsageTracker, TokenUsage

try:
    from anthropic import AsyncAnthropic  # type: ignore

    anthropic_available = True
except ImportError:
    anthropic_available = False


@dataclass
class AnthropicLLMResponse:
    success: bool
    content: str
    usage: Optional[TokenUsage] = None
    cost: Optional[Dict[str, float]] = None
    error: Optional[str] = None
    call_id: Optional[str] = None
    duration_ms: Optional[int] = None
    raw_model: Optional[str] = None


class AnthropicLLMClient:
    """
    Anthropic SDK 版本的 LLM Client。

    适用场景：
    - 你的上游网关/代理返回 Anthropic 风格响应（messages API）
    - 需要继续沿用本项目的 TokenUsageTracker 与 JSONL 调用日志
    """

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
        if not anthropic_available:
            raise ImportError("未安装 anthropic 库，无法初始化 AnthropicLLMClient（请 pip install anthropic）")

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

        if runtime_config.protocol != "anthropic_messages":
            raise ValueError(
                f"AnthropicLLMClient 只支持 anthropic_messages，"
                f"profile '{runtime_config.api_name}' 使用的是 {runtime_config.protocol}"
            )

        self.runtime_config = runtime_config
        self.api_name = runtime_config.api_name
        self.config = runtime_config.to_legacy_dict()

        self.provider = runtime_config.provider
        self.family = runtime_config.family
        self.protocol = runtime_config.protocol
        self.default_model = runtime_config.default_model
        self.default_temperature = runtime_config.temperature
        self.max_retries = runtime_config.max_retries
        self.retry_base_delay = runtime_config.retry_base_delay
        self.max_concurrent = runtime_config.max_concurrent

        resolved_api_key = runtime_config.api_key
        if not resolved_api_key:
            raise ValueError(f"缺少API密钥，请为 profile {self.api_name} 配置 api_key")

        base_url = runtime_config.api_url
        if not base_url:
            raise ValueError(f"缺少 api_url，请为 profile {self.api_name} 配置 api_url")

        # Anthropic SDK 会自行拼接 `/v1/messages` 等路径；
        # 如果用户把 api_url 配成 `.../v1`，会导致请求变成 `/v1/v1/messages` -> 404。
        # 这里做一次温和的兼容处理。
        base_url = str(base_url).rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[: -len("/v1")]

        self.client = AsyncAnthropic(
            api_key=resolved_api_key,
            base_url=base_url,
        )

        self.enable_tracking = enable_tracking
        if self.enable_tracking:
            default_log_file = runtime_config.log_file or f"logs/llm_activity_{self.api_name}.jsonl"
            self.tracker = TokenUsageTracker(log_file=default_log_file)
        else:
            self.tracker = None

        print("✅ AnthropicLLMClient 初始化成功:")
        print(f"   - Profile: {self.api_name}")
        print(f"   - Provider: {self.provider}")
        print(f"   - Protocol: {self.protocol}")
        print(f"   - Default Model: {self.default_model}")
        print(f"   - Tracking: {'Enabled' if enable_tracking else 'Disabled'}")

    @classmethod
    def from_profile(cls, profile_name: str, **kwargs: Any) -> "AnthropicLLMClient":
        return cls(api_name=profile_name, **kwargs)

    def _generate_call_id(self) -> str:
        return f"call_{uuid.uuid4().hex[:16]}"

    def _get_current_timestamp(self) -> str:
        return datetime.utcnow().isoformat() + "Z"

    @staticmethod
    def _extract_text_from_anthropic_response(resp: Any) -> str:
        """
        Anthropic messages.create 的 content 通常是 block 列表：
        - block.type == "text" 时，block.text 为文本
        兼容其它形态时尽量兜底为字符串拼接。
        """
        content = getattr(resp, "content", None)
        if content is None:
            return ""

        if isinstance(content, str):
            return content

        parts = []
        if isinstance(content, list):
            for block in content:
                block_type = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
                if block_type == "text":
                    text = getattr(block, "text", None) or (block.get("text") if isinstance(block, dict) else None)
                    if text:
                        parts.append(str(text))
                else:
                    # 兜底：尽量拼出可读文本（例如某些代理返回结构不同）
                    text = getattr(block, "text", None) or (block.get("text") if isinstance(block, dict) else None)
                    if text:
                        parts.append(str(text))
        return "\n".join(parts).strip()

    @staticmethod
    def _extract_usage_from_anthropic_response(resp: Any) -> TokenUsage:
        usage_obj = getattr(resp, "usage", None)
        prompt_tokens = int(getattr(usage_obj, "input_tokens", 0) or 0)
        completion_tokens = int(getattr(usage_obj, "output_tokens", 0) or 0)
        total_tokens = prompt_tokens + completion_tokens
        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    async def get_completion(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        stage: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        max_tokens: int = 1024,
    ) -> AnthropicLLMResponse:
        call_id = self._generate_call_id()
        timestamp_start = self._get_current_timestamp()
        start_time = asyncio.get_event_loop().time()

        model_name = model or self.default_model
        temperature_effective = (
            temperature
            if temperature is not None
            else self.default_temperature
        )

        call_record: Dict[str, Any] = {
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
            "sdk": "anthropic",
        }
        if metadata:
            call_record["metadata"] = metadata

        retries = 0
        last_error: Optional[str] = None

        while retries <= self.max_retries:
            try:
                print(f"[{call_id}] 调用API(Anthropic): {self.api_name}, 模型: {model_name}, 重试: {retries}/{self.max_retries}")

                resp = await self.client.messages.create(
                    model=model_name,
                    max_tokens=max_tokens,
                    temperature=temperature_effective,
                    messages=[{"role": "user", "content": prompt}],
                )

                resp_model = getattr(resp, "model", None)
                content = self._extract_text_from_anthropic_response(resp)
                if not content.strip():
                    raise ValueError("Empty response from Anthropic LLM")

                usage = self._extract_usage_from_anthropic_response(resp)

                cost = None
                if self.tracker:
                    cost = self.tracker.pricing_config.calculate_cost(
                        usage=usage,
                        model_name=model_name,
                        provider=self.provider,
                    )

                end_time = asyncio.get_event_loop().time()
                duration_ms = int((end_time - start_time) * 1000)
                timestamp_end = self._get_current_timestamp()

                call_record.update(
                    {
                        "timestamp_end": timestamp_end,
                        "duration_ms": duration_ms,
                        "status": "success",
                        "response": content,
                        "response_length": len(content),
                        "response_model": resp_model,
                        "usage": usage.__dict__,
                        "cost": cost,
                        "error": None,
                        "retry_count": retries,
                    }
                )

                if self.tracker:
                    self.tracker.log_call_record(call_record)

                print(f"[{call_id}] ✅ 调用成功 ({duration_ms}ms)")
                print(f"[{call_id}] Token: {usage.prompt_tokens}+{usage.completion_tokens}={usage.total_tokens}")
                if cost:
                    print(f"[{call_id}] Cost: ${cost.get('primary_cost', 0):.6f}")

                return AnthropicLLMResponse(
                    success=True,
                    content=content,
                    usage=usage,
                    cost=cost,
                    error=None,
                    call_id=call_id,
                    duration_ms=duration_ms,
                    raw_model=str(resp_model) if resp_model else None,
                )

            except Exception as e:
                last_error = str(e)
                print(f"[{call_id}] ❌ 调用失败: {last_error}")

            retries += 1
            if retries <= self.max_retries:
                delay = self.retry_base_delay * (2 ** (retries - 1))
                print(f"[{call_id}] 等待 {delay} 秒后重试...")
                await asyncio.sleep(delay)

        end_time = asyncio.get_event_loop().time()
        duration_ms = int((end_time - start_time) * 1000)
        timestamp_end = self._get_current_timestamp()

        call_record.update(
            {
                "timestamp_end": timestamp_end,
                "duration_ms": duration_ms,
                "status": "failure",
                "response": None,
                "response_length": None,
                "response_model": None,
                "usage": None,
                "cost": None,
                "error": last_error,
                "retry_count": retries - 1,
            }
        )

        if self.tracker:
            self.tracker.log_call_record(call_record)

        return AnthropicLLMResponse(
            success=False,
            content="",
            usage=None,
            cost=None,
            error=last_error,
            call_id=call_id,
            duration_ms=duration_ms,
            raw_model=None,
        )
