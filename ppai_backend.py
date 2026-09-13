#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""pp_ai 传输后端（transport backend）。

通过本地 pp_ai 包（import 名 ``pi_ai``）的 openai-completions 实现收发请求，
替代 native 路径的 AsyncOpenAI 传输。上层入口仍是 ``LLMClient`` 的
``get_completion`` / ``get_vision_completion`` / ``get_streaming_completion`` /
``stream_with_reasoning``，这些方法在 ``self.backend == "ppai"`` 时把调用分发到这里。

契约与 native 路径完全对齐：
- call_record 字段集合、usage 五字段、cost 结构（仍走
  ``token_usage_tracker.ModelPricingConfig``）逐字段一致；
- 流式产出 ``(kind, text)``，kind 只有 ``"reasoning"`` / ``"content"``；
- 错误先映射为携带 ``status_code``/``body`` 的异常，再交给既有的
  ``classify_llm_error`` 分类，保证 auth/quota → fallback_eligible、
  rate_limit/server/network → retryable 的判定与 native 一致；
- 重试与 ``_API_KEY_FALLBACK_PROFILES`` 调用期切换在本模块内镜像 native 结构，
  fallback 客户端沿用当前 backend。

usage 映射（实测确认）：pp_ai ``usage.input`` == wire 的 cache-miss 输入，
``usage.cache_read`` == cache-hit 输入；因此
``prompt_tokens = input + cache_read``，``prompt_cache_hit_tokens = cache_read``，
``prompt_cache_miss_tokens = input``。

tool calling 以 pi_agent ``StreamFn`` 形态由 ``agent_stream_fn.create_agent_stream_fn``
实现（tee 透传 13 种事件 + 流末收割 usage 建 call_record），本模块不再保留 stub。
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from .llm_error_classifier import LLMErrorClassification, classify_llm_error
from .token_usage_tracker import TokenUsage

_PI_AI = None


def _pi():
    """延迟导入 pi_ai，native 路径不依赖 pp_ai 安装。"""
    global _PI_AI
    if _PI_AI is None:
        try:
            import pi_ai
            from pi_ai.api import openai_completions
            from pi_ai.providers.all import get_builtin_model, get_builtin_providers
        except ImportError as exc:
            raise ImportError(
                "backend='ppai' 需要 pp_ai 包（import 名 pi_ai），"
                "请先 pip install -e repos/pp_ai"
            ) from exc
        _PI_AI = (pi_ai, openai_completions, get_builtin_model, get_builtin_providers)
    return _PI_AI


class PPAIProviderError(Exception):
    """携带 HTTP status/body 的异常，让 classify_llm_error 看到与 OpenAI SDK 异常相同的形状。"""

    def __init__(self, message: str, status_code: Optional[int] = None, body: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


def _raise_for_terminal(message: Any, model_name: str) -> None:
    """把 pp_ai 终止消息（stop_reason=error/aborted）映射成可分类异常。

    pp_ai 的 error_details.kind → 这里合成的 status/消息关键词 →
    classify_llm_error 的分类结果：
    - auth       → status 401/403（消息含 "authentication"）→ api_key_error/permission_denied，fallback_eligible
    - quota      → 消息含 "quota"（status 402 兜底）→ quota_exhausted，fallback_eligible
    - rate_limit → status 429 → rate_limited，retryable
                   （body 含 GoUsageLimitError 等额度标记时与 native 一样优先判 quota_exhausted）
    - server     → status 5xx → server_error，retryable
    - network    → builtin ConnectionError（类名匹配）→ connection_error，retryable
    - aborted/unknown → 无 status → unknown_error，不 retry 不 fallback
    """
    details = getattr(message, "error_details", None)
    error_message = getattr(message, "error_message", None) or ""
    if details is None:
        raise PPAIProviderError(
            f"pp_ai error ({model_name}): {error_message or 'unknown provider error'}"
        )

    kind = details.kind
    status = details.status
    body = details.body

    if kind == "network":
        # network 类错误的 error_message 可能是空串，补上模型上下文
        raise ConnectionError(
            f"pp_ai network error ({model_name}): {error_message or 'transport failure'}"
        )
    if kind == "auth":
        status = status or 401
        text = f"authentication failed: {error_message or body or 'auth error'}"
    elif kind == "quota":
        status = status or 402
        text = f"quota exhausted: {error_message or body or 'insufficient_quota'}"
    elif kind == "rate_limit":
        status = status or 429
        text = f"rate limit: {error_message or body or 'rate limited'}"
    elif kind == "server":
        status = status or 500
        text = f"server error: {error_message or body or 'upstream 5xx'}"
    elif kind == "aborted":
        text = f"aborted: {error_message or 'request aborted'}"
    else:
        text = f"unknown: {error_message or body or 'unknown pp_ai error'}"

    raise PPAIProviderError(f"pp_ai {kind} ({model_name}): {text}", status_code=status, body=body)


def _resolve_model(client, model_name: str, *, vision: bool = False):
    """构造 pp_ai Model：优先从 builtin catalog 取同 id 模型（拿到 compat/reasoning 等
    标志），再覆盖 base_url 指向当前 profile 的 endpoint；找不到则按 profile 构造裸模型。"""
    pi_ai, _, get_builtin_model, get_builtin_providers = _pi()

    candidates: List[str] = []
    for raw in (
        client.provider,
        (client.provider or "").replace("_", "-"),
        client.family,
        (client.provider or "").split("_")[0],
    ):
        if raw and raw not in candidates:
            candidates.append(raw)

    builtin = None
    for provider_id in candidates:
        builtin = get_builtin_model(provider_id, model_name)
        if builtin is not None:
            break
    if builtin is None:
        for provider_id in get_builtin_providers():
            if provider_id in candidates:
                continue
            builtin = get_builtin_model(provider_id, model_name)
            if builtin is not None:
                break

    if builtin is not None:
        overrides: Dict[str, Any] = {"base_url": client.runtime_config.api_url}
        if vision and "image" not in builtin.input:
            overrides["input"] = list(builtin.input) + ["image"]
        return replace(builtin, **overrides)

    return pi_ai.Model(
        id=model_name,
        api="openai-completions",
        provider=client.provider or "",
        base_url=client.runtime_config.api_url,
        reasoning=bool(client.reasoning_effort),
        input=["text", "image"] if vision else ["text"],
    )


def _openai_parts_to_user_content(parts: List[Any]) -> List[Any]:
    """OpenAI content parts（text / image_url data URL）→ pi_ai UserContent 列表。"""
    pi_ai, _, _, _ = _pi()
    content: List[Any] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            content.append(pi_ai.TextContent(text=part.get("text") or ""))
        elif part.get("type") == "image_url":
            url = (part.get("image_url") or {}).get("url") or ""
            if not url.startswith("data:") or ";base64," not in url:
                raise ValueError(f"ppai backend 仅支持 base64 data URL 图片，got: {url[:60]!r}")
            mime_type = url[5:].split(";base64,", 1)[0]
            data = url.split(";base64,", 1)[1]
            content.append(pi_ai.ImageContent(data=data, mime_type=mime_type))
    return content


def _context_from_messages(messages: List[Dict[str, Any]], system_prompt: Optional[str] = None):
    """OpenAI messages 数组 → pi_ai Context（system 并入 system_prompt）。"""
    pi_ai, _, _, _ = _pi()
    system_parts = [system_prompt] if system_prompt else []
    pi_messages: List[Any] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role == "system":
            if isinstance(content, str) and content:
                system_parts.append(content)
            continue
        if role == "user":
            if isinstance(content, list):
                user_content: Any = _openai_parts_to_user_content(content)
            else:
                user_content = content or ""
            pi_messages.append(pi_ai.UserMessage(content=user_content, timestamp=pi_ai.now_ms()))
        elif role == "assistant":
            if isinstance(content, list):
                text = "".join(
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            else:
                text = content or ""
            pi_messages.append(pi_ai.AssistantMessage(content=[pi_ai.TextContent(text=text)]))
    return pi_ai.Context(
        messages=pi_messages,
        system_prompt="\n".join(p for p in system_parts if p) or None,
    )


def _build_options(
    client,
    *,
    temperature: Optional[float],
    max_tokens: Optional[int],
    extra_body: Optional[Dict[str, Any]] = None,
    reasoning_effort: Optional[str] = None,
    signal: Optional[Any] = None,
):
    """native 的 create_kwargs → pp_ai OpenAICompletionsOptions。

    extra_body（如 DeepSeek 的 thinking 开关）并入 sampling_params（pp_ai 会原样
    update 进请求体）；max_retries=0，重试由 LLMClient 自己负责；signal 透传
    pi_ai 的协作式取消（agent loop 的 abort）。
    """
    _, openai_completions, _, _ = _pi()
    sampling = dict(client.extra_body or {})
    if extra_body:
        sampling.update(extra_body)
    return openai_completions.OpenAICompletionsOptions(
        api_key=client.runtime_config.api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort or client.reasoning_effort,
        sampling_params=sampling,
        max_retries=0,
        signal=signal,
    )


def _token_usage_from_pi(pi_usage: Any) -> Optional[TokenUsage]:
    """pp_ai Usage → TokenUsage；全零视为供应商未上报，返回 None。"""
    if pi_usage is None:
        return None
    if not (pi_usage.input or pi_usage.output or pi_usage.cache_read or pi_usage.total_tokens):
        return None
    prompt_tokens = pi_usage.input + pi_usage.cache_read
    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=pi_usage.output,
        total_tokens=pi_usage.total_tokens or (prompt_tokens + pi_usage.output),
        prompt_cache_hit_tokens=pi_usage.cache_read,
        prompt_cache_miss_tokens=pi_usage.input,
    )


async def _complete_once(
    client,
    *,
    messages: List[Dict[str, Any]],
    model_name: str,
    temperature: Optional[float],
    max_tokens: Optional[int],
    extra_body: Optional[Dict[str, Any]] = None,
    reasoning_effort: Optional[str] = None,
    vision: bool = False,
) -> Tuple[str, Optional[TokenUsage]]:
    """单次 pp_ai 调用（内部仍走 stream + 排空，取终止 AssistantMessage）。"""
    _, openai_completions, _, _ = _pi()
    model_obj = _resolve_model(client, model_name, vision=vision)
    context = _context_from_messages(messages)
    options = _build_options(
        client,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra_body,
        reasoning_effort=reasoning_effort,
    )
    event_stream = openai_completions.stream(model_obj, context, options)
    async for _event in event_stream:
        pass
    message = await event_stream.result()
    if message.stop_reason in ("error", "aborted"):
        _raise_for_terminal(message, model_name)
    content = "".join(block.text for block in message.content if block.type == "text")
    return content, _token_usage_from_pi(message.usage)


async def get_completion(
    client,
    *,
    prompt: str,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    stage: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    max_tokens: Optional[int] = None,
    system_prompt: Optional[str] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    reasoning_effort: Optional[str] = None,
):
    """ppai 后端的非流式 completion；结构镜像 native LLMClient.get_completion。"""
    from .llm_client import LLMResponse

    if client.protocol != "openai_chat":
        raise ValueError(f"ppai backend 仅支持 openai_chat 协议，当前: {client.protocol}")

    call_id = client._generate_call_id()
    timestamp_start = client._get_current_timestamp()
    start_time = asyncio.get_event_loop().time()

    model_name = model or client.default_model
    temperature_effective = (
        temperature if temperature is not None else client.default_temperature
    )

    call_record: Dict[str, Any] = {
        "call_id": call_id,
        "timestamp_start": timestamp_start,
        "api_name": client.api_name,
        "provider": client.provider,
        "family": client.family,
        "protocol": client.protocol,
        "model": model_name,
        "temperature": temperature_effective,
        "prompt": prompt,
        "prompt_length": len(prompt),
        "stage": stage,
    }
    if metadata:
        call_record["metadata"] = metadata

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    retries = 0
    last_error = None
    last_classification: Optional[LLMErrorClassification] = None

    while retries <= client.max_retries:
        try:
            print(f"[{call_id}] 调用API: {client.api_name}, 模型: {model_name}, 重试: {retries}/{client.max_retries} (backend=ppai)")

            content, usage = await _complete_once(
                client,
                messages=messages,
                model_name=model_name,
                temperature=temperature_effective,
                max_tokens=max_tokens,
                extra_body=extra_body,
                reasoning_effort=reasoning_effort,
            )

            if not content.strip():
                raise ValueError("Empty response from LLM")

            cost = None
            if usage and client.tracker:
                cost = client.tracker.pricing_config.calculate_cost(
                    usage=usage,
                    model_name=model_name,
                    provider=client.provider,
                )

            end_time = asyncio.get_event_loop().time()
            duration_ms = int((end_time - start_time) * 1000)
            timestamp_end = client._get_current_timestamp()

            call_record.update({
                "timestamp_end": timestamp_end,
                "duration_ms": duration_ms,
                "status": "success",
                "response": content,
                "response_length": len(content),
                "usage": usage.__dict__ if usage else None,
                "cost": cost,
                "error": None,
                "retry_count": retries,
            })

            if client.tracker:
                client.tracker.log_call_record(call_record)

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

            return LLMResponse(
                success=True,
                content=content,
                usage=usage,
                cost=cost,
                error=None,
                call_id=call_id,
                duration_ms=duration_ms,
            )

        except ValueError as e:
            if "Empty response from LLM" in str(e):
                last_error = f"空响应: {str(e)}"
                print(f"[{call_id}] ⚠️ 收到空响应，重试中...")
            else:
                last_error = f"值错误: {str(e)}"
                print(f"[{call_id}] ❌ 值错误: {e}")
                break

        except Exception as e:
            classification = classify_llm_error(e)
            last_classification = classification
            last_error = f"{classification.kind}: {classification.message}"

            fallback_profile = client._fallback_profile_for_error(classification)
            if fallback_profile:
                fallback_response = await client._run_completion_fallback(
                    classification,
                    prompt=prompt,
                    model=model,
                    temperature=temperature,
                    stage=stage,
                    metadata=metadata,
                    max_tokens=max_tokens,
                    system_prompt=system_prompt,
                )
                client._record_fallback_event(
                    call_record, classification, fallback_profile, retries, start_time
                )
                return fallback_response

            level = "⚠️" if classification.retryable else "❌"
            print(f"[{call_id}] {level} {last_error}")
            if not classification.retryable:
                break

        retries += 1
        if retries <= client.max_retries:
            delay = client.retry_base_delay * (2 ** (retries - 1))
            print(f"[{call_id}] 等待 {delay} 秒后重试...")
            await asyncio.sleep(delay)

    end_time = asyncio.get_event_loop().time()
    duration_ms = int((end_time - start_time) * 1000)
    timestamp_end = client._get_current_timestamp()

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
        "retry_count": retries,
    })

    if client.tracker:
        client.tracker.log_call_record(call_record)

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


async def get_vision_completion(
    client,
    *,
    prompt: Optional[str] = None,
    image_base64: Optional[Any] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    stage: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    system_prompt: Optional[str] = None,
    content_parts: Optional[list] = None,
):
    """ppai 后端的 vision completion；结构镜像 native LLMClient.get_vision_completion。"""
    from .llm_client import LLMResponse

    if client.protocol != "openai_chat":
        raise ValueError(f"ppai backend 仅支持 openai_chat 协议，当前: {client.protocol}")

    call_id = client._generate_call_id()
    timestamp_start = client._get_current_timestamp()
    start_time = asyncio.get_event_loop().time()

    model_name = model or client.default_model
    temperature_effective = (
        temperature if temperature is not None else client.default_temperature
    )

    if content_parts is not None:
        user_content = content_parts
        image_count = sum(
            1 for p in content_parts
            if isinstance(p, dict) and p.get("type") == "image_url"
        )
    else:
        images = (
            [image_base64] if isinstance(image_base64, str)
            else list(image_base64 or [])
        )
        user_content = [{"type": "text", "text": prompt or ""}]
        for b64 in images:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })
        image_count = len(images)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})

    call_record: Dict[str, Any] = {
        "call_id": call_id,
        "timestamp_start": timestamp_start,
        "api_name": client.api_name,
        "provider": client.provider,
        "model": model_name,
        "temperature": temperature_effective,
        "prompt": prompt if content_parts is None else "(multimodal content_parts)",
        "prompt_length": len(prompt) if prompt else 0,
        "has_image": image_count > 0,
        "image_count": image_count,
        "stage": stage,
    }
    if metadata:
        call_record["metadata"] = metadata

    retries = 0
    last_error = None
    last_classification: Optional[LLMErrorClassification] = None

    while retries <= client.max_retries:
        try:
            print(f"[{call_id}] 调用Vision API: {client.api_name}, 模型: {model_name}, 重试: {retries}/{client.max_retries} (backend=ppai)")

            content, usage = await _complete_once(
                client,
                messages=messages,
                model_name=model_name,
                temperature=temperature_effective,
                max_tokens=None,
                vision=True,
            )

            if not content.strip():
                raise ValueError("Empty response from LLM")

            cost = None
            if usage and client.tracker:
                cost = client.tracker.pricing_config.calculate_cost(
                    usage=usage,
                    model_name=model_name,
                    provider=client.provider,
                )

            end_time = asyncio.get_event_loop().time()
            duration_ms = int((end_time - start_time) * 1000)
            timestamp_end = client._get_current_timestamp()

            call_record.update({
                "timestamp_end": timestamp_end,
                "duration_ms": duration_ms,
                "status": "success",
                "response": content,
                "response_length": len(content),
                "usage": usage.__dict__ if usage else None,
                "cost": cost,
                "error": None,
                "retry_count": retries,
            })

            if client.tracker:
                client.tracker.log_call_record(call_record)

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

            return LLMResponse(
                success=True,
                content=content,
                usage=usage,
                cost=cost,
                error=None,
                call_id=call_id,
                duration_ms=duration_ms,
            )

        except ValueError as e:
            if "Empty response from LLM" in str(e):
                last_error = f"空响应: {str(e)}"
                print(f"[{call_id}] ⚠️ 收到空响应，重试中...")
            else:
                last_error = f"值错误: {str(e)}"
                print(f"[{call_id}] ❌ 值错误: {e}")
                break

        except Exception as e:
            classification = classify_llm_error(e)
            last_classification = classification
            last_error = f"{classification.kind}: {classification.message}"

            fallback_profile = client._fallback_profile_for_error(classification)
            if fallback_profile:
                fallback_response = await client._run_vision_fallback(
                    classification,
                    prompt=prompt,
                    image_base64=image_base64,
                    model=model,
                    temperature=temperature,
                    stage=stage,
                    metadata=metadata,
                    system_prompt=system_prompt,
                    content_parts=content_parts,
                )
                client._record_fallback_event(
                    call_record, classification, fallback_profile, retries, start_time
                )
                return fallback_response

            level = "⚠️" if classification.retryable else "❌"
            print(f"[{call_id}] {level} {last_error}")
            if not classification.retryable:
                break

        retries += 1
        if retries <= client.max_retries:
            delay = client.retry_base_delay * (2 ** (retries - 1))
            print(f"[{call_id}] 等待 {delay} 秒后重试...")
            await asyncio.sleep(delay)

    end_time = asyncio.get_event_loop().time()
    duration_ms = int((end_time - start_time) * 1000)
    timestamp_end = client._get_current_timestamp()

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
        "retry_count": retries,
    })

    if client.tracker:
        client.tracker.log_call_record(call_record)

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


async def stream_events(
    client,
    *,
    prompt: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    stage: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    max_tokens: Optional[int] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    record_sink: Optional[Dict[str, Any]] = None,
) -> AsyncGenerator[Tuple[str, str], None]:
    """ppai 后端的流式实现，对应 native LLMClient._stream_events。

    text_delta → ("content", text)，thinking_delta → ("reasoning", text)；
    pp_ai 的 thinking/text 块事件按 content_index 交错，但 delta 语义与 native
    的 reasoning_content/content 一致，直接顺序透传即可。流末回填 record_sink，
    schema 与 native 完全一致。
    """
    _, openai_completions, _, _ = _pi()

    if client.protocol != "openai_chat":
        raise ValueError(f"Streaming 仅支持 openai_chat 协议，当前: {client.protocol}")

    if messages is None and prompt is None:
        raise ValueError("prompt 和 messages 至少提供一个")

    call_id = client._generate_call_id()
    timestamp_start = client._get_current_timestamp()
    start_time = asyncio.get_event_loop().time()

    model_name = model or client.default_model
    temperature_effective = (
        temperature if temperature is not None else client.default_temperature
    )

    prompt_for_log = prompt or ""
    if not prompt_for_log and messages:
        last_content = messages[-1].get("content")
        if isinstance(last_content, str):
            prompt_for_log = last_content[:200]

    call_record: Dict[str, Any] = {
        "call_id": call_id,
        "timestamp_start": timestamp_start,
        "api_name": client.api_name,
        "provider": client.provider,
        "family": client.family,
        "protocol": client.protocol,
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

    full_content: List[str] = []
    usage: Optional[TokenUsage] = None
    error_msg = None
    error_classification: Optional[LLMErrorClassification] = None

    try:
        model_obj = _resolve_model(client, model_name)
        context = _context_from_messages(final_messages)
        options = _build_options(
            client,
            temperature=temperature_effective,
            max_tokens=max_tokens,
        )
        event_stream = openai_completions.stream(model_obj, context, options)
        async for event in event_stream:
            if event.type == "text_delta":
                full_content.append(event.delta)
                yield ("content", event.delta)
            elif event.type == "thinking_delta":
                yield ("reasoning", event.delta)

        message = await event_stream.result()
        if message.stop_reason in ("error", "aborted"):
            _raise_for_terminal(message, model_name)
        usage = _token_usage_from_pi(message.usage)

    except Exception as e:
        classification = classify_llm_error(e)
        error_classification = classification
        fallback_profile = client._fallback_profile_for_error(classification)
        if fallback_profile:
            client._annotate_stream_fallback_record(
                call_record, classification, fallback_profile, start_time
            )
            print(
                f"[{call_id}] 检测到 {classification.kind}，"
                f"切换官方 DeepSeek profile: {fallback_profile}"
            )
            try:
                fallback_client = client._build_fallback_client(fallback_profile)
                async for event in fallback_client._stream_events(
                    prompt=prompt,
                    model=model,
                    temperature=temperature,
                    stage=stage,
                    metadata=metadata,
                    max_tokens=max_tokens,
                    messages=messages,
                    record_sink=record_sink,
                ):
                    yield event
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
    if usage and client.tracker:
        cost = client.tracker.pricing_config.calculate_cost(
            usage=usage, model_name=model_name, provider=client.provider
        )

    call_record.update({
        "timestamp_end": client._get_current_timestamp(),
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

    if client.tracker:
        client.tracker.log_call_record(call_record)

    if record_sink is not None:
        record_sink.clear()
        record_sink.update(call_record)

    if not error_msg:
        print(f"[{call_id}] Streaming完成 ({duration_ms}ms, {len(content_str)} chars)")


__all__ = [
    "PPAIProviderError",
    "get_completion",
    "get_vision_completion",
    "stream_events",
]
