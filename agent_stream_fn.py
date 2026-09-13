#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""LLMClient → pi_agent ``StreamFn`` 适配器。

让 pi_agent（agent loop）的每次 LLM 调用都走 LLMClient 的计费/路由/fallback 管道：

- 路由：按 stage（默认 ``chat_interaction``）或显式 profile 经现有 resolve 机制拿
  model/base_url/key/temperature/extra_body/reasoning_effort；pi_agent 传入的
  ``model`` 只提供模型名覆盖，endpoint/key 始终来自 LLMClient profile。
- 计费：每次调用生成与 native 流式路径逐 key 一致的 call_record（含 metadata 共
  23 字段），追加进 token usage jsonl（``tracker.log_call_record``），cost 走
  ``ModelPricingConfig``；usage 映射复用 ``ppai_backend._token_usage_from_pi``，
  不复制第二份。
- 契约：返回的 callable 符合 pi_agent ``StreamFn``——绝不抛异常，失败编码为流内
  error 事件 + ``stop_reason="error"/"aborted"`` 的终止消息。
- fallback：终止消息 ``stop_reason="error"`` 且错误分类 fallback_eligible
  （auth/quota）且当前 profile 在 ``LLMClient._API_KEY_FALLBACK_PROFILES`` 里 →
  透明切换 fallback profile 重新流式（只切换一次，不重试），语义对齐
  ``ppai_backend.stream_events`` 的流式 fallback。
- 工具透传：``context.tools`` 原样交给 pi_ai，13 种 AssistantMessageEvent 原样
  透传给调用方；事件流做 tee——边透传边在流末收割 usage 建 call_record。
- 归集：``call_records_sink``（list，逐 attempt append）、``record_sink``（dict，
  回填最终 attempt 的 record）、``stream_fn.call_records``（accessor）三种取数方式，
  供 chat 端点按回合结算总成本。
- abort：``options.signal`` 透传到 pi_ai 传输层。

pi_ai/pi_agent 均为函数内延迟导入，native 使用者零新依赖。
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, List, Optional

from . import ppai_backend
from .llm_client import LLMClient
from .llm_error_classifier import LLMErrorClassification, classify_llm_error
from .token_usage_tracker import TokenUsage

# classify_llm_error 的 kind → pi_ai AssistantMessageErrorDetails.kind，
# 仅在泵协程遇到流外异常（按 StreamFn 契约需合成流内 error）时使用。
_CLASSIFICATION_TO_ERROR_KIND = {
    "api_key_error": "auth",
    "permission_denied": "auth",
    "quota_exhausted": "quota",
    "rate_limited": "rate_limit",
    "server_error": "server",
    "connection_error": "network",
}


def create_agent_stream_fn(
    stage: str = "chat_interaction",
    *,
    api_name: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
    record_sink: Optional[Dict[str, Any]] = None,
    call_records_sink: Optional[List[Dict[str, Any]]] = None,
    fallback_api: Optional[str] = None,
    **client_kwargs,
) -> Callable[..., Any]:
    """创建符合 pi_agent ``StreamFn`` 契约的 stream 函数。

    签名：``stream_fn(model, context, options=None) -> AssistantMessageEventStream``。

    Args:
        stage: stage routing 的阶段名（默认 ``chat_interaction``），显式给
            ``api_name`` 时忽略。
        api_name: 显式 profile 名，优先于 stage 路由。
        model/temperature: 覆盖路由结果的模型/温度。
        max_tokens: 每次 LLM 调用的 max_tokens 默认值；``options.max_tokens``
            （agent loop 配置）优先。
        metadata: 并入每条 call_record 的 metadata 字段。
        record_sink: 可选 dict，回填最终 attempt 的 call_record（对齐 native
            ``stream_with_reasoning`` 的 record_sink 语义）。
        call_records_sink: 可选 list，每次 attempt（含 fallback 前的失败 attempt）
            的 call_record 依次 append，供按回合归集结算。
        fallback_api: stage 无路由时的兜底 profile（对齐 ``LLMClient.from_stage``）。
        **client_kwargs: 透传 ``LLMClient`` 构造（如 ``log_file``、``max_retries``、
            ``api_key``）；``backend`` 缺省固定为 ``"ppai"``。

    返回的 callable 上挂有三个属性：``client``（LLMClient 实例）、``model``
    （按当前 profile resolve 出的 pi_ai Model，可直接放进 Agent initial_state）、
    ``call_records``（本工厂产生全部 call_record 的 list accessor）。
    """
    client_kwargs.setdefault("backend", "ppai")
    if api_name:
        client = LLMClient.from_profile(
            api_name, model=model, temperature=temperature, **client_kwargs
        )
    else:
        from .stage_routing import get_route_for_stage

        route = get_route_for_stage(stage, default=fallback_api)
        client = LLMClient(
            api_name=route.get("api_name") or fallback_api,
            model=model or route.get("model"),
            temperature=temperature if temperature is not None else route.get("temperature"),
            **client_kwargs,
        )

    call_records: List[Dict[str, Any]] = []
    base_metadata: Dict[str, Any] = {"source": "pi_agent.stream_fn"}
    if metadata:
        base_metadata.update(metadata)

    def stream_fn(model, context, options=None):
        """StreamFn 契约：同步返回 event stream，运行期失败只走流内 error。"""
        pi_ai, _, _, _ = ppai_backend._pi()
        from pi_ai.utils.event_stream import setup_error_stream
        from pi_ai.utils.tasks import spawn

        model_name = getattr(model, "id", None) or client.default_model
        out = pi_ai.create_assistant_message_event_stream()
        try:
            spawn(_pump(
                client=client,
                out=out,
                model_name=model_name,
                context=context,
                options=options,
                stage=None if api_name else stage,
                max_tokens=max_tokens,
                base_metadata=base_metadata,
                record_sink=record_sink,
                call_records=call_records,
                call_records_sink=call_records_sink,
            ))
        except Exception as exc:
            fallback_model = model if model is not None else stream_fn.model
            return setup_error_stream(fallback_model, exc)
        return out

    stream_fn.client = client  # type: ignore[attr-defined]
    stream_fn.call_records = call_records  # type: ignore[attr-defined]
    stream_fn.model = ppai_backend._resolve_model(client, model or client.default_model)  # type: ignore[attr-defined]
    return stream_fn


def _prompt_text_from_context(context: Any) -> str:
    """取最后一条 user 消息文本（截 200 字符）作为 call_record 的 prompt 日志字段。"""
    for msg in reversed(getattr(context, "messages", None) or []):
        if getattr(msg, "role", None) != "user":
            continue
        content = getattr(msg, "content", "")
        if isinstance(content, str):
            return content[:200]
        if isinstance(content, list):
            text = "".join(
                getattr(part, "text", "")
                for part in content
                if getattr(part, "type", None) == "text"
            )
            return text[:200]
    return ""


def _error_message_from_exception(pi_ai: Any, model_obj: Any, exc: BaseException,
                                  classification: LLMErrorClassification) -> Any:
    """流外异常 → 终止 AssistantMessage（stop_reason="error"），保持 StreamFn 契约。"""
    details = pi_ai.AssistantMessageErrorDetails(
        kind=_CLASSIFICATION_TO_ERROR_KIND.get(classification.kind, "unknown"),
        status=classification.status_code,
        body=classification.response_body,
    )
    return pi_ai.AssistantMessage(
        api=model_obj.api,
        provider=model_obj.provider,
        model=model_obj.id,
        content=[],
        stop_reason="error",
        error_message=str(exc) or type(exc).__name__,
        error_details=details,
    )


def _finalize_record(
    attempt_client: LLMClient,
    call_record: Dict[str, Any],
    record_sink: Optional[Dict[str, Any]],
    call_records: List[Dict[str, Any]],
    call_records_sink: Optional[List[Dict[str, Any]]],
) -> None:
    """写 jsonl + 回填三类 sink（最终 attempt 才回填 record_sink）。"""
    if attempt_client.tracker:
        attempt_client.tracker.log_call_record(call_record)
    if record_sink is not None:
        record_sink.clear()
        record_sink.update(call_record)
    call_records.append(call_record)
    if call_records_sink is not None:
        call_records_sink.append(call_record)


async def _pump(
    *,
    client: LLMClient,
    out: Any,
    model_name: str,
    context: Any,
    options: Any,
    stage: Optional[str],
    max_tokens: Optional[int],
    base_metadata: Dict[str, Any],
    record_sink: Optional[Dict[str, Any]],
    call_records: List[Dict[str, Any]],
    call_records_sink: Optional[List[Dict[str, Any]]],
) -> None:
    """泵协程：把 pi_ai 上游事件 tee 进 out，流末收割 usage 建 call_record。

    本体绝不向上抛异常——任何意外都编码成 out 里的 error 事件 + failure record。
    """
    pi_ai, openai_completions, _, _ = ppai_backend._pi()

    options_temperature = getattr(options, "temperature", None)
    temperature_eff = (
        options_temperature if options_temperature is not None else client.default_temperature
    )
    options_max_tokens = getattr(options, "max_tokens", None)
    max_tokens_eff = options_max_tokens if options_max_tokens is not None else max_tokens
    reasoning = getattr(options, "reasoning", None) or None
    signal = getattr(options, "signal", None)

    prompt_for_log = _prompt_text_from_context(context)
    attempt_client = client
    call_record: Dict[str, Any] = {}
    call_id = ""
    start_time = asyncio.get_event_loop().time()
    text_parts: List[str] = []
    usage: Optional[TokenUsage] = None
    error_msg: Optional[str] = None
    error_classification: Optional[LLMErrorClassification] = None
    pushed_start = False

    try:
        while True:
            call_id = attempt_client._generate_call_id()
            start_time = asyncio.get_event_loop().time()
            text_parts = []
            usage = None
            error_msg = None
            error_classification = None

            call_record = {
                "call_id": call_id,
                "timestamp_start": attempt_client._get_current_timestamp(),
                "api_name": attempt_client.api_name,
                "provider": attempt_client.provider,
                "family": attempt_client.family,
                "protocol": attempt_client.protocol,
                "model": model_name,
                "temperature": temperature_eff,
                "prompt": prompt_for_log,
                "prompt_length": len(prompt_for_log),
                "stage": stage,
                "streaming": True,
                "metadata": dict(base_metadata),
            }

            model_obj = ppai_backend._resolve_model(attempt_client, model_name)
            pi_options = ppai_backend._build_options(
                attempt_client,
                temperature=temperature_eff,
                max_tokens=max_tokens_eff,
                reasoning_effort=reasoning,
                signal=signal,
            )

            terminal_event = None
            message = None
            classification: Optional[LLMErrorClassification] = None
            try:
                upstream = openai_completions.stream(model_obj, context, pi_options)
                async for event in upstream:
                    if event.type in ("done", "error"):
                        # 扣住终止事件：先据此决定是否透明 fallback，再决定是否透传
                        terminal_event = event
                        continue
                    if event.type == "start":
                        if pushed_start:
                            # fallback 重流的第二个 start 不透传，避免调用方重复 append partial
                            continue
                        pushed_start = True
                    if event.type == "text_delta":
                        text_parts.append(event.delta)
                    out.push(event)
                message = await upstream.result()
            except Exception as exc:
                classification = classify_llm_error(exc)
                message = _error_message_from_exception(pi_ai, model_obj, exc, classification)
                terminal_event = pi_ai.ErrorEvent(reason="error", error=message)

            if message.stop_reason not in ("error", "aborted"):
                if terminal_event is not None:
                    out.push(terminal_event)
                usage = ppai_backend._token_usage_from_pi(message.usage)
                break

            if classification is None:
                try:
                    ppai_backend._raise_for_terminal(message, model_name)
                except Exception as exc:
                    classification = classify_llm_error(exc)
                else:  # pragma: no cover - _raise_for_terminal 必然抛出
                    classification = LLMErrorClassification(kind="unknown_error", message="unknown")

            error_msg = f"{classification.kind}: {classification.message}"
            error_classification = classification

            # 只切换一次：fallback attempt 自身失败不再二次切换（对齐现有流式 fallback）
            fallback_profile = (
                attempt_client._fallback_profile_for_error(classification)
                if attempt_client is client
                else None
            )
            if not fallback_profile:
                if terminal_event is not None:
                    out.push(terminal_event)
                break

            # 透明 fallback：原 attempt 记为 status="fallback"（写 jsonl + 归集，
            # 但不占 record_sink），随后用 fallback profile 重新流式。auth/quota
            # 失败发生在首个 SSE 之前，正常不会有内容事件已外泄。
            attempt_client._annotate_stream_fallback_record(
                call_record, classification, fallback_profile, start_time
            )
            print(
                f"[{call_id}] 检测到 {classification.kind}，"
                f"透明切换 fallback profile: {fallback_profile} (agent stream_fn)"
            )
            call_records.append(call_record)
            if call_records_sink is not None:
                call_records_sink.append(call_record)
            attempt_client = client._build_fallback_client(fallback_profile)

        end_time = asyncio.get_event_loop().time()
        duration_ms = int((end_time - start_time) * 1000)
        content_str = "".join(text_parts)

        cost = None
        if usage and attempt_client.tracker:
            cost = attempt_client.tracker.pricing_config.calculate_cost(
                usage=usage, model_name=model_name, provider=attempt_client.provider
            )

        call_record.update({
            "timestamp_end": attempt_client._get_current_timestamp(),
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
        _finalize_record(attempt_client, call_record, record_sink, call_records, call_records_sink)

        if not error_msg:
            print(f"[{call_id}] Agent stream 完成 ({duration_ms}ms, {len(content_str)} chars)")
        else:
            print(f"[{call_id}] Agent stream 失败: {error_msg}")

    except Exception as exc:  # 保底：适配器绝不向上抛，编码为流内 error + failure record
        classification = classify_llm_error(exc)
        if not out.done:
            try:
                model_obj = ppai_backend._resolve_model(attempt_client, model_name)
                message = _error_message_from_exception(pi_ai, model_obj, exc, classification)
                out.push(pi_ai.ErrorEvent(reason="error", error=message))
            except Exception:
                pass
        if call_record:
            try:
                content_str = "".join(text_parts)
                call_record.update({
                    "timestamp_end": attempt_client._get_current_timestamp(),
                    "duration_ms": int((asyncio.get_event_loop().time() - start_time) * 1000),
                    "status": "failure",
                    "response": content_str,
                    "response_length": len(content_str),
                    "usage": None,
                    "cost": None,
                    "error": f"{classification.kind}: {classification.message}",
                    "error_classification": classification.to_dict(),
                    "retry_count": 0,
                })
                _finalize_record(attempt_client, call_record, record_sink, call_records, call_records_sink)
            except Exception:
                pass


__all__ = ["create_agent_stream_fn"]
