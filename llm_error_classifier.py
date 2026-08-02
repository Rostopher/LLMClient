#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""将 LLM/OpenAI 兼容接口异常归一化，供重试和供应商 fallback 使用。

OpenCode Go 的部分错误会以 ``text/plain`` Content-Type 返回 JSON，因此这里
同时检查异常的 ``body``、``response`` 和字符串内容，而不依赖 Content-Type。
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional

try:  # openai 不是分类器的硬依赖，便于单元测试和降级诊断
    from openai import APIConnectionError as _OpenAIAPIConnectionError
except ImportError:  # pragma: no cover - 运行时由 LLMClient 负责提示依赖
    _OpenAIAPIConnectionError = None


_MAX_TEXT_LENGTH = 2000
_SECRET_PATTERNS = (
    (re.compile(r"sk-[A-Za-z0-9_-]{8,}"), "[REDACTED_API_KEY]"),
    (re.compile(r"(?i)(authorization\s*:\s*bearer\s+)[^\s,}]+"), r"\1[REDACTED_TOKEN]"),
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~-]{12,}"), r"\1[REDACTED_TOKEN]"),
)


@dataclass(frozen=True)
class LLMErrorClassification:
    """可记录、可审计的统一错误描述。"""

    kind: str
    message: str
    status_code: Optional[int] = None
    provider_error_type: Optional[str] = None
    retryable: bool = False
    api_key_related: bool = False
    fallback_eligible: bool = False
    response_body: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sanitize(value: Any, limit: int = _MAX_TEXT_LENGTH) -> str:
    if isinstance(value, (Mapping, list, tuple)):
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            text = str(value)
    else:
        text = str(value)
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    if len(text) > limit:
        return text[:limit] + "…"
    return text


def _payload_from_exception(error: BaseException) -> Any:
    for name in ("body", "response_body", "error_body"):
        value = getattr(error, name, None)
        if value is not None:
            return value

    response = getattr(error, "response", None)
    if response is None:
        return None
    if isinstance(response, Mapping):
        return response
    json_method = getattr(response, "json", None)
    if callable(json_method):
        try:
            return json_method()
        except Exception:  # pragma: no cover - 依赖第三方 response 实现
            pass
    text = getattr(response, "text", None)
    if text is not None:
        return text() if callable(text) else text
    return None


def _parse_json_text(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


def _find_values(value: Any, keys: set[str]) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in keys:
                found.append(item)
            found.extend(_find_values(item, keys))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.extend(_find_values(item, keys))
    return found


def _status_code(error: BaseException) -> Optional[int]:
    status = getattr(error, "status_code", None)
    if status is None:
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def _is_connection_error(error: BaseException) -> bool:
    if _OpenAIAPIConnectionError is not None and isinstance(error, _OpenAIAPIConnectionError):
        return True
    return error.__class__.__name__ in {
        "APIConnectionError",
        "ConnectError",
        "ConnectionError",
        "NetworkError",
    }


def _is_timeout(error: BaseException) -> bool:
    if isinstance(error, (asyncio.TimeoutError, TimeoutError)):
        return True
    name = error.__class__.__name__.lower()
    return "timeout" in name or name in {"apitimeouterror", "readtimeout"}


def classify_llm_error(error: BaseException) -> LLMErrorClassification:
    """分类 OpenAI SDK、OpenCode Go 或普通网络异常。

    ``fallback_eligible`` 仅表示“可尝试官方 DeepSeek”，并不表示所有 4xx
    都应该切换供应商：配额、限流、参数错误和区域限制会被明确区分。
    """

    raw_message = _sanitize(str(error) or error.__class__.__name__)
    status = _status_code(error)
    payload = _parse_json_text(_payload_from_exception(error))
    body_text = _sanitize(payload) if payload is not None else None

    if _is_timeout(error):
        return LLMErrorClassification(
            kind="timeout", message=raw_message, retryable=True, response_body=body_text
        )
    if _is_connection_error(error):
        return LLMErrorClassification(
            kind="connection_error", message=raw_message, retryable=True, response_body=body_text
        )

    type_values = _find_values(payload, {"type", "code", "error_type"}) if payload is not None else []
    message_values = _find_values(payload, {"message", "detail", "error_description"}) if payload is not None else []
    scalar_type_values = [
        str(value)
        for value in type_values
        if value is not None and not isinstance(value, (dict, list))
    ]
    provider_error_type = next(
        (value for value in scalar_type_values if value.lower() not in {"error", "api_error"}),
        scalar_type_values[0] if scalar_type_values else None,
    )
    provider_message = next(
        (str(value) for value in message_values if value is not None and not isinstance(value, (dict, list))),
        "",
    )
    detail_text = _sanitize(" ".join(part for part in (provider_error_type or "", provider_message, raw_message) if part))
    lowered = detail_text.lower()

    auth_hint = any(
        marker in lowered
        for marker in (
            "api key", "api_key", "apikey", "authentication", "unauthorized", "credential",
            "api密钥", "apikey_env", "缺少api", "密钥", "missing key", "invalid key",
            "token is invalid", "autherror",
        )
    )
    quota_hint = any(
        marker in lowered
        for marker in (
            "gousagelimiterror", "freeusagelimiterror", "quota", "insufficient_quota",
            "usage limit", "usage exceeded", "credits exhausted", "balance exhausted",
        )
    )

    if status == 401 or (status is None and auth_hint):
        return LLMErrorClassification(
            kind="api_key_error",
            message=detail_text,
            status_code=status,
            provider_error_type=_sanitize(provider_error_type) if provider_error_type else None,
            retryable=False,
            api_key_related=True,
            fallback_eligible=True,
            response_body=body_text,
        )
    if status == 403:
        return LLMErrorClassification(
            kind="permission_denied",
            message=detail_text,
            status_code=status,
            provider_error_type=_sanitize(provider_error_type) if provider_error_type else None,
            retryable=False,
            api_key_related=auth_hint,
            fallback_eligible=auth_hint,
            response_body=body_text,
        )
    if quota_hint:
        return LLMErrorClassification(
            kind="quota_exhausted",
            message=detail_text,
            status_code=status,
            provider_error_type=_sanitize(provider_error_type) if provider_error_type else None,
            retryable=False,
            response_body=body_text,
        )
    if status == 429:
        return LLMErrorClassification(
            kind="rate_limited",
            message=detail_text,
            status_code=status,
            provider_error_type=_sanitize(provider_error_type) if provider_error_type else None,
            retryable=True,
            response_body=body_text,
        )
    if status is not None and status >= 500:
        return LLMErrorClassification(
            kind="server_error", message=detail_text, status_code=status,
            provider_error_type=_sanitize(provider_error_type) if provider_error_type else None,
            retryable=True, response_body=body_text,
        )
    if status in {408, 409}:
        return LLMErrorClassification(
            kind="transient_api_error", message=detail_text, status_code=status,
            provider_error_type=_sanitize(provider_error_type) if provider_error_type else None,
            retryable=True, response_body=body_text,
        )
    if status == 400:
        return LLMErrorClassification(
            kind="bad_request", message=detail_text, status_code=status,
            provider_error_type=_sanitize(provider_error_type) if provider_error_type else None,
            response_body=body_text,
        )
    if status == 404:
        return LLMErrorClassification(
            kind="not_found", message=detail_text, status_code=status,
            provider_error_type=_sanitize(provider_error_type) if provider_error_type else None,
            response_body=body_text,
        )
    if status == 422:
        return LLMErrorClassification(
            kind="validation_error", message=detail_text, status_code=status,
            provider_error_type=_sanitize(provider_error_type) if provider_error_type else None,
            response_body=body_text,
        )

    return LLMErrorClassification(
        kind="api_error" if status is not None or error.__class__.__name__.endswith("APIError") else "unknown_error",
        message=detail_text,
        status_code=status,
        provider_error_type=_sanitize(provider_error_type) if provider_error_type else None,
        retryable=False,
        api_key_related=auth_hint,
        fallback_eligible=auth_hint,
        response_body=body_text,
    )


__all__ = ["LLMErrorClassification", "classify_llm_error"]
