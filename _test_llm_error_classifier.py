#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""LLM 错误分类器的最小行为测试。"""

from LLMClient.llm_error_classifier import classify_llm_error


class FakeAPIError(Exception):
    def __init__(self, message="", *, status_code=None, body=None, response=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        self.response = response


class APIConnectionError(Exception):
    pass


def test_invalid_api_key_is_fallback_eligible_and_redacted():
    error = FakeAPIError(
        "401 upstream",
        status_code=401,
        body={
            "type": "error",
            "error": {
                "type": "AuthError",
                "message": "Invalid API key: sk-super-secret-value",
            },
        },
    )
    result = classify_llm_error(error)

    assert result.kind == "api_key_error"
    assert result.status_code == 401
    assert result.api_key_related is True
    assert result.fallback_eligible is True
    assert result.provider_error_type == "AuthError"
    assert "sk-super-secret" not in result.message
    assert "[REDACTED_API_KEY]" in result.message


def test_permission_error_without_auth_hint_does_not_fallback():
    result = classify_llm_error(
        FakeAPIError(
            "country not supported",
            status_code=403,
            body={"error": {"type": "PermissionDenied", "message": "Country not supported"}},
        )
    )

    assert result.kind == "permission_denied"
    assert result.api_key_related is False
    assert result.fallback_eligible is False


def test_opencode_usage_limit_is_quota_not_api_key():
    result = classify_llm_error(
        FakeAPIError(
            "usage limit",
            status_code=429,
            body={"error": {"type": "GoUsageLimitError", "message": "Usage limit reached"}},
        )
    )

    assert result.kind == "quota_exhausted"
    assert result.provider_error_type == "GoUsageLimitError"
    assert result.retryable is False
    assert result.fallback_eligible is True


def test_usage_limit_marker_wins_even_when_gateway_uses_402():
    result = classify_llm_error(
        FakeAPIError(
            "payment required",
            status_code=402,
            body={"error": {"type": "FreeUsageLimitError", "message": "Subscribe to Go"}},
        )
    )

    assert result.kind == "quota_exhausted"
    assert result.fallback_eligible is True


def test_rate_limit_and_server_errors_are_retryable():
    rate = classify_llm_error(
        FakeAPIError("too many requests", status_code=429, body={"error": {"type": "RateLimitError"}})
    )
    server = classify_llm_error(FakeAPIError("bad gateway", status_code=502))

    assert rate.kind == "rate_limited"
    assert rate.retryable is True
    assert server.kind == "server_error"
    assert server.retryable is True


def test_connection_and_timeout_are_distinguished():
    connection = classify_llm_error(APIConnectionError("DNS failure"))
    timeout = classify_llm_error(TimeoutError("timed out"))

    assert connection.kind == "connection_error"
    assert connection.retryable is True
    assert connection.fallback_eligible is False
    assert timeout.kind == "timeout"
    assert timeout.retryable is True


def test_missing_api_key_configuration_is_auth_related():
    result = classify_llm_error(ValueError("缺少API密钥，请为 profile opencode_go_deepseek_pro 配置 api_key"))

    assert result.api_key_related is True
    assert result.fallback_eligible is True


def test_runtime_profile_missing_api_key_message_is_auth_related():
    result = classify_llm_error(
        ValueError(
            "LLM profile 'opencode_go_deepseek_flash' 缺少 api_key。"
            "请通过 --api-key、OPENCODE_GO_API_KEY 或配置文件提供。"
        )
    )

    assert result.kind == "api_key_error"
    assert result.fallback_eligible is True
