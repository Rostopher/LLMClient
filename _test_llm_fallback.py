#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""供应商 fallback 的行为测试（不触发真实网络）。"""

import asyncio
from types import SimpleNamespace

import LLMClient as llm_package
from LLMClient.llm_client import LLMClient, LLMResponse


class FakeAPIError(Exception):
    def __init__(self, message, status_code, body):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class FakeCompletions:
    def __init__(self, error):
        self.error = error

    async def create(self, **kwargs):
        raise self.error


class FakeClient:
    def __init__(self, error):
        self.chat = SimpleNamespace(completions=FakeCompletions(error))


def _client_with_failure(error, api_name="opencode_go_deepseek_pro"):
    client = LLMClient.__new__(LLMClient)
    client.api_name = api_name
    client.provider = "opencode_go"
    client.family = "deepseek"
    client.protocol = "openai_chat"
    client.default_model = (
        "deepseek-v4-flash" if api_name.endswith("_flash") else "deepseek-v4-pro"
    )
    client.default_temperature = 0.0
    client.max_retries = 0
    client.retry_base_delay = 0.0
    client.max_concurrent = None
    client.client = FakeClient(error)
    client.tracker = None
    client.enable_tracking = False
    client.runtime_config = SimpleNamespace(log_file=None)
    return client


def test_api_key_error_redirects_to_official_profile(monkeypatch):
    error = FakeAPIError(
        "unauthorized",
        401,
        {"error": {"type": "AuthError", "message": "Invalid API key: sk-do-not-log"}},
    )
    client = _client_with_failure(error)

    fallback = SimpleNamespace(
        get_completion=lambda **kwargs: asyncio.sleep(
            0,
            result=LLMResponse(success=True, content="official answer"),
        )
    )
    requested = []

    def build_fallback(profile):
        requested.append(profile)
        return fallback

    client._build_fallback_client = build_fallback
    response = asyncio.run(client.get_completion("hello"))

    assert response.success is True
    assert response.content == "official answer"
    assert response.fallback_used is True
    assert response.fallback_from == "opencode_go_deepseek_pro"
    assert response.fallback_reason == "api_key_error"
    assert requested == ["deepseek_official_chat"]
    assert "sk-do-not-log" not in str(response.error_classification)


def test_usage_limit_redirects_flash_to_official_profile():
    error = FakeAPIError(
        "usage limit",
        429,
        {
            "error": {
                "type": "GoUsageLimitError",
                "message": "5-hour usage limit reached",
            }
        },
    )
    client = _client_with_failure(error, api_name="opencode_go_deepseek_flash")

    fallback = SimpleNamespace(
        get_completion=lambda **kwargs: asyncio.sleep(
            0,
            result=LLMResponse(success=True, content="official flash answer"),
        )
    )
    requested = []

    def build_fallback(profile):
        requested.append(profile)
        return fallback

    client._build_fallback_client = build_fallback
    response = asyncio.run(client.get_completion("hello"))

    assert response.success is True
    assert response.content == "official flash answer"
    assert response.fallback_used is True
    assert response.fallback_from == "opencode_go_deepseek_flash"
    assert response.fallback_reason == "quota_exhausted"
    assert requested == ["deepseek_official_flash"]


def test_connection_error_is_reported_without_provider_fallback():
    class APIConnectionError(Exception):
        pass

    client = _client_with_failure(APIConnectionError("DNS failure"))
    response = asyncio.run(client.get_completion("hello"))

    assert response.success is False
    assert response.fallback_used is False
    assert response.error_classification["kind"] == "connection_error"


def test_streaming_api_key_error_redirects_before_final_record():
    error = FakeAPIError("unauthorized", 401, {"error": {"message": "Invalid API key"}})
    client = _client_with_failure(error)

    class Fallback:
        async def get_streaming_completion(self, **kwargs):
            yield "official"
            yield " stream"

    client._build_fallback_client = lambda profile: Fallback()

    async def collect():
        return [
            chunk
            async for chunk in client.get_streaming_completion(prompt="hello")
        ]

    assert asyncio.run(collect()) == ["official", " stream"]


def test_factory_redirects_when_go_key_is_missing(monkeypatch):
    calls = []

    class FakeFactoryClient:
        _API_KEY_FALLBACK_PROFILES = LLMClient._API_KEY_FALLBACK_PROFILES

        def __init__(self, **kwargs):
            calls.append(kwargs)
            if kwargs["api_name"] == "opencode_go_deepseek_flash":
                raise ValueError("缺少API密钥，请为 profile opencode_go_deepseek_flash 配置 api_key")

    monkeypatch.setattr(llm_package, "LLMClient", FakeFactoryClient)
    result = llm_package.create_llm_client(
        api_name="opencode_go_deepseek_flash", enable_tracking=False
    )

    assert isinstance(result, FakeFactoryClient)
    assert [call["api_name"] for call in calls] == [
        "opencode_go_deepseek_flash",
        "deepseek_official_flash",
    ]
