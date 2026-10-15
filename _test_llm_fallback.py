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
    client.extra_body = None
    client.reasoning_effort = None
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
        async def _stream_events(self, **kwargs):
            yield ("content", "official")
            yield ("content", " stream")

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


def _magua_fallback_spy(content="magua answer"):
    fallback = SimpleNamespace(
        get_completion=lambda **kwargs: asyncio.sleep(
            0, result=LLMResponse(success=True, content=content),
        )
    )
    requested = []

    def build_fallback(profile):
        requested.append(profile)
        return fallback

    return requested, build_fallback


def test_queue_timeout_on_official_flash_redirects_to_magua():
    error = FakeAPIError(
        "We were unable to start processing your request within the 900-second "
        "timeout limit. Please try again later.",
        None,
        {
            "message": "We were unable to start processing your request within "
            "the 900-second timeout limit. Please try again later."
        },
    )
    client = _client_with_failure(error, api_name="deepseek_official_flash")
    client.provider = "deepseek_official"

    requested, build_fallback = _magua_fallback_spy()
    client._build_fallback_client = build_fallback
    response = asyncio.run(client.get_completion("hello"))

    assert response.success is True
    assert response.content == "magua answer"
    assert response.fallback_used is True
    assert response.fallback_from == "deepseek_official_flash"
    assert response.fallback_reason == "provider_queue_timeout"
    assert requested == ["magua_deepseek_flash"]


class BadFormatCompletions:
    """模拟供应商排队超时后返回 200 但无 choices 的异常响应体。"""

    async def create(self, **kwargs):
        return SimpleNamespace(choices=[], usage=None)


class EmptyCompletions:
    async def create(self, **kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=""))],
            usage=None,
        )


def test_unexpected_response_format_redirects_to_magua():
    client = _client_with_failure(None, api_name="deepseek_official_flash")
    client.provider = "deepseek_official"
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=BadFormatCompletions()))

    requested, build_fallback = _magua_fallback_spy()
    client._build_fallback_client = build_fallback
    response = asyncio.run(client.get_completion("hello"))

    assert response.success is True
    assert response.fallback_used is True
    assert response.fallback_reason == "unexpected_response"
    assert requested == ["magua_deepseek_flash"]


def test_empty_response_retries_then_redirects_to_magua():
    client = _client_with_failure(None, api_name="deepseek_official_flash")
    client.provider = "deepseek_official"
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=EmptyCompletions()))
    client.max_retries = 0  # 直接走到最后一次重试仍为空的分支

    requested, build_fallback = _magua_fallback_spy()
    client._build_fallback_client = build_fallback
    response = asyncio.run(client.get_completion("hello"))

    assert response.success is True
    assert response.fallback_used is True
    assert response.fallback_reason == "empty_response"
    assert requested == ["magua_deepseek_flash"]


def test_unexpected_response_format_without_mapping_does_not_fallback():
    client = _client_with_failure(None, api_name="magua_deepseek_flash")
    client.provider = "magua"
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=BadFormatCompletions()))

    response = asyncio.run(client.get_completion("hello"))

    assert response.success is False
    assert response.fallback_used is False
    assert response.error_classification["kind"] == "unexpected_response"


def test_vision_queue_timeout_redirects_to_magua():
    error = FakeAPIError(
        "We were unable to start processing your request within the 900-second "
        "timeout limit. Please try again later.",
        None,
        {"message": "We were unable to start processing your request within "
        "the 900-second timeout limit. Please try again later."},
    )
    client = _client_with_failure(error, api_name="deepseek_official_flash_vision")
    client.provider = "deepseek_official"

    fallback = SimpleNamespace(
        get_vision_completion=lambda **kwargs: asyncio.sleep(
            0, result=LLMResponse(success=True, content="magua vision answer"),
        )
    )
    requested = []

    def build_fallback(profile):
        requested.append(profile)
        return fallback

    client._build_fallback_client = build_fallback
    response = asyncio.run(client.get_vision_completion(prompt="hello", image_base64="AAAA"))

    assert response.success is True
    assert response.content == "magua vision answer"
    assert response.fallback_used is True
    assert response.fallback_from == "deepseek_official_flash_vision"
    assert requested == ["magua_deepseek_flash"]


class FakeStreamCompletions:
    """记录请求参数并返回假流式 chunk 序列。"""

    def __init__(self, chunks):
        self.chunks = chunks
        self.captured = {}

    async def create(self, **kwargs):
        self.captured.update(kwargs)
        chunks = self.chunks

        class _Stream:
            def __aiter__(self_):
                async def gen():
                    for chunk in chunks:
                        yield chunk

                return gen()

        return _Stream()


def test_force_stream_aggregates_stream_for_get_completion():
    client = _client_with_failure(None, api_name="magua_deepseek_flash")
    client.provider = "magua"
    client.force_stream = True
    client.model_aliases = {"deepseek-v4-flash": "deepseek-v4.1-flash"}
    chunks = [
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(role="assistant", content=None))],
            usage=None,
        ),
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(reasoning_content="想", content="你好"))],
            usage=None,
        ),
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(reasoning_content=None, content="，世界"))],
            usage=None,
        ),
        SimpleNamespace(choices=[], usage=None),
    ]
    completions = FakeStreamCompletions(chunks)
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    response = asyncio.run(client.get_completion("hi", model="deepseek-v4-flash"))

    assert response.success is True
    assert response.content == "你好，世界"
    assert completions.captured["stream"] is True
    assert completions.captured["model"] == "deepseek-v4.1-flash"
