#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""LLMClient 流式重构回归测试。

验证 _stream_events / stream_with_reasoning / get_streaming_completion 三层关系：
- stream_with_reasoning 产出 ("reasoning"|"content", text) 二元组
- get_streaming_completion 保持旧契约：只 yield content 字符串
- reasoning 不混入 full_content（usage 记录语义不变）

不连真实 API：用 __new__ 绕过构造函数，注入假的 AsyncOpenAI client。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path


def _project_root() -> Path:
    try:
        return Path(__file__).resolve().parents[1]
    except NameError:  # pragma: no cover
        return Path.cwd().resolve()


def _install_project_path() -> None:
    project_root = _project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


_install_project_path()

from LLMClient.llm_client import LLMClient  # noqa: E402


class _FakeDelta:
    def __init__(self, content=None, reasoning=None):
        self.content = content
        if reasoning is not None:
            self.reasoning_content = reasoning


class _FakeChoice:
    def __init__(self, delta):
        self.delta = delta


class _FakeChunk:
    def __init__(self, delta=None):
        self.choices = [_FakeChoice(delta)] if delta is not None else []


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        async def _gen():
            for c in self._chunks:
                yield c

        return _gen()


class _FakeCompletions:
    def __init__(self, chunks):
        self._chunks = chunks

    async def create(self, **kwargs):
        assert kwargs.get("stream") is True
        return _FakeStream(self._chunks)


class _FakeChat:
    def __init__(self, chunks):
        self.completions = _FakeCompletions(chunks)


class _FakeOpenAIClient:
    def __init__(self, chunks):
        self.chat = _FakeChat(chunks)


def _make_client(chunks) -> LLMClient:
    client = LLMClient.__new__(LLMClient)
    client.protocol = "openai_chat"
    client.default_model = "fake-model"
    client.default_temperature = 0.7
    client.reasoning_effort = None
    client.extra_body = None
    client.tracker = None
    client.provider = "fake"
    client.api_name = "fake"
    client.family = None
    client.client = _FakeOpenAIClient(chunks)
    return client


_CHUNKS = [
    _FakeChunk(_FakeDelta(reasoning="先想")),
    _FakeChunk(_FakeDelta(reasoning="再想")),
    _FakeChunk(_FakeDelta(content="你")),
    _FakeChunk(_FakeDelta(content="好")),
    _FakeChunk(),  # 空 choices 的收尾 chunk
]


def test_stream_with_reasoning_yields_tuples():
    async def _run():
        events = []
        async for ev in _make_client(_CHUNKS).stream_with_reasoning(
            messages=[{"role": "user", "content": "hi"}]
        ):
            events.append(ev)
        return events

    events = asyncio.run(_run())
    assert events == [
        ("reasoning", "先想"),
        ("reasoning", "再想"),
        ("content", "你"),
        ("content", "好"),
    ], f"事件序列不符: {events}"


def test_get_streaming_completion_keeps_legacy_contract():
    async def _run():
        chunks = []
        async for text in _make_client(_CHUNKS).get_streaming_completion(
            messages=[{"role": "user", "content": "hi"}]
        ):
            chunks.append(text)
        return chunks

    chunks = asyncio.run(_run())
    assert chunks == ["你", "好"], f"旧契约被破坏: {chunks}"


def test_no_reasoning_model_still_works():
    chunks = [_FakeChunk(_FakeDelta(content="a")), _FakeChunk(_FakeDelta(content="b"))]

    async def _run():
        return [ev async for ev in _make_client(chunks).stream_with_reasoning(prompt="hi")]

    events = asyncio.run(_run())
    assert events == [("content", "a"), ("content", "b")]


def test_record_sink_filled_after_stream():
    """record_sink 在流结束时回填 call_record（含 call_id/status），供计费用。"""
    sink: dict = {}

    async def _run():
        return [
            ev
            async for ev in _make_client(_CHUNKS).stream_with_reasoning(
                messages=[{"role": "user", "content": "hi"}], record_sink=sink
            )
        ]

    events = asyncio.run(_run())
    assert events[-1] == ("content", "好")
    assert sink.get("call_id"), f"record_sink 未回填 call_id: {sink}"
    assert sink.get("status") == "success"
    assert sink.get("streaming") is True
    # 并发安全契约：sink 是调用方持有的 dict，client 实例上不应残留状态
    assert not hasattr(_make_client(_CHUNKS), "last_call_record")


def test_record_sink_optional_default_none():
    """不传 record_sink 时行为与原来一致。"""

    async def _run():
        return [
            ev
            async for ev in _make_client(_CHUNKS).stream_with_reasoning(
                messages=[{"role": "user", "content": "hi"}]
            )
        ]

    events = asyncio.run(_run())
    assert events[-1] == ("content", "好")
