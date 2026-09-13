#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ppai backend 的真实 API 验证脚本（DeepSeek 官方 profile，极小 max_tokens）。

覆盖：
1. stream_with_reasoning parity：native / ppai 同 prompt 各跑一次，断言
   (a) 两种 kind ("reasoning"/"content") 都出现；
   (b) record_sink 回填的 call_record 字段集合与 native 逐 key 完全一致；
   (c) usage 五字段齐全、cost 非空。
2. get_completion parity：两个 backend 的 LLMResponse 字段齐全、success=True。
3. fallback：ppai backend + 故意错误的 api_key 调 opencode_go_deepseek_flash，
   断言 fallback_used 且最终成功（调用期 fallback 到 deepseek_official_flash）。

直接运行：python LLMClient/_test_ppai_backend.py
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

PROFILE = "deepseek_official_flash"
PROMPT = "用一句中文解释什么是检索增强生成（RAG）。"
STREAM_MAX_TOKENS = 1500  # 思考+回答都要完整出来，压到够用的最小值
USAGE_FIELDS = {
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "prompt_cache_hit_tokens",
    "prompt_cache_miss_tokens",
}


def _make_client(backend: str, **overrides) -> LLMClient:
    kwargs = dict(api_name=PROFILE, backend=backend, max_retries=1, retry_base_delay=0.5)
    kwargs.update(overrides)
    return LLMClient(**kwargs)


async def _collect_stream(client: LLMClient, sink: dict):
    events = []
    async for kind, text in client.stream_with_reasoning(
        prompt=PROMPT,
        max_tokens=STREAM_MAX_TOKENS,
        metadata={"test": "ppai_parity"},
        record_sink=sink,
    ):
        assert kind in ("reasoning", "content"), f"非法 kind: {kind!r}"
        events.append((kind, text))
    return events


def _check_record(record: dict, tag: str) -> None:
    usage = record.get("usage")
    assert usage, f"[{tag}] record 缺 usage: {record.keys()}"
    missing = USAGE_FIELDS - set(usage.keys())
    assert not missing, f"[{tag}] usage 缺字段: {missing}"
    assert usage["prompt_tokens"] > 0 and usage["completion_tokens"] > 0, f"[{tag}] usage 异常: {usage}"
    assert (
        usage["prompt_tokens"]
        == usage["prompt_cache_hit_tokens"] + usage["prompt_cache_miss_tokens"]
    ), f"[{tag}] prompt_tokens != hit+miss: {usage}"
    cost = record.get("cost")
    assert cost and cost.get("primary_cost") is not None, f"[{tag}] record 缺 cost"
    assert record.get("status") == "success", f"[{tag}] status={record.get('status')}"


async def test_stream_parity() -> None:
    print("\n=== 1. stream_with_reasoning parity (native vs ppai) ===")
    native_sink: dict = {}
    ppai_sink: dict = {}

    native_events = await _collect_stream(_make_client("native"), native_sink)
    ppai_events = await _collect_stream(_make_client("ppai"), ppai_sink)

    for tag, events in (("native", native_events), ("ppai", ppai_events)):
        kinds = {kind for kind, _ in events}
        assert kinds == {"reasoning", "content"}, f"[{tag}] kind 集合不符: {kinds}"
        content = "".join(t for k, t in events if k == "content")
        assert content.strip(), f"[{tag}] content 为空"
        print(f"[{tag}] events={len(events)} reasoning_chunks={sum(1 for k, _ in events if k == 'reasoning')} "
              f"content_chars={len(content)}")

    native_keys = set(native_sink.keys())
    ppai_keys = set(ppai_sink.keys())
    assert native_keys == ppai_keys, (
        f"record_sink 字段集合不一致:\n"
        f"  only native: {sorted(native_keys - ppai_keys)}\n"
        f"  only ppai:   {sorted(ppai_keys - native_keys)}"
    )
    print(f"record_sink 字段集合一致 ({len(native_keys)} keys): {sorted(native_keys)}")

    _check_record(native_sink, "native")
    _check_record(ppai_sink, "ppai")
    print(f"native usage: {native_sink['usage']}")
    print(f"ppai   usage: {ppai_sink['usage']}")
    print(f"native cost: ¥{native_sink['cost']['primary_cost']:.6f} {native_sink['cost']['currency']}")
    print(f"ppai   cost: ¥{ppai_sink['cost']['primary_cost']:.6f} {ppai_sink['cost']['currency']}")
    print("✅ stream parity 通过")


async def test_completion_parity() -> None:
    print("\n=== 2. get_completion parity (native vs ppai) ===")
    for backend in ("native", "ppai"):
        response = await _make_client(backend).get_completion(
            prompt=PROMPT,
            max_tokens=200,
            system_prompt="你是一个简洁的中文助手。",
            metadata={"test": "ppai_parity"},
        )
        assert response.success, f"[{backend}] 调用失败: {response.error}"
        assert response.content.strip(), f"[{backend}] content 为空"
        assert response.call_id and response.duration_ms is not None
        assert response.usage is not None, f"[{backend}] 缺 usage"
        usage_dict = response.usage.__dict__
        assert USAGE_FIELDS <= set(usage_dict.keys()), f"[{backend}] usage 字段不齐: {usage_dict}"
        assert response.cost and response.cost.get("primary_cost") is not None, f"[{backend}] 缺 cost"
        print(f"[{backend}] ✅ success, {len(response.content)} chars, "
              f"usage={usage_dict}, cost=¥{response.cost['primary_cost']:.6f}")
    print("✅ get_completion parity 通过")


async def test_fallback() -> None:
    print("\n=== 3. ppai backend 调用期 fallback（错误 api_key → deepseek_official_flash）===")
    client = LLMClient(
        api_name="opencode_go_deepseek_flash",
        api_key="sk-ppai-deliberately-invalid-key",
        backend="ppai",
        max_retries=0,
    )
    response = await client.get_completion(prompt=PROMPT, max_tokens=200)
    print(f"success={response.success} fallback_used={response.fallback_used} "
          f"fallback_from={response.fallback_from} fallback_reason={response.fallback_reason}")
    assert response.fallback_used, "未触发 fallback"
    assert response.fallback_from == "opencode_go_deepseek_flash"
    assert response.success, f"fallback 后仍失败: {response.error}"
    assert response.content.strip()
    assert response.usage is not None and response.cost is not None
    print(f"fallback 后内容 {len(response.content)} chars，usage={response.usage.__dict__}")
    print("✅ 调用期 fallback 通过（构造期 fallback 未触发：显式 api_key 使构造成功）")


async def main() -> None:
    await test_stream_parity()
    await test_completion_parity()
    await test_fallback()
    print("\n🎉 全部 ppai backend 验证通过")


if __name__ == "__main__":
    asyncio.run(main())
