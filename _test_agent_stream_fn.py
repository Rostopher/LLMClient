#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""LLMClient→StreamFn 适配器（agent_stream_fn.create_agent_stream_fn）真实 API 验证。

真实 DeepSeek API，max_tokens 调小控成本。覆盖：

1. pi_agent.Agent + 适配器 stream_fn + toy 工具 get_current_time：
   (a) agent loop 完成且最终有文本回答；
   (b) 事件流出现 toolcall_* 事件与 tool_execution_start/end；
   (c) call_records 归集 ≥2 条（工具轮+最终轮），每条 23 字段齐全、cost 非空、
       usage 五字段正确（prompt_tokens == cache_hit + cache_miss）；
   (d) token usage jsonl 有对应追加（call_id 一一对应）。
2. 错误路径（无 fallback 映射的 profile + 坏 key）：不抛异常、终止消息
   stop_reason="error"、error_details.kind="auth"、call_record status="failure"
   且 error_classification.kind="api_key_error"。
3. 透明 fallback（opencode_go_deepseek_flash + 坏 key）：调用方无感知切换
   deepseek_official_flash 并成功；records 第 1 条 status="fallback"、
   第 2 条 status="success"，两条都写入同一 jsonl。

直接运行：python LLMClient/_test_agent_stream_fn.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
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

from LLMClient.agent_stream_fn import create_agent_stream_fn  # noqa: E402

EXPECTED_RECORD_KEYS = {
    # base（与 native 流式路径逐 key 一致）
    "call_id", "timestamp_start", "api_name", "provider", "family", "protocol",
    "model", "temperature", "prompt", "prompt_length", "stage", "streaming", "metadata",
    # finalize
    "timestamp_end", "duration_ms", "status", "response", "response_length",
    "usage", "cost", "error", "error_classification", "retry_count",
}
USAGE_FIELDS = {
    "prompt_tokens", "completion_tokens", "total_tokens",
    "prompt_cache_hit_tokens", "prompt_cache_miss_tokens",
}
BAD_KEY = "sk-agent-stream-fn-deliberately-invalid-key"


def _tmp_log(tag: str) -> str:
    return str(Path(tempfile.mkdtemp(prefix=f"agent_stream_fn_{tag}_")) / "token_usage.jsonl")


def _check_success_record(record: dict, tag: str) -> None:
    missing = EXPECTED_RECORD_KEYS - set(record.keys())
    extra = set(record.keys()) - EXPECTED_RECORD_KEYS
    assert not missing and not extra, f"[{tag}] record 字段集合不符: missing={missing} extra={extra}"
    usage = record.get("usage")
    assert usage, f"[{tag}] record 缺 usage"
    assert USAGE_FIELDS <= set(usage.keys()), f"[{tag}] usage 缺字段: {usage}"
    assert usage["prompt_tokens"] > 0 and usage["completion_tokens"] > 0, f"[{tag}] usage 异常: {usage}"
    assert (
        usage["prompt_tokens"]
        == usage["prompt_cache_hit_tokens"] + usage["prompt_cache_miss_tokens"]
    ), f"[{tag}] prompt_tokens != hit+miss: {usage}"
    cost = record.get("cost")
    assert cost and cost.get("primary_cost") is not None, f"[{tag}] record 缺 cost"


def _check_jsonl(log_file: str, records: list, tag: str) -> None:
    lines = Path(log_file).read_text().strip().splitlines()
    assert len(lines) >= len(records), f"[{tag}] jsonl 行数 {len(lines)} < records {len(records)}"
    logged = [json.loads(line) for line in lines]
    logged_ids = {r["call_id"] for r in logged}
    for record in records:
        assert record["call_id"] in logged_ids, f"[{tag}] jsonl 缺 call_id={record['call_id']}"
    print(f"[{tag}] jsonl 追加 {len(lines)} 行，{len(records)} 条归集 record 的 call_id 全部命中")


async def test_agent_tool_loop() -> None:
    print("\n=== 1. pi_agent Agent + 适配器 stream_fn + get_current_time 工具 ===")
    from pi_agent import Agent, AgentTool, AgentToolResult, MutableAgentState
    from pi_ai import TextContent

    log_file = _tmp_log("tool")
    call_records: list = []
    stream_fn = create_agent_stream_fn(
        stage="chat_interaction",
        max_tokens=2000,
        log_file=log_file,
        call_records_sink=call_records,
    )

    async def execute(tool_call_id, params, signal=None, on_update=None):
        return AgentToolResult(
            content=[TextContent(text="2026-09-11 10:00:00 UTC")],
            details={},
        )

    time_tool = AgentTool(
        name="get_current_time",
        description="获取当前日期和时间（无参数）",
        parameters={"type": "object", "properties": {}},
        execute=execute,
    )

    state = MutableAgentState(
        system_prompt=(
            "你是简洁的中文助手。涉及当前时间的问题必须先调用 get_current_time 工具，"
            "再基于工具结果用一句话回答。"
        ),
        model=stream_fn.model,
    )
    state.tools = [time_tool]
    agent = Agent(stream_fn, initial_state=state)

    events: list = []
    agent.subscribe(lambda event, signal=None: events.append(event))

    await agent.prompt("现在几点了？请先调用工具获取当前时间再回答。")

    event_types = [e.type for e in events]
    print(f"agent events: {event_types.count('turn_start')} turns, 共 {len(events)} 个事件")

    # (a) loop 完成且最终有文本回答
    assert agent.state.error_message is None, f"agent 出错: {agent.state.error_message}"
    final_text = ""
    for msg in reversed(agent.state.messages):
        if msg.role == "assistant":
            final_text = "".join(b.text for b in msg.content if b.type == "text")
            if final_text.strip():
                break
    assert final_text.strip(), "最终无文本回答"
    print(f"(a) ✅ 最终回答: {final_text.strip()[:80]}")

    # (b) toolcall 相关事件 + tool_execution_start/end
    toolcall_events = [
        e for e in events
        if e.type == "message_update" and e.assistant_message_event.type.startswith("toolcall")
    ]
    assert toolcall_events, "事件流缺 toolcall_* 事件"
    assert "tool_execution_start" in event_types, "缺 tool_execution_start"
    assert "tool_execution_end" in event_types, "缺 tool_execution_end"
    toolcall_kinds = sorted({e.assistant_message_event.type for e in toolcall_events})
    print(f"(b) ✅ toolcall 事件: {toolcall_kinds}; tool_execution_start/end 均出现")

    # (c) call_records 归集 ≥2 条，每条 23 字段、cost、usage 五字段
    assert len(call_records) >= 2, f"call_records 仅 {len(call_records)} 条（工具轮+最终轮应 ≥2）"
    for i, record in enumerate(call_records):
        assert record["status"] == "success", f"record[{i}] status={record['status']} error={record['error']}"
        _check_success_record(record, f"tool[{i}]")
    print(f"(c) ✅ 归集 {len(call_records)} 条 record，23 字段/usage/cost 全部校验通过")
    for i, record in enumerate(call_records):
        print(f"    record[{i}] model={record['model']} usage={record['usage']} "
              f"cost=¥{record['cost']['primary_cost']:.6f}")

    # (d) jsonl 对应追加
    _check_jsonl(log_file, call_records, "tool")
    print("(d) ✅ jsonl 追加校验通过")


async def test_auth_error_no_fallback() -> None:
    print("\n=== 2. 错误路径：坏 key + 无 fallback 映射的 profile ===")
    from pi_ai import Context, UserMessage

    log_file = _tmp_log("auth")
    call_records: list = []
    stream_fn = create_agent_stream_fn(
        api_name="deepseek_official_flash",  # 不在 _API_KEY_FALLBACK_PROFILES 的 key 侧
        api_key=BAD_KEY,
        max_tokens=500,
        max_retries=0,
        log_file=log_file,
        call_records_sink=call_records,
    )

    context = Context(messages=[UserMessage(content="你好")])
    stream = stream_fn(stream_fn.model, context, None)  # 契约：不抛异常
    events = [event async for event in stream]
    message = await stream.result()

    assert message.stop_reason == "error", f"stop_reason={message.stop_reason}"
    assert message.error_details is not None, "缺 error_details"
    assert message.error_details.kind == "auth", f"error kind={message.error_details.kind}"
    assert events[-1].type == "error", f"最后一个事件: {events[-1].type}"
    print(f"✅ 未抛异常；stop_reason=error, error_details.kind=auth, "
          f"error_message={str(message.error_message)[:60]!r}")

    assert len(call_records) == 1, f"records={len(call_records)}"
    record = call_records[0]
    assert record["status"] == "failure", f"status={record['status']}"
    assert record["error"], "record 缺 error"
    assert record["error_classification"], "record 缺 error_classification"
    assert record["error_classification"]["kind"] == "api_key_error", (
        f"error_classification={record['error_classification']}"
    )
    assert record["usage"] is None and record["cost"] is None
    missing = EXPECTED_RECORD_KEYS - set(record.keys())
    assert not missing, f"failure record 缺字段: {missing}"
    print(f"✅ failure record: status=failure, error={record['error'][:60]!r}, 23 字段齐全")
    _check_jsonl(log_file, call_records, "auth")


async def test_transparent_fallback() -> None:
    print("\n=== 3. 透明 fallback：坏 key 打 opencode_go → 无感知切换官方 DeepSeek ===")
    from pi_ai import Context, UserMessage

    log_file = _tmp_log("fallback")
    call_records: list = []
    stream_fn = create_agent_stream_fn(
        api_name="opencode_go_deepseek_flash",
        api_key=BAD_KEY,
        max_tokens=1500,
        max_retries=0,
        log_file=log_file,
        call_records_sink=call_records,
    )

    context = Context(messages=[UserMessage(content="用一句中文介绍北京。")])
    stream = stream_fn(stream_fn.model, context, None)
    events = [event async for event in stream]
    message = await stream.result()

    assert message.stop_reason == "stop", f"stop_reason={message.stop_reason} err={message.error_message}"
    text = "".join(b.text for b in message.content if b.type == "text")
    assert text.strip(), "fallback 后无文本"
    assert not [e for e in events if e.type == "error"], "调用方不应看到失败 attempt 的 error 事件"
    assert events[-1].type == "done", f"最后一个事件: {events[-1].type}"
    print(f"✅ 透明切换成功，调用方只见 done；回答: {text.strip()[:60]}")

    assert len(call_records) == 2, f"records={len(call_records)}（fallback attempt + 成功 attempt）"
    first, second = call_records
    assert first["api_name"] == "opencode_go_deepseek_flash"
    assert first["status"] == "fallback", f"first status={first['status']}"
    assert first["fallback_used"] and first["fallback_to"] == "deepseek_official_flash"
    assert first["fallback_reason"] in ("api_key_error", "permission_denied", "quota_exhausted")
    assert second["api_name"] == "deepseek_official_flash"
    assert second["status"] == "success", f"second status={second['status']}"
    _check_success_record(second, "fallback[1]")
    print(f"✅ records: [0] status=fallback → {first['fallback_to']}; "
          f"[1] status=success cost=¥{second['cost']['primary_cost']:.6f}")
    _check_jsonl(log_file, call_records, "fallback")


async def main() -> None:
    await test_agent_tool_loop()
    await test_auth_error_no_fallback()
    await test_transparent_fallback()
    print("\n🎉 全部 agent stream_fn 验证通过")


if __name__ == "__main__":
    asyncio.run(main())
