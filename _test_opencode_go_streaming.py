#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""通过项目 LLMClient 验证 OpenCode Go 的流式 Chat Completions。"""

from __future__ import annotations

import asyncio

from . import create_llm_client


async def probe_streaming_chat() -> None:
    client = create_llm_client(stage="chat_interaction", enable_tracking=False)
    chunks = []
    async for chunk in client.get_streaming_completion(
        messages=[{"role": "user", "content": "只回复 OK"}],
        max_tokens=128,
        stage="chat_interaction",
    ):
        chunks.append(chunk)

    content = "".join(chunks).strip()
    if not content:
        raise AssertionError("OpenCode Go streaming response is empty")
    print({"profile": client.api_name, "model": client.default_model, "content": content})


if __name__ == "__main__":
    asyncio.run(probe_streaming_chat())
