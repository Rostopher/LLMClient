#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""OpenCode Go 的 OpenAI Chat Completions 最小联网诊断。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


PACKAGE_DIR = Path(__file__).parent.resolve()
ENV_FILE = PACKAGE_DIR / ".env"
BASE_URL = "https://opencode.ai/zen/go/v1"
DEFAULT_MODELS = ("deepseek-v4-pro", "deepseek-v4-flash")


def _load_api_key() -> str:
    load_dotenv(ENV_FILE, override=False)
    api_key = os.getenv("OPENCODE_GO_API_KEY")
    if not api_key:
        raise RuntimeError(f"缺少 OPENCODE_GO_API_KEY，请配置到 {ENV_FILE}")
    return api_key


def probe_chat_completion(model: str) -> None:
    client = OpenAI(api_key=_load_api_key(), base_url=BASE_URL, timeout=60.0)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "只回复 OK"}],
        temperature=0,
        max_tokens=128,
    )
    choice = response.choices[0]
    usage = response.usage
    prompt_details = getattr(usage, "prompt_tokens_details", None) if usage else None
    completion_details = getattr(usage, "completion_tokens_details", None) if usage else None
    print(
        {
            "requested_model": model,
            "response_model": response.model,
            "object": response.object,
            "has_content": bool(choice.message.content),
            "has_reasoning": bool(getattr(choice.message, "reasoning_content", None)),
            "finish_reason": choice.finish_reason,
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
            "cached_tokens": getattr(prompt_details, "cached_tokens", None),
            "reasoning_tokens": getattr(completion_details, "reasoning_tokens", None),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", default=DEFAULT_MODELS)
    args = parser.parse_args()
    for model in args.models:
        probe_chat_completion(model)


if __name__ == "__main__":
    main()
