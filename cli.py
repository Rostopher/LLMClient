#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Command line utilities for LLMClient_v2."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Optional

from .cli_args import add_llm_cli_args, create_client_from_cli, resolve_llm_cli_config
from .llm_runtime_config import list_runtime_profiles


def _read_prompt(prompt: Optional[str], prompt_file: Optional[str]) -> str:
    if prompt_file:
        return Path(prompt_file).expanduser().resolve().read_text(encoding="utf-8")
    if prompt:
        return prompt
    return "你好，请用一句话说明你已正常工作。"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m LLMClient_v2.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("list-apis", help="List configured LLM profiles")
    add_llm_cli_args(list_parser)

    smoke_parser = sub.add_parser("smoke", help="Run one smoke-test LLM call")
    add_llm_cli_args(smoke_parser)
    smoke_parser.add_argument("--prompt", help="Prompt text")
    smoke_parser.add_argument("--prompt-file", help="Path to prompt text file")
    smoke_parser.add_argument("--max-tokens", type=int, default=512, help="Max output tokens for supported clients")
    smoke_parser.add_argument("--json", action="store_true", help="Print JSON result")
    smoke_parser.add_argument("--dry-run", action="store_true", help="Resolve config and prompt without calling API")

    return parser


def _command_list_apis(args: argparse.Namespace) -> int:
    profiles = list_runtime_profiles(
        config_file=getattr(args, "llm_config", None),
        env_file=getattr(args, "llm_env_file", None),
    )
    for name, profile in sorted(profiles.items()):
        provider = profile.get("provider", "unknown")
        family = profile.get("family", "")
        protocol = profile.get("protocol", "openai_chat")
        model = profile.get("default_model") or profile.get("model") or ""
        print(f"{name}\tprovider={provider}\tfamily={family}\tprotocol={protocol}\tmodel={model}")
    return 0


async def _command_smoke_async(args: argparse.Namespace) -> int:
    prompt = _read_prompt(args.prompt, args.prompt_file)
    config = resolve_llm_cli_config(args)
    if args.dry_run:
        payload = {
            "api_name": config.api_name,
            "provider": config.provider,
            "family": config.family,
            "protocol": config.protocol,
            "api_url": config.api_url,
            "model": config.default_model,
            "temperature": config.temperature,
            "max_retries": config.max_retries,
            "retry_base_delay": config.retry_base_delay,
            "log_file": config.log_file,
            "max_concurrent": config.max_concurrent,
            "prompt_length": len(prompt),
            "has_api_key": bool(config.api_key),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    client = create_client_from_cli(args)

    kwargs = {"max_tokens": args.max_tokens}

    response = await client.get_completion(prompt=prompt, metadata={"task": "llm_cli_smoke"}, **kwargs)
    payload = {
        "success": response.success,
        "api_name": config.api_name,
        "provider": config.provider,
        "protocol": config.protocol,
        "model": config.default_model,
        "call_id": response.call_id,
        "duration_ms": response.duration_ms,
        "error": response.error,
        "content": response.content,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("=" * 60)
        print(f"profile: {config.api_name}")
        print(f"provider/protocol/model: {config.provider}/{config.protocol}/{config.default_model}")
        print(f"success: {response.success}")
        print(f"call_id: {response.call_id}")
        if response.error:
            print(f"error: {response.error}")
        print("-" * 60)
        print(response.content)
    return 0 if response.success else 1


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "list-apis":
        return _command_list_apis(args)
    if args.command == "smoke":
        return asyncio.run(_command_smoke_async(args))
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
