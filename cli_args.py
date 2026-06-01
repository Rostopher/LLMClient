#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Reusable argparse helpers for scripts that use LLMClient_v2."""

from __future__ import annotations

import argparse
from typing import Optional

from .llm_runtime_config import LLMRuntimeConfig, resolve_runtime_config


def add_llm_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add common LLM runtime options to an existing parser."""
    group = parser.add_argument_group("LLM runtime")
    group.add_argument("--api", "--api-name", dest="llm_api", help="LLM profile name")
    group.add_argument("--llm-config", dest="llm_config", help="Path to YAML LLM profile config")
    group.add_argument("--env-file", dest="llm_env_file", help="Path to .env file")
    group.add_argument("--model", dest="llm_model", help="Override model name")
    group.add_argument("--temperature", dest="llm_temperature", type=float, help="Override temperature")
    group.add_argument("--max-retries", dest="llm_max_retries", type=int, help="Override max retries")
    group.add_argument(
        "--retry-base-delay",
        dest="llm_retry_base_delay",
        type=float,
        help="Override exponential retry base delay seconds",
    )
    group.add_argument("--base-url", dest="llm_base_url", help="Override API base URL")
    group.add_argument("--api-key", dest="llm_api_key", help="Override API key")
    group.add_argument("--api-key-env", dest="llm_api_key_env", help="Environment variable name for API key")
    group.add_argument("--log-file", dest="llm_log_file", help="LLM JSONL activity log path")
    group.add_argument("--max-concurrent", dest="llm_max_concurrent", type=int, help="Batch concurrency")
    group.add_argument(
        "--protocol",
        dest="llm_protocol",
        choices=["openai_chat", "openai_responses", "anthropic_messages"],
        help="Override wire protocol",
    )
    group.add_argument("--provider", dest="llm_provider", help="Override provider label")
    group.add_argument("--family", dest="llm_family", help="Override model family label")
    group.add_argument("--no-tracking", dest="llm_no_tracking", action="store_true", help="Disable token/log tracking")
    return parser


def resolve_llm_cli_config(args: argparse.Namespace, api_name: Optional[str] = None) -> LLMRuntimeConfig:
    """Resolve LLMRuntimeConfig from argparse namespace."""
    return resolve_runtime_config(
        api_name=api_name or getattr(args, "llm_api", None),
        config_file=getattr(args, "llm_config", None),
        env_file=getattr(args, "llm_env_file", None),
        model=getattr(args, "llm_model", None),
        temperature=getattr(args, "llm_temperature", None),
        max_retries=getattr(args, "llm_max_retries", None),
        retry_base_delay=getattr(args, "llm_retry_base_delay", None),
        base_url=getattr(args, "llm_base_url", None),
        api_key=getattr(args, "llm_api_key", None),
        api_key_env=getattr(args, "llm_api_key_env", None),
        log_file=getattr(args, "llm_log_file", None),
        max_concurrent=getattr(args, "llm_max_concurrent", None),
        protocol=getattr(args, "llm_protocol", None),
        provider=getattr(args, "llm_provider", None),
        family=getattr(args, "llm_family", None),
    )


def create_client_from_cli(args: argparse.Namespace):
    """Create the right client class for the resolved protocol."""
    from .anthropic_client import AnthropicLLMClient
    from .llm_client import LLMClient

    config = resolve_llm_cli_config(args)
    enable_tracking = not bool(getattr(args, "llm_no_tracking", False))
    if config.protocol == "anthropic_messages":
        return AnthropicLLMClient(runtime_config=config, enable_tracking=enable_tracking)
    return LLMClient(runtime_config=config, enable_tracking=enable_tracking)

