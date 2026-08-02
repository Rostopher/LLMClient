#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Minimal tests for runtime profile configuration."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from . import llm_runtime_config as runtime_config_module
from .llm_runtime_config import resolve_runtime_config


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_profile_env_resolution() -> None:
    os.environ.pop("YUNWU_GPT_API_KEY", None)
    os.environ.pop("YUNWU_API_KEY", None)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        config_path = tmp_path / "llm.yaml"
        env_path = tmp_path / ".env"

        _write(
            config_path,
            """
default_api: yunwu_gpt
apis:
  yunwu_gpt:
    provider: yunwu
    family: gpt
    protocol: openai_chat
    base_url: https://yunwu.ai/v1
    api_key_env: YUNWU_GPT_API_KEY
    default_model: gpt-5.5
    temperature: 0.3
""".strip(),
        )
        _write(env_path, "YUNWU_GPT_API_KEY=profile-key\nYUNWU_API_KEY=wrong-key\n")

        cfg = resolve_runtime_config(config_file=str(config_path), env_file=str(env_path))
        assert cfg.api_name == "yunwu_gpt"
        assert cfg.provider == "yunwu"
        assert cfg.family == "gpt"
        assert cfg.protocol == "openai_chat"
        assert cfg.api_key == "profile-key"
        assert cfg.default_model == "gpt-5.5"
        assert cfg.temperature == 0.3


def test_no_generic_provider_key_fallback() -> None:
    os.environ.pop("YUNWU_GEMINI_API_KEY", None)
    os.environ.pop("YUNWU_API_KEY", None)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        config_path = tmp_path / "llm.yaml"
        env_path = tmp_path / ".env"

        _write(
            config_path,
            """
default_api: yunwu_gemini
apis:
  yunwu_gemini:
    provider: yunwu
    family: gemini
    protocol: openai_chat
    base_url: https://yunwu.ai/v1
    api_key_env: YUNWU_GEMINI_API_KEY
    default_model: gemini-2.5-pro
""".strip(),
        )
        _write(env_path, "YUNWU_API_KEY=generic-key\n")

        try:
            resolve_runtime_config(config_file=str(config_path), env_file=str(env_path))
        except ValueError as exc:
            assert "YUNWU_GEMINI_API_KEY" in str(exc)
        else:
            raise AssertionError("generic provider key must not be used as fallback")


def test_explicit_overrides_win() -> None:
    os.environ.pop("DEEPSEEK_API_KEY", None)
    os.environ.pop("DEEPSEEK_MODEL", None)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        config_path = tmp_path / "llm.yaml"
        env_path = tmp_path / ".env"

        _write(
            config_path,
            """
default_api: deepseek
apis:
  deepseek:
    provider: deepseek_official
    family: deepseek
    protocol: openai_chat
    base_url: https://api.deepseek.com/v1
    api_key_env: DEEPSEEK_API_KEY
    default_model: deepseek-chat
    temperature: 0.7
""".strip(),
        )
        _write(env_path, "DEEPSEEK_API_KEY=env-key\nDEEPSEEK_MODEL=env-model\n")

        cfg = resolve_runtime_config(
            config_file=str(config_path),
            env_file=str(env_path),
            api_key="cli-key",
            model="cli-model",
            temperature=0.1,
            max_retries=2,
        )
        assert cfg.api_key == "cli-key"
        assert cfg.default_model == "cli-model"
        assert cfg.temperature == 0.1
        assert cfg.max_retries == 2


def test_default_loads_package_env() -> None:
    env_name = "LLMCLIENT_PACKAGE_ENV_TEST_KEY"
    os.environ.pop(env_name, None)
    original_package_dir = runtime_config_module.PACKAGE_DIR
    try:
        with tempfile.TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "LLMClient"
            package_dir.mkdir()
            _write(package_dir / ".env", f"{env_name}=package-key\n")
            runtime_config_module.PACKAGE_DIR = package_dir
            runtime_config_module._load_env_file(None)
            assert os.getenv(env_name) == "package-key"
    finally:
        runtime_config_module.PACKAGE_DIR = original_package_dir
        os.environ.pop(env_name, None)


if __name__ == "__main__":
    test_profile_env_resolution()
    test_no_generic_provider_key_fallback()
    test_explicit_overrides_win()
    test_default_loads_package_env()
    print("runtime config tests passed")
