#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Runtime profile configuration for LLMClient_v2.

The main unit is a profile: one callable endpoint configuration. A profile
keeps provider, protocol, API key, base URL, and default model separate so the
same model can be used through multiple providers or protocols.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml
from dotenv import load_dotenv


PACKAGE_DIR = Path(__file__).parent.resolve()
DEFAULT_CONFIG_FILE = PACKAGE_DIR / "llm_apis.yaml"


@dataclass(frozen=True)
class LLMRuntimeConfig:
    """Resolved profile configuration used by concrete clients."""

    api_name: str
    provider: str
    protocol: str
    api_url: str
    api_key: str
    default_model: str
    family: Optional[str] = None
    temperature: float = 0.7
    max_retries: int = 5
    retry_base_delay: float = 1.0
    log_file: Optional[str] = None
    max_concurrent: Optional[int] = None
    model_list: Optional[list[str]] = None
    source: str = "runtime"

    def to_legacy_dict(self) -> Dict[str, Any]:
        """Return a dict compatible with the older llm_config.py shape."""
        data: Dict[str, Any] = {
            "provider": self.provider,
            "protocol": self.protocol,
            "api_url": self.api_url,
            "api_key": self.api_key,
            "default_model": self.default_model,
            "temperature": self.temperature,
            "max_retries": self.max_retries,
            "retry_base_delay": self.retry_base_delay,
        }
        if self.family:
            data["family"] = self.family
        if self.log_file:
            data["log_file"] = self.log_file
        if self.max_concurrent is not None:
            data["max_concurrent"] = self.max_concurrent
        if self.model_list:
            data["model_list"] = list(self.model_list)
        return data


def _load_yaml_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"LLM config file must contain a mapping: {path}")
    return loaded


def _load_env_file(env_file: Optional[str]) -> None:
    if env_file:
        load_dotenv(Path(env_file).expanduser().resolve(), override=False)
        return

    # dotenv defaults to cwd traversal. Also try the project root near this package
    # so scripts launched from subdirectories still see the repository .env.
    load_dotenv(override=False)
    project_env = PACKAGE_DIR.parent / ".env"
    if project_env.exists():
        load_dotenv(project_env, override=False)


def _merged_config(config_file: Optional[str]) -> Dict[str, Any]:
    config_path = Path(config_file).expanduser().resolve() if config_file else DEFAULT_CONFIG_FILE
    data = _load_yaml_file(config_path)

    apis = data.setdefault("apis", {})
    if not isinstance(apis, dict):
        raise ValueError("'apis' in LLM config file must be a mapping")

    data.setdefault("default_api", os.getenv("LLM_DEFAULT_API"))
    return data


def _get_profile(data: Dict[str, Any], api_name: Optional[str]) -> tuple[str, Dict[str, Any]]:
    selected = api_name or os.getenv("LLM_DEFAULT_API") or data.get("default_api")
    if not selected:
        raise ValueError("No LLM profile selected. Set --api, LLM_DEFAULT_API, or default_api.")

    apis = data.get("apis") or {}
    if selected not in apis:
        raise ValueError(f"未知 LLM profile: {selected}，可用 profile: {sorted(apis.keys())}")
    profile = apis[selected]
    if not isinstance(profile, dict):
        raise ValueError(f"LLM profile must be a mapping: {selected}")
    return str(selected), dict(profile)


def _env_first(names: Iterable[Optional[str]]) -> Optional[str]:
    for name in names:
        if name and os.getenv(name):
            return os.getenv(name)
    return None


def _coerce_float(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number, got {value!r}") from exc


def _coerce_int(value: Any, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer, got {value!r}") from exc


def resolve_runtime_config(
    api_name: Optional[str] = None,
    *,
    config_file: Optional[str] = None,
    env_file: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_retries: Optional[int] = None,
    retry_base_delay: Optional[float] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    api_key_env: Optional[str] = None,
    log_file: Optional[str] = None,
    max_concurrent: Optional[int] = None,
    protocol: Optional[str] = None,
    provider: Optional[str] = None,
    family: Optional[str] = None,
) -> LLMRuntimeConfig:
    """
    Resolve a runtime profile.

    Precedence:
    CLI/explicit overrides > environment/.env > YAML config > legacy llm_config.py.
    """

    _load_env_file(env_file)
    data = _merged_config(config_file)
    resolved_name, profile = _get_profile(data, api_name)

    env_prefix = str(profile.get("env_prefix") or resolved_name).upper().replace("-", "_")
    configured_key_env = api_key_env or profile.get("api_key_env")

    resolved_api_key = (
        api_key
        or _env_first([configured_key_env, f"{env_prefix}_API_KEY"])
        or profile.get("api_key")
    )

    resolved_base_url = (
        base_url
        or _env_first([profile.get("base_url_env"), f"{env_prefix}_BASE_URL"])
        or profile.get("base_url")
        or profile.get("api_url")
    )

    resolved_model = (
        model
        or _env_first([profile.get("model_env"), f"{env_prefix}_MODEL"])
        or profile.get("default_model")
        or profile.get("model")
    )

    resolved_temperature = (
        temperature
        if temperature is not None
        else _env_first([profile.get("temperature_env"), f"{env_prefix}_TEMPERATURE"])
        or profile.get("temperature", 0.7)
    )
    resolved_max_retries = (
        max_retries
        if max_retries is not None
        else _env_first([profile.get("max_retries_env"), f"{env_prefix}_MAX_RETRIES"])
        or profile.get("max_retries", 5)
    )
    resolved_retry_base_delay = (
        retry_base_delay
        if retry_base_delay is not None
        else _env_first([profile.get("retry_base_delay_env"), f"{env_prefix}_RETRY_BASE_DELAY"])
        or profile.get("retry_base_delay", 1)
    )
    resolved_log_file = (
        log_file
        or _env_first([profile.get("log_file_env"), f"{env_prefix}_LOG_FILE", "LLM_LOG_FILE"])
        or profile.get("log_file")
    )
    resolved_max_concurrent = (
        max_concurrent
        if max_concurrent is not None
        else _env_first([profile.get("max_concurrent_env"), f"{env_prefix}_MAX_CONCURRENT", "LLM_MAX_CONCURRENT"])
        or profile.get("max_concurrent")
    )

    if not resolved_api_key:
        raise ValueError(
            f"LLM profile '{resolved_name}' 缺少 api_key。请通过 --api-key、"
            f"{configured_key_env or f'{env_prefix}_API_KEY'} 或配置文件提供。"
        )
    if not resolved_base_url:
        raise ValueError(f"LLM profile '{resolved_name}' 缺少 base_url/api_url")
    if not resolved_model:
        raise ValueError(f"LLM profile '{resolved_name}' 缺少 default_model/model")

    model_list = profile.get("model_list")
    if model_list is not None:
        if not isinstance(model_list, list) or not model_list:
            raise ValueError(f"LLM profile '{resolved_name}' 的 model_list 必须是非空 list")
        if resolved_model not in model_list:
            raise ValueError(
                f"LLM profile '{resolved_name}' 的 model='{resolved_model}' "
                f"不在 model_list 中: {model_list}"
            )

    return LLMRuntimeConfig(
        api_name=resolved_name,
        provider=str(provider or profile.get("provider", "unknown")),
        protocol=str(protocol or profile.get("protocol", "openai_chat")),
        api_url=str(resolved_base_url),
        api_key=str(resolved_api_key),
        default_model=str(resolved_model),
        family=str(family or profile.get("family")) if (family or profile.get("family")) else None,
        temperature=_coerce_float(resolved_temperature, "temperature"),
        max_retries=_coerce_int(resolved_max_retries, "max_retries"),
        retry_base_delay=_coerce_float(resolved_retry_base_delay, "retry_base_delay"),
        log_file=str(resolved_log_file) if resolved_log_file else None,
        max_concurrent=_coerce_int(resolved_max_concurrent, "max_concurrent")
        if resolved_max_concurrent is not None
        else None,
        model_list=list(model_list) if model_list else None,
        source="runtime",
    )


def overlay_runtime_config(config: LLMRuntimeConfig, **overrides: Any) -> LLMRuntimeConfig:
    """Apply explicit constructor overrides to an existing runtime config."""
    clean = {key: value for key, value in overrides.items() if value is not None}
    return replace(config, **clean)


def list_runtime_profiles(config_file: Optional[str] = None, env_file: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Return profile metadata without resolving API keys."""
    _load_env_file(env_file)
    data = _merged_config(config_file)
    return dict(data.get("apis") or {})


def get_default_api(config_file: Optional[str] = None) -> str:
    """Return the default API profile name from YAML or environment."""
    data = _merged_config(config_file)
    name = data.get("default_api")
    if not name:
        apis = data.get("apis") or {}
        if apis:
            name = next(iter(apis))
        else:
            raise ValueError("No profiles configured and no default_api set")
    return str(name)

