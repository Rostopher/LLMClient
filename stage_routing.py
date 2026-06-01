#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stage Routing & Pipeline 模块

两层设计：
  1. 底层路由表 — 从 llm_apis.yaml 的 stage_routing 节加载 stage→API 映射
  2. LLMPipeline — 高层抽象，将多个 stage 组织为 pipeline，通过
     context manager 自动注入正确的 LLMClient 并管理 token 追踪

使用示例::

    pipeline = LLMPipeline(
        name="paper_processing",
        stages={
            "metadata":  StageConfig("yunwu_gemini_25_pro_chat", "gemini-2.5-flash", temperature=0),
            "structure": StageConfig("yunwu_gemini_25_pro_chat", "gemini-2.5-flash", temperature=0),
            "summary":   StageConfig("yunwu_gemini_25_pro_chat", "gemini-3-pro-preview", temperature=0.7),
        },
        token_file="logs/paper_tokens.jsonl",
    )

    async with pipeline.stage("metadata") as client:
        meta = await client.get_json_completion(prompt)

    async with pipeline.stage("summary") as client:
        report = await client.get_completion(prompt)

    pipeline.print_summary()
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional, TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from .llm_client import LLMClient


PACKAGE_DIR = Path(__file__).parent.resolve()
DEFAULT_CONFIG_FILE = PACKAGE_DIR / "llm_apis.yaml"


# =========================================================================
# StageConfig — 单个 stage 的 LLM 配置
# =========================================================================

@dataclass(frozen=True)
class StageConfig:
    """一个 stage 的 LLM 路由配置"""
    api_name: str
    model: Optional[str] = None
    temperature: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "api_name": self.api_name,
            "model": self.model,
            "temperature": self.temperature,
        }


# =========================================================================
# 底层路由表（向后兼容）
# =========================================================================

_STAGE_ROUTING: Dict[str, StageConfig] = {}
_loaded = False


def _ensure_loaded(config_file: Optional[str] = None) -> None:
    global _loaded
    if _loaded:
        return
    load_stage_routing(config_file)


def load_stage_routing(config_file: Optional[str] = None) -> None:
    """从 YAML 配置文件加载 stage routing 映射"""
    global _STAGE_ROUTING, _loaded

    path = Path(config_file).resolve() if config_file else DEFAULT_CONFIG_FILE
    if not path.exists():
        _loaded = True
        return

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    raw = data.get("stage_routing") or {}
    if not isinstance(raw, dict):
        _loaded = True
        return

    _STAGE_ROUTING.clear()
    for stage_name, cfg in raw.items():
        if not isinstance(cfg, dict):
            continue
        _STAGE_ROUTING[stage_name] = StageConfig(
            api_name=cfg.get("api") or cfg.get("api_name") or "",
            model=cfg.get("model"),
            temperature=cfg.get("temperature"),
        )

    _loaded = True


def get_route_for_stage(
    stage_name: str,
    default: Optional[str] = None,
) -> Dict[str, Any]:
    """获取指定 stage 的路由配置（返回 dict，向后兼容）"""
    _ensure_loaded()
    cfg = _STAGE_ROUTING.get(stage_name)
    if cfg:
        return cfg.to_dict()
    return {"api_name": default, "model": None, "temperature": None}


def get_stage_config(stage_name: str) -> Optional[StageConfig]:
    """获取指定 stage 的 StageConfig 对象"""
    _ensure_loaded()
    return _STAGE_ROUTING.get(stage_name)


def get_api_for_stage(
    stage_name: str,
    default: Optional[str] = None,
) -> Optional[str]:
    """获取指定 stage 对应的 api_name"""
    return get_route_for_stage(stage_name, default=default).get("api_name") or default


def set_stage_route(
    stage_name: str,
    api_name: str,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
) -> None:
    """运行时动态设置/覆盖某个 stage 的路由"""
    _ensure_loaded()
    _STAGE_ROUTING[stage_name] = StageConfig(
        api_name=api_name, model=model, temperature=temperature,
    )


def list_stage_routing() -> Dict[str, Dict[str, Any]]:
    """返回当前所有 stage routing 配置的副本"""
    _ensure_loaded()
    return {k: v.to_dict() for k, v in _STAGE_ROUTING.items()}


def reload_stage_routing(config_file: Optional[str] = None) -> None:
    """强制重新加载 stage routing 配置"""
    global _loaded
    _loaded = False
    load_stage_routing(config_file)


# =========================================================================
# LLMPipeline — 高层 Pipeline 抽象
# =========================================================================

@dataclass
class _StageRecord:
    """Pipeline 内部用于记录单个 stage 执行统计"""
    name: str
    calls: int = 0
    tokens: int = 0
    cost: float = 0.0


class LLMPipeline:
    """
    多阶段 LLM 处理管线

    将多个 stage 的 LLM 配置集中定义，通过 context manager 自动：
      - 创建正确路由的 LLMClient
      - 绑定共享的 token 日志文件
      - 退出时汇总 token 使用和成本
    """

    def __init__(
        self,
        name: str,
        stages: Optional[Dict[str, StageConfig]] = None,
        token_file: Optional[str] = None,
        default_api: Optional[str] = None,
    ):
        """
        Args:
            name: pipeline 名称（用于日志标识）
            stages: stage 名称 → StageConfig 的映射。如果为 None，
                    则回退到 YAML stage_routing 全局配置。
            token_file: 整条 pipeline 共享的 token 日志文件路径
            default_api: stages 中未定义的 stage 回退到此 API
        """
        self.name = name
        self._stages: Dict[str, StageConfig] = dict(stages) if stages else {}
        self.token_file = token_file
        self.default_api = default_api
        self._records: Dict[str, _StageRecord] = {}

    def _resolve_stage(self, stage_name: str) -> StageConfig:
        """解析 stage 配置：优先 pipeline 内定义 → 全局 YAML → default_api"""
        if stage_name in self._stages:
            return self._stages[stage_name]

        global_cfg = get_stage_config(stage_name)
        if global_cfg:
            return global_cfg

        if self.default_api:
            return StageConfig(api_name=self.default_api)

        raise ValueError(
            f"Pipeline '{self.name}' 中未定义 stage '{stage_name}'，"
            f"且全局 stage_routing 中也找不到。"
            f"已定义的 stages: {sorted(self._stages.keys())}"
        )

    @asynccontextmanager
    async def stage(
        self,
        stage_name: str,
        **client_kwargs: Any,
    ) -> AsyncGenerator["LLMClient", None]:
        """
        进入某个 stage 并获取已配置好的 LLMClient。

        用法::

            async with pipeline.stage("metadata") as client:
                result = await client.get_json_completion(prompt)

        自动处理：
          - 从 stage 配置创建正确路由的 client
          - 设置全局 token file（如 pipeline 指定了 token_file）
          - 退出时汇总该 stage 的 token 使用
        """
        from .llm_client import LLMClient
        from .token_usage_tracker import (
            set_global_task_token_file,
            get_global_task_token_file,
            clear_global_task_token_file,
        )

        cfg = self._resolve_stage(stage_name)

        prev_token_file = get_global_task_token_file()
        if self.token_file:
            set_global_task_token_file(self.token_file)

        client = LLMClient(
            api_name=cfg.api_name,
            model=cfg.model,
            temperature=cfg.temperature,
            log_file=self.token_file,
            **client_kwargs,
        )

        try:
            yield client
        finally:
            if client.tracker:
                summary = client.tracker.get_session_summary()
                record = self._records.setdefault(
                    stage_name, _StageRecord(name=stage_name)
                )
                record.calls += summary.get("total_calls", 0)
                record.tokens += summary.get("total_usage", {}).get("total_tokens", 0)
                record.cost += summary.get("total_cost", 0.0)

            if self.token_file:
                if prev_token_file:
                    set_global_task_token_file(prev_token_file)
                else:
                    clear_global_task_token_file()

    def get_summary(self) -> Dict[str, Any]:
        """获取 pipeline 执行统计"""
        stages_summary = {}
        total_calls = 0
        total_tokens = 0
        total_cost = 0.0

        for name, record in self._records.items():
            stages_summary[name] = {
                "calls": record.calls,
                "tokens": record.tokens,
                "cost": record.cost,
            }
            total_calls += record.calls
            total_tokens += record.tokens
            total_cost += record.cost

        return {
            "pipeline": self.name,
            "total_calls": total_calls,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "stages": stages_summary,
        }

    def print_summary(self) -> None:
        """打印 pipeline 执行统计"""
        s = self.get_summary()
        print(f"\n{'='*50}")
        print(f"Pipeline: {s['pipeline']}")
        print(f"{'='*50}")
        print(f"总调用: {s['total_calls']}  总Token: {s['total_tokens']}  总成本: ${s['total_cost']:.6f}")

        if s["stages"]:
            print(f"\n{'Stage':<30} {'Calls':>6} {'Tokens':>10} {'Cost':>12}")
            print("-" * 60)
            for name, data in s["stages"].items():
                print(f"{name:<30} {data['calls']:>6} {data['tokens']:>10} ${data['cost']:>11.6f}")

        print(f"{'='*50}\n")

    @classmethod
    def from_yaml(
        cls,
        name: str,
        config_file: Optional[str] = None,
        token_file: Optional[str] = None,
        stage_names: Optional[list[str]] = None,
        default_api: Optional[str] = None,
    ) -> "LLMPipeline":
        """
        从 YAML 全局 stage_routing 创建 pipeline

        Args:
            name: pipeline 名称
            config_file: YAML 配置文件路径
            token_file: 共享 token 日志
            stage_names: 只包含指定的 stage（None 则全部包含）
            default_api: 回退 API
        """
        _ensure_loaded(config_file)

        if stage_names:
            stages = {
                k: v for k, v in _STAGE_ROUTING.items()
                if k in stage_names
            }
        else:
            stages = dict(_STAGE_ROUTING)

        return cls(
            name=name,
            stages=stages,
            token_file=token_file,
            default_api=default_api,
        )
