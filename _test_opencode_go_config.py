#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""OpenCode Go profile 与论文处理阶段路由的最小回归测试。"""

from __future__ import annotations

from pathlib import Path

import yaml

from .llm_runtime_config import resolve_runtime_config


PACKAGE_DIR = Path(__file__).parent.resolve()
CONFIG_FILE = PACKAGE_DIR / "llm_apis.yaml"


def test_opencode_go_profiles() -> None:
    expected = {
        "opencode_go_deepseek_pro": "deepseek-v4-pro",
        "opencode_go_deepseek_flash": "deepseek-v4-flash",
    }
    for profile_name, model in expected.items():
        config = resolve_runtime_config(
            api_name=profile_name,
            config_file=str(CONFIG_FILE),
            api_key="test-key",
        )
        assert config.provider == "opencode_go"
        assert config.family == "deepseek"
        assert config.protocol == "openai_chat"
        assert config.api_url == "https://opencode.ai/zen/go/v1"
        assert config.default_model == model


def test_paper_stages_route_to_opencode_go() -> None:
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    routing = config["stage_routing"]
    pro_stages = {
        "summary_report_generation",
        "deep_report_generation",
        "chat_interaction",
    }
    flash_stages = {
        "metadata_extraction",
        "attribute_tree",
        "section_detection",
        "paper_structure",
        "translation",
    }

    for stage in pro_stages:
        assert routing[stage]["api"] == "opencode_go_deepseek_pro"
        assert routing[stage]["model"] == "deepseek-v4-pro"
    for stage in flash_stages:
        assert routing[stage]["api"] == "opencode_go_deepseek_flash"
        assert routing[stage]["model"] == "deepseek-v4-flash"


if __name__ == "__main__":
    test_opencode_go_profiles()
    test_paper_stages_route_to_opencode_go()
    print("OpenCode Go config tests passed")
