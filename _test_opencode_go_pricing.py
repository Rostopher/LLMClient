#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""OpenCode Go 的供应商级 DeepSeek V4 美元定价回归。"""

from __future__ import annotations

from math import isclose

from .token_usage_tracker import ModelPricingConfig, TokenUsage, TokenUsageTracker


def test_opencode_go_deepseek_pricing() -> None:
    pricing = ModelPricingConfig()
    usage = TokenUsage(
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        total_tokens=2_000_000,
    )

    pro = pricing.calculate_cost(usage, "deepseek-v4-pro", provider="opencode_go")
    assert pro["currency"] == "USD"
    assert pro["input_cost"] == 0.435
    assert pro["output_cost"] == 0.87
    assert isclose(pro["standard_cost"], 1.305)

    flash = pricing.calculate_cost(usage, "deepseek-v4-flash", provider="opencode_go")
    assert flash["currency"] == "USD"
    assert flash["input_cost"] == 0.14
    assert flash["output_cost"] == 0.28
    assert isclose(flash["standard_cost"], 0.42)


def test_opencode_go_cached_read_pricing() -> None:
    pricing = ModelPricingConfig()
    usage = TokenUsage(
        prompt_tokens=1_000_000,
        completion_tokens=0,
        total_tokens=1_000_000,
        prompt_cache_hit_tokens=1_000_000,
        prompt_cache_miss_tokens=0,
    )

    pro = pricing.calculate_cost(usage, "deepseek-v4-pro", provider="opencode_go")
    assert pro["currency"] == "USD"
    assert pro["standard_cost"] == 0.003625

    flash = pricing.calculate_cost(usage, "deepseek-v4-flash", provider="opencode_go")
    assert flash["currency"] == "USD"
    assert flash["standard_cost"] == 0.0028


def test_openai_cached_tokens_usage_shape() -> None:
    response = {
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_tokens_details": {"cached_tokens": 75},
        }
    }
    usage = TokenUsageTracker.extract_usage_from_response(response)
    assert usage.prompt_cache_hit_tokens == 75
    assert usage.prompt_cache_miss_tokens == 25


if __name__ == "__main__":
    test_opencode_go_deepseek_pricing()
    test_opencode_go_cached_read_pricing()
    test_openai_cached_tokens_usage_shape()
    print("OpenCode Go pricing tests passed")
