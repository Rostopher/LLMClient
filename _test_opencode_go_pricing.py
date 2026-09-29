#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""OpenCode Go 复用 DeepSeek 官方模型定价的回归。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from math import isclose

from .token_usage_tracker import (
    ModelPricingConfig,
    TokenUsage,
    TokenUsageTracker,
    is_deepseek_peak_hour,
)

_BEIJING = timezone(timedelta(hours=8))
# 2026-09-29 周二：22:00 为空闲时段，10:00 为高峰时段；2026-09-27 为周日
_OFF_PEAK = datetime(2026, 9, 29, 22, 0, tzinfo=_BEIJING)
_PEAK = datetime(2026, 9, 29, 10, 0, tzinfo=_BEIJING)
_WEEKEND = datetime(2026, 9, 27, 10, 0, tzinfo=_BEIJING)


def test_opencode_go_reuses_official_deepseek_pricing() -> None:
    pricing = ModelPricingConfig()
    usage = TokenUsage(
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        total_tokens=2_000_000,
    )

    pro = pricing.calculate_cost(usage, "deepseek-v4-pro", provider="opencode_go", at=_OFF_PEAK)
    assert pro["currency"] == "CNY"
    assert pro["input_cost"] == 4.5
    assert pro["output_cost"] == 13.5
    assert isclose(pro["standard_cost"], 18.0)

    flash = pricing.calculate_cost(usage, "deepseek-v4-flash", provider="opencode_go", at=_OFF_PEAK)
    assert flash["currency"] == "CNY"
    assert flash["input_cost"] == 1.0
    assert flash["output_cost"] == 4.0
    assert isclose(flash["standard_cost"], 5.0)


def test_opencode_go_cached_read_pricing() -> None:
    pricing = ModelPricingConfig()
    usage = TokenUsage(
        prompt_tokens=1_000_000,
        completion_tokens=0,
        total_tokens=1_000_000,
        prompt_cache_hit_tokens=1_000_000,
        prompt_cache_miss_tokens=0,
    )

    pro = pricing.calculate_cost(usage, "deepseek-v4-pro", provider="opencode_go", at=_OFF_PEAK)
    assert pro["currency"] == "CNY"
    assert pro["standard_cost"] == 0.15

    flash = pricing.calculate_cost(usage, "deepseek-v4-flash", provider="opencode_go", at=_OFF_PEAK)
    assert flash["currency"] == "CNY"
    assert flash["standard_cost"] == 0.02


def test_deepseek_flash_official_name_pricing() -> None:
    """deepseek-flash（V4.1 Flash 官方 API 名）必须与 v4-flash 同价，不落 default。"""
    pricing = ModelPricingConfig()
    usage = TokenUsage(
        prompt_tokens=120_000,
        completion_tokens=2_000,
        total_tokens=122_000,
        prompt_cache_hit_tokens=118_000,
        prompt_cache_miss_tokens=2_000,
    )

    flash = pricing.calculate_cost(usage, "deepseek-flash", at=_OFF_PEAK)
    assert flash["currency"] == "CNY"
    expected = (118_000 * 0.02 + 2_000 * 1.0 + 2_000 * 4.0) / 1_000_000
    assert isclose(flash["standard_cost"], expected)

    legacy = pricing.calculate_cost(usage, "deepseek-v4-flash", at=_OFF_PEAK)
    vision = pricing.calculate_cost(usage, "deepseek-v4-flash-vision-exp", at=_OFF_PEAK)
    assert isclose(legacy["standard_cost"], flash["standard_cost"])
    assert isclose(vision["standard_cost"], flash["standard_cost"])


def test_deepseek_peak_hour_pricing() -> None:
    """工作日北京 9-12 / 14-18 高峰三项单价 ×2；周末与晚间按空闲价。"""
    assert is_deepseek_peak_hour(_PEAK)
    assert is_deepseek_peak_hour(datetime(2026, 9, 29, 15, 0, tzinfo=_BEIJING))
    assert not is_deepseek_peak_hour(_OFF_PEAK)
    assert not is_deepseek_peak_hour(_WEEKEND)

    pricing = ModelPricingConfig()
    usage = TokenUsage(
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        total_tokens=2_000_000,
        prompt_cache_hit_tokens=500_000,
        prompt_cache_miss_tokens=500_000,
    )

    off_peak = pricing.calculate_cost(usage, "deepseek-flash", at=_OFF_PEAK)
    assert off_peak["peak_pricing_applied"] is False
    # 0.5M×0.02 + 0.5M×1.0 + 1M×4.0 = 4.51
    assert isclose(off_peak["standard_cost"], 4.51)

    peak = pricing.calculate_cost(usage, "deepseek-flash", at=_PEAK)
    assert peak["peak_pricing_applied"] is True
    assert isclose(peak["standard_cost"], off_peak["standard_cost"] * 2)

    weekend = pricing.calculate_cost(usage, "deepseek-flash", at=_WEEKEND)
    assert weekend["peak_pricing_applied"] is False
    assert isclose(weekend["standard_cost"], off_peak["standard_cost"])


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
    test_opencode_go_reuses_official_deepseek_pricing()
    test_opencode_go_cached_read_pricing()
    test_deepseek_flash_official_name_pricing()
    test_deepseek_peak_hour_pricing()
    test_openai_cached_tokens_usage_shape()
    print("OpenCode Go pricing tests passed")
