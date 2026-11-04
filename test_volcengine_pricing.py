"""豆包计量回归：不访问供应商，不消耗额度。"""
from decimal import Decimal, ROUND_CEILING
from types import SimpleNamespace

import pytest

from .token_usage_tracker import ModelPricingConfig, TokenUsageTracker


MODEL = "doubao-seed-2-1-lite-260915"


def extract(prompt=1000, completion=100, **details):
    return TokenUsageTracker.extract_usage_from_response({
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                  "total_tokens": prompt + completion,
                  "prompt_tokens_details": {"cached_tokens": 0, **details}},
    })


def test_preserve_audio_and_original_usage():
    usage = extract(audio_tokens=200, cached_tokens=100, audio_cached_tokens=50)
    assert usage.audio_tokens == 200
    assert usage.audio_cached_tokens == 50
    assert usage.raw_usage["prompt_tokens_details"]["audio_tokens"] == 200
    combined = usage + usage
    assert combined.audio_tokens == 400
    assert combined.audio_cached_tokens == 100
    assert combined.raw_usage is None  # 聚合不能冒充单次供应商原始响应


def test_mixed_audio_cache_is_not_counted_twice():
    usage = extract(audio_tokens=200, cached_tokens=100, audio_cached_tokens=50)
    cost = ModelPricingConfig().calculate_cost(
        usage, MODEL, provider="volcengine", input_modality="audio")
    # 750文本未命中 + 150音频未命中 + 50文本缓存 + 50音频缓存 + 100输出
    assert cost["currency"] == "CNY"
    assert Decimal(str(cost["primary_cost"])) == Decimal("0.002798")
    assert Decimal(cost["primary_cost_exact"]) == Decimal("0.002798")
    assert cost["standard_cost_exact"] == cost["primary_cost_exact"]
    assert cost["audio_input_cost"] == 0.00192
    assert cost["price_version"] == "volc-public-cn-2026-11-04"


def test_plain_text_and_zero_usage():
    pricing = ModelPricingConfig()
    cost = pricing.calculate_cost(extract(1_000_000, 1_000_000), MODEL, provider="volcengine")
    assert cost["primary_cost"] == 3.5
    assert pricing.calculate_cost(extract(0, 0), MODEL, provider="volcengine")["primary_cost"] == 0


def test_sdk_optional_defaults_are_not_supplier_fields():
    from openai.types.completion_usage import CompletionUsage
    sdk_usage = CompletionUsage.model_validate({
        "prompt_tokens": 196, "completion_tokens": 12, "total_tokens": 208,
        "prompt_tokens_details": {"cached_tokens": 0},
    })
    usage = TokenUsageTracker.extract_usage_from_response(SimpleNamespace(usage=sdk_usage))
    assert "audio_tokens" not in usage.raw_usage["prompt_tokens_details"]
    assert ModelPricingConfig().calculate_cost(usage, MODEL, provider="volcengine")["primary_cost"] == 0.0001892


def test_implicit_audio_cache_missing_means_not_hit():
    # 官方：仅隐式缓存且命中音频缓存才返回 audio_cached_tokens。
    cost = ModelPricingConfig().calculate_cost(
        extract(audio_tokens=200, cached_tokens=100), MODEL,
        provider="volcengine", input_modality="audio")
    assert cost["primary_cost"] == 0.003246


@pytest.mark.parametrize("source", ["dict", "sdk"])
@pytest.mark.parametrize("partial_usage,modality", [
    ({}, "text"),
    ({"prompt_tokens_details": {}}, "text"),
    ({"prompt_tokens_details": None}, "text"),
    ({"prompt_tokens_details": {"audio_tokens": 200}}, "audio"),
])
def test_missing_supplier_cache_details_are_rejected(source, partial_usage, modality):
    # 直接构造实际响应，不用会自动补 cached_tokens=0 的 extract helper。
    raw = {"prompt_tokens": 1000, "completion_tokens": 0, "total_tokens": 1000,
           **partial_usage}
    if source == "sdk":
        from openai.types.completion_usage import CompletionUsage
        response = SimpleNamespace(usage=CompletionUsage.model_validate(raw))
    else:
        response = {"usage": raw}
    usage = TokenUsageTracker.extract_usage_from_response(response)
    with pytest.raises(ValueError):
        ModelPricingConfig().calculate_cost(
            usage, MODEL, provider="volcengine", input_modality=modality)


@pytest.mark.parametrize("details", [[], [0], "", "invalid", 0, False])
def test_non_dict_supplier_cache_details_are_rejected(details):
    usage = TokenUsageTracker.extract_usage_from_response({"usage": {
        "prompt_tokens": 1000, "completion_tokens": 0, "total_tokens": 1000,
        "prompt_tokens_details": details,
    }})
    with pytest.raises(ValueError, match="prompt_tokens_details"):
        ModelPricingConfig().calculate_cost(usage, MODEL, provider="volcengine")


@pytest.mark.parametrize("cached", [None, -1, 1.5, True, "0"])
def test_invalid_supplier_cached_count_is_rejected(cached):
    with pytest.raises(ValueError, match="cached_tokens"):
        ModelPricingConfig().calculate_cost(
            extract(cached_tokens=cached), MODEL, provider="volcengine")


def test_tracker_accumulates_known_audio_and_recovers_after_reset():
    tracker = TokenUsageTracker()
    known = extract(audio_tokens=200, audio_cached_tokens=0)
    for _ in range(2):
        tracker.log_call_record({"status": "success", "usage": known.__dict__})
    summary = tracker.get_session_summary()["total_usage"]
    assert summary["audio_tokens"] == 400
    assert summary["audio_cached_tokens"] == 0

    # 一条真实缺失记录会让累计值未知，后续已知记录也不能掩盖缺失。
    missing = extract()
    assert missing.audio_tokens is None
    assert missing.audio_cached_tokens is None
    tracker.log_call_record({"status": "success", "usage": missing.__dict__})
    tracker.log_call_record({"status": "success", "usage": known.__dict__})
    summary = tracker.get_session_summary()["total_usage"]
    assert summary["audio_tokens"] is None
    assert summary["audio_cached_tokens"] is None

    tracker.reset_session()
    for _ in range(2):
        tracker.log_call_record({"status": "success", "usage": known.__dict__})
    summary = tracker.get_session_summary()
    assert summary["total_calls"] == 2
    assert summary["total_usage"]["audio_tokens"] == 400
    assert summary["total_usage"]["audio_cached_tokens"] == 0


@pytest.mark.parametrize("details", [
    {"audio_tokens": 1001},
    {"audio_tokens": 200, "cached_tokens": 100, "audio_cached_tokens": 101},
    {"audio_tokens": 200, "cached_tokens": 900, "audio_cached_tokens": 0},
    {"audio_tokens": -1},
    {"audio_tokens": 1.5},
    {"audio_tokens": True},
])
def test_bad_usage_is_rejected(details):
    with pytest.raises(ValueError):
        ModelPricingConfig().calculate_cost(
            extract(**details), MODEL, provider="volcengine", input_modality="audio")


def test_missing_usage_and_missing_audio_are_not_zero_cost():
    pricing = ModelPricingConfig()
    empty = TokenUsageTracker.extract_usage_from_response({})
    with pytest.raises(ValueError):
        pricing.calculate_cost(empty, MODEL, provider="volcengine")
    with pytest.raises(ValueError):
        pricing.calculate_cost(extract(), MODEL, provider="volcengine", input_modality="audio")


def test_explicit_audio_cache_needs_full_breakdown():
    with pytest.raises(ValueError):
        ModelPricingConfig().calculate_cost(
            extract(audio_tokens=200, cached_tokens=100), MODEL,
            provider="volcengine", input_modality="audio", cache_mode="explicit")


def test_unknown_model_or_tier_never_uses_default_usd():
    pricing = ModelPricingConfig()
    with pytest.raises(ValueError):
        pricing.calculate_cost(extract(), "doubao-unknown", provider="volcengine")
    with pytest.raises(ValueError):
        pricing.calculate_cost(extract(), MODEL, provider="volcengine", service_tier="flex")
    # 旧供应商的 default 行为保持兼容。
    assert pricing.calculate_cost(extract(), "legacy-unknown")["currency"] == "USD"


def test_speech_characters_are_not_tokens():
    cost = ModelPricingConfig().calculate_speech_cost({"text_words": 11}, "seed-tts-2.0")
    assert cost["primary_cost"] == 0.0033
    assert cost["currency"] == "CNY"
    assert cost["billed_characters"] == 11


@pytest.mark.parametrize("provider", ["volcengine", "volcengine_ark", "ark", "doubao"])
@pytest.mark.parametrize("prompt,completion,expected", [
    (0, 0, "0"),
    (1, 0, "0.0000008"),
    (196, 12, "0.0001892"),
    (1_000_000, 1_000_000, "3.5"),
])
def test_chat_exact_costs_preserve_float_interface(provider, prompt, completion, expected):
    cost = ModelPricingConfig().calculate_cost(extract(prompt, completion), MODEL, provider=provider)
    for name in ("primary_cost", "standard_cost"):
        assert isinstance(cost[name], float)
        assert cost[name] == float(expected)
        exact = cost[f"{name}_exact"]
        assert isinstance(exact, str)
        assert "e" not in exact.lower()
        assert Decimal(exact) == Decimal(expected)


@pytest.mark.parametrize("characters,expected", [(0, "0"), (1, "0.0003"), (11, "0.0033")])
def test_speech_exact_costs_preserve_float_interface(characters, expected):
    cost = ModelPricingConfig().calculate_speech_cost({"text_words": characters}, "seed-tts-2.0")
    for name in ("primary_cost", "standard_cost"):
        assert isinstance(cost[name], float)
        assert cost[name] == float(expected)
        exact = cost[f"{name}_exact"]
        assert isinstance(exact, str)
        assert "e" not in exact.lower()
        assert Decimal(exact) == Decimal(expected)


@pytest.mark.parametrize("extra_token,expected_points", [(-1, 3), (0, 3), (1, 4)])
def test_chat_exact_costs_are_summed_before_credit_ceiling(extra_token, expected_points):
    pricing = ModelPricingConfig()
    # 两次调用恰好合计 0.01 元时，float 累计可能得到 0.010000000000000002。
    costs = [pricing.calculate_cost(extract(prompt, 0), MODEL, provider="volcengine")
             for prompt in (2464, 10036 + extra_token)]
    total = sum((Decimal(cost["primary_cost_exact"]) for cost in costs), Decimal(0))
    points = (total * Decimal(300)).to_integral_value(rounding=ROUND_CEILING)
    assert points == expected_points


def test_speech_exact_cost_uses_the_same_credit_ceiling_interface():
    cost = ModelPricingConfig().calculate_speech_cost({"text_words": 100}, "seed-tts-2.0")
    points = (Decimal(cost["primary_cost_exact"]) * Decimal(300)).to_integral_value(
        rounding=ROUND_CEILING)
    assert points == 9


@pytest.mark.parametrize("usage", [{}, {"text_words": None}, {"text_words": -1},
                                   {"text_words": 2.5}, {"text_words": True}])
def test_invalid_speech_usage(usage):
    with pytest.raises(ValueError):
        ModelPricingConfig().calculate_speech_cost(usage, "seed-tts-2.0")


def test_unknown_speech_resource():
    with pytest.raises(ValueError):
        ModelPricingConfig().calculate_speech_cost({"text_words": 11}, "seed-icl-2.0")
