"""Offline only; no inferred supplier billing rule."""
from decimal import Decimal
import pytest
from .asr_duration_pricing import duration_budget_cap, calculate_asr_duration_cost, MOCK_USAGE_SOURCE


def test_cap_rounds_money_conservatively_not_audio_usage():
    cap = duration_budget_cap(20)
    assert Decimal(cap['budget_cap_cny']) >= Decimal(20) / 3600
    assert Decimal(cap['budget_cap_cny']) <= Decimal('0.01')
    assert cap['billable_seconds'] is None and cap['billing_contract'] == 'unconfirmed'


@pytest.mark.parametrize('usage', [None, {'audio_info': {'duration': 6470}}, {'billable_seconds': 7}])
def test_no_production_usage_is_currently_pricable(usage):
    with pytest.raises(ValueError):
        calculate_asr_duration_cost(usage)


def test_mock_explicit_and_bounds():
    assert calculate_asr_duration_cost({'billable_seconds': 0}, usage_source=MOCK_USAGE_SOURCE)['synthetic']
    for value in (-1, 61, True, 6.47, '7'):
        with pytest.raises(ValueError):
            calculate_asr_duration_cost({'billable_seconds': value}, usage_source=MOCK_USAGE_SOURCE)
    for seconds in (0, 61, True, 20.1):
        with pytest.raises(ValueError):
            duration_budget_cap(seconds)
