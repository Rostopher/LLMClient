"""Seed ASR2 duration pricing. Audio metadata is NOT billable usage.

Official price: 1 CNY/hour, second precision. Per-call rounding and a trusted
billing field remain unconfirmed. Consequently only synthetic mock usage can
be priced here; production cannot enable charging by setting a boolean.
"""
from decimal import Decimal, ROUND_CEILING

RESOURCE_ID = 'volc.seedasr.sauc.duration'
PROVIDER = 'seed_asr2_bigmodel_async'
PRICE_VERSION = 'volc-seedasr2-duration-cn-2026-11-06'
RATE_CNY_PER_HOUR = Decimal('1')
MOCK_USAGE_SOURCE = 'local_mock_billable_seconds_v1'


def duration_budget_cap(max_seconds: int, resource_id: str = RESOURCE_ID):
    if resource_id != RESOURCE_ID or type(max_seconds) is not int or not 1 <= max_seconds <= 60:
        raise ValueError('Only verified ASR2 hourly resource and 1..60 second cap allowed')
    # Rounding up the MONEY cap protects a DECIMAL(24,12) reservation. This is
    # our budget policy, not a claim about supplier per-request rounding.
    cap = (Decimal(max_seconds) / 3600 * RATE_CNY_PER_HOUR).quantize(
        Decimal('0.000000000001'), rounding=ROUND_CEILING)
    return {'provider': PROVIDER, 'resource_id': resource_id, 'price_version': PRICE_VERSION,
            'rate_cny_per_hour': str(RATE_CNY_PER_HOUR), 'max_audio_seconds': max_seconds,
            'budget_cap_cny': str(cap), 'budget_policy': 'max_seconds_money_ceiling_12dp',
            'billing_contract': 'unconfirmed', 'billing_precision': 'seconds',
            'billable_seconds': None}


def calculate_asr_duration_cost(usage, resource_id=RESOURCE_ID, *, usage_source=None):
    if resource_id != RESOURCE_ID:
        raise ValueError('Unverified ASR resource')
    if usage_source != MOCK_USAGE_SOURCE:
        raise ValueError('Production billable-second source/rounding contract is unconfirmed')
    if not isinstance(usage, dict) or set(usage) != {'billable_seconds'}:
        raise ValueError('Explicit synthetic billable seconds required; duration_ms is metadata')
    seconds = usage['billable_seconds']
    if type(seconds) is not int or not 0 <= seconds <= 60:
        raise ValueError('Invalid synthetic seconds')
    cost = Decimal(seconds) / 3600 * RATE_CNY_PER_HOUR
    return {'primary_cost_exact': str(cost), 'currency': 'CNY', 'price_version': PRICE_VERSION,
            'resource_id': resource_id, 'billable_seconds': seconds, 'synthetic': True,
            'usage_source': MOCK_USAGE_SOURCE}
