"""Token estimation tests for context engineering."""

from __future__ import annotations

from rp.context_engineering.contracts import ContextSourceItem
from rp.context_engineering.estimation import (
    estimate_payload_tokens,
    estimate_source_item_tokens,
    estimate_source_items_tokens,
    estimate_text_tokens,
)


def test_text_estimate_uses_ceil_chars_div_four():
    assert estimate_text_tokens("12345") == 2


def test_empty_text_estimates_to_zero():
    assert estimate_text_tokens("") == 0
    assert estimate_text_tokens(None) == 0


def test_payload_estimate_is_stable_under_dict_key_ordering():
    left = estimate_payload_tokens({"b": 2, "a": 1})
    right = estimate_payload_tokens({"a": 1, "b": 2})

    assert left == right


def test_explicit_source_item_estimate_wins():
    item = ContextSourceItem(
        source_item_id="item-1",
        source_family="runtime_state",
        text="a" * 100,
        estimated_tokens=7,
    )

    assert estimate_source_item_tokens(item) == 7


def test_aggregate_estimate_sums_items():
    items = [
        ContextSourceItem(
            source_item_id="item-1",
            source_family="user_turn",
            text="1234",
        ),
        ContextSourceItem(
            source_item_id="item-2",
            source_family="assistant_turn",
            text="12345",
        ),
    ]

    assert estimate_source_items_tokens(items) == 3
