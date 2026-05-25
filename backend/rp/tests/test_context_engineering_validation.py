"""Validation policy tests for context engineering."""

from __future__ import annotations

from rp.context_engineering.policies import default_validation_policy
from rp.context_engineering.validation import (
    filter_allowed_recovery_refs,
    validate_payload_against_policy,
)


def test_payload_validation_reports_all_issue_families():
    policy = default_validation_policy(
        allowed_recovery_ref_prefixes=["draft:"],
        allowed_source_refs=["source:allowed"],
        forbidden_payload_fields=["analysis"],
        max_list_lengths={"summary_lines": 1},
        max_string_lengths={"title": 4, "summary_lines": 5},
        metadata={
            "allowed_payload_fields": [
                "summary_lines",
                "analysis",
                "title",
                "source_ref",
            ]
        },
    )
    report = validate_payload_against_policy(
        payload={
            "summary_lines": ["first line", "second line"],
            "analysis": "scratchpad",
            "title": "too long title",
            "source_ref": "source:denied",
            "unknown": True,
        },
        policy=policy,
    )

    codes = {issue.code for issue in report.issues}
    assert report.valid is False
    assert "unknown_payload_field" in codes
    assert "forbidden_payload_field" in codes
    assert "list_too_long" in codes
    assert "string_too_long" in codes
    assert "unsupported_source_ref" in codes


def test_filter_allowed_recovery_refs_keeps_supported_prefixes():
    policy = default_validation_policy(allowed_recovery_ref_prefixes=["draft:"])

    accepted, issues = filter_allowed_recovery_refs(
        refs=["draft:story_config", "bad:ref"],
        policy=policy,
    )

    assert accepted == ["draft:story_config"]
    assert [issue.code for issue in issues] == ["unsupported_recovery_ref"]
