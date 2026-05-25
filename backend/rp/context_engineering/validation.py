"""Common validation helpers for compact payloads and recovery refs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from rp.context_engineering.contracts import (
    ContextValidationIssue,
    ContextValidationPolicy,
    ContextValidationReport,
)


def _issue(
    code: str,
    message: str,
    *,
    field_path: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ContextValidationIssue:
    return ContextValidationIssue(
        code=code,
        message=message,
        field_path=field_path,
        metadata=dict(metadata or {}),
    )


def validate_payload_against_policy(
    *,
    payload: Mapping[str, Any],
    policy: ContextValidationPolicy,
) -> ContextValidationReport:
    """Validate all payload policy issues without stopping at the first one."""

    issues: list[ContextValidationIssue] = []
    allowed_fields = policy.metadata.get("allowed_payload_fields")
    allowed_set = (
        {str(field) for field in allowed_fields}
        if isinstance(allowed_fields, (list, tuple, set))
        else None
    )
    if policy.reject_unknown_fields and allowed_set is not None:
        for field in sorted(set(payload) - allowed_set):
            issues.append(
                _issue(
                    "unknown_payload_field",
                    f"Payload field is not allowed: {field}",
                    field_path=field,
                )
            )

    forbidden = set(policy.forbidden_payload_fields)
    for field in sorted(set(payload) & forbidden):
        issues.append(
            _issue(
                "forbidden_payload_field",
                f"Payload field is forbidden: {field}",
                field_path=field,
            )
        )

    for field, limit in policy.max_list_lengths.items():
        value = payload.get(field)
        if isinstance(value, list) and len(value) > int(limit):
            issues.append(
                _issue(
                    "list_too_long",
                    f"List field exceeds max length {limit}: {field}",
                    field_path=field,
                    metadata={"limit": int(limit), "actual": len(value)},
                )
            )

    for field, limit in policy.max_string_lengths.items():
        value = payload.get(field)
        if isinstance(value, str) and len(value) > int(limit):
            issues.append(
                _issue(
                    "string_too_long",
                    f"String field exceeds max length {limit}: {field}",
                    field_path=field,
                    metadata={"limit": int(limit), "actual": len(value)},
                )
            )
        if isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, str) and len(item) > int(limit):
                    issues.append(
                        _issue(
                            "string_too_long",
                            f"List string exceeds max length {limit}: {field}[{index}]",
                            field_path=f"{field}.{index}",
                            metadata={"limit": int(limit), "actual": len(item)},
                        )
                    )

    if policy.allowed_source_refs:
        allowed_source_refs = set(policy.allowed_source_refs)
        source_ref = payload.get("source_ref")
        source_refs = payload.get("source_refs")
        values: list[str] = []
        if isinstance(source_ref, str):
            values.append(source_ref)
        if isinstance(source_refs, list):
            values.extend(str(item) for item in source_refs)
        for value in values:
            if value not in allowed_source_refs:
                issues.append(
                    _issue(
                        "unsupported_source_ref",
                        f"Source ref is not allowed: {value}",
                        field_path="source_ref",
                    )
                )

    return ContextValidationReport(valid=not issues, issues=issues)


def filter_allowed_recovery_refs(
    *,
    refs: Sequence[str],
    policy: ContextValidationPolicy,
) -> tuple[list[str], list[ContextValidationIssue]]:
    """Filter recovery refs by adapter-owned allowed prefixes."""

    allowed_prefixes = tuple(policy.allowed_recovery_ref_prefixes)
    accepted: list[str] = []
    issues: list[ContextValidationIssue] = []
    for ref in refs:
        value = str(ref or "").strip()
        if not value:
            continue
        if allowed_prefixes and not value.startswith(allowed_prefixes):
            issues.append(
                _issue(
                    "unsupported_recovery_ref",
                    f"Recovery ref is not allowed: {value}",
                    field_path="recovery_refs",
                    metadata={"ref": value},
                )
            )
            continue
        if value not in accepted:
            accepted.append(value)
    return accepted, issues
