"""Compact artifact reuse/update/rebuild/fallback mechanics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, Protocol

from rp.context_engineering.contracts import (
    ContextArtifact,
    ContextCompactPromptRequest,
    ContextFallbackReport,
    ContextOperationRequest,
    ContextOperationResult,
    ContextReadManifest,
    ContextSourceItem,
    ContextValidationReport,
)
from rp.context_engineering.fingerprinting import (
    fingerprint_source_items,
    is_valid_prefix_artifact,
)
from rp.context_engineering.tracing import build_manifest_item, build_trace
from rp.context_engineering.validation import validate_payload_against_policy

CompactionAction = Literal["not_needed", "reused", "updated", "rebuilt"]


class ContextCompactPromptFailure(Exception):
    """Known model/provider compact failure that may use adapter fallback."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class CompactPromptRunner(Protocol):
    """Injected no-tools compact prompt runner."""

    async def run_compact_prompt(
        self,
        request: ContextCompactPromptRequest,
    ) -> dict[str, Any]: ...


def decide_compaction_action(
    *,
    dropped_items: Sequence[ContextSourceItem],
    previous_artifact: ContextArtifact | None,
) -> CompactionAction:
    """Choose reuse/update/rebuild based only on dropped source fingerprints."""

    if not dropped_items:
        return "not_needed"
    if previous_artifact is None:
        return "rebuilt"
    if not previous_artifact.validation_report.valid:
        return "rebuilt"
    fingerprint = fingerprint_source_items(dropped_items)
    if (
        previous_artifact.source_fingerprint == fingerprint
        and previous_artifact.source_item_count == len(dropped_items)
    ):
        return "reused"
    if is_valid_prefix_artifact(
        previous_artifact=previous_artifact,
        dropped_items=dropped_items,
    ):
        return "updated"
    return "rebuilt"


async def run_compact_operation(
    *,
    request: ContextOperationRequest,
    dropped_items: Sequence[ContextSourceItem],
    first_kept_source_item_id: str | None,
    compact_prompt_runner: CompactPromptRunner | None = None,
) -> ContextOperationResult:
    """Run a compact operation and return only validated model artifacts."""

    dropped = list(dropped_items)
    action = decide_compaction_action(
        dropped_items=dropped,
        previous_artifact=request.previous_artifact,
    )
    manifest = _manifest_for_dropped(dropped)
    empty_report = ContextValidationReport(valid=True)

    if action == "not_needed":
        return _operation_result(
            request=request,
            status="not_needed",
            dropped_items=dropped,
            manifest=manifest,
            validation_report=empty_report,
            artifact=None,
            fallback_report=None,
            summary_action="not_needed",
        )

    if action == "reused" and request.previous_artifact is not None:
        return _operation_result(
            request=request,
            status="reused",
            dropped_items=dropped,
            manifest=manifest,
            validation_report=request.previous_artifact.validation_report,
            artifact=request.previous_artifact,
            fallback_report=None,
            summary_action="reused",
        )

    fingerprint = fingerprint_source_items(dropped)
    prompt_items = _prompt_items_for_action(
        action=action,
        dropped=dropped,
        previous_artifact=request.previous_artifact,
    )
    if compact_prompt_runner is not None:
        prompt_request = ContextCompactPromptRequest(
            operation_id=request.operation_id,
            action=action,  # type: ignore[arg-type]
            schema_id=request.validation_policy.schema_id,
            source_fingerprint=fingerprint,
            source_item_count=len(dropped),
            dropped_items=prompt_items,
            previous_artifact_payload=(
                request.previous_artifact.payload
                if action == "updated" and request.previous_artifact is not None
                else None
            ),
            first_kept_source_item_id=first_kept_source_item_id,
            validation_policy=request.validation_policy,
            fallback_policy=request.fallback_policy,
            metadata=dict(request.metadata),
        )
        try:
            payload = await compact_prompt_runner.run_compact_prompt(prompt_request)
        except ContextCompactPromptFailure as exc:
            return _fallback_result(
                request=request,
                dropped_items=dropped,
                manifest=manifest,
                first_kept_source_item_id=first_kept_source_item_id,
                reason=exc.reason,
                summary_action=action,
            )
        artifact = _artifact_from_model_payload(
            request=request,
            payload=payload,
            source_fingerprint=fingerprint,
            source_item_count=len(dropped),
            first_kept_source_item_id=first_kept_source_item_id,
        )
        if artifact.validation_report.valid:
            return _operation_result(
                request=request,
                status=action,
                dropped_items=dropped,
                manifest=manifest,
                validation_report=artifact.validation_report,
                artifact=artifact,
                fallback_report=None,
                summary_action=action,
            )
        return _fallback_result(
            request=request,
            dropped_items=dropped,
            manifest=manifest,
            first_kept_source_item_id=first_kept_source_item_id,
            reason=_issue_reason(artifact.validation_report),
            summary_action=action,
        )

    return _fallback_result(
        request=request,
        dropped_items=dropped,
        manifest=manifest,
        first_kept_source_item_id=first_kept_source_item_id,
        reason="compact_prompt_runner_unavailable",
        summary_action=action,
    )


def _prompt_items_for_action(
    *,
    action: CompactionAction,
    dropped: list[ContextSourceItem],
    previous_artifact: ContextArtifact | None,
) -> list[ContextSourceItem]:
    if action == "updated" and previous_artifact is not None:
        return list(dropped[int(previous_artifact.source_item_count) :])
    return list(dropped)


def _artifact_from_model_payload(
    *,
    request: ContextOperationRequest,
    payload: dict[str, Any],
    source_fingerprint: str,
    source_item_count: int,
    first_kept_source_item_id: str | None,
) -> ContextArtifact:
    normalized = dict(payload)
    report = validate_payload_against_policy(
        payload=normalized,
        policy=request.validation_policy,
    )
    return ContextArtifact(
        artifact_id=f"{request.operation_id}:compact:{source_fingerprint[:12]}",
        artifact_kind="compact_summary",
        schema_id=request.validation_policy.schema_id or "context.compact.v1",
        schema_version="1",
        source_fingerprint=source_fingerprint,
        source_item_count=source_item_count,
        payload=normalized,
        recovery_refs=_payload_recovery_refs(normalized),
        first_kept_source_item_id=first_kept_source_item_id,
        created_by="model",
        validation_report=report,
    )


def _fallback_result(
    *,
    request: ContextOperationRequest,
    dropped_items: list[ContextSourceItem],
    manifest: ContextReadManifest,
    first_kept_source_item_id: str | None,
    reason: str,
    summary_action: str,
) -> ContextOperationResult:
    fallback_report = ContextFallbackReport(
        reason=reason[:240],
        user_visible_error_code=request.fallback_policy.user_visible_error_code,
        metadata={"first_kept_source_item_id": first_kept_source_item_id},
    )
    return _operation_result(
        request=request,
        status="fallback",
        dropped_items=dropped_items,
        manifest=manifest,
        validation_report=ContextValidationReport(valid=True),
        artifact=None,
        fallback_report=fallback_report,
        summary_action=summary_action,
        fallback_reason=reason[:240],
    )


def _operation_result(
    *,
    request: ContextOperationRequest,
    status: str,
    dropped_items: Sequence[ContextSourceItem],
    manifest: ContextReadManifest,
    validation_report: ContextValidationReport,
    artifact: ContextArtifact | None,
    fallback_report: ContextFallbackReport | None,
    summary_action: str,
    fallback_reason: str | None = None,
) -> ContextOperationResult:
    trace = build_trace(
        operation_id=request.operation_id,
        operation_kind=request.operation_kind,
        runtime_family=request.runtime_family,
        source_items=request.source_items,
        selected_items=[],
        read_manifest=manifest,
        summary_action=summary_action,
        fallback_reason=fallback_reason,
        provider_usage=dict((request.metadata or {}).get("provider_usage") or {}),
        metadata={
            **request.metadata,
            "dropped_source_count": len(dropped_items),
            "artifact_source_item_count": artifact.source_item_count if artifact else 0,
        },
    )
    return ContextOperationResult(
        operation_id=request.operation_id,
        status=status,  # type: ignore[arg-type]
        artifact=artifact,
        read_manifest=manifest,
        trace=trace,
        validation_report=validation_report,
        fallback_report=fallback_report,
    )


def _manifest_for_dropped(
    dropped_items: Sequence[ContextSourceItem],
) -> ContextReadManifest:
    manifest = ContextReadManifest()
    for item in dropped_items:
        manifest.omitted.append(
            build_manifest_item(
                item,
                decision="omitted",
                reason="compact_source",
            )
        )
    return manifest


def _payload_recovery_refs(payload: dict[str, Any]) -> list[str]:
    value = payload.get("recovery_refs")
    if not isinstance(value, list):
        return []
    refs: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text and text not in refs:
            refs.append(text)
    return refs


def _issue_reason(report: ContextValidationReport) -> str:
    if not report.issues:
        return "compact_payload_invalid"
    return ",".join(issue.code for issue in report.issues)[:240]
