"""Compaction operation tests for context engineering."""

from __future__ import annotations

import asyncio
from typing import Any

from rp.context_engineering.compaction import (
    CompactPromptRunner,
    ContextCompactPromptFailure,
    decide_compaction_action,
    run_compact_operation,
)
from rp.context_engineering.contracts import (
    ContextArtifact,
    ContextCompactPromptRequest,
    ContextOperationRequest,
    ContextSourceItem,
    ContextValidationIssue,
    ContextValidationReport,
)
from rp.context_engineering.fingerprinting import fingerprint_source_items
from rp.context_engineering.policies import (
    default_budget_policy,
    default_fallback_policy,
    default_placement_policy,
    default_validation_policy,
)


class _Runner(CompactPromptRunner):
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.requests: list[ContextCompactPromptRequest] = []

    async def run_compact_prompt(
        self,
        request: ContextCompactPromptRequest,
    ) -> dict[str, Any]:
        self.requests.append(request)
        return dict(self.payload)


def _item(index: int, *, recovery_refs: list[str] | None = None) -> ContextSourceItem:
    return ContextSourceItem(
        source_item_id=f"turn-{index}",
        source_family="user_turn",
        source_scope="scope",
        sequence_index=index,
        text=f"message {index}",
        recovery_refs=list(recovery_refs or []),
    )


def _request(
    dropped: list[ContextSourceItem],
    *,
    previous_artifact: ContextArtifact | None = None,
) -> ContextOperationRequest:
    return ContextOperationRequest(
        operation_id="op-compact",
        operation_kind="compact",
        runtime_family="test",
        source_items=dropped,
        budget_policy=default_budget_policy(),
        placement_policy=default_placement_policy(),
        validation_policy=default_validation_policy(
            allowed_recovery_ref_prefixes=["draft:"],
            forbidden_payload_fields=["analysis", "scratchpad"],
            metadata={
                "allowed_payload_fields": [
                    "summary_lines",
                    "draft_refs",
                    "recovery_hints",
                    "confirmed_points",
                    "open_threads",
                    "rejected_directions",
                    "must_not_infer",
                ]
            },
        ),
        fallback_policy=default_fallback_policy(),
        previous_artifact=previous_artifact,
    )


def _artifact(items: list[ContextSourceItem]) -> ContextArtifact:
    fingerprint = fingerprint_source_items(items)
    return ContextArtifact(
        artifact_id="artifact",
        artifact_kind="compact_summary",
        schema_id="test.v1",
        schema_version="1",
        source_fingerprint=fingerprint,
        source_item_count=len(items),
        payload={
            "source_fingerprint": fingerprint,
            "source_message_count": len(items),
            "summary_lines": ["previous"],
        },
        created_by="adapter",
        validation_report=ContextValidationReport(valid=True),
    )


def test_no_dropped_items_returns_not_needed():
    result = asyncio.run(
        run_compact_operation(
            request=_request([]),
            dropped_items=[],
            first_kept_source_item_id=None,
        )
    )

    assert result.status == "not_needed"
    assert result.artifact is None


def test_matching_previous_fingerprint_reuses_previous_artifact():
    dropped = [_item(0), _item(1)]
    artifact = _artifact(dropped)

    assert (
        decide_compaction_action(
            dropped_items=dropped,
            previous_artifact=artifact,
        )
        == "reused"
    )
    result = asyncio.run(
        run_compact_operation(
            request=_request(dropped, previous_artifact=artifact),
            dropped_items=dropped,
            first_kept_source_item_id="turn-2",
        )
    )

    assert result.status == "reused"
    assert result.artifact is artifact


def test_valid_prefix_updates_from_previous_artifact_plus_delta():
    dropped = [_item(0), _item(1), _item(2)]
    previous = _artifact(dropped[:2])
    runner = _Runner({"summary_lines": ["updated"]})

    result = asyncio.run(
        run_compact_operation(
            request=_request(dropped, previous_artifact=previous),
            dropped_items=dropped,
            first_kept_source_item_id="turn-3",
            compact_prompt_runner=runner,
        )
    )

    assert result.status == "updated"
    assert runner.requests[0].previous_artifact_payload is not None
    assert [item.source_item_id for item in runner.requests[0].dropped_items] == [
        "turn-2"
    ]
    assert result.artifact is not None
    assert result.artifact.first_kept_source_item_id == "turn-3"
    assert "source_fingerprint" not in result.artifact.payload
    assert "source_message_count" not in result.artifact.payload
    assert result.artifact.source_item_count == len(dropped)


def test_generic_model_payload_does_not_receive_setup_metadata():
    dropped = [_item(0), _item(1)]
    runner = _Runner({"summary": "ok"})
    request = ContextOperationRequest(
        operation_id="generic-compact",
        operation_kind="compact",
        runtime_family="generic",
        source_items=dropped,
        budget_policy=default_budget_policy(),
        placement_policy=default_placement_policy(),
        validation_policy=default_validation_policy(
            schema_id="generic_summary.v1",
            metadata={"allowed_payload_fields": ["summary"]},
        ),
        fallback_policy=default_fallback_policy(),
    )

    result = asyncio.run(
        run_compact_operation(
            request=request,
            dropped_items=dropped,
            first_kept_source_item_id="turn-2",
            compact_prompt_runner=runner,
        )
    )

    assert result.status == "rebuilt"
    assert result.artifact is not None
    assert result.artifact.payload == {"summary": "ok"}
    assert result.artifact.source_fingerprint
    assert result.artifact.source_item_count == 2


def test_invalid_previous_artifact_is_rebuilt_not_reused():
    dropped = [_item(0), _item(1)]
    previous = _artifact(dropped)
    previous = previous.model_copy(
        update={
            "validation_report": ContextValidationReport(
                valid=False,
                issues=[
                    ContextValidationIssue(
                        code="unsupported_recovery_ref",
                        message="Previous compact artifact is invalid.",
                    )
                ],
            )
        }
    )
    runner = _Runner({"summary_lines": ["rebuilt"]})

    result = asyncio.run(
        run_compact_operation(
            request=_request(dropped, previous_artifact=previous),
            dropped_items=dropped,
            first_kept_source_item_id="turn-2",
            compact_prompt_runner=runner,
        )
    )

    assert result.status == "rebuilt"
    assert [item.source_item_id for item in runner.requests[0].dropped_items] == [
        "turn-0",
        "turn-1",
    ]


def test_invalid_prefix_rebuilds_from_full_dropped_set():
    dropped = [_item(0), _item(1)]
    previous = _artifact([_item(99)])
    runner = _Runner({"summary_lines": ["rebuilt"]})

    result = asyncio.run(
        run_compact_operation(
            request=_request(dropped, previous_artifact=previous),
            dropped_items=dropped,
            first_kept_source_item_id="turn-2",
            compact_prompt_runner=runner,
        )
    )

    assert result.status == "rebuilt"
    assert [item.source_item_id for item in runner.requests[0].dropped_items] == [
        "turn-0",
        "turn-1",
    ]


def test_invalid_model_payload_records_fallback_without_kernel_payload():
    dropped = [_item(0)]
    runner = _Runner({"summary_lines": ["bad"], "analysis": "scratchpad"})

    result = asyncio.run(
        run_compact_operation(
            request=_request(dropped),
            dropped_items=dropped,
            first_kept_source_item_id="turn-1",
            compact_prompt_runner=runner,
        )
    )

    assert result.status == "fallback"
    assert result.artifact is None
    assert result.fallback_report is not None
    assert "forbidden_payload_field" in result.fallback_report.reason


def test_known_runner_failure_records_fallback_without_swallowing_all_exceptions():
    class _FailingRunner(CompactPromptRunner):
        async def run_compact_prompt(
            self,
            request: ContextCompactPromptRequest,
        ) -> dict[str, Any]:
            raise ContextCompactPromptFailure("model_output_not_json")

    dropped = [_item(0)]

    result = asyncio.run(
        run_compact_operation(
            request=_request(dropped),
            dropped_items=dropped,
            first_kept_source_item_id=None,
            compact_prompt_runner=_FailingRunner(),
        )
    )

    assert result.status == "fallback"
    assert result.artifact is None
    assert result.fallback_report is not None
    assert result.fallback_report.reason == "model_output_not_json"


def test_unexpected_runner_exception_propagates():
    class _BuggyRunner(CompactPromptRunner):
        async def run_compact_prompt(
            self,
            request: ContextCompactPromptRequest,
        ) -> dict[str, Any]:
            raise RuntimeError("programming bug")

    dropped = [_item(0, recovery_refs=["draft:story_config", "bad:ref"])]

    try:
        asyncio.run(
            run_compact_operation(
                request=_request(dropped),
                dropped_items=dropped,
                first_kept_source_item_id="turn-1",
                compact_prompt_runner=_BuggyRunner(),
            )
        )
    except RuntimeError as exc:
        assert str(exc) == "programming bug"
    else:
        raise AssertionError("expected unexpected runner exception to propagate")
