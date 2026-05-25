"""Setup adapter tests for context engineering."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from rp.agent_runtime.contracts import (
    SetupCompactRecoveryHint,
    SetupContextCompactSummary,
    SetupToolOutcome,
    SetupWorkingDigest,
)
from rp.context_engineering.adapters.setup import SetupContextEngineeringAdapter
from rp.context_engineering.compaction import CompactPromptRunner, run_compact_operation
from rp.context_engineering.contracts import ContextCompactPromptRequest
from rp.context_engineering.fingerprinting import fingerprint_source_items
from rp.context_engineering.selection import select_context_sections
from rp.models.setup_agent import SetupAgentDialogueMessage
from rp.services.setup_context_compaction_service import SetupContextCompactionService


def _history() -> list[SetupAgentDialogueMessage]:
    return [
        SetupAgentDialogueMessage(role="user", content="user asks"),
        SetupAgentDialogueMessage(role="assistant", content="assistant answers"),
        SetupAgentDialogueMessage(role="user", content="user follows up"),
        SetupAgentDialogueMessage(role="assistant", content="assistant clarifies"),
        SetupAgentDialogueMessage(role="user", content="older should drop"),
    ]


def _tool_outcome() -> SetupToolOutcome:
    return SetupToolOutcome(
        tool_name="rp_setup__setup.patch.foundation",
        success=True,
        summary="Updated foundation draft.",
        updated_refs=["foundation:magic-law"],
        relevance="draft",
        recorded_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
    )


def test_setup_history_maps_to_user_and_assistant_source_families():
    adapter = SetupContextEngineeringAdapter()
    request = adapter.build_stage_local_compact_request(
        history=_history()[:2],
        retained_tool_outcomes=[],
        working_digest=None,
        existing_summary=None,
        context_profile="standard",
        current_step="foundation",
        current_stage="foundation",
        estimated_input_tokens=123,
        previous_usage={"prompt_tokens": 100, "total_tokens": 200},
    )

    assert [item.source_family for item in request.source_items] == [
        "user_turn",
        "assistant_turn",
    ]
    assert {item.source_scope for item in request.source_items} == {
        "setup_stage:foundation"
    }
    assert [item.sequence_index for item in request.source_items] == [0, 1]


def test_setup_source_scope_falls_back_to_current_step():
    adapter = SetupContextEngineeringAdapter()
    request = adapter.build_stage_local_compact_request(
        history=_history()[:1],
        retained_tool_outcomes=[],
        working_digest=None,
        existing_summary=None,
        context_profile="standard",
        current_step="story_config",
        current_stage=None,
        estimated_input_tokens=None,
        previous_usage=None,
    )

    assert request.source_items[0].source_scope == "setup_step:story_config"


def test_setup_tool_outcome_and_working_digest_map_without_common_setup_enums():
    adapter = SetupContextEngineeringAdapter()
    request = adapter.build_stage_local_compact_request(
        history=[],
        retained_tool_outcomes=[_tool_outcome()],
        working_digest=SetupWorkingDigest(draft_refs=["foundation:magic-law"]),
        existing_summary=None,
        context_profile="compact",
        current_step="foundation",
        current_stage="foundation",
        estimated_input_tokens=None,
        previous_usage=None,
    )

    tool = request.source_items[0]
    digest = request.source_items[1]
    assert tool.source_family == "tool_outcome"
    assert tool.payload["relevance"] == "draft"
    assert digest.source_family == "runtime_state"
    assert request.validation_policy.allowed_recovery_ref_prefixes == [
        "draft:",
        "foundation:",
        "stage:",
    ]


def test_previous_setup_summary_maps_only_to_previous_artifact():
    adapter = SetupContextEngineeringAdapter()
    summary = SetupContextCompactSummary(
        source_fingerprint="fp-existing",
        source_message_count=2,
        summary_lines=["previous"],
        draft_refs=["foundation:magic-law"],
        recovery_hints=[
            SetupCompactRecoveryHint(
                ref="foundation:magic-law",
                reason="Recover exact law.",
            )
        ],
    )
    with_summary = adapter.build_stage_local_compact_request(
        history=_history()[:2],
        retained_tool_outcomes=[],
        working_digest=None,
        existing_summary=summary,
        context_profile="standard",
        current_step="foundation",
        current_stage="foundation",
        estimated_input_tokens=None,
        previous_usage=None,
    )
    without_summary = adapter.build_stage_local_compact_request(
        history=_history()[:2],
        retained_tool_outcomes=[],
        working_digest=None,
        existing_summary=None,
        context_profile="standard",
        current_step="foundation",
        current_stage="foundation",
        estimated_input_tokens=None,
        previous_usage=None,
    )

    assert with_summary.previous_artifact is not None
    assert all(
        item.source_family != "compact_artifact" for item in with_summary.source_items
    )
    assert fingerprint_source_items(with_summary.source_items) == (
        fingerprint_source_items(without_summary.source_items)
    )


def test_previous_setup_summary_validation_report_rejects_bad_refs():
    adapter = SetupContextEngineeringAdapter()
    summary = SetupContextCompactSummary(
        source_fingerprint="fp-existing",
        source_message_count=1,
        summary_lines=["previous"],
        draft_refs=["bad:ref"],
    )

    artifact = adapter.to_context_artifact(summary)

    assert artifact is not None
    assert artifact.validation_report.valid is False
    assert {issue.code for issue in artifact.validation_report.issues} == {
        "unsupported_recovery_ref"
    }


def test_result_maps_back_to_setup_summary_and_governance_metadata():
    class _Runner(CompactPromptRunner):
        async def run_compact_prompt(
            self,
            request: ContextCompactPromptRequest,
        ) -> dict[str, Any]:
            return {
                "summary_lines": ["model summary"],
                "draft_refs": ["foundation:magic-law"],
                "recovery_hints": [
                    {
                        "ref": "foundation:magic-law",
                        "reason": "Recover exact foundation law.",
                    }
                ],
            }

    adapter = SetupContextEngineeringAdapter()
    request = adapter.build_stage_local_compact_request(
        history=_history(),
        retained_tool_outcomes=[_tool_outcome()],
        working_digest=SetupWorkingDigest(draft_refs=["foundation:magic-law"]),
        existing_summary=None,
        context_profile="compact",
        current_step="foundation",
        current_stage="foundation",
        estimated_input_tokens=500,
        previous_usage={"prompt_tokens": 400, "total_tokens": 600},
    )
    selection = select_context_sections(request)
    result = asyncio.run(
        run_compact_operation(
            request=request,
            dropped_items=selection.compactable_dropped_items,
            first_kept_source_item_id=(
                selection.recent_raw_items[0].source_item_id
                if selection.recent_raw_items
                else None
            ),
            compact_prompt_runner=_Runner(),
        )
    )

    assert result.artifact is not None
    assert "source_fingerprint" not in result.artifact.payload
    assert "source_message_count" not in result.artifact.payload
    setup_summary = adapter.to_setup_compact_summary(result.artifact)
    metadata = adapter.to_setup_governance_metadata(result)
    report = adapter.to_setup_context_report(
        result=result,
        raw_history_count=5,
        raw_history_chars=50,
        user_edit_delta_count=1,
        prior_stage_handoff_count=0,
        context_profile="compact",
        profile_reasons=["history_count_threshold"],
    )

    assert setup_summary is not None
    assert setup_summary.source_message_count == 1
    assert metadata["raw_history_limit"] == 4
    assert metadata["kept_history_count"] == 4
    assert metadata["compacted_history_count"] == 1
    assert metadata["estimated_input_tokens"] == 500
    assert metadata["previous_prompt_tokens"] == 400
    assert metadata["previous_total_tokens"] == 600
    assert metadata["summary_strategy"] == "compact_prompt_summary"
    assert metadata["summary_action"] == "rebuilt"
    assert report.context_profile == "compact"
    assert report.summary_line_count == len(setup_summary.summary_lines)


def test_setup_service_bridge_keeps_common_model_artifact_payload_metadata_free():
    provider_payloads: list[dict[str, Any]] = []

    def _compact_prompt_provider(_messages):
        payload = {
            "summary_lines": ["model bridge summary"],
            "confirmed_points": ["Bridge keeps setup content fields."],
            "open_threads": ["Need exact draft detail."],
            "rejected_directions": ["Do not infer prior-stage raw discussion."],
            "draft_refs": ["foundation:magic-law"],
            "recovery_hints": [
                {
                    "ref": "foundation:magic-law",
                    "reason": "Recover exact foundation law.",
                }
            ],
            "must_not_infer": ["Do not treat compact history as committed truth."],
        }
        provider_payloads.append(dict(payload))
        return payload

    adapter = SetupContextEngineeringAdapter()
    service = SetupContextCompactionService(
        compact_prompt_provider=_compact_prompt_provider
    )
    request = adapter.build_stage_local_compact_request(
        history=_history(),
        retained_tool_outcomes=[_tool_outcome()],
        working_digest=SetupWorkingDigest(draft_refs=["foundation:magic-law"]),
        existing_summary=None,
        context_profile="compact",
        current_step="foundation",
        current_stage="foundation",
        estimated_input_tokens=500,
        previous_usage=None,
    )
    selection = select_context_sections(request)

    result = asyncio.run(
        service._run_common_compact_operation(
            request=request,
            selection=selection,
            retained_tool_outcomes=[_tool_outcome()],
            working_digest=SetupWorkingDigest(draft_refs=["foundation:magic-law"]),
            current_step="foundation",
            allow_async_provider=False,
        )
    )

    assert provider_payloads
    assert result.artifact is not None
    assert result.artifact.created_by == "model"
    assert "source_fingerprint" not in result.artifact.payload
    assert "source_message_count" not in result.artifact.payload
    assert result.artifact.payload["summary_lines"] == ["model bridge summary"]
    assert result.artifact.payload["confirmed_points"] == [
        "Bridge keeps setup content fields."
    ]
    assert result.artifact.payload["open_threads"] == ["Need exact draft detail."]
    assert result.artifact.payload["rejected_directions"] == [
        "Do not infer prior-stage raw discussion."
    ]
    assert result.artifact.payload["draft_refs"] == ["foundation:magic-law"]
    assert result.artifact.payload["recovery_hints"][0]["ref"] == (
        "foundation:magic-law"
    )
    assert result.artifact.payload["must_not_infer"] == [
        "Do not treat compact history as committed truth."
    ]

    service_summary = service.build_summary_from_common_selection(
        request=request,
        selection=selection,
        retained_tool_outcomes=[_tool_outcome()],
        working_digest=SetupWorkingDigest(draft_refs=["foundation:magic-law"]),
        existing_summary=None,
        current_step="foundation",
    )

    assert service_summary is not None
    assert service_summary.source_fingerprint == result.artifact.source_fingerprint
    assert service_summary.source_message_count == result.artifact.source_item_count
    assert service_summary.summary_lines == ["model bridge summary"]
    assert service_summary.confirmed_points == ["Bridge keeps setup content fields."]
    assert service_summary.draft_refs == ["foundation:magic-law"]
