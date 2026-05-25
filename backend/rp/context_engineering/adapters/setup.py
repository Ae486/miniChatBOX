"""SetupAgent adapter for the common context engineering kernel."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

from rp.agent_runtime.contracts import (
    SetupCompactRecoveryHint,
    SetupContextCompactSummary,
    SetupContextGovernanceReport,
    SetupToolOutcome,
    SetupWorkingDigest,
)
from rp.context_engineering.contracts import (
    ContextArtifact,
    ContextOperationRequest,
    ContextOperationResult,
    ContextSourceItem,
    ContextValidationReport,
)
from rp.context_engineering.policies import (
    default_budget_policy,
    default_fallback_policy,
    default_placement_policy,
    default_validation_policy,
)
from rp.context_engineering.validation import (
    filter_allowed_recovery_refs,
    validate_payload_against_policy,
)
from rp.models.setup_agent import SetupAgentDialogueMessage

_SETUP_ALLOWED_PAYLOAD_FIELDS = [
    "source_fingerprint",
    "source_message_count",
    "summary_lines",
    "confirmed_points",
    "open_threads",
    "rejected_directions",
    "draft_refs",
    "recovery_hints",
    "must_not_infer",
]


class SetupContextEngineeringAdapter:
    """Map setup stage-local workload into common context contracts."""

    _STANDARD_RAW_HISTORY_WINDOW = 6
    _COMPACT_RAW_HISTORY_WINDOW = 4
    _STANDARD_TOKEN_BUDGET = 2400
    _COMPACT_TOKEN_BUDGET = 600
    _COMPACT_PROMPT_MAX_TOKENS = 1200

    def build_stage_local_compact_request(
        self,
        *,
        history: Sequence[SetupAgentDialogueMessage],
        retained_tool_outcomes: Sequence[SetupToolOutcome],
        working_digest: SetupWorkingDigest | None,
        existing_summary: SetupContextCompactSummary | None,
        context_profile: Literal["standard", "compact"],
        current_step: str,
        current_stage: str | None = None,
        estimated_input_tokens: int | None,
        input_token_count_source: str | None = None,
        previous_usage: Mapping[str, Any] | None = None,
    ) -> ContextOperationRequest:
        """Build the common compact request without importing setup workspace truth."""

        source_scope = self._source_scope(
            current_step=current_step,
            current_stage=current_stage,
        )
        source_items: list[ContextSourceItem] = []
        for index, item in enumerate(history):
            source_items.append(
                ContextSourceItem(
                    source_item_id=f"setup.history.{index}",
                    source_family="user_turn"
                    if item.role == "user"
                    else "assistant_turn",
                    source_scope=source_scope,
                    sequence_index=index,
                    source_ref=f"setup:{current_step}:history:{index}",
                    text=item.content,
                    metadata={"role": item.role, "current_step": current_step},
                )
            )
        for index, item in enumerate(retained_tool_outcomes):
            source_items.append(
                ContextSourceItem(
                    source_item_id=f"setup.tool_outcome.{index}",
                    source_family="tool_outcome",
                    source_scope=source_scope,
                    sequence_index=len(history) + index,
                    source_ref=f"setup:{current_step}:tool_outcome:{index}",
                    recovery_refs=list(item.updated_refs),
                    text=item.summary,
                    payload={
                        "tool_name": item.tool_name,
                        "success": item.success,
                        "summary": item.summary,
                        "updated_refs": list(item.updated_refs),
                        "error_code": item.error_code,
                        "relevance": item.relevance,
                        "recorded_at": item.recorded_at.isoformat(),
                    },
                )
            )
        if working_digest is not None:
            source_items.append(
                ContextSourceItem(
                    source_item_id="setup.working_digest",
                    source_family="runtime_state",
                    source_scope=source_scope,
                    sequence_index=len(history) + len(retained_tool_outcomes),
                    source_ref=f"setup:{current_step}:working_digest",
                    recovery_refs=list(working_digest.draft_refs),
                    payload=working_digest.model_dump(mode="json", exclude_none=True),
                    metadata={"current_step": current_step},
                )
            )

        raw_history_limit = (
            self._COMPACT_RAW_HISTORY_WINDOW
            if context_profile == "compact"
            else self._STANDARD_RAW_HISTORY_WINDOW
        )
        previous_prompt_tokens = (
            previous_usage.get("prompt_tokens") if previous_usage else None
        )
        previous_total_tokens = (
            previous_usage.get("total_tokens") if previous_usage else None
        )
        previous_token_details = (
            previous_usage.get("token_details") if previous_usage else None
        )
        return ContextOperationRequest(
            operation_id=f"setup:{current_step}:stage_local_compact",
            operation_kind="compact",
            runtime_family="setup",
            source_items=source_items,
            budget_policy=default_budget_policy(
                operation_budget_tokens=(
                    self._COMPACT_TOKEN_BUDGET
                    if context_profile == "compact"
                    else self._STANDARD_TOKEN_BUDGET
                ),
                recent_window_items=raw_history_limit,
            ),
            placement_policy=default_placement_policy(),
            validation_policy=self._setup_validation_policy(),
            fallback_policy=default_fallback_policy(
                fallback_summary_line_limit=6,
            ),
            previous_artifact=self.to_context_artifact(existing_summary),
            metadata={
                "context_profile": context_profile,
                "current_step": current_step,
                "current_stage": current_stage,
                "source_scope": source_scope,
                "raw_history_limit": raw_history_limit,
                "raw_history_count": len(history),
                "raw_history_chars": sum(len(item.content or "") for item in history),
                "kept_history_count": min(len(history), raw_history_limit),
                "compacted_history_count": max(len(history) - raw_history_limit, 0),
                "retained_tool_outcome_count": len(retained_tool_outcomes),
                "estimated_input_tokens": estimated_input_tokens,
                "input_token_count_source": input_token_count_source,
                "previous_prompt_tokens": previous_prompt_tokens,
                "previous_completion_tokens": (
                    previous_usage.get("completion_tokens") if previous_usage else None
                ),
                "previous_total_tokens": previous_total_tokens,
                "previous_cached_tokens": (
                    previous_usage.get("cached_tokens") if previous_usage else None
                ),
                "previous_reasoning_tokens": (
                    previous_usage.get("reasoning_tokens") if previous_usage else None
                ),
                "previous_cache_creation_input_tokens": (
                    previous_usage.get("cache_creation_input_tokens")
                    if previous_usage
                    else None
                ),
                "previous_cache_read_input_tokens": (
                    previous_usage.get("cache_read_input_tokens")
                    if previous_usage
                    else None
                ),
                "previous_usage_source": (
                    previous_usage.get("source") if previous_usage else None
                ),
                "previous_token_details": (
                    dict(previous_token_details)
                    if isinstance(previous_token_details, Mapping)
                    else {}
                ),
            },
        )

    def to_context_artifact(
        self,
        summary: SetupContextCompactSummary | None,
    ) -> ContextArtifact | None:
        """Map setup compact summary to previous_artifact operation state."""

        if summary is None:
            return None
        payload = summary.model_dump(mode="json", exclude_none=True)
        validation_report = self.validate_setup_payload(payload)
        return ContextArtifact(
            artifact_id=f"setup:compact:{summary.source_fingerprint[:12]}",
            artifact_kind="compact_summary",
            schema_id="setup_context_compact_summary.v1",
            schema_version="1",
            source_fingerprint=summary.source_fingerprint,
            source_item_count=summary.source_message_count,
            payload=payload,
            recovery_refs=self._summary_refs(summary),
            created_by="adapter",
            validation_report=validation_report,
        )

    def to_setup_compact_summary(
        self,
        artifact: ContextArtifact | None,
    ) -> SetupContextCompactSummary | None:
        """Map common artifact payload back to the setup adapter schema."""

        if artifact is None:
            return None
        payload = dict(artifact.payload)
        payload["source_fingerprint"] = artifact.source_fingerprint
        payload["source_message_count"] = artifact.source_item_count
        payload.setdefault("summary_lines", [])
        payload.setdefault("confirmed_points", [])
        payload.setdefault("open_threads", [])
        payload.setdefault("rejected_directions", [])
        payload.setdefault("draft_refs", [])
        payload.setdefault("recovery_hints", [])
        payload.setdefault("must_not_infer", [])
        return SetupContextCompactSummary(**payload)

    def to_setup_governance_metadata(
        self,
        result: ContextOperationResult,
    ) -> dict[str, Any]:
        """Project common result trace into existing setup metadata keys."""

        metadata = result.trace.metadata
        summary_action = self._setup_summary_action(result)
        summary_strategy = self._setup_summary_strategy(result)
        return {
            "raw_history_limit": int(metadata.get("raw_history_limit") or 0),
            "kept_history_count": int(metadata.get("kept_history_count") or 0),
            "compacted_history_count": int(
                metadata.get("compacted_history_count") or 0
            ),
            "estimated_input_tokens": metadata.get("estimated_input_tokens"),
            "input_token_count_source": metadata.get("input_token_count_source"),
            "previous_prompt_tokens": metadata.get("previous_prompt_tokens"),
            "previous_completion_tokens": metadata.get("previous_completion_tokens"),
            "previous_total_tokens": metadata.get("previous_total_tokens"),
            "previous_cached_tokens": metadata.get("previous_cached_tokens"),
            "previous_reasoning_tokens": metadata.get("previous_reasoning_tokens"),
            "previous_cache_creation_input_tokens": metadata.get(
                "previous_cache_creation_input_tokens"
            ),
            "previous_cache_read_input_tokens": metadata.get(
                "previous_cache_read_input_tokens"
            ),
            "previous_usage_source": metadata.get("previous_usage_source"),
            "previous_token_details": metadata.get("previous_token_details") or {},
            "summary_strategy": summary_strategy,
            "summary_action": summary_action,
            "fallback_reason": (
                result.fallback_report.reason if result.fallback_report else None
            ),
        }

    def to_setup_context_report(
        self,
        *,
        result: ContextOperationResult,
        raw_history_count: int,
        raw_history_chars: int,
        user_edit_delta_count: int,
        prior_stage_handoff_count: int,
        context_profile: Literal["standard", "compact"],
        profile_reasons: Sequence[str],
    ) -> SetupContextGovernanceReport:
        """Build the transient setup context report from common trace data."""

        metadata = self.to_setup_governance_metadata(result)
        summary = self.to_setup_compact_summary(result.artifact)
        return SetupContextGovernanceReport(
            context_profile=context_profile,
            profile_reasons=list(profile_reasons),
            raw_history_count=raw_history_count,
            raw_history_chars=raw_history_chars,
            estimated_input_tokens=metadata.get("estimated_input_tokens"),
            input_token_count_source=metadata.get("input_token_count_source"),
            previous_prompt_tokens=metadata.get("previous_prompt_tokens"),
            previous_completion_tokens=metadata.get("previous_completion_tokens"),
            previous_total_tokens=metadata.get("previous_total_tokens"),
            previous_cached_tokens=metadata.get("previous_cached_tokens"),
            previous_reasoning_tokens=metadata.get("previous_reasoning_tokens"),
            previous_cache_creation_input_tokens=metadata.get(
                "previous_cache_creation_input_tokens"
            ),
            previous_cache_read_input_tokens=metadata.get(
                "previous_cache_read_input_tokens"
            ),
            previous_usage_source=metadata.get("previous_usage_source"),
            previous_token_details=dict(metadata.get("previous_token_details") or {}),
            user_edit_delta_count=user_edit_delta_count,
            prior_stage_handoff_count=prior_stage_handoff_count,
            raw_history_limit=int(metadata["raw_history_limit"]),
            kept_history_count=int(metadata["kept_history_count"]),
            compacted_history_count=int(metadata["compacted_history_count"]),
            retained_tool_outcome_count=int(
                result.trace.metadata.get("retained_tool_outcome_count") or 0
            ),
            summary_strategy=cast(
                Literal[
                    "none", "deterministic_prefix_summary", "compact_prompt_summary"
                ],
                metadata["summary_strategy"],
            ),
            summary_action=cast(
                Literal["none", "reused_existing", "updated_existing", "rebuilt"],
                metadata["summary_action"],
            ),
            summary_line_count=len(summary.summary_lines) if summary else 0,
            fallback_reason=metadata.get("fallback_reason"),
        )

    @staticmethod
    def _source_scope(*, current_step: str, current_stage: str | None) -> str:
        if current_stage:
            return f"setup_stage:{current_stage}"
        return f"setup_step:{current_step}"

    @staticmethod
    def _summary_refs(summary: SetupContextCompactSummary) -> list[str]:
        refs: list[str] = []
        for ref in summary.draft_refs:
            if ref not in refs:
                refs.append(ref)
        for hint in summary.recovery_hints:
            if hint.ref not in refs:
                refs.append(hint.ref)
        return refs

    @staticmethod
    def _setup_validation_policy():
        return default_validation_policy(
            schema_id="setup_context_compact_summary.v1",
            allowed_recovery_ref_prefixes=["draft:", "foundation:", "stage:"],
            forbidden_payload_fields=[
                "tool_calls",
                "draft_writes",
                "workspace_patch",
                "prior_stage_raw_discussion",
                "analysis",
                "scratchpad",
            ],
            max_list_lengths={
                "summary_lines": 6,
                "confirmed_points": 8,
                "open_threads": 4,
                "rejected_directions": 4,
                "draft_refs": 6,
                "recovery_hints": 6,
                "must_not_infer": 4,
            },
            metadata={"allowed_payload_fields": list(_SETUP_ALLOWED_PAYLOAD_FIELDS)},
        )

    @staticmethod
    def _setup_summary_action(
        result: ContextOperationResult,
    ) -> str:
        action = result.trace.summary_action or result.status
        if action == "not_needed" or result.artifact is None:
            return "none"
        if action == "reused":
            return "reused_existing"
        if action == "updated":
            return "updated_existing"
        return "rebuilt"

    @staticmethod
    def _setup_summary_strategy(
        result: ContextOperationResult,
    ) -> str:
        if result.artifact is None:
            return "none"
        if result.artifact.created_by == "model":
            return "compact_prompt_summary"
        return "deterministic_prefix_summary"

    @classmethod
    def validate_setup_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> ContextValidationReport:
        """Validate setup compact payload fields and setup-owned recovery refs."""

        policy = cls._setup_validation_policy()
        report = validate_payload_against_policy(payload=payload, policy=policy)
        _, ref_issues = filter_allowed_recovery_refs(
            refs=cls._payload_recovery_refs(payload),
            policy=policy,
        )
        if not ref_issues:
            return report
        return ContextValidationReport(
            valid=False,
            issues=[*report.issues, *ref_issues],
        )

    @staticmethod
    def _payload_recovery_refs(payload: Mapping[str, Any]) -> list[str]:
        refs: list[str] = []
        for field in ("draft_refs", "recovery_refs"):
            value = payload.get(field)
            if isinstance(value, list):
                refs.extend(str(item) for item in value)
        hints = payload.get("recovery_hints")
        if isinstance(hints, list):
            for item in hints:
                if isinstance(item, Mapping) and item.get("ref") is not None:
                    refs.append(str(item.get("ref")))
        return refs


def setup_recovery_hint(
    ref: str, reason: str, detail: str | None = None
) -> dict[str, Any]:
    """Small test/helper adapter shape for setup recovery hints."""

    return SetupCompactRecoveryHint(
        ref=ref,
        reason=reason,
        detail=detail,
    ).model_dump(mode="json", exclude_none=True)
