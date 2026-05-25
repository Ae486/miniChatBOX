"""Current-step context governor for setup runtime-v2 turns."""

from __future__ import annotations

from typing import Any

from rp.agent_runtime.contracts import (
    SetupContextCompactSummary,
    SetupCognitiveStateSnapshot,
    SetupCognitiveStateSummary,
    SetupToolOutcome,
    SetupWorkingDigest,
)
from rp.context_engineering.adapters.setup import SetupContextEngineeringAdapter
from rp.context_engineering.contracts import ContextSelectionResult
from rp.context_engineering.selection import select_context_sections
from rp.models.setup_agent import SetupAgentDialogueMessage
from rp.services.setup_context_compaction_service import SetupContextCompactionService


class SetupContextGovernorService:
    """Select prompt-visible step-local context without widening durable state."""

    _MAX_RETAINED_TOOL_OUTCOMES = 6
    _MAX_OPEN_QUESTIONS = 4
    _MAX_REJECTED_DIRECTIONS = 4
    _MAX_DRAFT_REFS = 6
    _MAX_COMMIT_BLOCKERS = 4
    _SUCCESS_RELEVANCE = {"cognitive", "draft", "question", "proposal", "asset"}

    def __init__(
        self,
        *,
        compaction_service: SetupContextCompactionService | None = None,
        context_adapter: SetupContextEngineeringAdapter | None = None,
    ) -> None:
        self._compaction_service = compaction_service or SetupContextCompactionService()
        self._context_adapter = context_adapter or SetupContextEngineeringAdapter()

    def govern_history(
        self,
        *,
        history: list[SetupAgentDialogueMessage],
        retained_tool_outcomes: list[SetupToolOutcome],
        working_digest: SetupWorkingDigest | None,
        existing_summary: SetupContextCompactSummary | None,
        context_profile: str,
        current_step: str | None = None,
        current_stage: str | None = None,
        estimated_input_tokens: int | None = None,
        input_token_count_source: str | None = None,
        previous_usage: dict[str, Any] | None = None,
    ) -> tuple[
        list[SetupAgentDialogueMessage],
        SetupContextCompactSummary | None,
        dict[str, Any],
    ]:
        return self._govern_history_sync(
            history=history,
            retained_tool_outcomes=retained_tool_outcomes,
            working_digest=working_digest,
            existing_summary=existing_summary,
            context_profile=context_profile,
            current_step=current_step,
            current_stage=current_stage,
            estimated_input_tokens=estimated_input_tokens,
            input_token_count_source=input_token_count_source,
            previous_usage=previous_usage,
        )

    async def govern_history_async(
        self,
        *,
        history: list[SetupAgentDialogueMessage],
        retained_tool_outcomes: list[SetupToolOutcome],
        working_digest: SetupWorkingDigest | None,
        existing_summary: SetupContextCompactSummary | None,
        context_profile: str,
        current_step: str | None = None,
        current_stage: str | None = None,
        estimated_input_tokens: int | None = None,
        input_token_count_source: str | None = None,
        previous_usage: dict[str, Any] | None = None,
    ) -> tuple[
        list[SetupAgentDialogueMessage],
        SetupContextCompactSummary | None,
        dict[str, Any],
    ]:
        return await self._govern_history_async(
            history=history,
            retained_tool_outcomes=retained_tool_outcomes,
            working_digest=working_digest,
            existing_summary=existing_summary,
            context_profile=context_profile,
            current_step=current_step,
            current_stage=current_stage,
            estimated_input_tokens=estimated_input_tokens,
            input_token_count_source=input_token_count_source,
            previous_usage=previous_usage,
        )

    def _govern_history_sync(
        self,
        *,
        history: list[SetupAgentDialogueMessage],
        retained_tool_outcomes: list[SetupToolOutcome],
        working_digest: SetupWorkingDigest | None,
        existing_summary: SetupContextCompactSummary | None,
        context_profile: str,
        current_step: str | None = None,
        current_stage: str | None = None,
        estimated_input_tokens: int | None = None,
        input_token_count_source: str | None = None,
        previous_usage: dict[str, Any] | None = None,
    ) -> tuple[
        list[SetupAgentDialogueMessage],
        SetupContextCompactSummary | None,
        dict[str, Any],
    ]:
        operation_request = self._context_adapter.build_stage_local_compact_request(
            history=history,
            retained_tool_outcomes=retained_tool_outcomes,
            working_digest=working_digest,
            existing_summary=existing_summary,
            context_profile=self._normalized_context_profile(context_profile),
            current_step=str(current_step or "unknown_step"),
            current_stage=current_stage,
            estimated_input_tokens=estimated_input_tokens,
            input_token_count_source=input_token_count_source,
            previous_usage=previous_usage,
        )
        selection = select_context_sections(operation_request)
        compact_summary = self._compaction_service.build_summary_from_common_selection(
            request=operation_request,
            selection=selection,
            retained_tool_outcomes=retained_tool_outcomes,
            working_digest=working_digest,
            existing_summary=existing_summary,
            current_step=current_step,
        )
        summary_decision = self._compaction_service.last_summary_decision()
        kept_history = self._history_from_recent_raw(
            history=history,
            selection=selection,
        )
        limit = int(operation_request.metadata.get("raw_history_limit") or 0)
        previous_prompt_tokens = (
            previous_usage.get("prompt_tokens") if previous_usage else None
        )
        previous_total_tokens = (
            previous_usage.get("total_tokens") if previous_usage else None
        )
        previous_token_details = (
            previous_usage.get("token_details") if previous_usage else None
        )
        return (
            kept_history,
            compact_summary,
            {
                "raw_history_limit": limit,
                "kept_history_count": len(kept_history),
                "compacted_history_count": len(selection.compactable_dropped_items),
                "estimated_input_tokens": estimated_input_tokens,
                "input_token_count_source": input_token_count_source,
                "previous_prompt_tokens": (
                    int(previous_prompt_tokens)
                    if previous_prompt_tokens is not None
                    else None
                ),
                "previous_total_tokens": (
                    int(previous_total_tokens)
                    if previous_total_tokens is not None
                    else None
                ),
                "previous_completion_tokens": self._optional_int(
                    previous_usage.get("completion_tokens") if previous_usage else None
                ),
                "previous_cached_tokens": self._optional_int(
                    previous_usage.get("cached_tokens") if previous_usage else None
                ),
                "previous_reasoning_tokens": self._optional_int(
                    previous_usage.get("reasoning_tokens") if previous_usage else None
                ),
                "previous_cache_creation_input_tokens": self._optional_int(
                    previous_usage.get("cache_creation_input_tokens")
                    if previous_usage
                    else None
                ),
                "previous_cache_read_input_tokens": self._optional_int(
                    previous_usage.get("cache_read_input_tokens")
                    if previous_usage
                    else None
                ),
                "previous_usage_source": (
                    previous_usage.get("source") if previous_usage else None
                ),
                "previous_token_details": (
                    dict(previous_token_details)
                    if isinstance(previous_token_details, dict)
                    else {}
                ),
                "summary_strategy": summary_decision.get("summary_strategy") or "none",
                "summary_action": summary_decision.get("summary_action") or "none",
                "fallback_reason": summary_decision.get("fallback_reason"),
            },
        )

    async def _govern_history_async(
        self,
        *,
        history: list[SetupAgentDialogueMessage],
        retained_tool_outcomes: list[SetupToolOutcome],
        working_digest: SetupWorkingDigest | None,
        existing_summary: SetupContextCompactSummary | None,
        context_profile: str,
        current_step: str | None = None,
        current_stage: str | None = None,
        estimated_input_tokens: int | None = None,
        input_token_count_source: str | None = None,
        previous_usage: dict[str, Any] | None = None,
    ) -> tuple[
        list[SetupAgentDialogueMessage],
        SetupContextCompactSummary | None,
        dict[str, Any],
    ]:
        operation_request = self._context_adapter.build_stage_local_compact_request(
            history=history,
            retained_tool_outcomes=retained_tool_outcomes,
            working_digest=working_digest,
            existing_summary=existing_summary,
            context_profile=self._normalized_context_profile(context_profile),
            current_step=str(current_step or "unknown_step"),
            current_stage=current_stage,
            estimated_input_tokens=estimated_input_tokens,
            input_token_count_source=input_token_count_source,
            previous_usage=previous_usage,
        )
        selection = select_context_sections(operation_request)
        compact_summary = (
            await self._compaction_service.build_summary_from_common_selection_async(
                request=operation_request,
                selection=selection,
                retained_tool_outcomes=retained_tool_outcomes,
                working_digest=working_digest,
                existing_summary=existing_summary,
                current_step=current_step,
            )
        )
        summary_decision = self._compaction_service.last_summary_decision()
        kept_history = self._history_from_recent_raw(
            history=history,
            selection=selection,
        )
        limit = int(operation_request.metadata.get("raw_history_limit") or 0)
        previous_prompt_tokens = (
            previous_usage.get("prompt_tokens") if previous_usage else None
        )
        previous_total_tokens = (
            previous_usage.get("total_tokens") if previous_usage else None
        )
        previous_token_details = (
            previous_usage.get("token_details") if previous_usage else None
        )
        return (
            kept_history,
            compact_summary,
            {
                "raw_history_limit": limit,
                "kept_history_count": len(kept_history),
                "compacted_history_count": len(selection.compactable_dropped_items),
                "estimated_input_tokens": estimated_input_tokens,
                "input_token_count_source": input_token_count_source,
                "previous_prompt_tokens": (
                    int(previous_prompt_tokens)
                    if previous_prompt_tokens is not None
                    else None
                ),
                "previous_total_tokens": (
                    int(previous_total_tokens)
                    if previous_total_tokens is not None
                    else None
                ),
                "previous_completion_tokens": self._optional_int(
                    previous_usage.get("completion_tokens") if previous_usage else None
                ),
                "previous_cached_tokens": self._optional_int(
                    previous_usage.get("cached_tokens") if previous_usage else None
                ),
                "previous_reasoning_tokens": self._optional_int(
                    previous_usage.get("reasoning_tokens") if previous_usage else None
                ),
                "previous_cache_creation_input_tokens": self._optional_int(
                    previous_usage.get("cache_creation_input_tokens")
                    if previous_usage
                    else None
                ),
                "previous_cache_read_input_tokens": self._optional_int(
                    previous_usage.get("cache_read_input_tokens")
                    if previous_usage
                    else None
                ),
                "previous_usage_source": (
                    previous_usage.get("source") if previous_usage else None
                ),
                "previous_token_details": (
                    dict(previous_token_details)
                    if isinstance(previous_token_details, dict)
                    else {}
                ),
                "summary_strategy": summary_decision.get("summary_strategy") or "none",
                "summary_action": summary_decision.get("summary_action") or "none",
                "fallback_reason": summary_decision.get("fallback_reason"),
            },
        )

    @staticmethod
    def _normalized_context_profile(context_profile: str):
        return "compact" if context_profile == "compact" else "standard"

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _history_from_recent_raw(
        *,
        history: list[SetupAgentDialogueMessage],
        selection: ContextSelectionResult,
    ) -> list[SetupAgentDialogueMessage]:
        indices: list[int] = []
        for item in selection.recent_raw_items:
            if item.sequence_index is None:
                continue
            index = int(item.sequence_index)
            if 0 <= index < len(history):
                indices.append(index)
        return [history[index] for index in sorted(dict.fromkeys(indices))]

    def build_initial_digest(
        self,
        *,
        cognitive_state: SetupCognitiveStateSnapshot | None,
        cognitive_state_summary: SetupCognitiveStateSummary | None,
        blocking_open_question_count: int,
        last_proposal_status: str | None,
    ) -> SetupWorkingDigest | None:
        base = (
            cognitive_state_summary.working_digest.model_copy(deep=True)
            if cognitive_state_summary is not None
            and cognitive_state_summary.working_digest is not None
            else SetupWorkingDigest()
        )
        discussion_state = (
            cognitive_state.discussion_state if cognitive_state is not None else None
        )
        truth_write = (
            cognitive_state.active_truth_write if cognitive_state is not None else None
        )
        if discussion_state is not None:
            base.next_focus = discussion_state.next_focus or base.next_focus
            base.rejected_directions = [
                item.label
                for item in discussion_state.candidate_directions
                if item.status == "discarded"
            ][: self._MAX_REJECTED_DIRECTIONS]
        if (
            cognitive_state_summary is not None
            and cognitive_state_summary.open_questions
        ):
            base.open_questions = list(
                cognitive_state_summary.open_questions[: self._MAX_OPEN_QUESTIONS]
            )
        draft_refs = list(base.draft_refs[: self._MAX_DRAFT_REFS])
        if truth_write is not None and truth_write.target_ref:
            target_ref = str(truth_write.target_ref)
            if target_ref not in draft_refs:
                draft_refs.append(target_ref)
        if cognitive_state_summary is not None:
            for outcome in cognitive_state_summary.tool_outcomes:
                for ref in outcome.updated_refs:
                    value = str(ref or "").strip()
                    if value and value not in draft_refs:
                        draft_refs.append(value)
                    if len(draft_refs) >= self._MAX_DRAFT_REFS:
                        break
                if len(draft_refs) >= self._MAX_DRAFT_REFS:
                    break
        base.draft_refs = draft_refs[: self._MAX_DRAFT_REFS]

        blockers = list(base.commit_blockers[: self._MAX_COMMIT_BLOCKERS])
        if blocking_open_question_count > 0:
            blockers.append(f"{blocking_open_question_count} blocking_open_question(s)")
        if cognitive_state_summary is not None and cognitive_state_summary.invalidated:
            blockers.append("cognitive_state_invalidated")
        if cognitive_state_summary is not None:
            blockers.extend(cognitive_state_summary.remaining_open_issues[:2])
        if str(last_proposal_status or "").lower() == "rejected":
            blockers.append("proposal_rejected")
        base.commit_blockers = list(dict.fromkeys(blockers))[
            : self._MAX_COMMIT_BLOCKERS
        ]

        has_content = any(
            (
                base.current_goal,
                base.next_focus,
                base.pending_obligation,
                base.open_questions,
                base.rejected_directions,
                base.draft_refs,
                base.commit_blockers,
            )
        )
        return base if has_content else None

    def retain_tool_outcomes(
        self,
        *,
        existing: list[SetupToolOutcome],
        latest_results: list[SetupToolOutcome] | None = None,
    ) -> list[SetupToolOutcome]:
        combined = [*existing, *(latest_results or [])]
        if not combined:
            return []

        failures: list[SetupToolOutcome] = []
        successes: list[SetupToolOutcome] = []
        seen_failure_keys: set[str] = set()
        seen_success_keys: set[str] = set()

        for item in reversed(combined):
            if not item.success:
                key = f"failure:{item.tool_name}:{item.error_code or ''}:{item.summary}"
                if key in seen_failure_keys:
                    continue
                seen_failure_keys.add(key)
                failures.append(item)
                continue
            if item.relevance not in self._SUCCESS_RELEVANCE:
                continue
            refs_key = ",".join(item.updated_refs)
            key = f"success:{item.tool_name}:{refs_key}"
            if key in seen_success_keys:
                continue
            seen_success_keys.add(key)
            successes.append(item)

        ordered = [*failures, *successes]
        return ordered[: self._MAX_RETAINED_TOOL_OUTCOMES]
