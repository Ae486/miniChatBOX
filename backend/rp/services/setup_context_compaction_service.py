"""Setup compact prompt runner and summary conversion helpers."""

from __future__ import annotations

import asyncio
import inspect
import json
import re
from collections.abc import Awaitable, Callable
from typing import Any

from models.chat import ChatMessage
from rp.agent_runtime.contracts import (
    SetupCompactRecoveryHint,
    SetupContextCompactSummary,
    SetupToolOutcome,
    SetupWorkingDigest,
)
from rp.agent_runtime.prompts.setup_agent import compact_prompt_system_prompt
from rp.context_engineering.compaction import (
    CompactPromptRunner,
    ContextCompactPromptFailure,
    run_compact_operation,
)
from rp.context_engineering.adapters.setup import SetupContextEngineeringAdapter
from rp.context_engineering.contracts import (
    ContextCompactPromptRequest,
    ContextOperationRequest,
    ContextOperationResult,
    ContextSelectionResult,
    ContextSourceItem,
)
from rp.context_engineering.fingerprinting import fingerprint_source_items
from rp.models.setup_agent import SetupAgentDialogueMessage

CompactPromptProvider = Callable[
    [list[ChatMessage]], dict[str, Any] | Awaitable[dict[str, Any]]
]


class SetupContextCompactionService:
    """Bridge setup compaction to the common kernel and setup prompt pass."""

    _SUMMARY_LINE_LIMIT = 6
    _CONFIRMED_POINT_LIMIT = 8
    _OPEN_THREAD_LIMIT = 4
    _REJECTED_DIRECTION_LIMIT = 4
    _DRAFT_REF_LIMIT = 6
    _RECOVERY_HINT_LIMIT = 6
    _MUST_NOT_INFER_LIMIT = 4
    _MAX_SUMMARY_CHARS = 180
    _VALID_DRAFT_REF_PREFIXES = ("draft:", "foundation:", "stage:")

    def __init__(
        self,
        *,
        compact_prompt_provider: CompactPromptProvider | None = None,
    ) -> None:
        self._compact_prompt_provider = compact_prompt_provider
        self._setup_context_adapter = SetupContextEngineeringAdapter()
        self._last_summary_decision: dict[str, Any] = {
            "summary_strategy": "none",
            "summary_action": "none",
            "fallback_reason": None,
        }

    def last_summary_decision(self) -> dict[str, Any]:
        """Return metadata for the most recent compact-summary build."""

        return dict(self._last_summary_decision)

    def build_summary_from_common_selection(
        self,
        *,
        request: ContextOperationRequest,
        selection: ContextSelectionResult,
        retained_tool_outcomes: list[SetupToolOutcome],
        working_digest: SetupWorkingDigest | None,
        existing_summary: SetupContextCompactSummary | None,
        current_step: str | None = None,
    ) -> SetupContextCompactSummary | None:
        """Build a setup summary from common selection and compaction mechanics."""

        result = asyncio.run(
            self._run_common_compact_operation(
                request=request,
                selection=selection,
                retained_tool_outcomes=retained_tool_outcomes,
                working_digest=working_digest,
                current_step=current_step,
                allow_async_provider=False,
            )
        )
        return self._summary_from_common_result(
            result=result,
            dropped_items=selection.compactable_dropped_items,
            retained_tool_outcomes=retained_tool_outcomes,
            working_digest=working_digest,
            existing_summary=existing_summary,
            provider_configured=self._compact_prompt_provider is not None,
        )

    async def build_summary_from_common_selection_async(
        self,
        *,
        request: ContextOperationRequest,
        selection: ContextSelectionResult,
        retained_tool_outcomes: list[SetupToolOutcome],
        working_digest: SetupWorkingDigest | None,
        existing_summary: SetupContextCompactSummary | None,
        current_step: str | None = None,
    ) -> SetupContextCompactSummary | None:
        """Async variant used by runtime-v2 setup launches."""

        result = await self._run_common_compact_operation(
            request=request,
            selection=selection,
            retained_tool_outcomes=retained_tool_outcomes,
            working_digest=working_digest,
            current_step=current_step,
            allow_async_provider=True,
        )
        return self._summary_from_common_result(
            result=result,
            dropped_items=selection.compactable_dropped_items,
            retained_tool_outcomes=retained_tool_outcomes,
            working_digest=working_digest,
            existing_summary=existing_summary,
            provider_configured=self._compact_prompt_provider is not None,
        )

    async def _run_common_compact_operation(
        self,
        *,
        request: ContextOperationRequest,
        selection: ContextSelectionResult,
        retained_tool_outcomes: list[SetupToolOutcome],
        working_digest: SetupWorkingDigest | None,
        current_step: str | None,
        allow_async_provider: bool,
    ) -> ContextOperationResult:
        runner = (
            _SetupCompactPromptRunner(
                service=self,
                all_dropped_items=list(selection.compactable_dropped_items),
                retained_tool_outcomes=retained_tool_outcomes,
                working_digest=working_digest,
                current_step=str(current_step or "unknown_step"),
                allow_async_provider=allow_async_provider,
            )
            if self._compact_prompt_provider is not None
            else None
        )
        first_kept_source_item_id = (
            selection.recent_raw_items[0].source_item_id
            if selection.recent_raw_items
            else None
        )
        return await run_compact_operation(
            request=request,
            dropped_items=selection.compactable_dropped_items,
            first_kept_source_item_id=first_kept_source_item_id,
            compact_prompt_runner=runner,
        )

    def _summary_from_common_result(
        self,
        *,
        result: ContextOperationResult,
        dropped_items: list[ContextSourceItem],
        retained_tool_outcomes: list[SetupToolOutcome],
        working_digest: SetupWorkingDigest | None,
        existing_summary: SetupContextCompactSummary | None,
        provider_configured: bool,
    ) -> SetupContextCompactSummary | None:
        summary_action = str(result.trace.summary_action or result.status)
        setup_summary_action = self._setup_summary_action(summary_action=summary_action)
        if result.status == "not_needed" or not dropped_items:
            self._last_summary_decision = {
                "summary_strategy": "none",
                "summary_action": "none",
                "fallback_reason": None,
            }
            return None
        if result.status == "reused" and existing_summary is not None:
            self._last_summary_decision = {
                "summary_strategy": "deterministic_prefix_summary",
                "summary_action": "reused_existing",
                "fallback_reason": None,
            }
            return existing_summary

        fallback_reason = (
            result.fallback_report.reason
            if result.fallback_report is not None
            else None
        )
        if (
            fallback_reason == "compact_prompt_runner_unavailable"
            and not provider_configured
        ):
            fallback_reason = None
        if result.artifact is not None and result.artifact.created_by == "model":
            summary = self._setup_context_adapter.to_setup_compact_summary(
                result.artifact
            )
            if summary is None:
                raise ValueError("compact_prompt_summary_missing_artifact")
            self._last_summary_decision = {
                "summary_strategy": "compact_prompt_summary",
                "summary_action": setup_summary_action,
                "fallback_reason": None,
            }
            return summary

        dropped_history = self._history_from_source_items(dropped_items)
        fingerprint = (
            result.artifact.source_fingerprint
            if result.artifact
            else fingerprint_source_items(dropped_items)
        )
        if setup_summary_action == "updated_existing" and existing_summary is not None:
            newly_compacted_history = self._newly_compacted_history_from_source_items(
                dropped_items=dropped_items,
                existing_summary=existing_summary,
            )
            summary = self._deterministic_incremental_summary(
                existing_summary=existing_summary,
                newly_compacted_history=newly_compacted_history,
                fingerprint=fingerprint,
                source_message_count=len(dropped_items),
                retained_tool_outcomes=retained_tool_outcomes,
                working_digest=working_digest,
            )
        else:
            summary = self._deterministic_summary(
                dropped_history=dropped_history,
                fingerprint=fingerprint,
                retained_tool_outcomes=retained_tool_outcomes,
                working_digest=working_digest,
            )
        self._last_summary_decision = {
            "summary_strategy": "deterministic_prefix_summary",
            "summary_action": setup_summary_action,
            "fallback_reason": fallback_reason,
        }
        return summary

    @staticmethod
    def _setup_summary_action(*, summary_action: str) -> str:
        if summary_action == "not_needed":
            return "none"
        if summary_action == "reused":
            return "reused_existing"
        if summary_action == "updated":
            return "updated_existing"
        return "rebuilt"

    def build_compact_prompt(
        self,
        *,
        dropped_history: list[SetupAgentDialogueMessage],
        existing_summary: SetupContextCompactSummary | None,
        working_digest: SetupWorkingDigest | None,
        retained_tool_outcomes: list[SetupToolOutcome],
        current_step: str,
        draft_refs: list[str],
        newly_compacted_history: list[SetupAgentDialogueMessage] | None = None,
        source_fingerprint: str | None = None,
        source_message_count: int | None = None,
    ) -> list[ChatMessage]:
        """Build the no-tools compact prompt pass for one setup stage."""

        incremental_update = (
            newly_compacted_history is not None and existing_summary is not None
        )
        newly_compacted_messages = (
            [
                item.model_dump(mode="json", exclude_none=True)
                for item in newly_compacted_history
            ]
            if incremental_update and newly_compacted_history is not None
            else None
        )
        payload = {
            "current_step": current_step,
            "dropped_current_step_fingerprint": source_fingerprint
            or fingerprint_source_items(self._source_items_from_history(dropped_history)),
            "dropped_current_step_message_count": source_message_count
            if source_message_count is not None
            else len(dropped_history),
            "incremental_update": incremental_update,
            "dropped_current_step_messages": (
                None
                if incremental_update
                else [
                    item.model_dump(mode="json", exclude_none=True)
                    for item in dropped_history
                ]
            ),
            "newly_compacted_current_step_messages": newly_compacted_messages,
            "previous_compact_summary": (
                existing_summary.model_dump(mode="json", exclude_none=True)
                if existing_summary is not None
                else None
            ),
            "working_digest": (
                working_digest.model_dump(mode="json", exclude_none=True)
                if working_digest is not None
                else None
            ),
            "retained_tool_outcomes": [
                item.model_dump(mode="json", exclude_none=True)
                for item in retained_tool_outcomes[:6]
            ],
            "draft_refs": list(draft_refs[: self._DRAFT_REF_LIMIT]),
        }
        return [
            ChatMessage(
                role="system",
                content=compact_prompt_system_prompt(),
            ),
            ChatMessage(
                role="user",
                content=json.dumps(payload, ensure_ascii=False, sort_keys=True),
            ),
        ]

    def validate_compact_prompt_summary(
        self,
        *,
        payload: dict[str, Any],
        source_fingerprint: str,
        source_message_count: int,
    ) -> SetupContextCompactSummary:
        """Validate and cap compact prompt-pass output before persistence."""

        if not isinstance(payload, dict):
            raise ValueError("compact_prompt_summary_not_json_object")
        allowed_fields = {
            "source_fingerprint",
            "source_message_count",
            "summary_lines",
            "confirmed_points",
            "open_threads",
            "rejected_directions",
            "draft_refs",
            "recovery_hints",
            "must_not_infer",
        }
        extra_fields = sorted(set(payload) - allowed_fields)
        if extra_fields:
            raise ValueError(
                f"compact_prompt_summary_forbidden_fields:{','.join(extra_fields)}"
            )
        payload_fingerprint = payload.get("source_fingerprint")
        if (
            payload_fingerprint is not None
            and str(payload_fingerprint) != source_fingerprint
        ):
            raise ValueError("compact_prompt_summary_source_fingerprint_mismatch")
        payload_message_count = payload.get("source_message_count")
        if payload_message_count is not None:
            try:
                count_matches = int(payload_message_count) == int(source_message_count)
            except (TypeError, ValueError):
                count_matches = False
            if not count_matches:
                raise ValueError("compact_prompt_summary_source_message_count_mismatch")

        draft_refs = self._string_list(
            payload.get("draft_refs"),
            limit=self._DRAFT_REF_LIMIT,
        )
        self._ensure_supported_refs(draft_refs)
        recovery_hints = self._recovery_hints(payload.get("recovery_hints"))
        self._ensure_supported_refs([item.ref for item in recovery_hints])
        return SetupContextCompactSummary(
            source_fingerprint=source_fingerprint,
            source_message_count=source_message_count,
            summary_lines=self._string_list(
                payload.get("summary_lines"),
                limit=self._SUMMARY_LINE_LIMIT,
            ),
            confirmed_points=self._string_list(
                payload.get("confirmed_points"),
                limit=self._CONFIRMED_POINT_LIMIT,
            ),
            open_threads=self._string_list(
                payload.get("open_threads"),
                limit=self._OPEN_THREAD_LIMIT,
            ),
            rejected_directions=self._string_list(
                payload.get("rejected_directions"),
                limit=self._REJECTED_DIRECTION_LIMIT,
            ),
            draft_refs=draft_refs,
            recovery_hints=recovery_hints,
            must_not_infer=self._string_list(
                payload.get("must_not_infer"),
                limit=self._MUST_NOT_INFER_LIMIT,
            ),
        )

    def _deterministic_summary(
        self,
        *,
        dropped_history: list[SetupAgentDialogueMessage],
        fingerprint: str,
        retained_tool_outcomes: list[SetupToolOutcome],
        working_digest: SetupWorkingDigest | None,
    ) -> SetupContextCompactSummary:
        draft_refs = self._draft_refs(
            retained_tool_outcomes=retained_tool_outcomes,
            working_digest=working_digest,
        )
        return SetupContextCompactSummary(
            source_fingerprint=fingerprint,
            source_message_count=len(dropped_history),
            summary_lines=self._summary_lines(dropped_history),
            open_threads=list(
                (working_digest.open_questions if working_digest else [])[
                    : self._OPEN_THREAD_LIMIT
                ]
            ),
            rejected_directions=list(
                (working_digest.rejected_directions if working_digest else [])[
                    : self._REJECTED_DIRECTION_LIMIT
                ]
            ),
            draft_refs=draft_refs,
            recovery_hints=[
                SetupCompactRecoveryHint(
                    ref=ref,
                    reason="recover_exact_draft_detail",
                    detail="Use setup.memory.open if exact setup fact detail is needed.",
                )
                for ref in draft_refs[: self._RECOVERY_HINT_LIMIT]
            ],
        )

    def _deterministic_incremental_summary(
        self,
        *,
        existing_summary: SetupContextCompactSummary,
        newly_compacted_history: list[SetupAgentDialogueMessage],
        fingerprint: str,
        source_message_count: int,
        retained_tool_outcomes: list[SetupToolOutcome],
        working_digest: SetupWorkingDigest | None,
    ) -> SetupContextCompactSummary:
        draft_refs = self._draft_refs(
            retained_tool_outcomes=retained_tool_outcomes,
            working_digest=working_digest,
        )
        merged_draft_refs = self._merge_string_lists(
            existing_summary.draft_refs,
            draft_refs,
            limit=self._DRAFT_REF_LIMIT,
        )
        return SetupContextCompactSummary(
            source_fingerprint=fingerprint,
            source_message_count=source_message_count,
            summary_lines=self._merge_string_lists(
                existing_summary.summary_lines,
                self._summary_lines(newly_compacted_history),
                limit=self._SUMMARY_LINE_LIMIT,
            ),
            confirmed_points=self._merge_string_lists(
                existing_summary.confirmed_points,
                [],
                limit=self._CONFIRMED_POINT_LIMIT,
            ),
            open_threads=self._merge_string_lists(
                existing_summary.open_threads,
                list(
                    (working_digest.open_questions if working_digest else [])[
                        : self._OPEN_THREAD_LIMIT
                    ]
                ),
                limit=self._OPEN_THREAD_LIMIT,
            ),
            rejected_directions=self._merge_string_lists(
                existing_summary.rejected_directions,
                list(
                    (working_digest.rejected_directions if working_digest else [])[
                        : self._REJECTED_DIRECTION_LIMIT
                    ]
                ),
                limit=self._REJECTED_DIRECTION_LIMIT,
            ),
            draft_refs=merged_draft_refs,
            recovery_hints=self._merge_recovery_hints(
                existing_summary.recovery_hints,
                merged_draft_refs,
            ),
            must_not_infer=self._merge_string_lists(
                existing_summary.must_not_infer,
                [],
                limit=self._MUST_NOT_INFER_LIMIT,
            ),
        )

    def _newly_compacted_history_from_source_items(
        self,
        *,
        dropped_items: list[ContextSourceItem],
        existing_summary: SetupContextCompactSummary,
    ) -> list[SetupAgentDialogueMessage]:
        previous_count = int(existing_summary.source_message_count or 0)
        if previous_count <= 0 or previous_count >= len(dropped_items):
            return []
        return self._history_from_source_items(dropped_items[previous_count:])

    @staticmethod
    def _history_from_source_items(
        items: list[ContextSourceItem],
    ) -> list[SetupAgentDialogueMessage]:
        history: list[SetupAgentDialogueMessage] = []
        for item in items:
            role = "assistant" if item.source_family == "assistant_turn" else "user"
            history.append(
                SetupAgentDialogueMessage(
                    role=role,
                    content=str(item.text or ""),
                )
            )
        return history

    @staticmethod
    def _source_items_from_history(
        history: list[SetupAgentDialogueMessage],
    ) -> list[ContextSourceItem]:
        return [
            ContextSourceItem(
                source_item_id=f"setup.history.{index}",
                source_family="user_turn"
                if item.role == "user"
                else "assistant_turn",
                sequence_index=index,
                text=item.content,
            )
            for index, item in enumerate(history)
        ]

    def _summary_lines(
        self,
        history: list[SetupAgentDialogueMessage],
    ) -> list[str]:
        lines: list[str] = []
        for item in history[: self._SUMMARY_LINE_LIMIT]:
            text = self._sanitize_summary_text(self._normalize_text(item.content))
            if not text:
                continue
            prefix = "User" if item.role == "user" else "Assistant"
            line = f"{prefix}: {text[: self._MAX_SUMMARY_CHARS]}"
            if line not in lines:
                lines.append(line)
        return lines

    def _draft_refs(
        self,
        *,
        retained_tool_outcomes: list[SetupToolOutcome],
        working_digest: SetupWorkingDigest | None,
    ) -> list[str]:
        refs: list[str] = []
        if working_digest is not None:
            for ref in working_digest.draft_refs:
                value = str(ref or "").strip()
                if value and self._is_supported_ref(value) and value not in refs:
                    refs.append(value)
                if len(refs) >= self._DRAFT_REF_LIMIT:
                    return refs
        for item in retained_tool_outcomes:
            for ref in item.updated_refs:
                value = str(ref or "").strip()
                if value and self._is_supported_ref(value) and value not in refs:
                    refs.append(value)
                if len(refs) >= self._DRAFT_REF_LIMIT:
                    return refs
        return refs

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "")).strip()

    @staticmethod
    def _sanitize_summary_text(text: str) -> str:
        return re.sub(
            r"\b[A-Z]{2,}(?:_[A-Z]{2,}){2,}\b",
            lambda match: match.group(0).replace("_", " "),
            text,
        )

    def _string_list(self, value: Any, *, limit: int) -> list[str]:
        if not isinstance(value, list):
            return []
        items: list[str] = []
        for item in value:
            text = self._normalize_text(str(item or ""))
            if not text or text in items:
                continue
            items.append(text[: self._MAX_SUMMARY_CHARS])
            if len(items) >= limit:
                break
        return items

    def _merge_string_lists(
        self,
        existing: list[str],
        incoming: list[str],
        *,
        limit: int,
    ) -> list[str]:
        merged: list[str] = []
        for item in [*existing, *incoming]:
            text = self._normalize_text(str(item or ""))
            if not text or text in merged:
                continue
            merged.append(text[: self._MAX_SUMMARY_CHARS])
        if len(merged) <= limit:
            return merged
        return merged[:limit]

    def _recovery_hints(self, value: Any) -> list[SetupCompactRecoveryHint]:
        if not isinstance(value, list):
            return []
        hints: list[SetupCompactRecoveryHint] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            ref = self._normalize_text(str(item.get("ref") or ""))
            reason = self._normalize_text(str(item.get("reason") or ""))
            if not ref or not reason:
                continue
            detail = item.get("detail")
            hints.append(
                SetupCompactRecoveryHint(
                    ref=ref,
                    reason=reason[: self._MAX_SUMMARY_CHARS],
                    detail=(
                        self._normalize_text(str(detail))[: self._MAX_SUMMARY_CHARS]
                        if detail is not None
                        else None
                    ),
                )
            )
            if len(hints) >= self._RECOVERY_HINT_LIMIT:
                break
        return hints

    def _merge_recovery_hints(
        self,
        existing: list[SetupCompactRecoveryHint],
        refs: list[str],
    ) -> list[SetupCompactRecoveryHint]:
        hints = list(existing)
        seen_refs = {item.ref for item in hints}
        for ref in refs:
            if ref in seen_refs:
                continue
            hints.append(
                SetupCompactRecoveryHint(
                    ref=ref,
                    reason="recover_exact_draft_detail",
                    detail="Use setup.memory.open if exact setup fact detail is needed.",
                )
            )
            seen_refs.add(ref)
        if len(hints) <= self._RECOVERY_HINT_LIMIT:
            return hints
        return hints[: self._RECOVERY_HINT_LIMIT]

    def _ensure_supported_refs(self, refs: list[str]) -> None:
        unsupported = [ref for ref in refs if not self._is_supported_ref(ref)]
        if unsupported:
            raise ValueError(
                f"compact_prompt_summary_unsupported_refs:{','.join(unsupported)}"
            )

    def _is_supported_ref(self, ref: str) -> bool:
        value = str(ref or "").strip()
        return any(
            value.startswith(prefix) and bool(value.removeprefix(prefix).strip())
            for prefix in self._VALID_DRAFT_REF_PREFIXES
        )

    @staticmethod
    def _fallback_reason(exc: Exception) -> str:
        text = str(exc).strip()
        if not text:
            return exc.__class__.__name__
        return text[:160]


class _SetupCompactPromptRunner(CompactPromptRunner):
    """Bridge common data-only compaction requests into setup's prompt schema."""

    def __init__(
        self,
        *,
        service: SetupContextCompactionService,
        all_dropped_items: list[ContextSourceItem],
        retained_tool_outcomes: list[SetupToolOutcome],
        working_digest: SetupWorkingDigest | None,
        current_step: str,
        allow_async_provider: bool,
    ) -> None:
        self._service = service
        self._all_dropped_items = all_dropped_items
        self._retained_tool_outcomes = retained_tool_outcomes
        self._working_digest = working_digest
        self._current_step = current_step
        self._allow_async_provider = allow_async_provider

    async def run_compact_prompt(
        self,
        request: ContextCompactPromptRequest,
    ) -> dict[str, Any]:
        if self._service._compact_prompt_provider is None:
            raise ContextCompactPromptFailure("compact_prompt_provider_unavailable")
        dropped_history = self._service._history_from_source_items(
            self._all_dropped_items
        )
        newly_compacted_history = (
            self._service._history_from_source_items(list(request.dropped_items))
            if request.action == "updated"
            else None
        )
        existing_summary = (
            SetupContextCompactSummary.model_validate(request.previous_artifact_payload)
            if request.previous_artifact_payload is not None
            else None
        )
        draft_refs = self._service._draft_refs(
            retained_tool_outcomes=self._retained_tool_outcomes,
            working_digest=self._working_digest,
        )
        prompt_messages = self._service.build_compact_prompt(
            dropped_history=dropped_history,
            newly_compacted_history=newly_compacted_history,
            existing_summary=existing_summary,
            working_digest=self._working_digest,
            retained_tool_outcomes=self._retained_tool_outcomes,
            current_step=self._current_step,
            draft_refs=draft_refs,
            source_fingerprint=request.source_fingerprint,
            source_message_count=request.source_item_count,
        )
        try:
            payload = self._service._compact_prompt_provider(prompt_messages)
            if inspect.isawaitable(payload):
                if not self._allow_async_provider:
                    close = getattr(payload, "close", None)
                    if callable(close):
                        close()
                    raise ContextCompactPromptFailure(
                        "compact_prompt_provider_returned_awaitable"
                    )
                payload = await payload
            summary = self._service.validate_compact_prompt_summary(
                payload=payload,
                source_fingerprint=request.source_fingerprint,
                source_message_count=request.source_item_count,
            )
        except ContextCompactPromptFailure:
            raise
        except ValueError as exc:
            raise ContextCompactPromptFailure(
                self._service._fallback_reason(exc)
            ) from exc
        artifact_payload: dict[str, Any] = summary.model_dump(
            mode="json",
            exclude_none=True,
        )
        artifact_payload.pop("source_fingerprint", None)
        artifact_payload.pop("source_message_count", None)
        return artifact_payload
