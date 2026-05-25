"""Unified RP runtime executor backed by a LangGraph agent loop."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable, Literal, cast

from models.chat import ChatCompletionRequest, ChatMessage, ProviderConfig
from services.langfuse_service import get_langfuse_service

from .contracts import (
    RpAgentTurnInput,
    RpAgentTurnResult,
    RuntimeProfile,
    SetupContextCompactSummary,
    SetupEventSinkSnapshot,
    SetupModelGatewayDiagnostics,
    SetupOutputInspection,
    RuntimeToolCall,
    RuntimeToolResult,
    SetupCognitiveStateSnapshot,
    SetupCognitiveStateSummary,
    SetupPendingObligation,
    SetupReflectionTicket,
    SetupReActTraceFrame,
    SetupToolOutcome,
    SetupTurnGoal,
    SetupWorkingDigest,
    SetupWorkingPlan,
)
from .events import RuntimeEvent, TypedSseEventAdapter
from .graph import build_runtime_graph
from .policies import (
    CompletionGuardPolicy,
    FinishPolicy,
    ReflectionTriggerPolicy,
    RepairDecisionPolicy,
    ToolFailureClassifier,
)
from .prompts.setup_agent import runtime_overlay_instruction
from .state import RpAgentRunState
from .tools import RuntimeToolExecutor
from rp.services.setup_context_governor import SetupContextGovernorService

_SENTINEL = object()


class _ProviderStreamPayloadError(ValueError):
    """Provider stream emitted a malformed typed-SSE payload."""


class RpAgentRuntimeExecutor:
    """Execute one agent turn on the shared runtime graph."""

    def __init__(
        self,
        *,
        tool_executor_factory: Callable[[RpAgentTurnInput], RuntimeToolExecutor],
        event_adapter: TypedSseEventAdapter | None = None,
    ) -> None:
        self._tool_executor_factory = tool_executor_factory
        self._event_adapter = event_adapter or TypedSseEventAdapter()
        self._last_result: RpAgentTurnResult | None = None

    @property
    def last_result(self) -> RpAgentTurnResult | None:
        return self._last_result

    async def run(
        self,
        turn_input: RpAgentTurnInput,
        profile: RuntimeProfile,
        *,
        llm_service: Any,
    ) -> RpAgentTurnResult:
        driver = _RuntimeRunDriver(
            llm_service=llm_service,
            tool_executor=self._tool_executor_factory(turn_input),
            profile=profile,
        )
        result = await driver.run(turn_input)
        self._last_result = result
        return result

    async def run_stream(
        self,
        turn_input: RpAgentTurnInput,
        profile: RuntimeProfile,
        *,
        llm_service: Any,
    ) -> AsyncIterator[str]:
        driver = _RuntimeRunDriver(
            llm_service=llm_service,
            tool_executor=self._tool_executor_factory(turn_input),
            profile=profile,
        )
        async for event in driver.run_stream(turn_input):
            line = self._event_adapter.to_sse_line(event)
            if line is not None:
                yield line
        self._last_result = driver.last_result


class _RuntimeRunDriver:
    """Per-run LangGraph driver and node implementation."""

    _PSEUDO_TOOL_CALL_TEXT_RE = re.compile(
        r"(?:tool_code\b|print\s*\(\s*default_api\.|default_api\.rp_setup__setup\.|rp_setup__setup\.[a-z0-9_.]+\s*\()",
        re.IGNORECASE,
    )

    _INSPECT_ROUTES = {
        "execute_tools",
        "reflect_if_needed",
        "finalize_success",
        "finalize_failure",
    }
    _ASSESS_ROUTES = {
        "derive_turn_goal",
        "reflect_if_needed",
        "finalize_success",
        "finalize_failure",
    }
    _REFLECT_ROUTES = {
        "derive_turn_goal",
        "finalize_success",
        "finalize_failure",
    }

    def __init__(
        self,
        *,
        llm_service: Any,
        tool_executor: RuntimeToolExecutor,
        profile: RuntimeProfile,
    ) -> None:
        self._llm_service = llm_service
        self._tool_executor = tool_executor
        self._profile = profile
        self._langfuse = get_langfuse_service()
        self._context_governor = SetupContextGovernorService()
        self._event_queue: asyncio.Queue[RuntimeEvent | object] | None = None
        self._event_sequence_no = 0
        self.last_result: RpAgentTurnResult | None = None

    async def run(self, turn_input: RpAgentTurnInput) -> RpAgentTurnResult:
        graph = self._build_graph()
        final_state = await graph.ainvoke(
            self._initial_state(turn_input),
            config=self._graph_invoke_config(),
        )
        self.last_result = self._state_to_result(final_state)
        return self.last_result

    async def run_stream(
        self,
        turn_input: RpAgentTurnInput,
    ) -> AsyncIterator[RuntimeEvent]:
        self._event_queue = asyncio.Queue()

        async def _runner() -> None:
            try:
                graph = self._build_graph()
                final_state = await graph.ainvoke(
                    self._initial_state(turn_input),
                    config=self._graph_invoke_config(),
                )
                self.last_result = self._state_to_result(final_state)
            except Exception as exc:  # pragma: no cover - defensive safety net
                error = {"message": str(exc), "type": "runtime_execution_failed"}
                await self._emit_event(
                    run_id=turn_input.workspace_id
                    or turn_input.session_id
                    or turn_input.profile_id,
                    event_type="error",
                    payload={"error": error},
                )
                await self._emit_event(
                    run_id=turn_input.workspace_id
                    or turn_input.session_id
                    or turn_input.profile_id,
                    event_type="done",
                    payload={},
                )
                self.last_result = RpAgentTurnResult(
                    status="failed",
                    finish_reason="runtime_execution_failed",
                    assistant_text="",
                    error=error,
                )
            finally:
                if self._event_queue is not None:
                    await self._event_queue.put(_SENTINEL)

        task = asyncio.create_task(_runner())
        try:
            while True:
                item = await self._event_queue.get()
                if item is _SENTINEL:
                    break
                if not isinstance(item, RuntimeEvent):
                    continue
                yield item
        finally:
            await task

    def _build_graph(self):
        return build_runtime_graph(
            prepare_input=self._prepare_input,
            derive_turn_goal=self._derive_turn_goal,
            plan_step_slice=self._plan_step_slice,
            build_model_request=self._build_model_request,
            call_model=self._call_model,
            inspect_model_output=self._inspect_model_output,
            execute_tools=self._execute_tools,
            apply_tool_results=self._apply_tool_results,
            assess_progress=self._assess_progress,
            reflect_if_needed=self._reflect_if_needed,
            finalize_success=self._finalize_success,
            finalize_failure=self._finalize_failure,
            route_after_inspect=self._route_after_inspect,
            route_after_assess=self._route_after_assess,
            route_after_reflect=self._route_after_reflect,
        )

    def _initial_state(self, turn_input: RpAgentTurnInput) -> RpAgentRunState:
        return {
            "run_id": uuid.uuid4().hex,
            "profile_id": turn_input.profile_id,
            "run_kind": turn_input.run_kind,
            "status": "received",
            "round_no": 0,
            "stream_mode": turn_input.stream,
            "turn_input": turn_input.model_dump(mode="json", exclude_none=True),
            "normalized_messages": [],
            "tool_definitions": [],
            "output_inspection": None,
            "pending_tool_calls": [],
            "tool_invocations": [],
            "tool_results": [],
            "latest_tool_batch": [],
            "assistant_text": "",
            "warnings": [],
            "finish_reason": None,
            "error": None,
            "turn_goal": None,
            "working_plan": None,
            "pending_obligation": None,
            "last_failure": None,
            "reflection_ticket": None,
            "completion_guard": None,
            "cognitive_state": self._context_bundle(turn_input).get("cognitive_state"),
            "cognitive_state_summary": self._context_bundle(turn_input).get(
                "cognitive_state_summary"
            ),
            "working_digest": self._context_bundle(turn_input).get("working_digest"),
            "tool_outcomes": list(
                self._context_bundle(turn_input).get("tool_outcomes") or []
            ),
            "compact_summary": self._context_bundle(turn_input).get("compact_summary"),
            "repair_route": None,
            "continue_reason": None,
            "loop_trace": [],
            "model_gateway_diagnostics": None,
            "next_action": "build_model_request",
            "schema_retry_count": 0,
            "pseudo_tool_retry_count": 0,
            "error_event_emitted": False,
        }

    def _prepare_input(self, state: RpAgentRunState) -> RpAgentRunState:
        turn_input = self._turn_input(state)
        normalized_messages: list[dict[str, Any]] = []
        system_prompt = str(turn_input.context_bundle.get("system_prompt") or "")
        if system_prompt:
            normalized_messages.append(
                ChatMessage(role="system", content=system_prompt).model_dump(
                    mode="json",
                    exclude_none=True,
                )
            )
        for message in turn_input.conversation_messages:
            normalized_messages.append(self._normalize_message_dict(message))
        normalized_messages.append(
            ChatMessage(
                role="user",
                content=turn_input.user_visible_request or "",
            ).model_dump(mode="json", exclude_none=True)
        )
        return {
            "status": "prepared",
            "normalized_messages": normalized_messages,
        }

    def _derive_turn_goal(self, state: RpAgentRunState) -> RpAgentRunState:
        turn_input = self._turn_input(state)
        context_bundle = self._context_bundle(turn_input)
        current_step = str(context_bundle.get("current_step") or "unknown_step")
        pending_obligation = self._pending_obligation(state)
        cognitive_state_summary = self._cognitive_state_summary(state)
        user_prompt = str(turn_input.user_visible_request or "").strip()
        current_snapshot = (
            self._context_packet(state).get("current_draft_snapshot") or {}
        )

        if pending_obligation is not None and pending_obligation.unresolved:
            if pending_obligation.obligation_type == "repair_tool_call":
                goal = SetupTurnGoal(
                    current_step=current_step,
                    goal_type="recover_from_tool_failure",
                    goal_summary=(
                        f"Repair the failed tool call for {pending_obligation.tool_name or 'setup tool'}."
                    ),
                    success_criteria=[
                        "Emit a corrected tool call or convert the turn into a targeted user question.",
                        "Clear the current repair obligation before finishing.",
                    ],
                )
            elif pending_obligation.obligation_type == "ask_user_for_missing_info":
                goal = SetupTurnGoal(
                    current_step=current_step,
                    goal_type="clarify_user_intent",
                    goal_summary="Ask the user for the missing information required to continue setup.",
                    success_criteria=[
                        "Ask one targeted user-facing clarification question.",
                        "Do not pretend the step is complete yet.",
                    ],
                )
            elif pending_obligation.obligation_type == "reconcile_after_user_edit":
                goal = SetupTurnGoal(
                    current_step=current_step,
                    goal_type="reconcile_after_user_edit",
                    goal_summary=(
                        "Reconcile the discussion state against the latest user-edited draft "
                        "before making more commit-oriented moves."
                    ),
                    success_criteria=[
                        "Refresh the discussion map from the latest draft context.",
                        "Do not treat stale truth candidates as ready for review.",
                    ],
                )
            elif pending_obligation.obligation_type == "continue_after_tool_failure":
                goal = SetupTurnGoal(
                    current_step=current_step,
                    goal_type="recover_from_tool_failure",
                    goal_summary=(
                        "Recover from the most recent tool failure and continue advancing the "
                        "same setup objective instead of switching to commit preparation."
                    ),
                    success_criteria=[
                        "Address the failed tool outcome in the current topic.",
                        "Continue discussion, ask the user, or retry with a better action as appropriate.",
                    ],
                )
            else:
                goal = SetupTurnGoal(
                    current_step=current_step,
                    goal_type="prepare_commit_intent",
                    goal_summary="Reassess whether the step is truly ready for review or commit.",
                    success_criteria=[
                        "Do not propose commit while unresolved blockers remain.",
                        "Continue discussion or ask a targeted question instead of forcing commit.",
                    ],
                )
        elif (
            cognitive_state_summary is not None and cognitive_state_summary.invalidated
        ):
            goal = SetupTurnGoal(
                current_step=current_step,
                goal_type="reconcile_after_user_edit",
                goal_summary=(
                    "The current discussion state is stale relative to the latest draft or "
                    "rejection feedback. Reconcile it before telling the user the draft is "
                    "ready for commit."
                ),
                success_criteria=[
                    "Use the latest draft and user edits when assessing readiness.",
                    "Explain unresolved readiness risks instead of implying commit is complete.",
                ],
            )
        elif self._user_requests_commit(user_prompt):
            goal = SetupTurnGoal(
                current_step=current_step,
                goal_type="prepare_commit_intent",
                goal_summary=(
                    "Respond to the user's commit intent by summarizing readiness and "
                    "surfacing unresolved concerns. Final commit is confirmed through "
                    "the UI commit button, not an agent tool."
                ),
                success_criteria=[
                    "State whether the current draft appears ready for user review.",
                    "Mention unresolved blockers or stale-state concerns as visible risks.",
                ],
            )
        elif not current_snapshot:
            goal = SetupTurnGoal(
                current_step=current_step,
                goal_type="fill_missing_step_fields",
                goal_summary="Fill the minimum missing fields for the current setup step.",
                success_criteria=[
                    "Identify the highest-priority missing information.",
                    "Either patch the draft or ask the user for the missing inputs.",
                ],
            )
        elif (
            cognitive_state_summary is not None
            and cognitive_state_summary.candidate_titles
            and cognitive_state_summary.truth_write_status is None
        ):
            goal = SetupTurnGoal(
                current_step=current_step,
                goal_type="advance_stage_entry_draft",
                goal_summary=(
                    "A chunk candidate is emerging; decide whether one candidate should now be "
                    "written into the current stage draft."
                ),
                success_criteria=[
                    "Use the current stage-entry draft tools when one candidate is stable enough.",
                    "Keep unresolved issues visible instead of pretending the draft is finalized.",
                ],
            )
        elif (
            cognitive_state_summary is not None
            and cognitive_state_summary.candidate_titles
        ):
            goal = SetupTurnGoal(
                current_step=current_step,
                goal_type="refine_chunk_candidate",
                goal_summary="Refine the current chunk candidates until they are truly usable.",
                success_criteria=[
                    "Clarify open issues on the active chunk candidates.",
                    "Promote only candidates that are specific enough to support draft writing.",
                ],
            )
        else:
            goal = SetupTurnGoal(
                current_step=current_step,
                goal_type="brainstorm_and_clarify",
                goal_summary="Advance the current setup discussion toward a more converged state.",
                success_criteria=[
                    "Prefer concrete progress over vague discussion.",
                    "Keep the turn aligned with the current setup step.",
                ],
            )

        return {
            "status": "goal_derived",
            "turn_goal": goal.model_dump(mode="json", exclude_none=True),
            "working_digest": self._build_working_digest_payload(
                state,
                turn_goal=goal,
            ),
        }

    def _plan_step_slice(self, state: RpAgentRunState) -> RpAgentRunState:
        goal_payload = state.get("turn_goal") or {}
        goal = SetupTurnGoal.model_validate(goal_payload)
        context_packet = self._context_packet(state)
        context_bundle = self._context_bundle(self._turn_input(state))
        pending_obligation = self._pending_obligation(state)
        cognitive_state_summary = self._cognitive_state_summary(state)

        missing_information = list(
            pending_obligation.required_fields if pending_obligation is not None else []
        )
        if not missing_information and not context_packet.get("current_draft_snapshot"):
            missing_information.append(f"{goal.current_step}:core_fields")
        for question_text in context_bundle.get("open_question_texts") or []:
            text = str(question_text or "").strip()
            if text and text not in missing_information:
                missing_information.append(text)

        discussion_actions: list[str] = []
        if cognitive_state_summary is not None and cognitive_state_summary.invalidated:
            discussion_actions.append("reconcile_discussion_state_from_latest_draft")
        if missing_information:
            discussion_actions.append("resolve_missing_information")
        if (
            cognitive_state_summary is not None
            and cognitive_state_summary.unresolved_conflicts
        ):
            discussion_actions.append("resolve_unresolved_conflicts")
        if not discussion_actions:
            discussion_actions.append("advance_current_step")

        plan = SetupWorkingPlan(
            missing_information=missing_information,
            discussion_actions=discussion_actions,
            candidate_targets=(
                list(cognitive_state_summary.candidate_titles)
                if cognitive_state_summary is not None
                else []
            ),
            draft_write_targets=(
                self._patch_targets_for_step(goal.current_step)
                if (
                    cognitive_state_summary is not None
                    and cognitive_state_summary.candidate_titles
                )
                else []
            ),
            patch_targets=self._patch_targets_for_step(goal.current_step),
            question_targets=list(missing_information),
            commit_readiness_checks=self._commit_readiness_checks(goal.current_step),
            current_priority=(
                goal.goal_summary
                if goal.goal_type not in {"patch_draft", "brainstorm_and_clarify"}
                else f"Advance {goal.current_step} with the highest-value patch."
            ),
        )
        return {
            "status": "planned",
            "working_plan": plan.model_dump(mode="json", exclude_none=True),
            "working_digest": self._build_working_digest_payload(
                state,
                working_plan=plan,
            ),
        }

    def _build_model_request(self, state: RpAgentRunState) -> RpAgentRunState:
        turn_input = self._turn_input(state)
        visible_tool_names = self._visible_tool_names(turn_input)
        tool_definitions = (
            self._model_facing_tool_definitions(
                self._tool_executor.get_openai_tool_definitions(
                    visible_tool_names=visible_tool_names
                ),
                turn_input=turn_input,
            )
            if self._profile.supports_tools
            else []
        )
        provider_payload = turn_input.metadata.get("provider") or {}
        provider = (
            ProviderConfig.model_validate(provider_payload)
            if provider_payload
            else None
        )
        request = ChatCompletionRequest(
            model=str(turn_input.metadata.get("model_name") or turn_input.model_id),
            model_id=turn_input.model_id,
            messages=self._build_request_messages(state),
            stream=bool(state.get("stream_mode")),
            stream_event_mode="typed" if state.get("stream_mode") else None,
            provider_id=turn_input.provider_id,
            provider=provider,
            enable_tools=bool(tool_definitions),
            tools=tool_definitions or None,
            tool_choice="auto" if tool_definitions else None,
        )
        return {
            "status": "model_request_built",
            "round_no": int(state.get("round_no") or 0) + 1,
            "tool_definitions": tool_definitions,
            "latest_request": request.model_dump(mode="json", exclude_none=True),
        }

    def _model_facing_tool_definitions(
        self,
        tool_definitions: list[dict[str, Any]],
        *,
        turn_input: RpAgentTurnInput,
    ) -> list[dict[str, Any]]:
        return list(tool_definitions)

    async def _call_model(self, state: RpAgentRunState) -> RpAgentRunState:
        request = ChatCompletionRequest.model_validate(state["latest_request"])
        if state.get("stream_mode"):
            return await self._call_model_stream(state, request)
        return await self._call_model_non_stream(request)

    async def _call_model_non_stream(
        self,
        request: ChatCompletionRequest,
    ) -> RpAgentRunState:
        try:
            with self._langfuse.start_as_current_observation(
                name="rp.runtime.model_call",
                as_type="generation",
                model=request.model,
                input={
                    "messages": request.model_dump(mode="json").get("messages", []),
                    "tool_count": len(request.tools or []),
                    "stream": False,
                },
                model_parameters={
                    "stream": False,
                    "tool_choice": request.tool_choice,
                },
            ) as generation:
                response = self._coerce_response_dict(
                    await self._llm_service.chat_completion(request)
                )
                usage_payload = response.get("usage")
                generation.update(
                    output=self._extract_message_payload(response),
                    usage_details=(
                        dict(usage_payload) if isinstance(usage_payload, dict) else None
                    ),
                )
        except Exception as exc:
            return self._model_gateway_failure_update(
                failure_kind="provider_request_error",
                message=str(exc),
                provider_error_type=type(exc).__name__,
            )
        return {
            "status": "model_called",
            "latest_response": response,
        }

    async def _call_model_stream(
        self,
        state: RpAgentRunState,
        request: ChatCompletionRequest,
    ) -> RpAgentRunState:
        accumulated_tool_calls: dict[int, dict[str, Any]] = {}
        accumulated_text = ""
        pending_text_chunks: list[str] = []
        suppress_pending_text = False
        error_payload: dict[str, Any] | None = None
        model_gateway_diagnostics: dict[str, Any] | None = None
        usage_details: dict[str, Any] | None = None
        run_id = str(state["run_id"])

        async def _flush_pending_text() -> None:
            nonlocal pending_text_chunks, suppress_pending_text
            if suppress_pending_text:
                pending_text_chunks = []
                return
            for chunk in pending_text_chunks:
                await self._emit_event(
                    run_id=run_id,
                    event_type="text_delta",
                    payload={"delta": chunk},
                )
            pending_text_chunks = []

        with self._langfuse.start_as_current_observation(
            name="rp.runtime.model_call.stream",
            as_type="generation",
            model=request.model,
            input={
                "messages": request.model_dump(mode="json").get("messages", []),
                "tool_count": len(request.tools or []),
                "stream": True,
            },
            model_parameters={
                "stream": True,
                "tool_choice": request.tool_choice,
            },
        ) as generation:
            try:
                async for line in self._llm_service.chat_completion_stream(request):
                    payload = self._parse_sse_payload(line)
                    if payload is None:
                        continue

                    event_type = str(payload.get("type") or "")
                    if event_type == "done":
                        await _flush_pending_text()
                        break
                    if event_type in {"thinking_delta", "text_delta"}:
                        delta = str(payload.get("delta") or "")
                        if event_type == "thinking_delta":
                            await self._emit_event(
                                run_id=run_id,
                                event_type="thinking_delta",
                                payload={"delta": delta},
                            )
                            continue
                        if event_type == "text_delta":
                            accumulated_text += delta
                            if accumulated_tool_calls:
                                suppress_pending_text = True
                                pending_text_chunks = []
                                continue
                            pending_text_chunks.append(delta)
                            if self._looks_like_pseudo_tool_call_text(
                                "".join(pending_text_chunks)
                            ):
                                suppress_pending_text = True
                                pending_text_chunks = []
                        continue
                    if event_type == "tool_call":
                        # A stream can deliver ordinary-looking text before the
                        # provider emits a real tool call. Once the output is mixed
                        # text+tool, the full buffered text remains private and
                        # OutputInspector owns the final classification.
                        suppress_pending_text = True
                        pending_text_chunks = []
                        raw_calls = payload.get("tool_calls", [])
                        self._merge_stream_tool_calls(accumulated_tool_calls, raw_calls)
                        await self._emit_event(
                            run_id=run_id,
                            event_type="tool_call",
                            payload={
                                "tool_calls": raw_calls
                                if isinstance(raw_calls, list)
                                else []
                            },
                        )
                        continue
                    if event_type == "error":
                        suppress_pending_text = True
                        pending_text_chunks = []
                        error_payload, model_gateway_diagnostics = (
                            self._model_gateway_error_payload(
                                failure_kind="provider_stream_error",
                                raw_error=payload.get("error"),
                            )
                        )
                        await self._emit_event(
                            run_id=run_id,
                            event_type="error",
                            payload={"error": error_payload},
                        )
                        break
                    if event_type == "usage":
                        usage_details = dict(payload)
                        await self._emit_event(
                            run_id=run_id,
                            event_type="usage",
                            payload=usage_details,
                        )
                        continue

                    model_gateway_diagnostics = self._merge_model_gateway_private_event(
                        model_gateway_diagnostics,
                        event_type=event_type or "unknown",
                        payload=payload,
                    )
            except Exception as exc:
                suppress_pending_text = True
                pending_text_chunks = []
                failure_kind = (
                    "provider_stream_parse_error"
                    if isinstance(exc, _ProviderStreamPayloadError)
                    else "provider_stream_exception"
                )
                error_payload, model_gateway_diagnostics = (
                    self._model_gateway_error_payload(
                        failure_kind=failure_kind,
                        raw_error={
                            "message": str(exc),
                            "type": type(exc).__name__,
                        },
                    )
                )
                await self._emit_event(
                    run_id=run_id,
                    event_type="error",
                    payload={"error": error_payload},
                )

            generation.update(
                output={
                    "content": accumulated_text or None,
                    "tool_calls": self._finalize_stream_tool_calls(
                        accumulated_tool_calls
                    ),
                    "error": error_payload,
                    "model_gateway_diagnostics": model_gateway_diagnostics,
                },
                usage_details=usage_details,
            )

        response_message = {
            "content": accumulated_text or None,
            "tool_calls": self._finalize_stream_tool_calls(accumulated_tool_calls),
        }
        latest_response: dict[str, Any] = {"message": response_message}
        if usage_details is not None:
            latest_response["usage"] = usage_details
        update: RpAgentRunState = {
            "status": "model_called",
            "latest_response": latest_response,
            "assistant_text": accumulated_text,
        }
        if error_payload is not None:
            update["error"] = error_payload
            update["finish_reason"] = "upstream_error"
            update["error_event_emitted"] = True
        if model_gateway_diagnostics is not None:
            update["model_gateway_diagnostics"] = model_gateway_diagnostics
        return update

    def _inspect_model_output(self, state: RpAgentRunState) -> RpAgentRunState:
        message_payload = self._extract_message_payload(
            state.get("latest_response") or {}
        )
        inspection = self._inspect_output_payload(state, message_payload)
        assistant_text = inspection.public_text_candidate
        runtime_tool_calls = list(inspection.tool_calls)
        normalized_messages = list(state.get("normalized_messages", []))
        if assistant_text and not runtime_tool_calls:
            normalized_messages.append(
                ChatMessage(role="assistant", content=assistant_text).model_dump(
                    mode="json",
                    exclude_none=True,
                )
            )
        update: RpAgentRunState = {
            "status": "model_inspected",
            "assistant_text": assistant_text,
            "output_inspection": inspection.model_dump(mode="json", exclude_none=True),
            "normalized_messages": normalized_messages,
            "pending_tool_calls": [
                call.model_dump(mode="json", exclude_none=True)
                for call in runtime_tool_calls
            ],
            "completion_guard": None,
            "continue_reason": None,
        }
        if state.get("error"):
            update["next_action"] = "finalize_failure"
            update["loop_trace"] = self._append_loop_trace(
                state,
                update,
                decision_site="inspect_model_output",
                tool_names=[call.tool_name for call in runtime_tool_calls],
            )
            return update
        if inspection.classification in {"pseudo_tool_text", "malformed_tool_call"}:
            pseudo_tool_retry_count = int(state.get("pseudo_tool_retry_count") or 0)
            warnings = list(state.get("warnings", []))
            warning = (
                "pseudo_tool_call_text_filtered"
                if inspection.classification == "pseudo_tool_text"
                else "malformed_tool_call_filtered"
            )
            if warning not in warnings:
                warnings.append(warning)
            update["warnings"] = warnings
            update["completion_guard"] = {
                "allow_finalize": False,
                "reason": (
                    "pseudo_tool_call_text_emitted"
                    if inspection.classification == "pseudo_tool_text"
                    else "malformed_tool_call_emitted"
                ),
                "required_action": "retry",
            }
            update["pseudo_tool_retry_count"] = pseudo_tool_retry_count + 1
            if pseudo_tool_retry_count >= 1:
                update["status"] = "failed"
                update["finish_reason"] = "repair_obligation_unfulfilled"
                update["error"] = {
                    "message": (
                        "The assistant repeatedly emitted invalid tool output "
                        "instead of a real tool call."
                    ),
                    "type": "repair_obligation_unfulfilled",
                    "details": inspection.private_diagnostics,
                }
                update["next_action"] = "finalize_failure"
                update["loop_trace"] = self._append_loop_trace(
                    state,
                    update,
                    decision_site="inspect_model_output",
                )
                return update
            update["reflection_ticket"] = SetupReflectionTicket(
                trigger="tool_failure",
                summary=(
                    "The assistant emitted invalid tool output instead of a real "
                    "tool call. Issue the setup tool call directly."
                ),
                required_decision="retry",
            ).model_dump(mode="json", exclude_none=True)
            update["continue_reason"] = "completion_guard_retry"
            update["next_action"] = "reflect_if_needed"
            update["loop_trace"] = self._append_loop_trace(
                state,
                update,
                decision_site="inspect_model_output",
            )
            return update
        if runtime_tool_calls:
            update["reflection_ticket"] = None
            update["continue_reason"] = "tool_call_batch_pending"
            update["next_action"] = "execute_tools"
            update["loop_trace"] = self._append_loop_trace(
                state,
                update,
                decision_site="inspect_model_output",
                tool_names=[call.tool_name for call in runtime_tool_calls],
            )
            return update
        decision = CompletionGuardPolicy.assess(
            assistant_text=assistant_text,
            pending_obligation=self._pending_obligation(state),
            reflection_ticket=self._reflection_ticket(state),
            cognitive_state_summary=self._cognitive_state_summary(state),
            prior_assistant_questions=self._recent_assistant_questions(state),
            working_digest=self._working_digest(state),
        )
        update["completion_guard"] = decision.get("completion_guard")
        if "pending_obligation" in decision:
            update["pending_obligation"] = decision.get("pending_obligation")
        if "reflection_ticket" in decision:
            update["reflection_ticket"] = decision.get("reflection_ticket")
        if decision.get("allow_finalize"):
            update["finish_reason"] = str(
                decision.get("finish_reason")
                or FinishPolicy.completed_text_finish_reason(assistant_text)
            )
            update["next_action"] = "finalize_success"
            update["loop_trace"] = self._append_loop_trace(
                state,
                update,
                decision_site="inspect_model_output",
            )
            return update
        update["continue_reason"] = "completion_guard_retry"
        update["next_action"] = "reflect_if_needed"
        update["loop_trace"] = self._append_loop_trace(
            state,
            update,
            decision_site="inspect_model_output",
        )
        return update

    async def _execute_tools(self, state: RpAgentRunState) -> RpAgentRunState:
        turn_input = self._turn_input(state)
        visible_tool_names = self._visible_tool_names(turn_input)
        call_batch = [
            RuntimeToolCall.model_validate(item)
            for item in state.get("pending_tool_calls", [])
        ]
        aggregated_invocations = list(state.get("tool_invocations", []))
        aggregated_results = list(state.get("tool_results", []))
        latest_batch: list[dict[str, Any]] = []
        run_id = str(state["run_id"])

        for call in call_batch:
            executable_call = self._normalize_tool_call_for_execution(
                call,
                turn_input=turn_input,
            )
            aggregated_invocations.append(
                executable_call.model_dump(mode="json", exclude_none=True)
            )
            await self._emit_event(
                run_id=run_id,
                event_type="tool_started",
                payload={
                    "call_id": executable_call.call_id,
                    "tool_name": executable_call.tool_name,
                },
            )
            with self._langfuse.start_as_current_observation(
                name=f"rp.runtime.tool:{executable_call.tool_name}",
                as_type="tool",
                input={
                    "tool_name": executable_call.tool_name,
                    "arguments": dict(executable_call.arguments),
                    "visible_tool_names": list(visible_tool_names),
                },
            ) as tool_observation:
                result = await self._tool_executor.execute_tool_call(
                    executable_call,
                    visible_tool_names=visible_tool_names,
                )
                tool_observation.update(
                    output={
                        "success": result.success,
                        "tool_name": result.tool_name,
                        "error_code": result.error_code,
                        "content_text": result.content_text,
                        "structured_payload": result.structured_payload,
                    }
                )
            latest_batch.append(result.model_dump(mode="json", exclude_none=True))
            aggregated_results.append(result.model_dump(mode="json", exclude_none=True))
            if result.success:
                await self._emit_event(
                    run_id=run_id,
                    event_type="tool_result",
                    payload={
                        "call_id": result.call_id,
                        "tool_name": result.tool_name,
                        "result": result.content_text,
                    },
                )
            else:
                await self._emit_event(
                    run_id=run_id,
                    event_type="tool_error",
                    payload={
                        "call_id": result.call_id,
                        "tool_name": result.tool_name,
                        "error": result.content_text,
                    },
                )
                if ToolFailureClassifier.classify(result) == "unrecoverable":
                    break

        return {
            "status": "tools_executed",
            "tool_invocations": aggregated_invocations,
            "tool_results": aggregated_results,
            "latest_tool_batch": latest_batch,
        }

    def _normalize_tool_call_for_execution(
        self,
        call: RuntimeToolCall,
        *,
        turn_input: RpAgentTurnInput,
    ) -> RuntimeToolCall:
        return call

    def _apply_tool_results(self, state: RpAgentRunState) -> RpAgentRunState:
        normalized_messages = list(state.get("normalized_messages", []))
        tool_calls = [
            RuntimeToolCall.model_validate(item)
            for item in state.get("pending_tool_calls", [])
        ]
        cognitive_state = state.get("cognitive_state")
        cognitive_state_summary = state.get("cognitive_state_summary")
        retained_tool_outcomes = self._tool_outcomes(state)
        latest_outcomes: list[SetupToolOutcome] = []
        if tool_calls:
            normalized_messages.append(
                ChatMessage(
                    role="assistant",
                    content=state.get("assistant_text") or "",
                    tool_calls=[
                        self._runtime_tool_call_to_openai(call) for call in tool_calls
                    ],
                ).model_dump(mode="json", exclude_none=True)
            )
        for item in state.get("latest_tool_batch", []):
            result = RuntimeToolResult.model_validate(item)
            normalized_messages.append(
                ChatMessage(
                    role="tool",
                    name=result.tool_name,
                    tool_call_id=result.call_id,
                    content=result.content_text,
                ).model_dump(mode="json", exclude_none=True)
            )
            content_payload = (
                result.structured_payload.get("content_payload")
                if isinstance(result.structured_payload, dict)
                else None
            )
            if isinstance(content_payload, dict):
                if isinstance(content_payload.get("cognitive_state_snapshot"), dict):
                    cognitive_state = content_payload["cognitive_state_snapshot"]
                if isinstance(content_payload.get("cognitive_state_summary"), dict):
                    cognitive_state_summary = content_payload["cognitive_state_summary"]
            latest_outcomes.append(self._tool_outcome_from_result(result))
        merged_tool_outcomes = self._context_governor.retain_tool_outcomes(
            existing=retained_tool_outcomes,
            latest_results=latest_outcomes,
        )
        return {
            "status": "tool_results_applied",
            "normalized_messages": normalized_messages,
            "pending_tool_calls": [],
            "cognitive_state": cognitive_state,
            "cognitive_state_summary": cognitive_state_summary,
            "tool_outcomes": [
                item.model_dump(mode="json", exclude_none=True)
                for item in merged_tool_outcomes
            ],
            "working_digest": self._build_working_digest_payload(
                state,
                cognitive_state=(
                    SetupCognitiveStateSnapshot.model_validate(cognitive_state)
                    if isinstance(cognitive_state, dict)
                    else None
                ),
                cognitive_state_summary=(
                    SetupCognitiveStateSummary.model_validate(cognitive_state_summary)
                    if isinstance(cognitive_state_summary, dict)
                    else None
                ),
                tool_outcomes=merged_tool_outcomes,
            ),
        }

    def _assess_progress(self, state: RpAgentRunState) -> RpAgentRunState:
        latest_batch = [
            RuntimeToolResult.model_validate(item)
            for item in state.get("latest_tool_batch", [])
        ]
        if not latest_batch:
            decision = CompletionGuardPolicy.assess(
                assistant_text=str(state.get("assistant_text") or ""),
                pending_obligation=self._pending_obligation(state),
                reflection_ticket=self._reflection_ticket(state),
                cognitive_state_summary=self._cognitive_state_summary(state),
                prior_assistant_questions=self._recent_assistant_questions(state),
                working_digest=self._working_digest(state),
            )
            update: RpAgentRunState = {
                "status": "assessed",
                "completion_guard": decision.get("completion_guard"),
                "continue_reason": None,
            }
            if "pending_obligation" in decision:
                update["pending_obligation"] = decision.get("pending_obligation")
            if "reflection_ticket" in decision:
                update["reflection_ticket"] = decision.get("reflection_ticket")
            if decision.get("allow_finalize"):
                update["next_action"] = "finalize_success"
                update["finish_reason"] = str(
                    decision.get("finish_reason")
                    or FinishPolicy.completed_text_finish_reason(
                        str(state.get("assistant_text") or "")
                    )
                )
                update["working_digest"] = self._build_working_digest_payload(
                    state,
                    pending_obligation=None,
                    reflection_ticket=None,
                )
                update["loop_trace"] = self._append_loop_trace(
                    state,
                    update,
                    decision_site="assess_progress",
                )
                return update
            update["continue_reason"] = "completion_guard_retry"
            update["next_action"] = "reflect_if_needed"
            update["working_digest"] = self._build_working_digest_payload(state)
            update["loop_trace"] = self._append_loop_trace(
                state,
                update,
                decision_site="assess_progress",
            )
            return update

        decision = RepairDecisionPolicy.assess(
            profile=self._profile,
            tool_results=latest_batch,
            prior_tool_outcomes=self._tool_outcomes(state),
            schema_retry_count=int(state.get("schema_retry_count") or 0),
            round_no=int(state.get("round_no") or 0),
        )
        warnings = list(state.get("warnings", []))
        for item in self._tool_result_warning_codes(latest_batch):
            if item not in warnings:
                warnings.append(item)
        if decision.get("warning"):
            warnings.append(str(decision["warning"]))
        for item in decision.get("warnings") or []:
            if item and item not in warnings:
                warnings.append(str(item))

        if decision["action"] == "continue":
            continue_update: RpAgentRunState = {
                "status": "assessed",
                "next_action": (
                    "reflect_if_needed"
                    if decision.get("reflection_ticket") is not None
                    else "derive_turn_goal"
                ),
                "warnings": warnings,
                "schema_retry_count": decision.get(
                    "schema_retry_count",
                    int(state.get("schema_retry_count") or 0),
                ),
                "pending_obligation": decision.get("pending_obligation"),
                "reflection_ticket": decision.get("reflection_ticket"),
                "completion_guard": decision.get("completion_guard"),
                "repair_route": (
                    decision.get("last_failure", {}).get("failure_category")
                    if isinstance(decision.get("last_failure"), dict)
                    else state.get("repair_route")
                ),
                "continue_reason": None,
                "working_digest": self._build_working_digest_payload(
                    state,
                    pending_obligation=(
                        SetupPendingObligation.model_validate(
                            decision["pending_obligation"]
                        )
                        if isinstance(decision.get("pending_obligation"), dict)
                        else None
                    ),
                    reflection_ticket=(
                        SetupReflectionTicket.model_validate(
                            decision["reflection_ticket"]
                        )
                        if isinstance(decision.get("reflection_ticket"), dict)
                        else None
                    ),
                ),
            }
            if "last_failure" in decision:
                continue_update["last_failure"] = decision.get("last_failure")
            if decision.get("reflection_ticket") is not None:
                continue_update["continue_reason"] = "commit_reassess_reflection"
            elif isinstance(decision.get("last_failure"), dict):
                continue_update["continue_reason"] = "tool_failure_follow_up"
            else:
                continue_update["continue_reason"] = "tool_result_follow_up"
            continue_update["loop_trace"] = self._append_loop_trace(
                state,
                continue_update,
                decision_site="assess_progress",
            )
            return continue_update

        failure_update: RpAgentRunState = {
            "status": "failed",
            "next_action": "finalize_failure",
            "finish_reason": str(decision.get("finish_reason") or "runtime_failed"),
            "warnings": warnings,
            "error": decision.get("error"),
            "last_failure": decision.get("last_failure"),
            "working_digest": self._build_working_digest_payload(state),
        }
        failure_update["loop_trace"] = self._append_loop_trace(
            state,
            failure_update,
            decision_site="assess_progress",
        )
        return failure_update

    def _reflect_if_needed(self, state: RpAgentRunState) -> RpAgentRunState:
        decision = ReflectionTriggerPolicy.assess(
            profile=self._profile,
            reflection_ticket=self._reflection_ticket(state),
            pending_obligation=self._pending_obligation(state),
            schema_retry_count=int(state.get("schema_retry_count") or 0),
            round_no=int(state.get("round_no") or 0),
        )
        warnings = list(state.get("warnings", []))
        if decision.get("warning"):
            warnings.append(str(decision["warning"]))

        if decision["action"] == "continue":
            continue_update: RpAgentRunState = {
                "status": "reflected",
                "next_action": "derive_turn_goal",
                "warnings": warnings,
                "reflection_ticket": None,
                "continue_reason": "reflection_retry",
                "working_digest": self._build_working_digest_payload(
                    state,
                    reflection_ticket=None,
                ),
            }
            continue_update["loop_trace"] = self._append_loop_trace(
                state,
                continue_update,
                decision_site="reflect_if_needed",
            )
            return continue_update

        failure_update: RpAgentRunState = {
            "status": "failed",
            "next_action": "finalize_failure",
            "finish_reason": str(decision.get("finish_reason") or "runtime_failed"),
            "warnings": warnings,
            "error": decision.get("error"),
            "reflection_ticket": None,
            "working_digest": self._build_working_digest_payload(
                state,
                reflection_ticket=None,
            ),
        }
        failure_update["loop_trace"] = self._append_loop_trace(
            state,
            failure_update,
            decision_site="reflect_if_needed",
        )
        return failure_update

    async def _finalize_success(self, state: RpAgentRunState) -> RpAgentRunState:
        await self._emit_event(
            run_id=str(state["run_id"]),
            event_type="done",
            payload={},
        )
        update: RpAgentRunState = {
            "status": "completed",
            "finish_reason": state.get("finish_reason")
            or FinishPolicy.completed_text_finish_reason(
                str(state.get("assistant_text") or "")
            ),
            "continue_reason": None,
        }
        update["loop_trace"] = self._append_loop_trace(
            state,
            update,
            decision_site="finalize_success",
        )
        return update

    async def _finalize_failure(self, state: RpAgentRunState) -> RpAgentRunState:
        error = state.get("error") or {
            "message": "runtime_failed",
            "type": str(state.get("finish_reason") or "runtime_failed"),
        }
        if not state.get("error_event_emitted"):
            await self._emit_event(
                run_id=str(state["run_id"]),
                event_type="error",
                payload={"error": error},
            )
        await self._emit_event(
            run_id=str(state["run_id"]),
            event_type="done",
            payload={},
        )
        update: RpAgentRunState = {
            "status": "failed",
            "error": error,
            "finish_reason": str(state.get("finish_reason") or "runtime_failed"),
            "continue_reason": None,
            "error_event_emitted": True,
        }
        update["loop_trace"] = self._append_loop_trace(
            state,
            update,
            decision_site="finalize_failure",
        )
        return update

    @classmethod
    def _route_after_inspect(cls, state: RpAgentRunState) -> str:
        return cls._safe_route(
            state,
            allowed=cls._INSPECT_ROUTES,
            default="finalize_success",
        )

    @classmethod
    def _route_after_assess(cls, state: RpAgentRunState) -> str:
        return cls._safe_route(
            state,
            allowed=cls._ASSESS_ROUTES,
            default="finalize_failure",
        )

    @classmethod
    def _route_after_reflect(cls, state: RpAgentRunState) -> str:
        return cls._safe_route(
            state,
            allowed=cls._REFLECT_ROUTES,
            default="finalize_failure",
        )

    @staticmethod
    def _safe_route(
        state: RpAgentRunState,
        *,
        allowed: set[str],
        default: str,
    ) -> str:
        next_action = str(state.get("next_action") or default)
        if next_action in allowed:
            return next_action

        state["finish_reason"] = "runtime_failed"
        state["error"] = {
            "message": f"Runtime selected unsupported next_action: {next_action}",
            "type": "runtime_failed",
        }
        state["next_action"] = "finalize_failure"
        return "finalize_failure"

    async def _emit_event(
        self,
        *,
        run_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        if self._event_queue is None:
            return
        self._event_sequence_no += 1
        await self._event_queue.put(
            RuntimeEvent(
                type=event_type,
                run_id=run_id,
                sequence_no=self._event_sequence_no,
                payload=payload,
            )
        )

    def _state_to_result(self, state: RpAgentRunState) -> RpAgentTurnResult:
        error = state.get("error")
        status: Literal["completed", "failed"] = "failed" if error else "completed"
        turn_input = self._turn_input(state)
        context_bundle = self._context_bundle(turn_input)
        return RpAgentTurnResult(
            status=status,
            finish_reason=str(
                state.get("finish_reason")
                or (
                    "runtime_failed"
                    if error
                    else FinishPolicy.completed_text_finish_reason(
                        str(state.get("assistant_text") or "")
                    )
                )
            ),
            assistant_text=str(state.get("assistant_text") or ""),
            tool_invocations=[
                RuntimeToolCall.model_validate(item)
                for item in state.get("tool_invocations", [])
            ],
            tool_results=[
                RuntimeToolResult.model_validate(item)
                for item in state.get("tool_results", [])
            ],
            warnings=list(state.get("warnings", [])),
            structured_payload={
                "profile_id": state.get("profile_id"),
                "round_no": state.get("round_no"),
                "skill_pack_name": turn_input.metadata.get("skill_pack_name"),
                "tool_invocation_count": len(state.get("tool_invocations", [])),
                "tool_result_count": len(state.get("tool_results", [])),
                "tool_scope": list(turn_input.tool_scope),
                "request_metrics": {
                    "system_prompt_chars": len(
                        str(context_bundle.get("system_prompt") or "")
                    ),
                    "user_prompt_chars": len(
                        str(turn_input.user_visible_request or "")
                    ),
                    "conversation_message_count": len(turn_input.conversation_messages),
                    "tool_scope_count": len(turn_input.tool_scope),
                },
                "request_context": {
                    "current_step": context_bundle.get("current_step"),
                    "step_readiness": context_bundle.get("step_readiness"),
                    "open_question_count": context_bundle.get("open_question_count"),
                    "blocking_open_question_count": context_bundle.get(
                        "blocking_open_question_count"
                    ),
                    "last_proposal_status": context_bundle.get("last_proposal_status"),
                    "cognitive_state_invalidated": context_bundle.get(
                        "cognitive_state_invalidated"
                    ),
                },
                "context_report": turn_input.metadata.get("context_report"),
                "context_pipeline": turn_input.metadata.get("context_pipeline"),
                "event_sink": SetupEventSinkSnapshot().model_dump(mode="json"),
                "model_gateway_diagnostics": state.get("model_gateway_diagnostics"),
                "latest_tool_batch": list(state.get("latest_tool_batch", [])),
                "latest_response": state.get("latest_response") or {},
                "output_inspection": state.get("output_inspection"),
                "turn_goal": state.get("turn_goal"),
                "working_plan": state.get("working_plan"),
                "pending_obligation": state.get("pending_obligation"),
                "last_failure": state.get("last_failure"),
                "reflection_ticket": state.get("reflection_ticket"),
                "completion_guard": state.get("completion_guard"),
                "cognitive_state": state.get("cognitive_state"),
                "cognitive_state_summary": state.get("cognitive_state_summary"),
                "working_digest": (
                    self._build_working_digest_payload(state)
                    or state.get("working_digest")
                ),
                "tool_outcomes": [
                    item.model_dump(mode="json", exclude_none=True)
                    for item in self._tool_outcomes(state)
                ],
                "compact_summary": state.get("compact_summary"),
                "repair_route": state.get("repair_route"),
                "continue_reason": state.get("continue_reason"),
                "loop_trace": list(state.get("loop_trace", [])),
            },
            error=error,
        )

    @staticmethod
    def _turn_input(state: RpAgentRunState) -> RpAgentTurnInput:
        return RpAgentTurnInput.model_validate(state["turn_input"])

    def _visible_tool_names(self, turn_input: RpAgentTurnInput) -> list[str]:
        if "tool_scope" in turn_input.model_fields_set:
            return list(turn_input.tool_scope)
        return list(self._profile.visible_tool_names)

    def _build_request_messages(self, state: RpAgentRunState) -> list[ChatMessage]:
        base_messages = [
            ChatMessage.model_validate(message)
            for message in state.get("normalized_messages", [])
        ]
        overlay_message = self._runtime_overlay_message(state)
        if overlay_message is None:
            return base_messages
        if base_messages and base_messages[0].role == "system":
            return [base_messages[0], overlay_message, *base_messages[1:]]
        return [overlay_message, *base_messages]

    def _runtime_overlay_message(self, state: RpAgentRunState) -> ChatMessage | None:
        payload = {
            "turn_goal": state.get("turn_goal"),
            "working_plan": state.get("working_plan"),
            "pending_obligation": state.get("pending_obligation"),
            "last_failure": state.get("last_failure"),
            "reflection_ticket": state.get("reflection_ticket"),
            "cognitive_state_summary": state.get("cognitive_state_summary"),
            "working_digest": self._build_working_digest_payload(state)
            or state.get("working_digest"),
            "tool_outcomes": [
                item.model_dump(mode="json", exclude_none=True)
                for item in self._tool_outcomes(state)
            ],
            "compact_summary": state.get("compact_summary"),
        }
        if all(value in (None, {}, []) for value in payload.values()):
            return None

        return ChatMessage(
            role="system",
            content=f"{runtime_overlay_instruction()}\n"
            f"{json.dumps(payload, ensure_ascii=False, sort_keys=True)}",
        )

    @staticmethod
    def _context_bundle(turn_input: RpAgentTurnInput) -> dict[str, Any]:
        return (
            dict(turn_input.context_bundle)
            if isinstance(turn_input.context_bundle, dict)
            else {}
        )

    def _context_packet(self, state: RpAgentRunState) -> dict[str, Any]:
        context_bundle = self._context_bundle(self._turn_input(state))
        packet = context_bundle.get("context_packet")
        return dict(packet) if isinstance(packet, dict) else {}

    @staticmethod
    def _user_requests_commit(user_prompt: str) -> bool:
        lowered = user_prompt.lower()
        return any(
            keyword in lowered
            for keyword in (
                "commit",
                "review",
                "freeze",
                "提交",
                "评审",
                "审核",
                "冻结",
                "确认提交",
                "发起评审",
                "发起审核",
            )
        )

    @staticmethod
    def _patch_targets_for_step(current_step: str) -> list[str]:
        return {
            "story_config": ["story_config_draft"],
            "writing_contract": ["writing_contract_draft"],
            "foundation": ["foundation_draft.entries"],
            "longform_blueprint": ["longform_blueprint_draft"],
        }.get(current_step, ["current_step_draft"])

    @staticmethod
    def _commit_readiness_checks(current_step: str) -> list[str]:
        return [
            f"{current_step}:blocking_questions_closed",
            f"{current_step}:draft_is_converged",
            f"{current_step}:user_feedback_addressed",
        ]

    @staticmethod
    def _pending_obligation(state: RpAgentRunState) -> SetupPendingObligation | None:
        payload = state.get("pending_obligation")
        if not isinstance(payload, dict):
            return None
        return SetupPendingObligation.model_validate(payload)

    @staticmethod
    def _reflection_ticket(state: RpAgentRunState) -> SetupReflectionTicket | None:
        payload = state.get("reflection_ticket")
        if not isinstance(payload, dict):
            return None
        return SetupReflectionTicket.model_validate(payload)

    @staticmethod
    def _cognitive_state(state: RpAgentRunState) -> SetupCognitiveStateSnapshot | None:
        payload = state.get("cognitive_state")
        if not isinstance(payload, dict):
            return None
        return SetupCognitiveStateSnapshot.model_validate(payload)

    @staticmethod
    def _cognitive_state_summary(
        state: RpAgentRunState,
    ) -> SetupCognitiveStateSummary | None:
        payload = state.get("cognitive_state_summary")
        if not isinstance(payload, dict):
            return None
        return SetupCognitiveStateSummary.model_validate(payload)

    @staticmethod
    def _working_digest(state: RpAgentRunState) -> SetupWorkingDigest | None:
        payload = state.get("working_digest")
        if not isinstance(payload, dict):
            return None
        return SetupWorkingDigest.model_validate(payload)

    @staticmethod
    def _tool_outcomes(state: RpAgentRunState) -> list[SetupToolOutcome]:
        items = state.get("tool_outcomes") or []
        return [
            SetupToolOutcome.model_validate(item)
            for item in items
            if isinstance(item, dict)
        ]

    @staticmethod
    def _compact_summary(state: RpAgentRunState) -> SetupContextCompactSummary | None:
        payload = state.get("compact_summary")
        if not isinstance(payload, dict):
            return None
        return SetupContextCompactSummary.model_validate(payload)

    def _recent_assistant_questions(self, state: RpAgentRunState) -> list[str]:
        turn_input = self._turn_input(state)
        questions: list[str] = []
        for item in turn_input.conversation_messages:
            if item.get("role") != "assistant":
                continue
            content = str(item.get("content") or "").strip()
            if FinishPolicy.looks_like_question(content):
                questions.append(content)
        return questions[-4:]

    def _build_working_digest_payload(
        self,
        state: RpAgentRunState,
        *,
        turn_goal: SetupTurnGoal | None = None,
        working_plan: SetupWorkingPlan | None = None,
        pending_obligation: SetupPendingObligation | None | object = _SENTINEL,
        reflection_ticket: SetupReflectionTicket | None | object = _SENTINEL,
        cognitive_state: SetupCognitiveStateSnapshot | None = None,
        cognitive_state_summary: SetupCognitiveStateSummary | None = None,
        tool_outcomes: list[SetupToolOutcome] | None = None,
    ) -> dict[str, Any] | None:
        digest = self._working_digest(state) or SetupWorkingDigest()
        goal = turn_goal or self._turn_goal_model(state)
        plan = working_plan or self._working_plan_model(state)
        obligation = (
            self._pending_obligation(state)
            if pending_obligation is _SENTINEL
            else pending_obligation
        )
        ticket = (
            self._reflection_ticket(state)
            if reflection_ticket is _SENTINEL
            else reflection_ticket
        )
        snapshot = cognitive_state or self._cognitive_state(state)
        summary = cognitive_state_summary or self._cognitive_state_summary(state)
        retained_outcomes = tool_outcomes or self._tool_outcomes(state)
        if goal is not None:
            digest.current_goal = goal.goal_summary
        if plan is not None:
            digest.next_focus = plan.current_priority or digest.next_focus
        if snapshot is not None and snapshot.discussion_state is not None:
            digest.next_focus = (
                snapshot.discussion_state.next_focus or digest.next_focus
            )
            digest.rejected_directions = [
                item.label
                for item in snapshot.discussion_state.candidate_directions
                if item.status == "discarded"
            ][:4]
        if summary is not None and summary.open_questions:
            digest.open_questions = list(summary.open_questions[:4])
        draft_refs = list(digest.draft_refs[:6])
        if snapshot is not None and snapshot.active_truth_write is not None:
            target_ref = str(snapshot.active_truth_write.target_ref or "").strip()
            if target_ref and target_ref not in draft_refs:
                draft_refs.append(target_ref)
        for item in retained_outcomes:
            for ref in item.updated_refs:
                value = str(ref or "").strip()
                if value and value not in draft_refs:
                    draft_refs.append(value)
                if len(draft_refs) >= 6:
                    break
            if len(draft_refs) >= 6:
                break
        digest.draft_refs = draft_refs[:6]
        digest.pending_obligation = (
            obligation.reason
            if isinstance(obligation, SetupPendingObligation) and obligation.unresolved
            else None
        )

        blockers: list[str] = []
        context_bundle = self._context_bundle(self._turn_input(state))
        blocking_count = int(context_bundle.get("blocking_open_question_count") or 0)
        if blocking_count > 0:
            blockers.append(f"{blocking_count} blocking_open_question(s)")
        if summary is not None and summary.invalidated:
            blockers.append("cognitive_state_invalidated")
        if summary is not None:
            blockers.extend(summary.remaining_open_issues[:2])
        if (
            isinstance(ticket, SetupReflectionTicket)
            and ticket.required_decision == "block_commit"
        ):
            blockers.append(ticket.summary)
        digest.commit_blockers = list(dict.fromkeys(blockers))[:4]

        if not any(
            (
                digest.current_goal,
                digest.next_focus,
                digest.pending_obligation,
                digest.open_questions,
                digest.rejected_directions,
                digest.draft_refs,
                digest.commit_blockers,
            )
        ):
            return None
        return digest.model_dump(mode="json", exclude_none=True)

    @staticmethod
    def _tool_relevance(
        tool_name: str, *, success: bool
    ) -> Literal[
        "cognitive",
        "draft",
        "question",
        "proposal",
        "read",
        "asset",
        "failure",
        "other",
    ]:
        if not success:
            return "failure"
        name = tool_name.removeprefix("rp_setup__")
        if name.startswith("setup.discussion.") or name.startswith("setup.chunk."):
            return "cognitive"
        if name.startswith("setup.stage_entry."):
            return "draft"
        if name.startswith("setup.question."):
            return "question"
        if name.startswith("setup.proposal."):
            return "proposal"
        if name.startswith("setup.asset."):
            return "asset"
        if name.startswith("setup.memory."):
            return "read"
        if name.startswith("setup.read."):
            return "read"
        return "other"

    def _tool_outcome_from_result(self, result: RuntimeToolResult) -> SetupToolOutcome:
        updated_refs: list[str] = []
        structured_payload = (
            result.structured_payload
            if isinstance(result.structured_payload, dict)
            else {}
        )
        content_payload = (
            structured_payload.get("content_payload")
            if isinstance(structured_payload.get("content_payload"), dict)
            else {}
        )
        raw_updated_refs = (
            content_payload.get("updated_refs")
            if isinstance(content_payload, dict)
            else None
        )
        if isinstance(raw_updated_refs, list):
            updated_refs = [str(item) for item in raw_updated_refs if item]
        summary = result.content_text
        raw_message = (
            content_payload.get("message")
            if isinstance(content_payload, dict)
            else None
        )
        if isinstance(raw_message, str) and raw_message:
            summary = raw_message
        elif not result.success:
            payload = ToolFailureClassifier.error_payload(result)
            summary = str(payload.get("message") or result.content_text)
        return SetupToolOutcome(
            tool_name=result.tool_name,
            success=result.success,
            summary=summary[:240],
            updated_refs=updated_refs[:6],
            error_code=result.error_code,
            relevance=self._tool_relevance(result.tool_name, success=result.success),
            recorded_at=datetime.now(timezone.utc),
        )

    def _append_loop_trace(
        self,
        state: RpAgentRunState,
        update: RpAgentRunState,
        *,
        decision_site: Literal[
            "inspect_model_output",
            "assess_progress",
            "reflect_if_needed",
            "finalize_success",
            "finalize_failure",
        ],
        tool_names: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        merged_state = cast(RpAgentRunState, dict(state))
        merged_state.update(update)
        existing = list(state.get("loop_trace", []))
        frame = self._build_loop_trace_frame(
            merged_state,
            decision_site=decision_site,
            tool_names=tool_names,
        )
        existing.append(frame.model_dump(mode="json", exclude_none=True))
        return existing

    def _build_loop_trace_frame(
        self,
        state: RpAgentRunState,
        *,
        decision_site: Literal[
            "inspect_model_output",
            "assess_progress",
            "reflect_if_needed",
            "finalize_success",
            "finalize_failure",
        ],
        tool_names: list[str] | None = None,
    ) -> SetupReActTraceFrame:
        resolved_tool_names = (
            list(tool_names)
            if tool_names is not None
            else self._trace_tool_names(state)
        )
        assistant_text = str(state.get("assistant_text") or "")
        terminal_kind = FinishPolicy.terminal_output_kind(assistant_text)
        if state.get("error"):
            action_kind = "error"
        elif resolved_tool_names:
            action_kind = "tool_batch"
        elif not assistant_text.strip():
            action_kind = "empty"
        else:
            action_kind = "assistant_text"
        if terminal_kind == "ask_user":
            assistant_text_kind = "question"
        elif terminal_kind == "text":
            assistant_text_kind = "text"
        else:
            assistant_text_kind = "empty"
        latest_batch = self._latest_tool_batch_results(state)
        updated_refs: list[str] = []
        for result in latest_batch:
            for ref in self._tool_outcome_from_result(result).updated_refs:
                if ref not in updated_refs:
                    updated_refs.append(ref)
        turn_goal_payload = state.get("turn_goal")
        working_plan_payload = state.get("working_plan")
        reflection_ticket_payload = state.get("reflection_ticket")
        return SetupReActTraceFrame(
            round_no=int(state.get("round_no") or 0),
            decision_site=decision_site,
            goal=dict(turn_goal_payload)
            if isinstance(turn_goal_payload, dict)
            else None,
            plan=(
                dict(working_plan_payload)
                if isinstance(working_plan_payload, dict)
                else None
            ),
            action={
                "kind": action_kind,
                "tool_names": resolved_tool_names,
                "assistant_text_kind": assistant_text_kind,
            },
            observation={
                "tool_result_count": len(latest_batch),
                "tool_failure_count": len(
                    [item for item in latest_batch if not item.success]
                ),
                "updated_refs": updated_refs,
                "warnings": list(state.get("warnings", [])),
            },
            reflection=(
                dict(reflection_ticket_payload)
                if isinstance(reflection_ticket_payload, dict)
                else None
            ),
            decision={
                "next_action": str(state.get("next_action") or ""),
                "continue_reason": state.get("continue_reason"),
                "finish_reason": state.get("finish_reason"),
                "repair_route": state.get("repair_route"),
            },
        )

    def _inspect_output_payload(
        self,
        state: RpAgentRunState,
        message_payload: dict[str, Any],
    ) -> SetupOutputInspection:
        if state.get("error"):
            return SetupOutputInspection(
                classification="provider_schema_error",
                public_text_candidate="",
                repair_observation={"reason": "upstream_error"},
                private_diagnostics={
                    "error": state.get("error"),
                    "finish_reason": state.get("finish_reason"),
                    "model_gateway_diagnostics": state.get("model_gateway_diagnostics"),
                },
                finish_reason_candidate=str(
                    state.get("finish_reason") or "upstream_error"
                ),
            )

        raw_content = message_payload.get("content")
        assistant_text = raw_content if isinstance(raw_content, str) else ""
        raw_tool_calls = [
            call
            # Some OpenAI-compatible providers serialize a plain assistant reply as
            # ``tool_calls: null`` rather than omitting the field. Treat that as no
            # tool call so text-only turn completion can finalize normally.
            for call in (message_payload.get("tool_calls") or [])
            if isinstance(call, dict)
        ]

        runtime_tool_calls: list[RuntimeToolCall] = []
        malformed_calls: list[dict[str, Any]] = []
        for position, call in enumerate(raw_tool_calls):
            tool_name = self._tool_name(call).strip()
            arguments, argument_error = self._parse_tool_arguments(call)
            if not tool_name or argument_error is not None:
                malformed_calls.append(
                    {
                        "position": position,
                        "tool_name": tool_name or None,
                        "error": argument_error or "missing_tool_name",
                    }
                )
                continue
            runtime_tool_calls.append(
                RuntimeToolCall(
                    call_id=self._tool_call_id(call),
                    tool_name=tool_name,
                    arguments=arguments,
                    source_round=int(state.get("round_no") or 0),
                )
            )

        if malformed_calls:
            return SetupOutputInspection(
                classification="malformed_tool_call",
                public_text_candidate="",
                repair_observation={
                    "reason": "malformed_tool_call",
                    "malformed_calls": malformed_calls,
                },
                private_diagnostics={
                    "malformed_calls": malformed_calls,
                    "raw_tool_call_count": len(raw_tool_calls),
                },
                continue_reason_candidate="completion_guard_retry",
                finish_reason_candidate="repair_obligation_unfulfilled",
            )

        if runtime_tool_calls:
            classification: Literal["mixed_text_and_tool_call", "real_tool_call"] = (
                "mixed_text_and_tool_call"
                if assistant_text.strip()
                else "real_tool_call"
            )
            return SetupOutputInspection(
                classification=classification,
                public_text_candidate="",
                tool_calls=runtime_tool_calls,
                private_diagnostics={
                    "raw_text_present": bool(assistant_text.strip()),
                    "tool_call_count": len(runtime_tool_calls),
                },
                continue_reason_candidate="tool_call_batch_pending",
            )

        if assistant_text and self._looks_like_pseudo_tool_call_text(assistant_text):
            return SetupOutputInspection(
                classification="pseudo_tool_text",
                public_text_candidate="",
                repair_observation={"reason": "pseudo_tool_call_text_emitted"},
                private_diagnostics={
                    "raw_text_length": len(assistant_text),
                    "detector": "pseudo_tool_call_text",
                },
                continue_reason_candidate="completion_guard_retry",
                finish_reason_candidate="repair_obligation_unfulfilled",
            )

        if not assistant_text.strip():
            return SetupOutputInspection(
                classification="empty_output",
                public_text_candidate="",
                repair_observation={"reason": "assistant_output_empty"},
                private_diagnostics={"raw_text_length": len(assistant_text)},
                continue_reason_candidate="completion_guard_retry",
            )

        return SetupOutputInspection(
            classification="normal_text",
            public_text_candidate=assistant_text,
        )

    def _graph_invoke_config(self) -> dict[str, Any]:
        # One semantic round can traverse multiple LangGraph nodes
        # (derive/plan/request/call/inspect plus optional tool/apply/assess/reflect),
        # so the framework recursion limit must stay comfortably above
        # profile.max_rounds or LangGraph will fail before runtime-owned stop
        # conditions such as max_rounds_exceeded can fire.
        return {"recursion_limit": self._graph_recursion_limit()}

    def _graph_recursion_limit(self) -> int:
        max_rounds = max(int(self._profile.max_rounds or 1), 1)
        return max(32, (max_rounds * 10) + 8)

    @classmethod
    def _looks_like_pseudo_tool_call_text(cls, text: str) -> bool:
        return bool(cls._PSEUDO_TOOL_CALL_TEXT_RE.search(str(text or "")))

    @staticmethod
    def _trace_tool_names(state: RpAgentRunState) -> list[str]:
        names: list[str] = []
        for item in state.get("pending_tool_calls", []):
            if not isinstance(item, dict):
                continue
            name = str(item.get("tool_name") or "").strip()
            if name:
                names.append(name)
        return names

    @staticmethod
    def _latest_tool_batch_results(state: RpAgentRunState) -> list[RuntimeToolResult]:
        results: list[RuntimeToolResult] = []
        for item in state.get("latest_tool_batch", []):
            if not isinstance(item, dict):
                continue
            results.append(RuntimeToolResult.model_validate(item))
        return results

    @staticmethod
    def _all_tool_results(state: RpAgentRunState) -> list[RuntimeToolResult]:
        results: list[RuntimeToolResult] = []
        for item in state.get("tool_results", []):
            if not isinstance(item, dict):
                continue
            results.append(RuntimeToolResult.model_validate(item))
        return results

    @staticmethod
    def _turn_goal_model(state: RpAgentRunState) -> SetupTurnGoal | None:
        payload = state.get("turn_goal")
        if not isinstance(payload, dict):
            return None
        return SetupTurnGoal.model_validate(payload)

    @staticmethod
    def _working_plan_model(state: RpAgentRunState) -> SetupWorkingPlan | None:
        payload = state.get("working_plan")
        if not isinstance(payload, dict):
            return None
        return SetupWorkingPlan.model_validate(payload)

    @staticmethod
    def _tool_result_warning_codes(tool_results: list[RuntimeToolResult]) -> list[str]:
        warnings: list[str] = []
        for result in tool_results:
            payload = result.structured_payload or {}
            for key in ("result_payload", "content_payload"):
                content = payload.get(key)
                if not isinstance(content, dict):
                    continue
                for item in content.get("warnings") or []:
                    warning = str(item or "").strip()
                    if warning and warning not in warnings:
                        warnings.append(warning)
        return warnings

    @staticmethod
    def _normalize_message_dict(message: dict[str, Any]) -> dict[str, Any]:
        return ChatMessage.model_validate(
            {
                "role": message.get("role"),
                "content": message.get("content"),
                "name": message.get("name"),
                "tool_calls": message.get("tool_calls"),
                "tool_call_id": message.get("tool_call_id"),
            }
        ).model_dump(mode="json", exclude_none=True)

    @staticmethod
    def _coerce_response_dict(response: Any) -> dict[str, Any]:
        if hasattr(response, "model_dump"):
            return response.model_dump(exclude_none=True)
        if isinstance(response, dict):
            return response
        raise TypeError(
            f"Unsupported chat completion response type: {type(response)!r}"
        )

    def _model_gateway_failure_update(
        self,
        *,
        failure_kind: str,
        message: str,
        provider_error_type: str | None = None,
        private_details: dict[str, Any] | None = None,
    ) -> RpAgentRunState:
        public_error, diagnostics = self._model_gateway_error_payload(
            failure_kind=failure_kind,
            raw_error={
                "message": message,
                "type": provider_error_type or failure_kind,
                "private_details": dict(private_details or {}),
            },
        )
        return {
            "status": "model_gateway_failed",
            "latest_response": {"message": {"content": "", "tool_calls": []}},
            "assistant_text": "",
            "error": public_error,
            "finish_reason": "upstream_error",
            "model_gateway_diagnostics": diagnostics,
        }

    @staticmethod
    def _model_gateway_error_payload(
        *,
        failure_kind: str,
        raw_error: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        raw_error_payload = raw_error if isinstance(raw_error, dict) else {}
        raw_message = str(
            raw_error_payload.get("message")
            or raw_error_payload.get("detail")
            or "model_gateway_failed"
        )
        provider_error_type = (
            str(raw_error_payload.get("type"))
            if raw_error_payload.get("type") is not None
            else None
        )
        diagnostics = SetupModelGatewayDiagnostics(
            failure_kind=failure_kind,
            message=raw_message,
            provider_error_type=provider_error_type,
            private_details={
                "raw_error": raw_error,
            },
        ).model_dump(mode="json", exclude_none=True)
        public_error = {
            "message": _RuntimeRunDriver._public_model_gateway_error_message(
                raw_message
            ),
            "type": "model_gateway_failed",
            "code": failure_kind,
            "failure_layer": "model_gateway",
        }
        return public_error, diagnostics

    @staticmethod
    def _public_model_gateway_error_message(message: str) -> str:
        raw_message = str(message or "").strip()
        if not raw_message:
            return "Model provider request failed."
        lowered = raw_message.lower()
        private_markers = (
            "api_key",
            "authorization",
            "bearer ",
            "private",
            "raw_provider_delta",
            "sk-",
            "stack",
            "stacktrace",
            "token",
            "trace",
            "traceback",
        )
        if any(marker in lowered for marker in private_markers):
            return "Model provider request failed."
        return raw_message[:240]

    @staticmethod
    def _merge_model_gateway_private_event(
        diagnostics: dict[str, Any] | None,
        *,
        event_type: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if diagnostics is None:
            diagnostics = SetupModelGatewayDiagnostics(
                failure_kind="private_stream_event",
                message="private provider stream events were filtered",
                private_details={"private_events": []},
            ).model_dump(mode="json", exclude_none=True)
        private_details = diagnostics.setdefault("private_details", {})
        private_events = private_details.setdefault("private_events", [])
        if isinstance(private_events, list):
            private_events.append({"type": event_type, "payload": dict(payload)})
        return diagnostics

    @staticmethod
    def _extract_message_payload(response: dict[str, Any]) -> dict[str, Any]:
        message = response.get("message")
        if isinstance(message, dict):
            return message
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict):
                nested = choice.get("message")
                if isinstance(nested, dict):
                    return nested
        return {}

    @staticmethod
    def _runtime_tool_call_to_openai(call: RuntimeToolCall) -> dict[str, Any]:
        return {
            "id": call.call_id,
            "type": "function",
            "function": {
                "name": call.tool_name,
                "arguments": json.dumps(
                    call.arguments,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            },
        }

    @staticmethod
    def _merge_stream_tool_calls(
        accumulators: dict[int, dict[str, Any]],
        raw_calls: Any,
    ) -> None:
        if not isinstance(raw_calls, list):
            return

        for position, raw_call in enumerate(raw_calls):
            if not isinstance(raw_call, dict):
                continue

            index = raw_call.get("index")
            key = index if isinstance(index, int) else position
            current = accumulators.setdefault(key, {"type": "function", "function": {}})

            for field, value in raw_call.items():
                if field in {"index", "function"}:
                    continue
                if value is None:
                    continue
                if isinstance(value, str) and not value:
                    continue
                current[field] = value

            raw_function = raw_call.get("function")
            if not isinstance(raw_function, dict):
                continue

            current_function = current.setdefault("function", {})
            for field, value in raw_function.items():
                if field == "arguments":
                    if isinstance(value, str):
                        existing = current_function.get("arguments")
                        current_function["arguments"] = (
                            existing if isinstance(existing, str) else ""
                        ) + value
                    elif value is not None:
                        current_function["arguments"] = value
                    continue
                if value is None:
                    continue
                if isinstance(value, str) and not value:
                    continue
                current_function[field] = value

    @staticmethod
    def _finalize_stream_tool_calls(
        accumulators: dict[int, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [accumulators[key] for key in sorted(accumulators)]

    @staticmethod
    def _tool_call_id(call: dict[str, Any]) -> str:
        existing = call.get("id")
        if existing:
            return str(existing)
        generated = f"call_{uuid.uuid4().hex[:8]}"
        call["id"] = generated
        return generated

    @staticmethod
    def _tool_name(call: dict[str, Any]) -> str:
        function = call.get("function")
        if isinstance(function, dict):
            return str(function.get("name") or "")
        return ""

    @staticmethod
    def _tool_arguments(call: dict[str, Any]) -> dict[str, Any]:
        arguments, _error = _RuntimeRunDriver._parse_tool_arguments(call)
        return arguments

    @staticmethod
    def _parse_tool_arguments(
        call: dict[str, Any],
    ) -> tuple[dict[str, Any], str | None]:
        function = call.get("function")
        raw_args = (
            function.get("arguments", "{}") if isinstance(function, dict) else "{}"
        )
        if isinstance(raw_args, dict):
            return raw_args, None
        if not isinstance(raw_args, str):
            return {}, "arguments_not_string_or_object"
        try:
            parsed = json.loads(raw_args)
        except json.JSONDecodeError:
            return {}, "arguments_json_decode_error"
        if not isinstance(parsed, dict):
            return {}, "arguments_not_object"
        return parsed, None

    @staticmethod
    def _parse_sse_payload(line: str) -> dict[str, Any] | None:
        if not line.startswith("data: "):
            return None
        data_str = line[6:].strip()
        if not data_str or data_str == "[DONE]":
            return None
        try:
            payload = json.loads(data_str)
        except json.JSONDecodeError:
            raise _ProviderStreamPayloadError(
                "Provider stream payload was invalid JSON."
            ) from None
        if not isinstance(payload, dict):
            raise _ProviderStreamPayloadError(
                "Provider stream payload was not an object."
            )
        return payload
