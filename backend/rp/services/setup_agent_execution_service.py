"""SetupAgent execution layer over the runtime-v2 setup harness."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from typing import Any, AsyncIterator, Literal, cast

from models.chat import ChatCompletionRequest, ChatMessage
from rp.agent_runtime.adapters import SetupRuntimeAdapter
from rp.agent_runtime.contracts import (
    RpAgentTurnResult,
    RpAgentTurnInput,
    RuntimeProfile,
    SetupContextCompactSummary,
    SetupContextGovernanceReport,
    SetupToolOutcome,
    SetupWorkingDigest,
)
from rp.agent_runtime.executor import RpAgentRuntimeExecutor
from rp.models.setup_agent import (
    SetupAgentTurnRequest,
    SetupAgentTurnResponse,
)
from rp.models.setup_handoff import SetupContextBuilderInput
from rp.models.setup_stage import SetupStageId
from rp.models.setup_workspace import SetupStepId
from rp.services.setup_agent_runtime_state_service import SetupAgentRuntimeStateService
from rp.services.setup_context_builder import SetupContextBuilder
from rp.services.setup_context_compaction_service import SetupContextCompactionService
from rp.services.setup_context_governor import SetupContextGovernorService
from rp.services.setup_workspace_service import SetupWorkspaceService
from services.litellm_service import LiteLLMService, get_litellm_service
from services.langfuse_service import get_langfuse_service
from services.model_registry import get_model_registry_service
from services.provider_registry import get_provider_registry_service

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _SetupTurnLaunch:
    """Thin outer-harness preflight result shared by turn/stream entrypoints."""

    request: SetupAgentTurnRequest
    workspace: Any
    current_step: SetupStepId
    current_stage: SetupStageId | None
    model_name: str
    provider: Any


@dataclass(frozen=True)
class _PreparedRuntimeV2Launch:
    """Runtime-v2 launch inputs derived from one prepared outer-harness turn."""

    turn_input: RpAgentTurnInput
    context_packet: Any
    profile: RuntimeProfile


class SetupAgentExecutionService:
    """Execute one SetupAgent turn against SetupWorkspace and setup tools."""

    _STANDARD_CONTEXT_TOKEN_BUDGET = 2400
    _COMPACT_CONTEXT_TOKEN_BUDGET = 600
    _COMPACT_HISTORY_COUNT_THRESHOLD = 8
    _COMPACT_HISTORY_CHARS_THRESHOLD = 4000
    _COMPACT_ESTIMATED_INPUT_TOKEN_THRESHOLD = 1800
    _COMPACT_OBSERVED_PROMPT_TOKEN_THRESHOLD = 1800
    _COMPACT_OBSERVED_TOTAL_TOKEN_THRESHOLD = 2400
    _COMPACT_USER_EDIT_THRESHOLD = 3
    _COMPACT_PROMPT_MAX_TOKENS = 1200

    def __init__(
        self,
        *,
        workspace_service: SetupWorkspaceService,
        context_builder: SetupContextBuilder,
        llm_service: LiteLLMService | None = None,
        runtime_executor: RpAgentRuntimeExecutor | None = None,
        adapter: SetupRuntimeAdapter | None = None,
        runtime_state_service: SetupAgentRuntimeStateService | None = None,
        context_governor: SetupContextGovernorService | None = None,
    ) -> None:
        self._workspace_service = workspace_service
        self._context_builder = context_builder
        self._llm_service = llm_service or get_litellm_service()
        self._runtime_executor = runtime_executor
        self._adapter = adapter
        self._runtime_state_service = runtime_state_service
        self._context_governor = context_governor or SetupContextGovernorService()
        self._last_runtime_result: RpAgentTurnResult | None = None
        self._last_runtime_usage_by_workspace_step: dict[
            tuple[str, str], dict[str, Any]
        ] = {}

    @property
    def last_runtime_result(self) -> RpAgentTurnResult | None:
        return self._last_runtime_result

    async def run_turn(self, request: SetupAgentTurnRequest) -> SetupAgentTurnResponse:
        launch = self._prepare_turn_launch(request)
        logger.info(
            "[SETUP_AGENT] run_turn_start workspace_id=%s story_id=%s model_id=%s provider_id=%s model=%s target_stage=%s target_step=%s history_count=%s",
            launch.request.workspace_id,
            launch.workspace.story_id,
            launch.request.model_id,
            launch.request.provider_id or "",
            launch.model_name,
            launch.current_stage.value if launch.current_stage is not None else "",
            launch.current_step.value,
            len(launch.request.history),
        )
        result = await self._run_turn_v2(launch=launch)
        logger.info(
            "[SETUP_AGENT] run_turn_done workspace_id=%s model_id=%s finish_reason=%s assistant_chars=%s",
            launch.request.workspace_id,
            launch.request.model_id,
            result.finish_reason,
            len(result.assistant_text),
        )
        adapter, _ = self._require_runtime_v2_components()
        return adapter.to_turn_response(result)

    async def run_turn_stream(
        self, request: SetupAgentTurnRequest
    ) -> AsyncIterator[str]:
        launch = self._prepare_turn_launch(request)
        logger.info(
            "[SETUP_AGENT] run_turn_stream_start workspace_id=%s story_id=%s model_id=%s provider_id=%s model=%s target_stage=%s target_step=%s history_count=%s",
            launch.request.workspace_id,
            launch.workspace.story_id,
            launch.request.model_id,
            launch.request.provider_id or "",
            launch.model_name,
            launch.current_stage.value if launch.current_stage is not None else "",
            launch.current_step.value,
            len(launch.request.history),
        )
        async for chunk in self._run_turn_stream_v2(launch=launch):
            yield chunk

    def _prepare_turn_launch(self, request: SetupAgentTurnRequest) -> _SetupTurnLaunch:
        workspace = self._require_workspace(request.workspace_id)
        current_step, current_stage = self._resolve_turn_selection(
            request=request,
            workspace=workspace,
        )
        self._ensure_agent_model_compatible(request.model_id)
        provider = self._resolve_provider(
            model_id=request.model_id,
            provider_id=request.provider_id,
        )
        model_name = self._resolve_model_name(
            model_id=request.model_id,
            fallback_provider_id=request.provider_id,
        )
        return _SetupTurnLaunch(
            request=request,
            workspace=workspace,
            current_step=current_step,
            current_stage=current_stage,
            model_name=model_name,
            provider=provider,
        )

    @classmethod
    def _resolve_turn_selection(
        cls,
        *,
        request: SetupAgentTurnRequest,
        workspace,
    ) -> tuple[SetupStepId, SetupStageId | None]:
        target_stage = request.target_stage
        if target_stage is not None:
            if target_stage not in workspace.stage_plan:
                raise ValueError(
                    "setup_target_stage_not_in_stage_plan:"
                    f"{target_stage.value}:{workspace.mode.value}"
                )
            expected_step = SetupWorkspaceService._LEGACY_STEP_FOR_STAGE[target_stage]
            if request.target_step is not None and request.target_step != expected_step:
                raise ValueError(
                    "setup_target_stage_step_mismatch:"
                    f"{target_stage.value}:{request.target_step.value}:{expected_step.value}"
                )
        current_stage = target_stage
        if current_stage is None and (
            request.target_step is None or request.target_step == workspace.current_step
        ):
            current_stage = workspace.current_stage
        if request.target_step is not None:
            current_step = request.target_step
        elif current_stage is not None:
            current_step = SetupWorkspaceService._LEGACY_STEP_FOR_STAGE[current_stage]
        else:
            current_step = workspace.current_step
        return current_step, current_stage

    @classmethod
    def _context_token_budget(
        cls,
        request: SetupAgentTurnRequest,
        *,
        estimated_input_tokens: int | None = None,
        previous_usage: dict[str, Any] | None = None,
    ) -> int:
        history_chars = sum(len(item.content or "") for item in request.history)
        estimated_tokens = (
            cls._estimate_input_tokens(request)
            if estimated_input_tokens is None
            else estimated_input_tokens
        )
        previous_prompt_tokens = (
            previous_usage.get("prompt_tokens")
            if isinstance(previous_usage, dict)
            else None
        )
        previous_total_tokens = (
            previous_usage.get("total_tokens")
            if isinstance(previous_usage, dict)
            else None
        )
        if (
            len(request.history) >= cls._COMPACT_HISTORY_COUNT_THRESHOLD
            or history_chars >= cls._COMPACT_HISTORY_CHARS_THRESHOLD
            or estimated_tokens >= cls._COMPACT_ESTIMATED_INPUT_TOKEN_THRESHOLD
            or (
                previous_prompt_tokens is not None
                and previous_prompt_tokens
                >= cls._COMPACT_OBSERVED_PROMPT_TOKEN_THRESHOLD
            )
            or (
                previous_total_tokens is not None
                and previous_total_tokens >= cls._COMPACT_OBSERVED_TOTAL_TOKEN_THRESHOLD
            )
            or len(request.user_edit_delta_ids) >= cls._COMPACT_USER_EDIT_THRESHOLD
        ):
            return cls._COMPACT_CONTEXT_TOKEN_BUDGET
        return cls._STANDARD_CONTEXT_TOKEN_BUDGET

    @classmethod
    def _context_profile_reasons(
        cls,
        request: SetupAgentTurnRequest,
        *,
        estimated_input_tokens: int | None = None,
        previous_usage: dict[str, Any] | None = None,
    ) -> list[str]:
        history_chars = sum(len(item.content or "") for item in request.history)
        estimated_tokens = (
            cls._estimate_input_tokens(request)
            if estimated_input_tokens is None
            else estimated_input_tokens
        )
        previous_prompt_tokens = (
            previous_usage.get("prompt_tokens")
            if isinstance(previous_usage, dict)
            else None
        )
        previous_total_tokens = (
            previous_usage.get("total_tokens")
            if isinstance(previous_usage, dict)
            else None
        )
        reasons: list[str] = []
        if len(request.history) >= cls._COMPACT_HISTORY_COUNT_THRESHOLD:
            reasons.append("history_count_threshold")
        if history_chars >= cls._COMPACT_HISTORY_CHARS_THRESHOLD:
            reasons.append("history_chars_threshold")
        if estimated_tokens >= cls._COMPACT_ESTIMATED_INPUT_TOKEN_THRESHOLD:
            reasons.append("estimated_input_tokens_threshold")
        if (
            previous_prompt_tokens is not None
            and previous_prompt_tokens >= cls._COMPACT_OBSERVED_PROMPT_TOKEN_THRESHOLD
        ) or (
            previous_total_tokens is not None
            and previous_total_tokens >= cls._COMPACT_OBSERVED_TOTAL_TOKEN_THRESHOLD
        ):
            reasons.append("observed_usage_threshold")
        if len(request.user_edit_delta_ids) >= cls._COMPACT_USER_EDIT_THRESHOLD:
            reasons.append("user_edit_threshold")
        if not reasons:
            reasons.append("default_standard_budget")
        return reasons

    @staticmethod
    def _estimate_input_tokens(request: SetupAgentTurnRequest) -> int:
        """Fallback approximation used when tokenizer preflight is unavailable."""

        char_count = len(request.user_prompt or "") + sum(
            len(item.content or "") for item in request.history
        )
        message_overhead = (len(request.history) + 1) * 8
        return max(1, (char_count + 3) // 4 + message_overhead)

    def _preflight_input_token_count(
        self,
        *,
        request: SetupAgentTurnRequest,
        model_name: str,
        provider: Any,
    ) -> tuple[int, str]:
        """Return current-turn pre-call token pressure and its source."""

        counter = getattr(self._llm_service, "count_request_tokens", None)
        if callable(counter):
            token_request = ChatCompletionRequest(
                model=model_name,
                model_id=request.model_id,
                messages=[
                    ChatMessage(role=item.role, content=item.content)
                    for item in request.history
                ]
                + [ChatMessage(role="user", content=request.user_prompt or "")],
                stream=False,
                provider_id=request.provider_id,
                provider=provider,
            )
            try:
                token_count = counter(token_request)
                if isinstance(token_count, dict):
                    prompt_tokens = token_count.get("prompt_tokens")
                    if prompt_tokens is not None:
                        return max(int(prompt_tokens), 0), str(
                            token_count.get("source") or "litellm_token_counter"
                        )
            except Exception as exc:  # pragma: no cover - defensive fallback path
                logger.debug(
                    "SetupAgent token preflight fell back to approximation: %s",
                    exc,
                )
        return self._estimate_input_tokens(request), "approx_chars_div_4"

    @staticmethod
    def _usage_from_runtime_result(
        result: RpAgentTurnResult | None,
    ) -> dict[str, Any] | None:
        if result is None or not isinstance(result.structured_payload, dict):
            return None
        latest_response = result.structured_payload.get("latest_response")
        if not isinstance(latest_response, dict):
            return None
        usage = (
            latest_response.get("usage")
            or latest_response.get("usage_metadata")
            or latest_response.get("usageMetadata")
        )
        if not isinstance(usage, dict):
            return None
        return SetupAgentExecutionService._normalize_provider_usage(usage)

    @staticmethod
    def _normalize_provider_usage(usage: dict[str, Any]) -> dict[str, Any] | None:
        """Normalize provider-returned token usage without discarding raw details."""

        prompt_tokens = SetupAgentExecutionService._first_int(
            usage,
            "prompt_tokens",
            "input_tokens",
            "prompt_token_count",
            "promptTokenCount",
            "inputTokenCount",
        )
        completion_tokens = SetupAgentExecutionService._first_int(
            usage,
            "completion_tokens",
            "output_tokens",
            "candidates_token_count",
            "candidatesTokenCount",
            "outputTokenCount",
        )
        total_tokens = SetupAgentExecutionService._first_int(
            usage,
            "total_tokens",
            "total_token_count",
            "totalTokenCount",
        )
        prompt_details = usage.get("prompt_tokens_details")
        if not isinstance(prompt_details, dict):
            prompt_details = {}
        completion_details = usage.get("completion_tokens_details")
        if not isinstance(completion_details, dict):
            completion_details = {}
        input_details = usage.get("input_token_details")
        if not isinstance(input_details, dict):
            input_details = {}
        output_details = usage.get("output_token_details")
        if not isinstance(output_details, dict):
            output_details = {}

        cached_tokens = SetupAgentExecutionService._first_available_int(
            SetupAgentExecutionService._first_int(
                usage,
                "cached_tokens",
                "cache_read_input_tokens",
                "cacheReadInputTokens",
            ),
            SetupAgentExecutionService._first_int(prompt_details, "cached_tokens"),
            SetupAgentExecutionService._first_int(
                input_details,
                "cache_read",
                "cached_tokens",
            ),
        )
        reasoning_tokens = SetupAgentExecutionService._first_available_int(
            SetupAgentExecutionService._first_int(
                usage,
                "reasoning_tokens",
                "thoughts_token_count",
                "thoughtsTokenCount",
            ),
            SetupAgentExecutionService._first_int(
                completion_details,
                "reasoning_tokens",
            ),
            SetupAgentExecutionService._first_int(
                output_details,
                "reasoning",
                "reasoning_tokens",
            ),
        )
        cache_creation_input_tokens = SetupAgentExecutionService._first_available_int(
            SetupAgentExecutionService._first_int(
                usage,
                "cache_creation_input_tokens",
                "cacheCreationInputTokens",
            ),
            SetupAgentExecutionService._first_int(
                input_details,
                "cache_creation",
            ),
        )
        cache_read_input_tokens = SetupAgentExecutionService._first_available_int(
            SetupAgentExecutionService._first_int(
                usage,
                "cache_read_input_tokens",
                "cacheReadInputTokens",
            ),
            SetupAgentExecutionService._first_int(input_details, "cache_read"),
            SetupAgentExecutionService._first_int(prompt_details, "cached_tokens"),
        )

        if (
            prompt_tokens is None
            and completion_tokens is None
            and total_tokens is None
            and cached_tokens is None
            and reasoning_tokens is None
            and cache_creation_input_tokens is None
            and cache_read_input_tokens is None
        ):
            return None
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "cached_tokens": cached_tokens,
            "reasoning_tokens": reasoning_tokens,
            "cache_creation_input_tokens": cache_creation_input_tokens,
            "cache_read_input_tokens": cache_read_input_tokens,
            "source": "provider_usage_metadata",
            "token_details": {
                "prompt_tokens_details": dict(prompt_details),
                "completion_tokens_details": dict(completion_details),
                "input_token_details": dict(input_details),
                "output_token_details": dict(output_details),
            },
            "raw_usage": dict(usage),
        }

    @staticmethod
    def _first_int(mapping: dict[str, Any], *keys: str) -> int | None:
        for key in keys:
            value = mapping.get(key)
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _first_available_int(*values: int | None) -> int | None:
        for value in values:
            if value is not None:
                return value
        return None

    def _previous_usage_for_turn(
        self,
        *,
        workspace_id: str,
        step_id: SetupStepId,
    ) -> dict[str, Any] | None:
        return self._last_runtime_usage_by_workspace_step.get(
            (workspace_id, step_id.value)
        )

    def _record_runtime_usage(
        self,
        *,
        workspace_id: str,
        step_id: SetupStepId,
        result: RpAgentTurnResult,
    ) -> None:
        usage = self._usage_from_runtime_result(result)
        if usage is None:
            return
        self._last_runtime_usage_by_workspace_step[(workspace_id, step_id.value)] = (
            usage
        )

    @staticmethod
    def _build_context_report(
        *,
        request: SetupAgentTurnRequest,
        context_packet,
        governance_metadata: dict[str, Any],
        retained_tool_outcomes: list[SetupToolOutcome],
        compact_summary: SetupContextCompactSummary | None,
        existing_summary: SetupContextCompactSummary | None,
    ) -> SetupContextGovernanceReport:
        raw_history_chars = sum(len(item.content or "") for item in request.history)
        summary_strategy = cast(
            Literal["none", "deterministic_prefix_summary", "compact_prompt_summary"],
            governance_metadata.get("summary_strategy") or "none",
        )
        summary_action = cast(
            Literal["none", "reused_existing", "updated_existing", "rebuilt"],
            governance_metadata.get("summary_action") or "none",
        )
        return SetupContextGovernanceReport(
            context_profile=cast(
                Literal["standard", "compact"], context_packet.context_profile
            ),
            profile_reasons=SetupAgentExecutionService._context_profile_reasons(
                request,
                estimated_input_tokens=governance_metadata.get(
                    "estimated_input_tokens"
                ),
                previous_usage={
                    "prompt_tokens": governance_metadata.get("previous_prompt_tokens"),
                    "total_tokens": governance_metadata.get("previous_total_tokens"),
                },
            ),
            raw_history_count=len(request.history),
            raw_history_chars=raw_history_chars,
            estimated_input_tokens=governance_metadata.get("estimated_input_tokens"),
            input_token_count_source=governance_metadata.get(
                "input_token_count_source"
            ),
            previous_prompt_tokens=governance_metadata.get("previous_prompt_tokens"),
            previous_completion_tokens=governance_metadata.get(
                "previous_completion_tokens"
            ),
            previous_total_tokens=governance_metadata.get("previous_total_tokens"),
            previous_cached_tokens=governance_metadata.get("previous_cached_tokens"),
            previous_reasoning_tokens=governance_metadata.get(
                "previous_reasoning_tokens"
            ),
            previous_cache_creation_input_tokens=governance_metadata.get(
                "previous_cache_creation_input_tokens"
            ),
            previous_cache_read_input_tokens=governance_metadata.get(
                "previous_cache_read_input_tokens"
            ),
            previous_usage_source=governance_metadata.get("previous_usage_source"),
            previous_token_details=dict(
                governance_metadata.get("previous_token_details") or {}
            ),
            user_edit_delta_count=len(request.user_edit_delta_ids),
            prior_stage_handoff_count=len(context_packet.prior_stage_handoffs),
            raw_history_limit=int(governance_metadata.get("raw_history_limit") or 0),
            kept_history_count=int(governance_metadata.get("kept_history_count") or 0),
            compacted_history_count=int(
                governance_metadata.get("compacted_history_count") or 0
            ),
            retained_tool_outcome_count=len(retained_tool_outcomes),
            summary_strategy=summary_strategy,
            summary_action=summary_action,
            summary_line_count=(
                len(compact_summary.summary_lines) if compact_summary is not None else 0
            ),
            fallback_reason=governance_metadata.get("fallback_reason"),
        )

    def _resolve_model_name(
        self, *, model_id: str, fallback_provider_id: str | None
    ) -> str:
        entry = get_model_registry_service().get_entry(model_id)
        if entry is None:
            raise ValueError(f"Model not found: {model_id}")
        if fallback_provider_id and entry.provider_id != fallback_provider_id:
            raise ValueError(
                f"Model {model_id} does not belong to provider {fallback_provider_id}"
            )
        if not entry.is_enabled:
            raise ValueError(f"Model is disabled: {model_id}")
        return entry.model_name

    def _ensure_agent_model_compatible(self, model_id: str) -> None:
        entry = get_model_registry_service().get_entry(model_id)
        if entry is None:
            raise ValueError(f"Model not found: {model_id}")

        profile = entry.capability_profile
        if profile is not None:
            mode = str(profile.mode or "chat").strip().lower()
            if mode not in {"chat", "responses"}:
                raise ValueError(
                    f"Model {model_id} is not compatible with SetupAgent (mode={mode})"
                )
            if profile.supports_function_calling is not True:
                raise ValueError(
                    f"Model {model_id} is not compatible with SetupAgent (missing function calling support)"
                )
            if profile.supported_openai_params:
                supported_params = {
                    str(item).strip().lower()
                    for item in profile.supported_openai_params
                }
                if not {"tools", "tool_choice"}.issubset(supported_params):
                    raise ValueError(
                        f"Model {model_id} is not compatible with SetupAgent (missing tools/tool_choice support)"
                    )
            return

        capabilities = {str(item).strip().lower() for item in entry.capabilities}
        if "tool" not in capabilities:
            raise ValueError(
                f"Model {model_id} is not compatible with SetupAgent (missing tool capability)"
            )

    def _resolve_provider(self, *, model_id: str, provider_id: str | None):
        entry = get_model_registry_service().get_entry(model_id)
        if entry is None:
            raise ValueError(f"Model not found: {model_id}")
        resolved_provider_id = provider_id or entry.provider_id
        provider_entry = get_provider_registry_service().get_entry(resolved_provider_id)
        if provider_entry is None:
            raise ValueError(f"Provider not found: {resolved_provider_id}")
        if not provider_entry.is_enabled:
            raise ValueError(f"Provider is disabled: {resolved_provider_id}")
        return provider_entry.to_runtime_provider()

    def _require_workspace(self, workspace_id: str):
        workspace = self._workspace_service.get_workspace(workspace_id)
        if workspace is None:
            raise ValueError(f"SetupWorkspace not found: {workspace_id}")
        return workspace

    def _require_runtime_v2_components(
        self,
    ) -> tuple[SetupRuntimeAdapter, RpAgentRuntimeExecutor]:
        if self._adapter is None or self._runtime_executor is None:
            raise RuntimeError("SetupAgent runtime v2 components are not configured")
        return self._adapter, self._runtime_executor

    async def _run_turn_v2(
        self,
        *,
        launch: _SetupTurnLaunch,
    ) -> RpAgentTurnResult:
        adapter, runtime_executor = self._require_runtime_v2_components()
        langfuse = get_langfuse_service()
        with langfuse.start_as_current_observation(
            name="rp.setup.runtime_v2",
            as_type="agent",
            input={
                "workspace_id": launch.request.workspace_id,
                "target_step": launch.current_step.value,
                "target_stage": (
                    launch.request.target_stage.value
                    if launch.request.target_stage is not None
                    else None
                ),
                "current_stage": (
                    launch.current_stage.value
                    if launch.current_stage is not None
                    else None
                ),
                "model_id": launch.request.model_id,
                "provider_id": launch.request.provider_id,
                "history_count": len(launch.request.history),
                "user_prompt": launch.request.user_prompt,
            },
        ) as observation:
            prepared = await self._prepare_runtime_v2_launch(
                adapter=adapter,
                launch=launch,
                stream=False,
            )
            observation.update(metadata=self._runtime_v2_observation_metadata(prepared))
            result = await runtime_executor.run(
                prepared.turn_input,
                prepared.profile,
                llm_service=self._llm_service,
            )
            self._record_runtime_usage(
                workspace_id=launch.workspace.workspace_id,
                step_id=launch.current_step,
                result=result,
            )
            self._persist_runtime_v2_governance(
                workspace=launch.workspace,
                context_packet=prepared.context_packet,
                step_id=launch.current_step,
                result=result,
            )
            observation.update(output=self._runtime_v2_observation_output(result))
        self._last_runtime_result = result
        return result

    async def _run_turn_stream_v2(
        self,
        *,
        launch: _SetupTurnLaunch,
    ) -> AsyncIterator[str]:
        adapter, runtime_executor = self._require_runtime_v2_components()
        langfuse = get_langfuse_service()
        with langfuse.start_as_current_observation(
            name="rp.setup.runtime_v2.stream",
            as_type="agent",
            input={
                "workspace_id": launch.request.workspace_id,
                "target_step": launch.current_step.value,
                "target_stage": (
                    launch.request.target_stage.value
                    if launch.request.target_stage is not None
                    else None
                ),
                "current_stage": (
                    launch.current_stage.value
                    if launch.current_stage is not None
                    else None
                ),
                "model_id": launch.request.model_id,
                "provider_id": launch.request.provider_id,
                "history_count": len(launch.request.history),
                "user_prompt": launch.request.user_prompt,
                "stream": True,
            },
        ) as observation:
            prepared = await self._prepare_runtime_v2_launch(
                adapter=adapter,
                launch=launch,
                stream=True,
            )
            observation.update(metadata=self._runtime_v2_observation_metadata(prepared))
            async for chunk in runtime_executor.run_stream(
                prepared.turn_input,
                prepared.profile,
                llm_service=self._llm_service,
            ):
                yield chunk
            self._last_runtime_result = runtime_executor.last_result
            if self._last_runtime_result is not None:
                self._record_runtime_usage(
                    workspace_id=launch.workspace.workspace_id,
                    step_id=launch.current_step,
                    result=self._last_runtime_result,
                )
                self._persist_runtime_v2_governance(
                    workspace=launch.workspace,
                    context_packet=prepared.context_packet,
                    step_id=launch.current_step,
                    result=self._last_runtime_result,
                )
                observation.update(
                    output=self._runtime_v2_observation_output(
                        self._last_runtime_result
                    )
                )

    async def _prepare_runtime_v2_launch(
        self,
        *,
        adapter: SetupRuntimeAdapter,
        launch: _SetupTurnLaunch,
        stream: bool,
    ) -> _PreparedRuntimeV2Launch:
        turn_input, context_packet = await self._build_runtime_v2_turn_input(
            adapter=adapter,
            request=launch.request,
            workspace=launch.workspace,
            model_name=launch.model_name,
            provider=launch.provider,
        )
        return _PreparedRuntimeV2Launch(
            turn_input=turn_input.model_copy(update={"stream": stream}),
            context_packet=context_packet,
            profile=adapter.build_runtime_profile(),
        )

    @staticmethod
    def _runtime_v2_observation_output(result: RpAgentTurnResult) -> dict[str, Any]:
        payload = (
            result.structured_payload
            if isinstance(result.structured_payload, dict)
            else {}
        )
        return {
            "finish_reason": result.finish_reason,
            "continue_reason": payload.get("continue_reason")
            if isinstance(payload, dict)
            else None,
            "assistant_text": result.assistant_text,
            "warnings": list(result.warnings),
            "tool_invocation_count": len(result.tool_invocations),
            "tool_result_count": len(result.tool_results),
            "loop_trace_count": len(payload.get("loop_trace") or [])
            if isinstance(payload, dict)
            else 0,
        }

    @staticmethod
    def _runtime_v2_observation_metadata(
        prepared: "_PreparedRuntimeV2Launch",
    ) -> dict[str, Any]:
        """Single-source Langfuse observation metadata for runtime-v2 turns."""

        return {
            "skill_pack_name": prepared.turn_input.metadata.get("skill_pack_name"),
            "context_pipeline": prepared.turn_input.metadata.get("context_pipeline"),
        }

    async def _build_runtime_v2_turn_input(
        self,
        *,
        adapter: SetupRuntimeAdapter,
        request: SetupAgentTurnRequest,
        workspace,
        model_name: str,
        provider,
    ):
        current_step, current_stage = self._resolve_turn_selection(
            request=request,
            workspace=workspace,
        )
        estimated_input_tokens, input_token_count_source = (
            self._preflight_input_token_count(
                request=request,
                model_name=model_name,
                provider=provider,
            )
        )
        previous_usage = self._previous_usage_for_turn(
            workspace_id=workspace.workspace_id,
            step_id=current_step,
        )
        token_budget = self._context_token_budget(
            request,
            estimated_input_tokens=estimated_input_tokens,
            previous_usage=previous_usage,
        )
        context_packet = self._context_builder.build(
            SetupContextBuilderInput(
                mode=workspace.mode.value,
                workspace_id=workspace.workspace_id,
                current_step=current_step.value,
                current_stage=(
                    current_stage.value if current_stage is not None else None
                ),
                user_prompt=request.user_prompt,
                user_edit_delta_ids=list(request.user_edit_delta_ids),
                token_budget=token_budget,
            )
        )
        cognitive_state = (
            self._runtime_state_service.reconcile_snapshot(
                workspace=workspace,
                context_packet=context_packet,
                step_id=current_step,
            )
            if self._runtime_state_service is not None
            else None
        )
        cognitive_state_summary = (
            self._runtime_state_service.summarize_for_prompt(cognitive_state)
            if self._runtime_state_service is not None
            else None
        )
        open_questions = [
            question
            for question in workspace.open_questions
            if question.step_id == current_step
        ]
        blocking_open_questions = [
            question
            for question in open_questions
            if question.status.value == "open" and question.severity.value == "blocking"
        ]
        last_proposal_status = None
        proposals = [
            proposal
            for proposal in workspace.commit_proposals
            if proposal.step_id == current_step
        ]
        if proposals:
            last_proposal_status = max(
                proposals, key=lambda item: item.created_at
            ).status.value

        retained_tool_outcomes = (
            list(cognitive_state_summary.tool_outcomes)
            if cognitive_state_summary is not None
            else []
        )
        working_digest = self._context_governor.build_initial_digest(
            cognitive_state=cognitive_state,
            cognitive_state_summary=cognitive_state_summary,
            blocking_open_question_count=len(blocking_open_questions),
            last_proposal_status=last_proposal_status,
        )
        existing_compact_summary = (
            cognitive_state_summary.compact_summary
            if cognitive_state_summary is not None
            else None
        )
        context_governor = self._context_governor
        if context_packet.context_profile == "compact":
            context_governor = self._build_compact_prompt_governor(
                model_name=model_name,
                model_id=request.model_id,
                provider=provider,
            )
        (
            governed_history,
            compact_summary,
            governance_metadata,
        ) = await context_governor.govern_history_async(
            history=list(request.history),
            retained_tool_outcomes=retained_tool_outcomes,
            working_digest=working_digest,
            existing_summary=existing_compact_summary,
            context_profile=context_packet.context_profile,
            current_step=current_step.value,
            current_stage=current_stage.value if current_stage is not None else None,
            estimated_input_tokens=estimated_input_tokens,
            input_token_count_source=input_token_count_source,
            previous_usage=previous_usage,
        )
        context_report = self._build_context_report(
            request=request,
            context_packet=context_packet,
            governance_metadata=governance_metadata,
            retained_tool_outcomes=retained_tool_outcomes,
            compact_summary=compact_summary,
            existing_summary=existing_compact_summary,
        )
        turn_input = adapter.build_turn_input(
            request=request,
            workspace=workspace,
            context_packet=context_packet,
            model_name=model_name,
            provider=provider,
            governed_history=governed_history,
            working_digest=working_digest,
            tool_outcomes=retained_tool_outcomes,
            compact_summary=compact_summary,
            governance_metadata=governance_metadata,
            context_report=context_report,
            cognitive_state=cognitive_state,
            cognitive_state_summary=cognitive_state_summary,
        )
        return turn_input, context_packet

    def _build_compact_prompt_governor(
        self,
        *,
        model_name: str,
        model_id: str,
        provider,
    ) -> SetupContextGovernorService:
        compaction_service = SetupContextCompactionService(
            compact_prompt_provider=lambda messages: self._run_compact_prompt_pass(
                messages=messages,
                model_name=model_name,
                model_id=model_id,
                provider=provider,
            )
        )
        return SetupContextGovernorService(compaction_service=compaction_service)

    async def _run_compact_prompt_pass(
        self,
        *,
        messages,
        model_name: str,
        model_id: str,
        provider,
    ) -> dict[str, Any]:
        request = ChatCompletionRequest(
            model=model_name,
            model_id=model_id,
            messages=messages,
            stream=False,
            provider=provider,
            max_tokens=self._COMPACT_PROMPT_MAX_TOKENS,
        )
        response = await self._llm_service.chat_completion(request)
        content = self._extract_compact_prompt_content(response)
        return json.loads(content)

    @staticmethod
    def _extract_compact_prompt_content(response: dict[str, Any]) -> str:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("compact_prompt_missing_choices")
        first = choices[0]
        if not isinstance(first, dict):
            raise ValueError("compact_prompt_invalid_choice")
        message = first.get("message")
        if not isinstance(message, dict):
            raise ValueError("compact_prompt_missing_message")
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        raise ValueError("compact_prompt_missing_content")

    def _persist_runtime_v2_governance(
        self,
        *,
        workspace,
        context_packet,
        step_id,
        result: RpAgentTurnResult,
    ) -> None:
        if self._runtime_state_service is None:
            return
        payload = (
            result.structured_payload
            if isinstance(result.structured_payload, dict)
            else {}
        )
        working_digest = (
            SetupWorkingDigest.model_validate(payload["working_digest"])
            if isinstance(payload.get("working_digest"), dict)
            else None
        )
        tool_outcomes = [
            SetupToolOutcome.model_validate(item)
            for item in payload.get("tool_outcomes", [])
            if isinstance(item, dict)
        ]
        compact_summary = (
            SetupContextCompactSummary.model_validate(payload["compact_summary"])
            if isinstance(payload.get("compact_summary"), dict)
            else None
        )
        self._runtime_state_service.persist_turn_governance(
            workspace=workspace,
            context_packet=context_packet,
            step_id=step_id,
            working_digest=working_digest,
            tool_outcomes=tool_outcomes,
            compact_summary=compact_summary,
        )
