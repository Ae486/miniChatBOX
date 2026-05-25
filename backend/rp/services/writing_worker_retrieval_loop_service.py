"""Thin bounded retrieval loop for WritingWorker E2."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from models.chat import ChatMessage
from rp.models.memory_contract_registry import MemoryRuntimeIdentity
from rp.models.retrieval_runtime_contracts import (
    WriterRetrievalExpandToolInput,
    WriterRetrievalSearchToolInput,
    WriterRetrievalUsageToolInput,
)
from rp.models.worker_memory import WorkerSourceRefBundle
from rp.services.runtime_retrieval_card_service import (
    RuntimeRetrievalCardService,
    RuntimeRetrievalCardServiceError,
)
from rp.services.runtime_retrieval_search_service import RuntimeRetrievalSearchService
from rp.services.story_llm_gateway import StoryLlmGateway
from rp.services.writer_retrieval_usage_guard_service import (
    WriterRetrievalUsageGuardService,
    WriterRetrievalUsageGuardServiceError,
)


WRITER_RETRIEVAL_SEARCH_TOOL = "retrieval.search"
WRITER_RETRIEVAL_EXPAND_TOOL = "retrieval.expand"
WRITER_RETRIEVAL_USAGE_TOOL = "retrieval.usage"


class WritingWorkerRetrievalLoopServiceError(ValueError):
    """Stable writer-retrieval loop error with a machine-readable code."""

    def __init__(self, code: str, detail: str):
        self.code = code
        super().__init__(f"{code}:{detail}")


@dataclass
class WriterRetrievalLoopResult:
    """Structured result of the bounded writer retrieval loop."""

    output_text: str
    usage_metadata: dict[str, Any] = field(default_factory=dict)
    tool_trace_refs: list[str] = field(default_factory=list)
    source_ref_bundle: WorkerSourceRefBundle = field(
        default_factory=WorkerSourceRefBundle
    )


@dataclass
class _LoopState:
    retrieval_generation: int = 0
    usage_generation: int = 0
    usage_material_id: str | None = None
    tool_trace_refs: list[str] = field(default_factory=list)


class WritingWorkerRetrievalLoopService:
    """Execute a bounded client-side tool loop for writer retrieval only."""

    def __init__(
        self,
        *,
        llm_gateway: StoryLlmGateway,
        runtime_retrieval_card_service: RuntimeRetrievalCardService,
        runtime_retrieval_search_service: RuntimeRetrievalSearchService | None = None,
        usage_guard_service: WriterRetrievalUsageGuardService | None = None,
    ) -> None:
        self._llm_gateway = llm_gateway
        self._runtime_retrieval_card_service = runtime_retrieval_card_service
        self._runtime_retrieval_search_service = (
            runtime_retrieval_search_service
            or RuntimeRetrievalSearchService(
                runtime_retrieval_card_service=runtime_retrieval_card_service
            )
        )
        self._usage_guard_service = (
            usage_guard_service
            or WriterRetrievalUsageGuardService(
                runtime_retrieval_card_service=runtime_retrieval_card_service
            )
        )

    async def run(
        self,
        *,
        identity: MemoryRuntimeIdentity,
        model_id: str,
        provider_id: str | None,
        messages: list[ChatMessage],
        max_retrieval_attempts: int,
        temperature: float | None,
        max_tokens: int | None,
        include_reasoning: bool | None = None,
    ) -> WriterRetrievalLoopResult:
        if max_retrieval_attempts <= 0:
            raise WritingWorkerRetrievalLoopServiceError(
                "writer_retrieval_attempt_limit_invalid",
                str(max_retrieval_attempts),
            )
        conversation = [message.model_copy(deep=True) for message in messages]
        attempts = 0
        rounds = 0
        max_rounds = max(4, (max_retrieval_attempts * 3) + 1)
        totals = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        state = _LoopState()

        while rounds < max_rounds:
            rounds += 1
            response = await self._llm_gateway.complete_with_tools(
                model_id=model_id,
                provider_id=provider_id,
                messages=conversation,
                tools=self._tool_definitions(),
                tool_choice="auto",
                temperature=temperature,
                max_tokens=max_tokens,
                include_reasoning=include_reasoning,
                extra_body={"parallel_tool_calls": False},
            )
            self._accumulate_usage(totals, response.get("usage"))
            choice = self._get_first_choice(response)
            if choice is None:
                raise WritingWorkerRetrievalLoopServiceError(
                    "writer_retrieval_model_response_missing_choice",
                    identity.turn_id,
                )
            message_payload = choice.get("message") or {}
            tool_calls = self._tool_calls_from_message(message_payload)
            if not tool_calls:
                output_text = self._message_text(message_payload)
                try:
                    self._usage_guard_service.ensure_final_output_allowed(
                        identity=identity,
                        retrieval_generation=state.retrieval_generation,
                        usage_generation=state.usage_generation,
                        usage_material_id=state.usage_material_id,
                    )
                except WriterRetrievalUsageGuardServiceError as exc:
                    raise WritingWorkerRetrievalLoopServiceError(
                        exc.code,
                        str(exc),
                    ) from exc
                return WriterRetrievalLoopResult(
                    output_text=output_text,
                    usage_metadata=totals,
                    tool_trace_refs=list(state.tool_trace_refs),
                    source_ref_bundle=self._runtime_retrieval_card_service.build_source_ref_bundle(
                        identity=identity
                    ),
                )

            assistant_message = self._build_assistant_tool_message(
                message_payload=message_payload,
                tool_calls=tool_calls,
            )
            tool_result_messages: list[ChatMessage] = []
            for tool_call in tool_calls:
                tool_name = self._tool_name(tool_call)
                if tool_name in {
                    WRITER_RETRIEVAL_SEARCH_TOOL,
                    WRITER_RETRIEVAL_EXPAND_TOOL,
                }:
                    if attempts >= max_retrieval_attempts:
                        raise WritingWorkerRetrievalLoopServiceError(
                            "writer_retrieval_attempt_limit_exceeded",
                            str(max_retrieval_attempts),
                        )
                    attempts += 1
                tool_content = await self._execute_tool_call(
                    identity=identity,
                    tool_call=tool_call,
                    state=state,
                    attempt_index=attempts,
                )
                tool_result_messages.append(
                    ChatMessage(
                        role="tool",
                        name=tool_name,
                        tool_call_id=self._tool_call_id(tool_call),
                        content=tool_content,
                    )
                )
            conversation.append(assistant_message)
            conversation.extend(tool_result_messages)
        raise WritingWorkerRetrievalLoopServiceError(
            "writer_retrieval_round_limit_exceeded",
            str(max_rounds),
        )

    async def _execute_tool_call(
        self,
        *,
        identity: MemoryRuntimeIdentity,
        tool_call: dict[str, Any],
        state: _LoopState,
        attempt_index: int,
    ) -> str:
        tool_name = self._tool_name(tool_call)
        arguments = self._tool_arguments(tool_call)
        state.tool_trace_refs.append(
            f"writer_tool:{tool_name}:{self._tool_call_id(tool_call)}"
        )
        try:
            if tool_name == WRITER_RETRIEVAL_SEARCH_TOOL:
                return await self._handle_search(
                    identity=identity,
                    arguments=arguments,
                    state=state,
                    attempt_index=attempt_index,
                )
            if tool_name == WRITER_RETRIEVAL_EXPAND_TOOL:
                return self._handle_expand(
                    identity=identity,
                    arguments=arguments,
                    state=state,
                )
            if tool_name == WRITER_RETRIEVAL_USAGE_TOOL:
                return self._handle_usage(
                    identity=identity,
                    arguments=arguments,
                    state=state,
                )
        except (
            RuntimeRetrievalCardServiceError,
            WriterRetrievalUsageGuardServiceError,
        ) as exc:
            raise WritingWorkerRetrievalLoopServiceError(exc.code, str(exc)) from exc
        raise WritingWorkerRetrievalLoopServiceError(
            "writer_retrieval_tool_not_allowed",
            tool_name,
        )

    async def _handle_search(
        self,
        *,
        identity: MemoryRuntimeIdentity,
        arguments: dict[str, Any],
        state: _LoopState,
        attempt_index: int,
    ) -> str:
        payload = WriterRetrievalSearchToolInput.model_validate(arguments)
        execution = await self._runtime_retrieval_search_service.search_for_writer(
            identity=identity,
            input_model=payload,
            actor="writer.retrieval",
            attempt_index=attempt_index,
        )
        if execution.materials or execution.miss_materials:
            state.retrieval_generation += 1
            for material_id in [
                *(material.material_id for material in execution.materials),
                *(material.material_id for material in execution.miss_materials),
            ]:
                state.tool_trace_refs.append(f"runtime_workspace:{material_id}")
        state.usage_material_id = None
        return execution.output.model_dump_json(
            exclude_none=True,
        )

    def _handle_expand(
        self,
        *,
        identity: MemoryRuntimeIdentity,
        arguments: dict[str, Any],
        state: _LoopState,
    ) -> str:
        payload = WriterRetrievalExpandToolInput.model_validate(arguments)
        expanded = self._runtime_retrieval_card_service.expand_cards_by_refs(
            identity=identity,
            card_refs=payload.card_short_ids,
            actor="writer.retrieval",
        )
        if expanded:
            state.retrieval_generation += 1
            state.usage_material_id = None
            for material in expanded:
                state.tool_trace_refs.append(
                    f"runtime_workspace:{material.material_id}"
                )
        return json.dumps(
            {
                "expanded": [self._expanded_tool_payload(item) for item in expanded],
            },
            ensure_ascii=False,
        )

    def _handle_usage(
        self,
        *,
        identity: MemoryRuntimeIdentity,
        arguments: dict[str, Any],
        state: _LoopState,
    ) -> str:
        payload = WriterRetrievalUsageToolInput.model_validate(arguments)
        usage = self._runtime_retrieval_card_service.record_writer_usage(
            identity=identity,
            used_card_ids=payload.used_card_short_ids,
            used_expanded_chunk_ids=payload.used_expanded_short_ids,
            missed_query_ids=payload.missed_query_short_ids,
            actor="writer.retrieval",
            knowledge_gaps=payload.knowledge_gaps,
        )
        state.usage_generation = state.retrieval_generation
        state.usage_material_id = usage.material_id
        state.tool_trace_refs.append(f"runtime_workspace:{usage.material_id}")
        return json.dumps(
            {
                "usage_short_id": usage.short_id,
                "used_card_short_ids": usage.payload.get("used_card_short_ids", []),
                "expanded_card_short_ids": usage.payload.get(
                    "expanded_card_short_ids", []
                ),
                "missed_query_short_ids": usage.payload.get(
                    "missed_query_short_ids", []
                ),
                "knowledge_gaps": usage.payload.get("knowledge_gaps", []),
            },
            ensure_ascii=False,
        )

    @staticmethod
    def _tool_definitions() -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": WRITER_RETRIEVAL_SEARCH_TOOL,
                    "description": (
                        "Search runtime memory with a concise query and return clean RAG "
                        "results. Use entity for one concrete subject; when comparing two "
                        "entities, prefer separate entity searches for each entity first. "
                        "Use relation for relationship questions with explicit anchors, "
                        "Use semantic for abstract background, rules, or broad context."
                    ),
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "query": {"type": "string"},
                            "mode": {
                                "type": "string",
                                "enum": [
                                    "entity",
                                    "relation",
                                    "semantic",
                                ],
                            },
                            "lexical_anchors": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "semantic_predicates": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": WRITER_RETRIEVAL_EXPAND_TOOL,
                    "description": "Expand already returned retrieval cards by short id.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "card_short_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            }
                        },
                        "required": ["card_short_ids"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": WRITER_RETRIEVAL_USAGE_TOOL,
                    "description": (
                        "Record which retrieval cards and expansions were used before final output."
                    ),
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "used_card_short_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "used_expanded_short_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "missed_query_short_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "knowledge_gaps": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "gap_id": {"type": ["string", "null"]},
                                        "query": {"type": "string"},
                                        "status": {"type": "string"},
                                        "impact": {"type": ["string", "null"]},
                                        "mode_policy_resolution": {
                                            "type": ["string", "null"]
                                        },
                                    },
                                    "required": ["query", "status"],
                                },
                            },
                        },
                        "required": [],
                    },
                },
            },
        ]

    @staticmethod
    def _get_first_choice(response: dict[str, Any]) -> dict[str, Any] | None:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        choice = choices[0]
        return choice if isinstance(choice, dict) else None

    @staticmethod
    def _tool_calls_from_message(
        message_payload: dict[str, Any],
    ) -> list[dict[str, Any]]:
        raw = message_payload.get("tool_calls") or []
        return [call for call in raw if isinstance(call, dict)]

    @staticmethod
    def _message_text(message_payload: dict[str, Any]) -> str:
        content = message_payload.get("content")
        if isinstance(content, list):
            return "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        return str(content or "")

    @classmethod
    def _build_assistant_tool_message(
        cls,
        *,
        message_payload: dict[str, Any],
        tool_calls: list[dict[str, Any]],
    ) -> ChatMessage:
        content = message_payload.get("content")
        if content is None and tool_calls:
            content = ""
        return ChatMessage(
            role="assistant",
            content=content,
            tool_calls=[cls._normalize_tool_call(call) for call in tool_calls],
        )

    @staticmethod
    def _normalize_tool_call(call: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(call)
        normalized["id"] = WritingWorkerRetrievalLoopService._tool_call_id(call)
        normalized["type"] = str(normalized.get("type") or "function")
        function = normalized.get("function")
        normalized["function"] = function if isinstance(function, dict) else {}
        return normalized

    @staticmethod
    def _tool_name(call: dict[str, Any]) -> str:
        function = call.get("function")
        if not isinstance(function, dict):
            return ""
        return str(function.get("name") or "").strip()

    @staticmethod
    def _tool_call_id(call: dict[str, Any]) -> str:
        return str(call.get("id") or call.get("tool_call_id") or "").strip()

    @staticmethod
    def _tool_arguments(call: dict[str, Any]) -> dict[str, Any]:
        function = call.get("function")
        if not isinstance(function, dict):
            return {}
        raw = function.get("arguments")
        if isinstance(raw, dict):
            return raw
        text = str(raw or "").strip()
        if not text:
            return {}
        return json.loads(text)

    @staticmethod
    def _accumulate_usage(totals: dict[str, int], usage: Any) -> None:
        if not isinstance(usage, dict):
            return
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        total_tokens = int(
            usage.get("total_tokens") or (prompt_tokens + completion_tokens)
        )
        totals["prompt_tokens"] += prompt_tokens
        totals["completion_tokens"] += completion_tokens
        totals["total_tokens"] += total_tokens

    @staticmethod
    def _expanded_tool_payload(material) -> dict[str, Any]:
        return {
            "short_id": material.short_id,
            "card_short_id": material.payload.get("card_short_id"),
            "title": material.payload.get("title"),
            "summary": material.payload.get("summary"),
            "text": material.payload.get("text"),
        }
