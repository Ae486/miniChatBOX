"""Factory wiring tests for the setup runtime-v2 execution path."""

from __future__ import annotations

import json
from typing import Any

import pytest

from models.chat import ProviderConfig
from rp.agent_runtime.contracts import (
    RpAgentTurnResult,
    RuntimeToolResult,
    SetupWorkingDigest,
)
from rp.agent_runtime.profiles import (
    build_setup_agent_capability_plan,
    build_setup_agent_tool_scope,
)
from rp.agent_runtime.executor import RpAgentRuntimeExecutor
from rp.agent_runtime.tools import RuntimeToolExecutor
from rp.models.setup_drafts import (
    SetupDraftEntry,
    SetupDraftSection,
    SetupStageDraftBlock,
)
from rp.models.setup_agent import SetupAgentDialogueMessage, SetupAgentTurnRequest
from rp.models.setup_handoff import SetupContextBuilderInput
from rp.models.setup_stage import SetupStageId
from rp.models.setup_workspace import SetupStepId, StoryMode
from rp.runtime.rp_runtime_factory import RpRuntimeFactory
from rp.agent_runtime.adapters import SetupRuntimeAdapter
from rp.services.setup_agent_execution_service import SetupAgentExecutionService
from rp.services.setup_agent_runtime_state_service import SetupAgentRuntimeStateService
from rp.services.setup_context_builder import SetupContextBuilder
from rp.services.setup_workspace_service import SetupWorkspaceService
from rp.tools.setup_tool_provider import SetupToolProvider


_RECOVERED_MAGIC_LAW_DETAIL = "Public spellcasting requires guild permits."
_RECOVERED_MEMORY_DETAIL = "Moonlit forest cities."
EXTERNAL_MEMORY_TOOLS = {
    "memory.get_state",
    "memory.get_summary",
    "memory.search_recall",
    "memory.search_archival",
    "memory.list_versions",
    "memory.read_provenance",
}


def _messages_text(messages: list[Any]) -> str:
    return "\n".join(str(message.content or "") for message in messages)


class _SetupProviderBackedToolExecutor(RuntimeToolExecutor):
    """Small test adapter that routes runtime setup calls into SetupToolProvider."""

    def __init__(self, *, provider: SetupToolProvider) -> None:
        self._provider = provider
        self.calls: list[tuple[Any, list[str]]] = []

    def get_openai_tool_definitions(
        self, *, visible_tool_names: list[str]
    ) -> list[dict[str, Any]]:
        allowed = set(visible_tool_names)
        return [
            tool.to_openai_tool()
            for tool in self._provider.list_tools()
            if (
                tool.name in allowed
                or any(alias in allowed for alias in tool.qualified_name_aliases)
            )
        ]

    async def execute_tool_call(
        self,
        call,
        *,
        visible_tool_names: list[str],
    ) -> RuntimeToolResult:
        self.calls.append((call, list(visible_tool_names)))
        tool_info = next(
            (
                tool
                for tool in self._provider.list_tools()
                if str(call.tool_name) in (*tool.qualified_name_aliases, tool.name)
            ),
            None,
        )
        raw_name = tool_info.name if tool_info is not None else str(call.tool_name)
        if raw_name not in visible_tool_names:
            return RuntimeToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                success=False,
                content_text=json.dumps(
                    {"error": {"code": "unknown_tool", "message": raw_name}},
                    sort_keys=True,
                ),
                error_code="UNKNOWN_TOOL",
            )
        result = await self._provider.call_tool(
            tool_name=raw_name,
            arguments=dict(call.arguments),
        )
        content_text = str(result.get("content") or "")
        content_payload = json.loads(content_text) if content_text else None
        structured_payload = {
            "server_id": self._provider.provider_id,
            "tool_name": raw_name,
            "qualified_name": tool_info.qualified_name if tool_info else call.tool_name,
            "raw_qualified_name": tool_info.raw_qualified_name
            if tool_info
            else raw_name,
        }
        if content_payload is not None:
            structured_payload["content_payload"] = content_payload
        return RuntimeToolResult(
            call_id=call.call_id,
            tool_name=tool_info.qualified_name if tool_info else call.tool_name,
            success=bool(result.get("success")),
            content_text=content_text,
            error_code=(
                str(result.get("error_code")) if result.get("error_code") else None
            ),
            structured_payload=structured_payload,
        )


class _MemoryOpenRecoveryLLM:
    def __init__(self, *, workspace_id: str) -> None:
        self.workspace_id = workspace_id
        self.requests: list[Any] = []
        self.round = 0
        self.recovered_from_tool_result = False
        self.recovered_detail: str | None = None

    async def chat_completion(self, request):
        self.round += 1
        self.requests.append(request)
        if self.round == 1:
            visible_text = _messages_text(request.messages)
            assert _RECOVERED_MAGIC_LAW_DETAIL not in visible_text
            assert "OLD_RAW_HISTORY_OUTSIDE_SUMMARY_WINDOW" not in visible_text
            assert "stage:world_background:magic_law:summary" in visible_text
            assert "recovery_hints" in visible_text
            assert any(
                "setup_memory_open" in item["function"]["name"]
                for item in (request.tools or [])
            )
            open_schema = next(
                item
                for item in (request.tools or [])
                if item["function"]["name"] == "rp_setup__setup_memory_open"
            )
            assert "ref" in open_schema["function"]["parameters"]["required"]
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_open_magic_law",
                                    "type": "function",
                                    "function": {
                                        "name": "rp_setup__setup_memory_open",
                                        "arguments": json.dumps(
                                            {
                                                "workspace_id": self.workspace_id,
                                                "ref": "stage:world_background:magic_law:summary",
                                                "max_chars": 1200,
                                            },
                                            sort_keys=True,
                                        ),
                                    },
                                }
                            ],
                        }
                    }
                ]
            }

        self.recovered_detail = self._recovered_detail_from_tool_messages(
            request.messages
        )
        self.recovered_from_tool_result = (
            self.recovered_detail == _RECOVERED_MAGIC_LAW_DETAIL
        )
        assert self.recovered_from_tool_result
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": (
                            "Recovered stage:world_background:magic_law:summary from memory open: "
                            f"{self.recovered_detail}"
                        ),
                    }
                }
            ]
        }

    @staticmethod
    def _recovered_detail_from_tool_messages(messages: list[Any]) -> str | None:
        for message in messages:
            if message.role != "tool":
                continue
            payload = json.loads(str(message.content or "{}"))
            if payload.get("opened_ref") != "stage:world_background:magic_law:summary":
                continue
            content = payload.get("content") or {}
            if content.get("type") == "text" and content.get("text"):
                return str(content["text"])
        return None


class _SetupMemorySearchThenOpenLLM:
    def __init__(self, *, workspace_id: str) -> None:
        self.workspace_id = workspace_id
        self.requests: list[Any] = []
        self.round = 0
        self.search_hit_ref: str | None = None
        self.search_carried_payload = False
        self.recovered_detail: str | None = None

    async def chat_completion(self, request):
        self.round += 1
        self.requests.append(request)
        if self.round == 1:
            visible_text = _messages_text(request.messages)
            assert _RECOVERED_MEMORY_DETAIL not in visible_text
            assert any(
                item["function"]["name"] == "rp_setup__setup_memory_search"
                for item in (request.tools or [])
            )
            assert any(
                item["function"]["name"] == "rp_setup__setup_memory_open"
                for item in (request.tools or [])
            )
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_memory_search",
                                    "type": "function",
                                    "function": {
                                        "name": "rp_setup__setup_memory_search",
                                        "arguments": json.dumps(
                                            {
                                                "workspace_id": self.workspace_id,
                                                "query": "Moonlit forest",
                                                "limit": 5,
                                            },
                                            sort_keys=True,
                                        ),
                                    },
                                }
                            ],
                        }
                    }
                ]
            }

        if self.round == 2:
            self.search_hit_ref = self._search_ref_from_tool_messages(request.messages)
            assert self.search_hit_ref == "stage:world_background:race_elf:summary"
            assert self.search_carried_payload is False
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_memory_read",
                                    "type": "function",
                                    "function": {
                                        "name": "rp_setup__setup_memory_open",
                                        "arguments": json.dumps(
                                            {
                                                "workspace_id": self.workspace_id,
                                                "ref": self.search_hit_ref,
                                                "max_chars": 1200,
                                            },
                                            sort_keys=True,
                                        ),
                                    },
                                }
                            ],
                        }
                    }
                ]
            }

        self.recovered_detail = self._recovered_detail_from_tool_messages(
            request.messages
        )
        assert self.recovered_detail == _RECOVERED_MEMORY_DETAIL
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": (
                            f"Recovered via setup memory: {self.recovered_detail}"
                        ),
                    }
                }
            ]
        }

    def _search_ref_from_tool_messages(self, messages: list[Any]) -> str | None:
        for message in messages:
            if message.role != "tool":
                continue
            payload = json.loads(str(message.content or "{}"))
            items = payload.get("items") or []
            if not items:
                continue
            encoded = json.dumps(payload, sort_keys=True)
            self.search_carried_payload = '"payload"' in encoded
            for item in items:
                if item.get("ref") == "stage:world_background:race_elf:summary":
                    assert item.get("scope") == "section"
                    assert item.get("navigation_summary")
                    return str(item["ref"])
        return None

    @staticmethod
    def _recovered_detail_from_tool_messages(messages: list[Any]) -> str | None:
        for message in messages:
            if message.role != "tool":
                continue
            payload = json.loads(str(message.content or "{}"))
            if payload.get("opened_ref") != "stage:world_background:race_elf:summary":
                continue
            content = payload.get("content") or {}
            if content.get("type") == "text" and content.get("text"):
                return str(content["text"])
        return None


class _CompactPromptLLM:
    def __init__(self, *, invalid_json: bool = False) -> None:
        self.invalid_json = invalid_json
        self.requests: list[Any] = []

    async def chat_completion(self, request):
        self.requests.append(request)
        visible_text = _messages_text(request.messages)
        assert "SetupStageCompactPrompt" in visible_text
        assert request.tools is None
        prompt_payload = json.loads(str(request.messages[1].content or "{}"))
        draft_refs = list(prompt_payload.get("draft_refs") or [])
        if self.invalid_json:
            content = "not json"
        else:
            content = json.dumps(
                {
                    "summary_lines": ["Prompt-pass compacted older setup discussion."],
                    "confirmed_points": ["Keep compact as context engineering."],
                    "open_threads": ["Need next setup focus."],
                    "rejected_directions": ["Do not add a separate compact agent."],
                    "draft_refs": draft_refs,
                    "recovery_hints": [
                        {
                            "ref": draft_refs[0],
                            "reason": "Recover exact draft detail if needed.",
                            "detail": "Use setup.memory.open.",
                        }
                    ]
                    if draft_refs
                    else [],
                    "must_not_infer": ["Do not infer old raw discussion details."],
                },
                sort_keys=True,
            )
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": content,
                    }
                }
            ]
        }


class _CountingCompactPromptLLM(_CompactPromptLLM):
    def __init__(self, *, prompt_tokens: int, source: str) -> None:
        super().__init__()
        self.prompt_tokens = prompt_tokens
        self.source = source
        self.token_count_requests: list[Any] = []

    def count_request_tokens(self, request):
        self.token_count_requests.append(request)
        return {
            "prompt_tokens": self.prompt_tokens,
            "source": self.source,
        }


def test_setup_runtime_factory_always_uses_runtime_v2(retrieval_session):
    service = RpRuntimeFactory(retrieval_session).build_setup_agent_execution_service()
    runner = RpRuntimeFactory(retrieval_session).build_setup_graph_runner()

    assert service._runtime_executor is not None
    assert service._adapter is not None
    assert runner._execution_service._runtime_executor is not None
    assert runner._execution_service._adapter is not None


def test_setup_runtime_factory_exposes_setup_memory_without_external_memory_os(
    retrieval_session,
):
    manager = RpRuntimeFactory(retrieval_session)._build_setup_mcp_manager(
        story_id="story-setup-memory-manager-scope"
    )

    tool_names = {tool.name for tool in manager.get_all_tools()}

    assert "setup.memory.search" in tool_names
    assert "setup.memory.open" in tool_names
    assert "setup.memory.read_refs" in tool_names
    assert {
        "memory.get_state",
        "memory.get_summary",
        "memory.search_recall",
        "memory.search_archival",
        "memory.list_versions",
        "memory.read_provenance",
    }.isdisjoint(tool_names)


def test_setup_agent_execution_service_uses_standard_context_budget_for_small_turn():
    request = SetupAgentTurnRequest(
        workspace_id="workspace-1",
        model_id="model-1",
        user_prompt="Help me continue setup.",
        history=[
            SetupAgentDialogueMessage(role="user", content="short"),
            SetupAgentDialogueMessage(role="assistant", content="short reply"),
        ],
        user_edit_delta_ids=[],
    )

    assert (
        SetupAgentExecutionService._context_token_budget(request)
        == SetupAgentExecutionService._STANDARD_CONTEXT_TOKEN_BUDGET
    )


def test_setup_agent_execution_service_switches_to_compact_budget_for_large_turn():
    request = SetupAgentTurnRequest(
        workspace_id="workspace-1",
        model_id="model-1",
        user_prompt="Continue with a compact context.",
        history=[
            SetupAgentDialogueMessage(role="user", content="x" * 2500),
            SetupAgentDialogueMessage(role="assistant", content="y" * 2000),
        ],
        user_edit_delta_ids=["delta-1", "delta-2", "delta-3"],
    )

    assert (
        SetupAgentExecutionService._context_token_budget(request)
        == SetupAgentExecutionService._COMPACT_CONTEXT_TOKEN_BUDGET
    )


def test_setup_agent_execution_service_reports_estimated_token_pressure():
    request = SetupAgentTurnRequest(
        workspace_id="workspace-1",
        model_id="model-1",
        user_prompt="x" * 7200,
        history=[],
        user_edit_delta_ids=[],
    )

    estimated_tokens = SetupAgentExecutionService._estimate_input_tokens(request)
    reasons = SetupAgentExecutionService._context_profile_reasons(
        request,
        estimated_input_tokens=estimated_tokens,
    )

    assert (
        SetupAgentExecutionService._context_token_budget(
            request,
            estimated_input_tokens=estimated_tokens,
        )
        == SetupAgentExecutionService._COMPACT_CONTEXT_TOKEN_BUDGET
    )
    assert "estimated_input_tokens_threshold" in reasons
    assert "history_chars_threshold" not in reasons


def test_setup_agent_execution_service_reports_observed_usage_pressure():
    request = SetupAgentTurnRequest(
        workspace_id="workspace-1",
        model_id="model-1",
        user_prompt="short",
        history=[],
        user_edit_delta_ids=[],
    )
    previous_usage: dict[str, int | None] = {
        "prompt_tokens": 1900,
        "completion_tokens": 10,
        "total_tokens": 1910,
    }

    reasons = SetupAgentExecutionService._context_profile_reasons(
        request,
        previous_usage=previous_usage,
    )

    assert (
        SetupAgentExecutionService._context_token_budget(
            request,
            previous_usage=previous_usage,
        )
        == SetupAgentExecutionService._COMPACT_CONTEXT_TOKEN_BUDGET
    )
    assert "observed_usage_threshold" in reasons


def test_setup_agent_execution_service_normalizes_openai_style_provider_usage():
    result = RpAgentTurnResult(
        status="completed",
        finish_reason="completed_text",
        assistant_text="Done.",
        structured_payload={
            "latest_response": {
                "usage": {
                    "prompt_tokens": 101,
                    "completion_tokens": 23,
                    "total_tokens": 124,
                    "prompt_tokens_details": {"cached_tokens": 17},
                    "completion_tokens_details": {"reasoning_tokens": 5},
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 17,
                    "provider_specific": {"kept": True},
                }
            }
        },
    )

    usage = SetupAgentExecutionService._usage_from_runtime_result(result)

    assert usage is not None
    assert usage["prompt_tokens"] == 101
    assert usage["input_tokens"] == 101
    assert usage["completion_tokens"] == 23
    assert usage["output_tokens"] == 23
    assert usage["total_tokens"] == 124
    assert usage["cached_tokens"] == 17
    assert usage["reasoning_tokens"] == 5
    assert usage["cache_creation_input_tokens"] == 0
    assert usage["cache_read_input_tokens"] == 17
    assert usage["source"] == "provider_usage_metadata"
    assert usage["raw_usage"]["provider_specific"] == {"kept": True}


def test_setup_agent_execution_service_normalizes_langchain_and_gemini_usage_shapes():
    langchain_usage = SetupAgentExecutionService._normalize_provider_usage(
        {
            "input_tokens": 44,
            "output_tokens": 12,
            "total_tokens": 56,
            "input_token_details": {"cache_read": 7, "cache_creation": 2},
            "output_token_details": {"reasoning": 4},
        }
    )
    gemini_usage = SetupAgentExecutionService._normalize_provider_usage(
        {
            "promptTokenCount": 30,
            "candidatesTokenCount": 9,
            "totalTokenCount": 39,
            "thoughtsTokenCount": 6,
        }
    )

    assert langchain_usage is not None
    assert langchain_usage["prompt_tokens"] == 44
    assert langchain_usage["completion_tokens"] == 12
    assert langchain_usage["cache_read_input_tokens"] == 7
    assert langchain_usage["cache_creation_input_tokens"] == 2
    assert langchain_usage["reasoning_tokens"] == 4
    assert gemini_usage is not None
    assert gemini_usage["prompt_tokens"] == 30
    assert gemini_usage["completion_tokens"] == 9
    assert gemini_usage["total_tokens"] == 39
    assert gemini_usage["reasoning_tokens"] == 6


@pytest.mark.asyncio
async def test_setup_agent_execution_service_v2_uses_litellm_token_counter_for_pressure(
    retrieval_session,
):
    workspace_service = SetupWorkspaceService(retrieval_session)
    context_builder = SetupContextBuilder(workspace_service)
    runtime_state_service = SetupAgentRuntimeStateService(retrieval_session)
    adapter = SetupRuntimeAdapter()
    llm = _CountingCompactPromptLLM(
        prompt_tokens=1900,
        source="litellm_token_counter",
    )
    service = SetupAgentExecutionService(
        workspace_service=workspace_service,
        context_builder=context_builder,
        adapter=adapter,
        runtime_executor=None,
        runtime_state_service=runtime_state_service,
        llm_service=llm,
    )
    workspace = workspace_service.create_workspace(
        story_id="story-token-counter-pressure",
        mode=StoryMode.LONGFORM,
    )
    request = SetupAgentTurnRequest(
        workspace_id=workspace.workspace_id,
        model_id="model-1",
        user_prompt="Continue setup.",
        history=[],
        user_edit_delta_ids=[],
    )

    turn_input, context_packet = await service._build_runtime_v2_turn_input(
        adapter=adapter,
        request=request,
        workspace=workspace,
        model_name="gpt-4o-mini",
        provider=ProviderConfig(
            type="openai",
            api_key="sk-test",
            api_url="https://example.com/v1",
            custom_headers={},
        ),
    )

    report = turn_input.metadata["context_report"]
    assert context_packet.context_profile == "compact"
    assert len(llm.token_count_requests) == 1
    assert len(llm.requests) == 0
    assert report["estimated_input_tokens"] == 1900
    assert report["input_token_count_source"] == "litellm_token_counter"
    assert "estimated_input_tokens_threshold" in report["profile_reasons"]


@pytest.mark.asyncio
async def test_setup_agent_execution_service_v2_builds_governed_history_for_compact_turn(
    retrieval_session,
):
    workspace_service = SetupWorkspaceService(retrieval_session)
    context_builder = SetupContextBuilder(workspace_service)
    runtime_state_service = SetupAgentRuntimeStateService(retrieval_session)
    adapter = SetupRuntimeAdapter()
    llm = _CompactPromptLLM()
    service = SetupAgentExecutionService(
        workspace_service=workspace_service,
        context_builder=context_builder,
        adapter=adapter,
        runtime_executor=None,
        runtime_state_service=runtime_state_service,
        llm_service=llm,
    )
    workspace = workspace_service.create_workspace(
        story_id="story-governed-history-1",
        mode=StoryMode.LONGFORM,
    )
    request = SetupAgentTurnRequest(
        workspace_id=workspace.workspace_id,
        model_id="model-1",
        user_prompt="Continue setup with a compact history.",
        history=[
            SetupAgentDialogueMessage(
                role="user" if index % 2 == 0 else "assistant",
                content=f"history message {index}",
            )
            for index in range(10)
        ],
        user_edit_delta_ids=["delta-1", "delta-2", "delta-3"],
    )

    turn_input, context_packet = await service._build_runtime_v2_turn_input(
        adapter=adapter,
        request=request,
        workspace=workspace,
        model_name="gpt-4o-mini",
        provider=ProviderConfig(
            type="openai",
            api_key="sk-test",
            api_url="https://example.com/v1",
            custom_headers={},
        ),
    )

    compact_request_text = _messages_text(llm.requests[0].messages)
    assert context_packet.context_profile == "compact"
    assert len(llm.requests) == 1
    assert "history message 0" in compact_request_text
    assert "history message 5" in compact_request_text
    assert "history message 6" not in compact_request_text
    assert llm.requests[0].tools is None
    assert len(turn_input.conversation_messages) == 4
    assert turn_input.conversation_messages[0]["content"] == "history message 6"
    assert turn_input.conversation_messages[-1]["content"] == "history message 9"
    assert "setup.world_background.write_entry" not in turn_input.tool_scope
    assert "setup.stage_entry.write" in turn_input.tool_scope
    assert "setup.truth.write" not in turn_input.tool_scope
    assert "setup.question.raise" not in turn_input.tool_scope
    assert "setup.proposal.commit" not in turn_input.tool_scope
    assert "setup.discussion.update_state" not in turn_input.tool_scope
    assert "setup.chunk.upsert" not in turn_input.tool_scope
    assert "setup.patch.foundation_entry" not in turn_input.tool_scope
    assert "setup.patch.story_config" not in turn_input.tool_scope
    assert "setup.patch.longform_blueprint" not in turn_input.tool_scope
    assert EXTERNAL_MEMORY_TOOLS.isdisjoint(turn_input.tool_scope)
    assert turn_input.context_bundle["governance_metadata"]["raw_history_limit"] == 4
    assert turn_input.context_bundle["governance_metadata"]["kept_history_count"] == 4
    assert (
        turn_input.context_bundle["governance_metadata"]["compacted_history_count"] == 6
    )
    assert turn_input.context_bundle["compact_summary"] is not None
    assert "context_report" not in turn_input.context_bundle
    assert turn_input.metadata["context_report"] is not None
    assert turn_input.metadata["context_report"]["context_profile"] == "compact"
    assert (
        "history_count_threshold"
        in turn_input.metadata["context_report"]["profile_reasons"]
    )
    assert (
        "user_edit_threshold"
        in turn_input.metadata["context_report"]["profile_reasons"]
    )
    assert turn_input.metadata["context_report"]["estimated_input_tokens"] is not None
    assert turn_input.metadata["context_report"].get("previous_prompt_tokens") is None
    assert (
        turn_input.metadata["context_report"]["summary_strategy"]
        == "compact_prompt_summary"
    )
    assert turn_input.metadata["context_report"]["summary_action"] == "rebuilt"
    assert turn_input.metadata["context_pipeline"]["final_request_message_order"] == [
        "stable_system_prompt",
        "runtime_overlay_system_message",
        "governed_history",
        "current_user",
    ]
    assert turn_input.metadata["context_pipeline"]["context_profile"] == "compact"
    assert (
        "context_report"
        in turn_input.metadata["context_pipeline"]["metadata_only_surfaces"]
    )
    assert (
        turn_input.metadata["context_pipeline"]["prompt_assembly"][
            "capability_guidance_source"
        ]
        == "SetupCapabilityPlan.prompt_guidance_fragments"
    )


@pytest.mark.asyncio
async def test_setup_agent_execution_service_v2_appends_external_mcp_tool_allowlist(
    retrieval_session,
):
    workspace_service = SetupWorkspaceService(retrieval_session)
    context_builder = SetupContextBuilder(workspace_service)
    runtime_state_service = SetupAgentRuntimeStateService(retrieval_session)
    service = SetupAgentExecutionService(
        workspace_service=workspace_service,
        context_builder=context_builder,
        runtime_state_service=runtime_state_service,
    )
    workspace = workspace_service.create_workspace(
        story_id="story-setup-external-mcp-scope",
        mode=StoryMode.LONGFORM,
    )
    request = SetupAgentTurnRequest(
        workspace_id=workspace.workspace_id,
        model_id="model-1",
        target_stage=SetupStageId.WORLD_BACKGROUND,
        user_prompt="Use enabled MCP if needed.",
        external_mcp_tool_allowlist=[
            "web__search",
            "web__search",
            "  web__fetch  ",
            "",
        ],
    )

    turn_input, _ = await service._build_runtime_v2_turn_input(
        adapter=SetupRuntimeAdapter(),
        request=request,
        workspace=workspace,
        model_name="gpt-4o-mini",
        provider=ProviderConfig(
            type="openai",
            api_key="sk-test",
            api_url="https://example.com/v1",
            custom_headers={},
        ),
    )

    assert "setup.memory.search" in turn_input.tool_scope
    assert "setup.stage_entry.write" in turn_input.tool_scope
    assert turn_input.tool_scope[-2:] == ["web__search", "web__fetch"]
    assert turn_input.metadata["external_mcp_tool_allowlist"] == [
        "web__search",
        "web__fetch",
    ]


@pytest.mark.asyncio
async def test_setup_agent_runtime_v2_recovers_compacted_draft_ref_detail(
    retrieval_session,
):
    workspace_service = SetupWorkspaceService(retrieval_session)
    context_builder = SetupContextBuilder(workspace_service)
    runtime_state_service = SetupAgentRuntimeStateService(retrieval_session)
    adapter = SetupRuntimeAdapter()
    compact_llm = _CompactPromptLLM()
    service = SetupAgentExecutionService(
        workspace_service=workspace_service,
        context_builder=context_builder,
        adapter=adapter,
        runtime_executor=None,
        runtime_state_service=runtime_state_service,
        llm_service=compact_llm,
    )
    workspace = workspace_service.create_workspace(
        story_id="story-compact-draft-ref-recovery-1",
        mode=StoryMode.LONGFORM,
    )
    workspace_service.patch_stage_draft(
        workspace_id=workspace.workspace_id,
        stage_id=SetupStageId.WORLD_BACKGROUND,
        draft=SetupStageDraftBlock(
            stage_id=SetupStageId.WORLD_BACKGROUND,
            entries=[
                SetupDraftEntry(
                    entry_id="magic_law",
                    entry_type="rule",
                    semantic_path="world_background.rule.magic_law",
                    title="Magic Law",
                    summary="Spellcasting permit law.",
                    tags=["law", "magic"],
                    sections=[
                        SetupDraftSection(
                            section_id="summary",
                            title="Summary",
                            kind="text",
                            content={"text": _RECOVERED_MAGIC_LAW_DETAIL},
                            retrieval_role="summary",
                        )
                    ],
                )
            ],
        ),
    )
    workspace = workspace_service.get_workspace(workspace.workspace_id)
    assert workspace is not None
    seed_packet = context_builder.build(
        SetupContextBuilderInput(
            mode=workspace.mode.value,
            workspace_id=workspace.workspace_id,
            current_step=SetupStepId.LONGFORM_BLUEPRINT.value,
            user_prompt="",
            user_edit_delta_ids=[],
            token_budget=SetupAgentExecutionService._COMPACT_CONTEXT_TOKEN_BUDGET,
        )
    )
    runtime_state_service.persist_turn_governance(
        workspace=workspace,
        context_packet=seed_packet,
        step_id=SetupStepId.LONGFORM_BLUEPRINT,
        working_digest=SetupWorkingDigest(
            current_goal="Continue blueprint planning from compact refs.",
            next_focus="Recover the exact magic-law constraint before using it.",
            draft_refs=["stage:world_background:magic_law:summary"],
        ),
        tool_outcomes=[],
        compact_summary=None,
    )
    old_raw_marker = "OLD_RAW_HISTORY_OUTSIDE_SUMMARY_WINDOW"
    request = SetupAgentTurnRequest(
        workspace_id=workspace.workspace_id,
        model_id="model-1",
        user_prompt="Use the recovered magic-law detail in the blueprint plan.",
        target_step=SetupStepId.LONGFORM_BLUEPRINT,
        history=[
            SetupAgentDialogueMessage(
                role="user" if index % 2 == 0 else "assistant",
                content=(
                    f"{old_raw_marker} compact candidate {index}"
                    if index == 0
                    else f"compact candidate history {index}"
                ),
            )
            for index in range(12)
        ],
        user_edit_delta_ids=["delta-1", "delta-2", "delta-3"],
    )

    turn_input, context_packet = await service._build_runtime_v2_turn_input(
        adapter=adapter,
        request=request,
        workspace=workspace,
        model_name="gpt-4o-mini",
        provider=ProviderConfig(
            type="openai",
            api_key="sk-test",
            api_url="https://example.com/v1",
            custom_headers={},
        ),
    )

    compact_summary = turn_input.context_bundle["compact_summary"]
    assert context_packet.context_profile == "compact"
    assert len(compact_llm.requests) == 1
    assert len(turn_input.conversation_messages) == 4
    assert all(
        old_raw_marker not in str(message.get("content") or "")
        for message in turn_input.conversation_messages
    )
    assert compact_summary is not None
    assert compact_summary["draft_refs"] == ["stage:world_background:magic_law:summary"]
    assert compact_summary["recovery_hints"][0]["ref"] == (
        "stage:world_background:magic_law:summary"
    )
    assert _RECOVERED_MAGIC_LAW_DETAIL not in json.dumps(
        turn_input.model_dump(mode="json", exclude_none=True),
        sort_keys=True,
    )

    provider = SetupToolProvider(
        workspace_service=workspace_service,
        context_builder=context_builder,
        runtime_state_service=runtime_state_service,
    )
    tool_executor = _SetupProviderBackedToolExecutor(provider=provider)
    llm = _MemoryOpenRecoveryLLM(workspace_id=workspace.workspace_id)
    executor = RpAgentRuntimeExecutor(tool_executor_factory=lambda _: tool_executor)

    result = await executor.run(
        turn_input,
        adapter.build_runtime_profile(),
        llm_service=llm,
    )

    assert result.status == "completed"
    assert result.assistant_text.endswith(_RECOVERED_MAGIC_LAW_DETAIL)
    assert llm.recovered_detail == _RECOVERED_MAGIC_LAW_DETAIL
    assert llm.recovered_from_tool_result is True
    assert tool_executor.calls[0][0].tool_name == "rp_setup__setup_memory_open"
    assert result.tool_results[0].success is True
    assert _RECOVERED_MAGIC_LAW_DETAIL in result.tool_results[0].content_text
    assert result.structured_payload["compact_summary"]["draft_refs"] == [
        "stage:world_background:magic_law:summary"
    ]


@pytest.mark.asyncio
async def test_setup_agent_runtime_v2_recovers_detail_through_session_memory(
    retrieval_session,
):
    workspace_service = SetupWorkspaceService(retrieval_session)
    context_builder = SetupContextBuilder(workspace_service)
    runtime_state_service = SetupAgentRuntimeStateService(retrieval_session)
    adapter = SetupRuntimeAdapter()
    service = SetupAgentExecutionService(
        workspace_service=workspace_service,
        context_builder=context_builder,
        adapter=adapter,
        runtime_executor=None,
        runtime_state_service=runtime_state_service,
    )
    workspace = workspace_service.create_workspace(
        story_id="story-setup-memory-e2e-1",
        mode=StoryMode.LONGFORM,
    )
    workspace_service.patch_stage_draft(
        workspace_id=workspace.workspace_id,
        stage_id=SetupStageId.WORLD_BACKGROUND,
        draft=SetupStageDraftBlock(
            stage_id=SetupStageId.WORLD_BACKGROUND,
            entries=[
                SetupDraftEntry(
                    entry_id="race_elf",
                    entry_type="race",
                    semantic_path="world_background.race.elf",
                    title="Elf",
                    summary="A forest ancestry with a recoverable city detail.",
                    tags=["forest"],
                    sections=[
                        SetupDraftSection(
                            section_id="summary",
                            title="Summary",
                            kind="text",
                            content={"text": _RECOVERED_MEMORY_DETAIL},
                            retrieval_role="summary",
                        )
                    ],
                )
            ],
        ),
    )
    workspace = workspace_service.get_workspace(workspace.workspace_id)
    assert workspace is not None
    request = SetupAgentTurnRequest(
        workspace_id=workspace.workspace_id,
        model_id="model-1",
        user_prompt=(
            "Recover the exact world-background detail about Moonlit forest "
            "before using it in the current setup discussion."
        ),
        target_stage=SetupStageId.CHARACTER_DESIGN,
        history=[],
        user_edit_delta_ids=[],
    )

    turn_input, context_packet = await service._build_runtime_v2_turn_input(
        adapter=adapter,
        request=request,
        workspace=workspace,
        model_name="gpt-4o-mini",
        provider=ProviderConfig(
            type="openai",
            api_key="sk-test",
            api_url="https://example.com/v1",
            custom_headers={},
        ),
    )

    assert context_packet.current_stage == SetupStageId.CHARACTER_DESIGN
    assert _RECOVERED_MEMORY_DETAIL not in json.dumps(
        turn_input.model_dump(mode="json", exclude_none=True),
        sort_keys=True,
    )
    provider = SetupToolProvider(
        workspace_service=workspace_service,
        context_builder=context_builder,
        runtime_state_service=runtime_state_service,
    )
    tool_executor = _SetupProviderBackedToolExecutor(provider=provider)
    llm = _SetupMemorySearchThenOpenLLM(workspace_id=workspace.workspace_id)
    executor = RpAgentRuntimeExecutor(tool_executor_factory=lambda _: tool_executor)

    result = await executor.run(
        turn_input,
        adapter.build_runtime_profile(),
        llm_service=llm,
    )

    assert result.status == "completed"
    assert result.assistant_text.endswith(_RECOVERED_MEMORY_DETAIL)
    assert llm.search_hit_ref == "stage:world_background:race_elf:summary"
    assert llm.search_carried_payload is False
    assert llm.recovered_detail == _RECOVERED_MEMORY_DETAIL
    assert [call[0].tool_name for call in tool_executor.calls] == [
        "rp_setup__setup_memory_search",
        "rp_setup__setup_memory_open",
    ]
    assert [tool_result.success for tool_result in result.tool_results] == [
        True,
        True,
    ]
    search_payload = json.loads(result.tool_results[0].content_text)
    open_payload = json.loads(result.tool_results[1].content_text)
    assert search_payload["items"][0]["ref"] == (
        "stage:world_background:race_elf:summary"
    )
    assert "payload" not in search_payload["items"][0]
    assert open_payload["content"]["text"] == _RECOVERED_MEMORY_DETAIL


@pytest.mark.asyncio
async def test_setup_agent_execution_service_v2_surfaces_previous_usage_pressure(
    retrieval_session,
):
    workspace_service = SetupWorkspaceService(retrieval_session)
    context_builder = SetupContextBuilder(workspace_service)
    runtime_state_service = SetupAgentRuntimeStateService(retrieval_session)
    adapter = SetupRuntimeAdapter()
    llm = _CompactPromptLLM()
    service = SetupAgentExecutionService(
        workspace_service=workspace_service,
        context_builder=context_builder,
        adapter=adapter,
        runtime_executor=None,
        runtime_state_service=runtime_state_service,
        llm_service=llm,
    )
    workspace = workspace_service.create_workspace(
        story_id="story-observed-usage-1",
        mode=StoryMode.LONGFORM,
    )
    service._record_runtime_usage(
        workspace_id=workspace.workspace_id,
        step_id=workspace.current_step,
        result=RpAgentTurnResult(
            status="completed",
            finish_reason="completed_text",
            assistant_text="Previous answer.",
            structured_payload={
                "latest_response": {
                    "usage": {
                        "prompt_tokens": 1901,
                        "completion_tokens": 8,
                        "total_tokens": 1909,
                        "prompt_tokens_details": {"cached_tokens": 13},
                        "completion_tokens_details": {"reasoning_tokens": 4},
                        "cache_creation_input_tokens": 2,
                        "cache_read_input_tokens": 13,
                    }
                }
            },
        ),
    )
    request = SetupAgentTurnRequest(
        workspace_id=workspace.workspace_id,
        model_id="model-1",
        user_prompt="Continue setup.",
        history=[],
        user_edit_delta_ids=[],
    )

    turn_input, context_packet = await service._build_runtime_v2_turn_input(
        adapter=adapter,
        request=request,
        workspace=workspace,
        model_name="gpt-4o-mini",
        provider=ProviderConfig(
            type="openai",
            api_key="sk-test",
            api_url="https://example.com/v1",
            custom_headers={},
        ),
    )

    report = turn_input.metadata["context_report"]
    assert context_packet.context_profile == "compact"
    assert len(llm.requests) == 0
    assert "observed_usage_threshold" in report["profile_reasons"]
    assert report["previous_prompt_tokens"] == 1901
    assert report["previous_completion_tokens"] == 8
    assert report["previous_total_tokens"] == 1909
    assert report["previous_cached_tokens"] == 13
    assert report["previous_reasoning_tokens"] == 4
    assert report["previous_cache_creation_input_tokens"] == 2
    assert report["previous_cache_read_input_tokens"] == 13
    assert report["previous_usage_source"] == "provider_usage_metadata"
    assert report["previous_token_details"]["prompt_tokens_details"] == {
        "cached_tokens": 13
    }


@pytest.mark.asyncio
async def test_setup_agent_execution_service_v2_does_not_share_observed_usage_across_workspaces(
    retrieval_session,
):
    workspace_service = SetupWorkspaceService(retrieval_session)
    context_builder = SetupContextBuilder(workspace_service)
    runtime_state_service = SetupAgentRuntimeStateService(retrieval_session)
    adapter = SetupRuntimeAdapter()
    service = SetupAgentExecutionService(
        workspace_service=workspace_service,
        context_builder=context_builder,
        adapter=adapter,
        runtime_executor=None,
        runtime_state_service=runtime_state_service,
    )
    noisy_workspace = workspace_service.create_workspace(
        story_id="story-observed-usage-noisy",
        mode=StoryMode.LONGFORM,
    )
    target_workspace = workspace_service.create_workspace(
        story_id="story-observed-usage-target",
        mode=StoryMode.LONGFORM,
    )
    service._record_runtime_usage(
        workspace_id=noisy_workspace.workspace_id,
        step_id=noisy_workspace.current_step,
        result=RpAgentTurnResult(
            status="completed",
            finish_reason="completed_text",
            assistant_text="Previous answer.",
            structured_payload={
                "latest_response": {
                    "usage": {
                        "prompt_tokens": 2200,
                        "completion_tokens": 8,
                        "total_tokens": 2600,
                    }
                }
            },
        ),
    )
    request = SetupAgentTurnRequest(
        workspace_id=target_workspace.workspace_id,
        model_id="model-1",
        user_prompt="Continue setup.",
        history=[],
        user_edit_delta_ids=[],
    )

    turn_input, context_packet = await service._build_runtime_v2_turn_input(
        adapter=adapter,
        request=request,
        workspace=target_workspace,
        model_name="gpt-4o-mini",
        provider=ProviderConfig(
            type="openai",
            api_key="sk-test",
            api_url="https://example.com/v1",
            custom_headers={},
        ),
    )

    report = turn_input.metadata["context_report"]
    assert context_packet.context_profile == "standard"
    assert "observed_usage_threshold" not in report["profile_reasons"]
    assert report.get("previous_prompt_tokens") is None
    assert report.get("previous_total_tokens") is None


@pytest.mark.asyncio
async def test_setup_agent_execution_service_v2_uses_target_step_for_tool_scope(
    retrieval_session,
):
    workspace_service = SetupWorkspaceService(retrieval_session)
    context_builder = SetupContextBuilder(workspace_service)
    runtime_state_service = SetupAgentRuntimeStateService(retrieval_session)
    adapter = SetupRuntimeAdapter()
    service = SetupAgentExecutionService(
        workspace_service=workspace_service,
        context_builder=context_builder,
        adapter=adapter,
        runtime_executor=None,
        runtime_state_service=runtime_state_service,
    )
    workspace = workspace_service.create_workspace(
        story_id="story-tool-scope-override-1",
        mode=StoryMode.LONGFORM,
    )
    request = SetupAgentTurnRequest(
        workspace_id=workspace.workspace_id,
        model_id="model-1",
        user_prompt="Adjust story config.",
        target_step=SetupStepId.STORY_CONFIG,
        history=[],
        user_edit_delta_ids=[],
    )

    turn_input, _ = await service._build_runtime_v2_turn_input(
        adapter=adapter,
        request=request,
        workspace=workspace,
        model_name="gpt-4o-mini",
        provider=ProviderConfig(
            type="openai",
            api_key="sk-test",
            api_url="https://example.com/v1",
            custom_headers={},
        ),
    )

    assert "setup.stage_entry.write" not in turn_input.tool_scope
    assert "setup.patch.story_config" not in turn_input.tool_scope
    assert "setup.patch.foundation_entry" not in turn_input.tool_scope


@pytest.mark.asyncio
async def test_setup_agent_execution_service_v2_uses_current_stage_metadata(
    retrieval_session,
):
    workspace_service = SetupWorkspaceService(retrieval_session)
    context_builder = SetupContextBuilder(workspace_service)
    runtime_state_service = SetupAgentRuntimeStateService(retrieval_session)
    adapter = SetupRuntimeAdapter()
    service = SetupAgentExecutionService(
        workspace_service=workspace_service,
        context_builder=context_builder,
        adapter=adapter,
        runtime_executor=None,
        runtime_state_service=runtime_state_service,
    )
    workspace = workspace_service.create_workspace(
        story_id="story-stage-turn-input-1",
        mode=StoryMode.LONGFORM,
    )
    request = SetupAgentTurnRequest(
        workspace_id=workspace.workspace_id,
        model_id="model-1",
        user_prompt="Continue world setup.",
        history=[],
        user_edit_delta_ids=[],
    )

    turn_input, context_packet = await service._build_runtime_v2_turn_input(
        adapter=adapter,
        request=request,
        workspace=workspace,
        model_name="gpt-4o-mini",
        provider=ProviderConfig(
            type="openai",
            api_key="sk-test",
            api_url="https://example.com/v1",
            custom_headers={},
        ),
    )

    assert context_packet.current_stage == SetupStageId.WORLD_BACKGROUND
    assert turn_input.context_bundle["current_step"] == "foundation"
    assert turn_input.context_bundle["current_stage"] == "world_background"
    assert turn_input.context_bundle["stage_state"]["stage_id"] == "world_background"
    capability_plan = turn_input.metadata["capability_plan"]
    assert capability_plan["stage_id"] == "world_background"
    assert capability_plan["step_id"] == "foundation"
    assert capability_plan["runtime_allowlist"] == turn_input.tool_scope
    assert "capability_plan" not in turn_input.context_bundle
    assert "skill_pack_name" not in turn_input.context_bundle
    assert "context_pipeline" not in turn_input.context_bundle
    assert "Active capability guidance:" in turn_input.context_bundle["system_prompt"]
    assert "setup.world_background.write_entry" not in turn_input.tool_scope
    assert "setup.stage_entry.write" in turn_input.tool_scope
    assert "setup.truth.write" not in turn_input.tool_scope
    assert "setup.question.raise" not in turn_input.tool_scope
    assert "setup.proposal.commit" not in turn_input.tool_scope
    assert "setup.discussion.update_state" not in turn_input.tool_scope
    assert "setup.chunk.upsert" not in turn_input.tool_scope
    assert "setup.patch.foundation_entry" not in turn_input.tool_scope
    assert "setup.patch.story_config" not in turn_input.tool_scope


@pytest.mark.asyncio
async def test_setup_agent_execution_service_v2_uses_target_stage_override(
    retrieval_session,
):
    workspace_service = SetupWorkspaceService(retrieval_session)
    context_builder = SetupContextBuilder(workspace_service)
    runtime_state_service = SetupAgentRuntimeStateService(retrieval_session)
    adapter = SetupRuntimeAdapter()
    service = SetupAgentExecutionService(
        workspace_service=workspace_service,
        context_builder=context_builder,
        adapter=adapter,
        runtime_executor=None,
        runtime_state_service=runtime_state_service,
    )
    workspace = workspace_service.create_workspace(
        story_id="story-target-stage-override-1",
        mode=StoryMode.LONGFORM,
    )
    request = SetupAgentTurnRequest(
        workspace_id=workspace.workspace_id,
        model_id="model-1",
        user_prompt="Continue character design.",
        target_stage=SetupStageId.CHARACTER_DESIGN,
        history=[],
        user_edit_delta_ids=[],
    )
    provider = ProviderConfig(
        type="openai",
        api_key="sk-test",
        api_url="https://example.com/v1",
        custom_headers={},
    )

    service._ensure_agent_model_compatible = lambda model_id: None  # type: ignore[method-assign]
    service._resolve_provider = (  # type: ignore[method-assign]
        lambda *, model_id, provider_id: provider
    )
    service._resolve_model_name = (  # type: ignore[method-assign]
        lambda *, model_id, fallback_provider_id: "gpt-4o-mini"
    )

    launch = service._prepare_turn_launch(request)
    turn_input, context_packet = await service._build_runtime_v2_turn_input(
        adapter=adapter,
        request=request,
        workspace=workspace,
        model_name="gpt-4o-mini",
        provider=provider,
    )

    assert launch.current_stage == SetupStageId.CHARACTER_DESIGN
    assert launch.current_step == SetupStepId.FOUNDATION
    assert context_packet.current_stage == SetupStageId.CHARACTER_DESIGN
    assert context_packet.current_step == "foundation"
    assert turn_input.context_bundle["current_step"] == "foundation"
    assert turn_input.context_bundle["current_stage"] == "character_design"
    assert turn_input.context_bundle["stage_state"]["stage_id"] == "character_design"
    assert turn_input.metadata["skill_pack_name"] == "character-design.v1"
    assert (
        turn_input.metadata["context_pipeline"]["prompt_assembly"][
            "active_skill_pack_name"
        ]
        == "character-design.v1"
    )
    assert turn_input.metadata["capability_plan"] == build_setup_agent_capability_plan(
        "foundation",
        current_stage="character_design",
    ).model_dump(mode="json", exclude_none=True)
    assert turn_input.tool_scope == build_setup_agent_tool_scope("character_design")


def test_setup_agent_execution_service_prepare_turn_launch_reuses_shared_preflight(
    retrieval_session,
    monkeypatch,
):
    workspace_service = SetupWorkspaceService(retrieval_session)
    context_builder = SetupContextBuilder(workspace_service)
    service = SetupAgentExecutionService(
        workspace_service=workspace_service,
        context_builder=context_builder,
    )
    workspace = workspace_service.create_workspace(
        story_id="story-launch-preflight-1",
        mode=StoryMode.LONGFORM,
    )
    request = SetupAgentTurnRequest(
        workspace_id=workspace.workspace_id,
        model_id="model-1",
        user_prompt="Adjust story config.",
        target_step=SetupStepId.STORY_CONFIG,
        history=[],
        user_edit_delta_ids=[],
    )
    provider = ProviderConfig(
        type="openai",
        api_key="sk-test",
        api_url="https://example.com/v1",
        custom_headers={},
    )
    seen_model_ids: list[str] = []

    monkeypatch.setattr(
        service,
        "_ensure_agent_model_compatible",
        lambda model_id: seen_model_ids.append(model_id),
    )
    monkeypatch.setattr(
        service,
        "_resolve_provider",
        lambda *, model_id, provider_id: provider,
    )
    monkeypatch.setattr(
        service,
        "_resolve_model_name",
        lambda *, model_id, fallback_provider_id: "gpt-4o-mini",
    )

    launch = service._prepare_turn_launch(request)

    assert seen_model_ids == ["model-1"]
    assert launch.workspace.workspace_id == workspace.workspace_id
    assert launch.current_step == SetupStepId.STORY_CONFIG
    assert launch.model_name == "gpt-4o-mini"
    assert launch.provider == provider


def test_setup_agent_execution_service_prepare_turn_launch_rejects_target_stage_step_mismatch(
    retrieval_session,
    monkeypatch,
):
    workspace_service = SetupWorkspaceService(retrieval_session)
    context_builder = SetupContextBuilder(workspace_service)
    service = SetupAgentExecutionService(
        workspace_service=workspace_service,
        context_builder=context_builder,
    )
    workspace = workspace_service.create_workspace(
        story_id="story-launch-stage-mismatch-1",
        mode=StoryMode.LONGFORM,
    )
    request = SetupAgentTurnRequest(
        workspace_id=workspace.workspace_id,
        model_id="model-1",
        user_prompt="Adjust setup.",
        target_stage=SetupStageId.CHARACTER_DESIGN,
        target_step=SetupStepId.STORY_CONFIG,
        history=[],
        user_edit_delta_ids=[],
    )
    provider = ProviderConfig(
        type="openai",
        api_key="sk-test",
        api_url="https://example.com/v1",
        custom_headers={},
    )

    monkeypatch.setattr(
        service,
        "_ensure_agent_model_compatible",
        lambda model_id: None,
    )
    monkeypatch.setattr(
        service,
        "_resolve_provider",
        lambda *, model_id, provider_id: provider,
    )
    monkeypatch.setattr(
        service,
        "_resolve_model_name",
        lambda *, model_id, fallback_provider_id: "gpt-4o-mini",
    )

    with pytest.raises(
        ValueError,
        match="setup_target_stage_step_mismatch:character_design:story_config:foundation",
    ):
        service._prepare_turn_launch(request)


def test_setup_agent_execution_service_prepare_turn_launch_rejects_target_stage_outside_stage_plan(
    retrieval_session,
    monkeypatch,
):
    workspace_service = SetupWorkspaceService(retrieval_session)
    context_builder = SetupContextBuilder(workspace_service)
    service = SetupAgentExecutionService(
        workspace_service=workspace_service,
        context_builder=context_builder,
    )
    workspace = workspace_service.create_workspace(
        story_id="story-launch-stage-plan-1",
        mode=StoryMode.ROLEPLAY,
    )
    request = SetupAgentTurnRequest(
        workspace_id=workspace.workspace_id,
        model_id="model-1",
        user_prompt="Adjust setup.",
        target_stage=SetupStageId.PLOT_BLUEPRINT,
        history=[],
        user_edit_delta_ids=[],
    )
    provider = ProviderConfig(
        type="openai",
        api_key="sk-test",
        api_url="https://example.com/v1",
        custom_headers={},
    )

    monkeypatch.setattr(
        service,
        "_ensure_agent_model_compatible",
        lambda model_id: None,
    )
    monkeypatch.setattr(
        service,
        "_resolve_provider",
        lambda *, model_id, provider_id: provider,
    )
    monkeypatch.setattr(
        service,
        "_resolve_model_name",
        lambda *, model_id, fallback_provider_id: "gpt-4o-mini",
    )

    with pytest.raises(
        ValueError,
        match="setup_target_stage_not_in_stage_plan:plot_blueprint:roleplay",
    ):
        service._prepare_turn_launch(request)


@pytest.mark.asyncio
async def test_setup_agent_execution_service_prepare_runtime_v2_launch_sets_stream_flag(
    retrieval_session,
    monkeypatch,
):
    workspace_service = SetupWorkspaceService(retrieval_session)
    context_builder = SetupContextBuilder(workspace_service)
    runtime_state_service = SetupAgentRuntimeStateService(retrieval_session)
    adapter = SetupRuntimeAdapter()
    service = SetupAgentExecutionService(
        workspace_service=workspace_service,
        context_builder=context_builder,
        adapter=adapter,
        runtime_executor=None,
        runtime_state_service=runtime_state_service,
    )
    workspace = workspace_service.create_workspace(
        story_id="story-runtime-launch-1",
        mode=StoryMode.LONGFORM,
    )
    request = SetupAgentTurnRequest(
        workspace_id=workspace.workspace_id,
        model_id="model-1",
        user_prompt="Continue foundation.",
        target_step=SetupStepId.FOUNDATION,
        history=[],
        user_edit_delta_ids=[],
    )
    provider = ProviderConfig(
        type="openai",
        api_key="sk-test",
        api_url="https://example.com/v1",
        custom_headers={},
    )
    monkeypatch.setattr(
        service, "_ensure_agent_model_compatible", lambda model_id: None
    )
    monkeypatch.setattr(
        service,
        "_resolve_provider",
        lambda *, model_id, provider_id: provider,
    )
    monkeypatch.setattr(
        service,
        "_resolve_model_name",
        lambda *, model_id, fallback_provider_id: "gpt-4o-mini",
    )

    launch = service._prepare_turn_launch(request)
    prepared = await service._prepare_runtime_v2_launch(
        adapter=adapter,
        launch=launch,
        stream=True,
    )

    assert prepared.turn_input.stream is True
    assert prepared.turn_input.context_bundle["current_step"] == "foundation"
    assert "setup.world_background.write_entry" not in prepared.turn_input.tool_scope
    assert "setup.stage_entry.write" in prepared.turn_input.tool_scope
    assert "setup.truth.write" not in prepared.turn_input.tool_scope
    assert "setup.question.raise" not in prepared.turn_input.tool_scope
    assert "setup.proposal.commit" not in prepared.turn_input.tool_scope
    assert "setup.discussion.update_state" not in prepared.turn_input.tool_scope
    assert "setup.chunk.upsert" not in prepared.turn_input.tool_scope
    assert "setup.patch.foundation_entry" not in prepared.turn_input.tool_scope
    assert "setup.patch.story_config" not in prepared.turn_input.tool_scope
    assert prepared.context_packet.workspace_id == workspace.workspace_id
    assert prepared.profile.profile_id == "setup_agent"
    metadata = service._runtime_v2_observation_metadata(prepared)
    assert metadata["context_pipeline"]["final_request_message_order"] == [
        "stable_system_prompt",
        "runtime_overlay_system_message",
        "governed_history",
        "current_user",
    ]
    assert "context_report" not in metadata["context_pipeline"]


@pytest.mark.asyncio
async def test_setup_agent_execution_service_v2_falls_back_when_compact_prompt_fails(
    retrieval_session,
):
    workspace_service = SetupWorkspaceService(retrieval_session)
    context_builder = SetupContextBuilder(workspace_service)
    runtime_state_service = SetupAgentRuntimeStateService(retrieval_session)
    adapter = SetupRuntimeAdapter()
    llm = _CompactPromptLLM(invalid_json=True)
    service = SetupAgentExecutionService(
        workspace_service=workspace_service,
        context_builder=context_builder,
        adapter=adapter,
        runtime_executor=None,
        runtime_state_service=runtime_state_service,
        llm_service=llm,
    )
    workspace = workspace_service.create_workspace(
        story_id="story-compact-prompt-fallback-1",
        mode=StoryMode.LONGFORM,
    )
    request = SetupAgentTurnRequest(
        workspace_id=workspace.workspace_id,
        model_id="model-1",
        user_prompt="Continue with a compact history.",
        history=[
            SetupAgentDialogueMessage(
                role="user" if index % 2 == 0 else "assistant",
                content=f"fallback history message {index}",
            )
            for index in range(10)
        ],
        user_edit_delta_ids=["delta-1", "delta-2", "delta-3"],
    )

    turn_input, context_packet = await service._build_runtime_v2_turn_input(
        adapter=adapter,
        request=request,
        workspace=workspace,
        model_name="gpt-4o-mini",
        provider=ProviderConfig(
            type="openai",
            api_key="sk-test",
            api_url="https://example.com/v1",
            custom_headers={},
        ),
    )

    report = turn_input.metadata["context_report"]
    compact_summary = turn_input.context_bundle["compact_summary"]
    assert context_packet.context_profile == "compact"
    assert len(llm.requests) == 1
    assert report["summary_strategy"] == "deterministic_prefix_summary"
    assert report["summary_action"] == "rebuilt"
    assert report["fallback_reason"] is not None
    assert compact_summary is not None
    assert compact_summary["summary_lines"][0].startswith("User: fallback history")
