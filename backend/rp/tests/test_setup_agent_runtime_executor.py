"""Unit tests for the setup-agent runtime executor."""

from __future__ import annotations

import json
from typing import Any

import pytest

from rp.agent_runtime.contracts import (
    RpAgentTurnInput,
    RuntimeProfile,
    RuntimeToolCall,
    RuntimeToolResult,
)
from rp.agent_runtime.events import RuntimeEvent, TypedSseEventAdapter
from rp.agent_runtime.executor import RpAgentRuntimeExecutor, _RuntimeRunDriver
from rp.agent_runtime.state import RpAgentRunState
from rp.agent_runtime.tools import RuntimeToolExecutor


def _turn_input(
    *,
    stream: bool = False,
    user_prompt: str = "Please help with setup.",
    conversation_messages: list[dict[str, Any]] | None = None,
    context_bundle: dict | None = None,
    tool_scope: list[str] | None = None,
) -> RpAgentTurnInput:
    runtime_context = {
        "system_prompt": "You are SetupAgent.",
        "current_step": "story_config",
        "context_packet": {
            "current_step": "story_config",
            "committed_summaries": ["world.magic.law"],
            "current_draft_snapshot": {"notes": ["existing"]},
            "prior_stage_handoffs": [
                {
                    "step_id": "foundation",
                    "commit_id": "commit-foundation-1",
                    "summary": "world.magic.law",
                    "committed_refs": ["draft:foundation"],
                    "spotlights": ["Magic Law"],
                    "chunk_descriptions": [
                        {
                            "chunk_ref": "foundation:magic-law",
                            "block_type": "foundation_entry",
                            "title": "Magic Law",
                            "description": "rule | world.magic.law - Public spellcasting is regulated by guild permits.",
                            "metadata": {
                                "domain": "rule",
                                "path": "world.magic.law",
                                "entry_id": "magic-law",
                            },
                        }
                    ],
                    "created_at": "2026-04-27T00:00:00Z",
                }
            ],
            "spotlights": ["Magic Law"],
            "user_prompt": user_prompt,
        },
        "open_question_count": 0,
        "blocking_open_question_count": 0,
        "open_question_texts": [],
        "has_prior_stage_handoffs": True,
        "prior_stage_handoff_count": 1,
        "prior_stage_handoff_steps": ["foundation"],
        "last_proposal_status": None,
    }
    if context_bundle:
        runtime_context.update(context_bundle)
    return RpAgentTurnInput(
        profile_id="setup_agent",
        run_kind="interactive_agent_turn",
        story_id="story-1",
        workspace_id="workspace-1",
        model_id="model-1",
        provider_id="provider-1",
        stream=stream,
        user_visible_request=user_prompt,
        conversation_messages=list(conversation_messages or []),
        context_bundle=runtime_context,
        tool_scope=(
            tool_scope if tool_scope is not None else ["setup.stage_entry.write"]
        ),
        metadata={
            "model_name": "gpt-4o-mini",
            "provider": {
                "type": "openai",
                "api_key": "sk-test",
                "api_url": "https://example.com/v1",
                "custom_headers": {},
            },
        },
    )


def _profile() -> RuntimeProfile:
    return RuntimeProfile(
        profile_id="setup_agent",
        visible_tool_names=["setup.stage_entry.write"],
        max_rounds=8,
        allow_stream=True,
        recovery_policy="setup_agent_v1",
        finish_policy="assistant_text_or_failure",
    )


def test_runtime_executor_respects_explicit_empty_tool_scope():
    driver = _RuntimeRunDriver(
        llm_service=object(),
        profile=_profile(),
        tool_executor=_FakeToolExecutor(),
    )

    assert driver._visible_tool_names(_turn_input(tool_scope=[])) == []


def test_runtime_executor_uses_tool_scope_without_removed_agent_tools():
    driver = _RuntimeRunDriver(
        llm_service=object(),
        profile=_profile(),
        tool_executor=_FakeToolExecutor(),
    )
    removed_tools = {
        "setup.proposal.commit",
        "setup.question.raise",
        "setup.discussion.update_state",
        "setup.chunk.upsert",
        "setup.truth.write",
        "setup.patch.story_config",
        "setup.patch.writing_contract",
        "setup.patch.foundation_entry",
        "setup.patch.longform_blueprint",
        "memory.get_state",
        "memory.get_summary",
        "memory.search_recall",
        "memory.search_archival",
        "memory.list_versions",
        "memory.read_provenance",
        "setup.read.workspace",
        "setup.read.step_context",
        "setup.read.draft_refs",
        "setup.truth_index.search",
        "setup.truth_index.read_refs",
    }
    turn_input = _turn_input(
        tool_scope=[
            "setup.stage_entry.write",
            "setup.memory.search",
            "setup.memory.open",
            "setup.memory.read_refs",
        ]
    )

    assert removed_tools.isdisjoint(driver._visible_tool_names(turn_input))


def _compact_exact_detail_context() -> dict[str, Any]:
    return {
        "compact_summary": {
            "source_fingerprint": "fp-compact",
            "source_message_count": 8,
            "summary_lines": ["Older foundation discussion was compacted."],
            "draft_refs": ["stage:world_background:magic-law:summary"],
            "recovery_hints": [
                {
                    "ref": "stage:world_background:magic-law:summary",
                    "reason": "Exact magic-law detail lives in the draft.",
                }
            ],
        },
    }


def _compact_exact_detail_context_without_refs() -> dict[str, Any]:
    return {
        "compact_summary": {
            "source_fingerprint": "fp-compact-no-refs",
            "source_message_count": 12,
            "summary_lines": [
                "Violet Harbor and Mira's contact mechanism were established earlier."
            ],
            "draft_refs": [],
            "recovery_hints": [],
        },
    }


def _memory_search_result() -> RuntimeToolResult:
    return RuntimeToolResult(
        call_id="call_search",
        tool_name="rp_setup__setup.memory.search",
        success=True,
        content_text=json.dumps(
            {
                "success": True,
                "items": [
                    {
                        "ref": "stage:character_design:mira-contact:signal",
                        "title": "Signal",
                        "path": "character_design / contact / mira / signal",
                        "scope": "section",
                        "navigation_summary": "Exact contact signal and hidden object.",
                        "message": (
                            "这是搜索候选，不是事实正文。需要使用该设定时，请 open 此 ref。"
                        ),
                    }
                ],
            }
        ),
        structured_payload={
            "content_payload": {
                "items": [
                    {
                        "ref": "stage:character_design:mira-contact:signal",
                        "title": "Signal",
                    }
                ]
            }
        },
    )


def _memory_open_result() -> RuntimeToolResult:
    return RuntimeToolResult(
        call_id="call_open",
        tool_name="rp_setup__setup.memory.open",
        success=True,
        content_text=json.dumps(
            {
                "success": True,
                "result_type": "content",
                "opened_ref": "stage:character_design:mira-contact:signal",
                "opened_path": "character_design / contact / mira / signal",
                "message": "当前打开的是四级内容节点，以下内容可作为回答或写入草稿的事实依据。",
                "content": {
                    "type": "key_value",
                    "title": "Signal",
                    "values": {
                        "signal": "blue lanterns unlock the tidewall lattice",
                        "object": "copper astrolabe named Lumen Key",
                    },
                },
            }
        ),
        structured_payload={
            "content_payload": {
                "success": True,
                "result_type": "content",
                "opened_ref": "stage:character_design:mira-contact:signal",
            }
        },
    )


def _draft_ref_open_result() -> RuntimeToolResult:
    return RuntimeToolResult(
        call_id="call_open",
        tool_name="rp_setup__setup.memory.open",
        success=True,
        content_text=json.dumps(
            {
                "success": True,
                "result_type": "content",
                "opened_ref": "stage:world_background:magic-law:summary",
                "opened_path": "world_background / rule / magic-law / summary",
                "message": "当前打开的是四级内容节点，以下内容可作为回答或写入草稿的事实依据。",
                "content": {
                    "type": "text",
                    "title": "Summary",
                    "text": "Public spellcasting is regulated by guild permits.",
                },
            }
        ),
        structured_payload={
            "content_payload": {
                "success": True,
                "result_type": "content",
                "opened_ref": "stage:world_background:magic-law:summary",
            }
        },
    )


def _tool_definition(name: str) -> dict[str, Any]:
    parameters = {"type": "object", "properties": {}}
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "test tool",
            "parameters": parameters,
        },
    }


def test_runtime_routes_invalid_next_action_to_runtime_failed():
    state: RpAgentRunState = {"next_action": "undefined_route"}

    route = _RuntimeRunDriver._route_after_assess(state)

    assert route == "finalize_failure"
    assert state["next_action"] == "finalize_failure"
    assert state["finish_reason"] == "runtime_failed"
    error = state["error"]
    assert error is not None
    assert error["type"] == "runtime_failed"


class _FakeToolExecutor(RuntimeToolExecutor):
    def __init__(self, *, results: list[RuntimeToolResult] | None = None) -> None:
        self._results = list(results or [])
        self.calls: list[tuple[Any, list[str]]] = []

    def get_openai_tool_definitions(
        self, *, visible_tool_names: list[str]
    ) -> list[dict[str, Any]]:
        return [
            _tool_definition(
                name if name.startswith("rp_setup__") else f"rp_setup__{name}"
            )
            for name in visible_tool_names
        ]

    async def execute_tool_call(
        self, call: RuntimeToolCall, *, visible_tool_names: list[str]
    ) -> RuntimeToolResult:
        self.calls.append((call, list(visible_tool_names)))
        if self._results:
            return self._results.pop(0)
        return RuntimeToolResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            success=True,
            content_text='{"success": true}',
            error_code=None,
        )


def _driver_for_inspection() -> _RuntimeRunDriver:
    return _RuntimeRunDriver(
        llm_service=object(),
        tool_executor=_FakeToolExecutor(),
        profile=_profile(),
    )


def _inspection_state(message: dict[str, Any]) -> RpAgentRunState:
    driver = _driver_for_inspection()
    state = driver._initial_state(_turn_input(tool_scope=["setup.stage_entry.write"]))
    state["latest_response"] = {"message": message}
    return state


def test_output_inspector_classifies_pseudo_tool_text_without_public_text():
    driver = _driver_for_inspection()
    state = _inspection_state(
        {
            "role": "assistant",
            "content": (
                "tool_code print(default_api.rp_setup__setup.stage_entry.write("
                '{"entry_type":"rule","title":"Magic Law","sections":[]}))'
            ),
        }
    )

    update = driver._inspect_model_output(state)

    assert update["assistant_text"] == ""
    assert update["pending_tool_calls"] == []
    assert update["next_action"] == "reflect_if_needed"
    assert update["output_inspection"]["classification"] == "pseudo_tool_text"
    assert update["continue_reason"] == "completion_guard_retry"


def test_output_inspector_routes_repeated_invalid_tool_output_to_active_taxonomy():
    driver = _driver_for_inspection()
    state = _inspection_state(
        {
            "role": "assistant",
            "content": (
                "tool_code print(default_api.rp_setup__setup.stage_entry.write("
                '{"entry_type":"rule","title":"Magic Law","sections":[]}))'
            ),
        }
    )
    state["pseudo_tool_retry_count"] = 1

    update = driver._inspect_model_output(state)

    assert update["status"] == "failed"
    assert update["finish_reason"] == "repair_obligation_unfulfilled"
    assert update["error"]["type"] == "repair_obligation_unfulfilled"
    assert update["output_inspection"]["classification"] == "pseudo_tool_text"


def test_output_inspector_mixed_text_and_tool_call_keeps_text_private():
    driver = _driver_for_inspection()
    state = _inspection_state(
        {
            "role": "assistant",
            "content": "tool_code print(default_api.rp_setup__setup.stage_entry.write(...))",
            "tool_calls": [
                {
                    "id": "call_patch",
                    "function": {
                        "name": "rp_setup__setup.stage_entry.write",
                        "arguments": '{"workspace_id":"workspace-1","entry_type":"rule","title":"Magic Law","sections":[]}',
                    },
                }
            ],
        }
    )

    update = driver._inspect_model_output(state)

    assert update["assistant_text"] == ""
    assert update["next_action"] == "execute_tools"
    assert update["continue_reason"] == "tool_call_batch_pending"
    assert update["output_inspection"]["classification"] == "mixed_text_and_tool_call"
    assert update["pending_tool_calls"][0]["tool_name"] == (
        "rp_setup__setup.stage_entry.write"
    )


def test_output_inspector_malformed_tool_call_enters_repair_route():
    driver = _driver_for_inspection()
    state = _inspection_state(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_bad",
                    "function": {
                        "name": "rp_setup__setup.stage_entry.write",
                        "arguments": "{not valid json",
                    },
                }
            ],
        }
    )

    update = driver._inspect_model_output(state)

    assert update["assistant_text"] == ""
    assert update["pending_tool_calls"] == []
    assert update["next_action"] == "reflect_if_needed"
    assert update["output_inspection"]["classification"] == "malformed_tool_call"
    assert update["completion_guard"]["reason"] == "malformed_tool_call_emitted"


def test_output_inspector_empty_output_cannot_finalize_success():
    driver = _driver_for_inspection()
    state = _inspection_state({"role": "assistant", "content": ""})

    update = driver._inspect_model_output(state)

    assert update["assistant_text"] == ""
    assert update["next_action"] == "reflect_if_needed"
    assert update["continue_reason"] == "completion_guard_retry"
    assert update["output_inspection"]["classification"] == "empty_output"
    assert update["completion_guard"]["reason"] == "assistant_output_empty"


class _FakeLangfuseObservation:
    def __init__(self, *, sink: list[dict], name: str) -> None:
        self._sink = sink
        self._name = name

    def __enter__(self):
        self._sink.append({"kind": "observation_enter", "name": self._name})
        return self

    def __exit__(self, exc_type, exc, tb):
        self._sink.append({"kind": "observation_exit", "name": self._name})
        return False

    def update(self, **kwargs):
        self._sink.append(
            {"kind": "observation_update", "name": self._name, "payload": kwargs}
        )

    def score(self, **kwargs):
        self._sink.append({"kind": "score", "name": self._name, "payload": kwargs})

    def score_trace(self, **kwargs):
        self._sink.append(
            {"kind": "score_trace", "name": self._name, "payload": kwargs}
        )

    def start_as_current_observation(self, **kwargs):
        return _FakeLangfuseObservation(
            sink=self._sink,
            name=str(kwargs.get("name") or "unknown"),
        )


class _FakeLangfuseService:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def start_as_current_observation(self, **kwargs):
        return _FakeLangfuseObservation(
            sink=self.events,
            name=str(kwargs.get("name") or "unknown"),
        )

    def propagate_attributes(self, **kwargs):
        return _FakeLangfuseObservation(
            sink=self.events,
            name="propagate",
        )


class _TextOnlyLLM:
    async def chat_completion(self, request):
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Direct setup answer.",
                    }
                }
            ]
        }


class _RecordingTextOnlyLLM:
    def __init__(self) -> None:
        self.requests: list[Any] = []

    async def chat_completion(self, request):
        self.requests.append(request)
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Recorded direct answer.",
                    }
                }
            ]
        }


class _CompactExactDetailTextThenOpenLLM:
    def __init__(self) -> None:
        self.round = 0

    async def chat_completion(self, request):
        self.round += 1
        if self.round == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "The exact magic-law content is permits only.",
                        }
                    }
                ]
            }
        if self.round == 2:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_open",
                                    "type": "function",
                                    "function": {
                                        "name": "rp_setup__setup.memory.open",
                                        "arguments": json.dumps(
                                            {
                                                "workspace_id": "workspace-1",
                                                "ref": "stage:world_background:magic-law:summary",
                                            }
                                        ),
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Read result says the exact magic-law content is guild permits.",
                    }
                }
            ]
        }


class _CompactExactDetailReadThenTextLLM:
    def __init__(self) -> None:
        self.round = 0

    async def chat_completion(self, request):
        self.round += 1
        if self.round == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_open",
                                    "type": "function",
                                    "function": {
                                        "name": "rp_setup__setup.memory.open",
                                        "arguments": json.dumps(
                                            {
                                                "workspace_id": "workspace-1",
                                                "ref": "stage:world_background:magic-law:summary",
                                            }
                                        ),
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "The exact magic-law content is guild permits.",
                    }
                }
            ]
        }


class _CompactExactDetailMutateThenOpenLLM:
    def __init__(self) -> None:
        self.round = 0

    async def chat_completion(self, request):
        self.round += 1
        if self.round == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_write",
                                    "type": "function",
                                    "function": {
                                        "name": "rp_setup__setup.stage_entry.write",
                                        "arguments": json.dumps(
                                            {
                                                "entry_type": "rule",
                                                "title": "Magic Law",
                                                "summary": "Guessed from compact.",
                                                "sections": [],
                                            }
                                        ),
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        if self.round == 2:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_open",
                                    "type": "function",
                                    "function": {
                                        "name": "rp_setup__setup.memory.open",
                                        "arguments": json.dumps(
                                            {
                                                "workspace_id": "workspace-1",
                                                "ref": "stage:world_background:magic-law:summary",
                                            }
                                        ),
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "I read the draft ref before mutating.",
                    }
                }
            ]
        }


class _CompactExactDetailMixedOpenMutationLLM:
    def __init__(self) -> None:
        self.round = 0

    async def chat_completion(self, request):
        self.round += 1
        if self.round == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_open_mixed",
                                    "type": "function",
                                    "function": {
                                        "name": "rp_setup__setup.memory.open",
                                        "arguments": json.dumps(
                                            {
                                                "workspace_id": "workspace-1",
                                                "ref": "stage:world_background:magic-law:summary",
                                            }
                                        ),
                                    },
                                },
                                {
                                    "id": "call_write_mixed",
                                    "type": "function",
                                    "function": {
                                        "name": "rp_setup__setup.stage_entry.write",
                                        "arguments": json.dumps(
                                            {
                                                "entry_type": "rule",
                                                "title": "Magic Law",
                                                "summary": "Mixed batch.",
                                                "sections": [],
                                            }
                                        ),
                                    },
                                },
                            ],
                        }
                    }
                ]
            }
        if self.round == 2:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "I received the opened draft detail and wrote the update.",
                        }
                    }
                ]
            }


class _ExactSessionDetailSearchThenOpenLLM:
    def __init__(self) -> None:
        self.round = 0

    async def chat_completion(self, request):
        self.round += 1
        if self.round == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_search",
                                    "type": "function",
                                    "function": {
                                        "name": "rp_setup__setup.memory.search",
                                        "arguments": json.dumps(
                                            {
                                                "workspace_id": "workspace-1",
                                                "query": "Mira Violet Harbor 确切暗号 物件",
                                                "limit": 5,
                                            }
                                        ),
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        if self.round == 2:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_open",
                                    "type": "function",
                                    "function": {
                                        "name": "rp_setup__setup.memory.open",
                                        "arguments": json.dumps(
                                            {
                                                "workspace_id": "workspace-1",
                                                "ref": "stage:character_design:mira-contact:signal",
                                            }
                                        ),
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": (
                            "暗号是 blue lanterns unlock the tidewall lattice，"
                            "藏在 copper astrolabe named Lumen Key 里。"
                        ),
                    }
                }
            ]
        }


class _ToolThenTextLLM:
    def __init__(self) -> None:
        self.round = 0

    async def chat_completion(self, request):
        self.round += 1
        if self.round == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_stage_entry",
                                    "type": "function",
                                    "function": {
                                        "name": "rp_setup__setup.stage_entry.write",
                                        "arguments": json.dumps(
                                            {
                                                "entry_type": "rule",
                                                "title": "Magic Law",
                                                "summary": "Public spellcasting requires permits.",
                                                "sections": [],
                                            }
                                        ),
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Tool succeeded and I finished the answer.",
                    }
                }
            ]
        }


class _SchemaRepairLLM:
    def __init__(self) -> None:
        self.round = 0

    async def chat_completion(self, request):
        self.round += 1
        arguments: dict[str, Any]
        if self.round == 1:
            arguments = {"workspace_id": "workspace-1"}
        elif self.round == 2:
            arguments = {
                "entry_type": "rule",
                "title": "Magic Law",
                "summary": "Public spellcasting requires permits.",
                "sections": [],
            }
        else:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Corrected the tool call and finished.",
                        }
                    }
                ]
            }
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": f"call_{self.round}",
                                "type": "function",
                                "function": {
                                    "name": "rp_setup__setup.stage_entry.write",
                                    "arguments": json.dumps(arguments),
                                },
                            }
                        ],
                    }
                }
            ]
        }


class _UnknownToolLLM:
    async def chat_completion(self, request):
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_unknown",
                                "type": "function",
                                "function": {
                                    "name": "rp_setup__setup.unknown",
                                    "arguments": json.dumps(
                                        {"workspace_id": "workspace-1"}
                                    ),
                                },
                            }
                        ],
                    }
                }
            ]
        }


class _AskUserAfterFailureLLM:
    def __init__(self) -> None:
        self.round = 0

    async def chat_completion(self, request):
        self.round += 1
        if self.round == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_ask_user",
                                    "type": "function",
                                    "function": {
                                        "name": "rp_setup__setup.stage_entry.write",
                                        "arguments": json.dumps(
                                            {"workspace_id": "workspace-1"}
                                        ),
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "I still need the missing story config details. Which style rules do you want me to lock in?",
                    }
                }
            ]
        }


class _ExplainInsteadOfRepairLLM:
    def __init__(self) -> None:
        self.round = 0

    async def chat_completion(self, request):
        self.round += 1
        if self.round == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_bad_patch",
                                    "type": "function",
                                    "function": {
                                        "name": "rp_setup__setup.stage_entry.write",
                                        "arguments": json.dumps(
                                            {"workspace_id": "workspace-1"}
                                        ),
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "I know the patch field is missing, so I cannot continue yet.",
                    }
                }
            ]
        }


class _RepeatedQuestionLLM:
    async def chat_completion(self, request):
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Which style rules do you want me to lock in for this draft?",
                    }
                }
            ]
        }


class _StageEntryFailureThenDiscussionLLM:
    def __init__(self) -> None:
        self.round = 0

    async def chat_completion(self, request):
        self.round += 1
        if self.round == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_stage_entry",
                                    "type": "function",
                                    "function": {
                                        "name": "rp_setup__setup.stage_entry.write",
                                        "arguments": json.dumps(
                                            {
                                                "entry_type": "rule",
                                                "title": "Magic Law",
                                                "summary": "Draft notes.",
                                                "sections": [],
                                            }
                                        ),
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "The draft write is not stable yet. We should continue refining this setup detail before review.",
                    }
                }
            ]
        }


class _StreamToolLLM:
    def __init__(self) -> None:
        self.round = 0

    async def chat_completion_stream(self, request):
        self.round += 1
        if self.round == 1:
            yield 'data: {"type":"text_delta","delta":"Checking draft."}\n\n'
            yield (
                'data: {"type":"tool_call","tool_calls":[{"id":"call_stage_entry","function":{"name":"rp_setup__setup.stage_entry.write","arguments":"{\\"entry_type\\":\\"rule\\",\\"title\\":\\"Magic Law\\",\\"sections\\":[]}"}}]}\n\n'
            )
            yield 'data: {"type":"done"}\n\n'
            return
        yield 'data: {"type":"text_delta","delta":"Applied and finalized."}\n\n'
        yield 'data: {"type":"done"}\n\n'


class _StreamUsageLLM:
    async def chat_completion_stream(self, request):
        yield 'data: {"type":"text_delta","delta":"Streaming answer."}\n\n'
        yield (
            'data: {"type":"usage","prompt_tokens":11,"completion_tokens":7,'
            '"total_tokens":18,"prompt_tokens_details":{"cached_tokens":3},'
            '"completion_tokens_details":{"reasoning_tokens":2},'
            '"cache_read_input_tokens":3}\n\n'
        )
        yield 'data: {"type":"done"}\n\n'


class _StreamProviderErrorLLM:
    async def chat_completion_stream(self, request):
        yield 'data: {"type":"text_delta","delta":"This must stay private."}\n\n'
        yield (
            'data: {"type":"error","error":{"message":"provider stack trace: token leaked","type":"ProviderSchemaError","raw_provider_delta":{"debug":true}}}\n\n'
        )


class _StreamMalformedPayloadLLM:
    async def chat_completion_stream(self, request):
        yield 'data: {"type":"text_delta","delta":"Buffered before parse failure."}\n\n'
        yield 'data: {"type":"text_delta","delta":\n\n'


class _StreamSafeProviderErrorLLM:
    async def chat_completion_stream(self, request):
        yield 'data: {"type":"error","error":{"message":"upstream timeout","type":"TimeoutError"}}\n\n'


class _StreamPrivateDeltaThenTextLLM:
    async def chat_completion_stream(self, request):
        yield 'data: {"type":"raw_provider_delta","delta":"private-token","debug":{"trace":"hidden"}}\n\n'
        yield 'data: {"type":"text_delta","delta":"Public answer."}\n\n'
        yield 'data: {"type":"done"}\n\n'


class _ProviderExceptionLLM:
    async def chat_completion(self, request):
        raise RuntimeError("provider schema explosion")


class _PseudoToolCodeTextLLM:
    def __init__(self) -> None:
        self.round = 0

    async def chat_completion(self, request):
        self.round += 1
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": (
                            "tool_code print(default_api.rp_setup__setup.stage_entry.write("
                            '{"entry_type":"rule","title":"Magic Law"}))'
                        ),
                    }
                }
            ]
        }


class _PseudoToolCodeStreamLLM:
    def __init__(self) -> None:
        self.round = 0

    async def chat_completion_stream(self, request):
        self.round += 1
        yield 'data: {"type":"text_delta","delta":"tool_code print(default_api."}\n\n'
        yield (
            'data: {"type":"text_delta","delta":"rp_setup__setup.stage_entry.write({\\"entry_type\\":\\"rule\\",\\"title\\":\\"Magic Law\\"}))"}\n\n'
        )
        yield 'data: {"type":"done"}\n\n'


@pytest.mark.asyncio
async def test_runtime_executor_returns_direct_answer_without_tools():
    tool_executor = _FakeToolExecutor()
    executor = RpAgentRuntimeExecutor(tool_executor_factory=lambda _: tool_executor)

    result = await executor.run(_turn_input(), _profile(), llm_service=_TextOnlyLLM())

    assert result.status == "completed"
    assert result.finish_reason == "completed_text"
    assert result.assistant_text == "Direct setup answer."
    assert result.tool_invocations == []
    assert result.tool_results == []
    assert result.structured_payload["working_digest"] is not None
    assert result.structured_payload["tool_outcomes"] == []
    assert result.structured_payload["compact_summary"] is None
    assert result.structured_payload["context_report"] is None
    assert result.structured_payload["continue_reason"] is None
    assert result.structured_payload["loop_trace"]
    assert (
        result.structured_payload["loop_trace"][-1]["decision"]["finish_reason"]
        == "completed_text"
    )


@pytest.mark.asyncio
async def test_runtime_executor_places_runtime_overlay_after_system_prompt():
    tool_executor = _FakeToolExecutor()
    llm = _RecordingTextOnlyLLM()
    executor = RpAgentRuntimeExecutor(tool_executor_factory=lambda _: tool_executor)

    turn_input = _turn_input(
        conversation_messages=[
            {"role": "user", "content": "governed history user"},
            {"role": "assistant", "content": "governed history assistant"},
        ],
        context_bundle={
            "cognitive_state_summary": {
                "current_step": "story_config",
                "invalidated": False,
                "invalidation_reasons": [],
                "open_questions": ["Which preset should be used?"],
            },
            "working_digest": {
                "current_goal": "Clarify setup",
                "next_focus": "Lock runtime preset",
                "open_questions": ["Which preset should be used?"],
                "draft_refs": ["draft:story_config"],
                "commit_blockers": [],
            },
            "tool_outcomes": [
                {
                    "tool_name": "rp_setup__setup.memory.open",
                    "success": True,
                    "summary": "Read current story config draft",
                    "updated_refs": ["draft:story_config"],
                    "relevance": "read",
                    "recorded_at": "2026-04-28T00:00:00Z",
                }
            ],
            "compact_summary": {
                "source_fingerprint": "fp-1",
                "source_message_count": 4,
                "summary_lines": ["Earlier draft discussion was compacted."],
                "open_threads": ["Need exact preset name."],
                "draft_refs": ["draft:story_config"],
                "recovery_hints": [
                    {
                        "ref": "draft:story_config",
                        "reason": "Need exact draft detail.",
                    }
                ],
            },
        },
    )
    turn_input.metadata["context_report"] = {
        "context_profile": "compact",
        "profile_reasons": [
            "history_count_threshold",
            "user_edit_threshold",
        ],
        "raw_history_count": 10,
        "raw_history_chars": 3200,
        "estimated_input_tokens": 900,
        "previous_prompt_tokens": 1200,
        "previous_total_tokens": 1600,
        "user_edit_delta_count": 3,
        "prior_stage_handoff_count": 1,
        "raw_history_limit": 4,
        "kept_history_count": 4,
        "compacted_history_count": 6,
        "retained_tool_outcome_count": 1,
        "summary_strategy": "deterministic_prefix_summary",
        "summary_action": "rebuilt",
        "summary_line_count": 1,
    }
    turn_input.metadata["context_pipeline"] = {
        "final_request_message_order": [
            "stable_system_prompt",
            "runtime_overlay_system_message",
            "governed_history",
            "current_user",
        ]
    }

    result = await executor.run(
        turn_input,
        _profile(),
        llm_service=llm,
    )

    assert result.status == "completed"
    assert llm.requests
    request = llm.requests[0]
    messages = request.messages
    assert messages[0].role == "system"
    assert messages[0].content == "You are SetupAgent."
    assert messages[1].role == "system"
    assert "Runtime turn state follows as JSON." in messages[1].content
    assert "setup.memory.open" in messages[1].content
    assert "creative design involving named setup objects" in messages[1].content
    assert "ground that object through setup.memory.search" in messages[1].content
    assert "context_packet" not in messages[1].content
    assert "context_report" not in messages[1].content
    assert messages[2].role == "user"
    assert messages[2].content == "governed history user"
    assert messages[3].role == "assistant"
    assert messages[3].content == "governed history assistant"
    assert messages[-1].role == "user"
    assert messages[-1].content == "Please help with setup."
    assert result.structured_payload["context_report"]["context_profile"] == "compact"
    assert result.structured_payload["context_pipeline"][
        "final_request_message_order"
    ] == [
        "stable_system_prompt",
        "runtime_overlay_system_message",
        "governed_history",
        "current_user",
    ]


@pytest.mark.asyncio
async def test_runtime_executor_does_not_guard_text_for_memory_recall():
    tool_executor = _FakeToolExecutor()
    llm = _CompactExactDetailTextThenOpenLLM()
    executor = RpAgentRuntimeExecutor(tool_executor_factory=lambda _: tool_executor)

    result = await executor.run(
        _turn_input(
            user_prompt="之前写入草稿的 magic-law 完整内容是什么？",
            context_bundle=_compact_exact_detail_context(),
            tool_scope=["setup.memory.open"],
        ),
        _profile(),
        llm_service=llm,
    )

    assert result.status == "completed"
    assert llm.round == 1
    assert result.assistant_text == "The exact magic-law content is permits only."
    assert tool_executor.calls == []
    assert "action_expectation" not in result.structured_payload
    assert result.structured_payload["completion_guard"]["reason"] == (
        "terminal_output_allowed"
    )
    assert "required_draft_ref_read_missing" not in result.warnings


@pytest.mark.asyncio
async def test_runtime_executor_allows_voluntary_memory_open_observation():
    tool_executor = _FakeToolExecutor(results=[_draft_ref_open_result()])
    llm = _CompactExactDetailReadThenTextLLM()
    executor = RpAgentRuntimeExecutor(tool_executor_factory=lambda _: tool_executor)

    result = await executor.run(
        _turn_input(
            user_prompt="之前写入草稿的 magic-law 完整内容是什么？",
            context_bundle=_compact_exact_detail_context(),
            tool_scope=["setup.memory.open"],
        ),
        _profile(),
        llm_service=llm,
    )

    assert result.status == "completed"
    assert llm.round == 2
    assert len(tool_executor.calls) == 1
    assert tool_executor.calls[0][0].tool_name == "rp_setup__setup.memory.open"
    assert "action_expectation" not in result.structured_payload
    assert "required_draft_ref_read_missing" not in result.warnings
    trace = result.structured_payload["loop_trace"]
    assert all("action_expectation" not in item["decision"] for item in trace)


@pytest.mark.asyncio
async def test_runtime_executor_does_not_block_mutation_for_memory_expectation():
    tool_executor = _FakeToolExecutor(
        results=[
            RuntimeToolResult(
                call_id="call_write",
                tool_name="rp_setup__setup.stage_entry.write",
                success=True,
                content_text='{"success": true}',
                error_code=None,
            ),
            _draft_ref_open_result(),
        ]
    )
    llm = _CompactExactDetailMutateThenOpenLLM()
    executor = RpAgentRuntimeExecutor(tool_executor_factory=lambda _: tool_executor)

    result = await executor.run(
        _turn_input(
            user_prompt="之前写入草稿的 magic-law 完整内容是什么？",
            context_bundle=_compact_exact_detail_context(),
            tool_scope=["setup.stage_entry.write", "setup.memory.open"],
        ),
        _profile(),
        llm_service=llm,
    )

    assert result.status == "completed"
    assert llm.round == 3
    executed_tool_names = [call.tool_name for call, _ in tool_executor.calls]
    assert executed_tool_names == [
        "rp_setup__setup.stage_entry.write",
        "rp_setup__setup.memory.open",
    ]
    assert "required_draft_ref_read_missing" not in result.warnings
    trace = result.structured_payload["loop_trace"]
    assert all("action_expectation" not in item["decision"] for item in trace)


@pytest.mark.asyncio
async def test_runtime_executor_does_not_block_mixed_memory_open_and_mutation_batch():
    tool_executor = _FakeToolExecutor(results=[_draft_ref_open_result()])
    llm = _CompactExactDetailMixedOpenMutationLLM()
    executor = RpAgentRuntimeExecutor(tool_executor_factory=lambda _: tool_executor)

    result = await executor.run(
        _turn_input(
            user_prompt="之前写入草稿的 magic-law 完整内容是什么？",
            context_bundle=_compact_exact_detail_context(),
            tool_scope=["setup.stage_entry.write", "setup.memory.open"],
        ),
        _profile(),
        llm_service=llm,
    )

    assert result.status == "completed"
    assert llm.round == 2
    executed_tool_names = [call.tool_name for call, _ in tool_executor.calls]
    assert executed_tool_names == [
        "rp_setup__setup.memory.open",
        "rp_setup__setup.stage_entry.write",
    ]
    trace = result.structured_payload["loop_trace"]
    assert all("action_expectation" not in item["decision"] for item in trace)


@pytest.mark.asyncio
async def test_runtime_executor_allows_voluntary_search_then_open_for_exact_session_detail():
    tool_executor = _FakeToolExecutor(
        results=[_memory_search_result(), _memory_open_result()]
    )
    llm = _ExactSessionDetailSearchThenOpenLLM()
    executor = RpAgentRuntimeExecutor(tool_executor_factory=lambda _: tool_executor)

    result = await executor.run(
        _turn_input(
            user_prompt="Mira 与 Violet Harbor 接头时的确切暗号是什么？她把暗号藏在哪个物件里？",
            context_bundle=_compact_exact_detail_context_without_refs(),
            tool_scope=["setup.memory.search", "setup.memory.open"],
        ),
        _profile(),
        llm_service=llm,
    )

    assert result.status == "completed"
    assert llm.round == 3
    assert result.assistant_text == (
        "暗号是 blue lanterns unlock the tidewall lattice，"
        "藏在 copper astrolabe named Lumen Key 里。"
    )
    executed_tool_names = [call.tool_name for call, _ in tool_executor.calls]
    assert executed_tool_names == [
        "rp_setup__setup.memory.search",
        "rp_setup__setup.memory.open",
    ]
    trace = result.structured_payload["loop_trace"]
    assert all("action_expectation" not in item["decision"] for item in trace)
    assert "action_expectation" not in result.structured_payload


@pytest.mark.asyncio
async def test_runtime_executor_only_sends_tool_schemas_from_turn_scope():
    tool_executor = _FakeToolExecutor()
    llm = _RecordingTextOnlyLLM()
    executor = RpAgentRuntimeExecutor(tool_executor_factory=lambda _: tool_executor)

    result = await executor.run(
        _turn_input(
            tool_scope=[
                "setup.memory.search",
                "setup.stage_entry.write",
            ]
        ),
        _profile(),
        llm_service=llm,
    )

    assert result.status == "completed"
    assert llm.requests
    tool_names = [item["function"]["name"] for item in (llm.requests[0].tools or [])]
    assert tool_names == [
        "rp_setup__setup.memory.search",
        "rp_setup__setup.stage_entry.write",
    ]


@pytest.mark.asyncio
async def test_runtime_executor_does_not_send_removed_draft_tool_schemas():
    tool_executor = _FakeToolExecutor()
    llm = _RecordingTextOnlyLLM()
    executor = RpAgentRuntimeExecutor(tool_executor_factory=lambda _: tool_executor)

    result = await executor.run(
        _turn_input(
            tool_scope=[
                "setup.stage_entry.write",
                "setup.memory.search",
            ]
        ),
        _profile(),
        llm_service=llm,
    )

    assert result.status == "completed"
    tool_names = {item["function"]["name"] for item in (llm.requests[0].tools or [])}
    assert tool_names == {
        "rp_setup__setup.stage_entry.write",
        "rp_setup__setup.memory.search",
    }
    assert "rp_setup__setup.truth.write" not in tool_names
    assert "rp_setup__setup.patch.story_config" not in tool_names


@pytest.mark.asyncio
async def test_runtime_executor_removes_inherited_strict_for_non_strict_models():
    tool_executor = _FakeToolExecutor()
    llm = _RecordingTextOnlyLLM()
    driver = _RuntimeRunDriver(
        llm_service=llm,
        tool_executor=tool_executor,
        profile=_profile(),
    )
    turn_input = _turn_input(tool_scope=["setup.stage_entry.write"])
    turn_input.metadata["model_name"] = "gemini-2.5-flash"
    inherited_strict_tool = _tool_definition("rp_setup__setup.stage_entry.write")
    inherited_strict_tool["function"]["strict"] = True

    tools = driver._model_facing_tool_definitions(
        [inherited_strict_tool],
        turn_input=turn_input,
    )

    function = tools[0]["function"]
    assert function["strict"] is True


@pytest.mark.asyncio
async def test_runtime_executor_ignores_removed_truth_write_schema_mode():
    tool_executor = _FakeToolExecutor()
    llm = _RecordingTextOnlyLLM()
    driver = _RuntimeRunDriver(
        llm_service=llm,
        tool_executor=tool_executor,
        profile=_profile(),
    )
    turn_input = _turn_input(tool_scope=["setup.stage_entry.write"])
    turn_input.metadata["capability_plan"] = {
        "model_schema_modes": {"setup.truth.write": "provider_default"}
    }
    inherited_tool = _tool_definition("rp_setup__setup.stage_entry.write")
    inherited_parameters = inherited_tool["function"]["parameters"]

    tools = driver._model_facing_tool_definitions(
        [inherited_tool],
        turn_input=turn_input,
    )

    function = tools[0]["function"]
    assert function["parameters"] == inherited_parameters
    assert function["description"] == "test tool"


@pytest.mark.asyncio
async def test_runtime_executor_filters_pseudo_tool_code_text_from_final_answer():
    tool_executor = _FakeToolExecutor()
    executor = RpAgentRuntimeExecutor(tool_executor_factory=lambda _: tool_executor)

    result = await executor.run(
        _turn_input(
            tool_scope=["setup.stage_entry.write"],
            context_bundle={
                "current_step": "foundation",
                "current_stage": "world_background",
                "context_packet": {
                    "workspace_id": "workspace-1",
                    "current_step": "foundation",
                    "current_stage": "world_background",
                    "current_draft_snapshot": {},
                },
            },
        ),
        _profile().model_copy(update={"max_rounds": 2}),
        llm_service=_PseudoToolCodeTextLLM(),
    )

    assert result.status == "failed"
    assert result.finish_reason == "repair_obligation_unfulfilled"
    assert result.assistant_text == ""
    assert result.tool_invocations == []
    assert "pseudo_tool_call_text_filtered" in result.warnings
    assert (
        result.structured_payload["completion_guard"]["reason"]
        == "pseudo_tool_call_text_emitted"
    )
    assert (
        result.structured_payload["output_inspection"]["classification"]
        == "pseudo_tool_text"
    )


@pytest.mark.asyncio
async def test_runtime_executor_stream_filters_pseudo_tool_code_text_from_typed_events():
    tool_executor = _FakeToolExecutor()
    executor = RpAgentRuntimeExecutor(tool_executor_factory=lambda _: tool_executor)

    chunks = []
    async for chunk in executor.run_stream(
        _turn_input(
            stream=True,
            tool_scope=["setup.stage_entry.write"],
            context_bundle={
                "current_step": "foundation",
                "current_stage": "world_background",
                "context_packet": {
                    "workspace_id": "workspace-1",
                    "current_step": "foundation",
                    "current_stage": "world_background",
                    "current_draft_snapshot": {},
                },
            },
        ),
        _profile().model_copy(update={"max_rounds": 2}),
        llm_service=_PseudoToolCodeStreamLLM(),
    ):
        chunks.append(chunk)

    payloads = [json.loads(chunk[6:]) for chunk in chunks if chunk.startswith("data: ")]
    assert all(payload.get("type") != "text_delta" for payload in payloads)
    assert payloads[-2]["type"] == "error"
    assert payloads[-1] == {"type": "done"}
    assert executor.last_result is not None
    assert executor.last_result.finish_reason == "repair_obligation_unfulfilled"


@pytest.mark.asyncio
async def test_runtime_executor_executes_tool_and_continues():
    tool_executor = _FakeToolExecutor(
        results=[
            RuntimeToolResult(
                call_id="call_stage_entry",
                tool_name="rp_setup__setup.stage_entry.write",
                success=True,
                content_text='{"success": true}',
                error_code=None,
            )
        ]
    )
    executor = RpAgentRuntimeExecutor(tool_executor_factory=lambda _: tool_executor)

    result = await executor.run(
        _turn_input(), _profile(), llm_service=_ToolThenTextLLM()
    )

    assert result.status == "completed"
    assert result.assistant_text == "Tool succeeded and I finished the answer."
    assert len(result.tool_invocations) == 1
    assert len(result.tool_results) == 1
    assert tool_executor.calls[0][0].tool_name == "rp_setup__setup.stage_entry.write"
    assert result.structured_payload["tool_outcomes"][0]["tool_name"] == (
        "rp_setup__setup.stage_entry.write"
    )
    assert result.structured_payload["tool_outcomes"][0]["success"] is True
    continue_reasons = [
        item["decision"].get("continue_reason")
        for item in result.structured_payload["loop_trace"]
    ]
    assert "tool_call_batch_pending" in continue_reasons
    assert "tool_result_follow_up" in continue_reasons


@pytest.mark.asyncio
async def test_runtime_executor_allows_one_schema_repair_retry():
    tool_executor = _FakeToolExecutor(
        results=[
            RuntimeToolResult(
                call_id="call_1",
                tool_name="rp_setup__setup.stage_entry.write",
                success=False,
                content_text='{"error":{"code":"schema_validation_failed"}}',
                error_code="SCHEMA_VALIDATION_FAILED",
            ),
            RuntimeToolResult(
                call_id="call_2",
                tool_name="rp_setup__setup.stage_entry.write",
                success=True,
                content_text='{"success": true}',
                error_code=None,
            ),
        ]
    )
    executor = RpAgentRuntimeExecutor(tool_executor_factory=lambda _: tool_executor)

    result = await executor.run(
        _turn_input(), _profile(), llm_service=_SchemaRepairLLM()
    )

    assert result.status == "completed"
    assert result.assistant_text == "Corrected the tool call and finished."
    assert len(result.tool_results) == 2
    assert "tool_schema_validation_retry" in result.warnings


@pytest.mark.asyncio
async def test_runtime_executor_switches_to_ask_user_when_failure_requires_user_input():
    tool_executor = _FakeToolExecutor(
        results=[
            RuntimeToolResult(
                call_id="call_ask_user",
                tool_name="rp_setup__setup.stage_entry.write",
                success=False,
                content_text=json.dumps(
                    {
                        "code": "schema_validation_failed",
                        "message": "Need user-selected style rules before patching story config.",
                        "details": {
                            "ask_user": True,
                            "errors": [{"type": "missing", "loc": ["title"]}],
                        },
                    }
                ),
                error_code="SCHEMA_VALIDATION_FAILED",
            )
        ]
    )
    executor = RpAgentRuntimeExecutor(tool_executor_factory=lambda _: tool_executor)

    result = await executor.run(
        _turn_input(),
        _profile(),
        llm_service=_AskUserAfterFailureLLM(),
    )

    assert result.status == "completed"
    assert result.finish_reason == "awaiting_user_input"
    assert "Which style rules do you want" in result.assistant_text
    assert "tool_failure_requires_user_input" in result.warnings


@pytest.mark.asyncio
async def test_runtime_executor_blocks_false_success_when_repair_obligation_is_unmet():
    tool_executor = _FakeToolExecutor(
        results=[
            RuntimeToolResult(
                call_id="call_bad_patch",
                tool_name="rp_setup__setup.stage_entry.write",
                success=False,
                content_text=json.dumps(
                    {
                        "code": "schema_validation_failed",
                        "message": "Stage entry title is missing.",
                        "details": {
                            "errors": [{"type": "missing", "loc": ["title"]}],
                        },
                    }
                ),
                error_code="SCHEMA_VALIDATION_FAILED",
            )
        ]
    )
    executor = RpAgentRuntimeExecutor(tool_executor_factory=lambda _: tool_executor)

    result = await executor.run(
        _turn_input(),
        _profile(),
        llm_service=_ExplainInsteadOfRepairLLM(),
    )

    assert result.status == "failed"
    assert result.finish_reason == "repair_obligation_unfulfilled"
    assert result.error is not None
    assert (
        result.structured_payload["completion_guard"]["reason"]
        == "repair_obligation_unresolved"
    )
    assert any(
        item["decision"].get("continue_reason") == "completion_guard_retry"
        for item in result.structured_payload["loop_trace"]
    )


@pytest.mark.asyncio
async def test_runtime_executor_blocks_repeated_question_at_initial_text_completion():
    tool_executor = _FakeToolExecutor()
    profile = _profile().model_copy(update={"max_rounds": 2})
    executor = RpAgentRuntimeExecutor(tool_executor_factory=lambda _: tool_executor)

    result = await executor.run(
        _turn_input(
            conversation_messages=[
                {
                    "role": "assistant",
                    "content": "Which style rules do you want me to lock in for this draft?",
                }
            ],
            context_bundle={
                "working_digest": {
                    "open_questions": ["Need the exact style rules."],
                }
            },
        ),
        profile,
        llm_service=_RepeatedQuestionLLM(),
    )

    assert result.status == "failed"
    assert result.finish_reason == "max_rounds_exceeded"
    assert (
        result.structured_payload["completion_guard"]["reason"]
        == "repeated_question_without_progress"
    )
    trace = result.structured_payload["loop_trace"]
    assert any(
        item["decision_site"] == "inspect_model_output"
        and item["action"]["kind"] == "assistant_text"
        and item["action"]["assistant_text_kind"] == "question"
        and item["decision"].get("continue_reason") == "completion_guard_retry"
        for item in trace
    )
    assert any(
        item["decision_site"] == "reflect_if_needed"
        and item["decision"].get("continue_reason") == "reflection_retry"
        for item in trace
    )
    assert trace[-1]["decision_site"] == "finalize_failure"
    assert trace[-1]["decision"]["finish_reason"] == "max_rounds_exceeded"


@pytest.mark.asyncio
async def test_runtime_executor_prefers_runtime_max_rounds_over_langgraph_recursion_limit():
    tool_executor = _FakeToolExecutor()
    executor = RpAgentRuntimeExecutor(tool_executor_factory=lambda _: tool_executor)

    result = await executor.run(
        _turn_input(
            conversation_messages=[
                {
                    "role": "assistant",
                    "content": "Which style rules do you want me to lock in for this draft?",
                }
            ],
            context_bundle={
                "working_digest": {
                    "open_questions": ["Need the exact style rules."],
                }
            },
        ),
        _profile().model_copy(update={"max_rounds": 5}),
        llm_service=_RepeatedQuestionLLM(),
    )

    assert result.status == "failed"
    assert result.finish_reason == "max_rounds_exceeded"
    assert result.error is not None
    assert result.error["type"] == "max_rounds_exceeded"


@pytest.mark.asyncio
async def test_runtime_executor_keeps_non_commit_tool_failure_in_recovery_semantics():
    tool_executor = _FakeToolExecutor(
        results=[
            RuntimeToolResult(
                call_id="call_stage_entry",
                tool_name="rp_setup__setup.stage_entry.write",
                success=False,
                content_text=json.dumps(
                    {
                        "code": "setup_tool_failed",
                        "message": "Stage entry write could not be applied yet.",
                        "details": {
                            "repair_strategy": "continue_discussion",
                        },
                    }
                ),
                error_code="SETUP_TOOL_FAILED",
            )
        ]
    )
    executor = RpAgentRuntimeExecutor(tool_executor_factory=lambda _: tool_executor)

    result = await executor.run(
        _turn_input(tool_scope=["setup.stage_entry.write"]),
        _profile(),
        llm_service=_StageEntryFailureThenDiscussionLLM(),
    )

    assert result.status == "completed"
    assert result.finish_reason == "continue_discussion"
    assert (
        result.structured_payload["turn_goal"]["goal_type"]
        == "recover_from_tool_failure"
    )
    assert (
        result.structured_payload["last_failure"]["failure_category"]
        == "continue_discussion"
    )
    assert result.structured_payload["repair_route"] == "continue_discussion"


@pytest.mark.asyncio
async def test_runtime_executor_fails_fast_on_unknown_tool():
    tool_executor = _FakeToolExecutor(
        results=[
            RuntimeToolResult(
                call_id="call_unknown",
                tool_name="rp_setup__setup.unknown",
                success=False,
                content_text='{"error":{"code":"unknown_tool"}}',
                error_code="UNKNOWN_TOOL",
            )
        ]
    )
    executor = RpAgentRuntimeExecutor(tool_executor_factory=lambda _: tool_executor)

    result = await executor.run(
        _turn_input(), _profile(), llm_service=_UnknownToolLLM()
    )

    assert result.status == "failed"
    assert result.finish_reason == "tool_error_unrecoverable"
    assert result.error is not None
    assert result.error["type"] == "tool_error_unrecoverable"


@pytest.mark.asyncio
async def test_runtime_executor_stream_preserves_typed_event_order():
    tool_executor = _FakeToolExecutor(
        results=[
            RuntimeToolResult(
                call_id="call_stage_entry",
                tool_name="rp_setup__setup.stage_entry.write",
                success=True,
                content_text='{"success": true}',
                error_code=None,
            )
        ]
    )
    executor = RpAgentRuntimeExecutor(tool_executor_factory=lambda _: tool_executor)

    chunks = []
    async for chunk in executor.run_stream(
        _turn_input(stream=True),
        _profile(),
        llm_service=_StreamToolLLM(),
    ):
        chunks.append(chunk)

    payloads = [json.loads(chunk[6:]) for chunk in chunks if chunk.startswith("data: ")]
    assert [payload["type"] for payload in payloads] == [
        "tool_call",
        "tool_started",
        "tool_result",
        "text_delta",
        "done",
    ]
    assert all(payload.get("delta") != "Checking draft." for payload in payloads)


def test_typed_sse_event_adapter_filters_private_provider_fields():
    event = RuntimeEvent(
        type="text_delta",
        run_id="run-1",
        sequence_no=1,
        payload={
            "delta": "public",
            "raw_provider_delta": {"token": "hidden"},
            "debug": {"trace": "hidden"},
        },
    )
    private_event = RuntimeEvent(
        type="raw_provider_delta",
        run_id="run-1",
        sequence_no=2,
        payload={"delta": "hidden"},
    )
    thinking_event = RuntimeEvent(
        type="thinking_delta",
        run_id="run-1",
        sequence_no=3,
        payload={"delta": "thinking", "raw_provider_delta": {"token": "hidden"}},
    )

    assert TypedSseEventAdapter.to_payload(event) == {
        "type": "text_delta",
        "delta": "public",
    }
    assert TypedSseEventAdapter.to_payload(thinking_event) == {
        "type": "thinking_delta",
        "delta": "thinking",
    }
    assert TypedSseEventAdapter.to_payload(private_event) is None
    assert TypedSseEventAdapter.to_sse_line(private_event) is None


@pytest.mark.asyncio
async def test_runtime_executor_stream_keeps_raw_provider_deltas_private():
    tool_executor = _FakeToolExecutor()
    executor = RpAgentRuntimeExecutor(tool_executor_factory=lambda _: tool_executor)

    chunks = []
    async for chunk in executor.run_stream(
        _turn_input(stream=True, tool_scope=[]),
        _profile(),
        llm_service=_StreamPrivateDeltaThenTextLLM(),
    ):
        chunks.append(chunk)

    payloads = [json.loads(chunk[6:]) for chunk in chunks if chunk.startswith("data: ")]
    assert [payload["type"] for payload in payloads] == ["text_delta", "done"]
    assert payloads[0]["delta"] == "Public answer."
    assert all("raw_provider_delta" not in payload for payload in payloads)
    assert executor.last_result is not None
    diagnostics = executor.last_result.structured_payload["model_gateway_diagnostics"]
    assert diagnostics["failure_layer"] == "model_gateway"
    assert (
        diagnostics["private_details"]["private_events"][0]["type"]
        == "raw_provider_delta"
    )


@pytest.mark.asyncio
async def test_runtime_executor_stream_provider_error_is_gateway_failure_private_diagnostics():
    tool_executor = _FakeToolExecutor()
    executor = RpAgentRuntimeExecutor(tool_executor_factory=lambda _: tool_executor)

    chunks = []
    async for chunk in executor.run_stream(
        _turn_input(stream=True, tool_scope=[]),
        _profile(),
        llm_service=_StreamProviderErrorLLM(),
    ):
        chunks.append(chunk)

    payloads = [json.loads(chunk[6:]) for chunk in chunks if chunk.startswith("data: ")]
    assert [payload["type"] for payload in payloads] == ["error", "done"]
    public_error = payloads[0]["error"]
    assert public_error == {
        "message": "Model provider request failed.",
        "type": "model_gateway_failed",
        "code": "provider_stream_error",
        "failure_layer": "model_gateway",
    }
    assert "provider stack trace" not in json.dumps(payloads, ensure_ascii=False)
    assert "This must stay private." not in json.dumps(payloads, ensure_ascii=False)
    assert executor.last_result is not None
    assert executor.last_result.status == "failed"
    assert executor.last_result.finish_reason == "upstream_error"
    diagnostics = executor.last_result.structured_payload["model_gateway_diagnostics"]
    assert diagnostics["failure_layer"] == "model_gateway"
    assert diagnostics["failure_kind"] == "provider_stream_error"
    assert "provider stack trace" in diagnostics["message"]
    assert (
        executor.last_result.structured_payload["output_inspection"]["classification"]
        == "provider_schema_error"
    )


@pytest.mark.asyncio
async def test_runtime_executor_stream_parse_error_is_gateway_failure():
    tool_executor = _FakeToolExecutor()
    executor = RpAgentRuntimeExecutor(tool_executor_factory=lambda _: tool_executor)

    chunks = []
    async for chunk in executor.run_stream(
        _turn_input(stream=True, tool_scope=[]),
        _profile(),
        llm_service=_StreamMalformedPayloadLLM(),
    ):
        chunks.append(chunk)

    payloads = [json.loads(chunk[6:]) for chunk in chunks if chunk.startswith("data: ")]
    assert [payload["type"] for payload in payloads] == ["error", "done"]
    public_error = payloads[0]["error"]
    assert public_error["type"] == "model_gateway_failed"
    assert public_error["code"] == "provider_stream_parse_error"
    assert public_error["failure_layer"] == "model_gateway"
    assert "Buffered before parse failure." not in json.dumps(
        payloads, ensure_ascii=False
    )
    assert executor.last_result is not None
    assert executor.last_result.status == "failed"
    assert executor.last_result.finish_reason == "upstream_error"
    diagnostics = executor.last_result.structured_payload["model_gateway_diagnostics"]
    assert diagnostics["failure_kind"] == "provider_stream_parse_error"
    assert (
        executor.last_result.structured_payload["output_inspection"]["classification"]
        == "provider_schema_error"
    )


@pytest.mark.asyncio
async def test_runtime_executor_stream_provider_error_can_expose_safe_summary():
    tool_executor = _FakeToolExecutor()
    executor = RpAgentRuntimeExecutor(tool_executor_factory=lambda _: tool_executor)

    chunks = []
    async for chunk in executor.run_stream(
        _turn_input(stream=True, tool_scope=[]),
        _profile(),
        llm_service=_StreamSafeProviderErrorLLM(),
    ):
        chunks.append(chunk)

    payloads = [json.loads(chunk[6:]) for chunk in chunks if chunk.startswith("data: ")]
    assert payloads[0]["type"] == "error"
    assert payloads[0]["error"]["message"] == "upstream timeout"
    assert payloads[0]["error"]["failure_layer"] == "model_gateway"


@pytest.mark.asyncio
async def test_runtime_executor_non_stream_provider_exception_is_gateway_failure():
    tool_executor = _FakeToolExecutor()
    executor = RpAgentRuntimeExecutor(tool_executor_factory=lambda _: tool_executor)

    result = await executor.run(
        _turn_input(tool_scope=[]),
        _profile(),
        llm_service=_ProviderExceptionLLM(),
    )

    assert result.status == "failed"
    assert result.finish_reason == "upstream_error"
    assert result.error is not None
    assert result.error["type"] == "model_gateway_failed"
    assert result.error["failure_layer"] == "model_gateway"
    diagnostics = result.structured_payload["model_gateway_diagnostics"]
    assert diagnostics["failure_kind"] == "provider_request_error"
    assert diagnostics["provider_error_type"] == "RuntimeError"
    assert "provider schema explosion" in diagnostics["message"]


@pytest.mark.asyncio
async def test_runtime_executor_emits_langfuse_generation_and_tool_observations(
    monkeypatch,
):
    fake_langfuse = _FakeLangfuseService()
    monkeypatch.setattr(
        "rp.agent_runtime.executor.get_langfuse_service",
        lambda: fake_langfuse,
    )
    tool_executor = _FakeToolExecutor()
    executor = RpAgentRuntimeExecutor(tool_executor_factory=lambda _: tool_executor)

    result = await executor.run(
        _turn_input(tool_scope=["setup.stage_entry.write"]),
        _profile(),
        llm_service=_ToolThenTextLLM(),
    )

    assert result.finish_reason == "completed_text"
    assert any(
        item["kind"] == "observation_enter" and item["name"] == "rp.runtime.model_call"
        for item in fake_langfuse.events
    )
    assert any(
        item["kind"] == "observation_enter"
        and item["name"] == "rp.runtime.tool:rp_setup__setup.stage_entry.write"
        for item in fake_langfuse.events
    )


@pytest.mark.asyncio
async def test_runtime_executor_stream_preserves_usage_in_latest_response():
    tool_executor = _FakeToolExecutor()
    executor = RpAgentRuntimeExecutor(tool_executor_factory=lambda _: tool_executor)

    chunks = []
    async for chunk in executor.run_stream(
        _turn_input(stream=True, tool_scope=[]),
        _profile(),
        llm_service=_StreamUsageLLM(),
    ):
        chunks.append(chunk)

    assert chunks
    assert executor.last_result is not None
    latest_response = executor.last_result.structured_payload["latest_response"]
    assert latest_response["usage"]["prompt_tokens"] == 11
    assert latest_response["usage"]["completion_tokens"] == 7
    assert latest_response["usage"]["total_tokens"] == 18
    assert latest_response["usage"]["prompt_tokens_details"]["cached_tokens"] == 3
    assert (
        latest_response["usage"]["completion_tokens_details"]["reasoning_tokens"] == 2
    )
    assert latest_response["usage"]["cache_read_input_tokens"] == 3
    assert executor.last_result.structured_payload["loop_trace"]
