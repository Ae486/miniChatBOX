"""Opt-in real-model behavior evals for setup tool-call paths.

The current write-path smoke exercises ``setup.stage_entry.write``. Compact
recovery verifies ``setup.memory.open`` over a stage-section ref, seeded through
the workspace service instead of the deleted legacy truth-write agent tool.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from models.model_registry import ModelCapabilityProfile, ModelRegistryEntry
from models.provider_registry import ProviderRegistryEntry
from rp.models.setup_agent import SetupAgentDialogueMessage, SetupAgentTurnRequest
from rp.models.setup_drafts import (
    SetupDraftEntry,
    SetupDraftSection,
    SetupStageDraftBlock,
)
from rp.models.setup_stage import SetupStageId
from rp.models.setup_workspace import SetupStepId, StoryMode
from rp.runtime.rp_runtime_factory import RpRuntimeFactory
from rp.services.setup_workspace_service import SetupWorkspaceService
from services.model_registry import get_model_registry_service
from services.provider_registry import get_provider_registry_service


_EXACT_MAGIC_LAW_DETAIL = "Public spellcasting requires guild permits."
_OLD_RAW_HISTORY_MARKER = "OLD_RAW_HISTORY_OUTSIDE_SUMMARY_WINDOW"
_VIOLET_HARBOR_SECRET_PHRASE = "blue lanterns unlock the tidewall lattice"
_MIRA_SECRET_HOLDER = "copper astrolabe named Lumen Key"
_RUN_ENV = "CHATBOX_RUN_REAL_SETUP_AGENT_COMPACT_EVAL"
_STAGE_ENTRY_RUN_ENV = "CHATBOX_RUN_REAL_SETUP_AGENT_STAGE_ENTRY_EVAL"
_MEMORY_BEHAVIOR_RUN_ENV = "CHATBOX_RUN_REAL_SETUP_AGENT_MEMORY_BEHAVIOR_EVAL"
_SETUP_PROVIDER_ID = "rp_setup"
LEGACY_WORLD_BACKGROUND_TOOL_NAMES = (
    "setup.world_background.list_entries",
    "setup.world_background.read_entry",
    "setup.world_background.write_entry",
    "setup.world_background.edit_entry",
    "setup.world_background.delete_entry",
)


def _load_local_registry_model_config(
    *,
    model_id: str,
    provider_id: str,
) -> dict[str, str] | None:
    storage_dir = Path(
        os.environ.get(
            "CHATBOX_REAL_SETUP_EVAL_REGISTRY_DIR",
            Path(__file__).resolve().parents[2] / "storage",
        )
    )
    models_path = storage_dir / "models.json"
    providers_path = storage_dir / "providers.json"
    if not models_path.exists() or not providers_path.exists():
        return None
    models = json.loads(models_path.read_text(encoding="utf-8"))
    providers = json.loads(providers_path.read_text(encoding="utf-8"))
    if not isinstance(models, list) or not isinstance(providers, list):
        return None

    model_entry = next(
        (item for item in models if isinstance(item, dict) and item.get("id") == model_id),
        None,
    )
    if model_entry is None:
        return None
    resolved_provider_id = provider_id or str(model_entry.get("provider_id") or "")
    provider_entry = next(
        (
            item
            for item in providers
            if isinstance(item, dict) and item.get("id") == resolved_provider_id
        ),
        None,
    )
    if provider_entry is None or not provider_entry.get("is_enabled", True):
        return None
    api_key = str(provider_entry.get("api_key") or "")
    model_name = str(model_entry.get("model_name") or "")
    if not api_key or not model_name:
        return None
    return {
        "provider_id": resolved_provider_id,
        "model_id": model_id,
        "provider_type": str(provider_entry.get("type") or "openai"),
        "api_url": str(provider_entry.get("api_url") or ""),
        "api_key": api_key,
        "model_name": model_name,
        "seed_registry": "true",
    }


def _require_real_setup_model_config(*, run_env: str = _RUN_ENV) -> dict[str, str]:
    if os.environ.get(run_env, "").strip().lower() not in {"1", "true", "yes"}:
        pytest.skip(
            f"Set {run_env}=1 plus CHATBOX_REAL_SETUP_EVAL_MODEL_NAME and "
            "CHATBOX_REAL_SETUP_EVAL_API_KEY, or set "
            "CHATBOX_REAL_SETUP_EVAL_MODEL_ID to use an existing local registry "
            "entry, to run the paid/network real-model eval."
        )

    model_id = os.environ.get("CHATBOX_REAL_SETUP_EVAL_MODEL_ID", "").strip()
    provider_id = os.environ.get("CHATBOX_REAL_SETUP_EVAL_PROVIDER_ID", "").strip()
    model_name = os.environ.get("CHATBOX_REAL_SETUP_EVAL_MODEL_NAME", "").strip()
    api_key = os.environ.get("CHATBOX_REAL_SETUP_EVAL_API_KEY", "").strip()
    if model_name and api_key:
        return {
            "provider_id": provider_id or "provider-real-setup-compact-eval",
            "model_id": model_id or "model-real-setup-compact-eval",
            "provider_type": os.environ.get(
                "CHATBOX_REAL_SETUP_EVAL_PROVIDER_TYPE",
                "openai",
            ).strip(),
            "api_url": os.environ.get(
                "CHATBOX_REAL_SETUP_EVAL_API_URL",
                "https://api.openai.com/v1",
            ).strip(),
            "api_key": api_key,
            "model_name": model_name,
            "seed_registry": "true",
        }

    if model_id:
        model_entry = get_model_registry_service().get_entry(model_id)
        if model_entry is None:
            seeded_config = _load_local_registry_model_config(
                model_id=model_id,
                provider_id=provider_id,
            )
            if seeded_config is not None:
                return seeded_config
            pytest.skip(f"Configured real setup model_id was not found: {model_id}")
        resolved_provider_id = provider_id or model_entry.provider_id
        provider_entry = get_provider_registry_service().get_entry(resolved_provider_id)
        if provider_entry is None or not provider_entry.is_enabled:
            pytest.skip(
                f"Configured real setup provider was not available: {resolved_provider_id}"
            )
        if not provider_entry.api_key or not model_entry.model_name:
            seeded_config = _load_local_registry_model_config(
                model_id=model_id,
                provider_id=resolved_provider_id,
            )
            if seeded_config is not None:
                return seeded_config
            pytest.skip(
                "Configured real setup provider/model exists but lacks api_key "
                f"or model_name: {model_id}"
            )
        return {
            "provider_id": resolved_provider_id,
            "model_id": model_id,
            "provider_type": provider_entry.type,
            "api_url": provider_entry.api_url or "",
            "api_key": provider_entry.api_key,
            "model_name": model_entry.model_name,
            "seed_registry": "true",
        }

    pytest.skip("No configured real setup model/provider env was provided.")
    raise AssertionError("unreachable after pytest.skip")


def _message_text(messages: list[Any]) -> str:
    return "\n".join(str(getattr(message, "content", "") or "") for message in messages)


def _response_payload(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json")
    return {}


def _assistant_content_from_response(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return str(message.get("content") or "")


def _tool_payload(tool: Any) -> dict[str, Any]:
    if isinstance(tool, dict):
        return tool
    if hasattr(tool, "model_dump"):
        return tool.model_dump(mode="json")
    return {}


def _setup_tool_name_variants(raw_tool_name: str) -> set[str]:
    normalized_raw_tool_name = raw_tool_name.replace(".", "_")
    return {
        raw_tool_name,
        f"{_SETUP_PROVIDER_ID}__{raw_tool_name}",
        f"{_SETUP_PROVIDER_ID}__{normalized_raw_tool_name}",
    }


def _matches_setup_tool_name(tool_name: Any, raw_tool_name: str) -> bool:
    return str(tool_name or "") in _setup_tool_name_variants(raw_tool_name)


def _setup_tool_function(tools: list[Any], raw_tool_name: str) -> dict[str, Any]:
    for tool in tools:
        payload = _tool_payload(tool)
        function = payload.get("function") or {}
        if _matches_setup_tool_name(function.get("name"), raw_tool_name):
            return function
    return {}


def _setup_tool_schema(tools: list[Any], raw_tool_name: str) -> dict[str, Any]:
    function = _setup_tool_function(tools, raw_tool_name)
    return function.get("parameters") or {}


def _memory_open_tool_schema(tools: list[Any]) -> dict[str, Any]:
    return _setup_tool_schema(tools, "setup.memory.open")


def _stage_entry_write_tool_schema(tools: list[Any]) -> dict[str, Any]:
    return _setup_tool_schema(tools, "setup.stage_entry.write")


def _contains_key(payload: Any, key: str) -> bool:
    if isinstance(payload, dict):
        return key in payload or any(_contains_key(value, key) for value in payload.values())
    if isinstance(payload, list):
        return any(_contains_key(item, key) for item in payload)
    return False


@pytest.mark.parametrize(
    "function_name",
    ["rp_setup__setup_memory_open", "rp_setup__setup.memory.open"],
)
def test_real_model_harness_finds_memory_open_schema_by_qualified_name_variant(
    function_name,
):
    schema = {"type": "object", "properties": {"ref": {"type": "string"}}}

    assert _memory_open_tool_schema(
        [{"function": {"name": function_name, "parameters": schema}}]
    ) == schema


@pytest.mark.parametrize(
    "function_name",
    ["rp_setup__setup_stage_entry_write", "rp_setup__setup.stage_entry.write"],
)
def test_real_model_harness_finds_stage_entry_schema_by_qualified_name_variant(
    function_name,
):
    schema = {"type": "object", "properties": {"entry_type": {"type": "string"}}}

    assert _stage_entry_write_tool_schema(
        [{"function": {"name": function_name, "parameters": schema}}]
    ) == schema


class _CapturingRealSetupLLMService:
    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate
        self.requests: list[Any] = []
        self.responses: list[dict[str, Any]] = []
        self.recovery_phase = False
        self.recovery_prompt_checked = False
        self.recovery_first_response: dict[str, Any] | None = None
        self.stage_entry_phase = False
        self.stage_entry_prompt_checked = False
        self.stage_entry_first_response: dict[str, Any] | None = None
        self.memory_behavior_phase = False
        self.memory_behavior_prompt_checked = False
        self.memory_behavior_first_response: dict[str, Any] | None = None
        self.memory_behavior_first_tool_names: list[str] = []

    def start_recovery_phase(self) -> None:
        self.recovery_phase = True

    def start_stage_entry_phase(self) -> None:
        self.stage_entry_phase = True

    def start_memory_behavior_phase(self) -> None:
        self.memory_behavior_phase = True

    async def chat_completion_stream(self, request: Any):
        raise AssertionError("stream path is not used by this behavior eval")

    async def chat_completion(self, request: Any):
        self.requests.append(request)
        is_recovery_pre_tool_request = False
        if self.stage_entry_phase and not self.stage_entry_prompt_checked:
            self._assert_stage_entry_round_prompt_visibility(request)
            self.stage_entry_prompt_checked = True
            is_stage_entry_pre_tool_request = True
        else:
            is_stage_entry_pre_tool_request = False
        if self.recovery_phase and not self.recovery_prompt_checked:
            if self._is_setup_compact_prompt_request(request):
                is_recovery_pre_tool_request = False
            else:
                self._assert_recovery_round_prompt_visibility(request)
                self.recovery_prompt_checked = True
                is_recovery_pre_tool_request = True
        if self.memory_behavior_phase and not self.memory_behavior_prompt_checked:
            self._assert_memory_behavior_round_prompt_visibility(request)
            self.memory_behavior_first_tool_names = _tool_names_from_request(request)
            self.memory_behavior_prompt_checked = True
            is_memory_behavior_pre_tool_request = True
        else:
            is_memory_behavior_pre_tool_request = False
        response = await self._delegate.chat_completion(request)
        payload = _response_payload(response)
        self.responses.append(payload)
        if is_recovery_pre_tool_request:
            self.recovery_first_response = payload
        if is_stage_entry_pre_tool_request:
            self.stage_entry_first_response = payload
        if is_memory_behavior_pre_tool_request:
            self.memory_behavior_first_response = payload
        return response

    @staticmethod
    def _assert_recovery_round_prompt_visibility(request: Any) -> None:
        visible_text = _message_text(list(request.messages or []))

        assert _EXACT_MAGIC_LAW_DETAIL not in visible_text
        assert _OLD_RAW_HISTORY_MARKER not in visible_text
        assert "stage:world_background:magic_law:summary" in visible_text
        assert "recovery_hints" in visible_text
        assert "setup.memory.open" in visible_text
        open_schema = _memory_open_tool_schema(list(request.tools or []))
        assert open_schema, (
            "setup.memory.open tool schema is not visible under raw or "
            "normalized qualified names"
        )
        assert "ref" in (open_schema.get("required") or [])

    @staticmethod
    def _is_setup_compact_prompt_request(request: Any) -> bool:
        if list(getattr(request, "tools", None) or []):
            return False
        visible_text = _message_text(list(request.messages or []))
        return (
            "dropped_current_step_messages" in visible_text
            and "dropped_current_step_fingerprint" in visible_text
        )

    @staticmethod
    def _assert_stage_entry_round_prompt_visibility(request: Any) -> None:
        tools = list(request.tools or [])
        stage_entry_write_schema = _stage_entry_write_tool_schema(tools)
        assert stage_entry_write_schema, (
            "setup.stage_entry.write schema is not visible under raw or normalized "
            "qualified names"
        )
        for tool_name in (
            "setup.stage_entry.list",
            "setup.stage_entry.read",
            "setup.stage_entry.edit",
            "setup.stage_entry.delete",
        ):
            assert _setup_tool_schema(tools, tool_name), (
                f"{tool_name} schema is not visible to the current-stage turn"
            )
        for tool_name in LEGACY_WORLD_BACKGROUND_TOOL_NAMES:
            assert not _setup_tool_schema(tools, tool_name), (
                f"{tool_name} must not be exposed after legacy cleanup"
            )
        assert not _setup_tool_schema(tools, "setup.truth.write"), (
            "setup.truth.write must not be the main visible write tool for "
            "world_background, character_design, or plot_blueprint"
        )
        properties = stage_entry_write_schema.get("properties") or {}
        assert "stage_id" not in properties
        assert "stage_id" not in (stage_entry_write_schema.get("required") or [])

    @staticmethod
    def _assert_memory_behavior_round_prompt_visibility(request: Any) -> None:
        visible_text = _message_text(list(request.messages or []))
        assert _VIOLET_HARBOR_SECRET_PHRASE not in visible_text
        assert _MIRA_SECRET_HOLDER not in visible_text
        assert "Violet Harbor" in visible_text
        assert "Mira" in visible_text
        for tool_name in ("setup.memory.search", "setup.memory.open"):
            assert _setup_tool_schema(list(request.tools or []), tool_name), (
                f"{tool_name} schema is not visible to the real-model probe"
            )


def _seed_real_model_registry(config: dict[str, str]) -> None:
    if config.get("seed_registry") == "false":
        return
    provider_service = get_provider_registry_service()
    model_service = get_model_registry_service()
    provider_service.upsert_entry(
        ProviderRegistryEntry(
            id=config["provider_id"],
            name="Real Setup Compact Eval Provider",
            type=config["provider_type"],
            api_key=config["api_key"],
            api_url=config["api_url"],
            custom_headers={},
            is_enabled=True,
            description="Opt-in real-model provider for compact recovery eval",
        )
    )
    model_service.upsert_entry(
        ModelRegistryEntry(
            id=config["model_id"],
            provider_id=config["provider_id"],
            model_name=config["model_name"],
            display_name="Real Setup Compact Eval Model",
            capabilities=["text", "tool"],
            capability_source="user_declared",
            capability_profile=ModelCapabilityProfile(
                known=True,
                provider_supported=True,
                capability_source="user_declared",
                transport_provider_type=config["provider_type"],
                mode="chat",
                supports_function_calling=True,
                supports_tool_choice=True,
                supported_openai_params=["tools", "tool_choice"],
                recommended_capabilities=["text", "tool"],
            ),
            is_enabled=True,
            description="Opt-in real-model setup eval model",
        )
    )


def _compact_recovery_history() -> list[SetupAgentDialogueMessage]:
    return [
        SetupAgentDialogueMessage(
            role="user" if index % 2 == 0 else "assistant",
            content=(
                f"{_OLD_RAW_HISTORY_MARKER} compact candidate {index}"
                if index == 0
                else f"compact candidate history {index}"
            ),
        )
        for index in range(12)
    ]


def _tool_invocations_named(result: Any, suffix: str) -> list[Any]:
    return [
        item
        for item in result.tool_invocations
        if _matches_setup_tool_name(item.tool_name, suffix)
    ]


def _tool_results_named(result: Any, suffix: str) -> list[Any]:
    return [
        item
        for item in result.tool_results
        if _matches_setup_tool_name(item.tool_name, suffix)
    ]


def test_real_model_harness_matches_stage_entry_tool_trace_name_variants():
    result = SimpleNamespace(
        tool_invocations=[
            SimpleNamespace(tool_name="rp_setup__setup_stage_entry_write")
        ],
        tool_results=[SimpleNamespace(tool_name="rp_setup__setup.stage_entry.write")],
    )

    assert _tool_invocations_named(result, "setup.stage_entry.write")
    assert _tool_results_named(result, "setup.stage_entry.write")


def _stage_magic_law_summary(workspace: Any) -> str | None:
    block = workspace.draft_blocks.get(SetupStageId.WORLD_BACKGROUND.value)
    if block is None:
        return None
    for entry in block.entries:
        if entry.entry_id != "magic_law":
            continue
        for section in entry.sections:
            if section.section_id != "summary":
                continue
            content = section.content
            if isinstance(content, dict) and content.get("text"):
                return str(content["text"])
    return None


def _real_recovery_prompt() -> str:
    return (
        "For the current blueprint decision I need the exact current draft detail "
        "for stage:world_background:magic_law:summary. Do not infer it from memory or compact summary. "
        "If the exact detail is not visible in the current prompt, call "
        "setup.memory.open with ref='stage:world_background:magic_law:summary'. "
        "After readback, answer with 'Recovered detail:' followed by the exact "
        "summary returned by the tool result."
    )


def _real_stage_entry_prompt(workspace_id: str) -> str:
    return (
        "Call setup.stage_entry.write exactly once to create one editable entry in "
        "the current setup stage draft. The current stage is supplied by the "
        "backend; do not include stage_id anywhere in the tool arguments. "
        f"The workspace_id is {workspace_id}. Use entry_type='location', "
        "title='霓虹湾', summary='霓虹湾是由潮汐墙、盐雾灯塔和走私航线维持秩序的海港城。', "
        "sections=[{'title':'地理与秩序','text':'潮汐墙保护内港，盐雾灯塔决定夜航窗口，走私航线由码头行会默许。'}], "
        "aliases=['Neon Bay'], tags=['harbor','smoke']. Do not ask the user "
        "to commit. After the tool result is successful, do not call "
        "any more tools; reply exactly: Stage entry write complete: 霓虹湾."
    )


def _real_memory_behavior_prompt() -> str:
    return (
        "我们要继续设计冲突场景。请回答：Mira 与 Violet Harbor 接头时的"
        "确切暗号是什么？她把暗号藏在哪个物件里？"
    )


def _memory_behavior_summary_history() -> list[SetupAgentDialogueMessage]:
    return [
        SetupAgentDialogueMessage(
            role="assistant",
            content=(
                "阶段摘要：此前已经完成 Violet Harbor 的港口秩序设定，以及 "
                "Mira 与港口线人的接头机制。摘要只保留主题，不展开暗号和"
                "物件细节。"
            ),
        )
    ]


def _seed_memory_behavior_probe_drafts(
    workspace_service: SetupWorkspaceService,
    *,
    workspace_id: str,
) -> None:
    workspace_service.patch_stage_draft(
        workspace_id=workspace_id,
        stage_id=SetupStageId.WORLD_BACKGROUND,
        draft=SetupStageDraftBlock(
            stage_id=SetupStageId.WORLD_BACKGROUND,
            entries=[
                SetupDraftEntry(
                    entry_id="violet_harbor",
                    entry_type="location",
                    semantic_path="world_background.location.violet_harbor",
                    title="Violet Harbor",
                    summary="Violet Harbor controls its tidewall through hidden signals.",
                    tags=["harbor", "tidewall", "signal"],
                    sections=[
                        SetupDraftSection(
                            section_id="emergency_signal",
                            title="Emergency Signal",
                            kind="text",
                            content={
                                "text": (
                                    "Mira's contact phrase is "
                                    f"'{_VIOLET_HARBOR_SECRET_PHRASE}'."
                                )
                            },
                            retrieval_role="detail",
                            tags=["signal", "contact"],
                        )
                    ],
                )
            ],
        ),
    )
    workspace_service.patch_stage_draft(
        workspace_id=workspace_id,
        stage_id=SetupStageId.CHARACTER_DESIGN,
        draft=SetupStageDraftBlock(
            stage_id=SetupStageId.CHARACTER_DESIGN,
            entries=[
                SetupDraftEntry(
                    entry_id="mira",
                    entry_type="character",
                    semantic_path="character_design.character.mira",
                    title="Mira",
                    summary="Mira is a harbor courier who hides operational secrets.",
                    tags=["courier", "harbor", "secret"],
                    sections=[
                        SetupDraftSection(
                            section_id="secret_holder",
                            title="Secret Holder",
                            kind="text",
                            content={
                                "text": (
                                    "Mira hides the contact phrase inside a "
                                    f"{_MIRA_SECRET_HOLDER}."
                                )
                            },
                            retrieval_role="detail",
                            tags=["secret", "prop"],
                        )
                    ],
                )
            ],
        ),
    )


def _memory_behavior_failure_diagnostic(
    *,
    llm: _CapturingRealSetupLLMService,
    result: Any,
) -> str:
    first_request_text = _message_text(list(llm.requests[0].messages or []))
    memory_search_visible = any(
        _matches_setup_tool_name(name, "setup.memory.search")
        for name in llm.memory_behavior_first_tool_names
    )
    memory_open_visible = any(
        _matches_setup_tool_name(name, "setup.memory.open")
        for name in llm.memory_behavior_first_tool_names
    )
    first_response = json.dumps(
        llm.memory_behavior_first_response or {},
        ensure_ascii=False,
    )
    failure_class = (
        "2.tool_description_or_memory_integration_gap"
        if not memory_search_visible or not memory_open_visible
        else "1.agent_reasoning_gap_or_prompt_policy_gap"
    )
    return (
        f"classification={failure_class}; "
        f"memory_search_visible={memory_search_visible}; "
        f"memory_open_visible={memory_open_visible}; "
        f"prompt_mentions_memory={'setup.memory' in first_request_text}; "
        f"assistant_text={result.assistant_text!r}; "
        f"first_response={first_response}"
    )


def _is_deepseek_config(config: dict[str, str]) -> bool:
    haystack = " ".join(
        [
            config.get("provider_id", ""),
            config.get("provider_type", ""),
            config.get("api_url", ""),
            config.get("model_id", ""),
            config.get("model_name", ""),
        ]
    ).lower()
    return "deepseek" in haystack


def _tool_names_from_request(request: Any) -> list[str]:
    names: list[str] = []
    for tool in list(getattr(request, "tools", None) or []):
        function = (_tool_payload(tool).get("function") or {})
        name = function.get("name")
        if name:
            names.append(str(name))
    return names


@pytest.mark.asyncio
async def test_real_deepseek_stage_entry_write_uses_current_stage_tool_chain(
    retrieval_session,
    monkeypatch,
):
    config = _require_real_setup_model_config(run_env=_STAGE_ENTRY_RUN_ENV)
    if not _is_deepseek_config(config):
        pytest.skip(
            "Configured real setup model is not DeepSeek. Set "
            "CHATBOX_REAL_SETUP_EVAL_MODEL_NAME or MODEL_ID to a DeepSeek model "
            "for this smoke."
        )
    get_provider_registry_service.cache_clear()
    import services.model_registry as model_registry_module
    from services.litellm_service import get_litellm_service as real_litellm_service

    model_registry_module._model_registry_service = None
    _seed_real_model_registry(config)

    llm = _CapturingRealSetupLLMService(real_litellm_service())
    llm.start_stage_entry_phase()
    monkeypatch.setattr(
        "rp.services.setup_agent_execution_service.get_litellm_service",
        lambda: llm,
    )

    workspace_service = SetupWorkspaceService(retrieval_session)
    workspace = workspace_service.create_workspace(
        story_id="story-real-deepseek-stage-entry",
        mode=StoryMode.LONGFORM,
    )
    service = RpRuntimeFactory(retrieval_session).build_setup_agent_execution_service()

    await service.run_turn(
        SetupAgentTurnRequest(
            workspace_id=workspace.workspace_id,
            model_id=config["model_id"],
            provider_id=config["provider_id"],
            target_step=SetupStepId.FOUNDATION,
            target_stage=SetupStageId.WORLD_BACKGROUND,
            user_prompt=_real_stage_entry_prompt(workspace.workspace_id),
        )
    )
    result = service.last_runtime_result

    assert result is not None, "real model/provider path did not return a runtime result"
    assert llm.requests, "provider/gateway/API-key failure: real model was not called"
    assert llm.responses, "provider/gateway failure: no model response was captured"
    assert llm.stage_entry_prompt_checked is True
    assert llm.stage_entry_first_response is not None
    first_request_tool_names = _tool_names_from_request(llm.requests[0])
    assert any(
        _matches_setup_tool_name(name, "setup.stage_entry.write")
        for name in first_request_tool_names
    ), "setup.stage_entry.write schema was not visible in the model request"
    exposed_world_background_tools = [
        name
        for name in first_request_tool_names
        if any(
            _matches_setup_tool_name(name, raw_tool_name)
            for raw_tool_name in LEGACY_WORLD_BACKGROUND_TOOL_NAMES
        )
    ]
    assert not exposed_world_background_tools, (
        "removed setup.world_background.* tools were exposed to the model: "
        + json.dumps(exposed_world_background_tools, ensure_ascii=False)
    )
    assert not any(
        _matches_setup_tool_name(name, "setup.truth.write")
        for name in first_request_tool_names
    ), "setup.truth.write was exposed as the main write tool for this stage"

    invocations = _tool_invocations_named(result, "setup.stage_entry.write")
    assert invocations, (
        "model did not issue a real setup.stage_entry.write tool call. "
        f"assistant_text={result.assistant_text!r}; "
        f"first_response={json.dumps(llm.stage_entry_first_response, ensure_ascii=False)}"
    )
    first_call_args = invocations[0].arguments
    assert not _contains_key(first_call_args, "stage_id"), (
        "model supplied stage_id even though current stage is backend-owned"
    )
    assert first_call_args["workspace_id"] == workspace.workspace_id
    assert first_call_args["entry_type"]
    assert first_call_args["title"]

    tool_results = _tool_results_named(result, "setup.stage_entry.write")
    assert tool_results, "backend did not return a setup.stage_entry.write tool result"
    assert any(item.success for item in tool_results), (
        "backend setup.stage_entry.write execution failed: "
        + json.dumps(
            [
                {
                    "success": item.success,
                    "error_code": item.error_code,
                    "content_text": item.content_text,
                }
                for item in tool_results
            ],
            ensure_ascii=False,
        )
    )

    refreshed = workspace_service.get_workspace(workspace.workspace_id)
    assert refreshed is not None
    world_block = refreshed.draft_blocks.get(SetupStageId.WORLD_BACKGROUND.value)
    assert world_block is not None, "backend did not create world_background draft block"
    assert len(world_block.entries) == 1
    entry = world_block.entries[0]
    assert entry.title == "霓虹湾"
    assert entry.semantic_path.startswith("world_background.")
    assert entry.sections
    assert SetupStageId.CHARACTER_DESIGN.value not in refreshed.draft_blocks
    assert SetupStageId.PLOT_BLUEPRINT.value not in refreshed.draft_blocks
    assert result.status == "completed"


@pytest.mark.asyncio
async def test_real_model_memory_behavior_probe_recovers_hidden_stage_detail(
    retrieval_session,
    monkeypatch,
):
    config = _require_real_setup_model_config(run_env=_MEMORY_BEHAVIOR_RUN_ENV)
    get_provider_registry_service.cache_clear()
    import services.model_registry as model_registry_module
    from services.litellm_service import get_litellm_service as real_litellm_service

    model_registry_module._model_registry_service = None
    _seed_real_model_registry(config)

    llm = _CapturingRealSetupLLMService(real_litellm_service())
    llm.start_memory_behavior_phase()
    monkeypatch.setattr(
        "rp.services.setup_agent_execution_service.get_litellm_service",
        lambda: llm,
    )

    workspace_service = SetupWorkspaceService(retrieval_session)
    workspace = workspace_service.create_workspace(
        story_id="story-real-memory-behavior-probe",
        mode=StoryMode.LONGFORM,
    )
    _seed_memory_behavior_probe_drafts(
        workspace_service,
        workspace_id=workspace.workspace_id,
    )
    service = RpRuntimeFactory(retrieval_session).build_setup_agent_execution_service()

    await service.run_turn(
        SetupAgentTurnRequest(
            workspace_id=workspace.workspace_id,
            model_id=config["model_id"],
            provider_id=config["provider_id"],
            target_stage=SetupStageId.PLOT_BLUEPRINT,
            user_prompt=_real_memory_behavior_prompt(),
            history=_memory_behavior_summary_history(),
            user_edit_delta_ids=[],
        )
    )
    result = service.last_runtime_result

    assert result is not None, "real model/provider path did not return a runtime result"
    assert llm.requests, "provider/gateway/API-key failure: real model was not called"
    assert llm.responses, "provider/gateway failure: no model response was captured"
    assert llm.memory_behavior_prompt_checked is True
    assert llm.memory_behavior_first_response is not None
    assert _VIOLET_HARBOR_SECRET_PHRASE not in _assistant_content_from_response(
        llm.memory_behavior_first_response
    )
    assert _MIRA_SECRET_HOLDER not in _assistant_content_from_response(
        llm.memory_behavior_first_response
    )

    diagnostic = _memory_behavior_failure_diagnostic(llm=llm, result=result)
    search_invocations = _tool_invocations_named(result, "setup.memory.search")
    open_invocations = _tool_invocations_named(result, "setup.memory.open")
    assert search_invocations, (
        "model did not proactively call setup.memory.search for a previously "
        f"established but hidden exact detail; {diagnostic}"
    )
    assert open_invocations, (
        "model found or had access to memory tooling but did not call "
        f"setup.memory.open before answering exact details; {diagnostic}"
    )

    open_results = _tool_results_named(result, "setup.memory.open")
    assert open_results and any(item.success for item in open_results), (
        "setup.memory.open was called but did not return a successful result; "
        + diagnostic
    )
    combined_open_payload = "\n".join(item.content_text for item in open_results)
    assert _VIOLET_HARBOR_SECRET_PHRASE in combined_open_payload
    assert _MIRA_SECRET_HOLDER in combined_open_payload
    assert _VIOLET_HARBOR_SECRET_PHRASE in result.assistant_text
    assert _MIRA_SECRET_HOLDER in result.assistant_text
    assert result.status == "completed"


@pytest.mark.asyncio
async def test_real_model_compact_recovery_reads_draft_ref_before_using_detail(
    retrieval_session,
    monkeypatch,
):
    config = _require_real_setup_model_config()
    get_provider_registry_service.cache_clear()
    import services.model_registry as model_registry_module
    from services.litellm_service import get_litellm_service as real_litellm_service

    model_registry_module._model_registry_service = None
    _seed_real_model_registry(config)

    llm = _CapturingRealSetupLLMService(real_litellm_service())
    monkeypatch.setattr(
        "rp.services.setup_agent_execution_service.get_litellm_service",
        lambda: llm,
    )

    workspace_service = SetupWorkspaceService(retrieval_session)
    workspace = workspace_service.create_workspace(
        story_id="story-real-compact-recovery",
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
                            content={"text": _EXACT_MAGIC_LAW_DETAIL},
                            retrieval_role="summary",
                        )
                    ],
                )
            ],
        ),
    )
    service = RpRuntimeFactory(retrieval_session).build_setup_agent_execution_service()

    workspace_after_write = workspace_service.get_workspace(workspace.workspace_id)
    assert workspace_after_write is not None
    assert _stage_magic_law_summary(workspace_after_write) == (
        _EXACT_MAGIC_LAW_DETAIL
    )

    llm.start_recovery_phase()
    await service.run_turn(
        SetupAgentTurnRequest(
            workspace_id=workspace.workspace_id,
            model_id=config["model_id"],
            provider_id=config["provider_id"],
            target_step=SetupStepId.LONGFORM_BLUEPRINT,
            user_prompt=_real_recovery_prompt(),
            history=_compact_recovery_history(),
            user_edit_delta_ids=["delta-1", "delta-2", "delta-3"],
        )
    )
    recovery_result = service.last_runtime_result

    assert recovery_result is not None
    assert llm.requests, "real model was not called"
    assert llm.responses, "real model response was not captured"
    assert llm.recovery_prompt_checked is True
    assert llm.recovery_first_response is not None
    assert _EXACT_MAGIC_LAW_DETAIL not in _assistant_content_from_response(
        llm.recovery_first_response
    )
    assert recovery_result.status == "completed"

    open_invocations = _tool_invocations_named(
        recovery_result,
        "setup.memory.open",
    )
    assert open_invocations, "setup.memory.open was not called"
    assert open_invocations[0].arguments["ref"] == (
        "stage:world_background:magic_law:summary"
    )

    open_results = _tool_results_named(recovery_result, "setup.memory.open")
    assert open_results and open_results[0].success is True
    assert _EXACT_MAGIC_LAW_DETAIL in open_results[0].content_text
    assert _EXACT_MAGIC_LAW_DETAIL in recovery_result.assistant_text

    context_report = recovery_result.structured_payload["context_report"]
    compact_summary = recovery_result.structured_payload["compact_summary"]
    assert context_report["context_profile"] == "compact"
    assert context_report["compacted_history_count"] > 0
    assert _EXACT_MAGIC_LAW_DETAIL not in json.dumps(
        compact_summary,
        sort_keys=True,
    )
