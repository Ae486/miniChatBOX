"""Tests for setup.memory tool registration, dispatch, and scope."""

from __future__ import annotations

import json

import pytest

from rp.agent_runtime.prompts.setup_agent import tool_description_text
from rp.agent_runtime.profiles import build_setup_agent_capability_plan
from rp.models.setup_drafts import (
    SetupDraftEntry,
    SetupDraftSection,
    SetupStageDraftBlock,
)
from rp.models.setup_stage import SetupStageId
from rp.models.setup_workspace import StoryMode
from rp.services.setup_agent_runtime_state_service import SetupAgentRuntimeStateService
from rp.services.setup_context_builder import SetupContextBuilder
from rp.services.setup_workspace_service import SetupWorkspaceService
from rp.tools.setup_tool_provider import SetupToolProvider
from rp.tools.setup_tool_registry import SETUP_TOOL_REGISTRY


def _seed_provider(retrieval_session):
    workspace_service = SetupWorkspaceService(retrieval_session)
    context_builder = SetupContextBuilder(workspace_service)
    runtime_state_service = SetupAgentRuntimeStateService(retrieval_session)
    provider = SetupToolProvider(
        workspace_service=workspace_service,
        context_builder=context_builder,
        runtime_state_service=runtime_state_service,
    )
    workspace = workspace_service.create_workspace(
        story_id="story-setup-memory-tools-1",
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
                    summary="Moonlit forest cities.",
                    tags=["forest"],
                    sections=[
                        SetupDraftSection(
                            section_id="summary",
                            title="Summary",
                            kind="text",
                            content={"text": "Moonlit forest cities."},
                            retrieval_role="summary",
                        )
                    ],
                )
            ],
        ),
    )
    return provider, workspace


def test_setup_session_memory_tools_are_registered_and_scoped():
    registry_names = [entry.name for entry in SETUP_TOOL_REGISTRY]
    plan = build_setup_agent_capability_plan(
        "foundation",
        current_stage="world_background",
    )

    assert "setup.memory.search" in registry_names
    assert "setup.memory.open" in registry_names
    assert "setup.memory.read_refs" in registry_names
    assert "setup.memory.search" in plan.runtime_allowlist
    assert "setup.memory.open" in plan.runtime_allowlist
    assert "setup.memory.read_refs" in plan.runtime_allowlist
    assert "setup.memory.search" in plan.active_tool_names
    assert "setup.memory.open" in plan.active_tool_names
    assert "setup.memory.read_refs" in plan.active_tool_names
    search_registration = next(
        entry for entry in SETUP_TOOL_REGISTRY if entry.name == "setup.memory.search"
    )
    open_registration = next(
        entry for entry in SETUP_TOOL_REGISTRY if entry.name == "setup.memory.open"
    )
    assert (
        "named setup objects before creative design" in search_registration.description
    )
    assert "not fact content" in search_registration.description
    assert "named setup object" in open_registration.description
    assert "clean structured fact content" in open_registration.description
    assert search_registration.description == tool_description_text(
        "setup.memory.search"
    )
    assert open_registration.description == tool_description_text("setup.memory.open")
    assert "memory.get_state" not in plan.runtime_allowlist
    assert "memory.get_summary" not in plan.runtime_allowlist
    assert "memory.search_recall" not in plan.runtime_allowlist
    assert "memory.search_archival" not in plan.runtime_allowlist
    assert "memory.list_versions" not in plan.runtime_allowlist
    assert "memory.read_provenance" not in plan.runtime_allowlist


@pytest.mark.asyncio
async def test_setup_memory_search_tool_returns_refs_without_payload(
    retrieval_session,
):
    provider, workspace = _seed_provider(retrieval_session)

    result = await provider.call_tool(
        tool_name="setup.memory.search",
        arguments={
            "workspace_id": workspace.workspace_id,
            "query": "Moonlit",
            "limit": 5,
        },
    )

    payload = json.loads(result["content"])
    assert result["success"] is True
    assert payload["items"][0]["ref"] == "stage:world_background:race_elf"
    assert payload["items"][0]["scope"] == "entry"
    assert payload["items"][0]["navigation_summary"] == "Moonlit forest cities."
    assert "payload" not in payload["items"][0]
    assert "source_kind" not in payload["items"][0]


@pytest.mark.asyncio
async def test_setup_memory_open_tool_opens_entry_then_section(
    retrieval_session,
):
    provider, workspace = _seed_provider(retrieval_session)

    entry_result = await provider.call_tool(
        tool_name="setup.memory.open",
        arguments={
            "workspace_id": workspace.workspace_id,
            "ref": "stage:world_background:race_elf",
        },
    )
    entry_payload = json.loads(entry_result["content"])

    assert entry_result["success"] is True
    assert entry_payload["result_type"] == "index"
    assert entry_payload["sections"][0]["ref"] == (
        "stage:world_background:race_elf:summary"
    )
    assert entry_payload.get("content") is None

    section_result = await provider.call_tool(
        tool_name="setup.memory.open",
        arguments={
            "workspace_id": workspace.workspace_id,
            "ref": "stage:world_background:race_elf:summary",
        },
    )
    section_payload = json.loads(section_result["content"])

    assert section_result["success"] is True
    assert section_payload["result_type"] == "content"
    assert section_payload["content"] == {
        "type": "text",
        "title": "Summary",
        "text": "Moonlit forest cities.",
    }
    assert section_payload.get("sections") is None


@pytest.mark.asyncio
async def test_setup_memory_read_refs_tool_dispatches_to_current_sources(
    retrieval_session,
):
    provider, workspace = _seed_provider(retrieval_session)

    result = await provider.call_tool(
        tool_name="setup.memory.read_refs",
        arguments={
            "workspace_id": workspace.workspace_id,
            "refs": [
                "stage:world_background:race_elf:summary",
                "stage:world_background:missing",
            ],
            "detail": "full",
            "max_chars": 1200,
        },
    )

    payload = json.loads(result["content"])
    found = {item["ref"]: item for item in payload["items"] if item["found"]}
    assert result["success"] is True
    assert payload["success"] is False
    assert payload["missing_refs"] == ["stage:world_background:missing"]
    assert found["stage:world_background:race_elf:summary"]["payload"]["content"] == {
        "text": "Moonlit forest cities."
    }
