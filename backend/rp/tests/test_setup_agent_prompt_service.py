"""Unit tests for SetupAgentPromptService SkillPack integration."""

from __future__ import annotations

import pytest

from rp.agent_runtime.contracts import SetupCapabilityGuidanceFragment
from rp.agent_runtime.prompts.setup_agent import (
    SYSTEM_PROMPT_TEMPLATE,
    capability_guidance_text,
    compact_prompt_system_prompt,
    stage_objective_text,
)
from rp.agent_runtime.profiles import (
    SETUP_AGENT_VISIBLE_TOOLS,
    build_setup_agent_capability_plan,
)
from rp.models.setup_handoff import SetupContextPacket
from rp.models.setup_stage import SetupStageId
from rp.models.setup_workspace import SetupStepId, StoryMode
from rp.services.setup_agent_prompt_service import SetupAgentPromptService


def _packet(
    *,
    current_stage: SetupStageId | None,
    current_step: SetupStepId = SetupStepId.FOUNDATION,
) -> SetupContextPacket:
    return SetupContextPacket(
        workspace_id="ws-test",
        current_step=current_step.value,
        current_stage=current_stage,
        user_prompt="hi",
    )


CHARACTER_DESIGN_LEGACY_OVERLAY_PROSE = (
    "Focus on stable character, relationship, group, and role facts."
)
PROMPT_TOOL_NAME_CANDIDATES = {
    *SETUP_AGENT_VISIBLE_TOOLS,
    "memory.get_state",
    "memory.get_summary",
    "memory.search_recall",
    "memory.search_archival",
    "memory.list_versions",
    "memory.read_provenance",
    "setup.proposal.commit",
    "setup.question.raise",
    "setup.discussion.update_state",
    "setup.chunk.upsert",
    "setup.truth.write",
    "setup.patch.story_config",
    "setup.patch.writing_contract",
    "setup.patch.foundation_entry",
    "setup.patch.longform_blueprint",
    "setup.read.workspace",
    "setup.read.step_context",
    "setup.read.draft_refs",
    "setup.truth_index.search",
    "setup.truth_index.read_refs",
    "setup.world_background.list_entries",
    "setup.world_background.read_entry",
    "setup.world_background.write_entry",
    "setup.world_background.edit_entry",
    "setup.world_background.delete_entry",
    "proposal.submit",
}


def test_setup_agent_prompt_assets_are_centralized_python_modules():
    assert "Fact grounding and creative design:" in SYSTEM_PROMPT_TEMPLATE
    assert "{workspace_snapshot}" in SYSTEM_PROMPT_TEMPLATE
    assert "named setup object" in capability_guidance_text(
        "setup_session_memory.search"
    )
    assert "world background" in stage_objective_text(SetupStageId.WORLD_BACKGROUND)
    assert "SetupStageCompactPrompt" in compact_prompt_system_prompt()


def test_character_design_stage_loads_skill_pack_into_system_prompt():
    service = SetupAgentPromptService()
    packet = _packet(current_stage=SetupStageId.CHARACTER_DESIGN)

    prompt = service.build_system_prompt(
        mode=StoryMode.LONGFORM,
        current_step=SetupStepId.FOUNDATION,
        current_stage=SetupStageId.CHARACTER_DESIGN,
        context_packet=packet,
    )

    assert "[Stage Skill Pack: character-design.v1]" in prompt
    assert "[/Stage Skill Pack]" in prompt
    assert "## Specialist hat" in prompt


def test_character_design_stage_inserts_specialist_hat_preamble():
    service = SetupAgentPromptService()
    packet = _packet(current_stage=SetupStageId.CHARACTER_DESIGN)

    prompt = service.build_system_prompt(
        mode=StoryMode.LONGFORM,
        current_step=SetupStepId.FOUNDATION,
        current_stage=SetupStageId.CHARACTER_DESIGN,
        context_packet=packet,
    )

    assert "For this turn, you operate in the 角色设定 stage." in prompt
    assert (
        "While in this stage, take on the perspective of the Specialist hat" in prompt
    )
    assert "never break the SetupAgent operating envelope above" in prompt


def test_character_design_stage_short_circuits_legacy_stage_overlay_prose():
    service = SetupAgentPromptService()
    packet = _packet(current_stage=SetupStageId.CHARACTER_DESIGN)

    prompt = service.build_system_prompt(
        mode=StoryMode.LONGFORM,
        current_step=SetupStepId.FOUNDATION,
        current_stage=SetupStageId.CHARACTER_DESIGN,
        context_packet=packet,
    )

    assert CHARACTER_DESIGN_LEGACY_OVERLAY_PROSE not in prompt


def test_character_design_stage_keeps_single_setup_agent_identity_declaration():
    service = SetupAgentPromptService()
    packet = _packet(current_stage=SetupStageId.CHARACTER_DESIGN)

    prompt = service.build_system_prompt(
        mode=StoryMode.LONGFORM,
        current_step=SetupStepId.FOUNDATION,
        current_stage=SetupStageId.CHARACTER_DESIGN,
        context_packet=packet,
    )

    assert prompt.count("You are SetupAgent") == 1
    assert "You are a senior dramatist" not in prompt


def test_no_skill_pack_when_current_stage_is_none_uses_capability_guidance():
    service = SetupAgentPromptService()
    packet = _packet(current_stage=None, current_step=SetupStepId.FOUNDATION)

    prompt = service.build_system_prompt(
        mode=StoryMode.LONGFORM,
        current_step=SetupStepId.FOUNDATION,
        current_stage=None,
        context_packet=packet,
    )

    assert "[Stage Skill Pack" not in prompt
    assert "Active capability guidance:" in prompt
    assert "setup.memory.search" in prompt
    assert "setup.memory.open" in prompt
    assert "setup.memory.read_refs" not in prompt
    assert "setup.patch.foundation_entry" not in prompt
    assert "setup.truth.write" not in prompt
    assert "proposal.submit" not in prompt


def test_system_prompt_separates_fact_grounding_from_creative_proposals():
    service = SetupAgentPromptService()
    packet = _packet(current_stage=SetupStageId.WORLD_BACKGROUND)

    prompt = service.build_system_prompt(
        mode=StoryMode.LONGFORM,
        current_step=SetupStepId.FOUNDATION,
        current_stage=SetupStageId.WORLD_BACKGROUND,
        context_packet=packet,
    )

    assert "Fact grounding and creative design:" in prompt
    assert "Confirmed setup facts must come from visible workspace context" in prompt
    assert "Indexes, navigation summaries, compact summaries" in prompt
    assert "Before designing around a named setup object" in prompt
    assert "Do not ask the user to restate recoverable setup facts" in prompt
    assert "present new details as proposals" in prompt
    assert "Do not reinterpret an existing setup name as a new invention" in prompt


@pytest.mark.parametrize(
    "stage_id",
    [
        SetupStageId.WORLD_BACKGROUND,
        SetupStageId.PLOT_BLUEPRINT,
        SetupStageId.WRITER_CONFIG,
        SetupStageId.WORKER_CONFIG,
        SetupStageId.OVERVIEW,
        SetupStageId.ACTIVATE,
        SetupStageId.RP_INTERACTION_CONTRACT,
        SetupStageId.TRPG_RULES,
    ],
)
def test_non_character_design_stages_do_not_load_any_skill_pack(stage_id):
    service = SetupAgentPromptService()
    packet = _packet(current_stage=stage_id)

    prompt = service.build_system_prompt(
        mode=StoryMode.LONGFORM,
        current_step=SetupStepId.FOUNDATION,
        current_stage=stage_id,
        context_packet=packet,
    )

    assert "[Stage Skill Pack" not in prompt
    assert "For this turn, you operate in the" not in prompt


def test_world_background_prompt_exposes_stage_entry_guidance():
    service = SetupAgentPromptService()
    packet = _packet(current_stage=SetupStageId.WORLD_BACKGROUND)

    prompt = service.build_system_prompt(
        mode=StoryMode.LONGFORM,
        current_step=SetupStepId.FOUNDATION,
        current_stage=SetupStageId.WORLD_BACKGROUND,
        context_packet=packet,
    )

    assert "setup.stage_entry.write as the primary draft write tool" in prompt
    assert "setup.stage_entry.list" in prompt
    assert "setup.stage_entry.read" in prompt
    assert "setup.stage_entry.edit" in prompt
    assert "setup.stage_entry.delete" in prompt
    assert "For canonical stage setup.truth.write calls" not in prompt
    assert "setup.truth.write" not in prompt
    assert "Use the setup.world_background.* tools" not in prompt
    assert "setup.world_background.write_entry" not in prompt
    assert "setup.world_background.edit_entry" not in prompt
    assert "setup.world_background.delete_entry" not in prompt


@pytest.mark.parametrize(
    "stage_id",
    [
        SetupStageId.WORLD_BACKGROUND,
        SetupStageId.PLOT_BLUEPRINT,
        SetupStageId.WRITER_CONFIG,
        SetupStageId.WORKER_CONFIG,
        SetupStageId.OVERVIEW,
        SetupStageId.ACTIVATE,
        SetupStageId.RP_INTERACTION_CONTRACT,
        SetupStageId.TRPG_RULES,
    ],
)
def test_non_character_design_stage_prompt_has_no_skill_pack_residue(stage_id):
    service = SetupAgentPromptService()
    packet = _packet(current_stage=stage_id)

    prompt = service.build_system_prompt(
        mode=StoryMode.LONGFORM,
        current_step=SetupStepId.FOUNDATION,
        current_stage=stage_id,
        context_packet=packet,
    )

    assert "[Stage Skill Pack" not in prompt
    assert "For this turn, you operate in the" not in prompt
    assert "Active capability guidance:" in prompt


def test_prompt_mentions_only_active_capability_tools_for_world_background():
    service = SetupAgentPromptService()
    packet = _packet(current_stage=SetupStageId.WORLD_BACKGROUND)
    plan = build_setup_agent_capability_plan(
        SetupStepId.FOUNDATION.value,
        current_stage=SetupStageId.WORLD_BACKGROUND.value,
    )

    prompt = service.build_system_prompt(
        mode=StoryMode.LONGFORM,
        current_step=SetupStepId.FOUNDATION,
        current_stage=SetupStageId.WORLD_BACKGROUND,
        context_packet=packet,
        capability_plan=plan,
    )

    mentioned_tools = {
        tool_name for tool_name in PROMPT_TOOL_NAME_CANDIDATES if tool_name in prompt
    }
    assert mentioned_tools.issubset(set(plan.active_tool_names))
    assert "setup.memory.search" in mentioned_tools
    assert "setup.memory.open" in mentioned_tools
    assert "setup.memory.read_refs" not in mentioned_tools
    assert "setup.stage_entry.write" in mentioned_tools
    assert "memory.get_state" not in mentioned_tools
    assert "memory.get_summary" not in mentioned_tools
    assert "memory.search_recall" not in mentioned_tools
    assert "memory.search_archival" not in mentioned_tools
    assert "memory.list_versions" not in mentioned_tools
    assert "memory.read_provenance" not in mentioned_tools
    assert "setup.truth.write" not in mentioned_tools
    assert "setup.proposal.commit" not in mentioned_tools
    assert "setup.question.raise" not in mentioned_tools
    assert "setup.discussion.update_state" not in mentioned_tools
    assert "setup.chunk.upsert" not in mentioned_tools
    assert "setup.world_background.write_entry" not in mentioned_tools
    assert "setup.read.workspace" not in mentioned_tools
    assert "setup.read.step_context" not in mentioned_tools
    assert "setup.read.draft_refs" not in mentioned_tools
    assert "setup.truth_index.search" not in mentioned_tools
    assert "setup.truth_index.read_refs" not in mentioned_tools
    assert "proposal.submit" not in mentioned_tools


def test_memory_capability_guidance_mentions_named_object_grounding():
    plan = build_setup_agent_capability_plan(
        SetupStepId.FOUNDATION.value,
        current_stage=SetupStageId.WORLD_BACKGROUND.value,
    )

    memory_guidance = next(
        fragment.text
        for fragment in plan.prompt_guidance_fragments
        if fragment.fragment_id == "setup_session_memory.search"
    )

    assert "named setup object" in memory_guidance
    assert "before creative design" in memory_guidance
    assert (
        "Search results and navigation_summary are navigation only" in memory_guidance
    )
    assert "setup.memory.open" in memory_guidance
    assert "clearly framed as proposals" in memory_guidance


def test_prompt_filters_inactive_guidance_from_supplied_capability_plan():
    service = SetupAgentPromptService()
    packet = _packet(current_stage=SetupStageId.WORLD_BACKGROUND)
    plan = build_setup_agent_capability_plan(
        SetupStepId.FOUNDATION.value,
        current_stage=SetupStageId.WORLD_BACKGROUND.value,
    )
    bad_plan = plan.model_copy(
        update={
            "prompt_guidance_fragments": [
                *plan.prompt_guidance_fragments,
                SetupCapabilityGuidanceFragment(
                    fragment_id="bad.world_background",
                    tool_names=["setup.world_background.write_entry"],
                    text="Use setup.world_background.write_entry directly.",
                ),
            ]
        },
        deep=True,
    )

    prompt = service.build_system_prompt(
        mode=StoryMode.LONGFORM,
        current_step=SetupStepId.FOUNDATION,
        current_stage=SetupStageId.WORLD_BACKGROUND,
        context_packet=packet,
        capability_plan=bad_plan,
    )

    assert "setup.world_background.write_entry" not in prompt


def test_system_prompt_contains_only_stable_context_packet_not_runtime_artifact_data():
    service = SetupAgentPromptService()
    packet = _packet(current_stage=SetupStageId.WORLD_BACKGROUND)
    plan = build_setup_agent_capability_plan(
        SetupStepId.FOUNDATION.value,
        current_stage=SetupStageId.WORLD_BACKGROUND.value,
    )

    prompt = service.build_system_prompt(
        mode=StoryMode.LONGFORM,
        current_step=SetupStepId.FOUNDATION,
        current_stage=SetupStageId.WORLD_BACKGROUND,
        context_packet=packet,
        capability_plan=plan,
    )

    assert '"workspace_id": "ws-test"' in prompt
    assert '"current_stage": "world_background"' in prompt
    assert "context_report" not in prompt
    assert "raw_history_limit" not in prompt
    assert "history_count_threshold" not in prompt
    assert "working_digest" not in prompt
    assert "tool_outcomes" not in prompt
    assert "source_fingerprint" not in prompt
    assert "summary_lines" not in prompt
    assert "loop_trace" not in prompt
    assert "continue_reason" not in prompt
    assert "raw_tool_retry_process" not in prompt
