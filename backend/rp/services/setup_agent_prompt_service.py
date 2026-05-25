"""Prompt assembly for the SetupAgent MVP execution layer."""

from __future__ import annotations

import json

from rp.agent_runtime.contracts import SetupCapabilityPlan
from rp.agent_runtime.prompts.setup_agent import (
    capability_guidance_text,
    render_setup_agent_system_prompt,
    render_specialist_preamble,
    stage_objective_text,
)
from rp.agent_runtime.profiles import build_setup_agent_capability_plan
from rp.agent_runtime.skill_packs import (
    get_skill_pack_for_stage,
    render_skill_pack,
)
from rp.models.setup_handoff import SetupContextPacket
from rp.models.setup_stage import SETUP_STAGE_MODULES, SetupStageId
from rp.models.setup_workspace import SetupStepId, StoryMode


class SetupAgentPromptService:
    """Build the stable system prompt stack for the SetupAgent MVP."""

    def build_system_prompt(
        self,
        *,
        mode: StoryMode,
        current_step: SetupStepId,
        current_stage: SetupStageId | None,
        context_packet: SetupContextPacket,
        capability_plan: SetupCapabilityPlan | None = None,
    ) -> str:
        capability_plan = capability_plan or build_setup_agent_capability_plan(
            current_step.value,
            current_stage=current_stage.value if current_stage is not None else None,
        )
        stage_overlay = self._stage_overlay(current_stage or current_step)
        skill_pack = get_skill_pack_for_stage(current_stage)
        specialist_preamble = (
            self._specialist_preamble(current_stage)
            if skill_pack is not None and current_stage is not None
            else ""
        )
        workspace_snapshot = json.dumps(
            context_packet.model_dump(mode="json", exclude_none=True),
            ensure_ascii=False,
            sort_keys=True,
        )
        capability_guidance = self._capability_guidance(capability_plan)
        return render_setup_agent_system_prompt(
            specialist_preamble=specialist_preamble,
            capability_guidance=capability_guidance,
            mode=mode.value,
            current_step=current_step.value,
            current_stage=(
                current_stage.value if current_stage is not None else current_step.value
            ),
            stage_overlay=stage_overlay,
            workspace_snapshot=workspace_snapshot,
        )

    @staticmethod
    def _capability_guidance(capability_plan: SetupCapabilityPlan) -> str:
        active_tools = set(capability_plan.active_tool_names)
        fragments = [
            f"- {fragment.text}"
            for fragment in capability_plan.prompt_guidance_fragments
            if fragment.text.strip() and set(fragment.tool_names).issubset(active_tools)
        ]
        if not fragments:
            return f"- {capability_guidance_text('no_active_tools')}"
        return "\n".join(fragments)

    @staticmethod
    def _specialist_preamble(stage_id: SetupStageId) -> str:
        module = SETUP_STAGE_MODULES[stage_id]
        return render_specialist_preamble(stage_display_name=module.display_name)

    @staticmethod
    def _stage_overlay(step_id: SetupStepId | SetupStageId) -> str:
        if isinstance(step_id, SetupStageId):
            skill_pack = get_skill_pack_for_stage(step_id)
            if skill_pack is not None:
                return render_skill_pack(skill_pack)
        return stage_objective_text(step_id)
