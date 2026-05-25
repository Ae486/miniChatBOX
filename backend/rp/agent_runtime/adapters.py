"""Project-layer adapters for the RP agent runtime."""

from __future__ import annotations

from typing import Any

from models.chat import ProviderConfig
from rp.agent_runtime.contracts import (
    RpAgentTurnInput,
    RpAgentTurnResult,
    RuntimeProfile,
    SetupContextCompactSummary,
    SetupContextGovernanceReport,
    SetupContextPipelineSnapshot,
    SetupCognitiveStateSnapshot,
    SetupCognitiveStateSummary,
    SetupPromptAssemblySnapshot,
    SetupToolOutcome,
    SetupWorkingDigest,
)
from rp.agent_runtime.profiles import (
    build_setup_agent_capability_plan,
    build_setup_agent_profile,
)
from rp.agent_runtime.skill_packs import get_skill_pack_for_stage
from rp.models.setup_agent import (
    SetupAgentDialogueMessage,
    SetupAgentTurnRequest,
    SetupAgentTurnResponse,
)
from rp.models.setup_handoff import SetupContextPacket
from rp.models.setup_workspace import (
    CommitProposalStatus,
    QuestionSeverity,
    QuestionStatus,
    SetupStepId,
)
from rp.services.setup_agent_prompt_service import SetupAgentPromptService


class SetupRuntimeAdapter:
    """Map setup-layer objects into runtime-internal contracts."""

    def __init__(
        self,
        *,
        prompt_service: SetupAgentPromptService | None = None,
    ) -> None:
        self._prompt_service = prompt_service or SetupAgentPromptService()

    def build_turn_input(
        self,
        *,
        request: SetupAgentTurnRequest,
        workspace,
        context_packet: SetupContextPacket,
        model_name: str,
        provider: ProviderConfig,
        governed_history: list[SetupAgentDialogueMessage] | None = None,
        working_digest: SetupWorkingDigest | None = None,
        tool_outcomes: list[SetupToolOutcome] | None = None,
        compact_summary: SetupContextCompactSummary | None = None,
        governance_metadata: dict[str, Any] | None = None,
        context_report: SetupContextGovernanceReport | None = None,
        cognitive_state: SetupCognitiveStateSnapshot | None = None,
        cognitive_state_summary: SetupCognitiveStateSummary | None = None,
    ) -> RpAgentTurnInput:
        current_step = SetupStepId(context_packet.current_step)
        current_stage = getattr(context_packet, "current_stage", None)
        selected_stage = (
            request.target_stage
            if request.target_stage is not None
            else (
                current_stage
                if request.target_step is None
                or request.target_step == workspace.current_step
                else None
            )
        )
        current_stage_value = (
            selected_stage.value
            if selected_stage is not None
            else (current_stage.value if current_stage is not None else None)
        )
        capability_plan = build_setup_agent_capability_plan(
            current_step.value,
            current_stage=selected_stage.value if selected_stage is not None else None,
        )
        external_mcp_tool_allowlist = self._external_mcp_tool_allowlist(request)
        runtime_tool_scope = self._ordered_unique(
            [
                *capability_plan.runtime_allowlist,
                *external_mcp_tool_allowlist,
            ]
        )
        system_prompt = self._prompt_service.build_system_prompt(
            mode=workspace.mode,
            current_step=current_step,
            current_stage=selected_stage,
            context_packet=context_packet,
            capability_plan=capability_plan,
        )
        open_questions = [
            question
            for question in workspace.open_questions
            if question.step_id == current_step
            and question.status == QuestionStatus.OPEN
        ]
        blocking_open_questions = [
            question
            for question in open_questions
            if question.severity == QuestionSeverity.BLOCKING
        ]
        latest_proposal = self._latest_step_proposal(
            workspace=workspace,
            current_step=current_step,
            current_stage=selected_stage,
        )
        step_state = next(
            (item for item in workspace.step_states if item.step_id == current_step),
            None,
        )
        stage_state = next(
            (
                item
                for item in workspace.stage_states
                if selected_stage is not None and item.stage_id == selected_stage
            ),
            None,
        )
        skill_pack = get_skill_pack_for_stage(selected_stage)
        context_pipeline = SetupContextPipelineSnapshot(
            context_profile=context_packet.context_profile,
            prompt_assembly=SetupPromptAssemblySnapshot(
                active_skill_pack_name=(
                    skill_pack.name if skill_pack is not None else None
                )
            ),
        )
        return RpAgentTurnInput(
            profile_id="setup_agent",
            run_kind="interactive_agent_turn",
            story_id=workspace.story_id,
            workspace_id=workspace.workspace_id,
            model_id=request.model_id,
            provider_id=request.provider_id,
            stream=False,
            user_visible_request=request.user_prompt,
            conversation_messages=[
                item.model_dump(mode="json")
                for item in (
                    governed_history
                    if governed_history is not None
                    else request.history
                )
            ],
            context_bundle={
                "system_prompt": system_prompt,
                "context_packet": context_packet.model_dump(
                    mode="json", exclude_none=True
                ),
                "mode": workspace.mode.value,
                "current_step": current_step.value,
                "current_stage": current_stage_value,
                "step_state": (
                    step_state.model_dump(mode="json", exclude_none=True)
                    if step_state is not None
                    else None
                ),
                "stage_state": (
                    stage_state.model_dump(mode="json", exclude_none=True)
                    if stage_state is not None
                    else None
                ),
                "step_readiness": workspace.readiness_status.step_readiness.get(
                    current_step.value
                ),
                "stage_readiness": (
                    workspace.readiness_status.step_readiness.get(current_stage_value)
                    if current_stage_value is not None
                    else None
                ),
                "open_question_count": len(open_questions),
                "blocking_open_question_count": len(blocking_open_questions),
                "open_question_texts": [
                    question.text for question in open_questions[:5]
                ],
                "has_user_edit_deltas": bool(context_packet.user_edit_deltas),
                "has_prior_stage_handoffs": bool(context_packet.prior_stage_handoffs),
                "prior_stage_handoff_count": len(context_packet.prior_stage_handoffs),
                "prior_stage_handoff_steps": [
                    handoff.step_id.value
                    for handoff in context_packet.prior_stage_handoffs
                ],
                "prior_stage_handoff_stages": [
                    (
                        handoff.stage_id.value
                        if handoff.stage_id is not None
                        else (
                            handoff.from_stage.value
                            if handoff.from_stage is not None
                            else handoff.step_id.value
                        )
                    )
                    for handoff in context_packet.prior_stage_handoffs
                ],
                "last_proposal_status": (
                    latest_proposal.status.value
                    if latest_proposal is not None
                    else None
                ),
                "has_rejected_commit_proposal": bool(
                    latest_proposal is not None
                    and latest_proposal.status == CommitProposalStatus.REJECTED
                ),
                "cognitive_state": (
                    cognitive_state.model_dump(mode="json", exclude_none=True)
                    if cognitive_state is not None
                    else None
                ),
                "cognitive_state_summary": (
                    cognitive_state_summary.model_dump(mode="json", exclude_none=True)
                    if cognitive_state_summary is not None
                    else None
                ),
                "cognitive_state_invalidated": bool(
                    cognitive_state_summary is not None
                    and cognitive_state_summary.invalidated
                ),
                "working_digest": (
                    working_digest.model_dump(mode="json", exclude_none=True)
                    if working_digest is not None
                    else None
                ),
                "tool_outcomes": [
                    item.model_dump(mode="json", exclude_none=True)
                    for item in (tool_outcomes or [])
                ],
                "compact_summary": (
                    compact_summary.model_dump(mode="json", exclude_none=True)
                    if compact_summary is not None
                    else None
                ),
                "governance_metadata": dict(governance_metadata or {}),
            },
            tool_scope=runtime_tool_scope,
            metadata={
                "model_name": model_name,
                "provider": provider.model_dump(mode="json", exclude_none=True),
                "capability_plan": capability_plan.model_dump(
                    mode="json", exclude_none=True
                ),
                "external_mcp_tool_allowlist": external_mcp_tool_allowlist,
                "skill_pack_name": (
                    skill_pack.name if skill_pack is not None else None
                ),
                "context_report": (
                    context_report.model_dump(mode="json", exclude_none=True)
                    if context_report is not None
                    else None
                ),
                "context_pipeline": context_pipeline.model_dump(
                    mode="json", exclude_none=True
                ),
            },
        )

    @staticmethod
    def to_turn_response(result: RpAgentTurnResult) -> SetupAgentTurnResponse:
        return SetupAgentTurnResponse(assistant_text=result.assistant_text)

    @staticmethod
    def build_runtime_profile() -> RuntimeProfile:
        return build_setup_agent_profile()

    @staticmethod
    def _external_mcp_tool_allowlist(
        request: SetupAgentTurnRequest,
    ) -> list[str]:
        return SetupRuntimeAdapter._ordered_unique(
            [
                item.strip()
                for item in request.external_mcp_tool_allowlist
                if item.strip()
            ]
        )

    @staticmethod
    def _ordered_unique(values: list[str]) -> list[str]:
        return list(dict.fromkeys(values))

    @staticmethod
    def _latest_step_proposal(*, workspace, current_step, current_stage=None):
        if current_stage is not None:
            stage_proposals = [
                proposal
                for proposal in workspace.commit_proposals
                if proposal.step_id == current_stage
            ]
            if stage_proposals:
                return max(stage_proposals, key=lambda item: item.created_at)
        proposals = [
            proposal
            for proposal in workspace.commit_proposals
            if proposal.step_id == current_step
        ]
        if not proposals:
            return None
        return max(proposals, key=lambda item: item.created_at)
