"""Stage and legacy-step objective prompt fragments for SetupAgent."""

from __future__ import annotations

from rp.models.setup_stage import SetupStageId
from rp.models.setup_workspace import SetupStepId

STAGE_OBJECTIVES: dict[SetupStageId | SetupStepId, str] = {
    SetupStageId.WORLD_BACKGROUND: (
        "- Focus on stable world background, rules, locations, history, factions, "
        "races, and other world facts.\n"
        "- Keep entries structured and retrieval-addressable.\n"
        "- Do not mix character-only details into this stage unless they define the world."
    ),
    SetupStageId.CHARACTER_DESIGN: (
        "- Focus on stable character, relationship, group, and role facts.\n"
        "- Use prior world handoffs as accepted context; do not replay old discussion.\n"
        "- Keep character entries separate from world-background entries."
    ),
    SetupStageId.PLOT_BLUEPRINT: (
        "- Focus on plot threads, foreshadowing, premise, conflict, arcs, and chapter plan.\n"
        "- Use accepted world and character handoffs as constraints."
    ),
    SetupStageId.WRITER_CONFIG: (
        "- Focus on POV, style, writing constraints, and task writing rules.\n"
        "- Do not turn the draft into one giant prompt blob."
    ),
    SetupStageId.WORKER_CONFIG: (
        "- Focus on worker policy, tool policy, and handoff rules.\n"
        "- Keep runtime configuration concise and explicit."
    ),
    SetupStageId.OVERVIEW: (
        "- Focus on review and activation readiness.\n"
        "- Do not add new foundation facts unless the user explicitly asks."
    ),
    SetupStageId.ACTIVATE: (
        "- Focus on review and activation readiness.\n"
        "- Do not add new foundation facts unless the user explicitly asks."
    ),
    SetupStepId.STORY_CONFIG: (
        "- Focus on story configuration and runtime profile convergence.\n"
        "- Do not modify mode.\n"
        "- Prefer clarification over premature commit."
    ),
    SetupStepId.WRITING_CONTRACT: (
        "- Focus on POV, style, and writing constraints.\n"
        "- Do not turn the draft into one giant prompt blob."
    ),
    SetupStepId.FOUNDATION: (
        "- Focus on stable world, character, and rule facts.\n"
        "- Prefer concrete entries over vague lore summaries."
    ),
}

DEFAULT_STAGE_OBJECTIVE = (
    "- Focus on longform blueprint convergence.\n"
    "- Prefer enough structure to activate later, not perfect completeness."
)


def stage_objective_text(step_id: SetupStepId | SetupStageId) -> str:
    return STAGE_OBJECTIVES.get(step_id, DEFAULT_STAGE_OBJECTIVE)
