"""Runtime-internal contracts for RP agent execution."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from rp.models.setup_stage import SetupStageId
from rp.models.setup_workspace import SetupStepId


class RuntimeProfile(BaseModel):
    """Fixed runtime behavior for one agent profile."""

    model_config = ConfigDict(extra="forbid")

    profile_id: str
    supports_tools: bool = True
    visible_tool_names: list[str] = Field(default_factory=list)
    max_rounds: int = 8
    allow_stream: bool = True
    recovery_policy: str = "default"
    finish_policy: str = "default"


class SetupCapabilityGuidanceFragment(BaseModel):
    """Prompt guidance tied to the active setup capability plan."""

    model_config = ConfigDict(extra="forbid")

    fragment_id: str
    tool_names: list[str] = Field(default_factory=list)
    text: str


class SetupCapabilityPlan(BaseModel):
    """Single turn-level contract for SetupAgent tool exposure."""

    model_config = ConfigDict(extra="forbid")

    stage_id: str | None = None
    step_id: str | None = None
    active_tool_names: list[str] = Field(default_factory=list)
    model_schema_modes: dict[str, str] = Field(default_factory=dict)
    runtime_allowlist: list[str] = Field(default_factory=list)
    prompt_guidance_fragments: list[SetupCapabilityGuidanceFragment] = Field(
        default_factory=list
    )
    candidate_exclusions: list[str] = Field(default_factory=list)
    snapshot_expectations: dict[str, Any] = Field(default_factory=dict)


SETUP_CONTEXT_PIPELINE_LAYERS: tuple[str, ...] = (
    "context_packet",
    "stage_local_context_governance",
    "runtime_adapter_bundle",
    "runtime_request_assembly",
)
SETUP_FINAL_REQUEST_MESSAGE_ORDER: tuple[str, ...] = (
    "stable_system_prompt",
    "runtime_overlay_system_message",
    "governed_history",
    "current_user",
)
SETUP_STABLE_SYSTEM_PROMPT_INPUTS: tuple[str, ...] = (
    "context_packet",
    "capability_plan",
    "skill_pack_prompt_layer",
)
SETUP_RUNTIME_OVERLAY_INPUTS: tuple[str, ...] = (
    "turn_goal",
    "working_plan",
    "pending_obligation",
    "last_failure",
    "reflection_ticket",
    "cognitive_state_summary",
    "working_digest",
    "tool_outcomes",
    "compact_summary",
)
SETUP_CONTEXT_PACKET_EXCLUDED_FIELDS: tuple[str, ...] = (
    "raw_history",
    "working_digest",
    "tool_outcomes",
    "compact_summary",
    "turn_goal",
    "working_plan",
    "loop_trace",
    "continue_reason",
    "latest_tool_results",
)
SETUP_RUNTIME_STATE_DURABLE_FIELDS: tuple[str, ...] = (
    "workspace_id",
    "current_step",
    "state_version",
    "discussion_state",
    "chunk_candidates",
    "active_truth_write",
    "working_digest",
    "tool_outcomes",
    "compact_summary",
    "invalidated",
    "invalidation_reasons",
    "source_basis",
)
SETUP_RUNTIME_STATE_TRANSIENT_EXCLUDED_FIELDS: tuple[str, ...] = (
    "context_report",
    "loop_trace",
    "continue_reason",
    "finish_reason",
    "output_inspection",
    "event_sink",
    "model_gateway_diagnostics",
    "latest_response",
    "raw_provider_delta",
    "raw_provider_deltas",
    "provider_delta",
    "raw_delta",
    "raw",
    "private_event",
    "private_events",
    "private_diagnostics",
    "private_details",
    "debug",
    "debug_payload",
    "provider_debug",
    "exception",
    "stack",
    "stacktrace",
    "trace",
)
SETUP_DURABLE_STATE_EXCLUDED_FIELDS = SETUP_RUNTIME_STATE_TRANSIENT_EXCLUDED_FIELDS
SETUP_PUBLIC_EVENT_TYPES: tuple[str, ...] = (
    "thinking_delta",
    "text_delta",
    "tool_call",
    "tool_started",
    "tool_result",
    "tool_error",
    "usage",
    "error",
    "done",
)
SETUP_PRIVATE_EVENT_SURFACES: tuple[str, ...] = (
    "raw_provider_delta",
    "provider_delta",
    "debug",
    "private_diagnostics",
    "model_gateway_diagnostics",
)


class SetupPromptAssemblySnapshot(BaseModel):
    """Debug/eval contract for stable setup prompt assembly inputs."""

    model_config = ConfigDict(extra="forbid")

    stable_system_prompt_inputs: list[str] = Field(
        default_factory=lambda: list(SETUP_STABLE_SYSTEM_PROMPT_INPUTS)
    )
    stable_system_prompt_excluded_inputs: list[str] = Field(
        default_factory=lambda: [
            "governed_history",
            "context_report",
            "working_digest",
            "tool_outcomes",
            "compact_summary",
            "loop_trace",
            "continue_reason",
            "raw_tool_retry_process",
        ]
    )
    capability_guidance_source: str = "SetupCapabilityPlan.prompt_guidance_fragments"
    skill_pack_boundary: str = "prompt_only_does_not_change_tool_scope"
    active_skill_pack_name: str | None = None


class SetupContextPipelineSnapshot(BaseModel):
    """Debug/eval contract for setup pre-model context assembly."""

    model_config = ConfigDict(extra="forbid")

    pipeline_name: str = "setup_context_pipeline"
    layers: list[str] = Field(
        default_factory=lambda: list(SETUP_CONTEXT_PIPELINE_LAYERS)
    )
    final_request_message_order: list[str] = Field(
        default_factory=lambda: list(SETUP_FINAL_REQUEST_MESSAGE_ORDER)
    )
    governed_history_source: str = "SetupContextGovernorService.govern_history"
    context_packet_excluded_fields: list[str] = Field(
        default_factory=lambda: list(SETUP_CONTEXT_PACKET_EXCLUDED_FIELDS)
    )
    runtime_overlay_inputs: list[str] = Field(
        default_factory=lambda: list(SETUP_RUNTIME_OVERLAY_INPUTS)
    )
    metadata_only_surfaces: list[str] = Field(
        default_factory=lambda: [
            "context_report",
            "capability_plan",
            "skill_pack_name",
            "context_pipeline",
        ]
    )
    durable_state_excluded_fields: list[str] = Field(
        default_factory=lambda: list(SETUP_DURABLE_STATE_EXCLUDED_FIELDS)
    )
    prompt_assembly: SetupPromptAssemblySnapshot = Field(
        default_factory=SetupPromptAssemblySnapshot
    )
    context_profile: Literal["standard", "compact"] | None = None


class SetupModelGatewayDiagnostics(BaseModel):
    """Private normalized provider/gateway failure diagnostics for one turn."""

    model_config = ConfigDict(extra="forbid")

    failure_layer: Literal["model_gateway"] = "model_gateway"
    failure_kind: str
    message: str
    provider_error_type: str | None = None
    private_details: dict[str, Any] = Field(default_factory=dict)


class SetupEventSinkSnapshot(BaseModel):
    """Debug/eval contract for setup runtime event visibility."""

    model_config = ConfigDict(extra="forbid")

    public_event_types: list[str] = Field(
        default_factory=lambda: list(SETUP_PUBLIC_EVENT_TYPES)
    )
    private_event_surfaces: list[str] = Field(
        default_factory=lambda: list(SETUP_PRIVATE_EVENT_SURFACES)
    )
    transcript_boundary: str = "TypedSseEventAdapter"


class RpAgentTurnInput(BaseModel):
    """Unified runtime input after project-specific adapter mapping."""

    model_config = ConfigDict(extra="forbid")

    profile_id: str
    run_kind: str
    story_id: str | None = None
    workspace_id: str | None = None
    session_id: str | None = None
    model_id: str
    provider_id: str | None = None
    stream: bool = False
    user_visible_request: str | None = None
    conversation_messages: list[dict[str, Any]] = Field(default_factory=list)
    context_bundle: dict[str, Any] = Field(default_factory=dict)
    tool_scope: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RuntimeToolCall(BaseModel):
    """Normalized model-declared tool call."""

    model_config = ConfigDict(extra="forbid")

    call_id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    source_round: int


class RuntimeToolResult(BaseModel):
    """Normalized tool execution outcome."""

    model_config = ConfigDict(extra="forbid")

    call_id: str
    tool_name: str
    success: bool
    content_text: str
    error_code: str | None = None
    structured_payload: dict[str, Any] | None = None


SetupOutputClassification = Literal[
    "real_tool_call",
    "normal_text",
    "pseudo_tool_text",
    "malformed_tool_call",
    "empty_output",
    "provider_schema_error",
    "mixed_text_and_tool_call",
]


class SetupOutputInspection(BaseModel):
    """Typed boundary between normalized model output and loop routing."""

    model_config = ConfigDict(extra="forbid")

    classification: SetupOutputClassification
    public_text_candidate: str = ""
    tool_calls: list[RuntimeToolCall] = Field(default_factory=list)
    repair_observation: dict[str, Any] | None = None
    private_diagnostics: dict[str, Any] = Field(default_factory=dict)
    continue_reason_candidate: str | None = None
    finish_reason_candidate: str | None = None


class RuntimeWorkingNote(BaseModel):
    """Internal-only runtime note."""

    model_config = ConfigDict(extra="forbid")

    kind: str
    message: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DiscussionCandidateDirection(BaseModel):
    """One candidate direction still under discussion in the current step."""

    model_config = ConfigDict(extra="forbid")

    direction_id: str
    label: str
    summary: str
    supporting_points: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    status: Literal["active", "selected", "discarded"] = "active"


class DiscussionState(BaseModel):
    """Runtime-private discussion map for the current setup step."""

    model_config = ConfigDict(extra="forbid")

    current_step: str
    discussion_topic: str
    confirmed_points: list[str] = Field(default_factory=list)
    candidate_directions: list[DiscussionCandidateDirection] = Field(
        default_factory=list
    )
    open_questions: list[str] = Field(default_factory=list)
    unresolved_conflicts: list[str] = Field(default_factory=list)
    user_preference_signals: list[str] = Field(default_factory=list)
    next_focus: str | None = None
    convergence_score: float | None = None


class ChunkCandidate(BaseModel):
    """Structured truth candidate distilled from discussion."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    current_step: str
    block_type: Literal[
        "story_config",
        "writing_contract",
        "foundation_entry",
        "longform_blueprint",
        "stage_draft",
    ]
    stage_id: SetupStageId | None = None
    target_ref: str | None = None
    title: str
    content: str
    detail_level: Literal["rough", "usable", "truth_candidate"]
    tags: list[str] = Field(default_factory=list)
    source_turn_refs: list[str] = Field(default_factory=list)
    unresolved_issues: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    confidence: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DraftTruthWrite(BaseModel):
    """Runtime-private structured write intent before commit proposal."""

    model_config = ConfigDict(extra="forbid")

    write_id: str
    current_step: str
    block_type: Literal[
        "story_config",
        "writing_contract",
        "foundation_entry",
        "longform_blueprint",
        "stage_draft",
    ]
    stage_id: SetupStageId | None = None
    target_ref: str | None = None
    operation: Literal["create", "merge", "replace"]
    payload: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    remaining_open_issues: list[str] = Field(default_factory=list)
    ready_for_review: bool = False


class SetupCognitiveSourceBasis(BaseModel):
    """Workspace-side source basis used to validate cross-turn cognitive state."""

    model_config = ConfigDict(extra="forbid")

    workspace_version: int
    draft_fingerprint: str | None = None
    pending_user_edit_delta_ids: list[str] = Field(default_factory=list)
    last_proposal_status: str | None = None
    current_step: str


class SetupWorkingDigest(BaseModel):
    """Thin stage-local control state carried across setup turns."""

    model_config = ConfigDict(extra="forbid")

    current_goal: str | None = None
    next_focus: str | None = None
    open_questions: list[str] = Field(default_factory=list)
    rejected_directions: list[str] = Field(default_factory=list)
    draft_refs: list[str] = Field(default_factory=list)
    pending_obligation: str | None = None
    commit_blockers: list[str] = Field(default_factory=list)


class SetupToolOutcome(BaseModel):
    """Final tool outcome retained as thin cross-turn context."""

    model_config = ConfigDict(extra="forbid")

    tool_name: str
    success: bool
    summary: str
    updated_refs: list[str] = Field(default_factory=list)
    error_code: str | None = None
    relevance: Literal[
        "cognitive",
        "draft",
        "question",
        "proposal",
        "read",
        "asset",
        "failure",
        "other",
    ] = "other"
    recorded_at: datetime


class SetupCompactRecoveryHint(BaseModel):
    """Draft-ref hint used to recover exact detail after stage-local compaction."""

    model_config = ConfigDict(extra="forbid")

    ref: str
    reason: str
    detail: str | None = None


class SetupContextCompactSummary(BaseModel):
    """Compacted carry-forward summary for trimmed older current-step history."""

    model_config = ConfigDict(extra="forbid")

    source_fingerprint: str
    source_message_count: int = 0
    summary_lines: list[str] = Field(default_factory=list)
    confirmed_points: list[str] = Field(default_factory=list)
    open_threads: list[str] = Field(default_factory=list)
    rejected_directions: list[str] = Field(default_factory=list)
    draft_refs: list[str] = Field(default_factory=list)
    recovery_hints: list[SetupCompactRecoveryHint] = Field(default_factory=list)
    must_not_infer: list[str] = Field(default_factory=list)


class SetupContextGovernanceReport(BaseModel):
    """Transient report describing one turn's stage-local context decision."""

    model_config = ConfigDict(extra="forbid")

    context_profile: Literal["standard", "compact"]
    profile_reasons: list[str] = Field(default_factory=list)
    raw_history_count: int = 0
    raw_history_chars: int = 0
    estimated_input_tokens: int | None = None
    input_token_count_source: str | None = None
    previous_prompt_tokens: int | None = None
    previous_completion_tokens: int | None = None
    previous_total_tokens: int | None = None
    previous_cached_tokens: int | None = None
    previous_reasoning_tokens: int | None = None
    previous_cache_creation_input_tokens: int | None = None
    previous_cache_read_input_tokens: int | None = None
    previous_usage_source: str | None = None
    previous_token_details: dict[str, Any] = Field(default_factory=dict)
    user_edit_delta_count: int = 0
    prior_stage_handoff_count: int = 0
    raw_history_limit: int = 0
    kept_history_count: int = 0
    compacted_history_count: int = 0
    retained_tool_outcome_count: int = 0
    summary_strategy: Literal[
        "none",
        "deterministic_prefix_summary",
        "compact_prompt_summary",
    ] = "none"
    summary_action: Literal[
        "none",
        "reused_existing",
        "updated_existing",
        "rebuilt",
    ] = "none"
    summary_line_count: int = 0
    fallback_reason: str | None = None


class SetupDraftRefReadInput(BaseModel):
    """Read-only draft-ref recovery request for setup compact summaries."""

    model_config = ConfigDict(extra="forbid")

    workspace_id: str
    step_id: SetupStepId
    stage_id: SetupStageId | None = None
    refs: list[str]
    detail: Literal["summary", "full"] = "summary"
    max_chars: int = 4000


class SetupDraftRefReadItem(BaseModel):
    """One read result for a requested setup draft ref."""

    model_config = ConfigDict(extra="forbid")

    ref: str
    found: bool
    block_type: str | None = None
    title: str | None = None
    summary: str | None = None
    payload: dict[str, Any] | None = None


class SetupDraftRefReadResult(BaseModel):
    """Read-only result payload for compact-summary draft-ref recovery."""

    model_config = ConfigDict(extra="forbid")

    success: bool
    items: list[SetupDraftRefReadItem] = Field(default_factory=list)
    missing_refs: list[str] = Field(default_factory=list)


class SetupCognitiveStateSnapshot(BaseModel):
    """Cross-turn runtime-private cognitive snapshot for one setup step."""

    model_config = ConfigDict(extra="forbid")

    workspace_id: str
    current_step: str
    state_version: int = 1
    discussion_state: DiscussionState | None = None
    chunk_candidates: list[ChunkCandidate] = Field(default_factory=list)
    active_truth_write: DraftTruthWrite | None = None
    working_digest: SetupWorkingDigest | None = None
    tool_outcomes: list[SetupToolOutcome] = Field(default_factory=list)
    compact_summary: SetupContextCompactSummary | None = None
    invalidated: bool = False
    invalidation_reasons: list[str] = Field(default_factory=list)
    source_basis: SetupCognitiveSourceBasis


class SetupCognitiveStateSummary(BaseModel):
    """Prompt-sized summary view of the current cognitive snapshot."""

    model_config = ConfigDict(extra="forbid")

    current_step: str
    invalidated: bool = False
    invalidation_reasons: list[str] = Field(default_factory=list)
    discussion_topic: str | None = None
    confirmed_points: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    unresolved_conflicts: list[str] = Field(default_factory=list)
    candidate_titles: list[str] = Field(default_factory=list)
    truth_write_status: str | None = None
    ready_for_review: bool = False
    remaining_open_issues: list[str] = Field(default_factory=list)
    working_digest: SetupWorkingDigest | None = None
    tool_outcomes: list[SetupToolOutcome] = Field(default_factory=list)
    compact_summary: SetupContextCompactSummary | None = None


class SetupTurnGoal(BaseModel):
    """Setup-only per-turn goal derived from step state and runtime obligations."""

    model_config = ConfigDict(extra="forbid")

    current_step: str
    goal_type: Literal[
        "brainstorm_and_clarify",
        "clarify_user_intent",
        "fill_missing_step_fields",
        "patch_draft",
        "reconcile_after_user_edit",
        "refine_chunk_candidate",
        "write_draft_truth",
        "prepare_commit_intent",
        "recover_from_tool_failure",
    ]
    goal_summary: str
    success_criteria: list[str] = Field(default_factory=list)


class SetupWorkingPlan(BaseModel):
    """Setup-only working plan for the current step slice."""

    model_config = ConfigDict(extra="forbid")

    missing_information: list[str] = Field(default_factory=list)
    discussion_actions: list[str] = Field(default_factory=list)
    candidate_targets: list[str] = Field(default_factory=list)
    draft_write_targets: list[str] = Field(default_factory=list)
    patch_targets: list[str] = Field(default_factory=list)
    question_targets: list[str] = Field(default_factory=list)
    commit_readiness_checks: list[str] = Field(default_factory=list)
    current_priority: str | None = None


class SetupPendingObligation(BaseModel):
    """Runtime-only unresolved work that blocks successful completion."""

    model_config = ConfigDict(extra="forbid")

    obligation_type: Literal[
        "repair_tool_call",
        "ask_user_for_missing_info",
        "continue_after_tool_failure",
        "reassess_commit_readiness",
        "reconcile_after_user_edit",
    ]
    reason: str
    tool_name: str | None = None
    required_fields: list[str] = Field(default_factory=list)
    unresolved: bool = True


class SetupLastFailure(BaseModel):
    """Most recent setup runtime failure interpreted into agent semantics."""

    model_config = ConfigDict(extra="forbid")

    failure_category: Literal[
        "auto_repair",
        "ask_user",
        "continue_discussion",
        "block_commit",
        "unrecoverable",
    ]
    message: str
    error_code: str | None = None
    tool_name: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class SetupReflectionTicket(BaseModel):
    """Lightweight reflection request inserted by runtime policies."""

    model_config = ConfigDict(extra="forbid")

    trigger: Literal[
        "tool_failure",
        "before_commit_proposal",
        "proposal_rejected",
    ]
    summary: str
    required_decision: Literal[
        "retry",
        "ask_user",
        "continue_discussion",
        "block_commit",
    ]


class SetupCompletionGuard(BaseModel):
    """Finalization guard result computed inside runtime only."""

    model_config = ConfigDict(extra="forbid")

    allow_finalize: bool
    reason: str
    required_action: Literal[
        "finalize_success",
        "execute_tools",
        "ask_user",
        "continue_discussion",
        "retry",
        "reflect",
        "finalize_failure",
    ]
    finish_reason: str | None = None


class SetupReActTraceFrame(BaseModel):
    """Thin runtime-authored semantic trace frame for one loop decision site."""

    model_config = ConfigDict(extra="forbid")

    round_no: int
    decision_site: Literal[
        "inspect_model_output",
        "assess_progress",
        "reflect_if_needed",
        "finalize_success",
        "finalize_failure",
    ]
    goal: dict[str, Any] | None = None
    plan: dict[str, Any] | None = None
    action: dict[str, Any] = Field(default_factory=dict)
    observation: dict[str, Any] = Field(default_factory=dict)
    reflection: dict[str, Any] | None = None
    decision: dict[str, Any] = Field(default_factory=dict)


class RpAgentTurnResult(BaseModel):
    """Final runtime outcome exposed back to the project layer."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["completed", "failed", "stopped"]
    finish_reason: str
    assistant_text: str = ""
    tool_invocations: list[RuntimeToolCall] = Field(default_factory=list)
    tool_results: list[RuntimeToolResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    structured_payload: dict[str, Any] = Field(default_factory=dict)
    error: dict[str, Any] | None = None
