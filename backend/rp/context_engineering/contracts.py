"""Runtime-agnostic contracts for pre-model context compaction."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ContextSourceFamily = Literal[
    "system_instruction",
    "user_turn",
    "assistant_turn",
    "tool_outcome",
    "runtime_state",
    "compact_artifact",
]

ContextVisibility = Literal[
    "model_visible",
    "metadata_only",
    "hidden",
    "forbidden",
]

ContextStability = Literal["stable_prefix", "volatile"]
ContextOperationKind = Literal["compact"]
ContextOperationStatus = Literal["not_needed", "reused", "updated", "rebuilt", "fallback"]

ContextManifestDecision = Literal[
    "selected",
    "omitted",
    "hidden",
    "forbidden",
    "metadata_only",
]


class ContextSourceItem(BaseModel):
    """One adapter-owned context source considered by the common kernel."""

    model_config = ConfigDict(extra="forbid")

    source_item_id: str
    source_family: ContextSourceFamily
    source_scope: str | None = None
    sequence_index: int | None = None
    visibility: ContextVisibility = "model_visible"
    stability: ContextStability = "volatile"
    source_ref: str | None = None
    recovery_refs: list[str] = Field(default_factory=list)
    text: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    estimated_tokens: int | None = None
    created_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_item_id")
    @classmethod
    def _source_item_id_must_not_be_blank(cls, value: str) -> str:
        if not str(value or "").strip():
            raise ValueError("source_item_id_required")
        return value


class ContextBudgetPolicy(BaseModel):
    """Budget limits and traceable pressure thresholds for one operation."""

    model_config = ConfigDict(extra="forbid")

    context_window_tokens: int | None = None
    response_reserve_tokens: int = 1024
    operation_budget_tokens: int | None = None
    recent_window_items: int | None = None
    source_family_token_caps: dict[str, int] = Field(default_factory=dict)
    source_family_item_caps: dict[str, int] = Field(default_factory=dict)


class ContextPlacementPolicy(BaseModel):
    """Provider-neutral placement slots."""

    model_config = ConfigDict(extra="forbid")

    ordered_slots: list[str]
    slot_by_source_family: dict[str, str] = Field(default_factory=dict)
    stable_prefix_slots: list[str] = Field(default_factory=list)


class ContextValidationPolicy(BaseModel):
    """Schema and field rules used before compact output is accepted."""

    model_config = ConfigDict(extra="forbid")

    schema_id: str | None = None
    allowed_recovery_ref_prefixes: list[str] = Field(default_factory=list)
    allowed_source_refs: list[str] = Field(default_factory=list)
    forbidden_payload_fields: list[str] = Field(default_factory=list)
    max_list_lengths: dict[str, int] = Field(default_factory=dict)
    max_string_lengths: dict[str, int] = Field(default_factory=dict)
    reject_unknown_fields: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextFallbackPolicy(BaseModel):
    """Single deterministic fallback signal used by initial compact callers."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["deterministic_fallback"] = "deterministic_fallback"
    fallback_summary_line_limit: int = 6
    user_visible_error_code: str | None = None


class ContextValidationIssue(BaseModel):
    """One structured validation issue."""

    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    field_path: str | None = None
    severity: Literal["error", "warning"] = "error"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextValidationReport(BaseModel):
    """Validation result with all issues retained."""

    model_config = ConfigDict(extra="forbid")

    valid: bool = True
    issues: list[ContextValidationIssue] = Field(default_factory=list)


class ContextFallbackReport(BaseModel):
    """Fallback evidence attached to an operation result."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["deterministic_fallback"] = "deterministic_fallback"
    reason: str
    used: bool = True
    user_visible_error_code: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextSection(BaseModel):
    """Provider-neutral serialized section built from selected sources."""

    model_config = ConfigDict(extra="forbid")

    section_id: str
    slot: str
    title: str
    content: str
    source_item_ids: list[str]
    stability: ContextStability
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextManifestItem(BaseModel):
    """Content-free evidence for a source selection decision."""

    model_config = ConfigDict(extra="forbid")

    source_item_id: str
    source_family: ContextSourceFamily
    source_scope: str | None = None
    source_ref: str | None = None
    visibility: ContextVisibility
    decision: ContextManifestDecision
    reason: str
    slot: str | None = None
    estimated_tokens: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextReadManifest(BaseModel):
    """Evidence of selected, omitted, and excluded source surfaces."""

    model_config = ConfigDict(extra="forbid")

    selected: list[ContextManifestItem] = Field(default_factory=list)
    omitted: list[ContextManifestItem] = Field(default_factory=list)
    hidden: list[ContextManifestItem] = Field(default_factory=list)
    forbidden: list[ContextManifestItem] = Field(default_factory=list)
    metadata_only: list[ContextManifestItem] = Field(default_factory=list)


class ContextBudgetDecision(BaseModel):
    """Trace event for a budget or retention choice."""

    model_config = ConfigDict(extra="forbid")

    decision: str
    reason: str
    source_item_id: str | None = None
    slot: str | None = None
    estimated_tokens: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextTrace(BaseModel):
    """Transient debug/eval trace for one context operation."""

    model_config = ConfigDict(extra="forbid")

    operation_id: str
    operation_kind: ContextOperationKind
    runtime_family: str
    estimate_method: str
    input_source_count: int
    selected_source_count: int
    omitted_source_count: int
    hidden_source_count: int
    forbidden_source_count: int
    metadata_only_source_count: int
    estimated_input_tokens: int
    selected_tokens: int
    budget_decisions: list[ContextBudgetDecision] = Field(default_factory=list)
    source_counts_by_family: dict[str, int] = Field(default_factory=dict)
    selected_counts_by_family: dict[str, int] = Field(default_factory=dict)
    summary_action: str | None = None
    fallback_reason: str | None = None
    cache_stability_notes: list[str] = Field(default_factory=list)
    provider_usage: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextArtifact(BaseModel):
    """Opaque compact artifact produced by a model runner or adapter."""

    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    artifact_kind: Literal["compact_summary"]
    schema_id: str
    schema_version: str
    source_fingerprint: str
    source_item_count: int
    payload: dict[str, Any]
    recovery_refs: list[str] = Field(default_factory=list)
    first_kept_source_item_id: str | None = None
    created_by: Literal["model", "adapter"]
    validation_report: ContextValidationReport
    fallback_report: ContextFallbackReport | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextSelectionResult(BaseModel):
    """Selection output and narrow handoff to optional compaction."""

    model_config = ConfigDict(extra="forbid")

    selected_items: list[ContextSourceItem] = Field(default_factory=list)
    recent_raw_items: list[ContextSourceItem] = Field(default_factory=list)
    compactable_dropped_items: list[ContextSourceItem] = Field(default_factory=list)
    sections: list[ContextSection] = Field(default_factory=list)
    read_manifest: ContextReadManifest
    trace: ContextTrace


class ContextOperationRequest(BaseModel):
    """Self-contained compact request created by a runtime adapter."""

    model_config = ConfigDict(extra="forbid")

    operation_id: str
    operation_kind: ContextOperationKind
    runtime_family: str
    source_items: list[ContextSourceItem]
    budget_policy: ContextBudgetPolicy
    placement_policy: ContextPlacementPolicy
    validation_policy: ContextValidationPolicy
    fallback_policy: ContextFallbackPolicy
    previous_artifact: ContextArtifact | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("operation_id", "runtime_family")
    @classmethod
    def _required_strings_must_not_be_blank(cls, value: str) -> str:
        if not str(value or "").strip():
            raise ValueError("value_required")
        return value


class ContextCompactPromptRequest(BaseModel):
    """Data-only prompt request for an injected no-tools compact runner."""

    model_config = ConfigDict(extra="forbid")

    operation_id: str
    action: Literal["updated", "rebuilt"]
    schema_id: str | None = None
    schema_version: str = "1"
    source_fingerprint: str
    source_item_count: int
    dropped_items: list[ContextSourceItem]
    previous_artifact_payload: dict[str, Any] | None = None
    first_kept_source_item_id: str | None = None
    validation_policy: ContextValidationPolicy
    fallback_policy: ContextFallbackPolicy
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextOperationResult(BaseModel):
    """Final output for a compact operation."""

    model_config = ConfigDict(extra="forbid")

    operation_id: str
    status: ContextOperationStatus
    sections: list[ContextSection] = Field(default_factory=list)
    artifact: ContextArtifact | None = None
    read_manifest: ContextReadManifest
    trace: ContextTrace
    validation_report: ContextValidationReport
    fallback_report: ContextFallbackReport | None = None
