"""Typed retrieval-runtime contracts shared by Runtime Workspace retrieval flow."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
)


def _require_non_blank(value: str | None, *, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")
    return normalized


def _normalize_optional_text(value: str | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_non_blank(value, field_name=field_name)


def _normalize_text_list(values: list[str], *, field_name: str) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return normalized


RuntimeRetrievalSearchMode = Literal[
    "entity",
    "relation",
    "semantic",
]

_LEGACY_RUNTIME_RETRIEVAL_MODE_MAP: dict[str, RuntimeRetrievalSearchMode] = {
    "entity_relation": "relation",
    "mixed": "semantic",
    "vague": "semantic",
}


class RetrievalKnowledgeGapItem(BaseModel):
    """Structured knowledge-gap contract kept on retrieval usage records."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    gap_id: str | None = None
    query_text: str = Field(alias="query")
    gap_kind: str = Field(alias="status")
    mode_policy_resolution: str | None = None
    notes: str | None = Field(default=None, alias="impact")

    @field_validator("gap_id", "mode_policy_resolution", "notes")
    @classmethod
    def _normalize_optional_text(
        cls,
        value: str | None,
        info: ValidationInfo,
    ) -> str | None:
        return _normalize_optional_text(
            value,
            field_name=info.field_name or "value",
        )

    @field_validator("query_text", "gap_kind")
    @classmethod
    def _normalize_required_text(cls, value: str, info: ValidationInfo) -> str:
        return _require_non_blank(value, field_name=info.field_name or "value")


class RuntimeRetrievalSearchInput(BaseModel):
    """LLM-facing runtime retrieval query expression.

    The model is allowed to describe what it needs to retrieve, but source
    routing, K values, filters, route weights, and rerank policy stay backend
    concerns.
    """

    model_config = ConfigDict(extra="forbid")

    query: str
    mode: RuntimeRetrievalSearchMode = Field(
        default="semantic",
        description=(
            "Retrieval intent hint only. Use entity for one concrete subject, "
            "relation for relationships between explicit anchors, and semantic "
            "for abstract background, rules, or broad context."
        ),
    )
    lexical_anchors: list[str] = Field(default_factory=list)
    semantic_predicates: list[str] = Field(default_factory=list)

    @field_validator("query")
    @classmethod
    def _normalize_required_text(cls, value: str, info: ValidationInfo) -> str:
        return _require_non_blank(value, field_name=info.field_name or "value")

    @field_validator("mode", mode="before")
    @classmethod
    def _normalize_mode(cls, value: Any) -> Any:
        if value is None:
            return "semantic"
        text = str(value or "").strip()
        if not text:
            return "semantic"
        return _LEGACY_RUNTIME_RETRIEVAL_MODE_MAP.get(text, text)

    @field_validator("lexical_anchors", "semantic_predicates")
    @classmethod
    def _normalize_hint_lists(
        cls,
        values: list[str],
        info: ValidationInfo,
    ) -> list[str]:
        return _normalize_text_list(values, field_name=info.field_name or "values")


class RuntimeRetrievalResultItem(BaseModel):
    """Clean RAG result item returned to runtime LLM callers."""

    model_config = ConfigDict(extra="forbid")

    result_id: str
    title: str | None = None
    summary: str | None = None
    excerpt: str | None = None
    text: str
    section: str | None = None

    @field_validator("result_id", "text")
    @classmethod
    def _normalize_required_text(cls, value: str, info: ValidationInfo) -> str:
        return _require_non_blank(value, field_name=info.field_name or "value")

    @field_validator("title", "summary", "excerpt", "section")
    @classmethod
    def _normalize_optional_text(
        cls,
        value: str | None,
        info: ValidationInfo,
    ) -> str | None:
        return _normalize_optional_text(
            value,
            field_name=info.field_name or "value",
        )


class RuntimeRetrievalSearchOutput(BaseModel):
    """Standard RAG search output for writer/worker/orchestrator tools."""

    model_config = ConfigDict(extra="forbid")

    query: str
    results: list[RuntimeRetrievalResultItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("query")
    @classmethod
    def _normalize_required_text(cls, value: str, info: ValidationInfo) -> str:
        return _require_non_blank(value, field_name=info.field_name or "value")

    @field_validator("warnings")
    @classmethod
    def _normalize_warnings(cls, values: list[str]) -> list[str]:
        return _normalize_text_list(values, field_name="warnings")


class WriterRetrievalSearchToolInput(RuntimeRetrievalSearchInput):
    """Writer-facing alias for the shared runtime retrieval search contract."""

    pass


class WriterRetrievalExpandToolInput(BaseModel):
    """Writer-facing expand tool contract using short ids only."""

    model_config = ConfigDict(extra="forbid")

    card_short_ids: list[str] = Field(default_factory=list)

    @field_validator("card_short_ids")
    @classmethod
    def _normalize_card_short_ids(
        cls,
        values: list[str],
    ) -> list[str]:
        normalized = _normalize_text_list(values, field_name="card_short_ids")
        if not normalized:
            raise ValueError("card_short_ids must not be empty")
        return normalized


class WriterRetrievalUsageToolInput(BaseModel):
    """Writer-facing usage tool contract using only short ids."""

    model_config = ConfigDict(extra="forbid")

    used_card_short_ids: list[str] = Field(default_factory=list)
    used_expanded_short_ids: list[str] = Field(default_factory=list)
    missed_query_short_ids: list[str] = Field(default_factory=list)
    knowledge_gaps: list[RetrievalKnowledgeGapItem] = Field(default_factory=list)

    @field_validator(
        "used_card_short_ids",
        "used_expanded_short_ids",
        "missed_query_short_ids",
    )
    @classmethod
    def _normalize_short_id_lists(
        cls,
        values: list[str],
        info: ValidationInfo,
    ) -> list[str]:
        return _normalize_text_list(values, field_name=info.field_name or "values")


class RetrievalUsageRecordPayload(BaseModel):
    """Stable payload contract for Runtime Workspace retrieval usage material."""

    model_config = ConfigDict(extra="forbid")

    used_card_short_ids: list[str] = Field(default_factory=list)
    expanded_card_short_ids: list[str] = Field(default_factory=list)
    unused_card_short_ids: list[str] = Field(default_factory=list)
    used_card_material_ids: list[str] = Field(default_factory=list)
    used_expanded_chunk_material_ids: list[str] = Field(default_factory=list)
    unused_card_material_ids: list[str] = Field(default_factory=list)
    missed_query_short_ids: list[str] = Field(default_factory=list)
    missed_query_material_ids: list[str] = Field(default_factory=list)
    knowledge_gaps: list[RetrievalKnowledgeGapItem] = Field(default_factory=list)

    @field_validator(
        "used_card_short_ids",
        "expanded_card_short_ids",
        "unused_card_short_ids",
        "used_card_material_ids",
        "used_expanded_chunk_material_ids",
        "unused_card_material_ids",
        "missed_query_short_ids",
        "missed_query_material_ids",
    )
    @classmethod
    def _normalize_text_lists(
        cls, values: list[str], info: ValidationInfo
    ) -> list[str]:
        return _normalize_text_list(values, field_name=info.field_name or "values")
