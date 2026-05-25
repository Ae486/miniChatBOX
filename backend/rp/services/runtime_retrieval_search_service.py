"""Runtime LLM-facing retrieval search service.

This service owns the writer-first RAG tool boundary: callers provide a clean
query expression, while backend policy decides which memory sources to search
and how to serialize the results for model consumption.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from rp.models.memory_contract_registry import MemoryRuntimeIdentity
from rp.models.memory_crud import (
    MemorySearchArchivalInput,
    MemorySearchRecallInput,
    RetrievalSearchResult,
)
from rp.models.retrieval_runtime_contracts import (
    RuntimeRetrievalResultItem,
    RuntimeRetrievalSearchInput,
    RuntimeRetrievalSearchOutput,
)
from rp.models.runtime_workspace_material import RuntimeWorkspaceMaterial

DEFAULT_RUNTIME_RETRIEVAL_FINAL_TOP_K = 5
DEFAULT_RUNTIME_RETRIEVAL_RELATION_TOP_K = 8
DEFAULT_RUNTIME_RETRIEVAL_SEMANTIC_TOP_K = 8
DEFAULT_RUNTIME_RETRIEVAL_COVERAGE_TOP_K = DEFAULT_RUNTIME_RETRIEVAL_RELATION_TOP_K
_ROUTE_DIRECT_HYBRID = "direct_hybrid"
_ROUTE_PLANNED_MULTI_ANCHOR_HYBRID = "planned_multi_anchor_hybrid"
_ROUTE_SEMANTIC_HYBRID = "semantic_hybrid"

_MODE_TARGET_TOP_K = {
    "entity": DEFAULT_RUNTIME_RETRIEVAL_FINAL_TOP_K,
    "relation": DEFAULT_RUNTIME_RETRIEVAL_RELATION_TOP_K,
    "semantic": DEFAULT_RUNTIME_RETRIEVAL_SEMANTIC_TOP_K,
}


class _RuntimeRetrievalCardServiceProtocol(Protocol):
    async def search_recall_to_cards(
        self,
        *,
        identity: MemoryRuntimeIdentity,
        input_model: MemorySearchRecallInput,
        actor: str,
        attempt_index: int = 1,
    ) -> tuple[
        RetrievalSearchResult,
        list[RuntimeWorkspaceMaterial],
        RuntimeWorkspaceMaterial | None,
    ]: ...

    async def search_archival_to_cards(
        self,
        *,
        identity: MemoryRuntimeIdentity,
        input_model: MemorySearchArchivalInput,
        actor: str,
        attempt_index: int = 1,
    ) -> tuple[
        RetrievalSearchResult,
        list[RuntimeWorkspaceMaterial],
        RuntimeWorkspaceMaterial | None,
    ]: ...


@dataclass(frozen=True)
class RuntimeRetrievalSearchExecution:
    """Search output plus internal Runtime Workspace material side effects."""

    output: RuntimeRetrievalSearchOutput
    materials: list[RuntimeWorkspaceMaterial] = field(default_factory=list)
    miss_materials: list[RuntimeWorkspaceMaterial] = field(default_factory=list)


@dataclass(frozen=True)
class RuntimeRetrievalQueryVariant:
    """One backend-owned query route planned before retrieval starts."""

    query_text: str
    top_k: int
    variant_kind: str
    anchor: str | None = None
    filters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeRetrievalSearchPlan:
    """Internal search plan hidden behind the LLM-facing input contract."""

    route: str
    top_k: int
    variants: list[RuntimeRetrievalQueryVariant]


@dataclass(frozen=True)
class _RankedMaterial:
    material: RuntimeWorkspaceMaterial
    route_index: int
    route_rank: int
    fusion_score: float


class RuntimeRetrievalQueryPlanner:
    """Resolve lightweight query hints into backend-owned retrieval variants."""

    def __init__(
        self,
        *,
        final_top_k: int | None,
        anchor_variant_top_k: int | None,
    ) -> None:
        self._configured_final_top_k = (
            None if final_top_k is None else max(1, int(final_top_k))
        )
        self._configured_anchor_variant_top_k = (
            None if anchor_variant_top_k is None else max(1, int(anchor_variant_top_k))
        )

    def plan(self, input_model: RuntimeRetrievalSearchInput) -> RuntimeRetrievalSearchPlan:
        normalized = RuntimeRetrievalSearchInput.model_validate(input_model)
        route = self._route(normalized)
        top_k = self._top_k_for_mode(normalized.mode)
        primary = RuntimeRetrievalQueryVariant(
            query_text=self._effective_query(normalized),
            top_k=top_k,
            variant_kind="primary",
            filters=self._filters_for_route(normalized=normalized, route=route),
        )
        variants = [primary]
        if route == _ROUTE_PLANNED_MULTI_ANCHOR_HYBRID:
            variants.extend(
                RuntimeRetrievalQueryVariant(
                    query_text=self._anchor_variant_query(
                        anchor=anchor,
                        base_query=normalized.query,
                        predicates=normalized.semantic_predicates,
                    ),
                    top_k=self._anchor_variant_top_k(top_k),
                    variant_kind="anchor",
                    anchor=anchor,
                    filters=self._filters_for_route(normalized=normalized, route=route),
                )
                for anchor in self.coverage_anchors(normalized)
            )
        return RuntimeRetrievalSearchPlan(route=route, top_k=top_k, variants=variants)

    def _top_k_for_mode(self, mode: str | None) -> int:
        if self._configured_final_top_k is not None:
            return self._configured_final_top_k
        return _MODE_TARGET_TOP_K.get(mode or "semantic", DEFAULT_RUNTIME_RETRIEVAL_SEMANTIC_TOP_K)

    def _anchor_variant_top_k(self, plan_top_k: int) -> int:
        if self._configured_anchor_variant_top_k is None:
            return plan_top_k
        return min(self._configured_anchor_variant_top_k, plan_top_k)

    @classmethod
    def _filters_for_route(
        cls,
        *,
        normalized: RuntimeRetrievalSearchInput,
        route: str,
    ) -> dict[str, Any]:
        return {
            "search_policy": {
                "hybrid": {
                    "route_weights": cls._route_weights(
                        normalized=normalized,
                        route=route,
                    )
                }
            }
        }

    @staticmethod
    def _route_weights(
        *,
        normalized: RuntimeRetrievalSearchInput,
        route: str,
    ) -> dict[str, float]:
        _ = normalized, route
        return {"retrieval.keyword": 1.0, "retrieval.semantic": 1.0}

    @classmethod
    def coverage_anchors(
        cls,
        input_model: RuntimeRetrievalSearchInput,
    ) -> list[str]:
        anchors = cls._dedupe_text(input_model.lexical_anchors)
        if len(anchors) < 2:
            return []
        return anchors[:4]

    @classmethod
    def _route(cls, input_model: RuntimeRetrievalSearchInput) -> str:
        mode = input_model.mode or "semantic"
        anchor_count = len(cls.coverage_anchors(input_model))
        if mode == "relation":
            if anchor_count >= 2:
                return _ROUTE_PLANNED_MULTI_ANCHOR_HYBRID
            return _ROUTE_DIRECT_HYBRID
        if mode == "entity":
            return _ROUTE_DIRECT_HYBRID
        return _ROUTE_SEMANTIC_HYBRID

    @classmethod
    def _effective_query(cls, input_model: RuntimeRetrievalSearchInput) -> str:
        parts = [
            input_model.query,
            *input_model.lexical_anchors,
            *input_model.semantic_predicates,
        ]
        return " ".join(cls._dedupe_text(parts))

    @classmethod
    def _anchor_variant_query(
        cls,
        *,
        anchor: str,
        base_query: str,
        predicates: Sequence[str],
    ) -> str:
        return " ".join(cls._dedupe_text([anchor, base_query, *predicates]))

    @staticmethod
    def _dedupe_text(values: Sequence[object]) -> list[str]:
        return RuntimeRetrievalSearchService._dedupe_text(values)


class RuntimeRetrievalSearchService:
    """Compose recall and archival searches behind one clean RAG output."""

    def __init__(
        self,
        *,
        runtime_retrieval_card_service: _RuntimeRetrievalCardServiceProtocol,
        final_top_k: int | None = None,
        coverage_top_k: int | None = None,
    ) -> None:
        self._runtime_retrieval_card_service = runtime_retrieval_card_service
        self._final_top_k = (
            DEFAULT_RUNTIME_RETRIEVAL_FINAL_TOP_K
            if final_top_k is None
            else max(1, int(final_top_k))
        )
        self._coverage_top_k = (
            DEFAULT_RUNTIME_RETRIEVAL_COVERAGE_TOP_K
            if coverage_top_k is None
            else max(1, int(coverage_top_k))
        )
        self._planner = RuntimeRetrievalQueryPlanner(
            final_top_k=None if final_top_k is None else self._final_top_k,
            anchor_variant_top_k=None if coverage_top_k is None else self._coverage_top_k,
        )

    async def search_for_writer(
        self,
        *,
        identity: MemoryRuntimeIdentity,
        input_model: RuntimeRetrievalSearchInput,
        actor: str,
        attempt_index: int = 1,
    ) -> RuntimeRetrievalSearchExecution:
        """Search writer-visible runtime sources and return model-clean results."""

        normalized = RuntimeRetrievalSearchInput.model_validate(input_model)
        plan = self._planner.plan(normalized)
        warnings: list[str] = []
        materials: list[RuntimeWorkspaceMaterial] = []
        primary_route_materials: list[list[RuntimeWorkspaceMaterial]] = []
        anchor_route_materials: list[list[RuntimeWorkspaceMaterial]] = []
        miss_materials: list[RuntimeWorkspaceMaterial] = []

        for variant in plan.variants:
            (
                recall_result,
                recall_cards,
                recall_miss,
            ) = await self._search_recall_cards(
                identity=identity,
                query_text=variant.query_text,
                top_k=variant.top_k,
                filters=variant.filters,
                actor=actor,
                attempt_index=attempt_index,
            )
            (
                archival_result,
                archival_cards,
                archival_miss,
            ) = await self._search_archival_cards(
                identity=identity,
                query_text=variant.query_text,
                top_k=variant.top_k,
                filters=variant.filters,
                actor=actor,
                attempt_index=attempt_index,
            )
            warnings.extend(getattr(recall_result, "warnings", []))
            warnings.extend(getattr(archival_result, "warnings", []))
            materials.extend(recall_cards)
            materials.extend(archival_cards)
            variant_route_materials = [recall_cards, archival_cards]
            if variant.variant_kind == "primary":
                primary_route_materials.extend(variant_route_materials)
            else:
                anchor_route_materials.extend(variant_route_materials)
            miss_materials.extend(
                material
                for material in (recall_miss, archival_miss)
                if material is not None
            )

        warnings = self._dedupe_text(warnings)
        ranked = self._rank_planned_materials(
            input_model=normalized,
            plan=plan,
            primary_route_materials=primary_route_materials,
            anchor_route_materials=anchor_route_materials,
        )
        results = [
            self._result_item_from_material(item.material)
            for item in ranked[: plan.top_k]
        ]
        if not results:
            warnings = self._dedupe_text([*warnings, "runtime_retrieval_no_results"])
        return RuntimeRetrievalSearchExecution(
            output=RuntimeRetrievalSearchOutput(
                query=normalized.query,
                results=results,
                warnings=warnings,
            ),
            materials=materials,
            miss_materials=miss_materials,
        )

    def _rank_planned_materials(
        self,
        *,
        input_model: RuntimeRetrievalSearchInput,
        plan: RuntimeRetrievalSearchPlan,
        primary_route_materials: list[list[RuntimeWorkspaceMaterial]],
        anchor_route_materials: list[list[RuntimeWorkspaceMaterial]],
    ) -> list[_RankedMaterial]:
        primary_ranked = self._rank_materials(route_materials=primary_route_materials)
        if plan.route != _ROUTE_PLANNED_MULTI_ANCHOR_HYBRID:
            return self._dedupe_ranked_materials(primary_ranked)
        anchor_ranked = self._rank_materials(route_materials=anchor_route_materials)
        return self._apply_anchor_coverage_guard(
            input_model=input_model,
            top_k=plan.top_k,
            primary_ranked=primary_ranked,
            anchor_ranked=anchor_ranked,
        )

    async def _search_recall_cards(
        self,
        *,
        identity: MemoryRuntimeIdentity,
        query_text: str,
        top_k: int,
        filters: dict[str, Any],
        actor: str,
        attempt_index: int,
    ) -> tuple[
        RetrievalSearchResult,
        list[RuntimeWorkspaceMaterial],
        RuntimeWorkspaceMaterial | None,
    ]:
        return await self._runtime_retrieval_card_service.search_recall_to_cards(
            identity=identity,
            input_model=MemorySearchRecallInput(
                query=query_text,
                scope="story",
                top_k=top_k,
                filters=copy.deepcopy(filters),
            ),
            actor=actor,
            attempt_index=attempt_index,
        )

    async def _search_archival_cards(
        self,
        *,
        identity: MemoryRuntimeIdentity,
        query_text: str,
        top_k: int,
        filters: dict[str, Any],
        actor: str,
        attempt_index: int,
    ) -> tuple[
        RetrievalSearchResult,
        list[RuntimeWorkspaceMaterial],
        RuntimeWorkspaceMaterial | None,
    ]:
        return await self._runtime_retrieval_card_service.search_archival_to_cards(
            identity=identity,
            input_model=MemorySearchArchivalInput(
                query=query_text,
                top_k=top_k,
                filters=copy.deepcopy(filters),
            ),
            actor=actor,
            attempt_index=attempt_index,
        )

    @classmethod
    def _coverage_anchors(
        cls,
        input_model: RuntimeRetrievalSearchInput,
    ) -> list[str]:
        return RuntimeRetrievalQueryPlanner.coverage_anchors(input_model)

    def _apply_anchor_coverage_guard(
        self,
        *,
        input_model: RuntimeRetrievalSearchInput,
        top_k: int,
        primary_ranked: list[_RankedMaterial],
        anchor_ranked: list[_RankedMaterial],
    ) -> list[_RankedMaterial]:
        primary_candidates = self._dedupe_ranked_materials(primary_ranked)
        anchor_candidates = self._dedupe_ranked_materials(anchor_ranked)
        selected = primary_candidates[:top_k]
        coverage_anchors = self._coverage_anchors(input_model)
        if len(coverage_anchors) < 2:
            return selected

        displaced: list[_RankedMaterial] = []
        for anchor in coverage_anchors:
            if self._anchor_is_covered(anchor=anchor, ranked=selected):
                continue
            candidate = self._first_anchor_candidate(
                anchor=anchor,
                candidates=anchor_candidates,
                selected=selected,
            )
            if candidate is None:
                candidate = self._first_anchor_candidate(
                    anchor=anchor,
                    candidates=primary_candidates,
                    selected=selected,
                )
            if candidate is None:
                continue
            if len(selected) >= top_k:
                displaced.append(
                    selected.pop(
                        self._replacement_index(
                            selected=selected,
                            coverage_anchors=coverage_anchors,
                        )
                    )
                )
            selected.append(candidate)
        return self._dedupe_ranked_materials(
            [
                *selected,
                *displaced,
                *primary_candidates,
                *anchor_candidates,
            ]
        )

    @classmethod
    def _first_anchor_candidate(
        cls,
        *,
        anchor: str,
        candidates: Iterable[_RankedMaterial],
        selected: Sequence[_RankedMaterial],
    ) -> _RankedMaterial | None:
        selected_keys = {cls._material_dedupe_key(item.material) for item in selected}
        for candidate in candidates:
            if cls._material_dedupe_key(candidate.material) in selected_keys:
                continue
            if cls._material_mentions_anchor(candidate.material, anchor=anchor):
                return candidate
        return None

    @classmethod
    def _replacement_index(
        cls,
        *,
        selected: Sequence[_RankedMaterial],
        coverage_anchors: Sequence[str],
    ) -> int:
        for index in range(len(selected) - 1, -1, -1):
            candidate_list = list(selected)
            candidate_list.pop(index)
            if all(
                cls._anchor_is_covered(anchor=anchor, ranked=candidate_list)
                for anchor in coverage_anchors
                if cls._anchor_is_covered(anchor=anchor, ranked=selected)
            ):
                return index
        return max(0, len(selected) - 1)

    @classmethod
    def _dedupe_ranked_materials(
        cls,
        ranked: Sequence[_RankedMaterial],
    ) -> list[_RankedMaterial]:
        deduped: list[_RankedMaterial] = []
        seen: set[str] = set()
        for item in ranked:
            key = cls._material_dedupe_key(item.material)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    @classmethod
    def _anchor_is_covered(
        cls,
        *,
        anchor: str,
        ranked: Sequence[_RankedMaterial],
    ) -> bool:
        return any(
            cls._material_mentions_anchor(item.material, anchor=anchor)
            for item in ranked
        )

    @classmethod
    def _material_mentions_anchor(
        cls,
        material: RuntimeWorkspaceMaterial,
        *,
        anchor: str,
    ) -> bool:
        normalized_anchor = anchor.strip()
        if not normalized_anchor:
            return False
        return (
            normalized_anchor.casefold()
            in " ".join(cls._material_search_texts(material)).casefold()
        )

    @classmethod
    def _material_search_texts(
        cls,
        material: RuntimeWorkspaceMaterial,
    ) -> list[str]:
        payload = dict(material.payload or {})
        metadata = payload.get("metadata")
        metadata_dict = dict(metadata) if isinstance(metadata, dict) else {}
        return cls._dedupe_text(
            [
                material.domain,
                material.domain_path,
                payload.get("title"),
                payload.get("summary"),
                payload.get("excerpt"),
                payload.get("text"),
                metadata_dict.get("entry_title"),
                metadata_dict.get("aliases"),
                metadata_dict.get("document_title"),
                metadata_dict.get("section_title"),
                metadata_dict.get("asset_title"),
                metadata_dict.get("tags"),
                metadata_dict.get("retrieval_role"),
                metadata_dict.get("semantic_path"),
                metadata_dict.get("entry_semantic_path"),
                metadata_dict.get("section_semantic_path"),
                metadata_dict.get("domain_path"),
            ]
        )

    @staticmethod
    def _material_dedupe_key(material: RuntimeWorkspaceMaterial) -> str:
        payload = dict(material.payload or {})
        metadata = payload.get("metadata")
        metadata_dict = dict(metadata) if isinstance(metadata, dict) else {}
        for value in (
            payload.get("hit_id"),
            metadata_dict.get("chunk_id"),
            metadata_dict.get("asset_id"),
        ):
            if isinstance(value, str) and value.strip():
                return value.strip()
        return str(material.material_id)

    @staticmethod
    def _rank_materials(
        *,
        route_materials: list[list[RuntimeWorkspaceMaterial]],
    ) -> list[_RankedMaterial]:
        ranked: list[_RankedMaterial] = []
        for route_index, materials in enumerate(route_materials):
            for route_rank, material in enumerate(materials, start=1):
                ranked.append(
                    _RankedMaterial(
                        material=material,
                        route_index=route_index,
                        route_rank=route_rank,
                        fusion_score=1.0 / (60.0 + route_rank),
                    )
                )
        return sorted(
            ranked,
            key=lambda item: (
                -item.fusion_score,
                item.route_index,
                item.route_rank,
                item.material.material_id,
            ),
        )

    @classmethod
    def _result_item_from_material(
        cls,
        material: RuntimeWorkspaceMaterial,
    ) -> RuntimeRetrievalResultItem:
        payload = dict(material.payload or {})
        metadata = payload.get("metadata")
        metadata_dict = dict(metadata) if isinstance(metadata, dict) else {}
        summary = cls._stored_summary(payload=payload)
        text = cls._first_text(
            payload.get("text"),
            payload.get("excerpt"),
            summary,
            payload.get("summary"),
            payload.get("title"),
            metadata_dict.get("entry_title"),
            metadata_dict.get("document_title"),
            material.domain_path,
        )
        if text is None:
            raise ValueError(
                f"Runtime retrieval material {material.material_id!r} has no text"
            )
        excerpt = None if summary is not None else cls._bounded_text(text)
        return RuntimeRetrievalResultItem(
            result_id=str(material.short_id or material.material_id),
            title=cls._first_text(
                payload.get("title"),
                metadata_dict.get("entry_title"),
                metadata_dict.get("document_title"),
            ),
            summary=summary,
            excerpt=excerpt,
            text=text,
            section=cls._first_text(
                metadata_dict.get("section_title"),
                metadata_dict.get("retrieval_role"),
            ),
        )

    @staticmethod
    def _stored_summary(*, payload: dict[str, Any]) -> str | None:
        if not payload.get("summary_source"):
            return None
        summary = payload.get("summary")
        if not isinstance(summary, str):
            return None
        return summary.strip() or None

    @staticmethod
    def _first_text(*values: object) -> str | None:
        for value in values:
            if not isinstance(value, str):
                continue
            normalized = value.strip()
            if normalized:
                return normalized
        return None

    @staticmethod
    def _bounded_text(value: str | None, *, limit: int = 600) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            return None
        if len(normalized) <= limit:
            return normalized
        return normalized[:limit].rstrip()

    @staticmethod
    def _dedupe_text(values: Sequence[object]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        pending = list(values)
        while pending:
            value = pending.pop(0)
            if isinstance(value, Mapping):
                pending[0:0] = list(value.values())
                continue
            if isinstance(value, list | tuple | set):
                pending[0:0] = list(value)
                continue
            if not isinstance(value, str):
                continue
            text = value.strip()
            if not text or text in seen:
                continue
            seen.add(text)
            normalized.append(text)
        return normalized
