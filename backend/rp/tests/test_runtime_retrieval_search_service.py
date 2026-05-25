"""Focused tests for the runtime LLM-facing retrieval search service."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rp.models.dsl import Domain
from rp.models.memory_contract_registry import MemoryRuntimeIdentity
from rp.models.memory_crud import (
    MemorySearchArchivalInput,
    MemorySearchRecallInput,
    RetrievalSearchResult,
)
from rp.models.retrieval_runtime_contracts import RuntimeRetrievalSearchInput
from rp.models.runtime_workspace_material import RuntimeWorkspaceMaterial
from rp.models.runtime_workspace_material import RuntimeWorkspaceMaterialKind
from rp.models.runtime_workspace_material import RuntimeWorkspaceMaterialVisibility
from rp.services.runtime_retrieval_search_service import RuntimeRetrievalSearchService


def _identity() -> MemoryRuntimeIdentity:
    return MemoryRuntimeIdentity(
        story_id="story-runtime-search",
        session_id="session-runtime-search",
        branch_head_id="branch-runtime-search",
        turn_id="turn-runtime-search",
        runtime_profile_snapshot_id="profile-runtime-search",
    )


def _route_weights(
    input_model: MemorySearchRecallInput | MemorySearchArchivalInput,
) -> dict[str, float]:
    return input_model.filters["search_policy"]["hybrid"]["route_weights"]


def _material(
    *,
    material_id: str,
    short_id: str,
    search_kind: str,
    title: str,
    text: str,
    rank: int,
    summary: str | None = None,
    summary_source: str | None = None,
    metadata: dict[str, object] | None = None,
) -> RuntimeWorkspaceMaterial:
    payload_metadata = {
        "section_title": "关系",
        "asset_id": "asset-hidden",
        "chunk_id": material_id,
        "score": 99,
    }
    payload_metadata.update(metadata or {})
    return RuntimeWorkspaceMaterial(
        material_id=material_id,
        material_kind=RuntimeWorkspaceMaterialKind.RETRIEVAL_CARD,
        identity=_identity(),
        domain=Domain.CHAPTER.value,
        domain_path=f"chapter.test.{short_id.lower()}",
        short_id=short_id,
        payload={
            "hit_id": material_id,
            "title": title,
            "text": text,
            "excerpt": text,
            "summary": summary,
            "summary_source": summary_source,
            "search_kind": search_kind,
            "rank": rank,
            "metadata": payload_metadata,
        },
        visibility=RuntimeWorkspaceMaterialVisibility.WRITER_VISIBLE.value,
        created_by="test",
    )


class _StubCardService:
    def __init__(self) -> None:
        self.recall_inputs: list[MemorySearchRecallInput] = []
        self.archival_inputs: list[MemorySearchArchivalInput] = []

    async def search_recall_to_cards(
        self,
        *,
        identity: MemoryRuntimeIdentity,
        input_model: MemorySearchRecallInput,
        actor: str,
        attempt_index: int = 1,
    ) -> tuple[RetrievalSearchResult, list[RuntimeWorkspaceMaterial], None]:
        _ = identity, actor, attempt_index
        self.recall_inputs.append(input_model)
        return (
            RetrievalSearchResult(
                query=input_model.query, hits=[], warnings=["recall warn"]
            ),
            [
                _material(
                    material_id="recall-card-1",
                    short_id="R1",
                    search_kind="recall",
                    title="回忆：林鸢",
                    text="林鸢在第一章救下夜紫林。",
                    rank=1,
                )
            ],
            None,
        )

    async def search_archival_to_cards(
        self,
        *,
        identity: MemoryRuntimeIdentity,
        input_model: MemorySearchArchivalInput,
        actor: str,
        attempt_index: int = 1,
    ) -> tuple[RetrievalSearchResult, list[RuntimeWorkspaceMaterial], None]:
        _ = identity, actor, attempt_index
        self.archival_inputs.append(input_model)
        return (
            RetrievalSearchResult(
                query=input_model.query, hits=[], warnings=["archival warn"]
            ),
            [
                _material(
                    material_id="archival-card-1",
                    short_id="R2",
                    search_kind="archival",
                    title="设定：林鸢与夜紫林",
                    text="林鸢与夜紫林是互相试探但彼此信任的同盟关系。",
                    rank=1,
                    summary="两人关系的已沉淀设定摘要。",
                    summary_source="entry_summary",
                )
            ],
            None,
        )


class _CoverageStubCardService:
    def __init__(self) -> None:
        self.recall_inputs: list[MemorySearchRecallInput] = []
        self.archival_inputs: list[MemorySearchArchivalInput] = []

    async def search_recall_to_cards(
        self,
        *,
        identity: MemoryRuntimeIdentity,
        input_model: MemorySearchRecallInput,
        actor: str,
        attempt_index: int = 1,
    ) -> tuple[RetrievalSearchResult, list[RuntimeWorkspaceMaterial], None]:
        _ = identity, actor, attempt_index
        self.recall_inputs.append(input_model)
        cards = [
            _material(
                material_id="recall-yakumo",
                short_id="R1",
                search_kind="recall",
                title="八云紫设定",
                text="八云紫擅长操纵境界，是幻想乡的重要角色。",
                rank=1,
            )
        ]
        if input_model.query.startswith("神绮 "):
            cards = [
                _material(
                    material_id="recall-shinki",
                    short_id="R3",
                    search_kind="recall",
                    title="神绮设定",
                    text="神绮是魔界的创造者，和八云紫的定位不同。",
                    rank=1,
                )
            ]
        return RetrievalSearchResult(query=input_model.query, hits=[]), cards, None

    async def search_archival_to_cards(
        self,
        *,
        identity: MemoryRuntimeIdentity,
        input_model: MemorySearchArchivalInput,
        actor: str,
        attempt_index: int = 1,
    ) -> tuple[RetrievalSearchResult, list[RuntimeWorkspaceMaterial], None]:
        _ = identity, actor, attempt_index
        self.archival_inputs.append(input_model)
        return (
            RetrievalSearchResult(query=input_model.query, hits=[]),
            [
                _material(
                    material_id="archival-yakumo",
                    short_id="R2",
                    search_kind="archival",
                    title="八云紫档案",
                    text="八云紫负责维护幻想乡的边界与秩序。",
                    rank=1,
                )
            ],
            None,
        )


class _PrimaryCoverageWithAnchorNoiseStubCardService:
    def __init__(self) -> None:
        self.recall_inputs: list[MemorySearchRecallInput] = []
        self.archival_inputs: list[MemorySearchArchivalInput] = []

    async def search_recall_to_cards(
        self,
        *,
        identity: MemoryRuntimeIdentity,
        input_model: MemorySearchRecallInput,
        actor: str,
        attempt_index: int = 1,
    ) -> tuple[RetrievalSearchResult, list[RuntimeWorkspaceMaterial], None]:
        _ = identity, actor, attempt_index
        self.recall_inputs.append(input_model)
        cards = [
            _material(
                material_id="primary-both",
                short_id="R1",
                search_kind="recall",
                title="角色甲与角色乙关系总览",
                text="角色甲和角色乙共同完成边境调查，互相信任但处理方式不同。",
                rank=1,
            ),
            _material(
                material_id="primary-alpha",
                short_id="R2",
                search_kind="recall",
                title="角色甲行动记录",
                text="角色甲负责正面交涉，并在调查中保护角色乙。",
                rank=2,
            ),
            _material(
                material_id="primary-beta",
                short_id="R3",
                search_kind="recall",
                title="角色乙行动记录",
                text="角色乙负责线索分析，并补足角色甲遗漏的信息。",
                rank=3,
            ),
        ]
        if input_model.query.startswith("角色甲 "):
            cards = [
                _material(
                    material_id="anchor-alpha-noise",
                    short_id="A1",
                    search_kind="recall",
                    title="角色甲单人卡",
                    text="角色甲的单人背景很完整，但没有提供角色乙相关关系信息。",
                    rank=1,
                )
            ]
        if input_model.query.startswith("角色乙 "):
            cards = [
                _material(
                    material_id="anchor-beta-noise",
                    short_id="A2",
                    search_kind="recall",
                    title="角色乙单人卡",
                    text="角色乙的单人背景很完整，但没有提供角色甲相关关系信息。",
                    rank=1,
                )
            ]
        return RetrievalSearchResult(query=input_model.query, hits=[]), cards, None

    async def search_archival_to_cards(
        self,
        *,
        identity: MemoryRuntimeIdentity,
        input_model: MemorySearchArchivalInput,
        actor: str,
        attempt_index: int = 1,
    ) -> tuple[RetrievalSearchResult, list[RuntimeWorkspaceMaterial], None]:
        _ = identity, input_model, actor, attempt_index
        self.archival_inputs.append(input_model)
        return RetrievalSearchResult(query=input_model.query, hits=[]), [], None


class _AliasCoverageStubCardService:
    def __init__(self) -> None:
        self.recall_inputs: list[MemorySearchRecallInput] = []
        self.archival_inputs: list[MemorySearchArchivalInput] = []

    async def search_recall_to_cards(
        self,
        *,
        identity: MemoryRuntimeIdentity,
        input_model: MemorySearchRecallInput,
        actor: str,
        attempt_index: int = 1,
    ) -> tuple[RetrievalSearchResult, list[RuntimeWorkspaceMaterial], None]:
        _ = identity, actor, attempt_index
        self.recall_inputs.append(input_model)
        cards = [
            _material(
                material_id="recall-reimu",
                short_id="R1",
                search_kind="recall",
                title="博丽神社巫女",
                text="博丽神社的巫女负责处理幻想乡异变。",
                rank=1,
                metadata={"aliases": ["灵梦"], "tags": ["巫女", "博丽"]},
            )
        ]
        if input_model.query.startswith("魔理沙 "):
            cards = [
                _material(
                    material_id="recall-marisa",
                    short_id="R3",
                    search_kind="recall",
                    title="普通魔法使",
                    text="魔理沙经常参与异变调查，并与灵梦形成行动对照。",
                    rank=1,
                    metadata={"aliases": ["雾雨魔理沙"], "tags": ["魔法使"]},
                )
            ]
        return RetrievalSearchResult(query=input_model.query, hits=[]), cards, None

    async def search_archival_to_cards(
        self,
        *,
        identity: MemoryRuntimeIdentity,
        input_model: MemorySearchArchivalInput,
        actor: str,
        attempt_index: int = 1,
    ) -> tuple[RetrievalSearchResult, list[RuntimeWorkspaceMaterial], None]:
        _ = identity, actor, attempt_index
        self.archival_inputs.append(input_model)
        return RetrievalSearchResult(query=input_model.query, hits=[]), [], None


class _StructuredMetadataCoverageStubCardService:
    def __init__(self) -> None:
        self.recall_inputs: list[MemorySearchRecallInput] = []
        self.archival_inputs: list[MemorySearchArchivalInput] = []

    async def search_recall_to_cards(
        self,
        *,
        identity: MemoryRuntimeIdentity,
        input_model: MemorySearchRecallInput,
        actor: str,
        attempt_index: int = 1,
    ) -> tuple[RetrievalSearchResult, list[RuntimeWorkspaceMaterial], None]:
        _ = identity, actor, attempt_index
        self.recall_inputs.append(input_model)
        cards = [
            _material(
                material_id="recall-scarlet",
                short_id="R1",
                search_kind="recall",
                title="红魔馆",
                text="洋馆内部保存了大量魔法书与契约记录。",
                rank=1,
                metadata={
                    "semantic_path": "place.红魔馆",
                    "entry_semantic_path": "world.place.红魔馆",
                    "retrieval_role": "据点",
                    "domain_path": "foundation.place.红魔馆",
                },
            )
        ]
        if input_model.query.startswith("帕秋莉 "):
            cards = [
                _material(
                    material_id="recall-patchouli",
                    short_id="R3",
                    search_kind="recall",
                    title="图书馆",
                    text="大图书馆的魔女常年研究五行与元素魔法。",
                    rank=1,
                    metadata={
                        "section_semantic_path": "character.帕秋莉.abilities",
                        "retrieval_role": "能力",
                    },
                )
            ]
        return RetrievalSearchResult(query=input_model.query, hits=[]), cards, None

    async def search_archival_to_cards(
        self,
        *,
        identity: MemoryRuntimeIdentity,
        input_model: MemorySearchArchivalInput,
        actor: str,
        attempt_index: int = 1,
    ) -> tuple[RetrievalSearchResult, list[RuntimeWorkspaceMaterial], None]:
        _ = identity, actor, attempt_index
        self.archival_inputs.append(input_model)
        return RetrievalSearchResult(query=input_model.query, hits=[]), [], None


@pytest.mark.parametrize(
    "backend_control",
    [
        {"search_kind": "archival"},
        {"top_k": 50},
        {"filters": {"source_families": ["foundation_entry"]}},
        {"rerank": True},
        {"rerank_top_n": 10},
        {"route_weights": {"keyword": 2.0, "semantic": 1.0}},
        {"candidate_top_k": 40},
    ],
)
def test_runtime_retrieval_search_input_rejects_backend_controls(
    backend_control: dict[str, object],
):
    payload = {
        "query": "林鸢和夜紫林的关系怎么样",
        **backend_control,
    }
    with pytest.raises(ValidationError):
        RuntimeRetrievalSearchInput.model_validate(payload)


def test_runtime_retrieval_search_input_schema_exposes_only_current_modes():
    schema = RuntimeRetrievalSearchInput.model_json_schema()

    assert schema["properties"]["mode"]["enum"] == [
        "entity",
        "relation",
        "semantic",
    ]


def test_runtime_retrieval_search_input_normalizes_hints():
    payload = RuntimeRetrievalSearchInput.model_validate(
        {
            "query": " 林鸢和夜紫林的关系怎么样 ",
            "mode": "entity_relation",
            "lexical_anchors": [" 林鸢 ", "夜紫林", "林鸢"],
            "semantic_predicates": ["关系", "关系", "  "],
        }
    )
    assert payload.query == "林鸢和夜紫林的关系怎么样"
    assert payload.mode == "relation"
    assert payload.lexical_anchors == ["林鸢", "夜紫林"]
    assert payload.semantic_predicates == ["关系"]


@pytest.mark.parametrize(
    "raw_mode, expected_mode",
    [
        ("entity_relation", "relation"),
        ("mixed", "semantic"),
        ("vague", "semantic"),
        (None, "semantic"),
    ],
)
def test_runtime_retrieval_search_input_normalizes_legacy_modes(
    raw_mode: str | None,
    expected_mode: str,
):
    payload = RuntimeRetrievalSearchInput.model_validate(
        {
            "query": "关系背景",
            "mode": raw_mode,
        }
    )

    assert payload.mode == expected_mode
    assert RuntimeRetrievalSearchInput.model_validate({"query": "关系背景"}).mode == "semantic"


@pytest.mark.asyncio
async def test_runtime_retrieval_search_service_merges_sources_and_cleans_output():
    card_service = _StubCardService()
    service = RuntimeRetrievalSearchService(
        runtime_retrieval_card_service=card_service,
        final_top_k=5,
    )

    execution = await service.search_for_writer(
        identity=_identity(),
        input_model=RuntimeRetrievalSearchInput(
            query="林鸢和夜紫林的关系怎么样",
            mode="entity",
            lexical_anchors=["林鸢", "夜紫林"],
            semantic_predicates=["关系"],
        ),
        actor="writer.retrieval",
    )

    assert (
        card_service.recall_inputs[0].query
        == "林鸢和夜紫林的关系怎么样 林鸢 夜紫林 关系"
    )
    assert (
        card_service.archival_inputs[0].query
        == "林鸢和夜紫林的关系怎么样 林鸢 夜紫林 关系"
    )
    assert _route_weights(card_service.recall_inputs[0]) == {
        "retrieval.keyword": 1.0,
        "retrieval.semantic": 1.0,
    }
    assert card_service.archival_inputs[0].filters == card_service.recall_inputs[0].filters
    assert card_service.archival_inputs[0].filters is not card_service.recall_inputs[0].filters
    assert [item.result_id for item in execution.output.results] == ["R1", "R2"]
    assert execution.output.results[0].summary is None
    assert execution.output.results[0].excerpt == "林鸢在第一章救下夜紫林。"
    assert execution.output.results[1].summary == "两人关系的已沉淀设定摘要。"
    assert execution.output.results[1].excerpt is None
    serialized = execution.output.model_dump(mode="json", exclude_none=True)
    result_keys = set(serialized["results"][0].keys())
    assert result_keys <= {
        "result_id",
        "title",
        "summary",
        "excerpt",
        "text",
        "section",
    }
    assert "score" not in serialized["results"][0]
    assert "metadata" not in serialized["results"][0]
    assert "search_kind" not in serialized["results"][0]
    assert "filters" not in serialized
    assert "search_policy" not in serialized
    assert execution.output.warnings == ["recall warn", "archival warn"]
    assert [material.material_id for material in execution.materials] == [
        "recall-card-1",
        "archival-card-1",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["relation", "entity_relation"])
async def test_runtime_retrieval_search_service_plans_multi_anchor_variants_upfront(
    mode: str,
):
    card_service = _CoverageStubCardService()
    service = RuntimeRetrievalSearchService(
        runtime_retrieval_card_service=card_service,
        final_top_k=2,
        coverage_top_k=1,
    )

    execution = await service.search_for_writer(
        identity=_identity(),
        input_model=RuntimeRetrievalSearchInput(
            query="八云紫和神绮有什么区别",
            mode=mode,
            lexical_anchors=["八云紫", "神绮"],
            semantic_predicates=["区别", "对比"],
        ),
        actor="writer.retrieval",
    )

    assert card_service.recall_inputs[0].query == (
        "八云紫和神绮有什么区别 八云紫 神绮 区别 对比"
    )
    assert card_service.archival_inputs[0].query == (
        "八云紫和神绮有什么区别 八云紫 神绮 区别 对比"
    )
    assert card_service.recall_inputs[1].query == (
        "八云紫 八云紫和神绮有什么区别 区别 对比"
    )
    assert card_service.archival_inputs[1].query == (
        "八云紫 八云紫和神绮有什么区别 区别 对比"
    )
    assert card_service.recall_inputs[2].query == (
        "神绮 八云紫和神绮有什么区别 区别 对比"
    )
    assert card_service.archival_inputs[2].query == (
        "神绮 八云紫和神绮有什么区别 区别 对比"
    )
    assert [item.top_k for item in card_service.recall_inputs] == [2, 1, 1]
    assert [item.top_k for item in card_service.archival_inputs] == [2, 1, 1]
    assert all(
        _route_weights(item) == {"retrieval.keyword": 1.0, "retrieval.semantic": 1.0}
        for item in [*card_service.recall_inputs, *card_service.archival_inputs]
    )
    assert [item.result_id for item in execution.output.results] == ["R1", "R3"]
    assert "八云紫" in execution.output.results[0].text
    assert "神绮" in execution.output.results[1].text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload, expected_weights",
    [
        (
            {
                "query": "八云紫和神绮有什么区别",
                "mode": "entity",
                "lexical_anchors": ["八云紫", "神绮"],
                "semantic_predicates": ["区别"],
            },
            {"retrieval.keyword": 1.0, "retrieval.semantic": 1.0},
        ),
        (
            {
                "query": "八云紫和神绮有什么区别",
                "mode": "mixed",
                "lexical_anchors": ["八云紫", "神绮"],
                "semantic_predicates": ["区别"],
            },
            {"retrieval.keyword": 1.0, "retrieval.semantic": 1.0},
        ),
        (
            {
                "query": "八云紫和神绮有什么区别",
                "mode": "entity_relation",
                "lexical_anchors": ["八云紫"],
                "semantic_predicates": ["区别"],
            },
            {"retrieval.keyword": 1.0, "retrieval.semantic": 1.0},
        ),
        (
            {
                "query": "八云紫和神绮有什么区别",
                "mode": "semantic",
                "lexical_anchors": ["八云紫", "神绮"],
                "semantic_predicates": ["区别"],
            },
            {"retrieval.keyword": 1.0, "retrieval.semantic": 1.0},
        ),
        (
            {
                "query": "八云紫和神绮有什么区别",
                "mode": "vague",
                "lexical_anchors": ["八云紫", "神绮"],
                "semantic_predicates": ["区别"],
            },
            {"retrieval.keyword": 1.0, "retrieval.semantic": 1.0},
        ),
        (
            {
                "query": "八云紫和神绮有什么区别",
                "mode": None,
                "lexical_anchors": ["八云紫", "神绮"],
                "semantic_predicates": ["区别"],
            },
            {"retrieval.keyword": 1.0, "retrieval.semantic": 1.0},
        ),
    ],
)
async def test_runtime_retrieval_search_service_does_not_plan_anchor_variants_for_non_multi_anchor_routes(
    payload: dict[str, object],
    expected_weights: dict[str, float],
):
    card_service = _CoverageStubCardService()
    service = RuntimeRetrievalSearchService(
        runtime_retrieval_card_service=card_service,
        final_top_k=2,
        coverage_top_k=1,
    )
    input_model = RuntimeRetrievalSearchInput.model_validate(payload)

    await service.search_for_writer(
        identity=_identity(),
        input_model=input_model,
        actor="writer.retrieval",
    )

    assert len(card_service.recall_inputs) == 1
    assert len(card_service.archival_inputs) == 1
    assert card_service.recall_inputs[0].query == (
        "八云紫和神绮有什么区别 八云紫 神绮 区别"
        if len(input_model.lexical_anchors) == 2
        else "八云紫和神绮有什么区别 八云紫 区别"
    )
    assert _route_weights(card_service.recall_inputs[0]) == expected_weights
    assert _route_weights(card_service.archival_inputs[0]) == expected_weights


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode, expected_weights",
    [
        ("mixed", {"retrieval.keyword": 1.0, "retrieval.semantic": 1.0}),
        ("semantic", {"retrieval.keyword": 1.0, "retrieval.semantic": 1.0}),
        ("vague", {"retrieval.keyword": 1.0, "retrieval.semantic": 1.0}),
        (None, {"retrieval.keyword": 1.0, "retrieval.semantic": 1.0}),
    ],
)
async def test_runtime_retrieval_search_service_uses_mode_aware_backend_weights(
    mode: str | None,
    expected_weights: dict[str, float],
):
    card_service = _CoverageStubCardService()
    service = RuntimeRetrievalSearchService(
        runtime_retrieval_card_service=card_service,
        final_top_k=2,
        coverage_top_k=1,
    )

    execution = await service.search_for_writer(
        identity=_identity(),
        input_model=RuntimeRetrievalSearchInput.model_validate(
            {
                "query": "八云紫的背景",
                "mode": mode,
                "lexical_anchors": ["八云紫"],
                "semantic_predicates": ["背景"],
            }
        ),
        actor="writer.retrieval",
    )

    assert len(card_service.recall_inputs) == 1
    assert len(card_service.archival_inputs) == 1
    assert _route_weights(card_service.recall_inputs[0]) == expected_weights
    assert _route_weights(card_service.archival_inputs[0]) == expected_weights
    serialized = execution.output.model_dump(mode="json", exclude_none=True)
    assert "search_policy" not in serialized
    assert all("search_policy" not in item for item in serialized["results"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload, expected_top_ks",
    [
        (
            {
                "query": "八云紫的背景",
                "mode": "entity",
                "lexical_anchors": ["八云紫", "神绮"],
                "semantic_predicates": ["背景"],
            },
            [5],
        ),
        (
            {
                "query": "八云紫和神绮有什么区别",
                "mode": "relation",
                "lexical_anchors": ["八云紫", "神绮"],
                "semantic_predicates": ["区别"],
            },
            [8, 8, 8],
        ),
        (
            {
                "query": "八云紫和神绮有什么区别",
                "mode": "semantic",
                "lexical_anchors": ["八云紫", "神绮"],
                "semantic_predicates": ["区别"],
            },
            [8],
        ),
        (
            {
                "query": "八云紫和神绮有什么区别",
                "mode": "mixed",
                "lexical_anchors": ["八云紫", "神绮"],
                "semantic_predicates": ["区别"],
            },
            [8],
        ),
    ],
)
async def test_runtime_retrieval_search_service_uses_mode_aware_top_k(
    payload: dict[str, object],
    expected_top_ks: list[int],
):
    card_service = _CoverageStubCardService()
    service = RuntimeRetrievalSearchService(
        runtime_retrieval_card_service=card_service,
    )

    await service.search_for_writer(
        identity=_identity(),
        input_model=RuntimeRetrievalSearchInput.model_validate(payload),
        actor="writer.retrieval",
    )

    assert [item.top_k for item in card_service.recall_inputs] == expected_top_ks
    assert [item.top_k for item in card_service.archival_inputs] == expected_top_ks


@pytest.mark.asyncio
async def test_runtime_retrieval_search_service_caps_planned_anchor_variants_at_four():
    card_service = _CoverageStubCardService()
    service = RuntimeRetrievalSearchService(
        runtime_retrieval_card_service=card_service,
        final_top_k=2,
        coverage_top_k=5,
    )

    await service.search_for_writer(
        identity=_identity(),
        input_model=RuntimeRetrievalSearchInput(
            query="多角色关系对照",
            mode="relation",
            lexical_anchors=["八云紫", "神绮", "灵梦", "魔理沙", "帕秋莉"],
            semantic_predicates=["关系"],
        ),
        actor="writer.retrieval",
    )

    assert len(card_service.recall_inputs) == 5
    assert len(card_service.archival_inputs) == 5
    assert [item.top_k for item in card_service.recall_inputs] == [2, 2, 2, 2, 2]
    assert all(
        _route_weights(item) == {"retrieval.keyword": 1.0, "retrieval.semantic": 1.0}
        for item in [*card_service.recall_inputs, *card_service.archival_inputs]
    )
    assert [item.query for item in card_service.recall_inputs[1:]] == [
        "八云紫 多角色关系对照 关系",
        "神绮 多角色关系对照 关系",
        "灵梦 多角色关系对照 关系",
        "魔理沙 多角色关系对照 关系",
    ]


@pytest.mark.asyncio
async def test_runtime_retrieval_search_service_planned_multi_anchor_runs_even_when_primary_covers_anchors():
    card_service = _StubCardService()
    service = RuntimeRetrievalSearchService(
        runtime_retrieval_card_service=card_service,
        final_top_k=2,
        coverage_top_k=1,
    )

    execution = await service.search_for_writer(
        identity=_identity(),
        input_model=RuntimeRetrievalSearchInput(
            query="林鸢和夜紫林的关系怎么样",
            mode="relation",
            lexical_anchors=["林鸢", "夜紫林"],
            semantic_predicates=["关系"],
        ),
        actor="writer.retrieval",
    )

    assert len(card_service.recall_inputs) == 3
    assert len(card_service.archival_inputs) == 3
    assert card_service.recall_inputs[1].query == (
        "林鸢 林鸢和夜紫林的关系怎么样 关系"
    )
    assert card_service.archival_inputs[2].query == (
        "夜紫林 林鸢和夜紫林的关系怎么样 关系"
    )
    assert [item.result_id for item in execution.output.results] == ["R1", "R2"]


@pytest.mark.asyncio
async def test_runtime_retrieval_search_service_keeps_primary_topk_when_it_covers_anchors():
    card_service = _PrimaryCoverageWithAnchorNoiseStubCardService()
    service = RuntimeRetrievalSearchService(
        runtime_retrieval_card_service=card_service,
        final_top_k=3,
        coverage_top_k=1,
    )

    execution = await service.search_for_writer(
        identity=_identity(),
        input_model=RuntimeRetrievalSearchInput(
            query="角色甲和角色乙的调查分工",
            mode="relation",
            lexical_anchors=["角色甲", "角色乙"],
            semantic_predicates=["调查分工"],
        ),
        actor="writer.retrieval",
    )

    assert len(card_service.recall_inputs) == 3
    assert len(card_service.archival_inputs) == 3
    assert [item.result_id for item in execution.output.results] == [
        "R1",
        "R2",
        "R3",
    ]


@pytest.mark.asyncio
async def test_runtime_retrieval_search_service_uses_alias_and_tag_metadata_for_planned_coverage_ranking():
    card_service = _AliasCoverageStubCardService()
    service = RuntimeRetrievalSearchService(
        runtime_retrieval_card_service=card_service,
        final_top_k=2,
        coverage_top_k=1,
    )

    execution = await service.search_for_writer(
        identity=_identity(),
        input_model=RuntimeRetrievalSearchInput(
            query="灵梦和魔理沙的行动差异",
            mode="relation",
            lexical_anchors=["灵梦", "魔理沙"],
            semantic_predicates=["行动差异"],
        ),
        actor="writer.retrieval",
    )

    assert len(card_service.recall_inputs) == 3
    assert len(card_service.archival_inputs) == 3
    assert card_service.recall_inputs[1].query == (
        "灵梦 灵梦和魔理沙的行动差异 行动差异"
    )
    assert card_service.recall_inputs[2].query == (
        "魔理沙 灵梦和魔理沙的行动差异 行动差异"
    )
    assert [item.result_id for item in execution.output.results] == ["R1", "R3"]


@pytest.mark.asyncio
async def test_runtime_retrieval_search_service_uses_structured_metadata_for_planned_coverage_ranking():
    card_service = _StructuredMetadataCoverageStubCardService()
    service = RuntimeRetrievalSearchService(
        runtime_retrieval_card_service=card_service,
        final_top_k=2,
        coverage_top_k=1,
    )

    execution = await service.search_for_writer(
        identity=_identity(),
        input_model=RuntimeRetrievalSearchInput(
            query="红魔馆和帕秋莉的关联",
            mode="relation",
            lexical_anchors=["红魔馆", "帕秋莉"],
            semantic_predicates=["关联"],
        ),
        actor="writer.retrieval",
    )

    assert len(card_service.recall_inputs) == 3
    assert len(card_service.archival_inputs) == 3
    assert card_service.recall_inputs[1].query == ("红魔馆 红魔馆和帕秋莉的关联 关联")
    assert card_service.recall_inputs[2].query == ("帕秋莉 红魔馆和帕秋莉的关联 关联")
    assert [item.result_id for item in execution.output.results] == ["R1", "R3"]
