"""Tests for retrieval-core search service."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from rp.models.dsl import Domain
from rp.models.memory_crud import (
    RetrievalHit,
    RetrievalQuery,
    RetrievalSearchResult,
    RetrievalTrace,
)
from rp.models.retrieval_records import SourceAsset
from rp.models.setup_workspace import StoryMode
from rp.retrieval.keyword_retriever import KeywordRetriever
from rp.retrieval.query_preprocessor import DefaultQueryPreprocessor
from rp.retrieval.rrf_fusion import reciprocal_rank_fusion
from rp.retrieval.rag_context_builder import RagContextBuilder
from rp.retrieval.reranker import SimpleMetadataReranker
from rp.services.retrieval_collection_service import RetrievalCollectionService
from rp.services.retrieval_document_service import RetrievalDocumentService
from rp.services.retrieval_ingestion_service import RetrievalIngestionService
from rp.services.retrieval_service import RetrievalService


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def test_default_query_preprocessor_normalizes_recall_filter_values():
    query = RetrievalQuery(
        query_id="rq-filter-normalize",
        query_kind="recall",
        story_id=" story-filter ",
        domains=[Domain.CHAPTER, Domain.CHAPTER],
        text_query=" filter text ",
        filters={
            "materialization_kinds": [
                " chapter_summary ",
                "",
                "chapter_summary",
                "continuity_note",
            ],
            "source_families": [
                " longform_story_runtime ",
                "longform_story_runtime",
                "alternate_source_family",
            ],
            "chapter_indices": [1, "2", " 2 ", "bad", None, True, "  "],
        },
        top_k=0,
    )

    normalized = DefaultQueryPreprocessor().preprocess(query)

    assert normalized.filters["materialization_kinds"] == [
        "chapter_summary",
        "continuity_note",
    ]
    assert normalized.filters["source_families"] == [
        "longform_story_runtime",
        "alternate_source_family",
    ]
    assert normalized.filters["chapter_indices"] == [1, 2]
    assert normalized.story_id == "story-filter"
    assert normalized.text_query == "filter text"
    assert normalized.domains == [Domain.CHAPTER]
    assert normalized.top_k == 1


def test_default_query_preprocessor_normalizes_narrative_and_archival_filters():
    query = RetrievalQuery(
        query_id="rq-narrative-filter-normalize",
        query_kind="recall",
        story_id="story-filter",
        text_query="filter text",
        filters={
            "scene_refs": [
                " chapter:1:scene:1 ",
                "",
                "chapter:1:scene:1",
                None,
            ],
            "character_refs": [" hero ", "villain", "hero", True],
            "pov_character_refs": [" hero "],
            "foreshadow_refs": [" clue-a "],
            "foreshadow_statuses": [" open ", "open"],
            "branch_ids": [" main "],
            "canon_statuses": [" canonical ", "draft"],
            "source_types": [" foundation_entry ", "foundation_entry"],
            "source_origins": [" setup_workspace "],
            "workspace_ids": [" workspace-1 "],
            "commit_ids": [" commit-1 "],
            "search_policy": {"profile": " LongForm ", "rerank": " ON "},
        },
    )

    normalized = DefaultQueryPreprocessor().preprocess(query)

    assert normalized.filters["scene_refs"] == ["chapter:1:scene:1"]
    assert normalized.filters["character_refs"] == ["hero", "villain"]
    assert normalized.filters["pov_character_refs"] == ["hero"]
    assert normalized.filters["foreshadow_refs"] == ["clue-a"]
    assert normalized.filters["foreshadow_statuses"] == ["open"]
    assert normalized.filters["branch_ids"] == ["main"]
    assert normalized.filters["canon_statuses"] == ["canonical", "draft"]
    assert normalized.filters["source_types"] == ["foundation_entry"]
    assert normalized.filters["source_origins"] == ["setup_workspace"]
    assert normalized.filters["workspace_ids"] == ["workspace-1"]
    assert normalized.filters["commit_ids"] == ["commit-1"]
    assert normalized.filters["search_policy"] == {
        "profile": "longform",
        "rerank": "on",
    }


def test_default_query_preprocessor_adds_structured_query_analysis():
    query = RetrievalQuery(
        query_id="rq-query-analysis",
        query_kind="archival",
        story_id="story-query-analysis",
        text_query="林鸢和夜紫林的关系",
        filters={},
    )

    normalized = DefaultQueryPreprocessor().preprocess(query)

    analysis = normalized.filters["query_analysis"]
    assert analysis["version"] == "structured_query_analysis_v1"
    assert analysis["intent"] == "relationship"
    assert analysis["entity_terms"] == ["林鸢", "夜紫林"]
    assert "关系" in analysis["intent_terms"]


def test_rrf_fusion_supports_route_weight_hints_without_leaking_metadata():
    fused = reciprocal_rank_fusion(
        [
            [
                {
                    "hit_id": "keyword-hit",
                    "query_id": "rq-weighted-rrf",
                    "layer": "archival",
                    "domain": "world_rule",
                    "domain_path": "world.keyword",
                    "excerpt_text": "keyword hit",
                    "score": 0.3,
                    "rank": 1,
                    "metadata": {},
                    "_rrf_weight": 3.0,
                }
            ],
            [
                {
                    "hit_id": "dense-hit",
                    "query_id": "rq-weighted-rrf",
                    "layer": "archival",
                    "domain": "world_rule",
                    "domain_path": "world.dense",
                    "excerpt_text": "dense hit",
                    "score": 0.9,
                    "rank": 1,
                    "metadata": {},
                }
            ],
        ]
    )

    assert fused[0]["hit_id"] == "keyword-hit"
    assert "_rrf_weight" not in fused[0]


def test_keyword_retriever_bypasses_postgres_simple_fts_for_structured_sparse_queries(
    retrieval_session,
):
    query = DefaultQueryPreprocessor().preprocess(
        RetrievalQuery(
            query_id="rq-pg-sparse-parity",
            query_kind="archival",
            story_id="story-pg-sparse-parity",
            text_query="林鸢和夜紫林的关系",
            filters={},
        )
    )

    assert KeywordRetriever._should_use_python_sparse_path(query) is True


def test_keyword_retriever_keeps_postgres_fts_available_for_plain_ascii_queries():
    query = RetrievalQuery(
        query_id="rq-pg-plain-fts",
        query_kind="archival",
        story_id="story-pg-plain-fts",
        text_query="ledger room sunrise",
        filters={},
    )

    assert KeywordRetriever._should_use_python_sparse_path(query) is False


class _FakeLangfuseObservation:
    def __init__(self, *, sink: list[dict], name: str) -> None:
        self._sink = sink
        self._name = name

    def __enter__(self):
        self._sink.append({"kind": "observation_enter", "name": self._name})
        return self

    def __exit__(self, exc_type, exc, tb):
        self._sink.append({"kind": "observation_exit", "name": self._name})
        return False

    def update(self, **kwargs):
        self._sink.append(
            {"kind": "observation_update", "name": self._name, "payload": kwargs}
        )

    def score(self, **kwargs):
        self._sink.append({"kind": "score", "name": self._name, "payload": kwargs})

    def score_trace(self, **kwargs):
        self._sink.append(
            {"kind": "score_trace", "name": self._name, "payload": kwargs}
        )

    def start_as_current_observation(self, **kwargs):
        return _FakeLangfuseObservation(
            sink=self._sink,
            name=str(kwargs.get("name") or "unknown"),
        )


class _FakeLangfuseService:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def start_as_current_observation(self, **kwargs):
        return _FakeLangfuseObservation(
            sink=self.events,
            name=str(kwargs.get("name") or "unknown"),
        )


@pytest.mark.asyncio
async def test_search_chunks_and_documents_use_real_store(retrieval_session):
    collection = RetrievalCollectionService(retrieval_session).ensure_story_collection(
        story_id="story-search",
        scope="story",
        collection_kind="archival",
    )
    RetrievalDocumentService(retrieval_session).upsert_source_asset(
        SourceAsset(
            asset_id="asset-search",
            story_id="story-search",
            mode=StoryMode.LONGFORM,
            collection_id=collection.collection_id,
            commit_id="commit-search",
            asset_kind="worldbook",
            source_ref="memory://asset-search",
            title="Archive Ledger",
            parse_status="queued",
            ingestion_status="queued",
            mapped_targets=["foundation"],
            metadata={
                "seed_sections": [
                    {
                        "section_id": "rule-1",
                        "title": "Archive Rule",
                        "path": "foundation.world.archive_rule",
                        "level": 1,
                        "page_no": 7,
                        "page_label": "VII",
                        "image_caption": "Diagram of the archive ledger seal.",
                        "text": "Archive keepers seal the ledger room before sunrise.",
                        "metadata": {
                            "domain": "world_rule",
                            "domain_path": "foundation.world.archive_rule",
                            "tags": ["archive"],
                        },
                    }
                ]
            },
            created_at=_utcnow(),
            updated_at=_utcnow(),
        )
    )
    retrieval_session.flush()
    RetrievalIngestionService(retrieval_session).ingest_asset(
        story_id="story-search",
        asset_id="asset-search",
        collection_id=collection.collection_id,
    )
    retrieval_session.commit()

    service = RetrievalService(retrieval_session)
    query = RetrievalQuery(
        query_id="rq-search",
        query_kind="archival",
        story_id="story-search",
        domains=[Domain.WORLD_RULE],
        text_query="ledger room sunrise",
        filters={"knowledge_collections": [collection.collection_id]},
        top_k=3,
    )

    chunk_result = await service.search_chunks(query)
    document_result = await service.search_documents(query)
    rag_result = await service.rag_context(query)

    assert chunk_result.hits
    assert chunk_result.trace is not None
    assert chunk_result.trace.route.startswith("retrieval.")
    assert chunk_result.trace.result_kind == "chunk"
    assert chunk_result.trace.retriever_routes
    assert "chunk_result_builder" in chunk_result.trace.pipeline_stages
    assert chunk_result.trace.filters_applied["rerank"] is False
    assert chunk_result.hits[0].metadata["asset_id"] == "asset-search"
    assert chunk_result.hits[0].metadata["collection_id"] == collection.collection_id
    assert chunk_result.hits[0].metadata["source_ref"] == "memory://asset-search"
    assert chunk_result.hits[0].metadata["commit_id"] == "commit-search"
    assert chunk_result.hits[0].metadata["section_id"] == "rule-1"
    assert chunk_result.hits[0].metadata["section_part"] == 0
    assert chunk_result.hits[0].metadata["domain"] == "world_rule"
    assert (
        chunk_result.hits[0].metadata["domain_path"] == "foundation.world.archive_rule"
    )
    assert chunk_result.hits[0].metadata["document_title"] == "Archive Ledger"
    assert chunk_result.hits[0].metadata["document_summary"]
    assert chunk_result.hits[0].metadata["page_no"] == 7
    assert chunk_result.hits[0].metadata["page_label"] == "VII"
    assert chunk_result.hits[0].metadata["page_ref"] == "VII (7)"
    assert (
        chunk_result.hits[0].metadata["image_caption"]
        == "Diagram of the archive ledger seal."
    )
    assert chunk_result.hits[0].metadata["context_header"]
    assert chunk_result.hits[0].metadata["contextual_text_version"] == "v2"
    assert "Page: VII (7)" in chunk_result.hits[0].metadata["contextual_text"]
    assert (
        "Image: Diagram of the archive ledger seal."
        in chunk_result.hits[0].metadata["contextual_text"]
    )
    assert chunk_result.hits[0].hit_id.startswith("chunk_")
    assert document_result.hits
    assert document_result.hits[0].metadata["result_kind"] == "document"
    assert document_result.hits[0].hit_id == "doc:asset-search"
    assert document_result.trace is not None
    assert document_result.trace.result_kind == "document"
    assert document_result.trace.retriever_routes
    assert "document_result_builder" in document_result.trace.pipeline_stages
    assert rag_result.hits
    assert rag_result.hits[0].metadata["result_kind"] == "rag"
    assert rag_result.hits[0].metadata["rag_context_header"]
    assert rag_result.hits[0].metadata["rag_document_summary"]
    assert rag_result.hits[0].excerpt_text.startswith("Context:")
    assert "Page: VII (7)" in rag_result.hits[0].excerpt_text
    assert (
        "Image: Diagram of the archive ledger seal." in rag_result.hits[0].excerpt_text
    )
    assert rag_result.trace is not None
    assert rag_result.trace.result_kind == "rag"
    assert "rag_context_builder" in rag_result.trace.pipeline_stages


@pytest.mark.asyncio
async def test_keyword_retriever_python_fallback_uses_bm25_sparse_ranking(retrieval_session):
    collection = RetrievalCollectionService(retrieval_session).ensure_story_collection(
        story_id="story-bm25",
        scope="story",
        collection_kind="archival",
    )
    document_service = RetrievalDocumentService(retrieval_session)
    for asset_id, title, text in (
        (
            "asset-bm25-exact",
            "Rare Anchor Exact",
            "The archive keeps the rare anchor ledger behind the dawn seal. "
            "Appendix material lists archive archive archive archive archive "
            "inventory inventory inventory inventory inventory catalog catalog "
            "catalog catalog catalog without repeating the anchor ledger pair.",
        ),
        (
            "asset-bm25-partial",
            "Rare Partial",
            "Rare rare rare archive inventory notes about shelves and catalog cards.",
        ),
        (
            "asset-bm25-noise",
            "Noise",
            "Kitchen inventory and garden weather notes.",
        ),
    ):
        document_service.upsert_source_asset(
            SourceAsset(
                asset_id=asset_id,
                story_id="story-bm25",
                mode=StoryMode.LONGFORM,
                collection_id=collection.collection_id,
                commit_id=f"commit-{asset_id}",
                asset_kind="worldbook",
                source_ref=f"memory://{asset_id}",
                title=title,
                parse_status="queued",
                ingestion_status="queued",
                mapped_targets=["foundation"],
                metadata={
                    "seed_sections": [
                        {
                            "section_id": asset_id,
                            "title": title,
                            "path": f"foundation.world.{asset_id}",
                            "level": 1,
                            "text": text,
                            "metadata": {
                                "domain": "world_rule",
                                "domain_path": f"foundation.world.{asset_id}",
                            },
                        }
                    ]
                },
                created_at=_utcnow(),
                updated_at=_utcnow(),
            )
        )
        retrieval_session.flush()
        RetrievalIngestionService(retrieval_session).ingest_asset(
            story_id="story-bm25",
            asset_id=asset_id,
            collection_id=collection.collection_id,
        )
    retrieval_session.commit()

    result = await KeywordRetriever(retrieval_session).search(
        RetrievalQuery(
            query_id="rq-bm25",
            query_kind="archival",
            story_id="story-bm25",
            domains=[Domain.WORLD_RULE],
            text_query="rare anchor ledger",
            filters={"knowledge_collections": [collection.collection_id]},
            top_k=3,
        )
    )

    assert result.trace is not None
    assert result.trace.route == "retrieval.keyword.bm25"
    assert result.hits[0].metadata["asset_id"] == "asset-bm25-exact"
    assert result.hits[0].score > result.hits[1].score


@pytest.mark.asyncio
async def test_keyword_retriever_python_fallback_supports_chinese_field_boosts(retrieval_session):
    collection = RetrievalCollectionService(retrieval_session).ensure_story_collection(
        story_id="story-cjk",
        scope="story",
        collection_kind="archival",
    )
    document_service = RetrievalDocumentService(retrieval_session)
    document_service.upsert_source_asset(
        SourceAsset(
            asset_id="asset-cjk-target",
            story_id="story-cjk",
            mode=StoryMode.LONGFORM,
            collection_id=collection.collection_id,
            commit_id="commit-cjk-target",
            asset_kind="worldbook",
            source_ref="memory://asset-cjk-target",
            title="林鸢",
            parse_status="queued",
            ingestion_status="queued",
            mapped_targets=["foundation"],
            metadata={
                "seed_sections": [
                    {
                        "section_id": "relationship",
                        "title": "关系",
                        "path": "character_design.character.lin_yuan.relationship",
                        "level": 1,
                        "text": "两人曾在旧城区共同调查遗失档案，后来成为互相信任的搭档。",
                        "metadata": {
                            "domain": "character",
                            "domain_path": "character_design.character.lin_yuan.relationship",
                            "entry_title": "林鸢",
                            "aliases": ["林鸢", "林鸢儿"],
                            "section_title": "关系",
                            "retrieval_role": "relationship",
                            "section_semantic_path": "character_design.character.lin_yuan.relationship",
                        },
                        "tags": ["夜紫林", "关系"],
                    }
                ]
            },
            created_at=_utcnow(),
            updated_at=_utcnow(),
        )
    )
    document_service.upsert_source_asset(
        SourceAsset(
            asset_id="asset-cjk-noise",
            story_id="story-cjk",
            mode=StoryMode.LONGFORM,
            collection_id=collection.collection_id,
            commit_id="commit-cjk-noise",
            asset_kind="worldbook",
            source_ref="memory://asset-cjk-noise",
            title="旧城区档案馆",
            parse_status="queued",
            ingestion_status="queued",
            mapped_targets=["foundation"],
            metadata={
                "seed_sections": [
                    {
                        "section_id": "summary",
                        "title": "概要",
                        "path": "world_background.location.archive",
                        "level": 1,
                        "text": "旧城区档案馆保存许多遗失档案，调查者经常在这里寻找线索。",
                        "metadata": {
                            "domain": "world_rule",
                            "domain_path": "world_background.location.archive",
                            "section_title": "概要",
                            "retrieval_role": "summary",
                        },
                        "tags": ["调查", "档案"],
                    }
                ]
            },
            created_at=_utcnow(),
            updated_at=_utcnow(),
        )
    )
    retrieval_session.flush()
    ingestion_service = RetrievalIngestionService(retrieval_session)
    ingestion_service.ingest_asset(
        story_id="story-cjk",
        asset_id="asset-cjk-target",
        collection_id=collection.collection_id,
    )
    ingestion_service.ingest_asset(
        story_id="story-cjk",
        asset_id="asset-cjk-noise",
        collection_id=collection.collection_id,
    )
    retrieval_session.commit()

    result = await KeywordRetriever(retrieval_session).search(
        RetrievalQuery(
            query_id="rq-cjk",
            query_kind="archival",
            story_id="story-cjk",
            text_query="林鸢和夜紫林的关系",
            filters={"knowledge_collections": [collection.collection_id]},
            top_k=2,
        )
    )

    assert result.trace is not None
    assert result.trace.route == "retrieval.keyword.bm25"
    assert result.hits[0].metadata["asset_id"] == "asset-cjk-target"
    assert result.hits[0].metadata["retrieval_role"] == "relationship"


@pytest.mark.asyncio
async def test_keyword_retriever_python_bm25_uses_existing_title_alias_tag_fields(
    retrieval_session,
):
    collection = RetrievalCollectionService(retrieval_session).ensure_story_collection(
        story_id="story-existing-fields",
        scope="story",
        collection_kind="archival",
    )
    document_service = RetrievalDocumentService(retrieval_session)
    document_service.upsert_source_asset(
        SourceAsset(
            asset_id="asset-existing-fields-target",
            story_id="story-existing-fields",
            mode=StoryMode.LONGFORM,
            collection_id=collection.collection_id,
            commit_id="commit-existing-fields-target",
            asset_kind="worldbook",
            source_ref="memory://asset-existing-fields-target",
            title="Generic Source",
            parse_status="queued",
            ingestion_status="queued",
            mapped_targets=["foundation"],
            metadata={
                "asset_title": "Moon Registry",
                "document_title": "Silver Archive",
                "aliases": {"asset": ["Asset Lantern"]},
                "tags": {"asset": ["lunar"]},
                "domain_path": {"asset": "celestial.path"},
                "seed_sections": [
                    {
                        "section_id": "field-target",
                        "title": "Plain Section",
                        "path": "foundation.world.existing_fields.target",
                        "level": 1,
                        "text": "A neutral passage that does not repeat the searchable labels.",
                        "metadata": {
                            "domain": "world_rule",
                            "domain_path": {
                                "root": "foundation",
                                "leaf": "silver.registry",
                            },
                            "document_title": "Silver Archive",
                            "asset_title": "Moon Registry",
                            "aliases": ("Lantern Alias",),
                            "tags": {"primary": ["ritual", "beacon"]},
                        },
                    }
                ],
            },
            created_at=_utcnow(),
            updated_at=_utcnow(),
        )
    )
    document_service.upsert_source_asset(
        SourceAsset(
            asset_id="asset-existing-fields-noise",
            story_id="story-existing-fields",
            mode=StoryMode.LONGFORM,
            collection_id=collection.collection_id,
            commit_id="commit-existing-fields-noise",
            asset_kind="worldbook",
            source_ref="memory://asset-existing-fields-noise",
            title="Generic Noise",
            parse_status="queued",
            ingestion_status="queued",
            mapped_targets=["foundation"],
            metadata={
                "seed_sections": [
                    {
                        "section_id": "field-noise",
                        "title": "Plain Section",
                        "path": "foundation.world.existing_fields.noise",
                        "level": 1,
                        "text": "A neutral passage about shelves and weather.",
                        "metadata": {
                            "domain": "world_rule",
                            "domain_path": "foundation.world.existing_fields.noise",
                        },
                    }
                ],
            },
            created_at=_utcnow(),
            updated_at=_utcnow(),
        )
    )
    retrieval_session.flush()
    ingestion_service = RetrievalIngestionService(retrieval_session)
    ingestion_service.ingest_asset(
        story_id="story-existing-fields",
        asset_id="asset-existing-fields-target",
        collection_id=collection.collection_id,
    )
    ingestion_service.ingest_asset(
        story_id="story-existing-fields",
        asset_id="asset-existing-fields-noise",
        collection_id=collection.collection_id,
    )
    retrieval_session.commit()

    result = await KeywordRetriever(retrieval_session).search(
        RetrievalQuery(
            query_id="rq-existing-fields",
            query_kind="archival",
            story_id="story-existing-fields",
            text_query=(
                "silver registry lantern alias ritual beacon "
                "asset lantern lunar celestial"
            ),
            filters={"knowledge_collections": [collection.collection_id]},
            top_k=2,
        )
    )

    assert result.hits
    assert result.hits[0].metadata["asset_id"] == "asset-existing-fields-target"


@pytest.mark.asyncio
async def test_keyword_retriever_structured_multiplier_uses_existing_metadata_fields(
    retrieval_session,
):
    collection = RetrievalCollectionService(retrieval_session).ensure_story_collection(
        story_id="story-structured-metadata-fields",
        scope="story",
        collection_kind="archival",
    )
    document_service = RetrievalDocumentService(retrieval_session)
    for asset_id, metadata in (
        (
            "asset-structured-fields-target",
            {
                "domain": "world_rule",
                "domain_path": {"root": "foundation", "leaf": "atlas.beacon"},
                "document_title": "Atlas Beacon",
                "asset_title": "Beacon Register",
                "tags": ["ritual"],
            },
        ),
        (
            "asset-structured-fields-noise",
            {
                "domain": "world_rule",
                "domain_path": "foundation.world.generic",
                "document_title": "Generic Register",
                "tags": ["misc"],
            },
        ),
    ):
        document_service.upsert_source_asset(
            SourceAsset(
                asset_id=asset_id,
                story_id="story-structured-metadata-fields",
                mode=StoryMode.LONGFORM,
                collection_id=collection.collection_id,
                commit_id=f"commit-{asset_id}",
                asset_kind="worldbook",
                source_ref=f"memory://{asset_id}",
                title="Shared Source",
                parse_status="queued",
                ingestion_status="queued",
                mapped_targets=["foundation"],
                metadata={
                    "seed_sections": [
                        {
                            "section_id": asset_id,
                            "title": "Shared Section",
                            "path": f"foundation.world.{asset_id}",
                            "level": 1,
                            "text": "shared neutral evidence",
                            "metadata": metadata,
                        }
                    ]
                },
                created_at=_utcnow(),
                updated_at=_utcnow(),
            )
        )
        retrieval_session.flush()
        RetrievalIngestionService(retrieval_session).ingest_asset(
            story_id="story-structured-metadata-fields",
            asset_id=asset_id,
            collection_id=collection.collection_id,
        )
    retrieval_session.commit()

    result = await KeywordRetriever(retrieval_session).search(
        RetrievalQuery(
            query_id="rq-structured-metadata-fields",
            query_kind="archival",
            story_id="story-structured-metadata-fields",
            text_query="shared neutral",
            filters={
                "knowledge_collections": [collection.collection_id],
                "query_analysis": {
                    "entity_terms": ["atlas"],
                    "intent_terms": ["ritual"],
                },
            },
            top_k=2,
        )
    )

    assert result.hits[0].metadata["asset_id"] == "asset-structured-fields-target"


@pytest.mark.asyncio
async def test_retrieval_service_uses_explicit_pipeline_slots(retrieval_session):
    class StubPreprocessor:
        def __init__(self) -> None:
            self.seen_queries: list[RetrievalQuery] = []

        def preprocess(self, query: RetrievalQuery) -> RetrievalQuery:
            self.seen_queries.append(query)
            return query.model_copy(update={"text_query": "normalized slot query"})

    class StubRetriever:
        def __init__(self, *, route: str, score: float) -> None:
            self.route = route
            self.score = score
            self.seen_queries: list[RetrievalQuery] = []

        async def search(self, query: RetrievalQuery) -> RetrievalSearchResult:
            self.seen_queries.append(query)
            return RetrievalSearchResult(
                query=query.text_query or "",
                hits=[
                    RetrievalHit(
                        hit_id=f"{self.route}:hit",
                        query_id=query.query_id,
                        layer="archival",
                        domain=Domain.WORLD_RULE,
                        domain_path="foundation.world.slot",
                        excerpt_text=f"excerpt from {self.route}",
                        score=self.score,
                        rank=1,
                        metadata={
                            "asset_id": "asset-slot",
                            "collection_id": "story-slot:archival",
                            "title": "Slot Rule",
                            "section_id": "slot-1",
                            "section_part": 0,
                            "source_ref": "memory://slot",
                            "commit_id": "commit-slot",
                        },
                    )
                ],
                trace=RetrievalTrace(
                    trace_id=f"trace-{self.route}",
                    query_id=query.query_id,
                    route=self.route,
                    candidate_count=1,
                    returned_count=1,
                    timings={f"{self.route}_ms": 1.0},
                ),
            )

    class StubFusionStrategy:
        def __init__(self) -> None:
            self.received_query: RetrievalQuery | None = None
            self.received_results: list[RetrievalSearchResult] = []

        def fuse(
            self,
            *,
            query: RetrievalQuery,
            retrieved_results: list[RetrievalSearchResult],
        ) -> RetrievalSearchResult:
            self.received_query = query
            self.received_results = list(retrieved_results)
            return retrieved_results[0].model_copy(
                update={
                    "query": query.text_query or "",
                    "trace": RetrievalTrace(
                        trace_id="trace-fused",
                        query_id=query.query_id,
                        route="fusion.stub",
                        candidate_count=2,
                        returned_count=1,
                        timings={"fusion_stub_ms": 1.0},
                    ),
                    "warnings": ["fused"],
                }
            )

    class StubReranker:
        def __init__(self) -> None:
            self.seen_query: RetrievalQuery | None = None
            self.seen_result: RetrievalSearchResult | None = None

        async def rerank(
            self,
            *,
            query: RetrievalQuery,
            result: RetrievalSearchResult,
        ) -> RetrievalSearchResult:
            self.seen_query = query
            self.seen_result = result
            return result.model_copy(
                update={"warnings": [*result.warnings, "reranked"]}
            )

    class StubChunkResultBuilder:
        def __init__(self) -> None:
            self.seen_query: RetrievalQuery | None = None
            self.seen_result: RetrievalSearchResult | None = None

        def build(
            self,
            *,
            query: RetrievalQuery,
            result: RetrievalSearchResult,
        ) -> RetrievalSearchResult:
            self.seen_query = query
            self.seen_result = result
            return result.model_copy(
                update={
                    "query": query.text_query or "",
                    "warnings": [*result.warnings, "built"],
                }
            )

    preprocessor = StubPreprocessor()
    retriever_a = StubRetriever(route="keyword.stub", score=0.8)
    retriever_b = StubRetriever(route="semantic.stub", score=0.7)
    fusion_strategy = StubFusionStrategy()
    reranker = StubReranker()
    chunk_builder = StubChunkResultBuilder()

    service = RetrievalService(
        retrieval_session,
        query_preprocessor=preprocessor,
        retrievers=[retriever_a, retriever_b],
        fusion_strategy=fusion_strategy,
        reranker=reranker,
        chunk_result_builder=chunk_builder,
    )
    query = RetrievalQuery(
        query_id="rq-slot",
        query_kind="archival",
        story_id="story-slot",
        domains=[Domain.WORLD_RULE],
        text_query=" raw slot query ",
        filters={},
        top_k=2,
    )

    result = await service.search_chunks(query)

    assert preprocessor.seen_queries[0].text_query == " raw slot query "
    assert retriever_a.seen_queries[0].text_query == "normalized slot query"
    assert retriever_b.seen_queries[0].text_query == "normalized slot query"
    assert fusion_strategy.received_query is not None
    assert fusion_strategy.received_query.text_query == "normalized slot query"
    assert len(fusion_strategy.received_results) == 2
    assert reranker.seen_query is not None
    assert reranker.seen_query.text_query == "normalized slot query"
    assert chunk_builder.seen_query is not None
    assert chunk_builder.seen_query.text_query == "normalized slot query"
    assert result.query == "normalized slot query"
    assert result.warnings == ["fused", "reranked", "built"]


@pytest.mark.asyncio
async def test_retrieval_service_expands_internal_candidate_pool_before_rerank(
    retrieval_session,
):
    class StubRetriever:
        def __init__(self) -> None:
            self.seen_top_k: int | None = None

        async def search(self, query: RetrievalQuery) -> RetrievalSearchResult:
            self.seen_top_k = query.top_k
            hits = [
                RetrievalHit(
                    hit_id=f"candidate-{index}",
                    query_id=query.query_id,
                    layer="archival",
                    domain=Domain.WORLD_RULE,
                    domain_path=f"foundation.world.{index}",
                    excerpt_text=f"Candidate {index}",
                    score=1.0 / index,
                    rank=index,
                    metadata={
                        "asset_id": f"asset-{index}",
                        "title": f"Candidate {index}",
                        "section_id": f"section-{index}",
                        "section_part": 0,
                    },
                )
                for index in range(1, query.top_k + 1)
            ]
            return RetrievalSearchResult(
                query=query.text_query or "",
                hits=hits,
                trace=RetrievalTrace(
                    trace_id="trace-candidate-pool",
                    query_id=query.query_id,
                    route="retrieval.stub.candidates",
                    result_kind="chunk",
                    retriever_routes=["retrieval.stub.candidates"],
                    pipeline_stages=["retrieve"],
                    candidate_count=len(hits),
                    returned_count=len(hits),
                ),
            )

    class PassthroughReranker:
        def __init__(self) -> None:
            self.seen_count: int | None = None

        async def rerank(
            self,
            *,
            query: RetrievalQuery,
            result: RetrievalSearchResult,
        ) -> RetrievalSearchResult:
            self.seen_count = len(result.hits)
            return result

    retriever = StubRetriever()
    reranker = PassthroughReranker()
    result = await RetrievalService(
        retrieval_session,
        retrievers=[retriever],
        reranker=reranker,
    ).search_chunks(
        RetrievalQuery(
            query_id="rq-candidate-pool",
            query_kind="archival",
            story_id="story-candidate-pool",
            text_query="candidate pool",
            filters={},
            top_k=3,
            rerank=True,
        )
    )

    assert retriever.seen_top_k == 20
    assert reranker.seen_count == 20
    assert len(result.hits) == 3
    assert result.trace is not None
    assert result.trace.details["candidate_pool"] == {
        "requested_top_k": 3,
        "retrieval_top_k": 20,
        "expanded": True,
    }


@pytest.mark.asyncio
async def test_retrieval_service_honors_explicit_candidate_pool_limit(
    retrieval_session,
):
    class StubRetriever:
        def __init__(self) -> None:
            self.seen_top_k: int | None = None

        async def search(self, query: RetrievalQuery) -> RetrievalSearchResult:
            self.seen_top_k = query.top_k
            return RetrievalSearchResult(
                query=query.text_query or "",
                hits=[],
                trace=RetrievalTrace(
                    trace_id="trace-candidate-pool-explicit",
                    query_id=query.query_id,
                    route="retrieval.stub.empty",
                    result_kind="chunk",
                    retriever_routes=["retrieval.stub.empty"],
                    pipeline_stages=["retrieve"],
                ),
            )

    retriever = StubRetriever()
    result = await RetrievalService(
        retrieval_session,
        retrievers=[retriever],
    ).search_chunks(
        RetrievalQuery(
            query_id="rq-candidate-pool-explicit",
            query_kind="archival",
            story_id="story-candidate-pool-explicit",
            text_query="candidate pool",
            filters={"search_policy": {"hybrid": {"candidate_top_k": 12}}},
            top_k=3,
        )
    )

    assert retriever.seen_top_k == 12
    assert result.trace is not None
    assert result.trace.details["candidate_pool"]["retrieval_top_k"] == 12


@pytest.mark.asyncio
async def test_retrieval_service_weights_keyword_route_for_structured_queries(
    retrieval_session,
):
    class StubRetriever:
        def __init__(self, *, route: str, hit_id: str) -> None:
            self.route = route
            self.hit_id = hit_id

        async def search(self, query: RetrievalQuery) -> RetrievalSearchResult:
            return RetrievalSearchResult(
                query=query.text_query or "",
                hits=[
                    RetrievalHit(
                        hit_id=self.hit_id,
                        query_id=query.query_id,
                        layer="archival",
                        domain=Domain.WORLD_RULE,
                        domain_path=f"foundation.world.{self.hit_id}",
                        excerpt_text=self.hit_id,
                        score=0.5,
                        rank=1,
                        metadata={
                            "asset_id": f"asset-{self.hit_id}",
                            "title": self.hit_id,
                            "section_id": self.hit_id,
                            "section_part": 0,
                        },
                    )
                ],
                trace=RetrievalTrace(
                    trace_id=f"trace-{self.hit_id}",
                    query_id=query.query_id,
                    route=self.route,
                    result_kind="chunk",
                    retriever_routes=[self.route],
                    pipeline_stages=["retrieve"],
                    candidate_count=1,
                    returned_count=1,
                ),
            )

    service = RetrievalService(
        retrieval_session,
        retrievers=[
            StubRetriever(route="retrieval.semantic.python", hit_id="dense-hit"),
            StubRetriever(route="retrieval.keyword.bm25", hit_id="keyword-hit"),
        ],
    )

    result = await service.search_chunks(
        RetrievalQuery(
            query_id="rq-structured-fusion",
            query_kind="archival",
            story_id="story-structured-fusion",
            text_query="林鸢和夜紫林的关系",
            filters={},
            top_k=2,
        )
    )

    assert result.hits[0].hit_id == "keyword-hit"
    assert result.trace is not None
    assert (
        result.trace.details["rrf"]["route_weights"]["retrieval.keyword.bm25"]
        == 2.0
    )
    assert (
        result.trace.details["rrf"]["route_weights"]["retrieval.semantic.python"]
        == 1.0
    )


@pytest.mark.asyncio
async def test_retrieval_service_consumes_runtime_style_route_weights(
    retrieval_session,
):
    class StubRetriever:
        def __init__(self, *, route: str, hit_id: str) -> None:
            self.route = route
            self.hit_id = hit_id

        async def search(self, query: RetrievalQuery) -> RetrievalSearchResult:
            return RetrievalSearchResult(
                query=query.text_query or "",
                hits=[
                    RetrievalHit(
                        hit_id=self.hit_id,
                        query_id=query.query_id,
                        layer="archival",
                        domain=Domain.WORLD_RULE,
                        domain_path=f"foundation.world.{self.hit_id}",
                        excerpt_text=self.hit_id,
                        score=0.5,
                        rank=1,
                        metadata={
                            "asset_id": f"asset-{self.hit_id}",
                            "section_id": self.hit_id,
                            "section_part": 0,
                        },
                    )
                ],
                trace=RetrievalTrace(
                    trace_id=f"trace-{self.hit_id}",
                    query_id=query.query_id,
                    route=self.route,
                    result_kind="chunk",
                    retriever_routes=[self.route],
                    pipeline_stages=["retrieve"],
                    candidate_count=1,
                    returned_count=1,
                ),
            )

    service = RetrievalService(
        retrieval_session,
        retrievers=[
            StubRetriever(route="retrieval.keyword.bm25", hit_id="keyword-hit"),
            StubRetriever(route="retrieval.semantic.python", hit_id="dense-hit"),
        ],
    )

    result = await service.search_chunks(
        RetrievalQuery(
            query_id="rq-runtime-style-fusion",
            query_kind="archival",
            story_id="story-runtime-style-fusion",
            text_query="broad context",
            filters={
                "search_policy": {
                    "hybrid": {
                        "route_weights": {
                            "retrieval.keyword": 1.0,
                            "retrieval.semantic": 1.5,
                        }
                    }
                }
            },
            top_k=2,
        )
    )

    assert result.hits[0].hit_id == "dense-hit"
    assert "_rrf_weight" not in result.hits[0].model_dump(mode="json")
    assert result.trace is not None
    assert result.trace.details["rrf"]["route_weights"] == {
        "retrieval.keyword.bm25": 1.0,
        "retrieval.semantic.python": 1.5,
    }


@pytest.mark.asyncio
async def test_retrieval_service_emits_langfuse_observation(retrieval_session):
    class StubRetriever:
        async def search(self, query: RetrievalQuery) -> RetrievalSearchResult:
            return RetrievalSearchResult(
                query=query.text_query or "",
                hits=[
                    RetrievalHit(
                        hit_id="chunk-observation",
                        query_id=query.query_id,
                        layer="archival",
                        domain=Domain.WORLD_RULE,
                        domain_path="foundation.world.obs",
                        excerpt_text="Observation friendly excerpt",
                        score=0.77,
                        rank=1,
                        metadata={
                            "asset_id": "asset-observation",
                            "title": "Observation Rule",
                            "section_id": "obs-1",
                            "section_part": 0,
                            "page_label": "IV",
                            "contextual_text_version": "v2",
                        },
                    )
                ],
                trace=RetrievalTrace(
                    trace_id="trace-observation",
                    query_id=query.query_id,
                    route="retrieval.keyword.stub",
                    result_kind="chunk",
                    retriever_routes=["retrieval.keyword.stub"],
                    pipeline_stages=["retrieve", "chunk_result_builder"],
                    candidate_count=1,
                    returned_count=1,
                    timings={"keyword_ms": 1.0},
                ),
            )

    class StubReranker:
        async def rerank(
            self, *, query: RetrievalQuery, result: RetrievalSearchResult
        ) -> RetrievalSearchResult:
            return result

    fake_langfuse = _FakeLangfuseService()
    service = RetrievalService(
        retrieval_session,
        retrievers=[StubRetriever()],
        reranker=StubReranker(),
        langfuse_service=fake_langfuse,
    )
    query = RetrievalQuery(
        query_id="rq-observation",
        query_kind="archival",
        story_id="story-observation",
        domains=[Domain.WORLD_RULE],
        text_query="observation query",
        filters={},
        top_k=1,
    )

    result = await service.search_chunks(query)

    assert result.hits[0].hit_id == "chunk-observation"
    observation_names = [
        item["name"]
        for item in fake_langfuse.events
        if item["kind"] == "observation_enter"
    ]
    assert "rp.retrieval.search_chunks" in observation_names
    updates = [
        item["payload"]["output"]
        for item in fake_langfuse.events
        if item["kind"] == "observation_update"
        and item["name"] == "rp.retrieval.search_chunks"
    ]
    assert updates
    observability = updates[-1]["observability"]
    assert updates[-1]["status"] == "ok"
    assert updates[-1]["search_kind"] == "chunks"
    assert observability["route"].startswith("retrieval.keyword.stub")
    assert observability["returned_count"] == 1
    assert observability["top_hits"][0]["asset_id"] == "asset-observation"
    assert observability["top_hits"][0]["block_view"]["source"] == "retrieval_store"
    assert observability["top_hits"][0]["block_view"]["block_id"].startswith(
        "retrieval.archival.rq-observation."
    )
    assert observability["top_hits"][0]["block_view"]["block_id"].endswith(
        ".chunk-observation"
    )


@pytest.mark.asyncio
async def test_simple_metadata_reranker_can_promote_metadata_aligned_hit(
    retrieval_session,
):
    class StubRetriever:
        async def search(self, query: RetrievalQuery) -> RetrievalSearchResult:
            return RetrievalSearchResult(
                query=query.text_query or "",
                hits=[
                    RetrievalHit(
                        hit_id="chunk-a",
                        query_id=query.query_id,
                        layer="archival",
                        domain=Domain.WORLD_RULE,
                        domain_path="foundation.world.misc",
                        excerpt_text="A generic rule.",
                        score=0.61,
                        rank=1,
                        metadata={
                            "asset_id": "asset-a",
                            "title": "Generic Rule",
                            "domain_path": "foundation.world.misc",
                            "section_id": "a",
                            "section_part": 0,
                        },
                    ),
                    RetrievalHit(
                        hit_id="chunk-b",
                        query_id=query.query_id,
                        layer="archival",
                        domain=Domain.WORLD_RULE,
                        domain_path="foundation.world.moon_gate",
                        excerpt_text="Moon gate rituals must be witnessed at dawn.",
                        score=0.58,
                        rank=2,
                        metadata={
                            "asset_id": "asset-b",
                            "title": "Moon Gate Ritual",
                            "domain_path": "foundation.world.moon_gate",
                            "tags": ["moon", "ritual"],
                            "section_id": "b",
                            "section_part": 0,
                        },
                    ),
                ],
                trace=RetrievalTrace(
                    trace_id="trace-keyword",
                    query_id=query.query_id,
                    route="keyword.stub",
                    result_kind="chunk",
                    retriever_routes=["keyword.stub"],
                    pipeline_stages=["retrieve"],
                    candidate_count=2,
                    returned_count=2,
                    timings={"keyword_ms": 1.0},
                ),
            )

    service = RetrievalService(retrieval_session, retrievers=[StubRetriever()])
    result = await service.search_chunks(
        RetrievalQuery(
            query_id="rq-rerank",
            query_kind="archival",
            story_id="story-rerank",
            domains=[Domain.WORLD_RULE],
            text_query="moon gate ritual",
            filters={},
            top_k=2,
            rerank=True,
        )
    )

    assert result.hits[0].hit_id == "chunk-b"
    assert result.hits[0].score > result.hits[1].score
    assert result.trace is not None
    assert result.trace.reranker_name == "simple_metadata"
    assert "rerank" in result.trace.pipeline_stages


@pytest.mark.asyncio
async def test_simple_metadata_reranker_records_narrative_scoring_trace():
    result = RetrievalSearchResult(
        query="scene continuity",
        hits=[
            RetrievalHit(
                hit_id="chunk-generic",
                query_id="rq-narrative-rerank",
                layer="recall",
                domain=Domain.CHAPTER,
                domain_path="recall.chapter.1.generic",
                excerpt_text="Generic continuity note.",
                score=0.7,
                rank=1,
                metadata={
                    "asset_id": "asset-generic",
                    "canon_status": "superseded",
                    "chapter_index": 8,
                },
            ),
            RetrievalHit(
                hit_id="chunk-scene",
                query_id="rq-narrative-rerank",
                layer="recall",
                domain=Domain.CHAPTER,
                domain_path="recall.chapter.1.scene",
                excerpt_text="Current scene continuity note.",
                score=0.62,
                rank=2,
                metadata={
                    "asset_id": "asset-scene",
                    "scene_ref": "chapter:1:scene:1",
                    "character_refs": ["hero"],
                    "canon_status": "canonical",
                    "chapter_index": 2,
                },
            ),
        ],
        trace=RetrievalTrace(
            trace_id="trace-narrative-rerank",
            query_id="rq-narrative-rerank",
            route="retrieval.hybrid.stub",
            result_kind="chunk",
            pipeline_stages=["retrieve", "fusion"],
            candidate_count=2,
            returned_count=2,
        ),
    )

    reranked = await SimpleMetadataReranker().rerank(
        query=RetrievalQuery(
            query_id="rq-narrative-rerank",
            query_kind="recall",
            story_id="story-narrative-rerank",
            domains=[Domain.CHAPTER],
            text_query="scene continuity",
            filters={
                "scene_refs": ["chapter:1:scene:1"],
                "character_refs": ["hero"],
                "search_policy": {
                    "profile": "longform",
                    "rerank": "on",
                    "context": {"current_chapter_index": 2},
                },
            },
            top_k=2,
            rerank=True,
        ),
        result=result,
    )

    assert reranked.hits[0].hit_id == "chunk-scene"
    assert reranked.trace is not None
    scoring = reranked.trace.details["narrative_scoring"]
    assert scoring["profile"] == "longform"
    rules = {item["hit_id"]: item for item in scoring["rules"]}
    assert rules["chunk-scene"]["boosts"]["scene_match"] == 0.2
    assert rules["chunk-scene"]["boosts"]["character_match"] == 0.12
    assert rules["chunk-scene"]["boosts"]["chapter_distance"] == 0.08
    assert rules["chunk-generic"]["penalties"]["non_canonical_status"] == -0.25


def test_rag_context_builder_records_budget_selected_and_excluded_hits():
    result = RetrievalSearchResult(
        query="budget",
        hits=[
            RetrievalHit(
                hit_id="chunk-a",
                query_id="rq-budget",
                layer="archival",
                domain=Domain.WORLD_RULE,
                domain_path="foundation.world.a",
                excerpt_text="First selected section.",
                score=0.9,
                rank=1,
                metadata={
                    "asset_id": "asset-a",
                    "source_family": "setup_source",
                    "domain": "world_rule",
                    "token_count": 3,
                },
            ),
            RetrievalHit(
                hit_id="chunk-a-dup",
                query_id="rq-budget",
                layer="archival",
                domain=Domain.WORLD_RULE,
                domain_path="foundation.world.a.dup",
                excerpt_text="Duplicate section.",
                score=0.8,
                rank=2,
                metadata={
                    "asset_id": "asset-a",
                    "source_family": "setup_source",
                    "domain": "world_rule",
                    "token_count": 2,
                },
            ),
            RetrievalHit(
                hit_id="chunk-b",
                query_id="rq-budget",
                layer="archival",
                domain=Domain.WORLD_RULE,
                domain_path="foundation.world.b",
                excerpt_text="Over budget section.",
                score=0.7,
                rank=3,
                metadata={
                    "asset_id": "asset-b",
                    "source_family": "setup_source",
                    "domain": "world_rule",
                    "token_count": 10,
                },
            ),
        ],
        trace=RetrievalTrace(
            trace_id="trace-budget",
            query_id="rq-budget",
            route="retrieval.hybrid.stub",
            result_kind="chunk",
            filters_applied={
                "filters": {
                    "search_policy": {
                        "context_budget": {
                            "max_tokens": 6,
                            "per_source_family": {"setup_source": 6},
                            "per_domain": {"world_rule": 6},
                        }
                    }
                }
            },
        ),
    )

    built = RagContextBuilder().build(result)

    assert [hit.hit_id for hit in built.hits] == ["chunk-a"]
    assert built.trace is not None
    budget = built.trace.details["context_budget"]
    assert budget["selected"][0]["hit_id"] == "chunk-a"
    excluded = {item["hit_id"]: item["reason"] for item in budget["excluded"]}
    assert excluded["chunk-a-dup"] == "duplicate_asset"
    assert excluded["chunk-b"] == "token_budget"


@pytest.mark.asyncio
async def test_search_chunks_short_circuits_empty_query_even_with_active_embeddings(
    retrieval_session,
):
    collection = RetrievalCollectionService(retrieval_session).ensure_story_collection(
        story_id="story-empty-query",
        scope="story",
        collection_kind="archival",
    )
    RetrievalDocumentService(retrieval_session).upsert_source_asset(
        SourceAsset(
            asset_id="asset-empty-query",
            story_id="story-empty-query",
            mode=StoryMode.LONGFORM,
            collection_id=collection.collection_id,
            commit_id="commit-empty-query",
            asset_kind="worldbook",
            source_ref="memory://asset-empty-query",
            title="Empty Query Asset",
            parse_status="queued",
            ingestion_status="queued",
            mapped_targets=["foundation"],
            metadata={
                "seed_sections": [
                    {
                        "section_id": "empty-1",
                        "title": "Rule",
                        "path": "foundation.world.empty_query_rule",
                        "level": 1,
                        "text": "Empty queries should not retrieve semantic neighbors.",
                        "metadata": {
                            "domain": "world_rule",
                            "domain_path": "foundation.world.empty_query_rule",
                        },
                    }
                ]
            },
            created_at=_utcnow(),
            updated_at=_utcnow(),
        )
    )
    retrieval_session.flush()
    RetrievalIngestionService(retrieval_session).ingest_asset(
        story_id="story-empty-query",
        asset_id="asset-empty-query",
        collection_id=collection.collection_id,
    )
    retrieval_session.commit()

    result = await RetrievalService(retrieval_session).search_chunks(
        RetrievalQuery(
            query_id="rq-empty-query",
            query_kind="archival",
            story_id="story-empty-query",
            domains=[Domain.WORLD_RULE],
            text_query="   ",
            filters={"knowledge_collections": [collection.collection_id]},
            top_k=3,
        )
    )

    assert result.hits == []
    assert "keyword_unavailable:empty_query" in result.warnings
    assert "dense_unavailable:empty_query" in result.warnings
    assert result.trace is not None
    assert result.trace.route == "retrieval.hybrid.empty"
