"""Sparse retrieval over persisted knowledge chunks."""

from __future__ import annotations

import math
import re
import time
import uuid
from collections import Counter
from collections.abc import Mapping
from typing import Any, cast

from sqlalchemy import text
from sqlmodel import select

from models.rp_retrieval_store import KnowledgeChunkRecord, KnowledgeCollectionRecord, SourceAssetRecord
from rp.models.memory_crud import RetrievalQuery, RetrievalSearchResult, RetrievalTrace
from .search_utils import build_chunk_hit, build_filters_applied, chunk_view_priority, row_matches_common_filters

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+", re.UNICODE)
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

_FIELD_WEIGHTS = {
    "title": 8.0,
    "asset_title": 6.0,
    "document_title": 6.0,
    "entry_title": 10.0,
    "aliases": 10.0,
    "tags": 5.0,
    "section_title": 4.0,
    "retrieval_role": 3.0,
    "semantic_path": 2.0,
    "domain_path": 2.0,
    "text": 1.0,
}


def _tokenize(text: str) -> list[str]:
    ascii_tokens = [token for token in _TOKEN_RE.findall(text.lower()) if token]
    cjk_chars = _CJK_RE.findall(text)
    cjk_bigrams = [
        "".join(cjk_chars[index : index + 2])
        for index in range(max(len(cjk_chars) - 1, 0))
    ]
    cjk_trigrams = [
        "".join(cjk_chars[index : index + 3])
        for index in range(max(len(cjk_chars) - 2, 0))
    ]
    return [*ascii_tokens, *cjk_chars, *cjk_bigrams, *cjk_trigrams]


def _weighted_tokens(text: str, *, weight: float) -> list[str]:
    tokens = _tokenize(text)
    if not tokens or weight <= 0.0:
        return []
    repeat_count = max(1, int(round(weight)))
    return tokens * repeat_count


def _flatten_metadata_value(value: object) -> list[str]:
    if isinstance(value, Mapping):
        mapping_values: list[str] = []
        for key in sorted(value, key=lambda item: str(item)):
            mapping_values.extend(_flatten_metadata_value(value[key]))
        return mapping_values
    if isinstance(value, list | tuple):
        sequence_values: list[str] = []
        for item in value:
            sequence_values.extend(_flatten_metadata_value(item))
        return sequence_values
    if isinstance(value, set):
        set_values: list[str] = []
        for item in sorted(value, key=lambda entry: str(entry)):
            set_values.extend(_flatten_metadata_value(item))
        return set_values
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def _metadata_text(metadata: Mapping[str, Any], *keys: str) -> str:
    values: list[str] = []
    for key in keys:
        values.extend(_flatten_metadata_value(metadata.get(key)))
    return " ".join(values)


def _field_weighted_tokens(field_values: dict[str, object]) -> list[str]:
    tokens: list[str] = []
    for field_name, weight in _FIELD_WEIGHTS.items():
        value = field_values.get(field_name)
        if value is None:
            continue
        text = " ".join(_flatten_metadata_value(value))
        tokens.extend(_weighted_tokens(text, weight=weight))
    return tokens


def _bm25_scores(
    *,
    query_terms: list[str],
    documents: list[tuple[str, list[str]]],
    k1: float = 1.5,
    b: float = 0.75,
) -> dict[str, float]:
    if not query_terms or not documents:
        return {}

    doc_count = len(documents)
    avg_doc_len = sum(len(tokens) for _, tokens in documents) / doc_count
    avg_doc_len = avg_doc_len or 1.0
    doc_frequencies: Counter[str] = Counter()
    term_frequencies: dict[str, Counter[str]] = {}
    doc_lengths: dict[str, int] = {}
    for chunk_id, tokens in documents:
        frequencies = Counter(tokens)
        term_frequencies[chunk_id] = frequencies
        doc_lengths[chunk_id] = len(tokens)
        for term in query_terms:
            if frequencies.get(term, 0) > 0:
                doc_frequencies[term] += 1

    scores: dict[str, float] = {}
    for chunk_id, frequencies in term_frequencies.items():
        doc_len = doc_lengths.get(chunk_id, 0) or 1
        score = 0.0
        for term in query_terms:
            tf = float(frequencies.get(term, 0))
            if tf <= 0.0:
                continue
            df = float(doc_frequencies.get(term, 0))
            idf = math.log(1.0 + (doc_count - df + 0.5) / (df + 0.5))
            denominator = tf + k1 * (1.0 - b + b * (doc_len / avg_doc_len))
            score += idf * ((tf * (k1 + 1.0)) / denominator)
        if score > 0.0:
            scores[chunk_id] = score
    return scores


def _contains_term(value: object, term: str) -> bool:
    normalized_term = str(term or "").strip().lower()
    if not normalized_term:
        return False
    if isinstance(value, list):
        return any(_contains_term(item, normalized_term) for item in value)
    if isinstance(value, tuple | set):
        return any(_contains_term(item, normalized_term) for item in value)
    if isinstance(value, Mapping):
        return any(_contains_term(item, normalized_term) for item in value.values())
    return normalized_term in str(value or "").lower()


def _has_cjk_text(text: str | None) -> bool:
    return bool(_CJK_RE.search(str(text or "")))


class KeywordRetriever:
    """Prefer PostgreSQL FTS and fall back to BM25 sparse scoring elsewhere."""

    def __init__(self, session) -> None:
        self._session = session

    async def search(self, query: RetrievalQuery) -> RetrievalSearchResult:
        started = time.perf_counter()
        if not (query.text_query or "").strip():
            warning = "keyword_unavailable:empty_query"
            return RetrievalSearchResult(
                query="",
                hits=[],
                trace=RetrievalTrace(
                    trace_id=f"trace_{uuid.uuid4().hex[:10]}",
                    query_id=query.query_id,
                    route="retrieval.keyword.empty",
                    result_kind="chunk",
                    filters_applied=build_filters_applied(query),
                    retriever_routes=["retrieval.keyword.empty"],
                    pipeline_stages=["retrieve"],
                    candidate_count=0,
                    returned_count=0,
                    timings={"keyword_ms": round((time.perf_counter() - started) * 1000, 3)},
                    warnings=[warning],
                ),
                warnings=[warning],
            )
        rows, warnings, route = self._load_ranked_rows(query)
        hits = [
            build_chunk_hit(
                query=query,
                chunk=chunk,
                asset=asset,
                collection=collection,
                score=score,
                rank=index,
            )
            for index, (chunk, asset, collection, score) in enumerate(rows[: query.top_k], start=1)
        ]
        return RetrievalSearchResult(
            query=query.text_query or "",
            hits=hits,
            trace=RetrievalTrace(
                trace_id=f"trace_{uuid.uuid4().hex[:10]}",
                query_id=query.query_id,
                route=route,
                result_kind="chunk",
                filters_applied=build_filters_applied(query),
                retriever_routes=[route],
                pipeline_stages=["retrieve"],
                candidate_count=len(rows),
                returned_count=len(hits),
                timings={"keyword_ms": round((time.perf_counter() - started) * 1000, 3)},
                warnings=warnings,
            ),
            warnings=warnings,
        )

    def _load_ranked_rows(
        self,
        query: RetrievalQuery,
    ) -> tuple[list[tuple[KnowledgeChunkRecord, SourceAssetRecord, KnowledgeCollectionRecord | None, float]], list[str], str]:
        if self._session.get_bind().dialect.name == "postgresql" and (query.text_query or "").strip():
            if self._should_use_python_sparse_path(query):
                return (
                    self._load_with_python_scoring(query),
                    ["fts_bypassed:structured_sparse_parity"],
                    "retrieval.keyword.bm25",
                )
            try:
                postgres_rows = self._load_with_postgres_fts(query)
                if postgres_rows:
                    return postgres_rows, [], "retrieval.keyword.fts"
            except Exception as exc:  # pragma: no cover - postgres only
                return self._load_with_python_scoring(query), [f"fts_unavailable:{type(exc).__name__}"], "retrieval.keyword.bm25"
        return self._load_with_python_scoring(query), [], "retrieval.keyword.bm25"

    @staticmethod
    def _should_use_python_sparse_path(query: RetrievalQuery) -> bool:
        if _has_cjk_text(query.text_query):
            return True
        analysis = query.filters.get("query_analysis")
        if not isinstance(analysis, dict):
            return False
        return bool(analysis.get("entity_terms") or analysis.get("intent_terms"))

    def _load_with_postgres_fts(
        self,
        query: RetrievalQuery,
    ) -> list[tuple[KnowledgeChunkRecord, SourceAssetRecord, KnowledgeCollectionRecord | None, float]]:
        conditions = ["c.is_active = true"]
        params: dict[str, Any] = {
            "query": query.text_query,
            "limit": max(query.top_k * 5, query.top_k),
        }
        if query.story_id not in {"", "*"}:
            conditions.append("c.story_id = :story_id")
            params["story_id"] = query.story_id
        if query.domains:
            conditions.append("c.domain = ANY(:domains)")
            params["domains"] = [domain.value for domain in query.domains]
        collection_ids = list(query.filters.get("knowledge_collections") or [])
        if collection_ids:
            conditions.append("c.collection_id = ANY(:collection_ids)")
            params["collection_ids"] = collection_ids
        elif query.query_kind == "archival":
            conditions.append("k.collection_kind = 'archival'")
        elif query.query_kind == "recall":
            conditions.append("k.collection_kind = 'recall'")

        sql = text(
            "SELECT c.chunk_id, "
            "ts_rank_cd(to_tsvector('simple', coalesce(c.title, '') || ' ' || coalesce(c.\"text\", '')), "
            "plainto_tsquery('simple', :query)) AS score "
            "FROM rp_knowledge_chunks c "
            "JOIN rp_source_assets a ON a.asset_id = c.asset_id "
            "LEFT JOIN rp_knowledge_collections k ON k.collection_id = c.collection_id "
            f"WHERE {' AND '.join(conditions)} "
            "AND to_tsvector('simple', coalesce(c.title, '') || ' ' || coalesce(c.\"text\", '')) @@ plainto_tsquery('simple', :query) "
            "ORDER BY score DESC "
            "LIMIT :limit"
        )
        ranked_ids = list(self._session.exec(sql, params).all())
        score_map = {row[0]: float(row[1]) for row in ranked_ids}
        if not score_map:
            return []
        return self._load_joined_rows(query, score_map)

    def _load_with_python_scoring(
        self,
        query: RetrievalQuery,
    ) -> list[tuple[KnowledgeChunkRecord, SourceAssetRecord, KnowledgeCollectionRecord | None, float]]:
        query_terms = list(dict.fromkeys(_tokenize(query.text_query or "")))
        if not query_terms:
            return []

        rows = self._iter_joined_rows(query)
        documents = [
            (chunk.chunk_id, self._tokens_for_chunk(chunk=chunk, asset=asset))
            for chunk, asset, _ in rows
        ]
        score_map = _bm25_scores(query_terms=query_terms, documents=documents)
        score_map = {
            chunk.chunk_id: score * self._structured_query_multiplier(
                query=query,
                chunk=chunk,
                asset=asset,
            )
            for chunk, asset, _ in rows
            if (score := score_map.get(chunk.chunk_id, 0.0)) > 0.0
        }
        return self._load_joined_rows(query, score_map)

    def _tokens_for_chunk(
        self,
        *,
        chunk: KnowledgeChunkRecord,
        asset: SourceAssetRecord,
    ) -> list[str]:
        metadata = chunk.metadata_json or {}
        asset_metadata = asset.metadata_json or {}
        seed_tags = _metadata_text(metadata, "tags")
        if not seed_tags:
            seed_sections = asset_metadata.get("seed_sections")
            if isinstance(seed_sections, list):
                seed_tags = " ".join(
                    " ".join(str(tag) for tag in item.get("tags") or [])
                    for item in seed_sections
                    if isinstance(item, dict)
                    and item.get("section_id") == metadata.get("section_id")
                )
        return _field_weighted_tokens(
            {
                "title": chunk.title or metadata.get("title") or "",
                "asset_title": _metadata_text(
                    {
                        "asset_title": asset.title,
                        "metadata_asset_title": metadata.get("asset_title"),
                        "asset_metadata_asset_title": asset_metadata.get("asset_title"),
                    },
                    "asset_title",
                    "metadata_asset_title",
                    "asset_metadata_asset_title",
                ),
                "document_title": _metadata_text(
                    {
                        "document_title": metadata.get("document_title"),
                        "asset_document_title": asset_metadata.get("document_title"),
                    },
                    "document_title",
                    "asset_document_title",
                ),
                "entry_title": _metadata_text(metadata, "entry_title"),
                "aliases": _metadata_text(
                    {
                        "metadata_aliases": metadata.get("aliases"),
                        "asset_metadata_aliases": asset_metadata.get("aliases"),
                    },
                    "metadata_aliases",
                    "asset_metadata_aliases",
                ),
                "tags": _metadata_text(
                    {
                        "metadata_or_seed_tags": seed_tags,
                        "asset_metadata_tags": asset_metadata.get("tags"),
                    },
                    "metadata_or_seed_tags",
                    "asset_metadata_tags",
                ),
                "section_title": _metadata_text(metadata, "section_title"),
                "retrieval_role": _metadata_text(metadata, "retrieval_role"),
                "semantic_path": _metadata_text(
                    metadata,
                    "semantic_path",
                    "entry_semantic_path",
                    "section_semantic_path",
                ),
                "domain_path": _metadata_text(
                    {
                        "chunk_domain_path": chunk.domain_path,
                        "metadata_domain_path": metadata.get("domain_path"),
                        "asset_metadata_domain_path": asset_metadata.get("domain_path"),
                    },
                    "chunk_domain_path",
                    "metadata_domain_path",
                    "asset_metadata_domain_path",
                ),
                "text": chunk.text,
            }
        )

    def _structured_query_multiplier(
        self,
        *,
        query: RetrievalQuery,
        chunk: KnowledgeChunkRecord,
        asset: SourceAssetRecord,
    ) -> float:
        analysis = query.filters.get("query_analysis")
        if not isinstance(analysis, dict):
            return 1.0

        metadata = chunk.metadata_json or {}
        asset_metadata = asset.metadata_json or {}
        entity_fields = (
            chunk.title,
            asset.title,
            metadata.get("title"),
            metadata.get("document_title"),
            metadata.get("asset_title"),
            asset_metadata.get("document_title"),
            asset_metadata.get("asset_title"),
            metadata.get("entry_title"),
            metadata.get("aliases"),
            asset_metadata.get("aliases"),
            metadata.get("tags"),
            asset_metadata.get("tags"),
            metadata.get("semantic_path"),
            metadata.get("entry_semantic_path"),
            metadata.get("section_semantic_path"),
            chunk.domain_path,
            metadata.get("domain_path"),
            asset_metadata.get("domain_path"),
        )
        intent_fields = (
            chunk.title,
            asset.title,
            metadata.get("document_title"),
            metadata.get("asset_title"),
            asset_metadata.get("document_title"),
            asset_metadata.get("asset_title"),
            metadata.get("section_title"),
            metadata.get("retrieval_role"),
            metadata.get("tags"),
            asset_metadata.get("tags"),
            metadata.get("semantic_path"),
            metadata.get("entry_semantic_path"),
            metadata.get("section_semantic_path"),
            chunk.domain_path,
            metadata.get("domain_path"),
            asset_metadata.get("domain_path"),
        )

        multiplier = 1.0
        entity_terms = [
            str(term).strip()
            for term in analysis.get("entity_terms") or []
            if str(term).strip()
        ]
        if entity_terms:
            entity_matches = sum(
                1
                for term in entity_terms
                if any(_contains_term(value, term) for value in entity_fields)
            )
            if entity_matches:
                multiplier += min(0.6, 0.3 * entity_matches)

        intent_terms = [
            str(term).strip()
            for term in analysis.get("intent_terms") or []
            if str(term).strip()
        ]
        if intent_terms:
            intent_matches = sum(
                1
                for term in intent_terms
                if any(_contains_term(value, term) for value in intent_fields)
            )
            if intent_matches:
                multiplier += min(0.2, 0.1 * intent_matches)

        if entity_terms and not any(
            any(_contains_term(value, term) for value in entity_fields)
            for term in entity_terms
        ):
            multiplier *= 0.9
        return multiplier

    def _load_joined_rows(
        self,
        query: RetrievalQuery,
        score_map: dict[str, float],
    ) -> list[tuple[KnowledgeChunkRecord, SourceAssetRecord, KnowledgeCollectionRecord | None, float]]:
        if not score_map:
            return []

        chunk_asset_id = cast(Any, KnowledgeChunkRecord.asset_id)
        chunk_collection_id = cast(Any, KnowledgeChunkRecord.collection_id)
        source_asset_id = cast(Any, SourceAssetRecord.asset_id)
        collection_id = cast(Any, KnowledgeCollectionRecord.collection_id)
        chunk_id = cast(Any, KnowledgeChunkRecord.chunk_id)
        stmt = (
            select(KnowledgeChunkRecord, SourceAssetRecord, KnowledgeCollectionRecord)
            .join(SourceAssetRecord, source_asset_id == chunk_asset_id)
            .join(
                KnowledgeCollectionRecord,
                collection_id == chunk_collection_id,
                isouter=True,
            )
            .where(chunk_id.in_(list(score_map.keys())))
        )
        rows = []
        for chunk, asset, collection in self._session.exec(stmt).all():
            if not row_matches_common_filters(chunk=chunk, asset=asset, collection=collection, query=query):
                continue
            score = float(score_map.get(chunk.chunk_id) or 0.0)
            if score <= 0.0:
                continue
            rows.append((chunk, asset, collection, score))
        rows.sort(
            key=lambda item: (
                -item[3],
                chunk_view_priority(item[0].metadata_json or {}),
                item[0].chunk_index,
            )
        )
        return rows

    def _iter_joined_rows(
        self,
        query: RetrievalQuery,
    ) -> list[tuple[KnowledgeChunkRecord, SourceAssetRecord, KnowledgeCollectionRecord | None]]:
        chunk_asset_id = cast(Any, KnowledgeChunkRecord.asset_id)
        chunk_collection_id = cast(Any, KnowledgeChunkRecord.collection_id)
        source_asset_id = cast(Any, SourceAssetRecord.asset_id)
        collection_id = cast(Any, KnowledgeCollectionRecord.collection_id)
        chunk_is_active = cast(Any, KnowledgeChunkRecord.is_active)
        chunk_story_id = cast(Any, KnowledgeChunkRecord.story_id)
        chunk_domain = cast(Any, KnowledgeChunkRecord.domain)
        stmt = (
            select(KnowledgeChunkRecord, SourceAssetRecord, KnowledgeCollectionRecord)
            .join(SourceAssetRecord, source_asset_id == chunk_asset_id)
            .join(
                KnowledgeCollectionRecord,
                collection_id == chunk_collection_id,
                isouter=True,
            )
            .where(chunk_is_active == True)  # noqa: E712
        )
        if query.story_id not in {"", "*"}:
            stmt = stmt.where(chunk_story_id == query.story_id)
        if query.domains:
            stmt = stmt.where(chunk_domain.in_([domain.value for domain in query.domains]))
        collection_ids = list(query.filters.get("knowledge_collections") or [])
        if collection_ids:
            stmt = stmt.where(chunk_collection_id.in_(collection_ids))
        rows = []
        for chunk, asset, collection in self._session.exec(stmt).all():
            if row_matches_common_filters(chunk=chunk, asset=asset, collection=collection, query=query):
                rows.append((chunk, asset, collection))
        return rows
