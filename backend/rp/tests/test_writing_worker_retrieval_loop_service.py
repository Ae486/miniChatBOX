"""Focused tests for writer-side bounded retrieval loop E2."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from rp.models.dsl import Domain
from rp.models.memory_crud import MemorySearchRecallInput
from rp.models.runtime_workspace_material import RuntimeWorkspaceMaterialKind
from rp.models.runtime_workspace_material import RuntimeWorkspaceMaterialVisibility
from rp.models.setup_workspace import StoryMode
from rp.models.story_runtime import LongformChapterPhase
from rp.models.writing_worker_contracts import WritingWorkerExecutionRequest
from rp.models.writing_runtime import WritingPacket
from rp.models.retrieval_records import SourceAsset
from rp.services.retrieval_collection_service import RetrievalCollectionService
from rp.services.retrieval_document_service import RetrievalDocumentService
from rp.services.retrieval_ingestion_service import RetrievalIngestionService
from rp.services.runtime_profile_snapshot_service import RuntimeProfileSnapshotService
from rp.services.runtime_retrieval_card_service import RuntimeRetrievalCardService
from rp.services.story_runtime_identity_service import StoryRuntimeIdentityService
from rp.services.story_session_service import StorySessionService
from rp.services.writing_worker_execution_service import WritingWorkerExecutionService
from rp.services.writing_worker_retrieval_loop_service import (
    WritingWorkerRetrievalLoopService,
    WritingWorkerRetrievalLoopServiceError,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def test_writer_retrieval_search_tool_schema_exposes_only_query_hints():
    tools = WritingWorkerRetrievalLoopService._tool_definitions()
    search_tool = next(
        tool for tool in tools if tool["function"]["name"] == "retrieval.search"
    )
    parameters = search_tool["function"]["parameters"]
    properties = parameters["properties"]

    assert parameters["additionalProperties"] is False
    assert set(properties) == {
        "query",
        "mode",
        "lexical_anchors",
        "semantic_predicates",
    }
    assert "search_kind" not in properties
    assert "top_k" not in properties
    assert "filters" not in properties
    assert "rerank" not in properties
    assert "route_weights" not in properties
    assert properties["mode"]["enum"] == ["entity", "relation", "semantic"]
    assert "separate entity searches" in search_tool["function"]["description"]
    assert "Use relation" in search_tool["function"]["description"]
    assert "Use semantic" in search_tool["function"]["description"]
    assert parameters["required"] == ["query"]


def _seed_story_runtime(retrieval_session):
    service = StorySessionService(retrieval_session)
    session = service.create_session(
        story_id="story-writer-loop",
        source_workspace_id="workspace-writer-loop",
        mode="longform",
        runtime_story_config={},
        writer_contract={},
        current_state_json={
            "chapter_digest": {"current_chapter": 1, "title": "Chapter One"},
            "narrative_progress": {
                "current_phase": "segment_drafting",
                "accepted_segments": 0,
            },
            "timeline_spine": [],
            "active_threads": [],
            "foreshadow_registry": [],
            "character_state_digest": {},
        },
        initial_phase=LongformChapterPhase.SEGMENT_DRAFTING,
    )
    service.create_chapter_workspace(
        session_id=session.session_id,
        chapter_index=1,
        phase=LongformChapterPhase.SEGMENT_DRAFTING,
        builder_snapshot_json={
            "foundation_digest": ["Found A"],
            "blueprint_digest": ["Blueprint A"],
            "current_outline_digest": ["Outline A"],
            "recent_segment_digest": ["Segment A"],
            "current_state_digest": ["State A"],
        },
    )
    service.commit()
    return service.get_session(session.session_id)


def _seed_recall_asset(retrieval_session, *, story_id: str):
    collection = RetrievalCollectionService(retrieval_session).ensure_story_collection(
        story_id=story_id,
        scope="story",
        collection_kind="recall",
    )
    RetrievalDocumentService(retrieval_session).upsert_source_asset(
        SourceAsset(
            asset_id="asset-writer-loop-recall",
            story_id=story_id,
            mode=StoryMode.LONGFORM,
            collection_id=collection.collection_id,
            asset_kind="accepted_story_segment",
            source_ref="memory://writer-loop-recall",
            title="Writer Retrieval Loop Recall",
            parse_status="queued",
            ingestion_status="queued",
            mapped_targets=["recall"],
            metadata={
                "seed_sections": [
                    {
                        "section_id": "seed:writer-loop-recall",
                        "title": "Writer Retrieval Loop Recall",
                        "path": "chapter.recall.writer-loop",
                        "level": 1,
                        "text": "The silver seal broke during the first storm at dusk.",
                        "metadata": {
                            "domain": Domain.CHAPTER.value,
                            "domain_path": "chapter.recall.writer-loop",
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
        story_id=story_id,
        asset_id="asset-writer-loop-recall",
        collection_id=collection.collection_id,
    )


def _build_identity(retrieval_session):
    session = _seed_story_runtime(retrieval_session)
    _seed_recall_asset(retrieval_session, story_id=session.story_id)
    snapshot_service = RuntimeProfileSnapshotService(retrieval_session)
    snapshot = snapshot_service.ensure_active_snapshot(
        session_id=session.session_id,
        created_from="test.writer_retrieval_loop",
    )
    identity = StoryRuntimeIdentityService(
        retrieval_session,
        runtime_profile_snapshot_service=snapshot_service,
    ).resolve_runtime_entry_identity(
        session_id=session.session_id,
        command_kind="write_next_segment",
        actor="story_runtime",
        requested_runtime_profile_snapshot_id=snapshot.runtime_profile_snapshot_id,
    )
    return identity


def _build_packet(identity) -> WritingPacket:
    return WritingPacket(
        packet_id="packet-writer-loop",
        identity=identity,
        session_id=identity.session_id,
        branch_head_id=identity.branch_head_id,
        turn_id=identity.turn_id,
        chapter_workspace_id="chapter-1",
        output_kind="story_segment",
        phase="segment_drafting",
        operation_mode="writing",
        system_sections=["You are the writing worker."],
        context_sections=[{"label": "core", "items": ["State A"]}],
        user_instruction="Write the next segment.",
        metadata={
            "writer_retrieval_allowed": True,
            "writer_max_retrieval_attempts": 2,
        },
    )


class _ToolLoopGateway:
    def __init__(self, responses: list[dict]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def supports_tools(self, **_kwargs) -> bool:
        return True

    async def complete_with_tools(self, **kwargs) -> dict:
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("No queued tool-loop response")
        return self._responses.pop(0)

    async def complete_text_with_usage(self, **kwargs):
        self.calls.append(kwargs)
        return "fallback text", {
            "prompt_tokens": 3,
            "completion_tokens": 5,
            "total_tokens": 8,
        }


class _OneShotGateway:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def complete_text_with_usage(self, **kwargs):
        self.calls.append(kwargs)
        return "one shot output", {
            "prompt_tokens": 7,
            "completion_tokens": 9,
            "total_tokens": 16,
        }


def _tool_call(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json_dumps(arguments),
        },
    }


def json_dumps(payload: dict) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False)


@pytest.mark.asyncio
async def test_writing_worker_execution_service_keeps_one_shot_path_when_retrieval_disabled(
    retrieval_session,
):
    identity = _build_identity(retrieval_session)
    packet = _build_packet(identity).model_copy(
        update={
            "metadata": {
                "writer_retrieval_allowed": False,
                "writer_max_retrieval_attempts": 0,
            }
        }
    )
    gateway = _OneShotGateway()
    service = WritingWorkerExecutionService(
        llm_gateway=gateway,
        runtime_retrieval_card_service=RuntimeRetrievalCardService(
            session=retrieval_session
        ),
    )

    result = await service.execute(
        request=WritingWorkerExecutionRequest(
            request_id="writer-one-shot",
            identity=identity,
            operation_mode="writing",
            packet=packet,
            writer_model_id="model",
            writer_provider_id="provider",
            retrieval_allowed=False,
            max_retrieval_attempts=0,
        )
    )

    assert result.output_text == "one shot output"
    assert result.writer_tool_trace_refs == []
    assert result.retrieval_source_ref_bundle.is_empty()


@pytest.mark.asyncio
async def test_writing_worker_execution_service_runs_search_expand_usage_then_final(
    retrieval_session,
):
    identity = _build_identity(retrieval_session)
    packet = _build_packet(identity)
    gateway = _ToolLoopGateway(
        [
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                _tool_call(
                                    "call_search",
                                    "retrieval.search",
                                    {
                                        "query": "storm",
                                        "mode": "semantic",
                                        "lexical_anchors": ["storm"],
                                        "semantic_predicates": [],
                                    },
                                )
                            ],
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                _tool_call(
                                    "call_expand",
                                    "retrieval.expand",
                                    {"card_short_ids": ["R1"]},
                                )
                            ],
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": 4,
                    "total_tokens": 12,
                },
            },
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                _tool_call(
                                    "call_usage",
                                    "retrieval.usage",
                                    {
                                        "used_card_short_ids": ["R1"],
                                        "used_expanded_short_ids": ["X1"],
                                        "knowledge_gaps": [
                                            {
                                                "query": "storm aftermath",
                                                "status": "insufficient_detail",
                                                "impact": "avoid naming the exact ruin",
                                                "mode_policy_resolution": "continue_conservatively",
                                            }
                                        ],
                                    },
                                )
                            ],
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 6,
                    "completion_tokens": 3,
                    "total_tokens": 9,
                },
            },
            {
                "choices": [
                    {
                        "message": {
                            "content": "Final segment grounded in retrieved evidence."
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 18,
                    "total_tokens": 30,
                },
            },
        ]
    )
    retrieval_service = RuntimeRetrievalCardService(session=retrieval_session)
    service = WritingWorkerExecutionService(
        llm_gateway=gateway,
        runtime_retrieval_card_service=retrieval_service,
    )

    result = await service.execute(
        request=WritingWorkerExecutionRequest(
            request_id="writer-loop-success",
            identity=identity,
            operation_mode="writing",
            packet=packet,
            writer_model_id="model",
            writer_provider_id="provider",
            retrieval_allowed=True,
            max_retrieval_attempts=2,
        )
    )

    assert result.output_text == "Final segment grounded in retrieved evidence."
    assert result.usage_metadata["total_tokens"] == 66
    assert len(result.writer_tool_trace_refs) >= 6
    assert result.retrieval_source_ref_bundle.retrieval_card_material_ids
    assert result.retrieval_source_ref_bundle.retrieval_expanded_chunk_material_ids
    assert result.retrieval_source_ref_bundle.retrieval_usage_material_ids

    usage_material = retrieval_service._workspace().require_material(
        identity=identity,
        material_id=result.retrieval_source_ref_bundle.retrieval_usage_material_ids[0],
    )
    assert (
        usage_material.material_kind
        == RuntimeWorkspaceMaterialKind.RETRIEVAL_USAGE_RECORD
    )
    assert usage_material.payload["used_card_short_ids"] == ["R1"]
    assert usage_material.payload["expanded_card_short_ids"] == ["R1"]


@pytest.mark.asyncio
async def test_writing_worker_execution_service_fails_closed_when_final_output_skips_usage(
    retrieval_session,
):
    identity = _build_identity(retrieval_session)
    packet = _build_packet(identity)
    gateway = _ToolLoopGateway(
        [
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                _tool_call(
                                    "call_search",
                                    "retrieval.search",
                                    {"query": "storm", "mode": "semantic"},
                                )
                            ],
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 9,
                    "completion_tokens": 4,
                    "total_tokens": 13,
                },
            },
            {
                "choices": [
                    {
                        "message": {
                            "content": "I wrote the answer without recording usage."
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": 7,
                    "total_tokens": 15,
                },
            },
        ]
    )
    service = WritingWorkerExecutionService(
        llm_gateway=gateway,
        runtime_retrieval_card_service=RuntimeRetrievalCardService(
            session=retrieval_session
        ),
    )

    with pytest.raises(WritingWorkerRetrievalLoopServiceError) as exc_info:
        await service.execute(
            request=WritingWorkerExecutionRequest(
                request_id="writer-loop-fail-closed",
                identity=identity,
                operation_mode="writing",
                packet=packet,
                writer_model_id="model",
                writer_provider_id="provider",
                retrieval_allowed=True,
                max_retrieval_attempts=2,
            )
        )
    assert exc_info.value.code == "writer_retrieval_usage_required_before_final_output"


@pytest.mark.asyncio
async def test_writing_worker_execution_service_enforces_attempt_limit(
    retrieval_session,
):
    identity = _build_identity(retrieval_session)
    packet = _build_packet(identity)
    gateway = _ToolLoopGateway(
        [
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                _tool_call(
                                    "call_search_1",
                                    "retrieval.search",
                                    {"query": "storm", "mode": "semantic"},
                                )
                            ],
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 3,
                    "total_tokens": 8,
                },
            },
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                _tool_call(
                                    "call_search_2",
                                    "retrieval.search",
                                    {"query": "storm aftermath", "mode": "semantic"},
                                )
                            ],
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 3,
                    "total_tokens": 8,
                },
            },
        ]
    )
    service = WritingWorkerExecutionService(
        llm_gateway=gateway,
        runtime_retrieval_card_service=RuntimeRetrievalCardService(
            session=retrieval_session
        ),
    )

    with pytest.raises(WritingWorkerRetrievalLoopServiceError) as exc_info:
        await service.execute(
            request=WritingWorkerExecutionRequest(
                request_id="writer-loop-attempt-limit",
                identity=identity,
                operation_mode="writing",
                packet=packet,
                writer_model_id="model",
                writer_provider_id="provider",
                retrieval_allowed=True,
                max_retrieval_attempts=1,
            )
        )
    assert exc_info.value.code == "writer_retrieval_attempt_limit_exceeded"


@pytest.mark.asyncio
async def test_runtime_retrieval_card_service_expand_cards_by_refs_accepts_short_ids(
    retrieval_session,
):
    identity = _build_identity(retrieval_session)
    service = RuntimeRetrievalCardService(session=retrieval_session)

    _, cards, _ = await service.search_recall_to_cards(
        identity=identity,
        input_model=MemorySearchRecallInput(
            query="storm",
            scope="story",
            domains=[Domain.CHAPTER],
        ),
        actor="writer.retrieval",
    )
    expanded = service.expand_cards_by_refs(
        identity=identity,
        card_refs=["R1"],
        actor="writer.retrieval",
    )
    assert cards[0].short_id == "R1"
    assert expanded[0].short_id == "X1"
    assert (
        expanded[0].visibility
        == RuntimeWorkspaceMaterialVisibility.WRITER_VISIBLE.value
    )
