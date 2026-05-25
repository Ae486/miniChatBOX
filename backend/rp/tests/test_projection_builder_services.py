"""Tests for Phase E3 settled projection refresh and builder context flow."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from types import SimpleNamespace
from typing import Any, cast

import pytest
from sqlalchemy import desc
from sqlmodel import select

from models.rp_story_store import (
    BranchHeadRecord,
    RuntimeWorkflowJobRecord,
    StoryTurnRecord,
)
from rp.models.dsl import Domain, Layer, ObjectRef
from rp.models.memory_contract_registry import MemoryRuntimeIdentity
from rp.models.memory_crud import RetrievalHit
from rp.models.projection_refresh import (
    ProjectionRefreshRequest,
    ProjectionRefreshServiceError,
)
from rp.models.story_runtime import (
    LongformChapterPhase,
    LongformTurnCommandKind,
    LongformTurnRequest,
    OrchestratorPlan,
    SpecialistResultBundle,
    StorySegmentStructuredMetadata,
    StoryArtifactStatus,
    StoryArtifactKind,
)
from rp.models.story_brainstorm import (
    BrainstormBatchSubmitRequest,
    BrainstormDiscussionRequest,
    BrainstormItemStatus,
    BrainstormItemUpdateRequest,
    BrainstormSessionStartRequest,
    BrainstormSummarizeRequest,
)
from rp.models.runtime_workspace_material import (
    RuntimeWorkspaceMaterial,
    RuntimeWorkspaceMaterialKind,
    RuntimeWorkspaceMaterialLifecycle,
)
from rp.models.runtime_identity import StoryTurnStatus
from rp.models.runtime_workflow_job import RuntimeWorkflowJobStatus
from rp.models.writing_worker_contracts import (
    WritingWorkerExecutionRequest,
    WritingWorkerExecutionResult,
)
from rp.models.writing_runtime import WritingPacket
from rp.graphs.story_graph_nodes import StoryGraphNodes
from rp.graphs.story_graph_runner import StoryGraphRunner
from rp.services.authoritative_state_view_service import AuthoritativeStateViewService
from rp.services.builder_projection_context_service import (
    BuilderProjectionContextService,
)
from rp.services.chapter_workspace_projection_adapter import (
    ChapterWorkspaceProjectionAdapter,
)
from rp.services.context_orchestration_service import ContextOrchestrationService
from rp.services.longform_orchestrator_service import LongformOrchestratorService
from rp.services.longform_specialist_service import LongformSpecialistService
from rp.services.memory_inspection_read_service import MemoryInspectionReadService
from rp.services.post_write_governance_service import PostWriteGovernanceService
from rp.services.post_write_scheduler_service import (
    POST_WRITE_MAINTENANCE_PHASE,
    PostWriteSchedulerService,
)
from rp.services.projection_state_service import ProjectionStateService
from rp.services.projection_refresh_service import ProjectionRefreshService
from rp.services.proposal_repository import ProposalRepository
from rp.services.rp_block_read_service import RpBlockReadService
from rp.services.runtime_profile_snapshot_service import RuntimeProfileSnapshotService
from rp.services.runtime_retrieval_card_service import RuntimeRetrievalCardService
from rp.services.runtime_workflow_job_service import RuntimeWorkflowJobService
from rp.services.runtime_workspace_material_service import (
    RuntimeWorkspaceMaterialService,
)
from rp.services.story_block_consumer_state_service import (
    StoryBlockConsumerStateService,
)
from rp.services.story_block_prompt_compile_service import (
    StoryBlockPromptCompileService,
)
from rp.services.story_brainstorm_service import StoryBrainstormService
from rp.services.story_block_prompt_context_service import (
    StoryBlockPromptContextService,
)
from rp.services.story_block_prompt_render_service import (
    StoryBlockPromptRenderService,
)
from rp.services.story_runtime_identity_service import StoryRuntimeIdentityService
from rp.services.story_session_core_state_adapter import StorySessionCoreStateAdapter
from rp.services.story_session_service import StorySessionService
from rp.services.story_turn_domain_service import StoryTurnDomainService
from rp.services.version_history_read_service import VersionHistoryReadService
from rp.services.worker_execution_service import WorkerExecutionOutcome
from rp.services.worker_scheduler_service import PRE_WRITE_CONTEXT_PHASE
from rp.services.writing_worker_execution_service import WritingWorkerExecutionService
from rp.services.writing_packet_builder import WritingPacketBuilder
from rp.models.worker_runtime_contracts import (
    WorkerExecutionItem,
    WorkerExecutionPlan,
    WorkerPlanSource,
    WorkerResult,
    WorkerResultStatus,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class _BrainstormTestGateway:
    async def complete_text_with_usage(self, **kwargs):
        system_prompt = str(kwargs["messages"][0].content)
        if "brainstorm_summarize" in system_prompt:
            return (
                '{"items":["保留钟楼债务伏笔","强化使者与旧账册的关联"]}',
                {"prompt_tokens": 8, "completion_tokens": 6, "total_tokens": 14},
            )
        return (
            "我们先把钟楼债务保留成未解释的异常细节。",
            {"prompt_tokens": 7, "completion_tokens": 10, "total_tokens": 17},
        )

    @staticmethod
    def extract_json_object(text: str):
        return json.loads(text)


def _seed_story_runtime(retrieval_session):
    service = StorySessionService(retrieval_session)
    session = service.create_session(
        story_id="story-1",
        source_workspace_id="workspace-1",
        mode="longform",
        runtime_story_config={},
        writer_contract={"style_rules": ["Lean"]},
        current_state_json={
            "chapter_digest": {"current_chapter": 1, "title": "Chapter One"},
            "narrative_progress": {
                "current_phase": "outline_drafting",
                "accepted_segments": 0,
            },
            "timeline_spine": [],
            "active_threads": [],
            "foreshadow_registry": [],
            "character_state_digest": {},
        },
        initial_phase=LongformChapterPhase.OUTLINE_DRAFTING,
    )
    service.create_chapter_workspace(
        session_id=session.session_id,
        chapter_index=1,
        phase=LongformChapterPhase.OUTLINE_DRAFTING,
        builder_snapshot_json={
            "foundation_digest": ["Found A"],
            "blueprint_digest": ["Blueprint A"],
            "current_outline_digest": ["Outline A"],
            "recent_segment_digest": ["Segment A"],
            "current_state_digest": ["State A"],
            "writer_hints": ["Persisted Hint"],
        },
    )
    service.commit()
    return (
        service.get_session(session.session_id),
        service.get_chapter_by_index(
            session_id=session.session_id,
            chapter_index=1,
        ),
        service,
    )


def _build_boundary_services(
    service: StorySessionService,
) -> tuple[
    AuthoritativeStateViewService,
    ProjectionStateService,
]:
    authoritative_state_view_service = AuthoritativeStateViewService(
        adapter=StorySessionCoreStateAdapter(service)
    )
    projection_state_service = ProjectionStateService(
        story_session_service=service,
        adapter=ChapterWorkspaceProjectionAdapter(service),
    )
    return authoritative_state_view_service, projection_state_service


class _NoopRegressionService:
    async def run_light_regression(
        self,
        *,
        session,
        chapter,
        accepted_artifact,
        model_id,
        provider_id,
        runtime_identity=None,
    ):
        return session, chapter

    async def run_heavy_regression(
        self, *, session, chapter, model_id, provider_id, runtime_identity=None
    ):
        return session, chapter


class _RecordingBlockConsumerStateService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def mark_consumer_synced(self, *, session_id: str, consumer_key: str):
        self.calls.append((session_id, consumer_key))
        return None


class _RecordingStoryLlmGateway:
    def __init__(self, response_text: str) -> None:
        self._response_text = response_text
        self.calls: list[dict] = []

    async def complete_text(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return self._response_text

    async def complete_text_with_usage(self, **kwargs):
        self.calls.append(kwargs)
        return self._response_text, {
            "prompt_tokens": 11,
            "completion_tokens": 17,
            "total_tokens": 28,
        }

    def supports_tools(self, **_kwargs) -> bool:
        return False

    @staticmethod
    def extract_json_object(raw: str) -> dict:
        return json.loads(raw)


class _StreamingStoryLlmGateway(_RecordingStoryLlmGateway):
    async def stream_text(self, **_kwargs):
        yield 'data: {"type":"text_delta","delta":"Streamed outline."}\n\n'
        yield (
            'data: {"type":"usage","prompt_tokens":42,'
            '"completion_tokens":11,"total_tokens":53}\n\n'
        )
        yield 'data: {"type":"done"}\n\n'


class _ToolLoopStoryLlmGateway(_RecordingStoryLlmGateway):
    def __init__(self, responses: list[dict]) -> None:
        super().__init__(response_text="")
        self._responses = list(responses)

    def supports_tools(self, **_kwargs) -> bool:
        return True

    async def complete_with_tools(self, **kwargs) -> dict:
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("No more tool-loop responses queued")
        return self._responses.pop(0)

    async def stream_text(self, **_kwargs):
        raise AssertionError(
            "buffered retrieval stream path should not call raw stream_text"
        )


def _tool_loop_call(
    call_id: str, name: str, arguments: dict[str, object]
) -> dict[str, object]:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }


class _StubMemoryOsService:
    def __init__(
        self,
        *,
        archival_hits: list[RetrievalHit] | None = None,
        recall_hits: list[RetrievalHit] | None = None,
    ) -> None:
        self._archival_hits = list(archival_hits or [])
        self._recall_hits = list(recall_hits or [])

    async def search_archival(self, *_args, **_kwargs):
        return SimpleNamespace(hits=list(self._archival_hits))

    async def search_recall(self, *_args, **_kwargs):
        return SimpleNamespace(hits=list(self._recall_hits))


class _RecordingMemoryOsService(_StubMemoryOsService):
    def __init__(self) -> None:
        super().__init__()
        self.archival_inputs: list[Any] = []
        self.recall_inputs: list[Any] = []

    async def search_archival(self, input_model, *_args, **_kwargs):
        self.archival_inputs.append(input_model)
        return await super().search_archival(input_model, *_args, **_kwargs)

    async def search_recall(self, input_model, *_args, **_kwargs):
        self.recall_inputs.append(input_model)
        return await super().search_recall(input_model, *_args, **_kwargs)


class _FailingMemoryOsService:
    async def search_archival(self, *_args, **_kwargs):
        raise AssertionError("retrieval should not run for this plan")

    async def search_recall(self, *_args, **_kwargs):
        raise AssertionError("retrieval should not run for this plan")


def _build_turn_domain_service(
    service: StorySessionService,
    *,
    orchestrator_service=None,
    specialist_service=None,
    writing_worker_execution_service=None,
    block_consumer_state_service=None,
    runtime_identity_service=None,
    runtime_workspace_material_service=None,
    worker_scheduler_service=None,
    worker_execution_service=None,
    context_orchestration_service=None,
    runtime_workflow_job_service=None,
    post_write_scheduler_service=None,
    post_write_governance_service=None,
) -> StoryTurnDomainService:
    authoritative_state_view_service, projection_state_service = (
        _build_boundary_services(service)
    )
    return StoryTurnDomainService(
        story_session_service=service,
        orchestrator_service=orchestrator_service or SimpleNamespace(),
        specialist_service=cast(
            LongformSpecialistService,
            specialist_service or SimpleNamespace(),
        ),
        builder_projection_context_service=BuilderProjectionContextService(
            projection_state_service
        ),
        projection_state_service=projection_state_service,
        writing_packet_builder=WritingPacketBuilder(),
        writing_worker_execution_service=cast(
            WritingWorkerExecutionService,
            writing_worker_execution_service or SimpleNamespace(),
        ),
        regression_service=_NoopRegressionService(),
        block_consumer_state_service=block_consumer_state_service,
        runtime_identity_service=runtime_identity_service,
        runtime_workspace_material_service=runtime_workspace_material_service,
        worker_scheduler_service=worker_scheduler_service,
        worker_execution_service=worker_execution_service,
        context_orchestration_service=context_orchestration_service,
        runtime_workflow_job_service=runtime_workflow_job_service,
        post_write_scheduler_service=post_write_scheduler_service,
        post_write_governance_service=post_write_governance_service,
    )


def test_authoritative_state_view_service_reads_session_scoped_objects(
    retrieval_session,
):
    session, _, service = _seed_story_runtime(retrieval_session)
    authoritative_state_view_service, _ = _build_boundary_services(service)

    chapter_digest = authoritative_state_view_service.get_chapter_digest(
        session_id=session.session_id
    )
    narrative_progress = authoritative_state_view_service.get_narrative_progress(
        session_id=session.session_id
    )

    assert chapter_digest == {"current_chapter": 1, "title": "Chapter One"}
    assert (
        narrative_progress["current_phase"]
        == LongformChapterPhase.OUTLINE_DRAFTING.value
    )


def test_projection_state_service_updates_slots_and_rollover(retrieval_session):
    session, chapter, service = _seed_story_runtime(retrieval_session)
    _, projection_state_service = _build_boundary_services(service)

    projection_state_service.set_current_outline(
        chapter_workspace_id=chapter.chapter_workspace_id,
        outline_text="Fresh Outline",
    )
    projection_state_service.append_recent_segment(
        chapter_workspace_id=chapter.chapter_workspace_id,
        excerpt="Segment B",
    )
    next_chapter = service.create_chapter_workspace(
        session_id=session.session_id,
        chapter_index=2,
        phase=LongformChapterPhase.OUTLINE_DRAFTING,
        chapter_goal="Chapter 2",
    )
    projection_state_service.seed_next_chapter(
        previous_chapter_workspace_id=chapter.chapter_workspace_id,
        next_chapter_workspace_id=next_chapter.chapter_workspace_id,
        next_chapter_index=2,
    )

    updated_chapter = service.get_chapter_workspace(chapter.chapter_workspace_id)
    seeded_chapter = service.get_chapter_workspace(next_chapter.chapter_workspace_id)

    assert updated_chapter is not None
    assert seeded_chapter is not None
    assert updated_chapter.builder_snapshot_json["current_outline_digest"] == [
        "Fresh Outline"
    ]
    assert updated_chapter.builder_snapshot_json["recent_segment_digest"] == [
        "Segment A",
        "Segment B",
    ]
    assert seeded_chapter.builder_snapshot_json["blueprint_digest"] == ["Blueprint A"]
    assert seeded_chapter.builder_snapshot_json["current_outline_digest"] == []
    assert seeded_chapter.builder_snapshot_json["recent_segment_digest"] == []
    assert seeded_chapter.builder_snapshot_json["chapter_index"] == 2


def test_builder_projection_context_service_ignores_writer_hints(retrieval_session):
    session, _, service = _seed_story_runtime(retrieval_session)
    _, projection_state_service = _build_boundary_services(service)
    context_service = BuilderProjectionContextService(projection_state_service)

    context_sections = context_service.build_context_sections(
        session_id=session.session_id
    )

    assert [section["label"] for section in context_sections] == [
        "foundation_digest",
        "blueprint_digest",
        "current_outline_digest",
        "recent_segment_digest",
        "current_state_digest",
    ]


def test_writing_packet_builder_uses_projection_sections_and_runtime_hints(
    retrieval_session,
):
    session, chapter, service = _seed_story_runtime(retrieval_session)
    _, projection_state_service = _build_boundary_services(service)
    context_service = BuilderProjectionContextService(projection_state_service)
    builder = WritingPacketBuilder()

    packet = builder.build(
        session=session,
        chapter=chapter,
        plan=OrchestratorPlan(
            output_kind=StoryArtifactKind.STORY_SEGMENT,
            writer_instruction="Write the next segment.",
        ),
        runtime_identity=None,
        operation_mode="writing",
        projection_context_sections=context_service.build_context_sections(
            session_id=session.session_id
        ),
        recent_raw_turn_sections=[
            {
                "label": "recent_raw_turns",
                "items": [
                    "user: Keep the storm callback present.",
                    "assistant: The envoy debt should resurface soon.",
                ],
            }
        ],
        runtime_writer_hints=["Runtime Hint"],
        runtime_retrieval_sections=[
            {
                "label": "retrieval_cards",
                "items": ["R1 [retrieval_card] Storm Callback: callback evidence."],
            }
        ],
        review_overlay_sections=[
            {
                "label": "review_overlay",
                "items": ["Preserve the bell tower rewrite intent."],
            }
        ],
        user_instruction="Write the next segment.",
    )

    assert [section.label for section in packet.core_view_sections] == [
        "foundation_digest",
        "blueprint_digest",
        "current_outline_digest",
        "recent_segment_digest",
        "current_state_digest",
    ]
    assert [section.label for section in packet.recent_raw_turn_sections] == [
        "recent_raw_turns"
    ]
    assert [section.label for section in packet.mode_sidecar_sections] == [
        "writer_hints"
    ]
    assert [section.label for section in packet.retrieval_card_sections] == [
        "retrieval_cards"
    ]
    assert [section.label for section in packet.review_overlay_sections] == [
        "review_overlay"
    ]
    assert [section["label"] for section in packet.context_sections] == [
        "foundation_digest",
        "blueprint_digest",
        "current_outline_digest",
        "recent_segment_digest",
        "current_state_digest",
        "recent_raw_turns",
        "writer_hints",
        "retrieval_cards",
        "review_overlay",
    ]
    assert packet.mode_sidecar_sections[-1].items == ["Runtime Hint"]
    assert packet.packet_summary_metadata["section_counts"] == {
        "core_view_sections": 5,
        "recent_raw_turn_sections": 1,
        "mode_sidecar_sections": 1,
        "retrieval_card_sections": 1,
        "review_overlay_sections": 1,
    }
    assert packet.metadata["packet_builder_boundary"] == "thin_context_packet_builder"
    assert packet.metadata["legacy_orchestrator_plan_role"] == "adapter_input"


def test_writing_packet_builder_has_no_raw_retrieval_hit_surface(
    retrieval_session,
):
    session, chapter, service = _seed_story_runtime(retrieval_session)
    _, projection_state_service = _build_boundary_services(service)
    context_service = BuilderProjectionContextService(projection_state_service)
    builder = WritingPacketBuilder()

    packet = builder.build(
        session=session,
        chapter=chapter,
        plan=OrchestratorPlan(
            output_kind=StoryArtifactKind.STORY_SEGMENT,
            writer_instruction="Write the next segment.",
        ),
        runtime_identity=None,
        operation_mode="writing",
        projection_context_sections=context_service.build_context_sections(
            session_id=session.session_id
        ),
        runtime_writer_hints=["Composed specialist continuity hint."],
        user_instruction="Write the next segment.",
    )
    payload = packet.model_dump(mode="json")

    assert "retrieval_hits" not in payload
    assert "archival_hits" not in payload
    assert "recall_hits" not in payload
    assert packet.mode_sidecar_sections[-1].label == "writer_hints"
    assert packet.mode_sidecar_sections[-1].source_kind == "worker_hint_digest"
    assert packet.mode_sidecar_sections[-1].items == [
        "Composed specialist continuity hint."
    ]


@pytest.mark.asyncio
async def test_writing_worker_execution_service_returns_structured_result_and_usage():
    packet = WritingPacket(
        packet_id="packet-1",
        session_id="session-1",
        branch_head_id="branch-1",
        turn_id="turn-1",
        chapter_workspace_id="chapter-1",
        output_kind="story_segment",
        phase="segment_drafting",
        operation_mode="rewrite",
        system_sections=["You are the writing worker."],
        context_sections=[{"label": "core", "items": ["State A"]}],
        user_instruction="Rewrite the segment.",
    )
    service = WritingWorkerExecutionService(
        llm_gateway=_RecordingStoryLlmGateway("Rewritten output.")
    )

    result = await service.execute(
        request=WritingWorkerExecutionRequest(
            request_id="writer-exec-1",
            operation_mode="rewrite",
            packet=packet,
            writer_model_id="model",
            writer_provider_id="provider",
        )
    )

    assert result.output_text == "Rewritten output."
    assert result.operation_mode == "rewrite"
    assert result.result_status == "completed"
    assert result.usage_metadata["total_tokens"] == 28


def test_projection_refresh_service_updates_settled_slots_only(retrieval_session):
    _, chapter, service = _seed_story_runtime(retrieval_session)
    refresh_service = ProjectionRefreshService(service)

    updated_chapter = refresh_service.refresh_from_bundle(
        chapter=chapter,
        bundle=SpecialistResultBundle(
            foundation_digest=["New Found"],
            blueprint_digest=["New Blueprint"],
            current_outline_digest=["New Outline"],
            recent_segment_digest=["New Segment"],
            current_state_digest=["New State"],
        ),
    )

    assert updated_chapter.builder_snapshot_json["foundation_digest"] == ["New Found"]
    assert "writer_hints" not in updated_chapter.builder_snapshot_json


def test_projection_refresh_service_requires_identity_for_explicit_runtime_request(
    retrieval_session,
):
    _, chapter, service = _seed_story_runtime(retrieval_session)
    refresh_service = ProjectionRefreshService(service)

    with pytest.raises(ProjectionRefreshServiceError) as exc_info:
        refresh_service.refresh_from_bundle(
            chapter=chapter,
            bundle=SpecialistResultBundle(
                foundation_digest=["New Found"],
                blueprint_digest=["New Blueprint"],
                current_outline_digest=["New Outline"],
                recent_segment_digest=["New Segment"],
                current_state_digest=["New State"],
            ),
            refresh_request=ProjectionRefreshRequest(),
        )

    assert exc_info.value.code == "projection_refresh_identity_required"


def test_story_turn_domain_service_build_packet_attaches_deterministic_read_manifest(
    retrieval_session,
):
    session, _, service = _seed_story_runtime(retrieval_session)
    service.create_discussion_entry(
        session_id=session.session_id,
        chapter_workspace_id=service.get_current_chapter(
            session.session_id
        ).chapter_workspace_id,
        role="user",
        content_text="Keep the storm callback alive.",
    )
    service.commit()
    snapshot_service = RuntimeProfileSnapshotService(retrieval_session)
    snapshot = snapshot_service.ensure_active_snapshot(
        session_id=session.session_id,
        created_from="test.packet_manifest",
    )
    identity_service = StoryRuntimeIdentityService(
        retrieval_session,
        runtime_profile_snapshot_service=snapshot_service,
    )
    runtime_identity = identity_service.resolve_runtime_entry_identity(
        session_id=session.session_id,
        command_kind=LongformTurnCommandKind.WRITE_NEXT_SEGMENT.value,
        actor="story_runtime",
        requested_runtime_profile_snapshot_id=snapshot.runtime_profile_snapshot_id,
    )
    RuntimeWorkspaceMaterialService(session=retrieval_session).record_material(
        RuntimeWorkspaceMaterial(
            material_id="packet-card-R1",
            material_kind=RuntimeWorkspaceMaterialKind.RETRIEVAL_CARD,
            identity=runtime_identity,
            domain="chapter",
            domain_path="runtime.packet.card",
            payload={
                "title": "Packet Card",
                "summary": "A recalled storm callback matters here.",
            },
            visibility="writer_visible",
            created_by="worker.retrieval",
        )
    )
    turn_domain_service = _build_turn_domain_service(
        service,
        runtime_identity_service=identity_service,
        runtime_workspace_material_service=RuntimeWorkspaceMaterialService(
            session=retrieval_session
        ),
    )
    bundle = SpecialistResultBundle(
        foundation_digest=["Found A"],
        blueprint_digest=["Blueprint A"],
        current_outline_digest=["Outline A"],
        recent_segment_digest=["Segment A"],
        current_state_digest=["State A"],
        writer_hints=["Hint A"],
    )

    packet_a = turn_domain_service.build_packet(
        session_id=session.session_id,
        plan=OrchestratorPlan(
            output_kind=StoryArtifactKind.STORY_SEGMENT,
            writer_instruction="Write the next segment.",
        ),
        specialist_bundle=bundle,
        runtime_identity=runtime_identity,
    )
    packet_b = turn_domain_service.build_packet(
        session_id=session.session_id,
        plan=OrchestratorPlan(
            output_kind=StoryArtifactKind.STORY_SEGMENT,
            writer_instruction="Write the next segment.",
        ),
        specialist_bundle=bundle,
        runtime_identity=runtime_identity,
    )

    manifest_a = packet_a.metadata["runtime_read_manifest"]
    manifest_b = packet_b.metadata["runtime_read_manifest"]

    assert manifest_a["manifest_id"] == manifest_b["manifest_id"]
    assert manifest_a["identity"]["turn_id"] == runtime_identity.turn_id
    assert manifest_a["active_branch_lineage"] == [runtime_identity.branch_head_id]
    assert [section["label"] for section in manifest_a["packet_sections"]] == [
        "foundation_digest",
        "blueprint_digest",
        "current_outline_digest",
        "recent_segment_digest",
        "current_state_digest",
        "recent_raw_turns",
        "writer_hints",
        "retrieval_cards",
    ]
    assert {item["packet_section_label"] for item in manifest_a["selected_refs"]} >= {
        "recent_raw_turns",
        "retrieval_cards",
    }
    assert manifest_a["retrieval_card_refs"] == ["packet-card-R1"]
    assert any(
        item["reason"] == "packet_visible_runtime_workspace_only"
        for item in manifest_a["omitted_refs"]
    )


def test_context_orchestration_service_builds_minimal_worker_context_packet(
    retrieval_session,
):
    session, chapter, service = _seed_story_runtime(retrieval_session)
    first = service.create_discussion_entry(
        session_id=session.session_id,
        chapter_workspace_id=chapter.chapter_workspace_id,
        role="user",
        content_text="Keep the storm callback alive.",
    )
    second = service.create_discussion_entry(
        session_id=session.session_id,
        chapter_workspace_id=chapter.chapter_workspace_id,
        role="assistant",
        content_text="The bell tower reveal can wait.",
    )
    service.commit()
    snapshot = RuntimeProfileSnapshotService(retrieval_session).ensure_active_snapshot(
        session_id=session.session_id,
        created_from="test.context_orchestration.worker_packet",
    )
    identity = MemoryRuntimeIdentity(
        story_id=session.story_id,
        session_id=session.session_id,
        branch_head_id="branch:test:main",
        turn_id="turn:test:context-packet",
        runtime_profile_snapshot_id=snapshot.runtime_profile_snapshot_id,
    )
    material_service = RuntimeWorkspaceMaterialService(session=retrieval_session)
    material_service.record_material(
        RuntimeWorkspaceMaterial(
            material_id="packet-card-R1",
            material_kind=RuntimeWorkspaceMaterialKind.RETRIEVAL_CARD,
            identity=identity,
            domain="chapter",
            domain_path="chapter.runtime.packet.card",
            short_id="R1",
            payload={
                "summary": "A recalled storm callback matters here.",
                "title": "Storm Callback",
            },
            visibility="writer_visible",
            created_by="worker.specialist",
        )
    )
    material_service.record_material(
        RuntimeWorkspaceMaterial(
            material_id="packet-expanded-X1",
            material_kind=RuntimeWorkspaceMaterialKind.RETRIEVAL_EXPANDED_CHUNK,
            identity=identity,
            domain="chapter",
            domain_path="chapter.runtime.packet.expanded",
            short_id="X1",
            payload={
                "summary": "Expanded detail: the seal broke during the first storm.",
                "text": "Expanded detail: the seal broke during the first storm.",
                "title": "Storm Callback Expanded",
            },
            lifecycle=RuntimeWorkspaceMaterialLifecycle.EXPANDED,
            visibility="writer_visible",
            created_by="worker.specialist",
        )
    )
    _, projection_state_service = _build_boundary_services(service)
    orchestration = ContextOrchestrationService(
        story_session_service=service,
        builder_projection_context_service=BuilderProjectionContextService(
            projection_state_service
        ),
        writing_packet_builder=WritingPacketBuilder(),
        runtime_workspace_material_service=material_service,
    )

    packet = orchestration.build_worker_context_packet(
        session=session,
        chapter=chapter,
        identity=identity,
        worker_id="longform_memory_worker",
        phase=PRE_WRITE_CONTEXT_PHASE,
        mode=session.mode,
        context_requirements={"focus": "continuity"},
        reason_codes=["selected_by_bootstrap_phase_policy"],
        budget_class="default",
    )

    assert packet.session_refs == [
        f"story_session:{session.session_id}",
        f"chapter_workspace:{chapter.chapter_workspace_id}",
    ]
    assert packet.recent_turn_refs == [first.entry_id, second.entry_id]
    assert packet.core_projection_refs == [
        "projection_slot:foundation_digest",
        "projection_slot:blueprint_digest",
        "projection_slot:current_outline_digest",
        "projection_slot:recent_segment_digest",
        "projection_slot:current_state_digest",
    ]
    assert packet.retrieval_refs == ["packet-card-R1", "packet-expanded-X1"]
    assert packet.workspace_refs == ["packet-card-R1", "packet-expanded-X1"]
    assert packet.packet_metadata["context_requirements"] == {"focus": "continuity"}
    assert packet.packet_metadata["worker_source_ref_bundle"] == {
        "retrieval_card_material_ids": ["packet-card-R1"],
        "retrieval_expanded_chunk_material_ids": ["packet-expanded-X1"],
        "retrieval_usage_material_ids": [],
    }


def test_story_turn_domain_service_build_packet_routes_through_context_orchestration_service(
    retrieval_session,
):
    session, chapter, service = _seed_story_runtime(retrieval_session)
    runtime_identity = MemoryRuntimeIdentity(
        story_id=session.story_id,
        session_id=session.session_id,
        branch_head_id="branch:test:main",
        turn_id="turn:test:writer",
        runtime_profile_snapshot_id="snapshot:test:writer",
    )
    packet = WritingPacketBuilder().build(
        session=session,
        chapter=chapter,
        plan=OrchestratorPlan(
            output_kind=StoryArtifactKind.STORY_SEGMENT,
            writer_instruction="Write the next segment.",
        ),
        runtime_identity=runtime_identity,
        operation_mode="writing",
        projection_context_sections=[
            {"label": "foundation_digest", "items": ["Found A"]}
        ],
        runtime_writer_hints=["Hint A"],
        user_instruction="Write the next segment.",
        packet_metadata={
            "runtime_identity": runtime_identity.model_dump(mode="json"),
        },
    )
    orchestration = _RecordingContextOrchestrationService(writing_packet=packet)
    turn_domain_service = _build_turn_domain_service(
        service,
        context_orchestration_service=orchestration,
    )
    bundle = SpecialistResultBundle(writer_hints=["Hint A"])

    actual = turn_domain_service.build_packet(
        session_id=session.session_id,
        plan=OrchestratorPlan(
            output_kind=StoryArtifactKind.STORY_SEGMENT,
            writer_instruction="Write the next segment.",
        ),
        specialist_bundle=bundle,
        runtime_identity=runtime_identity,
    )

    assert actual == packet
    assert len(orchestration.writer_calls) == 1
    writer_call = cast(dict[str, Any], orchestration.writer_calls[0])
    assert writer_call["session"].session_id == session.session_id
    assert writer_call["chapter"].chapter_workspace_id == (chapter.chapter_workspace_id)
    assert writer_call["specialist_bundle"] == bundle
    assert writer_call["runtime_identity"] == runtime_identity


def test_story_turn_domain_service_build_packet_includes_recent_raw_turn_window(
    retrieval_session,
):
    session, chapter, service = _seed_story_runtime(retrieval_session)
    first = service.create_discussion_entry(
        session_id=session.session_id,
        chapter_workspace_id=chapter.chapter_workspace_id,
        role="user",
        content_text="Keep the storm callback alive.",
    )
    second = service.create_discussion_entry(
        session_id=session.session_id,
        chapter_workspace_id=chapter.chapter_workspace_id,
        role="assistant",
        content_text="We should hold the bell tower reveal for later.",
    )
    service.create_discussion_entry(
        session_id=session.session_id,
        chapter_workspace_id=chapter.chapter_workspace_id,
        role="system",
        content_text="internal note should stay hidden",
    )
    latest = service.create_discussion_entry(
        session_id=session.session_id,
        chapter_workspace_id=chapter.chapter_workspace_id,
        role="user",
        content_text="Thread the envoy debt into the next beat.",
    )
    service.commit()
    turn_domain_service = _build_turn_domain_service(service)

    packet = turn_domain_service.build_packet(
        session_id=session.session_id,
        plan=OrchestratorPlan(
            output_kind=StoryArtifactKind.STORY_SEGMENT,
            writer_instruction="Write the next segment.",
        ),
        specialist_bundle=SpecialistResultBundle(
            foundation_digest=["Found A"],
            blueprint_digest=["Blueprint A"],
            current_outline_digest=["Outline A"],
            recent_segment_digest=["Segment A"],
            current_state_digest=["State A"],
            writer_hints=["Hint A"],
        ),
    )

    assert len(packet.recent_raw_turn_sections) == 1
    recent_raw_turns = packet.recent_raw_turn_sections[0]
    assert recent_raw_turns.label == "recent_raw_turns"
    assert recent_raw_turns.source_ref_ids == [
        first.entry_id,
        second.entry_id,
        latest.entry_id,
    ]
    assert recent_raw_turns.items == [
        "user: Keep the storm callback alive.",
        "assistant: We should hold the bell tower reveal for later.",
        "user: Thread the envoy debt into the next beat.",
    ]
    flattened_labels = [section["label"] for section in packet.context_sections]
    assert flattened_labels[:5] == [
        "foundation_digest",
        "blueprint_digest",
        "current_outline_digest",
        "recent_segment_digest",
        "current_state_digest",
    ]
    assert flattened_labels[5:7] == ["recent_raw_turns", "writer_hints"]


@pytest.mark.asyncio
async def test_writer_packet_excludes_brainstorm_runtime_workspace_scratch(
    retrieval_session,
):
    session, chapter, service = _seed_story_runtime(retrieval_session)
    service.create_discussion_entry(
        session_id=session.session_id,
        chapter_workspace_id=chapter.chapter_workspace_id,
        role="user",
        content_text="正常 writer 最近窗口应保留这句。",
    )
    service.commit()
    snapshot_service = RuntimeProfileSnapshotService(retrieval_session)
    snapshot = snapshot_service.ensure_active_snapshot(
        session_id=session.session_id,
        created_from="test.writer_packet.exclude_brainstorm_scratch",
    )
    identity_service = StoryRuntimeIdentityService(
        retrieval_session,
        runtime_profile_snapshot_service=snapshot_service,
    )
    runtime_identity = identity_service.resolve_runtime_entry_identity(
        session_id=session.session_id,
        command_kind=LongformTurnCommandKind.WRITE_NEXT_SEGMENT.value,
        actor="story_runtime",
        requested_runtime_profile_snapshot_id=snapshot.runtime_profile_snapshot_id,
    )
    brainstorm_identity = identity_service.resolve_runtime_entry_identity(
        session_id=session.session_id,
        command_kind="brainstorm",
        actor="writer",
        requested_runtime_profile_snapshot_id=snapshot.runtime_profile_snapshot_id,
    )
    brainstorm_service = StoryBrainstormService(
        story_session_service=service,
        runtime_workspace_material_service=RuntimeWorkspaceMaterialService(
            session=retrieval_session
        ),
        llm_gateway=_BrainstormTestGateway(),
    )
    started = brainstorm_service.start_session(
        BrainstormSessionStartRequest(identity=brainstorm_identity, actor="writer")
    )
    await brainstorm_service.discuss_session(
        brainstorm_id=started.brainstorm_id,
        request=BrainstormDiscussionRequest(
            identity=brainstorm_identity,
            actor="writer",
            prompt="保留钟楼债务暗线，不要现在揭晓。",
            model_id="test-model",
            provider_id="test-provider",
        ),
    )
    summarized = await brainstorm_service.summarize_session(
        brainstorm_id=started.brainstorm_id,
        request=BrainstormSummarizeRequest(
            identity=brainstorm_identity,
            actor="writer",
            dry_run_items=["保留钟楼债务伏笔", "强化使者与旧账册的关联"],
        ),
    )
    batch = summarized.batches[0]
    deleted = brainstorm_service.update_item(
        brainstorm_id=started.brainstorm_id,
        batch_id=batch.batch_id,
        item_id=batch.items[0].item_id,
        request=BrainstormItemUpdateRequest(
            identity=brainstorm_identity,
            actor="writer",
            status="deleted",
        ),
    )
    brainstorm_service.submit_batch(
        brainstorm_id=started.brainstorm_id,
        batch_id=batch.batch_id,
        request=BrainstormBatchSubmitRequest(
            identity=brainstorm_identity,
            actor="writer",
        ),
    )
    turn_domain_service = _build_turn_domain_service(
        service,
        runtime_identity_service=identity_service,
        runtime_workspace_material_service=RuntimeWorkspaceMaterialService(
            session=retrieval_session
        ),
    )
    packet = turn_domain_service.build_packet(
        session_id=session.session_id,
        plan=OrchestratorPlan(
            output_kind=StoryArtifactKind.STORY_SEGMENT,
            writer_instruction="Write the next segment.",
        ),
        specialist_bundle=SpecialistResultBundle(
            foundation_digest=["Found A"],
            blueprint_digest=["Blueprint A"],
            current_outline_digest=["Outline A"],
            recent_segment_digest=["Segment A"],
            current_state_digest=["State A"],
            writer_hints=["Hint A"],
        ),
        runtime_identity=runtime_identity,
    )

    serialized_sections = json.dumps(packet.context_sections, ensure_ascii=False)
    manifest = packet.metadata["runtime_read_manifest"]
    manifest_text = json.dumps(manifest, ensure_ascii=False)

    assert "保留钟楼债务暗线，不要现在揭晓。" not in serialized_sections
    assert "我们先把钟楼债务保留成未解释的异常细节。" not in serialized_sections
    assert "保留钟楼债务伏笔" not in serialized_sections
    assert "保留钟楼债务伏笔" not in manifest_text
    assert "正常 writer 最近窗口应保留这句。" in serialized_sections
    assert all(
        "brainstorm" not in str(ref.get("ref_id") or "")
        for ref in manifest["selected_refs"]
    )
    assert deleted.batches[0].items[0].status == BrainstormItemStatus.DELETED


def test_story_turn_domain_service_build_packet_records_runtime_workspace_writer_input_and_packet_refs(
    retrieval_session,
):
    session, chapter, service = _seed_story_runtime(retrieval_session)
    first = service.create_discussion_entry(
        session_id=session.session_id,
        chapter_workspace_id=chapter.chapter_workspace_id,
        role="user",
        content_text="Keep the storm callback alive.",
    )
    second = service.create_discussion_entry(
        session_id=session.session_id,
        chapter_workspace_id=chapter.chapter_workspace_id,
        role="assistant",
        content_text="Hold the bell tower reveal for a later beat.",
    )
    service.commit()
    snapshot_service = RuntimeProfileSnapshotService(retrieval_session)
    snapshot = snapshot_service.ensure_active_snapshot(
        session_id=session.session_id,
        created_from="test.runtime_workspace.packet_surface",
    )
    identity_service = StoryRuntimeIdentityService(
        retrieval_session,
        runtime_profile_snapshot_service=snapshot_service,
    )
    runtime_identity = identity_service.resolve_runtime_entry_identity(
        session_id=session.session_id,
        command_kind=LongformTurnCommandKind.WRITE_NEXT_SEGMENT.value,
        actor="story_runtime",
        requested_runtime_profile_snapshot_id=snapshot.runtime_profile_snapshot_id,
    )
    material_service = RuntimeWorkspaceMaterialService(session=retrieval_session)
    turn_domain_service = _build_turn_domain_service(
        service,
        runtime_identity_service=identity_service,
        runtime_workspace_material_service=material_service,
    )

    packet = turn_domain_service.build_packet(
        session_id=session.session_id,
        plan=OrchestratorPlan(
            output_kind=StoryArtifactKind.STORY_SEGMENT,
            writer_instruction="Write the next segment.",
        ),
        specialist_bundle=SpecialistResultBundle(
            foundation_digest=["Found A"],
            blueprint_digest=["Blueprint A"],
            current_outline_digest=["Outline A"],
            recent_segment_digest=["Segment A"],
            current_state_digest=["State A"],
            writer_hints=["Hint A"],
        ),
        runtime_identity=runtime_identity,
    )

    writer_inputs = material_service.list_materials(
        identity=runtime_identity,
        material_kind=RuntimeWorkspaceMaterialKind.WRITER_INPUT_REF,
    )
    packet_refs = material_service.list_materials(
        identity=runtime_identity,
        material_kind=RuntimeWorkspaceMaterialKind.PACKET_REF,
    )

    assert len(writer_inputs) == 1
    assert writer_inputs[0].payload["packet_id"] == packet.packet_id
    assert writer_inputs[0].payload["recent_turn_ref_ids"] == [
        first.entry_id,
        second.entry_id,
    ]
    assert (
        packet.metadata["runtime_workspace_writer_input_material_id"]
        == writer_inputs[0].material_id
    )
    assert len(packet_refs) == 1
    assert packet_refs[0].payload["packet_id"] == packet.packet_id
    assert (
        packet_refs[0].payload["runtime_read_manifest_id"]
        == packet.metadata["runtime_read_manifest_id"]
    )
    assert packet_refs[0].source_refs[0].entry_id == writer_inputs[0].material_id
    assert (
        packet.metadata["runtime_workspace_packet_material_id"]
        == packet_refs[0].material_id
    )


def test_story_turn_domain_service_build_packet_uses_runtime_workspace_retrieval_context_without_recording_usage(
    retrieval_session,
):
    session, _, service = _seed_story_runtime(retrieval_session)
    snapshot_service = RuntimeProfileSnapshotService(retrieval_session)
    snapshot = snapshot_service.ensure_active_snapshot(
        session_id=session.session_id,
        created_from="test.packet_retrieval_usage",
    )
    identity_service = StoryRuntimeIdentityService(
        retrieval_session,
        runtime_profile_snapshot_service=snapshot_service,
    )
    runtime_identity = identity_service.resolve_runtime_entry_identity(
        session_id=session.session_id,
        command_kind=LongformTurnCommandKind.WRITE_NEXT_SEGMENT.value,
        actor="story_runtime",
        requested_runtime_profile_snapshot_id=snapshot.runtime_profile_snapshot_id,
    )
    material_service = RuntimeWorkspaceMaterialService(session=retrieval_session)
    material_service.record_material(
        RuntimeWorkspaceMaterial(
            material_id="packet-card-R1",
            material_kind=RuntimeWorkspaceMaterialKind.RETRIEVAL_CARD,
            identity=runtime_identity,
            domain="chapter",
            domain_path="chapter.runtime.packet.card",
            short_id="R1",
            payload={
                "summary": "A recalled storm callback matters here.",
                "title": "Storm Callback",
            },
            visibility="writer_visible",
            created_by="worker.specialist",
        )
    )
    material_service.record_material(
        RuntimeWorkspaceMaterial(
            material_id="packet-expanded-X1",
            material_kind=RuntimeWorkspaceMaterialKind.RETRIEVAL_EXPANDED_CHUNK,
            identity=runtime_identity,
            domain="chapter",
            domain_path="chapter.runtime.packet.expanded",
            short_id="X1",
            payload={
                "summary": "Expanded detail: the seal broke during the first storm.",
                "text": "Expanded detail: the seal broke during the first storm.",
                "title": "Storm Callback Expanded",
            },
            lifecycle=RuntimeWorkspaceMaterialLifecycle.EXPANDED,
            visibility="writer_visible",
            created_by="worker.specialist",
        )
    )
    turn_domain_service = _build_turn_domain_service(
        service,
        runtime_identity_service=identity_service,
        runtime_workspace_material_service=material_service,
    )
    bundle = SpecialistResultBundle(
        foundation_digest=["Found A"],
        blueprint_digest=["Blueprint A"],
        current_outline_digest=["Outline A"],
        recent_segment_digest=["Segment A"],
        current_state_digest=["State A"],
        writer_hints=["Hint A"],
    )
    plan = OrchestratorPlan(
        output_kind=StoryArtifactKind.STORY_SEGMENT,
        writer_instruction="Write the next segment.",
    )

    packet = turn_domain_service.build_packet(
        session_id=session.session_id,
        plan=plan,
        specialist_bundle=bundle,
        runtime_identity=runtime_identity,
    )

    retrieval_sections = [
        section
        for section in packet.context_sections
        if section.get("label") == "retrieval_cards"
    ]
    assert len(retrieval_sections) == 1
    assert retrieval_sections[0]["items"] == [
        "R1 [retrieval_card] Storm Callback: A recalled storm callback matters here.",
        "X1 [retrieval_expanded_chunk] Storm Callback Expanded: Expanded detail: the seal broke during the first storm.",
    ]
    assert [section.label for section in packet.retrieval_card_sections] == [
        "retrieval_cards"
    ]
    bundle_metadata = packet.metadata["worker_source_ref_bundle"]
    assert bundle_metadata["retrieval_card_material_ids"] == ["packet-card-R1"]
    assert bundle_metadata["retrieval_expanded_chunk_material_ids"] == [
        "packet-expanded-X1"
    ]
    assert bundle_metadata["retrieval_usage_material_ids"] == []

    usage_records = material_service.list_materials(
        identity=runtime_identity,
        material_kind=RuntimeWorkspaceMaterialKind.RETRIEVAL_USAGE_RECORD,
    )
    assert usage_records == []

    response = turn_domain_service.persist_generated_artifact(
        request=LongformTurnRequest(
            session_id=session.session_id,
            command_kind=LongformTurnCommandKind.WRITE_NEXT_SEGMENT,
            model_id="model",
        ),
        packet=packet,
        plan=plan,
        text="Generated segment text.",
        specialist_bundle=bundle,
        pending_artifact_id=None,
    )
    artifact = service.get_artifact(response.artifact_id)

    usage_records = material_service.list_materials(
        identity=runtime_identity,
        material_kind=RuntimeWorkspaceMaterialKind.RETRIEVAL_USAGE_RECORD,
    )
    assert usage_records == []

    assert artifact is not None
    artifact_bundle = artifact.metadata["worker_source_ref_bundle"]
    assert artifact_bundle["retrieval_card_material_ids"] == ["packet-card-R1"]
    assert artifact_bundle["retrieval_expanded_chunk_material_ids"] == [
        "packet-expanded-X1"
    ]
    assert artifact_bundle["retrieval_usage_material_ids"] == []
    assert (
        artifact.metadata["runtime_read_manifest_id"]
        == packet.metadata["runtime_read_manifest_id"]
    )


def test_story_turn_domain_service_persist_generated_artifact_records_writer_output_ref(
    retrieval_session,
):
    session, _, service = _seed_story_runtime(retrieval_session)
    snapshot_service = RuntimeProfileSnapshotService(retrieval_session)
    snapshot = snapshot_service.ensure_active_snapshot(
        session_id=session.session_id,
        created_from="test.runtime_workspace.writer_output_ref",
    )
    identity_service = StoryRuntimeIdentityService(
        retrieval_session,
        runtime_profile_snapshot_service=snapshot_service,
    )
    runtime_identity = identity_service.resolve_runtime_entry_identity(
        session_id=session.session_id,
        command_kind=LongformTurnCommandKind.WRITE_NEXT_SEGMENT.value,
        actor="story_runtime",
        requested_runtime_profile_snapshot_id=snapshot.runtime_profile_snapshot_id,
    )
    material_service = RuntimeWorkspaceMaterialService(session=retrieval_session)
    turn_domain_service = _build_turn_domain_service(
        service,
        runtime_identity_service=identity_service,
        runtime_workspace_material_service=material_service,
    )
    bundle = SpecialistResultBundle(
        foundation_digest=["Found A"],
        blueprint_digest=["Blueprint A"],
        current_outline_digest=["Outline A"],
        recent_segment_digest=["Segment A"],
        current_state_digest=["State A"],
        writer_hints=["Hint A"],
    )
    plan = OrchestratorPlan(
        output_kind=StoryArtifactKind.STORY_SEGMENT,
        writer_instruction="Write the next segment.",
    )
    packet = turn_domain_service.build_packet(
        session_id=session.session_id,
        plan=plan,
        specialist_bundle=bundle,
        command_kind=LongformTurnCommandKind.WRITE_NEXT_SEGMENT,
        runtime_identity=runtime_identity,
    )

    response = turn_domain_service.persist_generated_artifact(
        request=LongformTurnRequest(
            session_id=session.session_id,
            command_kind=LongformTurnCommandKind.WRITE_NEXT_SEGMENT,
            model_id="model",
        ),
        packet=packet,
        plan=plan,
        writing_result=WritingWorkerExecutionResult(
            request_id="writer-exec-1",
            packet_id=packet.packet_id,
            turn_id=runtime_identity.turn_id,
            operation_mode="writing",
            output_text="Generated segment text.",
            output_kind=plan.output_kind.value,
            usage_metadata={
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
        ),
        specialist_bundle=bundle,
        pending_artifact_id=None,
    )

    writer_outputs = material_service.list_materials(
        identity=runtime_identity,
        material_kind=RuntimeWorkspaceMaterialKind.WRITER_OUTPUT_REF,
    )

    assert len(writer_outputs) == 1
    assert writer_outputs[0].payload["artifact_id"] == response.artifact_id
    assert writer_outputs[0].payload["packet_id"] == packet.packet_id
    assert writer_outputs[0].payload["operation_mode"] == "writing"
    assert writer_outputs[0].payload["artifact_kind"] == "story_segment"
    assert (
        writer_outputs[0].source_refs[0].entry_id
        == packet.metadata["runtime_workspace_packet_material_id"]
    )
    usage_materials = material_service.list_materials(
        identity=runtime_identity,
        material_kind=RuntimeWorkspaceMaterialKind.TOKEN_USAGE_METADATA,
    )
    assert len(usage_materials) == 1
    assert usage_materials[0].payload["usage_metadata"]["total_tokens"] == 30
    assert response.writing_result is not None
    assert response.writing_result.visible_output_ref == response.artifact_id
    assert (
        response.writing_result.writer_output_material_id
        == writer_outputs[0].material_id
    )
    assert (
        response.writing_result.token_usage_material_id
        == usage_materials[0].material_id
    )


def test_story_turn_domain_service_persist_generated_artifact_registers_obligations(
    retrieval_session,
):
    session, _, service = _seed_story_runtime(retrieval_session)
    snapshot_service = RuntimeProfileSnapshotService(retrieval_session)
    snapshot = snapshot_service.ensure_active_snapshot(
        session_id=session.session_id,
        created_from="test.post_write.creation_time_obligations",
    )
    identity_service = StoryRuntimeIdentityService(
        retrieval_session,
        runtime_profile_snapshot_service=snapshot_service,
    )
    runtime_identity = identity_service.resolve_runtime_entry_identity(
        session_id=session.session_id,
        command_kind=LongformTurnCommandKind.WRITE_NEXT_SEGMENT.value,
        actor="story_runtime",
        requested_runtime_profile_snapshot_id=snapshot.runtime_profile_snapshot_id,
    )
    material_service = RuntimeWorkspaceMaterialService(session=retrieval_session)
    workflow_job_service = RuntimeWorkflowJobService(retrieval_session)
    turn_domain_service = _build_turn_domain_service(
        service,
        runtime_identity_service=identity_service,
        runtime_workspace_material_service=material_service,
    )
    bundle = SpecialistResultBundle(
        foundation_digest=["Found A"],
        blueprint_digest=["Blueprint A"],
        current_outline_digest=["Outline A"],
        recent_segment_digest=["Segment A"],
        current_state_digest=["State A"],
        writer_hints=["Hint A"],
    )
    plan = OrchestratorPlan(
        output_kind=StoryArtifactKind.STORY_SEGMENT,
        writer_instruction="Write the next segment.",
    )
    packet = turn_domain_service.build_packet(
        session_id=session.session_id,
        plan=plan,
        specialist_bundle=bundle,
        command_kind=LongformTurnCommandKind.WRITE_NEXT_SEGMENT,
        runtime_identity=runtime_identity,
    )

    response = turn_domain_service.persist_generated_artifact(
        request=LongformTurnRequest(
            session_id=session.session_id,
            command_kind=LongformTurnCommandKind.WRITE_NEXT_SEGMENT,
            model_id="model",
        ),
        packet=packet,
        plan=plan,
        writing_result=WritingWorkerExecutionResult(
            request_id="writer-exec-f1",
            packet_id=packet.packet_id,
            turn_id=runtime_identity.turn_id,
            operation_mode="writing",
            output_text="Generated segment text.",
            output_kind=plan.output_kind.value,
            usage_metadata={"total_tokens": 30},
        ),
        specialist_bundle=bundle,
        pending_artifact_id=None,
    )
    first_jobs = workflow_job_service.list_jobs_for_turn(
        turn_id=runtime_identity.turn_id
    )
    second_jobs = workflow_job_service.ensure_creation_time_obligations(
        identity=runtime_identity
    )
    turn = retrieval_session.get(StoryTurnRecord, runtime_identity.turn_id)

    assert turn is not None
    assert turn.status == "post_write_pending"
    assert turn.visible_output_ref == response.artifact_id
    assert turn.selected_output_ref == response.artifact_id
    assert turn.writer_completed_at is not None
    assert sorted(job.job_kind for job in first_jobs) == [
        "required_post_write_analysis",
        "runtime_workspace_finalize",
    ]
    assert {job.creation_mode for job in first_jobs} == {"creation_time_obligation"}
    assert {job.required_for_turn_completion for job in first_jobs} == {True}
    assert {job.status for job in first_jobs} == {"pending"}
    assert {job.metadata_json["obligation_owner"] for job in first_jobs} == {
        "story_turn_domain.persist_generated_artifact"
    }
    assert {job.job_id for job in second_jobs} == {job.job_id for job in first_jobs}


@pytest.mark.asyncio
async def test_story_turn_domain_service_trigger_post_write_defers_pending_obligations(
    retrieval_session,
):
    session, _, service = _seed_story_runtime(retrieval_session)
    snapshot_service = RuntimeProfileSnapshotService(retrieval_session)
    snapshot = snapshot_service.ensure_active_snapshot(
        session_id=session.session_id,
        created_from="test.post_write.trigger_minimal",
    )
    identity_service = StoryRuntimeIdentityService(
        retrieval_session,
        runtime_profile_snapshot_service=snapshot_service,
    )
    runtime_identity = identity_service.resolve_runtime_entry_identity(
        session_id=session.session_id,
        command_kind=LongformTurnCommandKind.WRITE_NEXT_SEGMENT.value,
        actor="story_runtime",
        requested_runtime_profile_snapshot_id=snapshot.runtime_profile_snapshot_id,
    )
    material_service = RuntimeWorkspaceMaterialService(session=retrieval_session)
    workflow_job_service = RuntimeWorkflowJobService(retrieval_session)
    turn_domain_service = _build_turn_domain_service(
        service,
        runtime_identity_service=identity_service,
        runtime_workspace_material_service=material_service,
    )
    bundle = SpecialistResultBundle(
        foundation_digest=["Found A"],
        blueprint_digest=["Blueprint A"],
        current_outline_digest=["Outline A"],
        recent_segment_digest=["Segment A"],
        current_state_digest=["State A"],
        writer_hints=["Hint A"],
    )
    plan = OrchestratorPlan(
        output_kind=StoryArtifactKind.STORY_SEGMENT,
        writer_instruction="Write the next segment.",
    )
    packet = turn_domain_service.build_packet(
        session_id=session.session_id,
        plan=plan,
        specialist_bundle=bundle,
        command_kind=LongformTurnCommandKind.WRITE_NEXT_SEGMENT,
        runtime_identity=runtime_identity,
    )
    turn_domain_service.persist_generated_artifact(
        request=LongformTurnRequest(
            session_id=session.session_id,
            command_kind=LongformTurnCommandKind.WRITE_NEXT_SEGMENT,
            model_id="model",
        ),
        packet=packet,
        plan=plan,
        writing_result=WritingWorkerExecutionResult(
            request_id="writer-exec-f1-trigger",
            packet_id=packet.packet_id,
            turn_id=runtime_identity.turn_id,
            operation_mode="writing",
            output_text="Generated segment text.",
            output_kind=plan.output_kind.value,
        ),
        specialist_bundle=bundle,
        pending_artifact_id=None,
    )

    trigger_result = await turn_domain_service.trigger_post_write(
        runtime_identity=runtime_identity
    )
    trigger_result_again = await turn_domain_service.trigger_post_write(
        runtime_identity=runtime_identity
    )
    turn = retrieval_session.get(StoryTurnRecord, runtime_identity.turn_id)
    jobs = workflow_job_service.list_jobs_for_turn(turn_id=runtime_identity.turn_id)

    assert trigger_result["run_kind"] == "minimal_only"
    assert trigger_result["settled"] is True
    assert trigger_result["settlement_reason"] == "required_jobs_deferred_by_policy"
    assert turn is not None
    assert turn.status == "settled"
    assert turn.settlement_reason == "required_jobs_deferred_by_policy"
    assert turn.settled_at is not None
    assert sorted(job.job_kind for job in jobs) == [
        "required_post_write_analysis",
        "runtime_workspace_finalize",
    ]
    assert {job.status for job in jobs} == {"deferred"}
    assert {job.completion_reason for job in jobs} == {
        "post_write_full_schedule_services_missing"
    }
    trigger_metadata = cast(dict[str, Any], trigger_result_again["metadata_json"])
    assert set(trigger_metadata["job_ids"]) == {job.job_id for job in jobs}


@pytest.mark.asyncio
async def test_story_turn_domain_service_trigger_post_write_waits_for_finalize(
    retrieval_session,
):
    session, _, service = _seed_story_runtime(retrieval_session)
    snapshot_service = RuntimeProfileSnapshotService(retrieval_session)
    snapshot = snapshot_service.ensure_active_snapshot(
        session_id=session.session_id,
        created_from="test.post_write.pre_finalize_guard",
    )
    identity_service = StoryRuntimeIdentityService(
        retrieval_session,
        runtime_profile_snapshot_service=snapshot_service,
    )
    runtime_identity = identity_service.resolve_runtime_entry_identity(
        session_id=session.session_id,
        command_kind=LongformTurnCommandKind.WRITE_NEXT_SEGMENT.value,
        actor="story_runtime",
        requested_runtime_profile_snapshot_id=snapshot.runtime_profile_snapshot_id,
    )
    workflow_job_service = RuntimeWorkflowJobService(retrieval_session)
    turn_domain_service = _build_turn_domain_service(
        service,
        runtime_identity_service=identity_service,
        runtime_workflow_job_service=workflow_job_service,
    )

    trigger_result = await turn_domain_service.trigger_post_write(
        runtime_identity=runtime_identity
    )
    turn = retrieval_session.get(StoryTurnRecord, runtime_identity.turn_id)
    jobs = workflow_job_service.list_jobs_for_turn(turn_id=runtime_identity.turn_id)

    assert trigger_result == {
        "run_kind": "skipped",
        "reason": "writer_output_not_finalized",
        "turn_status": "started",
    }
    assert turn is not None
    assert turn.status == "started"
    assert jobs == []


def test_runtime_workflow_settlement_blocks_cancelled_required_job(
    retrieval_session,
):
    session, _, _ = _seed_story_runtime(retrieval_session)
    snapshot_service = RuntimeProfileSnapshotService(retrieval_session)
    snapshot = snapshot_service.ensure_active_snapshot(
        session_id=session.session_id,
        created_from="test.post_write.cancelled_blocks_settlement",
    )
    identity_service = StoryRuntimeIdentityService(
        retrieval_session,
        runtime_profile_snapshot_service=snapshot_service,
    )
    runtime_identity = identity_service.resolve_runtime_entry_identity(
        session_id=session.session_id,
        command_kind=LongformTurnCommandKind.WRITE_NEXT_SEGMENT.value,
        actor="story_runtime",
        requested_runtime_profile_snapshot_id=snapshot.runtime_profile_snapshot_id,
    )
    workflow_job_service = RuntimeWorkflowJobService(retrieval_session)
    jobs = workflow_job_service.ensure_creation_time_obligations(
        identity=runtime_identity
    )
    jobs[0].status = RuntimeWorkflowJobStatus.CANCELLED.value
    jobs[1].status = RuntimeWorkflowJobStatus.COMPLETED.value
    retrieval_session.add(jobs[0])
    retrieval_session.add(jobs[1])
    retrieval_session.commit()

    settlement = workflow_job_service.evaluate_turn_settlement(
        turn_id=runtime_identity.turn_id
    )

    assert settlement.eligible is False
    assert settlement.blocking_job_ids == (jobs[0].job_id,)


@pytest.mark.asyncio
async def test_story_turn_domain_service_full_post_write_governs_worker_results(
    retrieval_session,
):
    session, _, service = _seed_story_runtime(retrieval_session)
    snapshot_service = RuntimeProfileSnapshotService(retrieval_session)
    snapshot = snapshot_service.ensure_active_snapshot(
        session_id=session.session_id,
        created_from="test.post_write.full_schedule",
    )
    identity_service = StoryRuntimeIdentityService(
        retrieval_session,
        runtime_profile_snapshot_service=snapshot_service,
    )
    runtime_identity = identity_service.resolve_runtime_entry_identity(
        session_id=session.session_id,
        command_kind=LongformTurnCommandKind.WRITE_NEXT_SEGMENT.value,
        actor="story_runtime",
        requested_runtime_profile_snapshot_id=snapshot.runtime_profile_snapshot_id,
    )
    material_service = RuntimeWorkspaceMaterialService(session=retrieval_session)
    workflow_job_service = RuntimeWorkflowJobService(retrieval_session)
    worker_plan = WorkerExecutionPlan(
        plan_id="worker-plan-f2-post-write",
        identity=runtime_identity,
        plan_source=WorkerPlanSource.POST_WRITE_REQUIRED,
        phase=POST_WRITE_MAINTENANCE_PHASE,
        selected_workers=[
            WorkerExecutionItem(
                worker_id="LongformMemoryWorker",
                must_run=True,
                allow_degrade=False,
                blocking=True,
                async_allowed=False,
                reason_codes=["post_write_full_schedule_required"],
            )
        ],
    )
    worker_result = WorkerResult(
        worker_id="LongformMemoryWorker",
        phase=POST_WRITE_MAINTENANCE_PHASE,
        result_status=WorkerResultStatus.COMPLETED,
        projection_refresh_requests=[{"reason": "retrieval_used"}],
        proposal_candidates=[
            {
                "candidate_kind": "legacy_state_patch",
                "payload": {
                    "chapter_digest": {
                        "current_chapter": 1,
                        "title": "Chapter One Revised",
                    }
                },
            }
        ],
        recall_candidates=[
            {"candidate_kind": "legacy_recall_summary", "text": "Recall candidate"}
        ],
        evidence_refs=["retrieval-usage-f2"],
        metadata={"context_packet_ref": "worker-context-f2"},
    )
    specialist_bundle = SpecialistResultBundle(
        foundation_digest=["Found F2"],
        blueprint_digest=["Blueprint F2"],
        current_outline_digest=["Outline F2"],
        recent_segment_digest=["Segment F2"],
        current_state_digest=["State F2"],
        writer_hints=["Hint F2"],
    )
    worker_execution_service = _RecordingWorkerExecutionService(
        WorkerExecutionOutcome(
            plan=worker_plan,
            worker_results=[worker_result],
            specialist_bundle=specialist_bundle,
        )
    )
    worker_scheduler_service = _RecordingWorkerSchedulerService(worker_plan)
    proposal_workflow_service = _RecordingProposalWorkflowService()
    post_write_scheduler_service = PostWriteSchedulerService(
        worker_scheduler_service=worker_scheduler_service,
        runtime_workspace_material_service=material_service,
    )
    post_write_governance_service = PostWriteGovernanceService(
        runtime_workflow_job_service=workflow_job_service,
        projection_refresh_service=ProjectionRefreshService(service),
        proposal_workflow_service=proposal_workflow_service,
    )
    turn_domain_service = _build_turn_domain_service(
        service,
        runtime_identity_service=identity_service,
        runtime_workspace_material_service=material_service,
        runtime_workflow_job_service=workflow_job_service,
        worker_execution_service=worker_execution_service,
        post_write_scheduler_service=post_write_scheduler_service,
        post_write_governance_service=post_write_governance_service,
    )
    bundle = SpecialistResultBundle(
        foundation_digest=["Found A"],
        blueprint_digest=["Blueprint A"],
        current_outline_digest=["Outline A"],
        recent_segment_digest=["Segment A"],
        current_state_digest=["State A"],
        writer_hints=["Hint A"],
    )
    plan = OrchestratorPlan(
        output_kind=StoryArtifactKind.STORY_SEGMENT,
        writer_instruction="Write the next segment.",
    )
    packet = turn_domain_service.build_packet(
        session_id=session.session_id,
        plan=plan,
        specialist_bundle=bundle,
        command_kind=LongformTurnCommandKind.WRITE_NEXT_SEGMENT,
        runtime_identity=runtime_identity,
    )
    turn_domain_service.persist_generated_artifact(
        request=LongformTurnRequest(
            session_id=session.session_id,
            command_kind=LongformTurnCommandKind.WRITE_NEXT_SEGMENT,
            model_id="model",
        ),
        packet=packet,
        plan=plan,
        writing_result=WritingWorkerExecutionResult(
            request_id="writer-exec-f2-full",
            packet_id=packet.packet_id,
            turn_id=runtime_identity.turn_id,
            operation_mode="writing",
            output_text="Generated segment with retrieved evidence.",
            output_kind=plan.output_kind.value,
        ),
        specialist_bundle=bundle,
        pending_artifact_id=None,
    )
    material_service.record_material(
        RuntimeWorkspaceMaterial(
            material_id="retrieval-usage-f2",
            material_kind=RuntimeWorkspaceMaterialKind.RETRIEVAL_USAGE_RECORD,
            identity=runtime_identity,
            domain="chapter",
            domain_path="chapter.runtime.retrieval.usage",
            payload={
                "used_card_material_ids": ["retrieval-card-f2"],
                "knowledge_gaps": [],
            },
            visibility="runtime_private",
            created_by="test",
        )
    )

    trigger_result = await turn_domain_service.trigger_post_write(
        runtime_identity=runtime_identity,
        model_id="model",
        orchestrator_plan=plan,
    )
    turn = retrieval_session.get(StoryTurnRecord, runtime_identity.turn_id)
    jobs = workflow_job_service.list_jobs_for_turn(turn_id=runtime_identity.turn_id)
    jobs_by_kind = {job.job_kind: job for job in jobs}

    assert trigger_result["run_kind"] == "full_schedule"
    assert trigger_result["settled"] is True
    assert trigger_result["settlement_reason"] == "all_required_jobs_completed"
    assert trigger_result["projection_refresh_job_refs"]
    assert trigger_result["proposal_job_refs"]
    assert trigger_result["materialization_job_refs"]
    assert turn is not None
    assert turn.status == "settled"
    assert turn.settlement_reason == "all_required_jobs_completed"
    assert jobs_by_kind["required_post_write_analysis"].status == "completed"
    assert jobs_by_kind["runtime_workspace_finalize"].status == "completed"
    assert jobs_by_kind["projection_refresh"].status == "completed"
    assert jobs_by_kind["proposal_submit"].status == "completed"
    assert jobs_by_kind["recall_materialization"].status == "deferred"
    assert jobs_by_kind["recall_materialization"].required_for_turn_completion is False
    assert jobs_by_kind["projection_refresh"].creation_mode == "derived"
    assert jobs_by_kind["proposal_submit"].creation_mode == "derived"
    assert jobs_by_kind["projection_refresh"].metadata_json["refresh_reason"] == (
        "retrieval_used"
    )
    job_order = [job.job_kind for job in jobs]
    assert job_order.index("projection_refresh") < job_order.index("proposal_submit")
    assert proposal_workflow_service.calls
    governance_metadata = cast(
        Any,
        proposal_workflow_service.calls[0]["governance_metadata"],
    )
    assert governance_metadata.worker_id == "LongformMemoryWorker"
    assert governance_metadata.phase == POST_WRITE_MAINTENANCE_PHASE
    assert worker_scheduler_service.calls == [
        {"identity": runtime_identity, "phase": POST_WRITE_MAINTENANCE_PHASE}
    ]
    assert worker_execution_service.calls


@pytest.mark.asyncio
async def test_story_turn_domain_service_rewrite_preserves_prior_draft_until_accept(
    retrieval_session,
):
    session, _, service = _seed_story_runtime(retrieval_session)
    snapshot_service = RuntimeProfileSnapshotService(retrieval_session)
    snapshot = snapshot_service.ensure_active_snapshot(
        session_id=session.session_id,
        created_from="test.longform.rewrite_candidates",
    )
    identity_service = StoryRuntimeIdentityService(
        retrieval_session,
        runtime_profile_snapshot_service=snapshot_service,
    )
    material_service = RuntimeWorkspaceMaterialService(session=retrieval_session)
    turn_domain_service = _build_turn_domain_service(
        service,
        runtime_identity_service=identity_service,
        runtime_workspace_material_service=material_service,
    )
    bundle = SpecialistResultBundle(
        foundation_digest=["Found A"],
        blueprint_digest=["Blueprint A"],
        current_outline_digest=["Outline A"],
        recent_segment_digest=["Segment A"],
        current_state_digest=["State A"],
        writer_hints=["Hint A"],
    )
    plan = OrchestratorPlan(
        output_kind=StoryArtifactKind.STORY_SEGMENT,
        writer_instruction="Write the next segment.",
    )

    first_identity = identity_service.resolve_runtime_entry_identity(
        session_id=session.session_id,
        command_kind=LongformTurnCommandKind.WRITE_NEXT_SEGMENT.value,
        actor="story_runtime",
        requested_runtime_profile_snapshot_id=snapshot.runtime_profile_snapshot_id,
    )
    first_packet = turn_domain_service.build_packet(
        session_id=session.session_id,
        plan=plan,
        specialist_bundle=bundle,
        command_kind=LongformTurnCommandKind.WRITE_NEXT_SEGMENT,
        runtime_identity=first_identity,
    )
    first_response = turn_domain_service.persist_generated_artifact(
        request=LongformTurnRequest(
            session_id=session.session_id,
            command_kind=LongformTurnCommandKind.WRITE_NEXT_SEGMENT,
            model_id="model",
        ),
        packet=first_packet,
        plan=plan,
        writing_result=WritingWorkerExecutionResult(
            request_id="writer-exec-first",
            packet_id=first_packet.packet_id,
            turn_id=first_identity.turn_id,
            operation_mode="writing",
            output_text="First draft.",
            output_kind=plan.output_kind.value,
        ),
        specialist_bundle=bundle,
        pending_artifact_id=None,
    )

    rewrite_identity = identity_service.resolve_runtime_entry_identity(
        session_id=session.session_id,
        command_kind=LongformTurnCommandKind.REWRITE_PENDING_SEGMENT.value,
        actor="story_runtime",
        requested_runtime_profile_snapshot_id=snapshot.runtime_profile_snapshot_id,
    )
    rewrite_packet = turn_domain_service.build_packet(
        session_id=session.session_id,
        plan=plan,
        specialist_bundle=bundle,
        command_kind=LongformTurnCommandKind.REWRITE_PENDING_SEGMENT,
        runtime_identity=rewrite_identity,
    )
    rewrite_response = turn_domain_service.persist_generated_artifact(
        request=LongformTurnRequest(
            session_id=session.session_id,
            command_kind=LongformTurnCommandKind.REWRITE_PENDING_SEGMENT,
            model_id="model",
            target_artifact_id=first_response.artifact_id,
        ),
        packet=rewrite_packet,
        plan=plan,
        writing_result=WritingWorkerExecutionResult(
            request_id="writer-exec-rewrite",
            packet_id=rewrite_packet.packet_id,
            turn_id=rewrite_identity.turn_id,
            operation_mode="rewrite",
            output_text="Second draft.",
            output_kind=plan.output_kind.value,
        ),
        specialist_bundle=bundle,
        pending_artifact_id=first_response.artifact_id,
    )

    story_segments = [
        item
        for item in service.list_artifacts(
            chapter_workspace_id=rewrite_response.chapter_workspace_id
        )
        if item.artifact_kind == StoryArtifactKind.STORY_SEGMENT
    ]
    assert [
        (item.artifact_id, item.status, item.revision) for item in story_segments
    ] == [
        (first_response.artifact_id, StoryArtifactStatus.DRAFT, 1),
        (rewrite_response.artifact_id, StoryArtifactStatus.DRAFT, 2),
    ]

    accepted = await turn_domain_service.accept_pending_segment(
        request=LongformTurnRequest(
            session_id=session.session_id,
            command_kind=LongformTurnCommandKind.ACCEPT_PENDING_SEGMENT,
            model_id="model",
            target_artifact_id=first_response.artifact_id,
        ),
        runtime_identity=rewrite_identity,
    )
    accepted_segments = [
        item
        for item in service.list_artifacts(
            chapter_workspace_id=accepted.chapter_workspace_id
        )
        if item.artifact_kind == StoryArtifactKind.STORY_SEGMENT
    ]
    assert [(item.artifact_id, item.status) for item in accepted_segments] == [
        (first_response.artifact_id, StoryArtifactStatus.ACCEPTED),
        (rewrite_response.artifact_id, StoryArtifactStatus.SUPERSEDED),
    ]


def test_orchestrator_fallback_uses_projection_slots_not_writer_hints(
    retrieval_session,
):
    session, chapter, service = _seed_story_runtime(retrieval_session)
    authoritative_state_view_service, projection_state_service = (
        _build_boundary_services(service)
    )
    orchestrator_service = LongformOrchestratorService(
        authoritative_state_view_service=authoritative_state_view_service,
        projection_state_service=projection_state_service,
    )
    plan = orchestrator_service._fallback_plan(
        session=session,
        chapter=chapter,
        command_kind=LongformTurnCommandKind.WRITE_NEXT_SEGMENT,
        user_prompt=None,
        projection_snapshot=projection_state_service.build_planner_projection(
            session_id=session.session_id
        ),
    )

    assert "Persisted Hint" not in " ".join(plan.archival_queries)
    assert any("Outline A" in query for query in plan.archival_queries)


def test_orchestrator_accept_fallback_does_not_request_retrieval(
    retrieval_session,
):
    session, chapter, service = _seed_story_runtime(retrieval_session)
    authoritative_state_view_service, projection_state_service = (
        _build_boundary_services(service)
    )
    orchestrator_service = LongformOrchestratorService(
        authoritative_state_view_service=authoritative_state_view_service,
        projection_state_service=projection_state_service,
    )

    plan = orchestrator_service._fallback_plan(
        session=session,
        chapter=chapter,
        command_kind=LongformTurnCommandKind.ACCEPT_PENDING_SEGMENT,
        user_prompt=None,
        projection_snapshot=projection_state_service.build_planner_projection(
            session_id=session.session_id
        ),
    )

    assert plan.needs_retrieval is False
    assert plan.archival_queries == []
    assert plan.recall_queries == []


@pytest.mark.asyncio
async def test_specialist_skips_memory_os_when_plan_does_not_need_retrieval(
    retrieval_session,
):
    session, chapter, service = _seed_story_runtime(retrieval_session)
    authoritative_state_view_service, projection_state_service = (
        _build_boundary_services(service)
    )
    specialist_service = LongformSpecialistService(
        authoritative_state_view_service=authoritative_state_view_service,
        projection_state_service=projection_state_service,
        memory_os_factory=lambda _story_id: _FailingMemoryOsService(),
    )

    bundle = await specialist_service.analyze(
        session=session,
        chapter=chapter,
        plan=OrchestratorPlan(
            output_kind=StoryArtifactKind.STORY_SEGMENT,
            needs_retrieval=False,
            archival_queries=["should not run"],
            recall_queries=["should not run"],
            writer_instruction="Accept the pending segment.",
        ),
        command_kind=LongformTurnCommandKind.ACCEPT_PENDING_SEGMENT,
        model_id="model",
        provider_id=None,
        user_prompt=None,
        accepted_segments=[],
        pending_artifact=None,
    )

    assert bundle.state_patch_proposals["narrative_progress"]["last_command"] == (
        LongformTurnCommandKind.ACCEPT_PENDING_SEGMENT.value
    )


def test_story_segment_structured_metadata_normalizes_latest_foreshadow_update():
    metadata = StorySegmentStructuredMetadata(
        foreshadow_status_updates=[
            {
                "foreshadow_id": "envoy_debt",
                "status": "active",
                "summary": "  bell tower debt  ",
            },
            {
                "foreshadow_id": "   ",
                "status": "closed",
            },
            {
                "foreshadow_id": "envoy_debt",
                "status": "resolved",
                "summary": "bell tower debt",
                "resolution": " Settled at the river gate. ",
            },
        ]
    )

    assert metadata.to_artifact_metadata() == {
        "foreshadow_status_updates": [
            {
                "foreshadow_id": "envoy_debt",
                "status": "resolved",
                "summary": "bell tower debt",
                "resolution": "Settled at the river gate.",
            }
        ]
    }


def test_story_segment_structured_metadata_schema_stays_typed_and_degrades_bad_items():
    schema = SpecialistResultBundle.model_json_schema()
    assert (
        schema["$defs"]["StorySegmentStructuredMetadata"]["properties"][
            "foreshadow_status_updates"
        ]["items"]["$ref"]
        == "#/$defs/ForeshadowStatusUpdateMetadata"
    )

    bundle = SpecialistResultBundle.model_validate(
        {
            "story_segment_metadata": {
                "foreshadow_status_updates": [
                    {
                        "foreshadow_id": "envoy_debt",
                        "status": "resolved",
                        "summary": " bell tower debt ",
                    },
                    "not-a-dict",
                    {
                        "foreshadow_id": "   ",
                        "status": "closed",
                    },
                ],
                "unsupported_family": [{"ignored": True}],
            }
        }
    )

    assert bundle.story_segment_metadata.to_artifact_metadata() == {
        "foreshadow_status_updates": [
            {
                "foreshadow_id": "envoy_debt",
                "status": "resolved",
                "summary": "bell tower debt",
            }
        ]
    }


def test_story_turn_accept_outline_updates_projection_via_boundary_service(
    retrieval_session,
):
    session, chapter, service = _seed_story_runtime(retrieval_session)
    turn_domain_service = _build_turn_domain_service(service)
    outline = service.create_artifact(
        session_id=session.session_id,
        chapter_workspace_id=chapter.chapter_workspace_id,
        artifact_kind=StoryArtifactKind.CHAPTER_OUTLINE,
        status=StoryArtifactStatus.DRAFT,
        content_text="Accepted Outline Text",
    )

    response = turn_domain_service.accept_outline(
        request=LongformTurnRequest(
            session_id=session.session_id,
            command_kind=LongformTurnCommandKind.ACCEPT_OUTLINE,
            model_id="model",
            target_artifact_id=outline.artifact_id,
        )
    )
    updated_chapter = service.get_current_chapter(session.session_id)

    assert updated_chapter is not None
    assert updated_chapter.builder_snapshot_json["current_outline_digest"] == [
        "Accepted Outline Text"
    ]
    assert response.current_phase == LongformChapterPhase.SEGMENT_DRAFTING


class _AsyncOrchestratorService:
    async def plan(self, **kwargs):
        return OrchestratorPlan(
            output_kind=StoryArtifactKind.STORY_SEGMENT,
            writer_instruction=kwargs.get("user_prompt") or "Write the next segment.",
        )


class _AsyncSpecialistService:
    async def analyze(self, **_kwargs):
        return SpecialistResultBundle(
            foundation_digest=["Found A"],
            blueprint_digest=["Blueprint A"],
            current_outline_digest=["Outline A"],
            recent_segment_digest=["Segment A"],
            current_state_digest=["State A"],
            writer_hints=["Hint A"],
        )


class _FailingAsyncSpecialistService:
    async def analyze(self, **_kwargs):
        raise AssertionError("legacy specialist path should not run")


class _RecordingWorkerSchedulerService:
    def __init__(self, plan: WorkerExecutionPlan) -> None:
        self.plan = plan
        self.calls: list[dict[str, object]] = []

    def build_plan(self, *, identity: MemoryRuntimeIdentity, phase: str):
        self.calls.append({"identity": identity, "phase": phase})
        return self.plan


class _RecordingWorkerExecutionService:
    def __init__(self, outcome: WorkerExecutionOutcome) -> None:
        self.outcome = outcome
        self.calls: list[dict[str, object]] = []

    async def execute_plan(self, **kwargs):
        self.calls.append(kwargs)
        return self.outcome


class _RecordingProposalWorkflowService:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def submit_and_route(self, input_model, **kwargs):
        self.calls.append({"input_model": input_model, **kwargs})
        return SimpleNamespace(proposal_id=f"proposal-{len(self.calls)}")


class _RecordingContextOrchestrationService:
    def __init__(
        self,
        *,
        writing_packet=None,
        worker_context_packet=None,
    ) -> None:
        self.writing_packet = writing_packet
        self.worker_context_packet = worker_context_packet
        self.writer_calls: list[dict[str, Any]] = []
        self.worker_calls: list[dict[str, Any]] = []

    def build_writing_packet(self, **kwargs):
        self.writer_calls.append(kwargs)
        return self.writing_packet

    def build_worker_context_packet(self, **kwargs):
        self.worker_calls.append(kwargs)
        return self.worker_context_packet


@pytest.mark.asyncio
async def test_story_turn_domain_service_marks_consumers_synced(retrieval_session):
    session, _, service = _seed_story_runtime(retrieval_session)
    recorder = _RecordingBlockConsumerStateService()
    turn_domain_service = _build_turn_domain_service(
        service,
        orchestrator_service=_AsyncOrchestratorService(),
        specialist_service=_AsyncSpecialistService(),
        block_consumer_state_service=recorder,
    )

    plan = await turn_domain_service.orchestrator_plan(
        session_id=session.session_id,
        command_kind=LongformTurnCommandKind.WRITE_NEXT_SEGMENT,
        model_id="model",
        provider_id=None,
        user_prompt="Continue.",
        target_artifact_id=None,
    )
    bundle = await turn_domain_service.specialist_analyze(
        session_id=session.session_id,
        command_kind=LongformTurnCommandKind.WRITE_NEXT_SEGMENT,
        model_id="model",
        provider_id=None,
        user_prompt="Continue.",
        plan=plan,
        pending_artifact_id=None,
        accepted_segment_ids=[],
    )
    _ = turn_domain_service.build_packet(
        session_id=session.session_id,
        plan=plan,
        specialist_bundle=bundle,
    )

    assert recorder.calls == [
        (session.session_id, "story.orchestrator"),
        (session.session_id, "story.specialist"),
        (session.session_id, "story.writer_packet"),
    ]


@pytest.mark.asyncio
async def test_story_turn_domain_service_routes_specialist_analyze_through_worker_services_when_configured(
    retrieval_session,
):
    session, _, service = _seed_story_runtime(retrieval_session)
    runtime_identity = MemoryRuntimeIdentity(
        story_id=session.story_id,
        session_id=session.session_id,
        branch_head_id="branch:test:main",
        turn_id="turn:test:1",
        runtime_profile_snapshot_id="snapshot:test:1",
    )
    execution_plan = WorkerExecutionPlan(
        plan_id="worker-plan-test",
        identity=runtime_identity,
        plan_source=WorkerPlanSource.DETERMINISTIC_FALLBACK,
        phase=PRE_WRITE_CONTEXT_PHASE,
    )
    returned_bundle = SpecialistResultBundle(
        foundation_digest=["Worker Found"],
        blueprint_digest=["Worker Blueprint"],
        current_outline_digest=["Worker Outline"],
        recent_segment_digest=["Worker Segment"],
        current_state_digest=["Worker State"],
        writer_hints=["Worker Hint"],
    )
    scheduler = _RecordingWorkerSchedulerService(execution_plan)
    execution = _RecordingWorkerExecutionService(
        WorkerExecutionOutcome(
            plan=execution_plan,
            worker_results=[],
            specialist_bundle=returned_bundle,
        )
    )
    turn_domain_service = _build_turn_domain_service(
        service,
        specialist_service=_FailingAsyncSpecialistService(),
        worker_scheduler_service=scheduler,
        worker_execution_service=execution,
    )
    orchestrator_plan = OrchestratorPlan(
        output_kind=StoryArtifactKind.STORY_SEGMENT,
        writer_instruction="Write the next segment.",
    )

    bundle = await turn_domain_service.specialist_analyze(
        session_id=session.session_id,
        command_kind=LongformTurnCommandKind.WRITE_NEXT_SEGMENT,
        model_id="model",
        provider_id=None,
        user_prompt="Continue.",
        plan=orchestrator_plan,
        pending_artifact_id=None,
        accepted_segment_ids=[],
        runtime_identity=runtime_identity,
    )

    assert bundle.writer_hints == ["Worker Hint"]
    assert scheduler.calls == [
        {
            "identity": runtime_identity,
            "phase": PRE_WRITE_CONTEXT_PHASE,
        }
    ]
    assert len(execution.calls) == 1
    assert execution.calls[0]["plan"] == execution_plan
    assert execution.calls[0]["orchestrator_plan"] == orchestrator_plan
    assert execution.calls[0]["accepted_segments"] == []


@pytest.mark.asyncio
async def test_story_graph_runner_pins_runtime_identity_before_special_command(
    retrieval_session,
):
    session, chapter, service = _seed_story_runtime(retrieval_session)
    service.update_chapter_workspace(
        chapter_workspace_id=chapter.chapter_workspace_id,
        phase=LongformChapterPhase.OUTLINE_REVIEW,
    )
    service.update_session(
        session_id=session.session_id,
        current_phase=LongformChapterPhase.OUTLINE_REVIEW,
    )
    outline = service.create_artifact(
        session_id=session.session_id,
        chapter_workspace_id=chapter.chapter_workspace_id,
        artifact_kind=StoryArtifactKind.CHAPTER_OUTLINE,
        status=StoryArtifactStatus.DRAFT,
        content_text="Pinned Identity Outline",
    )
    service.commit()
    identity_service = StoryRuntimeIdentityService(
        retrieval_session,
        runtime_profile_snapshot_service=RuntimeProfileSnapshotService(
            retrieval_session
        ),
    )
    runner = StoryGraphRunner(
        nodes=StoryGraphNodes(
            domain_service=_build_turn_domain_service(
                service,
                runtime_identity_service=identity_service,
            )
        )
    )

    response = await runner.run_turn(
        LongformTurnRequest(
            session_id=session.session_id,
            command_kind=LongformTurnCommandKind.ACCEPT_OUTLINE,
            model_id="model",
            target_artifact_id=outline.artifact_id,
        )
    )
    debug = runner.get_runtime_debug(session_id=session.session_id)
    turn = retrieval_session.exec(
        select(StoryTurnRecord)
        .where(StoryTurnRecord.session_id == session.session_id)
        .order_by(desc(StoryTurnRecord.created_at))
    ).first()
    branch = retrieval_session.exec(
        select(BranchHeadRecord).where(
            BranchHeadRecord.session_id == session.session_id
        )
    ).first()

    assert response.current_phase == LongformChapterPhase.SEGMENT_DRAFTING
    assert branch is not None
    assert turn is not None
    assert turn.branch_head_id == branch.branch_head_id
    assert turn.runtime_profile_snapshot_id
    assert turn.status == "completed"
    assert turn.completed_at is not None
    expected_thread_id = StoryTurnDomainService.build_graph_thread_id(
        session_id=session.session_id,
        branch_head_id=branch.branch_head_id,
    )
    assert debug["thread_id"] == expected_thread_id + ":rp_story"
    assert debug["branch_head_id"] == branch.branch_head_id
    meaningful_state = debug["latest_meaningful_checkpoint"]["state"]
    assert meaningful_state["graph_thread_id"] == expected_thread_id
    assert meaningful_state["branch_head_id"] == branch.branch_head_id
    assert meaningful_state["turn_id"] == turn.turn_id
    assert (
        meaningful_state["runtime_profile_snapshot_id"]
        == turn.runtime_profile_snapshot_id
    )
    assert meaningful_state["runtime_identity"]["turn_id"] == turn.turn_id


@pytest.mark.asyncio
async def test_story_graph_runner_uses_active_branch_for_graph_thread_binding(
    retrieval_session,
):
    session, chapter, service = _seed_story_runtime(retrieval_session)
    service.update_chapter_workspace(
        chapter_workspace_id=chapter.chapter_workspace_id,
        phase=LongformChapterPhase.OUTLINE_REVIEW,
    )
    service.update_session(
        session_id=session.session_id,
        current_phase=LongformChapterPhase.OUTLINE_REVIEW,
    )
    sibling_branch = BranchHeadRecord(
        branch_head_id=f"branch:{session.session_id}:sibling",
        story_id=session.story_id,
        session_id=session.session_id,
        branch_name="sibling",
        parent_branch_head_id=f"branch:{session.session_id}:main",
        forked_from_turn_id=None,
        head_turn_id=None,
        status="active",
        visibility_scope="active_lineage",
    )
    retrieval_session.add(sibling_branch)
    retrieval_session.flush()
    service.update_session(
        session_id=session.session_id,
        active_branch_head_id=sibling_branch.branch_head_id,
    )
    outline = service.create_artifact(
        session_id=session.session_id,
        chapter_workspace_id=chapter.chapter_workspace_id,
        artifact_kind=StoryArtifactKind.CHAPTER_OUTLINE,
        status=StoryArtifactStatus.DRAFT,
        content_text="Sibling Branch Outline",
    )
    service.commit()
    identity_service = StoryRuntimeIdentityService(
        retrieval_session,
        runtime_profile_snapshot_service=RuntimeProfileSnapshotService(
            retrieval_session
        ),
    )
    runner = StoryGraphRunner(
        nodes=StoryGraphNodes(
            domain_service=_build_turn_domain_service(
                service,
                runtime_identity_service=identity_service,
            )
        )
    )

    response = await runner.run_turn(
        LongformTurnRequest(
            session_id=session.session_id,
            command_kind=LongformTurnCommandKind.ACCEPT_OUTLINE,
            model_id="model",
            target_artifact_id=outline.artifact_id,
        )
    )
    debug = runner.get_runtime_debug(session_id=session.session_id)
    turn = retrieval_session.exec(
        select(StoryTurnRecord)
        .where(StoryTurnRecord.session_id == session.session_id)
        .order_by(desc(StoryTurnRecord.created_at))
    ).first()

    assert response.current_phase == LongformChapterPhase.SEGMENT_DRAFTING
    assert turn is not None
    assert turn.branch_head_id == sibling_branch.branch_head_id
    expected_thread_id = StoryTurnDomainService.build_graph_thread_id(
        session_id=session.session_id,
        branch_head_id=sibling_branch.branch_head_id,
    )
    assert debug["thread_id"] == expected_thread_id + ":rp_story"
    assert debug["branch_head_id"] == sibling_branch.branch_head_id
    assert (
        debug["latest_meaningful_checkpoint"]["state"]["graph_thread_id"]
        == expected_thread_id
    )


def test_graph_thread_binding_reports_rollback_visible_turn_head(
    retrieval_session,
):
    session, _, service = _seed_story_runtime(retrieval_session)
    snapshot_service = RuntimeProfileSnapshotService(retrieval_session)
    snapshot = snapshot_service.ensure_active_snapshot(
        session_id=session.session_id,
        created_from="test.graph_binding.rollback",
    )
    identity_service = StoryRuntimeIdentityService(
        retrieval_session,
        runtime_profile_snapshot_service=snapshot_service,
    )
    identities = []
    for index in range(3):
        identity = identity_service.resolve_runtime_entry_identity(
            session_id=session.session_id,
            command_kind=f"continue-{index}",
            actor="story_runtime",
            requested_runtime_profile_snapshot_id=snapshot.runtime_profile_snapshot_id,
        )
        identity_service.update_turn_status(
            turn_id=identity.turn_id,
            status=StoryTurnStatus.SETTLED,
            visible_output_ref=f"artifact:{index}",
            selected_output_ref=f"artifact:{index}",
            settlement_reason="test_settled",
        )
        identities.append(identity)
    target_identity = identities[1]

    identity_service.rollback_to_turn(
        session_id=session.session_id,
        target_turn_id=target_identity.turn_id,
        actor="user",
    )
    domain_service = _build_turn_domain_service(
        service,
        runtime_identity_service=identity_service,
    )

    binding = domain_service.resolve_graph_thread_binding(session_id=session.session_id)

    expected_thread_id = StoryTurnDomainService.build_graph_thread_id(
        session_id=session.session_id,
        branch_head_id=target_identity.branch_head_id,
    )
    assert binding["branch_head_id"] == target_identity.branch_head_id
    assert binding["graph_thread_id"] == expected_thread_id
    assert binding["visible_turn_head_id"] == target_identity.turn_id
    assert binding["last_settled_turn_id"] == target_identity.turn_id


@pytest.mark.asyncio
async def test_story_graph_runner_marks_pinned_turn_failed_on_command_error(
    retrieval_session,
):
    session, _, service = _seed_story_runtime(retrieval_session)
    identity_service = StoryRuntimeIdentityService(
        retrieval_session,
        runtime_profile_snapshot_service=RuntimeProfileSnapshotService(
            retrieval_session
        ),
    )
    runner = StoryGraphRunner(
        nodes=StoryGraphNodes(
            domain_service=_build_turn_domain_service(
                service,
                runtime_identity_service=identity_service,
            )
        )
    )

    with pytest.raises(ValueError):
        await runner.run_turn(
            LongformTurnRequest(
                session_id=session.session_id,
                command_kind=LongformTurnCommandKind.ACCEPT_OUTLINE,
                model_id="model",
            )
        )
    turn = retrieval_session.exec(
        select(StoryTurnRecord)
        .where(StoryTurnRecord.session_id == session.session_id)
        .order_by(desc(StoryTurnRecord.created_at))
    ).first()

    assert turn is not None
    assert turn.status == "failed"
    assert turn.completed_at is not None


@pytest.mark.asyncio
async def test_story_graph_runner_stream_persists_usage_metadata_into_writing_result(
    retrieval_session,
):
    session, chapter, service = _seed_story_runtime(retrieval_session)
    service.update_chapter_workspace(
        chapter_workspace_id=chapter.chapter_workspace_id,
        phase=LongformChapterPhase.OUTLINE_REVIEW,
    )
    service.update_session(
        session_id=session.session_id,
        current_phase=LongformChapterPhase.OUTLINE_REVIEW,
    )
    outline = service.create_artifact(
        session_id=session.session_id,
        chapter_workspace_id=chapter.chapter_workspace_id,
        artifact_kind=StoryArtifactKind.CHAPTER_OUTLINE,
        status=StoryArtifactStatus.DRAFT,
        content_text="Draft Outline",
    )
    service.commit()

    identity_service = StoryRuntimeIdentityService(
        retrieval_session,
        runtime_profile_snapshot_service=RuntimeProfileSnapshotService(
            retrieval_session
        ),
    )
    writing_worker_execution_service = WritingWorkerExecutionService(
        llm_gateway=_StreamingStoryLlmGateway("unused"),
    )
    runner = StoryGraphRunner(
        nodes=StoryGraphNodes(
            domain_service=_build_turn_domain_service(
                service,
                orchestrator_service=_AsyncOrchestratorService(),
                specialist_service=_AsyncSpecialistService(),
                runtime_identity_service=identity_service,
                runtime_workspace_material_service=RuntimeWorkspaceMaterialService(
                    session=retrieval_session
                ),
                writing_worker_execution_service=writing_worker_execution_service,
            )
        )
    )

    accepted_outline = await runner.run_turn(
        LongformTurnRequest(
            session_id=session.session_id,
            command_kind=LongformTurnCommandKind.ACCEPT_OUTLINE,
            model_id="model",
            target_artifact_id=outline.artifact_id,
        )
    )
    assert accepted_outline.current_phase == LongformChapterPhase.SEGMENT_DRAFTING

    chunks = []
    async for chunk in runner.run_turn_stream(
        LongformTurnRequest(
            session_id=session.session_id,
            command_kind=LongformTurnCommandKind.WRITE_NEXT_SEGMENT,
            model_id="model",
            user_prompt="Continue.",
        )
    ):
        chunks.append(chunk)

    assert any('"type":"usage"' in chunk for chunk in chunks)

    snapshot = service.build_chapter_snapshot(
        session_id=session.session_id,
        chapter_index=1,
    )
    draft_segment = next(
        item
        for item in snapshot.artifacts
        if item.artifact_kind == StoryArtifactKind.STORY_SEGMENT
        and item.status == StoryArtifactStatus.DRAFT
    )
    assert draft_segment.content_text == "Streamed outline."
    assert draft_segment.metadata["command_kind"] == "write_next_segment"
    assert snapshot.chapter.pending_segment_artifact_id == draft_segment.artifact_id
    assert snapshot.chapter.builder_snapshot_json["recent_segment_digest"][-1] == (
        "Streamed outline."
    )

    turn = retrieval_session.exec(
        select(StoryTurnRecord)
        .where(StoryTurnRecord.session_id == session.session_id)
        .where(StoryTurnRecord.command_kind == "write_next_segment")
        .order_by(desc(StoryTurnRecord.created_at))
    ).first()
    assert turn is not None
    assert turn.status == "settled"
    assert turn.settlement_reason == "required_jobs_deferred_by_policy"
    branch = retrieval_session.get(BranchHeadRecord, turn.branch_head_id)
    assert branch is not None
    graph_binding = branch.metadata_json["graph_checkpoint_bindings_by_turn_id"][
        turn.turn_id
    ]
    assert graph_binding["turn_id"] == turn.turn_id
    assert graph_binding["branch_head_id"] == turn.branch_head_id
    assert graph_binding["runtime_profile_snapshot_id"] == (
        turn.runtime_profile_snapshot_id
    )
    assert graph_binding["checkpoint_ns"] == "rp_story"
    assert graph_binding["checkpoint_id"]
    assert graph_binding["captured_after_node"] == "finalize_turn"
    assert graph_binding["source"] == "langgraph_checkpoint"
    assert graph_binding["graph_thread_id"] == (
        f"story_session:{session.session_id}:branch_head:{turn.branch_head_id}"
    )
    assert branch.metadata_json["graph_checkpoint_binding"] == graph_binding
    workflow_jobs = retrieval_session.exec(
        select(RuntimeWorkflowJobRecord).where(
            RuntimeWorkflowJobRecord.turn_id == turn.turn_id
        )
    ).all()
    assert {job.job_kind for job in workflow_jobs} == {
        "required_post_write_analysis",
        "runtime_workspace_finalize",
    }
    assert {job.status for job in workflow_jobs} == {"deferred"}
    debug = runner.get_runtime_debug(session_id=session.session_id)
    trigger_state = debug["latest_meaningful_checkpoint"]["state"]["post_write_trigger"]
    assert trigger_state["run_kind"] == "minimal_only"
    assert trigger_state["settled"] is True

    material_service = RuntimeWorkspaceMaterialService(session=retrieval_session)
    identity = MemoryRuntimeIdentity(
        story_id=session.story_id,
        session_id=session.session_id,
        branch_head_id=turn.branch_head_id,
        turn_id=turn.turn_id,
        runtime_profile_snapshot_id=turn.runtime_profile_snapshot_id,
    )
    usage_materials = material_service.list_materials(
        identity=identity,
        material_kind=RuntimeWorkspaceMaterialKind.TOKEN_USAGE_METADATA,
    )
    assert len(usage_materials) == 1
    assert usage_materials[0].payload["usage_metadata"] == {
        "prompt_tokens": 42,
        "completion_tokens": 11,
        "total_tokens": 53,
    }
    writer_output_materials = material_service.list_materials(
        identity=identity,
        material_kind=RuntimeWorkspaceMaterialKind.WRITER_OUTPUT_REF,
    )
    packet_materials = material_service.list_materials(
        identity=identity,
        material_kind=RuntimeWorkspaceMaterialKind.PACKET_REF,
    )
    writer_input_materials = material_service.list_materials(
        identity=identity,
        material_kind=RuntimeWorkspaceMaterialKind.WRITER_INPUT_REF,
    )
    assert len(writer_output_materials) == 1
    assert (
        writer_output_materials[0].payload["artifact_id"] == draft_segment.artifact_id
    )
    assert writer_output_materials[0].payload["artifact_kind"] == (
        StoryArtifactKind.STORY_SEGMENT.value
    )
    assert len(packet_materials) == 1
    assert (
        packet_materials[0].payload["packet_id"]
        == (writer_output_materials[0].payload["packet_id"])
    )
    assert packet_materials[0].payload["output_kind"] == (
        StoryArtifactKind.STORY_SEGMENT.value
    )
    assert packet_materials[0].payload["runtime_read_manifest_id"]
    assert writer_output_materials[0].source_refs[0].entry_id == (
        packet_materials[0].material_id
    )
    assert len(writer_input_materials) == 1
    assert (
        writer_input_materials[0].payload["packet_id"]
        == (packet_materials[0].payload["packet_id"])
    )
    assert writer_input_materials[0].payload["output_kind"] == (
        StoryArtifactKind.STORY_SEGMENT.value
    )
    assert writer_input_materials[0].payload["user_instruction_preview"] == (
        "Continue."
    )
    assert writer_input_materials[0].material_id in {
        source_ref.entry_id for source_ref in packet_materials[0].source_refs
    }
    assert usage_materials[0].payload["artifact_id"] == draft_segment.artifact_id
    assert (
        usage_materials[0].payload["packet_id"]
        == (packet_materials[0].payload["packet_id"])
    )
    assert writer_output_materials[0].material_id in {
        source_ref.entry_id for source_ref in usage_materials[0].source_refs
    }


@pytest.mark.asyncio
async def test_story_graph_runner_stream_buffers_when_writer_retrieval_loop_is_enabled(
    retrieval_session,
    monkeypatch,
):
    original_build_writing_packet = ContextOrchestrationService.build_writing_packet

    def _build_retrieval_enabled_packet(self, *args, **kwargs):
        packet = original_build_writing_packet(self, *args, **kwargs)
        return packet.model_copy(
            update={
                "metadata": {
                    **dict(packet.metadata),
                    "writer_retrieval_allowed": True,
                    "writer_max_retrieval_attempts": 2,
                }
            }
        )

    monkeypatch.setattr(
        ContextOrchestrationService,
        "build_writing_packet",
        _build_retrieval_enabled_packet,
    )

    session, chapter, service = _seed_story_runtime(retrieval_session)
    service.update_chapter_workspace(
        chapter_workspace_id=chapter.chapter_workspace_id,
        phase=LongformChapterPhase.OUTLINE_REVIEW,
    )
    service.update_session(
        session_id=session.session_id,
        current_phase=LongformChapterPhase.OUTLINE_REVIEW,
    )
    outline = service.create_artifact(
        session_id=session.session_id,
        chapter_workspace_id=chapter.chapter_workspace_id,
        artifact_kind=StoryArtifactKind.CHAPTER_OUTLINE,
        status=StoryArtifactStatus.DRAFT,
        content_text="Draft Outline",
    )
    service.commit()

    async def _fake_search_recall_to_cards(
        self,
        *,
        identity,
        input_model,
        actor,
        attempt_index,
    ):
        workspace = self._workspace()
        material = workspace.get_material(
            identity=identity,
            material_id="buffered-loop-card",
        )
        if material is None:
            material = workspace.record_material(
                RuntimeWorkspaceMaterial(
                    material_id="buffered-loop-card",
                    material_kind=RuntimeWorkspaceMaterialKind.RETRIEVAL_CARD,
                    identity=identity,
                    domain="chapter",
                    domain_path="chapter.runtime.retrieval.card",
                    short_id="R1",
                    payload={
                        "title": "Storm Callback",
                        "summary": "Retrieved evidence for the buffered writer loop.",
                        "query_text": getattr(input_model, "query", None),
                        "search_kind": "recall",
                    },
                    visibility="writer_visible",
                    created_by=actor,
                )
            ).material
        return SimpleNamespace(hits=[], warnings=[]), [material], None

    monkeypatch.setattr(
        RuntimeRetrievalCardService,
        "search_recall_to_cards",
        _fake_search_recall_to_cards,
    )

    identity_service = StoryRuntimeIdentityService(
        retrieval_session,
        runtime_profile_snapshot_service=RuntimeProfileSnapshotService(
            retrieval_session
        ),
    )
    retrieval_service = RuntimeRetrievalCardService(session=retrieval_session)
    gateway = _ToolLoopStoryLlmGateway(
        [
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                _tool_loop_call(
                                    "call_search",
                                    "retrieval.search",
                                    {"query": "storm", "mode": "semantic"},
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
                                _tool_loop_call(
                                    "call_usage",
                                    "retrieval.usage",
                                    {
                                        "used_card_short_ids": ["R1"],
                                    },
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
                    {"message": {"content": "Retrieved segment grounded in evidence."}}
                ],
                "usage": {
                    "prompt_tokens": 6,
                    "completion_tokens": 3,
                    "total_tokens": 9,
                },
            },
        ]
    )
    writing_worker_execution_service = WritingWorkerExecutionService(
        llm_gateway=gateway,
        runtime_retrieval_card_service=retrieval_service,
    )
    runner = StoryGraphRunner(
        nodes=StoryGraphNodes(
            domain_service=_build_turn_domain_service(
                service,
                orchestrator_service=_AsyncOrchestratorService(),
                specialist_service=_AsyncSpecialistService(),
                runtime_identity_service=identity_service,
                runtime_workspace_material_service=RuntimeWorkspaceMaterialService(
                    session=retrieval_session
                ),
                writing_worker_execution_service=writing_worker_execution_service,
            )
        )
    )

    accepted_outline = await runner.run_turn(
        LongformTurnRequest(
            session_id=session.session_id,
            command_kind=LongformTurnCommandKind.ACCEPT_OUTLINE,
            model_id="model",
            target_artifact_id=outline.artifact_id,
        )
    )
    assert accepted_outline.current_phase == LongformChapterPhase.SEGMENT_DRAFTING

    chunks: list[str] = []
    async for chunk in runner.run_turn_stream(
        LongformTurnRequest(
            session_id=session.session_id,
            command_kind=LongformTurnCommandKind.WRITE_NEXT_SEGMENT,
            model_id="model",
            user_prompt="Continue.",
        )
    ):
        chunks.append(chunk)

    body = "".join(chunks)
    assert '"type": "text_delta"' in body
    assert "Retrieved segment grounded in evidence." in body
    assert '"type": "usage"' in body

    turn = retrieval_session.exec(
        select(StoryTurnRecord)
        .where(StoryTurnRecord.session_id == session.session_id)
        .where(StoryTurnRecord.command_kind == "write_next_segment")
        .order_by(desc(StoryTurnRecord.created_at))
    ).first()
    assert turn is not None
    identity = MemoryRuntimeIdentity(
        story_id=session.story_id,
        session_id=session.session_id,
        branch_head_id=turn.branch_head_id,
        turn_id=turn.turn_id,
        runtime_profile_snapshot_id=turn.runtime_profile_snapshot_id,
    )
    snapshot = service.build_chapter_snapshot(
        session_id=session.session_id,
        chapter_index=1,
    )
    draft_segment = next(
        item
        for item in snapshot.artifacts
        if item.artifact_kind == StoryArtifactKind.STORY_SEGMENT
        and item.status == StoryArtifactStatus.DRAFT
    )
    assert draft_segment.metadata["worker_source_ref_bundle"][
        "retrieval_usage_material_ids"
    ]

    writer_output_materials = RuntimeWorkspaceMaterialService(
        session=retrieval_session
    ).list_materials(
        identity=identity,
        material_kind=RuntimeWorkspaceMaterialKind.WRITER_OUTPUT_REF,
    )
    assert writer_output_materials
    usage_materials = RuntimeWorkspaceMaterialService(
        session=retrieval_session
    ).list_materials(
        identity=identity,
        material_kind=RuntimeWorkspaceMaterialKind.TOKEN_USAGE_METADATA,
    )
    assert usage_materials


@pytest.mark.asyncio
async def test_specialist_retrieval_inputs_carry_runtime_identity_filters(
    retrieval_session,
):
    session, chapter, service = _seed_story_runtime(retrieval_session)
    authoritative_state_view_service, projection_state_service = (
        _build_boundary_services(service)
    )
    recorder = _RecordingMemoryOsService()
    specialist_service = LongformSpecialistService(
        authoritative_state_view_service=authoritative_state_view_service,
        projection_state_service=projection_state_service,
        memory_os_factory=lambda story_id, **_kwargs: recorder,
    )
    runtime_identity = MemoryRuntimeIdentity(
        story_id=session.story_id,
        session_id=session.session_id,
        branch_head_id="branch:test:main",
        turn_id="turn:test:1",
        runtime_profile_snapshot_id="snapshot:test:1",
    )

    await specialist_service.analyze(
        session=session,
        chapter=chapter,
        plan=OrchestratorPlan(
            output_kind=StoryArtifactKind.STORY_SEGMENT,
            needs_retrieval=True,
            archival_queries=["archive pin"],
            recall_queries=["recall pin"],
            writer_instruction="Write the next segment.",
        ),
        command_kind=LongformTurnCommandKind.WRITE_NEXT_SEGMENT,
        model_id="model",
        provider_id=None,
        user_prompt="Continue.",
        accepted_segments=[],
        pending_artifact=None,
        runtime_identity=runtime_identity,
    )

    assert recorder.archival_inputs
    assert recorder.recall_inputs
    assert (
        recorder.archival_inputs[0].filters["runtime_identity"][
            "runtime_profile_snapshot_id"
        ]
        == "snapshot:test:1"
    )
    assert recorder.recall_inputs[0].filters["runtime_identity"]["turn_id"] == (
        "turn:test:1"
    )


@pytest.mark.asyncio
async def test_orchestrator_and_specialist_include_block_prompt_context(
    retrieval_session,
):
    session, chapter, service = _seed_story_runtime(retrieval_session)
    authoritative_state_view_service, projection_state_service = (
        _build_boundary_services(service)
    )
    proposal_repository = ProposalRepository(retrieval_session)
    version_history_read_service = VersionHistoryReadService(
        adapter=StorySessionCoreStateAdapter(service),
        proposal_repository=proposal_repository,
    )
    builder_projection_context_service = BuilderProjectionContextService(
        projection_state_service
    )
    memory_inspection_read_service = MemoryInspectionReadService(
        story_session_service=service,
        builder_projection_context_service=builder_projection_context_service,
        proposal_repository=proposal_repository,
        version_history_read_service=version_history_read_service,
    )
    rp_block_read_service = RpBlockReadService(
        story_session_service=service,
        builder_projection_context_service=builder_projection_context_service,
        memory_inspection_read_service=memory_inspection_read_service,
    )
    consumer_state_service = StoryBlockConsumerStateService(
        session=retrieval_session,
        story_session_service=service,
        rp_block_read_service=rp_block_read_service,
    )
    block_prompt_context_service = StoryBlockPromptContextService(
        rp_block_read_service=rp_block_read_service,
        story_block_consumer_state_service=consumer_state_service,
    )
    block_prompt_render_service = StoryBlockPromptRenderService()
    block_prompt_compile_service = StoryBlockPromptCompileService(
        story_block_prompt_context_service=block_prompt_context_service,
        story_block_prompt_render_service=block_prompt_render_service,
        story_block_consumer_state_service=consumer_state_service,
    )
    orchestrator_gateway = _RecordingStoryLlmGateway(
        response_text=json.dumps(
            {
                "output_kind": StoryArtifactKind.STORY_SEGMENT.value,
                "needs_retrieval": True,
                "archival_queries": ["archive policy"],
                "recall_queries": ["storm callback"],
                "specialist_focus": ["segment continuity"],
                "writer_instruction": "Write the next segment.",
                "notes": ["gateway"],
            }
        )
    )
    specialist_gateway = _RecordingStoryLlmGateway(
        response_text=SpecialistResultBundle(
            foundation_digest=["Found A"],
            blueprint_digest=["Blueprint A"],
            current_outline_digest=["Outline A"],
            recent_segment_digest=["Segment A"],
            current_state_digest=["State A"],
            writer_hints=["Hint A"],
            story_segment_metadata=StorySegmentStructuredMetadata(
                foreshadow_status_updates=[
                    {
                        "foreshadow_id": "envoy_debt",
                        "status": "resolved",
                        "summary": "bell tower debt",
                    }
                ]
            ),
        ).model_dump_json()
    )
    archival_hits = [
        RetrievalHit(
            hit_id="chunk-archive-1",
            query_id="rq-archive-1",
            layer=Layer.ARCHIVAL.value,
            domain=Domain.WORLD_RULE,
            domain_path="foundation.world.rules.archive_policy",
            knowledge_ref=ObjectRef(
                object_id="world_rule.archive_policy",
                layer=Layer.ARCHIVAL,
                domain=Domain.WORLD_RULE,
                domain_path="foundation.world.rules.archive_policy",
                scope="story",
                revision=2,
            ),
            excerpt_text="Archive policy says all relics must be sealed at dusk.",
            score=0.91,
            rank=1,
            metadata={"title": "Archive Policy"},
            provenance_refs=["prov:archive-policy"],
        )
    ]
    recall_hits = [
        RetrievalHit(
            hit_id="recall-note-1",
            query_id="rq-recall-1",
            layer=Layer.RECALL.value,
            domain=Domain.CHAPTER,
            domain_path="chapter.one.scene.callback",
            knowledge_ref=ObjectRef(
                object_id="recall.note.storm_callback",
                layer=Layer.RECALL,
                domain=Domain.CHAPTER,
                domain_path="chapter.one.scene.callback",
                scope="story",
                revision=1,
            ),
            excerpt_text="Earlier in chapter one, the seal broke during the storm.",
            score=0.77,
            rank=1,
            metadata={
                "title": "Storm Callback",
                "layer": "recall",
                "source_family": "longform_story_runtime",
                "materialization_event": "heavy_regression.chapter_close",
                "materialization_kind": "accepted_story_segment",
                "materialized_to_recall": True,
                "chapter_index": 1,
                "artifact_id": "artifact-storm",
                "artifact_revision": 2,
                "asset_id": "recall_detail_artifact-storm",
                "asset_kind": "accepted_story_segment",
                "source_ref": (
                    "story_session:session-1:chapter:1:artifact:artifact-storm"
                ),
            },
            provenance_refs=["prov:storm-callback"],
        )
    ]
    orchestrator_service = LongformOrchestratorService(
        llm_gateway=orchestrator_gateway,
        authoritative_state_view_service=authoritative_state_view_service,
        projection_state_service=projection_state_service,
        story_block_prompt_compile_service=block_prompt_compile_service,
        story_block_prompt_context_service=block_prompt_context_service,
    )
    specialist_service = LongformSpecialistService(
        llm_gateway=specialist_gateway,
        authoritative_state_view_service=authoritative_state_view_service,
        projection_state_service=projection_state_service,
        memory_os_factory=lambda _story_id: _StubMemoryOsService(
            archival_hits=archival_hits,
            recall_hits=recall_hits,
        ),
        story_block_prompt_compile_service=block_prompt_compile_service,
        story_block_prompt_context_service=block_prompt_context_service,
    )

    plan = await orchestrator_service.plan(
        session=session,
        chapter=chapter,
        command_kind=LongformTurnCommandKind.WRITE_NEXT_SEGMENT,
        model_id="model",
        provider_id=None,
        user_prompt="Continue.",
        target_artifact_id=None,
    )
    bundle = await specialist_service.analyze(
        session=session,
        chapter=chapter,
        plan=plan,
        command_kind=LongformTurnCommandKind.WRITE_NEXT_SEGMENT,
        model_id="model",
        provider_id=None,
        user_prompt="Continue.",
        accepted_segments=[],
        pending_artifact=None,
    )

    orchestrator_payload = json.loads(
        orchestrator_gateway.calls[0]["messages"][1].content
    )
    specialist_payload = json.loads(specialist_gateway.calls[0]["messages"][1].content)
    orchestrator_system_prompt = orchestrator_gateway.calls[0]["messages"][0].content
    specialist_system_prompt = specialist_gateway.calls[0]["messages"][0].content

    assert orchestrator_payload["block_context"]["consumer_key"] == "story.orchestrator"
    assert specialist_payload["block_context"]["consumer_key"] == "story.specialist"
    assert (
        orchestrator_payload["authoritative_state"]["chapter_digest"]["title"]
        == "Chapter One"
    )
    assert orchestrator_payload["projection_state"]["current_outline_digest"] == [
        "Outline A"
    ]
    assert specialist_payload["projection_state"]["current_outline_digest"] == [
        "Outline A"
    ]
    assert {
        block["label"]
        for block in orchestrator_payload["block_context"]["attached_blocks"]
    } == {
        "chapter.current",
        "character.state_digest",
        "foreshadow.registry",
        "narrative_progress.current",
        "plot_thread.active",
        "timeline.event_spine",
        "projection.foundation_digest",
        "projection.blueprint_digest",
        "projection.current_outline_digest",
        "projection.recent_segment_digest",
        "projection.current_state_digest",
    }
    assert (
        specialist_payload["block_context"]["attached_blocks"]
        == orchestrator_payload["block_context"]["attached_blocks"]
    )
    assert specialist_payload["archival_hits"][0]["excerpt_text"] == (
        "Archive policy says all relics must be sealed at dusk."
    )
    assert specialist_payload["recall_hits"][0]["excerpt_text"] == (
        "Earlier in chapter one, the seal broke during the storm."
    )
    assert (
        specialist_payload["recall_hits"][0]["source_family"]
        == "longform_story_runtime"
    )
    assert (
        specialist_payload["recall_hits"][0]["materialization_kind"]
        == "accepted_story_segment"
    )
    assert (
        specialist_payload["recall_hits"][0]["materialization_event"]
        == "heavy_regression.chapter_close"
    )
    assert (
        specialist_payload["recall_hits"][0]["metadata"]["artifact_id"]
        == "artifact-storm"
    )
    assert specialist_payload["archival_block_views"][0]["source"] == "retrieval_store"
    assert (
        specialist_payload["archival_block_views"][0]["label"]
        == "world_rule.archive_policy"
    )
    assert (
        specialist_payload["archival_block_views"][0]["data_json"]["excerpt_text"]
        == "Archive policy says all relics must be sealed at dusk."
    )
    assert specialist_payload["recall_block_views"][0]["source"] == "retrieval_store"
    assert (
        specialist_payload["recall_block_views"][0]["label"]
        == "recall.note.storm_callback"
    )
    assert (
        specialist_payload["recall_block_views"][0]["metadata"]["query_id"]
        == "rq-recall-1"
    )
    assert (
        specialist_payload["recall_block_views"][0]["metadata"]["materialization_kind"]
        == "accepted_story_segment"
    )
    assert (
        specialist_payload["recall_block_views"][0]["data_json"]["source_family"]
        == "longform_story_runtime"
    )
    assert "[BLOCK_PROMPT_CONTEXT]" in orchestrator_system_prompt
    assert 'label="chapter.current"' in orchestrator_system_prompt
    assert 'label="projection.current_outline_digest"' in orchestrator_system_prompt
    assert "[BLOCK_PROMPT_CONTEXT]" in specialist_system_prompt
    assert 'label="chapter.current"' in specialist_system_prompt
    assert "gateway" in plan.notes
    assert "adapter_input:legacy_orchestrator_plan" in plan.notes
    assert "not_canonical_worker_plan" in plan.notes
    assert bundle.writer_hints == ["Hint A"]
    assert [
        item.model_dump(mode="json", exclude_none=True)
        for item in bundle.story_segment_metadata.foreshadow_status_updates
    ] == [
        {
            "foreshadow_id": "envoy_debt",
            "status": "resolved",
            "summary": "bell tower debt",
        }
    ]
