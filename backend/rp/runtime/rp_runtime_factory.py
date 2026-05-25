"""Shared dependency factory for RP runtime entrypoints."""

from __future__ import annotations

from config import get_settings
from sqlmodel import Session

from rp.agent_runtime.adapters import SetupRuntimeAdapter
from rp.agent_runtime.executor import RpAgentRuntimeExecutor
from rp.agent_runtime.tools import RuntimeToolExecutor
from rp.graphs.activation_graph_nodes import ActivationGraphNodes
from rp.graphs.activation_graph_runner import ActivationGraphRunner
from rp.graphs.setup_graph_nodes import SetupGraphNodes
from rp.graphs.setup_graph_runner import SetupGraphRunner
from rp.graphs.story_graph_nodes import StoryGraphNodes
from rp.graphs.story_graph_runner import StoryGraphRunner
from rp.services.authoritative_compatibility_mirror_service import (
    AuthoritativeCompatibilityMirrorService,
)
from rp.services.authoritative_state_view_service import AuthoritativeStateViewService
from rp.services.archival_evolution_service import ArchivalEvolutionService
from rp.services.longform_orchestrator_service import LongformOrchestratorService
from rp.services.longform_regression_service import LongformRegressionService
from rp.services.longform_specialist_service import LongformSpecialistService
from rp.services.builder_projection_context_service import (
    BuilderProjectionContextService,
)
from rp.services.chapter_workspace_projection_adapter import (
    ChapterWorkspaceProjectionAdapter,
)
from rp.services.longform_chapter_runtime_service import (
    LongformChapterRuntimeService,
)
from rp.services.core_state_dual_write_service import CoreStateDualWriteService
from rp.services.core_state_as_of_resolver import CoreStateAsOfResolver
from rp.services.core_state_store_repository import CoreStateStoreRepository
from rp.services.context_orchestration_service import ContextOrchestrationService
from rp.services.draft_materialization_service import DraftMaterializationService
from rp.services.legacy_state_patch_proposal_builder import (
    LegacyStatePatchProposalBuilder,
)
from rp.services.local_tool_provider_registry import LocalToolProviderRegistry
from rp.services.memory_inspection_read_service import MemoryInspectionReadService
from rp.services.memory_inspection_service import MemoryInspectionService
from rp.services.memory_trace_read_service import MemoryTraceReadService
from rp.services.memory_os_service import MemoryOsService
from rp.services.minimal_retrieval_ingestion_service import (
    MinimalRetrievalIngestionService,
)
from rp.services.post_write_apply_handler import PostWriteApplyHandler
from rp.services.post_write_governance_service import PostWriteGovernanceService
from rp.services.post_write_scheduler_service import PostWriteSchedulerService
from rp.services.projection_compatibility_mirror_service import (
    ProjectionCompatibilityMirrorService,
)
from rp.services.projection_read_service import ProjectionReadService
from rp.services.projection_state_service import ProjectionStateService
from rp.services.projection_refresh_service import ProjectionRefreshService
from rp.services.provenance_read_service import ProvenanceReadService
from rp.services.proposal_apply_service import ProposalApplyService
from rp.services.proposal_repository import ProposalRepository
from rp.services.proposal_workflow_service import ProposalWorkflowService
from rp.services.recall_character_long_history_ingestion_service import (
    RecallCharacterLongHistoryIngestionService,
)
from rp.services.recall_continuity_note_ingestion_service import (
    RecallContinuityNoteIngestionService,
)
from rp.services.recall_detail_ingestion_service import RecallDetailIngestionService
from rp.services.recall_retired_foreshadow_ingestion_service import (
    RecallRetiredForeshadowIngestionService,
)
from rp.services.recall_lifecycle_service import RecallLifecycleService
from rp.services.recall_scene_transcript_ingestion_service import (
    RecallSceneTranscriptIngestionService,
)
from rp.services.revision_overlay_service import RevisionOverlayService
from rp.services.retrieval_document_service import RetrievalDocumentService
from rp.services.retrieval_broker import RetrievalBroker
from rp.services.recall_summary_ingestion_service import RecallSummaryIngestionService
from rp.services.runtime_read_manifest_service import (
    BranchVisibilityResolver,
    RuntimeReadManifestService,
)
from rp.services.runtime_retrieval_card_service import RuntimeRetrievalCardService
from rp.services.runtime_config_control_service import RuntimeConfigControlService
from rp.services.runtime_workflow_job_service import RuntimeWorkflowJobService
from rp.services.runtime_workspace_material_service import (
    RuntimeWorkspaceMaterialService,
)
from rp.services.rp_block_read_service import RpBlockReadService
from rp.services.setup_agent_execution_service import SetupAgentExecutionService
from rp.services.setup_agent_runtime_state_service import SetupAgentRuntimeStateService
from rp.services.setup_context_builder import SetupContextBuilder
from rp.services.setup_runtime_controller import SetupRuntimeController
from rp.services.setup_workspace_service import SetupWorkspaceService
from rp.services.story_activation_service import StoryActivationService
from rp.services.story_block_consumer_state_service import (
    StoryBlockConsumerStateService,
)
from rp.services.story_block_mutation_service import StoryBlockMutationService
from rp.services.story_block_prompt_compile_service import (
    StoryBlockPromptCompileService,
)
from rp.services.story_block_prompt_context_service import (
    StoryBlockPromptContextService,
)
from rp.services.story_block_prompt_render_service import (
    StoryBlockPromptRenderService,
)
from rp.services.story_brainstorm_service import StoryBrainstormService
from rp.services.story_runtime_controller import StoryRuntimeController
from rp.services.story_runtime_debug_query_service import (
    StoryRuntimeDebugQueryService,
)
from rp.services.story_runtime_identity_service import StoryRuntimeIdentityService
from rp.services.story_runtime_migration_service import StoryRuntimeMigrationService
from rp.services.story_runtime_workspace_facade import (
    StoryRuntimeWorkspaceFacade,
)
from rp.services.story_session_core_state_adapter import StorySessionCoreStateAdapter
from rp.services.story_session_service import StorySessionService
from rp.services.story_state_apply_service import StoryStateApplyService
from rp.services.story_turn_domain_service import StoryTurnDomainService
from rp.services.runtime_profile_snapshot_service import RuntimeProfileSnapshotService
from rp.services.version_history_read_service import VersionHistoryReadService
from rp.services.worker_execution_service import WorkerExecutionService
from rp.services.worker_memory_service import WorkerMemoryService
from rp.services.worker_registry_service import WorkerRegistryService
from rp.services.worker_scheduler_service import WorkerSchedulerService
from rp.services.writing_packet_builder import WritingPacketBuilder
from rp.services.writing_worker_execution_service import WritingWorkerExecutionService
from rp.tools.setup_tool_provider import SetupToolProvider
from rp.tools.setup_scoped_mcp_manager import SetupScopedMcpManager
from services.mcp_manager import McpManager, get_mcp_manager


class RpRuntimeFactory:
    """Centralize RP runtime dependency wiring before deeper graph migration."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def _build_setup_workspace_service(self) -> SetupWorkspaceService:
        return SetupWorkspaceService(self._session)

    def _build_setup_context_builder(
        self,
        *,
        workspace_service: SetupWorkspaceService,
    ) -> SetupContextBuilder:
        return SetupContextBuilder(workspace_service)

    def _build_setup_runtime_state_service(self) -> SetupAgentRuntimeStateService:
        return SetupAgentRuntimeStateService(self._session)

    def _build_setup_runtime_controller(
        self,
        *,
        workspace_service: SetupWorkspaceService,
        context_builder: SetupContextBuilder,
        runtime_state_service: SetupAgentRuntimeStateService | None = None,
    ) -> SetupRuntimeController:
        return SetupRuntimeController(
            workspace_service=workspace_service,
            context_builder=context_builder,
            retrieval_ingestion_service=MinimalRetrievalIngestionService(self._session),
            runtime_state_service=(
                runtime_state_service or self._build_setup_runtime_state_service()
            ),
        )

    def build_setup_runtime_controller(self) -> SetupRuntimeController:
        workspace_service = self._build_setup_workspace_service()
        context_builder = self._build_setup_context_builder(
            workspace_service=workspace_service
        )
        return self._build_setup_runtime_controller(
            workspace_service=workspace_service,
            context_builder=context_builder,
            runtime_state_service=self._build_setup_runtime_state_service(),
        )

    def _build_setup_mcp_manager(
        self,
        *,
        story_id: str,
        external_tool_allowlist: list[str] | None = None,
    ):
        workspace_service = self._build_setup_workspace_service()
        context_builder = self._build_setup_context_builder(
            workspace_service=workspace_service
        )
        runtime_state_service = self._build_setup_runtime_state_service()
        registry = LocalToolProviderRegistry()
        registry.register(
            SetupToolProvider(
                workspace_service=workspace_service,
                context_builder=context_builder,
                runtime_state_service=runtime_state_service,
            )
        )
        setup_tool_manager = McpManager(
            storage_path=None,
            local_tool_provider_registry=registry,
            register_default_local_providers=False,
        )
        if not external_tool_allowlist:
            return setup_tool_manager
        return SetupScopedMcpManager(
            setup_tool_manager=setup_tool_manager,
            external_mcp_manager=get_mcp_manager(),
            external_tool_allowlist=external_tool_allowlist,
        )

    def _build_setup_runtime_executor(self) -> RpAgentRuntimeExecutor:
        return RpAgentRuntimeExecutor(
            tool_executor_factory=lambda turn_input: RuntimeToolExecutor(
                mcp_manager=self._build_setup_mcp_manager(
                    story_id=str(turn_input.story_id or ""),
                    external_tool_allowlist=self._external_tool_allowlist(
                        turn_input.metadata
                    ),
                )
            )
        )

    @staticmethod
    def _external_tool_allowlist(metadata: dict) -> list[str]:
        raw_items = metadata.get("external_mcp_tool_allowlist")
        if not isinstance(raw_items, list):
            return []
        return list(
            dict.fromkeys(
                item.strip()
                for item in raw_items
                if isinstance(item, str) and item.strip()
            )
        )

    @staticmethod
    def _use_core_state_store_write() -> bool:
        return bool(get_settings().rp_memory_core_state_store_write_enabled)

    @staticmethod
    def _use_core_state_store_read() -> bool:
        return bool(get_settings().rp_memory_core_state_store_read_enabled)

    @classmethod
    def _use_core_state_store_write_switch(cls) -> bool:
        settings = get_settings()
        return bool(
            settings.rp_memory_core_state_store_write_enabled
            and settings.rp_memory_core_state_store_write_switch_enabled
        )

    def _build_core_state_dual_write_service(self) -> CoreStateDualWriteService | None:
        if not self._use_core_state_store_write():
            return None
        return CoreStateDualWriteService(
            repository=CoreStateStoreRepository(self._session)
        )

    @staticmethod
    def _build_authoritative_compatibility_mirror_service(
        *,
        story_session_service: StorySessionService,
    ) -> AuthoritativeCompatibilityMirrorService:
        return AuthoritativeCompatibilityMirrorService(
            story_session_service=story_session_service
        )

    @staticmethod
    def _build_projection_compatibility_mirror_service(
        *,
        story_session_service: StorySessionService,
    ) -> ProjectionCompatibilityMirrorService:
        return ProjectionCompatibilityMirrorService(
            story_session_service=story_session_service
        )

    def _build_setup_execution_service(
        self,
        *,
        workspace_service: SetupWorkspaceService,
        context_builder: SetupContextBuilder,
    ) -> SetupAgentExecutionService:
        runtime_state_service = self._build_setup_runtime_state_service()
        return SetupAgentExecutionService(
            workspace_service=workspace_service,
            context_builder=context_builder,
            runtime_executor=self._build_setup_runtime_executor(),
            adapter=SetupRuntimeAdapter(),
            runtime_state_service=runtime_state_service,
        )

    def build_setup_agent_execution_service(self) -> SetupAgentExecutionService:
        workspace_service = self._build_setup_workspace_service()
        context_builder = self._build_setup_context_builder(
            workspace_service=workspace_service
        )
        return self._build_setup_execution_service(
            workspace_service=workspace_service,
            context_builder=context_builder,
        )

    def build_setup_graph_runner(self) -> SetupGraphRunner:
        workspace_service = self._build_setup_workspace_service()
        context_builder = self._build_setup_context_builder(
            workspace_service=workspace_service
        )
        execution_service = self._build_setup_execution_service(
            workspace_service=workspace_service,
            context_builder=context_builder,
        )
        return SetupGraphRunner(
            nodes=SetupGraphNodes(
                workspace_service=workspace_service,
                execution_service=execution_service,
            ),
            execution_service=execution_service,
        )

    def build_activation_graph_runner(self) -> ActivationGraphRunner:
        workspace_service = self._build_setup_workspace_service()
        context_builder = self._build_setup_context_builder(
            workspace_service=workspace_service
        )
        setup_controller = self._build_setup_runtime_controller(
            workspace_service=workspace_service,
            context_builder=context_builder,
        )
        story_session_service = StorySessionService(self._session)
        return ActivationGraphRunner(
            nodes=ActivationGraphNodes(
                workspace_service=workspace_service,
                setup_controller=setup_controller,
                story_session_service=story_session_service,
            ),
        )

    def build_story_turn_domain_service(self) -> StoryTurnDomainService:
        story_session_service = StorySessionService(self._session)
        authoritative_mirror_service = (
            self._build_authoritative_compatibility_mirror_service(
                story_session_service=story_session_service
            )
        )
        projection_mirror_service = self._build_projection_compatibility_mirror_service(
            story_session_service=story_session_service
        )
        core_state_dual_write_service = self._build_core_state_dual_write_service()
        core_state_store_repository = CoreStateStoreRepository(self._session)
        core_state_as_of_resolver = CoreStateAsOfResolver(
            session=self._session,
            repository=core_state_store_repository,
        )
        store_read_enabled = self._use_core_state_store_read()
        store_write_switch_enabled = self._use_core_state_store_write_switch()
        core_state_adapter = StorySessionCoreStateAdapter(story_session_service)
        authoritative_state_view_service = AuthoritativeStateViewService(
            adapter=core_state_adapter,
            core_state_store_repository=core_state_store_repository,
            store_read_enabled=store_read_enabled,
        )
        projection_state_service = ProjectionStateService(
            story_session_service=story_session_service,
            adapter=ChapterWorkspaceProjectionAdapter(story_session_service),
            core_state_dual_write_service=core_state_dual_write_service,
            core_state_store_repository=core_state_store_repository,
            store_read_enabled=store_read_enabled,
            core_state_store_write_switch_enabled=store_write_switch_enabled,
            projection_compatibility_mirror_service=projection_mirror_service,
        )
        proposal_repository = ProposalRepository(self._session)
        version_history_read_service = VersionHistoryReadService(
            adapter=core_state_adapter,
            proposal_repository=proposal_repository,
            core_state_store_repository=core_state_store_repository,
            store_read_enabled=store_read_enabled,
        )
        proposal_apply_service = ProposalApplyService(
            story_session_service=story_session_service,
            proposal_repository=proposal_repository,
            story_state_apply_service=StoryStateApplyService(),
            core_state_dual_write_service=core_state_dual_write_service,
            core_state_store_write_switch_enabled=store_write_switch_enabled,
            authoritative_compatibility_mirror_service=authoritative_mirror_service,
            core_state_as_of_resolver=core_state_as_of_resolver,
        )
        proposal_workflow_service = ProposalWorkflowService(
            proposal_repository=proposal_repository,
            proposal_apply_service=proposal_apply_service,
            post_write_apply_handler=PostWriteApplyHandler(),
        )
        builder_projection_context_service = BuilderProjectionContextService(
            projection_state_service
        )
        memory_inspection_read_service = MemoryInspectionReadService(
            story_session_service=story_session_service,
            builder_projection_context_service=builder_projection_context_service,
            proposal_repository=proposal_repository,
            version_history_read_service=version_history_read_service,
            core_state_store_repository=core_state_store_repository,
            store_read_enabled=store_read_enabled,
        )
        rp_block_read_service = RpBlockReadService(
            story_session_service=story_session_service,
            builder_projection_context_service=builder_projection_context_service,
            core_state_store_repository=core_state_store_repository,
            memory_inspection_read_service=memory_inspection_read_service,
            store_read_enabled=store_read_enabled,
        )
        block_consumer_state_service = StoryBlockConsumerStateService(
            session=self._session,
            story_session_service=story_session_service,
            rp_block_read_service=rp_block_read_service,
        )
        block_prompt_context_service = StoryBlockPromptContextService(
            rp_block_read_service=rp_block_read_service,
            story_block_consumer_state_service=block_consumer_state_service,
        )
        block_prompt_render_service = StoryBlockPromptRenderService()
        block_prompt_compile_service = StoryBlockPromptCompileService(
            story_block_prompt_context_service=block_prompt_context_service,
            story_block_prompt_render_service=block_prompt_render_service,
            story_block_consumer_state_service=block_consumer_state_service,
        )
        orchestrator_service = LongformOrchestratorService(
            authoritative_state_view_service=authoritative_state_view_service,
            projection_state_service=projection_state_service,
            story_block_prompt_compile_service=block_prompt_compile_service,
            story_block_prompt_context_service=block_prompt_context_service,
            story_block_prompt_render_service=block_prompt_render_service,
        )
        runtime_profile_snapshot_service = RuntimeProfileSnapshotService(self._session)
        runtime_workspace_material_service = RuntimeWorkspaceMaterialService(
            session=self._session,
        )
        runtime_retrieval_card_service = RuntimeRetrievalCardService(
            runtime_workspace_material_service=runtime_workspace_material_service,
            session=self._session,
        )
        specialist_service = LongformSpecialistService(
            authoritative_state_view_service=authoritative_state_view_service,
            projection_state_service=projection_state_service,
            memory_os_factory=(
                lambda story_id,
                session_id=None,
                runtime_identity=None: MemoryOsService(
                    retrieval_broker=RetrievalBroker(
                        default_story_id=story_id,
                        runtime_identity=runtime_identity,
                        session=self._session,
                    )
                )
            ),
            runtime_retrieval_card_service=runtime_retrieval_card_service,
            story_block_prompt_compile_service=block_prompt_compile_service,
            story_block_prompt_context_service=block_prompt_context_service,
            story_block_prompt_render_service=block_prompt_render_service,
        )
        projection_refresh_service = ProjectionRefreshService(
            story_session_service,
            core_state_dual_write_service=core_state_dual_write_service,
            core_state_store_write_switch_enabled=store_write_switch_enabled,
            projection_compatibility_mirror_service=projection_mirror_service,
        )
        regression_service = LongformRegressionService(
            story_session_service=story_session_service,
            orchestrator_service=orchestrator_service,
            specialist_service=specialist_service,
            proposal_workflow_service=proposal_workflow_service,
            legacy_state_patch_proposal_builder=LegacyStatePatchProposalBuilder(),
            projection_refresh_service=projection_refresh_service,
            recall_summary_ingestion_service=RecallSummaryIngestionService(
                self._session
            ),
            recall_detail_ingestion_service=RecallDetailIngestionService(self._session),
            recall_continuity_note_ingestion_service=(
                RecallContinuityNoteIngestionService(self._session)
            ),
            recall_character_long_history_ingestion_service=(
                RecallCharacterLongHistoryIngestionService(self._session)
            ),
            recall_retired_foreshadow_ingestion_service=(
                RecallRetiredForeshadowIngestionService(self._session)
            ),
        )
        writing_packet_builder = WritingPacketBuilder()
        writing_worker_execution_service = WritingWorkerExecutionService(
            runtime_retrieval_card_service=runtime_retrieval_card_service,
        )
        runtime_workspace_facade = StoryRuntimeWorkspaceFacade(
            runtime_workspace_material_service=runtime_workspace_material_service,
        )
        worker_registry_service = WorkerRegistryService(
            self._session,
            runtime_profile_snapshot_service=runtime_profile_snapshot_service,
        )
        worker_scheduler_service = WorkerSchedulerService(
            worker_registry_service=worker_registry_service
        )
        runtime_workflow_job_service = RuntimeWorkflowJobService(self._session)
        runtime_read_manifest_service = RuntimeReadManifestService(
            session=self._session,
            runtime_workspace_material_service=runtime_workspace_material_service,
            core_state_as_of_resolver=core_state_as_of_resolver,
        )
        longform_chapter_runtime_service = LongformChapterRuntimeService(
            story_session_service=story_session_service,
            workspace_material_service=runtime_workspace_material_service,
            session=self._session,
        )
        context_orchestration_service = ContextOrchestrationService(
            story_session_service=story_session_service,
            builder_projection_context_service=builder_projection_context_service,
            writing_packet_builder=writing_packet_builder,
            runtime_workspace_material_service=runtime_workspace_material_service,
            runtime_read_manifest_service=runtime_read_manifest_service,
            longform_chapter_runtime_service=longform_chapter_runtime_service,
        )
        worker_execution_service = WorkerExecutionService(
            worker_registry_service=worker_registry_service,
            longform_specialist_service=specialist_service,
            context_orchestration_service=context_orchestration_service,
            runtime_workspace_facade=runtime_workspace_facade,
        )
        post_write_scheduler_service = PostWriteSchedulerService(
            worker_scheduler_service=worker_scheduler_service,
            runtime_workspace_material_service=runtime_workspace_material_service,
        )
        post_write_governance_service = PostWriteGovernanceService(
            runtime_workflow_job_service=runtime_workflow_job_service,
            projection_refresh_service=projection_refresh_service,
            core_state_as_of_resolver=core_state_as_of_resolver,
            proposal_workflow_service=proposal_workflow_service,
            legacy_state_patch_proposal_builder=LegacyStatePatchProposalBuilder(),
        )
        turn_domain_service = StoryTurnDomainService(
            story_session_service=story_session_service,
            orchestrator_service=orchestrator_service,
            specialist_service=specialist_service,
            builder_projection_context_service=builder_projection_context_service,
            projection_state_service=projection_state_service,
            writing_packet_builder=writing_packet_builder,
            writing_worker_execution_service=writing_worker_execution_service,
            regression_service=regression_service,
            block_consumer_state_service=block_consumer_state_service,
            recall_scene_transcript_ingestion_service=(
                RecallSceneTranscriptIngestionService(self._session)
            ),
            runtime_identity_service=StoryRuntimeIdentityService(
                self._session,
                runtime_profile_snapshot_service=runtime_profile_snapshot_service,
            ),
            runtime_read_manifest_service=runtime_read_manifest_service,
            runtime_workspace_material_service=runtime_workspace_material_service,
            worker_scheduler_service=worker_scheduler_service,
            worker_execution_service=worker_execution_service,
            context_orchestration_service=context_orchestration_service,
            runtime_workspace_facade=runtime_workspace_facade,
            runtime_workflow_job_service=runtime_workflow_job_service,
            post_write_scheduler_service=post_write_scheduler_service,
            post_write_governance_service=post_write_governance_service,
            longform_chapter_runtime_service=longform_chapter_runtime_service,
        )
        return turn_domain_service

    def _build_story_activation_service(self) -> StoryActivationService:
        workspace_service = self._build_setup_workspace_service()
        context_builder = self._build_setup_context_builder(
            workspace_service=workspace_service
        )
        setup_controller = self._build_setup_runtime_controller(
            workspace_service=workspace_service,
            context_builder=context_builder,
        )
        story_activation_service = StoryActivationService(
            setup_controller=setup_controller,
            workspace_service=workspace_service,
            story_session_service=StorySessionService(self._session),
            core_state_dual_write_service=self._build_core_state_dual_write_service(),
            runtime_profile_snapshot_service=RuntimeProfileSnapshotService(
                self._session
            ),
        )
        return story_activation_service

    def build_story_runtime_controller(self) -> StoryRuntimeController:
        story_session_service = StorySessionService(self._session)
        story_activation_service = self._build_story_activation_service()
        proposal_repository = ProposalRepository(self._session)
        core_state_adapter = StorySessionCoreStateAdapter(story_session_service)
        authoritative_mirror_service = (
            self._build_authoritative_compatibility_mirror_service(
                story_session_service=story_session_service
            )
        )
        projection_mirror_service = self._build_projection_compatibility_mirror_service(
            story_session_service=story_session_service
        )
        core_state_store_repository = CoreStateStoreRepository(self._session)
        core_state_as_of_resolver = CoreStateAsOfResolver(
            session=self._session,
            repository=core_state_store_repository,
        )
        core_state_dual_write_service = self._build_core_state_dual_write_service()
        store_read_enabled = self._use_core_state_store_read()
        store_write_switch_enabled = self._use_core_state_store_write_switch()
        projection_adapter = ChapterWorkspaceProjectionAdapter(story_session_service)
        projection_state_service = ProjectionStateService(
            story_session_service=story_session_service,
            adapter=projection_adapter,
            core_state_dual_write_service=core_state_dual_write_service,
            core_state_store_repository=core_state_store_repository,
            store_read_enabled=store_read_enabled,
            core_state_store_write_switch_enabled=store_write_switch_enabled,
            projection_compatibility_mirror_service=projection_mirror_service,
        )
        builder_projection_context_service = BuilderProjectionContextService(
            projection_state_service
        )
        version_history_read_service = VersionHistoryReadService(
            adapter=core_state_adapter,
            proposal_repository=proposal_repository,
            core_state_store_repository=core_state_store_repository,
            store_read_enabled=store_read_enabled,
        )
        provenance_read_service = ProvenanceReadService(
            adapter=core_state_adapter,
            proposal_repository=proposal_repository,
            core_state_store_repository=core_state_store_repository,
            store_read_enabled=store_read_enabled,
        )
        projection_read_service = ProjectionReadService(
            adapter=projection_adapter,
            core_state_store_repository=core_state_store_repository,
            store_read_enabled=store_read_enabled,
        )
        memory_inspection_read_service = MemoryInspectionReadService(
            story_session_service=story_session_service,
            builder_projection_context_service=builder_projection_context_service,
            proposal_repository=proposal_repository,
            version_history_read_service=version_history_read_service,
            core_state_store_repository=core_state_store_repository,
            store_read_enabled=store_read_enabled,
        )
        proposal_apply_service = ProposalApplyService(
            story_session_service=story_session_service,
            proposal_repository=proposal_repository,
            story_state_apply_service=StoryStateApplyService(),
            core_state_dual_write_service=core_state_dual_write_service,
            core_state_store_write_switch_enabled=store_write_switch_enabled,
            authoritative_compatibility_mirror_service=authoritative_mirror_service,
            core_state_as_of_resolver=core_state_as_of_resolver,
        )
        proposal_workflow_service = ProposalWorkflowService(
            proposal_repository=proposal_repository,
            proposal_apply_service=proposal_apply_service,
            post_write_apply_handler=PostWriteApplyHandler(),
        )
        rp_block_read_service = RpBlockReadService(
            story_session_service=story_session_service,
            builder_projection_context_service=builder_projection_context_service,
            core_state_store_repository=core_state_store_repository,
            memory_inspection_read_service=memory_inspection_read_service,
            store_read_enabled=store_read_enabled,
        )
        block_consumer_state_service = StoryBlockConsumerStateService(
            session=self._session,
            story_session_service=story_session_service,
            rp_block_read_service=rp_block_read_service,
        )
        block_mutation_service = StoryBlockMutationService(
            story_session_service=story_session_service,
            rp_block_read_service=rp_block_read_service,
            memory_inspection_read_service=memory_inspection_read_service,
            proposal_apply_service=proposal_apply_service,
            proposal_workflow_service=proposal_workflow_service,
        )
        memory_inspection_service = MemoryInspectionService(
            memory_inspection_read_service=memory_inspection_read_service,
            rp_block_read_service=rp_block_read_service,
            story_block_mutation_service=block_mutation_service,
            branch_visibility_resolver=BranchVisibilityResolver(self._session),
            runtime_workspace_material_service=RuntimeWorkspaceMaterialService(
                session=self._session,
            ),
            retrieval_document_service=RetrievalDocumentService(self._session),
            recall_lifecycle_service=RecallLifecycleService(self._session),
            archival_evolution_service=ArchivalEvolutionService(self._session),
        )
        runtime_workspace_material_service = RuntimeWorkspaceMaterialService(
            session=self._session,
        )
        runtime_read_manifest_service = RuntimeReadManifestService(
            session=self._session,
            runtime_workspace_material_service=runtime_workspace_material_service,
            core_state_as_of_resolver=core_state_as_of_resolver,
        )
        memory_trace_read_service = MemoryTraceReadService(
            session=self._session,
            runtime_read_manifest_service=runtime_read_manifest_service,
        )
        story_runtime_debug_query_service = StoryRuntimeDebugQueryService(
            session=self._session,
            story_session_service=story_session_service,
            runtime_profile_snapshot_service=RuntimeProfileSnapshotService(
                self._session
            ),
            memory_trace_read_service=memory_trace_read_service,
            runtime_workflow_job_service=RuntimeWorkflowJobService(self._session),
            branch_visibility_resolver=BranchVisibilityResolver(self._session),
        )
        story_runtime_migration_service = StoryRuntimeMigrationService(
            debug_query_service=story_runtime_debug_query_service,
        )
        runtime_profile_snapshot_service = RuntimeProfileSnapshotService(self._session)
        worker_registry_service = WorkerRegistryService(
            self._session,
            runtime_profile_snapshot_service=runtime_profile_snapshot_service,
        )
        story_brainstorm_service = StoryBrainstormService(
            story_session_service=story_session_service,
            runtime_workspace_material_service=runtime_workspace_material_service,
            proposal_workflow_service=proposal_workflow_service,
            rp_block_read_service=rp_block_read_service,
            worker_registry_service=worker_registry_service,
            worker_memory_service=WorkerMemoryService(session=self._session),
            core_state_as_of_resolver=core_state_as_of_resolver,
        )
        return StoryRuntimeController(
            story_session_service=story_session_service,
            story_activation_service=story_activation_service,
            version_history_read_service=version_history_read_service,
            provenance_read_service=provenance_read_service,
            projection_read_service=projection_read_service,
            memory_inspection_read_service=memory_inspection_read_service,
            memory_inspection_service=memory_inspection_service,
            rp_block_read_service=rp_block_read_service,
            story_block_mutation_service=block_mutation_service,
            story_block_consumer_state_service=block_consumer_state_service,
            recall_scene_transcript_ingestion_service=(
                RecallSceneTranscriptIngestionService(self._session)
            ),
            runtime_profile_snapshot_service=runtime_profile_snapshot_service,
            story_runtime_identity_service=StoryRuntimeIdentityService(
                self._session,
                runtime_profile_snapshot_service=runtime_profile_snapshot_service,
            ),
            runtime_config_control_service=RuntimeConfigControlService(self._session),
            story_runtime_debug_query_service=story_runtime_debug_query_service,
            story_runtime_migration_service=story_runtime_migration_service,
            draft_materialization_service=DraftMaterializationService(),
            revision_overlay_service=RevisionOverlayService(
                workspace_material_service=runtime_workspace_material_service,
                session=self._session,
            ),
            story_brainstorm_service=story_brainstorm_service,
        )

    def build_story_graph_runner(self) -> StoryGraphRunner:
        return StoryGraphRunner(
            nodes=StoryGraphNodes(
                domain_service=self.build_story_turn_domain_service(),
            ),
        )
