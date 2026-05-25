"""Eval runner for setup-first in-process execution."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from models.rp_retrieval_store import (
    EmbeddingRecordRecord,
    IndexJobRecord,
    KnowledgeChunkRecord,
    ParsedDocumentRecord,
    SourceAssetRecord,
)
from models.rp_story_store import ChapterWorkspaceRecord, StorySessionRecord
from models.rp_setup_store import SetupPendingUserEditDeltaRecord
from models.rp_setup_store import SetupWorkspaceRecord
from rp.agent_runtime.contracts import SetupCognitiveStateSnapshot
from rp.models.memory_crud import RetrievalQuery
from rp.models.retrieval_records import SourceAsset
from rp.models.setup_agent import SetupAgentTurnRequest
from rp.models.setup_drafts import (
    FoundationEntry,
    LongformBlueprintDraft,
    StoryConfigDraft,
    WritingContractDraft,
)
from rp.models.setup_handoff import ActivationCheckResult
from rp.models.setup_workspace import SetupStepId, SetupWorkspace, StoryMode
from rp.models.story_runtime import LongformChapterPhase
from rp.runtime.rp_runtime_factory import RpRuntimeFactory
from rp.retrieval.embedder import Embedder
from rp.services.minimal_retrieval_ingestion_service import MinimalRetrievalIngestionService
from rp.services.retrieval_maintenance_service import RetrievalMaintenanceService
from rp.services.retrieval_collection_service import RetrievalCollectionService
from rp.services.retrieval_document_service import RetrievalDocumentService
from rp.services.retrieval_ingestion_service import RetrievalIngestionService
from rp.services.retrieval_service import RetrievalService
from rp.services.setup_agent_runtime_state_service import SetupAgentRuntimeStateService
from rp.services.setup_workspace_service import SetupWorkspaceService
from rp.services.story_activation_service import StoryActivationService
from rp.services.story_session_service import StorySessionService
from sqlmodel import Session, select

from rp.observability.langfuse_scores import emit_ragas_metric_scores
from services.langfuse_service import get_langfuse_service

from .fixtures import ensure_registry_fixtures, normalize_story_mode
from .graders.deterministic import (
    evaluate_deterministic_scores,
    evaluate_diagnostic_expectation_scores,
)
from .graders.subjective import (
    build_subjective_hook_artifacts,
    evaluate_subjective_hook_scores,
)
from .models import EvalCase, EvalFailure, EvalRun, EvalRunResult, utcnow
from .ragas_adapter import (
    RagasRunConfig,
    ragas_available,
    resolve_metric_objects,
    result_to_records,
    run_ragas_evaluation,
)
from .ragas_generation import (
    build_ragas_response_generation_config,
    generate_eval_rag_response,
)
from .ragas_reporting import build_ragas_artifacts, build_ragas_report
from .ragas_runtime import resolve_ragas_runtime_bindings
from .ragas_samples import build_ragas_sample_from_eval_result
from .replay import save_replay
from .reporting import attach_diagnostic_expectation_results, build_report
from .trace_capture import (
    build_activation_trace,
    build_retrieval_trace,
    build_setup_trace,
)


class EvalRunner:
    """Run eval cases against the current RP runtime in-process."""

    def __init__(
        self,
        session: Session,
    ) -> None:
        self._session = session
        self._factory = RpRuntimeFactory(session)
        self._workspace_service = SetupWorkspaceService(session)
        self._runtime_state_service = SetupAgentRuntimeStateService(session)

    async def run_case(self, case: EvalCase) -> EvalRunResult:
        case = self._materialize_isolated_case(case)
        if case.scope == "setup":
            return await self._run_setup_case(case)
        if case.scope == "retrieval":
            return await self._run_retrieval_case(case)
        if case.scope == "activation":
            return await self._run_activation_case(case)
        raise NotImplementedError(f"Unsupported eval scope: {case.scope!r}")

    def _materialize_isolated_case(self, case: EvalCase) -> EvalCase:
        """Clone the case input so each eval run starts from isolated IDs."""

        request = dict(case.input.request)
        workspace_seed = dict(case.input.workspace_seed)
        workspace_seed = self._merge_runtime_story_config_overrides(
            workspace_seed=workspace_seed,
            env_overrides=dict(case.input.env_overrides),
        )
        suffix = f"--run-{uuid4().hex[:8]}"

        workspace_id = request.get("workspace_id")
        if workspace_id:
            request["workspace_id"] = f"{workspace_id}{suffix}"

        base_story_id = str(workspace_seed.get("story_id") or case.case_id.replace(".", "-"))
        workspace_seed["story_id"] = f"{base_story_id}{suffix}"

        return case.model_copy(
            deep=True,
            update={
                "input": case.input.model_copy(
                    deep=True,
                    update={
                        "request": request,
                        "workspace_seed": workspace_seed,
                    },
                )
            },
        )

    def _merge_runtime_story_config_overrides(
        self,
        *,
        workspace_seed: dict,
        env_overrides: dict,
    ) -> dict:
        """Inject retrieval runtime bindings into story_config when eval overrides request it."""

        story_config_patch = {
            "retrieval_embedding_model_id": env_overrides.get("retrieval_embedding_model_id"),
            "retrieval_embedding_provider_id": env_overrides.get("retrieval_embedding_provider_id"),
            "retrieval_rerank_model_id": env_overrides.get("retrieval_rerank_model_id"),
            "retrieval_rerank_provider_id": env_overrides.get("retrieval_rerank_provider_id"),
        }
        story_config_patch = {
            key: str(value)
            for key, value in story_config_patch.items()
            if value is not None and str(value).strip()
        }
        if not story_config_patch:
            return workspace_seed

        drafts = dict(workspace_seed.get("drafts") or {})
        story_config = dict(drafts.get("story_config") or {})
        story_config.update(story_config_patch)
        drafts["story_config"] = story_config
        return {
            **workspace_seed,
            "drafts": drafts,
        }

    async def _run_setup_case(self, case: EvalCase) -> EvalRunResult:
        run_id = uuid4().hex
        trace_id = uuid4().hex
        started_at = utcnow()
        request_payload = deepcopy(case.input.request)

        fixture_ids = ensure_registry_fixtures(request_payload)
        request_payload["provider_id"] = fixture_ids["provider_id"]
        request_payload["model_id"] = fixture_ids["model_id"]
        workspace = self._prepare_setup_workspace(case=case, request_payload=request_payload)
        request_payload["workspace_id"] = workspace.workspace_id
        request = SetupAgentTurnRequest.model_validate(request_payload)

        graph_runner = self._factory.build_setup_graph_runner()
        workspace_before = self._workspace_service.get_workspace(request.workspace_id)
        runtime_events: list[dict] = []
        runtime_result = None
        activation_check: ActivationCheckResult | None = None
        runtime_debug: dict[str, Any] | None = None
        failure = None
        run_status = "completed"

        try:
            if case.runtime_target.mode == "http":
                http_result = await self._run_setup_case_via_http(
                    case=case,
                    request=request,
                )
                runtime_result = http_result["runtime_result"]
                runtime_events = list(http_result["runtime_events"])
                runtime_debug = http_result["runtime_debug"]
                failure = http_result["failure"]
                run_status = "failed" if failure is not None else "completed"
            else:
                if case.runtime_target.stream:
                    async for chunk in graph_runner.run_turn_stream(request):
                        payload = _parse_typed_payload(chunk)
                        if payload is not None and case.trace_hooks.capture_runtime_events:
                            runtime_events.append(payload)
                else:
                    await graph_runner.run_turn(request)
                runtime_result = graph_runner.last_runtime_result
                if case.trace_hooks.capture_graph_debug:
                    runtime_debug = graph_runner.get_runtime_debug(
                        workspace_id=request.workspace_id
                    )
        except Exception as exc:
            runtime_result = graph_runner.last_runtime_result if runtime_result is None else runtime_result
            run_status = "failed"
            failure = _classify_exception_failure(exc, runtime_result)

        if runtime_result is None:
            run_status = "failed"
            failure = failure or EvalFailure(
                layer="infra",
                code="missing_runtime_result",
                message="Setup graph finished without exposing RpAgentTurnResult",
                retryable=False,
                source="runner",
            )
        elif failure is None and runtime_result.status == "failed":
            run_status = "failed"
            failure = _classify_runtime_failure(runtime_result)

        if bool(case.trace_hooks.capture_activation_snapshot):
            controller = self._factory.build_setup_runtime_controller()
            activation_check = controller.run_activation_check(
                workspace_id=request.workspace_id
            )

        finished_at = utcnow()
        workspace_after = self._workspace_service.get_workspace(request.workspace_id)
        runtime_result_payload = (
            runtime_result.model_dump(mode="json")
            if runtime_result is not None
            else {}
        )
        if isinstance(activation_check, ActivationCheckResult):
            runtime_result_payload["activation_check"] = activation_check.model_dump(
                mode="json"
            )
        trace, artifacts = build_setup_trace(
            trace_id=trace_id,
            run_id=run_id,
            case_id=case.case_id,
            story_id=workspace.story_id,
            request=request,
            runtime_result=runtime_result,
            runtime_events=runtime_events,
            workspace_before=workspace_before if case.trace_hooks.capture_workspace_before_after else None,
            workspace_after=workspace_after if case.trace_hooks.capture_workspace_before_after else None,
            runtime_debug=runtime_debug,
            activation_check=activation_check.model_dump(mode="json")
            if isinstance(activation_check, ActivationCheckResult)
            else None,
            capture_tool_sequence=bool(case.trace_hooks.capture_tool_sequence),
            stream_mode=bool(case.runtime_target.stream),
            started_at=started_at,
            finished_at=finished_at,
        )
        scores = evaluate_deterministic_scores(
            case=case,
            run_id=run_id,
            root_span_id=trace.spans[0].span_id if trace.spans else None,
            runtime_result=runtime_result_payload,
            workspace_before=(
                workspace_before.model_dump(mode="json") if workspace_before else None
            ),
            workspace_after=(
                workspace_after.model_dump(mode="json") if workspace_after else None
            ),
            graph_debug=runtime_debug,
            runtime_events=runtime_events,
        )
        subjective_scores = await evaluate_subjective_hook_scores(
            case=case,
            run_id=run_id,
            root_span_id=trace.spans[0].span_id if trace.spans else None,
            runtime_result=runtime_result_payload,
            workspace_before=(
                workspace_before.model_dump(mode="json") if workspace_before else None
            ),
            workspace_after=(
                workspace_after.model_dump(mode="json") if workspace_after else None
            ),
            graph_debug=runtime_debug,
            runtime_events=runtime_events,
            judge_enabled=bool(case.input.env_overrides.get("enable_subjective_judges")),
            judge_model_id=(
                str(case.input.env_overrides.get("judge_model_id") or request.model_id)
                if (case.input.env_overrides.get("judge_model_id") or request.model_id)
                else None
            ),
            judge_provider_id=(
                str(case.input.env_overrides.get("judge_provider_id") or request.provider_id)
                if (case.input.env_overrides.get("judge_provider_id") or request.provider_id)
                else None
            ),
        )
        scores.extend(subjective_scores)
        artifacts.extend(build_subjective_hook_artifacts(scores=subjective_scores))

        run = EvalRun(
            run_id=run_id,
            case_id=case.case_id,
            scope=case.scope,
            status=run_status,
            started_at=started_at,
            finished_at=finished_at,
            runtime_target=case.runtime_target.graph_id,
            baseline_tags=list(case.baseline.baseline_tags),
            trace_id=trace_id,
            failure=failure,
            metadata={
                "workspace_id": request.workspace_id,
                "story_id": workspace.story_id,
                "setup_step": (
                    request.target_step.value
                    if request.target_step is not None
                    else workspace.current_step.value
                ),
                "target_stage": (
                    request.target_stage.value
                    if request.target_stage is not None
                    else None
                ),
                "model_id": request.model_id,
                "provider_id": request.provider_id,
                "stream_mode": bool(case.runtime_target.stream),
                "diagnostic_profile": (
                    case.input.diagnostic_profile or "setup_outcome_v1"
                ),
            },
        )
        result = EvalRunResult(
            case=case,
            run=run,
            trace=trace,
            artifacts=artifacts,
            scores=scores,
            runtime_result=runtime_result_payload,
        )
        self._finalize_report(result)
        replay_path = self._maybe_save_replay(case=case, result=result)
        if replay_path is not None:
            result.run.metadata["replay_path"] = replay_path.as_posix()
            result.report["replay_path"] = replay_path.as_posix()
        return result

    async def _run_setup_case_via_http(
        self,
        *,
        case: EvalCase,
        request: SetupAgentTurnRequest,
    ) -> dict[str, Any]:
        from api import rp_setup as rp_setup_api
        from main import create_app

        graph_runner = self._factory.build_setup_graph_runner()
        app = create_app()
        app.dependency_overrides[rp_setup_api._setup_graph_runner] = lambda: graph_runner

        runtime_events: list[dict] = []
        failure: EvalFailure | None = None
        runtime_debug: dict[str, Any] | None = None
        path = self._render_runtime_entrypoint(
            entrypoint=case.runtime_target.entrypoint,
            workspace_id=request.workspace_id,
        )

        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(
                    app=app,
                    raise_app_exceptions=False,
                ),
                base_url="http://eval.local",
            ) as client:
                request_payload = request.model_dump(mode="json")
                if case.runtime_target.stream:
                    async with client.stream(
                        "POST",
                        path,
                        json=request_payload,
                        headers={"X-Request-Id": f"eval-{uuid4().hex[:12]}"},
                    ) as response:
                        body_parts: list[str] = []
                        async for chunk in response.aiter_text():
                            body_parts.append(chunk)
                        if response.status_code >= 400:
                            failure = self._classify_setup_http_failure(
                                response=response,
                                runtime_result=graph_runner.last_runtime_result,
                            )
                        for line in "".join(body_parts).splitlines():
                            payload = _parse_typed_payload(line)
                            if (
                                payload is not None
                                and case.trace_hooks.capture_runtime_events
                            ):
                                runtime_events.append(payload)
                else:
                    response = await client.post(
                        path,
                        json=request_payload,
                        headers={"X-Request-Id": f"eval-{uuid4().hex[:12]}"},
                    )
                    if response.status_code >= 400:
                        failure = self._classify_setup_http_failure(
                            response=response,
                            runtime_result=graph_runner.last_runtime_result,
                        )
        finally:
            if case.trace_hooks.capture_graph_debug:
                runtime_debug = graph_runner.get_runtime_debug(
                    workspace_id=request.workspace_id
                )
            app.dependency_overrides.clear()

        return {
            "runtime_result": graph_runner.last_runtime_result,
            "runtime_events": runtime_events,
            "runtime_debug": runtime_debug,
            "failure": failure,
        }

    @staticmethod
    def _render_runtime_entrypoint(*, entrypoint: str, workspace_id: str) -> str:
        try:
            return entrypoint.format(workspace_id=workspace_id)
        except KeyError:
            return entrypoint

    @staticmethod
    def _classify_setup_http_failure(
        *,
        response: httpx.Response,
        runtime_result,
    ) -> EvalFailure:
        if runtime_result is not None and runtime_result.status == "failed":
            return _classify_runtime_failure(runtime_result)

        message = response.text
        code = f"http_{response.status_code}"
        try:
            payload = response.json()
        except ValueError:
            payload = None

        if isinstance(payload, dict):
            detail = payload.get("detail")
            if isinstance(detail, dict):
                error = detail.get("error")
                if isinstance(error, dict):
                    message = str(error.get("message") or message)
                    code = str(error.get("code") or code)

        return EvalFailure(
            layer="infra",
            code=code,
            message=message,
            retryable=response.status_code >= 500,
            source="http_route",
        )

    async def _run_retrieval_case(self, case: EvalCase) -> EvalRunResult:
        run_id = uuid4().hex
        trace_id = uuid4().hex
        started_at = utcnow()
        request_payload = deepcopy(case.input.request)
        workspace = self._prepare_setup_workspace(case=case, request_payload=request_payload)
        workspace_before = self._workspace_service.get_workspace(workspace.workspace_id)

        commit_id = str(
            request_payload.get("commit_id")
            or self._latest_commit_id(workspace_before)
            or ""
        )
        if not commit_id:
            raise ValueError(
                f"Retrieval eval requires a commit_id or accepted commit seed: {case.case_id}"
            )

        ingestion_service = MinimalRetrievalIngestionService(self._session)
        completed_job_ids = ingestion_service.ingest_commit(
            workspace_id=workspace.workspace_id,
            commit_id=commit_id,
        )
        self._workspace_service.refresh_readiness(workspace.workspace_id)
        maintenance_payload: dict | None = None

        if bool(case.input.env_overrides.get("force_stub_embeddings_for_story")):
            self._force_stub_embeddings_for_story(workspace.story_id)

        if bool(case.input.env_overrides.get("create_failed_reindex_jobs_for_story")):
            failure_count = int(
                case.input.env_overrides.get("create_failed_reindex_jobs_count") or 1
            )
            self._create_failed_reindex_jobs_for_story(
                story_id=workspace.story_id,
                count=max(1, failure_count),
            )

        recall_seed_assets = request_payload.get("recall_seed_assets")
        if isinstance(recall_seed_assets, list):
            self._seed_recall_assets_for_retrieval_eval(
                story_id=workspace.story_id,
                workspace_id=workspace.workspace_id,
                commit_id=commit_id,
                assets=recall_seed_assets,
            )

        if bool(case.input.env_overrides.get("run_backfill_story_embeddings")):
            maintenance_service = RetrievalMaintenanceService(self._session)
            backfill_jobs = maintenance_service.backfill_story_embeddings(story_id=workspace.story_id)
            snapshot = maintenance_service.get_story_snapshot(story_id=workspace.story_id)
            maintenance_payload = {
                "operation": "backfill_story_embeddings",
                "jobs": [job.model_dump(mode="json") for job in backfill_jobs],
                "story_snapshot": snapshot.model_dump(mode="json"),
            }

        if bool(case.input.env_overrides.get("run_retry_story_failed_jobs")):
            maintenance_service = RetrievalMaintenanceService(self._session)
            retry_limit = case.input.env_overrides.get("retry_story_failed_jobs_limit")
            retry_batch = maintenance_service.retry_story_failed_jobs(
                story_id=workspace.story_id,
                limit=(int(retry_limit) if retry_limit is not None else None),
            )
            snapshot = maintenance_service.get_story_snapshot(story_id=workspace.story_id)
            maintenance_payload = {
                "operation": "retry_story_failed_jobs",
                "retry_batch": retry_batch.model_dump(mode="json"),
                "story_snapshot": snapshot.model_dump(mode="json"),
            }

        query_input = request_payload.get("query")
        query_result_payload: dict | None = None
        if isinstance(query_input, dict):
            query_model = RetrievalQuery.model_validate(
                {
                    "query_id": query_input.get("query_id") or f"rq_{uuid4().hex[:10]}",
                    "query_kind": query_input.get("query_kind") or "archival",
                    "story_id": query_input.get("story_id") or workspace.story_id,
                    "scope": query_input.get("scope"),
                    "domains": query_input.get("domains") or [],
                    "text_query": query_input.get("text_query"),
                    "filters": query_input.get("filters") or {},
                    "top_k": query_input.get("top_k") or 5,
                    "rerank": bool(query_input.get("rerank", False)),
                    "required_refs": query_input.get("required_refs") or [],
                    "optional_refs": query_input.get("optional_refs") or [],
                }
            )
            retrieval_service = RetrievalService(self._session)
            search_kind = str(query_input.get("search_kind") or "chunks")
            if search_kind == "documents":
                query_result = await retrieval_service.search_documents(query_model)
            elif search_kind == "rag":
                query_result = await retrieval_service.rag_context(query_model)
            else:
                query_result = await retrieval_service.search_chunks(query_model)
            query_result_payload = query_result.model_dump(mode="json")

        finished_at = utcnow()
        retrieval_truth = self._build_retrieval_truth(story_id=workspace.story_id)
        retrieval_result_payload = {
            "finish_reason": "retrieval_completed",
            "workspace_id": workspace.workspace_id,
            "story_id": workspace.story_id,
            "commit_id": commit_id,
            "completed_job_ids": completed_job_ids,
            "query_input": query_input or {},
            "query_result": query_result_payload,
            "maintenance": maintenance_payload or {},
        }
        trace, artifacts = build_retrieval_trace(
            trace_id=trace_id,
            run_id=run_id,
            case_id=case.case_id,
            workspace_id=workspace.workspace_id,
            story_id=workspace.story_id,
            commit_id=commit_id,
            started_at=started_at,
            finished_at=finished_at,
            retrieval_result=retrieval_result_payload,
            retrieval_truth=retrieval_truth,
        )
        scores = evaluate_deterministic_scores(
            case=case,
            run_id=run_id,
            root_span_id=trace.spans[0].span_id if trace.spans else None,
            runtime_result=retrieval_result_payload,
            workspace_before=(
                workspace_before.model_dump(mode="json") if workspace_before else None
            ),
            workspace_after=retrieval_truth,
            graph_debug=None,
            runtime_events=[],
        )
        subjective_scores = await evaluate_subjective_hook_scores(
            case=case,
            run_id=run_id,
            root_span_id=trace.spans[0].span_id if trace.spans else None,
            runtime_result=retrieval_result_payload,
            workspace_before=(
                workspace_before.model_dump(mode="json") if workspace_before else None
            ),
            workspace_after=retrieval_truth,
            graph_debug=None,
            runtime_events=[],
            judge_enabled=bool(case.input.env_overrides.get("enable_subjective_judges")),
            judge_model_id=(
                str(case.input.env_overrides.get("judge_model_id"))
                if case.input.env_overrides.get("judge_model_id")
                else None
            ),
            judge_provider_id=(
                str(case.input.env_overrides.get("judge_provider_id"))
                if case.input.env_overrides.get("judge_provider_id")
                else None
            ),
        )
        scores.extend(subjective_scores)
        artifacts.extend(build_subjective_hook_artifacts(scores=subjective_scores))
        run = EvalRun(
            run_id=run_id,
            case_id=case.case_id,
            scope=case.scope,
            status="completed",
            started_at=started_at,
            finished_at=finished_at,
            runtime_target=case.runtime_target.graph_id,
            baseline_tags=list(case.baseline.baseline_tags),
            trace_id=trace_id,
            metadata={
                "workspace_id": workspace.workspace_id,
                "story_id": workspace.story_id,
                "commit_id": commit_id,
            },
        )
        result = EvalRunResult(
            case=case,
            run=run,
            trace=trace,
            artifacts=artifacts,
            scores=scores,
            runtime_result=retrieval_result_payload,
        )
        result.artifacts.extend(self._maybe_build_ragas_artifacts(case=case, result=result))
        self._finalize_report(result)
        if isinstance(result.report.get("ragas"), dict):
            result.run.metadata["ragas_status"] = result.report["ragas"].get("status")
        replay_path = self._maybe_save_replay(case=case, result=result)
        if replay_path is not None:
            result.run.metadata["replay_path"] = replay_path.as_posix()
            result.report["replay_path"] = replay_path.as_posix()
        return result

    def _seed_recall_assets_for_retrieval_eval(
        self,
        *,
        story_id: str,
        workspace_id: str,
        commit_id: str,
        assets: list[Any],
    ) -> None:
        """Seed explicit Recall fixtures for retrieval policy eval cases."""

        collection = RetrievalCollectionService(self._session).ensure_story_collection(
            story_id=story_id,
            scope="story",
            collection_kind="recall",
        )
        document_service = RetrievalDocumentService(self._session)
        ingestion_service = RetrievalIngestionService(self._session)
        now = utcnow()
        for index, raw_asset in enumerate(assets):
            if not isinstance(raw_asset, dict):
                continue
            metadata = dict(raw_asset.get("metadata") or {})
            asset_id = str(raw_asset.get("asset_id") or f"eval-recall-{index}")
            text = str(raw_asset.get("text") or metadata.get("text") or "")
            if not text:
                continue
            domain = str(metadata.get("domain") or raw_asset.get("domain") or "chapter")
            domain_path = str(
                metadata.get("domain_path")
                or raw_asset.get("domain_path")
                or f"recall.eval.{asset_id}"
            )
            section_metadata = {
                "domain": domain,
                "domain_path": domain_path,
                **metadata,
            }
            document_service.upsert_source_asset(
                SourceAsset(
                    asset_id=asset_id,
                    story_id=story_id,
                    mode=StoryMode.LONGFORM,
                    collection_id=collection.collection_id,
                    workspace_id=workspace_id,
                    commit_id=commit_id,
                    asset_kind=str(
                        metadata.get("materialization_kind") or "recall_eval_fixture"
                    ),
                    source_ref=str(
                        metadata.get("source_ref") or f"eval://recall/{asset_id}"
                    ),
                    title=str(raw_asset.get("title") or asset_id),
                    parse_status="queued",
                    ingestion_status="queued",
                    mapped_targets=["recall"],
                    metadata={
                        **metadata,
                        "seed_sections": [
                            {
                                "section_id": f"seed:{asset_id}",
                                "title": str(raw_asset.get("title") or asset_id),
                                "path": domain_path,
                                "level": 1,
                                "text": text,
                                "metadata": section_metadata,
                            }
                        ],
                    },
                    created_at=now,
                    updated_at=now,
                )
            )
            self._session.flush()
            ingestion_service.ingest_asset(
                story_id=story_id,
                asset_id=asset_id,
                collection_id=collection.collection_id,
            )

    async def _run_activation_case(self, case: EvalCase) -> EvalRunResult:
        run_id = uuid4().hex
        trace_id = uuid4().hex
        started_at = utcnow()
        request_payload = deepcopy(case.input.request)
        workspace = self._prepare_setup_workspace(case=case, request_payload=request_payload)
        workspace_before = self._workspace_service.get_workspace(workspace.workspace_id)
        activation_check: ActivationCheckResult | None = None
        activation_result_model = None
        session_truth: dict = {"session": None, "chapter": None, "snapshot": None}
        failure: EvalFailure | None = None
        run_status = "completed"

        if bool(case.input.env_overrides.get("run_pending_retrieval_ingestion_for_all_commits")):
            ingestion_service = MinimalRetrievalIngestionService(self._session)
            for commit_id in self._list_commit_ids(workspace_before):
                ingestion_service.ingest_commit(
                    workspace_id=workspace.workspace_id,
                    commit_id=commit_id,
                )
            self._workspace_service.refresh_readiness(workspace.workspace_id)

        controller = self._factory.build_setup_runtime_controller()
        activation_check = controller.run_activation_check(workspace_id=workspace.workspace_id)
        activation_runner = self._factory.build_activation_graph_runner()
        workspace_after = self._workspace_service.get_workspace(workspace.workspace_id)
        try:
            activation_result_model = activation_runner.activate_workspace(
                workspace_id=workspace.workspace_id
            )
            workspace_after = self._workspace_service.get_workspace(workspace.workspace_id)
            session_truth = self._build_session_truth(activation_result_model.session_id)
            activation_result_payload = {
                "finish_reason": "activation_completed",
                "activation_check": (
                    activation_check.model_dump(mode="json")
                    if isinstance(activation_check, ActivationCheckResult)
                    else {}
                ),
                "activation_result": activation_result_model.model_dump(mode="json"),
            }
        except Exception as exc:
            run_status = "failed"
            failure = _classify_activation_failure(
                exc=exc,
                activation_check=activation_check,
            )
            workspace_after = self._workspace_service.get_workspace(workspace.workspace_id)
            activation_result_payload = {
                "finish_reason": "activation_failed",
                "activation_check": (
                    activation_check.model_dump(mode="json")
                    if isinstance(activation_check, ActivationCheckResult)
                    else {}
                ),
                "activation_result": {},
                "error": {
                    "message": str(exc),
                    "type": exc.__class__.__name__,
                },
            }
        finished_at = utcnow()
        trace, artifacts = build_activation_trace(
            trace_id=trace_id,
            run_id=run_id,
            case_id=case.case_id,
            workspace_id=workspace.workspace_id,
            started_at=started_at,
            finished_at=finished_at,
            activation_result=activation_result_payload,
            session_truth=session_truth,
        )
        if workspace_before is not None:
            artifacts.append(
                self._artifact(
                    run_id=run_id,
                    kind="workspace_before",
                    name="SetupWorkspaceBefore",
                    payload=workspace_before.model_dump(mode="json"),
                )
            )
        if workspace_after is not None:
            artifacts.append(
                self._artifact(
                    run_id=run_id,
                    kind="workspace_after",
                    name="SetupWorkspaceAfter",
                    payload=workspace_after.model_dump(mode="json"),
                )
            )
        scores = evaluate_deterministic_scores(
            case=case,
            run_id=run_id,
            root_span_id=trace.spans[0].span_id if trace.spans else None,
            runtime_result=activation_result_payload,
            workspace_before=(
                workspace_before.model_dump(mode="json") if workspace_before else None
            ),
            workspace_after=session_truth,
            graph_debug=None,
            runtime_events=[],
        )
        subjective_scores = await evaluate_subjective_hook_scores(
            case=case,
            run_id=run_id,
            root_span_id=trace.spans[0].span_id if trace.spans else None,
            runtime_result=activation_result_payload,
            workspace_before=(
                workspace_before.model_dump(mode="json") if workspace_before else None
            ),
            workspace_after=session_truth,
            graph_debug=None,
            runtime_events=[],
            judge_enabled=bool(case.input.env_overrides.get("enable_subjective_judges")),
            judge_model_id=(
                str(case.input.env_overrides.get("judge_model_id"))
                if case.input.env_overrides.get("judge_model_id")
                else None
            ),
            judge_provider_id=(
                str(case.input.env_overrides.get("judge_provider_id"))
                if case.input.env_overrides.get("judge_provider_id")
                else None
            ),
        )
        scores.extend(subjective_scores)
        artifacts.extend(build_subjective_hook_artifacts(scores=subjective_scores))
        run = EvalRun(
            run_id=run_id,
            case_id=case.case_id,
            scope=case.scope,
            status=run_status,
            started_at=started_at,
            finished_at=finished_at,
            runtime_target=case.runtime_target.graph_id,
            baseline_tags=list(case.baseline.baseline_tags),
            trace_id=trace_id,
            failure=failure,
            metadata={
                "workspace_id": workspace.workspace_id,
                "story_id": workspace.story_id,
                "session_id": (
                    activation_result_model.session_id if activation_result_model is not None else None
                ),
            },
        )
        result = EvalRunResult(
            case=case,
            run=run,
            trace=trace,
            artifacts=artifacts,
            scores=scores,
            runtime_result=activation_result_payload,
        )
        self._finalize_report(result)
        replay_path = self._maybe_save_replay(case=case, result=result)
        if replay_path is not None:
            result.run.metadata["replay_path"] = replay_path.as_posix()
            result.report["replay_path"] = replay_path.as_posix()
        return result

    def _finalize_report(self, result: EvalRunResult) -> None:
        result.report = build_report(result)
        root_span = result.trace.spans[0] if result.trace.spans else None
        diagnostic_scores = evaluate_diagnostic_expectation_scores(
            case=result.case,
            run_id=result.run.run_id,
            root_span_id=root_span.span_id if root_span is not None else None,
            report=result.report,
            run_metadata=result.run.metadata,
            trace_root_attributes=root_span.attributes if root_span is not None else None,
        )
        if diagnostic_scores:
            result.scores.extend(diagnostic_scores)
            attach_diagnostic_expectation_results(result.report, diagnostic_scores)

    def _prepare_setup_workspace(
        self,
        *,
        case: EvalCase,
        request_payload: dict,
    ) -> SetupWorkspace:
        workspace_id = str(request_payload.get("workspace_id") or "")
        workspace = self._workspace_service.get_workspace(workspace_id) if workspace_id else None
        if workspace is None:
            seed = case.input.workspace_seed
            controller = self._factory.build_setup_runtime_controller()
            workspace = controller.create_workspace(
                story_id=str(seed.get("story_id") or case.case_id.replace(".", "-")),
                mode=normalize_story_mode(seed.get("mode")),
            )
        self._apply_setup_seed(workspace_id=workspace.workspace_id, seed=case.input.workspace_seed)
        return self._workspace_service.get_workspace(workspace.workspace_id) or workspace

    def _apply_setup_seed(self, *, workspace_id: str, seed: dict) -> None:
        drafts = dict(seed.get("drafts") or {})
        if "story_config" in drafts:
            self._workspace_service.patch_story_config(
                workspace_id=workspace_id,
                patch=StoryConfigDraft.model_validate(drafts["story_config"]),
            )
        if "writing_contract" in drafts:
            self._workspace_service.patch_writing_contract(
                workspace_id=workspace_id,
                patch=WritingContractDraft.model_validate(drafts["writing_contract"]),
            )
        if "foundation" in drafts:
            foundation_payload = drafts["foundation"]
            entries = foundation_payload.get("entries") or []
            for item in entries:
                self._workspace_service.patch_foundation_entry(
                    workspace_id=workspace_id,
                    entry=FoundationEntry.model_validate(item),
                )
        if "longform_blueprint" in drafts:
            self._workspace_service.patch_longform_blueprint(
                workspace_id=workspace_id,
                patch=LongformBlueprintDraft.model_validate(drafts["longform_blueprint"]),
            )

        current_step = seed.get("current_step")
        if current_step:
            record = self._session.get(SetupWorkspaceRecord, workspace_id)
            if record is None:
                raise ValueError(f"Workspace missing while seeding current_step: {workspace_id}")
            record.current_step = str(current_step)
            self._session.add(record)
            self._session.commit()

        rejected_proposals = seed.get("rejected_proposals") or []
        for item in rejected_proposals:
            proposal = self._workspace_service.propose_commit(
                workspace_id=workspace_id,
                step_id=SetupStepId(str(item["step_id"])),
                target_draft_refs=list(item.get("target_draft_refs") or []),
                reason=item.get("reason"),
            )
            self._workspace_service.reject_commit(
                workspace_id=workspace_id,
                proposal_id=proposal.proposal_id,
            )

        accepted_proposals = seed.get("accepted_proposals") or []
        for item in accepted_proposals:
            proposal = self._workspace_service.propose_commit(
                workspace_id=workspace_id,
                step_id=SetupStepId(str(item["step_id"])),
                target_draft_refs=list(item.get("target_draft_refs") or []),
                reason=item.get("reason"),
            )
            self._workspace_service.accept_commit(
                workspace_id=workspace_id,
                proposal_id=proposal.proposal_id,
            )

        pending_user_edit_deltas = seed.get("pending_user_edit_deltas") or []
        for item in pending_user_edit_deltas:
            self._session.add(
                SetupPendingUserEditDeltaRecord(
                    delta_id=str(item.get("delta_id") or uuid4().hex),
                    workspace_id=workspace_id,
                    step_id=str(item["step_id"]),
                    target_block=str(item["target_block"]),
                    target_ref=str(item["target_ref"]),
                    changes_json=list(item.get("changes") or []),
                    created_at=utcnow(),
                    consumed_at=None,
                )
            )
        if pending_user_edit_deltas:
            self._session.commit()

        cognitive_states = seed.get("cognitive_states") or []
        if isinstance(cognitive_states, dict):
            cognitive_states = [cognitive_states]
        for item in cognitive_states:
            snapshot = SetupCognitiveStateSnapshot.model_validate(item).model_copy(
                update={"workspace_id": workspace_id}
            )
            self._runtime_state_service.save_snapshot(snapshot)

        if drafts or current_step or rejected_proposals or accepted_proposals or pending_user_edit_deltas:
            self._workspace_service.refresh_readiness(workspace_id)
        activated_session = seed.get("activated_session")
        if activated_session:
            self._seed_activated_session(
                workspace_id=workspace_id,
                seed=dict(activated_session),
            )

    def _seed_activated_session(self, *, workspace_id: str, seed: dict) -> None:
        workspace = self._workspace_service.get_workspace(workspace_id)
        if workspace is None:
            raise ValueError(f"Workspace missing while seeding activated session: {workspace_id}")
        if workspace.activated_story_session_id:
            return

        initial_phase = LongformChapterPhase(
            str(seed.get("current_phase") or LongformChapterPhase.OUTLINE_DRAFTING.value)
        )
        story_session_service = StorySessionService(self._session)
        session = story_session_service.create_session(
            story_id=workspace.story_id,
            source_workspace_id=workspace_id,
            mode=workspace.mode.value,
            runtime_story_config=(
                dict(seed.get("runtime_story_config") or {})
                or (
                    workspace.story_config_draft.model_dump(mode="json")
                    if workspace.story_config_draft is not None
                    else {}
                )
            ),
            writer_contract=(
                dict(seed.get("writer_contract") or {})
                or (
                    workspace.writing_contract_draft.model_dump(mode="json")
                    if workspace.writing_contract_draft is not None
                    else {}
                )
            ),
            current_state_json=(
                dict(seed.get("current_state_json") or {})
                or StoryActivationService._initial_current_state(workspace)
            ),
            initial_phase=initial_phase,
        )
        story_session_service.create_chapter_workspace(
            session_id=session.session_id,
            chapter_index=int(seed.get("chapter_index") or 1),
            phase=initial_phase,
            chapter_goal=(
                str(seed.get("chapter_goal"))
                if seed.get("chapter_goal") is not None
                else StoryActivationService._chapter_goal(workspace, chapter_index=1)
            ),
            builder_snapshot_json=(
                dict(seed.get("builder_snapshot_json") or {})
                or StoryActivationService._initial_builder_snapshot(workspace)
            ),
        )
        self._workspace_service.mark_activated_story_session(
            workspace_id=workspace_id,
            session_id=session.session_id,
        )

    def _maybe_save_replay(
        self,
        *,
        case: EvalCase,
        result: EvalRunResult,
    ) -> Path | None:
        env = case.input.env_overrides
        save_replay_path = env.get("save_replay_path")
        save_replay_dir = env.get("save_replay_dir")
        save_on_failure = bool(env.get("save_replay_on_failure"))

        target_path: Path | None = None
        if save_replay_path:
            target_path = Path(str(save_replay_path))
        elif save_replay_dir:
            safe_case_id = case.case_id.replace("/", "_").replace("\\", "_")
            target_path = Path(str(save_replay_dir)) / f"{safe_case_id}--{result.run.run_id}.json"
        elif save_on_failure and result.run.status == "failed":
            target_path = (
                Path("artifacts")
                / "rp-eval-replays"
                / f"{case.case_id.replace('/', '_')}--{result.run.run_id}.json"
            )

        if target_path is None:
            return None
        return save_replay(target_path, result)

    def _maybe_build_ragas_artifacts(
        self,
        *,
        case: EvalCase,
        result: EvalRunResult,
    ) -> list:
        env = case.input.env_overrides
        config = RagasRunConfig(
            enabled=bool(env.get("enable_ragas")),
            metrics=tuple(env.get("ragas_metrics") or RagasRunConfig().metrics),
            response=(
                str(env.get("ragas_response"))
                if env.get("ragas_response") is not None
                else None
            ),
            reference=(
                str(env.get("ragas_reference"))
                if env.get("ragas_reference") is not None
                else None
            ),
            llm_model_id=(
                str(env.get("ragas_llm_model_id"))
                if env.get("ragas_llm_model_id") is not None
                else None
            ),
            llm_provider_id=(
                str(env.get("ragas_llm_provider_id"))
                if env.get("ragas_llm_provider_id") is not None
                else None
            ),
            embedding_model_id=(
                str(env.get("ragas_embedding_model_id"))
                if env.get("ragas_embedding_model_id") is not None
                else None
            ),
            embedding_provider_id=(
                str(env.get("ragas_embedding_provider_id"))
                if env.get("ragas_embedding_provider_id") is not None
                else None
            ),
        )
        generation_config = build_ragas_response_generation_config(env)

        samples = []
        records: list[dict] = []
        error: dict | None = None
        status = "not_requested"
        available = ragas_available()
        runtime_metadata: dict = {}
        generation_metadata: dict = {}

        if config.enabled:
            if not available:
                status = "dependency_missing"
                error = {
                    "type": "RagasUnavailable",
                    "message": "ragas dependency is not installed in the current environment.",
                }
            else:
                try:
                    sample = build_ragas_sample_from_eval_result(
                        result,
                        response=config.response,
                        reference=config.reference,
                    )
                    if generation_config.enabled:
                        if config.response is not None:
                            generation_metadata = {
                                "enabled": True,
                                "status": "skipped_explicit_response",
                            }
                        else:
                            generated_response = generate_eval_rag_response(
                                sample=sample,
                                config=generation_config,
                            )
                            generation_metadata = dict(generated_response.metadata)
                            sample = sample.model_copy(
                                deep=True,
                                update={
                                    "response": generated_response.response,
                                    "metadata": {
                                        **sample.metadata,
                                        "ragas_generation": generation_metadata,
                                    },
                                },
                            )
                    samples = [sample]
                    runtime = resolve_ragas_runtime_bindings(
                        session=self._session,
                        story_id=str(sample.metadata.get("story_id") or ""),
                        env_overrides={
                            **env,
                            "ragas_llm_model_id": config.llm_model_id,
                            "ragas_llm_provider_id": config.llm_provider_id,
                            "ragas_embedding_model_id": config.embedding_model_id,
                            "ragas_embedding_provider_id": config.embedding_provider_id,
                        },
                        metric_names=config.metrics,
                    )
                    runtime_metadata = dict(runtime.metadata)
                    metric_objects = resolve_metric_objects(
                        config.metrics,
                        llm=runtime.llm,
                        embeddings=runtime.embeddings,
                    )
                    raw_result = run_ragas_evaluation(
                        samples=samples,
                        metric_objects=metric_objects,
                        llm=runtime.llm,
                        embeddings=runtime.embeddings,
                    )
                    records = result_to_records(raw_result)
                    status = "completed"
                except Exception as exc:
                    status = "failed"
                    error = {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }

        report = build_ragas_report(
            enabled=config.enabled,
            available=available,
            status=status,
            metric_names=config.metrics,
            samples=samples,
            records=records,
            error=error,
        )
        if runtime_metadata:
            report["runtime"] = runtime_metadata
        if generation_metadata:
            report["generation"] = generation_metadata
        if bool(env.get("sync_ragas_to_langfuse")):
            self._sync_ragas_report_to_langfuse(result=result, report=report)
        return build_ragas_artifacts(
            run_id=result.run.run_id,
            samples=samples,
            records=records,
            report=report,
        )

    def _sync_ragas_report_to_langfuse(
        self,
        *,
        result: EvalRunResult,
        report: dict[str, Any],
    ) -> None:
        langfuse = get_langfuse_service()
        runtime = report.get("runtime")
        if not isinstance(runtime, dict):
            runtime = {}
        story_id = str(
            runtime.get("story_id")
            or result.run.metadata.get("story_id")
            or result.run.metadata.get("session_id")
            or result.run.run_id
        )
        metadata = {
            "scope": result.case.scope,
            "case_id": result.case.case_id,
            "run_id": result.run.run_id,
            "metric_names": list(report.get("metric_names") or []),
        }
        with langfuse.propagate_attributes(
            session_id=story_id,
            tags=["rp", "eval", "retrieval", "ragas"],
            metadata=metadata,
            trace_name="rp.eval.ragas",
        ):
            with langfuse.start_as_current_observation(
                name="rp.eval.ragas",
                as_type="eval",
                input={
                    "case_id": result.case.case_id,
                    "run_id": result.run.run_id,
                    "metric_names": list(report.get("metric_names") or []),
                    "sample_count": int(report.get("sample_count") or 0),
                },
            ) as observation:
                observation.update(
                    output={
                        "status": report.get("status"),
                        "metric_summary": dict(report.get("metric_summary") or {}),
                    }
                )
                emit_ragas_metric_scores(observation, report=report)

    @staticmethod
    def _artifact(*, run_id: str, kind: str, name: str, payload: dict):
        from .models import EvalArtifact

        return EvalArtifact(
            artifact_id=f"{run_id}:artifact:{kind}",
            run_id=run_id,
            kind=kind,
            name=name,
            payload=payload,
        )

    @staticmethod
    def _latest_commit_id(workspace: SetupWorkspace | None) -> str | None:
        if workspace is None or not workspace.accepted_commits:
            return None
        latest = max(workspace.accepted_commits, key=lambda item: item.created_at)
        return latest.commit_id

    @staticmethod
    def _list_commit_ids(workspace: SetupWorkspace | None) -> list[str]:
        if workspace is None:
            return []
        return [item.commit_id for item in workspace.accepted_commits]

    def _build_retrieval_truth(self, *, story_id: str) -> dict:
        source_assets = self._session.exec(
            select(SourceAssetRecord).where(SourceAssetRecord.story_id == story_id)
        ).all()
        parsed_documents = self._session.exec(
            select(ParsedDocumentRecord).where(ParsedDocumentRecord.story_id == story_id)
        ).all()
        chunks = self._session.exec(
            select(KnowledgeChunkRecord).where(KnowledgeChunkRecord.story_id == story_id)
        ).all()
        embeddings = self._session.exec(select(EmbeddingRecordRecord)).all()
        index_jobs = self._session.exec(
            select(IndexJobRecord).where(IndexJobRecord.story_id == story_id)
        ).all()
        chunk_ids = {item.chunk_id for item in chunks}
        asset_ids = {item.asset_id for item in source_assets}
        return {
            "source_assets": [
                {
                    "asset_id": item.asset_id,
                    "collection_id": item.collection_id,
                    "parse_status": item.parse_status,
                    "ingestion_status": item.ingestion_status,
                    "commit_id": item.commit_id,
                }
                for item in source_assets
            ],
            "parsed_documents": [
                {
                    "parsed_document_id": item.parsed_document_id,
                    "asset_id": item.asset_id,
                    "parser_kind": item.parser_kind,
                }
                for item in parsed_documents
                if item.asset_id in asset_ids
            ],
            "chunks": [
                {
                    "chunk_id": item.chunk_id,
                    "asset_id": item.asset_id,
                    "domain": item.domain,
                    "domain_path": item.domain_path,
                    "is_active": item.is_active,
                }
                for item in chunks
            ],
            "embeddings": [
                {
                    "embedding_id": item.embedding_id,
                    "chunk_id": item.chunk_id,
                    "vector_dim": item.vector_dim,
                    "status": item.status,
                    "is_active": item.is_active,
                }
                for item in embeddings
                if item.chunk_id in chunk_ids
            ],
            "index_jobs": [
                {
                    "job_id": item.job_id,
                    "asset_id": item.asset_id,
                    "job_kind": item.job_kind,
                    "job_state": item.job_state,
                    "error_message": item.error_message,
                }
                for item in index_jobs
            ],
        }

    def _force_stub_embeddings_for_story(self, story_id: str) -> None:
        rows = self._session.exec(select(EmbeddingRecordRecord)).all()
        for record in rows:
            chunk = self._session.get(KnowledgeChunkRecord, record.chunk_id) if record.chunk_id else None
            if chunk is None or chunk.story_id != story_id:
                continue
            record.embedding_model = RetrievalIngestionService._STUB_EMBEDDING_MODEL
            record.vector_dim = 0
            record.embedding_vector = None
            self._session.add(record)
        self._session.commit()

    def _create_failed_reindex_jobs_for_story(self, *, story_id: str, count: int) -> None:
        class _InvalidEmbedder(Embedder):
            def __init__(self) -> None:
                super().__init__(fallback_dim=8)

            def embed(self, chunks):
                records = super().embed(chunks)
                self.last_warnings = ["forced_invalid_embedding"]
                return [
                    record.model_copy(update={"vector_dim": 0, "embedding_vector": None})
                    for record in records
                ]

        asset_ids = sorted(
            {
                item.asset_id
                for item in self._session.exec(
                    select(SourceAssetRecord).where(SourceAssetRecord.story_id == story_id)
                ).all()
            }
        )
        if not asset_ids:
            return
        service = RetrievalIngestionService(self._session, embedder=_InvalidEmbedder())
        for _ in range(count):
            service.reindex_asset(story_id=story_id, asset_id=asset_ids[0])
            self._session.commit()

    def _build_session_truth(self, session_id: str) -> dict:
        service = StorySessionService(self._session)
        session = service.get_session(session_id)
        chapter = service.get_current_chapter(session_id)
        snapshot = (
            service.build_chapter_snapshot(
                session_id=session_id,
                chapter_index=session.current_chapter_index,
            )
            if session is not None
            else None
        )
        session_count_for_story = (
            len(
                self._session.exec(
                    select(StorySessionRecord).where(StorySessionRecord.story_id == session.story_id)
                ).all()
            )
            if session is not None
            else 0
        )
        chapter_count_for_session = (
            len(
                self._session.exec(
                    select(ChapterWorkspaceRecord).where(
                        ChapterWorkspaceRecord.session_id == session_id
                    )
                ).all()
            )
            if session is not None
            else 0
        )
        return {
            "session": session.model_dump(mode="json") if session is not None else None,
            "chapter": chapter.model_dump(mode="json") if chapter is not None else None,
            "snapshot": snapshot.model_dump(mode="json") if snapshot is not None else None,
            "session_count_for_story": session_count_for_story,
            "chapter_count_for_session": chapter_count_for_session,
        }


def _parse_typed_payload(line: str) -> dict | None:
    stripped = line.strip()
    if not stripped.startswith("data: "):
        return None
    payload = stripped[6:]
    if not payload or payload == "[DONE]":
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def _classify_runtime_failure(runtime_result) -> EvalFailure:
    finish_reason = str(runtime_result.finish_reason)
    layer = "agent"
    if finish_reason in {"runtime_execution_failed", "upstream_error"}:
        layer = "infra"
    return EvalFailure(
        layer=layer,
        code=finish_reason,
        message=str(runtime_result.error or finish_reason),
        retryable=layer == "infra",
        source="runtime_result",
    )


def _classify_exception_failure(exc: Exception, runtime_result) -> EvalFailure:
    if runtime_result is not None and runtime_result.status == "failed":
        return _classify_runtime_failure(runtime_result)
    return EvalFailure(
        layer="infra",
        code=exc.__class__.__name__,
        message=str(exc),
        retryable=False,
        source="runner_exception",
    )


def _classify_activation_failure(
    *,
    exc: Exception,
    activation_check: ActivationCheckResult | None,
) -> EvalFailure:
    if isinstance(activation_check, ActivationCheckResult) and not activation_check.ready:
        return EvalFailure(
            layer="deterministic",
            code="activation_not_ready",
            message=str(exc),
            retryable=False,
            source="activation_check",
        )
    return EvalFailure(
        layer="infra",
        code=exc.__class__.__name__,
        message=str(exc),
        retryable=False,
        source="activation_runner",
    )
