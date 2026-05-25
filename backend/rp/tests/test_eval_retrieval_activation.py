from __future__ import annotations

from pathlib import Path

import pytest

from rp.eval.case_loader import load_case
from rp.eval.ragas_generation import RagasGeneratedResponse
from rp.eval.runner import EvalRunner


def _case_path(*parts: str) -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "eval"
        / "cases"
        / Path(*parts)
    )


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
        self._sink.append({"kind": "observation_update", "name": self._name, "payload": kwargs})

    def score_trace(self, **kwargs):
        self._sink.append({"kind": "score_trace", "name": self._name, "payload": kwargs})


class _FakeLangfuseContext:
    def __init__(self, *, sink: list[dict], payload: dict) -> None:
        self._sink = sink
        self._payload = payload

    def __enter__(self):
        self._sink.append({"kind": "propagate_enter", "payload": self._payload})
        return self

    def __exit__(self, exc_type, exc, tb):
        self._sink.append({"kind": "propagate_exit", "payload": self._payload})
        return False


class _FakeLangfuseService:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def propagate_attributes(self, **kwargs):
        return _FakeLangfuseContext(sink=self.events, payload=kwargs)

    def start_as_current_observation(self, **kwargs):
        return _FakeLangfuseObservation(
            sink=self.events,
            name=str(kwargs.get("name") or "unknown"),
        )


@pytest.mark.asyncio
async def test_eval_runner_supports_retrieval_case_from_file(retrieval_session):
    case = load_case(
        "H:/chatboxapp/backend/rp/eval/cases/retrieval/ingestion/commit_ingestion_and_query.v1.json"
    )
    result = await EvalRunner(retrieval_session).run_case(case)

    assert result.run.status == "completed"
    assert result.report["finish_reason"] == "retrieval_completed"
    assert result.report["assertion_summary"]["fail"] == 0
    assert result.report["ragas"]["status"] == "not_requested"
    retrieval_truth = next(item for item in result.artifacts if item.kind == "retrieval_truth")
    assert retrieval_truth.payload["chunks"]


@pytest.mark.asyncio
async def test_eval_runner_supports_retrieval_trace_and_provenance_case(retrieval_session):
    case = load_case(_case_path("retrieval", "search", "query_trace_and_provenance.v1.json"))
    result = await EvalRunner(retrieval_session).run_case(case)

    assert result.run.status == "completed"
    assert result.report["finish_reason"] == "retrieval_completed"
    assert result.report["assertion_summary"]["fail"] == 0
    assert result.report["ragas"]["status"] == "not_requested"
    retrieval_result = next(item for item in result.artifacts if item.kind == "retrieval_result")
    assert retrieval_result.payload["query_result"]["trace"]["pipeline_stages"]
    assert retrieval_result.payload["query_result"]["hits"][0]["provenance_refs"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case_relpath",
    [
        ("retrieval", "search", "archival_source_filter_policy.v1.json"),
        ("retrieval", "search", "recall_scene_policy.v1.json"),
        ("retrieval", "search", "active_foreshadow_policy.v1.json"),
        ("retrieval", "search", "branch_canon_isolation_policy.v1.json"),
        ("retrieval", "search", "rag_context_budget_policy.v1.json"),
    ],
)
async def test_eval_runner_supports_retrieval_policy_cases(
    retrieval_session,
    case_relpath,
):
    case = load_case(_case_path(*case_relpath))
    result = await EvalRunner(retrieval_session).run_case(case)

    assert result.run.status == "completed"
    assert result.report["finish_reason"] == "retrieval_completed"
    assert result.report["assertion_summary"]["fail"] == 0


@pytest.mark.asyncio
async def test_eval_runner_marks_ragas_dependency_missing_when_enabled(
    retrieval_session,
    monkeypatch,
):
    case = load_case(_case_path("retrieval", "search", "query_trace_and_provenance.v1.json"))
    case = case.model_copy(
        deep=True,
        update={
            "input": case.input.model_copy(
                deep=True,
                update={
                    "env_overrides": {
                        **case.input.env_overrides,
                        "enable_ragas": True,
                    }
                },
            )
        },
    )
    monkeypatch.setattr("rp.eval.runner.ragas_available", lambda: False)

    result = await EvalRunner(retrieval_session).run_case(case)

    assert result.report["ragas"]["status"] == "dependency_missing"
    assert result.report["ragas"]["enabled"] is True
    assert result.run.metadata["ragas_status"] == "dependency_missing"


@pytest.mark.asyncio
async def test_eval_runner_records_completed_ragas_metrics_when_execution_succeeds(
    retrieval_session,
    monkeypatch,
):
    case = load_case(_case_path("retrieval", "search", "query_trace_and_provenance.v1.json"))
    case = case.model_copy(
        deep=True,
        update={
            "input": case.input.model_copy(
                deep=True,
                update={
                    "env_overrides": {
                        **case.input.env_overrides,
                        "enable_ragas": True,
                        "ragas_llm_model_id": "model-judge",
                        "ragas_llm_provider_id": "provider-judge",
                    }
                },
            )
        },
    )
    monkeypatch.setattr("rp.eval.runner.ragas_available", lambda: True)
    monkeypatch.setattr(
        "rp.eval.runner.resolve_metric_objects",
        lambda metrics, **kwargs: ["metric"],
    )
    monkeypatch.setattr(
        "rp.eval.runner.resolve_ragas_runtime_bindings",
        lambda **kwargs: type(
            "_Runtime",
            (),
            {
                "llm": object(),
                "embeddings": None,
                "metadata": {"llm": {"model_id": "model-judge"}},
            },
        )(),
    )
    monkeypatch.setattr(
        "rp.eval.runner.run_ragas_evaluation",
        lambda **kwargs: [{"context_precision": 0.88}],
    )

    result = await EvalRunner(retrieval_session).run_case(case)

    assert result.report["ragas"]["status"] == "completed"
    assert result.report["ragas"]["metric_summary"]["context_precision"] == 0.88
    assert result.report["ragas"]["runtime"]["llm"]["model_id"] == "model-judge"


@pytest.mark.asyncio
async def test_eval_runner_generates_eval_only_ragas_response_when_enabled(
    retrieval_session,
    monkeypatch,
):
    case = load_case(_case_path("retrieval", "search", "query_trace_and_provenance.v1.json"))
    case = case.model_copy(
        deep=True,
        update={
            "input": case.input.model_copy(
                deep=True,
                update={
                    "env_overrides": {
                        **case.input.env_overrides,
                        "enable_ragas": True,
                        "ragas_generate_response": True,
                        "ragas_generator_model_id": "model-generator",
                        "ragas_generator_provider_id": "provider-generator",
                        "ragas_llm_model_id": "model-judge",
                        "ragas_llm_provider_id": "provider-judge",
                    }
                },
            )
        },
    )
    monkeypatch.setattr("rp.eval.runner.ragas_available", lambda: True)
    monkeypatch.setattr(
        "rp.eval.runner.generate_eval_rag_response",
        lambda *, sample, config: RagasGeneratedResponse(
            response="Generated answer grounded in retrieved contexts.",
            metadata={
                "enabled": True,
                "status": "completed",
                "model_id": config.model_id,
                "provider_id": config.provider_id,
            },
        ),
    )
    monkeypatch.setattr(
        "rp.eval.runner.resolve_metric_objects",
        lambda metrics, **kwargs: ["metric"],
    )
    monkeypatch.setattr(
        "rp.eval.runner.resolve_ragas_runtime_bindings",
        lambda **kwargs: type(
            "_Runtime",
            (),
            {
                "llm": object(),
                "embeddings": None,
                "metadata": {"llm": {"model_id": "model-judge"}},
            },
        )(),
    )

    def _fake_ragas_evaluation(**kwargs):
        samples = kwargs["samples"]
        assert samples[0].response == "Generated answer grounded in retrieved contexts."
        return [{"faithfulness": 0.91}]

    monkeypatch.setattr(
        "rp.eval.runner.run_ragas_evaluation",
        _fake_ragas_evaluation,
    )

    result = await EvalRunner(retrieval_session).run_case(case)

    assert result.report["ragas"]["status"] == "completed"
    assert result.report["ragas"]["generation"]["status"] == "completed"
    assert result.report["ragas"]["generation"]["model_id"] == "model-generator"
    assert result.report["ragas"]["metric_summary"]["faithfulness"] == 0.91


@pytest.mark.asyncio
async def test_eval_runner_can_sync_ragas_metrics_to_langfuse(
    retrieval_session,
    monkeypatch,
):
    fake_langfuse = _FakeLangfuseService()
    case = load_case(_case_path("retrieval", "search", "query_trace_and_provenance.v1.json"))
    case = case.model_copy(
        deep=True,
        update={
            "input": case.input.model_copy(
                deep=True,
                update={
                    "env_overrides": {
                        **case.input.env_overrides,
                        "enable_ragas": True,
                        "sync_ragas_to_langfuse": True,
                        "ragas_llm_model_id": "model-judge",
                        "ragas_llm_provider_id": "provider-judge",
                    }
                },
            )
        },
    )
    monkeypatch.setattr("rp.eval.runner.ragas_available", lambda: True)
    monkeypatch.setattr(
        "rp.eval.runner.resolve_metric_objects",
        lambda metrics, **kwargs: ["metric"],
    )
    monkeypatch.setattr(
        "rp.eval.runner.resolve_ragas_runtime_bindings",
        lambda **kwargs: type(
            "_Runtime",
            (),
            {
                "llm": object(),
                "embeddings": None,
                "metadata": {"llm": {"model_id": "model-judge"}, "story_id": "story-ragas"},
            },
        )(),
    )
    monkeypatch.setattr(
        "rp.eval.runner.run_ragas_evaluation",
        lambda **kwargs: [{"context_precision": 0.88}],
    )
    monkeypatch.setattr(
        "rp.eval.runner.get_langfuse_service",
        lambda: fake_langfuse,
    )

    result = await EvalRunner(retrieval_session).run_case(case)

    assert result.report["ragas"]["status"] == "completed"
    assert any(
        item["kind"] == "score_trace"
        and item["payload"]["name"] == "retrieval.ragas.context_precision"
        and item["payload"]["value"] == 0.88
        for item in fake_langfuse.events
    )
    assert any(
        item["kind"] == "score_trace"
        and item["payload"]["name"] == "retrieval.ragas.status"
        and item["payload"]["value"] == "completed"
        for item in fake_langfuse.events
    )


@pytest.mark.asyncio
async def test_eval_runner_surfaces_failed_retrieval_ingestion_jobs(
    retrieval_session,
    monkeypatch,
):
    monkeypatch.setattr(
        "rp.services.retrieval_ingestion_service.Chunker.chunk",
        lambda self, document, **kwargs: [],
    )

    case = load_case(
        _case_path("retrieval", "ingestion", "commit_ingestion_no_chunks_failed.v1.json")
    )
    result = await EvalRunner(retrieval_session).run_case(case)

    assert result.run.status == "completed"
    assert result.report["finish_reason"] == "retrieval_completed"
    assert result.report["assertion_summary"]["fail"] == 0
    retrieval_truth = next(item for item in result.artifacts if item.kind == "retrieval_truth")
    assert retrieval_truth.payload["index_jobs"][0]["job_state"] == "failed"
    assert retrieval_truth.payload["source_assets"][0]["ingestion_status"] == "failed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case_relpath", "expected_operation"),
    [
        (
            ("retrieval", "maintenance", "story_backfill_stub_embeddings.v1.json"),
            "backfill_story_embeddings",
        ),
        (
            ("retrieval", "maintenance", "story_retry_failed_jobs.v1.json"),
            "retry_story_failed_jobs",
        ),
    ],
)
async def test_eval_runner_supports_retrieval_maintenance_cases(
    retrieval_session,
    case_relpath,
    expected_operation,
):
    case = load_case(_case_path(*case_relpath))
    result = await EvalRunner(retrieval_session).run_case(case)

    assert result.run.status == "completed"
    assert result.report["finish_reason"] == "retrieval_completed"
    assert result.report["assertion_summary"]["fail"] == 0
    retrieval_result = next(item for item in result.artifacts if item.kind == "retrieval_result")
    assert retrieval_result.payload["maintenance"]["operation"] == expected_operation


@pytest.mark.asyncio
async def test_eval_runner_supports_activation_case_from_file(retrieval_session):
    case = load_case(
        "H:/chatboxapp/backend/rp/eval/cases/activation/bootstrap/ready_workspace_activation.v1.json"
    )
    result = await EvalRunner(retrieval_session).run_case(case)

    assert result.run.status == "completed"
    assert result.report["finish_reason"] == "activation_completed"
    assert result.report["assertion_summary"]["fail"] == 0
    session_truth = next(item for item in result.artifacts if item.kind == "session_truth")
    assert session_truth.payload["session"]["session_state"] == "active"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case_relpath", "expected_status", "expected_finish_reason", "expected_failure_layer"),
    [
        (
            ("activation", "gate", "unfinished_ingestion_blocks_activation.v1.json"),
            "failed",
            "activation_failed",
            "deterministic",
        ),
        (
            ("activation", "gate", "missing_blueprint_blocks_activation.v1.json"),
            "failed",
            "activation_failed",
            "deterministic",
        ),
        (
            ("activation", "bootstrap", "idempotent_existing_session_activation.v1.json"),
            "completed",
            "activation_completed",
            None,
        ),
        (
            ("activation", "bootstrap", "bootstrap_seed_complete.v1.json"),
            "completed",
            "activation_completed",
            None,
        ),
    ],
)
async def test_eval_runner_supports_extended_activation_cases(
    retrieval_session,
    case_relpath,
    expected_status,
    expected_finish_reason,
    expected_failure_layer,
):
    case = load_case(_case_path(*case_relpath))
    result = await EvalRunner(retrieval_session).run_case(case)

    assert result.run.status == expected_status
    assert result.report["finish_reason"] == expected_finish_reason
    assert result.report["failure_layer"] == expected_failure_layer
    assert result.report["assertion_summary"]["fail"] == 0

    if "idempotent_existing_session_activation" in case.case_id:
        session_truth = next(item for item in result.artifacts if item.kind == "session_truth")
        assert session_truth.payload["session_count_for_story"] == 1
        assert session_truth.payload["chapter_count_for_session"] == 1

    if "bootstrap_seed_complete" in case.case_id:
        session_truth = next(item for item in result.artifacts if item.kind == "session_truth")
        assert session_truth.payload["chapter"]["builder_snapshot_json"]["foundation_digest"]
        assert session_truth.payload["chapter"]["builder_snapshot_json"]["blueprint_digest"]
