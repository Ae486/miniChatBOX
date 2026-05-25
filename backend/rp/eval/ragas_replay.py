"""Run RAGAS over replay payloads, including sampled retrieval traces."""

from __future__ import annotations

from typing import Any

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
from .ragas_reporting import build_ragas_report
from .ragas_runtime import resolve_ragas_runtime_bindings
from .ragas_samples import build_ragas_sample_from_replay_payload


def run_ragas_on_replay_payload(
    *,
    session,
    replay_payload: dict[str, Any],
    env_overrides: dict[str, Any],
) -> dict[str, Any]:
    config = RagasRunConfig(
        enabled=bool(env_overrides.get("enable_ragas", True)),
        metrics=tuple(env_overrides.get("ragas_metrics") or RagasRunConfig().metrics),
        response=(
            str(env_overrides.get("ragas_response"))
            if env_overrides.get("ragas_response") is not None
            else None
        ),
        reference=(
            str(env_overrides.get("ragas_reference"))
            if env_overrides.get("ragas_reference") is not None
            else None
        ),
        llm_model_id=(
            str(env_overrides.get("ragas_llm_model_id"))
            if env_overrides.get("ragas_llm_model_id") is not None
            else None
        ),
        llm_provider_id=(
            str(env_overrides.get("ragas_llm_provider_id"))
            if env_overrides.get("ragas_llm_provider_id") is not None
            else None
        ),
        embedding_model_id=(
            str(env_overrides.get("ragas_embedding_model_id"))
            if env_overrides.get("ragas_embedding_model_id") is not None
            else None
        ),
        embedding_provider_id=(
            str(env_overrides.get("ragas_embedding_provider_id"))
            if env_overrides.get("ragas_embedding_provider_id") is not None
            else None
        ),
    )
    generation_config = build_ragas_response_generation_config(env_overrides)
    samples = []
    records: list[dict[str, Any]] = []
    error: dict[str, Any] | None = None
    runtime_metadata: dict[str, Any] = {}
    generation_metadata: dict[str, Any] = {}
    available = ragas_available()

    if not config.enabled:
        status = "not_requested"
    elif not available:
        status = "dependency_missing"
        error = {
            "type": "missing_dependency",
            "message": "ragas dependency is not installed in the current environment.",
        }
    else:
        try:
            sample = build_ragas_sample_from_replay_payload(
                replay_payload,
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
                session=session,
                story_id=str(sample.metadata.get("story_id") or ""),
                env_overrides={
                    **env_overrides,
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
    return report


def attach_ragas_report_to_replay(
    *,
    replay_payload: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, Any]:
    updated = dict(replay_payload)
    existing_report = updated.get("report")
    if not isinstance(existing_report, dict):
        existing_report = {}
    updated["report"] = {
        **existing_report,
        "ragas": report,
    }
    sampled_trace = updated.get("sampled_trace")
    if isinstance(sampled_trace, dict):
        updated["sampled_trace"] = {
            **sampled_trace,
            "ragas_report": report,
        }
    return updated
