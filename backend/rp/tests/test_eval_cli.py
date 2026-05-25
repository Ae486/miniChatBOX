from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path

import pytest

from models.model_registry import ModelRegistryEntry
from models.provider_registry import ProviderRegistryEntry
from rp.eval.case_loader import load_case
from rp.eval.cli import _apply_cli_env_overrides, _print_payload, build_parser, main
from rp.eval.ragas_generation import RagasGeneratedResponse
from services.model_registry import get_model_registry_service
from services.provider_registry import get_provider_registry_service
import services.model_registry as model_registry_module


def _case_path(*parts: str) -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "eval"
        / "cases"
        / Path(*parts)
    )


def _seed_judge_model_registry() -> None:
    get_provider_registry_service.cache_clear()
    model_registry_module._model_registry_service = None
    provider_service = get_provider_registry_service()
    provider_service.upsert_entry(
        ProviderRegistryEntry(
            id="provider-eval",
            name="Eval Provider",
            type="openai",
            api_key="sk-test",
            api_url="https://example.com/v1",
            custom_headers={},
            is_enabled=True,
        )
    )
    model_service = get_model_registry_service()
    model_service.upsert_entry(
        ModelRegistryEntry(
            id="model-eval",
            provider_id="provider-eval",
            model_name="gpt-4o-mini",
            display_name="Eval Model",
            capabilities=["text", "tool"],
            is_enabled=True,
        )
    )


class _FakeJudgeLLMService:
    def __init__(self, *, response_payload: dict) -> None:
        self._response_payload = response_payload

    async def chat_completion(self, request):
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(self._response_payload),
                    }
                }
            ]
        }

    def chat_completion_stream(self, request):
        raise AssertionError("stream path not expected")


class _FakeLangfuseService:
    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.flush_calls = 0

    def flush(self) -> None:
        self.flush_calls += 1


def test_eval_cli_run_suite_writes_bundle_and_passes_thresholds(
    retrieval_session,
    monkeypatch,
    tmp_path,
    capsys,
):
    monkeypatch.setattr(
        "rp.eval.cli._open_eval_session",
        lambda: nullcontext(retrieval_session),
    )

    exit_code = main(
        [
            "run-suite",
            str(_case_path("retrieval", "ingestion", "commit_ingestion_and_query.v1.json")),
            "--output-dir",
            str(tmp_path / "suite-output"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "threshold_passed=True" in captured.out
    assert "top_reason_codes=" in captured.out
    assert "top_primary_suspects=" in captured.out
    assert (tmp_path / "suite-output" / "suite-summary.json").exists()
    assert (tmp_path / "suite-output" / "derived-summary.json").exists()
    assert (tmp_path / "suite-output" / "thresholds.json").exists()
    assert (tmp_path / "suite-output" / "summary.md").exists()


def test_eval_cli_flushes_langfuse_before_exit(tmp_path, monkeypatch, capsys):
    fake_langfuse = _FakeLangfuseService()
    monkeypatch.setattr("rp.eval.cli.get_langfuse_service", lambda: fake_langfuse)

    sample_input = tmp_path / "sample.json"
    sample_input.write_text(
        json.dumps(
            {
                "query": "who holds the archive ledger",
                "retrieved_contexts": ["The archive ledger is sealed before sunrise."],
                "response": "The archive ledger is sealed before sunrise.",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "sampled-replay",
            str(sample_input),
            str(tmp_path / "replay.json"),
        ]
    )

    capsys.readouterr()
    assert exit_code == 0
    assert fake_langfuse.flush_calls == 1


def test_eval_cli_run_case_repeat_count_writes_analysis_bundle(
    retrieval_session,
    monkeypatch,
    tmp_path,
    capsys,
):
    monkeypatch.setattr(
        "rp.eval.cli._open_eval_session",
        lambda: nullcontext(retrieval_session),
    )

    exit_code = main(
        [
            "run-case",
            str(_case_path("retrieval", "ingestion", "commit_ingestion_and_query.v1.json")),
            "--output-dir",
            str(tmp_path / "repeat-output"),
            "--repeat-count",
            "2",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "repeat_cases=1" in captured.out
    payload = json.loads((tmp_path / "repeat-output" / "suite-summary.json").read_text(encoding="utf-8"))
    assert payload["run_count"] == 2
    assert payload["items"][0]["attempt_index"] == 1
    assert (tmp_path / "repeat-output" / "derived-summary.json").exists()
    assert (tmp_path / "repeat-output" / "thresholds.json").exists()
    assert (tmp_path / "repeat-output" / "summary.md").exists()


def test_eval_cli_run_case_can_be_repeated_without_story_collision(
    retrieval_session,
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(
        "rp.eval.cli._open_eval_session",
        lambda: nullcontext(retrieval_session),
    )

    first_exit_code = main(
        [
            "run-case",
            str(_case_path("retrieval", "search", "query_trace_and_provenance.v1.json")),
        ]
    )
    capsys.readouterr()
    second_exit_code = main(
        [
            "run-case",
            str(_case_path("retrieval", "search", "query_trace_and_provenance.v1.json")),
        ]
    )
    capsys.readouterr()

    assert first_exit_code == 0
    assert second_exit_code == 0


def test_eval_cli_applies_retrieval_runtime_overrides():
    case = load_case(str(_case_path("retrieval", "search", "query_trace_and_provenance.v1.json")))
    parser = build_parser()
    args = parser.parse_args(
        [
            "run-case",
            str(_case_path("retrieval", "search", "query_trace_and_provenance.v1.json")),
            "--retrieval-embedding-model-id",
            "embed-model-id",
            "--retrieval-embedding-provider-id",
            "embed-provider-id",
            "--retrieval-rerank-model-id",
            "rerank-model-id",
            "--retrieval-rerank-provider-id",
            "rerank-provider-id",
        ]
    )

    updated = _apply_cli_env_overrides(case, args)

    assert updated.input.env_overrides["retrieval_embedding_model_id"] == "embed-model-id"
    assert updated.input.env_overrides["retrieval_embedding_provider_id"] == "embed-provider-id"
    assert updated.input.env_overrides["retrieval_rerank_model_id"] == "rerank-model-id"
    assert updated.input.env_overrides["retrieval_rerank_provider_id"] == "rerank-provider-id"


def test_eval_cli_applies_sync_ragas_to_langfuse_override():
    case = load_case(str(_case_path("retrieval", "search", "query_trace_and_provenance.v1.json")))
    parser = build_parser()
    args = parser.parse_args(
        [
            "run-case",
            str(_case_path("retrieval", "search", "query_trace_and_provenance.v1.json")),
            "--sync-ragas-to-langfuse",
        ]
    )

    updated = _apply_cli_env_overrides(case, args)

    assert updated.input.env_overrides["sync_ragas_to_langfuse"] is True


def test_eval_cli_parser_supports_langfuse_suite_compare_and_replay_sync_flags():
    parser = build_parser()

    suite_args = parser.parse_args(
        [
            "run-suite",
            str(_case_path("retrieval", "search", "query_trace_and_provenance.v1.json")),
            "--sync-suite-to-langfuse",
        ]
    )
    suite_all_args = parser.parse_args(
        [
            "run-suite",
            str(_case_path("retrieval", "search", "query_trace_and_provenance.v1.json")),
            "--sync-all-to-langfuse",
        ]
    )
    compare_args = parser.parse_args(
        [
            "compare",
            "current.json",
            "baseline.json",
            "--sync-comparison-to-langfuse",
        ]
    )
    replay_args = parser.parse_args(
        [
            "replay",
            "replay.json",
            "--sync-replay-to-langfuse",
        ]
    )
    replay_ragas_args = parser.parse_args(
        [
            "replay",
            "replay.json",
            "--run-ragas",
            "--write-back",
        ]
    )
    sampled_replay_args = parser.parse_args(
        [
            "sampled-replay",
            "sample.json",
            "replay.json",
        ]
    )

    assert suite_args.sync_suite_to_langfuse is True
    assert suite_all_args.sync_all_to_langfuse is True
    assert compare_args.sync_comparison_to_langfuse is True
    assert replay_args.sync_replay_to_langfuse is True
    assert replay_ragas_args.run_ragas is True
    assert replay_ragas_args.write_back is True
    assert sampled_replay_args.command == "sampled-replay"


def test_eval_cli_compare_reports_changed_cases(tmp_path, capsys):
    baseline_dir = tmp_path / "baseline"
    current_dir = tmp_path / "current"
    baseline_dir.mkdir()
    current_dir.mkdir()

    (baseline_dir / "suite-summary.json").write_text(
        json.dumps(
            {
                "suite_id": "baseline",
                "items": [
                    {
                        "case_id": "setup.case.a",
                        "run_id": "baseline-1",
                        "report": {
                            "case_id": "setup.case.a",
                            "status": "completed",
                            "finish_reason": "completed_text",
                            "failure_layer": None,
                            "hard_failures": [],
                            "assertion_summary": {"pass": 1, "fail": 0, "warn": 0, "skip": 0},
                            "reason_codes": ["prompt.missing_step_targeting"],
                            "primary_suspects": ["instruction_prompt_skill"],
                            "outcome_chain": {"transcript_status": "pass"},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (current_dir / "suite-summary.json").write_text(
        json.dumps(
            {
                "suite_id": "current",
                "items": [
                    {
                        "case_id": "setup.case.a",
                        "run_id": "current-1",
                        "report": {
                            "case_id": "setup.case.a",
                            "status": "completed",
                            "finish_reason": "awaiting_user_input",
                            "failure_layer": None,
                            "hard_failures": [
                                "finish_reason_changed",
                                "diagnostic.reason_code_presence",
                            ],
                            "assertion_summary": {"pass": 0, "fail": 2, "warn": 0, "skip": 0},
                            "reason_codes": [
                                "prompt.missing_step_targeting",
                                "controller.commit_proposal_blocked",
                            ],
                            "primary_suspects": ["decision_policy"],
                            "outcome_chain": {"transcript_status": "warn"},
                            "recommended_next_action": "tighten_commit_readiness_checks_and_review_block_messages",
                            "diagnostic_expectation_results": [
                                {
                                    "score_name": "diagnostic.reason_code_presence",
                                    "status": "fail",
                                    "expected": ["readiness.blocked_by_open_setup_prerequisites"],
                                    "actual": ["prompt.missing_step_targeting"],
                                }
                            ],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["compare", str(current_dir), str(baseline_dir), "--json"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["changed_cases"][0]["case_id"] == "setup.case.a"
    assert payload["drift_summary"]["changed_finish_reason_case_ids"] == ["setup.case.a"]
    assert payload["drift_summary"]["changed_reason_code_case_ids"] == ["setup.case.a"]
    assert payload["drift_summary"]["changed_primary_suspect_case_ids"] == ["setup.case.a"]
    assert payload["drift_summary"]["changed_outcome_chain_case_ids"] == ["setup.case.a"]
    assert payload["drift_summary"]["changed_recommended_next_action_case_ids"] == [
        "setup.case.a"
    ]
    assert payload["drift_summary"]["changed_diagnostic_expectation_case_ids"] == [
        "setup.case.a"
    ]


def test_eval_cli_print_payload_includes_diagnostic_context(capsys):
    _print_payload(
        {
            "case_id": "setup.case.summary",
            "status": "completed",
            "report": {
                "case_id": "setup.case.summary",
                "status": "completed",
                "finish_reason": "awaiting_user_input",
                "failure_layer": "deterministic",
                "assertion_summary": {"pass": 3, "fail": 1, "warn": 0, "skip": 0},
                "reason_codes": ["prompt.missing_step_targeting"],
                "primary_suspects": ["instruction_prompt_skill"],
                "recommended_next_action": "ask_for_missing_setup_inputs",
                "outcome_chain_display": ["transcript=warn", "readiness=fail"],
                "evidence_refs": ["artifact:tool_sequence", "artifact:readiness_snapshot"],
            },
        },
        as_json=False,
    )
    captured = capsys.readouterr()

    assert "failure_layer=deterministic" in captured.out
    assert "reason_codes=prompt.missing_step_targeting" in captured.out
    assert "primary_suspects=instruction_prompt_skill" in captured.out
    assert "next_action=ask_for_missing_setup_inputs" in captured.out
    assert "outcome_chain=transcript=warn;readiness=fail" in captured.out
    assert "evidence_refs=artifact:tool_sequence,artifact:readiness_snapshot" in captured.out


def test_eval_cli_print_payload_includes_langfuse_sync_summary(capsys):
    _print_payload(
        {
            "suite": {"items": []},
            "summary": {
                "run_count": 1,
                "case_count": 1,
                "failed_run_count": 0,
                "repeat_case_ids": [],
                "assertion_fail_total": 0,
                "assertion_warn_total": 0,
                "executed_judge_hook_total": 0,
                "pending_judge_hook_total": 0,
                "diagnostic_summary": {},
            },
            "thresholds": {"passed": True},
            "langfuse_sync": {
                "suite_replay_sync_count": 2,
                "comparison_synced": True,
            },
        },
        as_json=False,
    )
    captured = capsys.readouterr()

    assert "langfuse_replays_synced=2" in captured.out
    assert "langfuse_comparison_synced=True" in captured.out


def test_eval_cli_run_suite_writes_comparison_markdown(
    retrieval_session,
    monkeypatch,
    tmp_path,
    capsys,
):
    monkeypatch.setattr(
        "rp.eval.cli._open_eval_session",
        lambda: nullcontext(retrieval_session),
    )

    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    (baseline_dir / "suite-summary.json").write_text(
        json.dumps(
            {
                "suite_id": "baseline",
                "items": [
                    {
                        "case_id": "retrieval.ingestion.commit_ingestion_and_query.v1",
                        "run_id": "baseline-1",
                        "report": {
                            "case_id": "retrieval.ingestion.commit_ingestion_and_query.v1",
                            "status": "completed",
                            "finish_reason": "retrieval_completed",
                            "failure_layer": None,
                            "hard_failures": ["baseline-hard-failure"],
                            "assertion_summary": {"pass": 4, "fail": 1, "warn": 0, "skip": 0},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "run-suite",
            str(_case_path("retrieval", "ingestion", "commit_ingestion_and_query.v1.json")),
            "--output-dir",
            str(tmp_path / "suite-output"),
            "--baseline-dir",
            str(baseline_dir),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "suite run_count=" in captured.out
    assert "compare added=" in captured.out
    comparison_markdown = (tmp_path / "suite-output" / "comparison.md").read_text(encoding="utf-8")
    assert "# RP Eval Comparison" in comparison_markdown
    assert "hard_failure_drifts" in comparison_markdown


def test_eval_cli_run_suite_sync_all_to_langfuse_updates_summary(
    retrieval_session,
    monkeypatch,
    tmp_path,
    capsys,
):
    monkeypatch.setattr(
        "rp.eval.cli._open_eval_session",
        lambda: nullcontext(retrieval_session),
    )
    monkeypatch.setattr(
        "rp.eval.cli.get_langfuse_service",
        lambda: _FakeLangfuseService(enabled=True),
    )
    monkeypatch.setattr(
        "rp.eval.cli.sync_suite_bundle_to_langfuse",
        lambda **kwargs: {
            "suite_summary_synced": True,
            "suite_replay_sync_count": 1,
            "comparison_synced": False,
        },
    )

    exit_code = main(
        [
            "run-suite",
            str(_case_path("retrieval", "ingestion", "commit_ingestion_and_query.v1.json")),
            "--output-dir",
            str(tmp_path / "suite-sync-output"),
            "--sync-all-to-langfuse",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "langfuse_replays_synced=1" in captured.out
    assert "langfuse_comparison_synced=False" in captured.out


def test_eval_cli_sync_flags_require_langfuse_enabled(capsys, monkeypatch):
    monkeypatch.setattr(
        "rp.eval.cli.get_langfuse_service",
        lambda: _FakeLangfuseService(enabled=False),
    )

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "run-suite",
                str(_case_path("retrieval", "search", "query_trace_and_provenance.v1.json")),
                "--sync-all-to-langfuse",
            ]
        )
    captured = capsys.readouterr()

    assert exc_info.value.code == 2
    assert "需要已启用 Langfuse" in captured.err


def test_eval_cli_can_enable_subjective_judges(
    retrieval_session,
    monkeypatch,
    tmp_path,
    capsys,
):
    _seed_judge_model_registry()
    monkeypatch.setattr(
        "rp.eval.cli._open_eval_session",
        lambda: nullcontext(retrieval_session),
    )
    monkeypatch.setattr(
        "rp.services.story_llm_gateway.get_litellm_service",
        lambda: _FakeJudgeLLMService(
            response_payload={
                "status": "pass",
                "label": "good",
                "score": 0.88,
                "explanation": "The query is concrete and aligned with archival retrieval.",
            }
        ),
    )

    exit_code = main(
        [
            "run-case",
            str(_case_path("retrieval", "search", "query_trace_and_provenance.v1.json")),
            "--output-dir",
            str(tmp_path / "judge-output"),
            "--enable-subjective-judges",
            "--judge-model-id",
            "model-eval",
            "--judge-provider-id",
            "provider-eval",
        ]
    )
    capsys.readouterr()

    assert exit_code == 0
    suite_summary = json.loads(
        (tmp_path / "judge-output" / "suite-summary.json").read_text(encoding="utf-8")
    )
    report = suite_summary["items"][0]["report"]
    assert report["pending_judge_hook_ids"] == []
    assert report["subjective_hook_summary"]["executed"] == 1


def test_eval_cli_can_enable_ragas_with_dependency_missing(
    retrieval_session,
    monkeypatch,
    tmp_path,
    capsys,
):
    monkeypatch.setattr(
        "rp.eval.cli._open_eval_session",
        lambda: nullcontext(retrieval_session),
    )
    monkeypatch.setattr("rp.eval.runner.ragas_available", lambda: False)

    exit_code = main(
        [
            "run-case",
            str(_case_path("retrieval", "search", "query_trace_and_provenance.v1.json")),
            "--output-dir",
            str(tmp_path / "ragas-output"),
            "--enable-ragas",
            "--ragas-metric",
            "context_precision",
            "--ragas-metric",
            "faithfulness",
        ]
    )
    capsys.readouterr()

    assert exit_code == 0
    suite_summary = json.loads(
        (tmp_path / "ragas-output" / "suite-summary.json").read_text(encoding="utf-8")
    )
    report = suite_summary["items"][0]["report"]
    assert report["ragas"]["status"] == "dependency_missing"
    assert report["ragas"]["metric_names"] == ["context_precision", "faithfulness"]


def test_eval_cli_passes_ragas_runtime_overrides(
    retrieval_session,
    monkeypatch,
    tmp_path,
    capsys,
):
    monkeypatch.setattr(
        "rp.eval.cli._open_eval_session",
        lambda: nullcontext(retrieval_session),
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
                "embeddings": object(),
                "metadata": {
                    "llm": {"model_id": "model-ragas-judge"},
                    "embeddings": {"model_id": "model-ragas-embed"},
                },
            },
        )(),
    )
    monkeypatch.setattr(
        "rp.eval.runner.run_ragas_evaluation",
        lambda **kwargs: [{"faithfulness": 0.93}],
    )

    exit_code = main(
        [
            "run-case",
            str(_case_path("retrieval", "search", "query_trace_and_provenance.v1.json")),
            "--output-dir",
            str(tmp_path / "ragas-runtime-output"),
            "--enable-ragas",
            "--ragas-metric",
            "faithfulness",
            "--ragas-llm-model-id",
            "model-ragas-judge",
            "--ragas-llm-provider-id",
            "provider-ragas-judge",
            "--ragas-embedding-model-id",
            "model-ragas-embed",
            "--ragas-embedding-provider-id",
            "provider-ragas-embed",
        ]
    )
    capsys.readouterr()

    assert exit_code == 0
    suite_summary = json.loads(
        (tmp_path / "ragas-runtime-output" / "suite-summary.json").read_text(encoding="utf-8")
    )
    report = suite_summary["items"][0]["report"]
    assert report["ragas"]["status"] == "completed"
    assert report["ragas"]["runtime"]["llm"]["model_id"] == "model-ragas-judge"
    assert report["ragas"]["runtime"]["embeddings"]["model_id"] == "model-ragas-embed"


def test_eval_cli_passes_ragas_generation_overrides(
    retrieval_session,
    monkeypatch,
    tmp_path,
    capsys,
):
    monkeypatch.setattr(
        "rp.eval.cli._open_eval_session",
        lambda: nullcontext(retrieval_session),
    )
    monkeypatch.setattr("rp.eval.runner.ragas_available", lambda: True)
    monkeypatch.setattr(
        "rp.eval.runner.generate_eval_rag_response",
        lambda *, sample, config: RagasGeneratedResponse(
            response="Generated eval-only RAG answer.",
            metadata={
                "enabled": True,
                "status": "completed",
                "model_id": config.model_id,
                "provider_id": config.provider_id,
                "max_contexts": config.max_contexts,
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
                "metadata": {"llm": {"model_id": "model-ragas-judge"}},
            },
        )(),
    )
    monkeypatch.setattr(
        "rp.eval.runner.run_ragas_evaluation",
        lambda **kwargs: [{"faithfulness": 0.94}],
    )

    exit_code = main(
        [
            "run-case",
            str(_case_path("retrieval", "search", "query_trace_and_provenance.v1.json")),
            "--output-dir",
            str(tmp_path / "ragas-generation-output"),
            "--enable-ragas",
            "--ragas-metric",
            "faithfulness",
            "--ragas-llm-model-id",
            "model-ragas-judge",
            "--ragas-llm-provider-id",
            "provider-ragas-judge",
            "--ragas-generate-response",
            "--ragas-generator-model-id",
            "model-generator",
            "--ragas-generator-provider-id",
            "provider-generator",
            "--ragas-generator-max-contexts",
            "3",
        ]
    )
    capsys.readouterr()

    assert exit_code == 0
    suite_summary = json.loads(
        (tmp_path / "ragas-generation-output" / "suite-summary.json").read_text(
            encoding="utf-8"
        )
    )
    report = suite_summary["items"][0]["report"]
    assert report["ragas"]["status"] == "completed"
    assert report["ragas"]["generation"]["model_id"] == "model-generator"
    assert report["ragas"]["generation"]["provider_id"] == "provider-generator"
    assert report["ragas"]["generation"]["max_contexts"] == 3
