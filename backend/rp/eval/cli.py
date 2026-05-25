"""CLI entrypoints for local RP eval execution and comparison."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from sqlmodel import Session

from services.database import create_db_and_tables, get_engine
from services.langfuse_service import get_langfuse_service

from .case_loader import load_case, load_cases
from .comparison import compare_suite_outputs, evaluate_suite_thresholds, summarize_suite
from .langfuse_sync import (
    sync_comparison_to_langfuse,
    sync_replay_to_langfuse,
    sync_suite_bundle_to_langfuse,
    sync_suite_summary_to_langfuse,
)
from .ragas_replay import attach_ragas_report_to_replay, run_ragas_on_replay_payload
from .replay import SampledRetrievalTrace, load_replay, save_sampled_retrieval_replay
from .reporting import render_comparison_markdown, render_suite_markdown
from .runner import EvalRunner
from .suite import EvalSuiteRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rp-eval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_case = subparsers.add_parser("run-case", help="Run a single eval case.")
    run_case.add_argument("path", help="Case file path.")
    run_case.add_argument("--output-dir", help="Optional report/replay bundle directory.")
    run_case.add_argument("--repeat-count", type=int, help="Override per-case repeat count.")
    run_case.add_argument("--max-fails", type=int, default=0, help="Maximum allowed assertion fail count.")
    run_case.add_argument("--max-warns", type=int, help="Maximum allowed assertion warn count.")
    run_case.add_argument("--enable-subjective-judges", action="store_true", help="Enable configured LLM subjective judges.")
    run_case.add_argument("--judge-model-id", help="Model id used for subjective judge execution.")
    run_case.add_argument("--judge-provider-id", help="Provider id used for subjective judge execution.")
    run_case.add_argument("--enable-ragas", action="store_true", help="Enable retrieval-side RAGAS metrics.")
    run_case.add_argument("--ragas-metric", action="append", default=[], help="RAGAS metric name to execute. Repeat the flag for multiple metrics.")
    run_case.add_argument("--ragas-response", help="Optional explicit response text passed into RAGAS samples.")
    run_case.add_argument("--ragas-reference", help="Optional explicit reference text passed into RAGAS samples.")
    run_case.add_argument("--ragas-llm-model-id", help="Model id used as the evaluator LLM for RAGAS metrics.")
    run_case.add_argument("--ragas-llm-provider-id", help="Provider id used as the evaluator LLM provider for RAGAS metrics.")
    run_case.add_argument("--ragas-embedding-model-id", help="Embedding model id used by embedding-based RAGAS metrics.")
    run_case.add_argument("--ragas-embedding-provider-id", help="Embedding provider id used by embedding-based RAGAS metrics.")
    run_case.add_argument("--ragas-generate-response", action="store_true", help="Generate an eval-only RAG response from retrieved contexts before RAGAS judging.")
    run_case.add_argument("--ragas-generator-model-id", help="Model id used to generate eval-only RAG responses.")
    run_case.add_argument("--ragas-generator-provider-id", help="Provider id used to generate eval-only RAG responses.")
    run_case.add_argument("--ragas-generator-max-contexts", type=int, help="Maximum retrieved contexts passed to the eval-only RAG generator.")
    run_case.add_argument("--ragas-generator-max-context-chars", type=int, help="Maximum total context characters passed to the eval-only RAG generator.")
    run_case.add_argument("--ragas-generator-max-tokens", type=int, help="Maximum output tokens for the eval-only RAG generator.")
    run_case.add_argument("--ragas-generator-temperature", type=float, help="Temperature for the eval-only RAG generator.")
    run_case.add_argument("--sync-ragas-to-langfuse", action="store_true", help="Sync completed RAGAS metrics to Langfuse scores when Langfuse is enabled.")
    run_case.add_argument("--retrieval-embedding-model-id", help="Embedding model id used by the retrieval main chain.")
    run_case.add_argument("--retrieval-embedding-provider-id", help="Embedding provider id used by the retrieval main chain.")
    run_case.add_argument("--retrieval-rerank-model-id", help="Rerank model id used by the retrieval main chain.")
    run_case.add_argument("--retrieval-rerank-provider-id", help="Rerank provider id used by the retrieval main chain.")
    run_case.add_argument("--json", action="store_true", help="Print full JSON payload.")

    run_suite = subparsers.add_parser("run-suite", help="Run one case or a case directory.")
    run_suite.add_argument("path", help="Case file or directory path.")
    run_suite.add_argument("--output-dir", help="Optional report/replay bundle directory.")
    run_suite.add_argument("--baseline-dir", help="Optional baseline suite directory.")
    run_suite.add_argument("--repeat-count", type=int, help="Override repeat count for every case in the suite.")
    run_suite.add_argument("--max-fails", type=int, default=0, help="Maximum allowed assertion fail count.")
    run_suite.add_argument("--max-warns", type=int, help="Maximum allowed assertion warn count.")
    run_suite.add_argument("--enable-subjective-judges", action="store_true", help="Enable configured LLM subjective judges.")
    run_suite.add_argument("--judge-model-id", help="Model id used for subjective judge execution.")
    run_suite.add_argument("--judge-provider-id", help="Provider id used for subjective judge execution.")
    run_suite.add_argument("--enable-ragas", action="store_true", help="Enable retrieval-side RAGAS metrics.")
    run_suite.add_argument("--ragas-metric", action="append", default=[], help="RAGAS metric name to execute. Repeat the flag for multiple metrics.")
    run_suite.add_argument("--ragas-response", help="Optional explicit response text passed into RAGAS samples.")
    run_suite.add_argument("--ragas-reference", help="Optional explicit reference text passed into RAGAS samples.")
    run_suite.add_argument("--ragas-llm-model-id", help="Model id used as the evaluator LLM for RAGAS metrics.")
    run_suite.add_argument("--ragas-llm-provider-id", help="Provider id used as the evaluator LLM provider for RAGAS metrics.")
    run_suite.add_argument("--ragas-embedding-model-id", help="Embedding model id used by embedding-based RAGAS metrics.")
    run_suite.add_argument("--ragas-embedding-provider-id", help="Embedding provider id used by embedding-based RAGAS metrics.")
    run_suite.add_argument("--ragas-generate-response", action="store_true", help="Generate an eval-only RAG response from retrieved contexts before RAGAS judging.")
    run_suite.add_argument("--ragas-generator-model-id", help="Model id used to generate eval-only RAG responses.")
    run_suite.add_argument("--ragas-generator-provider-id", help="Provider id used to generate eval-only RAG responses.")
    run_suite.add_argument("--ragas-generator-max-contexts", type=int, help="Maximum retrieved contexts passed to the eval-only RAG generator.")
    run_suite.add_argument("--ragas-generator-max-context-chars", type=int, help="Maximum total context characters passed to the eval-only RAG generator.")
    run_suite.add_argument("--ragas-generator-max-tokens", type=int, help="Maximum output tokens for the eval-only RAG generator.")
    run_suite.add_argument("--ragas-generator-temperature", type=float, help="Temperature for the eval-only RAG generator.")
    run_suite.add_argument("--sync-ragas-to-langfuse", action="store_true", help="Sync completed RAGAS metrics to Langfuse scores when Langfuse is enabled.")
    run_suite.add_argument("--retrieval-embedding-model-id", help="Embedding model id used by the retrieval main chain.")
    run_suite.add_argument("--retrieval-embedding-provider-id", help="Embedding provider id used by the retrieval main chain.")
    run_suite.add_argument("--retrieval-rerank-model-id", help="Rerank model id used by the retrieval main chain.")
    run_suite.add_argument("--retrieval-rerank-provider-id", help="Rerank provider id used by the retrieval main chain.")
    run_suite.add_argument(
        "--allow-soft-fail",
        action="append",
        default=[],
        help="Case id allowed to fail deterministic assertions without tripping the threshold gate.",
    )
    run_suite.add_argument("--json", action="store_true", help="Print full JSON payload.")
    run_suite.add_argument("--sync-suite-to-langfuse", action="store_true", help="Sync suite summary and thresholds to Langfuse.")
    run_suite.add_argument(
        "--sync-all-to-langfuse",
        action="store_true",
        help=(
            "Sync the full suite bundle to Langfuse: suite summary, every replay in the suite, "
            "and comparison drift when --baseline-dir is provided. Requires replay bundles."
        ),
    )

    compare = subparsers.add_parser("compare", help="Compare two suite outputs.")
    compare.add_argument("current", help="Current suite output directory or suite-summary.json.")
    compare.add_argument("baseline", help="Baseline suite output directory or suite-summary.json.")
    compare.add_argument("--json", action="store_true", help="Print full JSON payload.")
    compare.add_argument("--sync-comparison-to-langfuse", action="store_true", help="Sync comparison drift summary to Langfuse.")

    replay = subparsers.add_parser("replay", help="Inspect one replay bundle.")
    replay.add_argument("path", help="Replay JSON path.")
    replay.add_argument("--json", action="store_true", help="Print full JSON payload.")
    replay.add_argument("--sync-replay-to-langfuse", action="store_true", help="Sync the replay payload to Langfuse as a sampled eval trace.")
    replay.add_argument("--run-ragas", action="store_true", help="Run RAGAS against the replay payload.")
    replay.add_argument("--write-back", action="store_true", help="Write the generated RAGAS report back into the replay file.")
    replay.add_argument("--ragas-metric", action="append", default=[], help="RAGAS metric name to execute. Repeat the flag for multiple metrics.")
    replay.add_argument("--ragas-response", help="Optional explicit response text passed into RAGAS samples.")
    replay.add_argument("--ragas-reference", help="Optional explicit reference text passed into RAGAS samples.")
    replay.add_argument("--ragas-llm-model-id", help="Model id used as the evaluator LLM for RAGAS metrics.")
    replay.add_argument("--ragas-llm-provider-id", help="Provider id used as the evaluator LLM provider for RAGAS metrics.")
    replay.add_argument("--ragas-embedding-model-id", help="Embedding model id used by embedding-based RAGAS metrics.")
    replay.add_argument("--ragas-embedding-provider-id", help="Embedding provider id used by embedding-based RAGAS metrics.")
    replay.add_argument("--ragas-generate-response", action="store_true", help="Generate an eval-only RAG response from retrieved contexts before RAGAS judging.")
    replay.add_argument("--ragas-generator-model-id", help="Model id used to generate eval-only RAG responses.")
    replay.add_argument("--ragas-generator-provider-id", help="Provider id used to generate eval-only RAG responses.")
    replay.add_argument("--ragas-generator-max-contexts", type=int, help="Maximum retrieved contexts passed to the eval-only RAG generator.")
    replay.add_argument("--ragas-generator-max-context-chars", type=int, help="Maximum total context characters passed to the eval-only RAG generator.")
    replay.add_argument("--ragas-generator-max-tokens", type=int, help="Maximum output tokens for the eval-only RAG generator.")
    replay.add_argument("--ragas-generator-temperature", type=float, help="Temperature for the eval-only RAG generator.")

    sampled_replay = subparsers.add_parser("sampled-replay", help="Build a replay bundle from a normalized sampled retrieval trace JSON.")
    sampled_replay.add_argument("input", help="Sampled retrieval trace JSON path.")
    sampled_replay.add_argument("output", help="Output replay JSON path.")
    sampled_replay.add_argument("--json", action="store_true", help="Print full JSON payload.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run-case":
            return asyncio.run(_run_case_command(args))
        if args.command == "run-suite":
            _require_langfuse_enabled(
                parser,
                requested=bool(
                    getattr(args, "sync_suite_to_langfuse", False)
                    or getattr(args, "sync_all_to_langfuse", False)
                ),
                flag_names=["--sync-suite-to-langfuse", "--sync-all-to-langfuse"],
            )
            return asyncio.run(_run_suite_command(args))
        if args.command == "compare":
            payload = compare_suite_outputs(args.current, args.baseline)
            if getattr(args, "sync_comparison_to_langfuse", False):
                _require_langfuse_enabled(
                    parser,
                    requested=True,
                    flag_names=["--sync-comparison-to-langfuse"],
                )
                sync_comparison_to_langfuse(comparison=payload)
            _print_payload(payload, as_json=args.json)
            return 0
        if args.command == "replay":
            payload = load_replay(args.path)
            if getattr(args, "run_ragas", False):
                with _open_eval_session() as session:
                    ragas_report = run_ragas_on_replay_payload(
                        session=session,
                        replay_payload=payload,
                        env_overrides=_replay_ragas_env_overrides(args),
                    )
                payload = attach_ragas_report_to_replay(
                    replay_payload=payload,
                    report=ragas_report,
                )
                if getattr(args, "write_back", False):
                    Path(args.path).write_text(
                        json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
            if getattr(args, "sync_replay_to_langfuse", False):
                _require_langfuse_enabled(
                    parser,
                    requested=True,
                    flag_names=["--sync-replay-to-langfuse"],
                )
                sync_replay_to_langfuse(replay_payload=payload)
            _print_payload(payload, as_json=args.json)
            return 0
        if args.command == "sampled-replay":
            sample_payload = _load_json_file(args.input)
            sample = SampledRetrievalTrace.model_validate(sample_payload)
            replay_path = save_sampled_retrieval_replay(args.output, sample)
            payload = load_replay(replay_path)
            _print_payload(payload, as_json=args.json)
            return 0
        parser.error(f"Unsupported command: {args.command}")
        return 2
    finally:
        _flush_langfuse_for_cli()


async def _run_case_command(args) -> int:
    if args.output_dir or args.repeat_count:
        payload = await _run_suite_like_case(
            args.path,
            output_dir=args.output_dir,
            repeat_count=args.repeat_count,
            args=args,
        )
        result_payload = {
            "suite": payload["suite"],
            "summary": payload["summary"],
            "thresholds": evaluate_suite_thresholds(
                payload["suite"],
                max_fail=int(args.max_fails),
                max_warn=args.max_warns,
            ),
        }
        if args.output_dir:
            _write_analysis_bundle(
                output_dir=args.output_dir,
                payload=result_payload,
            )
        raw_suite_payload = payload.get("suite")
        suite_payload = raw_suite_payload if isinstance(raw_suite_payload, dict) else {}
        raw_suite_items = suite_payload.get("items")
        suite_items = raw_suite_items if isinstance(raw_suite_items, list) else []
        first_item = suite_items[0] if suite_items else None
        result_payload = {
            **result_payload,
            "case": first_item.get("report") if isinstance(first_item, dict) else None,
        }
        _print_payload(result_payload, as_json=args.json)
        return 0 if result_payload["thresholds"]["passed"] else 1

    with _open_eval_session() as session:
        case = load_case(args.path)
        case = _apply_cli_env_overrides(case, args)
        result = await EvalRunner(session).run_case(case)

    payload = {
        "case_id": result.case.case_id,
        "run_id": result.run.run_id,
        "status": result.run.status,
        "report": result.report,
    }
    _print_payload(payload, as_json=args.json)
    fail_total = int(result.report.get("assertion_summary", {}).get("fail", 0))
    return 0 if result.run.status != "failed" and fail_total == 0 else 1


async def _run_suite_command(args) -> int:
    payload = await _run_suite_like_case(
        args.path,
        output_dir=args.output_dir,
        repeat_count=args.repeat_count,
        args=args,
    )
    raw_suite_payload = payload.get("suite")
    suite_payload = raw_suite_payload if isinstance(raw_suite_payload, dict) else {}
    result_payload = {
        "suite": suite_payload,
        "summary": payload["summary"],
        "thresholds": evaluate_suite_thresholds(
            suite_payload,
            max_fail=int(args.max_fails),
            max_warn=args.max_warns,
            allowed_soft_fail_case_ids=set(args.allow_soft_fail or []),
        ),
    }
    if args.baseline_dir:
        result_payload["comparison"] = compare_suite_outputs(
            suite_payload,
            args.baseline_dir,
        )
    if getattr(args, "sync_suite_to_langfuse", False):
        sync_suite_summary_to_langfuse(
            suite_payload=suite_payload,
            summary=result_payload["summary"],
            thresholds=result_payload["thresholds"],
        )
        result_payload["langfuse_sync"] = {
            "suite_summary_synced": True,
            "suite_replay_sync_count": 0,
            "comparison_synced": False,
        }
    if getattr(args, "sync_all_to_langfuse", False):
        sync_summary = sync_suite_bundle_to_langfuse(
            suite_payload=suite_payload,
            summary=result_payload["summary"],
            thresholds=result_payload["thresholds"],
            comparison=result_payload.get("comparison"),
        )
        result_payload["langfuse_sync"] = sync_summary
    if args.output_dir:
        _write_analysis_bundle(
            output_dir=args.output_dir,
            payload=result_payload,
        )
    _print_payload(result_payload, as_json=args.json)
    return 0 if result_payload["thresholds"]["passed"] else 1


async def _run_suite_like_case(
    path: str,
    *,
    output_dir: str | None,
    repeat_count: int | None = None,
    args=None,
) -> dict[str, object]:
    cases = [_apply_cli_env_overrides(case, args) for case in load_cases(path)]
    with _open_eval_session() as session:
        suite = EvalSuiteRunner(session)
        suite_result = await suite.run_cases(
            cases,
            output_dir=output_dir,
            repeat_override=repeat_count,
        )

    suite_payload = suite_result.model_dump(mode="json")
    summary = summarize_suite(suite_payload)
    thresholds = evaluate_suite_thresholds(suite_payload)
    return {
        "suite": suite_payload,
        "summary": summary,
        "thresholds": thresholds,
    }


def _apply_cli_env_overrides(case, args):
    if args is None:
        return case
    env_overrides = dict(case.input.env_overrides)
    if getattr(args, "enable_subjective_judges", False):
        env_overrides["enable_subjective_judges"] = True
    if getattr(args, "judge_model_id", None):
        env_overrides["judge_model_id"] = args.judge_model_id
    if getattr(args, "judge_provider_id", None):
        env_overrides["judge_provider_id"] = args.judge_provider_id
    if getattr(args, "enable_ragas", False):
        env_overrides["enable_ragas"] = True
    if getattr(args, "ragas_metric", None):
        env_overrides["ragas_metrics"] = list(args.ragas_metric)
    if getattr(args, "ragas_response", None):
        env_overrides["ragas_response"] = args.ragas_response
    if getattr(args, "ragas_reference", None):
        env_overrides["ragas_reference"] = args.ragas_reference
    if getattr(args, "ragas_llm_model_id", None):
        env_overrides["ragas_llm_model_id"] = args.ragas_llm_model_id
    if getattr(args, "ragas_llm_provider_id", None):
        env_overrides["ragas_llm_provider_id"] = args.ragas_llm_provider_id
    if getattr(args, "ragas_embedding_model_id", None):
        env_overrides["ragas_embedding_model_id"] = args.ragas_embedding_model_id
    if getattr(args, "ragas_embedding_provider_id", None):
        env_overrides["ragas_embedding_provider_id"] = args.ragas_embedding_provider_id
    if getattr(args, "ragas_generate_response", False):
        env_overrides["ragas_generate_response"] = True
    if getattr(args, "ragas_generator_model_id", None):
        env_overrides["ragas_generator_model_id"] = args.ragas_generator_model_id
    if getattr(args, "ragas_generator_provider_id", None):
        env_overrides["ragas_generator_provider_id"] = args.ragas_generator_provider_id
    if getattr(args, "ragas_generator_max_contexts", None):
        env_overrides["ragas_generator_max_contexts"] = args.ragas_generator_max_contexts
    if getattr(args, "ragas_generator_max_context_chars", None):
        env_overrides["ragas_generator_max_context_chars"] = args.ragas_generator_max_context_chars
    if getattr(args, "ragas_generator_max_tokens", None):
        env_overrides["ragas_generator_max_tokens"] = args.ragas_generator_max_tokens
    if getattr(args, "ragas_generator_temperature", None) is not None:
        env_overrides["ragas_generator_temperature"] = args.ragas_generator_temperature
    if getattr(args, "sync_ragas_to_langfuse", False):
        env_overrides["sync_ragas_to_langfuse"] = True
    if getattr(args, "retrieval_embedding_model_id", None):
        env_overrides["retrieval_embedding_model_id"] = args.retrieval_embedding_model_id
    if getattr(args, "retrieval_embedding_provider_id", None):
        env_overrides["retrieval_embedding_provider_id"] = args.retrieval_embedding_provider_id
    if getattr(args, "retrieval_rerank_model_id", None):
        env_overrides["retrieval_rerank_model_id"] = args.retrieval_rerank_model_id
    if getattr(args, "retrieval_rerank_provider_id", None):
        env_overrides["retrieval_rerank_provider_id"] = args.retrieval_rerank_provider_id
    if env_overrides == dict(case.input.env_overrides):
        return case
    return case.model_copy(
        deep=True,
        update={
            "input": case.input.model_copy(
                deep=True,
                update={"env_overrides": env_overrides},
            )
        },
    )


def _replay_ragas_env_overrides(args) -> dict[str, Any]:
    env_overrides: dict[str, Any] = {
        "enable_ragas": True,
    }
    if getattr(args, "ragas_metric", None):
        env_overrides["ragas_metrics"] = list(args.ragas_metric)
    if getattr(args, "ragas_response", None):
        env_overrides["ragas_response"] = args.ragas_response
    if getattr(args, "ragas_reference", None):
        env_overrides["ragas_reference"] = args.ragas_reference
    if getattr(args, "ragas_llm_model_id", None):
        env_overrides["ragas_llm_model_id"] = args.ragas_llm_model_id
    if getattr(args, "ragas_llm_provider_id", None):
        env_overrides["ragas_llm_provider_id"] = args.ragas_llm_provider_id
    if getattr(args, "ragas_embedding_model_id", None):
        env_overrides["ragas_embedding_model_id"] = args.ragas_embedding_model_id
    if getattr(args, "ragas_embedding_provider_id", None):
        env_overrides["ragas_embedding_provider_id"] = args.ragas_embedding_provider_id
    if getattr(args, "ragas_generate_response", False):
        env_overrides["ragas_generate_response"] = True
    if getattr(args, "ragas_generator_model_id", None):
        env_overrides["ragas_generator_model_id"] = args.ragas_generator_model_id
    if getattr(args, "ragas_generator_provider_id", None):
        env_overrides["ragas_generator_provider_id"] = args.ragas_generator_provider_id
    if getattr(args, "ragas_generator_max_contexts", None):
        env_overrides["ragas_generator_max_contexts"] = args.ragas_generator_max_contexts
    if getattr(args, "ragas_generator_max_context_chars", None):
        env_overrides["ragas_generator_max_context_chars"] = args.ragas_generator_max_context_chars
    if getattr(args, "ragas_generator_max_tokens", None):
        env_overrides["ragas_generator_max_tokens"] = args.ragas_generator_max_tokens
    if getattr(args, "ragas_generator_temperature", None) is not None:
        env_overrides["ragas_generator_temperature"] = args.ragas_generator_temperature
    return env_overrides


def _flush_langfuse_for_cli() -> None:
    """CLI is a short-lived process; flush queued Langfuse events before exit."""
    get_langfuse_service().flush()


def _write_analysis_bundle(*, output_dir: str, payload: dict[str, Any]) -> None:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for name, item in (
        ("derived-summary.json", payload.get("summary")),
        ("thresholds.json", payload.get("thresholds")),
        ("comparison.json", payload.get("comparison")),
    ):
        if item is None:
            continue
        (target_dir / name).write_text(
            json.dumps(item, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    summary = payload.get("summary")
    thresholds = payload.get("thresholds")
    if isinstance(summary, dict) and isinstance(thresholds, dict):
        (target_dir / "summary.md").write_text(
            render_suite_markdown(summary=summary, thresholds=thresholds),
            encoding="utf-8",
        )
    comparison = payload.get("comparison")
    if isinstance(comparison, dict):
        (target_dir / "comparison.md").write_text(
            render_comparison_markdown(comparison=comparison),
            encoding="utf-8",
        )


def _load_json_file(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at path: {path}")
    return payload


@contextmanager
def _open_eval_session() -> Iterator[Session]:
    create_db_and_tables()
    with Session(get_engine()) as session:
        yield session


def _print_payload(payload: dict, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if "suite" in payload:
        summary = payload["summary"]
        thresholds = payload["thresholds"]
        langfuse_sync = payload.get("langfuse_sync")
        if not isinstance(langfuse_sync, dict):
            langfuse_sync = {}
        diagnostic_summary = dict(summary.get("diagnostic_summary") or {})
        top_reason_codes = _top_counter_keys(diagnostic_summary.get("reason_codes"))
        top_primary_suspects = _top_counter_keys(diagnostic_summary.get("primary_suspects"))
        top_next_actions = _top_counter_keys(
            diagnostic_summary.get("recommended_next_actions"),
            limit=1,
        )
        print(
            f"suite run_count={summary['run_count']} "
            f"cases={summary['case_count']} "
            f"failed_runs={summary['failed_run_count']} "
            f"repeat_cases={len(summary['repeat_case_ids'])} "
            f"fails={summary['assertion_fail_total']} "
            f"warns={summary['assertion_warn_total']} "
            f"diagnostic_expectation_failures={sum(int(value or 0) for value in dict(diagnostic_summary.get('diagnostic_expectation_failures') or {}).values())} "
            f"top_reason_codes={','.join(top_reason_codes) or 'none'} "
            f"top_primary_suspects={','.join(top_primary_suspects) or 'none'} "
            f"top_next_actions={','.join(top_next_actions) or 'none'} "
            f"executed_judges={summary.get('executed_judge_hook_total', 0)} "
            f"pending_judges={summary.get('pending_judge_hook_total', 0)} "
            f"langfuse_replays_synced={int(langfuse_sync.get('suite_replay_sync_count') or 0)} "
            f"langfuse_comparison_synced={bool(langfuse_sync.get('comparison_synced'))} "
            f"threshold_passed={thresholds['passed']}"
        )
        if "comparison" not in payload:
            return

    if "comparison" in payload:
        comparison = payload["comparison"]
        print(
            f"compare added={len(comparison['added_case_ids'])} "
            f"removed={len(comparison['removed_case_ids'])} "
            f"changed={len(comparison['changed_cases'])} "
            f"finish_reason_drifts={len(comparison['drift_summary']['changed_finish_reason_case_ids'])} "
            f"reason_code_drifts={len(comparison['drift_summary'].get('changed_reason_code_case_ids') or [])} "
            f"primary_suspect_drifts={len(comparison['drift_summary'].get('changed_primary_suspect_case_ids') or [])} "
            f"outcome_chain_drifts={len(comparison['drift_summary'].get('changed_outcome_chain_case_ids') or [])} "
            f"next_action_drifts={len(comparison['drift_summary'].get('changed_recommended_next_action_case_ids') or [])} "
            f"diagnostic_expectation_drifts={len(comparison['drift_summary'].get('changed_diagnostic_expectation_case_ids') or [])} "
            f"judge_status_drifts={len(comparison['drift_summary'].get('changed_subjective_status_case_ids') or [])}"
        )
        return

    if "report" in payload:
        report = payload["report"]
        case = payload.get("case")
        run = payload.get("run")
        case_id = (
            payload.get("case_id")
            or (case.get("case_id") if isinstance(case, dict) else None)
            or report.get("case_id")
            or "unknown"
        )
        status = (
            payload.get("status")
            or (run.get("status") if isinstance(run, dict) else None)
            or report.get("status")
            or "unknown"
        )
        outcome_chain = report.get("outcome_chain_display") or [
            f"{key}={value}"
            for key, value in dict(report.get("outcome_chain") or {}).items()
        ]
        evidence_refs = list(report.get("evidence_refs") or [])
        print(
            f"case {case_id} status={status} "
            f"finish_reason={report.get('finish_reason') or 'n/a'} "
            f"failure_layer={report.get('failure_layer') or 'none'} "
            f"fails={report.get('assertion_summary', {}).get('fail', 0)} "
            f"reason_codes={','.join(report.get('reason_codes') or []) or 'none'} "
            f"primary_suspects={','.join(report.get('primary_suspects') or []) or 'none'} "
            f"next_action={report.get('recommended_next_action') or 'n/a'} "
            f"outcome_chain={';'.join(outcome_chain) or 'none'} "
            f"evidence_refs={','.join(evidence_refs) or 'none'}"
        )
        return

    if "case" in payload and "run" in payload:
        print(
            f"replay case_id={payload['case'].get('case_id')} "
            f"run_id={payload['run'].get('run_id')} "
            f"status={payload['run'].get('status')}"
        )
        return

    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _top_counter_keys(counter: Any, *, limit: int = 3) -> list[str]:
    if not isinstance(counter, dict):
        return []
    return [
        str(key)
        for key, _ in sorted(
            counter.items(),
            key=lambda item: (-int(item[1] or 0), str(item[0])),
        )[:limit]
    ]


def _require_langfuse_enabled(
    parser: argparse.ArgumentParser,
    *,
    requested: bool,
    flag_names: list[str],
) -> None:
    if not requested:
        return
    if get_langfuse_service().enabled:
        return
    joined_flags = ", ".join(flag_names)
    parser.error(
        f"{joined_flags} 需要已启用 Langfuse。请先设置 "
        "LANGFUSE_ENABLED=true、LANGFUSE_PUBLIC_KEY、LANGFUSE_SECRET_KEY，"
        "必要时再设置 LANGFUSE_BASE_URL。"
    )


if __name__ == "__main__":
    sys.exit(main())
