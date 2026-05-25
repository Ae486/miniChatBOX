"""Eval-only RAG response generation before RAGAS judging.

This module is intentionally scoped to offline evaluation. It builds a stable
answer from retrieved contexts so RAGAS can judge an end-to-end RAG sample
without changing writer/worker/orchestrator runtime behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import litellm

from models.chat import ChatCompletionRequest, ChatMessage
from services.litellm_service import LiteLLMService
from services.model_registry import get_model_registry_service
from services.provider_registry import get_provider_registry_service

from .ragas_samples import RagasRetrievalSample

RAGAS_EVAL_RESPONSE_PROMPT_VERSION = "ragas_eval_response_v1"

_DEFAULT_MAX_CONTEXTS = 5
_DEFAULT_MAX_CONTEXT_CHARS = 8000
_DEFAULT_MAX_TOKENS = 512
_DEFAULT_TEMPERATURE = 0.0


@dataclass(frozen=True)
class RagasResponseGenerationConfig:
    """Explicit opt-in config for eval-only response generation."""

    enabled: bool = False
    model_id: str | None = None
    provider_id: str | None = None
    max_contexts: int = _DEFAULT_MAX_CONTEXTS
    max_context_chars: int = _DEFAULT_MAX_CONTEXT_CHARS
    max_tokens: int = _DEFAULT_MAX_TOKENS
    temperature: float = _DEFAULT_TEMPERATURE


@dataclass(frozen=True)
class RagasGeneratedResponse:
    """Generated answer plus reproducibility metadata."""

    response: str
    metadata: dict[str, Any]


def build_ragas_response_generation_config(
    env_overrides: dict[str, Any],
) -> RagasResponseGenerationConfig:
    """Normalize CLI/case env overrides for eval-only answer generation."""

    return RagasResponseGenerationConfig(
        enabled=bool(env_overrides.get("ragas_generate_response")),
        model_id=_optional_text(env_overrides.get("ragas_generator_model_id")),
        provider_id=_optional_text(env_overrides.get("ragas_generator_provider_id")),
        max_contexts=_positive_int(
            env_overrides.get("ragas_generator_max_contexts"),
            default=_DEFAULT_MAX_CONTEXTS,
        ),
        max_context_chars=_positive_int(
            env_overrides.get("ragas_generator_max_context_chars"),
            default=_DEFAULT_MAX_CONTEXT_CHARS,
        ),
        max_tokens=_positive_int(
            env_overrides.get("ragas_generator_max_tokens"),
            default=_DEFAULT_MAX_TOKENS,
        ),
        temperature=_float_value(
            env_overrides.get("ragas_generator_temperature"),
            default=_DEFAULT_TEMPERATURE,
        ),
    )


def generate_eval_rag_response(
    *,
    sample: RagasRetrievalSample,
    config: RagasResponseGenerationConfig,
) -> RagasGeneratedResponse:
    """Generate one deterministic RAG answer for RAGAS evaluation only."""

    if not config.enabled:
        raise ValueError("RAGAS response generation is not enabled")
    if not config.model_id:
        raise ValueError("ragas_generator_model_id is required when generation is enabled")
    if not config.provider_id:
        raise ValueError(
            "ragas_generator_provider_id is required when generation is enabled"
        )

    model_entry = get_model_registry_service().get_entry(config.model_id)
    if model_entry is None or not model_entry.is_enabled:
        raise ValueError(f"ragas generator model not found or disabled: {config.model_id}")

    provider_entry = get_provider_registry_service().get_entry(config.provider_id)
    if provider_entry is None or not provider_entry.is_enabled:
        raise ValueError(
            f"ragas generator provider not found or disabled: {config.provider_id}"
        )

    provider = provider_entry.to_runtime_provider()
    contexts = _bounded_contexts(
        sample.retrieved_contexts,
        max_contexts=config.max_contexts,
        max_context_chars=config.max_context_chars,
    )
    messages = [
        ChatMessage(
            role="system",
            content=(
                "You are an evaluation-only RAG answer generator. "
                "Answer using only the provided contexts. "
                "If the contexts do not contain enough evidence, say so clearly. "
                "Do not add unstated facts."
            ),
        ),
        ChatMessage(
            role="user",
            content=_build_generation_user_prompt(
                query=sample.query,
                contexts=contexts,
            ),
        ),
    ]
    service = LiteLLMService()
    request = ChatCompletionRequest(
        model=model_entry.model_name,
        model_id=model_entry.id,
        provider_id=provider_entry.id,
        provider=provider,
        messages=messages,
        stream=False,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    kwargs = service._build_completion_kwargs(request)
    kwargs["stream"] = False
    raw_response = litellm.completion(**kwargs)
    payload = (
        raw_response.model_dump()
        if hasattr(raw_response, "model_dump")
        else dict(raw_response)
    )
    response_text = _extract_message_text(payload).strip()
    if not response_text:
        raise ValueError("RAGAS generator returned an empty response")

    return RagasGeneratedResponse(
        response=response_text,
        metadata={
            "enabled": True,
            "status": "completed",
            "prompt_version": RAGAS_EVAL_RESPONSE_PROMPT_VERSION,
            "model_id": model_entry.id,
            "model_name": model_entry.model_name,
            "provider_id": provider_entry.id,
            "provider_type": provider_entry.type,
            "max_contexts": config.max_contexts,
            "max_context_chars": config.max_context_chars,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "input_context_count": len(sample.retrieved_contexts),
            "used_context_count": len(contexts),
        },
    )


def _build_generation_user_prompt(*, query: str, contexts: list[str]) -> str:
    rendered_contexts = "\n\n".join(
        f"[Context {index}]\n{context}"
        for index, context in enumerate(contexts, start=1)
    )
    if not rendered_contexts:
        rendered_contexts = "[No retrieved context]"
    return (
        "Question:\n"
        f"{query}\n\n"
        "Retrieved contexts:\n"
        f"{rendered_contexts}\n\n"
        "Write a concise answer grounded only in the retrieved contexts."
    )


def _bounded_contexts(
    contexts: list[str],
    *,
    max_contexts: int,
    max_context_chars: int,
) -> list[str]:
    selected: list[str] = []
    remaining = max_context_chars
    for context in contexts[:max_contexts]:
        text = str(context).strip()
        if not text or remaining <= 0:
            continue
        if len(text) > remaining:
            text = text[:remaining].rstrip()
        selected.append(text)
        remaining -= len(text)
    return selected


def _extract_message_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return ""
    message = first_choice.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                str(item.get("text") or item.get("content") or "")
                for item in content
                if isinstance(item, dict)
            )
    text = first_choice.get("text")
    return str(text) if text is not None else ""


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _positive_int(value: Any, *, default: int) -> int:
    if value is None:
        return default
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"Expected positive integer, got {value!r}")
    return parsed


def _float_value(value: Any, *, default: float) -> float:
    if value is None:
        return default
    return float(value)
