"""Project-side runtime bindings for executing RAGAS with current backend registries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import instructor
import litellm
from models.model_registry import ModelRegistryEntry
from models.provider_registry import ProviderRegistryEntry
from ragas.llms import LiteLLMStructuredLLM
from rp.services.retrieval_runtime_config_service import RetrievalRuntimeConfigService
from services.litellm_service import LiteLLMService
from services.model_registry import get_model_registry_service
from services.provider_registry import get_provider_registry_service

from .ragas_adapter import parse_ragas_metrics

_EMBEDDING_REQUIRED_METRICS = {"response_relevancy"}
_LLM_REQUIRED_METRICS = {
    "context_precision",
    "context_recall",
    "response_relevancy",
    "faithfulness",
}


@dataclass(frozen=True)
class RagasRuntimeBindings:
    llm: Any | None
    embeddings: Any | None
    metadata: dict[str, Any]


class ProjectRagasEmbeddings:
    """Project-native embedding adapter aligned with current backend provider routing."""

    def __init__(
        self,
        *,
        service: LiteLLMService,
        provider,
        model: str,
        dimensions: int | None = None,
    ) -> None:
        self._service = service
        self._provider = provider
        self._model = model
        self._dimensions = dimensions

    def embed_query(self, text: str) -> list[float]:
        return self._embed_texts([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed_texts(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await _run_in_thread(self.embed_query, text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await _run_in_thread(self.embed_documents, texts)

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = self._service.embedding(
            provider=self._provider,
            model=self._model,
            input_texts=texts,
            encoding_format="float",
            dimensions=self._dimensions,
        )
        data = response.get("data")
        if not isinstance(data, list) or not data:
            raise ValueError(f"Embedding response missing data: {response}")
        vectors: list[list[float]] = []
        for item in data:
            if not isinstance(item, dict):
                raise ValueError(f"Embedding item is not an object: {item!r}")
            vector = item.get("embedding")
            if not isinstance(vector, list):
                raise ValueError(f"Embedding item missing vector: {item!r}")
            vectors.append([float(value) for value in vector])
        return vectors


def resolve_ragas_runtime_bindings(
    *,
    session,
    story_id: str | None,
    env_overrides: dict[str, Any],
    metric_names: tuple[str, ...] | list[str],
) -> RagasRuntimeBindings:
    normalized_metrics = set(parse_ragas_metrics(metric_names))
    needs_llm = bool(normalized_metrics.intersection(_LLM_REQUIRED_METRICS))
    needs_embeddings = bool(normalized_metrics.intersection(_EMBEDDING_REQUIRED_METRICS))

    runtime_config = (
        RetrievalRuntimeConfigService(session).resolve_story_config(story_id=story_id or "")
        if story_id
        else None
    )
    metadata: dict[str, Any] = {
        "metric_names": sorted(normalized_metrics),
        "story_id": story_id,
    }

    llm = None
    if needs_llm:
        llm_binding = _resolve_model_binding(
            model_id=env_overrides.get("ragas_llm_model_id") or env_overrides.get("judge_model_id"),
            provider_id=env_overrides.get("ragas_llm_provider_id") or env_overrides.get("judge_provider_id"),
            role="ragas_llm",
        )
        llm = _build_ragas_llm(binding=llm_binding)
        metadata["llm"] = llm_binding.metadata

    embeddings = None
    if needs_embeddings:
        embedding_binding = _resolve_model_binding(
            model_id=(
                env_overrides.get("ragas_embedding_model_id")
                or (runtime_config.embedding_model_id if runtime_config else None)
            ),
            provider_id=(
                env_overrides.get("ragas_embedding_provider_id")
                or (runtime_config.embedding_provider_id if runtime_config else None)
            ),
            role="ragas_embedding",
        )
        embeddings = _build_ragas_embeddings(binding=embedding_binding)
        metadata["embeddings"] = embedding_binding.metadata

    return RagasRuntimeBindings(
        llm=llm,
        embeddings=embeddings,
        metadata=metadata,
    )


@dataclass(frozen=True)
class _ResolvedModelBinding:
    model_entry: ModelRegistryEntry
    provider_entry: ProviderRegistryEntry
    runtime_provider: Any
    litellm_model: str
    api_base: str | None
    metadata: dict[str, Any]


def _resolve_model_binding(
    *,
    model_id: Any,
    provider_id: Any,
    role: str,
) -> _ResolvedModelBinding:
    resolved_model_id = str(model_id or "").strip()
    if not resolved_model_id:
        raise ValueError(f"{role} model_id is required")

    model_entry = get_model_registry_service().get_entry(resolved_model_id)
    if model_entry is None or not model_entry.is_enabled:
        raise ValueError(f"{role} model not found or disabled: {resolved_model_id}")

    resolved_provider_id = str(provider_id or model_entry.provider_id or "").strip()
    if not resolved_provider_id:
        raise ValueError(f"{role} provider_id is required")

    provider_entry = get_provider_registry_service().get_entry(resolved_provider_id)
    if provider_entry is None or not provider_entry.is_enabled:
        raise ValueError(f"{role} provider not found or disabled: {resolved_provider_id}")

    lite = LiteLLMService()
    runtime_provider = provider_entry.to_runtime_provider()
    return _ResolvedModelBinding(
        model_entry=model_entry,
        provider_entry=provider_entry,
        runtime_provider=runtime_provider,
        litellm_model=lite._get_litellm_model(runtime_provider, model_entry.model_name),
        api_base=lite._get_api_base(runtime_provider),
        metadata={
            "model_id": model_entry.id,
            "model_name": model_entry.model_name,
            "provider_id": provider_entry.id,
            "provider_type": provider_entry.type,
        },
    )


def _build_ragas_llm(*, binding: _ResolvedModelBinding) -> LiteLLMStructuredLLM:
    mode = _select_instructor_mode(binding.model_entry)
    client = instructor.from_litellm(
        litellm.completion,
        mode=mode,
    )
    kwargs: dict[str, Any] = {
        "api_key": binding.runtime_provider.api_key,
        "timeout": 60,
    }
    if binding.api_base:
        kwargs["api_base"] = binding.api_base
    if binding.runtime_provider.custom_headers:
        kwargs["extra_headers"] = binding.runtime_provider.custom_headers
    return LiteLLMStructuredLLM(
        client=client,
        model=binding.litellm_model,
        provider=binding.provider_entry.type,
        **kwargs,
    )


def _build_ragas_embeddings(*, binding: _ResolvedModelBinding) -> ProjectRagasEmbeddings:
    return ProjectRagasEmbeddings(
        service=LiteLLMService(),
        provider=binding.runtime_provider,
        model=binding.model_entry.model_name,
    )


def _select_instructor_mode(model_entry: ModelRegistryEntry) -> instructor.Mode:
    model_name = str(model_entry.model_name or "").strip().lower()
    display_name = str(model_entry.display_name or "").strip().lower()
    if "deepseek" in model_name or "deepseek" in display_name:
        return instructor.Mode.JSON
    capabilities = {str(item).strip().lower() for item in model_entry.capabilities}
    if "tool" in capabilities:
        return instructor.Mode.TOOLS
    return instructor.Mode.JSON


async def _run_in_thread(fn, *args):
    import asyncio

    return await asyncio.to_thread(fn, *args)
