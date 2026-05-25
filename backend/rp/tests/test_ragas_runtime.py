from __future__ import annotations

import instructor

from models.model_registry import ModelRegistryEntry
from models.provider_registry import ProviderRegistryEntry
from rp.eval.ragas_runtime import (
    ProjectRagasEmbeddings,
    _select_instructor_mode,
    resolve_ragas_runtime_bindings,
)
from rp.models.retrieval_runtime_config import RetrievalRuntimeConfig


class _FakeLiteLLMService:
    def _get_litellm_model(self, provider, model: str) -> str:
        return f"{provider.type}/{model}"

    def _get_api_base(self, provider) -> str:
        return provider.api_url

    def embedding(
        self,
        *,
        provider,
        model: str,
        input_texts,
        encoding_format=None,
        dimensions=None,
    ):
        texts = input_texts if isinstance(input_texts, list) else [input_texts]
        return {
            "data": [
                {"embedding": [float(len(text))], "index": index}
                for index, text in enumerate(texts)
            ]
        }


class _FakeRegistry:
    def __init__(self, entries):
        self._entries = {entry.id: entry for entry in entries}

    def get_entry(self, entry_id: str):
        return self._entries.get(entry_id)


def test_project_ragas_embeddings_exposes_query_and_documents_api():
    wrapper = ProjectRagasEmbeddings(
        service=_FakeLiteLLMService(),
        provider=object(),
        model="embed-model",
    )

    assert wrapper.embed_query("abc") == [3.0]
    assert wrapper.embed_documents(["a", "abcd"]) == [[1.0], [4.0]]


def test_ragas_deepseek_judge_prefers_json_mode_over_tools():
    model = ModelRegistryEntry(
        id="model-deepseek",
        provider_id="provider-deepseek",
        model_name="deepseek-v4-flash",
        display_name="DeepSeek",
        capabilities=["tool"],
        is_enabled=True,
    )

    assert _select_instructor_mode(model) == instructor.Mode.JSON


def test_resolve_ragas_runtime_bindings_uses_judge_llm_and_story_embedding_defaults(
    monkeypatch,
):
    llm_provider = ProviderRegistryEntry(
        id="provider-llm",
        name="Judge Provider",
        type="openai",
        api_key="sk-test",
        api_url="https://judge.example/v1/chat/completions",
        custom_headers={},
        is_enabled=True,
    )
    embedding_provider = ProviderRegistryEntry(
        id="provider-embed",
        name="Embed Provider",
        type="openai",
        api_key="sk-embed",
        api_url="https://embed.example/v1/embeddings",
        custom_headers={},
        is_enabled=True,
    )
    llm_model = ModelRegistryEntry(
        id="model-judge",
        provider_id="provider-llm",
        model_name="gpt-4o-mini",
        display_name="Judge",
        capabilities=["tool"],
        is_enabled=True,
    )
    embedding_model = ModelRegistryEntry(
        id="model-embed",
        provider_id="provider-embed",
        model_name="text-embedding-3-small",
        display_name="Embed",
        capabilities=["embedding"],
        is_enabled=True,
    )

    monkeypatch.setattr(
        "rp.eval.ragas_runtime.get_model_registry_service",
        lambda: _FakeRegistry([llm_model, embedding_model]),
    )
    monkeypatch.setattr(
        "rp.eval.ragas_runtime.get_provider_registry_service",
        lambda: _FakeRegistry([llm_provider, embedding_provider]),
    )

    class _FakeRuntimeConfigService:
        def __init__(self, session) -> None:
            self._session = session

        def resolve_story_config(self, *, story_id: str):
            assert story_id == "story-ragas"
            return RetrievalRuntimeConfig(
                embedding_model_id="model-embed",
                embedding_provider_id="provider-embed",
            )

    monkeypatch.setattr(
        "rp.eval.ragas_runtime.RetrievalRuntimeConfigService",
        _FakeRuntimeConfigService,
    )
    monkeypatch.setattr(
        "rp.eval.ragas_runtime.instructor.from_litellm",
        lambda *args, **kwargs: {"client": "ok", "mode": kwargs.get("mode")},
    )

    class _FakeLiteLLMStructuredLLM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(
        "rp.eval.ragas_runtime.LiteLLMStructuredLLM",
        _FakeLiteLLMStructuredLLM,
    )
    monkeypatch.setattr(
        "rp.eval.ragas_runtime.LiteLLMService",
        lambda: _FakeLiteLLMService(),
    )

    bindings = resolve_ragas_runtime_bindings(
        session=object(),
        story_id="story-ragas",
        env_overrides={
            "judge_model_id": "model-judge",
            "judge_provider_id": "provider-llm",
        },
        metric_names=("response_relevancy",),
    )

    assert bindings.metadata["llm"]["model_id"] == "model-judge"
    assert bindings.metadata["embeddings"]["model_id"] == "model-embed"
    assert bindings.metadata["metric_names"] == ["response_relevancy"]
    assert bindings.llm.kwargs["model"] == "openai/gpt-4o-mini"
    assert bindings.embeddings.embed_query("abc") == [3.0]
