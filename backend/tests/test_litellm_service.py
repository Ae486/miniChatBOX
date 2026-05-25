"""Tests for LiteLLM service."""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
import pytest

from models.chat import (
    AttachedFile,
    ChatCompletionRequest,
    ChatMessage,
    CircuitBreakerConfig,
    ProviderConfig,
)
from services.litellm_service import LiteLLMService, get_http_status_for_exception


@pytest.fixture
def service():
    with patch("services.litellm_service.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(
            llm_request_timeout=120.0,
            llm_stream_timeout=20.0,
            llm_num_retries=2,
            llm_allowed_fails=0,
            llm_cooldown_time=0.0,
            use_litellm_router=True,
            debug=False,
        )
        return LiteLLMService()


@pytest.fixture
def openai_provider():
    return ProviderConfig(type="openai", api_key="sk-test", api_url="https://api.openai.com/v1")


@pytest.fixture
def sample_request(openai_provider):
    return ChatCompletionRequest(
        model="gpt-4o",
        messages=[ChatMessage(role="user", content="Hello")],
        stream=False,
        temperature=0.7,
        provider=openai_provider,
    )


class TestGetLitellmModel:
    def test_adds_prefix(self, service, openai_provider):
        assert service._get_litellm_model(openai_provider, "gpt-4") == "openai/gpt-4"

    def test_deepseek(self, service):
        p = ProviderConfig(type="deepseek", api_key="", api_url="https://api.deepseek.com/v1")
        assert service._get_litellm_model(p, "deepseek-chat") == "deepseek/deepseek-chat"

    def test_gemini(self, service):
        p = ProviderConfig(type="gemini", api_key="", api_url="https://generativelanguage.googleapis.com")
        assert service._get_litellm_model(p, "gemini-2.0-flash") == "gemini/gemini-2.0-flash"

    def test_claude(self, service):
        p = ProviderConfig(type="claude", api_key="", api_url="https://api.anthropic.com")
        assert service._get_litellm_model(p, "claude-3-opus") == "anthropic/claude-3-opus"

    def test_already_prefixed(self, service, openai_provider):
        assert service._get_litellm_model(openai_provider, "openai/custom-model") == "openai/custom-model"

    def test_openai_compatible_vendor_model_gets_openai_prefix(self, service, openai_provider):
        assert (
            service._get_litellm_model(openai_provider, "Qwen/Qwen3-Embedding-8B")
            == "openai/Qwen/Qwen3-Embedding-8B"
        )

    def test_unknown_provider_defaults_openai(self, service):
        p = ProviderConfig(type="unknown", api_key="", api_url="https://custom.com")
        assert service._get_litellm_model(p, "model") == "openai/model"


class TestGetApiBase:
    """Core fix: strip endpoint paths Flutter appends (e.g., /chat/completions)."""

    def test_strips_chat_completions(self, service):
        p = ProviderConfig(type="openai", api_key="", api_url="https://api.openai.com/v1/chat/completions")
        assert service._get_api_base(p) == "https://api.openai.com/v1"

    def test_strips_messages(self, service):
        p = ProviderConfig(type="claude", api_key="", api_url="https://api.anthropic.com/v1/messages")
        assert service._get_api_base(p) == "https://api.anthropic.com/v1"

    def test_strips_rerank(self, service):
        p = ProviderConfig(type="openai", api_key="", api_url="https://api.siliconflow.com/v1/rerank")
        assert service._get_api_base(p) == "https://api.siliconflow.com/v1"

    def test_no_stripping_needed(self, service):
        p = ProviderConfig(type="openai", api_key="", api_url="https://my-proxy.com/v1")
        assert service._get_api_base(p) == "https://my-proxy.com/v1"

    def test_base_only(self, service):
        p = ProviderConfig(type="openai", api_key="", api_url="https://my-proxy.com")
        assert service._get_api_base(p) == "https://my-proxy.com"

    def test_force_mode_hash(self, service):
        p = ProviderConfig(type="openai", api_key="", api_url="https://api.example.com/custom#")
        assert service._get_api_base(p) == "https://api.example.com/custom"

    def test_trailing_slash_stripped(self, service):
        p = ProviderConfig(type="openai", api_key="", api_url="https://api.example.com/v1/")
        assert service._get_api_base(p) == "https://api.example.com/v1"

    def test_empty_url_returns_none(self, service):
        p = ProviderConfig(type="openai", api_key="", api_url="")
        assert service._get_api_base(p) is None


class TestBuildCompletionKwargs:
    def test_basic_kwargs(self, service, sample_request):
        kwargs = service._build_completion_kwargs(sample_request)
        assert kwargs["model"] == "openai/gpt-4o"
        assert kwargs["api_key"] == "sk-test"
        assert kwargs["api_base"] == "https://api.openai.com/v1"
        assert kwargs["temperature"] == 0.7

    def test_full_url_gets_stripped(self, service):
        p = ProviderConfig(type="openai", api_key="sk-test", api_url="https://my-proxy.com/v1/chat/completions")
        req = ChatCompletionRequest(
            model="gpt-4o",
            messages=[ChatMessage(role="user", content="Hello")],
            provider=p,
        )
        kwargs = service._build_completion_kwargs(req)
        assert kwargs["api_base"] == "https://my-proxy.com/v1"

    def test_extra_body(self, service, openai_provider):
        req = ChatCompletionRequest(
            model="gemini-2.0-flash",
            messages=[ChatMessage(role="user", content="Hi")],
            provider=openai_provider,
            extra_body={"google": {"thinking_config": {"include_thoughts": True}}},
        )
        kwargs = service._build_completion_kwargs(req)
        assert kwargs["extra_body"]["google"]["thinking_config"]["include_thoughts"] is True

    def test_custom_headers(self, service):
        p = ProviderConfig(
            type="openai",
            api_key="sk-test",
            api_url="https://api.example.com",
            custom_headers={"X-Custom": "value"},
        )
        req = ChatCompletionRequest(
            model="gpt-4",
            messages=[ChatMessage(role="user", content="Hi")],
            provider=p,
        )
        kwargs = service._build_completion_kwargs(req)
        assert kwargs["extra_headers"] == {"X-Custom": "value"}

    def test_none_params_excluded(self, service, openai_provider):
        req = ChatCompletionRequest(
            model="gpt-4",
            messages=[ChatMessage(role="user", content="Hi")],
            provider=openai_provider,
        )
        kwargs = service._build_completion_kwargs(req)
        assert "temperature" not in kwargs
        assert "max_tokens" not in kwargs

    def test_default_proxy_style_params_are_normalized_to_direct_chain_behavior(
        self, service, openai_provider
    ):
        req = ChatCompletionRequest(
            model="gpt-4",
            messages=[
                ChatMessage(role="system", content=""),
                ChatMessage(role="user", content="Hi"),
            ],
            provider=openai_provider,
            temperature=0.7,
            max_tokens=2048,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        kwargs = service._build_completion_kwargs(req)
        assert kwargs["temperature"] == 0.7
        assert kwargs["max_tokens"] == 2048
        assert "top_p" not in kwargs
        assert "frequency_penalty" not in kwargs
        assert "presence_penalty" not in kwargs
        assert kwargs["messages"] == [{"role": "user", "content": "Hi"}]

    def test_include_reasoning_passed(self, service, openai_provider):
        req = ChatCompletionRequest(
            model="deepseek-reasoner",
            messages=[ChatMessage(role="user", content="Hi")],
            provider=openai_provider,
            include_reasoning=True,
        )
        kwargs = service._build_completion_kwargs(req)
        assert kwargs["include_reasoning"] is True

    def test_include_reasoning_not_set(self, service, openai_provider):
        req = ChatCompletionRequest(
            model="gpt-4",
            messages=[ChatMessage(role="user", content="Hi")],
            provider=openai_provider,
        )
        kwargs = service._build_completion_kwargs(req)
        assert "include_reasoning" not in kwargs

    def test_gemini_does_not_forward_include_reasoning(self, service):
        provider = ProviderConfig(
            type="gemini",
            api_key="sk-test",
            api_url="https://generativelanguage.googleapis.com/v1beta/openai",
        )
        req = ChatCompletionRequest(
            model="gemini-2.5-pro",
            messages=[ChatMessage(role="user", content="Hi")],
            provider=provider,
            include_reasoning=True,
        )
        kwargs = service._build_completion_kwargs(req)
        assert "include_reasoning" not in kwargs
        assert kwargs["extra_body"]["google"]["thinking_config"]["include_thoughts"] is True


class TestTokenCounting:
    def test_count_request_tokens_uses_litellm_token_counter(
        self,
        service,
        openai_provider,
    ):
        request = ChatCompletionRequest(
            model="gpt-4o",
            messages=[ChatMessage(role="user", content="Hello")],
            stream=False,
            provider=openai_provider,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "setup_memory_open",
                        "description": "Open a setup memory ref.",
                        "parameters": {
                            "type": "object",
                            "properties": {"ref": {"type": "string"}},
                            "required": ["ref"],
                        },
                    },
                }
            ],
            tool_choice="auto",
        )

        with patch(
            "services.litellm_service.litellm.token_counter",
            return_value=42,
        ) as token_counter:
            result = service.count_request_tokens(request)

        assert result == {
            "prompt_tokens": 42,
            "source": "litellm_token_counter",
            "model": "openai/gpt-4o",
            "includes_tools": True,
        }
        token_counter.assert_called_once()
        kwargs = token_counter.call_args.kwargs
        assert kwargs["model"] == "openai/gpt-4o"
        assert kwargs["messages"] == [{"role": "user", "content": "Hello"}]
        assert kwargs["tools"] == request.tools
        assert kwargs["tool_choice"] == "auto"


class TestNonChatKwargs:
    def test_build_embedding_kwargs_uses_openai_prefix_for_openai_compatible_vendor_model(
        self, service, openai_provider
    ):
        kwargs = service.build_embedding_kwargs(
            provider=openai_provider,
            model="Qwen/Qwen3-Embedding-8B",
            input_texts="hi",
        )
        assert kwargs["model"] == "openai/Qwen/Qwen3-Embedding-8B"

    def test_build_direct_embedding_kwargs_uses_embeddings_endpoint(self, service):
        provider = ProviderConfig(
            type="openai",
            api_key="sk-test",
            api_url="https://api.siliconflow.com/v1/chat/completions",
            custom_headers={"X-Test": "1"},
        )
        url, headers, payload = service._build_direct_embedding_kwargs(
            provider=provider,
            model="Qwen/Qwen3-Embedding-8B",
            input_texts="hi",
            encoding_format="float",
            dimensions=1024,
        )
        assert url == "https://api.siliconflow.com/v1/embeddings"
        assert headers["Authorization"] == "Bearer sk-test"
        assert headers["X-Test"] == "1"
        assert payload == {
            "model": "Qwen/Qwen3-Embedding-8B",
            "input": "hi",
            "encoding_format": "float",
            "dimensions": 1024,
        }

    def test_build_direct_rerank_kwargs_uses_base_rerank_endpoint(self, service):
        provider = ProviderConfig(
            type="openai",
            api_key="sk-test",
            api_url="https://api.siliconflow.com/v1/chat/completions",
            custom_headers={"X-Test": "1"},
        )
        url, headers, payload = service._build_direct_rerank_kwargs(
            provider=provider,
            model="Qwen/Qwen3-VL-Reranker-8B",
            query="hi",
            documents=["hello", "hi"],
            top_n=1,
        )
        assert url == "https://api.siliconflow.com/v1/rerank"
        assert headers["Authorization"] == "Bearer sk-test"
        assert headers["X-Test"] == "1"
        assert payload["model"] == "Qwen/Qwen3-VL-Reranker-8B"
        assert payload["return_documents"] is False
        assert payload["top_n"] == 1

    def test_files_are_merged_into_messages_before_litellm_call(
        self, service, openai_provider, tmp_path
    ):
        text_file = tmp_path / "notes.txt"
        text_file.write_text("Alpha content", encoding="utf-8")
        req = ChatCompletionRequest(
            model="gpt-4o",
            messages=[ChatMessage(role="user", content="Use the file")],
            provider=openai_provider,
            files=[
                AttachedFile(
                    path=str(text_file),
                    mime_type="text/plain",
                    name="notes.txt",
                )
            ],
        )

        kwargs = service._build_completion_kwargs(req)
        assert isinstance(kwargs["messages"][0]["content"], list)
        assert kwargs["messages"][0]["content"][0]["type"] == "text"
        assert "Alpha content" in kwargs["messages"][0]["content"][0]["text"]

    def test_tools_and_tool_choice_forwarded_to_kwargs(
        self, service, openai_provider
    ):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web",
                    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                },
            }
        ]
        req = ChatCompletionRequest(
            model="gpt-4o",
            messages=[ChatMessage(role="user", content="Search tokyo")],
            provider=openai_provider,
            tools=tools,
            tool_choice="auto",
        )
        kwargs = service._build_completion_kwargs(req)
        assert kwargs["tools"] == tools
        assert kwargs["tool_choice"] == "auto"

    def test_tools_omitted_when_none(self, service, openai_provider):
        req = ChatCompletionRequest(
            model="gpt-4o",
            messages=[ChatMessage(role="user", content="Hi")],
            provider=openai_provider,
        )
        kwargs = service._build_completion_kwargs(req)
        assert "tools" not in kwargs
        assert "tool_choice" not in kwargs


class _BrokenDelta:
    def __init__(self):
        self.content = "hello"


class _BrokenChoice:
    def __init__(self):
        self.delta = _BrokenDelta()
        self.finish_reason = None


class _BrokenChunk:
    def __init__(self):
        self.id = "chunk-1"
        self.model = "gemini-2.5-flash"
        self.choices = [_BrokenChoice()]

    def model_dump(self, **kwargs):
        raise RuntimeError("broken serializer")


class TestStreamChunkCoercion:
    def test_coerce_stream_chunk_falls_back_to_attribute_extraction(self, service):
        chunk = _BrokenChunk()

        chunk_dict = service._coerce_stream_chunk(chunk)

        assert chunk_dict["id"] == "chunk-1"
        assert chunk_dict["model"] == "gemini-2.5-flash"
        assert chunk_dict["choices"][0]["delta"]["content"] == "hello"


class TestRouterConfig:
    def test_build_router_model_list(self, service, sample_request):
        model_list = service._build_router_model_list(sample_request)
        assert model_list[0]["model_name"] == "gpt-4o"
        litellm_params = model_list[0]["litellm_params"]
        assert litellm_params["model"] == "openai/gpt-4o"
        assert litellm_params["api_base"] == "https://api.openai.com/v1"
        assert litellm_params["timeout"] == 120.0
        assert litellm_params["stream_timeout"] == 20.0

    def test_build_router_model_list_uses_auto_mode_fallback_timeout_override(
        self, service
    ):
        provider = ProviderConfig(
            type="openai",
            api_key="sk-test",
            api_url="https://api.openai.com/v1",
            backend_mode="auto",
            fallback_timeout_ms=7000,
        )
        request = ChatCompletionRequest(
            model="gpt-4o",
            messages=[ChatMessage(role="user", content="Hello")],
            provider=provider,
        )

        model_list = service._build_router_model_list(request)
        litellm_params = model_list[0]["litellm_params"]

        assert litellm_params["stream_timeout"] == 7.0

    @patch("services.litellm_service.litellm.Router")
    def test_router_is_cached_per_provider_and_model(
        self, mock_router_cls, service, sample_request
    ):
        first = service._get_router(sample_request)
        second = service._get_router(sample_request)
        assert first is second
        mock_router_cls.assert_called_once()

    @patch("services.litellm_service.litellm.Router")
    def test_router_uses_auto_mode_circuit_breaker_overrides(
        self, mock_router_cls, service
    ):
        provider = ProviderConfig(
            type="openai",
            api_key="sk-test",
            api_url="https://api.openai.com/v1",
            backend_mode="auto",
            circuit_breaker=CircuitBreakerConfig(
                failure_threshold=4,
                open_ms=45000,
            ),
        )
        request = ChatCompletionRequest(
            model="gpt-4o",
            messages=[ChatMessage(role="user", content="Hello")],
            provider=provider,
        )

        service._get_router(request)

        router_kwargs = mock_router_cls.call_args.kwargs
        assert router_kwargs["allowed_fails"] == 4
        assert router_kwargs["cooldown_time"] == 45.0

    @patch("services.litellm_service.litellm.Router")
    def test_streaming_router_is_not_cached(
        self, mock_router_cls, service, sample_request
    ):
        sample_request.stream = True

        service._get_router(sample_request, allow_cached=False)
        service._get_router(sample_request, allow_cached=False)

        assert mock_router_cls.call_count == 2


class TestChatCompletion:
    @pytest.mark.asyncio
    @patch("services.litellm_service.litellm.Router")
    async def test_non_streaming(self, mock_router_cls, service, sample_request):
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "id": "chatcmpl-test",
            "choices": [{"message": {"role": "assistant", "content": "Hi!"}}],
        }
        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock(return_value=mock_response)
        mock_router_cls.return_value = mock_router

        result = await service.chat_completion(sample_request)
        assert result["id"] == "chatcmpl-test"
        mock_router.acompletion.assert_awaited_once()
        assert mock_router.acompletion.call_args[1]["stream"] is False

    @pytest.mark.asyncio
    async def test_missing_provider(self, service):
        req = ChatCompletionRequest(
            model="gpt-4",
            messages=[ChatMessage(role="user", content="Hi")],
        )
        with pytest.raises(ValueError, match="Provider configuration is required"):
            await service.chat_completion(req)

    @pytest.mark.asyncio
    @patch("services.litellm_service.litellm.acompletion")
    async def test_falls_back_to_plain_acompletion_when_router_disabled(
        self, mock_acompletion, service, sample_request
    ):
        service.settings.use_litellm_router = False
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "id": "chatcmpl-test",
            "choices": [{"message": {"role": "assistant", "content": "Hi!"}}],
        }
        mock_acompletion.return_value = mock_response

        result = await service.chat_completion(sample_request)
        assert result["id"] == "chatcmpl-test"
        mock_acompletion.assert_called_once()


class TestChatCompletionStream:
    @staticmethod
    def _extract_contents(chunks):
        contents = []
        for chunk in chunks:
            if not chunk.startswith("data: {"):
                continue
            payload = json.loads(chunk[6:])
            choices = payload.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            content = delta.get("content")
            if isinstance(content, str):
                contents.append(content)
        return contents

    @pytest.mark.asyncio
    @patch("services.litellm_service.litellm.Router")
    async def test_streaming(self, mock_router_cls, service, sample_request):
        chunk1 = MagicMock()
        chunk1.model_dump.return_value = {"choices": [{"delta": {"content": "Hello"}}]}
        chunk2 = MagicMock()
        chunk2.model_dump.return_value = {"choices": [{"delta": {"content": " world"}}]}

        async def mock_aiter():
            yield chunk1
            yield chunk2

        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock(return_value=mock_aiter())
        mock_router_cls.return_value = mock_router
        sample_request.stream = True

        chunks = []
        async for chunk in service.chat_completion_stream(sample_request):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[0].startswith("data: ")
        assert chunks[-1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    @patch("services.litellm_service.litellm.Router")
    async def test_streaming_reasoning_is_normalized_to_content_chunks(
        self, mock_router_cls, service, sample_request
    ):
        chunk1 = MagicMock()
        chunk1.model_dump.return_value = {
            "id": "chatcmpl-test",
            "model": "deepseek-r1",
            "choices": [{"delta": {"reasoning_content": "先分析"}}],
        }
        chunk2 = MagicMock()
        chunk2.model_dump.return_value = {
            "id": "chatcmpl-test",
            "model": "deepseek-r1",
            "choices": [{"delta": {"content": "最终答案"}}],
        }

        async def mock_aiter():
            yield chunk1
            yield chunk2

        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock(return_value=mock_aiter())
        mock_router_cls.return_value = mock_router
        sample_request.stream = True
        sample_request.model = "deepseek-r1"
        sample_request.provider.type = "deepseek"

        chunks = []
        async for chunk in service.chat_completion_stream(sample_request):
            chunks.append(chunk)

        assert any('"content": "<think>"' in chunk for chunk in chunks)
        assert self._extract_contents(chunks) == ["<think>", "先分析", "</think>", "最终答案"]
        assert chunks[-1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    @patch("services.litellm_service.litellm.Router")
    async def test_error_before_stream(self, mock_router_cls, service, sample_request):
        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock(
            side_effect=litellm.AuthenticationError(
                message="Invalid API key", llm_provider="openai", model="gpt-4o"
            )
        )
        mock_router_cls.return_value = mock_router
        sample_request.stream = True

        chunks = []
        async for chunk in service.chat_completion_stream(sample_request):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert '"error"' in chunks[0]
        assert chunks[-1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    @patch("services.litellm_service.litellm.Router")
    async def test_error_mid_stream(self, mock_router_cls, service, sample_request):
        chunk1 = MagicMock()
        chunk1.model_dump.return_value = {"choices": [{"delta": {"content": "Hi"}}]}

        async def mock_aiter_with_error():
            yield chunk1
            raise litellm.APIConnectionError(
                message="Connection lost", llm_provider="openai", model="gpt-4o"
            )

        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock(return_value=mock_aiter_with_error())
        mock_router_cls.return_value = mock_router
        sample_request.stream = True

        chunks = []
        async for chunk in service.chat_completion_stream(sample_request):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert '"error"' in chunks[1]
        assert chunks[-1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    @patch("services.litellm_service.litellm.Router")
    async def test_stream_error_invalidates_cached_router(self, mock_router_cls, service, sample_request):
        class _ClosableStream:
            def __init__(self):
                self.closed = False
                self._emitted = False

            def __aiter__(self):
                return self

            async def __anext__(self):
                if not self._emitted:
                    self._emitted = True
                    yield_chunk = MagicMock()
                    yield_chunk.model_dump.return_value = {
                        "choices": [{"delta": {"content": "Hi"}}]
                    }
                    return yield_chunk
                raise litellm.APIConnectionError(
                    message="Connection lost", llm_provider="openai", model="gpt-4o"
                )

            async def aclose(self):
                self.closed = True

        stream = _ClosableStream()
        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock(return_value=stream)
        mock_router_cls.return_value = mock_router
        sample_request.stream = True

        service._routers[service._router_cache_key(sample_request)] = mock_router

        chunks = []
        async for chunk in service.chat_completion_stream(sample_request):
            chunks.append(chunk)

        assert '"error"' in chunks[1]
        assert service._router_cache_key(sample_request) not in service._routers
        assert stream.closed is True

    @pytest.mark.asyncio
    @patch("services.litellm_service.litellm.Router")
    async def test_streaming_typed_mode_emits_typed_payloads(
        self, mock_router_cls, service, sample_request
    ):
        chunk1 = MagicMock()
        chunk1.model_dump.return_value = {
            "id": "chatcmpl-test",
            "model": "deepseek-r1",
            "choices": [{"delta": {"reasoning_content": "先分析"}}],
        }
        chunk2 = MagicMock()
        chunk2.model_dump.return_value = {
            "id": "chatcmpl-test",
            "model": "deepseek-r1",
            "choices": [{"delta": {"content": "最终答案"}}],
        }

        async def mock_aiter():
            yield chunk1
            yield chunk2

        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock(return_value=mock_aiter())
        mock_router_cls.return_value = mock_router
        sample_request.stream = True
        sample_request.stream_event_mode = "typed"
        sample_request.model = "deepseek-r1"
        sample_request.provider.type = "deepseek"

        chunks = []
        async for chunk in service.chat_completion_stream(sample_request):
            chunks.append(chunk)

        assert 'data: {"type": "thinking_delta", "delta": "\\u5148\\u5206\\u6790"}\n\n' in chunks
        assert 'data: {"type": "text_delta", "delta": "\\u6700\\u7ec8\\u7b54\\u6848"}\n\n' in chunks
        assert chunks[-1] == 'data: {"type": "done"}\n\n'

    @pytest.mark.asyncio
    @patch("services.litellm_service.litellm.Router")
    async def test_streaming_typed_mode_emits_usage_payload(
        self, mock_router_cls, service, sample_request
    ):
        chunk1 = MagicMock()
        chunk1.model_dump.return_value = {
            "id": "chatcmpl-test",
            "model": "gpt-4o-mini",
            "choices": [{"delta": {"content": "最终答案"}}],
        }
        chunk2 = MagicMock()
        chunk2.model_dump.return_value = {
            "id": "chatcmpl-test",
            "model": "gpt-4o-mini",
            "choices": [],
            "usage": {
                "prompt_tokens": 42,
                "completion_tokens": 11,
                "total_tokens": 53,
            },
        }

        async def mock_aiter():
            yield chunk1
            yield chunk2

        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock(return_value=mock_aiter())
        mock_router_cls.return_value = mock_router
        sample_request.stream = True
        sample_request.stream_event_mode = "typed"
        sample_request.model = "gpt-4o-mini"
        sample_request.provider.type = "openai"

        chunks = []
        async for chunk in service.chat_completion_stream(sample_request):
            chunks.append(chunk)

        assert 'data: {"type": "usage", "prompt_tokens": 42, "completion_tokens": 11, "total_tokens": 53}\n\n' in chunks
        assert chunks[-1] == 'data: {"type": "done"}\n\n'


class TestExceptionMapping:
    def test_auth_error(self):
        exc = litellm.AuthenticationError(
            message="Invalid key", llm_provider="openai", model="gpt-4"
        )
        status, code = get_http_status_for_exception(exc)
        assert status == 401

    def test_unknown_exception(self):
        status, code = get_http_status_for_exception(RuntimeError("unknown"))
        assert status == 500
