"""The vendor adapters, exercised against fakes instead of credentials.

Every adapter here takes its client as a constructor argument, which is what makes
this file possible: the request/response translation can be tested for real
without a network call, a key, or a vendor SDK installed. That matters more than
usual for this layer, because translation bugs are silent — an adapter that sends
a parameter under the wrong name, or in the wrong place in the config, produces a
plausible-looking response and a wrong answer rather than an exception.

The three LLM adapters all owe callers the same thing, so the shared behaviour is
parametrised rather than repeated:

* a structured request must actually produce a validated object, not prose
* an endpoint that ignores structured-output hints must still be handled, because
  a compatible gateway is a supported deployment and it will ignore them
* a schema mismatch is a ``ProviderError`` naming the field, not a bare
  ``ValidationError`` from three frames down
* provider faults are reported as retryable or not, because the router decides
  whether to try again based on that flag
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel

from app.core.exceptions import ProviderError
from app.providers.embedding.openai import OpenAIEmbeddingProvider
from app.providers.llm.gemini import GeminiLLMProvider
from app.providers.llm.openai import OpenAILLMProvider
from app.providers.search.tavily import TavilySearchProvider


class Pricing(BaseModel):
    plan: str
    price: str


# --------------------------------------------------------------- fakes


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Message(content)
        self.finish_reason = "stop"


class _Usage:
    prompt_tokens = 11
    completion_tokens = 7


class _Completion:
    def __init__(self, content: str) -> None:
        self.choices = [_Choice(content)]
        self.usage = _Usage()


class _OpenAIClient:
    """Stands in for ``AsyncOpenAI``, recording what it was asked for."""

    def __init__(self, content: str = '{"plan": "Pro", "price": "$99"}') -> None:
        self.content = content
        self.kwargs: dict[str, Any] = {}
        self.chat = self
        self.completions = self

    async def create(self, **kwargs: Any) -> _Completion:
        self.kwargs = kwargs
        return _Completion(self.content)


class _GeminiResult:
    def __init__(self, text: str) -> None:
        self.text = text
        self.usage_metadata = type(
            "Usage", (), {"prompt_token_count": 11, "candidates_token_count": 7}
        )()


class _GeminiModels:
    def __init__(self, text: str) -> None:
        self.text = text
        self.kwargs: dict[str, Any] = {}

    async def generate_content(self, **kwargs: Any) -> _GeminiResult:
        self.kwargs = kwargs
        return _GeminiResult(self.text)


class _GeminiClient:
    def __init__(self, text: str) -> None:
        self.aio = self
        self.models = _GeminiModels(text)


def _openai(content: str = '{"plan": "Pro", "price": "$99"}') -> OpenAILLMProvider:
    return OpenAILLMProvider("sk-test", client=_OpenAIClient(content))


def _gemini(content: str = '{"plan": "Pro", "price": "$99"}') -> GeminiLLMProvider:
    return GeminiLLMProvider("g-test", client=_GeminiClient(content))


#: Both LLM adapters, constructed the same way, so shared rules are stated once.
LLM_ADAPTERS = [
    pytest.param(_openai, _OpenAIClient, id="openai"),
    pytest.param(_gemini, _GeminiModels, id="gemini"),
]


class TestStructuredRequests:
    @pytest.mark.parametrize(("build", "inner"), LLM_ADAPTERS)
    async def test_a_schema_is_validated_before_the_response_is_returned(
        self, build: Any, inner: Any
    ) -> None:
        provider = build('{"plan": "Pro", "price": "$99"}')
        result = await provider.generate("extract the pricing", schema=Pricing)

        assert result.provider in {"openai", "gemini"}
        assert json.loads(result.content)["plan"] == "Pro"

    @pytest.mark.parametrize(("build", "inner"), LLM_ADAPTERS)
    async def test_prose_where_json_was_required_is_a_provider_error(
        self, build: Any, inner: Any
    ) -> None:
        """A gateway that ignores the structured-output hint still gets caught.

        This is the case the local validation exists for. Without it the failure
        would surface later as a schema error in some unrelated part of the run.
        """
        provider = build("Sure! The plan costs $99 per month.")
        with pytest.raises(ProviderError, match="did not return JSON"):
            await provider.generate("extract the pricing", schema=Pricing)

    @pytest.mark.parametrize(("build", "inner"), LLM_ADAPTERS)
    async def test_a_fenced_answer_is_still_read(
        self, build: Any, inner: Any
    ) -> None:
        """Models emit fences even when told not to."""
        provider = build('```json\n{"plan": "Pro", "price": "$99"}\n```')
        result = await provider.generate("extract the pricing", schema=Pricing)

        assert result.content.strip().startswith("```")

    @pytest.mark.parametrize(("build", "inner"), LLM_ADAPTERS)
    async def test_a_schema_mismatch_names_the_missing_field(
        self, build: Any, inner: Any
    ) -> None:
        provider = build('{"plan": "Pro"}')
        with pytest.raises(ProviderError, match="price"):
            await provider.generate("extract the pricing", schema=Pricing)

    @pytest.mark.parametrize(("build", "inner"), LLM_ADAPTERS)
    async def test_a_prose_request_is_returned_untouched(
        self, build: Any, inner: Any
    ) -> None:
        """A prompt that wanted prose must not be failed for not producing JSON."""
        provider = build("Acme Cloud is a mid-market hosting provider.")
        result = await provider.generate("summarise", schema=None)

        assert result.content == "Acme Cloud is a mid-market hosting provider."

    @pytest.mark.parametrize(("build", "inner"), LLM_ADAPTERS)
    async def test_a_provider_fault_is_reported_as_retryable(
        self, build: Any, inner: Any
    ) -> None:
        """The router decides whether to retry from this flag."""
        provider = build("{}")

        async def boom(**_: Any) -> None:
            raise RuntimeError("connection reset")

        if provider.name == "openai":
            provider._client.chat.completions.create = boom
        else:
            provider._client.models.generate_content = boom

        with pytest.raises(ProviderError) as excinfo:
            await provider.generate("hello", schema=Pricing)
        assert excinfo.value.retryable is True


class TestOpenAIRequestShape:
    async def test_the_system_prompt_is_sent_as_a_system_message(self) -> None:
        client = _OpenAIClient()
        provider = OpenAILLMProvider("sk-test", client=client)

        await provider.generate("the user prompt", system="the system prompt")

        assert client.kwargs["messages"] == [
            {"role": "system", "content": "the system prompt"},
            {"role": "user", "content": "the user prompt"},
        ]

    async def test_no_schema_means_no_response_format(self) -> None:
        """Sending the parameter unsolicited gets a refusal from strict gateways."""
        client = _OpenAIClient()
        await OpenAILLMProvider("sk-test", client=client).generate("hi")

        assert "response_format" not in client.kwargs

    async def test_a_schema_asks_for_json_object_mode(self) -> None:
        client = _OpenAIClient()
        provider = OpenAILLMProvider("sk-test", client=client)

        await provider.generate("hi", schema=Pricing)

        assert client.kwargs["response_format"] == {"type": "json_object"}

    async def test_the_json_requirement_is_stated_in_words_too(self) -> None:
        """A gateway that drops response_format still gets the instruction."""
        client = _OpenAIClient()
        provider = OpenAILLMProvider("sk-test", client=client)

        await provider.generate("extract the pricing", schema=Pricing)

        sent = client.kwargs["messages"][-1]["content"]
        assert "extract the pricing" in sent
        assert "JSON" in sent

    async def test_usage_is_carried_through_for_cost_tracking(self) -> None:
        result = await _openai().generate("hi")
        assert (result.prompt_tokens, result.completion_tokens) == (11, 7)

    async def test_a_blank_key_is_refused_at_construction(self) -> None:
        """A provider built without a key must fail before any request is made."""
        with pytest.raises(ProviderError, match="OPENAI_API_KEY"):
            OpenAILLMProvider("   ", client=_OpenAIClient())


class TestGeminiRequestShape:
    async def test_the_schema_goes_inside_the_config(self) -> None:
        """Gemini takes the response schema in the config, not as an argument.

        Sent as a sibling of ``config`` it is not a recognised field, so the
        request is rejected — or silently ignored, and the run gets prose.
        """
        models = _GeminiModels('{"plan": "Pro", "price": "$99"}')
        provider = GeminiLLMProvider("g-test", client=_GeminiClient("{}"))
        provider._client.models = models

        await provider.generate("extract the pricing", schema=Pricing)

        config = models.kwargs["config"]
        assert config["response_mime_type"] == "application/json"
        assert config["response_schema"] is Pricing
        assert "response_schema" not in models.kwargs

    async def test_an_inline_dict_schema_becomes_an_instruction(self) -> None:
        """A raw dict schema is rejected by the API, so it is stated in words."""
        models = _GeminiModels('{"plan": "Pro"}')
        provider = GeminiLLMProvider("g-test", client=_GeminiClient("{}"))
        provider._client.models = models
        schema = {"type": "object", "properties": {"plan": {"type": "string"}}}

        await provider.generate("extract", schema=schema)

        assert "response_schema" not in models.kwargs["config"]
        assert "plan" in models.kwargs["contents"]

    async def test_a_pydantic_schema_is_not_restated_in_the_prompt(self) -> None:
        """A second copy of a large schema is a second thing to fall out of sync."""
        models = _GeminiModels('{"plan": "Pro", "price": "$99"}')
        provider = GeminiLLMProvider("g-test", client=_GeminiClient("{}"))
        provider._client.models = models

        await provider.generate("extract", schema=Pricing)

        assert models.kwargs["contents"] == "extract"

    async def test_the_system_prompt_goes_in_the_config(self) -> None:
        models = _GeminiModels("{}")
        provider = GeminiLLMProvider("g-test", client=_GeminiClient("{}"))
        provider._client.models = models

        await provider.generate("user", system="system")

        assert models.kwargs["config"]["system_instruction"] == "system"

    async def test_a_blank_key_is_refused_at_construction(self) -> None:
        with pytest.raises(ProviderError, match="GEMINI_API_KEY"):
            GeminiLLMProvider("  ", client=_GeminiClient("{}"))


class TestModelRoles:
    @pytest.mark.parametrize(("build", "inner"), LLM_ADAPTERS)
    async def test_a_role_selects_its_configured_model(
        self, build: Any, inner: Any
    ) -> None:
        provider = build("{}")
        provider.set_profiles({"fast": "small", "reasoning": "big"})

        assert provider.model_for("fast") == "small"
        assert provider.model_for("reasoning") == "big"
        assert provider.model_for("final") == provider._default_model

    async def test_the_openai_role_reaches_the_request(self) -> None:
        client = _OpenAIClient()
        provider = OpenAILLMProvider("sk-test", client=client, default_model="base")
        provider.set_profiles({"reasoning": "big-model"})

        await provider.generate("hi", model_role="reasoning")

        assert client.kwargs["model"] == "big-model"


class TestEmbeddingAdapter:
    async def test_vectors_come_back_in_input_order(self) -> None:
        client = _EmbeddingClient([[0.1, 0.2], [0.3, 0.4]])
        provider = OpenAIEmbeddingProvider("sk-test", client=client)

        assert await provider.embed_documents(["a", "b"]) == [[0.1, 0.2], [0.3, 0.4]]

    async def test_a_miscounted_response_is_refused(self) -> None:
        """One vector for two inputs would silently misalign every chunk.

        A vector returned for the wrong chunk is worse than no vector at all: the
        retrieval still runs, and it ranks by an unrelated embedding.
        """
        client = _EmbeddingClient([[0.1, 0.2]])
        provider = OpenAIEmbeddingProvider("sk-test", client=client)

        with pytest.raises(ProviderError, match="number of inputs"):
            await provider.embed_documents(["a", "b"])

    async def test_a_shuffled_response_is_reordered_by_index(self) -> None:
        """The API does not promise order, and a misordered corpus misranks."""
        client = _EmbeddingClient([[0.3, 0.4], [0.1, 0.2]], indices=[1, 0])
        provider = OpenAIEmbeddingProvider("sk-test", client=client)

        assert await provider.embed_documents(["a", "b"]) == [[0.1, 0.2], [0.3, 0.4]]

    async def test_a_query_returns_one_vector(self) -> None:
        provider = OpenAIEmbeddingProvider("sk-test", client=_EmbeddingClient([[0.1]]))
        assert await provider.embed_query("a") == [0.1]

    async def test_no_texts_makes_no_request(self) -> None:
        """A caller with nothing to index should not be charged for a call."""
        client = _EmbeddingClient([])
        provider = OpenAIEmbeddingProvider("sk-test", client=client)

        assert await provider.embed_documents([]) == []
        assert client.calls == 0

    async def test_a_blank_key_is_refused_at_construction(self) -> None:
        with pytest.raises(ProviderError, match="OPENAI_API_KEY"):
            OpenAIEmbeddingProvider("", client=_EmbeddingClient([]))


class _Embedding:
    def __init__(self, embedding: list[float], index: int) -> None:
        self.embedding = embedding
        self.index = index


class _EmbeddingClient:
    def __init__(self, vectors: list[list[float]], indices: list[int] | None = None) -> None:
        self.vectors = vectors
        self.indices = indices or list(range(len(vectors)))
        self.calls = 0
        self.embeddings = self

    async def create(self, **kwargs: Any) -> Any:
        self.calls += 1
        data = [
            _Embedding(vector, index)
            for vector, index in zip(self.vectors, self.indices, strict=True)
        ]
        return type("Resp", (), {"data": data})()


class TestSearchAdapter:
    async def test_results_become_hits_with_their_rank_preserved(self) -> None:
        client = _TavilyClient(
            {
                "results": [
                    {"url": "https://b.example", "title": "B", "content": "bee"},
                    {"url": "https://a.example", "title": "A", "content": "ay"},
                ]
            }
        )
        hits = await TavilySearchProvider("tv-test", client=client).search("q")

        assert [hit.url for hit in hits] == ["https://b.example", "https://a.example"]
        assert [hit.rank for hit in hits] == [1, 2]
        assert hits[0].provider == "tavily"

    async def test_a_result_with_no_url_is_dropped(self) -> None:
        """A hit that cannot be fetched cannot become a source."""
        client = _TavilyClient(
            {"results": [{"url": "", "title": "x", "content": "y"}]}
        )
        assert await TavilySearchProvider("tv-test", client=client).search("q") == []

    async def test_an_empty_result_set_is_not_an_error(self) -> None:
        """No results is a legitimate answer; only the run's coverage suffers."""
        client = _TavilyClient({"results": []})
        assert await TavilySearchProvider("tv-test", client=client).search("q") == []

    async def test_a_provider_fault_names_the_query(self) -> None:
        client = _TavilyClient({}, error=RuntimeError("rate limited"))
        with pytest.raises(ProviderError, match="rate limited"):
            await TavilySearchProvider("tv-test", client=client).search("acme cloud")

    async def test_a_blank_key_is_refused_at_construction(self) -> None:
        with pytest.raises(ProviderError, match="TAVILY_API_KEY"):
            TavilySearchProvider("", client=_TavilyClient({}))


class _TavilyClient:
    def __init__(self, payload: dict[str, Any], error: Exception | None = None) -> None:
        self.payload = payload
        self.error = error
        self.kwargs: dict[str, Any] = {}

    async def search(self, **kwargs: Any) -> dict[str, Any]:
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.payload
