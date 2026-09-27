"""The offline providers, exercised by calling them.

Found while reviewing the branch before publishing. The offline suite consisted
of four ``isinstance`` assertions - the class was the right type and nothing more.
Every method was untested, and two of them did not work:

- ``OfflineSearchProvider.add`` assigned ``hit.metadata`` on a ``SearchHit``, which
  is ``extra="forbid"`` with ``validate_assignment=True``. Every call raised
  ``ValidationError``, so an offline search result could not be registered at all.
- ``_synthesize`` put ``_offline`` and ``_source`` into the generated JSON, and
  every domain model is ``extra="forbid"``, so ``parse_structured`` rejected every
  synthesized answer. The offline LLM could not produce a single parseable result.

The consequence was that the pipeline these providers exist to make testable had
never actually run. ``isinstance`` passing tells you the constructor worked; only
calling the method tells you the provider does.

The rule this file follows: assert on behaviour, not on shape. Where a provider is
supposed to refuse something, the test asserts the refusal. Where it is supposed
to be deterministic, the test calls it twice.
"""

from __future__ import annotations

import asyncio

import pytest

from app.providers.offline import (
    OfflineBrowserProvider,
    OfflineEmbeddingProvider,
    OfflineGraphStore,
    OfflineLLMProvider,
    OfflineSearchProvider,
    OfflineVectorStore,
    OfflineWebProvider,
    cosine_similarity,
)
from app.schemas.company import CompanyProfile
from app.schemas.pricing import PricingAnalysis
from app.schemas.provider import GraphEntity, GraphRelationship, RetrievedChunk
from app.utils.structured import parse_structured


def run(coro: object) -> object:
    """Drive one coroutine, so the tests read as sequences of calls."""
    return asyncio.run(coro)  # type: ignore[arg-type]


class TestOfflineSearchCanActuallyBeRegistered:
    def test_a_result_can_be_added(self) -> None:
        """The call that raised on every invocation before the fix."""
        provider = OfflineSearchProvider()
        hit = provider.add(
            "https://acme.com/pricing", title="Acme pricing", terms=["acme", "pricing"]
        )
        assert hit.url == "https://acme.com/pricing"
        assert hit.domain == "acme.com"

    def test_registered_terms_decide_the_ranking(self) -> None:
        provider = OfflineSearchProvider()
        provider.add("https://acme.com/pricing", terms=["acme", "pricing"])
        provider.add("https://beta.io", terms=["beta"])
        found = run(provider.search("acme pricing plans"))
        assert [hit.url for hit in found] == ["https://acme.com/pricing"]
        assert found[0].rank == 1

    def test_an_unregistered_query_returns_nothing(self) -> None:
        """A search must never manufacture a source - the offline path included."""
        provider = OfflineSearchProvider()
        provider.add("https://acme.com", terms=["acme"])
        assert run(provider.search("something else entirely")) == []

    def test_hits_are_ranked_and_deterministic(self) -> None:
        provider = OfflineSearchProvider()
        for url, terms in [("https://a.com", ["crm"]), ("https://b.com", ["crm", "pricing"])]:
            provider.add(url, terms=terms)
        first = [hit.url for hit in run(provider.search("crm pricing"))]
        second = [hit.url for hit in run(provider.search("crm pricing"))]
        assert first == second == ["https://b.com", "https://a.com"]


class TestOfflineLLMProducesParseableAnswers:
    @pytest.mark.parametrize("schema", [PricingAnalysis, CompanyProfile])
    def test_a_synthesized_answer_satisfies_its_schema(self, schema: type) -> None:
        """No extra keys: a domain model is ``extra="forbid"`` and rejected them.

        The ``_offline``/``_source`` markers broke every one of these.
        """
        response = run(OfflineLLMProvider().generate("a prompt", schema=schema))
        parsed = parse_structured(response.content, schema)
        assert isinstance(parsed, schema)

    def test_provenance_is_carried_in_the_envelope_not_the_payload(self) -> None:
        """The marker moved out of the JSON, so it must still be observable."""
        response = run(OfflineLLMProvider().generate("a prompt", schema=PricingAnalysis))
        assert response.provider == "offline"
        assert response.model.startswith("offline-")
        assert "_offline" not in response.content

    def test_a_synthesized_string_can_never_pass_for_retrieved_text(self) -> None:
        """A fixture that looked like real evidence would be worse than none."""
        response = run(OfflineLLMProvider().generate("a prompt", schema=PricingAnalysis))
        parsed = parse_structured(response.content, PricingAnalysis)
        rendered = parsed.model_dump_json()
        assert "[offline-fixture]" in rendered

    def test_with_no_schema_the_answer_says_it_is_a_fixture(self) -> None:
        response = run(OfflineLLMProvider().generate("a prompt"))
        assert response.content.startswith("[offline-fixture]")

    def test_token_counts_are_reported(self) -> None:
        response = run(OfflineLLMProvider().generate("x" * 400, schema=PricingAnalysis))
        assert response.prompt_tokens > 0
        assert response.completion_tokens > 0
        assert response.total_tokens == response.prompt_tokens + response.completion_tokens

    def test_a_fixed_answer_is_returned_verbatim(self) -> None:
        """What lets a test drive a specific branch of a caller."""
        provider = OfflineLLMProvider(
            content='{"company": "Acme", "pricing_model": "per seat"}'
        )
        response = run(provider.generate("a prompt", schema=PricingAnalysis))
        parsed = parse_structured(response.content, PricingAnalysis)
        assert parsed.company == "Acme"
        assert parsed.pricing_model == "per seat"

    def test_every_prompt_is_recorded(self) -> None:
        provider = OfflineLLMProvider()
        run(provider.generate("first prompt"))
        run(provider.generate("second prompt"))
        assert provider.prompts == ["first prompt", "second prompt"]


class TestOfflineWebServesFixturesOnly:
    def test_a_registered_page_is_returned(self) -> None:
        provider = OfflineWebProvider()
        provider.add("https://acme.com/", text="Acme sells a CRM.")
        result = run(provider.fetch("https://acme.com/"))
        assert result.ok
        assert result.status == 200
        assert "Acme sells a CRM." in result.text

    def test_fixtures_are_matched_on_the_canonical_url(self) -> None:
        """A tracking parameter must not make a registered page look absent."""
        provider = OfflineWebProvider()
        provider.add("https://acme.com/", text="Acme sells a CRM.")
        result = run(provider.fetch("https://acme.com/?utm_source=newsletter"))
        assert result.ok

    def test_an_unregistered_page_is_not_found_not_invented(self) -> None:
        result = run(OfflineWebProvider().fetch("https://acme.com/"))
        assert result.status == 404
        assert result.retrieval_status == "not_found"
        assert not result.ok

    def test_extraction_of_a_missing_page_refuses(self) -> None:
        with pytest.raises(LookupError):
            run(OfflineWebProvider().extract("https://acme.com/", PricingAnalysis))


class TestOfflineBrowserRendersTheWebProvider:
    def test_navigation_returns_the_page(self) -> None:
        web = OfflineWebProvider()
        web.add("https://acme.com/", text="rendered")
        browser = OfflineBrowserProvider(web=web)
        run(browser.open("https://acme.com/"))
        assert run(browser.extract(PricingAnalysis)) is not None

    def test_extracting_with_nothing_open_refuses(self) -> None:
        with pytest.raises(RuntimeError):
            run(OfflineBrowserProvider().extract(PricingAnalysis))

    def test_close_clears_the_current_page(self) -> None:
        browser = OfflineBrowserProvider()
        run(browser.open("https://acme.com/"))
        run(browser.close())
        with pytest.raises(RuntimeError):
            run(browser.extract(PricingAnalysis))


class TestOfflineEmbeddings:
    def test_similar_text_embeds_nearby(self) -> None:
        """The property retrieval depends on, which the test suite assumes."""
        provider = OfflineEmbeddingProvider()
        vectors = run(
            provider.embed_documents(
                ["competitive pricing for a crm plan", "cheap crm pricing tiers"]
            )
        )
        similarity = cosine_similarity(vectors[0], vectors[1])
        assert similarity > 0.3

    def test_unrelated_text_embeds_farther_away(self) -> None:
        provider = OfflineEmbeddingProvider()
        left, right = run(provider.embed_documents(["pricing plan cost", "zebra giraffe"]))
        assert cosine_similarity(left, right) < 1.0

    def test_the_same_text_always_embeds_identically(self) -> None:
        provider = OfflineEmbeddingProvider()
        first = run(provider.embed_query("pricing"))
        second = run(provider.embed_query("pricing"))
        assert first == second

    def test_a_query_and_a_document_share_the_vector_space(self) -> None:
        """A query and a document must be comparable, or retrieval is noise."""
        provider = OfflineEmbeddingProvider()
        as_query = run(provider.embed_query("pricing"))
        as_document = run(provider.embed_documents(["pricing"]))[0]
        assert as_query == as_document

    def test_cosine_similarity_of_a_vector_with_itself_is_one(self) -> None:
        vector = run(OfflineEmbeddingProvider().embed_query("pricing"))
        assert cosine_similarity(vector, vector) == pytest.approx(1.0)

    def test_mismatched_lengths_score_zero_rather_than_raising(self) -> None:
        assert cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0]) == 0.0


class TestOfflineVectorStore:
    def _chunks(self) -> tuple[list[RetrievedChunk], list[list[float]]]:
        embedder = OfflineEmbeddingProvider()
        chunks = [
            RetrievedChunk(id="c1", text="crm pricing plans", source_id="s1"),
            RetrievedChunk(id="c2", text="zebra giraffe habitat", source_id="s2"),
        ]
        return chunks, run(embedder.embed_documents([chunk.text for chunk in chunks]))

    def test_a_stored_chunk_comes_back(self) -> None:
        store = OfflineVectorStore()
        chunks, vectors = self._chunks()
        run(store.upsert("docs", chunks, vectors))
        found = run(store.query("docs", vectors[0], limit=5))
        assert found[0].id == "c1"
        assert found[0].score > 0

    def test_count_reflects_what_was_stored(self) -> None:
        store = OfflineVectorStore()
        chunks, vectors = self._chunks()
        run(store.upsert("docs", chunks, vectors))
        assert run(store.count("docs")) == 2

    def test_collections_are_isolated(self) -> None:
        store = OfflineVectorStore()
        chunks, vectors = self._chunks()
        run(store.upsert("docs", chunks, vectors))
        assert run(store.count("other")) == 0

    def test_deleting_a_collection_empties_it(self) -> None:
        store = OfflineVectorStore()
        chunks, vectors = self._chunks()
        run(store.upsert("docs", chunks, vectors))
        run(store.delete_collection("docs"))
        assert run(store.count("docs")) == 0

    def test_mismatched_lengths_are_refused(self) -> None:
        """A silent zip would drop chunks and make the report quietly incomplete."""
        with pytest.raises(ValueError):
            run(OfflineVectorStore().upsert("docs", self._chunks()[0], [[0.1]]))

    def test_a_query_against_an_empty_store_returns_nothing(self) -> None:
        assert run(OfflineVectorStore().query("nothing", [0.1, 0.2])) == []


class TestOfflineGraphStore:
    def test_an_entity_can_be_found_by_name(self) -> None:
        store = OfflineGraphStore()
        run(store.upsert_entities([GraphEntity(id="p1", name="Product", type="Product")]))
        assert [entity.name for entity in run(store.query("Product"))] == ["Product"]

    def test_upserting_an_entity_twice_merges_its_sources(self) -> None:
        """Losing the earlier sources would quietly break attribution."""
        store = OfflineGraphStore()
        run(store.upsert_entities(
            [GraphEntity(id="p1", name="Product", type="Product", source_ids=["s1"])]
        ))
        run(store.upsert_entities(
            [GraphEntity(id="p1", name="Product", type="Product", source_ids=["s2"])]
        ))
        assert store.entities["p1"].source_ids == ["s1", "s2"]

    def test_a_relationship_is_stored(self) -> None:
        store = OfflineGraphStore()
        run(store.upsert_relationships([
            GraphRelationship(id="r1", type="OFFERS", start_id="c1", end_id="p1")
        ]))
        assert store.relationships["r1"].type == "OFFERS"
