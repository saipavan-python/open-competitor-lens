"""Retrieval: what an analyst can see, and what it is therefore allowed to say.

The two properties the plan requires are here — a metadata filter can never leak
another company's chunk, and chunking never separates a heading from its text or
drops provenance. The rest of this file tests the things that make those two
true in practice rather than by accident: that the filter runs *before* ranking,
that an unmatched filter returns nothing rather than something close, and that a
missing metadata key excludes a chunk instead of admitting it.

That last one is the one that bites. A permissive filter looks correct in every
happy-path test and then returns a competitor's pricing page as the target's,
because the target's chunks were the ones that never recorded a `category`.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.core.constants import PageType
from app.rag.chunking import chunk_document, chunk_documents
from app.rag.indexing import build_index, tokenize
from app.rag.metadata import UnknownFilterKeyError, matches, matches_all, validate_filters
from app.rag.reranking import rerank
from app.rag.retrieval import all_sources, candidate_positions, retrieve
from app.schemas.document import Document
from app.services.rag_service import RagService


def _document(
    doc_id: str,
    *,
    company: str | None = None,
    competitor: str | None = None,
    category: str | None = None,
    content: str = "",
    title: str | None = None,
    page_type: PageType = PageType.OTHER,
    source_id: str | None = None,
) -> Document:
    metadata = {"category": category} if category else {}
    return Document(
        id=doc_id,
        run_id="run-1",
        source_id=source_id or f"src-{doc_id}",
        url=f"https://{company or competitor}.example/{doc_id}",
        canonical_url=f"https://{company or competitor}.example/{doc_id}",
        company=company,
        competitor=competitor,
        title=title or f"Page {doc_id}",
        page_type=page_type,
        content=content,
        metadata=metadata,
        retrieved_at=datetime.now(UTC),
    )


def _corpus() -> list[Document]:
    return [
        _document(
            "acme-pricing",
            company="Acme",
            category="pricing",
            title="Acme pricing",
            page_type=PageType.PRICING,
            content=(
                "# Pricing\n\n"
                "## Starter\n\n"
                "Starter plans start at $10 per month for one seat. "
                "Enterprise pricing is available on request.\n\n"
                "## Pro\n\n"
                "Pro costs $49 per seat per month and includes SSO."
            ),
        ),
        _document(
            "beta-pricing",
            competitor="Beta",
            category="pricing",
            title="Beta pricing",
            page_type=PageType.PRICING,
            content=(
                "# Pricing\n\n"
                "## Starter\n\n"
                "Beta starter pricing begins at $20 per month for one seat. "
                "No enterprise tier is offered."
            ),
        ),
        _document(
            "acme-about",
            company="Acme",
            category="website",
            title="Acme about",
            page_type=PageType.ABOUT,
            content=(
                "# About Acme\n\n"
                "Acme was founded in 2011 and builds deployment tooling for "
                "platform engineering teams."
            ),
        ),
    ]


class TestMetadataFilters:
    """The plan's first required failure, stated several ways."""

    def test_a_competitor_filter_never_returns_another_competitor(self) -> None:
        index = build_index(chunk_documents(_corpus()))
        results = retrieve(
            index, "pricing plans", filters={"competitor": "Beta", "category": "pricing"}
        )
        assert results
        assert {chunk.competitor for chunk in results} == {"Beta"}

    def test_a_company_filter_never_returns_a_competitor(self) -> None:
        index = build_index(chunk_documents(_corpus()))
        results = retrieve(index, "pricing", filters={"company": "Acme"})
        assert {chunk.competitor for chunk in results} == {None}

    def test_the_entity_filter_spans_target_and_competitor(self) -> None:
        index = build_index(chunk_documents(_corpus()))
        assert retrieve(index, "pricing", filters={"entity": "Beta"})
        assert retrieve(index, "pricing", filters={"entity": "Acme"})

    def test_a_case_different_filter_still_matches(self) -> None:
        index = build_index(chunk_documents(_corpus()))
        assert retrieve(index, "pricing", filters={"competitor": "beta"})

    def test_a_filter_that_matches_nothing_returns_nothing(self) -> None:
        """Never a top-up of loosely related results from another company."""
        index = build_index(chunk_documents(_corpus()))
        assert retrieve(index, "pricing", filters={"competitor": "Gamma"}) == []

    def test_a_chunk_missing_the_filtered_key_is_excluded(self) -> None:
        """Absent metadata is not a wildcard.

        This is the case that would otherwise let an untagged chunk answer a
        question asked about a specific company.
        """
        index = build_index(chunk_documents(_corpus()))
        for chunk in index.documents:
            chunk.competitor = None
            chunk.company = None
        assert retrieve(index, "pricing", filters={"competitor": "Beta"}) == []

    def test_an_empty_metadata_value_is_not_a_match(self) -> None:
        index = build_index(chunk_documents(_corpus()))
        for chunk in index.documents:
            chunk.metadata["category"] = ""
        assert retrieve(index, "pricing", filters={"category": "pricing"}) == []

    def test_several_filters_must_all_hold(self) -> None:
        index = build_index(chunk_documents(_corpus()))
        assert retrieve(
            index, "pricing", filters={"competitor": "Beta", "category": "website"}
        ) == []

    def test_matched_by_names_the_filters_that_admitted_a_chunk(self) -> None:
        index = build_index(chunk_documents(_corpus()))
        found = retrieve(
            index, "pricing", filters={"competitor": "Beta", "category": "pricing"}
        )
        assert set(found[0].matched_by) == {"competitor", "category"}


class TestFilterValidation:
    def test_an_unknown_filter_key_is_rejected(self) -> None:
        """A typo must fail at the call, not silently match nothing."""
        with pytest.raises(UnknownFilterKeyError, match="not a filterable"):
            validate_filters({"competitr": "Beta"})

    def test_filter_keys_are_normalised(self) -> None:
        assert validate_filters({"  Competitor ": "Beta"}) == {"competitor": "Beta"}

    def test_no_filters_validates_to_an_empty_mapping(self) -> None:
        assert validate_filters(None) == {}
        assert validate_filters({}) == {}

    def test_a_filter_value_is_stringified(self) -> None:
        assert validate_filters({"company": 42}) == {"company": "42"}


class TestFilterBeforeRank:
    def test_filtering_happens_before_any_scoring(self) -> None:
        """Asserted on the candidate set, not inferred from the output.

        If ranking ran first, a rival's richer pricing page would take the top
        slots and the filter would then discard them, reporting "not publicly
        available" while the right chunks were sitting in the discarded set.
        """
        index = build_index(chunk_documents(_corpus()))
        every = candidate_positions(index, None)
        filtered = candidate_positions(index, {"competitor": "Beta"})
        assert len(filtered) < len(every)
        assert all(index.documents[p].competitor == "Beta" for p in filtered)

    def test_a_rival_with_a_richer_page_does_not_starve_the_target(self) -> None:
        index = build_index(chunk_documents(_corpus()))
        results = retrieve(index, "pricing per month", filters={"company": "Acme"})
        assert results
        assert any("Starter" in chunk.text for chunk in results)

    def test_k_caps_the_result(self) -> None:
        index = build_index(chunk_documents(_corpus()))
        assert len(retrieve(index, "pricing", k=1)) == 1

    def test_a_zero_k_returns_nothing(self) -> None:
        index = build_index(chunk_documents(_corpus()))
        assert retrieve(index, "pricing", k=0) == []

    def test_an_empty_index_returns_nothing(self) -> None:
        assert retrieve(build_index([]), "pricing") == []

    def test_a_query_sharing_no_terms_returns_nothing(self) -> None:
        """A zero-score chunk is a lexical accident, not a match."""
        index = build_index(chunk_documents(_corpus()))
        assert retrieve(index, "zzzzqqqq nonexistent") == []

    def test_results_are_ordered_by_descending_score(self) -> None:
        index = build_index(chunk_documents(_corpus()))
        scores = [chunk.score for chunk in retrieve(index, "pricing plans seat")]
        assert scores == sorted(scores, reverse=True)

    def test_retrieval_is_deterministic(self) -> None:
        index = build_index(chunk_documents(_corpus()))
        first = [chunk.id for chunk in retrieve(index, "pricing plans")]
        second = [chunk.id for chunk in retrieve(index, "pricing plans")]
        assert first == second


class TestChunking:
    """The plan's second required failure."""

    def test_a_heading_never_ends_up_in_another_chunk(self) -> None:
        content = "# Pricing\n\n" + "\n\n".join(
            f"## Plan {index}\n\nPlan {index} costs ${index} per month. "
            f"It includes seats and support."
            for index in range(12)
        )
        document = _document("long", company="Acme", content=content)
        chunks = chunk_document(document, max_chars=400, overlap=50)
        assert len(chunks) > 1
        for chunk in chunks:
            if chunk.heading:
                assert chunk.heading in chunk.text or chunk.text.startswith("#")

    def test_every_chunk_keeps_its_source_metadata(self) -> None:
        document = _document(
            "acme-pricing", company="Acme", category="pricing", content="x " * 2000
        )
        chunks = chunk_document(document, max_chars=300)
        assert chunks
        for chunk in chunks:
            assert chunk.source_id == document.source_id
            assert chunk.document_id == document.id
            assert chunk.run_id == document.run_id
            assert chunk.competitor == document.competitor
            assert chunk.company == document.company
            assert chunk.metadata == document.metadata
            assert chunk.url == document.canonical_url

    def test_a_chunks_metadata_is_a_copy_not_a_reference(self) -> None:
        """A later mutation of the document must not rewrite indexed meaning."""
        document = _document("acme", company="Acme", category="pricing", content="hello")
        chunk = chunk_document(document)[0]
        document.metadata["category"] = "website"
        assert chunk.metadata["category"] == "pricing"

    def test_chunk_ids_are_unique_and_stable(self) -> None:
        document = _document("acme", company="Acme", content="alpha beta gamma")
        first = [chunk.id for chunk in chunk_document(document)]
        second = [chunk.id for chunk in chunk_document(document)]
        assert first == second
        assert len(set(first)) == len(first)

    def test_ordinals_run_in_order(self) -> None:
        document = _document("acme", company="Acme", content="word " * 3000)
        chunks = chunk_document(document, max_chars=300)
        assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))

    def test_an_empty_document_yields_no_chunks(self) -> None:
        assert chunk_document(_document("empty", company="Acme", content="   ")) == []

    def test_no_chunk_exceeds_the_limit(self) -> None:
        document = _document("acme", company="Acme", content="word " * 3000)
        for chunk in chunk_document(document, max_chars=400):
            assert len(chunk.text) <= 400

    def test_a_chunk_knows_the_entity_it_is_about(self) -> None:
        competitor = _document("beta", competitor="Beta", content="Beta pricing is $20")
        assert chunk_document(competitor)[0].entity == "Beta"


class TestIndexing:
    def test_tokenize_drops_stopwords_and_shortens_case(self) -> None:
        assert tokenize("The Plans and Prices of Acme") == [
            "plans", "prices", "acme",
        ]

    def test_an_identical_chunk_id_replaces_rather_than_duplicates(self) -> None:
        document = _document("acme", company="Acme", content="pricing for acme")
        first = chunk_document(document)
        second = chunk_document(_document("acme", company="Acme", content="pricing for acme"))
        index = build_index(first + second)
        assert index.size == len(first)

    def test_reindexing_replaces_a_changed_chunk(self) -> None:
        index = build_index(chunk_document(_document("acme", company="Acme", content="old text here")))
        index.add(chunk_document(_document("acme", company="Acme", content="brand new text here")))
        assert index.size == 1
        assert "brand new" in index.documents[0].text

    def test_removing_a_chunk_keeps_the_others_retrievable(self) -> None:
        index = build_index(chunk_documents(_corpus()))
        victim = next(c for c in index.documents if c.competitor == "Beta")
        rebuilt = build_index([c for c in index.documents if c.id != victim.id])
        assert retrieve(rebuilt, "pricing", filters={"company": "Acme"})

    def test_an_index_id_is_stable_for_the_same_content(self) -> None:
        chunks = chunk_documents(_corpus())
        assert build_index(chunks).index_id == build_index(chunk_documents(_corpus())).index_id


class TestReranking:
    def test_a_pricing_page_outranks_a_page_merely_mentioning_pricing(self) -> None:
        index = build_index(chunk_documents(_corpus()))
        found = rerank("pricing plans", retrieve(index, "pricing plans"))
        assert found
        assert found[0].page_type is PageType.PRICING

    def test_reranking_preserves_every_chunk(self) -> None:
        """Reordering only. Dropping a chunk here would lose evidence silently."""
        index = build_index(chunk_documents(_corpus()))
        found = retrieve(index, "pricing")
        assert len(rerank("pricing", found)) == len(found)

    def test_reranking_does_not_resurrect_a_filtered_out_chunk(self) -> None:
        index = build_index(chunk_documents(_corpus()))
        found = retrieve(index, "pricing", filters={"company": "Acme"})
        assert {chunk.competitor for chunk in rerank("pricing", found)} == {None}

    def test_reranking_is_deterministic(self) -> None:
        index = build_index(chunk_documents(_corpus()))
        found = retrieve(index, "pricing")
        assert [c.id for c in rerank("pricing", found)] == [
            c.id for c in rerank("pricing", found)
        ]


class TestAllSources:
    def test_all_sources_deduplicates_and_sorts(self) -> None:
        index = build_index(chunk_documents(_corpus()))
        found = retrieve(index, "pricing")
        assert all_sources(found) == sorted(set(all_sources(found)))

    def test_all_sources_is_empty_for_no_chunks(self) -> None:
        assert all_sources([]) == []


class TestRagService:
    def test_indexing_returns_an_id_and_reports_its_size(self) -> None:
        service = RagService()
        index_id = service.index(_corpus())
        assert index_id
        assert service.size > 0
        assert not service.is_empty

    def test_retrieval_before_indexing_returns_nothing(self) -> None:
        """A missing index is not an error: it is an empty category."""
        assert RagService().retrieve("pricing") == []

    def test_retrieve_applies_filters(self) -> None:
        service = RagService()
        service.index(_corpus())
        found = service.retrieve("pricing", filters={"competitor": "Beta"})
        assert {chunk.competitor for chunk in found} == {"Beta"}

    def test_retrieve_many_merges_without_duplicating_a_chunk(self) -> None:
        service = RagService()
        service.index(_corpus())
        found = service.retrieve_many(["pricing", "pricing plans", "seat pricing"])
        assert len({chunk.id for chunk in found}) == len(found)

    def test_sources_for_is_the_analysts_permitted_set(self) -> None:
        service = RagService()
        service.index(_corpus())
        found = service.retrieve("pricing", filters={"competitor": "Beta"})
        assert service.sources_for(found) == ["src-beta-pricing"]

    def test_reindexing_replaces_rather_than_accumulates(self) -> None:
        """A stale page from an earlier phase must not answer about a re-crawl."""
        service = RagService()
        service.index(_corpus())
        first = service.size
        service.index(_corpus())
        assert service.size == first

    def test_chunks_exposes_the_indexed_pieces(self) -> None:
        service = RagService()
        service.index(_corpus())
        assert all(chunk.text for chunk in service.chunks())

    def test_reranking_can_be_switched_off(self) -> None:
        service = RagService()
        service.index(_corpus())
        assert service.retrieve("pricing", rerank=False) is not None


class TestMatchesHelpers:
    def test_matches_compares_case_insensitively(self) -> None:
        index = build_index(chunk_documents(_corpus()))
        beta = next(c for c in index.documents if c.competitor == "Beta")
        assert matches(beta, "competitor", "BETA")

    def test_matches_all_reports_no_keys_on_failure(self) -> None:
        index = build_index(chunk_documents(_corpus()))
        acme = next(c for c in index.documents if c.company == "Acme")
        assert matches_all(acme, {"competitor": "Beta"}) == (False, [])
