"""The contract all nine analysts inherit, tested once for all of them.

Two of the plan's required failures live here, and both are about the same thing:
an analyst's output is the last point at which the retrieved set is known, so
every check that depends on it has to happen inside the analyst.

* **A citation to a source the analyst never retrieved is stripped**, and the
  run is marked partial with a warning naming it. The alternative — persisting it
  — produces a report citing a URL that was never fetched, which no later stage
  can detect because every later stage trusts the analyst.
* **A payload asserting a number with no evidence is refused.** For all nine
  analysts, not just the ones where a number looks risky.

The tests run the checks across all nine because a check that only the pricing
analyst applies is not a contract, it is a coincidence. Anything parameterised
over `ANALYSTS` here would pass even if eight subclasses overrode it.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from app.agents import ANALYSTS, ANALYSTS_BY_CATEGORY, analyst_for
from app.agents.analysis_base import (
    NO_EVIDENCE,
    NO_LLM,
    SCHEMA_ERROR,
    UNPARSABLE,
    UNSUPPORTED_CITATION,
    UNSUPPORTED_NUMBER,
    CategoryAnalyst,
    numeric_leaves,
    validate_citations,
    validate_payload_numbers,
)
from app.agents.base import AgentContext, AgentStatus
from app.core.constants import PageType
from app.providers.base import LLMProvider
from app.schemas.document import Document
from app.schemas.provider import LLMResponse
from app.services.rag_service import RagService

CONTEXT = AgentContext(company="Acme", research_goal="compare pricing")


def _corpus() -> list[Document]:
    return [
        Document(
            id="acme-pricing",
            run_id="run-1",
            source_id="src-acme",
            url="https://acme.example/pricing",
            canonical_url="https://acme.example/pricing",
            company="Acme",
            title="Acme pricing",
            page_type=PageType.PRICING,
            content=(
                "# Pricing\n\n## Starter\n\n"
                "Starter plans start at $10 per month for one seat."
            ),
            retrieved_at=datetime.now(UTC),
        ),
        Document(
            id="beta-pricing",
            run_id="run-1",
            source_id="src-beta",
            url="https://beta.example/pricing",
            canonical_url="https://beta.example/pricing",
            competitor="Beta",
            title="Beta pricing",
            page_type=PageType.PRICING,
            content=(
                "# Pricing\n\n## Starter\n\n"
                "Beta starter pricing begins at $20 per month."
            ),
            retrieved_at=datetime.now(UTC),
        ),
    ]


def _rag() -> RagService:
    service = RagService()
    service.index(_corpus())
    return service


class StubLLM(LLMProvider):
    """Returns canned content, recording what it was asked."""

    name = "stub"

    def __init__(self, content: str) -> None:
        self._content = content
        self.prompts: list[str] = []
        self.systems: list[str | None] = []
        self.schemas: list[object] = []

    async def generate(self, prompt, *, system=None, schema=None, **kwargs) -> LLMResponse:
        self.prompts.append(prompt)
        self.systems.append(system)
        self.schemas.append(schema)
        return LLMResponse(content=self._content, model="stub", provider="stub")


def _analyst(cls, *, llm=None, entity: str = "Acme") -> CategoryAnalyst:
    return cls(rag=_rag(), llm=llm, entity=entity)


def _payload(analyst, **overrides) -> dict:
    """A minimal schema-valid payload for one analyst.

    Built by introspection rather than written out, because the nine schemas do
    not agree on their required fields: the website analysis keys the company as
    ``name``, the market analysis requires nothing at all, and all nine forbid
    extra keys. A payload written for one of them is therefore not valid for the
    other eight — which is the assumption behind a version of this helper that
    passed six analysts and failed the seventh for no reason anyone wanted.
    """
    fields = analyst.output_schema.model_fields
    payload = {name: "Acme" for name, field in fields.items() if field.is_required()}
    if "confidence" in fields and "confidence" not in payload:
        payload["confidence"] = "low"
    if "evidence" in fields:
        payload["evidence"] = []
    payload.update(overrides)
    return payload


def _json(payload: dict) -> str:
    return json.dumps(payload)


class TestTheNineAnalysts:
    def test_there_are_exactly_nine(self) -> None:
        assert len(ANALYSTS) == 9

    def test_every_analyst_declares_a_category_prompt_queries_and_schema(self) -> None:
        for analyst in ANALYSTS:
            assert analyst.category
            assert analyst.prompt_name
            assert analyst.retrieval_queries
            assert analyst.output_schema is not None

    def test_every_analyst_prompt_is_registered(self) -> None:
        from app.prompts import names

        registered = set(names())
        for analyst in ANALYSTS:
            assert analyst.prompt_name in registered, analyst.name

    def test_every_analyst_schema_matches_its_prompt(self) -> None:
        """The analyst must validate into the same shape the prompt demanded."""
        from app.prompts import get

        for analyst in ANALYSTS:
            assert get(analyst.prompt_name).OUTPUT_SCHEMA is analyst.output_schema

    def test_categories_are_unique(self) -> None:
        categories = [analyst.category for analyst in ANALYSTS]
        assert len(set(categories)) == len(categories)

    def test_every_analyst_can_be_looked_up_by_category(self) -> None:
        for category in ANALYSTS_BY_CATEGORY:
            assert analyst_for(category) is ANALYSTS_BY_CATEGORY[category]

    def test_an_unknown_category_raises_with_the_real_ones_listed(self) -> None:
        with pytest.raises(KeyError, match="pricing"):
            analyst_for("not-a-category")

    def test_no_analyst_overrides_the_verification(self) -> None:
        """The checks live on the base; a subclass cannot opt out of them."""
        for analyst in ANALYSTS:
            assert "verify_and_build" not in vars(analyst)
            assert "validate_payload_numbers" not in vars(analyst)
            assert "validate_citations" not in vars(analyst)

    def test_a_missing_declaration_is_an_import_error(self) -> None:
        with pytest.raises(TypeError, match="must declare"):

            class Incomplete(CategoryAnalyst):
                category = "incomplete"

    def test_retrieval_queries_are_not_empty_strings(self) -> None:
        for analyst in ANALYSTS:
            assert all(query.strip() for query in analyst.retrieval_queries)


class TestUnsupportedCitation:
    """The plan's third required failure."""

    def test_a_citation_outside_the_retrieved_set_is_stripped(self) -> None:
        payload = _payload(ANALYSTS_BY_CATEGORY["pricing"])
        payload["evidence"] = ["src-acme", "src-never-retrieved"]
        removed = validate_citations(payload, {"src-acme"})
        assert removed == ["src-never-retrieved"]
        assert payload["evidence"] == ["src-acme"]

    def test_the_analyst_reports_partial_with_a_named_warning(self) -> None:
        analyst = _analyst(ANALYSTS_BY_CATEGORY["pricing"])
        payload = _payload(ANALYSTS_BY_CATEGORY["pricing"])
        payload["evidence"] = ["src-acme", "src-invented"]
        result = analyst.verify_and_build(payload, {"src-acme"})
        assert result.status is AgentStatus.PARTIAL
        assert any(UNSUPPORTED_CITATION in w for w in result.warnings)
        assert "src-invented" in " ".join(result.warnings)

    def test_the_stripped_id_does_not_survive_into_the_payload(self) -> None:
        """Stripping and reporting must happen in the same pass.

        Cleaning the payload but not reporting it hides a systematic problem;
        reporting without cleaning leaves the fabricated id in the data.
        """
        analyst = _analyst(ANALYSTS_BY_CATEGORY["pricing"])
        payload = _payload(ANALYSTS_BY_CATEGORY["pricing"])
        payload["evidence"] = ["src-invented"]
        result = analyst.verify_and_build(payload, {"src-acme"})
        assert result.payload.evidence == []
        assert any(UNSUPPORTED_CITATION in w for w in result.warnings)

    def test_a_wholly_invented_citation_leaves_nothing_and_still_warns(self) -> None:
        analyst = _analyst(ANALYSTS_BY_CATEGORY["pricing"])
        payload = _payload(ANALYSTS_BY_CATEGORY["pricing"])
        payload["evidence"] = ["made-up"]
        result = analyst.verify_and_build(payload, {"src-acme"})
        assert result.payload.evidence == []
        assert any(UNSUPPORTED_CITATION in w for w in result.warnings)
        assert any(NO_EVIDENCE in w for w in result.warnings)

    def test_a_genuine_citation_is_left_alone(self) -> None:
        analyst = _analyst(ANALYSTS_BY_CATEGORY["pricing"])
        payload = _payload(ANALYSTS_BY_CATEGORY["pricing"])
        payload["evidence"] = ["src-acme"]
        result = analyst.verify_and_build(payload, {"src-acme"})
        assert result.status is AgentStatus.OK
        assert result.payload.evidence == ["src-acme"]
        assert result.warnings == []

    def test_every_analyst_strips_an_invented_citation(self) -> None:
        """All nine, not the one where it is easiest to imagine."""
        for analyst in ANALYSTS:
            instance = _analyst(analyst)
            payload = _payload(analyst)
            payload["evidence"] = ["src-invented"]
            result = instance.verify_and_build(payload, {"src-acme"})
            assert result.status is AgentStatus.PARTIAL, analyst.name
            assert any(
                UNSUPPORTED_CITATION in w for w in result.warnings
            ), analyst.name
            assert not result.payload.evidence, analyst.name

    def test_a_companions_own_source_is_not_a_citation_for_a_rival(self) -> None:
        """Retrieval is scoped per entity, so Beta's pages cannot cite Acme's."""
        beta = _analyst(ANALYSTS_BY_CATEGORY["pricing"], entity="Beta")
        found = beta.retrieve()
        assert {chunk.source_id for chunk in found} == {"src-beta"}
        payload = _payload(ANALYSTS_BY_CATEGORY["pricing"])
        payload["evidence"] = ["src-acme"]
        result = beta.verify_and_build(payload, {"src-beta"})
        assert any(UNSUPPORTED_CITATION in w for w in result.warnings)


class TestNumbersNeedEvidence:
    """The plan's fourth required failure."""

    def test_a_number_with_no_evidence_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no evidence"):
            validate_payload_numbers({"company": "Acme", "evidence": [], "count": 3})

    def test_evidence_licenses_a_number(self) -> None:
        validate_payload_numbers({"company": "Acme", "evidence": ["src-acme"], "count": 3})

    def test_the_check_reaches_into_nested_structures(self) -> None:
        with pytest.raises(ValueError, match="plans"):
            validate_payload_numbers(
                {"company": "Acme", "evidence": [], "plans": [{"price": 10}]}
            )

    def test_a_boolean_is_held_to_the_same_rule(self) -> None:
        """A free tier inferred from silence is a guess like any other."""
        assert numeric_leaves({"has_free_tier": True}) == ["has_free_tier"]
        with pytest.raises(ValueError, match="has_free_tier"):
            validate_payload_numbers({"has_free_tier": True, "evidence": []})

    def test_key_factual_keys_are_exempt(self) -> None:
        """`company` labels the analysis; it does not assert anything."""
        validate_payload_numbers({"company": "Acme", "market": "cloud", "evidence": []})

    def test_every_analyst_refuses_an_unevidenced_number(self) -> None:
        for analyst in ANALYSTS:
            instance = _analyst(analyst)
            result = instance.verify_and_build(
                {"company": "Acme", "evidence": [], "made_up_number": 7},
                {"src-acme"},
            )
            assert result.status is AgentStatus.FAILED, analyst.name
            assert UNSUPPORTED_NUMBER in result.warnings, analyst.name

    def test_a_refused_number_names_the_field(self) -> None:
        analyst = _analyst(ANALYSTS_BY_CATEGORY["pricing"])
        result = analyst.verify_and_build(
            {"company": "Acme", "evidence": [], "invented": 99}, {"src-acme"}
        )
        assert "invented" in " ".join(result.errors)


class TestTheAnalysedEntity:
    def test_retrieval_is_scoped_to_the_analysts_entity(self) -> None:
        acme = _analyst(ANALYSTS_BY_CATEGORY["pricing"], entity="Acme")
        beta = _analyst(ANALYSTS_BY_CATEGORY["pricing"], entity="Beta")
        assert {c.source_id for c in acme.retrieve()} == {"src-acme"}
        assert {c.source_id for c in beta.retrieve()} == {"src-beta"}

    def test_the_prompt_names_the_company_under_analysis(self) -> None:
        beta = _analyst(ANALYSTS_BY_CATEGORY["pricing"], entity="Beta")
        messages = beta.build_messages(CONTEXT, beta.retrieve())
        assert "Beta" in messages[0].content or "Beta" in messages[1].content

    def test_a_rival_is_listed_as_a_competitor_not_as_the_company(self) -> None:
        beta = _analyst(ANALYSTS_BY_CATEGORY["pricing"], entity="Beta")
        context = beta.prompt_context(CONTEXT, beta.retrieve())
        assert context.company == "Beta"
        assert context.competitors == ["Beta"]

    def test_the_target_is_not_listed_as_its_own_competitor(self) -> None:
        acme = _analyst(ANALYSTS_BY_CATEGORY["pricing"], entity="Acme")
        assert acme.prompt_context(CONTEXT, acme.retrieve()).competitors == []


class TestTheRun:
    async def test_a_clean_analysis_is_ok(self) -> None:
        payload = _payload(ANALYSTS_BY_CATEGORY["pricing"])
        payload["evidence"] = ["src-acme"]
        result = await _analyst(
            ANALYSTS_BY_CATEGORY["pricing"], llm=StubLLM(_json(payload))
        ).run(CONTEXT)
        assert result.status is AgentStatus.OK
        assert result.payload is not None

    async def test_the_model_is_handed_the_schema_and_the_retrieved_text(self) -> None:
        llm = StubLLM('{"company": "Acme", "evidence": []}')
        analyst = _analyst(ANALYSTS_BY_CATEGORY["pricing"], llm=llm)
        await analyst.run(CONTEXT)
        assert llm.schemas[0] is analyst.output_schema
        assert "Starter plans start at $10" in llm.prompts[0]
        assert "src-acme" in llm.prompts[0]

    async def test_a_competitor_pricing_page_cannot_reach_an_acme_run(self) -> None:
        """The whole point of per-entity retrieval, asserted end to end."""
        llm = StubLLM('{"company": "Acme", "evidence": ["src-beta"]}')
        analyst = _analyst(ANALYSTS_BY_CATEGORY["pricing"], llm=llm, entity="Acme")
        result = await analyst.run(CONTEXT)
        assert any(UNSUPPORTED_CITATION in w for w in result.warnings)
        assert "src-beta" not in " ".join(llm.prompts[0])

    async def test_prose_instead_of_json_is_a_failure_not_a_crash(self) -> None:
        llm = StubLLM("Acme's starter plan costs about ten dollars a month.")
        result = await _analyst(
            ANALYSTS_BY_CATEGORY["pricing"], llm=llm
        ).run(CONTEXT)
        assert result.status is AgentStatus.FAILED
        assert UNPARSABLE in result.warnings

    async def test_a_fenced_json_block_is_accepted(self) -> None:
        """Providers routinely wrap JSON; the parser must not be brittle."""
        payload = _payload(ANALYSTS_BY_CATEGORY["pricing"])
        payload["evidence"] = ["src-acme"]
        result = await _analyst(
            ANALYSTS_BY_CATEGORY["pricing"],
            llm=StubLLM(f"```json\n{_json(payload)}\n```"),
        ).run(CONTEXT)
        assert result.status is AgentStatus.OK

    async def test_a_payload_the_schema_rejects_is_a_stated_failure(self) -> None:
        payload = _payload(ANALYSTS_BY_CATEGORY["pricing"])
        payload["evidence"] = ["src-acme"]
        payload["plans"] = 7
        result = await _analyst(
            ANALYSTS_BY_CATEGORY["pricing"], llm=StubLLM(_json(payload))
        ).run(CONTEXT)
        assert result.status is AgentStatus.FAILED
        assert SCHEMA_ERROR in result.warnings

    async def test_no_evidence_is_partial_rather_than_failed(self) -> None:
        """The category is genuinely not publicly available: that is a finding.

        Gamma is absent from the corpus, so its analyst retrieves nothing. A
        well-formed answer with no evidence behind it is a legitimate result —
        the alternative, failing, would report a gap in public information as a
        failure of the system.
        """
        empty = _payload(ANALYSTS_BY_CATEGORY["pricing"])
        result = await _analyst(
            ANALYSTS_BY_CATEGORY["pricing"], llm=StubLLM(_json(empty)), entity="Gamma"
        ).run(AgentContext(company="Gamma", research_goal="compare pricing"))
        assert result.status is AgentStatus.PARTIAL
        assert result.succeeded
        assert any(NO_EVIDENCE in w for w in result.warnings)
        assert result.retrieved_source_ids == []

    async def test_no_llm_analyses_nothing_but_says_so(self) -> None:
        result = await _analyst(ANALYSTS_BY_CATEGORY["pricing"]).run(CONTEXT)
        assert NO_LLM in result.warnings
        assert result.payload is None

    async def test_the_result_carries_its_permitted_sources(self) -> None:
        payload = _payload(ANALYSTS_BY_CATEGORY["pricing"])
        payload["evidence"] = ["src-acme"]
        result = await _analyst(
            ANALYSTS_BY_CATEGORY["pricing"], llm=StubLLM(_json(payload))
        ).run(CONTEXT)
        assert result.retrieved_source_ids == ["src-acme"]


class TestPassageConversion:
    def test_chunks_become_passages_that_keep_their_source_id(self) -> None:
        """The source id is the model's only handle for citing, and the only
        handle the citation check has for verifying it."""
        analyst = _analyst(ANALYSTS_BY_CATEGORY["pricing"])
        passages = analyst.to_passages(analyst.retrieve())
        assert passages
        assert passages[0].source_id == "src-acme"
        assert passages[0].url.startswith("https://acme.example")

    def test_a_passage_is_not_an_evidence_record(self) -> None:
        """Presenting retrieved text as evidence would let a report cite a chunk
        nobody has read."""
        analyst = _analyst(ANALYSTS_BY_CATEGORY["pricing"])
        context = analyst.prompt_context(CONTEXT, analyst.retrieve())
        assert context.passages
        assert context.evidence == []
        rendered = context.evidence_block()
        assert "[P1]" in rendered
        assert "[E1]" not in rendered

    def test_a_chunk_cannot_launder_its_own_source_type(self) -> None:
        """Source authority belongs to the source, not to the text sliced from it."""
        analyst = _analyst(ANALYSTS_BY_CATEGORY["pricing"])
        chunks = analyst.retrieve()
        chunks[0].metadata["source_type"] = "official_site"
        passages = analyst.to_passages(chunks)
        assert passages[0].source_type is not None

    def test_an_unrecognised_source_type_is_left_unknown(self) -> None:
        analyst = _analyst(ANALYSTS_BY_CATEGORY["pricing"])
        chunks = analyst.retrieve()
        chunks[0].metadata["source_type"] = "not_a_real_type"
        assert analyst.to_passages(chunks)[0].source_type is None
