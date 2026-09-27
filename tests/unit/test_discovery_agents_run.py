"""The two discovery agents as the graph will actually call them.

``resolve()`` and ``select()`` are the parts worth testing hard, and they are
tested elsewhere. What is here is the seam between them and the run: whether an
agent that has been handed a context and nothing else can gather what it needs,
and whether a shortfall is reported rather than papered over.

The two failure shapes that matter are the same for both agents. A search or
provider that is down must not raise out of ``run()``, because the graph treats an
exception as "this node is broken" and a missing domain or an empty competitor
pool as "this node ran and found nothing" — and only the second of those is
something the report can honestly state. And a result that is incomplete must
come back ``PARTIAL`` with the count in the warning, so the run's own accounting
of what it did not find is written down by the step that knows.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.agents.base import AgentContext, AgentStatus
from app.agents.company_agent import CompanyAgent
from app.agents.competitor_agent import CandidateSignals, CompetitorAgent, Validator
from app.core.constants import PageType, RetrievalStatus, SourceType
from app.schemas.competitor import CompetitorCandidate, ValidationSignal
from app.schemas.evidence import Source
from app.schemas.search import SearchHit
from app.utils.normalization import company_identity_key


def _hit(url: str, *, title: str = "Acme Cloud") -> SearchHit:
    return SearchHit(
        url=url,
        title=title,
        snippet="",
        domain=None,
        rank=1,
        provider="test",
        retrieved_at=datetime.now(UTC),
    )


def _source(domain: str = "acmecloud.com") -> Source:
    return Source(
        id=f"src-{domain}",
        url=f"https://{domain}/",
        canonical_url=f"https://{domain}/",
        title="Acme Cloud",
        text="Acme Cloud provides managed hosting.",
        domain=domain,
        source_type=SourceType.OFFICIAL_SITE,
        page_type=PageType.HOMEPAGE,
        retrieval_status=RetrievalStatus.OK,
    )


class _Search:
    """A search provider that returns fixed hits, or fails."""

    def __init__(self, hits: list[SearchHit] | None = None, *, error: Exception | None = None):
        self.hits = hits or []
        self.error = error
        self.queries: list[str] = []

    async def search(self, query: str, *, limit: int = 10, **_: object) -> list[SearchHit]:
        self.queries.append(query)
        if self.error:
            raise self.error
        return self.hits


class TestCompanyAgentRun:
    async def test_it_searches_for_the_company_and_resolves(self) -> None:
        search = _Search([_hit("https://acmecloud.com/", title="Acme Cloud")])
        agent = CompanyAgent()

        result = await agent.run(AgentContext(company="Acme Cloud", search=search))

        assert result.status is AgentStatus.OK
        assert result.payload.domains == ["acmecloud.com"]
        assert search.queries == ["Acme Cloud official site"]

    async def test_a_search_outage_falls_back_to_retrieved_pages(self) -> None:
        """A dead search provider must not end the run.

        The run may already have the company's own homepage, which is a stronger
        anchor than any search hit. So the outage degrades to a warning and the
        resolution proceeds on what was fetched.
        """
        search = _Search(error=RuntimeError("search is down"))
        agent = CompanyAgent()

        result = await agent.run(
            AgentContext(
                company="Acme Cloud", search=search, extras={"sources": [_source()]}
            )
        )

        assert result.status is AgentStatus.OK
        assert result.payload.domains == ["acmecloud.com"]

    async def test_a_search_outage_with_no_pages_is_partial_not_an_exception(
        self,
    ) -> None:
        agent = CompanyAgent()
        search = _Search(error=RuntimeError("search is down"))

        result = await agent.run(AgentContext(company="Acme Cloud", search=search))

        assert result.status is AgentStatus.PARTIAL
        assert result.succeeded
        assert result.warnings

    async def test_an_unresolvable_company_is_partial_with_a_stated_reason(self) -> None:
        """The gap is recorded, so the report can name it."""
        agent = CompanyAgent()
        search = _Search([_hit("https://a.example/", title="Something Else")])

        result = await agent.run(AgentContext(company="Acme Cloud", search=search))

        assert result.status is AgentStatus.PARTIAL
        assert result.payload is None
        assert result.warnings

    async def test_supplied_hits_are_used_instead_of_a_second_search(self) -> None:
        """A graph that already searched must not pay for it twice."""
        search = _Search([_hit("https://wrong.example/", title="Wrong")])
        agent = CompanyAgent()

        result = await agent.run(
            AgentContext(
                company="Acme Cloud",
                search=search,
                extras={"hits": [_hit("https://acmecloud.com/", title="Acme Cloud")]},
            )
        )

        assert result.payload.domains == ["acmecloud.com"]
        assert search.queries == []

    async def test_only_the_winning_domain_is_reported_as_retrieved(self) -> None:
        """A run must not cite a page it did not anchor on.

        Both pages were retrieved, but anchoring on one and then citing the other
        as supporting evidence would overstate what the run established.
        """
        agent = CompanyAgent()
        result = await agent.run(
            AgentContext(
                company="Acme Cloud",
                extras={
                    "sources": [_source("acmecloud.com"), _source("globex.com")],
                    "hits": [_hit("https://acmecloud.com/", title="Acme Cloud")],
                },
            )
        )

        assert result.payload.domains == ["acmecloud.com"]
        assert result.retrieved_source_ids == ["src-acmecloud.com"]

    async def test_it_needs_no_search_provider_at_all(self) -> None:
        """Resolving from a retrieved page is a complete answer on its own."""
        result = await CompanyAgent().run(
            AgentContext(company="Acme Cloud", extras={"sources": [_source()]})
        )

        assert result.status is AgentStatus.OK
        assert result.payload.domains == ["acmecloud.com"]

    def test_its_task_type_is_one_the_router_knows(self) -> None:
        """An unregistered task type raises at routing time, not wiring time.

        This was ``company_resolution``, which is in neither ``TASK_ROLES`` nor
        the prompt registry, so ``ModelRouter.role_for`` rejected it.
        """
        from app.prompts import PROMPTS
        from app.services.model_router import TASK_ROLES

        assert CompanyAgent.task_type in TASK_ROLES
        assert CompanyAgent.task_type in PROMPTS

    def test_its_output_schema_matches_its_prompt(self) -> None:
        from app.prompts import PROMPTS

        agent = CompanyAgent()
        assert PROMPTS[agent.task_type].OUTPUT_SCHEMA is agent.output_schema


def _validator(sources: list[Source] | None = None) -> Validator:
    # The key must come from the same normaliser the validator uses internally, or
    # self-exclusion silently stops matching and the target is returned as one of
    # its own competitors.
    return Validator(
        target_identity_key=company_identity_key("Acme Cloud"),
        sources={source.id: source for source in (sources or [])},
    )


def _signals(source_id: str = "src-acmecloud.com") -> list[ValidationSignal]:
    """Signals backed by a page the validator actually has.

    The source id has to match one in the validator's map: a signal citing a page
    the validator cannot see is not evidence, so a candidate carrying only those
    is correctly rejected and the test would be asserting the wrong thing.
    """
    return [
        ValidationSignal(signal="same_industry", supported=True, source_ids=[source_id]),
        ValidationSignal(
            signal="product_overlap", supported=True, source_ids=[source_id]
        ),
    ]


def _candidate(name: str = "Beta Metrics") -> CompetitorCandidate:
    return CompetitorCandidate(
        name=name, website=f"https://{name.split()[0].lower()}.example"
    )


class TestCompetitorAgentRun:
    async def test_a_supplied_pool_is_validated_and_truncated(self) -> None:
        agent = CompetitorAgent(validator=_validator([_source()]))
        pool = [
            CandidateSignals(candidate=_candidate(f"Company {i}"), signals=_signals())
            for i in range(5)
        ]

        result = await agent.run(
            AgentContext(
                company="Acme Cloud",
                competitor_count=2,
                extras={"candidates": pool},
            )
        )

        assert result.status is AgentStatus.OK
        assert len(result.payload.competitors) == 2

    async def test_finding_fewer_than_asked_is_partial_with_the_count(self) -> None:
        """The shortfall is written down by the step that knows it."""
        agent = CompetitorAgent(validator=_validator([_source()]))

        result = await agent.run(
            AgentContext(
                company="Acme Cloud",
                competitor_count=5,
                extras={
                    "candidates": [
                        CandidateSignals(
                            candidate=_candidate(), signals=_signals()
                        )
                    ]
                },
            )
        )

        assert result.status is AgentStatus.PARTIAL
        assert any("1 survived" in warning for warning in result.warnings)

    async def test_an_empty_pool_is_partial_rather_than_an_empty_success(self) -> None:
        agent = CompetitorAgent(validator=_validator([_source()]))

        result = await agent.run(AgentContext(company="Acme Cloud", extras={}))

        assert result.status is AgentStatus.PARTIAL
        assert result.payload is None
        assert result.warnings

    async def test_a_pool_where_everything_fails_validation_is_reported(self) -> None:
        """A run with no competitors has no comparison to make; say so."""
        agent = CompetitorAgent(validator=_validator([_source()]))
        pool = [
            CandidateSignals(candidate=_candidate(), signals=[]) for _ in range(3)
        ]

        result = await agent.run(
            AgentContext(company="Acme Cloud", extras={"candidates": pool})
        )

        assert result.status is AgentStatus.PARTIAL
        assert result.payload.competitors == []
        assert any("rejected" in warning for warning in result.warnings)

    async def test_rejected_candidates_are_kept_in_the_payload(self) -> None:
        """A reader can only ask why a company is absent if it was recorded."""
        agent = CompetitorAgent(validator=_validator([_source()]))
        pool = [
            CandidateSignals(candidate=_candidate(), signals=[]),
            CandidateSignals(candidate=_candidate("Gamma"), signals=_signals()),
        ]

        result = await agent.run(
            AgentContext(company="Acme Cloud", extras={"candidates": pool})
        )

        assert len(result.payload.rejected) == 1
        assert result.payload.rejected[0].candidate.name == "Beta Metrics"

    async def test_without_an_llm_it_reports_rather_than_raising(self) -> None:
        """Discovery needs a model; validation does not.

        With no pool supplied and no model there is nothing to validate, so the
        honest outcome is an empty result the graph can route on — not an
        exception from a node that is otherwise entirely offline-capable.
        """
        agent = CompetitorAgent(validator=_validator([_source()]))

        result = await agent.run(AgentContext(company="Acme Cloud", extras={}))

        assert result.succeeded
        assert result.status is AgentStatus.PARTIAL

    async def test_a_discovery_call_that_fails_is_reported_not_raised(self) -> None:
        class _BrokenLLM:
            async def generate(
                self, prompt: str, *, system: str | None = None, schema: object = None
            ) -> object:
                raise RuntimeError("model is down")

        agent = CompetitorAgent(validator=_validator([_source()]))

        result = await agent.run(
            AgentContext(company="Acme Cloud", llm=_BrokenLLM(), extras={})
        )

        assert result.succeeded
        assert result.status is AgentStatus.PARTIAL

    async def test_a_discovery_answer_with_no_json_is_reported_not_raised(self) -> None:
        class _ProseLLM:
            async def generate(
                self, prompt: str, *, system: str | None = None, schema: object = None
            ) -> object:
                from app.schemas.provider import LLMResponse

                return LLMResponse(
                    content="I could not think of any competitors.",
                    model="m",
                    provider="stub",
                )

        agent = CompetitorAgent(validator=_validator([_source()]))

        result = await agent.run(
            AgentContext(company="Acme Cloud", llm=_ProseLLM(), extras={})
        )

        assert result.succeeded
        assert result.status is AgentStatus.PARTIAL

    async def test_a_discovery_answer_becomes_a_validated_pool(self) -> None:
        """The full path: model proposes, validator decides, no model in between.

        The model is allowed to propose generously, but only what the validator
        accepts reaches the payload — which is the property that makes an
        uncritical discovery step safe to have in the pipeline at all.
        """
        from app.schemas.provider import LLMResponse

        class _DiscoveryLLM:
            async def generate(
                self, prompt: str, *, system: str | None = None, schema: object = None
            ) -> LLMResponse:
                assert "Acme Cloud" in prompt
                return LLMResponse(
                    content=(
                        '{"candidates": ['
                        '{"name": "Beta Metrics", "website": "https://beta.example"},'
                        '{"name": "Acme Cloud", "website": "https://acmecloud.com"}'
                        "]}"
                    ),
                    model="m",
                    provider="stub",
                )

        agent = CompetitorAgent(validator=_validator([_source()]))
        result = await agent.run(
            AgentContext(
                company="Acme Cloud",
                llm=_DiscoveryLLM(),
                competitor_count=3,
                extras={"sources": [_source()]},
            )
        )

        # Neither candidate is corroborated by a retrieved page naming it, so both
        # are rejected: an unevidenced candidate must not survive on the model's
        # say-so alone.
        assert result.payload.competitors == []
        assert {item.candidate.name for item in result.payload.rejected} == {
            "Beta Metrics",
            "Acme Cloud",
        }

    async def test_a_candidate_named_on_a_retrieved_page_can_be_accepted(self) -> None:
        """The positive path: the model proposes, a page corroborates, it survives.

        This is the only way a discovered candidate reaches the report — the model
        proposing it is not enough, and a page naming it is not automatically
        enough either, it has to be enough *to the validator*.
        """
        from app.schemas.provider import LLMResponse

        class _DiscoveryLLM:
            async def generate(
                self, prompt: str, *, system: str | None = None, schema: object = None
            ) -> LLMResponse:
                return LLMResponse(
                    content='{"candidates": [{"name": "Beta Metrics"}]}',
                    model="m",
                    provider="stub",
                )

        source = _source()
        source.text = (
            "Acme Cloud provides managed hosting. Beta Metrics is a close "
            "alternative for the same buyer."
        )
        agent = CompetitorAgent(validator=_validator([source]))

        result = await agent.run(
            AgentContext(
                company="Acme Cloud",
                llm=_DiscoveryLLM(),
                competitor_count=1,
                extras={"sources": [source]},
            )
        )

        assert [item.name for item in result.payload.competitors] == ["Beta Metrics"]
        assert result.payload.competitors[0].signals[0].source_ids == [source.id]

    async def test_the_target_itself_is_never_returned_as_a_competitor(self) -> None:
        """Self-exclusion is a correctness rule, not a preference."""
        agent = CompetitorAgent(validator=_validator([_source()]))

        result = await agent.run(
            AgentContext(
                company="Acme Cloud",
                extras={
                    "candidates": [
                        CandidateSignals(
                            candidate=_candidate("Acme Cloud"), signals=_signals()
                        )
                    ]
                },
            )
        )

        assert result.payload.competitors == []

    def test_its_task_type_names_a_registered_prompt(self) -> None:
        from app.prompts import PROMPTS
        from app.services.model_router import TASK_ROLES

        assert CompetitorAgent.task_type in TASK_ROLES
        assert CompetitorAgent.task_type in PROMPTS

    def test_it_declares_a_schema_that_holds_the_whole_answer(self) -> None:
        """Not one competitor: a set plus the rejected ones."""
        from app.schemas.competitor import CompetitorSelection

        agent = CompetitorAgent(validator=_validator())
        assert agent.output_schema is CompetitorSelection
        assert "competitors" in agent.output_schema.model_fields
        assert "rejected" in agent.output_schema.model_fields


class TestBothAgentsStayUsableWithoutAProvider:
    @pytest.mark.parametrize(
        "agent",
        [
            pytest.param(CompanyAgent(), id="company"),
            pytest.param(CompetitorAgent(validator=_validator()), id="competitor"),
        ],
    )
    async def test_an_empty_context_never_raises(self, agent: object) -> None:
        """Both must be callable with nothing, because the graph calls them early.

        A node that raises on an empty context cannot be the first node in a
        pipeline, and neither of these needs a provider to explain its own failure.
        """
        result = await agent.run(AgentContext(company="Acme Cloud"))  # type: ignore[attr-defined]

        assert result.succeeded
        assert result.agent == agent.name  # type: ignore[attr-defined]
