"""Working out who the user is actually asking about.

A user arrives with a name, and a name is not an identity. "Acme" might be the
group, the product, the trading name, or a division of a larger company whose
own site would answer the question better. So this agent's job is to pick the
single official domain to crawl first, and to be honest when it cannot.

**The official domain is the anchor for the whole run.** Identity resolution,
source authority, and the self-exclusion in competitor validation are all keyed
off it, so guessing here poisons everything downstream. A wrong-but-confident
domain means the run confidently profiles the wrong company.

That asymmetry is why everything here is conservative:

* A domain on the official registrable domain (or its subdomains) outranks
  anything a search engine suggested.
* Candidates that merely *mention* the name rank below ones that are named after
  it, because a page about a company is not the company.
* When no candidate is convincing, the result is ``PARTIAL`` with a reason, not a
  best guess wearing a confidence score. A missing domain is recoverable; a
  wrong one is not.

The name is the only thing this can be built from, so when there is not enough
to go on it says so in ``rejected_reason`` rather than picking the most popular
result.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import Field

from app.agents.base import Agent, AgentContext, AgentResult, AgentStatus
from app.core.logging import get_logger
from app.schemas.common import DomainModel
from app.schemas.company import CompanyIdentity
from app.schemas.evidence import Source
from app.schemas.search import SearchHit
from app.utils.normalization import company_identity_key
from app.utils.urls import canonicalize_url, registrable_domain

log = get_logger(__name__)

#: How close a candidate's name has to be to the queried name to count as a
#: spelling of it, rather than a different company that happens to be indexed
#: nearby. Deliberately tight: a loose ratio is how "Acme" acquires "Acme Digital"
#: as an alias, and from there as a self-exclusion.
NAME_SIMILARITY_FLOOR = 0.75

#: Below this, nothing is known well enough to crawl confidently.
MIN_ACCEPTABLE_CONFIDENCE = 0.5

#: How many search hits to consider when finding the company's own site. Small:
#: the official site is nearly always on the first page, and a wide search here
#: mostly returns directories and listicles that only add ranking noise.
HIT_LIMIT = 10


class CompanyCandidate(DomainModel):
    """A possible official home for the queried company."""

    domain: str
    title: str | None = None
    snippet: str | None = None
    name_match: float = Field(ge=0.0, le=1.0, default=0.0)
    source_id: str | None = None

    @property
    def url(self) -> str:
        return f"https://{self.domain}"


class CompanyResolution(DomainModel):
    """Which domain won, and what it was chosen over."""

    identity: CompanyIdentity | None = None
    domain: str | None = None
    url: str | None = None
    confidence: float = 0.0
    status: str = str(AgentStatus.OK)
    rejected_reason: str | None = None
    considered: list[CompanyCandidate] = Field(default_factory=list)


def name_similarity(left: str, right: str) -> float:
    """How much two company names look like the same name, in ``[0, 1]``.

    Scores on normalised spellings rather than raw strings, so case, spacing and
    legal suffixes are not differences: "Acme", "ACME  Cloud" and "Acme Cloud
    Private Limited" are all one name.

    Spaces are also ignored entirely for a full match, because the most common
    real difference between a company's name and its own domain is *nothing but a
    space* — ``acmecloud.com`` labels to ``acmecloud``, which is "Acme Cloud"
    written without the space. Comparing character sequences catches that
    immediately, where token overlap alone would not, and treating it as a full
    match matters because it is the single most common way a correct domain gets
    rejected for looking unrelated to its own company.

    Partial containment is scored but deliberately capped *below* the
    acceptance floor. "Acme" and "Acme Cloud" are related without being the same
    company, and "Acme Digital" is a competitor that must not inherit "Acme" as
    an alias — which is exactly the failure that would make it exclude itself
    from the comparison later.
    """
    left_key = company_identity_key(left)
    right_key = company_identity_key(right)
    if not left_key or not right_key:
        return 0.0
    if left_key == right_key:
        return 1.0

    left_flat = left_key.replace(" ", "")
    right_flat = right_key.replace(" ", "")
    if left_flat and left_flat == right_flat:
        return 1.0

    left_tokens = set(left_key.split())
    right_tokens = set(right_key.split())
    if shared := left_tokens & right_tokens:
        return len(shared) / len(left_tokens | right_tokens)

    shorter, longer = sorted((left_flat, right_flat), key=len)
    if shorter and longer.startswith(shorter):
        return 0.6
    return 0.0


class CompanyAgent(Agent):
    """Picks the official domain to crawl first."""

    name = "company_resolution"
    goal = "Determine the official domain for the company the user asked about."
    output_schema = CompanyIdentity
    #: Must match a key in ``TASK_ROLES`` and a module in the prompt registry.
    #: This was ``"company_resolution"``, which is in neither, so routing this
    #: agent's model raised ``ValueError: Unknown task type`` at run time rather
    #: than at wiring time. The registered name is the prompt's.
    task_type = "company_identification"

    def __init__(self, *, hits: Sequence[SearchHit] = ()) -> None:
        self._hits = list(hits)

    def candidates_from(self, hits: Sequence[SearchHit]) -> list[CompanyCandidate]:
        """Turn search hits into candidate official domains."""
        candidates: list[CompanyCandidate] = []
        for hit in hits:
            canonical = canonicalize_url(hit.url or "")
            domain = registrable_domain(canonical) if canonical else None
            if not domain:
                continue
            candidates.append(
                CompanyCandidate(
                    domain=domain,
                    title=hit.title,
                    snippet=hit.snippet,
                )
            )
        return candidates

    def rank(
        self, candidates: Sequence[CompanyCandidate], *, query_name: str
    ) -> list[CompanyCandidate]:
        """Order candidates by how much they look like the queried company.

        Scored on the first label of the domain, not the whole domain: including
        the TLD compares "Acme Cloud" against "acmecloud com", which differs by
        an invented token and scores nothing. The label is the part a company
        actually chose as its name.

        Sorted on the score, then the domain as a deterministic tiebreak — two
        runs over the same hits must rank identically, or the same company
        resolves differently for no reason a reader could see.
        """
        scored = [
            candidate.model_copy(
                update={
                    "name_match": name_similarity(
                        candidate.domain.split(".")[0], query_name
                    )
                }
            )
            for candidate in candidates
        ]
        return sorted(scored, key=lambda item: (-item.name_match, item.domain))

    def resolve(
        self,
        query_name: str,
        hits: Sequence[SearchHit] = (),
        *,
        sources: Sequence[Source] = (),
    ) -> CompanyResolution:
        """Choose the official domain, or explain why none was chosen.

        An official page the run has already retrieved always wins outright. It
        is the only candidate whose authority is not a matter of opinion, and
        preferring it over a better-ranking search hit keeps the anchor of the
        run on something that was actually fetched.
        """
        retrieved = self._from_retrieved(query_name, sources)
        if retrieved is not None:
            return retrieved

        candidates = self.rank(self.candidates_from(hits), query_name=query_name)
        if not candidates:
            return CompanyResolution(
                identity=None,
                domain=None,
                url=None,
                confidence=0.0,
                status=str(AgentStatus.PARTIAL),
                rejected_reason=(
                    f"no candidate domain could be derived from {query_name!r}; "
                    f"nothing was crawled, so there is no official site to anchor on"
                ),
                considered=[],
            )

        best = candidates[0]
        # Both bounds, and they are not the same check. The floor rejects scores
        # that are merely adequate; the confidence minimum rejects a run that
        # could not gather much evidence at all, so a weak score is not
        # surprising there and should not also be reported as a weak match.
        if (
            best.name_match < NAME_SIMILARITY_FLOOR
            or best.name_match < MIN_ACCEPTABLE_CONFIDENCE
        ):
            return CompanyResolution(
                identity=None,
                domain=None,
                url=None,
                confidence=best.name_match,
                status=str(AgentStatus.PARTIAL),
                rejected_reason=(
                    f"best candidate {best.domain!r} matched {query_name!r} only "
                    f"{best.name_match:.2f}, below the {NAME_SIMILARITY_FLOOR} floor; "
                    f"refusing to profile a company on a weak guess"
                ),
                considered=candidates,
            )

        return CompanyResolution(
            identity=CompanyIdentity(canonical_name=query_name, domains=[best.domain]),
            domain=best.domain,
            url=best.url,
            confidence=best.name_match,
            status=str(AgentStatus.OK),
            considered=candidates,
        )

    def _from_retrieved(
        self, query_name: str, sources: Sequence[Source]
    ) -> CompanyResolution | None:
        """An official domain this run has already retrieved, if there is one.

        A fetched page is the only candidate whose authority is not a matter of
        opinion, so one matching the queried name — by its title or by the label
        in its own domain — outranks anything a search engine suggested. The
        domain label is checked as well as the title because a company whose site
        is ``acmecloud.com`` very often titles its pages "Something Else
        entirely", and the domain is the stronger signal of the two.
        """
        best: Source | None = None
        best_score = 0.0
        for source in sources:
            if not source.domain or not source.text:
                continue
            label = source.domain.split(".")[0]
            score = max(
                name_similarity(source.title or "", query_name),
                name_similarity(label, query_name),
            )
            if score < NAME_SIMILARITY_FLOOR:
                continue
            # Domain as tiebreak so two equally good pages resolve the same way
            # on every run.
            if best is None or score > best_score or (
                score == best_score and source.domain < (best.domain or "")
            ):
                best, best_score = source, score

        if best is None:
            return None
        return CompanyResolution(
            identity=CompanyIdentity(
                canonical_name=query_name, domains=[best.domain or ""]
            ),
            domain=best.domain,
            url=best.canonical_url or best.url,
            confidence=1.0,
            status=str(AgentStatus.OK),
        )

    async def run(self, ctx: AgentContext) -> AgentResult:
        """Find candidates for the company, then resolve them to one domain.

        This is the async adapter over :meth:`resolve`, which stays pure so the
        ranking rules can be tested without a search provider. The agent gathers
        its own candidates rather than demanding them from the caller, so it is
        runnable on its own; a graph that has already searched passes its hits in
        ``ctx.extras["hits"]`` and no second search happens.
        """
        hits = await self._hits_for(ctx)
        sources = list(ctx.extras.get("sources", []) or [])
        resolution = self.resolve(ctx.company, hits, sources=sources)

        # Only the pages on the winning domain count as retrieved-and-used. A run
        # that anchored on globex.com must not report acmecloud.com as evidence
        # just because both were in the same candidate pool.
        used = [
            source.id
            for source in sources
            if source.domain and source.domain == resolution.domain
        ]
        result = AgentResult(
            agent=self.name,
            status=AgentStatus(resolution.status),
            payload=resolution.identity,
            retrieved_source_ids=used,
        )
        if resolution.domain is None:
            # Recorded as a shortfall, not raised: the run can continue by saying
            # the target's own site was never found, and a report that names that
            # gap is worth more than an exception.
            result.warn(
                resolution.rejected_reason
                or f"no official domain could be resolved for {ctx.company!r}"
            )
        return result

    async def _hits_for(self, ctx: AgentContext) -> list[SearchHit]:
        """Search hits to resolve against, from the caller or from a search."""
        supplied = list(ctx.extras.get("hits", []) or []) or list(self._hits)
        if supplied:
            return supplied
        if ctx.search is None:
            # Not an error here. Resolving against an already-retrieved official
            # page needs no search, and a run that fetched the homepage first
            # should be able to resolve from that alone.
            return []
        try:
            return list(
                await ctx.search.search(f"{ctx.company} official site", limit=HIT_LIMIT)
            )
        except Exception as exc:  # noqa: BLE001 - a search outage is not fatal
            log.warning(
                "company search failed; resolving from retrieved pages only",
                extra={"company": ctx.company, "error": str(exc)},
            )
            return []
