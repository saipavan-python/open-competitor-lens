"""Competitor discovery's other half: deciding who is actually a competitor.

Discovery is deliberately generous. It returns candidates it is only mildly
confident about, because the step that follows is where weak candidates are
supposed to die — and a discovery step that pre-filters aggressively just hides
the candidates a reader would have wanted to see rejected. So the pool is built
at two to four times the requested count, and this module throws most of it away.

**The bar is one primary-source-backed signal.** Not a majority of signals, not a
score threshold: one signal that a retrieved, authoritative page actually
supports. The reason is the asymmetry of the error. A candidate that does not
compete, admitted on a search snippet, becomes a row in the comparison table
that looks exactly as well-sourced as a row built from a vendor's own pricing
page — and nobody downstream can tell, because by then the snippet has been
promoted into a source. Requiring one primary-backed signal makes the failure
mode a missing competitor rather than an invented one, and a missing competitor
is visible: the reader sees a short list.

**Identity is checked first.** A company cannot compete with itself, and neither
can its parent or its trading name. That check is exact normalised equality
rather than substring matching, because a substring rule would also drop "Acme
Digital" as though it were "Acme" — which is a real competitor lost for no reason.

Nothing here needs an LLM or a network. Validation is the step that must keep
working when a provider is down, because it is the last gate before a claim
enters the report.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from app.agents.base import Agent, AgentContext, AgentResult, AgentStatus
from app.core.constants import (
    NON_PRIMARY_SOURCE_TYPES,
    RetrievalStatus,
    SourceType,
)
from app.core.logging import get_logger
from app.prompts import PROMPTS
from app.prompts.base import PromptContext
from app.schemas.competitor import (
    CandidateSignals,
    CompetitorCandidate,
    CompetitorSelection,
    ValidationOutcome,
    ValidationSignal,
)
from app.schemas.evidence import Source
from app.utils.normalization import normalize_company_name
from app.utils.structured import parse_structured

log = get_logger(__name__)

#: A snippet is a discovery artefact. Until somebody retrieves the page behind
#: it, it is a third party's description of somebody else, and it cannot carry a
#: "these two companies compete" claim on its own.
_NON_PRIMARY = frozenset(NON_PRIMARY_SOURCE_TYPES) | {SourceType.SEARCH_SNIPPET}

#: The signals a candidate is scored on. Each must name the sources that
#: support it; a signal with no source is an opinion.
VALIDATION_SIGNALS: tuple[str, ...] = (
    "same_industry",
    "same_target_segment",
    "product_overlap",
    "substitutable",
    "listed_as_alternative",
    "co_occurrence",
    "same_geography",
)

#: How many candidates to consider per competitor requested. Wide enough that
#: validation is genuinely lossy, bounded so a run cannot fan out without limit.
POOL_MIN_MULTIPLE = 2
POOL_MAX_MULTIPLE = 4

#: The absolute ceiling on the pool, whatever the multiples are set to. A caller
#: raising ``min_multiple`` to chase under-recall gets a wider pool, but never
#: an unbounded one - the cost of the pool is a discovery call per candidate.
POOL_HARD_CAP_MULTIPLE = 6


class Validator:
    """Decides whether a candidate genuinely competes with the target."""

    def __init__(
        self,
        *,
        target_identity_key: str,
        sources: Mapping[str, Source] | None = None,
        required_signals: int = 1,
    ) -> None:
        self._target_key = target_identity_key
        self._sources: dict[str, Source] = dict(sources or {})
        self._required = required_signals

    def with_sources(self, sources: Mapping[str, Source]) -> Validator:
        """A copy that knows about more retrieved sources."""
        merged = {**self._sources, **sources}
        return Validator(
            target_identity_key=self._target_key,
            sources=merged,
            required_signals=self._required,
        )

    def validate(self, subject: CandidateSignals) -> ValidationOutcome:
        candidate = subject.candidate
        supported = [signal for signal in subject.signals if signal.supported]

        # Identity first: no amount of evidence makes a company its own competitor.
        if self._is_self(candidate):
            return ValidationOutcome(
                candidate=candidate,
                accepted=False,
                reason="self",
                signals=list(subject.signals),
                primary_signals=0,
            )

        if not supported:
            return self._reject(candidate, subject, "no supporting signals")

        primary = [signal for signal in supported if self._is_primary(signal)]
        if len(primary) < self._required:
            return self._reject(
                candidate,
                subject,
                f"no primary-source-backed signal (only a search snippet or an "
                f"unclassified page supports {len(supported)} signal(s))",
                primary_signals=len(primary),
            )

        return ValidationOutcome(
            candidate=candidate,
            accepted=True,
            reason=self._reason(primary),
            signals=list(supported),
            primary_signals=len(primary),
        )

    # -- helpers ----------------------------------------------------------

    def _reject(
        self,
        candidate: CompetitorCandidate,
        subject: CandidateSignals,
        reason: str,
        *,
        primary_signals: int = 0,
    ) -> ValidationOutcome:
        log.info("competitor rejected", extra={"name": candidate.name, "reason": reason})
        return ValidationOutcome(
            candidate=candidate,
            accepted=False,
            reason=reason,
            signals=list(subject.signals),
            primary_signals=primary_signals,
        )

    def _is_primary(self, signal: ValidationSignal) -> bool:
        """Whether a signal is backed by at least one authoritative retrieved page.

        A signal citing a source id the run never fetched contributes nothing.
        That is the same rule the evidence gate applies, for the same reason: an
        unverifiable citation is not weaker evidence, it is not evidence.

        Authority is a question about *what kind of page this is*, so that is all
        that is checked. Whether the page actually says anything is the
        evidence gate's job, several steps later — deciding it here would mean
        judging content at the structural gate, where a partial result is
        indistinguishable from a wrong one.
        """
        for source_id in signal.source_ids:
            source = self._sources.get(source_id)
            if source is None:
                continue
            if source.retrieval_status is not RetrievalStatus.OK:
                continue
            if source.source_type is None or source.source_type in _NON_PRIMARY:
                continue
            return True
        return False

    def _is_self(self, candidate: CompetitorCandidate) -> bool:
        """Whether this candidate is the target under another spelling.

        Compared as a set intersection rather than a single derived key, because
        the least-qualified spelling of a company is not stable: folding a
        candidate's website into the key can shorten it to a bare domain label
        and lose the name that actually identifies it. Every spelling the
        candidate is known by is therefore collected, and any overlap with the
        target's key counts.

        This will drop a distinct company that happens to sit on the target's own
        domain — which is almost always the target anyway, or a division of it.
        The opposite error, keeping a company's parent or trading name in the
        comparison as though it were a rival, is the one worth paying for.
        """
        spellings = _spellings(candidate)
        if not spellings:
            return False
        return self._target_key in spellings

    def _reason(self, primary: Sequence[ValidationSignal]) -> str:
        names = ", ".join(sorted({signal.signal for signal in primary}))
        return f"primary-source-backed: {names}"


def _spellings(candidate: CompetitorCandidate) -> set[str]:
    """Every normalised spelling this candidate is known by.

    A bare domain contributes its first label as well as itself, because
    ``acme.com`` is a spelling of ``Acme`` and not of ``Acme Com`` — the same
    rule :func:`company_identity_key` applies, kept here so the validator can
    compare the full set rather than the single least-qualified key.
    """
    raw: list[str] = [candidate.name]
    if candidate.website:
        raw.append(candidate.website)
    if candidate.identity:
        raw.append(candidate.identity.canonical_name)
        raw.extend(candidate.identity.aliases)
        raw.extend(candidate.identity.domains)

    spellings: set[str] = set()
    for value in raw:
        if not value:
            continue
        normalised = normalize_company_name(value)
        if normalised:
            spellings.add(normalised)
        if "." in value and " " not in value.strip():
            label = value.strip().split(".")[0]
            from_label = normalize_company_name(label)
            if from_label:
                spellings.add(from_label)
    return spellings


class CompetitorAgent(Agent):
    """Builds a candidate pool and selects the competitors worth reporting on."""

    name = "competitor_discovery"
    goal = "Find the companies that genuinely compete with the target."
    #: The selection, not a single competitor. This was ``ValidatedCompetitor``,
    #: which is one element of the answer: an agent asked for three competitors
    #: returns a set plus the rejected candidates, and a graph consuming the
    #: declared schema would have found a list where it expected one object.
    output_schema = CompetitorSelection
    task_type = "competitor_discovery"

    def __init__(
        self,
        *,
        validator: Validator,
        min_multiple: int = POOL_MIN_MULTIPLE,
        max_multiple: int = POOL_MAX_MULTIPLE,
    ) -> None:
        self._validator = validator
        self._min_multiple = min_multiple
        self._max_multiple = max_multiple

    def pool_size_for(self, competitor_count: int) -> int:
        """How many candidates to gather for a requested competitor count.

        Between ``min_multiple`` and ``max_multiple`` times the count, and never
        more than :data:`POOL_HARD_CAP_MULTIPLE` times: wide enough that
        validation has something to reject, bounded so a large requested count
        cannot fan out without limit.

        The previous version was ``min(count * max_multiple, count * 3)``, where
        the ``3`` was a literal. Two things followed from that, both of them
        silent. ``min_multiple`` was never read, so a caller raising it to widen
        the pool got no change at all; and the literal capped the pool below
        ``max_multiple``, so raising *that* was equally inert. A parameter whose
        only documented job is to be tuned, and which cannot be, is worse than
        none: the tuning looks like it is working.
        """
        count = max(1, competitor_count)
        wanted = count * max(self._min_multiple, self._max_multiple)
        return min(wanted, count * POOL_HARD_CAP_MULTIPLE)

    def select(
        self, pool: Sequence[CandidateSignals], *, competitor_count: int
    ) -> CompetitorSelection:
        """Validate every candidate, then keep the best ``competitor_count``."""
        outcomes = [self._validator.validate(subject) for subject in pool]
        accepted = [outcome for outcome in outcomes if outcome.accepted]
        rejected = [outcome for outcome in outcomes if not outcome.accepted]

        # Strongest evidence first, so a truncation keeps the best-supported
        # candidates rather than whichever happened to be discovered first.
        accepted.sort(
            key=lambda outcome: (-outcome.primary_signals, outcome.candidate.name)
        )
        kept = accepted[:competitor_count]

        status = AgentStatus.OK if len(kept) >= competitor_count else AgentStatus.PARTIAL
        if status is AgentStatus.PARTIAL:
            log.info(
                "fewer competitors validated than requested",
                extra={"requested": competitor_count, "found": len(kept)},
            )
        return CompetitorSelection(
            competitors=[outcome.to_competitor() for outcome in kept],
            rejected=rejected,
            status=str(status),
        )

    async def run(self, ctx: AgentContext) -> AgentResult:
        """Gather a candidate pool, then keep the best of it.

        Discovery is wide and uncritical; validation is where a weak candidate
        dies. Both halves are here rather than split across the graph, because a
        discovery step with no validation attached is exactly the failure this
        system exists to prevent — and because it keeps the "every candidate is
        judged" invariant in one place.

        A pool supplied in ``ctx.extras["candidates"]`` is used as-is, which is how
        a graph that already ran discovery avoids paying for it twice.
        """
        pool = await self._pool(ctx)
        if not pool:
            return AgentResult(
                agent=self.name,
                status=AgentStatus.PARTIAL,
                warnings=[
                    f"no competitor candidates were found for {ctx.company!r}, so "
                    f"there was nothing to validate"
                ],
            )

        selection = self.select(pool, competitor_count=ctx.competitor_count)
        result = AgentResult(
            agent=self.name,
            status=AgentStatus(selection.status),
            payload=selection,
        )
        if len(selection.competitors) < ctx.competitor_count:
            result.warn(
                f"asked for {ctx.competitor_count} competitors and "
                f"{len(selection.competitors)} survived validation out of "
                f"{len(pool)} candidates"
            )
        if not selection.competitors and selection.rejected:
            result.warn(
                f"all {len(selection.rejected)} candidate(s) were rejected; the "
                f"report will have no competitors to compare against"
            )
        return result

    async def _pool(self, ctx: AgentContext) -> list[CandidateSignals]:
        """The candidate pool, from the caller or from a discovery call."""
        supplied = list(ctx.extras.get("candidates", []) or [])
        if supplied:
            return supplied
        return await self._discover(ctx)

    async def _discover(self, ctx: AgentContext) -> list[CandidateSignals]:
        """Propose candidates with the discovery prompt, and gather signals.

        Discovery is one LLM call and no validation: the model is asked to
        brainstorm names, and every name it returns goes to the validator, which
        runs with no model and no network. That ordering is deliberate — the part
        that decides what counts as a competitor must not be the part that can be
        talked into it.
        """
        if ctx.llm is None:
            log.warning(
                "competitor discovery has no llm; no candidates proposed",
                extra={"company": ctx.company},
            )
            return []

        prompt = PROMPTS[self.task_type]
        messages = prompt.build(
            PromptContext(
                company=ctx.company,
                research_goal=ctx.research_goal,
                competitors=list(ctx.extras.get("competitors", []) or []),
                sources=list(ctx.extras.get("sources", []) or []),
            )
        )
        try:
            response = await ctx.llm.generate(
                messages[-1].content,
                system=messages[0].content,
                schema=prompt.OUTPUT_SCHEMA,
            )
        except Exception as exc:  # noqa: BLE001 - one call must not end the run
            log.warning(
                "competitor discovery call failed",
                extra={"company": ctx.company, "error": str(exc)},
            )
            return []

        try:
            discovered = parse_structured(response.content, prompt.OUTPUT_SCHEMA)
        except Exception as exc:  # noqa: BLE001 - reported as an empty pool
            log.warning(
                "competitor discovery returned unusable JSON",
                extra={"company": ctx.company, "error": str(exc)},
            )
            return []

        candidates = list(getattr(discovered, "candidates", []) or [])
        wanted = self.pool_size_for(ctx.competitor_count)
        return [
            CandidateSignals(
                candidate=candidate,
                signals=self._signals_for(ctx, candidate),
            )
            for candidate in candidates[:wanted]
        ]

    def _signals_for(
        self, ctx: AgentContext, candidate: CompetitorCandidate
    ) -> list[ValidationSignal]:
        """Signals for one candidate, from the pages the run already retrieved.

        Only retrieved pages count. A candidate is validated against evidence, and
        a search snippet is not evidence — so a candidate the model proposed and
        nothing else corroborates ends up rejected, which is the correct outcome
        for an unevidenced claim.

        Self-exclusion is deliberately *not* handled here. It is a fact about the
        candidate's identity rather than about any page, so it is the validator's
        job and it applies whether or not the run retrieved anything. Re-deciding
        it from page text would make the target's own exclusion depend on whether
        a page happened to have body copy.
        """
        name = candidate.name.strip().lower()
        if not name:
            return []
        signals: list[ValidationSignal] = []
        for source in ctx.extras.get("sources", []) or []:
            text = (source.text or "").lower()
            if not text or name not in text:
                continue
            # One page naming the candidate is enough to propose the signal. The
            # validator decides whether it is *sufficient*, which is the judgement
            # that belongs to it rather than to the gathering step.
            signals.append(
                ValidationSignal(
                    signal="co_occurrence",
                    supported=True,
                    source_ids=[source.id],
                    detail="the candidate is named on a retrieved page",
                )
            )
            break
        return signals
