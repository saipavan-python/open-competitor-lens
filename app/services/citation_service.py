"""Citation indexing: turning evidence back into links a reader can follow.

Evidence is stored one claim at a time, each pointing at a single source. That
is the right shape for verification and the wrong shape for rendering: a report
cannot carry four hundred inline references, and it cannot answer "what backs
this section" without a reverse index from source to the claims it supports.

This service holds that index and nothing else. It performs no verification --
:class:`~app.services.evidence_service.EvidenceService` has already refused
anything unsupportable by the time a citation exists -- so a citation built here
can be assumed to point at a page the run actually retrieved. What it does
enforce is the *other* direction of the guarantee: a source that no evidence
cites is reported as an orphan rather than quietly included, because a
bibliography entry supporting nothing is a claim the report is not making, and
shipping it would imply a breadth of sourcing the run never achieved.
"""

from __future__ import annotations

from typing import NamedTuple

from app.core.exceptions import ValidationError
from app.core.logging import get_logger
from app.schemas.evidence import Citation, Evidence, Source

log = get_logger(__name__)


class CitationResolution(NamedTuple):
    """The outcome of resolving a citation index against the retrieved sources.

    A ``NamedTuple`` rather than a bare list because both halves are needed: the
    cited sources become the report's bibliography, and the orphans are the
    pages the run fetched but ended up not citing. Both are needed, and
    returning only the first would mean the second was computed and thrown away.
    """

    resolved: list[Source]
    orphans: list[Source]


class CitationService:
    """Builds and resolves the source-to-claims index for one run."""

    def __init__(self) -> None:
        self._known: dict[str, Source] = {}

    def build_index(
        self, sources: list[Source], evidence: list[Evidence]
    ) -> dict[str, list[str]]:
        """Map each cited source id to the claim ids it supports.

        Evidence naming a source that is not in ``sources`` is dropped rather
        than indexed. Indexing it would produce an entry the resolver cannot
        turn into a link, so the citation would render as a dead reference --
        the exact hallucinated-citation failure the evidence gate exists to
        prevent, reappearing downstream of it. Such entries are counted and
        logged so the mismatch is visible rather than merely absent.

        Claim ids are de-duplicated while preserving order, because the same
        evidence may legitimately arrive twice (a re-planned graph step, a
        retried node) and a bibliography entry listing one claim twice is
        noise.
        """
        known = {source.id: source for source in sources}
        self._known = known

        index: dict[str, list[str]] = {}
        dangling = 0
        for item in evidence:
            if item.source_id not in known:
                dangling += 1
                continue
            claims = index.setdefault(item.source_id, [])
            if item.id not in claims:
                claims.append(item.id)

        if dangling:
            log.warning(
                "evidence cited sources that were not retrieved",
                extra={"dropped": dangling},
            )
        return index

    def resolve(
        self, citation_index: dict[str, list[str]], sources: list[Source] | None = None
    ) -> CitationResolution:
        """Resolve an index to the sources it names, plus the uncited ones.

        ``sources`` defaults to those seen by the last
        :meth:`build_index`, so a caller that has just built an index can
        resolve it without threading the source list through a second time.
        Passing it explicitly is how a caller resolves against sources other
        than the ones currently indexed.
        """
        catalogue = (
            {source.id: source for source in sources}
            if sources is not None
            else dict(self._known)
        )
        cited = {source_id for source_id in citation_index if source_id in catalogue}
        resolved = [
            source
            for source in catalogue.values()
            if source.id in cited
        ]
        orphans = [source for source in catalogue.values() if source.id not in cited]
        return CitationResolution(resolved, orphans)

    def to_citation(self, evidence: Evidence, source: Source) -> Citation:
        """Render one evidence as a citation pointing at a public URL.

        The source is passed in rather than looked up so the caller stays in
        control of which retrieval backs the citation, but the ids must agree:
        citing source A's text under source B's URL is a fabricated citation
        that is otherwise indistinguishable from a real one, and it is the
        failure this check exists to catch.
        """
        if evidence is None:
            raise ValidationError(
                "cannot build a citation without evidence to cite"
            )
        if source is None:
            raise ValidationError("cannot build a citation without a source")
        if evidence.source_id != source.id:
            raise ValidationError(
                f"evidence {evidence.id!r} cites source "
                f"{evidence.source_id!r} but was rendered against "
                f"{source.id!r}"
            )
        return Citation(
            claim_id=evidence.id,
            source_id=source.id,
            url=source.url,
            title=source.title,
            source_type=source.source_type,
            retrieved_at=source.retrieved_at,
            quote=evidence.evidence_text,
        )
