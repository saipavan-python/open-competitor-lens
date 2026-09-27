"""What every one of the nine analysts does, so none of them can forget it.

Each analyst answers a different question, so each declares its own queries,
prompt and output schema. Everything *around* that is identical and lives here:
retrieve, filter, compress, call the model, then check what came back. That last
step is the reason this class exists.

**A model asked to cite will cite.** Given retrieved text it will produce fluent
analysis, and asked to support it with sources it will produce source ids — some
of them real. A plausible-looking ``src-8f2a…`` that was never retrieved is
indistinguishable in the output from a real one, and it survives every later
check, because every later check trusts the analyst. So the check happens here,
at the only point where the retrieved set is still in hand:

* every cited id must be in the set retrieved for *this* analyst, for *this*
  entity;
* ids that are not are **stripped from the payload**, and the run is marked
  ``PARTIAL`` with an ``unsupported_citation`` warning naming them.

Stripping rather than rejecting is deliberate. The analysis around a bad citation
is often sound, and discarding it loses real work over a reference that can simply
be dropped. Rejecting wholesale would mean one hallucinated id throws away
everything the analyst found, and would train every future author of an analyst
to be careless, because the strict version looks like it punishes honesty.

**Numbers require evidence.** A numeric value anywhere in a payload with no
evidence behind it is a fabrication until proven otherwise, and a number is the
easiest fabrication to produce by accident: a model asked for pricing that finds
no pricing page will helpfully supply a plausible figure. So the numeric check
refuses that payload, and it applies to all nine analysts by construction, because
they all inherit it.

Neither check is optional and neither has an override. A subclass supplies
queries, a prompt and a schema; it does not get to decide whether the base
verifies the answer.

**Retrieval is scoped to one entity.** Every analyst here analyses one company,
so its filters are keyed on ``entity`` — target or competitor. Spelling out
``company=`` or ``competitor=`` per analyst is how one of them ends up quietly
searching the target while supposedly analysing a competitor, and the resulting
analysis is not obviously wrong to a reader; it is just about the wrong company.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from typing import Any, ClassVar

from pydantic import ValidationError

from app.agents.base import AgentContext, AgentResult, AgentStatus
from app.core.constants import SourceType
from app.core.logging import get_logger
from app.prompts import build as build_prompt
from app.prompts.base import PromptContext
from app.providers.base import LLMProvider
from app.schemas.common import DomainModel
from app.schemas.document import RetrievedChunk
from app.schemas.evidence import Passage
from app.schemas.provider import LLMResponse
from app.services.rag_service import RagService
from app.utils.json import safe_json_loads

log = get_logger(__name__)

UNSUPPORTED_CITATION = "unsupported_citation"
UNSUPPORTED_NUMBER = "unsupported_number"
NO_EVIDENCE = "no_evidence"
NO_LLM = "no_llm"
SCHEMA_ERROR = "schema_error"
UNPARSABLE = "unparsable_response"

#: Retrieved passages handed to the model. Enough to answer a category question,
#: small enough that nine categories over the same corpus stay affordable.
MAX_CONTEXT_CHUNKS = 8

#: Keys naming *what was looked at* rather than asserting a fact about the
#: world, and so are not claims needing evidence of their own. `company` and
#: `market` are here for the same reason: they label the analysis, they do not
#: assert anything.
NON_FACTUAL_KEYS: frozenset[str] = frozenset(
    {
        "company",
        "market",
        "confidence",
        "evidence",
        "pages_checked",
        "categories_covered",
        "currency",
        "pricing_page_url",
        "positioning",
        "title_tag",
    }
)

#: The four things every analyst must declare. Checked at class-creation time so
#: a missing one is an import error rather than a runtime surprise inside a graph.
REQUIRED_DECLARATIONS: tuple[str, ...] = (
    "category",
    "prompt_name",
    "retrieval_queries",
    "output_schema",
)


class PayloadError(ValueError):
    """A payload failed the analyst's own validation."""


class UnsupportedNumberError(PayloadError):
    """A payload asserts a number with no evidence behind it."""


def numeric_leaves(value: Any, path: str = "") -> list[str]:
    """Every path in a payload holding a real number.

    ``bool`` is deliberately counted, not exempted. It is a subclass of ``int``
    in Python, and ``has_free_tier = True`` is a fact the model asserted from
    reading a page — so it is held to the same rule as a price. Exempting it
    would leave the check applying to prices but not to the free tier, which is
    backwards: the free tier is exactly the field a model infers from silence.
    """
    found: list[str] = []
    if isinstance(value, bool):
        return [path or "<root>"]
    if isinstance(value, (int, float)):
        return [path or "<root>"]
    if isinstance(value, dict):
        for key, item in value.items():
            if key in NON_FACTUAL_KEYS:
                continue
            found.extend(numeric_leaves(item, f"{path}.{key}" if path else str(key)))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(numeric_leaves(item, f"{path}[{index}]"))
    return found


def validate_payload_numbers(payload: dict[str, Any]) -> None:
    """Refuse a payload that asserts a number with no evidence behind it."""
    if payload.get("evidence"):
        return
    numbers = numeric_leaves(payload)
    if numbers:
        raise UnsupportedNumberError(
            f"payload asserts {len(numbers)} numeric or boolean value(s) "
            f"({', '.join(numbers[:5])}) with no evidence; a number with nothing "
            f"behind it is a guess, and the not-public sentinel exists precisely "
            f"so that guessing is never necessary"
        )


def validate_citations(
    payload: dict[str, Any], allowed: set[str], *, analyst: str = ""
) -> list[str]:
    """Strip citation ids absent from ``allowed``; return what was removed.

    Mutates ``payload`` in place and returns the removed ids, so the caller both
    persists a payload containing only real references and reports what was
    dropped. Doing both in one pass avoids the two failure modes of doing them
    separately: cleaning silently, or warning while the bad id stays in the data.
    """
    removed: list[str] = []
    for key in ("evidence", "source_ids", "sources"):
        ids = payload.get(key)
        if not isinstance(ids, list):
            continue
        kept: list[Any] = []
        for value in ids:
            if isinstance(value, str) and value in allowed:
                kept.append(value)
            else:
                removed.append(str(value))
        if len(kept) != len(ids):
            payload[key] = kept
    if removed:
        log.warning(
            "analyst cited sources it did not retrieve",
            extra={
                "analyst": analyst,
                "entity": payload.get("company"),
                "removed": sorted(set(removed)),
            },
        )
    return sorted(set(removed))


class CategoryAnalyst(ABC):
    """Base for the nine category analysts.

    Subclasses declare three things — a category, a prompt name, and retrieval
    queries — plus an ``output_schema`` taken from the registered prompt. The
    retrieve → filter → compress → analyse sequence and both verification checks
    are inherited, because a check implemented per analyst is a check nine
    authors will implement nine ways.
    """

    category: ClassVar[str]
    prompt_name: ClassVar[str]
    retrieval_queries: ClassVar[tuple[str, ...]]
    output_schema: ClassVar[type[DomainModel]]
    name: ClassVar[str] = ""
    goal: ClassVar[str] = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        missing = [
            declaration
            for declaration in REQUIRED_DECLARATIONS
            if not getattr(cls, declaration, None)
        ]
        if missing:
            raise TypeError(
                f"{cls.__name__} must declare {', '.join(missing)}; every analyst "
                f"needs a category, a prompt, its retrieval queries and a schema"
            )

    def __init__(
        self,
        *,
        rag: RagService,
        llm: LLMProvider | None = None,
        entity: str | None = None,
        max_chunks: int = MAX_CONTEXT_CHUNKS,
    ) -> None:
        self._rag = rag
        self._llm = llm
        self._entity = entity
        self._max_chunks = max_chunks

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(entity={self._entity!r})"

    @property
    def entity(self) -> str | None:
        return self._entity

    # -- context ----------------------------------------------------------

    def filters(self) -> dict[str, str]:
        """Metadata filters scoping retrieval to this analyst's subject."""
        return {"entity": self._entity} if self._entity else {}

    def retrieve(self) -> list[RetrievedChunk]:
        """This analyst's context: its queries, filtered to its subject."""
        return self._rag.retrieve_many(
            list(self.retrieval_queries), filters=self.filters(), k=self._max_chunks
        )

    def to_passages(self, chunks: Sequence[RetrievedChunk]) -> list[Passage]:
        """Render retrieved chunks as the prompt's input passages.

        The conversion happens here, at the only boundary between retrieval and
        prompting, because a passage is not evidence and must not be able to
        masquerade as one. Each keeps its ``source_id``, which is the only handle
        the model has for citing it and the only handle the citation check has for
        verifying that citation.
        """
        return [
            Passage(
                source_id=chunk.source_id,
                url=chunk.url,
                text=chunk.text,
                title=chunk.title,
                heading=chunk.heading,
                source_type=self._source_type(chunk),
            )
            for chunk in chunks[: self._max_chunks]
        ]

    @staticmethod
    def _source_type(chunk: RetrievedChunk) -> SourceType | None:
        """A chunk's source type, from its metadata.

        Not a schema field on ``Chunk``, because the authority question belongs to
        the source rather than to the text sliced from it — and a chunk that
        claimed to be from an official site would be a nice way to launder a
        directory listing into primary evidence.

        An unrecognised value is ``None`` rather than a guess: metadata arrives
        from crawled pages, and a page that labels itself
        ``source_type=homepage`` has not thereby become an official site.
        """
        raw = chunk.metadata.get("source_type")
        if not raw:
            return None
        try:
            return SourceType(raw)
        except ValueError:
            return None

    def permitted_sources(self, chunks: Sequence[RetrievedChunk]) -> set[str]:
        """The source ids this analyst's output may cite."""
        return {chunk.source_id for chunk in chunks if chunk.source_id}

    # -- the shared sequence ----------------------------------------------

    async def run(self, ctx: AgentContext) -> AgentResult:
        """retrieve → filter → compress → analyse, then verify what came back."""
        chunks = self.retrieve()
        if not chunks:
            return self._no_evidence_result()

        if self._llm is None:
            log.warning(
                "analyst has no llm",
                extra={"analyst": self.name, "entity": self._entity},
            )
            return AgentResult(
                agent=self.name,
                status=AgentStatus.PARTIAL,
                errors=[
                    f"{self.name} retrieved {len(chunks)} chunk(s) for "
                    f"{self._entity} but has no LLM configured, so it analysed "
                    f"nothing"
                ],
                warnings=[NO_LLM],
            )

        messages = self.build_messages(ctx, chunks)
        response: LLMResponse = await self._llm.generate(
            messages[-1].content,
            system=messages[0].content,
            schema=self.output_schema,
        )
        payload = self._parse(response)
        if payload is None:
            return AgentResult(
                agent=self.name,
                status=AgentStatus.FAILED,
                errors=[
                    f"{self.name} returned no parsable JSON object: "
                    f"{response.content[:200]!r}"
                ],
                warnings=[UNPARSABLE],
            )

        return self.verify_and_build(payload, self.permitted_sources(chunks), chunks)

    def verify_and_build(
        self,
        payload: dict[str, Any],
        allowed: set[str],
        chunks: Sequence[RetrievedChunk] = (),
    ) -> AgentResult:
        """Run both checks, then build the output schema. Never raises.

        A payload the schema rejects is a failure with a stated reason rather
        than an exception: the graph routes on status, and an exception here would
        take down a category the run can still report as missing.
        """
        try:
            validate_payload_numbers(payload)
        except UnsupportedNumberError as exc:
            return AgentResult(
                agent=self.name,
                status=AgentStatus.FAILED,
                errors=[str(exc)],
                warnings=[UNSUPPORTED_NUMBER],
            )

        removed = validate_citations(payload, allowed, analyst=self.name)
        warnings = [f"{UNSUPPORTED_CITATION}: {', '.join(removed)}"] if removed else []
        if not payload.get("evidence"):
            warnings.append(
                f"{NO_EVIDENCE}: the analysis asserts nothing this run retrieved"
            )

        try:
            model = self.output_schema.model_validate(payload)
        except ValidationError as exc:
            return AgentResult(
                agent=self.name,
                status=AgentStatus.FAILED,
                errors=[f"{self.name} returned a payload the schema rejected: {exc}"],
                warnings=[SCHEMA_ERROR],
            )

        log.info(
            "analyst complete",
            extra={
                "analyst": self.name,
                "entity": self._entity,
                "category": self.category,
                "chunks": len(chunks),
                "stripped": len(removed),
            },
        )
        return AgentResult(
            agent=self.name,
            status=AgentStatus.PARTIAL if warnings else AgentStatus.OK,
            payload=model,
            warnings=warnings,
            # The sources this analysis is entitled to cite, carried on the result
            # so the next step can check its work without re-deriving it.
            retrieved_source_ids=sorted(allowed),
        )

    # -- prompting --------------------------------------------------------

    def prompt_context(
        self, ctx: AgentContext, chunks: Sequence[RetrievedChunk]
    ) -> PromptContext:
        """The prompt's view of this run, scoped to this analyst's one entity."""
        analysing_competitor = bool(
            self._entity and self._entity != ctx.company
        )
        return PromptContext(
            company=self._entity or ctx.company,
            research_goal=ctx.research_goal or self.goal,
            competitors=[self._entity] if analysing_competitor else [],
            passages=self.to_passages(chunks),
        )

    def build_messages(
        self, ctx: AgentContext, chunks: Sequence[RetrievedChunk]
    ) -> list[Any]:
        """The two chat messages for this analyst, from its registered prompt."""
        return build_prompt(
            self.prompt_name, self.prompt_context(ctx, list(chunks))
        )

    # -- internals --------------------------------------------------------

    def _no_evidence_result(self) -> AgentResult:
        """Nothing retrieved for this category.

        A finding, not a failure: the category genuinely is not publicly
        available and the report says so. Marking this ``FAILED`` would either
        abort the run or put error text where a reader expects a section.
        """
        log.info(
            "analyst found no evidence",
            extra={"analyst": self.name, "entity": self._entity},
        )
        return AgentResult(
            agent=self.name,
            status=AgentStatus.PARTIAL,
            errors=[
                f"no evidence retrieved for {self.category} "
                f"({self._entity or 'unspecified entity'})"
            ],
            warnings=[NO_EVIDENCE],
        )

    @staticmethod
    def _parse(response: LLMResponse) -> dict[str, Any] | None:
        """The response's JSON object, or ``None`` if there isn't one.

        ``LLMResponse`` carries ``content`` and nothing parsed, so the JSON is
        extracted here. Providers that can guarantee structured output still
        return it as content, which is why this is tolerant of a fenced block.
        """
        parsed = safe_json_loads(response.content)
        return dict(parsed) if isinstance(parsed, dict) else None
