"""The agent contract.

An agent here is a single, bounded unit of work: it has a name, a stated goal,
declared input and output schemas, a timeout and an iteration ceiling. It is not
an autonomous agent in the tool-using sense — there is no free-form loop, no
self-directed tool selection, and no model deciding what to do next. Each step of
the research pipeline is one of these, and the orchestrator (Task 12) sequences
them.

That is a deliberate narrowing. The failure mode of a tool-using agent in a system
whose whole purpose is verifiable evidence is that it takes a route nobody
approved, retrieves pages nobody recorded, and produces claims the evidence gate
then refuses — or worse, ones it cannot check. Constraining each step to a
declared input shape, a declared output schema and a bounded time budget means
the graph stays a readable description of what the run does, and every claim has
a path back to a retrieval.

Two result states matter. ``ok`` means the step did its job. ``partial`` means it
did enough to be useful and the shortfall is recorded — a run that found three
competitors when asked for five is a report with a gap in it, which is very
different from a run that quietly claimed five. ``failed`` means it did not do
its job, and the orchestrator decides whether the run can continue without it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic import BaseModel as _BaseModel

from app.core.constants import StrEnum
from app.core.logging import get_logger
from app.providers.base import LLMProvider, SearchProvider

log = get_logger(__name__)

#: Ceiling on how long a single agent may run.
DEFAULT_AGENT_TIMEOUT = 120.0

#: Ceiling on refinement rounds. One is the norm; more exists only for the steps
#: that genuinely iterate, and a hard ceiling is what stops a misbehaving node
#: from consuming a run's entire budget.
DEFAULT_MAX_ITERATIONS = 1


class AgentStatus(StrEnum):
    """How well an agent did its job."""

    OK = "ok"
    PARTIAL = "partial"
    FAILED = "failed"


class AgentResult(_BaseModel):
    """What one agent produced, and how far it got.

    ``payload`` is untyped at this level because its concrete type is the agent's
    declared ``output_schema``, which every subclass fixes. Narrowing it here
    would mean a union of every agent's output and would validate the same data
    twice.
    """

    model_config = ConfigDict(extra="forbid")

    agent: str
    status: AgentStatus = AgentStatus.OK
    payload: Any = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    retrieved_source_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    iterations: int = 0
    model: str | None = None
    duration_ms: int = 0

    @property
    def succeeded(self) -> bool:
        return self.status is not AgentStatus.FAILED

    def warn(self, message: str) -> None:
        """Record a shortfall.

        Warnings are accumulated rather than raised because the interesting cases
        are cumulative — three analysts each quietly citing a source they were not
        given is a report with a systematic problem, and raising on the first one
        would hide the other two.
        """
        self.warnings.append(message)
        if self.status is AgentStatus.OK:
            self.status = AgentStatus.PARTIAL
        log.warning("agent warning", extra={"agent": self.agent, "warning": message})


@dataclass
class AgentContext:
    """What an agent is given to work with.

    A dataclass rather than a Pydantic model because it holds provider
    *protocols*, which are not model types — a model here would either reject
    them or coerce them into something they are not.
    """

    company: str
    research_goal: str = ""
    competitor_count: int = 3
    llm: LLMProvider | None = None
    search: SearchProvider | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def require_llm(self) -> LLMProvider:
        if self.llm is None:
            raise RuntimeError(
                f"agent for {self.company!r} needs an LLM provider; none was supplied"
            )
        return self.llm

    def require_search(self) -> SearchProvider:
        if self.search is None:
            raise RuntimeError(
                f"agent for {self.company!r} needs a search provider; none was supplied"
            )
        return self.search


class Agent(ABC):
    """One bounded step of the research pipeline."""

    #: Stable identifier, used as the node name in the graph.
    name: str = ""
    #: What this step is for, in one sentence, for a human reading a trace.
    goal: str = ""
    input_schema: type[BaseModel] | None = None
    output_schema: type[BaseModel] | None = None
    timeout: float = DEFAULT_AGENT_TIMEOUT
    max_iterations: int = DEFAULT_MAX_ITERATIONS

    @abstractmethod
    async def run(self, ctx: AgentContext) -> AgentResult:
        """Do the work and report what happened."""

    def _result(self, **kwargs: Any) -> AgentResult:
        return AgentResult(agent=self.name or type(self).__name__, **kwargs)
