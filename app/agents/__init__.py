"""Agents: the steps of the analysis, each one a boundary that can refuse.

An agent is a named unit of work with a declared output schema and a budget. The
base class holds what every agent needs — the run's context, a status that can
express partial success, and a result that carries a payload *or* a refusal — so
that no individual agent has to reinvent the answer to "what do I return when I
could not do the job?".

That question is the reason this layer exists in this shape. In a system whose
whole purpose is not to make things up, "I could not determine this" has to be
as easy to return as an answer, or it will get replaced by a confident one. An
agent that cannot report a failure structurally will report a success with
hollow contents instead.
"""

from app.agents.base import Agent, AgentContext, AgentResult, AgentStatus
from app.agents.company_agent import (
    CompanyAgent,
    CompanyCandidate,
    CompanyResolution,
    name_similarity,
)
from app.agents.competitor_agent import (
    VALIDATION_SIGNALS,
    CandidateSignals,
    CompetitorAgent,
    CompetitorSelection,
    ValidationOutcome,
    Validator,
)
from app.agents.planner import PlannerAgent, PlannerInput

__all__ = [
    "Agent",
    "AgentContext",
    "AgentResult",
    "AgentStatus",
    "CandidateSignals",
    "CompanyAgent",
    "CompanyCandidate",
    "CompanyResolution",
    "CompetitorAgent",
    "CompetitorSelection",
    "PlannerAgent",
    "PlannerInput",
    "VALIDATION_SIGNALS",
    "ValidationOutcome",
    "Validator",
    "name_similarity",
]
