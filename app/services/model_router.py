"""Cost-aware model routing.

One model for the whole run would be either too slow and too expensive for the
easy steps, or too cheap for the hard ones. This router splits the work across
the three profiles an operator configures in ``Settings`` — ``fast``,
``reasoning`` and ``final`` — so the expensive model is spent on the steps whose
output a reader actually relies on.

The single most important property here is that the router holds no model
identifiers of its own. Every id it returns came out of ``Settings``. A hardcoded
default would be invisible to an operator who deployed a different model, would
be a second source of truth to keep in sync with configuration, and would make
the cost of a run impossible to predict from settings alone. The task-type to
role mapping below is the only thing this module decides; which model fills each
role is configuration, not code.

The second property is that an unrecognised task type raises. The tempting
alternative is to default to the cheap model, which means a typo in an agent's
task name sends the final report to a small model and produces a plausible,
fluent, wrong report. Failing at the routing table is much cheaper than that.
"""

from __future__ import annotations

from app.core.config import Settings, get_settings
from app.core.exceptions import ValidationError
from app.core.logging import get_logger

log = get_logger(__name__)

#: The three configured profiles, cheapest first.
ROLES: tuple[str, ...] = ("fast", "reasoning", "final")

#: Which profile does which job.
#:
#: ``fast`` is for mechanical, high-volume work whose output is checked against
#: the source text anyway: writing search queries, pulling a quote out of a page,
#: scoring a candidate. ``reasoning`` is for judgement calls that shape the rest
#: of the run. ``final`` is reserved for the report a person reads, because it
#: is the one output where a weak model is visible to the end user.
TASK_ROLES: dict[str, str] = {
    # -- mechanical, high volume, mechanically checkable
    "query_generation": "fast",
    "source_extraction": "fast",
    "evidence_extraction": "fast",
    "competitor_validation": "fast",
    # -- judgement that shapes the run
    "research_planning": "reasoning",
    "company_identification": "reasoning",
    "competitor_discovery": "reasoning",
    "website_analysis": "reasoning",
    "product_analysis": "reasoning",
    "pricing_analysis": "reasoning",
    "marketing_analysis": "reasoning",
    "seo_analysis": "reasoning",
    "social_analysis": "reasoning",
    "review_analysis": "reasoning",
    "technology_analysis": "reasoning",
    "market_analysis": "reasoning",
    "comparison": "reasoning",
    "synthesis": "reasoning",
    # -- the output a person reads
    "final_report": "final",
}

#: Complexity levels a caller may request.
COMPLEXITIES: tuple[str, ...] = ("low", "high")


class ModelRouter:
    """Resolves a task type to a configured model id."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings if settings is not None else get_settings()

    def role_for(self, task_type: str, *, complexity: str = "low") -> str:
        """The profile a task runs on, before any model id is looked up.

        Exposed separately from :meth:`select` so a caller that only wants to
        know the tier — for logging or budgeting — does not have to hold a
        model id it has no use for.
        """
        if task_type not in TASK_ROLES:
            raise ValidationError(
                f"Unknown task type {task_type!r}. Add it to TASK_ROLES or the "
                f"job will be routed by guesswork. Known types: "
                f"{', '.join(sorted(TASK_ROLES))}."
            )
        if complexity not in COMPLEXITIES:
            raise ValidationError(
                f"Unknown complexity {complexity!r}; expected one of "
                f"{', '.join(COMPLEXITIES)}."
            )
        role = TASK_ROLES[task_type]
        if complexity == "high" and role == "fast":
            # A caller who says this step is harder than usual gets the
            # reasoning tier. The cap is deliberate: nothing is promoted to
            # ``final`` by complexity, so raising the complexity of a cheap step
            # can never quietly promote it to the model reserved for the report.
            role = "reasoning"
        return role

    def select(self, task_type: str, *, complexity: str = "low") -> str:
        """The configured model id for ``task_type``.

        Always a value from :class:`~app.core.config.Settings`; this method
        never constructs a model id.
        """
        role = self.role_for(task_type, complexity=complexity)
        model = self._settings.model_for_role(role)
        log.debug(
            "model selected", extra={"task_type": task_type, "role": role, "model": model}
        )
        return model
