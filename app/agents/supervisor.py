"""Deciding whether a run that lost a step is still worth finishing.

A pipeline that aborts on the first failure produces nothing, and a pipeline that
never fails produces a report with holes in it and no mention of them. The
supervisor is the point between those two failures: it decides which steps were
load-bearing and which were merely disappointing.

The distinction that matters is between a step everything downstream depends on
and a step that fills one section of one report. Failing to identify the target
company means there is no report to write, so the run stops. Failing to analyse
customer reviews means one section says the data was not available — a worse
report, but a usable one, and one that is honest about the gap.

So ``FAILED`` is reserved for a hard stop, ``PARTIAL`` for a run that completed
with gaps, and only genuinely completed work reports ``COMPLETED``. A step that
later succeeds clears its own failure, so a retried node that recovers does not
leave a stale error in the report.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.constants import RunStatus
from app.core.logging import get_logger

log = get_logger(__name__)

#: Steps without which the run has nothing to report. Failing one of these ends
#: the run; every other step is a section of a report rather than the report.
HARD_DEPENDENCIES: frozenset[str] = frozenset(
    {
        "company_identification",
        "competitor_discovery",
    }
)


@dataclass
class Supervisor:
    """Tracks what a run completed, failed and warned about.

    Mutable state on purpose: it is the run's own record of itself, updated as
    nodes finish, and the graph holds one for the length of a run.
    """

    failed_steps: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    completed: set[str] = field(default_factory=set)
    hard_failures: set[str] = field(default_factory=set)

    @property
    def status(self) -> RunStatus:
        """The run status implied by what has happened so far."""
        if self.hard_failures:
            return RunStatus.FAILED
        if self.failed_steps or self.warnings:
            return RunStatus.PARTIAL
        return RunStatus.COMPLETED

    @property
    def can_continue(self) -> bool:
        """Whether the remaining steps can still produce something worth reading."""
        return not self.hard_failures

    @property
    def progress(self) -> int:
        """Progress as a percentage, never negative and never above 100.

        A hard failure reports 0: the report it would have produced does not
        exist, and a percentage implying mostly-done work would misdescribe a run
        that stopped before it began.
        """
        if self.hard_failures:
            return 0
        if self.warnings:
            return 100
        return 100

    def record_ok(self, step: str) -> None:
        """A step finished. Clears any earlier failure of the same step."""
        self.completed.add(step)
        self.failed_steps = [item for item in self.failed_steps if not item.startswith(f"{step}:")]
        self.hard_failures.discard(step)
        log.debug("step completed", extra={"step": step})

    def record_failure(self, step: str, reason: str) -> None:
        """A step could not do its job.

        Recorded once per step: a retried node that fails the same way should not
        put the same error in the report twice.
        """
        entry = f"{step}: {reason}"
        if not any(item.startswith(f"{step}:") for item in self.failed_steps):
            self.failed_steps.append(entry)
        if step in HARD_DEPENDENCIES:
            self.hard_failures.add(step)
        log.warning("step failed", extra={"step": step, "reason": reason})

    def record_warning(self, step: str, message: str) -> None:
        """A step finished with a shortfall worth carrying into the report."""
        entry = f"{step}: {message}"
        if entry not in self.warnings:
            self.warnings.append(entry)
        log.info("step warning", extra={"step": step, "warning": message})

    def summary(self) -> dict[str, object]:
        """The run's own account of itself, for the report and the API."""
        return {
            "status": self.status.value,
            "progress": self.progress,
            "failed_steps": list(self.failed_steps),
            "warnings": list(self.warnings),
            "completed": sorted(self.completed),
        }
