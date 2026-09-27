"""Report repository re-export.

``ReportRepository`` lives beside the evidence repository because a report and
its evidence are written in the same transaction. This module keeps the
documented import path ``app.repositories.report_repository`` working.
"""

from app.repositories.evidence_repository import ReportRepository

__all__ = ["ReportRepository"]
