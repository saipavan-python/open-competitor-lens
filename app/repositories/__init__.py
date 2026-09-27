"""Persistence layer. Each repository wraps one table and owns its queries."""

from app.repositories.company_repository import CompanyRepository
from app.repositories.competitor_repository import CompetitorRepository
from app.repositories.evidence_repository import (
    EvidenceRepository,
    FindingRepository,
    ReportRepository,
)
from app.repositories.research_repository import ResearchRunRepository
from app.repositories.source_repository import SourceRepository

__all__ = [
    "CompanyRepository",
    "CompetitorRepository",
    "EvidenceRepository",
    "FindingRepository",
    "ReportRepository",
    "ResearchRunRepository",
    "SourceRepository",
]
