"""SQLAlchemy models. Importing this package registers every table on ``Base``."""

from __future__ import annotations

from sqlalchemy import MetaData

from app.core.database import Database
from app.models.analysis import (
    MarketInsightModel,
    OpportunityModel,
    PricingPlanModel,
    ProductModel,
    ReportModel,
    ReviewModel,
    SocialProfileModel,
    TechnologySignalModel,
)
from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.models.company import CompanyModel, CompetitorModel
from app.models.research_run import ResearchQuestionModel, ResearchRunModel
from app.models.source import (
    DocumentModel,
    EvidenceModel,
    FindingModel,
    SourceModel,
)


async def create_schema(db: Database, *, metadata: MetaData | None = None) -> None:
    """Create every table on a :class:`~app.core.database.Database`.

    Lives here rather than on ``Database`` itself because ``core`` is a lower
    layer and cannot import the models it would be creating. Production
    deployments should use Alembic; this is the developer and test path.
    """
    await db.create_all(metadata or Base.metadata)


async def drop_schema(db: Database, *, metadata: MetaData | None = None) -> None:
    await db.drop_all(metadata or Base.metadata)


__all__ = [
    "Base",
    "CompanyModel",
    "CompetitorModel",
    "DocumentModel",
    "EvidenceModel",
    "FindingModel",
    "MarketInsightModel",
    "OpportunityModel",
    "PricingPlanModel",
    "ProductModel",
    "ReportModel",
    "ResearchQuestionModel",
    "ResearchRunModel",
    "ReviewModel",
    "SocialProfileModel",
    "SourceModel",
    "TechnologySignalModel",
    "TimestampMixin",
    "UUIDPrimaryKeyMixin",
    "create_schema",
    "drop_schema",
]
