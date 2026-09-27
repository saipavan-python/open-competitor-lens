"""SQLAlchemy models. Importing this package registers every table on ``Base``."""

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
]
