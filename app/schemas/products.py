"""Product, service and feature schemas."""

from __future__ import annotations

from pydantic import Field, field_validator

from app.core.constants import Confidence
from app.schemas.common import NOT_PUBLIC, DomainModel


class ProductOffering(DomainModel):
    """One product or service line as publicly described."""

    name: str
    description: str | None = None
    features: list[str] = Field(default_factory=list)
    target_segments: list[str] = Field(default_factory=list)
    use_cases: list[str] = Field(default_factory=list)
    deployment_model: str | None = None
    business_model: str | None = None
    pricing_model: str | None = None
    differentiators: list[str] = Field(default_factory=list)
    url: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _name_required(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("ProductOffering.name is required")
        return value.strip()


class ProductAnalysis(DomainModel):
    """The product agent's output for one company."""

    company: str
    products: list[ProductOffering] = Field(default_factory=list)
    services: list[str] = Field(default_factory=list)
    features: list[str] = Field(default_factory=list)
    segments: list[str] = Field(default_factory=list)
    use_cases: list[str] = Field(default_factory=list)
    differentiators: list[str] = Field(default_factory=list)
    positioning: str = NOT_PUBLIC
    notes: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    confidence: Confidence = Confidence.LOW

    @field_validator("company")
    @classmethod
    def _company_required(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("ProductAnalysis.company is required")
        return value.strip()

    @property
    def feature_set(self) -> set[str]:
        """Union of all features across offerings, for feature comparison."""
        collected: set[str] = set(self.features)
        for product in self.products:
            collected.update(feature.casefold() for feature in product.features)
        return collected
