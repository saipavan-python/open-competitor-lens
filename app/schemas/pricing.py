"""Pricing schemas.

The governing rule: a price is recorded only when it is publicly displayed.
Hidden pricing becomes ``price=None`` plus an explicit ``price_display`` string
and a ``pricing_model`` of "Contact sales" - never an estimate.
"""

from __future__ import annotations

from pydantic import Field, field_validator

from app.core.constants import Confidence
from app.schemas.common import NOT_PUBLIC, DomainModel


class PricingPlan(DomainModel):
    """One publicly listed plan."""

    name: str
    price: float | None = None
    currency: str | None = None
    billing_period: str | None = None
    price_display: str | None = None
    features: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    is_free: bool = False
    is_contact_sales: bool = False
    evidence_ids: list[str] = Field(default_factory=list)

    @field_validator("price")
    @classmethod
    def _price_non_negative(cls, value: float | None) -> float | None:
        if value is not None and value < 0:
            raise ValueError("A published price cannot be negative")
        return value

    @field_validator("name")
    @classmethod
    def _name_required(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("PricingPlan.name is required")
        return value.strip()

    def display(self) -> str:
        """Human-readable price, never a fabricated number."""
        if self.price_display:
            return self.price_display
        if self.is_free:
            return "Free"
        if self.is_contact_sales or self.price is None:
            return "Contact sales"
        return f"{self.price} {self.currency or ''}".strip()


class PricingAnalysis(DomainModel):
    """The pricing agent's output for one company."""

    company: str
    plans: list[PricingPlan] = Field(default_factory=list)
    pricing_model: str = NOT_PUBLIC
    billing_periods: list[str] = Field(default_factory=list)
    has_free_tier: bool | None = None
    has_trial: bool | None = None
    has_enterprise_tier: bool | None = None
    pricing_page_url: str | None = None
    currency: str | None = None
    notes: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    confidence: Confidence = Confidence.LOW

    @property
    def lowest_published_price(self) -> float | None:
        prices = [plan.price for plan in self.plans if plan.price is not None]
        return min(prices) if prices else None
