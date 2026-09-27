"""Company profile and identity schemas."""

from __future__ import annotations

from pydantic import Field, field_validator

from app.core.constants import Confidence
from app.schemas.common import DomainModel


class CompanyIdentity(DomainModel):
    """Normalised identity for a company, so aliases do not create duplicates.

    "Acme", "Acme Ltd.", "acme.com" and "CompanyX" are one entity. The
    competitor validator compares ``identity_key`` to avoid validating a company
    as a competitor of itself.
    """

    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)
    country: str | None = None
    industry: str | None = None

    @field_validator("canonical_name")
    @classmethod
    def _name_required(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("CompanyIdentity.canonical_name is required")
        return value.strip()


class CompanyProfile(DomainModel):
    """What public research established about the target company."""

    name: str
    official_website: str | None = None
    description: str | None = None
    industry: str | None = None
    country: str | None = None
    products: list[str] = Field(default_factory=list)
    services: list[str] = Field(default_factory=list)
    target_segments: list[str] = Field(default_factory=list)
    value_proposition: str | None = None
    confidence: Confidence = Confidence.LOW
    evidence: list[str] = Field(default_factory=list)
    identity: CompanyIdentity | None = None
    discovery_queries: list[str] = Field(default_factory=list)
    candidate_domains: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _name_required(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("CompanyProfile.name is required")
        return value.strip()
