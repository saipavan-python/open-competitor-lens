"""Social presence schemas.

``public_metrics`` is ``None`` unless a figure was actually read off a public
page during this run. It is never populated from model memory.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from app.core.constants import SOCIAL_PLATFORMS, Confidence
from app.schemas.common import DomainModel

PlatformName = Literal[
    "linkedin", "youtube", "instagram", "facebook", "x", "tiktok", "github", "other"
]


class SocialProfile(DomainModel):
    """One official public profile found for a company."""

    platform: PlatformName
    profile_url: str
    handle: str | None = None
    content_themes: list[str] = Field(default_factory=list)
    visible_activity: str | None = None
    public_metrics: dict[str, int] | None = None
    metrics_observed_at: str | None = None
    observations: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)

    @field_validator("profile_url")
    @classmethod
    def _url_required(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("SocialProfile.profile_url is required")
        return value.strip()


class SocialAnalysis(DomainModel):
    """All official public social profiles found for one company."""

    company: str
    profiles: list[SocialProfile] = Field(default_factory=list)
    not_found: list[str] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    confidence: Confidence = Confidence.LOW
    evidence: list[str] = Field(default_factory=list)

    @property
    def platforms_present(self) -> list[str]:
        return [profile.platform for profile in self.profiles]


def platform_for_url(url: str) -> PlatformName | None:
    """Map a profile URL to a known platform name, or ``None``."""
    lowered = url.casefold()
    for name, host in SOCIAL_PLATFORMS.items():
        if host in lowered:
            return name  # type: ignore[return-value]
    return None
