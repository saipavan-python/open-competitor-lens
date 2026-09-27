"""Provider-facing schemas: what every LLM call is handed and must return."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ModelRole = Literal["fast", "reasoning", "final", "default"]


class ChatMessage(BaseModel):
    """One message in a provider request."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str


class LLMResponse(BaseModel):
    """A provider's answer, with the metadata needed to audit cost."""

    model_config = ConfigDict(extra="forbid")

    content: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_ms: int = 0
    finish_reason: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMRequest(BaseModel):
    """Everything a provider needs for one generation call."""

    model_config = ConfigDict(extra="forbid")

    messages: list[ChatMessage]
    model_role: ModelRole = "default"
    temperature: float | None = None
    max_tokens: int | None = None
    response_schema: dict[str, Any] | None = None
    timeout: int | None = None


class ProviderResult(BaseModel):
    """Envelope stamped onto every provider response for observability."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    provider: str
    model: str | None = None
    duration_ms: int = 0
    cached: bool = False
    degraded: bool = False
    warnings: list[str] = Field(default_factory=list)
    payload: Any = None


class RetrievedChunk(BaseModel):
    """One chunk returned by the vector store, with its citation metadata."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    id: str
    text: str
    score: float = 0.0
    source_id: str | None = None
    url: str | None = None
    company: str | None = None
    competitor: str | None = None
    page_type: str | None = None
    category: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class GraphEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    type: str
    properties: dict[str, Any] = Field(default_factory=dict)
    source_ids: list[str] = Field(default_factory=list)


class GraphRelationship(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    type: str
    start_id: str
    end_id: str
    properties: dict[str, Any] = Field(default_factory=dict)
    source_ids: list[str] = Field(default_factory=list)
    confidence: str = "low"


class ChangeEvent(BaseModel):
    """A detected change between two observations of a competitor."""

    model_config = ConfigDict(extra="forbid")

    company: str
    change_type: str
    previous_value: dict[str, Any] = Field(default_factory=dict)
    new_value: dict[str, Any] = Field(default_factory=dict)
    detected_at: str
    source: str | None = None
    previous_hash: str | None = None
    new_hash: str | None = None


class CrawlTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    priority: int = 100
    reason: str | None = None
    depth: int = 0


class CrawlBudget(BaseModel):
    """Limits for one crawl.

    ``timeout`` is a float because it is compared against a monotonic clock:
    whole-second resolution cannot express a half-second budget, and truncating
    it would make short budgets silently un-enforceable.
    """

    model_config = ConfigDict(extra="forbid")

    max_pages: int = 40
    max_concurrent: int = 5
    timeout: float = 300.0
    max_document_bytes: int = 5_000_000


class FetchResult(BaseModel):
    """One retrieved page.

    A retrieval that failed is still a ``FetchResult``, not an exception: a
    competitor having an unreachable pricing page is a finding, not a crash, and
    the report has to be able to say so. ``retrieval_status`` and the reason
    fields are what let it distinguish "the page says nothing" from "we could not
    read the page" — a distinction that matters a great deal to the spec's
    anti-hallucination rules.
    """

    model_config = ConfigDict(extra="forbid")

    url: str
    final_url: str | None = None
    status: int | None = None
    content_type: str | None = None
    text: str = ""
    html: str = ""
    title: str | None = None
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    retrieval_status: Literal["ok", "failed", "blocked", "not_found", "skipped"] = "ok"
    content_hash: str | None = None
    error: str | None = None
    provider: str | None = None
    duration_ms: int = 0
    page_type: str | None = None
    content_length: int = 0

    @property
    def ok(self) -> bool:
        return self.retrieval_status == "ok"

    @property
    def is_snippet_only(self) -> bool:
        """A fetch result is retrieved content, never a search snippet."""
        return False
