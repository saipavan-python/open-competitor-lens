"""Application configuration.

Every tunable value in the system lives here. Nothing else in the codebase may
hardcode a timeout, a model name, a concurrency limit or a credential.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, computed_field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: The role-to-model settings. Used to tell "the deployment chose this model for
#: this role" apart from "this is the default, and a vendor model should win".
_ROLE_FIELDS = frozenset({"fast_model", "reasoning_model", "final_model"})

ResearchDepth = Literal["quick", "standard", "deep"]

#: Page budgets per research depth (spec section 96-98). Used by
#: :meth:`Settings.pages_for_depth` so a run never carries its own constants.
DEPTH_PAGE_BUDGETS: dict[str, int] = {
    "quick": 10,
    "standard": 40,
    "deep": 120,
}


class Settings(BaseSettings):
    """Runtime configuration loaded from the environment or a ``.env`` file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------ app
    app_env: Literal["development", "staging", "production", "test"] = "development"
    app_name: str = "AI Competitor Analysis Agent"
    log_level: str = "INFO"
    log_json: bool = False
    api_key: str | None = None
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # ------------------------------------------------------------- database
    database_url: str = "sqlite+aiosqlite:///./competitor_analysis.db"
    db_echo: bool = False

    # ------------------------------------------------------------- providers
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    #: Overrides the API endpoint. Set for any OpenAI-compatible gateway, proxy
    #: or local model server; the adapter speaks chat completions, so anything
    #: implementing that protocol works without a second provider. ``None`` uses
    #: the SDK default.
    openai_base_url: str | None = None
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.0-flash"
    tavily_api_key: str | None = None
    firecrawl_api_key: str | None = None

    llm_provider: str = "auto"
    search_provider: str = "auto"
    web_provider: str = "auto"
    vector_store_provider: str = "auto"
    graph_store_provider: str = "memory"

    # --------------------------------------------------------- model routing
    #: Defaults are OpenAI ids, which is the wrong vendor's namespace for a
    #: Gemini or gateway deployment. ``_seed_roles_from_vendor_model`` replaces
    #: them when the deployment has not chosen roles itself.
    fast_model: str = "gpt-4o-mini"
    reasoning_model: str = "gpt-4o"
    final_model: str = "gpt-4o"

    @model_validator(mode="after")
    def _seed_roles_from_vendor_model(self) -> Settings:
        """Let the configured vendor model stand in for unset role defaults.

        Every model call goes through ``model_for_role``, so the role map decides
        which model actually runs — and it defaults to OpenAI ids. A deployment
        that set ``GEMINI_API_KEY`` and ``GEMINI_MODEL`` therefore selected the
        Gemini adapter and then asked it for ``gpt-4o``: the vendor model setting
        was dead, and the run failed on a model that vendor has never heard of.
        The same applies to any OpenAI-compatible gateway, where ``gpt-4o`` need
        not exist.

        Seeded per role rather than all at once, so a deployment that overrides
        just one role gets the vendor model for the rest instead of the stock
        OpenAI defaults. An explicitly configured role is never touched, which is
        what makes deliberately mixing vendors per role possible.
        """
        vendor_model = {
            "openai": self.openai_model,
            "gemini": self.gemini_model,
        }.get(self.resolved_llm_provider)
        if not vendor_model:
            return self
        for role in _ROLE_FIELDS:
            if role not in self.model_fields_set:
                setattr(self, role, vendor_model)
        return self

    # ----------------------------------------------------------- vector/graph
    chroma_persist_directory: str = "./data/chroma"
    graph_rag_enabled: bool = False
    enable_browser_use: bool = True
    enable_reranking: bool = False

    # ---------------------------------------------------------------- tracing
    langchain_tracing_v2: bool = False
    langchain_api_key: str | None = None
    langchain_project: str = "ai-competitor-analysis"

    # ----------------------------------------------------------------- redis
    redis_url: str | None = None

    # ----------------------------------------------------------------- budget
    max_search_results: int = Field(default=10, gt=0)
    max_queries: int = Field(default=20, gt=0)
    max_search_queries_per_competitor: int = Field(default=15, gt=0)
    max_pages_per_competitor: int = Field(default=40, gt=0)
    max_document_bytes: int = Field(default=5_000_000, gt=0)
    max_document_chars: int = Field(default=100_000, gt=0)
    max_llm_tokens: int = Field(default=12_000, gt=0)
    max_browser_steps: int = Field(default=20, gt=0)
    max_report_repair_rounds: int = Field(default=2, ge=0)

    # ------------------------------------------------------------ concurrency
    max_concurrent_requests: int = Field(default=5, gt=0)
    max_concurrent_competitors: int = Field(default=3, gt=0)
    max_concurrent_agents: int = Field(default=5, gt=0)

    # --------------------------------------------------------------- timeouts
    http_timeout: int = Field(default=30, gt=0)
    search_timeout: int = Field(default=30, gt=0)
    browser_timeout: int = Field(default=60, gt=0)
    llm_timeout: int = Field(default=120, gt=0)
    crawl_timeout: int = Field(default=300, gt=0)

    # --------------------------------------------------------------- research
    research_depth: ResearchDepth = "standard"
    report_language: str = "en"
    competitor_count: int = Field(default=5, ge=1, le=50)
    max_redirects: int = Field(default=3, ge=0)
    allowed_url_ports: list[int] = Field(default_factory=lambda: [80, 443, 8080, 8443])
    #: Identifies the crawler honestly. This used to advertise "respects
    #: robots.txt", and nothing in the codebase ever fetched or checked
    #: robots.txt. Site operators see this string in their logs and may make
    #: access decisions on the strength of it, so it states only what is true.
    user_agent: str = "open-competitor-lens/0.1 (+competitive research)"

    @field_validator("cors_origins", "allowed_url_ports", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        """Accept ``a,b,c`` from the environment as well as a JSON list."""
        if isinstance(value, str) and not value.strip().startswith("["):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value

    @field_validator("log_level")
    @classmethod
    def _upper_level(cls, value: str) -> str:
        return value.upper()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def resolved_llm_provider(self) -> str:
        """Pick a concrete LLM provider from config + available credentials."""
        if self.llm_provider != "auto":
            return self.llm_provider
        if self.openai_api_key:
            return "openai"
        if self.gemini_api_key:
            return "gemini"
        return "offline"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def resolved_search_provider(self) -> str:
        if self.search_provider != "auto":
            return self.search_provider
        if self.tavily_api_key:
            return "tavily"
        if self.firecrawl_api_key:
            return "firecrawl"
        return "offline"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def resolved_web_provider(self) -> str:
        if self.web_provider != "auto":
            return self.web_provider
        if self.firecrawl_api_key:
            return "firecrawl"
        return "http"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def resolved_vector_store_provider(self) -> str:
        if self.vector_store_provider != "auto":
            return self.vector_store_provider
        return "chroma"

    def pages_for_depth(self, depth: str | None = None) -> int:
        """Page budget for a research depth, clamped by the global maximum."""
        level = depth or self.research_depth
        return min(DEPTH_PAGE_BUDGETS.get(level, DEPTH_PAGE_BUDGETS["standard"]),
                   self.max_pages_per_competitor * self.competitor_count)

    def effective_page_budget(self, depth: str | None = None) -> int:
        """Per-competitor page budget after applying the depth preset."""
        return max(1, min(self.pages_for_depth(depth), self.max_pages_per_competitor))

    def model_for_role(self, role: str) -> str:
        """Resolve a model role (``fast``/``reasoning``/``final``) to a model id."""
        return {"fast": self.fast_model, "reasoning": self.reasoning_model,
                "final": self.final_model}.get(role, self.fast_model)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()


def reset_settings_cache() -> None:
    """Drop the cached settings. Tests call this after mutating the env."""
    get_settings.cache_clear()
