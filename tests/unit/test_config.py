"""RED: configuration, exception hierarchy, and logging setup."""

from __future__ import annotations

import logging

import pytest
from pydantic import ValidationError as PydanticValidationError

from app.core.config import Settings, get_settings, reset_settings_cache
from app.core.exceptions import (
    AppError,
    ConfigurationError,
    ProviderError,
    ProviderNotConfiguredError,
    RateLimitError,
    SSRFError,
    ValidationError,
)
from app.core.logging import configure_logging, get_logger


class TestSettingsDefaults:
    def test_loads_without_any_env_file(self) -> None:
        settings = get_settings()
        assert settings.app_name == "AI Competitor Analysis Agent"

    def test_budget_defaults_match_spec(self) -> None:
        settings = get_settings()
        assert settings.max_search_results == 10
        assert settings.max_queries == 20
        assert settings.max_pages_per_competitor == 40
        assert settings.max_search_queries_per_competitor == 15
        assert settings.max_browser_steps == 20
        assert settings.max_report_repair_rounds == 2
        assert settings.max_concurrent_requests == 5
        assert settings.max_concurrent_competitors == 3
        assert settings.max_concurrent_agents == 5

    def test_timeout_defaults_match_spec(self) -> None:
        settings = get_settings()
        assert settings.http_timeout == 30
        assert settings.search_timeout == 30
        assert settings.browser_timeout == 60
        assert settings.llm_timeout == 120
        assert settings.crawl_timeout == 300

    def test_document_limits_match_spec(self) -> None:
        settings = get_settings()
        assert settings.max_document_bytes == 5_000_000
        assert settings.max_document_chars == 100_000
        assert settings.max_llm_tokens == 12_000

    def test_feature_flags_default_off(self) -> None:
        settings = get_settings()
        assert settings.graph_rag_enabled is False
        assert settings.langchain_tracing_v2 is False

    def test_offline_mode_used_when_no_llm_key(self) -> None:
        assert get_settings().resolved_llm_provider == "offline"

    def test_openai_selected_when_key_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-value")
        reset_settings_cache()
        assert get_settings().resolved_llm_provider == "openai"


class TestSettingsCacheIsolation:
    def test_get_settings_returns_same_instance(self) -> None:
        assert get_settings() is get_settings()

    def test_reset_cache_reloads_from_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        first = get_settings()
        monkeypatch.setenv("MAX_PAGES_PER_COMPETITOR", "7")
        reset_settings_cache()
        second = get_settings()
        assert second is not first
        assert second.max_pages_per_competitor == 7


class TestSettingsValidation:
    def test_rejects_non_positive_page_budget(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MAX_PAGES_PER_COMPETITOR", "0")
        reset_settings_cache()
        with pytest.raises(PydanticValidationError):
            get_settings()

    def test_research_depth_presets_are_ordered(self) -> None:
        settings = get_settings()
        assert settings.pages_for_depth("quick") < settings.pages_for_depth("standard")
        assert settings.pages_for_depth("standard") < settings.pages_for_depth("deep")

    def test_settings_is_typed(self) -> None:
        assert isinstance(get_settings(), Settings)


class TestExceptionHierarchy:
    @pytest.mark.parametrize(
        "exc_type",
        [
            ConfigurationError,
            ProviderNotConfiguredError,
            ProviderError,
            RateLimitError,
            SSRFError,
            ValidationError,
        ],
    )
    def test_all_errors_subclass_app_error(self, exc_type: type[AppError]) -> None:
        assert issubclass(exc_type, AppError)

    def test_error_carries_diagnostic_fields(self) -> None:
        error = ProviderError("boom", retryable=True, provider="tavily", step="search")
        assert error.retryable is True
        assert error.provider == "tavily"
        assert error.step == "search"
        assert str(error) == "boom"

    def test_error_defaults_to_not_retryable(self) -> None:
        assert ProviderError("boom").retryable is False

    def test_rate_limit_is_retryable(self) -> None:
        assert RateLimitError("429").retryable is True

    def test_not_configured_names_the_missing_variable(self) -> None:
        error = ProviderNotConfiguredError("tavily", "TAVILY_API_KEY")
        assert "TAVILY_API_KEY" in str(error)
        assert error.retryable is False

    def test_to_dict_is_json_safe_and_hides_nothing_sensitive(self) -> None:
        payload = ProviderError("boom", provider="openai").to_dict()
        assert payload["type"] == "ProviderError"
        assert payload["message"] == "boom"
        assert payload["provider"] == "openai"


class TestLogging:
    def test_configure_logging_is_idempotent(self) -> None:
        configure_logging()
        configure_logging()
        assert logging.getLogger("app").handlers

    def test_logger_binds_run_context(self) -> None:
        configure_logging()
        log = get_logger(__name__, run_id="abc", node="discover_competitors")
        log.info("hello", extra={"competitor_count": 5})
        assert log is not None
