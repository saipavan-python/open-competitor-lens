"""The model router decides which model does which job.

Routing is a cost decision, and the failure mode is not "too expensive" -- it is
"a cheap model silently did the hardest job in the run". A final report written
by a small model is cheaper than the report is worth, and nobody notices until
a reader notices.

So the tests here are about the *mapping*, not the model names. A router that
hardcodes ``gpt-4o`` passes a naive "does it return a model id" test and fails
the moment an operator configures a cheaper deployment, which is the only
reason for having settings at all.
"""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.core.exceptions import ValidationError
from app.services.model_router import TASK_ROLES, ModelRouter


@pytest.fixture()
def settings() -> Settings:
    return Settings(
        fast_model="fast-model-x",
        reasoning_model="reasoning-model-x",
        final_model="final-model-x",
    )


@pytest.fixture()
def router(settings: Settings) -> ModelRouter:
    return ModelRouter(settings)


class TestRoutingTable:
    def test_query_generation_and_the_final_report_use_different_models(
        self, router: ModelRouter
    ) -> None:
        assert router.select("query_generation") != router.select("final_report")

    def test_every_selected_model_is_a_value_present_in_settings(
        self, router: ModelRouter, settings: Settings
    ) -> None:
        known = {settings.fast_model, settings.reasoning_model, settings.final_model}
        for task_type in TASK_ROLES:
            assert router.select(task_type) in known, task_type

    def test_the_router_invents_no_model_id_of_its_own(
        self, router: ModelRouter, settings: Settings
    ) -> None:
        """A model id the router supplied itself would be unauditable."""
        known = {settings.fast_model, settings.reasoning_model, settings.final_model}
        for task_type in TASK_ROLES:
            for complexity in ("low", "high"):
                assert router.select(task_type, complexity=complexity) in known

    def test_an_unknown_task_type_is_refused_rather_than_guessed(
        self, router: ModelRouter
    ) -> None:
        """Silently routing an unrecognised job to the fast model is the bug."""
        with pytest.raises(ValidationError) as excinfo:
            router.select("task_that_does_not_exist")
        assert "task_that_does_not_exist" in str(excinfo.value)

    def test_a_task_type_may_declare_its_own_role(self) -> None:
        assert TASK_ROLES["final_report"] == "final"
        assert TASK_ROLES["query_generation"] == "fast"


class TestComplexity:
    def test_a_high_complexity_request_is_never_routed_to_the_cheap_model(
        self, router: ModelRouter
    ) -> None:
        assert router.select("query_generation", complexity="high") != router.select(
            "query_generation"
        )

    def test_raising_complexity_never_downgrades_the_final_report(
        self, router: ModelRouter
    ) -> None:
        """The final report is already the top tier; complexity cannot exceed it."""
        assert router.select("final_report", complexity="high") == router.select(
            "final_report"
        )

    def test_an_unknown_complexity_is_refused(self, router: ModelRouter) -> None:
        with pytest.raises(ValidationError):
            router.select("query_generation", complexity="extreme")


class TestConfiguration:
    def test_changing_the_settings_changes_the_routing(
        self, router: ModelRouter, settings: Settings
    ) -> None:
        """Proves the router reads settings rather than a private constant."""
        before = router.select("query_generation")
        settings.fast_model = "a-different-fast-model"
        assert router.select("query_generation") == "a-different-fast-model"
        assert before == "fast-model-x"

    def test_the_router_does_not_need_credentials_to_route(
        self, settings: Settings
    ) -> None:
        """Routing is a pure lookup and must work before any provider is set up."""
        unconfigured = Settings(openai_api_key=None, gemini_api_key=None)
        assert ModelRouter(unconfigured).select("final_report") == unconfigured.final_model
