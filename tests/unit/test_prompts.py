"""The prompt contract: what every prompt must carry, whatever it is for.

Each prompt in this package does a different job, so the interesting properties
are the ones that are true of *all* of them. A prompt that quietly omits the
evidence block is not caught by anything downstream — the model will simply
answer from memory, fluently, and the result will look like every other step in
the run. The only place that failure is visible is here.

The tests are deliberately structural. They do not check that a prompt says
anything in particular about pricing or SEO; they check that every prompt is
handed the evidence, is told what shape to answer in, cannot carry a credential,
and is bound to a real output schema and a real routing table entry.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from app.core.constants import ClaimType, Confidence, RetrievalStatus, SourceType
from app.prompts import PROMPTS, build, get, names
from app.prompts.base import (
    COMMON_INSTRUCTION,
    EVIDENCE_HEADER,
    SCHEMA_HEADER,
    PromptContext,
    PromptLeakError,
    assert_no_secrets,
)
from app.schemas.evidence import Evidence, Source
from app.services.model_router import TASK_ROLES


def _source(source_id: str = "src-1") -> Source:
    return Source(
        id=source_id,
        url=f"https://acme.com/{source_id}",
        title="Pricing",
        text="Acme Cloud plans start at $10 per seat per month.",
        retrieval_status=RetrievalStatus.OK,
        source_type=SourceType.OFFICIAL_SITE,
        company="Acme",
    )


def _evidence(source_id: str = "src-1") -> Evidence:
    return Evidence(
        id=f"ev-{source_id}",
        claim="Acme charges $10 per seat",
        source_id=source_id,
        source_url=f"https://acme.com/{source_id}",
        evidence_text="start at $10 per seat",
        claim_type=ClaimType.OBSERVED,
        confidence=Confidence.HIGH,
        claim_key="acme.price",
    )


def _ctx(**kwargs: object) -> PromptContext:
    defaults: dict[str, object] = {
        "evidence": [_evidence()],
        "sources": [_source()],
    }
    defaults.update(kwargs)
    return PromptContext(
        company="Acme",
        research_goal="compare pricing",
        **defaults,  # type: ignore[arg-type]
    )


ALL = names()


class TestRegistry:
    def test_the_plan_specifies_seventeen_prompts(self) -> None:
        assert len(PROMPTS) == 17

    def test_every_registered_prompt_exposes_the_contract(self) -> None:
        for name in ALL:
            module = get(name)
            assert callable(module.build), name
            assert isinstance(module.TASK_TYPE, str), name
            assert issubclass(module.OUTPUT_SCHEMA, BaseModel), name
            assert module.SECTIONS, name

    def test_every_prompt_routes_to_a_configured_role(self) -> None:
        """A prompt with no routing entry would fail at send time, not here."""
        for name in ALL:
            assert get(name).TASK_TYPE in TASK_ROLES, name

    def test_an_unknown_prompt_name_is_refused_with_a_useful_message(self) -> None:
        with pytest.raises(KeyError) as excinfo:
            get("no_such_prompt")
        assert "final_report" in str(excinfo.value)

    def test_build_renders_a_registered_prompt(self) -> None:
        messages = build("final_report", _ctx())
        assert [m.role for m in messages] == ["system", "user"]


class TestRequiredSections:
    def test_every_prompt_carries_the_evidence_header(self) -> None:
        for name in ALL:
            rendered = "\n".join(m.content for m in get(name).build(_ctx()))
            assert EVIDENCE_HEADER in rendered, name

    def test_every_prompt_carries_the_output_schema_header(self) -> None:
        for name in ALL:
            rendered = "\n".join(m.content for m in get(name).build(_ctx()))
            assert SCHEMA_HEADER in rendered, name

    def test_every_prompt_carries_the_evidence_rules(self) -> None:
        for name in ALL:
            rendered = "\n".join(m.content for m in get(name).build(_ctx()))
            assert "Use only the sources" in rendered, name

    def test_every_prompt_names_its_output_schema(self) -> None:
        for name in ALL:
            rendered = "\n".join(m.content for m in get(name).build(_ctx()))
            assert get(name).OUTPUT_SCHEMA.__name__ in rendered, name


class TestEvidenceRendering:
    def test_evidence_reaches_the_model_verbatim(self) -> None:
        rendered = "\n".join(m.content for m in build("final_report", _ctx()))
        assert "start at $10 per seat" in rendered
        assert "src-1" in rendered

    def test_evidence_indices_are_stable_and_distinct(self) -> None:
        ctx = _ctx(
            evidence=[_evidence("src-1"), _evidence("src-2")],
            sources=[_source("src-1"), _source("src-2")],
        )
        rendered = "\n".join(m.content for m in build("final_report", ctx))
        assert "[E1]" in rendered and "[E2]" in rendered

    def test_an_empty_evidence_block_says_so(self) -> None:
        """An empty block must read as empty, not be omitted."""
        ctx = PromptContext(company="Acme")
        rendered = "\n".join(m.content for m in build("final_report", ctx))
        assert "none retrieved" in rendered

    def test_the_research_goal_reaches_the_model(self) -> None:
        rendered = "\n".join(m.content for m in build("final_report", _ctx()))
        assert "compare pricing" in rendered


class TestNoCredentialLeaks:
    def test_the_guard_catches_a_planted_secret(self, monkeypatch: pytest.MonkeyPatch
                                              ) -> None:
        monkeypatch.setenv("ACME_TEST_API_KEY", "sk-planted-value-1234")
        with pytest.raises(PromptLeakError) as excinfo:
            assert_no_secrets("connecting with sk-planted-value-1234")
        assert "ACME_TEST_API_KEY" in str(excinfo.value)

    def test_the_guard_ignores_short_and_non_secret_values(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ACME_TEST_API_KEY", "short")
        monkeypatch.setenv("ACME_TEST_TOKEN", "a" * 20)
        assert assert_no_secrets("nothing secret here") == "nothing secret here"

    def test_no_prompt_leaks_a_real_credential_from_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The real risk: a prompt interpolating settings or a URL with a key.

        A sentinel is planted first. The previous version built its list of
        secrets from ``os.environ`` and asserted none of them appeared in any
        prompt - on CI, and on any machine without credentials exported, that
        list was empty, the inner loop never ran, and the test passed without
        checking anything. It was the project's headline anti-leak test and it
        verified nothing in normal use.
        """
        sentinel = "sentinel-" + "k" * 24
        monkeypatch.setenv("OPENAI_API_KEY", sentinel)
        monkeypatch.setenv("ACME_DEPLOY_TOKEN", sentinel)
        for name in ALL:
            rendered = "\n".join(m.content for m in get(name).build(_ctx()))
            assert sentinel not in rendered, f"the {name} prompt leaked a credential"

    def test_the_context_cannot_carry_arbitrary_attributes(self) -> None:
        """A closed model is what stops ``ctx.settings`` becoming a leak."""
        with pytest.raises(ValueError):
            PromptContext(company="Acme", settings=object())  # type: ignore[call-arg]


class TestReportSpecifics:
    def test_the_final_report_carries_the_reporting_rules(self) -> None:
        rendered = "\n".join(m.content for m in build("final_report", _ctx()))
        assert "Show the gaps" in rendered

    def test_other_prompts_do_not_carry_the_reporting_rules(self) -> None:
        """Otherwise every prompt pays for a rule only the report needs."""
        rendered = "\n".join(m.content for m in build("seo_analysis", _ctx()))
        assert "Show the gaps" not in rendered

    def test_the_reporting_rules_precede_the_data_in_the_system_block(self) -> None:
        system = build("final_report", _ctx())[0].content
        assert system.index("REPORTING RULES") < system.index("RESEARCH GOAL")


class TestInstructionText:
    def test_the_common_instruction_is_not_empty(self) -> None:
        assert len(COMMON_INSTRUCTION) > 200
        assert EVIDENCE_HEADER in COMMON_INSTRUCTION
