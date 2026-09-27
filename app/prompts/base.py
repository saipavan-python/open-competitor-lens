"""The prompt layer: one place where every instruction to a model is written.

Two rules govern everything in this package.

**The evidence is the ground truth, and it arrives in the prompt.** A model asked
to describe a company's pricing will produce a plausible price whether or not
anyone supplied one. The only reliable defence is to put the retrieved text in
front of the model and make the absence of a fact cheap to report. So every
prompt renders an ``AVAILABLE EVIDENCE`` block, verbatim, and the shared
instruction below tells the model that anything not in that block does not exist
as far as this answer is concerned.

**A prompt is not the enforcement mechanism.** Everything the shared instruction
*asks* for is also *checked* in code downstream: :mod:`app.services.evidence_service`
re-verifies each quote against stored source text and refuses a citation naming
a source the run never fetched. The instruction exists to get a useful answer,
not to keep the system honest. That split is deliberate — a system whose
correctness depends on a model obeying prose has no correctness, only a
suggestion.

The shared instruction is therefore written to match the rules the code already
enforces, not to invent stricter ones the pipeline would then reject. When a rule
below and a validator in :mod:`app.schemas.evidence` disagree, the validator is
right and this text is a bug.
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.core.logging import get_logger
from app.schemas.evidence import Evidence, Passage, Source
from app.schemas.provider import ChatMessage

log = get_logger(__name__)

#: Header introducing the retrieved evidence. The prompt tests assert on this
#: literal, because a silently renamed section means the model stops being told
#: what it is allowed to rely on and nothing fails.
EVIDENCE_HEADER = "AVAILABLE EVIDENCE"

#: Header introducing the required output shape.
SCHEMA_HEADER = "OUTPUT SCHEMA"

#: Environment variables whose values must never reach a prompt.
_SECRET_ENV_MARKERS = ("API_KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
_MIN_SECRET_LENGTH = 8


class PromptLeakError(ValueError):
    """A rendered prompt contained a value that looks like a credential.

    Raised at build time rather than at send time because a prompt is built once
    and sent many times; catching it here means the leak never reaches a
    provider, a log line, or a trace.
    """


class PromptContext(BaseModel):
    """Everything a prompt is allowed to know about the run.

    Deliberately a closed model (``extra="forbid"``): a prompt builder that
    silently accepted an arbitrary attribute would be one refactor away from
    interpolating a settings object into a prompt, which is the credential leak
    :func:`assert_no_secrets` exists to catch.
    """

    model_config = ConfigDict(extra="forbid")

    company: str
    research_goal: str = ""
    competitors: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    #: Raw retrieved text, for the analyst prompts that run *before* any claim
    #: has been verified. See :class:`~app.schemas.evidence.Passage` for why
    #: these are not evidence and must not be cited as such.
    passages: list[Passage] = Field(default_factory=list)
    facts: dict[str, Any] = Field(default_factory=dict)
    instructions: str = ""

    def source(self, source_id: str) -> Source | None:
        for candidate in self.sources:
            if candidate.id == source_id:
                return candidate
        return None

    def evidence_block(self) -> str:
        """Render the evidence, verbatim, for the model to read.

        Quotes are reproduced exactly as stored rather than reformatted: the
        evidence gate verifies quotes against this text, so a rendering that
        tidied whitespace here would produce prompts whose quotes the gate then
        rejects. Each claim carries its index so the model can cite ``[E3]``
        and have that index mean one specific thing.

        Retrieved passages render instead, and are indexed ``[P1]`` rather than
        ``[E1]``, when there is no verified evidence — which is the state every
        analyst runs in, since the evidence for a category is what the analyst
        produces. The two index spaces are kept distinct so a model cannot cite
        a passage as though it were a verified finding.
        """
        if self.passages and not self.evidence:
            return self._passage_block()
        if not self.evidence:
            return (
                f"{EVIDENCE_HEADER}\n"
                "(none retrieved yet — every field you cannot support from "
                "evidence must be reported as unknown)"
            )
        lines = [EVIDENCE_HEADER]
        for position, item in enumerate(self.evidence, start=1):
            source = self.source(item.source_id)
            lines.append(
                f"[E{position}] source_id={item.source_id} "
                f"url={item.source_url} "
                f"source_type={item.source_type.value if item.source_type else 'unknown'}"
            )
            if source is not None and source.retrieved_at:
                lines.append(f"    retrieved_at={source.retrieved_at.isoformat()}")
            lines.append(f"    claim: {item.claim}")
            if item.evidence_text:
                lines.append(f'    quote: "{item.evidence_text}"')
            if item.published_at:
                lines.append(f"    published_at={item.published_at.isoformat()}")
        return "\n".join(lines)

    def _passage_block(self) -> str:
        """Retrieved text, labelled so it is read as input rather than as a finding."""
        lines = [
            EVIDENCE_HEADER,
            "RETRIEVED PASSAGES — these are pages this run fetched and has not "
            "yet verified. Cite them by their source_id, quote them exactly, and "
            "report as unknown anything they do not state.",
        ]
        for position, passage in enumerate(self.passages, start=1):
            lines.append(passage.label(position))
            if passage.title:
                lines.append(f"    title: {passage.title}")
            if passage.heading:
                lines.append(f"    section: {passage.heading}")
            lines.append(f'    text: "{passage.text}"')
        return "\n".join(lines)


#: The instruction every prompt carries. See the module docstring: each rule
#: here has a validator behind it, and the validator wins any disagreement.
COMMON_INSTRUCTION = f"""\
EVIDENCE RULES — these are absolute, and they are checked after you answer:

1. Use only the sources in the {EVIDENCE_HEADER} block. A fact you cannot point
   at there does not exist for this answer, even if you are certain of it.
2. Never invent a source id, URL, figure, date, feature name or customer. If it
   is not in the block, you did not find it.
3. Quote verbatim. A quotation must appear character-for-character in the source
   text you were given; paraphrasing inside quotation marks is a fabricated
   quote.
4. Report anything unestablished as unknown, using the unknown sentinel value
   the output schema provides. "Unknown" is a valid, useful answer. A confident
   guess is not.
5. An inference is your reasoning, not something a source says. Label it as an
   inference and never give one high confidence.
6. A search snippet or an unclassified source cannot support high confidence on
   its own, however confident you feel.
7. If the evidence is insufficient for a field, say so. Do not fill the gap from
   general knowledge, and do not smooth over a conflict between two sources —
   report both and the disagreement between them.
"""


def assert_no_secrets(text: str) -> str:
    """Refuse to return a prompt containing a credential from the environment.

    Checked on the rendered text rather than trusted to review, because the
    realistic failure is not a deliberate leak: it is a future prompt that
    interpolates ``settings.model_dump()`` or a URL with an embedded key, which
    looks like a normal field in review and would otherwise ship silently.
    """
    for name, value in os.environ.items():
        if not value or len(value) < _MIN_SECRET_LENGTH:
            continue
        if not any(marker in name.upper() for marker in _SECRET_ENV_MARKERS):
            continue
        if value in text:
            raise PromptLeakError(
                f"rendered prompt contains the value of {name}; prompts must "
                f"never carry credentials"
            )
    return text


def schema_block(schema: type[BaseModel], *, example: str = "") -> str:
    """Render a required output shape as a labelled section."""
    body = f"{SCHEMA_HEADER}\n{schema.__name__}\n"
    if example:
        body += f"\nExample of the required shape (values are illustrative):\n{example}\n"
    return body


def build_messages(
    *,
    purpose: str,
    context: PromptContext,
    schema: type[BaseModel],
    sections: list[str],
    schema_example: str = "",
    extra_system: str = "",
) -> list[ChatMessage]:
    """Assemble the two messages every prompt in this package is made of.

    A system message carrying the role, the evidence rules and the per-task
    instructions, and a user message carrying the run's actual data. The split
    matters for auditing: the system block is identical across every call of a
    given prompt, so a reviewer can read what the model was told once and be
    confident it was told that for all of them.

    ``extra_system`` inserts additional rules in the middle of the system block,
    after the task purpose and before the run-specific goal. It exists so a
    prompt with extra rules can place them deliberately; the alternative,
    concatenating onto the finished string, both appends after the goal and
    invites repeating text that is already there.
    """
    system_parts = [
        f"You are analysing {context.company} as part of an evidence-backed "
        f"competitor analysis.",
        "",
        COMMON_INSTRUCTION,
        "",
        f"YOUR TASK: {purpose}",
    ]
    if extra_system:
        system_parts += ["", extra_system]
    if context.research_goal:
        system_parts += ["", f"RESEARCH GOAL FOR THIS RUN: {context.research_goal}"]
    if sections:
        system_parts += ["", "COVER, IN THIS ORDER:"] + [
            f"  {index}. {section}" for index, section in enumerate(sections, start=1)
        ]
    if context.instructions:
        system_parts += ["", context.instructions]

    user_parts = [
        f"COMPANY: {context.company}",
    ]
    if context.competitors:
        user_parts.append(f"COMPETITORS UNDER CONSIDERATION: {', '.join(context.competitors)}")
    user_parts += [
        "",
        context.evidence_block(),
        "",
        schema_block(schema, example=schema_example),
    ]
    if context.facts:
        user_parts += ["", "CONTEXT PROVIDED BY EARLIER STEPS:"] + [
            f"  {key}: {value}" for key, value in sorted(context.facts.items())
        ]

    messages = [
        ChatMessage(role="system", content=assert_no_secrets("\n".join(system_parts))),
        ChatMessage(role="user", content=assert_no_secrets("\n".join(user_parts))),
    ]
    log.debug("prompt built", extra={"purpose": purpose, "messages": len(messages)})
    return messages
