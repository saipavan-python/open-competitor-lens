"""Prompt text that is shared but not universal.

:mod:`app.prompts.base` holds the evidence rules that *every* prompt carries,
because every prompt in this system is subject to the same fabrication risks.
This module holds the smaller set of instructions that apply to one or two
prompts, kept apart from the builder machinery so that the text a model reads
can be reviewed without reading the code that assembles it.
"""

from __future__ import annotations

#: Additional rules for the one prompt whose output is read by a person.
#:
#: The evidence rules in :mod:`app.prompts.base` stop a report from containing
#: something untrue. These stop it from containing something true that implies
#: more than the analysis established — which no validator downstream can catch,
#: because the report is prose and the fabrications are in what it leaves out.
REPORTING_INSTRUCTION = """\
REPORTING RULES:

1. Write from the evidence in front of you. If it does not support a point, do
   not make the point.
2. Show the gaps. An unknown cell stays unknown; a section you could not
   research is named as unresearched. A report that is silently incomplete
   invites the reader to mistake your silence for their finding.
3. Report disagreements between sources instead of choosing a side. If the
   official page and a news report give different prices, both appear, with the
   conflict stated.
4. Distinguish what a source says from what you conclude. Mark the latter as
   inference. A reader who cannot tell which sentences are sourced will treat
   all of them as sourced.
5. State the limits of the analysis in the report itself — what was not
   retrieved and what remains unverified. This is not hedging; it is what lets a
   reader decide how far to trust the rest.
6. Do not recommend a course of action that depends on a fact you could not
   verify. If the recommendation needs it, say what would have to be checked
   first.
"""
