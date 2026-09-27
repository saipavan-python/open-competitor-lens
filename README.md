# open-competitor-lens

**Evidence-backed competitive analysis: every claim in a report traced to a
retrieved source.**

> **Status: pre-alpha.** The analysis library is complete and tested. The
> orchestrator, HTTP API, CLI and Docker image are **not built yet** — see
> [What is not built yet](#what-is-not-built-yet). There is nothing to run.
> This README is deliberately blunt about that, because a repository that says
> "competitor analysis agent" and then cannot analyse a competitor is worse than
> one that never claimed to.

---

## The idea

Most "AI competitive analysis" tools produce fluent prose with citations
attached. The citations are decorative: the model states a price, a market
share, or a growth rate, and the page linked underneath says nothing of the
sort. The number is still there, and it still reads as authoritative.

This project treats that as the primary failure mode to design against. The
central rule is:

> A claim may only appear in a report if a retrieved page contains it, and the
> quote attached to it appears verbatim in that page's stored text.

Everything else — the evidence records, the fact checker, the quality gate, the
`NOT_PUBLIC` convention — exists to make that one property hold, or to make the
places where it fails visible instead of silent.

## What exists today

A library of the parts, each independently tested, with no wiring between them yet.

| Area | Modules | What it does |
|---|---|---|
| **Retrieval** | `providers/web`, `providers/search` | HTTP and Firecrawl fetchers, Tavily search, SSRF guard, redirects, dedupe, content cache, change detection |
| **Extraction** | `services/extraction_service` | Schema-constrained LLM extraction with quote location against the retrieved text |
| **Evidence** | `schemas/evidence`, `agents/evidence_agent` | Evidence records that reject unsupported confidence combinations at construction |
| **Discovery** | `agents/company_agent`, `agents/competitor_agent` | Official-domain resolution; competitor proposal followed by validation |
| **Analysis** | `agents/analysis_base` + 9 analysts | Per-category extraction, each verifying its own output against permitted sources |
| **Assembly** | `agents/comparison`, `agents/opportunity`, `agents/report` | Comparison matrix, opportunities, report bundle and renderers — **zero model calls**, by design |
| **Verification** | `agents/fact_checker`, `services/quality_service` | Mechanical rules: numbers, superlatives, dangling citations, refuted claims, bounded repair |
| **Infrastructure** | `core`, `utils`, `models`, `repositories` | Config, redacting JSON logging, typed exceptions, SQLAlchemy models, SSRF guard, text utilities |

### Two decisions worth calling out

**Assembly stages make no model calls.** `ComparisonAgent`, `OpportunityAgent`,
`EvidenceAgent`, `ReportAgent` and `FactCheckerAgent` are all deterministic. A
quality stage that calls a model to judge the model's own output is a model
agreeing with itself, so the final gate compares text against evidence it can
point at.

**A failed retrieval is a finding, not a crash.** A competitor's pricing page
being unreachable is something the report must be able to *say*, so every
provider returns a `FetchResult` carrying a `retrieval_status` instead of raising.
The report can then distinguish "the page says nothing" from "we could not read
the page".

## What is not built yet

These are planned work that has not been started. Nothing in the list above is
wired to any of them:

- **Orchestrator** — no LangGraph workflow exists. `langgraph` is a declared
  dependency with no graph in the tree.
- **HTTP API** — no `app/main.py`, no routes, no app entry point.
- **CLI** — no `run.py`, so the plan's `python run.py analyze --company ...` does
  not work.
- **GraphRAG** — `providers/offline.py` has an in-memory `OfflineGraphStore`; the
  real `providers/graph/` and `graph_rag/` modules do not exist.
- **Docker** — no `Dockerfile` or `docker-compose.yml`. **The Docker build is
  unverified**: Docker is not installed on the machine this was written on, so no
  claim is made that an image builds.
- **Live provider verification** — every adapter is tested against injected fake
  SDK clients (request shape, response validation, error mapping). None has been
  exercised against a real endpoint, because no API keys were available. Request
  correctness against the real APIs is therefore *assumed*, not demonstrated.

## Installation

```bash
git clone https://github.com/saipavan-python/open-competitor-lens.git
cd open-competitor-lens
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

## Running

You cannot run an analysis yet — there is no entry point. What you can do:

```bash
pytest -q                  # 1124 tests
ruff check app tests       # lint
python -c "import app.agents.competitor_agent"   # imports resolve
```

The test suite is the only executable documentation of the behaviour so far, and
it is written to be read: each regression test names the defect it exists to
catch.

## Configuration

Copy `.env.example` to `.env` and fill in what you need. Everything is optional
at import time; a missing key disables that provider rather than failing to start,
and a disabled provider degrades the run instead of ending it.

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | OpenAI chat + embeddings |
| `GEMINI_API_KEY` | Gemini chat |
| `TAVILY_API_KEY` | Web search |
| `FIRECRAWL_API_KEY` | Rendered-page retrieval |
| `OPENAI_BASE_URL` | Any OpenAI-compatible gateway |
| `OPENAI_MODEL` / `GEMINI_MODEL` | Vendor model; seeds every unset model role |
| `DATABASE_URL` | Defaults to local SQLite |
| `LOG_JSON` | Structured JSON logs instead of text |
| `GRAPH_RAG_ENABLED` | Off by default; off means the graph store is never touched |

Model roles (`FAST_MODEL`, `REASONING_MODEL`, `FINAL_MODEL`) are seeded from the
configured vendor model and may be set individually, so per-role vendor mixing
works.

## Architecture

Imports flow one way, and this is enforced by a 133-case AST test rather than by
convention — `tests/unit/test_layering.py` parses every module and fails the build
on a backwards import:

```
core → schemas → utils → models/rag → repositories → providers → services
     → agents/graphs/prompts → api/cli
```

The rule it enforces is the interesting part: a **service** that holds an injected
LLM must not import from the provider package it was handed. That is why
`parse_structured` lives in `utils/` rather than in a provider's private module —
the extraction service needs it, and reaching into `providers/llm/_shared.py` to
get it would invert the dependency.

## Security

A crawler that fetches URLs chosen by a language model is an SSRF engine unless
something stops it, so that is treated as a first-class concern:

- Per-hop URL validation on every redirect (the guard is on the request, not the
  entry point).
- The **browser** provider installs the guard as a route handler, covering
  redirects and in-page fetches, with a second check on the landing URL before
  the body is read.
- Blocked: loopback, private, link-local (including `169.254.169.254`),
  carrier-grade NAT, multicast, reserved, IPv4-mapped, decimal/hex IP
  obfuscation, `user:pass@host`, non-HTTP schemes, IDN homographs.
- Streaming byte caps on every fetcher, including rendered pages.
- Credential and absolute-path redaction on both log formatters, and on the API
  error body.
- Prompt rendering is refused if it contains any credential visible in the
  environment *or* in loaded settings.

## Honest notes

A whole-branch review before the first public push found eleven further defects,
all fixed, and two are worth recording because they describe how this kind of
system fails:

- The numeric anti-hallucination gate was defeated by a single `/`. `$50/mo`
  tokenised to nothing, so an invented price passed the checker. 1015 passing
  tests could not see it. The test suite now asserts the cases explicitly, and
  11 of them fail against the old code.
- The offline providers had never successfully run. They existed to make the
  pipeline testable without credentials, and both were broken — so the pipeline
  they existed for had never actually executed. The only tests for them were
  `isinstance` assertions, which pass when a constructor works and say nothing
  about whether the provider does.

Both are recorded in the commit history, because the second one is the general
warning: a test suite that has never been made to fail is not evidence.

## Licence

Not yet chosen. Add one before treating this as a released project.
