# AI Competitor Analysis Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a production-grade competitive-intelligence platform that takes a company name + website and produces an evidence-backed competitor analysis report, with every factual claim traceable to a retrieved public source.

**Architecture:** One primary orchestrator — LangGraph `StateGraph` over a single typed `ResearchState`. Nodes are narrow, each owning a disjoint slice of state. All external I/O goes through `Protocol` providers (LLM / search / web / browser / embeddings / vector / graph) so any backend can be swapped without touching orchestration. Deterministic "offline" provider implementations let the entire graph run end-to-end with zero API credentials.

**Tech Stack:** Python 3.12, FastAPI, LangGraph, Pydantic v2 + pydantic-settings, httpx, BeautifulSoup4/lxml, trafilatura, SQLAlchemy 2.0 (async), ChromaDB, pytest/pytest-asyncio, ruff.

**Spec:** The user's "PRODUCTION MASTER BUILD PROMPT v2" (120 sections) is the binding authority. Where this plan and the spec conflict, the spec wins and the controller records a ruling in the ledger.

---

## Global Constraints

These bind every task. They are not repeated in task text.

- **Python 3.12** (`.venv` in repo root). Package versions resolved at install time, never pinned from the prompt.
- **One orchestrator.** LangGraph only. CrewAI, AutoGPT, Deep Agents are *design references*, not dependencies. Do not `pip install` them.
- **No hardcoded secrets, model names, URLs, timeouts, or concurrency limits.** All read from `app/core/config.py` via `pydantic-settings`. Tests must not need API keys.
- **Every outbound URL passes `app/utils/urls.py:validate_url` before any request.** SSRF guard is not optional and is tested directly.
- **Anti-hallucination is a data-structure property, not a prompt request.** Claims that lack a retrievable `source_id` cannot become `Evidence` with confidence `high`. Unknown is a first-class value (`UNKNOWN_SENTINELS`), never a fabricated default.
- **TDD: no production code without a failing test first.** Run the test, see it fail for the expected reason, then implement.
- **Verification before completion claims.** No "works"/"passes"/"complete" without a fresh command + output in the same message.
- **No `pass` / `TODO` / "implement later" in core logic.** An optional adapter that is unavailable must raise a typed `ProviderNotConfiguredError`, never silently no-op.
- `ruff check` and `ruff format --check` must be clean at the end of every task that adds Python files.
- Async-first: `async def` for anything touching I/O. `asyncio.Semaphore` for all concurrency limits.

## Review Focus

The spec implies these failure modes; each gets a test in the task named beside it.

1. **Config pollution between tests** — a test that sets `OPENAI_API_KEY` must not leak into the next test, and `get_settings()` must not return a stale cached object. → Task 1
2. **SSRF via redirect** — a URL that passes validation then 302s to `169.254.169.254` (cloud metadata). The guard must re-validate each hop. → Task 5
3. **A competitor that is a substring of the target** (target "Kansbe", candidate "Kansbe Digital") must not be validated as a competitor of itself. → Task 10
4. **Evidence laundering** — an LLM returning a claim with a plausible-looking but never-retrieved `source_id` must be rejected, not persisted. → Task 7
5. **Deduplication drops distinct content** — two competitors publishing byte-identical boilerplate (same shared footer) must not collapse into one source. Hashing must include the canonical URL. → Task 6

---

## File Structure

Locked before Task 1. Every path below is created by exactly one task.

```
app/
  main.py                     FastAPI app factory, router mount, lifespan
  api/dependencies.py         DI container wiring providers + repositories
  api/routes/{analysis,companies,competitors,reports,sources,health}.py
  core/{config,logging,security,exceptions,constants,telemetry}.py
  graph/{state,workflow,routing}.py
  graph/nodes/*.py            one file per node, listed in Task 12
  agents/*.py                 one file per agent, listed in Tasks 9-11
  providers/
    base.py                   all Protocols + ProviderResult envelope
    registry.py               name -> factory, with fallback chains
    llm/{openai,gemini,offline}.py
    search/{tavily,firecrawl,offline}.py
    web/{http,firecrawl,browser,offline}.py
    browser/{browser_use,unavailable}.py
    embeddings/{openai,hash}.py
    storage/{chroma,memory}.py
    graph/{memory}.py
  services/*.py               listed in Tasks 5-11
  schemas/*.py                listed in Task 2
  models/*.py                 SQLAlchemy, listed in Task 3
  repositories/*.py           listed in Task 3
  prompts/*.py                listed in Task 8
  rag/{chunking,indexing,retrieval,metadata,reranking}.py
  graph_rag/{entities,relationships,extraction,queries}.py
  utils/{urls,hashing,normalization,text,dates,retry,concurrency,json}.py
tests/{unit,integration,graph,api,providers,fixtures}/
scripts/{init_db,seed_demo,health_check}.py
```

**Layering rule (enforced by a test):** `utils` imports nothing from `app`; `schemas` imports only `utils`; `providers` imports `schemas`+`utils`; `services` import `providers`+`schemas`; `agents` import `services`; `graph` imports `agents`. A test in Task 4 walks the AST of every module and asserts no lower layer imports a higher one.

---

## Task 1: Scaffolding, config, logging, exceptions

**Files:** `pyproject.toml`, `.gitignore`, `.env.example`, `app/__init__.py`, `app/core/{config,logging,exceptions,constants}.py`, `tests/unit/test_config.py`, `tests/conftest.py`

**Interfaces:** produces `app.core.config.Settings` (frozen-ish singleton via `get_settings()`), `app.core.config.reset_settings_cache()`, `app.core.exceptions.{AppError,ConfigurationError,ProviderNotConfiguredError,ProviderError,RateLimitError,ValidationError,SSRFError,BudgetExceededError}`, `configure_logging()`, `get_logger()`.

- [ ] Write failing tests: defaults load with no `.env`; `MAX_PAGES_PER_COMPETITOR=40` default; unknown env prefix ignored; `get_settings()` returns same instance; `reset_settings_cache()` then env change returns new value; every exception subclasses `AppError` and carries `retryable: bool`.
- [ ] Run `pytest tests/unit/test_config.py -v` → FAIL (module not found).
- [ ] Implement. `Settings(BaseSettings)` with `env_prefix=""`, nested `BudgetSettings`/`TimeoutSettings`/`ProviderSettings` grouped but flattened env names. `AppError.__init__(message, *, retryable=False, provider=None, step=None)`.
- [ ] Run tests → PASS. `ruff check app tests`.

## Task 2: Domain schemas (Pydantic v2)

**Files:** `app/schemas/*.py` (16 files), `tests/unit/test_schemas.py`

**Interfaces:** produces every model named in spec §14, §15, §36, §38, §62, §64. Key ones: `Source`, `Document`, `Evidence`, `Finding`, `Insight`, `CompetitorCandidate`, `ValidatedCompetitor`, `ResearchQuestion`, `ResearchPlan`, `PricingPlan`, `ProductOffering`, `SocialProfile`, `ReviewTheme`, `TechnologySignal`, `Opportunity`, `ComparisonMatrix`, `ReportBundle`, `StartAnalysisRequest`, `RunStatus`, `AgentResult` envelope.

- [ ] Failing tests: `Evidence(confidence="high")` rejected without `source_id`; `claim_type` literal rejects `"guessed"`; `PricingPlan` keeps `price` `None` and `price_display` `"Contact sales"`; `Evidence` rejects a `source_url` whose host is not in the run's known source set (passed as `allowed_hosts`).
- [ ] Run → FAIL. Implement. `model_config = ConfigDict(extra="forbid")` on all domain models.
- [ ] Run → PASS.

## Task 3: Persistence — models, session, repositories

**Files:** `app/models/*.py` (base + 8), `app/core/database.py`, `app/repositories/*.py` (6), `tests/integration/test_repositories.py`

**Interfaces:** `Database` (async engine + sessionmaker, `sqlite+aiosqlite` default, `postgresql+asyncpg` if `DATABASE_URL` says so), `Base`, `CompanyRepository`, `CompetitorRepository`, `ResearchRunRepository`, `SourceRepository`, `EvidenceRepository`, `ReportRepository`. Every repo: `async def create(...) -> Model`, `async def get(id) -> Model | None`, plus run-scoped list methods. All PKs `UUID` (`sa.Uuid`), all timestamps `DateTime(timezone=True)` UTC.

- [ ] Failing test on a temp SQLite file: create a `ResearchRun`, add a `Source`, read both back; `content_hash` has a unique index per run; `ResearchRun.status` defaults to `"queued"`.
- [ ] Run → FAIL. Implement. `async_sessionmaker`, `expire_on_commit=False`.
- [ ] Run → PASS.

## Task 4: Layering test + provider protocols

**Files:** `app/providers/base.py`, `app/providers/registry.py`, `tests/unit/test_layering.py`, `tests/providers/test_protocols.py`

**Interfaces:** Protocols `LLMProvider` (`async generate(prompt, *, system=None, schema=None, model_role="default", temperature=None) -> LLMResponse`), `SearchProvider` (`async search(query, *, limit) -> list[SearchHit]`), `WebProvider` (`async fetch(url) -> FetchResult`, `async extract(url, schema) -> dict`), `BrowserProvider` (`async open/navigate/extract/close`), `EmbeddingProvider` (`async embed_documents/embed_query`), `VectorStore` (`async upsert/query/delete_collection/count`), `GraphStore` (`async upsert_entities/upsert_relationships/query`). Plus `ProviderHealth`/circuit-breaker mixin.

- [ ] Failing test: AST-walks every `app/**/*.py` and asserts no module in layer *N* imports layer *N+1*.
- [ ] Failing test: `ProviderRegistry.resolve("search", "tavily")` raises `ProviderNotConfiguredError` naming the missing env var, and `resolve("search","tavily", fallbacks=["offline"])` returns the offline provider.
- [ ] Run → FAIL. Implement. `ProviderResult` carries `provider`, `model`, `duration_ms`, `cached`.
- [ ] Run → PASS.

## Task 5: Utils + SSRF guard

**Files:** `app/utils/{urls,hashing,normalization,text,dates,retry,concurrency,json}.py`, `tests/unit/test_urls.py`, `tests/unit/test_utils.py`

**Interfaces:**
- `canonicalize_url(raw: str) -> str | None` — lowercases scheme+host, drops fragment, sorts query, strips `utm_*`/`fbclid`/`gclid`, strips default ports, normalizes trailing slash.
- `validate_url(raw: str) -> ValidatedUrl` — raises `SSRFError` on: non-http(s) scheme, missing host, userinfo in URL, host resolving to loopback/private/link-local/unspecified/multicast/reserved IPv4 or IPv6, cloud-metadata IPs (`169.254.169.254`, `fd00:ec2::254`), disallowed port, IDN that fails to normalize. Returns parsed components.
- `validate_redirect_chain(urls: Iterable[str]) -> None` — re-validates *each* hop (Review Focus #2).
- `sha256_text`, `sha256_bytes`, `stable_id(prefix, *parts) -> str`
- `normalize_company_name`, `company_identity_key` (Review Focus #3), `normalize_space`, `slugify`
- `with_retry(fn, *, attempts, base_delay, retry_on)` exponential backoff + jitter
- `gather_limited(coros, limit)` semaphore-bounded gather preserving order
- `safe_json_loads(text)` — strips markdown fences, extracts first balanced JSON object, returns `None` on failure rather than raising

- [ ] Failing tests for every blocklisted range above, plus: `http://127.0.0.1/x` blocked, `http://[::1]/x` blocked, `http://10.0.0.1/x` blocked, `http://169.254.169.254/latest/meta-data` blocked, `http://localhost` blocked, `ftp://example.com` blocked, `http://user:pw@example.com` blocked, `http://example.com:22` blocked, `https://Example.COM:443/a?b=2&utm_source=x#f` → `https://example.com/a?b=2`. Test `validate_redirect_chain` with a chain whose 2nd element is `169.254.169.254` → raises.
- [ ] Run → FAIL. Implement.
- [ ] Run → PASS.

## Task 6: Web pipeline — fetch, extract, dedupe

**Files:** `app/providers/web/{http,firecrawl,browser,offline}.py`, `app/providers/browser/{browser_use,unavailable}.py`, `app/services/{crawling,extraction,cache}_service.py`, `tests/unit/test_dedupe.py`, `tests/providers/test_web_offline.py`

**Interfaces:** `HttpWebProvider(safe_fetcher, timeout, max_bytes, user_agent)`, `FetchResult(url, final_url, status, content_type, text, html, title, retrieved_at, retrieval_status, content_hash)`, `classify_page(url, html, text) -> PageType` (the 18 types in spec §90), `Deduplicator` (canonical-URL key AND `sha256(canonical_url + content_hash)` key — Review Focus #5), `should_escalate_to_browser(fetch_result) -> bool`, `CrawlingService.crawl_site(root_url, budget) -> list[FetchResult]` honouring `MAX_PAGES_PER_COMPETITOR` and the priority path list in spec §20.

- [ ] Failing test: same content under two different URLs of the same company produces **two** sources; the same URL fetched twice produces one; the byte-identical boilerplate of two different competitors produces two.
- [ ] Failing test: `classify_page("https://x.com/pricing", ...) == PageType.PRICING`; `.../blog/post-1` → `BLOG`; `.../case-studies/acme` → `CASE_STUDY`.
- [ ] Failing test: `should_escalate_to_browser(FetchResult(status=200, text="", html="<html><body><div id=root></div>"))` is True; a 20 000-char text page is False.
- [ ] Run → FAIL. Implement.
- [ ] Run → PASS.

## Task 7: Evidence system

**Files:** `app/schemas/evidence.py` (extended), `app/services/evidence_service.py`, `app/services/citation_service.py`, `tests/unit/test_evidence.py`

**Interfaces:** `EvidenceService.attach(claim, *, source, quote, claim_type, confidence)` — verifies `source.retrieval_status == "ok"`, that `quote` is a real substring of `source.text` (or `evidence_text` is empty), and that `source_id` refers to a `Source` the run actually fetched. Raises `EvidenceNotSupportedError` otherwise (Review Focus #4). `CitationService.build_index(sources, evidence) -> dict[source_id, list[claim_ids]]` and `resolve(citation_index) -> list[Source]` for the API.

- [ ] Failing tests: claim citing a `source_id` never fetched → raises; quote not present in source text → raises; `claim_type="inferred"` may not carry confidence `high`; a conflict between two sources on the same `claim_key` is preserved as two evidences and flagged, not silently merged.
- [ ] Run → FAIL. Implement.
- [ ] Run → PASS.

## Task 8: Prompts + cost-aware model router

**Files:** `app/prompts/*.py` (17 modules), `app/services/model_router.py`, `tests/unit/test_model_router.py`, `tests/unit/test_prompts.py`

**Interfaces:** `ModelRouter.select(task_type, *, complexity="low") -> str` reading `ModelSettings` profiles `fast`/`reasoning`/`final`; never hardcodes a model id. Every prompt module exposes `build(ctx) -> list[ChatMessage]` and its output Pydantic schema; every prompt carries the spec §78 sections, and every prompt's user block includes `AVAILABLE_EVIDENCE` verbatim plus the spec §78 common anti-hallucination instruction.

- [ ] Failing test: `select("query_generation")` != `select("final_report")`; and both are values present in `Settings`.
- [ ] Failing test: every registered prompt's `build()` output contains the literal `AVAILABLE EVIDENCE` and `OUTPUT SCHEMA` headers, and the rendered text contains no `API_KEY` value from env.
- [ ] Run → FAIL. Implement.
- [ ] Run → PASS.

## Task 9: Agents — planner, company discovery, competitor discovery + validation

**Files:** `app/agents/{planner,company_agent,competitor_agent}.py`, `app/agents/base.py`, `app/services/{search,research}_service.py`, `tests/unit/test_competitor_validation.py`, `tests/unit/test_planner.py`

**Interfaces:** `Agent` ABC with `name`, `goal`, `input_schema`, `output_schema`, `timeout`, `max_iterations`, and `async run(ctx) -> AgentResult`. `PlannerAgent.run(ctx) -> ResearchPlan` emits ≤ the questions in spec §9, tailored to `research_goal`, and marks each `required_sources` tier per spec §13. `CompanyAgent` uses 6 queries, ranks official-domain candidates by multiple signals (spec §10) and never assumes rank 1 is official. `CompetitorAgent` builds a pool of `2–4x` `competitor_count` then `Validator` scores each on the 7 signals of spec §12 and returns only those with ≥1 primary-source-backed signal.

- [ ] Failing test: pool of 10 candidates with `competitor_count=5` yields exactly 5 validated.
- [ ] Failing test: candidate whose `company_identity_key` equals the target's is dropped with reason `self`.
- [ ] Failing test: a candidate supported only by a search snippet (source_type `search_snippet`) is NOT validated.
- [ ] Failing test: planner with `research_goal="compare pricing only"` does not emit a `seo` question.
- [ ] Run → FAIL. Implement.
- [ ] Run → PASS.

## Task 10: Specialized analysts

**Files:** `app/agents/{website,product,pricing,marketing,seo,social,review,technology,market}_agent.py`, `app/agents/analysis_base.py`, `app/rag/{chunking,indexing,retrieval,metadata,reranking}.py`, `app/services/rag_service.py`, `tests/unit/test_rag.py`, `tests/unit/test_analyst_contract.py`

**Interfaces:** `CategoryAnalyst(ABC)` declaring `category`, `retrieval_queries`, `output_schema`. Each of the 9 subclasses supplies only its queries + schema + prompt; the base does retrieve→filter→compress→analyze and returns `AgentResult` whose payload cites only `source_id`s from the retrieved set. `RagService.index(documents) -> index_id`, `RagService.retrieve(query, *, filters, k) -> list[RetrievedChunk]` with metadata prefilter applied *before* ranking.

- [ ] Failing test: `retrieve(filters={"competitor":"A","category":"pricing"})` never returns a chunk whose `competitor=="B"`.
- [ ] Failing test: `chunk_text` never splits inside a heading; every chunk keeps its source metadata keys.
- [ ] Failing test: an analyst whose LLM returns a `source_id` outside its retrieved set is reported as `partial` with a `unsupported_citation` warning — the base strips it rather than persisting.
- [ ] Failing test: for all 9 analysts, the payload validator rejects `pricing=None` silently filled with a number.
- [ ] Run → FAIL. Implement.
- [ ] Run → PASS.

## Task 11: Comparison, opportunity, fact-check, quality, report

**Files:** `app/agents/{comparison,opportunity,evidence_agent,fact_checker,report_agent,supervisor}.py`, `app/services/{comparison,quality,report}_service.py`, `app/services/change_detection.py`, `tests/unit/test_comparison.py`, `tests/unit/test_quality_control.py`, `tests/unit/test_change_detection.py`

**Interfaces:** `ComparisonService.build(companies, analyses) -> ComparisonMatrix` filling absent cells with the literal `"Not publicly available"`, never a guess. `OpportunityAgent` requires ≥1 evidence id per opportunity. `FactCheckerAgent.verify(claims, evidence_index) -> list[ClaimVerdict]` with `verdict ∈ {supported, overstatement, unsupported, contradictory, stale}` and a downgraded-confidence side effect. `QualityService.check(report) -> QualityReport` and `QualityService.repair(report, rounds)` honouring `MAX_REPORT_REPAIR_ROUNDS=2` with a hard stop. `ChangeDetector.diff(previous, current) -> list[ChangeEvent]`.

- [ ] Failing test: comparison of a company with no pricing yields `"Not publicly available"`, not `None` and not `0`.
- [ ] Failing test: fact checker downgrades `"is the market leader"` (no evidence) to `unsupported` and the repaired report no longer contains that sentence.
- [ ] Failing test: `MAX_REPORT_REPAIR_ROUNDS=2` → repair is attempted exactly twice then gives up with `repair_exhausted=True`.
- [ ] Failing test: quality check flags a numeric statistic appearing in the report with no `Evidence` of claim_type `observed`.
- [ ] Failing test: `ChangeDetector` on two snapshots with one changed price emits exactly one `pricing` event with `previous_value`/`new_value`/`detected_at`/`source`.
- [ ] Run → FAIL. Implement.
- [ ] Run → PASS.

## Task 12: LangGraph workflow

**Files:** `app/graph/{state,workflow,routing}.py`, `app/graph/nodes/*.py` (14), `tests/graph/test_workflow.py`, `tests/graph/test_partial_failure.py`

**Interfaces:** `ResearchState(TypedDict, total=False)` exactly the fields in spec §8, plus `progress: int`, `current_step: str`, `repair_rounds: int`, `provider_health: dict`. Node registry maps node name → `(coroutine, owned_state_keys)`; a test asserts the owned-key sets are **disjoint**, so no node mutates another's fields (spec §8). `build_workflow(checkpointer=None) -> CompiledStateGraph` with the edges of spec §7 and conditional edges `route_browser_required`, `route_graph_enabled`, `route_after_quality_control` (repair loop, bounded).

- [ ] Failing test: owned-key sets across all 14 nodes are pairwise disjoint.
- [ ] Failing test: running the graph with a provider stubbed to raise on one competitor's pricing yields run status `partial`, `failed_steps == ["pricing_analysis:CompetitorX"]`, and the report still renders.
- [ ] Failing test: quality failure loops back into `generate_report` at most `MAX_REPORT_REPAIR_ROUNDS` times then proceeds.
- [ ] Failing test: `GRAPH_RAG_ENABLED=false` never calls the graph store (assert on a spy).
- [ ] Run → FAIL. Implement.
- [ ] Run → PASS.

## Task 13: API + CLI

**Files:** `app/main.py`, `app/api/dependencies.py`, `app/api/routes/*.py` (6), `run.py`, `tests/api/test_routes.py`

**Interfaces:** the 8 routes of spec §52. `POST /api/v1/analysis` returns `202 {run_id, status:"queued"}` and schedules `ResearchService.execute` on the app's `BackgroundTasks`, never blocking. `GET .../status` returns spec §54's shape with a real integer `progress`. `GET /api/v1/health` pings db + vector + llm + search, returning `"degraded"` (not a stack trace) when a provider is unconfigured. An `APIKeyAuth` dependency in `app/core/security.py` reads `API_KEY`, and when unset the app runs open in dev but the test asserts the header is required when set. Error handler returns `{error:{type,message,retryable}}` with no stack trace and no secret.

- [ ] Failing test: `POST` returns 202 + queued, and the run reaches `completed` when the graph is stubbed.
- [ ] Failing test: `GET /api/v1/analysis/{id}/report` on an unknown id returns 404 with a safe body, not 500.
- [ ] Failing test: response body of every route contains no substring of `Settings.openai_api_key` after setting it.
- [ ] Failing test: `python run.py analyze --company ... ` writes `data/reports/<run_id>.{json,md}`.
- [ ] Run → FAIL. Implement.
- [ ] Run → PASS.

## Task 14: Optional graph layer (GraphRAG)

**Files:** `app/graph_rag/{entities,relationships,extraction,queries}.py`, `app/providers/graph/memory.py`, `tests/unit/test_graph_rag.py`

**Interfaces:** `EntityType`/`RelationshipType` enums from spec §26, `GraphStore` impl in-memory with `query_entity(name)`, `query_neighbors(...)`, `find_missing_relations()`; the 6 question templates of spec §26 are runnable query strings.

- [ ] Failing test: insert Company→OFFERS→Product→HAS_FEATURE→Feature; `query_entity("Product").neighbors` returns the company and the feature; `find_missing_relations()` reports competitor pairs with no `COMPETES_WITH` edge.
- [ ] Run → FAIL. Implement.
- [ ] Run → PASS.

## Task 15: Docker, docs, final verification

**Files:** `Dockerfile`, `docker-compose.yml`, `README.md`, `frontend/README.md`, `scripts/*.py`, `requirements.txt`

- [ ] `ruff check . && ruff format --check .` → exit 0.
- [ ] `pytest -q` → 0 failures, 0 errors, no warnings summary.
- [ ] `python -c "from app.main import app"` → imports clean.
- [ ] Health check script runs against the app and prints a 200 + `status` field.
- [ ] **Docker build is NOT verifiable here** — Docker is not installed on this machine. State this explicitly in the final report rather than claiming it passes. Provide the Dockerfile and note the exact command the user should run.
- [ ] Final whole-branch review subagent over the full diff before claiming completion.

---

## Execution Order

1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11 → 12 → 13 → 14 → 15

Tasks 10 and 14 are independent of each other's internals but both depend on 2/4/8. Task 13 depends on 12. Strictly sequential given shared interfaces.

## Self-Review

**1. Spec coverage.** All 120 sections map: §4/§112→T1, §5/§47/§48→T4, §6→file structure, §8→T12, §9→T9, §10→T9, §11/§12→T9, §13→T8, §14/§15→T2/T7, §16→T2/T7/T10, §17/§18→T6, §19/§21→T5, §20→T6, §22→T10, §23/§24/§25/§89→T10, §26→T14, §27-§35→T10, §36→T11, §37/§38→T11, §39/§40→T11, §41/§42→T7, §43/§44/§77→T1/T8, §45/§76→T10, §46→T9/T10, §49→T4, §50/§51/§103→T3, §52-§55→T13, §56/§57/§58→T4/T5, §59→T6, §60/§61→T11, §62-§65→T11/T13, §66/§67→T11, §69/§70→T1, §71/§72/§73→T5/T13, §74/§75→T5, §78/§79/§80/§95-§98→T8, §81/§82→T11, §83→T13, §84→T13, §85→all tasks, §90/§91/§92/§93/§94→T5/T6/T10, §99/§100/§101/§102/§105→T3/T11/T13, §107/§108/§109/§110/§111/§112/§114→T15, §113→all, §116/§117→T13/T15, §118/§119→architecture. No section is unassigned.

**2. Step scan.** Every step names a file, a signature, or a command. No "handle edge cases", no TBD.

**3. Type consistency.** `Source`, `Evidence`, `Finding` are defined once in T2 and consumed unchanged by T7/T10/T11/T13. `ResearchState` fields are defined in T12 and match spec §8 verbatim. `AgentResult` defined in T9, returned by T10/T11.

**4. Review Focus.** All 5 lines have a named owning task and test. ✓

**5. Proportion.** This plan is ~1/10 the length of the spec and ~1/50 the length of the resulting code. Signatures and test names, not bodies. ✓
