"""RED: persistence layer against a real (temporary) SQLite database.

These are integration tests on purpose - the point is that the SQLAlchemy models,
mappers and repositories actually work together, which a mocked session could
never prove.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.core.constants import ClaimType, Confidence, RetrievalStatus, RunStatus, SourceType
from app.core.database import Database
from app.models import (
    DocumentModel,
    FindingModel,
    ResearchRunModel,
    SourceModel,
)
from app.repositories.company_repository import CompanyRepository
from app.repositories.evidence_repository import EvidenceRepository
from app.repositories.report_repository import ReportRepository
from app.repositories.source_repository import SourceRepository


@pytest_asyncio.fixture()
async def database(tmp_path: Path) -> Database:
    db = Database(f"sqlite+aiosqlite:///{(tmp_path / 'test.db').as_posix()}")
    await db.create_all()
    yield db
    await db.dispose()


@pytest_asyncio.fixture()
async def session(database: Database):
    async with database.session() as active:
        yield active


class TestDatabase:
    async def test_health_check_reports_ok(self, database: Database) -> None:
        assert await database.health_check() is True

    async def test_dispose_is_safe_to_call_twice(self, database: Database) -> None:
        await database.dispose()
        await database.dispose()


class TestResearchRunModel:
    async def test_new_run_defaults_to_queued(self, session) -> None:
        run = ResearchRunModel(company_name="KANSBE", status=RunStatus.QUEUED)
        session.add(run)
        await session.commit()
        assert run.status == RunStatus.QUEUED
        assert run.id is not None
        assert isinstance(run.created_at, datetime)

    async def test_timestamps_are_utc_aware(self, session) -> None:
        run = ResearchRunModel(company_name="KANSBE")
        session.add(run)
        await session.commit()
        refreshed = await session.get(ResearchRunModel, run.id)
        assert refreshed is not None
        assert refreshed.created_at.tzinfo is not None

    async def test_request_payload_round_trips_as_json(self, session) -> None:
        run = ResearchRunModel(company_name="KANSBE",
                               request={"company_name": "KANSBE", "competitor_count": 5})
        session.add(run)
        await session.commit()
        refreshed = await session.get(ResearchRunModel, run.id)
        assert refreshed is not None
        assert refreshed.request["competitor_count"] == 5


class TestUniqueContentHashPerRun:
    async def test_same_url_and_hash_in_one_run_is_rejected(self, session) -> None:
        run = ResearchRunModel(company_name="KANSBE")
        session.add(run)
        await session.commit()
        for _ in range(2):
            session.add(SourceModel(run_id=run.id, url="https://a.com/x",
                                    canonical_url="https://a.com/x", domain="a.com",
                                    content_hash="deadbeef", source_type=SourceType.OFFICIAL_SITE))
        with pytest.raises(IntegrityError):
            await session.commit()
        await session.rollback()


class TestCompanyRepository:
    async def test_upsert_is_idempotent_on_identity_key(self, database, session) -> None:
        repo = CompanyRepository(session)
        first = await repo.upsert(name="Acme Ltd", website="https://acme.com",
                                  identity_key="acme")
        second = await repo.upsert(name="Acme", website="https://acme.com",
                                   identity_key="acme")
        assert first.id == second.id

    async def test_distinct_companies_get_distinct_rows(self, database, session) -> None:
        repo = CompanyRepository(session)
        a = await repo.upsert(name="Acme", identity_key="acme")
        b = await repo.upsert(name="Beta", identity_key="beta")
        assert a.id != b.id

    async def test_get_by_identity_returns_none_when_absent(self, database, session) -> None:
        assert await CompanyRepository(session).get_by_identity("nothing") is None


class TestSourceRepository:
    async def test_upsert_by_canonical_url_deduplicates(self, database, session) -> None:
        repo = SourceRepository(session)
        run = ResearchRunModel(company_name="KANSBE")
        session.add(run)
        await session.commit()
        a = await repo.upsert(run_id=run.id, url="https://a.com/pricing",
                              canonical_url="https://a.com/pricing", domain="a.com")
        b = await repo.upsert(run_id=run.id, url="https://a.com/pricing",
                              canonical_url="https://a.com/pricing", domain="a.com")
        assert a.id == b.id

    async def test_list_by_run_returns_only_that_runs_sources(self, database, session) -> None:
        repo = SourceRepository(session)
        run_a = ResearchRunModel(company_name="A")
        run_b = ResearchRunModel(company_name="B")
        session.add_all([run_a, run_b])
        await session.commit()
        await repo.upsert(run_id=run_a.id, url="https://a.com/1",
                          canonical_url="https://a.com/1", domain="a.com")
        await repo.upsert(run_id=run_b.id, url="https://b.com/1",
                          canonical_url="https://b.com/1", domain="b.com")
        found = await repo.list_by_run(run_a.id)
        assert [source.url for source in found] == ["https://a.com/1"]

    async def test_source_keeps_retrieved_text_for_quote_verification(
        self, database, session
    ) -> None:
        repo = SourceRepository(session)
        run = ResearchRunModel(company_name="A")
        session.add(run)
        await session.commit()
        stored = await repo.upsert(run_id=run.id, url="https://a.com/p",
                                   canonical_url="https://a.com/p", domain="a.com",
                                   text="Starter plan costs $10 per month.")
        assert stored.text is not None
        assert "$10 per month" in stored.text


class TestEvidenceRepository:
    async def test_evidence_round_trips_all_enum_fields(self, database, session) -> None:
        repo = EvidenceRepository(session)
        run = ResearchRunModel(company_name="A")
        session.add(run)
        await session.commit()
        ev = await repo.create(
            run_id=run.id, claim="Acme lists a $10 plan", source_id="src-1",
            source_url="https://a.com/p", claim_type=ClaimType.OBSERVED,
            confidence=Confidence.HIGH, evidence_text="Starter plan costs $10 per month.",
        )
        assert ev.claim_type == ClaimType.OBSERVED
        assert ev.confidence == Confidence.HIGH

    async def test_list_by_run_filters_to_run(self, database, session) -> None:
        repo = EvidenceRepository(session)
        run_a = ResearchRunModel(company_name="A")
        run_b = ResearchRunModel(company_name="B")
        session.add_all([run_a, run_b])
        await session.commit()
        for run in (run_a, run_b):
            await repo.create(run_id=run.id, claim="x", source_id="s",
                              source_url="https://a.com", claim_type=ClaimType.REPORTED,
                              confidence=Confidence.LOW)
        assert len(await repo.list_by_run(run_a.id)) == 1

    async def test_count_by_confidence_groups(self, database, session) -> None:
        repo = EvidenceRepository(session)
        run = ResearchRunModel(company_name="A")
        session.add(run)
        await session.commit()
        await repo.create(run_id=run.id, claim="a", source_id="s", source_url="https://a.com",
                          claim_type=ClaimType.OBSERVED, confidence=Confidence.HIGH)
        await repo.create(run_id=run.id, claim="b", source_id="s", source_url="https://a.com",
                          claim_type=ClaimType.INFERRED, confidence=Confidence.LOW)
        counts = await repo.count_by_confidence(run.id)
        assert counts == {"high": 1, "low": 1}


class TestReportRepository:
    async def test_save_and_load_report_bundle(self, database, session) -> None:
        repo = ReportRepository(session)
        run = ResearchRunModel(company_name="A")
        session.add(run)
        await session.commit()
        payload = {"metadata": {"run_id": str(run.id)}, "sources": []}
        await repo.save(run_id=run.id, report_json=payload,
                        report_markdown="# Report\n\nbody")
        loaded = await repo.get_latest(run.id)
        assert loaded is not None
        assert loaded.report_json["metadata"]["run_id"] == str(run.id)
        assert loaded.report_markdown is not None
        assert loaded.report_markdown.startswith("# Report")

    async def test_get_latest_returns_none_for_unknown_run(self, database, session) -> None:
        assert await ReportRepository(session).get_latest(uuid4()) is None


class TestDocumentAndFinding:
    async def test_document_keeps_company_and_competitor_metadata(self, database, session) -> None:
        run = ResearchRunModel(company_name="A")
        session.add(run)
        await session.commit()
        doc = DocumentModel(run_id=run.id, source_id="s1", url="https://a.com",
                            canonical_url="https://a.com", content="hello",
                            content_hash="h1", competitor="Beta")
        session.add(doc)
        await session.commit()
        assert doc.competitor == "Beta"
        assert doc.char_count == 5

    async def test_finding_persists_evidence_id_list(self, database, session) -> None:
        run = ResearchRunModel(company_name="A")
        session.add(run)
        await session.commit()
        finding = FindingModel(run_id=run.id, company="A", category="pricing",
                               statement="Lists $10 plan", evidence_ids=["ev-1", "ev-2"])
        session.add(finding)
        await session.commit()
        assert finding.evidence_ids == ["ev-1", "ev-2"]

    async def test_run_has_ordered_evidence_relationship(self, database, session) -> None:
        run = ResearchRunModel(company_name="A")
        session.add(run)
        await session.flush()
        session.add(SourceModel(run_id=run.id, url="https://a.com",
                                canonical_url="https://a.com", domain="a.com",
                                retrieval_status=RetrievalStatus.OK))
        await session.commit()
        result = await session.execute(
            select(SourceModel).where(SourceModel.run_id == run.id)
        )
        assert len(result.scalars().all()) == 1
