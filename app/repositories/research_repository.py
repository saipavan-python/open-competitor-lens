"""Research run repository - the run lifecycle and its questions."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.constants import ProgressStage, RunStatus
from app.models.base import utcnow
from app.models.research_run import ResearchQuestionModel, ResearchRunModel


class ResearchRunRepository:
    """Persistence and lifecycle transitions for research runs."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self, *, company_name: str, website: str | None = None,
        industry: str | None = None, country: str | None = None,
        request: dict[str, Any] | None = None, research_goal: str | None = None,
        report_language: str = "en", research_depth: str = "standard",
        enable_graph_rag: bool = False,
    ) -> ResearchRunModel:
        run = ResearchRunModel(
            company_name=company_name, website=website, industry=industry,
            country=country, request=request or {}, research_goal=research_goal,
            report_language=report_language, research_depth=research_depth,
            enable_graph_rag=enable_graph_rag, status=RunStatus.QUEUED,
        )
        self._session.add(run)
        await self._session.flush()
        return run

    async def get(self, run_id: str) -> ResearchRunModel | None:
        return await self._session.get(ResearchRunModel, run_id)

    async def set_status(self, run_id: str, status: RunStatus, *, error: str | None = None
                         ) -> ResearchRunModel | None:
        run = await self.get(run_id)
        if run is None:
            return None
        run.status = status
        if error:
            run.errors = [*run.errors, {"message": error, "at": utcnow().isoformat()}]
        await self._session.flush()
        return run

    async def set_progress(
        self, run_id: str, *, stage: ProgressStage, progress: int, current_step: str | None = None
    ) -> ResearchRunModel | None:
        run = await self.get(run_id)
        if run is None:
            return None
        run.stage = stage
        run.progress = max(0, min(100, progress))
        run.current_step = current_step
        if run.started_at is None and status_is_running(run):
            run.started_at = utcnow()
        await self._session.flush()
        return run

    async def add_warning(self, run_id: str, warning: str) -> None:
        run = await self.get(run_id)
        if run is not None and warning not in run.warnings:
            run.warnings = [*run.warnings, warning]
            await self._session.flush()

    async def add_failed_step(self, run_id: str, step: str) -> None:
        run = await self.get(run_id)
        if run is not None and step not in run.failed_steps:
            run.failed_steps = [*run.failed_steps, step]
            await self._session.flush()

    async def add_completed_step(self, run_id: str, step: str) -> None:
        run = await self.get(run_id)
        if run is not None and step not in run.completed_steps:
            run.completed_steps = [*run.completed_steps, step]
            await self._session.flush()

    async def save_questions(self, run_id: str, questions: list[dict[str, Any]]) -> None:
        result = await self._session.execute(
            select(ResearchQuestionModel).where(ResearchQuestionModel.run_id == run_id)
        )
        for existing in result.scalars().all():
            await self._session.delete(existing)
        for payload in questions:
            self._session.add(ResearchQuestionModel(
                run_id=run_id,
                question_id=payload.get("id", ""),
                question=payload.get("question", ""),
                category=str(payload.get("category", "other")),
                priority=payload.get("priority", "medium"),
                competitor=payload.get("competitor"),
                status=payload.get("status", "pending"),
                required_sources=payload.get("required_sources", []),
            ))
        await self._session.flush()

    async def list_questions(self, run_id: str) -> list[ResearchQuestionModel]:
        result = await self._session.execute(
            select(ResearchQuestionModel).where(ResearchQuestionModel.run_id == run_id)
        )
        return list(result.scalars().all())

    async def list_runs(self, *, limit: int = 50, offset: int = 0) -> list[ResearchRunModel]:
        result = await self._session.execute(
            select(ResearchRunModel).order_by(ResearchRunModel.created_at.desc())
            .limit(limit).offset(offset)
        )
        return list(result.scalars().all())


def status_is_running(run: ResearchRunModel) -> bool:
    return run.status in (RunStatus.RUNNING, RunStatus.QUEUED)
