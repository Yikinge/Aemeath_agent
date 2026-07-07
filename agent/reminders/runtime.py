"""Deterministic reminder scheduler/runtime.

Unlike proactive commitments, reminder jobs are explicit user requests. Once a
job is scheduled, delivery should not be blocked by cooldown, quota, mood, or
LLM re-judgement.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from typing import Any

from agent.memory.store import Store

SendFn = Callable[[str, str, str], Awaitable[None]]
DEFAULT_MISFIRE_GRACE_SECONDS = 24 * 60 * 60
DEFAULT_RECONCILE_INTERVAL_SECONDS = 30


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class ReminderRuntime:
    """Schedules and fires `reminder_job` rows."""

    def __init__(self, store: Store, *, send_fn: SendFn | None = None, scheduler: Any = None) -> None:
        self.store = store
        self.send_fn = send_fn
        self.scheduler = scheduler
        self._reconcile_task: asyncio.Task | None = None

    def configure(self, *, send_fn: SendFn | None = None, scheduler: Any = None) -> None:
        if send_fn is not None:
            self.send_fn = send_fn
        if scheduler is not None:
            self.scheduler = scheduler

    async def recover(self, namespace: str = "default") -> None:
        """Load scheduled jobs after startup and register/repair them."""
        now = datetime.now(timezone.utc)
        for job in await self.store.pending_reminder_jobs(namespace):
            due = _parse_dt(job.get("due_at_utc"))
            if due is None:
                continue
            grace = int(job.get("misfire_grace_seconds") or DEFAULT_MISFIRE_GRACE_SECONDS)
            if due <= now:
                age = (now - due).total_seconds()
                if age <= grace:
                    asyncio.create_task(self.fire(job["id"]))
                else:
                    await self.store.mark_reminder_expired(job["id"], f"missed by {int(age)}s")
                continue
            self.schedule(job)

    async def reconcile_due(self, namespace: str = "default") -> None:
        """Repair due reminders from SQLite.

        APScheduler date jobs are in-memory. If the event loop is paused or a
        date job is skipped as a misfire, the database remains authoritative:
        scheduled/failed rows whose due time has arrived should still be
        delivered or expired deterministically.
        """
        now = datetime.now(timezone.utc)
        for job in await self.store.pending_reminder_jobs(namespace):
            due = _parse_dt(job.get("due_at_utc"))
            if due is None or due > now:
                continue
            grace = int(job.get("misfire_grace_seconds") or DEFAULT_MISFIRE_GRACE_SECONDS)
            age = (now - due).total_seconds()
            if age <= grace:
                asyncio.create_task(self.fire(job["id"]))
            else:
                await self.store.mark_reminder_expired(job["id"], f"missed by {int(age)}s")

    def start_reconciler(
        self, *, namespace: str = "default",
        interval_seconds: int = DEFAULT_RECONCILE_INTERVAL_SECONDS,
    ) -> None:
        """Start a lightweight DB sweeper for due reminders."""
        if self._reconcile_task is not None and not self._reconcile_task.done():
            return
        self._reconcile_task = asyncio.create_task(
            self._reconcile_loop(namespace, max(1, interval_seconds))
        )

    async def stop_reconciler(self) -> None:
        if self._reconcile_task is None:
            return
        self._reconcile_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._reconcile_task

    async def _reconcile_loop(self, namespace: str, interval_seconds: int) -> None:
        while True:
            await self.reconcile_due(namespace)
            await asyncio.sleep(interval_seconds)

    def schedule(self, job: dict) -> None:
        """Register one future reminder with APScheduler when available."""
        if self.scheduler is None:
            return
        due = _parse_dt(job.get("due_at_utc"))
        if due is None:
            return
        now = datetime.now(timezone.utc)
        if due <= now:
            asyncio.create_task(self.fire(job["id"]))
            return
        self.scheduler.add_job(
            self.fire,
            "date",
            run_date=due,
            args=[job["id"]],
            id=f"reminder:{job['id']}",
            replace_existing=True,
            misfire_grace_time=int(job.get("misfire_grace_seconds") or DEFAULT_MISFIRE_GRACE_SECONDS),
        )

    def schedule_retry(self, job: dict) -> None:
        if self.scheduler is None:
            return
        retry_count = int(job.get("retry_count") or 0)
        delay = min(300, 30 * max(1, retry_count))
        self.scheduler.add_job(
            self.fire,
            "date",
            run_date=datetime.now(timezone.utc) + timedelta(seconds=delay),
            args=[job["id"]],
            id=f"reminder-retry:{job['id']}",
            replace_existing=True,
            misfire_grace_time=300,
        )

    async def fire(self, job_id: str) -> None:
        """Claim and deliver one reminder exactly once."""
        job = await self.store.claim_reminder_job(job_id)
        if job is None:
            return
        message = (job.get("message") or job.get("title") or "").strip()
        channel = job.get("delivery_channel") or "telegram"
        target = job.get("delivery_target") or ""
        try:
            if self.send_fn is None:
                raise RuntimeError("reminder delivery target is not configured")
            await self.send_fn(channel, target, message)
            await self.store.add_delivery_log(
                source_type="reminder",
                source_id=job_id,
                channel=channel,
                target=target,
                payload=message,
                status="sent",
                namespace=job.get("namespace") or "default",
            )
            await self.store.mark_reminder_sent(job_id)
        except Exception as exc:
            await self.store.add_delivery_log(
                source_type="reminder",
                source_id=job_id,
                channel=channel,
                target=target,
                payload=message,
                status="failed",
                error=str(exc),
                namespace=job.get("namespace") or "default",
            )
            await self.store.mark_reminder_failed(job_id, str(exc))
            failed = await self.store.get_reminder_job(job_id)
            if failed and int(failed.get("retry_count") or 0) < int(failed.get("max_retries") or 3):
                self.schedule_retry(failed)
