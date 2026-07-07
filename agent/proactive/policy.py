"""Policy gates and post-guards for soft proactive interactions."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, datetime

ACTIVE_START, ACTIVE_END = 8, 23
COOLDOWN_SEC = 30 * 60
DAILY_QUOTA = 5
MIN_CONFIDENCE = 0.4
MOOD_LOW_VALENCE = -0.3
LOW_MOOD_COOLDOWN_MULT = 1.5
LOW_MOOD_ALLOWED = {"care_check_in", "event_check_in"}


@dataclass
class GateResult:
    passed: bool
    reasons: list[str]
    low_mood: bool
    used: int
    quota: int


def in_active_hours(now: datetime) -> bool:
    return ACTIVE_START <= now.hour < ACTIVE_END


async def pre_gate(store, namespace: str, *, force: bool = False) -> GateResult:
    now = datetime.now()
    mood = await store.recent_mood(namespace)
    low = mood["valence"] is not None and mood["valence"] < MOOD_LOW_VALENCE
    today = date.today().isoformat()
    quota = max(1, DAILY_QUOTA // 2) if low else DAILY_QUOTA
    used, _ = await store.budget_used_today(today, DAILY_QUOTA)
    reasons: list[str] = []

    if not force and not in_active_hours(now):
        reasons.append("outside_active_hours")

    cooldown = COOLDOWN_SEC * (LOW_MOOD_COOLDOWN_MULT if low else 1.0)
    last = await store.last_message_ts(namespace)
    if not force and last and (time.time() - last) < cooldown:
        reasons.append("cooldown")

    if not force and used >= quota:
        reasons.append("daily_quota")

    return GateResult(
        passed=force or not reasons,
        reasons=reasons,
        low_mood=low,
        used=used,
        quota=quota,
    )


def filter_candidates(candidates: list[dict], *, low_mood: bool, force: bool = False) -> list[dict]:
    out: list[dict] = []
    for c in candidates:
        confidence = float(c.get("confidence") if c.get("confidence") is not None else 0.7)
        if not force and confidence < MIN_CONFIDENCE:
            continue
        if not force and int(c.get("attempts") or 0) >= 3:
            continue
        if low_mood and c.get("kind") not in LOW_MOOD_ALLOWED:
            continue
        out.append(c)
    return out


def post_guard(message: str | None, recent_messages: list[str] | None = None) -> tuple[bool, str]:
    text = (message or "").strip()
    if not text:
        return False, "empty_message"
    if len(text) > 240:
        return False, "too_long"
    normalized = " ".join(text.split())
    for prev in recent_messages or []:
        if normalized and normalized == " ".join((prev or "").split()):
            return False, "duplicate_recent"
    return True, "pass"
