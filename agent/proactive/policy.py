"""Policy gates and post-guards for soft proactive interactions."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

ACTIVE_START, ACTIVE_END = 8, 23
COOLDOWN_SEC = 30 * 60
DAILY_QUOTA = 5
MIN_CONFIDENCE = 0.4
MOOD_LOW_VALENCE = -0.3
LOW_MOOD_COOLDOWN_MULT = 1.5
LOW_MOOD_ALLOWED = {"care_check_in", "event_check_in"}
_FALLBACK_FRESHNESS = {
    "care_check_in": timedelta(days=1),
    "deadline_check": timedelta(days=2),
    "event_check_in": timedelta(days=3),
}
_PROFILE_META_WORDS = ("补充画像", "用户画像", "建立档案", "记录下来", "为了了解你", "完善资料")
_QUESTION_HINT_RE = re.compile(r"[？?]|(?:吗|呢|什么|怎么|哪(?:个|种|些)?|有没有|会不会|喜欢不喜欢)")


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


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def candidate_is_timely(candidate: dict, now: str | datetime) -> bool:
    """Reject old inferred follow-ups when the extractor omitted an expiry.

    An explicit window remains authoritative. The fallback only applies to
    time-sensitive kinds with no end time, preventing a forgotten event from
    resurfacing weeks later.
    """
    kind = str(candidate.get("kind") or "")
    ttl = _FALLBACK_FRESHNESS.get(kind)
    if ttl is None:
        return True
    now_dt = _parse_time(now) if isinstance(now, str) else now.astimezone(timezone.utc)
    if now_dt is None:
        return True
    explicit_end = _parse_time(candidate.get("due_window_end") or candidate.get("expires_at"))
    if explicit_end is not None:
        return explicit_end >= now_dt
    start = _parse_time(
        candidate.get("due_window_start") or candidate.get("due_at") or candidate.get("event_at")
    )
    return start is None or now_dt <= start + ttl


def filter_candidates(
    candidates: list[dict], *, low_mood: bool, force: bool = False,
    now: str | datetime | None = None,
) -> list[dict]:
    out: list[dict] = []
    for c in candidates:
        confidence = float(c.get("confidence") if c.get("confidence") is not None else 0.7)
        if not force and confidence < MIN_CONFIDENCE:
            continue
        if not force and int(c.get("attempts") or 0) >= 3:
            continue
        if not force and now is not None and not candidate_is_timely(c, now):
            continue
        if low_mood and c.get("kind") not in LOW_MOOD_ALLOWED:
            continue
        out.append(c)
    return out


def post_guard(
    message: str | None, recent_messages: list[str] | None = None, *, kind: str | None = None,
) -> tuple[bool, str]:
    text = (message or "").strip()
    if not text:
        return False, "empty_message"
    if len(text) > 240:
        return False, "too_long"
    normalized = " ".join(text.split())
    for prev in recent_messages or []:
        if normalized and normalized == " ".join((prev or "").split()):
            return False, "duplicate_recent"
    if kind == "profile_discovery":
        if any(word in text for word in _PROFILE_META_WORDS):
            return False, "profile_meta_language"
        if text.count("？") + text.count("?") > 1:
            return False, "too_many_questions"
        if not _QUESTION_HINT_RE.search(text):
            return False, "not_a_question"
    return True, "pass"
