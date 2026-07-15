"""Regression tests for timestamp-aware history and natural proactive discovery."""

from __future__ import annotations

from datetime import datetime, timezone

from agent.memory.consolidator import Consolidator
from agent.memory.models import ProfileFact
from agent.memory.service import MemoryService
from agent.orchestration.loop import (
    _history_for_model, _history_time_context, _strip_leaked_time_prefix,
)
from agent.proactive.engine import ProactiveEngine
from agent.proactive.policy import candidate_is_timely, post_guard
from tests.conftest import PromptRouter


def _stack(store, tmp_path, router):
    consolidator = Consolidator(store, router, str(tmp_path / "MEMORY.md"))
    memory = MemoryService(store, router, consolidator, consolidate_threshold=1)
    engine = ProactiveEngine(store, router, memory, "你是爱弥斯", "Asia/Shanghai")
    return memory, engine


def test_history_time_is_internal_and_content_stays_clean():
    old = datetime(2026, 7, 7, 2, 6, tzinfo=timezone.utc).timestamp()
    rows = [{"role": "assistant", "content": "今天10:10提醒你看股票", "ts": old}]
    history = _history_for_model(rows)
    metadata = _history_time_context(rows, "Asia/Shanghai")
    assert history[0]["content"] == "今天10:10提醒你看股票"
    assert "消息发送于" not in history[0]["content"]
    assert "2026-07-07 10:06" in metadata
    assert "不要在回复中复述" in metadata


def test_leaked_history_time_prefix_is_removed_before_delivery():
    reply = "[该消息发送于 2026-07-14 15:04，Asia/Shanghai]\n哦？今天收盘怎么样？"
    assert _strip_leaked_time_prefix(reply) == "哦？今天收盘怎么样？"
    history = _history_for_model([{"role": "assistant", "content": reply, "ts": 0}])
    assert history[0]["content"] == "哦？今天收盘怎么样？"


def test_old_follow_up_without_expiry_is_not_timely():
    candidate = {
        "kind": "event_check_in",
        "event_at": "2026-07-07T10:10:00+08:00",
        "due_window_end": None,
        "expires_at": None,
    }
    assert candidate_is_timely(candidate, "2026-07-13T10:25:00+08:00") is False


def test_explicit_active_window_overrides_fallback_freshness():
    candidate = {
        "kind": "event_check_in",
        "event_at": "2026-07-08",
        "due_window_end": "2026-07-15T23:59:00+08:00",
        "expires_at": "2026-07-15T23:59:00+08:00",
    }
    assert candidate_is_timely(candidate, "2026-07-14T14:00:00+08:00") is True


def test_profile_question_guard_blocks_awkward_meta_language_and_interviewing():
    assert post_guard(
        "为了补充画像，你平时几点下班？", kind="profile_discovery",
    ) == (False, "profile_meta_language")
    assert post_guard(
        "你平时几点下班？周末做什么？", kind="profile_discovery",
    ) == (False, "too_many_questions")
    assert post_guard(
        "顺口问一句，你平时下班后一般怎么放松呀？", kind="profile_discovery",
    ) == (True, "pass")


async def test_profile_discovery_skips_known_and_recent_topics(store, tmp_path):
    router = PromptRouter()
    memory, engine = _stack(store, tmp_path, router)
    await store.add_fact(ProfileFact(category="routine", key="workout_schedule", value="练5休2"))
    await store.add_fact(ProfileFact(category="preference", key="favorite_game", value="鸣潮"))

    first = await engine._profile_candidate("default")
    assert first is not None and first["topic"] == "work_study"

    await store.add_proactive_decision_trace(
        namespace="default", trigger_source="profile_discovery", gate_result="pass",
        candidate_id="profile:work_study", candidate_kind="profile_discovery",
        llm_notify=True, post_guard_result="pass", final_sent=True,
        message="最近主要在忙工作还是学习呀？",
    )
    second = await engine._profile_candidate("default")
    assert second is not None and second["topic"] == "food_taste"
    assert len(await memory.list_facts()) == 2


async def test_profile_decision_prompt_contains_current_time_and_natural_rules(store, tmp_path):
    router = PromptRouter()
    _, engine = _stack(store, tmp_path, router)
    candidate = {
        "id": "profile:work_study",
        "kind": "profile_discovery",
        "content": "探索主题：工作学习。自然了解最近主要在忙什么。",
        "confidence": 0.65,
        "attempts": 0,
    }
    await engine._decide(candidate, "default")
    prompt = router.calls[-1]
    assert "当前时间：" in prompt
    assert "只问一个问题" in prompt
    assert "不提画像、记忆、档案" in prompt


async def test_stale_follow_up_is_not_written_into_working_memory(store, tmp_path):
    router = PromptRouter()
    memory, _ = _stack(store, tmp_path, router)
    await store.add_commitment(
        "event_check_in", "追问很久以前的旧提醒", due_at=None,
        event_at="2020-01-01", expires_at=None,
    )
    await memory.consolidator.refresh_memory_md()
    assert "追问很久以前的旧提醒" not in memory.consolidator.read_memory_md()
