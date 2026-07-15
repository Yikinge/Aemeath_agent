"""确定性单测：承诺生命周期（§9）+ 去重 + due/opportunistic + 跨天锚点 + MEM-7 情绪调制。

用按 prompt 应答的 PromptRouter，不调真 LLM。
"""

from __future__ import annotations

import json

from agent.memory.consolidator import Consolidator
from agent.memory.service import MemoryService
from agent.proactive.engine import ProactiveEngine
from agent.proactive.policy import filter_candidates
from tests.conftest import PromptRouter

_PAST = "2020-01-01"     # 远古日期 → 一定"到期"


def _stack(store, tmp_path, router):
    cons = Consolidator(store, router, str(tmp_path / "MEMORY.md"))
    mem = MemoryService(store, router, cons, consolidate_threshold=1)
    eng = ProactiveEngine(store, router, mem, persona="你是爱弥斯")
    return cons, mem, eng


# ---------- A2 / #5：due 必须有时间；无日期 open_loop 不自动到期 ----------

async def test_undated_open_loop_not_due(store):
    await store.add_commitment("open_loop", "找机会问问换工作的事", due_at=None)
    assert await store.due_commitments("2026-06-20") == []          # 不算到期
    assert len(await store.opportunistic_commitments("2026-06-20")) == 1  # 走低频 opportunistic


async def test_timed_commitment_is_due(store):
    await store.add_commitment("event_check_in", "面试关心", due_at=None, event_at=_PAST)
    due = await store.due_commitments("2026-06-20")
    assert len(due) == 1 and due[0]["event_at"] == _PAST


async def test_due_commitment_compares_timezone_offsets(store):
    await store.add_commitment(
        "care_check_in", "北京时间九点关心",
        due_at="2026-06-25T09:00:00+08:00",
        event_at=None, expires_at="2026-06-25T15:00:00+08:00",
    )
    due = await store.due_commitments("2026-06-25T02:00:00+00:00")
    assert len(due) == 1 and due[0]["content"] == "北京时间九点关心"


async def test_due_at_takes_precedence_over_event_at(store):
    await store.add_commitment(
        "event_check_in", "面试后关心结果",
        due_at="2026-06-25T17:00:00+08:00",
        event_at="2026-06-25T15:00:00+08:00",
        expires_at="2026-06-30T23:59:59+08:00",
    )
    assert await store.due_commitments("2026-06-25T08:00:00+00:00") == []
    due = await store.due_commitments("2026-06-25T09:00:00+00:00")
    assert len(due) == 1 and due[0]["content"] == "面试后关心结果"


# ---------- A1 / #4：(kind, canonical_key) 签名去重 ----------

async def test_commitment_signature_dedup(store, tmp_path):
    router = PromptRouter(commitments=json.dumps([{
        "kind": "event_check_in", "content": "2026-06-24 用户有重要面试",
        "event_at": "2026-06-24", "canonical_key": "event:interview:2026-06-24"}]))
    cons, _, _ = _stack(store, tmp_path, router)
    await store.add_pending_intake("下周三面试", source_at="2026-06-18T10:00:00+08:00", timezone="Asia/Shanghai")
    await cons.consolidate()
    await store.add_pending_intake("对了那个面试", source_at="2026-06-19T10:00:00+08:00", timezone="Asia/Shanghai")
    await cons.consolidate()
    assert len(await store.list_all_commitments(status="open")) == 1   # 同 canonical_key 不重复


async def test_commitment_window_fields_are_persisted(store, tmp_path):
    router = PromptRouter(commitments=json.dumps([{
        "kind": "event_check_in",
        "content": "2026-06-24 面试后关心结果",
        "event_at": "2026-06-24T15:00:00+08:00",
        "due_window_start": "2026-06-24T18:00:00+08:00",
        "due_window_end": "2026-06-25T12:00:00+08:00",
        "dedupe_key": "interview:2026-06-24",
        "confidence": 0.92,
    }]))
    cons, _, _ = _stack(store, tmp_path, router)
    await store.add_pending_intake("下周三下午三点面试，之后问问我怎么样",
                                   source_at="2026-06-18T10:00:00+08:00",
                                   timezone="Asia/Shanghai")
    await cons.consolidate()
    c = (await store.list_all_commitments(status="open"))[0]
    assert c["due_window_start"] == "2026-06-24T18:00:00+08:00"
    assert c["due_window_end"] == "2026-06-25T12:00:00+08:00"
    assert c["dedupe_key"] == "interview:2026-06-24"
    assert c["confidence"] == 0.92


# ---------- B / #2：跨天 pending 各按自己的 source_at 锚点 ----------

async def test_cross_day_per_item_anchoring(store, tmp_path):
    router = PromptRouter(commitments=json.dumps([{
        "kind": "deadline_check", "content": "今天要交报告", "due_hint": "none"}]))
    cons, _, _ = _stack(store, tmp_path, router)
    # 两条不同天的 pending，一起 consolidate
    await store.add_pending_intake("今天要交报告", source_at="2026-06-18T10:00:00+08:00", timezone="Asia/Shanghai")
    await store.add_pending_intake("今天要交报告", source_at="2026-06-20T10:00:00+08:00", timezone="Asia/Shanghai")
    await cons.consolidate()
    commits = await store.list_all_commitments(status="open")
    event_dates = {c["event_at"] for c in commits}
    # 每条"今天"锚到各自的 source_at（旧的 lump 实现会把两条都锚到最后一条 = 6/20）
    assert "2026-06-18" in event_dates and "2026-06-20" in event_dates


# ---------- §9 / #1：发出转 sent（即时从 MEMORY.md 移除）→ 用户回应闭成 done ----------

async def test_sent_drops_from_md_then_closes_on_reply(store, tmp_path):
    router = PromptRouter()
    cons, mem, eng = _stack(store, tmp_path, router)
    cid = await store.add_commitment(
        "event_check_in", "面试后关心结果", due_at=None, event_at=_PAST,
        expires_at="2099-01-01",
    )
    await cons.refresh_memory_md()
    assert "面试后关心结果" in cons.read_memory_md()      # 一开始在「当前开放回路」

    res = await eng.tick(force=True)
    assert res["sent"] is True
    sent = await store.list_all_commitments(status="sent")
    assert len(sent) == 1 and sent[0]["id"] == cid        # 转 sent，不是 done
    assert "面试后关心结果" not in cons.read_memory_md()  # 立即从 MEMORY.md 移除（修 #1）

    closed = await mem.close_sent_commitments()           # 用户来消息 → 闭合
    assert closed == 1
    assert (await store.list_all_commitments(status="done"))[0]["id"] == cid


# ---------- A4 / #1：遗忘画像后即时刷新 MEMORY.md ----------

async def test_forget_fact_refreshes_md(store, tmp_path):
    router = PromptRouter(profile=json.dumps(
        [{"category": "bio", "key": "name", "value": "小王"}]))
    cons, mem, _ = _stack(store, tmp_path, router)
    await store.add_pending_intake("我叫小王", source_at="2026-06-18T10:00:00+08:00", timezone="Asia/Shanghai")
    await cons.consolidate()
    assert "小王" in cons.read_memory_md()
    fact = (await store.all_active_facts())[0]
    await mem.forget(fact.id)
    assert "小王" not in cons.read_memory_md()             # 遗忘后 MEMORY.md 立即不含


# ---------- D / MEM-7：情绪低落时压制催事类、保留关心类 ----------

async def test_low_mood_suppresses_deadline_nag(store, tmp_path):
    router = PromptRouter()
    _, _, eng = _stack(store, tmp_path, router)
    await store.add_commitment("deadline_check", "催交报告", due_at=None, event_at=_PAST)
    await store.add_mood(valence=-0.6, arousal=0.5, signals=["累"], note="很低落")
    res = await eng.tick(force=True)
    assert res["sent"] is False                            # 低落 → 不催 deadline


async def test_low_mood_keeps_care_checkin(store, tmp_path):
    router = PromptRouter()
    _, _, eng = _stack(store, tmp_path, router)
    await store.add_commitment("care_check_in", "关心一下最近状态", due_at=None, event_at=_PAST)
    await store.add_mood(valence=-0.6, arousal=0.5, signals=["累"], note="很低落")
    res = await eng.tick(force=True)
    assert res["sent"] is True                             # 低落 → 仍发关心


async def test_proactive_notify_false_does_not_send(store, tmp_path):
    router = PromptRouter()
    router.complete = lambda messages, **kwargs: _async_json({
        "notify": False,
        "outcome": "no_change",
        "summary": "现在不打扰",
        "reason": "刚聊过类似内容",
    })
    _, _, eng = _stack(store, tmp_path, router)
    await store.add_commitment("event_check_in", "面试后关心结果", due_at=None, event_at=_PAST)
    res = await eng.tick(force=True)
    assert res["sent"] is False
    assert (await store.list_all_commitments(status="open"))[0]["attempts"] == 1


async def test_low_confidence_candidate_is_skipped(store, tmp_path):
    candidates = [{"kind": "event_check_in", "content": "弱候选", "confidence": 0.2}]
    assert filter_candidates(candidates, low_mood=False, force=False) == []


async def test_post_guard_blocks_duplicate_recent_message(store, tmp_path):
    msg = "想起你之前说的面试，过来轻轻问一句：后来感觉怎么样？"
    router = PromptRouter()
    router.complete = lambda messages, **kwargs: _async_json({
        "notify": True,
        "outcome": "progress",
        "summary": "想关心面试结果",
        "notification_text": msg,
        "reason": "窗口到期",
    })
    _, _, eng = _stack(store, tmp_path, router)
    cid = await store.add_commitment("event_check_in", "面试后关心结果", due_at=None, event_at=_PAST)
    await store.add_proactive_candidate("p1", cid, msg, 0.8, "old", "sent")
    res = await eng.tick(force=True)
    assert res["sent"] is False
    assert res["reason"] == "post_guard:duplicate_recent"


async def _async_json(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False)
