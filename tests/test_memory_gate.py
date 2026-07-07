"""确定性单测：MemoryGate 任务感知注入门控（方案 §8.2）。纯函数，不调 LLM/store。

验收对照方案 P2：
- 工具/计算/搜索任务 → 不灌情绪叙事、敏感记忆不进工具上下文；
- 陪伴/复盘任务 → 允许 mood/event；
- 问日期 → 无时间锚点的事件类不注入；
- 过期记忆不注入。
"""

from __future__ import annotations

from agent.memory.gate import (
    TASK_COMPANION, TASK_NEUTRAL, TASK_TOOL, classify_task, gate_memories,
)
from agent.memory.models import MemoryHit, MemoryItem
from agent.memory.types import EVENT_MEMORY


def _hit(content="x", *, memory_type=EVENT_MEMORY, sensitivity="low",
         event_at=None, expires_at=None, score=2.0) -> MemoryHit:
    it = MemoryItem(content=content, memory_type=memory_type, sensitivity=sensitivity,
                    event_at=event_at, expires_at=expires_at)
    return MemoryHit(item=it, score=score)


def _decmap(decisions):
    return {d["memory_id"]: d for d in decisions}


# ---------- classify_task ----------

def test_classify_tool():
    assert classify_task("帮我算一下 3+5 等于多少")["kind"] == TASK_TOOL


def test_classify_companion_overrides_tool():
    # 既像工具又像情绪 → 按陪伴处理，别误删情绪记忆
    assert classify_task("最近压力好大，帮我算算还要熬几天")["kind"] == TASK_COMPANION


def test_classify_neutral_and_date():
    t = classify_task("煤球明天要去打疫苗吗")
    assert t["kind"] == TASK_NEUTRAL and t["mentions_date"] is True


# ---------- 门控规则 ----------

def test_tool_task_skips_emotional_narrative():
    hits = [_hit("2026-06-15 被领导骂很累", memory_type=EVENT_MEMORY)]
    kept, dec = gate_memories(hits, "帮我算下 3+5", now="2026-06-20T00:00:00+00:00")
    assert kept == []
    assert _decmap(dec)[hits[0].item.id]["reason"] == "tool_task_skip_emotional"


def test_tool_task_skips_sensitive_memory():
    hits = [_hit("体检报告异常", memory_type="entity_fact", sensitivity="personal")]
    kept, dec = gate_memories(hits, "搜一下今天天气", now="2026-06-20T00:00:00+00:00")
    assert kept == [] and _decmap(dec)[hits[0].item.id]["reason"] == "sensitive_not_for_tool"


def test_companion_task_keeps_mood_and_event():
    hits = [_hit("2026-06-15 工作受挫情绪低落", memory_type=EVENT_MEMORY, sensitivity="care")]
    kept, _ = gate_memories(hits, "唉我最近压力好大，想聊聊", now="2026-06-20T00:00:00+00:00")
    assert len(kept) == 1


def test_date_query_filters_unanchored_event():
    anchored = _hit("2026-06-24 有面试", memory_type=EVENT_MEMORY, event_at="2026-06-24")
    floating = _hit("说过想换工作", memory_type=EVENT_MEMORY, event_at=None)
    kept, dec = gate_memories([anchored, floating], "我下周几号有安排来着", now="2026-06-20T00:00:00+00:00")
    ids = {h.item.id for h in kept}
    assert anchored.item.id in ids and floating.item.id not in ids
    assert _decmap(dec)[floating.item.id]["reason"] == "date_query_needs_anchor"


def test_expired_memory_skipped():
    hits = [_hit("短期计划", expires_at="2026-06-01T00:00:00+00:00")]
    kept, dec = gate_memories(hits, "随便聊聊", now="2026-06-20T00:00:00+00:00")
    assert kept == [] and _decmap(dec)[hits[0].item.id]["reason"] == "expired"


def test_neutral_pet_memory_recalled():
    hits = [_hit("用户养了一只叫煤球的橘猫", memory_type=EVENT_MEMORY, sensitivity="low")]
    kept, _ = gate_memories(hits, "煤球最近怎么样", now="2026-06-20T00:00:00+00:00")
    assert len(kept) == 1


def test_memory_lookup_overrides_tool_skip():
    # "我上次说面试是几点" 含工具词"几点"，但这是记忆查询 → 应允许 event_memory 注入
    assert classify_task("我上次说面试是几点来着")["memory_lookup"] is True
    hit = _hit("2026-06-24 15:00 有面试", memory_type=EVENT_MEMORY, event_at="2026-06-24T15:00:00+08:00")
    kept, _ = gate_memories([hit], "我上次说面试是几点来着", now="2026-06-20T00:00:00+00:00")
    assert len(kept) == 1   # 不被 tool_task_skip_emotional 误删
