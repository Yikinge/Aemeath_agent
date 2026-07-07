"""确定性单测：route 分类把不同条目配进正确的层（方案 §6.2）。

用按 prompt 内容应答的 PromptRouter，不调真 LLM。核心验收：
- 未来计划 → commitment（带绝对 event_at），**不进** narrative；
- 单轮情绪 / discard → 不进 narrative，也不落向量碎片；
- 已发生事件 → 进 narrative（带 event_at + canonical_key）。
"""

from __future__ import annotations

import json

from agent.memory.consolidator import Consolidator
from tests.conftest import PromptRouter

_THU = "2026-06-18T10:00:00+08:00"   # 周四


def _md(tmp_path):
    return str(tmp_path / "MEMORY.md")


async def _run(store, consolidator, text, source_at=_THU):
    await store.add_pending_intake(text, source_at=source_at, timezone="Asia/Shanghai")
    return await consolidator.consolidate()


async def test_future_plan_goes_to_commitment_not_narrative(store, tmp_path):
    router = PromptRouter(
        route=json.dumps([{"memory_type": "active_commitment",
                           "content": "2026-06-24 用户有重要面试", "importance": 8}]),
        commitments=json.dumps([{"kind": "event_check_in",
                                 "content": "2026-06-24 用户有重要面试，面试后关心结果和感受",
                                 "due_hint": "none"}]),
    )
    c = Consolidator(store, router, _md(tmp_path))
    await _run(store, c, "下周三我有个重要面试")

    # 不进叙事层
    assert await store.list_narratives() == []
    assert await store.active_memory_items() == []
    # 进 commitment，且 event_at 是绝对日期
    commits = await store.list_all_commitments(status="open")
    assert len(commits) == 1
    assert commits[0]["event_at"] == "2026-06-24"

    # MEMORY.md：不出现"下周三"，出现绝对日期
    md = c.read_memory_md()
    assert "下周三" not in md and "2026-06-24" in md
    assert "## 当前开放回路" in md


async def test_mood_and_discard_not_stored_as_narrative(store, tmp_path):
    router = PromptRouter(
        route=json.dumps([
            {"memory_type": "mood_observation", "content": "用户今天工作受挫、情绪低落", "importance": 4},
            {"memory_type": "discard", "content": "帮我算个数", "importance": 1},
        ]),
    )
    c = Consolidator(store, router, _md(tmp_path))
    await _run(store, c, "唉今天上班被领导骂了，好累，对了帮我算下 3+5")

    assert await store.list_narratives() == []
    assert await store.active_memory_items() == []


async def test_event_memory_becomes_narrative_with_anchor(store, tmp_path):
    router = PromptRouter(
        route=json.dumps([{"memory_type": "event_memory",
                           "content": "2026-06-14 用户上周末去爬了香山",
                           "canonical_key": "event:hike:xiangshan",
                           "event_at": "2026-06-14", "keywords": ["香山", "爬山"], "importance": 6}]),
    )
    c = Consolidator(store, router, _md(tmp_path))
    await _run(store, c, "我上周末去爬了香山")

    narrs = await store.list_narratives()
    assert len(narrs) == 1
    assert narrs[0]["canonical_key"] == "event:hike:xiangshan"
    assert narrs[0]["event_at"] == "2026-06-14"
    items = await store.active_memory_items()
    assert len(items) == 1 and items[0].memory_type == "event_memory"


async def test_low_importance_event_filtered(store, tmp_path):
    # importance 1 → 0.1 < 价值闸门 0.3 → 不入叙事层
    router = PromptRouter(
        route=json.dumps([{"memory_type": "event_memory",
                           "content": "用户喝了杯水", "importance": 1}]),
    )
    c = Consolidator(store, router, _md(tmp_path))
    await _run(store, c, "刚喝了杯水")
    assert await store.list_narratives() == []
