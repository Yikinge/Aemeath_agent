"""确定性单元测试：检索三因子打分 + MemoryBank 遗忘曲线（不调 LLM）。

这层是 CI 的主力——纯函数 + 临时 SQLite，快且稳。LLM 抽取质量的评测另放（手动 eval）。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from agent.memory.models import MemoryItem
from agent.memory.retrieve import _age_days, _minmax, _recency, retrieve


def _iso(days_ago: float = 0.0) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()


def _item(content: str = "x", **kw) -> MemoryItem:
    return MemoryItem(content=content, embedding=None, **kw)


# ---------- min-max 归一化（Generative Agents） ----------

def test_minmax_basic():
    assert _minmax([1.0, 2.0, 3.0]) == [0.0, 0.5, 1.0]


def test_minmax_all_equal_gives_full():
    # 全相同 → 该因子不参与区分，给满分（不影响排序）
    assert _minmax([5.0, 5.0, 5.0]) == [1.0, 1.0, 1.0]


def test_minmax_empty():
    assert _minmax([]) == []


# ---------- recency 遗忘曲线（MemoryBank R=exp(-t/(TAU·strength))） ----------

def test_recency_fresh_near_one():
    assert _recency(_item(created_at=_iso(0))) > 0.99


def test_recency_decays_with_age():
    assert _recency(_item(created_at=_iso(60))) < _recency(_item(created_at=_iso(10)))


def test_strength_slows_decay():
    weak = _recency(_item(created_at=_iso(30), strength=1.0))
    strong = _recency(_item(created_at=_iso(30), strength=8.0))
    assert strong > weak  # 被强化过 → 同龄衰减更慢（间隔重复）


def test_last_accessed_overrides_created():
    # MemoryBank：t = 距上次使用，而非距创建
    it = _item(created_at=_iso(60), last_accessed=_iso(1))
    assert _age_days(it) < 2


# ---------- 三因子排序（端到端，临时 store） ----------

async def test_retrieve_recency_breaks_tie(store, router):
    # 相关性、重要性相同（都靠关键词命中、importance 相同），只差新旧 → 新的排前
    await store.add_memory_item(_item("新", keywords=["猫"], importance=0.8, created_at=_iso(0)))
    await store.add_memory_item(_item("旧", keywords=["猫"], importance=0.8, created_at=_iso(60)))
    hits = await retrieve(store, router, "猫", k=5)
    assert [h.item.content for h in hits] == ["新", "旧"]


async def test_retrieve_importance_breaks_tie(store, router):
    # 相关性、新旧相同，只差重要性 → 重要的排前
    await store.add_memory_item(_item("重要", keywords=["猫"], importance=0.9, created_at=_iso(0)))
    await store.add_memory_item(_item("普通", keywords=["猫"], importance=0.3, created_at=_iso(0)))
    hits = await retrieve(store, router, "猫", k=5)
    assert hits[0].item.content == "重要"


async def test_retrieve_recall_floor_filters_irrelevant(store, router):
    # 完全不相关（无关键词命中、无向量）→ 被准入门槛挡掉
    await store.add_memory_item(_item("不相关", keywords=["天气"], importance=0.9, created_at=_iso(0)))
    hits = await retrieve(store, router, "讨论量子物理", k=5)
    assert hits == []


async def test_retrieve_components_present(store, router):
    await store.add_memory_item(_item("有猫", keywords=["猫"], importance=0.5, created_at=_iso(0)))
    hits = await retrieve(store, router, "猫", k=5)
    assert hits and set(hits[0].components) == {"relevance", "recency", "importance", "final"}
