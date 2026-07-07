"""混合检索 + 三因子打分（B1，TDD §4.5）。

召回：向量语义 + Lorebook 关键词，混合成"相关性"，并设相关性准入门槛
（对话注入场景：不相关的不被"新/重要"救回）。
排序：Generative Agents 三因子范式 —— relevance + recency + importance，
各自 min-max 归一化后等权相加；其中 recency 用 MemoryBank 遗忘曲线
R = exp(-t/(TAU·strength))（t = 距上次命中天数，被检索则 strength 增强）。

单用户量级，向量用纯 Python 余弦（零原生依赖）；量大了换 sqlite-vec，本模块接口不变。
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from agent.gateway.router import LLMRouter
from agent.memory.models import MemoryHit, MemoryItem
from agent.memory.store import Store

_RECALL_FLOOR = 0.15     # 相关性准入：低于此与当前话题无关，不进候选（不被新/重要救回）
_TAU_DAYS = 30.0         # MemoryBank R=e^(-t/S) 的基准时间常数；原版 1 天对长期陪伴衰减过快，取 30 天
_DEFAULT_K = 5


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):  # 维度不一致（如切换过 embedder）直接不算相似
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _keyword_score(query: str, item: MemoryItem) -> float:
    """Lorebook 思路：关键词出现在当前消息里就唤起这条记忆。"""
    if not item.keywords:
        return 0.0
    hit = sum(1 for k in item.keywords if k and k in query)
    return hit / len(item.keywords)


def _age_days(item: MemoryItem) -> float:
    """距上次命中（无则距创建）的天数 —— MemoryBank 的 t = days since last use。"""
    ref = item.last_accessed or item.created_at
    try:
        t = datetime.fromisoformat(ref)
    except (ValueError, TypeError):
        return 0.0
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - t).total_seconds() / 86400.0)


def _recency(item: MemoryItem) -> float:
    """MemoryBank 遗忘曲线 R = exp(-t/(TAU·strength))；strength 越大衰减越慢（间隔重复）。"""
    s = max(0.1, item.strength)
    return math.exp(-_age_days(item) / (_TAU_DAYS * s))


def _minmax(vals: list[float]) -> list[float]:
    """Generative Agents：每个因子 min-max 归一化到 [0,1]。全相同则该因子不参与区分（给满分）。"""
    if not vals:
        return []
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return [1.0 for _ in vals]
    return [(v - lo) / (hi - lo) for v in vals]


async def retrieve(
    store: Store, router: LLMRouter, query: str, *, namespace: str = "default", k: int = _DEFAULT_K
) -> list[MemoryHit]:
    items = await store.active_memory_items(namespace)  # 时效过滤：只取 valid_until IS NULL
    # 生命周期硬过滤（方案 §8.3）：已过期的（expires_at <= now）不进打分，省 token、防误注入
    now = datetime.now(timezone.utc).isoformat()
    items = [it for it in items if not it.expires_at or it.expires_at > now]
    if not items:
        return []

    qvec = (await router.embed([query]))[0]

    # ① 相关性 + 准入门槛
    pool: list[tuple[MemoryItem, float]] = []
    for it in items:
        v = max(0.0, _cosine(qvec, it.embedding)) if it.embedding else 0.0
        kw = _keyword_score(query, it)
        rel = v + kw - v * kw  # 概率 OR：语义或关键词任一强即可唤起
        if rel >= _RECALL_FLOOR:
            pool.append((it, rel))
    if not pool:
        return []

    # ② 三因子 → ③ min-max 归一化 + 等权求和（Generative Agents）
    n_rel = _minmax([rel for _, rel in pool])
    n_rec = _minmax([_recency(it) for it, _ in pool])
    n_imp = _minmax([it.importance for it, _ in pool])

    hits: list[MemoryHit] = []
    for (it, _), r, c, m in zip(pool, n_rel, n_rec, n_imp):
        final = r + c + m
        hits.append(MemoryHit(
            item=it, score=final,
            components={"relevance": round(r, 3), "recency": round(c, 3),
                        "importance": round(m, 3), "final": round(final, 3)},
        ))

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:k]
