"""记忆类型与生命周期常量（方案 §4）+ route 阶段产出的候选。

分类不是为了概念完整，而是为了决定**写入 / 过期 / 召回 / 注入**策略：
每条记忆都要知道它是什么、从哪来、何时有效、何时过期、该不该注入。
"""

from __future__ import annotations

from dataclasses import dataclass, field

# route 候选的 memory_type（决定配进哪一层；DISCARD = 不值得记）
PROFILE_FACT = "profile_fact"
ENTITY_FACT = "entity_fact"
PREFERENCE = "preference"
EVENT_MEMORY = "event_memory"
ACTIVE_COMMITMENT = "active_commitment"
MOOD_OBSERVATION = "mood_observation"
INSIGHT = "insight"
DISCARD = "discard"

# 只有这些类型会被配进叙事层（其余各有专职抽取器或直接丢弃）
NARRATIVE_TYPES = {EVENT_MEMORY, INSIGHT}

# narrative_note.status 生命周期（方案 §5.2）
STATUS_ACTIVE = "active"
STATUS_ARCHIVED = "archived"
STATUS_EXPIRED = "expired"
STATUS_SUPERSEDED = "superseded"

# 叙事 resolve 的判定（方案 §6.3）
RESOLVE_VERDICTS = ("SAME", "MERGE", "UPDATE", "EXPIRE", "NEW")


@dataclass
class RoutedMemory:
    """route 阶段产出的一条带分类的候选记忆（方案 §6.2）。

    取代旧的 MemoryCandidate「只有 content/keywords/importance」——现在每条候选自带
    类型、规范键、事件时间、过期时间、置信度和「为什么值得记」。
    """

    memory_type: str
    content: str
    canonical_key: str | None = None
    event_at: str | None = None
    expires_at: str | None = None
    keywords: list[str] = field(default_factory=list)
    importance: float = 0.5      # 0..1
    confidence: float = 0.6      # 0..1
    sensitivity: str = "low"     # low / personal / care —— 敏感度，P2 注入门控用
    reason: str = ""
