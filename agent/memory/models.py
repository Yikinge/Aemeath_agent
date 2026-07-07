"""记忆数据模型（S2 画像 + S3 语义记忆）。对应 TDD §4.2。"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    return uuid.uuid4().hex


@dataclass
class ProfileFact:
    """一条结构化画像事实，带双时间（valid_until/superseded_by）+ 可信度 + 来源。"""

    category: str
    key: str
    value: str
    confidence: float = 0.5
    sensitivity: str = "low"
    source: str = "msg"            # msg / inferred / user_edit
    namespace: str = "default"
    id: str = field(default_factory=new_id)
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    valid_until: str | None = None      # None = 仍有效；被更新/移除时写上
    superseded_by: str | None = None    # 被哪条新事实取代（可回溯）
    canonical_key: str | None = None    # 实体规范键（如 entity:pet:cat）：把拆散的同实体事实并起来


@dataclass
class FactCandidate:
    """画像抽取阶段产出的候选事实（还没决定 ADD/UPDATE/NOOP）。"""

    category: str
    key: str
    value: str
    confidence: float = 0.6
    canonical_key: str | None = None


@dataclass
class MemoryItem:
    """向量召回层：一段可按 query 命中的文本碎片（"记忆碎片"）。

    人类可读的「源」（叙事笔记/画像事实）由 source_table+source_id 反向指回；
    用户在那侧遗忘后，本表通过 forget_chunks_of 级联失效。
    """

    content: str
    kind: str = "episodic"              # narrative / profile / episodic / semantic
    keywords: list[str] = field(default_factory=list)
    embedding: list[float] | None = None
    confidence: float = 0.6
    source: str = "msg"
    namespace: str = "default"
    id: str = field(default_factory=new_id)
    created_at: str = field(default_factory=now_iso)
    valid_until: str | None = None
    source_table: str | None = None     # 'narrative_note' / 'profile_fact' / None
    source_id: str | None = None
    embedder_version: str | None = None
    importance: float = 0.5             # 写入时 LLM 打分（0..1），检索三因子之一
    strength: float = 1.0              # 记忆强度；被检索命中则增强（间隔重复）
    last_accessed: str | None = None   # 上次被检索命中的时间
    # 生命周期 / 时间锚点 / 来源（方案 §5.3）：检索时可按类型和有效期过滤
    canonical_key: str | None = None    # 同一事件/实体的合并键（如 entity:pet:cat）
    memory_type: str | None = None      # event_memory / insight / …（route 分类）
    event_at: str | None = None         # 事件发生时间（绝对）
    expires_at: str | None = None       # 过期时间（短期情绪/未来计划），过了不再召回
    sensitivity: str = "low"            # low / personal / care —— 敏感度（P2 门控用）
    source_message_id: str | None = None


@dataclass
class MemoryCandidate:
    """语义记忆抽取阶段产出的候选（旧版，保留兼容；新链路用 types.RoutedMemory）。"""

    content: str
    keywords: list[str] = field(default_factory=list)
    importance: float = 0.5            # LLM 评估的长期重要性（0..1）


@dataclass
class CommitmentCandidate:
    """S6 承诺/开放回路候选：需要以后主动跟进的事。"""

    kind: str                          # care_check_in / open_loop / event_check_in / deadline_check
    content: str
    due_at: str | None = None
    due_window_start: str | None = None
    due_window_end: str | None = None
    sensitivity: str = "routine"       # routine / personal / care
    event_at: str | None = None        # 事件发生时间（绝对，由 normalize 给）
    expires_at: str | None = None      # 过了就不必再跟进
    canonical_key: str | None = None
    dedupe_key: str | None = None
    confidence: float = 0.7
    reason: str = ""


@dataclass
class MemoryHit:
    """一次检索命中的记忆 + 分数（便于注入规划与调试）。"""

    item: MemoryItem
    score: float
    components: dict | None = None   # {relevance, recency, importance, final} 打分分解（B2 Trace 用）


@dataclass
class IngestResult:
    """一次 ingest 的结果，便于日志与验收。"""

    added: list[ProfileFact] = field(default_factory=list)
    updated: list[ProfileFact] = field(default_factory=list)
    noop: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    memories: list[MemoryItem] = field(default_factory=list)
    commitments: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"画像 新增 {len(self.added)} · 更新 {len(self.updated)} · "
            f"无变化 {len(self.noop)} · 矛盾 {len(self.contradictions)}；"
            f"语义记忆 +{len(self.memories)}；承诺 +{len(self.commitments)}"
        )
