"""存储层：对话原文 + 画像(S2) + 语义记忆(S3) + 确认门(S5) + 主动引擎(S6)。

单 SQLite 文件，零运维。向量存 JSON 列、检索用纯 Python 余弦；量大了再换 sqlite-vec。
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timedelta, timezone

import aiosqlite

from agent.memory.models import MemoryItem, ProfileFact, now_iso

SCHEMA = """
CREATE TABLE IF NOT EXISTS message (
    id        TEXT PRIMARY KEY,
    namespace TEXT NOT NULL DEFAULT 'default',
    role      TEXT NOT NULL,
    content   TEXT NOT NULL,
    ts        REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_message_ns_ts ON message(namespace, ts);

CREATE TABLE IF NOT EXISTS profile_fact (
    id            TEXT PRIMARY KEY,
    namespace     TEXT NOT NULL DEFAULT 'default',
    category      TEXT NOT NULL,
    key           TEXT NOT NULL,
    value         TEXT NOT NULL,
    confidence    REAL NOT NULL DEFAULT 0.5,
    sensitivity   TEXT NOT NULL DEFAULT 'low',
    source        TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    valid_until   TEXT,
    superseded_by TEXT,
    canonical_key TEXT                  -- 实体规范键（如 entity:pet:cat）：把拆散的 pet_name/pet_species 并到一条
);
CREATE INDEX IF NOT EXISTS idx_fact_active ON profile_fact(namespace, key, valid_until);

-- 向量召回层（"记忆碎片"）：机器索引、按 query 命中后才注入；不是直接展示给人看的
-- source_table/source_id 反向指回人类可读层（profile_fact / narrative_note），用户在那侧遗忘后这边可级联失效
CREATE TABLE IF NOT EXISTS memory_item (
    id               TEXT PRIMARY KEY,
    namespace        TEXT NOT NULL DEFAULT 'default',
    content          TEXT NOT NULL,
    kind             TEXT NOT NULL DEFAULT 'episodic',
    source           TEXT,
    confidence       REAL DEFAULT 0.6,
    created_at       TEXT NOT NULL,
    valid_until      TEXT,
    keywords         TEXT,
    embedding        TEXT,
    source_table     TEXT,         -- 'narrative_note' / 'profile_fact' / null(legacy)
    source_id        TEXT,
    embedder_version TEXT,         -- 切换 embedder 后可批量识别需要重嵌的行
    importance       REAL DEFAULT 0.5,   -- 写入时 LLM 打分（0..1），检索三因子之一
    strength         REAL DEFAULT 1.0,   -- 记忆强度，被检索命中则增强（间隔重复）
    last_accessed    TEXT,                -- 上次被检索命中时间
    -- 生命周期/时间锚点/来源（方案 §5.3）：检索时可按类型与有效期硬过滤
    canonical_key     TEXT,
    memory_type       TEXT,
    event_at          TEXT,
    expires_at        TEXT,
    sensitivity       TEXT DEFAULT 'low',
    source_message_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_mem_active ON memory_item(namespace, valid_until);
-- idx_mem_source 依赖后补列 source_table，移到 _migrate 之后建（旧库升级安全）

-- 人类可读层 · 叙事笔记（具体小事/事件/重要时刻/共同话题），用户可在控制台直接编辑/遗忘
CREATE TABLE IF NOT EXISTS narrative_note (
    id            TEXT PRIMARY KEY,
    namespace     TEXT NOT NULL DEFAULT 'default',
    kind          TEXT NOT NULL DEFAULT 'event',   -- event / milestone / inside_joke / journal / insight
    content       TEXT NOT NULL,
    importance    REAL NOT NULL DEFAULT 0.5,
    source        TEXT NOT NULL DEFAULT 'consolidator', -- consolidator / user_edit / journal / reflection
    created_at    TEXT NOT NULL,
    valid_until   TEXT,
    -- 生命周期/时间锚点/来源/合并（方案 §5.2）
    canonical_key TEXT,                            -- 同一事件/实体的合并键（如 event:interview:2026-06-24）
    summary       TEXT,
    event_at      TEXT,                            -- 事件发生时间（绝对）
    observed_at   TEXT,                            -- 用户告诉 agent 的时间
    source_at     TEXT,
    expires_at    TEXT,                            -- 短期事件/计划的过期时间
    confidence    REAL DEFAULT 0.6,
    sensitivity   TEXT DEFAULT 'low',
    status        TEXT NOT NULL DEFAULT 'active',  -- active / archived / expired / superseded
    superseded_by TEXT,                            -- 被哪条新叙事取代（MERGE/UPDATE 可回溯）
    metadata_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_narr_active ON narrative_note(namespace, valid_until, importance);
CREATE INDEX IF NOT EXISTS idx_narr_canon ON narrative_note(namespace, canonical_key, status);

-- 写入缓冲（PENDING）：对话先入此表；Consolidator 达阈值/定时把这里批量跑成结构化记忆
-- 目的是保护 prompt cache：MEMORY.md 不每条消息都改，攒一波再合
CREATE TABLE IF NOT EXISTS pending_intake (
    id            TEXT PRIMARY KEY,
    namespace     TEXT NOT NULL DEFAULT 'default',
    user_text     TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'pending',  -- pending / processed
    created_at    TEXT NOT NULL,
    processed_at  TEXT,
    -- 来源/时间锚点（方案 §5.1）：解析"明天/下周三"需要 source_at + timezone，并支持精确回溯
    source_at     TEXT,
    timezone      TEXT,
    message_id    TEXT,
    source_role   TEXT
);
CREATE INDEX IF NOT EXISTS idx_pending_status ON pending_intake(namespace, status);

-- 情绪时间线（MEM-4 / ACT-6 信号源）
CREATE TABLE IF NOT EXISTS mood_log (
    id        TEXT PRIMARY KEY,
    namespace TEXT NOT NULL DEFAULT 'default',
    ts        TEXT NOT NULL,
    valence   REAL,           -- -1..1 (负面..正面)
    arousal   REAL,           -- 0..1 (平静..激动)
    signals   TEXT,           -- json: ["累","加班","委屈"...]
    note      TEXT
);
CREATE INDEX IF NOT EXISTS idx_mood_ts ON mood_log(namespace, ts);

-- 调试用 Trace：每轮对话和每次主动心跳都留痕（含完整 system prompt + 召回片段 + 决策原因）
CREATE TABLE IF NOT EXISTS turn_trace (
    id              TEXT PRIMARY KEY,
    namespace       TEXT NOT NULL DEFAULT 'default',
    ts              TEXT NOT NULL,
    user_text       TEXT,
    stable_prefix   TEXT,
    dynamic_suffix  TEXT,
    retrieved       TEXT,                -- json [{id,score,content}]
    reply           TEXT,
    latency_ms      INTEGER
);
CREATE INDEX IF NOT EXISTS idx_turn_ts ON turn_trace(namespace, ts);

CREATE TABLE IF NOT EXISTS tick_trace (
    id            TEXT PRIMARY KEY,
    namespace     TEXT NOT NULL DEFAULT 'default',
    ts            TEXT NOT NULL,
    sent          INTEGER NOT NULL,      -- 0/1
    reason        TEXT,
    commitment_id TEXT,
    message       TEXT
);
CREATE INDEX IF NOT EXISTS idx_tick_ts ON tick_trace(namespace, ts);

-- 工具调用留痕（TOOL-0 可观测）：每次工具调用记一行，控制台看「为什么调、传了啥、结果」
CREATE TABLE IF NOT EXISTS tool_trace (
    id         TEXT PRIMARY KEY,
    namespace  TEXT NOT NULL DEFAULT 'default',
    ts         TEXT NOT NULL,
    step       INTEGER,
    tool_name  TEXT NOT NULL,
    source     TEXT,                  -- builtin / mcp:<server> / skill
    arguments  TEXT,                  -- json
    result     TEXT,                  -- 截断后的结果
    ok         INTEGER NOT NULL,      -- 1 成功 / 0 异常
    ms         INTEGER
);
CREATE INDEX IF NOT EXISTS idx_tool_trace_ts ON tool_trace(namespace, ts);

CREATE TABLE IF NOT EXISTS memory_history (
    id TEXT PRIMARY KEY, target_table TEXT, target_id TEXT,
    prev_value TEXT, new_value TEXT, actor TEXT, reason TEXT, ts TEXT
);

CREATE TABLE IF NOT EXISTS contradiction_item (
    id TEXT PRIMARY KEY, namespace TEXT NOT NULL DEFAULT 'default',
    new_fact_ref TEXT, conflicting_fact_ref TEXT,
    status TEXT DEFAULT 'pending', created_at TEXT
);

-- S5 确认门：外发动作前置待确认
CREATE TABLE IF NOT EXISTS pending_action (
    id TEXT PRIMARY KEY, namespace TEXT DEFAULT 'default',
    action_type TEXT, summary TEXT, payload TEXT,
    status TEXT DEFAULT 'pending', created_at TEXT
);

-- 明确提醒/定时任务：用户显式要求“几点提醒我”时写这里，不再混入 commitment。
CREATE TABLE IF NOT EXISTS reminder_job (
    id TEXT PRIMARY KEY,
    namespace TEXT NOT NULL DEFAULT 'default',
    kind TEXT NOT NULL DEFAULT 'one_shot',
    title TEXT NOT NULL,
    message TEXT NOT NULL,
    trigger_type TEXT NOT NULL,          -- date / interval / cron
    trigger_spec TEXT NOT NULL,
    due_at_utc TEXT,
    timezone TEXT NOT NULL DEFAULT 'UTC',
    original_time_text TEXT,
    delivery_channel TEXT NOT NULL DEFAULT 'telegram',
    delivery_target TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'scheduled',
    source_message_id TEXT,
    tool_call_id TEXT,
    idempotency_key TEXT UNIQUE,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    fired_at TEXT,
    sent_at TEXT,
    acknowledged_at TEXT,
    retry_count INTEGER NOT NULL DEFAULT 0,
    max_retries INTEGER NOT NULL DEFAULT 3,
    last_error TEXT,
    misfire_grace_seconds INTEGER NOT NULL DEFAULT 86400,
    policy_json TEXT NOT NULL DEFAULT '{}',
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_reminder_status_due ON reminder_job(namespace, status, due_at_utc);

-- 所有外发留痕：reminder/proactive/scheduler/manual 后续都可以统一写这里。
CREATE TABLE IF NOT EXISTS delivery_log (
    id TEXT PRIMARY KEY,
    namespace TEXT NOT NULL DEFAULT 'default',
    source_type TEXT NOT NULL,
    source_id TEXT NOT NULL,
    channel TEXT NOT NULL,
    target TEXT NOT NULL,
    payload TEXT NOT NULL,
    status TEXT NOT NULL,
    attempted_at TEXT NOT NULL,
    sent_at TEXT,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_delivery_source ON delivery_log(source_type, source_id);

-- S5 审计：动作执行留痕
CREATE TABLE IF NOT EXISTS audit_log (
    id TEXT PRIMARY KEY, kind TEXT, summary TEXT, actor TEXT, ts TEXT
);

-- S6 承诺/开放回路：决定「主动说什么」
CREATE TABLE IF NOT EXISTS commitment (
    id TEXT PRIMARY KEY, namespace TEXT DEFAULT 'default',
    kind TEXT, sensitivity TEXT, source TEXT,
    content TEXT, due_at TEXT, status TEXT DEFAULT 'open', created_at TEXT,
    -- 时间锚点/生命周期/来源（方案 §5.4）："下周三面试" → event_at=2026-06-24
    -- 状态机：open →(主动发出 check-in)→ sent →(用户回应/nightly 兜底)→ done
    event_at TEXT, completed_at TEXT, expires_at TEXT,
    source_message_id TEXT, canonical_key TEXT, sent_at TEXT,
    due_window_start TEXT, due_window_end TEXT,
    confidence REAL DEFAULT 0.7, dedupe_key TEXT,
    attempts INTEGER NOT NULL DEFAULT 0, last_attempt_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_commitment_window ON commitment(namespace, status, due_window_start, due_window_end);

-- S6 主动候选：每条带 reason（TRUST-3 可回溯）
CREATE TABLE IF NOT EXISTS proactive_candidate (
    id TEXT PRIMARY KEY, namespace TEXT DEFAULT 'default',
    commitment_id TEXT, content TEXT, score REAL, reason TEXT,
    status TEXT, created_at TEXT
);

CREATE TABLE IF NOT EXISTS proactive_decision_trace (
    id TEXT PRIMARY KEY,
    namespace TEXT NOT NULL DEFAULT 'default',
    ts TEXT NOT NULL,
    trigger_source TEXT,
    gate_result TEXT,
    gate_reasons TEXT,
    candidate_id TEXT,
    candidate_kind TEXT,
    llm_notify INTEGER,
    llm_outcome TEXT,
    llm_summary TEXT,
    llm_reason TEXT,
    next_check_after TEXT,
    post_guard_result TEXT,
    final_sent INTEGER NOT NULL,
    message TEXT
);
CREATE INDEX IF NOT EXISTS idx_proactive_decision_ts ON proactive_decision_trace(namespace, ts);

-- S6 频率预算：每日主动条数上限
CREATE TABLE IF NOT EXISTS frequency_budget (
    date TEXT PRIMARY KEY, base_quota INTEGER, used INTEGER DEFAULT 0
);

-- 杂项 kv（如 Telegram chat_id）
CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT);
"""

_FACT_COLS = (
    "id, namespace, category, key, value, confidence, sensitivity, "
    "source, created_at, updated_at, valid_until, superseded_by, canonical_key"
)
_MEM_COLS = (
    "id, namespace, content, kind, source, confidence, created_at, valid_until, "
    "keywords, embedding, source_table, source_id, embedder_version, "
    "importance, strength, last_accessed, "
    "canonical_key, memory_type, event_at, expires_at, sensitivity, source_message_id"
)


def commitment_signature(kind: str, canonical_key: str | None, event_at: str | None, content: str) -> str:
    """承诺去重签名：同一件事不同措辞也算一条。优先 canonical_key，退 event_at，再退 content。"""
    if canonical_key:
        return f"{kind}|ck:{canonical_key}"
    if event_at:
        return f"{kind}|ev:{event_at}"
    return f"{kind}|c:{content.strip()}"


def _commitment_due_start(c: dict) -> str | None:
    return c.get("due_window_start") or c.get("due_at") or c.get("event_at")


def _commitment_due_end(c: dict) -> str | None:
    return c.get("due_window_end") or c.get("expires_at")


def _row_to_fact(r: tuple) -> ProfileFact:
    return ProfileFact(
        id=r[0], namespace=r[1], category=r[2], key=r[3], value=r[4], confidence=r[5],
        sensitivity=r[6], source=r[7], created_at=r[8], updated_at=r[9],
        valid_until=r[10], superseded_by=r[11], canonical_key=r[12],
    )


def _row_to_mem(r: tuple) -> MemoryItem:
    return MemoryItem(
        id=r[0], namespace=r[1], content=r[2], kind=r[3], source=r[4], confidence=r[5],
        created_at=r[6], valid_until=r[7],
        keywords=json.loads(r[8]) if r[8] else [],
        embedding=json.loads(r[9]) if r[9] else None,
        source_table=r[10], source_id=r[11], embedder_version=r[12],
        importance=r[13] if r[13] is not None else 0.5,
        strength=r[14] if r[14] is not None else 1.0,
        last_accessed=r[15],
        canonical_key=r[16], memory_type=r[17], event_at=r[18], expires_at=r[19],
        sensitivity=r[20] if r[20] is not None else "low", source_message_id=r[21],
    )


def _parse_time_for_compare(value: str | None) -> datetime | None:
    """把 date/datetime 字符串转成 UTC datetime，用于到期判断。

    SQLite 字符串比较无法正确比较 "2026-06-25T09:00:00+08:00" 和
    "2026-06-25T02:00:00+00:00"。这里统一 parse 后转 UTC；纯日期按当天 00:00 UTC
    处理，主要用于 event_at=YYYY-MM-DD 的整日事件。
    """
    if not value:
        return None
    s = str(value).strip()
    try:
        if len(s) == 10 and s[4] == "-" and s[7] == "-":
            dt = datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
        else:
            dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _not_expired(expires_at: str | None, now_dt: datetime) -> bool:
    exp = _parse_time_for_compare(expires_at)
    return exp is None or exp > now_dt


def _is_due(event_at: str | None, due_at: str | None, now_dt: datetime) -> bool:
    due = _parse_time_for_compare(due_at) or _parse_time_for_compare(event_at)
    return due is not None and due <= now_dt


class Store:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def init(self) -> None:
        self._db = await aiosqlite.connect(self.db_path)
        await self._db.executescript(SCHEMA)
        await self._migrate()
        await self._db.commit()

    async def _migrate(self) -> None:
        """旧库平滑升级：给各表补新列；保证已有数据不丢（沿用项目既有的前向安全风格）。"""
        async def _add_cols(table: str, specs: tuple[tuple[str, str], ...]) -> None:
            cur = await self._db.execute(f"PRAGMA table_info({table})")
            have = {r[1] for r in await cur.fetchall()}
            for col, ddl in specs:
                if col not in have:
                    await self._db.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")

        await _add_cols("memory_item", (
            ("source_table", "TEXT"), ("source_id", "TEXT"), ("embedder_version", "TEXT"),
            ("importance", "REAL DEFAULT 0.5"), ("strength", "REAL DEFAULT 1.0"), ("last_accessed", "TEXT"),
            # 方案 §5.3
            ("canonical_key", "TEXT"), ("memory_type", "TEXT"), ("event_at", "TEXT"),
            ("expires_at", "TEXT"), ("sensitivity", "TEXT DEFAULT 'low'"), ("source_message_id", "TEXT"),
        ))
        await _add_cols("narrative_note", (  # 方案 §5.2
            ("canonical_key", "TEXT"), ("summary", "TEXT"), ("event_at", "TEXT"),
            ("observed_at", "TEXT"), ("source_at", "TEXT"), ("expires_at", "TEXT"),
            ("confidence", "REAL DEFAULT 0.6"), ("sensitivity", "TEXT DEFAULT 'low'"),
            ("status", "TEXT NOT NULL DEFAULT 'active'"), ("superseded_by", "TEXT"), ("metadata_json", "TEXT"),
        ))
        await _add_cols("pending_intake", (  # 方案 §5.1
            ("source_at", "TEXT"), ("timezone", "TEXT"), ("message_id", "TEXT"), ("source_role", "TEXT"),
        ))
        await _add_cols("commitment", (  # 方案 §5.4 + §9 状态机
            ("event_at", "TEXT"), ("completed_at", "TEXT"), ("expires_at", "TEXT"),
            ("source_message_id", "TEXT"), ("canonical_key", "TEXT"), ("sent_at", "TEXT"),
            ("due_window_start", "TEXT"), ("due_window_end", "TEXT"),
            ("confidence", "REAL DEFAULT 0.7"), ("dedupe_key", "TEXT"),
            ("attempts", "INTEGER NOT NULL DEFAULT 0"), ("last_attempt_at", "TEXT"),
        ))
        await _add_cols("profile_fact", (("canonical_key", "TEXT"),))  # 方案 §7
        # 依赖后补列的索引：必须在补列之后建，否则旧库 executescript 阶段会因列不存在而失败
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_mem_source ON memory_item(source_table, source_id)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_narr_canon ON narrative_note(namespace, canonical_key, status)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_commitment_window "
            "ON commitment(namespace, status, due_window_start, due_window_end)"
        )

    @property
    def db(self) -> aiosqlite.Connection:
        assert self._db is not None, "Store 未初始化"
        return self._db

    # ---------- 对话原文（S1） ----------

    async def add_message(self, role: str, content: str, namespace: str = "default") -> None:
        await self.db.execute(
            "INSERT INTO message (id, namespace, role, content, ts) VALUES (?, ?, ?, ?, ?)",
            (uuid.uuid4().hex, namespace, role, content, time.time()),
        )
        await self.db.commit()

    async def recent_messages(self, n: int = 20, namespace: str = "default") -> list[dict]:
        cur = await self.db.execute(
            "SELECT role, content FROM message WHERE namespace = ? ORDER BY ts DESC LIMIT ?",
            (namespace, n),
        )
        rows = await cur.fetchall()
        return [{"role": r, "content": c} for r, c in reversed(rows)]

    async def last_message_ts(self, namespace: str = "default") -> float | None:
        cur = await self.db.execute(
            "SELECT MAX(ts) FROM message WHERE namespace = ?", (namespace,)
        )
        row = await cur.fetchone()
        return row[0] if row and row[0] is not None else None

    # ---------- 画像事实（S2） ----------

    async def add_fact(self, fact: ProfileFact) -> None:
        await self.db.execute(
            f"INSERT INTO profile_fact ({_FACT_COLS}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                fact.id, fact.namespace, fact.category, fact.key, fact.value, fact.confidence,
                fact.sensitivity, fact.source, fact.created_at, fact.updated_at,
                fact.valid_until, fact.superseded_by, fact.canonical_key,
            ),
        )
        await self.db.commit()

    async def active_facts_by_key(
        self, key: str, namespace: str = "default", canonical_key: str | None = None
    ) -> list[ProfileFact]:
        """取同一属性的活跃事实：canonical_key 命中 OR key 命中（把拆散的同实体事实并起来）。"""
        if canonical_key:
            cur = await self.db.execute(
                f"SELECT {_FACT_COLS} FROM profile_fact "
                "WHERE namespace=? AND valid_until IS NULL AND (canonical_key=? OR key=?) "
                "ORDER BY created_at DESC",
                (namespace, canonical_key, key),
            )
        else:
            cur = await self.db.execute(
                f"SELECT {_FACT_COLS} FROM profile_fact "
                "WHERE namespace=? AND key=? AND valid_until IS NULL ORDER BY created_at DESC",
                (namespace, key),
            )
        return [_row_to_fact(r) for r in await cur.fetchall()]

    async def all_active_facts(self, namespace: str = "default") -> list[ProfileFact]:
        cur = await self.db.execute(
            f"SELECT {_FACT_COLS} FROM profile_fact "
            "WHERE namespace=? AND valid_until IS NULL ORDER BY category, updated_at DESC",
            (namespace,),
        )
        return [_row_to_fact(r) for r in await cur.fetchall()]

    async def all_facts(self, namespace: str = "default") -> list[ProfileFact]:
        """All profile facts, including superseded ones, for audit/journal rebuilds."""
        cur = await self.db.execute(
            f"SELECT {_FACT_COLS} FROM profile_fact "
            "WHERE namespace=? ORDER BY updated_at DESC, created_at DESC",
            (namespace,),
        )
        return [_row_to_fact(r) for r in await cur.fetchall()]

    async def touch_fact(self, fact_id: str, confidence: float) -> None:
        await self.db.execute(
            "UPDATE profile_fact SET confidence=?, updated_at=? WHERE id=?",
            (confidence, now_iso(), fact_id),
        )
        await self.db.commit()

    async def supersede_fact(self, old_id: str, new_id: str) -> None:
        ts = now_iso()
        await self.db.execute(
            "UPDATE profile_fact SET valid_until=?, superseded_by=?, updated_at=? WHERE id=?",
            (ts, new_id, ts, old_id),
        )
        await self.db.commit()

    async def get_fact(self, fact_id: str) -> ProfileFact | None:
        cur = await self.db.execute(
            f"SELECT {_FACT_COLS} FROM profile_fact WHERE id=?", (fact_id,)
        )
        r = await cur.fetchone()
        return _row_to_fact(r) if r else None

    async def update_fact_fields(
        self, fact_id: str, value: str | None = None, confidence: float | None = None
    ) -> None:
        """控制台编辑：只动 value/confidence，其它字段不动。"""
        sets, args = [], []
        if value is not None:
            sets.append("value=?"); args.append(value)
        if confidence is not None:
            sets.append("confidence=?"); args.append(confidence)
        if not sets:
            return
        sets.append("updated_at=?"); args.append(now_iso())
        args.append(fact_id)
        await self.db.execute(f"UPDATE profile_fact SET {', '.join(sets)} WHERE id=?", args)
        await self.db.commit()

    async def forget_fact(self, fact_id: str) -> None:
        """被遗忘权：软删（写 valid_until），从此不再注入/检索；history 留痕。"""
        await self.db.execute(
            "UPDATE profile_fact SET valid_until=?, updated_at=? WHERE id=? AND valid_until IS NULL",
            (now_iso(), now_iso(), fact_id),
        )
        await self.db.commit()

    async def clear_fact_validity(self, fact_id: str) -> None:
        """让一条挂起事实重新生效（矛盾解决：选 keep new 时用）。"""
        await self.db.execute(
            "UPDATE profile_fact SET valid_until=NULL, updated_at=? WHERE id=?",
            (now_iso(), fact_id),
        )
        await self.db.commit()

    # ---------- 语义记忆（S3） ----------

    async def add_memory_item(self, item: MemoryItem) -> None:
        await self.db.execute(
            f"INSERT INTO memory_item ({_MEM_COLS}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                item.id, item.namespace, item.content, item.kind, item.source, item.confidence,
                item.created_at, item.valid_until,
                json.dumps(item.keywords, ensure_ascii=False),
                json.dumps(item.embedding) if item.embedding is not None else None,
                item.source_table, item.source_id, item.embedder_version,
                item.importance, item.strength, item.last_accessed,
                item.canonical_key, item.memory_type, item.event_at, item.expires_at,
                item.sensitivity, item.source_message_id,
            ),
        )
        await self.db.commit()

    async def forget_chunks_of(self, source_table: str, source_id: str) -> None:
        """级联：人类可读层（narrative/profile）被遗忘后，对应的向量碎片也失效。"""
        await self.db.execute(
            "UPDATE memory_item SET valid_until=? WHERE source_table=? AND source_id=? AND valid_until IS NULL",
            (now_iso(), source_table, source_id),
        )
        await self.db.commit()

    async def active_memory_items(self, namespace: str = "default") -> list[MemoryItem]:
        cur = await self.db.execute(
            f"SELECT {_MEM_COLS} FROM memory_item "
            "WHERE namespace=? AND valid_until IS NULL ORDER BY created_at DESC",
            (namespace,),
        )
        return [_row_to_mem(r) for r in await cur.fetchall()]

    async def memory_contents(self, namespace: str = "default") -> set[str]:
        cur = await self.db.execute(
            "SELECT content FROM memory_item WHERE namespace=? AND valid_until IS NULL", (namespace,)
        )
        return {r[0] for r in await cur.fetchall()}

    async def get_memory_item(self, item_id: str) -> MemoryItem | None:
        cur = await self.db.execute(f"SELECT {_MEM_COLS} FROM memory_item WHERE id=?", (item_id,))
        r = await cur.fetchone()
        return _row_to_mem(r) if r else None

    async def forget_memory(self, item_id: str) -> None:
        await self.db.execute(
            "UPDATE memory_item SET valid_until=? WHERE id=? AND valid_until IS NULL",
            (now_iso(), item_id),
        )
        await self.db.commit()

    async def reinforce_memories(self, ids: list[str]) -> None:
        """被检索命中 → strength += 1，刷新 last_accessed（MemoryBank 间隔重复：常被想起的衰减变慢）。"""
        if not ids:
            return
        qs = ",".join("?" for _ in ids)
        await self.db.execute(
            f"UPDATE memory_item SET strength = strength + 1, last_accessed = ? WHERE id IN ({qs})",
            (now_iso(), *ids),
        )
        await self.db.commit()

    # ---------- 人类可读层 · 叙事笔记 ----------

    async def add_narrative(
        self, content: str, kind: str = "event", importance: float = 0.5,
        source: str = "consolidator", namespace: str = "default",
        *, canonical_key: str | None = None, summary: str | None = None,
        event_at: str | None = None, observed_at: str | None = None,
        source_at: str | None = None, expires_at: str | None = None,
        confidence: float = 0.6, sensitivity: str = "low",
        status: str = "active", metadata_json: str | None = None,
        created_at: str | None = None,
    ) -> str:
        nid = uuid.uuid4().hex
        await self.db.execute(
            "INSERT INTO narrative_note (id, namespace, kind, content, importance, source, created_at, "
            "canonical_key, summary, event_at, observed_at, source_at, expires_at, confidence, "
            "sensitivity, status, metadata_json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (nid, namespace, kind, content, importance, source, created_at or now_iso(),
             canonical_key, summary, event_at, observed_at, source_at, expires_at, confidence,
             sensitivity, status, metadata_json),
        )
        await self.db.commit()
        return nid

    async def active_narratives_with_vec(self, namespace: str = "default") -> list[dict]:
        """叙事 resolve 用：活跃叙事 + 其向量碎片（按 source_id 关联），供相似度比对。"""
        cur = await self.db.execute(
            "SELECT n.id, n.kind, n.content, n.importance, n.canonical_key, n.event_at, "
            "       m.id, m.embedding, m.keywords "
            "FROM narrative_note n "
            "LEFT JOIN memory_item m "
            "  ON m.source_table='narrative_note' AND m.source_id=n.id AND m.valid_until IS NULL "
            "WHERE n.namespace=? AND n.valid_until IS NULL AND COALESCE(n.status,'active')='active'",
            (namespace,),
        )
        out: list[dict] = []
        for r in await cur.fetchall():
            out.append({
                "id": r[0], "kind": r[1], "content": r[2], "importance": r[3],
                "canonical_key": r[4], "event_at": r[5],
                "mem_id": r[6], "embedding": json.loads(r[7]) if r[7] else None,
                "keywords": json.loads(r[8]) if r[8] else [],
            })
        return out

    async def supersede_narrative(
        self, old_id: str, new_id: str | None, status: str = "superseded"
    ) -> None:
        """旧叙事被合并/更新/过期：软删 + 标 status + 记 superseded_by；级联失效其向量碎片。"""
        await self.db.execute(
            "UPDATE narrative_note SET valid_until=?, status=?, superseded_by=? "
            "WHERE id=? AND valid_until IS NULL",
            (now_iso(), status, new_id, old_id),
        )
        await self.db.commit()
        await self.forget_chunks_of("narrative_note", old_id)

    async def list_narratives(
        self, namespace: str = "default", kind: str | None = None, n: int = 200
    ) -> list[dict]:
        sql = ("SELECT id, kind, content, importance, source, created_at, "
               "canonical_key, event_at, expires_at, COALESCE(status,'active') "
               "FROM narrative_note WHERE namespace=? AND valid_until IS NULL "
               "AND COALESCE(status,'active')='active'")
        args: list = [namespace]
        if kind:
            sql += " AND kind=?"; args.append(kind)
        sql += " ORDER BY created_at DESC LIMIT ?"; args.append(n)
        cur = await self.db.execute(sql, args)
        cols = ("id", "kind", "content", "importance", "source", "created_at",
                "canonical_key", "event_at", "expires_at", "status")
        return [dict(zip(cols, r)) for r in await cur.fetchall()]

    async def narrative_contents(self, namespace: str = "default") -> set[str]:
        cur = await self.db.execute(
            "SELECT content FROM narrative_note WHERE namespace=? AND valid_until IS NULL",
            (namespace,),
        )
        return {r[0] for r in await cur.fetchall()}

    async def get_narrative(self, nid: str) -> dict | None:
        cur = await self.db.execute(
            "SELECT id, kind, content, importance, source, created_at, valid_until "
            "FROM narrative_note WHERE id=?", (nid,)
        )
        r = await cur.fetchone()
        if not r:
            return None
        return dict(zip(("id", "kind", "content", "importance", "source", "created_at", "valid_until"), r))

    async def forget_narrative(self, nid: str) -> None:
        await self.db.execute(
            "UPDATE narrative_note SET valid_until=? WHERE id=? AND valid_until IS NULL",
            (now_iso(), nid),
        )
        await self.db.commit()

    async def top_narratives_for_md(
        self, namespace: str = "default", limit: int = 30
    ) -> list[dict]:
        cur = await self.db.execute(
            "SELECT id, kind, content, importance, created_at FROM narrative_note "
            "WHERE namespace=? AND valid_until IS NULL AND COALESCE(status,'active')='active' "
            "AND kind!='journal' "
            "ORDER BY importance DESC, created_at DESC LIMIT ?",
            (namespace, limit),
        )
        cols = ("id", "kind", "content", "importance", "created_at")
        return [dict(zip(cols, r)) for r in await cur.fetchall()]

    # ---------- PENDING · 写入缓冲 ----------

    async def add_pending_intake(
        self, user_text: str, namespace: str = "default", *,
        source_at: str | None = None, timezone: str | None = None,
        message_id: str | None = None, source_role: str = "user",
    ) -> None:
        await self.db.execute(
            "INSERT INTO pending_intake "
            "(id, namespace, user_text, status, created_at, source_at, timezone, message_id, source_role) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (uuid.uuid4().hex, namespace, user_text, "pending", now_iso(),
             source_at or now_iso(), timezone, message_id, source_role),
        )
        await self.db.commit()

    async def pending_count(self, namespace: str = "default") -> int:
        cur = await self.db.execute(
            "SELECT COUNT(*) FROM pending_intake WHERE namespace=? AND status='pending'",
            (namespace,),
        )
        r = await cur.fetchone()
        return int(r[0]) if r else 0

    async def take_pending(self, namespace: str = "default") -> list[dict]:
        """原子取一批 pending：拿出来同时打标 processing；processed 在 mark_processed 完成。"""
        # 上一次进程如果在 take_pending 之后、mark_pending_processed 之前崩溃，
        # 会留下 processing 卡死项。单进程本地 agent 启动/巩固前可安全重置。
        await self.reset_stale_processing(namespace)
        cur = await self.db.execute(
            "SELECT id, user_text, created_at, source_at, timezone, message_id FROM pending_intake "
            "WHERE namespace=? AND status='pending' ORDER BY created_at",
            (namespace,),
        )
        rows = await cur.fetchall()
        if not rows:
            return []
        ids = [r[0] for r in rows]
        await self.db.execute(
            f"UPDATE pending_intake SET status='processing' WHERE id IN ({','.join('?'*len(ids))})",
            ids,
        )
        await self.db.commit()
        return [
            {"id": r[0], "user_text": r[1], "created_at": r[2],
             "source_at": r[3], "timezone": r[4], "message_id": r[5]}
            for r in rows
        ]

    async def mark_pending_processed(self, ids: list[str]) -> None:
        if not ids:
            return
        await self.db.execute(
            f"UPDATE pending_intake SET status='processed', processed_at=? "
            f"WHERE id IN ({','.join('?'*len(ids))})",
            [now_iso(), *ids],
        )
        await self.db.commit()

    async def reset_stale_processing(self, namespace: str = "default") -> int:
        cur = await self.db.execute(
            "UPDATE pending_intake SET status='pending' WHERE namespace=? AND status='processing'",
            (namespace,),
        )
        await self.db.commit()
        return cur.rowcount

    async def list_pending(self, namespace: str = "default", n: int = 100) -> list[dict]:
        cur = await self.db.execute(
            "SELECT id, user_text, status, created_at, processed_at FROM pending_intake "
            "WHERE namespace=? ORDER BY created_at DESC LIMIT ?",
            (namespace, n),
        )
        cols = ("id", "user_text", "status", "created_at", "processed_at")
        return [dict(zip(cols, r)) for r in await cur.fetchall()]

    # ---------- 情绪 ----------

    async def add_mood(
        self, valence: float | None, arousal: float | None,
        signals: list[str], note: str | None = None, namespace: str = "default",
        ts: str | None = None,
    ) -> str:
        mid = uuid.uuid4().hex
        await self.db.execute(
            "INSERT INTO mood_log (id, namespace, ts, valence, arousal, signals, note) "
            "VALUES (?,?,?,?,?,?,?)",
            (mid, namespace, ts or now_iso(), valence, arousal,
             json.dumps(signals, ensure_ascii=False), note),
        )
        await self.db.commit()
        return mid

    async def recent_mood(self, namespace: str = "default", hours: int = 48) -> dict:
        """最近 hours 小时的情绪聚合（MEM-7 用）：返回均值 valence/arousal 及样本数。"""
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        cur = await self.db.execute(
            "SELECT valence, arousal FROM mood_log WHERE namespace=? AND ts>=?",
            (namespace, cutoff),
        )
        rows = await cur.fetchall()
        vals = [r[0] for r in rows if r[0] is not None]
        aros = [r[1] for r in rows if r[1] is not None]
        return {
            "valence": sum(vals) / len(vals) if vals else None,
            "arousal": sum(aros) / len(aros) if aros else None,
            "n": len(rows),
        }

    async def list_mood(self, namespace: str = "default", n: int = 100) -> list[dict]:
        cur = await self.db.execute(
            "SELECT id, ts, valence, arousal, signals, note FROM mood_log "
            "WHERE namespace=? ORDER BY ts DESC LIMIT ?",
            (namespace, n),
        )
        out = []
        for r in await cur.fetchall():
            out.append({
                "id": r[0], "ts": r[1], "valence": r[2], "arousal": r[3],
                "signals": json.loads(r[4]) if r[4] else [], "note": r[5],
            })
        return out

    # ---------- Trace ----------

    async def add_turn_trace(
        self, namespace: str, user_text: str, stable_prefix: str,
        dynamic_suffix: str, retrieved: list[dict], reply: str, latency_ms: int,
    ) -> None:
        await self.db.execute(
            "INSERT INTO turn_trace (id, namespace, ts, user_text, stable_prefix, "
            "dynamic_suffix, retrieved, reply, latency_ms) VALUES (?,?,?,?,?,?,?,?,?)",
            (uuid.uuid4().hex, namespace, now_iso(), user_text, stable_prefix,
             dynamic_suffix, json.dumps(retrieved, ensure_ascii=False), reply, latency_ms),
        )
        await self.db.commit()

    async def list_turn_trace(self, namespace: str = "default", n: int = 50) -> list[dict]:
        cur = await self.db.execute(
            "SELECT id, ts, user_text, stable_prefix, dynamic_suffix, retrieved, reply, latency_ms "
            "FROM turn_trace WHERE namespace=? ORDER BY ts DESC LIMIT ?",
            (namespace, n),
        )
        out = []
        for r in await cur.fetchall():
            out.append({
                "id": r[0], "ts": r[1], "user_text": r[2],
                "stable_prefix": r[3], "dynamic_suffix": r[4],
                "retrieved": json.loads(r[5]) if r[5] else [],
                "reply": r[6], "latency_ms": r[7],
            })
        return out

    async def add_tick_trace(
        self, namespace: str, sent: bool, reason: str,
        commitment_id: str | None, message: str | None,
    ) -> None:
        await self.db.execute(
            "INSERT INTO tick_trace (id, namespace, ts, sent, reason, commitment_id, message) "
            "VALUES (?,?,?,?,?,?,?)",
            (uuid.uuid4().hex, namespace, now_iso(), 1 if sent else 0, reason, commitment_id, message),
        )
        await self.db.commit()

    async def add_proactive_decision_trace(
        self, *,
        namespace: str = "default",
        trigger_source: str = "interval",
        gate_result: str = "",
        gate_reasons: list[str] | None = None,
        candidate_id: str | None = None,
        candidate_kind: str | None = None,
        llm_notify: bool | None = None,
        llm_outcome: str | None = None,
        llm_summary: str | None = None,
        llm_reason: str | None = None,
        next_check_after: str | None = None,
        post_guard_result: str | None = None,
        final_sent: bool = False,
        message: str | None = None,
    ) -> None:
        await self.db.execute(
            "INSERT INTO proactive_decision_trace ("
            "id, namespace, ts, trigger_source, gate_result, gate_reasons, candidate_id, "
            "candidate_kind, llm_notify, llm_outcome, llm_summary, llm_reason, next_check_after, "
            "post_guard_result, final_sent, message"
            ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                uuid.uuid4().hex, namespace, now_iso(), trigger_source, gate_result,
                json.dumps(gate_reasons or [], ensure_ascii=False), candidate_id, candidate_kind,
                None if llm_notify is None else (1 if llm_notify else 0), llm_outcome,
                llm_summary, llm_reason, next_check_after, post_guard_result,
                1 if final_sent else 0, message,
            ),
        )
        await self.db.commit()

    async def list_tick_trace(self, namespace: str = "default", n: int = 50) -> list[dict]:
        cur = await self.db.execute(
            "SELECT id, ts, sent, reason, commitment_id, message FROM tick_trace "
            "WHERE namespace=? ORDER BY ts DESC LIMIT ?",
            (namespace, n),
        )
        cols = ("id", "ts", "sent", "reason", "commitment_id", "message")
        return [dict(zip(cols, r)) for r in await cur.fetchall()]

    async def add_tool_trace(
        self, namespace: str, step: int, tool_name: str, source: str | None,
        arguments: str, result: str | None, ok: bool, ms: int | None,
    ) -> None:
        await self.db.execute(
            "INSERT INTO tool_trace (id, namespace, ts, step, tool_name, source, arguments, result, ok, ms) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (uuid.uuid4().hex, namespace, now_iso(), step, tool_name, source,
             arguments, (result or "")[:2000], 1 if ok else 0, ms),
        )
        await self.db.commit()

    async def list_tool_trace(self, namespace: str = "default", n: int = 100) -> list[dict]:
        cur = await self.db.execute(
            "SELECT id, ts, step, tool_name, source, arguments, result, ok, ms FROM tool_trace "
            "WHERE namespace=? ORDER BY ts DESC LIMIT ?",
            (namespace, n),
        )
        cols = ("id", "ts", "step", "tool_name", "source", "arguments", "result", "ok", "ms")
        return [dict(zip(cols, r)) for r in await cur.fetchall()]

    # ---------- 审计 / 矛盾 ----------

    async def add_history(
        self, target_table: str, target_id: str, prev_value: str | None,
        new_value: str | None, actor: str, reason: str,
    ) -> None:
        await self.db.execute(
            "INSERT INTO memory_history (id, target_table, target_id, prev_value, new_value, actor, reason, ts) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (uuid.uuid4().hex, target_table, target_id, prev_value, new_value, actor, reason, now_iso()),
        )
        await self.db.commit()

    async def add_contradiction(self, cid: str, namespace: str, new_ref: str, conflict_ref: str) -> None:
        await self.db.execute(
            "INSERT INTO contradiction_item (id, namespace, new_fact_ref, conflicting_fact_ref, status, created_at) "
            "VALUES (?,?,?,?,'pending',?)",
            (cid, namespace, new_ref, conflict_ref, now_iso()),
        )
        await self.db.commit()

    async def list_pending_contradictions(self, namespace: str = "default") -> list[dict]:
        cur = await self.db.execute(
            "SELECT id, new_fact_ref, conflicting_fact_ref, created_at FROM contradiction_item "
            "WHERE namespace=? AND status='pending' ORDER BY created_at",
            (namespace,),
        )
        return [
            {"id": r[0], "new_fact_ref": r[1], "conflicting_fact_ref": r[2], "created_at": r[3]}
            for r in await cur.fetchall()
        ]

    async def set_contradiction_status(self, cid: str, status: str) -> None:
        await self.db.execute("UPDATE contradiction_item SET status=? WHERE id=?", (status, cid))
        await self.db.commit()

    async def recent_history(self, n: int = 200, target_id: str | None = None) -> list[dict]:
        if target_id:
            cur = await self.db.execute(
                "SELECT id, target_table, target_id, prev_value, new_value, actor, reason, ts "
                "FROM memory_history WHERE target_id=? ORDER BY ts DESC LIMIT ?",
                (target_id, n),
            )
        else:
            cur = await self.db.execute(
                "SELECT id, target_table, target_id, prev_value, new_value, actor, reason, ts "
                "FROM memory_history ORDER BY ts DESC LIMIT ?",
                (n,),
            )
        cols = ("id", "target_table", "target_id", "prev_value", "new_value", "actor", "reason", "ts")
        return [dict(zip(cols, r)) for r in await cur.fetchall()]

    # ---------- 确认门 / 审计（S5） ----------

    async def add_pending_action(
        self, aid: str, action_type: str, summary: str, payload: str, namespace: str = "default"
    ) -> None:
        await self.db.execute(
            "INSERT INTO pending_action (id, namespace, action_type, summary, payload, status, created_at) "
            "VALUES (?,?,?,?,?,'pending',?)",
            (aid, namespace, action_type, summary, payload, now_iso()),
        )
        await self.db.commit()

    async def latest_pending_action(self, namespace: str = "default") -> dict | None:
        cur = await self.db.execute(
            "SELECT id, action_type, summary, payload FROM pending_action "
            "WHERE namespace=? AND status='pending' ORDER BY created_at DESC LIMIT 1",
            (namespace,),
        )
        r = await cur.fetchone()
        return {"id": r[0], "action_type": r[1], "summary": r[2], "payload": r[3]} if r else None

    async def list_pending_actions(
        self, namespace: str = "default", status: str | None = None, n: int = 100,
    ) -> list[dict]:
        sql = "SELECT id, action_type, summary, payload, status, created_at FROM pending_action WHERE namespace=?"
        args: list = [namespace]
        if status:
            sql += " AND status=?"; args.append(status)
        sql += " ORDER BY created_at DESC LIMIT ?"; args.append(n)
        cur = await self.db.execute(sql, args)
        cols = ("id", "action_type", "summary", "payload", "status", "created_at")
        return [dict(zip(cols, r)) for r in await cur.fetchall()]

    async def get_pending_action(self, aid: str) -> dict | None:
        cur = await self.db.execute(
            "SELECT id, action_type, summary, payload, status FROM pending_action WHERE id=?", (aid,)
        )
        r = await cur.fetchone()
        if not r:
            return None
        return {"id": r[0], "action_type": r[1], "summary": r[2], "payload": r[3], "status": r[4]}

    async def set_pending_status(self, aid: str, status: str) -> None:
        await self.db.execute("UPDATE pending_action SET status=? WHERE id=?", (status, aid))
        await self.db.commit()

    async def add_audit(self, kind: str, summary: str, actor: str) -> None:
        await self.db.execute(
            "INSERT INTO audit_log (id, kind, summary, actor, ts) VALUES (?,?,?,?,?)",
            (uuid.uuid4().hex, kind, summary, actor, now_iso()),
        )
        await self.db.commit()

    async def recent_audit(self, n: int = 200) -> list[dict]:
        cur = await self.db.execute(
            "SELECT id, kind, summary, actor, ts FROM audit_log ORDER BY ts DESC LIMIT ?", (n,)
        )
        cols = ("id", "kind", "summary", "actor", "ts")
        return [dict(zip(cols, r)) for r in await cur.fetchall()]

    # ---------- 承诺 / 主动（S6） ----------

    async def add_commitment(
        self, kind: str, content: str, due_at: str | None,
        sensitivity: str = "routine", source: str = "inferred", namespace: str = "default",
        *, event_at: str | None = None, expires_at: str | None = None,
        source_message_id: str | None = None, canonical_key: str | None = None,
        due_window_start: str | None = None, due_window_end: str | None = None,
        confidence: float = 0.7, dedupe_key: str | None = None,
        created_at: str | None = None,
    ) -> str:
        cid = uuid.uuid4().hex
        confidence = min(1.0, max(0.0, float(confidence if confidence is not None else 0.7)))
        dedupe_key = dedupe_key or canonical_key
        due_window_start = due_window_start or due_at or event_at
        due_window_end = due_window_end or expires_at
        await self.db.execute(
            "INSERT INTO commitment (id, namespace, kind, sensitivity, source, content, due_at, "
            "status, created_at, event_at, expires_at, source_message_id, canonical_key, "
            "due_window_start, due_window_end, confidence, dedupe_key) "
            "VALUES (?,?,?,?,?,?,?,'open',?,?,?,?,?,?,?,?,?)",
            (cid, namespace, kind, sensitivity, source, content, due_at, created_at or now_iso(),
             event_at, expires_at, source_message_id, canonical_key,
             due_window_start, due_window_end, confidence, dedupe_key),
        )
        await self.db.commit()
        return cid

    async def open_commitment_contents(self, namespace: str = "default") -> set[str]:
        cur = await self.db.execute(
            "SELECT content FROM commitment WHERE namespace=? AND status='open'", (namespace,)
        )
        return {r[0] for r in await cur.fetchall()}

    async def list_all_commitments(
        self, namespace: str = "default", status: str | None = None, n: int = 200
    ) -> list[dict]:
        sql = ("SELECT id, kind, sensitivity, source, content, due_at, status, created_at, "
               "event_at, expires_at, completed_at, canonical_key, due_window_start, "
               "due_window_end, confidence, dedupe_key, attempts, last_attempt_at, sent_at "
               "FROM commitment WHERE namespace=?")
        args: list = [namespace]
        if status:
            sql += " AND status=?"; args.append(status)
        sql += " ORDER BY created_at DESC LIMIT ?"; args.append(n)
        cur = await self.db.execute(sql, args)
        cols = ("id", "kind", "sensitivity", "source", "content", "due_at", "status", "created_at",
                "event_at", "expires_at", "completed_at", "canonical_key", "due_window_start",
                "due_window_end", "confidence", "dedupe_key", "attempts", "last_attempt_at", "sent_at")
        return [dict(zip(cols, r)) for r in await cur.fetchall()]

    async def get_commitment(self, cid: str) -> dict | None:
        cur = await self.db.execute(
            "SELECT id, kind, sensitivity, source, content, due_at, status, created_at, "
            "event_at, expires_at, completed_at, canonical_key, due_window_start, "
            "due_window_end, confidence, dedupe_key, attempts, last_attempt_at, sent_at "
            "FROM commitment WHERE id=?",
            (cid,),
        )
        r = await cur.fetchone()
        if not r:
            return None
        cols = ("id", "kind", "sensitivity", "source", "content", "due_at", "status", "created_at",
                "event_at", "expires_at", "completed_at", "canonical_key", "due_window_start",
                "due_window_end", "confidence", "dedupe_key", "attempts", "last_attempt_at", "sent_at")
        return dict(zip(cols, r))

    async def open_commitments_for_md(self, now: str, namespace: str = "default") -> list[dict]:
        """MEMORY.md「当前开放回路」用：open 且未过期，按 event_at/due_at 升序（最近的在前）。"""
        cur = await self.db.execute(
            "SELECT id, kind, content, due_at, event_at, sensitivity, due_window_start, due_window_end, confidence FROM commitment "
            "WHERE namespace=? AND status='open' AND (expires_at IS NULL OR expires_at>?) "
            "ORDER BY COALESCE(due_window_start, event_at, due_at, '9999') ASC",
            (namespace, now),
        )
        cols = ("id", "kind", "content", "due_at", "event_at", "sensitivity",
                "due_window_start", "due_window_end", "confidence")
        return [dict(zip(cols, r)) for r in await cur.fetchall()]

    async def due_commitments(self, now: str, namespace: str = "default") -> list[dict]:
        """主动引擎用：**有明确时间**（event_at/due_at）且已到期、未过期的 open 承诺。
        无时间的 open_loop 不算到期（避免"找机会再问"立刻被触发，见 opportunistic_commitments）。"""
        now_dt = _parse_time_for_compare(now) or datetime.now(timezone.utc)
        cur = await self.db.execute(
            "SELECT id, kind, content, due_at, event_at, sensitivity, expires_at, "
            "due_window_start, due_window_end, confidence, attempts FROM commitment "
            "WHERE namespace=? AND status='open' ORDER BY created_at",
            (namespace,),
        )
        out: list[dict] = []
        for r in await cur.fetchall():
            expires_at = r[6]
            start = r[7] or r[3] or r[4]
            end = r[8] or expires_at
            if not _not_expired(expires_at, now_dt):
                continue
            if not _is_due(None, start, now_dt):
                continue
            end_dt = _parse_time_for_compare(end)
            if end_dt is not None and end_dt < now_dt:
                continue
            out.append({"id": r[0], "kind": r[1], "content": r[2], "due_at": r[3],
                        "event_at": r[4], "sensitivity": r[5], "expires_at": expires_at,
                        "due_window_start": start, "due_window_end": end,
                        "confidence": r[9] if r[9] is not None else 0.7,
                        "attempts": r[10] if r[10] is not None else 0})
        out.sort(key=lambda c: _parse_time_for_compare(c.get("due_window_start")) or now_dt)
        return out

    async def opportunistic_commitments(self, now: str, namespace: str = "default") -> list[dict]:
        """无明确时间的 open_loop（"找机会再问"）：走低频策略，不随 due 一起立刻触发。"""
        now_dt = _parse_time_for_compare(now) or datetime.now(timezone.utc)
        cur = await self.db.execute(
            "SELECT id, kind, content, due_at, event_at, sensitivity, expires_at, confidence, attempts FROM commitment "
            "WHERE namespace=? AND status='open' "
            "AND event_at IS NULL AND due_at IS NULL "
            "ORDER BY created_at",
            (namespace,),
        )
        out: list[dict] = []
        for r in await cur.fetchall():
            if not _not_expired(r[6], now_dt):
                continue
            out.append({"id": r[0], "kind": r[1], "content": r[2], "due_at": r[3],
                        "event_at": r[4], "sensitivity": r[5], "expires_at": r[6],
                        "confidence": r[7] if r[7] is not None else 0.7,
                        "attempts": r[8] if r[8] is not None else 0})
        return out

    async def open_commitment_signatures(self, namespace: str = "default") -> set[str]:
        """去重签名集（open+sent）：优先 (kind, canonical_key)，退 (kind, event_at)，再退 content。"""
        cur = await self.db.execute(
            "SELECT kind, canonical_key, event_at, content, dedupe_key FROM commitment "
            "WHERE namespace=? AND status IN ('open','sent')",
            (namespace,),
        )
        sigs: set[str] = set()
        for kind, ck, ev, content, dedupe_key in await cur.fetchall():
            sigs.add(dedupe_key or commitment_signature(kind, ck, ev, content))
        return sigs

    async def mark_commitment_sent(self, cid: str) -> None:
        """主动 check-in 发出 → 转 sent（不再 refire、从开放回路移除），记 sent_at。"""
        await self.db.execute(
            "UPDATE commitment SET status='sent', sent_at=?, attempts=attempts+1, last_attempt_at=? WHERE id=?",
            (now_iso(), now_iso(), cid),
        )
        await self.db.commit()

    async def mark_commitment_attempted(self, cid: str) -> None:
        await self.db.execute(
            "UPDATE commitment SET attempts=attempts+1, last_attempt_at=? WHERE id=?",
            (now_iso(), cid),
        )
        await self.db.commit()

    async def list_sent_commitments(self, namespace: str = "default") -> list[dict]:
        cur = await self.db.execute(
            "SELECT id, kind, content, event_at FROM commitment "
            "WHERE namespace=? AND status='sent'",
            (namespace,),
        )
        cols = ("id", "kind", "content", "event_at")
        return [dict(zip(cols, r)) for r in await cur.fetchall()]

    async def expire_stale_sent(self, cutoff: str, namespace: str = "default") -> int:
        """nightly 兜底：sent 超期未回应（sent_at<cutoff）→ done。返回闭合条数。"""
        cur = await self.db.execute(
            "UPDATE commitment SET status='done', completed_at=? "
            "WHERE namespace=? AND status='sent' AND (sent_at IS NULL OR sent_at<?)",
            (now_iso(), namespace, cutoff),
        )
        await self.db.commit()
        return cur.rowcount

    async def stale_sent_commitments(self, cutoff: str, namespace: str = "default") -> list[dict]:
        cur = await self.db.execute(
            "SELECT id, kind, content, event_at, sent_at FROM commitment "
            "WHERE namespace=? AND status='sent' AND (sent_at IS NULL OR sent_at<?)",
            (namespace, cutoff),
        )
        cols = ("id", "kind", "content", "event_at", "sent_at")
        return [dict(zip(cols, r)) for r in await cur.fetchall()]

    async def set_commitment_status(self, cid: str, status: str) -> None:
        # 闭合（done/completed）时记 completed_at，便于事后回沉 event_memory（方案 §9）
        if status in ("done", "completed"):
            await self.db.execute(
                "UPDATE commitment SET status=?, completed_at=? WHERE id=?", (status, now_iso(), cid)
            )
        else:
            await self.db.execute("UPDATE commitment SET status=? WHERE id=?", (status, cid))
        await self.db.commit()

    async def add_proactive_candidate(
        self, cid: str, commitment_id: str, content: str, score: float, reason: str,
        status: str, namespace: str = "default",
    ) -> None:
        await self.db.execute(
            "INSERT INTO proactive_candidate (id, namespace, commitment_id, content, score, reason, status, created_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (cid, namespace, commitment_id, content, score, reason, status, now_iso()),
        )
        await self.db.commit()

    async def list_proactive(self, namespace: str = "default", n: int = 100) -> list[dict]:
        cur = await self.db.execute(
            "SELECT id, commitment_id, content, score, reason, status, created_at "
            "FROM proactive_candidate WHERE namespace=? ORDER BY created_at DESC LIMIT ?",
            (namespace, n),
        )
        cols = ("id", "commitment_id", "content", "score", "reason", "status", "created_at")
        return [dict(zip(cols, r)) for r in await cur.fetchall()]

    # ---------- 精确提醒（独立于 commitment / proactive heartbeat） ----------

    async def add_reminder_job(
        self, *,
        title: str,
        message: str,
        trigger_type: str,
        trigger_spec: str,
        due_at_utc: str | None,
        timezone: str,
        original_time_text: str | None,
        delivery_channel: str,
        delivery_target: str,
        namespace: str = "default",
        kind: str = "one_shot",
        source_message_id: str | None = None,
        tool_call_id: str | None = None,
        idempotency_key: str | None = None,
        misfire_grace_seconds: int = 86400,
        policy_json: str = "{}",
        metadata_json: str = "{}",
    ) -> str:
        if idempotency_key:
            cur = await self.db.execute(
                "SELECT id FROM reminder_job WHERE idempotency_key=?", (idempotency_key,)
            )
            row = await cur.fetchone()
            if row:
                return row[0]
        rid = uuid.uuid4().hex
        ts = now_iso()
        await self.db.execute(
            "INSERT INTO reminder_job ("
            "id, namespace, kind, title, message, trigger_type, trigger_spec, due_at_utc, "
            "timezone, original_time_text, delivery_channel, delivery_target, status, "
            "source_message_id, tool_call_id, idempotency_key, created_at, updated_at, "
            "misfire_grace_seconds, policy_json, metadata_json"
            ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                rid, namespace, kind, title, message, trigger_type, trigger_spec, due_at_utc,
                timezone, original_time_text, delivery_channel, delivery_target, "scheduled",
                source_message_id, tool_call_id, idempotency_key, ts, ts,
                misfire_grace_seconds, policy_json, metadata_json,
            ),
        )
        await self.db.commit()
        return rid

    def _reminder_cols(self) -> tuple[str, tuple[str, ...]]:
        cols = (
            "id", "namespace", "kind", "title", "message", "trigger_type", "trigger_spec",
            "due_at_utc", "timezone", "original_time_text", "delivery_channel", "delivery_target",
            "status", "source_message_id", "tool_call_id", "idempotency_key", "created_at",
            "updated_at", "fired_at", "sent_at", "acknowledged_at", "retry_count", "max_retries",
            "last_error", "misfire_grace_seconds", "policy_json", "metadata_json",
        )
        return ", ".join(cols), cols

    async def get_reminder_job(self, rid: str) -> dict | None:
        select_cols, cols = self._reminder_cols()
        cur = await self.db.execute(f"SELECT {select_cols} FROM reminder_job WHERE id=?", (rid,))
        row = await cur.fetchone()
        return dict(zip(cols, row)) if row else None

    async def pending_reminder_jobs(self, namespace: str = "default") -> list[dict]:
        select_cols, cols = self._reminder_cols()
        cur = await self.db.execute(
            f"SELECT {select_cols} FROM reminder_job "
            "WHERE namespace=? AND status IN ('pending','scheduled','failed') "
            "AND retry_count < max_retries ORDER BY due_at_utc ASC",
            (namespace,),
        )
        return [dict(zip(cols, r)) for r in await cur.fetchall()]

    async def list_reminder_jobs(
        self, namespace: str = "default", status: str | None = None, n: int = 200
    ) -> list[dict]:
        select_cols, cols = self._reminder_cols()
        args: list = [namespace]
        sql = f"SELECT {select_cols} FROM reminder_job WHERE namespace=?"
        if status:
            sql += " AND status=?"; args.append(status)
        sql += " ORDER BY COALESCE(due_at_utc, created_at) ASC LIMIT ?"; args.append(n)
        cur = await self.db.execute(sql, args)
        return [dict(zip(cols, r)) for r in await cur.fetchall()]

    async def claim_reminder_job(self, rid: str) -> dict | None:
        """scheduled/failed -> firing；返回 claim 后的 job。重复触发会拿不到。"""
        ts = now_iso()
        cur = await self.db.execute(
            "UPDATE reminder_job SET status='firing', fired_at=?, updated_at=? "
            "WHERE id=? AND status IN ('pending','scheduled','failed') RETURNING id",
            (ts, ts, rid),
        )
        row = await cur.fetchone()
        await self.db.commit()
        return await self.get_reminder_job(rid) if row else None

    async def mark_reminder_sent(self, rid: str) -> None:
        ts = now_iso()
        await self.db.execute(
            "UPDATE reminder_job SET status='sent', sent_at=?, updated_at=?, last_error=NULL WHERE id=?",
            (ts, ts, rid),
        )
        await self.db.commit()

    async def mark_reminder_failed(self, rid: str, error: str) -> None:
        await self.db.execute(
            "UPDATE reminder_job SET status='failed', retry_count=retry_count+1, "
            "last_error=?, updated_at=? WHERE id=?",
            (error[:1000], now_iso(), rid),
        )
        await self.db.commit()

    async def mark_reminder_expired(self, rid: str, reason: str = "") -> None:
        await self.db.execute(
            "UPDATE reminder_job SET status='expired', last_error=?, updated_at=? WHERE id=?",
            (reason[:1000], now_iso(), rid),
        )
        await self.db.commit()

    async def cancel_reminder_job(self, rid: str) -> bool:
        cur = await self.db.execute(
            "UPDATE reminder_job SET status='cancelled', updated_at=? "
            "WHERE id=? AND status NOT IN ('sent','cancelled','expired')",
            (now_iso(), rid),
        )
        await self.db.commit()
        return cur.rowcount > 0

    async def add_delivery_log(
        self, *,
        source_type: str,
        source_id: str,
        channel: str,
        target: str,
        payload: str,
        status: str,
        namespace: str = "default",
        error: str | None = None,
    ) -> None:
        ts = now_iso()
        await self.db.execute(
            "INSERT INTO delivery_log (id, namespace, source_type, source_id, channel, target, "
            "payload, status, attempted_at, sent_at, error) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                uuid.uuid4().hex, namespace, source_type, source_id, channel, target,
                payload, status, ts, ts if status == "sent" else None, error,
            ),
        )
        await self.db.commit()

    async def budget_used_today(self, date: str, base_quota: int) -> tuple[int, int]:
        """返回 (今日已用, 配额)。不存在则建。"""
        cur = await self.db.execute(
            "SELECT base_quota, used FROM frequency_budget WHERE date=?", (date,)
        )
        r = await cur.fetchone()
        if r is None:
            await self.db.execute(
                "INSERT INTO frequency_budget (date, base_quota, used) VALUES (?,?,0)", (date, base_quota)
            )
            await self.db.commit()
            return 0, base_quota
        return r[1], r[0]

    async def increment_budget(self, date: str) -> None:
        await self.db.execute("UPDATE frequency_budget SET used = used + 1 WHERE date=?", (date,))
        await self.db.commit()

    # ---------- kv ----------

    async def kv_set(self, key: str, value: str) -> None:
        await self.db.execute(
            "INSERT INTO kv (k, v) VALUES (?,?) ON CONFLICT(k) DO UPDATE SET v=?", (key, value, value)
        )
        await self.db.commit()

    async def kv_get(self, key: str) -> str | None:
        cur = await self.db.execute("SELECT v FROM kv WHERE k=?", (key,))
        r = await cur.fetchone()
        return r[0] if r else None

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
