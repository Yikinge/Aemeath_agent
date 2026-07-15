"""Consolidator —— 记忆巩固管线（akashic Deep Dream / letta sleeptime 思想）。

ingest 只 buffer 到 pending_intake；本模块把缓冲批量跑成带生命周期的结构化记忆：

  normalize 时间锚点
  → 画像 extract→resolve（含实体合并）
  → route 分类：event_memory 进叙事(resolve 去重/合并)，未来计划进 commitment，情绪进 mood，其余丢弃
  → 重写分区式 MEMORY.md

把"高频改写"集中到 consolidate 周期里，让 SOUL.md + MEMORY.md 的 system prompt prefix
保持稳定，命中下游 provider 的 prompt cache（Anthropic 5min TTL ≈ 90% 折扣）。
"""

from __future__ import annotations

from pathlib import Path

from agent.gateway.router import LLMRouter
from agent.memory.extract import (
    _MEM_MIN_IMPORTANCE, extract_mood, extract_profile_facts, reflect_insights, route_memories,
)
from agent.memory.journal import DailyJournal
from agent.memory.models import (
    FactCandidate, IngestResult, MemoryItem, ProfileFact, new_id, now_iso,
)
from agent.memory.normalize import absolutize, normalize_time, time_anchor_hint
from agent.memory.retrieve import _age_days, _cosine
from agent.memory.store import Store, commitment_signature
from agent.memory.types import EVENT_MEMORY, RESOLVE_VERDICTS
from agent.proactive.commitments import extract_commitments
from agent.proactive.policy import candidate_is_timely

_PRUNE_MAX_STRENGTH = 1.0   # 从没被强化过（strength 仍为初始值）
_PRUNE_MAX_IMPORTANCE = 0.4  # 不重要
_PRUNE_MIN_AGE_DAYS = 60.0   # 且很久没被用到 → 才剪（三条都满足，保守）

_CONTRADICTION_CONF = 0.7   # 旧事实置信度 ≥ 此值且与新值冲突 → 进矛盾队列

# 画像分区（MEMORY.md）：稳定画像 vs 互动偏好
_PROFILE_CATS = {"bio", "value", "social", "entity", "routine", "other"}
_PREF_CATS = {"preference", "taste", "taboo"}


def _norm(s: str) -> str:
    return s.strip().lower()


def _date_part(s: str | None) -> str | None:
    """从 event_at（date 或 datetime）取日期部分用于展示。"""
    if not s:
        return None
    return s[:10]


class Consolidator:
    def __init__(
        self, store: Store, router: LLMRouter, md_path: str, *,
        timezone: str = "Asia/Shanghai", similarity_threshold: float = 0.82,
        core_max_commitments: int = 5, recent_mood_days: int = 14,
    ) -> None:
        self.store = store
        self.router = router
        self.md_path = Path(md_path)
        self.md_path.parent.mkdir(parents=True, exist_ok=True)
        # 给用户看的"db 全量台账"（不注入模型），跟 MEMORY.md 同目录、随它一起刷新
        self.snapshot_path = self.md_path.parent / "agent_db.md"
        self.journal = DailyJournal(self.md_path.parent / "journal", timezone_name=timezone)
        self.timezone = timezone
        self.similarity_threshold = similarity_threshold
        self.core_max_commitments = core_max_commitments
        self.recent_mood_days = recent_mood_days

    # ---------- 主入口 ----------

    async def consolidate(self, namespace: str = "default") -> IngestResult:
        """把 pending_intake 跑成结构化记忆；返回本批的统计。"""
        batch = await self.store.take_pending(namespace)
        result = IngestResult()
        if not batch:
            return result

        try:
            # 逐条处理：每条消息都用自己的 source_at/message_id，避免断线积压或跨天 pending 造成日报时间漂移。
            for item in batch:
                src = item.get("source_at") or now_iso()
                tz = item.get("timezone") or self.timezone
                message_id = item.get("message_id")
                msgs = [{"role": "user", "content": item["user_text"]}]
                for cand in await extract_profile_facts(self.router, msgs):
                    await self._resolve(cand, namespace, result, observed_at=src)
                hint = time_anchor_hint(item["user_text"], src, tz)
                await self._ingest_routed(msgs, hint, src, tz, namespace, result, message_id=message_id)
                await self._ingest_commitments(msgs, hint, src, tz, namespace, result, message_id=message_id)
                await self._ingest_mood(msgs, namespace, source_at=src)

            # 5) 重写 MEMORY.md：有结构化变更时刷新；首次运行即使没有抽到长期记忆，
            # 也生成基础骨架，方便用户在 Docker volume 里直接看到文件。
            if result.added or result.updated or result.memories or result.commitments or not self.md_path.exists():
                await self.refresh_memory_md(namespace)
        finally:
            await self.store.mark_pending_processed([b["id"] for b in batch])

        return result

    # ---------- 画像 resolve（含实体合并） ----------

    async def _resolve(
        self, cand: FactCandidate, namespace: str, result: IngestResult, *,
        observed_at: str | None = None,
    ) -> None:
        existing = await self.store.active_facts_by_key(cand.key, namespace, cand.canonical_key)

        if not existing:
            fact = self._make_fact(cand, namespace, observed_at=observed_at)
            await self.store.add_fact(fact)
            await self.store.add_history("profile_fact", fact.id, None, fact.value, "agent", "ADD")
            self.journal.append(self.journal.profile_entry(fact, "ADD"))
            result.added.append(fact)
            return

        cur = existing[0]
        if _norm(cur.value) == _norm(cand.value):
            await self.store.touch_fact(cur.id, max(cur.confidence, cand.confidence))
            result.noop.append(cur.key)
            return

        # 实体补充信息 → MERGE 合并，不丢旧值（修 #3：煤球 + 橘猫 → "煤球（橘猫）"）
        same_entity = cand.category == "entity" and cand.canonical_key and cur.canonical_key == cand.canonical_key
        if same_entity:
            if _norm(cand.value) in _norm(cur.value):   # 新值已被旧值包含 → 不重复
                await self.store.touch_fact(cur.id, max(cur.confidence, cand.confidence))
                result.noop.append(cur.key)
                return
            merged = await self._merge_values(cur.key, cur.value, cand.value)
            fact = ProfileFact(
                category=cand.category, key=cur.key, value=merged,
                confidence=max(cur.confidence, cand.confidence), source="msg",
                namespace=namespace, canonical_key=cand.canonical_key or cur.canonical_key,
                created_at=observed_at or now_iso(), updated_at=observed_at or now_iso(),
            )
            await self.store.add_fact(fact)
            await self.store.supersede_fact(cur.id, fact.id)
            await self.store.add_history("profile_fact", cur.id, cur.value, merged, "agent", "MERGE")
            self.journal.append(self.journal.profile_entry(fact, "MERGE", previous=cur.value))
            result.updated.append(fact)
            return

        verdict = await self._judge(cand.key, cur.value, cand.value)

        if verdict == "SAME":
            await self.store.touch_fact(cur.id, max(cur.confidence, cand.confidence))
            result.noop.append(cur.key)
            return

        if verdict == "CONFLICT" and cur.confidence >= _CONTRADICTION_CONF:
            pending = self._make_fact(cand, namespace, observed_at=observed_at)
            pending.valid_until = pending.created_at  # 标记为「未生效」
            await self.store.add_fact(pending)
            await self.store.add_contradiction(new_id(), namespace, pending.id, cur.id)
            await self.store.add_history(
                "profile_fact", cur.id, cur.value, cand.value, "agent", "CONTRADICTION"
            )
            self.journal.append(self.journal.maintenance_entry(
                f"profile-contradiction:{pending.id}",
                f"发现画像矛盾，暂缓生效：{cand.key} 旧值「{cur.value}」/ 新值「{cand.value}」。",
                occurred_at=pending.created_at,
            ))
            result.contradictions.append(cur.key)
            return

        fact = self._make_fact(cand, namespace, observed_at=observed_at)
        await self.store.add_fact(fact)
        await self.store.supersede_fact(cur.id, fact.id)
        await self.store.add_history("profile_fact", cur.id, cur.value, fact.value, "agent", "UPDATE")
        self.journal.append(self.journal.profile_entry(fact, "UPDATE", previous=cur.value))
        result.updated.append(fact)

    async def _judge(self, key: str, old: str, new: str) -> str:
        if not self.router.live("fast"):
            return "CONFLICT"
        prompt = (
            f"同一属性「{key}」出现了两个值。\n旧值：{old}\n新值：{new}\n"
            "判断关系，只回一个词：\n"
            "SAME = 实质相同（同义，或一个是另一个更具体/简略的表述）\n"
            "UPDATE = 用户情况发生了变化，新值应取代旧值\n"
            "CONFLICT = 直接矛盾且无法判断哪个为真\n"
            "只输出 SAME / UPDATE / CONFLICT 之一。"
        )
        raw = (await self.router.complete(
            [{"role": "user", "content": prompt}], task="fast"
        )).upper()
        for v in ("SAME", "UPDATE", "CONFLICT"):
            if v in raw:
                return v
        return "CONFLICT"

    async def _merge_values(self, key: str, old: str, new: str) -> str:
        """合并同一实体的两条属性值（如 名字 + 品种 → "煤球（橘猫）"）。离线兜底用括号拼。"""
        if not self.router.live("fast"):
            return f"{old}（{new}）"
        prompt = (
            f"同一实体的属性「{key}」有两条信息：{old}；{new}。\n"
            "合并成一条简洁、不重复、信息完整的中文值（如把名字和品种合到一起）。只输出合并后的值。"
        )
        raw = (await self.router.complete([{"role": "user", "content": prompt}], task="fast")).strip()
        return raw or f"{old}（{new}）"

    @staticmethod
    def _make_fact(cand: FactCandidate, namespace: str, observed_at: str | None = None) -> ProfileFact:
        ts = observed_at or now_iso()
        return ProfileFact(
            category=cand.category, key=cand.key, value=cand.value,
            confidence=cand.confidence, source="msg", namespace=namespace,
            canonical_key=cand.canonical_key, created_at=ts, updated_at=ts,
        )

    # ---------- route → 叙事（resolve 去重/合并，方案 §6.3） ----------

    async def _ingest_routed(
        self, new_messages: list[dict], time_hint: str, anchor_src: str, anchor_tz: str,
        namespace: str, result: IngestResult, *, message_id: str | None = None,
    ) -> None:
        existing = await self.store.narrative_contents(namespace)
        cands = await route_memories(self.router, new_messages, time_hint, existing)
        for cand in cands:
            if cand.memory_type != EVENT_MEMORY:
                continue   # 只有 event_memory 配进叙事层；实体/情绪/计划各有专职处理
            if cand.importance < _MEM_MIN_IMPORTANCE:
                continue   # 价值闸门
            cand.content = absolutize(cand.content, anchor_src, anchor_tz)  # 兜底防漂移
            # event_at 以本地 normalize 为权威（LLM 给的只作兜底）
            nt = normalize_time(cand.content, anchor_src, anchor_tz)
            event_at = nt.primary_event_at or cand.event_at
            await self._resolve_narrative(cand, event_at, namespace, result, source_at=anchor_src, message_id=message_id)

    async def _resolve_narrative(
        self, cand, event_at: str | None, namespace: str, result: IngestResult, *,
        source_at: str | None = None, message_id: str | None = None,
    ) -> None:
        vec = (await self.router.embed([cand.content]))[0]
        existing = await self.store.active_narratives_with_vec(namespace)

        match = None
        if cand.canonical_key:
            match = next((e for e in existing if e["canonical_key"] == cand.canonical_key), None)
        if match is None:
            best, best_sim = None, 0.0
            for e in existing:
                if not e["embedding"]:
                    continue
                sim = _cosine(vec, e["embedding"])
                if sim > best_sim:
                    best, best_sim = e, sim
            if best is not None and best_sim >= self.similarity_threshold:
                match = best

        if match is None:
            await self._store_narrative(
                cand, event_at, vec, namespace, result, reason="ADD",
                source_at=source_at, message_id=message_id,
            )
            return

        verdict = await self._judge_narrative(match["content"], cand.content)

        if verdict == "SAME":
            if match["mem_id"]:
                await self.store.reinforce_memories([match["mem_id"]])
            await self.store.add_history("narrative_note", match["id"], match["content"], cand.content, "agent", "SAME")
            return
        if verdict == "EXPIRE":
            await self.store.supersede_narrative(match["id"], None, "expired")
            await self.store.add_history("narrative_note", match["id"], match["content"], None, "agent", "EXPIRE")
            await self._store_narrative(
                cand, event_at, vec, namespace, result, reason="ADD",
                source_at=source_at, message_id=message_id,
            )
            return

        # MERGE / UPDATE / NEW
        if verdict == "MERGE":
            merged = await self._merge_content(match["content"], cand.content)
            cand.content = merged
            cand.canonical_key = cand.canonical_key or match["canonical_key"]
            vec = (await self.router.embed([cand.content]))[0]
        if verdict in ("MERGE", "UPDATE"):
            nid = await self._store_narrative(
                cand, event_at, vec, namespace, result, reason=verdict,
                source_at=source_at, message_id=message_id,
            )
            await self.store.supersede_narrative(match["id"], nid, "superseded")
            await self.store.add_history("narrative_note", match["id"], match["content"], cand.content, "agent", verdict)
        else:  # NEW
            await self._store_narrative(
                cand, event_at, vec, namespace, result, reason="ADD",
                source_at=source_at, message_id=message_id,
            )

    async def _store_narrative(
        self, cand, event_at: str | None, vec, namespace: str, result: IngestResult, *,
        reason: str, source_at: str | None = None, message_id: str | None = None,
    ) -> str:
        embedder_v = self.router.embed_model or "fallback"
        sensitivity = getattr(cand, "sensitivity", "low") or "low"
        observed_at = source_at or now_iso()
        nid = await self.store.add_narrative(
            cand.content, kind="event", importance=cand.importance, source="consolidator",
            namespace=namespace, canonical_key=cand.canonical_key, event_at=event_at,
            observed_at=observed_at, source_at=source_at, confidence=cand.confidence,
            expires_at=cand.expires_at, sensitivity=sensitivity, created_at=observed_at,
        )
        await self.store.add_history("narrative_note", nid, None, cand.content, "agent", reason)
        chunk = MemoryItem(
            content=cand.content, kind="narrative", keywords=cand.keywords,
            embedding=vec, source="msg", namespace=namespace, importance=cand.importance,
            source_table="narrative_note", source_id=nid, embedder_version=embedder_v,
            canonical_key=cand.canonical_key, memory_type=EVENT_MEMORY,
            event_at=event_at, expires_at=cand.expires_at, sensitivity=sensitivity,
            source_message_id=message_id, created_at=observed_at,
        )
        await self.store.add_memory_item(chunk)
        self.journal.append(self.journal.narrative_entry(
            nid, cand.content, reason=reason, event_at=event_at, created_at=observed_at,
        ))
        result.memories.append(chunk)
        return nid

    async def _judge_narrative(self, old: str, new: str) -> str:
        """叙事 resolve 判定：SAME/MERGE/UPDATE/EXPIRE/NEW（方案 §6.3）。"""
        if not self.router.live("fast"):
            return "NEW"   # 离线兜底：不擅自合并（宁可暂时重复，也不丢信息）
        prompt = (
            "下面两条是关于同一个人的记忆，判断它们的关系，只回一个词：\n"
            f"已有：{old}\n新的：{new}\n"
            "SAME = 同一件事/同一信息，实质重复\n"
            "MERGE = 同一件事/同一实体的补充，应合并成更完整的一条\n"
            "UPDATE = 新信息使旧的过时，应替代\n"
            "EXPIRE = 旧的是已结束的短期状态/已过的计划\n"
            "NEW = 不同的事，各自保留\n"
            "只输出 SAME / MERGE / UPDATE / EXPIRE / NEW 之一。"
        )
        raw = (await self.router.complete([{"role": "user", "content": prompt}], task="fast")).upper()
        for v in RESOLVE_VERDICTS:
            if v in raw:
                return v
        return "NEW"

    async def _merge_content(self, old: str, new: str) -> str:
        if not self.router.live("fast"):
            return new
        prompt = (
            "把下面两条关于同一件事/同一实体的记忆合并成一句更完整、不重复的中文陈述"
            "（保留必要的绝对日期）：\n"
            f"1）{old}\n2）{new}\n只输出合并后的一句话。"
        )
        raw = (await self.router.complete([{"role": "user", "content": prompt}], task="fast")).strip()
        return raw or new

    # ---------- 承诺 / 情绪 ----------

    async def _ingest_commitments(
        self, new_messages: list[dict], time_hint: str, anchor_src: str, anchor_tz: str,
        namespace: str, result: IngestResult, *, message_id: str | None = None,
    ) -> None:
        sigs = await self.store.open_commitment_signatures(namespace)
        cands = await extract_commitments(self.router, new_messages, None, time_hint)
        # 先把每条的绝对 event_at 解析出来（normalize 为权威、absolutize 兜底防漂移）
        prepared: list[tuple] = []
        for c in cands:
            c.content = absolutize(c.content, anchor_src, anchor_tz)
            nt = normalize_time(c.content, anchor_src, anchor_tz)
            prepared.append((c, nt.primary_event_at or c.event_at))

        # 防线：同一批里若既有"带时间承诺"又有"无时间 open_loop"，且二者语义相近（同一件事），
        # 丢掉那个 open_loop——否则它会被 opportunistic 反复翻出来、和正点的 event_check_in 撞车。
        # 跨批的冗余靠抽取 prompt 那条「一件事只抽一条」兜（这里只处理最常见的同批同句重复）。
        # 只有确实可能撞车时才 embed，省调用。
        has_undated_ol = any(c.kind == "open_loop" and not ev and not c.due_at for c, ev in prepared)
        has_dated = any(ev or c.due_at for c, ev in prepared)
        vecs = (await self.router.embed([c.content for c, _ in prepared])
                if has_undated_ol and has_dated else [])
        dated_idx = [i for i, (c, ev) in enumerate(prepared) if ev or c.due_at]

        for i, (c, event_at) in enumerate(prepared):
            if (c.kind == "open_loop" and not event_at and not c.due_at and vecs
                    and any(_cosine(vecs[i], vecs[j]) >= self.similarity_threshold for j in dated_idx)):
                continue  # 已被同批带时间承诺覆盖 → 丢弃，不入库
            sig = c.dedupe_key or commitment_signature(c.kind, c.canonical_key, event_at, c.content)
            if sig in sigs:                 # 语义去重：同一件事不同措辞算一条
                continue
            sigs.add(sig)
            cid = await self.store.add_commitment(
                c.kind, c.content, c.due_at or event_at, c.sensitivity, "inferred", namespace,
                event_at=event_at, expires_at=c.expires_at, canonical_key=c.canonical_key,
                due_window_start=c.due_window_start or c.due_at or event_at,
                due_window_end=c.due_window_end or c.expires_at,
                confidence=c.confidence, dedupe_key=c.dedupe_key or sig,
                source_message_id=message_id, created_at=anchor_src,
            )
            self.journal.append(self.journal.commitment_entry(
                cid, c.content, status="open", occurred_at=anchor_src,
            ))
            result.commitments.append(c.content)

    async def _ingest_mood(
        self, new_messages: list[dict], namespace: str, *, source_at: str | None = None,
    ) -> None:
        mood = await extract_mood(self.router, new_messages)
        if mood is None:
            return
        mid = await self.store.add_mood(
            valence=mood["valence"], arousal=mood["arousal"],
            signals=mood["signals"], note=mood["note"], namespace=namespace,
            ts=source_at,
        )
        if mood.get("note"):
            self.journal.append(self.journal.mood_entry(mid, mood, occurred_at=source_at))

    # ---------- MEMORY.md 重写（分区式核心摘要，方案 §7） ----------

    async def refresh_memory_md(self, namespace: str = "default") -> str:
        facts = await self.store.all_active_facts(namespace)
        profile = [f for f in facts if f.category in _PROFILE_CATS]
        prefs = [f for f in facts if f.category in _PREF_CATS]
        now = now_iso()
        commits = [
            c for c in await self.store.open_commitments_for_md(now, namespace)
            if candidate_is_timely(c, now)
        ][: self.core_max_commitments]
        insights = (await self.store.list_narratives(namespace, kind="insight"))[:3]
        mood_line = await self._recent_mood_line(namespace)

        lines: list[str] = ["# MEMORY", ""]

        lines.append("## 稳定画像")
        lines.extend([f"- {f.key}：{f.value}" for f in profile] or ["- 暂无"])
        lines.append("")

        lines.append("## 互动偏好")
        lines.extend([f"- {f.key}：{f.value}" for f in prefs] or ["- 暂无"])
        lines.append("")

        lines.append("## 当前开放回路")
        if commits:
            for c in commits:
                d = _date_part(c.get("event_at")) or _date_part(c.get("due_at"))
                prefix = f"{d} " if d and (not d or d not in c["content"]) else ""
                lines.append(f"- {prefix}{c['content']}")
        else:
            lines.append("- 暂无")
        lines.append("")

        lines.append("## 近期状态")
        lines.append(f"- {mood_line}" if mood_line else "- 暂无")
        lines.append("")

        lines.append("## 长期洞察")
        lines.extend([f"- {n['content']}" for n in insights] or ["- 暂无"])
        lines.append("")

        lines.append(f"_(更新于 {now_iso()})_")
        text = "\n".join(lines).rstrip() + "\n"
        tmp = self.md_path.with_suffix(self.md_path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(self.md_path)  # 原子写
        await self.write_db_snapshot(namespace)  # 顺手刷新给用户看的全量台账
        return text

    async def write_db_snapshot(self, namespace: str = "default") -> str:
        """把 agent.db 里所有人类可读的记忆导出成一份**全量台账** md（给用户看，不注入模型）。

        区别于 MEMORY.md（给模型的精简摘要、只取 top-N）：这份列出当前实际存了哪些
        画像/叙事/承诺/情绪/向量碎片，让你随时知道"写进 db 的到底是什么"。
        随 MEMORY.md 一起刷新（每次记忆变更）；也可 `python -m agent.cli --snapshot` 手动生成。
        """
        facts = await self.store.all_active_facts(namespace)
        narrs = await self.store.list_narratives(namespace)
        commits = await self.store.list_all_commitments(namespace)
        moods = await self.store.list_mood(namespace, n=30)
        chunks = await self.store.active_memory_items(namespace)
        pending = await self.store.pending_count(namespace)

        L: list[str] = [
            "# agent.db 记忆全量快照",
            "",
            "> db 里**实际存了什么**的完整台账（给你看的，不注入模型）。",
            "> 模型看到的精简版在 MEMORY.md；这份每次记忆变更后自动刷新。",
            "",
            f"## 画像 profile_fact（{len(facts)}）",
        ]
        if facts:
            for f in facts:
                ck = f"，键 {f.canonical_key}" if f.canonical_key else ""
                L.append(f"- [{f.category}] {f.key} = {f.value}（置信 {f.confidence:.2f}{ck}）")
        else:
            L.append("- 暂无")

        events = [n for n in narrs if n["kind"] == "event"]
        insights = [n for n in narrs if n["kind"] == "insight"]
        rest = [n for n in narrs if n["kind"] not in ("event", "insight")]
        L += ["", f"## 叙事 narrative_note（{len(narrs)}）", f"### 事件 event（{len(events)}）"]
        L += [f"- {n['content']}" + (f"  @{n['event_at'][:10]}" if n.get("event_at") else "")
              for n in events] or ["- 暂无"]
        L += [f"### 洞察 insight（{len(insights)}）"]
        L += ([f"- {n['content']}" for n in insights] or ["- 暂无"])
        if rest:
            L += [f"### 其他（{len(rest)}）"] + [f"- ({n['kind']}) {n['content']}" for n in rest]

        L += ["", f"## 承诺 commitment（{len(commits)}）"]
        if commits:
            for c in commits:
                when = c.get("event_at") or c.get("due_at") or ""
                when = f"  @{when[:10]}" if when else ""
                L.append(f"- [{c['status']}] ({c['kind']}) {c['content']}{when}")
        else:
            L.append("- 暂无")

        L += ["", f"## 情绪 mood_log（近 {len(moods)} 条）"]
        if moods:
            for m in moods:
                v = f"{m['valence']:+.2f}" if m["valence"] is not None else "—"
                sig = "、".join(m["signals"]) if m["signals"] else ""
                L.append(f"- {m['ts'][:16]}  v={v}  {sig}  {m['note'] or ''}".rstrip())
        else:
            L.append("- 暂无")

        L += ["", f"## 向量碎片 memory_item（{len(chunks)}，语义召回用）"]
        if chunks:
            for it in chunks:
                tag = it.memory_type or it.kind
                L.append(f"- [{tag}] {it.content}（importance {it.importance:.2f}，strength {it.strength:.1f}）")
        else:
            L.append("- 暂无")

        L += ["", "## 其他", f"- 待巩固 pending_intake：{pending} 条",
              "", f"_(更新于 {now_iso()})_"]

        text = "\n".join(L).rstrip() + "\n"
        tmp = self.snapshot_path.with_suffix(self.snapshot_path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(self.snapshot_path)  # 原子写
        return text

    async def _recent_mood_line(self, namespace: str) -> str | None:
        """近期状态：最近 recent_mood_days 内最新一条带 note 的情绪，做一句摘要（不堆原句）。"""
        moods = await self.store.list_mood(namespace, n=50)
        cutoff_days = self.recent_mood_days
        for m in moods:  # 已按 ts DESC
            if _iso_age_days(m["ts"]) > cutoff_days:
                break
            note = m.get("note")
            if not note:
                continue
            v = m.get("valence")
            tone = "情绪平稳" if v is None else ("情绪偏好" if v > 0.2 else "情绪偏低落" if v < -0.2 else "情绪平稳")
            return f"截至 {_date_part(m['ts'])}：{note}（{tone}）"
        return None

    def read_memory_md(self) -> str:
        if not self.md_path.exists():
            return ""
        return self.md_path.read_text(encoding="utf-8")

    # ---------- 反思（A2） / 剪枝（C2）：nightly 低频跑 ----------

    async def reflect(self, namespace: str = "default") -> list[str]:
        """Generative Agents reflection：回看画像+叙事，归纳高层洞察存成 narrative(kind=insight)。"""
        facts = await self.store.all_active_facts(namespace)
        narrs = await self.store.top_narratives_for_md(namespace, limit=30)
        if len(facts) + len(narrs) < 3:   # 素材太少，不硬挤洞察
            return []
        material = (
            "【画像】\n" + "\n".join(f"- {f.key}: {f.value}" for f in facts)
            + "\n【小事】\n" + "\n".join(f"- {n['content']}" for n in narrs)
        )
        existing = await self.store.narrative_contents(namespace)
        out: list[str] = []
        for ins in await reflect_insights(self.router, material):
            if ins in existing:           # 不重复归纳同一条洞察
                continue
            nid = await self.store.add_narrative(
                ins, kind="insight", importance=0.85, source="reflection", namespace=namespace,
                confidence=0.85,
            )
            await self.store.add_history("narrative_note", nid, None, ins, "agent", "REFLECT")
            self.journal.append(self.journal.maintenance_entry(
                f"insight:{nid}", f"反思洞察：{ins}",
            ))
            out.append(ins)
        return out

    async def prune(self, namespace: str = "default") -> list[str]:
        """保守剪枝：从没被强化、不重要、且很久没用到的碎片 → 软删（history 可回溯）。"""
        pruned: list[str] = []
        for it in await self.store.active_memory_items(namespace):
            if it.strength > _PRUNE_MAX_STRENGTH or it.importance >= _PRUNE_MAX_IMPORTANCE:
                continue                  # 被强化过 / 重要 → 留
            if _age_days(it) < _PRUNE_MIN_AGE_DAYS:
                continue                  # 不够老 → 留
            await self.store.forget_memory(it.id)
            await self.store.add_history("memory_item", it.id, it.content, None, "agent", "PRUNE")
            self.journal.append(self.journal.maintenance_entry(
                f"prune:{it.id}", f"剪枝低强度记忆碎片：{it.content}",
            ))
            pruned.append(it.id)
        return pruned

    async def expire_stale_sent(self, cutoff: str, namespace: str = "default") -> int:
        """Nightly sent->done fallback with journal projection."""
        stale = await self.store.stale_sent_commitments(cutoff, namespace)
        count = await self.store.expire_stale_sent(cutoff, namespace)
        for c in stale:
            self.journal.append(self.journal.commitment_entry(
                c["id"], c["content"], status="expired",
            ))
        return count


def _iso_age_days(ts: str) -> float:
    from datetime import datetime, timezone
    try:
        t = datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return 0.0
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - t).total_seconds() / 86400.0)
