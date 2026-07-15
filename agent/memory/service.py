"""MemoryService —— 记忆服务的唯一对外契约（TDD §4.3 + 新分层）。

新分层（人类可读 / 向量召回）后的职责：
- 写入：ingest 只往 pending_intake 缓冲，达阈值或被手动触发时调 Consolidator
- 读取：retrieve（向量召回层）+ assemble_system_prompt（拼可缓存的 prefix + 动态 suffix）
- 信任层：list / edit / forget / history / resolve_contradiction（控制台用）

resolve 管线本身（extract→resolve / 叙事 / 情绪 / 关系 / MEMORY.md 重写）已下沉到 Consolidator。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent.gateway.router import LLMRouter
from agent.memory.consolidator import Consolidator
from agent.memory.gate import gate_memories
from agent.memory.models import (
    IngestResult, MemoryHit, MemoryItem, ProfileFact,
)
from agent.memory.retrieve import retrieve
from agent.memory.store import Store


@dataclass
class WorkingMemory:
    """组装好的 system prompt 两段：稳定 prefix（命中 cache）+ 动态 suffix（按 query 召回）。"""
    stable_prefix: str
    dynamic_suffix: str
    retrieved: list[MemoryHit]
    gate_decisions: list[dict] = field(default_factory=list)  # 每条命中的 inject/skip + reason（P2 可回溯）

    def as_system(self) -> str:
        if self.dynamic_suffix:
            return f"{self.stable_prefix}\n\n{self.dynamic_suffix}"
        return self.stable_prefix


class MemoryService:
    def __init__(
        self, store: Store, router: LLMRouter, consolidator: Consolidator,
        consolidate_threshold: int = 2,
    ) -> None:
        self.store = store
        self.router = router
        self.consolidator = consolidator
        self.consolidate_threshold = consolidate_threshold

    # ---------- 写入（缓冲） ----------

    async def ingest(
        self, new_messages: list[dict], namespace: str = "default", *,
        source_at: str | None = None, timezone: str | None = None,
        message_id: str | None = None,
    ) -> IngestResult:
        """把这一轮的 user 文本写进 pending；达阈值才真正巩固。
        非 user 角色忽略（assistant 自己说的不抽取）。
        source_at/timezone 是解析"明天/下周三"的锚点，落进 pending 供 Consolidator 归一化。"""
        user_text = "\n".join(
            m["content"] for m in new_messages if m.get("role") == "user"
        ).strip()
        if user_text:
            await self.store.add_pending_intake(
                user_text, namespace,
                source_at=source_at, timezone=timezone, message_id=message_id,
            )

        if await self.store.pending_count(namespace) >= self.consolidate_threshold:
            return await self.consolidator.consolidate(namespace)
        return IngestResult()

    async def flush(self, namespace: str = "default") -> IngestResult:
        """强制清空 pending（cli.drain / nightly job / 用户手动触发）。"""
        return await self.consolidator.consolidate(namespace)

    # ---------- 承诺生命周期（§9：open → sent → done，闭合即刷新 core memory） ----------

    async def mark_commitment_sent(self, cid: str, namespace: str = "default") -> None:
        """主动 check-in 发出 → 转 sent，并立即刷新 MEMORY.md（从「当前开放回路」移除，修 #1）。"""
        cur = await self.store.get_commitment(cid)
        await self.store.mark_commitment_sent(cid)
        if cur:
            self.consolidator.journal.append(self.consolidator.journal.commitment_entry(
                cid, cur["content"], status="sent",
            ))
        await self.consolidator.refresh_memory_md(namespace)

    async def complete_commitment(self, cid: str, namespace: str = "default") -> None:
        """闭合承诺（done + completed_at）并刷新 MEMORY.md。统一闭合入口。"""
        cur = await self.store.get_commitment(cid)
        await self.store.set_commitment_status(cid, "done")
        if cur:
            self.consolidator.journal.append(self.consolidator.journal.commitment_entry(
                cid, cur["content"], status="done",
            ))
        await self.consolidator.refresh_memory_md(namespace)

    async def close_sent_commitments(self, namespace: str = "default") -> int:
        """用户来了新消息 → 把已发出待回应（sent）的承诺闭成 done。
        用户这条回复随后经正常 ingest→consolidate 自然回沉 event_memory（不硬造）。返回闭合条数。"""
        sent = await self.store.list_sent_commitments(namespace)
        if not sent:
            return 0
        for c in sent:
            await self.store.set_commitment_status(c["id"], "done")
            self.consolidator.journal.append(self.consolidator.journal.commitment_entry(
                c["id"], c["content"], status="done",
            ))
        await self.consolidator.refresh_memory_md(namespace)
        return len(sent)

    # ---------- 读取 / 注入 ----------

    async def list_facts(self, namespace: str = "default") -> list[ProfileFact]:
        return await self.store.all_active_facts(namespace)

    async def list_memories(self, namespace: str = "default") -> list[MemoryItem]:
        return await self.store.active_memory_items(namespace)

    async def list_narratives(self, namespace: str = "default") -> list[dict]:
        return await self.store.list_narratives(namespace)

    async def list_mood(self, namespace: str = "default", n: int = 100) -> list[dict]:
        return await self.store.list_mood(namespace, n)

    async def retrieve(self, query: str, *, namespace: str = "default", k: int = 5) -> list[MemoryHit]:
        return await retrieve(self.store, self.router, query, namespace=namespace, k=k)

    async def assemble_system_prompt(
        self, persona: str, query: str = "", namespace: str = "default",
        now_hint: str | None = None,
    ) -> WorkingMemory:
        """两段拼装：
        - stable_prefix：persona + MEMORY.md → 跨多轮稳定，命中 prompt cache
        - dynamic_suffix：当前时间（每轮变）+ 按 query 召回的相关记忆碎片
        now_hint 放进动态段、不进缓存前缀：让模型每轮知道真实"现在"，又不破坏 prompt cache。
        """
        memory_md = self.consolidator.read_memory_md().strip()
        prefix_parts = [persona]
        if memory_md:
            prefix_parts.append(f"【关于用户的工作记忆 · MEMORY.md】\n{memory_md}")
        stable_prefix = "\n\n".join(prefix_parts)

        hits = await self.retrieve(query, namespace=namespace, k=5) if query else []
        # P2 MemoryGate：召回后按本轮任务再门控（工具任务少灌情绪、敏感记忆不进工具上下文、问日期需锚点）
        kept, decisions = gate_memories(hits, query) if hits else ([], [])
        dyn_parts: list[str] = []
        if now_hint:
            dyn_parts.append(
                f"【当前时间】现在是 {now_hint}。涉及日期/时间一律以此为准，不要自己编。\n"
                "【历史时间规则】历史消息的真实日期由内部时间索引提供。旧消息里的几点、今天、明天和提醒，"
                "只属于那条消息的日期；除非当前仍存在有效任务或用户主动提起，否则不要把过期提醒、"
                "旧计划或已发生事件当成今天的待办，也不要主动追问。"
            )
        if kept:
            # MemoryBank 间隔重复：只强化**真正被注入**的碎片（被门控跳过的不算用到）
            await self.store.reinforce_memories([h.item.id for h in kept])
            recall_lines = [f"- {h.item.content}（{h.score:.2f}）" for h in kept]
            dyn_parts.append("【为本轮额外召回（按相关度）】\n" + "\n".join(recall_lines))
        dynamic_suffix = "\n\n".join(dyn_parts)
        return WorkingMemory(
            stable_prefix=stable_prefix, dynamic_suffix=dynamic_suffix,
            retrieved=hits, gate_decisions=decisions,
        )

    # 旧接口保留以兼容（供 ProactiveEngine 内部使用 / 后向）
    async def render_working_memory(self, query: str = "", namespace: str = "default") -> str:
        wm = await self.assemble_system_prompt(persona="", query=query, namespace=namespace)
        # 仅返回记忆部分（去掉 persona 占位）
        return (wm.stable_prefix + ("\n\n" + wm.dynamic_suffix if wm.dynamic_suffix else "")).strip()

    async def pending_contradictions(self, namespace: str = "default") -> list[dict]:
        return await self.store.list_pending_contradictions(namespace)

    # ---------- 信任层：看/改/删 + 版本历史（TRUST-1/3） ----------

    async def edit_fact(
        self, fact_id: str, *, value: str | None = None, confidence: float | None = None,
        actor: str = "user",
    ) -> ProfileFact | None:
        cur = await self.store.get_fact(fact_id)
        if cur is None:
            return None
        await self.store.update_fact_fields(fact_id, value=value, confidence=confidence)
        if value is not None and value != cur.value:
            await self.store.add_history("profile_fact", fact_id, cur.value, value, actor, "EDIT")
        await self.consolidator.refresh_memory_md(cur.namespace)  # 画像在 MEMORY.md 里，改完即刷新
        return await self.store.get_fact(fact_id)

    async def forget(self, fact_id: str, actor: str = "user") -> bool:
        cur = await self.store.get_fact(fact_id)
        if cur is None or cur.valid_until is not None:
            return False
        await self.store.forget_fact(fact_id)
        await self.store.add_history("profile_fact", fact_id, cur.value, None, actor, "FORGET")
        await self.consolidator.refresh_memory_md(cur.namespace)
        return True

    async def forget_memory(self, item_id: str, actor: str = "user") -> bool:
        cur = await self.store.get_memory_item(item_id)
        if cur is None or cur.valid_until is not None:
            return False
        await self.store.forget_memory(item_id)
        await self.store.add_history("memory_item", item_id, cur.content, None, actor, "FORGET")
        return True

    async def forget_narrative(self, nid: str, actor: str = "user", namespace: str = "default") -> bool:
        cur = await self.store.get_narrative(nid)
        if cur is None or cur.get("valid_until"):
            return False
        await self.store.forget_narrative(nid)
        # 级联：对应的向量碎片也失效
        await self.store.forget_chunks_of("narrative_note", nid)
        await self.store.add_history("narrative_note", nid, cur["content"], None, actor, "FORGET")
        # insight 类叙事会进 MEMORY.md，遗忘后刷新（event 类不在 prefix 里，刷一下也无妨）
        await self.consolidator.refresh_memory_md(namespace)
        return True

    async def history(self, target_id: str | None = None, n: int = 200) -> list[dict]:
        return await self.store.recent_history(n=n, target_id=target_id)

    async def resolve_contradiction(self, cid: str, keep: str, actor: str = "user") -> bool:
        pendings = await self.store.list_pending_contradictions()
        item = next((p for p in pendings if p["id"] == cid), None)
        if item is None:
            return False

        new_ref, old_ref = item["new_fact_ref"], item["conflicting_fact_ref"]
        if keep == "new":
            new_fact = await self.store.get_fact(new_ref)
            old_fact = await self.store.get_fact(old_ref)
            if new_fact is None or old_fact is None:
                return False
            await self.store.clear_fact_validity(new_ref)
            await self.store.supersede_fact(old_ref, new_ref)
            await self.store.add_history(
                "profile_fact", old_ref, old_fact.value, new_fact.value, actor, "CONTRADICTION_RESOLVED_NEW"
            )
        elif keep == "old":
            new_fact = await self.store.get_fact(new_ref)
            if new_fact is not None:
                await self.store.add_history(
                    "profile_fact", new_ref, new_fact.value, None, actor, "CONTRADICTION_REJECTED_NEW"
                )
        else:
            return False

        await self.store.set_contradiction_status(cid, "resolved")
        await self.consolidator.refresh_memory_md()  # 矛盾解决可能换了生效画像 → 刷新 core memory
        return True
