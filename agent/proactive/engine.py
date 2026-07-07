"""主动引擎（S6，护城河 2）：心跳决定「何时醒」，承诺决定「说什么」，情绪决定「分寸」。

一次 tick：活跃时段 × 冷却 × 频率预算（何时醒）→ 取到期承诺打分（说什么）
→ 用爱弥斯口吻生成一句主动消息 → 记 candidate(带 reason) + 扣预算 + **转 sent 待回应（§9）**。

MEM-7 情绪驱动（护城河）：用户最近情绪明显低落时，收敛主动——只发关心、压制催事、
配额减半、冷却拉长、口吻更软。研究证据：低唤醒情境要克制，主动关心"状态"让人觉得被陪着。
每条主动消息都带 reason（TRUST-3 可回溯）。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date

from agent.gateway.router import LLMRouter
from agent.memory.models import new_id, now_iso
from agent.memory.service import MemoryService
from agent.memory.store import Store
from agent.proactive.policy import DAILY_QUOTA, filter_candidates, post_guard, pre_gate

# 不同 kind 的基础分（重要性）
_KIND_SCORE = {"care_check_in": 0.9, "deadline_check": 0.8, "event_check_in": 0.6, "open_loop": 0.5}
_KIND_ORDER = ["care_check_in", "deadline_check", "event_check_in", "open_loop"]


def _pick(due: list[dict]) -> dict:
    return sorted(
        due,
        key=lambda c: (
            _KIND_ORDER.index(c["kind"]) if c["kind"] in _KIND_ORDER else 99,
            -float(c.get("confidence") if c.get("confidence") is not None else 0.7),
        ),
    )[0]


@dataclass
class ProactiveDecision:
    notify: bool
    notification_text: str
    outcome: str = "progress"
    summary: str = ""
    reason: str = ""
    priority: str = "normal"
    next_check_after: str | None = None


def _json_obj(raw: str) -> dict | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        val = json.loads(raw)
        return val if isinstance(val, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        return None
    try:
        val = json.loads(match.group(0))
        return val if isinstance(val, dict) else None
    except json.JSONDecodeError:
        return None


def _decision_from_raw(raw: str, fallback_reason: str) -> ProactiveDecision:
    data = _json_obj(raw)
    if data is None:
        text = (raw or "").strip()
        return ProactiveDecision(
            notify=bool(text),
            notification_text=text,
            summary=text[:120],
            reason=fallback_reason,
        )
    notify = bool(data.get("notify"))
    text = str(data.get("notification_text") or data.get("notificationText") or "").strip()
    return ProactiveDecision(
        notify=notify,
        notification_text=text,
        outcome=str(data.get("outcome") or ("progress" if notify else "no_change")),
        summary=str(data.get("summary") or "").strip(),
        reason=str(data.get("reason") or fallback_reason).strip(),
        priority=str(data.get("priority") or "normal").strip(),
        next_check_after=(
            str(data.get("next_check_after") or data.get("nextCheck")).strip()
            if data.get("next_check_after") or data.get("nextCheck") else None
        ),
    )


class ProactiveEngine:
    def __init__(self, store: Store, router: LLMRouter, memory: MemoryService, persona: str) -> None:
        self.store = store
        self.router = router
        self.memory = memory
        self.persona = persona

    async def tick(self, namespace: str = "default", force: bool = False) -> dict:
        """跑一次心跳。返回 {sent, message, reason}。force=True 跳过时段/冷却/预算（调试用）。"""
        gate = await pre_gate(self.store, namespace, force=force)
        if not gate.passed:
            return await self._skip(namespace, "gate:" + ",".join(gate.reasons), gate_reasons=gate.reasons)

        # 取到期承诺；低落时只留关心类，压制催事类
        due = await self.store.due_commitments(now_iso(), namespace)
        candidates = filter_candidates(due, low_mood=gate.low_mood, force=force)
        if not candidates:
            # 无到期项 → 无明确时间的 open_loop 走低频策略：仅今天还没主动过、且非低落时找机会问一句
            if force or (not gate.low_mood and gate.used == 0):
                candidates = filter_candidates(
                    await self.store.opportunistic_commitments(now_iso(), namespace),
                    low_mood=gate.low_mood,
                    force=force,
                )
        if not candidates:
            reason = "暂时没有要跟进的事" + ("（情绪低落，仅留关心类）" if gate.low_mood else "")
            return await self._skip(namespace, reason, gate_reasons=["no_candidate"])

        commitment = _pick(candidates)
        score = max(
            _KIND_SCORE.get(commitment["kind"], 0.5),
            float(commitment.get("confidence") if commitment.get("confidence") is not None else 0.7),
        )

        decision = await self._decide(commitment, namespace, gate.low_mood)
        reason = decision.reason or f"{commitment['kind']}｜{commitment['content']}"
        if not decision.notify:
            await self.store.mark_commitment_attempted(commitment["id"])
            await self.store.add_proactive_decision_trace(
                namespace=namespace, gate_result="pass", gate_reasons=[],
                candidate_id=commitment["id"], candidate_kind=commitment["kind"],
                llm_notify=False, llm_outcome=decision.outcome, llm_summary=decision.summary,
                llm_reason=decision.reason, next_check_after=decision.next_check_after,
                post_guard_result="notified_false", final_sent=False,
            )
            await self.store.add_tick_trace(namespace, False, "notify=false｜" + reason, commitment["id"], None)
            return {"sent": False, "message": None, "reason": "notify=false｜" + reason}

        recent = [p["content"] for p in await self.store.list_proactive(namespace, n=5)]
        ok, guard_reason = post_guard(decision.notification_text, recent)
        if not ok:
            await self.store.mark_commitment_attempted(commitment["id"])
            await self.store.add_proactive_decision_trace(
                namespace=namespace, gate_result="pass", gate_reasons=[],
                candidate_id=commitment["id"], candidate_kind=commitment["kind"],
                llm_notify=True, llm_outcome=decision.outcome, llm_summary=decision.summary,
                llm_reason=decision.reason, next_check_after=decision.next_check_after,
                post_guard_result=guard_reason, final_sent=False,
                message=decision.notification_text,
            )
            await self.store.add_tick_trace(namespace, False, "post_guard:" + guard_reason, commitment["id"], None)
            return {"sent": False, "message": None, "reason": "post_guard:" + guard_reason}

        await self.store.add_proactive_candidate(
            new_id(), commitment["id"], decision.notification_text, score, reason, "sent", namespace
        )
        today = date.today().isoformat()
        await self.store.increment_budget(today)
        # §9：发出 = 转 sent（待回应），不直接 done；并刷新 MEMORY.md 把它从「当前开放回路」移除
        await self.memory.mark_commitment_sent(commitment["id"], namespace)
        await self.store.add_proactive_decision_trace(
            namespace=namespace, gate_result="pass", gate_reasons=[],
            candidate_id=commitment["id"], candidate_kind=commitment["kind"],
            llm_notify=True, llm_outcome=decision.outcome, llm_summary=decision.summary,
            llm_reason=decision.reason, next_check_after=decision.next_check_after,
            post_guard_result="pass", final_sent=True, message=decision.notification_text,
        )
        await self.store.add_tick_trace(namespace, True, reason, commitment["id"], decision.notification_text)
        return {"sent": True, "message": decision.notification_text, "reason": reason}

    async def _decide(self, commitment: dict, namespace: str, low_mood: bool = False) -> ProactiveDecision:
        wm = await self.memory.assemble_system_prompt(
            self.persona, query=commitment["content"], namespace=namespace,
        )
        tone = (
            "对方最近情绪偏低落——语气更软、更短，多一分关心，别催事别灌任务，给点空间。\n"
            if low_mood else ""
        )
        system = (
            wm.as_system() + "\n\n"
            + "现在是你【主动】找用户。你必须先判断该不该打扰，再决定是否发送。\n"
            + f"惦记的事：{commitment['content']}\n"
            + f"类型：{commitment.get('kind')}；置信度：{commitment.get('confidence', 0.7)}；已尝试：{commitment.get('attempts', 0)} 次。\n"
            + tone
            + "只输出 JSON，不要 Markdown，不要多余解释。格式：\n"
            + '{"notify": true/false, "outcome": "no_change|progress|done|blocked|needs_attention", '
              '"summary": "内部摘要", "notification_text": "若发送，给用户看的短消息", '
              '"reason": "为什么现在适合或不适合发", "priority": "low|normal|high", '
              '"next_check_after": "可选 ISO 时间"}\n'
            + "发送标准：有明确价值才 notify=true；没必要、太打扰、信息不足就 notify=false。"
        )
        raw = await self.router.complete(
            [{"role": "system", "content": system},
             {"role": "user", "content": "（系统：请做一次主动互动决策）"}]
        )
        return _decision_from_raw(raw, f"{commitment['kind']}｜{commitment['content']}")

    async def _skip(self, namespace: str, reason: str, gate_reasons: list[str] | None = None) -> dict:
        await self.store.add_proactive_decision_trace(
            namespace=namespace, gate_result="skip", gate_reasons=gate_reasons or [reason],
            post_guard_result=None, final_sent=False,
        )
        await self.store.add_tick_trace(namespace, False, reason, None, None)
        return {"sent": False, "message": None, "reason": reason}
