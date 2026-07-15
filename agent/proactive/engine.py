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
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from agent.gateway.router import LLMRouter
from agent.memory.models import new_id, now_iso
from agent.memory.normalize import current_time_hint
from agent.memory.service import MemoryService
from agent.memory.store import Store
from agent.proactive.policy import DAILY_QUOTA, filter_candidates, post_guard, pre_gate

# 不同 kind 的基础分（重要性）
_KIND_SCORE = {
    "care_check_in": 0.9,
    "deadline_check": 0.8,
    "event_check_in": 0.6,
    "open_loop": 0.5,
    "profile_discovery": 0.45,
}
_KIND_ORDER = ["care_check_in", "deadline_check", "event_check_in", "open_loop", "profile_discovery"]

_PROFILE_TOPIC_COOLDOWN_DAYS = 45
_PROFILE_TOPICS = (
    ("daily_rhythm", "日常节奏", "自然了解用户平时一天的节奏、作息或下班后的日常。"),
    ("work_study", "工作学习", "自然了解用户目前主要在忙的工作、学习方向或最近投入的事情。"),
    ("leisure", "放松方式", "自然了解用户平时如何放松、周末喜欢做什么。"),
    ("food_taste", "饮食口味", "自然了解用户常吃的东西、喜欢的口味或饮品。"),
    ("current_goal", "近期目标", "轻松了解用户最近想推进的一件事，不要求宏大目标。"),
    ("social_circle", "重要关系", "在不打探隐私的前提下，了解用户常相处的人或重要陪伴。"),
    ("interaction_style", "互动偏好", "自然了解用户更喜欢怎样聊天、被提醒或被关心。"),
)


def _topic_is_covered(topic: str, facts: list) -> bool:
    pairs = [(str(f.category).lower(), str(f.key).lower()) for f in facts]
    if topic == "daily_rhythm":
        return any(cat == "routine" for cat, _ in pairs)
    if topic == "work_study":
        return any(
            cat == "bio" and any(word in key for word in ("job", "work", "career", "school", "major", "study"))
            for cat, key in pairs
        )
    if topic == "leisure":
        return any(
            cat in {"preference", "taste"}
            and any(word in key for word in ("hobby", "game", "music", "movie", "book", "sport", "leisure"))
            for cat, key in pairs
        )
    if topic == "food_taste":
        return any(
            cat in {"preference", "taste"}
            and any(word in key for word in ("food", "drink", "coffee", "tea", "flavor", "cuisine"))
            for cat, key in pairs
        )
    if topic == "current_goal":
        return any(any(word in key for word in ("goal", "focus", "target", "plan")) for _, key in pairs)
    if topic == "social_circle":
        return any(cat in {"social", "entity"} for cat, _ in pairs)
    if topic == "interaction_style":
        return any(
            cat in {"preference", "taboo"}
            and any(word in key for word in ("chat", "communication", "reminder", "interaction"))
            for cat, key in pairs
        )
    return False


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
    def __init__(
        self, store: Store, router: LLMRouter, memory: MemoryService, persona: str,
        timezone_name: str = "Asia/Shanghai",
    ) -> None:
        self.store = store
        self.router = router
        self.memory = memory
        self.persona = persona
        self.timezone = timezone_name

    async def tick(self, namespace: str = "default", force: bool = False) -> dict:
        """跑一次心跳。返回 {sent, message, reason}。force=True 跳过时段/冷却/预算（调试用）。"""
        gate = await pre_gate(self.store, namespace, force=force)
        if not gate.passed:
            return await self._skip(namespace, "gate:" + ",".join(gate.reasons), gate_reasons=gate.reasons)

        # 取到期承诺；低落时只留关心类，压制催事类
        now = now_iso()
        due = await self.store.due_commitments(now, namespace)
        candidates = filter_candidates(due, low_mood=gate.low_mood, force=force, now=now)
        if not candidates:
            # 没有时效事项时，每天至多一次自然探索尚缺的画像维度。
            # force 用于调试已有候选，不凭空制造一条画像问题。
            if not force and not gate.low_mood and gate.used == 0:
                profile_candidate = await self._profile_candidate(namespace)
                if profile_candidate is not None:
                    candidates = [profile_candidate]
        if not candidates:
            # 画像暂不适合问时，再考虑无明确时间的 open_loop。
            if force or (not gate.low_mood and gate.used == 0):
                candidates = filter_candidates(
                    await self.store.opportunistic_commitments(now, namespace),
                    low_mood=gate.low_mood,
                    force=force, now=now,
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
        is_profile = commitment["kind"] == "profile_discovery"
        trigger_source = "profile_discovery" if is_profile else "interval"
        if not decision.notify:
            if not is_profile:
                await self.store.mark_commitment_attempted(commitment["id"])
            await self.store.add_proactive_decision_trace(
                trigger_source=trigger_source,
                namespace=namespace, gate_result="pass", gate_reasons=[],
                candidate_id=commitment["id"], candidate_kind=commitment["kind"],
                llm_notify=False, llm_outcome=decision.outcome, llm_summary=decision.summary,
                llm_reason=decision.reason, next_check_after=decision.next_check_after,
                post_guard_result="notified_false", final_sent=False,
            )
            await self.store.add_tick_trace(namespace, False, "notify=false｜" + reason, commitment["id"], None)
            return {"sent": False, "message": None, "reason": "notify=false｜" + reason}

        recent = [p["content"] for p in await self.store.list_proactive(namespace, n=5)]
        ok, guard_reason = post_guard(
            decision.notification_text, recent, kind=commitment["kind"],
        )
        if not ok:
            if not is_profile:
                await self.store.mark_commitment_attempted(commitment["id"])
            await self.store.add_proactive_decision_trace(
                trigger_source=trigger_source,
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
        if not is_profile:
            await self.memory.mark_commitment_sent(commitment["id"], namespace)
        await self.store.add_proactive_decision_trace(
            trigger_source=trigger_source,
            namespace=namespace, gate_result="pass", gate_reasons=[],
            candidate_id=commitment["id"], candidate_kind=commitment["kind"],
            llm_notify=True, llm_outcome=decision.outcome, llm_summary=decision.summary,
            llm_reason=decision.reason, next_check_after=decision.next_check_after,
            post_guard_result="pass", final_sent=True, message=decision.notification_text,
        )
        await self.store.add_tick_trace(namespace, True, reason, commitment["id"], decision.notification_text)
        return {"sent": True, "message": decision.notification_text, "reason": reason}

    async def _profile_candidate(self, namespace: str) -> dict | None:
        facts = await self.memory.list_facts(namespace)
        now = datetime.now(timezone.utc)
        cutoff = (now - timedelta(days=_PROFILE_TOPIC_COOLDOWN_DAYS)).isoformat()
        attempt_cutoff = (now - timedelta(days=1)).isoformat()
        recently_asked = await self.store.recent_profile_question_topics(
            cutoff, namespace, attempt_since=attempt_cutoff,
        )
        for key, label, guidance in _PROFILE_TOPICS:
            if key in recently_asked or _topic_is_covered(key, facts):
                continue
            return {
                "id": f"profile:{key}",
                "kind": "profile_discovery",
                "content": f"探索主题：{label}。{guidance}",
                "topic": key,
                "confidence": 0.65,
                "attempts": 0,
            }
        return None

    async def _decide(self, commitment: dict, namespace: str, low_mood: bool = False) -> ProactiveDecision:
        now_hint = current_time_hint(self.timezone)
        wm = await self.memory.assemble_system_prompt(
            self.persona, query=commitment["content"], namespace=namespace,
            now_hint=now_hint,
        )
        recent_rows = await self.store.recent_messages_with_timestamps(6, namespace)
        recent_context = self._recent_context(recent_rows)
        tone = (
            "对方最近情绪偏低落——语气更软、更短，多一分关心，别催事别灌任务，给点空间。\n"
            if low_mood else ""
        )
        kind = commitment.get("kind")
        task_rule = (
            "这是一次低频的画像探索，不是旧事跟进。围绕给定主题问一个轻松、具体、容易回答的问题。\n"
            "要求：只问一个问题；不提画像、记忆、档案或收集信息；不盘问隐私；不预设答案；"
            "不要硬把已有爱好套进问题。已有信息只用于避免重复，最近对话只用于判断语气和是否自然。\n"
            if kind == "profile_discovery" else
            "这是已有事项的跟进。必须核对当前时间与事项的 event/due/window；过期、错过窗口、"
            "缺少现实价值时 notify=false。提到过去事件时写清日期，绝不能把旧消息里的几点当成今天。\n"
        )
        system = (
            wm.as_system() + "\n\n"
            + "现在是你【主动】找用户。你必须先判断该不该打扰，再决定是否发送。\n"
            + f"当前时间：{now_hint}\n"
            + f"候选事项：{commitment['content']}\n"
            + f"类型：{kind}；event_at={commitment.get('event_at')}；due_at={commitment.get('due_at')}；"
              f"window={commitment.get('due_window_start')}..{commitment.get('due_window_end')}；"
              f"expires_at={commitment.get('expires_at')}；已尝试={commitment.get('attempts', 0)}。\n"
            + task_rule
            + (f"最近对话（均带真实时间）：\n{recent_context}\n" if recent_context else "")
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

    def _recent_context(self, rows: list[dict]) -> str:
        try:
            tz = ZoneInfo(self.timezone)
        except Exception:
            tz = ZoneInfo("UTC")
        lines: list[str] = []
        for row in rows:
            sent_at = datetime.fromtimestamp(float(row["ts"]), tz).strftime("%Y-%m-%d %H:%M")
            content = " ".join(str(row["content"]).split())[:180]
            lines.append(f"- {sent_at} {row['role']}: {content}")
        return "\n".join(lines)

    async def _skip(self, namespace: str, reason: str, gate_reasons: list[str] | None = None) -> dict:
        await self.store.add_proactive_decision_trace(
            namespace=namespace, gate_result="skip", gate_reasons=gate_reasons or [reason],
            post_guard_result=None, final_sent=False,
        )
        await self.store.add_tick_trace(namespace, False, reason, None, None)
        return {"sent": False, "message": None, "reason": reason}
