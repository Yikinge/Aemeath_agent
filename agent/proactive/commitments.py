"""承诺/开放回路抽取（S6，TDD §5.2）：决定「以后主动说什么」。

从用户的话里抽出值得日后主动跟进的开放回路。和画像/语义记忆抽取一样，跑在 ingest 后台。
"""

from __future__ import annotations

import json
import re

from agent.gateway.router import LLMRouter
from agent.memory.models import CommitmentCandidate

_SYSTEM = """从用户最新的话里抽取「需要以后主动跟进」的开放回路（commitment）。
只输出 JSON 数组，每个元素：
{"kind","content","event_at","due_at","due_window_start","due_window_end","expires_at","canonical_key","dedupe_key","confidence","sensitivity","reason"}
- kind: care_check_in（流露情绪/辛苦/压力，过会儿关心一下）/ open_loop（在做、没了结的事，找机会问）/ event_check_in（未来事件/计划，事后问问怎么样）/ deadline_check（有明确截止时间）
- content: 一句话写清以后主动跟进什么（中文）；涉及日期**用绝对日期**（见下方【时间锚点】），别写"下周三/明天"
- event_at: 事件/计划发生的绝对时间（YYYY-MM-DD 或带时区 datetime）；没有给 null
- due_at: 该在什么时候主动跟进的绝对时间；不急/找机会给 null
- due_window_start / due_window_end: 适合主动跟进的时间窗口。比如"下午3点面试，之后问问"应从面试结束后或当天晚些时候开始，而不是面试开始前。没有时间窗口给 null
- expires_at: 过了就不必再跟进的绝对时间；长期没有给 null
- canonical_key: 同一件事的稳定键（如 event:interview:2026-06-24、loop:job_hunting）；没有给 null
- dedupe_key: 去重键，优先用稳定短键，如 "interview:2026-06-24"、"job_hunting"，没有给 null
- confidence: 0 到 1，表示这条跟进机会是否真的值得以后主动提。弱候选低于 0.5
- sensitivity: routine / personal（私密）/ care（情绪脆弱时刻）
- reason: 一句话「为什么要跟进」
- **一件事只抽一条**：同一件事/同一事件，若有明确时间或属于具体事件（已用 event_check_in/deadline_check 并给了 event_at 或 due_at），就**不要再额外抽一个 open_loop**。open_loop 只留给**真正没有具体事件、也没有时间**的持续性事项（如"在找工作""在减肥""在学吉他"）。
- **跳过明确提醒/定时请求**：用户明确说"几点提醒我"、"明天上午9点叫我"、"每天/每周定时发"、"schedule/remind me/check in at 3" 这类属于 reminder/scheduler 工具，不要抽 commitment；如果助手已经说已设置提醒，也不要重复抽。
- commitment 只用于 agent 推测出来的温和跟进机会，不用于用户明确授权的闹钟。
没有就返回 []。只输出 JSON。"""

_KINDS = {"care_check_in", "open_loop", "event_check_in", "deadline_check"}
_CARE_KINDS = {"care_check_in"}


def _nullable(v) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return None if not s or s.lower() in ("null", "none", "n/a") else s


def _confidence(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return 0.7
    return min(1.0, max(0.0, x))


async def extract_commitments(
    router: LLMRouter, new_messages: list[dict],
    existing: set[str] | None = None, time_hint: str = "",
) -> list[CommitmentCandidate]:
    user_text = "\n".join(m["content"] for m in new_messages if m.get("role") == "user").strip()
    if not user_text:
        return []

    user_content = f"{time_hint}\n\n{user_text}" if time_hint else user_text
    raw = await router.complete(
        [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user_content}],
        task="fast",
    )
    match = re.search(r"\[.*\]", raw.strip(), re.S)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []

    out: list[CommitmentCandidate] = []
    for it in data if isinstance(data, list) else []:
        if not isinstance(it, dict):
            continue
        content = str(it.get("content", "")).strip()
        if not content:
            continue
        if existing and content in existing:   # 廉价预去重；语义去重由 consolidator 用 (kind,canonical_key) 做
            continue
        kind = str(it.get("kind", "open_loop")).strip()
        if kind not in _KINDS:
            kind = "open_loop"
        sens = str(it.get("sensitivity", "")).strip().lower()
        if sens not in ("routine", "personal", "care"):
            sens = "care" if kind in _CARE_KINDS else "routine"
        out.append(
            CommitmentCandidate(
                kind=kind, content=content,
                event_at=_nullable(it.get("event_at")), due_at=_nullable(it.get("due_at")),
                due_window_start=_nullable(it.get("due_window_start")),
                due_window_end=_nullable(it.get("due_window_end")),
                expires_at=_nullable(it.get("expires_at")),
                canonical_key=_nullable(it.get("canonical_key")),
                dedupe_key=_nullable(it.get("dedupe_key")),
                confidence=_confidence(it.get("confidence")),
                sensitivity=sens, reason=str(it.get("reason", "")).strip(),
            )
        )
    return out
