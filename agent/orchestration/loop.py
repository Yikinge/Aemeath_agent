"""轻量编排（S5，TDD §7）：统一 run_turn()，被 CLI / Telegram 共用。

注入策略（cache 友好）：
  system = [稳定 prefix(SOUL+MEMORY.md) || 动态 suffix(本轮召回)]
  history = 最近 N 条消息
  user    = 本轮文本
为下游 provider 的 prompt cache 留好稳定边界（Anthropic 5min TTL）。每轮写 turn_trace 留痕调试。
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from agent.gateway.router import LLMRouter
from agent.memory.models import now_iso
from agent.memory.normalize import current_time_hint
from agent.memory.service import MemoryService
from agent.memory.store import Store
from agent.tools.loop import run_tool_loop
from agent.tools.builtin.commitments import add_reminder, parse_reminder_request, render_reminder_reply
from agent.tools.registry import ToolContext, ToolRegistry
from agent.trust.confirm import ConfirmGate, parse_decision


_LEAKED_TIME_PREFIX_RE = re.compile(
    r"^\s*[\[【](?:该)?消息发送于[^\]】\n]*[\]】]\s*",
    re.IGNORECASE,
)


def _history_for_model(rows: list[dict]) -> list[dict]:
    """Keep history free of timestamp labels, including artifacts already stored."""
    return [
        {"role": row["role"], "content": _strip_leaked_time_prefix(str(row["content"]))}
        for row in rows
    ]


def _history_time_context(rows: list[dict], timezone: str) -> str:
    """Render timestamps as internal system metadata, separate from conversation text."""
    try:
        tz = ZoneInfo(timezone)
    except Exception:
        tz = ZoneInfo("UTC")
    lines: list[str] = []
    for index, row in enumerate(rows, start=1):
        sent_at = datetime.fromtimestamp(float(row["ts"]), tz)
        label = sent_at.strftime("%Y-%m-%d %H:%M")
        lines.append(f"{index}. {row['role']} @ {label}")
    return (
        "【内部历史时间索引】以下序号对应随后历史消息的顺序，仅用于判断时间先后。"
        "不要在回复中复述、引用或展示本索引。\n" + "\n".join(lines)
    )


def _strip_leaked_time_prefix(reply: str) -> str:
    """Last-resort guard against models imitating an internal timestamp label."""
    cleaned = reply or ""
    while True:
        updated = _LEAKED_TIME_PREFIX_RE.sub("", cleaned, count=1)
        if updated == cleaned:
            return cleaned.strip() or "我在。"
        cleaned = updated


class Orchestrator:
    def __init__(
        self, store: Store, router: LLMRouter, memory: MemoryService,
        persona: str, confirm: ConfirmGate,
        registry: ToolRegistry | None = None, tool_deny: set[str] | None = None,
        mcp=None, reminders=None, timezone: str = "Asia/Shanghai",
    ) -> None:
        self.store = store
        self.router = router
        self.memory = memory
        self.persona = persona
        self.confirm = confirm
        self.registry = registry
        self.tool_deny = tool_deny or set()
        self.mcp = mcp
        self.reminders = reminders
        self.timezone = timezone
        self._tasks: set = set()

    async def run_turn(
        self, text: str, namespace: str = "default", *,
        source_at: str | None = None, message_id: str | None = None,
    ) -> str:
        # 确认门：若有待确认的外发动作、且用户在表态 → 先处理它
        pending = await self.store.latest_pending_action(namespace)
        if pending is not None:
            decision = parse_decision(text)
            if decision is not None:
                return await self.confirm.resolve(pending, decision, namespace)

        t0 = time.monotonic()
        source_at = source_at or now_iso()   # 用户发出消息的绝对时刻 = 解析"明天/下周三"的锚点
        await self.store.add_message("user", text, namespace)
        # §9：用户来消息了 → 把已发出待回应的主动 check-in 闭合（这条回复随后经 consolidate 自然回沉 event_memory）
        await self.memory.close_sent_commitments(namespace)
        history_rows = await self.store.recent_messages_with_timestamps(20, namespace)
        history = _history_for_model(history_rows)
        history_time_context = _history_time_context(history_rows, self.timezone)

        direct_reminder = parse_reminder_request(text)
        if direct_reminder is not None and "add_reminder" not in self.tool_deny:
            ctx = ToolContext(
                self.store, self.memory, self.router, self.confirm, namespace,
                registry=self.registry, mcp=self.mcp, reminders=self.reminders, persona=self.persona,
            )
            t_tool = time.monotonic()
            raw_reply = await add_reminder(ctx, **direct_reminder)
            reply = await render_reminder_reply(
                self.router, self.persona, raw_reply, user_text=text,
            )
            await self.store.add_tool_trace(
                namespace, 0, "add_reminder", "builtin",
                json.dumps(direct_reminder, ensure_ascii=False), raw_reply, True,
                int((time.monotonic() - t_tool) * 1000),
            )
            await self.store.add_message("assistant", reply, namespace)
            latency_ms = int((time.monotonic() - t0) * 1000)
            await self.store.add_turn_trace(namespace, text, "", "", [], reply, latency_ms)
            self._spawn(self.memory.ingest(
                [{"role": "user", "content": text}], namespace,
                source_at=source_at, timezone=self.timezone, message_id=message_id,
            ))
            return reply

        # 工作记忆 prefix（SOUL + MEMORY.md）+ 动态 suffix（当前时间 + 召回）
        wm = await self.memory.assemble_system_prompt(
            self.persona, query=text, namespace=namespace,
            now_hint=current_time_hint(self.timezone),
        )

        messages = [{
            "role": "system",
            "content": wm.as_system() + "\n\n" + history_time_context,
        }, *history]
        if self.registry is not None and self.registry.to_openai_tools(deny=self.tool_deny):
            ctx = ToolContext(
                self.store, self.memory, self.router, self.confirm, namespace,
                registry=self.registry, mcp=self.mcp, reminders=self.reminders, persona=self.persona,
            )
            reply = await run_tool_loop(
                self.router, self.registry, messages, ctx,
                on_trace=self._make_tracer(namespace), deny=self.tool_deny,
            )
        else:
            reply = await self.router.complete(messages)
        reply = _strip_leaked_time_prefix(reply)
        await self.store.add_message("assistant", reply, namespace)
        latency_ms = int((time.monotonic() - t0) * 1000)

        gate_by_id = {d["memory_id"]: d for d in wm.gate_decisions}
        await self.store.add_turn_trace(
            namespace, text, wm.stable_prefix, wm.dynamic_suffix,
            [{"id": h.item.id, "score": h.score, "content": h.item.content,
              "components": h.components,
              "gate": gate_by_id.get(h.item.id, {}).get("decision"),
              "gate_reason": gate_by_id.get(h.item.id, {}).get("reason")} for h in wm.retrieved],
            reply, latency_ms,
        )

        # 后台异步缓冲到 pending_intake；达阈值时 consolidator 在内自动跑
        self._spawn(self.memory.ingest(
            [{"role": "user", "content": text}], namespace,
            source_at=source_at, timezone=self.timezone, message_id=message_id,
        ))
        # 热更新：消息处理完成后，若 mcp.json 变了就后台重载（不阻塞回复）
        if self.mcp is not None:
            self._spawn(self.mcp.maybe_reload(self.registry))
        return reply

    def _make_tracer(self, namespace: str):
        """产出 on_trace 回调：把每次工具调用落进 tool_trace（控制台可观测）。"""
        async def trace(step: int, call: dict, result: str, ok: bool, ms: int) -> None:
            tool = self.registry.get(call["name"]) if self.registry else None
            await self.store.add_tool_trace(
                namespace, step, call["name"], tool.source if tool else None,
                call.get("arguments") or "{}", result, ok, ms,  # arguments 本就是 JSON 串，勿再编码
            )
        return trace

    def _spawn(self, coro) -> None:
        t = asyncio.create_task(coro)
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)

    async def drain(self) -> None:
        """等所有 fire-and-forget 完成，并强制 flush 一次（清空 pending）。"""
        if self._tasks:
            await asyncio.gather(*self._tasks)
        await self.memory.flush()
