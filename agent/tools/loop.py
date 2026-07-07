"""工具调用循环（TOOL-0 核心）：与编排解耦，单独可测。

call LLM(带 tools) → 有 tool_calls? → 执行 → 回灌结果 → 再 call → 直到出文本 / 触 max_steps。
危险工具命中 → 中断循环、转确认门（P0 简化：确认后单独执行一次，不追求循环无缝续接）。
循环里的 assistant(tool_calls) / role=tool 消息是临时的，不写进正式对话历史。
"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable

from agent.gateway.router import LLMRouter
from agent.tools.builtin.commitments import render_reminder_reply
from agent.tools.registry import ConfirmationRequired, ToolContext, ToolRegistry

# DeepSeek 多步偏弱，保守起步；强模型可调高
MAX_STEPS = 4

# on_trace 回调签名：(step, call, result, ok, ms) -> None
TraceFn = Callable[[int, dict, str, bool, int], Awaitable[None]]


def _safe_json(raw: str) -> dict:
    try:
        val = json.loads(raw) if raw else {}
        return val if isinstance(val, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _assistant_msg(turn) -> dict:
    """把带 tool_calls 的助手回合，还原成 OpenAI 上下文消息塞回去。"""
    return {
        "role": "assistant",
        "content": turn.content or None,
        "tool_calls": [
            {
                "id": c["id"],
                "type": "function",
                "function": {"name": c["name"], "arguments": c["arguments"]},
            }
            for c in turn.tool_calls
        ],
    }


def _tool_msg(call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _last_user_text(messages: list[dict]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return str(msg.get("content") or "")
    return ""


async def run_tool_loop(
    router: LLMRouter,
    registry: ToolRegistry,
    messages: list[dict],
    ctx: ToolContext,
    *,
    on_trace: TraceFn | None = None,
    max_steps: int = MAX_STEPS,
    deny: set[str] | None = None,
) -> str:
    """跑工具循环，返回给用户的最终文本。messages 会被原地追加临时消息。"""
    if not registry.to_openai_tools(deny=deny):
        return await router.complete(messages)  # 没有可用工具 → 退回普通对话

    for step in range(max_steps):
        # 每轮重算：search_tools 激活的懒工具，下一轮即可见可调（MCP-B 按需加载）
        tools = registry.to_openai_tools(deny=deny)
        turn = await router.chat(messages, tools=tools)
        if not turn.wants_tools:
            return turn.content

        messages.append(_assistant_msg(turn))
        for call in turn.tool_calls:
            args = _safe_json(call["arguments"])
            t0 = time.monotonic()
            try:
                result = await registry.dispatch(call["name"], args, ctx)
            except ConfirmationRequired as cr:
                # 危险工具：中断循环，转确认门，把待执行动作暂存到 pending_action
                ms = int((time.monotonic() - t0) * 1000)
                if ctx.confirm is None:
                    note = "[已跳过：危险工具但未配置确认门]"
                    messages.append(_tool_msg(call["id"], note))
                    if on_trace:
                        await on_trace(step, call, note, False, ms)
                    continue
                if on_trace:
                    await on_trace(step, call, f"[待确认] {cr.summary}", True, ms)
                return await ctx.confirm.request(
                    "tool_call", cr.summary,
                    {"name": cr.name, "arguments": cr.arguments, "namespace": ctx.namespace},
                    ctx.namespace,
                )
            ms = int((time.monotonic() - t0) * 1000)
            messages.append(_tool_msg(call["id"], result))
            if on_trace:
                await on_trace(step, call, result, not result.startswith("[工具错误]"), ms)
            if call["name"] == "add_reminder" and not result.startswith("[工具错误]"):
                return await render_reminder_reply(
                    ctx.router, ctx.persona, result, user_text=_last_user_text(messages)
                )

    # 触顶兜底：再要一次纯文本收尾，别把工具中间态丢给用户
    return await router.complete(messages)
