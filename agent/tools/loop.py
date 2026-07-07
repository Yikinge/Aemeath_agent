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

# DeepSeek 多步工具链路有时需要 search -> fetch -> summarize，给它多一点空间。
MAX_STEPS = 6
_MAX_OBSERVATION_CHARS = 1200

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


def _looks_like_tool_markup(text: str) -> bool:
    markers = ("<｜｜DSML｜｜tool_calls>", "<｜｜DSML｜｜invoke", '"tool_calls"', '"function_call"')
    return any(m in (text or "") for m in markers)


def _clip(text: str, limit: int = _MAX_OBSERVATION_CHARS) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n...[truncated]"


def _compact_observations(observations: list[dict]) -> str:
    parts = []
    for i, obs in enumerate(observations, 1):
        status = "ok" if obs["ok"] else "error"
        parts.append(
            f"[{i}] tool={obs['tool']} status={status}\n"
            f"args={json.dumps(obs['args'], ensure_ascii=False)}\n"
            f"result:\n{_clip(obs['result'], 1000)}"
        )
    return "\n\n".join(parts)


def _extract_useful_lines(text: str, limit: int = 8) -> list[str]:
    lines = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.lower().startswith(("for more information", "error:", "traceback")):
            continue
        lines.append(line)
        if len(lines) >= limit:
            break
    return lines


def _best_effort_from_observations(observations: list[dict]) -> str:
    usable = [o for o in observations if o["ok"] and not str(o["result"]).startswith("[工具错误]")]
    if not usable:
        return (
            "我刚才查资料时工具调用轮数到上限了，而且没有拿到可用结果。"
            "你可以换个更具体的问题再让我查一次，比如限定数据源、市场或时间范围。"
        )

    bullets: list[str] = []
    for obs in usable:
        lines = _extract_useful_lines(obs["result"], limit=5)
        if not lines:
            continue
        bullets.append(f"- {obs['tool']}：{_clip(' / '.join(lines), 280)}")
        if len(bullets) >= 5:
            break

    if not bullets:
        bullets = [f"- {o['tool']}：拿到了结果，但内容太长，没来得及完整整理。" for o in usable[:3]]

    return (
        "我查资料时工具调用轮数到上限了，但已经拿到一些线索：\n"
        + "\n".join(bullets)
        + "\n\n基于这些线索，结论还不够稳。我可以继续缩小范围再查一次，或者你指定一个数据源，我直接给你整理成结论。"
    )


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

    observations: list[dict] = []
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
            observations.append({
                "tool": call["name"],
                "args": args,
                "result": result,
                "ok": not result.startswith("[工具错误]"),
            })
            if on_trace:
                await on_trace(step, call, result, not result.startswith("[工具错误]"), ms)
            if call["name"] == "add_reminder" and not result.startswith("[工具错误]"):
                return await render_reminder_reply(
                    ctx.router, ctx.persona, result, user_text=_last_user_text(messages)
                )

    # 触顶兜底：再要一次纯文本收尾，别把工具中间态丢给用户
    messages.append({
        "role": "system",
        "content": (
            "工具调用次数已经到上限。现在必须只基于已有工具结果直接回答用户；"
            "不要再调用工具，不要输出 XML/JSON/tool_calls/代码块。\n\n"
            "如果信息不完整，也要给 best-effort 答复：先说已经查到什么，再说明不确定性，"
            "最后给下一步可以怎么继续查。\n\n"
            f"已有工具结果摘要：\n{_compact_observations(observations)}"
        ),
    })
    final = await router.complete(messages)
    if _looks_like_tool_markup(final):
        return _best_effort_from_observations(observations)
    return final
