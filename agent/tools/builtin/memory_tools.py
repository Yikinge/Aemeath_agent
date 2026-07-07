"""记忆工具：让模型主动查长期记忆、主动记一条要点。直接强化护城河（记忆）。"""

from __future__ import annotations

from agent.tools.registry import ToolContext, tool


@tool
async def search_memory(ctx: ToolContext, query: str) -> str:
    """按关键词/问题检索关于用户的长期记忆（画像、过往小事、偏好）。
    当需要回忆用户以前说过的事、确认某个细节时调用。"""
    hits = await ctx.memory.retrieve(query, namespace=ctx.namespace, k=5)
    if not hits:
        return "（没有检索到相关记忆）"
    # 命中即强化（MemoryBank 间隔重复）：和注入路径一致
    await ctx.store.reinforce_memories([h.item.id for h in hits])
    return "\n".join(f"- {h.item.content}（相关度 {h.score:.2f}）" for h in hits)


@tool
async def remember_fact(ctx: ToolContext, text: str) -> str:
    """主动记下一条用户透露的、值得长期记住的要点。会进巩固缓冲，由 Consolidator 决定如何归档。
    仅在用户明确说了值得记的新信息时用，别滥用。"""
    text = (text or "").strip()
    if not text:
        return "（没有可记的内容）"
    await ctx.store.add_pending_intake(text, ctx.namespace)
    return f"好，记下了：{text}"
