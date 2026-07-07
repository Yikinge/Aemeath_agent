"""工具搜索（MCP-B 规模治理）：让模型按需发现并启用"懒加载"的工具。

工具一多（尤其接了多个 MCP server），全量注入 schema 会撑爆上下文。策略：
MCP 工具默认懒加载、不进 tools[]；模型需要某能力时先调 search_tools 搜一下，
命中的工具被激活，下一步即可直接调用。常驻原生工具不受影响。
"""

from __future__ import annotations

from agent.tools.registry import ToolContext, tool


@tool
async def search_tools(ctx: ToolContext, query: str) -> str:
    """搜索并启用当前未直接装载的工具（多来自已接入的 MCP）。当你需要某种能力、但工具列表里
    没看到对应工具时，先用它搜一下（如"网页搜索""文件""天气"）。命中的工具会被启用，随后可直接调用。"""
    reg = ctx.registry
    if reg is None:
        return "（无法搜索工具）"
    if not reg.has_lazy():
        return "当前没有需要按需加载的工具，已有工具都可直接调用。"
    hits = reg.search_catalog(query)
    if not hits:
        return f"没找到和「{query}」相关的可用工具。换个说法再搜，或先用 list_mcp_servers 看看接入了哪些 MCP。"
    reg.activate([t.name for t in hits])
    return "已启用以下工具，现在可以直接调用：\n" + "\n".join(
        f"- {t.name}：{t.description}" for t in hits
    )
