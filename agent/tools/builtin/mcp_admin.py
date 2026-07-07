"""MCP 自助安装工具：聊天里贴一段标准 mcpServers JSON，agent 自己把 MCP 装上。

install_mcp 标 dangerous=True——它会运行命令/连外部服务，必须经确认门（用户回「确认」才装）。
依赖 ctx.mcp（MCPManager）与 ctx.registry（实时改注册表）；二者由编排层注入。
"""

from __future__ import annotations

from agent.tools.mcp_config import normalize_mcp_json
from agent.tools.registry import ToolContext, tool


@tool(dangerous=True)
async def install_mcp(ctx: ToolContext, config_json: str) -> str:
    """安装/接入一个 MCP 服务器。当用户贴出 MCP 配置 JSON（含 mcpServers、或 command、或 url）
    并希望接入时调用。config_json：原样传入用户给的那段 JSON 字符串。"""
    if ctx.mcp is None or ctx.registry is None:
        return "[未启用 MCP 运行时管理，无法安装]"
    try:
        cfgs = normalize_mcp_json(config_json)
    except Exception as e:
        return f"[配置解析失败] {e}\n请确认贴的是合法 JSON（支持 mcpServers / 单个 command|url 写法）。"

    lines: list[str] = []
    for cfg in cfgs:
        where = cfg.get("command") or cfg.get("url") or "?"
        try:
            names = await ctx.mcp.add_server(cfg, ctx.registry)
            tools = "、".join(n.split("__", 2)[-1] for n in names) or "（无工具）"
            lines.append(f"✅ {cfg['name']}（{cfg['transport']} · {where}）→ {len(names)} 个工具：{tools}")
        except Exception as e:
            lines.append(f"❌ {cfg['name']} 接入失败：{e}")
    return "MCP 安装结果：\n" + "\n".join(lines) + "\n（已存盘，重启自动重连。直接说需求我就能用这些新工具了。）"


@tool
async def list_mcp_servers(ctx: ToolContext) -> str:
    """列出当前已接入的 MCP 服务器及各自工具数。想知道装了哪些 MCP 时调用。"""
    if ctx.mcp is None:
        return "（未启用 MCP）"
    if not ctx.mcp.connected:
        return "还没接入任何 MCP 服务器。贴一段 mcpServers JSON 我可以帮你装。"
    return "已接入的 MCP：\n" + "\n".join(
        f"- {name}：{cnt} 个工具" for name, cnt in ctx.mcp.connected.items()
    )
