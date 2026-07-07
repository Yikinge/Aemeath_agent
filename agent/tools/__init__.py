"""能力层（TDD §3「工具集统一注册」的落地）。

一个注册表（ToolRegistry）+ 一个循环（run_tool_loop）+ 三个来源（原生/MCP/Skill）。
三来源最终都变成注册表里的一个 Tool，共用同一条执行路径。详见 工具与技能设计文档.md。
"""

from agent.tools.registry import (
    ConfirmationRequired,
    Tool,
    ToolContext,
    ToolRegistry,
    tool,
)

__all__ = [
    "ConfirmationRequired",
    "Tool",
    "ToolContext",
    "ToolRegistry",
    "tool",
]
