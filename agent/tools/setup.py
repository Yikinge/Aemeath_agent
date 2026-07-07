"""统一装配本地工具来源（原生 + 技能）成一个 ToolRegistry。

cli.py / main.py 都调 build_local_registry()，避免各自重复接线。
MCP 因需异步连接，由 MCPManager 单独处理（cli/main 各自 load_servers + connect_all）。
"""

from __future__ import annotations

import logging
from pathlib import Path

from agent.tools.builtin import discover_builtin
from agent.tools.registry import ToolRegistry
from agent.tools.skills import SkillRegistry

log = logging.getLogger(__name__)


def build_local_registry(
    skills_dir: str | None = None, *, lazy_mcp: bool = False, max_eager: int = 25,
) -> ToolRegistry:
    """同步装配本地来源（原生工具 + 技能）。lazy_mcp/max_eager 设定 MCP-B 懒加载策略。"""
    registry = ToolRegistry()
    n_builtin = discover_builtin(registry)
    log.info("原生工具 %d 个", n_builtin)
    if skills_dir:
        n_skill = SkillRegistry().scan(Path(skills_dir), registry)
        log.info("技能 %d 个", n_skill)
    registry.set_lazy_policy(lazy_mcp, max_eager)
    return registry
