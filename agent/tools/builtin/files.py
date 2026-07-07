"""读取本机文本文件：让她能"看一眼你指的那个文件"。只读、单用户本地，故不标危险；超大截断。"""

from __future__ import annotations

import os
from pathlib import Path

from agent.tools.registry import ToolContext, tool

_MAX_CHARS = 8000


@tool
async def read_file(ctx: ToolContext, path: str) -> str:
    """读取本机一个文本文件的内容。用户让你看/读某个文件（给了路径）时调用。只读文本，超大自动截断。"""
    p = Path(os.path.expanduser(path))
    if not p.exists() or not p.is_file():
        return f"[找不到文件] {path}"
    try:
        data = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"[读不了] {e}"
    if len(data) > _MAX_CHARS:
        data = data[:_MAX_CHARS] + f"\n…[已截断，共 {len(data)} 字符]"
    return data or "（空文件）"
