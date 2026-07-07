"""人格加载（S4，TDD §6.1）：读 SOUL.md 注入 system prompt。

SOUL.md 缺失时回退到 config 里的占位人格。后续可加「agent 自我编辑 SOUL.md」。
"""

from __future__ import annotations

from pathlib import Path


def load_soul(path: str, fallback: str) -> str:
    p = Path(path)
    if p.exists():
        text = p.read_text(encoding="utf-8").strip()
        if text:
            return text
    return fallback
