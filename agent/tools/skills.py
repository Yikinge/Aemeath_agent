"""Skill 系统（TOOL-3，drop-in #3）：Anthropic SKILL.md 约定 + 渐进披露。

加一个 skill = 把含 SKILL.md 的文件夹丢进 skills 目录，自动挂载成一个工具 skill__<folder>。
- 渐进披露：平时 tools[] 里只暴露 frontmatter 的 description（成本低）；
  模型调用时 handler 才返回 SKILL.md 正文指令注入上下文。
- frontmatter 用极简手写解析（key: value），不引入 pyyaml 依赖。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from agent.tools.registry import Tool, ToolContext, ToolRegistry

log = logging.getLogger(__name__)


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """解析 `---` 围栏内的 key: value 元数据，返回 (meta, body)。无 frontmatter 则 meta 为空。"""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text.strip()
    meta: dict[str, str] = {}
    i = 1
    while i < len(lines) and lines[i].strip() != "---":
        line = lines[i]
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
        i += 1
    body = "\n".join(lines[i + 1:]).strip()
    return meta, body


def _ascii_slug(s: str) -> str:
    """工具名要 ascii（部分 provider 限制 [a-zA-Z0-9_-]）；用文件夹名最稳。"""
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", s.strip()).strip("-").lower()
    return slug or "skill"


def _make_loader(body: str, folder: Path):
    async def handler(ctx: ToolContext, args: dict) -> str:
        return (
            "【技能已加载，请按以下步骤执行】\n"
            f"{body}\n\n"
            f"（技能目录：{folder}，可读取其中的模板/脚本等附带文件）"
        )

    return handler


class SkillRegistry:
    def scan(self, root: Path, registry: ToolRegistry) -> int:
        """扫描 root 下每个含 SKILL.md 的子目录，注册成 skill__<folder> 工具。返回挂载数。"""
        if not root.exists():
            return 0
        n = 0
        for md in sorted(root.glob("*/SKILL.md")):
            try:
                meta, body = parse_frontmatter(md.read_text(encoding="utf-8"))
            except Exception as e:
                log.warning("skill %s 解析失败，跳过：%s", md, e)
                continue
            desc = (meta.get("description") or "").strip()
            if not desc:
                log.warning("skill %s 缺 description，跳过", md)
                continue
            display = meta.get("name") or md.parent.name
            registry.register(Tool(
                name=f"skill__{_ascii_slug(md.parent.name)}",
                description=f"[技能·{display}] {desc}",  # 渐进披露：只暴露描述
                parameters={"type": "object", "properties": {}},
                handler=_make_loader(body, md.parent),
                dangerous=False,
                source="skill",
            ))
            n += 1
        return n
