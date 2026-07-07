"""工具注册表 + 统一契约（TOOL-0）。

- Tool：一个能力的统一表示 {name, description, parameters(JSON Schema), handler, dangerous, source}
- ToolContext：每次调用注入的依赖（store/memory/router/confirm + namespace）
- ToolRegistry：登记、产出 OpenAI tools[]、按名分发
- @tool：把一个 async 函数登记成原生工具（从签名自动生成 schema）

三个来源（原生/MCP/Skill）都把自己包成 Tool 注册进来，dispatch 统一执行。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agent.tools.schema import build_schema, first_line

if TYPE_CHECKING:  # 仅类型注解，避免运行时循环导入
    from agent.gateway.router import LLMRouter
    from agent.memory.service import MemoryService
    from agent.memory.store import Store
    from agent.reminders.runtime import ReminderRuntime
    from agent.tools.mcp_client import MCPManager
    from agent.trust.confirm import ConfirmGate


@dataclass
class ToolContext:
    """每次工具调用注入的依赖与作用域。原生工具通过它访问记忆/存储等。

    registry / mcp 让"管理类"工具（如 install_mcp）能在运行时改注册表、连新 server。
    """

    store: "Store"
    memory: "MemoryService"
    router: "LLMRouter"
    confirm: "ConfirmGate | None" = None
    namespace: str = "default"
    registry: "ToolRegistry | None" = None
    mcp: "MCPManager | None" = None
    reminders: "ReminderRuntime | None" = None
    persona: str | None = None


# 统一的工具执行签名：拿到上下文 + 参数 dict，返回文本结果（喂回模型）
Handler = Callable[[ToolContext, dict], Awaitable[str]]


@dataclass
class Tool:
    name: str                       # 唯一；MCP/Skill 带前缀防撞
    description: str                # 进 tools[]，模型据此决定是否调用
    parameters: dict                # JSON Schema（object）
    handler: Handler                # async (ctx, args) -> 文本
    dangerous: bool = False         # True → 执行前走 ConfirmGate
    source: str = "builtin"         # builtin / mcp:<server> / skill
    lazy: bool = False              # True → 不默认注入 tools[]，由 search_tools 按需激活（MCP-B 规模治理）


class ConfirmationRequired(Exception):
    """危险工具被调用、但尚未经用户确认时抛出；工具循环据此转确认门。"""

    def __init__(self, name: str, arguments: dict, summary: str) -> None:
        super().__init__(summary)
        self.name = name
        self.arguments = arguments
        self.summary = summary


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._activated: set[str] = set()    # 被 search_tools 激活的懒工具
        self._lazy_mcp = False               # 懒加载策略（MCP-B）
        self._max_eager = 25                 # 工具总数超过此值则把 MCP 工具转懒

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        if tool.source.startswith("mcp"):    # 运行时 install_mcp 新增的也即时套用懒策略
            self._reapply_lazy()

    def remove_by_source(self, source: str) -> list[str]:
        """移除某来源的全部工具（热更新下线一个 MCP server 时用）。返回被移除的工具名。"""
        names = [n for n, t in self._tools.items() if t.source == source]
        for n in names:
            del self._tools[n]
        self._activated -= set(names)
        return names

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def names(self) -> list[str]:
        return list(self._tools)

    # ---------- 懒加载 / 工具搜索（MCP-B 规模治理） ----------

    def set_lazy_policy(self, lazy_mcp: bool, max_eager: int) -> None:
        self._lazy_mcp = lazy_mcp
        self._max_eager = max_eager
        self._reapply_lazy()

    def _reapply_lazy(self) -> None:
        """按策略给 MCP 来源的工具打/清 lazy：显式开启、或工具总数超阈值则转懒。"""
        should = self._lazy_mcp or len(self._tools) > self._max_eager
        for t in self._tools.values():
            if t.source.startswith("mcp"):
                t.lazy = should
        self._activated &= set(self._tools)  # 清掉已不存在的激活项

    def activate(self, names: list[str]) -> None:
        self._activated.update(n for n in names if n in self._tools)

    def search_catalog(self, query: str, limit: int = 8) -> list[Tool]:
        """在懒工具里按关键词检索（名字 + 描述子串/词命中），给 search_tools 用。"""
        q = query.lower().strip()
        words = [w for w in q.split() if w]
        scored: list[tuple[int, Tool]] = []
        for t in self._tools.values():
            if not t.lazy:
                continue
            hay = (t.name + " " + t.description).lower()
            score = (3 if q and q in hay else 0) + sum(1 for w in words if w in hay)
            if score:
                scored.append((score, t))
        scored.sort(key=lambda x: -x[0])
        return [t for _, t in scored[:limit]]

    def has_lazy(self) -> bool:
        return any(t.lazy for t in self._tools.values())

    def to_openai_tools(self, deny: set[str] | None = None) -> list[dict]:
        """产出注入给模型的 tools[]：含 eager 工具 + 已激活的 lazy 工具，排除 deny。

        懒工具（默认 MCP）未激活则不注入，省上下文；没有任何懒工具时连 search_tools 也不占位。
        """
        deny = deny or set()
        has_lazy = self.has_lazy()
        out = []
        for t in self._tools.values():
            if t.name in deny:
                continue
            if t.lazy and t.name not in self._activated:
                continue
            if t.name == "search_tools" and not has_lazy:
                continue                      # 没有懒工具 → search_tools 无意义，别占位
            out.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            })
        return out

    async def dispatch(
        self, name: str, arguments: dict, ctx: ToolContext, *, allow_dangerous: bool = False,
    ) -> str:
        """按名执行。未知工具/异常 → 返回错误文本（喂回模型，不崩 agent）。
        危险工具未获许可 → 抛 ConfirmationRequired 交由上层走确认门。"""
        tool = self._tools.get(name)
        if tool is None:
            return f"[工具错误] 未知工具：{name}"
        if tool.dangerous and not allow_dangerous:
            raise ConfirmationRequired(name, arguments, _summarize(tool, arguments))
        try:
            return await tool.handler(ctx, arguments or {})
        except ConfirmationRequired:
            raise
        except Exception as e:  # 单个工具失败不该让整轮崩
            return f"[工具错误] {name} 执行失败：{e}"


def _summarize(tool: Tool, arguments: dict) -> str:
    args = "，".join(f"{k}={v}" for k, v in (arguments or {}).items()) or "（无参数）"
    return f"调用工具 {tool.name}（{args}）"


def tool(
    _fn: Callable | None = None, *,
    name: str | None = None,
    description: str | None = None,
    dangerous: bool = False,
    parameters: dict | None = None,
):
    """把一个 `async def fn(ctx, **params)` 登记成原生工具。

    - name：默认取函数名
    - description：默认取 docstring 首行
    - parameters：默认从签名自动生成（跳过 ctx）；复杂场景可显式传
    - dangerous：有外部副作用（写/发/花钱）置 True，执行前走确认门

    用法：
        @tool(dangerous=False)
        async def get_time(ctx) -> str:
            "返回当前时间。"
            ...
    被装饰的函数会带上 `_tool` 属性；discover 时收集注册。
    """

    def deco(fn: Callable):
        async def handler(ctx: ToolContext, args: dict) -> str:
            return await fn(ctx, **(args or {}))

        fn._tool = Tool(  # type: ignore[attr-defined]
            name=name or fn.__name__,
            description=description or first_line(fn.__doc__),
            parameters=parameters if parameters is not None else build_schema(fn),
            handler=handler,
            dangerous=dangerous,
            source="builtin",
        )
        return fn

    return deco(_fn) if _fn is not None else deco
