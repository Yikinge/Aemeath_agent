"""MCP 客户端（TOOL-2，drop-in #2）：config 驱动接入外部 MCP 服务器。

加一个 server = config 加一段 [[mcp.servers]]，它的工具全自动注册（命名空间 mcp__<server>__<tool>）。
- 三种传输：
    stdio           本地子进程，给 command/args（官方 server 多为这种）
    sse             连一个 URL（老的 HTTP 传输，正被官方淘汰，仍有存量 server 在用）
    http            连一个 URL（Streamable HTTP，2025-03 规范，远程/托管 server 的新标准）
  传输只决定"怎么通信"，注册/调用逻辑三种完全一致。
- 错误隔离：单个 server 连不上/调用失败都不拖垮 agent。
- 可靠性（MCP-A）：每次调用带超时（卡死的 server 不拖垮整轮）+ 失败自动重连重试 + 结果超长截断。
- 生命周期：session 进程级常驻（AsyncExitStack），退出时 aclose()。
- mcp SDK 未安装时整体跳过（保持"没装全依赖也能跑"）。
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import AsyncExitStack

from agent.tools.registry import Tool, ToolContext, ToolRegistry

log = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30.0      # 单次工具调用超时（秒）；卡死的 server 不该拖垮整轮对话
_MAX_RESULT_CHARS = 8000     # 工具结果回灌模型前的上限，防一次调用把上下文塞爆


def _flatten(result, limit: int = _MAX_RESULT_CHARS) -> str:
    """把 MCP call_tool 的 content blocks 摊平成文本，并对超长结果截断。"""
    blocks = getattr(result, "content", None) or []
    parts = []
    for c in blocks:
        text = getattr(c, "text", None)
        parts.append(text if text is not None else str(c))
    out = "\n".join(parts) if parts else "（工具无输出）"
    if len(out) > limit:
        out = out[:limit] + f"\n…[结果已截断，共 {len(out)} 字符，超出 {limit} 上限]"
    return out


class MCPManager:
    def __init__(
        self, servers: list[dict], store_path: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self.servers = servers or []
        self.store_path = store_path           # MCP 配置文件（data/mcp.json）：手编/聊天装/热更新都读写它
        self.timeout = timeout
        self._stack: AsyncExitStack | None = None
        self.connected: dict[str, int] = {}    # server name -> 接入工具数
        self._sessions: dict[str, object] = {} # server name -> 活的 ClientSession（重连时整体替换）
        self._cfgs: dict[str, dict] = {}       # server name -> cfg（重连/热更新要用）
        self._last_mtime = self._file_mtime()  # 配置文件 mtime 基线（热更新比对用）
        self._cmd_q: asyncio.Queue | None = None   # 命令队列：把 stack 操作委托给专属 worker task
        self._run_task: asyncio.Task | None = None

    def _file_mtime(self) -> float | None:
        try:
            return os.path.getmtime(self.store_path) if self.store_path else None
        except OSError:
            return None

    def _ensure_stack(self) -> bool:
        """确保 mcp SDK 可用并初始化 AsyncExitStack。返回是否可用。"""
        try:
            import mcp  # noqa: F401
        except ImportError:
            log.warning("未安装 mcp SDK（pip install mcp），跳过 MCP 接入")
            return False
        if self._stack is None:
            self._stack = AsyncExitStack()
        return True

    # ---------- 专属 worker：stack 的建/用/关都在同一 task（避开 anyio 跨任务报错） ----------

    async def start(self, registry: ToolRegistry) -> None:
        """非阻塞启动：建命令队列 + 专属 worker task，server 在后台连，对话立即可用。"""
        if not self._ensure_stack():
            return
        self._cmd_q = asyncio.Queue()
        self._run_task = asyncio.create_task(self._run(registry))

    async def _run(self, registry: ToolRegistry) -> None:
        """worker：连初始 server → 处理委托命令(增改/重连/热更新) → 收尾在本 task 内关栈。"""
        try:
            for cfg in [s for s in self.servers if s.get("enabled", True)]:
                try:
                    await self._connect_one(cfg, registry)
                except Exception as e:
                    log.warning("MCP %s 接入失败，跳过：%s", cfg.get("name", "?"), e)
            if self.connected:
                log.info("MCP 启动接入 %d 个 server / %d 个工具",
                         len(self.connected), sum(self.connected.values()))
            while True:
                fn, fut = await self._cmd_q.get()
                if fn is None:               # 关闭哨兵
                    break
                try:
                    fut.set_result(await fn())
                except Exception as e:
                    fut.set_exception(e)
        finally:
            if self._stack is not None:      # 与建栈/用栈同一 task，关闭无跨任务问题
                try:
                    await self._stack.aclose()
                except Exception as e:
                    log.debug("关闭 MCP 栈：%s", e)
                self._stack = None

    async def _submit(self, fn):
        """把一个会操作 stack 的协程交给 worker 执行（保证同任务）；没起 worker 则当前任务直接跑。"""
        if self._cmd_q is None:
            return await fn()
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        await self._cmd_q.put((fn, fut))
        return await fut

    async def connect_all(self, registry: ToolRegistry) -> None:
        """逐个连接、列举工具、注册代理。失败的 server 记日志后跳过。"""
        enabled = [s for s in self.servers if s.get("enabled", True)]
        if not enabled or not self._ensure_stack():
            return
        for cfg in enabled:
            try:
                await self._connect_one(cfg, registry)
            except Exception as e:  # 连接/列举失败 → 跳过该 server，不影响其余
                log.warning("MCP %s 接入失败，跳过：%s", cfg.get("name", "?"), e)

    async def _connect_one(self, cfg: dict, registry: ToolRegistry) -> list[str]:
        """连一个 server、列举并注册其工具。返回注册的工具名列表（失败抛异常）。"""
        from mcp import ClientSession

        name = cfg.get("name") or "server"
        read, write = await self._open_streams(cfg)
        session = await self._stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        self._sessions[name] = session
        self._cfgs[name] = cfg
        names: list[str] = []
        for t in (await session.list_tools()).tools:
            tname = f"mcp__{name}__{t.name}"
            registry.register(Tool(
                name=tname,
                description=t.description or t.name,
                parameters=t.inputSchema or {"type": "object", "properties": {}},
                handler=self._proxy(name, t.name),   # 代理按 name 查活 session，重连后自动指向新连接
                dangerous=self._is_dangerous(cfg, t.name),
                source=f"mcp:{name}",
            ))
            names.append(tname)
        self.connected[name] = len(names)
        log.info("MCP %s（%s）接入成功，注册 %d 个工具",
                 name, cfg.get("transport", "stdio"), len(names))
        return names

    async def add_server(self, cfg: dict, registry: ToolRegistry, *, persist: bool = True) -> list[str]:
        """运行时接入一个 server（聊天里贴 JSON 装的）：连接+注册，可选持久化。

        实际连接经 _submit 交 worker task 执行（同任务建栈）。失败抛异常，交 install_mcp 报告。
        """
        async def _do() -> list[str]:
            if not self._ensure_stack():
                raise RuntimeError("未安装 mcp SDK（pip install mcp）")
            registry.remove_by_source(f"mcp:{cfg.get('name')}")   # 同名覆盖：先清旧工具
            names = await self._connect_one(cfg, registry)
            self.servers = [s for s in self.servers if s.get("name") != cfg.get("name")] + [cfg]
            if persist and self.store_path:
                from agent.tools.mcp_config import save_server
                save_server(self.store_path, cfg)
                self._last_mtime = self._file_mtime()  # 自己写的，别触发热更新重复 reconcile
            return names

        return await self._submit(_do)

    # ---------- 启动异步 + 热更新 ----------

    async def maybe_reload(self, registry: ToolRegistry) -> dict | None:
        """配置文件变了就热重载（在每轮消息处理完成后调）。没变/无文件 → None。"""
        if not self.store_path:
            return None
        mtime = self._file_mtime()
        if mtime is None or mtime == self._last_mtime:
            return None
        self._last_mtime = mtime
        log.info("检测到 %s 变化，热重载 MCP", self.store_path)
        return await self.reconcile(registry)

    async def reconcile(self, registry: ToolRegistry) -> dict:
        """把已连接的 server 对齐到配置文件：新增的连上、删掉的下线、改了的重连。

        连接动作经 _submit 交 worker task 执行（同任务建栈）。
        """
        from agent.tools.mcp_config import load_servers

        async def _do() -> dict:
            if not self._ensure_stack():
                return {"added": [], "removed": [], "changed": []}
            desired = {s["name"]: s for s in load_servers(self.store_path) if s.get("enabled", True)}
            added, removed, changed = [], [], []

            for name in list(self._cfgs):
                if name not in desired:           # 被删 → 下线（旧 session 不主动关，进程退出回收）
                    registry.remove_by_source(f"mcp:{name}")
                    self._sessions.pop(name, None)
                    self._cfgs.pop(name, None)
                    self.connected.pop(name, None)
                    removed.append(name)

            for name, cfg in desired.items():
                if name not in self._cfgs:        # 新增
                    try:
                        await self._connect_one(cfg, registry)
                        added.append(name)
                    except Exception as e:
                        log.warning("热更新接入 %s 失败：%s", name, e)
                elif self._cfgs[name] != cfg:     # 改了 → 重连
                    registry.remove_by_source(f"mcp:{name}")
                    try:
                        await self._connect_one(cfg, registry)
                        changed.append(name)
                    except Exception as e:
                        log.warning("热更新重连 %s 失败：%s", name, e)

            if added or removed or changed:
                log.info("MCP 热更新：新增 %s 下线 %s 重连 %s", added, removed, changed)
            return {"added": added, "removed": removed, "changed": changed}

        return await self._submit(_do)

    async def _open_streams(self, cfg: dict):
        """按 transport 打开 (read, write) 流。三种传输在此分流，其余逻辑不区分。"""
        transport = cfg.get("transport", "stdio")
        if transport == "stdio":
            from mcp import StdioServerParameters
            from mcp.client.stdio import stdio_client
            params = StdioServerParameters(
                command=cfg["command"], args=cfg.get("args", []), env=cfg.get("env"),
            )
            read, write = await self._stack.enter_async_context(stdio_client(params))
            return read, write
        if transport == "sse":
            from mcp.client.sse import sse_client
            read, write = await self._stack.enter_async_context(
                sse_client(cfg["url"], headers=cfg.get("headers")),
            )
            return read, write
        if transport in ("http", "streamable_http", "streamable-http"):
            from mcp.client.streamable_http import streamablehttp_client
            # Streamable HTTP 返回三元组（多一个 get_session_id 回调），这里用不到
            read, write, _ = await self._stack.enter_async_context(
                streamablehttp_client(cfg["url"], headers=cfg.get("headers")),
            )
            return read, write
        raise ValueError(f"未知 transport：{transport!r}（支持 stdio/sse/http）")

    def _proxy(self, server_name: str, tool_name: str):
        """代理 handler：按 server_name 查活 session 调用，带超时 + 失败重连重试（MCP-A）。"""
        async def handler(ctx: ToolContext, args: dict) -> str:
            return await self._call_with_recovery(server_name, tool_name, args or {})

        return handler

    async def _call_with_recovery(self, name: str, tool_name: str, args: dict) -> str:
        session = self._sessions.get(name)
        if session is None:
            return f"[工具错误] MCP {name} 未连接"
        timeout = self._cfgs.get(name, {}).get("timeout", self.timeout)
        try:
            res = await asyncio.wait_for(session.call_tool(tool_name, args), timeout)
            return _flatten(res)
        except asyncio.TimeoutError:
            return f"[工具超时] {name}.{tool_name} 超过 {timeout}s 未返回，已跳过本次调用。"
        except Exception as e:  # 多半是连接断了 → 重连一次再重试（重连交 worker task）
            log.warning("MCP %s.%s 调用失败，尝试重连重试：%s", name, tool_name, e)
            if await self._submit(lambda: self._reconnect(name)):
                try:
                    res = await asyncio.wait_for(self._sessions[name].call_tool(tool_name, args), timeout)
                    return _flatten(res)
                except Exception as e2:
                    return f"[工具错误] {name}.{tool_name} 重连后仍失败：{e2}"
            return f"[工具错误] {name}.{tool_name} 调用失败、且重连未成功：{e}"

    async def _reconnect(self, name: str) -> bool:
        """重连一个掉线的 server：开新 session 替换 self._sessions[name]，已注册工具的代理自动指向新连接。

        旧 session 不主动 aclose（其上下文可能在别的 task 里建立，跨任务关闭会报错）；进程退出时统一回收。
        """
        cfg = self._cfgs.get(name)
        if cfg is None or self._stack is None:
            return False
        try:
            from mcp import ClientSession
            read, write = await self._open_streams(cfg)
            session = await self._stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self._sessions[name] = session
            log.info("MCP %s 重连成功", name)
            return True
        except Exception as e:
            log.warning("MCP %s 重连失败：%s", name, e)
            return False

    @staticmethod
    def _is_dangerous(cfg: dict, tool_name: str) -> bool:
        """哪些 MCP 工具算危险（走确认门）：可在 config 用 dangerous_tools 白名单，
        或 dangerous=true 把整个 server 标危险。默认放行（只读类）。"""
        if "dangerous_tools" in cfg:
            return tool_name in cfg["dangerous_tools"]
        return bool(cfg.get("dangerous", False))

    async def aclose(self) -> None:
        """优雅关闭：起了 worker 就发哨兵让它在自己 task 内关栈；否则（同步路径）直接关。"""
        if self._cmd_q is not None and self._run_task is not None:
            await self._cmd_q.put((None, None))    # 哨兵 → worker 收尾关栈
            try:
                await self._run_task
            except Exception:
                pass
            self._cmd_q = None
            self._run_task = None
        elif self._stack is not None:              # connect_all/--tools 同步路径：同任务直接关
            try:
                await self._stack.aclose()
            except Exception as e:
                log.warning("关闭 MCP 连接时出错（忽略）：%s", e)
            self._stack = None
