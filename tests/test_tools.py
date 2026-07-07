"""确定性测试：工具注册表 + schema 生成 + 工具循环 + 危险确认 + 发现/技能。

不调真 LLM：用 ScriptedToolRouter 预置每一轮的 AssistantTurn（含/不含 tool_calls）。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agent.gateway.router import AssistantTurn, LLMRouter
from agent.tools.builtin import discover_builtin
from agent.tools.loop import run_tool_loop
from agent.tools.registry import (
    ConfirmationRequired,
    Tool,
    ToolContext,
    ToolRegistry,
    tool,
)
from agent.tools.mcp_config import (
    cfg_to_entry,
    load_servers,
    normalize_mcp_json,
    save_server,
)
from agent.tools.schema import build_schema
from agent.tools.skills import SkillRegistry, parse_frontmatter
from agent.trust.confirm import ConfirmGate


# ---------- 假网关：按脚本逐轮返回 ----------

class ScriptedToolRouter(LLMRouter):
    def __init__(self, turns: list[AssistantTurn], *, final: str = "最终回复",
                 always: AssistantTurn | None = None) -> None:
        super().__init__("fake/none", "fake/none")
        self._turns = list(turns)
        self.final = final
        self.always = always          # 设了就每轮都返回它（测 max_steps）
        self.chat_calls = 0

    async def chat(self, messages, *, tools=None, task="default", **kwargs) -> AssistantTurn:
        self.chat_calls += 1
        if self.always is not None:
            return self.always
        return self._turns.pop(0) if self._turns else AssistantTurn(self.final)

    async def complete(self, messages, *, task="default", **kwargs) -> str:
        return self.final

    def live(self, task: str = "default") -> bool:
        return True


def _call(name: str, args: str = "{}", cid: str = "c1") -> AssistantTurn:
    return AssistantTurn(content="", tool_calls=[{"id": cid, "name": name, "arguments": args}])


def _ctx(store=None, confirm=None, router=None, persona=None) -> ToolContext:
    return ToolContext(store=store, memory=None, router=router, confirm=confirm, persona=persona)


async def _noop(ctx, args):
    return ""


# ---------- schema 生成 ----------

def test_build_schema_types_and_required():
    async def fn(ctx, query: str, k: int = 5):
        ...
    schema = build_schema(fn)
    assert schema["properties"] == {"query": {"type": "string"}, "k": {"type": "integer"}}
    assert schema["required"] == ["query"]          # 有默认值的 k 不必填；ctx 被跳过


def test_tool_decorator_builds_spec():
    @tool(dangerous=True)
    async def do_thing(ctx, x: str):
        "干一件事。"
        return x
    spec = do_thing._tool
    assert spec.name == "do_thing" and spec.dangerous and spec.description == "干一件事。"


# ---------- 注册表分发 ----------

async def test_dispatch_known_and_unknown():
    reg = ToolRegistry()

    async def handler(ctx, args):
        return f"hi {args.get('who','')}"

    reg.register(Tool("greet", "打招呼", {"type": "object"}, handler))
    assert await reg.dispatch("greet", {"who": "你"}, _ctx()) == "hi 你"
    assert "未知工具" in await reg.dispatch("nope", {}, _ctx())


async def test_dispatch_swallows_handler_error():
    reg = ToolRegistry()

    async def boom(ctx, args):
        raise ValueError("炸了")

    reg.register(Tool("boom", "", {"type": "object"}, boom))
    out = await reg.dispatch("boom", {}, _ctx())
    assert "执行失败" in out and "炸了" in out


async def test_dangerous_raises_confirmation():
    reg = ToolRegistry()

    async def handler(ctx, args):
        return "done"

    reg.register(Tool("send", "发消息", {"type": "object"}, handler, dangerous=True))
    with pytest.raises(ConfirmationRequired):
        await reg.dispatch("send", {"to": "老板"}, _ctx())
    # 获许可后放行
    assert await reg.dispatch("send", {}, _ctx(), allow_dangerous=True) == "done"


# ---------- 工具循环 ----------

async def test_loop_executes_tool_then_returns_text():
    reg = ToolRegistry()
    seen = {}

    async def handler(ctx, args):
        seen["args"] = args
        return "12:00"

    reg.register(Tool("clock", "时间", {"type": "object"}, handler))
    router = ScriptedToolRouter([_call("clock", '{"tz":"local"}')], final="现在 12:00")

    traces = []
    async def on_trace(step, call, result, ok, ms):
        traces.append((call["name"], result, ok))

    out = await run_tool_loop(router, reg, [{"role": "user", "content": "几点了"}], _ctx(),
                              on_trace=on_trace)
    assert out == "现在 12:00"
    assert seen["args"] == {"tz": "local"}          # 参数 JSON 被正确解析回 dict
    assert traces == [("clock", "12:00", True)]
    assert router.chat_calls == 2                    # 第一轮要工具，第二轮出文本


async def test_loop_no_tools_falls_back_to_complete():
    reg = ToolRegistry()                              # 空注册表
    router = ScriptedToolRouter([], final="普通回复")
    out = await run_tool_loop(router, reg, [{"role": "user", "content": "hi"}], _ctx())
    assert out == "普通回复"


async def test_loop_max_steps_guard():
    reg = ToolRegistry()

    async def handler(ctx, args):
        return "again"

    reg.register(Tool("loopy", "", {"type": "object"}, handler))
    # always：每轮都要求调工具 → 必须靠 max_steps 兜底收尾
    router = ScriptedToolRouter([], final="收尾", always=_call("loopy"))
    out = await run_tool_loop(router, reg, [{"role": "user", "content": "x"}], _ctx(), max_steps=3)
    assert out == "收尾"
    assert router.chat_calls == 3                     # 恰好 max_steps 次带工具调用


async def test_loop_dangerous_routes_to_confirm(store):
    reg = ToolRegistry()

    async def handler(ctx, args):
        return "已发送"

    reg.register(Tool("send_msg", "发消息给老板", {"type": "object"}, handler, dangerous=True))
    confirm = ConfirmGate(store)
    router = ScriptedToolRouter([_call("send_msg", '{"text":"请假"}')])
    out = await run_tool_loop(
        router, reg, [{"role": "user", "content": "帮我请假"}],
        _ctx(store=store, confirm=confirm),
    )
    assert "点头" in out                               # 返回的是确认话术
    pending = await store.latest_pending_action()
    assert pending and pending["action_type"] == "tool_call"


async def test_loop_returns_add_reminder_result_without_rewriting(store):
    reg = ToolRegistry()

    async def handler(ctx, args):
        return "设好了，今天 16:00 我提醒你：喝水。"

    reg.register(Tool("add_reminder", "提醒", {"type": "object"}, handler))
    router = ScriptedToolRouter([_call("add_reminder", '{"content":"喝水"}')], final="别用这个最终回复")

    out = await run_tool_loop(
        router, reg, [{"role": "user", "content": "4点提醒我喝水"}], _ctx(store=store)
    )

    assert out == "设好了，今天 16:00 我提醒你：喝水。"
    assert router.chat_calls == 1


async def test_loop_styles_add_reminder_result_with_persona(store):
    reg = ToolRegistry()

    async def handler(ctx, args):
        return "设好了，今天 16:00 我提醒你：喝水。"

    reg.register(Tool("add_reminder", "提醒", {"type": "object"}, handler))
    router = ScriptedToolRouter(
        [_call("add_reminder", '{"content":"喝水"}')],
        final="妥，今天 16:00 我盯着，喝水这事跑不掉 (๑•̀ㅂ•́)و",
    )

    out = await run_tool_loop(
        router, reg, [{"role": "user", "content": "4点提醒我喝水"}],
        _ctx(store=store, router=router, persona="说话俏皮、短。"),
    )

    assert out == "妥，今天 16:00 我盯着，喝水这事跑不掉 (๑•̀ㅂ•́)و"
    assert router.chat_calls == 1


# ---------- 原生工具发现 ----------

def test_discover_builtin_registers_core_tools():
    reg = ToolRegistry()
    n = discover_builtin(reg)
    assert n >= 4
    names = set(reg.names())
    assert {"get_time", "search_memory", "remember_fact", "add_reminder"} <= names


# ---------- 技能（渐进披露） ----------

def test_parse_frontmatter():
    meta, body = parse_frontmatter("---\nname: 测试\ndescription: 一段说明\n---\n正文内容")
    assert meta == {"name": "测试", "description": "一段说明"}
    assert body == "正文内容"


async def test_skill_scan_progressive_disclosure(tmp_path):
    d = tmp_path / "demo-skill"
    d.mkdir()
    (d / "SKILL.md").write_text(
        "---\nname: 演示\ndescription: 触发时这么做\n---\n步骤一\n步骤二", encoding="utf-8"
    )
    reg = ToolRegistry()
    assert SkillRegistry().scan(tmp_path, reg) == 1
    t = reg.get("skill__demo-skill")
    assert t is not None and t.source == "skill"
    # 渐进披露：tools[] 里只暴露描述，不含正文步骤
    assert "触发时这么做" in t.description and "步骤一" not in t.description
    # 调用时才返回正文
    body = await t.handler(_ctx(), {})
    assert "步骤一" in body and "步骤二" in body


def test_skill_scan_skips_when_no_description(tmp_path):
    d = tmp_path / "bad"
    d.mkdir()
    (d / "SKILL.md").write_text("---\nname: 没描述\n---\n正文", encoding="utf-8")
    reg = ToolRegistry()
    assert SkillRegistry().scan(tmp_path, reg) == 0   # 缺 description → 跳过


# ---------- MCP-B：懒加载 / 工具搜索 ----------

def test_lazy_policy_filters_and_search_tools_visibility():
    reg = ToolRegistry()
    reg.register(Tool("get_time", "时间", {"type": "object"}, _noop, source="builtin"))
    reg.register(Tool("search_tools", "搜工具", {"type": "object"}, _noop, source="builtin"))
    reg.register(Tool("mcp__ddg__search", "网页搜索", {"type": "object"}, _noop, source="mcp:ddg"))

    # 默认不超阈值、lazy_mcp=false → 全 eager；无懒工具 → search_tools 被隐藏
    names = {t["function"]["name"] for t in reg.to_openai_tools()}
    assert "mcp__ddg__search" in names and "search_tools" not in names

    # 开 lazy_mcp → mcp 工具转懒、不默认注入；search_tools 现身
    reg.set_lazy_policy(lazy_mcp=True, max_eager=25)
    names = {t["function"]["name"] for t in reg.to_openai_tools()}
    assert "mcp__ddg__search" not in names and "search_tools" in names and "get_time" in names

    # 搜索命中 → 激活 → 之后注入
    hits = reg.search_catalog("网页 搜索")
    assert hits and hits[0].name == "mcp__ddg__search"
    reg.activate([h.name for h in hits])
    assert "mcp__ddg__search" in {t["function"]["name"] for t in reg.to_openai_tools()}


def test_max_eager_auto_lazies_mcp():
    reg = ToolRegistry()
    reg.set_lazy_policy(lazy_mcp=False, max_eager=2)
    for i in range(3):
        reg.register(Tool(f"mcp__s__{i}", "d", {"type": "object"}, _noop, source="mcp:s"))
    assert reg.has_lazy()        # 总数 3 > max_eager 2 → 自动转懒


# ---------- MCP-A：超时 / 重连 / 截断 ----------

async def test_mcp_call_times_out():
    from agent.tools.mcp_client import MCPManager

    class SlowSession:
        async def call_tool(self, name, args):
            await asyncio.sleep(1)

    mgr = MCPManager([], timeout=0.05)
    mgr._sessions["s"] = SlowSession()
    mgr._cfgs["s"] = {}
    out = await mgr._call_with_recovery("s", "t", {})
    assert "超时" in out


async def test_mcp_call_reconnects_on_failure(monkeypatch):
    from agent.tools.mcp_client import MCPManager

    class FlakySession:
        def __init__(self, ok):
            self.ok = ok

        async def call_tool(self, name, args):
            if not self.ok:
                raise RuntimeError("connection closed")
            return type("R", (), {"content": [type("B", (), {"text": "ok"})()]})()

    mgr = MCPManager([], timeout=1)
    mgr._sessions["s"] = FlakySession(ok=False)
    mgr._cfgs["s"] = {}

    async def fake_reconnect(name):
        mgr._sessions[name] = FlakySession(ok=True)
        return True

    monkeypatch.setattr(mgr, "_reconnect", fake_reconnect)
    out = await mgr._call_with_recovery("s", "t", {})
    assert out == "ok"           # 第一次失败 → 重连 → 重试成功


def test_flatten_truncates_long_output():
    from agent.tools.mcp_client import _flatten

    res = type("R", (), {"content": [type("B", (), {"text": "x" * 9000})()]})()
    out = _flatten(res, limit=100)
    assert len(out) < 9000 and "截断" in out


# ---------- 新增原生工具：计算器 / 读文件 / 列提醒 ----------

async def test_calculate_tool():
    from agent.tools.builtin.calc import calculate
    assert (await calculate._tool.handler(_ctx(), {"expression": "(2+3)*4"})).endswith("= 20")


async def test_calculate_rejects_unsafe():
    from agent.tools.builtin.calc import calculate
    out = await calculate._tool.handler(_ctx(), {"expression": "__import__('os')"})
    assert "算不了" in out          # 函数调用/导入被拒，不执行任意代码


async def test_read_file_tool(tmp_path):
    from agent.tools.builtin.files import read_file
    f = tmp_path / "a.txt"
    f.write_text("你好世界", encoding="utf-8")
    assert await read_file._tool.handler(_ctx(), {"path": str(f)}) == "你好世界"
    miss = await read_file._tool.handler(_ctx(), {"path": str(tmp_path / "nope.txt")})
    assert "找不到" in miss


def test_discover_includes_new_native_tools():
    reg = ToolRegistry()
    discover_builtin(reg)
    names = set(reg.names())
    assert {"calculate", "read_file", "search_tools", "list_reminders", "install_mcp"} <= names


# ---------- MCP 传输路由（stdio / sse / http）----------

class _FakeCM:
    """假 async 上下文管理器：enter 返回预置的流元组。"""

    def __init__(self, ret):
        self.ret = ret

    async def __aenter__(self):
        return self.ret

    async def __aexit__(self, *a):
        return False


# ---------- 聊天里贴 JSON 装 MCP：归一化 + 持久化 + 工具守卫 ----------

def test_normalize_mcpservers_stdio():
    cfgs = normalize_mcp_json('{"mcpServers": {"ddg": {"command": "duckduckgo-mcp-server", "args": []}}}')
    assert cfgs == [{"name": "ddg", "enabled": True, "transport": "stdio",
                     "command": "duckduckgo-mcp-server", "args": []}]


def test_normalize_url_infers_http_and_keeps_headers():
    cfg = normalize_mcp_json('{"mcpServers": {"gh": {"url": "https://x/mcp", "headers": {"Authorization": "Bearer t"}}}}')[0]
    assert cfg["transport"] == "http" and cfg["url"] == "https://x/mcp"
    assert cfg["headers"] == {"Authorization": "Bearer t"}


def test_normalize_type_sse_respected():
    cfg = normalize_mcp_json('{"mcpServers": {"s": {"url": "https://x/sse", "type": "sse"}}}')[0]
    assert cfg["transport"] == "sse"


def test_normalize_single_server_form():
    cfg = normalize_mcp_json('{"name": "gh", "url": "https://x"}')[0]
    assert cfg["name"] == "gh" and cfg["transport"] == "http"


def test_normalize_rejects_no_command_or_url():
    with pytest.raises(ValueError):
        normalize_mcp_json('{"mcpServers": {"bad": {"foo": 1}}}')


async def test_add_server_persists(tmp_path, monkeypatch):
    from agent.tools.mcp_client import MCPManager

    store_path = str(tmp_path / "mcp.json")
    mgr = MCPManager([], store_path=store_path)

    async def fake_connect_one(cfg, registry):       # 不真连 server
        return [f"mcp__{cfg['name']}__t1"]

    monkeypatch.setattr(mgr, "_connect_one", fake_connect_one)
    monkeypatch.setattr(mgr, "_ensure_stack", lambda: True)

    names = await mgr.add_server({"name": "ddg", "transport": "stdio", "command": "x"}, ToolRegistry())
    assert names == ["mcp__ddg__t1"]
    saved = load_servers(store_path)                  # 存盘且读回（标准 mcpServers 格式）
    assert len(saved) == 1 and saved[0]["name"] == "ddg"


async def test_install_mcp_requires_runtime_handles():
    from agent.tools.builtin.mcp_admin import install_mcp

    # ctx 没有 registry/mcp（默认 None）→ 拒绝并提示，不炸
    out = await install_mcp._tool.handler(_ctx(), {"config_json": "{}"})
    assert "未启用" in out


# ---------- 单一真相源 data/mcp.json（标准 mcpServers 格式）----------

def test_load_servers_standard_format(tmp_path):
    f = tmp_path / "mcp.json"
    f.write_text('{"mcpServers": {"ddg": {"command": "duckduckgo-mcp-server"}, '
                 '"gh": {"url": "https://x/mcp", "type": "sse"}}}', encoding="utf-8")
    by = {s["name"]: s for s in load_servers(str(f))}
    assert by["ddg"]["transport"] == "stdio" and by["ddg"]["command"] == "duckduckgo-mcp-server"
    assert by["gh"]["transport"] == "sse" and by["gh"]["url"] == "https://x/mcp"


def test_cfg_to_entry_roundtrip():
    cfg = {"name": "gh", "transport": "http", "url": "https://x", "headers": {"A": "1"}}
    entry = cfg_to_entry(cfg)
    assert entry == {"url": "https://x", "headers": {"A": "1"}}     # http 不写 type
    cfg2 = {"name": "s", "transport": "sse", "url": "https://y"}
    assert cfg_to_entry(cfg2) == {"url": "https://y", "type": "sse"}


def test_save_server_writes_standard_and_merges(tmp_path):
    p = str(tmp_path / "mcp.json")
    save_server(p, {"name": "ddg", "transport": "stdio", "command": "x", "args": []})
    save_server(p, {"name": "gh", "transport": "http", "url": "https://x"})
    import json
    data = json.loads(Path(p).read_text(encoding="utf-8"))
    assert set(data["mcpServers"]) == {"ddg", "gh"}       # 合并而非覆盖整个文件
    assert data["mcpServers"]["ddg"] == {"command": "x"}  # args=[] 不写


def test_registry_remove_by_source():
    reg = ToolRegistry()
    reg.register(Tool("mcp__s__a", "", {"type": "object"}, _noop, source="mcp:s"))
    reg.register(Tool("mcp__s__b", "", {"type": "object"}, _noop, source="mcp:s"))
    reg.register(Tool("get_time", "", {"type": "object"}, _noop, source="builtin"))
    removed = reg.remove_by_source("mcp:s")
    assert set(removed) == {"mcp__s__a", "mcp__s__b"}
    assert reg.names() == ["get_time"]


# ---------- 热更新（reconcile / maybe_reload）----------

async def test_reconcile_adds_and_removes(tmp_path, monkeypatch):
    from agent.tools.mcp_client import MCPManager

    p = str(tmp_path / "mcp.json")
    save_server(p, {"name": "ddg", "transport": "stdio", "command": "x"})
    mgr = MCPManager([], store_path=p)
    reg = ToolRegistry()

    async def fake_connect_one(cfg, registry):
        registry.register(Tool(f"mcp__{cfg['name']}__t", "", {"type": "object"}, _noop, source=f"mcp:{cfg['name']}"))
        mgr._cfgs[cfg["name"]] = cfg
        return [f"mcp__{cfg['name']}__t"]

    monkeypatch.setattr(mgr, "_connect_one", fake_connect_one)
    monkeypatch.setattr(mgr, "_ensure_stack", lambda: True)

    # 文件里有 ddg，当前未连 → reconcile 应新增
    res = await mgr.reconcile(reg)
    assert res["added"] == ["ddg"] and "mcp__ddg__t" in reg.names()

    # 文件改成只剩 gh → ddg 下线、gh 新增
    save_server(p, {"name": "gh", "transport": "stdio", "command": "y"})
    import json
    data = json.loads(Path(p).read_text(encoding="utf-8"))
    del data["mcpServers"]["ddg"]
    Path(p).write_text(json.dumps(data), encoding="utf-8")
    res = await mgr.reconcile(reg)
    assert "ddg" in res["removed"] and "gh" in res["added"]
    assert "mcp__ddg__t" not in reg.names() and "mcp__gh__t" in reg.names()


async def test_maybe_reload_gates_on_mtime(tmp_path, monkeypatch):
    from agent.tools.mcp_client import MCPManager

    p = tmp_path / "mcp.json"
    p.write_text('{"mcpServers": {}}', encoding="utf-8")
    mgr = MCPManager([], store_path=str(p))   # __init__ 记下基线 mtime

    called = {"n": 0}
    async def fake_reconcile(registry):
        called["n"] += 1
        return {"added": [], "removed": [], "changed": []}
    monkeypatch.setattr(mgr, "reconcile", fake_reconcile)

    assert await mgr.maybe_reload(ToolRegistry()) is None   # 没变 → 不重载
    assert called["n"] == 0
    import os, time
    os.utime(p, (time.time() + 10, time.time() + 10))       # 改 mtime
    await mgr.maybe_reload(ToolRegistry())
    assert called["n"] == 1                                  # 变了 → 触发一次


def test_install_mcp_is_dangerous():
    from agent.tools.builtin.mcp_admin import install_mcp
    assert install_mcp._tool.dangerous is True       # 必须走确认门


async def test_mcp_open_streams_routes_by_transport(monkeypatch):
    from contextlib import AsyncExitStack

    import mcp.client.sse as sse_mod
    import mcp.client.stdio as stdio_mod
    import mcp.client.streamable_http as http_mod

    from agent.tools.mcp_client import MCPManager

    calls: dict = {}
    monkeypatch.setattr(stdio_mod, "stdio_client",
                        lambda params, **k: calls.__setitem__("stdio", params) or _FakeCM(("R", "W")))
    monkeypatch.setattr(sse_mod, "sse_client",
                        lambda url, headers=None, **k: calls.__setitem__("sse", (url, headers)) or _FakeCM(("R", "W")))
    # Streamable HTTP 返回三元组，_open_streams 要正确解包成 (read, write)
    monkeypatch.setattr(http_mod, "streamablehttp_client",
                        lambda url, headers=None, **k: calls.__setitem__("http", (url, headers)) or _FakeCM(("R", "W", None)))

    mgr = MCPManager([])
    mgr._stack = AsyncExitStack()
    try:
        assert await mgr._open_streams({"transport": "stdio", "command": "x", "args": []}) == ("R", "W")
        assert await mgr._open_streams(
            {"transport": "sse", "url": "http://s", "headers": {"A": "1"}}) == ("R", "W")
        assert calls["sse"] == ("http://s", {"A": "1"})
        assert await mgr._open_streams({"transport": "http", "url": "http://h"}) == ("R", "W")
        assert calls["http"] == ("http://h", None)            # 3 元组被解成 2
        assert await mgr._open_streams({"transport": "streamable_http", "url": "http://h2"}) == ("R", "W")
        with pytest.raises(ValueError):
            await mgr._open_streams({"transport": "carrier-pigeon"})
    finally:
        await mgr._stack.aclose()
