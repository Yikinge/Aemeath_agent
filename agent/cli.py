"""本地 CLI：对话(S1) + 记忆(S2/S3) + 人格(S4) + 编排/确认门(S5) + 主动引擎(S6) + 控制台(S7)。

和 main.py（Telegram）共用同一套 Orchestrator / MemoryService / ProactiveEngine / Consolidator。

  交互对话:   python -m agent.cli
  脚本演示:   python -m agent.cli --script
  查看记忆:   python -m agent.cli --facts            # 画像 / 叙事 / 承诺 / 关系 / pending
  检索测试:   python -m agent.cli --recall "猫"
  手动巩固:   python -m agent.cli --consolidate      # 把 pending 全跑完 + 重写 MEMORY.md
  工作记忆:   python -m agent.cli --memory-md        # 打印当前 MEMORY.md（注入 system prompt 的内容）
  记忆台账:   python -m agent.cli --snapshot         # 导出/打印 agent.db 全量记忆 → data/agent_db.md
  每日记忆:   python -m agent.cli --journal [日期]    # 查看 data/journal/YYYY-MM-DD.md
  重建日记:   python -m agent.cli --rebuild-journal 2026-07-02
  主动心跳:   python -m agent.cli --tick [--force]   # 看她现在会不会主动找你、说什么
  工具清单:   python -m agent.cli --tools            # 列出已注册的原生/MCP/技能工具
  确认门演示: python -m agent.cli --demo-confirm
  本地控制台: python -m agent.cli --console          # 看/改/删记忆 + 审计（S7）
"""

from __future__ import annotations

import asyncio
import sys

from agent.config import load_config
from agent.gateway.router import LLMRouter
from agent.memory.consolidator import Consolidator
from agent.memory.service import MemoryService
from agent.memory.store import Store
from agent.orchestration.loop import Orchestrator
from agent.persona.soul import load_soul
from agent.proactive.engine import ProactiveEngine
from agent.tools.mcp_client import MCPManager
from agent.tools.mcp_config import load_servers
from agent.tools.registry import ToolContext
from agent.tools.setup import build_local_registry
from agent.trust.confirm import ConfirmGate


async def _ext_message_stub(payload: dict) -> str:
    return f"(模拟外发) 已发送给 {payload.get('to', '?')}：{payload.get('text', '')}"


async def _print_facts(memory: MemoryService, store: Store) -> None:
    facts = await memory.list_facts()
    print(f"\n画像事实 {len(facts)} 条：")
    for f in facts:
        print(f"  [{f.category}] {f.key} = {f.value}")
    narrs = await memory.list_narratives()
    print(f"叙事笔记 {len(narrs)} 条：")
    for n in narrs:
        print(f"  · [{n['kind']}] {n['content']}")
    chunks = await memory.list_memories()
    print(f"向量碎片 {len(chunks)} 条（机器索引）")
    moods = await memory.list_mood(n=5)
    if moods:
        print(f"最近情绪 {len(moods)} 条：")
        for m in moods:
            v = f"{m['valence']:+.2f}" if m['valence'] is not None else "—"
            a = f"{m['arousal']:.2f}" if m['arousal'] is not None else "—"
            print(f"  · {m['ts']}  v={v} a={a}  {m['signals']}  {m['note'] or ''}")
    commits = await store.list_all_commitments(status="open")  # 全部未闭合承诺
    print(f"待跟进承诺 {len(commits)} 条：")
    for c in commits:
        when = f" @ {c['event_at']}" if c.get("event_at") else ""
        print(f"  ({c['kind']}) {c['content']}{when}")
    pending = await store.pending_count()
    if pending:
        print(f"⚠ 还有 {pending} 条 user 文本在 pending（运行 --consolidate 立即巩固）")


async def _print_recall(memory: MemoryService, query: str) -> None:
    print(f"\nquery「{query}」召回：")
    hits = await memory.retrieve(query, k=5)
    for h in hits or []:
        print(f"  {h.score:.3f}  {h.item.content}")
    if not hits:
        print("  （无相关记忆）")


async def _amain() -> None:
    cfg = load_config()
    store = Store(cfg.db_path)
    await store.init()
    router = LLMRouter(
        cfg.default_model, cfg.fast_model,
        embed_model=cfg.embed_model, embed_base_url=cfg.embed_base_url, embed_api_key=cfg.embed_api_key,
    )
    consolidator = Consolidator(
        store, router, cfg.memory_md_path,
        timezone=cfg.memory_timezone, similarity_threshold=cfg.narrative_similarity_threshold,
        core_max_commitments=cfg.core_max_commitments, recent_mood_days=cfg.recent_mood_days,
    )
    memory = MemoryService(store, router, consolidator, cfg.consolidate_threshold)
    persona = load_soul(cfg.soul_path, cfg.system_prompt)
    confirm = ConfirmGate(store)
    confirm.register("external_message", _ext_message_stub)

    # 工具/技能注册表（MCP 留到交互/脚本前再连，省短命命令的启动开销）。
    # MCPManager 早建好（不连接，零开销）：合并 config.toml + 运行时装的 server，供 install_mcp 运行时用。
    registry = None
    mcp_manager: MCPManager | None = None
    if cfg.tools_enabled:
        registry = build_local_registry(
            cfg.skills_dir, lazy_mcp=cfg.tools_lazy_mcp, max_eager=cfg.tools_max_eager,
        )
        servers = load_servers(cfg.mcp_config_path)   # 单一真相源 data/mcp.json
        mcp_manager = MCPManager(servers, store_path=cfg.mcp_config_path, timeout=cfg.mcp_timeout)

        async def _tool_executor(payload: dict) -> str:
            ns = payload.get("namespace", "default")
            tctx = ToolContext(store, memory, router, confirm, namespace=ns,
                               registry=registry, mcp=mcp_manager)
            return await registry.dispatch(
                payload["name"], payload.get("arguments", {}), tctx, allow_dangerous=True
            )

        confirm.register("tool_call", _tool_executor)

    orch = Orchestrator(
        store, router, memory, persona, confirm,
        registry=registry, tool_deny=set(cfg.tool_deny), mcp=mcp_manager,
        timezone=cfg.memory_timezone,
    )
    engine = ProactiveEngine(store, router, memory, persona, cfg.memory_timezone)

    print(f"[模型] {cfg.default_model}  [向量] {cfg.embed_model if router.embed_live() else '降级'}  "
          f"[人格] {'SOUL.md' if persona != cfg.system_prompt else 'config'}  "
          f"[工具] {len(registry.all()) if registry else 0}")

    if "--tools" in sys.argv:
        # 临时连一下 MCP，好把清单列全（其余短命命令不连，省启动）
        if mcp_manager is not None:
            await mcp_manager.connect_all(registry)
        if not registry or not registry.all():
            print("（没有已注册的工具；检查 config 的 [tools].enabled 与 agent/skills/）")
        else:
            eager = registry.to_openai_tools(deny=set(cfg.tool_deny))
            print(f"\n已注册工具 {len(registry.all())} 个（默认注入 {len(eager)} 个，懒加载的用到才注入）：")
            for t in registry.all():
                flag = " ⚠危险" if t.dangerous else ""
                lazy = " 💤懒" if t.lazy else ""
                print(f"  [{t.source}] {t.name}{flag}{lazy} — {t.description}")
        if mcp_manager is not None:
            await mcp_manager.aclose()
        await store.close()
        return
    if "--facts" in sys.argv:
        await _print_facts(memory, store)
        await store.close()
        return
    if "--recall" in sys.argv:
        i = sys.argv.index("--recall")
        await _print_recall(memory, sys.argv[i + 1] if i + 1 < len(sys.argv) else "")
        await store.close()
        return
    if "--tick" in sys.argv:
        res = await engine.tick(force="--force" in sys.argv)
        if res["sent"]:
            print(f"\n爱弥斯(主动) < {res['message']}\n[reason] {res['reason']}")
        else:
            print(f"\n[不打扰] {res['reason']}")
        await store.close()
        return
    if "--consolidate" in sys.argv:
        from datetime import datetime, timedelta, timezone
        before = await store.pending_count()
        res = await memory.flush()
        insights = await consolidator.reflect()      # A2 反思
        pruned = await consolidator.prune()          # C2 剪枝
        cutoff = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        await consolidator.expire_stale_sent(cutoff) # §9 兜底闭合超期 sent
        await consolidator.refresh_memory_md()
        print(f"\n巩固完成（pending {before} → 0）：{res.summary()}")
        if insights:
            print(f"反思洞察 +{len(insights)}：")
            for ins in insights:
                print(f"  · {ins}")
        if pruned:
            print(f"剪枝 {len(pruned)} 条（强度低、不重要、久未用）")
        print(f"MEMORY.md → {cfg.memory_md_path}")
        await store.close()
        return
    if "--memory-md" in sys.argv:
        text = consolidator.read_memory_md()
        print(text if text else "(MEMORY.md 还没生成，先 --consolidate 或聊几句)")
        await store.close()
        return
    if "--snapshot" in sys.argv:
        text = await consolidator.write_db_snapshot()
        print(text)
        print(f"→ 已写入 {consolidator.snapshot_path}")
        await store.close()
        return
    if "--journal" in sys.argv:
        i = sys.argv.index("--journal")
        date = sys.argv[i + 1] if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--") else None
        text = consolidator.journal.read_day(date)
        print(text if text else "(当天每日记忆还没有生成)")
        print(f"→ {consolidator.journal.path_for(date)}")
        await store.close()
        return
    if "--rebuild-journal" in sys.argv:
        i = sys.argv.index("--rebuild-journal")
        date = sys.argv[i + 1] if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--") else None
        if not date:
            from datetime import datetime
            date = datetime.now().strftime("%Y-%m-%d")
        text = await consolidator.journal.rebuild_day(store, date)
        print(text)
        print(f"→ 已重建 {consolidator.journal.path_for(date)}")
        await store.close()
        return
    if "--demo-confirm" in sys.argv:
        prompt = await confirm.request(
            "external_message", "发条消息给老板：今天身体不舒服，想请个假", {"to": "老板", "text": "今天请假"}
        )
        print(f"\n爱弥斯 < {prompt}")
        print("你 > 确认")
        pending = await store.latest_pending_action()
        print(f"爱弥斯 < {await confirm.resolve(pending, True)}")
        await store.close()
        return
    if "--console" in sys.argv:
        # 单独跑控制台：浏览器开 http://host:port，Ctrl+C 退出
        from agent.console.app import serve as serve_console

        print(f"控制台启动：http://{cfg.console_host}:{cfg.console_port}  （Ctrl+C 退出）")
        try:
            await serve_console(
                store, memory, consolidator, cfg.console_host, cfg.console_port, registry=registry,
            )
        finally:
            await store.close()
        return

    # 启动异步：MCP 后台连接，不阻塞——对话立即可用，工具边连边出现（循环每轮重算 tools[]）。
    # 总是起 worker（即便 0 server）：聊天里 install_mcp 也走 worker task，避免跨任务建栈。
    if mcp_manager is not None:
        await mcp_manager.start(registry)
        if mcp_manager.servers:
            print(f"[MCP] 后台连接 {len(mcp_manager.servers)} 个 server 中…（不影响对话）")

    if "--script" in sys.argv:
        for text in ["你好呀", "我叫小王，养了只橘猫叫煤球", "唉今天上班被领导骂了，好累", "下周三我有个重要面试"]:
            print(f"\n你 > {text}")
            print(f"爱弥斯 < {await orch.run_turn(text)}")
        await orch.drain()
        await _print_facts(memory, store)
        print("\n--- 主动心跳（force 跳过冷却）---")
        res = await engine.tick(force=True)
        print(f"爱弥斯(主动) < {res['message']}\n[reason] {res['reason']}" if res["sent"]
              else f"[不打扰] {res['reason']}")
    else:
        print("进入对话（输入 exit 退出）")
        loop = asyncio.get_event_loop()
        while True:
            try:
                text = (await loop.run_in_executor(None, input, "\n你 > ")).strip()
            except (EOFError, KeyboardInterrupt):
                break
            if text in {"exit", "quit", ":q"}:
                break
            if not text:
                continue
            print(f"爱弥斯 < {await orch.run_turn(text)}")
        await orch.drain()

    if mcp_manager is not None:
        await mcp_manager.aclose()   # 发哨兵让 worker 在自己 task 内关栈（无跨任务报错）
    await store.close()


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
