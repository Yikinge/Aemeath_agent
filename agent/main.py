"""Telegram 入口：对话(S1) + 记忆(S2/S3) + 人格(S4) + 编排/确认门(S5) + 主动引擎(S6) + 控制台(S7)。

被动回复走 Orchestrator.run_turn；主动消息由 APScheduler 定时心跳触发，推送到你的会话；
本地控制台跑在同一事件循环里（http://127.0.0.1:8765 默认），看/改/删记忆 + 审计。
运行：python -m agent.main
"""

from __future__ import annotations

import asyncio
import os
import threading
import time

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram.ext import Application

from agent.channels.telegram_bot import build_app
from agent.config import load_config
from agent.console.app import serve as serve_console
from agent.gateway.router import LLMRouter
from agent.memory.consolidator import Consolidator
from agent.memory.service import MemoryService
from agent.memory.store import Store
from agent.orchestration.loop import Orchestrator
from agent.persona.soul import load_soul
from agent.proactive.engine import ProactiveEngine
from agent.reminders import ReminderRuntime
from agent.tools.mcp_client import MCPManager
from agent.tools.mcp_config import load_servers
from agent.tools.registry import ToolContext
from agent.tools.setup import build_local_registry
from agent.trust.confirm import ConfirmGate

_HEARTBEAT_MINUTES = 15  # 多久跑一次主动心跳
_TG_HEALTH_INTERVAL_SEC = int(os.environ.get("TELEGRAM_HEALTH_INTERVAL_SEC", "60"))
_TG_HEALTH_MAX_FAILURES = int(os.environ.get("TELEGRAM_HEALTH_MAX_FAILURES", "3"))
_TG_HEALTH_TIMEOUT_SEC = int(os.environ.get("TELEGRAM_HEALTH_TIMEOUT_SEC", "20"))
_TG_PENDING_RESTART_SEC = int(os.environ.get("TELEGRAM_PENDING_RESTART_SEC", "600"))
_LOOP_WATCHDOG_INTERVAL_SEC = int(os.environ.get("LOOP_WATCHDOG_INTERVAL_SEC", "10"))
_LOOP_STALL_RESTART_SEC = int(os.environ.get("LOOP_STALL_RESTART_SEC", "180"))
_REMINDER_RECONCILE_INTERVAL_SEC = int(os.environ.get("REMINDER_RECONCILE_INTERVAL_SEC", "30"))


async def _ext_message_stub(payload: dict) -> str:
    return f"(模拟外发) 已发送给 {payload.get('to', '?')}：{payload.get('text', '')}"


def _env_enabled(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in {"0", "false", "no", "off"}


async def _telegram_health_watchdog(app: Application) -> None:
    """Restart the process when Telegram connectivity stays broken.

    Docker Compose has restart: unless-stopped, so exiting is the safest way to
    recover a stuck long-polling connection after proxy/network trouble.
    """
    if not _env_enabled("TELEGRAM_HEALTH_ENABLED", "1"):
        return
    failures = 0
    while True:
        await asyncio.sleep(_TG_HEALTH_INTERVAL_SEC)
        try:
            await asyncio.wait_for(app.bot.get_me(), timeout=_TG_HEALTH_TIMEOUT_SEC)
            info = await asyncio.wait_for(app.bot.get_webhook_info(), timeout=_TG_HEALTH_TIMEOUT_SEC)
            pending_count = int(info.pending_update_count or 0)
            last_update_at = float(app.bot_data.get("telegram_last_update_monotonic") or 0.0)
            stale_for = time.monotonic() - last_update_at if last_update_at else float("inf")
            if pending_count > 0 and stale_for >= _TG_PENDING_RESTART_SEC:
                stale_desc = "从未收到过消息" if last_update_at == 0 else f"{stale_for:.0f}s 未收到新消息"
                print(
                    "Telegram 轮询疑似卡住："
                    f"pending_update_count={pending_count}，{stale_desc}；"
                    "退出进程交给 Docker restart 策略自愈。",
                    flush=True,
                )
                os._exit(71)
            if failures:
                print("Telegram 健康检查恢复。", flush=True)
            failures = 0
        except Exception as exc:
            failures += 1
            print(
                f"Telegram 健康检查失败 {failures}/{_TG_HEALTH_MAX_FAILURES}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            if failures >= _TG_HEALTH_MAX_FAILURES:
                print("Telegram 连续不可达，退出进程交给 Docker restart 策略自愈。", flush=True)
                os._exit(70)


async def _loop_watchdog_pulse(state: dict[str, float]) -> None:
    while True:
        state["last_wall"] = time.time()
        await asyncio.sleep(max(1, _LOOP_WATCHDOG_INTERVAL_SEC))


def _start_loop_stall_watchdog(state: dict[str, float]) -> None:
    """Exit when the asyncio loop stops pulsing.

    Telegram reconnect checks run inside the same event loop as reminders. If
    that loop is wedged or the host wakes after a long pause, an external thread
    is the only in-process path that can notice and hand recovery to Docker.
    """
    if not _env_enabled("LOOP_WATCHDOG_ENABLED", "1"):
        return

    def watch() -> None:
        while True:
            time.sleep(max(1, _LOOP_WATCHDOG_INTERVAL_SEC))
            age = time.time() - state.get("last_wall", time.time())
            if age >= _LOOP_STALL_RESTART_SEC:
                print(
                    f"事件循环心跳停滞 {age:.0f}s，退出进程交给 Docker restart 策略自愈。",
                    flush=True,
                )
                os._exit(72)

    threading.Thread(target=watch, name="loop-stall-watchdog", daemon=True).start()


def main() -> None:
    cfg = load_config()
    store = Store(cfg.db_path)
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
    reminder_runtime = ReminderRuntime(store)

    # 工具/技能注册表（原生+技能同步装；MCP 在 post_init 异步连接）。
    # MCPManager 合并 config.toml + 运行时装的 server，并带 store_path 供 install_mcp 用。
    registry = None
    mcp_manager = None
    if cfg.tools_enabled:
        registry = build_local_registry(
            cfg.skills_dir, lazy_mcp=cfg.tools_lazy_mcp, max_eager=cfg.tools_max_eager,
        )
        servers = load_servers(cfg.mcp_config_path)   # 单一真相源 data/mcp.json
        mcp_manager = MCPManager(servers, store_path=cfg.mcp_config_path, timeout=cfg.mcp_timeout)

        async def _tool_executor(payload: dict) -> str:
            ns = payload.get("namespace", "default")
            tctx = ToolContext(store, memory, router, confirm, namespace=ns,
                               registry=registry, mcp=mcp_manager, reminders=reminder_runtime)
            return await registry.dispatch(
                payload["name"], payload.get("arguments", {}), tctx, allow_dangerous=True
            )

        confirm.register("tool_call", _tool_executor)

    orch = Orchestrator(
        store, router, memory, persona, confirm,
        registry=registry, tool_deny=set(cfg.tool_deny), mcp=mcp_manager,
        reminders=reminder_runtime, timezone=cfg.memory_timezone,
    )
    engine = ProactiveEngine(store, router, memory, persona, cfg.memory_timezone)

    async def on_message(
        text: str, user: str | None, chat_id: int,
        source_at: str | None = None, message_id: str | None = None,
    ) -> str:
        await store.kv_set("telegram_chat_id", str(chat_id))  # 记住往哪推主动消息
        return await orch.run_turn(text, source_at=source_at, message_id=message_id)

    async def post_init(app: Application) -> None:
        await store.init()
        # 启动异步：MCP 后台连接，不阻塞 bot 启动；工具边连边可用（worker task 持有生命周期）。
        # 总是起 worker（即便 0 server）：聊天里 install_mcp 也走 worker task。
        if mcp_manager is not None:
            await mcp_manager.start(registry)
        scheduler = AsyncIOScheduler()

        async def send_reminder(channel: str, target: str, message: str) -> None:
            if channel != "telegram":
                raise RuntimeError(f"unsupported reminder channel: {channel}")
            await app.bot.send_message(int(target), message)

        reminder_runtime.configure(send_fn=send_reminder, scheduler=scheduler)
        await reminder_runtime.recover()
        reminder_runtime.start_reconciler(interval_seconds=_REMINDER_RECONCILE_INTERVAL_SEC)
        asyncio.create_task(_telegram_health_watchdog(app))
        loop_watchdog_state = {"last_wall": time.time()}
        _start_loop_stall_watchdog(loop_watchdog_state)
        asyncio.create_task(_loop_watchdog_pulse(loop_watchdog_state))

        async def heartbeat() -> None:
            chat_id = await store.kv_get("telegram_chat_id")
            if not chat_id:
                return
            res = await engine.tick()
            if res["sent"]:
                await app.bot.send_message(int(chat_id), res["message"])

        async def nightly_consolidate() -> None:
            # 凌晨 02:00 Deep Dream：清 pending + 反思归纳洞察(A2) + 保守剪枝(C2)
            # + §9 兜底闭合超期未回应的 sent 承诺 + 重写 MEMORY.md
            from datetime import datetime, timedelta, timezone
            await memory.flush()
            await consolidator.reflect()
            await consolidator.prune()
            cutoff = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
            await consolidator.expire_stale_sent(cutoff)
            await consolidator.refresh_memory_md()

        scheduler.add_job(heartbeat, "interval", minutes=_HEARTBEAT_MINUTES)
        scheduler.add_job(nightly_consolidate, "cron", hour=2, minute=0)
        scheduler.start()

        if cfg.console_enabled:
            # 作为后台 task 与 Telegram 轮询共享同一个事件循环；进程退出时一起回收
            asyncio.create_task(
                serve_console(
                    store, memory, consolidator, cfg.console_host, cfg.console_port, registry=registry,
                )
            )
            print(f"控制台: http://{cfg.console_host}:{cfg.console_port}")

    app = build_app(cfg.telegram_token, cfg.allow_from, on_message, post_init=post_init)
    print(f"agent 启动：Telegram 轮询中，主动心跳每 {_HEARTBEAT_MINUTES} 分钟一次…（Ctrl+C 退出）")
    app.run_polling()


if __name__ == "__main__":
    main()
