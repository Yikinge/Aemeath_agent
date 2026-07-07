"""本地 Web 控制台（S7，TRUST-1 看/改/删 + TRUST-3 可解释/审计）。

只在 127.0.0.1 上提供：单用户自托管，不做鉴权（出本机即非预期）。
JSON API + 内嵌 HTML 单页，全部走 MemoryService 契约，不直接碰存储。
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from agent.console.static import INDEX_HTML
from agent.memory.consolidator import Consolidator
from agent.memory.service import MemoryService
from agent.memory.store import Store


class FactPatch(BaseModel):
    value: str | None = None
    confidence: float | None = None


class ContradictionResolve(BaseModel):
    keep: str  # "new" / "old"


def create_app(
    store: Store, memory: MemoryService, consolidator: Consolidator, namespace: str = "default",
    registry=None,
) -> FastAPI:
    app = FastAPI(title="个人智能体 · 控制台", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return INDEX_HTML

    # ---------- 概览 ----------
    @app.get("/api/stats")
    async def stats() -> dict:
        facts = await memory.list_facts(namespace)
        narrs = await memory.list_narratives(namespace)
        mems = await memory.list_memories(namespace)
        cms = await store.list_all_commitments(namespace, status="open")
        pa = await store.list_pending_actions(namespace, status="pending")
        cd = await memory.pending_contradictions(namespace)
        used, quota = await store.budget_used_today(date.today().isoformat(), 0)
        pending = await store.pending_count(namespace)
        return {
            "facts": len(facts), "narratives": len(narrs), "memories": len(mems),
            "open_commitments": len(cms), "pending_actions": len(pa), "contradictions": len(cd),
            "proactive_used_today": used, "proactive_quota_today": quota,
            "pending_intake": pending,
        }

    # ---------- 画像事实 ----------
    @app.get("/api/facts")
    async def list_facts() -> list[dict]:
        return [asdict(f) for f in await memory.list_facts(namespace)]

    @app.patch("/api/facts/{fact_id}")
    async def edit_fact(fact_id: str, patch: FactPatch) -> dict:
        out = await memory.edit_fact(
            fact_id, value=patch.value, confidence=patch.confidence, actor="user"
        )
        if out is None:
            raise HTTPException(404, "fact not found")
        return asdict(out)

    @app.delete("/api/facts/{fact_id}")
    async def forget_fact(fact_id: str) -> dict:
        ok = await memory.forget(fact_id, actor="user")
        if not ok:
            raise HTTPException(404, "fact not found or already forgotten")
        return {"ok": True}

    # ---------- 语义记忆 ----------
    @app.get("/api/memories")
    async def list_mems() -> list[dict]:
        out = []
        for m in await memory.list_memories(namespace):
            d = asdict(m)
            d.pop("embedding", None)  # 向量不展示，省带宽
            out.append(d)
        return out

    @app.delete("/api/memories/{item_id}")
    async def forget_mem(item_id: str) -> dict:
        ok = await memory.forget_memory(item_id, actor="user")
        if not ok:
            raise HTTPException(404, "memory not found or already forgotten")
        return {"ok": True}

    # ---------- 承诺 / 主动消息 ----------
    @app.get("/api/commitments")
    async def commitments(status: str | None = None) -> list[dict]:
        return await store.list_all_commitments(namespace, status=status)

    @app.get("/api/proactive")
    async def proactive() -> list[dict]:
        return await store.list_proactive(namespace)

    # ---------- 待确认动作（TRUST-2） ----------
    @app.get("/api/pending-actions")
    async def pending_actions(status: str | None = "pending") -> list[dict]:
        items = await store.list_pending_actions(namespace, status=status)
        for it in items:
            if it.get("payload"):
                try:
                    it["payload"] = json.loads(it["payload"])
                except json.JSONDecodeError:
                    pass
        return items

    @app.post("/api/pending-actions/{aid}/cancel")
    async def cancel_action(aid: str) -> dict:
        pa = await store.get_pending_action(aid)
        if pa is None or pa["status"] != "pending":
            raise HTTPException(404, "action not pending")
        await store.set_pending_status(aid, "cancelled")
        await store.add_audit(pa["action_type"], pa["summary"], "console:cancel")
        return {"ok": True}

    # ---------- 矛盾队列（MEM-3） ----------
    @app.get("/api/contradictions")
    async def contradictions() -> list[dict]:
        out = []
        for c in await memory.pending_contradictions(namespace):
            new = await store.get_fact(c["new_fact_ref"])
            old = await store.get_fact(c["conflicting_fact_ref"])
            out.append({
                "id": c["id"], "created_at": c["created_at"],
                "key": (old.key if old else new.key if new else "?"),
                "new": asdict(new) if new else None,
                "old": asdict(old) if old else None,
            })
        return out

    @app.post("/api/contradictions/{cid}/resolve")
    async def resolve(cid: str, body: ContradictionResolve) -> dict:
        if body.keep not in ("new", "old"):
            raise HTTPException(400, "keep must be 'new' or 'old'")
        ok = await memory.resolve_contradiction(cid, body.keep, actor="user")
        if not ok:
            raise HTTPException(404, "contradiction not found")
        return {"ok": True}

    # ---------- 审计 / 历史 ----------
    @app.get("/api/audit")
    async def audit(n: int = 200) -> list[dict]:
        return await store.recent_audit(n)

    @app.get("/api/history")
    async def history(target_id: str | None = None, n: int = 200) -> list[dict]:
        return await memory.history(target_id=target_id, n=n)

    @app.get("/api/messages")
    async def messages(n: int = 50) -> list[dict]:
        return await store.recent_messages(n, namespace)

    # ---------- 工作记忆 MEMORY.md（注入 system prompt 的稳定 prefix） ----------
    @app.get("/api/working-memory")
    async def working_memory() -> dict:
        return {"content": consolidator.read_memory_md(), "path": str(consolidator.md_path)}

    @app.post("/api/working-memory/refresh")
    async def refresh_md() -> dict:
        text = await consolidator.refresh_memory_md(namespace)
        return {"ok": True, "length": len(text)}

    # ---------- 叙事笔记（人类可读层） ----------
    @app.get("/api/narratives")
    async def narratives(kind: str | None = None) -> list[dict]:
        return await store.list_narratives(namespace, kind=kind)

    @app.delete("/api/narratives/{nid}")
    async def forget_narrative(nid: str) -> dict:
        ok = await memory.forget_narrative(nid, actor="user")
        if not ok:
            raise HTTPException(404, "narrative not found or already forgotten")
        return {"ok": True}

    # ---------- PENDING + 手动巩固 ----------
    @app.get("/api/pending")
    async def pending() -> dict:
        return {
            "count": await store.pending_count(namespace),
            "items": await store.list_pending(namespace, n=50),
        }

    @app.post("/api/pending/consolidate")
    async def consolidate_now() -> dict:
        res = await memory.flush(namespace)
        await consolidator.refresh_memory_md(namespace)
        return {
            "ok": True,
            "added": len(res.added), "updated": len(res.updated),
            "memories": len(res.memories), "commitments": len(res.commitments),
            "contradictions": len(res.contradictions),
        }

    # ---------- 情绪 ----------
    @app.get("/api/mood")
    async def mood(n: int = 100) -> list[dict]:
        return await store.list_mood(namespace, n)

    # ---------- Trace（调试） ----------
    @app.get("/api/turn-trace")
    async def turn_trace(n: int = 30) -> list[dict]:
        return await store.list_turn_trace(namespace, n)

    @app.get("/api/tick-trace")
    async def tick_trace(n: int = 30) -> list[dict]:
        return await store.list_tick_trace(namespace, n)

    # ---------- 工具 / 技能（TOOL-4 可观测） ----------
    @app.get("/api/tools")
    async def tools() -> list[dict]:
        if registry is None:
            return []
        return [
            {"name": t.name, "description": t.description, "source": t.source,
             "dangerous": t.dangerous, "parameters": t.parameters}
            for t in registry.all()
        ]

    @app.get("/api/tool-trace")
    async def tool_trace(n: int = 100) -> list[dict]:
        return await store.list_tool_trace(namespace, n)

    return app


async def serve(
    store: Store, memory: MemoryService, consolidator: Consolidator,
    host: str = "127.0.0.1", port: int = 8787, namespace: str = "default", registry=None,
) -> None:
    """在已有事件循环里跑控制台（被 main.py 的 post_init 拉起来）。"""
    import uvicorn

    app = create_app(store, memory, consolidator, namespace=namespace, registry=registry)
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    await server.serve()
