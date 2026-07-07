"""确认门（S5，TRUST-2）：外发动作前置 pending-action，用户点头后才执行。

内部动作（读/整理记忆）直接放行；外部动作（发消息/预订/花钱）必须经此门。
真实外部工具以后 register 一个 executor 即可接入，业务流程不变。
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable

from agent.memory.models import new_id
from agent.memory.store import Store

Executor = Callable[[dict], Awaitable[str]]

_YES = {"确认", "确定", "好", "好的", "可以", "行", "同意", "发吧", "执行", "yes", "y", "ok"}
_NO = {"取消", "算了", "不", "不要", "不用", "别", "no", "n"}


def parse_decision(text: str) -> bool | None:
    """把用户回话解析成 确认(True)/取消(False)/无关(None)。"""
    t = text.strip().lower().rstrip("。.!！~")
    if t in _YES:
        return True
    if t in _NO:
        return False
    return None


class ConfirmGate:
    def __init__(self, store: Store) -> None:
        self.store = store
        self._executors: dict[str, Executor] = {}

    def register(self, action_type: str, fn: Executor) -> None:
        self._executors[action_type] = fn

    async def request(
        self, action_type: str, summary: str, payload: dict, namespace: str = "default"
    ) -> str:
        """登记一个待确认的外部动作，返回给用户的确认文案。"""
        await self.store.add_pending_action(
            new_id(), action_type, summary, json.dumps(payload, ensure_ascii=False), namespace
        )
        return f"⚠️ 这件事要发到外部，得你点头：{summary}\n回「确认」我就去做，回「取消」就算了。"

    async def resolve(self, pending: dict, approve: bool, namespace: str = "default") -> str:
        """用户表态后执行或取消，并写审计。"""
        await self.store.set_pending_status(pending["id"], "approved" if approve else "cancelled")
        if not approve:
            await self.store.add_audit(pending["action_type"], pending["summary"], "user:cancel")
            return "好，那就先不弄了。"

        payload = json.loads(pending["payload"]) if pending["payload"] else {}
        fn = self._executors.get(pending["action_type"])
        result = await fn(payload) if fn else "（已记录，但还没接对应执行器）"
        await self.store.add_audit(pending["action_type"], pending["summary"], "user:approve")
        return f"✅ 搞定：{pending['summary']}\n{result}"
