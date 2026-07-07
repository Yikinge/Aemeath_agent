"""测试夹具：临时 SQLite Store + 一个不联网的 FakeRouter。

确定性测试不调真 LLM：FakeRouter.complete 返回预置的脚本化响应（按调用顺序或默认），
embed 走内置兜底向量。这样 ingest→consolidate→retrieve 整条链都能在 CI 里稳定跑。
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from agent.gateway.router import LLMRouter
from agent.memory.store import Store


@pytest_asyncio.fixture
async def store(tmp_path):
    s = Store(str(tmp_path / "test.db"))
    await s.init()
    yield s
    await s.close()


class FakeRouter(LLMRouter):
    """可编程的假网关：complete 返回 scripted 队列里的下一条（用于 mock 抽取/反思的 JSON）。

    embed 继承父类的兜底哈希向量（不联网）。scripted 用完后返回 "[]"（安全空结果）。
    """

    def __init__(self, scripted: list[str] | None = None) -> None:
        super().__init__("fake/none", "fake/none")
        self._scripted = list(scripted or [])

    async def complete(self, messages, *, task: str = "default", **kwargs) -> str:
        return self._scripted.pop(0) if self._scripted else "[]"

    def live(self, task: str = "default") -> bool:
        return True


@pytest.fixture
def router():
    """默认假网关（抽取一律返回空，用于只测确定性逻辑的场景）。"""
    return FakeRouter()


class PromptRouter(LLMRouter):
    """按 prompt 内容路由的假网关：一次 consolidate 里会发生多次 LLM 调用（画像/路由/承诺/情绪/
    判定/合并），位置脚本队列脆弱，故按 system/user 文本判别各返回对应的预置响应。

    embed 继承父类兜底哈希向量（确定性）；live=True 让 resolve 的 judge/merge 走真分支。
    """

    def __init__(self, *, profile: str = "[]", route: str = "[]", commitments: str = "[]",
                 mood: str | None = None, verdict: str = "NEW", merged: str | None = None) -> None:
        super().__init__("fake/none", "fake/none")
        self.profile, self.route, self.commitments = profile, route, commitments
        self.mood, self.verdict, self.merged = mood, verdict, merged
        self.calls: list[str] = []

    async def complete(self, messages, *, task: str = "default", **kwargs) -> str:
        sys = next((m["content"] for m in messages if m.get("role") == "system"), "")
        usr = next((m["content"] for m in messages if m.get("role") == "user"), "")
        self.calls.append(sys + "\n" + usr)
        if "画像抽取器" in sys:
            return self.profile
        if "记忆路由器" in sys:
            return self.route
        if "需要以后主动跟进" in sys:
            return self.commitments
        if "情绪状态" in sys:
            return self.mood if self.mood is not None else '{"valence":null,"arousal":null,"signals":[],"note":""}'
        if "合并成一" in usr:               # 叙事(合并成一句) / 画像实体(合并成一条)
            return self.merged or "MERGED"
        if "SAME / MERGE / UPDATE / EXPIRE / NEW" in usr:   # 叙事 resolve 判定
            return self.verdict
        if "SAME / UPDATE / CONFLICT" in usr:                # 画像 resolve 判定
            return self.verdict
        return "[]"

    def live(self, task: str = "default") -> bool:
        return True
