"""确定性测试：精确提醒从 commitment 拆出，走 reminder_job + runtime。"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from agent.gateway.router import LLMRouter
from agent.reminders import ReminderRuntime
from agent.tools.builtin.commitments import add_reminder, parse_reminder_request
from agent.tools.registry import ToolContext


class CaptureSender:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str, str]] = []

    async def __call__(self, channel: str, target: str, message: str) -> None:
        self.sent.append((channel, target, message))


class StyleRouter(LLMRouter):
    def __init__(self, reply: str) -> None:
        super().__init__("fake/none", "fake/none")
        self.reply = reply

    async def complete(self, messages, *, task: str = "default", **kwargs) -> str:
        return self.reply

    def live(self, task: str = "default") -> bool:
        return True


async def test_add_reminder_with_exact_time_creates_reminder_job(store):
    await store.kv_set("telegram_chat_id", "12345")
    ctx = ToolContext(store=store, memory=None, router=None)

    out = await add_reminder(ctx, content="喝水", at="明天上午9点")

    assert "设好了" in out
    jobs = await store.list_reminder_jobs()
    assert len(jobs) == 1
    assert jobs[0]["title"] == "喝水"
    assert jobs[0]["delivery_target"] == "12345"
    assert jobs[0]["status"] == "scheduled"
    assert await store.list_all_commitments(status="open") == []


async def test_add_reminder_with_relative_duration_creates_reminder_job(store):
    await store.kv_set("telegram_chat_id", "12345")
    ctx = ToolContext(store=store, memory=None, router=None)

    out = await add_reminder(ctx, content="学习", at="2分钟后")

    assert "设好了" in out
    jobs = await store.list_reminder_jobs()
    assert len(jobs) == 1
    assert jobs[0]["title"] == "学习"


async def test_add_reminder_styles_delivery_message_with_persona(store):
    await store.kv_set("telegram_chat_id", "12345")
    router = StyleRouter("到点啦，喝水这事别装没看见 (๑•̀ㅂ•́)و")
    ctx = ToolContext(store=store, memory=None, router=router, persona="说话俏皮、短。")

    await add_reminder(ctx, content="喝水", at="明天上午9点")

    jobs = await store.list_reminder_jobs()
    assert jobs[0]["message"] == "到点啦，喝水这事别装没看见 (๑•̀ㅂ•́)و"


def test_parse_reminder_request_duration_and_clock():
    assert parse_reminder_request("2分钟后提醒我学习") == {
        "content": "学习",
        "at": "2分钟后",
        "kind": "deadline_check",
    }
    assert parse_reminder_request("3点20提醒我学习") == {
        "content": "学习",
        "at": "3点20",
        "kind": "deadline_check",
    }
    assert parse_reminder_request("等会4点钟提醒我喝水") == {
        "content": "喝水",
        "at": "4点",
        "kind": "deadline_check",
    }
    assert parse_reminder_request("麻烦等会过三分钟提醒我吃药") == {
        "content": "吃药",
        "at": "过三分钟",
        "kind": "deadline_check",
    }
    assert parse_reminder_request("能不能等会2点55提醒我一下，我要看a股尾盘") == {
        "content": "看a股尾盘",
        "at": "2点55",
        "kind": "deadline_check",
    }


async def test_add_reminder_without_exact_time_falls_back_to_commitment(store):
    ctx = ToolContext(store=store, memory=None, router=None)

    out = await add_reminder(ctx, content="找机会问问我找工作的进展", when="none")

    assert "记下了" in out
    assert await store.list_reminder_jobs() == []
    commits = await store.list_all_commitments(status="open")
    assert len(commits) == 1
    assert commits[0]["kind"] == "open_loop"


async def test_reminder_runtime_fires_and_marks_sent(store):
    sender = CaptureSender()
    rid = await store.add_reminder_job(
        title="喝水",
        message="到点啦，喝水。",
        trigger_type="date",
        trigger_spec="2020-01-01T00:00:00+00:00",
        due_at_utc="2020-01-01T00:00:00+00:00",
        timezone="Asia/Shanghai",
        original_time_text="过去",
        delivery_channel="telegram",
        delivery_target="12345",
    )

    runtime = ReminderRuntime(store, send_fn=sender)
    await runtime.fire(rid)

    assert sender.sent == [("telegram", "12345", "到点啦，喝水。")]
    job = await store.get_reminder_job(rid)
    assert job is not None and job["status"] == "sent" and job["sent_at"]


async def test_reminder_recover_sends_recent_misfire(store):
    sender = CaptureSender()
    due = (datetime.now(timezone.utc) - timedelta(minutes=44)).isoformat()
    rid = await store.add_reminder_job(
        title="吃饭",
        message="到点啦，吃饭。",
        trigger_type="date",
        trigger_spec=due,
        due_at_utc=due,
        timezone="Asia/Shanghai",
        original_time_text="8点",
        delivery_channel="telegram",
        delivery_target="12345",
    )

    runtime = ReminderRuntime(store, send_fn=sender)
    await runtime.recover()
    await asyncio.sleep(0.05)

    assert sender.sent == [("telegram", "12345", "到点啦，吃饭。")]
    job = await store.get_reminder_job(rid)
    assert job is not None and job["status"] == "sent"


async def test_reminder_reconciler_sends_due_job_without_scheduler(store):
    sender = CaptureSender()
    due = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    rid = await store.add_reminder_job(
        title="拉伸",
        message="到点啦，拉伸。",
        trigger_type="date",
        trigger_spec=due,
        due_at_utc=due,
        timezone="Asia/Shanghai",
        original_time_text="刚才",
        delivery_channel="telegram",
        delivery_target="12345",
    )

    runtime = ReminderRuntime(store, send_fn=sender, scheduler=None)
    await runtime.reconcile_due()
    await asyncio.sleep(0.05)

    assert sender.sent == [("telegram", "12345", "到点啦，拉伸。")]
    job = await store.get_reminder_job(rid)
    assert job is not None and job["status"] == "sent"
