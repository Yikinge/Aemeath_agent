"""提醒/承诺工具：让模型把"以后要主动跟进的事"写进 commitment 表，喂给主动引擎。

直接强化护城河（主动）：模型显式记下的开放回路，到点由 ProactiveEngine 主动找用户。
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from agent.memory.models import now_iso
from agent.memory.normalize import absolutize, normalize_time
from agent.gateway.router import LLMRouter
from agent.tools.registry import ToolContext, tool

_KINDS = {"care_check_in", "open_loop", "event_check_in", "deadline_check"}
_TZ = "Asia/Shanghai"
_NUM_RE = r"\d+|[一二两三四五六七八九十]+"
_AFTER_RE = re.compile(rf"(?:过|再过)?\s*({_NUM_RE})\s*(秒钟?|分钟|分|小时|个小时|天)\s*(?:后|以后|之后)?")
_COLON_TIME_RE = re.compile(r"\b(\d{1,2}):([0-5]\d)\b")
_CLOCK_RE = re.compile(
    rf"((?:今天|明天|后天|大后天|下周[一二三四五六日天]|周[一二三四五六日天])?\s*"
    rf"(?:凌晨|早上|早晨|上午|中午|下午|傍晚|晚上|晚|夜里)?\s*"
    rf"(?:{_NUM_RE})\s*点\s*(?:半|[0-5]?\d|[一二三四五六七八九十]+)?\s*分?)"
    r"\s*钟?"
)
_REMIND_WORD_RE = re.compile(r"(提醒我|叫我|喊我|通知我|提醒一下我|帮我提醒|记得提醒我|提醒)")
_FILLER_RE = re.compile(
    r"(麻烦|拜托|劳驾|请|能不能|可不可以|可以|帮我|帮忙|等会儿|等会|待会儿|待会|一会儿|一会|"
    r"到时候|的时候|一下|再)"
)


def _due(when: str) -> str | None:
    """把粗粒度时间提示映射成到期时间；'none' 表示不急、找机会跟。"""
    now = datetime.now(timezone.utc)
    return {
        "soon": now.isoformat(),
        "tomorrow": (now + timedelta(days=1)).isoformat(),
        "next_week": (now + timedelta(days=7)).isoformat(),
    }.get(when)


def _cn_to_int(s: str) -> int | None:
    if s.isdigit():
        return int(s)
    digits = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
              "六": 6, "七": 7, "八": 8, "九": 9}
    if s == "十":
        return 10
    if "十" in s:
        left, _, right = s.partition("十")
        tens = digits.get(left, 1) if left else 1
        ones = digits.get(right, 0) if right else 0
        return tens * 10 + ones
    return digits.get(s)


def _source_dt(source_at_iso: str) -> datetime:
    try:
        dt = datetime.fromisoformat(source_at_iso)
    except ValueError:
        dt = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(ZoneInfo(_TZ))


def _relative_due(text: str, source_at_iso: str) -> str | None:
    m = _AFTER_RE.search(text or "")
    if not m:
        return None
    n = _cn_to_int(m.group(1))
    if n is None:
        return None
    unit = m.group(2)
    if unit.startswith("秒"):
        delta = timedelta(seconds=n)
    elif unit in ("分钟", "分"):
        delta = timedelta(minutes=n)
    elif "小时" in unit:
        delta = timedelta(hours=n)
    else:
        delta = timedelta(days=n)
    return (_source_dt(source_at_iso) + delta).isoformat()


def _normalize_colon_time(text: str) -> str:
    return _COLON_TIME_RE.sub(lambda m: f"{m.group(1)}点{m.group(2)}分", text or "")


def _to_utc_iso(value: str) -> str:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _fmt_local(value: str) -> str:
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return value
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(ZoneInfo(_TZ)).strftime("%Y-%m-%d %H:%M")


def _fmt_when(value: str) -> str:
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return value
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    local = dt.astimezone(ZoneInfo(_TZ))
    now = datetime.now(ZoneInfo(_TZ)).date()
    if local.date() == now:
        return local.strftime("今天 %H:%M")
    if local.date() == now + timedelta(days=1):
        return local.strftime("明天 %H:%M")
    return local.strftime("%Y-%m-%d %H:%M")


def _clean_content(text: str) -> str:
    content = _FILLER_RE.sub(" ", text or "")
    content = re.sub(r"^[\s，。,.、：:]+", " ", content)
    content = re.sub(r"^(?:我)?(?:要|需要|得|该|想要)\s*", "", content.strip())
    content = re.sub(r"\s+", " ", content).strip(" ，。,.、：:")
    return content


def _confirmation(content: str, due_at: str, *, has_target: bool) -> str:
    note = "" if has_target else "（我先存下了；等 Telegram 对话目标确认后才能主动发出去。）"
    return f"设好了，{_fmt_when(due_at)} 我提醒你：{content}。{note}".strip()


async def render_reminder_reply(
    router: LLMRouter | None,
    persona: str | None,
    base_reply: str,
    *,
    user_text: str = "",
) -> str:
    """把确定性提醒结果用人格润色；只改语气，不改事实。"""
    base_reply = (base_reply or "").strip()
    if not base_reply or router is None or not persona or not router.live():
        return base_reply
    prompt = (
        "你只负责把一条【提醒设置成功】的系统回执改写成符合人格的自然回复。\n"
        "硬性规则：\n"
        "- 必须保留回执里的时间和提醒事项，不得改写事实。\n"
        "- 只能说已经设好，不要说自己不能主动提醒、不要建议手机闹钟、不要解释系统限制。\n"
        "- 1 到 2 句，短，口语；可以有一个符合人格的颜文字。\n"
        "- 不要列表，不要 Markdown，不要引号。\n\n"
        f"用户原话：{user_text or '（无）'}\n"
        f"系统回执：{base_reply}\n"
        "请直接输出给用户看的最终回复。"
    )
    try:
        out = (await router.complete(
            [{"role": "system", "content": persona}, {"role": "user", "content": prompt}],
            task="fast",
        )).strip()
    except Exception:
        return base_reply
    if not out or len(out) > 160:
        return base_reply
    return out


async def render_reminder_delivery(
    router: LLMRouter | None,
    persona: str | None,
    base_message: str,
) -> str:
    """把到点外发提醒用人格润色；提醒事实来自 base_message。"""
    base_message = (base_message or "").strip()
    if not base_message or router is None or not persona or not router.live():
        return base_message
    prompt = (
        "你只负责把一条【到点提醒】改写成符合人格的短消息。\n"
        "硬性规则：\n"
        "- 必须保留提醒事项，不得添加新的任务或时间。\n"
        "- 明确表达到点了/该做这件事了。\n"
        "- 1 句，短，口语；可以有一个符合人格的颜文字。\n"
        "- 不要列表，不要 Markdown，不要引号。\n\n"
        f"基准提醒：{base_message}\n"
        "请直接输出要发给用户的提醒。"
    )
    try:
        out = (await router.complete(
            [{"role": "system", "content": persona}, {"role": "user", "content": prompt}],
            task="fast",
        )).strip()
    except Exception:
        return base_message
    if not out or len(out) > 120:
        return base_message
    return out


_REMINDER_PARAMS = {
    "type": "object",
    "properties": {
        "content": {"type": "string",
                    "description": "要提醒/跟进的事，一句话。如「提醒下班」「问问面试结果」。不要把时间塞进这里。"},
        "at": {"type": "string",
               "description": "具体时间（自然语言）。用户说了钟点或多久以后就**务必**填，"
                              "如「下午5点10分」「15:20」「明天上午9点」「2分钟后」「下周三下午3点」。没有具体时间就留空。"},
        "when": {"type": "string", "enum": ["soon", "tomorrow", "next_week", "none"],
                 "description": "粗粒度时间，仅当没有具体钟点(at 留空)时用：马上/明天/下周/不急找机会"},
        "kind": {"type": "string",
                 "enum": ["deadline_check", "event_check_in", "open_loop", "care_check_in"],
                 "description": "deadline_check=到点提醒(有明确时间)；event_check_in=事后问问怎么样；"
                                "open_loop=没时间、找机会问；care_check_in=情绪关心"},
    },
    "required": ["content"],
}


@tool(
    description="登记一件要提醒/跟进的事。用户说「X点提醒我做Y」时，务必把时间填进 at 参数；有明确时间会走精确定时器，不走主动心跳。",
    parameters=_REMINDER_PARAMS,
)
async def add_reminder(
    ctx: ToolContext, content: str, at: str = "", when: str = "none", kind: str = "open_loop"
) -> str:
    """登记一件要主动提醒/跟进的事。

    有明确时间时写入 reminder_job，走确定性 scheduler；没有明确时间时才回落为
    commitment，让软主动以后找机会跟进。
    """
    content = (content or "").strip()
    if not content:
        return "（没有可登记的事）"
    if kind not in _KINDS:
        kind = "open_loop"
    # 防漂移：相对日期就地转绝对（与 consolidator 同口径）
    src = now_iso()
    content = absolutize(content, src, _TZ)
    # 解析具体时间：优先用 at（用户说的钟点），其次从 content 里找
    time_text = _normalize_colon_time(at.strip() or content)
    relative_at = _relative_due(time_text, src)
    nt = normalize_time(time_text, src, _TZ)
    event_at = nt.primary_event_at
    due_at = _due(when) or relative_at or event_at

    # 有具体触发时间：这是用户授权的硬提醒，必须进入 reminder_job，不再混入 commitment。
    if due_at and "T" in due_at:
        due_utc = _to_utc_iso(due_at)
        chat_id = await ctx.store.kv_get("telegram_chat_id")
        target = chat_id or "default"
        msg = await render_reminder_delivery(
            ctx.router, ctx.persona, f"到点啦，{content}。"
        )
        idem_src = f"{ctx.namespace}|{target}|{content}|{due_utc}"
        rid = await ctx.store.add_reminder_job(
            title=content,
            message=msg,
            trigger_type="date",
            trigger_spec=due_utc,
            due_at_utc=due_utc,
            timezone=_TZ,
            original_time_text=at or when,
            delivery_channel="telegram",
            delivery_target=target,
            namespace=ctx.namespace,
            idempotency_key=hashlib.sha256(idem_src.encode("utf-8")).hexdigest(),
        )
        job = await ctx.store.get_reminder_job(rid)
        if ctx.reminders is not None and job is not None:
            ctx.reminders.schedule(job)
        return _confirmation(content, due_at, has_target=bool(chat_id))

    # 没有触发时间：这不是闹钟，只是以后找机会跟进的开放回路。
    await ctx.store.add_commitment(
        kind=kind, content=content, due_at=due_at,
        sensitivity="care" if kind == "care_check_in" else "routine",
        source="tool", namespace=ctx.namespace, event_at=event_at,
    )
    return f"记下了，之后我会找个合适的时机跟你提：{content}。"


@tool
async def list_reminders(ctx: ToolContext) -> str:
    """列出当前精确提醒；用户问待跟进事项时仍可由记忆/主动系统回答。"""
    items = await ctx.store.list_reminder_jobs(ctx.namespace, status="scheduled")
    if not items:
        return "现在没有已设定的精确提醒。"
    lines = []
    for item in items:
        due = item.get("due_at_utc") or item.get("trigger_spec") or ""
        lines.append(f"- {item['title']}（{due}）")
    return "已设定的精确提醒：\n" + "\n".join(lines)


def parse_reminder_request(text: str) -> dict | None:
    """确定性兜底：识别常见中文提醒请求，避免模型漏调 add_reminder。

    覆盖：
    - 2分钟后提醒我学习 / 过两分钟再提醒我学习
    - 3点20提醒我学习 / 下午3点20提醒我学习 / 15:20提醒我学习
    """
    raw = (text or "").strip()
    if not raw or not _REMIND_WORD_RE.search(raw):
        return None
    after = _AFTER_RE.search(raw)
    clock = _CLOCK_RE.search(_normalize_colon_time(raw))
    colon = _COLON_TIME_RE.search(raw)
    if not (after or clock or colon):
        return None
    at = after.group(0) if after else (clock.group(1) if clock else f"{colon.group(1)}点{colon.group(2)}分")
    content = raw
    for m in [after, clock, colon]:
        if m:
            content = content.replace(m.group(0), " ")
    content = _REMIND_WORD_RE.sub(" ", content)
    content = _clean_content(content)
    return {
        "content": content or "这件事",
        "at": at,
        "kind": "deadline_check",
    }
