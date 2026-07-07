"""时间归一化（方案 P0 §6.1）：把中文相对时间锚到 source_at + 时区，转绝对 ISO。

长期记忆必须存绝对时间——"下周三""今天"在写入当天对，作为长期记忆会漂移。
本模块**纯规则、零三方依赖**（只用 stdlib zoneinfo），可被确定性单测覆盖；
复杂/口语化表达留给上层 LLM 写进 content，但结构化的 event_at 一律以本模块为权威。

P0 覆盖：今天/明天/后天/大后天/昨天/前天、本周X/这周X/下周X/下下周X/上周X/周末、
X月X号/日（含跨年）、下个月X号、以及"上午/下午/晚上 + N点(半)"时间点。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - py<3.9 不会发生（本项目 3.12+）
    ZoneInfo = None  # type: ignore

_WEEKDAY_CN = ["一", "二", "三", "四", "五", "六", "日"]
_WEEKDAY = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6, "末": 5}

_CN_DIGIT = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
             "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}

# 命名相对日 → 相对今天的天数偏移（按长度降序匹配，避免"后天"吃掉"大后天"）
_NAMED_DAYS = {
    "大后天": 3, "后天": 2, "明天": 1, "明日": 1, "明儿": 1,
    "今天": 0, "今日": 0, "今儿": 0, "本日": 0,
    "昨天": -1, "昨日": -1, "昨儿": -1, "前天": -2,
}

# 绝对 ISO / 中文绝对日期（最高置信，优先于任何相对表达）：2026-06-24 / 2026/6/24 / 2026年6月24日
_ISO_RE = re.compile(r"(\d{4})\s*[-/年]\s*(\d{1,2})\s*[-/月]\s*(\d{1,2})\s*日?")
_WEEK_RE = re.compile(r"(下下|下|这|本|上)?\s*(?:周|星期|礼拜)\s*([一二三四五六日天末])")
_MD_RE = re.compile(r"(\d{1,2}|[一二三四五六七八九十]+)\s*月\s*(\d{1,2}|[一二三四五六七八九十]+)\s*[号日]")
_MONTH_REL_RE = re.compile(r"(下下个?月|下个?月|这个?月|本月)\s*(\d{1,2}|[一二三四五六七八九十]+)\s*[号日]")
_TIME_RE = re.compile(
    r"(凌晨|早上|早晨|上午|中午|下午|傍晚|晚上|晚|夜里)?\s*"
    r"(\d{1,2}|[一二三四五六七八九十两]+)\s*点\s*(半|[0-5]?\d|[一二三四五六七八九十]+)?\s*分?"
)


def _cn_to_int(s: str | None) -> int | None:
    """中文/阿拉伯数字 → int（覆盖 0~59，够日/号/点/分用）。"""
    if not s:
        return None
    s = s.strip()
    if not s:
        return None
    if s.isdigit():
        return int(s)
    if s == "十":
        return 10
    if "十" in s:
        tens, _, ones = s.partition("十")
        t = _CN_DIGIT.get(tens, 1) if tens else 1
        o = _CN_DIGIT.get(ones, 0) if ones else 0
        return t * 10 + o
    return _CN_DIGIT.get(s)


@dataclass
class TimeMention:
    original: str
    normalized: str       # "YYYY-MM-DD" 或带时区的 "YYYY-MM-DDTHH:MM:SS+08:00"
    kind: str             # "date" | "datetime"
    confidence: float


@dataclass
class NormalizedTime:
    today: date
    time_mentions: list[TimeMention] = field(default_factory=list)
    primary_event_at: str | None = None     # 下游 event_at 的权威值
    primary_kind: str | None = None         # "date" | "datetime" | None

    def hint(self) -> str:
        """给 route/commitment prompt 的「时间锚点」提示，要求 LLM 用绝对日期写 content。"""
        wd = _WEEKDAY_CN[self.today.weekday()]
        parts = [f"今天={self.today.isoformat()}(周{wd})"]
        for m in self.time_mentions:
            parts.append(f"「{m.original}」={m.normalized}")
        return "【时间锚点】" + "；".join(parts)


def _safe_zone(name: str | None):
    if ZoneInfo is not None:
        try:
            return ZoneInfo(name or "Asia/Shanghai")
        except Exception:
            try:
                return ZoneInfo("UTC")
            except Exception:
                pass
    return timezone.utc


def current_time_hint(tz_name: str = "Asia/Shanghai") -> str:
    """注入对话 system prompt 的"现在"：让模型知道真实当前时间，不靠权重瞎猜。
    形如 `2026-06-25（星期三）14:30，Asia/Shanghai`。"""
    now = datetime.now(_safe_zone(tz_name))
    return f"{now.strftime('%Y-%m-%d')}（星期{_WEEKDAY_CN[now.weekday()]}）{now.strftime('%H:%M')}，{tz_name}"


def _to_local(iso: str | None, tz) -> datetime:
    try:
        t = datetime.fromisoformat(iso) if iso else datetime.now(timezone.utc)
    except (ValueError, TypeError):
        t = datetime.now(timezone.utc)
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t.astimezone(tz)


def _resolve_weekday(prefix: str | None, char: str, today: date) -> tuple[date, float]:
    target = _WEEKDAY[char]
    base = today.weekday()
    delta = target - base
    conf = 0.9
    if prefix == "上":
        delta -= 7
    elif prefix == "下":
        delta += 7
    elif prefix == "下下":
        delta += 14
    elif prefix in ("本", "这"):
        pass                       # 本周该天（可能已过）
    else:
        conf = 0.6 if char == "末" else 0.75
        if delta < 0:              # 裸「周X」且已过 → 默认指将到来的那个
            delta += 7
    return today + timedelta(days=delta), conf


def _resolve_month_day(m: int, d: int, today: date) -> date | None:
    if not (1 <= m <= 12 and 1 <= d <= 31):
        return None
    year = today.year
    try:
        cand = date(year, m, d)
    except ValueError:
        return None
    if cand < today:               # 已过 → 取明年同月日（跨年）
        try:
            cand = date(year + 1, m, d)
        except ValueError:
            return None
    return cand


def _resolve_month_rel(prefix: str, d: int, today: date) -> date | None:
    if not 1 <= d <= 31:
        return None
    off = {"本月": 0}.get(prefix)
    if off is None:
        if prefix.startswith("下下"):
            off = 2
        elif prefix.startswith("下"):
            off = 1
        else:                      # 这个月/这月
            off = 0
    month0 = today.month - 1 + off
    year = today.year + month0 // 12
    month = month0 % 12 + 1
    try:
        return date(year, month, d)
    except ValueError:
        return None


def normalize_time(text: str, source_at_iso: str | None, tz_name: str = "Asia/Shanghai") -> NormalizedTime:
    """从一句话里抽出时间表达并锚成绝对时间。读不出时间则 mentions 空、primary 为 None。"""
    tz = _safe_zone(tz_name)
    base = _to_local(source_at_iso, tz)
    today = base.date()
    res = NormalizedTime(today=today)

    dates: list[tuple[str, date, float]] = []
    seen: set[str] = set()

    def _add_date(original: str, d: date, conf: float) -> None:
        if original in seen:
            return
        seen.add(original)
        dates.append((original, d, conf))
        res.time_mentions.append(TimeMention(original, d.isoformat(), "date", conf))

    # 0) 绝对 ISO / 中文绝对日期（最高置信，优先级最高）
    for mt in _ISO_RE.finditer(text):
        y, m, d = int(mt.group(1)), _cn_to_int(mt.group(2)), _cn_to_int(mt.group(3))
        if m and d and 1 <= m <= 12 and 1 <= d <= 31:
            try:
                _add_date(mt.group(0).replace(" ", ""), date(y, m, d), 1.0)
            except ValueError:
                pass

    # 1) 命名相对日（长度降序，先匹配"大后天"）
    for name in sorted(_NAMED_DAYS, key=len, reverse=True):
        if name in text:
            _add_date(name, today + timedelta(days=_NAMED_DAYS[name]), 0.95)

    # 2) 周X
    for mt in _WEEK_RE.finditer(text):
        d, conf = _resolve_weekday(mt.group(1), mt.group(2), today)
        _add_date(mt.group(0).replace(" ", ""), d, conf)

    # 3) X月X号
    for mt in _MD_RE.finditer(text):
        m, d = _cn_to_int(mt.group(1)), _cn_to_int(mt.group(2))
        if m and d:
            resolved = _resolve_month_day(m, d, today)
            if resolved:
                _add_date(mt.group(0).replace(" ", ""), resolved, 0.9)

    # 4) 下个月X号 / 这个月X号
    for mt in _MONTH_REL_RE.finditer(text):
        d = _cn_to_int(mt.group(2))
        if d:
            resolved = _resolve_month_rel(mt.group(1), d, today)
            if resolved:
                _add_date(mt.group(0).replace(" ", ""), resolved, 0.8)

    # 5) 时间点（取第一个）
    time_part: tuple[str, int, int, float] | None = None
    tm = _TIME_RE.search(text)
    if tm:
        hour = _cn_to_int(tm.group(2))
        if hour is not None and 0 <= hour <= 24:
            period = tm.group(1)
            raw_min = tm.group(3)
            minute = 30 if raw_min == "半" else (_cn_to_int(raw_min) or 0)
            minute = minute if 0 <= minute < 60 else 0
            conf = 0.6
            ambiguous = False
            if period in ("下午", "傍晚", "晚上", "晚", "夜里"):
                if hour < 12:
                    hour += 12
                conf = 0.85
            elif period in ("上午", "早上", "早晨", "凌晨", "中午"):
                conf = 0.85
            else:
                ambiguous = 1 <= hour <= 11   # 没带"上午/下午"的 12 小时制模糊小时，留待下面按"已过则下午"消歧
            if hour == 24:
                hour = 0
            if 0 <= hour <= 23:
                time_part = (tm.group(0).replace(" ", ""), hour, minute, conf, ambiguous)

    # 选 primary：最高置信的日期；时间点附到它（无日期则附到今天）
    primary_date: date | None = None
    if dates:
        primary_date = max(dates, key=lambda x: x[2])[1]
    if time_part:
        d0 = primary_date or today
        h, mi, ambiguous = time_part[1], time_part[2], time_part[4]
        dt = datetime.combine(d0, time(hour=h, minute=mi), tzinfo=tz)
        # 12 小时制消歧：没带"上午/下午"的模糊小时（1~11），若已是过去时刻 → 当下午（+12），
        # 取"接下来那次"，符合提醒/计划的前瞻语义（修"今天5点下班"被存成 05:00 的 bug）。
        if ambiguous and dt <= base:
            dt = dt + timedelta(hours=12)
        res.primary_event_at = dt.isoformat()
        res.primary_kind = "datetime"
        res.time_mentions.append(TimeMention(time_part[0], dt.isoformat(), "datetime", time_part[3]))
    elif primary_date is not None:
        res.primary_event_at = primary_date.isoformat()
        res.primary_kind = "date"

    return res


def time_anchor_hint(text: str, source_at_iso: str | None, tz_name: str = "Asia/Shanghai") -> str:
    """便捷封装：直接拿到注入 prompt 的时间锚点提示（没有时间表达时返回 ""）。"""
    nt = normalize_time(text, source_at_iso, tz_name)
    return nt.hint() if nt.time_mentions else ""


def absolutize(text: str, source_at_iso: str | None, tz_name: str = "Asia/Shanghai") -> str:
    """把文本里的相对日期词（今天/明天/下周三…）就地替换成绝对日期，确定性兜底防漂移。

    逐 mention 处理（不再整段跳过）：只替换 **date 类**相对词，不动"下午3点"这类纯时刻；
    若文本里已有一个**不同值**的绝对 ISO 日期，则信任它、跳过该相对词，避免
    "2026-06-14 用户2026-06-13去爬山"这种双日期矛盾。相对词与已有绝对日期同值时正常替换。
    """
    if not text:
        return text
    nt = normalize_time(text, source_at_iso, tz_name)
    # 文本里原文本身就是绝对 ISO 的日期值（如 "2026-06-24"）
    abs_vals = {m.normalized for m in nt.time_mentions if _ISO_RE.search(m.original)}
    out = text
    for m in nt.time_mentions:
        if m.kind != "date" or _ISO_RE.search(m.original):
            continue                                  # 时刻类 / 本身已是绝对 → 不动
        if abs_vals and m.normalized not in abs_vals:
            continue                                  # 与已有绝对日期矛盾 → 信任绝对的
        if m.original in out:
            out = out.replace(m.original, m.normalized)
    return out
