"""确定性单测：中文相对时间归一化（方案 P0 §6.1）。纯函数，不调 LLM。

锚点统一用 2026-06-18T10:00:00+08:00（周四），对照方案验收条款。
"""

from __future__ import annotations

from agent.memory.normalize import absolutize, normalize_time, time_anchor_hint

_THU = "2026-06-18T10:00:00+08:00"   # 周四
_MON = "2026-06-15T16:49:41+08:00"   # 周一（方案示例 source_at）


def _primary(text: str, src: str = _THU) -> str | None:
    return normalize_time(text, src, "Asia/Shanghai").primary_event_at


# ---------- 方案核心验收：未来计划用绝对日期 ----------

def test_next_wednesday_from_thursday():
    # 周四说"下周三" → 下一周的周三 = 2026-06-24
    assert _primary("下周三我有个重要面试") == "2026-06-24"


def test_tomorrow():
    assert _primary("明天再说") == "2026-06-19"


def test_today_and_yesterday():
    assert _primary("今天被领导骂了") == "2026-06-18"
    assert _primary("昨天加班到很晚") == "2026-06-17"


def test_day_after_tomorrow_priority_over_after():
    # "大后天"不能被"后天"吃掉
    assert _primary("大后天出发") == "2026-06-21"


# ---------- 时间点：绑定日期 + 时区偏移 ----------

def test_afternoon_time_binds_to_date_and_tz():
    # 周一说"明天下午3点" → 2026-06-16T15:00:00+08:00
    assert normalize_time("明天下午3点有面试", _MON).primary_event_at == "2026-06-16T15:00:00+08:00"


def test_evening_half_past():
    assert _primary("晚上8点半吃饭", _MON) == "2026-06-15T20:30:00+08:00"


def test_bare_time_uses_today():
    nt = normalize_time("下午两点开会", _THU)
    assert nt.primary_event_at == "2026-06-18T14:00:00+08:00"
    assert nt.primary_kind == "datetime"


# ---------- 裸小时（没带上午/下午）的 12 小时制消歧 ----------

def test_bare_hour_past_assumes_pm():
    # 周一 16:49 说"今天5点下班" → 5点已过 → 当下午 17:00（修 AM/PM 漏洞，原会存成 05:00）
    assert _primary("今天5点下班提醒我", _MON) == "2026-06-15T17:00:00+08:00"


def test_bare_hour_future_same_day_stays_am():
    # 周四 10:00 说"11点开会" → 11点还没到 → 保留上午 11:00，不臆测
    assert _primary("11点开会", _THU) == "2026-06-18T11:00:00+08:00"


def test_bare_hour_future_date_stays_am():
    # "明天5点"是未来日 → 不臆测下午，保留 05:00（消歧只对"同日已过"的小时生效）
    assert _primary("明天5点的火车", _MON) == "2026-06-16T05:00:00+08:00"


# ---------- 月日 / 跨年 / 下个月 ----------

def test_month_day_this_year():
    assert _primary("6月24号面试") == "2026-06-24"


def test_month_day_rolls_to_next_year_when_passed():
    # 周四是 6-18，说"1月5号"已过 → 取明年
    assert _primary("1月5号有安排") == "2027-01-05"


def test_next_month():
    assert _primary("下个月5号") == "2026-07-05"


# ---------- 周X 边界 ----------

def test_weekend_is_saturday():
    assert _primary("周末去爬山") == "2026-06-20"


def test_last_week_is_negative():
    assert _primary("上周五交了报告") == "2026-06-12"


def test_bare_weekday_rolls_forward_when_passed():
    # 周四说"周三"（本周三已过）→ 默认指将到来的下周三
    assert _primary("周三见") == "2026-06-24"


# ---------- 无时间表达 ----------

def test_no_time_returns_none():
    nt = normalize_time("我喜欢喝美式咖啡", _THU)
    assert nt.primary_event_at is None and nt.time_mentions == []


def test_hint_empty_when_no_time():
    assert time_anchor_hint("我喜欢喝美式咖啡", _THU) == ""


def test_hint_contains_anchor_and_absolute():
    hint = time_anchor_hint("下周三我有面试", _THU)
    assert "2026-06-18" in hint and "2026-06-24" in hint


# ---------- absolutize：相对日期词就地改写（兜底防漂移） ----------

def test_absolutize_replaces_relative_day_word():
    # care 类承诺内容里的"今天"会漏进 MEMORY.md → 兜底替换成绝对日期
    out = absolutize("问候用户今天被领导骂后的情绪", _THU)
    assert "今天" not in out and "2026-06-18" in out


def test_absolutize_skips_when_iso_present():
    # 已含绝对日期 → 信任上游，不改写（避免双日期噪声）
    text = "2026-06-14 用户上周末去爬了香山"
    assert absolutize(text, _THU) == text


def test_absolutize_keeps_clock_time():
    # 只动 date 类，不动"下午3点"这种纯时刻
    out = absolutize("明天下午3点面试", _MON)
    assert "明天" not in out and "下午3点" in out and "2026-06-16" in out


def test_absolutize_noop_without_time():
    assert absolutize("我喜欢喝咖啡", _THU) == "我喜欢喝咖啡"
