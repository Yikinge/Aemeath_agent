"""时间工具：让模型能确定"现在几点/今天几号/星期几"。"""

from __future__ import annotations

from datetime import datetime

from agent.tools.registry import ToolContext, tool

_WEEK = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]


@tool
async def get_time(ctx: ToolContext) -> str:
    """返回当前本地日期、时间与星期。需要知道现在几点、今天几号、星期几时调用。"""
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S") + " " + _WEEK[now.weekday()]
