"""MemoryGate（方案 §8.2）：检索命中进入上下文**之前**的任务感知门控。

retrieve.py 已按相关度/时效（valid_until、expires_at）做了硬过滤；本模块再按「这一轮在干嘛」
决定每条命中该不该注入——工具/计算/搜索任务别灌情绪叙事、敏感记忆别进工具上下文、问日期时
只留有时间锚点的记忆。纯函数 + 启发式，确定性可单测；每条命中产出 inject/skip + reason（可回溯）。
"""

from __future__ import annotations

from datetime import datetime, timezone

from agent.memory.models import MemoryHit
from agent.memory.types import EVENT_MEMORY, MOOD_OBSERVATION

# 任务类型
TASK_TOOL = "tool"          # 计算/搜索/代码/查询等工具性任务
TASK_COMPANION = "companion"  # 陪伴/情绪/复盘
TASK_NEUTRAL = "neutral"

_TOOL_HINTS = (
    "算", "计算", "等于", "多少", "几点", "搜", "查一下", "查查", "查询", "天气",
    "翻译", "代码", "报错", "bug", "函数", "编译", "运行", "安装", "提醒我", "日程", "+", "-", "*", "/", "=",
)
_COMPANION_HINTS = (
    "累", "难受", "难过", "开心", "高兴", "压力", "焦虑", "心情", "烦", "委屈", "崩溃",
    "怎么办", "陪", "聊聊", "倾诉", "复盘", "最近", "情绪", "孤独", "想你",
)
_DATE_WORDS = (
    "今天", "明天", "昨天", "后天", "前天", "下周", "上周", "这周", "本周", "下个月",
    "几号", "几月", "什么时候", "日期", "周一", "周二", "周三", "周四", "周五", "周六", "周日",
)
# 记忆查询意图：用户在回忆"我之前说过的事"——即使句子里有"几点/多少"等工具词，也该允许记忆注入
_LOOKUP_HINTS = (
    "我说过", "我跟你说过", "我告诉过你", "记不记得", "还记得", "你记得", "记得我",
    "上次", "之前说", "上回", "我提过",
)


def classify_task(query: str) -> dict:
    """启发式判定这一轮的任务类型 + 是否在问日期 + 是否记忆查询（不调 LLM）。"""
    q = query or ""
    is_tool = any(h in q for h in _TOOL_HINTS)
    is_comp = any(h in q for h in _COMPANION_HINTS)
    is_lookup = any(h in q for h in _LOOKUP_HINTS)
    # 情绪/陪伴信号优先（既像工具又像情绪时，按陪伴处理，避免误删情绪记忆）
    if is_comp:
        kind = TASK_COMPANION
    elif is_tool:
        kind = TASK_TOOL
    else:
        kind = TASK_NEUTRAL
    return {"kind": kind, "mentions_date": any(w in q for w in _DATE_WORDS),
            "memory_lookup": is_lookup}


def gate_memories(
    hits: list[MemoryHit], query: str, *, now: str | None = None
) -> tuple[list[MemoryHit], list[dict]]:
    """对召回命中做任务感知门控。返回 (要注入的命中, 决策 trace)。"""
    now = now or datetime.now(timezone.utc).isoformat()
    task = classify_task(query)
    kind, mentions_date, lookup = task["kind"], task["mentions_date"], task["memory_lookup"]
    # 记忆查询（"我上次说过…"）：即使像工具任务，也允许记忆注入，不走工具剔除
    tool_skip = kind == TASK_TOOL and not lookup

    kept: list[MemoryHit] = []
    decisions: list[dict] = []
    for h in hits:
        it = h.item
        mtype = it.memory_type
        sens = (it.sensitivity or "low")
        reason = None

        if it.expires_at and it.expires_at <= now:
            reason = "expired"                       # 防御：过期不注入
        elif tool_skip and sens in ("personal", "care"):
            reason = "sensitive_not_for_tool"        # 敏感记忆不进工具上下文
        elif tool_skip and mtype in (EVENT_MEMORY, MOOD_OBSERVATION):
            reason = "tool_task_skip_emotional"      # 工具任务少灌情绪叙事
        elif mentions_date and mtype == EVENT_MEMORY and not it.event_at:
            reason = "date_query_needs_anchor"       # 问日期时，无时间锚点的事件类不注入

        decision = "skip" if reason else "inject"
        decisions.append({
            "memory_id": it.id, "decision": decision,
            "reason": reason or f"{kind}_relevant", "score": round(h.score, 3),
        })
        if decision == "inject":
            kept.append(h)
    return kept, decisions
