"""抽取器（Evaluators）：从新对话里抽出候选记忆。

- 画像抽取（S2）：稳定结构化属性 → FactCandidate
- 语义记忆抽取（S3）：不好结构化的小事/事件 → MemoryCandidate

优先用 LLM；离线 / 无 Key / 解析失败时退回启发式正则，保证没配模型也能跑。
"""

from __future__ import annotations

import json
import re

from agent.gateway.router import LLMRouter
from agent.memory.models import FactCandidate, MemoryCandidate
from agent.memory.types import DISCARD, RoutedMemory

# ---------------- 画像抽取（S2） ----------------

_FACT_SYSTEM = """你是个人智能体的「画像抽取器」。从用户的话里抽取**长期稳定**的画像事实与实体。
只输出 JSON 数组，每个元素 {"category","key","value","confidence","canonical_key"}：
- category: bio(身份基本信息) / preference(喜欢) / taste(口味) / value(价值观) / taboo(雷区不喜欢) / routine(作息规律) / social(人际) / entity(长期实体/关系/物件，如宠物、家人) / other
- key: 稳定的英文 snake_case 属性名（如 name / favorite_coffee / sleep_time / job / pet_cat），同一属性/同一实体**务必只用一个 key**
- value: 简洁的中文值；**同一实体的所有属性合写进一个 value**（如养了叫煤球的橘猫 → key=pet_cat, value="煤球（橘猫）"，不要拆成 pet_name/pet_species 多条）
- confidence: 0~1，明确陈述给 0.9，推断给 0.5
- canonical_key: 实体类给稳定规范键（如 entity:pet:cat、entity:family:mother）；非实体给 null
只抽**确定且持久**的事实/实体；闲聊、一次性的事、已发生的具体事件、未来计划、助手的话都不要抽。没有就返回 []。只输出 JSON。"""


async def extract_profile_facts(router: LLMRouter, new_messages: list[dict]) -> list[FactCandidate]:
    user_text = _user_text(new_messages)
    if not user_text:
        return []
    raw = await router.complete(
        [{"role": "system", "content": _FACT_SYSTEM}, {"role": "user", "content": user_text}],
        task="fast",
    )
    parsed = _parse(raw, _to_fact)
    return parsed if parsed is not None else _heuristic_facts(user_text)


# ---------------- 记忆路由（route，方案 §6.2，取代旧的 dumb 去重抽取） ----------------

_ROUTE_SYSTEM = """你是个人智能体的「记忆路由器」。从用户最近的话里挑出值得长期处理的条目，并给每条**分类**。
只输出 JSON 数组，每个元素：
{"memory_type","content","canonical_key","event_at","expires_at","keywords","importance","confidence","sensitivity","reason"}

memory_type（决定这条怎么存）：
- entity_fact: 长期实体/关系/物件（养的宠物、家人、重要物件）——会并进画像，不当叙事。
- event_memory: **已经发生**、带情绪或意义的具体事件（上周爬了香山、昨天搬家）。
- active_commitment: **未来**计划/约定/有截止、事后值得跟进的事（下周三面试、月底交报告）。
- mood_observation: 单轮情绪状态（今天很累、有点焦虑）。
- discard: 寒暄、客套、一次性请求（算个数、搜一下）、助手的话、信息不全——不值得记。

规则：
- content 一律用**绝对日期**改写（见下方【时间锚点】），不要出现"今天/明天/下周三"这类相对词。
- canonical_key：同一事件/实体给稳定键，便于日后合并。例：event:interview:2026-06-24、entity:pet:cat。没有给 null。
- event_at：事件/计划的绝对时间（YYYY-MM-DD 或带时区 datetime）；没有给 null。
- expires_at：短期情绪/短效计划的过期时间；长期记忆给 null。
- importance：1~10 整数（人生大事/强烈情绪/重要关系 8~10；日常计划 4~6；琐碎一过性 1~3）。
- sensitivity：low（普通可示人）/ personal（私密，如健康/财务/家庭隐私）/ care（情绪脆弱时刻）——默认 low。
- confidence：0~1；reason：一句话「为什么值得记 / 为什么这么分类」。
没有值得记的就返回 []。只输出 JSON。"""

_MEM_MIN_IMPORTANCE = 0.3   # 价值闸门：event_memory importance（0..1）低于此值不入叙事层


async def route_memories(
    router: LLMRouter, new_messages: list[dict],
    time_hint: str = "", existing: set[str] | None = None,
) -> list[RoutedMemory]:
    """对新消息里的可记条目分类（方案 §6.2）。返回去掉 discard 的候选；真正的去重交给叙事 resolve。"""
    user_text = _user_text(new_messages)
    if not user_text:
        return []
    user_content = f"{time_hint}\n\n{user_text}" if time_hint else user_text
    raw = await router.complete(
        [{"role": "system", "content": _ROUTE_SYSTEM}, {"role": "user", "content": user_content}],
        task="fast",
    )
    cands = _parse(raw, _to_routed) or []
    out = [c for c in cands if c.memory_type != DISCARD]
    if existing:  # 廉价预去重；语义合并由 resolve 负责
        out = [c for c in out if c.content not in existing]
    return out


# ---------------- 情绪抽取（MEM-4，喂主动 check-in 的核心信号） ----------------

_MOOD_SYSTEM = """从用户这段话里读出他/她当前的情绪状态。只输出一个 JSON 对象：
{"valence": -1..1, "arousal": 0..1, "signals": ["..."], "note": "一句话总结"}
- valence: -1 很差 / 0 中性 / +1 很好
- arousal: 0 平静 / 1 激动（情绪强度，不分好坏）
- signals: 0~4 个关键词（中文），如 "加班"/"累"/"开心"/"焦虑"，没有就给 []
- note: 一句不超过 20 字的中文概括；中性闲聊给 ""
若完全读不出情绪信号（纯事实/纯客套），返回 {"valence":null,"arousal":null,"signals":[],"note":""}。
只输出 JSON。"""


async def extract_mood(router: LLMRouter, new_messages: list[dict]) -> dict | None:
    """读出 (valence, arousal, signals, note)；读不出返 None。"""
    user_text = _user_text(new_messages)
    if not user_text:
        return None
    raw = await router.complete(
        [{"role": "system", "content": _MOOD_SYSTEM}, {"role": "user", "content": user_text}],
        task="fast",
    )
    match = re.search(r"\{.*\}", raw.strip(), re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    val = data.get("valence")
    aro = data.get("arousal")
    signals = data.get("signals", [])
    note = data.get("note", "")
    # 读不出任何信号则不要写库，避免噪声把曲线压平
    if val is None and aro is None and not signals and not note:
        return None
    return {
        "valence": float(val) if val is not None else None,
        "arousal": float(aro) if aro is not None else None,
        "signals": [str(s) for s in signals if str(s).strip()] if isinstance(signals, list) else [],
        "note": str(note).strip() or None,
    }


# ---------------- 反思（A2，Generative Agents reflection） ----------------

_REFLECT_SYSTEM = """你在回顾关于某个人的一批记忆。请归纳 1~3 条更高层的「洞察」——
不是复述具体事实，而是提炼规律 / 性格 / ta 在意的东西 / 近期状态趋势
（例如"很重视家人"、"最近工作压力偏大"、"对宠物投入很深"）。
每条要有概括性、对长期理解这个人有价值，且能从给到的材料里站得住。
只输出 JSON 数组，每个元素 {"insight"}（中文一句话）。归纳不出有把握的就返回 []。只输出 JSON。"""


def _to_insight(it):
    if not isinstance(it, dict):
        return None
    s = str(it.get("insight", "")).strip()
    return s or None


async def reflect_insights(router: LLMRouter, material: str) -> list[str]:
    """从一批画像+叙事材料里归纳高层洞察。用强模型（不走 fast）。"""
    if not material.strip():
        return []
    raw = await router.complete(
        [{"role": "system", "content": _REFLECT_SYSTEM}, {"role": "user", "content": material}],
    )
    return _parse(raw, _to_insight) or []


# ---------------- 公共解析 ----------------

def _user_text(new_messages: list[dict]) -> str:
    return "\n".join(m["content"] for m in new_messages if m.get("role") == "user").strip()


def _parse(raw: str, mapper):
    match = re.search(r"\[.*\]", raw.strip(), re.S)  # 容忍 ```json 包裹/前后多余文字
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None
    out = []
    for it in data:
        mapped = mapper(it)
        if mapped is not None:
            out.append(mapped)
    return out


def _nullable(v) -> str | None:
    """把 ""/"null"/"none"/None 统一成 None（LLM 常把空值写成字符串 "null"）。"""
    if v is None:
        return None
    s = str(v).strip()
    return None if not s or s.lower() in ("null", "none", "n/a") else s


def _to_fact(it) -> FactCandidate | None:
    if not isinstance(it, dict):
        return None
    try:
        return FactCandidate(
            category=str(it["category"]).strip().lower(),
            key=str(it["key"]).strip().lower(),
            value=str(it["value"]).strip(),
            confidence=float(it.get("confidence", 0.6)),
            canonical_key=_nullable(it.get("canonical_key")),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _to_routed(it) -> RoutedMemory | None:
    if not isinstance(it, dict):
        return None
    content = str(it.get("content", "")).strip()
    mtype = str(it.get("memory_type", "")).strip().lower()
    if not content or not mtype:
        return None
    kws = it.get("keywords", [])
    keywords = [str(k).strip() for k in kws if str(k).strip()] if isinstance(kws, list) else []
    try:
        imp10 = float(it.get("importance", 5))
    except (TypeError, ValueError):
        imp10 = 5.0
    try:
        conf = float(it.get("confidence", 0.6))
    except (TypeError, ValueError):
        conf = 0.6
    sens = str(it.get("sensitivity", "low")).strip().lower()
    if sens not in ("low", "personal", "care"):
        sens = "low"
    return RoutedMemory(
        memory_type=mtype, content=content,
        canonical_key=_nullable(it.get("canonical_key")),
        event_at=_nullable(it.get("event_at")), expires_at=_nullable(it.get("expires_at")),
        keywords=keywords,
        importance=min(1.0, max(0.0, imp10 / 10.0)),
        confidence=min(1.0, max(0.0, conf)),
        sensitivity=sens,
        reason=str(it.get("reason", "")).strip(),
    )


def _to_memory(it) -> MemoryCandidate | None:
    if not isinstance(it, dict):
        return None
    content = str(it.get("content", "")).strip()
    if not content:
        return None
    kws = it.get("keywords", [])
    keywords = [str(k).strip() for k in kws if str(k).strip()] if isinstance(kws, list) else []
    try:
        imp10 = float(it.get("importance", 5))
    except (TypeError, ValueError):
        imp10 = 5.0
    importance = min(1.0, max(0.0, imp10 / 10.0))  # 1~10 → 0..1
    return MemoryCandidate(content=content, keywords=keywords, importance=importance)


# ---------------- 启发式兜底（仅画像） ----------------

_STOP = "[^\\s，。,.!！？?、]{1,12}"
_PATTERNS = [
    ("bio", "name", re.compile(rf"我(?:叫|的名字是)\s*({_STOP})")),
    ("preference", "like", re.compile(rf"我(?:很|超|特别)?(?:喜欢|爱)(?:喝|吃|玩|看|听)?\s*({_STOP})")),
    ("taboo", "dislike", re.compile(rf"我(?:很|超|特别)?(?:不喜欢|讨厌|受不了)\s*({_STOP})")),
    ("routine", "sleep_time", re.compile(r"每天\s*(\d{1,2})\s*点(?:睡觉|睡|入睡)")),
]


def _heuristic_facts(text: str) -> list[FactCandidate]:
    out: list[FactCandidate] = []
    for category, key, pat in _PATTERNS:
        m = pat.search(text)
        if m:
            out.append(FactCandidate(category=category, key=key, value=m.group(1).strip(), confidence=0.6))
    return out
