"""记忆质量评测（方案 P3）：跑 golden_memory_cases.jsonl，看抽取/路由/归一化/resolve 的质量。

和确定性单测分开（LLM 输出有方差，不进 CI）。每次改抽取 prompt 或 resolve 逻辑后手动跑：
    python -m agent.eval.memory_eval                 # 跑全部
    python -m agent.eval.memory_eval rel_date short  # 只跑 id 含这些子串的用例

用真 fast 模型 + 真/兜底 embedding；没配 Key 时优雅跳过（不报错）。每个用例独立临时库。
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

from agent.config import load_config
from agent.gateway.router import LLMRouter
from agent.memory.consolidator import Consolidator
from agent.memory.store import Store

_CASES = Path(__file__).resolve().parent.parent.parent / "tests" / "golden_memory_cases.jsonl"
_RELATIVE_WORDS = ("今天", "明天", "昨天", "后天", "前天", "下周", "上周", "这周", "本周", "下个月", "大后天")


def _load_cases(filters: list[str]) -> list[dict]:
    cases = [json.loads(line) for line in _CASES.read_text(encoding="utf-8").splitlines() if line.strip()]
    if filters:
        cases = [c for c in cases if any(f in c["id"] for f in filters)]
    return cases


async def _run_case(case: dict, router: LLMRouter, tz: str) -> list[str]:
    """跑一个用例，返回未通过的检查项（空 = 全过）。"""
    tmp = tempfile.mktemp(suffix=".db")
    md = tempfile.mktemp(suffix=".md")
    store = Store(tmp)
    await store.init()
    cons = Consolidator(store, router, md, timezone=tz)
    try:
        for msg in case["messages"]:
            await store.add_pending_intake(msg, source_at=case.get("source_at"), timezone=tz)
            await cons.consolidate()           # 逐条巩固 → 第二条能与第一条 resolve

        facts = await store.all_active_facts()
        narrs = await store.list_narratives()
        commits = await store.list_all_commitments(status="open")
        moods = await store.list_mood(n=10)
        md_text = cons.read_memory_md()
        return _check(case.get("expect", {}), facts, narrs, commits, moods, md_text)
    finally:
        await store.close()
        for p in (tmp, md):
            Path(p).unlink(missing_ok=True)


def _check(exp, facts, narrs, commits, moods, md_text) -> list[str]:
    fails: list[str] = []

    def bad(name, detail):
        fails.append(f"{name}: {detail}")

    if "commitment_event_at" in exp:
        want = exp["commitment_event_at"]
        got = [c.get("event_at") for c in commits]
        if want not in got:
            bad("commitment_event_at", f"想要 {want}，实际 open 承诺 event_at={got}")
    if exp.get("has_open_commitment") is True and not commits:
        bad("has_open_commitment", "没有 open 承诺")
    if exp.get("has_open_commitment") is False and commits:
        bad("has_open_commitment", f"不该有 open 承诺，却有 {[c['content'] for c in commits]}")
    if exp.get("no_open_commitment") and commits:
        bad("no_open_commitment", f"有 {len(commits)} 条 open 承诺")
    if "entity_value_contains" in exp:
        sub = exp["entity_value_contains"]
        if not any(sub in f.value for f in facts):
            bad("entity_value_contains", f"画像里没有含「{sub}」的事实：{[f.value for f in facts]}")
    if "profile_category_in" in exp:
        cats = set(exp["profile_category_in"])
        if not any(f.category in cats for f in facts):
            bad("profile_category_in", f"没有 {cats} 类画像，实际 {[f.category for f in facts]}")
    if "narrative_count_max" in exp and len(narrs) > exp["narrative_count_max"]:
        bad("narrative_count_max", f"叙事 {len(narrs)} 条 > {exp['narrative_count_max']}：{[n['content'] for n in narrs]}")
    if exp.get("no_narrative") and narrs:
        bad("no_narrative", f"不该有叙事，却有 {[n['content'] for n in narrs]}")
    if exp.get("mood_logged") and not moods:
        bad("mood_logged", "没记到情绪")
    if exp.get("no_memory"):
        if facts or narrs or commits:
            bad("no_memory", f"应什么都不沉淀，却有 画像{len(facts)}/叙事{len(narrs)}/承诺{len(commits)}")
    if exp.get("md_no_relative"):
        leaked = [w for w in _RELATIVE_WORDS if w in md_text]
        if leaked:
            bad("md_no_relative", f"MEMORY.md 漏了相对词 {leaked}")
    if "md_contains" in exp and exp["md_contains"] not in md_text:
        bad("md_contains", f"MEMORY.md 不含「{exp['md_contains']}」")
    return fails


async def _amain(filters: list[str]) -> int:
    cfg = load_config()
    router = LLMRouter(
        cfg.default_model, cfg.fast_model,
        embed_model=cfg.embed_model, embed_base_url=cfg.embed_base_url, embed_api_key=cfg.embed_api_key,
    )
    if not router.live("fast"):
        print("⚠ 未配置 fast 模型 API Key —— 记忆质量评测需要真模型，跳过。")
        print("  配好 DEEPSEEK_API_KEY（或对应 provider）后再跑：python -m agent.eval.memory_eval")
        return 0

    cases = _load_cases(filters)
    print(f"记忆质量评测 · {len(cases)} 个用例 · fast={cfg.fast_model}\n" + "─" * 60)
    passed = 0
    for case in cases:
        try:
            fails = await _run_case(case, router, cfg.memory_timezone)
        except Exception as e:  # 单个用例炸了不影响其余
            fails = [f"运行异常: {e!r}"]
        if not fails:
            passed += 1
            print(f"✅ {case['id']:32} {case['desc']}")
        else:
            print(f"❌ {case['id']:32} {case['desc']}")
            for f in fails:
                print(f"      - {f}")
    print("─" * 60)
    print(f"通过 {passed}/{len(cases)}")
    return 0


def main() -> None:
    filters = [a for a in sys.argv[1:] if not a.startswith("-")]
    raise SystemExit(asyncio.run(_amain(filters)))


if __name__ == "__main__":
    main()
