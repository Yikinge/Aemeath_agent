"""确定性单测：叙事 resolve + 画像实体合并（方案 §6.3 / §7）。

核心验收：
- 同一件事说两遍（SAME）→ 只保留一条 canonical 叙事，不新增；
- 补充信息（MERGE）→ 合并成一条更完整的，旧的被 supersede（不再活跃）；
- 同一实体被拆成不同 key（pet_cat / pet_species）→ 靠 canonical_key 并成一条活跃画像。
"""

from __future__ import annotations

import json

from agent.memory.consolidator import Consolidator
from tests.conftest import PromptRouter

_THU = "2026-06-18T10:00:00+08:00"


def _event(content, canonical="event:hike:xiangshan", imp=6):
    return json.dumps([{"memory_type": "event_memory", "content": content,
                        "canonical_key": canonical, "importance": imp}])


async def _run(store, c, text):
    await store.add_pending_intake(text, source_at=_THU, timezone="Asia/Shanghai")
    return await c.consolidate()


async def test_same_narrative_not_duplicated(store, tmp_path):
    router = PromptRouter(route=_event("用户上周去爬了香山"))
    c = Consolidator(store, router, str(tmp_path / "MEMORY.md"))
    await _run(store, c, "我上周去爬了香山")
    assert len(await store.list_narratives()) == 1
    mem_before = (await store.active_memory_items())[0]

    # 第二次同一件事（不同措辞，同 canonical_key）→ judge SAME → 不新增，强化原碎片
    router.route = _event("用户上周末爬了香山，挺开心")
    router.verdict = "SAME"
    await _run(store, c, "对了上周末爬香山玩得挺开心")

    narrs = await store.list_narratives()
    assert len(narrs) == 1                       # 仍只有一条 canonical 叙事
    mem_after = await store.get_memory_item(mem_before.id)
    assert mem_after.strength > mem_before.strength   # 命中被强化（间隔重复）


async def test_merge_narrative_into_one_canonical(store, tmp_path):
    router = PromptRouter(route=_event("用户养了一只橘猫", canonical="entity:pet:cat"))
    c = Consolidator(store, router, str(tmp_path / "MEMORY.md"))
    await _run(store, c, "我养了只橘猫")

    router.route = _event("用户的橘猫叫煤球", canonical="entity:pet:cat")
    router.verdict = "MERGE"
    router.merged = "用户养了一只叫煤球的橘猫"
    await _run(store, c, "我的橘猫叫煤球")

    narrs = await store.list_narratives()
    assert len(narrs) == 1                       # 合并后只剩一条
    assert narrs[0]["content"] == "用户养了一只叫煤球的橘猫"
    assert len(await store.active_memory_items()) == 1   # 旧碎片已级联失效


async def test_profile_entity_keys_merge_via_canonical(store, tmp_path):
    # 同一只猫被拆成 pet_cat / pet_species，但共享 canonical_key → 合并成一条、且不丢信息（#3）
    router = PromptRouter(profile=json.dumps(
        [{"category": "entity", "key": "pet_cat", "value": "煤球", "canonical_key": "entity:pet:cat"}]))
    c = Consolidator(store, router, str(tmp_path / "MEMORY.md"))
    await _run(store, c, "我养了只猫叫煤球")

    router.profile = json.dumps(
        [{"category": "entity", "key": "pet_species", "value": "橘猫", "canonical_key": "entity:pet:cat"}])
    router.merged = "煤球（橘猫）"   # 实体 MERGE 合并值
    await _run(store, c, "我的猫是橘猫")

    facts = await store.all_active_facts()
    cat_facts = [f for f in facts if f.canonical_key == "entity:pet:cat"]
    assert len(cat_facts) == 1                       # 不再 pet_cat / pet_species 两条并存
    assert "煤球" in cat_facts[0].value and "橘猫" in cat_facts[0].value  # 合并不丢信息
