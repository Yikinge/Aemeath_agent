import json

import pytest

from agent.memory.consolidator import Consolidator
from agent.memory.journal import JournalEntry
from agent.memory.service import MemoryService


@pytest.mark.asyncio
async def test_consolidation_projects_daily_journal(store, tmp_path):
    from tests.conftest import PromptRouter

    router = PromptRouter(
        profile=json.dumps([
            {
                "category": "preference",
                "key": "memory_design_style",
                "value": "偏好结构化、可审计的记忆设计",
                "confidence": 0.9,
                "canonical_key": None,
            }
        ], ensure_ascii=False),
        route=json.dumps([
            {
                "memory_type": "event_memory",
                "content": "用户在 2026-07-02 讨论每日 Markdown 记忆设计",
                "canonical_key": "event:daily_journal_design:2026-07-02",
                "event_at": "2026-07-02",
                "expires_at": None,
                "keywords": ["每日记忆", "Markdown"],
                "importance": 8,
                "confidence": 0.9,
                "sensitivity": "low",
                "reason": "正在做长期记忆架构设计",
            }
        ], ensure_ascii=False),
        commitments=json.dumps([
            {
                "kind": "open_loop",
                "content": "继续完善每日记忆总方案",
                "event_at": None,
                "due_at": None,
                "due_window_start": None,
                "due_window_end": None,
                "expires_at": None,
                "canonical_key": "loop:daily_journal_design",
                "dedupe_key": "daily_journal_design",
                "confidence": 0.8,
                "sensitivity": "routine",
                "reason": "用户要求继续实现总方案",
            }
        ], ensure_ascii=False),
        mood='{"valence":0.2,"arousal":0.4,"signals":["设计"],"note":"专注于记忆架构"}',
    )
    cons = Consolidator(store, router, str(tmp_path / "MEMORY.md"))

    await store.add_pending_intake(
        "我们来设计每日 Markdown 记忆",
        source_at="2026-07-02T10:00:00+08:00",
        timezone="Asia/Shanghai",
    )
    await cons.consolidate()

    text = cons.journal.read_day("2026-07-02")
    assert "# Daily Memory: 2026-07-02" in text
    assert "[10:00] 用户在 2026-07-02 讨论每日 Markdown 记忆设计" in text
    assert "## 事件" in text
    assert "用户在 2026-07-02 讨论每日 Markdown 记忆设计" in text
    assert "## 画像与偏好变化" in text
    assert "偏好结构化、可审计的记忆设计" in text
    assert "## 开放回路" in text
    assert "[10:00] [open] 继续完善每日记忆总方案" in text
    assert "继续完善每日记忆总方案" in text
    assert "## 状态" in text
    assert "[10:00] 专注于记忆架构" in text
    assert "专注于记忆架构" in text

    rows = await store.list_all_commitments()
    assert rows[0]["created_at"] == "2026-07-02T10:00:00+08:00"
    moods = await store.list_mood()
    assert moods[0]["ts"] == "2026-07-02T10:00:00+08:00"


@pytest.mark.asyncio
async def test_journal_append_is_idempotent(store, tmp_path):
    cons = Consolidator(store, router=None, md_path=str(tmp_path / "MEMORY.md"))  # type: ignore[arg-type]
    entry = JournalEntry(
        section="整理日志",
        marker="unit:once",
        content="同一条整理日志只写一次。",
        occurred_at="2026-07-02T10:00:00+08:00",
    )

    assert cons.journal.append(entry) is True
    assert cons.journal.append(entry) is False

    text = cons.journal.read_day("2026-07-02")
    assert text.count("同一条整理日志只写一次。") == 1


@pytest.mark.asyncio
async def test_commitment_status_changes_project_to_journal(store, router, tmp_path):
    cons = Consolidator(store, router, str(tmp_path / "MEMORY.md"))
    memory = MemoryService(store, router, cons, consolidate_threshold=99)
    cid = await store.add_commitment(
        "open_loop",
        "跟进每日记忆实现",
        None,
        namespace="default",
        canonical_key="loop:daily_journal_impl",
    )
    cons.journal.append(cons.journal.commitment_entry(cid, "跟进每日记忆实现", status="open"))

    await memory.mark_commitment_sent(cid)
    await memory.complete_commitment(cid)

    today = cons.journal.path_for().read_text(encoding="utf-8")
    assert "[open] 跟进每日记忆实现" in today
    assert "[sent] 已主动跟进：跟进每日记忆实现" in today
    assert "[done] 已闭合：跟进每日记忆实现" in today


@pytest.mark.asyncio
async def test_rebuild_day_from_structured_tables(store, router, tmp_path):
    cons = Consolidator(store, router, str(tmp_path / "MEMORY.md"))
    await store.add_narrative(
        "用户在 2026-07-02 确认每日记忆采用 DB 投影方案",
        kind="event",
        event_at="2026-07-02",
    )
    path = cons.journal.path_for("2026-07-02")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# broken\n\n手工写坏的内容\n", encoding="utf-8")

    text = await cons.journal.rebuild_day(store, "2026-07-02")

    assert "# Daily Memory: 2026-07-02" in text
    assert "手工写坏的内容" not in text
    assert "用户在 2026-07-02 确认每日记忆采用 DB 投影方案" in text
