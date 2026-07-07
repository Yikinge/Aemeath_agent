"""Daily Markdown memory projection.

The structured SQLite tables remain the source of truth.  This module writes a
human-readable, per-day journal from committed memory changes so the user can
review "what happened today" without inflating MEMORY.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from agent.memory.models import ProfileFact, now_iso


SECTIONS = (
    "事件",
    "开放回路",
    "画像与偏好变化",
    "状态",
    "整理日志",
)


@dataclass(frozen=True)
class JournalEntry:
    section: str
    marker: str
    content: str
    occurred_at: str | None = None


class DailyJournal:
    """Append-only, idempotent writer for data/journal/YYYY-MM-DD.md."""

    def __init__(self, root_dir: str | Path, *, timezone_name: str = "Asia/Shanghai") -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.timezone_name = timezone_name
        try:
            self.tz = ZoneInfo(timezone_name)
        except Exception:
            self.tz = ZoneInfo("Asia/Shanghai")

    def path_for(self, date_str: str | None = None) -> Path:
        date_str = date_str or self._date_for(now_iso())
        return self.root_dir / f"{date_str}.md"

    def read_day(self, date_str: str | None = None) -> str:
        path = self.path_for(date_str)
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def append(self, entry: JournalEntry) -> bool:
        if entry.section not in SECTIONS:
            raise ValueError(f"unknown journal section: {entry.section}")
        date_str = self._date_for(entry.occurred_at)
        path = self.path_for(date_str)
        text = self._ensure_document(path, date_str)
        marker = f"<!-- {entry.marker} -->"
        if marker in text:
            return False

        item = f"{marker}\n- {self._format_time(entry.occurred_at)}{entry.content.strip()}\n"
        updated = self._insert_into_section(text, entry.section, item)
        path.write_text(updated, encoding="utf-8")
        return True

    async def rebuild_day(self, store, date_str: str, namespace: str = "default") -> str:
        """Recreate one day from current structured tables.

        This is intentionally a projection, not a verbatim replay of old file
        edits. Runtime append entries capture merge/prune details; rebuild gives
        a clean current-state view for the selected day.
        """
        path = self.path_for(date_str)
        path.write_text(self._blank_document(date_str), encoding="utf-8")

        for n in await store.list_narratives(namespace, n=1000):
            event_day = self._date_for(n.get("event_at") or n.get("created_at"))
            created_day = self._date_for(n.get("created_at"))
            if date_str not in {event_day, created_day}:
                continue
            if n["kind"] == "event":
                occurred = n.get("created_at") if created_day == date_str else f"{date_str}T00:00:00"
                self.append(JournalEntry(
                    section="事件",
                    marker=f"narrative:{n['id']}",
                    content=n["content"],
                    occurred_at=occurred,
                ))
            elif n["kind"] == "insight":
                self.append(JournalEntry(
                    section="整理日志",
                    marker=f"insight:{n['id']}",
                    content=f"反思洞察：{n['content']}",
                    occurred_at=n.get("created_at"),
                ))

        for c in await store.list_all_commitments(namespace, n=1000):
            created_day = self._date_for(c.get("created_at"))
            if created_day == date_str:
                self.append(JournalEntry(
                    section="开放回路",
                    marker=f"commitment:{c['id']}:open",
                    content=f"[open] {c['content']}",
                    occurred_at=c.get("created_at"),
                ))
            if c.get("sent_at") and self._date_for(c.get("sent_at")) == date_str:
                self.append(JournalEntry(
                    section="开放回路",
                    marker=f"commitment:{c['id']}:sent",
                    content=f"[sent] 已主动跟进：{c['content']}",
                    occurred_at=c.get("sent_at"),
                ))
            if c.get("completed_at") and self._date_for(c.get("completed_at")) == date_str:
                self.append(JournalEntry(
                    section="开放回路",
                    marker=f"commitment:{c['id']}:done",
                    content=f"[done] 已闭合：{c['content']}",
                    occurred_at=c.get("completed_at"),
                ))

        for f in await store.all_facts(namespace):
            if self._date_for(f.created_at) == date_str:
                self.append(self.profile_entry(f, "ADD"))
            elif self._date_for(f.updated_at) == date_str:
                self.append(self.profile_entry(f, "UPDATE"))

        for m in await store.list_mood(namespace, n=1000):
            if self._date_for(m.get("ts")) != date_str:
                continue
            note = (m.get("note") or "").strip()
            if not note:
                continue
            self.append(JournalEntry(
                section="状态",
                marker=f"mood:{m['id']}",
                content=self._mood_text(m),
                occurred_at=m.get("ts"),
            ))
        return self.read_day(date_str)

    def profile_entry(self, fact: ProfileFact, reason: str, previous: str | None = None) -> JournalEntry:
        if previous:
            content = f"[{reason}] {fact.key}：{previous} -> {fact.value}"
        else:
            content = f"[{reason}] {fact.key}：{fact.value}"
        return JournalEntry(
            section="画像与偏好变化",
            marker=f"profile:{fact.id}:{reason.lower()}",
            content=content,
            occurred_at=fact.updated_at or fact.created_at,
        )

    def narrative_entry(
        self, narrative_id: str, content: str, *, reason: str,
        event_at: str | None = None, created_at: str | None = None,
    ) -> JournalEntry:
        label = "" if reason == "ADD" else f"[{reason}] "
        return JournalEntry(
            section="事件",
            marker=f"narrative:{narrative_id}",
            content=f"{label}{content}",
            occurred_at=created_at or event_at,
        )

    def commitment_entry(
        self, commitment_id: str, content: str, *, status: str,
        occurred_at: str | None = None,
    ) -> JournalEntry:
        labels = {
            "open": "[open]",
            "sent": "[sent] 已主动跟进：",
            "done": "[done] 已闭合：",
            "expired": "[done] 超期兜底闭合：",
        }
        prefix = labels.get(status, f"[{status}] ")
        return JournalEntry(
            section="开放回路",
            marker=f"commitment:{commitment_id}:{status}",
            content=f"{prefix} {content}".replace("： ", "："),
            occurred_at=occurred_at,
        )

    def mood_entry(self, mood_id: str, mood: dict, *, occurred_at: str | None = None) -> JournalEntry:
        return JournalEntry(
            section="状态",
            marker=f"mood:{mood_id}",
            content=self._mood_text(mood),
            occurred_at=occurred_at,
        )

    def maintenance_entry(self, marker: str, content: str, *, occurred_at: str | None = None) -> JournalEntry:
        return JournalEntry(
            section="整理日志",
            marker=marker,
            content=content,
            occurred_at=occurred_at,
        )

    def _ensure_document(self, path: Path, date_str: str) -> str:
        if not path.exists():
            text = self._blank_document(date_str)
            path.write_text(text, encoding="utf-8")
            return text
        text = path.read_text(encoding="utf-8")
        changed = False
        if not text.startswith("# Daily Memory:"):
            text = f"# Daily Memory: {date_str}\n\n" + text.strip() + "\n"
            changed = True
        for section in SECTIONS:
            if f"## {section}" not in text:
                text = text.rstrip() + f"\n\n## {section}\n- 暂无\n"
                changed = True
        if changed:
            path.write_text(text, encoding="utf-8")
        return text

    @staticmethod
    def _blank_document(date_str: str) -> str:
        body = [f"# Daily Memory: {date_str}", ""]
        for section in SECTIONS:
            body.extend([f"## {section}", "- 暂无", ""])
        return "\n".join(body).rstrip() + "\n"

    def _insert_into_section(self, text: str, section: str, item: str) -> str:
        lines = text.splitlines()
        header = f"## {section}"
        try:
            start = lines.index(header)
        except ValueError:
            return text.rstrip() + f"\n\n{header}\n{item}"

        end = len(lines)
        for i in range(start + 1, len(lines)):
            if lines[i].startswith("## "):
                end = i
                break
        block = lines[start + 1:end]
        block = [line for line in block if line.strip() != "- 暂无"]
        if block and block[-1].strip():
            block.append("")
        block.extend(item.rstrip().splitlines())
        new_lines = lines[:start + 1] + block + lines[end:]
        return "\n".join(new_lines).rstrip() + "\n"

    def _date_for(self, value: str | None) -> str:
        dt = self._parse_dt(value)
        return dt.astimezone(self.tz).date().isoformat()

    def _format_time(self, value: str | None) -> str:
        dt = self._parse_dt(value)
        local = dt.astimezone(self.tz)
        return f"[{local.strftime('%H:%M')}] "

    @staticmethod
    def _mood_text(mood: dict) -> str:
        note = (mood.get("note") or "").strip()
        valence = mood.get("valence")
        if valence is None:
            tone = "状态未明"
        elif valence > 0.2:
            tone = "情绪偏好"
        elif valence < -0.2:
            tone = "情绪偏低落"
        else:
            tone = "情绪平稳"
        signals = "、".join(mood.get("signals") or [])
        suffix = f"；信号：{signals}" if signals else ""
        return f"{note or '记录到一次情绪状态'}（{tone}{suffix}）"

    def _parse_dt(self, value: str | None) -> datetime:
        if not value:
            return datetime.now(timezone.utc)
        s = str(value).strip()
        try:
            if len(s) == 10 and s[4] == "-" and s[7] == "-":
                return datetime.fromisoformat(s).replace(tzinfo=self.tz)
            dt = datetime.fromisoformat(s)
        except ValueError:
            return datetime.now(timezone.utc)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=self.tz)
        return dt
