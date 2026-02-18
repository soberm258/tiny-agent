from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from agent.prompts.memory import MEMORY_UPDATE_SYSTEM_PROMPT


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryStore:
    def __init__(self, db_path: str = "data/agent_memory/memory.sqlite") -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    @contextmanager
    def _connect_cm(self) -> sqlite3.Connection:
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            raise
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _ensure_schema(self) -> None:
        with self._connect_cm() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS threads ("
                "thread_id TEXT PRIMARY KEY,"
                "created_at TEXT,"
                "updated_at TEXT"
                ");"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS messages ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "thread_id TEXT,"
                "role TEXT,"
                "content TEXT,"
                "meta_json TEXT,"
                "created_at TEXT"
                ");"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_thread_time ON messages(thread_id, created_at);")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS summaries ("
                "thread_id TEXT PRIMARY KEY,"
                "summary TEXT,"
                "facts_json TEXT,"
                "updated_at TEXT"
                ");"
            )

    def _touch_thread(self, thread_id: str) -> None:
        now = _now_iso()
        with self._connect_cm() as conn:
            conn.execute(
                "INSERT INTO threads(thread_id, created_at, updated_at) VALUES(?,?,?) "
                "ON CONFLICT(thread_id) DO UPDATE SET updated_at=excluded.updated_at;",
                (thread_id, now, now),
            )

    def append_message(self, thread_id: str, role: str, content: str, meta: Dict[str, Any] | None = None) -> None:
        self._touch_thread(thread_id)
        with self._connect_cm() as conn:
            conn.execute(
                "INSERT INTO messages(thread_id, role, content, meta_json, created_at) VALUES(?,?,?,?,?);",
                (thread_id, role, content, json.dumps(meta or {}, ensure_ascii=False), _now_iso()),
            )

    def get_recent_messages(self, thread_id: str, limit: int = 20) -> List[Dict[str, str]]:
        with self._connect_cm() as conn:
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE thread_id=? ORDER BY id DESC LIMIT ?;",
                (thread_id, int(limit)),
            ).fetchall()
        rows.reverse()
        return [{"role": str(r), "content": str(c)} for (r, c) in rows]

    def get_summary(self, thread_id: str) -> Tuple[str, Dict[str, Any]]:
        with self._connect_cm() as conn:
            row = conn.execute("SELECT summary, facts_json FROM summaries WHERE thread_id=?;", (thread_id,)).fetchone()
        if not row:
            return "", {}
        summary, facts_json = row
        try:
            facts = json.loads(facts_json or "{}")
        except Exception:
            facts = {}
        return str(summary or ""), facts if isinstance(facts, dict) else {}

    def list_threads(self, limit: int = 50) -> List[Dict[str, str]]:
        limit = int(limit) if limit is not None else 50
        limit = max(1, min(200, limit))
        with self._connect_cm() as conn:
            rows = conn.execute(
                "SELECT thread_id, updated_at FROM threads ORDER BY updated_at DESC LIMIT ?;",
                (limit,),
            ).fetchall()

        out: List[Dict[str, str]] = []
        for thread_id, updated_at in rows:
            tid = str(thread_id or "").strip()
            if not tid:
                continue
            title = ""
            with self._connect_cm() as conn:
                row = conn.execute(
                    "SELECT content FROM messages WHERE thread_id=? AND role='user' ORDER BY id DESC LIMIT 1;",
                    (tid,),
                ).fetchone()
            if row:
                title = str(row[0] or "").strip().replace("\r", " ").replace("\n", " ")
                title = " ".join([x for x in title.split(" ") if x]).strip()[:20]
            if not title:
                title = tid[-8:]
            out.append({"thread_id": tid, "updated_at": str(updated_at or ""), "title": title})
        return out

    def upsert_summary(self, thread_id: str, summary: str, facts: Dict[str, Any]) -> None:
        self._touch_thread(thread_id)
        with self._connect_cm() as conn:
            conn.execute(
                "INSERT INTO summaries(thread_id, summary, facts_json, updated_at) VALUES(?,?,?,?) "
                "ON CONFLICT(thread_id) DO UPDATE SET summary=excluded.summary, facts_json=excluded.facts_json, updated_at=excluded.updated_at;",
                (thread_id, summary, json.dumps(facts or {}, ensure_ascii=False), _now_iso()),
            )

    def update_summary_with_llm(
        self,
        *,
        thread_id: str,
        prev_summary: str,
        prev_facts: Dict[str, Any],
        user_text: str,
        assistant_text: str,
    ) -> None:
        from langchain_openai import ChatOpenAI

        model_id = os.environ.get("LLM_MODEL_ID", "Qwen/Qwen3-32B")
        base_url = os.environ.get("LLM_BASE_URL", "https://api-inference.modelscope.cn/v1")
        api_key = os.environ.get("LLM_API_KEY", "")
        model = ChatOpenAI(
            model=model_id,
            base_url=base_url,
            api_key=api_key,
            streaming=False,
            extra_body={"enable_thinking": False},
        )

        payload = (
            "【旧摘要】\n"
            f"{prev_summary}\n\n"
            "【旧关键事实JSON】\n"
            f"{json.dumps(prev_facts or {}, ensure_ascii=False)}\n\n"
            "【本轮用户】\n"
            f"{user_text}\n\n"
            "【本轮助手】\n"
            f"{assistant_text}\n"
        )
        try:
            resp = model.invoke(
                [
                    {"role": "system", "content": MEMORY_UPDATE_SYSTEM_PROMPT},
                    {"role": "user", "content": payload},
                ]
            )
            text = getattr(resp, "content", "") if resp is not None else ""
        except Exception:
            return

        def _extract_json(s: str) -> str:
            t = str(s or "").lstrip("\ufeff").strip()
            if t.startswith("```") and "```" in t[3:]:
                parts = t.split("```")
                t = parts[1] if len(parts) >= 2 else t
                t = t.strip()
                if t.lower().startswith("json"):
                    t = "\n".join(t.splitlines()[1:]).strip()
            if not t.startswith("{"):
                a, b = t.find("{"), t.rfind("}")
                if 0 <= a < b:
                    t = t[a : b + 1]
            return t

        try:
            obj = json.loads(_extract_json(text))
            summary = str(obj.get("summary") or "").strip()
            facts = obj.get("facts") or {}
            facts = facts if isinstance(facts, dict) else {}
        except Exception:
            summary = str(text or "").strip()
            facts = prev_facts or {}
        self.upsert_summary(thread_id, summary, facts)
