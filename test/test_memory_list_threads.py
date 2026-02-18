from __future__ import annotations

import os
import tempfile

from agent.memory.store import MemoryStore


def test_list_threads_title_from_last_user_message() -> None:
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "memory.sqlite")
        store = MemoryStore(db_path=db_path)
        store.append_message("t1", "user", "第一条用户消息：这是一个会话标题示例")
        store.append_message("t1", "assistant", "好的")
        store.append_message("t2", "user", "另一个会话")
        threads = store.list_threads(limit=10)
        ids = [t["thread_id"] for t in threads]
        assert "t1" in ids and "t2" in ids
        t1 = [t for t in threads if t["thread_id"] == "t1"][0]
        assert t1["title"].startswith("第一条用户消息")

