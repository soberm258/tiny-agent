from agent.memory import MemoryStore
from pathlib import Path
import uuid


def test_memory_store_roundtrip() -> None:
    base = Path("data") / "agent_memory"
    base.mkdir(parents=True, exist_ok=True)
    db_path = base / f"test-memory-{uuid.uuid4().hex}.sqlite"
    store = MemoryStore(str(db_path))
    thread_id = f"t-{uuid.uuid4().hex}"
    store.append_message(thread_id, "user", "你好，我想咨询合同法问题。")
    store.append_message(thread_id, "assistant", "我会先检索法条再回答。")
    msgs = store.get_recent_messages(thread_id, limit=10)
    assert msgs[-1]["role"] == "assistant"
    assert "检索" in msgs[-1]["content"]

    store.upsert_summary(thread_id, "摘要", {"constraints": [], "preferences": [], "open_questions": []})
    summary, facts = store.get_summary(thread_id)
    assert summary == "摘要"
    assert isinstance(facts, dict)

    try:
        db_path.unlink(missing_ok=True)
    except Exception:
        pass
