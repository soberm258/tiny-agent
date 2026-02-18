from __future__ import annotations

from script.evidence import parse_rag_observation


def test_parse_rag_observation_basic() -> None:
    raw = (
        "[1] 这是第一条证据内容。\n"
        "source=data/raw_data/law/xingfa.txt | 中华人民共和国刑法 | 第一百三十三条之一\n"
        "[2] 这是第二条证据内容，可能有多行。\n"
        "补充一行。\n"
        "source=https://example.com/a\n"
    )
    items = parse_rag_observation(raw)
    assert len(items) == 2
    assert items[0].tool == "rag_search"
    assert "第一条证据" in items[0].snippet
    assert "xingfa" in items[0].source
    assert items[1].source == "https://example.com/a"
