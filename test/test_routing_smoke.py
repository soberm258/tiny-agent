from __future__ import annotations

from agent.routing import should_use_deep_research


def test_should_use_deep_research_keywords() -> None:
    assert should_use_deep_research("帮我生成一份详细报告")
    assert should_use_deep_research("deep research 一下这个问题")
    assert not should_use_deep_research("你好，解释一下这个概念")

