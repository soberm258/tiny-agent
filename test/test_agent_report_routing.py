from agent.routing import should_use_deep_research


def test_should_use_deep_research_keywords() -> None:
    assert should_use_deep_research("请给我一份详细报告：关于不可抗力的法律责任")
    assert should_use_deep_research("写一份报告，包含引用")
    assert not should_use_deep_research("什么是不可抗力？请简单解释")
