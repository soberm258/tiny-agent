import sys

sys.path.append(".")

import pytest

def test_deep_research_report_body_has_no_source() -> None:
    try:
        from agent.tools.deep_research import _render_markdown_report
    except ModuleNotFoundError as e:
        pytest.skip(f"缺少依赖，跳过：{e}", allow_module_level=False)

    from agent.tools.schemas import Evidence

    evidence = [Evidence(source_type="rag", source=f"rag:law:{i}", title=f"T{i}", snippet=f"S{i}") for i in range(1, 6)]
    sections = {
        "一、任务与结论摘要": "结论A。[1]\n结论B。[5]",
        "二、背景与问题拆解": "背景说明。[1]",
        "三、证据与分析": "分析段落。[1] [5]",
        "四、风险、不确定性与需要补充的信息": "不确定性说明。",
    }
    md, _m, _old = _render_markdown_report(sections=sections, evidence=evidence, used_ids=[1, 5])
    body, refs = md.split("## 五、引用列表", 1)
    assert "source=" not in body
    assert "source=" in refs
    assert "[5]" not in body
    assert "[2]" in body


def test_rag_law_title_prefers_snippet_article() -> None:
    try:
        from agent.tools.deep_research import _rag_items_to_evidence
    except ModuleNotFoundError as e:
        pytest.skip(f"缺少依赖，跳过：{e}", allow_module_level=False)

    obs = (
        "[1] 第一百三十三条 之一 在道路上醉酒驾驶机动车的，处拘役，并处罚金。\n"
        "source=data\\raw_data\\law\\xingfa.txt | 中华人民共和国刑法 | 第二编 分则 | 第二章 危害公共安全罪 | 未分节 | 第一百三十三条\n"
    )
    evs = _rag_items_to_evidence(obs=obs, db="law", query="醉驾 危险驾驶罪 条款", max_items=5)
    assert evs
    assert "中华人民共和国刑法" in evs[0].title
    assert "第一百三十三条之一" in evs[0].title
    assert "source=" not in evs[0].snippet
