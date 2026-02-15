_REPORT_KEYWORDS = ("详细报告", "研究报告", "调研", "完整分析", "写一份报告", "深度报告", "deep research")


def should_use_deep_research(text: str) -> bool:
    t = (text or "").strip().lower()
    return any(k.lower() in t for k in _REPORT_KEYWORDS)

