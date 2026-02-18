from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Literal, Optional


@dataclass(frozen=True)
class EvidenceItem:
    tool: str
    title: str
    snippet: str
    source: str
    kind: Literal["rag", "web", "other"] = "other"


_RE_RAG_ITEM = re.compile(r"^\[(\d+)\]\s*(.*)$")


def parse_rag_observation(text: str) -> List[EvidenceItem]:
    """
    解析 tinyrag 的 rag_search Observation 文本，提取结构化证据条目。

    典型格式：
      [1] 证据文本...
      source=...
      [2] ...
      source=...
    """
    t = (text or "").strip()
    if not t:
        return []
    lines = [ln.rstrip() for ln in t.splitlines() if ln.strip()]
    out: List[EvidenceItem] = []
    cur_rank: Optional[int] = None
    cur_text_lines: List[str] = []
    for ln in lines:
        m = _RE_RAG_ITEM.match(ln)
        if m:
            cur_rank = int(m.group(1))
            cur_text_lines = [m.group(2).strip()]
            continue
        if ln.startswith("source=") and cur_rank is not None:
            source = ln[len("source=") :].strip()
            snippet = "\n".join([x for x in cur_text_lines if x]).strip()
            title = f"RAG[{cur_rank}]"
            out.append(EvidenceItem(tool="rag_search", title=title, snippet=snippet, source=source, kind="rag"))
            cur_rank = None
            cur_text_lines = []
            continue
        if cur_rank is not None:
            cur_text_lines.append(ln.strip())
    return out

