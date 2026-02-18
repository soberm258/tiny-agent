from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    title: str = Field(default="")
    url: str = Field(description="网页 URL")
    snippet: str = Field(default="")
    provider: Literal["tavily", "serpapi"]


class Evidence(BaseModel):
    source_type: Literal["rag", "web"]
    source: str = Field(description="source=路径或URL")
    title: str = Field(default="")
    snippet: str = Field(default="")
    meta: Dict[str, Any] = Field(default_factory=dict)

    def cite_line(self, idx: int) -> str:
        title = self.title.strip() or "证据"
        snippet = (self.snippet or "").strip()
        if len(snippet) > 240:
            snippet = snippet[:240].rstrip() + "…"
        return f"[{idx}] {title} source={self.source}\n{snippet}".strip()
    
    def full_text(self,idx: int) -> str:
        title = self.title.strip() or "证据"
        snippet = (self.snippet or "").strip()
        return f"[{idx}] {title}\nsource_type={self.source_type} \nsource={self.source}\n{snippet}".strip()

class DeepResearchPlanItem(BaseModel):
    question: str
    rag_query: str = Field(default="")
    web_query: str = Field(default="")


class DeepResearchPlan(BaseModel):
    sub_questions: list[DeepResearchPlanItem]

