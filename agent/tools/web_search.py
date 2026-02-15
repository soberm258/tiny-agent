from __future__ import annotations

import json
import os
import ssl
import urllib.parse
import urllib.error
import urllib.request
from typing import Any, Dict, List

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from agent.tools.schemas import SearchResult


def _http_json(url: str, *, method: str = "GET", body: Dict[str, Any] | None = None, timeout_s: int = 10) -> Any:
    data = None
    headers = {"Accept": "application/json", "User-Agent": "tiny-agents/1.0"}
    if body is not None:
        raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
        data = raw
        headers["Content-Type"] = "application/json; charset=utf-8"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    ctx = ssl.create_default_context()
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    try:
        with urllib.request.urlopen(req, timeout=timeout_s, context=ctx) as resp:
            text = resp.read().decode("utf-8", errors="replace")
        return json.loads(text)
    except Exception as e:
        return {"error": str(e), "url": url}


def _tavily_search(query: str, max_results: int) -> List[SearchResult]:
    key = os.environ["TAVILY_API_KEY"]
    obj = _http_json(
        "https://api.tavily.com/search",
        method="POST",
        body={"api_key": key, "query": query, "max_results": int(max_results), "include_raw_content": False},
    )
    out: List[SearchResult] = []
    # title: str = Field(default="")
    # url: str = Field(description="网页 URL")
    # snippet: str = Field(default="")
    # provider: Literal["tavily", "serpapi"]
    for it in (obj or {}).get("results", []) if isinstance(obj, dict) else []:
        url = str(it.get("url") or "")
        if not url:
            continue
        out.append(
            SearchResult(
                title=str(it.get("title") or ""),
                url=url,
                snippet=str(it.get("content") or it.get("snippet") or ""),
                provider="tavily",
            )
        )
    return out


def _serpapi_search(query: str, max_results: int) -> List[SearchResult]:
    key = os.environ["SERPAPI_API_KEY"]
    params = {
        "engine": "google",
        "q": query,
        "api_key": key,
        "num": int(max_results),
    }
    url = "https://serpapi.com/search.json?" + urllib.parse.urlencode(params)
    obj = _http_json(url)
    out: List[SearchResult] = []
    organic = (obj or {}).get("organic_results", []) if isinstance(obj, dict) else []
    for it in organic:
        link = str(it.get("link") or "")
        if not link:
            continue
        out.append(
            SearchResult(
                title=str(it.get("title") or ""),
                url=link,
                snippet=str(it.get("snippet") or ""),
                provider="serpapi",
            )
        )
    return out


def _merge_dedupe(results: List[SearchResult], max_results: int) -> List[SearchResult]:
    seen: Dict[str, SearchResult] = {}
    for r in results:
        if r.url not in seen:
            seen[r.url] = r
    return list(seen.values())[: int(max_results)]


class WebSearchInput(BaseModel):
    query: str = Field(description="搜索查询（必填）")
    max_results: int = Field(default=5, ge=1, le=10, description="返回条数（1~10）")


@tool("web_search", args_schema=WebSearchInput)
def web_search(query: str, max_results: int = 5) -> str:
    """在线搜索聚合工具（Tavily + SerpAPI）。

    输入 query，输出 JSON 字符串：{"query": "...", "results": [{"title","url","snippet","provider"}]}。
    适用于“最新/近期/网页公开信息”等需要联网补充证据的场景。
    """
    q = (query or "").strip()
    max_results = int(max_results) if max_results is not None else 5
    tavily = _tavily_search(q, max_results=max_results)
    serp = _serpapi_search(q, max_results=max_results)
    merged = _merge_dedupe([*tavily, *serp], max_results=max_results)
    payload = {"query": q, "results": [r.model_dump() for r in merged]}
    return json.dumps(payload, ensure_ascii=False)
