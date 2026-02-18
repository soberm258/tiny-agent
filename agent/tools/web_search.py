from __future__ import annotations

import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from agent.tools.schemas import SearchResult


_EXECUTOR = ThreadPoolExecutor(max_workers=2)


def _run(coro: Any, *, timeout_s: float = 30.0) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(asyncio.wait_for(coro, timeout=timeout_s))

    fut = _EXECUTOR.submit(lambda: asyncio.run(asyncio.wait_for(coro, timeout=timeout_s)))
    return fut.result(timeout=timeout_s + 2.0)


def _as_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, list):
        parts: List[str] = []
        for it in obj:
            if isinstance(it, str):
                parts.append(it)
            elif isinstance(it, dict):
                if "text" in it:
                    parts.append(str(it.get("text") or ""))
                elif "content" in it:
                    parts.append(str(it.get("content") or ""))
                else:
                    parts.append(json.dumps(it, ensure_ascii=False))
            else:
                parts.append(str(it))
        s = "\n".join([p for p in parts if p.strip()]).strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                parsed = json.loads(s)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
        return {}
    if isinstance(obj, str):
        s = obj.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                parsed = json.loads(s)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
        return {}
    content = getattr(obj, "content", None)
    if content is not None:
        return _as_dict(content)
    text = getattr(obj, "text", None)
    if text is not None:
        return _as_dict(text)
    try:
        dumped = json.loads(json.dumps(obj, ensure_ascii=False))
        return dumped if isinstance(dumped, dict) else {}
    except Exception:
        return {}


def _extract_items(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ("organic_results", "news_results", "results", "items", "data"):
        v = obj.get(key)
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict)]
        if isinstance(v, dict):
            for sub_key in ("organic_results", "results", "items"):
                vv = v.get(sub_key)
                if isinstance(vv, list):
                    return [x for x in vv if isinstance(x, dict)]
    return []


async def _mcp_serpapi_search(query: str, *, max_results: int, timeout_s: float) -> List[SearchResult]:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    api_key = (os.environ.get("SERPAPI_API_KEY") or "").strip()
    client = MultiServerMCPClient(
        {
            "serpapi": {
                "transport": "http",
                "url": f"https://mcp.serpapi.com/{api_key}/mcp",
            }
        }
    )
    tools = await asyncio.wait_for(client.get_tools(), timeout=timeout_s)
    search_tool = next(
        (t for t in tools if str(getattr(t, "name", "") or "").strip().lower() in {"search", "serpapi_search"}),
        None,
    )
    if search_tool is None:
        search_tool = next((t for t in tools if "search" in str(getattr(t, "name", "") or "").lower()), None)
    if search_tool is None and len(tools) == 1:
        search_tool = tools[0]
    if search_tool is None:
        return []

    raw = await asyncio.wait_for(
        search_tool.ainvoke(
            {
                "params": {"q": query, "num": int(max_results)},
                "mode": "complete",
            }
        ),
        timeout=timeout_s,
    )
    obj = _as_dict(raw)
    items = _extract_items(obj)

    out: List[SearchResult] = []
    for it in items:
        url = str(it.get("link") or it.get("url") or it.get("source") or it.get("canonical_link") or "").strip()
        if not url:
            continue
        out.append(
            SearchResult(
                title=str(it.get("title") or ""),
                url=url,
                snippet=str(it.get("snippet") or it.get("content") or ""),
                provider="serpapi",
            )
        )
        if len(out) >= int(max_results):
            break
    return out


class WebSearchInput(BaseModel):
    query: str = Field(description="搜索查询（必填）")
    max_results: int = Field(default=5, ge=1, le=10, description="返回条数（1~10）")


@tool("web_search", args_schema=WebSearchInput)
def web_search(query: str, max_results: int = 5) -> str:
    """在线搜索工具（MCP + SerpAPI）。
    当用户输入"在线搜索"、"网络搜索"、"网页搜索"等关键词时，系统会调用此工具。
    输入 query，输出 JSON 字符串：{"query": "...", "results": [{"title","url","snippet","provider"}]}。
    """
    q = (query or "").strip()
    max_results = int(max_results) if max_results is not None else 5

    api_key = (os.environ.get("SERPAPI_API_KEY") or "").strip()
    if not api_key:
        return json.dumps(
            {"query": q, "results": [], "error": "缺少 SERPAPI_API_KEY 环境变量。"},
            ensure_ascii=False,
        )

    try:
        results = _run(_mcp_serpapi_search(q, max_results=max_results, timeout_s=25.0), timeout_s=30.0)
        payload = {"query": q, "results": [r.model_dump() for r in (results or [])]}
        return json.dumps(payload, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"query": q, "results": [], "error": str(e)}, ensure_ascii=False)
