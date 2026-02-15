from __future__ import annotations

import json
import ssl
import urllib.request
from typing import Any, Dict, Tuple

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from agent.tools.html_extract import extract_text_from_html


def _decode_bytes(raw: bytes, charset: str | None) -> str:
    if not raw:
        return ""
    if charset:
        try:
            return raw.decode(charset, errors="replace")
        except Exception:
            pass
    try:
        text = raw.decode("utf-8", errors="replace")
        if text.count("\ufffd") <= 2:
            return text
    except Exception:
        pass
    for enc in ("gb18030", "gbk", "big5"):
        try:
            text = raw.decode(enc, errors="replace")
            if text.count("\ufffd") <= 2:
                return text
        except Exception:
            continue
    return raw.decode("utf-8", errors="replace")


def _fetch_url(url: str, timeout_s: int = 10) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "tiny-agents/1.0"})
    ctx = ssl.create_default_context()
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    with urllib.request.urlopen(req, timeout=timeout_s, context=ctx) as resp:
        raw = resp.read()
        charset = None
        try:
            charset = resp.headers.get_content_charset()
        except Exception:
            charset = None
    return _decode_bytes(raw, charset)


class WebFetchInput(BaseModel):
    url: str = Field(description="网页 URL（必填）")
    max_chars: int = Field(default=12000, ge=500, le=50000, description="正文截断长度（500~50000）")


@tool("web_fetch", args_schema=WebFetchInput)
def web_fetch(url: str, max_chars: int = 12000) -> str:
    """抓取网页并提取正文片段。

    输入为 URL，输出为 JSON 字符串，包含 url/title/excerpt/source 等字段。
    用于在在线检索后把网页内容拉取为可引用证据，避免直接基于搜索摘要下结论。
    """
    u = (url or "").strip()
    try:
        html = _fetch_url(u)
        title, text = extract_text_from_html(html)
        excerpt = text[: int(max_chars)].rstrip()
        payload: Dict[str, Any] = {"url": u, "title": title, "excerpt": excerpt, "source": u}
    except Exception as e:
        payload = {"url": u, "title": "", "excerpt": "", "source": u, "error": str(e)}
    return json.dumps(payload, ensure_ascii=False)
