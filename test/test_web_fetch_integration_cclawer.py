from __future__ import annotations

import argparse
import os
import re
import ssl
import urllib.request
from pathlib import Path


URL = "https://www.cclawer.com/index/article/903.html"


def _fetch_html(url: str, *, timeout_s: int = 20) -> str:
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
    if charset:
        try:
            return raw.decode(charset, errors="replace")
        except Exception:
            pass
    for enc in ("utf-8", "gb18030", "gbk", "big5"):
        try:
            text = raw.decode(enc, errors="replace")
            if text.count("\ufffd") <= 2:
                return text
        except Exception:
            continue
    return raw.decode("utf-8", errors="replace")


def _extract_main_text(html: str) -> str:
    try:
        import trafilatura  # type: ignore

        text = trafilatura.extract(
            html,
            output_format="txt",
            include_comments=False,
            include_tables=False,
            favor_precision=True,
        )
        return str(text or "").strip()
    except Exception:
        pass

    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(html or "", "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
            tag.decompose()
        text = "\n".join(s.strip() for s in soup.get_text("\n").splitlines() if s.strip())
        return text.strip()
    except Exception:
        return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=URL)
    ap.add_argument("--timeout", type=int, default=20)
    ap.add_argument("--out", default="data/tmp/cclawer_903_extracted.txt")
    ap.add_argument("--preview-chars", type=int, default=1200)
    args = ap.parse_args()

    url = str(args.url or "").strip()
    if not url:
        print("URL 不能为空。")
        return 2

    html = _fetch_html(url, timeout_s=int(args.timeout))
    text = _extract_main_text(html)
    text = re.sub(r"\s+", " ", text).strip()

    out_path = Path(str(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")

    print(f"URL: {url}")
    print(f"抽取文本长度: {len(text)}")
    print("预览：")
    print(text[: int(args.preview_chars)])
    print("")
    print(f"已写入文件（UTF-8）：{out_path.as_posix()}")

    if not text:
        print("抽取结果为空：如果你想要更高质量的正文抽取，建议安装 trafilatura。")
        print("安装命令：pip install trafilatura")
        return 1

    if "酒驾" not in text and "醉驾" not in text:
        print("提示：抽取内容未命中关键词“酒驾/醉驾”，可能抓到了非正文或页面结构发生变化。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
