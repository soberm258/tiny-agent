from __future__ import annotations

import re
from typing import Tuple

from bs4 import BeautifulSoup


def extract_text_from_html(html: str) -> Tuple[str, str]:
    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()
    title = ""
    if soup.title and soup.title.string:
        title = str(soup.title.string).strip()
    text = " ".join(s.strip() for s in soup.get_text("\n").splitlines() if s.strip())
    text = re.sub(r"\s+", " ", text).strip()
    return title, text

