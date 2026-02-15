from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class MessageRecord:
    role: str
    content: str
    meta: Optional[Dict[str, Any]] = None

