from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Literal, Optional

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessageChunk, ToolMessage, ToolMessageChunk
from langchain_openai import ChatOpenAI

from tinyrag.langchain_tools import rag_search

from agent.memory import MemoryStore
from agent.prompts.main_agent import build_main_agent_system_prompt
from agent.routing import should_use_deep_research
from agent.tools.deep_research import deep_research
from agent.tools.web_fetch import web_fetch
from agent.tools.web_search import web_search

from script.evidence import EvidenceItem, parse_rag_observation


def _to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for blk in content:
            if isinstance(blk, str):
                parts.append(blk)
            elif isinstance(blk, dict):
                if "text" in blk:
                    parts.append(str(blk.get("text") or ""))
                elif "content" in blk:
                    parts.append(str(blk.get("content") or ""))
                else:
                    parts.append(str(blk))
            else:
                parts.append(str(blk))
        return "".join(parts)
    return str(content)


def _is_tool_call_message_chunk(chunk: AIMessageChunk) -> bool:
    tool_calls = getattr(chunk, "tool_calls", None)
    if tool_calls:
        return True
    extra = getattr(chunk, "additional_kwargs", None) or {}
    if isinstance(extra, dict) and extra.get("tool_calls"):
        return True
    return False


@dataclass(frozen=True)
class TurnEvent:
    type: Literal["token", "evidence", "final"]
    text: str = ""
    assistant_text: str = ""
    routed_to: Literal["chat", "deep_research"] = "chat"
    evidence: List[EvidenceItem] | None = None


@dataclass
class Runtime:
    agent: Any
    store: MemoryStore
    system_prompt: str


def build_runtime() -> Runtime:
    load_dotenv()
    model_id = os.environ.get("LLM_MODEL_ID", "Qwen/Qwen3-32B")
    base_url = os.environ.get("LLM_BASE_URL", "https://api-inference.modelscope.cn/v1")
    api_key = os.environ.get("LLM_API_KEY", "")
    model = ChatOpenAI(
        model=model_id,
        base_url=base_url,
        api_key=api_key,
        streaming=True,
        extra_body={"enable_thinking": False},
    )
    tools = [
        rag_search,
        web_search,
        web_fetch,
        deep_research,
    ]
    system_prompt = build_main_agent_system_prompt()
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
    )
    store = MemoryStore()
    return Runtime(agent=agent, store=store, system_prompt=system_prompt)


def _try_json(text: str) -> Any:
    s = (text or "").strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def _evidence_from_tool(tool_name: str, tool_output_text: str) -> List[EvidenceItem]:
    name = (tool_name or "").strip()
    raw = (tool_output_text or "").strip()
    if not raw:
        return []

    if name == "rag_search":
        return parse_rag_observation(raw)

    obj = _try_json(raw)
    if isinstance(obj, dict):
        if name == "web_fetch":
            url = str(obj.get("url") or obj.get("source") or "").strip()
            title = str(obj.get("title") or "web_fetch").strip()
            text = str(obj.get("text") or obj.get("excerpt") or "").strip()
            snippet = text if len(text) <= 600 else text[:600].rstrip() + "……"
            return [EvidenceItem(tool=name, title=title, snippet=snippet, source=url, kind="web")]
        if name == "web_search":
            results = obj.get("results") or obj.get("items") or []
            items: List[EvidenceItem] = []
            if isinstance(results, list):
                for i, it in enumerate(results[:5], start=1):
                    if not isinstance(it, dict):
                        continue
                    title = str(it.get("title") or f"web_search[{i}]").strip()
                    url = str(it.get("url") or it.get("link") or "").strip()
                    snippet = str(it.get("snippet") or it.get("summary") or "").strip()
                    snippet = snippet if len(snippet) <= 300 else snippet[:300].rstrip() + "……"
                    items.append(EvidenceItem(tool=name, title=title, snippet=snippet, source=url, kind="web"))
            return items

    snippet = raw if len(raw) <= 600 else raw[:600].rstrip() + "……"
    return [EvidenceItem(tool=name or "tool", title=name or "tool", snippet=snippet, source="", kind="other")]


def _evidence_key(it: EvidenceItem) -> str:
    raw = (
        f"{it.kind}\n{it.tool}\n{it.source}".encode("utf-8", errors="ignore")
    )
    return hashlib.sha1(raw).hexdigest()


def stream_turn(runtime: Runtime, *, thread_id: str, user_input: str) -> Iterator[TurnEvent]:
    store = runtime.store
    system_prompt = runtime.system_prompt
    agent = runtime.agent

    user_input = (user_input or "").strip()
    if not user_input:
        yield TurnEvent(type="final", assistant_text="", routed_to="chat", evidence=[])
        return

    prev_summary, prev_facts = store.get_summary(thread_id)
    history = store.get_recent_messages(thread_id, limit=20)

    if should_use_deep_research(user_input):
        task = (
            "【用户请求】\n"
            f"{user_input}\n\n"
            "【长期记忆摘要】\n"
            f"{prev_summary}\n\n"
            "【关键事实JSON】\n"
            f"{json.dumps(prev_facts or {}, ensure_ascii=False)}\n\n"
            "【输出要求】\n"
            "生成一份详细完备的法务备忘录式报告，结论必须引用证据。正文只用 [n] 编号引用，source=... 统一放在“## 五、引用列表”。\n"
            "【可用工具】\n"
            "- 本地：rag_search(db_name=law/case)\n"
            "- 在线：web_search + web_fetch\n"
        )
        report = deep_research.invoke({"task": task, "audience": "法律咨询"})
        report_text = str(report or "")
        store.append_message(thread_id, "user", user_input)
        store.append_message(thread_id, "assistant", report_text)
        store.update_summary_with_llm(
            thread_id=thread_id,
            prev_summary=prev_summary,
            prev_facts=prev_facts,
            user_text=user_input,
            assistant_text=report_text,
        )
        yield TurnEvent(type="final", assistant_text=report_text, routed_to="deep_research", evidence=[])
        return

    memory_injection = (
        "长期记忆摘要："
        + (prev_summary or "（空）")
        + "\n关键事实（JSON）："
        + json.dumps(prev_facts or {}, ensure_ascii=False)
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": memory_injection},
        *history,
        {"role": "user", "content": user_input},
    ]

    assistant_parts: List[str] = []
    evidence: List[EvidenceItem] = []
    evidence_keys: set[str] = set()
    tool_buf: Dict[str, Dict[str, Any]] = {}

    for chunk, meta in agent.stream({"messages": messages}, stream_mode="messages"):
        if isinstance(chunk, ToolMessage):
            tool_name = str(getattr(chunk, "name", "") or "").strip()
            if not tool_name:
                tool_name = str((meta or {}).get("name") or (meta or {}).get("tool_name") or "").strip()
            tool_output = _to_text(getattr(chunk, "content", ""))
            before = len(evidence)
            for it in _evidence_from_tool(tool_name, tool_output):
                k = _evidence_key(it)
                if k in evidence_keys:
                    continue
                evidence_keys.add(k)
                evidence.append(it)
            if len(evidence) != before:
                yield TurnEvent(type="evidence", routed_to="chat", evidence=list(evidence))
            continue

        if isinstance(chunk, ToolMessageChunk):
            tool_name = str(getattr(chunk, "name", "") or "").strip()
            if not tool_name:
                tool_name = str((meta or {}).get("name") or (meta or {}).get("tool_name") or "").strip()
            tool_call_id = str(getattr(chunk, "tool_call_id", "") or "").strip() or str(getattr(chunk, "id", "") or "").strip()
            if not tool_call_id:
                tool_call_id = f"tool_chunk_{len(tool_buf) + 1}"
            buf = tool_buf.get(tool_call_id)
            if buf is None:
                buf = {"name": tool_name, "parts": []}
                tool_buf[tool_call_id] = buf
            if tool_name and not str(buf.get("name") or "").strip():
                buf["name"] = tool_name
            buf["parts"].append(_to_text(getattr(chunk, "content", "")))
            continue

        if isinstance(chunk, AIMessageChunk) and not _is_tool_call_message_chunk(chunk):
            text = _to_text(getattr(chunk, "content", ""))
            if text:
                assistant_parts.append(text)
                yield TurnEvent(type="token", text=text)

    for buf in tool_buf.values():
        tool_name = str(buf.get("name") or "").strip()
        parts = buf.get("parts") or []
        if not isinstance(parts, list):
            parts = [str(parts)]
        tool_output = "".join([_to_text(x) for x in parts]).strip()
        if tool_name and tool_output:
            before = len(evidence)
            for it in _evidence_from_tool(tool_name, tool_output):
                k = _evidence_key(it)
                if k in evidence_keys:
                    continue
                evidence_keys.add(k)
                evidence.append(it)
            if len(evidence) != before:
                yield TurnEvent(type="evidence", routed_to="chat", evidence=list(evidence))

    assistant_text = "".join(assistant_parts).strip()
    store.append_message(thread_id, "user", user_input)
    store.append_message(thread_id, "assistant", assistant_text)
    store.update_summary_with_llm(
        thread_id=thread_id,
        prev_summary=prev_summary,
        prev_facts=prev_facts,
        user_text=user_input,
        assistant_text=assistant_text,
    )
    yield TurnEvent(type="final", assistant_text=assistant_text, routed_to="chat", evidence=evidence)


def __main__() -> None:
    runtime = build_runtime()
    thread_id = os.environ.get("AGENT_THREAD_ID", "1")
    text = os.environ.get("AGENT_TEXT", "").strip()
    if not text:
        print("请通过环境变量 AGENT_TEXT 传入本轮用户输入。")
        return
    for ev in stream_turn(runtime, thread_id=thread_id, user_input=text):
        if ev.type == "token":
            print(ev.text, end="", flush=True)
        elif ev.type == "final":
            if not ev.assistant_text:
                print("", flush=True)
            else:
                print("", flush=True)
            if ev.evidence:
                print("\n【本轮证据】")
                for i, it in enumerate(ev.evidence, start=1):
                    print(f"{i}. {it.title}")
                    if it.source:
                        print(f"   source={it.source}")
            break


if __name__ == "__main__":
    __main__()
