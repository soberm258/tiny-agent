from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Dict, List, Literal, TypedDict, Tuple

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from tinyrag.langchain_tools import rag_search

from agent.prompts.deep_research import (
    DEEP_RESEARCH_PLANNER_PROMPT,
    DEEP_RESEARCH_VERIFIER_PROMPT,
    DEEP_RESEARCH_WRITER_PROMPT,
)
from agent.prompts.report_template import REPORT_TEMPLATE
from agent.tools.schemas import DeepResearchPlan, DeepResearchPlanItem, Evidence
from agent.tools.web_fetch import web_fetch
from agent.tools.web_search import web_search


class DeepResearchInput(BaseModel):
    task: str = Field(description="主Agent传入的任务描述（必填）")
    audience: str = Field(default="法律咨询", description="报告面向对象")


class _State(TypedDict, total=False):
    task: str  # 主任务（来自主 Agent，可能含输出要求/上下文/记忆注入）
    audience: str  # 报告面向对象（影响写作口吻与详略）
    plan: List[DeepResearchPlanItem]  # Planner 子问题清单（驱动 research 检索）  question:str, rag_query:str, web_query:str
    evidence: List[Evidence]  # 证据池（Research 产出；去重+坏编码过滤后）
    citation_map: Dict[int, int]  # 引用重编号映射：old_id -> new_id
    citation_old_ids: List[int]  # 参与重编号的 old_id（按正文首次出现顺序）
    sections_json: Dict[str, Any]  # Writer JSON sections（四段正文，不含引用列表）
    used_evidence_ids: List[int]  # Writer 声明使用的证据 id（会与正文引用合并）
    draft: str  # 最终 Markdown 报告（模板+连续编号+引用列表）
    passed: bool  # Verifier 是否通过
    issues: List[str]  # Verifier 问题列表（用于“质检提示”）
    round: int  # 回合计数（控制最多补充检索/重写次数）


def _build_model(streaming: bool = False) -> ChatOpenAI:
    model_id = os.environ.get("DEEP_RESEARCH_MODEL_ID") or os.environ.get("LLM_MODEL_ID", "Qwen/Qwen3-32B")
    base_url = os.environ.get("LLM_BASE_URL", "https://api-inference.modelscope.cn/v1")
    api_key = os.environ.get("LLM_API_KEY", "")
    return ChatOpenAI(
        model=model_id,
        base_url=base_url,
        api_key=api_key,
        streaming=streaming,
        extra_body={"enable_thinking": bool(streaming)},
    )


def _truncate(text: str, max_len: int) -> str:
    t = str(text or "")
    if len(t) <= max_len:
        return t
    return t[:max_len].rstrip() + "…"


def _extract_first_source(text: str) -> str:
    m = re.search(r"(source=[^\n\r]+)", str(text or ""))
    return m.group(1).replace("source=", "", 1).strip() if m else ""


def _parse_rag_observation_items(obs: str) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    cur_text_lines: List[str] = []
    cur_source = ""
    cur_rank = ""

    def flush() -> None:
        nonlocal cur_text_lines, cur_source, cur_rank
        text = "\n".join([x for x in cur_text_lines if str(x).strip()]).strip()
        src = str(cur_source or "").strip()
        if text:
            out: Dict[str, str] = {"text": text, "source": src}
            if cur_rank:
                out["rank"] = cur_rank
            items.append(out)
        cur_text_lines = []
        cur_source = ""
        cur_rank = ""

    for raw in str(obs or "").splitlines():
        line = raw.rstrip("\n").rstrip("\r")
        if not line.strip():
            continue
        if line.startswith("error="):
            continue

        m = re.match(r"^\[(\d+)\]\s*(.*)$", line)
        if m:
            flush()
            cur_rank = m.group(1)
            cur_text_lines.append(m.group(2))
            continue

        if line.startswith("source="):
            cur_source = line[len("source=") :].strip()
            flush()
            continue

        if cur_text_lines:
            cur_text_lines.append(line)

    flush()
    return items


def _extract_article_from_text(text: str) -> str:
    t = re.sub(r"\s+", "", str(text or ""))
    m = re.search(r"(第[一二三四五六七八九十百千万零〇0-9]+条(?:之[一二三四五六七八九十百千万零〇0-9]+)?)", t)
    return m.group(1) if m else ""


def _law_title_from_source_and_snippet(*, source: str, snippet: str) -> str:
    src = str(source or "")
    parts = [p.strip() for p in src.split("|")] if "|" in src else []
    doc = parts[1] if len(parts) >= 2 else ""
    src_article = parts[-1] if parts else ""
    snip_article = _extract_article_from_text(snippet)
    article = snip_article or _extract_article_from_text(src_article) or src_article
    doc = doc.strip()
    article = article.strip()
    if doc and article:
        return f"{doc} {article}"
    return article or doc or "rag_search(law)"


def _normalize_for_dedup(text: str) -> str:
    t = str(text or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[，。；；、,.!?！？:：()（）\\[\\]【】]", "", t)
    return t


def _is_bad_mojibake(text: str) -> bool:
    """判断文本中是否存在严重的乱码情况（如大量的 � 字符）"""
    t = str(text or "")
    if not t:
        return False
    bad = t.count("\ufffd")
    if bad >= 6:
        return True
    if len(t) >= 120 and (bad / max(len(t), 1)) >= 0.02:
        return True
    return False


def _dedupe_evidence(evidence: List[Evidence]) -> List[Evidence]:
    """基于 source_type + source + snippet 内容 + title 做去重"""
    seen: set[str] = set()
    out: List[Evidence] = []
    for ev in evidence:
        src = str(ev.source or "").strip()
        title = str(ev.title or "").strip()
        snip = _normalize_for_dedup(ev.snippet)[:800]
        key = f"{ev.source_type}|{src}|{hashlib.sha1(snip.encode('utf-8', errors='ignore')).hexdigest()}|{title[:80]}"
        if key in seen:
            continue
        seen.add(key)
        out.append(ev)
    return out


def _rag_items_to_evidence(*, obs: str, db: str, query: str, max_items: int = 5) -> List[Evidence]:
    parsed = _parse_rag_observation_items(obs)
    if not parsed:
        return [
            Evidence(
                source_type="rag",
                source=f"rag:{db}",
                title=f"rag_search({db})",
                snippet=_truncate(str(obs or ""), 5000),
                meta={"query": query},
            )
        ]

    out: List[Evidence] = []
    for it in parsed[: int(max_items)]:
        src = str(it.get("source") or "").strip() or f"rag:{db}"
        text = str(it.get("text") or "").strip()
        text = re.sub(r"^\[\d+\]\s*", "", text).strip()
        text = re.sub(r"source=[^\n\r]+", "", text).strip()
        rank = str(it.get("rank") or "").strip()

        title = ""
        if db == "law":
            title = _law_title_from_source_and_snippet(source=src, snippet=text)
        elif "|" in src:
            title = src.split("|")[-1].strip()
        if not title:
            title = f"rag_search({db})"
        ####
        #
        #
        # print(f"\ntext:\n{text}\n")

        out.append(
            Evidence(
                source_type="rag",
                source=src,
                title=title,
                snippet=_truncate(text, 10000),
                meta={"query": query, "rank": rank} if rank else {"query": query},
            )
        )
    return out


def _extract_citation_ids(text: str) -> List[int]:
    ids: List[int] = []
    for m in re.finditer(r"\[(\d+)\]", str(text or "")):
        try:
            ids.append(int(m.group(1)))
        except Exception:
            continue
    return ids


def _extract_citation_ids_in_order(text: str) -> List[int]:
    seen: set[int] = set()
    out: List[int] = []
    for m in re.finditer(r"\[(\d+)\]", str(text or "")):
        try:
            cid = int(m.group(1))
        except Exception:
            continue
        if cid in seen:
            continue
        seen.add(cid)
        out.append(cid)
    return out


def _renumber_sections_and_refs(
    *,
    body_sections: Dict[str, str],
    required_keys: List[str],
    evidence_by_old_id: Dict[int, Evidence],
    used_old_ids: List[int],
) -> Tuple[Dict[str, str], List[str], Dict[int, int], List[int]]:
    in_body_order: List[int] = []
    for k in required_keys:
        in_body_order.extend(_extract_citation_ids_in_order(body_sections.get(k, "")))
    in_body_order = [i for i in in_body_order if i in evidence_by_old_id]
    used_old_ids = [i for i in used_old_ids if i in evidence_by_old_id]
    all_old_ids = list(dict.fromkeys([*in_body_order, *used_old_ids]))
    citation_map: Dict[int, int] = {old: new for new, old in enumerate(all_old_ids, start=1)}

    def repl(match: re.Match) -> str:
        old = int(match.group(1))
        new = citation_map.get(old)
        if not new:
            return match.group(0)
        return f"[{new}]"

    new_body: Dict[str, str] = {}
    for k in required_keys:
        t = str(body_sections.get(k, "") or "")
        t = re.sub(r"\[(\d+)\]", repl, t)
        t = re.sub(r"\]\[", "] [", t)
        new_body[k] = t.strip()

    refs_lines: List[str] = []
    for old_id in all_old_ids:
        new_id = citation_map[old_id]
        ev = evidence_by_old_id[old_id]
        title = ev.title.strip() or "证据"
        snippet = re.sub(r"\s+", " ", str(ev.snippet or "").strip())
        if len(snippet) > 140:
            snippet = snippet[:140].rstrip() + "…"
        line = f"[{new_id}] {title} source={ev.source}".strip()
        if snippet:
            line = f"{line}\n证据摘录：{snippet}"
        refs_lines.append(line)

    return new_body, refs_lines, citation_map, all_old_ids


def _render_markdown_report(
    *, sections: Dict[str, str], evidence: List[Evidence], used_ids: List[int]
) -> Tuple[str, Dict[int, int], List[int]]:
    required_keys = [
        "一、任务与结论摘要",
        "二、背景与问题拆解",
        "三、证据与分析",
        "四、风险、不确定性与需要补充的信息",
    ]
    evidence_by_id = {i: ev for i, ev in enumerate(evidence, start=1)}
    used_ids = [int(x) for x in used_ids if isinstance(x, int) or str(x).isdigit()]
    used_ids = [i for i in used_ids if i in evidence_by_id]
    used_ids = list(dict.fromkeys(used_ids))

    body_sections: Dict[str, str] = {}
    for k in required_keys:
        raw = str(sections.get(k, "") or "").strip()
        body_sections[k] = raw

    cited_ids: List[int] = []
    for k in required_keys:
        cited_ids.extend(_extract_citation_ids(body_sections[k]))
    cited_ids = [i for i in dict.fromkeys(cited_ids) if i in evidence_by_id]
    used_ids = list(dict.fromkeys([*used_ids, *cited_ids]))

    body_sections, refs_lines, citation_map, citation_old_ids = _renumber_sections_and_refs(
        body_sections=body_sections,
        required_keys=required_keys,
        evidence_by_old_id=evidence_by_id,
        used_old_ids=used_ids,
    )

    md = (
        "# 法务备忘录\n\n"
        "## 一、任务与结论摘要\n"
        f"{body_sections['一、任务与结论摘要']}\n\n"
        "## 二、背景与问题拆解\n"
        f"{body_sections['二、背景与问题拆解']}\n\n"
        "## 三、证据与分析\n"
        f"{body_sections['三、证据与分析']}\n\n"
        "## 四、风险、不确定性与需要补充的信息\n"
        f"{body_sections['四、风险、不确定性与需要补充的信息']}\n\n"
        "## 五、引用列表\n"
        + ("\n\n".join(refs_lines).strip() or "（无）")
        + "\n"
    )
    return md, citation_map, citation_old_ids


def _parse_writer_json(text: str) -> Tuple[Dict[str, str], List[int]]:
    obj = json.loads(text)
    sections = obj.get("sections") or {}
    used = obj.get("used_evidence_ids") or []
    if not isinstance(sections, dict):
        raise ValueError("sections 必须是对象")
    if not isinstance(used, list):
        raise ValueError("used_evidence_ids 必须是数组")
    sections_str: Dict[str, str] = {str(k): str(v) for k, v in sections.items()}
    required_keys = {
        "一、任务与结论摘要",
        "二、背景与问题拆解",
        "三、证据与分析",
        "四、风险、不确定性与需要补充的信息",
    }
    if not required_keys.issubset(set(sections_str.keys())):
        raise ValueError("sections 缺少必需章节键")
    used_ids: List[int] = []
    for x in used:
        try:
            used_ids.append(int(x))
        except Exception:
            continue
    return sections_str, used_ids


def _parse_plan(text: str) -> List[DeepResearchPlanItem]:
    try:
        obj = json.loads(text)
        plan = DeepResearchPlan.model_validate(obj)
        return plan.sub_questions
    except Exception:
        items: List[DeepResearchPlanItem] = []
        for line in (text or "").splitlines():
            line = line.strip()
            if not line:
                continue
            items.append(DeepResearchPlanItem(question=line, rag_query=line, web_query=line))
        return items[:6]


def _extract_db_hint(task: str) -> List[Literal["law", "case"]]:
    t = (task or "").lower()
    m = re.search(r"(db|数据库)\s*[:=]\s*([a-z,]+)", t)
    hint = m.group(2) if m else ""
    if "case" in hint:
        return ["law", "case"] if "law" in hint else ["case"]
    return ["law"]


def _plan_node(state: _State) -> _State:
    model = _build_model(streaming=False)
    task = state.get("task", "")
    resp = model.invoke(
        [
            {"role": "system", "content": DEEP_RESEARCH_PLANNER_PROMPT},
            {"role": "user", "content": task},
        ]
    )
    # 理想情况下llm输出： resp['content'] = {"sub_questions":[{"question": "...", "rag_query": "...", "web_query": "..."}...]}
    # 但是如果大模型不听话也需要考虑处理 _parse_plan
    plan_items = _parse_plan(getattr(resp, "content", "") or "")[:4]
    return {"plan": plan_items, "round": int(state.get("round") or 0)}


def _research_node(state: _State) -> _State:
    task = state.get("task", "")
    dbs = _extract_db_hint(task)
    plan_items = state.get("plan") or []
    evidence: List[Evidence] = list(state.get("evidence") or [])
    for item in plan_items[:4]:
        q_rag = (item.rag_query or item.question or "").strip()
        q_web = (item.web_query or item.question or "").strip()

        for db in dbs:
            rag_out = rag_search.invoke({"query": q_rag, "topk": 5, "db_name": db, "is_hyde": False})
            evidence.extend(_rag_items_to_evidence(obs=str(rag_out), db=str(db), query=q_rag, max_items=5))

        web_out = web_search.invoke({"query": q_web, "max_results": 5})
        try:
            web_obj = json.loads(web_out)
            results = web_obj.get("results") or []
        except Exception:
            results = []
        for r in results[:2]:
            url = str(r.get("url") or "").strip()
            if not url:
                continue
            ev = Evidence(
                source_type="web",
                source=url,
                title=str(r.get("title") or ""),
                snippet=_truncate(str(r.get("snippet") or ""), 5000),
                meta={"query": q_web, "provider": str(r.get("provider") or "")},
            )
            evidence.append(ev)
            fetched = web_fetch.invoke({"url": url, "max_chars": 10000})
            try:
                fobj = json.loads(fetched)
                if fobj.get("error"):
                    excerpt = f"抓取失败：{str(fobj.get('error') or '')}".strip()
                else:
                    excerpt = str(fobj.get("excerpt") or "")
                title = str(fobj.get("title") or "")
            except Exception:
                title = ""
                excerpt = str(fetched)
            if excerpt:
                ev.title = title or ev.title
                # ev.snippet = _truncate(excerpt, 3000)
                ev.snippet = _truncate(excerpt, 5000)
                if _is_bad_mojibake(ev.title) or _is_bad_mojibake(ev.snippet):
                    ev.meta["bad_encoding"] = True

    evidence = _dedupe_evidence(evidence)
    evidence = [e for e in evidence if not bool((e.meta or {}).get("bad_encoding"))]
    return {"evidence": evidence}


def _write_node(state: _State) -> _State:
    model = _build_model(streaming=False)
    task = state.get("task", "")
    audience = state.get("audience", "法律咨询")
    evidence = state.get("evidence") or []
    evidence_payload = [
        {
            "id": i,
            "source_type": ev.source_type,
            "source": ev.source,
            "title": ev.title,
            "snippet": _truncate(ev.snippet,5000),
            "meta": ev.meta,
        }
        for i, ev in enumerate(evidence, start=1)
    ]
    user_payload = (
        f"【面向对象】{audience}\n\n"
        f"【任务】\n{task}\n\n"
        f"【报告模板（必须严格使用标题，不得改写）】\n{REPORT_TEMPLATE}\n\n"
        f"【证据池（JSON，id 为引用编号）】\n{json.dumps(evidence_payload, ensure_ascii=False)}\n"
    )
    def _issues_to_str(issues: List[str]) -> str:
        return "\n".join([f"{i+1}. {str(iss).strip()}" for i, iss in enumerate(issues)])
    def call_writer(extra: str = "") -> str:
        resp = model.invoke(
            [
                {"role": "system", "content": DEEP_RESEARCH_WRITER_PROMPT},
                {"role": "user", "content": user_payload +
                 (("\n\n刚刚生成的结果有如下问题，必须解决:\n"+_issues_to_str(state.get("issues"))) if state.get("issues") else "")
                 +(("\n\n" + extra) if extra else "")},
            ]
        )
        return getattr(resp, "content", "") or ""

    raw = call_writer()
    try:
        sections, used_ids = _parse_writer_json(raw)
    except Exception:
        raw = call_writer("你刚才输出无法解析为严格 JSON。请只输出严格 JSON，不要附加任何说明。")
        sections, used_ids = _parse_writer_json(raw)

    body_keys = (
        "一、任务与结论摘要",
        "二、背景与问题拆解",
        "三、证据与分析",
        "四、风险、不确定性与需要补充的信息",
    )
    if any("source=" in str(sections.get(k, "") or "") for k in body_keys):
        raw = call_writer("你的正文里出现了 source=...，这是禁止的。请删除正文中的 source=...，正文只保留 [n] 引用编号，并输出严格 JSON。")
        sections, used_ids = _parse_writer_json(raw)

    allowed = set(range(1, len(evidence) + 1))
    cited = []
    for k in (
        "一、任务与结论摘要",
        "二、背景与问题拆解",
        "三、证据与分析",
        "四、风险、不确定性与需要补充的信息",
    ):
        cited.extend(_extract_citation_ids(sections.get(k, "")))
    cited = list(dict.fromkeys(cited))
    invalid_cited = [i for i in cited if i not in allowed]
    invalid_used = [i for i in used_ids if i not in allowed]
    if invalid_cited or invalid_used:
        raw = call_writer(
            "你的 JSON 中出现了无效引用编号。请只使用证据池中存在的编号（范围 1~"
            f"{len(evidence)}），并重新输出严格 JSON。"
        )
        sections, used_ids = _parse_writer_json(raw)

    draft_md, citation_map, citation_old_ids = _render_markdown_report(sections=sections, evidence=evidence, used_ids=used_ids)
    return {
        "sections_json": sections,
        "used_evidence_ids": used_ids,
        "draft": draft_md,
        "citation_map": citation_map,
        "citation_old_ids": citation_old_ids,
    }


def _verify_node(state: _State) -> _State:
    model = _build_model(streaming=False)
    draft = state.get("draft", "")
    evidence = state.get("evidence") or []
    citation_old_ids = [int(x) for x in (state.get("citation_old_ids") or []) if isinstance(x, int) or str(x).isdigit()]
    evidence_by_old_id = {i: ev for i, ev in enumerate(evidence, start=1)}
    if citation_old_ids:
        cited_evs = [evidence_by_old_id[i] for i in citation_old_ids if i in evidence_by_old_id]
    else:
        cited_evs = evidence
    cites = "\n".join(e.full_text(i) for i, e in enumerate(cited_evs, start=1))
    payload = f"【报告草稿】\n{draft}\n\n【证据池（供对照）】\n{cites}\n"
    resp = model.invoke(
        [
            {"role": "system", "content": DEEP_RESEARCH_VERIFIER_PROMPT},
            {"role": "user", "content": payload},
        ]
    )
    text = getattr(resp, "content", "") or ""
    try:
        obj = json.loads(text)
        passed = bool(obj.get("passed"))
        issues = obj.get("issues") or []
        needed = obj.get("needed_queries") or []
        issues = issues if isinstance(issues, list) else [str(issues)]
        needed = needed if isinstance(needed, list) else [str(needed)]
    except Exception:
        passed = False
        issues = ["质检输出无法解析为JSON，建议补充证据并重写。"]
        needed = []
    return {
        "passed": passed,
        "issues": [str(x) for x in issues],
        "needed_queries": [str(x) for x in needed],
    }


def _needs_more(state: _State) -> str:
    if state.get("passed"):
        return "end"
    if int(state.get("round") or 0) >= 1:
        return "end"
    return "more"


def _research_more_node(state: _State) -> _State:
    needed = state.get("needed_queries") or []
    plan_items = [DeepResearchPlanItem(question=q, rag_query=q, web_query=q) for q in needed[:3] if str(q).strip()]
    return {"plan": plan_items, "round": int(state.get("round") or 0) + 1}


def _build_graph():
    g = StateGraph(_State)
    g.add_node("plan", _plan_node)
    g.add_node("research", _research_node)
    g.add_node("write", _write_node)
    g.add_node("verify", _verify_node)
    g.add_node("research_more", _research_more_node)
    g.set_entry_point("plan")
    g.add_edge("plan", "research")
    g.add_edge("research", "write")
    g.add_edge("write", "verify")
    g.add_conditional_edges("verify", _needs_more, {"more": "research_more", "end": END})
    g.add_edge("research_more", "research")
    return g.compile()


@tool("deep_research", args_schema=DeepResearchInput)
def deep_research(task: str, audience: str = "法律咨询") -> str:
    """多 Agent 深度调研与报告生成工具。

    主 Agent 只负责聊天与少量检索；当用户明确要求“详细报告/研究报告/调研/完整分析”时，
    必须调用本工具。本工具内部采用 Planner/Research/Writer/Verifier 的编排生成法务备忘录，
    并要求结论与引用证据对齐（正文仅编号 [n]，source=... 统一放在“## 五、引用列表”）。
    """
    graph = _build_graph()
    state: _State = {"task": (task or "").strip(), "audience": (audience or "法律咨询").strip(), "round": 0}
    out: _State = graph.invoke(state)
    draft = out.get("draft") or ""
    passed = bool(out.get("passed"))
    issues = out.get("issues") or []

    if passed:
        return str(draft).strip()

    issues_lines = [str(x).strip() for x in issues if str(x).strip()]
    issues_text = "\n".join(f"问题{idx}：{line}" for idx, line in enumerate(issues_lines, start=1))
    return f"{draft}\n\n---\n\n【质检提示】\n{issues_text}".strip()
