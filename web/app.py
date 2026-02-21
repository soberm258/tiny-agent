from __future__ import annotations

import os
import re
import uuid
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, TypedDict

import gradio as gr

from agent.memory import MemoryStore

from script.agent_turn import EvidenceItem, TurnEvent, build_runtime, stream_turn


_CSS_PATH = Path(__file__).parent / "assets" / "style.css"


def _load_css() -> str:
    if _CSS_PATH.exists():
        return _CSS_PATH.read_text(encoding="utf-8")
    return ""


def _new_thread_id() -> str:
    return uuid.uuid4().hex[:16]


def _compact_text(s: str, *, limit: int = 20) -> str:
    t = re.sub(r"\s+", " ", (s or "").strip())
    if not t:
        return ""
    return t[:limit]


class _ChatMsg(TypedDict):
    role: str
    content: str


def _messages_to_chatbot_messages(messages: List[Dict[str, str]]) -> List[_ChatMsg]:
    out: List[_ChatMsg] = []
    for m in messages or []:
        role = str(m.get("role") or "").strip()
        content = str(m.get("content") or "")
        if role not in ("user", "assistant", "system"):
            continue
        if role == "system":
            continue
        out.append({"role": role, "content": content})
    return out


def _render_evidence_md(items: List[EvidenceItem]) -> str:
    if not items:
        return "（证据池为空。）"
    lines: List[str] = []
    lines.append(f"### 证据池（{len(items)}）")
    for i, it in enumerate(items, start=1):
        title = _compact_text(it.title, limit=40) or it.tool
        src = _compact_text(it.source, limit=120)
        snippet = (it.snippet or "").strip()
        snippet = snippet if len(snippet) <= 200 else snippet[:200].rstrip() + "……"
        lines.append(f"{i}. ")
        if src:
            lines.append(f"来源：{src}")
        if snippet:
            lines.append(f"摘要：{snippet}\n")
    return "\n".join(lines).strip()


def _evidence_key(it: EvidenceItem) -> str:
    raw = f"{it.kind}\n{it.tool}\n{it.source}\n{it.title}\n{it.snippet}".encode("utf-8", errors="ignore")
    return hashlib.sha1(raw).hexdigest()


def _merge_evidence_pool(pool: List[EvidenceItem], incoming: List[EvidenceItem]) -> List[EvidenceItem]:
    cur = list(pool or [])
    seen = {_evidence_key(x) for x in cur}
    for it in incoming or []:
        k = _evidence_key(it)
        if k in seen:
            continue
        seen.add(k)
        cur.append(it)
    return cur


@dataclass
class UIState:
    thread_id: str
    evidence_pool: List[EvidenceItem]


def _init_ui_state() -> UIState:
    return UIState(thread_id=_new_thread_id(), evidence_pool=[])


def _list_thread_choices(store: MemoryStore, *, limit: int = 50) -> List[Tuple[str, str]]:
    threads = store.list_threads(limit=limit)
    choices: List[Tuple[str, str]] = []
    for th in threads:
        tid = str(th.get("thread_id") or "").strip()
        title = str(th.get("title") or "").strip()
        if not tid:
            continue
        label = f"{title}  ·  {tid}" if title else tid
        choices.append((label, tid))
    return choices


def _load_thread(store: MemoryStore, thread_id: str) -> Tuple[List[_ChatMsg], str]:
    history = store.get_recent_messages(thread_id, limit=50)
    summary, _facts = store.get_summary(thread_id)
    chat_pairs = _messages_to_chatbot_messages(history)
    summary_md = (summary or "").strip() or "（暂无长期记忆摘要）"
    return chat_pairs, summary_md


def main() -> None:
    runtime = build_runtime()
    store = runtime.store
    css = _load_css()

    with gr.Blocks(title="Tiny Agents · 法务助理", fill_width=True) as demo:
        ui_state = gr.State(_init_ui_state())

        gr.HTML(
            """
            <div id="topbar">
              <div class="topbar-left">
                <div class="topbar-title">Tiny Agents · 法务助理</div>
                <div class="topbar-sub">这是一个基于本地 RAG 和在线证据的智能法务助理。</div>
              </div>
              <div class="topbar-right">需要详细报告时直接说“详细报告/研究报告”。</div>
            </div>
            """
        )

        with gr.Row(equal_height=True, elem_id="main_row"):
            with gr.Column(scale=28, min_width=320, elem_id="left_col"):
                with gr.Group(elem_id="left_panel"):
                    gr.Markdown("### 历史对话")
                    thread_dropdown = gr.Dropdown(
                        choices=_list_thread_choices(store),
                        value=None,
                        label="",
                        interactive=True,
                        elem_id="thread_dropdown",
                    )
                    new_thread_btn = gr.Button("新建对话", variant="primary", elem_id="new_thread_btn")
                    gr.Markdown("### 长期记忆摘要")
                    summary_box = gr.Markdown("（暂无长期记忆摘要）", elem_id="memory_panel")

            with gr.Column(scale=72, min_width=560, elem_id="right_col"):
                with gr.Group(elem_id="right_panel"):
                    gr.Markdown("### 对话")
                    chatbot = gr.Chatbot(
                        label="",
                        height=520,
                        buttons=["copy", "copy_all"],
                        elem_id="chat_panel",
                    )
                    with gr.Row(elem_id="composer_row"):
                        user_text = gr.Textbox(
                            label="",
                            placeholder="输入问题…",
                            lines=2,
                            elem_id="composer_text",
                        )
                        send_btn = gr.Button("发送", variant="secondary", elem_id="composer_send")

        with gr.Row(equal_height=True, elem_id="evidence_row"):
            with gr.Column(scale=1, min_width=920):
                with gr.Group(elem_id="evidence_panel_wrap"):
                    gr.Markdown("### 证据池")
                    evidence_md = gr.Markdown("（证据池为空。开始对话后会自动累积检索证据。）", elem_id="evidence_panel")

        def _on_new_thread(state: UIState) -> Tuple[UIState, List[_ChatMsg], str, Any]:
            new_state = UIState(thread_id=_new_thread_id(), evidence_pool=[])
            choices = _list_thread_choices(store)
            return new_state, [], "（暂无长期记忆摘要）", gr.update(choices=choices, value=None)

        def _on_switch_thread(thread_id: str, state: UIState) -> Tuple[UIState, List[_ChatMsg], str]:
            if not thread_id:
                return state, [], "（暂无长期记忆摘要）"
            new_state = UIState(thread_id=str(thread_id), evidence_pool=[])
            pairs, summary_md = _load_thread(store, new_state.thread_id)
            return new_state, pairs, summary_md

        def _on_send(
            text: str,
            state: UIState,
            chat_pairs: List[_ChatMsg],
            summary_md_in: str,
        ) -> Iterable[Tuple[List[_ChatMsg], UIState, str, Any, str, str]]:
            user_in = (text or "").strip()
            cur_state = state
            thread_dropdown_initial = gr.update()
            if not user_in:
                yield (
                    chat_pairs,
                    cur_state,
                    _render_evidence_md(cur_state.evidence_pool),
                    thread_dropdown_initial,
                    summary_md_in,
                    "",
                )
                return

            local_pairs = list(chat_pairs or [])
            local_pairs.append({"role": "user", "content": user_in})
            local_pairs.append({"role": "assistant", "content": ""})
            yield (
                local_pairs,
                cur_state,
                _render_evidence_md(cur_state.evidence_pool),
                thread_dropdown_initial,
                summary_md_in,
                "",
            )

            assistant_buf: List[str] = []
            evidence_pool: List[EvidenceItem] = list(cur_state.evidence_pool or [])

            for ev in stream_turn(runtime, thread_id=cur_state.thread_id, user_input=user_in):
                if ev.type == "token" and ev.text:
                    assistant_buf.append(ev.text)
                    local_pairs[-1] = {"role": "assistant", "content": "".join(assistant_buf)}
                    yield (
                        local_pairs,
                        cur_state,
                        _render_evidence_md(evidence_pool),
                        thread_dropdown_initial,
                        summary_md_in,
                        "",
                    )
                elif ev.type == "evidence":
                    evidence_pool = _merge_evidence_pool(evidence_pool, list(ev.evidence or []))
                    new_state = UIState(
                        thread_id=cur_state.thread_id,
                        evidence_pool=evidence_pool,
                    )
                    cur_state = new_state
                    yield (
                        local_pairs,
                        cur_state,
                        _render_evidence_md(evidence_pool),
                        thread_dropdown_initial,
                        summary_md_in,
                        "",
                    )
                elif ev.type == "final":
                    local_pairs[-1] = {"role": "assistant", "content": ev.assistant_text}
                    new_state = UIState(
                        thread_id=cur_state.thread_id,
                        evidence_pool=evidence_pool,
                    )
                    cur_state = new_state
                    pairs, summary_md = _load_thread(store, new_state.thread_id)
                    thread_dropdown_final = gr.update(choices=_list_thread_choices(store))
                    yield (
                        pairs,
                        cur_state,
                        _render_evidence_md(evidence_pool),
                        thread_dropdown_final,
                        summary_md,
                        "",
                    )

        new_thread_btn.click(
            _on_new_thread,
            inputs=[ui_state],
            outputs=[ui_state, chatbot, summary_box, thread_dropdown],
        )
        thread_dropdown.change(
            _on_switch_thread,
            inputs=[thread_dropdown, ui_state],
            outputs=[ui_state, chatbot, summary_box],
        )
        send_btn.click(
            _on_send,
            inputs=[user_text, ui_state, chatbot, summary_box],
            outputs=[chatbot, ui_state, evidence_md, thread_dropdown, summary_box, user_text],
        )
        user_text.submit(
            _on_send,
            inputs=[user_text, ui_state, chatbot, summary_box],
            outputs=[chatbot, ui_state, evidence_md, thread_dropdown, summary_box, user_text],
        )

    port = int(os.environ.get("WEB_PORT") or 7860)
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=port,
        width="100%",
        css=css,
        js="""
        (() => {
          const kick = () => {
            try { window.dispatchEvent(new Event("resize")); } catch (e) {}
          };
          const burst = () => {
            kick();
            requestAnimationFrame(kick);
            setTimeout(kick, 120);
          };
          setTimeout(burst, 60);
          setTimeout(burst, 240);
          setTimeout(burst, 900);
        })();
        """,
    )


if __name__ == "__main__":
    main()
