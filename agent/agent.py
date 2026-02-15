import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import AIMessageChunk, ToolMessageChunk


from tinyrag.langchain_tools import rag_search

from agent.memory import MemoryStore
from agent.prompts.main_agent import build_main_agent_system_prompt
from agent.routing import should_use_deep_research
from agent.tools.deep_research import deep_research
from agent.tools.web_fetch import web_fetch
from agent.tools.web_search import web_search


def _to_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
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


def __main__():
    load_dotenv()
    model_id = os.environ.get("LLM_MODEL_ID", "Qwen/Qwen3-32B")
    base_url = os.environ.get("LLM_BASE_URL", "https://api-inference.modelscope.cn/v1")
    api_key = os.environ.get("LLM_API_KEY", "")
    model = ChatOpenAI(
        model=model_id,
        base_url=base_url,
        api_key=api_key,
        streaming=True,
        extra_body={"enable_thinking": True},
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
    thread_id = os.environ.get("AGENT_THREAD_ID", "1")
    store = MemoryStore()
    while True:
        user_input = input("用户: ")
        if user_input.lower() in ["exit", "quit"]:
            print("退出对话。")
            break

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
                "- 本地：rag_search(db : law,case)\n"
                "- 在线：web_search + web_fetch\n"
            )
            report = deep_research.invoke({"task": task, "audience": "法律咨询"})
            print("助手: ", end="", flush=True)
            print(str(report), flush=True)
            store.append_message(thread_id, "user", user_input)
            store.append_message(thread_id, "assistant", str(report))
            store.update_summary_with_llm(
                thread_id=thread_id,
                prev_summary=prev_summary,
                prev_facts=prev_facts,
                user_text=user_input,
                assistant_text=str(report),
            )
            continue

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

        print("助手: ", end="", flush=True)
        assistant_text_parts = []
        for chunk, _meta in agent.stream({"messages": messages}, stream_mode="messages"):
            if isinstance(chunk, ToolMessageChunk):
                continue
            if isinstance(chunk, AIMessageChunk) and not _is_tool_call_message_chunk(chunk):
                text = _to_text(getattr(chunk, "content", ""))
                if text:
                    assistant_text_parts.append(text)
                    print(text, end="", flush=True)
        print("", flush=True)
        assistant_text = "".join(assistant_text_parts).strip()
        store.append_message(thread_id, "user", user_input)
        store.append_message(thread_id, "assistant", assistant_text)
        store.update_summary_with_llm(
            thread_id=thread_id,
            prev_summary=prev_summary,
            prev_facts=prev_facts,
            user_text=user_input,
            assistant_text=assistant_text,
        )
if __name__ == "__main__":
    __main__()
    
