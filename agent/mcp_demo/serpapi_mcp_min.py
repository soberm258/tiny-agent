from __future__ import annotations

import ast
import asyncio
import json
import os

from dotenv import load_dotenv


async def main() -> None:
    load_dotenv()
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except Exception as e:
        print("缺少依赖：langchain-mcp-adapters。")
        print('请先安装：pip install "langchain-mcp-adapters"')
        print(f"导入错误：{e}")
        return
    
    api_key = (os.environ.get("SERPAPI_API_KEY") or "").strip()
    if not api_key:
        print("请先设置环境变量 SERPAPI_API_KEY（不要把 key 写进代码或日志）。")
        return

    client = MultiServerMCPClient(
        {
            "serpapi": {
                "transport": "http",
                "url": f"https://mcp.serpapi.com/{api_key}/mcp",
            }
        }
    )

    tools = await client.get_tools()
    names = [getattr(t, "name", "") for t in tools]
    print("MCP tools：", ", ".join([n for n in names if n]))

    search_tool = None
    for t in tools:
        if getattr(t, "name", "") == "search":
            search_tool = t
            break

    if search_tool is None:
        print("未找到 tool=search，请检查 MCP 服务返回的工具列表。")
        return

    result = await search_tool.ainvoke(
        {
            "params": {"q": "醉驾 危险驾驶罪 缓刑 条件",
                       "json_restrictor":"organic_results"},
            "mode": "complete",
        }
    )

    print(type(result))
    print(type(result[0]))
    print(result[0])

    # 解析内部的 text 字段
    for item in result:
        if isinstance(item.get("text"), str):
            # 这里需要处理内部 JSON 字符串
            text_content = item["text"]
            # 解析内部 JSON
            try:
                inner_data = json.loads(text_content)
                item["text"] = inner_data
                print("内部 JSON 解析成功")
            except json.JSONDecodeError as e:
                print(f"内部 JSON 解析失败: {e}")
    print(result[0]['text']['organic_results'][0]['title'])
    print(result[0]['text']['organic_results'][0]['snippet']) 
    print(json.dumps(result[0]['text']['organic_results'], ensure_ascii=False, indent=2))
    
if __name__ == "__main__":
    asyncio.run(main())

