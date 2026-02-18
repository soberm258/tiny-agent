import json
import os
import sys

sys.path.append(".")


def test_web_search_missing_api_key_returns_error() -> None:
    try:
        from agent.tools.web_search import web_search
    except ModuleNotFoundError as e:
        import pytest

        pytest.skip(f"缺少依赖，跳过：{e}")

    old = os.environ.pop("SERPAPI_API_KEY", None)
    try:
        out = web_search("测试", max_results=3)
        obj = json.loads(out)
        assert obj.get("results") == []
        assert "error" in obj
    finally:
        if old is not None:
            os.environ["SERPAPI_API_KEY"] = old
