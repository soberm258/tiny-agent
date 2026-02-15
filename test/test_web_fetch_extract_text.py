from agent.tools.html_extract import extract_text_from_html


def test_extract_text_from_html_keeps_chinese() -> None:
    html = """
    <html>
      <head><title>测试页面</title><style>.x{}</style></head>
      <body>
        <script>var a=1;</script>
        <h1>标题</h1>
        <p>这是中文内容，不应该变成乱码。</p>
      </body>
    </html>
    """
    title, text = extract_text_from_html(html)
    assert title == "测试页面"
    assert "这是中文内容" in text
    assert "�" not in text
