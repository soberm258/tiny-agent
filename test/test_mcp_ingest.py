import ast
import json

# 使用原始字符串（r前缀），但这样会保留 \n 作为两个字符
s = r"""[{'type': 'text', 'text': '{\n  "organic_results": [\n    {\n      "position": 1,\n      "title": "最高人民法院最高人民检察院公 安部司法部关于办理醉酒危险驾驶 ...",\n      "link": "https://www.spp.gov.cn/spp/xwfbh/wsfbt/202312/t20231218_637161.shtml",\n      "displayed_link": "www.spp.gov.cn › spp › xwfbh › wsfbt",\n      "snippet": "第四条 在道路上驾驶机动车，经呼气酒精含量检测， 显示血液酒精含量达到80毫克/100毫升以上的，公安机关应当依照刑事诉讼法和本意见的规定决定是否立案。对 ...",\n      "date": "Dec 18, 2023"\n    }\n]\n}', 'id': 'lc_f5ffa57f-9f84-4c2f-9e77-1ccd5db30163'}]"""

# 解析最外层
data = ast.literal_eval(s)
print(f"外层解析成功，数据类型: {type(data)}")
print(f"元素数量: {len(data)}")

# 解析内部的 text 字段
for item in data:
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

print(data[0]['text']['organic_results'][0]['title'])
print(data[0]['text']['organic_results'][0]['snippet'])

file_path = "test\\test.jsonl"
with open(file_path, 'w', encoding='utf-8') as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')