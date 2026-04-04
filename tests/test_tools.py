import json
from backend.services.tools import clean_output, parse_manual_tool_calls


def test_clean_output():
    # 测试有完整 think 标签的内容
    text1 = "<think>思考过程</think>\n最终回答"
    assert clean_output(text1) == "最终回答"

    # 测试没有 think 标签的内容
    text2 = "正常回答"
    assert clean_output(text2) == "正常回答"

    # 测试不完整的 think 标签内容
    text3 = "<think>部分思考最终回答"
    assert clean_output(text3) == "部分思考最终回答"

    # 测试空输入
    assert clean_output("") == ""
    assert clean_output(None) == ""


def test_parse_manual_tool_calls_xml_format():
    content = '<tool_call>{"name": "get_current_time", "arguments": {}}</tool_call>'
    calls = parse_manual_tool_calls(content)
    assert len(calls) == 1
    assert calls[0].function.name == "get_current_time"
    assert json.loads(calls[0].function.arguments) == {}


def test_parse_manual_tool_calls_raw_json():
    content = '{"name": "query_book_info", "arguments": {"book_name": "深度学习"}}'
    calls = parse_manual_tool_calls(content)
    assert len(calls) == 1
    assert calls[0].function.name == "query_book_info"
    assert json.loads(calls[0].function.arguments) == {"book_name": "深度学习"}


def test_parse_manual_tool_calls_markdown_json():
    content = """```json
{
  "name": "recommend_books",
  "arguments": {
    "topic": "Python"
  }
}
```"""
    calls = parse_manual_tool_calls(content)
    assert len(calls) == 1
    assert calls[0].function.name == "recommend_books"
    assert json.loads(calls[0].function.arguments) == {"topic": "Python"}


def test_parse_manual_tool_calls_empty_or_invalid():
    assert len(parse_manual_tool_calls("")) == 0
    assert len(parse_manual_tool_calls("这不是合法的 JSON")) == 0
