import datetime
import json
import re
import uuid
from typing import Any
from backend.core.db import get_db_connection


# ================= 1. 定义本地功能函数 =================
def get_current_time():
    """获取当前的时间，返回格式为 HH:MM:SS"""
    now = datetime.datetime.now()
    return {"current_time": now.strftime("%H:%M:%S")}


def get_current_date():
    """获取当前的日期，返回格式为 YYYY-MM-DD"""
    today = datetime.date.today()
    return {"current_date": today.strftime("%Y-%m-%d")}


def recommend_books(topic):
    """根据用户感兴趣的主题推荐相关书籍"""
    print(f"\n[SQLite DB] 正在查询与 '{topic}' 相关的推荐书籍...")
    conn = get_db_connection()
    cursor = conn.cursor()
    # 扩大搜索范围，使得不仅搜索主题，也搜索书名和作者
    search_term = f"%{topic}%"
    cursor.execute(
        "SELECT title, author FROM books WHERE topic LIKE ? OR title LIKE ? OR author LIKE ?",
        (search_term, search_term, search_term),
    )
    rows = cursor.fetchall()
    conn.close()

    recommended_books = [
        {"title": row["title"], "author": row["author"]} for row in rows
    ]

    if not recommended_books:
        return {
            "topic": topic,
            "error": f"抱歉，没有找到与 '{topic}' 相关的书籍。请尝试其他关键词。",
        }

    return {
        "topic": topic,
        "recommended_books": recommended_books,
    }


def query_book_info(book_name):
    """查询指定书籍的馆藏信息（索书号、位置、借阅状态）"""
    print(f"\n[SQLite DB] 正在查询书籍 '{book_name}' 的馆藏状态...")
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT call_number, location, status FROM books WHERE title LIKE ?",
        ("%" + book_name + "%",),
    )
    row = cursor.fetchone()
    conn.close()

    if row:
        return {
            "book_name": book_name,
            "call_number": row["call_number"],
            "location": row["location"],
            "status": row["status"],
        }
    else:
        return {"book_name": book_name, "error": "未查询到该书籍的馆藏信息"}


AVAILABLE_FUNCTIONS = {
    "get_current_time": get_current_time,
    "get_current_date": get_current_date,
    "recommend_books": recommend_books,
    "query_book_info": query_book_info,
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前的准确时间",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_date",
            "description": "获取当前的准确日期",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_books",
            "description": "根据用户感兴趣的主题推荐相关书籍",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "用户感兴趣的主题，例如：人工智能、历史、Python等",
                    }
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_book_info",
            "description": "查询指定书籍的馆藏信息（索书号、位置、借阅状态）",
            "parameters": {
                "type": "object",
                "properties": {
                    "book_name": {"type": "string", "description": "书籍的名称"}
                },
                "required": ["book_name"],
            },
        },
    },
]


# ================= 2. 辅助解析与清理函数 =================
def clean_output(text: str) -> str:
    """清理输出中可能带有的大模型思考过程 <think>"""
    if text:
        # 清除完整的 <think>...</think>
        text = re.sub(r"<think>.*?</think>\n*", "", text, flags=re.DOTALL)
        # 如果还有未闭合的 <think>，去掉标签保留内容，以免输出空白
        text = text.replace("<think>", "").replace("</think>", "")
        return text.strip()
    return ""


def parse_manual_tool_calls(content: str):
    """手动从文本中解析可能的 XML 格式 tool_call 或纯 JSON 格式"""
    tool_calls: list[Any] = []
    if not content:
        return tool_calls

    class DummyFunction:
        def __init__(self, name, arguments):
            self.name = name
            self.arguments = (
                json.dumps(arguments, ensure_ascii=False)
                if isinstance(arguments, dict)
                else arguments
            )

    class DummyToolCall:
        def __init__(self, function):
            self.id = "call_" + str(uuid.uuid4())[:8]
            self.type = "function"
            self.function = function

    # 1. 尝试匹配 <tool_call> ... </tool_call>
    pattern = r"<tool_call>\s*({.*?})\s*</tool_call>"
    matches = re.findall(pattern, content, flags=re.DOTALL)

    if matches:
        for match in matches:
            try:
                tool_data = json.loads(match)
                if "name" in tool_data:
                    func = DummyFunction(
                        tool_data["name"], tool_data.get("arguments", {})
                    )
                    tool_calls.append(DummyToolCall(func))
            except json.JSONDecodeError:
                continue
        if tool_calls:
            return tool_calls

    # 2. 尝试解析纯 JSON 文本（先清理掉 <think> 标签）
    content_clean = clean_output(content)

    # 尝试寻找 ```json ... ``` 代码块
    json_blocks = re.findall(
        r"```(?:json)?\s*(\{.*?\})\s*```", content_clean, flags=re.DOTALL
    )
    if json_blocks:
        for block in json_blocks:
            try:
                tool_data = json.loads(block)
                if "name" in tool_data and (
                    "arguments" in tool_data or "parameters" in tool_data
                ):
                    args = tool_data.get("arguments", tool_data.get("parameters", {}))
                    func = DummyFunction(tool_data["name"], args)
                    tool_calls.append(DummyToolCall(func))
            except json.JSONDecodeError:
                continue
        if tool_calls:
            return tool_calls

    # 3. 尝试直接从文本中提取 {...}
    start = content_clean.find("{")
    end = content_clean.rfind("}")
    if start != -1 and end != -1 and end > start:
        possible_json = content_clean[start : end + 1]
        try:
            tool_data = json.loads(possible_json)
            if "name" in tool_data and (
                "arguments" in tool_data or "parameters" in tool_data
            ):
                args = tool_data.get("arguments", tool_data.get("parameters", {}))
                func = DummyFunction(tool_data["name"], args)
                tool_calls.append(DummyToolCall(func))
        except json.JSONDecodeError:
            pass

    return tool_calls
