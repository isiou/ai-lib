import datetime
import json
import re
import uuid
import logging
from typing import Any
from backend.core.db import get_db_connection

logger = logging.getLogger(__name__)


# 获取当前精确时间
def get_current_time():
    now = datetime.datetime.now()
    return {"current_time": now.strftime("%H:%M:%S")}


# 获取当前具体日期
def get_current_date():
    today = datetime.date.today()
    return {"current_date": today.strftime("%Y-%m-%d")}


# 依据用户话题通过数据库检索推荐书籍
def recommend_books(topic):
    logger.info(f"[SQLite DB] 正在查询与 '{topic}' 相关的推荐书籍...")
    conn = get_db_connection()
    cursor = conn.cursor()
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
            "error": f"抱歉 未找到与 '{topic}' 相关的书籍 请尝试其他关键词",
        }
    return {"topic": topic, "recommended_books": recommended_books}


# 根据书名精确查询图书馆馆藏相关状态
def query_book_info(book_name):
    logger.info(f"[SQLite DB] 正在查询书籍 '{book_name}' 的馆藏状态...")
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

# 注册给 LLM 使用的工具规范描述列表
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
                        "description": "用户感兴趣的主题 例如人工智能 历史 Python等",
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
            "description": "查询指定书籍的馆藏信息 包含索书号 位置 借阅状态等",
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


# 清理模型回复中携带的多余思考过程标签
def clean_output(text: str) -> str:
    if text:
        text = re.sub("<think>.*?</think>\\n*", "", text, flags=re.DOTALL)
        text = text.replace("<think>", "").replace("</think>", "")
        return text.strip()
    return ""


# 自研兜底解析器 用于捕捉模型未遵循标准 API 而输出的文本型工具请求
def parse_manual_tool_calls(content: str):
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

    # 优先尝试匹配内置的自定义 XML 工具块
    pattern = "<tool_call>\\s*({.*?})\\s*</tool_call>"
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

    # 若匹配不到则清理思考文本后进一步搜寻代码块
    content_clean = clean_output(content)
    json_blocks = re.findall(
        "```(?:json)?\\s*(\\{.*?\\})\\s*```", content_clean, flags=re.DOTALL
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

    # 最后一招直接截取文本中的疑似 JSON 对象
    start = content_clean.find("{")
    end = content_clean.rfind("}")
    if start != -1 and end != -1 and (end > start):
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
