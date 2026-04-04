import json
import datetime
from openai import OpenAI
import re
import uuid


# 1. 定义本地功能函数示例
def get_current_time():
    now = datetime.datetime.now()
    return {"current_time": now.strftime("%H:%M:%S")}


def get_current_date():
    today = datetime.date.today()
    return {"current_date": today.strftime("%Y-%m-%d")}


# 2. 统一维护的函数注册表
AVAILABLE_FUNCTIONS = {
    "get_current_time": get_current_time,
    "get_current_date": get_current_date,
}

# 3. 统一管理并定义 Tools
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前的实时准确时间",
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
]

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="none")


def clean_output(text):
    # 清理可能存在的 <think> 标签
    if text:
        return re.sub(r"<think>.*?</think>\n*", "", text, flags=re.DOTALL).strip()
    return ""


def parse_manual_tool_calls(content):
    # 手动从文本中解析可能的 XML 格式 tool_call
    tool_calls = []
    if not content:
        return tool_calls

    # 匹配 <tool_call> { "name": "...", "arguments": {...} } </tool_call>
    pattern = r"<tool_call>\s*({.*?})\s*</tool_call>"
    matches = re.findall(pattern, content, flags=re.DOTALL)

    for match in matches:
        try:
            tool_data = json.loads(match)
            if "name" in tool_data:
                # 伪造 OpenAI 的 ToolCall 对象结构，用于后续逻辑
                class DummyFunction:
                    def __init__(self, name, arguments):
                        self.name = name
                        self.arguments = (
                            json.dumps(arguments)
                            if isinstance(arguments, dict)
                            else arguments
                        )

                class DummyToolCall:
                    def __init__(self, function):
                        self.id = "call_" + str(uuid.uuid4())[:8]
                        self.type = "function"
                        self.function = function

                func = DummyFunction(tool_data["name"], tool_data.get("arguments", {}))
                tool_calls.append(DummyToolCall(func))
        except json.JSONDecodeError:
            continue

    return tool_calls


def chat_with_tools(query, model="outputs/qwen_full"):
    # 构建 System Prompt 以引导模型正确使用 Tool Calling
    system_prompt = """作为一个多功能的人工智能助手，你可以使用以下工具来帮助用户解决问题：
    1. `get_current_time`: 获取当前的准确时间。
    2. `get_current_date`: 获取当前的准确日期。
    注意：如果需要使用工具，**必须严格输出如下格式**：
    `<tool_call>{"name": "工具名称", "arguments": {}}</tool_call>`
    若不需要使用工具，直接回答用户的问题。"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    try:
        # 第一轮对话
        # 触发大模型思考并可能调用工具
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.1,
        )

        response_message = response.choices[0].message

        # 尝试获取标准 tool_calls
        tool_calls = response_message.tool_calls
        content = response_message.content or ""

        # 尝试手动从 content 中提取
        if not tool_calls and "<tool_call>" in content:
            tool_calls = parse_manual_tool_calls(content)

        if tool_calls:
            print("[\033[94m模型尝试调用工具...\033[0m]")

            # 追加助手消息
            if not response_message.tool_calls:
                messages.append({"role": "assistant", "content": content})
            else:
                messages.append(response_message)

            for tool_call in tool_calls:
                function_name = tool_call.function.name

                args = {}
                if tool_call.function.arguments:
                    try:
                        args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        pass

                print(f"\033[95m正在执行函数: {function_name}({args})\033[0m")

                if function_name in AVAILABLE_FUNCTIONS:
                    func = AVAILABLE_FUNCTIONS[function_name]
                    function_response = func()

                    print(f"\033[92m函数返回结果: {function_response}\033[0m")

                    # 追加工具调用结果消息
                    tool_msg_content = json.dumps(function_response, ensure_ascii=False)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": tool_msg_content,
                        }
                    )
                else:
                    print(f"未找到函数 {function_name}")
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": json.dumps(
                                {"error": f"Function {function_name} not found"}
                            ),
                        }
                    )

            # 第二轮对话
            # 将工具返回给模型并生成最终回答
            second_response = client.chat.completions.create(
                model=model, messages=messages, temperature=0.5
            )

            final_reply = clean_output(second_response.choices[0].message.content)
            print(f"[\033[96massistant\033[0m]:\n{final_reply}\n")
        else:
            final_reply = clean_output(response_message.content)
            print(f"[\033[96massistant\033[0m]:\n{final_reply}\n")

    except Exception as e:
        print(f"[\033[91m发生错误\033[0m]: {str(e)}")


if __name__ == "__main__":
    print("============ 测试 Function Calling 统一接口 ============")
    test_queries = [
        "你好，请问现在几点了？",
        "今天是几月几号？",
        "你能告诉我今天日期和现在的时间分别是什么吗？",
        "你是谁？",
    ]

    for q in test_queries:
        chat_with_tools(q)
