import openai
import json


# 示例函数
def search_books(keyword, limit=10):
    mock_db_results = {
        "计算机网络": [
            {
                "书名": "计算机网络（第8版）",
                "作者": "谢希仁",
                "索书号": "TP393/123",
                "状态": "在馆",
            },
            {
                "书名": "自顶向下方法",
                "作者": "Kurose",
                "索书号": "TP393.4/456",
                "状态": "被借出",
            },
            {
                "书名": "TCP/IP 协议",
                "作者": "工业出版社",
                "索书号": "TP341.1/84",
                "状态": "在馆",
            },
        ],
        "Python": [
            {
                "书名": "Python编程从入门到实践",
                "作者": "Eric Matthes",
                "索书号": "TP311.56/888",
                "状态": "在馆",
            }
        ],
    }

    results = mock_db_results.get(keyword, [])
    if not results:
        return {"error": f"没有找到关于 '{keyword}' 的书籍"}

    # 限制返回数量
    return {"keyword": keyword, "results": results[:limit]}


# 工具清单
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_books",
            "description": "从图书馆数据库搜索书籍。当用户询问任何书籍推荐、查找书籍时，必须使用此工具获取准确信息。",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "搜索关键词，例如提取用户问题中的主题词：'计算机网络'、'Python' 等",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "返回结果数量限制",
                        "default": 10,
                    },
                },
                "required": ["keyword"],
            },
        },
    },
]

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="none")


def test_tool_calling_loop():
    with open("prompt/system.md", "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()

    messages = [
        {
            "role": "system",
            "content": system_prompt
            + "[!important]当用户询问书籍推荐、查找书籍、图书馆有哪些书时必须立即调用 search_books 工具，无需进行多余回复",
        },
        {
            "role": "system",
            "content": "若用户询问图书推荐等介绍时，优先调用工具进行推荐！",
        },
        {"role": "user", "content": "我想学习计算机网络，图书馆中有什么推荐的书籍吗？"},
    ]

    print(f"用户输入: \n{messages[-1]['content']}")

    # 第一次请求
    print("**第一次请求**")
    # required
    tool_choice = "auto"
    print(f"tool_choice: {tool_choice}")
    response = client.chat.completions.create(
        model="outputs/qwen_full",
        messages=messages,
        tools=tools,
        # tool_choice="required",
        tool_choice=tool_choice,
        temperature=0.1,
    )

    response_msg = response.choices[0].message

    if response_msg.tool_calls:
        print(f"模型决定调用工具: \n{response_msg}")

        # 将工具调用信息加入上下文
        messages.append(response_msg)

        for tool_call in response_msg.tool_calls:
            func_name = tool_call.function.name
            print(f"工具: {func_name}")
            print(f"参数: {tool_call.function.arguments}")

            if func_name == "search_books":
                # 解析参数
                args = json.loads(tool_call.function.arguments)

                # 本地执行函数
                func_result = search_books(
                    keyword=args.get("keyword"), limit=args.get("limit", 10)
                )

                # 将执行结果作为 tool 角色添加回消息列表
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": func_name,
                        "content": json.dumps(func_result, ensure_ascii=False),
                    }
                )

        # 整合后进行第二次请求
        print("**第二次请求**")
        second_response = client.chat.completions.create(
            model="outputs/qwen_full",
            messages=messages,
            temperature=0.7,
            max_tokens=4096,
            extra_body={"enable_thinking": False},
        )

        final_reply = second_response.choices[0].message.content
        print(f"最终回复: \n{final_reply}")

    else:
        print(f"模型未使用工具进行回复: \n{response_msg.content}")


if __name__ == "__main__":
    test_tool_calling_loop()
