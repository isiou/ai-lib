import pickle
import numpy as np
import faiss
import re
import json
import datetime
import uuid
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI

print("============ 初始化综合测试模块 (RAG + Function Calling) ============")
try:
    # 加载模型
    emb_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    reranker_model = CrossEncoder("BAAI/bge-reranker-base")

    # 加载索引
    index = faiss.read_index("data/rag/faiss_index.bin")

    # 加载知识库
    with open("data/rag/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
except Exception as e:
    print(e)
    exit(1)

# 创建连接对象
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="none")


# 本地函数
# 仅供测试用
def get_current_time():
    now = datetime.datetime.now()
    return {"current_time": now.strftime("%H:%M:%S")}


def get_current_date():
    today = datetime.date.today()
    return {"current_date": today.strftime("%Y-%m-%d")}


def recommend_books(topic):
    # 书籍推荐
    return {
        "topic": topic,
        "recommended_books": [
            {"title": "深度学习", "author": "Ian Goodfellow"},
            {"title": "人工智能", "author": "Stuart Russell"},
        ],
    }


def query_book_info(book_name):
    # 查询指定书籍
    return {
        "book_name": book_name,
        "call_number": "TP311.56/123",
        "location": "二楼/南区/51号架",
        "status": "在馆可借",
    }


# 工具函数注册
AVAILABLE_FUNCTIONS = {
    "get_current_time": get_current_time,
    "get_current_date": get_current_date,
    "recommend_books": recommend_books,
    "query_book_info": query_book_info,
}

# 描述
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
            "description": "查询指定书籍的馆藏信息",
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


# 处理 <think>
def clean_output(text):
    if text:
        text = re.sub(r"<think>.*?</think>\n*", "", text, flags=re.DOTALL)
        text = text.replace("<think>", "").replace("</think>", "")
        return text.strip()
    return ""


# 手动解析 tool_call
def parse_manual_tool_calls(content):
    tool_calls = []
    if not content:
        return tool_calls

    pattern = r"<tool_call>\s*({.*?})\s*</tool_call>"
    matches = re.findall(pattern, content, flags=re.DOTALL)

    for match in matches:
        try:
            tool_data = json.loads(match)
            if "name" in tool_data:

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


# 交互
def multi_turn_chat(
    conversation_turns, top_k=5, rerank_top_k=2, model="outputs/qwen_full"
):

    system_prompt = """
    # 角色设定
    你是的名字是厦小嘉，厦门大学嘉庚学院图书馆的专属智能助手。
    你性格温和、热情、乐于助人，语气总是充满亲和力。

    # 行为准则
    - 交流风格：自然流畅，不要机械地说“根据参考资料”。
    - 业务解答：请以提供的【参考资料】作为唯一事实依据。
    - 坚守边界：专业领域仅限于“图书馆及阅读相关业务”。
    - 主动服务：当用户对某本书感兴趣或你推荐了书籍后，你可以主动询问用户是否需要查询该书的具体馆藏信息（索书号、位置等）。

    # 工具使用规范
    你有以下必须使用的工具：
    1. `get_current_time`: 用于获取当前时间；
    2. `get_current_date`: 用于获取当前日期；
    3. `recommend_books`: 根据主题推荐书籍。传入参数 `topic` 为兴趣关键字；
    4. `query_book_info`: 查询书籍馆藏信息。传入参数 `book_name` 为书籍名称。

    如果需要调用工具，你**必须严格输出如下格式来调用相应的工具**，**不得添加任何额外字符**：
    `<tool_call>{"name": "工具名称", "arguments": {"参数名": "参数值"}}</tool_call>`

    **如果需要调用工具，请直接输出 <tool_call> XML，不需要任何多余的解释**。如果不需要工具则直接回答问题。 /nothink"""

    messages = [{"role": "system", "content": system_prompt}]

    for query in conversation_turns:
        print(f"[\033[93muser\033[0m]: {query}")

        user_prompt = f"{query} /nothink"
        messages.append({"role": "user", "content": user_prompt})

        try:
            # 推理阶段
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.1,
            )

            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls
            content = response_message.content or ""

            if not tool_calls and "<tool_call>" in content:
                tool_calls = parse_manual_tool_calls(content)

            if tool_calls:
                print("[\033[94m模型决定调用工具\033[0m]")

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

                    print(f" -> 正在执行函数: \033[95m{function_name}({args})\033[0m")

                    if function_name in AVAILABLE_FUNCTIONS:
                        func = AVAILABLE_FUNCTIONS[function_name]
                        function_response = func(**args) if args else func()
                        print(f" -> 函数返回结果: \033[92m{function_response}\033[0m")

                        tool_msg_content = json.dumps(
                            function_response, ensure_ascii=False
                        )
                        if not response_message.tool_calls:
                            messages.append(
                                {
                                    "role": "user",
                                    "content": f"工具 {function_name} 返回了结果：\n{tool_msg_content}\n请根据此结果直接回答用户的问题。",
                                }
                            )
                        else:
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "name": function_name,
                                    "content": tool_msg_content,
                                }
                            )
                    else:
                        print(f" -> 错误：未找到函数 {function_name}")
                        if not response_message.tool_calls:
                            messages.append(
                                {
                                    "role": "user",
                                    "content": f"工具 {function_name} 不存在，请直接回答用户。",
                                }
                            )
                        else:
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
                second_response = client.chat.completions.create(
                    model=model, messages=messages, temperature=0.5
                )
                raw_reply = second_response.choices[0].message.content
                final_reply = clean_output(raw_reply)
                print(f"[\033[96massistant\033[0m]:\n{final_reply}\n")

                # 保存助手的回复到历史记录
                messages.append({"role": "assistant", "content": final_reply})
            else:
                raw_reply = response_message.content
                final_reply = clean_output(raw_reply)
                print(f"[\033[96massistant\033[0m]:\n{final_reply}\n")

                # 保存助手的回复到历史记录
                messages.append({"role": "assistant", "content": final_reply})

        except Exception as e:
            print(f"[\033[91m发生错误\033[0m]: {str(e)}")


def integrated_chat(query, top_k=5, rerank_top_k=2, model="outputs/qwen_full"):
    print(f"\n{'=' * 50}")
    print(f"[\033[93muser\033[0m]: {query}")

    # --- RAG 检索阶段 ---
    # 编码查询
    query_emb = emb_model.encode([query], normalize_embeddings=True).astype(np.float32)

    # 1. 向量检索初筛
    distances, indices = index.search(query_emb, top_k)

    retrieved_texts = []
    for idx in indices[0]:
        if idx != -1:
            retrieved_texts.append(chunks[idx])

    # 2. 重排序精排
    if retrieved_texts:
        cross_inp = [[query, text] for text in retrieved_texts]
        scores = reranker_model.predict(cross_inp)

        doc_score_pairs = sorted(
            zip(retrieved_texts, scores), key=lambda x: x[1], reverse=True
        )

        # 只保留 score > 0.5 的高置信度片段
        retrieved_texts = [
            doc for doc, score in doc_score_pairs[:rerank_top_k] if score > 0.5
        ]

    context = "\n".join(retrieved_texts) if retrieved_texts else "无匹配的参考资料"
    if retrieved_texts:
        print(f"[\033[92mRAG 已检索到 {len(retrieved_texts)} 条参考知识\033[0m]")

    # --- 构造 Prompt ---
    system_prompt = """# 角色设定
你是“厦小嘉”，厦门大学嘉庚学院图书馆的专属智能助手。
你性格温和、热情、乐于助人，语气总是充满亲和力。

# 行为准则
- 交流风格：自然流畅，不要机械地说“根据参考资料”。
- 业务解答：请以提供的【参考资料】作为唯一事实依据。
- 坚守边界：专业领域仅限于“图书馆及阅读相关业务”。

# 工具使用规范
你有两个必须使用的工具：
1. `get_current_time`: 用于获取当前时间（几点）。
2. `get_current_date`: 用于获取当前日期（几月几号）。

如果用户的问题中询问了日期或时间，你必须立刻输出如下格式来调用相应的工具：
<tool_call>
{"name": "工具名称", "arguments": {}}
</tool_call>

注意：如果需要调用工具，请直接输出 <tool_call> XML，不需要任何多余的解释。如果不需要工具，就直接回答问题。"""

    if retrieved_texts:
        user_prompt = f"【参考资料】\n{context}\n\n【用户问题】\n{query} /no_think\n"
    else:
        user_prompt = f"{query} /no_think"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        # --- LLM 推理阶段 ---
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.1,
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        content = response_message.content or ""

        if not tool_calls and "<tool_call>" in content:
            tool_calls = parse_manual_tool_calls(content)

        if tool_calls:
            print("[\033[94m模型决定调用工具\033[0m]")

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

                print(f" -> 正在执行函数: \033[95m{function_name}({args})\033[0m")

                if function_name in AVAILABLE_FUNCTIONS:
                    func = AVAILABLE_FUNCTIONS[function_name]
                    function_response = func()
                    print(f" -> 函数返回结果: \033[92m{function_response}\033[0m")

                    tool_msg_content = json.dumps(function_response, ensure_ascii=False)
                    if not response_message.tool_calls:
                        # 对于手动解析的情况，使用 user 角色返回结果，并采用明确的提示
                        messages.append(
                            {
                                "role": "user",
                                "content": f"工具 {function_name} 返回了结果：\n{tool_msg_content}\n请根据此结果直接回答用户的问题。",
                            }
                        )
                    else:
                        # 对于标准 API 调用的情况
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": function_name,
                                "content": tool_msg_content,
                            }
                        )
                else:
                    print(f" -> 错误：未找到函数 {function_name}")
                    if not response_message.tool_calls:
                        messages.append(
                            {
                                "role": "user",
                                "content": f"工具 {function_name} 不存在，请直接回答用户。",
                            }
                        )
                    else:
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
            second_response = client.chat.completions.create(
                model=model, messages=messages, temperature=0.5
            )
            raw_reply = second_response.choices[0].message.content
            print(f"[Debug] Raw reply: {raw_reply!r}")
            final_reply = clean_output(raw_reply)
            print(f"[\033[96massistant\033[0m]:\n{final_reply}\n")
        else:
            raw_reply = response_message.content
            print(f"[Debug] Raw reply: {raw_reply!r}")
            final_reply = clean_output(raw_reply)
            print(f"[\033[96massistant\033[0m]:\n{final_reply}\n")

    except Exception as e:
        print(f"[\033[91m发生错误\033[0m]: {str(e)}")


if __name__ == "__main__":
    print("\n[ 单轮问答测试 ]")
    test_queries = [
        "你是谁？",
        "你好，请问现在几点了？",
        "如果我借书逾期了，罚款怎么算？每天多少钱？",
        "你知道今天是几号吗，顺便告诉我图书馆什么时候开门？",
        "研讨室在几楼？怎么预约？",
    ]

    for q in test_queries:
        integrated_chat(q)

    # 运行多轮对话测试
    multi_turn_conversation = [
        "我想学习人工智能，你有什么推荐的吗？",
        "好的，帮我查一下第一本《深度学习》的信息。",
    ]
    multi_turn_chat(multi_turn_conversation)
