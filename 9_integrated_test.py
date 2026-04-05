import pickle
import numpy as np
import faiss
import re
import json
import datetime
import os
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI

# 1. 初始化模型与基础组件
emb_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
reranker_model = CrossEncoder("BAAI/bge-reranker-base")
index = faiss.read_index("data/rag/faiss_index.bin")
with open("data/rag/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="none")


# 2. 定义可用工具集合与描述清单
class Tools:
    @staticmethod
    def get_current_time():
        return {"current_time": datetime.datetime.now().strftime("%H:%M:%S")}

    @staticmethod
    def get_current_date():
        return {"current_date": datetime.date.today().strftime("%Y-%m-%d")}

    @staticmethod
    def recommend_books(topic: str):
        return {
            "topic": topic,
            "recommended_books": [
                {"title": "深入理解计算机系统", "author": "Randal E. Bryant"},
                {"title": "算法导论", "author": "Thomas H. Cormen"},
            ],
        }

    @staticmethod
    def query_book_info(book_name: str):
        return {
            "book_name": book_name,
            "call_number": "TP311.56/123",
            "location": "二楼/南区/51号架",
            "status": "在馆可借",
        }


AVAILABLE_FUNCTIONS = {
    "get_current_time": Tools.get_current_time,
    "get_current_date": Tools.get_current_date,
    "recommend_books": Tools.recommend_books,
    "query_book_info": Tools.query_book_info,
}

TOOLS_SCHEMA = [
    {
        "name": "get_current_time",
        "description": "获取当前的准确时间（几点）",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_current_date",
        "description": "获取当前的准确日期（几号）",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "recommend_books",
        "description": "根据用户感兴趣的主题推荐相关书籍",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "用户感兴趣的主题，例如：人工智能、计算机网络等",
                }
            },
            "required": ["topic"],
        },
    },
    {
        "name": "query_book_info",
        "description": "查询指定书籍的馆藏信息(包括位置、索书号、状态)",
        "parameters": {
            "type": "object",
            "properties": {
                "book_name": {
                    "type": "string",
                    "description": "书籍的名称，如：深度学习",
                }
            },
            "required": ["book_name"],
        },
    },
]


def get_system_prompt():
    """读取基础 system.md 并附加工具调用的强化指令"""
    base_prompt = ""
    if os.path.exists("prompt/system.md"):
        with open("prompt/system.md", "r", encoding="utf-8") as f:
            base_prompt = f.read().strip()

    tool_instructions = f"""

# 工具使用指南 (Tool Calling)
你可以使用以下工具清单来辅助你回答用户的问题：
{json.dumps(TOOLS_SCHEMA, ensure_ascii=False, indent=2)}

【极其重要/强制规范】：
当用户询问的问题涉及当前时间、日期、查询具体书籍位置或需要推荐书籍时，你**绝对不能自己编造答案**。你必须且只能输出以下 XML 格式来调用工具：
<tool_call>
{{"name": "工具名称", "arguments": {{"参数名": "参数值"}}}}
</tool_call>

【工具调用示例】：
用户：现在几点了？
厦小嘉：
<tool_call>
{{"name": "get_current_time", "arguments": {{}}}}
</tool_call>

用户：我想学习人工智能，推荐几本书
厦小嘉：
<tool_call>
{{"name": "recommend_books", "arguments": {{"topic": "人工智能"}}}}
</tool_call>

用户：查询《红楼梦》的位置
厦小嘉：
<tool_call>
{{"name": "query_book_info", "arguments": {{"book_name": "红楼梦"}}}}
</tool_call>

注意：如果不需要调用工具（即正常的寒暄或已经可以根据常识/参考资料直接回答），请直接给出友好的自然语言回复，绝不要输出 `<tool_call>` 标签。
"""
    return base_prompt + tool_instructions


def clean_output(text):
    if text:
        text = re.sub(r"<think>.*?</think>\n*", "", text, flags=re.DOTALL)
        return text.replace("<think>", "").replace("</think>", "").strip()
    return ""


def parse_manual_tool_calls(content):
    """解析大模型输出的 XML 工具标签"""
    if not content:
        return None
    pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
    match = re.search(pattern, content, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
    return None


def retrieve_rag_context(query, top_k=5, rerank_top_k=2):
    """封装 RAG 检索逻辑"""
    query_emb = emb_model.encode([query], normalize_embeddings=True).astype(np.float32)
    distances, indices = index.search(query_emb, top_k)

    retrieved_texts = [chunks[idx] for idx in indices[0] if idx != -1]

    if retrieved_texts:
        cross_inp = [[query, text] for text in retrieved_texts]
        scores = reranker_model.predict(cross_inp)
        doc_score_pairs = sorted(
            zip(retrieved_texts, scores), key=lambda x: x[1], reverse=True
        )
        retrieved_texts = [
            doc for doc, score in doc_score_pairs[:rerank_top_k] if score > 0.5
        ]

    return "\n".join(retrieved_texts) if retrieved_texts else ""


def agent_chat(query: str, chat_history: list = None):
    """
    核心 Agent Loop: 模型分析 -> XML工具解析 -> 工具执行 -> 结果喂入二次推理 -> 输出自然语言
    """
    if chat_history is None:
        chat_history = []

    print(f"\n{'=' * 50}")
    print(f"[\033[93m用户\033[0m]: {query}")

    # 1. RAG 检索
    context = retrieve_rag_context(query)
    if context:
        print("[\033[92m系统\033[0m]: 触发 RAG 知识检索，已附加至上下文。")
        user_prompt = f"【参考资料】\n{context}\n\n【用户问题】\n{query} /nothink"
    else:
        user_prompt = f"{query} /nothink"

    # 2. 构建对话信息
    messages = [{"role": "system", "content": get_system_prompt()}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_prompt})

    try:
        # 3. 第一次推理：意图识别与工具分配
        response = client.chat.completions.create(
            model="outputs/qwen_full",
            messages=messages,
            temperature=0.1,  # 低温有助于更稳定输出 XML
            max_tokens=1024,
        )

        content = response.choices[0].message.content or ""
        print(f"[\033[90mDebug Raw\033[0m]: {content!r}")

        # 4. 后端拦截：检查是否需要调用工具
        tool_call_data = parse_manual_tool_calls(content)

        if tool_call_data and "name" in tool_call_data:
            func_name = tool_call_data["name"]
            args = tool_call_data.get("arguments", {})
            print(
                f"[\033[94m系统\033[0m]: 模型判断需要使用工具 -> \033[95m{func_name}({args})\033[0m"
            )

            # 执行工具
            if func_name in AVAILABLE_FUNCTIONS:
                func = AVAILABLE_FUNCTIONS[func_name]
                tool_result = func(**args) if args else func()
                print(f"[\033[94m系统\033[0m]: 工具返回结果 -> {tool_result}")

                # 5. 组装结果进行第二次推理
                # 将模型的工具调用指令作为 assistant 补充进去
                messages.append(
                    {
                        "role": "assistant",
                        "content": f"<tool_call>\n{json.dumps(tool_call_data, ensure_ascii=False)}\n</tool_call>",
                    }
                )
                # 将工具返回结果以 user (或 tool_result) 的身份重新喂入
                messages.append(
                    {
                        "role": "user",
                        "content": f"工具 {func_name} 返回了结果：\n{json.dumps(tool_result, ensure_ascii=False)}\n请结合此结果，用自然语言回答我刚才的问题。不要提及你调用了工具。",
                    }
                )

                print("[\033[94m系统\033[0m]: 结果已交回模型进行二次理解整合...")
                second_response = client.chat.completions.create(
                    model="outputs/qwen_full",
                    messages=messages,
                    temperature=0.6,
                )
                final_reply = clean_output(second_response.choices[0].message.content)
                print(f"[\033[96m厦小嘉\033[0m]:\n{final_reply}\n")

                # 将结果存入历史用于多轮对话
                chat_history.append({"role": "user", "content": query})
                chat_history.append({"role": "assistant", "content": final_reply})
            else:
                print(f"[\033[91m系统\033[0m]: 工具 {func_name} 不存在。")
        else:
            # 未触发工具调用，直接输出 RAG 或常识推理结果
            final_reply = clean_output(content)
            print(f"[\033[96m厦小嘉 (未调工具)\033[0m]:\n{final_reply}\n")

            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": final_reply})

        return chat_history
    except Exception as e:
        print(f"[\033[91m系统\033[0m] 发生错误: {str(e)}")
        return chat_history


if __name__ == "__main__":
    print(
        "\n\033[1m====== 厦小嘉 综合 Agent 测试流水线 (RAG + 强约束 Tool Calling) ======\033[0m"
    )

    # 测试集设计：混合闲聊、RAG问答、单工具查询、连续多轮工具查询
    queries = [
        "你好，你是谁？",  # 触发：System Prompt 常识
        "图书馆什么时候开门？",  # 触发：RAG 检索
        "请问现在是几点？",  # 触发：get_current_time
        "我想学习计算机网络，可以推荐几本书吗？",  # 触发：recommend_books
        "那帮我查一下《算法导论》在图书馆的什么位置？",  # 触发：query_book_info + 多轮上下文
    ]

    history = []
    for q in queries:
        history = agent_chat(q, history)
