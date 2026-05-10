import openai
import pickle
import numpy as np
import faiss
import re
import json
import datetime
import os
from sentence_transformers import SentenceTransformer, CrossEncoder


emb_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
reranker_model = CrossEncoder("BAAI/bge-reranker-base")
index = faiss.read_index("data/rag/faiss_index.bin")
with open("data/rag/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

client = openai.OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="none")


# 示例函数
def get_current_time():
    return {"current_time": datetime.datetime.now().strftime("%H:%M:%S")}


def get_current_date():
    return {"current_date": datetime.date.today().strftime("%Y-%m-%d")}


def recommend_books(topic: str):
    print(f"后端执行: 正在连接数据库推荐书籍... 主题: '{topic}'")
    return {
        "topic": topic,
        "recommended_books": [
            {"title": "深入理解计算机系统", "author": "Randal E. Bryant"},
            {"title": "算法导论", "author": "Thomas H. Cormen"},
        ],
    }


def query_book_info(book_name: str):
    print(f"后端执行: 正在连接数据库查询书籍... 书名: '{book_name}'")
    return {
        "book_name": book_name,
        "call_number": "TP311.56/123",
        "location": "二楼/南区/51号架",
        "status": "在馆可借",
    }


# 工具清单
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前的准确时间（几点）。当用户询问时间时，必须使用此工具获取准确信息。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_date",
            "description": "获取当前的准确日期（几号）。当用户询问日期时，必须使用此工具获取准确信息。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_books",
            "description": "根据用户感兴趣的主题推荐相关书籍。当用户询问任何书籍推荐时，必须使用此工具。",
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
    },
    {
        "type": "function",
        "function": {
            "name": "query_book_info",
            "description": "查询指定书籍的馆藏信息(包括位置、索书号、状态)。当用户查找书籍时，必须使用此工具。",
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
    },
]


# 清理输出
def clean_output(text):
    if text:
        text = re.sub(r"<think>.*?</think>\n*", "", text, flags=re.DOTALL)
        return text.replace("<think>", "").replace("</think>", "").strip()
    return ""


# RAG 召回
def retrieve_rag_context(query, top_k=5, rerank_top_k=2):
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


# 意图关键字检测
TOOL_TRIGGER_KEYWORDS = {
    "get_current_time": ["几点", "时间", "现在是", "当前时间", "什么时候了"],
    "get_current_date": ["几号", "日期", "今天是", "当前日期", "星期", "几月"],
    "recommend_books": ["推荐", "有什么书", "书籍", "书单", "想学", "看什么"],
    "query_book_info": ["在哪", "位置", "哪里", "查一下", "馆藏", "索书号", "借阅"],
}


# 判断意图决定是否强制使用工具
def detect_required_tool(query: str):
    for tool_name, keywords in TOOL_TRIGGER_KEYWORDS.items():
        if any(kw in query for kw in keywords):
            print(f"意图预判命中: \n{tool_name}")
            return {"type": "function", "function": {"name": tool_name}}
    return "auto"


# 获得系统提示词
def get_system_prompt():
    if os.path.exists("prompt/system.md"):
        with open("prompt/system.md", "r", encoding="utf-8") as f:
            base_prompt = f.read().strip()
    return base_prompt


# 主流程
def test_tool_calling_loop(query: str, chat_history: list = None):
    if chat_history is None:
        chat_history = []

    print(f"\n{'=' * 50}")
    print(f"用户输入: \n{query}")

    # RAG 检索
    context = retrieve_rag_context(query)

    strong_reminder = "【系统指令】如果问题涉及以下内容如查询当前时间、当前日期、推荐书籍或查询书籍馆藏位置等工具清单中所列举的工具时，你必须立即调用工具获取真实数据，绝不允许自行编造答案。"

    if context:
        print("触发 RAG")
        user_prompt = f"【参考资料】\n{context}\n\n【用户问题】\n{query}"
    else:
        user_prompt = f"{query}"

    messages = [{"role": "system", "content": get_system_prompt() + strong_reminder}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_prompt})

    # 第一次请求
    print("**第一次请求**")
    # 判定意图
    tool_choice = detect_required_tool(query)
    print(f"tool_choice: {tool_choice}")
    response = client.chat.completions.create(
        model="outputs/qwen_full",
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        temperature=0.5,
        max_tokens=1024,
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

            args = {}
            if tool_call.function.arguments:
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    pass

            if func_name == "get_current_time":
                func_result = get_current_time()
            elif func_name == "get_current_date":
                func_result = get_current_date()
            elif func_name == "recommend_books":
                func_result = recommend_books(topic=args.get("topic"))
            elif func_name == "query_book_info":
                func_result = query_book_info(book_name=args.get("book_name"))
            else:
                print(f"工具 {func_name} 不存在。")
                func_result = {"error": f"Function {func_name} not found"}

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
            temperature=0.6,
            max_tokens=4096,
        )

        final_reply = clean_output(second_response.choices[0].message.content)
        print(f"最终回复: \n{final_reply}")

    else:
        final_reply = clean_output(response_msg.content)
        print(f"模型未使用工具进行回复: \n{final_reply}")

    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": final_reply})
    return chat_history


if __name__ == "__main__":
    queries = [
        "你好，你是谁？",
        "图书馆什么时候开门？",
        "请问现在是几点？",
        "我想学习计算机网络，可以推荐几本书吗？",
        "那帮我查一下《算法导论》在图书馆的什么位置？",
        "我想知道图书馆每天开门的时间",
    ]

    history = []
    for q in queries:
        history = test_tool_calling_loop(q, history)
