import pickle
import numpy as np
import faiss
import re
from sentence_transformers import SentenceTransformer
from openai import OpenAI

print("===== 正在初始化 RAG 模块 =====")
try:
    # 加载 RAG 模型
    emb_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

    # 加载索引
    index = faiss.read_index("data/rag/faiss_index.bin")

    # 加载知识库
    with open("data/rag/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    print("RAG 知识库与模型加载成功")
except Exception as e:
    print("加载 RAG 发生错误")
    print(e)
    exit(1)

# 初始化 vLLM 客户端
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="none")


def chat_with_rag(query, top_k=2):
    # 编码查询
    query_emb = emb_model.encode([query], normalize_embeddings=True).astype(np.float32)

    # 向量检索
    distances, indices = index.search(query_emb, top_k)

    retrieved_texts = []
    for idx in indices[0]:
        if idx != -1:
            retrieved_texts.append(chunks[idx])

    # 构造增强 Prompt
    context = "\n".join(retrieved_texts)
    print(f"\n[\033[92m检索到的参考知识\033[0m]：\n{context}\n")

    system_prompt = f"""你是一个图书馆智能助手，请根据以下参考资料回答用户的问题。
    如果参考资料中没有相关信息，请根据你的原有知识回答。如果回答，必须保持礼貌和专业。
    参考资料：
    {context}"""

    # 调用大模型
    response = client.chat.completions.create(
        model="outputs/qwen_full",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        temperature=0.7,
    )

    reply = response.choices[0].message.content

    # 正则过滤 <think></think> 标签
    reply_cleaned = re.sub(r"<think>.*?</think>\n*", "", reply, flags=re.DOTALL).strip()

    print(f"[\033[96massistant: \033[0m]\n{reply_cleaned}\n")


if __name__ == "__main__":
    queries = [
        "你是谁？",
        "今天热不热啊？",
        "如果我借书逾期了，罚款怎么算？每天多少钱？",
        "研讨室在几楼？",
        "当前有多少馆藏？",
        "怎么样借书？",
        "图书馆什么时候开门？"
    ]

    for q in queries:
        print("=" * 50)
        print(f"\033[93muser: {q}\033[0m")
        chat_with_rag(q)
