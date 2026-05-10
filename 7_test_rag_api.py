import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI
import re


# 加载向量模型和重排模型
emb_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
reranker_model = CrossEncoder("BAAI/bge-reranker-base")

# FAISS 向量库
index = faiss.read_index("data/rag/faiss_index.bin")
with open("data/rag/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# 创建本地 API 连接对象
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="none")


def chat_with_rag(query, top_k=5, rerank_top_k=2):
    # 将输入进行向量化
    query_emb = emb_model.encode([query], normalize_embeddings=True).astype(np.float32)
    distances, indices = index.search(query_emb, top_k)

    retrieved_texts = []
    for idx in indices[0]:
        if idx != -1:
            retrieved_texts.append(chunks[idx])

    # 将召回内容进行重排
    if retrieved_texts:
        cross_inp = [[query, text] for text in retrieved_texts]
        scores = reranker_model.predict(cross_inp)
        doc_score_pairs = sorted(
            zip(retrieved_texts, scores), key=lambda x: x[1], reverse=True
        )
        # 重排序得分
        print("重排序得分详情列表: ")
        for doc, score in doc_score_pairs:
            print(f"Score: {score:.4f} | {doc}")
        print()

        # 裁剪出高度相关的结果区间进行下游对话构建
        retrieved_texts = [
            doc for doc, score in doc_score_pairs[:rerank_top_k] if score > 0.5
        ]

    context = "\n".join(retrieved_texts) if retrieved_texts else "无匹配的参考资料"
    print(f"归因后参考资料: \n{context}")
    print()

    with open("prompt/system.md", "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()

    user_prompt = f"/nothink\n【参考资料】\n{context}\n【用户问题】\n{query}"
    print(f"重组后的用户提示词: \n{user_prompt}")
    print()

    # 发送请求获取推断结果
    response = client.chat.completions.create(
        model="outputs/qwen_full",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=2048,
        temperature=0.7,
    )

    reply = response.choices[0].message.content
    # 去除 <think>
    reply_cleaned = re.sub("<think>.*?</think>\\n*", "", reply, flags=re.DOTALL).strip()
    print(f"收到回复: \n{reply_cleaned}")
    print()


if __name__ == "__main__":
    queries = [
        "你是谁？",
        "今天热不热啊？",
        "如果我借书逾期了，罚款怎么算？每天多少钱？",
        "研讨室在几楼？",
        "当前有多少馆藏？",
        "怎么样借书？",
        "图书馆什么时候开门？",
        "怎么样预约研讨室？",
        "图书馆可以喝饮料吗？",
        "简单给我说一下图书馆有哪些规章制度。",
        "预约书找不到怎么办",
        "什么我的书籍超期了，没有收到邮件？",
    ]
    for q in queries:
        print(f"用户输入: \n{q}")
        print()
        chat_with_rag(q)
        print("=" * 100)
