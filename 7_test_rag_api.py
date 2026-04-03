import pickle
import numpy as np
import faiss
import re
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI

print("============ 正在初始化 RAG 与重排模块 ============")
try:
    # 加载模型
    emb_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    reranker_model = CrossEncoder("BAAI/bge-reranker-base")

    # 加载索引
    index = faiss.read_index("data/rag/faiss_index.bin")

    # 加载知识库
    with open("data/rag/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    print("RAG 知识库与模型加载成功")
except Exception as e:
    print("加载 RAG 时发生错误")
    print(e)
    exit(1)

# 创建连接对象
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="none")


def chat_with_rag(query, top_k=5, rerank_top_k=2):
    # 编码查询
    query_emb = emb_model.encode([query], normalize_embeddings=True).astype(np.float32)

    # 1. 向量检索初筛
    # 召回 top_k 个
    distances, indices = index.search(query_emb, top_k)

    retrieved_texts = []
    for idx in indices[0]:
        if idx != -1:
            retrieved_texts.append(chunks[idx])

    # 2. 重排序精排
    if retrieved_texts:
        cross_inp = [[query, text] for text in retrieved_texts]
        scores = reranker_model.predict(cross_inp)

        # 根据重排序分数进行降序排列
        doc_score_pairs = sorted(
            zip(retrieved_texts, scores), key=lambda x: x[1], reverse=True
        )

        # 打印分数用于调试
        print(f"[\033[94m重排序得分详情: \033[0m]")
        for doc, score in doc_score_pairs:
            print(f"  Score: {score:.4f} | {doc}")

        # 选取前 rerank_top_k 个片段，并且只保留 score > 0.5 的高置信度片段
        retrieved_texts = [
            doc for doc, score in doc_score_pairs[:rerank_top_k] if score > 0.5
        ]

    # 构造增强 Prompt
    context = "\n".join(retrieved_texts) if retrieved_texts else "无匹配的参考资料"
    print(f"\n[\033[92m归因后参考知识: \033[0m]\n{context}\n")

    system_prompt = """你的名字是厦小嘉，是厦门大学嘉庚学院图书馆的智能助手，用于帮助用户解决相关咨询问题等。
    规则一：如果【参考资料】中给出了与问题相关的明确信息，必须严格根据参考资料准确地回答。
    规则二：如果【参考资料】中无相关资料或无法解答该问题时，且问题是关于图书馆业务（如借书、预约、开馆时间等）时，请运用你作为图书馆助手的常识给出通用的解答。
    规则三：如果用户提出问题与图书馆业务无关时，请不要自作主张进行回答。
    规则四：涉及政治、意识形态、色情等敏感信息时，拒绝回答！"""

    user_prompt = f"""【参考资料】{context}\n【用户问题】{query}\n"""

    # 调用模型
    response = client.chat.completions.create(
        model="outputs/qwen_full",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        # 温度设为 0.7
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
        "图书馆什么时候开门？",
        "怎么样预约研讨室？",
    ]

    for q in queries:
        print("=" * 50)
        print(f"\033[93muser: {q}\033[0m")
        chat_with_rag(q)
