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
        print("[\033[94m重排序得分详情: \033[0m]")
        for doc, score in doc_score_pairs:
            print(f"  Score: {score:.4f} | {doc}")

        # 选取前 rerank_top_k 个片段，并且只保留 score > 0.5 的高置信度片段
        retrieved_texts = [
            doc for doc, score in doc_score_pairs[:rerank_top_k] if score > 0.5
        ]

    # 构造增强 Prompt
    context = "\n".join(retrieved_texts) if retrieved_texts else "无匹配的参考资料"
    print(f"\n[\033[92m归因后参考资料: \033[0m]\n{context}\n")

    system_prompt = """# 角色设定
    你是“厦小嘉”，厦门大学嘉庚学院图书馆的专属智能助手。
    你性格温和、热情、乐于助人，语气总是充满亲和力，交流时就像一位图书馆里贴心、专业的管理员。
    你拥有强大的对话能力和常识，但你清楚自己的职责边界。

    # 行为准则
    - 交流风格：自然流畅，像人类客服一样对话，把参考资料里的干瘪文字转化为温暖的口语。绝不要机械地说“根据参考资料”这类机器人味很重的话。
    - 业务解答：当被问到具体的业务规定时，请以提供的【参考资料】作为唯一事实依据。如果参考资料为空，但问题属于借还书、阅读等图书馆常见场景，请运用你的专业常识给出友好的引导。
    - 坚守边界：你的专业领域仅限于“图书馆及阅读相关业务”。遇到闲聊、天气、医疗、编程等完全无关的问题时，请用你温柔的性格巧妙婉拒，主动把话题拉回图书馆。
    - 安全底线：拒绝回答任何违规、敏感问题。"""

    user_prompt = f"""【参考资料】\n{context}\n\n【用户问题】\n{query}\n/no_think"""

    # 调用模型
    response = client.chat.completions.create(
        model="outputs/qwen_full",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        # 平衡自然度
        temperature=0.5,
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
