import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def main():
    kb_path = "data/knowledge.txt"
    if not os.path.exists(kb_path):
        print("未识别到知识库文件: ", {kb_path})
        return

    # 分割文件行
    with open(kb_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    # BAAI/bge-small-zh-v1.5 适用于中文表征
    model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

    print("计算知识库向量...")
    # 按行计算向量
    embeddings = model.encode(lines, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    # 创建 FAISS 索引
    print("构建 FAISS 向量库...")
    dim = embeddings.shape[1]
    # 因为 embeddings 已经做过归一化故内积等价于余弦相似度
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # 保存向量库与原始文本
    os.makedirs("data/rag", exist_ok=True)
    faiss.write_index(index, "data/rag/faiss_index.bin")

    with open("data/rag/chunks.pkl", "wb") as f:
        pickle.dump(lines, f)

    print("RAG 向量库构建完成")
    print(f"共存入 {len(lines)} 条知识片段")


if __name__ == "__main__":
    main()
