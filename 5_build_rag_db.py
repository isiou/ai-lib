import os
import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def parse_knowledge_base(kb_dir):
    chunks = []
    if not os.path.exists(kb_dir):
        print("未识别到知识库目录")
        return chunks

    for filename in os.listdir(kb_dir):
        filepath = os.path.join(kb_dir, filename)
        if os.path.isfile(filepath):
            try:
                # json 格式解析
                if filename.endswith(".json"):
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            chunks.extend(
                                [
                                    str(item).strip()
                                    for item in data
                                    if str(item).strip()
                                ]
                            )
                        elif isinstance(data, dict):
                            for v in data.values():
                                if str(v).strip():
                                    chunks.append(str(v).strip())
                # txt 或 md 文件解析
                elif filename.endswith(".txt") or filename.endswith(".md"):
                    with open(filepath, "r", encoding="utf-8") as f:
                        lines = [line.strip() for line in f if line.strip()]
                        chunks.extend(lines)
                # pdf 文件解析
                elif filename.endswith(".pdf"):
                    try:
                        import fitz

                        with fitz.open(filepath) as doc:
                            for page in doc:
                                text = page.get_text()
                                if text:
                                    # 按换行符切分
                                    chunks.extend(
                                        [
                                            line.strip()
                                            for line in text.split("\n")
                                            if line.strip()
                                        ]
                                    )
                    except ImportError as e:
                        print(e)
                # docx 文件解析
                elif filename.endswith(".docx"):
                    try:
                        from docx import Document

                        doc = Document(filepath)
                        for para in doc.paragraphs:
                            if para.text.strip():
                                chunks.append(para.text.strip())
                    except ImportError as e:
                        print(e)
                else:
                    print(f"{filename} 为未知格式文件将跳过处理")
            except Exception as e:
                print(f"处理文件 {filename} 时发生错误: {e}")

    return chunks


def main():
    kb_dir = "data/knowledge_base"
    lines = parse_knowledge_base(kb_dir)

    if not lines:
        print("未从知识库目录中提取到任何内容")
        return

    # BAAI/bge-small-zh-v1.5 适用于中文表征
    model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

    print(f"计算知识库向量共 {len(lines)} 条片段...")

    # 按行和块计算向量
    embeddings = model.encode(lines, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    # 创建 FAISS 索引
    print("构建 FAISS 向量库...")
    dim = embeddings.shape[1]
    # embeddings 已经做过归一化故内积等价于余弦相似度
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
