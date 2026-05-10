import os
import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# 测试知识目录
kb_dir = "data/knowledge_base"


# 解析知识库数据
def parse_knowledge_base(kb_dir):
    chunks = []
    if not os.path.exists(kb_dir):
        return chunks

    for filename in os.listdir(kb_dir):
        filepath = os.path.join(kb_dir, filename)
        if os.path.isfile(filepath):
            try:
                # .json
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
                # .md/.txt
                elif filename.endswith(".txt") or filename.endswith(".md"):
                    with open(filepath, "r", encoding="utf-8") as f:
                        lines = [line.strip() for line in f if line.strip()]
                        chunks.extend(lines)
                # .pdf
                elif filename.endswith(".pdf"):
                    try:
                        import fitz
                        import re

                        with fitz.open(filepath) as doc:
                            full_text = ""
                            for page in doc:
                                full_text += page.get_text() + "\n"

                            lines = full_text.split("\n")
                            current_chunk = ""
                            for line in lines:
                                line = line.strip()
                                if not line:
                                    continue

                                is_new_item = re.match(
                                    r"^(\d+\.|[一二三四五六七八九十]、)", line
                                )
                                if is_new_item and current_chunk:
                                    chunks.append(current_chunk)
                                    current_chunk = line
                                elif current_chunk:
                                    if (
                                        len(current_chunk) > 400
                                        and current_chunk[-1] in "。！？.!?"
                                    ):
                                        chunks.append(current_chunk)
                                        current_chunk = line
                                    else:
                                        current_chunk += line
                                else:
                                    current_chunk = line
                            if current_chunk:
                                chunks.append(current_chunk)
                    except ImportError as e:
                        print(e)
                # .docx
                elif filename.endswith(".docx"):
                    try:
                        from docx import Document

                        doc = Document(filepath)
                        for para in doc.paragraphs:
                            if para.text.strip():
                                chunks.append(para.text.strip())
                    except ImportError as e:
                        print(e)
            except Exception as e:
                print(e)
    return chunks


# 构建 FAISS 索引
def main():
    lines = parse_knowledge_base(kb_dir)

    if not lines:
        return
    else:
        print(f"知识库共 {len(lines)} 条片段")

    # 调用嵌入向量打分
    model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

    embeddings = model.encode(lines, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    dim = embeddings.shape[1]

    # 建立内积或 L2 暴力全空间遍历树
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # 本地化保存
    os.makedirs("data/rag", exist_ok=True)
    faiss.write_index(index, "data/rag/faiss_index.bin")
    with open("data/rag/chunks.pkl", "wb") as f:
        pickle.dump(lines, f)

    print("RAG 向量库构建完成")


if __name__ == "__main__":
    main()
