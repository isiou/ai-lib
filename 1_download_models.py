from modelscope import snapshot_download
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder


def main():
    # 对话模型
    model_dir = snapshot_download("Qwen/Qwen3-0.6B")
    if model_dir:
        print(model_dir)

    # 向量化模型
    emb_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    if emb_model:
        print(emb_model)

    # 重排序模型
    reranker_model = CrossEncoder("BAAI/bge-reranker-base")
    if reranker_model:
        print(reranker_model)

    print("模型下载完成")


if __name__ == "__main__":
    main()
