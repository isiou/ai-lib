from modelscope import snapshot_download
from sentence_transformers import SentenceTransformer


def main():
    model_dir = snapshot_download("Qwen/Qwen3-0.6B")
    print("主对话模型加载成功")
    # 模型下载位置
    # print(model_dir)

    emb_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    print("向量化模型加载成功")


if __name__ == "__main__":
    main()
