from modelscope import snapshot_download
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder

# 采用的模型配置
BASE_MODEL = "Qwen/Qwen3-0.6B"
EMB_MODEL = "BAAI/bge-small-zh-v1.5"
REK_MODEL = "BAAI/bge-reranker-base"


def main():
    base_res = snapshot_download(BASE_MODEL)

    emb_res = SentenceTransformer(EMB_MODEL)

    rek_res = CrossEncoder(REK_MODEL)

    if base_res and emb_res and rek_res:
        print("模型下载完成")


if __name__ == "__main__":
    main()
