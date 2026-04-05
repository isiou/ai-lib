from pydantic_settings import BaseSettings


# 系统全局配置中心 统一管理所有的环境变量及默认路径
class Settings(BaseSettings):
    VLLM_API_BASE: str = "http://127.0.0.1:8000/v1"
    MODEL_NAME: str = "outputs/qwen_full"
    EMB_MODEL_NAME: str = "BAAI/bge-small-zh-v1.5"
    RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-base"
    FAISS_INDEX_PATH: str = "data/rag/faiss_index.bin"
    CHUNKS_PATH: str = "data/rag/chunks.pkl"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8080
    DB_PATH: str = "data/library.db"
    USER_KNOWLEDGE_PATH: str = "data/knowledge_base/user_inputs.txt"


settings = Settings()
