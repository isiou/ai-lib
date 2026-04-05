import os
import pickle
import faiss
import logging
import numpy as np
from typing import Any, Optional
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import AsyncOpenAI
from backend.core.config import settings

logger = logging.getLogger(__name__)


# 承载后端生命周期内全局依赖对象的核心状态类
class AppState:
    def __init__(self):
        self.emb_model: Optional[SentenceTransformer] = None
        self.reranker_model: Optional[CrossEncoder] = None
        self.index: Any = None
        self.chunks: Any = None
        self.client: Optional[AsyncOpenAI] = None

    # 初始化嵌入模型 重排序模型 向量库索引及客户端连接
    def initialize(self):
        logger.info("Initializing models and data...")
        self.emb_model = SentenceTransformer(settings.EMB_MODEL_NAME)
        self.reranker_model = CrossEncoder(settings.RERANKER_MODEL_NAME)
        self.index = faiss.read_index(settings.FAISS_INDEX_PATH)
        with open(settings.CHUNKS_PATH, "rb") as f:
            self.chunks = pickle.load(f)
        self.client = AsyncOpenAI(base_url=settings.VLLM_API_BASE, api_key="none")
        logger.info("Models initialized successfully.")

    # 提供向量化与落盘一条龙的知识录入方法
    def add_knowledge(self, text: str):
        if not self.emb_model or not self.index or self.chunks is None:
            raise RuntimeError("Models and data are not fully initialized.")

        # 先编码后压库 持久化二进制索引与反序列化块数据
        embedding = self.emb_model.encode([text], normalize_embeddings=True)
        embedding = np.array(embedding, dtype=np.float32)
        self.index.add(embedding)
        self.chunks.append(text)
        faiss.write_index(self.index, settings.FAISS_INDEX_PATH)
        with open(settings.CHUNKS_PATH, "wb") as f:
            pickle.dump(self.chunks, f)

        # 并行写入原文档日志以便追溯审核
        os.makedirs(os.path.dirname(settings.USER_KNOWLEDGE_PATH), exist_ok=True)
        with open(settings.USER_KNOWLEDGE_PATH, "a", encoding="utf-8") as f:
            f.write(text + "\n\n")


# 提供全局唯一的共享单例
app_state = AppState()


# 通过依赖注入风格暴露单例状态入口
def get_app_state() -> AppState:
    return app_state
