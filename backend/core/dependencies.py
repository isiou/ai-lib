import pickle
import faiss
import logging
from typing import Any, Optional, Type
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import AsyncOpenAI
from backend.core.config import settings

logger = logging.getLogger(__name__)


class AppState:
    emb_model: Optional[SentenceTransformer] = None
    reranker_model: Optional[CrossEncoder] = None
    index: Any = None
    chunks: Any = None
    client: Optional[AsyncOpenAI] = None

    @classmethod
    def initialize(cls):
        logger.info("Initializing models and data...")
        cls.emb_model = SentenceTransformer(settings.EMB_MODEL_NAME)
        cls.reranker_model = CrossEncoder(settings.RERANKER_MODEL_NAME)
        cls.index = faiss.read_index(settings.FAISS_INDEX_PATH)
        with open(settings.CHUNKS_PATH, "rb") as f:
            cls.chunks = pickle.load(f)
        cls.client = AsyncOpenAI(base_url=settings.VLLM_API_BASE, api_key="none")
        logger.info("Models initialized successfully.")


def get_app_state() -> Type[AppState]:
    return AppState
