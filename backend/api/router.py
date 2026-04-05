from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from backend.schemas.chat import ChatRequest, KnowledgeRequest
from backend.services.chat import process_chat_stream
from backend.core.dependencies import get_app_state
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


# 核心对话接口 接收用户提问并返回 RAG 强化后的流式响应
@router.post("/chat")
async def chat_endpoint(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty")
    try:
        return StreamingResponse(process_chat_stream(req), media_type="text/plain")
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# 知识录入接口 动态接受新的片段并编码持久化到向量库中
@router.post("/knowledge")
async def add_knowledge_endpoint(req: KnowledgeRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    state = get_app_state()
    try:
        state.add_knowledge(text)
        return {"status": "success", "message": "知识片段已成功添加至向量库"}
    except Exception as e:
        logger.error(f"Error adding knowledge: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
