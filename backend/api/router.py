from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from backend.schemas.chat import ChatRequest
from backend.services.chat import process_chat_stream
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/chat")
async def chat_endpoint(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty")

    try:
        return StreamingResponse(process_chat_stream(req), media_type="text/plain")
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
