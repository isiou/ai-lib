from pydantic import BaseModel
from typing import List


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    top_k: int = 5
    rerank_top_k: int = 2


class ChatResponse(BaseModel):
    reply: str
