from pydantic import BaseModel
from typing import List


# 消息体数据模型定义
class Message(BaseModel):
    role: str
    content: str


# 对话请求参数统一结构
class ChatRequest(BaseModel):
    messages: List[Message]
    top_k: int = 5
    rerank_top_k: int = 2


# 对话响应体定义
class ChatResponse(BaseModel):
    reply: str


# 知识录入请求统一结构
class KnowledgeRequest(BaseModel):
    text: str
