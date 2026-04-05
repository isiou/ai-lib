import json
import logging
import numpy as np
import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator
from backend.core.config import settings
from backend.core.dependencies import get_app_state
from backend.schemas.chat import ChatRequest
from backend.services.tools import AVAILABLE_FUNCTIONS, TOOLS, parse_manual_tool_calls

logger = logging.getLogger(__name__)


def construct_system_prompt(retrieved_texts: Optional[List[str]] = None) -> str:
    context = "\n".join(retrieved_texts) if retrieved_texts else "无匹配的参考资料"
    prompt = f"# 角色设定\n你是“厦小嘉”，厦门大学嘉庚学院图书馆的专属智能助手。\n你性格温和、热情、乐于助人，语气总是充满亲和力。\n\n# 行为准则\n- 交流风格：自然流畅，不要机械地说“根据参考资料”。\n- 业务解答：请以提供的【参考资料】以及工具调用的返回结果作为事实依据。当你通过工具获取到了书籍的索书号、位置或状态等信息时，请直接且完整地告知用户，绝对不要拒绝提供。\n- 坚守边界：专业领域仅限于“图书馆及阅读相关业务”。\n- 主动服务：当用户对某本书感兴趣或你推荐了书籍后，你可以主动询问用户是否需要查询该书的具体馆藏信息（索书号、位置等）。\n\n【参考资料】\n{context}\n"
    return prompt


async def process_chat_stream(req: ChatRequest) -> AsyncGenerator[str, None]:
    state = get_app_state()
    user_query = req.messages[-1].content
    logger.info(f"RAG Retrieval for query: {user_query}")

    assert state.emb_model is not None, "Embedding model not initialized"
    assert state.index is not None, "FAISS index not initialized"
    assert state.reranker_model is not None, "Reranker model not initialized"
    assert state.client is not None, "OpenAI client not initialized"

    query_emb = state.emb_model.encode([user_query], normalize_embeddings=True).astype(
        np.float32
    )

    distances, indices = state.index.search(query_emb, req.top_k)
    retrieved_texts = []
    for idx in indices[0]:
        if idx != -1:
            retrieved_texts.append(state.chunks[idx])

    if retrieved_texts:
        cross_inp = [[user_query, text] for text in retrieved_texts]
        scores = state.reranker_model.predict(cross_inp)
        doc_score_pairs = sorted(
            zip(retrieved_texts, scores), key=lambda x: x[1], reverse=True
        )
        retrieved_texts = [
            doc for doc, score in doc_score_pairs[: req.rerank_top_k] if score > 0.5
        ]
        logger.info(f"Retrieved {len(retrieved_texts)} chunks after reranking.")

    system_prompt = construct_system_prompt(retrieved_texts)
    messages_for_llm: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt}
    ]

    for msg in req.messages:
        messages_for_llm.append({"role": msg.role, "content": msg.content})

    # 第一阶段：分析并决定是否需要调用工具
    response = await state.client.chat.completions.create(
        model=settings.MODEL_NAME,
        messages=messages_for_llm,  # type: ignore
        tools=TOOLS,
        tool_choice="auto",
        temperature=0.7,
        stream=False,
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    content = response_message.content or ""

    if not tool_calls:
        tool_calls = parse_manual_tool_calls(content)

    if tool_calls:
        logger.info("模型决定调用工具")

        # 将助手的工具调用请求原样追加到上下文
        if getattr(response_message, "tool_calls", None):
            messages_for_llm.append(response_message.model_dump(exclude_none=True))
        else:
            messages_for_llm.append({"role": "assistant", "content": content})

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            args = {}
            if tool_call.function.arguments:
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    pass
            logger.info(f"Executing function: {function_name}({args})")

            if function_name in AVAILABLE_FUNCTIONS:
                func = AVAILABLE_FUNCTIONS[function_name]
                function_response = func(**args) if args else func()
                tool_msg_content = json.dumps(function_response, ensure_ascii=False)

                if getattr(response_message, "tool_calls", None):
                    messages_for_llm.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": tool_msg_content,
                        }
                    )
                else:
                    messages_for_llm.append(
                        {
                            "role": "user",
                            "content": f"系统返回了工具调用的结果：\n{tool_msg_content}\n\n请直接把上面的真实数据告诉用户！不要要求用户自行查询！",
                        }
                    )
            else:
                logger.error(f"Function {function_name} not found")
                if getattr(response_message, "tool_calls", None):
                    messages_for_llm.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": json.dumps(
                                {"error": f"Function {function_name} not found"}
                            ),
                        }
                    )
                else:
                    messages_for_llm.append(
                        {
                            "role": "user",
                            "content": f"系统返回：工具 {function_name} 不存在，请向用户道歉并直接回答。",
                        }
                    )

        if not getattr(response_message, "tool_calls", None):
            messages_for_llm.append(
                {
                    "role": "user",
                    "content": "强制要求：刚才工具已返回了真实数据。你必须直接将查询到的数据告诉用户，绝不允许回答“无法提供”、“需自行查询”或类似拒绝话术！",
                }
            )

        # 第二阶段：工具调用完毕，将结果交给大模型，获取最终回复并流式返回给用户
        second_response: Any = await state.client.chat.completions.create(
            model=settings.MODEL_NAME,
            messages=messages_for_llm,  # type: ignore
            temperature=0.7,
            stream=True,
        )
        async for chunk in second_response:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield delta
    else:
        # 如果不需要调用工具，则直接返回第一阶段生成的文本
        chunk_size = 4
        for i in range(0, len(content), chunk_size):
            yield content[i : i + chunk_size]
            await asyncio.sleep(0.02)
