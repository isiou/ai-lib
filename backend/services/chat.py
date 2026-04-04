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
    prompt = f"""# 角色设定
你是“厦小嘉”，厦门大学嘉庚学院图书馆的专属智能助手。
你性格温和、热情、乐于助人，语气总是充满亲和力。

# 行为准则
- 交流风格：自然流畅，不要机械地说“根据参考资料”。
- 业务解答：请以提供的【参考资料】以及工具调用的返回结果作为事实依据。当你通过工具获取到了书籍的索书号、位置或状态等信息时，请直接且完整地告知用户，绝对不要拒绝提供。
- 坚守边界：专业领域仅限于“图书馆及阅读相关业务”。
- 主动服务：当用户对某本书感兴趣或你推荐了书籍后，你可以主动询问用户是否需要查询该书的具体馆藏信息（索书号、位置等）。

# 工具使用规范
你具备以下查询工具，遇到对应问题时【必须无条件调用】，绝不能自己编造答案：
1. `get_current_time`: 遇到“现在几点”、“当前时间”等问题时，必须调用此工具。
2. `get_current_date`: 遇到“今天几号”、“今天星期几”等问题时，必须调用此工具。
3. `recommend_books`: 根据主题推荐书籍。传入参数 `topic`。
4. `query_book_info`: 查询特定书籍的馆藏信息（如：在哪、索书号是多少）。传入参数 `book_name`。

遇到需要工具解答的问题时，你必须立刻且仅输出如下格式来调用相应的工具：
<tool_call>
{{"name": "工具名称", "arguments": {{"参数名": "参数值"}}}}
</tool_call>

注意：如果需要调用工具，请直接输出纯 JSON 或 <tool_call> XML，不要附加任何其他思考或解释文字。如果问题不需要工具，再直接回答。

【参考资料】
{context}
"""
    return prompt


async def process_chat_stream(req: ChatRequest) -> AsyncGenerator[str, None]:
    state = get_app_state()
    user_query = req.messages[-1].content

    # --- RAG 检索阶段 ---
    logger.info(f"RAG Retrieval for query: {user_query}")

    # Assert models are loaded to satisfy type checker
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

    # 构造历史对话上下文
    messages_for_llm: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt}
    ]

    # 注入 Few-Shot 示例，以最标准的规范教会小模型怎么使用工具
    few_shot_examples = [
        {"role": "user", "content": "你好，请问现在几点了？"},
        {
            "role": "assistant",
            "content": '<tool_call>\n{"name": "get_current_time", "arguments": {}}\n</tool_call>',
        },
        {
            "role": "user",
            "content": '系统返回了工具调用的结果：\n{"current_time": "14:30:00"}\n\n请直接把上面的真实数据告诉用户！不要要求用户自行查询！',
        },
        {"role": "assistant", "content": "当前时间是下午两点半。"},
        {"role": "user", "content": "有什么推荐的科幻小说吗？"},
        {
            "role": "assistant",
            "content": '<tool_call>\n{"name": "recommend_books", "arguments": {"topic": "科幻小说"}}\n</tool_call>',
        },
        {
            "role": "user",
            "content": '系统返回了工具调用的结果：\n{"topic": "科幻小说", "recommended_books": [{"title": "三体", "author": "刘慈欣"}]}\n\n请直接把上面的真实数据告诉用户！不要要求用户自行查询！',
        },
        {"role": "assistant", "content": "为您推荐刘慈欣的《三体》。"},
        {"role": "user", "content": "这本书在哪？"},
        {
            "role": "assistant",
            "content": '<tool_call>\n{"name": "query_book_info", "arguments": {"book_name": "三体"}}\n</tool_call>',
        },
        {
            "role": "user",
            "content": '系统返回了工具调用的结果：\n{"book_name": "三体", "call_number": "I247.5/88", "location": "五楼文学阅览室", "status": "在馆可借"}\n\n请直接把上面的真实数据告诉用户！不要要求用户自行查询！',
        },
        {
            "role": "assistant",
            "content": "《三体》这本书在五楼文学阅览室，索书号是 I247.5/88，目前状态是在馆可借。",
        },
    ]
    messages_for_llm.extend(few_shot_examples)

    for i, msg in enumerate(req.messages):
        content = msg.content
        if i == len(req.messages) - 1 and msg.role == "user":
            content = f"{content} /no_think"
        messages_for_llm.append({"role": msg.role, "content": content})

    # LLM 第一轮推断 (非流式，以捕获潜在的工具调用)
    response = await state.client.chat.completions.create(
        model=settings.MODEL_NAME,
        messages=messages_for_llm,
        tools=TOOLS,
        tool_choice="auto",
        temperature=0.1,
        stream=False,
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    content = response_message.content or ""

    if not tool_calls:
        tool_calls = parse_manual_tool_calls(content)

    if tool_calls:
        logger.info("模型决定调用工具")

        # 为了兼容 openai client 类型，转换为字典格式存储到 message 历史中
        if not response_message.tool_calls:
            messages_for_llm.append({"role": "assistant", "content": content})
        else:
            messages_for_llm.append(response_message)  # type: ignore

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
                function_response = func(**args) if args else func()  # type: ignore

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
                    # 对于手动解析的工具调用，直接将工具结果作为用户新一轮的输入
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

        if getattr(response_message, "tool_calls", None):
            # 强力注入用户指令，防止模型出现对齐拒答（幻觉），针对标准 tool_calls 流程
            messages_for_llm.append(
                {
                    "role": "user",
                    "content": "强制要求：刚才工具已返回了真实数据。你必须直接将查询到的数据告诉用户，绝不允许回答“无法提供”、“需自行查询”或类似拒绝话术！",
                }
            )

        # LLM 第二轮推断 (流式输出最终结果)
        second_response = await state.client.chat.completions.create(
            model=settings.MODEL_NAME,
            messages=messages_for_llm,
            temperature=0.5,
            stream=True,
        )

        async for chunk in second_response:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield delta
    else:
        # 伪造流式输出，给用户平滑的打字机体验
        chunk_size = 4
        for i in range(0, len(content), chunk_size):
            yield content[i : i + chunk_size]
            await asyncio.sleep(0.02)
