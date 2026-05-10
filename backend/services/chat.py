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

# ── 意图预判 ──────────────────────────────────────────────────────────────────

# 各工具对应的触发关键词
TOOL_TRIGGER_KEYWORDS = {
    "get_current_time": ["几点", "时间", "现在是", "当前时间", "什么时候了"],
    "get_current_date": ["几号", "日期", "今天是", "当前日期", "星期", "几月"],
    "recommend_books": ["推荐", "有什么书", "书籍", "书单", "想学", "看什么"],
    "query_book_info": ["在哪", "位置", "哪里", "查一下", "馆藏", "索书号", "借阅"],
}


def detect_required_tool(query: str):
    """
    根据关键词预判用户意图，返回对应的 tool_choice 配置。
    若命中关键词则强制指定工具（required），否则交由模型自行决策（auto）。
    """
    for tool_name, keywords in TOOL_TRIGGER_KEYWORDS.items():
        if any(kw in query for kw in keywords):
            logger.info(f"意图预判命中: 强制调用工具 [{tool_name}]")
            return {"type": "function", "function": {"name": tool_name}}
    return "auto"


# ── 系统提示词构造 ────────────────────────────────────────────────────────────


def construct_system_prompt(retrieved_texts: Optional[List[str]] = None) -> str:
    context = "\n".join(retrieved_texts) if retrieved_texts else "无匹配的参考资料"
    prompt = (
        f"# 角色设定\n"
        f"你是“厦小嘉”，厦门大学嘉庚学院图书馆的专属智能助手。\n"
        f"你性格温和、热情、乐于助人，语气总是充满亲和力。\n\n"
        f"# 核心指令\n"
        f"1. 业务解答：你必须且只能使用提供的【参考资料】或工具返回的结果来回答用户的问题。\n"
        f"2. 严禁编造：如果【参考资料】中没有包含答案，请直接致歉并表示不知情，绝对不要自己脑补、猜测或编造任何理由！\n"
        f"3. 提取事实：对于参考资料中包含的具体步骤（如超期处理、找不到书、预约等），请直接、准确地将参考资料里的规定告诉用户，不要画蛇添足地添加未提及的前提或解释。\n"
        f"4. 交流风格：自然流畅，直接回答结果，绝对不要说“根据参考资料”这类机械前缀。\n\n"
        f"【参考资料】\n"
        f"{context}\n"
    )
    return prompt


# ── 核心对话与 RAG 流程 ──────────────────────────────────────────────────────


async def process_chat_stream(req: ChatRequest) -> AsyncGenerator[str, None]:
    state = get_app_state()
    user_query = req.messages[-1].content
    logger.info(f"RAG Retrieval for query: {user_query}")

    assert state.emb_model is not None, "Embedding model not initialized"
    assert state.index is not None, "FAISS index not initialized"
    assert state.reranker_model is not None, "Reranker model not initialized"
    assert state.client is not None, "OpenAI client not initialized"

    # 执行向量查询与相关度排序过滤
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

    strong_reminder = "\n\n(系统强制指令：1. 若问题涉及查询时间、推荐或查询书籍馆藏，请务必调用工具；2. 若属于图书馆业务解答，你必须完全按照【参考资料】中提供的方法回复，绝不允许自行编造任何多余的解释或前提条件！)"

    # 聚合历史上下文及强制工具指令
    for msg in req.messages[:-1]:
        messages_for_llm.append({"role": msg.role, "content": msg.content})

    last_message = req.messages[-1].content
    messages_for_llm.append(
        {"role": "user", "content": f"{last_message}{strong_reminder} /nothink"}
    )

    tool_choice = detect_required_tool(user_query)
    logger.info(f"tool_choice determined as: {tool_choice}")

    # 第一阶段：分析并决定是否需要调用工具
    response = await state.client.chat.completions.create(
        model=settings.MODEL_NAME,
        messages=messages_for_llm,  # type: ignore
        tools=TOOLS,
        tool_choice=tool_choice,
        temperature=0.1,  # Lower temperature for tool calling stage
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

            # 本地工具执行与上下文挂载
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
