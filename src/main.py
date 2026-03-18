"""
系统主入口模块 (Main)

整个 RAG + ReAct Agent 多轮对话系统系统的核心入口，
负责初始化所有依赖服务、加载配置、实例化各功能模块，并对外提供统一的异步请求处理函数 handle_request_async，
同时为命令行测试提供同步包装器 handle_request
如需 api 测试请直接运行 api.py (根目录)

主要功能：
    1. 环境初始化：加载 .env 配置文件，初始化 LangSmith 监控。
    2. 服务连接：建立 Redis、Milvus、Neo4j、Ollama 等外部服务的连接。
    3. 模块实例化：按顺序创建查询改写、高频缓存、短期记忆、长期记忆、检索工具、验证工具、翻译工具及 ReAct Agent 实例。
    4. 异步请求处理：定义 handle_request_async，实现完整的对话处理流程：
     - 查询改写 → 高频缓存检查 → 短期记忆加载 → 长期记忆预取 → 消息构造 → Agent 调用 → 异步长期记忆存储 → 缓存和短期记忆更新。
    5. 后台任务管理：通过全局 background_tasks集合追踪异步长期记忆存储任务，确保数据持久化。
    7. 测试入口：当直接运行该文件时，执行一次测试对话，验证系统功能。

架构亮点：
    - 模块化组装：所有核心模块通过依赖注入方式组装，职责清晰，易于替换和测试。
    - 异步非阻塞：利用 asyncio实现高并发处理，长期记忆存储等耗时操作异步化，不阻塞主请求响应。
    - 后台任务追踪：通过全局集合管理异步任务，确保服务关闭前任务完成，防止数据丢失。
    - 分层日志：集成 LangSmith 进行链路追踪，结合本地 logger 输出结构化日志到控制台，便于调试和监控。
    - 配置分离：所有敏感信息和可变参数通过环境变量和配置文件管理，符合 12-factor 原则。
    - 错误处理：关键服务连接失败时主动抛出异常，避免系统在不可用状态下运行。

Author: Ke Meng
Created: 2026-03-17
Last Modified: 2026-03-18
"""

import redis
import asyncio
from pymilvus import Collection, connections
from langchain_milvus import Milvus
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


import logging

logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为 INFO，这样所有 INFO 及以上日志都会显示
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 添加当前模块的 logger
logger = logging.getLogger(__name__)

# 配置文件导入
from config.config_path import (
    REDIS_CONFIG,
    OLLAMA_CONFIG,
    MILVUS_CONFIG,
    MILVUS_COLLECTION,
    API_CONFIG,
    NEO4J_CONFIG,
    RETRIEVAL_CONFIG,
    TRANSLATOR_CONFIG,
    CACHE_CONFIG,
    MEMORY_CONFIG,
    REWRITER_CONFIG,
    RERANKER_CONFIG,
    LONG_TERM_MEMORY_CONFIG,
)
# 前置模块导入
from src.core.ReAct_Agent.tools.agent import create_react_agent
from src.core.query_rewrite.query_rewriter import QueryRewriter
from src.core.high_frequency_query_cache.redis_bloom import HighFreqCache
from src.core.memory_short.redis_short_memory import ShortTermMemory
from src.core.memory_long.long_term_memory import LongTermMemory

# 工具集导入
from src.core.ReAct_Agent.tools.retriever import RetrieverTool, BGEReranker
from src.core.ReAct_Agent.tools.relation_verifier import create_relation_verifier_tool
from src.core.ReAct_Agent.tools.translate import create_translator_tool

# langsmith监控导入
import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Lang验证
api_key = os.getenv("LANGCHAIN_API_KEY")
if api_key:
    print(f"✅ LangSmith API Key 已加载: {api_key[:5]}...")
else:
    print("⚠️ 未找到 LangSmith API Key，请检查 .env 文件")


# 在 main.py 顶部添加
background_tasks = set()

# ========== 1. 初始化 Redis 客户端 ==========
redis_client = redis.Redis(**REDIS_CONFIG)
try:
    redis_client.ping()
    print("✅ Redis 连接成功")
except redis.ConnectionError:
    raise RuntimeError("❌ Redis 连接失败")

# ========== 2. 初始化嵌入模型 ==========
embeddings = OllamaEmbeddings(
    model=OLLAMA_CONFIG['embed_model'],
    base_url=OLLAMA_CONFIG['base_url']
)

# ========== 3. 获取 pymilvus Collection 实例 ==========
connections.connect(**MILVUS_CONFIG)
milvus_collection = Collection(MILVUS_COLLECTION)


# ========== 初始化重排器 ==========
reranker = BGEReranker(
    model_path=RERANKER_CONFIG['model_path'],
    device=RERANKER_CONFIG['device']
)

# ========== 4. 初始化 LLM ==========
llm = ChatOpenAI(
    model=API_CONFIG['model'],
    openai_api_key=API_CONFIG['api_key'],
    openai_api_base=API_CONFIG['base_url'],
    temperature=API_CONFIG['temperature'],
    streaming=False,
)

# ========== 5. 创建检索工具（移除 vectorstore）==========
retriever = RetrieverTool(
    embedding_model=embeddings,          # 嵌入模型
    milvus_collection=milvus_collection,  # 用于向量检索和 BM25 的 Milvus 集合
    reranker=reranker,
    vector_top_k=RETRIEVAL_CONFIG['vector_top_k'],
    bm25_top_k=RETRIEVAL_CONFIG['bm25_top_k'],
    rrf_final_top_k=RETRIEVAL_CONFIG['rrf_final_top_k'],
    absolute_low=RETRIEVAL_CONFIG.get('absolute_low', -5.0),  # 绝对下限阈值，低于此值返回空结果
    vector_search_params={"metric_type": "COSINE", "params": {"nprobe": 128}}
)
retriever_tool = retriever.as_tool()

# ========== 6. 创建验证工具 ==========
verifier_tool = create_relation_verifier_tool(
    neo4j_url=NEO4J_CONFIG['url'],
    neo4j_user=NEO4J_CONFIG['user'],
    neo4j_password=NEO4J_CONFIG['password'],
    default_min_confidence=0.5,
)

# ========== 7. 创建翻译工具 ==========
translator_tool = create_translator_tool(
    model_path=TRANSLATOR_CONFIG['model_path'],
    device=TRANSLATOR_CONFIG['device']
)

# ========== 8. 初始化长期记忆相关组件 ==========
# 8.1 记忆管理器专用 LLM（用小模型）
manager_llm = ChatOllama(
    model=LONG_TERM_MEMORY_CONFIG['manager_model'],
    base_url=LONG_TERM_MEMORY_CONFIG.get('manager_base_url', OLLAMA_CONFIG['base_url']),
    temperature=0.3,
    max_tokens=512,
)


# 新代码：
long_term_memory = LongTermMemory(
    milvus_uri=MILVUS_CONFIG['uri'],
    collection_name=LONG_TERM_MEMORY_CONFIG['collection_name'],
    vector_dim=LONG_TERM_MEMORY_CONFIG['vector_dim'],
    embed_model=OLLAMA_CONFIG['embed_model'],               # "nomic-embed-text"
    llm_model=LONG_TERM_MEMORY_CONFIG['manager_model'],     # "qwen3-4b-rewrite"
    ollama_base_url=OLLAMA_CONFIG['base_url'],               # Ollama 地址
    top_k=LONG_TERM_MEMORY_CONFIG['top_k'],
)

# 8.4 获取记忆工具并加入工具列表
memory_tools = long_term_memory.get_tools()
tools = [retriever_tool, verifier_tool, translator_tool] + memory_tools

# ========== 9. 创建 Agent（传入 store）==========
agent_app = create_react_agent(
    llm=llm,
    tools=tools,
)

# ========== 10. 初始化前置模块 ==========
rewriter = QueryRewriter(
    redis_client=redis_client,
    max_history_turns=REWRITER_CONFIG['max_history_turns'],
    ollama_base_url=OLLAMA_CONFIG['base_url'],
    ollama_model=OLLAMA_CONFIG['model'],
    rewrite_timeout=REWRITER_CONFIG['timeout'],
    max_retries=REWRITER_CONFIG['max_retries'],
    enable_warmup=True
)

cache = HighFreqCache(
    redis_client=redis_client,
    threshold=CACHE_CONFIG['threshold'],
    window_seconds=CACHE_CONFIG['window_seconds'],
    bloom_capacity=CACHE_CONFIG['bloom_capacity'],
    bloom_error_rate=CACHE_CONFIG['bloom_error_rate']
)

# ========== 11. 初始化短期记忆管理器 ==========
memory = ShortTermMemory(
    redis_client=redis_client,
    max_history_turns=MEMORY_CONFIG['max_history_turns'],
    expire_seconds=MEMORY_CONFIG['expire_seconds']
)


# ========== 12. 异步请求处理函数 ==========
async def handle_request_async(user_input: str, session_id: str, user_id: str = "default_user") -> str:
    """
    处理用户请求的异步版本（支持长期记忆）
    """

    logger.info(f"🚀 开始处理请求: user_input='{user_input}', session_id='{session_id}', user_id='{user_id}'")

    # 1. 查询改写（返回中英文双版本）
    rewritten_dict = rewriter.rewrite(user_input, session_id)
    en_query = rewritten_dict["en_query"]
    zh_query = rewritten_dict["zh_query"]
    logger.info(f"✍️ 查询改写结果: en='{en_query}', zh='{zh_query}'")

    # 2. 检查高频缓存
    cached = cache.get(en_query)
    if cached is not None:
        logger.info(f"🎯 缓存命中，直接返回缓存答案")
        memory.add_turn(session_id, user_input, cached)
        return cached
    else:
        logger.info("❌ 缓存未命中")

    # 3. 加载短期记忆
    short_term_msgs = memory.get_messages(session_id)  # 现在有了这个方法
    logger.info(f"📚 加载短期记忆: 共 {len(short_term_msgs)} 条消息")

    # 4. 预取长期记忆（使用中文查询）
    memories = await long_term_memory.retrieve_relevant(
        user_id=user_id,
        query=zh_query,
        memory_type=None,
    )
    logger.info(f"🧠 加载长期记忆: 共 {len(memories)} 条相关记忆")

    # 5. 构造初始消息
    initial_messages = short_term_msgs[:]
    if memories:
        mem_text = "；".join([m["content"] for m in memories])
        initial_messages.append(SystemMessage(content=f"用户长期记忆：{mem_text}"))
    initial_messages.append(HumanMessage(content=en_query))

    # 6. 调用 Agent
    logger.info("🤖 开始调用 Agent...")
    result = await agent_app.ainvoke({
        "messages": initial_messages,
        "step_count": 0,
        "max_steps": 5
    })
    final_answer = result.get("final_answer", "抱歉，我无法回答这个问题。")
    logger.info(f"✅ Agent 调用完成，最终答案: {final_answer[:200]}...")

    # 7. 异步处理长期记忆
    task = asyncio.create_task(
        long_term_memory.process_and_store(
            user_id=user_id,
            user_input=user_input,
            assistant_output=final_answer,
        )
    )
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)
    logger.debug("长期记忆异步处理任务已提交")

    # 8. 更新缓存和短期记忆
    cache.update(en_query, final_answer)
    memory.add_turn(session_id, user_input, final_answer)
    logger.info(f"💾 缓存和短期记忆已更新")
    return final_answer


# ========== 13. 同步包装器（支持后台任务） ==========
def handle_request(user_input: str, session_id: str, user_id: str = "default_user") -> str:
    # 创建新的事件循环
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        # 运行主异步处理函数
        result = loop.run_until_complete(
            handle_request_async(user_input, session_id, user_id)
        )
        # 等待所有后台任务完成（最多5秒）
        if background_tasks:
            # 等待所有任务，超时5秒
            done, pending = loop.run_until_complete(
                asyncio.wait(background_tasks, timeout=10)
            )
            if pending:
                logger.warning(f"有 {len(pending)} 个后台任务未在5秒内完成")
        return result
    finally:
        loop.close()


# ========== 14. 测试运行 ==========
if __name__ == "__main__":
    test_query = "有什么好看的科幻电视剧推荐吗？"
    test_session = "test_session_123"
    test_user = "test_user"
    answer = handle_request(test_query, test_session, test_user)
    logger.info(f"最终答案: {answer}")
