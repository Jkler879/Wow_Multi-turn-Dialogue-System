"""
长期记忆模块 (Long Term Memory)
- ReAct Agent 的前置模块，负责用户长期记忆的存储和检索。

该模块负责长期记忆注入和存储，全程异步执行，不阻塞Agent主程序任务，基于 Mem0 框架 + Milvus 向量数据库实现
1. 长期记忆注入阶段：
    - 根据用户输入的 user_id 进入 Milvus 专用长期记忆集合检索相关记忆, 与知识库集合逻辑分离
    - 返回与用户输入语义最相关的Top_K（上限3条）,Mem0 内部已经通过混合排序（语义+重要性+时效性）确保了返回结果的整体质量
    - 检索到的 Top_K 注入上下文供 Agent 使用。

2. 长期记忆存储阶段：
    - 在生成最终回复后异步执行，通过本地模型 Qwen3-1.7b 判断对话中是否包含需要存储进长期记忆的内容
    - 如包含，通过 Prompt 引导 LLM 生成结构化输出（包含记忆本身、记忆类型、重要性标签等元数据）
    - 通过本地嵌入模型 nomic-embed-text（知识库向量化、用户输入向量化相同模型）向量化后存入 Milvus 长期记忆集合
    - Mem0框架自带记忆去重逻辑，避免存储完全一致的长期记忆

功能：
    - 异步接口: 整个长期记忆模块皆异步接口通过 fastapi 执行，不阻塞主程序 Agent 任务
    - 结构化输出: 通过 Prompt 引导 LLM 生成结构化输出，确保记忆包含记忆类型 (preference/fact/episodic)和重要性 (0.0-1.0) 标签, 便于后续检索时按类型过滤或按重要性排序。
    - LLM输出解析: 检索时自动解析 LLM 输出标签，返回结构化的元数据。
    - 容错机制: 启动时确保集合加载，运行时每次操作前也检查加载状态，避免 Milvus 集合未加载错误。

依赖：
    - pymilvus: Milvus 向量数据库
    - mem0: 长期记忆管理框架
    - ollama: 本地 LLM 服务（Qwen3-1.7b, nomic-embed-text）

Author: Ke Meng
Created: 2026-03-07
Version: 1.1.0
Last Modified: 2026-03-18

变更记录：
    - 1.1.0 (2026-03-18):
                        - 改动1、通过自定义 prompt 引导模型输出包含记忆类型、重要性评分等元数据，后续查询支持按类型过滤
                        - 改动2、Qwen3-4b 降级为 Qwen3-1.7b 模型，足够支撑长期记忆的提取
                        - 改动3、整个长期记忆模块通过fastapi完全异步执行，函数异步化

    - 1.0.1 (2026-03-17):
                        - 升级为 Mem0 框架完全替换 Langmem，代码重构（Langmem框架过于demo，无法支撑生产级长期记忆）
    - 1.0.0 (2026-03-07):
                        - 初始版本，基于 Langmem + Milvus 实现。
"""

import asyncio
import logging
import re
from typing import List, Optional, Dict, Any
from pymilvus import Collection, utility
from mem0 import Memory

logger = logging.getLogger(__name__)


class LongTermMemory:
    """
    生产级长期记忆管理器（Mem0 实现）。

    集合需提前用 `long_memory_collection.py` 创建，运行时只负责使用。

    Attributes:
        top_k (int): 默认检索返回的记忆条数。
        collection_name (str): Milvus 集合名称。
        memory (Memory): Mem0 客户端实例。
    """

    def __init__(
        self,
        milvus_uri: str,
        collection_name: str,
        vector_dim: int,
        embed_model: str,
        llm_model: str,
        ollama_base_url: str,
        top_k: int = 5,
    ):
        self.top_k = top_k
        self.collection_name = collection_name

        # 自定义事实提取 Prompt：要求 LLM 在记忆文本末尾添加类型和重要性标签
        custom_extraction_prompt = """
        你是一个个人信息组织者，负责从对话中提取重要信息。你的输出必须是一个 JSON 对象，包含一个 "facts" 键，其值是一个**字符串列表**。
        每个字符串代表一条记忆，用简洁的一句话表达，并且**必须在末尾附加 [类型: preference/fact/episodic] 和 [重要性: 0.0-1.0] 标签**。

        例如：
        输入：我喜欢看科幻电影。
        输出：{"facts": ["用户喜欢看科幻电影。 [类型: preference] [重要性: 0.8]"]}

        输入：我的生日是5月20日。
        输出：{"facts": ["用户的生日是5月20日。 [类型: fact] [重要性: 0.9]"]}

        输入：昨天我和朋友去了故宫。
        输出：{"facts": ["用户昨天和朋友去了故宫。 [类型: episodic] [重要性: 0.6]"]}

        输入：你好，今天天气不错。
        输出：{"facts": []}

        请分析以下对话，并按照上述格式返回 JSON 对象。
        """

        config = {
            "vector_store": {
                "provider": "milvus",
                "config": {
                    "collection_name": collection_name,
                    "url": milvus_uri,
                    "token": "",
                    "embedding_model_dims": vector_dim,
                }
            },
            "llm": {
                "provider": "ollama",
                "config": {
                    "model": llm_model,
                    "temperature": 0.3,
                    "max_tokens": 512,
                    "ollama_base_url": ollama_base_url,
                }
            },
            "embedder": {
                "provider": "ollama",
                "config": {
                    "model": embed_model,
                    "ollama_base_url": ollama_base_url,
                }
            },
            "custom_fact_extraction_prompt": custom_extraction_prompt,  # 注入自定义 Prompt
            "version": "v1.1",
        }

        logger.info(f"初始化 Mem0 长期记忆模块，集合：{collection_name}")
        self.memory = Memory.from_config(config)

        # 启动时加载集合
        self._ensure_collection_loaded()

    def _ensure_collection_loaded(self) -> None:
        """
        确保 Milvus 集合已加载。

        使用 pymilvus.utility 检查集合是否存在，若存在则调用 Collection.load()。
        load() 是幂等操作，已加载的集合不会重复加载。
        """
        try:
            if utility.has_collection(self.collection_name):
                collection = Collection(self.collection_name)
                collection.load()
                logger.debug(f"集合 {self.collection_name} 已加载")
            else:
                logger.warning(f"集合 {self.collection_name} 不存在，请先运行创建脚本")
        except Exception as e:
            logger.error(f"加载集合失败: {e}")

    async def retrieve_relevant(
        self,
        user_id: str,
        query: str,
        memory_type: Optional[str] = None,
        config: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        检索与用户问题相关的长期记忆。

        流程：
            1. 确保集合已加载。
            2. 构建过滤条件（user_id 和可选的 memory_type）。
            3. 调用 Mem0 的 search 方法（同步，在线程池中执行）。
            4. 解析返回结果，提取内容、分数，并从内容中解析类型和重要性标签。
            5. 按分数降序排序，取前 top_k 条返回。

        Args:
            user_id: 用户 ID。
            query: 查询文本（通常为中文问题）。
            memory_type: 可选，按记忆类型过滤（preference/fact/episodic）。
            config: 保留参数，未使用。

        Returns:
            记忆列表，每个元素包含：
                - content: 纯净的记忆文本（已移除标签）
                - score: 相关性分数（0-1）
                - metadata: 字典，包含 memory_type, importance, timestamp, user_id 等
        """
        self._ensure_collection_loaded()

        filters = {"user_id": user_id}
        if memory_type:
            filters["memory_type"] = memory_type

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: self.memory.search(
                query=query,
                user_id=user_id,
                limit=self.top_k * 2,
                filters=filters,
            )
        )

        logger.info(f"检索到 {len(results)} 条相关记忆")

        formatted = []
        for r in results:
            try:
                # 从结果中提取基本数据
                if hasattr(r, 'metadata'):
                    original_metadata = r.metadata or {}
                    content = r.memory if hasattr(r, 'memory') else str(r)
                    score = getattr(r, 'score', 0.0)
                elif isinstance(r, dict):
                    original_metadata = r.get('metadata', {})
                    content = r.get('memory', r.get('text', ''))
                    score = r.get('score', 0.0)
                else:
                    original_metadata = {}
                    content = str(r)
                    score = 0.0

                if not content:
                    continue

                # 解析内容中的类型和重要性标签
                memory_type_parsed = "unknown"
                importance_parsed = 0.5
                if content:
                    # 正则匹配 [类型: xxx] 和 [重要性: x.x]
                    type_match = re.search(r'\[类型:\s*(\w+)\]', content)
                    if type_match:
                        memory_type_parsed = type_match.group(1)
                        # 从原内容中移除标签，保持纯净内容
                        content = re.sub(r'\s*\[类型:\s*\w+\]', '', content)
                    imp_match = re.search(r'\[重要性:\s*([0-9.]+)\]', content)
                    if imp_match:
                        importance_parsed = float(imp_match.group(1))
                        content = re.sub(r'\s*\[重要性:\s*[0-9.]+\]', '', content)
                    content = content.strip()

                # 合并元数据：保留原有字段，添加解析出的字段
                merged_metadata = {
                    "memory_type": memory_type_parsed,
                    "importance": importance_parsed,
                    "timestamp": original_metadata.get("timestamp", 0),
                    "user_id": original_metadata.get("user_id", user_id),
                    # 同时保留其他原始元数据，避免丢失
                    **{k: v for k, v in original_metadata.items() if k not in ["memory_type", "importance", "timestamp", "user_id"]}
                }

                formatted.append({
                    "content": content,
                    "score": score,
                    "metadata": merged_metadata,
                })
            except Exception as e:
                logger.warning(f"解析检索结果失败: {e}")

        formatted.sort(key=lambda x: x["score"], reverse=True)
        return formatted[:self.top_k]

    async def process_and_store(
        self,
        user_id: str,
        user_input: str,
        assistant_output: str,
        config: Optional[Dict] = None,
    ) -> None:
        """
        处理并存储长期记忆。

        将本轮对话提交给 Mem0，由 Mem0 内部根据以下规则决定是否存储：
            - 对话中是否包含可记忆的信息（由 LLM 判断）
            - 新信息与已有记忆的关系（新增、更新、合并）
            - 自动去重，避免重复存储

        Args:
            user_id: 用户 ID。
            user_input: 本轮用户输入。
            assistant_output: 本轮助手回复。
            config: 保留参数，未使用。
        """
        self._ensure_collection_loaded()

        messages = [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": assistant_output},
        ]

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.memory.add(messages, user_id=user_id)
            )
            logger.info(f"记忆存储成功: {result}")
        except Exception as e:
            logger.error(f"存储记忆失败: {e}", exc_info=True)

    def get_tools(self) -> List:
        """返回 Mem0 自带的工具（如果需要），但通常直接使用上述接口，因此返回空列表。"""
        return []
