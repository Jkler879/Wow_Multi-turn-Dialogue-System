"""
查询改写模块 (Query Rewriter)
- ReAct Agent前置模块，不属于ReAct 工具集

该模块负责判断用户输入的session_id是否存在对话历史记录
- 如存在：从Redis导入短期记忆（上限五轮），送入本地模型Qwen3-4b-rewrite（已量化）执行指代消解，并生成中英文结果。
（英文结果用于检索后续高频查询缓存，中文结果用于长期记忆检索）
- 如不存在：即当前对话无历史记录，直接调用模型生成英文翻译供后续模块使用。

已通过测试用例评估指代消解质量，测试代码在当前路径同级目录 query_rewrite_test_case.py

主要功能：
    - 从 Redis 获取会话历史（Hash + Sorted Set 存储）
    - 调用本地 Qwen3-4B-rewrite 量化模型（Ollama）生成中英文改写
    - 入口函数添加LangSmith监控装饰器，监控日志通过LangSmith平台存储
    - 已添加后台模型预热、超时重试、本地logger日志

依赖：
    - redis: 历史存储
    - requests: 调用 Ollama模型
    - langsmith: 监控追踪

Author: Ke Meng
Created: 2026-01-20
Version: 1.0.1
Last Modified: 2026-03-18

变更记录：
    - 1.0.1 (2026-03-18):
                        改动1、本地模型从Qwen3-1.7b升级为Qwen3-4b，小模型指代消解测试后效果不佳
                        改动2、few-shot添加无对话历史示例，指导模型学习无历史对话的输出格式
                        改动3、Redis 数据存储格式从 list升级为 Hash + Sorted Set
                        改动4、添加 LangSmith监控 + 本地 logger监控，方便代码集成时的调试

    - 1.0.0 (2026-01-20): 初始版本
"""

import json
import logging
import threading
import time
from typing import Dict, List, Optional, Union

import redis
import requests
from langsmith import traceable  # LangSmith监控

logger = logging.getLogger(__name__)


# 生产级 few-shot 示例（用于英文改写，同时示范中英文输出格式）
PRODUCTION_FEW_SHOT_EXAMPLES = [
    """# 无历史对话（重要）
对话历史：
无对话历史。
用户当前问题：你好吗？
输出格式：
EN: How are you?
ZH: 你好吗？""",

    """# 跨多轮指代最近提及实体（重要）
对话历史：
用户：推荐一部科幻电影。
系统：《银翼杀手》（Blade Runner）是经典之作。
用户：导演是谁？
系统：雷德利·斯科特（Ridley Scott）执导的。
用户：他还拍过其他电影吗？
系统：比如《异形》（Alien）也很出名。
用户：它讲了什么？
输出格式：
EN: What is the plot of the movie "Alien"?
ZH: 电影《异形》讲了什么？""",

    """# 人称代词指代
对话历史：
用户：我想了解关于时间旅行的科幻作品。
系统：时间旅行是科幻作品中常见的主题，比如电影《回到未来》（Back to the Future）。
用户：它讲述了一个什么样的故事？
输出格式：
EN: What is the plot of the movie "Back to the Future"?
ZH: 电影《回到未来》讲述了什么故事？""",

    """# 中文电影名翻译示例
对话历史：
用户：最近有什么好看的国产科幻片？
系统：《流浪地球2》（The Wandering Earth 2）口碑很好。
用户：它讲的是什么？
输出格式：
EN: What is the plot of the movie "The Wandering Earth 2"?
ZH: 电影《流浪地球2》讲了什么？""",
]


class QueryRewriter:
    """
    查询改写器，一次模型调用同时生成英文和中文版本。

    Attributes:
        redis_client (redis.Redis): Redis 客户端实例。
        max_history_turns (int): 最大历史对话轮数。
        ollama_base_url (str): Ollama API 基础 URL。
        ollama_model (str): Ollama 模型名称。
        rewrite_timeout (int): API 调用超时时间（秒）。
        max_retries (int): 最大重试次数。
        few_shot_examples (List[str]): 少样本示例列表。
    """

    # 结构化System Prompt
    SYSTEM_PROMPT = (
        "你是一个专业的查询改写助手。你的核心任务是根据对话历史和用户当前的问题，生成一个独立、完整、可直接用于英文知识库检索的问题，并同时提供对应的中文翻译。\n"
        "请遵循以下规则：\n"
        "1. **语言要求**：英文问题必须准确、完整，可用于检索；中文翻译应与英文问题语义一致。\n"
        "2. **专有名词翻译**：电影名、人名、书名等专有名词应使用公认的标准英文名称（例如《流浪地球2》应译为 \"The Wandering Earth 2\"），中文翻译中使用常用名称（如“流浪地球2”）。\n"
        "3. **输出格式**：必须输出两行，第一行以 'EN:' 开头，第二行以 'ZH:' 开头。不要输出任何解释、注释或多余的文字。\n"
        "4. **无历史处理**：如果对话历史为空，请直接将用户当前问题转换为合适的英文检索问题，并翻译成中文。\n"
        "5. **指代消解**：如果用户问题中包含指代（如“它”、“这个”等），结合对话历史将其替换为明确的实体。\n"
        "6. **完整性**：确保生成的问题包含所有必要的上下文，不依赖对话历史也能理解。\n"
        "7. **时序优先**：当指代词可能对应多个历史实体时，选择最近一次对话中提到的实体。\n"
        "\n"
        "下面是一些示例，请参考它们的输出格式。"
    )

    def __init__(
        self,
        redis_client: redis.Redis,
        max_history_turns: int = 3,
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "qwen3-4b-rewrite",
        rewrite_timeout: int = 180,
        max_retries: int = 3,
        few_shot_examples: Optional[Union[str, List[str]]] = None,
        enable_warmup: bool = True,
    ):
        self.redis_client = redis_client
        self.max_history_turns = max_history_turns
        self.ollama_base_url = ollama_base_url.rstrip('/')
        self.ollama_model = ollama_model
        self.rewrite_timeout = rewrite_timeout
        self.max_retries = max_retries

        # 处理 few-shot 示例
        if few_shot_examples is None:
            self.few_shot_examples = PRODUCTION_FEW_SHOT_EXAMPLES
        elif isinstance(few_shot_examples, str):
            self.few_shot_examples = [few_shot_examples]
        else:
            self.few_shot_examples = few_shot_examples

        if enable_warmup:
            threading.Thread(target=self._warmup, daemon=True).start()

    def _warmup(self):
        """
        预热模型，发送一次空请求以减少首次推理延迟。

        Returns:
            None
        """
        try:
            self._call_ollama("Hello", allow_empty_return=True)
            logger.info("模型预热完成")
        except Exception as e:
            logger.warning(f"模型预热失败: {e}")

    def _get_chat_history(self, session_id: str) -> str:
        """
        从 Redis 获取指定会话的最近历史记录。

        Args:
            session_id (str): 会话 ID。

        Returns:
            str: 多行字符串，每行格式为 "用户: ..." 或 "系统: ..."。

        Raises:
            Exception: Redis 操作失败时记录日志但返回空字符串。
        """
        set_key = f"session:{session_id}:history"
        try:
            # 获取最近 max_history_turns * 2 条消息ID（按时间降序）
            msg_ids = self.redis_client.zrevrange(set_key, 0, self.max_history_turns * 2 - 1)
            if not msg_ids:
                return ""

            # 按时间正序排列
            msg_ids.reverse()

            history_lines = []
            for msg_id in msg_ids:
                msg_key = f"session:{session_id}:msg:{msg_id}"
                msg_data = self.redis_client.hgetall(msg_key)
                if not msg_data:
                    continue
                role = msg_data.get('role')
                content = msg_data.get('content', '')
                if role == 'user':
                    history_lines.append(f"用户: {content}")
                elif role == 'assistant':
                    history_lines.append(f"系统: {content}")
                else:
                    logger.warning(f"未知角色 {role} 的消息: {msg_id}")

            return "\n".join(history_lines)
        except Exception as e:
            logger.error(f"从 Redis 获取历史失败: {e}")
            return ""

    def _format_few_shot(self) -> str:
        return "\n\n".join(self.few_shot_examples)

    def _call_ollama(self, user_prompt: str, allow_empty_return: bool = False) -> Optional[str]:
        """
        调用 Ollama生成响应，支持重试和超时。

        Args:
            user_prompt (str): 发送给模型的提示词。
            allow_empty_return (bool): 是否允许超时后返回 None（用于预热）。

        Returns:
            Optional[str]: 模型响应的文本，失败时返回 None。
        """
        url = f"{self.ollama_base_url}/api/chat"
        payload = {
            "model": self.ollama_model,
            "messages": [{"role": "user", "content": user_prompt}],
            "think": False,
            "stream": False,
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.post(url, json=payload, timeout=self.rewrite_timeout)
                response.raise_for_status()
                data = response.json()
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Ollama 完整响应: " + json.dumps(data, indent=2, ensure_ascii=False))
                if "message" in data and "content" in data["message"]:
                    return data["message"]["content"].strip()
                else:
                    logger.error(f"Ollama 返回格式异常: {data}")
                    return None
            except requests.exceptions.Timeout:
                logger.warning(f"Ollama 调用超时 (尝试 {attempt+1}/{self.max_retries})")
                if attempt == self.max_retries - 1:
                    if not allow_empty_return:
                        logger.error("Ollama 调用最终超时失败")
                    return None
                time.sleep(2 ** attempt)
            except requests.exceptions.RequestException as e:
                logger.error(f"Ollama 调用失败 (尝试 {attempt+1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    return None
                time.sleep(2 ** attempt)
        return None

    def _parse_response(self, response: str) -> Dict[str, str]:
        """
        解析模型响应，提取英文和中文查询。

        Args:
            response (str): 模型返回的原始文本。

        Returns:
            Dict[str, str]: 包含 'en' 和 'zh' 的字典，若解析失败则值为 None。
        """
        lines = response.split('\n')
        en_query = None
        zh_query = None
        for line in lines:
            line = line.strip()
            if line.startswith('EN:'):
                en_query = line[3:].strip()
            elif line.startswith('ZH:'):
                zh_query = line[3:].strip()
        # 如果找不到标记，尝试取第一行和第二行
        if en_query is None and len(lines) >= 1:
            en_query = lines[0].strip()
        if zh_query is None and len(lines) >= 2:
            zh_query = lines[1].strip()
        return {"en": en_query, "zh": zh_query}

    @traceable   # 入口函数添加LangSmith装饰器
    def rewrite(self, user_input: str, session_id: str) -> Dict[str, str]:
        """
        执行查询改写，返回中英文双版本。

        Args:
            user_input (str): 用户当前输入。
            session_id (str): 会话 ID。

        Returns:
            Dict[str, str]: 包含 'en_query' 和 'zh_query' 的字典。
        """
        history = self._get_chat_history(session_id)
        logger.info(f"📥 用户原始输入: {user_input}")
        logger.info(f"📚 会话 {session_id} 的历史: {history if history else '无历史'}")
        history_part = history if history else "无对话历史。"

        logger.debug(f"会话 {session_id} 的历史内容: {history_part}")

        few_shot_part = self._format_few_shot()
        prompt = f"""{few_shot_part}

对话历史：
{history_part}

用户当前问题：{user_input}

请按照要求输出两行：EN: ... 和 ZH: ...
"""

        response = self._call_ollama(prompt)

        if response:
            parsed = self._parse_response(response)
            en_query = parsed["en"] if parsed["en"] else user_input
            zh_query = parsed["zh"] if parsed["zh"] else user_input
            logger.info(f"会话 {session_id} 改写成功: EN='{en_query}', ZH='{zh_query}'")
        else:
            logger.warning(f"会话 {session_id} 改写失败，使用原始输入")
            en_query = user_input
            zh_query = user_input
        logger.info(f"🔄 英文改写结果: {en_query}")
        logger.info(f"🀄️ 中文改写结果: {zh_query}")
        return {"en_query": en_query, "zh_query": zh_query}
