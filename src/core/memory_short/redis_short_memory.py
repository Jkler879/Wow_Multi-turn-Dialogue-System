"""
短期记忆模块 (Redis Short Memory)
- ReAct Agent前置模块，不属于ReAct 工具集

该模块负责存储和读取会话的短期历史（对话上下文）
1、通过用户输入的session_id判断有无对话历史数据：
- 如果有，调取所有短期对话（最高不超过5轮）,作为上下文传递给后续ReAct Agent 模块。
- 如果无，直接将改写后的查询传递给后续ReAct Agent模块。
2、后续系统最终回复后将消息存入短期记忆。

主要功能：
    - 基于 Hash + Sorted Set 存储短期记忆数据
    - 判断用户输入的session_id是否有历史记录, 有则提取注入上下文
    - 系统最终回复后，将用户输入和系统回复存入短期记忆


依赖：
    - redis： 短期记忆存储
    - langchain: 封装成 langchain 接口


Author: Ke Meng
Created: 2026-03-15
Version: 1.0.0
Last Modified: 2026-03-18


变更记录：
    - 1.0.1 (2026-03-18):
                        改动 1、Redis数据存储类型由 list 升级为Redis Hash + Sorted Set，优化读取速度和添加时序性
                            键设计：
                            - Sorted Set: session:{session_id}:history (成员为消息ID)
                            - Hash: session:{session_id}:msg:{msg_id} (存储消息内容)
                            - 消息格式：每条包含 role 和 content

    - 1.0.0 (2026-03-15): 初始版本

"""

import logging
import time
import json
from typing import List, Optional
from uuid import uuid4
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

logger = logging.getLogger(__name__)


class ShortTermMemory:
    """短期记忆管理器（基于Hash + Sorted Set）"""

    def __init__(self, redis_client, max_history_turns: int = 5, expire_seconds: int = 86400):
        """
        :param redis_client: 已连接的 Redis 客户端（decode_responses=True）
        :param max_history_turns: 最多保留的对话轮数（每轮含用户和系统两条消息）
        :param expire_seconds: 会话过期时间（秒），默认24小时
        """
        self.redis = redis_client
        self.max_turns = max_history_turns
        self.expire = expire_seconds
        self.history_set_key = "session:{session_id}:history"          # Sorted Set
        self.msg_prefix = "session:{session_id}:msg:"                  # Hash 前缀

    def _get_history_key(self, session_id: str) -> str:
        """获取存储消息顺序的 Sorted Set 键名"""
        return f"session:{session_id}:history"

    def _get_msg_key(self, session_id: str, msg_id: str) -> str:
        """获取存储单条消息的 Hash 键名"""
        return f"session:{session_id}:msg:{msg_id}"

    def _generate_msg_id(self) -> str:
        """生成唯一消息ID（基于时间戳和随机数）"""
        return f"{int(time.time() * 1000)}_{uuid4().hex[:8]}"

    def get_history(self, session_id: str) -> List[str]:
        """
        获取指定会话的最近历史（最多 max_turns 轮），返回字符串列表，
        每条格式为 "用户: {content}" 或 "系统: {content}"，保持与旧版兼容。
        """
        set_key = self._get_history_key(session_id)
        try:
            # 获取最近 max_turns * 2 条消息的ID（按时间戳降序，最后的是最新的）
            msg_ids = self.redis.zrevrange(set_key, 0, self.max_turns * 2 - 1)
            if not msg_ids:
                return []

            # 按时间正序返回（最早的在前）
            msg_ids.reverse()

            messages = []
            for msg_id in msg_ids:
                msg_key = self._get_msg_key(session_id, msg_id)
                # 获取 Hash 中的字段
                msg_data = self.redis.hgetall(msg_key)
                if not msg_data:
                    continue
                role = msg_data.get('role', 'unknown')
                content = msg_data.get('content', '')
                if role == 'user':
                    messages.append(f"用户: {content}")
                elif role == 'assistant':
                    messages.append(f"系统: {content}")
                else:
                    logger.warning(f"未知角色 {role} 的消息: {msg_id}")
            return messages
        except Exception as e:
            logger.error(f"读取历史失败: {e}")
            return []

    def get_messages(self, session_id: str) -> List[BaseMessage]:
        """
        获取指定会话的最近历史，转换为 LangChain 消息列表。
        每条历史格式为 "用户: {input}" 或 "系统: {output}"。
        """
        history = self.get_history(session_id)
        messages = []
        for line in history:
            if line.startswith("用户:"):
                content = line[3:].strip()
                messages.append(HumanMessage(content=content))
            elif line.startswith("系统:"):
                content = line[3:].strip()
                messages.append(AIMessage(content=content))
            else:
                logger.warning(f"无法识别的历史消息格式: {line}")
        return messages

    def add_turn(self, session_id: str, user_input: str, assistant_output: str) -> None:
        """
        向会话历史中添加一轮对话
        :param session_id: 会话ID
        :param user_input: 用户输入
        :param assistant_output: 系统回复
        """
        set_key = self._get_history_key(session_id)
        now = time.time()

        try:
            # 使用管道保证原子性
            pipe = self.redis.pipeline()

            # 存储用户消息
            user_msg_id = self._generate_msg_id()
            user_msg_key = self._get_msg_key(session_id, user_msg_id)
            pipe.hset(user_msg_key, mapping={
                'role': 'user',
                'content': user_input,
                'timestamp': now
            })
            pipe.zadd(set_key, {user_msg_id: now})

            # 存储系统消息
            sys_msg_id = self._generate_msg_id()
            sys_msg_key = self._get_msg_key(session_id, sys_msg_id)
            pipe.hset(sys_msg_key, mapping={
                'role': 'assistant',
                'content': assistant_output,
                'timestamp': now + 0.001  # 微小偏移保证顺序
            })
            pipe.zadd(set_key, {sys_msg_id: now + 0.001})

            # 修剪到最近 max_turns 轮（即 max_turns * 2 条消息）
            # 获取当前消息总数
            total = self.redis.zcard(set_key)
            if total > self.max_turns * 2:
                # 移除最早的 (total - max_turns*2) 条消息
                remove_count = total - self.max_turns * 2
                # 获取要移除的消息ID（按分数升序，最早的）
                old_msg_ids = self.redis.zrange(set_key, 0, remove_count - 1)
                if old_msg_ids:
                    pipe.zrem(set_key, *old_msg_ids)
                    # 同时删除对应的 Hash
                    for msg_id in old_msg_ids:
                        msg_key = self._get_msg_key(session_id, msg_id)
                        pipe.delete(msg_key)

            # 设置会话过期时间
            pipe.expire(set_key, self.expire)

            pipe.execute()
            logger.debug(f"会话 {session_id} 历史已更新")
        except Exception as e:
            logger.error(f"写入历史失败: {e}")

    def clear_history(self, session_id: str) -> None:
        """清空指定会话的历史（管理用）"""
        set_key = self._get_history_key(session_id)
        try:
            # 获取所有消息ID
            msg_ids = self.redis.zrange(set_key, 0, -1)
            pipe = self.redis.pipeline()
            if msg_ids:
                for msg_id in msg_ids:
                    msg_key = self._get_msg_key(session_id, msg_id)
                    pipe.delete(msg_key)
                pipe.zrem(set_key, *msg_ids)
            pipe.execute()
            logger.info(f"🗑️ 会话 {session_id} 历史已清空")
        except Exception as e:
            logger.error(f"清空历史失败: {e}")