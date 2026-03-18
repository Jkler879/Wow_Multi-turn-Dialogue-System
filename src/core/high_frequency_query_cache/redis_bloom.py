"""
高频查询缓存模块（High Frequency Query Cache）
- ReAct Agent前置模块，不属于ReAct 工具集


该模块负责对改写后的用户查询进行高级归一化处理，避免只有完全一致的字符串才能命中缓存。
归一化处理后通过 Redis 布隆过滤器 + 计数器实现高频查询缓存，减少重复计算。
如缓存命中，则直接返回最终答案。
如缓存未命中，则进入后续ReAct Agent模块。


已通过测试用例评估高频查询功能，测试代码在当前路径同级目录 redis_bloom_test_case.py

主要功能：
    - 高级查询归一化（全角半角符号、货币符号、中英文数字类型等）
    - 基于 Redis 布隆过滤器快速判断查询是否曾出现
    - 统计查询频率，超过阈值后直接返回缓存
    - 提供 get/update/clear 接口

依赖：
    - redis: 历史存储
    - cn2an: 中英文数字转换
    - bloom: 布隆过滤器，redis-stack自带


Author: Ke Meng
Created: 2026-01-22
Version: 1.0.1
Last Modified: 2026-03-18


变更记录：
    - 1.0.1 (2026-03-18):
                        改动1、引入cn2an库解决中英文数字类型转换，去除自定义词典（过于繁琐且效果不佳）

    - 1.0.0 (2026-01-22): 初始版本

"""

import logging
import re
import redis
from typing import Optional, Callable

# 导入cn2an库（能够将“一百二十三”、“壹佰贰拾叁”、“1百23”等不同形式的中文数字或混合表达式，准确地转换为阿拉伯数字“123”）
try:
    import cn2an
    CN2AN_AVAILABLE = True
except ImportError:
    CN2AN_AVAILABLE = False
    # 可以在此处记录一个警告日志，但不要中断程序
    print("Warning: cn2an library not installed. Chinese number normalization will be disabled.")

logger = logging.getLogger(__name__)


def _build_full_to_half_map():
    """构建全角字符到半角字符的映射字典"""
    return {
        0xff01: 0x21, 0xff02: 0x22, 0xff03: 0x23, 0xff04: 0x24,
        0xff05: 0x25, 0xff06: 0x26, 0xff07: 0x27, 0xff08: 0x28,
        0xff09: 0x29, 0xff0a: 0x2a, 0xff0b: 0x2b, 0xff0c: 0x2c,
        0xff0d: 0x2d, 0xff0e: 0x2e, 0xff0f: 0x2f, 0xff10: 0x30,
        0xff11: 0x31, 0xff12: 0x32, 0xff13: 0x33, 0xff14: 0x34,
        0xff15: 0x35, 0xff16: 0x36, 0xff17: 0x37, 0xff18: 0x38,
        0xff19: 0x39, 0xff1a: 0x3a, 0xff1b: 0x3b, 0xff1c: 0x3c,
        0xff1d: 0x3d, 0xff1e: 0x3e, 0xff1f: 0x3f, 0xff20: 0x40,
        0xff21: 0x61, 0xff22: 0x62, 0xff23: 0x63, 0xff24: 0x64,
        0xff25: 0x65, 0xff26: 0x66, 0xff27: 0x67, 0xff28: 0x68,
        0xff29: 0x69, 0xff2a: 0x6a, 0xff2b: 0x6b, 0xff2c: 0x6c,
        0xff2d: 0x6d, 0xff2e: 0x6e, 0xff2f: 0x6f, 0xff30: 0x70,
        0xff31: 0x71, 0xff32: 0x72, 0xff33: 0x73, 0xff34: 0x74,
        0xff35: 0x75, 0xff36: 0x76, 0xff37: 0x77, 0xff38: 0x78,
        0xff39: 0x79, 0xff3a: 0x7a, 0xff3b: 0x5b, 0xff3c: 0x5c,
        0xff3d: 0x5d, 0xff3e: 0x5e, 0xff3f: 0x5f, 0xff40: 0x60,
        0xff41: 0x61, 0xff42: 0x62, 0xff43: 0x63, 0xff44: 0x64,
        0xff45: 0x65, 0xff46: 0x66, 0xff47: 0x67, 0xff48: 0x68,
        0xff49: 0x69, 0xff4a: 0x6a, 0xff4b: 0x6b, 0xff4c: 0x6c,
        0xff4d: 0x6d, 0xff4e: 0x6e, 0xff4f: 0x6f, 0xff50: 0x70,
        0xff51: 0x71, 0xff52: 0x72, 0xff53: 0x73, 0xff54: 0x74,
        0xff55: 0x75, 0xff56: 0x76, 0xff57: 0x77, 0xff58: 0x78,
        0xff59: 0x79, 0xff5a: 0x7a, 0xff5b: 0x7b, 0xff5c: 0x7c,
        0xff5d: 0x7d, 0xff5e: 0x7e
}


def _build_currency_map():
    """构建货币符号统一映射字典（键为正则表达式，值为标准符号）"""
    return {
        r'\$': '$', r'usd\b': '$', r'dollar\b': '$', r'dollars\b': '$',
        r'€': '€', r'eur\b': '€', r'euro\b': '€', r'euros\b': '€',
        r'£': '£', r'gbp\b': '£', r'pound\b': '£', r'pounds\b': '£',
        r'¥': '¥', r'cny\b': '¥', r'jpy\b': '¥', r'yuan\b': '¥', r'yen\b': '¥',
    }


# 在模块顶部调用一次，得到最终的字典常量
_full_to_half = _build_full_to_half_map()
CURRENCY_MAP = _build_currency_map()


def advanced_normalize(text: str) -> str:
    """
    对查询文本进行高级归一化处理。

    处理步骤包括：
        1. 去除首尾空格并转小写
        2. 全角字符转半角
        3. 货币符号统一
        4. 中文数字转阿拉伯数字（cn2an库）
        5. 去除千位分隔符
        6. 科学计数法规范化
        7. 括号周围空格规范化
        8. 合并多余空格

    Args:
        text: 原始查询字符串。

    Returns:
        归一化后的字符串。
    """
    if not text:
        return ""

    # 1. 基础清理
    s = text.strip().lower()

    # 2. 全角转半角
    s = s.translate(_full_to_half)

    # 3. 货币符号统一
    for pattern, replacement in CURRENCY_MAP.items():
        s = re.sub(pattern, replacement, s, flags=re.IGNORECASE)

    # 4. 数字格式化：使用 cn2an 将中文数字部分转换为阿拉伯数字
    if CN2AN_AVAILABLE:
        try:
            # 使用 smart 模式，它可以处理像 "1百23" 这样的混合表达
            s = cn2an.transform(s, "cn2an")
        except Exception as e:
            # 如果转换失败（例如，输入中没有中文数字），则忽略
            logger.debug(f"cn2an transform failed: {e}")
    else:
        logger.warning("cn2an not installed, Chinese numbers may not be normalized.")

    # 5. 去除千位分隔符（逗号或空格）
    s = re.sub(r'(?<=\d)[, ](?=\d)', '', s)

    # 6. 科学计数法转换
    def _sci_to_float(match):
        try:
            f = float(match.group(0))
            if f.is_integer():
                return str(int(f))
            else:
                return f"{f:.10f}".rstrip('0').rstrip('.')
        except:
            return match.group(0)
    s = re.sub(r'\d+\.?\d*[eE][+-]?\d+', _sci_to_float, s)

    # 7. 规范化括号周围空格（新增）
    # 确保左括号前有空格（如果前面是字母或数字）
    s = re.sub(r'([a-z0-9])\(', r'\1 (', s)
    # 确保右括号后有空格（如果后面是字母或数字）
    s = re.sub(r'\)([a-z0-9])', r') \1', s)

    # 8. 合并多余空格
    s = re.sub(r'\s+', ' ', s).strip()

    return s


class HighFreqCache:
    """
    高频查询缓存器，基于 Redis 布隆过滤器和计数器实现。

    核心逻辑：
        - 对查询进行归一化（可通过 normalize_func 自定义）
        - 使用布隆过滤器快速判断查询是否曾出现（减少 Redis 读压力）
        - 通过计数器统计查询频率，窗口内超过阈值则缓存响应
        - 后续相同查询直接返回缓存，避免重复计算

    """

    def __init__(
        self,
        redis_client: redis.Redis,
        bloom_key: str = "bf:queries",
        counter_prefix: str = "cnt:",
        cache_prefix: str = "cache:",
        threshold: int = 5,
        window_seconds: int = 1800,      # 30分钟内超过5次重复查询定义为高频查询
        normalize_func: Optional[Callable[[str], str]] = None,
        bloom_error_rate: float = 0.01,  # 布隆过滤器误判率1%,后续计数器精确计数,误判率可忽略
        bloom_capacity: int = 1000000,   # 布隆过滤器的不重复查询上限：100万
    ):
        """
        :param redis_client: Redis客户端（已连接）
        :param bloom_key: 布隆过滤器键名
        :param counter_prefix: 计数器键前缀
        :param cache_prefix: 缓存键前缀
        :param threshold: 高频阈值
        :param window_seconds: 计数窗口（秒）
        :param normalize_func: 自定义归一化函数，若为None则使用内置的advanced_normalize
        :param bloom_error_rate: 布隆过滤器期望误判率
        :param bloom_capacity: 布隆过滤器预期元素数量
        """
        self.redis = redis_client
        self.bloom_key = bloom_key
        self.counter_prefix = counter_prefix
        self.cache_prefix = cache_prefix
        self.threshold = threshold
        self.window = window_seconds
        self.normalize = normalize_func if normalize_func else advanced_normalize

        self._ensure_bloom_filter(bloom_error_rate, bloom_capacity)

    def _ensure_bloom_filter(self, error_rate: float, capacity: int):
        """确保布隆过滤器存在"""
        try:
            self.redis.execute_command('BF.INFO', self.bloom_key)
        except redis.exceptions.ResponseError:
            self.redis.execute_command('BF.RESERVE', self.bloom_key, error_rate, capacity)
            logger.info(f"布隆过滤器已创建: {self.bloom_key}")

    def _get_counter_key(self, norm_query: str) -> str:
        return f"{self.counter_prefix}{norm_query}"

    def _get_cache_key(self, norm_query: str) -> str:
        return f"{self.cache_prefix}{norm_query}"

    def get(self, query: str) -> Optional[str]:
        """获取查询的缓存响应"""
        logger.info(f"[Cache] 接收到的查询改写模块输出: {query}")
        norm = self.normalize(query)
        logger.info(f"[Cache] Normalized query: {norm}")
        exists = self.redis.execute_command('BF.EXISTS', self.bloom_key, norm)
        if not exists:
            logger.info(f"[Cache] Query '{norm}' not in bloom filter, cache miss")
            return None

        counter_key = self._get_counter_key(norm)
        count = self.redis.get(counter_key)
        if count is None or int(count) < self.threshold:
            logger.info(f"[Cache] Query '{norm}' count {count} below threshold {self.threshold}, cache miss")
            return None

        cache_key = self._get_cache_key(norm)
        cached = self.redis.get(cache_key)

        if cached:
            logger.info(f"[Cache] HIT! Returning cached response for '{norm}': {cached}")
        else:
            logger.warning(f"[Cache] Query '{norm}' passed threshold but cache missing (expired?)")
        return cached

    def update(self, query: str, response: str):
        """更新查询的计数和缓存"""
        norm = self.normalize(query)
        counter_key = self._get_counter_key(norm)
        cache_key = self._get_cache_key(norm)

        pipe = self.redis.pipeline()
        pipe.incr(counter_key)
        pipe.expire(counter_key, self.window)
        pipe.execute_command('BF.ADD', self.bloom_key, norm)
        results = pipe.execute()
        new_count = results[0]
        logger.info(f"[Cache] Query '{norm}' new count: {new_count}")

        if new_count == self.threshold:
            self.redis.setex(cache_key, self.window, response)
            logger.info(f"查询 '{norm}' 达到阈值 {self.threshold}，已缓存响应")
        elif new_count > self.threshold:
            if not self.redis.exists(cache_key):
                self.redis.setex(cache_key, self.window, response)
            else:
                logger.debug(f"[Cache] Query '{norm}' already cached")

    def clear(self, query: str):
        """清除特定查询的计数和缓存（管理用）"""
        norm = self.normalize(query)
        pipe = self.redis.pipeline()
        pipe.delete(self._get_counter_key(norm))
        pipe.delete(self._get_cache_key(norm))
        pipe.execute()
