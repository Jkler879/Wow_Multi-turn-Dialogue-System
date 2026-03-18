"""
高频查询缓存模块完整测试（带详细输出）
使用真实 Redis，每个测试独立运行，并打印测试过程。
"""

import unittest
import time
import redis
from redis_bloom import HighFreqCache


class TestHighFreqCacheVerbose(unittest.TestCase):
    """高频查询缓存测试（详细输出版）"""

    @classmethod
    def setUpClass(cls):
        """类级别初始化，打印开始标记"""
        print("\n" + "="*60)
        print("开始执行高频查询缓存测试")
        print("="*60)

    def setUp(self):
        """每个测试前准备"""
        print(f"\n--- 准备测试: {self._testMethodName} ---")
        self.redis_client = redis.Redis(host='localhost', port=6380, decode_responses=True)
        self.redis_client.flushdb()
        self.cache = HighFreqCache(
            redis_client=self.redis_client,
            threshold=3,
            window_seconds=2,
            bloom_capacity=1000,
            bloom_error_rate=0.01
        )
        print("Redis 已清空，缓存实例已创建")

    def tearDown(self):
        """每个测试后清理"""
        self.redis_client.flushdb()
        self.redis_client.close()
        print(f"--- 测试 {self._testMethodName} 完成，数据已清理 ---")

    def test_basic_get_update(self):
        """基础流程：更新三次后应能命中缓存"""
        print("测试: 基础 get/update 流程")
        query = "what is the plot of inception"
        for i in range(3):
            self.cache.update(query, f"response{i}")
            print(f"第 {i+1} 次更新后，get() 结果: {self.cache.get(query)}")
            if i < 2:
                self.assertIsNone(self.cache.get(query), f"第 {i+1} 次后不应有缓存")
        # 第三次后应有缓存
        cached = self.cache.get(query)
        print(f"最终缓存结果: {cached}")
        self.assertEqual(cached, "response2", "缓存内容应为最后一次更新的响应")

    def test_threshold_not_reached(self):
        """未达到阈值时不应返回缓存"""
        print("测试: 未达阈值无缓存")
        query = "not frequent"
        for i in range(2):
            self.cache.update(query, f"r{i}")
            print(f"第 {i+1} 次更新后 get: {self.cache.get(query)}")
            self.assertIsNone(self.cache.get(query), f"第 {i+1} 次后不应有缓存")
        print("阈值未达到，测试通过")

    def test_normalization_hit(self):
        """归一化后语义相同的查询应命中缓存"""
        print("测试: 归一化命中")
        q1 = "Inception (2010) plot"
        q2 = "INCEPTION（２０１０）PLOT"  # 全角括号和数字
        print(f"原始查询 q1: {q1}")
        print(f"原始查询 q2: {q2}")
        for i in range(3):
            self.cache.update(q1, "inception plot")
            print(f"q1 第 {i+1} 次更新")
        print(f"q1 归一化后: {self.cache.normalize(q1)}")
        print(f"q2 归一化后: {self.cache.normalize(q2)}")
        result = self.cache.get(q2)
        print(f"使用 q2 查询缓存结果: {result}")
        self.assertEqual(result, "inception plot", "q2 应命中 q1 的缓存")

    def test_counter_expiration(self):
        """计数器窗口过期后应重置"""
        print("测试: 计数器过期重置")
        query = "expire test"
        norm = self.cache.normalize(query)
        cnt_key = self.cache._get_counter_key(norm)

        self.cache.update(query, "r1")
        self.cache.update(query, "r2")
        cnt = self.redis_client.get(cnt_key)
        print(f"两次更新后计数器值: {cnt}")
        self.assertEqual(int(cnt), 2, "计数器应为2")

        print("等待窗口过期 (3秒)...")
        time.sleep(3)

        self.cache.update(query, "r3")
        cnt = self.redis_client.get(cnt_key)
        print(f"过期后第一次更新计数器: {cnt}")
        self.assertEqual(int(cnt), 1, "过期后计数器应从1开始")

        # 再更新两次，触发缓存
        self.cache.update(query, "r3")  # 第二次
        self.cache.update(query, "r3")  # 第三次
        cached = self.cache.get(query)
        print(f"再更新两次后缓存结果: {cached}")
        self.assertEqual(cached, "r3", "过期后累计3次应触发缓存")

    def test_bloom_filter_reject(self):
        """布隆过滤器应快速拒绝未出现过的查询"""
        print("测试: 布隆过滤器快速拒绝")
        query = "never seen before"
        print(f"查询 '{query}' 从未出现过，get 应返回 None")
        result = self.cache.get(query)
        print(f"get 结果: {result}")
        self.assertIsNone(result, "新查询应返回 None")


if __name__ == '__main__':
    # 使用 verbosity=2 运行，获得更详细的输出
    unittest.main(verbosity=2)
