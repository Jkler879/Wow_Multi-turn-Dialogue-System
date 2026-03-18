# config.py
import os
from dotenv import load_dotenv
load_dotenv()
from config.paths import paths  # 导入路径管理器


# ==================== Redis 配置 ====================
REDIS_CONFIG = {
    'host': os.getenv('REDIS_HOST', 'localhost'),
    'port': int(os.getenv('REDIS_PORT', 6380)),
    'db': int(os.getenv('REDIS_DB', 0)),
    'decode_responses': True,
    'socket_connect_timeout': 5,
    'socket_timeout': 5,
    # 如果需要密码：'password': os.getenv('REDIS_PASSWORD', None),
}

# Redis 键前缀常量（用于逻辑隔离）
REDIS_PREFIX = {
    'session': 'session:',          # 短期记忆历史
    'counter': 'cnt:',              # 高频查询计数器
    'cache': 'cache:',              # 高频缓存响应
    'bloom': 'bf:',                 # 布隆过滤器
    'pref': 'pref:',                # 用户偏好缓存
    'token': 'token:',              # 临时令牌
}


# ==================== Milvus 配置 ====================
MILVUS_CONFIG = {
    'uri': os.getenv('MILVUS_URI', 'http://localhost:19530'),
    # 如果使用用户名密码：
    # 'user': os.getenv('MILVUS_USER', ''),
    # 'password': os.getenv('MILVUS_PASSWORD', ''),
    # 'token': os.getenv('MILVUS_TOKEN', None),
}

# Milvus 知识库集合名称
MILVUS_COLLECTION = os.getenv('MILVUS_COLLECTION', 'wow_knowledge_base_v2')


# ==================== Ollama 配置 ====================
OLLAMA_CONFIG = {
    'base_url': os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
    'model': os.getenv('OLLAMA_MODEL', 'qwen3-4b-rewrite'),          # 用于查询改写
    'embed_model': os.getenv('EMBED_MODEL', 'nomic-embed-text'),     # 嵌入模型
    'long_memory_model': os.getenv('OLLAMA_MEMORY_MODEL', 'qwen3-1.7b')  # 长期记忆模型
}


# ==================== LLM API 配置 ====================
API_CONFIG = {
    'model': os.getenv('API_MODEL', 'qwen-plus'),  # qwen-plus api模型
    'api_key': "OPENAI_API_KEY",  # 环境变量读取
    'base_url': os.getenv('OPENAI_API_BASE', 'https://dashscope.aliyuncs.com/compatible-mode/v1'),  # qwen api调用中国大陆地址
    'temperature': float(os.getenv('API_TEMPERATURE', 0.7)),
}


# ==================== Neo4j 配置 ====================
NEO4J_CONFIG = {
    'url': os.getenv('NEO4J_URL', 'bolt://localhost:7687'),
    'user': os.getenv('NEO4J_USER', 'neo4j'),
    'password': os.getenv('NEO4J_PASSWORD', 'NEO4J_PASSWORD'),  # 环境变量读取
    'database': os.getenv('NEO4J_DATABASE', 'neo4j'),
}


# ==================== 重排器配置 ====================
RERANKER_CONFIG = {
    'model_path': str(paths.reranker_model),  # 重排模型
    'device': os.getenv('RERANKER_DEVICE', 'cpu'),
}

# ==================== 检索工具参数 ====================
RETRIEVAL_CONFIG = {
    'vector_top_k': int(os.getenv('VECTOR_TOP_K', 10)),          # 向量检索返回候选数
    'bm25_top_k': int(os.getenv('BM25_TOP_K', 10)),              # BM25全文检索返回候选数
    'rrf_k': int(os.getenv('RRF_K', 60)),                        # RRF 常数
    'rrf_final_top_k': int(os.getenv('RRF_FINAL_TOP_K', 10)),    # RRF 融合后保留数（送入重排）
}


# ==================== 翻译工具参数 ====================
TRANSLATOR_CONFIG = {
    'model_name': os.getenv('TRANSLATOR_MODEL', 'Helsinki-NLP/opus-mt-en-zh'),
    'device': os.getenv('TRANSLATOR_DEVICE', 'cpu'),
    'model_path': str(paths.translator_model),  # 使用 paths 统一管理
}


# ==================== 高频缓存配置 ====================
CACHE_CONFIG = {
    'threshold': int(os.getenv('CACHE_THRESHOLD', 5)),           # 高频阈值
    'window_seconds': int(os.getenv('CACHE_WINDOW', 1800)),      # 计数窗口（秒）
    'bloom_capacity': int(os.getenv('BLOOM_CAPACITY', 1000000)),  # 布隆过滤器预期容量
    'bloom_error_rate': float(os.getenv('BLOOM_ERROR_RATE', 0.01)),  # 误判率
}


# ==================== 短期记忆配置 ====================
MEMORY_CONFIG = {
    'max_history_turns': int(os.getenv('MAX_HISTORY_TURNS', 5)),  # 保留对话轮数
    'expire_seconds': int(os.getenv('MEMORY_EXPIRE', 86400)),    # 过期时间（秒，默认24h）
}

# ==================== 长期记忆配置 ====================
LONG_TERM_MEMORY_CONFIG = {
    # Milvus 配置
    'collection_name': os.getenv('LTM_COLLECTION', 'user_long_term_memory'),  # milvus长期记忆集合，和知识库集合逻辑隔离
    'milvus_uri': os.getenv('MILVUS_URI', 'http://localhost:19530'),
    'vector_dim': int(os.getenv('LTM_VECTOR_DIM', 768)),  # 向量模型维度

    # 检索参数
    'top_k': int(os.getenv('LTM_TOP_K', 3)),  # 长期记忆最多返回上限
    'score_threshold': float(os.getenv('LTM_SCORE_THRESHOLD', 0.7)),  #
    'importance_weight': float(os.getenv('LTM_IMPORTANCE_WEIGHT', 0.2)),
    'time_decay_weight': float(os.getenv('LTM_TIME_DECAY_WEIGHT', 0.1)),

    # Memory Manager 模型（用小模型）
    'manager_model': OLLAMA_CONFIG['long_memory_model'],  # Ollama qwen3-1.7b
    'manager_base_url': OLLAMA_CONFIG['base_url'],
}

# ==================== 查询改写器配置 ====================
REWRITER_CONFIG = {
    'max_history_turns': MEMORY_CONFIG['max_history_turns'],     # 与记忆配置同步
    'timeout': int(os.getenv('REWRITER_TIMEOUT', 90)),  # 查询改写超时时间
    'max_retries': int(os.getenv('REWRITER_MAX_RETRIES', 3)),  # 重试次数上限
}
