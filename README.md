graph TD
    User(用户) --> API[FastAPI接口]

    subgraph 请求处理流程
        API --> QW[查询改写 Qwen3-4B]
        QW --> Cache[高频缓存 Redis Bloom Filter]
        Cache -->|命中| DirectReturn[直接返回缓存答案]
        DirectReturn --> User
        Cache -->|未命中| ShortMem[短期记忆 Redis]
        ShortMem --> LongMemRet[长期记忆检索 Mem0+Milvus]
        LongMemRet --> MsgConstruct[构造初始消息]
        MsgConstruct --> Agent[ReAct Agent LangGraph]
    end

    subgraph ReAct Agent
        Agent --> LLM[LLM推理 Qwen-plus]
        LLM --> Parse[解析输出]
        Parse -->|调用工具| Tools[工具集]
        Tools -->|返回观察| LLM
        Parse -->|最终答案| FinalAns[最终答案]
    end

    subgraph 工具集
        Tools --> Retriever[检索工具]
        Tools --> Verifier[关系验证工具]
        Tools --> Translator[翻译工具]

        Retriever --> VectorSearch[向量检索 IVF_RABITQ]
        Retriever --> FullTextSearch[全文检索 BM25]
        VectorSearch --> RRF[RRF融合]
        FullTextSearch --> RRF
        RRF --> Rerank[BGE重排]
        Rerank --> RetResult[返回文档+分数]

        Verifier --> Neo4j[(Neo4j知识图谱)]
        Neo4j --> VeriResult[返回实体关系、证据、置信度]

        Translator --> Helsinki[Helsinki-NLP模型]
    end

    FinalAns -->|返回用户| User
    FinalAns --> AsyncLongMem[异步长期记忆存储]
    FinalAns --> UpdateCache[更新缓存及短期记忆]

    subgraph 长期记忆存储
        AsyncLongMem --> Decide[判断是否值得存储 qwen3-1.7b]
        Decide -->|是| Vectorize[向量化]
        Vectorize --> StoreMem[存入Milvus]
        Decide -->|否| Discard[丢弃]
    end

    subgraph 数据预处理与异步入库解耦
        RawData[原始数据] --> Chunking[文档分块]
        Chunking --> EntityExt[实体提取 all-mpnet-base-v2+LLM]
        EntityExt --> KBData[知识库数据含元数据]
        KBData --> Stream[Redis Stream消息队列]

        EntityExt --> RelationExt[实体关系抽取 LLM]
        RelationExt --> KGData[知识图谱三元组]
        KGData --> Stream

        Stream --> MilvusConsumer[写入Milvus]
        Stream --> Neo4jConsumer[写入Neo4j]
    end
