from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
import asyncio


class StreamPrinter:
    """自定义回调处理器用于实时打印流式输出"""

    def __init__(self):
        self.buffer = ""

    def __call__(self, token: str):
        print(token, end="", flush=True)
        self.buffer += token


def create_streaming_chain():
    """创建流式处理链"""
    # 1. 创建组件 - 启用流式支持
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key="sk-1c8dc5bdda034300ac759617ea625722",
        base_url="https://api.deepseek.com/v1",
        streaming=True,  # True启用流式输出，False为正常响应
        temperature=0.7
    )

    # 2. 创建提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant"),
        ("user", "{text}")
    ])

    # 3. 输出解析器
    output_parser = StrOutputParser()

    # 4. 使用 LCEL 管道操作符创建链
    chain = prompt | llm | output_parser

    return chain


def sync_stream_example():
    """同步流式输出示例"""
    print("=== 同步流式输出 ===")
    chain = create_streaming_chain()

    # 创建流式处理器
    stream_printer = StreamPrinter()

    print("AI: ", end="", flush=True)

    # chain.stream流式调用
    for chunk in chain.stream({"text": "请用中文介绍一下人工智能的主要应用领域"}):
        stream_printer(chunk)

    print("\n" + "=" * 50)


async def async_stream_example():
    """异步流式输出示例"""
    print("=== 异步流式输出 ===")
    chain = create_streaming_chain()

    print("AI: ", end="", flush=True)

    # 异步流式调用 chain.astream()
    async for chunk in chain.astream({"text": "请用中文解释机器学习的基本概念"}):
        print(chunk, end="", flush=True)

    print("\n" + "=" * 50)


def batch_stream_example():
    """批量流式处理示例"""
    print("=== 批量流式处理 ===")
    chain = create_streaming_chain()

    questions = [
        "什么是深度学习？",
        "自然语言处理有哪些应用？",
        "计算机视觉的发展现状如何？"
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n问题 {i}: {question}")
        print("回答: ", end="", flush=True)
        # chain.batch() - 批量处理
        # 1、需要真正的逐字流式 → 使用 stream() 或异步并发
        # 2、需要最高性能的批量处理 → 使用 batch() + streaming=False
        for chunk in chain.stream({"text": question}):
            print(chunk, end="", flush=True)
        print()


def advanced_lcel_features():
    """展示高级 LCEL 特性"""
    print("=== 高级 LCEL 特性 ===")

    from langchain_core.runnables import RunnableLambda, RunnableParallel

    # 创建带有预处理和后处理的链
    chain = (
        # 输入处理
            {"original_text": RunnablePassthrough()}
            | RunnableLambda(lambda x: {"text": f"请详细回答: {x['original_text']}"})

            # 主要处理管道
            | ChatPromptTemplate.from_messages([
        ("system", "You are an expert AI assistant"),
        ("user", "{text}")
    ])
            | ChatOpenAI(
        model="deepseek-chat",
        api_key="sk-1c8dc5bdda034300ac759617ea625722",
        base_url="https://api.deepseek.com/v1",
        streaming=True
    )
            | StrOutputParser()

            # 后处理
            | RunnableLambda(lambda x: f"✨ {x} ✨")
    )

    print("AI: ", end="", flush=True)
    for chunk in chain.stream("机器学习的类型有哪些？"):
        print(chunk, end="", flush=True)
    print("\n" + "=" * 50)


if __name__ == "__main__":
    # 运行同步流式示例
    sync_stream_example()

    # 运行异步流式示例
    asyncio.run(async_stream_example())

    # 运行批量流式示例
    batch_stream_example()

    # 运行高级特性示例
    advanced_lcel_features()
