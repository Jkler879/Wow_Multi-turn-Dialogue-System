"""
ReAct Agent 模块 (ReAct Agent Module)

该模块基于 LangGraph 框架实现了一个生产级的 ReAct Agent，自主决策多轮对话中的复杂推理和多工具调用。
采用 LanGraph 开发优势在于将智能体建模为一个可组合的状态图，可以对多步推理、工具调用、循环控制进行精细化编排。
通过循环执行“思考-行动-观察”的过程自主选择工具集执行，最终生成答案。

同时 LanGraph 内置状态持久化，保证每一轮输入输出不丢失、可监控。

主要功能：
    1. 状态管理和持久化：定义 AgentState 包含查询、消息历史、步数、最大步数和最终答案，支持通过 add_messages 注解自动合并消息。

    2. 系统提示词管理：设计结构化的 System Prompt:
       - 核心规则（输出格式、工具调用语言、工具输入要求）
       - 行为准则 (基于检索到的事实、长短期记忆上下文理解、无效循环)
       - 决策流程 (检索优先、明确的最终回复标识符)
       - 重要提醒 (json输出格式、禁止插入自主分析)

       Agent执行前会动态注入每个工具的自然语言描述 (tools_descriptions)

    3. LLM 输出解析与修复：使用 Pydantic 模型定义期望的输出结构（工具调用或最终答案），
       并借助 OutputFixingParser 实现自动格式错误修复，极大提升鲁棒性。

    4. Agent 节点：负责调用 LLM 生成下一步行动或最终答案，每轮皆包含步数检查，
       并返回原始消息供后续解析。

    5. 工具节点：解析 Agent 输出，执行具体工具调用，并将结果以 `HumanMessage`
       形式返回给 Agent，实现“观察”步骤。

    6. 循环控制：根据当前状态决定下一步是继续调用工具、输出最终答案还是强制停止，
       并内置最大步数 5 保护，防止无限循环。

    7. 最终答案提取：从状态中提取最终答案，并返回结构化结果。

容错处理：
    - 循环步数上限保护：超过 max_steps=5 后强制停止，返回默认答案。
    - 输出格式异常处理：利用 OutputFixingParser 自动修复格式错误，若修复失败默认再修复一次，再次失败则强制终止。
    - 工具调用失败处理：捕获工具执行异常，将错误信息返回给 Agent 作为观察结果。

版本记录：
    - 1.0.0 (2026-03-07): 初始版本，基于 LangGraph 构建基础 ReAct Agent。
    - 1.1.0 (2026-03-08): 引入 Pydantic 结构化输出和 `OutputFixingParser`，增强格式鲁棒性。

Author: Ke Meng
Created: 2026-01-15
Last Modified: 2026-03-18
"""

import json
import logging
from typing import TypedDict, List, Any, Literal, Optional, Sequence, Annotated, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain_classic.output_parsers import OutputFixingParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langgraph.store.base import BaseStore
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

logger = logging.getLogger(__name__)


# ==================== 状态定义 ====================
class AgentState(TypedDict):
    """ReAct Agent 状态（含循环计数）"""
    query: str  # 改写后的查询（前置模块传入）
    messages: Annotated[Sequence[BaseMessage], add_messages]  # 消息历史列表
    step_count: int  # 当前循环步数
    max_steps: int  # 最大允许步数
    final_answer: Optional[str]                # 最终答案（用于提取）


# ==================== 系统提示词 ====================
SYSTEM_PROMPT = """你是一个专业的多轮对话助手，可以调用以下工具来帮助用户解决问题：

{tools_descriptions}

---

## 核心规则

### 1. 输出格式(严格遵循)
- **调用工具时**（每行独立）：
    Action: 工具名
    Action Input: 有效的 JSON 对象

- **直接回答时**：
    Final Answer: 你的最终答案（必须用中文）

- **错误示例（包含分析）**：
    检索结果中，第一条提到了《Fringe》... 因此我推荐你观看《Fringe》。（这是错误的，必须直接输出Final Answer）


### 2. 工具调用语言
- `knowledge_retriever` 的 `query` 参数**必须使用英文**。直接使用用户问题（已由前置模块改写为英文），**严禁自行翻译**。

### 3. 工具输入要求
- 每个工具的详细输入格式已在工具描述中说明，请严格遵循。
- **特别提醒**：`relation_verifier` 的实体必须从已检索到的文档中提取，而非用户问题。

---

## 行为准则

1. **基于事实**：所有回答必须严格依据工具返回的结果，严禁编造信息。
2. **语言要求**：最终答案必须用中文。若生成的答案是英文，调用 `translator` 翻译。
3. **多轮连贯**：结合对话历史理解用户意图，保持上下文一致。
4. **处理不确定性**：
   - **检索结果为空**（工具返回 `[]`）：直接输出 `Final Answer: 未找到相关信息。`，**不要再次调用工具**。
   - **检索结果有文档但分数较低**（例如最高分 < 0.2）：如实告知用户“以下信息可能不完全相关，仅供参考”，然后基于这些信息谨慎回答。
   - **分数较高时**（例如最高分 > 0.5）：可以自信地依据高分文档回答。
   - **验证结果置信度低**：说明“可能存在但不十分确定”，并引用证据原文。
5. **避免无效循环**：连续两次同一工具结果为空，请直接输出最终答案。
6. **错误处理**：工具执行失败时，如果输出格式错误，你将收到“格式错误”提示，并消耗步数。请严格遵守格式，避免无效循环。

---

## 决策流程

1. **检索优先**：首先调用 `knowledge_retriever`。
2. **评估是否需要验证**：如果检索结果包含实体且用户问题涉及实体间关系，调用 `relation_verifier`；否则直接回答。
3. **回答**：
   - 若检索结果足够，生成最终答案。
   - 若答案含有英文，调用 `translator` 翻译后输出。
   - 若检索结果为空或分数极低，按行为准则第4条处理。
4. **终止条件**：输出 `Final Answer` 后对话结束。

---

## 重要提醒

- **JSON 必须有效**：工具输入必须是合法的 JSON，例如 {{"query": "..."}}，不要使用单引号或缺少引号。
- **禁止输出任何解释**：除了 `Action:` / `Final Answer:` 行，严禁输出任何思考过程、分析或多余文本。
- **遵循工具用途**：只调用适合当前场景的工具，不要滥用。
- **工具调用后必须立即输出**：收到工具返回后，你必须直接输出 `Final Answer:` 或下一次 `Action:`，不得插入分析。

请严格按照以上规则进行思考与输出。"""


# ==================== 构建提示模板 ====================
def create_agent_prompt() -> ChatPromptTemplate:
    """创建包含 tools_descriptions 占位符的提示模板"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),  # 此时 SYSTEM_PROMPT 中有 {tools_descriptions}
        MessagesPlaceholder(variable_name="messages"),
        ("human", "继续你的思考。")
    ])
    return prompt


# ==================== 创建 Agent 主函数 ====================
def create_react_agent(llm: BaseChatModel, tools: List[BaseTool], store: BaseStore = None) -> Any:
    """
    创建 LangGraph ReAct Agent
    :param llm: LLM 实例
    :param tools: 工具列表
    :param store: 存储后端（用于长期记忆）
    :return: 编译后的图应用
    """
    logger.info(f"创建 ReAct Agent，LLM: {llm.__class__.__name__}, 工具数量: {len(tools)}")
    for tool in tools:
        logger.debug(f"注册工具: {tool.name}")

    tools_dict = {tool.name: tool for tool in tools}
    tools_descriptions = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])
    prompt = create_agent_prompt()

    # ==================== Pydantic 模型用于解析 ====================
    class AgentActionModel(BaseModel):
        """工具调用模型"""
        tool: str = Field(description="工具名称")
        tool_input: dict = Field(description="工具输入，必须是字典格式")

    class AgentFinishModel(BaseModel):
        """最终答案模型"""
        output: str = Field(description="最终答案")

    class AgentOutput(BaseModel):
        """Agent 输出模型：要么调用工具，要么输出最终答案"""
        action: Optional[AgentActionModel] = Field(default=None, description="如果调用工具，填写此项")
        finish: Optional[AgentFinishModel] = Field(default=None, description="如果结束，填写此项")

    # ==================== 带修复功能的解析器 ====================
    def _parse_with_fixing(output: str) -> Union[AgentAction, AgentFinish, None]:
        """
        使用 OutputFixingParser 解析 LLM 输出，转换为 AgentAction 或 AgentFinish
        返回 None 表示无法解析
        """
        base_parser = PydanticOutputParser(pydantic_object=AgentOutput)
        fixing_parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)

        try:
            parsed: AgentOutput = fixing_parser.parse(output)
            if parsed.action:
                return AgentAction(tool=parsed.action.tool, tool_input=parsed.action.tool_input, log=output)
            elif parsed.finish:
                return AgentFinish(return_values={"output": parsed.finish.output}, log=output)
            else:
                logger.error("解析结果既无 action 也无 finish")
                return None
        except OutputParserException as e:
            logger.error(f"修复解析器最终失败: {e}")
            return None

    # ==================== Agent 节点 ====================
    def agent_node(state: AgentState) -> dict:
        """Agent 节点：调用 LLM 生成下一步行动或最终答案"""
        logger.info(f"🤖 Agent 节点开始，当前步数: {state.get('step_count', 0)}")
        if logger.isEnabledFor(logging.DEBUG):
            messages = state["messages"]
            logger.debug(f"输入消息数量: {len(messages)}")
            if messages:
                last = messages[-1].content[:200] if len(messages[-1].content) > 200 else messages[-1].content
                logger.debug(f"最后一条消息预览: {last}")

        # 步数检查
        if state.get("step_count", 0) >= state.get("max_steps", 5):
            logger.warning(f"步数超限强制结束: {state.get('step_count')} >= {state.get('max_steps')}")
            return {
                "messages": [AIMessage(content="Final Answer: 已达到最大尝试次数，无法获得有效答案。")],
                "step_count": state["step_count"] + 1
            }

        messages = state["messages"]
        if not messages:
            messages = [HumanMessage(content=state["query"])]

        chain = prompt | llm
        logger.info("调用 LLM 生成下一步...")
        response = chain.invoke({
            "messages": messages,
            "tools_descriptions": tools_descriptions
        })
        ai_message = response.content
        logger.info(f"LLM 原始输出 (长度 {len(ai_message)}): {ai_message[:200]}{'...' if len(ai_message) > 200 else ''}")

        # 返回原始消息，解析由后续节点负责
        return {
            "messages": [AIMessage(content=ai_message)],
            "step_count": state.get("step_count", 0) + 1
        }

    # ==================== 工具节点 ====================
    def tool_node(state: AgentState) -> dict:
        """执行工具调用，返回观察结果（作为 HumanMessage）"""
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            return {"messages": []}

        parsed = _parse_with_fixing(last_message.content)
        if parsed is None:
            error_msg = "解析工具调用失败"
            logger.error(error_msg)
            return {"messages": [HumanMessage(content=f"观察结果: {error_msg}")]}

        if isinstance(parsed, AgentFinish):
            # 理论上不会发生，但安全处理
            return {"messages": []}
        elif not isinstance(parsed, AgentAction):
            logger.error(f"非AgentAction类型: {type(parsed)}")
            return {"messages": []}

        tool_name = parsed.tool
        tool_input = parsed.tool_input
        logger.info(f"🔧 工具调用开始: {tool_name}, 输入参数: {tool_input}")

        if tool_name not in tools_dict:
            error_msg = f"未知工具: {tool_name}"
            logger.error(error_msg)
            return {"messages": [HumanMessage(content=f"观察结果: {error_msg}")]}

        try:
            result = tools_dict[tool_name].invoke(tool_input)
            content = json.dumps(result, ensure_ascii=False) if isinstance(result, dict) else str(result)
            logger.info(f"✅ 工具调用成功: {tool_name}, 结果预览: {content[:200]}{'...' if len(content) > 200 else ''}")
        except Exception as e:
            content = f"工具执行出错: {e}"
            logger.error(f"❌ 工具调用失败: {tool_name}, 错误: {e}")

        return {"messages": [HumanMessage(content=f"观察结果: {content}")]}

    # ==================== 循环控制 ====================
    def should_continue(state: AgentState) -> Literal["continue", "stop", "force_stop"]:
        step = state.get("step_count", 0)
        max_steps = state.get("max_steps", 5)
        logger.info(f"当前步数: {step}, 最大步数: {max_steps}")

        if step >= max_steps:
            logger.warning("达到最大步数，强制停止")
            return "force_stop"

        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            logger.debug("上一条消息非AIMessage，继续")
            return "continue"

        parsed = _parse_with_fixing(last_message.content)
        if parsed is None:
            logger.error("解析LLM输出失败，强制停止")
            return "force_stop"

        if isinstance(parsed, AgentFinish):
            logger.info("检测到 Final Answer，停止")
            return "stop"
        elif isinstance(parsed, AgentAction):
            logger.info("检测到工具调用，继续")
            return "continue"
        else:
            logger.warning(f"未知解析类型: {type(parsed)}，强制继续")
            return "continue"

    # ==================== 提取最终答案 ====================
    def extract_final_answer(state: AgentState) -> dict:
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage):
            parsed = _parse_with_fixing(last_message.content)
            if parsed is not None and isinstance(parsed, AgentFinish):
                final = parsed.return_values.get("output", "")
                logger.info(f"最终答案: {final[:200]}{'...' if len(final) > 200 else ''}")
                return {"final_answer": final}
        logger.warning("无法提取最终答案，返回默认值")
        return {"final_answer": "抱歉，无法获得有效答案，请稍后重试。"}

    # ==================== 构建图 ====================
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tool", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tool",
            "stop": "final",
            "force_stop": "final"
        }
    )
    graph.add_edge("tool", "agent")
    graph.add_node("final", extract_final_answer)
    graph.add_edge("final", END)

    return graph.compile(store=store)