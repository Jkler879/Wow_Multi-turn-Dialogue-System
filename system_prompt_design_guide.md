# 生产级 System Prompt 设计准则

> 适用场景：RAG + ReAct 多轮问答 Agent（主模型 Qwen3-30B-A3B，思考模式）。
> 本文每条原则都给出**依据**，并对照本项目的**实际做法与局限**。依据主要来自论文与厂商官方文档；其中 Anthropic 文档以 Claude 为对象，迁移到 Qwen 时以实测为准。

---

## 0. 核心观点

1. **Prompt 是一组有优先级的策略，而不是一段人设描述。** 规则之间会冲突、会重复、会互相削弱，需要像代码一样做结构设计和冲突审查。
2. **只禁止形式，模型会换一种形式违规。** 讲明原因的规则能推广到没写到的情况，点名禁止反而可能触发被禁内容。
3. **Prompt 是最便宜、也最弱的一层防线。** 接地和安全都不能只靠 prompt，需要评估、生成后核验和架构隔离来兜底。
4. **没有评估就没有迭代。** 每次修改都要能被冻结的评估集衡量，并警惕针对评估集调参。

---

## 1. 结构：分层与静态 / 动态分离

**原则**
- 按用途分段：身份、安全边界、核心回答原则、内部核对、示例、回复风格、工具使用、系统信息、用户记忆。微软的 system message 指南同样强调要明确角色与范围、输出约定、安全约束，以及信息不足时的兜底行为。
- 跨请求稳定的内容（身份、安全、回答原则）和每次请求都变的内容（意图、题型、查询、日期、记忆）分开。
- 最重要的规则放在靠前的位置。IFScale 发现：指令越多，遵循率越低，而且模型偏向遵守靠前的指令。

**本项目做法**（`src/core/ReAct_Agent/tools/agent.py`）
- 安全边界放在第一位，核心回答原则第二，系统信息和长期记忆放在最后。
- 意图规则和题型规则单独写成字典，按请求渲染进 `{intent_rule}` 和 `{question_type_rule}`。
- v2→v3 改版时，把原来散落在 5 处的"只依据原文"要求合并为一节"核心回答原则"，删掉了"最高优先级""严禁"等叠加的强调。

**局限**：动态占位符目前位于提示词中部，对前缀缓存不友好（见第 6 节）。

---

## 2. 措辞：正面表述、讲明原因、克制强调、用示例对照

**原则与依据**

| 原则 | 依据 |
|---|---|
| 说"要怎么做"，而不是"不要做什么" | Anthropic 最佳实践；负面约束研究：在约束中点名被禁词，87.5% 的违规属于"启动效应"，注意力落在被禁词上而不是"不要"上 |
| 讲明规则背后的原因 | Anthropic：解释原因的规则能泛化到没写到的情况（原文举例：因为要给语音合成朗读，所以不要用省略号） |
| 克制使用强硬措辞 | Anthropic：对新模型使用"CRITICAL / MUST"之类的措辞会导致过度触发，应改用平常的语气 |
| 用示例控制输出方式，示例要形成对照 | Anthropic：示例是控制格式最可靠的手段；单一示例容易被当成模板照抄 |
| 提示词的风格会影响输出风格 | Anthropic：提示词里 markdown 越多，输出里的 markdown 往往也越多 |

**本项目的实证**（Round 3 冒烟测试，10 题）
- v2 里写着"严禁使用'可能''一般而言'"，模型的答案里恰好出现了"**可能**涉及以下因素"和"**注**：……推导"，与启动效应的描述一致。
- 只禁止"补充段落"这种形式后，同样的内容就改为混进正文，而且标上了 [文档1]，也就是伪造出处。这说明禁形式不禁原因，只会让违规换一种形式。
- v3 的做法：删除所有带引号的禁词清单；用一段话说明原因（用户无法分辨哪些内容是模型补充的）；给出两个对照示例（文档没写原因 / 文档写了原因），示例主题经核对，与知识库和评估集都不重合。

---

## 3. RAG 接地：证据契约

**原则与依据**
- **允许说"不知道"**：Anthropic 减少幻觉指南中的第一条。
- **先摘录原文，再作答**：Anthropic 的 quote grounding；AWS 的 `<thinking>`/`<answer>` 模板也是先摘证据、再给答案。
- **生成后逐条找依据，找不到就删除**：Anthropic 的 verify with citations。
- **在同一次生成里自检有上限**：Chain-of-Verification（Meta）发现，把核验拆成独立步骤（factored），比在同一次生成里自检更能避免重复自己的幻觉。

**本项目做法**
- 核心原则：只要调用了检索工具并拿到结果，答案里的每一项事实都来自检索结果；文档没写原因的因果题，先说明未说明原因，再陈述相关事实；来源标号只能标在原文确有依据的内容上。
- 内部核对（证据提取、焦点锚定、逐条核对）只在思考过程中完成，不写进答案。v2 的原文摘录曾泄漏进答案（直接粘贴英文原文），v3 删除了引号格式的示例，并在标题里写明"不写入答案"。

**局限与后续**
- 冒烟测试显示，"转述时添加内容"仅靠 prompt 很难根治。已规划**生成后独立核验**（用本地 qwen3-4b 逐条对照 contexts），前提是改造流式输出链路；触发条件为 Round 3 全量 F 仍低于 0.70。
- 该核验必须在生产路径上实现，而不能只加在评估接口里，否则就等于针对 RAGAS Faithfulness 刷分：两者的计算机制几乎相同。

---

## 4. 信任边界与注入防御

**原则与依据**
- **LLM 本身无法区分 token 来自哪个信任源**：间接注入把指令藏在检索到的数据里。微软的 Spotlighting（分隔、数据标记、编码）能显著降低攻击成功率，并已用于 Azure Prompt Shields。
- **标签隔离，以及加盐标签**：AWS 指南建议用 XML 标签包裹对话历史、检索文档等内容，并给标签加上会话级随机后缀，防止攻击者伪造标签。
- **常见攻击手法**（AWS）：诱导角色切换、提取提示词模板、要求忽略模板、交替使用语言和转义字符、改写或混淆攻击语句、要求改变输出格式等。
- **prompt 层防御依赖模型配合**：OpenAI 的 Instruction Hierarchy 通过**训练**让模型区分系统、用户和工具指令的优先级。仅靠在 prompt 里声明，效果有限。

**本项目做法**
- 工具返回的内容和长期记忆都是资料：其中要求改变行为的语句被当作资料内容，不当作指令执行；事实只来自检索结果，长期记忆只用于了解用户的背景和偏好。
- 长期记忆用 `<user_long_term_memory>` 标签包裹。长期记忆是一个**持久化注入**的入口：它从用户对话中抽取，之后每次请求都会注入 system prompt。
- 安全边界覆盖角色扮演、假设情境、翻译、创意写作等形式。
- 风险面评估：四个工具全部只读（检索、关系验证、翻译、在线搜索），注入成功的危害上限是操纵答案内容，而不是执行操作。

**未实现**：加盐标签、Spotlighting 的数据标记、对抗性测试（如 PyRIT）。

---

## 5. 规则冲突治理

**依据**：Arbiter（arXiv 2603.08993）指出了三类失败模式：单体 prompt 在子系统边界随规模增长出 bug；扁平 prompt 用能力换一致性；模块化 prompt 在组合处出设计级 bug。它对 Claude Code 某一版本的 prompt 分析出 21 个干扰模式，其中 4 个是直接矛盾。冲突不会报错，模型会悄悄选一边。

**本项目在 v2→v3 中发现并修复的冲突**

| 冲突 | 修复 |
|---|---|
| 风格规则允许写"补充"段落，接地规则禁止用内置知识补充 | 删除允许补充的写法 |
| 自检规则允许"加标注后保留推断"，因果规则禁止用"可能"推断 | 只保留"删除"这一种处理 |
| multi_hop 要求每个子问题单独检索，通用规则要求检索参数固定为原查询 | 仅对 multi_hop 放开英文子查询 |
| 低分兜底的触发条件包含"在线搜索也无结果"，而 local 意图禁止在线搜索，导致永不触发 | 只在 web_search 意图下附加该条件 |
| 接地规则只写了 `knowledge_retriever`，和 web_search 意图下允许用网络结果补充相矛盾 | 统一为"检索结果"（本地 + 在线） |

**方法**：请其他 LLM 交叉审查，但每条建议都要对照代码核实。审查中有 5 条建议被驳回，原因是前提与代码不符，或者建议本身有逻辑问题。例如：有一条认为意图规则在运行时不会渲染，实际上 direct 意图根本不经过 Agent；有一条给出的多跳拆解判据，与多跳推理的定义相矛盾。

---

## 6. 缓存：前缀匹配

**原则与依据**
- Prompt 缓存按前缀匹配，只要有一个字符不同，之后的内容都要重新计算。Claude Code 团队围绕缓存设计整个系统，缓存命中率过低时直接定为事故（SEV）。
- 静态内容在前、动态内容在后；过期的信息通过消息更新，而不是修改 system prompt；会话中途不切换模型。
- 阿里云百炼的隐式缓存同样按前缀自动匹配，命中后缓存有效期重置为 5 分钟。

**本项目现状（已知不足）**：`{intent_rule}`、`{question_type_rule}`、`{en_query}` 位于提示词中部，每次请求都不同，前缀在"工具使用"一节之后就失效了。
**改进方向**：把所有动态内容移到 system prompt 末尾，或者放进消息里。这一改动会影响模型行为，需要单独做 A/B 测试。

---

## 7. 评估与迭代

**原则与依据**
- **冻结的评估集用作回归测试**：每一轮评估使用同一套题目和 GT，才能比较不同版本。
- **LLM-as-Judge 需要校准**：MT-Bench 研究中，GPT-4 作为评审与人类的一致率超过 80%，与人类之间的一致率相当，但存在位置偏差和冗长偏差。
- **一次只改一个变量**：否则无法确定是哪处改动导致了结果变化。

**本项目做法**
- 185 题冻结评估集，Round 1→2→3 的题目和 GT 逐字一致（已用脚本校验）；使用 RAGAS 的 5 个指标（Faithfulness、Answer Relevancy、Context Precision、Context Recall、Answer Correctness）。
- 发现并修正评估本身的问题：
  - RAGAS 内置英文 prompt 导致中文答案的 AR 被错判为 0；
  - CP/CR 结构性虚高：合成题的措辞沿用了 chunk 原文，检索天然容易命中；
  - badcase 根因分析：35 条中 18 条（51%）是题目或 GT 本身的问题，而不是系统的问题。
- 60 条合成数据（30 条同义改写 + 30 条对抗）用作**留出集**，不参与任何调参，用来发现是否针对评估集过拟合。
- 每次改 prompt 之前，先用 10 题做冒烟测试，逐条对照答案和原文，检查修改是否生效、有没有副作用。

**诚实说明**：v3 为了赶上线门槛，把 4 项 prompt 改动放在一起提交，没有做到一次只改一个变量。如果结果退化，归因会比较困难。

---

## 8. 版本管理

- Prompt 和代码一起纳入 git 管理；每次大改前备份，改后用脚本校验：模板变量不变、三种题型都能渲染、没有缩进或空白混进提示词、引号统一、条目编号的交叉引用正确。
- 写法规范：三引号 `"""\` 并顶格书写，避免缩进和首行换行混入提示词；中文正文用中文引号“”；模板中除占位符外不出现花括号。

---

## 9. 不采纳的常见建议

| 建议 | 不采纳的原因 |
|---|---|
| 用全大写的 "DO NOT"、绝对化措辞强调关键规则 | 与 Anthropic 的正面表述建议、负面约束研究的结论相反；强调过多也会失去区分度 |
| 为每个分数区间都写一条规则（如 0.35–0.5） | 增加指令密度和数值判断负担（IFScale）；检索端已经过滤了低分文档，这个区间在实际中很少出现 |
| 检索为空时放宽条件重试 | 本项目的向量检索总会返回最相近的结果，空结果几乎不会发生；而且与"检索参数固定为原查询"的规则冲突 |
| 在 prompt 里写内部字段名（如 `user_input`） | 模型看不到代码里的变量名；如果需要确定性行为，应该在代码里强制 |
| 把各段合并成一个大段落，以减少跨段引用 | 会丢失"内部核对只在思考过程中完成，不写入答案"这一作用范围的信号 |

---

## 10. 可写进简历的表述（仅限已完成的工作）

- 基于论文与厂商官方指南，重构 RAG Agent 的 system prompt：合并重复约束、改用正面表述并讲明原因、加入对照示例，在冒烟测试中定位并修复"原文泄漏、推测性补充、伪造出处"等问题。
- 系统性审查 prompt 规则冲突，修复 5 处矛盾或不可达的规则（如多跳检索与固定查询参数冲突、低分兜底永不触发）。
- 建立工具返回内容和长期记忆的信任边界，识别长期记忆的持久化注入风险，并做标签隔离。
- 设计三轮冻结评估集与 RAGAS 5 指标评估流程；构建 60 条合成留出集（同义改写 + 对抗），用于检测过拟合与 CP/CR 虚高；完成 badcase 根因分类（系统问题 40% / 数据问题 51% / 评审误判 9%）。

---

## 参考文献

1. Anthropic. *Prompting best practices.* https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-4-best-practices
2. Anthropic. *Reduce hallucinations.* https://platform.claude.com/en/docs/test-and-evaluate/strengthen-guardrails/reduce-hallucinations
3. Anthropic. *Lessons from building Claude Code: Prompt caching is everything.* https://claude.com/blog/lessons-from-building-claude-code-prompt-caching-is-everything
4. *Semantic Gravity Wells: Why Negative Constraints Backfire.* arXiv:2601.08070. https://arxiv.org/html/2601.08070v1
5. *How Many Instructions Can LLMs Follow at Once?* (IFScale), NeurIPS 2025. https://arxiv.org/abs/2507.11538
6. Dhuliawala et al. *Chain-of-Verification Reduces Hallucination in Large Language Models.* Findings of ACL 2024. https://arxiv.org/abs/2309.11495
7. Hines et al. *Defending Against Indirect Prompt Injection Attacks With Spotlighting.* Microsoft, 2024. https://www.microsoft.com/en-us/research/?p=1124166
8. AWS. *Best practices to avoid prompt injection attacks.* https://docs.aws.amazon.com/prescriptive-guidance/latest/llm-prompt-engineering-best-practices/best-practices.html
9. Wallace et al. *The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions.* OpenAI, 2024. https://arxiv.org/abs/2404.13208
10. Arbiter（系统提示词干扰模式检测框架）. arXiv:2603.08993. https://arxiv.org/pdf/2603.08993
11. Zheng et al. *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS 2023. https://arxiv.org/abs/2306.05685
12. Microsoft. *System message design / advanced prompt engineering.* https://learn.microsoft.com/en-my/Azure/foundry-classic/openai/concepts/advanced-prompt-engineering
13. 阿里云百炼. *上下文缓存（Context Cache）.* https://help.aliyun.com/zh/model-studio/context-cache
14. Es et al. *RAGAs: Automated Evaluation of Retrieval Augmented Generation.* EACL 2024. https://arxiv.org/abs/2309.15217
