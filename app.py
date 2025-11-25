import asyncio
from llama_index.core.agent import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import Settings

from core.llm_factory import ModelFactory
from tools.market_data import MarketDataManager
from tools.quant_analysis import QuantAnalyzer
from tools.knowledge_base import FinancialKnowledgeBase
from prompts.system_prompts import AGENT_SYSTEM_PROMPT, AGENT_CONTEXT_INJECTION
from utils.logger import get_logger

logger = get_logger("MainApp")

async def main():
    # 1. 初始化模型
    ModelFactory.init_models()

    # 2. 初始化工具
    logger.info("初始化工具箱...")
    knowledge_base = FinancialKnowledgeBase()
    market_tool = MarketDataManager().get_tool()
    quant_tool = QuantAnalyzer().get_tool()
    
    all_tools = [tool for tool in [market_tool, quant_tool] if tool]

    # 3. 构建 Agent
    agent = ReActAgent(
        tools=all_tools,
        llm=Settings.llm,
        verbose=True,
        system_prompt=AGENT_SYSTEM_PROMPT,
        context=AGENT_CONTEXT_INJECTION,
        memory=ChatMemoryBuffer.from_defaults(token_limit=4096)
    )

    conversation_history = []

    def build_agent_input(user_query: str, rag_context: str) -> str:
        sections = []
        if conversation_history:
            recent_history = conversation_history[-3:]
            formatted_history = "\n".join(
                f"👤用户: {turn['user']}\n🤖顾问: {turn['assistant']}" for turn in recent_history
            )
            sections.append("【历史对话】\n" + formatted_history)
        if rag_context:
            sections.append("【financial_theory_tool 检索摘要】\n" + rag_context)
        sections.append("【当前用户问题】\n" + user_query)
        sections.append("请先依据理论摘要建立即时观点，再视需要调用 market_data_tool 或 quant_analysis_tool，最后以专业但易懂的投资顾问口吻输出结论。")
        return "\n\n".join(sections)

    # 4. 交互循环
    print("\n🤖 量化投资顾问已就绪 (输入 'exit' 退出)")
    while True:
        user_input = input("\n👤 用户: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        
        try:
            rag_context = knowledge_base.query_raw(user_input)

            enriched_input = build_agent_input(user_input, rag_context)
            response = await agent.run(enriched_input)
            conversation_history.append({"user": user_input, "assistant": str(response)})
            if len(conversation_history) > 5:
                conversation_history.pop(0)
            print(f"\n🤖 顾问: {response}")
        except Exception as e:
            logger.error(f"运行出错: {e}")

if __name__ == "__main__":
    asyncio.run(main())