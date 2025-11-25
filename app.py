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
    market_tool = MarketDataManager().get_tool()
    quant_tool = QuantAnalyzer().get_tool()
    rag_tool = FinancialKnowledgeBase().get_tool()
    
    all_tools = [tool for tool in [market_tool, quant_tool, rag_tool] if tool]

    # 3. 构建 Agent
    agent = ReActAgent(
        tools=all_tools,
        llm=Settings.llm,
        verbose=True,
        system_prompt=AGENT_SYSTEM_PROMPT,
        context=AGENT_CONTEXT_INJECTION,
        memory=ChatMemoryBuffer.from_defaults(token_limit=4096)
    )

    # 4. 交互循环
    print("\n🤖 量化投资顾问已就绪 (输入 'exit' 退出)")
    while True:
        user_input = input("\n👤 用户: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        
        try:
            response = await agent.run(user_input)
            print(f"\n🤖 顾问: {response}")
        except Exception as e:
            logger.error(f"运行出错: {e}")

if __name__ == "__main__":
    asyncio.run(main())