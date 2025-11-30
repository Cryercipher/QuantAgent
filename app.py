import asyncio

from core.agent_runtime import runtime, logger


async def main():
    await runtime.ensure_ready()
    print("\n🤖 量化投资顾问已就绪 (输入 'exit' 退出)")
    while True:
        user_input = input("\n👤 用户: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        try:
            result = await runtime.run_chat(user_input)
            print(f"\n🤖 顾问: {result['answer']}")
        except Exception as exc:
            logger.error("运行出错: %s", exc)
            print("\n🤖 顾问: 系统繁忙，请稍后重试。")

if __name__ == "__main__":
    asyncio.run(main())