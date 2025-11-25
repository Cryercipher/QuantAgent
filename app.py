import asyncio
import os
from datetime import datetime
from importlib import import_module
from llama_index.core.agent import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import Settings, set_global_handler

from core.llm_factory import ModelFactory
from tools.market_data import MarketDataManager
from tools.quant_analysis import QuantAnalyzer
from tools.knowledge_base import FinancialKnowledgeBase
from prompts.system_prompts import AGENT_SYSTEM_PROMPT, AGENT_CONTEXT_INJECTION
from utils.logger import get_logger
from config.settings import PHOENIX_ENABLED, PHOENIX_HOST, PHOENIX_PORT

logger = get_logger("MainApp")


def init_phoenix_monitor():
    if not PHOENIX_ENABLED:
        logger.info("Phoenix 监控已关闭。设置 PHOENIX_ENABLED=true 以启用。")
        return None
    try:
        phoenix_module = import_module("phoenix")
    except Exception as exc:
        logger.warning(f"未安装 phoenix，跳过监控：{exc}")
        return None

    os.environ.setdefault("PHOENIX_HOST", PHOENIX_HOST)
    os.environ.setdefault("PHOENIX_PORT", str(PHOENIX_PORT))
    os.environ.setdefault("PHOENIX_GRPC_PORT", str(PHOENIX_GRPC_PORT))

    try:
        session = phoenix_module.launch_app()
        session_name = getattr(session, "session_name", "N/A")
        set_global_handler("arize_phoenix")
        logger.info(
            f"Phoenix 监控已启动：http://{PHOENIX_HOST}:{PHOENIX_PORT} (session={session_name})"
        )
        return session
    except RuntimeError as exc:
        logger.error(
            "Phoenix 启动失败（可能为端口冲突：HTTP=%s, gRPC=%s）：%s",
            PHOENIX_PORT,
            PHOENIX_GRPC_PORT,
            exc,
        )
        logger.info(
            "可通过设置环境变量 PHOENIX_GRPC_PORT/PHOENIX_PORT 或 PHOENIX_ENABLED=false 来规避。"
        )
        return None
    except Exception as exc:
        logger.error(f"Phoenix 启动失败：{exc}")
        return None


def _get_llm_tokenizer():
    llm = getattr(Settings, "llm", None)
    if llm is None:
        return None
    for attr in ("tokenizer", "_tokenizer"):
        tok = getattr(llm, attr, None)
        if tok is not None:
            return tok
    return None


def _count_tokens(text: str, tokenizer=None) -> int:
    if not text:
        return 0
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text))
        except Exception as exc:
            logger.warning(f"token 统计失败，使用近似值: {exc}")
    # 简单近似：中文文本平均 2 字符 ≈ 1 token
    return max(1, len(text) // 2)

async def main():
    # 1. 初始化模型
    ModelFactory.init_models()

    # 启动 Phoenix 监控（如可用）
    init_phoenix_monitor()

    # 2. 初始化工具
    logger.info("初始化工具箱...")
    tool_cache: dict[str, dict] = {}
    last_focus = {"ts_code": None}

    def cache_tool_result(ts_code: str, category: str, summary: str, metadata=None):
        if not ts_code or not summary:
            return
        metadata = metadata or {}
        entry = tool_cache.setdefault(
            ts_code,
            {"ts_code": ts_code, "summaries": {}, "last_updated": None, "name": metadata.get("name")},
        )
        if metadata.get("name"):
            entry["name"] = metadata["name"]
        updated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
        entry["summaries"][category] = {
            "summary": summary,
            "updated_at": updated_at,
        }
        combined_segments = []
        for cat, payload in entry["summaries"].items():
            combined_segments.append(
                f"{ts_code} | {cat} @ {payload['updated_at']}: {payload['summary']}"
            )
        entry["summary"] = "\n".join(combined_segments)
        entry["last_updated"] = updated_at
        last_focus["ts_code"] = ts_code

    def get_focus_hint() -> str:
        ts_code = last_focus.get("ts_code")
        if not ts_code:
            return ""
        entry = tool_cache.get(ts_code)
        if not entry:
            return ""
        display_name = entry.get("name") or ts_code
        summary = entry.get("summary")
        hint = f"最近聚焦标的：{display_name}（{ts_code}）。"
        if summary:
            hint += f"\n缓存要点：\n{summary}"
        return hint

    def get_cache_snippets(limit: int = 3) -> list[str]:
        if not tool_cache:
            return []
        sorted_entries = sorted(
            tool_cache.values(),
            key=lambda item: item.get("last_updated") or "",
            reverse=True,
        )
        snippets = []
        for entry in sorted_entries[:limit]:
            snippets.append(entry.get("summary", ""))
        return [s for s in snippets if s]

    knowledge_base = FinancialKnowledgeBase()
    market_tool = MarketDataManager(cache_callback=cache_tool_result).get_tool()
    quant_tool = QuantAnalyzer(cache_callback=cache_tool_result).get_tool()

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

    prompt_tokenizer = _get_llm_tokenizer()

    conversation_history = []

    def build_agent_input(user_query: str, rag_context: str) -> str:
        sections = []
        if conversation_history:
            recent_history = conversation_history[-3:]
            formatted_history = "\n".join(
                f"👤用户: {turn['user']}\n🤖顾问: {turn['assistant']}" for turn in recent_history
            )
            sections.append("【历史对话】\n" + formatted_history)
        focus_hint = get_focus_hint()
        if focus_hint:
            sections.append("【上下文提醒】\n" + focus_hint)
        if rag_context:
            sections.append("【知识库检索摘要】\n" + rag_context)
        sections.append("【当前用户问题】\n" + user_query)
        cache_snippets = get_cache_snippets()
        if cache_snippets:
            sections.append("【历史工具缓存】\n" + "\n".join(cache_snippets))
        sections.append(
            "请严格遵循：\n"
            "1) 先将【知识库检索摘要】概括为 1-2 条要点，放入“【理论依据】”，句中标注“知识库”；若只有默认风险准则，也必须引用默认要点再继续。\n"
            "2) 若需要数据，调用 market_data_tool / quant_analysis_tool，并在“【数据洞察】”部分注明来源与结论。\n"
            "3) 在“【顾问建议】”部分整合理论+数据，说明仓位/止损/风险提示，如信息不足需明确说明。"
        )
        return "\n\n".join(sections)

    def build_fallback_response(user_query: str) -> str:
        focus_hint = get_focus_hint()
        cache_snippets = get_cache_snippets(limit=1)
        theory = "【理论依据】知识库响应出现异常，临时沿用默认风险控制准则（分散配置、设定止损、控制杠杆）。"
        if focus_hint:
            theory += f" 近期上下文：{focus_hint.replace(chr(10), ' ')}"
        if cache_snippets:
            data = "【数据洞察】暂无新增工具结果，沿用缓存：\n" + cache_snippets[0]
        else:
            data = "【数据洞察】当前缺少可复用的行情/量化数据，请稍后重试工具查询。"
        advice = (
            "【顾问建议】暂无法完成完整分析，建议先根据默认风控原则评估问题："
            f"“{user_query}”。若需即时结果，请重试或缩短提问，并可指定标的与期望仓位。"
        )
        return "\n".join([theory, data, advice])

    # 4. 交互循环
    print("\n🤖 量化投资顾问已就绪 (输入 'exit' 退出)")
    while True:
        user_input = input("\n👤 用户: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        
        try:
            rag_context = knowledge_base.query_raw(user_input)

            enriched_input = build_agent_input(user_input, rag_context)
            token_count = _count_tokens(enriched_input, tokenizer=prompt_tokenizer)
            logger.info(
                f"[PromptStats] tokens={token_count} chars={len(enriched_input)} history_turns={len(conversation_history)}"
            )
            try:
                response = await asyncio.wait_for(agent.run(enriched_input), timeout=90)
            except asyncio.TimeoutError:
                logger.error("Agent 响应超时，返回兜底建议。")
                response = build_fallback_response(user_input)
            except Exception as agent_exc:
                logger.error(f"Agent 执行异常: {agent_exc}")
                response = build_fallback_response(user_input)
            conversation_history.append({"user": user_input, "assistant": str(response)})
            if len(conversation_history) > 5:
                conversation_history.pop(0)
            print(f"\n🤖 顾问: {response}")
        except Exception as e:
            logger.error(f"运行出错: {e}")
            fallback = build_fallback_response(user_input)
            conversation_history.append({"user": user_input, "assistant": fallback})
            if len(conversation_history) > 5:
                conversation_history.pop(0)
            print(f"\n🤖 顾问: {fallback}")

if __name__ == "__main__":
    asyncio.run(main())