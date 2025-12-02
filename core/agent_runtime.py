import asyncio
import os
from datetime import datetime
from importlib import import_module
from typing import Any, Dict, List, Optional

from llama_index.core import Settings, set_global_handler
from llama_index.core.agent import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer

from config.settings import PHOENIX_ENABLED, PHOENIX_HOST, PHOENIX_PORT
from core.llm_factory import ModelFactory
from prompts.system_prompts import AGENT_SYSTEM_PROMPT
from tools.chart_visualizer import CandlestickChartTool
from tools.knowledge_base import FinancialKnowledgeBase
from tools.quant_analysis import MarketInsightTool
from utils.logger import get_logger

logger = get_logger("AgentRuntime")


def init_phoenix_monitor():
    if not PHOENIX_ENABLED:
        logger.info("Phoenix 监控已关闭。设置 PHOENIX_ENABLED=true 以启用。")
        return None
    try:
        phoenix_module = import_module("phoenix")
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning(f"未安装 phoenix，跳过监控：{exc}")
        return None

    os.environ.setdefault("PHOENIX_HOST", PHOENIX_HOST)
    os.environ.setdefault("PHOENIX_PORT", str(PHOENIX_PORT))

    try:
        session = phoenix_module.launch_app()
        session_name = getattr(session, "session_name", "N/A")
        set_global_handler("arize_phoenix")
        logger.info(
            "Phoenix 监控已启动：http://%s:%s (session=%s)",
            PHOENIX_HOST,
            PHOENIX_PORT,
            session_name,
        )
        return session
    except RuntimeError as exc:
        logger.error(
            "Phoenix 启动失败（可能为端口冲突：HTTP=%s）：%s",
            PHOENIX_PORT,
            exc,
        )
        logger.info(
            "可通过设置环境变量 PHOENIX_PORT 或 PHOENIX_ENABLED=false 来规避。"
        )
        return None
    except Exception as exc:  # pragma: no cover - unexpected
        logger.error("Phoenix 启动失败：%s", exc)
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
        except Exception as exc:  # pragma: no cover - tokenizer edge cases
            logger.warning("token 统计失败，使用近似值: %s", exc)
    return max(1, len(text) // 2)


class AgentRuntime:
    def __init__(self):
        self._agent: Optional[ReActAgent] = None
        self._prompt_tokenizer = None
        self.tool_cache: Dict[str, Dict[str, Any]] = {}
        self.last_focus: Dict[str, Optional[str]] = {"ts_code": None}
        self._phoenix_session = None

    async def ensure_ready(self):
        if self._agent is not None:
            return
        logger.info("初始化模型与工具...")
        ModelFactory.init_models()
        self._phoenix_session = init_phoenix_monitor()
        self._setup_agent()

    def _setup_agent(self):
        knowledge_base = FinancialKnowledgeBase()
        theory_tool = knowledge_base.get_tool()
        
        market_insight = MarketInsightTool(cache_callback=self._cache_tool_result)
        market_tool = market_insight.get_tool()
        search_tool = market_insight.get_search_tool() # 新增搜索工具
        
        chart_tool = CandlestickChartTool(
            cache_callback=self._cache_tool_result
        ).get_tool()
        
        tools = [tool for tool in [theory_tool, market_tool, search_tool, chart_tool] if tool]
        self._agent = ReActAgent(
            tools=tools,
            llm=Settings.llm,
            verbose=True,
            system_prompt=AGENT_SYSTEM_PROMPT,
            memory=ChatMemoryBuffer.from_defaults(token_limit=4096),
        )
        self._prompt_tokenizer = _get_llm_tokenizer()

    def _cache_tool_result(
        self, ts_code: str, category: str, summary: str, metadata: Optional[dict] = None
    ):
        if not ts_code or not summary:
            return
        metadata = metadata or {}
        entry = self.tool_cache.setdefault(
            ts_code,
            {
                "ts_code": ts_code,
                "summaries": {},
                "last_updated": None,
                "name": metadata.get("name"),
            },
        )
        if metadata.get("name"):
            entry["name"] = metadata["name"]
        updated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
        entry["summaries"][category] = {"summary": summary, "updated_at": updated_at}
        segments = []
        for cat, payload in entry["summaries"].items():
            segments.append(
                f"{ts_code} | {cat} @ {payload['updated_at']}: {payload['summary']}"
            )
        entry["summary"] = "\n".join(segments)
        entry["last_updated"] = updated_at
        self.last_focus["ts_code"] = ts_code

    def get_focus_hint(self) -> str:
        ts_code = self.last_focus.get("ts_code")
        if not ts_code:
            return ""
        entry = self.tool_cache.get(ts_code)
        if not entry:
            return ""
        display_name = entry.get("name") or ts_code
        summary = entry.get("summary")
        hint = f"最近聚焦标的：{display_name}（{ts_code}）。"
        if summary:
            hint += f"\n缓存要点：\n{summary}"
        return hint

    def get_focus_entry(self) -> Optional[Dict[str, Any]]:
        ts_code = self.last_focus.get("ts_code")
        if not ts_code:
            return None
        return self.tool_cache.get(ts_code)

    def get_cache_snippets(self, limit: int = 3) -> List[str]:
        if not self.tool_cache:
            return []
        sorted_entries = sorted(
            self.tool_cache.values(),
            key=lambda item: item.get("last_updated") or "",
            reverse=True,
        )
        snippets = [entry.get("summary", "") for entry in sorted_entries[:limit]]
        return [s for s in snippets if s]

    def _build_agent_input(self, user_query: str) -> str:
        sections = []
        focus_hint = self.get_focus_hint()
        if focus_hint:
            sections.append("【上下文提醒】\n" + focus_hint)
        sections.append("【当前用户问题】\n" + user_query)
        cache_snippets = self.get_cache_snippets()
        if cache_snippets:
            sections.append("【历史工具缓存】\n" + "\n".join(cache_snippets))
        return "\n\n".join(sections)

    def _build_fallback_response(self, user_query: str) -> str:
        focus_hint = self.get_focus_hint()
        cache_snippets = self.get_cache_snippets(limit=1)
        theory = (
            "【理论依据】知识库响应出现异常，临时沿用默认风险控制准则（分散配置、设定止损、控制杠杆）。"
        )
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

    async def run_chat(self, user_query: str, timeout: int = 90) -> Dict[str, Any]:
        await self.ensure_ready()
        enriched_input = self._build_agent_input(user_query)
        token_count = _count_tokens(enriched_input, tokenizer=self._prompt_tokenizer)
        logger.info("[PromptStats] tokens=%s chars=%s", token_count, len(enriched_input))
        try:
            response = await asyncio.wait_for(
                self._agent.run(enriched_input), timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.error("Agent 响应超时，返回兜底建议。")
            response = self._build_fallback_response(user_query)
        except Exception as exc:
            logger.error("Agent 执行异常: %s", exc)
            response = self._build_fallback_response(user_query)
        answer_text = self._normalize_answer(response)
        payload = {
            "answer": answer_text,
            "focus_hint": self.get_focus_hint(),
            "focus_entry": self.get_focus_entry(),
            "tool_cache": self.tool_cache,
        }
        return payload

    @staticmethod
    def _normalize_answer(response: Any) -> str:
        if response is None:
            return ""
        if isinstance(response, str):
            return response
        if hasattr(response, "response") and isinstance(response.response, str):
            return response.response
        return str(response)


runtime = AgentRuntime()
