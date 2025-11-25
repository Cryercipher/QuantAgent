#!/usr/bin/env python3
"""Standalone harness to replay the curated QuantAgent test suite."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from llama_index.core import Settings
from llama_index.core.agent import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer

from core.llm_factory import ModelFactory
from prompts.system_prompts import AGENT_SYSTEM_PROMPT
from tools.knowledge_base import FinancialKnowledgeBase
from tools.quant_analysis import MarketInsightTool

os.environ.setdefault("QUANT_AGENT_LOG_LEVEL", "WARNING")

RESULTS_DIR = ROOT_DIR / "tests" / "results"


@dataclass
class TestCase:
    id: str
    category: str
    prompts: List[str]
    expectation: str


TEST_CASES: dict[str, TestCase] = {
    case.id: case
    for case in [
        TestCase(
            id="U-01",
            category="unit-routing",
            prompts=["什么是量化对冲策略？它的风险在哪里？"],
            expectation="Only financial_theory_tool should be invoked.",
        ),
        TestCase(
            id="U-02",
            category="unit-routing",
            prompts=["帮我查一下宁德时代现在的股价。"],
            expectation="Only quant_analysis_tool should be invoked.",
        ),
        TestCase(
            id="U-03",
            category="unit-routing",
            prompts=["你好，你是谁？"],
            expectation="No tool invocations; respond with persona only.",
        ),
        TestCase(
            id="I-01",
            category="integration",
            prompts=["我是稳健型投资者，最近想买点银行股，招商银行值得入手吗？"],
            expectation="financial_theory_tool + quant_analysis_tool with persona-aware advice.",
        ),
        TestCase(
            id="I-02",
            category="integration",
            prompts=["我是激进型选手，想博短线，你看中信证券现在的走势怎么样？"],
            expectation="quant_analysis_tool (and optionally theory) focusing on trend/momentum.",
        ),
        TestCase(
            id="E-01",
            category="edge",
            prompts=["如何判断一只股票是否被高估？"],
            expectation="Only theory tool; no fabricated ticker or data lookup.",
        ),
        TestCase(
            id="E-02",
            category="edge",
            prompts=["帮我分析一下‘老干妈’这只股票。"],
            expectation="Quant tool reports 'not found'; agent explains gracefully.",
        ),
        TestCase(
            id="E-03",
            category="edge",
            prompts=["今天天气怎么样？"],
            expectation="Out-of-scope reply without calling financial tools.",
        ),
        TestCase(
            id="C-01",
            category="context",
            prompts=["茅台最近表现咋样？", "那五粮液呢？"],
            expectation="Maintain context and call quant tool for each ticker in one session.",
        ),
    ]
}


def _ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _default_output_base() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / f"test_results_{_ts()}"


def list_cases() -> None:
    print("Available test cases:\n")
    for case in TEST_CASES.values():
        preview = " | ".join(case.prompts)
        print(f"- {case.id:<4} [{case.category}] -> {preview}")
    print()


def _response_text(output: Any) -> str:
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    for attr in ("response", "raw", "message"):
        value = getattr(output, attr, None)
        if isinstance(value, str):
            return value
    return str(output)


class AgentHarness:
    """Initializes shared resources and runs conversations per test case."""

    def __init__(self) -> None:
        ModelFactory.init_models()
        kb = FinancialKnowledgeBase()
        theory_tool = kb.get_tool()
        quant_tool = MarketInsightTool().get_tool()
        if not theory_tool or not quant_tool:
            raise RuntimeError("Both financial_theory_tool and quant_analysis_tool must be available.")
        self.tools = [theory_tool, quant_tool]

    def _new_agent(self) -> ReActAgent:
        return ReActAgent(
            tools=self.tools,
            llm=Settings.llm,
            verbose=False,
            system_prompt=AGENT_SYSTEM_PROMPT,
            memory=ChatMemoryBuffer.from_defaults(token_limit=4096),
        )

    async def run_case(self, case: TestCase) -> dict:
        agent = self._new_agent()
        turn_logs = []
        status = "ok"
        error_message = None
        start = time.time()

        for prompt in case.prompts:
            try:
                response = await agent.run(prompt)
                turn_logs.append({"user": prompt, "agent": _response_text(response).strip()})
            except Exception as exc:  # noqa: BLE001
                status = "error"
                error_message = str(exc)
                turn_logs.append({"user": prompt, "agent": f"[ERROR] {exc}"})
                break

        return {
            "case": case.id,
            "category": case.category,
            "expectation": case.expectation,
            "status": status,
            "error": error_message,
            "duration_sec": round(time.time() - start, 2),
            "turns": turn_logs,
        }

    async def run_cases(self, selected_ids: List[str]) -> List[dict]:
        results = []
        for case_id in selected_ids:
            case = TEST_CASES[case_id]
            results.append(await self.run_case(case))
        return results


def _render_markdown(generated_at: str, results: List[dict]) -> str:
    lines = ["# QuantAgent Test Results", "", f"Generated at {generated_at}", ""]
    for res in results:
        lines.extend([f"## {res['case']} · {res['category']}", ""])
        if res.get("expectation"):
            lines.append(f"_Expectation_: {res['expectation']}")
            lines.append("")
        if res.get("status") != "ok":
            lines.append(f"**Status:** {res['status']} ({res.get('error')})")
            lines.append("")
        for idx, turn in enumerate(res["turns"], start=1):
            lines.append(f"**Turn {idx} — User:** {turn['user']}")
            lines.append("")
            lines.append(turn["agent"] or "[No response]")
            lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _write_outputs(generated_at: str, results: List[dict], output_base: Path, fmt: str) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    if fmt in {"markdown", "both"}:
        md_path = output_base.with_suffix(".md")
        md_path.write_text(_render_markdown(generated_at, results), encoding="utf-8")
        print(f"Markdown transcript saved to {md_path}")
    if fmt in {"json", "both"}:
        json_path = output_base.with_suffix(".json")
        payload = {"generated_at": generated_at, "results": results}
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON transcript saved to {json_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay scripted QuantAgent test cases (non-interactive).")
    parser.add_argument("--cases", help="Comma-separated case IDs to run (default: all).")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path prefix (extension is added automatically).",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json", "both"],
        default="markdown",
        help="Result file format (default: markdown).",
    )
    parser.add_argument("--list", action="store_true", help="List available cases and exit.")
    return parser.parse_args()


def _select_cases(arg: str | None) -> List[str]:
    if not arg:
        return list(TEST_CASES.keys())
    if "," in arg:
        selected = [cid.strip() for cid in arg.split(",") if cid.strip()]
    else:
        selected = [cid.strip() for cid in arg.split() if cid.strip()]
    missing = [case_id for case_id in selected if case_id not in TEST_CASES]
    if missing:
        raise SystemExit(f"Unknown case IDs: {', '.join(missing)}")
    return selected


def main() -> None:
    args = _parse_args()
    if args.list:
        list_cases()
        return

    selected_ids = _select_cases(args.cases)
    output_base = args.output.with_suffix("") if args.output else _default_output_base()
    harness = AgentHarness()

    print(f"Running {len(selected_ids)} case(s) in standalone mode...\n")
    results = asyncio.run(harness.run_cases(selected_ids))
    generated_at = _ts()
    _write_outputs(generated_at, results, output_base, args.format)


if __name__ == "__main__":
    main()
