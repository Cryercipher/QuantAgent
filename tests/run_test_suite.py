#!/usr/bin/env python3
"""Lightweight harness to replay the curated QuantAgent test suite.

The script launches `app.py` as a subprocess for each test case, feeds the
configured turns (prompts), captures stdout/stderr, and stores artifacts for
manual inspection. This does not attempt to auto-grade the semantics; instead it
produces structured logs so humans (or downstream scripts) can evaluate
robustness, routing, and answer quality.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "tests" / "results"


def _ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _default_output_path() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / f"test_results_{_ts()}.json"


def _readable_timeout(seconds: int) -> str:
    return f"{seconds}s" if seconds % 60 else f"{seconds // 60}m"


def _ensure_turns(turns_or_prompt: List[str] | str) -> List[str]:
    if isinstance(turns_or_prompt, str):
        return [turns_or_prompt]
    return list(turns_or_prompt)


@dataclass
class TestCase:
    id: str
    category: str
    prompt: List[str]
    expectation: str

    @property
    def input_block(self) -> str:
        # Feed every turn sequentially, then exit the agent loop.
        turns = [text.strip() for text in self.prompt if text.strip()]
        turns.append("exit")
        return "\n".join(turns) + "\n"

    def to_public_dict(self) -> dict:
        data = asdict(self)
        data["prompt"] = self.prompt
        return data


TEST_CASES: dict[str, TestCase] = {
    case.id: case
    for case in [
        TestCase(
            id="U-01",
            category="unit-routing",
            prompt=_ensure_turns("什么是量化对冲策略？它的风险在哪里？"),
            expectation="Only financial_theory_tool should be invoked.",
        ),
        TestCase(
            id="U-02",
            category="unit-routing",
            prompt=_ensure_turns("帮我查一下宁德时代现在的股价。"),
            expectation="Only quant_analysis_tool should be invoked.",
        ),
        TestCase(
            id="U-03",
            category="unit-routing",
            prompt=_ensure_turns("你好，你是谁？"),
            expectation="No tool invocations; respond with persona only.",
        ),
        TestCase(
            id="I-01",
            category="integration",
            prompt=_ensure_turns("我是稳健型投资者，最近想买点银行股，招商银行值得入手吗？"),
            expectation="financial_theory_tool + quant_analysis_tool with persona-aware advice.",
        ),
        TestCase(
            id="I-02",
            category="integration",
            prompt=_ensure_turns("我是激进型选手，想博短线，你看中信证券现在的走势怎么样？"),
            expectation="quant_analysis_tool (and optionally theory) focusing on trend/momentum.",
        ),
        TestCase(
            id="E-01",
            category="edge",
            prompt=_ensure_turns("如何判断一只股票是否被高估？"),
            expectation="Only theory tool; no fabricated ticker or data lookup.",
        ),
        TestCase(
            id="E-02",
            category="edge",
            prompt=_ensure_turns("帮我分析一下‘老干妈’这只股票。"),
            expectation="Quant tool reports 'not found'; agent explains gracefully.",
        ),
        TestCase(
            id="E-03",
            category="edge",
            prompt=_ensure_turns("今天天气怎么样？"),
            expectation="Out-of-scope reply without calling financial tools.",
        ),
        TestCase(
            id="C-01",
            category="context",
            prompt=_ensure_turns([
                "茅台最近表现咋样？",
                "那五粮液呢？",
            ]),
            expectation="Maintain context and call quant tool for each ticker in one session.",
        ),
    ]
}


def list_cases() -> None:
    print("Available test cases:\n")
    for case in TEST_CASES.values():
        turns_preview = " | ".join(case.prompt)
        print(f"- {case.id:<4} [{case.category}] -> {turns_preview}")
    print()


def run_case(case: TestCase, timeout: int) -> dict:
    start_ts = time.time()
    proc = subprocess.Popen(
        [sys.executable, "app.py"],
        cwd=ROOT_DIR,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = proc.communicate(case.input_block, timeout=timeout)
        status = "ok" if proc.returncode == 0 else "error"
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        status = "timeout"
    duration = time.time() - start_ts

    return {
        "case": case.id,
        "category": case.category,
        "expectation": case.expectation,
        "status": status,
        "exit_code": proc.returncode,
        "duration_sec": round(duration, 2),
        "stdout": stdout,
        "stderr": stderr,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run QuantAgent scripted test cases.")
    parser.add_argument(
        "--cases",
        help="Comma-separated list of case IDs to run (default: all).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=210,
        help="Per-case timeout in seconds (default: 210).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to write JSON results (default: tests/results/test_results_<ts>.json).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available cases and exit.",
    )
    args = parser.parse_args()

    if args.list:
        list_cases()
        return

    selected_ids = (
        [case_id.strip() for case_id in args.cases.split(",") if case_id.strip()]
        if args.cases
        else list(TEST_CASES.keys())
    )

    missing = [case_id for case_id in selected_ids if case_id not in TEST_CASES]
    if missing:
        parser.error(f"Unknown case IDs: {', '.join(missing)}")

    output_path = args.output or _default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    print(f"Running {len(selected_ids)} case(s) with timeout {_readable_timeout(args.timeout)}...\n")
    for case_id in selected_ids:
        case = TEST_CASES[case_id]
        print(f"[+] {case.id}: {case.expectation}")
        result = run_case(case, timeout=args.timeout)
        print(f"    -> status={result['status']} exit_code={result['exit_code']} duration={result['duration_sec']}s")
        results.append(result)

    payload = {
        "generated_at": _ts(),
        "cases": [TEST_CASES[cid].to_public_dict() for cid in selected_ids],
        "results": results,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nDetailed logs saved to {output_path}")


if __name__ == "__main__":
    main()
