#!/usr/bin/env python3
"""Run baseline evaluation (clean prompts, no ambiguity) across benchmark items.

Sends each item's clean `prompt` to an LLM, runs the output in a Docker sandbox
against the original `test_code`, and records pass/fail.

Usage:
    # Quick sanity check — 3 MBPP items, gpt-5.4
    python scripts/run_baseline_eval.py --limit 3

    # Full MBPP baseline, 1 sample per item
    python scripts/run_baseline_eval.py --source mbpp --model gpt-5.4

    # Full DS-1000 baseline (requires ambicode-ds1000 Docker image)
    python scripts/run_baseline_eval.py --source ds1000 --model claude-sonnet

    # pass@3 sampling (3 samples per item, temperature 0.8)
    python scripts/run_baseline_eval.py --n-samples 3 --temperature 0.8

Results are saved incrementally to data/results/baseline_<model>_<timestamp>.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.model import BenchmarkItem
from src.util.llm import LLMClient, ModelConfig
from src.util.sandbox import Sandbox

# ── Paths ─────────────────────────────────────────────────────────────────────

BENCHMARK_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "benchmark" / "benchmark.jsonl"
)
RESULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "results"

# ── Prompt helpers ─────────────────────────────────────────────────────────────

# System prompts must be IDENTICAL between baseline and perturbed eval —
# only the user prompt differs (clean vs. perturbed). See run_perturbed_eval.py
# for rationale (Mode B "lightweight permission" — naturalistic, no meta-prompt).
# AC behavior should not be triggered on clean prompts; equal prompts ensure
# any AC seen on perturbed runs is attributable to ambiguity, not the system message.

SYSTEM_PROMPT_MBPP = (
    "You are a helpful Python programming assistant. "
    "If anything about the user's request is unclear, you may ask a clarifying question. "
    "Otherwise, write the requested Python code and wrap it in "
    "@@CODE_START@@ and @@CODE_END@@ markers."
)

SYSTEM_PROMPT_DS1000 = (
    "You are a helpful Python programming assistant solving a data science problem. "
    "If anything about the user's request is unclear, you may ask a clarifying question. "
    "Otherwise, write executable Python code that uses the variables already defined "
    "in the execution context (e.g. df, X, data) — do not read from files or "
    "re-initialize provided data. Store your final answer in a variable named `result`. "
    "Wrap your code in @@CODE_START@@ and @@CODE_END@@ markers."
)


def parse_response(text: str) -> dict:
    """Extract think_block, prose, and code from a raw model response.

    Priority for code extraction:
    1. @@CODE_START@@ / @@CODE_END@@ markers (instructed in system prompt)
    2. First ```python ... ``` fence (fallback for models that ignore markers)
    3. Entire text (last resort)
    """
    # 1. think_block: <think>...</think> (DeepSeek-R1, QwQ, etc.)
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    think_block = think_match.group(1).strip() if think_match else ""

    # 2. code: @@CODE_START@@ / @@CODE_END@@ markers, else ```python fence
    marker_match = re.search(r"@@CODE_START@@(.*?)@@CODE_END@@", text, re.DOTALL)
    if marker_match:
        code = marker_match.group(1).strip()
    else:
        fence_match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
        if fence_match:
            code = fence_match.group(1).strip()
        else:
            # Some models (e.g. Claude) emit HTML code tags
            html_match = re.search(r"<code>\s*(.*?)\s*</code>", text, re.DOTALL)
            code = html_match.group(1).strip() if html_match else text.strip()

    # 3. prose: everything outside think block and code section
    stripped = text
    if think_match:
        stripped = stripped.replace(think_match.group(0), "")
    if marker_match:
        stripped = stripped.replace(marker_match.group(0), "")
    else:
        stripped = re.sub(r"```python.*?```", "", stripped, flags=re.DOTALL)
        stripped = re.sub(r"<code>.*?</code>", "", stripped, flags=re.DOTALL)
    prose = stripped.strip()

    return {"think_block": think_block, "prose": prose, "code": code}


def extract_function_name(test_code: str) -> str | None:
    m = re.search(r"assert\s+(\w+)\s*\(", test_code)
    return m.group(1) if m else None


def extract_example_call(test_code: str) -> str | None:
    """Extract the first function call from test assertions as a usage example.

    From: assert add_lists([5, 6, 7], (9, 10)) == (9, 10, 5, 6, 7)
    Returns: add_lists([5, 6, 7], (9, 10))
    """
    m = re.search(r"assert\s+(\w+\(.*?\))\s*==", test_code)
    return m.group(1) if m else None


def build_mbpp_prompt(item: BenchmarkItem) -> str:
    """Build MBPP prompt with function name and example call signature.

    Shows the model both the function name and how it will be called
    (argument count and types), fixing wrong-signature TypeErrors.
    """
    func_name = extract_function_name(item.test_code)
    example_call = extract_example_call(item.test_code)

    if func_name and example_call:
        return (
            f"{item.prompt}\n"
            f"The function should be named `{func_name}`.\n"
            f"Example call: {example_call}"
        )
    if func_name:
        return f"{item.prompt}\nThe function should be named `{func_name}`."
    return item.prompt


def wrap_ds1000_solution(code: str) -> str:
    """Wrap raw model output as the DS-1000 __SOLUTION__ string expected by the harness.

    Uses repr() instead of a raw triple-quoted string so that any triple-quotes
    or backslashes inside the model's code don't break the outer string delimiter.
    """
    return f"__SOLUTION__ = {repr(code)}"


# ── Per-item evaluation ────────────────────────────────────────────────────────


def evaluate_item(
    item: BenchmarkItem,
    client: LLMClient,
    config: ModelConfig,
    sandbox_default: Sandbox,
    sandbox_ds1000: Sandbox | None,
    n_samples: int,
) -> dict:
    """Call LLM n_samples times, run each output in sandbox, record results."""
    samples = []

    for idx in range(n_samples):
        sample: dict = {"sample": idx, "passed": False, "exit_code": -1,
                        "timed_out": False, "stderr": "", "raw_response": "",
                        "think_block": "", "prose": "", "generated_code": "",
                        "latency_s": 0.0}
        try:
            if item.source == "ds1000":
                if sandbox_ds1000 is None:
                    sample["stderr"] = "DS-1000 sandbox not available (--skip-ds1000)"
                    samples.append(sample)
                    continue
                prompt = item.prompt
                resp = client.call(config, prompt=prompt, system=SYSTEM_PROMPT_DS1000)
                parsed = parse_response(resp.choices[0])
                wrapped = wrap_ds1000_solution(parsed["code"])
                result = sandbox_ds1000.run(wrapped, item.test_code, timeout_s=60)
            else:
                # MBPP
                prompt = build_mbpp_prompt(item)
                resp = client.call(config, prompt=prompt, system=SYSTEM_PROMPT_MBPP)
                parsed = parse_response(resp.choices[0])
                result = sandbox_default.run(parsed["code"], item.test_code)

            sample.update({
                "passed": result.passed,
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
                "stderr": result.stderr[:400] if result.stderr else "",
                "raw_response": resp.choices[0],
                "think_block": parsed["think_block"],
                "prose": parsed["prose"],
                "generated_code": parsed["code"],
                "latency_s": resp.latency_s,
            })

        except Exception as exc:
            sample["stderr"] = str(exc)[:400]

        samples.append(sample)

    pass_count = sum(s["passed"] for s in samples)
    return {
        "task_id": item.task_id,
        "anchor_task_id": item.anchor_task_id,
        "source": item.source,
        "library": item.library,
        "ambiguity_type": item.ambiguity_type,
        "risk_level": item.risk_level,
        "n_samples": n_samples,
        "pass_count": pass_count,
        "pass_rate": round(pass_count / n_samples, 4) if n_samples else 0.0,
        "samples": samples,
    }


# ── Summary printing ───────────────────────────────────────────────────────────


def print_summary(records: list[dict], model_id: str) -> None:
    total_pass = sum(r["pass_count"] for r in records)
    total_samp = sum(r["n_samples"] for r in records)

    print(f"\n{'='*55}")
    print(f"Model:        {model_id}")
    print(f"Items:        {len(records)}")
    print(f"Overall pass@1: {total_pass/total_samp:.1%}  ({total_pass}/{total_samp})")

    # By source
    print(f"\n── By source ──────────────────────────────")
    for src in ["mbpp", "ds1000"]:
        sub = [r for r in records if r["source"] == src]
        if sub:
            p = sum(r["pass_count"] for r in sub)
            t = sum(r["n_samples"] for r in sub)
            print(f"  {src:8s}  {p/t:.1%}  ({p}/{t})  [{len(sub)} items]")

    # By ambiguity type
    print(f"\n── By ambiguity type ──────────────────────")
    for amb in ["coreferential", "syntactic", "scopal",
                "collective_distributive", "elliptical"]:
        sub = [r for r in records if r["ambiguity_type"] == amb]
        if sub:
            p = sum(r["pass_count"] for r in sub)
            t = sum(r["n_samples"] for r in sub)
            print(f"  {amb:25s}  {p/t:.1%}  ({p}/{t})")

    # By risk level
    print(f"\n── By risk level ──────────────────────────")
    for risk in ["low", "high"]:
        sub = [r for r in records if r["risk_level"] == risk]
        if sub:
            p = sum(r["pass_count"] for r in sub)
            t = sum(r["n_samples"] for r in sub)
            print(f"  {risk:5s}  {p/t:.1%}  ({p}/{t})")

    # Per-item breakdown
    print(f"\n── Per-item results ───────────────────────")
    for r in records:
        status = "✓" if r["pass_count"] == r["n_samples"] else (
            "~" if r["pass_count"] > 0 else "✗"
        )
        fails = [s for s in r["samples"] if not s["passed"]]
        err_hint = ""
        if fails and fails[0]["stderr"]:
            last_line = fails[0]["stderr"].strip().split("\n")[-1]
            err_hint = f"  ← {last_line[:60]}"
        print(f"  {status} {r['task_id']:10s}  {r['pass_count']}/{r['n_samples']}"
              f"  ({r['source']}, {r['ambiguity_type'][:12]}){err_hint}")


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baseline evaluation: clean prompts → LLM → sandbox."
    )
    parser.add_argument(
        "--model", default="gpt-5.4",
        help="Model alias from config/models.yaml (default: gpt-5.4)"
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--n-samples", type=int, default=1,
        help="Samples per item (1 = pass@1, 5 = pass@5 estimate)"
    )
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument(
        "--source", choices=["mbpp", "ds1000", "all"], default="all",
        help="Filter by source benchmark (default: all)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Evaluate only the first N items (for quick sanity checks)"
    )
    parser.add_argument(
        "--skip-ds1000", action="store_true",
        help="Skip DS-1000 items (avoids needing the ambicode-ds1000 Docker image)"
    )
    args = parser.parse_args()

    # Load benchmark items
    items: list[BenchmarkItem] = []
    with open(BENCHMARK_PATH) as f:
        for line in f:
            if line.strip():
                items.append(BenchmarkItem.from_dict(json.loads(line)))

    if args.source != "all":
        items = [it for it in items if it.source == args.source]
    if args.skip_ds1000:
        items = [it for it in items if it.source != "ds1000"]
    if args.limit:
        items = items[: args.limit]

    print(f"Benchmark items to evaluate: {len(items)}")
    if not items:
        print("No items match the filters. Exiting.")
        return

    # Initialize LLM + sandboxes
    client = LLMClient()
    config = ModelConfig(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    sandbox_default = Sandbox()

    sandbox_ds1000: Sandbox | None = None
    if not args.skip_ds1000 and any(it.source == "ds1000" for it in items):
        try:
            sandbox_ds1000 = Sandbox(image="ambicode-ds1000")
        except Exception as exc:
            print(f"[warn] Could not init ambicode-ds1000 sandbox: {exc}")
            print("[warn] DS-1000 items will be skipped. "
                  "Run: docker build -t ambicode-ds1000 -f docker/ds1000.Dockerfile .")

    print(f"Model:       {config.model_id}")
    print(f"Temperature: {args.temperature}  |  n_samples: {args.n_samples}")
    print()

    # Output file (incremental write so results survive partial runs)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_model = args.model.replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"baseline_{safe_model}_{timestamp}.jsonl"

    records: list[dict] = []
    t_start = time.time()

    with open(out_path, "w") as out_f:
        for i, item in enumerate(items, 1):
            print(
                f"[{i:2d}/{len(items)}] {item.task_id}  "
                f"({item.source}, {item.ambiguity_type}, {item.risk_level})...",
                end="  ",
                flush=True,
            )
            t0 = time.time()
            record = evaluate_item(
                item, client, config,
                sandbox_default, sandbox_ds1000,
                args.n_samples,
            )
            elapsed = time.time() - t0

            status = (
                "PASS" if record["pass_count"] == record["n_samples"]
                else f"PARTIAL {record['pass_count']}/{record['n_samples']}"
                if record["pass_count"] > 0
                else "FAIL"
            )
            print(f"{status}  ({elapsed:.1f}s)")

            # Show error hint on failure
            fails = [s for s in record["samples"] if not s["passed"] and s["stderr"]]
            if fails:
                last_line = fails[0]["stderr"].strip().split("\n")[-1]
                print(f"         ↳ {last_line[:80]}")

            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            records.append(record)

    total_elapsed = time.time() - t_start
    print_summary(records, config.model_id)
    print(f"\nTotal time:  {total_elapsed:.1f}s")
    print(f"Results:     {out_path}")


if __name__ == "__main__":
    main()
