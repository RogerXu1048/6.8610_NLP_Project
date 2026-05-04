#!/usr/bin/env python3
"""Phase 2: Run inference on perturbed (ambiguous) prompts.

Sends each item's `perturbed_prompt` to an LLM, runs the generated code against
both test_a and test_b via dual-blind sandbox execution, and records pass@k(A)
and pass@k(B) separately.

Usage:
    # Quick sanity check — 3 items, 1 sample
    python scripts/run_perturbed_eval.py --limit 3

    # Full run, 5 samples per item, temperature 0.8
    python scripts/run_perturbed_eval.py --model gpt-5.4 --n-samples 5 --temperature 0.8

    # MBPP only, skip DS-1000
    python scripts/run_perturbed_eval.py --source mbpp --skip-ds1000

Results saved to data/results/perturbed_<model>_<timestamp>.jsonl
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

# ── System prompts ─────────────────────────────────────────────────────────────
# Same wording as run_baseline_eval.py — model sees the ambiguous prompt naturally
# and produces thinking / assumptions / clarification questions organically.

SYSTEM_PROMPT_MBPP = (
    "You are a professional Python programmer. Read the problem carefully and write a correct Python solution. "
    "Wrap your final code in @@CODE_START@@ and @@CODE_END@@ markers exactly as shown."
)

SYSTEM_PROMPT_DS1000 = (
    "You are a professional Python programmer solving a data science problem. "
    "Write executable Python code (NOT a bare function definition). "
    "The problem's variables (e.g. df, X, data) are already defined in your execution context — use them directly. "
    "Do NOT read from files or re-initialize data that is already provided. "
    "Store your final answer in a variable named `result`. "
    "Wrap your final code in @@CODE_START@@ and @@CODE_END@@ markers exactly as shown."
)

# ── Response parsing ───────────────────────────────────────────────────────────


def parse_response(text: str) -> dict:
    """Extract think_block, prose, and code from a raw model response.

    Code extraction priority:
    1. @@CODE_START@@ / @@CODE_END@@ markers (instructed in system prompt)
    2. First ```python ... ``` fence (fallback for non-compliant models)
    3. HTML <code>...</code> tags
    4. Entire text (last resort)
    """
    # 1. think_block: <think>...</think> (DeepSeek-R1, QwQ, etc.)
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    think_block = think_match.group(1).strip() if think_match else ""

    # 2. code extraction
    marker_match = re.search(r"@@CODE_START@@(.*?)@@CODE_END@@", text, re.DOTALL)
    if marker_match:
        code = marker_match.group(1).strip()
    else:
        fence_match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
        if fence_match:
            code = fence_match.group(1).strip()
        else:
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


def wrap_ds1000_solution(code: str) -> str:
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
    """Call LLM n_samples times, run each against test_a and test_b, record results."""
    samples = []

    for idx in range(n_samples):
        sample: dict = {
            "sample": idx,
            "passed_a": False, "passed_b": False,
            "exit_code_a": -1, "exit_code_b": -1,
            "timed_out_a": False, "timed_out_b": False,
            "stderr_a": "", "stderr_b": "",
            "raw_response": "", "think_block": "", "prose": "",
            "generated_code": "", "latency_s": 0.0,
        }
        try:
            if item.source == "ds1000":
                if sandbox_ds1000 is None:
                    sample["stderr_a"] = "DS-1000 sandbox not available (--skip-ds1000)"
                    samples.append(sample)
                    continue
                resp = client.call(
                    config, prompt=item.perturbed_prompt, system=SYSTEM_PROMPT_DS1000
                )
                parsed = parse_response(resp.choices[0])
                wrapped = wrap_ds1000_solution(parsed["code"])
                result_a, result_b = sandbox_ds1000.run_dual_blind(
                    wrapped, item.test_a, item.test_b, timeout_s=60
                )
            else:
                resp = client.call(
                    config, prompt=item.perturbed_prompt, system=SYSTEM_PROMPT_MBPP
                )
                parsed = parse_response(resp.choices[0])
                result_a, result_b = sandbox_default.run_dual_blind(
                    parsed["code"], item.test_a, item.test_b
                )

            sample.update({
                "passed_a": result_a.passed,
                "passed_b": result_b.passed,
                "exit_code_a": result_a.exit_code,
                "exit_code_b": result_b.exit_code,
                "timed_out_a": result_a.timed_out,
                "timed_out_b": result_b.timed_out,
                "stderr_a": result_a.stderr[:400] if result_a.stderr else "",
                "stderr_b": result_b.stderr[:400] if result_b.stderr else "",
                "raw_response": resp.choices[0],
                "think_block": parsed["think_block"],
                "prose": parsed["prose"],
                "generated_code": parsed["code"],
                "latency_s": resp.latency_s,
            })

        except Exception as exc:
            sample["stderr_a"] = str(exc)[:400]

        samples.append(sample)

    pass_a_count = sum(s["passed_a"] for s in samples)
    pass_b_count = sum(s["passed_b"] for s in samples)
    pass_neither_count = sum(
        not s["passed_a"] and not s["passed_b"] for s in samples
    )

    return {
        "task_id": item.task_id,
        "anchor_task_id": item.anchor_task_id,
        "source": item.source,
        "library": item.library,
        "ambiguity_type": item.ambiguity_type,
        "risk_level": item.risk_level,
        "perturbed_prompt": item.perturbed_prompt,
        "interpretation_a": item.interpretation_a,
        "interpretation_b": item.interpretation_b,
        "n_samples": n_samples,
        "pass_a_count": pass_a_count,
        "pass_b_count": pass_b_count,
        "pass_neither_count": pass_neither_count,
        "pass_a_rate": round(pass_a_count / n_samples, 4) if n_samples else 0.0,
        "pass_b_rate": round(pass_b_count / n_samples, 4) if n_samples else 0.0,
        "samples": samples,
    }


# ── Summary printing ───────────────────────────────────────────────────────────


def print_summary(records: list[dict], model_id: str) -> None:
    total_a = sum(r["pass_a_count"] for r in records)
    total_b = sum(r["pass_b_count"] for r in records)
    total_neither = sum(r["pass_neither_count"] for r in records)
    total_samp = sum(r["n_samples"] for r in records)

    print(f"\n{'='*60}")
    print(f"Model:       {model_id}")
    print(f"Items:       {len(records)}")
    print(f"pass@k(A):   {total_a/total_samp:.1%}  ({total_a}/{total_samp})")
    print(f"pass@k(B):   {total_b/total_samp:.1%}  ({total_b}/{total_samp})")
    print(f"neither:     {total_neither/total_samp:.1%}  ({total_neither}/{total_samp})")

    print(f"\n── By source ──────────────────────────────────────────")
    for src in ["mbpp", "ds1000"]:
        sub = [r for r in records if r["source"] == src]
        if sub:
            a = sum(r["pass_a_count"] for r in sub)
            b = sum(r["pass_b_count"] for r in sub)
            t = sum(r["n_samples"] for r in sub)
            print(f"  {src:8s}  A:{a/t:.1%}  B:{b/t:.1%}  [{len(sub)} items]")

    print(f"\n── By ambiguity type ──────────────────────────────────")
    for amb in ["coreferential", "syntactic", "scopal",
                "collective_distributive", "elliptical"]:
        sub = [r for r in records if r["ambiguity_type"] == amb]
        if sub:
            a = sum(r["pass_a_count"] for r in sub)
            b = sum(r["pass_b_count"] for r in sub)
            t = sum(r["n_samples"] for r in sub)
            print(f"  {amb:25s}  A:{a/t:.1%}  B:{b/t:.1%}  [{len(sub)} items]")

    print(f"\n── By risk level ──────────────────────────────────────")
    for risk in ["low", "high"]:
        sub = [r for r in records if r["risk_level"] == risk]
        if sub:
            a = sum(r["pass_a_count"] for r in sub)
            b = sum(r["pass_b_count"] for r in sub)
            t = sum(r["n_samples"] for r in sub)
            print(f"  {risk:5s}  A:{a/t:.1%}  B:{b/t:.1%}  [{len(sub)} items]")

    print(f"\n── Per-item results ───────────────────────────────────")
    for r in records:
        a_str = f"A:{r['pass_a_count']}/{r['n_samples']}"
        b_str = f"B:{r['pass_b_count']}/{r['n_samples']}"
        print(f"  {r['task_id']:10s}  {a_str}  {b_str}  "
              f"({r['source']}, {r['ambiguity_type'][:12]})")


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2: Inference on perturbed (ambiguous) prompts."
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
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument(
        "--source", choices=["mbpp", "ds1000", "all"], default="all"
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

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_model = args.model.replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"perturbed_{safe_model}_{timestamp}.jsonl"

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

            a_str = f"A:{record['pass_a_count']}/{record['n_samples']}"
            b_str = f"B:{record['pass_b_count']}/{record['n_samples']}"
            print(f"{a_str}  {b_str}  ({elapsed:.1f}s)")

            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            records.append(record)

    total_elapsed = time.time() - t_start
    print_summary(records, config.model_id)
    print(f"\nTotal time:  {total_elapsed:.1f}s")
    print(f"Results:     {out_path}")


if __name__ == "__main__":
    main()
