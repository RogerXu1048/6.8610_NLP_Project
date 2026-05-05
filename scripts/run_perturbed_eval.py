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

# System prompts use Mode B "lightweight permission":
# the model is granted permission to ask a clarifying question if anything is
# unclear, but is NOT told that the prompt may be ambiguous. Detection of
# ambiguity is left to the model. The wording mirrors a generic deployed
# assistant context — no meta-prompt enumerating SA/EA/AC options.

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

# ── Response parsing ───────────────────────────────────────────────────────────


_CODE_PATTERNS = [
    re.compile(r"^\s*import\s+\w+", re.MULTILINE),
    re.compile(r"^\s*from\s+\w+\s+import", re.MULTILINE),
    re.compile(r"^\s*def\s+\w+\s*\(", re.MULTILINE),
    re.compile(r"^\s*class\s+\w+", re.MULTILINE),
    re.compile(r"^\s*return\s+", re.MULTILINE),
    re.compile(r"^\s*\w+\s*=\s*[^=]", re.MULTILINE),  # assignment, not ==
]


def _looks_like_code(text: str) -> bool:
    """Heuristic: does the text look like Python code (vs. a clarification question)?

    Used as a fallback when the model produced code but forgot the @@CODE_START@@
    markers. We require:
      - at least one strong code pattern (import/def/class/return/assignment), AND
      - the text does not look like a pure question (doesn't end with '?').
    """
    if not text.strip():
        return False
    has_pattern = any(p.search(text) for p in _CODE_PATTERNS)
    is_question = text.strip().endswith("?")
    return has_pattern and not is_question


def parse_response(text: str) -> dict:
    """Extract think_block, prose, and code from a raw model response.

    Code extraction priority:
    1. @@CODE_START@@ / @@CODE_END@@ markers (instructed in system prompt)
    2. First ```python ... ``` fence
    3. HTML <code>...</code> tags
    4. Heuristic: text looks like Python code (def/import/etc. + not a question)
    5. Empty (model produced no code — likely a clarification question)

    The system prompt instructs models to wrap code in markers. When markers,
    fences, and HTML tags are all absent, we use a regex heuristic to decide
    whether the bare text is code-with-missing-markers or pure prose. This is
    important for two reasons:
      - AC detection: putting a question into `code` would cause SyntaxError
        in the sandbox AND make the judge mis-label as SA/EA.
      - Recovering forgotten-marker cases: avoids losing valid code on the
        rare occasions when a model omits markers but writes valid code.
    """
    # 1. think_block: <think>...</think> (DeepSeek-R1, QwQ, etc.)
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    think_block = think_match.group(1).strip() if think_match else ""

    # Strip the think block before deciding code/prose split
    body = text
    if think_match:
        body = body.replace(think_match.group(0), "")

    # 2. code extraction
    marker_match = re.search(r"@@CODE_START@@(.*?)@@CODE_END@@", body, re.DOTALL)
    fence_match = re.search(r"```python\s*(.*?)```", body, re.DOTALL)
    html_match = re.search(r"<code>\s*(.*?)\s*</code>", body, re.DOTALL)
    if marker_match:
        code = marker_match.group(1).strip()
    elif fence_match:
        code = fence_match.group(1).strip()
    elif html_match:
        code = html_match.group(1).strip()
    elif _looks_like_code(body):
        code = body.strip()
    else:
        code = ""

    # 3. prose: body minus the matched code section
    stripped = body
    if marker_match:
        stripped = stripped.replace(marker_match.group(0), "")
    elif fence_match:
        stripped = re.sub(r"```python.*?```", "", stripped, flags=re.DOTALL)
    elif html_match:
        stripped = re.sub(r"<code>.*?</code>", "", stripped, flags=re.DOTALL)
    # If we used the heuristic fallback, body == code, so prose is empty.
    # If code is empty (pure prose), prose is the full body.
    prose = stripped.strip() if (marker_match or fence_match or html_match or not code) else ""

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
                # test_a is harness format (expects __SOLUTION__);
                # test_b is self-contained (defines its own data + asserts).
                # Cannot pass __SOLUTION__ wrapper to test_b — it would be a
                # bare string and the test's function calls would NameError.
                wrapped = wrap_ds1000_solution(parsed["code"])
                result_a = sandbox_ds1000.run(wrapped, item.test_a, timeout_s=60)
                result_b = sandbox_ds1000.run(parsed["code"], item.test_b, timeout_s=60)
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

    # Aggregate counts (samples can satisfy 0, 1, or both tests)
    pass_a_count = sum(s["passed_a"] for s in samples)
    pass_b_count = sum(s["passed_b"] for s in samples)
    pass_either_count = sum(s["passed_a"] or s["passed_b"] for s in samples)

    # Mutually-exclusive 4-way decomposition — used to determine which
    # interpretation the model "chose" on each sample:
    #   chose_a:    code satisfies test_a only       -> model picked A
    #   chose_b:    code satisfies test_b only       -> model picked B
    #   pass_both:  code satisfies both tests        -> tests cannot distinguish
    #   pass_neither: code satisfies neither         -> code error / wrong choice
    chose_a_count = sum(s["passed_a"] and not s["passed_b"] for s in samples)
    chose_b_count = sum(s["passed_b"] and not s["passed_a"] for s in samples)
    pass_both_count = sum(s["passed_a"] and s["passed_b"] for s in samples)
    pass_neither_count = sum(
        not s["passed_a"] and not s["passed_b"] for s in samples
    )
    # Sanity: chose_a + chose_b + pass_both + pass_neither == n_samples

    # Interpretation bias: of the samples that made a clear choice (A xor B),
    # what fraction picked A? None when model never made a clear choice.
    decisive = chose_a_count + chose_b_count
    interp_a_bias = (
        round(chose_a_count / decisive, 4) if decisive else None
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
        # Test-level counts (a sample can contribute to multiple)
        "pass_a_count": pass_a_count,
        "pass_b_count": pass_b_count,
        "pass_either_count": pass_either_count,
        "pass_a_rate": round(pass_a_count / n_samples, 4) if n_samples else 0.0,
        "pass_b_rate": round(pass_b_count / n_samples, 4) if n_samples else 0.0,
        "pass_either_rate": round(pass_either_count / n_samples, 4) if n_samples else 0.0,
        # Choice decomposition (mutually exclusive, sums to n_samples)
        "chose_a_count": chose_a_count,
        "chose_b_count": chose_b_count,
        "pass_both_count": pass_both_count,
        "pass_neither_count": pass_neither_count,
        "chose_a_rate": round(chose_a_count / n_samples, 4) if n_samples else 0.0,
        "chose_b_rate": round(chose_b_count / n_samples, 4) if n_samples else 0.0,
        "both_pass_rate": round(pass_both_count / n_samples, 4) if n_samples else 0.0,
        "neither_rate": round(pass_neither_count / n_samples, 4) if n_samples else 0.0,
        "interp_a_bias": interp_a_bias,
        "samples": samples,
    }


# ── Summary printing ───────────────────────────────────────────────────────────


def print_summary(records: list[dict], model_id: str) -> None:
    total_a = sum(r["pass_a_count"] for r in records)
    total_b = sum(r["pass_b_count"] for r in records)
    total_either = sum(r.get("pass_either_count", 0) for r in records)
    total_chose_a = sum(r.get("chose_a_count", 0) for r in records)
    total_chose_b = sum(r.get("chose_b_count", 0) for r in records)
    total_both = sum(r.get("pass_both_count", 0) for r in records)
    total_neither = sum(r["pass_neither_count"] for r in records)
    total_samp = sum(r["n_samples"] for r in records)

    print(f"\n{'='*60}")
    print(f"Model:       {model_id}")
    print(f"Items:       {len(records)}")
    print(f"\n── Test pass rates (a sample can satisfy both tests) ──")
    print(f"pass@k(A):       {total_a/total_samp:.1%}  ({total_a}/{total_samp})")
    print(f"pass@k(B):       {total_b/total_samp:.1%}  ({total_b}/{total_samp})")
    print(f"pass@k(either):  {total_either/total_samp:.1%}  ({total_either}/{total_samp})")
    print(f"\n── Choice decomposition (mutually exclusive, sums to 100%) ──")
    print(f"chose A only:    {total_chose_a/total_samp:.1%}  ({total_chose_a}/{total_samp})")
    print(f"chose B only:    {total_chose_b/total_samp:.1%}  ({total_chose_b}/{total_samp})")
    print(f"both pass:       {total_both/total_samp:.1%}  ({total_both}/{total_samp})  (tests can't distinguish)")
    print(f"neither pass:    {total_neither/total_samp:.1%}  ({total_neither}/{total_samp})  (code error or wrong choice)")
    decisive = total_chose_a + total_chose_b
    if decisive:
        print(f"\nInterpretation bias (chose A / decisive): "
              f"{total_chose_a/decisive:.1%}  ({total_chose_a}/{decisive})")

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
