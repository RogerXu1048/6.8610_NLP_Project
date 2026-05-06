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
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    "Otherwise, structure your code in TWO parts:\n"
    "1. Define a function `g(...)` that takes the relevant input variables as parameters "
    "and returns the answer. The function should be reusable on different inputs.\n"
    "2. Call it: `result = g(<the variables already defined in the execution context, "
    "e.g. df, X, data>)` so `result` holds the final answer.\n"
    "Do not read from files or re-initialize data that is already provided. "
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


def _extract_ds1000_setup(perturbed_prompt: str) -> str:
    """Extract the first <code>...</code> setup block from a DS-1000 prompt.

    DS-1000 prompts have the structure:
        Problem: ...
        <code>
        import pandas as pd
        df = pd.DataFrame(...)
        </code>
        result = ... # put solution in this variable
        BEGIN SOLUTION
        <code>
        [model puts code here]

    The first <code> block defines the variables (df, X, a, b, c, ...) the
    model is expected to use. We need this for test_b execution because:
      - LLM code is a fragment that uses these variables
      - test_b is "self-contained" but uses different variable names
      - Concatenating LLM code + test_b -> NameError on the LLM's variables
    Prepending this setup makes LLM code runnable; test_b then checks `result`.

    Returns "" if no setup block found (1/36 items in our benchmark).
    """
    m = re.search(r"<code>\s*(.*?)\s*</code>", perturbed_prompt, re.DOTALL)
    return m.group(1) if m else ""


def _extract_example_assertion(test_code: str) -> str | None:
    """Extract first `assert <call> == <expected>` from test_code as a usage example."""
    import ast
    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            t = node.test
            if (
                isinstance(t, ast.Compare)
                and len(t.ops) == 1
                and isinstance(t.ops[0], ast.Eq)
                and isinstance(t.left, ast.Call)
            ):
                try:
                    return ast.unparse(t).strip()
                except Exception:
                    return None
    return None


def inject_example_into_mbpp_prompt(perturbed_prompt: str, test_code: str) -> str:
    """Inject `Example: assert <call> == <expected>` into the docstring of an
    MBPP-style perturbed prompt. Keeps the structure aligned with the baseline
    prompt (which also includes this example) so the only difference between
    baseline and perturbed is the docstring text — not whether an example is shown.

    Returns the prompt unchanged if no docstring or no assertion can be found
    (e.g., DS-1000 prompts, which have a different format).
    """
    example = _extract_example_assertion(test_code)
    if not example:
        return perturbed_prompt
    # Only inject if the prompt has a docstring closer
    closer_match = re.search(r'(\n\s*)"""\s*$', perturbed_prompt)
    if not closer_match:
        return perturbed_prompt
    indent = closer_match.group(1).lstrip("\n")
    insertion = f"\n{indent}\n{indent}Example: assert {example}"
    return (
        perturbed_prompt[: closer_match.start()]
        + insertion
        + perturbed_prompt[closer_match.start():]
    )


def ensure_humaneval_test_invocation(test_code: str, entry_point: str | None) -> str:
    """Append `check(<entry_point>)` to HumanEval test_code if absent.

    HumanEval's test_code is a bare `def check(candidate)` definition with no
    top-level invocation. Without an explicit call, the sandbox script defines
    the function and exits 0 with no assertion executed.
    """
    if not entry_point:
        return test_code
    top_level_call = re.compile(r"^\s*check\s*\(", re.MULTILINE)
    if top_level_call.search(test_code):
        return test_code
    return test_code.rstrip() + f"\ncheck({entry_point})\n"


def _build_ds1000_for_test_b(llm_code: str, perturbed_prompt: str) -> str:
    """Build code to run against test_b: setup + LLM code, both in try/except.

    Setup may contain calls (e.g. `load_data()`) that fail in sandbox; wrapping
    in try/except ensures partial setup still defines whatever variables it can.
    LLM code is also wrapped so a failed line doesn't kill `result` if it was
    defined earlier.
    """
    import textwrap
    setup = _extract_ds1000_setup(perturbed_prompt)
    parts = []
    if setup:
        indented = textwrap.indent(setup.rstrip(), "    ")
        parts.append(f"try:\n{indented}\nexcept Exception:\n    pass\n")
    parts.append(llm_code)
    return "\n".join(parts)


# ── Per-item evaluation ────────────────────────────────────────────────────────


def _run_one_perturbed_sample(
    idx: int,
    item: BenchmarkItem,
    client: LLMClient,
    config: ModelConfig,
    sandbox_default: Sandbox,
    sandbox_ds1000: Sandbox | None,
) -> dict:
    """Single (LLM-call + dual sandbox) for one perturbed sample. Returns a sample dict.

    Designed to be safe to call from multiple threads in parallel: each call uses
    a fresh `client.call(...)` (the underlying OpenAI SDK is thread-safe) and
    spawns its own short-lived Docker containers via the Sandbox.
    """
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
                return sample
            resp = client.call(
                config, prompt=item.perturbed_prompt, system=SYSTEM_PROMPT_DS1000
            )
            parsed = parse_response(resp.choices[0])
            # test_a is harness format (expects __SOLUTION__);
            # test_b is self-contained (defines its own data + asserts).
            wrapped = wrap_ds1000_solution(parsed["code"])
            result_a = sandbox_ds1000.run(wrapped, item.test_a, timeout_s=60)
            code_for_b = _build_ds1000_for_test_b(parsed["code"], item.perturbed_prompt)
            result_b = sandbox_ds1000.run(code_for_b, item.test_b, timeout_s=60)
        elif item.source == "humaneval":
            resp = client.call(
                config, prompt=item.perturbed_prompt, system=SYSTEM_PROMPT_MBPP
            )
            parsed = parse_response(resp.choices[0])
            test_a = ensure_humaneval_test_invocation(item.test_a, item.entry_point)
            test_b = ensure_humaneval_test_invocation(item.test_b, item.entry_point)
            result_a, result_b = sandbox_default.run_dual_blind(
                parsed["code"], test_a, test_b
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
    return sample


def evaluate_item(
    item: BenchmarkItem,
    client: LLMClient,
    config: ModelConfig,
    sandbox_default: Sandbox,
    sandbox_ds1000: Sandbox | None,
    n_samples: int,
    sample_workers: int = 5,
) -> dict:
    """Run n_samples (LLM-call + sandbox) tuples concurrently, then aggregate.

    `sample_workers` caps how many of the n_samples run in parallel. With
    n_samples=5 and sample_workers=5 (default), all 5 LLM calls go out at once
    — provided the API accepts concurrent calls. Set sample_workers=1 to fall
    back to v1 sequential behavior.
    """
    samples: list[dict | None] = [None] * n_samples
    workers = max(1, min(sample_workers, n_samples))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(
                _run_one_perturbed_sample,
                i, item, client, config, sandbox_default, sandbox_ds1000,
            ): i for i in range(n_samples)
        }
        for fut in as_completed(futs):
            i = futs[fut]
            samples[i] = fut.result()
    samples = [s for s in samples if s is not None]

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


def _pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al., 2021)."""
    from math import comb
    if k > n or n - c < k:
        return 1.0 if c >= 1 and k <= n else (0.0 if c == 0 else 1.0)
    return 1.0 - comb(n - c, k) / comb(n, k)


def _mean_pass_at_k(records: list[dict], count_field: str, k: int) -> float | None:
    """Mean per-item pass@k over a count field. None if any n < k."""
    if not records or any(r["n_samples"] < k for r in records):
        return None
    return sum(
        _pass_at_k(r["n_samples"], r[count_field], k) for r in records
    ) / len(records)


def print_summary(records: list[dict], model_id: str, k_values: list[int] = (1, 3)) -> None:
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
    print(f"\n── Aggregate pass rates (a sample can satisfy both tests) ──")
    print(f"pass@k(A):       {total_a/total_samp:.1%}  ({total_a}/{total_samp})")
    print(f"pass@k(B):       {total_b/total_samp:.1%}  ({total_b}/{total_samp})")
    print(f"pass@k(either):  {total_either/total_samp:.1%}  ({total_either}/{total_samp})")

    # Unbiased pass@k for each k in k_values
    for k in k_values:
        a_k = _mean_pass_at_k(records, "pass_a_count", k)
        b_k = _mean_pass_at_k(records, "pass_b_count", k)
        e_k = _mean_pass_at_k(records, "pass_either_count", k)
        if a_k is None:
            print(f"\n── pass@{k} (unbiased) ──  (n_samples < {k} for some item, skipped)")
            continue
        print(f"\n── pass@{k} (unbiased) ──")
        print(f"pass@{k}(A):       {a_k:.1%}")
        print(f"pass@{k}(B):       {b_k:.1%}")
        print(f"pass@{k}(either):  {e_k:.1%}")

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
    parser.add_argument(
        "--benchmark", type=Path, default=BENCHMARK_PATH,
        help=f"Benchmark JSONL file (default: {BENCHMARK_PATH})"
    )
    parser.add_argument(
        "--sample-workers", type=int, default=5,
        help="Concurrent LLM calls per item (default: 5 = full n_samples parallelism)"
    )
    args = parser.parse_args()

    # Load benchmark items
    items: list[BenchmarkItem] = []
    with open(args.benchmark) as f:
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
                sample_workers=args.sample_workers,
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
