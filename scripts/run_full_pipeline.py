#!/usr/bin/env python3
"""End-to-end AmbiCode-Eval pipeline: baseline → perturbed → classify → analyze.

Runs all four phases in sequence for a single model and saves all outputs to
data/results/. Judge model is auto-selected to avoid same-family circularity:
  - Claude models are judged by gpt-5.4-mini
  - All other models are judged by claude-haiku (Claude Haiku 4.5 via OpenRouter)

Usage:
    # Full run, 5 samples per item, MBPP only
    python scripts/run_full_pipeline.py \\
        --model claude-sonnet --n-samples 5 --temperature 0.8 --skip-ds1000

    # Full run including DS-1000 (requires Docker + ambicode-ds1000 image)
    python scripts/run_full_pipeline.py --model claude-sonnet --n-samples 5 --temperature 0.8

    # Quick sanity check — 3 items, 1 sample
    python scripts/run_full_pipeline.py --model claude-sonnet --limit 3 --skip-ds1000

    # Override judge model manually
    python scripts/run_full_pipeline.py --model gpt-5.4 --judge-model gpt-5.4-mini
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPTS_DIR.parent / "data" / "results"

# Auto-selection: Claude models must not be judged by any Claude model.
_CLAUDE_JUDGE_FALLBACK = "gpt-5.4-mini"
_DEFAULT_JUDGE = "claude-haiku"


def _auto_judge(tested_model: str) -> str:
    """Pick a judge that is not in the same model family as the tested model."""
    if "claude" in tested_model.lower():
        print(
            f"[pipeline] Tested model is Claude family — using {_CLAUDE_JUDGE_FALLBACK} "
            f"as judge to avoid same-family circularity."
        )
        return _CLAUDE_JUDGE_FALLBACK
    return _DEFAULT_JUDGE


def _run(cmd: list[str], step_name: str) -> None:
    """Run a subprocess command and exit on failure."""
    print(f"\n{'='*65}")
    print(f"  {step_name}")
    print(f"  {' '.join(cmd)}")
    print(f"{'='*65}\n")
    t0 = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[pipeline] {step_name} failed (exit {result.returncode}). Stopping.")
        sys.exit(result.returncode)
    print(f"\n[pipeline] {step_name} done in {elapsed:.1f}s")


def _latest_file(pattern: str) -> str:
    """Return the most recent file matching a glob pattern in RESULTS_DIR."""
    matches = sorted(RESULTS_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No file found matching '{pattern}' in {RESULTS_DIR}. "
            "Did the previous step complete successfully?"
        )
    return str(matches[-1])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end AmbiCode-Eval pipeline for one model."
    )
    parser.add_argument(
        "--model", required=True,
        help="Model alias from config/models.yaml (e.g. claude-sonnet, gpt-5.4)"
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument(
        "--n-samples", type=int, default=5,
        help="Samples per item (default: 5 for pass@5)"
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
        "--judge-model", default=None,
        help="Override auto-selected judge model for SA/EA/AC classification"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory for analysis plots (default: data/results/analysis_<model>_<ts>/)"
    )
    parser.add_argument(
        "--benchmark", default=None,
        help="Benchmark JSONL file (default: data/benchmark/benchmark.jsonl)"
    )
    parser.add_argument(
        "--sample-workers", type=int, default=5,
        help="Concurrent LLM calls per item (default: 5 = full n_samples parallelism)"
    )
    args = parser.parse_args()

    judge_model = args.judge_model or _auto_judge(args.model)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or str(
        RESULTS_DIR / f"analysis_{args.model}_{timestamp}"
    )

    python = sys.executable
    safe_model = args.model.replace("/", "_")

    # ── Shared flags for eval scripts ─────────────────────────────────────────
    eval_flags: list[str] = [
        "--model", args.model,
        "--temperature", str(args.temperature),
        "--n-samples", str(args.n_samples),
        "--max-tokens", str(args.max_tokens),
        "--source", args.source,
    ]
    if args.limit:
        eval_flags += ["--limit", str(args.limit)]
    if args.skip_ds1000:
        eval_flags.append("--skip-ds1000")
    if args.benchmark:
        eval_flags += ["--benchmark", args.benchmark]
    eval_flags += ["--sample-workers", str(args.sample_workers)]

    print(f"\n{'#'*65}")
    print(f"  AmbiCode-Eval Full Pipeline")
    print(f"  Tested model : {args.model}")
    print(f"  Judge model  : {judge_model}")
    print(f"  n_samples    : {args.n_samples}  temperature: {args.temperature}")
    print(f"  Started at   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*65}")

    t_total = time.time()

    # ── Step 1: Baseline eval ─────────────────────────────────────────────────
    _run(
        [python, str(SCRIPTS_DIR / "run_baseline_eval.py")] + eval_flags,
        "Step 1/4 — Baseline evaluation (clean prompts)"
    )

    # ── Step 2: Perturbed eval ────────────────────────────────────────────────
    _run(
        [python, str(SCRIPTS_DIR / "run_perturbed_eval.py")] + eval_flags,
        "Step 2/4 — Perturbed evaluation (ambiguous prompts)"
    )

    # ── Step 3: SA/EA/AC classification ──────────────────────────────────────
    perturbed_file = _latest_file(f"perturbed_{safe_model}_*.jsonl")
    _run(
        [python, str(SCRIPTS_DIR / "run_classification.py"),
         "--input", perturbed_file,
         "--judge-model", judge_model,
         "--max-tokens", "256"],
        "Step 3/4 — SA/EA/AC behavioral classification"
    )

    # ── Step 4: Analysis + plots ──────────────────────────────────────────────
    baseline_file = _latest_file(f"baseline_{safe_model}_*.jsonl")
    classified_file = _latest_file(f"classified_{safe_model}_*.jsonl")
    _run(
        [python, str(SCRIPTS_DIR / "analyze_results.py"),
         "--baseline", baseline_file,
         "--perturbed", perturbed_file,
         "--classified", classified_file,
         "--label", args.model,
         "--output", output_dir],
        "Step 4/4 — Analysis and plot generation"
    )

    total_elapsed = time.time() - t_total
    print(f"\n{'#'*65}")
    print(f"  Pipeline complete in {total_elapsed/60:.1f} min")
    print(f"  Results : {RESULTS_DIR}/")
    print(f"  Plots   : {output_dir}/")
    print(f"{'#'*65}\n")


if __name__ == "__main__":
    main()
