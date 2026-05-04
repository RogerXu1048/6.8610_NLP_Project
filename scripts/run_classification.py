#!/usr/bin/env python3
"""Phase 3: Classify model responses as SA / EA / AC using an LLM judge.

Reads perturbed eval JSONL, runs each sample's response through a judge model
that answers three boolean questions, then derives the behavioral label
deterministically:

    Q1=Y              → AC  (Active Clarification — asked a question)
    Q1=N, Q2=Y, Q3=Y  → EA  (Explicit Assumption — stated premise + code)
    Q1=N, Q2=Y, Q3=N  → SA  (Silent Assumption — code only)
    Q1=N, Q2=N        → unclassifiable  (no code, no question)

Usage:
    python scripts/run_classification.py \\
        --input data/results/perturbed_gpt-5.4_20260504_120000.jsonl

    # If testing a Claude model, specify a non-Claude judge:
    python scripts/run_classification.py \\
        --input data/results/perturbed_claude-sonnet_*.jsonl \\
        --judge-model gpt-5.4-mini

Output: data/results/classified_<model>_<timestamp>.jsonl
        (same structure as input + behavior_label/q1/q2/q3/rationale per sample)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.prompts import get_prompt, render_prompt
from src.util.llm import LLMClient, ModelConfig
from src.util.parsing import parse_json_response

# ── Paths ─────────────────────────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "results"

# ── Label derivation ──────────────────────────────────────────────────────────


def derive_label(q1: bool, q2: bool, q3: bool) -> str:
    """Deterministic SA/EA/AC label from the three boolean rubric answers."""
    if q1:
        return "AC"
    if q2 and q3:
        return "EA"
    if q2 and not q3:
        return "SA"
    return "unclassifiable"


# ── Per-sample classification ─────────────────────────────────────────────────


def classify_sample(
    sample: dict,
    item: dict,
    client: LLMClient,
    config: ModelConfig,
) -> dict:
    """Call LLM judge and attach behavior fields to the sample dict."""
    system = get_prompt("classification.sa_ea_ac_judge.system")

    code = sample.get("generated_code", "").strip()
    code_block_display = f"```python\n{code}\n```" if code else "(none)"

    task_prompt = render_prompt(
        "classification.sa_ea_ac_judge.task",
        perturbed_prompt=item.get("perturbed_prompt", "(not available)"),
        think_block=sample.get("think_block") or "(none)",
        prose=sample.get("prose") or "(none)",
        code_block=code_block_display,
    )

    try:
        resp = client.call(config, prompt=task_prompt, system=system)
        parsed = parse_json_response(resp.choices[0])
        q1 = bool(parsed.get("q1_question_present", False))
        q2 = bool(parsed.get("q2_code_present", False))
        q3 = bool(parsed.get("q3_explicit_assumption", False))
        label = derive_label(q1, q2, q3)
        rationale = parsed.get("rationale", "")
    except Exception as exc:
        q1, q2, q3 = False, bool(code), False
        label = "error"
        rationale = str(exc)[:300]

    return {
        **sample,
        "behavior_label": label,
        "behavior_q1_question": q1,
        "behavior_q2_code": q2,
        "behavior_q3_assumption": q3,
        "behavior_rationale": rationale,
    }


# ── I/O helpers ───────────────────────────────────────────────────────────────


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def resolve_input_path(raw: str) -> Path:
    """Resolve a path string, expanding shell globs against RESULTS_DIR."""
    p = Path(raw)
    if p.exists():
        return p
    # Try glob expansion (user may have passed a pattern)
    matches = sorted(RESULTS_DIR.glob(p.name))
    if matches:
        return matches[-1]  # most recent
    raise FileNotFoundError(f"No file found matching: {raw}")


# ── Summary ───────────────────────────────────────────────────────────────────


def print_summary(label_counts: dict[str, int], judge_model_id: str) -> None:
    total = sum(label_counts.values())
    print(f"\n{'='*55}")
    print(f"Judge model: {judge_model_id}")
    print(f"Samples classified: {total}")
    print(f"\n── Label distribution ─────────────────────────────────")
    for label in ["SA", "EA", "AC", "unclassifiable", "error"]:
        count = label_counts.get(label, 0)
        if count:
            bar = "█" * int(count / total * 30)
            print(f"  {label:15s}  {count:4d}  ({count/total:.1%})  {bar}")


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3: SA/EA/AC behavioral classification."
    )
    parser.add_argument(
        "--input", required=True,
        help="Perturbed eval JSONL path (data/results/perturbed_*.jsonl)"
    )
    parser.add_argument(
        "--judge-model", default="claude-haiku",
        help="Judge model alias (default: claude-haiku). "
             "Use a non-Claude model if the tested model is Claude."
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N items (for quick sanity checks)"
    )
    args = parser.parse_args()

    input_path = resolve_input_path(args.input)
    records = load_jsonl(input_path)
    if args.limit:
        records = records[: args.limit]

    client = LLMClient()
    config = ModelConfig(
        model=args.judge_model,
        temperature=0.0,
        max_tokens=args.max_tokens,
    )

    print(f"Input:       {input_path.name}")
    print(f"Items:       {len(records)}")
    print(f"Judge model: {config.model_id}")
    print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem.replace("perturbed_", "classified_")
    out_path = RESULTS_DIR / f"{stem}.jsonl"

    label_counts: dict[str, int] = {}
    t_start = time.time()

    with open(out_path, "w") as out_f:
        for i, item in enumerate(records, 1):
            n_samples = len(item.get("samples", []))
            print(
                f"[{i:2d}/{len(records)}] {item['task_id']}  "
                f"({item['source']}, {item['ambiguity_type']})  "
                f"{n_samples} sample(s)...",
                end="  ",
                flush=True,
            )
            t0 = time.time()

            augmented_samples = []
            item_labels = []
            for sample in item.get("samples", []):
                classified = classify_sample(sample, item, client, config)
                augmented_samples.append(classified)
                lbl = classified["behavior_label"]
                item_labels.append(lbl)
                label_counts[lbl] = label_counts.get(lbl, 0) + 1

            elapsed = time.time() - t0
            print(f"{item_labels}  ({elapsed:.1f}s)")

            augmented_item = {**item, "samples": augmented_samples}
            out_f.write(json.dumps(augmented_item) + "\n")
            out_f.flush()

    print_summary(label_counts, config.model_id)
    print(f"\nTotal time: {time.time() - t_start:.1f}s")
    print(f"Output:     {out_path}")


if __name__ == "__main__":
    main()
