#!/usr/bin/env python3
"""Analyze Phase 2 + Phase 3 results and produce tables and plots.

Auto-discovers result files in data/results/ grouped by model name, or accepts
explicit paths. Prints summary tables, saves PNG plots, and persists metrics to
data/results/metrics_summary.json (merged across runs) and metrics_summary.csv.

Usage:
    # Auto-discover all runs in data/results/
    python scripts/analyze_results.py

    # Point to specific files for one model
    python scripts/analyze_results.py \\
        --baseline  data/results/baseline_gpt-5.4_*.jsonl \\
        --perturbed data/results/perturbed_gpt-5.4_*.jsonl \\
        --classified data/results/classified_gpt-5.4_*.jsonl \\
        --label "GPT-5.4"

    # Save plots to a directory
    python scripts/analyze_results.py --output data/results/analysis/
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "results"
AMBIGUITY_TYPES = [
    "coreferential", "syntactic", "scopal",
    "collective_distributive", "elliptical",
]
LABELS = ["SA", "EA", "AC", "unclassifiable", "error"]

# pass@k values computed by default. k must satisfy k <= n_samples for every
# item; values where n < k are silently skipped (with a printed warning).
K_VALUES = [1, 3]


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al., 2021, "Evaluating LLMs on Code").

    Probability that at least one of k randomly drawn samples (without
    replacement) from n total passes, given c of the n actually passed.

        pass@k = 1 - C(n - c, k) / C(n, k)

    Edge cases:
      - c == 0          -> 0   (no successes possible)
      - c == n          -> 1   (every sample passes)
      - n - c < k       -> 1   (fewer than k failures, so at least one of any
                                 k must succeed)
      - k > n           -> raises ValueError (caller should pre-filter)
    """
    if k > n:
        raise ValueError(f"pass@{k} requires n >= {k}, got n={n}")
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def mean_pass_at_k(records: list[dict], count_field: str, k: int) -> float | None:
    """Mean pass@k across items (per-item pass@k → average over items).

    Returns None if any item has n_samples < k (cannot estimate).
    """
    if not records:
        return None
    if any(r["n_samples"] < k for r in records):
        return None
    return sum(
        pass_at_k(r["n_samples"], r[count_field], k) for r in records
    ) / len(records)
LBL_COLORS = {
    "SA": "#EF5350",
    "EA": "#66BB6A",
    "AC": "#42A5F5",
    "unclassifiable": "#BDBDBD",
    "error": "#000000",
}

# ── I/O helpers ───────────────────────────────────────────────────────────────


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def resolve_glob(pattern: str) -> Path | None:
    """Resolve a glob pattern, returning the most recent match."""
    p = Path(pattern)
    if p.exists():
        return p
    matches = sorted(RESULTS_DIR.glob(p.name))
    return matches[-1] if matches else None


def extract_model_name(filename: str) -> str:
    """Parse model name from result filenames like baseline_gpt-5.4_20260504.jsonl."""
    parts = filename.split("_")
    if len(parts) >= 3:
        return "_".join(parts[1:-2]) if len(parts) > 3 else parts[1]
    return filename


def auto_discover(results_dir: Path) -> dict[str, dict[str, Path]]:
    """Group result files by model name: {model: {baseline/perturbed/classified: Path}}."""
    groups: dict[str, dict[str, Path]] = defaultdict(dict)
    for prefix in ("baseline", "perturbed", "classified"):
        for f in sorted(results_dir.glob(f"{prefix}_*.jsonl")):
            model = extract_model_name(f.stem)
            if prefix not in groups[model] or f > groups[model][prefix]:
                groups[model][prefix] = f
    return dict(groups)


# ── Metric helpers ────────────────────────────────────────────────────────────


def pass_rates(records: list[dict], key: str = "pass_count") -> dict[str, float | None]:
    """Compute pass@k for baseline records.

    Returns a dict with:
      - rate           : aggregate c/n across all samples (== pass@1 estimate)
      - pass_at_<k>    : per-item pass@k averaged across items, for k in K_VALUES
    """
    total = sum(r["n_samples"] for r in records)
    passed = sum(r[key] for r in records)
    out: dict[str, float | None] = {
        "rate": (passed / total if total else 0.0),
    }
    for k in K_VALUES:
        out[f"pass_at_{k}"] = mean_pass_at_k(records, key, k)
    return out


def perturbed_rates(records: list[dict]) -> dict[str, float]:
    """Compute aggregate pass / choice rates across all perturbed records.

    Returns a dict with:
      - pass_a_rate / pass_b_rate / pass_either_rate
            test-level pass rates (a sample can contribute to multiple).
      - chose_a_rate / chose_b_rate / both_pass_rate / neither_rate
            mutually-exclusive choice decomposition (sums to 1):
              chose_a:   passed_a only -> model picked A
              chose_b:   passed_b only -> model picked B
              both_pass: tests cannot distinguish (don't use for bias)
              neither:   code failed both
      - interp_a_bias = chose_a / (chose_a + chose_b), or None if no decisive
            samples — what fraction of decisive samples picked A.

    Falls back to deriving from samples for legacy files missing aggregate fields.
    """
    total = sum(r["n_samples"] for r in records)
    if not total:
        return {
            "pass_a_rate": 0.0, "pass_b_rate": 0.0, "pass_either_rate": 0.0,
            "chose_a_rate": 0.0, "chose_b_rate": 0.0,
            "both_pass_rate": 0.0, "neither_rate": 0.0,
            "interp_a_bias": None,
        }

    pass_a = sum(r["pass_a_count"] for r in records)
    pass_b = sum(r["pass_b_count"] for r in records)
    chose_a = chose_b = both = 0
    pass_either = 0
    # For pass@k we also need per-item counts of each event. Backfill missing
    # aggregate fields onto records so pass_at_k() can work uniformly.
    for r in records:
        if "chose_a_count" in r:
            chose_a += r["chose_a_count"]
            chose_b += r["chose_b_count"]
            both += r.get("pass_both_count", 0)
            pass_either += r.get("pass_either_count", 0)
        else:
            ca = cb = bo = pe = 0
            for s in r.get("samples", []):
                pa, pb = s.get("passed_a"), s.get("passed_b")
                if pa and pb:
                    bo += 1
                elif pa:
                    ca += 1
                elif pb:
                    cb += 1
                if pa or pb:
                    pe += 1
            r["chose_a_count"] = ca
            r["chose_b_count"] = cb
            r["pass_both_count"] = bo
            r["pass_either_count"] = pe
            chose_a += ca
            chose_b += cb
            both += bo
            pass_either += pe
    neither = total - chose_a - chose_b - both
    decisive = chose_a + chose_b

    out: dict[str, float | None] = {
        # Test-level pass rates (aggregate c/n; equivalent to pass@1 estimator)
        "pass_a_rate": pass_a / total,
        "pass_b_rate": pass_b / total,
        "pass_either_rate": pass_either / total,
        # Choice decomposition
        "chose_a_rate": chose_a / total,
        "chose_b_rate": chose_b / total,
        "both_pass_rate": both / total,
        "neither_rate": neither / total,
        "interp_a_bias": (chose_a / decisive) if decisive else None,
    }

    # Unbiased pass@k for k in K_VALUES (per-item average; None if any n < k).
    for k in K_VALUES:
        out[f"pass_a_at_{k}"] = mean_pass_at_k(records, "pass_a_count", k)
        out[f"pass_b_at_{k}"] = mean_pass_at_k(records, "pass_b_count", k)
        out[f"pass_either_at_{k}"] = mean_pass_at_k(records, "pass_either_count", k)
        out[f"chose_a_at_{k}"] = mean_pass_at_k(records, "chose_a_count", k)
        out[f"chose_b_at_{k}"] = mean_pass_at_k(records, "chose_b_count", k)

    return out


def label_distribution(classified: list[dict]) -> dict[str, float]:
    """Fraction of samples with each SA/EA/AC label."""
    counts: dict[str, int] = defaultdict(int)
    total = 0
    for item in classified:
        for s in item.get("samples", []):
            lbl = s.get("behavior_label", "error")
            counts[lbl] += 1
            total += 1
    return {lbl: counts[lbl] / total for lbl in LABELS} if total else {}


def label_by_risk_level(classified: list[dict]) -> dict[str, dict[str, float]]:
    """For each risk level, fraction of samples with each SA/EA/AC label."""
    buckets: dict[str, dict[str, int]] = {
        risk: {lbl: 0 for lbl in LABELS} | {"total": 0}
        for risk in ("low", "high")
    }
    for item in classified:
        risk = item.get("risk_level", "low")
        if risk not in buckets:
            continue
        for s in item.get("samples", []):
            lbl = s.get("behavior_label")
            if lbl in LABELS:
                buckets[risk][lbl] += 1
                buckets[risk]["total"] += 1
    result = {}
    for risk, b in buckets.items():
        t = b["total"]
        if t:
            result[risk] = {lbl: b[lbl] / t for lbl in LABELS}
    return result


def label_by_correctness(classified: list[dict]) -> dict[str, dict[str, float]]:
    """For each label, fraction of samples that passed_a / passed_b / neither."""
    buckets: dict[str, dict[str, int]] = {
        lbl: {"pass_a": 0, "pass_b": 0, "neither": 0, "total": 0}
        for lbl in LABELS
    }
    for item in classified:
        for s in item.get("samples", []):
            lbl = s.get("behavior_label")
            if lbl not in buckets:
                continue
            buckets[lbl]["total"] += 1
            if s.get("passed_a"):
                buckets[lbl]["pass_a"] += 1
            elif s.get("passed_b"):
                buckets[lbl]["pass_b"] += 1
            else:
                buckets[lbl]["neither"] += 1
    result = {}
    for lbl, b in buckets.items():
        t = b["total"]
        if t:
            result[lbl] = {
                "pass_a": b["pass_a"] / t,
                "pass_b": b["pass_b"] / t,
                "neither": b["neither"] / t,
                "total": t,
            }
    return result


def _item_pass_either_rate(record: dict) -> float:
    """Get pass_either_rate from a perturbed record, falling back to samples."""
    if "pass_either_rate" in record:
        return record["pass_either_rate"]
    samples = record.get("samples", [])
    if not samples:
        return 0.0
    either = sum(1 for s in samples if s.get("passed_a") or s.get("passed_b"))
    return either / len(samples)


def compute_tax_by_type(data: dict) -> dict[str, float]:
    """Return {ambiguity_type: avg_tax_pp} based on pass_either (per-item, then mean)."""
    perturbed = data.get("perturbed_records", [])
    baseline = data.get("baseline_records", [])
    if not perturbed or not baseline:
        return {}
    baseline_map = {r["task_id"]: r for r in baseline}
    type_taxes: dict[str, list[float]] = defaultdict(list)
    for r in perturbed:
        b = baseline_map.get(r["task_id"])
        if b:
            amb = r.get("ambiguity_type", "")
            if amb:
                type_taxes[amb].append(
                    (b.get("pass_rate", 0) - _item_pass_either_rate(r)) * 100
                )
    return {amb: sum(v) / len(v) for amb, v in type_taxes.items()}


# ── Metrics persistence ───────────────────────────────────────────────────────


def save_metrics(runs: dict[str, dict], results_dir: Path) -> None:
    """Persist computed metrics to JSON + CSV, merging with any prior model runs."""
    summary_path = results_dir / "metrics_summary.json"
    csv_path = results_dir / "metrics_summary.csv"

    # Load existing summary so prior model runs are preserved
    existing: dict = {}
    if summary_path.exists():
        with open(summary_path) as f:
            existing = json.load(f)

    for model, data in runs.items():
        baseline_rate = data.get("baseline_rate")
        perturbed_a = data.get("perturbed_a_rate")
        perturbed_b = data.get("perturbed_b_rate")
        perturbed_either = data.get("perturbed_either_rate")

        # Primary Ambiguity Tax: baseline - pass_either
        # (a sample is "successful" if it passes EITHER interpretation, since
        # both are valid given the ambiguous prompt). Falls back to pass_a if
        # pass_either is unavailable (e.g. legacy results file).
        if baseline_rate is not None and perturbed_either is not None:
            tax_pp = (baseline_rate - perturbed_either) * 100
        elif baseline_rate is not None and perturbed_a is not None:
            tax_pp = (baseline_rate - perturbed_a) * 100
        else:
            tax_pp = None

        # Conditional pass@k: how often the model produced code matching each
        # interpretation. tax_a_pp / tax_b_pp report the per-interpretation tax.
        tax_a_pp = (
            (baseline_rate - perturbed_a) * 100
            if baseline_rate is not None and perturbed_a is not None
            else None
        )
        tax_b_pp = (
            (baseline_rate - perturbed_b) * 100
            if baseline_rate is not None and perturbed_b is not None
            else None
        )

        # Unbiased pass@k metrics + corresponding tax (uses pass_either as
        # primary "pass" definition under ambiguity).
        pass_at_k_metrics: dict[str, float | None] = {}
        for k in K_VALUES:
            base_k = data.get(f"baseline_pass_at_{k}")
            either_k = data.get(f"pass_either_at_{k}")
            pass_at_k_metrics[f"baseline_pass_at_{k}"] = base_k
            pass_at_k_metrics[f"pass_a_at_{k}"] = data.get(f"pass_a_at_{k}")
            pass_at_k_metrics[f"pass_b_at_{k}"] = data.get(f"pass_b_at_{k}")
            pass_at_k_metrics[f"pass_either_at_{k}"] = either_k
            pass_at_k_metrics[f"chose_a_at_{k}"] = data.get(f"chose_a_at_{k}")
            pass_at_k_metrics[f"chose_b_at_{k}"] = data.get(f"chose_b_at_{k}")
            pass_at_k_metrics[f"ambiguity_tax_at_{k}_pp"] = (
                (base_k - either_k) * 100
                if base_k is not None and either_k is not None
                else None
            )

        existing[model] = {
            "baseline_rate": baseline_rate,
            # Test-level pass rates (aggregate; equivalent to pass@1)
            "perturbed_a_rate": perturbed_a,
            "perturbed_b_rate": perturbed_b,
            "perturbed_either_rate": perturbed_either,
            "ambiguity_tax_pp": tax_pp,
            "tax_a_pp": tax_a_pp,
            "tax_b_pp": tax_b_pp,
            # Choice decomposition (mutually exclusive, sums to 1)
            "chose_a_rate": data.get("chose_a_rate"),
            "chose_b_rate": data.get("chose_b_rate"),
            "both_pass_rate": data.get("both_pass_rate"),
            "neither_rate": data.get("neither_rate"),
            "interp_a_bias": data.get("interp_a_bias"),
            # Unbiased pass@k metrics for k in K_VALUES
            **pass_at_k_metrics,
            "label_dist": data.get("label_dist", {}),
            "label_by_risk": data.get("label_by_risk", {}),
            "tax_by_type": compute_tax_by_type(data),
            "per_item_deltas": data.get("per_item_deltas", []),
        }

    with open(summary_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"Saved metrics JSON: {summary_path}")

    # pass@k field names, ordered for stable CSV columns
    pass_at_k_fields: list[str] = []
    for k in K_VALUES:
        pass_at_k_fields.extend([
            f"baseline_pass_at_{k}",
            f"pass_a_at_{k}",
            f"pass_b_at_{k}",
            f"pass_either_at_{k}",
            f"chose_a_at_{k}",
            f"chose_b_at_{k}",
            f"ambiguity_tax_at_{k}_pp",
        ])

    # Write CSV (one row per model, flat fields only)
    csv_rows = []
    for model, m in existing.items():
        row = {
            "model": model,
            "baseline_rate": m.get("baseline_rate", ""),
            "perturbed_a_rate": m.get("perturbed_a_rate", ""),
            "perturbed_b_rate": m.get("perturbed_b_rate", ""),
            "perturbed_either_rate": m.get("perturbed_either_rate", ""),
            "ambiguity_tax_pp": m.get("ambiguity_tax_pp", ""),
            "tax_a_pp": m.get("tax_a_pp", ""),
            "tax_b_pp": m.get("tax_b_pp", ""),
            "chose_a_rate": m.get("chose_a_rate", ""),
            "chose_b_rate": m.get("chose_b_rate", ""),
            "both_pass_rate": m.get("both_pass_rate", ""),
            "neither_rate": m.get("neither_rate", ""),
            "interp_a_bias": m.get("interp_a_bias", ""),
        }
        for f in pass_at_k_fields:
            row[f] = m.get(f, "")
        for lbl in LABELS:
            row[lbl] = m.get("label_dist", {}).get(lbl, "")
        for amb in AMBIGUITY_TYPES:
            row[f"tax_{amb}_pp"] = m.get("tax_by_type", {}).get(amb, "")
        csv_rows.append(row)

    fieldnames = (
        [
            "model",
            "baseline_rate",
            "perturbed_a_rate",
            "perturbed_b_rate",
            "perturbed_either_rate",
            "ambiguity_tax_pp",
            "tax_a_pp",
            "tax_b_pp",
            "chose_a_rate",
            "chose_b_rate",
            "both_pass_rate",
            "neither_rate",
            "interp_a_bias",
        ]
        + pass_at_k_fields
        + LABELS
        + [f"tax_{a}_pp" for a in AMBIGUITY_TYPES]
    )
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved metrics CSV:  {csv_path}")


# ── Table printing ────────────────────────────────────────────────────────────


def print_pass_at_k_table(runs: dict[str, dict]) -> None:
    """Show unbiased pass@k (Chen et al. 2021) for each k in K_VALUES."""
    cols = ["Clean", "Pert-A", "Pert-B", "Pert-Either", "Tax(pp)"]
    for k in K_VALUES:
        header_cols = "  ".join(f"{c:>10}" for c in cols)
        header = f"{'Model':<22} {header_cols}"
        print(f"\n{'='*len(header)}")
        print(f"Table — pass@{k} (unbiased estimator)  "
              f"Tax = baseline − pass_either, both at @{k}")
        print(f"{'='*len(header)}")
        print(header)
        print("-" * len(header))
        for model, data in sorted(runs.items()):
            base = data.get(f"baseline_pass_at_{k}")
            pa = data.get(f"pass_a_at_{k}")
            pb = data.get(f"pass_b_at_{k}")
            pe = data.get(f"pass_either_at_{k}")
            tax = (base - pe) * 100 if base is not None and pe is not None else None
            cells = [
                f"{base:.1%}" if base is not None else "—",
                f"{pa:.1%}" if pa is not None else "—",
                f"{pb:.1%}" if pb is not None else "—",
                f"{pe:.1%}" if pe is not None else "—",
                f"{tax:+.1f}" if tax is not None else "—",
            ]
            print(f"  {model:<20} " + "  ".join(f"{c:>10}" for c in cells))


def print_ambiguity_tax_table(runs: dict[str, dict]) -> None:
    header = (
        f"{'Model':<22} {'Clean':>8} {'Pert-A':>8} {'Pert-B':>8} "
        f"{'Pert-Either':>12} {'Tax(pp)':>9}"
    )
    print(f"\n{'='*len(header)}")
    print("Table 1 — Ambiguity Tax  (Tax = Clean − Pert-Either)")
    print(f"{'='*len(header)}")
    print(header)
    print("-" * len(header))
    for model, data in sorted(runs.items()):
        clean_r = data.get("baseline_rate", float("nan"))
        pert_a = data.get("perturbed_a_rate", float("nan"))
        pert_b = data.get("perturbed_b_rate", float("nan"))
        pert_e = data.get("perturbed_either_rate", float("nan"))
        # Primary tax uses pass_either; falls back to pass_a if either is unavailable
        if pert_e == pert_e and clean_r == clean_r:
            tax = (clean_r - pert_e) * 100
        elif pert_a == pert_a and clean_r == clean_r:
            tax = (clean_r - pert_a) * 100
        else:
            tax = float("nan")
        clean_s = f"{clean_r:.1%}" if clean_r == clean_r else "—"
        a_s = f"{pert_a:.1%}" if pert_a == pert_a else "—"
        b_s = f"{pert_b:.1%}" if pert_b == pert_b else "—"
        e_s = f"{pert_e:.1%}" if pert_e == pert_e else "—"
        tax_s = f"{tax:+.1f}" if tax == tax else "—"
        print(f"  {model:<20} {clean_s:>8} {a_s:>8} {b_s:>8} {e_s:>12} {tax_s:>9}")


def print_interpretation_choice_table(runs: dict[str, dict]) -> None:
    """Show the mutually-exclusive 4-way choice decomposition + interp bias."""
    header = (
        f"{'Model':<22} {'Chose-A':>8} {'Chose-B':>8} "
        f"{'Both':>7} {'Neither':>8} {'A-bias':>8}"
    )
    print(f"\n{'='*len(header)}")
    print("Table — Interpretation Choice  "
          "(mutually exclusive; A-bias = chose_a / decisive)")
    print(f"{'='*len(header)}")
    print(header)
    print("-" * len(header))
    for model, data in sorted(runs.items()):
        ca = data.get("chose_a_rate")
        cb = data.get("chose_b_rate")
        if ca is None or cb is None:
            continue
        both = data.get("both_pass_rate", 0)
        neither = data.get("neither_rate", 0)
        bias = data.get("interp_a_bias")
        bias_s = f"{bias:.1%}" if bias is not None else "—"
        print(
            f"  {model:<20} {ca:>7.1%} {cb:>7.1%} {both:>6.1%} "
            f"{neither:>7.1%} {bias_s:>8}"
        )


def print_sa_ea_ac_table(runs: dict[str, dict]) -> None:
    header = (
        f"{'Model':<22} {'SA':>7} {'EA':>7} {'AC':>7} {'Unclass':>9} {'Error':>7}"
    )
    print(f"\n{'='*len(header)}")
    print("Table 2 — Behavioral Label Distribution  "
          "(Error = judge LLM call failed)")
    print(f"{'='*len(header)}")
    print(header)
    print("-" * len(header))
    for model, data in sorted(runs.items()):
        dist = data.get("label_dist", {})
        if not dist:
            continue
        sa = f"{dist.get('SA', 0):.1%}"
        ea = f"{dist.get('EA', 0):.1%}"
        ac = f"{dist.get('AC', 0):.1%}"
        uc = f"{dist.get('unclassifiable', 0):.1%}"
        er = f"{dist.get('error', 0):.1%}"
        print(f"  {model:<20} {sa:>7} {ea:>7} {ac:>7} {uc:>9} {er:>7}")


def print_label_correctness_table(runs: dict[str, dict]) -> None:
    print(f"\n{'='*65}")
    print("Table 3 — Behavioral Label × Interpretation Correctness")
    print(f"{'='*65}")
    for model, data in sorted(runs.items()):
        lbc = data.get("label_by_correctness", {})
        if not lbc:
            continue
        print(f"\n  {model}")
        print(f"  {'Label':<16} {'pass_a':>8} {'pass_b':>8} {'neither':>8} {'n':>6}")
        print(f"  {'-'*50}")
        for lbl in LABELS:
            row = lbc.get(lbl)
            if not row:
                continue
            print(
                f"  {lbl:<16} "
                f"{row['pass_a']:>7.1%} "
                f"{row['pass_b']:>8.1%} "
                f"{row['neither']:>8.1%} "
                f"{row['total']:>6}"
            )


def print_perturbed_by_type(runs: dict[str, dict]) -> None:
    print(f"\n{'='*70}")
    print("Table 4 — pass@k(A) / pass@k(B) by Ambiguity Type")
    print(f"{'='*70}")
    for model, data in sorted(runs.items()):
        perturbed = data.get("perturbed_records", [])
        if not perturbed:
            continue
        print(f"\n  {model}")
        print(f"  {'Ambiguity Type':<28} {'pass_a':>8} {'pass_b':>8} {'n':>5}")
        print(f"  {'-'*55}")
        for amb in AMBIGUITY_TYPES:
            sub = [r for r in perturbed if r.get("ambiguity_type") == amb]
            if not sub:
                continue
            total = sum(r["n_samples"] for r in sub)
            a = sum(r["pass_a_count"] for r in sub)
            b = sum(r["pass_b_count"] for r in sub)
            print(f"  {amb:<28} {a/total:>7.1%} {b/total:>8.1%} {len(sub):>5}")


# ── Plotting ──────────────────────────────────────────────────────────────────


def save_plots(runs: dict[str, dict], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[warn] matplotlib not installed — skipping plots. pip install matplotlib")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    models = sorted(runs.keys())

    # ── Plot: pass@k(A) vs pass@k(B) vs pass@k(either) per model ──────────────
    fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.6), 4))
    x = np.arange(len(models))
    a_rates = [runs[m].get("perturbed_a_rate", 0) for m in models]
    b_rates = [runs[m].get("perturbed_b_rate", 0) for m in models]
    either_rates = [runs[m].get("perturbed_either_rate", 0) for m in models]
    clean_rates = [runs[m].get("baseline_rate", None) for m in models]
    width = 0.25
    ax.bar(x - width, a_rates, width, label="pass@k(A)", color="#2196F3")
    ax.bar(x, b_rates, width, label="pass@k(B)", color="#FF9800")
    ax.bar(x + width, either_rates, width, label="pass@k(either)", color="#4CAF50")
    for xi, cr in zip(x, clean_rates):
        if cr is not None:
            ax.axhline(y=cr, xmin=(xi - 0.5) / len(models),
                       xmax=(xi + 0.5) / len(models),
                       color="black", linewidth=1.2, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Pass Rate")
    ax.set_title("Conditional pass@k(A) / pass@k(B) / pass@k(either) by Model\n(dashed = clean baseline)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    _save(plt, output_dir / "plot_pass_rates.png")

    # ── Plot E: Ambiguity Tax per model (uses pass@k(either)) ─────────────────
    # Tax = baseline - pass_either: code is "successful" if it satisfies EITHER
    # interpretation (since both are valid given the ambiguous prompt).
    models_with_both = [
        m for m in models
        if runs[m].get("baseline_rate") is not None
        and runs[m].get("perturbed_either_rate") is not None
    ]
    if models_with_both:
        fig, ax = plt.subplots(figsize=(max(5, len(models_with_both) * 1.4), 4))
        taxes = [
            (runs[m]["baseline_rate"] - runs[m]["perturbed_either_rate"]) * 100
            for m in models_with_both
        ]
        bar_colors = ["#EF5350" if t >= 0 else "#42A5F5" for t in taxes]
        xm = np.arange(len(models_with_both))
        ax.bar(xm, taxes, color=bar_colors)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(xm)
        ax.set_xticklabels(models_with_both, rotation=15, ha="right")
        ax.set_ylabel("Ambiguity Tax (pp)")
        ax.set_title("Ambiguity Tax by Model\n(baseline pass@k − perturbed pass@k(either))")
        plt.tight_layout()
        _save(plt, output_dir / "plot_ambiguity_tax_all_models.png")

    # ── Plot D: Tax by ambiguity type, one subplot per model ─────────────────
    models_with_tax = [m for m in models if compute_tax_by_type(runs[m])]
    if models_with_tax:
        ncols = len(models_with_tax)
        fig, axes = plt.subplots(1, ncols, figsize=(max(6, ncols * 4), 4), squeeze=False)
        for col, model in enumerate(models_with_tax):
            ax = axes[0][col]
            tbt = compute_tax_by_type(runs[model])
            vals = [tbt.get(a, 0) for a in AMBIGUITY_TYPES]
            bar_colors = ["#EF5350" if v >= 0 else "#42A5F5" for v in vals]
            ax.bar(AMBIGUITY_TYPES, vals, color=bar_colors)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_ylabel("Ambiguity Tax (pp)")
            ax.set_title(f"{model}\nTax by Ambiguity Type")
            ax.set_ylim(
                min(min(vals) - 5, -5),
                max(max(vals) + 5, 5),
            )
            plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
        plt.tight_layout()
        _save(plt, output_dir / "plot_tax_by_type_per_model.png")

    # ── Plot F: Tax by ambiguity type, aggregated across all models ───────────
    agg_type_taxes: dict[str, list[float]] = defaultdict(list)
    for data in runs.values():
        for amb, tax in compute_tax_by_type(data).items():
            agg_type_taxes[amb].append(tax)

    fig, ax = plt.subplots(figsize=(8, 4))
    agg_vals = [
        sum(agg_type_taxes[a]) / len(agg_type_taxes[a]) if agg_type_taxes[a] else 0
        for a in AMBIGUITY_TYPES
    ]
    bar_colors = ["#EF5350" if v >= 0 else "#42A5F5" for v in agg_vals]
    ax.bar(AMBIGUITY_TYPES, agg_vals, color=bar_colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Ambiguity Tax (pp)")
    ax.set_title(
        f"Ambiguity Tax by Type — Aggregated ({len(models)} model(s))\n"
        "(baseline pass@k − perturbed pass@k(A))"
    )
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    _save(plt, output_dir / "plot_tax_by_type_aggregated.png")

    # ── Plot A: Per-item pass@k delta (one subplot per model) ────────────────
    models_with_deltas = [m for m in models if runs[m].get("per_item_deltas")]
    if models_with_deltas:
        ncols = len(models_with_deltas)
        fig, axes = plt.subplots(1, ncols, figsize=(max(8, ncols * 7), 5), squeeze=False)
        for col, model in enumerate(models_with_deltas):
            ax = axes[0][col]
            items_d = sorted(
                runs[model]["per_item_deltas"], key=lambda d: d["delta"], reverse=True
            )
            labels_x = [d["task_id"].replace("AMBI/", "") for d in items_d]
            deltas = [d["delta"] * 100 for d in items_d]
            colors_bar = ["#EF5350" if d >= 0 else "#42A5F5" for d in deltas]
            ax.bar(range(len(deltas)), deltas, color=colors_bar)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xticks(range(len(labels_x)))
            ax.set_xticklabels(labels_x, rotation=90, fontsize=6)
            ax.set_xlabel("Item")
            ax.set_ylabel("Δ pass@k (pp)")
            ax.set_title(
                f"{model}\nper-item Δ pass@k = clean − pass@k(either)\n"
                f"(red=hurt, blue=helped)"
            )
        plt.tight_layout()
        _save(plt, output_dir / "plot_delta_per_item.png")

    # ── Plot B: SA/EA/AC stacked bar — one bar per model ─────────────────────
    models_with_labels = [m for m in models if runs[m].get("label_dist")]
    if models_with_labels:
        fig, ax = plt.subplots(figsize=(max(5, len(models_with_labels) * 1.4), 4))
        xm = np.arange(len(models_with_labels))
        bottoms = np.zeros(len(models_with_labels))
        for lbl in LABELS:
            vals = [runs[m]["label_dist"].get(lbl, 0) for m in models_with_labels]
            ax.bar(xm, vals, bottom=bottoms, label=lbl, color=LBL_COLORS[lbl])
            bottoms += np.array(vals)
        ax.set_xticks(xm)
        ax.set_xticklabels(models_with_labels, rotation=15, ha="right")
        ax.set_ylabel("Fraction of Samples")
        ax.set_title("SA / EA / AC by Model")
        ax.legend(loc="upper right")
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        _save(plt, output_dir / "plot_sa_ea_ac.png")

    # ── Plot G: SA/EA/AC aggregated across all models ─────────────────────────
    if models_with_labels:
        agg_dist: dict[str, list[float]] = defaultdict(list)
        for m in models_with_labels:
            for lbl in LABELS:
                agg_dist[lbl].append(runs[m]["label_dist"].get(lbl, 0))
        fig, ax = plt.subplots(figsize=(4, 4))
        bottom = 0.0
        for lbl in LABELS:
            val = sum(agg_dist[lbl]) / len(agg_dist[lbl])
            ax.bar(["All Models"], [val], bottom=bottom, label=lbl, color=LBL_COLORS[lbl])
            bottom += val
        ax.set_ylabel("Fraction of Samples")
        ax.set_title(f"SA / EA / AC — Aggregated\n({len(models_with_labels)} model(s))")
        ax.legend(loc="upper right")
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        _save(plt, output_dir / "plot_sa_ea_ac_aggregated.png")

    # ── Plot C: SA/EA/AC by risk — one subplot per model ─────────────────────
    models_with_risk = [m for m in models if runs[m].get("label_by_risk")]
    if models_with_risk:
        ncols = len(models_with_risk)
        fig, axes = plt.subplots(1, ncols, figsize=(max(6, ncols * 4), 4), squeeze=False)
        for col, model in enumerate(models_with_risk):
            ax = axes[0][col]
            risk_data = runs[model]["label_by_risk"]
            risk_levels = [r for r in ("low", "high") if r in risk_data]
            xr = np.arange(len(risk_levels))
            bottoms = np.zeros(len(risk_levels))
            for lbl in LABELS:
                vals = [risk_data[r].get(lbl, 0) for r in risk_levels]
                ax.bar(xr, vals, bottom=bottoms, label=lbl, color=LBL_COLORS[lbl], width=0.5)
                bottoms += np.array(vals)
            ax.set_xticks(xr)
            ax.set_xticklabels([f"{r.capitalize()} Risk" for r in risk_levels])
            ax.set_ylabel("Fraction of Samples")
            ax.set_title(f"{model}\nSA/EA/AC by Risk Level")
            ax.legend(loc="upper right", fontsize=8)
            ax.set_ylim(0, 1.05)
        plt.tight_layout()
        _save(plt, output_dir / "plot_sa_ea_ac_by_risk.png")

    # ── Plot H: SA/EA/AC by risk — aggregated across all models ──────────────
    if models_with_risk:
        agg_risk: dict[str, dict[str, list[float]]] = {
            r: defaultdict(list) for r in ("low", "high")
        }
        for m in models_with_risk:
            for risk, dist in runs[m]["label_by_risk"].items():
                for lbl in LABELS:
                    agg_risk[risk][lbl].append(dist.get(lbl, 0))

        risk_levels = [r for r in ("low", "high") if agg_risk[r]]
        fig, ax = plt.subplots(figsize=(5, 4))
        xr = np.arange(len(risk_levels))
        bottoms = np.zeros(len(risk_levels))
        for lbl in LABELS:
            vals = [
                sum(agg_risk[r][lbl]) / len(agg_risk[r][lbl]) if agg_risk[r][lbl] else 0
                for r in risk_levels
            ]
            ax.bar(xr, vals, bottom=bottoms, label=lbl, color=LBL_COLORS[lbl], width=0.5)
            bottoms += np.array(vals)
        ax.set_xticks(xr)
        ax.set_xticklabels([f"{r.capitalize()} Risk" for r in risk_levels])
        ax.set_ylabel("Fraction of Samples")
        ax.set_title(
            f"SA/EA/AC by Risk Level — Aggregated\n({len(models_with_risk)} model(s))"
        )
        ax.legend(loc="upper right", fontsize=8)
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        _save(plt, output_dir / "plot_sa_ea_ac_by_risk_aggregated.png")


def _save(plt, path: Path) -> None:
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Phase 2+3 results.")
    parser.add_argument(
        "--baseline", nargs="*", default=None,
        help="Baseline eval JSONL path(s) (data/results/baseline_*.jsonl)"
    )
    parser.add_argument(
        "--perturbed", nargs="*", default=None,
        help="Perturbed eval JSONL path(s) (data/results/perturbed_*.jsonl)"
    )
    parser.add_argument(
        "--classified", nargs="*", default=None,
        help="Classification JSONL path(s) (data/results/classified_*.jsonl)"
    )
    parser.add_argument(
        "--label", nargs="*", default=None,
        help="Model label(s) corresponding to each file set (optional)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Directory to save plots (default: data/results/analysis/)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else RESULTS_DIR / "analysis"

    # Auto-discover if no explicit paths given
    if args.perturbed is None and args.baseline is None and args.classified is None:
        discovered = auto_discover(RESULTS_DIR)
        if not discovered:
            print("No result files found in data/results/. Run an eval script first.")
            sys.exit(0)
        print(f"Auto-discovered {len(discovered)} model run(s): {list(discovered.keys())}")
        file_groups = {
            model: {
                "baseline": data.get("baseline"),
                "perturbed": data.get("perturbed"),
                "classified": data.get("classified"),
            }
            for model, data in discovered.items()
        }
    else:
        labels = args.label or []
        perturbed_files = args.perturbed or []
        baseline_files = args.baseline or []
        classified_files = args.classified or []
        n = max(len(perturbed_files), len(baseline_files), len(classified_files))
        file_groups = {}
        for i in range(n):
            lbl = labels[i] if i < len(labels) else f"run_{i+1}"
            file_groups[lbl] = {
                "baseline": Path(baseline_files[i]) if i < len(baseline_files) else None,
                "perturbed": Path(perturbed_files[i]) if i < len(perturbed_files) else None,
                "classified": Path(classified_files[i]) if i < len(classified_files) else None,
            }

    # Load files and compute metrics
    runs: dict[str, dict] = {}
    for model, files in file_groups.items():
        data: dict = {}

        if files.get("baseline") and files["baseline"].exists():
            recs = load_jsonl(files["baseline"])
            br = pass_rates(recs, key="pass_count")
            data["baseline_rate"] = br["rate"]
            for k in K_VALUES:
                data[f"baseline_pass_at_{k}"] = br[f"pass_at_{k}"]
            data["baseline_records"] = recs

        if files.get("perturbed") and files["perturbed"].exists():
            recs = load_jsonl(files["perturbed"])
            rates = perturbed_rates(recs)
            # Aggregate (== pass@1) rates
            data["perturbed_a_rate"] = rates["pass_a_rate"]
            data["perturbed_b_rate"] = rates["pass_b_rate"]
            data["perturbed_either_rate"] = rates["pass_either_rate"]
            data["chose_a_rate"] = rates["chose_a_rate"]
            data["chose_b_rate"] = rates["chose_b_rate"]
            data["both_pass_rate"] = rates["both_pass_rate"]
            data["neither_rate"] = rates["neither_rate"]
            data["interp_a_bias"] = rates["interp_a_bias"]
            # Unbiased pass@k for each k in K_VALUES
            for k in K_VALUES:
                for field in ("pass_a", "pass_b", "pass_either",
                              "chose_a", "chose_b"):
                    data[f"{field}_at_{k}"] = rates.get(f"{field}_at_{k}")
            data["perturbed_records"] = recs

        if files.get("classified") and files["classified"].exists():
            recs = load_jsonl(files["classified"])
            data["label_dist"] = label_distribution(recs)
            data["label_by_correctness"] = label_by_correctness(recs)
            data["label_by_risk"] = label_by_risk_level(recs)
            data["classified_records"] = recs

        if "baseline_records" in data and "perturbed_records" in data:
            baseline_map = {r["task_id"]: r for r in data["baseline_records"]}
            per_item = []
            for r in data["perturbed_records"]:
                b = baseline_map.get(r["task_id"])
                if b:
                    either_rate = _item_pass_either_rate(r)
                    per_item.append({
                        "task_id": r["task_id"],
                        # Primary delta: baseline - pass_either (model produced
                        # valid code for at least one interpretation).
                        "delta": b.get("pass_rate", 0) - either_rate,
                        "delta_a": b.get("pass_rate", 0) - r.get("pass_a_rate", 0),
                        "delta_b": b.get("pass_rate", 0) - r.get("pass_b_rate", 0),
                        "pass_a_rate": r.get("pass_a_rate", 0),
                        "pass_b_rate": r.get("pass_b_rate", 0),
                        "pass_either_rate": either_rate,
                        "ambiguity_type": r.get("ambiguity_type", ""),
                        "risk_level": r.get("risk_level", ""),
                    })
            data["per_item_deltas"] = per_item

        if data:
            runs[model] = data

    if not runs:
        print("No data loaded. Check file paths.")
        sys.exit(1)

    # Print tables
    models_with_perturbed = [m for m in runs if "perturbed_a_rate" in runs[m]]
    models_with_labels = [m for m in runs if "label_dist" in runs[m]]
    if models_with_perturbed:
        print_ambiguity_tax_table(runs)
        print_pass_at_k_table(runs)
        print_interpretation_choice_table(runs)
        print_perturbed_by_type(runs)
    if models_with_labels:
        print_sa_ea_ac_table(runs)
        print_label_correctness_table(runs)

    # Persist metrics (merges with prior runs)
    save_metrics(runs, RESULTS_DIR)

    # Save plots
    save_plots(runs, output_dir)


if __name__ == "__main__":
    main()
