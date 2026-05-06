"""Aggregate all 6-model results into the data structures the milestone notebook uses.

Inputs
------
- data/results/baseline_<safe_model>_<ts>.jsonl
- data/results/perturbed_<safe_model>_<ts>.jsonl
- data/results/classified_<safe_model>_<ts>.jsonl

For each model we auto-detect the *most recent* triple (baseline, perturbed,
classified) — but only triples whose perturbed file covers the full
benchmark_v2_full.jsonl (48 items). That filter avoids pulling in older v1
results when v2_full results exist.

Outputs
-------
- data/results/milestone/summary.json         (top-line metrics × model)
- data/results/milestone/per_item.csv         (long-format: task × model)
- data/results/milestone/per_sample.csv       (sample-level: task × model × sample)
- data/results/milestone/by_type.csv          (model × ambiguity_type tax)
- data/results/milestone/by_risk.csv          (model × risk_level tax)
- data/results/milestone/bootstrap_ci.json    (bootstrap 95% CI for tax/pass_either)

Notes on metric definitions (consistent with paper):
- pass@k uses the unbiased Chen et al. 2021 estimator on the n_samples per item.
- Tax = baseline_pass@k − pass_either@k (a sample is "successful" if it satisfies
  EITHER interpretation A or B — both are valid given the ambiguous prompt).
- A-bias = chose_a / (chose_a + chose_b) among decisive samples (excludes "both"
  and "neither" rows; computed at the sample level then averaged across items).
- Behavior rates are computed at the sample level.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from math import comb
from pathlib import Path
from typing import Iterable, Optional

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "results"
OUT_DIR = RESULTS / "milestone"
BENCHMARK_PATH = ROOT / "data" / "benchmark" / "benchmark_v2_full.jsonl"

K_VALUES = (1, 3)
LABELS = ("SA", "EA", "AC", "unclassifiable", "error")
AMBIG_TYPES = (
    "coreferential", "syntactic", "scopal", "collective_distributive", "elliptical",
)
SOURCES = ("mbpp", "ds1000", "humaneval")
RISK_LEVELS = ("low", "high")
BOOTSTRAP_SAMPLES = 2000

# ── 5 models we evaluate (match models.yaml aliases) ──
# gpt-5.5 replaced gpt-5.4 (routing issue forced gpt-5.4 → gpt-5.4-mini).
# qwen-3.6-plus skipped due to throughput.
MILESTONE_MODELS = (
    "gpt-5.5",
    "claude-sonnet",
    "claude-opus",
    "gemini-3.1-pro",
    "deepseek-v4-pro",
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe(model: str) -> str:
    """run_full_pipeline.py's own model→filename transform: replace '/' with '_'."""
    return model.replace("/", "_")


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("task_id", "").startswith("SUMMARY"):
            continue
        rows.append(r)
    return rows


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator of pass@k (Chen et al. 2021)."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def mean_pass_at_k(records: Iterable[dict], count_field: str, k: int) -> float:
    vals = [
        pass_at_k(r["n_samples"], r.get(count_field, 0), k)
        for r in records if r.get("n_samples", 0) >= k
    ]
    return sum(vals) / len(vals) if vals else 0.0


# ── File discovery ────────────────────────────────────────────────────────────

@dataclass
class ResultTriple:
    model: str
    baseline_path: Path
    perturbed_path: Path
    classified_path: Path
    perturbed_n_items: int


_TS_RE = re.compile(r"_(\d{8}_\d{6})\.jsonl$")


def _ts(path: Path) -> str:
    m = _TS_RE.search(path.name)
    return m.group(1) if m else ""


def discover_results(
    model: str,
    benchmark_size: int,
    require_full: bool = True,
) -> Optional[ResultTriple]:
    """Find latest (baseline, perturbed, classified) for a model.

    If require_full=True, prefer the most recent triple whose perturbed file
    covers the FULL benchmark (benchmark_size items). If no full-coverage
    triple exists we fall back to the latest available — this lets the script
    work on partial results during a milestone run.
    """
    safe = _safe(model)
    pert_files = sorted(RESULTS.glob(f"perturbed_{safe}_*.jsonl"), key=_ts, reverse=True)

    candidates = []
    for pf in pert_files:
        ts = _ts(pf)
        bf = RESULTS / f"baseline_{safe}_{ts}.jsonl"
        cf = RESULTS / f"classified_{safe}_{ts}.jsonl"
        # baseline timestamp can be slightly earlier (it runs first); fall back
        # to the latest baseline for this model whose ts ≤ pf.ts
        if not bf.exists():
            base_files = sorted(
                (b for b in RESULTS.glob(f"baseline_{safe}_*.jsonl") if _ts(b) <= ts),
                key=_ts,
                reverse=True,
            )
            if not base_files:
                continue
            bf = base_files[0]
        if not cf.exists():
            continue

        n = sum(
            1 for line in pf.read_text().splitlines()
            if line.strip() and not json.loads(line).get("task_id", "").startswith("SUMMARY")
        )
        candidates.append(ResultTriple(model, bf, pf, cf, n))

    if not candidates:
        return None

    if require_full:
        full = [c for c in candidates if c.perturbed_n_items == benchmark_size]
        if full:
            return full[0]

    return candidates[0]


# ── Per-model metric computation ──────────────────────────────────────────────

@dataclass
class PerItem:
    """Joined per-item record (perturbed + baseline + behavior counts)."""
    task_id: str
    anchor_task_id: str
    source: str
    ambiguity_type: str
    risk_level: str
    n_samples: int

    # baseline
    baseline_pass_count: int = 0
    # perturbed
    pass_a_count: int = 0
    pass_b_count: int = 0
    pass_either_count: int = 0
    chose_a_count: int = 0
    chose_b_count: int = 0
    pass_both_count: int = 0
    pass_neither_count: int = 0

    # behavior (counts across n_samples)
    behavior_counts: dict = field(default_factory=dict)


def join_per_item(triple: ResultTriple) -> list[PerItem]:
    base = {r["task_id"]: r for r in _load_jsonl(triple.baseline_path)}
    pert = {r["task_id"]: r for r in _load_jsonl(triple.perturbed_path)}
    cls_rows = _load_jsonl(triple.classified_path)
    cls = {r["task_id"]: r for r in cls_rows}

    items = []
    for tid, p in pert.items():
        b = base.get(tid, {})
        c = cls.get(tid, {})

        bcounts = {lbl: 0 for lbl in LABELS}
        for s in c.get("samples", []):
            lbl = s.get("behavior_label", "unclassifiable")
            if lbl not in bcounts:
                lbl = "unclassifiable"
            bcounts[lbl] += 1

        items.append(PerItem(
            task_id=tid,
            anchor_task_id=p.get("anchor_task_id", ""),
            source=p.get("source", "unknown"),
            ambiguity_type=p.get("ambiguity_type", "unknown"),
            risk_level=p.get("risk_level", "unknown"),
            n_samples=p["n_samples"],
            baseline_pass_count=b.get("pass_count", 0),
            pass_a_count=p.get("pass_a_count", 0),
            pass_b_count=p.get("pass_b_count", 0),
            pass_either_count=p.get("pass_either_count", 0),
            chose_a_count=p.get("chose_a_count", 0),
            chose_b_count=p.get("chose_b_count", 0),
            pass_both_count=p.get("pass_both_count", 0),
            pass_neither_count=p.get("pass_neither_count", 0),
            behavior_counts=bcounts,
        ))
    return items


def aggregate_metrics(items: list[PerItem]) -> dict:
    """Compute model-level aggregate metrics."""
    out: dict = {"n_items": len(items)}

    # Pass rates @ k=1, 3
    for k in K_VALUES:
        out[f"baseline_pass_at_{k}"] = mean_pass_at_k(
            [{"n_samples": r.n_samples, "c": r.baseline_pass_count} for r in items], "c", k)
        out[f"pass_a_at_{k}"] = mean_pass_at_k(
            [{"n_samples": r.n_samples, "c": r.pass_a_count} for r in items], "c", k)
        out[f"pass_b_at_{k}"] = mean_pass_at_k(
            [{"n_samples": r.n_samples, "c": r.pass_b_count} for r in items], "c", k)
        out[f"pass_either_at_{k}"] = mean_pass_at_k(
            [{"n_samples": r.n_samples, "c": r.pass_either_count} for r in items], "c", k)
        out[f"chose_a_at_{k}"] = mean_pass_at_k(
            [{"n_samples": r.n_samples, "c": r.chose_a_count} for r in items], "c", k)
        out[f"chose_b_at_{k}"] = mean_pass_at_k(
            [{"n_samples": r.n_samples, "c": r.chose_b_count} for r in items], "c", k)
        out[f"ambiguity_tax_at_{k}_pp"] = (
            out[f"baseline_pass_at_{k}"] - out[f"pass_either_at_{k}"]
        ) * 100

    # Choice decomposition (mutually exclusive — sums to 1)
    total_samples = sum(r.n_samples for r in items)
    if total_samples > 0:
        sum_chose_a = sum(r.chose_a_count for r in items)
        sum_chose_b = sum(r.chose_b_count for r in items)
        sum_both = sum(r.pass_both_count for r in items)
        sum_neither = sum(r.pass_neither_count for r in items)
        out["chose_a_rate"] = sum_chose_a / total_samples
        out["chose_b_rate"] = sum_chose_b / total_samples
        out["both_pass_rate"] = sum_both / total_samples
        out["neither_rate"] = sum_neither / total_samples
        decisive = sum_chose_a + sum_chose_b
        out["a_bias"] = sum_chose_a / decisive if decisive else None

    # Behavior distribution (sample-level)
    total_b = defaultdict(int)
    n_total = 0
    for r in items:
        for lbl, cnt in r.behavior_counts.items():
            total_b[lbl] += cnt
        n_total += sum(r.behavior_counts.values())
    out["label_dist"] = (
        {lbl: total_b[lbl] / n_total for lbl in LABELS} if n_total else {lbl: 0.0 for lbl in LABELS}
    )

    return out


def aggregate_by_group(
    items: list[PerItem],
    group_key: str,
    group_values: tuple[str, ...],
) -> dict[str, dict]:
    """Per-group aggregate (e.g. by ambiguity_type, risk_level, source)."""
    out: dict[str, dict] = {}
    for grp in group_values:
        sub = [r for r in items if getattr(r, group_key) == grp]
        if sub:
            out[grp] = aggregate_metrics(sub)
        else:
            out[grp] = {"n_items": 0}
    return out


# ── Bootstrap confidence intervals (item-level resample) ──────────────────────

def bootstrap_tax_ci(
    items: list[PerItem],
    k: int,
    n_boot: int = BOOTSTRAP_SAMPLES,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap 95% CI for ambiguity tax @ k by resampling items with replacement.

    Returns (point, low, high) in percentage points.
    """
    import random
    rng = random.Random(seed)
    n = len(items)
    if n == 0:
        return (0.0, 0.0, 0.0)

    # Pre-compute per-item pass@k for baseline and pass_either
    base_at_k = [pass_at_k(r.n_samples, r.baseline_pass_count, k) for r in items]
    eit_at_k = [pass_at_k(r.n_samples, r.pass_either_count, k) for r in items]

    point_tax_pp = (sum(base_at_k) - sum(eit_at_k)) / n * 100

    boots = []
    indices = list(range(n))
    for _ in range(n_boot):
        sample_idx = [rng.choice(indices) for _ in range(n)]
        b_mean = sum(base_at_k[i] for i in sample_idx) / n
        e_mean = sum(eit_at_k[i] for i in sample_idx) / n
        boots.append((b_mean - e_mean) * 100)
    boots.sort()
    low = boots[int(n_boot * alpha / 2)]
    high = boots[int(n_boot * (1 - alpha / 2))]
    return (point_tax_pp, low, high)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-full", action="store_true", default=True,
                        help="Only use triples that cover the full benchmark.")
    parser.add_argument("--no-require-full", dest="require_full", action="store_false")
    parser.add_argument("--benchmark", type=Path, default=BENCHMARK_PATH)
    parser.add_argument("--n-boot", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    benchmark = _load_jsonl(args.benchmark)
    benchmark_size = len(benchmark)
    print(f"Benchmark: {args.benchmark} ({benchmark_size} items)")
    print(f"Looking for results for {len(MILESTONE_MODELS)} models...")
    print()

    triples: dict[str, ResultTriple] = {}
    for m in MILESTONE_MODELS:
        t = discover_results(m, benchmark_size, require_full=args.require_full)
        if t is None:
            print(f"  ✗ {m}: no result triple found")
            continue
        coverage = "FULL" if t.perturbed_n_items == benchmark_size else f"PARTIAL ({t.perturbed_n_items})"
        print(f"  ✓ {m}: {coverage}")
        print(f"      baseline:   {t.baseline_path.name}")
        print(f"      perturbed:  {t.perturbed_path.name}")
        print(f"      classified: {t.classified_path.name}")
        triples[m] = t

    if not triples:
        print("\nNo results found. Aborting.")
        return

    print()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Per-model aggregation ────────────────────────────────────────────────
    summary: dict = {"models": list(triples.keys()), "benchmark_size": benchmark_size,
                     "per_model": {}, "bootstrap_ci": {}}
    per_item_rows: list[dict] = []
    by_type_rows: list[dict] = []
    by_risk_rows: list[dict] = []
    by_source_rows: list[dict] = []

    for m, t in triples.items():
        print(f"Aggregating {m}...")
        items = join_per_item(t)
        agg = aggregate_metrics(items)

        # Per-group
        agg["by_type"] = aggregate_by_group(items, "ambiguity_type", AMBIG_TYPES)
        agg["by_risk"] = aggregate_by_group(items, "risk_level", RISK_LEVELS)
        agg["by_source"] = aggregate_by_group(items, "source", SOURCES)
        summary["per_model"][m] = agg

        # Bootstrap CIs
        ci = {}
        for k in K_VALUES:
            point, lo, hi = bootstrap_tax_ci(items, k, n_boot=args.n_boot)
            ci[f"tax_at_{k}_pp"] = {"point": point, "low": lo, "high": hi}
        summary["bootstrap_ci"][m] = ci

        # Long-format per_item rows for CSV
        for r in items:
            row = {
                "model": m,
                "task_id": r.task_id,
                "anchor_task_id": r.anchor_task_id,
                "source": r.source,
                "ambiguity_type": r.ambiguity_type,
                "risk_level": r.risk_level,
                "n_samples": r.n_samples,
                "baseline_pass_count": r.baseline_pass_count,
                "pass_a_count": r.pass_a_count,
                "pass_b_count": r.pass_b_count,
                "pass_either_count": r.pass_either_count,
                "chose_a_count": r.chose_a_count,
                "chose_b_count": r.chose_b_count,
                "pass_both_count": r.pass_both_count,
                "pass_neither_count": r.pass_neither_count,
                "baseline_rate": r.baseline_pass_count / r.n_samples,
                "pass_either_rate": r.pass_either_count / r.n_samples,
                "tax_pp": (r.baseline_pass_count - r.pass_either_count) / r.n_samples * 100,
            }
            for lbl in LABELS:
                row[f"behavior_{lbl}"] = r.behavior_counts.get(lbl, 0)
            per_item_rows.append(row)

        for atype, sub in agg["by_type"].items():
            if sub.get("n_items", 0) > 0:
                by_type_rows.append({
                    "model": m, "ambiguity_type": atype,
                    "n_items": sub["n_items"],
                    "baseline_pass_at_1": sub.get("baseline_pass_at_1", 0),
                    "pass_either_at_1": sub.get("pass_either_at_1", 0),
                    "tax_at_1_pp": sub.get("ambiguity_tax_at_1_pp", 0),
                    "tax_at_3_pp": sub.get("ambiguity_tax_at_3_pp", 0),
                    "pass_a_at_1": sub.get("pass_a_at_1", 0),
                    "pass_b_at_1": sub.get("pass_b_at_1", 0),
                })
        for risk, sub in agg["by_risk"].items():
            if sub.get("n_items", 0) > 0:
                by_risk_rows.append({
                    "model": m, "risk_level": risk,
                    "n_items": sub["n_items"],
                    "tax_at_1_pp": sub.get("ambiguity_tax_at_1_pp", 0),
                    "tax_at_3_pp": sub.get("ambiguity_tax_at_3_pp", 0),
                })
        for src, sub in agg["by_source"].items():
            if sub.get("n_items", 0) > 0:
                by_source_rows.append({
                    "model": m, "source": src,
                    "n_items": sub["n_items"],
                    "tax_at_1_pp": sub.get("ambiguity_tax_at_1_pp", 0),
                    "tax_at_3_pp": sub.get("ambiguity_tax_at_3_pp", 0),
                })

    # ── Write outputs ────────────────────────────────────────────────────────
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float)
    )

    def _write_csv(rows: list[dict], name: str) -> None:
        if not rows:
            return
        path = args.out_dir / name
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    _write_csv(per_item_rows, "per_item.csv")
    _write_csv(by_type_rows, "by_type.csv")
    _write_csv(by_risk_rows, "by_risk.csv")
    _write_csv(by_source_rows, "by_source.csv")

    print()
    print(f"Wrote outputs to {args.out_dir}/:")
    for name in ["summary.json", "per_item.csv", "by_type.csv", "by_risk.csv", "by_source.csv"]:
        p = args.out_dir / name
        if p.exists():
            print(f"  {name:20s}  {p.stat().st_size:>8d} bytes")

    # ── Print top-line table ─────────────────────────────────────────────────
    print()
    print("=" * 96)
    print(f"{'Model':<20} {'tax@1 (95% CI)':>20} {'tax@3 (95% CI)':>20} {'A-bias':>8} {'SA':>6} {'EA':>6} {'AC':>6}")
    print("-" * 96)
    for m in summary["models"]:
        agg = summary["per_model"][m]
        ci = summary["bootstrap_ci"][m]
        c1 = ci["tax_at_1_pp"]; c3 = ci["tax_at_3_pp"]
        ld = agg["label_dist"]
        ab = agg.get("a_bias")
        ab_str = f"{ab:.1%}" if ab is not None else "n/a"
        print(f"{m:<20} "
              f"{c1['point']:>+6.1f} [{c1['low']:>+5.1f},{c1['high']:>+5.1f}]   "
              f"{c3['point']:>+6.1f} [{c3['low']:>+5.1f},{c3['high']:>+5.1f}]   "
              f"{ab_str:>8} "
              f"{ld.get('SA',0):>6.1%} {ld.get('EA',0):>6.1%} {ld.get('AC',0):>6.1%}")
    print("=" * 96)


if __name__ == "__main__":
    main()