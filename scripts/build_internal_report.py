"""Generate notebooks/internal_result_report.ipynb — the team-facing deep-dive.

Audience: us / collaborators / TAs reviewing internals. Not the poster, not the
paper. Free to be exploratory, raw, and noisy. Goal is to surface things worth
investigating further rather than to present polished claims.

Sections (kept in sync with the section numbers below):
  1. Data inventory + sanity checks
  2. Per-model aggregate distributions
  3. Cross-model item-level agreement
  4. Choice decomposition (chose_a / chose_b / both / neither)
  5. Failure mode dissection (parse / runtime / wrong-read / schema)
  6. AC sample gallery (every clarification question across all 5 models)
  7. EA sample gallery (a curated subset of explicit-assumption responses)
  8. Hardest / easiest / most-variable items
  9. Sonnet negative-tax anomaly — per-item drilldown with code samples
  10. Latency profile
  11. Interpretation-B pickup analysis
  12. Open questions / next experiments

Re-generate by running:  python scripts/build_internal_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "internal_result_report.ipynb"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.rstrip("\n").split("\n")],
    }


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in src.rstrip("\n").split("\n")],
    }


CELLS: list[dict] = []


# ── 0. Title + setup ─────────────────────────────────────────────────────────

CELLS.append(md(r"""
# AmbiCode-Eval — Internal Result Report

*Audience: project team. Free to be exploratory.*

This notebook is the team-facing deep-dive into the v2-full benchmark
evaluation (5 models × 48 items × n=5 × T=0.8). It lives next to the
milestone analysis notebook, but is intentionally rougher: more raw tables,
more diagnostic plots, more "huh, that's weird" callouts.

**Companion notebook:** [`milestone_analysis.ipynb`](./milestone_analysis.ipynb) — the polished, poster-facing version.

**What's in here that's *not* in the milestone notebook:**

- Sanity checks (do the choice-decomposition counts add up? are there missing samples?)
- Per-model raw distributions
- A complete gallery of every AC sample (only ~10–15 across all models — small enough to enumerate)
- A curated EA gallery (what assumptions do models *actually* declare?)
- Failure-mode taxonomy applied to the "neither" cases (parse / runtime / wrong-read / DS-1000 schema)
- Hardest / easiest / most-variable items
- Sonnet anomaly drill-down with side-by-side code from 5 models
- Latency profile (per-model wall-clock cost)
- B-pickup analysis (when does anyone actually pick interpretation B?)

> ⚠️ Numbers in this notebook should match the milestone notebook to within rounding. If they don't, stop and figure out why.
"""))

CELLS.append(code(r"""
# ── Setup ──────────────────────────────────────────────────────────────────
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from scripts.plot_style import (
    setup_style, MODEL_COLORS, MODEL_ORDER, BEHAVIOR_COLORS, DIVERGING_CMAP,
    model_palette, shorten_model_name,
)
setup_style()

DATA = ROOT / "data" / "results" / "milestone"
RESULTS = ROOT / "data" / "results"

summary    = json.loads((DATA / "summary.json").read_text())
per_item   = pd.read_csv(DATA / "per_item.csv")
by_type    = pd.read_csv(DATA / "by_type.csv")
by_risk    = pd.read_csv(DATA / "by_risk.csv")
by_source  = pd.read_csv(DATA / "by_source.csv")

MODELS = [m for m in MODEL_ORDER if m in summary["models"]]
print(f"Loaded {len(MODELS)} models: {MODELS}")
"""))

CELLS.append(code(r"""
# Locate the latest classified_<model>_<ts>.jsonl per milestone model and load
# sample-level records. We need them for behavior gallery + failure dissection.

_TS_RE = re.compile(r"_(\d{8}_\d{6})\.jsonl$")

def _ts(p: Path) -> str:
    m = _TS_RE.search(p.name)
    return m.group(1) if m else ""

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


classified_files: dict[str, Path] = {}
for m in MODELS:
    safe = m.replace("/", "_")
    candidates = sorted(RESULTS.glob(f"classified_{safe}_*.jsonl"), key=_ts, reverse=True)
    for c in candidates:
        n = sum(1 for ln in c.read_text().splitlines()
                if ln.strip() and not json.loads(ln).get("task_id", "").startswith("SUMMARY"))
        if n == summary["benchmark_size"]:
            classified_files[m] = c
            break

# Map: model -> list of items (each item has `samples` list)
classified: dict[str, list[dict]] = {m: _load_jsonl(p) for m, p in classified_files.items()}

# Long DataFrame at the SAMPLE level (for gallery + failure dissection)
def to_sample_df():
    rows = []
    for m, items in classified.items():
        for it in items:
            for s in it["samples"]:
                rows.append({
                    "model": m,
                    "task_id": it["task_id"],
                    "anchor_task_id": it.get("anchor_task_id", ""),
                    "source": it["source"],
                    "ambiguity_type": it["ambiguity_type"],
                    "risk_level": it["risk_level"],
                    "interpretation_a": it.get("interpretation_a", ""),
                    "interpretation_b": it.get("interpretation_b", ""),
                    "perturbed_prompt": it.get("perturbed_prompt", ""),
                    "sample": s.get("sample"),
                    "passed_a": s.get("passed_a", False),
                    "passed_b": s.get("passed_b", False),
                    "behavior_label": s.get("behavior_label", ""),
                    "behavior_q1_question": s.get("behavior_q1_question", False),
                    "behavior_q2_code": s.get("behavior_q2_code", False),
                    "behavior_q3_assumption": s.get("behavior_q3_assumption", False),
                    "behavior_rationale": s.get("behavior_rationale", ""),
                    "raw_response": s.get("raw_response", ""),
                    "generated_code": s.get("generated_code", ""),
                    "prose": s.get("prose", ""),
                    "stderr_a": s.get("stderr_a", ""),
                    "stderr_b": s.get("stderr_b", ""),
                    "exit_code_a": s.get("exit_code_a", -1),
                    "exit_code_b": s.get("exit_code_b", -1),
                    "timed_out_a": s.get("timed_out_a", False),
                    "timed_out_b": s.get("timed_out_b", False),
                    "latency_s": s.get("latency_s", 0.0),
                })
    return pd.DataFrame(rows)

samples = to_sample_df()
print(f"Sample-level frame: {len(samples)} rows  ({samples.groupby('model').size().to_dict()})")
"""))


# ── 1. Data inventory + sanity checks ────────────────────────────────────────

CELLS.append(md(r"""
## 1. Data inventory + sanity checks

Goal: be sure we have what we think we have, and that the choice
decomposition counts (chose_a + chose_b + both + neither) add up to
n_samples for every (model, item).
"""))

CELLS.append(code(r"""
# 1.1 Coverage table
inv = (per_item.groupby("model")
                 .agg(items=("task_id", "nunique"),
                      total_samples=("n_samples", "sum"),
                      mbpp=("source", lambda s: (s == "mbpp").sum()),
                      ds1000=("source", lambda s: (s == "ds1000").sum()),
                      humaneval=("source", lambda s: (s == "humaneval").sum()))
                 .reindex(MODELS))
print("Per-model coverage:")
inv
"""))

CELLS.append(code(r"""
# 1.2 Choice decomposition sanity:
# chose_a + chose_b + pass_both + pass_neither should equal n_samples per row.
sanity = per_item.copy()
sanity["sum_decomp"] = (sanity["chose_a_count"] + sanity["chose_b_count"]
                       + sanity["pass_both_count"] + sanity["pass_neither_count"])
mismatches = sanity[sanity["sum_decomp"] != sanity["n_samples"]]
if mismatches.empty:
    print("✓  Choice decomposition adds up to n_samples for ALL (model, item) rows.")
else:
    print(f"✗  {len(mismatches)} rows where decomposition ≠ n_samples:")
    print(mismatches[["model", "task_id", "sum_decomp", "n_samples"]].head(10).to_string(index=False))
"""))

CELLS.append(code(r"""
# 1.3 Behavior labels sanity:
# behavior_<lbl> across {SA, EA, AC, unclassifiable, error} should sum to n_samples.
b_cols = [c for c in per_item.columns if c.startswith("behavior_")]
sanity = per_item.copy()
sanity["sum_b"] = sanity[b_cols].sum(axis=1)
mismatches = sanity[sanity["sum_b"] != sanity["n_samples"]]
if mismatches.empty:
    print(f"✓  Behavior labels sum to n_samples across {b_cols}")
else:
    print(f"✗  {len(mismatches)} rows where behavior labels don't sum:")
    print(mismatches[["model", "task_id", "sum_b", "n_samples"] + b_cols].head(5))

# Also a quick: any judge errors?
print()
err_count = per_item["behavior_error"].sum()
unclass_count = per_item["behavior_unclassifiable"].sum()
total_samp = per_item["n_samples"].sum()
print(f"behavior=error      : {err_count}/{total_samp}  ({err_count/total_samp*100:.1f}%)")
print(f"behavior=unclassifiable: {unclass_count}/{total_samp}  ({unclass_count/total_samp*100:.1f}%)")
"""))


# ── 2. Per-model aggregate distributions ─────────────────────────────────────

CELLS.append(md(r"""
## 2. Per-model aggregate distributions

What does the *distribution* of per-item baseline pass rate, pass_either, and tax look like for each model? Headline numbers are means; here we look at spread.
"""))

CELLS.append(code(r"""
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)

metrics = [
    ("baseline_rate",     "Baseline pass rate (clean prompt)",     (0, 1)),
    ("pass_either_rate",  "Perturbed pass_either rate",            (0, 1)),
    ("tax_pp",            "Per-item tax (pp)  perturbed→clean",    (-100, 100)),
]

for ax, (col, title, xlim) in zip(axes, metrics):
    for m in MODELS:
        sub = per_item[per_item["model"] == m][col]
        ax.hist(sub, bins=20, range=xlim, alpha=0.45,
                label=shorten_model_name(m), color=MODEL_COLORS[m])
    ax.set_title(title)
    ax.set_xlabel(col)

axes[0].set_ylabel("Number of items")
axes[-1].axvline(0, color="#222", linewidth=0.8)
axes[0].legend(loc="upper center", bbox_to_anchor=(1.6, -0.15), ncol=5, frameon=False)
plt.tight_layout()
plt.show()
"""))

CELLS.append(code(r"""
# Per-model summary: mean / median / p25 / p75 of tax_pp
desc = (per_item.groupby("model")["tax_pp"]
                  .agg(["mean", "median",
                       lambda x: x.quantile(0.25),
                       lambda x: x.quantile(0.75),
                       "std", "count"])
                  .rename(columns={"<lambda_0>": "p25", "<lambda_1>": "p75"})
                  .reindex(MODELS).round(2))
print("Per-item tax_pp distribution (per model):")
desc
"""))


# ── 3. Cross-model item-level agreement ──────────────────────────────────────

CELLS.append(md(r"""
## 3. Cross-model item-level agreement

For each of the 48 items, on how many of the 5 models did `pass_either` exceed 0.5? An item that "passes" all 5 is universally easy; one that fails all 5 is universally hard. The shape of this distribution tells us how much the benchmark's difficulty is item-driven vs model-driven.
"""))

CELLS.append(code(r"""
# Per-item: how many models scored pass_either_rate > 0.5?
pe = per_item.pivot(index="task_id", columns="model", values="pass_either_rate")[MODELS]
n_models_passed = (pe > 0.5).sum(axis=1)

fig, ax = plt.subplots(figsize=(8, 4))
counts = n_models_passed.value_counts().reindex(range(0, len(MODELS) + 1), fill_value=0).sort_index()
ax.bar(counts.index, counts.values, color="#444", edgecolor="black")
for x, y in zip(counts.index, counts.values):
    ax.text(x, y + 0.4, str(y), ha="center", fontsize=10)
ax.set_xlabel("Number of models with pass_either_rate > 0.5")
ax.set_ylabel("Items")
ax.set_xticks(range(0, len(MODELS) + 1))
ax.set_title("Item difficulty distribution: how many of 5 models can satisfy either reading?")
plt.tight_layout()
plt.show()

print(f"\nItems all 5 fail (universally hard): {(n_models_passed == 0).sum()}")
print(f"Items all 5 pass (universally easy): {(n_models_passed == 5).sum()}")
print(f"Items 1-4 pass (model-dependent): {((n_models_passed >= 1) & (n_models_passed <= 4)).sum()}")
"""))

CELLS.append(code(r"""
# List the universally-hard and universally-easy items
hard = pe[n_models_passed == 0].index.tolist()
easy = pe[n_models_passed == len(MODELS)].index.tolist()
meta = (per_item[per_item["model"] == MODELS[0]]
        [["task_id", "anchor_task_id", "source", "ambiguity_type", "risk_level"]]
        .set_index("task_id"))

print(f"Universally-HARD items ({len(hard)}):")
print(meta.loc[hard].to_string() if hard else "  (none)")
print()
print(f"Universally-EASY items ({len(easy)}):")
print(meta.loc[easy].to_string() if easy else "  (none)")
"""))


# ── 4. Choice decomposition ──────────────────────────────────────────────────

CELLS.append(md(r"""
## 4. Choice decomposition — where does the n_samples budget go?

For each model, what fraction of *samples* end up in each of the 4 mutually-exclusive buckets:
- **chose_a**: passed test_a but not test_b → model picked interpretation A
- **chose_b**: passed test_b but not test_a → model picked interpretation B
- **both**: passed both → tests cannot distinguish (unfortunate, but happens)
- **neither**: passed neither → model produced wrong / failing code

Headline shape: a model that handles ambiguity well has high (chose_a + chose_b) and low neither.
"""))

CELLS.append(code(r"""
# Stacked horizontal bar of choice decomposition per model
fig, ax = plt.subplots(figsize=(11, 4.5))

cats = ["chose_a", "chose_b", "both", "neither"]
labels_pretty = {
    "chose_a": "Chose A",
    "chose_b": "Chose B",
    "both":    "Both pass (tests can't distinguish)",
    "neither": "Neither (code error / wrong reading)",
}
colors = {
    "chose_a": "#4285F4",
    "chose_b": "#F39C12",
    "both":    "#BDC3C7",
    "neither": "#E74C3C",
}

y = np.arange(len(MODELS))
left = np.zeros(len(MODELS))
for cat in cats:
    rate_field = "neither_rate" if cat == "neither" else f"{cat}_rate" if cat != "both" else "both_pass_rate"
    vals = np.array([summary["per_model"][m][rate_field] for m in MODELS]) * 100
    ax.barh(y, vals, left=left, color=colors[cat], edgecolor="white",
            linewidth=0.7, label=labels_pretty[cat])
    for yi, (l, v) in enumerate(zip(left, vals)):
        if v >= 4:
            ax.text(l + v/2, yi, f"{v:.1f}", ha="center", va="center", fontsize=9,
                    color="white" if cat in {"chose_a", "neither"} else "#222",
                    fontweight="bold")
    left += vals

ax.set_yticks(y)
ax.set_yticklabels([shorten_model_name(m) for m in MODELS])
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xlabel("% of all samples")
ax.set_title("Choice decomposition (per model, mutually exclusive)")
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
ax.grid(axis="x", linestyle=":", alpha=0.5)
plt.tight_layout()
plt.show()

# Tabular form
decomp = pd.DataFrame({
    m: {
        "chose_a%":  summary["per_model"][m]["chose_a_rate"] * 100,
        "chose_b%":  summary["per_model"][m]["chose_b_rate"] * 100,
        "both%":     summary["per_model"][m]["both_pass_rate"] * 100,
        "neither%":  summary["per_model"][m]["neither_rate"] * 100,
    } for m in MODELS
}).T.round(1).reindex(MODELS)
print("\nDecomposition (% of samples):")
decomp
"""))


# ── 5. Failure mode dissection ───────────────────────────────────────────────

CELLS.append(md(r"""
## 5. Failure mode dissection — what's *in* the "neither" pile?

A sample that satisfied neither test_a nor test_b can fail for distinct reasons:

- **parse error** — generated_code is empty or unparseable
- **timeout** — sandbox timed out
- **runtime error** — code ran but raised an exception (NameError, TypeError, …)
- **assertion failed** — code ran cleanly but produced the wrong output (wrong reading / brittle test schema)

Knowing the breakdown matters: an "assertion failed" share is real ambiguity-induced wrong reading; a "runtime error" share is plumbing.
"""))

CELLS.append(code(r"""
def classify_failure(s: pd.Series) -> str:
    # Return the failure category for a sample that didn't pass either test.
    if s["passed_a"] or s["passed_b"]:
        return "passed"
    code = s.get("generated_code", "") or ""
    if not code.strip() or len(code.strip()) < 5:
        return "parse_or_empty"
    if s.get("timed_out_a") or s.get("timed_out_b"):
        return "timeout"
    stderr_a = (s.get("stderr_a") or "").lower()
    stderr_b = (s.get("stderr_b") or "").lower()
    runtime_markers = ("nameerror", "typeerror", "valueerror", "attributeerror",
                       "keyerror", "indexerror", "modulenotfounderror", "importerror",
                       "syntaxerror", "zerodivisionerror", "recursionerror")
    if any(mk in stderr_a or mk in stderr_b for mk in runtime_markers):
        return "runtime_error"
    if "assertionerror" in stderr_a or "assertionerror" in stderr_b:
        return "assertion_failed"
    return "other"


fail = samples[~(samples["passed_a"] | samples["passed_b"])].copy()
fail["fail_cat"] = fail.apply(classify_failure, axis=1)

# Per-model breakdown (% of failed samples)
fail_pivot = (fail.groupby(["model", "fail_cat"]).size()
                   .unstack(fill_value=0)
                   .reindex(MODELS))
fail_pivot["TOTAL_FAILED"] = fail_pivot.sum(axis=1)
fail_pivot_pct = (fail_pivot.iloc[:, :-1].div(fail_pivot["TOTAL_FAILED"], axis=0) * 100).round(1)
fail_pivot_pct["TOTAL_FAILED"] = fail_pivot["TOTAL_FAILED"]
print("Failure category breakdown (% of FAILED samples per model):")
fail_pivot_pct
"""))

CELLS.append(code(r"""
# Bar chart of failure mode mix per model
fail_cats = ["assertion_failed", "runtime_error", "timeout", "parse_or_empty", "other"]
fc_colors = {
    "assertion_failed": "#3498DB",   # blue — wrong reading
    "runtime_error":    "#E74C3C",   # red  — plumbing
    "timeout":          "#9B59B6",
    "parse_or_empty":   "#7F8C8D",
    "other":            "#95A5A6",
}

fig, ax = plt.subplots(figsize=(11, 4.5))
y = np.arange(len(MODELS))
left = np.zeros(len(MODELS))
for cat in fail_cats:
    if cat not in fail_pivot_pct.columns:
        continue
    vals = fail_pivot_pct[cat].reindex(MODELS).fillna(0).values
    ax.barh(y, vals, left=left, color=fc_colors[cat], edgecolor="white",
            linewidth=0.7, label=cat.replace("_", " "))
    for yi, (l, v) in enumerate(zip(left, vals)):
        if v >= 6:
            ax.text(l + v/2, yi, f"{v:.0f}%", ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")
    left += vals

ax.set_yticks(y)
ax.set_yticklabels([shorten_model_name(m) for m in MODELS])
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xlabel("% of FAILED samples (per model)")
ax.set_title("Failure mode mix among samples that satisfied neither A nor B")
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
plt.tight_layout()
plt.show()
"""))

CELLS.append(md(r"""
**Reading.** The blue ("assertion failed") share is the proper "wrong-reading" failure — the code ran but produced output matching neither A nor B. The red ("runtime error") share is plumbing noise we should reduce in future iterations (especially in DS-1000, where NameError on `df` / `X` is a recurring story).
"""))


# ── 6. AC sample gallery ─────────────────────────────────────────────────────

CELLS.append(md(r"""
## 6. AC sample gallery — every clarification question across all 5 models

AC samples are rare (0–1.7% per model). All of them fit on a few screens. We dump the full prose / question for each, so we can read them.
"""))

CELLS.append(code(r"""
ac = samples[samples["behavior_label"] == "AC"].sort_values(["model", "task_id"])
print(f"Total AC samples across all models: {len(ac)}")
print(f"Per model: {ac.groupby('model').size().to_dict()}")
"""))

CELLS.append(code(r"""
# Print one entry per AC sample
def render_ac(row):
    print("─" * 80)
    print(f"  {row['model']:<20}  {row['task_id']:<12}  ({row['ambiguity_type']}, {row['source']})")
    print(f"  PERTURBED PROMPT:")
    print("    " + (row["perturbed_prompt"] or "")[:400].replace("\n", "\n    "))
    print(f"  INTERP_A: {row['interpretation_a'][:160]}")
    print(f"  INTERP_B: {row['interpretation_b'][:160]}")
    print(f"  MODEL OUTPUT (raw response):")
    raw = row["raw_response"] or ""
    print("    " + raw[:600].replace("\n", "\n    "))
    print()

for _, r in ac.iterrows():
    render_ac(r)
"""))


# ── 7. EA sample gallery ─────────────────────────────────────────────────────

CELLS.append(md(r"""
## 7. EA sample gallery — what assumptions do models declare?

EA samples are more common (6–17% per model). We can't enumerate all, but we sample 2 per model and look at the *prose* field — that's the model's declaration of which reading it picked.
"""))

CELLS.append(code(r"""
import random
random.seed(42)

ea = samples[samples["behavior_label"] == "EA"].copy()
print(f"Total EA samples: {len(ea)}\n")

# Sample up to 2 per model, spread across ambiguity types if possible
selected = []
for m in MODELS:
    sub = ea[ea["model"] == m]
    if len(sub) == 0:
        continue
    pick = sub.sample(min(2, len(sub)), random_state=42)
    selected.extend(pick.to_dict("records"))

for r in selected:
    print("─" * 80)
    print(f"  {r['model']:<20}  {r['task_id']:<12}  ({r['ambiguity_type']}, "
          f"passed_a={r['passed_a']}, passed_b={r['passed_b']})")
    code_str = (r["generated_code"] or "")[:400]
    prose_str = (r["prose"] or "")[:400]
    if prose_str.strip():
        print(f"  PROSE: {prose_str[:300].replace(chr(10), ' ')}")
    print(f"  CODE (first 400 chars):")
    print("    " + code_str.replace("\n", "\n    "))
    print()
"""))


# ── 8. Hardest / easiest / most-variable items ───────────────────────────────

CELLS.append(md(r"""
## 8. Hardest / easiest / most-variable items

Three slices:
- **Hardest** = highest mean tax across 5 models
- **Easiest** = lowest mean tax (most negative)
- **Most variable** = highest std of tax across models — these reveal *interaction* between item and model
"""))

CELLS.append(code(r"""
tax_mat = per_item.pivot(index="task_id", columns="model", values="tax_pp")[MODELS]
tax_summary = pd.DataFrame({
    "mean_tax": tax_mat.mean(axis=1).round(1),
    "std_tax":  tax_mat.std(axis=1).round(1),
}).join(meta)

print("TOP 8 hardest items (highest mean tax):")
print(tax_summary.sort_values("mean_tax", ascending=False).head(8).to_string())
print()
print("TOP 8 easiest items (lowest / most negative mean tax):")
print(tax_summary.sort_values("mean_tax").head(8).to_string())
print()
print("TOP 8 most VARIABLE items (highest std across models):")
print(tax_summary.sort_values("std_tax", ascending=False).head(8).to_string())
"""))

CELLS.append(md(r"""
**Read with caveat.** "Hardest" items are tax-positive across the board — they should be the headline items in the poster. "Easiest" items are tax-negative across the board — these may be benchmark artifacts (Stage-1 perturbation accidentally clarifying a vague clean prompt). "Most variable" items are where a specific model's reading bias kicks in — useful for failure case studies.
"""))


# ── 9. Sonnet anomaly drill-down ─────────────────────────────────────────────

CELLS.append(md(r"""
## 9. Sonnet negative-tax anomaly drill-down

The milestone notebook flagged Sonnet's negative tax. Here we triangulate: per-item, how does Sonnet's tax compare to the **mean of the other 4 models**? Items where Sonnet diverges the most from peers are the ones to investigate.
"""))

CELLS.append(code(r"""
# Per-item: Sonnet's tax minus mean of the other 4 models
others = [m for m in MODELS if m != "claude-sonnet"]
peer_mean = tax_mat[others].mean(axis=1)
delta = (tax_mat["claude-sonnet"] - peer_mean).rename("sonnet_minus_peers")

frame = pd.concat([tax_mat[["claude-sonnet"]].rename(columns={"claude-sonnet": "sonnet_tax"}),
                    peer_mean.rename("peers_mean_tax"), delta], axis=1).join(meta)
frame = frame.sort_values("sonnet_minus_peers")

print("Top 8 items where Sonnet is HELPED MORE than peers (negative delta):")
print(frame.head(8)[["sonnet_tax", "peers_mean_tax", "sonnet_minus_peers",
                     "ambiguity_type", "anchor_task_id"]].round(1).to_string())
print()
print("Top 8 items where Sonnet is HURT MORE than peers (positive delta):")
print(frame.tail(8)[["sonnet_tax", "peers_mean_tax", "sonnet_minus_peers",
                     "ambiguity_type", "anchor_task_id"]].round(1).to_string())
"""))

CELLS.append(code(r"""
# For ONE of the most extreme negative-delta items, dump the actual generated code from each model
target = frame.head(1).index[0]
print(f"Drilling into {target} (anchor: {meta.loc[target, 'anchor_task_id']})")
print(f"Type: {meta.loc[target, 'ambiguity_type']}, Source: {meta.loc[target, 'source']}")
print()

item_samples = samples[samples["task_id"] == target]
prompts_done = set()
for m in MODELS:
    rows_m = item_samples[item_samples["model"] == m]
    if rows_m.empty:
        continue
    if m not in prompts_done:
        # Print the perturbed prompt only once
        if not prompts_done:
            print("PERTURBED PROMPT:")
            print("  " + (rows_m.iloc[0]["perturbed_prompt"] or "")[:500].replace("\n", "\n  "))
            print()
            print(f"INTERP_A: {rows_m.iloc[0]['interpretation_a'][:200]}")
            print(f"INTERP_B: {rows_m.iloc[0]['interpretation_b'][:200]}")
            print()
        prompts_done.add(m)
    # Dump first sample's code from this model
    s = rows_m.iloc[0]
    print(f"── {m}  (sample 0)  passed_a={s['passed_a']}, passed_b={s['passed_b']}, label={s['behavior_label']} ──")
    print("  " + (s["generated_code"] or "")[:500].replace("\n", "\n  "))
    print()
"""))


# ── 10. Latency profile ──────────────────────────────────────────────────────

CELLS.append(md(r"""
## 10. Latency profile — what's the wall-clock cost of each model?

Per-sample latency (seconds). Mean / median / p95 give us a sense of how much
slower reasoning models are.
"""))

CELLS.append(code(r"""
lat = (samples.groupby("model")["latency_s"]
                  .agg(["mean", "median",
                       lambda x: x.quantile(0.95),
                       "max", "count"])
                  .rename(columns={"<lambda_0>": "p95"})
                  .reindex(MODELS).round(2))
print("Per-sample latency (seconds):")
lat
"""))

CELLS.append(code(r"""
# Per-model latency boxplot
fig, ax = plt.subplots(figsize=(9, 4.5))
data = [samples[samples["model"] == m]["latency_s"].values for m in MODELS]
bp = ax.boxplot(data, patch_artist=True, showmeans=True, widths=0.6,
                medianprops=dict(color="black", linewidth=1.5))
for patch, m in zip(bp["boxes"], MODELS):
    patch.set_facecolor(MODEL_COLORS[m])
    patch.set_alpha(0.85)
ax.set_xticklabels([shorten_model_name(m) for m in MODELS], rotation=15, ha="right")
ax.set_ylabel("Latency (s) per LLM call")
ax.set_title("Per-sample latency distribution")
ax.set_yscale("log")
plt.tight_layout()
plt.show()
"""))


# ── 11. B-pickup analysis ────────────────────────────────────────────────────

CELLS.append(md(r"""
## 11. Interpretation-B pickup analysis

Headline finding: A-bias is 73-81%. But on which items does *anyone* actually pick interpretation B? What types?
"""))

CELLS.append(code(r"""
# For each item, count how many of the 5 models picked B at least once
b_count_per_item = (per_item.assign(picked_b=lambda d: d["chose_b_count"] > 0)
                              .groupby("task_id")["picked_b"].sum())

print(f"Items where ≥1 model picked B at least once: "
      f"{(b_count_per_item >= 1).sum()}/{len(b_count_per_item)}")
print(f"Items where ≥3 models picked B at least once: "
      f"{(b_count_per_item >= 3).sum()}")
print()

b_friendly = b_count_per_item[b_count_per_item >= 3].index
print("Items where 3+ models successfully picked B (B-friendly items):")
print(meta.loc[b_friendly].to_string())
"""))

CELLS.append(code(r"""
# Among (model, item) combos that have chose_b > 0, what's the distribution by ambiguity_type?
b_picked = per_item[per_item["chose_b_count"] > 0]
print("(model, item) pairs where B was picked at least once, by ambiguity_type:")
ct = b_picked.groupby(["ambiguity_type", "model"]).size().unstack(fill_value=0).reindex(columns=MODELS, fill_value=0)
ct["TOTAL"] = ct.sum(axis=1)
ct
"""))


# ── 12. Open questions ───────────────────────────────────────────────────────

CELLS.append(md(r"""
## 12. Open questions / next experiments

1. **Sonnet anomaly mechanism.** Hypotheses to test:
   - (a) **Brittle baseline + diverse sampling**. Re-run baseline ONLY for Sonnet at n=20 on the negative-delta items; if Sonnet's baseline variance is much higher than others, hypothesis (a) is supported.
   - (b) **Paraphrase robustness**. Compare Sonnet's per-item baseline to the others' baselines. If Sonnet's clean-prompt pass rate is systematically lower while others are higher on the *same* items, hypothesis (b) gets a vote.
2. **Reasoning vs non-reasoning AC gap.** Add Qwen 3.6 Plus (reasoning) and re-evaluate. If AC remains 0% for Qwen, the "reasoning → AC > 0" claim weakens; if it sits at 1–2%, the claim strengthens.
3. **Test schema brittleness in DS-1000.** §5 should report the exact fraction of failures attributable to runtime / NameError / wrong-format. If it's > 30% of failures on DS-1000, the v3 priority is rewriting tests with format tolerance.
4. **B-friendly items audit.** The handful of items where multiple models successfully picked B (§11) tell us where the perturbation is genuinely two-way. Pull these into a "controlled-difficulty" subset and re-compute A-bias.
5. **A-bias controlled subset.** Re-compute A-bias only on items where Stage-2 entropy ≈ 1 (genuine bilateral ambiguity). If A-bias drops from 73–81% to ~50%, the residual bias is construction artifact.
6. **n_samples sensitivity.** Bootstrap CIs span 25 pp at n=5. At n=10 we'd expect tighter; one model at n=10 (probably gpt-5.5) would calibrate how much bigger an effect we're missing.
"""))


# ── Build ─────────────────────────────────────────────────────────────────────

NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(NOTEBOOK, indent=1, ensure_ascii=False))
    print(f"Wrote {OUT}  ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()