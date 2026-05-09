# AmbiCode-Eval — Project Status

*Last updated: 2026-05-06*

## Current State — **PROJECT COMPLETE**

All planned milestones for the Spring 2026 6.8610 deliverable are done:

- ✅ **Phase 1 — benchmark construction**: `benchmark_v2_full.jsonl` (48 items)
- ✅ **Phases 2–4 — model evaluation**: 5 SOTA models on the v2_full benchmark
- ✅ **Phase 5 — analysis**: cross-model aggregation + bootstrap CIs
- ✅ **Notebooks**: milestone (poster-facing) + internal report (team deep-dive)
- ✅ **Docs**: findings, audit, v2 generation pipeline, benchmark guide, this status file
- 🟡 **Poster**: design phase — 8-block landscape layout drafted, content TBD

### Headline numbers (5 models, 48 items, n=5, T=0.8)

| Model | Tax @1 | Tax @3 | A-bias | SA / EA / AC |
|---|---|---|---|---|
| GPT-5.5 | +10.8 pp | +15.4 pp | 79.5% | 90.4 / 9.6 / 0.0% |
| Claude Sonnet 4.6 | −7.1 pp | −11.9 pp | 75.2% | 87.9 / 12.1 / 0.0% |
| Claude Opus 4.6 | +6.7 pp | +3.7 pp | 80.7% | 82.1 / 17.1 / 0.8% |
| Gemini 3.1 Pro | +10.8 pp | +11.7 pp | 77.9% | 87.9 / 6.2 / 1.7% |
| DeepSeek V4 Pro | +11.2 pp | +12.9 pp | 73.8% | 81.2 / 15.8 / 1.2% |

### Three findings (full prose in [findings.md](findings.md), figures in [milestone notebook](../notebooks/milestone_analysis.ipynb))

1. **Anti-calibration** — 4 / 5 models go *more* silent on high-risk items than on low-risk; AC drops to 0% on high-risk for every model.
2. **AC = deliberation product** — Opus and Gemini AC samples take 3× longer than SA samples; DeepSeek inverts (its SA latency is the longest of any model–behavior, AC is shorter than SA), so reasoning is necessary but not sufficient for AC > 0%.
3. **No single best model on ambiguity** — every model has its own weak ambiguity type; no type is uniformly hard or uniformly easy across models.

## Benchmark Coverage (Phase 1)

| Dimension | Current | Notes |
|---|---|---|
| Total items | 62 | Exceeds original 50 target |
| collective_distributive | 19 | 16 low, 3 high |
| syntactic | 14 | 10 low, 4 high |
| coreferential | 11 | 8 low, 3 high |
| elliptical | 10 | 7 low, 3 high |
| scopal | 8 | 5 low, 3 high |
| Source: MBPP | 26 | |
| Source: DS-1000 | 36 | Pandas (22), Sklearn (5), Scipy (4), Numpy (4), Pytorch (1) |
| Risk: low | 46 | 74% |
| Risk: high | 16 | 26% |

## Pipeline Architecture

### Phase 1 — Benchmark Construction (DONE)

```
Anchor Selection (1,421 tasks scored)
    |
DS-1000 Normalization (845 tasks normalized; Matplotlib excluded)
    |
Stage 1 — Perturbation Generation
    3 SOTA models (gpt-5.4, claude-sonnet, gemini-3.1-pro) generate
      perturbed_prompt + interpretation_a + interpretation_b
    |
Stage 2 — Entropy Gate
    5 judge models vote on which interpretation the prompt conveys.
    Shannon entropy H >= 0.72 required (at least 4–1 judge split).
    |
Stage 3 — Reference Solution B + Test Generation
    Same generator model writes ref_solution_b + test_b.
    DS-1000: generates self-contained code for test_b.
    |
Stage 4 — Exclusivity Gate (Docker sandbox, no LLM calls)
    Verifies 2x2 matrix (with cross-format adapters for DS-1000).
```

### Phases 2–4 — Model Evaluation (DONE)

```
Phase 2: Baseline Inference (run_baseline_eval.py)
    LLM with clean prompt -> sandbox -> pass_count per item
    |
Phase 2: Perturbed Inference (run_perturbed_eval.py)
    LLM with ambiguous prompt -> dual-blind sandbox (test_a + test_b)
    Records: pass_a/b/either_count, chose_a/b_count, both/neither
    |
Phase 3: Behavioral Classification (run_classification.py)
    Judge LLM answers Q1/Q2/Q3 -> deterministic SA/EA/AC label
    Auto-judge selection: Claude family judged by gpt-5.4-mini, others by claude-haiku
    |
Phase 4: Aggregation + Plots (analyze_results.py)
    Computes 3 metric layers:
      - Aggregate (test-level) rates
      - Choice decomposition (mutually exclusive, sums to 1)
      - Unbiased pass@k (Chen et al. 2021) for k in [1, 3]
    Outputs CSV/JSON summary + 8+ PNG plots
```

## Evaluation Metrics

### Three layers of granularity

1. **Test-level (aggregate)**: `pass_a_rate`, `pass_b_rate`, `pass_either_rate` — a sample can contribute to multiple
2. **Choice decomposition**: `chose_a_rate`, `chose_b_rate`, `both_pass_rate`, `neither_rate`, `interp_a_bias` — mutually exclusive
3. **Unbiased pass@k**: `pass_either_at_1`, `pass_either_at_3`, `ambiguity_tax_at_k_pp` — per-item pass@k averaged across items

### Top-line metrics

- **Ambiguity Tax** = `baseline_pass@k − pass_either@k`
  Code is "successful" if it satisfies either interpretation (both are valid given the ambiguous prompt)
- **Interpretation Bias** = `chose_a / (chose_a + chose_b)` among decisive samples
- **Behavioral Distribution** = fraction SA / EA / AC / unclassifiable / error (judge-failed)

## First Full Evaluation Run (2026-05-05)

Run config: 62 items, n=5 samples, T=0.8, Mode B prompt, all audit fixes applied.

| Metric | gpt-5.4 | claude-sonnet |
|---|---|---|
| baseline pass@1 | 50.6% | 47.1% |
| pass_either@1 | 45.8% | 57.7% |
| Ambiguity Tax @1 | +4.8 pp | −10.6 pp |
| interp_a_bias | 82.6% | 74.3% |
| SA / EA / AC | 91.3 / 8.7 / 0% | 89.0 / 10.3 / 0.6% |

Cleanest signal: gpt-5.4 on DS-1000 coreferential/scopal items shows tax of +25–35 pp.
Aggregate tax is diluted by ~6 MBPP items where Stage-1 perturbation accidentally
disambiguates a vague baseline (false-negative tax). Full breakdown in
[findings.md](findings.md).

## Recent Audit Fixes

A logic audit before running models identified and fixed several bugs:

| Bug | File | Fix |
|---|---|---|
| **DS-1000 dual-blind**: same `__SOLUTION__`-wrapped string used for both test_a (harness) and test_b (self-contained) → all `pass_b` systematically False | `run_perturbed_eval.py` | Run wrapped on test_a, raw code on test_b |
| **Ambiguity Tax** counted `pass_a` only → penalized models that correctly chose interpretation B | `analyze_results.py` | Tax now uses `pass_either` (either interpretation is valid) |
| **Conditional metrics** missing — `pass_a_rate` was treated as "P(chose A)" but actually counted samples passing both tests too | `run_perturbed_eval.py`, `analyze_results.py` | Added `chose_a/b/both/neither_count` (mutually exclusive) and `interp_a_bias` |
| **Classification "error" label** silently dropped (not in LABELS list) → judge failures invisible in stats | `analyze_results.py` | Added "error" to LABELS, displayed in tables |
| **pass@k** was computed as aggregate `c/n` (≡ pass@1) | `analyze_results.py` | Added unbiased Chen-et-al estimator with K_VALUES=[1,3] |
| **`parse_response` fallback** put entire response into `code` field → AC questions sandbox-failed AND mis-classified | `run_perturbed_eval.py` | Heuristic: text must look like code (def/import/etc) AND not end in `?` |
| **MBPP baseline/perturbed asymmetry**: baseline used plain text, perturbed used `def` + docstring → unfair comparison | `run_baseline_eval.py` | Baseline now uses same `def` + docstring structure (no leaking example) |

## System Prompt Design

The pipeline uses **Mode B "lightweight permission"**:

```
You are a helpful Python programming assistant.
If anything about the user's request is unclear, you may ask a clarifying question.
Otherwise, write the requested Python code and wrap it in @@CODE_START@@ and @@CODE_END@@ markers.
```

**Rationale**:
- Naturalistic (matches a real deployed assistant)
- Does not pre-announce "this might be ambiguous" (no meta-prompt contamination)
- Grants permission to ask without strongly encouraging it
- Same prompt for baseline and perturbed → AC differences attributable to ambiguity

**Empirical AC rate**: with this prompt + temperature=0.8, modern coding LLMs (gpt-5.4, claude-sonnet) typically show <5% AC. This reflects real default behavior — instruction-tuned models prefer writing code.

## Known Limitations

### 1. HumanEval Excluded
HumanEval prompts contain `>>>` example blocks that disambiguate perturbations. 0% Stage 4 pass rate during development.

### 2. Matplotlib DS-1000 Excluded (155 tasks)
Image-based pixel comparison cannot be normalized into assertion-style tests.

### 3. Scopal Ambiguity Under-represented
Only 36 candidates total in raw data. Scopal ambiguity is rare in code prompts.

### 4. AC Rate is Low by Construction
Modern instruction-tuned LLMs rarely ask clarification questions even when given permission. AC=0% is a valid observation, not a bug. To trigger more AC behavior would require meta-prompts that contaminate the measurement. **Confirmed empirically (2026-05-05)**: across 620 perturbed samples on gpt-5.4 + claude-sonnet, only 2 AC events (both on AMBI/021).

### 5b. Pipeline v2 + HumanEval addition (2026-05-05)

A reformed perturbation generation pipeline (4 reforms — opt-out, info
conservation, bilateral naturalness gate, new Stage-1.5 quality gate) was
implemented and run on 127 high-feasibility HumanEval anchors. See
[benchmark_generated_v2.md](benchmark_generated_v2.md) for the design and
[findings.md](findings.md) §"v2 First Application to HumanEval" for funnel.

**Headline outcomes**:
- 2/127 items survived the full pipeline (`benchmark_humaneval_v2.jsonl`),
  both scopal: AMBI/063 (`prime_fib`) and AMBI/064 (`is_equal_to_sum_even`).
- Merged with the v2-evaluation benchmark to produce
  `benchmark_v2_full.jsonl` (48 items: 19 MBPP + 27 DS-1000 + 2 HumanEval).
- A separate Stage-4 bug was discovered and fixed: HumanEval's
  `def check(candidate)` was never invoked in the sandbox script, so all
  4 cells of the 2×2 trivially passed. The patch appends
  `check({entry_point})` for HumanEval items. The same patch is now in
  `run_baseline_eval.py` and `run_perturbed_eval.py` so the merged
  benchmark can be evaluated correctly.
- Cross-model behavior on the new opt-out channel: Gemini-3.1-pro used it
  31× of 127 (24%); gpt-5.4 and claude-sonnet-4-6 never used it. This is
  itself a paper-worthy finding about instruction-following bias.

### 5. Stage 1 Perturbation Quality + Other Benchmark Issues
A full manual audit (2026-05-05, see [benchmark_audit.md](benchmark_audit.md))
identified five categories of quality issues across ~25 of the 62 items:

1. **Information leakage** in perturbation (9 items) — perturbed prompt is clearer than clean
2. **Duplicated anchors** (9 anchor groups, ~17 items) — same source task admitted multiple times
3. **Dark items** (9 items) — both clean and perturbed pass ≤20%, so they contribute no signal
4. **Contrived interpretation B** (5 items) — Stage 1 invented a second reading that's unnatural
5. **Meta-prompting** (1 item, AMBI/039) — perturbation explicitly announces ambiguity

**Resolution → benchmark v2** at `data/benchmark/benchmark_v2.jsonl` (46 items):
- 10 duplicates dropped (kept one per anchor group)
- 6 dark items dropped (Ambiguity Tax undefined under floor effect)
- 7 items had `perturbed_prompt` rewritten to remove leakage / meta-prompts
- DS-1000 elliptical (8→4) and MBPP syntactic (8→4) coverage thinned;
  plan is to backfill from HumanEval-sourced items in a follow-up Stage 1 run.

## Sample Workflow

```bash
# 1. Build benchmark (Phase 1 — already done; benchmark_v2_full.jsonl is in git)
python scripts/run_perturbation.py        # if you want to rebuild end-to-end via v2 pipeline

# 2. Evaluate ALL 5 SOTA models in parallel
./scripts/run_milestone_eval.sh           # ~80 min on a Mac with Docker

# 3. Aggregate + bootstrap CIs
python scripts/build_milestone_analysis.py

# 4. Regenerate notebooks from their Python sources
python scripts/build_milestone_notebook.py
python scripts/build_internal_report.py
```

Outputs land in `data/results/milestone/` (`summary.json`, `per_item.csv`, `by_*.csv`, and figures in PNG + PDF).

## Final deliverables

| Artifact | What it is | Where |
|---|---|---|
| Benchmark | 48 items × full schema | `data/benchmark/benchmark_v2_full.jsonl` |
| 5-model results | summary + per-item tables | `data/results/milestone/` |
| Milestone notebook | 11 figures, 7 RQs, 3 case studies | `notebooks/milestone_analysis.ipynb` |
| Internal report | sanity checks + AC gallery + failure dissection | `notebooks/internal_result_report.ipynb` |
| Poster figures | 14 PNG (300 dpi) + 14 PDF (vector) | `data/results/milestone/figures/` |