# AmbiCode-Eval — Project Status

*Last updated: 2026-05-04*

## Current State

**Phase 1 + Phases 2–4 complete.**

- **62 verified benchmark items** in `data/benchmark/benchmark.jsonl` (MBPP 26 + DS-1000 36)
- **Full evaluation pipeline** runs end-to-end for any model: baseline → perturbed → classify → analyze
- **Cross-model analysis** computes Ambiguity Tax, conditional pass@k, interpretation bias, and behavioral distributions

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
Modern instruction-tuned LLMs rarely ask clarification questions even when given permission. AC=0% is a valid observation, not a bug. To trigger more AC behavior would require meta-prompts that contaminate the measurement.

## Sample Workflow

```bash
# 1. Build benchmark (Phase 1 — already done; benchmark.jsonl is in git)
python scripts/run_scaled_pipeline.py

# 2. Evaluate one model end-to-end (Phases 2-4)
python scripts/run_full_pipeline.py \
    --model gpt-5.4 --n-samples 5 --temperature 0.8

# 3. Repeat for additional models
python scripts/run_full_pipeline.py --model claude-sonnet --n-samples 5 --temperature 0.8

# 4. Cross-model report
python scripts/analyze_results.py    # auto-discovers all results
```

Outputs land in `data/results/` (jsonl per phase + plots + `metrics_summary.csv`).