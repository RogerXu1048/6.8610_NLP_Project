# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**AmbiCode-Eval** — a benchmark of **48 verified items** (`benchmark_v2_full.jsonl`) measuring how LLMs handle linguistically ambiguous coding prompts. Quantifies the "Ambiguity Tax" (pass@k drop from ambiguity injection) and classifies model behavior into Silent Assumption / Explicit Assumption / Active Clarification.

**Status (2026-05-06)**: project complete. 5 SOTA models evaluated end-to-end (GPT-5.5, Claude Sonnet 4.6, Claude Opus 4.6, Gemini 3.1 Pro, DeepSeek V4 Pro). Two notebooks delivered: poster-facing (`milestone_analysis.ipynb`) and team-facing (`internal_result_report.ipynb`).

**Three findings**: (1) anti-calibration — 4 / 5 models go *more* silent on high-risk items; AC = 0% across all models on high-risk; (2) AC is a deliberation product (3 × SA latency for Opus / Gemini) but reasoning is necessary not sufficient (DeepSeek inverts); (3) no single best model on ambiguity — every model has its own weak ambiguity type.

Target models: GPT, Claude, Gemini, DeepSeek, Qwen — all called via OpenRouter.

## Commands

```bash
pip install -e ".[dev]"        # install project + dev dependencies
pytest                          # run all tests
pytest tests/ -k "test_name"   # run a single test
```

Requires a `.env` file with `OPENROUTER_API_KEY` (see `.env.example`).
Docker must be running for sandbox execution.
Use the base Anaconda env (`/Users/ender_yang/opt/anaconda3/bin/python3`).

## Architecture

```
src/
├── data/
│   ├── model.py        # BenchmarkTask + BenchmarkItem dataclasses
│   ├── loaders.py      # Per-source loaders (HumanEval, MBPP, DS-1000) via HuggingFace
│   ├── ds1000_normalizer.py  # DS-1000 harness → concatenation format converter
│   └── store.py        # BenchmarkStore — load, filter, save/reload from JSONL
├── pipeline/
│   ├── prompts.py      # Prompt loader — reads from config/prompts.yaml
│   ├── anchor_selection.py  # Phase 1 anchor scoring pipeline
│   ├── perturbation.py      # Shared: ambiguity type defs, anchor loading/selection
│   ├── stage1_perturbation.py   # Stage 1: SOTA models generate perturbed prompts
│   ├── stage2_entropy_gate.py   # Stage 2: judge models vote → entropy filter
│   ├── stage3_test_generation.py # Stage 3: generate ref_solution_b + test_b
│   └── stage4_exclusivity_gate.py # Stage 4: Docker sandbox 2×2 verification
├── util/
│   ├── llm.py          # Unified LLM client — OpenRouter (OpenAI-compatible API)
│   ├── sandbox.py      # Docker-based Python sandbox — no network, mem/pid limits
│   ├── parsing.py      # Shared JSON extraction from LLM responses
│   └── pipeline_runner.py  # Generic concurrent pipeline runner with JSONL output
config/
├── models.yaml         # Model alias → OpenRouter ID registry
├── pipeline.yaml       # Pipeline parameters (judge models, concurrency, etc.)
└── prompts.yaml        # All system/task prompts (single source of truth)
scripts/
├── download_data.py    # Download all benchmarks to data/raw/
├── normalize_ds1000.py # Normalize DS-1000 tasks for concatenation execution
├── run_anchor_selection.py  # Phase 1.1b: Run anchor selection
├── run_perturbation.py      # Phase 1.2-1.5: Run 4-stage perturbation pipeline
├── run_scaled_pipeline.py   # Phase 1: Scaled pipeline — 10 parallel workers
├── run_baseline_eval.py     # Phase 2: Baseline (clean prompt) inference
├── run_perturbed_eval.py    # Phase 2: Perturbed (ambiguous prompt) inference
├── run_classification.py    # Phase 3: SA/EA/AC behavioral classification
├── analyze_results.py       # Phase 4: Aggregate, metrics, plots
└── run_full_pipeline.py     # Phases 2-4 end-to-end orchestrator
docker/
└── ds1000.Dockerfile   # Docker image with data science packages for DS1000
docs/
├── data_guide.md       # Full pipeline guide for all phases
├── benchmark_guide.md  # Benchmark item format + downstream usage
└── project_status.md   # Current status, known issues, scaling plan
data/
├── raw/                # Downloaded benchmark JSONL (gitignored)
├── intermediate/       # Pipeline intermediate outputs (gitignored)
├── benchmark/          # Final benchmark items (benchmark.jsonl tracked in git)
└── results/            # Phase 2-4 outputs: baseline/perturbed/classified JSONL + plots
```

### Key Design Decisions

- **All prompts** live in `config/prompts.yaml` — never hardcoded in Python
- **Model registry** in `config/models.yaml` — aliases map to OpenRouter IDs
- **Pipeline config** in `config/pipeline.yaml` — judge models, concurrency, token limits
- **Combined evaluation** — each judge makes ONE API call covering ambiguity + risk + feasibility (not 3 separate calls) for speed
- **Structured boolean rubrics** — all scoring uses yes/no questions, not subjective scales, for cross-model consistency

### Data Layer (`src/data/`)

- `BenchmarkTask` — unified dataclass for raw benchmark tasks across all sources
- `BenchmarkItem` — extends with perturbation fields + quality gate (Phase 1 final deliverable)
- `BenchmarkStore` — in-memory store with `filter(source=, library=)`, `save()` / `load_local()` JSONL
- Loaders normalise HumanEval (164), MBPP-sanitized (257), DS-1000 (1000) into `BenchmarkTask`
- `ds1000_normalizer` — converts DS-1000 harness format to concatenation-friendly format (845 non-Matplotlib tasks)

### Pipeline (`src/pipeline/`)

- `prompts.py` — `get_prompt(path)`, `render_prompt(path, **vars)`, `load_pipeline_config()`
- `anchor_selection.py` — scores each anchor with N judges via combined evaluation call; aggregates into `AnchorResult`; writes to JSONL incrementally with progress tracking
- `perturbation.py` — shared constants (`AMBIGUITY_TYPE_DEFS`), `load_anchor_results()`, `select_anchors()`
- `stage1_perturbation.py` — SOTA models generate perturbed_prompt + interpretation_a + interpretation_b
- `stage2_entropy_gate.py` — judge models vote A/B on perturbed prompts, compute Shannon entropy, filter H >= 0.72
- `stage3_test_generation.py` — generate ref_solution_b + test_b for entropy-passed items
- `stage4_exclusivity_gate.py` — Docker sandbox runs 2×2 matrix (ref_a/b × test_a/b), all 4 must hold

### LLM Client (`src/util/llm.py`)

- Uses OpenRouter as the single gateway to all model families
- `LLMClient.call()` returns `LLMResponse` with `choices` list (one per `n`)
- Supports temperature sampling (n>1) for pass@k

### Sandbox (`src/util/sandbox.py`)

- Each execution runs in a fresh Docker container (`python:3.11-slim`)
- Containers have: no network, 256MB memory limit, 64 PID limit, configurable timeout
- `run(code, test_code)` — concatenates code + tests, returns `SandboxResult`
- `run_dual_blind(code, test_a, test_b)` — runs against both interpretations
- `validate_quality_gate_a()` — checks strict exclusivity of reference solutions

## Pipeline Phases

1. **Data** (Phase 1): Anchor selection → ambiguity injection → reference solutions → test authoring → quality gate
2. **Inference** (Phase 2): Two-condition sampling (clean + perturbed) with sandbox execution
3. **Classification** (Phase 3): LLM-as-Judge → SA/EA/AC labels (Q1/Q2/Q3 rubric → deterministic mapping)
4. **Analysis** (Phase 4): Ambiguity Tax, conditional pass@k, behavioral distributions, plots

## Evaluation Metrics (Phase 4)

Three metric layers, all in `data/results/metrics_summary.{json,csv}`:

1. **Test-level (aggregate)** — sample can satisfy both tests:
   `pass_a_rate`, `pass_b_rate`, `pass_either_rate`
2. **Choice decomposition** — mutually exclusive, sums to 1:
   `chose_a_rate` (passed_a only), `chose_b_rate` (passed_b only), `both_pass_rate` (tests can't distinguish), `neither_rate`, `interp_a_bias = chose_a / (chose_a + chose_b)`
3. **Unbiased pass@k** (Chen et al. 2021) for `K_VALUES = [1, 3]`:
   `baseline_pass_at_k`, `pass_a/b/either_at_k`, `chose_a/b_at_k`, `ambiguity_tax_at_k_pp`

**Top-line Ambiguity Tax** = `baseline_pass@k − pass_either@k` (success = code satisfies *either* valid interpretation).

## System Prompt (Mode B "lightweight permission")

Same prompt for baseline and perturbed runs; perturbed-only AC is attributable to ambiguity:

```
You are a helpful Python programming assistant.
If anything about the user's request is unclear, you may ask a clarifying question.
Otherwise, write the requested Python code and wrap it in @@CODE_START@@ and @@CODE_END@@ markers.
```

Naturalistic — no meta-prompt that pre-announces "this might be ambiguous" or enumerates SA/EA/AC options.

## Current Status — PROJECT COMPLETE (2026-05-06)

- **Phase 1 (benchmark construction)**: DONE — `data/benchmark/benchmark_v2_full.jsonl` (48 items: 19 MBPP + 27 DS-1000 + 2 HumanEval)
  - v1 → v2 cleanup documented in `docs/benchmark_audit.md`
  - v2 generation pipeline (4 reforms: opt-out / info conservation / bilateral naturalness / Stage-1.5 quality gate) documented in `docs/benchmark_generated_v2.md`
  - HumanEval re-attempted with v2 pipeline; 2 items survived all stages
- **Phases 2–4 (evaluation)**: DONE — 5 SOTA models on `benchmark_v2_full.jsonl` (n=5, T=0.8)
  - Models: GPT-5.5, Claude Sonnet 4.6, Claude Opus 4.6, Gemini 3.1 Pro, DeepSeek V4 Pro
  - Per-item LLM-call parallelism (`--sample-workers 5`) — n=5 samples issued concurrently per item
  - HumanEval `check()` invocation patch in `stage4_exclusivity_gate.py` + both eval scripts
- **Phase 5 (analysis)**: DONE
  - `scripts/build_milestone_analysis.py` — aggregates results + bootstrap 95% CIs (B=2000)
  - `scripts/plot_style.py` — consistent matplotlib style + per-model brand colors
  - `data/results/milestone/` — `summary.json`, `per_item.csv`, `by_type.csv`, `by_risk.csv`, 14 figures (PNG + PDF)
- **Notebooks**: DONE
  - `notebooks/milestone_analysis.ipynb` — poster-facing (55 cells, 14 figures, 11 RQs/sub-sections)
  - `notebooks/internal_result_report.ipynb` — team-facing (37 cells, sanity checks + AC sample gallery + failure-mode dissection)
  - Both regenerated from Python sources (`scripts/build_milestone_notebook.py`, `scripts/build_internal_report.py`)
- **Poster**: design phase. Layout brainstormed in chat (3-column landscape 48"W × 36"H, 8-block); content TBD.

### Three findings (full detail in `docs/findings.md`)

1. **Anti-calibration** — 4/5 models go *more* silent on high-risk items than on low-risk; AC drops to 0% on high-risk for every model. Only Gemini is calibrated (+2.8 pp), marginally.
2. **AC = deliberation product** — Opus AC/SA latency = 3.49×, Gemini = 3.05×; DeepSeek inverts (SA latency = 41 s, AC < SA). Reasoning is necessary but not sufficient for AC > 0%.
3. **No single best model on ambiguity** — every model has its own weak type; "ambiguity-handling ability" is a multi-dimensional skill bundle, not a single number.

### Methodology limitation surfaced (Sonnet anomaly)

`tax = baseline − pass_either` is interpretable as ambiguity-handling ability *only when* the baseline reflects the model's ability on the canonical reading. Sonnet's negative aggregate tax (-7.1 pp @1) is a phrasing-brittleness artifact (Spearman ρ = +0.52 between Sonnet's tax shortfall vs peers and Sonnet's baseline shortfall vs peers, p < 0.001). Mitigation: report a "shared-baseline cohort" (items where all models achieve baseline ≥ τ) alongside the aggregate.

See `docs/findings.md` for full prose, `docs/project_status.md` for milestone status, `notebooks/milestone_analysis.ipynb` §9.7 for the limitation analysis.
