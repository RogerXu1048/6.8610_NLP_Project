# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**AmbiCode-Eval** — a benchmark of 62 tasks measuring how LLMs handle linguistically ambiguous coding prompts. Quantifies the "Ambiguity Tax" (pass@k drop from ambiguity injection) and classifies model behavior into Silent Assumption / Explicit Assumption / Active Clarification.

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
2. **Inference** (Phase 2): Two-condition batch sampling (clean + perturbed) with temperature sampling
3. **Classification** (Phase 3): LLM-as-Judge with structured boolean rubric → SA/EA/AC labels
4. **Execution** (Phase 4): Dual-blind sandbox against Test-A and Test-B
5. **Analysis** (Phase 5): Ambiguity Tax, behavioral distributions, conditional pass@k

## Current Status

- **Phase 1 (Benchmark Construction)**: DONE — 62 verified items in `data/benchmark/benchmark.jsonl`
  - Sources: MBPP (26), DS-1000 (36)
  - All 5 ambiguity types and both risk levels represented
  - DS-1000 normalization done (845 tasks, Matplotlib excluded)
  - Scaled pipeline built (`scripts/run_scaled_pipeline.py`)
- **Phases 2–4 (Inference, Classification, Analysis)**: DONE
  - Phase 2 — `run_baseline_eval.py` + `run_perturbed_eval.py`: two-condition sampling with sandbox execution
  - Phase 3 — `run_classification.py`: LLM-as-Judge with 3-question rubric → SA/EA/AC labels
  - Phase 4 — `analyze_results.py`: Ambiguity Tax, behavioral distributions, conditional pass@k, plots
  - End-to-end: `run_full_pipeline.py` runs all 4 phases for one model
  - Judge auto-selection avoids same-family circularity (Claude models judged by gpt-5.4-mini; others by claude-haiku)
- **Phase 5 (Cross-Model Analysis)**: Available via `analyze_results.py` with `--baseline/--perturbed/--classified` lists across multiple models
- See `docs/project_status.md` for full details
