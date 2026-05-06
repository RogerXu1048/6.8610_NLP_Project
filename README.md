# AmbiCode-Eval

A benchmark of **62 tasks** measuring how LLMs handle linguistically ambiguous coding prompts.

Quantifies the **Ambiguity Tax** (pass@k drop from ambiguity injection) and classifies model behavior into Silent Assumption / Explicit Assumption / Active Clarification.

> **Status (2026-05-05)**: Phase 1 + Phases 2–4 complete. First full evaluation
> run on `gpt-5.4` and `claude-sonnet-4-6` (n=5, T=0.8). See
> [`docs/findings.md`](docs/findings.md) for results.
>
> **v2 generation pipeline** (4 reforms — opt-out, info conservation, bilateral
> naturalness, quality gate) reformed; HumanEval re-attempted; produced 2 new
> scopal items (`benchmark_humaneval_v2.jsonl`). Merged benchmark
> `benchmark_v2_full.jsonl` is 48 items (19 MBPP + 27 DS-1000 + 2 HumanEval).
> See [`docs/benchmark_generated_v2.md`](docs/benchmark_generated_v2.md).

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Set up API key
cp .env.example .env   # edit with your OPENROUTER_API_KEY

# Docker (required for sandbox execution)
docker build -t ambicode-ds1000 -f docker/ds1000.Dockerfile .
```

### Load the Benchmark

```python
import json
from src.data.model import BenchmarkItem

with open("data/benchmark/benchmark.jsonl") as f:
    items = [BenchmarkItem.from_dict(json.loads(line)) for line in f if line.strip()]
```

### Run a Simple Evaluation

```python
from src.util.llm import LLMClient, ModelConfig
from src.util.sandbox import Sandbox
from src.data.ds1000_normalizer import _wrap_solution_as_string

client = LLMClient()
sandbox = Sandbox()                                  # MBPP / HumanEval
sandbox_ds = Sandbox(image="ambicode-ds1000")        # DS-1000
config = ModelConfig(model="gpt-5.4-mini", temperature=0.8, max_tokens=1024)

item = items[0]

# Baseline (clean prompt)
resp = client.call(config, prompt=item.prompt, system=SYSTEM_PROMPT)
result = sandbox.run(resp.choices[0], item.test_code)

# Perturbed (ambiguous prompt) — dual-blind execution
resp = client.call(config, prompt=item.perturbed_prompt, system=SYSTEM_PROMPT)
if item.source == "ds1000":
    # test_a is the harness (needs __SOLUTION__); test_b is self-contained
    wrapped = _wrap_solution_as_string(resp.choices[0])
    result_a = sandbox_ds.run(wrapped, item.test_a, timeout_s=60)
    result_b = sandbox_ds.run(resp.choices[0], item.test_b, timeout_s=60)
else:
    result_a, result_b = sandbox.run_dual_blind(
        resp.choices[0], item.test_a, item.test_b
    )
```

See `notebooks/evaluation_demo.ipynb` for a complete walkthrough.

### Run the Full Evaluation Pipeline

End-to-end evaluation (baseline → perturbed → classify → analyze) for one model:

```bash
# Single model, full pipeline
python scripts/run_full_pipeline.py \
    --model anthropic/claude-sonnet-4-6 \
    --n-samples 5 \
    --temperature 0.8

# Quick sanity check (3 items, MBPP only)
python scripts/run_full_pipeline.py --model gpt-5.4 --limit 3 --skip-ds1000
```

The pipeline auto-selects a judge model to avoid same-family circularity:
- Claude models → judged by `gpt-5.4-mini`
- All other models → judged by `claude-haiku`

Run individual phases if needed:

```bash
# Phase 2: Baseline (clean prompt) inference
python scripts/run_baseline_eval.py --model gpt-5.4 --n-samples 5

# Phase 2: Perturbed (ambiguous prompt) inference
python scripts/run_perturbed_eval.py --model gpt-5.4 --n-samples 5

# Phase 3: SA/EA/AC behavioral classification
python scripts/run_classification.py \
    --input data/results/perturbed_gpt-5.4_<timestamp>.jsonl \
    --judge-model claude-haiku

# Phase 4: Aggregate + plots
python scripts/analyze_results.py
```

Cross-model analysis (combine multiple models into one report):

```bash
python scripts/analyze_results.py \
    --baseline   data/results/baseline_anthropic_claude-sonnet-4-6_*.jsonl \
                 data/results/baseline_gpt-5.4_*.jsonl \
    --perturbed  data/results/perturbed_anthropic_claude-sonnet-4-6_*.jsonl \
                 data/results/perturbed_gpt-5.4_*.jsonl \
    --classified data/results/classified_anthropic_claude-sonnet-4-6_*.jsonl \
                 data/results/classified_gpt-5.4_*.jsonl \
    --label "claude-sonnet-4-6" "gpt-5.4" \
    --output data/results/analysis_combined
```

Outputs:
- `data/results/baseline_<model>_<ts>.jsonl` — per-item pass rates against `test_code`
- `data/results/perturbed_<model>_<ts>.jsonl` — pass@k(A) and pass@k(B) under dual-blind
- `data/results/classified_<model>_<ts>.jsonl` — SA/EA/AC labels per sample
- `data/results/analysis_<model>_<ts>/*.png` — Ambiguity Tax, behavioral distribution, per-item deltas
- `data/results/metrics_summary.csv` / `.json` — aggregated metrics (one row per model)

## Benchmark Overview

Each benchmark item contains:

| Layer | Fields | Purpose |
|-------|--------|---------|
| **Anchor** | `prompt`, `canonical_solution`, `test_code` | Baseline condition |
| **Perturbation** | `perturbed_prompt`, `interpretation_a/b`, `ref_solution_a/b`, `test_a/b` | Experimental condition |
| **Quality Gates** | `quality_gate_a` (sandbox exclusivity), `quality_gate_b` (entropy gate) | Validation |

### Distribution

| Ambiguity Type | Low Risk | High Risk | Total |
|----------------|----------|-----------|-------|
| Coreferential | 8 | 3 | 11 |
| Syntactic | 10 | 4 | 14 |
| Scopal | 5 | 3 | 8 |
| Collective/Distributive | 16 | 3 | 19 |
| Elliptical | 7 | 3 | 10 |
| **Total** | **46** | **16** | **62** |

Sources: MBPP (26), DS-1000 (36).

## Source-Specific Prompting

Each benchmark source requires dedicated prompt engineering for fair evaluation. **Failing to apply these strategies will cause false negatives** (correct code that fails tests due to naming/format mismatches).

### MBPP Items

MBPP prompts are natural language descriptions that do **not** specify the expected function name.
Tests call a specific function name (e.g., `add_lists`).

**Required**: Extract the function name from `test_code` and append it to the prompt.

```python
import re

def extract_function_name(test_code):
    m = re.search(r'assert\s+(\w+)\s*\(', test_code)
    return m.group(1) if m else None

func_name = extract_function_name(item.test_code)
prompt = f"{item.prompt}\nThe function should be named `{func_name}`."
```

**Output format**: Complete, self-contained Python function.

**Sandbox**: `Sandbox()` (default `python:3.11-slim` image).

### DS-1000 Items

DS-1000 prompts are data science problems with embedded setup code and example data.
The benchmark uses a **normalized format**:

- `canonical_solution`: code fragment wrapped as `__SOLUTION__ = r"""..."""`
- `test_code`: original DS-1000 harness + `test_execution(__SOLUTION__)`
- `ref_solution_b` / `test_b`: self-contained (include their own imports and test data)

**Required**: Wrap LLM output as `__SOLUTION__` when testing against `test_a`.

```python
from src.data.ds1000_normalizer import _wrap_solution_as_string

# Testing against test_a (harness format)
wrapped = _wrap_solution_as_string(llm_output)
result_a = sandbox_ds.run(wrapped, item.test_a, timeout_s=60)

# Testing against test_b (self-contained)
result_b = sandbox_ds.run(llm_output, item.test_b, timeout_s=60)
```

**Output format**: Code fragment (not a complete function).

**Sandbox**: `Sandbox(image="ambicode-ds1000")` — requires building the Docker image.

### Common: Code Extraction

`scripts/run_perturbed_eval.py` extracts code from raw LLM responses with this priority:

1. `@@CODE_START@@ ... @@CODE_END@@` markers (system-prompt-instructed)
2. `` ```python ... ``` `` fence (markdown fallback)
3. `<code>...</code>` HTML tags
4. **Heuristic**: text contains code patterns (`def`/`import`/`return`/`=`) AND not a question
5. Empty (response is pure prose — likely a clarification question)

If none of 1–4 match, `code = ""` so the response is treated as a clarification (AC) rather than being mis-fed to the sandbox.

### System Prompt Design

The pipeline uses **Mode B "lightweight permission"** — naturalistic, model self-detects ambiguity:

```
You are a helpful Python programming assistant.
If anything about the user's request is unclear, you may ask a clarifying question.
Otherwise, write the requested Python code and wrap it in @@CODE_START@@ and @@CODE_END@@ markers.
```

**Why this wording**:
- Does not pre-announce "this prompt may be ambiguous" (which would be a meta-prompt biasing toward AC)
- Does not enumerate SA/EA/AC as options (same reason)
- Grants permission to ask without strongly encouraging it
- Identical for baseline and perturbed prompts (so AC behavior is attributable to ambiguity, not prompt difference)

**Empirical finding**: with this prompt, modern instruction-tuned coding models still default to writing code; AC rates are typically <5%. This reflects real deployment behavior. To raise AC rates, you would need to explicitly enumerate options (which contaminates the measurement).

## Evaluation Metrics

The pipeline reports three layers of metrics. Use them at increasing granularity.

### 1. Test-level rates (a sample can satisfy both tests)

| Metric | Formula | Meaning |
|---|---|---|
| `pass_a_rate` | `Σ pass_a_count / Σ n_samples` | Fraction of samples whose code passes test_a |
| `pass_b_rate` | `Σ pass_b_count / Σ n_samples` | Fraction passing test_b |
| `pass_either_rate` | `Σ pass_either_count / Σ n_samples` | Fraction passing test_a OR test_b |

### 2. Choice decomposition (mutually exclusive — sums to 100%)

| Metric | Condition | Interpretation |
|---|---|---|
| `chose_a_rate` | `passed_a=T, passed_b=F` | Model picked interpretation A |
| `chose_b_rate` | `passed_b=T, passed_a=F` | Model picked interpretation B |
| `both_pass_rate` | both pass | Tests cannot distinguish (don't use for bias analysis) |
| `neither_rate` | neither pass | Code error or wrong choice |
| `interp_a_bias` | `chose_a / (chose_a + chose_b)` | Among decisive samples, fraction picking A (50% = unbiased) |

### 3. Unbiased pass@k (Chen et al. 2021)

Per-item pass@k = `1 - C(n-c, k) / C(n, k)`, then averaged across items.

| Metric | Meaning |
|---|---|
| `baseline_pass_at_1`, `baseline_pass_at_3` | Baseline pass@k on clean prompts |
| `pass_a_at_k`, `pass_b_at_k`, `pass_either_at_k` | Conditional pass@k for each interpretation |
| `chose_a_at_k`, `chose_b_at_k` | pass@k of the choice decomposition events |
| `ambiguity_tax_at_k_pp` | `(baseline_pass_at_k − pass_either_at_k) × 100` |

**Default** computes both `k=1` and `k=3`. `n_samples=5` is the minimum to support pass@3.

### Top-line metrics

- **Ambiguity Tax** = `baseline_pass@k − pass_either@k` (model is "successful" if it produces valid code for either valid interpretation)
- **Interpretation Bias** = `interp_a_bias` (whether the model systematically prefers one reading)
- **Behavioral Distribution** = SA / EA / AC fractions from the LLM-judge classifier

## Project Structure

```
data/
  benchmark/benchmark.jsonl     # The benchmark (62 items)
  raw/                          # Raw benchmark sources (gitignored)
  intermediate/                 # Pipeline intermediate outputs (gitignored)

src/
  data/                         # Data models, loaders, DS-1000 normalizer
  pipeline/                     # 4-stage perturbation pipeline
  util/                         # LLM client, Docker sandbox, parsing

config/
  models.yaml                   # Model alias -> OpenRouter ID registry
  pipeline.yaml                 # Pipeline parameters
  prompts.yaml                  # All LLM prompts (single source of truth)

scripts/
  download_data.py              # Download raw benchmarks
  normalize_ds1000.py           # Normalize DS-1000 format
  run_anchor_selection.py       # Phase 1.1b: Score anchors for ambiguity potential
  run_perturbation.py           # Phase 1.2-1.5: 4-stage perturbation pipeline
  run_scaled_pipeline.py        # Phase 1: Scaled pipeline (10 parallel workers)
  run_baseline_eval.py          # Phase 2: Baseline (clean prompt) inference
  run_perturbed_eval.py         # Phase 2: Perturbed (ambiguous prompt) inference
  run_classification.py         # Phase 3: SA/EA/AC behavioral classification
  analyze_results.py            # Phase 4: Aggregate metrics + plots
  run_full_pipeline.py          # Phases 2-4 end-to-end orchestrator

notebooks/
  benchmark_demo.ipynb          # Explore benchmark structure
  evaluation_demo.ipynb         # Full evaluation walkthrough

docs/
  benchmark_guide.md            # Detailed benchmark format + usage
  data_guide.md                 # Pipeline construction guide
  project_status.md             # Current status + known limitations

docker/
  ds1000.Dockerfile             # Docker image for DS-1000 execution
```

## Reproducing the Pipeline

### Phase 1 — Build the Benchmark

```bash
# 1. Download raw data
python scripts/download_data.py

# 2. Normalize DS-1000
python scripts/normalize_ds1000.py

# 3. Run anchor selection (scores 1,421 tasks)
python scripts/run_anchor_selection.py

# 4. Run scaled pipeline (generates benchmark items)
python scripts/run_scaled_pipeline.py --dry-run    # preview
python scripts/run_scaled_pipeline.py              # run
```

### Phases 2-4 — Evaluate Models

```bash
# End-to-end for one model
python scripts/run_full_pipeline.py --model gpt-5.4 --n-samples 5 --temperature 0.8

# Repeat for additional models
python scripts/run_full_pipeline.py --model anthropic/claude-sonnet-4-6 --n-samples 5
python scripts/run_full_pipeline.py --model gemini-3.1-pro --n-samples 5

# Phase 5: Cross-model comparison
python scripts/analyze_results.py    # auto-discovers all baseline/perturbed/classified files
```

## Requirements

- Python 3.9+
- Docker Desktop (for sandbox execution)
- OpenRouter API key (for LLM calls)
- ~$5-10 in API credits for full benchmark construction

## License

MIT 6.8610 NLP course project.
