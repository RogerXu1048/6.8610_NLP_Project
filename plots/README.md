# plots/

Curated, presentation-ready figures for the AmbiCode-Eval report and poster.

```
plots/
├── findings/       ← 17 finding figures (PNG + PDF, 14 milestone + 3 deep-dive)
└── pipeline/       ← 2 pipeline diagrams rendered from docs/pipeline_diagrams.md
                       (PNG @ 300 dpi + SVG vector for InDesign / LaTeX)
```

## What's where

### `findings/`

Source: regenerated from `notebooks/milestone_analysis.ipynb` and saved by `scripts/plot_style.py::save_for_poster`. Each figure exists as both **PNG** (300 dpi raster) and **PDF** (vector — embeds Helvetica via `pdf.fonttype=42` for Illustrator-editability).

| File | Used in | Section |
|---|---|---|
| `rq1_all_metrics_overview` | README headline | §3 RQ1 |
| `rq1_headline_tax` | poster supporting | §3 RQ1 — bootstrap CI |
| `rq2_tax_by_source` / `tax_by_type` / `tax_by_risk` | poster / paper | §4 RQ2 breakdown |
| `rq3_behavior_stacked` | overview of SA/EA/AC | §5 RQ3 |
| `rq3_behavior_by_risk` | **Finding 1 (anti-calibration)** | §5.1 |
| `rq3_calibration_delta` | numeric supporting | §5.1 |
| `rq3_ea_ac_breakout` | EA + AC reasoning vs not | §5 |
| `rq3_latency_by_behavior` | **Finding 2 (deliberation)** | §5.2 |
| `rq3_ea_vs_sa_choices` | EA → reading composition | §5.3 |
| `rq3_behavior_vs_pass` | SA-dominant vs EA-dominant pass rate | §5.3 |
| `rq4_type_model_heatmap` | **Finding 3 (heterogeneous skill)** | §6 |
| `rq5_item_model_heatmap` | per-item × per-model | §7 |
| `rq6_a_bias` | A-bias 73–81% | §8 |
| `rq9_sonnet_per_item` | Sonnet drilldown | §9 (now embedded in §9.7 Limitations) |
| `limitation_baseline_brittleness` | **methodology caveat** | §9.7 |

### `pipeline/`

Source: rendered from the Mermaid blocks in [`docs/pipeline_diagrams.md`](../docs/pipeline_diagrams.md) by [`scripts/render_diagrams.sh`](../scripts/render_diagrams.sh).

| File | Diagram |
|---|---|
| `benchmark_pipeline.png` / `.svg` | How a raw coding task becomes a verified benchmark item — anchor selection → Stage 1 → Stage 1.5 quality gate → Stage 2 bilateral naturalness → Stage 3 → Stage 4 sandbox exclusivity |
| `evaluation_pipeline.png` / `.svg` | How a model is evaluated — Phase 2 (baseline + perturbed dual-blind) → Phase 3 (Q1Q2Q3 → SA/EA/AC) → Phase 4 (3-layer metrics + bootstrap CI) |

## How to regenerate

### Findings figures (after re-running an evaluation)

```bash
python scripts/build_milestone_analysis.py        # aggregate
python scripts/build_milestone_notebook.py        # regenerate notebook
jupyter nbconvert --execute --inplace notebooks/milestone_analysis.ipynb
# Notebook's save_for_poster() writes PNG + PDF to plots/findings/
```

### Pipeline diagrams (after editing docs/pipeline_diagrams.md)

```bash
./scripts/render_diagrams.sh
# extracts each ```mermaid block and renders to plots/pipeline/<NAME>.{png,svg}
# the <NAME> comes from a `%% NAME: foo` comment on the first line of the block
```

The Mermaid source file is the single source of truth. Edit `docs/pipeline_diagrams.md` (plain text), push, and either re-run the render script or rely on GitHub's native Mermaid renderer.

## Color scheme

Defined once in [`scripts/plot_style.py`](../scripts/plot_style.py):

| Concept | Color |
|---|---|
| GPT-5.5 | `#10A37F` (OpenAI teal) |
| Claude Sonnet | `#D97757` (Anthropic warm orange) |
| Claude Opus | `#7B3F00` (darker brown) |
| Gemini 3.1 Pro | `#4285F4` (Google blue) |
| DeepSeek V4 Pro | `#553C9A` (purple) |
| SA / EA / AC | `#888888` / `#F39C12` / `#27AE60` |

Diverging heatmaps use `RdBu_r` (red = hurts, blue = helps, white = ~0).

When making a NEW figure for the poster: please use `setup_style()` and `MODEL_COLORS` from `plot_style.py` so the visual language stays consistent.
