# AmbiCode-Eval — Findings

*Last updated: 2026-05-05*

First full evaluation run of the 62-item benchmark on two SOTA models, with all
audit fixes applied (DS-1000 dual-blind correct, MBPP baseline/perturbed
structurally aligned, no leaking example, Mode B "lightweight permission"
system prompt).

## Run Configuration

| Field | Value |
|---|---|
| Models | `gpt-5.4`, `claude-sonnet-4-6` |
| Items | 62 (MBPP 26 + DS-1000 36) |
| Samples per item | n=5 |
| Temperature | 0.8 |
| System prompt | Mode B (lightweight permission) |
| Judges | gpt-5.4 → claude-haiku; claude-sonnet → gpt-5.4-mini (avoids same-family circularity) |

## Top-Line Numbers

| Metric | gpt-5.4 | claude-sonnet |
|---|---|---|
| baseline pass@1 | **50.6%** | 47.1% |
| baseline pass@3 | 58.5% | 50.8% |
| pass_either@1 | 45.8% | 57.7% |
| pass_either@3 | 56.6% | 61.3% |
| **Ambiguity Tax @1** | **+4.8 pp** | **−10.6 pp** |
| **Ambiguity Tax @3** | +1.9 pp | −10.5 pp |
| interp_a_bias (decisive samples) | 82.6% | 74.3% |
| SA / EA / AC | 91.3% / 8.7% / **0%** | 89.0% / 10.3% / **0.6%** |

Two unexpected observations:
1. **claude-sonnet shows negative tax** — perturbation seems to *help* on average. This is not a real finding; it is dominated by Phase 1 perturbation-quality artifacts (see "False-Negative Tax Items" below).
2. **AC ≈ 0** for both models. claude-sonnet asked clarifying questions in only 2 of 310 perturbed samples (both on AMBI/021). gpt-5.4 never asked.

## Per-Source Breakdown (most diagnostic view)

| Model / Source | baseline | A | B | either | tax@1 |
|---|---|---|---|---|---|
| gpt-5.4 / **MBPP** | 44.6% | 50.0% | 10.8% | 60.8% | **−16.2 pp** |
| gpt-5.4 / **DS-1000** | 55.0% | 29.4% | 7.8% | 35.0% | **+20.0 pp** |
| claude / **MBPP** | 79.2% | 77.7% | 6.9% | 84.6% | −5.4 pp |
| claude / **DS-1000** | 23.9% | 17.8% | 20.6% | 38.3% | −14.4 pp |

Only one of the four (model × source) cells shows a clean positive tax —
**gpt-5.4 on DS-1000, +20 pp**. The negative-tax cells are explained almost
entirely by the false-negative items below.

## Tax by Ambiguity Type (pass@1, pp)

| Type | gpt-5.4 | claude-sonnet |
|---|---|---|
| coreferential | **+34.5** | −12.7 |
| scopal | **+27.5** | 0.0 |
| elliptical | +2.0 | −34.0 |
| collective_distributive | −8.4 | 0.0 |
| syntactic | −11.4 | −12.9 |

**coreferential** and **scopal** are the cleanest tax signals on gpt-5.4:
the model can satisfy the original prompt but loses ~25–35 pp once the
referent or scope is rewritten ambiguously. claude is more robust on these
types but shows large *negative* tax on elliptical, again driven by
perturbation-quality artifacts.

## Interpretation Bias

| Model / Source | interp_a_bias | choseA | choseB | neither |
|---|---|---|---|---|
| gpt-5.4 overall | 82.6% | 36.8% | 7.7% | 54.2% |
| gpt-5.4 / MBPP | 82.3% | 50.0% | 10.8% | — |
| gpt-5.4 / DS-1000 | 83.1% | 27.2% | 5.6% | — |
| claude overall | 74.3% | 42.9% | 14.8% | 42.3% |
| claude / MBPP | 91.8% | 77.7% | 6.9% | — |
| claude / **DS-1000** | **46.4%** | 17.8% | 20.6% | — |

Two findings worth keeping for the writeup:

1. **Both models prefer interpretation A** by default (the first reading the
   Stage-1 generator produced). Average bias ≈ 75–83% versus a 50% null.
2. **claude-sonnet is essentially unbiased on DS-1000** (46.4%, near 50/50)
   while strongly A-biased on MBPP (91.8%). gpt-5.4 stays ~82% on both.
   This is a model-specific, domain-conditional behavior that the benchmark
   surfaces.

## SA / EA / AC Distribution

### gpt-5.4 (310 samples)

| Slice | SA | EA | AC |
|---|---|---|---|
| overall | 91.3% | 8.7% | 0% |
| MBPP | 81.5% | 18.5% | 0% |
| DS-1000 | 98.3% | 1.7% | 0% |
| low risk | 89.1% | 10.9% | 0% |
| high risk | 97.5% | 2.5% | 0% |

### claude-sonnet (310 samples)

| Slice | SA | EA | AC |
|---|---|---|---|
| overall | 89.0% | 10.3% | 0.6% |
| MBPP | 88.5% | 11.5% | 0% |
| DS-1000 | 89.4% | 9.4% | 1.1% |
| coreferential | 70.9% | 25.5% | 3.6% |

**Headline finding**: under Mode B "lightweight permission", modern
instruction-tuned coding LLMs almost never ask clarifying questions. Both AC
events for claude were on the same item (AMBI/021, a coreferential DS-1000
task). High-risk items show *less* AC, not more — the opposite of what the
"safety-conscious LLM" narrative would predict.

EA (explicit assumption) rises on MBPP for gpt-5.4 (18.5%) — these are mostly
docstring-style "assuming X means Y" comments inside the generated code. On
DS-1000 the format leaves no room for prose, so EA drops to 1.7%.

## False-Negative Tax Items (Phase 1 quality issue)

Items where `baseline=0%` but `pass_either ≥ 0.8` — perturbation
*increased* solvability. These items dominate the negative-tax averages.

| task_id | source | type | base | either | A | B |
|---|---|---|---|---|---|---|
| AMBI/006 | mbpp | collective_distributive | 0.00 | **1.00** | 0.00 | 1.00 |
| AMBI/010 | mbpp | collective_distributive | 0.00 | **1.00** | 1.00 | 0.00 |
| AMBI/045 | mbpp | collective_distributive | 0.00 | **1.00** | 1.00 | 0.00 |
| AMBI/028 | mbpp | syntactic | 0.20 | 1.00 | 1.00 | 0.00 |
| AMBI/033 | ds1000 | syntactic | 0.00 | 0.80 | 0.00 | 0.80 |
| AMBI/062 | mbpp | elliptical | 0.00 | 0.80 | 0.00 | 0.80 |

Hypothesis: the Stage-1 SOTA generator, while writing the perturbed prompt,
inadvertently disambiguates by adding type hints, restating the goal, or
giving worked examples in the docstring. The original MBPP prompt is so
under-specified that even gpt-5.4 cannot solve it (baseline=0%), but the
"perturbed" version is *clearer* than clean.

This is a **Phase 1 data-quality issue**, not a model behavior. Two ways to
report it:

- (a) Flag and exclude these items from the headline tax number
- (b) Audit them, push the offending perturbations back through Stage 1 with
  stricter prompts (no hints, no examples)

For the writeup, the cleanest move is to report two numbers: tax including
all items, and tax excluding the ≤6 items where `baseline ≤ 0.2`. The
exclusion set is small enough to list explicitly.

## True-Positive Tax Items (real ambiguity signal)

Items where `baseline=1.00` but `pass_either ≤ 0.2` — perturbation broke
behavior cleanly. These are the strongest evidence that the benchmark
measures ambiguity tax.

| task_id | source | type | base | either |
|---|---|---|---|---|
| AMBI/024 | ds1000 | coreferential | 1.00 | 0.00 |
| AMBI/036 | ds1000 | scopal | 1.00 | 0.00 |
| AMBI/040 | ds1000 | scopal | 1.00 | 0.00 |
| AMBI/043 | ds1000 | collective_distributive | 1.00 | 0.00 |
| AMBI/049 | ds1000 | collective_distributive | 0.60 | 0.00 |

All five are DS-1000. Three are coreferential/scopal — the same types that
showed up as the cleanest tax signals in the per-type table above.

## Headline Findings (one-paragraph each)

1. **Ambiguity Tax exists and is type-conditional**: on the cleanest slice
   (DS-1000, gpt-5.4, coreferential + scopal items), pass@1 drops by 25–35 pp
   when prompts are made ambiguous. Aggregate tax is muddier because (a)
   different ambiguity types have different difficulty profiles and (b) some
   Phase-1 perturbations leak information.

2. **Interpretation bias is the second-order finding**: both models prefer the
   first interpretation the generator produced (75–83% bias). claude-sonnet
   on DS-1000 is the lone exception, near 50/50 — suggesting the bias is not
   uniform across domains.

3. **Active Clarification is essentially zero in deployment-realistic
   prompting**. Even with explicit permission, modern coding LLMs default to
   guessing (SA ~90%) rather than asking (AC <1%). This is a robust finding
   across two models and 620 generated samples.

4. **Phase 1 perturbation quality is the binding constraint** for tightening
   the headline tax number. The 6 "false-negative" items pull the average
   down by 5–10 pp on MBPP.

## Audit Followup (2026-05-05)

A full manual audit of all 62 items was conducted after this run. Findings
are recorded separately in [`benchmark_audit.md`](benchmark_audit.md). Headline:
about 16 of 62 items have issues serious enough to flag for exclusion
(information leakage in the perturbation, duplicated anchors that are also
dark, or one meta-prompting case). A "clean subset" of ~38–46 items is
identified. Resolution path (rebuild vs. flag-and-exclude vs. report-both) is
pending discussion.

## v2 Evaluation Results (2026-05-05)

After building `benchmark_v2.jsonl` (46 items: dropped 10 dups + 6 dark, fixed
7 leak/meta items), re-evaluated both models. Same config (n=5, T=0.8, Mode B).

### Headline numbers — v1 vs v2

| Metric | v1 (62) | v2 (46) | Δ |
|---|---|---|---|
| gpt-5.4 tax@1 | +4.8 | **+8.7** | +3.9 ↑ |
| gpt-5.4 tax@3 | +1.9 | **+17.8** | +15.9 ↑↑ |
| claude tax@1 | −10.6 | **−2.2** | +8.4 ↑ |
| claude tax@3 | −10.5 | −7.6 | +2.9 ↑ |
| gpt-5.4 A-bias | 82.6% | 77.7% | −4.9 |
| claude A-bias | 74.3% | 73.2% | −1.1 |
| gpt-5.4 SA / EA / AC | 91.3 / 8.7 / 0% | 93.9 / 6.1 / **0%** | — |
| claude SA / EA / AC | 89.0 / 10.3 / 0.6% | 89.6 / 10.4 / **0%** | — |

The v2 cleanup boosts gpt-5.4's tax@3 signal by **~10×** and flips claude's
MBPP cell from −5.4 pp to **+8.4 pp**. **AC ≈ 0** is now confirmed across two
benchmark versions and 920+ samples — the strongest standalone finding.

### Per-source v1 → v2

| Model / Source | v1 tax@1 | v2 tax@1 |
|---|---|---|
| gpt-5.4 / MBPP | −16.2 | **+6.3** |
| gpt-5.4 / DS-1000 | +20.0 | +10.4 |
| claude / MBPP | −5.4 | **+8.4** |
| claude / DS-1000 | −14.4 | −9.6 |

Three of four cells flipped to positive or shrank toward zero. The remaining
negative tax in claude-DS-1000 reflects claude's domain-specific willingness
to try interpretation B (A-bias = 53.6% on DS-1000 in v2), not perturbation
leakage.

### v2 fix verification — did the rewrites work?

| task_id | v1 (base→either) | v2 (base→either) | Verdict |
|---|---|---|---|
| AMBI/006 | 0% → 100% | 0% → 0% | ✅ leak closed |
| AMBI/010 | 0% → 100% | 0% → 0% | ✅ leak closed |
| AMBI/033 | 0% → 80% | 0% → 0% | ✅ leak closed |
| AMBI/034 | 0% → 60% | 0% → 20% | 🟡 reduced but not eliminated |
| AMBI/039 | 100% → 100% | 100% → 100% | ✅ meta-prompt removed without changing behavior |
| AMBI/051 | 40% → 80% | 20% → **100%** | ❌ regressed; the surviving "Clarification:" sentence still leaks |
| AMBI/062 | 0% → 80% | 0% → **100%** | ❌ anchor structurally broken — should drop, not fix |

5 of 7 fixes worked. AMBI/051 has a second leak ("Clarification:..." line)
not caught in the audit; AMBI/062's MBPP anchor is fundamentally broken
(canonical interpretation of "colon of a tuple" violates linguistic intuition).

## Pipeline v2 — First Application to HumanEval (2026-05-05)

After implementing the four Stage-1/1.5/2 reforms documented in
[`benchmark_generated_v2.md`](benchmark_generated_v2.md), the new pipeline ran
end-to-end on 127 high-feasibility HumanEval anchors. Funnel:

| Stage | Output | Pass rate |
|---|---|---|
| Stage 1 (3 generators × 127) | 381 generations | — |
| ↳ errors | 21 | |
| ↳ opt-outs (reform B) | **31 — all Gemini-3.1-pro** | gpt-5.4/claude never opted out |
| ↳ valid | 329 | |
| Stage 1.5 quality gate (reform D) | 29 / 329 | **9%** |
| Stage 2 bilateral naturalness (reform A) | 3 / 29 | **10%** of S1.5 survivors |
| Stage 4 exclusivity (after `check()` invocation patch) | **2 / 3** | **67%** |
| **Final benchmark items** (`benchmark_humaneval_v2.jsonl`) | **2** (AMBI/063 & /064, both scopal) | **1.6%** yield |

### Cross-model behavior on opt-out (B reform)

| Generator | Stage 1 ok | opt-outs | errors |
|---|---|---|---|
| gpt-5.4 | 126 | 0 | 1 |
| claude-sonnet | 127 | 0 | 0 |
| **gemini-3.1-pro** | 76 | **31** | 20 |

Gemini is the only generator that takes the opt-out option seriously when an
anchor doesn't admit natural ambiguity; the other two always force a B
reading even with the rule explicitly written into the prompt. This is a
finding worth reporting — either gpt-5.4 / claude-sonnet are genuinely worse
at recognizing "no natural ambiguity here," or they have a stronger
instruction-following bias toward producing the requested output even when a
refusal channel exists.

### Implications

The reforms achieve their stated goals:
- **Reform C (info conservation)** + **Reform A (bilateral naturalness)**
  together filter ~91% of v1-style perturbations as either leaky or
  contrived-B.
- **Reform B (opt-out)** provides an upstream channel that one of three
  generators uses (Gemini), shrinking the working set before downstream cost.
- **Reform D (quality gate)** correctly catches what Stage 2 then re-confirms
  with 5 judges — the single-judge gate is fast and cheap, the 5-judge gate
  is the strict second pass.

But the run also surfaces a Stage-3 limitation: HumanEval/37 (sort_even,
mutate-vs-return-new) is a *genuine* coreferential ambiguity that survived
Stages 1/1.5/2 but failed Stage 4 because the canonical test_a checks return
values only — it can't observe the mutation distinction. Strengthening
Stage-3 test generation to negate test_a when needed is the natural next
reform.

## Failure Mode Analysis on v2 (per-task forensic)

To understand *why* tasks fail (beyond aggregate tax), inspected stderr and
generated code for representative items in three buckets:

| Bucket | Count | Definition |
|---|---|---|
| LOW BASELINE | 13 / 46 | avg baseline pass < 40% — model fails the clean prompt |
| LOW pass_A | 9 / 46 | avg baseline ≥ 40% but pass_a < 30% — perturbation breaks A reading |
| LOW pass_B | **30 / 46** | pass_b < 10% — model never matches B |

### Five distinct failure modes

| Mode | Description | Example items | Fixable? |
|---|---|---|---|
| **F1 — Brittle DS-1000 schema** | Model logic correct, but output format mismatches (DataFrame vs Series, exact column order, exact index type). Test rejects valid solutions. | AMBI/049, /022, /036, /021 | Re-author tests to accept multiple equivalent formats |
| **F2 — Function wrapping mismatch** | Model wraps code in `def g(...)` but DS-1000 harness expects code inline OR expects `X` predefined. NameError on `features` / `X`. | AMBI/017, /052 | Update harness or normalization to inject required globals |
| **F3 — Genuine ambiguity signal (good)** | Model picks interp B because perturbation actually shifted reading; test_a fails, test_b passes. | AMBI/036, /043, /049, /003, /056 | Don't fix — this *is* the measurement |
| **F4 — Contrived B (Stage-1 limitation)** | Interp B is something no programmer would write. pass_b ≈ 0 by construction; "ambiguity" exists only on paper. | AMBI/007, /008, /039 (and ~17 more) | Stage-1 prompt redesign — see below |
| **F5 — Vague clean prompt** | Original anchor under-specifies (e.g., doesn't say "nested tuple"). Even baseline fails. | AMBI/010, /021 | Anchor selection / curation |

### Distribution of failure modes per bucket (estimated from samples)

| Mode | LOW BASE | LOW pa | LOW pb |
|---|---|---|---|
| F1 (schema) | ~6 | ~4 | ~5 |
| F2 (wrapping) | ~2 | 0 | 0 |
| F3 (real signal) | 0 | ~3 | ~3 |
| F4 (contrived B) | ~1 | ~1 | **~20** |
| F5 (vague clean) | ~4 | ~1 | ~2 |

### Key takeaways

1. **The "30 of 46 items have pass_b ≈ 0" finding is dominated by F4** —
   Stage 1 invented unnatural B readings for prompts that only have one
   natural reading. This is a *methodological* limitation, not a model
   behavior we want to report.

2. **F3 is the real signal**: only ~5–7 items show a clean ambiguity tax
   pattern (model can solve clean, fails A, passes B). These items are
   the strongest evidence the benchmark measures what it claims to. They are
   disproportionately DS-1000 + scopal/coll_dist.

3. **F1 + F2 (DS-1000 plumbing)** account for ~11 items of noise that
   *aren't* about ambiguity at all. Should be acknowledged as test-quality
   debt, not blamed on the model.

4. **Implication for Stage 1 redesign**: the binding constraint on benchmark
   quality is whether Stage 1 can produce two interpretations that are *both
   linguistically natural*. When it can't, item is wasted (F4). A stricter
   Stage-1 gate that rejects forced B's would shrink the benchmark but raise
   the per-item signal density substantially.

## Open Questions

- How much of the negative tax is fixed by re-running Stage 1 with a stricter
  prompt that forbids worked examples in the perturbed prompt?
- Does AC rise meaningfully under a Mode-B prompt that asks "is the request
  ambiguous?" upfront? (Would no longer be naturalistic, but useful as an
  upper-bound measurement.)
- Adding a third model (e.g. Gemini, DeepSeek) — does the interp_a_bias
  pattern hold, or is it generator-specific?

## Artifact Pointers

- Run timestamps: `gpt-5.4` 20260505_115459, `claude-sonnet` 20260505_115507
- Per-item records: `data/results/perturbed_*_20260505_*.jsonl`
- Aggregate metrics: `data/results/metrics_summary.{csv,json}`
- Plots: `data/results/analysis_<model>_20260505_*/plot_*.png`