# Benchmark Generation Pipeline v2 — Design

*Last updated: 2026-05-06* — design 2026-05-05; HumanEval first run captured below; the 5-model evaluation that consumes the v2-built benchmark is documented in [`findings.md`](findings.md).

This document describes the v2 redesign of the Phase-1 benchmark generation
pipeline. The motivation is failure-mode analysis on the v1 benchmark
([findings.md](findings.md), [benchmark_audit.md](benchmark_audit.md)) which
identified four root causes for low signal-to-noise:

| Root cause | v1 effect | Items affected |
|---|---|---|
| Cat 1 — information leakage in perturbation | negative tax | 9 / 62 |
| Cat 4 — contrived interpretation B | pass_b ≈ 0 | ~22 / 62 |
| Cat 5 — meta-prompt announcing ambiguity | contaminates AC | 1 / 62 |
| Phase-1 has no opt-out for "no real ambiguity" | drives Cat 4 | systemic |

The v2 pipeline introduces **four reforms (A/B/C/D)** that target these causes
upstream rather than after the fact. The downstream stages (Stage 3
test-generation, Stage 4 exclusivity gate, Phase-2 evaluation, Phase-3
classification, Phase-4 analysis) are unchanged.

## Reform Summary

| | Reform | Stage | What changes |
|---|---|---|---|
| **A** | Bilateral naturalness gate | 2 | Replace "which is more natural" voting with two independent yes/no judgments per judge: "is A natural?" + "is B natural?". Pass requires both ≥3/5. |
| **B** | Stage-1 opt-out | 1 | Generation prompt allows returning `null` perturbed_prompt with a reason when the anchor admits only one natural reading. Forced ambiguity is rejected upstream. |
| **C** | Information conservation rules | 1 | Generation prompt adds a hard constraint list: no added type signatures, no added examples, no clarifying parentheticals, no simplification of verbose originals. |
| **D** | Stage 1.5 quality gate | new | Independent LLM judge inspects each generated perturbation against three flags (leakage, B-naturalness, distinguishability). Reject if any flag fails. |

## Pipeline Architecture (v1 → v2)

```
v1                                   v2
──                                   ──
Anchor selection (unchanged)         Anchor selection (unchanged)
        |                                    |
Stage 1: 3 SOTA generators           Stage 1: 3 SOTA generators
  - C: information conservation        - + reform C: info-conservation rules
  - B: opt-out option                  - + reform B: opt-out option
        |                                    |
                                     ★ Stage 1.5: quality gate (NEW, reform D)
                                       1 cheap judge × 3 binary flags
                                       reject if any fails
                                            |
Stage 2: 5 judges, "which is        Stage 2: 5 judges, "is A natural?"
  more natural?" → entropy           + "is B natural?" (reform A)
  ≥0.72 passes                       both ≥3/5 passes
        |                                    |
Stage 3 (test_b generation)          Stage 3 (test_b generation, unchanged)
        |                                    |
Stage 4 (sandbox exclusivity)        Stage 4 (sandbox exclusivity, unchanged)
        |                                    |
benchmark.jsonl                      benchmark_humaneval_v2.jsonl
```

## Reform A — Bilateral Naturalness Gate (Stage 2)

**Problem**: v1 entropy gate measures *judge disagreement* on the natural reading.
A 3-2 vote split can mean either (a) genuine ambiguity, or (b) two judges
misread a contrived B. The gate cannot distinguish.

**Fix**: each judge answers two independent yes/no questions:

```
Q1. A typical Python developer reads this prompt cold and writes code that
    realizes Interpretation A ({interpretation_a}). Would they naturally
    do this? (yes/no)

Q2. Same setup, but Interpretation B ({interpretation_b}). Would they
    naturally do this? (yes/no)

Both answers are independent — both can be yes, both can be no.
```

**Pass criterion**: `yes_a ≥ 3 AND yes_b ≥ 3` (both interpretations endorsed
as natural by majority of judges, independently).

**Why this is stronger**: under v1, an item where 4 judges all see only A but
1 misreads as B will pass entropy ≥ 0.72 (4-1 split). Under v2, that same
item gets `yes_a=5, yes_b=1` → fails (yes_b < 3). Conversely, an item where
all 5 judges see both readings as plausible passes both v1 and v2.

The Shannon entropy is still computed and stored (for diagnostic use) but is
no longer the pass criterion.

## Reform B — Stage-1 Opt-Out

**Problem**: Stage-1 generators are required to produce two interpretations
for every anchor. When an anchor has only one natural reading, the LLM is
forced to invent a contrived B (Cat-4 failure mode).

**Fix**: the generation prompt explicitly grants permission to refuse:

```
If the original prompt has only ONE natural interpretation (no productive
linguistic ambiguity is possible without violating these rules), respond:

  {"perturbed_prompt": null, "reason": "<one sentence explaining why no
   natural ambiguity exists for this prompt>"}

This is preferable to forcing a contrived second reading. A rejected anchor
saves all downstream cost; a contrived ambiguity wastes 5 judge calls,
1 sandbox run, and produces a benchmark item with pass_b ≈ 0.
```

`Stage1Result` accommodates `perturbed_prompt: None` and downstream stages
skip such generations early.

## Reform C — Information Conservation

**Problem**: Stage-1 LLMs add information not in the clean prompt — function
signatures with named parameters, worked examples, clarifying
parentheticals, restructured prose. This makes the perturbed prompt
*easier* than clean (negative tax).

**Fix**: hard rules block the most common leak patterns:

```
INFORMATION CONSERVATION (critical):
The perturbed prompt MUST NOT contain information absent from the original:
  1. NO new type signatures, parameter names, or argument counts.
     (e.g. clean says "more than one list" → do NOT write `(l1, l2, l3)`)
  2. NO worked examples (input/output pairs) not in the original.
  3. NO clarifying parentheticals or assumptions like "(assume we keep the
     first)".
  4. NO restating the goal in clearer language than the original used.
  5. If the original is verbose or confusing, the perturbed version MUST be
     equally verbose/confusing — only the ambiguity-injecting words may
     change.
  6. NO meta-language about the prompt itself ("interpret as you see fit",
     "could be read multiple ways", etc.).
```

These rules complement the existing rule about handling doctest examples.
Together they should catch v1's Cat-1 and Cat-5 failures.

## Reform D — Stage 1.5 Quality Gate (new stage)

**Problem**: even with reforms A/B/C, the generation step can still produce
flawed perturbations (LLMs don't always follow rules). v1's two existing
gates (Stage 2 entropy, Stage 4 exclusivity) check different properties:
entropy checks judge disagreement; exclusivity checks 2×2 sandbox behavior.
Neither catches "perturbed prompt leaks information" or "interp B is code
no one would write".

**Fix**: insert a new gate between Stage 1 and Stage 2. One cheap judge
(claude-haiku) reads `(clean_prompt, perturbed_prompt, interpretation_a,
interpretation_b)` and answers three binary questions:

```
Q1 (LEAKAGE): Does the perturbed prompt contain information NOT present in
    the clean prompt? Examples of leakage: a function signature exposing
    parameter count when clean said "multiple"; a worked example; a
    clarifying assumption; restated goal in clearer words.
    answer: yes / no

Q2 (B-NATURALNESS): Would a senior Python programmer, reading the perturbed
    prompt cold, plausibly write code matching interpretation B? (Not "is
    interpretation B grammatically valid" — but "would anyone actually
    produce this code?")
    answer: yes / no

Q3 (DISTINGUISHABILITY): Would interpretation A's code and interpretation
    B's code produce DIFFERENT outputs on a typical input? (Not just on
    edge cases.)
    answer: yes / no
```

**Pass criterion**: Q1 = "no" AND Q2 = "yes" AND Q3 = "yes". Any other
combination rejects the generation.

This gate is the v2 pipeline's signature contribution — it directly
addresses every named v1 failure mode that originates upstream of Stage 2.

**Cost**: one extra cheap-model call per Stage-1 generation (3 generations
× ~150 candidate anchors = ~450 calls per pipeline run, ~$0.50 with
claude-haiku).

## Config Schema Changes

`config/pipeline.yaml`:

```yaml
# NEW section between perturbation and entropy_gate
quality_gate:
  judge_model: claude-haiku        # one judge is enough (we're checking objective things)
  temperature: 0.0
  max_tokens: 256
  max_workers: 12
  output_dir: data/intermediate/quality_gate

# entropy_gate: unchanged keys; behavior changes per reform A
entropy_gate:
  judge_models: [...]              # same 5 judges
  judges_per_task: 5
  min_yes_per_side: 3              # NEW: was min_entropy: 0.72
  temperature: 0.0
  max_tokens: 256                  # was 128 — bilateral response is longer
  max_workers: 12
  output_dir: data/intermediate/entropy_gate
```

`config/prompts.yaml`:

- `perturbation.task`: extended with reforms B + C
- `entropy_gate.task`: rewritten for bilateral voting (reform A)
- `quality_gate` (new section): system + task prompts for Stage 1.5

## Output Schema Changes

**Stage1Result.generations[i]**: gains `reason: str | None` field. When
`perturbed_prompt is None`, `reason` explains the opt-out.

**New Stage1_5Result** (analogous to Stage2Result):
```python
@dataclass
class QualityGateResult:
    generator_model: str
    perturbed_prompt: str
    interpretation_a: str
    interpretation_b: str
    leakage: bool          # Q1 result
    b_natural: bool        # Q2 result
    distinguishable: bool  # Q3 result
    judge_reasoning: str
    passed: bool           # !leakage and b_natural and distinguishable
```

**EntropyResult** (Stage 2): keeps the name and most fields. Adds:
```python
yes_a: int            # NEW: judges saying A is natural
yes_b: int            # NEW: judges saying B is natural
# count_a, count_b: kept as aliases for yes_a, yes_b for back-compat
# entropy: still computed for diagnostics
# passed: now (yes_a >= 3 and yes_b >= 3)
```

## Migration Plan

The v2 pipeline is committed to in-place: existing modules
(`stage1_perturbation.py`, `stage2_entropy_gate.py`) are updated, a new
`stage1_5_quality_gate.py` is added, and the orchestrator
(`run_perturbation.py`) is extended to chain the new stage.

Reproducing v1 results from raw anchors will no longer work. v1 outputs
already on disk (`benchmark.jsonl`, `benchmark_v2.jsonl` — the *evaluation*
v2, derived from the v1 generation pipeline) are preserved untouched.

## First Application: HumanEval

HumanEval (164 tasks, 163 with feasibility ≥ 0.6) was excluded from v1
because doctest examples in the prompts disambiguated all generated
perturbations (0% Stage 4 pass rate). With reforms A–D, the explicit
example-handling rule + the quality gate should pass items where Stage 1
correctly removes or generalizes the examples.

**Selection plan for the first run**:
- 164 HumanEval anchors, all 5 ambiguity types
- min_feasibility ≥ 0.6 (current default)
- Both risk levels (HumanEval is 162 low / 2 high — small but report what we get)
- Output goes to `data/benchmark/benchmark_humaneval_v2.jsonl`

**Success criterion**: Stage 4 pass rate > 30% (vs. 0% on v1 pipeline).
Anything above this proves the reforms unlocked HumanEval as a benchmark
source. The expected ambiguity-type distribution (HumanEval anchors are
80 syntactic / 49 coreferential / others) suggests the resulting items will
complement the existing MBPP+DS-1000 benchmark in syntactic coverage.

## First Run Results (2026-05-05)

Pipeline executed end-to-end on 127 high-feasibility HumanEval anchors
(default selection: `min_feasibility ≥ 2.0`, `risk_level = low`).

### Funnel

| Stage | Output | Pass rate |
|---|---|---|
| Stage 1 (3 generators × 127) | 381 candidate generations | — |
| ↳ Stage 1 errors | 21 (mostly Gemini truncated JSON) | |
| ↳ Stage 1 opt-outs (reform B) | **31 (all from Gemini-3.1-pro)** | gpt-5.4 / claude never opted out |
| ↳ Stage 1 valid | 329 | |
| Stage 1.5 quality gate (reform D) | 29 / 329 passed | **9%** |
| ↳ failure: leak only | 62 | |
| ↳ failure: contrived-B only | 100 | |
| ↳ failure: leak + contrived-B | 121 | |
| Stage 2 bilateral naturalness (reform A) | 3 / 29 passed | **10%** of S1.5 survivors |
| Stage 4 exclusivity gate | **2 / 3 passed** | **67%** of S2 survivors |
| **Final benchmark items** | **2** | **1.6% yield** of Stage-1 anchors |

### Surviving items (`data/benchmark/benchmark_humaneval_v2.jsonl`)

| task_id | anchor | type | interpretations |
|---|---|---|---|
| AMBI/063 | HumanEval/39 (`prime_fib`) | scopal | A: n-th integer that is *both* prime *and* fibonacci. B: tuple (n-th prime, n-th fib). |
| AMBI/064 | HumanEval/138 (`is_equal_to_sum_even`) | scopal | A: n is sum of *any* 4 positive evens. B: n is a *single* even multiple repeated 4 times (n divisible by 8). |

Both surviving generations came from `gemini-3.1-pro` and have judges' bilateral votes `yes_a=4, yes_b=3` (4 of 5 judges deem A natural, 3 of 5 deem B natural — passes the `min_yes_per_side=3` threshold).

### Diagnostic observations

1. **Reforms B and D are doing the heavy lifting**:
   - Gemini's 31 opt-outs (24% of its 127 attempts) are exactly the contrived-B
     cases v1 would have admitted. gpt-5.4 and claude-sonnet, despite identical
     prompt instructions, never used the opt-out — they always force a B reading.
     This is itself an interesting cross-model behavior.
   - Stage 1.5 rejects 91% of generations. Of those, the dominant failure is
     `contrived-B` (alone or combined with leak), confirming the v1 audit's
     finding that Cat-4 was the largest source of pass_b ≈ 0 noise.

2. **Stage 2 (bilateral) catches what Stage 1.5 misses**: 26 of 29 Stage-1.5
   survivors fail Stage 2. Most failures are `(yes_a=5, yes_b=0)` or
   `(yes_a=0, yes_b=5)` — the single Stage-1.5 judge sometimes calls B natural,
   but the 5-judge majority disagrees. Belt-and-suspenders works.

3. **Stage 4 catches a different class of issue**: HumanEval/37 (sort_even,
   coreferential, `mutate vs. return new`) passed Stages 1/1.5/2 but failed
   exclusivity because test_a checks only the return value (which is identical
   for both A and B). The reading is genuinely ambiguous, but the test
   harness can't observe the distinction. **This is a Stage-3 test-design
   limitation, not a v2 reform issue.** Possible follow-up: prompt Stage 3
   to also negate test_a for known mutating cases (or generate a stricter
   test_a that observes side effects).

4. **Yield is low but quality is high**: 2 items out of 127 anchors is a 1.6%
   yield, but each survivor is a clean ambiguity that passed five independent
   gates (opt-out + quality + bilateral × 5 + 2×2 sandbox exclusivity).
   For comparison, v1 admitted ~36 HumanEval items at Stage 1 and 0 at Stage 4
   (without the test-invocation patch this run also caught).

### Stage 4 patch — HumanEval `check()` invocation

Discovered during this run: HumanEval `test_code` defines `def check(candidate)` but never calls it, so the v1 sandbox would silently exit 0 with no assertion executed (every 2×2 cell trivially "passing"). The patch in `stage4_exclusivity_gate.py` appends `check({entry_point})` when `task.source == "humaneval"` and the test text doesn't already contain a top-level `check(` call. With this patch, the 3 Stage-2 survivors run their assertions and 2 cleanly distinguish A from B.

### What this means for the benchmark

- **HumanEval is still a thin source** even with v2 reforms — its prompts are well-specified by design, so few admit natural ambiguity. The 2 surviving items are valuable additions but won't supplant MBPP/DS-1000 as the bulk of the benchmark.
- The reforms work as advertised: where v1 would have admitted dozens of contrived-B / leaky items from HumanEval, v2 admits only the cleanly ambiguous ones.
- For broader coverage, the Stage 3 weakness (tests not always observing the A/B distinction) and the Stage 1 cross-model behavior (gpt/claude never opting out) are the two productive follow-up directions.