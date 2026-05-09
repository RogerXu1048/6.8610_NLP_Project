# Benchmark Quality Audit

*Last updated: 2026-05-06* — audit done 2026-05-05; v2 build documented below; final benchmark in use is `benchmark_v2_full.jsonl` (48 items, includes 2 HumanEval items added via the v2 generation pipeline).

A manual review of all 62 items in `data/benchmark/benchmark.jsonl`, conducted
after the first full evaluation run on gpt-5.4 + claude-sonnet exposed
unexpected patterns (notably negative Ambiguity Tax on MBPP). The audit
compared `prompt`, `perturbed_prompt`, `interpretation_a/b`, and `test_a/b`
side-by-side, cross-referenced against per-item baseline and perturbed pass
rates from both models.

The findings below are descriptive — no benchmark items have been
modified, removed, or re-tagged yet.

## Summary

| Category | Count | Severity | Effect on metrics |
|---|---|---|---|
| 1. Information leakage in perturbation | 9 | 🔴 high | Inflates pass_either; produces negative Ambiguity Tax |
| 2. Duplicated anchors | 9 anchor groups (~17 items) | 🟠 medium | Inflates n_items; biases per-type averages |
| 3. Dark items (no model passes either condition) | 9 | 🟠 medium | Adds noise, no signal |
| 4. Forced/contrived interpretation B | 5 | 🟡 low | Drives pass_b ≈ 0 for that item |
| 5. Meta-prompting (perturbation announces ambiguity) | 1 | 🔴 high | Contaminates SA/EA/AC measurement |

Some items belong to more than one category. The unique union covers about
**25–30 of 62 items** (≈45%).

## Category 1 — Information Leakage in Perturbation

The Stage-1 SOTA generator (rewriting clean → perturbed) sometimes adds
information not present in the original prompt: a function signature exposing
arg counts, a worked-out docstring, a clarifying parenthetical, or a simply
better-written version of an awkward original. These items show
`baseline ≪ pass_either` (perturbation *helps*).

| task_id | source | type | baseline (gpt/claude) | either (gpt/claude) | What leaks |
|---|---|---|---|---|---|
| AMBI/006 | mbpp | coll_dist | 0% / 80% | 100% / 100% | `def convert_list_dictionary(l1, l2, l3)` reveals exactly 3 args; clean only says "more than one list" |
| AMBI/010 | mbpp | coll_dist | 0% / 0% | 100% / 100% | "inner tuple elements at each index" is much clearer than "index wise multiplication" |
| AMBI/028 | mbpp | syntactic | 20% / 100% | 100% / 100% | Perturbed docstring spells out "return the total count of elements from the list that appear in the tuple" — basically the answer |
| AMBI/033 | ds1000 | syntactic | 0% / 40% | 80% / 100% | Clean is verbose and confusing; perturbed simplifies it (gpt baseline=0% confirms clean is hard to parse) |
| AMBI/034 | ds1000 | syntactic | 0% / 100% | 60% / 100% | Same pattern as AMBI/033 — perturbed is more digestible than the long original |
| AMBI/045 | mbpp | coll_dist | 0% / 100% | 100% / 100% | "tuples of tuples" appears in perturbed but not in clean ("two tuples") |
| AMBI/046 | mbpp | coll_dist | 0% / 60% | 60% / 100% | Same anchor as AMBI/006; same arg-count leak |
| AMBI/051 | ds1000 | coll_dist | 40% / 20% | 80% / 100% | Adds explicit "(Assume we want to keep the first row of any overlapping set)" — directly resolves one of the two interpretations |
| AMBI/062 | mbpp | elliptical | 0% / 0% | 80% / 80% | Original "colon of a tuple" is incoherent; perturbed "slice from m to n" reads like a normal slicing problem |

**Why the gates missed this**: the Stage 4 exclusivity gate (Docker 2×2)
verifies internal consistency among `ref_a/b` and `test_a/b`. The Stage 2
entropy gate verifies that *judges* are split on which interpretation the
perturbed prompt conveys. Neither gate compares the perturbed prompt to the
clean prompt for information conservation.

## Category 2 — Duplicated Anchors

`run_anchor_selection.py` did not deduplicate at the anchor level. Several
MBPP/DS-1000 source tasks made it into the benchmark twice or three times,
each with a different Stage-1 perturbation. The "62 verified items" count
overstates independent task coverage.

| Source anchor | Items | Notes |
|---|---|---|
| MBPP `max_sum_increasing_subseq` | AMBI/005, AMBI/018 | Same task, different pronoun-resolution perturbations |
| MBPP `count_Occurrence` | AMBI/011, AMBI/028, AMBI/029 | One of the count_Occurrence perturbations is a leak (AMBI/028) |
| MBPP `maximize_elements` | AMBI/003, AMBI/045 | AMBI/045 is a leak |
| MBPP `convert_list_dictionary` | AMBI/006, AMBI/046 | Both are leaks |
| DS-1000 frequent-value-per-row | AMBI/027, AMBI/037 | Both are dark (0/0 across models) |
| DS-1000 unique ID per `a` | AMBI/040, AMBI/041 | Both have clean signal |
| DS-1000 reverse + concatenate lists | AMBI/043, AMBI/044 | Both have clean signal |
| DS-1000 remove overlapping rows | AMBI/050, AMBI/051 | AMBI/050 dark, AMBI/051 leaks |
| DS-1000 row/column z-score | AMBI/053, AMBI/055 | Both dark |

Effect on metrics: per-ambiguity-type averages over-weight whatever the
duplicated anchor's pattern is. For example, the three `count_Occurrence`
items contribute disproportionately to the "syntactic" category.

## Category 3 — Dark Items (no signal)

Both models pass ≤20% on both clean and perturbed conditions. These items
carry no Ambiguity Tax signal and only add variance.

| task_id | source | type | Why dark |
|---|---|---|---|
| AMBI/012 | mbpp | syntactic | "max product of an increasing subsequence" — the canonical interpretation is unusual; neither model finds it |
| AMBI/027 | ds1000 | syntactic | Frequent-value-per-row with weird output schema; tests too brittle |
| AMBI/032 | mbpp | syntactic | Re-arrange-array test cases inconsistent (e.g. expected output `[-1,-3,-7,4,5,6,2,8,9]` puts 2 after 6 in original-order positives) — possible canonical bug |
| AMBI/037 | ds1000 | scopal | Same anchor as AMBI/027 |
| AMBI/050 | ds1000 | coll_dist | Overlapping-rows date filter — too underspecified to solve cleanly |
| AMBI/053 | ds1000 | elliptical | Z-score with two-level row index output — output format brittle |
| AMBI/055 | ds1000 | elliptical | Same anchor as AMBI/053 |
| AMBI/059 | ds1000 | elliptical | value_counts string-format output — brittle |
| AMBI/061 | ds1000 | elliptical | Shift NaN to left — both clean & perturbed prompts unclear about target |

Note: an item being dark is not always Stage-1's fault. AMBI/032's tests are
internally inconsistent (a Phase-1 issue earlier, not the perturbation).
AMBI/053/055/059/061 are inherently brittle DS-1000 problems with
output-format-sensitive testing.

## Category 4 — Forced / Contrived Interpretation B

Stage 1 was prompted to invent two readings. When the prompt is genuinely
two-way ambiguous, this works. When only one reading is natural, Stage 1 has
to *invent* the second — and the invented one is something no real model
would produce. These items show pass_b ≈ 0 across all conditions.

| task_id | What's contrived |
|---|---|
| AMBI/008 | Interp B reads "sublists" as "all contiguous slices of a flat list" — nobody does this |
| AMBI/054 | Interp B returns a tuple of triples (AND_result, ele1, ele2) — unnatural output shape for "elementwise AND" |
| AMBI/057 | Interp B says "ignore any leftover row/column AND the last complete patch" — the second clause is invented |
| AMBI/058 | Interp B reads ellipsis "formatted as row and column" as "tuple of two arrays" — possible but a stretch |
| AMBI/062 | Interp A is the contrived one (original MBPP canonical "append n to nested list at index m" is bizarre); the perturbed B "slice from m to n" is the natural reading |

These don't bias the headline tax much (since pass_b is just ~0 either way),
but they inflate `interp_a_bias` artificially — there's no real B option to
choose.

## Category 5 — Meta-Prompting

Stage 1 occasionally generated a perturbed prompt that *announces* the
ambiguity, defeating the Mode B "lightweight permission" experimental design.

| task_id | The leak |
|---|---|
| AMBI/039 | Perturbed docstring contains: *"interpret this as you see fit based on what 'even' modifies in context."* — explicitly tells the model the prompt is ambiguous, biasing toward EA/AC. |

Only one item, but a serious one — it contaminates the SA/EA/AC measurement
for that sample.

## Effect on the First Full Run (2026-05-05)

Most of the negative Ambiguity Tax is concentrated in the Category-1 items:

- gpt-5.4 MBPP tax = −16.2 pp; remove the 4 MBPP leaks (AMBI/006/010/028/045/046, plus their duplicates) and the average shifts upward by an estimated 8–10 pp.
- gpt-5.4 elliptical tax = +2.0 pp on a per-type average that includes AMBI/062's leak (−100 pp delta).
- claude-sonnet's headline −10.6 pp tax is partially driven by AMBI/033 (+100 pp delta) on the DS-1000 syntactic side.

A back-of-envelope re-aggregation excluding the 9 Category-1 + 1 Category-5
items (10 of 62) would lift gpt-5.4's overall pass@1 tax from +4.8 pp to
roughly +12–15 pp — closer to the per-type signal seen on coreferential
(+34.5 pp) and scopal (+27.5 pp).

## v2 Build Result (2026-05-05)

`data/benchmark/benchmark_v2.jsonl` was built from v1 by `scripts/build_benchmark_v2.py`.

**Operations applied**:
- Dropped **10 items** (one duplicate kept per anchor group):
  AMBI/018, /027, /028, /029, /041, /044, /045, /046, /050, /053
- Dropped **6 dark items** (both models score ≤20% on baseline AND pass_either):
  AMBI/012, /032, /037, /055, /059, /061
- Fixed **7 items** (rewrote `perturbed_prompt` only — tests, interpretations, ref solutions untouched):
  AMBI/006, /010, /033, /034, /039, /051, /062

**Result**: 46 items, retains all 5 ambiguity types and both risk levels.

| | v1 | v2 |
|---|---|---|
| Total | 62 | 46 |
| MBPP | 26 | 19 |
| DS-1000 | 36 | 27 |
| coll_dist | 19 | 15 |
| syntactic | 14 | 9 |
| coreferential | 11 | 10 |
| elliptical | 10 | 6 |
| scopal | 8 | 6 |
| low risk | 46 | 34 |
| high risk | 16 | 12 |

**Coverage caveat**: DS-1000 elliptical drops from 8 → 4, MBPP syntactic from 8 → 4. Plan is to
backfill these slots from HumanEval items selected for ambiguity potential in a follow-up Stage 1 run.

### Fix Rationale (per item)

| task_id | Issue | Fix |
|---|---|---|
| AMBI/006 | `def convert_list_dictionary(l1, l2, l3)` revealed exactly 3 args | Changed signature to `def convert_list_dictionary(*lists)` |
| AMBI/010 | docstring "inner tuple elements at each index" disambiguated structure | Reverted docstring to clean's wording: "Perform index-wise multiplication of tuple elements in the given two tuples." |
| AMBI/033 | perturbation simplified clean and changed thresholds (artificial ambiguity) | Replaced with a real syntactic binding ambiguity: "with thresholds 3 and 2" — does (3,2) bind {Qu1:3, Qu2/Qu3:2} or {Qu1/Qu2:3, Qu3:2}? |
| AMBI/034 | full clean context replaced by a single sentence | Restored the long original context verbatim; injected only the ambiguous final sentence ("predictions on the combined features in a DataFrame") so the PP attachment ambiguity is the *only* edit |
| AMBI/039 | meta-prompt: "interpret this as you see fit based on what 'even' modifies" | Removed the second docstring paragraph; "even binomial coefficients" remains genuinely scopally ambiguous |
| AMBI/051 | "(Assume we want to keep the first row of any overlapping set)" disambiguated | Removed the parenthetical; greedy vs all-pairs overlap rule remains undetermined |
| AMBI/062 | "Get the colon of a tuple at m with n" — "at m with n" steers toward slice | Reverted to clean's "Get a colon of a tuple"; both the canonical (insert-into-inner-list) and slice readings remain plausible |

### Dedup Selection (per anchor group)

| Group (anchor) | Kept | Dropped | Reason |
|---|---|---|---|
| max_sum_increasing_subseq | AMBI/005 | AMBI/018 | 005's "it" pronoun ambiguity is cleaner |
| count_Occurrence | AMBI/011 | AMBI/028, AMBI/029 | 028 had Cat-1 leak; 011 / 029 nearly identical, 011 simpler |
| maximize_elements | AMBI/003 | AMBI/045 | 045 had Cat-1 leak |
| convert_list_dictionary | AMBI/006 | AMBI/046 | both leaked; kept 006 and fixed |
| frequent value | AMBI/037 | AMBI/027 | both dark, but 037 has the genuine scopal reading |
| unique ID per `a` | AMBI/040 | AMBI/041 | 040 has stronger tax signal (100→0 vs 100→60) |
| reverse + concat | AMBI/043 | AMBI/044 | 043 has cleanest tax signal (100→0 on both models) |
| overlapping rows | AMBI/051 | AMBI/050 | 050 is dark; 051 becomes useful after fix |
| z-score | AMBI/055 | AMBI/053 | both dark; row z-score slightly cleaner |

### Dark Item Exclusion Criterion

Dark = `max(baseline_pass@1) ≤ 0.20` AND `max(pass_either@1) ≤ 0.20` across both
evaluated models (gpt-5.4 and claude-sonnet-4-6, n=5, T=0.8). Under this floor
condition the Ambiguity Tax metric (`baseline − pass_either`) is undefined or
near-zero by construction and contributes no signal. The criterion is
pre-registered (defined before the v2 evaluation run) and applied symmetrically
to all candidate items.

### What v2 Does NOT Change

- Cat-4 contrived-B items remain in v2: AMBI/008, /054, /057, /058, /062. These have low pass_b but still test the model's reading of interpretation A.
- All `test_a`, `test_b`, `ref_solution_a/b`, `interpretation_a/b`, `quality_gate_*` fields are unchanged — the fixes only touch `perturbed_prompt`.

## What This Audit Does NOT Address

- We have not re-run Stage 1 to verify that a stricter prompt (no examples,
  no type hints, no clarifying parentheticals) actually fixes the leakage.
- We have not checked whether the duplicated anchors arose because Stage 4
  accepted multiple Stage-1 outputs from the same source task, or because
  anchor selection itself returned the same task multiple times. Either way
  the dedupe should happen at anchor selection time.
- We have not yet decided the resolution path (rebuild vs. flag-and-exclude
  vs. report two numbers).

## Suggested Next Decisions (for discussion)

1. **Rebuild path**: re-run Stage 1 with `forbid_examples=true,
   forbid_type_hints=true, paraphrase_only=true` for the leaking items, push
   through Stages 2/3/4 again. Cost: a few API dollars + half a day.
2. **Flag path**: add a `quality_flag` field to each `BenchmarkItem` listing
   the audit categories that apply. Analysis scripts grow a `--exclude-flags`
   option. Cost: low; doesn't touch upstream pipeline.
3. **Report-both path**: keep `benchmark.jsonl` untouched; in the paper,
   report tax over (a) all 62 items and (b) a clean subset of ~30, with
   the audit categories explained as known limitations. Cost: only a paper
   section.

The cleanest move for now is **Flag path**: it keeps `benchmark.jsonl` as the
single source of truth, makes the analysis explicit about what's being
included, and leaves Rebuild as an option for later if reviewers push back.

## Per-Item Verdict Table

For reference. ✓ = clean ambiguity signal, ⚠ = some issue, ✗ = exclude.

| task_id | source | type | verdict | notes |
|---|---|---|---|---|
| AMBI/001 | mbpp | coll_dist | ✓ | clean |
| AMBI/002 | mbpp | syntactic | ✓ | clean |
| AMBI/003 | mbpp | coll_dist | ⚠ | duplicate anchor (AMBI/045) |
| AMBI/004 | mbpp | coll_dist | ✓ | clean |
| AMBI/005 | mbpp | coreferential | ⚠ | duplicate anchor (AMBI/018) |
| AMBI/006 | mbpp | coll_dist | ✗ | Cat 1 leak + duplicate (AMBI/046) |
| AMBI/007 | mbpp | coll_dist | ✓ | clean |
| AMBI/008 | mbpp | coll_dist | ⚠ | Cat 4 contrived B |
| AMBI/009 | mbpp | syntactic | ✓ | clean |
| AMBI/010 | mbpp | coll_dist | ✗ | Cat 1 leak |
| AMBI/011 | mbpp | syntactic | ⚠ | duplicate anchor (AMBI/028, AMBI/029) |
| AMBI/012 | mbpp | syntactic | ✗ | Cat 3 dark |
| AMBI/013 | mbpp | coll_dist | ✓ | clean |
| AMBI/014 | mbpp | coll_dist | ✓ | clean |
| AMBI/015 | ds1000 | coreferential | ✓ | clean |
| AMBI/016 | ds1000 | coreferential | ✓ | clean |
| AMBI/017 | ds1000 | coreferential | ✓ | clean (low pass on both) |
| AMBI/018 | mbpp | coreferential | ⚠ | duplicate anchor (AMBI/005) |
| AMBI/019 | ds1000 | coreferential | ✓ | clean |
| AMBI/020 | ds1000 | coreferential | ✓ | clean |
| AMBI/021 | ds1000 | coreferential | ✓ | clean (only AC events came from this item) |
| AMBI/022 | ds1000 | coreferential | ✓ | clean |
| AMBI/023 | ds1000 | coreferential | ✓ | clean (good tax signal) |
| AMBI/024 | ds1000 | coreferential | ⚠ | both interpretations contrived |
| AMBI/025 | mbpp | syntactic | ✓ | clean |
| AMBI/026 | ds1000 | syntactic | ✓ | clean |
| AMBI/027 | ds1000 | syntactic | ✗ | Cat 3 dark + duplicate (AMBI/037) |
| AMBI/028 | mbpp | syntactic | ✗ | Cat 1 leak + duplicate |
| AMBI/029 | mbpp | syntactic | ⚠ | duplicate anchor (AMBI/011) |
| AMBI/030 | ds1000 | syntactic | ✓ | clean |
| AMBI/031 | ds1000 | syntactic | ✓ | clean |
| AMBI/032 | mbpp | syntactic | ✗ | Cat 3 dark, possible canonical bug |
| AMBI/033 | ds1000 | syntactic | ✗ | Cat 1 leak |
| AMBI/034 | ds1000 | syntactic | ✗ | Cat 1 leak |
| AMBI/035 | mbpp | scopal | ✓ | clean |
| AMBI/036 | ds1000 | scopal | ✓ | clean (excellent tax signal: 100→0) |
| AMBI/037 | ds1000 | scopal | ✗ | Cat 3 dark + duplicate (AMBI/027) |
| AMBI/038 | ds1000 | scopal | ✓ | clean |
| AMBI/039 | mbpp | scopal | ✗ | Cat 5 meta-prompt |
| AMBI/040 | ds1000 | scopal | ⚠ | duplicate anchor (AMBI/041) |
| AMBI/041 | ds1000 | scopal | ⚠ | duplicate anchor (AMBI/040) |
| AMBI/042 | ds1000 | scopal | ✓ | clean |
| AMBI/043 | ds1000 | coll_dist | ⚠ | duplicate anchor (AMBI/044), but otherwise clean signal |
| AMBI/044 | ds1000 | coll_dist | ⚠ | duplicate anchor (AMBI/043) |
| AMBI/045 | mbpp | coll_dist | ✗ | Cat 1 leak + duplicate (AMBI/003) |
| AMBI/046 | mbpp | coll_dist | ✗ | Cat 1 leak + duplicate (AMBI/006) |
| AMBI/047 | ds1000 | coll_dist | ✓ | clean |
| AMBI/048 | mbpp | coll_dist | ✓ | clean |
| AMBI/049 | ds1000 | coll_dist | ✓ | clean (excellent tax signal) |
| AMBI/050 | ds1000 | coll_dist | ✗ | Cat 3 dark + duplicate (AMBI/051) |
| AMBI/051 | ds1000 | coll_dist | ✗ | Cat 1 leak + duplicate (AMBI/050) |
| AMBI/052 | ds1000 | coll_dist | ✓ | clean |
| AMBI/053 | ds1000 | elliptical | ✗ | Cat 3 dark + duplicate (AMBI/055) |
| AMBI/054 | mbpp | elliptical | ⚠ | Cat 4 contrived B |
| AMBI/055 | ds1000 | elliptical | ✗ | Cat 3 dark + duplicate (AMBI/053) |
| AMBI/056 | ds1000 | elliptical | ✓ | clean |
| AMBI/057 | ds1000 | elliptical | ⚠ | Cat 4 contrived B |
| AMBI/058 | ds1000 | elliptical | ⚠ | Cat 4 contrived B (mild) |
| AMBI/059 | ds1000 | elliptical | ✗ | Cat 3 dark |
| AMBI/060 | ds1000 | elliptical | ✓ | clean (excellent tax signal) |
| AMBI/061 | ds1000 | elliptical | ✗ | Cat 3 dark |
| AMBI/062 | mbpp | elliptical | ✗ | Cat 1 leak |

**Counts**: ✓ clean = 30, ⚠ minor issue (mostly duplicate anchors) = 16, ✗ exclude = 16. A "clean subset" if we exclude only ✗ would be 46 items; if we also dedupe the ⚠ duplicate-anchor groups (keep 1 per group), about 38 items.