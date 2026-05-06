#!/usr/bin/env python3
"""Run the perturbation pipeline (Stage 1 + 1.5 + 2 + 3 + 4).

Stage 1   : SOTA models generate perturbed prompts + interpretations (with opt-out)
Stage 1.5 : Quality gate — independent judge checks leakage / B-naturalness / distinguishability
Stage 2   : Bilateral naturalness gate — 5 judges, "is A natural?" + "is B natural?"
Stage 3   : Generate ref_solution_b + test_b for passed perturbations
Stage 4   : Sandbox exclusivity gate — verify 2x2 matrix (ref_a/b × test_a/b)

Usage:
    python scripts/run_perturbation.py                                # full pipeline
    python scripts/run_perturbation.py --stage 1                      # one stage at a time
    python scripts/run_perturbation.py --stage 1.5                    # Stage 1.5 only
    python scripts/run_perturbation.py --max-tasks 10                 # limit tasks
    python scripts/run_perturbation.py --task-ids HumanEval/32        # specific tasks
"""

import argparse
import copy
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.store import BenchmarkStore
from src.pipeline.perturbation import load_anchor_results, select_anchors
from src.pipeline.prompts import load_pipeline_config
from src.pipeline.stage1_perturbation import run_stage1, Stage1Result
from src.pipeline.stage1_5_quality_gate import run_stage1_5, load_stage1_5_results
from src.pipeline.stage2_entropy_gate import run_stage2, load_stage1_results
from src.pipeline.stage3_test_generation import load_stage2_results, run_stage3
from src.pipeline.stage4_exclusivity_gate import load_stage3_results, run_stage4


def main():
    parser = argparse.ArgumentParser(description="Run perturbation pipeline")
    parser.add_argument(
        "--stage", type=str, choices=["1", "1.5", "2", "3", "4"], default=None,
        help="Run only this stage (default: all five)",
    )
    parser.add_argument(
        "--anchor-results", type=Path,
        default=Path("data/intermediate/anchor_selection/anchor_results.jsonl"),
        help="Path to anchor_results.jsonl (Stage 1 input)",
    )
    parser.add_argument(
        "--stage1-results", type=Path, default=None,
        help="Path to stage1_results.jsonl (Stage 1.5/2 input, auto-detected if omitted)",
    )
    parser.add_argument(
        "--stage1-5-results", type=Path, default=None,
        help="Path to stage1_5_results.jsonl (Stage 2 filter, auto-detected if omitted)",
    )
    parser.add_argument(
        "--stage2-results", type=Path, default=None,
        help="Path to stage2_results.jsonl (Stage 3 input, auto-detected if omitted)",
    )
    parser.add_argument(
        "--stage3-results", type=Path, default=None,
        help="Path to stage3_results.jsonl (Stage 4 input, auto-detected if omitted)",
    )
    parser.add_argument("--risk-level", type=str, default=None)
    parser.add_argument("--min-feasibility", type=float, default=None)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--task-ids", nargs="*", help="Specific task IDs")
    parser.add_argument("--output", "-o", type=Path, help="Output directory override")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_pipeline_config()
    run_s1 = args.stage is None or args.stage == "1"
    run_s1_5 = args.stage is None or args.stage == "1.5"
    run_s2 = args.stage is None or args.stage == "2"
    run_s3 = args.stage is None or args.stage == "3"
    run_s4 = args.stage is None or args.stage == "4"

    # Load raw data (needed by Stage 1, 1.5, 3, and 4)
    store = None
    task_map = None
    if run_s1 or run_s1_5 or run_s3 or run_s4:
        store = BenchmarkStore.load_local("data/raw")
        task_map = {t.task_id: t for t in store.all_tasks()}
        print(f"Loaded {len(store)} raw tasks for lookup")

    # ── Stage 1 ──────────────────────────────────────────────────────────
    s1_results = None
    if run_s1:
        pert_config = config["perturbation"]

        print(f"\nLoading anchor results from {args.anchor_results}...")
        anchor_results = load_anchor_results(args.anchor_results)
        print(f"Loaded {len(anchor_results)} anchor results")

        # Select anchors
        risk_level = args.risk_level or pert_config["risk_level"]
        min_feas = args.min_feasibility if args.min_feasibility is not None else pert_config["min_feasibility_score"]
        max_tasks = args.max_tasks or pert_config["max_tasks"]

        if args.task_ids:
            id_set = set(args.task_ids)
            selected = [r for r in anchor_results if r.task_id in id_set]
            print(f"Selected {len(selected)} tasks by ID")
        else:
            selected = select_anchors(
                anchor_results,
                min_feasibility=min_feas,
                risk_level=risk_level,
                max_tasks=max_tasks,
            )
            print(f"Selected {len(selected)} tasks (risk={risk_level}, feas>={min_feas})")

        if not selected:
            print("No tasks matched selection criteria.")
            return

        print()
        s1_results = run_stage1(
            tasks=task_map,
            anchor_results=selected,
            config=config,
            output_dir=args.output,
        )

    # ── Stage 1.5 — Quality Gate ─────────────────────────────────────────
    s1_5_results = None
    if run_s1_5:
        if s1_results is None:
            s1_path = args.stage1_results
            if s1_path is None:
                s1_path = Path(config["perturbation"]["output_dir"]) / "stage1_results.jsonl"
            print(f"\nLoading Stage 1 results from {s1_path}...")
            s1_results = load_stage1_results(s1_path)
            print(f"Loaded {len(s1_results)} Stage 1 results")

        if task_map is None:
            store = BenchmarkStore.load_local("data/raw")
            task_map = {t.task_id: t for t in store.all_tasks()}
            print(f"Loaded {len(store)} raw tasks for lookup")

        print()
        s1_5_results = run_stage1_5(
            tasks=task_map,
            stage1_results=s1_results,
            config=config,
            output_dir=args.output,
        )

    # ── Stage 2 — Bilateral Naturalness Gate ─────────────────────────────
    s2_results = None
    if run_s2:
        if s1_results is None:
            s1_path = args.stage1_results
            if s1_path is None:
                s1_path = Path(config["perturbation"]["output_dir"]) / "stage1_results.jsonl"
            print(f"\nLoading Stage 1 results from {s1_path}...")
            s1_results = load_stage1_results(s1_path)
            print(f"Loaded {len(s1_results)} Stage 1 results")

        # Load Stage 1.5 results to filter Stage 1 generations down to those
        # that passed the quality gate. If no Stage 1.5 file exists, Stage 2
        # will see all non-errored / non-opted-out generations (legacy behavior).
        if s1_5_results is None:
            s1_5_path = args.stage1_5_results
            if s1_5_path is None:
                s1_5_path = Path(config["quality_gate"]["output_dir"]) / "stage1_5_results.jsonl"
            if s1_5_path.exists():
                print(f"Loading Stage 1.5 results from {s1_5_path}...")
                s1_5_results = load_stage1_5_results(s1_5_path)
                print(f"Loaded {len(s1_5_results)} Stage 1.5 results")
            else:
                print(f"No Stage 1.5 results found at {s1_5_path}; "
                      f"Stage 2 will audit ALL non-errored generations.")

        if s1_5_results is not None:
            # Build {task_id: {generator_model: passed}} from Stage 1.5
            qg_pass: dict[str, dict[str, bool]] = {}
            for r in s1_5_results:
                qg_pass[r.task_id] = {
                    qr["generator_model"]: bool(qr.get("passed", False))
                    for qr in r.quality_results
                }
            # Filter each Stage-1 result's generations to only those that passed
            filtered_s1: list[Stage1Result] = []
            kept = dropped = 0
            for r in s1_results:
                pass_map = qg_pass.get(r.task_id, {})
                kept_gens = []
                for g in r.generations:
                    if g.get("error") or g.get("opted_out") or not g.get("perturbed_prompt"):
                        continue
                    if pass_map.get(g["model"], False):
                        kept_gens.append(g)
                        kept += 1
                    else:
                        dropped += 1
                # Keep the Stage1Result shell even if all gens were dropped
                fr = copy.deepcopy(r)
                fr.generations = kept_gens
                filtered_s1.append(fr)
            print(f"Stage 1.5 filter: kept {kept} generations, "
                  f"dropped {dropped} that failed the quality gate")
            s1_input_for_s2 = filtered_s1
        else:
            s1_input_for_s2 = s1_results

        print()
        s2_results = run_stage2(
            stage1_results=s1_input_for_s2,
            config=config,
            output_dir=args.output,
        )

    # ── Stage 3 ──────────────────────────────────────────────────────────
    s3_results = None
    if run_s3:
        if s2_results is None:
            s2_path = args.stage2_results
            if s2_path is None:
                s2_path = Path(config["entropy_gate"]["output_dir"]) / "stage2_results.jsonl"
            print(f"\nLoading Stage 2 results from {s2_path}...")
            s2_results = load_stage2_results(s2_path)
            print(f"Loaded {len(s2_results)} Stage 2 results")

        if task_map is None:
            store = BenchmarkStore.load_local("data/raw")
            task_map = {t.task_id: t for t in store.all_tasks()}
            print(f"Loaded {len(store)} raw tasks for lookup")

        print()
        s3_results = run_stage3(
            tasks=task_map,
            stage2_results=s2_results,
            config=config,
            output_dir=args.output,
        )

    # ── Stage 4 ──────────────────────────────────────────────────────────
    s4_results = None
    if run_s4:
        if s3_results is None:
            s3_path = args.stage3_results
            if s3_path is None:
                s3_path = Path(config["test_generation"]["output_dir"]) / "stage3_results.jsonl"
            print(f"\nLoading Stage 3 results from {s3_path}...")
            s3_results = load_stage3_results(s3_path)
            print(f"Loaded {len(s3_results)} Stage 3 results")

        if task_map is None:
            store = BenchmarkStore.load_local("data/raw")
            task_map = {t.task_id: t for t in store.all_tasks()}
            print(f"Loaded {len(store)} raw tasks for lookup")

        print()
        s4_results = run_stage4(
            tasks=task_map,
            stage3_results=s3_results,
            config=config,
            output_dir=args.output,
        )

    # ── Summary ──────────────────────────────────────────────────────────
    # Show Stage 2 summary if we ran it
    if run_s2 and s2_results:
        print(f"\n{'='*60}")
        print("PIPELINE SUMMARY")
        print(f"{'='*60}")

        all_entropies = [
            er["entropy"]
            for r in s2_results for er in r.entropy_results
        ]
        if all_entropies:
            print(f"\nEntropy distribution across {len(all_entropies)} generations:")
            bins = {"H=0 (5-0)": 0, "0<H<0.72 (fail)": 0,
                    "0.72<=H<0.97 (4-1)": 0, "H>=0.97 (3-2)": 0}
            for h in all_entropies:
                if h == 0:
                    bins["H=0 (5-0)"] += 1
                elif h < 0.72:
                    bins["0<H<0.72 (fail)"] += 1
                elif h < 0.97:
                    bins["0.72<=H<0.97 (4-1)"] += 1
                else:
                    bins["H>=0.97 (3-2)"] += 1
            for label, count in bins.items():
                bar = "#" * count
                print(f"  {label:25s}: {count:3d}  {bar}")

        gen_stats: dict[str, dict] = {}
        for r in s2_results:
            for er in r.entropy_results:
                m = er["generator_model"]
                if m not in gen_stats:
                    gen_stats[m] = {"total": 0, "passed": 0}
                gen_stats[m]["total"] += 1
                if er.get("passed"):
                    gen_stats[m]["passed"] += 1

        if gen_stats:
            print(f"\nPass rate by generator:")
            for m, s in gen_stats.items():
                pct = s["passed"] / s["total"] * 100 if s["total"] else 0
                print(f"  {m:25s}: {s['passed']}/{s['total']} ({pct:.0f}%)")

        passed_types = Counter(
            r.best_ambiguity_type
            for r in s2_results if r.any_passed
        )
        if passed_types:
            print(f"\nAmbiguity types (tasks with >= 1 passing generation):")
            for atype, count in passed_types.most_common():
                print(f"  {atype:25s}: {count}")

    # Show Stage 3 summary if we ran it
    if run_s3 and s3_results:
        print(f"\n{'='*60}")
        print("STAGE 3 — REF SOLUTION B + TEST GENERATION SUMMARY")
        print(f"{'='*60}")

        ok = [r for r in s3_results if not r.error]
        err = [r for r in s3_results if r.error]
        print(f"\n  Total: {len(s3_results)}  Success: {len(ok)}  Errors: {len(err)}")

        if ok:
            avg_ref = sum(len(r.ref_solution_b) for r in ok) / len(ok)
            avg_test = sum(len(r.test_b) for r in ok) / len(ok)
            print(f"  Avg ref_solution_b: {avg_ref:.0f} chars")
            print(f"  Avg test_b: {avg_test:.0f} chars")

            # Per-model stats
            model_stats: dict[str, dict] = {}
            for r in ok:
                m = r.test_gen_model
                if m not in model_stats:
                    model_stats[m] = {"count": 0, "ref_lens": [], "test_lens": []}
                model_stats[m]["count"] += 1
                model_stats[m]["ref_lens"].append(len(r.ref_solution_b))
                model_stats[m]["test_lens"].append(len(r.test_b))
            print(f"\n  By model:")
            for m, s in model_stats.items():
                avg_r = sum(s["ref_lens"]) / s["count"]
                avg_t = sum(s["test_lens"]) / s["count"]
                print(f"    {m:25s}: {s['count']} items, ref_b avg {avg_r:.0f}, test_b avg {avg_t:.0f} chars")

            # Show sample
            sample = ok[0]
            print(f"\n  --- Sample: {sample.task_id} (gen={sample.generator_model}) ---")
            print(f"  Interp B: {sample.interpretation_b[:150]}")
            print(f"  ref_solution_b preview:")
            for line in sample.ref_solution_b.split("\n")[:8]:
                print(f"    {line}")
            if sample.ref_solution_b.count("\n") > 8:
                print(f"    ... ({sample.ref_solution_b.count(chr(10))} lines total)")
            print(f"  test_b preview:")
            for line in sample.test_b.split("\n")[:8]:
                print(f"    {line}")
            if sample.test_b.count("\n") > 8:
                print(f"    ... ({sample.test_b.count(chr(10))} lines total)")

    # Show Stage 4 summary if we ran it
    if run_s4 and s4_results:
        print(f"\n{'='*60}")
        print("STAGE 4 — EXCLUSIVITY GATE SUMMARY")
        print(f"{'='*60}")

        passed = [r for r in s4_results if r.passed]
        failed = [r for r in s4_results if not r.passed and not r.error]
        errors = [r for r in s4_results if r.error]

        print(f"\n  Total: {len(s4_results)}  Passed: {len(passed)}  Failed: {len(failed)}  Errors: {len(errors)}")

        if passed:
            print(f"\n  Passed items:")
            for r in passed:
                print(f"    {r.task_id:25s}  {r.best_ambiguity_type:22s}  gen={r.generator_model}")

        if failed:
            print(f"\n  Failed items (2x2 matrix):")
            for r in failed:
                def mark(expected, actual):
                    return ("P" if actual else "F") + ("✓" if actual == expected else "✗")
                print(
                    f"    {r.task_id:25s}  gen={r.generator_model:18s}  "
                    f"ra·ta={mark(True, r.ref_a_test_a)}  "
                    f"ra·tb={mark(False, r.ref_a_test_b)}  "
                    f"rb·ta={mark(False, r.ref_b_test_a)}  "
                    f"rb·tb={mark(True, r.ref_b_test_b)}"
                )


if __name__ == "__main__":
    main()
