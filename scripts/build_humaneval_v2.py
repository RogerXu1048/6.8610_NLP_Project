"""Assemble benchmark_humaneval_v2.jsonl from the v2-pipeline Stage 4 output.

Loads passed Stage 4 results, joins with the original BenchmarkTask data,
and writes BenchmarkItems with task_ids continuing from the v1 numbering
(v1 ended at AMBI/062, so HumanEval items start at AMBI/063).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.model import BenchmarkItem
from src.data.store import BenchmarkStore

S4 = ROOT / "data/intermediate/humaneval_v2/stage4_results.jsonl"
S2 = ROOT / "data/intermediate/humaneval_v2/stage2_results.jsonl"
OUT = ROOT / "data/benchmark/benchmark_humaneval_v2.jsonl"
START_ID = 63  # v1 ended at AMBI/062


def main() -> None:
    s4_results = [json.loads(line) for line in S4.read_text().splitlines() if line.strip()]
    passed = [r for r in s4_results if r.get("passed")]
    print(f"Stage 4 passed: {len(passed)} items")

    # Stage 2 has the entropy_results we want to attach as quality_gate_b_votes.
    s2_results = {
        r["task_id"]: r
        for r in (json.loads(line) for line in S2.read_text().splitlines() if line.strip())
    }

    store = BenchmarkStore.load_local("data/raw")
    task_map = {t.task_id: t for t in store.all_tasks()}

    items = []
    item_id = START_ID
    for s4 in passed:
        anchor_id = s4["task_id"]
        task = task_map.get(anchor_id)
        if task is None:
            print(f"  SKIP {anchor_id}: not found in raw data")
            continue

        # Find the matching Stage-2 entropy result for the surviving generator
        s2_record = s2_results.get(anchor_id, {})
        gen_model = s4.get("generator_model")
        s2_match = next(
            (
                er for er in s2_record.get("entropy_results", [])
                if er.get("generator_model") == gen_model and er.get("passed")
            ),
            None,
        )
        votes_record = {
            "yes_a": s2_match.get("yes_a") if s2_match else None,
            "yes_b": s2_match.get("yes_b") if s2_match else None,
            "entropy": s2_match.get("entropy") if s2_match else None,
            "votes": s2_match.get("votes", []) if s2_match else [],
        }

        item = BenchmarkItem(
            task_id=f"AMBI/{item_id:03d}",
            anchor_task_id=anchor_id,
            source=task.source,
            prompt=task.prompt,
            canonical_solution=task.canonical_solution,
            test_code=task.test_code,
            entry_point=task.entry_point,
            library=task.library,
            perturbed_prompt=s4["perturbed_prompt"],
            ambiguity_type=s4["best_ambiguity_type"],
            risk_level="low",  # all HumanEval anchors with feas≥0.6 are low-risk
            interpretation_a=s4["interpretation_a"],
            interpretation_b=s4["interpretation_b"],
            ref_solution_a=task.canonical_solution,
            ref_solution_b=s4["ref_solution_b"],
            test_a=task.test_code,
            test_b=s4["test_b"],
            quality_gate_a=True,
            quality_gate_b=True,
            quality_gate_b_votes=votes_record,
        )
        items.append(item)
        item_id += 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for item in items:
            f.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")

    print(f"Wrote {len(items)} items → {OUT}")
    for it in items:
        print(f"  {it.task_id}  ←  {it.anchor_task_id}  "
              f"({it.ambiguity_type}, {it.risk_level})")


if __name__ == "__main__":
    main()