"""Stage 4 — Exclusivity Gate (Sandbox Verification).

Runs a 2x2 matrix of sandbox executions for each Stage 3 output:

    |              | test_a (original) | test_b (generated) |
    |--------------|-------------------|--------------------|
    | ref_a (canon)| PASS              | FAIL               |
    | ref_b (gen)  | FAIL              | PASS               |

All 4 must hold for the item to pass. This proves:
  - Two distinct interpretations exist
  - Each has a working solution
  - The tests can tell them apart

No LLM calls — only Docker sandbox executions.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from src.data.model import BenchmarkTask
from src.pipeline.prompts import load_pipeline_config
from src.pipeline.stage3_test_generation import TestGenOutput
from src.util.pipeline_runner import run_pipeline
from src.util.sandbox import Sandbox, SandboxResult


# ── Result types ─────────────────────────────────────────────────────────────

@dataclass
class ExclusivityResult:
    """Stage 4 result for one item."""
    task_id: str
    source: str
    entry_point: Optional[str] = None
    library: Optional[str] = None
    best_ambiguity_type: str = ""
    generator_model: str = ""
    entropy: float = 0.0

    # 2x2 matrix results
    ref_a_test_a: bool = False    # canonical passes original tests
    ref_a_test_b: bool = False    # canonical against test_b (should FAIL)
    ref_b_test_a: bool = False    # ref_solution_b against original tests (should FAIL)
    ref_b_test_b: bool = False    # ref_solution_b passes test_b

    # Error details for failed runs
    ref_a_test_a_stderr: str = ""
    ref_a_test_b_stderr: str = ""
    ref_b_test_a_stderr: str = ""
    ref_b_test_b_stderr: str = ""

    # Overall gate result
    passed: bool = False          # all 4 checks correct

    # Carry forward for final output
    perturbed_prompt: str = ""
    interpretation_a: str = ""
    interpretation_b: str = ""
    ref_solution_b: str = ""
    test_b: str = ""

    error: Optional[str] = None   # sandbox-level error (Docker failure etc.)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ExclusivityResult:
        return cls(**d)


# ── Input loading ───────────────────────────────────────────────────────────

def load_stage3_results(path: str | Path) -> list[TestGenOutput]:
    """Load Stage 3 results from JSONL."""
    results = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(TestGenOutput.from_dict(json.loads(line)))
    return results


# ── Single item verification ────────────────────────────────────────────────

def verify_exclusivity(
    sandbox: Sandbox,
    ref_a: str,
    ref_b: str,
    test_a: str,
    test_b: str,
    timeout_s: int = 30,
) -> dict:
    """Run the 2x2 matrix and return results.

    Returns dict with keys: ref_a_test_a, ref_a_test_b, ref_b_test_a, ref_b_test_b,
    plus stderr for each, and overall 'passed'.
    """
    r1 = sandbox.run(ref_a, test_a, timeout_s=timeout_s)
    r2 = sandbox.run(ref_a, test_b, timeout_s=timeout_s)
    r3 = sandbox.run(ref_b, test_a, timeout_s=timeout_s)
    r4 = sandbox.run(ref_b, test_b, timeout_s=timeout_s)

    # Exclusivity: ref_a passes test_a, fails test_b
    #              ref_b fails test_a, passes test_b
    passed = (
        r1.passed
        and not r2.passed
        and not r3.passed
        and r4.passed
    )

    return {
        "ref_a_test_a": r1.passed,
        "ref_a_test_b": r2.passed,
        "ref_b_test_a": r3.passed,
        "ref_b_test_b": r4.passed,
        "ref_a_test_a_stderr": r1.stderr[:500],
        "ref_a_test_b_stderr": r2.stderr[:500],
        "ref_b_test_a_stderr": r3.stderr[:500],
        "ref_b_test_b_stderr": r4.stderr[:500],
        "passed": passed,
    }


# ── Pipeline runner ─────────────────────────────────────────────────────────

def run_stage4(
    tasks: dict[str, BenchmarkTask],
    stage3_results: list[TestGenOutput],
    config: dict | None = None,
    output_dir: str | Path | None = None,
) -> list[ExclusivityResult]:
    """Run Stage 4 exclusivity gate on all Stage 3 outputs.

    Args:
        tasks: Mapping of task_id -> BenchmarkTask (raw data lookup).
        stage3_results: Output from Stage 3 (only items without error).
        config: Pipeline config (loaded from pipeline.yaml if None).
        output_dir: Output directory override.

    Returns:
        List of ExclusivityResult.
    """
    if config is None:
        config = load_pipeline_config()

    eg_config = config["exclusivity_gate"]
    if output_dir is None:
        output_dir = Path(eg_config["output_dir"])
    output_dir = Path(output_dir)

    timeout_s = eg_config.get("timeout_s", 30)
    max_workers = eg_config.get("max_workers", 4)

    # Filter to only successful Stage 3 items
    valid_items = [r for r in stage3_results if not r.error and r.ref_solution_b and r.test_b]

    print(f"Stage 4 — Exclusivity Gate (Sandbox)")
    print(f"  Items to verify: {len(valid_items)}")
    print(f"  Timeout: {timeout_s}s per execution")
    print(f"  Workers: {max_workers}")
    print(f"  Runs per item: 4 (2x2 matrix)")
    print()

    if not valid_items:
        print("  No valid items to process.")
        return []

    # Create sandboxes per Docker image (different images for different sources)
    docker_images = eg_config.get("docker_images", {})
    default_image = docker_images.get("default", "python:3.11-slim")
    mem_limit = eg_config.get("mem_limit", "256m")
    sandboxes: dict[str, Sandbox] = {}

    def get_sandbox(source: str) -> Sandbox:
        image = docker_images.get(source, default_image)
        if image not in sandboxes:
            sandboxes[image] = Sandbox(image=image, timeout_s=timeout_s, mem_limit=mem_limit)
        return sandboxes[image]

    def process_fn(index: int, item: TestGenOutput) -> dict | None:
        task = tasks.get(item.task_id)
        if task is None:
            print(f"  SKIP {item.task_id}: not found in raw data")
            return None

        # Build complete executable code from prompt + solution
        # HumanEval: prompt has function sig + docstring, solution has the body
        # MBPP: solution is self-contained
        # DS1000 (normalized): ref_a is __SOLUTION__ wrapper, test_a is harness;
        #   Stage 3 generates self-contained ref_b and test_b.
        #   Cross-format pairs need adaptation:
        #     ref_a × test_b: use original code fragment (not __SOLUTION__ wrapper)
        #     ref_b × test_a: wrap ref_b as __SOLUTION__ for the harness
        if task.source == "humaneval":
            ref_a = task.prompt + task.canonical_solution
        else:
            ref_a = task.canonical_solution

        ref_b = item.ref_solution_b
        test_a = task.test_code
        test_b = item.test_b

        # HumanEval test_code defines `def check(candidate)` but never calls it.
        # Append an explicit invocation so the asserts actually run.
        # Use a regex that matches a top-level call (not the def) by requiring
        # the line to start with `check(` (no leading `def `).
        if task.source == "humaneval" and task.entry_point:
            import re
            entry = task.entry_point
            invocation = f"\ncheck({entry})\n"
            top_level_call = re.compile(r"^\s*check\s*\(", re.MULTILINE)
            # The def line is `def check(candidate)` — has `def ` prefix on its line —
            # so a re match for `^\s*check(` won't hit it.
            if not top_level_call.search(test_a):
                test_a = test_a.rstrip() + invocation
            if not top_level_call.search(test_b):
                test_b = test_b.rstrip() + invocation

        if task.source == "ds1000" and task.metadata.get("normalized"):
            from src.data.ds1000_normalizer import _wrap_solution_as_string

            # ref_a × test_b: wrap original solution with imports + try/except
            # so function defs survive even if inline execution crashes on
            # undefined variables (e.g. df, X). test_b then calls the function.
            imports = task.metadata.get("exec_context_imports", "")
            original_sol = task.metadata.get("original_solution", "")
            ref_a_for_test_b = (
                f"{imports}\n\n"
                f"try:\n"
                + "".join(f"    {ln}\n" for ln in original_sol.split("\n"))
                + "except:\n"
                + "    pass\n"
            )
            # ref_b × test_a: wrap ref_b as __SOLUTION__ for the harness
            ref_b_for_test_a = _wrap_solution_as_string(item.ref_solution_b)

        try:
            sb = get_sandbox(task.source)
            if task.source == "ds1000" and task.metadata.get("normalized"):
                # Run each pair with the correctly formatted ref
                r1 = sb.run(ref_a, test_a, timeout_s=timeout_s)           # __SOLUTION__ + harness
                r2 = sb.run(ref_a_for_test_b, test_b, timeout_s=timeout_s) # original fragment + self-contained test
                r3 = sb.run(ref_b_for_test_a, test_a, timeout_s=timeout_s) # wrapped ref_b + harness
                r4 = sb.run(ref_b, test_b, timeout_s=timeout_s)           # self-contained + self-contained
                passed = r1.passed and not r2.passed and not r3.passed and r4.passed
                matrix = {
                    "ref_a_test_a": r1.passed,
                    "ref_a_test_b": r2.passed,
                    "ref_b_test_a": r3.passed,
                    "ref_b_test_b": r4.passed,
                    "ref_a_test_a_stderr": r1.stderr[:500],
                    "ref_a_test_b_stderr": r2.stderr[:500],
                    "ref_b_test_a_stderr": r3.stderr[:500],
                    "ref_b_test_b_stderr": r4.stderr[:500],
                    "passed": passed,
                }
            else:
                matrix = verify_exclusivity(sb, ref_a, ref_b, test_a, test_b, timeout_s)
        except Exception as e:
            result = ExclusivityResult(
                task_id=item.task_id,
                source=item.source,
                entry_point=item.entry_point,
                library=item.library,
                best_ambiguity_type=item.best_ambiguity_type,
                generator_model=item.generator_model,
                entropy=item.entropy,
                perturbed_prompt=item.perturbed_prompt,
                interpretation_a=item.interpretation_a,
                interpretation_b=item.interpretation_b,
                ref_solution_b=item.ref_solution_b,
                test_b=item.test_b,
                error=str(e)[:300],
            ).to_dict()
            result["__progress__"] = (
                f"{item.task_id:25s}  gen={item.generator_model:18s}  "
                f"ERROR: {str(e)[:60]}"
            )
            return result

        # Build status string for the 2x2 matrix
        def check(expected, actual, label):
            ok = actual == expected
            mark = "✓" if ok else "✗"
            return f"{label}={'P' if actual else 'F'}{mark}"

        checks = [
            check(True,  matrix["ref_a_test_a"], "ra·ta"),
            check(False, matrix["ref_a_test_b"], "ra·tb"),
            check(False, matrix["ref_b_test_a"], "rb·ta"),
            check(True,  matrix["ref_b_test_b"], "rb·tb"),
        ]
        gate = "PASS" if matrix["passed"] else "FAIL"

        result = ExclusivityResult(
            task_id=item.task_id,
            source=item.source,
            entry_point=item.entry_point,
            library=item.library,
            best_ambiguity_type=item.best_ambiguity_type,
            generator_model=item.generator_model,
            entropy=item.entropy,
            ref_a_test_a=matrix["ref_a_test_a"],
            ref_a_test_b=matrix["ref_a_test_b"],
            ref_b_test_a=matrix["ref_b_test_a"],
            ref_b_test_b=matrix["ref_b_test_b"],
            ref_a_test_a_stderr=matrix["ref_a_test_a_stderr"],
            ref_a_test_b_stderr=matrix["ref_a_test_b_stderr"],
            ref_b_test_a_stderr=matrix["ref_b_test_a_stderr"],
            ref_b_test_b_stderr=matrix["ref_b_test_b_stderr"],
            passed=matrix["passed"],
            perturbed_prompt=item.perturbed_prompt,
            interpretation_a=item.interpretation_a,
            interpretation_b=item.interpretation_b,
            ref_solution_b=item.ref_solution_b,
            test_b=item.test_b,
        ).to_dict()

        # Build debug lines showing stderr for unexpected results
        debug_lines = []
        expectations = [
            ("ra·ta", True,  matrix["ref_a_test_a"], matrix["ref_a_test_a_stderr"]),
            ("ra·tb", False, matrix["ref_a_test_b"], matrix["ref_a_test_b_stderr"]),
            ("rb·ta", False, matrix["ref_b_test_a"], matrix["ref_b_test_a_stderr"]),
            ("rb·tb", True,  matrix["ref_b_test_b"], matrix["ref_b_test_b_stderr"]),
        ]
        for label, expected, actual, stderr in expectations:
            if actual != expected and stderr:
                # Show first meaningful line of stderr
                err_line = stderr.strip().split("\n")[-1][:120]
                debug_lines.append(f"    {label}: {err_line}")

        progress = (
            f"{item.task_id:25s}  gen={item.generator_model:18s}  "
            f"{gate}  [{' | '.join(checks)}]"
        )
        if debug_lines:
            progress += "\n" + "\n".join(debug_lines)

        result["__progress__"] = progress
        return result

    raw_results = run_pipeline(
        items=valid_items,
        process_fn=process_fn,
        output_path=output_dir / "stage4_results.jsonl",
        max_workers=max_workers,
        label="Stage 4",
    )

    results = [ExclusivityResult.from_dict(r) for r in raw_results]

    # Summary
    passed = sum(1 for r in results if r.passed)
    errors = sum(1 for r in results if r.error)
    failed = len(results) - passed - errors

    print(f"\n  Passed: {passed}/{len(results)}")
    if failed:
        # Show failure breakdown
        fail_reasons = {"ra·ta=F": 0, "ra·tb=P": 0, "rb·ta=P": 0, "rb·tb=F": 0}
        for r in results:
            if not r.passed and not r.error:
                if not r.ref_a_test_a:
                    fail_reasons["ra·ta=F"] += 1
                if r.ref_a_test_b:
                    fail_reasons["ra·tb=P"] += 1
                if r.ref_b_test_a:
                    fail_reasons["rb·ta=P"] += 1
                if not r.ref_b_test_b:
                    fail_reasons["rb·tb=F"] += 1
        print(f"  Failure breakdown:")
        for reason, count in fail_reasons.items():
            if count:
                print(f"    {reason}: {count}")

    return results
