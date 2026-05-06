"""Stage 1.5 — Quality Gate.

For each Stage-1 generation that produced a perturbed prompt (i.e. didn't
opt out, didn't error), one cheap judge model reads:
  - the original clean prompt
  - the perturbed prompt
  - interpretation A and B

and answers three independent yes/no questions:

  Q1 LEAKAGE        — does perturbed contain info not in clean?
  Q2 B_NATURALNESS  — would a senior programmer plausibly write code matching B?
  Q3 DISTINGUISH    — do A and B produce different outputs on typical inputs?

Pass criterion: leakage=no AND b_natural=yes AND distinguishable=yes.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from src.data.model import BenchmarkTask
from src.pipeline.prompts import get_prompt, load_pipeline_config, render_prompt
from src.pipeline.stage1_perturbation import Stage1Result
from src.util.llm import LLMClient, ModelConfig
from src.util.parsing import parse_json_response
from src.util.pipeline_runner import run_pipeline


# ── Result types ─────────────────────────────────────────────────────────────

@dataclass
class QualityGateResult:
    """Quality-gate verdict for ONE Stage-1 generation."""
    generator_model: str
    perturbed_prompt: str = ""
    interpretation_a: str = ""
    interpretation_b: str = ""

    # Judge answers (None if judge errored)
    leakage: Optional[bool] = None         # Q1: True = leak detected (BAD)
    b_natural: Optional[bool] = None       # Q2: True = naturally writeable (GOOD)
    distinguishable: Optional[bool] = None # Q3: True = differ on typical input (GOOD)
    judge_reasoning: str = ""

    passed: bool = False                    # all three flags pass
    judge_error: Optional[str] = None


@dataclass
class Stage1_5Result:
    """Stage 1.5 results for one anchor task (one verdict per Stage-1 generation)."""
    task_id: str
    source: str
    entry_point: Optional[str] = None
    library: Optional[str] = None
    best_ambiguity_type: str = ""

    # One per Stage-1 generation that produced a perturbed_prompt
    quality_results: list[dict] = field(default_factory=list)

    # Summary
    any_passed: bool = False
    n_evaluated: int = 0    # generations that reached the gate (not opted-out, not errored)
    n_passed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Stage1_5Result:
        return cls(**d)


# ── Single judge call ───────────────────────────────────────────────────────

def _yn(value: object) -> Optional[bool]:
    """Parse 'yes'/'no' (case-insensitive) into bool. Anything else → None."""
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    if v == "yes":
        return True
    if v == "no":
        return False
    return None


def judge_single(
    client: LLMClient,
    clean_prompt: str,
    perturbed_prompt: str,
    interpretation_a: str,
    interpretation_b: str,
    generator_model: str,
    config: dict,
) -> QualityGateResult:
    """One judge inspects one Stage-1 generation."""
    qg_config = config["quality_gate"]
    judge_alias = qg_config["judge_model"]

    system = get_prompt("quality_gate.system")
    prompt = render_prompt(
        "quality_gate.task",
        clean_prompt=clean_prompt,
        perturbed_prompt=perturbed_prompt,
        interpretation_a=interpretation_a,
        interpretation_b=interpretation_b,
    )

    mc = ModelConfig(
        model=judge_alias,
        temperature=qg_config["temperature"],
        max_tokens=qg_config["max_tokens"],
    )

    raw = ""
    try:
        resp = client.call(mc, prompt=prompt, system=system)
        raw = resp.choices[0]
        parsed = parse_json_response(raw)

        leakage = _yn(parsed.get("leakage"))
        b_natural = _yn(parsed.get("b_natural"))
        distinguishable = _yn(parsed.get("distinguishable"))

        if leakage is None or b_natural is None or distinguishable is None:
            raise ValueError(
                f"Invalid yes/no in response. "
                f"leakage={parsed.get('leakage')!r}, "
                f"b_natural={parsed.get('b_natural')!r}, "
                f"distinguishable={parsed.get('distinguishable')!r}"
            )

        passed = (not leakage) and b_natural and distinguishable

        return QualityGateResult(
            generator_model=generator_model,
            perturbed_prompt=perturbed_prompt,
            interpretation_a=interpretation_a,
            interpretation_b=interpretation_b,
            leakage=leakage,
            b_natural=b_natural,
            distinguishable=distinguishable,
            judge_reasoning=str(parsed.get("reasoning", ""))[:500],
            passed=passed,
        )
    except Exception as e:
        error_msg = str(e)
        if raw:
            error_msg += f"\n    RAW: {raw[:300]}"
        return QualityGateResult(
            generator_model=generator_model,
            perturbed_prompt=perturbed_prompt,
            interpretation_a=interpretation_a,
            interpretation_b=interpretation_b,
            judge_error=error_msg,
        )


# ── Pipeline runner ─────────────────────────────────────────────────────────

def load_stage1_5_results(path: str | Path) -> list[Stage1_5Result]:
    """Load Stage 1.5 results from JSONL."""
    results = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(Stage1_5Result.from_dict(json.loads(line)))
    return results


def run_stage1_5(
    tasks: dict[str, BenchmarkTask],
    stage1_results: list[Stage1Result],
    config: dict | None = None,
    output_dir: str | Path | None = None,
) -> list[Stage1_5Result]:
    """Run Stage 1.5 quality gate on all Stage-1 results.

    Args:
        tasks: Mapping of task_id -> BenchmarkTask (used to look up the clean prompt).
        stage1_results: Output from Stage 1.
        config: Pipeline config (loaded from pipeline.yaml if None).
        output_dir: Output directory override.

    Returns:
        List of Stage1_5Result in input order.
    """
    if config is None:
        config = load_pipeline_config()

    if output_dir is None:
        output_dir = Path(config["quality_gate"]["output_dir"])
    output_dir = Path(output_dir)

    qg_config = config["quality_gate"]
    judge_alias = qg_config["judge_model"]
    max_workers = qg_config.get("max_workers", 12)
    client = LLMClient()

    # Count generations that reached this gate (opted-out / errored generations skip)
    eligible_total = sum(
        1
        for r in stage1_results
        for g in r.generations
        if not g.get("error") and not g.get("opted_out") and g.get("perturbed_prompt")
    )

    print(f"Stage 1.5 — Quality Gate")
    print(f"  Tasks: {len(stage1_results)}")
    print(f"  Generations to audit: {eligible_total}")
    print(f"  Judge: {judge_alias}")
    print(f"  Workers: {max_workers}")
    print()

    def process_fn(index: int, s1: Stage1Result) -> dict | None:
        task = tasks.get(s1.task_id)
        if task is None:
            print(f"  SKIP {s1.task_id}: not found in raw data")
            return None

        eligible = [
            g for g in s1.generations
            if not g.get("error") and not g.get("opted_out") and g.get("perturbed_prompt")
        ]
        if not eligible:
            # Nothing to audit — record an empty result so downstream can see.
            stage1_5 = Stage1_5Result(
                task_id=s1.task_id,
                source=s1.source,
                entry_point=s1.entry_point,
                library=s1.library,
                best_ambiguity_type=s1.best_ambiguity_type,
                quality_results=[],
                any_passed=False,
                n_evaluated=0,
                n_passed=0,
            ).to_dict()
            stage1_5["__progress__"] = (
                f"{s1.task_id:25s}  no-eligible (all opted-out / errored)"
            )
            return stage1_5

        verdicts: list[QualityGateResult] = []
        for g in eligible:
            v = judge_single(
                client,
                clean_prompt=task.prompt,
                perturbed_prompt=g["perturbed_prompt"],
                interpretation_a=g["interpretation_a"],
                interpretation_b=g["interpretation_b"],
                generator_model=g["model"],
                config=config,
            )
            verdicts.append(v)

        clean = [asdict(v) for v in verdicts]
        n_passed = sum(1 for v in verdicts if v.passed)

        result = Stage1_5Result(
            task_id=s1.task_id,
            source=s1.source,
            entry_point=s1.entry_point,
            library=s1.library,
            best_ambiguity_type=s1.best_ambiguity_type,
            quality_results=clean,
            any_passed=n_passed > 0,
            n_evaluated=len(verdicts),
            n_passed=n_passed,
        ).to_dict()

        # Concise per-generation breakdown for the progress line
        details = []
        for v in verdicts:
            if v.judge_error:
                tag = "ERR"
            elif v.passed:
                tag = "PASS"
            else:
                # Surface which flag(s) caused the failure
                fails = []
                if v.leakage is True:
                    fails.append("leak")
                if v.b_natural is False:
                    fails.append("contrived-B")
                if v.distinguishable is False:
                    fails.append("indistinct")
                tag = "FAIL[" + ",".join(fails) + "]" if fails else "FAIL"
            details.append(f"{v.generator_model.split('/')[-1]}={tag}")

        result["__progress__"] = (
            f"{s1.task_id:25s}  pass={n_passed}/{len(verdicts)}  [{' '.join(details)}]"
        )
        return result

    raw_results = run_pipeline(
        items=stage1_results,
        process_fn=process_fn,
        output_path=output_dir / "stage1_5_results.jsonl",
        max_workers=max_workers,
        label="Stage 1.5",
    )

    results = [Stage1_5Result.from_dict(r) for r in raw_results]

    # Summary
    total_eval = sum(r.n_evaluated for r in results)
    total_pass = sum(r.n_passed for r in results)
    tasks_with_pass = sum(1 for r in results if r.any_passed)
    print(f"\n  Generations passed: {total_pass}/{total_eval}")
    print(f"  Tasks with >= 1 pass: {tasks_with_pass}/{len(results)}")

    return results