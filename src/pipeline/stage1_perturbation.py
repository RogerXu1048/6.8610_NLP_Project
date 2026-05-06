"""Stage 1 — Perturbation Generation.

Each SOTA model generates for a given anchor task:
  - perturbed_prompt: the clean prompt rewritten with injected ambiguity
  - interpretation_a: one-sentence explanation (original meaning)
  - interpretation_b: one-sentence explanation (new ambiguous meaning)

Lightweight call (~500 tokens output). No test generation at this stage.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from src.data.model import BenchmarkTask
from src.pipeline.anchor_selection import AnchorResult
from src.pipeline.perturbation import AMBIGUITY_TYPE_DEFS
from src.pipeline.prompts import get_prompt, load_pipeline_config, render_prompt
from src.util.llm import LLMClient, ModelConfig
from src.util.parsing import parse_json_response
from src.util.pipeline_runner import run_pipeline


# ── Result types ─────────────────────────────────────────────────────────────

@dataclass
class GeneratorOutput:
    """One SOTA model's perturbation generation.

    Three terminal states:
      1. Success — perturbed_prompt populated, opted_out=False, error=None
      2. Opt-out — perturbed_prompt="", opted_out=True, reason populated, error=None
      3. Error  — perturbed_prompt="", opted_out=False, error populated
    """
    model: str
    perturbed_prompt: str = ""
    interpretation_a: str = ""
    interpretation_b: str = ""
    raw_response: str = ""
    opted_out: bool = False        # NEW (reform B): generator declined to perturb
    reason: str = ""               # NEW (reform B): explanation for opt-out
    error: Optional[str] = None


@dataclass
class Stage1Result:
    """Stage 1 result for one anchor task (all generator outputs)."""
    task_id: str
    source: str
    entry_point: Optional[str] = None
    library: Optional[str] = None

    # From anchor selection
    best_ambiguity_type: str = ""
    feasibility_score: float = 0.0
    risk_level: str = ""
    perturbation_sketches: list[str] = field(default_factory=list)

    # Generator outputs (one per SOTA model, raw_response stripped for storage)
    generations: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Stage1Result:
        return cls(**d)


# ── Single generation call ──────────────────────────────────────────────────

def generate_single(
    client: LLMClient,
    task: BenchmarkTask,
    anchor: AnchorResult,
    model_alias: str,
    config: dict,
) -> GeneratorOutput:
    """One SOTA model generates a perturbation (prompt + interpretations)."""
    ambiguity_type = anchor.best_ambiguity_type
    type_def = AMBIGUITY_TYPE_DEFS.get(ambiguity_type, "")

    sketches = anchor.perturbation_sketches.get(ambiguity_type, [])
    sketches_text = "\n".join(f"  - {s}" for s in sketches) if sketches else "  (none)"

    system = get_prompt("perturbation.system")
    prompt = render_prompt(
        "perturbation.task",
        ambiguity_type=ambiguity_type,
        ambiguity_type_definition=type_def,
        prompt=task.prompt,
        canonical_solution=task.canonical_solution,
        best_ambiguity_type=ambiguity_type,
        perturbation_sketches=sketches_text,
    )

    mc = ModelConfig(
        model=model_alias,
        temperature=config["perturbation"]["temperature"],
        max_tokens=config["perturbation"]["max_tokens"],
    )

    raw = ""
    try:
        resp = client.call(mc, prompt=prompt, system=system)
        raw = resp.choices[0]
        parsed = parse_json_response(raw)

        # Reform B: handle opt-out (perturbed_prompt is null/None)
        pp = parsed.get("perturbed_prompt", "")
        if pp is None or (isinstance(pp, str) and pp.strip() == ""):
            return GeneratorOutput(
                model=model_alias,
                perturbed_prompt="",
                interpretation_a="",
                interpretation_b="",
                raw_response=raw,
                opted_out=True,
                reason=parsed.get("reason", "")[:500],  # cap to 500 chars
            )

        return GeneratorOutput(
            model=model_alias,
            perturbed_prompt=pp,
            interpretation_a=parsed.get("interpretation_a", ""),
            interpretation_b=parsed.get("interpretation_b", ""),
            raw_response=raw,
            opted_out=False,
            reason=parsed.get("reason", "")[:500],
        )
    except Exception as e:
        error_msg = str(e)
        if raw:
            error_msg += f"\n    RAW_RESPONSE: {raw[:300]}"
        return GeneratorOutput(model=model_alias, raw_response=raw, error=error_msg)


# ── Per-task runner (all models concurrently) ───────────────────────────────

def generate_for_task(
    client: LLMClient,
    task: BenchmarkTask,
    anchor: AnchorResult,
    config: dict,
    models: list[str] | None = None,
) -> list[GeneratorOutput]:
    """Run all SOTA models on a single anchor task concurrently."""
    if models is None:
        models = config["perturbation"]["generator_models"]

    outputs: dict[str, GeneratorOutput] = {}

    def _call(m):
        outputs[m] = generate_single(client, task, anchor, m, config)

    with ThreadPoolExecutor(max_workers=len(models)) as pool:
        for m in models:
            pool.submit(_call, m)

    return [outputs[m] for m in models]


# ── Pipeline runner ─────────────────────────────────────────────────────────

def run_stage1(
    tasks: dict[str, BenchmarkTask],
    anchor_results: list[AnchorResult],
    config: dict | None = None,
    output_dir: str | Path | None = None,
) -> list[Stage1Result]:
    """Run Stage 1 perturbation generation for all selected anchors.

    Args:
        tasks: Mapping of task_id -> BenchmarkTask (raw data lookup).
        anchor_results: Pre-selected anchor results to process.
        config: Pipeline config (loaded from pipeline.yaml if None).
        output_dir: Output directory override.

    Returns:
        List of Stage1Result in input order.
    """
    if config is None:
        config = load_pipeline_config()

    if output_dir is None:
        output_dir = Path(config["perturbation"]["output_dir"])
    output_dir = Path(output_dir)

    models = config["perturbation"]["generator_models"]
    max_workers = config["perturbation"].get("max_workers", 8)
    client = LLMClient()

    print(f"Stage 1 — Perturbation Generation")
    print(f"  Tasks: {len(anchor_results)}")
    print(f"  Models: {', '.join(models)}")
    print(f"  Workers: {max_workers}")
    print()

    def process_fn(index: int, anchor: AnchorResult) -> dict | None:
        task = tasks.get(anchor.task_id)
        if task is None:
            print(f"  SKIP {anchor.task_id}: not found in raw data")
            return None

        generations = generate_for_task(client, task, anchor, config, models)

        sketches = anchor.perturbation_sketches.get(anchor.best_ambiguity_type, [])
        clean_gens = [
            {k: v for k, v in asdict(g).items() if k != "raw_response"}
            for g in generations
        ]

        ok = sum(1 for g in generations if not g.error)
        err = len(generations) - ok

        result = Stage1Result(
            task_id=anchor.task_id,
            source=anchor.source,
            entry_point=anchor.entry_point,
            library=anchor.library,
            best_ambiguity_type=anchor.best_ambiguity_type,
            feasibility_score=anchor.feasibility_score,
            risk_level=anchor.risk_level,
            perturbation_sketches=sketches,
            generations=clean_gens,
        ).to_dict()

        err_str = f"  err={err}" if err else ""
        result["__progress__"] = (
            f"{anchor.task_id:25s}  "
            f"type={anchor.best_ambiguity_type:22s}  "
            f"ok={ok}/{len(models)}{err_str}"
        )
        return result

    raw_results = run_pipeline(
        items=anchor_results,
        process_fn=process_fn,
        output_path=output_dir / "stage1_results.jsonl",
        max_workers=max_workers,
        label="Stage 1",
    )

    return [Stage1Result.from_dict(r) for r in raw_results]
