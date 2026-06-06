"""Audit paper-replication artifacts across the full Tjong pipeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .paper_metrics import PAPER_NAME


@dataclass(frozen=True)
class JsonExpectation:
    path: str
    expected: Any | None = None
    minimum: float | None = None


@dataclass(frozen=True)
class StageSpec:
    name: str
    description: str
    files: tuple[str, ...] = ()
    json_file: str | None = None
    expectations: tuple[JsonExpectation, ...] = ()
    next_action: str = ""
    optional: bool = False
    alternatives: tuple[tuple[str, ...], ...] = field(default_factory=tuple)


def record_count_expectation(path: str, expected_records: int) -> JsonExpectation:
    if expected_records > 0:
        return JsonExpectation(path, expected_records)
    return JsonExpectation(path, minimum=1)


def stage_specs(expected_records: int) -> list[StageSpec]:
    return [
        StageSpec(
            name="paper_extraction",
            description="OCR/page-text extraction records the architecture and training constants.",
            files=(
                "papers/tjong_cit2_12298/notes/extraction.md",
                "papers/tjong_cit2_12298/ocr/figure_ocr.md",
            ),
            next_action="keep OCR notes with code changes; rerun paper compliance after edits",
        ),
        StageSpec(
            name="converted_botzone_corpus",
            description="Available Tziakcha corpus is converted into Botzone-like MCR logs.",
            files=("data/raw/tziakcha_all_sources_botzone_20260605b.jsonl",),
            json_file="runs/tjong_validate_tziakcha_all_sources_botzone_20260605b.json",
            expectations=(
                record_count_expectation("records", expected_records),
                JsonExpectation("error_count", 0),
                JsonExpectation("valid_for_paper_supervised", True),
            ),
            next_action="apply k8s/tjong-tziakcha-all-botzone-merge-cpu-20260605b.yaml",
            alternatives=(("papers/tjong_cit2_12298/artifacts/tjong_validate_tziakcha_all_sources_botzone_20260605b.json",),),
        ),
        StageSpec(
            name="supervised_tensor",
            description="Converted corpus is tensorized into the paper-shaped supervised dataset.",
            files=("data/processed/tjong/tziakcha_all_sources_botzone_20260605b_tensorized.pt",),
            json_file="runs/tjong_tensorize_all_sources_20260605b_summary.json",
            expectations=(
                JsonExpectation("examples", minimum=1),
                record_count_expectation("stats.matches", expected_records),
                JsonExpectation("corpus_validation.error_count", 0),
            ),
            next_action="apply k8s/tjong-tziakcha-all-sources-prep-cpu-20260605c.yaml",
            alternatives=(("papers/tjong_cit2_12298/artifacts/tjong_tensorize_all_sources_20260605b_summary.json",),),
        ),
        StageSpec(
            name="selfplay_initdata",
            description="Botzone-like logs are converted to official-judge initdata seeds.",
            files=("data/processed/tjong/tziakcha_all_sources_botzone_20260605b_initdata.jsonl",),
            json_file="runs/tjong_validate_tziakcha_all_sources_initdata_20260605b.json",
            expectations=(
                record_count_expectation("records", expected_records),
                JsonExpectation("error_count", 0),
                JsonExpectation("valid_for_selfplay_initdata", True),
            ),
            next_action="apply k8s/tjong-tziakcha-all-sources-prep-cpu-20260605c.yaml",
            alternatives=(("papers/tjong_cit2_12298/artifacts/tjong_validate_tziakcha_all_sources_initdata_20260605b.json",),),
        ),
        StageSpec(
            name="supervised_training",
            description="125-epoch supervised training finishes and deterministic paper metric gate runs.",
            files=(
                "models/tjong_supervised_all_sources_20260605b.pt",
                "runs/tjong_supervised_all_sources_20260605b_metrics.json",
                "runs/tjong_supervised_all_sources_20260605b_eval.json",
                "runs/tjong_supervised_all_sources_20260605b_paper_metric_gate.json",
            ),
            json_file="runs/tjong_supervised_all_sources_20260605b_paper_metric_gate.json",
            expectations=(
                JsonExpectation("passed", True),
                JsonExpectation("metrics_determinism.required", True),
                JsonExpectation("metrics_determinism.passed", True),
                JsonExpectation("metrics_determinism.deterministic_eval", True),
                JsonExpectation("metrics_determinism.strict_deterministic", True),
            ),
            next_action="apply k8s/tjong-supervised-rl-l40-all-sources-20260605b.yaml after corrected tensorization completes",
        ),
        StageSpec(
            name="supervised_checkpoint_sweep",
            description="Final and epoch checkpoints are evaluated deterministically on the same full tensor set.",
            files=("runs/tjong_supervised_all_sources_checkpoint_sweep_20260605b.json",),
            json_file="runs/tjong_supervised_all_sources_checkpoint_sweep_20260605b.json",
            expectations=(JsonExpectation("checkpoint_count", minimum=1),),
            next_action="apply k8s/tjong-supervised-eval-sweep-l40-20260605b.yaml after supervised training",
        ),
        StageSpec(
            name="selfplay_rollout",
            description="Paper-scale official-judge self-play is merged, quality-gated, and tensorized.",
            files=(
                "data/processed/tjong/tjong_selfplay_20260605b_raw.jsonl",
                "data/processed/tjong/tjong_selfplay_20260605b_fan_items.jsonl",
                "data/processed/tjong/tjong_selfplay_20260605b_tensorized.pt",
                "runs/tjong_selfplay_shard_merge_20260605b_summary.json",
            ),
            json_file="runs/tjong_selfplay_shard_merge_20260605b_summary.json",
            expectations=(
                record_count_expectation("games", expected_records),
                JsonExpectation("hu_rate", minimum=0.99),
            ),
            next_action="apply sharded self-play rollout, wait, then apply k8s/tjong-selfplay-merge-cpu-20260605b.yaml",
        ),
        StageSpec(
            name="fan_backward_rollout",
            description="Self-play tensors receive faithful fan-backward rewards and frozen old policy values/log-probs.",
            files=(
                "data/processed/tjong/tjong_selfplay_20260605b_fan_backward_tensorized.pt",
                "data/processed/tjong/tjong_selfplay_20260605b_fan_backward_rollouts.pt",
                "runs/tjong_rollout_build_20260605b_summary.json",
            ),
            json_file="runs/tjong_rollout_build_20260605b_summary.json",
            expectations=(JsonExpectation("actual_reward_source", "fan_backward_reward"),),
            next_action="run the preparation section of k8s/tjong-ppo-l40-20260605b.yaml",
        ),
        StageSpec(
            name="ppo_training",
            description="PPO training consumes faithful fan-backward rollouts and writes the final RL checkpoint.",
            files=(
                "models/tjong_ppo_20260605b.pt",
                "runs/tjong_ppo_20260605b_metrics.json",
            ),
            json_file="runs/tjong_ppo_20260605b_metrics.json",
            expectations=(JsonExpectation("phase", "ppo"),),
            next_action="apply k8s/tjong-ppo-l40-20260605b.yaml after self-play and rollout preparation",
        ),
    ]


def infer_expected_records(repo_root: Path, expected_records: int) -> int:
    if expected_records > 0:
        return int(expected_records)
    for relative_path in (
        "runs/tjong_validate_tziakcha_all_sources_botzone_20260605b.json",
        "papers/tjong_cit2_12298/artifacts/tjong_validate_tziakcha_all_sources_botzone_20260605b.json",
    ):
        path = repo_root / relative_path
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        try:
            records = int(data.get("records", 0))
        except (TypeError, ValueError):
            records = 0
        if records > 0:
            return records
    return 0


def audit_pipeline(repo_root: Path, *, expected_records: int) -> dict[str, Any]:
    expected_records = infer_expected_records(repo_root, expected_records)
    stages = [audit_stage(repo_root, spec) for spec in stage_specs(expected_records)]
    first_incomplete = next((stage for stage in stages if not stage["passed"] and not stage["optional"]), None)
    return {
        "paper": PAPER_NAME,
        "format": "tjong_pipeline_status_v1",
        "repo_root": str(repo_root),
        "expected_records": int(expected_records),
        "passed": first_incomplete is None,
        "first_incomplete_stage": first_incomplete["name"] if first_incomplete else None,
        "next_action": first_incomplete["next_action"] if first_incomplete else None,
        "stages": stages,
    }


def audit_stage(repo_root: Path, spec: StageSpec) -> dict[str, Any]:
    file_checks = [check_file(repo_root, path) for path in spec.files]
    json_check = check_json(repo_root, spec.json_file, spec.expectations) if spec.json_file else None
    alternative_checks = []
    for group in spec.alternatives:
        group_checks = [check_file(repo_root, path) for path in group]
        if group:
            alternative_checks.append({"paths": list(group), "passed": all(item["passed"] for item in group_checks)})
    has_primary = all(item["passed"] for item in file_checks) and (json_check is None or json_check["passed"])
    has_alternative = any(item["passed"] for item in alternative_checks)
    passed = bool(has_primary or has_alternative)
    return {
        "name": spec.name,
        "description": spec.description,
        "passed": passed,
        "optional": spec.optional,
        "next_action": "" if passed else spec.next_action,
        "files": file_checks,
        "json": json_check,
        "alternatives": alternative_checks,
    }


def check_file(repo_root: Path, relative_path: str) -> dict[str, Any]:
    path = repo_root / relative_path
    exists = path.exists()
    size = path.stat().st_size if exists and path.is_file() else None
    return {
        "path": relative_path,
        "exists": exists,
        "size": size,
        "passed": bool(exists and (size is None or size > 0)),
    }


def check_json(repo_root: Path, relative_path: str | None, expectations: tuple[JsonExpectation, ...]) -> dict[str, Any]:
    if relative_path is None:
        return {"passed": True}
    path = repo_root / relative_path
    if not path.exists():
        return {"path": relative_path, "exists": False, "passed": False, "expectations": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"path": relative_path, "exists": True, "passed": False, "error": str(exc), "expectations": []}
    checks = [check_expectation(data, expectation) for expectation in expectations]
    return {
        "path": relative_path,
        "exists": True,
        "passed": all(check["passed"] for check in checks),
        "expectations": checks,
    }


def check_expectation(data: dict[str, Any], expectation: JsonExpectation) -> dict[str, Any]:
    observed = dotted_get(data, expectation.path)
    passed = True
    if expectation.expected is not None:
        passed = observed == expectation.expected
    if expectation.minimum is not None:
        try:
            passed = passed and float(observed) >= float(expectation.minimum)
        except (TypeError, ValueError):
            passed = False
    return {
        "path": expectation.path,
        "observed": observed,
        "expected": expectation.expected,
        "minimum": expectation.minimum,
        "passed": bool(passed),
    }


def dotted_get(data: dict[str, Any], path: str) -> Any:
    current: Any = data
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".", type=Path)
    parser.add_argument("--expected-records", type=int, default=0)
    parser.add_argument("--summary-out", default=None, type=Path)
    parser.add_argument("--fail-if-incomplete", action="store_true")
    args = parser.parse_args()
    summary = audit_pipeline(args.repo_root, expected_records=args.expected_records)
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 1 if args.fail_if_incomplete and not summary["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
