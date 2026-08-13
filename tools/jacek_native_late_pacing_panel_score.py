#!/usr/bin/env python3
"""Score late-pacing checkpoints and the frozen late-trap replay panel.

This tool deliberately keeps three evidence domains separate:

* ``validation`` evaluates every checkpoint on the *same* reconstructed H62
  validation samples.  Candidate-local validation metrics are never compared
  to the incumbent because adding games can move deterministic split bounds.
* ``panel`` compares fixed-work auditor-v3 rows on the frozen 96 trap / 96
  matched-winning-control panel.  Archived moves locate states only; the
  incumbent's own selected root action is the fixed comparison action.
* ``penalty`` independently validates the sparse +2-to-+4 anti-greed predicate
  and checks that the 0.11/0.15 overlays only change the final root argmax.

No command reads CodinGame ``matches.json`` or any protected/sealed bank.
Outputs are canonical, write-once JSON evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import statistics
import sys
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOL_DIR = pathlib.Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))


SCHEMA = "papersoccer.jacek-native-late-pacing-score/v2"
LEGACY_AUDIT_SCHEMA = "jacek-native-decision-audit-v3"
PENALTY_AUDIT_SCHEMA = "jacek-native-decision-audit-v4"
PRIMARY_SUPPORTED_ADVANCE_PENALTY = 0.11
GUARD_SUPPORTED_ADVANCE_PENALTY = 0.15
SUPPORTED_ADVANCE_SPARSE_EDGE_LIMIT = 48
SUPPORTED_ADVANCE_START_PROGRESS = 2
SUPPORTED_ADVANCE_ENDPOINT_PROGRESS_THRESHOLD = 4
MAXIMUM_SUPPORTED_ADVANCE_PENALTY = 0.20
PANEL_SCHEMAS = {
    "papersoccer.jacek-native-late-trap-panel/v1",
    "papersoccer.jacek-native-late-trap-panel/v2",
}
V2_PANEL_SCHEMA = "papersoccer.jacek-native-late-trap-panel/v2"
V2_LEGACY_MISSING_STATE = "fnv1a64:6ebbac0d7afc5221"
EXPECTED_H62 = {
    "games": 22_238,
    "split_games": {"train": 17_779, "validation": 2_230, "test": 2_229},
    "split_samples": {
        "train": 1_755_307,
        "validation": 198_858,
        "test": 197_724,
    },
}
EXPECTED_H62_EARLY_SAMPLES = 21_444
EXPECTED_H62_EARLY_OUTCOME_MSE = 0.9992715120315552
EARLY_OUTCOME_RATIO_LIMIT = 1.02
FLOAT_TOLERANCE = 1e-7
FORBIDDEN_PATH_TOKENS = (
    "matches.json", "protected", "sealed", "prospective", "final-bank",
    "final_bank",
)


def training_dependencies():
    """Load the optional numerical stack only for the validation subcommand."""
    try:
        import numpy as numpy_module
        import train_jacek_native_round2 as trainer_module
    except ModuleNotFoundError as error:
        raise ValueError(
            "validation mode requires the workspace NumPy runtime"
        ) from error
    return numpy_module, trainer_module


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(
        value, sort_keys=True, allow_nan=False, ensure_ascii=False,
        separators=(",", ":"),
    ) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: pathlib.Path) -> str:
    return sha256_bytes(path.read_bytes())


def explicit_file(path: pathlib.Path, label: str) -> pathlib.Path:
    resolved = path.resolve(strict=True)
    rendered = str(resolved).lower()
    if any(token in rendered for token in FORBIDDEN_PATH_TOKENS):
        raise ValueError(f"{label} path contains forbidden evidence: {path}")
    if not resolved.is_file():
        raise ValueError(f"{label} is not an explicit file: {path}")
    return resolved


def write_exclusive(path: pathlib.Path, payload: Mapping[str, Any]) -> None:
    raw = canonical_json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as output:
            output.write(raw)
    except FileExistsError as error:
        raise ValueError(f"refusing to replace existing evidence: {path}") from error


def load_canonical_json(
    path: pathlib.Path, label: str, *, trainer_model: bool = False,
) -> tuple[dict, str]:
    path = explicit_file(path, label)
    raw = path.read_bytes()
    value = json.loads(raw)
    canonical_value = value
    if trainer_model and isinstance(value, dict):
        provenance = value.get("provenance")
        generation = (
            provenance.get("generation")
            if isinstance(provenance, Mapping) else None
        )
        depths = (
            generation.get("opening_depths")
            if isinstance(generation, Mapping) else None
        )
        if isinstance(depths, Mapping):
            normalized_depths: dict[int, Any] = {}
            for key, count in depths.items():
                try:
                    numeric_key = int(key)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"{label} opening-depth key is not canonical"
                    ) from error
                if (
                    not isinstance(key, str)
                    or key != str(numeric_key)
                    or numeric_key < 0
                    or numeric_key in normalized_depths
                ):
                    raise ValueError(
                        f"{label} opening-depth key is not canonical"
                    )
                normalized_depths[numeric_key] = count
            canonical_generation = dict(generation)
            canonical_generation["opening_depths"] = normalized_depths
            canonical_provenance = dict(provenance)
            canonical_provenance["generation"] = canonical_generation
            canonical_value = dict(value)
            canonical_value["provenance"] = canonical_provenance
    if (
        not isinstance(value, dict)
        or canonical_json_bytes(canonical_value) != raw
    ):
        raise ValueError(f"{label} is not canonical JSON: {path}")
    return value, sha256_bytes(raw)


def load_jsonl(path: pathlib.Path, label: str) -> tuple[list[dict], str]:
    path = explicit_file(path, label)
    raw = path.read_bytes()
    rows = []
    for number, line in enumerate(raw.splitlines(), 1):
        if not line:
            raise ValueError(f"blank {label} row at line {number}")
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"non-object {label} row at line {number}")
        rows.append(row)
    if not rows:
        raise ValueError(f"{label} is empty")
    return rows, sha256_bytes(raw)


def row_key(row: Mapping[str, Any]) -> tuple[str, int, str]:
    return str(row["game_id"]), int(row["turn_index"]), str(row["state_id"])


def checkpoint(artifact: Mapping[str, Any], seed: int) -> Mapping[str, Any]:
    matches = [item for item in artifact.get("checkpoints", ())
               if item.get("seed") == seed]
    if len(matches) != 1:
        raise ValueError(f"model does not retain exactly one seed {seed}")
    return matches[0]


def parameters(item: Mapping[str, Any]) -> dict[str, Any]:
    np, _ = training_dependencies()
    model = item.get("model")
    if not isinstance(model, dict) or set(model) != {"w1", "w2", "w3"}:
        raise ValueError("checkpoint model tensors are malformed")
    result = {}
    for name in ("w1", "w2", "w3"):
        tensor = model[name]
        shape = tensor.get("shape")
        values = tensor.get("values")
        if not isinstance(shape, list) or not isinstance(values, list):
            raise ValueError(f"checkpoint {name} tensor is malformed")
        array = np.asarray(values, dtype=np.float32)
        if math.prod(shape) != len(array):
            raise ValueError(f"checkpoint {name} tensor shape is stale")
        result[name] = array.reshape(shape)
    return result


def stored_early_metric(artifact: Mapping[str, Any], seed: int) -> tuple[int, float]:
    reports = [item for item in artifact.get("seed_reports", ())
               if item.get("seed") == seed]
    if len(reports) != 1:
        raise ValueError(f"model does not contain one report for seed {seed}")
    early = reports[0]["quantized_metrics"]["validation"][
        "turn_calibration"
    ]["turns_0_11"]
    return int(early["samples"]), float(early["outcome_mse"])


def load_frozen_h62_datasets(
    current_round2: Sequence[pathlib.Path],
    archived_round1: Sequence[pathlib.Path],
    archived_restart: Sequence[pathlib.Path],
):
    """Load the historical strict-current corpus after live source advances.

    The production loader normally also requires every source and the local
    compiler to equal today's workspace.  That is correct during training but
    impossible for a historical held-out scorer once ``bot.cpp`` advances.
    This narrow wrapper disables only that local-HEAD/compiler comparison.
    The normal loader still validates the canonical strict-current manifests,
    content-addressed source declarations, archived binaries, checkpoint
    payloads, every shard byte/hash, schedules, restart provenance, samples,
    deterministic split assignment and cross-split canonical purge.
    """
    _, trainer = training_dependencies()
    contract = trainer.corpus_contract
    original = contract._validate_round2_build_contract

    def validate_archived(raw, directory, _verify_local_build):
        return original(raw, directory, False)

    contract._validate_round2_build_contract = validate_archived
    try:
        return trainer.load_datasets(
            current_round2,
            archived_round1_paths=archived_round1,
            archived_restart_round2_paths=archived_restart,
        )
    finally:
        contract._validate_round2_build_contract = original


def score_validation(arguments: argparse.Namespace) -> dict:
    _, trainer = training_dependencies()
    baseline, baseline_sha = load_canonical_json(
        arguments.baseline_model, "baseline model", trainer_model=True,
    )
    candidate, candidate_sha = load_canonical_json(
        arguments.candidate_model, "candidate model", trainer_model=True,
    )
    archived_round1 = [explicit_file(path, "archived round-one corpus")
                       for path in arguments.archived_round1]
    current_round2 = [explicit_file(path, "strict-current round-two corpus")
                      for path in arguments.current_round2]
    archived_restart = [explicit_file(path, "archived restart corpus")
                        for path in arguments.archived_restart_round2]
    datasets, report = load_frozen_h62_datasets(
        current_round2, archived_round1, archived_restart
    )
    provenance = baseline.get("provenance") or {}
    for field, expected in EXPECTED_H62.items():
        if report.get(field) != expected or provenance.get(field) != expected:
            raise ValueError(f"frozen H62 {field} does not match its artifact")
    for field in ("corpus_sha256", "source_sha256"):
        if report.get(field) != provenance.get(field):
            raise ValueError(f"frozen H62 {field} does not match its artifact")

    validation = datasets["validation"]
    baseline_parameters = parameters(checkpoint(baseline, arguments.baseline_seed))
    baseline_metrics = trainer.metrics(baseline_parameters, validation)
    baseline_early = baseline_metrics["turn_calibration"]["turns_0_11"]
    stored_samples, stored_mse = stored_early_metric(
        baseline, arguments.baseline_seed
    )
    if (
        int(baseline_early["samples"]) != EXPECTED_H62_EARLY_SAMPLES
        or stored_samples != EXPECTED_H62_EARLY_SAMPLES
        or abs(float(baseline_early["outcome_mse"]) - stored_mse)
        > FLOAT_TOLERANCE
        or abs(stored_mse - EXPECTED_H62_EARLY_OUTCOME_MSE)
        > FLOAT_TOLERANCE
    ):
        raise ValueError("recomputed H62 early metric does not match frozen evidence")

    ceiling = float(baseline_early["outcome_mse"]) * EARLY_OUTCOME_RATIO_LIMIT
    scored = []
    for seed in arguments.candidate_seed:
        item = checkpoint(candidate, seed)
        metrics = trainer.metrics(parameters(item), validation)
        early = metrics["turn_calibration"]["turns_0_11"]
        phase = metrics["phase_metrics"]["turns_0_11"]
        mse = float(early["outcome_mse"])
        scored.append({
            "seed": seed,
            "checkpoint_sha256": item["checkpoint_sha256"],
            "early_samples": int(early["samples"]),
            "early_outcome_mse": mse,
            "early_outcome_mse_ratio": (
                mse / float(baseline_early["outcome_mse"])
            ),
            "early_unweighted_exact_overridden_combined_target_mse": (
                phase["unweighted_combined_target_mse"]
            ),
            "gate_early_outcome_mse": {
                "passed": mse <= ceiling,
                "ceiling": ceiling,
            },
        })
    return {
        "schema": SCHEMA,
        "mode": "frozen-h62-validation",
        "baseline": {
            "model_path": arguments.baseline_model.as_posix(),
            "model_sha256": baseline_sha,
            "seed": arguments.baseline_seed,
            "checkpoint_sha256": checkpoint(
                baseline, arguments.baseline_seed
            )["checkpoint_sha256"],
            "early_samples": int(baseline_early["samples"]),
            "early_outcome_mse": float(baseline_early["outcome_mse"]),
        },
        "candidate_model": {
            "path": arguments.candidate_model.as_posix(),
            "sha256": candidate_sha,
        },
        "frozen_validation": {
            "corpus_sha256": report["corpus_sha256"],
            "source_sha256": report["source_sha256"],
            **EXPECTED_H62,
        },
        "thresholds": {
            "early_outcome_mse_ratio_maximum": EARLY_OUTCOME_RATIO_LIMIT,
            "early_outcome_mse_ceiling": ceiling,
        },
        "candidates": scored,
    }


def panel_records(panel: Mapping[str, Any]):
    if panel.get("schema") not in PANEL_SCHEMAS:
        raise ValueError("late-trap panel schema is unsupported")
    trap_rows = list(panel.get("trap_states", ()))
    control_rows = list(panel.get("matched_winning_controls", ()))
    traps = {str(item["auditor_state_id"]): item for item in trap_rows}
    controls = {str(item["auditor_state_id"]): item for item in control_rows}
    identities_are_unique = len(traps) == 96 and len(controls) == 96
    for field in ("state_id", "canonical_key"):
        trap_ids = {str(item[field]) for item in trap_rows}
        control_ids = {str(item[field]) for item in control_rows}
        identities_are_unique = identities_are_unique and (
            len(trap_ids) == 96
            and len(control_ids) == 96
            and not trap_ids & control_ids
        )
    if not identities_are_unique or set(traps) & set(controls):
        raise ValueError("late-trap panel must contain 96+96 unique states")
    canonical_traps = {str(item["state_id"]): item for item in traps.values()}
    pairs = []
    for control in controls.values():
        trap = canonical_traps.get(str(control["matched_trap_state_id"]))
        if trap is None:
            raise ValueError("matched control references an unknown trap")
        pairs.append((str(trap["auditor_state_id"]),
                      str(control["auditor_state_id"])))
    return traps, controls, pairs


def runtime_identity(path: pathlib.Path) -> dict[str, str]:
    path = explicit_file(path, "runtime")
    raw = path.read_bytes()
    lines = raw.decode("ascii").splitlines()
    if len(lines) != 7 or lines[0] != "papersoccer.jacek-native-runtime-model/v1":
        raise ValueError(f"runtime is malformed: {path}")
    return {
        "runtime_sha256": sha256_bytes(raw),
        "model_sha256": lines[3],
        "packed_weights_sha256": lines[4],
    }


def audit_index(
    rows: Sequence[Mapping[str, Any]], expected_states: set[str],
    runtime: Mapping[str, str], penalty: float,
    required_schema: str = LEGACY_AUDIT_SCHEMA,
) -> dict[str, Mapping[str, Any]]:
    result = {}
    for row in rows:
        if row.get("schema_version") != required_schema:
            raise ValueError(f"panel audit is not {required_schema} evidence")
        if (
            row.get("audit_mode") != "fixed-work"
            or row.get("fixed_work_limit") != 30_000
            or row.get("root_reply_width") != 0
            or abs(float(row.get("supported_advance_penalty", -1)) - penalty)
            > 1e-12
            or row.get("search_deadline_reached")
            or row.get("search_expansion_cap_reached")
            or row.get("search_exact_reply_refuted_actions")
        ):
            raise ValueError("panel audit does not use the frozen safe profile")
        for field, expected in runtime.items():
            if row.get(field) != expected:
                raise ValueError(f"panel audit has wrong {field}")
        state = str(row["state_id"])
        if state in result:
            raise ValueError(f"duplicate panel state: {state}")
        result[state] = row
    if set(result) != expected_states:
        raise ValueError("panel audit does not cover exactly the frozen 192 states")
    return result


def root_actions(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    actions = row.get("search_root_action_diagnostics")
    if not isinstance(actions, list) or len(actions) != row["search_root_actions"]:
        raise ValueError("full root-action diagnostics are incomplete")
    encoded = [str(item["encoded"]) for item in actions]
    if len(encoded) != len(set(encoded)):
        raise ValueError("root-action diagnostics contain duplicate encodings")
    return actions


def action_map(row: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(item["encoded"]): item for item in root_actions(row)}


MODEL_INDEPENDENT_ROW_FIELDS = (
    "game_id", "state_id", "transcript_prefix", "turn_index",
    "own_decision_index", "candidate_player", "winner", "result",
    "audit_mode", "fixed_work_limit", "max_actions", "max_partial_paths",
    "max_expansions", "root_reply_width", "exploration",
    "first_play_urgency", "first_time_limit_ms", "later_time_limit_ms",
    "time_limit_ms", "pre_action_used_edges", "diagnostic_root_actions",
    "diagnostic_root_partial_paths", "diagnostic_root_tactical_proof_paths",
    "diagnostic_root_completed_actions", "diagnostic_root_duplicate_boundaries",
    "diagnostic_root_fifo_extractions", "diagnostic_root_lifo_extractions",
    "diagnostic_root_tactical_actions", "diagnostic_root_tactical_classes_found",
    "diagnostic_root_tactical_proof_truncated", "diagnostic_root_truncations",
    "diagnostic_root_maximum_deque_size", "diagnostic_root_deadline_reached",
    "diagnostic_root_exhaustive", "actual_exact_retained_ordinal",
    "actual_boundary_retained_ordinal", "actual_retained_action",
    "actual_tactical_class",
)


def verify_model_independent(
    baseline: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
) -> None:
    for state, left in baseline.items():
        right = candidate[state]
        for field in MODEL_INDEPENDENT_ROW_FIELDS:
            if left.get(field) != right.get(field):
                raise ValueError(f"model-independent field drift at {state}: {field}")
        left_actions = root_actions(left)
        right_actions = root_actions(right)
        if len(left_actions) != len(right_actions):
            raise ValueError(f"root action count drift at {state}")
        for before, after in zip(left_actions, right_actions):
            for field in ("encoded", "tactical_class", "start_progress",
                          "endpoint_progress"):
                if before.get(field) != after.get(field):
                    raise ValueError(f"root generator drift at {state}: {field}")


def mean(values: Sequence[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def median(values: Sequence[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def fixed_action_values(
    baseline: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
    states: Iterable[str],
) -> list[tuple[str, float, float]]:
    values = []
    for state in states:
        encoded = str(baseline[state]["chosen_action"])
        before = action_map(baseline[state]).get(encoded)
        after = action_map(candidate[state]).get(encoded)
        if before is None or after is None:
            raise ValueError(f"frozen incumbent action is absent at {state}")
        values.append((state, float(before["initial_value"]),
                       float(after["initial_value"])))
    return values


def cohort_delta(
    values: Sequence[tuple[str, float, float]],
    metadata: Mapping[str, Mapping[str, Any]], field: str,
) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for state, before, after in values:
        grouped[str(metadata[state][field])].append(after - before)
    return {key: {"states": len(items), "mean_delta": mean(items)}
            for key, items in sorted(grouped.items())}


def score_one_model_panel(
    label: str,
    baseline: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
    traps: Mapping[str, Mapping[str, Any]],
    controls: Mapping[str, Mapping[str, Any]],
    pairs: Sequence[tuple[str, str]],
) -> dict:
    verify_model_independent(baseline, candidate)
    trap_unsolved = [state for state in traps if not baseline[state]["search_solved"]]
    control_unsolved = [state for state in controls
                        if not baseline[state]["search_solved"]]
    if len(trap_unsolved) != 94 or len(control_unsolved) != 85:
        raise ValueError("frozen exact-solved trap/control inventory changed")
    trap_values = fixed_action_values(baseline, candidate, trap_unsolved)
    control_values = fixed_action_values(baseline, candidate, control_unsolved)

    def signs(values):
        before = sum(left > 0.0 for _, left, _ in values) / len(values)
        after = sum(right > 0.0 for _, _, right in values) / len(values)
        return before, after

    trap_before_sign, trap_after_sign = signs(trap_values)
    control_before_sign, control_after_sign = signs(control_values)
    trap_deltas = [after - before for _, before, after in trap_values]
    control_deltas = [after - before for _, before, after in control_values]

    pair_changes = []
    for trap_state, control_state in pairs:
        trap_encoded = str(baseline[trap_state]["chosen_action"])
        control_encoded = str(baseline[control_state]["chosen_action"])
        bt = float(action_map(baseline[trap_state])[trap_encoded]["initial_value"])
        bc = float(action_map(baseline[control_state])[control_encoded]["initial_value"])
        ct = float(action_map(candidate[trap_state])[trap_encoded]["initial_value"])
        cc = float(action_map(candidate[control_state])[control_encoded]["initial_value"])
        pair_changes.append((cc - ct) - (bc - bt))

    trap_exact_churn = sum(
        baseline[state]["chosen_action"] != candidate[state]["chosen_action"]
        for state in traps
    )
    control_exact_churn = sum(
        baseline[state]["chosen_action"] != candidate[state]["chosen_action"]
        for state in controls
    )
    trap_boundary_churn = sum(
        baseline[state]["chosen_boundary_retained_ordinal"]
        != candidate[state]["chosen_boundary_retained_ordinal"]
        for state in traps
    )
    control_boundary_churn = sum(
        baseline[state]["chosen_boundary_retained_ordinal"]
        != candidate[state]["chosen_boundary_retained_ordinal"]
        for state in controls
    )

    solved_trap_losses = [state for state in traps
                          if baseline[state]["search_solved"] and
                          baseline[state]["search_solved_winner"]
                          != baseline[state]["candidate_player"]]
    solved_trap_wins = [state for state in traps
                        if baseline[state]["search_solved"] and
                        baseline[state]["search_solved_winner"]
                        == baseline[state]["candidate_player"]]
    solved_control_wins = [state for state in controls
                           if baseline[state]["search_solved"] and
                           baseline[state]["search_solved_winner"]
                           == baseline[state]["candidate_player"]]
    mate_loss_actions = [state for state in traps
                         if baseline[state].get("chosen_final_backed_value")
                         is not None and
                         float(baseline[state]["chosen_final_backed_value"]) < -100]
    baseline_proven_actions = []
    for state in baseline:
        for action in root_actions(baseline[state]):
            if action["proven_win"]:
                baseline_proven_actions.append((state, str(action["encoded"])))
    exact_signals_ok = (
        len(solved_trap_losses) == 1
        and all(candidate[state]["search_solved"] and
                candidate[state]["search_solved_winner"]
                != candidate[state]["candidate_player"]
                for state in solved_trap_losses)
        and len(solved_trap_wins) == 1
        and all(candidate[state]["search_solved"] and
                candidate[state]["search_solved_winner"]
                == candidate[state]["candidate_player"]
                for state in solved_trap_wins)
        and len(solved_control_wins) == 11
        and all(candidate[state]["search_solved"] and
                candidate[state]["search_solved_winner"]
                == candidate[state]["candidate_player"]
                for state in solved_control_wins)
        and len(mate_loss_actions) == 2
        and all(
            float(action_map(candidate[state])[
                str(baseline[state]["chosen_action"])
            ]["value"]) < -100
            for state in mate_loss_actions
        )
        and all(action_map(candidate[state])[encoded]["proven_win"]
                for state, encoded in baseline_proven_actions)
        and all(
            not any(action["proven_win"] for action in root_actions(candidate[state]))
            or action_map(candidate[state])[
                str(candidate[state]["chosen_action"])
            ]["proven_win"]
            for state in candidate
        )
    )

    role_changes = Counter()
    color_changes = Counter()
    for state, meta in traps.items():
        if baseline[state]["chosen_action"] != candidate[state]["chosen_action"]:
            role_changes[str(meta["role"])] += 1
            color_changes[str(meta["candidate_player"])] += 1

    hard = {
        "lost_trap_false_positive_sign_rate_improvement": {
            "passed": trap_before_sign - trap_after_sign >= 0.25,
            "required_absolute_improvement": 0.25,
            "baseline": trap_before_sign,
            "candidate": trap_after_sign,
        },
        "matched_control_positive_classification_retention": {
            "passed": control_before_sign - control_after_sign <= 0.03,
            "maximum_absolute_degradation": 0.03,
            "baseline": control_before_sign,
            "candidate": control_after_sign,
        },
        "exact_proof_retention": {"passed": exact_signals_ok},
    }
    secondary = {
        "trap_mean_delta": {
            "passed": mean(trap_deltas) <= -0.15,
            "maximum": -0.15,
            "actual": mean(trap_deltas),
        },
        "control_mean_delta": {
            "passed": mean(control_deltas) >= -0.05,
            "minimum": -0.05,
            "actual": mean(control_deltas),
        },
        "paired_gap": {
            "passed": mean(pair_changes) >= 0.10
            and median(pair_changes) >= 0.05
            and sum(value > 0 for value in pair_changes) >= 58,
            "mean_improvement_minimum": 0.10,
            "median_improvement_minimum": 0.05,
            "improved_pairs_minimum": 58,
            "mean_improvement": mean(pair_changes),
            "median_improvement": median(pair_changes),
            "improved_pairs": sum(value > 0 for value in pair_changes),
        },
        "action_churn": {
            "passed": trap_exact_churn >= 12 and control_exact_churn <= 24
            and (trap_exact_churn - control_exact_churn) / 96 >= 0.08,
            "trap_exact": trap_exact_churn,
            "control_exact": control_exact_churn,
            "trap_boundary": trap_boundary_churn,
            "control_boundary": control_boundary_churn,
            "trap_role_changes": dict(sorted(role_changes.items())),
            "trap_color_changes": dict(sorted(color_changes.items())),
        },
    }
    return {
        "label": label,
        "hard_gates": hard,
        "hard_passed": all(item["passed"] for item in hard.values()),
        "secondary_gates": secondary,
        "secondary_passed": all(item["passed"] for item in secondary.values()),
        "fixed_incumbent_action": {
            "trap_states": len(trap_values),
            "control_states": len(control_values),
            "trap_mean_delta": mean(trap_deltas),
            "control_mean_delta": mean(control_deltas),
            "trap_false_positive_sign_rate": trap_after_sign,
            "control_positive_classification_rate": control_after_sign,
            "trap_delta_by_role": cohort_delta(
                trap_values, traps, "role"
            ),
            "trap_delta_by_color": cohort_delta(
                trap_values, traps, "candidate_player"
            ),
        },
        "proof_inventory": {
            "baseline_root_solved_trap_losses": len(solved_trap_losses),
            "baseline_root_solved_trap_wins": len(solved_trap_wins),
            "baseline_root_solved_control_wins": len(solved_control_wins),
            "baseline_chosen_mate_loss_actions": len(mate_loss_actions),
            "baseline_proven_win_actions": len(baseline_proven_actions),
            "retained": exact_signals_ok,
        },
    }


def parse_candidate(values: Sequence[Sequence[str]]):
    result = []
    labels = set()
    for label, runtime, audit in values:
        if not label or label in labels:
            raise ValueError("candidate labels must be unique and nonempty")
        labels.add(label)
        result.append((label, pathlib.Path(runtime), pathlib.Path(audit)))
    if not result:
        raise ValueError("at least one candidate audit is required")
    return result


def verify_legacy_equivalence(
    panel: Mapping[str, Any],
    legacy_rows: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    expected = set(baseline)
    rows_by_state: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in legacy_rows:
        rows_by_state[str(row["state_id"])].append(row)
    covered = set(rows_by_state)
    missing = expected - covered
    extras = covered - expected
    if panel.get("schema") == V2_PANEL_SCHEMA:
        required_missing = {V2_LEGACY_MISSING_STATE}
        if (
            len(covered) != 191
            or missing != required_missing
            or extras
        ):
            raise ValueError(
                "legacy K0 audit must cover the exact 191-state v2 subset"
            )
    elif covered != expected:
        raise ValueError("legacy K0 audit does not cover the frozen panel")

    ignored = {"schema_version", "search_elapsed_ms"}

    def matches(old: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
        for field, value in old.items():
            if field in ignored:
                continue
            actual = current.get(field)
            if field == "initial_best_value":
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or isinstance(actual, bool)
                    or not isinstance(actual, (int, float))
                    or not math.isfinite(float(value))
                    or not math.isfinite(float(actual))
                    or value != float(format(actual, ".9g"))
                ):
                    return False
            elif actual != value:
                return False
        return True

    for state in sorted(covered):
        matching_rows = [
            row for row in rows_by_state[state]
            if matches(row, baseline[state])
        ]
        if len(matching_rows) != 1:
            raise ValueError(f"auditor-v3 K0 legacy drift at {state}")
    return {
        "expected_states": len(expected),
        "covered_unique_states": len(covered),
        "compared_states": len(covered),
        "legacy_rows": len(legacy_rows),
        "missing_states": sorted(missing),
    }


def score_panel(arguments: argparse.Namespace) -> dict:
    panel, panel_sha = load_canonical_json(arguments.panel, "late-trap panel")
    traps, controls, pairs = panel_records(panel)
    expected = set(traps) | set(controls)
    baseline_rows, baseline_sha = load_jsonl(arguments.baseline_audit,
                                               "baseline audit")
    baseline_runtime = runtime_identity(arguments.baseline_runtime)
    baseline = audit_index(baseline_rows, expected, baseline_runtime, 0.0)

    legacy_coverage = None
    if arguments.legacy_baseline is not None:
        legacy_rows, legacy_sha = load_jsonl(arguments.legacy_baseline,
                                              "legacy baseline audit")
        legacy_coverage = verify_legacy_equivalence(
            panel, legacy_rows, baseline
        )
    else:
        legacy_sha = None

    scored = []
    candidates = parse_candidate(arguments.candidate)
    for label, runtime_path, audit_path in candidates:
        rows, audit_sha = load_jsonl(audit_path, f"{label} audit")
        identity = runtime_identity(runtime_path)
        indexed = audit_index(rows, expected, identity, 0.0)
        score = score_one_model_panel(
            label, baseline, indexed, traps, controls, pairs
        )
        score["evidence"] = {
            "runtime_path": runtime_path.as_posix(),
            **identity,
            "audit_path": audit_path.as_posix(),
            "audit_sha256": audit_sha,
        }
        scored.append(score)
    return {
        "schema": SCHEMA,
        "mode": "frozen-late-trap-model-panel",
        "panel": {"path": arguments.panel.as_posix(), "sha256": panel_sha},
        "baseline": {
            "runtime_path": arguments.baseline_runtime.as_posix(),
            **baseline_runtime,
            "audit_path": arguments.baseline_audit.as_posix(),
            "audit_sha256": baseline_sha,
            "legacy_k0_path": (
                arguments.legacy_baseline.as_posix()
                if arguments.legacy_baseline else None
            ),
            "legacy_k0_sha256": legacy_sha,
            "legacy_k0_coverage": legacy_coverage,
        },
        "inventory": {
            "trap_states": len(traps),
            "matched_winning_controls": len(controls),
            "pairs": len(pairs),
            "trap_roles": dict(sorted(Counter(
                str(item["role"]) for item in traps.values()
            ).items())),
        },
        "candidates": scored,
    }


def independently_eligible(
    pre_action_used_edges: int, action: Mapping[str, Any]
) -> bool:
    return (
        pre_action_used_edges < SUPPORTED_ADVANCE_SPARSE_EDGE_LIMIT
        and int(action["start_progress"]) == SUPPORTED_ADVANCE_START_PROGRESS
        and int(action["endpoint_progress"])
        >= SUPPORTED_ADVANCE_ENDPOINT_PROGRESS_THRESHOLD
        and action["tactical_class"] == "safe-handoff"
        and not bool(action["solved"])
        and not bool(action["proven_win"])
    )


def offline_argmax(row: Mapping[str, Any], penalty: float) -> str:
    actions = root_actions(row)
    best = None
    best_score = -math.inf
    for action in actions:
        score = float(action["unpenalized_final_score"])
        if independently_eligible(int(row["pre_action_used_edges"]), action):
            score -= penalty
        encoded = str(action["encoded"])
        if best is None or score > best_score or (
            score == best_score and encoded < best
        ):
            best = encoded
            best_score = score
    assert best is not None
    return best


def verify_penalty_overlay(
    zero: Mapping[str, Mapping[str, Any]],
    overlay: Mapping[str, Mapping[str, Any]], penalty: float,
) -> None:
    for state, before in zero.items():
        after = overlay[state]
        for field in MODEL_INDEPENDENT_ROW_FIELDS:
            if before.get(field) != after.get(field):
                raise ValueError(f"penalty changed root input at {state}: {field}")
        left = root_actions(before)
        right = root_actions(after)
        if len(left) != len(right):
            raise ValueError(f"penalty changed root action count at {state}")
        for base_action, action in zip(left, right):
            for field in (
                "encoded", "value", "initial_value", "visits",
                "selection_visits", "tactical_class", "solved",
                "proven_win", "exact_reply_refuted", "start_progress",
                "endpoint_progress", "unpenalized_final_score",
            ):
                if base_action.get(field) != action.get(field):
                    raise ValueError(
                        f"penalty changed search at {state}: {field}"
                    )
            expected = independently_eligible(
                int(before["pre_action_used_edges"]), action
            )
            if bool(action["supported_advance_eligible"]) != expected:
                raise ValueError(f"telemetry eligibility mismatch at {state}")
            applied = float(action["supported_advance_penalty"])
            if abs(applied - (penalty if expected else 0.0)) > 1e-12:
                raise ValueError(f"wrong supported-advance penalty at {state}")
            expected_score = float(action["unpenalized_final_score"]) - applied
            if abs(float(action["penalized_final_score"]) - expected_score) > 1e-9:
                raise ValueError(f"wrong penalized score at {state}")
        if after["chosen_action"] != offline_argmax(before, penalty):
            raise ValueError(f"penalty selection differs from offline argmax: {state}")


def score_penalty(arguments: argparse.Namespace) -> dict:
    panel, panel_sha = load_canonical_json(arguments.panel, "late-trap panel")
    traps, controls, _ = panel_records(panel)
    expected = set(traps) | set(controls)
    runtime = runtime_identity(arguments.runtime)
    zero_rows, zero_sha = load_jsonl(arguments.zero_audit, "zero-penalty audit")
    primary_rows, primary_sha = load_jsonl(
        arguments.penalty_11_audit, "0.11 audit"
    )
    guard_rows, guard_sha = load_jsonl(
        arguments.penalty_15_audit, "0.15 audit"
    )
    zero = audit_index(
        zero_rows, expected, runtime, 0.0, PENALTY_AUDIT_SCHEMA
    )
    primary = audit_index(
        primary_rows, expected, runtime,
        PRIMARY_SUPPORTED_ADVANCE_PENALTY, PENALTY_AUDIT_SCHEMA,
    )
    guard = audit_index(
        guard_rows, expected, runtime,
        GUARD_SUPPORTED_ADVANCE_PENALTY, PENALTY_AUDIT_SCHEMA,
    )
    for rows in (zero, primary, guard):
        for state, row in rows.items():
            if (
                row.get("supported_advance_sparse_edge_limit")
                != SUPPORTED_ADVANCE_SPARSE_EDGE_LIMIT
                or row.get("supported_advance_start_progress")
                != SUPPORTED_ADVANCE_START_PROGRESS
                or row.get("supported_advance_endpoint_progress_threshold")
                != SUPPORTED_ADVANCE_ENDPOINT_PROGRESS_THRESHOLD
                or abs(float(row.get("maximum_supported_advance_penalty", -1))
                       - MAXIMUM_SUPPORTED_ADVANCE_PENALTY) > 1e-12
            ):
                raise ValueError(
                    f"penalty audit has wrong anti-greed contract at {state}"
                )
    # Penalty-zero telemetry must itself be independently exact.
    for state, row in zero.items():
        for action in root_actions(row):
            if bool(action["supported_advance_eligible"]) != independently_eligible(
                int(row["pre_action_used_edges"]), action
            ):
                raise ValueError(f"zero-penalty eligibility mismatch at {state}")
            if float(action["supported_advance_penalty"]) != 0.0:
                raise ValueError(f"zero-penalty audit applied a penalty at {state}")
        if row["chosen_action"] != offline_argmax(row, 0.0):
            raise ValueError(f"zero-penalty selection differs from argmax: {state}")
    verify_penalty_overlay(
        zero, primary, PRIMARY_SUPPORTED_ADVANCE_PENALTY
    )
    verify_penalty_overlay(zero, guard, GUARD_SUPPORTED_ADVANCE_PENALTY)

    eligible_actions = {kind: 0 for kind in ("trap", "control")}
    eligible_roots = {kind: 0 for kind in ("trap", "control")}
    eligible_selected = {kind: 0 for kind in ("trap", "control")}
    for kind, states in (("trap", traps), ("control", controls)):
        for state in states:
            eligible = [action for action in root_actions(zero[state])
                        if independently_eligible(
                            int(zero[state]["pre_action_used_edges"]), action
                        )]
            eligible_actions[kind] += len(eligible)
            eligible_roots[kind] += bool(eligible)
            eligible_selected[kind] += any(
                action["encoded"] == zero[state]["chosen_action"]
                for action in eligible
            )

    def overlay_report(rows: Mapping[str, Mapping[str, Any]], penalty: float):
        trap_changes = [state for state in traps
                        if rows[state]["chosen_action"]
                        != zero[state]["chosen_action"]]
        control_changes = [state for state in controls
                           if rows[state]["chosen_action"]
                           != zero[state]["chosen_action"]]
        jacek_traps = [state for state, meta in traps.items()
                       if meta["opponent"]["name"] == "jacek" and any(
                           action["encoded"] == zero[state]["chosen_action"]
                           and independently_eligible(
                               int(zero[state]["pre_action_used_edges"]), action
                           ) for action in root_actions(zero[state])
                       )]
        changed_jacek = [state for state in jacek_traps if state in trap_changes]
        changed_colors = sorted({
            int(traps[state]["candidate_player"]) for state in changed_jacek
        })
        safe_changes = all(
            not action_map(rows[state])[rows[state]["chosen_action"]][
                "exact_reply_refuted"
            ]
            for state in trap_changes + control_changes
        )
        gates = {
            "trap_intervention": {
                "passed": len(trap_changes) == 2,
                "changed": len(trap_changes), "required": 2,
            },
            "jacek_intervention": {
                "passed": len(jacek_traps) == 2 and len(changed_jacek) == 2,
                "eligible_selected": len(jacek_traps),
                "changed": len(changed_jacek), "required": 2,
            },
            "both_colors": {
                "passed": changed_colors == [0, 1],
                "changed_candidate_players": changed_colors,
                "required_candidate_players": [0, 1],
            },
            "control_retention": {
                "passed": len(control_changes) == 0,
                "changed": len(control_changes), "required": 0,
                "retained": 96, "required_retained": 96,
            },
            "no_exact_refutation": {
                "passed": safe_changes,
            },
        }
        return {
            "penalty": penalty,
            "trap_changed_states": sorted(trap_changes),
            "control_changed_states": sorted(control_changes),
            "gates": gates,
            "passed": all(item["passed"] for item in gates.values()),
        }

    report_primary = overlay_report(
        primary, PRIMARY_SUPPORTED_ADVANCE_PENALTY
    )
    report_guard = overlay_report(guard, GUARD_SUPPORTED_ADVANCE_PENALTY)
    primary_changes = set(report_primary["trap_changed_states"])
    primary_changes.update(report_primary["control_changed_states"])
    guard_changes = set(report_guard["trap_changed_states"])
    guard_changes.update(report_guard["control_changed_states"])
    identical = primary_changes == guard_changes
    selected = (
        PRIMARY_SUPPORTED_ADVANCE_PENALTY
        if report_primary["passed"] and report_guard["passed"] and identical
        else None
    )
    return {
        "schema": SCHEMA,
        "mode": "frozen-supported-advance-panel",
        "panel": {"path": arguments.panel.as_posix(), "sha256": panel_sha},
        "runtime": {"path": arguments.runtime.as_posix(), **runtime},
        "audits": {
            "0": {"path": arguments.zero_audit.as_posix(), "sha256": zero_sha},
            "0.11": {"path": arguments.penalty_11_audit.as_posix(),
                     "sha256": primary_sha},
            "0.15": {"path": arguments.penalty_15_audit.as_posix(),
                     "sha256": guard_sha},
        },
        "eligibility": {
            "independent_predicate": (
                "pre_action_used_edges<48 && start_progress==2 && "
                "endpoint_progress>=4 && tactical_class==safe-handoff && "
                "!solved && !proven_win"
            ),
            "maximum_penalty": MAXIMUM_SUPPORTED_ADVANCE_PENALTY,
            "eligible_actions": eligible_actions,
            "eligible_roots": eligible_roots,
            "eligible_selected_roots": eligible_selected,
        },
        "overlays": [report_primary, report_guard],
        "change_sets_identical": identical,
        "selected_penalty": selected,
        "qualified": selected is not None,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="mode", required=True)

    validation = subparsers.add_parser("validation")
    validation.add_argument("--baseline-model", type=pathlib.Path, required=True)
    validation.add_argument("--baseline-seed", type=int, required=True)
    validation.add_argument("--candidate-model", type=pathlib.Path, required=True)
    validation.add_argument("--candidate-seed", type=int, action="append",
                            required=True)
    validation.add_argument("--archived-round1", type=pathlib.Path, nargs="+",
                            required=True)
    validation.add_argument("--current-round2", type=pathlib.Path, nargs="+",
                            required=True)
    validation.add_argument("--archived-restart-round2", type=pathlib.Path,
                            nargs="+", required=True)
    validation.add_argument("--output", type=pathlib.Path, required=True)

    panel = subparsers.add_parser("panel")
    panel.add_argument("--panel", type=pathlib.Path, required=True)
    panel.add_argument("--baseline-runtime", type=pathlib.Path, required=True)
    panel.add_argument("--baseline-audit", type=pathlib.Path, required=True)
    panel.add_argument("--legacy-baseline", type=pathlib.Path)
    panel.add_argument("--candidate", nargs=3, action="append", metavar=(
        "LABEL", "RUNTIME", "AUDIT"), required=True)
    panel.add_argument("--output", type=pathlib.Path, required=True)

    penalty = subparsers.add_parser("penalty")
    penalty.add_argument("--panel", type=pathlib.Path, required=True)
    penalty.add_argument("--runtime", type=pathlib.Path, required=True)
    penalty.add_argument("--zero-audit", type=pathlib.Path, required=True)
    penalty.add_argument("--penalty-11-audit", type=pathlib.Path, required=True)
    penalty.add_argument("--penalty-15-audit", type=pathlib.Path, required=True)
    penalty.add_argument("--output", type=pathlib.Path, required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        if arguments.mode == "validation":
            report = score_validation(arguments)
        elif arguments.mode == "panel":
            report = score_panel(arguments)
        else:
            report = score_penalty(arguments)
        write_exclusive(arguments.output, report)
        print(f"wrote {arguments.output} sha256={sha256(arguments.output)}")
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
