#!/usr/bin/env python3
"""Prepare the fresh 512-game representation pilot; run bounded mechanical fits.

Production game collection is deliberately a separate root checkpoint. This
module never reads old corpora, holdouts, or public/live replays and never starts
game collection. Mechanical teacher predictions exercise training plumbing,
not search-quality supervision or a qualification claim.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time
import base64

for _key in ("MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"
os.environ["PAPERSOCCER_COMPACT_TRAINING_THREADS_FIXED_BEFORE_NUMPY"] = "1"
import numpy as np

import compact_representation as representation
import compact_value_bfm_train as compact
import jacek_replay_train as teacher_module
import jacek_replay_features as features

TEACHER_SHA256 = "f7bdb201a377c04531f1ba98fd73457f7f77961aa0f0f9b1ac32c59b6e85ee75"
SCHEMA = "papersoccer.compact-representation.pilot.v1"
DEFAULT_TEACHER = representation.ROOT / "results/top_three_20260916/controls/teacher/model.runtime"


def record(path: Path) -> dict:
    data = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def teacher(path: Path):
    identity = record(path)
    if identity["sha256"] != TEACHER_SHA256 or identity["bytes"] != 4864000:
        raise ValueError("accepted teacher identity mismatch")
    parameters, metadata = teacher_module.load_runtime(path)
    if metadata["artifact_sha256"] != TEACHER_SHA256: raise ValueError("loaded teacher mismatch")
    return parameters, identity


def prepare(path: Path, teacher_path: Path) -> dict:
    # Hashing suffices for a plan; load/parse is checked before any predictions.
    identity = record(teacher_path)
    if identity["sha256"] != TEACHER_SHA256: raise ValueError("accepted teacher mismatch")
    plan = {"schema": SCHEMA, "stage": "prepared-awaiting-root-checkpoint",
            "game_collection_enabled": False, "teacher": identity,
            "new_games": 512, "root_groups": 256, "colors_per_root": 2,
            "split": {"train_root_groups": 128, "validation_root_groups": 128,
                      "train_games": 256, "validation_games": 256},
            "minimum_canonical_independent_early_validation_groups": 100,
            "early_max_edges": 12, "whole_trajectory_split_isolation": True,
            "require_exhaustive_successors_for_validation_regret": True,
            "profiles": list(representation.PROFILES), "ranking_weights": [0.0, 0.1],
            "matched_seed": 2026091604, "max_workers": 1,
            "production_training_input": "fresh-generated-game-linked-complete-turn-search-labels-only",
            "old_or_live_corpora_allowed": False, "mechanical_predictions_are_production_labels": False,
            "training_trajectory_schema": "papersoccer.top-three.training-game.v1",
            "fresh_game_admission": "fail on insufficient independent early groups; do not count the two colors as two roots",
            "source": record(Path(__file__)), "format_source": record(Path(representation.__file__))}
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = representation.canonical(plan)
    if path.exists() and path.read_bytes() != payload: raise ValueError("pilot plan already frozen with another identity")
    path.write_bytes(payload)
    return plan


def tensors(runtime: dict):
    values, hidden, width = representation.validate(runtime)
    architecture = compact.Architecture("representation-" + runtime["profile"], hidden, 8, "isolated-research")
    flat = np.asarray(values, dtype=np.int8)
    counts = [6301 * hidden, hidden * 8, 8]
    cuts = np.cumsum([0, *counts])
    integer = {name: flat[cuts[index]:cuts[index + 1]].reshape(architecture.shapes[name]).copy()
               for index, name in enumerate(("w1", "w2", "w3"))}
    scales = {name: np.float32(scale) for name, scale in zip(("w1", "w2", "w3"), runtime["scales"])}
    return architecture, compact.QuantizedWeights(integer, scales), width


def quantize(parameters, scales, width):
    bound = (1 << (width - 1)) - 1
    integer = {name: np.clip(np.rint(values / scales[name]), -bound, bound).astype(np.int8)
               for name, values in parameters.items()}
    return compact.QuantizedWeights(integer, scales)


def runtime_from_quantized(profile, quantized, lineage):
    values = np.concatenate([quantized.integer[name].ravel() for name in ("w1", "w2", "w3")]).astype(int).tolist()
    return representation.document(profile, values, [float(quantized.scales[name]) for name in ("w1", "w2", "w3")], lineage)


def parameter_digest(parameters):
    return hashlib.sha256(b"".join(np.asarray(parameters[name], dtype="<f4").tobytes() for name in ("w1", "w2", "w3"))).hexdigest()


def feature_key(indices):
    normal = features.validate_active(indices)
    mirrored = features.reflect_active(normal)
    return hashlib.sha256(np.asarray(min(normal, mirrored), dtype="<u2").tobytes()).hexdigest()


def ranking_loss_gradient(predictions, targets, signs):
    """Best-teacher-versus-all-rivals logistic loss in the parent's frame."""
    values = predictions * signs
    teacher_values = targets * signs
    best = int(np.argmax(teacher_values))
    rivals = [index for index in range(len(values)) if teacher_values[best] - teacher_values[index] > 1e-4]
    gradient = np.zeros_like(values)
    if not rivals: return 0.0, gradient
    loss = 0.0
    temperature = np.float32(0.2)
    for rival in rivals:
        delta = float((values[best] - values[rival]) / temperature)
        loss += float(np.logaddexp(0.0, -delta)) / len(rivals)
        derivative = -float(1.0 / (1.0 + np.exp(np.clip(delta, -80, 80)))) / float(temperature) / len(rivals)
        gradient[best] += derivative * signs[best]
        gradient[rival] -= derivative * signs[rival]
    return loss, gradient


def fit(groups, *, profile, ranking_weight, steps, seed, teacher_identity):
    runtime = representation.initialize(profile)
    architecture, quantized, width = tensors(runtime)
    parameters = quantized.effective()
    scales = {name: np.float32(scale / (2 if width == 4 else 1)) for name, scale in quantized.scales.items()}
    if architecture.hidden_one == 16:
        # Zero incoming AND outgoing weights would permanently kill the new
        # units. Clone four incoming columns and retain zero outgoing rows.
        parameters["w1"][:, 12:16] = parameters["w1"][:, :4]
    initial = parameter_digest(parameters)
    # Intentionally aggressive mechanical-only rate: force code transitions in
    # a handful of updates rather than mistaking float-only changes for a
    # exercised quantized training path. It is not a production recipe.
    optimizer = compact.AdamW(parameters, learning_rate=0.02, weight_decay=0.0)
    initial_integer = {name: value.copy() for name, value in quantize(parameters, scales, width).integer.items()}
    rng = np.random.default_rng(seed)
    train = [group for group in groups if group["split"] == "train"]
    if not train: raise ValueError("training groups absent")
    losses = []
    for step in range(steps):
        group = train[int(rng.integers(len(train)))]
        active = [np.asarray(row["active"], dtype=np.uint16) for row in group["successors"]]
        targets = np.asarray([row["teacher_value"] for row in group["successors"]], dtype=np.float32)
        signs = np.asarray([1 if row["value_mover"] == group["mover"] else -1 for row in group["successors"]], dtype=np.float32)
        quantized = quantize(parameters, scales, width)
        predictions, cache = compact.forward(parameters, architecture, active, quantized=quantized)
        scalar_loss, gradient = compact._weighted_huber_loss_gradient(predictions, targets, np.ones(len(active), dtype=np.float32))
        ranking_loss = 0.0
        if ranking_weight:
            ranking_loss, ranking_gradient = ranking_loss_gradient(predictions, targets, signs)
            gradient += np.float32(ranking_weight) * ranking_gradient
        gradients = compact._network_gradients(parameters, architecture, active, cache, gradient, quantized.effective())
        for name in gradients:
            np.clip(gradients[name], -5, 5, out=gradients[name])
        optimizer.update(parameters, gradients)
        losses.append({"scalar": scalar_loss, "ranking": ranking_loss})
    quantized = quantize(parameters, scales, width)
    final = parameter_digest(parameters)
    if initial == final: raise ValueError("mechanical fit performed no parameter update")
    lineage = {"kind": "mechanical-training-smoke", "production_eligible": False,
               "teacher_runtime_sha256": teacher_identity["sha256"], "seed": seed,
               "ranking_weight": ranking_weight, "optimizer_steps": steps,
               "mechanical_learning_rate": 0.02,
               "changed_integer_weights": sum(int(np.count_nonzero(quantized.integer[name] != initial_integer[name])) for name in initial_integer),
               "initial_parameters_sha256": initial, "final_parameters_sha256": final,
               "h16_extra_units": "cloned-input-zero-output" if architecture.hidden_one == 16 else None,
               "label_type": "static-accepted-teacher-prediction-not-search"}
    return runtime_from_quantized(profile, quantized, lineage), {**lineage, "losses": losses}


def validate_group_isolation(groups):
    states = {"train": set(), "validation": set()}
    roots = {"train": set(), "validation": set()}
    early = set()
    for group in groups:
        split = group["split"]
        if split not in states: raise ValueError("only train/validation splits are accepted")
        roots[split].add(group["root_group_id"])
        for row in group["successors"]:
            indices = row["active"]
            if indices != sorted(set(indices)) or not indices or indices[-1] >= 6301 or indices[0] < 0:
                raise ValueError("invalid active features")
            states[split].add(feature_key(indices))
        if split == "validation" and group["edges"] <= 12 and group["exhaustive"] and len(group["successors"]) > 1:
            early.add(feature_key(group["parent_active"]))
    if roots["train"] & roots["validation"] or states["train"] & states["validation"]:
        raise ValueError("whole-root or canonical active-feature overlap across splits")
    return {"train_roots": len(roots["train"]), "validation_roots": len(roots["validation"]),
            "early_validation_root_groups": len(early)}


def smoke(output: Path, teacher_path: Path, feature_probe: Path, *, steps=8):
    if not 1 <= steps <= 32: raise ValueError("mechanical step limit")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    large, identity = teacher(teacher_path)
    process = subprocess.run([str(feature_probe.resolve()), "--groups", "32"], check=True,
                             capture_output=True, text=True, timeout=60)
    raw = [json.loads(line) for line in process.stdout.splitlines()]
    # Mechanical rows are legal states, but not a new game corpus. Keep each
    # state fingerprint in one split and purge cross-split feature overlap.
    groups, seen = [], set()
    for index, group in enumerate(raw):
        if len(group["successors"]) < 2 or group["state_hash"] in seen: continue
        seen.add(group["state_hash"])
        group["root_group_id"] = str(group["state_hash"])
        group["split"] = "validation" if index % 4 == 0 else "train"
        active = [np.asarray(row["active"], dtype=np.uint16) for row in group["successors"]]
        values, _ = teacher_module.forward(large, active)
        for row, value in zip(group["successors"], values): row["teacher_value"] = float(value)
        groups.append(group)
    training_features = {feature_key(row["active"]) for group in groups if group["split"] == "train" for row in group["successors"]}
    groups = [group for group in groups if group["split"] == "train" or all(feature_key(row["active"]) not in training_features for row in group["successors"])]
    coverage = validate_group_isolation(groups)
    corpus = {"schema": SCHEMA, "classification": "mechanical-only-not-production-data", "teacher": identity,
              "feature_probe": record(feature_probe), "groups": groups, "coverage": coverage}
    (output / "mechanical-groups.json").write_bytes(representation.canonical(corpus))
    reports = []
    for profile in representation.PROFILES:
        for weight in (0.0, 0.1):
            runtime, trained = fit(groups, profile=profile, ranking_weight=weight, steps=steps,
                                   seed=2026091604, teacher_identity=identity)
            source, exported = representation.export(runtime)
            stem = f"{profile}-{'scalar' if weight == 0 else 'ranking'}"
            (output / (stem + ".runtime.json")).write_bytes(representation.canonical(runtime))
            (output / (stem + ".cpp")).write_bytes(source)
            reports.append({"name": stem, "training": trained, "export": exported})
    report = {"schema": SCHEMA, "stage": "mechanical-smoke-complete", "production_eligible": False,
              "teacher": identity, "coverage": coverage, "models": reports,
              "elapsed_seconds": time.perf_counter() - started, "native_parity_verified": False,
              "new_games_generated": 0, "formal_early_coverage_pass": False}
    (output / "smoke.json").write_bytes(representation.canonical(report))
    return report


def native_parity(runtime, probe: Path, active):
    values, hidden, width = representation.validate(runtime)
    compressed, lengths, _ = representation.compress(values, width)
    architecture, quantized, _ = tensors(runtime)
    header = " ".join([str(hidden), str(width), *(format(scale, ".9g") for scale in runtime["scales"]),
                       base64.b64encode(compressed).decode("ascii"), runtime["payload_sha256"],
                       *map(str, lengths)])
    rows = "".join(str(len(row)) + " " + " ".join(map(str, row)) + "\n" for row in active)
    process = subprocess.run([str(probe.resolve()), "--predict-model"], input=header + "\n" + rows,
                             text=True, capture_output=True, check=True, timeout=120)
    native = [int(line) for line in process.stdout.splitlines()]
    expected = [int(np.asarray(compact.scalar_quantized_forward(quantized, architecture, row), dtype=np.float32).view(np.uint32)) for row in active]
    if native != expected:
        mismatches = [(index, left, right) for index, (left, right) in enumerate(zip(native, expected)) if left != right]
        raise ValueError(f"native/scalar float32 parity failed: {mismatches[:5]}, lengths {len(native)}/{len(expected)}")
    return {"profile": runtime["profile"], "runtime_body_sha256": runtime["body_sha256"],
            "rows": len(active), "bit_exact": True, "native_probe": record(probe),
            "outputs_sha256": hashlib.sha256(np.asarray(expected, dtype="<u4").tobytes()).hexdigest()}


def verify_smoke(directory: Path, probe: Path):
    corpus = json.loads((directory / "mechanical-groups.json").read_bytes())
    if corpus.get("classification") != "mechanical-only-not-production-data": raise ValueError("not a mechanical corpus")
    active = [group["parent_active"] for group in corpus["groups"]]
    active += [row["active"] for group in corpus["groups"] for row in group["successors"]]
    # Isolated feature rows ensure the largest signed four-bit weights and each
    # matrix boundary are exercised independently of the legal-state sample.
    active += [[index] for index in range(16)] + [[6300], [0, 6300]]
    smoke_report = json.loads((directory / "smoke.json").read_bytes())
    results = []
    for model in smoke_report["models"]:
        path = directory / (model["name"] + ".runtime.json")
        runtime = json.loads(path.read_bytes())
        source, exported = representation.export(runtime)
        if (source != (directory / (model["name"] + ".cpp")).read_bytes()
                or any(exported[key] != model["export"][key] for key in ("source_sha256", "source_bytes", "payload_sha256"))):
            raise ValueError("mechanical source/runtime changed")
        results.append(native_parity(runtime, probe, active))
    # Full four-bit alphabet, including values not present in the incumbent.
    runtime = representation.initialize("h12-b4")
    values, _, _ = representation.validate(runtime)
    values[:15] = list(range(-7, 8))
    boundary = representation.document("h12-b4", values, runtime["scales"], {"kind": "signed-domain-fixture"})
    results.append(native_parity(boundary, probe, active))
    report = {"schema": SCHEMA, "stage": "mechanical-native-parity-complete", "bit_exact": True,
              "production_eligible": False, "models": results, "corpus": record(directory / "mechanical-groups.json"),
              "smoke": record(directory / "smoke.json"), "native_inference_rows": sum(row["rows"] for row in results)}
    (directory / "native-parity.json").write_bytes(representation.canonical(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "smoke"):
        sub = subs.add_parser(command)
        sub.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
        sub.add_argument("--output", type=Path, required=True)
        if command == "smoke":
            sub.add_argument("--feature-probe", type=Path, required=True)
            sub.add_argument("--steps", type=int, default=8)
    parity = subs.add_parser("verify-smoke")
    parity.add_argument("--directory", type=Path, required=True)
    parity.add_argument("--probe", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare": result = prepare(args.output, args.teacher)
    elif args.command == "smoke": result = smoke(args.output, args.teacher, args.feature_probe, steps=args.steps)
    else: result = verify_smoke(args.directory, args.probe)
    print(json.dumps({key: value for key, value in result.items() if key not in ("models", "groups")}, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
