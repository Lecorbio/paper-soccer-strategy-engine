#!/usr/bin/env python3
"""Post-training proof-aware diagnostics; never a model selection or game gate.

Runtime proof flags come ONLY from one root expansion of the exact compact
source. Deeper teacher proofs are targets, never runtime proof overrides.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import shutil
import subprocess
import sys
import time
import gzip

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"
os.environ["PAPERSOCCER_COMPACT_TRAINING_THREADS_FIXED_BEFORE_NUMPY"] = "1"

try:
    from . import top_three_experiments as experiments, jacek_replay_features as features
except ImportError:
    import top_three_experiments as experiments
    import jacek_replay_features as features

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "tools/top_three_one_ply_probe.cpp"
SCHEMA = "papersoccer.top-three.proof-aware-diagnostic.v1"


def record(path):
    return experiments.raw_record(path)


def bound(item):
    path = Path(item["path"])
    if record(path)["sha256"] != item["sha256"] or ("bytes" in item and path.stat().st_size != item["bytes"]):
        raise ValueError("diagnostic input identity changed")
    return path


def physical_key(state):
    words = [0] * 5
    for edge in state.used_segments:
        index = features.EDGE_INDEX[edge]
        words[index // 64] |= 1 << (index % 64)
    return ("".join(f"{word:016x}" for word in words) + f":{features.POINT_INDEX[state.ball]}:{state.to_move}:"
            + f"{-1 if state.winner is None else state.winner}:{len(state.used_segments)}")


def successor(root, action):
    child = copy.deepcopy(root)
    features.apply_complete_turn(child, child.to_move, action)
    return child


def teacher_map(group, root):
    result = {}
    for row in group["successors"]:
        action = row["transcript"] if root.to_move == 0 else "".join(str((int(d) + 4) % 8) for d in row["transcript"])
        child = successor(root, action)
        if row["value_mover"] != child.to_move or list(features.encode_active(child)) != row["active"]:
            raise ValueError("teacher successor frame/features disagree with the exact boundary")
        key = physical_key(child)
        value = row["teacher_value"] * (1 if child.to_move == root.to_move else -1)
        if not math.isfinite(value) or not -1 <= value <= 1:
            raise ValueError("teacher value is outside its normalized mover frame")
        if key in result and result[key]["value"] != value:
            raise ValueError("boundary aliases have conflicting teacher values")
        result[key] = {"value": value, "active": row["active"], "teacher_solved": row["proof"]["solved"]}
    return result


def choose(rows, field):
    return min(rows, key=lambda row: (-row[field], row["order"])) if rows else None


def analyse_group(group, prefix, probe, float_mover_values=None, teacher_labels=None):
    root = experiments.state(prefix)
    if (probe["schema"] != "papersoccer.top-three.one-ply-probe.v1" or probe["group_id"] != group["group_id"] or probe["root_key"] != physical_key(root)
            or probe["expansions"] != 1 or probe["max_depth"] != 1):
        raise ValueError("probe is not the expected one-ply root")
    # A paired diagnostic can reuse this already validated immutable map for
    # both sources. The ordinary single-source path still validates it here.
    labels = teacher_map(group, root) if teacher_labels is None else teacher_labels
    retained = []
    seen = set()
    for raw in probe["actions"]:
        row = dict(raw)
        child = successor(root, row["action"])
        key = physical_key(child)
        if key != row["key"] or key in seen or row["value_mover"] != child.to_move:
            raise ValueError("probe retained boundary identity changed or duplicated")
        if row["terminal"] != (child.winner is not None):
            raise ValueError("probe terminal marker changed")
        seen.add(key)
        if not row["solved"] and row["proof_parent_value"] != row["raw_parent_value"]:
            raise ValueError("unsolved runtime action received a proof override")
        if float_mover_values is not None:
            row["float_parent_value"] = float_mover_values[key] * (1 if child.to_move == root.to_move else -1)
            row["float_proof_value"] = row["proof_parent_value"] if row["solved"] else row["float_parent_value"]
        retained.append(row)
    if not retained:
        raise ValueError("probe retained no actions")
    quantized_choice = choose(retained, "proof_parent_value")
    if quantized_choice["key"] != probe["chosen_key"]:
        raise ValueError("one-ply diagnostic choice differs from actual deployed root scoring")
    raw_choice = choose(retained, "raw_parent_value")
    unresolved = [r for r in retained if not r["solved"]]
    root_proved_win = any(r["solved"] and r["proof_parent_value"] > 0 for r in retained)
    unresolved_choice = choose(unresolved, "raw_parent_value")
    teacher_values = [r["value"] for r in labels.values()]
    comparable = group["exhaustive"] and len(teacher_values) > 1 and max(teacher_values) - min(teacher_values) > 1e-4
    def regret(chosen, candidates):
        if chosen is None or chosen["key"] not in labels or not candidates:
            return None
        return max(labels[key]["value"] for key in candidates) - labels[chosen["key"]]["value"]
    common = seen & labels.keys()
    if group["exhaustive"] and seen - labels.keys():
        raise ValueError("exhaustive teacher group omits a legal retained runtime boundary")
    unknown_keys = {r["key"] for r in unresolved} & labels.keys()
    unresolved_comparable = (not root_proved_win and group["exhaustive"] and len(unknown_keys) > 1 and
                            max(labels[k]["value"] for k in unknown_keys) - min(labels[k]["value"] for k in unknown_keys) > 1e-4)
    result = {"group_id": group["group_id"], "root_group_id": group["root_group_id"], "edges": group["edges"],
              "teacher_exhaustive": group["exhaustive"],
              "teacher_successors": len(labels), "retained_actions": len(retained), "retained_labelled_actions": len(common),
              "teacher_boundary_retention": len(common) / len(labels), "unlabelled_retained_actions": len(seen - labels.keys()),
              "runtime_proved_actions": sum(r["solved"] for r in retained),
              "runtime_root_proved_winning": root_proved_win,
              "teacher_solved_but_runtime_unresolved": sum(labels[r["key"]]["teacher_solved"] for r in unresolved if r["key"] in labels),
              "proof_aware_regret": regret(quantized_choice, labels) if comparable else None,
              "raw_retained_regret": regret(raw_choice, labels) if comparable else None,
              "proof_aware_retained_set_regret": regret(quantized_choice, common) if comparable else None,
              "unresolved_only_regret": regret(unresolved_choice, unknown_keys) if unresolved_comparable else None,
              "proof_changes_quantized_choice": float(raw_choice["key"] != quantized_choice["key"]),
              "proof_aware_float_quantized_flip": None, "unresolved_float_quantized_flip": None,
              "raw_float_quantized_flip": None, "chosen_key": quantized_choice["key"],
              "raw_chosen_key": raw_choice["key"], "runtime_choice_tactical": quantized_choice["tactical"]}
    if float_mover_values is not None:
        result["proof_aware_float_quantized_flip"] = float(choose(retained, "float_proof_value")["key"] != quantized_choice["key"])
        result["raw_float_quantized_flip"] = float(choose(retained, "float_parent_value")["key"] != raw_choice["key"])
        if len(unresolved) >= 2 and not root_proved_win:
            result["unresolved_float_quantized_flip"] = float(choose(unresolved, "float_parent_value")["key"] != unresolved_choice["key"])
    return result


def aggregate(rows):
    metrics = ("proof_aware_regret", "raw_retained_regret", "proof_aware_retained_set_regret", "unresolved_only_regret",
               "proof_changes_quantized_choice", "proof_aware_float_quantized_flip", "unresolved_float_quantized_flip",
               "raw_float_quantized_flip", "teacher_boundary_retention")
    result = {}
    for phase in ("overall", "early"):
        selected = [r for r in rows if phase == "overall" or r["edges"] <= 12]
        result[phase] = {}
        for key in metrics:
            by_root = defaultdict(list)
            for row in selected:
                if row[key] is not None: by_root[row["root_group_id"]].append(row[key])
            values = [sum(v) / len(v) for v in by_root.values()]
            result[phase][key] = {"mean": sum(values) / len(values) if values else None,
                                  "roots": len(values), "positions": sum(map(len, by_root.values()))}
    return result


def coverage_counts(rows):
    return {"teacher_actions": sum(r["teacher_successors"] for r in rows),
            "runtime_retained_actions": sum(r["retained_actions"] for r in rows),
            "matched_boundaries": sum(r["retained_labelled_actions"] for r in rows),
            "runtime_proved_actions": sum(r["runtime_proved_actions"] for r in rows),
            "teacher_solved_but_runtime_unresolved": sum(r["teacher_solved_but_runtime_unresolved"] for r in rows),
            "positions_with_runtime_proved_winning_action": sum(r["runtime_root_proved_winning"] for r in rows)}


def build(source, expected_sha, directory, compiler):
    source = Path(source).resolve()
    if record(source)["sha256"] != expected_sha: raise ValueError("candidate source hash mismatch")
    directory = Path(directory).resolve()
    signature = experiments.compiler_signature(compiler)
    source_copy, probe_copy = directory / "submission.cpp", directory / "probe.cpp"
    experiments.campaign.immutable(source_copy, source.read_bytes())
    experiments.campaign.immutable(probe_copy, PROBE.read_bytes())
    binary = directory / "probe"
    plan = {"source": record(source_copy), "probe": record(probe_copy), "compiler": signature}
    experiments.emit(directory / "plan.json", plan)
    receipt = directory / "build.json"
    if receipt.exists():
        value = experiments.read(receipt)
        if value["plan"] != plan: raise ValueError("diagnostic build changed")
        bound(value["binary"])
        return value
    command = [signature["executable"]["path"], "-std=c++20", "-O3", "-DNDEBUG", "-ffp-contract=off",
               f'-DTOP_THREE_COMPACT_SOURCE="{source_copy}"', str(probe_copy), "-o", str(binary)]
    process = subprocess.run(command, capture_output=True, text=True, timeout=180)
    if process.returncode: raise RuntimeError(process.stderr)
    value = {"plan": plan, "binary": record(binary), "command": command}
    experiments.emit(receipt, value)
    return value


class Probe:
    def __init__(self, binary):
        self.child = subprocess.Popen([str(bound(binary))], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=subprocess.DEVNULL, bufsize=0)
        os.set_blocking(self.child.stdout.fileno(), False)

    def query(self, group_id, prefix):
        self.child.stdin.write(f"{group_id}\t{prefix or '-'}\n".encode())
        self.child.stdin.flush()
        received = bytearray()
        started = time.monotonic()
        with selectors.DefaultSelector() as selector:
            selector.register(self.child.stdout, selectors.EVENT_READ)
            while b"\n" not in received:
                if time.monotonic() - started > 15: raise TimeoutError("one-ply probe watchdog")
                if selector.select(.1):
                    chunk = os.read(self.child.stdout.fileno(), 65536)
                    if not chunk: raise RuntimeError("one-ply probe exited")
                    received.extend(chunk)
                    if len(received) > 2 * 1024**2: raise ValueError("one-ply response exceeds2MiB")
        if received.count(b"\n") != 1 or not received.endswith(b"\n"): raise ValueError("extra probe output")
        return json.loads(received)

    def close(self):
        self.child.stdin.close()
        if self.child.poll() is None:
            self.child.terminate()
            try: self.child.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.child.kill(); self.child.wait()
        self.child.stdout.close()


def run(args):
    # Numerical helpers are imported only for this post-training command.
    if str(Path(__file__).resolve().parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
    import numpy as np
    import compact_representation as representation
    import compact_representation_pilot as mechanics
    import compact_representation_train as training
    runtime = json.loads(bound({"path": str(args.runtime.resolve()), "sha256": args.runtime_sha256}).read_bytes())
    architecture, quantized, width = mechanics.tensors(runtime)
    checkpoint = bound({"path": str(args.checkpoint.resolve()), "sha256": args.checkpoint_sha256})
    with np.load(checkpoint, allow_pickle=False) as data:
        parameters = {name: data["parameter_" + name].copy() for name in ("w1", "w2", "w3")}
    if any(parameters[name].shape != architecture.shapes[name] or parameters[name].dtype != np.dtype("float32")
           or not np.all(np.isfinite(parameters[name])) for name in parameters):
        raise ValueError("floating checkpoint tensor contract changed")
    float_digest = mechanics.parameter_digest(parameters)
    claimed_digest = runtime.get("lineage", {}).get("parameters_sha256")
    if claimed_digest is not None and claimed_digest != float_digest:
        raise ValueError("floating checkpoint is not the selected model's saved master parameters")
    if runtime.get("lineage", {}).get("kind") == "fresh-512-search-label-training" and claimed_digest is None:
        raise ValueError("trained model lacks its float-master binding")
    rebuilt = mechanics.quantize(parameters, quantized.scales, width)
    if any(not np.array_equal(rebuilt.integer[name], quantized.integer[name]) for name in parameters):
        raise ValueError("floating checkpoint does not quantize to the selected runtime")
    corpus_path = bound({"path": str(args.corpus.resolve()), "sha256": args.corpus_sha256})
    corpus = json.loads(corpus_path.read_bytes())
    if corpus.get("schema") != "papersoccer.compact-representation.fresh-data.v1" or corpus.get("kind") != "fresh-complete-turn-search-corpus":
        raise ValueError("expected fresh complete-turn corpus")
    label_plan = json.loads(bound(corpus["label_plan"]).read_bytes())
    positions = {row["canonical_state"]: row for row in label_plan["positions"]}
    groups = [item for item in corpus["group_artifacts"] if item["split"] == "validation"]
    root_ids = sorted({item["root_group_id"] for item in groups})
    if args.maximum_roots:
        root_ids = root_ids[:args.maximum_roots]
    groups = sorted((g for g in groups if g["root_group_id"] in root_ids), key=lambda g: (g["root_group_id"], g["group_id"]))
    if not groups: raise ValueError("no validation groups selected")
    output = args.output.resolve()
    built = build(args.source, args.source_sha256, output / "build", args.compiler)
    if record(args.source)["sha256"] != args.source_sha256:
        raise ValueError("source changed during diagnostic build")
    plan = {"schema": SCHEMA, "purpose": "post-training-diagnostic-only", "selection_rule_changed": False,
            "source": record(args.source), "runtime": record(args.runtime), "checkpoint": record(checkpoint),
            "corpus": record(corpus_path), "label_plan": corpus["label_plan"], "groups": groups,
            "probe_build": built, "max_expansions": 1, "clock_limited": False, "maximum_roots": args.maximum_roots,
            "diagnostic_tie_break": "deployed retained-action generation order; frozen raw-regret selection is not recomputed or replaced",
            "float_parameters_sha256": float_digest,
            "float_reference": "selected-training-master" if claimed_digest else "external-quantization-matched-fixture",
            "numpy_version": np.__version__, "python": record(sys.executable),
            "root_order": root_ids, "workers": 1, "qualified_for_games": False,
            "helpers": {"driver": record(__file__), "features": record(features.__file__),
                        "experiments": record(experiments.__file__), "representation": record(representation.__file__),
                        "math_core": record(training.math_core.__file__),
                        "mechanics": record(mechanics.__file__), "training_math": record(training.__file__)}}
    experiments.emit(output / "plan.json", plan)
    if (output / "result.json").exists():
        saved = experiments.read(output / "result.json")
        positions_path = bound(saved["positions"])
        bound(saved["raw_probe"])
        rows = [json.loads(line) for line in positions_path.read_text().splitlines()]
        if (saved["plan"] != record(output / "plan.json") or saved["metrics"] != aggregate(rows) or
                saved["coverage_counts"] != coverage_counts(rows) or saved["position_count"] != len(rows)):
            raise ValueError("cached diagnostic report differs from its bound evidence")
        return saved
    if (output / "claim.json").exists(): raise ValueError("interrupted diagnostic is preserved; use explicit recovery")
    experiments.emit(output / "claim.json", {"plan": record(output / "plan.json")})
    probe = Probe(built["binary"])
    rows = []
    try:
        with (output / "raw-probe.jsonl").open("xb") as raw_output:
            for item in groups:
                group = json.loads(gzip.decompress(bound(item["artifact"]).read_bytes()))
                position = positions[group["canonical_state"]]
                if (group["group_id"] != item["group_id"] or group["root_group_id"] != item["root_group_id"] or group["split"] != "validation"
                        or position["mover"] != group["mover"]): raise ValueError("group/root binding mismatch")
                native = probe.query(group["group_id"], position["prefix"])
                if native["payload_sha256"] != runtime["payload_sha256"] or native["runtime_body_sha256"] != runtime["body_sha256"]:
                    raise ValueError("embedded source model differs from supplied runtime")
                raw_output.write((json.dumps(native, sort_keys=True) + "\n").encode())
                root = experiments.state(position["prefix"])
                children = [successor(root, row["action"]) for row in native["actions"]]
                active = [np.asarray(features.encode_active(child), dtype=np.uint16) for child in children]
                qvalues, _ = training.native_order_forward(parameters, architecture, active, quantized)
                float_values, _ = training.native_order_forward(parameters, architecture, active)
                if [int(x) for x in qvalues.view(np.uint32)] != [row["raw_mover_bits"] for row in native["actions"]]:
                    raise ValueError("diagnostic inference differs from exact exported source")
                float_map = {physical_key(child): float(value) for child, value in zip(children, float_values)}
                rows.append(analyse_group(group, position["prefix"], native, float_map))
                if len(rows) % 32 == 0:
                    print(json.dumps({"diagnostic_positions": len(rows), "requested_positions": len(groups)}), file=sys.stderr, flush=True)
    finally:
        probe.close()
    experiments.campaign.immutable(output / "positions.jsonl", b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows))
    result = {"schema": SCHEMA, "plan": record(output / "plan.json"), "positions": record(output / "positions.jsonl"),
              "raw_probe": record(output / "raw-probe.jsonl"), "position_count": len(rows), "root_groups": len(root_ids),
              "coverage_counts": coverage_counts(rows),
              "metrics": aggregate(rows), "source_selection_unchanged": True, "game_qualified": False,
              "claim_boundary": "One root expansion with deployed action/proof caps and no deadline truncation; deeper teacher proofs never override runtime NN values. This is not full BFM or game strength."}
    experiments.emit(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "runtime", "checkpoint", "corpus"):
        parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiler", default="/usr/bin/clang++")
    parser.add_argument("--maximum-roots", type=int, default=0)
    args = parser.parse_args()
    if args.maximum_roots < 0: parser.error("maximum-roots cannot be negative")
    result = run(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
