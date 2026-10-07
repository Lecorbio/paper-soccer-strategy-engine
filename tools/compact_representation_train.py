#!/usr/bin/env python3
"""Six matched fresh-data training arms; root-balanced updates and native parity."""
from __future__ import annotations
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

for _name in ("MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"
os.environ["PAPERSOCCER_COMPACT_TRAINING_THREADS_FIXED_BEFORE_NUMPY"] = "1"
import numpy as np
import compact_representation as representation
import compact_representation_pilot as mechanics
import compact_value_bfm_train as math_core
import jacek_replay_train as archive_math

SCHEMA = "papersoccer.compact-representation.production-training.v1"
RECIPE = {"seed": 2026091604, "profiles": ["h12-b3", "h12-b4", "h16-b3"], "ranking_weights": [0.0, 0.1],
          "float_epochs": 1, "qat_epochs": 4, "float_learning_rate": 0.00006, "qat_learning_rate": 0.00025,
          "optimizer": "AdamW-continuous-moments", "weight_decay": 0.00001, "global_gradient_clip": 5.0,
          "updates": "one-per-training-root-per-epoch; mean-of-position-losses-within-root",
          "huber_delta": 0.25, "ranking_temperature": 0.2,
          "quantization_scales": "frozen-incumbent-scales; four-bit uses half scales",
          "selection": ["quantized-root-balanced-validation-regret", "quantized-root-balanced-validation-Huber", "epoch"],
          "hard_source_limit": 99999, "preferred_source_target": 95000,
          "native_parity_required": True, "maximum_finalists": 2, "maximum_rss_bytes": 12 * 1024**3,
          "offline_regret_improvement_required": False, "exclude_unchanged_quantized_runtime": True,
          "deduplicate_numerical_runtime": True,
          "native_threads": 1, "game_qualification": False}


def record(path):
    path = Path(path).resolve()
    payload = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


def bound(item):
    path = Path(item["path"])
    if record(path) != item: raise ValueError("training artifact identity changed")
    return path


def once(path, value):
    payload = value if isinstance(value, bytes) else representation.canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload: raise ValueError("immutable training artifact changed")
    else: path.write_bytes(payload)


def rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    value = int(value if sys.platform == "darwin" else value * 1024)
    if value >= RECIPE["maximum_rss_bytes"]: raise MemoryError("training/aggregation exceeded 12 GiB")
    return value


class Corpus:
    def __init__(self, path):
        self.identity = record(path)
        self.manifest = json.loads(path.read_bytes())
        if (self.manifest.get("schema") != "papersoccer.compact-representation.fresh-data.v1"
                or self.manifest.get("kind") != "fresh-complete-turn-search-corpus"
                or self.manifest.get("teacher_runtime_sha256") != mechanics.TEACHER_SHA256
                or self.manifest.get("production_training_admitted") is not True
                or self.manifest.get("independent_comparable_early_validation_root_groups", 0) < 100):
            raise ValueError("fresh corpus is not admitted by the independent early validation gate")
        self.roots = {"train": defaultdict(list), "validation": defaultdict(list)}
        for item in self.manifest["group_artifacts"]:
            if item["split"] not in self.roots: raise ValueError("unknown split")
            self.roots[item["split"]][item["root_group_id"]].append(item)
        if set(self.roots["train"]) & set(self.roots["validation"]): raise ValueError("training root crosses validation")
        if not self.roots["train"] or len(self.roots["validation"]) < 100: raise ValueError("insufficient independent roots")
        for split in self.roots:
            for rows in self.roots[split].values(): rows.sort(key=lambda item: item["group_id"])
        rss()

    def root(self, split, root_id):
        result = []
        for item in self.roots[split][root_id]:
            path = bound(item["artifact"])
            group = json.loads(gzip.decompress(path.read_bytes()))
            if group["root_group_id"] != root_id or group["split"] != split or group["group_id"] != item["group_id"]:
                raise ValueError("group index disagrees with bound group content")
            result.append(group)
        rss()
        return result


def native_order_forward(parameters, architecture, active, quantized=None):
    """Literal float32 operations, with the deployment integer accumulator in QAT."""
    count, hidden = len(active), architecture.hidden_one
    first_pre = np.zeros((count, hidden), dtype=np.float32)
    for row, indices in enumerate(active):
        if quantized is None:
            for index in indices: first_pre[row] = np.asarray(first_pre[row] + parameters["w1"][index], dtype=np.float32)
        else:
            total = np.sum(quantized.integer["w1"][indices], axis=0, dtype=np.int32)
            first_pre[row] = total.astype(np.float32) * quantized.scales["w1"]
    first = math_core.first_activation(first_pre)
    second_pre = np.zeros((count, 8), dtype=np.float32)
    for input_index in range(hidden):
        for output_index in range(8):
            if quantized is None:
                term = np.asarray(first[:, input_index] * parameters["w2"][input_index, output_index], dtype=np.float32)
            else:
                scaled = np.asarray(first[:, input_index] * quantized.scales["w2"], dtype=np.float32)
                term = np.asarray(scaled * quantized.integer["w2"][input_index, output_index], dtype=np.float32)
            second_pre[:, output_index] = np.asarray(second_pre[:, output_index] + term, dtype=np.float32)
    second = math_core.second_activation(second_pre)
    output_pre = np.zeros(count, dtype=np.float32)
    for index in range(8):
        if quantized is None: term = np.asarray(second[:, index] * parameters["w3"][index], dtype=np.float32)
        else:
            scaled = np.asarray(second[:, index] * quantized.scales["w3"], dtype=np.float32)
            term = np.asarray(scaled * quantized.integer["w3"][index], dtype=np.float32)
        output_pre = np.asarray(output_pre + term, dtype=np.float32)
    output = np.asarray([math_core._fast_tanh_scalar(value) for value in output_pre], dtype=np.float32)
    if not np.all(np.isfinite(output)): raise FloatingPointError("nonfinite native-order forward")
    return output, (first_pre, first, second_pre, second, output_pre)


def root_rows(groups):
    active, targets, signs, slices = [], [], [], []
    for group in groups:
        begin = len(active)
        for row in group["successors"]:
            active.append(np.asarray(row["active"], dtype=np.uint16))
            targets.append(row["teacher_value"])
            signs.append(1 if row["value_mover"] == group["mover"] else -1)
        if len(active) == begin: raise ValueError("root contains an empty action group")
        slices.append(slice(begin, len(active)))
    return active, np.asarray(targets, dtype=np.float32), np.asarray(signs, dtype=np.float32), slices


def loss_gradient(predictions, targets, signs, slices, ranking_weight):
    gradient = np.zeros_like(predictions)
    scalar_total, ranking_total = 0.0, 0.0
    for indices in slices:
        scalar, local = math_core._weighted_huber_loss_gradient(predictions[indices], targets[indices], np.ones(len(predictions[indices]), dtype=np.float32))
        ranking = 0.0
        if ranking_weight:
            ranking, rank_gradient = mechanics.ranking_loss_gradient(predictions[indices], targets[indices], signs[indices])
            local += np.float32(ranking_weight) * rank_gradient
        gradient[indices] = local / np.float32(len(slices))
        scalar_total += scalar / len(slices)
        ranking_total += ranking / len(slices)
    return scalar_total, ranking_total, gradient


def metrics(corpus, parameters, architecture, quantized):
    roots, early_roots, hubs, flips = [], [], [], []
    for root_id in sorted(corpus.roots["validation"]):
        groups = corpus.root("validation", root_id)
        active, targets, signs, slices = root_rows(groups)
        predictions, _ = native_order_forward(parameters, architecture, active, quantized)
        float_predictions, _ = native_order_forward(parameters, architecture, active)
        regrets, early, losses, changed = [], [], [], []
        for group, span in zip(groups, slices):
            loss, _ = math_core._weighted_huber_loss_gradient(predictions[span], targets[span], np.ones(len(predictions[span]), dtype=np.float32))
            losses.append(loss)
            if not group["exhaustive"] or len(predictions[span]) < 2: continue
            teacher = targets[span] * signs[span]
            if float(np.max(teacher) - np.min(teacher)) <= 1e-4: continue
            chosen = int(np.argmax(predictions[span] * signs[span]))
            float_chosen = int(np.argmax(float_predictions[span] * signs[span]))
            regret = float(np.max(teacher) - teacher[chosen])
            regrets.append(regret)
            changed.append(float(chosen != float_chosen))
            if group["edges"] <= 12: early.append(regret)
        hubs.append(float(np.mean(losses)))
        if regrets: roots.append(float(np.mean(regrets)))
        if early: early_roots.append(float(np.mean(early)))
        if changed: flips.append(float(np.mean(changed)))
    if len(early_roots) < 100: raise ValueError("actual evaluated early validation root coverage below 100")
    return {"regret": float(np.mean(roots)), "early_regret": float(np.mean(early_roots)),
            "huber": float(np.mean(hubs)), "float_quantized_action_flip_rate": float(np.mean(flips)),
            "validation_roots": len(hubs), "comparable_roots": len(roots), "early_roots": len(early_roots)}


def initialize(profile):
    architecture, original, width = mechanics.tensors(representation.initialize(profile))
    parameters = original.effective()
    scales = {name: np.float32(value / (2 if width == 4 else 1)) for name, value in original.scales.items()}
    if architecture.hidden_one == 16: parameters["w1"][:, 12:16] = parameters["w1"][:, :4]
    return architecture, parameters, scales, width


def checkpoint(path, parameters, optimizer):
    arrays = {"parameter_" + key: value for key, value in parameters.items()}
    arrays.update({"first_" + key: value for key, value in optimizer.first.items()})
    arrays.update({"second_" + key: value for key, value in optimizer.second.items()})
    arrays["step"] = np.asarray([optimizer.step], dtype=np.int64)
    once(path, archive_math.deterministic_npz(arrays))
    return record(path)


def restore(item, parameters, optimizer):
    with np.load(bound(item), allow_pickle=False) as arrays:
        for key in parameters:
            for prefix in ("parameter_", "first_", "second_"):
                value = arrays[prefix + key]
                if value.shape != parameters[key].shape or value.dtype != np.dtype("float32") or not np.all(np.isfinite(value)):
                    raise ValueError("checkpoint tensor shape/dtype/finiteness changed")
            parameters[key][...] = arrays["parameter_" + key]
            optimizer.first[key][...] = arrays["first_" + key]
            optimizer.second[key][...] = arrays["second_" + key]
        optimizer.step = int(arrays["step"][0])


def train_arm(corpus, profile, ranking_weight, output, binding, compiler):
    architecture, parameters, scales, width = initialize(profile)
    initial_digest = mechanics.parameter_digest(parameters)
    initial_codes = mechanics.quantize(parameters, scales, width)
    initial_metrics = metrics(corpus, parameters, architecture, initial_codes)
    optimizer = math_core.AdamW(parameters, learning_rate=RECIPE["float_learning_rate"], weight_decay=RECIPE["weight_decay"])
    once(output / "arm.json", {"schema": SCHEMA, "binding": binding, "profile": profile, "ranking_weight": ranking_weight,
                               "initial_parameters_sha256": initial_digest, "initial_metrics": initial_metrics})
    root_ids = sorted(corpus.roots["train"])
    outcomes = []
    for epoch in range(5):
        directory = output / f"epoch-{epoch:02d}"
        receipt_path = directory / "receipt.json"
        optimizer.learning_rate = np.float32(RECIPE["float_learning_rate"] if epoch == 0 else RECIPE["qat_learning_rate"])
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_bytes())
            if (receipt["binding"] != binding or receipt["epoch"] != epoch
                    or receipt.get("profile") != profile or receipt.get("ranking_weight") != ranking_weight
                    or receipt.get("initial_parameters_sha256") != initial_digest):
                raise ValueError("checkpoint belongs to another run or training arm")
            restore(receipt["checkpoint"], parameters, optimizer)
            if (optimizer.step != len(root_ids) * (epoch + 1)
                    or mechanics.parameter_digest(parameters) != receipt["updates"]["parameters_sha256"]):
                raise ValueError("checkpoint optimizer steps or parameter digest changed")
            bound(receipt["runtime"]); bound(receipt["source"])
            if epoch: outcomes.append(receipt)
            continue
        if (directory / "claim.json").exists(): raise ValueError("interrupted training epoch requires explicit recovery")
        order = np.random.default_rng(RECIPE["seed"] + epoch).permutation(len(root_ids))
        once(directory / "claim.json", {"binding": binding, "epoch": epoch, "root_order": [root_ids[index] for index in order]})
        before = {key: value.copy() for key, value in parameters.items()}
        losses, ranks, gradient_updates = [], [], 0
        started = time.perf_counter()
        for index in order:
            groups = corpus.root("train", root_ids[index])
            active, targets, signs, slices = root_rows(groups)
            quantized = None if epoch == 0 else mechanics.quantize(parameters, scales, width)
            predictions, cache = native_order_forward(parameters, architecture, active, quantized)
            scalar_loss, ranking_loss, output_gradient = loss_gradient(predictions, targets, signs, slices, ranking_weight)
            effective = parameters if quantized is None else quantized.effective()
            gradients = math_core._network_gradients(parameters, architecture, active, cache, output_gradient, effective)
            norm = math.sqrt(sum(float(np.sum(value.astype(np.float64)**2)) for value in gradients.values()))
            if norm > RECIPE["global_gradient_clip"]:
                ratio = np.float32(RECIPE["global_gradient_clip"] / norm)
                for value in gradients.values(): value *= ratio
            if any(np.any(value) for value in gradients.values()): gradient_updates += 1
            optimizer.update(parameters, gradients)
            losses.append(scalar_loss); ranks.append(ranking_loss)
            rss()
        quantized = mechanics.quantize(parameters, scales, width)
        validation = metrics(corpus, parameters, architecture, quantized)
        changes = {key: int(np.count_nonzero(parameters[key] != before[key])) for key in parameters}
        integer_changes = {key: int(np.count_nonzero(quantized.integer[key] != initial_codes.integer[key])) for key in parameters}
        lineage = {"kind": "fresh-512-search-label-training", "recipe": RECIPE, "binding": binding,
                   "profile": profile, "ranking_weight": ranking_weight, "epoch": epoch,
                   "initial_parameters_sha256": initial_digest, "parameters_sha256": mechanics.parameter_digest(parameters),
                   "optimizer_steps": optimizer.step, "epoch_nonzero_gradient_updates": gradient_updates,
                   "epoch_changed_float_parameters": changes, "changed_integer_weights_vs_initial": integer_changes}
        runtime = mechanics.runtime_from_quantized(profile, quantized, lineage)
        source, exported = representation.export(runtime, RECIPE["hard_source_limit"])
        once(directory / "runtime.json", representation.canonical(runtime))
        once(directory / "submission.cpp", source)
        receipt = {"schema": SCHEMA, "binding": binding, "epoch": epoch, "profile": profile,
                   "ranking_weight": ranking_weight, "initial_parameters_sha256": initial_digest, "validation": validation,
                   "checkpoint": checkpoint(directory / "checkpoint.npz", parameters, optimizer),
                   "runtime": record(directory / "runtime.json"), "source": record(directory / "submission.cpp"),
                   "export": exported, "updates": lineage, "mean_scalar_loss": float(np.mean(losses)),
                   "mean_ranking_loss": float(np.mean(ranks)), "elapsed_seconds": time.perf_counter() - started,
                   "peak_rss_bytes": rss()}
        once(receipt_path, receipt)
        print(json.dumps({"arm": output.name, "epoch": epoch, "validation": validation, "source_bytes": len(source), "updates": optimizer.step}), flush=True)
        if epoch: outcomes.append(receipt)
    eligible = [row for row in outcomes if row["export"]["deployable_size"]
                and any(row["updates"]["changed_integer_weights_vs_initial"].values())]
    if not eligible:
        result = {"schema": SCHEMA, "arm": output.name, "selected": None,
                  "failure": "no source-valid QAT checkpoint with a changed quantized runtime"}
    else:
        selected = min(eligible, key=lambda row: (row["validation"]["regret"], row["validation"]["huber"], row["epoch"]))
        runtime = json.loads(bound(selected["runtime"]).read_bytes())
        source = bound(selected["source"])
        probe_source = representation.DIRECTORY / "inference_probe.cpp"
        probe = output / "selected-probe"
        command = [compiler, "-std=c++20", "-O3", "-DNDEBUG", f'-DCOMPACT_REPRESENTATION_SOURCE="{source}"', str(probe_source), "-o", str(probe)]
        process = subprocess.run(command, text=True, capture_output=True, timeout=180)
        if process.returncode: raise RuntimeError(process.stderr)
        active = []
        for root_id in sorted(corpus.roots["validation"]):
            for group in corpus.root("validation", root_id):
                active.append(group["parent_active"])
                active.extend(row["active"] for row in group["successors"][:4])
            if len(active) >= 256: break
        parity = mechanics.native_parity(runtime, probe, active)
        # Also test the actual embedded header, independent of dynamic loading.
        selected_architecture, selected_quantized, _ = mechanics.tensors(runtime)
        expected = [int(np.asarray(math_core.scalar_quantized_forward(selected_quantized, selected_architecture, row), dtype=np.float32).view(np.uint32)) for row in active]
        stdin = "".join(str(len(row)) + " " + " ".join(map(str, row)) + "\n" for row in active)
        native = subprocess.run([str(probe), "--predict"], input=stdin, text=True, capture_output=True, check=True, timeout=120)
        if [int(line) for line in native.stdout.splitlines()] != expected: raise ValueError("embedded selected export differs from Python scalar")
        result = {"schema": SCHEMA, "arm": output.name, "selected": selected, "native_parity": parity,
                  "compile_command": command, "compiler": record(Path(compiler)), "probe": record(probe),
                  "initial_metrics": initial_metrics, "offline_regret_improved": selected["validation"]["regret"] < initial_metrics["regret"],
                  "numerical_runtime_sha256": hashlib.sha256(representation.canonical({key: runtime[key] for key in ("architecture", "weight_bits", "scales", "payload_sha256")})).hexdigest(),
                  "game_qualified": False}
    once(output / "result.json", result)
    return result


def prepare(corpus_path, output):
    corpus = Corpus(corpus_path)
    frozen = output / "frozen_repo"
    tools = Path(__file__).resolve().parent
    names = ["compact_representation_train.py", "compact_representation_pilot.py", "compact_representation.py", "compact_huffman.py",
             "compact_value_bfm_train.py", "jacek_replay_train.py", "jacek_replay_corpus.py", "jacek_replay_features.py"]
    sources = []
    for name in names:
        target = frozen / "tools" / name
        once(target, (tools / name).read_bytes()); target.chmod(0o444)
        sources.append(record(target))
    for relative in ("submissions/codingame/bots/compact_value_bfm/discrete_v3_deployment.cpp",
                     "submissions/codingame/bots/compact_representation/runtime_codec.hpp",
                     "submissions/codingame/bots/compact_representation/inference_probe.cpp"):
        target = frozen / relative
        once(target, (representation.ROOT / relative).read_bytes()); target.chmod(0o444)
        sources.append(record(target))
    plan = {"schema": SCHEMA, "corpus": corpus.identity, "recipe": RECIPE, "sources": sources,
            "train_roots": len(corpus.roots["train"]), "validation_roots": len(corpus.roots["validation"]),
            "numpy_version": np.__version__, "python": record(Path(sys.executable).resolve()), "first_update_started": False}
    once(output / "plan.json", plan)
    return plan


def run(plan_path, output, compiler):
    plan = json.loads(plan_path.read_bytes())
    if plan.get("schema") != SCHEMA or plan.get("recipe") != RECIPE: raise ValueError("production recipe changed")
    for source in plan["sources"]: bound(source)
    if record(Path(sys.executable).resolve()) != plan["python"] or np.__version__ != plan["numpy_version"]:
        raise ValueError("training numerical runtime changed")
    if not any(Path(source["path"]).resolve() == Path(__file__).resolve() for source in plan["sources"]):
        raise ValueError("execute the frozen production trainer, not workspace source")
    source_by_name = {Path(item["path"]).name: item for item in plan["sources"]}
    for module in (mechanics, representation, math_core, archive_math,
                   representation.incumbent, mechanics.features, archive_math.corpus):
        loaded = Path(module.__file__).resolve()
        expected = source_by_name.get(loaded.name)
        if expected is None or loaded != Path(expected["path"]).resolve() or record(loaded) != expected:
            raise ValueError(f"loaded helper is outside the frozen closure: {loaded.name}")
    corpus = Corpus(bound(plan["corpus"]))
    compiler = shutil.which(compiler) or compiler
    binding = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    outcomes = []
    for profile in RECIPE["profiles"]:
        for weight in RECIPE["ranking_weights"]:
            name = profile + ("-scalar" if weight == 0 else "-ranking")
            try: result = train_arm(corpus, profile, weight, output / "arms" / name, binding, compiler)
            except Exception as error:
                result = {"schema": SCHEMA, "arm": name, "selected": None, "failure": str(error)}
                once(output / "arms" / name / "failure.json", result)
            outcomes.append(result)
    candidates = [row for row in outcomes if row.get("selected") and row["native_parity"]["bit_exact"]
                  and all(math.isfinite(row["selected"]["validation"][key]) for key in ("regret", "huber"))]
    ordered = sorted(candidates, key=lambda row: (row["selected"]["validation"]["regret"], row["selected"]["validation"]["huber"], row["selected"]["export"]["source_bytes"], row["arm"]))
    finalists, identities = [], set()
    for row in ordered:
        if row["numerical_runtime_sha256"] not in identities:
            finalists.append(row); identities.add(row["numerical_runtime_sha256"])
        if len(finalists) == 2: break
    result = {"schema": SCHEMA, "complete": True, "recipe": RECIPE, "outcomes": outcomes,
              "finalists": [row["arm"] for row in finalists], "game_qualified": False, "peak_rss_bytes": rss()}
    once(output / "complete.json", result)
    print(json.dumps({"complete": True, "finalists": result["finalists"], "failures": [row["arm"] for row in outcomes if row.get("failure")]}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    prep = subs.add_parser("prepare")
    prep.add_argument("--corpus", type=Path, required=True); prep.add_argument("--output", type=Path, required=True)
    train = subs.add_parser("run")
    train.add_argument("--plan", type=Path, required=True); train.add_argument("--output", type=Path, required=True)
    train.add_argument("--compiler", default="/usr/bin/clang++")
    args = parser.parse_args()
    if args.command == "prepare": print(json.dumps(prepare(args.corpus, args.output), sort_keys=True))
    else: run(args.plan, args.output, args.compiler)
