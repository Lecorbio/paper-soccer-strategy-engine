"""Bounded exact-parent, optimizer and full-anchor checks before expansion."""
from __future__ import annotations

import argparse
import ast
import copy
import gzip
import hashlib
import json
from pathlib import Path

from tools import rank_two_focused_training_v5 as retained
from tools import rank_two_focused_training_v6 as candidate
from tools import rank_two_focused_expansion_label_data_v1 as pipeline

campaign, storage, np, core = candidate.campaign, candidate.storage, candidate.np, candidate.core
SCHEMA = campaign.SCHEMA + ".warm-start-checks.v1"
NUMERICAL = ["quantize", "forward", "root_rows", "objective", "update_root", "diagnostics",
             "calibrate", "stopping", "runtime", "validation", "checkpoint", "restore"]


def function_bodies(path):
    return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(Path(path).read_text()).body
            if isinstance(node, ast.FunctionDef)}


def allocate():
    architecture = candidate.Architecture("dd8", 8, 8)
    parameters = {name: np.zeros(shape, dtype=np.float32) for name, shape in architecture.shapes.items()}
    return architecture, parameters, core.AdamW(parameters, learning_rate=.001, weight_decay=1e-5)


def packed(parameters, optimizer):
    return storage.pack(storage.arrays(parameters, optimizer))[1]


def run(plan_path):
    plan = campaign.read(plan_path)
    pipeline.check_current(plan, "experiment", launch=True)
    if plan["schema"] != SCHEMA + ".plan" or plan["producer"] != campaign.record(__file__):
        raise ValueError("warm-start check producer differs")
    for reference in plan["inputs"].values():
        campaign.verify(reference)
    out = Path(plan["output"])
    if (out / "CLAIM.json").exists():
        raise RuntimeError("spent warm-start diagnostic claim retained")
    binding = campaign.record(plan_path)
    campaign.immutable(out / "CLAIM.json", dict(plan=binding, new_games=0,
        production_training_started=False, diagnostic_optimizer_updates=1))
    campaign.immutable(out / "RESERVED_EXPOSURE.json", dict(schema=SCHEMA + ".exposure",
        plan=binding, training_eligible=False, before_next_fresh_bank=True))
    old = function_bodies(campaign.verify(plan["inputs"]["retained_trainer"]))
    new = function_bodies(campaign.verify(plan["inputs"]["expansion_trainer"]))
    if any(old[name] != new[name] for name in NUMERICAL):
        raise AssertionError("retained numerical functions changed")
    declaration = campaign.read(campaign.verify(plan["declaration"]))
    parent = declaration["preserve_checkpoint"]
    architecture, parameters, optimizer = allocate()
    anchor = candidate.warm_start(plan, binding, parameters, optimizer, out)
    _, parent_raw = storage.decode(parent, declaration["parent_plan"])
    if packed(parameters, optimizer) != parent_raw or optimizer.step != parent["optimizer_step"]:
        raise AssertionError("warm start changed parameters or optimizer")
    _, anchor_raw = storage.decode(anchor, binding)
    if anchor_raw != parent_raw or anchor["storage"]["mode"] != "full" or anchor["storage"]["parent"] is not None:
        raise AssertionError("new-plan full anchor differs from exact parent")
    _, copied, copied_optimizer = allocate()
    for value in storage.arrays(copied, copied_optimizer).values():
        value.fill(np.float32(.123))
    storage.restore(anchor, copied, copied_optimizer, binding)
    if packed(copied, copied_optimizer) != parent_raw:
        raise AssertionError("full anchor roundtrip changed arrays")
    rejected = []
    for name, change in (("wrong-parent-plan", lambda value: value.update(parent_plan={**value["parent_plan"], "sha256": "0" * 64})),
                         ("wrong-parent-seed", lambda value: value.update(seed=value["seed"] + 1))):
        bad = copy.deepcopy(plan)
        change(bad)
        try:
            candidate.warm_start(bad, binding, copied, copied_optimizer, out / name)
        except ValueError:
            rejected.append(name)
        else:
            raise AssertionError("unbound warm start accepted: " + name)
    previous = campaign.read(campaign.verify(plan["previous_corpus"]))
    item = next(item for item in previous["group_artifacts"] if item["split"] == "train")
    group = json.loads(gzip.decompress(pipeline.data.bound(item["artifact"]).read_bytes()))
    campaign.immutable(out / "FIXTURE.json", dict(group=group, artifact=item["artifact"],
        kind="already-indexed-TRAIN-only-diagnostic", training_eligible=False, new_game=False))
    first = retained.update_root(parameters, architecture, optimizer, [group])
    second = candidate.update_root(copied, architecture, copied_optimizer, [group])
    if first != second or packed(parameters, optimizer) != packed(copied, copied_optimizer):
        raise AssertionError("one useful-parent optimizer continuation differs")
    old_delta_parent = campaign.read(campaign.verify(parent["storage"]["parent"]))
    if old_delta_parent["storage"]["depth"] >= storage.MAX_DEPTH:
        raise AssertionError("cross-plan rejection fixture must exercise the delta branch")
    try:
        storage.checkpoint(out / "illegal-cross-plan.psc", parameters, optimizer, parent["cursor"], binding, parent=old_delta_parent)
    except ValueError:
        rejected.append("implicit-cross-plan-delta")
    else:
        raise AssertionError("cross-plan delta accepted without full transition anchor")
    cursor = dict(phase="diagnostic", epoch=0, next_root=1,
                  rng_seed=parent["cursor"]["rng_seed"] + 1, root_order=[group["root_group_id"]])
    delta = storage.checkpoint(out / "continuation.psc", parameters, optimizer, cursor, binding, parent=anchor)
    storage.restore(delta, copied, copied_optimizer, binding)
    if packed(copied, copied_optimizer) != packed(parameters, optimizer) or copied_optimizer.step != optimizer.step:
        raise AssertionError("new-plan continuation checkpoint roundtrip differs")
    campaign.verify(plan["parent_plan"])
    pipeline.data.bound(parent["storage"]["producer"]) if "bytes" in parent["storage"]["producer"] else campaign.verify(parent["storage"]["producer"])
    campaign.verify(parent["checkpoint"])
    result = dict(passed=True, plan=binding, retained_numerical_functions_unchanged=NUMERICAL,
        exact_parent_arrays_optimizer_step_and_cursor=True, new_plan_full_anchor=True,
        full_anchor_roundtrip=True, diagnostic_continuation_bit_exact=True,
        continuation_checkpoint_roundtrip=True, rejected=rejected,
        parent_step=parent["optimizer_step"], continued_step=optimizer.step,
        parent_array_sha256=hashlib.sha256(parent_raw).hexdigest(),
        anchor=candidate.campaign.record(Path(anchor["checkpoint"]["path"]).with_suffix(".json")),
        diagnostic_checkpoint=campaign.record(Path(delta["checkpoint"]["path"]).with_suffix(".json")),
        fixture=campaign.record(out / "FIXTURE.json"), new_games=0,
        production_training_started=False, source_exports=0)
    campaign.immutable(out / "RESULT.json", result)
    campaign.immutable(out / "EXPOSURE_PENDING.json", dict(schema=SCHEMA + ".exposure",
        input=campaign.record(out / "FIXTURE.json"), result=campaign.record(out / "RESULT.json"),
        retained_artifacts=[result["anchor"], result["diagnostic_checkpoint"]],
        training_eligible=False, before_next_fresh_bank=True, new_games=0))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.plan)))


if __name__ == "__main__":
    main()
