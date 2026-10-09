"""Check retained numerical code and reject mismatched teacher provenance."""
from __future__ import annotations

import argparse
import ast
import copy
import json
from pathlib import Path

from tools import rank_two_focused_teacher_budget_v1 as pipeline
from tools import rank_two_focused_teacher_budget_corpus_v1 as corpus

campaign = pipeline.campaign
SCHEMA = pipeline.SCHEMA + ".checks"


def rejects(function):
    try:
        function()
    except (ValueError, KeyError):
        return True
    raise AssertionError("invalid source/budget/feature binding was accepted")


def functions(path):
    return {node.name: ast.dump(node, include_attributes=False)
            for node in ast.parse(Path(path).read_text()).body if isinstance(node, ast.FunctionDef)}


def run(plan_path):
    plan = campaign.read(plan_path)
    pipeline.check_current(plan, "experiment", launch=True)
    for ref in plan["inputs"].values():
        campaign.verify(ref)
    if plan["producer"] != campaign.record(__file__):
        raise ValueError("teacher-budget checks producer differs")
    new = functions(plan["inputs"]["trainer"]["path"])
    old = functions(plan["inputs"]["retained_trainer"]["path"])
    names = ["initialize", "quantize", "forward", "root_rows", "objective", "update_root",
             "diagnostics", "calibrate", "stopping", "checkpoint", "restore", "runtime",
             "validation", "warm_start"]
    if any(new[name] != old[name] for name in names):
        raise ValueError("retained numerical or warm-start implementation changed")
    stream = functions(plan["inputs"]["streaming_trainer"]["path"])
    if new["calibrate_stream"] != stream["calibrate_stream"]:
        raise ValueError("verified streaming calibration implementation changed")
    old_plan = campaign.read(campaign.verify(plan["original_teacher_plan"]))
    position = old_plan["positions"][plan["original_position_index"]]
    native = json.loads(campaign.verify(plan["native_output"]).read_text())
    expected = pipeline.retained.normalized_label(copy.deepcopy(native), position, old_plan)
    actual = pipeline.normalized_label(copy.deepcopy(native), position, old_plan)
    if actual != expected:
        raise ValueError("parameterized normalizer changed a retained native target")
    cases = []
    for name, mutate in (
        ("wrong-node-budget", lambda p: p.update(nodes=256000)),
        ("wrong-bundle", lambda p: p.update(bundle_sha256="0"*64)),
    ):
        changed = copy.deepcopy(old_plan); mutate(changed)
        rejects(lambda: pipeline.normalized_label(copy.deepcopy(native), position, changed));cases.append(name)
    changed = dict(position, mover=1-position["mover"])
    rejects(lambda: pipeline.normalized_label(copy.deepcopy(native), changed, old_plan));cases.append("wrong-parent-mover")
    higher = copy.deepcopy(expected);higher["work_budget"]["max_tree_nodes"] = 256000
    corpus.paired_groups(expected, higher, position)
    for name, mutate in (
        ("wrong-parent-features", lambda g: g["parent_active"].append(6300)),
        ("wrong-successor-features", lambda g: g["successors"][0]["active"].append(6300)),
        ("wrong-successor-mover", lambda g: g["successors"][0].update(value_mover=1-g["successors"][0]["value_mover"])),
        ("changed-action-enumeration", lambda g: g["successors"].pop()),
        ("changed-exhaustiveness", lambda g: g.update(exhaustive=not g["exhaustive"])),
    ):
        changed = copy.deepcopy(higher);mutate(changed)
        rejects(lambda: corpus.paired_groups(expected, changed, position));cases.append(name)
    solved = next((i for i,r in enumerate(expected["successors"]) if r["proof"]["solved"]),None)
    if solved is not None:
        changed = copy.deepcopy(higher);changed["successors"][solved]["teacher_value"] *= -1
        rejects(lambda: corpus.paired_groups(expected, changed, position));cases.append("changed-exact-proof-target")
    result = dict(schema=SCHEMA, passed=True, plan=campaign.record(plan_path),
        unchanged_numeric_and_warm_start_functions=names, streaming_calibration_AST_equal=True,
        retained_normalizer_native_values_equal=True, rejected_cases=cases,
        structural_pair_fixture_is_synthetic=True, native_output=plan["native_output"],
        production_teacher_labels_generated=0, training_updates=0, new_games=0,
        outcome_targets=False, per_game_strategy_mining=False)
    campaign.immutable(Path(plan["output"])/"RESULT.json",result)
    campaign.immutable(Path(plan["output"])/"EXPOSURE_PENDING.json",dict(schema=SCHEMA+".exposure",
        retained_artifacts=[campaign.record(plan_path),plan["native_output"],plan["original_teacher_plan"]],
        training_eligible=False, before_next_fresh_bank=True, new_states=False))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,required=True)
    args=parser.parse_args();result=run(args.plan)
    print(json.dumps({"passed":result["passed"],"rejected_cases":result["rejected_cases"]}))


if __name__ == "__main__":
    main()
