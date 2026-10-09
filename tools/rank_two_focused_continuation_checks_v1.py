"""Check continuation bindings and exact shared warm-start state before fitting."""
from __future__ import annotations

import argparse
import ast
import copy
import json
from pathlib import Path

from tools import rank_two_focused_training_v9 as training
from tools import rank_two_focused_training_v8 as predecessor
from tools import compact_representation_train as reader
from tools import rank_two_focused_campaign_v1 as campaign
from tools import rank_two_focused_work_v1 as work

SCHEMA = campaign.SCHEMA + ".continuation-preflight.v1"
UNCHANGED = ("initialize", "quantize", "forward", "root_rows", "objective",
             "update_root", "diagnostics", "calibrate", "calibrate_stream",
             "stopping", "checkpoint", "restore", "runtime", "validation", "warm_start")


def functions(path):
    return {node.name: ast.dump(node, include_attributes=False)
            for node in ast.parse(Path(path).read_text()).body
            if isinstance(node, ast.FunctionDef)}


def run(plan_path):
    plan = campaign.read(plan_path)
    work.checked(plan, "experiment", launch=True)
    if plan["producer"] != campaign.record(__file__):
        raise ValueError("continuation preflight producer differs")
    for reference in plan["inputs"].values():
        campaign.verify(reference)
    old, new = functions(predecessor.__file__), functions(training.__file__)
    if any(old[name] != new[name] for name in UNCHANGED):
        raise ValueError("numerical or warm-start implementation changed")
    output = Path(plan["output"])
    if (output / "CLAIM.json").exists():
        raise RuntimeError("spent preflight claim retained")
    campaign.immutable(output / "CLAIM.json", dict(plan=campaign.record(plan_path),
        production_training_updates=0, new_games=0))
    reports = []
    for reference in plan["arms"]:
        arm = campaign.read(campaign.verify(reference))
        corpus = reader.Corpus(campaign.verify(arm["corpus"]))
        training.validate_corpus_binding(arm, corpus)
        rejected = []
        mutations = (
            ("float_rate", lambda p: p["recipe"].update(float_learning_rate=.0002)),
            ("loss", lambda p: p["recipe"].update(ranking_weight=.5)),
            ("epoch_cap", lambda p: p["recipe"].update(float_maximum=81)),
            ("arm", lambda p: p.update(arm="undeclared")),
            ("corpus", lambda p: p.update(corpus=p["diagnostic64_corpus"])),
            ("declaration", lambda p: p.update(declaration=p["supervision_plan"])),
        )
        for label, mutate in mutations:
            changed = copy.deepcopy(arm)
            mutate(changed)
            try:
                training.validate_corpus_binding(changed, corpus)
            except (ValueError, PermissionError, KeyError):
                rejected.append(label)
            else:
                raise AssertionError("mutated continuation binding accepted: " + label)
        architecture, parameters = training.initialize("dd8", 0)
        optimizer = training.core.AdamW(parameters, learning_rate=.001, weight_decay=1e-5)
        anchor = training.warm_start(arm, reference, parameters, optimizer,
                                     output / arm["arm"])
        if (anchor["storage"]["raw_sha256"] != arm["parent_checkpoint"]["storage"]["raw_sha256"]
                or anchor["parameter_sha256"] != arm["parent_checkpoint"]["parameter_sha256"]
                or anchor["optimizer_step"] != arm["parent_checkpoint"]["optimizer_step"]):
            raise ValueError("shared warm-start parameters or moments changed")
        reports.append(dict(arm=arm["arm"], plan=reference, anchor=anchor,
                            rejected_mutations=rejected, training_updates=0))
    if len(reports) != 2 or reports[0]["anchor"]["storage"]["raw_sha256"] != reports[1]["anchor"]["storage"]["raw_sha256"]:
        raise ValueError("two arms must begin from byte-identical shared state")
    result = dict(schema=SCHEMA, passed=True, plan=campaign.record(plan_path),
        arms=reports, unchanged_numerical_functions=list(UNCHANGED),
        shared_parameters_moments_cursor_identical=True, rejected_mutations=12,
        production_training_updates=0, new_games=0, mechanical_warm_start_anchors=2)
    campaign.immutable(output / "RESULT.json", result)
    campaign.immutable(output / "EXPOSURE_PENDING.json", dict(
        schema=SCHEMA + ".exposure", retained_artifacts=[campaign.record(output / "RESULT.json")],
        retained_plans=[campaign.record(plan_path)], before_next_fresh_bank=True,
        training_eligible=False, new_games=0, states_generated=0))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.plan)
    print(json.dumps(dict(passed=result["passed"], rejected_mutations=12,
                          production_training_updates=0, new_games=0)))


if __name__ == "__main__":
    main()
