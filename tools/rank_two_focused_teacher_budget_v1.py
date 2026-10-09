"""Bind matched teacher budgets to retained positions and a common warm start."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

from tools import rank_two_focused_campaign_v1 as campaign
from tools import rank_two_focused_data_v3 as retained
from tools import rank_two_focused_work_v1 as work

data, labels = retained.data, retained.labels
SCHEMA = campaign.SCHEMA + ".teacher-budget.v1"
CAMPAIGN_ID = retained.CAMPAIGN_ID


def check_current(plan, action, launch=False):
    current = work.checked(plan, action, launch=launch)
    if (current.get("focused_family") != "dd8"
            or current.get("focused_family_lock") != plan["family_lock"]
            or current.get("focused_round_02_declaration") != plan["declaration"]):
        raise PermissionError("matched teacher round/family binding changed")
    declaration = campaign.read(campaign.verify(plan["declaration"]))
    if (declaration["principal_intervention"] != "matched-teacher-node-budget"
            or declaration["architecture"] != [6301, 8, 8, 1]
            or declaration["teacher_nodes"] != [64000, 256000]
            or declaration["outcomes_as_targets"] is not False):
        raise ValueError("declared matched teacher recipe differs")
    return current


def parent_key(position):
    return (position["game_id"], position["root_group_id"], position["split"],
            position["canonical_state"], position["mover"], tuple(position["parent_active"]))


def normalized_label(native, position, plan):
    native = labels.validate_complete_turn_action_group(native)
    group, source = native["group"], native["group"]["source_binding"]
    if (native["source_bundle_body_sha256"] != plan["bundle_sha256"]
            or native["teacher"]["artifact_sha256"] != data.TEACHER_SHA256
            or group["work_budget"]["max_tree_nodes"] != plan["nodes"]
            or group["work_budget"]["max_time_ms"] != 0
            or source["campaign_id"] != CAMPAIGN_ID):
        raise ValueError("teacher/budget/recipe binding differs")
    prefix = "/".join(row["action"] for row in source["prefix"])
    if (prefix != position["prefix"] or group["parent_mover"] != position["mover"]
            or any(source[key] != position[key] for key in
                   ("position_id", "root_group_id", "group_id", "split", "winner", "source"))):
        raise ValueError("label is not bound to the exact retained parent")
    return dict(group_id=group["parent_identity"], root_group_id=position["root_group_id"],
                game_id=position["game_id"], split=position["split"], mover=position["mover"],
                edges=position["edges"], parent_active=position["parent_active"],
                canonical_state=position["canonical_state"], exhaustive=group["successors_exhaustive"],
                teacher_value=group["root_value"], successors=group["successors"],
                teacher=native["teacher"], work_budget=group["work_budget"])


def teacher_command(plan, digest):
    return [plan["native_teacher"]["path"], "--model", plan["teacher"]["path"],
            "--model-sha256", data.TEACHER_SHA256, "--campaign-id", CAMPAIGN_ID,
            "--tree-nodes", str(plan["nodes"]), "--time-ms", "0", "--max-actions", "250",
            "--max-partial-paths", "50000", "--exploration", "0.5", "--fpu", "0.5",
            "--emit-action-groups", "--source-bundle-body-sha256", digest]


def verify_teacher_plan(plan):
    check_current(plan, "label")
    declaration = campaign.read(campaign.verify(plan["declaration"]))
    body = {key: value for key, value in plan.items() if key not in ("bundle_sha256", "command")}
    if hashlib.sha256(campaign.canonical(body)).hexdigest() != plan["bundle_sha256"]:
        raise ValueError("teacher source bundle body changed")
    if (plan["producer"] != campaign.record(__file__)
            or plan["teacher"] != declaration["teacher"]
            or plan["native_teacher"] != declaration["native_teacher"]
            or plan["nodes"] != 256000 or plan["outcomes_as_targets"] is not False
            or len(plan["positions"]) != declaration["retained_parent_count"]
            or len(plan["matched_baseline_groups"]) != len(plan["positions"])
            or len(plan["positions"]) > 6144
            or plan["command"] != teacher_command(plan, plan["bundle_sha256"])):
        raise ValueError("retained matched parent or teacher recipe differs")
    return True


def prepare(plan_path):
    plan = campaign.read(plan_path)
    check_current(plan, "label", launch=True)
    if plan["producer"] != campaign.record(__file__):
        raise ValueError("matched-parent preparation producer differs")
    for ref in plan["inputs"].values():
        campaign.verify(ref)
    declaration = campaign.read(campaign.verify(plan["declaration"]))
    corpus = campaign.read(campaign.verify(declaration["retained_corpus"]))
    out = Path(plan["output"])
    campaign.immutable(out / "CLAIM.json", dict(plan=campaign.record(plan_path),
        parent_rule="every retained eligible group; exact physical parent; no scores or outcomes"))
    original = {}
    for ref in declaration["original_teacher_plans"]:
        original_plan = campaign.read(campaign.verify(ref))
        for index, position in enumerate(original_plan["positions"]):
            key = parent_key(position)
            if key in original:
                raise ValueError("retained parent has ambiguous original provenance")
            original[key] = (position, ref, index)
    positions, matched, provenance = [], [], []
    for item in sorted(corpus["group_artifacts"], key=lambda x: (x["split"] != "train", x["root_group_id"], x["group_id"])):
        check_current(plan, "label")
        group = json.loads(gzip.decompress(data.bound(item["artifact"]).read_bytes()))
        if any(group[key] != item[key] for key in ("group_id", "root_group_id", "split", "edges")):
            raise ValueError("retained corpus/group content differs")
        position, reference, index = original[parent_key(group)]
        if group["edges"] != position["edges"] or group["work_budget"]["max_tree_nodes"] != 64000:
            raise ValueError("retained physical parent/baseline budget differs")
        positions.append(position); matched.append(item)
        provenance.append(dict(original_plan=reference, original_index=index,
                               original_position_id=position["position_id"], baseline_group=item))
    counts = Counter(row["split"] for row in positions)
    roots = {split: sorted({p["root_group_id"] for p in positions if p["split"] == split})
             for split in ("train", "validation")}
    if (len(positions) != 4545 or counts != {"train": 3891, "validation": 654}
            or len(roots["train"]) != 512 or len(roots["validation"]) != 128
            or set(roots["train"]) & set(roots["validation"])
            or len({p["position_id"] for p in positions}) != len(positions)):
        raise ValueError("complete retained512TRAIN/128VALIDATION matched parent coverage differs")
    positions_file = out / "positions.tsv"
    payload = data.HEADER + "".join("\t".join(str(p[k]) for k in
        ("position_id", "root_group_id", "group_id", "source", "split", "winner", "mover", "prefix")) + "\n"
        for p in positions)
    campaign.immutable(positions_file, payload.encode("ascii"))
    provenance_ref = campaign.immutable(out / "PARENT_PROVENANCE.json", dict(parents=provenance,
        roots=roots, counts=dict(counts), retained_corpus=declaration["retained_corpus"],
        outcome_fields_are_provenance_only=True, no_new_games=True))
    sources_ref = campaign.immutable(out / "RETAINED_GAMES.json", dict(
        original_teacher_plans=declaration["original_teacher_plans"],
        retained_corpus=declaration["retained_corpus"], parent_provenance=provenance_ref, new_games=0))
    body = dict(schema=SCHEMA + ".teacher-plan", producer=campaign.record(__file__),
        campaign_current=plan["campaign_current"], owner_thread_id=plan["owner_thread_id"],
        activation=plan["activation"], family_lock=plan["family_lock"], declaration=plan["declaration"],
        positions=positions, matched_baseline_groups=matched, positions_file=data.record(positions_file),
        games=data.record(Path(sources_ref["path"])), parent_provenance=provenance_ref,
        teacher=declaration["teacher"], native_teacher=declaration["native_teacher"], nodes=256000,
        maximum_workers=10, maximum_rss_bytes=18*1024**3, maximum_position_cpu_seconds=480,
        maximum_position_wall_seconds=490, numerical_threads=1, outcomes_as_targets=False,
        validator=campaign.record(labels.__file__), lifecycle=plan["inputs"]["work_guard"],
        resource_budget=declaration["stage_resources"]["teacher"],
        inputs=dict(label_runner=declaration["producers"]["labels"], declaration=plan["declaration"],
            retained_corpus=declaration["retained_corpus"], parent_provenance=provenance_ref,
            original_teacher_01=declaration["original_teacher_plans"][0],
            original_teacher_02=declaration["original_teacher_plans"][1]))
    digest = hashlib.sha256(campaign.canonical(body)).hexdigest()
    teacher_plan = dict(body, bundle_sha256=digest, command=teacher_command(body, digest))
    verify_teacher_plan(teacher_plan)
    teacher_ref = campaign.immutable(out / "TEACHER_PLAN.json", teacher_plan)
    campaign.immutable(out / "EXPOSURE_PENDING.json", dict(schema=SCHEMA + ".teacher-exposure",
        plan=teacher_ref, training_eligible=False, before_next_fresh_bank=True,
        all_retained_parent_inputs_preserved=True))
    result = dict(passed=True, plan=campaign.record(plan_path), teacher_plan=teacher_ref,
        parent_provenance=provenance_ref, matched_parents=len(positions), split_counts=dict(counts),
        train_roots=512, validation_roots=128, new_games=0, outcome_targets=False,
        exact_parent_and_feature_identity=True)
    campaign.immutable(out / "RESULT.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.plan)
    print(json.dumps({key: result[key] for key in ("passed", "matched_parents", "new_games")}))


if __name__ == "__main__":
    main()
