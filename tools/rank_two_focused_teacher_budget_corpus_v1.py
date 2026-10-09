"""Build paired training corpora with identical parents and common validation."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

from tools import rank_two_focused_teacher_budget_v1 as pipeline
from tools import rank_two_focused_labels_v5 as labels
from tools import rank_two_focused_expansion_corpus_v1 as retained

campaign, data = pipeline.campaign, pipeline.data
SCHEMA = campaign.SCHEMA + ".teacher-budget-corpus.v1"


def group_content(item):
    return retained.group_content(item)


def paired_groups(old, new, position):
    if any(old[k] != new[k] for k in ("canonical_state", "parent_active", "root_group_id",
                                     "game_id", "split", "mover", "edges")):
        raise ValueError("matched teacher parent identities differ")
    if old["work_budget"]["max_tree_nodes"] != 64000 or new["work_budget"]["max_tree_nodes"] != 256000:
        raise ValueError("declared matched teacher budgets differ")
    def successors(group):
        values = {row["transcript"]: row for row in group["successors"]}
        if len(values) != len(group["successors"]):
            raise ValueError("duplicate successor action")
        return values
    a, b = successors(old), successors(new)
    if set(a) != set(b) or old["exhaustive"] != new["exhaustive"]:
        raise ValueError("fixed complete-turn enumeration changed across teacher budgets")
    for action in sorted(a):
        x, y = a[action], b[action]
        if x["active"] != y["active"] or x["value_mover"] != y["value_mover"]:
            raise ValueError("native successor feature or mover identity changed")
        if x["proof"]["solved"] and (not y["proof"]["solved"]
                or x["proof"]["proven_winner"] != y["proof"]["proven_winner"]
                or x["teacher_value"] != y["teacher_value"]):
            raise ValueError("previous exact terminal/proof target contradicted")
    # Keep paired group IDs identical so root position averaging and update order
    # cannot differ merely because a source bundle identity changed.
    new = dict(new, group_id=old["group_id"])
    return old, new


def run(plan_path):
    plan = campaign.read(plan_path)
    pipeline.check_current(plan, "train", launch=True)
    if plan["producer"] != campaign.record(__file__):
        raise ValueError("paired corpus producer differs")
    for ref in plan["inputs"].values():
        campaign.verify(ref)
    teacher_path = campaign.verify(plan["teacher_plan"])
    teacher_plan = campaign.read(teacher_path)
    pipeline.verify_teacher_plan(teacher_plan)
    execution = campaign.read(campaign.verify(plan["label_execution"]))
    recipe = campaign.read(Path(plan["label_execution"]["path"]).with_name("EXECUTION.json"))
    if (not execution["passed"] or execution["plan"] != plan["teacher_plan"]
            or execution["requested_parents"] != len(teacher_plan["positions"])
            or len(execution["parent_receipts"]) != len(teacher_plan["positions"])
            or recipe["producer"] != campaign.record(labels.__file__)
            or recipe["plan"] != plan["teacher_plan"]):
        raise ValueError("complete matching teacher execution required")
    out = Path(plan["output"])
    campaign.immutable(out / "CLAIM.json", dict(plan=campaign.record(plan_path),
        validation_rule="common256k labels; all structural exclusions shared; no score filtering"))
    paired, unsupported, changed = [], [], Counter()
    for index, (position, item, ref) in enumerate(zip(teacher_plan["positions"],
            teacher_plan["matched_baseline_groups"], execution["parent_receipts"])):
        pipeline.check_current(plan, "train")
        result = campaign.read(campaign.verify(ref))
        if (result["position_id"] != position["position_id"]
                or result["plan_body_sha256"] != teacher_plan["bundle_sha256"]):
            raise ValueError("teacher parent receipt order/body differs")
        if result["unsupported"]:
            unsupported.append(dict(index=index, position_id=position["position_id"], receipt=ref))
            continue
        if not result["success"]:
            raise ValueError("unexpected label failure is retained")
        with data.bound(result["output"]).open() as stream:
            new = pipeline.normalized_label(json.loads(stream.readline()), position, teacher_plan)
            if stream.readline():
                raise ValueError("multiple parent outputs")
        old = group_content(item)
        old, new = paired_groups(old, new, position)
        a = {r["transcript"]: r for r in old["successors"]}
        b = {r["transcript"]: r for r in new["successors"]}
        changed["parents"] += 1
        changed["successors"] += len(a)
        changed["value_changes"] += sum(a[k]["teacher_value"] != b[k]["teacher_value"] for k in a)
        changed["proof_changes"] += sum(a[k]["proof"] != b[k]["proof"] for k in a)
        paired.append((item, old, new, ref))
    train_keys, validation_keys, reserved_keys = set(), set(), set()
    prior = campaign.read(campaign.verify(plan["original_corpus"]))
    for item in prior["group_artifacts"]:
        if item["split"] == "validation":
            reserved_keys.update(retained.feature_keys(group_content(item)))
    for item, old, new, ref in paired:
        keys = retained.feature_keys(new)
        if item["split"] == "train":
            train_keys.update(keys)
        else:
            validation_keys.update(keys)
    if train_keys & (validation_keys | reserved_keys) or validation_keys & reserved_keys:
        raise ValueError("matched training/validation/reserved reflected feature leakage")
    early = {item["root_group_id"] for item, old, new, ref in paired
             if item["split"] == "validation" and retained.comparable_early(new)}
    roots = {split: sorted({item["root_group_id"] for item, old, new, ref in paired
                          if item["split"] == split}) for split in ("train", "validation")}
    if len(roots["train"]) != 512 or len(roots["validation"]) != 128 or len(early) < 100:
        raise ValueError("paired512TRAIN/128VALIDATION independent early gate failed")
    groups64, groups256, diagnostic = [], [], []
    for item, old, new, ref in paired:
        path = out / "groups256" / (item["group_id"] + ".json.gz")
        campaign.immutable(path, gzip.compress(campaign.canonical(new), mtime=0))
        high = dict(item, comparable_early=retained.comparable_early(new), artifact=data.record(path))
        groups256.append(high)
        diagnostic.append(item)
        groups64.append(item if item["split"] == "train" else high)
    base = dict(schema=data.SCHEMA, kind="fresh-complete-turn-search-corpus",
        teacher_runtime_sha256=data.TEACHER_SHA256, teacher_budget_declaration=plan["declaration"],
        retained_corpus=plan["retained_corpus"], label_plan=plan["teacher_plan"],
        execution=plan["label_execution"], importer_source=campaign.record(__file__),
        storage="per-position-gzip-json-streamed-by-root", production_training_admitted=True,
        training_started=False, outcomes_as_targets=False, old_validation_reserved=True,
        validation_overlap_groups_removed=0, shared_unsupported_positions=unsupported,
        independent_comparable_early_validation_root_groups=len(early),
        validation_teacher_nodes=256000, common_parent_and_update_order=True,
        train_root_group_ids=roots["train"], eligible_validation_root_group_ids=roots["validation"],
        exact_native_targets_and_mover_signs_preserved=True)
    low_ref = campaign.immutable(out / "CORPUS64.json", dict(base,
        training_teacher_nodes=64000, group_artifacts=groups64))
    high_ref = campaign.immutable(out / "CORPUS256.json", dict(base,
        training_teacher_nodes=256000, group_artifacts=groups256))
    diag_ref = campaign.immutable(out / "DIAGNOSTIC64.json", dict(base,
        training_teacher_nodes=64000, validation_teacher_nodes=64000, group_artifacts=diagnostic))
    result = dict(passed=True, plan=campaign.record(plan_path), corpora={"teacher64":low_ref,"teacher256":high_ref},
        diagnostic64=diag_ref, label_execution=plan["label_execution"], matched_parents=len(paired),
        unsupported_positions=unsupported, label_changes=dict(changed), train_roots=512, validation_roots=128,
        comparable_early_roots=len(early), common256_validation=True,
        training_validation_reflected_features_disjoint=True, reserved_validation_preserved=True,
        exact_targets_and_mover_signs=True, new_games=0)
    campaign.immutable(out / "RESULT.json", result)
    campaign.immutable(out / "EXPOSURE_PENDING.json", dict(schema=SCHEMA + ".exposure",
        input=plan["label_execution"], retained_artifacts=[campaign.record(plan_path), low_ref, high_ref, diag_ref],
        training_eligible=False, before_next_fresh_bank=True))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.plan)
    print(json.dumps({k:result[k] for k in ("passed","matched_parents","comparable_early_roots")}))


if __name__ == "__main__":
    main()
