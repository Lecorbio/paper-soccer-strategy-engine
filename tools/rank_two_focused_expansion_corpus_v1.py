"""Merge retained TRAIN roots with declared expansion and fresh validation."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

from tools import rank_two_focused_campaign_v1 as campaign
from tools import rank_two_focused_expansion_label_data_v1 as pipeline
from tools import rank_two_focused_labels_v4 as labels

data = pipeline.data
SCHEMA = campaign.SCHEMA + ".expansion-corpus.v1"


def checked(plan, launch=False):
    current = pipeline.check_current(plan, "train", launch=launch)
    if current["focused_round_01_teacher_plan"] != plan["teacher_plan"]:
        raise ValueError("declared teacher plan changed")
    for reference in plan["inputs"].values():
        campaign.verify(reference)
    return current


def group_content(item):
    group = json.loads(gzip.decompress(data.bound(item["artifact"]).read_bytes()))
    if any(group[key] != item[key] for key in ("group_id", "root_group_id", "split", "edges")):
        raise ValueError("bound label group metadata differs")
    return group


def feature_keys(group):
    return {data.feature_key(group["parent_active"])} | {
        data.feature_key(row["active"]) for row in group["successors"]}


def comparable_early(group):
    values = [row["teacher_value"] * (1 if row["value_mover"] == group["mover"] else -1)
              for row in group["successors"]]
    return (group["edges"] <= 12 and group["exhaustive"] and len(values) > 1
            and max(values) - min(values) > 1e-4)


def check_selection_rules():
    # Fixed split guards depend only on feature identities, never losses/outcomes.
    train, reserved = {"training-parent", "training-child"}, {"reserved-parent"}
    assert not ({"fresh-parent", "fresh-child"} & (train | reserved))
    assert {"fresh-parent", "training-child"} & train
    assert {"fresh-parent", "reserved-parent"} & reserved
    active = pipeline.rules.encode_active(pipeline.rules.ReplayState())
    assert data.feature_key(active) == data.feature_key(pipeline.rules.reflect_active(active))
    return dict(passed=True, training_and_reserved_feature_overlap_rejected=True,
                reflected_feature_identity=True, outcomes_or_metrics_used=False,
                synthetic_case=dict(prefix="", training_eligible=False))


def run(plan_path):
    plan = campaign.read(plan_path)
    checked(plan, launch=True)
    if plan["schema"] != SCHEMA + ".plan" or plan["producer"] != campaign.record(__file__):
        raise ValueError("expansion corpus producer/plan differs")
    out = Path(plan["output"])
    if (out / "CLAIM.json").exists():
        raise RuntimeError("retained corpus preparation claim requires explicit classification")
    campaign.immutable(out / "CLAIM.json", dict(plan=campaign.record(plan_path),
        outcomes_as_targets=False, split_filter="frozen feature identities only"))
    campaign.immutable(out / "RESERVED_EXPOSURE.json", dict(schema=SCHEMA + ".exposure",
        plan=campaign.record(plan_path), input=plan["label_execution"],
        training_eligible=False, before_next_fresh_bank=True))
    teacher_plan = campaign.read(campaign.verify(plan["teacher_plan"]))
    pipeline.verify_teacher_plan(teacher_plan)
    if teacher_plan["previous_corpus"] != plan["previous_corpus"]:
        raise ValueError("retained teacher corpus binding differs")
    execution = campaign.read(campaign.verify(plan["label_execution"]))
    if execution["requested_parents"] != len(teacher_plan["positions"]):
        raise ValueError("declared teacher parent coverage differs")
    labels.import_corpus(campaign.verify(plan["teacher_plan"]),
                         campaign.verify(plan["label_execution"]), out / "NEW_CORPUS.json")
    fresh = campaign.read(out / "NEW_CORPUS.json")
    previous = campaign.read(campaign.verify(plan["previous_corpus"]))
    for manifest in (previous, fresh):
        if manifest["teacher_runtime_sha256"] != data.TEACHER_SHA256 or manifest["outcomes_as_targets"] is not False:
            raise ValueError("same accepted teacher and teacher-only targets required")
    declaration = campaign.read(campaign.verify(plan["declaration"]))
    generation = campaign.read(campaign.verify(teacher_plan["generation_plan"]))
    bank = campaign.read(campaign.verify(generation["bank"]))
    previous_train_roots = {row["root_group_id"] for row in previous["group_artifacts"] if row["split"] == "train"}
    previous_validation_roots = {row["root_group_id"] for row in previous["group_artifacts"] if row["split"] == "validation"}
    fresh_train_roots = {row["root_group_id"] for row in bank["rows"] if row["split"] == "train"}
    fresh_validation_roots = {row["root_group_id"] for row in bank["rows"] if row["split"] == "validation"}
    if (len(previous_train_roots) != 128 or len(previous_validation_roots) != 128
            or len(fresh_train_roots) != 384 or len(fresh_validation_roots) != 128
            or (previous_train_roots | previous_validation_roots) & (fresh_train_roots | fresh_validation_roots)):
        raise ValueError("retained128/fresh384/independent128 root bindings differ")
    training_keys, reserved_keys, parent_keys, ids = set(), set(), set(), set()
    merged, excluded, validation_roots, early_roots = [], [], set(), set()
    checks = check_selection_rules()
    for index, item in enumerate(previous["group_artifacts"]):
        if index % 32 == 0:
            pipeline.check_current(plan, "train")
        group = group_content(item)
        keys = feature_keys(group)
        if item["split"] == "validation":
            reserved_keys.update(keys)
            continue
        if item["split"] != "train":
            raise ValueError("unknown retained label split")
        training_keys.update(keys)
        parent_keys.add(data.feature_key(group["parent_active"]))
        if item["group_id"] in ids:
            raise ValueError("duplicate retained training label identity")
        ids.add(item["group_id"]); merged.append(item)
    if training_keys & reserved_keys:
        raise ValueError("retained corpus violates its original validation separation")
    order = sorted(fresh["group_artifacts"], key=lambda row: (row["split"] != "train", row["root_group_id"], row["group_id"]))
    for index, item in enumerate(order):
        if index % 32 == 0:
            pipeline.check_current(plan, "train")
        group = group_content(item)
        keys = feature_keys(group)
        parent_key = data.feature_key(group["parent_active"])
        reason = None
        if item["split"] == "train":
            if item["root_group_id"] not in fresh_train_roots:
                raise ValueError("new TRAIN label lacks its frozen root")
            if keys & reserved_keys:
                reason = "previous-validation-features-remain-reserved"
            elif parent_key in parent_keys:
                reason = "duplicate-retained-or-new-training-parent"
            if reason is None:
                training_keys.update(keys); parent_keys.add(parent_key)
        elif item["split"] == "validation":
            if item["root_group_id"] not in fresh_validation_roots:
                raise ValueError("new VALIDATION label lacks its frozen root")
            if keys & (training_keys | reserved_keys):
                reason = "fresh-validation-overlaps-training-or-reserved-features"
            if reason is None:
                validation_roots.add(item["root_group_id"])
                if comparable_early(group):
                    early_roots.add(item["root_group_id"])
        else:
            raise ValueError("unknown expansion label split")
        if reason:
            excluded.append(dict(group=item, reason=reason))
            continue
        if item["group_id"] in ids:
            raise ValueError("merged label identity collision")
        ids.add(item["group_id"]); merged.append(item)
    expected_train = previous_train_roots | fresh_train_roots
    actual_train = {row["root_group_id"] for row in merged if row["split"] == "train"}
    if actual_train != expected_train or len(actual_train) != declaration["training_roots_after"]:
        raise ValueError("expanded training does not retain every declared512 root")
    if len(validation_roots) < 100 or len(early_roots) < 100:
        raise ValueError("insufficient independent comparable early validation roots after all split guards")
    if previous_validation_roots & {row["root_group_id"] for row in merged}:
        raise ValueError("reserved previous validation root entered the merged corpus")
    campaign.immutable(out / "SPLIT_EXCLUSIONS.json", dict(groups=excluded,
        new_import_excluded_group_ids=fresh["excluded_group_ids"],
        selection_uses_outcomes_or_validation_metrics=False,
        previous_validation_roots_reserved=sorted(previous_validation_roots)))
    manifest = dict(schema=data.SCHEMA, kind="fresh-complete-turn-search-corpus",
        teacher_runtime_sha256=data.TEACHER_SHA256, label_plan=data.record(campaign.verify(plan["teacher_plan"])),
        execution=plan["label_execution"], group_artifacts=merged,
        storage="per-position-gzip-json-streamed-by-root", outcomes_as_targets=False,
        generation_plan=teacher_plan["generation_plan"], previous_corpus=plan["previous_corpus"],
        expansion_declaration=plan["declaration"], expansion_merge_plan=campaign.record(plan_path),
        new_corpus=campaign.record(out / "NEW_CORPUS.json"), old_validation_reserved=True,
        retained_train_root_group_ids=sorted(previous_train_roots),
        fresh_train_root_group_ids=sorted(fresh_train_roots),
        fresh_validation_root_group_ids=sorted(fresh_validation_roots),
        eligible_validation_root_group_ids=sorted(validation_roots),
        reserved_previous_validation_root_group_ids=sorted(previous_validation_roots),
        split_exclusions=campaign.record(out / "SPLIT_EXCLUSIONS.json"),
        validation_overlap_groups_removed=len(excluded) + fresh["validation_overlap_groups_removed"],
        independent_comparable_early_validation_root_groups=len(early_roots),
        production_training_admitted=True, training_started=False,
        importer_source=data.record(__file__))
    corpus = campaign.immutable(out / "CORPUS.json", manifest)
    campaign.immutable(out / "CHECKS.json", checks)
    result = dict(passed=True, plan=campaign.record(plan_path), corpus=corpus,
        teacher_execution=plan["label_execution"], teacher_parents=len(teacher_plan["positions"]),
        valid_labels=execution["valid_labels"], unsupported_labels=execution["unsupported_labels"],
        train_roots=len(actual_train), validation_roots=len(validation_roots),
        generated_fresh_validation_roots=len(fresh_validation_roots),
        comparable_early_validation_roots=len(early_roots), group_count=len(merged),
        split_groups=dict(Counter(row["split"] for row in merged)),
        excluded_group_count=len(excluded) + fresh["validation_overlap_groups_removed"],
        previous_training_roots_retained=len(previous_train_roots),
        previous_validation_roots_reserved=len(previous_validation_roots),
        exact_native_targets_and_mover_signs_preserved=True, outcomes_as_targets=False,
        training_validation_reflected_features_disjoint=True, checks=campaign.record(out / "CHECKS.json"))
    campaign.immutable(out / "RESULT.json", result)
    campaign.immutable(out / "EXPOSURE_PENDING.json", dict(schema=SCHEMA + ".exposure",
        input=plan["label_execution"], result=campaign.record(out / "RESULT.json"),
        cases=[campaign.record(out / "SPLIT_EXCLUSIONS.json"), campaign.record(out / "CHECKS.json")],
        all_teacher_claims_outputs_and_excluded_groups_retained=True,
        training_eligible=False, before_next_fresh_bank=True))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.plan)))


if __name__ == "__main__":
    main()
