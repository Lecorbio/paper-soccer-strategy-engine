"""Distinguish conflicting proofs from independent-search proof coverage."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from tools import rank_two_focused_teacher_budget_v1 as pipeline
from tools import rank_two_focused_teacher_budget_corpus_v1 as corpus

campaign, data = pipeline.campaign, pipeline.data
SCHEMA = pipeline.SCHEMA + ".proof-audit.v1"


def run(plan_path):
    plan = campaign.read(plan_path)
    pipeline.check_current(plan, "experiment", launch=True)
    if plan["producer"] != campaign.record(__file__):
        raise ValueError("proof audit producer changed")
    for ref in plan["inputs"].values():
        campaign.verify(ref)
    teacher = campaign.read(campaign.verify(plan["teacher_plan"]))
    execution = campaign.read(campaign.verify(plan["teacher_execution"]))
    provenance = campaign.read(campaign.verify(teacher["parent_provenance"]))["parents"]
    original_plans, counts, samples = {}, Counter(), []
    if not len(execution["parent_receipts"]) == len(teacher["positions"]) == len(provenance) == 4545:
        raise ValueError("complete paired parent coverage required")
    for index, (position, item, ref, origin) in enumerate(zip(teacher["positions"],
            teacher["matched_baseline_groups"], execution["parent_receipts"], provenance)):
        pipeline.check_current(plan, "experiment")
        source_path = origin["original_plan"]["path"]
        if source_path not in original_plans:
            original_plans[source_path] = campaign.read(campaign.verify(origin["original_plan"]))
        source_plan = original_plans[source_path]
        source_position = source_plan["positions"][origin["original_index"]]
        raw_path = Path(source_path).parent / "execution/parents" / f"{origin['original_index']:05d}" / "RESULT.json"
        prior_result = campaign.read(raw_path)
        current_result = campaign.read(campaign.verify(ref))
        if not prior_result["success"] or not current_result["success"]:
            raise ValueError("audit requires completed valid labels")
        prior_native = json.loads(data.bound(prior_result["output"]).read_text())
        current_native = json.loads(data.bound(current_result["output"]).read_text())
        old = pipeline.retained.normalized_label(prior_native, source_position, source_plan)
        new = pipeline.normalized_label(current_native, position, teacher)
        if old != corpus.group_content(item):
            raise ValueError("baseline corpus differs from its original native label")
        if any(old[k] != new[k] for k in ("canonical_state", "parent_active", "root_group_id", "game_id", "split", "mover", "edges")):
            raise ValueError("paired parent identity differs")
        a = {row["transcript"]:row for row in old["successors"]}
        b = {row["transcript"]:row for row in new["successors"]}
        if set(a) != set(b) or old["exhaustive"] != new["exhaustive"]:
            raise ValueError("complete-turn enumeration changed")
        counts["parents"] += 1
        counts["native_seed_differences"] += old["work_budget"]["seed"] != new["work_budget"]["seed"]
        root_old,root_new=prior_native["group"],current_native["group"]
        counts["prior_root_proof_unavailable_in256"] += root_old["root_solved"] and not root_new["root_solved"]
        counts["additional256_root_proofs"] += not root_old["root_solved"] and root_new["root_solved"]
        if root_old["root_solved"] and root_new["root_solved"] and root_old["proven_winner"] != root_new["proven_winner"]:
            counts["opposing_root_proofs"] += 1
            if len(samples)<12:samples.append(dict(index=index,position_id=position["position_id"],kind="opposing-root-proofs"))
        for action,x in a.items():
            y=b[action]
            if x["active"] != y["active"] or x["value_mover"] != y["value_mover"]:
                raise ValueError("successor feature or mover identity differs")
            counts["successors"] += 1
            sx,sy=x["proof"]["solved"],y["proof"]["solved"]
            counts["prior_successor_proof_unavailable_in256"] += sx and not sy
            counts["additional256_successor_proofs"] += not sx and sy
            if sx and sy and (x["proof"]["proven_winner"] != y["proof"]["proven_winner"] or x["teacher_value"] != y["teacher_value"]):
                counts["opposing_successor_proofs"] += 1
                if len(samples)<12:samples.append(dict(index=index,position_id=position["position_id"],action=action,kind="opposing-successor-proofs"))
            elif sx and not sy and len(samples)<12:
                samples.append(dict(index=index,position_id=position["position_id"],action=action,
                    kind="prior-known-proof-versus-new-unknown",known_proof=x["proof"],
                    known_value=x["teacher_value"],new_estimate=y["teacher_value"],
                    native64_output=prior_result["output"],native256_output=current_result["output"]))
        if index%64==0:
            campaign.atomic(Path(plan["output"])/"PROGRESS.json",dict(completed_parents=index+1,total_parents=4545))
    for key in ("opposing_root_proofs","opposing_successor_proofs"):
        counts.setdefault(key,0)
    result=dict(schema=SCHEMA,passed=True,plan=campaign.record(plan_path),counts=dict(counts),samples=samples,
        genuine_proof_contradictions=counts["opposing_root_proofs"]+counts["opposing_successor_proofs"],
        all4545_parents_checked=True,new_games=0,new_teacher_queries=0,training_updates=0,
        gameplay_strategy_mining=False,unknown_is_not_an_opposing_proof=True,
        monotonic_proof_coverage_not_assumed=True)
    campaign.immutable(Path(plan["output"])/"RESULT.json",result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,required=True)
    args=parser.parse_args();result=run(args.plan)
    print(json.dumps({"passed":result["passed"],"counts":result["counts"]}))


if __name__ == "__main__":
    main()
