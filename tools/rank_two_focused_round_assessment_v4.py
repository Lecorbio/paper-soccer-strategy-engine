"""Archive every declared paired slot and assess only complete whole-root panels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools import rank_two_focused_matched_assessment_v1 as retained

campaign, runtime, work = retained.campaign, retained.runtime, retained.work
SCHEMA = campaign.SCHEMA + ".round-assessment.v4"


def run(plan_path):
    plan = campaign.read(plan_path)
    if plan["producer"] != campaign.record(__file__):
        raise ValueError("round assessment producer differs")
    current = work.checked(plan, "experiment", launch=True)
    for reference in plan["inputs"].values():
        campaign.verify(reference)
    gameplay = campaign.read(campaign.verify(plan["gameplay_plan"]))
    result = campaign.read(campaign.verify(plan["result"]))
    bank = campaign.read(campaign.verify(gameplay["bank"]))
    bundle = campaign.read(campaign.verify(gameplay["runtime"]))
    if (not result["passed"] or result["own_failures"] != 0
            or result["completed_games"] != gameplay["games"]
            or result["plan"] != plan["gameplay_plan"]
            or len(result["root_receipts"]) != gameplay["roots"]
            or result["bank"] != gameplay["bank"] or result["runtime"] != gameplay["runtime"]
            or result["canceled_unclaimed_slots"] != 0
            or result["original_declared_slots"] != gameplay["games"]
            or not result["all_declared_games_archived"] or len(bank["rows"]) != gameplay["roots"]):
        raise ValueError("complete exact promotion panel required")
    if (gameplay["offline_roster"] != current["focused_active_offline_roster"]
            or gameplay["roster_approval"] != current["focused_h62_roster_exception_approval"]
            or gameplay["runtime"] != current["focused_round_04_runtime_bundle"]):
        raise ValueError("approved roster/runtime binding differs")
    if len(gameplay["candidates"]) != 1:
        raise ValueError("one frozen round candidate required")
    candidate = gameplay["candidates"][0]
    names, roots, seen, decisions = ["control", candidate], [], set(), 0
    for index, (row, reference) in enumerate(zip(bank["rows"], result["root_receipts"])):
        work.checked(plan, "experiment")
        root = campaign.read(campaign.verify(reference))
        order = names[index % 2:] + names[:index % 2]
        if (row["index"] != index or root["row"] != row or root["plan"] != plan["gameplay_plan"]
                or root["execution_order"] != order or set(root["arms"]) != set(names)
                or not root["control_reused_once_per_root_color"]):
            raise ValueError("root/control/order binding differs")
        wins = {}
        for name in names:
            if len(root["arms"][name]) != 2:
                raise ValueError("both colors required")
            pair = []
            actor = bundle["actors"][name]
            opponent = bundle["actors"]["opponent:" + row["opponent"]]
            for color, game_ref in enumerate(root["arms"][name]):
                if game_ref["path"] in seen:
                    raise ValueError("game reused across slots")
                seen.add(game_ref["path"])
                path = campaign.verify(game_ref)
                document, claim = campaign.read(path), campaign.read(path.with_name("CLAIM.json"))
                if claim["actor"] != name:
                    raise ValueError("game assigned to another source")
                pair.append(retained.validate_game(document, claim, row, color, actor, opponent,
                                                   plan["gameplay_plan"], gameplay["runtime"]))
                decisions += len(document["game"]["decisions"])
            wins[name] = sum(pair) / 2
        roots.append(dict(cluster_id=row["cluster_id"], opponent=row["opponent"],
                          delta=wins[candidate] - wins["control"]))
    if (len(seen) != gameplay["games"] or len({row["cluster_id"] for row in roots}) != gameplay["roots"]
            or len({row["state_sha256"] for row in bank["rows"]}) != gameplay["roots"]):
        raise ValueError("independent roots or paired games differ")
    recipe = gameplay["evaluation_recipe"]
    stats = retained.bootstrap({candidate: [row["delta"] for row in roots]},
                               recipe["bootstrap_seed"], recipe["bootstrap_repetitions"])[candidate]
    by_opponent = {name: sum(row["delta"] for row in roots if row["opponent"] == name) /
                   sum(row["opponent"] == name for row in roots) for name in sorted({row["opponent"] for row in roots})}
    if gameplay["stage"] == "pilot":
        qualified = stats["upper95"] >= 0
    elif gameplay["stage"] == "development":
        qualified = stats["mean"] >= .03 and min(by_opponent.values()) >= -.05
    else:
        raise ValueError("undeclared promotion stage")
    out = Path(plan["output"])
    archive = campaign.immutable(out / "ARCHIVE.json", dict(passed=True,
        gameplay_plan=plan["gameplay_plan"], result=plan["result"], bank=gameplay["bank"],
        runtime=gameplay["runtime"], root_receipts=result["root_receipts"], validated_games=len(seen),
        validated_decisions=decisions, zero_operational_failures=True,
        all_source_roster_runtime_bindings_verified=True, protocol_replay_verified=True,
        control_reused_once_per_root_color=True, all_declared_games_archived=True))
    value = dict(schema=SCHEMA, passed=True, stage=gameplay["stage"], source=plan["source"],
        plan=campaign.record(plan_path), archive=archive, statistics=stats,
        per_opponent_mean=by_opponent, qualified_for_next_stage=qualified,
        playing_strength_promoted=False, experimental_live_source=False,
        favorable_subsets_used=False, individual_game_strategy_mining=False, new_games=0)
    campaign.immutable(out / "RESULT.json", value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.plan)
    print(json.dumps({key: result[key] for key in ("passed", "stage", "qualified_for_next_stage")}))


if __name__ == "__main__":
    main()
