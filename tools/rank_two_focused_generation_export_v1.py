"""Finish a spent generation from exact receipts, without launching games."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tools import rank_two_focused_campaign_v1 as campaign
from tools import rank_two_focused_data_v3 as data
from tools import rank_two_focused_work_v1 as work


def complete(plan_path):
    plan = campaign.read(plan_path)
    work.checked(plan, "archive", launch=True)
    if plan["producer"] != campaign.record(__file__):
        raise ValueError("completion exporter source changed")
    generation = campaign.read(campaign.verify(plan["generation_plan"]))
    campaign.verify(plan["original_envelopes"])
    output = Path(generation["output"])
    receipts, envelopes = [], []
    for row in generation["rows"]:
        for color in row["colors"]:
            directory = output / "games" / (row["root_group_id"] + "-p" + str(color))
            claim, envelope = campaign.read(directory / "CLAIM.json"), campaign.read(directory / "RESULT.json")
            if (claim["plan"] != plan["generation_plan"] or envelope["plan"] != plan["generation_plan"]
                    or claim["row"] != row or claim["color"] != color
                    or envelope["game"]["failure"] is not None
                    or envelope["candidate"] != generation["actors"]["rank_3"]
                    or envelope["opponent"] != generation["actors"][row["opponent"]]):
                raise ValueError("retained generation identity or complete game differs")
            data.games.training_trajectory(envelope)
            receipts.append(campaign.record(directory / "RESULT.json"))
            envelopes.append(envelope)
    if len(envelopes) != 512:
        raise ValueError("exact512 retained trajectories required")
    corrected = output / "envelopes-compact-v1.jsonl"
    campaign.immutable(corrected, b"".join((json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
                                           + "\n").encode() for row in envelopes))
    exported = data.games.export_training(corrected, output / "games.jsonl")
    data.data.load_games(output / "games.jsonl")
    result = dict(passed=True, plan=plan["generation_plan"], games=512,
                  trajectories=campaign.record(output / "games.jsonl"), export=exported,
                  generation_failures=0, source_model_failure=False, completion_producer=campaign.record(__file__),
                  completion_plan=campaign.record(plan_path), new_games=0, replayed_games=0,
                  retained_game_receipts=receipts, strength_eligible=False)
    campaign.immutable(output / "RESULT.json", result)
    campaign.immutable(output / "COMPLETED_EXPOSURE.json", dict(
        schema=data.SCHEMA + ".completed-training-trajectories", generation_plan=plan["generation_plan"],
        result=campaign.record(output / "RESULT.json"), input=campaign.record(corrected),
        game_receipts=receipts, training_eligible=False, before_next_fresh_bank=True, games_retained=512,
        original_malformed_bundle_retained=plan["original_envelopes"]))
    return dict(passed=True, games_retained=512, replayed_games=0, completion=campaign.record(output / "RESULT.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(complete(args.plan)))


if __name__ == "__main__":
    main()
