"""Add one admitted focused-round source to the reviewed cached runtime roster."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools import rank_two_focused_campaign_v1 as campaign
from tools import rank_two_focused_runtime_v6 as runtime
from tools import rank_two_focused_work_v1 as work

SCHEMA = campaign.SCHEMA + ".focused-round-runtime.v1"


def build(plan_path):
    plan = campaign.read(plan_path)
    current = work.checked(plan, "experiment", launch=True)
    if plan["producer"] != campaign.record(__file__) or plan["runtime_producer"] != campaign.record(runtime.__file__):
        raise ValueError("round runtime assembly/adapter binding differs")
    for reference in plan["inputs"].values():
        campaign.verify(reference)
    frozen = campaign.read(campaign.verify(plan["sources"]))
    if len(frozen["models"]) != 1 or frozen["models"][0]["profile"] != current["focused_family"]:
        raise ValueError("one selected-family round source required")
    model = frozen["models"][0]
    runtime.source_allowed(current, model["source"]["sha256"])
    parent = campaign.read(campaign.verify(plan["parent_bundle"]))
    if parent["producer"] != plan["runtime_producer"]:
        raise ValueError("reviewed parent runtime producer differs")
    actors, builds = {}, {}
    for name, actor in parent["actors"].items():
        if name.startswith("candidate:"):
            continue
        original = parent["builds"][name]
        runtime.verify_build_receipt(original, actor, parent["compiler"])
        actors[name] = actor
        builds[name] = original
    manifest = campaign.read(campaign.verify(plan["manifest"]))
    if actors["control"]["source"]["sha256"] != manifest["control"]["source"]["sha256"]:
        raise ValueError("reused incumbent control identity differs")
    for name, expected in manifest["opponents"].items():
        actor = actors["opponent:" + name]
        if (actor["source"]["sha256"] != expected["source"]["sha256"]
                or actor["family"] != expected["family"] or actor["clocks_ms"] != expected["clocks_ms"]):
            raise ValueError("approved opponent roster identity differs")
    tag = "candidate:" + model["profile"] + "-seed-" + str(model["seed"])
    actor = dict(source=runtime.e.raw_record(model["source"]["path"]), family="turn_action_v2",
                 clocks_ms=[550, 140], response_policy="strict-focused-response")
    aliases = campaign.read(campaign.verify(model["export"]))["aliases"]
    output = Path(plan["output"])
    if (output / "CLAIM.json").exists():
        raise RuntimeError("spent runtime assembly claim retained")
    campaign.immutable(output / "CLAIM.json", dict(plan=campaign.record(plan_path), source=model["source"]))
    actors[tag] = actor
    builds[tag] = runtime.build(actor, aliases, output / "workers" / tag.replace(":", "-"), parent["compiler"])
    bundle = dict(parent, sources=plan["sources"], manifest=plan["manifest"], actors=actors, builds=builds,
        plan=campaign.record(plan_path), parent_bundle=plan["parent_bundle"], assembly_producer=campaign.record(__file__),
        excluded_declared_models=[], single_selected_family_arm=True,
        reused_control_and_opponent_builds=True, new_games=0)
    campaign.immutable(output / "bundle.json", bundle)
    return dict(passed=True, bundle=campaign.record(output / "bundle.json"), actors=len(actors), candidate=tag,
                new_games=0, runtime_producer=plan["runtime_producer"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.plan)
    campaign.immutable(Path(campaign.read(args.plan)["output"]) / "RESULT.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
