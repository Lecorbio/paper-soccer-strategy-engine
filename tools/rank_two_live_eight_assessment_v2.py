"""Assess complete live blocks without conditioning on identifiable resamples."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from tools import rank_two_live_eight_v1 as retained
from tools import rank_two_focused_work_v1 as work

SCHEMA = retained.SCHEMA + ".live-assessment.v2"


def score_windows(windows, primary, repetitions=10000, seed=2026101023):
    result = retained.score_windows(windows, primary, repetitions, seed)
    eligible = result["eligible_bootstrap_repetitions"]
    usable = eligible == repetitions and result["interval95"] is not None
    result.update(schema=SCHEMA, confidence_interval_usable=usable,
                  unidentified_bootstrap_repetitions=repetitions-eligible)
    if not usable:
        result["conditional_interval_diagnostic"] = result["interval95"]
        result["interval95"] = None
        result["live_gate_passed"] = False
        result["inference_limitation"] = (
            "At least one whole-window draw lacks a primary opponent/color cell; "
            "dropping such draws cannot provide the declared confidence interval.")
    # Later games are descriptive but every own failure still blocks promotion.
    all_own = sum(bool(game.get("own_failure")) for window in windows
                  for game in window["games"] + window.get("supplemental_games", []))
    for window in windows:
        observed = sum(bool(game.get("own_failure")) for game in
                       window["games"] + window.get("supplemental_games", []))
        if "own_failures" in window and window["own_failures"] != observed:
            raise ValueError("window own-failure summary differs from retained games")
    result["own_failures"] = all_own
    if all_own:
        result["live_gate_passed"] = False
    return result


def run(plan_path):
    plan = retained.read(plan_path)
    role = plan["role"]
    if role not in ("exploration", "qualification"):
        raise ValueError("explicit assessment purpose required")
    action = "experiment" if role == "exploration" else "qualification-check"
    current = work.checked(plan, action, launch=True)
    if plan["producer"] != retained.record(__file__):
        raise ValueError("versioned assessment producer changed")
    if any(not item.get("all90_archived") for item in current["live_eight_known_windows"].values()):
        raise PermissionError("complete live archives before assessment")
    if plan["primary_roster"] != current["live_eight_primary_roster"]:
        raise ValueError("frozen primary roster differs")
    if (plan["bootstrap_repetitions"] != retained.PLAN["bootstrap_repetitions"] or
            plan["bootstrap_seed"] != retained.PLAN["bootstrap_seed"]):
        raise ValueError("frozen bootstrap recipe differs")
    packets = [retained.read(retained.verify(ref)) for ref in plan["packets"]]
    expected = [slot["attempt_id"] for packet in packets for slot in packet["slots"]]
    windows = [retained.read(retained.verify(ref)) for ref in plan["windows"]]
    actual = [Path(ref["path"]).parent.name for ref in plan["windows"]]
    if actual != expected or any(p["candidate_sha256"] != plan["source_sha256"] for p in packets):
        raise ValueError("all declared windows of one exact source must be assessed")
    expected_kind = "exploration" if role == "exploration" else "qualification"
    if not packets or packets[0]["kind"] != expected_kind or len(packets) > 2:
        raise ValueError("assessment purpose differs from its frozen primary block")
    if len(packets) == 2 and packets[1]["kind"] != "coverage":
        raise ValueError("only the single declared coverage extension may be appended")
    if any(window["purpose"] != role for window in windows):
        raise ValueError("exploration and qualification windows cannot be substituted")
    for window in windows:
        source = retained.BASELINE if window["arm"] == "B" else plan["source_sha256"]
        if window["source_sha256"] != source:
            raise ValueError("exact arm source differs")
    if role == "qualification":
        if len(windows) not in (8, 10) or any(w.get("training_designated") for w in windows):
            raise ValueError("independent qualification block required; all its games are excluded from TRAIN")
        finalist = retained.read(retained.verify(current["live_eight_finalist"]))
        if finalist["source"]["sha256"] != plan["source_sha256"]:
            raise ValueError("qualification source differs from the frozen finalist")
    primary = retained.read(retained.verify(plan["primary_roster"]))["opponents"]
    result = score_windows(windows, primary, plan["bootstrap_repetitions"], plan["bootstrap_seed"])
    result.update(role=role, source_sha256=plan["source_sha256"], packets=plan["packets"],
                  windows=plan["windows"], producer=plan["producer"],
                  all_declared_games=sum(len(w["games"]) for w in windows),
                  ranks=[w["completed_rank"] for w in windows],
                  scores=[w["score"] for w in windows],
                  qualified_for_deployment=False)
    result["numeric_live_gate_passed"] = result["live_gate_passed"]
    if role == "exploration":
        result["live_gate_passed"] = False
    output = Path(plan["output"])
    assessment = retained.immutable(output/"ASSESSMENT.json", result)
    retained.immutable(output/"RESULT.json", dict(passed=True, assessment=assessment,
                       complete_window_pairs=len(windows)//2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.plan)))
