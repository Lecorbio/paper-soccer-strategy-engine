"""Receipt-bound live exploration on the original focused campaign clock."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import random

from tools import rank_two_focused_campaign_v2 as old

ROOT, BASELINE = old.ROOT, old.BASELINE
read, verify, record = old.read, old.verify, old.record
immutable, atomic, locked, now = old.immutable, old.atomic, old.locked, old.now
actual_usage, resource_guard = old.actual_usage, old.resource_guard
SCHEMA = "papersoccer.live-eight.v1"
FEATURE_SCHEMA = "papersoccer.jacek-replay-bfm.features.v1:edge316+vertex105x57:mover-relative-rotate180:true-turn-distance+free-degree"
ACTIVATIONS = ["square-leaky-0.01", "leaky-relu-0.01", "fast-tanh-rational-v1"]
RESEARCH = {"freeze", "enqueue", "train", "generate", "label", "experiment", "prepare-bank", "admit"}
WINDOW_ACTIONS = {"claim", "submit"}
TERMINAL = {"observe", "archive", "classify", "restore", "checkpoint", "qualification-check"}
PLAN = dict(architecture=[6301, 8, 8, 1], maximum_live_versions=12,
            reserved_late_versions=2, late_release_hours=96, research_reserve_hours=48,
            source_target=99000, source_hard_limit=100000, weight_bits=[4, 6, 7],
            clock_ladder=[[550, 140], [750, 160], [900, 170], [990, 180]],
            exploration_order=list("BCCB"), coverage_extension=list("BC"),
            qualification_order=list("BCCBBCCB"), primary_opponents=6,
            minimum_clean_per_opponent_arm=8, minimum_live_uplift=.03,
            opponent_floor=-.05, bootstrap_repetitions=10000, bootstrap_seed=2026101023,
            formal_windows=3, minimum_top_two_windows=2, private_games=12,
            local_confirmation_roots=800, local_confirmation_games=3200,
            local_strength_veto=False, old_strength_stop_diagnostic_override=True,
            live_train_mass=.5, training_roots_maximum=2048, validation_roots=128,
            minimum_early_validation_roots=100, teacher_nodes=256000,
            deeper_teacher_nodes_maximum=1000000, float_minimum=10, float_maximum=80,
            qat_minimum=8, qat_maximum=32, patience=5, tolerance=1e-4,
            float_learning_rate=.001, qat_learning_rate=.00025,
            weight_decay=1e-5, gradient_clip=5., ranking_weight=.25,
            numerical_threads=1, maximum_workers=10, maximum_rss_bytes=18*1024**3,
            weekly_used_exclusive_maximum=70, recovery_minutes=5)


def activation(state):
    doc = read(verify(state["live_eight_activation"]))
    approval = read(verify(doc["approval"]))
    if (approval.get("approved_by") != "human" or approval.get("plan") != PLAN
            or doc["original_activation"] != state["focused_activation"]
            or doc["owner_thread_id"] != state["owner_thread_id"]):
        raise PermissionError("human-approved extension binding differs")
    original = read(verify(doc["original_activation"]))
    if (doc["deadline_utc"] != original["deadline_utc"]
            or state["deadline_utc"] != original["deadline_utc"]
            or state["started_at_utc"] != original["started_at_utc"]):
        raise PermissionError("original campaign clock cannot reset")
    expected = dt.datetime.fromisoformat(doc["deadline_utc"]) - dt.timedelta(hours=48)
    if doc["research_cutoff_utc"] != expected.isoformat():
        raise ValueError("final qualification reserve differs")
    verify(approval["contract"])
    return doc


def banned(state, closures, source):
    exact = {row["source_sha256"] for key in ("operationally_failed", "retired_implementations")
             for row in closures.get(key, [])}
    entry = state.get("frozen_sources", {}).get(source, {})
    if (source in exact or entry.get("operationally_banned")
            or entry.get("semantic_identity") in state.get("focused_banned_semantic_identities", [])):
        raise PermissionError("exact source or semantic operational ban remains permanent")


def source_admission(state, source):
    if source == BASELINE:
        verify(state["incumbent"])
        return
    entry = state.get("frozen_sources", {}).get(source)
    if not entry:
        raise PermissionError("exact source is not frozen")
    text = verify(entry["source"]).read_text()
    text.encode("ascii")
    if not 0 < len(text) <= PLAN["source_target"]:
        raise ValueError("source character target exceeded")
    admission = read(verify(entry["admission"]))
    if (admission.get("passed") is not True or admission.get("own_failures") != 0
            or admission.get("source") != entry["source"]
            or admission.get("runtime") != entry.get("runtime")
            or admission.get("activation") != state["focused_activation"]
            or len(set(admission.get("canonical_state_sha256", []))) < 2):
        raise PermissionError("source-bound safety admission incomplete")
    for name in ("correctness", "sanitizers", "decoder", "native_parity", "fixed_work", "whole_response"):
        proof = read(verify(admission[name]))
        if proof.get("passed") is not True or proof.get("source") != entry["source"]:
            raise PermissionError("source proof differs: " + name)
        if name == "whole_response" and (proof.get("clocks_ms") != admission["clocks_ms"]
                                         or proof.get("own_failures") != 0):
            raise PermissionError("whole-response clock proof differs")
    if admission["clocks_ms"] not in PLAN["clock_ladder"]:
        raise PermissionError("undeclared timing tuple")
    if admission["clocks_ms"] != [550, 140]:
        proof = read(verify(admission["timing_experiment"]))
        if not proof.get("passed") or proof.get("source") != entry["source"] or proof.get("clocks_ms") != admission["clocks_ms"]:
            raise PermissionError("higher clocks lack exact-source certification")
    model = read(verify(entry["runtime"]))
    if model.get("feature_schema") != FEATURE_SCHEMA or model.get("activations") != ACTIVATIONS:
        raise PermissionError("features or activations changed")
    if model.get("architecture") != PLAN["architecture"]:
        control = state.get("live_eight_retimed_control")
        if (source != control or model.get("architecture") != [6301, 12, 8, 1]
                or model.get("weight_bits") != 4
                or entry.get("incumbent_tensor_identity") != state.get("live_eight_incumbent_tensor_identity")):
            raise PermissionError("only the exact retimed incumbent may use another width")
    if model.get("biases") is not False or model.get("weight_bits") not in PLAN["weight_bits"]:
        raise PermissionError("network representation differs")


def validate(state, closures, owner, action, used_percent, source=None, attempt=None, at=None):
    at = at or now()
    if state.get("owner_thread_id") != owner or state.get("ownership_status") != "active" or state.get("worktree") != str(ROOT):
        raise PermissionError("active execution owner or worktree differs")
    if action not in RESEARCH | WINDOW_ACTIONS | TERMINAL:
        raise PermissionError("unknown extension action")
    if action != "checkpoint" and (type(used_percent) not in (int, float)
            or not math.isfinite(used_percent) or not 0 <= used_percent < 70):
        raise PermissionError("weekly usage unavailable or reserve reached")
    doc = activation(state)
    if state.get("human_pause", {}).get("status") == "paused" and action not in TERMINAL:
        raise PermissionError("explicit human resumption required")
    if action in RESEARCH:
        if state.get("research_closed") or at >= dt.datetime.fromisoformat(doc["research_cutoff_utc"]):
            raise PermissionError("research cutoff reached; qualification or terminal work only")
        if state.get("incident"):
            raise PermissionError("resolve and restore the incident before research")
        if action == "prepare-bank" and (state.get("pending_exposures") or not state.get("live_eight_ancestry_validated")):
            raise PermissionError("validate layered ancestry and close new exposures before fresh banks")
    if source is not None:
        banned(state, closures, source)
    if action in {"admit", "claim", "submit"}:
        if source is None:
            raise PermissionError("exact source required")
        source_admission(state, source)
    versions = state.get("live_eight_versions", {})
    if len(versions) > 12:
        raise PermissionError("live version ceiling exceeded")
    if action == "admit" and source != BASELINE and source not in versions:
        late = dt.datetime.fromisoformat(doc["deadline_utc"]) - dt.timedelta(hours=96)
        if len(versions) >= (12 if at >= late else 10):
            raise PermissionError("live version cap or reserved late slots reached")
    if action in WINDOW_ACTIONS:
        window = state.get("live_eight_declared_windows", {}).get(attempt)
        if not window or window["source_sha256"] != source:
            raise PermissionError("exact predeclared source/window required")
        verify(window["plan"])
        if window.get("attested") or attempt in state.get("live_eight_completed_attempts", []):
            raise PermissionError("attested window cannot be repeated")
        purpose = window["purpose"]
        if purpose not in ("exploration", "qualification", "private", "formal", "restoration"):
            raise PermissionError("unknown window purpose")
        if purpose in ("qualification", "private", "formal"):
            finalist = read(verify(state["live_eight_finalist"]))
            if finalist["source"]["sha256"] != source:
                raise PermissionError("qualification source differs from frozen finalist")
        if purpose in ("private", "formal"):
            assessed = read(verify(state["live_eight_final_live_assessment"]))
            if assessed.get("source_sha256") != source or assessed.get("live_gate_passed") is not True:
                raise PermissionError("independent live qualification incomplete")
        if purpose == "formal":
            panel = read(verify(state["live_eight_private_panel"]))
            if (panel.get("source_sha256") != source or panel.get("passed") is not True
                    or panel.get("games") != 12 or panel.get("own_failures") != 0):
                raise PermissionError("private qualification incomplete")
            if state.get("live_eight_formal_sources", {}).get(source) != window.get("block"):
                raise PermissionError("one bound formal block per source required")
        if purpose == "exploration" and at >= dt.datetime.fromisoformat(doc["research_cutoff_utc"]):
            raise PermissionError("exploration cutoff reached")
        if purpose != "restoration" and at >= dt.datetime.fromisoformat(doc["deadline_utc"]):
            raise PermissionError("campaign deadline reached")
        if state.get("incident") and (source != BASELINE or purpose != "restoration"):
            raise PermissionError("incident allows only incumbent restoration")
        if any(not value.get("all90_archived") for value in state.get("live_eight_known_windows", {}).values()):
            raise PermissionError("archive every known window before another upload")
        if action == "claim" and state.get("active_claims"):
            raise PermissionError("unknown prior claim retained")
        if action == "submit" and state.get("active_claims", {}).get(attempt, {}).get("source_sha256") != source:
            raise PermissionError("submit requires its durable exact-source claim")
    return True


def guard(base, action, owner, usage_path, source=None, attempt=None, budget=None, at=None):
    base = Path(base)
    used = actual_usage(read(usage_path), at)
    if budget is not None:
        resource_guard(budget)
    with locked(base):
        state = read(base / "CURRENT.json")
        if state["control_guard"] != record(__file__):
            raise PermissionError("current controller binding differs")
        for key in ("focused_activation", "live_eight_approval", "campaign_contract", "closures"):
            verify(state[key])
        validate(state, read(verify(state["closures"])), owner, action, used, source, attempt, at)
        state["focused_usage_observation"] = record(usage_path)
        atomic(base / "CURRENT.json", state)
    return dict(passed=True, action=action, owner_thread_id=owner, source_sha256=source,
                attempt_id=attempt, used_percent=used, usage=record(usage_path),
                checked_at=(at or now()).isoformat(), controller=record(__file__), budget=budget)


def activate(base, approval_path, usage_path, owner, at=None):
    base, at = Path(base), at or now()
    used = actual_usage(read(usage_path), at)
    approval = read(approval_path)
    if approval.get("approved_by") != "human" or approval.get("owner_thread_id") != owner or approval.get("plan") != PLAN:
        raise PermissionError("approved live-eight plan required")
    verify(approval["contract"])
    with locked(base):
        state = read(base / "CURRENT.json")
        if state.get("live_eight_activation"):
            return read(verify(state["live_eight_activation"]))
        if state.get("owner_thread_id") != owner or state.get("ownership_status") != "active" or state.get("worktree") != str(ROOT):
            raise PermissionError("predecessor ownership differs")
        if not state.get("delivery_complete") or not state.get("research_closed"):
            raise PermissionError("audited closed predecessor required")
        if any(state.get(k) for k in ("active_jobs", "active_claims", "active_helper", "incident", "pending_exposures", "arena_attempt")):
            raise PermissionError("unresolved predecessor work")
        original = read(verify(state["focused_activation"]))
        cutoff = dt.datetime.fromisoformat(original["deadline_utc"]) - dt.timedelta(hours=48)
        if at >= cutoff:
            raise PermissionError("no remaining research window")
        out = base / "live-eight-v1"
        predecessor = immutable(out / "PREDECESSOR_CURRENT.json", (base / "CURRENT.json").read_bytes())
        doc = dict(schema=SCHEMA + ".activation", owner_thread_id=owner,
                   original_activation=state["focused_activation"], approval=record(approval_path),
                   predecessor_current=predecessor, deadline_utc=original["deadline_utc"],
                   research_cutoff_utc=cutoff.isoformat(), extension_started_at_utc=at.isoformat(),
                   used_percent=used, usage=record(usage_path))
        ref = immutable(out / "ACTIVATION.json", doc)
        state.update(live_eight_activation=ref, live_eight_approval=record(approval_path),
                     campaign_contract=approval["contract"], control_guard=record(__file__),
                     research_closed=False, delivery_complete=False, recovery_retired=False,
                     live_eight_versions={}, live_eight_declared_windows={}, live_eight_known_windows={},
                     live_eight_completed_attempts=[], live_eight_blocks=[], phase="live-eight-preparation",
                     focused_usage_observation=record(usage_path), recovery_minutes=5,
                     next_action="Validate extension, layered ancestry and current live identities; prepare first BCCB diagnostic.")
        atomic(base / "CURRENT.json", state)
    return doc


def select_primary(board, owner_id):
    rows = board.get("users", board.get("rows", [])) if isinstance(board, dict) else board
    rows = [row for row in rows if row.get("codingamer", {}).get("userId", row.get("userId")) != owner_id]
    def name(row):
        return row.get("codingamer", {}).get("pseudo", row.get("pseudo", ""))
    required = []
    for nickname in ("jacek", "marchete"):
        matches = [row for row in rows if name(row).casefold() == nickname]
        if len(matches) != 1:
            raise ValueError("primary opponent identity unavailable: " + nickname)
        required.extend(matches)
    ordered = sorted(rows, key=lambda r: r["rank"])
    selected = required + [row for row in ordered if row not in required][:4]
    if len(selected) != 6:
        raise ValueError("six actual primary opponents required")
    return [dict(name=name(row), user_id=row.get("codingamer", {}).get("userId", row.get("userId")),
                 agent_id=row["agentId"], rank=row["rank"]) for row in selected]


def score_windows(windows, primary, repetitions=10000, seed=2026101023):
    """Version/color matching with paired whole-window bootstrap units."""
    if len(primary) != 6 or len({p["user_id"] for p in primary}) != 6:
        raise ValueError("six frozen opponent identities required")
    if len(windows) not in (4, 6, 8, 10):
        raise ValueError("complete declared window block required")
    pairs, own_failures = [], 0
    seen = set()
    for w in windows:
        if not w.get("complete") or len(w["games"]) != 90:
            raise ValueError("all ninety games must be archived")
        for g in w["games"]:
            if g["game_id"] in seen:
                raise ValueError("duplicate attested game")
            seen.add(g["game_id"])
            own_failures += bool(g.get("own_failure"))
    for index in range(0, len(windows), 2):
        a, b = windows[index:index+2]
        if {a["arm"], b["arm"]} != {"B", "C"}:
            raise ValueError("paired window arm order differs")
        cell = {"B": {}, "C": {}}
        for w in (a, b):
            for g in w["games"]:
                if g.get("own_failure") or g.get("opponent_failure") or not g.get("clean"):
                    continue
                key = (g["opponent_user_id"], g["opponent_submission_id"], g["color"])
                cell[w["arm"]].setdefault(key, []).append(float(g["won"]))
        pairs.append(cell)
    ids = [p["user_id"] for p in primary]
    def calculate(chosen):
        combined = {"B": {}, "C": {}}
        for pair in chosen:
            # Unmatched version/color rows are descriptive, never silently pooled.
            for key in pair["B"].keys() & pair["C"].keys():
                for arm in ("B", "C"):
                    combined[arm].setdefault(key, []).extend(pair[arm][key])
        deltas, coverage = {}, {}
        for opponent in ids:
            colors, counts = [], {"B": 0, "C": 0}
            for color in (0, 1):
                keys = [k for k in combined["B"] if k[0] == opponent and k[2] == color]
                if not keys:
                    return None
                denominator = sum(len(combined["B"][k]) for k in keys)
                colors.append(sum(len(combined["B"][k])/denominator *
                    (sum(combined["C"][k])/len(combined["C"][k]) - sum(combined["B"][k])/len(combined["B"][k])) for k in keys))
                for arm in counts:
                    counts[arm] += sum(len(combined[arm][k]) for k in keys)
            deltas[str(opponent)] = sum(colors)/2
            coverage[str(opponent)] = counts
        return dict(mean=sum(deltas.values())/6, opponent_delta=deltas, coverage=coverage)
    point = calculate(pairs)
    sufficient = bool(point) and all(min(c.values()) >= 8 for c in point["coverage"].values())
    rng, draws = random.Random(seed), []
    for _ in range(repetitions):
        result = calculate([pairs[rng.randrange(len(pairs))] for _ in pairs])
        if result is not None:
            draws.append(result["mean"])
    draws.sort()
    interval = [draws[int(.025*(len(draws)-1))], draws[int(.975*(len(draws)-1))]] if draws else None
    qualified = bool(sufficient and not own_failures and interval and len(draws) == repetitions
                     and point["mean"] >= .03 and interval[0] > 0
                     and min(point["opponent_delta"].values()) >= -.05)
    return dict(schema=SCHEMA + ".live-assessment", point=point, sufficient=sufficient,
                own_failures=own_failures, interval95=interval, bootstrap_repetitions=repetitions,
                eligible_bootstrap_repetitions=len(draws), bootstrap_seed=seed,
                bootstrap_unit="paired complete calibration windows", live_gate_passed=qualified,
                qualified_for_deployment=False, correlation_warning="Conditional on observed windows and frozen opponent identities")


def training_allowed(window, game):
    return (window.get("purpose") == "exploration" and window.get("training_designated") is True
            and window.get("extension") == SCHEMA and window.get("complete") is True
            and game.get("clean") is True and not game.get("own_failure") and not game.get("opponent_failure"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--usage", type=Path, required=True)
    parser.add_argument("--action", choices=sorted(RESEARCH | WINDOW_ACTIONS | TERMINAL), default="observe")
    parser.add_argument("--source"); parser.add_argument("--attempt")
    args = parser.parse_args()
    print(json.dumps(guard(args.base, args.action, args.owner, args.usage, args.source, args.attempt)))


if __name__ == "__main__":
    main()
