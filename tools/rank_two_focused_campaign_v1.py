"""Bounded, cumulative network research on the existing campaign queue."""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "papersoccer.focused-network-campaign.v1"
BASELINE = "78e731a81ce7cc702a200d0832a6c75e967a11b12452051724ab1387c226e4b8"
FAMILIES = {"dd8": [6301, 8, 8, 1], "dd12wide": [6301, 12, 16, 1]}
SEEDS = [2026100701, 2026100702]
RECIPE = dict(float_learning_rate=1e-3, qat_learning_rate=2.5e-4,
              weight_decay=1e-5, gradient_clip=5., huber_delta=.25,
              ranking_weight=.25, float_min_epochs=10, float_max_epochs=40,
              qat_min_epochs=8, qat_max_epochs=16, patience=5,
              improvement_tolerance=1e-4, weight_bits=4, outcome_weight=0.,
              calibration_percentiles=[95, 99, 100], calibration_passes=2,
              calibration_split="train", fixed_scales_during_qat=True)
PLAN = dict(schema=SCHEMA + ".plan", duration_days=14, preparation_days=4,
            selection_days=3, families=FAMILIES, seeds=SEEDS, recipe=RECIPE,
            generation_seed=2026100710, selection_seed=2026100711,
            bootstrap_seed=2026100712, bootstrap_repetitions=10000,
            train_roots=128, validation_roots=128, trajectories=512,
            maximum_parents=6144, teacher_nodes=64000,
            minimum_early_validation_roots=100, selection_roots=256,
            selection_games=2560, comparison_clocks_ms=[550, 140],
            source_limit=100000, stress_source_limit=98000,
            nonpayload_target=43000, maximum_rounds=4, maximum_arms=2,
            maximum_training_roots=2048, maximum_live_sources=6,
            maximum_workers=10, maximum_rss_bytes=18*1024**3,
            numerical_threads=1, recovery_minutes=5,
            weekly_used_exclusive_maximum=70,
            checkpoint_selection=["native_regret", "teacher_huber", "epoch"],
            family_selection="dd12wide only when paired lower95 > 0; otherwise safe dd8",
            promotion="existing-pilot-development-private-formal-confirmation-gates")
RESEARCH = {"freeze", "enqueue", "train", "generate", "label", "experiment",
            "prepare-bank", "admit", "claim", "submit"}
TERMINAL = {"observe", "archive", "classify", "restore", "checkpoint",
            "qualification-check"}


def now():
    return dt.datetime.now(dt.timezone.utc)


def read(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest()}


def verify(reference):
    if record(reference["path"]) != reference:
        raise ValueError("changed bound input: " + reference["path"])
    return Path(reference["path"])


def immutable(path, value):
    path = Path(path)
    payload = value if isinstance(value, bytes) else canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError("immutable receipt already differs: " + str(path))
    else:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            import os
            os.fsync(stream.fileno())
    return record(path)


def atomic(path, value):
    import os
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    with temporary.open("xb") as stream:
        stream.write(canonical(value)); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextlib.contextmanager
def locked(directory):
    with (Path(directory) / "ownership.lock").open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def validate(state, closures, owner, action, used_percent, source=None, attempt=None, at=None):
    if state.get("owner_thread_id") != owner or state.get("ownership_status") != "active":
        raise PermissionError("active execution owner differs")
    if state.get("worktree") != str(ROOT) or state.get("recovery_minutes") != 5:
        raise ValueError("worktree/recovery binding differs")
    if not state.get("focused_activation") or action not in RESEARCH | TERMINAL:
        raise PermissionError("focused campaign inactive or unknown action")
    if action != "checkpoint":
        if (type(used_percent) not in (int, float) or not math.isfinite(used_percent)
                or not 0 <= used_percent <= 100):
            raise PermissionError("weekly usage unknown; defer to next wake")
        if used_percent >= 70:
            raise PermissionError("weekly reserve reached; checkpoint and pause recovery")
    at = at or now()
    if at.tzinfo is None:
        raise ValueError("timezone-aware timestamp required")
    if len(state.get("focused_live_sources", {})) > 6:
        raise ValueError("live source ceiling exceeded")
    if action in RESEARCH:
        if state.get("human_pause", {}).get("status") == "paused":
            raise PermissionError("human resumption required")
        if state.get("research_closed") or at >= dt.datetime.fromisoformat(state["deadline_utc"]):
            raise PermissionError("research deadline reached; terminal work only")
        if not state.get("focused_setup_ready") and at >= dt.datetime.fromisoformat(state["preparation_deadline_utc"]):
            raise PermissionError("preparation timebox reached; retain work and report blocker")
        if state.get("focused_setup_ready") and not state.get("focused_family") and at >= dt.datetime.fromisoformat(state["selection_deadline_utc"]):
            raise PermissionError("matched study timebox reached; retain incomplete evidence")
        restoration = source == BASELINE and state.get("restoration_authorization") and action in {"claim", "submit"}
        if state.get("incident") and not restoration:
            raise PermissionError("incident requires restoration and classification")
        if action in {"train", "generate", "label", "prepare-bank", "admit"} and not state.get("focused_setup_ready"):
            raise PermissionError("common runtime/training preparation is not admitted")
        if action == "prepare-bank" and state.get("pending_exposures"):
            raise PermissionError("pending exposure must be incrementally indexed first")
    banned = {row["source_sha256"] for key in ("operationally_failed", "retired_implementations")
              for row in closures.get(key, [])}
    if source is not None and action in {"admit", "claim", "submit"}:
        if source in banned:
            raise PermissionError("exact failed/retired source remains banned")
        ref = state.get("frozen_sources", {}).get(source)
        if source == BASELINE:
            ref = state["incumbent"]
        if ref is None:
            raise PermissionError("source is not frozen in the current campaign")
        if "source" in ref:
            ref = ref["source"]
        if len(verify(ref).read_bytes().decode("utf-8")) >= 100000:
            raise ValueError("submission source limit")
        if action == "admit" and source != BASELINE and source not in state.get("focused_live_sources", {}) and len(state.get("focused_live_sources", {})) >= 6:
            raise PermissionError("six live versions already admitted")
    if action == "claim" and state.get("active_claims"):
        raise PermissionError("unresolved claim retained; never duplicate a window")
    if action == "submit":
        claim = state.get("active_claims", {}).get(attempt)
        if not claim or claim.get("source_sha256") != source:
            raise PermissionError("submission requires its exact bound claim")
    return True


def activate(base, audit_path, approval_path, usage_path, model_path, owner, at=None):
    base = Path(base).resolve()
    with locked(base):
        state = read(base / "CURRENT.json")
        if state.get("focused_activation"):
            return read(verify(state["focused_activation"]))
        if state.get("owner_thread_id") != owner or state.get("ownership_status") != "active":
            raise PermissionError("predecessor owner differs")
        if state.get("worktree") != str(ROOT):
            raise ValueError("existing checkout required")
        if not state.get("delivery_complete") or not state.get("research_closed"):
            raise ValueError("predecessor is not audited and closed")
        if any(state.get(k) for k in ("active_jobs", "active_claims", "active_helper", "incident", "arena_attempt")):
            raise ValueError("unresolved predecessor work")
        if any((base.parent / n).exists() for n in ("STOP.json", "RESTORE_REQUIRED.json")):
            raise ValueError("unresolved predecessor restoration")
        approval, audit = read(approval_path), read(audit_path)
        usage, model = read(usage_path), read(model_path)
        if approval.get("approved_by") != "human" or approval.get("owner_thread_id") != owner or approval.get("plan") != PLAN:
            raise ValueError("approved successor plan differs")
        verify(approval["contract"])
        if not audit.get("passed") or not audit.get("all90_archived") or not audit.get("no_pending_execution"):
            raise ValueError("predecessor audit incomplete")
        for key in ("source_sha256", "editor_sha256", "deployed_sha256"):
            if audit.get(key) != BASELINE:
                raise ValueError("source/editor/deployed identity differs")
        if audit["predecessor_current"] != record(base / "CURRENT.json"):
            raise ValueError("predecessor changed after fresh audit")
        active = read(verify(audit["active"]))
        if active["agent_id"] != audit["agent_id"] or active["submission_id"] != audit["submission_id"] or active["source_sha256"] != BASELINE:
            raise ValueError("fresh deployed identity differs")
        expected_profile = approval.get("execution_profile")
        if (not isinstance(expected_profile, dict)
                or set(expected_profile) != {"model", "effort", "service_tier"}
                or not all(isinstance(expected_profile[key], str) and expected_profile[key]
                           for key in ("model", "effort"))
                or any(model.get(key) != value for key, value in expected_profile.items())):
            raise ValueError("actual execution profile differs from human approval")
        if usage.get("window_duration_mins") != 10080 or usage.get("ordinary_usage_allowed") is not True or not 0 <= usage["used_percent"] < 70:
            raise PermissionError("fresh weekly usage does not permit activation")
        at = at or now()
        if at.tzinfo is None:
            raise ValueError("timezone-aware activation timestamp required")
        age = (at - dt.datetime.fromisoformat(usage["observed_at"])).total_seconds()
        if not 0 <= age <= 120:
            raise PermissionError("activation usage observation is stale")
        out = base / "focused-network-v1"
        predecessor = immutable(out / "PREDECESSOR_FINAL_CURRENT.json", (base / "CURRENT.json").read_bytes())
        activation = dict(schema=SCHEMA + ".activation", owner_thread_id=owner,
                          predecessor_current=predecessor, predecessor_audit=record(audit_path),
                          approval=record(approval_path), usage=record(usage_path), model=record(model_path),
                          started_at_utc=at.isoformat(), deadline_utc=(at + dt.timedelta(days=14)).isoformat(),
                          preparation_deadline_utc=(at + dt.timedelta(days=4)).isoformat())
        activation_ref = immutable(out / "ACTIVATION.json", activation)
        state.update(focused_activation=activation_ref, campaign_contract=approval["contract"],
                     focused_plan=record(approval_path), control_guard=record(__file__),
                     predecessor_final_current=predecessor, phase="focused-runtime-preparation",
                     started_at_utc=activation["started_at_utc"], deadline_utc=activation["deadline_utc"],
                     preparation_deadline_utc=activation["preparation_deadline_utc"],
                     candidate_count=0, completed_candidates=0, frozen_sources={}, focused_live_sources={},
                     focused_rounds=[], focused_family=None, focused_setup_ready=None,
                     active_jobs=[], active_claims={}, active_helper=None, incident=None,
                     active_packet=None, arena_attempt=None, restoration_authorization=None,
                     research_closed=False, delivery_complete=False, recovery_retired=False,
                     next_wave_status="closed-predecessor", recovery_minutes=5,
                     primary_model=record(model_path), weekly_used_exclusive_maximum=70,
                     next_action="Implement compact common runtime and matched training preflights; no production training or fresh gameplay bank before focused_setup_ready.")
        atomic(base / "CURRENT.json", state)
        return activation


def architecture_decision(scores, feasible, repetitions=10000, seed=2026100712):
    """Bootstrap roots, keeping both seeds and both colors paired."""
    if set(feasible) != set(FAMILIES):
        raise ValueError("both family feasibility decisions required")
    if not any(feasible.values()):
        return dict(family=None, reason="neither-family-mechanically-admitted", strength_claim=False)
    if not all(feasible.values()):
        return dict(family=next(k for k in FAMILIES if feasible[k]), reason="only-feasible-family", strength_claim=False)
    expected = {f"{family}-{seed_value}" for family in FAMILIES for seed_value in SEEDS}
    if set(scores) != expected:
        raise ValueError("all four frozen seed checkpoints required")
    sizes = {len(rows) for rows in scores.values()}
    if len(sizes) != 1 or not next(iter(sizes)):
        raise ValueError("paired root coverage differs")
    for rows in scores.values():
        if any(len(row) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in row) for row in rows):
            raise ValueError("two color scores in [0,1] required")
    differences = []
    seed_means = {}
    for seed_value in SEEDS:
        small, large = scores[f"dd8-{seed_value}"], scores[f"dd12wide-{seed_value}"]
        delta = [sum(b)/2 - sum(a)/2 for a, b in zip(small, large, strict=True)]
        seed_means[str(seed_value)] = sum(delta)/len(delta)
        differences.append(delta)
    roots = [(a+b)/2 for a, b in zip(*differences, strict=True)]
    rng = random.Random(seed)
    boot = sorted(sum(roots[rng.randrange(len(roots))] for _ in roots)/len(roots) for _ in range(repetitions))
    lower, upper = boot[int(.025*(repetitions-1))], boot[int(.975*(repetitions-1))]
    return dict(family="dd12wide" if lower > 0 else "dd8", reason="paired-root-study",
                mean_difference=sum(roots)/len(roots), lower95=lower, upper95=upper,
                seed_differences=seed_means, roots=len(roots), repetitions=repetitions,
                strength_claim="conditional on two frozen seeds and this shared panel")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    sub = parser.add_subparsers(dest="action", required=True)
    activate_parser = sub.add_parser("activate")
    for name in ("audit", "approval", "usage", "model"):
        activate_parser.add_argument("--" + name, type=Path, required=True)
    activate_parser.add_argument("--owner", required=True)
    sub.add_parser("status")
    args = parser.parse_args()
    if args.action == "activate":
        value = activate(args.base, args.audit, args.approval, args.usage, args.model, args.owner)
    else:
        state = read(args.base / "CURRENT.json")
        value = {key: state.get(key) for key in ("phase", "focused_family", "active_jobs", "active_claims", "deadline_utc", "next_action")}
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
