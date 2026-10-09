"""Receipt-bound setup, experiment and restoration guards for focused research."""
from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

from tools import rank_two_focused_campaign_v1 as retained

ROOT, BASELINE, FAMILIES = retained.ROOT, retained.BASELINE, retained.FAMILIES
SCHEMA = retained.SCHEMA + ".controller.v2"
read, verify, record = retained.read, retained.verify, retained.record
immutable, atomic, locked, now = retained.immutable, retained.atomic, retained.locked, retained.now


def actual_usage(document, at=None, fresh=True):
    at = at or now()
    raw = document.get("actual_response", {})
    buckets = raw.get("rateLimitsByLimitId") or {"default": raw.get("rateLimits")}
    reported = []
    for bucket in buckets.values():
        if not isinstance(bucket, dict):
            continue
        for name in ("primary", "secondary"):
            window = bucket.get(name)
            if isinstance(window, dict) and window.get("windowDurationMins") == 10080:
                value = window.get("usedPercent")
                if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 100:
                    raise PermissionError("actual weekly usage unknown; defer")
                reported.append(value)
    if not reported or raw.get("ordinaryUsageAllowed") is not True:
        raise PermissionError("actual weekly usage unavailable or ordinary usage prohibited")
    used = max(reported)
    if document.get("window_duration_mins") != 10080 or document.get("used_percent") != used:
        raise ValueError("weekly usage receipt does not match actual reported windows")
    observed = dt.datetime.fromisoformat(document["observed_at"].replace("Z", "+00:00"))
    age = (at - observed).total_seconds()
    if observed.tzinfo is None or age < 0 or (fresh and age > 120):
        raise PermissionError("fresh actual usage receipt required before launch or admission")
    if used >= 70:
        raise PermissionError("weekly reserve reached; checkpoint and pause recovery")
    return used


def resource_guard(budget):
    if (type(budget.get("workers")) is not int or not 1 <= budget["workers"] <= 10
            or type(budget.get("rss_bytes")) is not int or not 0 < budget["rss_bytes"] <= 18 * 1024**3
            or budget.get("numerical_threads") != 1
            or type(budget.get("wall_seconds")) not in (int, float) or not math.isfinite(budget["wall_seconds"]) or budget["wall_seconds"] <= 0
            or type(budget.get("cpu_capacity_seconds")) not in (int, float) or not math.isfinite(budget["cpu_capacity_seconds"]) or budget["cpu_capacity_seconds"] <= 0):
        raise ValueError("declared fixed worker/memory/thread/time budget differs")
    return True


def setup_document(state):
    ref = state.get("focused_setup_ready")
    if not ref:
        raise PermissionError("focused setup has not been admitted")
    setup = read(verify(ref))
    if (setup.get("passed") is not True or setup.get("activation") != state["focused_activation"]
            or setup.get("owner_thread_id") != state["owner_thread_id"]):
        raise ValueError("focused setup admission binding differs")
    for reference in setup["inputs"].values():
        verify(reference)
    start = dt.datetime.fromisoformat(setup["selection_started_at_utc"])
    activation = read(verify(state["focused_activation"]))
    latest = dt.datetime.fromisoformat(activation["started_at_utc"]) + dt.timedelta(days=7)
    deadline = min(start + dt.timedelta(days=3), latest)
    if state.get("selection_deadline_utc") != deadline.isoformat():
        raise ValueError("matched study clock changed")
    return setup


def restoration_binding(state, source, attempt=None):
    if source != BASELINE or not state.get("incident") or not state.get("restoration_authorization"):
        return None
    incident = state["incident"]
    if not isinstance(incident, dict) or "receipt" not in incident:
        raise ValueError("restoration requires a bound focused incident")
    if state["incumbent"]["sha256"] != BASELINE:
        raise PermissionError("mandatory restoration incumbent identity differs")
    verify(incident["receipt"])
    if not isinstance(state["restoration_authorization"], dict) or set(state["restoration_authorization"]) != {"path", "sha256"}:
        raise PermissionError("restoration authorization must be an immutable bound receipt")
    authorization = read(verify(state["restoration_authorization"]))
    if (authorization.get("owner_thread_id") != state["owner_thread_id"]
            or authorization.get("activation") != state["focused_activation"]
            or authorization.get("incident") != incident["receipt"]
            or authorization.get("incumbent") != state["incumbent"]
            or authorization.get("source_sha256") != BASELINE
            or authorization.get("purpose") != "mandatory-restoration"
            or authorization.get("one_window_only") is not True
            or authorization.get("attempt_id") != attempt):
        raise PermissionError("restoration authorization is not bound to this incident and attempt")
    previous = state.get("focused_restorations", {}).get(incident["receipt"]["sha256"], {})
    if previous.get("finished") or previous.get("attested_window_id"):
        raise PermissionError("one restoration window already attested; finish its archive")
    return authorization


def source_admission(state, source):
    frozen = state.get("frozen_sources", {}).get(source)
    if not isinstance(frozen, dict) or "admission" not in frozen:
        raise PermissionError("exact source lacks focused mechanical admission")
    source_ref = frozen["source"]
    if source_ref["sha256"] != source:
        raise ValueError("frozen source key differs from its bytes")
    verify(source_ref)
    admission = read(verify(frozen["admission"]))
    if (admission.get("passed") is not True or admission.get("source") != source_ref
            or admission.get("runtime") != frozen.get("runtime") or admission.get("own_failures") != 0
            or len(set(admission.get("canonical_state_sha256", []))) < 2
            or admission.get("activation") != state["focused_activation"]):
        raise PermissionError("source-bound safety/canonical-state admission incomplete")
    for name in ("correctness", "sanitizers", "decoder", "native_parity", "fixed_work", "whole_response"):
        proof = read(verify(admission[name]))
        if proof.get("passed") is not True or proof.get("source") != source_ref:
            raise PermissionError("exact source safety proof incomplete: " + name)
        if name == "whole_response" and (proof.get("clocks_ms") != admission.get("clocks_ms")
                                         or proof.get("own_failures") != 0):
            raise PermissionError("whole-response proof does not match admitted clocks")
    clocks = admission["clocks_ms"]
    if len(clocks) != 2 or any(type(value) is not int for value in clocks) or not 0 < clocks[0] <= 990 or not 0 < clocks[1] <= 180:
        raise ValueError("whole-response clock ceiling exceeded")
    if clocks != [550, 140]:
        experiment = read(verify(admission["timing_experiment"]))
        if experiment.get("passed") is not True or experiment.get("clocks_ms") != clocks or experiment.get("source") != source_ref:
            raise PermissionError("larger clocks lack a separately declared exact-source experiment")
    semantic = frozen.get("semantic_identity")
    if not isinstance(semantic, str) or semantic in state.get("focused_banned_semantic_identities", []):
        raise PermissionError("semantic identity missing or operationally banned")
    return admission


def validate(state, closures, owner, action, used_percent, source=None, attempt=None, at=None):
    at = at or now()
    if source == BASELINE and state.get("incumbent", {}).get("sha256") != BASELINE:
        raise PermissionError("exact incumbent source identity differs")
    authorization = restoration_binding(state, source, attempt) if action in ("claim", "submit", "restore") else None
    if authorization:
        # Mandatory restoration remains terminal work after either timebox.
        retained.validate(state, closures, owner, "restore", used_percent, at=at)
        verify(state["incumbent"])
        if action == "claim" and state.get("active_claims"):
            raise PermissionError("unknown prior claim retained before restoration")
        if action == "submit":
            claim = state.get("active_claims", {}).get(attempt)
            if not claim or claim.get("source_sha256") != BASELINE or claim.get("restoration_authorization") != state["restoration_authorization"]:
                raise PermissionError("restoration submission requires its exact incident-bound claim")
        return True
    if action == "restore":
        raise PermissionError("restoration needs its bound incident authorization")
    retained.validate(state, closures, owner, action, used_percent, source, attempt, at)
    if action in retained.RESEARCH and state.get("focused_setup_ready"):
        setup_document(state)
    if action in ("admit", "claim", "submit") and source != BASELINE:
        source_admission(state, source)
    if action == "claim":
        declaration = state.get("focused_declared_windows", {}).get(attempt)
        if not declaration or declaration.get("source_sha256") != source:
            raise PermissionError("claim lacks the exact predeclared slot/source")
        verify(declaration["plan"])
        if declaration.get("attested_window_id") or attempt in state.get("focused_completed_attempts", []):
            raise PermissionError("attested/completed window cannot be duplicated")
        if any(not window.get("all90_archived") for window in state.get("focused_known_windows", {}).values()):
            raise PermissionError("archive every known window before the next declared slot")
    return True


def guard(base, action, owner, usage_path, source=None, attempt=None, budget=None, at=None):
    base = Path(base)
    at = at or now()
    usage = read(usage_path)
    used = actual_usage(usage, at)
    if budget is not None:
        resource_guard(budget)
    with locked(base):
        state = read(base / "CURRENT.json")
        if state["control_guard"] != record(__file__):
            raise PermissionError("this successor controller is not currently bound")
        for key in ("focused_plan", "campaign_contract", "focused_activation", "control_guard", "closures"):
            verify(state[key])
        validate(state, read(verify(state["closures"])), owner, action, used, source, attempt, at)
        state["focused_usage_observation"] = record(usage_path)
        atomic(base / "CURRENT.json", state)
    return dict(passed=True, owner_thread_id=owner, action=action, source_sha256=source,
                usage=record(usage_path), used_percent=used, checked_at=at.isoformat(),
                activation=state["focused_activation"], controller=record(__file__), budget=budget)


def admit_setup(base, owner, usage_path, inputs, at=None):
    base = Path(base)
    at = at or now()
    used = actual_usage(read(usage_path), at)
    required = {"native", "independent", "production_safety", "pipeline_checks", "ancestry", "controller_checks"}
    if set(inputs) != required:
        raise ValueError("all six common preparation receipts required")
    documents = {key: read(verify(reference)) for key, reference in inputs.items()}
    if any(doc.get("passed") is not True for doc in documents.values()):
        raise PermissionError("preparation receipt failed or incomplete")
    if documents["controller_checks"].get("controller") != record(__file__):
        raise PermissionError("controller unit receipt belongs to another implementation")
    native = documents["native"]
    if not all(native.get(name) is True for name in ("compiler_tokens_equal", "fixed_work_incumbent_equal", "native_tensors_equal")):
        raise PermissionError("native compiler/tensor/search parity is incomplete")
    for family in FAMILIES:
        profile = native["profiles"][family]
        if profile["characters"] >= 98000 or profile["nonpayload_characters"] > 43000:
            raise PermissionError("common high-entropy size admission failed")
    if documents["production_safety"].get("own_failures") != 0 or documents["production_safety"].get("whole_response_envelopes_ms") != [550, 140]:
        raise PermissionError("common isolated whole-response admission incomplete")
    production_build = read(verify(documents["production_safety"]["exact_builds"]))
    for family in FAMILIES:
        expected = record(Path(inputs["native"]["path"]).parent / family / "submission.cpp")
        if production_build["builds"][family]["source"] != expected:
            raise PermissionError("timing source differs from the compact native fixture")
    independent = documents["independent"]
    if independent.get("both_orientations") is not True or any(
            independent["profiles"][family].get("sanitizers") is not True
            or independent["profiles"][family].get("maximum_inference_error") != 0 for family in FAMILIES):
        raise PermissionError("independent orientation/feature/inference/sanitizer admission incomplete")
    if documents["ancestry"].get("historical_protected_reads") is not False or documents["ancestry"].get("historical_ancestry_rescan") is not False:
        raise PermissionError("incremental ancestry receipt does not preserve the historical boundary")
    if any(documents["pipeline_checks"].get(key, 1) != 0 for key in ("errors", "failures")):
        raise PermissionError("pipeline unit checks contain errors")
    with locked(base):
        state = read(base / "CURRENT.json")
        validate(state, read(verify(state["closures"])), owner, "freeze", used, at=at)
        if state["control_guard"] != record(__file__):
            raise PermissionError("setup requires the currently bound successor controller")
        if state.get("focused_setup_ready"):
            return setup_document(state)
        if any(state.get(key) for key in ("active_jobs", "active_claims", "active_helper", "incident", "arena_attempt", "pending_exposures")):
            raise PermissionError("setup still has unresolved work, claims or exposure")
        if at >= dt.datetime.fromisoformat(state["preparation_deadline_utc"]):
            raise PermissionError("preparation timebox expired; retain work and report blocker")
        activation = read(verify(state["focused_activation"]))
        latest = dt.datetime.fromisoformat(activation["started_at_utc"]) + dt.timedelta(days=7)
        deadline = min(at + dt.timedelta(days=3), latest)
        setup = dict(schema=SCHEMA + ".setup-admission", passed=True, owner_thread_id=owner,
                     activation=state["focused_activation"], inputs=inputs, usage=record(usage_path),
                     controller=record(__file__), selection_started_at_utc=at.isoformat(),
                     selection_deadline_utc=deadline.isoformat(), campaign_deadline_unchanged=state["deadline_utc"])
        reference = immutable(base / "focused-network-v1/SETUP_READY.json", setup)
        state.update(focused_setup_ready=reference, focused_usage_observation=record(usage_path),
                     selection_started_at_utc=at.isoformat(), selection_deadline_utc=deadline.isoformat(),
                     phase="focused-matched-study", next_action="Freeze the common seed-bound training-data generation; keep both widths paired.")
        atomic(base / "CURRENT.json", state)
    return setup
