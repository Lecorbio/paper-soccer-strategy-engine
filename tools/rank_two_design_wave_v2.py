"""Activate an approved design wave only after its predecessor is audited."""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
from pathlib import Path

from tools import rank_two_live_v3 as live

SCHEMA = "papersoccer.design-wave.activation.v2"


def guard_module(state):
    spec = importlib.util.spec_from_file_location("design_wave_control", live.verify(state["control_guard"]))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def readiness(state, approval, plan, audit, active_source, *, stop=False, restore=False):
    """Pure validation; no clock/count reset while earlier work is unresolved."""
    if state.get("design_wave_activation"):
        raise ValueError("design wave already activated")
    if approval.get("approved_by") != "human" or approval.get("owner_thread_id") != state["owner_thread_id"]:
        raise ValueError("human approval/owner mismatch")
    if plan.get("status") != "approved-predecessor-open" or plan.get("owner_thread_id") != state["owner_thread_id"]:
        raise ValueError("approved plan/owner mismatch")
    if plan.get("maximum_frozen_sources") != 6 or plan.get("duration_days") != 7:
        raise ValueError("approved wave bounds changed")
    if plan.get("clocks_ms") != [990, 180]:
        raise ValueError("approved response budgets changed")
    if state["active_claims"] or state["active_jobs"] or state.get("helper") or state.get("incident") or stop or restore:
        raise ValueError("predecessor work, helper, incident or restoration remains unresolved")
    required = ("passed", "declared_comparison_closed", "restoration_complete", "all_played_windows_archived")
    if any(audit.get(key) is not True for key in required):
        raise ValueError("incomplete predecessor audit")
    incumbent = state["incumbent"]["sha256"]
    if any(audit.get(key) != incumbent for key in ("active_source_sha256", "editor_source_sha256", "deployed_source_sha256")):
        raise ValueError("predecessor source/editor/deployed identity differs")
    if active_source != incumbent or not audit.get("comparison_result") or not audit.get("deployed_submission"):
        raise ValueError("incumbent active identity or comparison receipt missing")
    return incumbent


def activate(root, audit_path, used_percent, owner=None, at=None):
    if not owner:
        raise ValueError("explicit execution owner required")
    root = Path(root).resolve()
    base = root / "campaign-next-v1"
    state = live.read(base / "CURRENT.json")
    control = guard_module(state)
    with control.locked(base):
        state = live.read(base / "CURRENT.json")
        control.validate(state, live.read(live.verify(state["closures"])), owner,
                         "checkpoint", used_percent, at=at)
        approval = live.read(live.verify(state["approved_next_wave"]))
        plan = live.read(live.verify(state["next_wave_plan"]))
        live.refs(approval)
        live.refs(plan)
        if approval["plan"] != state["next_wave_plan"] or approval["contract"] != state["next_wave_contract"]:
            raise ValueError("current approved wave bindings differ")
        audit = live.read(audit_path)
        live.refs(audit)
        active = live.read(root / "active.json")
        readiness(state, approval, plan, audit, active["source_sha256"],
                  stop=(root / "STOP.json").exists(), restore=(root / "RESTORE_REQUIRED.json").exists())
        result = live.read(live.verify(audit["comparison_result"]))
        predecessor_plan = state["native025_extension_packet"]
        expected_result = Path(predecessor_plan["path"]).with_name("result.json")
        if Path(audit["comparison_result"]["path"]).resolve() != expected_result.resolve():
            raise ValueError("audit belongs to a different predecessor comparison")
        if result.get("plan") != predecessor_plan:
            raise ValueError("closed comparison/plan identity differs")
        window_refs = result.get("windows", result.get("retained", []))
        if not window_refs:
            raise ValueError("predecessor contains no archived played windows")
        for ref in window_refs:
            window = live.read(live.verify(ref))
            coverage = window.get("coverage", {})
            if (window.get("status") != "complete" or not coverage.get("full_window_accounted")
                    or coverage.get("accepted_games") != 90):
                raise ValueError("predecessor window archive incomplete")
        deployed = live.read(live.verify(audit["deployed_submission"]))
        if (deployed.get("source_sha256") != state["incumbent"]["sha256"]
                or not deployed.get("agent_id") or not deployed.get("submission_id")):
            raise ValueError("deployed incumbent submission lacks exact attestation")
        if deployed.get("agent_id") != active.get("agent_id"):
            raise ValueError("incumbent active agent differs from audited submission")
        if result.get("own_failures", 0) and not audit.get("failed_source_classification"):
            raise ValueError("failed comparison requires retained classification")
        now = at or dt.datetime.now(dt.timezone.utc)
        if now.tzinfo is None:
            raise ValueError("activation timestamp needs timezone")
        out = base / "design-wave-v1"
        snapshot = live.emit(out / "PREDECESSOR_FINAL_CURRENT.json", state)
        record = {"schema": SCHEMA, "approval": state["approved_next_wave"],
                  "plan": state["next_wave_plan"], "predecessor_audit": live.record(audit_path),
                  "predecessor_current": snapshot, "started_at_utc": now.isoformat(),
                  "deadline_utc": (now + dt.timedelta(days=7)).isoformat(),
                  "maximum_frozen_sources": 6, "owner_thread_id": owner}
        activation = live.emit(out / "ACTIVATION.json", record)
        state.update(design_wave_activation=activation, campaign_contract=state["next_wave_contract"],
                     predecessor_final_current=snapshot, next_wave_status="active",
                     started_at_utc=record["started_at_utc"], deadline_utc=record["deadline_utc"],
                     first_live_review_due_utc=(now + dt.timedelta(hours=48)).isoformat(),
                     candidate_count=0, completed_candidates=0, frozen_sources={},
                     research_closed=False, phase="design-wave-d32-budget-preparation",
                     active_packet=None, arena_attempt=None,
                     next_action="Guard and bind candidate1 historical D32 weights with modern990/180 deadlines; measurement-only550/140 comparator. Follow approved design-wave plan.")
        control.atomic(base / "CURRENT.json", state)
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--used-percent", type=float, required=True)
    parser.add_argument("--owner", required=True)
    args = parser.parse_args()
    print(json.dumps(activate(args.root, args.audit, args.used_percent, owner=args.owner)))


if __name__ == "__main__":
    main()
