"""Retain zero-trace recipe exposures without copying an unchanged census."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools import rank_two_focused_exposure_v4 as retained
from tools import rank_two_focused_work_v1 as work

campaign = retained.campaign
SCHEMA = campaign.SCHEMA + ".metadata-exposure.v1"


def run(plan_path):
    plan = campaign.read(plan_path)
    work.checked(plan, "archive", launch=True)
    if plan["producer"] != campaign.record(__file__):
        raise ValueError("metadata exposure producer changed")
    for ref in plan["inputs"].values():
        campaign.verify(ref)
    seed = campaign.read(campaign.verify(plan["seed_ready"]))
    out = Path(plan["output"])
    campaign.immutable(out / "CLAIM.json", dict(plan=campaign.record(plan_path),
        seed_ready=plan["seed_ready"], database_copy=False))
    normalized = retained.prepare(plan["base"], plan["seed_ready"]["path"],
        plan["exposures"], out / "normalized")
    if normalized["rows"] != 0:
        raise ValueError("nonzero new exposure traces require the existing incremental SQLite extension")
    closure = campaign.read(campaign.verify(normalized["closure"]))
    if closure["rows"] != 0 or not closure["all_payload_inputs_hash_verified_before_census"]:
        raise ValueError("zero-trace metadata identity proof incomplete")
    prior = campaign.read(campaign.verify(seed["prior"]))
    with retained.prior_validator.prior_census(prior):
        pass
    receipt = campaign.immutable(out / "METADATA_RECEIPT.json", dict(passed=True,
        plan=campaign.record(plan_path), normalized_plan=campaign.record(out / "normalized/PLAN.json"),
        normalized_closure=normalized["closure"], new_normalized_rows=0,
        existing_database=prior["inputs"]["database"], original_census_receipt=seed["receipt"],
        exposed_positions_unchanged=True, database_unchanged=True, teacher_keysets_unchanged=True,
        new_games=0, historical_ancestry_rescan=False, historical_protected_reads=False,
        all_previous_databases_retained=True, exposures=plan["exposures"]))
    own_exposure = campaign.immutable(out / "RECIPE_EXPOSURE.json", dict(
        schema=SCHEMA+".recipe", kind="recipe-only", retained_plans=[campaign.record(plan_path)],
        retained_artifacts=[receipt], new_games=0, training_eligible=False,
        before_next_fresh_bank=True))
    import io
    own_normalizer = retained.Normalizer(io.StringIO())
    own_normalizer.bind(own_exposure, parse=True)
    if own_normalizer.rows != 0:
        raise ValueError("metadata admission recipe unexpectedly exposed new positions")
    ready = {**seed, "schema": SCHEMA+".ready", "passed": True,
        "plan": campaign.record(plan_path), "inherited_ready": plan["seed_ready"],
        "metadata_extension_receipt": receipt, "new_normalized_rows": 0,
        "exposures": seed["exposures"]+[x for x in [*plan["exposures"], own_exposure] if x not in seed["exposures"]],
        "rows_replayed": 0, "games_run": 0, "all_previous_databases_retained": True}
    campaign.immutable(out / "ready.json", ready)
    return ready


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    result = run(parser.parse_args().plan)
    print(json.dumps(dict(passed=result["passed"], new_normalized_rows=0,
        database_unchanged=True, exposures=len(result["exposures"])) ))


if __name__ == "__main__":
    main()
