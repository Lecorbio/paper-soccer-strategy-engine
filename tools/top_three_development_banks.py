#!/usr/bin/env python3
"""Deterministic unprotected development roots behind explicit learning exclusions.

This module never reads games, model labels, live replays, or protected banks.
It consumes caller-hash-bound metadata inventories and prior DEVELOPMENT banks.
It cannot create a sealed/final bank or run a game.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path
import platform
import random
import re
import sys

try:
    from . import top_three_experiments as experiments
    from . import top_three_campaign as campaign
    from . import jacek_replay_features as features
except ImportError:
    import top_three_experiments as experiments
    import top_three_campaign as campaign
    import jacek_replay_features as features

SCHEMA = "papersoccer.top-three.development-bank-generator.v1"
SCHEMA_V2 = "papersoccer.top-three.development-bank-generator.v2"
INVENTORY_SCHEMA = "papersoccer.compact-representation.exclusions.v1"
EXPOSURE_SCHEMA = "papersoccer.top-three.played-exclusions.v1"
DEPTHS = (8, 12, 20, 40)
PAIRS = {"screen": 32, "confirmation": 100}
MAX_PER_SLOT = 512
MAX_PROPOSALS = 20000
HEX = re.compile(r"[0-9a-f]{64}")


class ProposalBudgetError(ValueError):
    def __init__(self, message, rows, receipts):
        super().__init__(message)
        self.rows, self.receipts = rows, receipts


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def record(path):
    return experiments.raw_record(Path(path))


def load_bound(path, expected):
    if not isinstance(expected, str) or HEX.fullmatch(expected) is None:
        raise ValueError("explicit 64-hex expected hash required")
    path = Path(path).resolve()
    raw = path.read_bytes()
    if sha(raw) != expected:
        raise ValueError("exclusion/bank hash mismatch")
    return json.loads(raw), {"path": str(path), "sha256": expected}


def sorted_hashes(values, label):
    if not isinstance(values, list) or not values or any(not isinstance(v, str) or HEX.fullmatch(v) is None for v in values):
        raise ValueError(f"{label} must be a nonempty 64-hex inventory")
    if values != sorted(set(values)):
        raise ValueError(f"{label} is not sorted/unique")
    return set(values)


def feature_key(position):
    active = features.encode_active(position)
    active = min(active, features.reflect_active(active))
    return sha(b"".join(int(index).to_bytes(2, "little") for index in active))


def learning_inventory(path, expected):
    value, identity = load_bound(path, expected)
    if value.get("schema") != INVENTORY_SCHEMA or value.get("complete_coverage") is not True:
        raise ValueError("learning inventory is absent, incomplete, or has the wrong schema")
    if (value.get("includes_overlap_removed_groups") is not True or
            value.get("includes_prefix_and_terminal_boundaries") is not True):
        raise ValueError("inventory omits discarded groups or trajectory boundaries")
    if (value.get("state_function_sha256") != sha(inspect.getsource(experiments.fingerprint).encode()) or
            value.get("sources", {}).get("features", {}).get("sha256") != record(features.__file__)["sha256"]):
        raise ValueError("inventory fingerprint/feature algorithm is incompatible")
    if value.get("game_count") != 512 or value.get("root_group_count") != 256 or value.get("splits") != ["train", "validation"]:
        raise ValueError("inventory does not cover both complete pilot splits")
    if type(value.get("supervised_group_count")) is not int or value["supervised_group_count"] < 1:
        raise ValueError("supervised parent/successor coverage is missing")
    clusters = value.get("root_group_ids")
    if not isinstance(clusters, list) or len(clusters) != 256 or clusters != sorted(set(clusters)) or any(not isinstance(c, str) or not c for c in clusters):
        raise ValueError("pilot root group inventory is incomplete")
    inputs = value.get("inputs", {})
    for role in ("games", "labels"):
        item = inputs.get(role, {})
        if (not isinstance(item.get("path"), str) or not item["path"] or
                HEX.fullmatch(str(item.get("sha256", ""))) is None):
            raise ValueError("input identity receipts are missing")
        # These are provenance references, deliberately NEVER opened here.
    states = sorted_hashes(value.get("canonical_state_sha256"), "state hashes")
    feature_keys = sorted_hashes(value.get("canonical_feature_sha256"), "feature hashes")
    return {"states": states, "features": feature_keys, "clusters": set(clusters)}, identity


def exposure_inventory(path, expected, campaign_root=None):
    """Read only hash-bound metadata; never open scope/input trajectory refs."""
    value, identity = load_bound(path, expected)
    if (value.get("schema") != EXPOSURE_SCHEMA or value.get("complete_coverage") is not True or
            value.get("includes_primitive_states") is not True or value.get("includes_turn_boundaries") is not True):
        raise ValueError("played exposure inventory is incomplete or incompatible")
    root_value = value.get("campaign_root")
    if not isinstance(root_value, str) or not Path(root_value).is_absolute():
        raise ValueError("exposure campaign_root must be absolute")
    root = Path(root_value).resolve()
    if campaign_root is not None and root != Path(campaign_root).resolve():
        raise ValueError("exposure inventory belongs to another current campaign")
    if (value.get("state_function_sha256") != sha(inspect.getsource(experiments.fingerprint).encode()) or
            value.get("sources", {}).get("features", {}).get("sha256") != record(features.__file__)["sha256"]):
        raise ValueError("exposure fingerprint/feature algorithm is incompatible")
    inputs = value.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("exposure input identity list is missing")
    for item in [value.get("scope_manifest"), *inputs]:
        if (not isinstance(item, dict) or not isinstance(item.get("path"), str) or
                not Path(item["path"]).is_absolute() or HEX.fullmatch(str(item.get("sha256", ""))) is None):
            raise ValueError("exposure provenance identity is invalid")
        if not Path(item["path"]).resolve().is_relative_to(root):
            raise ValueError("exposure input is outside the current campaign")
        # Deliberately do not read or verify these referenced payload bytes.
    clusters = value.get("root_group_ids")
    if (not isinstance(clusters, list) or any(not isinstance(x, str) or not x for x in clusters)
            or clusters != sorted(set(clusters))):
        raise ValueError("exposure root groups must be sorted and unique")
    return {"states": sorted_hashes(value.get("canonical_state_sha256"), "exposure states"),
            "features": sorted_hashes(value.get("canonical_feature_sha256"), "exposure features"),
            "clusters": set(clusters)}, identity


def candidate_identity(argument):
    path, expected = argument
    path = Path(path).resolve()
    if not isinstance(expected, str) or HEX.fullmatch(expected) is None or record(path)["sha256"] != expected:
        raise ValueError("reserved candidate source hash mismatch")
    return {"path": str(path), "sha256": expected}


def exclusions(inventories, prior_banks, stage, exposures=(), campaign_root=None):
    if stage not in PAIRS:
        raise ValueError("only unprotected screen/confirmation stages are supported")
    if not inventories:
        raise ValueError("complete learning exclusion inventory required before generating roots")
    result = {"states": set(), "features": set(), "clusters": set()}
    inventory_refs, bank_refs, exposure_refs = [], [], []
    for path, expected in inventories:
        values, reference = learning_inventory(path, expected)
        inventory_refs.append(reference)
        for key in result:
            result[key].update(values[key])
    for path, expected in exposures:
        values, reference = exposure_inventory(path, expected, campaign_root)
        exposure_refs.append(reference)
        for key in result:
            result[key].update(values[key])
    screen_seen = False
    known_hashes = {item["sha256"] for item in inventory_refs}
    exposure_hashes = {item["sha256"] for item in exposure_refs}
    prior_hashes = {expected for _, expected in prior_banks}
    for path, expected in prior_banks:
        previous, reference = load_bound(path, expected)
        if (previous.get("schema") != experiments.SCHEMA or previous.get("kind") != "bank"
                or previous.get("stage") not in PAIRS or previous.get("protected_data") is not False):
            raise ValueError("only explicit prior unprotected development banks are eligible exclusions")
        required = {item["sha256"] for item in previous.get("learning_inventories", [])}
        if not required <= known_hashes:
            raise ValueError("supply every learning inventory used by the previous development bank")
        if not {item["sha256"] for item in previous.get("exposure_inventories", [])} <= exposure_hashes:
            raise ValueError("supply every exposure inventory used by the previous development bank")
        if exposures and not {item["sha256"] for item in previous.get("exclusions", [])} <= prior_hashes:
            raise ValueError("supply every prior bank used by the previous development bank")
        for row in previous["rows"]:
            position = experiments.state(row["transcript"])
            key = experiments.fingerprint(position)
            if position.winner is not None or not position.used_segments or key != row["state_sha256"]:
                raise ValueError("previous bank contains an invalid root")
            result["states"].add(key)
            result["features"].add(feature_key(position))
            result["clusters"].add(row["cluster_id"])
        screen_seen |= previous["stage"] == "screen"
        bank_refs.append(reference)
    if stage == "confirmation" and not screen_seen:
        raise ValueError("confirmation requires an explicitly hash-bound prior screen bank")
    return result, inventory_refs, bank_refs


def slots(opponents, depths, per_depth):
    return [{"opponent": opponent, "drawn_edges": depth, "physical_mover": (index + depth_index + opponent_index) % 2,
             "depth_slot": index}
            for opponent_index, opponent in enumerate(opponents)
            for depth_index, depth in enumerate(depths) for index in range(per_depth)]


def proposal_seed(seed, stage, slot, attempt):
    return int.from_bytes(hashlib.sha256(canonical([SCHEMA, seed, stage, slot, attempt])).digest()[:16], "little")


def propose(seed, depth):
    """One bounded random legal walk; reject rather than cross a turn boundary."""
    rng = random.Random(seed)
    position = features.ReplayState()
    actions, action = [], ""
    for _ in range(depth):
        mover = position.to_move
        options = []
        x, y = position.ball
        for direction, (dx, dy) in enumerate(features.DIRECTION_DELTAS):
            if features._legal_destination(position, (x + dx, y + dy)):
                options.append(direction)
        if not options:
            return None, None, "no-legal-edge"
        direction = options[rng.randrange(len(options))]
        features.apply_primitive(position, direction)
        action += str(direction)
        if position.winner is not None:
            return None, None, "terminal"
        if position.to_move != mover:
            actions.append(action)
            action = ""
    if action:
        return None, None, "incomplete-turn-boundary"
    return position, "/".join(actions), None


def generate_rows(opponents, depths, per_depth, seed, stage, excluded,
                  max_per_slot=MAX_PER_SLOT, max_proposals=MAX_PROPOSALS):
    """Pure bounded generator, also used with tiny nonproduction test dimensions."""
    if type(seed) is not int or not 0 <= seed < 2**64 or max_per_slot < 1 or max_proposals < 1:
        raise ValueError("invalid explicit seed/proposal bounds")
    states, feature_keys, clusters = set(), set(), set()
    rows, receipts = [], []
    for slot_index, slot in enumerate(slots(opponents, depths, per_depth)):
        cluster = "top-three-" + stage + ":" + sha(canonical([seed, slot]))
        if cluster in excluded["clusters"] or cluster in clusters:
            raise ValueError("root cluster was already exposed; choose a new generation seed")
        accepted = False
        for attempt in range(max_per_slot):
            if len(receipts) >= max_proposals:
                raise ProposalBudgetError("aggregate proposal budget exhausted; no bank qualifies", rows, receipts)
            rng_seed = proposal_seed(seed, stage, slot, attempt)
            position, transcript, reason = propose(rng_seed, slot["drawn_edges"])
            receipt = {"slot": slot_index, "attempt": attempt, "proposal_seed": str(rng_seed), **slot}
            if position is not None:
                state_hash = experiments.fingerprint(position)
                feature_hash = feature_key(position)
                receipt.update(state_sha256=state_hash, canonical_feature_sha256=feature_hash,
                               transcript_sha256=sha(transcript.encode("ascii")),
                               proposed_physical_mover=position.to_move)
                if position.to_move != slot["physical_mover"]:
                    reason = "wrong-physical-mover"
                elif state_hash in excluded["states"]:
                    reason = "excluded-state"
                elif feature_hash in excluded["features"]:
                    reason = "excluded-neural-features"
                elif state_hash in states:
                    reason = "duplicate-state"
                elif feature_hash in feature_keys:
                    reason = "duplicate-neural-features"
                else:
                    reason = "accepted"
                    row = {"opponent": slot["opponent"], "transcript": transcript, "cluster_id": cluster,
                           "state_sha256": state_hash, "drawn_edges": slot["drawn_edges"],
                           "physical_mover": position.to_move, "canonical_feature_sha256": feature_hash,
                           "proposal_slot": slot_index, "proposal_attempt": attempt, "proposal_seed": str(rng_seed)}
                    rows.append(row)
                    states.add(state_hash); feature_keys.add(feature_hash); clusters.add(cluster)
                    accepted = True
            receipt["status"] = reason
            receipts.append(receipt)
            if accepted:
                break
        if not accepted:
            raise ProposalBudgetError("per-root proposal budget exhausted; no bank qualifies", rows, receipts)
    return rows, receipts


def validate_generated_contract(bank, names):
    """Extra validation called by the normal qualification runner before games."""
    version = bank.get("generator_schema")
    if version not in (SCHEMA, SCHEMA_V2):
        raise ValueError("wrong generated-bank contract")
    stage = bank["stage"]
    if stage not in PAIRS or bank["pairs_per_opponent"] != PAIRS[stage]:
        raise ValueError("unsupported generated-bank stage")
    inventory_args = [(r["path"], r["sha256"]) for r in bank["learning_inventories"]]
    prior_args = [(r["path"], r["sha256"]) for r in bank["exclusions"]]
    exposure_args = []
    if version == SCHEMA_V2:
        refs = bank.get("exposure_inventories")
        reservation = bank.get("candidate_source")
        root = bank.get("campaign_root")
        if not refs or not isinstance(reservation, dict) or not isinstance(root, str) or not Path(root).is_absolute():
            raise ValueError("v2 bank requires exposure inventories and candidate source reservation")
        candidate_identity((reservation["path"], reservation["sha256"]))
        exposure_args = [(r["path"], r["sha256"]) for r in refs]
    excluded, _, _ = exclusions(inventory_args, prior_args, stage, exposure_args, bank.get("campaign_root"))
    plan = experiments.read(experiments.verify(bank["generation_plan"]))
    if (plan["schema"] != version or plan["stage"] != stage or plan["seed"] != bank["seed"] or
            plan["learning_inventories"] != bank["learning_inventories"] or plan["prior_banks"] != bank["exclusions"] or
            set(plan["opponents"]) != set(names) or plan["depths"] != list(DEPTHS) or plan["per_depth"] != PAIRS[stage] // 4):
        raise ValueError("generation plan differs from the frozen bank")
    if version == SCHEMA_V2 and any(plan.get(key) != bank.get(key) for key in ("exposure_inventories", "candidate_source", "campaign_root")):
        raise ValueError("v2 exposure/source reservation differs from generation plan")
    if (type(bank["seed"]) is not int or not 0 <= bank["seed"] < 2**64 or
            plan["max_per_slot"] != MAX_PER_SLOT or plan["max_proposals"] != MAX_PROPOSALS):
        raise ValueError("generation seed/proposal contract changed")
    if set(plan["sources"]) != {"generator", "experiments", "rules", "campaign"}:
        raise ValueError("generator source closure is incomplete")
    for source in plan["sources"].values():
        experiments.verify(source)
    expected_slots = slots(plan["opponents"], DEPTHS, PAIRS[stage] // 4)
    if len(bank["rows"]) != len(expected_slots):
        raise ValueError("generated root count changed")
    inputs = experiments.verify(bank["input"])
    expected_input = b"".join(canonical({k: row[k] for k in ("opponent", "transcript", "cluster_id")}) for row in bank["rows"])
    if inputs.read_bytes() != expected_input:
        raise ValueError("root input export differs from bank rows")
    features_seen, states_seen, clusters_seen = set(), set(), set()
    for index, (row, slot) in enumerate(zip(bank["rows"], expected_slots)):
        if type(row["proposal_attempt"]) is not int or not 0 <= row["proposal_attempt"] < plan["max_per_slot"]:
            raise ValueError("invalid proposal attempt")
        position = experiments.state(row["transcript"])
        state_hash, features_hash = experiments.fingerprint(position), feature_key(position)
        cluster = "top-three-" + stage + ":" + sha(canonical([bank["seed"], slot]))
        if (position.winner is not None or position.to_move != slot["physical_mover"] or
                len(position.used_segments) != slot["drawn_edges"] or row["opponent"] != slot["opponent"] or
                row["physical_mover"] != position.to_move or row["state_sha256"] != state_hash or
                row["canonical_feature_sha256"] != features_hash or row["cluster_id"] != cluster or row["proposal_slot"] != index or
                row["proposal_seed"] != str(proposal_seed(bank["seed"], stage, slot, row["proposal_attempt"]))):
            raise ValueError("generated root/seed/mover identity changed")
        if (state_hash in states_seen or state_hash in excluded["states"] or
                features_hash in features_seen or features_hash in excluded["features"] or
                cluster in clusters_seen or cluster in excluded["clusters"]):
            raise ValueError("generated root overlaps an excluded or duplicate state/feature/cluster")
        states_seen.add(state_hash); features_seen.add(features_hash); clusters_seen.add(cluster)
    receipts_path = experiments.verify(bank["proposal_receipts"])
    receipts = [json.loads(line) for line in receipts_path.read_text().splitlines()]
    if len(receipts) != bank["proposal_count"] or len(receipts) > plan["max_proposals"]:
        raise ValueError("proposal count exceeds or differs from frozen budget")
    slot_index, attempt = 0, 0
    allowed = {"accepted", "terminal", "no-legal-edge", "incomplete-turn-boundary", "wrong-physical-mover",
               "excluded-state", "excluded-neural-features", "duplicate-state", "duplicate-neural-features"}
    for receipt in receipts:
        if slot_index >= len(expected_slots):
            raise ValueError("proposal stream continues after its final accepted root")
        slot = expected_slots[slot_index]
        if (receipt["slot"] != slot_index or receipt["attempt"] != attempt or
                any(receipt[key] != value for key, value in slot.items()) or
                receipt["proposal_seed"] != str(proposal_seed(bank["seed"], stage, slot, attempt)) or
                receipt["status"] not in allowed or attempt >= plan["max_per_slot"]):
            raise ValueError("proposal sequence, seed, or disposition changed")
        if receipt["status"] == "accepted":
            if receipt["transcript_sha256"] != sha(bank["rows"][slot_index]["transcript"].encode("ascii")):
                raise ValueError("accepted proposal transcript binding changed")
            slot_index += 1
            attempt = 0
        else:
            attempt += 1
    if slot_index != len(expected_slots):
        raise ValueError("proposal stream has an unfilled slot")
    accepted = [row for row in receipts if row["status"] == "accepted"]
    if [(r["slot"], r["state_sha256"], r["canonical_feature_sha256"], r["proposal_seed"]) for r in accepted] != [
            (r["proposal_slot"], r["state_sha256"], r["canonical_feature_sha256"], r["proposal_seed"]) for r in bank["rows"]]:
        raise ValueError("proposal receipts do not account for every accepted root")
    if bank["rejection_counts"] != dict(Counter(r["status"] for r in receipts if r["status"] != "accepted")):
        raise ValueError("rejection summary differs from proposal receipts")
    if version == SCHEMA_V2:
        replay_rows, replay_receipts = generate_rows(plan["opponents"], DEPTHS, plan["per_depth"], bank["seed"], stage,
                                                     excluded, plan["max_per_slot"], plan["max_proposals"])
        if replay_rows != bank["rows"] or replay_receipts != receipts:
            raise ValueError("v2 deterministic proposal replay changed")


def freeze(root, output, stage, seed, inventories, prior_banks=(), exposures=(), candidate_source=None):
    # No output or proposal is allocated before a complete approved inventory.
    version = SCHEMA_V2 if exposures or candidate_source is not None else SCHEMA
    if version == SCHEMA_V2 and (not exposures or candidate_source is None):
        raise ValueError("v2 requires both explicit exposure inventory and candidate source")
    reserved = candidate_identity(candidate_source) if candidate_source is not None else None
    excluded, inventory_refs, bank_refs = exclusions(inventories, prior_banks, stage, exposures, root)
    exposure_refs = [{"path": str(Path(path).resolve()), "sha256": expected} for path, expected in exposures]
    extra = {} if version == SCHEMA else {"exposure_inventories": exposure_refs, "candidate_source": reserved,
                                        "campaign_root": str(Path(root).resolve())}
    manifest = campaign.verify(Path(root))
    opponents = list(campaign.CONTROLS)
    if set(manifest["controls"]) != set(opponents):
        raise ValueError("frozen seven-opponent roster changed")
    output = Path(output).resolve()
    sources = {}
    for label, source in (("generator", Path(__file__)), ("experiments", Path(experiments.__file__)),
                          ("rules", Path(features.__file__)), ("campaign", Path(campaign.__file__))):
        raw = source.read_bytes()
        target = output / "producers" / source.name
        campaign.immutable(target, raw)
        sources[label] = record(target)
    plan = {"schema": version, "stage": stage, "seed": seed, "opponents": opponents, "depths": list(DEPTHS),
            "per_depth": PAIRS[stage] // 4, "max_per_slot": MAX_PER_SLOT, "max_proposals": MAX_PROPOSALS,
            "learning_inventories": inventory_refs, "prior_banks": bank_refs, "sources": sources,
            "campaign": record(Path(root) / "campaign.json"), "purpose": "unprotected-development-only",
            "python": {"implementation": platform.python_implementation(), "version": sys.version,
                       "executable": record(sys.executable)}, "games_run": 0, **extra}
    experiments.emit(output / "plan.json", plan)
    try:
        rows, receipts = generate_rows(opponents, DEPTHS, PAIRS[stage] // 4, seed, stage, excluded)
    except ProposalBudgetError as error:
        campaign.immutable(output / "failed-proposals.jsonl", b"".join(canonical(r) for r in error.receipts))
        experiments.emit(output / "failure.json", {"schema": version, "plan": record(output / "plan.json"),
                         "reason": str(error), "partial_roots": len(error.rows), "bank_created": False,
                         "proposal_receipts": record(output / "failed-proposals.jsonl")})
        raise
    proposal_content = b"".join(canonical(r) for r in receipts)
    campaign.immutable(output / "proposals.jsonl", proposal_content)
    inputs = b"".join(canonical({k: row[k] for k in ("opponent", "transcript", "cluster_id")}) for row in rows)
    campaign.immutable(output / "roots.jsonl", inputs)
    bank = {"schema": experiments.SCHEMA, "kind": "bank", "stage": stage,
            "pairs_per_opponent": PAIRS[stage], "input": record(output / "roots.jsonl"),
            "exclusions": bank_refs, "rows": rows, "canonical_rules": record(features.__file__), "protected_data": False,
            "generator_schema": version, "seed": seed, "learning_inventories": inventory_refs,
            "generation_plan": record(output / "plan.json"), "proposal_receipts": record(output / "proposals.jsonl"),
            "proposal_count": len(receipts), "rejection_counts": dict(Counter(r["status"] for r in receipts if r["status"] != "accepted")), **extra}
    experiments.validate_bank(bank, opponents)
    experiments.emit(output / "bank.json", bank)
    return {"bank": record(output / "bank.json"), "roots": len(rows), "proposals": len(receipts),
            "stage": stage, "games_run": 0, "protected_bank_created": False}


def check(path, replay_proposals=False):
    bank = experiments.read(path)
    names = list(campaign.CONTROLS)
    experiments.validate_bank(bank, names)
    if replay_proposals and bank.get("generator_schema") != SCHEMA_V2:
        plan = experiments.read(bank["generation_plan"]["path"])
        excluded, _, _ = exclusions([(r["path"], r["sha256"]) for r in bank["learning_inventories"]],
                                    [(r["path"], r["sha256"]) for r in bank["exclusions"]], bank["stage"])
        rows, receipts = generate_rows(plan["opponents"], plan["depths"], plan["per_depth"], bank["seed"], bank["stage"], excluded,
                                       plan["max_per_slot"], plan["max_proposals"])
        if rows != bank["rows"] or sha(b"".join(canonical(r) for r in receipts)) != bank["proposal_receipts"]["sha256"]:
            raise ValueError("deterministic proposal replay changed")
    return {"status": "verified", "bank": record(path), "roots": len(bank["rows"]),
            "proposal_replay": replay_proposals or bank.get("generator_schema") == SCHEMA_V2}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("freeze")
    create.add_argument("--root", type=Path, default=campaign.DEFAULT)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--stage", choices=PAIRS, required=True)
    create.add_argument("--seed", type=int, required=True)
    create.add_argument("--inventory", nargs=2, action="append", metavar=("PATH", "SHA256"), required=True)
    create.add_argument("--prior-bank", nargs=2, action="append", metavar=("PATH", "SHA256"), default=[])
    create.add_argument("--exposure-inventory", nargs=2, action="append", metavar=("PATH", "SHA256"), default=[])
    create.add_argument("--candidate-source", nargs=2, metavar=("PATH", "SHA256"))
    verify = commands.add_parser("check")
    verify.add_argument("--bank", type=Path, required=True)
    verify.add_argument("--replay-proposals", action="store_true")
    args = parser.parse_args()
    result = (freeze(args.root, args.output, args.stage, args.seed, args.inventory, args.prior_bank,
                     args.exposure_inventory, args.candidate_source)
              if args.command == "freeze" else check(args.bank, args.replay_proposals))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
