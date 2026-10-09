"""Validate declared expansion trajectories and freeze bounded teacher parents."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import re

from tools import rank_two_focused_data_v3 as retained

campaign, data, games, rules = retained.campaign, retained.data, retained.games, retained.rules
SCHEMA = campaign.SCHEMA + ".expansion-label-data.v1"
CAMPAIGN_ID = retained.CAMPAIGN_ID
normalized_label = retained.normalized_label


def check_current(plan, action, launch=False):
    current = retained.check_current(plan, action, launch=launch)
    if (current.get("focused_family") != "dd8"
            or current.get("focused_family_lock") != plan["family_lock"]
            or current.get("focused_round_01_declaration") != plan["declaration"]
            or current.get("focused_active_offline_roster") != plan["offline_roster"]):
        raise PermissionError("locked family, round or approved roster changed")
    declaration = campaign.read(campaign.verify(plan["declaration"]))
    if (declaration["principal_intervention"] != "training-coverage-expansion"
            or declaration["architecture"] != [6301, 8, 8, 1]
            or declaration["root_generation_seed"] != 2026100801
            or declaration["parents_per_trajectory"] != {"early_max": 4, "later_max": 2}
            or declaration["new_teacher_parents_max"] != 6144
            or declaration["teacher_nodes"] != 64000
            or declaration["outcomes_as_targets"] is not False):
        raise ValueError("declared expansion teacher recipe differs")
    return current


def validate_games(rows, quotas, seed, actors):
    """Retain the original trajectory checks with explicit root quotas/bindings."""
    if len(rows) != 2 * sum(quotas.values()):
        raise ValueError("declared expansion game count differs")
    ids, roots, states, features = set(), defaultdict(list), {}, {}
    for game in rows:
        if (game.get("schema") != "papersoccer.top-three.training-game.v1"
                or game.get("training_eligible") is not True
                or game.get("source_kind") != "fresh-generated"
                or game.get("rule_terminal") is not True
                or game.get("split") not in ("train", "validation")
                or type(game.get("winner")) is not int or game["winner"] not in (0, 1)
                or type(game.get("candidate_player")) is not int or game["candidate_player"] not in (0, 1)
                or type(game.get("seed")) is not int or game["seed"] != seed):
            raise ValueError("game is not an eligible declared fresh trajectory")
        for key in ("game_id", "root_group_id", "opponent"):
            value = game.get(key)
            if not isinstance(value, str) or not value or any(not 32 <= ord(ch) <= 126 for ch in value):
                raise ValueError("game identity must be printable ASCII")
        if set(game.get("source_sha256", {})) != {"candidate", "opponent"} or any(
                not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
                for value in game["source_sha256"].values()):
            raise ValueError("game source identities absent")
        if game["opponent"] not in actors or game["source_sha256"] != {
                "candidate": actors["rank_3"]["source"]["sha256"],
                "opponent": actors[game["opponent"]]["source"]["sha256"]}:
            raise ValueError("declared physical generation source changed")
        if game.get("search_profiles") != {"candidate": [50, 10], "opponent": [50, 10]}:
            raise ValueError("declared training clock profiles changed")
        if game["game_id"] in ids:
            raise ValueError("duplicate generated game")
        ids.add(game["game_id"])
        root = games.state(game["root_transcript"])
        state_key = games.fingerprint(root)
        feature_key = data.feature_key(rules.encode_active(root))
        if (state_key != game["root_state_sha256"] or root.winner is not None
                or game.get("root_drawn_edges") != len(root.used_segments)):
            raise ValueError("root identity changed")
        for mapping, key in ((states, state_key), (features, feature_key)):
            if key in mapping and mapping[key] != game["root_group_id"]:
                raise ValueError("canonical/reflected root relabelled as independent")
            mapping[key] = game["root_group_id"]
        prefix = game["root_transcript"]
        if prefix and not game["transcript"].startswith(prefix + "/"):
            raise ValueError("game does not extend its root")
        if games.state(game["transcript"]).winner != game["winner"]:
            raise ValueError("game does not reach declared terminal")
        roots[game["root_group_id"]].append(game)
    for paired in roots.values():
        if (len(paired) != 2 or {game["candidate_player"] for game in paired} != {0, 1}
                or len({game["split"] for game in paired}) != 1
                or len({game["root_state_sha256"] for game in paired}) != 1
                or len({game["opponent"] for game in paired}) != 1
                or len({json.dumps(game["source_sha256"], sort_keys=True) for game in paired}) != 1):
            raise ValueError("root pair colors/split/source identity changed")
    if Counter(paired[0]["split"] for paired in roots.values()) != quotas:
        raise ValueError("declared training/validation root quotas differ")
    return rows


def sample_positions(rows, early_max, later_max):
    """Sample before labels, retain every duplicate and its exclusion reason."""
    selected = []
    for game in sorted(rows, key=lambda row: (row["split"] != "train", row["game_id"])):
        actions = game["transcript"].split("/")
        start = len(game["root_transcript"].split("/")) if game["root_transcript"] else 0
        state, candidates = rules.ReplayState(), []
        for turn, action in enumerate(actions):
            if turn >= start and state.winner is None:
                active = rules.encode_active(state)
                candidates.append(dict(position_id=game["game_id"] + ":" + str(turn),
                    game_id=game["game_id"], root_group_id=game["root_group_id"],
                    group_id=game["root_group_id"], source="top-three-fresh-generated",
                    split=game["split"], winner=game["winner"], mover=state.to_move,
                    prefix="/".join(actions[:turn]), edges=len(state.used_segments),
                    parent_active=list(active), canonical_state=games.fingerprint(state),
                    canonical_features=data.feature_key(active)))
            rules.apply_complete_turn(state, state.to_move, action)
        early = [row for row in candidates if row["edges"] <= 12][:early_max]
        later = [row for row in candidates if row["edges"] > 12]
        if len(later) > later_max:
            later = [later[index * (len(later) - 1) // (later_max - 1)] for index in range(later_max)]
        selected.extend(early + later)
    unique, excluded, states, features = [], [], set(), set()
    for row in selected:
        if row["canonical_state"] in states or row["canonical_features"] in features:
            excluded.append(dict(position=row, reason="train-first-canonical-or-reflected-parent-duplicate"))
        else:
            states.add(row["canonical_state"]); features.add(row["canonical_features"])
            unique.append(row)
    return unique, excluded, selected


def audit_generation(plan):
    generation_path = campaign.verify(plan["generation_plan"])
    generation = campaign.read(generation_path)
    result = campaign.read(campaign.verify(plan["generation_result"]))
    completion = campaign.read(campaign.verify(plan["queue_completion"]))
    if (result.get("passed") is not True or result.get("failure") is not None
            or result["plan"] != plan["generation_plan"] or result["games"] != 1024
            or result["strength_eligible"] is not False or result["outcomes_as_targets"] is not False
            or completion["result"] != plan["generation_result"] or completion.get("passed") is not True):
        raise ValueError("complete training-only generation binding differs")
    queue_result = campaign.read(campaign.verify(completion["queue_result"]))
    if queue_result["status"] != "complete" or queue_result["exit_code"] != 0 or queue_result["reason"] is not None:
        raise ValueError("generation queue did not finish cleanly")
    declaration = campaign.read(campaign.verify(plan["declaration"]))
    bank = campaign.read(campaign.verify(generation["bank"]))
    builds = campaign.read(campaign.verify(generation["builds"]))
    roster = campaign.read(campaign.verify(plan["offline_roster"]))
    for name, actor in builds["actors"].items():
        expected = roster["control"] if name == "rank_3" else roster["opponents"][name]
        if actor != {**expected, "clocks_ms": [50, 10]}:
            raise ValueError("training-only actor source/family binding differs")
        campaign.verify(actor["source"])
    path = data.bound(plan["games"])
    if result["trajectories"] != campaign.record(path):
        raise ValueError("physical training export changed")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    quotas = {"train": declaration["fresh_train_roots"], "validation": declaration["fresh_validation_roots"]}
    validate_games(rows, quotas, declaration["root_generation_seed"], builds["actors"])
    envelopes = [json.loads(line) for line in data.bound(plan["envelopes"]).read_text().splitlines()]
    wanted = [(row, color) for row in bank["rows"] for color in (0, 1)]
    if len(wanted) != len(rows) or len(envelopes) != len(rows):
        raise ValueError("bank/receipt/export slot count differs")
    receipts = []
    output = Path(generation["output"])
    for (root, color), envelope, row in zip(wanted, envelopes, rows):
        check_current(plan, "label")
        identifier = root["root_group_id"] + "-p" + str(color)
        directory = output / "games" / identifier
        claim = campaign.read(directory / "CLAIM.json")
        expected_claim = dict(plan=plan["generation_plan"], row=root, color=color,
                             source=builds["actors"]["rank_3"]["source"],
                             opponent=builds["actors"][root["opponent"]])
        if (claim != expected_claim or campaign.read(directory / "RESULT.json") != envelope
                or envelope["plan"] != plan["generation_plan"] or row["game_id"] != identifier
                or row["root_group_id"] != root["root_group_id"] or row["split"] != root["split"]
                or row["candidate_player"] != color or row["root_transcript"] != root["transcript"]
                or row["root_state_sha256"] != root["state_sha256"]
                or games.training_trajectory(envelope) != row):
            raise ValueError("claimed training trajectory and export differ")
        receipts.append(dict(claim=campaign.record(directory / "CLAIM.json"),
                             result=campaign.record(directory / "RESULT.json")))
    if len(list((output / "games").glob("*/CLAIM.json"))) != len(receipts):
        raise ValueError("unknown training game claim retained")
    return rows, dict(passed=True, generation_plan=plan["generation_plan"],
        generation_result=plan["generation_result"], queue_completion=plan["queue_completion"],
        games=plan["games"], envelopes=plan["envelopes"], game_receipts=receipts,
        complete_games=len(rows), operational_failures=0, unknown_claims=0,
        split_roots=quotas, both_colors=True, outcomes_as_targets=False,
        strength_eligible=False, producer=campaign.record(__file__))


def verify_teacher_plan(plan):
    check_current(plan, "label")
    declaration = campaign.read(campaign.verify(plan["declaration"]))
    body = {key: value for key, value in plan.items() if key not in ("bundle_sha256", "command")}
    if hashlib.sha256(campaign.canonical(body)).hexdigest() != plan["bundle_sha256"]:
        raise ValueError("teacher source bundle body changed")
    if (plan["producer"] != campaign.record(__file__) or plan["teacher"] != declaration["teacher"]
            or plan["native_teacher"] != declaration["native_teacher"] or plan["nodes"] != 64000
            or plan["outcomes_as_targets"] is not False or len(plan["positions"]) > 6144
            or plan["command"] != teacher_command(plan, plan["bundle_sha256"])):
        raise ValueError("teacher source, recipe or bounded positions differ")
    return True


def teacher_command(body, digest):
    return [body["native_teacher"]["path"], "--model", body["teacher"]["path"],
        "--model-sha256", data.TEACHER_SHA256, "--campaign-id", CAMPAIGN_ID,
        "--tree-nodes", "64000", "--time-ms", "0", "--max-actions", "250",
        "--max-partial-paths", "50000", "--exploration", "0.5", "--fpu", "0.5",
        "--emit-action-groups", "--source-bundle-body-sha256", digest]


def preparation_checks(rows, actors, output):
    groups = []
    for split in ("train", "validation"):
        root = next(row["root_group_id"] for row in rows if row["split"] == split)
        groups.extend(row for row in rows if row["root_group_id"] == root)
    quotas = {"train": 1, "validation": 1}
    seed = groups[0]["seed"]
    validate_games(groups, quotas, seed, actors)
    cases, checks = [], ["explicit-root-quotas-and-source-binding"]
    mutations = {
        "changed-seed": lambda value: value[0].update(seed=seed + 1),
        "changed-candidate-source": lambda value: value[0]["source_sha256"].update(candidate="0" * 64),
        "changed-training-clocks": lambda value: value[0]["search_profiles"].update(candidate=[550, 140]),
        "duplicate-game-identity": lambda value: value[0].update(game_id=value[1]["game_id"]),
        "missing-paired-color": lambda value: value[0].update(candidate_player=value[1]["candidate_player"]),
        "changed-root-state": lambda value: value[0].update(root_state_sha256="0" * 64),
        "wrong-terminal-winner": lambda value: value[0].update(winner=1 - value[0]["winner"]),
        "cross-split-root": lambda value: value[0].update(split="validation"),
    }
    for name, mutate in mutations.items():
        changed = copy.deepcopy(groups)
        mutate(changed)
        try:
            validate_games(changed, quotas, seed, actors)
        except ValueError as error:
            checks.append(name)
            cases.append(dict(name=name, rows=changed, rejected=str(error), synthetic=True, training_eligible=False))
        else:
            raise AssertionError("invalid trajectory accepted: " + name)
    sampled, excluded, all_sampled = sample_positions(groups, 4, 8)
    if sampled != data.sample_positions(groups):
        raise AssertionError("parameterized sampler differs at retained4/8 quotas")
    checks.append("retained-sampler-parity-at-original-quota")
    _, _, bounded = sample_positions(groups, 4, 2)
    for game in groups:
        selected = [row for row in bounded if row["game_id"] == game["game_id"]]
        assert sum(row["edges"] <= 12 for row in selected) <= 4
        assert sum(row["edges"] > 12 for row in selected) <= 2
        for row in selected:
            physical = games.state(row["prefix"])
            assert physical.to_move == row["mover"] and games.fingerprint(physical) == row["canonical_state"]
            assert list(rules.encode_active(physical)) == row["parent_active"]
    checks.append("declared4/2-parent-quota-and-physical-mover-features")
    campaign.immutable(output / "CHECK_CASES.json", dict(cases=cases, strength_eligible=False,
        synthetic=True, outcomes_as_targets=False))
    campaign.immutable(output / "PREPARATION_CHECKS.json", dict(passed=True, checks=checks,
        cases=campaign.record(output / "CHECK_CASES.json"), number_of_checks=len(checks)))


def prepare(plan_path):
    plan = campaign.read(plan_path)
    check_current(plan, "label", launch=True)
    if plan["schema"] != SCHEMA + ".preparation" or plan["producer"] != campaign.record(__file__):
        raise ValueError("expansion parent preparation producer differs")
    for reference in plan["inputs"].values():
        campaign.verify(reference)
    declaration = campaign.read(campaign.verify(plan["declaration"]))
    output = Path(plan["output"])
    rows, audit = audit_generation(plan)
    campaign.immutable(output / "GENERATION_ARCHIVE.json", audit)
    generation = campaign.read(campaign.verify(plan["generation_plan"]))
    actors = campaign.read(campaign.verify(generation["builds"]))["actors"]
    preparation_checks(rows, actors, output)
    quota = declaration["parents_per_trajectory"]
    positions, excluded, selected = sample_positions(rows, quota["early_max"], quota["later_max"])
    if len(selected) > declaration["new_teacher_parents_max"] or len(positions) > 6144:
        raise ValueError("declared teacher parent cap exceeded")
    early_roots = {row["root_group_id"] for row in positions if row["split"] == "validation" and row["edges"] <= 12}
    if len(early_roots) < 100:
        raise ValueError("fewer than100 independent early validation roots before labels")
    campaign.immutable(output / "SAMPLE_EXCLUSIONS.json", dict(excluded=excluded,
        train_first_precedence=True, canonical_and_reflected_parent_dedup=True))
    campaign.immutable(output / "ALL_SAMPLED_PARENTS.jsonl", b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode() for row in selected))
    payload = data.HEADER + "".join("\t".join(str(row[key]) for key in
        ("position_id", "root_group_id", "group_id", "source", "split", "winner", "mover", "prefix"))
        + "\n" for row in positions)
    campaign.immutable(output / "positions.tsv", payload.encode("ascii"))
    body = dict(schema=SCHEMA + ".teacher-plan", generation_plan=plan["generation_plan"],
        games=plan["games"], teacher=declaration["teacher"], native_teacher=declaration["native_teacher"],
        positions_file=data.record(output / "positions.tsv"), positions=positions, nodes=64000,
        campaign_id=CAMPAIGN_ID, outcomes_as_targets=False, producer=campaign.record(__file__),
        validator=campaign.record(retained.labels.__file__), lifecycle=plan["inputs"]["work_guard"],
        activation=plan["activation"], owner_thread_id=plan["owner_thread_id"],
        campaign_current=plan["campaign_current"], family_lock=plan["family_lock"],
        declaration=plan["declaration"], offline_roster=plan["offline_roster"],
        maximum_position_cpu_seconds=120, maximum_workers=10, maximum_rss_bytes=18 * 1024**3,
        numerical_threads=1, preparation=campaign.record(plan_path),
        generation_archive=campaign.record(output / "GENERATION_ARCHIVE.json"),
        previous_corpus=plan["previous_corpus"], sampling_exclusions=campaign.record(output / "SAMPLE_EXCLUSIONS.json"),
        all_sampled_parents=campaign.record(output / "ALL_SAMPLED_PARENTS.jsonl"),
        preparation_checks=campaign.record(output / "PREPARATION_CHECKS.json"),
        resource_budget=declaration["stage_resources"]["teacher"], inputs=plan["inputs"])
    digest = hashlib.sha256(campaign.canonical(body)).hexdigest()
    teacher_plan = {**body, "bundle_sha256": digest, "command": teacher_command(body, digest)}
    verify_teacher_plan(teacher_plan)
    campaign.immutable(output / "PLAN.json", teacher_plan)
    result = dict(passed=True, preparation=campaign.record(plan_path), teacher_plan=campaign.record(output / "PLAN.json"),
        generation_archive=campaign.record(output / "GENERATION_ARCHIVE.json"),
        games=len(rows), parents=len(positions), sampled_before_dedup=len(selected), excluded_parents=len(excluded),
        split_parents=dict(Counter(row["split"] for row in positions)),
        independent_early_validation_roots_before_labels=len(early_roots), outcomes_as_targets=False,
        previous_corpus_retained=plan["previous_corpus"], old_validation_reserved=True)
    campaign.immutable(output / "PREPARATION_RESULT.json", result)
    campaign.immutable(output / "EXPOSURE_PENDING.json", dict(
        schema=SCHEMA + ".parent-preparation-exposure", input=plan["games"],
        generation_plan=plan["generation_plan"], result=campaign.record(output / "PREPARATION_RESULT.json"),
        inputs=campaign.record(output / "ALL_SAMPLED_PARENTS.jsonl"),
        cases=campaign.record(output / "CHECK_CASES.json"),
        training_eligible=False, before_next_fresh_bank=True, all_excluded_parents_retained=True))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.plan)))


if __name__ == "__main__":
    main()
