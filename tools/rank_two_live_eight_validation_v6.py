"""Continuation05 validation with its immutable declared seed and sampling recipe."""
from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import random
import re
import time

from tools import rank_two_live_eight_v1 as c
from tools import rank_two_live_eight_ancestry_v1 as ancestry
from tools import rank_two_live_eight_data_v2 as references
from tools import rank_two_live_eight_validation_sampling_v1 as bounded_sampling
from tools import rank_two_focused_data_v3 as retained
from tools import rank_two_focused_work_v1 as work
from tools import rank_two_focused_teacher_budget_v1 as teacher

campaign = c.old.retained
data, games, rules, generation = retained.data, retained.games, retained.rules, retained.generation
features, experiments = rules, games
SCHEMA = c.SCHEMA + '.fresh-validation.v1'
SEED = 2026101237


def verify_recipe(declaration):
    expected = dict(colors=[0, 1], maximum_early_parents_per_root=2,
        minimum_comparable_early_roots=100, no_seed_or_root_replacement=True,
        outcomes_as_targets=False, root_depths=[8, 10], roots=128, seed=SEED)
    if (declaration.get('fresh_validation_recipe') != expected
            or declaration.get('architecture') != [6301, 8, 8, 1]
            or declaration.get('outcomes_as_targets') is not False):
        raise PermissionError('immutable continuation05 validation recipe required')


def check_current(plan, action, launch=False):
    state = work.checked(plan, action, launch=launch)
    if ('continuation_declaration' in plan and
            plan['continuation_declaration'] != state.get('live_eight_continuation_05_declaration')):
        raise PermissionError('current continuation05 declaration differs')
    if any(not window.get('all90_archived') for window in state.get('live_eight_known_windows', {}).values()):
        raise PermissionError('archive the active live window before validation work')
    return state


def freeze(base, output, budget, storage):
    base, output = Path(base), Path(output)
    state = c.read(base / 'CURRENT.json')
    preliminary = dict(campaign_current=str(base / 'CURRENT.json'),
                       activation=state['focused_activation'], owner_thread_id=state['owner_thread_id'])
    check_current(preliminary, 'generate', launch=True)
    declaration_ref = state['live_eight_continuation_05_declaration']
    verify_recipe(c.read(c.verify(declaration_ref)))
    if state.get('pending_exposures'):
        raise PermissionError('close all spent exposures before drawing validation roots')
    validation = ancestry.validate(base, output / 'ANCESTRY_VALIDATED.json')
    approval = c.read(c.verify(state['focused_plan']))
    roster = c.read(c.verify(approval['roster_manifest']))
    if roster['control']['source'] != state['incumbent'] or len(roster['opponents']) != 8:
        raise ValueError('the unchanged incumbent and generator roster are required')
    actors = {name: games.descriptor(c.verify(item['source']), item['family'], [50, 10])
              for name, item in roster['opponents'].items()}
    actors['rank_3'] = games.descriptor(c.verify(state['incumbent']), 'turn_action_v2', [50, 10])
    rng, seen, rows = random.Random(SEED), set(), []
    with ancestry.databases(state) as databases:
        for index in range(128):
            depth = (8, 10)[index % 2]
            for attempt in range(10000):
                row = generation.fresh_root(rng, depth, seen)
                feature = data.feature_key(rules.encode_active(games.state(row['transcript'])))
                if feature in seen:
                    continue
                blocked = any(db.execute('SELECT 1 FROM keys WHERE category=? AND value=?',
                              (category, key)).fetchone() is not None
                              for db in databases for category, key in
                              (('states', row['state_sha256']), ('features', feature)))
                seen.add(feature)
                if not blocked:
                    break
            else:
                raise ValueError('frozen validation draw quota exhausted; no silent seed replacement')
            root = f'live-eight-validation-05-{index:04d}'
            rows.append({**row, 'feature_sha256': feature, 'root_group_id': root,
                         'cluster_id': root, 'opponent': sorted(roster['opponents'])[index % 8],
                         'split': 'validation', 'colors': [0, 1]})
    producers = {name: c.record(path) for name, path in dict(
        orchestrator=__file__, generator=generation.__file__, sampler=data.__file__,
        driver=games.__file__, rules=rules.__file__, worker=games.WORKER,
        lifecycle=work.__file__, ancestry=ancestry.__file__).items()}
    producers['controller'] = state['control_guard']
    plan = dict(schema=SCHEMA + '.generation', **preliminary, seed=SEED,
                continuation_declaration=declaration_ref,
                candidate='rank_3', actors=actors, rows=rows, new_games=256,
                root_depths=[8, 10], split_roots={'validation': 128},
                training_clocks_ms=[50, 10], strength_eligible=False, outcomes_as_targets=False,
                training_eligible=False, roster=approval['roster_manifest'],
                layered_ancestry=validation, producers=producers, output=str(output),
                maximum_workers=1, numerical_threads=1, maximum_game_wall_seconds=120,
                resource_budget=budget, storage=storage)
    ref = c.immutable(output / 'PLAN.json', plan)
    exposure = c.immutable(output / 'EXPOSURE_PENDING.json', dict(
        schema=SCHEMA + '.reserved-validation-roots', generation_plan=ref,
        before_next_fresh_bank=True, training_eligible=False, protected_gameplay_reads=False))
    with c.locked(base):
        current = c.read(base / 'CURRENT.json')
        current.setdefault('pending_exposures', []).append(exposure)
        c.atomic(base / 'CURRENT.json', current)
    return ref


def completed_exposure(plan_path, output):
    plan = c.read(plan_path)
    return c.immutable(Path(output) / 'COMPLETE_EXPOSURE.json', dict(
        schema=SCHEMA + '.played-validation-exposure', generation_plan=c.record(plan_path),
        game_receipts=[c.record(Path(output) / 'games' / (row['root_group_id'] + '-p' + str(color)) / 'RESULT.json')
                       for row in plan['rows'] for color in row['colors']],
        before_next_fresh_bank=True, training_eligible=False))


def load_games(path):
    games = [json.loads(line) for line in Path(path).read_text().splitlines()]
    if len(games) != 256: raise ValueError("fresh validation requires exactly256complete games")
    ids, roots, keys = set(), defaultdict(list), {}
    for game in games:
        if (game.get("schema") != "papersoccer.top-three.training-game.v1"
                or game.get("training_eligible") is not False or game.get("source_kind") != "fresh-generated"
                or game.get("rule_terminal") is not True or game.get("split") != "validation"
                or type(game.get("winner")) is not int or game.get("winner") not in (0, 1)
                or type(game.get("candidate_player")) is not int or game.get("candidate_player") not in (0, 1)
                or type(game.get("seed")) is not int or game["seed"] < 0):
            raise ValueError("game is not an eligible fresh training trajectory")
        for key in ("game_id", "root_group_id", "opponent"):
            value = game.get(key)
            if not isinstance(value, str) or not value or any(ord(character) < 32 or ord(character) > 126 for character in value):
                raise ValueError("game identity must be printable ASCII")
        if set(game.get("source_sha256", {})) != {"candidate", "opponent"} or any(
                not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
                for value in game["source_sha256"].values()): raise ValueError("game source identities absent")
        profiles = game.get("search_profiles", {})
        if set(profiles) != {"candidate", "opponent"} or any(
                not isinstance(value, list) or len(value) != 2
                or any(type(clock) is not int or clock < 1 for clock in value)
                or value[0] > 1000 or value[1] > 200 for value in profiles.values()):
            raise ValueError("game generation clock profiles invalid")
        if game["game_id"] in ids: raise ValueError("duplicate generated game")
        ids.add(game["game_id"])
        root = experiments.state(game["root_transcript"])
        key = experiments.fingerprint(root)
        if key != game["root_state_sha256"] or root.winner is not None or game.get("root_drawn_edges") != len(root.used_segments):
            raise ValueError("root identity changed")
        if key in keys and keys[key] != game["root_group_id"]: raise ValueError("canonical root has two group identities")
        keys[key] = game["root_group_id"]
        prefix = game["root_transcript"]
        if prefix and not game["transcript"].startswith(prefix + "/"): raise ValueError("game does not extend its root")
        terminal = experiments.state(game["transcript"])
        if terminal.winner != game["winner"]: raise ValueError("game does not reach declared terminal")
        roots[game["root_group_id"]].append(game)
    for paired in roots.values():
        if (len(paired) != 2 or {game["candidate_player"] for game in paired} != {0, 1}
                or len({game["split"] for game in paired}) != 1
                or len({game["root_state_sha256"] for game in paired}) != 1
                or len({game["opponent"] for game in paired}) != 1
                or len({json.dumps(game["source_sha256"], sort_keys=True) for game in paired}) != 1):
            raise ValueError("root pair colors/split/source identity changed")
    split_roots = Counter(paired[0]["split"] for paired in roots.values())
    if split_roots != {"validation": 128}: raise ValueError("pilot root quota changed")
    if len({game["opponent"] for game in games}) < 4: raise ValueError("pilot requires a multi-opponent distribution")
    if any(row["seed"] != SEED or row["source_sha256"]["candidate"] != c.BASELINE for row in games):
        raise ValueError("validation seed or unchanged incumbent differs")
    return games


def prepare_labels(generation_plan, result_path, output, label_runner, budget):
    plan = c.read(generation_plan)
    check_current(plan, 'label', launch=True)
    result = c.read(result_path)
    if not result['passed'] or result['plan'] != c.record(generation_plan):
        raise ValueError('completed source-bound validation generation required')
    trajectories = load_games(result['trajectories']['path'])
    positions = bounded_sampling.early_parents(data.sample_positions(trajectories))
    if len({row['root_group_id'] for row in positions}) != 128:
        raise ValueError('all128 frozen validation roots require early parents')
    if len(positions) > 3072 or any(position['split'] != 'validation' for position in positions):
        raise ValueError('validation sampler exceeds its fixed quota')
    state = c.read(plan['campaign_current'])
    declaration = c.read(c.verify(state['live_eight_live_data_precision_declaration']))
    output = Path(output)
    c.immutable(output / 'POSITIONS.jsonl', b''.join(
        json.dumps(row, sort_keys=True, separators=(',', ':')).encode('ascii') + b'\n'
        for row in positions))
    body = dict(schema=SCHEMA + '.teacher-plan', positions=positions,
        positions_file=data.record(output / 'POSITIONS.jsonl'), games=data.record(result['trajectories']['path']),
        generation_plan=c.record(generation_plan), generation_result=c.record(result_path),
        teacher=declaration['teacher'], native_teacher=declaration['native_teacher'],
        nodes=256000, outcomes_as_targets=False, scope='validation-only',
        producer=c.record(__file__), validator=c.record(retained.labels.__file__), lifecycle=c.record(work.__file__),
        activation=plan['activation'], owner_thread_id=plan['owner_thread_id'],
        campaign_current=plan['campaign_current'], maximum_workers=9, numerical_threads=1,
        maximum_rss_bytes=18 * 1024**3, maximum_position_cpu_seconds=480,
        maximum_position_wall_seconds=490, resource_budget=budget,
        inputs={'label_runner': c.record(label_runner), 'declaration': state['live_eight_live_data_precision_declaration'],
                'continuation_declaration': plan['continuation_declaration']})
    digest = hashlib.sha256(c.old.retained.canonical(body)).hexdigest()
    return c.immutable(output / 'TEACHER_PLAN.json', dict(body, bundle_sha256=digest,
                       command=teacher.teacher_command(body, digest)))


def verify_teacher_plan(plan):
    check_current(plan, 'label')
    generation_plan = c.read(c.verify(plan['generation_plan']))
    check_current(generation_plan, 'label')
    generated = c.read(c.verify(plan['generation_result']))
    verify_recipe(c.read(c.verify(generation_plan['continuation_declaration'])))
    if (generation_plan['seed'] != SEED or generated.get('passed') is not True
            or generated['plan'] != plan['generation_plan']):
        raise ValueError('fresh validation generation binding differs')
    body = {key: value for key, value in plan.items() if key not in ('bundle_sha256', 'command')}
    if (hashlib.sha256(c.old.retained.canonical(body)).hexdigest() != plan['bundle_sha256']
        or plan['scope'] != 'validation-only' or plan['nodes'] != 256000
        or plan['outcomes_as_targets'] is not False
        or any(position['split'] != 'validation' for position in plan['positions'])
        or len({position['root_group_id'] for position in plan['positions']}) != 128):
        raise ValueError('validation-only teacher identity differs')
    for key in ('teacher', 'native_teacher', 'positions_file', 'games'):
        data.bound(plan[key])


def normalized_label(native, position, plan):
    return teacher.normalized_label(native, position, plan)


def aggregate(plan_path, execution_path, train_features, output):
    plan, execution = c.read(plan_path), c.read(execution_path)
    if not execution['passed'] or execution['plan'] != c.record(plan_path):
        raise ValueError('complete validation teacher execution required')
    output = Path(output); artifacts = []; early = set(); comparable = set(); excluded = []
    for position, ref in zip(plan['positions'], execution['parent_receipts'], strict=True):
        receipt = c.read(c.verify(ref))
        if not receipt['success']:
            if not receipt['unsupported']:
                raise ValueError('unexpected validation label failure')
            excluded.append(dict(position=position['position_id'], reason='unsupported'))
            continue
        native = json.loads(data.bound(references.legacy_reference(receipt['output'])).read_text())
        group = normalized_label(native, position, plan)
        keys = [data.feature_key(group['parent_active']), *[data.feature_key(child['active']) for child in group['successors']]]
        if any(key in train_features for key in keys):
            excluded.append(dict(position=position['position_id'], reason='TRAIN/reflected-feature-overlap'))
            continue
        values = [child['teacher_value'] * (1 if child['value_mover'] == group['mover'] else -1) for child in group['successors']]
        is_comparable = group['exhaustive'] and len(values) > 1 and max(values) - min(values) > 1e-4
        if is_comparable:
            comparable.add(group['root_group_id'])
            if group['edges'] <= 12:
                early.add(group['root_group_id'])
        path = output / 'groups' / (position['position_id'].replace(':', '-') + '.json.gz')
        c.immutable(path, gzip.compress(c.old.retained.canonical(group), mtime=0))
        artifacts.append(dict(group_id=group['group_id'], root_group_id=group['root_group_id'],
            split='validation', edges=group['edges'], comparable_early=is_comparable and group['edges'] <= 12,
            artifact=data.record(path)))
    roots = {row['root_group_id'] for row in artifacts}
    c.immutable(output / 'DIAGNOSTICS.json', dict(roots=len(roots), comparable_roots=len(comparable),
        early_roots=len(early), exclusions=excluded, outcomes_as_targets=False,
        selection_uses_student_or_game_strength=False))
    if len(roots) != 128 or len(early) < 100:
        raise ValueError('128roots/100comparable early admission failed; all diagnostics retained')
    return c.immutable(output / 'RESULT.json', dict(passed=True, roots=128, early_roots=len(early),
        group_artifacts=artifacts, exclusions=excluded, plan=c.record(plan_path),
        execution=c.record(execution_path), protected_gameplay_reads=False,
        outcomes_as_targets=False, training_eligible=False))


def generate(plan_path):
    plan = campaign.read(plan_path)
    check_current(plan, "generate", launch=True)
    if plan["schema"] != SCHEMA + ".generation" or plan["seed"] != SEED or plan["candidate"] != "rank_3":
        raise ValueError("matched generation recipe differs")
    for reference in plan["producers"].values():
        campaign.verify(reference)
    verify_recipe(c.read(c.verify(plan['continuation_declaration'])))
    output = Path(plan["output"])
    names = sorted(plan["actors"])
    builds = {name: games.build_worker(plan["actors"][name], output / "builds", "/usr/bin/clang++")
              for name in names}
    campaign.immutable(output / "builds.json", builds)
    envelopes, started = [], time.monotonic()
    for row in plan["rows"]:
        for color in row["colors"]:
            # Ownership/pause is checked at each boundary; an interrupted claim
            # remains spent and cannot be silently replayed.
            check_current(plan, "generate")
            identifier = row["root_group_id"] + "-p" + str(color)
            directory = output / "games" / identifier
            result_path = directory / "RESULT.json"
            if result_path.exists():
                envelope = campaign.read(result_path)
                if envelope["plan"] != campaign.record(plan_path):
                    raise ValueError("retained trajectory plan changed")
                games.training_trajectory(envelope)
            else:
                if (directory / "CLAIM.json").exists():
                    raise RuntimeError("unknown game claim retained; no automatic replay")
                campaign.immutable(directory / "CLAIM.json", dict(
                    plan=campaign.record(plan_path), row=row, color=color,
                    builds={name: builds[name] for name in ("rank_3", row["opponent"])}))
                begin = time.monotonic()
                def decision_guard():
                    if time.monotonic() - begin >= plan["maximum_game_wall_seconds"]:
                        raise RuntimeError("declared per-game wall cap reached")
                played = games.play(row, color, plan["actors"]["rank_3"], plan["actors"][row["opponent"]],
                                    builds["rank_3"], builds[row["opponent"]], decision_guard=decision_guard)
                envelope = dict(plan=campaign.record(plan_path), game_id=identifier, seed=plan["seed"],
                                split=row["split"], opponent_name=row["opponent"],
                                candidate=plan["actors"]["rank_3"], opponent=plan["actors"][row["opponent"]],
                                root=dict(source_kind="fresh-generated", root_group_id=row["root_group_id"],
                                          transcript=row["transcript"]), game=played,
                                generation_wall_seconds=time.monotonic() - begin)
                campaign.immutable(result_path, envelope)
                campaign.immutable(directory / "EXPOSURE.json", dict(
                    schema=SCHEMA + ".played-trajectory-exposure", input=campaign.record(result_path),
                    generation_plan=campaign.record(plan_path), training_eligible=False,
                    before_next_fresh_bank=True, includes_partial=played["failure"] is not None))
                if played["failure"] is not None:
                    raise RuntimeError("training trajectory operational failure; exact receipt retained")
                games.training_trajectory(envelope)
            envelopes.append(envelope)
            campaign.atomic(output / "PROGRESS.json", dict(games_completed=len(envelopes),
                            wall_seconds=time.monotonic() - started, total_games=256))
    envelope_path = output / "envelopes.jsonl"
    campaign.immutable(envelope_path, b"".join(
        json.dumps(row, sort_keys=True, separators=(',', ':')).encode('ascii') + b'\n'
        for row in envelopes))
    exported = games.export_training(envelope_path, output / "exporter-raw-games.jsonl")
    validation_rows = [json.loads(line) for line in (output / "exporter-raw-games.jsonl").read_text().splitlines()]
    for row in validation_rows:
        if row["split"] != "validation":
            raise ValueError("validation bank cannot supply TRAIN")
        row["training_eligible"] = False
        row["validation_eligible"] = True
    c.immutable(output / "games.jsonl", b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n"
        for row in validation_rows))
    load_games(output / "games.jsonl")
    result = dict(passed=True, plan=campaign.record(plan_path), games=256,
                  trajectories=campaign.record(output / "games.jsonl"), export=exported,
                  wall_seconds=time.monotonic() - started, strength_eligible=False)
    campaign.immutable(output / "RESULT.json", result)
    completed_exposure(plan_path, output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.plan)))


if __name__ == '__main__':
    main()
