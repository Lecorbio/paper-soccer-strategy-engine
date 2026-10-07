#!/usr/bin/env python3
"""Freeze reproducible controls and inspect source-bound top-three experiments.

This campaign is independent of closed historical promotion/training campaigns.
No command uploads a bot, opens protected replay content, or starts training.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'results/top_three_20260916'
SCHEMA = 'papersoccer.top-three-campaign.v1'
CONTROLS = {
    'rank_4': ('submissions/codingame/bots/rank_4/submission.cpp', None,
               '5c7ebbb38e3b08940eb26ca8cd7585dc5cbce5ad949dfd595bfb0eaab1de53c9'),
    'compact_deployed': ('submissions/codingame/bots/compact_value_bfm/discrete_v3_deployment.cpp', None,
                         'add71c369052f232209d69c3b40b6bb459a2d7326ef15c5980377b1526fb8ea9'),
    'h62': ('submissions/codingame/bots/jacek_native_bfm/submission.cpp',
            'a7dd201dbaf32b98f6d661fe4b076c4c769e1815',
            '653eba7d4b5f9b3e8737a6fb50bf16945e416bcdbc53e72520a6ee68acbbef90'),
    **{name: (f'submissions/codingame/bots/{name}/submission.cpp', None, None)
       for name in ('rank_4_jacek_hybrid', 'rank_4_fullturn_bfm', 'challenger', 'neural_puct')},
}
TEACHER_SHA = 'f7bdb201a377c04531f1ba98fd73457f7f77961aa0f0f9b1ac32c59b6e85ee75'


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()


def immutable(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f'immutable artifact differs: {path}')
        return
    with path.open('xb') as handle:
        handle.write(data)
    path.chmod(0o444)


def record(path: Path, relative_to: Path) -> dict:
    data = path.read_bytes()
    return {'path': str(path.relative_to(relative_to)), 'bytes': len(data), 'sha256': digest(data)}


def verify_record(root: Path, item: dict) -> Path:
    path = (root / item['path']).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('artifact escapes campaign root')
    data = path.read_bytes()
    if digest(data) != item['sha256'] or len(data) != item['bytes']:
        raise ValueError(f'artifact identity changed: {path}')
    return path


def freeze(root: Path, teacher: Path | None = None) -> dict:
    if (root / 'campaign.json').exists():
        manifest = verify(root)
        if teacher and (digest(teacher.read_bytes()) != TEACHER_SHA
                        or manifest['models'].get('teacher', {}).get('sha256') != TEACHER_SHA):
            raise ValueError('existing campaign cannot be rebound to another teacher')
        return manifest
    controls = {}
    for name, (relative, revision, expected) in CONTROLS.items():
        data = (subprocess.check_output(['git', 'show', f'{revision}:{relative}'], cwd=ROOT)
                if revision else (ROOT / relative).read_bytes())
        data.decode('ascii')
        if expected and digest(data) != expected:
            raise ValueError(f'{name} does not match verified historical identity')
        if len(data) >= 100000:
            raise ValueError(f'{name} exceeds submission limit')
        destination = root / 'controls' / name / 'submission.cpp'
        immutable(destination, data)
        controls[name] = record(destination, root) | {
            'origin': relative, 'origin_revision': revision,
            'clocks_ms': [800, 165] if name in ('rank_4', 'rank_4_jacek_hybrid', 'rank_4_fullturn_bfm')
                         else [800, 155] if name in ('compact_deployed', 'h62') else [650, 130]}
    h62 = ROOT / 'models/jacek_native_history62_champion.runtime'
    if digest(h62.read_bytes()) != '17038c104bf79c4d5c4c47f09ea144acdeb5dc8e2b01137d46f6b0c589d304c3':
        raise ValueError('H62 runtime changed')
    h62_copy = root / 'controls/h62/model.runtime'
    immutable(h62_copy, h62.read_bytes())
    model_records = {'h62': record(h62_copy, root)}
    if teacher:
        data = teacher.read_bytes()
        if digest(data) != TEACHER_SHA:
            raise ValueError('teacher is not the accepted local teacher')
        destination = root / 'controls/teacher/model.runtime'
        immutable(destination, data)
        model_records['teacher'] = record(destination, root)
    manifest = {
        'schema': SCHEMA, 'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'controls': controls, 'models': model_records,
        'resource_limits': {'workers': 4, 'aggregate_rss_bytes': 12 * 1024**3,
                            'authoritative_timing_workers': 1, 'paid_compute_authorized': False},
        'qualification': {'screen_pairs_per_opponent': 32, 'confirmation_pairs_per_opponent': 100,
                          'screen_minimum_uplift': 0.03, 'maximum_opponent_regression': 0.05,
                          'confirmation_paired_lower_above': 0.0, 'external_deadlines_ms': [1000, 200]},
        'success': {'live_rank_at_most': 3, 'calibration_complete': True, 'own_operational_failures': 0},
    }
    immutable(root / 'campaign.json', canonical(manifest))
    return manifest


def verify(root: Path) -> dict:
    manifest = json.loads((root / 'campaign.json').read_text())
    if manifest['schema'] != SCHEMA or set(manifest['controls']) != set(CONTROLS):
        raise ValueError('wrong campaign schema or control roster')
    for item in (*manifest['controls'].values(), *manifest['models'].values()):
        verify_record(root, item)
    for name, (_, _, expected) in CONTROLS.items():
        if expected and manifest['controls'][name]['sha256'] != expected:
            raise ValueError('historical control hash changed')
    if ('teacher' in manifest['models']
            and manifest['models']['teacher']['sha256'] != TEACHER_SHA):
        raise ValueError('accepted teacher identity changed')
    return manifest


def api(service: str, body) -> tuple[bytes, object]:
    request = urllib.request.Request('https://www.codingame.com/services/' + service,
        data=json.dumps(body).encode(), headers={'Content-Type': 'application/json', 'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read()
    return raw, json.loads(raw)


def live_snapshot(root: Path, pseudo: str = 'Lecorbio') -> dict:
    verify(root)
    raw, leaderboard = api('Leaderboards/getFilteredPuzzleLeaderboard',
        ['paper-soccer', None, 'global', {'active': False, 'column': '', 'filter': ''}])
    payload = root / 'live/public_metadata' / (digest(raw) + '.json')
    immutable(payload, raw)
    users = leaderboard['users']
    selected = [row for row in users if row.get('codingamer', {}).get('pseudo') == pseudo]
    if len(selected) != 1:
        raise ValueError('expected exactly one public user')
    user = selected[0]
    battle_raw, battles = api('gamesPlayersRankingRemoteService/findLastBattlesByAgentId', [user['agentId'], None])
    battle_path = root / 'live/public_metadata' / (digest(battle_raw) + '.json')
    immutable(battle_path, battle_raw)
    receipt = {'observed_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
               'pseudo': pseudo, 'rank': user['rank'], 'score': user['score'],
               'agent_id': user['agentId'], 'in_progress': user.get('inProgress'),
               'percentage': user.get('percentage'), 'leaderboard_count': leaderboard['count'],
               'leaderboard': record(payload, root), 'battles': record(battle_path, root),
               'source_binding': 'public-metadata-only; requires separate editor copyback and submission attestation'}
    immutable(root / 'live/snapshots' / (digest(canonical(receipt)) + '.json'), canonical(receipt))
    return receipt


def build(root: Path, names: list[str], compiler: str) -> dict:
    manifest = verify(root)
    out = {}
    for name in names:
        source = verify_record(root, manifest['controls'][name])
        executable = root / 'build' / name
        executable.parent.mkdir(parents=True, exist_ok=True)
        command = [compiler, '-std=c++20', '-O3', '-DNDEBUG', str(source), '-o', str(executable)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=300)
        if result.returncode:
            raise RuntimeError(result.stderr)
        entry = {'source': manifest['controls'][name], 'executable': record(executable, root),
                 'compiler': subprocess.check_output([compiler, '--version'], text=True), 'command': command}
        out[name] = entry
        immutable(root / 'build/receipts' / (digest(canonical(entry)) + '.json'), canonical(entry))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT)
    commands = parser.add_subparsers(dest='command', required=True)
    freezer = commands.add_parser('freeze'); freezer.add_argument('--teacher', type=Path)
    commands.add_parser('verify')
    builder = commands.add_parser('build'); builder.add_argument('names', nargs='+', choices=tuple(CONTROLS))
    builder.add_argument('--compiler', default='/usr/bin/clang++')
    live = commands.add_parser('live-snapshot'); live.add_argument('--pseudo', default='Lecorbio')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == 'freeze': result = freeze(root, args.teacher)
    elif args.command == 'verify': result = verify(root)
    elif args.command == 'build': result = build(root, args.names, args.compiler)
    else: result = live_snapshot(root, args.pseudo)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
