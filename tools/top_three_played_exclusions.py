#!/usr/bin/env python3
"""Explicit current-campaign exposure census; exports hashes, never training rows."""
from __future__ import annotations
import argparse
import collections
import copy
import hashlib
import inspect
import json
from pathlib import Path
import resource
import sys

try:
    from . import top_three_experiments as e, top_three_development_banks as banks
    from . import top_three_collect_replays as public
except ImportError:
    import top_three_experiments as e
    import top_three_development_banks as banks
    import top_three_collect_replays as public

SCHEMA = 'papersoccer.top-three.played-exclusions.v1'
SCOPE_SCHEMA = 'papersoccer.top-three.played-exclusion-scope.v1'


class Census:
    def __init__(self, campaign_root):
        self.root = Path(campaign_root).resolve()
        self.inputs, self.states, self.features, self.groups = {}, set(), set(), set()
        self.cache, self.ledger = {}, []
        self.counts = collections.Counter()

    def read(self, reference, jsonl=False):
        path = Path(reference['path'])
        path = (e.ROOT / path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('exposure input escapes explicit current campaign')
        raw = path.read_bytes()
        if len(raw) > 256 * 1024**2 or hashlib.sha256(raw).hexdigest() != reference['sha256']:
            raise ValueError('exposure input hash/size mismatch')
        item = {'path': str(path), 'sha256': reference['sha256']}
        if str(path) in self.inputs and self.inputs[str(path)] != item:
            raise ValueError('one exposure path has conflicting identities')
        self.inputs[str(path)] = item
        if len(self.inputs) > 5000:
            raise ValueError('exposure input file bound exceeded')
        return ([json.loads(line) for line in raw.splitlines() if line.strip()] if jsonl else json.loads(raw)), item

    def observe(self, state):
        key = e.fingerprint(state)
        if key not in self.states:
            self.states.add(key)
            self.features.add(banks.feature_key(state))

    def trace(self, turns, reference, locator, kind, group=None):
        if group:
            self.groups.add(str(group))
        normalized = []
        for turn in turns:
            if isinstance(turn, str): normalized.append([None, turn])
            else: normalized.append([turn.get('player_id', turn.get('player')), turn['action']])
        key = hashlib.sha256(e.campaign.canonical(normalized)).hexdigest()
        if key not in self.cache:
            state = e.rules.ReplayState()
            self.observe(state)
            primitive, boundaries, reason = 0, 1, None
            for player, action in normalized:
                if state.winner is not None:
                    reason = 'turn-after-terminal'; break
                mover = state.to_move
                if player is not None and player != mover:
                    reason = 'wrong-player'; break
                for direction in action:
                    if direction not in '01234567' or state.winner is not None or state.to_move != mover:
                        reason = 'invalid-or-overlong-output'; break
                    try:
                        e.rules.apply_primitive(state, direction)
                    except ValueError:
                        reason = 'illegal-edge'; break
                    self.observe(state)
                    primitive += 1
                if reason: break
                if not action or state.winner is None and state.to_move == mover:
                    reason = 'empty-or-incomplete-turn'; break
                boundaries += 1
            self.cache[key] = {'primitive_rows': primitive + 1, 'boundary_rows': boundaries,
                               'rule_terminal': state.winner is not None, 'truncated_tail': reason}
        result = self.cache[key]
        self.counts[kind] += 1
        self.counts['trajectory_instances'] += 1
        self.counts['observed_primitive_rows'] += result['primitive_rows']
        self.counts['observed_boundary_rows'] += result['boundary_rows']
        self.counts['terminal_trajectory_instances' if result['rule_terminal'] else 'partial_trajectory_instances'] += 1
        self.counts['truncated_tail_instances'] += result['truncated_tail'] is not None
        self.ledger.append({'input': reference, 'locator': locator, 'kind': kind, 'trace_sha256': key, **result})
        if len(self.ledger) % 100 == 0:
            memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            memory = int(memory if sys.platform == 'darwin' else memory * 1024)
            if memory > 12 * 1024**3: raise MemoryError('exclusion census exceeded12GiB')
            print(json.dumps({'trajectories': len(self.ledger), 'unique_states': len(self.states), 'rss_bytes': memory}), flush=True)

    @staticmethod
    def split(transcript):
        return [] if transcript in ('', '-') else transcript.split('/')

    def native(self, node, reference, locator):
        game = node.get('game', node)
        if game is None: raise ValueError('native receipt has no observed game')
        if game.get('schema') != 'papersoccer.codingame-match.v1':
            raise ValueError('unexpected native referee schema')
        self.trace(game['actions'], reference, locator, 'native-game')

    def decision(self, row, reference, locator, default_prefix=None):
        prefix = row.get('prefix', default_prefix)
        if prefix is None:
            tokens = row.get('request', '').split()
            if len(tokens) != 3 or tokens[0] not in ('0', '1'):
                raise ValueError('decision omits a recoverable protocol prefix')
            prefix = '' if tokens[0] == '0' and tokens[2] == '-' else tokens[2]
        action = row.get('action')
        if action is None and row.get('stdout'):
            action = row['stdout'].splitlines()[0].split('\t')[0]
        turns = self.split(prefix)
        if action: turns.append(action)
        self.trace(turns, reference, locator, 'decision-probe')

    def entry(self, spec):
        kind = spec['kind']
        payload, reference = self.read(spec['input'], jsonl=kind.endswith('-jsonl'))
        node = payload
        for part in spec.get('pointer', '').split('/'):
            if part: node = node[int(part)] if isinstance(node, list) else node[part]
        if kind in ('experiment-result', 'pair-result', 'native-panel'):
            field, child = {'experiment-result': ('root_artifacts', 'experiment-root'),
                            'pair-result': ('pair_artifacts', 'pair-root'),
                            'native-panel': ('artifacts', 'native-game')}[kind]
            for item in node[field]: self.entry({'input': item, 'kind': child})
        elif kind in ('experiment-root', 'pair-root'):
            arms = node['arms'] if kind == 'experiment-root' else {key: node[key] for key in ('candidate', 'control')}
            for name, games in arms.items():
                for index, game in enumerate(games):
                    self.trace(self.split(game['transcript']), reference, f'{name}/{index}', 'development-game', node['root']['cluster_id'])
        elif kind == 'native-game': self.native(node, reference, spec.get('pointer', ''))
        elif kind == 'native-bundle':
            for index, game in enumerate(node['games']): self.native(game, reference, str(index))
        elif kind == 'opening-game':
            if 'actions' in node['game']: self.native(node, reference, '')
            else: self.trace(self.split(node['game']['transcript']), reference, '/game', 'opening-deviation-game')
        elif kind in ('transcripts', 'transcripts-jsonl'):
            values = node if isinstance(node, list) else [node]
            for index, row in enumerate(values):
                self.trace(self.split(row['transcript']), reference, str(index), spec.get('role', 'recorded-game'), row.get('root_group_id'))
        elif kind == 'arena-manifest':
            for index, item in enumerate(node['games']):
                child, child_ref = self.read({'path': item['record_path'], 'sha256': item['record_sha256']})
                if child != item['record']: raise ValueError('arena embedded record differs from its bound artifact')
                self.trace(child['replay']['observed_turns'], child_ref, '/replay/observed_turns', 'arena-game', 'arena:' + str(child['game_id']))
        elif kind == 'public-details':
            for item in node['dispositions']:
                raw_ref = {'path': str(Path(reference['path']).parent / 'raw' / (item['raw_sha256'] + '.json')), 'sha256': item['raw_sha256']}
                raw, raw_ref = self.read(raw_ref)
                if raw['gameId'] != item['game_id']: raise ValueError('public detail identity mismatch')
                turns = public.arena.parse_frames(raw['frames'], raw['gameId'])['turns']
                self.trace(turns, raw_ref, '/frames', 'public-development-detail', 'public:' + str(raw['gameId']))
        elif kind == 'decisions':
            for index, row in enumerate(node if isinstance(node, list) else [node]):
                self.decision(row, reference, str(index), spec.get('default_prefix'))
        elif kind == 'opening-probes-jsonl':
            for index, row in enumerate(node):
                if row.get('type') != 'position': continue
                self.trace(self.split(row['prefix']), reference, str(index), 'opening-probe-parent')
                for n, candidate in enumerate(row['candidates']):
                    turns = [*self.split(row['prefix']), candidate['action']]
                    self.trace(turns, reference, f'{index}/candidate/{n}', 'opening-probe-child')
                    for role in ('worst_alpha_reply', 'worst_compact_reply'):
                        reply = candidate.get(role)
                        if reply: self.trace([*turns, reply], reference, f'{index}/candidate/{n}/{role}', 'opening-probe-reply')
        elif kind == 'teacher-jsonl':
            for index, row in enumerate(node):
                group = row['group']
                prefix = [turn['action'] for turn in group['source_binding']['prefix']]
                self.trace(prefix, reference, f'{index}/parent', 'additional-supervised-parent')
                parent = e.state('/'.join(prefix))
                mover = parent.to_move
                for n, child in enumerate(group['successors']):
                    action = child['transcript']
                    if mover == 1: action = ''.join(str((int(c)+4) % 8) for c in action)
                    state = copy.deepcopy(parent)
                    e.rules.apply_complete_turn(state, state.to_move, action)
                    self.states.add(e.fingerprint(state))
                    self.features.add(banks.sha(b''.join(int(x).to_bytes(2, 'little') for x in min(tuple(child['active']), e.rules.reflect_active(child['active'])))))
                    self.counts['additional_supervised_successor_rows'] += 1
                self.ledger.append({'input': reference, 'locator': f'{index}/successors',
                    'kind': 'additional-supervised-successors', 'rows': len(group['successors'])})
        else:
            raise ValueError('unsupported explicit exposure parser: ' + kind)


def export(scope_path, expected, output):
    raw = Path(scope_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected: raise ValueError('scope manifest hash mismatch')
    scope = json.loads(raw)
    if scope.get('schema') != SCOPE_SCHEMA or scope.get('historical_protected_reads') is not False:
        raise ValueError('explicit current-campaign scope required')
    census = Census(scope['campaign_root'])
    scope_ref = e.raw_record(scope_path)
    census.read(scope_ref)
    source_files = {'exporter': Path(__file__), 'features': Path(e.rules.__file__),
                    'experiments': Path(e.__file__), 'banks': Path(banks.__file__), 'arena_parser': public.PARSER}
    sources = {name: e.raw_record(path) for name, path in source_files.items()}
    for spec in scope['entries']: census.entry(spec)
    for item in sources.values(): e.verify(item)
    output = Path(output)
    ledger_path = output.with_suffix('.ledger.jsonl')
    e.campaign.immutable(ledger_path, b''.join(banks.canonical(row) for row in census.ledger))
    value = {'schema': SCHEMA, 'campaign_root': str(census.root), 'complete_coverage': True,
             'scope_manifest': scope_ref, 'inputs': sorted(census.inputs.values(), key=lambda x: x['path']),
             'canonical_state_sha256': sorted(census.states), 'canonical_feature_sha256': sorted(census.features),
             'root_group_ids': sorted(census.groups), 'state_function_sha256': hashlib.sha256(inspect.getsource(e.fingerprint).encode()).hexdigest(),
             'sources': sources, 'includes_primitive_states': True, 'includes_turn_boundaries': True,
             'includes_terminal_and_valid_operational_prefixes': True,
             'rejected_action_policy': 'conservatively retain every legal primitive prefix; stop at the first invalid or post-terminal instruction',
             'counts': dict(census.counts), 'unique_transcripts': len(census.cache),
             'trajectory_ledger': e.raw_record(ledger_path), 'training_eligible': False,
             'historical_protected_reads': False, 'games_run': 0}
    e.emit(output, value)
    return {'inventory': e.raw_record(output), 'states': len(census.states), 'features': len(census.features),
            'counts': dict(census.counts), 'unique_transcripts': len(census.cache), 'inputs': len(census.inputs)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', type=Path, required=True)
    parser.add_argument('--scope-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.scope, args.scope_sha256, args.output), indent=2))


if __name__ == '__main__': main()
