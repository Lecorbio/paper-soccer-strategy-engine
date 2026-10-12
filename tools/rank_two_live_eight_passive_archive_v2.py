"""Resume a classified frozen exclusion archive without replacing prior receipts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tools import rank_two_live_eight_v1 as c
from tools import rank_two_focused_work_v1 as work
from tools import top_three_collect_replays as retained


class FrozenWindowCollector(retained.arena.ArenaBatchCollector):
    def __init__(self, plan, snapshot, **kwargs):
        super().__init__(**kwargs)
        self.plan = plan
        self.snapshot = snapshot
        self.allowed_ids = set(plan['game_ids'])

    def _fetch(self, *, run_id, request_schema, payload):
        work.checked(self.plan, 'archive')
        if request_schema in ('leaderboard-v1', 'agent-battles-v1'):
            key = 'leaderboard_receipt' if request_schema == 'leaderboard-v1' else 'battles_receipt'
            receipt = self.snapshot[key]
            if receipt['request']['body'] != payload:
                raise ValueError('frozen metadata request differs')
            raw = Path(self.plan['snapshot']['path']).parent / receipt['raw_path']
            content = raw.read_bytes()
            if hashlib.sha256(content).hexdigest() != receipt['raw_sha256']:
                raise ValueError('frozen raw metadata hash differs')
            response = retained.shared.ApiResponse(body=content, status=receipt['status'], attempts=receipt['attempts'])
            return self._store_response(run_id=run_id, request_schema=request_schema,
                payload=payload, response=response, fetched_at=receipt['fetched_at_utc'])
        if (request_schema != 'game-detail-v1' or not isinstance(payload, list) or len(payload) != 2
                or payload[1] is not None or type(payload[0]) is not int or payload[0] not in self.allowed_ids):
            raise PermissionError('only the frozen exclusion games may be fetched')
        return super()._fetch(run_id=run_id, request_schema=request_schema, payload=payload)


def run(plan_path):
    plan = c.read(plan_path)
    work.checked(plan, 'archive', launch=True)
    c.resource_guard(plan['resource_budget'])
    if plan['producer'] != c.record(__file__) or plan['training_eligible'] is not False:
        raise PermissionError('exclusion-only producer binding differs')
    for ref in plan['inputs'].values():
        c.verify(ref)
    snapshot = c.read(c.verify(plan['snapshot']))
    if snapshot['user']['agentId'] != plan['agent_id'] or snapshot['user'].get('percentage') != 100:
        raise ValueError('frozen incumbent agent is not complete')
    if len(plan['game_ids']) != len(set(plan['game_ids'])) or set(plan['game_ids']) != {g['gameId'] for g in snapshot['battles']}:
        raise ValueError('frozen later-game frontier differs')
    for game in snapshot['battles']:
        focus = [p for p in game['players'] if p.get('playerAgentId') == plan['agent_id']]
        if not game.get('done') or len(focus) != 1 or focus[0]['submissionId'] != plan['submission_id']:
            raise ValueError('frozen game/submission identity differs')
    out = Path(plan['output'])
    if (out / 'CLAIM.json').exists():
        raise RuntimeError('spent archive claim retained; classify before retrying')
    c.immutable(out / 'CLAIM.json', dict(plan=c.record(plan_path), game_ids=plan['game_ids'], new_submissions=0))
    predecessor = c.read(c.verify(plan['predecessor_plan']))
    classification = c.read(c.verify(plan['classification']))
    if classification.get('kind') != 'archive-resource-thread-limit' or classification.get('source_operational_failure') is not False:
        raise PermissionError('explicit non-source resource classification required')
    for key in ('source', 'snapshot', 'registry', 'game_ids', 'agent_id', 'submission_id', 'run_id'):
        if plan[key] != predecessor[key]:
            raise PermissionError('recovery changes the frozen archive frontier')
    prior_out = Path(predecessor['output'])
    claim = c.read(prior_out / 'CLAIM.json')
    if claim['plan'] != plan['predecessor_plan'] or claim['game_ids'] != plan['game_ids']:
        raise PermissionError('predecessor archival claim differs')
    collector = FrozenWindowCollector(plan, snapshot, repository=c.ROOT, data_root=prior_out / 'archive',
        maximum_workers=4, api=retained.shared.PublicApi(maximum_attempts=4),
        exclusion_registry_path=c.verify(plan['registry']), exclusion_registry_sha256=plan['registry']['sha256'])
    binding = collector.bind_source(agent_id=plan['agent_id'], submission_id=plan['submission_id'],
        source_path=c.verify(plan['source']), expected_source_sha256=c.BASELINE)
    collected = collector.collect(run_id=plan['run_id'], binding=binding, expected_games=len(plan['game_ids']))
    manifest_path = Path(collected['manifest_path'])
    if not manifest_path.is_absolute():
        manifest_path = c.ROOT / manifest_path
    manifest = retained.arena.validate_export_manifest(manifest_path, plan['registry']['sha256'])
    accepted = [row['record'] for row in manifest['games'] if row['record']['status'] == 'accepted']
    if not manifest['coverage']['full_window_accounted'] or {row['game_id'] for row in accepted} != set(plan['game_ids']):
        raise ValueError('every frozen exclusion game must be retained and classified')
    exposure = c.immutable(out / 'EXPOSURE.json', dict(schema=c.SCHEMA+'.played-exposure',
        games=[dict(game_id=row['game_id'], turns=row['replay']['valid_turns']) for row in accepted],
        training_eligible=False, source=plan['source'], manifest=c.record(manifest_path)))
    result = c.immutable(out / 'RESULT.json', dict(schema=c.SCHEMA+'.passive-incumbent-archive', passed=True,
        plan=c.record(plan_path), snapshot=plan['snapshot'], manifest=c.record(manifest_path), exposure=exposure,
        games=len(accepted), game_ids=plan['game_ids'], new_submissions=0, training_eligible=False,
        own_failures=sum(row['operational']['focus_status'] != 'ok' for row in accepted),
        all_operational_classifications_retained=True, frozen_metadata_used=True))
    with c.locked(plan['base']):
        state = c.read(Path(plan['base']) / 'CURRENT.json')
        if exposure not in state['pending_exposures']:
            state['pending_exposures'].append(exposure)
        state['live_eight_pre_precision_passive_archive_result'] = result
        c.atomic(Path(plan['base']) / 'CURRENT.json', state)
    return dict(passed=True, games=len(accepted), result=result, exposure=exposure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    print(json.dumps(run(parser.parse_args().plan)))
