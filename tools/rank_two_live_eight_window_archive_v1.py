"""Queue-bound archival of the full declared live window and later exposures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools import rank_two_live_eight_v1 as c
from tools import rank_two_live_eight_windows_v1 as windows
from tools import rank_two_focused_work_v1 as work


def run(plan_path):
    plan = c.read(plan_path)
    state = work.checked(plan, 'archive', launch=True)
    c.resource_guard(plan['resource_budget'])
    if plan['producer'] != c.record(__file__):
        raise PermissionError('window archive producer changed')
    for reference in plan['inputs'].values():
        c.verify(reference)
    declared = state['live_eight_declared_windows'][plan['attempt_id']]
    if declared['source_sha256'] != plan['source']['sha256'] or not declared.get('attested'):
        raise PermissionError('exact attested source/window required')
    result = windows.close(Path(plan['base']), plan['attempt_id'])
    if not result['complete'] or len(result['games']) != 90:
        raise ValueError('every declared chronological game must be retained')
    root = Path(plan['base']) / 'live-eight-v1/attempts' / plan['attempt_id']
    receipt = c.immutable(Path(plan['output']) / 'RESULT.json', dict(
        schema=c.SCHEMA+'.window-archive-completion', passed=True,
        plan=c.record(plan_path), result=c.record(root/'result.json'),
        exposure=c.record(root/'EXPOSURE.json'), manifest=result['manifest'],
        declared_games=len(result['games']), supplemental_games=len(result['supplemental_games']),
        own_failures=result['own_failures'], opponent_failures=sum(g['opponent_failure'] for g in result['games']),
        completed_rank=result['completed_rank'], score=result['score'],
        all_operational_classifications_retained=True, playing_strength_qualified=False))
    return dict(passed=True, receipt=receipt, own_failures=result['own_failures'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    print(json.dumps(run(parser.parse_args().plan)))
