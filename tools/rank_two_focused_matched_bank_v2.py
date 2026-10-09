"""Freeze one fresh shared panel, retaining every bounded proposal and partial."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3

from tools import rank_two_focused_campaign_v1 as campaign
from tools import rank_two_focused_runtime_v6 as runtime
from tools import rank_two_focused_work_v1 as work
from tools import top_three_development_banks as proposals

SCHEMA = campaign.SCHEMA + '.matched-bank.v2'


def run(plan_path):
    plan = campaign.read(plan_path)
    if plan['producer'] != campaign.record(__file__):
        raise ValueError('matched bank producer differs')
    current = work.checked(plan, 'prepare-bank', launch=True)
    if current['pending_exposures'] or current['focused_ancestry_ready'] != plan['ancestry']:
        raise PermissionError('close exact predecessor exposures before fresh bank')
    for reference in plan['inputs'].values():
        campaign.verify(reference)
    if (plan['seed'] != 2026100711 or plan['roots'] != 256 or plan['drawn_edges'] != 10
            or plan['original_declared_games'] != 2560 or plan['execution_games'] != 2048
            or plan['canceled_unclaimed_games'] != 512 or plan['selection']['superiority_claim_allowed']):
        raise ValueError('immutable original panel and source-ban exception differ')
    bundle = campaign.read(campaign.verify(plan['runtime']))
    frozen = campaign.read(campaign.verify(plan['sources']))
    if len(frozen['models']) != 4 or len(plan['candidates']) != 3:
        raise ValueError('four frozen originals and three unbanned sources required')
    if plan['candidates'] != sorted(name for name in bundle['actors'] if name.startswith('candidate:')):
        raise ValueError('panel candidate roster differs')
    for name in plan['candidates']:
        runtime.source_allowed(current, bundle['actors'][name]['source']['sha256'])
    excluded = bundle['excluded_declared_models']
    if len(excluded) != 1 or excluded[0]['model']['source'] != plan['excluded_source']:
        raise ValueError('permanent source-ban exclusion differs')
    ready = campaign.read(campaign.verify(plan['ancestry']))
    prior = campaign.read(campaign.verify(ready['prior']))
    database = campaign.verify(prior['inputs']['database'])
    output = Path(plan['output'])
    output.mkdir(parents=True, exist_ok=True)
    if (output/'CLAIM.json').exists():
        raise RuntimeError('spent bank claim retained; no automatic replay')
    campaign.immutable(output/'CLAIM.json', dict(plan=campaign.record(plan_path), ancestry=plan['ancestry']))
    campaign.immutable(output/'EXPOSURE_PENDING.json', dict(schema=SCHEMA+'.exposure',
        generation_plan=campaign.record(plan_path), proposal_receipts=campaign.record(plan_path),
        before_next_fresh_bank=True, training_eligible=False, new_games=0))
    previous=campaign.read(campaign.verify(plan['previous_proposals']))
    floors=[0]*256
    for reference in previous['proposal_receipts']:
        prior=campaign.read(campaign.verify(reference))
        floors[prior['slot']]=max(floors[prior['slot']],prior['attempt']+1)
    if floors!=plan['attempt_floors']:raise ValueError('fixed-seed proposal continuation differs')
    rows, seen_states, seen_features, drawn = [], set(), set(), 0
    with sqlite3.connect(database.as_uri()+'?mode=ro', uri=True) as db:
        db.execute('PRAGMA cache_size=-16384')
        for index in range(plan['roots']):
            work.checked(plan, 'experiment')
            slot = dict(opponent=plan['opponents'][index % 8], drawn_edges=10,
                        physical_mover=(index//8) % 2, depth_slot=index)
            for attempt in range(floors[index],floors[index]+plan['maximum_per_slot']):
                if drawn >= plan['maximum_proposals']:
                    raise RuntimeError('bounded matched bank proposal cap reached; retain partials')
                seed = proposals.proposal_seed(plan['seed'], plan['proposal_stage'], slot, attempt)
                directory = output/'proposals'/f'{drawn:06}'
                campaign.immutable(directory/'CLAIM.json', dict(proposal_seed=seed, drawn_edges=10,
                                                              slot=index, attempt=attempt))
                position, prefix, error = proposals.propose(seed, 10)
                drawn += 1
                record = dict(proposal_seed=seed, drawn_edges=10, slot=index, attempt=attempt,
                              status=error or 'complete', training_eligible=False)
                accepted = False
                if error is None:
                    state = runtime.e.fingerprint(position)
                    feature = proposals.feature_key(position)
                    overlap = (state in seen_states or feature in seen_features or any(
                        db.execute('SELECT 1 FROM keys WHERE category=? AND value=?',
                                   (category,key)).fetchone() is not None
                        for category,key in (('states',state),('features',feature))))
                    accepted = position.to_move == slot['physical_mover'] and not overlap
                    record.update(prefix=prefix, state_sha256=state, feature_sha256=feature,
                        accepted=accepted, rejection='ancestry-overlap' if overlap else
                        ('mover-mismatch' if not accepted else None))
                campaign.immutable(directory/'RESULT.json', record)
                if accepted:
                    if runtime.e.fingerprint(runtime.e.state(prefix)) != state:
                        raise ValueError('proposal and complete-turn reconstruction differ')
                    seen_states.add(state); seen_features.add(feature)
                    rows.append(dict(index=index, transcript=prefix, state_sha256=state,
                        feature_sha256=feature, cluster_id=f'focused-match-v2-{index:04}',
                        opponent=slot['opponent'], physical_mover=slot['physical_mover'],
                        drawn_edges=10, colors=[0,1], proposal=campaign.record(directory/'RESULT.json')))
                    break
            else:
                raise RuntimeError('matched root draw exhausted; no substitute seed or depth')
            campaign.atomic(output/'PROGRESS.json', dict(completed_roots=len(rows), proposals=drawn))
    if len(rows) != 256 or len(seen_states) != 256 or len(seen_features) != 256:
        raise ValueError('independent canonical/feature roots incomplete')
    bank = dict(schema=SCHEMA, passed=True, plan=campaign.record(plan_path), rows=rows,
        seed=plan['seed'], sources=plan['sources'], runtime=plan['runtime'],
        ancestry=plan['ancestry'], candidates=plan['candidates'],
        control_reused_once_per_root_color=True, bootstrap=plan['selection'],
        previous_quarantined_bank=plan['previous_bank'],fixed_seed_stream_continued=True,
        original_declared_games=2560, execution_games=2048, canceled_unclaimed_games=512,
        excluded_source=plan['excluded_source'], new_games=0, training_eligible=False)
    campaign.immutable(output/'BANK.json', bank)
    proposal_refs=[campaign.record(p) for p in sorted((output/'proposals').glob('*/RESULT.json'))]
    campaign.immutable(output/'COMPLETED_EXPOSURE.json', dict(schema=SCHEMA+'.completed-exposure',
        bank=campaign.record(output/'BANK.json'), proposal_receipts=proposal_refs,
        before_next_fresh_bank=True, training_eligible=False, new_games=0))
    result=dict(schema=SCHEMA+'.result', passed=True, roots=256, proposals=drawn,
        bank=campaign.record(output/'BANK.json'), plan=campaign.record(plan_path), new_games=0,
        all_proposals_and_partials_retained=True, ancestry_rescan=False, protected_reads=False)
    campaign.immutable(output/'RESULT.json', result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();result=run(args.plan)
    print(json.dumps(dict(passed=result['passed'],roots=result['roots'],proposals=result['proposals'])))


if __name__ == '__main__':
    main()
