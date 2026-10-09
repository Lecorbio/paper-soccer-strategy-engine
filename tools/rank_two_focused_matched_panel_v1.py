"""Serial, source-bound shared panel with durable game and cancellation slots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools import rank_two_focused_campaign_v1 as campaign
from tools import rank_two_focused_runtime_v5 as runtime
from tools import rank_two_focused_work_v1 as work

SCHEMA = campaign.SCHEMA + '.matched-panel.v1'


def binding_check(plan):
    current = work.checked(plan, 'experiment')
    if current['focused_runtime_v5_bundle'] != plan['runtime']:
        raise PermissionError('reviewed runtime binding changed')
    for source in plan['candidate_sources']:
        runtime.source_allowed(current, source['sha256'])
    return current


def run(plan_path):
    plan = campaign.read(plan_path)
    if plan['producer'] != campaign.record(__file__):
        raise ValueError('panel producer changed')
    work.checked(plan, 'experiment', launch=True)
    binding_check(plan)
    for reference in plan['inputs'].values():
        campaign.verify(reference)
    bank = campaign.read(campaign.verify(plan['bank']))
    bundle = campaign.read(campaign.verify(plan['runtime']))
    if (bank['runtime'] != plan['runtime'] or bank['candidates'] != plan['candidates']
            or len(bank['rows']) != 256 or bank['execution_games'] != 2048
            or bank['canceled_unclaimed_games'] != 512 or plan['games'] != 2048
            or plan['source_clocks_ms'] != [550,140]):
        raise ValueError('exact shared bank/source/runtime/clock binding differs')
    output = Path(plan['output'])
    identity = campaign.record(plan_path)
    completed, root_receipts = 0, []
    all_game_refs = []
    for row in bank['rows']:
        binding_check(plan)
        if runtime.e.fingerprint(runtime.e.state(row['transcript'])) != row['state_sha256']:
            raise ValueError('canonical panel root reconstruction differs')
        arms = ['control',*plan['candidates']]
        offset = row['index'] % len(arms)
        order = arms[offset:] + arms[:offset]
        root_dir = output/'roots'/f'{row["index"]:04}'
        arm_refs = {}
        for name in order:
            arm_refs[name] = []
            actor = bundle['actors'][name]
            opponent_name = 'opponent:'+row['opponent']
            opponent = bundle['actors'][opponent_name]
            actor_build = bundle['builds'][name]
            opponent_build = bundle['builds'][opponent_name]
            runtime.verify_build_receipt(actor_build, actor, bundle['compiler'])
            runtime.verify_build_receipt(opponent_build, opponent, bundle['compiler'])
            for color in (0,1):
                binding_check(plan)
                directory = root_dir/name.replace(':','-')/f'p{color}'
                claim = dict(plan=identity, row=row, actor=name, source=actor['source'],
                             opponent=opponent, color=color, runtime=plan['runtime'])
                if (directory/'RESULT.json').exists():
                    previous = campaign.read(directory/'RESULT.json')
                    if previous['claim'] != claim or previous['game']['failure'] is not None:
                        raise ValueError('retained game differs or failed; never replay')
                else:
                    if (directory/'CLAIM.json').exists():
                        raise RuntimeError('spent unknown game claim retained; no automatic replay')
                    campaign.immutable(directory/'CLAIM.json', claim)
                    game = runtime.play(row,color,actor,opponent,actor_build,opponent_build)
                    campaign.immutable(directory/'RESULT.json', dict(schema=SCHEMA+'.game',claim=claim,game=game))
                    if game['failure'] is not None:
                        failure = dict(schema=SCHEMA+'.failure', passed=False,
                            failed_game=campaign.record(directory/'RESULT.json'), plan=identity,
                            source=actor['source'], details=game['failure_details'],
                            own_failure=name.startswith('candidate:') and
                            game['failure_details'].get('role')=='candidate',
                            completed_games=completed, unknown_claims_preserved=True,
                            all_unclaimed_slots_must_be_canceled=True)
                        campaign.immutable(output/'RESULT.json',failure)
                        campaign.immutable(output/'EXPOSURE_PENDING.json',dict(schema=SCHEMA+'.failed-exposure',
                            bank=plan['bank'],report=campaign.record(output/'RESULT.json'),
                            game_receipts=[*all_game_refs,campaign.record(directory/'RESULT.json')],
                            before_next_fresh_bank=True,training_eligible=False))
                        return failure
                reference = campaign.record(directory/'RESULT.json')
                arm_refs[name].append(reference);all_game_refs.append(reference);completed += 1
                campaign.atomic(output/'PROGRESS.json',dict(completed_games=completed,
                    completed_roots=len(root_receipts),declared_games=2048,
                    current_root=row['index'],last_game=reference))
        cancellation = dict(status='canceled-unclaimed', source=bank['excluded_source'],
            reason='permanent own-failure exact-source/semantic ban before shared panel dispatch',
            slots=[dict(root=row['cluster_id'],color=color) for color in (0,1)],
            observed_games=0,substitutions=0)
        root_receipts.append(campaign.immutable(root_dir/'RESULT.json',dict(schema=SCHEMA+'.root',
            plan=identity,row=row,execution_order=order,arms=arm_refs,canceled=cancellation,
            control_reused_once_per_root_color=True)))
    if completed != 2048 or len(root_receipts) != 256:
        raise ValueError('declared panel is incomplete')
    result=dict(schema=SCHEMA,passed=True,plan=identity,bank=plan['bank'],runtime=plan['runtime'],
        completed_games=2048,canceled_unclaimed_slots=512,original_declared_slots=2560,
        root_receipts=root_receipts,source_bans_preserved=True,own_failures=0,
        all_declared_games_archived=True,family_locked=False,superiority_claim=False,
        evaluation_recipe=plan['evaluation_recipe'],training_eligible=False)
    campaign.immutable(output/'RESULT.json',result)
    campaign.immutable(output/'EXPOSURE_PENDING.json',dict(schema=SCHEMA+'.completed-exposure',
        bank=plan['bank'],report=campaign.record(output/'RESULT.json'),game_receipts=all_game_refs,
        before_next_fresh_bank=True,training_eligible=False))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();result=run(args.plan)
    print(json.dumps(dict(passed=result['passed'],completed_games=result['completed_games'])))


if __name__ == '__main__':
    main()
