"""Declared, isolated production-main timing on the frozen matched sources."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from tools import rank_two_focused_campaign_v1 as campaign
from tools import rank_two_focused_coverage_v1 as coverage
from tools import rank_two_focused_work_v1 as work

SCHEMA = campaign.SCHEMA + '.matched-production-safety.v2'


def checked(plan_path, action, launch=True):
    plan = campaign.read(plan_path)
    if (plan['producer'] != campaign.record(__file__)
            or plan['game_producer'] != campaign.record(coverage.__file__)):
        raise ValueError('bound production timing producer changed')
    work.checked(plan,action,launch=launch)
    frozen = campaign.read(campaign.verify(plan['sources']))
    if frozen['models'] != plan['models']:
        raise ValueError('the four source identities differ from the timing declaration')
    return plan


def build(plan_path):
    plan = checked(plan_path,'experiment')
    output = Path(plan['output'])
    output.mkdir(parents=True,exist_ok=True)
    builds = {}
    for model in plan['models']:
        work.checked(plan,'experiment')
        tag = model['profile'] + '-seed-' + str(model['seed'])
        directory = output/tag
        directory.mkdir(exist_ok=True)
        source = campaign.verify(model['source'])
        binary = directory/'main'
        command = ['/usr/bin/clang++','-std=c++20','-O3','-ffp-contract=off',str(source),'-o',str(binary)]
        result = subprocess.run(command,capture_output=True,text=True,timeout=120)
        campaign.immutable(directory/'compiler.stderr',result.stderr.encode())
        if result.returncode:
            raise ValueError('unchanged production source compilation failed')
        builds[tag] = dict(source=model['source'],runtime=model['runtime'],
                           binary=campaign.record(binary),flags=command[1:4])
    report = dict(schema=SCHEMA+'.build',passed=True,compiled=True,builds=builds,
                  source_modified=False,sources=plan['sources'],plan=campaign.record(plan_path),
                  compiler=campaign.record('/usr/bin/clang++'),new_games=0)
    campaign.immutable(output/'BUILD.json',report)
    return report


def certify(plan_path):
    plan = checked(plan_path,'experiment')
    builds = campaign.read(campaign.verify(plan['build']))
    if builds['sources'] != plan['sources'] or builds.get('source_modified') is not False:
        raise ValueError('timing binaries are not the unchanged frozen sources')
    if plan['clocks_ms'] != [550,140] or plan['direction_offsets'] != [0,3]:
        raise ValueError('declared matched clocks/coverage changed')
    output = Path(plan['output'])
    output.mkdir(parents=True,exist_ok=True)
    reports, all_receipts = [], []
    for model in plan['models']:
        tag = model['profile'] + '-seed-' + str(model['seed'])
        entry = builds['builds'][tag]
        if entry['source'] != model['source'] or entry['runtime'] != model['runtime']:
            raise ValueError('source/binary/runtime identity differs')
        campaign.verify(entry['source'])
        campaign.verify(entry['binary'])
        games = []
        for color in (0,1):
            for offset in plan['direction_offsets']:
                work.checked(plan,'experiment')
                directory = output/tag/('p'+str(color)+'-offset'+str(offset))
                # The frozen generator constructs color one's first input
                # before launch; the measured envelope contains bot work only.
                result = coverage.game(entry,color,offset,directory)
                ref = campaign.record(directory/'RESULT.json')
                if not result['complete'] or result.get('failure'):
                    raise ValueError('production-main mechanical game failed')
                games.append(dict(color=color,offset=offset,receipt=ref,
                    first_max_ms=result['first_max_ms'],later_max_ms=result['later_max_ms'],
                    longest_response_edges=result['longest_response_edges']))
                all_receipts.append(ref)
        longest = max(row['longest_response_edges'] for row in games)
        directory = output/tag
        proof = dict(schema=SCHEMA+'.whole-response',passed=longest>=8,source=model['source'],
            runtime=model['runtime'],build=campaign.record(Path(plan['build']['path'])),
            plan=campaign.record(plan_path),clocks_ms=[550,140],own_failures=0,isolated=True,
            games=games,complete_games=len(games),cold_warm_both_colors=True,
            long_turn_observed=longest>=8,longest_response_edges=longest,
            source_modified=False,playing_strength_qualified=False)
        reference = campaign.immutable(directory/'whole_response.json',proof)
        reports.append(dict(profile=model['profile'],seed=model['seed'],source=model['source'],
                            runtime=model['runtime'],whole_response=reference,passed=proof['passed']))
    passed = all(row['passed'] for row in reports)
    result = dict(schema=SCHEMA,passed=passed,models=reports,sources=plan['sources'],
        plan=campaign.record(plan_path),isolated=True,mechanical_games=len(all_receipts),own_failures=0,
        whole_response_envelopes_ms=[550,140],playing_strength_qualified=False)
    campaign.immutable(output/'RESULT.json',result)
    campaign.immutable(output/'EXPOSURE_PENDING.json',dict(schema=SCHEMA+'.exposure',
        report=campaign.record(output/'RESULT.json'),game_receipts=all_receipts,
        recipe=campaign.record(__file__),before_next_fresh_bank=True,training_eligible=False))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['build','certify'])
    parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args()
    result=build(args.plan) if args.mode=='build' else certify(args.plan)
    print(json.dumps(dict(passed=result['passed'],mode=args.mode)))


if __name__=='__main__':
    main()
