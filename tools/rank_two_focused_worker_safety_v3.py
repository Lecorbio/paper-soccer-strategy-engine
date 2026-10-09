"""Isolated whole-response cold/warm safety for the reviewed runtime_v4."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from tools import rank_two_focused_campaign_v1 as campaign
from tools import rank_two_focused_runtime_v6 as runtime
from tools import rank_two_focused_work_v1 as work

SCHEMA=campaign.SCHEMA+'.runtime-v6-worker-safety.v3'


def run(plan_path):
    plan=campaign.read(plan_path)
    if plan['producer']!=campaign.record(__file__) or plan['runtime_producer']!=campaign.record(runtime.__file__):
        raise ValueError('worker safety/runtime producer changed')
    work.checked(plan,'experiment',launch=True)
    bundle=campaign.read(campaign.verify(plan['bundle']))
    if bundle['producer']!=plan['runtime_producer'] or bundle['sources']!=plan['sources']:
        raise ValueError('exact runtime/source binding differs')
    output=Path(plan['output']);output.mkdir(parents=True,exist_ok=True)
    reports=[];exposures=[]
    for name in plan['actors']:
        current=work.checked(plan,'experiment')
        actor=bundle['actors'][name];build=bundle['builds'][name]
        runtime.verify_build_receipt(build,actor,bundle['compiler'])
        if name.startswith('candidate:'):runtime.source_allowed(current,actor['source']['sha256'])
        directory=output/name.replace(':','-');directory.mkdir(exist_ok=True)
        if (directory/'CLAIM.json').exists():raise RuntimeError('spent worker safety claim retained')
        campaign.immutable(directory/'CLAIM.json',dict(plan=campaign.record(plan_path),actor=name,source=actor['source']))
        rows=[];failure=None;details=None
        def query(worker,case,cold_started=None):
            prefix=case['prefix'];state=runtime.e.state(prefix)
            counts=runtime.e.prior_decision_counts(prefix)
            first=counts[state.to_move]==0;budget=actor['clocks_ms'][0 if first else 1]
            external=1000 if first else 200
            limit=external if actor['response_policy']=='frozen-opponent-engine-budget' else budget
            action,ms=worker.choose(prefix,budget,external)
            total=ms if cold_started is None else 1000*(time.monotonic()-cold_started)
            row=dict(prefix=prefix,action=action,elapsed_ms=total,query_ms=ms,budget_ms=budget,
                     response_limit_ms=limit,response_policy=actor['response_policy'],cold=cold_started is not None,first=first,source=actor['source'])
            rows.append(row)
            campaign.atomic(directory/'PROGRESS.json',dict(actor=name,source=actor['source'],responses=rows,complete=False))
            if total>limit:raise runtime.retained.WorkerResponseFailure('cold worker envelope exceeded',dict(kind='cold_worker_envelope_exceeded',**row))
            runtime.e.rules.apply_complete_turn(state,state.to_move,action)
        try:
            for case in plan['cases']:
                worker=None
                try:
                    current=work.checked(plan,'experiment')
                    if name.startswith('candidate:'):runtime.source_allowed(current,actor['source']['sha256'])
                    started=time.monotonic();worker=runtime.Worker(build);query(worker,case,started)
                finally:
                    if worker:worker.close()
            worker=None
            try:
                worker=runtime.Worker(build)
                for case in plan['cases']:
                    current=work.checked(plan,'experiment')
                    if name.startswith('candidate:'):runtime.source_allowed(current,actor['source']['sha256'])
                    query(worker,case)
            finally:
                if worker:worker.close()
        except (RuntimeError,ValueError,OSError,BrokenPipeError) as error:
            failure=type(error).__name__+': '+str(error);details=getattr(error,'details',None)
        result=dict(schema=SCHEMA+'.actor',passed=failure is None,actor=name,source=actor['source'],
            binary=build['binary'],runtime=plan['bundle'],responses=rows,failure=failure,failure_details=details,
            cold_max_ms=max((row['elapsed_ms'] for row in rows if row['cold']),default=0),
            warm_max_ms=max((row['elapsed_ms'] for row in rows if not row['cold']),default=0),
            arbitrary_later_root_covered=any(row['cold'] and not row['first'] for row in rows),
            long_prefix_covered=any(len(row['prefix'].replace('/',''))>=40 for row in rows),
            strength_eligible=False,training_eligible=False)
        reference=campaign.immutable(directory/'RESULT.json',result)
        exposures.append(reference);reports.append(dict(actor=name,receipt=reference,passed=result['passed']))
        if failure:
            campaign.immutable(output/'EXPOSURE_PENDING.json',dict(schema=SCHEMA+'.exposure',cases=plan['cases'],
                report=reference,recipe=campaign.record(__file__),before_next_fresh_bank=True,training_eligible=False))
            campaign.immutable(output/'RESULT.json',dict(schema=SCHEMA,passed=False,actors=reports,
                failure_actor=name,source=actor['source'],own_failure=name.startswith('candidate:'),
                plan=campaign.record(plan_path),runtime=plan['bundle'],new_games=0))
            return False
    result=dict(schema=SCHEMA,passed=True,actors=reports,runtime=plan['bundle'],plan=campaign.record(plan_path),
        sources=plan['sources'],isolated=True,own_failures=0,whole_response_cold_warm_long_roots=True,new_games=0)
    campaign.immutable(output/'RESULT.json',result)
    campaign.immutable(output/'EXPOSURE_PENDING.json',dict(schema=SCHEMA+'.exposure',cases=plan['cases'],
        inputs=exposures,recipe=campaign.record(__file__),before_next_fresh_bank=True,training_eligible=False))
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(dict(passed=run(args.plan))))


if __name__=='__main__':main()
