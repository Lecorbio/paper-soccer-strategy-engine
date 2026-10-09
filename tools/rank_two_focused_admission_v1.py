"""Exact-source mechanical evidence for the four frozen matched networks."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

from tools import rank_two_focused_campaign_v1 as campaign
from tools import rank_two_focused_export_v2 as exporter
from tools import rank_two_focused_native_v3 as native
from tools import rank_two_focused_training_v5 as training
from tools import rank_two_focused_work_v1 as work
from tools import jacek_replay_features as rules
from tools import top_three_experiments as experiments

np = training.np
SCHEMA = campaign.SCHEMA + '.source-mechanics.v1'


def tensors(runtime):
    values, _ = exporter.validate(runtime)
    shape = runtime['architecture']
    architecture = training.Architecture(runtime['profile'], shape[1], shape[2])
    flat = np.asarray(values, dtype=np.int8)
    first, second = shape[0] * shape[1], shape[0] * shape[1] + shape[1] * shape[2]
    integer = dict(w1=flat[:first].reshape(shape[0],shape[1]),
                   w2=flat[first:second].reshape(shape[1],shape[2]),w3=flat[second:])
    quantized = training.core.QuantizedWeights(integer,
        {name:np.float32(scale) for name,scale in zip(('w1','w2','w3'),runtime['scales'],strict=True)})
    return architecture, quantized, values


def proof(directory, name, source, **values):
    return campaign.immutable(directory / (name + '.json'),
        dict(schema=SCHEMA + '.' + name, passed=True, source=source, **values))


def run(plan_path):
    plan = campaign.read(plan_path)
    if plan['producer'] != campaign.record(__file__):
        raise ValueError('mechanical verification producer changed')
    work.checked(plan, 'experiment', launch=True)
    frozen = campaign.read(campaign.verify(plan['sources']))
    examples = campaign.read(campaign.verify(plan['cases']))['cases']
    if frozen['models'] != plan['models']:
        raise ValueError('the four frozen sources differ from the declaration')
    if sorted({case['mover'] for case in examples}) != [0,1]:
        raise ValueError('both player orientations required')
    nonterminal = [row for row in examples if not row['terminal']]
    canonical = sorted({experiments.fingerprint(experiments.state(row['prefix'])) for row in nonterminal})
    if canonical != plan['canonical_state_sha256'] or len(canonical) < 2:
        raise ValueError('predeclared canonical states differ')
    output = Path(plan['output'])
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for model in plan['models']:
        work.checked(plan, 'experiment')
        tag = model['profile'] + '-seed-' + str(model['seed'])
        directory = output / tag
        directory.mkdir(exist_ok=True)
        if (directory / 'CLAIM.json').exists():
            raise RuntimeError('spent verification claim retained; classify before repeating')
        for name in ('source', 'runtime', 'export'):
            campaign.verify(model[name])
        campaign.immutable(directory / 'CLAIM.json',dict(plan=campaign.record(plan_path),source=model['source']))
        source = Path(model['source']['path']).read_text()
        runtime, report = campaign.read(model['runtime']['path']), campaign.read(model['export']['path'])
        if report['source_sha256'] != model['source']['sha256'] or not report['nonpayload_target_passed'] or not report['below_hard_limit']:
            raise ValueError('exact source size/identity differs')
        regenerated, token_report = exporter.export(runtime, exporter.macro_names(), verify_tokens=True)
        if regenerated != source:
            raise ValueError('source regenerated from frozen tensor differs')
        campaign.immutable(directory / 'TOKEN_PROOFS.json',token_report)
        normal = native.build(source,report,directory/'normal')
        sanitized = native.build(source,report,directory/'sanitized',sanitize=True)
        architecture, quantized, values = tensors(runtime)
        decoded = list(map(int,native.query(normal,['weights'])[0].split()))
        if decoded != values:
            raise ValueError('independent native decoded tensor differs')
        active = [np.asarray(row['active'],dtype=np.intp) for row in nonterminal]
        expected,_ = training.forward(quantized.effective(),architecture,active,quantized)
        actual = np.asarray([float(row) for row in native.query(normal,
            ['eval '+str(len(row))+' '+' '.join(map(str,row)) for row in active])],dtype=np.float32)
        if not np.allclose(expected,actual,rtol=0,atol=1e-6):
            raise ValueError('native tensor/inference parity differs')
        features = native.query(normal,['features '+row['prefix'] for row in nonterminal])
        if [list(map(int,row.split())) for row in features] != [row['active'] for row in nonterminal]:
            raise ValueError('native mover-relative feature parity differs')
        requests = ['state '+row['prefix'] for row in examples]
        states = native.query(normal,requests)
        if native.query(sanitized,requests) != states:
            raise ValueError('sanitized state/make-unmake differs')
        for case, answer in zip(examples,states,strict=True):
            if answer.startswith('ERROR'):
                raise ValueError('native rule or make/unmake check failed')
            mover,terminal,_ = map(int,answer.split())
            if mover != case['mover'] or terminal != int(case['terminal']):
                raise ValueError('native rule labels differ')
        transport = []
        for length in range(1,18):
            raw = bytes(range(length))
            encoded = base64.b85encode(raw,pad=True).decode()
            answer = native.query(sanitized,['decode '+str(length)+' '+encoded])[0]
            if list(map(int,answer.split())) != list(raw):
                raise ValueError('independent Base85 tail decoder differs')
            transport.append(length)
        errors = native.query(sanitized,['decode 4 ~~~~~','decode 4 0000"','decode 1 VPa!s'])
        if any(not row.startswith('ERROR') for row in errors):
            raise ValueError('malformed transport accepted')
        search_requests = ['search '+str(nodes)+' '+case['prefix']
            for nodes in plan['search_nodes'] for case in nonterminal[:plan['search_case_count']]]
        chosen = native.query(normal,search_requests)
        if native.query(sanitized,search_requests) != chosen:
            raise ValueError('sanitized search/proof/fallback differs')
        search_cases = []
        for request,answer in zip(search_requests,chosen,strict=True):
            if answer.startswith('ERROR'):
                raise ValueError('bounded search or fallback failed')
            prefix = request.split(' ',2)[2]
            state = experiments.state(prefix)
            action = answer.split()[0]
            rules.apply_complete_turn(state,state.to_move,action)
            search_cases.append(dict(prefix=prefix,action=action,statistics=answer,request=request))
        original,_,_,_ = exporter.uncompressed_source(runtime,report['payload_kind'])
        reference = native.build(original,dict(aliases={}),directory/'reference')
        if native.query(reference,search_requests) != chosen:
            raise ValueError('compact runtime changes incumbent fixed-work search with identical tensor')
        details = campaign.immutable(directory/'DETAILS.json',dict(source=model['source'],runtime=model['runtime'],
            cases=plan['cases'],inference_error=float(np.max(np.abs(actual-expected))),
            search_cases=search_cases,normal_build=campaign.record(directory/'normal/BUILD.json'),
            sanitized_build=campaign.record(directory/'sanitized/BUILD.json'),
            reference_build=campaign.record(directory/'reference/BUILD.json'),
            compiler_tokens=campaign.record(directory/'TOKEN_PROOFS.json')))
        proofs = dict(correctness=proof(directory,'correctness',model['source'],details=details,
                          canonical_state_sha256=canonical,both_orientations=True,legal_fallback=True,
                          terminal_and_make_unmake=True),
                      sanitizers=proof(directory,'sanitizers',model['source'],details=details),
                      decoder=proof(directory,'decoder',model['source'],details=details,
                          tensor_values=len(values),base85_tail_lengths=transport),
                      native_parity=proof(directory,'native_parity',model['source'],details=details,
                          feature_queries=len(nonterminal),inference_queries=len(active)),
                      fixed_work=proof(directory,'fixed_work',model['source'],details=details,
                          identical_weights=True,search_queries=len(search_requests)))
        reports.append(dict(profile=model['profile'],seed=model['seed'],source=model['source'],
                            runtime=model['runtime'],proofs=proofs,details=details))
    result = dict(schema=SCHEMA,passed=True,sources=plan['sources'],plan=campaign.record(plan_path),
                  models=reports,canonical_state_sha256=canonical,whole_response_certified=False,
                  playing_strength_qualified=False,new_games=0)
    campaign.immutable(output/'RESULT.json',result)
    campaign.immutable(output/'EXPOSURE_PENDING.json',dict(schema=SCHEMA+'.exposure',
        cases=plan['cases'],report=campaign.record(output/'RESULT.json'),
        inputs=[row['details'] for row in reports],before_next_fresh_bank=True,training_eligible=False))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args()
    result=run(args.plan)
    print(json.dumps(dict(passed=result['passed'],models=len(result['models']))))


if __name__=='__main__':
    main()
