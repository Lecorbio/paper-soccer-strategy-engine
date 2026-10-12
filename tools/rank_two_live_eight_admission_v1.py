"""Bind existing exact-source mechanical and isolated production-clock proofs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tools import rank_two_live_eight_v1 as c
from tools import rank_two_live_eight_model_v1 as model_codec


def require_identity(proof, model):
    if proof.get('passed') is not True or proof.get('source') != model['source']:
        raise PermissionError('exact source proof differs')
    if 'runtime' in proof and proof['runtime'] != model['runtime']:
        raise PermissionError('proof tensor runtime differs')
    if 'clocks_ms' in proof and proof['clocks_ms'] != model['clocks_ms']:
        raise PermissionError('proof declared response clocks differ')


def prepare(base, usage):
    base = Path(base)
    state = c.read(base / 'CURRENT.json')
    c.guard(base, 'freeze', state['owner_thread_id'], usage)
    if state.get('pending_exposures') or state.get('active_jobs') or state.get('active_claims'):
        raise PermissionError('finish and close spent work before source preparation')
    ancestry = c.read(c.verify(state['live_eight_ancestry_validated']))
    if not ancestry.get('passed') or ancestry['layers'] != state['live_eight_census_layers']:
        raise PermissionError('current full layered ancestry validation required')
    mechanics = c.read(c.verify(state['live_eight_clock_ladder_mechanical_complete']))
    responses = c.read(c.verify(state['live_eight_clock_ladder_response_complete']))
    clocks = state['live_eight_highest_safe_common_clocks']
    if clocks != [990, 180] or responses['highest_safe_common_clocks_ms'] != clocks:
        raise PermissionError('saved highest safe common response tuple differs')
    frozen = []
    for row in responses['source_bound_models']:
        model = row['model']
        if model['clocks_ms'] != clocks:
            continue
        source = c.verify(model['source']).read_bytes()
        source.decode('ascii')
        export = c.read(c.verify(model['export']))
        runtime = c.read(c.verify(model['runtime']))
        model_codec.validate(runtime)
        if (not 0 < len(source) <= 99000 or export['characters'] != len(source)
                or export['source_sha256'] != model['source']['sha256']
                or export['tensor_sha256'] != runtime['payload_sha256']
                or export['clocks_ms'] != clocks):
            raise PermissionError('source/export/tensor binding differs')
        matching = [x for x in mechanics['models'] if x['source'] == model['source']]
        if len(matching) != 1 or matching[0]['runtime'] != model['runtime'] or matching[0]['clocks_ms'] != clocks:
            raise PermissionError('unique source-bound mechanical proof required')
        mechanical = matching[0]
        details = c.read(c.verify(mechanical['details']))
        if details['source'] != model['source'] or details['runtime'] != model['runtime']:
            raise PermissionError('source mechanics details differ')
        c.verify(details['compiler_tokens'])
        for proof in mechanical['proofs'].values():
            require_identity(c.read(c.verify(proof)), model)
        correctness = c.read(c.verify(mechanical['proofs']['correctness']))
        if len(set(correctness['canonical_state_sha256'])) < 2:
            raise PermissionError('independent canonical state coverage required')
        whole = c.read(c.verify(row['whole_response']))
        require_identity(whole, model)
        if (whole['own_failures'] != 0 or whole.get('isolated') is not True
                or whole.get('cold_warm_both_colors') is not True or whole.get('long_turn_observed') is not True
                or {(case['color'], case['offset']) for case in whole['cases']} != {(0, 0), (0, 3), (1, 0), (1, 3)}):
            raise PermissionError('isolated production response coverage differs')
        for case in whole['cases']:
            proof = c.read(c.verify(case['receipt']))
            require_identity(proof, model)
            if proof.get('own_failures') != 0 or proof.get('isolated') is not True:
                raise PermissionError('a spent production case failed')
            c.verify(proof['game'])
        build = c.read(c.verify(row['build']))
        if build['source'] != model['source'] or build['runtime'] != model['runtime']:
            raise PermissionError('production build source identity differs')
        c.verify(build['binary'])
        if model['profile'] == 'control':
            comparison = details['incumbent_comparison']
            if (comparison['source'] != state['incumbent'] or comparison['identical_tensor'] is not True
                    or comparison['identical_fixed_work'] is not True):
                raise PermissionError('retimed control differs from original incumbent')
            c.verify(state['live_eight_incumbent_tensor_identity'])
        # Remove training lineage before hashing the full uncompressed program:
        # re-encoding, aliases, or checkpoint provenance cannot change this identity.
        values, _ = model_codec.validate(runtime)
        canonical_runtime = model_codec.document(runtime['profile'], values, runtime['scales'], runtime['weight_bits'])
        semantic_source, _, _ = model_codec.uncompressed_source(canonical_runtime, clocks)
        semantic = hashlib.sha256(semantic_source.encode('ascii')).hexdigest()
        out = base / 'live-eight-v1/source-admissions-v1' / model['source']['sha256']
        admission = c.immutable(out / 'ADMISSION.json', dict(schema=c.SCHEMA+'.source-admission.v1',
            passed=True, source=model['source'], runtime=model['runtime'], activation=state['focused_activation'],
            clocks_ms=clocks, own_failures=0, canonical_state_sha256=correctness['canonical_state_sha256'],
            **mechanical['proofs'], whole_response=row['whole_response'], timing_experiment=row['whole_response'],
            production_build=row['build'], mechanics_details=mechanical['details'],
            ancestry=state['live_eight_ancestry_validated'], playing_strength_qualified=False,
            producer=c.record(__file__)))
        entry = dict(model, admission=admission, source_characters=len(source),
            nonpayload_characters=export['nonpayload_characters'], semantic_identity=semantic,
            semantic_identity_method='sha256 of lineage-free full uncompressed runtime program at declared clocks',
            playing_strength_qualified=False, experimental_live_source=True)
        if model['profile'] == 'control':
            entry['incumbent_tensor_identity'] = state['live_eight_incumbent_tensor_identity']
        receipt = c.immutable(out / 'FROZEN_SOURCE.json', entry)
        frozen.append(dict(entry, frozen_receipt=receipt))
    if len(frozen) != 2 or {row['profile'] for row in frozen} != {'dd8', 'control'}:
        raise PermissionError('one selected 8/8 and one exact retimed control required')
    candidate_state = dict(state)
    candidate_state['frozen_sources'] = dict(state['frozen_sources'])
    candidate_state['live_eight_retimed_control'] = next(row['source']['sha256'] for row in frozen if row['profile'] == 'control')
    for entry in frozen:
        candidate_state['frozen_sources'][entry['source']['sha256']] = entry
        c.banned(candidate_state, c.read(c.verify(state['closures'])), entry['source']['sha256'])
        c.source_admission(candidate_state, entry['source']['sha256'])
    with c.locked(base):
        current = c.read(base / 'CURRENT.json')
        if current['live_eight_ancestry_validated'] != state['live_eight_ancestry_validated'] or current['pending_exposures'] or current['active_jobs']:
            raise PermissionError('ancestry changed during source preparation')
        for entry in frozen:
            previous = current['frozen_sources'].get(entry['source']['sha256'])
            if previous is not None and previous != entry:
                raise PermissionError('immutable frozen source entry already differs')
            current['frozen_sources'][entry['source']['sha256']] = entry
        current['live_eight_retimed_control'] = candidate_state['live_eight_retimed_control']
        current['live_eight_990_source_admission_preparer'] = c.record(__file__)
        current['live_eight_990_prepared_sources'] = [row['frozen_receipt'] for row in frozen]
        c.atomic(base / 'CURRENT.json', current)
    return dict(passed=True, frozen_sources=[row['frozen_receipt'] for row in frozen], live_versions_consumed=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--usage', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.base, args.usage)))
