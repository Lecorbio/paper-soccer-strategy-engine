"""Admit one continuation source after its own mechanics, timing and ancestry."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tools import rank_two_live_eight_v1 as c
from tools import rank_two_live_eight_model_v1 as codec
from tools.rank_two_live_eight_admission_v1 import require_identity


def selected_model(responses, clocks):
    if (not responses['passed'] or responses['own_failures'] != 0
            or not responses['all_spent_cases_archived']
            or responses['isolated_production_games'] != 16
            or responses['highest_safe_common_clocks_ms'] != clocks
            or clocks not in c.PLAN['clock_ladder']):
        raise PermissionError('complete source-bound continuation response certification required')
    rows = responses['source_bound_models']
    if (len(rows) != 4 or {tuple(row['model']['clocks_ms']) for row in rows}
            != {tuple(clock) for clock in c.PLAN['clock_ladder']}):
        raise PermissionError('the complete declared clock ladder is required')
    selected = [row for row in rows if row['model']['clocks_ms'] == clocks]
    if len(selected) != 1 or selected[0]['model']['profile'] != 'dd8':
        raise PermissionError('one exact eight/eight continuation source required')
    return selected[0]


def prepare(base, usage):
    base = Path(base)
    state = c.read(base / 'CURRENT.json')
    c.guard(base, 'freeze', state['owner_thread_id'], usage)
    if state.get('pending_exposures') or state.get('active_jobs') or state.get('active_claims'):
        raise PermissionError('archive and close every spent exposure before source preparation')
    ancestry = c.read(c.verify(state['live_eight_ancestry_validated']))
    if not ancestry['passed'] or ancestry['layers'] != state['live_eight_census_layers']:
        raise PermissionError('current full layered ancestry validation required')
    completion = c.read(c.verify(state['live_eight_next_clock_mechanical_complete']))
    mechanics = c.read(c.verify(completion['result']))
    responses = c.read(c.verify(state['live_eight_next_clock_response_complete']))
    clocks = state['live_eight_next_highest_safe_clocks']
    row = selected_model(responses, clocks)
    model = row['model']
    native = c.read(c.verify(state['live_eight_next_native_precision_assessment']))
    if model['runtime'] != native['runtime'] or model['seed'] != native['common_seed']:
        raise PermissionError('response source differs from the current native selection')
    source = c.verify(model['source']).read_bytes()
    export = c.read(c.verify(model['export']))
    runtime = c.read(c.verify(model['runtime']))
    values, _ = codec.validate(runtime)
    if (not source.isascii() or not 0 < len(source) <= 99000
            or export['characters'] != len(source) or export['source_sha256'] != model['source']['sha256']
            or export['tensor_sha256'] != runtime['payload_sha256'] or export['clocks_ms'] != clocks
            or runtime['architecture'] != c.PLAN['architecture']):
        raise PermissionError('source, tensor, architecture or clock binding differs')
    matching = [item for item in mechanics['models'] if item['source'] == model['source']]
    if len(matching) != 1:
        raise PermissionError('unique exact-source mechanical proof required')
    mechanical = matching[0]
    require_identity(dict(mechanical, passed=True), model)
    details = c.read(c.verify(mechanical['details']))
    require_identity(dict(details, passed=True), model)
    c.verify(details['compiler_tokens'])
    for reference in mechanical['proofs'].values():
        require_identity(c.read(c.verify(reference)), model)
    correctness = c.read(c.verify(mechanical['proofs']['correctness']))
    if len(set(correctness['canonical_state_sha256'])) < 2:
        raise PermissionError('independent canonical state coverage required')
    whole = c.read(c.verify(row['whole_response']))
    require_identity(whole, model)
    if (whole['own_failures'] != 0 or not whole['isolated'] or not whole['cold_warm_both_colors']
            or not whole['long_turn_observed'] or whole['first_max_ms'] >= clocks[0]
            or whole['later_max_ms'] >= clocks[1]
            or {(case['color'], case['offset']) for case in whole['cases']}
            != {(0, 0), (0, 3), (1, 0), (1, 3)}):
        raise PermissionError('isolated whole-response coverage differs')
    for case in whole['cases']:
        proof = c.read(c.verify(case['receipt']))
        require_identity(proof, model)
        if proof['own_failures'] != 0 or not proof['isolated']:
            raise PermissionError('a spent response case failed')
        c.verify(proof['game'])
    build = c.read(c.verify(row['build']))
    require_identity(build, model)
    c.verify(build['binary'])
    canonical = codec.document(runtime['profile'], values, runtime['scales'], runtime['weight_bits'])
    full_source, _, _ = codec.uncompressed_source(canonical, clocks)
    semantic = hashlib.sha256(full_source.encode('ascii')).hexdigest()
    output = base / 'live-eight-v1/source-admissions-v1' / model['source']['sha256']
    admission = c.immutable(output / 'ADMISSION.json', dict(schema=c.SCHEMA + '.source-admission.v1',
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
    frozen = c.immutable(output / 'FROZEN_SOURCE.json', entry)
    candidate = dict(state, frozen_sources=dict(state['frozen_sources']))
    candidate['frozen_sources'][model['source']['sha256']] = entry
    c.banned(candidate, c.read(c.verify(state['closures'])), model['source']['sha256'])
    c.source_admission(candidate, model['source']['sha256'])
    with c.locked(base):
        current = c.read(base / 'CURRENT.json')
        if (current['live_eight_ancestry_validated'] != state['live_eight_ancestry_validated']
                or current.get('pending_exposures') or current.get('active_jobs') or current.get('active_claims')):
            raise PermissionError('source preparation state changed')
        previous = current['frozen_sources'].get(model['source']['sha256'])
        if previous is not None and previous != entry:
            raise PermissionError('immutable source entry already differs')
        current['frozen_sources'][model['source']['sha256']] = entry
        current['live_eight_next_prepared_source'] = frozen
        current['live_eight_next_admission_preparer'] = c.record(__file__)
        c.atomic(base / 'CURRENT.json', current)
    return dict(passed=True, frozen_source=frozen, live_versions_consumed=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--usage', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.base, args.usage)))
