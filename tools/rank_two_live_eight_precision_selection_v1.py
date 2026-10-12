"""Select a native precision proxy from one completed continuation pair."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import unittest

from tools import rank_two_live_eight_v1 as c
from tools import rank_two_focused_work_v1 as work

SCHEMA = c.SCHEMA + '.native-precision-selection.v1'
RULE = ['native_validation_regret', 'teacher_huber', 'earliest_epoch']
MATCHED = ('common_float_parent', 'seed', 'corpus', 'validation', 'live_roots',
           'live_training_provenance', 'selection')


def select(arms):
    if len(arms) != 2 or {arm['bits'] for arm in arms} != {4, 6}:
        raise ValueError('one completed four-bit and six-bit arm required')
    first = arms[0]
    for arm in arms:
        if any(arm[key] != first[key] for key in MATCHED):
            raise ValueError('parent, seed, corpus, validation or provenance differs')
        if arm['selection'] != RULE or arm['calibration_split'] != 'train':
            raise ValueError('frozen selection or TRAIN-only calibration differs')
        if not 8 <= arm['epochs'] <= 32 or arm['early_roots'] < 100:
            raise ValueError('epoch bounds or comparable early coverage differs')
        if not 0 < arm['characters'] <= 99000:
            raise ValueError('source is not feasible under the declared target')
        if any(not math.isfinite(arm[key]) or arm[key] < 0 for key in ('regret', 'huber')):
            raise ValueError('finite nonnegative native metrics required')
    return min(arms, key=lambda arm: (arm['regret'], arm['huber'], arm['epoch']))


def run(plan_path):
    plan = c.read(plan_path)
    state = work.checked(plan, 'experiment', launch=True)
    if plan['producer'] != c.record(__file__) or state.get('active_claims'):
        raise PermissionError('bound producer and no unresolved live claim required')
    for reference in plan['inputs'].values():
        c.verify(reference)
    directory = Path(plan['output'])
    if (directory / 'CLAIM.json').exists():
        raise RuntimeError('spent selection claim must not be repeated')
    declaration = c.read(c.verify(plan['declaration']))
    if plan['declaration'] != state['live_eight_next_precision_declaration']:
        raise ValueError('selection belongs to a different continuation')
    suite = unittest.defaultTestLoader.loadTestsFromName(
        'tests.codingame.test_rank_two_live_eight_precision_selection_v1')
    tests = unittest.TextTestRunner(verbosity=1).run(suite)
    c.immutable(directory / 'UNIT_TESTS.json', dict(passed=tests.wasSuccessful(), tests=tests.testsRun))
    if not tests.wasSuccessful():
        raise ValueError('native precision selection tests failed')
    c.immutable(directory / 'CLAIM.json', dict(plan=c.record(plan_path), declaration=plan['declaration']))
    arms = []
    for bits in (4, 6):
        completion_ref = plan['completions'][str(bits)]
        if completion_ref != state[f'live_eight_next_qat{bits}_complete']:
            raise ValueError('current precision completion differs')
        completion = c.read(c.verify(completion_ref))
        if not completion['passed'] or not completion['all_update_journals_verified']:
            raise ValueError('complete retained epochs and update journals required')
        receipt = c.read(c.verify(completion['selected']))
        training = c.read(c.verify(receipt['plan']))
        calibration = c.read(Path(training['output']) / 'CALIBRATION.json')
        for key in ('source', 'runtime', 'export', 'native'):
            c.verify(receipt[key])
        export = c.read(c.verify(receipt['export']))
        source = c.verify(receipt['source']).read_bytes()
        if (not source.isascii() or len(source) != export['characters']
                or export['source_sha256'] != receipt['source']['sha256']
                or export['weight_bits'] != bits or training['weight_bits'] != bits
                or calibration['weight_bits'] != bits):
            raise ValueError('native export/source identity differs')
        arm = dict(bits=bits, archive=completion_ref, selected=completion['selected'],
            source=receipt['source'], runtime=receipt['runtime'], export=receipt['export'],
            native=receipt['native'], common_float_parent=training['float_parent'],
            seed=training['seed'], corpus=training['corpus'], validation=training['fresh_validation'],
            live_roots=training['live_roots'], live_training_provenance=training['live_training_provenance'],
            selection=training['selection'], calibration_split=calibration['split'],
            epochs=completion['epochs_retained'], early_roots=receipt['metrics']['early_roots'],
            regret=receipt['metrics']['regret'], huber=receipt['metrics']['huber'],
            epoch=receipt['epoch'], characters=export['characters'])
        if (arm['common_float_parent'] != declaration['common_float_parent']
                or arm['corpus'] != declaration['corpus'] or arm['validation'] != declaration['validation']
                or arm['seed'] != declaration['shared_root_seed']):
            raise ValueError('declared common parent or validation differs')
        arms.append(arm)
    chosen = select(arms)
    summary = [{k: v for k, v in arm.items() if k != 'live_roots'} for arm in arms]
    result = dict(schema=SCHEMA, passed=True, plan=c.record(plan_path), producer=c.record(__file__),
        declaration=plan['declaration'], arms=summary, selection=RULE,
        selected=chosen['selected'], selected_bits=chosen['bits'], source=chosen['source'],
        runtime=chosen['runtime'], export=chosen['export'], characters=chosen['characters'],
        validation=chosen['validation'], corpus=chosen['corpus'],
        common_float_parent=chosen['common_float_parent'], common_seed=chosen['seed'],
        training_root_mass=dict(retained=.5, live=.5), clock_certification_pending=True,
        live_strength_qualified=False, private_qualified=False, formal_qualified=False)
    c.immutable(directory / 'RESULT.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    result = run(parser.parse_args().plan)
    print(json.dumps(dict(passed=result['passed'], selected_bits=result['selected_bits'],
                          characters=result['characters'])))
