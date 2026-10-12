"""Select audited precision runtimes on a complete frozen validation bank."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import unittest

from tools import rank_two_live_eight_v1 as c
from tools import rank_two_focused_work_v1 as work
from tools import rank_two_live_eight_qat_runtime_v6 as qat_runtime

SCHEMA = c.SCHEMA + '.native-precision-selection.v3'
RULE = ['native_validation_regret', 'teacher_huber', 'earliest_epoch']
MATCHED = ('common_float_parent', 'seed', 'corpus', 'validation', 'live_roots',
           'live_training_provenance', 'selection', 'validation_root_accounting')


def select(arms):
    if len(arms) != 2 or {arm['bits'] for arm in arms} != {4, 6}:
        raise ValueError('one completed four-bit and six-bit arm required')
    first = arms[0]
    for arm in arms:
        if any(arm[key] != first[key] for key in MATCHED):
            raise ValueError('parent, seed, corpus, validation or provenance differs')
        if arm['selection'] != RULE or arm['calibration_split'] != 'train':
            raise ValueError('frozen selection or TRAIN-only calibration differs')
        counts = arm['validation_root_accounting']
        if ((counts['frozen'], counts['evaluated'], counts['excluded']) != (128, 128, 0)
                or counts['comparable_early'] != arm['early_roots']):
            raise ValueError('complete frozen/evaluated/excluded validation accounting differs')
        if (not 8 <= arm['epochs'] <= 32 or not 100 <= arm['early_roots'] <= 128
                or not 0 <= arm['epoch'] < arm['epochs']):
            raise ValueError('epoch bounds or comparable early coverage differs')
        if not 0 < arm['characters'] <= 99000:
            raise ValueError('source is not feasible under the declared target')
        if any(not math.isfinite(arm[key]) or arm[key] < 0 for key in ('regret', 'huber')):
            raise ValueError('finite nonnegative native metrics required')
    return min(arms, key=lambda arm: (arm['regret'], arm['huber'], arm['epoch']))


def validate_completion(completion, proof, runtime, numeric, receipt, bits, parent_step):
    flags = ('passed', 'all_root_journals_moments_native128_and_runtime_tensors_verified',
             'atomic_head_exact_last_epoch', 'calibration_TRAIN_only',
             'first_epoch_native_full_streamed_exact')
    if (completion.get('passed') is not True or not all(proof.get(k) is True for k in flags)
            or runtime['producer'] != c.record(qat_runtime.__file__)
            or runtime['numeric_plan'] != receipt['plan']
            or proof['numeric_plan'] != receipt['plan']
            or proof['runtime_plan'] != completion['plan']
            or proof['numeric_result'] != completion['numeric_result']
            or proof['selected'] != completion['selected']
            or numeric['selected'] != completion['selected']
            or numeric.get('passed') is not True or numeric['phase'] != 'qat'
            or numeric['weight_bits'] != bits or completion['weight_bits'] != bits
            or completion['selected_epoch'] != receipt['epoch']
            or proof['selected_epoch'] != receipt['epoch']
            or completion['selected_optimizer_step'] != receipt['checkpoint']['optimizer_step']
            or proof['initial_optimizer_step'] != parent_step):
        raise ValueError('exact audited runtime, numeric plan, parent or selection differs')
    epochs, updates = completion['epochs'], completion['updates']
    if (not 8 <= epochs <= 32 or proof['epochs'] != epochs or proof['updates'] != updates
            or len(proof['verified_epochs']) != epochs
            or len(proof['calibration_trials']) != 18
            or proof['final_optimizer_step'] != parent_step + updates
            or updates <= 0 or updates % epochs):
        raise ValueError('complete epoch, update or calibration accounting differs')
    roots = updates // epochs
    if (not 0 < roots <= 2048 or proof['calibration_root_contributions'] != 18 * roots
            or completion['selected_optimizer_step'] != parent_step + roots * (receipt['epoch'] + 1)):
        raise ValueError('bounded TRAIN contributions or selected optimizer step differs')
    return roots


def verify_retention(proof, selected, roots):
    selected_seen = False
    for reference in proof['verified_epochs']:
        epoch = c.read(c.verify(reference))
        if (not all(epoch.get(k) is True for k in
                    ('passed', 'full_optimizer_moments_verified', 'tensor_and_export_verified'))
                or epoch['updates'] != roots or len(epoch['journal_receipts']) != roots
                or epoch['validation_roots'] != 128):
            raise ValueError('retained epoch proof is incomplete')
        for key in ('receipt', 'native', 'runtime', 'source', 'root_order', 'checkpoint'):
            c.verify(epoch[key])
        for journal in epoch['journal_receipts']:
            c.verify(journal)
        selected_seen |= epoch['receipt'] == selected
    if not selected_seen:
        raise ValueError('selected epoch is absent from the complete retained proof')
    for reference in proof['calibration_trials']:
        trial = c.read(c.verify(reference))
        if (not all(trial.get(k) is True for k in ('passed', 'TRAIN_only', 'weighted_loss_exact_sum'))
                or len(trial['root_contributions']) != roots or trial['roots'] != roots):
            raise ValueError('TRAIN-only calibration contribution proof is incomplete')
        c.verify(trial['recipe'])
        c.verify(trial['result'])
        for contribution in trial['root_contributions']:
            c.verify(contribution)


def run(plan_path):
    plan = c.read(plan_path)
    state = work.checked(plan, 'experiment', launch=True)
    c.resource_guard(plan['resource_budget'])
    if (plan['producer'] != c.record(__file__) or state.get('active_claims')
            or state.get('pending_exposures') or plan['new_games'] != 0
            or plan['protected_gameplay_reads'] is not False):
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
        'tests.codingame.test_rank_two_live_eight_precision_selection_v3')
    tests = unittest.TextTestRunner(verbosity=1).run(suite)
    c.immutable(directory / 'UNIT_TESTS.json', dict(passed=tests.wasSuccessful(), tests=tests.testsRun))
    if not tests.wasSuccessful():
        raise ValueError('native precision selection tests failed')
    c.immutable(directory / 'CLAIM.json', dict(plan=c.record(plan_path), declaration=plan['declaration']))
    if declaration['common_float_parent'] != state['live_eight_next_common_float_selected_parent']:
        raise ValueError('current common float parent differs')
    parent = c.read(c.verify(declaration['common_float_parent']))
    arms = []
    for bits in (4, 6):
        completion_ref = plan['completions'][str(bits)]
        if completion_ref != state[f'live_eight_next_qat{bits}_complete']:
            raise ValueError('current precision completion differs')
        completion = c.read(c.verify(completion_ref))
        receipt = c.read(c.verify(completion['selected']))
        proof = c.read(c.verify(completion['verification']))
        runtime_plan = c.read(c.verify(completion['plan']))
        numeric = c.read(c.verify(completion['numeric_result']))
        roots = validate_completion(completion, proof, runtime_plan, numeric, receipt,
                                    bits, parent['optimizer_step'])
        verify_retention(proof, completion['selected'], roots)
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
            epochs=completion['epochs'], early_roots=receipt['metrics']['early_roots'],
            regret=receipt['metrics']['regret'], huber=receipt['metrics']['huber'],
            validation_root_accounting=receipt['metrics']['validation_root_accounting'],
            epoch=receipt['epoch'], characters=export['characters'])
        if (arm['common_float_parent'] != declaration['common_float_parent']
                or arm['corpus'] != declaration['corpus'] or arm['validation'] != declaration['validation']
                or arm['seed'] != declaration['shared_root_seed']):
            raise ValueError('declared common parent or validation differs')
        if arm['validation_root_accounting'] != declaration['validation_root_accounting']:
            raise ValueError('declared complete bank accounting differs')
        arms.append(arm)
    chosen = select(arms)
    summary = [{k: v for k, v in arm.items() if k != 'live_roots'} for arm in arms]
    result = dict(schema=SCHEMA, passed=True, plan=c.record(plan_path), producer=c.record(__file__),
        declaration=plan['declaration'], arms=summary, selection=RULE,
        selected=chosen['selected'], selected_bits=chosen['bits'], source=chosen['source'],
        runtime=chosen['runtime'], export=chosen['export'], characters=chosen['characters'],
        validation=chosen['validation'], corpus=chosen['corpus'],
        validation_root_accounting=chosen['validation_root_accounting'],
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
