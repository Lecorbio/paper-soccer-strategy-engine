"""Run matched QAT with durable TRAIN-only calibration and strict native validation."""
import os
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'

import argparse
import json
from pathlib import Path

from tools import rank_two_live_eight_v1 as c
from tools import rank_two_live_eight_training_v5 as training
from tools import rank_two_live_eight_calibration_v4 as calibration
from tools import rank_two_live_eight_validation_stream_v1 as streaming
from tools import rank_two_focused_work_v1 as work


def validate_binding(plan, numeric, state, verified):
    bits = numeric['weight_bits']
    if (numeric['producer'] != c.record(training.__file__) or numeric['phase_mode'] != 'qat'
            or bits not in (4, 6) or numeric['profile'] != 'dd8'
            or state.get('active_jobs') != [plan['job_id']]
            or state.get(f'live_eight_continuation_04_qat{bits}_runtime_plan') != c.record(plan['path'])
            or numeric['corpus'] != state.get('live_eight_next_corpus')
            or numeric['fresh_validation'] != state.get('live_eight_next_fresh_validation')
            or numeric['float_parent'] != verified.get('selected_parent')
            or not all(verified.get(key) is True for key in
                ('passed', 'all_epochs_native128parity_retained',
                 'all_journals_root_orders_full_moments_verified', 'atomic_head_exact_last_epoch'))):
        raise PermissionError('sole current matched QAT with audited common float parent required')


def run(plan_path):
    plan = c.read(plan_path)
    state = work.checked(plan, 'train', launch=True)
    c.resource_guard(plan['resource_budget'])
    if plan['producer'] != c.record(__file__):
        raise PermissionError('bound QAT runtime required')
    for ref in plan['inputs'].values():
        c.verify(ref)
    numeric = c.read(c.verify(plan['numeric_plan']))
    verification = c.read(c.verify(plan['inputs']['common_float_verification']))
    validate_binding(dict(plan, path=str(Path(plan_path).resolve())), numeric, state, verification)
    output, numeric_output = Path(plan['output']), Path(numeric['output'])
    if (output / 'EXECUTION_CLAIM.json').exists() or (numeric_output / 'resume/HEAD.json').exists():
        raise PermissionError('spent QAT requires classified recovery, never fresh replay')
    c.immutable(output / 'EXECUTION_CLAIM.json', dict(plan=c.record(plan_path),
        numeric_plan=plan['numeric_plan'], common_float_verification=plan['inputs']['common_float_verification'],
        claimed_at=c.now().isoformat(), numerical_threads=1, complete_update_journals_never_replayed=True))
    original_calibration, original_validation = training.calibration, training.validation

    def validate(corpus, parameters, architecture, bits, scales=None, native=None):
        head = c.read(numeric_output / 'resume/HEAD.json')
        checkpoint = c.read(c.verify(head['receipt']))
        cursor = checkpoint['cursor']
        if (checkpoint['plan'] != plan['numeric_plan'] or cursor['phase'] != 'qat'
                or cursor['next_root'] != len(cursor['root_order']) or scales is None):
            raise PermissionError('exact fully updated QAT epoch required before validation')
        epoch = numeric_output / 'qat' / ('epoch-%02d' % cursor['epoch'])
        full = None
        if cursor['epoch'] == 0:
            full = original_validation(corpus, parameters, architecture, bits, scales, native)
            c.immutable(output / 'FIRST_EPOCH_FULL_ROOT_METRICS.json', full)
        result = streaming.validation(corpus, parameters, architecture, bits, scales, native,
            numeric_output / 'validation-journal-v1' / ('qat-epoch-%02d' % cursor['epoch']),
            dict(numeric_plan=plan['numeric_plan'], checkpoint=c.record(epoch / 'state.json'),
                 native=c.record(epoch / 'native/probe')),
            lambda: work.checked(plan, 'train'))
        if full is not None:
            if result != full or len(result['roots']) != 128:
                raise ValueError('complete128-root QAT full/streamed metrics differ')
            c.immutable(output / 'FIRST_EPOCH_VALIDATION_PROOF.json', dict(passed=True,
                numeric_plan=plan['numeric_plan'], weight_bits=bits, compared_roots=128,
                exact_metrics_equal=True, exact_native_predictions=True,
                checkpoint=c.record(epoch / 'state.json'), native=c.record(epoch / 'native/probe'),
                full_metrics=c.record(output / 'FIRST_EPOCH_FULL_ROOT_METRICS.json'),
                bounded_metrics=c.record(numeric_output / 'validation-journal-v1/qat-epoch-00/RESULT.json')))
        return result

    training.calibration = lambda parameters, architecture, corpus, roots, bits, masses: calibration.calibration(
        parameters, architecture, corpus, roots, bits, masses,
        numeric_output / 'calibration-journal-v1', plan['numeric_plan'],
        lambda: work.checked(plan, 'train'))
    training.validation = validate
    try:
        result = training.fit(c.verify(plan['numeric_plan']))
    finally:
        training.calibration, training.validation = original_calibration, original_validation
    completed = dict(passed=result['passed'], numeric_plan=plan['numeric_plan'],
        numeric_result=c.record(numeric_output / 'RESULT.json'), weight_bits=numeric['weight_bits'],
        train_only_calibration=True, calibration_root_receipts_retained=True, numerical_threads=1,
        maximum_validation_prediction_rows=2048,
        first_epoch_validation_proof=c.record(output / 'FIRST_EPOCH_VALIDATION_PROOF.json'),
        validation_root_accounting=result['validation_root_accounting'])
    c.immutable(output / 'RESULT.json', completed)
    return completed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    print(json.dumps(run(parser.parse_args().plan)))
