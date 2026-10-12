"""Run the unchanged float update recipe with bounded durable native validation."""
import os
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'

import argparse
import json
from pathlib import Path

from tools import rank_two_live_eight_v1 as c
from tools import rank_two_live_eight_training_v5 as training
from tools import rank_two_live_eight_validation_stream_v1 as streaming
from tools import rank_two_live_eight_float_preflight_v1 as preflight
from tools import rank_two_focused_work_v1 as work


def validate_preflight(report, numeric_ref, numeric):
    accounting = report.get('validation_root_accounting', {})
    if (not all(report.get(key) is True for key in ('passed', 'exact_metrics_equal',
            'exact_native_predictions', 'optimizer_and_parameters_unchanged'))
            or report.get('numeric_plan') != numeric_ref or report.get('corpus') != numeric['corpus']
            or report.get('float_parent') != numeric['float_parent']
            or report.get('compared_roots') != 128 or report.get('training_updates') != 0
            or any(accounting.get(key) != value for key, value in
                   dict(frozen=128, evaluated=128, excluded=0).items())
            or accounting.get('comparable_early', 0) < 100):
        raise PermissionError('independent exact complete128-root float preflight required')


def run(plan_path):
    plan = c.read(plan_path)
    state = work.checked(plan, 'train', launch=True)
    c.resource_guard(plan['resource_budget'])
    if plan['producer'] != c.record(__file__):
        raise PermissionError('bound float runtime required')
    for ref in plan['inputs'].values():
        c.verify(ref)
    numeric = c.read(c.verify(plan['numeric_plan']))
    preflight.check_numeric(plan, numeric, state)
    if (state.get('active_jobs') != [plan['job_id']]
            or state.get('live_eight_continuation_04_float_runtime_plan') != c.record(plan_path)):
        raise PermissionError('sole current execution must bind this exact runtime plan')
    report = c.read(c.verify(plan['inputs']['validation_preflight']))
    validate_preflight(report, plan['numeric_plan'], numeric)
    output = Path(plan['output'])
    numeric_output = Path(numeric['output'])
    if (output / 'EXECUTION_CLAIM.json').exists() or (numeric_output / 'resume/HEAD.json').exists():
        raise PermissionError('spent float work requires classified recovery, never fresh replay')
    c.immutable(output / 'EXECUTION_CLAIM.json', dict(plan=c.record(plan_path),
        numeric_plan=plan['numeric_plan'], validation_preflight=plan['inputs']['validation_preflight'],
        claimed_at=c.now().isoformat(), numerical_threads=1, complete_update_journals_never_replayed=True))
    original_validation = training.validation

    def validate(corpus, parameters, architecture, bits, scales=None, native=None):
        head = c.read(numeric_output / 'resume/HEAD.json')
        checkpoint = c.read(c.verify(head['receipt']))
        cursor = checkpoint['cursor']
        if (checkpoint['plan'] != plan['numeric_plan'] or cursor['phase'] != 'float'
                or cursor['next_root'] != len(cursor['root_order']) or scales is not None):
            raise PermissionError('complete unchanged float epoch required before validation')
        epoch = numeric_output / 'float' / ('epoch-%02d' % cursor['epoch'])
        return streaming.validation(corpus, parameters, architecture, bits, None, native,
            numeric_output / 'validation-journal-v1' / ('float-epoch-%02d' % cursor['epoch']),
            dict(numeric_plan=plan['numeric_plan'], checkpoint=c.record(epoch / 'state.json'),
                 native=c.record(epoch / 'native/float-probe')),
            lambda: work.checked(plan, 'train'))

    training.validation = validate
    try:
        result = training.fit(c.verify(plan['numeric_plan']))
    finally:
        training.validation = original_validation
    completed = dict(passed=result['passed'], numeric_plan=plan['numeric_plan'],
        numeric_result=c.record(numeric_output / 'RESULT.json'), numerical_threads=1,
        maximum_validation_prediction_rows=2048,
        validation_root_accounting=result['validation_root_accounting'],
        gradient_recipe_unchanged=True, every_epoch_and_full_optimizer_state_retained=True)
    c.immutable(output / 'RESULT.json', completed)
    return completed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    print(json.dumps(run(parser.parse_args().plan)))
