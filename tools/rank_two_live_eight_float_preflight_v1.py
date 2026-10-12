"""Check unchanged float validation through full-root and bounded native paths."""
import os
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'

import argparse
from pathlib import Path

from tools import rank_two_live_eight_v1 as c
from tools import rank_two_live_eight_training_v5 as training
from tools import rank_two_live_eight_validation_stream_v1 as streaming
from tools import rank_two_focused_work_v1 as work


def check_numeric(plan, numeric, state):
    declaration_ref = state.get('live_eight_continuation_04_declaration')
    if (numeric['producer'] != c.record(training.__file__) or numeric['profile'] != 'dd8'
            or numeric['phase_mode'] != 'float' or numeric['float_maximum'] != 80
            or numeric['corpus'] != state.get('live_eight_next_corpus')
            or numeric['fresh_validation'] != state.get('live_eight_next_fresh_validation')
            or plan['declaration'] != declaration_ref or numeric['inputs']['declaration'] != declaration_ref
            or state.get('pending_exposures') or state.get('active_claims')):
        raise PermissionError('current closed declaration-bound float continuation required')
    declaration = c.read(c.verify(declaration_ref))
    if (numeric['float_parent'] != declaration['float_parent']
            or numeric['seed'] != declaration['train_seed']
            or declaration['architecture'] != [6301, 8, 8, 1]):
        raise PermissionError('retained float parent, moments, seed or fixed width differs')


def run(plan_path):
    plan = c.read(plan_path)
    state = work.checked(plan, 'train', launch=True)
    c.resource_guard(plan['resource_budget'])
    if plan['producer'] != c.record(__file__):
        raise PermissionError('bound float preflight producer required')
    for ref in plan['inputs'].values():
        c.verify(ref)
    numeric = c.read(c.verify(plan['numeric_plan']))
    check_numeric(plan, numeric, state)
    corpus = training.reader.Corpus(c.verify(numeric['corpus']))
    training.validate_plan(numeric, corpus, state)
    architecture, parameters = training.retained.initialize('dd8', numeric['seed'])
    optimizer = training.retained.core.AdamW(parameters, learning_rate=.001, weight_decay=1e-5)
    parent = c.read(c.verify(numeric['float_parent']))
    if parent['cursor']['phase'] != 'float':
        raise PermissionError('complete useful float checkpoint required')
    training.storage.restore(parent, parameters, optimizer, parent['plan'])
    before = training.storage.arrays(parameters, optimizer)
    out = Path(plan['output'])
    binary = training.float_probe(parameters, out / 'native')

    def native(root, active):
        requests = ['eval ' + str(len(row)) + ' ' + ' '.join(map(str, row)) for row in active]
        return [float(value) for value in training.native_probe.query(binary, requests)]

    complete = training.validation(corpus, parameters, architecture, numeric['weight_bits'], None, native)
    complete_ref = c.immutable(out / 'FULL_ROOT_METRICS.json', complete)
    bounded = streaming.validation(corpus, parameters, architecture, numeric['weight_bits'], None, native,
        out / 'roots', dict(numeric_plan=plan['numeric_plan'], float_parent=numeric['float_parent'],
                           native=c.record(binary)), lambda: work.checked(plan, 'train'))
    if bounded != complete or training.storage.arrays(parameters, optimizer) != before:
        raise ValueError('bounded metrics or retained parameter/optimizer state differs')
    if len(bounded['roots']) != 128 or bounded['early_roots'] < 100:
        raise PermissionError('strict complete128-root native preflight required')
    report = dict(passed=True, numeric_plan=plan['numeric_plan'], corpus=numeric['corpus'],
        float_parent=numeric['float_parent'], native=c.record(binary),
        complete_metrics=complete_ref, bounded_metrics=c.record(out / 'roots/RESULT.json'),
        exact_metrics_equal=True, exact_native_predictions=True,
        optimizer_and_parameters_unchanged=True, optimizer_step=int(optimizer.step),
        compared_roots=128, validation_root_accounting=bounded['validation_root_accounting'],
        maximum_prediction_rows=2048, numerical_threads=1, training_updates=0)
    c.immutable(out / 'RESULT.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    print(run(parser.parse_args().plan))
