"""Fixed-budget architecture training with observed-parent outcome supervision."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

for key in ('MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'OMP_NUM_THREADS',
            'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
os.environ['PAPERSOCCER_COMPACT_TRAINING_THREADS_FIXED_BEFORE_NUMPY'] = '1'
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

from tools import compact_representation_train as legacy
from tools import compact_representation_pilot as mechanics
from tools import compact_value_bfm_train as core
from tools import rank_two_design_network_v1 as net
from tools import rank_two_live_v3 as live
from tools import rank_two_design_wave_v2 as wave

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = ROOT / 'results/rank_two_20260927/live-loss-v3/campaign-next-v1/phase-b-training-v2'
SCHEMA = 'papersoccer.design-training.v2'
RECIPE = {'float_epochs': 1, 'qat_epochs': 4, 'seed': 2026093007,
          'float_learning_rate': 6e-5, 'qat_learning_rate': 2.5e-4,
          'weight_decay': 1e-5, 'gradient_clip': 5, 'huber_delta': .25,
          'weight_bits': 4, 'native_threads': 1}
ARMS = {'dd8-teacher': ('dd8', 0., 0.), 'dd8-mixed': ('dd8', .2, 0.),
        'dd12wide-ranking': ('dd12wide', 0., .25)}


def load_frozen(path, name, digest):
    if live.record(path)['sha256'] != digest:
        raise ValueError('changed frozen numerical implementation')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def support():
    driver = load_frozen(HISTORICAL / 'producer.py', 'design_frozen_forward',
                         '48787fe3285af1b1e0f68e86fe43de4d4669b4e49d3735f7d466c1e0847ca005')
    return driver, driver.helpers


@dataclass(frozen=True)
class Architecture:
    inputs: int
    hidden_one: int
    hidden_two: int

    @property
    def shapes(self):
        return {'w1': (self.inputs, self.hidden_one),
                'w2': (self.hidden_one, self.hidden_two), 'w3': (self.hidden_two,)}


def tensors(runtime):
    _, values, scales = net.validate(runtime)
    architecture = Architecture(*runtime['architecture'][:3])
    flat = np.asarray(values, dtype=np.int8)
    cuts = np.cumsum([0, *(math.prod(architecture.shapes[key]) for key in ('w1', 'w2', 'w3'))])
    integer = {key: flat[cuts[i]:cuts[i + 1]].reshape(architecture.shapes[key]).copy()
               for i, key in enumerate(('w1', 'w2', 'w3'))}
    return architecture, core.QuantizedWeights(integer, dict(zip(('w1', 'w2', 'w3'), map(np.float32, scales))))


class ObservedParents:
    """Only original played parent states may receive weak outcome targets."""
    def __init__(self, positions, roster):
        self.rows = {}
        allowed = {split: set(roster[split + '_roots']) for split in ('train', 'validation')}
        for row in positions:
            split = row['split']
            if split not in allowed:
                raise ValueError('protected split cannot enter observed-parent labels')
            if row['root_group_id'] not in allowed[split]:
                continue
            if type(row['winner']) is not int or row['winner'] not in (0, 1) or row['mover'] not in (0, 1):
                raise ValueError('observed outcome/mover is not a contest player')
            key = (split, row['root_group_id'], row['game_id'], row['canonical_state'], row['mover'])
            prior = self.rows.get(key)
            if prior is not None and prior != row['winner']:
                raise ValueError('conflicting observed outcome identity')
            self.rows[key] = row['winner']

    def target(self, group, outcome_weight):
        if outcome_weight not in (0., .2):
            raise ValueError('unapproved outcome mixture')
        key = (group['split'], group['root_group_id'], group['game_id'], group['canonical_state'], group['mover'])
        if key not in self.rows:
            raise ValueError('parent state has no exact original played-position identity')
        teacher = float(group['teacher_value'])
        if not math.isfinite(teacher) or not -1 <= teacher <= 1:
            raise ValueError('parent teacher value outside numerical contract')
        proof = group.get('proof', {})
        if proof.get('solved'):
            winner = proof.get('proven_winner')
            if type(winner) is not int or winner not in (0, 1) or teacher != (1. if winner == group['mover'] else -1.):
                raise ValueError('proved parent value/winner perspective differs')
            return teacher
        if group.get('terminal') and abs(teacher) != 1:
            raise ValueError('terminal parent lacks exact outcome target')
        if group.get('terminal') or abs(teacher) == 1:
            return teacher
        result = 1. if self.rows[key] == group['mover'] else -1.
        return (1. - outcome_weight) * teacher + outcome_weight * result


def rows(groups, parents, outcome_weight, with_parent):
    active, targets, signs, spans = legacy.root_rows(groups)
    parent_indices = []
    if with_parent:
        active = list(active)
        target_list, sign_list = targets.tolist(), signs.tolist()
        for group in groups:
            parent_indices.append(len(active))
            active.append(np.asarray(group['parent_active'], dtype=np.uint16))
            target_list.append(parents.target(group, outcome_weight))
            sign_list.append(1.)
        targets, signs = np.asarray(target_list, dtype=np.float32), np.asarray(sign_list, dtype=np.float32)
    return active, targets, signs, spans, parent_indices


def objective(groups, predictions, targets, signs, spans, parent_indices, ranking_weight, helpers):
    if not parent_indices:
        h, ranking, gradient, ranking_only = helpers.objective(groups, predictions, targets, signs, spans, ranking_weight)
        return h, ranking, gradient, ranking_only
    if ranking_weight != 0 or len(parent_indices) != len(groups):
        raise ValueError('8x8 parent pair has no ranking loss and exactly one parent per group')
    gradient = np.zeros_like(predictions)
    total = 0.
    for span, parent in zip(spans, parent_indices, strict=True):
        child_loss, child_grad = core._weighted_huber_loss_gradient(predictions[span], targets[span], np.ones(span.stop - span.start, dtype=np.float32))
        parent_loss, parent_grad = core._weighted_huber_loss_gradient(predictions[parent:parent + 1], targets[parent:parent + 1], np.ones(1, dtype=np.float32))
        gradient[span] = np.float32(.5 / len(groups)) * child_grad
        gradient[parent] = np.float32(.5 / len(groups)) * parent_grad[0]
        total += .5 * (child_loss + parent_loss) / len(groups)
    if not np.all(np.isfinite(gradient)):
        raise FloatingPointError('nonfinite parent-supervision gradient')
    return total, 0., gradient, np.zeros_like(gradient)


class Corpus:
    def __init__(self, path, roster):
        self.original = legacy.Corpus(path)
        self.roots = {split: {root: self.original.roots[split][root] for root in roster[split + '_roots']}
                      for split in ('train', 'validation')}
        if any(len(self.roots[split]) != 128 for split in self.roots):
            raise ValueError('audited128/128 roster required')

    def root(self, split, root):
        if root not in self.roots[split]:
            raise ValueError('root outside audited roster')
        return self.original.root(split, root)


def paused(plan):
    state = live.read(plan['campaign_current'])
    return (state['ownership_status'] != 'active' or state['owner_thread_id'] != plan['owner_thread_id']
            or state.get('human_pause', {}).get('status') == 'paused'
            or state.get('design_wave_activation') != plan['wave_activation'])


def validate_wave_inputs(plan, approved):
    for key in ('corpus', 'roster', 'played_positions', 'deployment_template'):
        if plan.get(key) != approved.get(key):
            raise ValueError('training changed approved ' + key)
    if plan.get('network_support') != live.record(net.__file__):
        raise ValueError('training architecture implementation differs')


def train(plan_path):
    plan = live.read(plan_path)
    live.refs(plan)
    if plan.get('schema') != SCHEMA or plan['id'] not in ARMS or plan['recipe'] != RECIPE:
        raise ValueError('unapproved training arm or fixed recipe')
    profile, outcome_weight, ranking_weight = ARMS[plan['id']]
    if plan['profile'] != profile or plan['outcome_weight'] != outcome_weight or plan['ranking_weight'] != ranking_weight:
        raise ValueError('arm/profile/objective differs')
    if plan['producer'] != live.record(__file__) or plan['python'] != live.record(Path(sys.executable).resolve()) or plan['numpy_version'] != np.__version__:
        raise ValueError('frozen producer/numerical runtime differs')
    state = live.read(plan['campaign_current'])
    if state.get('design_wave_activation') != plan['wave_activation'] or state.get('next_wave_status') != 'active':
        raise PermissionError('new wave not activated after predecessor audit')
    validate_wave_inputs(plan, live.read(live.verify(state['next_wave_plan'])))
    control = wave.guard_module(state)
    control.validate(state, live.read(live.verify(state['closures'])), plan['owner_thread_id'], 'train', plan['used_percent'])
    driver, helpers = support()
    roster = live.read(live.verify(plan['roster']))
    corpus = Corpus(live.verify(plan['corpus']), roster)
    positions = live.read(live.verify(plan['played_positions']))['positions']
    parents = ObservedParents(positions, roster)
    if not live.read(live.verify(plan['native_preflight']))['passed'] or not live.read(live.verify(plan['objective_preflight']))['passed']:
        raise ValueError('native/objective preflight incomplete')
    architecture, initial = tensors(net.initialize(profile))
    parameters, scales = initial.effective(), initial.scales.copy()
    optimizer = core.AdamW(parameters, learning_rate=RECIPE['float_learning_rate'], weight_decay=RECIPE['weight_decay'])
    binding = live.record(plan_path)
    out = Path(plan['output'])
    out.mkdir(parents=True, exist_ok=True)
    live.emit(out / 'arm.json', {'plan': binding, 'profile': profile, 'initial_parameters_sha256': mechanics.parameter_digest(parameters),
                                'scales': {k: float(v) for k, v in scales.items()}, 'original_labels_unchanged': True})
    root_ids = sorted(corpus.roots['train'])
    reference = live.read(live.verify(plan['reference_native_validation']))
    receipts = []
    for epoch in range(5):
        if paused(plan):
            raise PermissionError('campaign paused or ownership/wave changed')
        ep = out / f'epoch-{epoch:02d}'
        ep.mkdir(exist_ok=True)
        optimizer.learning_rate = np.float32(RECIPE['float_learning_rate'] if epoch == 0 else RECIPE['qat_learning_rate'])
        if (ep / 'receipt.json').exists():
            receipt = live.read(ep / 'receipt.json')
            live.refs(receipt)
            if receipt['plan'] != binding or receipt['epoch'] != epoch:
                raise ValueError('completed checkpoint binding differs')
            legacy.restore(receipt['checkpoint'], parameters, optimizer)
            if optimizer.step != 128 * (epoch + 1) or mechanics.parameter_digest(parameters) != receipt['parameters_sha256']:
                raise ValueError('completed checkpoint/update identity differs')
            receipts.append(receipt)
            continue
        if (ep / 'claim.json').exists():
            raise RuntimeError('unknown epoch claim retained; never replay')
        order = np.random.default_rng(RECIPE['seed'] + epoch).permutation(128)
        ids = [root_ids[i] for i in order]
        live.emit(ep / 'claim.json', {'plan': binding, 'epoch': epoch, 'root_order': ids})
        for root in ids:
            if paused(plan):
                raise PermissionError('pause/ownership/wave change retains unfinished claim')
            groups = corpus.root('train', root)
            active, targets, signs, spans, parent_indices = rows(groups, parents, outcome_weight, profile == 'dd8')
            quantized = None if epoch == 0 else mechanics.quantize(parameters, scales, 4)
            predictions, cache = driver.native_order_forward(parameters, architecture, active, quantized)
            _, _, derivative, _ = objective(groups, predictions, targets, signs, spans, parent_indices, ranking_weight, helpers)
            gradients = core._network_gradients(parameters, architecture, active, cache, derivative,
                                                parameters if quantized is None else quantized.effective())
            if any(not np.all(np.isfinite(x)) for x in gradients.values()):
                raise FloatingPointError('nonfinite gradient before clipping/update')
            norm = math.sqrt(sum(float(np.sum(x.astype(np.float64) ** 2)) for x in gradients.values()))
            if norm > RECIPE['gradient_clip']:
                for x in gradients.values():
                    x *= np.float32(RECIPE['gradient_clip'] / norm)
            optimizer.update(parameters, gradients)
            if any(not np.all(np.isfinite(x)) for x in (*parameters.values(), *optimizer.first.values(), *optimizer.second.values())):
                raise FloatingPointError('nonfinite parameter/optimizer state')
        quantized = mechanics.quantize(parameters, scales, 4)
        values = np.concatenate([quantized.integer[k].ravel() for k in ('w1', 'w2', 'w3')]).astype(int).tolist()
        runtime = net.document(profile, values, [float(scales[k]) for k in ('w1', 'w2', 'w3')], trained=True,
                               lineage={'plan': binding, 'epoch': epoch, 'outcome_weight': outcome_weight, 'ranking_weight': ranking_weight})
        raw, report = net.export(runtime, plan['deployment_template'])
        live.emit(ep / 'runtime.json', runtime)
        with (ep / 'submission.cpp').open('xb') as stream:
            stream.write(raw)
        metrics, native = helpers.native_metrics(ep / 'submission.cpp', runtime, corpus, ep / 'native-validation', plan,
                                                 driver.native_order_forward, tensors, reference)
        receipt = {'schema': SCHEMA, 'plan': binding, 'epoch': epoch, 'optimizer_steps': optimizer.step,
                   'parameters_sha256': mechanics.parameter_digest(parameters), 'checkpoint': legacy.checkpoint(ep / 'checkpoint.npz', parameters, optimizer),
                   'source': live.record(ep / 'submission.cpp'), 'runtime': live.record(ep / 'runtime.json'),
                   'export': report, 'validation': metrics, 'native_validation': live.record(ep / 'native-validation/native-validation.json')}
        live.emit(ep / 'receipt.json', receipt)
        receipts.append(receipt)
        print(json.dumps({'arm': plan['id'], 'epoch': epoch, 'optimizer_steps': optimizer.step, 'source_characters': len(raw), 'validation': metrics}), flush=True)
    if optimizer.step != 640:
        raise ValueError('fixed640-update budget incomplete')
    eligible = [x for x in receipts if x['epoch'] and x['export']['deployable_size']]
    chosen = min(eligible, key=lambda x: (x['validation']['regret'], x['validation']['huber'], x['epoch'])) if eligible else None
    result = {'schema': SCHEMA, 'plan': binding, 'status': 'complete', 'updates': 640,
              'epochs': [live.record(out / f'epoch-{i:02d}/receipt.json') for i in range(5)],
              'selected': chosen, 'playing_strength_qualified': False, 'new_games': 0}
    live.emit(out / 'result.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    result = train(args.plan)
    print(json.dumps({'status': result['status'], 'updates': result['updates'], 'selected_epoch': None if result['selected'] is None else result['selected']['epoch']}))
