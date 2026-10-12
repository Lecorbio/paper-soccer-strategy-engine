"""Stream TRAIN successors in bounded batches with exact full-root error reduction."""
import hashlib
import gzip
import gc
import json
from pathlib import Path

from tools import rank_two_live_eight_training_v5 as training
from tools import rank_two_live_eight_calibration_v1 as previous
from tools import rank_two_live_eight_group_stream_v1 as group_stream

c, np = training.campaign, training.np


def bounded_predict(parameters, architecture, active, quantized=None, maximum=2048):
    if type(maximum) is not int or maximum < 1:
        raise ValueError('positive prediction batch bound required')
    predictions = np.empty(len(active), dtype=np.float32)
    for start in range(0, len(active), maximum):
        end = min(start + maximum, len(active))
        predictions[start:end] = training.retained.forward(parameters, architecture, active[start:end], quantized)[0]
    return predictions


def root_mse(parameters, architecture, corpus, root, quantized):
    errors = []
    def evaluate(metadata, batch):
        if metadata['split'] != 'train':
            raise PermissionError('TRAIN-only scale calibration required')
        active = training.retained.root_rows([{**metadata, 'successors': batch}])[0]
        target = bounded_predict(parameters, architecture, active)
        prediction = bounded_predict(parameters, architecture, active, quantized)
        errors.append(np.asarray((prediction - target) ** 2, dtype=np.float32))
    if hasattr(corpus, 'roots'):
        for item in corpus.roots['train'][root]:
            batch = []
            with gzip.open(training.reader.bound(item['artifact']), 'rt') as stream:
                for metadata, child in group_stream.successors(stream, root, item['group_id']):
                    batch.append(child)
                    if len(batch) == 2048:
                        evaluate(metadata, batch)
                        batch = []
                if batch:
                    evaluate(metadata, batch)
            gc.collect()
    else:
        for group in corpus.root('train', root):
            evaluate(group, group['successors'])
    if not errors:
        raise ValueError('complete nonempty root required')
    return float(np.mean(np.concatenate(errors)))


def calibration(parameters, architecture, corpus, roots, bits, masses, output, binding, boundary=lambda: None):
    output = Path(output)
    digest = hashlib.sha256(b''.join(parameters[k].tobytes() for k in ('w1', 'w2', 'w3'))).hexdigest()
    c.immutable(output / 'BINDING.json', dict(plan=binding, parameter_sha256=digest,
        corpus=corpus.identity, roots=roots, masses=masses, bits=bits, split='train',
        numerical_threads=1, producer=c.record(previous.__file__)))
    c.immutable(output / 'IMPLEMENTATION-v4.json', dict(producer=c.record(__file__),
        original_recipe=c.record(previous.__file__), prediction_batch_rows=2048,
        root_contributions_and_numpy_mean_unchanged=True, streamed_successors=True))
    maximum = (1 << (bits - 1)) - 1
    candidates = {k: [np.float32(max(float(np.percentile(np.abs(v), p)) / maximum, 1e-8))
                     for p in (95, 99, 100)] for k, v in parameters.items()}
    scales = {k: v[-1] for k, v in candidates.items()}
    trials = []
    for iteration in range(2):
        for layer in ('w1', 'w2', 'w3'):
            choices = []
            for candidate_index, candidate in enumerate(candidates[layer]):
                proposed = {**scales, layer: candidate}
                directory = output / ('pass-%d-%s-%d' % (iteration, layer, candidate_index))
                recipe = c.immutable(directory / 'RECIPE.json', dict(plan=binding,
                    parameter_sha256=digest, pass_index=iteration, layer=layer,
                    scales={k: float(v) for k, v in proposed.items()}))
                q = training.model.quantize(parameters, proposed, bits)
                loss = 0.
                for index, root in enumerate(roots):
                    boundary()
                    path = directory / ('root-%04d.json' % index)
                    if path.exists():
                        saved = c.read(path)
                        if saved['recipe'] != recipe or saved['root'] != root:
                            raise ValueError('retained calibration root binding changed')
                        contribution = saved['weighted_float_mse']
                    else:
                        contribution = masses[root] * root_mse(parameters, architecture, corpus, root, q) / len(roots)
                        c.immutable(path, dict(recipe=recipe, root=root, weighted_float_mse=contribution))
                        gc.collect()
                    loss += contribution
                    c.atomic(output / 'PROGRESS.json', dict(pass_index=iteration, layer=layer,
                        candidate_index=candidate_index, completed_roots=index + 1,
                        total_roots=len(roots), completed_trials=len(trials), split='train'))
                trial = dict(pass_index=iteration, layer=layer, scale=float(candidate), weighted_float_mse=loss)
                c.immutable(directory / 'RESULT.json', trial)
                trials.append(trial)
                choices.append((loss, float(candidate), proposed))
            scales = min(choices, key=lambda item: (item[0], item[1]))[2]
    return scales, dict(split='train', weight_bits=bits, passes=2, percentiles=[95, 99, 100],
                       trials=trials, selected={k: float(v) for k, v in scales.items()})
