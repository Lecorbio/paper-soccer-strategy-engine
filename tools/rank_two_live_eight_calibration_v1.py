"""Retain each completed TRAIN calibration root under an exact numerical binding."""
import hashlib
import json
from pathlib import Path

from tools import rank_two_live_eight_training_v5 as training

c, np = training.campaign, training.np


def calibration(parameters, architecture, corpus, roots, bits, masses, output, binding, boundary=lambda: None):
    output = Path(output)
    digest = hashlib.sha256(b''.join(parameters[k].tobytes() for k in ('w1', 'w2', 'w3'))).hexdigest()
    c.immutable(output / 'BINDING.json', dict(plan=binding, parameter_sha256=digest,
        corpus=corpus.identity, roots=roots, masses=masses, bits=bits, split='train',
        numerical_threads=1, producer=c.record(__file__)))
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
                        groups = corpus.root('train', root)
                        if any(g['split'] != 'train' for g in groups):
                            raise PermissionError('TRAIN-only scale calibration required')
                        active, *_ = training.retained.root_rows(groups)
                        target = training.retained.forward(parameters, architecture, active)[0]
                        prediction = training.retained.forward(parameters, architecture, active, q)[0]
                        contribution = masses[root] * float(np.mean((prediction - target) ** 2)) / len(roots)
                        c.immutable(path, dict(recipe=recipe, root=root, weighted_float_mse=contribution))
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
