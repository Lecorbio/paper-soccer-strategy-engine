"""Bound validation predictions while preserving exact full-group/root scoring."""
import gc
import gzip
import hashlib
from pathlib import Path

from tools import rank_two_live_eight_group_stream_v2 as stream
from tools import rank_two_live_eight_training_v5 as training

c, np = training.campaign, training.np


def root_metrics(corpus, root, parameters, architecture, bits, scales, native, maximum):
    quantized = None if scales is None else training.model.quantize(parameters, scales, bits)
    groups, targets, signs, predictions, spans = [], [], [], [], []
    offset = 0
    for item in corpus.roots['validation'][root]:
        local_target, local_sign, local_prediction = [], [], []
        metadata, batch = None, []

        def evaluate(meta, children):
            active, target, sign, _ = training.retained.root_rows([{**meta, 'successors': children}])
            prediction = training.retained.forward(parameters, architecture, active, quantized)[0]
            if native is not None:
                actual = np.asarray(native(root, active), dtype=np.float32)
                if actual.shape != prediction.shape or not np.array_equal(actual, prediction):
                    raise ValueError('native/reference float predictions differ')
                prediction = actual
            local_target.append(target)
            local_sign.append(sign)
            local_prediction.append(prediction)

        with gzip.open(training.reader.bound(item['artifact']), 'rt') as source:
            for metadata, child in stream.successors(source, root, item['group_id'], 'validation'):
                batch.append(child)
                if len(batch) == maximum:
                    evaluate(metadata, batch)
                    batch = []
            if batch:
                evaluate(metadata, batch)
        if not local_target:
            raise ValueError('root contains an empty action group')
        target = np.concatenate(local_target)
        sign = np.concatenate(local_sign)
        prediction = np.concatenate(local_prediction)
        groups.append(dict(exhaustive=metadata['exhaustive'], edges=metadata['edges']))
        spans.append(slice(offset, offset + len(target)))
        offset += len(target)
        targets.append(target)
        signs.append(sign)
        predictions.append(prediction)
        del batch, child, metadata, local_target, local_sign, local_prediction
        gc.collect()
    target, sign, prediction = map(np.concatenate, (targets, signs, predictions))
    huber, ranking, _ = training.retained.objective(groups, prediction, target, sign, spans)
    regrets, early = [], False
    for group, span in zip(groups, spans, strict=True):
        teacher = target[span] * sign[span]
        if group['exhaustive'] and len(teacher) > 1 and float(np.max(teacher) - np.min(teacher)) > 1e-4:
            picked = int(np.argmax(prediction[span] * sign[span]))
            regrets.append(float(np.max(teacher) - teacher[picked]))
            early |= group['edges'] <= 12
    return dict(root=root, comparable=bool(regrets), early=early,
        regret=sum(regrets) / len(regrets) if regrets else None, huber=huber, ranking=ranking)


def validation(corpus, parameters, architecture, bits, scales=None, native=None,
               output=None, binding=None, boundary=lambda: None, maximum=2048):
    if type(maximum) is not int or not 1 <= maximum <= 2048:
        raise ValueError('prediction/native request batch must be bounded by2048')
    directory = None if output is None else Path(output)
    root_ids = sorted(corpus.roots['validation'])
    recipe = None
    if directory is not None:
        if binding is None:
            raise PermissionError('durable validation requires exact epoch/plan binding')
        recipe = c.immutable(directory / 'BINDING.json', dict(binding=binding,
            corpus=corpus.identity, parameter_sha256=hashlib.sha256(
                b''.join(parameters[k].tobytes() for k in ('w1', 'w2', 'w3'))).hexdigest(),
            bits=bits, scales=None if scales is None else {k: float(v) for k, v in scales.items()},
            root_ids=root_ids, producer=c.record(__file__), partition_reader=c.record(stream.__file__),
            maximum_prediction_rows=maximum, scoring='unchanged complete group/root reductions'))
    rows, early = [], 0
    for index, root in enumerate(root_ids):
        boundary()
        path = None if directory is None else directory / ('root-%04d.json' % index)
        if path is not None and path.exists():
            saved = c.read(path)
            if saved['recipe'] != recipe or saved['metrics']['root'] != root:
                raise ValueError('retained validation root binding differs')
            metrics = saved['metrics']
        else:
            metrics = root_metrics(corpus, root, parameters, architecture, bits, scales, native, maximum)
            if path is not None:
                c.immutable(path, dict(recipe=recipe, metrics=metrics, native_parity_checked=native is not None))
        early += metrics['early']
        if metrics['comparable']:
            rows.append({key: metrics[key] for key in ('root', 'regret', 'huber', 'ranking')})
        gc.collect()
    accounting = training.validation_accounting(corpus, {row['root'] for row in rows}, early)
    result = dict(regret=sum(row['regret'] for row in rows) / len(rows),
        huber=sum(row['huber'] for row in rows) / len(rows), early_roots=early,
        roots=rows, validation_root_accounting=accounting)
    if directory is not None:
        c.immutable(directory / 'RESULT.json', result)
    return result
