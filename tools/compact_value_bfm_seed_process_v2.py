#!/usr/bin/env python3
"""Opt-in, source-bound spawn workers for future v2 training phases.

No existing phase is opted in by this module. Production use requires the
separate real-smoke equivalence and memory measurements described in the runbook.
Only the coordinator exports sources and appends aggregate/ledger receipts.
"""
from __future__ import annotations

import concurrent.futures
import copy
import hashlib
import multiprocessing
import os
from pathlib import Path
import resource
import sys

# Keep this module free of numerical imports. A spawned interpreter inherits
# these settings before importing its main module, not merely at initialization.
ENVIRONMENT = {name: '1' for name in ('MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
    'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}
MARKER = 'PAPERSOCCER_COMPACT_TRAINING_THREADS_FIXED_BEFORE_NUMPY'
MODE = {'mode': 'spawn-v2', 'maximum_workers': 2}
MODE2 = MODE
MODE4 = {'mode': 'spawn-v2', 'maximum_workers': 4}
_WORKER = None


def _modules():
    from tools import compact_value_bfm_campaign_v2 as campaign
    from tools import compact_value_bfm_train as trainer
    return campaign, trainer


def normalize_executor(setting):
    if not isinstance(setting, dict) or type(setting.get('maximum_workers')) is not int or setting not in (MODE2, MODE4):
        raise ValueError('unrecognized frozen training executor')
    return dict(setting)


def executor_mode(plan):
    setting = plan.get('training_executor')
    if setting is None:
        return 'threads'
    setting = normalize_executor(setting)
    if setting == MODE4:
        from tools import compact_value_bfm_training_resources_v2 as resources
        if resources.expected_workers(plan) != 4:
            raise ValueError('four training workers lack their source-bound resource authorization')
    return 'spawn-v2'


def filter_early_anchor(anchor, rows):
    """Reproduce the original filter, including order and dataset provenance."""
    campaign, trainer = _modules()
    import numpy as np
    active = set()
    for row in rows:
        if row['split'] != 'train' or row['drawn_edges'] > 6:
            continue
        state = campaign.features.ReplayState()
        if row['prefix']:
            for action in row['prefix'].split('/'):
                campaign.features.apply_complete_turn(state, state.to_move, action)
        for rotate in (False, True):
            for reflect in (False, True):
                transformed = campaign.openings.transform_state(
                    state, rotate=rotate, reflect=reflect)
                active.add(tuple(campaign.features.encode_active(transformed)))
    keep, removed = [], []
    for index in range(len(anchor)):
        (removed if tuple(anchor.active_row(index)) in active else keep).append(index)
    evidence = {'original_rows': len(anchor), 'removed_rows': len(removed),
        'kept_indices_sha256': trainer._array_identity(np.asarray(keep, dtype='<i8')),
        'removed_indices_sha256': trainer._array_identity(np.asarray(removed, dtype='<i8')),
        'early_active_sha256': hashlib.sha256(campaign.raw(
            [list(map(int, row)) for row in sorted(active)])).hexdigest()}
    if removed:
        if not keep:
            raise ValueError('early overlap filter removed the whole anchor pool')
        selected = [anchor.active_row(index) for index in keep]
        indptr = np.zeros(len(selected) + 1, dtype=np.int64)
        indptr[1:] = np.cumsum([len(row) for row in selected])
        # Match the original v2 constructor exactly; core inputs have no sidecar.
        if anchor.teacher_predictions is not None:
            raise ValueError('v2 core anchor unexpectedly has teacher sidecars')
        anchor = trainer.Dataset(indptr, np.concatenate(selected), anchor.targets[keep],
            anchor.weights[keep], anchor.group_ids[keep], anchor.split,
            anchor.source_manifest_sha256, anchor.source_npz_sha256, anchor.source_route)
    return anchor, evidence


def input_identity(inputs, anchor_filter):
    campaign, trainer = _modules()
    labels = inputs.successor_rankings
    if labels is None:
        raise ValueError('spawn v2 requires exhaustive successor rankings')
    ordering = {}
    for split in ('train', 'validation'):
        groups = getattr(labels, split)
        ordering[split] = {'groups': len(groups), 'sha256': hashlib.sha256(campaign.raw([
            [group.group_id, group.parent_mover, len(group.successors), group.successors_exhaustive]
            for group in groups])).hexdigest()}
    body = {'datasets': {name: trainer.dataset_identity(getattr(inputs, name))
            for name in ('new', 'anchor', 'common_adjudicator', 'canonical_validation')},
        'source_routes': {key: list(value) for key, value in inputs.source_routes.items()},
        'paired_row_validation': inputs.paired_row_validation,
        'split_isolation': inputs.split_isolation, 'input_audit': inputs.input_audit,
        'anchor_filter': anchor_filter, 'ranking': {'artifact_sha256': labels.artifact_sha256,
            'body_sha256': labels.body_sha256, 'schema': labels.artifact_schema,
            'source_bundle_body_sha256': labels.source_bundle_body_sha256,
            'teacher': dict(labels.teacher), 'ordered_groups': ordering}}
    return {**body, 'body_sha256': hashlib.sha256(campaign.raw(body)).hexdigest()}


def reconstruct_inputs(root, phase):
    """Read only audited artifacts; fresh scalar validation is never canonical."""
    campaign, trainer = _modules()
    from tools import compact_value_bfm_teacher_training as adapter
    from tools import compact_value_bfm_ranking_store as ranking_store
    root = Path(root)
    plan = campaign.read(root / 'campaign.json')
    audit = campaign.read(root / phase / 'training-input-audit.json')
    if (audit['schema'] != campaign.ID + '.training-input-audit.v2'
            or audit['bundle'] != plan['bundle'] or audit['protected_tests_opened'] is not False):
        raise ValueError('training reconstruction audit changed')
    bundle = trainer.FrozenBundle.load(campaign.verify(audit['bundle']))
    campaign.verify(audit['exclusion_index'])
    labels_path = campaign.verify(audit['labels'])
    positions_path = campaign.verify(audit['position_closure'])
    if labels_path != (root / phase / 'labels.json').resolve() or positions_path != (root / phase / 'positions.json').resolve():
        raise ValueError('training reconstruction phase paths changed')
    labels = campaign.read(labels_path)
    if campaign.verify(labels['positions']) != positions_path:
        raise ValueError('label position binding changed')
    index = campaign.verify(audit['ranking_store'])
    if campaign.read(index)['sources'] != [labels['merged']]:
        raise ValueError('audited ranking source changed')
    rankings = ranking_store.RankingStore(index, bundle).labels()
    shard = audit['shards']['train']
    manifest, npz = campaign.verify(shard['manifest']), campaign.verify(shard['npz'])
    new = trainer.load_shard(adapter._ExternalShardView(manifest, npz), manifest.name)
    if new.split != 'train':
        raise ValueError('audited new scalar shard is not training data')
    anchor, common, canonical_validation, routes = adapter._load_core_inputs(bundle)
    anchor, filtering = filter_early_anchor(anchor, campaign.read(positions_path)['rows'])
    if filtering['removed_rows'] != audit['anchor_duplicates_removed']:
        raise ValueError('reconstructed anchor filtering changed')
    inputs = trainer.TrainingInputs(new=new, anchor=anchor, common_adjudicator=common,
        canonical_validation=canonical_validation,
        source_routes={**routes, 'new': (shard['manifest']['path'],)},
        paired_row_validation={'external_source_bound': True},
        split_isolation={'closure_audit': audit['body_sha256']}, input_audit=audit,
        successor_rankings=rankings)
    for dataset in (new, anchor, common, canonical_validation):
        for name in ('indptr', 'indices', 'targets', 'weights', 'group_ids'):
            getattr(dataset, name).flags.writeable = False
    return bundle, inputs, input_identity(inputs, filtering)


def source_closure():
    campaign, _trainer = _modules()
    # Freeze the Python tools directory as a superset of all local imports,
    # including dynamically loaded helpers. Production runs use git snapshots.
    return [campaign.record(path) for path in sorted((campaign.REPO / 'tools').glob('*.py'))]


def roster_binding(base, seed, weight):
    """Only these two fields vary in the maintained v1 successor binding.

    Deriving the roster avoids rescanning every mapped successor nine times in
    the coordinator. Tests compare every entry with training_binding itself;
    each actual seed still recomputes its full binding in the unchanged trainer.
    """
    _campaign, trainer = _modules()
    weight = trainer._ranking_weight(weight)
    body = copy.deepcopy(base)
    body.pop('body_sha256')
    if (body['schema'] != 'papersoccer.compact-value-bfm-training-binding.v1'
            or seed not in trainer.FIXED_SEEDS):
        raise ValueError('spawn binding roster changed')
    body['seed'] = seed
    body['successor_ranking']['loss_weight'] = weight
    return trainer.body_hashed(body)


def freeze_spec(root, phase, bundle, inputs, anchor_filter, ranking_weights, seeds,
                *, qat_profile='standard-v1'):
    campaign, trainer = _modules()
    from tools import compact_value_bfm_intervention_v2 as intervention
    root = Path(root).resolve()
    plan = campaign.read(root / 'campaign.json')
    if executor_mode(plan) != 'spawn-v2':
        raise ValueError('spawn execution was not frozen in this phase')
    profile = trainer.resolve_qat_profile(qat_profile)
    executor = normalize_executor(plan['training_executor'])
    if profile.name != intervention.expected_qat_profile(plan):
        raise ValueError('spawn QAT profile differs from frozen phase')
    if (tuple(ranking_weights) != tuple(sorted(set(ranking_weights)))
            or 0.0 not in ranking_weights or not set(ranking_weights) <= {0.0, .10, .25}
            or tuple(seeds) not in (trainer.FIXED_SEEDS[:1], trainer.FIXED_SEEDS)):
        raise ValueError('spawn seed/lambda roster changed')
    architecture, arm = trainer.ARCHITECTURES['capacity-12x8'], trainer.ARMS['search-target']
    initial = campaign.verify(plan['inputs']['attempt_one_initial_checkpoint'])
    base = trainer.training_binding(bundle, inputs, architecture, arm, seeds[0],
        None, ranking_weights[0], initial, profile)
    jobs = []
    for weight in ranking_weights:
        for seed in seeds:
            jobs.append({'weight': weight, 'seed': seed,
                'directory': str(root / phase / 'training' / f'lambda-{weight:.2f}'),
                'binding': roster_binding(base, seed, weight)})
    path = root / phase / 'seed-process-spec.json'
    resource_binding = {} if executor == MODE2 else {
        'training_resource_authorization': plan['training_resource_authorization']}
    campaign.seal(path, {'schema': campaign.ID + '.seed-process-spec.v2',
        'phase_contract': campaign.record(root / 'campaign.json'), 'phase': phase,
        'input_audit': campaign.record(root / phase / 'training-input-audit.json'),
        'initial_checkpoint': campaign.record(initial), 'executor': executor,
        'qat_profile': profile.name, 'qat_profile_contract': trainer.qat_profile_contract(profile),
        'sources': source_closure(), 'python': {'executable': str(Path(sys.executable).resolve()),
            'version': sys.version}, 'reconstruction': input_identity(inputs, anchor_filter),
        'ranking_weights': list(ranking_weights), 'seeds': list(seeds), 'jobs': jobs, **resource_binding})
    return path


def seed_thread(function, *arguments):
    """Each child owns its limiter outside its one-seed thread roster."""
    _campaign, trainer = _modules()
    with trainer.native_thread_execution_scope() as execution:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(function, *arguments, execution).result()


def _initialize(spec_path):
    global _WORKER
    campaign, trainer = _modules()
    from tools import compact_value_bfm_intervention_v2 as intervention
    spec = campaign.read(spec_path)
    if spec['schema'] != campaign.ID + '.seed-process-spec.v2':
        raise ValueError('spawn worker specification changed')
    executor = normalize_executor(spec['executor'])
    if source_closure() != spec['sources']:
        raise ValueError('spawn worker source closure changed')
    if spec['python'] != {'executable': str(Path(sys.executable).resolve()), 'version': sys.version}:
        raise ValueError('spawn worker Python runtime changed')
    root = campaign.verify(spec['phase_contract']).parent
    plan = campaign.read(root / 'campaign.json')
    if executor_mode(plan) != 'spawn-v2' or normalize_executor(plan['training_executor']) != executor:
        raise ValueError('spawn worker phase contract changed')
    if executor == MODE4 and spec.get('training_resource_authorization') != plan['training_resource_authorization']:
        raise ValueError('spawn worker resource authorization changed')
    if campaign.verify(spec['input_audit']) != root / spec['phase'] / 'training-input-audit.json':
        raise ValueError('spawn worker audit route changed')
    trainer.validate_qat_profile_contract(spec['qat_profile_contract'], expected_name=spec['qat_profile'])
    if (spec['qat_profile'] != intervention.expected_qat_profile(plan)
            or spec['initial_checkpoint'] != plan['inputs']['attempt_one_initial_checkpoint']):
        raise ValueError('spawn worker initialization/profile changed')
    with trainer.native_thread_execution_scope():
        bundle, inputs, identity = reconstruct_inputs(root, spec['phase'])
    if identity != spec['reconstruction']:
        raise ValueError('spawn worker input reconstruction differs from coordinator')
    initial = campaign.verify(spec['initial_checkpoint'])
    weights, seeds = spec['ranking_weights'], spec['seeds']
    if (weights != sorted(set(weights)) or 0.0 not in weights or not set(weights) <= {0.0, .1, .25}
            or tuple(seeds) not in (trainer.FIXED_SEEDS[:1], trainer.FIXED_SEEDS)):
        raise ValueError('spawn worker roster changed')
    architecture, arm = trainer.ARCHITECTURES['capacity-12x8'], trainer.ARMS['search-target']
    with trainer.native_thread_execution_scope():
        base = trainer.training_binding(bundle, inputs, architecture, arm, seeds[0],
            None, weights[0], initial, spec['qat_profile'])
    expected_jobs = [{'weight': weight, 'seed': seed,
        'directory': str(root / spec['phase'] / 'training' / f'lambda-{weight:.2f}'),
        'binding': roster_binding(base, seed, weight)} for weight in weights for seed in seeds]
    if expected_jobs != spec['jobs']:
        raise ValueError('spawn worker seed bindings changed')
    _WORKER = (spec, bundle, inputs, initial)


def _train(job, execution):
    campaign, trainer = _modules()
    spec, bundle, inputs, initial = _WORKER
    architecture, arm = trainer.ARCHITECTURES['capacity-12x8'], trainer.ARMS['search-target']
    binding = job['binding']
    directory = Path(job['directory'])
    receipt = trainer.train_seed_candidate(bundle, inputs, architecture, arm, job['seed'],
        directory, ranking_weight=job['weight'], initial_checkpoint=initial,
        qat_profile=spec['qat_profile'], resume=True, _native_thread_execution=execution)
    reference = trainer._seed_reference_path(directory, architecture, arm, job['seed'])
    if trainer._load_seed_receipt_from_reference(directory, reference, binding) != receipt:
        raise ValueError('spawn seed reference differs from returned receipt')
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {'reference': campaign.record(reference), 'binding_sha256': binding['body_sha256'],
        'process': {'pid': os.getpid(), 'peak_rss': usage.ru_maxrss,
            'peak_rss_units': 'bytes' if sys.platform == 'darwin' else 'KiB',
            'minor_page_faults': usage.ru_minflt, 'major_page_faults': usage.ru_majflt},
        'native_thread_execution': execution}


def _run(job):
    if _WORKER is None or job not in _WORKER[0]['jobs']:
        raise ValueError('seed is outside frozen worker roster')
    return seed_thread(_train, job)


class SpawnSeedExecutor:
    """Persistent spawn children with a frozen roster; no child replacement."""
    def __init__(self, spec_path):
        campaign, _trainer = _modules()
        self.path = Path(spec_path)
        self.spec = campaign.read(self.path)
        self.pool = None
        self.ordinal = 0
        self.evidence = []

    @property
    def settings(self):
        return normalize_executor(self.spec.get('executor', MODE2))

    def __enter__(self):
        campaign, _trainer = _modules()
        if self.settings == MODE4:
            plan = campaign.read(campaign.verify(self.spec['phase_contract']))
            if executor_mode(plan) != 'spawn-v2' or plan['training_executor'] != MODE4:
                raise ValueError('four-worker roster does not match its authorized phase')
            if self.spec.get('training_resource_authorization') != plan['training_resource_authorization']:
                raise ValueError('four-worker roster resource authorization changed')
        # Spawn imports __main__ before initializer; set these before start.
        os.environ.update(ENVIRONMENT)
        os.environ[MARKER] = '1'
        self.pool = concurrent.futures.ProcessPoolExecutor(max_workers=self.settings['maximum_workers'],
            mp_context=multiprocessing.get_context('spawn'), initializer=_initialize,
            initargs=(str(self.path),))
        return self

    def run_weight(self, weight):
        if self.settings != MODE2:
            raise ValueError('four-worker execution requires the flattened run_roster entry point')
        if (self.ordinal >= len(self.spec['ranking_weights'])
                or weight != self.spec['ranking_weights'][self.ordinal]):
            raise ValueError('spawn lambda execution order changed')
        jobs = [job for job in self.spec['jobs'] if job['weight'] == weight]
        returned = list(self.pool.map(_run, jobs))
        receipts = self._collect(jobs, returned)
        self.ordinal += 1
        return receipts

    def run_roster(self):
        if self.settings != MODE4 or self.ordinal != 0:
            raise ValueError('flattened four-worker roster can execute exactly once')
        self.ordinal = len(self.spec['ranking_weights'])
        jobs = self.spec['jobs']
        returned = list(self.pool.map(_run, jobs))
        return self._collect(jobs, returned)

    def _collect(self, jobs, returned):
        campaign, trainer = _modules()
        receipts = []
        for job, result in zip(jobs, returned, strict=True):
            reference = campaign.verify(result['reference'])
            expected = trainer._seed_reference_path(Path(job['directory']),
                trainer.ARCHITECTURES['capacity-12x8'], trainer.ARMS['search-target'], job['seed'])
            if (reference != expected or result['binding_sha256'] != job['binding']['body_sha256']):
                raise ValueError('spawn result seed reference changed')
            receipt = trainer._load_seed_receipt_from_reference(Path(job['directory']), reference, job['binding'])
            if receipt['native_thread_execution'] != result['native_thread_execution']:
                raise ValueError('spawn result numerical limiter evidence changed')
            receipts.append(receipt)
            self.evidence.append({'weight': job['weight'], 'seed': job['seed'], **result})
        return receipts

    def __exit__(self, *_exception):
        # ProcessPoolExecutor marks a crashed pool broken, never retries jobs.
        # Waiting here joins every child before the caller can create a new pool.
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.pool = None
