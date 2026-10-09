"""Freeze the four matched arms after exact shared-data/ancestry admission."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

from tools import rank_two_focused_campaign_v1 as campaign
from tools import rank_two_focused_checkpoint_v1 as storage
from tools import rank_two_focused_training_v5 as trainer
from tools import rank_two_focused_export_v2 as exporter
from tools import rank_two_focused_native_v3 as native
from tools import rank_two_focused_labels_v3 as labels
from tools import rank_two_focused_prior_v1 as prior
from tools import rank_two_focused_work_v1 as work
from tools import compact_representation_train as reader
from tools import jacek_replay_features as features

SCHEMA = campaign.SCHEMA + '.matched-training-plans.v1'
BUDGET = dict(workers=3, rss_bytes=8 * 1024**3, numerical_threads=1,
              wall_seconds=8 * 3600, cpu_capacity_seconds=145000)
ORDER = [('dd8', 2026100701), ('dd12wide', 2026100701),
         ('dd8', 2026100702), ('dd12wide', 2026100702)]


def freeze(base, owner, usage_path, output):
    base, output = Path(base).resolve(), Path(output).resolve()
    state = campaign.read(base / 'CURRENT.json')
    if (state.get('ownership_status') != 'active' or state.get('owner_thread_id') != owner
            or state.get('active_jobs') or state.get('active_helper') or state.get('active_claims')
            or state.get('pending_exposures')):
        raise PermissionError('sole idle owner and closed exposure ledger required before plan freeze')
    spec = importlib.util.spec_from_file_location('focused_plan_guard', campaign.verify(state['control_guard']))
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    launch_guard = guard.guard(base, 'freeze', owner, usage_path, budget=BUDGET)
    approved = campaign.read(campaign.verify(state['focused_plan']))
    if (approved['plan']['recipe'] != campaign.RECIPE or approved['plan']['families'] != campaign.FAMILIES
            or approved['plan']['seeds'] != campaign.SEEDS):
        raise ValueError('the four-arm declaration differs from the human-approved comparison')
    if state.get('focused_initial_training_plans'):
        previous = campaign.read(campaign.verify(state['focused_initial_training_plans']))
        if previous['producer'] != campaign.record(__file__):
            raise ValueError('existing training-plan producer differs')
        for reference in previous['plans']:
            campaign.verify(reference)
        return previous
    implementation = campaign.read(campaign.verify(state['focused_storage_implementation']))
    if (implementation.get('passed') is not True
            or implementation['checkpoint_producer'] != campaign.record(storage.__file__)
            or implementation['training_producer'] != campaign.record(trainer.__file__)):
        raise ValueError('verified lossless training implementation required')
    corpus_archive = campaign.read(campaign.verify(state['focused_corpus_complete']))
    if (corpus_archive.get('passed') is not True or corpus_archive['early_validation_roots'] < 100
            or corpus_archive['train_roots'] != 128 or corpus_archive['validation_roots'] != 128):
        raise ValueError('shared teacher-only corpus admission differs')
    ancestry = campaign.read(campaign.verify(state['focused_ancestry_ready']))
    if ancestry.get('passed') is not True:
        raise ValueError('completed incremental ancestry required')
    with prior.prior_census(campaign.read(campaign.verify(ancestry['prior']))):
        pass
    teacher_plan = campaign.read(campaign.verify(state['focused_teacher_plan']))
    teacher_archive = campaign.read(campaign.verify(state['focused_teacher_complete']))
    label_execution = teacher_archive['execution']
    if 'bytes' in label_execution:
        labels.data.bound(label_execution)
        label_execution = campaign.record(label_execution['path'])
    corpus = reader.Corpus(campaign.verify(corpus_archive['corpus']))
    template = dict(schema=trainer.SCHEMA + '.plan', producer=campaign.record(trainer.__file__),
        checkpoint_storage=campaign.record(storage.__file__),
        work_guard=campaign.record(work.__file__), feature_schema=campaign.record(features.__file__),
        recipe=campaign.RECIPE, initialization='fresh-common-gaussian-method; no incumbent truncation',
        parent_checkpoint=None, numpy_version=trainer.np.__version__,
        python=campaign.record(Path(sys.executable).resolve()),
        generation_plan=state['focused_generation_plan'], label_plan=state['focused_teacher_plan'],
        label_execution=label_execution, label_producer=campaign.record(labels.__file__),
        teacher=teacher_plan['teacher'], corpus=corpus_archive['corpus'],
        ancestry_ready=state['focused_ancestry_ready'], activation=state['focused_activation'],
        owner_thread_id=owner, campaign_current=str(base / 'CURRENT.json'),
        exporter=campaign.record(exporter.__file__), native_validation=campaign.record(native.__file__),
        incumbent_source=approved['incumbent'], roster_manifest=approved['roster_manifest'],
        clocks_ms=[550, 140], resource_budget=BUDGET,
        float_selection=['root_balanced_validation_objective', 'earliest_epoch'],
        exported_selection=['native_regret', 'teacher_huber', 'earliest_source_valid_epoch'],
        source_limits=dict(nonpayload=43000, stress=98000, hard=100000),
        setup_admission=state['focused_setup_ready'], storage_admission=state['focused_storage_implementation'],
        experimental_live_source=False, outcomes_as_targets=False)
    trainer.validate_corpus_binding(template, corpus)
    output.mkdir(parents=True, exist_ok=True)
    campaign.immutable(output / 'FREEZE_GUARD.json', launch_guard)
    plans = []
    for profile, seed in ORDER:
        directory = output / (profile + '-seed-' + str(seed))
        plan = dict(template, profile=profile, shape=campaign.FAMILIES[profile], seed=seed,
                    output=str(directory / 'execution'))
        plans.append(campaign.immutable(directory / 'PLAN.json', plan))
    bundle = dict(schema=SCHEMA, producer=campaign.record(__file__), passed=True,
        plans=plans, order=[dict(profile=profile, seed=seed) for profile, seed in ORDER],
        corpus=template['corpus'], ancestry_ready=template['ancestry_ready'], recipe=campaign.RECIPE,
        activation=state['focused_activation'], owner_thread_id=owner, usage=campaign.record(usage_path),
        same_teacher_data_optimizer_root_order_selection=True, fresh_initialization=True,
        new_games=0, new_training_runs=0)
    reference = campaign.immutable(output / 'PLANSET.json', bundle)
    with campaign.locked(base):
        current = campaign.read(base / 'CURRENT.json')
        if current.get('focused_initial_training_plans') or current['owner_thread_id'] != owner:
            raise RuntimeError('plan-freeze ownership changed; immutable plans retained')
        current['focused_initial_training_plans'] = reference
        current['next_action'] = 'launch the first of four matched arms with fresh usage/source/resource guard'
        campaign.atomic(base / 'CURRENT.json', current)
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--owner', required=True)
    parser.add_argument('--usage', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = freeze(args.base, args.owner, args.usage, args.output)
    import json
    print(json.dumps(dict(passed=result['passed'], plans=len(result['plans']))))


if __name__ == '__main__':
    main()
