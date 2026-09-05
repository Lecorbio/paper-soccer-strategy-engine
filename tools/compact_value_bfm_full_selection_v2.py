#!/usr/bin/env python3
"""Freeze full-round model selection from six completed, bound training seeds.

This command performs no training or game execution. Full-round ranking metrics
are diagnostics: the pilot's regret-improvement requirement is not applied a
second time. Actual multi-opponent and final qualification remain mandatory.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from pathlib import Path
import sys

if __name__ == '__main__':
    for key in ('MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'OMP_NUM_THREADS',
                'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[key] = '1'
    os.environ['PAPERSOCCER_COMPACT_TRAINING_THREADS_FIXED_BEFORE_NUMPY'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import compact_value_bfm_campaign_v2 as campaign
from tools import compact_value_bfm_full_v2 as full
from tools import compact_value_bfm_pilot_selection_v2 as pilot_selection
from tools import compact_value_bfm_ranking_store as storage
from tools import compact_value_bfm_teacher_training as selection
from tools import compact_value_bfm_train as trainer


def validate_context(root, context, phase):
    """Reopen actual pilot admission and the unchanged original initialization."""
    root, context = Path(root).resolve(), Path(context).resolve()
    contract = campaign.read(context / 'campaign.json')
    parent_path = campaign.verify(contract['parent_campaign'])
    if parent_path.resolve() != root / 'campaign.json' or Path(contract['heavy_stage_root']).resolve() != root:
        raise ValueError('full selection parent campaign changed')
    parent = campaign.read(parent_path)
    attempt = contract['attempt']
    expected_phase = f'attempt-{attempt:03d}-full'
    if (isinstance(attempt, bool) or attempt < 1 or phase != expected_phase
            or context != root / 'phases' / expected_phase or contract.get('phase') != 'full'
            or contract.get('full_games') != 10000 or contract.get('policy') != parent['policy']
            or parent['policy'] != campaign.POLICY):
        raise ValueError('full selection context or phase changed')
    pilot_context_path = campaign.verify(contract['pilot_context'])
    pilot_context = pilot_context_path.parent
    pilot_phase = f'attempt-{attempt:03d}-pilot'
    pilot_contract = campaign.read(pilot_context_path)
    if (pilot_context_path.resolve() != root / 'phases' / pilot_phase / 'campaign.json'
            or pilot_contract.get('attempt') != attempt or pilot_contract.get('phase') != 'pilot'
            or pilot_contract.get('parent_campaign') != contract['parent_campaign']):
        raise ValueError('full selection pilot context changed')
    outcome_path = campaign.verify(contract['admitted_pilot'])
    if outcome_path.resolve() != pilot_context / pilot_phase / 'pilot-outcome.json':
        raise ValueError('full selection admitted pilot path changed')
    outcome = full.admitted_pilot(pilot_context, pilot_phase)
    selected = outcome['selected']
    weight = selected['lambda']
    roster = {'lambdas': [0, weight], 'seeds': list(trainer.FIXED_SEEDS)}
    if weight not in (.1, .25) or contract.get('full_training_roster') != roster:
        raise ValueError('full selection must use scalar and admitted ranking recipe')
    expected_inputs = {**pilot_contract['inputs'], 'attempt_zero_runtime': selected['runtime']}
    initial = parent['inputs']['attempt_one_initial_checkpoint']
    if (contract['inputs'] != expected_inputs
            or pilot_contract['inputs']['attempt_one_initial_checkpoint'] != initial
            or contract.get('bundle') != parent['bundle'] or pilot_contract.get('bundle') != parent['bundle']
            or contract.get('candidate_lineage') != {
                'mandatory_training': True, 'initial_float': initial,
                'generation_student': selected['runtime'], 'pilot_source': selected['source'],
                'smoke_weights_reused': False}):
        raise ValueError('full selection lost its original float or admitted student lineage')
    for key in ('attempt_one_initial_checkpoint', 'teacher_runtime', 'teacher_manifest'):
        if contract['inputs'][key] != parent['inputs'][key]:
            raise ValueError('full selection initial checkpoint or accepted teacher changed')
        campaign.verify(contract['inputs'][key])
    for item in (*pilot_contract['exclusions'], *outcome['development_exclusions']):
        if item not in contract['exclusions']:
            raise ValueError('full selection lost accumulated exclusions')
    for item in contract['exclusions']:
        campaign.verify(item)
    return contract, parent


def policy(contract):
    return {
        'schema': campaign.ID + '.full-selection-policy.v2',
        **contract['full_training_roster'],
        'seed_selection': pilot_selection.SELECTION_POLICY['seed_selection'],
        'canonical_retention': 'maintained-offline-advancement-gate',
        'source_limit_exclusive': 95000, 'source_reserve_target': 2000,
        'early_max_drawn_edges': 12, 'minimum_comparable_groups': 100,
        'minimum_comparable_fraction': .8, 'early_uses_same_coverage_floors': True,
        'ranking_metrics': 'diagnostic-only;pilot-regret-and-flip-gate-not-repeated',
        'coverage': 'reported-per-stratum;insufficient-evidence-is-not-a-passing-metric',
        'advancement': 'selected-ranking-seed-passes-canonical-retention-and-source-reserve',
        'strength_control': 'immutable-original-deployed-source-and-runtime',
        'scalar_control': 'fresh-full-round-scalar-model-selected-by-maintained-validation-key',
        'multi_opponent_strength_and_final_qualification_required': True,
        'offline_selection_is_not_qualification': True,
    }


def prepare_policy(context, phase, contract):
    """Bind the selector, its validators and the exact maintained source exporter."""
    modules = (campaign, full, pilot_selection, storage, selection, trainer,
               selection.model_exporter, selection.source_exporter)
    files = {Path(__file__).resolve(), *(Path(module.__file__).resolve() for module in modules)}
    exporter = selection.source_exporter
    config = json.loads(exporter.CONFIG.read_text())
    manifest = exporter.contained(exporter.HERE, config.get('sources', 'sources.txt'), 'sources')
    files.update((exporter.CONFIG, manifest))
    files.update(exporter.contained(exporter.ROOT, line.strip(), 'source')
                 for line in manifest.read_text().splitlines()
                 if line.strip() and not line.lstrip().startswith('#'))
    path = Path(context) / phase / 'full-selection-policy.json'
    campaign.seal(path, {**policy(contract), 'context': campaign.record(Path(context) / 'campaign.json'),
                         'source_closure': [campaign.record(p) for p in sorted(files)]})
    return path


def validate_roster(training, weights):
    records = training.get('results', [])
    expected = {(weight, seed) for weight in weights for seed in trainer.FIXED_SEEDS}
    actual = [(row['weight'], row['seed']) for row in records]
    if (training.get('schema') != campaign.ID + '.training.v2'
            or training.get('smoke') is not False or training.get('mandatory_training_verified') is not True
            or len(records) != 6 or len(set(actual)) != 6 or set(actual) != expected):
        raise ValueError('full selection requires all six real nonsmoke trainings exactly once')
    return records


def verify_source_export(row):
    source = campaign.verify(row['source']).read_bytes()
    source.decode('ascii')
    runtime = campaign.verify(row['runtime'])
    if source != selection._runtime_source(runtime):
        raise ValueError('full seed source does not reproduce its exact runtime export')
    reserve = 95000 - len(source)
    if reserve <= 0 or reserve != row.get('source_reserve'):
        raise ValueError('full seed source size evidence changed')


def verify_master_updates(row, initial, parameters, architecture, quantized):
    updates = trainer._parameter_update_evidence(initial, parameters)
    if (updates != row.get('master_updates') or set(updates) != {'w1', 'w2', 'w3'}
            or any(not value['changed'] or not math.isfinite(value['l2_delta']) or value['l2_delta'] <= 0
                   or value['before_sha256'] == value['after_sha256'] for value in updates.values())):
        raise ValueError('full seed lacks actual finite nonzero all-layer master updates')
    initial_quantized = trainer.quantize_fixed(initial, architecture, quantized.scales)
    changes = trainer._quantized_update_evidence(initial_quantized, quantized)
    if changes != row.get('quantized_changes_vs_initialization') or not any(v['changed_codes'] for v in changes.values()):
        raise ValueError('full seed quantized initialization was reused or changed')


def validate_seed_corpus_binding(binding, audit):
    new = binding.get('datasets', {}).get('new', {})
    shard = audit['shards']['train']
    settings = binding.get('settings', {})
    expected_settings = {'seeds': list(trainer.FIXED_SEEDS), 'batch_size': 256,
                         'new_rows_per_batch': 64, 'anchor_rows_per_batch': 192,
                         'new_loss_share': .25, 'anchor_loss_share': .75, 'qat_epochs': 4,
                         'qat_profile': trainer.STANDARD_QAT_PROFILE}
    if (new.get('source_manifest_sha256') != shard['manifest']['sha256']
            or new.get('source_npz_sha256') != shard['npz']['sha256']
            or binding.get('source_routes', {}).get('new') != [shard['manifest']['path']]
            or any(settings.get(key) != value for key, value in expected_settings.items())):
        raise ValueError('full seed new-shard or approved batch/QAT binding changed')


def validate_inputs(context, phase, contract, training, bundle):
    directory = Path(context) / phase
    audit_path = campaign.verify(training['input_audit'])
    if audit_path.resolve() != directory / 'training-input-audit.json':
        raise ValueError('full training input audit belongs to another phase')
    audit = campaign.read(audit_path)
    positions_path = directory / 'positions.json'
    labels_path = directory / 'labels.json'
    index_path = directory / 'ranking-store/index.json'
    if (audit.get('bundle') != contract['bundle'] or audit.get('protected_tests_opened') is not False
            or audit.get('position_closure') != campaign.record(positions_path)
            or audit.get('labels') != campaign.record(labels_path)
            or audit.get('ranking_store') != campaign.record(index_path)):
        raise ValueError('full training audit changed its corpus or bundle')
    labels = campaign.read(labels_path)
    plan = campaign.read(campaign.verify(labels['plan']))
    positions = campaign.read(positions_path)
    position_plan = campaign.read(campaign.verify(positions['plan']))
    games_path = campaign.verify(position_plan['games'])
    games = campaign.read(games_path)
    if (labels.get('all_groups_exhaustive') is not True or labels.get('all_native_labels_validated') is not True
            or labels.get('positions') != campaign.record(positions_path)
            or labels.get('teacher') != contract['inputs']['teacher_runtime']
            or plan.get('context') != campaign.record(Path(context) / 'campaign.json')
            or plan.get('phase') != phase or plan.get('bundle') != contract['bundle']
            or plan.get('teacher') != contract['inputs']['teacher_runtime']
            or plan.get('student') != contract['inputs']['attempt_zero_runtime']
            or (plan.get('shallow_nodes'), plan.get('deep_nodes'), plan.get('deep_fraction')) != (64000, 500000, .25)
            or positions.get('all_retained_groups_preflighted') is not True
            or position_plan.get('context') != campaign.record(Path(context) / 'campaign.json')
            or position_plan.get('phase') != phase or position_plan.get('validation_first') is not True
            or games_path.resolve() != directory / 'games.json' or games.get('games') != 10000
            or len(games.get('rows', [])) != 10000):
        raise ValueError('full selection requires the completed bound 10000-game native corpus')
    campaign.verify(audit['exclusion_index'])
    for shard in audit['shards'].values():
        for item in shard.values():
            campaign.verify(item)
    store = storage.RankingStore(index_path, bundle)
    if store.document['sources'] != [labels['merged']]:
        raise ValueError('full ranking store differs from completed native labels')
    rankings = store.labels()
    if len(rankings.train) + len(rankings.validation) != labels['groups']:
        raise ValueError('full ranking store lost native label groups')
    return audit, rankings


def validate_seed(context, phase, row, contract, audit, rankings, bundle, initial):
    """Validate real seed references and recompute all master/payload updates."""
    directory = Path(context) / phase / 'training' / f'lambda-{row["weight"]:.2f}'
    receipt = row['seed_receipt']
    binding = receipt['binding']
    trainer.verify_body_hash(binding, schema='papersoccer.compact-value-bfm-training-binding.v1', label='full seed binding')
    architecture = trainer.ARCHITECTURES['capacity-12x8']
    arm = trainer.ARMS['search-target']
    ranking = binding.get('successor_ranking', {})
    validate_seed_corpus_binding(binding, audit)
    initial_identity = trainer._parameter_identity(initial, architecture)
    if (binding.get('source_bundle_body_sha256') != bundle.body_sha256
            or binding.get('seed') != row['seed'] or binding.get('architecture') != {
                'name': architecture.name, 'dimensions': list(architecture.dimensions),
                'biases': False, 'activations': list(trainer.ACTIVATIONS)}
            or binding.get('arm') != dataclasses.asdict(arm)
            or binding.get('input_audit') != audit
            or binding.get('split_isolation') != {'closure_audit': audit['body_sha256']}
            or ranking.get('artifact_sha256') != rankings.artifact_sha256
            or ranking.get('body_sha256') != rankings.body_sha256
            or ranking.get('source_bundle_body_sha256') != bundle.body_sha256
            or ranking.get('teacher') != rankings.teacher
            or ranking.get('schema') != storage.SCHEMA
            or ranking.get('loss_weight') != row['weight']
            or ranking.get('train_groups') != len(rankings.train)
            or ranking.get('validation_groups') != len(rankings.validation)
            or ranking.get('skipped_nonexhaustive_groups') != 0
            or ranking.get('initial_checkpoint') != {
                **contract['inputs']['attempt_one_initial_checkpoint'],
                'parameters': initial_identity}
            or receipt.get('float_training', {}).get('initialization') != {
                'kind': 'frozen-float-checkpoint', 'seed_affects': 'row-order-only',
                'parameters': initial_identity}):
        raise ValueError('full seed training binding changed its corpus, architecture or original float')
    reference = trainer._seed_reference_path(directory, architecture, arm, row['seed'])
    loaded = trainer._load_seed_receipt_from_reference(directory, reference, binding)
    if loaded != receipt or receipt.get('protected_tests_opened') is not False:
        raise ValueError('full training row differs from its completed seed reference')
    for row_key, seed_key in (('runtime', 'quantized_runtime'), ('float_checkpoint', 'float_checkpoint')):
        item = receipt[seed_key]
        expected = trainer._output_artifact(directory, item['path'], expected_sha256=item['sha256'], label=seed_key)
        if campaign.verify(row[row_key]).resolve() != expected.resolve():
            raise ValueError('full seed row substituted a trained artifact')
    arch, quantized, runtime_selection, runtime_document = trainer.load_runtime(campaign.verify(row['runtime']))
    parameters = trainer.load_float_checkpoint(campaign.verify(row['float_checkpoint']), architecture)
    if arch != architecture or runtime_selection['source_bundle_body_sha256'] != bundle.body_sha256:
        raise ValueError('full seed runtime architecture or bundle changed')
    verify_master_updates(row, initial, parameters, architecture, quantized)
    if receipt['float_training'].get('per_layer_update_evidence') != row['master_updates']:
        raise ValueError('full seed warm-up update evidence differs from its checkpoint')
    verify_source_export(row)
    gate = trainer.offline_advancement_gate(receipt['float_validation'], receipt['quantized_validation'])
    if receipt.get('offline_gate') != gate:
        raise ValueError('full seed canonical retention verdict changed')
    selection._ranking_metrics(receipt)
    return {'reference': campaign.record(reference), 'runtime_body_sha256': runtime_document['body_sha256'],
            'payload_sha256': runtime_document['quantization']['payload_sha256'],
            'parameters': parameters, 'architecture': arch, 'quantized': quantized}


def ranking_diagnostic(control, candidate):
    """Coverage controls interpretation; it adds no new full admission gate."""
    result = {}
    for stratum in ('overall', 'early'):
        baseline, evaluated = control[stratum], candidate[stratum]
        # Reuse maintained type/range validation for both reports.
        for metrics in (baseline, evaluated):
            if metrics['groups']:
                selection._ranking_metrics({'quantized_validation': {'successor_ranking': metrics}})
        covered = pilot_selection.coverage(baseline) and pilot_selection.coverage(evaluated)
        result[stratum] = {
            'control_coverage_passed': pilot_selection.coverage(baseline),
            'candidate_coverage_passed': pilot_selection.coverage(evaluated),
            'sufficient_comparable_evidence': covered,
            'regret_reduction': selection._regret_reduction(baseline['mean_teacher_regret'], evaluated['mean_teacher_regret']) if covered else None,
            'flip_rate_increase': evaluated['float_vs_quantized_action_flip_rate'] - baseline['float_vs_quantized_action_flip_rate'] if covered else None,
            'used_as_full_advancement_gate': False,
        }
    return result


def select_arms(records, evidence, rankings):
    arms = []
    early = tuple(group for group in rankings.validation
                  if sum(len(turn['action']) for turn in group.evidence['source_binding']['prefix']) <= 12)
    for weight in sorted({row['weight'] for row in records}):
        candidates = [row for row in records if row['weight'] == weight]
        receipt = selection._selected_seed([row['seed_receipt'] for row in candidates])
        chosen = next(row for row in candidates if row['seed'] == receipt['seed'])
        item = evidence[weight, chosen['seed']]
        early_metrics = (trainer.successor_ranking_metrics(item['parameters'], item['architecture'], early, quantized=item['quantized'])
                         if early else {'groups': 0, 'comparable_groups': 0, 'mean_teacher_regret': None,
                                        'top1_agreement': None, 'float_vs_quantized_action_flip_rate': None,
                                        'status': 'no-eligible-early-validation-groups'})
        arms.append({'lambda': weight, 'seed': chosen['seed'],
            'runtime': chosen['runtime'], 'source': chosen['source'], 'float_checkpoint': chosen['float_checkpoint'],
            'seed_reference': item['reference'], 'runtime_body_sha256': item['runtime_body_sha256'],
            'payload_sha256': item['payload_sha256'], 'canonical_retention_passed': receipt['offline_gate']['passed'],
            'source_reserve': chosen['source_reserve'], 'overall': receipt['quantized_validation']['successor_ranking'],
            'early': early_metrics,
            'eligible_for_multi_opponent': receipt['offline_gate']['passed'] is True and chosen['source_reserve'] >= 2000})
    return arms


def assess(root, context, phase):
    root, context = Path(root).resolve(), Path(context).resolve()
    contract, parent = validate_context(root, context, phase)
    policy_path = prepare_policy(context, phase, contract)
    training_path = context / phase / 'training.json'
    training = campaign.read(training_path)
    records = validate_roster(training, contract['full_training_roster']['lambdas'])
    for key in ('datasets', 'source_routes'):
        if any(row['seed_receipt']['binding'][key] != records[0]['seed_receipt']['binding'][key] for row in records[1:]):
            raise ValueError('full training seeds used different scalar inputs')
    for weight in contract['full_training_roster']['lambdas']:
        selection._validate_seed_roster(
            [row['seed_receipt'] for row in sorted(records, key=lambda row: row['seed']) if row['weight'] == weight],
            ranking_weight=weight, qat_profile=trainer.STANDARD_QAT_PROFILE,
            teacher_ranking_profile=selection.pipeline.STANDARD_TEACHER_RANKING_PROFILE)
    campaign.verify(training['producer'])
    bundle = trainer.FrozenBundle.load(campaign.verify(contract['bundle']))
    audit, rankings = validate_inputs(context, phase, contract, training, bundle)
    initial = trainer.load_float_checkpoint(campaign.verify(contract['inputs']['attempt_one_initial_checkpoint']),
                                            trainer.ARCHITECTURES['capacity-12x8'])
    evidence = {(row['weight'], row['seed']): validate_seed(context, phase, row, contract, audit, rankings, bundle, initial)
                for row in records}
    arms = select_arms(records, evidence, rankings)
    scalar, candidate = arms
    diagnostics = ranking_diagnostic(scalar, candidate)
    selected = candidate if candidate['eligible_for_multi_opponent'] else None
    frozen_control = {'source': parent['inputs']['discrete_v3_deployment.cpp'],
                      'runtime': parent['inputs']['attempt_zero_runtime'],
                      'role': 'frozen-deployed-strength-control'}
    for key in ('source', 'runtime'):
        campaign.verify(frozen_control[key])
    return campaign.seal(context / phase / 'full-model-selection.json', {
        'schema': campaign.ID + '.full-model-selection.v2', 'policy': campaign.record(policy_path),
        'context': campaign.record(context / 'campaign.json'), 'parent_campaign': contract['parent_campaign'],
        'admitted_pilot': contract['admitted_pilot'], 'training': campaign.record(training_path),
        'input_audit': training['input_audit'], 'ranking_store': audit['ranking_store'],
        'seed_references': [evidence[row['weight'], row['seed']]['reference'] for row in records],
        'arms': arms, 'scalar_control': scalar, 'frozen_deployed_control': frozen_control,
        'selected': selected, 'diagnostics': diagnostics,
        'eligible_for_multi_opponent': selected is not None,
        'status': 'full-model-selected-awaiting-strength-evaluation' if selected else 'full-model-offline-rejected',
        'game_banks_opened': False, 'qualification_passed': False, 'campaign_success': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--context', type=Path, required=True)
    parser.add_argument('--phase', required=True)
    parser.add_argument('command', choices=('prepare', 'assess'))
    args = parser.parse_args()
    root, context = args.root.resolve(), args.context.resolve()
    with campaign.lease(root):
        if args.command == 'prepare':
            contract, _ = validate_context(root, context, args.phase)
            prepare_policy(context, args.phase, contract)
            print('full selection policy frozen; no training or game execution', flush=True)
            return
        with trainer.native_thread_execution_scope():
            result = assess(root, context, args.phase)
    print(json.dumps({'status': result['status'], 'eligible_for_multi_opponent': result['eligible_for_multi_opponent'],
                      'qualification_passed': False, 'campaign_success': False}), flush=True)


if __name__ == '__main__':
    main()
