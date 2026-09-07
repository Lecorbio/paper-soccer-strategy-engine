from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools import compact_value_bfm_attribution_v2 as attribution
from tools import compact_value_bfm_campaign_v2 as campaign
from tools import compact_value_bfm_intervention_v2 as intervention
from tools import compact_value_bfm_pilot_v2 as pilot


def rewrite(path, document):
    path.unlink()
    return campaign.seal(path, {key: value for key, value in document.items() if key != 'body_sha256'})


def fixture(root, *, category='qat-and-scales'):
    root = root.resolve()
    inputs = {}
    for name in ('attempt_one_initial_checkpoint', 'teacher_runtime', 'attempt_zero_runtime'):
        source = root / 'inputs' / name
        campaign.once(source, ('original ' + name).encode())
        inputs[name] = campaign.record(source)
    parent = campaign.seal(root / 'campaign.json', {'policy': campaign.POLICY, 'inputs': inputs, 'exclusions': []})
    rows, previous = [], []
    domain = campaign.legacy.FEATURE_FINGERPRINT_DOMAIN
    for attempt in (1, 2):
        phase = f'attempt-{attempt:03d}-pilot'; context = root / 'phases' / phase
        inherited = root / 'exclusions' / f'prior-{attempt}.json'
        campaign.seal(inherited, {'role': 'protected' if attempt == 1 else 'prior-validation',
                                 'domain': domain, 'fingerprints': [str(attempt) * 64]})
        contract = campaign.seal(context / 'campaign.json', {'attempt': attempt, 'phase': 'pilot',
            'policy': campaign.POLICY, 'parent_campaign': campaign.record(root / 'campaign.json'),
            'inputs': inputs, 'exclusions': [campaign.record(inherited)]})
        for name in ('pilot-outcome.json', 'training.json', 'model-selection.json', 'positions.json', 'games.json'):
            campaign.seal(context / phase / name, {'bound': name, 'attempt': attempt})
        records = {key: campaign.record(context / phase / name) for key, name in (
            ('outcome', 'pilot-outcome.json'), ('training', 'training.json'), ('selection', 'model-selection.json'))}
        previous.append({'attempt': attempt, 'phase': phase, 'context': context, 'contract': contract,
                         **records, 'screen': None})
        rows.append({'attempt': attempt, 'completed_attempt_count': 1, 'outcome': records['outcome'],
                     'phases': [{'phase': phase, 'training': records['training'], 'selection': records['selection']}]})
    document = campaign.seal(root / 'attribution/after-two-attempts.json', {
        'schema': campaign.ID + '.attribution.v2', 'completed_unsuccessful_trained_attempts': 2,
        'attempts': rows, 'policy': attribution.POLICY, 'protected_results_used': False,
        'live_results_used': False, 'new_training_started': False, 'campaign_success': False,
        'recommendation': {'category': category, 'existing_profile_to_consider': intervention.PROFILE if category == 'qat-and-scales' else None,
                           'selected_execution_profile': None, 'attempt_three_may_start': False},
        'maintained_profile_menu': {'qat_and_scales': {intervention.PROFILE: intervention.approved_profile()}}})
    campaign.seal(root / 'baseline-engine-comparison.json', {'same_weights': True, 'all_checks_passed': True, 'exclusions': []})
    campaign.seal(root / 'exclusions/anchor-derived.json', {'domain': domain, 'fingerprints': {}})
    campaign.seal(root / 'exclusions/prior-search-validation.json', {'role': 'prior-validation', 'domain': domain, 'fingerprints': []})
    return root, parent, document, previous


def checked_attribution(root):
    return mock.patch.object(intervention, 'validated_attribution',
                             side_effect=lambda ignored: campaign.read(root / 'attribution/after-two-attempts.json'))


def pilot_patches(root, previous):
    return (
        checked_attribution(root),
        mock.patch.object(pilot, 'validate_smoke', return_value={'smoke': 'already-validated'}),
        mock.patch.object(pilot, 'smoke_exclusions', return_value=[]),
        mock.patch.object(pilot.attempts, 'failed_attempt', side_effect=lambda ignored, n: previous[n - 1]),
        mock.patch.object(pilot.attempts, 'collect_fingerprints',
                          side_effect=lambda item: ({('prior-validation', campaign.legacy.FEATURE_FINGERPRINT_DOMAIN):
                                                     {str(item['attempt'] + 2) * 64}}, 'not-opened')),
    )


def fourth_fixture(root, *, freeze_intervention=True):
    """Synthetic completed1/2/3 metadata and the actual fourth-profile binding."""
    from tests.codingame.test_compact_value_bfm_attribution_v2 import fixture as phase_fixture
    root = root.resolve(); inputs = {}
    for key in ('attempt_one_initial_checkpoint', 'teacher_runtime', 'attempt_zero_runtime'):
        source = root / 'inputs' / key; campaign.once(source, ('fixed ' + key).encode())
        inputs[key] = campaign.record(source)
    parent = campaign.seal(root / 'campaign.json', {'policy': campaign.POLICY, 'inputs': inputs, 'exclusions': []})
    previous = []
    for attempt in (1, 2, 3):
        if attempt == 3:
            with mock.patch.object(attribution.attempts, 'failed_attempt', side_effect=lambda _, n: previous[n - 1]):
                attribution.produce(root)
                third_intervention = intervention.prepare(root)
        row = phase_fixture(root, attempt, float_sign=.88, quantized_sign=.86)
        for name in ('positions.json', 'games.json'):
            campaign.seal(row['context'] / row['phase'] / name, {'fixture': name})
        exclusion = root / 'exclusions' / f'prior-{attempt}.json'
        campaign.seal(exclusion, {'role': 'prior-validation', 'domain': campaign.legacy.FEATURE_FINGERPRINT_DOMAIN,
                                  'fingerprints': [str(attempt) * 64]})
        contract = {'attempt': attempt, 'phase': 'pilot', 'policy': campaign.POLICY,
                    'parent_campaign': campaign.record(root / 'campaign.json'), 'inputs': inputs,
                    'exclusions': [campaign.record(exclusion)]}
        if attempt == 3:
            contract.update({'qat_profile': intervention.PROFILE, 'qat_profile_contract': intervention.approved_profile(),
                'intervention': campaign.record(intervention.path(root)), 'completed_unsuccessful_trained_attempts': 2,
                'previous_failed_attempts': third_intervention['previous_failed_attempts'],
                'pilot_games': 2000, 'pilot_training_roster': {'lambdas': [0, .1, .25], 'seeds': list(attribution.trainer.FIXED_SEEDS)},
                'candidate_lineage': {'mandatory_training': True, 'initial_float': inputs['attempt_one_initial_checkpoint'],
                    'generation_student': inputs['attempt_zero_runtime'], 'smoke_weights_reused': False}})
            training = campaign.read(row['training']['path']); training['schema'] = campaign.ID + '.training.v2'
            for seed in training['results']:
                receipt = seed['seed_receipt']; receipt.update({'qat_profile': intervention.PROFILE,
                    'qat_profile_contract': intervention.approved_profile()})
                receipt['offline_gate'] = attribution.trainer.offline_advancement_gate(
                    receipt['float_validation'], receipt['quantized_validation'])
            rewrite(Path(row['training']['path']), training); row['training'] = campaign.record(row['training']['path'])
            selected = campaign.read(row['selection']['path']); selected.update({
                'schema': campaign.ID + '.pilot-model-selection.v2', 'training': row['training'],
                'status': 'offline-rejected-before-rank4-screen', 'selected': None,
                'pilot_admitted': False, 'campaign_success': False})
            rewrite(Path(row['selection']['path']), selected); row['selection'] = campaign.record(row['selection']['path'])
        outcome = {'schema': campaign.ID + '.pilot-outcome.v2', 'status': 'offline-rejected',
                   'admitted': False, 'campaign_success': False, 'selection': row['selection']}
        rewrite(Path(row['outcome']['path']), outcome); row['outcome'] = campaign.record(row['outcome']['path'])
        row['contract'] = campaign.seal(row['context'] / 'campaign.json', contract)
        previous.append(row)
    campaign.seal(root / 'baseline-engine-comparison.json', {'same_weights': True, 'all_checks_passed': True, 'exclusions': []})
    campaign.seal(root / 'exclusions/anchor-derived.json', {'domain': campaign.legacy.FEATURE_FINGERPRINT_DOMAIN, 'fingerprints': {}})
    campaign.seal(root / 'exclusions/prior-search-validation.json', {'role': 'prior-validation',
        'domain': campaign.legacy.FEATURE_FINGERPRINT_DOMAIN, 'fingerprints': []})
    with mock.patch.object(attribution.attempts, 'failed_attempt', side_effect=lambda _, n: previous[n - 1]):
        report = attribution.produce(root, after_attempts=3)
        document = intervention.prepare(root, attempt=4) if freeze_intervention else None
    contract4 = None if document is None else {**{k: v for k, v in parent.items() if k != 'body_sha256'},
        'parent_campaign': campaign.record(root / 'campaign.json'), 'attempt': 4, 'phase': 'pilot',
        'qat_profile': document['qat_profile'], 'qat_profile_contract': document['qat_profile_contract'],
        'intervention': campaign.record(intervention.path(root, 4)), 'completed_unsuccessful_trained_attempts': 3,
        'previous_failed_attempts': document['previous_failed_attempts'], 'pilot_games': 2000,
        'pilot_training_roster': {'lambdas': [0, .1, .25], 'seeds': list(attribution.trainer.FIXED_SEEDS)},
        'candidate_lineage': {'mandatory_training': True, 'initial_float': document['initial_float'],
            'generation_student': document['pilot_generation_student'], 'smoke_weights_reused': False}}
    return root, parent, report, previous, contract4


class InterventionTests(unittest.TestCase):
    def test_no_frozen_attribution_means_no_third_context_or_intervention(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.object(intervention, 'validated_attribution') as validate:
                with self.assertRaisesRegex(ValueError, 'frozen validated'):
                    pilot.prepare(root, 3)
            validate.assert_not_called()
            self.assertFalse((root / 'phases').exists())
            self.assertFalse((root / 'interventions').exists())

    def test_asserted_attribution_never_bypasses_actual_outcome_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _, _, _ = fixture(Path(tmp))
            with mock.patch.object(intervention, 'validated_attribution', side_effect=ValueError('second outcome incomplete')):
                with self.assertRaisesRegex(ValueError, 'second outcome incomplete'):
                    pilot.prepare(root, 3)
            self.assertFalse((root / 'phases/attempt-003-pilot').exists())

    def test_search_teacher_and_missing_recommendations_remain_closed(self):
        for category in ('harder-teacher-ranking', 'one-search-intervention', None):
            with self.subTest(category=category), tempfile.TemporaryDirectory() as tmp:
                root, _, _, _ = fixture(Path(tmp), category=category)
                with checked_attribution(root):
                    with self.assertRaisesRegex(ValueError, 'only the approved QAT/scales'):
                        pilot.prepare(root, 3)
                self.assertFalse((root / 'phases/attempt-003-pilot').exists())
                self.assertFalse(intervention.path(root).exists())

    def test_profile_is_the_exact_existing_recipe_with_unchanged_architecture_and_schedule(self):
        profile = intervention.approved_profile()
        self.assertEqual(profile, intervention.trainer.qat_profile_contract('refined-adaptive-scales-v1'))
        self.assertEqual(profile['schedule']['float_warmup_epochs'], 1)
        self.assertEqual(profile['schedule']['qat_epochs'], 4)
        self.assertEqual(profile['quantization']['bits'], 3)
        changed = copy.deepcopy(profile); changed['schedule']['qat_epochs'] = 5
        original = intervention.trainer.qat_profile_contract
        with mock.patch.object(intervention.trainer, 'qat_profile_contract',
                               side_effect=lambda name: changed if name == intervention.PROFILE else original(name)):
            with self.assertRaisesRegex(ValueError, 'architecture or training schedule'):
                intervention.approved_profile()

    def test_prepare_and_validate_freeze_original_inputs_without_starting_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, parent, _, _ = fixture(Path(tmp))
            with checked_attribution(root):
                document = intervention.prepare(root)
                self.assertEqual(document, intervention.prepare(root))
                self.assertEqual(document, intervention.validate(root))
            self.assertEqual(document['initial_float'], parent['inputs']['attempt_one_initial_checkpoint'])
            self.assertEqual(document['teacher_runtime'], parent['inputs']['teacher_runtime'])
            self.assertEqual(document['unchanged_campaign_policy'], campaign.POLICY)
            self.assertEqual(document['single_changed_training_setting'], 'qat_profile')
            self.assertFalse(document['new_training_started'])
            self.assertFalse(document['qualification_passed'])

    def test_resume_preserves_verified_historical_intervention_producer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _, _, _ = fixture(Path(tmp))
            with checked_attribution(root):
                document = intervention.prepare(root)
                document['producers']['intervention'] = campaign.copy_checked(
                    campaign.verify(document['producers']['intervention']), root / 'historical/intervention.py')
                rewrite(intervention.path(root), document)
                historical = campaign.read(intervention.path(root))
                self.assertEqual(historical, intervention.prepare(root))
                self.assertEqual(historical, intervention.validate(root))
                Path(document['producers']['intervention']['path']).write_bytes(b'changed producer')
                with self.assertRaisesRegex(ValueError, 'changed artifact'):
                    intervention.validate(root)

    def test_prepare_third_attempt_accumulates_both_closures_and_preserves_old_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, parent, _, previous = fixture(Path(tmp))
            before = {path: path.read_bytes() for row in previous for path in row['context'].rglob('*') if path.is_file()}
            a, b, c, d, e = pilot_patches(root, previous)
            with a, b, c, d, e:
                prepared = pilot.prepare(root, 3)
                self.assertEqual(prepared, pilot.prepare(root, 3))
                self.assertEqual(intervention.expected_qat_profile(prepared), intervention.PROFILE)
            self.assertEqual(before, {path: path.read_bytes() for row in previous for path in row['context'].rglob('*') if path.is_file()})
            self.assertEqual(prepared['policy'], campaign.POLICY)
            self.assertEqual(prepared['inputs'], parent['inputs'])
            self.assertEqual(prepared['pilot_games'], 2000)
            self.assertEqual(prepared['pilot_training_roster'], {'lambdas': [0, .1, .25], 'seeds': [20260907, 20260908, 20260909]})
            self.assertEqual(prepared['completed_unsuccessful_trained_attempts'], 2)
            self.assertEqual([row['attempt'] for row in prepared['previous_failed_attempts']], [1, 2])
            exclusions = campaign.exclusion_sets(prepared); domain = campaign.legacy.FEATURE_FINGERPRINT_DOMAIN
            self.assertEqual(exclusions['protected', domain], {'1' * 64})
            self.assertEqual(exclusions['prior-validation', domain], {str(n) * 64 for n in (2, 3, 4)})

    def test_resume_cannot_drop_a_previous_attempt_or_carry_exclusion(self):
        for mutation in ('attempt', 'exclusion'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root, _, _, previous = fixture(Path(tmp))
                a, b, c, d, e = pilot_patches(root, previous)
                with a, b, c, d, e:
                    prepared = pilot.prepare(root, 3)
                    changed = copy.deepcopy(prepared)
                    if mutation == 'attempt':
                        changed['previous_failed_attempts'].pop()
                    else:
                        target = campaign.read(campaign.verify(changed['previous_failed_attempts'][0]['carry']))['artifacts'][0]
                        changed['exclusions'].remove(target)
                    rewrite(root / 'phases/attempt-003-pilot/campaign.json', changed)
                    with self.assertRaisesRegex(ValueError, 'prior-attempt outcome|accumulated isolation'):
                        pilot.prepare(root, 3)

    def test_explicit_spawn_executor_is_frozen_and_none_preserves_it_on_resume(self):
        from tools import compact_value_bfm_seed_process_v2 as seed_process
        with tempfile.TemporaryDirectory() as tmp:
            root, _, _, previous = fixture(Path(tmp))
            a, b, c, d, e = pilot_patches(root, previous)
            with a, b, c, d, e:
                prepared = pilot.prepare(root, 3, training_executor='spawn-v2')
                self.assertEqual(prepared['training_executor'], seed_process.MODE)
                self.assertEqual(prepared, pilot.prepare(root, 3))
                self.assertEqual(prepared, pilot.prepare(root, 3, training_executor='spawn-v2'))
                with self.assertRaisesRegex(ValueError, 'cannot change its frozen training executor'):
                    pilot.prepare(root, 3, training_executor='threads')
                self.assertEqual(prepared, campaign.read(root / 'phases/attempt-003-pilot/campaign.json'))

    def test_default_thread_executor_cannot_be_changed_after_freeze(self):
        from tools import compact_value_bfm_seed_process_v2 as seed_process
        with tempfile.TemporaryDirectory() as tmp:
            root, _, _, previous = fixture(Path(tmp))
            a, b, c, d, e = pilot_patches(root, previous)
            with a, b, c, d, e:
                prepared = pilot.prepare(root, 3)
                self.assertNotIn('training_executor', prepared)
                self.assertEqual(seed_process.executor_mode(prepared), 'threads')
                self.assertEqual(prepared, pilot.prepare(root, 3, training_executor='threads'))
                with self.assertRaisesRegex(ValueError, 'cannot change its frozen training executor'):
                    pilot.prepare(root, 3, training_executor='spawn-v2')

    def test_downstream_profile_rejects_forged_standard_and_unbound_third_attempts(self):
        self.assertEqual(intervention.expected_qat_profile({'attempt': 1}), 'standard-v1')
        with self.assertRaisesRegex(ValueError, 'first two standard'):
            intervention.expected_qat_profile({'attempt': 2, 'qat_profile': intervention.PROFILE})
        with self.assertRaisesRegex(ValueError, 'third attempt requires'):
            intervention.expected_qat_profile({'attempt': 3})
        with self.assertRaisesRegex(ValueError, 'no approved'):
            intervention.expected_qat_profile({'attempt': 4})

    def test_frozen_recipe_input_and_prior_identity_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _, _, previous = fixture(Path(tmp))
            a, b, c, d, e = pilot_patches(root, previous)
            with a, b, c, d, e:
                prepared = pilot.prepare(root, 3)
            for key, value in (('qat_profile', 'standard-v1'), ('pilot_games', 1000),
                               ('previous_failed_attempts', prepared['previous_failed_attempts'][:1])):
                altered = copy.deepcopy(prepared); altered[key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    intervention.expected_qat_profile(altered)
            altered = copy.deepcopy(prepared)
            altered['inputs']['attempt_one_initial_checkpoint'] = altered['inputs']['attempt_zero_runtime']
            with self.assertRaisesRegex(ValueError, 'frozen inputs'):
                intervention.expected_qat_profile(altered)

    def test_downstream_profile_requires_the_complete_verified_producer_closure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _, _, previous = fixture(Path(tmp))
            a, b, c, d, e = pilot_patches(root, previous)
            with a, b, c, d, e:
                prepared = pilot.prepare(root, 3)
            document = campaign.read(intervention.path(root)); document['producers'] = {}
            rewrite(intervention.path(root), document)
            prepared['intervention'] = campaign.record(intervention.path(root))
            with self.assertRaisesRegex(ValueError, 'source closure is incomplete'):
                intervention.expected_qat_profile(prepared)
            with checked_attribution(root), self.assertRaisesRegex(ValueError, 'source closure is incomplete'):
                intervention.validate(root)

    def test_full_stage_may_use_admitted_pilot_student_without_changing_original_float_or_qat(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _, _, previous = fixture(Path(tmp))
            a, b, c, d, e = pilot_patches(root, previous)
            with a, b, c, d, e:
                prepared = pilot.prepare(root, 3)
            full = copy.deepcopy(prepared); full['phase'] = 'full'
            full['inputs']['attempt_zero_runtime'] = {'path': '/admitted/pilot.runtime', 'sha256': 'b' * 64, 'bytes': 42}
            self.assertEqual(intervention.expected_qat_profile(full), intervention.PROFILE)

    def test_poison_protected_and_live_recommendation_fields_are_not_followed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _, document, _ = fixture(Path(tmp))
            poison = root / 'protected-metrics.json'; poison.write_bytes(b'POISON')
            document['protected_result'] = {'path': str(poison)}
            document['live_score'] = {'path': str(poison)}
            rewrite(root / 'attribution/after-two-attempts.json', document)
            actual = Path.read_bytes
            def read_bytes(path):
                if path == poison:
                    raise AssertionError('protected metric opened')
                return actual(path)
            with checked_attribution(root), mock.patch.object(Path, 'read_bytes', read_bytes):
                result = intervention.prepare(root)
            self.assertFalse(result['protected_metrics_used_for_intervention'])
            self.assertFalse(result['live_metrics_used_for_intervention'])


class FourthInterventionTests(unittest.TestCase):
    def test_incomplete_third_blocks_before_validators_or_metric_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for attempt in (1, 2):
                phase = f'attempt-{attempt:03d}-pilot'
                campaign.seal(root / 'phases' / phase / phase / 'pilot-outcome.json', {'status': 'offline-rejected'})
            with mock.patch.object(attribution.attempts, 'failed_attempt') as failed, \
                    mock.patch.object(attribution, 'third_pilot_retention') as facts:
                with self.assertRaisesRegex(ValueError, 'three completed'):
                    attribution.produce(root, after_attempts=3)
            failed.assert_not_called(); facts.assert_not_called()
            self.assertFalse(attribution.path(root, after_attempts=3).exists())

    def test_after_three_reproduces_all_failures_and_keeps_after_two_frozen(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _, report, previous, contract = fourth_fixture(Path(tmp))
            old = attribution.path(root).read_bytes()
            with mock.patch.object(attribution.attempts, 'failed_attempt', side_effect=lambda _, n: previous[n - 1]) as checked:
                self.assertEqual(report, attribution.validate(root, after_attempts=3))
            self.assertEqual([call.args[1] for call in checked.call_args_list], [1, 2, 3])
            self.assertEqual(attribution.path(root).read_bytes(), old)
            self.assertEqual(set(campaign.read(attribution.path(root))['maintained_profile_menu']['qat_and_scales']),
                             {intervention.STANDARD, intervention.PROFILE})
            self.assertEqual(report['third_pilot_retention']['retention_failed_seed_count'], 9)
            self.assertFalse(report['recommendation']['causal_attribution_proven'])
            self.assertFalse(report['recommendation']['attempt_four_may_start'])
            self.assertEqual(intervention.expected_qat_profile(contract), intervention.RETENTION_PROFILE)

    @staticmethod
    def change_third(previous, mutate):
        row = previous[-1]; training = campaign.read(row['training']['path']); mutate(training)
        rewrite(Path(row['training']['path']), training); row['training'] = campaign.record(row['training']['path'])
        selected = campaign.read(row['selection']['path']); selected['training'] = row['training']
        rewrite(Path(row['selection']['path']), selected); row['selection'] = campaign.record(row['selection']['path'])
        outcome = campaign.read(row['outcome']['path']); outcome['selection'] = row['selection']
        rewrite(Path(row['outcome']['path']), outcome); row['outcome'] = campaign.record(row['outcome']['path'])

    def test_missing_duplicate_passing_or_forged_third_seed_cannot_authorize(self):
        for fault in ('missing', 'duplicate', 'passing', 'forged-gate', 'wrong-profile'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as tmp:
                root, _, _, previous, _ = fourth_fixture(Path(tmp), freeze_intervention=False)
                def mutate(training):
                    if fault == 'missing': training['results'].pop()
                    elif fault == 'duplicate': training['results'][-1] = copy.deepcopy(training['results'][0])
                    else:
                        receipt = training['results'][0]['seed_receipt']
                        if fault == 'passing':
                            receipt['quantized_validation'] = copy.deepcopy(receipt['float_validation'])
                            receipt['offline_gate'] = attribution.trainer.offline_advancement_gate(
                                receipt['float_validation'], receipt['quantized_validation'])
                        elif fault == 'forged-gate': receipt['offline_gate']['errors'] = ['invented failure']
                        else: receipt['qat_profile'] = intervention.STANDARD
                self.change_third(previous, mutate)
                with mock.patch.object(attribution.attempts, 'failed_attempt', side_effect=lambda _, n: previous[n - 1]):
                    with self.assertRaises(ValueError): attribution.body_after_three(root)
                self.assertFalse(intervention.path(root, 4).exists())

    def test_protected_terminal_and_third_full_are_rejected_before_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _, _, previous, _ = fourth_fixture(Path(tmp), freeze_intervention=False)
            full = root / 'phases/attempt-003-full'; full.mkdir()
            with mock.patch.object(attribution.attempts, 'failed_attempt') as failed:
                with self.assertRaisesRegex(ValueError, 'third pilot'): attribution.completed_three(root)
            failed.assert_not_called(); full.rmdir()
            full = root / 'phases/attempt-001-full'
            campaign.seal(full / full.name / 'attempt-outcome.json', {'schema': 'protected-terminal',
                'protected_result': {'path': '/must-not-open/protected.json'}, 'status': 'completed-unsuccessful'})
            with mock.patch.object(attribution.attempts, 'failed_attempt') as failed:
                with self.assertRaisesRegex(ValueError, 'unprotected failures'): attribution.completed_three(root)
            failed.assert_not_called()

    def test_unknown_protected_fields_are_not_opened_or_copied(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _, _, previous, _ = fourth_fixture(Path(tmp), freeze_intervention=False)
            poison = root / 'protected/poison.json'; campaign.once(poison, b'POISON PRIVATE METRICS')
            secret = campaign.record(poison)
            def mutate(training):
                training['protected_result'] = secret
                for row in training['results']:
                    row['seed_receipt']['float_validation']['protected_result'] = secret
                    row['seed_receipt']['quantized_validation']['canonical_validation']['live_result'] = secret
            self.change_third(previous, mutate)
            old_read = campaign.read; old_verify = campaign.verify
            def read(path):
                if Path(path) == poison: raise AssertionError('protected source opened')
                return old_read(path)
            def verify(record):
                if Path(record['path']) == poison: raise AssertionError('protected source hashed')
                return old_verify(record)
            with mock.patch.object(attribution.attempts, 'failed_attempt', side_effect=lambda _, n: previous[n - 1]), \
                    mock.patch.object(campaign, 'read', side_effect=read), mock.patch.object(campaign, 'verify', side_effect=verify):
                report = attribution.body_after_three(root)
                self.assertNotIn(str(poison), campaign.raw(report).decode())
                redirected = {**report['third_pilot_retention']['sources'], 'training': secret}
                with self.assertRaisesRegex(ValueError, 'fixed unprotected'):
                    attribution.third_pilot_retention(root, redirected)

    def test_exact_fourth_profile_preserves_shapes_and_rejects_changed_lr_or_reference(self):
        profile = intervention.approved_profile(4); prior = intervention.approved_profile()
        self.assertEqual(profile['schedule'], {**prior['schedule'], 'qat_learning_rate': .0000625})
        self.assertEqual(profile['quantization'], prior['quantization'])
        original = intervention.trainer.qat_profile_contract
        for field in ('rate', 'reference'):
            changed = copy.deepcopy(profile)
            if field == 'rate': changed['schedule']['qat_learning_rate'] = .000125
            else: changed['scale_selection']['retention_policy']['reference'] = 'current-epoch-float'
            with mock.patch.object(intervention.trainer, 'qat_profile_contract',
                    side_effect=lambda name: changed if name == intervention.RETENTION_PROFILE else original(name)):
                with self.assertRaisesRegex(ValueError, 'approved recipe'): intervention.approved_profile(4)

    def test_fourth_resume_preserves_producers_and_rejects_changed_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _, report, previous, contract = fourth_fixture(Path(tmp))
            document = campaign.read(intervention.path(root, 4))
            copied = campaign.copy_checked(Path(document['producers']['intervention']['path']), root / 'snapshot/intervention.py')
            document['producers']['intervention'] = copied; rewrite(intervention.path(root, 4), document)
            contract['intervention'] = campaign.record(intervention.path(root, 4))
            self.assertEqual(intervention.expected_qat_profile(contract), intervention.RETENTION_PROFILE)
            with mock.patch.object(attribution.attempts, 'failed_attempt', side_effect=lambda _, n: previous[n - 1]):
                self.assertEqual(intervention.prepare(root, attempt=4), intervention.validate(root, attempt=4))
            Path(copied['path']).write_bytes(b'changed source')
            with self.assertRaisesRegex(ValueError, 'changed artifact'): intervention.expected_qat_profile(contract)

    def test_fourth_binding_rejects_missing_prior_foreign_profile_and_fifth(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _, _, _, contract = fourth_fixture(Path(tmp))
            original_document = campaign.read(intervention.path(root, 4))
            for field in ('prior', 'profile', 'attempt', 'producer', 'facts'):
                changed = copy.deepcopy(contract)
                if field == 'prior': changed['previous_failed_attempts'].pop()
                elif field == 'profile': changed['qat_profile'] = intervention.PROFILE
                elif field == 'attempt': changed['attempt'] = 5
                else:
                    document = copy.deepcopy(original_document)
                    if field == 'producer': document['producers'] = {}
                    else: document['third_pilot_retention_sources']['training'] = {'path': '/forbidden/protected.json'}
                    rewrite(intervention.path(root, 4), document); changed['intervention'] = campaign.record(intervention.path(root, 4))
                with self.subTest(field=field), self.assertRaises((ValueError, KeyError)):
                    intervention.expected_qat_profile(changed)
            with self.assertRaises(ValueError): pilot.context_root(root, 5, intervention={'attempt': 5})

    def test_fourth_pilot_prepare_carries_all_three_and_resume_cannot_drop_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, parent, _, previous, _ = fourth_fixture(Path(tmp))
            with mock.patch.object(attribution.attempts, 'failed_attempt', side_effect=lambda _, n: previous[n - 1]), \
                    mock.patch.object(pilot, 'validate_smoke', return_value={'smoke': 'bound'}), \
                    mock.patch.object(pilot, 'smoke_exclusions', return_value=[]), \
                    mock.patch.object(pilot.attempts, 'collect_fingerprints', side_effect=lambda row:
                        ({('prior-validation', campaign.legacy.FEATURE_FINGERPRINT_DOMAIN): {str(row['attempt'] + 3) * 64}}, 'not-opened')):
                contract = pilot.prepare(root, 4)
                self.assertEqual(contract, pilot.prepare(root, 4))
                self.assertEqual(contract['completed_unsuccessful_trained_attempts'], 3)
                self.assertEqual([row['attempt'] for row in contract['previous_failed_attempts']], [1, 2, 3])
                self.assertEqual(contract['inputs'], parent['inputs'])
                self.assertEqual(contract['qat_profile'], intervention.RETENTION_PROFILE)
                for row in previous:
                    self.assertTrue(all(item in contract['exclusions'] for item in row['contract']['exclusions']))
                target = root / 'phases/attempt-004-pilot/campaign.json'
                altered = copy.deepcopy(contract)
                altered['exclusions'].remove(previous[-1]['contract']['exclusions'][0]); rewrite(target, altered)
                with self.assertRaises(ValueError): pilot.prepare(root, 4)


if __name__ == '__main__':
    unittest.main()
