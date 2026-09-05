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


if __name__ == '__main__':
    unittest.main()
