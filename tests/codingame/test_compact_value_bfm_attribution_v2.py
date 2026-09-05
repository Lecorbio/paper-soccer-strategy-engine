from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools import compact_value_bfm_attribution_v2 as attribution
from tools import compact_value_bfm_campaign_v2 as campaign


def ranking(regret=.2, flip=.01, groups=200):
    return {'groups': groups, 'comparable_groups': groups, 'mean_teacher_regret': regret,
            'top1_agreement': .8, 'float_vs_quantized_action_flip_rate': flip}


def seed(weight, ordinal=20260907, *, float_sign=.88, quantized_sign=.88, regret=.2):
    reports = {}
    for frame, sign in (('float_validation', float_sign), ('quantized_validation', quantized_sign)):
        reports[frame] = {name: {'sign_accuracy': sign, 'weighted_huber': .04}
                         for name in ('common_adjudicator', 'canonical_validation')}
        reports[frame]['successor_ranking'] = ranking(regret=regret, flip=0 if frame == 'float_validation' else .01)
    return {'weight': weight, 'seed': ordinal, 'seed_receipt': {
        **reports, 'offline_gate': {'passed': quantized_sign >= .8613 and float_sign - quantized_sign < .005}}}


def fixture(root, attempt, *, full=False, float_sign=.88, quantized_sign=.88, regret=.2, pilot=None):
    root = root.resolve()
    phase = f'attempt-{attempt:03d}-' + ('full' if full else 'pilot')
    context = root / 'phases' / phase
    weights = (0, .1) if full else (0, .1, .25)
    rows = [seed(weight, ordinal, float_sign=float_sign, quantized_sign=quantized_sign,
                 regret=.3 if weight == 0 else regret)
            for weight in weights for ordinal in (20260907, 20260908, 20260909)]
    training = context / phase / 'training.json'
    campaign.seal(training, {'smoke': False, 'mandatory_training_verified': True, 'results': rows})
    arms = [{'lambda': weight, 'seed': 20260907, 'source_reserve': 2800,
             'canonical_retention_passed': row['seed_receipt']['offline_gate']['passed'],
             'overall': ranking(regret=.3 if weight == 0 else regret),
             'early': ranking(regret=.3 if weight == 0 else regret)}
            for weight, row in zip(weights, rows[::3])]
    selection = context / phase / ('full-model-selection.json' if full else 'model-selection.json')
    campaign.seal(selection, {'arms': arms})
    outcome = context / phase / ('attempt-outcome.json' if full else 'pilot-outcome.json')
    campaign.seal(outcome, {'status': 'completed-unsuccessful' if full else 'offline-rejected'})
    result = {'attempt': attempt, 'phase': phase, 'context': context,
              'training': campaign.record(training), 'selection': campaign.record(selection),
              'outcome': campaign.record(outcome), 'screen': None}
    if full:
        result.update({'stage': 'full', 'pilot': pilot, 'rejection_stage': 'full-offline',
                       'stages': {}, 'source_selection': None, 'suites': [], 'development': None})
    return result


def rewrite(record, mutate):
    path = Path(record['path'])
    document = campaign.read(path)
    document.pop('body_sha256')
    mutate(document)
    path.unlink()
    campaign.seal(path, document)
    return campaign.record(path)


class AttributionTests(unittest.TestCase):
    def test_incomplete_second_attempt_blocks_before_any_validator_or_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root, 1)
            campaign.seal(root / 'smoke-064/smoke-completion-corrected.json', {'complete': True})
            with mock.patch.object(attribution.attempts, 'failed_attempt') as validate, \
                    mock.patch.object(attribution, 'phase_evidence') as metrics:
                with self.assertRaisesRegex(ValueError, 'two completed'):
                    attribution.produce(root)
            validate.assert_not_called(); metrics.assert_not_called()
            self.assertFalse((root / 'attribution').exists())

    def test_existing_incomplete_full_cannot_fall_back_to_rejected_pilot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root, 1); fixture(root, 2)
            (root / 'phases/attempt-002-full').mkdir()
            with mock.patch.object(attribution.attempts, 'failed_attempt') as validate:
                with self.assertRaisesRegex(ValueError, 'two completed'):
                    attribution.completed_pair(root)
            validate.assert_not_called()

    def test_completed_receipts_still_require_both_actual_outcome_validators(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = fixture(root, 1); fixture(root, 2)
            with mock.patch.object(attribution.attempts, 'failed_attempt', side_effect=[first, ValueError('missing seed')]) as checked:
                with self.assertRaisesRegex(ValueError, 'missing seed'):
                    attribution.produce(root)
            self.assertEqual(checked.call_args_list, [mock.call(root.resolve(), 1), mock.call(root.resolve(), 2)])
            self.assertFalse((root / 'attribution').exists())

    def test_full_and_its_pilot_are_one_attempt_and_all_seed_diagnostics_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pilot = fixture(root, 1)
            first = fixture(root, 1, full=True, pilot=pilot)
            second = fixture(root, 2, quantized_sign=.86)
            with mock.patch.object(attribution.attempts, 'failed_attempt', side_effect=lambda _, n: (first, second)[n - 1]):
                report = attribution.produce(root)
                self.assertEqual(report, attribution.validate(root))
            self.assertEqual(report['completed_unsuccessful_trained_attempts'], 2)
            self.assertEqual([len(row['phases']) for row in report['attempts']], [2, 1])
            self.assertEqual([len(phase['seeds']) for phase in report['attempts'][0]['phases']], [9, 6])
            self.assertEqual(report['recommendation']['category'], 'qat-and-scales')
            self.assertIsNone(report['recommendation']['selected_execution_profile'])
            self.assertFalse(report['recommendation']['attempt_three_may_start'])
            self.assertFalse(report['campaign_success'])

    def test_quantization_relative_loss_and_absolute_boundary_are_separate_from_float_quality(self):
        metrics = attribution.seed_metrics(seed(.1, float_sign=.863, quantized_sign=.860))
        effect = metrics['quantization_effects']['canonical_validation']
        self.assertTrue(effect['quantization_boundary_crossing'])
        self.assertFalse(effect['relative_retention_failed'])
        metrics = attribution.seed_metrics(seed(.1, float_sign=.90, quantized_sign=.89))
        effect = metrics['quantization_effects']['canonical_validation']
        self.assertFalse(effect['quantization_boundary_crossing'])
        self.assertTrue(effect['relative_retention_failed'])

    def test_float_deficit_or_covered_ranking_deficit_recommends_teacher_category(self):
        for arguments in ({'float_sign': .85, 'quantized_sign': .85}, {'regret': .29}):
            with self.subTest(arguments=arguments), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                rows = [fixture(root, n, **arguments) for n in (1, 2)]
                with mock.patch.object(attribution.attempts, 'failed_attempt', side_effect=rows):
                    report = attribution.body(root)
                self.assertEqual(report['recommendation']['category'], 'harder-teacher-ranking')
                self.assertIsNone(report['recommendation']['selected_execution_profile'])

    def test_healthy_validation_and_actual_unprotected_strength_failure_recommends_one_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for n in (1, 2):
                base = fixture(root, n)
                row = fixture(root, n, full=True, pilot=base)
                row['rejection_stage'] = 'development'
                rows.append(row)
            evidence = [{'attempt': row['attempt'], 'rejection_stage': row['rejection_stage'],
                         'phases': [attribution.phase_evidence(row)]} for row in rows]
            report = attribution.recommendation(evidence)
            self.assertEqual(report['category'], 'one-search-intervention')
            self.assertIsNone(report['existing_profile_to_consider'])
            self.assertFalse(report['attempt_three_may_start'])

    def test_no_supported_signal_remains_explicitly_unresolved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [fixture(root, n) for n in (1, 2)]
            evidence = [{'attempt': row['attempt'], 'rejection_stage': 'offline-rejected',
                         'phases': [attribution.phase_evidence(row)]} for row in rows]
            report = attribution.recommendation(evidence)
            self.assertIsNone(report['category'])
            self.assertEqual(report['status'], 'insufficient-unprotected-attribution-evidence')

    def test_unknown_protected_live_metric_paths_and_poison_values_are_never_followed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [fixture(root, n, quantized_sign=.86) for n in (1, 2)]
            secret = root / 'protected/results.json'
            secret.parent.mkdir()
            secret.write_text('POISON PROTECTED TRANSCRIPT AND METRICS')
            record = campaign.record(secret)
            for row in rows:
                def poison(document):
                    document['protected_result'] = record
                    document['live_score'] = {'path': str(secret), 'score': 9999999}
                    for seed in document['results']:
                        seed['seed_receipt']['float_validation']['protected_metrics'] = record
                        seed['seed_receipt']['quantized_validation']['canonical_validation']['live_transcripts'] = record
                row['training'] = rewrite(row['training'], poison)
                row['protected_result'] = record
            original = Path.read_bytes
            opened = []
            def read(path):
                opened.append(path)
                if path == secret:
                    raise AssertionError('protected content opened')
                return original(path)
            with mock.patch.object(attribution.attempts, 'failed_attempt', side_effect=rows), \
                    mock.patch.object(Path, 'read_bytes', read):
                report = attribution.body(root)
            self.assertNotIn(secret, opened)
            serialized = json.dumps(report)
            self.assertNotIn('POISON', serialized)
            self.assertNotIn(str(secret), serialized)
            self.assertNotIn('9999999', serialized)
            self.assertEqual(report['recommendation']['category'], 'qat-and-scales')

    def test_redirected_metric_source_and_symlink_are_rejected_before_any_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            secret = root / 'protected/result.json'; secret.parent.mkdir(); secret.write_text('SECRET')
            record = campaign.record(secret)
            expected = root / 'training.json'
            with mock.patch.object(campaign, 'verify') as verify, mock.patch.object(campaign, 'read') as read:
                with self.assertRaisesRegex(ValueError, 'fixed unprotected'):
                    attribution.bound_document(record, expected)
            verify.assert_not_called(); read.assert_not_called()
            expected.symlink_to(secret)
            with mock.patch.object(campaign, 'verify') as verify:
                with self.assertRaisesRegex(ValueError, 'fixed unprotected'):
                    attribution.bound_document({**record, 'path': str(expected)}, expected)
            verify.assert_not_called()

    def test_malformed_metric_and_different_float_quantized_groups_fail_closed(self):
        row = seed(.1)
        row['seed_receipt']['float_validation']['canonical_validation']['sign_accuracy'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'invalid unprotected'):
            attribution.seed_metrics(row)
        row = seed(.1)
        row['seed_receipt']['quantized_validation']['successor_ranking']['comparable_groups'] = 199
        with self.assertRaisesRegex(ValueError, 'different groups'):
            attribution.seed_metrics(row)

    def test_downstream_projection_keeps_only_completed_unprotected_summaries(self):
        poison = {'path': '/forbidden/protected-transcripts.json', 'value': 'POISON'}
        shares = {name: 1 / len(attribution.instrumentation.CATEGORIES)
                  for name in attribution.instrumentation.CATEGORIES}
        source = {'category_profile': {'variants': {'baseline': {'shares': shares, 'protected': poison}}},
                  'retained_variants': ['baseline'],
                  'throughput_and_latency': {'combined': {'throughput_gain': .2, 'p95_regression': .02, 'passed': True}},
                  'clocked_strength': {'baseline_wins': 600, 'paired_win_deltas': {'combined': 3}}, 'live': poison}
        suite = {'passed': False, 'equal_weight_improvement': .02, 'paired_95_interval': [-.01, .05],
                 'failures': [], 'opponents': {'rank_4': {'root_pairs': 32, 'candidate_win_rate': .6,
                    'control_win_rate': .58, 'improvement': .02, 'protected': poison}}, 'live': poison}
        development = {'passed': False, 'games': 1000, 'candidate_wins': 540, 'failures': 0,
                       'candidate_wins_by_color': [270, 270], 'paired_lower_95': .49, 'protected': poison}
        previous = {'stage': 'full', 'pilot': {'screen': (None, {'result': {'games': 200,
                    'candidate_wins': 110, 'failures': 0, 'live': poison}})}, 'source_selection': source,
                    'suites': [(suite, None, None)], 'development': development,
                    'stages': {'search': {'unprotected': 'search'}, 'screen': {'unprotected': 'screen'},
                               'development': {'unprotected': 'development'}}}
        with mock.patch.object(campaign, 'read', side_effect=AssertionError('no new result reads')), \
                mock.patch.object(campaign, 'verify', side_effect=AssertionError('no arbitrary source follows')):
            report = attribution.downstream_evidence(previous)
        self.assertNotIn('POISON', json.dumps(report))
        self.assertEqual(report['search']['baseline_wins'], 600)
        self.assertEqual(report['suites'][0]['paired_95_interval'], [-.01, .05])
        self.assertEqual(report['development']['wins_by_color'], [270, 270])
        self.assertFalse(report['search']['category_shares_authorize_speed_retention'])

    def test_sparse_ranking_coverage_cannot_support_ranking_recommendation(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = fixture(Path(tmp), 1, regret=.29)
            def sparse(document):
                for arm in document['arms']:
                    arm['overall'] = ranking(regret=.29, groups=12)
                    arm['early'] = ranking(regret=.29, groups=12)
            row['selection'] = rewrite(row['selection'], sparse)
            evidence = [{'attempt': 1, 'rejection_stage': 'offline-rejected',
                         'phases': [attribution.phase_evidence(row)]}]
            report = attribution.recommendation(evidence)
            self.assertIsNone(report['category'])
            self.assertEqual(len(report['coverage_gaps']), 4)

    def test_receipt_tampering_is_detected_by_reproducing_all_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [fixture(root, n, quantized_sign=.86) for n in (1, 2)]
            with mock.patch.object(attribution.attempts, 'failed_attempt', side_effect=lambda _, n: rows[n - 1]):
                attribution.produce(root)
                path = root / 'attribution/after-two-attempts.json'
                rewrite(campaign.record(path), lambda document: document['recommendation'].update({'attempt_three_may_start': True}))
                with self.assertRaisesRegex(ValueError, 'differs from verified'):
                    attribution.validate(root)

    def test_menu_uses_existing_profiles_without_changing_schedule_or_combining_search(self):
        menu = attribution.profile_menu()
        refined = menu['qat_and_scales']['refined-adaptive-scales-v1']
        self.assertEqual(refined['schedule']['float_warmup_epochs'], 1)
        self.assertEqual(refined['schedule']['qat_epochs'], 4)
        self.assertEqual(refined['quantization']['bits'], 3)
        self.assertEqual(menu['teacher_ranking']['hardest-5pct-2m-v1']['deep_tree_nodes'], 2000000)
        self.assertEqual(len(menu['single_search']), 3)
        self.assertFalse(menu['combine_search_profiles'])
        self.assertFalse(menu['cross_turn_persistence'])


if __name__ == '__main__':
    unittest.main()
