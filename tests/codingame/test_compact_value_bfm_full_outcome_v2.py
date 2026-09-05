"""Completed full-attempt rejection and lossless cross-attempt isolation."""
from collections import defaultdict
from contextlib import ExitStack
import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools import compact_value_bfm_full_outcome_v2 as outcome

campaign = outcome.campaign
attempts = outcome.attempts


def rewrite(path, body):
    if path.exists():
        path.unlink()
    return campaign.seal(path, {key: value for key, value in body.items() if key != 'body_sha256'})


class FullSelectionReopeningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.phase = 'attempt-001-full'
        self.context = self.root / 'phases' / self.phase
        self.directory = self.context / self.phase
        artifact = self.root / 'frozen-input'
        campaign.once(artifact, b'fixed input')
        self.record = campaign.record(artifact)
        campaign.seal(self.root / 'campaign.json', {'frozen': True})
        self.contract = campaign.seal(self.context / 'campaign.json', {
            'parent_campaign': campaign.record(self.root / 'campaign.json'),
            'admitted_pilot': self.record, 'bundle': self.record,
            'inputs': {'attempt_one_initial_checkpoint': self.record},
            'full_training_roster': {'lambdas': [0, .1], 'seeds': list(outcome.full_selection.trainer.FIXED_SEEDS)}})
        self.parent = {'inputs': {'discrete_v3_deployment.cpp': self.record, 'attempt_zero_runtime': self.record}}
        campaign.seal(self.directory / 'full-selection-policy.json', {
            **outcome.full_selection.policy(self.contract), 'context': campaign.record(self.context / 'campaign.json'),
            'source_closure': [self.record]})
        self.rows = [{'weight': weight, 'seed': seed,
                      'seed_receipt': {'binding': {'datasets': {'fixed': 'dataset'}, 'source_routes': {'fixed': 'route'}}}}
                     for weight in (0, .1) for seed in outcome.full_selection.trainer.FIXED_SEEDS]
        self.training = {'schema': campaign.ID + '.training.v2', 'smoke': False,
                         'mandatory_training_verified': True, 'results': self.rows,
                         'producer': self.record, 'input_audit': self.record}
        campaign.seal(self.directory / 'training.json', self.training)
        self.scalar = {'lambda': 0, 'eligible_for_multi_opponent': True}
        self.candidate = {'lambda': .1, 'eligible_for_multi_opponent': False}
        self.document = {'schema': campaign.ID + '.full-model-selection.v2',
            'policy': campaign.record(self.directory / 'full-selection-policy.json'),
            'context': campaign.record(self.context / 'campaign.json'), 'parent_campaign': self.contract['parent_campaign'],
            'admitted_pilot': self.record, 'training': campaign.record(self.directory / 'training.json'),
            'input_audit': self.record, 'ranking_store': self.record, 'seed_references': [self.record] * 6,
            'arms': [self.scalar, self.candidate], 'scalar_control': self.scalar,
            'frozen_deployed_control': {'source': self.record, 'runtime': self.record, 'role': 'frozen-deployed-strength-control'},
            'selected': None, 'diagnostics': {'diagnostic': True}, 'eligible_for_multi_opponent': False,
            'status': 'full-model-offline-rejected', 'game_banks_opened': False,
            'qualification_passed': False, 'campaign_success': False}
        campaign.seal(self.directory / 'full-model-selection.json', self.document)
        self.stack = self.enterContext(ExitStack())
        full = outcome.full_selection
        patches = [mock.patch.object(full, 'validate_context', return_value=(self.contract, self.parent)),
            mock.patch.object(full.selection, '_validate_seed_roster'),
            mock.patch.object(full.trainer.FrozenBundle, 'load', return_value='bundle'),
            mock.patch.object(full, 'validate_inputs', return_value=({'ranking_store': self.record}, 'rankings')),
            mock.patch.object(full.trainer, 'load_float_checkpoint', return_value='initial'),
            mock.patch.object(full, 'select_arms', return_value=[self.scalar, self.candidate]),
            mock.patch.object(full, 'ranking_diagnostic', return_value={'diagnostic': True})]
        for patch in patches:
            self.stack.enter_context(patch)
        self.seed_validator = self.stack.enter_context(mock.patch.object(full, 'validate_seed', return_value={'reference': self.record}))

    def validate(self):
        return outcome.validate_full_selection(self.root, self.context, self.phase)

    def test_all_six_seed_validators_reopened_before_accepting_summary(self):
        self.assertEqual(self.validate()[1], campaign.read(self.directory / 'full-model-selection.json'))
        self.assertEqual(self.seed_validator.call_count, 6)
        self.assertEqual({(call.args[2]['weight'], call.args[2]['seed']) for call in self.seed_validator.call_args_list},
                         {(weight, seed) for weight in (0, .1) for seed in outcome.full_selection.trainer.FIXED_SEEDS})

    def test_missing_duplicate_or_smoke_training_cannot_close_full_attempt(self):
        good = copy.deepcopy(self.training)
        for mutation in ('missing', 'duplicate', 'smoke'):
            body = copy.deepcopy(good)
            if mutation == 'missing': body['results'].pop()
            if mutation == 'duplicate': body['results'][-1] = body['results'][-2]
            if mutation == 'smoke': body['smoke'] = True
            rewrite(self.directory / 'training.json', body)
            document = {**self.document, 'training': campaign.record(self.directory / 'training.json')}
            rewrite(self.directory / 'full-model-selection.json', document)
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, 'six real nonsmoke'):
                self.validate()

    def test_actual_seed_validation_failure_cannot_be_relabelled_as_rejection(self):
        self.seed_validator.side_effect = ValueError('quantized initialization reused')
        with self.assertRaisesRegex(ValueError, 'initialization reused'):
            self.validate()

    def test_resealed_false_summary_and_changed_policy_source_are_rejected(self):
        rewrite(self.directory / 'full-model-selection.json', {**self.document, 'seed_references': [self.record] * 5})
        with self.assertRaisesRegex(ValueError, 'all six completed'):
            self.validate()
        rewrite(self.directory / 'full-model-selection.json', self.document)
        Path(self.record['path']).write_bytes(b'replacement')
        with self.assertRaisesRegex(ValueError, 'changed artifact'):
            self.validate()


class TerminalStageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.phase = 'attempt-001-full'
        self.context = self.root / 'phases' / self.phase
        self.directory = self.context / self.phase
        self.model = {'selected': {'source': 'trained'}, 'eligible_for_multi_opponent': True,
                      'status': 'full-model-selected-awaiting-strength-evaluation'}
        self.source = {'selected': {'source': 'exact source'}, 'required_ablation_complete': True,
                       'required_category_profile_complete': True, 'required_profiling_complete': True,
                       'incomplete_requirements': [], 'eligible_for_multi_opponent': True,
                       'status': 'source-selected-awaiting-multi-opponent'}
        campaign.seal(self.directory / 'search/search-selection.json', self.source)
        self.search_record = campaign.record(self.directory / 'search/search-selection.json')

    def run_stage(self):
        return outcome.completed_rejection(self.root, self.context, self.phase, self.model)

    def make_suite(self, stage, passed):
        assessment = campaign.seal(self.directory / 'multi-opponent' / stage / 'assessment.json', {'passed': passed})
        return assessment, {'selection': self.search_record}, {'pairs': []}

    def test_missing_search_and_incomplete_category_profile_do_not_count(self):
        with self.assertRaises((KeyError, FileNotFoundError, ValueError)):
            self.run_stage()
        for key in ('required_ablation_complete', 'required_category_profile_complete', 'required_profiling_complete'):
            with mock.patch.object(outcome.search, 'validate_selection', return_value={**self.source, key: False}):
                with self.assertRaisesRegex(ValueError, 'incomplete profiling'):
                    self.run_stage()

    def test_completed_source_rejection_requires_all_profiles_and_no_later_claim(self):
        rejected = {**self.source, 'selected': None, 'eligible_for_multi_opponent': False, 'status': 'search-strength-rejected'}
        with mock.patch.object(outcome.search, 'validate_selection', return_value=rejected), \
                mock.patch.object(outcome.suite, '_completed_suite') as suite_validator:
            self.assertEqual(self.run_stage()[0], 'search')
            campaign.seal(self.directory / 'multi-opponent/screen/execution-claim.json', {'spent': True})
            with self.assertRaisesRegex(ValueError, 'cannot hide played evidence'):
                self.run_stage()
        suite_validator.assert_not_called()

    def test_generic_failed_summary_cannot_replace_actual_suite_validator(self):
        self.make_suite('screen', False)
        with mock.patch.object(outcome.search, 'validate_selection', return_value=self.source):
            with self.assertRaises(KeyError):
                self.run_stage()

    def test_completed_screen_or_confirmation_failure_closes_only_that_stage(self):
        screen = self.make_suite('screen', False)
        with mock.patch.object(outcome.search, 'validate_selection', return_value=self.source), \
                mock.patch.object(outcome.suite, '_completed_suite', return_value=screen) as suites, \
                mock.patch.object(outcome.development, 'completed_development') as dev:
            result = self.run_stage()
        self.assertEqual(result[0], 'screen')
        self.assertEqual(suites.call_count, 1)
        dev.assert_not_called()
        screen = ({'passed': True}, screen[1], screen[2])
        confirmation = self.make_suite('confirmation', False)
        with mock.patch.object(outcome.search, 'validate_selection', return_value=self.source), \
                mock.patch.object(outcome.suite, '_completed_suite', side_effect=[screen, confirmation]):
            result = self.run_stage()
        self.assertEqual(result[0], 'confirmation')
        self.assertEqual(len(result[3]), 2)

    def test_missing_development_is_incomplete_and_passing_development_is_not_rejection(self):
        screen, confirmation = self.make_suite('screen', True), self.make_suite('confirmation', True)
        with mock.patch.object(outcome.search, 'validate_selection', return_value=self.source), \
                mock.patch.object(outcome.suite, '_completed_suite', side_effect=[screen, confirmation]), \
                mock.patch.object(outcome.development, 'completed_development', side_effect=ValueError('incomplete shard')):
            with self.assertRaisesRegex(ValueError, 'incomplete shard'):
                self.run_stage()
        campaign.seal(self.directory / 'development/assessment.json', {'passed': True})
        for passed in (True, False):
            status = 'development-passed-awaiting-exact-source-ci-and-protected-gates' if passed else 'development-rejected'
            with mock.patch.object(outcome.search, 'validate_selection', return_value=self.source), \
                    mock.patch.object(outcome.suite, '_completed_suite', side_effect=[screen, confirmation]), \
                    mock.patch.object(outcome.development, 'completed_development', return_value={'passed': passed, 'status': status}):
                if passed:
                    with self.assertRaisesRegex(ValueError, 'passing or incomplete'):
                        self.run_stage()
                else:
                    self.assertEqual(self.run_stage()[0], 'development')

    def test_incomplete_full_stage_cannot_fall_back_to_pilot_rejection(self):
        with mock.patch.object(attempts, 'failed_pilot') as pilot:
            with self.assertRaisesRegex(ValueError, 'no verified terminal outcome'):
                attempts.failed_attempt(self.root, 1)
        pilot.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'intervention binding'):
            attempts.failed_attempt(self.root, 3)


class FullCarryTests(unittest.TestCase):
    def test_next_pilot_preserves_original_float_teacher_and_both_corpora(self):
        from tools import compact_value_bfm_pilot_v2 as pilot
        from tests.codingame.test_compact_value_bfm_pilot_v2 import fixture
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pilot_context, pilot_phase, pilot_contract, _, _ = fixture(root)
            parent = campaign.read(root / 'campaign.json')
            rewrite(root / 'campaign.json', {**parent, 'exclusions': []})
            phase = 'attempt-001-full'; context = root / 'phases' / phase
            protected = root / 'old-protected-fingerprints.json'
            domain = campaign.legacy.FEATURE_FINGERPRINT_DOMAIN
            campaign.seal(protected, {'role': 'protected', 'domain': domain, 'fingerprints': ['a' * 64]})
            full_contract = campaign.seal(context / 'campaign.json', {**pilot_contract, 'phase': 'full',
                'parent_campaign': campaign.record(root / 'campaign.json'), 'exclusions': [campaign.record(protected)]})
            for name in ('positions.json', 'games.json', 'attempt-outcome.json', 'training.json', 'full-model-selection.json'):
                campaign.seal(context / phase / name, {'frozen': name})
            previous = {'stage': 'full', 'attempt': 1, 'context': context, 'phase': phase, 'contract': full_contract,
                        'rejection_stage': 'full-offline', 'seed_references': [], 'stages': {},
                        'pilot': {key: campaign.record(pilot_context / pilot_phase / name) for key, name in
                                  (('outcome', 'pilot-outcome.json'), ('training', 'training.json'), ('selection', 'model-selection.json'))},
                        **{key: campaign.record(context / phase / name) for key, name in
                           (('outcome', 'attempt-outcome.json'), ('training', 'training.json'), ('selection', 'full-model-selection.json'))}}
            campaign.seal(root / 'baseline-engine-comparison.json', {'same_weights': True, 'all_checks_passed': True, 'exclusions': []})
            campaign.seal(root / 'exclusions/anchor-derived.json', {'domain': domain, 'fingerprints': {}})
            campaign.seal(root / 'exclusions/prior-search-validation.json', {'role': 'prior-validation', 'domain': domain, 'fingerprints': []})
            before = {path: path.read_bytes() for path in context.rglob('*') if path.is_file()}
            values = defaultdict(set, {('prior-train', domain): {'b' * 64, 'c' * 64},
                                       ('prior-validation', domain): {'d' * 64, 'e' * 64},
                                       ('mixed-development', domain): {'f' * 64}})
            with mock.patch.object(pilot, 'validate_smoke', return_value={'smoke': 'bound'}), \
                    mock.patch.object(pilot, 'smoke_exclusions', return_value=[]), \
                    mock.patch.object(outcome, 'failed_full', return_value=previous), \
                    mock.patch.object(outcome, 'collect_fingerprints', return_value=values):
                prepared = pilot.prepare(root, 2)
                self.assertEqual(prepared, pilot.prepare(root, 2))
            self.assertEqual(before, {path: path.read_bytes() for path in context.rglob('*') if path.is_file()})
            self.assertEqual(prepared['completed_unsuccessful_trained_attempts'], 1)
            self.assertEqual(prepared['inputs'], parent['inputs'])
            self.assertEqual(prepared['candidate_lineage']['initial_float'], parent['inputs']['attempt_one_initial_checkpoint'])
            self.assertEqual(prepared['candidate_lineage']['generation_student'], parent['inputs']['attempt_zero_runtime'])
            carried = campaign.read(campaign.verify(prepared['previous_failed_attempts'][0]['carry']))
            self.assertTrue(carried['pilot_and_full_corpora_included'])
            self.assertEqual(carried['completed_attempt_count'], 1)
            excluded = campaign.exclusion_sets(prepared)
            for role in ('prior-train', 'prior-validation', 'mixed-development'):
                self.assertTrue(values[role, domain] <= excluded[role, domain])
            self.assertEqual(excluded['protected', domain], {'a' * 64})

    def test_pilot_full_search_candidate_control_and_terminal_boundaries_are_unioned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = campaign.features.ReplayState()
            transcript = []
            while state.winner is None:
                campaign.features.apply_complete_turn(state, state.to_move, '0')
                transcript.append('0')
            terminal = campaign.fingerprints(state)[campaign.legacy.FEATURE_FINGERPRINT_DOMAIN]
            raw = root / 'search-result.json'
            campaign.once(raw, campaign.raw({'games': [{'root_transcript': '0', 'transcript': '/'.join(transcript)}]}))
            execution = root / 'search-execution.json'
            campaign.seal(execution, {'raw': campaign.record(raw)})
            strength = root / 'strength.json'
            campaign.seal(strength, {'executions': {'rejected-variant': campaign.record(execution)}})
            pair = root / 'pair.json'
            native = root / 'native.jsonl'
            campaign.once(native, b'two already validated native games')
            campaign.seal(pair, {'root': {'transcript': '0'}, 'opponent': 'rank_4',
                                 'arms': {arm: {'output': campaign.record(native)} for arm in ('candidate', 'control')}})
            domain = campaign.legacy.FEATURE_FINGERPRINT_DOMAIN
            previous = {'pilot': {'phase': 'pilot'}, 'source_selection': {'strength': campaign.record(strength)},
                        'suites': [({}, {}, {'pairs': [campaign.record(pair)]})], 'development': None}
            values = defaultdict(set, {('prior-train', domain): {'full-train'}, ('prior-validation', domain): {'full-val'}})
            pilot_values = defaultdict(set, {('prior-train', domain): {'pilot-train'}, ('prior-validation', domain): {'pilot-val'}})
            games = {(0, 0): {'trajectory': '/'.join(transcript), 'root_transcript': '0'}}
            with mock.patch.object(attempts, 'collect_fingerprints', side_effect=[(values, ''), (pilot_values, '')]) as corpora, \
                    mock.patch.object(outcome.suite, 'checked_games', return_value=games) as checked:
                actual = outcome.collect_fingerprints(previous)
            self.assertEqual(corpora.call_args_list[0].kwargs, {'expected_games': 10000})
            self.assertEqual(actual['prior-train', domain], {'pilot-train', 'full-train'})
            self.assertEqual(actual['prior-validation', domain], {'pilot-val', 'full-val'})
            self.assertEqual(checked.call_count, 2)  # Both candidate and deployed control are preserved.
            self.assertIn(terminal, actual['mixed-development', domain])
            initial = campaign.fingerprints(campaign.features.ReplayState())[domain]
            self.assertNotIn(initial, actual['mixed-development', domain])

    def test_terminal_receipt_is_reproduced_and_an_incomplete_bridge_cannot_write_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            phase = 'attempt-001-full'; context = root / 'phases' / phase
            for path in (context / 'campaign.json', context / phase / 'positions.json', context / phase / 'games.json'):
                campaign.seal(path, {'frozen': path.name})
            previous = {'stage': 'full', 'attempt': 1, 'context': context, 'phase': phase,
                        'rejection_stage': 'full-offline', 'training': {}, 'selection': {}, 'seed_references': [],
                        'stages': {}, 'pilot': {'outcome': {}, 'training': {}, 'selection': {}}}
            with mock.patch.object(outcome, 'validated_evidence', side_effect=ValueError('missing sixth seed')):
                with self.assertRaisesRegex(ValueError, 'missing sixth'):
                    outcome.record_outcome(root, 1)
            path = context / phase / 'attempt-outcome.json'
            self.assertFalse(path.exists())
            with mock.patch.object(outcome, 'validated_evidence', return_value=previous):
                document = outcome.record_outcome(root, 1)
                self.assertEqual(document, outcome.record_outcome(root, 1))
                self.assertEqual(outcome.failed_full(root, 1)['outcome'], campaign.record(path))
                rewrite(path, {**document, 'rejection_stage': 'protected'})
                with self.assertRaisesRegex(ValueError, 'differs from completed unprotected'):
                    outcome.failed_full(root, 1)
            self.assertEqual(document['completed_attempt_count'], 1)
            self.assertTrue(document['pilot_and_full_are_one_attempt'])
            self.assertFalse(document['protected_results_used'])


if __name__ == '__main__':
    unittest.main()
