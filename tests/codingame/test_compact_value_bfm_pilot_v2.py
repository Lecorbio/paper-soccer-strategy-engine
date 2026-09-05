"""Failed-attempt proof, immutable continuation, and cross-attempt isolation."""
import copy
from collections import defaultdict
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

from tools import compact_value_bfm_attempt_v2 as attempts
from tools import compact_value_bfm_pilot_v2 as pilot

campaign = attempts.campaign


def rewrite(path, body):
    if path.exists():
        path.unlink()
    return campaign.seal(path, {key: value for key, value in body.items() if key != 'body_sha256'})


def fixture(root, *, eligible=False):
    phase = 'attempt-001-pilot'; context = root / 'phases' / phase; directory = context / phase
    artifact = root / 'input'; campaign.once(artifact, b'frozen input')
    inputs = {key: campaign.record(artifact) for key in ('attempt_one_initial_checkpoint', 'teacher_runtime', 'attempt_zero_runtime')}
    campaign.seal(root / 'campaign.json', {'inputs': inputs})
    contract = campaign.seal(context / 'campaign.json', {'attempt': 1, 'phase': 'pilot', 'policy': campaign.POLICY,
        'parent_campaign': campaign.record(root / 'campaign.json'), 'inputs': inputs, 'exclusions': []})
    rows = []; arms = []
    for weight in (0, .1, .25):
        metrics = {'groups': 150, 'comparable_groups': 130, 'mean_teacher_regret': .2 if weight == 0 else .16,
            'float_vs_quantized_action_flip_rate': .1, 'top1_agreement': .8, 'pairwise_loss': .1, 'loss_weight': weight}
        for seed in attempts.selection.trainer.FIXED_SEEDS:
            scalar = {'objective_weighted_huber': .04, 'weighted_huber': .04, 'sign_accuracy': .9 if eligible else .8}
            validation = {'successor_ranking': metrics, 'common_adjudicator': dict(scalar), 'canonical_validation': dict(scalar)}
            receipt = {'seed': seed, 'float_validation': copy.deepcopy(validation), 'quantized_validation': validation}
            receipt['offline_gate'] = attempts.selection.trainer.offline_advancement_gate(receipt['float_validation'], validation)
            rows.append({'weight': weight, 'seed': seed, 'source': campaign.record(artifact), 'runtime': campaign.record(artifact),
                'float_checkpoint': campaign.record(artifact), 'source_reserve': 2300, 'seed_receipt': receipt})
        selected = rows[-3]
        arms.append({'lambda': weight, 'seed': selected['seed'], 'canonical_retention_passed': eligible,
            'overall': metrics, 'early': copy.deepcopy(metrics), **{key: selected[key] for key in
                ('source', 'runtime', 'float_checkpoint', 'source_reserve')}})
    training = campaign.seal(directory / 'training.json', {'results': rows, 'smoke': False, 'mandatory_training_verified': True})
    campaign.seal(directory / 'selection-policy.json', attempts.selection.SELECTION_POLICY)
    comparisons = [attempts.selection.compare_candidate(arms[0], arm) for arm in arms[1:]]
    selection = campaign.seal(directory / 'model-selection.json', {'policy': campaign.record(directory / 'selection-policy.json'),
        'training': campaign.record(directory / 'training.json'), 'arms': arms, 'comparisons': comparisons,
        'selected': arms[1] if eligible else None, 'pilot_admitted': False, 'campaign_success': False,
        'status': 'model-selected-before-rank4-screen' if eligible else 'offline-rejected-before-rank4-screen'})
    campaign.seal(directory / 'pilot-outcome.json', {'selection': campaign.record(directory / 'model-selection.json'),
        'status': 'offline-rejected', 'admitted': False, 'campaign_success': False})
    return context, phase, contract, training, selection


class FailedPilotProofTests(unittest.TestCase):
    def setUp(self):
        self.early_loader = mock.patch.object(attempts, 'load_early_groups', return_value=('held-out-group',))
        self.early_evaluator = mock.patch.object(attempts, 'evaluate_early',
            side_effect=lambda row, early: row['seed_receipt']['quantized_validation']['successor_ranking'])
        self.early_loader.start(); self.early_evaluator.start()
        self.addCleanup(self.early_loader.stop); self.addCleanup(self.early_evaluator.stop)

    def test_attempt_zero_and_unbound_third_attempt_are_forbidden(self):
        with self.assertRaisesRegex(ValueError, 'attempt zero'):
            pilot.context_root(Path('/tmp/campaign'), 0)
        self.assertEqual(pilot.context_root(Path('/tmp/campaign'), 2).name, 'attempt-002-pilot')
        with self.assertRaisesRegex(ValueError, 'unprotected attribution intervention'):
            pilot.context_root(Path('/tmp/campaign'), 3)

    def test_training_boolean_cannot_replace_all_nine_seed_completions(self):
        with tempfile.TemporaryDirectory() as tmp:
            context = Path(tmp)
            campaign.seal(context / 'pilot/training.json', {'results': [], 'smoke': False, 'mandatory_training_verified': True})
            with self.assertRaisesRegex(ValueError, 'all nine completed'):
                attempts.validate_training(context, 'pilot', {})

    def test_smoke_never_counts_as_a_failed_trained_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            context = Path(tmp)
            campaign.seal(context / 'pilot/training.json', {'results': [], 'smoke': True, 'mandatory_training_verified': True})
            with self.assertRaisesRegex(ValueError, 'nonsmoke'):
                attempts.validate_training(context, 'pilot', {})

    def test_completed_offline_rejection_reproduces_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); context, phase, _, training, _ = fixture(root)
            with mock.patch.object(attempts, 'validate_training', return_value=training) as validator:
                previous = attempts.failed_pilot(root, 1)
            validator.assert_called_once()
            self.assertEqual(previous['outcome'], campaign.record(context / phase / 'pilot-outcome.json'))
            self.assertIsNone(previous['screen'])

    def test_forged_rejection_cannot_hide_an_eligible_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); _, _, _, training, _ = fixture(root, eligible=True)
            with mock.patch.object(attempts, 'validate_training', return_value=training):
                with self.assertRaisesRegex(ValueError, 'eligible or spent screen'):
                    attempts.failed_pilot(root, 1)

    def test_selection_is_bound_to_the_actual_trained_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            context, phase, contract, training, selection = fixture(Path(tmp))
            forged = copy.deepcopy(selection); forged['arms'][1]['source']['sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'completed seed evidence'):
                attempts.validate_selection(forged, training, context, phase, contract)

    def test_resealed_retention_cannot_change_a_seed_verdict(self):
        with tempfile.TemporaryDirectory() as tmp:
            context, phase, contract, training, selection = fixture(Path(tmp), eligible=True)
            forged = copy.deepcopy(training)
            forged['results'][-1]['seed_receipt']['offline_gate']['passed'] = False
            path = context / phase / 'training.json'; rewrite(path, forged)
            selection['training'] = campaign.record(path)
            rewrite(context / phase / 'model-selection.json', selection)
            with self.assertRaisesRegex(ValueError, 'canonical retention verdict does not reproduce'):
                attempts.validate_selection(campaign.read(context / phase / 'model-selection.json'), campaign.read(path),
                    context, phase, contract)

    def test_rejection_cannot_hide_an_interrupted_screen(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); context, phase, _, training, _ = fixture(root)
            campaign.seal(context / phase / 'rank4-screen/execution-claim.json', {'retry_allowed': False})
            with mock.patch.object(attempts, 'validate_training', return_value=training):
                with self.assertRaisesRegex(ValueError, 'spent screen'):
                    attempts.failed_pilot(root, 1)
            (context / phase / 'pilot-outcome.json').unlink()
            with self.assertRaisesRegex(ValueError, 'claimed screen remains spent'):
                attempts.failed_pilot(root, 1)

    def test_full_stage_cannot_be_abandoned_through_a_pilot_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fixture(root)
            (root / 'phases/attempt-001-full').mkdir()
            with self.assertRaisesRegex(ValueError, 'full-stage attempt outcomes'):
                attempts.failed_pilot(root, 1)

    def test_tampered_outcome_and_wrong_parent_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); context, phase, contract, _, _ = fixture(root)
            outcome = context / phase / 'pilot-outcome.json'
            outcome.write_bytes(outcome.read_bytes().replace(b'offline-rejected', b'forged-rejected'))
            with self.assertRaisesRegex(ValueError, 'changed receipt'):
                attempts.failed_pilot(root, 1)
            other = root / 'other/campaign.json'; campaign.seal(other, {'inputs': contract['inputs']})
            contract['parent_campaign'] = campaign.record(other); rewrite(context / 'campaign.json', contract)
            with self.assertRaisesRegex(ValueError, 'expected stage'):
                attempts.failed_pilot(root, 1)


class AttemptCarryTests(unittest.TestCase):
    def test_prepare_second_attempt_preserves_initialization_and_accumulated_exclusions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); context, phase, contract, training, _ = fixture(root)
            directory = context / phase
            for name in ('positions.json', 'games.json'):
                campaign.seal(directory / name, {'frozen': name})
            protected = root / 'exclusions/protected.json'
            campaign.seal(protected, {'role': 'protected', 'domain': campaign.legacy.FEATURE_FINGERPRINT_DOMAIN,
                'fingerprints': ['a' * 64]})
            contract['exclusions'] = [campaign.record(protected)]
            rewrite(context / 'campaign.json', contract)
            campaign.seal(root / 'baseline-engine-comparison.json', {'same_weights': True, 'all_checks_passed': True, 'exclusions': []})
            campaign.seal(root / 'exclusions/anchor-derived.json', {'domain': campaign.legacy.FEATURE_FINGERPRINT_DOMAIN,
                'fingerprints': {'prior-train': ['b' * 64]}})
            campaign.seal(root / 'exclusions/prior-search-validation.json', {'role': 'prior-validation',
                'domain': campaign.legacy.FEATURE_FINGERPRINT_DOMAIN, 'fingerprints': ['c' * 64]})
            parent = campaign.read(root / 'campaign.json'); parent['exclusions'] = []
            rewrite(root / 'campaign.json', parent)
            contract['parent_campaign'] = campaign.record(root / 'campaign.json'); rewrite(context / 'campaign.json', contract)
            previous = {'attempt': 1, 'context': context, 'phase': phase, 'contract': contract, 'screen': None,
                'outcome': campaign.record(directory / 'pilot-outcome.json'), 'training': campaign.record(directory / 'training.json'),
                'selection': campaign.record(directory / 'model-selection.json')}
            values = {('prior-validation', campaign.legacy.FEATURE_FINGERPRINT_DOMAIN): {'d' * 64}}
            with mock.patch.object(pilot, 'validate_smoke', return_value={'smoke': 'bound'}), \
                    mock.patch.object(pilot, 'smoke_exclusions', return_value=[]), \
                    mock.patch.object(attempts, 'failed_pilot', return_value=previous), \
                    mock.patch.object(attempts, 'collect_fingerprints', return_value=(values, 'not-opened')):
                prepared = pilot.prepare(root, 2)
                resumed = pilot.prepare(root, 2)
            self.assertEqual(prepared, resumed)
            self.assertEqual(prepared['candidate_lineage']['initial_float'], parent['inputs']['attempt_one_initial_checkpoint'])
            self.assertEqual(prepared['candidate_lineage']['generation_student'], parent['inputs']['attempt_zero_runtime'])
            self.assertFalse(prepared['candidate_lineage']['smoke_weights_reused'])
            self.assertEqual(prepared['completed_unsuccessful_trained_attempts'], 1)
            self.assertEqual(prepared['pilot_training_roster']['lambdas'], [0, .1, .25])
            excluded = campaign.exclusion_sets(prepared)
            domain = campaign.legacy.FEATURE_FINGERPRINT_DOMAIN
            self.assertEqual(excluded['protected', domain], {'a' * 64})
            self.assertEqual(excluded['prior-validation', domain], {'c' * 64, 'd' * 64})

    def test_validation_root_ancestors_stay_out_and_all_successors_stay_in(self):
        state, prefix = campaign.fresh_root(8, random.Random(42))
        actions = prefix.split('/'); action, successor = campaign.successors(state)[0]
        fps, final = attempts.trajectory_fingerprints(prefix + '/' + action, len(actions))
        domain = campaign.legacy.FEATURE_FINGERPRINT_DOMAIN
        self.assertNotIn(campaign.fingerprints(campaign.features.ReplayState())[domain], fps[domain])
        self.assertIn(campaign.fingerprints(state)[domain], fps[domain])
        self.assertIn(campaign.fingerprints(successor)[domain], fps[domain])
        self.assertEqual(campaign.fingerprints(final), campaign.fingerprints(successor))

    def test_terminal_feature_boundary_is_carried(self):
        state = campaign.features.ReplayState(); actions = []
        while state.winner is None:
            campaign.features.apply_complete_turn(state, state.to_move, '0'); actions.append('0')
        transcript = '/'.join(actions)
        fps, state = attempts.trajectory_fingerprints(transcript, 0)
        self.assertIsNotNone(state.winner)
        self.assertIn(campaign.fingerprints(state)[campaign.legacy.FEATURE_FINGERPRINT_DOMAIN],
            fps[campaign.legacy.FEATURE_FINGERPRINT_DOMAIN])

    def test_new_context_preserves_role_isolation_without_old_receipt_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); old = root / 'old'; phase = 'pilot'; destination = root / 'new'
            for name in ('positions.json', 'games.json', 'pilot-outcome.json', 'training.json', 'model-selection.json'):
                campaign.seal(old / phase / name, {'frozen': name})
            before = {path: path.read_bytes() for path in old.rglob('*') if path.is_file()}
            previous = {'attempt': 1, 'context': old, 'phase': phase,
                **{key: campaign.record(old / phase / name) for key, name in
                    (('outcome', 'pilot-outcome.json'), ('training', 'training.json'), ('selection', 'model-selection.json'))}}
            state = campaign.features.ReplayState(); fps = campaign.fingerprints(state)
            values = defaultdict(set)
            for role in ('prior-train', 'prior-validation', 'mixed-development', 'protected'):
                for domain, value in fps.items():
                    values[role, domain].add(value)
            with mock.patch.object(attempts, 'collect_fingerprints', return_value=(values, 'not-opened')):
                artifacts, receipt = attempts.carry_failed_pilot(root, previous, destination)
                repeated = attempts.carry_failed_pilot(root, previous, destination)
            self.assertEqual((artifacts, receipt), repeated)
            self.assertEqual(before, {path: path.read_bytes() for path in old.rglob('*') if path.is_file()})
            exclusions = campaign.exclusion_sets({'exclusions': artifacts})
            domain = campaign.legacy.FEATURE_FINGERPRINT_DOMAIN
            for role in ('prior-validation', 'mixed-development', 'protected'):
                self.assertEqual(campaign.rejection(state, 'train', {(role, domain): exclusions[role, domain]}), role)
            self.assertIsNone(campaign.rejection(state, 'train', {('prior-train', domain): exclusions['prior-train', domain]}))
            self.assertEqual(campaign.rejection(state, 'validation', {('prior-train', domain): exclusions['prior-train', domain]}), 'prior-train')
            changed = Path(artifacts[0]['path']); changed.write_bytes(changed.read_bytes() + b' ')
            with self.assertRaisesRegex(ValueError, 'changed artifact'):
                campaign.verify(artifacts[0])


class RecomputedSelectionTests(unittest.TestCase):
    def test_resealed_early_metrics_must_match_real_mapped_holdout_and_model(self):
        from tests.codingame.test_compact_value_bfm_ranking_store import RankingStoreTests
        trainer = attempts.selection.trainer
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); context, phase, contract, training, selected = fixture(root)
            _, bundle, _ = RankingStoreTests().fixture(root)
            index = attempts.storage.build_store([root / 'labels.jsonl'], context / phase / 'ranking-store', bundle)
            campaign.once(root / 'bundle-manifest.json', b'synthetic bundle; identity supplied by fixture')
            contract['bundle'] = campaign.record(root / 'bundle-manifest.json')
            rewrite(context / 'campaign.json', contract)
            campaign.seal(context / phase / 'training-input-audit.json', {'ranking_store': campaign.record(index), 'bundle': contract['bundle']})
            training['input_audit'] = campaign.record(context / phase / 'training-input-audit.json')
            architecture = trainer.ARCHITECTURES['capacity-12x8']
            parameters = trainer.initialize_parameters(architecture, 20260907)
            checkpoint = trainer.write_float_checkpoint(root / 'model', parameters, architecture)
            quantized = trainer.quantize_fixed(parameters, architecture, {name: .1 for name in parameters})
            runtime = trainer.write_runtime(root / 'model', architecture, quantized, arm=trainer.ARMS['search-target'],
                seed=20260907, float_epoch=1, qat_epoch=4, source_bundle_body_sha256=bundle.body_sha256)
            for row in training['results']:
                row['runtime'] = campaign.record(runtime); row['float_checkpoint'] = campaign.record(checkpoint)
            with mock.patch.object(trainer.FrozenBundle, 'load', return_value=bundle):
                early = attempts.load_early_groups(context, phase, contract, training)
                self.assertEqual(len(early), 1)  # The separate training group is excluded.
                real_metrics = attempts.evaluate_early(training['results'][0], early)
                for arm in selected['arms']:
                    arm['runtime'] = campaign.record(runtime); arm['float_checkpoint'] = campaign.record(checkpoint)
                    arm['early'] = dict(real_metrics)
                selected['comparisons'] = [attempts.selection.compare_candidate(selected['arms'][0], arm) for arm in selected['arms'][1:]]
                rewrite(context / phase / 'training.json', training)
                selected['training'] = campaign.record(context / phase / 'training.json')
                rewrite(context / phase / 'model-selection.json', selected)
                self.assertIsNone(attempts.validate_selection(campaign.read(context / phase / 'model-selection.json'),
                    campaign.read(context / phase / 'training.json'), context, phase, contract))
                selected['arms'][1]['early']['mean_teacher_regret'] += .01
                selected['comparisons'] = [attempts.selection.compare_candidate(selected['arms'][0], arm) for arm in selected['arms'][1:]]
                rewrite(context / phase / 'model-selection.json', selected)
                with self.assertRaisesRegex(ValueError, 'early metrics do not reproduce'):
                    attempts.validate_selection(campaign.read(context / phase / 'model-selection.json'),
                        campaign.read(context / phase / 'training.json'), context, phase, contract)

    def test_numeric_environment_is_checked_before_loading_the_model(self):
        trainer = attempts.selection.trainer
        with mock.patch.object(trainer, 'native_thread_execution_scope', side_effect=ValueError('wrong preimport environment')), \
                mock.patch.object(trainer, 'load_runtime') as load:
            with self.assertRaisesRegex(ValueError, 'wrong preimport environment'):
                attempts.evaluate_early({}, ())
        load.assert_not_called()


class PlayedScreenClosureTests(unittest.TestCase):
    def test_resealed_played_exclusions_cannot_drop_a_continuation_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            campaign.seal(directory / 'rank4-screen/execution.json', {'raw': 'source-bound test execution'})
            raw = {'config': {'trajectory_schema': 'papersoccer.compact-value-bfm-rank4-trajectories.v1'},
                'games': [{'root_transcript': '0', 'transcript': '0/0/0'}]}
            bank = {'tsv': {'sha256': 'a' * 64}, 'exclusions': []}
            values, _ = attempts.trajectory_fingerprints('0/0/0', 1)
            records = []
            for ordinal, (domain, fingerprints) in enumerate(sorted(values.items())):
                path = directory / 'rank4-screen' / f'played-exclusion-{ordinal}.json'
                campaign.seal(path, {'schema': campaign.ID + '.pilot-screen-played-exclusions.v2', 'role': 'mixed-development',
                    'domain': domain, 'fingerprints': sorted(fingerprints), 'execution': campaign.record(directory / 'rank4-screen/execution.json'),
                    'bank_sha256': bank['tsv']['sha256'], 'contains_transcripts': False, 'contains_labels': False, 'contains_metrics': False,
                    'includes_all_played_postroot_boundaries': True, 'includes_terminal_features': True})
                records.append(campaign.record(path))
            outcome = {'played_trajectory_closure_preserved': True, 'development_exclusions': records}
            attempts.validate_played_exclusions(directory, outcome, bank, raw)
            path = Path(records[0]['path']); altered = campaign.read(path); altered['fingerprints'].pop()
            rewrite(path, altered); records[0] = campaign.record(path)
            with self.assertRaisesRegex(ValueError, 'do not reproduce all played boundaries'):
                attempts.validate_played_exclusions(directory, outcome, bank, raw)

    def test_root_only_screen_cannot_close_a_v2_attempt(self):
        with self.assertRaisesRegex(ValueError, 'every source-bound played trajectory'):
            attempts.validate_played_exclusions(Path('/unused'), {}, {}, {'config': {}})


if __name__ == '__main__':
    unittest.main()
