import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from tools import compact_value_bfm_full_selection_v2 as full_selection


campaign = full_selection.campaign
trainer = full_selection.trainer


def metrics(regret=.2, *, groups=200, comparable=180, flip=.1):
    return {'groups': groups, 'comparable_groups': comparable, 'mean_teacher_regret': regret,
            'top1_agreement': .8, 'float_vs_quantized_action_flip_rate': flip,
            'pairwise_loss': .1, 'loss_weight': 0.0}


def seed_receipt(seed, *, passed=True, loss=.1, regret=.2):
    validation = {'objective_weighted_huber': loss, 'sign_accuracy': .9}
    return {'seed': seed, 'offline_gate': {'passed': passed},
            'quantized_validation': {'common_adjudicator': dict(validation),
                                    'canonical_validation': dict(validation),
                                    'successor_ranking': metrics(regret)}}


class FullContextTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.phase = 'attempt-001-full'
        self.context = self.root / 'phases' / self.phase
        self.pilot = self.root / 'phases/attempt-001-pilot'
        self.inputs = {}
        for key in ('attempt_one_initial_checkpoint', 'teacher_runtime', 'teacher_manifest',
                    'attempt_zero_runtime', 'discrete_v3_deployment.cpp'):
            path = self.root / 'inputs' / key
            campaign.once(path, key.encode())
            self.inputs[key] = campaign.record(path)
        bundle_path = self.root / 'bundle.json'
        campaign.once(bundle_path, b'opaque frozen bundle')
        self.parent = {'inputs': self.inputs, 'bundle': campaign.record(bundle_path),
                       'policy': campaign.POLICY}
        campaign.seal(self.root / 'campaign.json', self.parent)
        pilot_contract = {**self.parent, 'attempt': 1, 'phase': 'pilot', 'exclusions': [],
                          'parent_campaign': campaign.record(self.root / 'campaign.json')}
        campaign.seal(self.pilot / 'campaign.json', pilot_contract)
        runtime_path = self.pilot / 'admitted.runtime'
        source_path = self.pilot / 'admitted.cpp'
        campaign.once(runtime_path, b'admitted student')
        campaign.once(source_path, b'admitted source')
        selected = {'lambda': .1, 'runtime': campaign.record(runtime_path), 'source': campaign.record(source_path)}
        self.outcome = {'status': 'pilot-admitted', 'selected': selected, 'admitted': True,
                        'games': 200, 'wins': 105, 'failures': 0, 'development_exclusions': []}
        outcome_path = self.pilot / 'attempt-001-pilot/pilot-outcome.json'
        campaign.seal(outcome_path, self.outcome)
        self.contract = {
            **pilot_contract, 'phase': 'full', 'full_games': 10000,
            'heavy_stage_root': str(self.root), 'pilot_context': campaign.record(self.pilot / 'campaign.json'),
            'admitted_pilot': campaign.record(outcome_path),
            'inputs': {**self.inputs, 'attempt_zero_runtime': selected['runtime']},
            'full_training_roster': {'lambdas': [0, .1], 'seeds': list(trainer.FIXED_SEEDS)},
            'candidate_lineage': {'mandatory_training': True,
                'initial_float': self.inputs['attempt_one_initial_checkpoint'],
                'generation_student': selected['runtime'], 'pilot_source': selected['source'],
                'smoke_weights_reused': False},
        }

    def validate(self):
        campaign.seal(self.context / 'campaign.json', self.contract)
        return full_selection.validate_context(self.root, self.context, self.phase)

    def test_original_float_and_actual_admission_are_reopened(self):
        with mock.patch.object(full_selection.full, 'admitted_pilot', return_value=self.outcome) as admission:
            contract, parent = self.validate()
        admission.assert_called_once_with(self.pilot, 'attempt-001-pilot')
        self.assertEqual(contract['inputs']['attempt_one_initial_checkpoint'], parent['inputs']['attempt_one_initial_checkpoint'])
        self.assertNotEqual(contract['inputs']['attempt_zero_runtime'], parent['inputs']['attempt_zero_runtime'])

    def test_hash_valid_replacement_initialization_cannot_be_selected(self):
        replacement = self.root / 'replacement.float.npz'
        campaign.once(replacement, b'different checkpoint')
        self.contract['inputs']['attempt_one_initial_checkpoint'] = campaign.record(replacement)
        self.contract['candidate_lineage']['initial_float'] = campaign.record(replacement)
        with mock.patch.object(full_selection.full, 'admitted_pilot', return_value=self.outcome):
            with self.assertRaisesRegex(ValueError, 'original float'):
                self.validate()

    def test_other_phase_and_changed_parent_are_rejected(self):
        self.contract['heavy_stage_root'] = str(self.root / 'other')
        with self.assertRaisesRegex(ValueError, 'parent campaign'):
            self.validate()

    def test_unadmitted_pilot_is_not_authorized_by_full_context(self):
        path = Path(self.contract['admitted_pilot']['path'])
        path.unlink()
        campaign.seal(path, {**self.outcome, 'status': 'pilot-screen-rejected', 'admitted': False, 'wins': 104})
        self.contract['admitted_pilot'] = campaign.record(path)
        with self.assertRaisesRegex(ValueError, 'actually admitted pilot'):
            self.validate()

    def test_extra_ranking_recipe_does_not_enter_full_selection(self):
        self.contract['full_training_roster']['lambdas'] = [0, .1, .25]
        with mock.patch.object(full_selection.full, 'admitted_pilot', return_value=self.outcome):
            with self.assertRaisesRegex(ValueError, 'admitted ranking recipe'):
                self.validate()

    def test_standard_attempt_cannot_substitute_a_registered_intervention_profile(self):
        self.contract['qat_profile'] = 'refined-adaptive-scales-v1'
        self.contract['qat_profile_contract'] = trainer.qat_profile_contract(self.contract['qat_profile'])
        with self.assertRaises(ValueError):
            self.validate()


class FullRosterAndSelectionTests(unittest.TestCase):
    def roster(self):
        return {'schema': campaign.ID + '.training.v2', 'smoke': False, 'mandatory_training_verified': True,
                'results': [{'weight': weight, 'seed': seed} for weight in (0, .1) for seed in trainer.FIXED_SEEDS]}

    def test_smoke_missing_and_duplicate_seeds_cannot_satisfy_roster(self):
        good = self.roster()
        self.assertEqual(len(full_selection.validate_roster(good, [0, .1])), 6)
        variants = [copy.deepcopy(good) for _ in range(4)]
        variants[0]['smoke'] = True
        variants[1]['results'].pop()
        variants[2]['results'][-1] = variants[2]['results'][-2]
        variants[3]['results'].append(variants[3]['results'][-1])
        for invalid in variants:
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, 'six real nonsmoke'):
                    full_selection.validate_roster(invalid, [0, .1])

    def test_hash_valid_other_corpus_cannot_supply_full_scalar_targets(self):
        audit = {'shards': {'train': {'manifest': {'sha256': 'full-manifest', 'path': '/full/manifest.json'},
                                    'npz': {'sha256': 'full-npz'}}}}
        binding = {'datasets': {'new': {'source_manifest_sha256': 'full-manifest', 'source_npz_sha256': 'full-npz'}},
                   'source_routes': {'new': ['/full/manifest.json']},
                   'settings': {'seeds': list(trainer.FIXED_SEEDS), 'batch_size': 256,
                                'new_rows_per_batch': 64, 'anchor_rows_per_batch': 192,
                                'new_loss_share': .25, 'anchor_loss_share': .75, 'qat_epochs': 4,
                                'qat_profile': trainer.STANDARD_QAT_PROFILE}}
        full_selection.validate_seed_corpus_binding(binding, audit)
        changed_profile = copy.deepcopy(binding)
        changed_profile['settings']['qat_profile'] = 'refined-adaptive-scales-v1'
        with self.assertRaisesRegex(ValueError, 'batch/QAT'):
            full_selection.validate_seed_corpus_binding(changed_profile, audit)
        full_selection.validate_seed_corpus_binding(changed_profile, audit,
                                                    qat_profile='refined-adaptive-scales-v1')
        binding['datasets']['new']['source_npz_sha256'] = 'valid-pilot-npz'
        with self.assertRaisesRegex(ValueError, 'new-shard'):
            full_selection.validate_seed_corpus_binding(binding, audit)

    def test_maintained_seed_key_prefers_passing_seed_without_second_regret_gate(self):
        records = []
        evidence = {}
        for weight in (0, .1):
            for index, seed in enumerate(trainer.FIXED_SEEDS):
                # The best diagnostic loss fails retention. Among passing seeds,
                # the maintained key chooses seed 08 even when 09 has lower regret.
                receipt = seed_receipt(seed, passed=index > 0, loss=(.01, .1, .2)[index],
                                       regret=.2 if weight == 0 else (.8, .7, .1)[index])
                records.append({'weight': weight, 'seed': seed, 'seed_receipt': receipt,
                                'runtime': {'path': str(seed)}, 'source': {'path': f'{weight}-{seed}'},
                                'float_checkpoint': {}, 'source_reserve': 2500})
                evidence[weight, seed] = {'parameters': {}, 'architecture': None, 'quantized': None,
                                         'reference': {}, 'runtime_body_sha256': 'runtime', 'payload_sha256': 'payload'}
        groups = [SimpleNamespace(evidence={'source_binding': {'prefix': [{'action': '0' * 12}]}}),
                  SimpleNamespace(evidence={'source_binding': {'prefix': [{'action': '0' * 13}]}})]
        rankings = SimpleNamespace(validation=groups)
        thin = metrics(.8, groups=1, comparable=1, flip=.7)
        with mock.patch.object(trainer, 'successor_ranking_metrics', return_value=thin) as evaluate:
            scalar, candidate = full_selection.select_arms(records, evidence, rankings)
        self.assertEqual(scalar['seed'], 20260908)
        self.assertEqual(candidate['seed'], 20260908)
        self.assertTrue(candidate['eligible_for_multi_opponent'])
        self.assertEqual(evaluate.call_args.args[2], (groups[0],))
        diagnostic = full_selection.ranking_diagnostic(scalar, candidate)
        self.assertLess(diagnostic['overall']['regret_reduction'], 0)
        self.assertFalse(diagnostic['overall']['used_as_full_advancement_gate'])
        self.assertFalse(diagnostic['early']['sufficient_comparable_evidence'])
        self.assertIsNone(diagnostic['early']['regret_reduction'])

    def test_empty_early_evidence_is_explicit_and_never_scored_as_a_pass(self):
        empty = {'groups': 0, 'comparable_groups': 0, 'mean_teacher_regret': None,
                 'float_vs_quantized_action_flip_rate': None}
        control = {'overall': metrics(), 'early': empty}
        candidate = {'overall': metrics(.1), 'early': empty}
        evidence = full_selection.ranking_diagnostic(control, candidate)
        self.assertFalse(evidence['early']['control_coverage_passed'])
        self.assertFalse(evidence['early']['candidate_coverage_passed'])
        self.assertIsNone(evidence['early']['flip_rate_increase'])

    def test_source_reserve_and_canonical_failure_remain_vetoes(self):
        for passed, reserve in ((False, 2800), (True, 1999)):
            records = []
            evidence = {}
            for weight in (0, .1):
                for seed in trainer.FIXED_SEEDS:
                    records.append({'weight': weight, 'seed': seed, 'seed_receipt': seed_receipt(seed, passed=passed),
                                    'runtime': {}, 'source': {}, 'float_checkpoint': {}, 'source_reserve': reserve})
                    evidence[weight, seed] = {'parameters': {}, 'architecture': None, 'quantized': None,
                                             'reference': {}, 'runtime_body_sha256': 'runtime', 'payload_sha256': 'payload'}
            scalar, candidate = full_selection.select_arms(records, evidence, SimpleNamespace(validation=()))
            self.assertFalse(candidate['eligible_for_multi_opponent'])
            self.assertEqual(candidate['early']['status'], 'no-eligible-early-validation-groups')


class ActualArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.architecture = trainer.ARCHITECTURES['capacity-12x8']
        self.initial = {name: np.zeros(shape, dtype=np.float32) for name, shape in (
            ('w1', (6301, 12)), ('w2', (12, 8)), ('w3', (8,)))}
        self.parameters = {name: value + np.float32(.1 if name == 'w1' else .001)
                           for name, value in self.initial.items()}
        self.quantized = trainer.quantize_fixed(self.parameters, self.architecture, {name: .1 for name in self.parameters})

    def row(self):
        initial_quantized = trainer.quantize_fixed(self.initial, self.architecture, self.quantized.scales)
        return {'master_updates': trainer._parameter_update_evidence(self.initial, self.parameters),
                'quantized_changes_vs_initialization': trainer._quantized_update_evidence(initial_quantized, self.quantized)}

    def test_real_all_layer_updates_do_not_require_code_changes_in_every_layer(self):
        row = self.row()
        full_selection.verify_master_updates(row, self.initial, self.parameters, self.architecture, self.quantized)
        self.assertEqual(row['quantized_changes_vs_initialization']['w2']['changed_codes'], 0)
        self.assertEqual(row['quantized_changes_vs_initialization']['w3']['changed_codes'], 0)

    def test_unchanged_master_layer_and_reused_payload_are_rejected(self):
        broken = {**self.parameters, 'w3': self.initial['w3']}
        row = self.row()
        row['master_updates'] = trainer._parameter_update_evidence(self.initial, broken)
        with self.assertRaisesRegex(ValueError, 'all-layer master'):
            full_selection.verify_master_updates(row, self.initial, broken, self.architecture, self.quantized)
        unchanged = trainer.quantize_fixed(self.initial, self.architecture, self.quantized.scales)
        row = self.row()
        row['quantized_changes_vs_initialization'] = trainer._quantized_update_evidence(unchanged, unchanged)
        with self.assertRaisesRegex(ValueError, 'initialization was reused'):
            full_selection.verify_master_updates(row, self.initial, self.parameters, self.architecture, unchanged)

    def test_exact_runtime_export_is_reproduced_and_hash_valid_source_substitution_fails(self):
        runtime = trainer.write_runtime(self.directory, self.architecture, self.quantized,
                                        arm=trainer.ARMS['search-target'], seed=20260907,
                                        float_epoch=1, qat_epoch=1, source_bundle_body_sha256='a' * 64)
        source = self.directory / 'candidate.cpp'
        source.write_bytes(full_selection.selection._runtime_source(runtime))
        row = {'source': campaign.record(source), 'runtime': campaign.record(runtime),
               'source_reserve': 95000 - source.stat().st_size}
        full_selection.verify_source_export(row)
        source.write_bytes(source.read_bytes() + b'\n// altered source\n')
        row['source'] = campaign.record(source)
        row['source_reserve'] = 95000 - source.stat().st_size
        with self.assertRaisesRegex(ValueError, 'exact runtime export'):
            full_selection.verify_source_export(row)

    def test_other_phase_input_audit_is_rejected_even_with_valid_hash(self):
        context = self.directory / 'full'
        audit = self.directory / 'pilot/training-input-audit.json'
        campaign.seal(audit, {'schema': 'valid-but-other-phase'})
        with self.assertRaisesRegex(ValueError, 'another phase'):
            full_selection.validate_inputs(context, 'full', {}, {'input_audit': campaign.record(audit)}, None)


if __name__ == '__main__':
    unittest.main()
