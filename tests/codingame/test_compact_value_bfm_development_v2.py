import copy
import json
from pathlib import Path
import random
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from tools import compact_value_bfm_campaign_v2 as campaign
from tools import compact_value_bfm_development_v2 as development
from tools import compact_value_bfm_opponent_suite_v2 as suite


def root_row(seed=31):
    state, transcript = campaign.fresh_root(12, random.Random(seed))
    fps = campaign.fingerprints(state)
    return {'opening_id': fps[campaign.legacy.STATE_FINGERPRINT_DOMAIN],
            'transcript': transcript, 'plies': 12, 'fingerprints': fps}


def game_fixture(double_wins=275):
    rows = [{'opening_id': str(index)} for index in range(500)]
    games = [{'pair_index': index, 'opening_id': str(index), 'candidate_player': color,
              'winner': color if index < double_wins else 1 - color, 'failure': None}
             for index in range(500) for color in (0, 1)]
    return games, rows


def fixture_prerequisites(root):
    context, phase = root / 'context', 'full'
    parent_path = root / 'campaign.json'
    campaign.seal(parent_path, {'fixture': 'parent'})
    campaign.seal(context / 'campaign.json', {'parent_campaign': campaign.record(parent_path)})
    for name in ('positions', 'games'):
        campaign.seal(context / phase / (name + '.json'), {'fixture': name})
    source_path = context / phase / 'source.cpp'; source_path.write_text('full exported source')
    runtime_path = context / phase / 'runtime.json'
    runtime = campaign.seal(runtime_path, {'quantization': {'payload_sha256': 'b' * 64}})
    seed_path = context / phase / 'seed.json'; campaign.seal(seed_path, {'fixture': 'seed'})
    model = {'source': campaign.record(source_path), 'runtime': campaign.record(runtime_path),
             'runtime_body_sha256': runtime['body_sha256'], 'payload_sha256': 'b' * 64,
             'canonical_retention_passed': True, 'source_reserve': 94000, 'lambda': .1,
             'seed': 20260907, 'seed_reference': campaign.record(seed_path)}
    training_path = context / phase / 'training.json'
    campaign.seal(training_path, {'smoke': False, 'mandatory_training_verified': True,
        'results': [{'weight': .1, 'seed': 20260907, 'source': model['source'], 'runtime': model['runtime']}]})
    full_path = context / phase / 'full-model-selection.json'
    campaign.seal(full_path, {'eligible_for_multi_opponent': True,
        'context': campaign.record(context / 'campaign.json'), 'selected': model,
        'training': campaign.record(training_path), 'seed_references': [campaign.record(seed_path)]})
    selected_path = context / phase / 'search/search-selection.json'
    source = {**model, 'candidate_search_profile': 'standard-v1', 'search_variant': 'baseline', 'compile_time_macros': []}
    source_selection = campaign.seal(selected_path, {'full_model_selection': campaign.record(full_path), 'selected': source})
    previous = []
    for stage in ('screen', 'confirmation'):
        directory = context / phase / 'multi-opponent' / stage
        campaign.seal(directory / 'assessment.json', {'passed': True})
        claim = directory / 'execution-claim.json'
        campaign.seal(claim, {'sources': {'candidate': model['source']}})
        previous.append(({'passed': True}, {'selection': campaign.record(selected_path)}, {'claim': campaign.record(claim)}))
    return context, phase, previous, source_selection


class DevelopmentTests(unittest.TestCase):
    def test_both_passing_suites_and_search_selection_bind_one_trained_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            context, phase, previous, source = fixture_prerequisites(Path(temporary))
            with patch.object(development.full_selection, 'verify_source_export'), \
                    patch.object(development.search, 'validate_selection', return_value=source), \
                    patch.object(suite, '_completed_suite', side_effect=previous):
                _, selected, inputs, _ = development.prerequisites(context, phase)
            self.assertEqual(selected['source'], source['selected']['source'])
            self.assertNotEqual(inputs['selection'], inputs['full_model_selection'])
            for changed, message in [('pass', 'passing suites'), ('source', 'selections differ')]:
                broken = copy.deepcopy(previous)
                if changed == 'pass': broken[1][0]['passed'] = False
                else: broken[1][1]['selection'] = inputs['full_model_selection']
                with patch.object(development.full_selection, 'verify_source_export'), \
                        patch.object(suite, '_completed_suite', side_effect=broken), \
                        self.assertRaisesRegex(ValueError, message):
                    development.prerequisites(context, phase)
            different = copy.deepcopy(source); different['selected']['source'] = {'sha256': 'wrong'}
            with patch.object(development.full_selection, 'verify_source_export'), \
                    patch.object(development.search, 'validate_selection', return_value=different), \
                    patch.object(suite, '_completed_suite', side_effect=previous), \
                    self.assertRaisesRegex(ValueError, 'validated search selection'):
                development.prerequisites(context, phase)

    def test_canonical_bank_rejects_duplicate_and_relabelled_states(self):
        row = root_row()
        development.validate_bank_rows([row], pairs=1)
        with self.assertRaisesRegex(ValueError, 'reused a canonical'):
            development.validate_bank_rows([row, row], pairs=2)
        broken = copy.deepcopy(row); broken['opening_id'] = 'new-name-same-state'
        with self.assertRaisesRegex(ValueError, 'canonical ID'):
            development.validate_bank_rows([broken], pairs=1)
        broken = copy.deepcopy(row); broken['plies'] = 13
        with self.assertRaisesRegex(ValueError, 'progress'):
            development.validate_bank_rows([broken], pairs=1)

    def test_paired_bootstrap_preserves_two_color_cluster_and_strict_threshold(self):
        games, rows = game_fixture()
        result = development.assess_games(games, rows, 'frozen-claim')
        expected = development.maintained.paired_bootstrap_lower_95(
            {'games': games}, seed_material='frozen-claim', samples=20000)
        self.assertEqual(result['paired_lower_95'], expected)
        self.assertEqual(result['candidate_wins_by_color'], [275, 275])
        self.assertTrue(result['passed'])
        # Exactly one win per pair has no root-pair sampling variance.
        for game in games: game['winner'] = 0
        tied = development.assess_games(games, rows, 'frozen-claim')
        self.assertEqual(tied['paired_lower_95'], .5)
        self.assertFalse(tied['passed'])
        games, rows = game_fixture()
        with patch.object(development.maintained, 'paired_bootstrap_lower_95', return_value=.5):
            self.assertFalse(development.assess_games(games, rows, 'claim')['passed'])

    def test_color_floor_and_any_failure_prevent_pass(self):
        games, rows = game_fixture()
        # Retain the total while making one color fall below its floor.
        for index in range(11): games[2 * index + 1]['winner'] = 0
        for index in range(275, 286): games[2 * index]['winner'] = 0
        result = development.assess_games(games, rows, 'claim')
        self.assertEqual(result['candidate_wins'], 550)
        self.assertEqual(result['candidate_wins_by_color'], [286, 264])
        self.assertFalse(result['passed'])
        games, rows = game_fixture(300); games[-1]['failure'] = 'rank4_timeout'
        self.assertFalse(development.assess_games(games, rows, 'claim')['passed'])
        with self.assertRaisesRegex(ValueError, 'schedule'):
            development.assess_games(games[:-1], rows, 'claim')
        duplicate = list(games); duplicate[-1] = duplicate[-2]
        with self.assertRaisesRegex(ValueError, 'schedule'):
            development.assess_games(duplicate, rows, 'claim')

    def test_trajectory_closure_has_root_terminal_and_no_pre_root_ancestors(self):
        root = root_row(); state = suite.replay(root['transcript'])
        rng = random.Random(888); actions = []
        while state.winner is None:
            mover = state.to_move; action = ''
            while state.winner is None and state.to_move == mover:
                direction = rng.choice(campaign.openings.legal_directions(state))
                action += str(direction); campaign.features.apply_primitive(state, direction)
            actions.append(action)
        transcript = '/'.join([root['transcript']] + actions)
        closure = list(development.boundaries(transcript, root['transcript']))
        self.assertEqual(len(closure), len(actions) + 1)
        self.assertEqual(closure[0], root['fingerprints'])
        self.assertEqual(closure[-1], campaign.fingerprints(state))
        self.assertNotIn(campaign.legacy.STATE_FINGERPRINT_DOMAIN, closure[-1])
        self.assertNotIn(campaign.fingerprints(campaign.features.ReplayState()), closure)
        self.assertEqual(list(development.boundaries(root['transcript'], root['transcript'])), [root['fingerprints']])
        with self.assertRaisesRegex(ValueError, 'prefix'):
            list(development.boundaries('0/0', root['transcript']))

    def test_current_and_prior_played_state_collisions_both_reject(self):
        target = {'state': {'a', 'b'}, 'feature': {'c'}}
        with patch.object(suite, '_current_collisions', return_value={'state': {'a'}, 'feature': set()}), \
                patch.object(development, 'suite_boundaries', return_value=iter([
                    {'state': 'b', 'feature': 'c'}, {'state': 'unrelated'}])):
            self.assertEqual(development.collisions(Path('/context'), 'phase', target, []),
                             {'state': {'a', 'b'}, 'feature': {'c'}})

    def test_spent_claim_stops_before_any_new_bank_or_subprocess(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); directory = root / 'phase/development'
            campaign.seal(directory / 'execution-claim.json', {'spent': True})
            with patch.object(development, 'prerequisites') as ready, \
                    patch.object(development.subprocess, 'run') as execute, \
                    self.assertRaisesRegex(ValueError, 'spent'):
                development.run(root, root, 'phase')
            ready.assert_not_called(); execute.assert_not_called()

    def test_shard_timeout_preserves_progress_and_cannot_restart(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with patch.object(development.subprocess, 'run', side_effect=subprocess.TimeoutExpired(['fixture'], 1)):
                binding = development.execute_shard(directory, 0, ['fixture'])
            receipt = campaign.read(campaign.verify(binding))
            self.assertTrue(receipt['timeout']); self.assertIsNone(receipt['raw'])
            campaign.verify(receipt['stdout']); campaign.verify(receipt['stderr'])
            with self.assertRaises(FileExistsError):
                development.execute_shard(directory, 0, ['fixture'])

    def test_compiler_and_native_commands_use_frozen_inputs_and_disjoint_pairs(self):
        directory = Path('/frozen/development')
        compile_command = development.compile_command(directory / 'build', Path('/compiler'))
        self.assertIn('-DCOMPACT_VALUE_BFM_CANDIDATE_SOURCE="/frozen/development/build/candidate.cpp"', compile_command)
        bank = {'tsv': {'path': '/bank.tsv', 'sha256': 'b'}}
        build = {'binary': {'path': '/binary'}, 'candidate': {'sha256': 's'},
                 'sources': {'candidate': {'path': '/candidate.cpp'}, 'rank4': {'path': '/rank4.cpp'}}}
        for ordinal in range(4):
            command = development.shard_command(directory, ordinal, bank, build)
            self.assertEqual(command[command.index('--pair-offset') + 1], str(ordinal * 125))
            self.assertEqual(command[command.index('--pair-count') + 1], '125')
            self.assertEqual(command[command.index('--mode') + 1], 'actual-clock')
            self.assertIn('--include-trajectories', command)

    def test_finalization_rejects_raw_clock_payload_and_execution_grafts(self):
        with tempfile.TemporaryDirectory() as temporary:
            context = Path(temporary).resolve(); phase = 'full'; directory = context / phase / 'development'
            selected = {'source': {'sha256': 'source'}, 'runtime_body_sha256': 'runtime',
                        'payload_sha256': 'payload', 'candidate_search_profile': 'standard-v1'}
            bank = {'tsv': {'path': '/bank.tsv', 'sha256': 'bank'}}
            build = {'binary': {'path': '/binary'}, 'candidate': {'sha256': 'source'},
                     'sources': {'candidate': {'path': '/candidate.cpp'}, 'rank4': {'path': '/rank4.cpp'}}}
            campaign.seal(directory / 'bank.json', bank)
            campaign.seal(directory / 'seed-claim.json', {'producers': {}})
            commands = [development.shard_command(directory, index, bank, build) for index in range(4)]
            claim = {'bank': campaign.record(directory / 'bank.json'), 'inputs': {}, 'selected': selected,
                     'policy': development.POLICY, 'producers': {}, 'nice': 0, 'threads': campaign.THREADS,
                     'build': {'path': str(directory / 'build/build.json')}, 'commands': commands}
            campaign.seal(directory / 'execution-claim.json', claim)
            output = directory / 'shards/0'; output.mkdir(parents=True)
            (output / 'result.json').write_text('{}')
            campaign.seal(output / 'execution.json', {'ordinal': 0, 'command': commands[0],
                'timeout': False, 'returncode': 0, 'raw': campaign.record(output / 'result.json')})
            raw = {'config': {'mode': 'actual-clock', 'pair_offset': 0, 'pair_count': 125,
                             'max_turns': 320, 'minimum_candidate_wins': -1, 'minimum_wins_per_color': -1},
                   'bindings': {'candidate_runtime_body_sha256': 'runtime', 'candidate_payload_sha256': 'payload'},
                   'result': {'passed': True}}
            original_verify = campaign.verify
            def verify(binding):
                if binding == bank['tsv']: return Path('/bank.tsv')
                return original_verify(binding)
            for group, key, value in [('config', 'mode', 'fixed-work'), ('config', 'pair_offset', 125),
                                      ('bindings', 'candidate_payload_sha256', 'other-model')]:
                broken = copy.deepcopy(raw); broken[group][key] = value
                with patch.object(development, 'prerequisites', return_value=({}, selected, {}, [])), \
                        patch.object(development, 'validate_bank', return_value=bank), \
                        patch.object(development, 'validate_build', return_value=build), \
                        patch.object(campaign, 'verify', side_effect=verify), \
                        patch.object(development.gate, 'validate_result', return_value=broken) as validator, \
                        self.assertRaisesRegex(ValueError, 'configuration, model payload'):
                    development.finalize(context, phase)
                self.assertTrue(validator.call_args.kwargs['require_trajectories'])
                self.assertEqual(validator.call_args.kwargs['expected_candidate_sha256'], 'source')
            # A receipt copied from a different binary/command cannot reach the raw validator.
            bad_command = copy.deepcopy(commands); bad_command[0][0] = '/other-binary'
            altered = {**claim, 'commands': bad_command}
            original_read = campaign.read
            def read(path):
                if Path(path) == directory / 'execution-claim.json': return altered
                return original_read(path)
            with patch.object(development, 'prerequisites', return_value=({}, selected, {}, [])), \
                    patch.object(development, 'validate_bank', return_value=bank), \
                    patch.object(development, 'validate_build', return_value=build), \
                    patch.object(campaign, 'read', side_effect=read), \
                    self.assertRaisesRegex(ValueError, 'execution commands changed'):
                development.finalize(context, phase)


if __name__ == '__main__':
    unittest.main()
