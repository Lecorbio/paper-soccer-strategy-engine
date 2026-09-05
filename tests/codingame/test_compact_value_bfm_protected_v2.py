import copy
import json
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

from tools import compact_value_bfm_campaign_v2 as campaign
from tools import compact_value_bfm_protected_v2 as protected


def games_fixture(wins=(267, 260)):
    rows = [{'opening_id': str(index)} for index in range(500)]
    games = [{'pair_index': index, 'opening_id': str(index), 'candidate_player': color,
              'winner': color if index < wins[color] else 1 - color, 'failure': None}
             for index in range(500) for color in (0, 1)]
    return games, rows


def file(root, name, text='fixture'):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return campaign.record(path)


def shard_fixture(directory, *, complete=True):
    selected = {'source': file(directory, 'candidate.cpp'), 'runtime_body_sha256': 'runtime',
                'payload_sha256': 'payload', 'candidate_search_profile': 'standard-v1'}
    build = {'binary': file(directory, 'gate.bin'), 'candidate': selected['source'],
             'sources': {'candidate': selected['source'], 'rank4': file(directory, 'rank4.cpp')}}
    games, rows = games_fixture()
    bank = {'tsv': file(directory, 'bank.tsv'), 'rows': rows}
    execution = {'commands': [protected.shard_command(directory, index, bank, build) for index in range(100)]}
    campaign.seal(directory / 'execution-claim.json', execution)
    campaign.seal(directory / 'shards/000/claim.json', protected._shard_expected(directory, 0, execution))
    raw = {'config': {'mode': 'actual-clock', 'pair_offset': 0, 'pair_count': 5,
                      'candidate_clocks_ms': [800, 155], 'rank4_clocks_ms': [800, 165],
                      'max_turns': 320, 'minimum_candidate_wins': -1, 'minimum_wins_per_color': -1},
           'bindings': {'candidate_runtime_body_sha256': 'runtime', 'candidate_payload_sha256': 'payload',
                        'candidate_source_bytes': build['sources']['candidate']['bytes'],
                        'rank4_source_bytes': build['sources']['rank4']['bytes']},
           'result': {'passed': True}, 'games': games[:10]}
    if complete:
        output = directory / 'shards/000'
        campaign.seal(output / 'receipt.json', {'claim': campaign.record(output / 'claim.json'),
            'returncode': 0, 'timeout': False,
            'raw': file(output, 'result.json', json.dumps(raw)),
            'stdout': file(output, 'stdout.log', ''), 'stderr': file(output, 'stderr.log', ''),
            'trajectory_progress': None})
    return selected, build, bank, execution, raw


class ProtectedTests(unittest.TestCase):
    def test_fixed_527_and_260_thresholds_and_any_failure(self):
        games, rows = games_fixture()
        result = protected.assess_games(games, rows)
        self.assertTrue(result['passed'])
        self.assertEqual(result['candidate_wins'], 527)
        for wins in ((266, 260), (300, 259), (259, 300)):
            rejected, bank = games_fixture(wins)
            self.assertFalse(protected.assess_games(rejected, bank)['passed'])
        games[-1]['failure'] = 'rank4_timeout'
        self.assertFalse(protected.assess_games(games, rows)['passed'])
        with self.assertRaisesRegex(ValueError, 'both colors'):
            protected.assess_games(games[:-1], rows)
        games[-1] = games[-2]
        with self.assertRaisesRegex(ValueError, 'both colors'):
            protected.assess_games(games, rows)

    def test_production_policy_has_no_small_gate_or_worker_override(self):
        self.assertEqual(protected.POLICY['pairs_per_gate'], 500)
        self.assertEqual(protected.POLICY['games_per_gate'], 1000)
        self.assertEqual(protected.POLICY['workers'], 4)
        self.assertEqual(protected.POLICY['shards'] * protected.POLICY['pairs_per_shard'], 500)
        self.assertEqual(protected.POLICY['external_deadlines_ms'], [1000, 200])
        self.assertEqual(protected.POLICY['candidate_clocks_ms'], [800, 155])
        with self.assertRaisesRegex(ValueError, '500 canonical'):
            protected.assess_games([], [])

    def test_source_release_rejection_happens_before_any_protected_materialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with patch.object(protected, 'prerequisites', side_effect=ValueError('release is not ready')), \
                    patch.object(protected.os, 'urandom') as entropy, \
                    patch.object(protected, '_prepare_exclusions') as exclusions, \
                    patch.object(protected.subprocess, 'run') as execute, \
                    self.assertRaisesRegex(ValueError, 'not ready'):
                protected.run(root, root, 'phase')
            entropy.assert_not_called(); exclusions.assert_not_called(); execute.assert_not_called()
            self.assertFalse((root / 'phase/protected').exists())

    def test_prerequisites_require_true_release_eligibility(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); context = root / 'context'
            frozen = campaign.seal(context / 'phase/release/freeze.json', {'eligible_for_protected': False})
            from tools import compact_value_bfm_release_v2 as release
            with patch.object(release, 'validate', return_value=frozen), \
                    self.assertRaisesRegex(ValueError, 'source freeze'):
                protected.prerequisites(root, context, 'phase')

    def test_os_256_bit_seed_is_write_once_and_resume_never_draws_again(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            with patch.object(protected.os, 'urandom', return_value=b'a' * 32) as entropy:
                first = protected._seed_claim(directory, 'a', {'frozen': 'source'}, {'exclusions': 'hashes'}, {})
            entropy.assert_called_once_with(32)
            self.assertEqual(first['seed_hex'], (b'a' * 32).hex())
            with patch.object(protected.os, 'urandom') as entropy:
                self.assertEqual(first, protected._seed_claim(directory, 'a', {'frozen': 'source'}, {'exclusions': 'hashes'}, {}))
            entropy.assert_not_called()
            with self.assertRaisesRegex(ValueError, 'frozen source'):
                protected._seed_claim(directory, 'a', {'frozen': 'different'}, {'exclusions': 'hashes'}, {})

    def test_gate_b_seed_independent_and_bound_to_gate_a(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with patch.object(protected.os, 'urandom', return_value=b'a' * 32):
                protected._seed_claim(root / 'a', 'a', {}, {}, {})
            campaign.seal(root / 'a/bank.json', {'claim': campaign.record(root / 'a/seed-claim.json')})
            campaign.seal(root / 'a/assessment.json', {'bank': campaign.record(root / 'a/bank.json')})
            prior = campaign.record(root / 'a/assessment.json')
            with patch.object(protected.os, 'urandom', return_value=b'a' * 32), \
                    self.assertRaisesRegex(ValueError, 'independent protected seeds'):
                protected._seed_claim(root / 'b', 'b', {}, {}, {}, prior)
            self.assertFalse((root / 'b/seed-claim.json').exists())
            with patch.object(protected.os, 'urandom', return_value=b'b' * 32):
                second = protected._seed_claim(root / 'b', 'b', {}, {}, {}, prior)
            self.assertEqual(second['previous_gate'], prior)
            self.assertNotEqual(second['seed_hex'], (b'a' * 32).hex())

    def test_validation_cannot_create_a_missing_seed_or_bank(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with patch.object(protected.os, 'urandom') as entropy, \
                    self.assertRaisesRegex(ValueError, 'seed claim is missing'):
                protected._seed_claim(root, 'a', {}, {}, {}, create=False)
            entropy.assert_not_called()
            with patch.object(protected, 'prerequisites') as ready, \
                    self.assertRaisesRegex(ValueError, 'completed dual assessment'):
                protected.validate(root, root, 'phase')
            ready.assert_not_called()

    def test_hash_only_exclusions_use_prior_current_and_previous_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); state, feature = protected.DOMAINS
            bindings = []
            for name, domain, values in [('prior', state, ['a' * 64]), ('previous', state, ['b' * 64])]:
                path = root / (name + '.json')
                # The unrelated raw provenance is deliberately nonexistent. Hash
                # exclusion loading must never open old protected raw evidence.
                campaign.seal(path, {'domain': domain, 'fingerprints': values,
                    'execution': {'path': '/must-never-read-protected-raw.json'}})
                bindings.append(campaign.record(path))
            current = root / 'current.gz'
            protected.stream.write_gzip(current, [{'domain-ignored': 'fixture'}])
            target = {state: {'a' * 64, 'b' * 64}, feature: {'c' * 64}}
            with patch.object(protected.stream, 'read_gzip', return_value=iter([{feature: 'c' * 64}])):
                result = protected._collision_sets({'fragments': bindings[:1], 'current': campaign.record(current)},
                                                   target, bindings[1:])
            self.assertEqual(result, target)
            with self.assertRaisesRegex(ValueError, 'fingerprint changed'):
                protected._collision_sets({'fragments': [], 'current': campaign.record(current)}, target)

    def test_shards_cover_disjoint_five_pair_slices_and_actual_clocks(self):
        directory = Path('/protected/gate-a')
        bank = {'tsv': {'path': '/bank', 'sha256': 'bank'}}
        build = {'binary': {'path': '/binary'}, 'candidate': {'sha256': 'source'},
                 'sources': {'candidate': {'path': '/source'}, 'rank4': {'path': '/rank4'}}}
        pairs = []
        for ordinal in range(100):
            command = protected.shard_command(directory, ordinal, bank, build)
            offset = int(command[command.index('--pair-offset') + 1])
            self.assertEqual(command[command.index('--pair-count') + 1], '5')
            self.assertEqual(command[command.index('--mode') + 1], 'actual-clock')
            self.assertIn('--include-trajectories', command)
            pairs.extend(range(offset, offset + 5))
        self.assertEqual(pairs, list(range(500)))
        for bad in (-1, 100, True, 1.0):
            with self.assertRaises(ValueError): protected.shard_command(directory, bad, bank, build)

    def test_claim_without_completion_is_spent_and_never_executes(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            selected, build, bank, execution, _ = shard_fixture(directory, complete=False)
            with patch.object(protected.subprocess, 'run') as run, \
                    self.assertRaisesRegex(protected.SpentShardError, 'spent'):
                protected._execute_shard(directory, 0, execution, bank, build, selected)
            run.assert_not_called()

    def test_completed_shard_reuses_only_independently_validated_raw(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            selected, build, bank, execution, raw = shard_fixture(directory)
            with patch.object(protected.gate, 'validate_result', return_value=raw) as check, \
                    patch.object(protected.subprocess, 'run') as run:
                self.assertEqual(protected._execute_shard(directory, 0, execution, bank, build, selected), raw)
            run.assert_not_called()
            self.assertTrue(check.call_args.kwargs['require_trajectories'])
            self.assertEqual(check.call_args.kwargs['trajectory_bank'], Path(bank['tsv']['path']))
            self.assertEqual(check.call_args.kwargs['expected_candidate_sha256'], selected['source']['sha256'])
            self.assertEqual(check.call_args.kwargs['expected_candidate_search_profile'], 'standard-v1')

    def test_changed_payload_clock_shard_schedule_and_raw_hash_all_spend_claim(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            selected, build, bank, execution, raw = shard_fixture(directory)
            for group, key, value in [('bindings', 'candidate_payload_sha256', 'other'),
                                      ('bindings', 'candidate_runtime_body_sha256', 'other'),
                                      ('bindings', 'candidate_source_bytes', 999),
                                      ('config', 'mode', 'fixed-work'),
                                      ('config', 'pair_offset', 5),
                                      ('config', 'pair_count', 500),
                                      ('config', 'candidate_clocks_ms', [1, 1])]:
                changed = copy.deepcopy(raw); changed[group][key] = value
                with patch.object(protected.gate, 'validate_result', return_value=changed), \
                        self.assertRaisesRegex(protected.SpentShardError, 'configuration or compiled model'):
                    protected.validate_shard(directory, 0, execution, bank, build, selected)
            changed = copy.deepcopy(raw); changed['games'][0]['opening_id'] = 'other-root'
            with patch.object(protected.gate, 'validate_result', return_value=changed), \
                    self.assertRaisesRegex(protected.SpentShardError, 'schedule'):
                protected.validate_shard(directory, 0, execution, bank, build, selected)
            (directory / 'shards/000/result.json').write_text('changed')
            with patch.object(protected.gate, 'validate_result') as check, \
                    self.assertRaisesRegex(protected.SpentShardError, 'changed artifact'):
                protected.validate_shard(directory, 0, execution, bank, build, selected)
            check.assert_not_called()

    def test_orphaned_output_and_timeout_are_never_retried(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            selected, build, bank, execution, _ = shard_fixture(directory)
            output = directory / 'shards/001'; output.mkdir()
            (output / 'result.json').write_text('unclaimed')
            with patch.object(protected.subprocess, 'run') as run, \
                    self.assertRaisesRegex(protected.SpentShardError, 'orphaned'):
                protected._execute_shard(directory, 1, execution, bank, build, selected)
            run.assert_not_called()
            with patch.object(protected.subprocess, 'run', side_effect=subprocess.TimeoutExpired(['fixture'], 1)), \
                    self.assertRaisesRegex(protected.SpentShardError, 'spent'):
                protected._execute_shard(directory, 2, execution, bank, build, selected)
            receipt = campaign.read(directory / 'shards/002/receipt.json')
            self.assertTrue(receipt['timeout']); self.assertIsNone(receipt['raw'])
            with patch.object(protected.subprocess, 'run') as run, \
                    self.assertRaises(protected.SpentShardError):
                protected._execute_shard(directory, 2, execution, bank, build, selected)
            run.assert_not_called()

    def test_gate_b_never_prepares_until_all_gate_a_shards_validate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); selected = {'fixture': 'model'}
            ready = ({}, selected, {}, {})
            execution = {'commands': [['fixture']] * 100}
            first = root / 'phase/protected/gate-a'
            campaign.seal(first / 'shards/000/claim.json', {'fixture': 'spent'})
            with patch.object(protected, 'prerequisites', return_value=ready), \
                    patch.object(protected, '_prepare_exclusions', return_value={}), \
                    patch.object(protected, '_prepare_bank', return_value={}) as bank, \
                    patch.object(protected, '_build', return_value={}), \
                    patch.object(protected, '_execution_claim', return_value=execution), \
                    patch.object(protected, 'validate_shard', side_effect=protected.SpentShardError('spent')), \
                    patch.object(protected, '_execute_shard') as run, \
                    self.assertRaises(protected.SpentShardError):
                protected._process(root, root, 'phase', execute=True)
            self.assertEqual(bank.call_count, 1)
            self.assertEqual(bank.call_args.args[2], 'a')
            run.assert_not_called()

    def test_gate_b_uses_complete_a_fingerprints_regardless_of_a_score(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            ready = ({}, {'fixture': 'frozen-source'}, {'fixture': 'freeze'}, {})
            finished = []
            def finalize(context, phase, gate_id, *args):
                finished.append(gate_id)
                result = {'gate': gate_id, 'passed': gate_id == 'b', 'protected_exclusions': []}
                return campaign.seal(protected._directory(context, phase, gate_id) / 'assessment.json', result)
            def bank(context, phase, gate_id, ready, exclusions, previous, **kwargs):
                if gate_id == 'b':
                    self.assertEqual(finished, ['a'])
                    self.assertFalse(previous['passed'])
                    self.assertEqual(previous['gate'], 'a')
                return {}
            with patch.object(protected, 'prerequisites', return_value=ready), \
                    patch.object(protected, '_prepare_exclusions', return_value={}), \
                    patch.object(protected, '_prepare_bank', side_effect=bank), \
                    patch.object(protected, '_build', return_value={}), \
                    patch.object(protected, '_execution_claim', return_value={}), \
                    patch.object(protected, '_execute_shard') as execute, \
                    patch.object(protected, '_finalize_gate', side_effect=finalize), \
                    patch.object(protected.os, 'getpriority', return_value=0):
                result = protected._process(root, root, 'phase', execute=True)
            self.assertEqual(finished, ['a', 'b'])
            self.assertEqual(execute.call_count, 200)
            self.assertFalse(result['passed']); self.assertFalse(result['campaign_success'])
            self.assertEqual(result['status'], 'protected-rejected')

    def test_full_proposal_fingerprints_and_postroot_terminal_features_survive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); directory = root / 'phase/protected/gate-a'
            state, feature = protected.DOMAINS
            proposals = directory / 'proposals.json'
            campaign.seal(proposals, {'rows': [{'fingerprints': {state: 'a' * 64, feature: 'b' * 64}}]})
            bank = {'proposals': campaign.record(proposals), 'rows': games_fixture()[1]}
            campaign.seal(directory / 'bank.json', bank)
            campaign.seal(directory / 'execution-claim.json', {'fixture': 'execution'})
            for ordinal in range(100):
                campaign.seal(directory / 'shards' / f'{ordinal:03d}' / 'receipt.json', {'fixture': ordinal})
            games, _ = games_fixture()
            for row in games:
                row.update(transcript='synthetic-complete', root_transcript='synthetic-root')
            checked = [{'config': {'pair_offset': index * 5}, 'games': games[index * 10:index * 10 + 10]}
                       for index in range(100)]
            with patch.object(protected, 'validate_shard', side_effect=checked), \
                    patch.object(protected.development, 'boundaries', return_value=[
                        {state: 'c' * 64, feature: 'd' * 64}, {feature: 'e' * 64}]):
                result = protected._finalize_gate(root, 'phase', 'a', ({}, {}, {}, {}), bank, {}, {})
            exclusions = {campaign.read(campaign.verify(binding))['domain']:
                          campaign.read(campaign.verify(binding)) for binding in result['protected_exclusions']}
            self.assertEqual(exclusions[state]['fingerprints'], ['a' * 64, 'c' * 64])
            self.assertEqual(exclusions[feature]['fingerprints'], ['b' * 64, 'd' * 64, 'e' * 64])
            for exclusion in exclusions.values():
                self.assertTrue(exclusion['includes_all_proposals'])
                self.assertTrue(exclusion['includes_terminal_features'])
                self.assertFalse(exclusion['contains_metrics'])
                self.assertFalse(exclusion['contains_transcripts'])


if __name__ == '__main__':
    unittest.main()
