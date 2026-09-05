import copy
import json
from pathlib import Path
import random
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from tools import compact_value_bfm_campaign_v2 as campaign
from tools import compact_value_bfm_opponent_suite_v2 as suite


def root_row(seed=42, edges=8):
    state, transcript = campaign.fresh_root(edges, random.Random(seed))
    fps = campaign.fingerprints(state)
    return {'root_id': fps[campaign.legacy.STATE_FINGERPRINT_DOMAIN], 'transcript': transcript,
        'edges': edges, 'fingerprints': fps}


def completed_games(root):
    state = suite.replay(root['transcript'])
    rng = random.Random(89); turns = []
    while state.winner is None:
        player = state.to_move; action = ''
        while state.winner is None and state.to_move == player:
            direction = rng.choice(campaign.openings.legal_directions(state))
            action += str(direction); campaign.features.apply_primitive(state, direction)
        turns.append(action)
    return [{'schema': 'papersoccer.compact-state-evaluation.v2', 'root_id': root['root_id'],
        'candidate_player': color, 'winner': state.winner, 'failure': '', 'turns': len(turns),
        'root_transcript': root['transcript'], 'root_edges': root['edges'],
        'first_budget_ms': 800, 'later_budget_ms': 155,
        'opponent_first_budget_ms': 800, 'opponent_later_budget_ms': 165,
        'trajectory': '/'.join([root['transcript']] + turns),
        'candidate_latency_ms': [1.0, 2.0], 'candidate_max_ms': 2.0} for color in (0, 1)]


class OpponentSuiteTests(unittest.TestCase):
    def test_bank_checks_canonical_states_not_arbitrary_ids(self):
        rows = {name: [root_row(1000 * i + depth, depth) for depth in suite.DEPTHS]
            for i, name in enumerate(campaign.OPPONENTS)}
        suite.validate_bank_rows(rows, 4)
        duplicate = copy.deepcopy(rows)
        duplicate[campaign.OPPONENTS[1]][0] = copy.deepcopy(rows[campaign.OPPONENTS[0]][0])
        duplicate[campaign.OPPONENTS[1]][0]['root_id'] = 'different-name-same-state'
        with self.assertRaisesRegex(ValueError, 'canonical root reused'):
            suite.validate_bank_rows(duplicate, 4)
        rows[campaign.OPPONENTS[0]][0]['edges'] = 12
        with self.assertRaisesRegex(ValueError, 'progress'):
            suite.validate_bank_rows(rows, 4)

    def test_raw_winner_root_and_both_actor_clocks_are_verified(self):
        root = root_row(); rows = completed_games(root)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'games.jsonl'
            def check(values):
                path.write_text(''.join(json.dumps(row) + '\n' for row in values))
                return suite.checked_games(path, root, 'rank_4')
            self.assertEqual(len(check(rows)), 2)
            for field, value, message in [('winner', 1 - rows[0]['winner'], 'winner'),
                    ('root_transcript', '0', 'root or actor clocks'),
                    ('opponent_later_budget_ms', 155, 'actor clocks'),
                    ('later_budget_ms', 1, 'actor clocks'),
                    ('turns', rows[0]['turns'] - 1, 'prefix')]:
                broken = copy.deepcopy(rows); broken[0][field] = value
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, message): check(broken)
            with self.assertRaisesRegex(ValueError, 'schedule'): check(rows[:1])

    def test_subprocess_watchdog_preserves_both_failed_arms(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary); row = root_row()
            binary = directory / 'binary'; binary.write_bytes(b'fixture')
            tsv = directory / 'root.tsv'; tsv.write_text('root_id\ttranscript\n')
            build = directory / 'build.json'
            campaign.seal(build, {'binary': campaign.record(binary)})
            builds = {arm + ':rank_4': campaign.record(build) for arm in ('candidate', 'control')}
            bank = {'tsvs': {'rank_4:' + row['root_id']: campaign.record(tsv)}}
            with patch.object(suite.subprocess, 'run', side_effect=subprocess.TimeoutExpired(['binary'], 180)):
                receipt = suite.execute_pair(directory, 'rank_4', row, 1, bank, builds)
            result = campaign.read(campaign.verify(receipt))
            self.assertEqual(result['arm_order'], ['control', 'candidate'])
            self.assertTrue(all(arm['timeout'] and arm['returncode'] is None for arm in result['arms'].values()))
            for arm in result['arms'].values():
                campaign.verify(arm['output']); campaign.verify(arm['stderr'])
            with self.assertRaises(FileExistsError):
                suite.execute_pair(directory, 'rank_4', row, 1, bank, builds)

    def test_manifest_cannot_graft_another_execution_claim(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'screen'; directory.mkdir()
            claim = directory / 'claim.json'
            campaign.seal(claim, {'bank': {'sha256': 'a'}, 'builds': {}, 'policy': suite.POLICY})
            manifest = directory / 'manifest.json'
            campaign.seal(manifest, {'claim': campaign.record(claim), 'bank': {'sha256': 'b'}, 'builds': {}})
            campaign.seal(directory / 'assessment.json', {'manifest': campaign.record(manifest)})
            with self.assertRaisesRegex(ValueError, 'differs from execution claim'):
                suite._completed_suite(directory)

    def test_compiler_command_keeps_replay_correction_for_four_sources(self):
        for name in campaign.OPPONENTS:
            command = suite.compile_command(Path('/frozen'), Path('/compiler'), name)
            self.assertEqual('-DCAMPAIGN_OPPONENT_REPLAY_CORRECTION' in command, name in campaign.OPPONENTS[:4])
            self.assertIn('-DCAMPAIGN_CANDIDATE_SOURCE="/frozen/candidate.cpp"', command)

    def test_build_cannot_substitute_compiler_input_or_output_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);manifest=root/'build.json'
            candidate=root/'elsewhere.cpp';candidate.write_text('fixture')
            binary=root/'other.bin';binary.write_bytes(b'fixture')
            campaign.seal(manifest,{'binary':campaign.record(binary),'sources':{'candidate':campaign.record(candidate)}})
            with self.assertRaisesRegex(ValueError,'compiler output'):
                suite.validate_build(campaign.record(manifest),{},campaign.record(candidate),'rank_4',{})
            other=root/'second';other.mkdir();(other/'adapter.bin').write_bytes(b'fixture')
            campaign.seal(other/'build.json',{'binary':campaign.record(other/'adapter.bin'),'sources':{'candidate':campaign.record(candidate)}})
            with self.assertRaisesRegex(ValueError,'compiler input path'):
                suite.validate_build(campaign.record(other/'build.json'),{},campaign.record(candidate),'rank_4',{})

    def test_resume_isolation_rejects_played_search_states(self):
        root=root_row();domain=campaign.legacy.STATE_FINGERPRINT_DOMAIN
        with tempfile.TemporaryDirectory() as temporary:
            context=Path(temporary).resolve()
            campaign.seal(context/'campaign.json',{'exclusions':[]})
            bank={'stage':'screen','rows':{'rank_4':[root]}}
            empty={key:set() for key in root['fingerprints']}
            with patch.object(suite,'_current_collisions',return_value=empty),patch.object(suite,'search_boundaries',return_value=iter([{domain:root['fingerprints'][domain]}])):
                with self.assertRaisesRegex(ValueError,'played evaluation states'):
                    suite.validate_bank_isolation(context,'phase',bank,{})


if __name__ == '__main__': unittest.main()
