import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from tools import compact_value_bfm_release_v2 as release
from submissions.codingame.bots.compact_value_bfm import feature_parity

campaign = release.campaign


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.context = self.root / 'context'
        self.phase = 'attempt-001-full'
        self.output = release.directory(self.context, self.phase)
        campaign.seal(self.root / 'campaign.json', {'fixture': 'parent'})
        campaign.seal(self.context / 'campaign.json', {'parent_campaign': campaign.record(self.root / 'campaign.json')})
        source = self.context / self.phase / 'selected.cpp'
        campaign.once(source, b'int main(){bool first_decision=true;const int budget_ms=first_decision?800:155;return budget_ms;}\n')
        runtime_path = source.with_suffix('.runtime.json')
        runtime = campaign.seal(runtime_path, {'quantization': {'payload_sha256': 'b' * 64}})
        selection_path = self.context / self.phase / 'search/search-selection.json'
        campaign.seal(selection_path, {'fixture': 'selection'})
        self.selected = {'source': campaign.record(source), 'runtime': campaign.record(runtime_path),
            'runtime_body_sha256': runtime['body_sha256'], 'payload_sha256': 'b' * 64,
            'candidate_search_profile': 'standard-v1', 'source_selection': campaign.record(selection_path)}
        self.result = {'selected': self.selected, 'passed': True, 'games': 1000, 'pairs': 500,
            'candidate_wins': 550, 'candidate_wins_by_color': [275, 275], 'failures': 0,
            'paired_lower_95': .51, 'protected_gates_passed': False, 'campaign_success': False,
            'policy': release.development.POLICY,
            'status': 'development-passed-awaiting-exact-source-ci-and-protected-gates'}
        campaign.seal(self.context / self.phase / 'development/assessment.json', self.result)
        self.identity = {'selected_source': self.selected['source'], 'runtime': self.selected['runtime'],
            'runtime_body_sha256': self.selected['runtime_body_sha256'], 'payload_sha256': self.selected['payload_sha256'],
            'source_selection': self.selected['source_selection']}
        for patcher in (mock.patch.object(release.development, 'completed_development', return_value=self.result),
                        mock.patch.object(release.ci, '_selection_identity', return_value=self.identity)):
            patcher.start(); self.addCleanup(patcher.stop)

    def ready(self):
        return release.prerequisites(self.root, self.context, self.phase)

    def test_completed_development_is_reopened_and_frozen_source_is_bound(self):
        ready = self.ready()
        self.assertEqual(ready['selected'], self.selected)
        release.development.completed_development.assert_called_once_with(self.context, self.phase)
        self.assertEqual(ready['source_selection'], self.selected['source_selection'])
        self.assertNotIn('qualification_passed', ready)

    def test_no_incomplete_or_lower_development_threshold_can_publish(self):
        for key, value in (('passed', False), ('games', 999), ('candidate_wins', 549),
                           ('candidate_wins_by_color', [264, 286]), ('failures', 1),
                           ('paired_lower_95', .5), ('paired_lower_95', float('nan')),
                           ('paired_lower_95', True), ('campaign_success', True)):
            with self.subTest(key=key, value=value):
                changed = {**self.result, key: value}
                with mock.patch.object(release.development, 'completed_development', return_value=changed), \
                        self.assertRaisesRegex(ValueError, 'completed 550'):
                    self.ready()
        with mock.patch.object(release.development, 'completed_development', side_effect=ValueError('incomplete raw development')), \
                self.assertRaisesRegex(ValueError, 'incomplete raw'):
            self.ready()

    def test_swapped_source_runtime_or_internal_clock_cannot_publish(self):
        self.identity['payload_sha256'] = 'c' * 64
        with self.assertRaisesRegex(ValueError, 'exact search-selected'):
            self.ready()
        self.identity['payload_sha256'] = self.selected['payload_sha256']
        source = Path(self.selected['source']['path'])
        source.write_bytes(source.read_bytes().replace(b'?800:155', b'?800:165'))
        self.selected['source'] = campaign.record(source)
        self.identity['selected_source'] = self.selected['source']
        with self.assertRaisesRegex(ValueError, 'internal clocks'):
            self.ready()

    def git(self, repository, *args):
        return subprocess.run(['git', '-C', str(repository), *args], capture_output=True, check=True).stdout

    def repository(self):
        repository = self.root / 'git'
        repository.mkdir()
        self.git(repository, 'init', '-b', 'feature/release-test')
        self.git(repository, 'remote', 'add', 'origin', release.ci.maintained.REPOSITORY_URL + '.git')
        target = repository / release.ci.RELEASE_SOURCE
        campaign.once(target, b'previous candidate\n')
        maintained = repository / release.ci.SUPPORTED_SOURCE
        campaign.once(maintained, b'frozen maintained source\n')
        self.git(repository, 'add', '.')
        self.git(repository, '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
            '-c', 'commit.gpgsign=false', 'commit', '-m', 'Freeze fixture')
        return repository, target, maintained

    def test_stage_preserves_previous_source_and_only_stages_candidate(self):
        repository, target, maintained = self.repository()
        result = release.stage(self.root, self.context, self.phase, repository=repository)
        self.assertEqual(target.read_bytes(), Path(self.selected['source']['path']).read_bytes())
        self.assertEqual(maintained.read_bytes(), b'frozen maintained source\n')
        self.assertEqual(self.git(repository, 'diff', '--cached', '--name-only').decode().strip(), release.ci.RELEASE_SOURCE)
        claim = campaign.read(campaign.verify(result['claim']))
        self.assertEqual(campaign.verify(claim['previous']).read_bytes(), b'previous candidate\n')
        self.assertEqual(release.stage(self.root, self.context, self.phase, repository=repository), result)
        self.assertFalse(result['campaign_success'])

    def test_stage_rejects_unrelated_target_changes_and_requires_clean_initial_index(self):
        repository, target, _ = self.repository()
        target.write_bytes(b'user edit\n')
        with self.assertRaisesRegex(ValueError, 'clean Git'):
            release.stage(self.root, self.context, self.phase, repository=repository)
        target.write_bytes(b'previous candidate\n')
        release.stage(self.root, self.context, self.phase, repository=repository)
        target.write_bytes(b'another user edit\n')
        with self.assertRaisesRegex(ValueError, 'unrelated change'):
            release.stage(self.root, self.context, self.phase, repository=repository)

    def test_publication_uses_read_only_exact_commit_ci_bridge(self):
        repository, _, _ = self.repository()
        with mock.patch.object(release.ci, 'freeze_publication', return_value={'fixture': 'publication'}) as freeze:
            release.publication(self.root, self.context, self.phase, repository=repository)
        self.assertEqual(freeze.call_args.kwargs['source'], repository / release.ci.RELEASE_SOURCE)
        self.assertEqual(freeze.call_args.kwargs['source_selection'], self.selected['source_selection']['path'])
        self.assertEqual(self.git(repository, 'status', '--porcelain'), b'')

    def test_ci_for_another_publication_or_developed_runtime_is_rejected(self):
        path = self.output / 'publication.json'
        campaign.seal(path, {'fixture': 'publication'})
        ci_path = self.output / 'ci.json'; campaign.seal(ci_path, {'fixture': 'ci'})
        published = {'source': {'path': release.ci.RELEASE_SOURCE},
            'selected_source': self.selected['source'], 'source_selection': self.selected['source_selection'],
            **{key: self.selected[key] for key in ('runtime', 'runtime_body_sha256', 'payload_sha256')}}
        evidence = {'publication': campaign.record(path)}
        with mock.patch.object(release.ci, 'validate_ci', return_value=evidence), \
                mock.patch.object(release.ci, 'validate_publication', return_value=published):
            ready, _ = release._ci_ready(self.root, self.context, self.phase, ci_path)
            self.assertEqual(ready['ci'], campaign.record(ci_path))
            published['payload_sha256'] = 'd' * 64
            with self.assertRaisesRegex(ValueError, 'differs from'):
                release._ci_ready(self.root, self.context, self.phase, ci_path)

    def plan_fixture(self):
        repository = self.output / 'repository'; repository.mkdir(parents=True, exist_ok=True)
        toolchain = {name: {'command_path': '/fixture/' + name} for name in ('gcc', 'clang', 'python', 'cmake', 'ctest', 'node')}
        snapshot = {'repository': str(repository)}
        plan = {**self.ready(), 'toolchain': toolchain,
            'commands': release.commands(self.output, snapshot, toolchain, self.selected)}
        return plan, repository

    def test_plan_requires_three_fresh_candidate_panels_full_parity_and_actual_clocks(self):
        plan, _ = self.plan_fixture()
        steps = {step['name']: step for step in plan['commands']}
        for panel in release.PANELS:
            command = steps[panel + '-compile']['argv']
            self.assertIn('--parallel', command)
            self.assertEqual(command[command.index('--parallel') + 1], '1')
            self.assertTrue(all(target in command for target in release.TARGETS))
            self.assertTrue(steps[panel + '-inference']['argv'][-1].endswith(release.PREFIX + 'inference_probe'))
            self.assertTrue(all(name in steps[panel + '-ctest']['markers'] for name in release.TESTS))
            self.assertIn('compact feature parity passed states=4096', steps[panel + '-ctest']['markers'])
        for color in (0, 1):
            command = steps['empty-' + str(color)]['argv']
            self.assertEqual(command[-4:], ['--first-timeout-ms', '1000', '--later-timeout-ms', '200'])
        self.assertEqual(release.POLICY['workers'], 1)
        self.assertFalse(release.POLICY['retry_interrupted_or_failed_command'])
        self.assertIn('delta-inference', steps)
        self.assertEqual(release.POLICY['actual_model_full_delta_states'], 4096)

    def test_source_snapshot_and_preparation_are_exact_and_resumable(self):
        repository, target, _ = self.repository()
        target.write_bytes(Path(self.selected['source']['path']).read_bytes())
        closure = ['tools/compact_value_bfm_release_v2.py',
            'submissions/codingame/bots/rank_4/submission.cpp',
            'submissions/codingame/bots/compact_value_bfm/timing_probe.cpp',
            'submissions/codingame/bots/compact_value_bfm/inference_probe.cpp']
        for relative in closure:
            campaign.once(repository / relative, (campaign.REPO / relative).read_bytes())
        self.git(repository, 'add', '.')
        self.git(repository, '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
            '-c', 'commit.gpgsign=false', 'commit', '-m', 'Freeze exact release fixture')
        published = {'repository_path': str(repository), 'commit': self.git(repository, 'rev-parse', 'HEAD').decode().strip()}
        ready = self.ready()
        for name in ('publication', 'ci'):
            campaign.seal(self.output / (name + '.json'), {'fixture': name})
            ready[name] = campaign.record(self.output / (name + '.json'))
        def tool(path, family=None):
            return {'command_path': str(path), 'family': family}
        with mock.patch.object(release, '_ci_ready', return_value=(ready, published)), \
                mock.patch.object(release, '_tool', side_effect=tool):
            plan = release.prepare(self.root, self.context, self.phase, ci_path=Path(ready['ci']['path']),
                python_path='/fixture/python', gcc_path='/fixture/gcc', clang_path='/fixture/clang',
                cmake_path='/fixture/cmake', ctest_path='/fixture/ctest', node_path='/fixture/node')
            self.assertFalse(plan['eligible_for_protected'])
            self.assertEqual(release.validate_plan(self.root, self.context, self.phase), plan)
            generated = (self.output / 'delta.cpp').read_text()
            self.assertIn('model.evaluate_delta(delta_previous, features)', generated)
            self.assertIn(str(self.output / 'repository' / release.ci.RELEASE_SOURCE), generated)
            # A newly injected source file must not enter subsequent CMake globs.
            extra = self.output / 'repository/extra.cpp'; extra.write_text('uncommitted input\n')
            with self.assertRaisesRegex(ValueError, 'extra uncommitted'):
                release.validate_plan(self.root, self.context, self.phase)
            extra.unlink()
            frozen = self.output / 'repository' / release.ci.RELEASE_SOURCE
            frozen.write_bytes(b'changed committed bytes\n')
            with self.assertRaisesRegex(ValueError, 'committed blob'):
                release.validate_plan(self.root, self.context, self.phase)

    def command_fixture(self, code='print("passed")'):
        plan, _ = self.plan_fixture()
        step = {'name': 'fixture', 'argv': [sys.executable, '-c', code], 'markers': ['passed']}
        path = self.output / 'fixture-plan.json'; campaign.seal(path, plan)
        return plan, path, step

    def test_completed_command_resumes_without_execution_and_binds_raw_streams(self):
        plan, path, step = self.command_fixture()
        receipt = release._run_step(plan, path, step)
        with mock.patch.object(release.maintained, 'run_command', side_effect=AssertionError('must not rerun')):
            self.assertEqual(release._run_step(plan, path, step), receipt)
        Path(receipt['stdout']['path']).write_bytes(b'changed\n')
        with self.assertRaisesRegex(ValueError, 'changed artifact'):
            release.validate_step(plan, path, step)

    def test_claimed_incomplete_or_failed_command_is_never_silently_retried(self):
        plan, path, step = self.command_fixture()
        (self.output / 'steps' / step['name']).mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, 'already claimed'):
            release._run_step(plan, path, step)
        step['name'] = 'failed'; step['argv'] = [sys.executable, '-c', 'raise SystemExit(2)']
        with self.assertRaises(release.maintained.PreflightError):
            release._run_step(plan, path, step)
        self.assertTrue((self.output / 'steps/failed/execution.json').is_file())
        with mock.patch.object(release.maintained, 'run_command', side_effect=AssertionError('must not rerun')), \
                self.assertRaises(release.maintained.PreflightError):
            release._run_step(plan, path, step)

    def test_timing_uses_actual_deadlines_without_historical_900_180_thresholds(self):
        plan, _ = self.plan_fixture()
        step = {'name': 'timing-0'}
        raw = b'player0_first budget_ms=800 elapsed_us=950000 nodes=1\nplayer0_later budget_ms=155 elapsed_us=190000 nodes=1\nplayer0_later_initial budget_ms=155 elapsed_us=190000 nodes=1\n'
        release._validate_output(plan, step, raw)
        with self.assertRaisesRegex(ValueError, 'actual external deadline'):
            release._validate_output(plan, step, raw.replace(b'190000', b'200000'))
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            release._validate_output(plan, step, raw.splitlines()[0])

    def test_runtime_identity_requires_both_embedded_and_decoded_payload_identity(self):
        plan, _ = self.plan_fixture()
        step = {'name': 'identity'}
        raw = ('\n'.join([self.selected['runtime_body_sha256'], 'b' * 64, 'b' * 64, '6301,12,8,1']) + '\n').encode()
        release._validate_output(plan, step, raw)
        for replacement in (raw.replace(b'b' * 64, b'a' * 64, 1), raw.replace(b'12,8', b'192,32')):
            with self.assertRaisesRegex(ValueError, 'identity differs'):
                release._validate_output(plan, step, replacement)

    def empty_fixture(self, color):
        ids = ['candidate', 'rank4'] if color == 0 else ['rank4', 'candidate']
        state = campaign.features.ReplayState(); rows = []; previous = '-'; counts = [0, 0]
        for turn in range(300):
            if state.winner is not None: break
            player = state.to_move
            action = feature_parity.choose_turn(state, 31, turn)
            rows.append({'turn': turn, 'player': player, 'botId': ids[player], 'opponentAction': previous,
                'action': action, 'accepted': True, 'failureClassification': None, 'durationMicros': 10,
                'deadlineMillis': 1000 if counts[player] == 0 else 200})
            previous = action; counts[player] += 1
        self.assertIn(state.winner, (0, 1))
        return {'schema': 'papersoccer.codingame-match.v1',
            'participants': {key: {'id': ids[player], 'player': player,
                'executable': release.PREFIX + 'submission' if ids[player] == 'candidate' else release.RANK4}
                for player, key in enumerate(('playerOne', 'playerTwo'))},
            'rules': {'width': 8, 'height': 10, 'goalRule': 'OwnGoalsAllowed', 'blockedRule': 'MoverLoses'},
            'timeouts': {'firstMillis': 1000, 'laterMillis': 200}, 'actions': rows,
            'outcome': {'forfeit': None, 'winnerId': ids[state.winner], 'loserId': ids[1 - state.winner]}}

    def test_empty_process_evidence_replays_both_colors_and_rejects_forfeits_or_fake_histories(self):
        for color in (0, 1):
            game = self.empty_fixture(color)
            self.assertEqual(release.validate_empty(json.dumps(game).encode(), color)['color'], color)
            variants = []
            changed = copy.deepcopy(game); changed['outcome']['forfeit'] = {'classification': 'timeout'}; variants.append(changed)
            changed = copy.deepcopy(game); changed['actions'][0]['opponentAction'] = '00'; variants.append(changed)
            changed = copy.deepcopy(game); changed['timeouts']['laterMillis'] = 220; variants.append(changed)
            changed = copy.deepcopy(game); changed['actions'] = changed['actions'][:-1]; variants.append(changed)
            changed = copy.deepcopy(game); changed['actions'][0]['durationMicros'] = 1000000; variants.append(changed)
            for changed in variants:
                with self.assertRaises(ValueError):
                    release.validate_empty(json.dumps(changed).encode(), color)

    def test_release_exclusions_include_both_initial_and_terminal_boundaries(self):
        executions = []
        games = []
        for color in (0, 1):
            game = self.empty_fixture(color); games.append(game)
            output = self.output / 'steps' / f'empty-{color}'
            campaign.once(output / 'stdout.log', json.dumps(game).encode())
            campaign.seal(output / 'execution.json', {'stdout': campaign.record(output / 'stdout.log')})
            executions.append(campaign.record(output / 'execution.json'))
        campaign.seal(self.output / 'preflight.json', {'executions': executions})
        frozen = {'preflight': campaign.record(self.output / 'preflight.json')}
        with mock.patch.object(release, 'validate', return_value=frozen) as validate:
            boundaries = list(release.preflight_boundaries(self.root, self.context, self.phase))
        validate.assert_called_once_with(self.root, self.context, self.phase)
        self.assertEqual(len(boundaries), sum(len(game['actions']) + 1 for game in games))
        initial = campaign.fingerprints(campaign.features.ReplayState())
        self.assertEqual(boundaries[0], initial)
        self.assertEqual(boundaries[len(games[0]['actions']) + 1], initial)
        final = campaign.features.ReplayState()
        for row in games[-1]['actions']:
            campaign.features.apply_complete_turn(final, final.to_move, row['action'])
        self.assertIn(final.winner, (0, 1))
        self.assertEqual(boundaries[-1], campaign.fingerprints(final))
        campaign.once(self.output / 'incomplete.json', b'{}')
        with mock.patch.object(release, 'validate', return_value={'preflight': campaign.record(self.output / 'incomplete.json')}):
            with self.assertRaises((KeyError, ValueError)):
                list(release.preflight_boundaries(self.root, self.context, self.phase))

    def test_cmake_evidence_requires_exact_harness_compiler_and_sanitizers(self):
        plan, repository = self.plan_fixture()
        candidate = repository / release.ci.RELEASE_SOURCE
        campaign.once(candidate, Path(self.selected['source']['path']).read_bytes())
        panel = 'clang-sanitized'; build = self.output / 'build' / panel
        build.mkdir(parents=True)
        compiler = plan['toolchain']['clang']['command_path']
        (build / 'CMakeCache.txt').write_text('Python3_EXECUTABLE:FILEPATH=/fixture/python\nPAPERSOCCER_ENABLE_SANITIZERS:BOOL=ON\n'
            'CMAKE_CXX_COMPILER:FILEPATH=' + compiler + '\nPAPERSOCCER_COMPACT_VALUE_BFM_RELEASE_SOURCE:FILEPATH=' + str(candidate) + '\n')
        entries = []
        for target in release.TARGETS:
            source = candidate
            if target != release.PREFIX + 'submission':
                source = build / 'compact-release-checks' / (target.removeprefix(release.PREFIX) + '.cpp')
                original = repository / release.maintained.BOT_RELATIVE / source.name
                campaign.once(original, b'#include "submission.cpp"\n')
                campaign.once(source, ('#include ' + json.dumps(str(candidate)) + '\n').encode())
            entries.append({'file': str(source), 'arguments': [compiler, '-fsanitize=address,undefined', '-fno-sanitize-recover=all', '-c', str(source)]})
            campaign.once(build / 'CMakeFiles' / (target + '.dir/link.txt'), (compiler + ' -fsanitize=address,undefined\n').encode())
        compile_path = build / 'compile_commands.json'; compile_path.write_text(json.dumps(entries))
        evidence = release.build_evidence(plan, {'panel': panel})
        self.assertEqual(len(evidence), 10)
        entries[0]['arguments'].remove('-fsanitize=address,undefined')
        compile_path.write_text(json.dumps(entries))
        with self.assertRaisesRegex(ValueError, 'sanitizer instrumentation'):
            release.build_evidence(plan, {'panel': panel})
        entries[0]['arguments'].insert(1, '-fsanitize=address,undefined')
        compile_path.write_text(json.dumps(entries))
        harness = build / 'compact-release-checks/submission_test.cpp'
        harness.write_text('#include "other.cpp"\n')
        with self.assertRaisesRegex(ValueError, 'exact selected bytes'):
            release.build_evidence(plan, {'panel': panel})

    def test_freeze_cannot_claim_readiness_without_every_completed_command(self):
        plan, path, step = self.command_fixture()
        release._run_step(plan, path, step)
        plan['commands'] = [step]
        for name in ('publication', 'ci'):
            campaign.seal(self.output / (name + '.json'), {'fixture': name})
            plan[name] = campaign.record(self.output / (name + '.json'))
        actual_plan_path = self.output / 'preflight-plan.json'
        campaign.seal(actual_plan_path, plan)
        # The prior command was bound to a different plan, so a copied green
        # command cannot satisfy this release plan.
        with mock.patch.object(release, 'validate_plan', return_value=plan), \
                self.assertRaisesRegex(ValueError, 'claim changed'):
            release._freeze_body(self.root, self.context, self.phase)


if __name__ == '__main__':
    unittest.main()
