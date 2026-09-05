from collections import defaultdict
from contextlib import ExitStack, nullcontext
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from tools import compact_value_bfm_terminal_outcome_v2 as terminal

campaign = terminal.campaign


def sealed(path, body):
    campaign.seal(path, body)
    return campaign.record(path)


def fixture(root, passed=False):
    context = root / 'phases/attempt-001-full'; phase = context.name; directory = context / phase
    parent = sealed(root / 'campaign.json', {'frozen': 'parent'})
    contract = campaign.seal(context / 'campaign.json', {'parent_campaign': parent, 'exclusions': []})
    source_path = directory / 'source.cpp'; source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text('synthetic-source')
    runtime = sealed(directory / 'runtime.json', {'quantization': {'payload_sha256': 'a' * 64}})
    selected = {'source': campaign.record(source_path), 'runtime': runtime,
                'runtime_body_sha256': campaign.read(campaign.verify(runtime))['body_sha256'],
                'payload_sha256': 'a' * 64, 'candidate_search_profile': 'standard-v1'}
    training = sealed(directory / 'training.json', {'smoke': False, 'mandatory_training_verified': True})
    model = sealed(directory / 'full-model-selection.json', {'selected': selected})
    for name in ('positions', 'games'): sealed(directory / (name + '.json'), {'fixture': name})
    pilot = {'outcome': sealed(root / 'pilot-outcome.json', {'admitted': True}),
             'training': training, 'selection': model, 'screen': None}
    development = {'passed': True, 'selected': selected, 'games': 1000, 'candidate_wins': 570,
                   'failures': 0, 'candidate_wins_by_color': [285, 285], 'paired_lower_95': .53}
    dev = sealed(directory / 'development/assessment.json', development)
    previous = {'stage': 'full', 'terminal_outcome': True, 'attempt': 1, 'context': context, 'phase': phase,
        'contract': contract, 'training': training, 'selection': model, 'screen': None, 'pilot': pilot,
        'rejection_stage': 'post-development', 'stages': {'development': dev}, 'source_selection': None,
        'suites': [], 'development': development, 'seed_references': []}
    freeze = sealed(directory / 'release/freeze.json', {'selected': selected, 'eligible_for_protected': True})
    gates, exclusions = [], []
    for number, gate_id in enumerate(('a', 'b')):
        rows = []
        for index, domain in enumerate(terminal.DOMAINS):
            rows.append(sealed(directory / f'protected/gate-{gate_id}/exclusion-{index}.json', {
                'role': 'protected', 'domain': domain, 'fingerprints': [str(number + index + 1) * 64],
                'includes_all_proposals': True, 'includes_all_played_postroot_boundaries': True,
                'includes_terminal_features': True, 'contains_labels': False, 'contains_metrics': False,
                'contains_transcripts': False}))
        gates.append(sealed(directory / f'protected/gate-{gate_id}/assessment.json', {'protected_exclusions': rows}))
        exclusions.extend(rows)
    qualified = campaign.seal(directory / 'protected/assessment.json', {'selected': selected, 'freeze': freeze,
        'passed': passed, 'status': 'protected-passed-awaiting-source-bound-live' if passed else 'protected-rejected',
        'campaign_success': False, 'gates': gates, 'protected_exclusions': exclusions,
        'protected_metric_that_must_not_escape': 991234.5})
    return previous, selected, qualified


class TerminalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.previous, self.selected, self.qualified = fixture(self.root)
        self.context, self.phase = self.previous['context'], self.previous['phase']

    def validators(self, qualified=None):
        stack = ExitStack()
        stack.enter_context(patch.object(terminal, '_unprotected_context', return_value=(self.previous, self.selected)))
        stack.enter_context(patch.object(terminal.protected, 'validate', return_value=qualified or self.qualified))
        stack.enter_context(patch.object(terminal.release, 'validate', return_value=campaign.read(self.context / self.phase / 'release/freeze.json')))
        return stack

    def test_terminal_context_reopens_full_training_pilot_and_passing_unprotected_chain(self):
        directory = self.context / self.phase
        source = sealed(directory / 'search/search-selection.json', {'selected': self.selected})
        model = {'selected': self.selected, 'eligible_for_multi_opponent': True,
                 'training': self.previous['training'], 'seed_references': []}
        inputs = {'selection': source, 'full_model_selection': self.previous['selection'],
            'screen': sealed(directory / 'multi-opponent/screen/assessment.json', {'passed': True}),
            'confirmation': sealed(directory / 'multi-opponent/confirmation/assessment.json', {'passed': True})}
        with patch.object(terminal.full.full_selection.trainer, 'native_thread_execution_scope', return_value=nullcontext()), \
                patch.object(terminal.full, 'validate_full_selection', return_value=(self.previous['contract'], model)) as trained, \
                patch.object(terminal.full, 'validate_pilot', return_value=self.previous['pilot']) as pilot, \
                patch.object(terminal.development, 'prerequisites', return_value=(self.previous['contract'], self.selected, inputs, [])), \
                patch.object(terminal.development, 'completed_development', return_value=self.previous['development']) as dev:
            previous, selected = terminal._unprotected_context(self.root, 1)
        trained.assert_called_once_with(self.root, self.context, self.phase)
        pilot.assert_called_once_with(self.root, 1, self.previous['contract'])
        dev.assert_called_once_with(self.context, self.phase)
        self.assertEqual(selected, self.selected)
        self.assertEqual(set(previous['stages']), {'search', 'screen', 'confirmation', 'development'})
        self.assertEqual(previous['rejection_stage'], 'post-development')

    def test_complete_protected_failure_counts_once_and_returns_only_unprotected_metrics(self):
        from tools import compact_value_bfm_attribution_v2 as attribution
        with self.validators(), patch.object(terminal.live, 'assess') as live_assess:
            result = terminal.record_outcome(self.root, 1)
            previous = terminal.failed_terminal(self.root, 1)
        live_assess.assert_not_called()
        self.assertEqual(result['completed_attempt_count'], 1)
        self.assertTrue(result['pilot_and_full_are_one_attempt'])
        self.assertEqual(previous['rejection_stage'], 'post-development')
        self.assertFalse(result['qualification_metrics_used_for_attribution'])
        self.assertFalse(result['qualification_references_followed_for_attribution'])
        projected = attribution.downstream_evidence(previous)
        self.assertEqual(projected['development']['candidate_wins'], 570)
        for value in (result, previous, projected):
            encoded = json.dumps(value, default=str)
            self.assertNotIn('991234.5', encoded)
            self.assertNotIn('protected_metric_that_must_not_escape', encoded)
        self.assertNotIn('protected', previous['stages'])
        self.assertNotIn('live', previous['stages'])

    def test_partial_protected_failure_does_not_create_terminal_outcome(self):
        with patch.object(terminal, '_unprotected_context', return_value=(self.previous, self.selected)), \
                patch.object(terminal.protected, 'validate', side_effect=ValueError('spent shard')), \
                self.assertRaisesRegex(ValueError, 'spent shard'):
            terminal.record_outcome(self.root, 1)
        self.assertFalse((self.context / self.phase / 'attempt-outcome.json').exists())

    def test_mismatched_qualified_source_and_later_live_evidence_block_protected_closure(self):
        wrong = copy.deepcopy(self.qualified); wrong['selected']['source'] = {'sha256': 'wrong'}
        with self.validators(wrong), self.assertRaisesRegex(ValueError, 'exact developed source'):
            terminal.record_outcome(self.root, 1)
        terminal.live.directory(self.root, self.selected['source']['sha256']).mkdir(parents=True)
        with self.validators(), self.assertRaisesRegex(ValueError, 'cannot hide later'):
            terminal.record_outcome(self.root, 1)

    def test_completed_receipt_is_immutable_and_revalidated(self):
        with self.validators():
            first = terminal.record_outcome(self.root, 1)
            self.assertEqual(first, terminal.record_outcome(self.root, 1))
            before = (self.context / self.phase / 'attempt-outcome.json').read_bytes()
            with patch.object(terminal.protected, 'validate', side_effect=ValueError('changed binary')), \
                    self.assertRaisesRegex(ValueError, 'changed binary'):
                terminal.failed_terminal(self.root, 1)
            self.assertEqual(before, (self.context / self.phase / 'attempt-outcome.json').read_bytes())

    def live_fixture(self, **changes):
        output = terminal.live.directory(self.root, self.selected['source']['sha256'])
        authorization = sealed(output / 'authorization.json', {'fixture': 'authorization'})
        auth = {'context': str(self.context), 'phase': self.phase, 'qualified_source': self.selected['source'],
                'runtime': self.selected['runtime'], 'payload_sha256': self.selected['payload_sha256']}
        document = {'schema': campaign.ID + '.live-assessment.v2',
            'status': 'completed-live-attempt-below-objective', 'campaign_success': False,
            'exact_games': 90, 'calibration_complete': True, 'precise_score_required': False,
            'clean_window': True, 'training_eligible': False, 'identical_source_reupload_allowed': False,
            'authorization': authorization, **changes}
        sealed(output / 'assessment.json', document)
        return output, auth, campaign.read(output / 'assessment.json')

    def test_completed_calibrated_live_failure_requires_source_bound_validator(self):
        output, auth, assessment = self.live_fixture()
        with patch.object(terminal.live, 'assess', return_value=assessment) as assess, \
                patch.object(terminal.live, 'revalidate_authorization', return_value=auth):
            self.assertEqual(terminal._live_rejection(self.root, self.context, self.phase, self.selected),
                             campaign.record(output / 'assessment.json'))
        assess.assert_called_once_with(self.root, self.selected['source']['sha256'])
        auth['runtime'] = {'different': 'runtime'}
        with patch.object(terminal.live, 'assess', return_value=assessment), \
                patch.object(terminal.live, 'revalidate_authorization', return_value=auth), \
                self.assertRaisesRegex(ValueError, 'another full attempt or source'):
            terminal._live_rejection(self.root, self.context, self.phase, self.selected)

    def test_unclean_complete_live_window_is_definitive_even_when_score_precision_is_inconclusive(self):
        output, auth, assessment = self.live_fixture(clean_window=False, precise_score_required=True)
        with patch.object(terminal.live, 'assess', return_value=assessment), \
                patch.object(terminal.live, 'revalidate_authorization', return_value=auth):
            self.assertEqual(terminal._live_rejection(self.root, self.context, self.phase, self.selected),
                             campaign.record(output / 'assessment.json'))

    def test_incomplete_ambiguous_and_precision_inconclusive_live_are_not_failures(self):
        with patch.object(terminal.live, 'assess') as assess, self.assertRaisesRegex(ValueError, 'completed unambiguous'):
            terminal._live_rejection(self.root, self.context, self.phase, self.selected)
        assess.assert_not_called()
        output, auth, assessment = self.live_fixture()
        for key, value in [('precise_score_required', True), ('calibration_complete', False),
                           ('campaign_success', True), ('exact_games', 89),
                           ('status', 'score-precision-inconclusive')]:
            altered = {**assessment, key: value}
            with patch.object(terminal.live, 'assess', return_value=altered), \
                    self.assertRaisesRegex(ValueError, 'not a failed attempt'):
                terminal._live_rejection(self.root, self.context, self.phase, self.selected)
        sealed(output / 'assessment-precision-inconclusive.json', {'inconclusive': True})
        with patch.object(terminal.live, 'assess') as assess, self.assertRaisesRegex(ValueError, 'completed unambiguous'):
            terminal._live_rejection(self.root, self.context, self.phase, self.selected)
        assess.assert_not_called()

    def test_protected_carry_contains_both_gates_all_proposals_and_terminal_feature_domains(self):
        evidence = {'protected': campaign.record(self.context / self.phase / 'protected/assessment.json')}
        values = terminal._protected_fingerprints(evidence)
        self.assertEqual(values['protected', terminal.DOMAINS[0]], {'1' * 64, '2' * 64})
        self.assertEqual(values['protected', terminal.DOMAINS[1]], {'2' * 64, '3' * 64})
        path = Path(self.qualified['protected_exclusions'][0]['path']); path.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'changed artifact'):
            terminal._protected_fingerprints(evidence)

    def test_zero_turn_operational_ending_keeps_initial_state_and_feature_hashes(self):
        identity = types.SimpleNamespace(source_sha256='a' * 64, agent_id=1, submission_id=2)
        record = {'schema': terminal.live.collector.GENERIC_GAME_SCHEMA, 'status': 'accepted',
            'source_sha256': identity.source_sha256, 'focus': {'agent_id': 1, 'submission_id': 2},
            'replay': {'valid_transcript': '', 'valid_turns': [], 'rules_validation': {
                'valid_turns': [], 'valid_turn_count': 0, 'status': 'invalid'}},
            'operational': {'classification': 'operationally-terminated'}}
        self.assertEqual(terminal._live_record_fingerprints(record, identity),
                         {domain: {value} for domain, value in campaign.fingerprints(campaign.features.ReplayState()).items()})
        record['focus']['submission_id'] = 3
        with self.assertRaisesRegex(ValueError, 'source identity'):
            terminal._live_record_fingerprints(record, identity)

    def test_live_replay_uses_existing_validator_and_preserves_terminal_feature(self):
        feature, state = terminal.DOMAINS[1], terminal.DOMAINS[0]
        fake = {'replay': {'valid_transcript': 'synthetic'}}
        values = defaultdict(set, {state: {'a' * 64}, feature: {'b' * 64, 'c' * 64}})
        with patch.object(terminal.live.collector, '_canonical_live_boundaries', return_value=['a' * 64]) as canonical, \
                patch.object(terminal.attempts, 'trajectory_fingerprints', return_value=(values, object())) as replay:
            self.assertEqual(terminal._live_record_fingerprints(fake, 'identity'), values)
        canonical.assert_called_once_with(fake, identity='identity')
        replay.assert_called_once_with('synthetic', 0)
        with patch.object(terminal.live.collector, '_canonical_live_boundaries', return_value=['d' * 64]), \
                patch.object(terminal.attempts, 'trajectory_fingerprints', return_value=(values, object())), \
                self.assertRaisesRegex(ValueError, 'boundaries disagree'):
            terminal._live_record_fingerprints(fake, 'identity')

    def test_live_extraction_revalidates_exact90_source_session_and_manifest(self):
        output = terminal.live.directory(self.root, self.selected['source']['sha256'])
        window = output / 'window/live-window.reference.json'
        window_binding = sealed(window, {'fixture': 'window'})
        assessment = sealed(output / 'assessment.json', {'window': window_binding})
        evidence = {'selected_source': self.selected['source'], 'live': assessment}
        identity = types.SimpleNamespace(source_sha256=self.selected['source']['sha256'])
        records = [{'focus': {'session_id': 'session'}} for _ in range(90)]
        receipt = {'collector_manifest': {'path': str(output / 'manifest.json'), 'sha256': 'bound'},
                   'game_ids': list(range(1, 91))}
        def context():
            stack = ExitStack()
            stack.enter_context(patch.object(terminal.live, 'live_window', return_value=(
                {'exclusion_binding': {'path': '/excluded'}}, {'test_session_handle': 'session'}, window, receipt)))
            stack.enter_context(patch.object(terminal.live.collector, 'load_live_identity', return_value=(identity, {'registry': {'sha256': 'registry'}}, None)))
            stack.enter_context(patch.object(terminal.live.collector, 'verify_generic_result', return_value={'records': records}))
            return stack
        with context(), patch.object(terminal, '_live_record_fingerprints', return_value={terminal.DOMAINS[1]: {'a' * 64}}) as replay:
            values = terminal._live_fingerprints(self.root, evidence)
        self.assertEqual(replay.call_count, 90)
        self.assertEqual(values['live', terminal.DOMAINS[1]], {'a' * 64})
        records[0]['focus']['session_id'] = 'other-submission-session'
        with context(), patch.object(terminal, '_live_record_fingerprints') as replay, \
                self.assertRaisesRegex(ValueError, 'session or exact90'):
            terminal._live_fingerprints(self.root, evidence)
        replay.assert_not_called()

    def test_next_attempt_carries_all_roles_without_changing_prior_evidence(self):
        with self.validators():
            terminal.record_outcome(self.root, 1)
            previous = terminal.failed_terminal(self.root, 1)
            _, evidence = terminal._validated(self.root, 1)
        files = {path: path.read_bytes() for path in self.root.rglob('*') if path.is_file()}
        state, feature = terminal.DOMAINS
        corpus = defaultdict(set, {('prior-train', state): {'a' * 64},
                                   ('prior-validation', feature): {'b' * 64}})
        live = defaultdict(set, {('live', state): {'c' * 64}, ('live', feature): {'d' * 64}})
        release = defaultdict(set, {('mixed-development', state): {'e' * 64}})
        with patch.object(terminal, 'failed_terminal', return_value=previous), \
                patch.object(terminal, '_validated', return_value=(previous, evidence)), \
                patch.object(terminal.full, 'collect_fingerprints', return_value=corpus), \
                patch.object(terminal, '_live_fingerprints', return_value=live), \
                patch.object(terminal, '_release_fingerprints', return_value=release):
            bindings, index = terminal.carry_failed_terminal(self.root, previous, self.root / 'next')
        received = {(row['role'], row['domain']): set(row['fingerprints'])
                    for row in (campaign.read(campaign.verify(binding)) for binding in bindings)}
        for key, members in {**corpus, **live, **release}.items(): self.assertTrue(members <= received[key])
        self.assertEqual(received['protected', state], {'1' * 64, '2' * 64})
        self.assertEqual(received['protected', feature], {'2' * 64, '3' * 64})
        document = campaign.read(campaign.verify(index))
        self.assertEqual(document['completed_attempt_count'], 1)
        self.assertTrue(document['protected_never_exempt'])
        for path, raw in files.items(): self.assertEqual(raw, path.read_bytes())

    def test_dispatchers_route_only_the_terminal_schema_and_keep_attempt_three_guarded(self):
        with self.validators(): terminal.record_outcome(self.root, 1)
        sentinel = {'terminal_outcome': True, 'attempt': 1}
        with patch.object(terminal, 'failed_terminal', return_value=sentinel) as validate, \
                patch.object(terminal.full, 'failed_full') as old:
            self.assertEqual(terminal.attempts.failed_attempt(self.root, 1), sentinel)
        validate.assert_called_once_with(self.root, 1); old.assert_not_called()
        with patch.object(terminal, 'carry_failed_terminal', return_value='carry') as carry:
            self.assertEqual(terminal.attempts.carry_failed_attempt(self.root, sentinel, self.root / 'next'), 'carry')
        carry.assert_called_once()
        for attempt in (3, 4):
            with self.assertRaisesRegex(ValueError, 'intervention binding'):
                terminal.attempts.failed_attempt(self.root, attempt)


if __name__ == '__main__':
    unittest.main()
