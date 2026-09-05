import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

from tools import compact_value_bfm_search_v2 as search

campaign = search.campaign


def probe(plan, mode='fixed', milliseconds=10, nodes=100, trace='exact'):
    return {'schema': 'papersoccer.compact-engine-version-probe.v2', 'mode': mode,
            'payload_sha256': plan['model']['payload_sha256'], 'all_actions_legal': True,
            'all_root_actions_legal': True, 'actual_model_full_delta_bit_exact': True,
            'all_root_actions_full_delta_bit_exact': True,
            'rows': [{'id': row['root_id'], 'action': '0', 'milliseconds': milliseconds,
                      'nodes': nodes, 'expansions': 10, 'generated_successors': 20,
                      'evaluated_successors': 20, 'fixed_trace': trace} for row in plan['roots']]}


class ProfilingCorpusTests(unittest.TestCase):
    def test_only_canonical_unique_full_train_states_enter_frozen_depth_schedule(self):
        rows = []
        rng = random.Random(20260909)
        for depth in search.POLICY['profiling_depths']:
            for number in range(search.POLICY['profiling_roots_per_depth'] + 1):
                state, transcript = campaign.fresh_root(depth, rng)
                rows.append({'position_id': f'{depth}-{number}', 'split': 'train', 'drawn_edges': depth,
                    'prefix': transcript, 'canonical': campaign.fingerprints(state)[campaign.legacy.STATE_FINGERPRINT_DOMAIN]})
        rows.append({**rows[0], 'position_id': 'duplicate'})
        rows.append({**rows[0], 'position_id': 'heldout', 'split': 'validation'})
        actual = search.profiling_rows({'rows': list(reversed(rows))})
        self.assertEqual(len(actual), 32)
        self.assertEqual(len({row['canonical'] for row in actual}), 32)
        self.assertNotIn('heldout', {row['position_id'] for row in actual})
        self.assertEqual(actual, search.profiling_rows({'rows': rows}))
        bad = [row for row in rows if row['split'] != 'validation']
        for row in bad:
            row['canonical'] = '0000000000000000000000000000000000000000000000000000000000000000'
        with self.assertRaisesRegex(ValueError, 'canonical identity'):
            search.profiling_rows({'rows': bad})


class MeasurementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name).resolve()
        self.plan = {'profile': 'standard-v1', 'roots': [{'root_id': 'root-a'}, {'root_id': 'root-b'}],
                     'model': {'payload_sha256': 'a' * 64}, 'compiler': {'path': '/frozen/compiler'},
                     'probe_source': {'path': '/frozen/probe.cpp'}, 'roots_tsv': {'path': '/frozen/roots.tsv'},
                     'variants': {name: {'source': {'path': '/frozen/' + name + '.cpp'}}
                                  for name in search.maintained.SEARCH_VARIANT_ORDER}}
        campaign.seal(self.output / 'plan.json', self.plan)

    def make_measurement(self, *, mutation=None, omit=False):
        builds = {}
        for name in self.plan['variants']:
            binary = self.output / 'builds' / name / 'probe.bin'
            campaign.once(binary, name.encode())
            builds[name] = {'source': self.plan['variants'][name]['source'], 'binary': campaign.record(binary),
                            'command': search.compile_command(self.plan, name, binary)}
        schedule = search.measurement_schedule(self.plan)
        claim = self.output / 'measurement-claim.json'
        campaign.seal(claim, {'plan': campaign.record(self.output / 'plan.json'), 'policy': search.POLICY,
                             'environment': campaign.THREADS, 'workers': 1, 'process_nice': 0,
                             'builds': builds, 'schedule': [list(row) for row in schedule]})
        runs = []
        for repeat, mode, name in schedule:
            raw = probe(self.plan, mode)
            if mutation:
                mutation(raw, repeat, mode, name)
            path = self.output / f'{repeat}-{mode}-{name}.json'
            campaign.once(path, campaign.raw(raw))
            runs.append({'repeat': repeat, 'mode': mode, 'variant': name, 'output': campaign.record(path),
                         'returncode': 0, 'command': [builds[name]['binary']['path'], self.plan['roots_tsv']['path'], mode]})
        if omit:
            runs.pop()
        campaign.seal(self.output / 'measurement.json', {'schema': campaign.ID + '.search-measurement.v2',
            'plan': campaign.record(self.output / 'plan.json'), 'claim': campaign.record(claim), 'runs': runs})

    def test_all_four_arms_and_all_repetitions_are_required(self):
        self.make_measurement(omit=True)
        with self.assertRaisesRegex(ValueError, 'schedule changed'):
            search.validate_measurement(self.plan, self.output)

    def test_complete_native_rows_recompute_speed_and_exact_trace_evidence(self):
        self.make_measurement()
        comparisons, rows = search.validate_measurement(self.plan, self.output)
        self.assertEqual(len(comparisons), 3)
        self.assertTrue(all(not row['passed'] for row in comparisons.values()))
        self.assertEqual(len(rows['baseline']['fixed']), 6)

    def test_source_valid_but_semantically_changed_variant_is_not_speed_evidence(self):
        def mutate(raw, repeat, mode, name):
            if name == 'combined' and mode == 'fixed':
                raw['rows'][0]['fixed_trace'] = 'different visits'
        self.make_measurement(mutation=mutate)
        with self.assertRaisesRegex(ValueError, 'fixed-work semantics'):
            search.validate_measurement(self.plan, self.output)

    def test_all_arms_sharing_nondeterministic_repeats_still_fail(self):
        def mutate(raw, repeat, mode, name):
            if repeat == 1 and mode == 'fixed':
                raw['rows'][0]['fixed_trace'] = 'shared repeated drift'
        self.make_measurement(mutation=mutate)
        with self.assertRaisesRegex(ValueError, 'nondeterministic'):
            search.validate_measurement(self.plan, self.output)

    def test_wrong_payload_root_order_or_omitted_root_invariant_fails(self):
        good = probe(self.plan)
        for field, value in (('payload_sha256', 'b' * 64), ('all_root_actions_full_delta_bit_exact', False)):
            bad = {**good, field: value}
            with self.assertRaisesRegex(ValueError, 'invariants differ'):
                search.validate_probe(bad, self.plan, 'fixed')
        good['rows'].reverse()
        with self.assertRaisesRegex(ValueError, 'roots or invariants'):
            search.validate_probe(good, self.plan, 'fixed')

    def test_nonfinite_negative_and_boolean_timing_are_rejected(self):
        for value in (float('nan'), float('inf'), -1, 0, True):
            bad = probe(self.plan, milliseconds=value)
            with self.assertRaisesRegex(ValueError, 'invalid native profiling row'):
                search.validate_probe(bad, self.plan, 'fixed')

    def test_latency_tail_can_veto_large_aggregate_throughput_gain(self):
        baseline = [{'milliseconds': 100, 'nodes': 1000}] * 20
        fast_with_tail = [{'milliseconds': 20, 'nodes': 1000}] * 18 + [{'milliseconds': 106, 'nodes': 1000}] * 2
        result = search.measured_comparison(baseline, fast_with_tail)
        self.assertGreater(result['throughput_gain'], .1)
        self.assertGreater(result['p95_regression'], .05)
        self.assertFalse(result['passed'])

    def test_actual_ten_percent_gain_is_required_and_five_percent_tail_is_allowed(self):
        baseline = [{'milliseconds': 100, 'nodes': 1000}] * 20
        slow = [{'milliseconds': 99, 'nodes': 1000}] * 20
        fast = [{'milliseconds': 80, 'nodes': 1000}] * 18 + [{'milliseconds': 105, 'nodes': 1000}] * 2
        self.assertFalse(search.measured_comparison(baseline, slow)['passed'])
        self.assertTrue(search.measured_comparison(baseline, fast)['passed'])


class SourceSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.context = self.root / 'context'
        self.phase = 'full'
        self.output = search.directory(self.context, self.phase)
        self.plan = {'profile': 'standard-v1', 'model': {'source': {'path': '/trained.cpp'},
                     'runtime': {'path': '/runtime.json'}, 'lambda': .1, 'seed': 20260908,
                     'runtime_body_sha256': 'r', 'payload_sha256': 'p', 'seed_reference': {'path': '/seed.json'}},
                     'full_model_selection': {'path': '/full-model-selection.json'},
                     'variants': {name: {'source': {'path': '/' + name + '.cpp', 'bytes': 92500},
                         'metadata': search.maintained._search_variant_metadata('standard-v1', name)}
                         for name in search.maintained.SEARCH_VARIANT_ORDER}}
        for path in ('plan.json', 'measurement.json', 'strength/index.json'):
            campaign.seal(self.output / path, {'fixture': path})

    def body(self, *, failed=(), speed=(), strength=(), timing=True, category_complete=True):
        names = list(self.plan['variants'])
        comparisons = {name: {'passed': name in speed} for name in names if name != 'baseline'}
        summaries = {name: {'candidate_wins': 100 + names.index(name)} for name in names}
        documents = {name: {'name': name} for name in names}
        retention = {'retained_variants': ['baseline', *strength]}
        with mock.patch.object(search, 'validate_plan', return_value=self.plan), \
             mock.patch.object(search, 'validate_measurement', return_value=(comparisons, {})), \
             mock.patch.object(search, 'validate_strength', return_value=(documents, {})), \
             mock.patch.object(search.maintained, '_result_summary', side_effect=lambda doc: summaries[doc['name']]), \
             mock.patch.object(search.maintained, '_standard_variant_cleanliness', return_value={}), \
             mock.patch.object(search.maintained, '_select_complete_search_variant', return_value=retention), \
             mock.patch.object(search.maintained, '_zero_failures', side_effect=lambda summary: summary not in [summaries[n] for n in failed]), \
             mock.patch.object(search.maintained, '_actual_clock_timing_clean', return_value=timing), \
             mock.patch.object(search, 'category_profile_status', return_value={
                 'complete': category_complete, 'receipt': None, 'reason': 'test category status'}):
            return search.selection_body(self.root, self.context, self.phase)

    def test_clocked_strength_and_speed_are_both_required(self):
        arm = 'no-feature-sort-only'
        for speed, strength in (((arm,), ()), ((), (arm,))):
            self.assertEqual(self.body(speed=speed, strength=strength)['selected']['search_variant'], 'baseline')
        selected = self.body(speed=(arm,), strength=(arm,))['selected']
        self.assertEqual(selected['search_variant'], arm)
        self.assertEqual(selected['runtime'], self.plan['model']['runtime'])
        self.assertEqual(selected['payload_sha256'], 'p')
        self.assertEqual(selected['source']['path'], '/' + arm + '.cpp')
        self.assertEqual(selected['candidate_search_profile'], 'standard-v1')

    def test_combined_requires_both_individual_speed_results(self):
        body = self.body(speed=('combined',), strength=('combined',))
        self.assertEqual(body['selected']['search_variant'], 'baseline')
        names = tuple(name for name in self.plan['variants'] if name != 'baseline')
        self.assertEqual(self.body(speed=names, strength=names)['selected']['search_variant'], 'combined')

    def test_failed_or_clock_dirty_baseline_cannot_be_a_fallback(self):
        self.assertIsNone(self.body(failed=('baseline',))['selected'])
        self.assertIsNone(self.body(timing=False)['selected'])

    def test_unmeasured_baseline_cannot_satisfy_source_selection(self):
        with mock.patch.object(search, 'validate_plan', return_value=self.plan), \
             mock.patch.object(search, 'validate_measurement', side_effect=ValueError('missing actual profiling')):
            with self.assertRaisesRegex(ValueError, 'missing actual profiling'):
                search.selection_body(self.root, self.context, self.phase)

    def test_missing_category_timers_preserve_ablation_but_block_advancement(self):
        body = self.body(category_complete=False)
        self.assertTrue(body['required_ablation_complete'])
        self.assertFalse(body['required_category_profile_complete'])
        self.assertFalse(body['required_profiling_complete'])
        self.assertFalse(body['eligible_for_multi_opponent'])
        self.assertIsNone(body['selected'])
        self.assertEqual(body['evaluated_candidate']['search_variant'], 'baseline')
        self.assertEqual(body['status'], 'awaiting-source-bound-category-profile')
        with mock.patch.object(search, 'selection_body', return_value=body):
            search.assess(self.root, self.context, self.phase)
        self.assertTrue((self.output / 'search-assessment-incomplete.json').is_file())
        self.assertFalse((self.output / 'search-selection.json').exists())

    def test_asserted_category_receipt_cannot_replace_missing_native_instrumentation(self):
        self.assertFalse(search.category_profile_status(self.plan, self.output)['complete'])
        campaign.seal(self.output / 'category-profile.json', {'passed': True, 'complete': True})
        with self.assertRaisesRegex(ValueError, 'source-bound native execution receipt'):
            search.category_profile_status(self.plan, self.output)

    def test_final_selection_preserves_existing_incomplete_receipt(self):
        incomplete = self.body(category_complete=False)
        with mock.patch.object(search, 'selection_body', return_value=incomplete):
            search.assess(self.root, self.context, self.phase)
        path = self.output / 'search-assessment-incomplete.json'
        original = path.read_bytes()
        complete = self.body(category_complete=True)
        with mock.patch.object(search, 'selection_body', return_value=complete):
            search.assess(self.root, self.context, self.phase)
        self.assertEqual(path.read_bytes(), original)
        self.assertTrue(campaign.read(self.output / 'search-selection.json')['required_profiling_complete'])

    def test_independent_profiles_have_one_treatment_macro_each_and_stay_default_off(self):
        self.assertEqual(search.POLICY['default_throughput_profile'], 'standard-v1')
        for profile, macro in search.maintained.SEARCH_THROUGHPUT_PROFILE_MACROS.items():
            roster = search.maintained.active_search_variants(profile)
            if macro is None:
                self.assertEqual(len(roster), 4)
            else:
                self.assertEqual(len(roster), 8)
                for base in search.maintained.SEARCH_VARIANT_ORDER:
                    self.assertEqual(roster[base], search.maintained.SEARCH_VARIANTS[base])
                    self.assertEqual(roster[base + '--' + profile], (*roster[base], macro))


class StrengthBindingTests(unittest.TestCase):
    def setUp(self):
        from tools import compact_value_bfm_search_strength_v2 as producer
        self.producer = producer
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name).resolve()
        for filename in ('plan.json', 'runtime.json', 'compiler', 'strength/bank.json', 'strength/bank.tsv'):
            campaign.once(self.output / filename, filename.encode())
        self.bank = {'tsv': campaign.record(self.output / 'strength/bank.tsv')}
        runtime = campaign.record(self.output / 'runtime.json')
        self.plan = {'profile': 'standard-v1', 'model': {'runtime': runtime, 'runtime_body_sha256': 'r' * 64,
            'payload_sha256': 'p' * 64, 'lambda': .1, 'seed': 20260908, 'source': {}},
            'compiler': campaign.record(self.output / 'compiler'),
            'gate_source': campaign.record(search.GATE_SOURCE),
            'rank4_source': campaign.record(search.GATE_SOURCE.parent.parent / 'rank_4/submission.cpp'),
            'full_model_selection': {}, 'full_model_selection_body_sha256': 'f' * 64, 'variants': {}}
        for name in search.maintained.SEARCH_VARIANT_ORDER:
            source = self.output / 'sources' / (name + '.cpp')
            campaign.once(source, name.encode())
            self.plan['variants'][name] = {'source': campaign.record(source),
                'metadata': search.maintained._search_variant_metadata('standard-v1', name)}

    def fixture(self, mutate=None):
        index = {'schema': campaign.ID + '.search-strength-index.v2',
            'plan': campaign.record(self.output / 'plan.json'), 'bank': self.bank['tsv'],
            'bank_receipt': campaign.record(self.output / 'strength/bank.json'), 'executions': {}}
        documents = {}
        for name, arm in self.plan['variants'].items():
            destination = self.output / 'strength' / name
            binary = destination / 'gate.bin'
            campaign.once(binary, name.encode())
            raw_path = destination / 'result.json'
            raw = {'config': {**search.maintained.GATE_CONFIGURATION,
                'candidate_search_profile': 'standard-v1', 'pair_count': 500,
                'minimum_candidate_wins': 550, 'minimum_wins_per_color': -1},
                'bindings': {'candidate_runtime_body_sha256': 'r' * 64, 'candidate_payload_sha256': 'p' * 64},
                'result': {'passed': True}}
            command = self.producer.command(self.plan, name, binary, self.bank, raw_path)
            claim = {'schema': campaign.ID + '.search-strength-claim.v2', 'plan': index['plan'], 'variant': name,
                'bank': index['bank'], 'bank_receipt': index['bank_receipt'], 'policy': self.producer.POLICY,
                'source': arm['source'], 'runtime': self.plan['model']['runtime'], 'compiler': self.plan['compiler'],
                'workers': 1, 'environment': campaign.THREADS, 'retry_allowed': False, 'process_nice': 0,
                'binary': campaign.record(binary), 'gate_source': self.plan['gate_source'],
                'rank4_source': self.plan['rank4_source'],
                'compile_command': self.producer.compile_command(self.plan, name, binary), 'command': command}
            execution = {'schema': campaign.ID + '.search-strength-execution.v2', 'returncode': 0}
            paths = {'claim': destination / 'claim.json', 'execution': destination / 'execution.json', 'raw': raw_path}
            if mutate and name == 'baseline':
                mutate(claim, execution, raw, paths)
            campaign.once(paths['raw'], campaign.raw(raw))
            campaign.seal(paths['claim'], claim)
            execution.update({'claim': campaign.record(paths['claim']), 'raw': campaign.record(paths['raw'])})
            campaign.seal(paths['execution'], execution)
            index['executions'][name] = campaign.record(paths['execution'])
            documents[str(paths['raw'])] = raw
        campaign.seal(self.output / 'strength/index.json', index)
        return documents

    def validate(self, documents):
        # The native result validator and bank isolator have their own tests;
        # this fixture tests the consumer's exact execution bindings around them.
        with mock.patch.object(self.producer, 'validate_bank', return_value=self.bank), \
             mock.patch.object(search.maintained.gate_support, 'validate_bank', return_value={}), \
             mock.patch.object(search.maintained.gate_support, 'validate_result',
                               side_effect=lambda path, **kwargs: documents[str(path)]):
            return search.validate_strength(self.plan, self.output)

    def test_producer_commands_and_complete_per_variant_bindings_are_accepted(self):
        documents, requests = self.validate(self.fixture())
        self.assertEqual(set(documents), set(search.maintained.SEARCH_VARIANT_ORDER))
        self.assertEqual(requests['baseline']['configuration']['candidate_shuffle_seed'], 1)
        self.assertEqual(requests['baseline']['configuration']['minimum_candidate_wins'], 550)

    def test_omitted_seed_or_changed_minimum_wins_are_rejected(self):
        def mutate(claim, execution, raw, paths):
            index = claim['command'].index('--candidate-seed')
            del claim['command'][index:index + 2]
        with self.assertRaisesRegex(ValueError, 'command differs'):
            self.validate(self.fixture(mutate))

    def test_pilot_win_threshold_cannot_substitute_for_full_search_configuration(self):
        def mutate(claim, execution, raw, paths):
            raw['config']['minimum_candidate_wins'] = 105
        with self.assertRaisesRegex(ValueError, 'model/runtime binding changed'):
            self.validate(self.fixture(mutate))

    def test_passed_native_result_requires_zero_exit_code(self):
        def mutate(claim, execution, raw, paths):
            execution['returncode'] = 2
        with self.assertRaisesRegex(ValueError, 'model/runtime binding changed'):
            self.validate(self.fixture(mutate))

    def test_failed_native_result_requires_exit_code_two(self):
        def mutate(claim, execution, raw, paths):
            raw['result']['passed'] = False
        with self.assertRaisesRegex(ValueError, 'model/runtime binding changed'):
            self.validate(self.fixture(mutate))

    def test_failed_native_result_with_exit_code_two_is_valid_failure_evidence(self):
        def mutate(claim, execution, raw, paths):
            raw['result']['passed'] = False
            execution['returncode'] = 2
        documents, _ = self.validate(self.fixture(mutate))
        self.assertFalse(documents['baseline']['result']['passed'])

    def test_other_variant_execution_path_cannot_supply_a_valid_receipt(self):
        def mutate(claim, execution, raw, paths):
            paths['execution'] = self.output / 'elsewhere/execution.json'
        with self.assertRaisesRegex(ValueError, 'worker or plan binding'):
            self.validate(self.fixture(mutate))

    def test_other_claim_path_cannot_supply_a_valid_claim(self):
        def mutate(claim, execution, raw, paths):
            paths['claim'] = self.output / 'elsewhere/claim.json'
        with self.assertRaisesRegex(ValueError, 'worker or plan binding'):
            self.validate(self.fixture(mutate))

    def test_other_raw_path_cannot_supply_a_valid_result(self):
        def mutate(claim, execution, raw, paths):
            paths['raw'] = self.output / 'elsewhere/result.json'
        with self.assertRaisesRegex(ValueError, 'another variant execution'):
            self.validate(self.fixture(mutate))

    def test_other_binary_path_cannot_supply_the_same_binary_bytes(self):
        def mutate(claim, execution, raw, paths):
            path = self.output / 'elsewhere/gate.bin'
            campaign.once(path, campaign.verify(claim['binary']).read_bytes())
            claim['binary'] = campaign.record(path)
        with self.assertRaisesRegex(ValueError, 'compile/source closure'):
            self.validate(self.fixture(mutate))

    def test_other_bank_receipt_is_rejected(self):
        def mutate(claim, execution, raw, paths):
            claim['bank_receipt'] = {'different': 'bank'}
        with self.assertRaisesRegex(ValueError, 'worker or plan binding'):
            self.validate(self.fixture(mutate))

    def test_changed_producer_policy_is_rejected(self):
        def mutate(claim, execution, raw, paths):
            claim['policy'] = {**claim['policy'], 'workers': 2}
        with self.assertRaisesRegex(ValueError, 'worker or plan binding'):
            self.validate(self.fixture(mutate))

    def test_changed_process_priority_is_rejected(self):
        def mutate(claim, execution, raw, paths):
            claim['process_nice'] = 10
        with self.assertRaisesRegex(ValueError, 'worker or plan binding'):
            self.validate(self.fixture(mutate))

    def test_boolean_false_cannot_claim_integer_nice_zero(self):
        def mutate(claim, execution, raw, paths):
            claim['process_nice'] = False
        with self.assertRaisesRegex(ValueError, 'worker or plan binding'):
            self.validate(self.fixture(mutate))


if __name__ == '__main__':
    unittest.main()
