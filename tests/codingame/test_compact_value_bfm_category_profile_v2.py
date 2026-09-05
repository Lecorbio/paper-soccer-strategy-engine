import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from tools import compact_value_bfm_category_profile_v2 as categories

campaign = categories.campaign
search = categories.search
instrumentation = categories.instrumentation


class CategoryEvidenceTests(unittest.TestCase):
    """Exercise real source transformation and validators with synthetic executions."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.context = self.root / 'context'
        self.phase = 'full'
        self.output = search.directory(self.context, self.phase)
        source = (campaign.REPO / 'submissions/codingame/bots/compact_value_bfm/submission.cpp').read_bytes()
        _, manifest = instrumentation.instrument_source(source, instrumentation.sha(source))
        self.plan = {'profile': 'standard-v1', 'roots': [{'root_id': 'a'}, {'root_id': 'b'}],
                     'model': {'payload_sha256': manifest['payload_sha256']}, 'variants': {}}
        for name, macros in search.maintained.active_search_variants('standard-v1').items():
            path = self.output / 'sources' / (name + '.cpp')
            campaign.once(path, search.maintained._variant_source(source, macros))
            self.plan['variants'][name] = {'source': campaign.record(path)}
        for filename, key in (('compiler', 'compiler'), ('roots.tsv', 'roots_tsv')):
            path = self.root / filename
            campaign.once(path, filename.encode())
            self.plan[key] = campaign.record(path)
        runtime = self.root / 'runtime.json'
        campaign.seal(runtime, {'payload_sha256': manifest['payload_sha256']})
        self.plan['model']['runtime'] = campaign.record(runtime)
        self.plan['probe_source'] = campaign.record(search.PROBE)
        campaign.seal(self.output / 'plan.json', self.plan)
        self.ordinary_measurements()

    def raw(self, mode='fixed'):
        return {'schema': 'papersoccer.compact-engine-version-probe.v2', 'mode': mode,
            'payload_sha256': self.plan['model']['payload_sha256'],
            'all_actions_legal': True, 'all_root_actions_legal': True,
            'actual_model_full_delta_bit_exact': True, 'all_root_actions_full_delta_bit_exact': True,
            'rows': [{'id': row['root_id'], 'action': '0', 'fixed_trace': 'identical',
                      'nodes': 100, 'expansions': 20, 'generated_successors': 30,
                      'evaluated_successors': 30, 'milliseconds': 10.0} for row in self.plan['roots']]}

    def ordinary_measurements(self):
        builds = {}
        for name in self.plan['variants']:
            binary = self.output / 'builds' / name / 'probe.bin'
            campaign.once(binary, name.encode())
            builds[name] = {'source': self.plan['variants'][name]['source'], 'binary': campaign.record(binary),
                            'command': search.compile_command(self.plan, name, binary)}
        claim = self.output / 'measurement-claim.json'
        campaign.seal(claim, {'plan': campaign.record(self.output / 'plan.json'), 'policy': search.POLICY,
            'environment': campaign.THREADS, 'workers': 1, 'process_nice': 0, 'builds': builds,
            'schedule': [list(row) for row in search.measurement_schedule(self.plan)]})
        runs = []
        for repeat, mode, name in search.measurement_schedule(self.plan):
            path = self.output / 'measurements' / f'{repeat}-{mode}-{name}.json'
            campaign.once(path, campaign.raw(self.raw(mode)))
            runs.append({'repeat': repeat, 'mode': mode, 'variant': name, 'output': campaign.record(path),
                         'returncode': 0, 'command': [builds[name]['binary']['path'], self.plan['roots_tsv']['path'], mode]})
        campaign.seal(self.output / 'measurement.json', {'schema': campaign.ID + '.search-measurement.v2',
            'plan': campaign.record(self.output / 'plan.json'), 'claim': campaign.record(claim), 'runs': runs})

    def fake_process(self, args, **kwargs):
        self.assertEqual(kwargs['env'] | campaign.THREADS, kwargs['env'])
        if '-o' in args:
            Path(args[-1]).write_bytes(campaign.raw(args))
            return subprocess.CompletedProcess(args, 0, b'', b'')
        self.assertEqual(args[-1], 'fixed')
        name = Path(args[0]).parent.name
        raw = self.raw()
        raw.update({'schema': instrumentation.PROBE_SCHEMA,
                    'source_sha256': self.plan['variants'][name]['source']['sha256'],
                    'instrumentation_version': instrumentation.VERSION,
                    'attribution_only': True, 'eligible_for_speed_gate': False})
        for row in raw['rows']:
            times = dict.fromkeys(instrumentation.CATEGORIES, 1_000_000)
            times['residual_search'] = 24_000_000
            row.update({'milliseconds': 30.0, 'category_exclusive_ns': times,
                        'category_calls': dict.fromkeys(instrumentation.CATEGORIES, 1),
                        'total_search_ns': 30_000_000, 'category_sum_ns': 30_000_000, 'reconciled': True})
        return subprocess.CompletedProcess(args, 0, campaign.raw(raw), b'')

    def produce(self, process=None):
        with mock.patch.object(search, 'validate_plan', return_value=self.plan), \
             mock.patch.object(categories.os, 'getpriority', return_value=0), \
             mock.patch.object(categories.platform, 'platform', return_value='fixture-platform'), \
             mock.patch.object(categories.subprocess, 'run', side_effect=process or self.fake_process):
            return categories.produce(self.root, self.context, self.phase)

    @staticmethod
    def replace(path, body):
        path.unlink()
        return campaign.seal(path, {key: value for key, value in body.items() if key != 'body_sha256'})

    def test_complete_four_arm_batch_reproduces_trace_shares_and_separate_overhead(self):
        before = search.validate_measurement(self.plan, self.output)[0]
        receipt = self.produce()
        self.assertEqual(len(receipt['runs']), 12)
        self.assertTrue(receipt['complete'])
        self.assertFalse(receipt['timing_retention_input'])
        for value in receipt['variants'].values():
            self.assertEqual(value['rows'], 6)
            self.assertAlmostEqual(sum(value['shares'].values()), 1.0)
            self.assertAlmostEqual(value['shares']['residual_search'], .8)
            self.assertEqual(value['instrumentation_overhead']['observed_elapsed_ratio_minus_one'], 2.0)
        status = search.category_profile_status(self.plan, self.output)
        self.assertTrue(status['complete'])
        self.assertEqual(status['receipt'], campaign.record(self.output / 'category-profile.json'))
        self.assertEqual(search.validate_measurement(self.plan, self.output)[0], before)
        with mock.patch.object(search, 'validate_plan', return_value=self.plan), \
             mock.patch.object(categories.subprocess, 'run', side_effect=AssertionError('completed batch rerun')):
            self.assertEqual(categories.produce(self.root, self.context, self.phase), receipt)

    def test_partial_claim_and_failed_native_execution_are_never_retried(self):
        def failure(args, **kwargs):
            if '-o' in args:
                return self.fake_process(args, **kwargs)
            return subprocess.CompletedProcess(args, 1, b'{"partial":true}', b'failed')
        with self.assertRaises(subprocess.CalledProcessError):
            self.produce(failure)
        self.assertTrue((self.output / 'category/claim.json').exists())
        self.assertFalse((self.output / 'category-profile.json').exists())
        with self.assertRaisesRegex(ValueError, 'no automatic retry'):
            self.produce()

    def test_arbitrary_summary_missing_run_and_modified_share_cannot_authorize(self):
        receipt = self.produce()
        path = self.output / 'category-profile.json'
        for body in ({'schema': categories.SCHEMA, 'complete': True},
                     {**receipt, 'runs': receipt['runs'][:-1]}):
            self.replace(path, body)
            with self.assertRaises(ValueError):
                categories.validate(self.plan, self.output)
        altered = copy.deepcopy(receipt)
        altered['variants']['baseline']['shares']['first_layer'] = .99
        self.replace(path, altered)
        with self.assertRaisesRegex(ValueError, 'shares or overhead'):
            categories.validate(self.plan, self.output)

    def test_resealed_derivative_change_cannot_replace_deterministic_transformation(self):
        receipt = self.produce()
        claim_path = Path(receipt['claim']['path'])
        claim = campaign.read(claim_path)
        build = claim['builds']['baseline']
        derivative = Path(build['instrumented_source']['path'])
        derivative.write_bytes(derivative.read_bytes() + b'\n')
        build['instrumented_source'] = campaign.record(derivative)
        self.replace(claim_path, claim)
        receipt['claim'] = campaign.record(claim_path)
        self.replace(self.output / 'category-profile.json', receipt)
        with self.assertRaisesRegex(ValueError, 'transformation anchors'):
            categories.validate(self.plan, self.output)

    def test_byte_identical_relocated_validators_accept_original_execution_paths(self):
        receipt = self.produce()
        original = campaign.read(Path(receipt['claim']['path']))['producers']
        relocated = {}
        for name, record in original.items():
            path = self.root / 'another-snapshot' / (name + Path(record['path']).suffix)
            campaign.once(path, campaign.verify(record).read_bytes())
            relocated[name] = campaign.record(path)
        with mock.patch.object(categories, 'producer_records', return_value=relocated), \
             mock.patch.object(categories, 'PROBE', Path(relocated['category_probe']['path'])):
            self.assertEqual(categories.validate(self.plan, self.output), receipt)
            changed = Path(relocated['category_probe']['path'])
            changed.write_bytes(changed.read_bytes() + b'\n')
            relocated['category_probe'] = campaign.record(changed)
            with self.assertRaisesRegex(ValueError, 'source closure'):
                categories.validate(self.plan, self.output)

    def test_relocated_validation_still_verifies_original_producer_files(self):
        receipt = self.produce()
        original = campaign.read(Path(receipt['claim']['path']))['producers']
        relocated = {}
        for name, record in original.items():
            path = self.root / 'old-snapshot' / (name + Path(record['path']).suffix)
            campaign.once(path, campaign.verify(record).read_bytes())
            relocated[name] = campaign.record(path)
        # Relocate only recorded provenance; it must remain verifiable even
        # when the presently executing implementations have identical bytes.
        claim_path = Path(receipt['claim']['path'])
        claim = campaign.read(claim_path)
        claim['producers'] = relocated
        for build in claim['builds'].values():
            build['command'][-3] = relocated['category_probe']['path']
        self.replace(claim_path, claim)
        receipt['claim'] = campaign.record(claim_path)
        receipt = self.replace(self.output / 'category-profile.json', receipt)
        self.assertEqual(categories.validate(self.plan, self.output), receipt)
        changed = Path(relocated['instrumentation']['path'])
        changed.write_bytes(changed.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'changed artifact'):
            categories.validate(self.plan, self.output)

    def test_original_trace_node_count_or_category_loss_stops_receipt_creation(self):
        for field, value in (('fixed_trace', 'changed'), ('nodes', 101), ('category_sum_ns', 1)):
            raw = self.raw()
            name = 'baseline'
            attributed = json.loads(self.fake_process(
                [str(self.output / 'category/builds' / name / 'probe.bin'), 'roots', 'fixed'],
                env=campaign.THREADS).stdout)
            attributed['rows'][0][field] = value
            source = campaign.verify(self.plan['variants'][name]['source'])
            _, manifest = instrumentation.instrument_source(source.read_bytes(), campaign.sha(source))
            with self.assertRaises(ValueError):
                categories.validate_raw(attributed, raw, self.plan, manifest)

    def test_category_profiling_does_not_accept_independent_treatment_rosters(self):
        plan = {**self.plan, 'profile': 'cache-v1'}
        with self.assertRaisesRegex(ValueError, 'four standard search arms'):
            categories.schedule(plan)


if __name__ == '__main__':
    unittest.main()
