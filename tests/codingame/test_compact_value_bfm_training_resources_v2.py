from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools import compact_value_bfm_campaign_v2 as campaign
from tools import compact_value_bfm_training_resources_v2 as resources


def authorize(root):
    resources.authorize(root, authorized_at_utc='2026-09-05T10:00:00Z', user_request=resources.USER_REQUEST)
    return campaign.record(resources.authorization_path(root))


def root_fixture(root):
    root = root.resolve()
    campaign.seal(root / 'campaign.json', {'policy': campaign.POLICY, 'fixture': True})
    return root


def context(root, record):
    return {'parent_campaign': campaign.record(root / 'campaign.json'),
            'training_executor': {'mode': 'spawn-v2', 'maximum_workers': 4},
            'training_resource_authorization': record}


class TrainingResourceTests(unittest.TestCase):
    def test_explicit_authorization_preserves_original_policy_and_freezes_bounded_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = root_fixture(Path(tmp)); before = (root / 'campaign.json').read_bytes()
            record = authorize(root); document = resources.validate_authorization(record, root)
            self.assertEqual(document['training']['maximum_seed_workers'], 4)
            self.assertEqual(document['training']['native_threads_per_worker'], 1)
            self.assertEqual(document['benchmark']['temporary_total_seed_workers'], 6)
            self.assertFalse(document['benchmark']['timing_is_speed_evidence'])
            self.assertFalse(document['benchmark']['current_training_restart_authorized'])
            self.assertEqual(campaign.POLICY['workers']['training_seeds_max'], 2)
            self.assertEqual(before, (root / 'campaign.json').read_bytes())
            self.assertEqual(record, authorize(root))

    def test_default_and_existing_two_workers_need_no_new_authority(self):
        with mock.patch.object(resources, 'validate_authorization', side_effect=AssertionError('no auth needed')):
            self.assertEqual(resources.expected_workers({}), 2)
            self.assertEqual(resources.expected_workers({'training_executor': {'mode': 'spawn-v2', 'maximum_workers': 2}}), 2)

    def test_four_workers_require_exact_source_bound_campaign_authorization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = root_fixture(Path(tmp))
            value = context(root, None)
            with self.assertRaisesRegex(ValueError, 'source-bound user'):
                resources.expected_workers(value)
            value['training_resource_authorization'] = authorize(root)
            self.assertEqual(resources.expected_workers(value), 4)
            value['training_executor']['maximum_workers'] = 8
            with self.assertRaisesRegex(ValueError, 'two- or four-worker'):
                resources.expected_workers(value)

    def test_other_campaign_authority_is_rejected_before_opening_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = root_fixture(Path(tmp) / 'one'); other = root_fixture(Path(tmp) / 'two')
            record = authorize(other)
            with mock.patch.object(campaign, 'verify', side_effect=AssertionError('wrong path read')):
                with self.assertRaisesRegex(ValueError, 'fixed campaign path'):
                    resources.validate_authorization(record, root)

    def test_resealed_worker_numeric_thread_and_benchmark_cap_drift_is_rejected(self):
        for section, key, value in (('training', 'maximum_seed_workers', 8),
                                    ('training', 'native_threads_per_worker', 2),
                                    ('benchmark', 'temporary_total_seed_workers', 8)):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                root = root_fixture(Path(tmp)); record = authorize(root)
                document = campaign.read(Path(record['path'])); document.pop('body_sha256')
                document[section][key] = value
                Path(record['path']).unlink(); campaign.seal(Path(record['path']), document)
                with self.assertRaisesRegex(ValueError, 'four-worker scope'):
                    resources.validate_authorization(campaign.record(Path(record['path'])), root)

    def test_user_request_and_timestamp_are_required_before_any_new_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = root_fixture(Path(tmp))
            with self.assertRaisesRegex(ValueError, 'explicit more-cores'):
                resources.authorize(root, authorized_at_utc='2026-09-05T10:00:00Z', user_request='background task')
            self.assertFalse((root / 'training-resources').exists())
            with self.assertRaisesRegex(ValueError, 'identify UTC'):
                resources.authorize(root, authorized_at_utc='2026-09-05T10:00:00', user_request=resources.USER_REQUEST)

    def test_new_mode_four_and_immutable_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = root_fixture(Path(tmp)); record = authorize(root)
            fields = resources.execution_fields(root, training_workers=4)
            self.assertEqual(fields['training_resource_authorization'], record)
            frozen = {'parent_campaign': campaign.record(root / 'campaign.json'), **fields}
            resources.check_resume(frozen, root)
            resources.check_resume(frozen, root, training_executor='spawn-v2')
            resources.check_resume(frozen, root, training_workers=4)
            with self.assertRaisesRegex(ValueError, 'cannot change its frozen'):
                resources.check_resume(frozen, root, training_workers=2)
            with self.assertRaisesRegex(ValueError, 'only as authorized spawned'):
                resources.execution_fields(root, training_executor='threads', training_workers=4)

    def test_new_pilot_can_freeze_four_without_touching_previous_contexts(self):
        from tools import compact_value_bfm_pilot_v2 as pilot
        from tests.codingame.test_compact_value_bfm_intervention_v2 import fixture, pilot_patches
        with tempfile.TemporaryDirectory() as tmp:
            root, _, _, previous = fixture(Path(tmp)); authorize(root)
            before = {path: path.read_bytes() for row in previous for path in row['context'].rglob('*') if path.is_file()}
            a, b, c, d, e = pilot_patches(root, previous)
            with a, b, c, d, e:
                prepared = pilot.prepare(root, 3, training_workers=4)
                self.assertEqual(resources.expected_workers(prepared), 4)
                self.assertEqual(prepared, pilot.prepare(root, 3))
                with self.assertRaisesRegex(ValueError, 'cannot change its frozen'):
                    pilot.prepare(root, 3, training_workers=2)
            self.assertEqual(before, {path: path.read_bytes() for row in previous for path in row['context'].rglob('*') if path.is_file()})

    def test_future_full_can_use_authorized_four_even_when_pilot_used_threads(self):
        from tests.codingame.test_compact_value_bfm_full_selection_v2 import FullContextTests
        from tools import compact_value_bfm_full_v2 as full
        from tools import compact_value_bfm_full_selection_v2 as selection
        case = FullContextTests(); case.setUp(); self.addCleanup(case.doCleanups)
        root = case.root; authorize(root)
        pilot_path = case.pilot / 'campaign.json'
        parent = campaign.read(pilot_path); parent.pop('body_sha256'); parent['heavy_stage_root'] = str(root)
        pilot_path.unlink(); campaign.seal(pilot_path, parent)
        before = pilot_path.read_bytes()
        for name in ('anchor-derived.json', 'prior-search-validation.json'):
            campaign.seal(root / 'exclusions' / name, {'fixture': True})
        with mock.patch.object(full, 'admitted_pilot', return_value=case.outcome), \
                mock.patch.object(full, 'pilot_fingerprints', return_value=[]):
            context_path, phase, frozen = full.prepare(root, case.pilot, 'attempt-001-pilot', training_workers=4)
            self.assertEqual(resources.expected_workers(frozen), 4)
            self.assertEqual(before, pilot_path.read_bytes())
            self.assertEqual((context_path, phase, frozen), full.prepare(root, case.pilot, 'attempt-001-pilot'))
            self.assertEqual(selection.validate_context(root, context_path, phase)[0], frozen)
            with self.assertRaisesRegex(ValueError, 'cannot change its frozen'):
                full.prepare(root, case.pilot, 'attempt-001-pilot', training_workers=2)

    def test_four_worker_full_selection_does_not_accept_missing_authority(self):
        from tests.codingame.test_compact_value_bfm_full_selection_v2 import FullContextTests
        from tools import compact_value_bfm_full_selection_v2 as selection
        case = FullContextTests(); case.setUp(); self.addCleanup(case.doCleanups)
        case.contract['training_executor'] = {'mode': 'spawn-v2', 'maximum_workers': 4}
        with mock.patch.object(selection.full, 'admitted_pilot', return_value=case.outcome):
            with self.assertRaisesRegex(ValueError, 'source-bound user resource authorization'):
                case.validate()


if __name__ == '__main__':
    unittest.main()
