import copy
import fcntl
import gzip
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from tools import compact_value_bfm_parallel_store_v2 as parallel
os.environ.update(parallel.ENV)
from tools import compact_value_bfm_ranking_store as serial
from tests.codingame.test_compact_value_bfm_training import bundle_fixture
from tests.codingame import test_jacek_replay_corpus as corpus_tests


class ReversePool:
    """Finish each bounded batch backwards while its caller consumes forward."""
    finished = []
    largest_pending = 0
    instances = []

    def __init__(self, *, initargs, **_kwargs):
        claim = serial.campaign.read(serial.campaign.verify(initargs[0]))
        value = claim['bundle']
        bundle = serial.trainer.FrozenBundle(Path(value['manifest_path']),
            bytes.fromhex(value['manifest_payload_hex']), value['manifest'])
        parallel._WORKER = claim, bundle
        self.pending = []; self.joined = False
        self.instances.append(self)

    def submit(self, function, job):
        pool = self
        class Future:
            done = False
            value = None
            failure = None
            def result(self):
                for other, callback, argument in reversed(pool.pending):
                    if not other.done:
                        try: other.value = callback(argument)
                        except BaseException as error: other.failure = error
                        other.done = True; pool.finished.append(argument['ordinal'])
                pool.pending[:] = [item for item in pool.pending if not item[0].done]
                if self.failure is not None: raise self.failure
                return self.value
        future = Future(); self.pending.append((future, function, job))
        type(self).largest_pending = max(type(self).largest_pending, len(self.pending))
        return future

    def shutdown(self, **_kwargs):
        self.joined = True


class ParallelRankingStoreTests(unittest.TestCase):
    def setUp(self):
        ReversePool.finished = []; ReversePool.largest_pending = 0; ReversePool.instances = []

    def rows(self):
        rows = [corpus_tests.JacekReplayCorpusTests.complete_turn_action_group_row(
            position_id='position:' + f'{ordinal + 1:064x}',
            split='validation' if ordinal % 2 else 'train', root_action=action)
            for ordinal, action in enumerate(('0', '1', '2', '3', '5', '6'))]
        for ordinal, row in enumerate(rows):
            row['source_bundle_body_sha256'] = 'a' * 64
            item = row['group']['successors'][0]
            item['visits'] = (1 << 40) + ordinal; item['selection_visits'] = ordinal
            if ordinal % 2:
                item['teacher_value'] = -1. if ordinal == 1 else 1.
                item['proof'] = {'solved': True, 'proven_winner': 1 if ordinal == 1 else 0}
                item['termination'] = {'reason': 'subtree-solved', 'value_status': 'exact-sign'}
        # A second legal successor exercises per-group offsets and exact ID order.
        row = rows[0]; item = copy.deepcopy(row['group']['successors'][0])
        state = serial.corpus._prefix_state(row['group']['source_binding']['prefix'])
        serial.corpus.features.apply_complete_turn(state, row['group']['parent_mover'], '1')
        item.update(successor_id=serial.corpus._mover_canonical_position_identity(state),
            active=list(serial.corpus.features.encode_active(state)), transcript='5',
            teacher_value=-0.0, value_mover=state.to_move)
        row['group']['successors'].append(item)
        row['group']['successors'].sort(key=lambda value: value['successor_id'])
        return [serial.corpus.validate_complete_turn_action_group(row) for row in rows]

    def source(self, root, rows, name='labels.jsonl'):
        path = root / name
        payload = b''.join(serial.corpus.canonical_json_bytes(row) for row in rows)
        if name.endswith('.gz'):
            with gzip.open(path, 'wb') as stream: stream.write(payload)
        else: path.write_bytes(payload)
        return path

    def normalized(self, index):
        document = serial.campaign.read(index)
        for name in ('body_sha256', 'builder', 'parallel_build'): document.pop(name, None)
        document['arrays'] = {name: {key: value for key, value in record.items() if key != 'path'}
                              for name, record in document['arrays'].items()}
        return serial.campaign.raw(document)

    def assert_equal(self, left, right, bundle):
        self.assertEqual(self.normalized(left), self.normalized(right))
        a, b = serial.campaign.read(left), serial.campaign.read(right)
        for name in ('indices', 'successors', 'transcripts'):
            self.assertEqual(Path(a['arrays'][name]['path']).read_bytes(), Path(b['arrays'][name]['path']).read_bytes())
        x, y = serial.RankingStore(left, bundle).labels(), serial.RankingStore(right, bundle).labels()
        for first, second in zip((*x.train, *x.validation), (*y.train, *y.validation), strict=True):
            self.assertEqual(first.evidence, second.evidence)
            for l, r in zip(first.successors, second.successors, strict=True):
                self.assertEqual(l.successor_id, r.successor_id)
                self.assertEqual(l.active.tobytes(), r.active.tobytes())
                self.assertEqual(l.evidence, r.evidence)
                self.assertEqual(l.teacher_value, r.teacher_value)
                self.assertEqual(l.value_mover, r.value_mover)

    def test_actual_four_spawn_workers_match_serial_arrays_and_normalized_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); rows = self.rows(); bundle = bundle_fixture(root)
            sources = [self.source(root, rows[:3]), self.source(root, rows[3:], 'last.jsonl.gz')]
            baseline = serial.build_store(sources, root / 'serial', bundle)
            result = parallel.build_store(sources, root / 'parallel', bundle, root=root, chunk_bytes=1)
            self.assert_equal(baseline, result, bundle)
            completion = serial.campaign.read(result.parent / 'parallel-complete.json')
            self.assertEqual(completion['chunks'], 6)
            self.assertTrue(completion['all_submitted_workers_joined'])
            self.assertEqual(completion['settings']['max_pending'], 4)
            self.assertTrue(completion['worker_memory_observations'])
            self.assertFalse(completion['capacity_approval_claimed'])
            self.assertEqual(parallel.build_store(sources, result.parent, bundle, root=root, chunk_bytes=1), result)

    def test_reverse_completion_preserves_global_order_and_bounded_pending(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); rows = self.rows(); bundle = bundle_fixture(root)
            sources = [self.source(root, rows[:2]), self.source(root, rows[2:], 'second.jsonl')]
            baseline = serial.build_store(sources, root / 'serial', bundle)
            with mock.patch.object(parallel.concurrent.futures, 'ProcessPoolExecutor', ReversePool):
                result = parallel.build_store(sources, root / 'parallel', bundle, root=root, workers=8, chunk_bytes=1)
            self.assertEqual(ReversePool.finished, [5, 4, 3, 2, 1, 0])
            self.assertLessEqual(ReversePool.largest_pending, 8)
            self.assertTrue(all(pool.joined for pool in ReversePool.instances))
            self.assert_equal(baseline, result, bundle)

    def test_multiple_groups_per_chunk_and_large_group_never_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); rows = self.rows(); bundle = bundle_fixture(root)
            source = self.source(root, rows)
            size = max(len(serial.corpus.canonical_json_bytes(row)) for row in rows)
            baseline = serial.build_store([source], root / 'serial', bundle)
            with mock.patch.object(parallel.concurrent.futures, 'ProcessPoolExecutor', ReversePool):
                result = parallel.build_store([source], root / 'parallel', bundle, root=root,
                    chunk_bytes=size * 3, max_group_bytes=size * 4)
            self.assert_equal(baseline, result, bundle)
            self.assertLess(serial.campaign.read(result.parent / 'parallel-complete.json')['chunks'], len(rows))

    def test_universal_newlines_and_boundaries_match_serial_reader(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); rows = self.rows(); bundle = bundle_fixture(root)
            source = root / 'mixed.jsonl'
            source.write_bytes(b''.join(serial.corpus.canonical_json_bytes(row).rstrip(b'\n') + ending
                for row, ending in zip(rows, (b'\r', b'\r\n', b'\n', b'\r', b'\r\n', b''), strict=True)))
            baseline = serial.build_store([source], root / 'serial', bundle)
            with mock.patch.object(parallel.concurrent.futures, 'ProcessPoolExecutor', ReversePool):
                result = parallel.build_store([source], root / 'parallel', bundle, root=root, chunk_bytes=1)
            self.assert_equal(baseline, result, bundle)
        self.assertEqual(list(parallel.raw_lines(io.BytesIO(b'a\r\nb\rc\nd'), 3)), [b'a\n', b'b\n', b'c\n', b'd'])
        with self.assertRaisesRegex(ValueError, 'max_group_bytes'):
            list(parallel.raw_lines(io.BytesIO(b'abcd\n'), 3))

    def test_utf16_source_is_not_silently_accepted_by_json_bytes_parser(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); source = root / 'utf16.jsonl'; bundle = bundle_fixture(root)
            source.write_bytes(serial.corpus.canonical_json_bytes(self.rows()[0]).decode().encode('utf-16'))
            with self.assertRaises(UnicodeError): serial.build_store([source], root / 'serial', bundle)
            with mock.patch.object(parallel.concurrent.futures, 'ProcessPoolExecutor', ReversePool):
                with self.assertRaises(UnicodeError): parallel.build_store([source], root / 'parallel', bundle, root=root)

    def test_duplicate_and_cross_split_groups_are_globally_rejected(self):
        for cross_split in (False, True):
            with self.subTest(cross_split=cross_split), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); row = self.rows()[0]; other = copy.deepcopy(row)
                if cross_split:
                    other['split'] = 'validation'; other['group']['source_binding']['split'] = 'validation'
                sources = [self.source(root, [row]), self.source(root, [other], 'second.jsonl')]
                with mock.patch.object(parallel.concurrent.futures, 'ProcessPoolExecutor', ReversePool):
                    with self.assertRaisesRegex(ValueError, 'duplicate/cross-split'):
                        parallel.build_store(sources, root / 'parallel', bundle_fixture(root), root=root, chunk_bytes=1)
                self.assertFalse((root / 'parallel/index.json').exists())
                self.assertTrue((root / 'parallel/parallel-failure.json').exists())

    def test_mixed_teacher_bundle_schema_and_nonexhaustive_fail_without_index(self):
        for field in ('teacher', 'bundle', 'schema', 'nonexhaustive'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); rows = self.rows()[:2]
                if field == 'teacher': rows[1]['teacher']['source_sha256'] = '9' * 64
                if field == 'bundle': rows[1]['source_bundle_body_sha256'] = '9' * 64
                if field == 'schema': rows[1]['feature_schema'] = 'other'
                if field == 'nonexhaustive': rows[1]['group']['successors_exhaustive'] = False
                source = self.source(root, rows)
                with mock.patch.object(parallel.concurrent.futures, 'ProcessPoolExecutor', ReversePool):
                    with self.assertRaises(ValueError):
                        parallel.build_store([source], root / 'parallel', bundle_fixture(root), root=root, chunk_bytes=1)
                self.assertFalse((root / 'parallel/index.json').exists())

    def test_partial_claim_failures_and_limits_require_fresh_output(self):
        for failure in ('raw-limit', 'output-limit', 'worker'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); source = self.source(root, self.rows()); output = root / 'parallel'
                kwargs = {'chunk_bytes': 1}
                if failure == 'raw-limit': kwargs['max_group_bytes'] = 1
                if failure == 'output-limit': kwargs['max_chunk_output_bytes'] = 1
                with mock.patch.object(parallel.concurrent.futures, 'ProcessPoolExecutor', ReversePool):
                    with mock.patch.object(parallel, 'transcribe', side_effect=RuntimeError('worker failed')) if failure == 'worker' else context_noop():
                        with self.assertRaises((ValueError, RuntimeError)):
                            parallel.build_store([source], output, bundle_fixture(root), root=root, **kwargs)
                self.assertTrue((output / 'parallel-claim.json').exists())
                self.assertTrue((output / 'parallel-failure.json').exists())
                self.assertFalse((output / '.scratch').exists())
                with self.assertRaisesRegex(ValueError, 'partially claimed'):
                    parallel.build_store([source], output, bundle_fixture(root), root=root)

    def test_global_lease_precedes_source_access_and_output_claim(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); output = root / 'parallel'
            with (root / '.heavy-stage.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):
                    parallel.build_store([root / 'missing.jsonl'], output, bundle_fixture(root), root=root)
            self.assertFalse(output.exists())

    def test_phase_authority_resolves_global_lease_and_invalid_limits_do_not_claim(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); phase = root / 'phase'; phase.mkdir()
            serial.campaign.seal(root / 'campaign.json', {'schema': 'fixture'})
            serial.campaign.seal(phase / 'campaign.json', {'heavy_stage_root': str(root),
                'parent_campaign': serial.campaign.record(root / 'campaign.json')})
            with (root / '.heavy-stage.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):
                    parallel.build_store([root / 'missing'], phase / 'store', bundle_fixture(root), root=phase)
            for kwargs in ({'workers': 3}, {'workers': True}, {'max_pending': 9}, {'max_group_bytes': 1}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    parallel.build_store([], root / 'bad', bundle_fixture(root), root=root, **kwargs)
            self.assertFalse((root / 'bad').exists())

    def test_foreign_loaded_provider_and_tools_namespace_are_rejected(self):
        for provider in (serial, serial.campaign, serial.corpus, serial.trainer, parallel.modules()[-1]):
            with self.subTest(provider=provider.__name__), mock.patch.object(provider, '__file__', '/foreign/' + Path(provider.__file__).name):
                with self.assertRaisesRegex(ValueError, 'exact source namespace'): parallel.source_closure()
        with mock.patch.dict(parallel.sys.modules, {'tools.foreign_fixture': SimpleNamespace(__file__='/foreign/helper.py')}):
            with self.assertRaisesRegex(ValueError, 'foreign tools provider'): parallel.source_closure()
        with mock.patch.object(parallel.sys.modules['tools'], '__path__', ['/foreign/tools']):
            with self.assertRaisesRegex(ValueError, 'namespace is mixed'): parallel.source_closure()

    def test_completed_tamper_or_different_request_and_legacy_store_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); source = self.source(root, self.rows()); bundle = bundle_fixture(root)
            legacy = serial.build_store([source], root / 'legacy', bundle)
            with self.assertRaisesRegex(ValueError, 'partially claimed'):
                parallel.build_store([source], legacy.parent, bundle, root=root)
            with mock.patch.object(parallel.concurrent.futures, 'ProcessPoolExecutor', ReversePool):
                result = parallel.build_store([source], root / 'parallel', bundle, root=root)
            with self.assertRaisesRegex(ValueError, 'completed claim changed'):
                parallel.build_store([source], result.parent, bundle, root=root, workers=8)
            document = serial.campaign.read(result)
            path = Path(document['arrays']['successors']['path']); path.write_bytes(path.read_bytes() + b'bad')
            with self.assertRaisesRegex(ValueError, 'changed artifact'):
                parallel.build_store([source], result.parent, bundle, root=root)


def context_noop():
    import contextlib
    return contextlib.nullcontext()


if __name__ == '__main__': unittest.main()
