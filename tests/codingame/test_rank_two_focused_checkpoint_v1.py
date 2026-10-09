import ast
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools import rank_two_focused_checkpoint_v1 as codec
from tools import rank_two_focused_training_v4 as retained
from tools import rank_two_focused_training_v5 as training

np, campaign = codec.np, codec.campaign
CURSOR = dict(phase='float', epoch=0, next_root=1, rng_seed=2026100701, root_order=['root-a', 'root-b'])
NUMERIC_GROUP = dict(mover=0, exhaustive=True, split='train', successors=[
    dict(active=[1, 4, 17], teacher_value=.2, value_mover=0),
    dict(active=[2, 9, 34], teacher_value=-.4, value_mover=1)])


class PreservedDirectory:
    def __enter__(self):
        root=Path(os.environ['PAPERSOCCER_CHECKPOINT_FIXTURE_DIR']);root.mkdir(parents=True,exist_ok=True)
        self.path=tempfile.mkdtemp(prefix='fixture-',dir=root)
        return self.path

    def __exit__(self,*args):
        return False


def fixture_directory():
    if os.environ.get('PAPERSOCCER_CHECKPOINT_FIXTURE_DIR'):
        return PreservedDirectory()
    return tempfile.TemporaryDirectory(prefix='focused-checkpoint-unit-',dir=campaign.ROOT/'results')


def model(profile):
    architecture, parameters = training.initialize(profile, 2026100701)
    optimizer = training.core.AdamW(parameters, learning_rate=.001, weight_decay=1e-5)
    return architecture, parameters, optimizer


class LosslessCheckpointTests(unittest.TestCase):
    def setUp(self):
        codec.CACHE.clear()

    def test_restore_weights_optimizer_cursor_and_next_update_are_bit_exact(self):
        for profile in ('dd8', 'dd12wide'):
            with self.subTest(profile=profile), fixture_directory() as tmp:
                root = Path(tmp)
                binding = campaign.immutable(root / 'PLAN.json', dict(profile=profile, seed=2026100701))
                arch, parameters, optimizer = model(profile)
                training.update_root(parameters, arch, optimizer, [NUMERIC_GROUP])
                receipt = codec.checkpoint(root / 'checkpoint.psc', parameters, optimizer, CURSOR, binding)
                expected = {name:value.tobytes() for name, value in codec.arrays(parameters, optimizer).items()}
                _, restored, other = model(profile)
                codec.CACHE.clear()
                self.assertEqual(codec.restore(receipt, restored, other, binding), CURSOR)
                self.assertEqual(other.step, optimizer.step)
                self.assertEqual(expected, {name:value.tobytes() for name, value in codec.arrays(restored, other).items()})
                before = retained.update_root(parameters, arch, optimizer, [NUMERIC_GROUP])
                after = training.update_root(restored, arch, other, [NUMERIC_GROUP])
                self.assertEqual(before, after)
                self.assertEqual({name:value.tobytes() for name,value in codec.arrays(parameters, optimizer).items()},
                                 {name:value.tobytes() for name,value in codec.arrays(restored, other).items()})

    def test_xor_chains_have_periodic_full_anchors_and_preserve_every_checkpoint(self):
        with fixture_directory() as tmp:
            root = Path(tmp)
            binding = campaign.immutable(root / 'PLAN.json', dict(profile='dd8', seed=2026100701))
            _, parameters, optimizer = model('dd8')
            parent, receipts = None, []
            for step in range(35):
                for value in codec.arrays(parameters, optimizer).values():
                    value[:] = np.nextafter(value, np.float32(np.inf))
                optimizer.step = step
                parent = codec.checkpoint(root / ('root-' + str(step) + '.psc'), parameters, optimizer,
                                          dict(CURSOR, next_root=step + 1), binding, parent)
                receipts.append(parent)
            self.assertGreater(sum(r['storage']['mode'] == 'xor' for r in receipts), 30)
            self.assertEqual(receipts[32]['storage']['mode'], 'full')
            self.assertLessEqual(max(r['storage']['depth'] for r in receipts), codec.MAX_DEPTH)
            self.assertLess(sum(r['storage']['bytes'] for r in receipts), sum(r['storage']['raw_bytes'] for r in receipts) // 4)
            codec.CACHE.clear()
            _, restored, other = model('dd8')
            codec.restore(receipts[-1], restored, other, binding)
            self.assertEqual(other.step, optimizer.step)
            self.assertEqual({name:value.tobytes() for name,value in codec.arrays(parameters, optimizer).items()},
                             {name:value.tobytes() for name,value in codec.arrays(restored, other).items()})
            for receipt in receipts:
                campaign.verify(receipt['checkpoint'])

    def test_changed_parent_is_rejected_even_when_child_is_cached(self):
        with fixture_directory() as tmp:
            root = Path(tmp)
            binding = campaign.immutable(root / 'PLAN.json', dict(profile='dd8'))
            _, parameters, optimizer = model('dd8')
            parent = codec.checkpoint(root / 'parent.psc', parameters, optimizer, CURSOR, binding)
            parameters['w1'][:] = np.nextafter(parameters['w1'], np.float32(np.inf))
            child = codec.checkpoint(root / 'child.psc', parameters, optimizer, CURSOR, binding, parent)
            self.assertEqual(child['storage']['mode'], 'xor')
            with (root / 'parent.json').open('a') as stream:
                stream.write(' ')
            with self.assertRaisesRegex(ValueError, 'changed bound'):
                codec.decode(child, binding)

    def test_forged_cursor_is_rejected_by_binary_header_with_and_without_cache(self):
        with fixture_directory() as tmp:
            root = Path(tmp)
            binding = campaign.immutable(root / 'PLAN.json', dict(profile='dd8'))
            _, parameters, optimizer = model('dd8')
            receipt = codec.checkpoint(root / 'checkpoint.psc', parameters, optimizer, CURSOR, binding)
            forged = copy.deepcopy(receipt)
            forged['cursor']['next_root'] = 123
            forged['storage']['cursor']['next_root'] = 123
            (root / 'checkpoint.json').write_text(json.dumps(forged))
            for clear_cache in (False, True):
                if clear_cache:
                    codec.CACHE.clear()
                with self.assertRaisesRegex(ValueError, 'header/receipt'):
                    codec.decode(forged, binding)

    def test_changed_payload_and_cross_plan_restore_are_rejected(self):
        with fixture_directory() as tmp:
            root = Path(tmp)
            binding = campaign.immutable(root / 'PLAN.json', dict(profile='dd8'))
            _, parameters, optimizer = model('dd8')
            receipt = codec.checkpoint(root / 'checkpoint.psc', parameters, optimizer, CURSOR, binding)
            with self.assertRaisesRegex(ValueError, 'checkpoint/plan'):
                codec.decode(receipt, dict(path='different', sha256='different'))
            with (root / 'checkpoint.psc').open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'changed bound'):
                codec.decode(receipt, binding)

    def test_nonfinite_arrays_and_low_disk_space_are_rejected_before_writing(self):
        with fixture_directory() as tmp:
            root = Path(tmp)
            binding = campaign.immutable(root / 'PLAN.json', dict(profile='dd8'))
            _, parameters, optimizer = model('dd8')
            with mock.patch.object(codec.shutil, 'disk_usage', return_value=mock.Mock(free=1)):
                with self.assertRaises(codec.DiskBudgetExceeded):
                    codec.checkpoint(root / 'low-space.psc', parameters, optimizer, CURSOR, binding)
            self.assertFalse((root / 'low-space.psc').exists())
            parameters['w1'][0, 0] = np.float32(np.nan)
            with self.assertRaisesRegex(ValueError, 'finite float32'):
                codec.checkpoint(root / 'invalid.psc', parameters, optimizer, CURSOR, binding)
            self.assertFalse((root / 'invalid.psc').exists())

    def test_trainer_preserves_all_numerical_loss_calibration_and_selection_functions(self):
        names = ('initialize', 'forward', 'root_rows', 'objective', 'update_root', 'diagnostics',
                 'calibrate', 'stopping', 'runtime', 'validation', 'validate_corpus_binding')
        def functions(path):
            return {node.name:ast.dump(node, include_attributes=False) for node in ast.parse(Path(path).read_text()).body
                    if isinstance(node, ast.FunctionDef)}
        before, after = functions(retained.__file__), functions(training.__file__)
        for name in names:
            self.assertEqual(before[name], after[name], name)


if __name__ == '__main__':
    unittest.main()
