import copy
import gzip
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from tools import rank_two_live_eight_group_stream_v2 as stream
from tools import rank_two_live_eight_validation_stream_v1 as validation
from tests.codingame.test_rank_two_focused_training_v1 import groups

t = validation.training


class StreamedValidationTests(unittest.TestCase):
    def corpus(self, directory):
        documents = []
        items = []
        for index, original in enumerate(groups()):
            document = copy.deepcopy(original)
            document.update(root_group_id='root', group_id='group-%d' % index,
                split='validation', edges=8 + index)
            document['successors'] *= 1031
            documents.append(document)
            path = Path(directory) / ('group-%d.json.gz' % index)
            path.write_bytes(gzip.compress(json.dumps(document, sort_keys=True).encode()))
            items.append(dict(group_id=document['group_id'], artifact=t.reader.record(path)))
        return SimpleNamespace(identity=dict(path='unit-corpus', sha256='fixed'),
            roots=dict(validation=dict(root=items)), manifest={},
            root=lambda split, root: copy.deepcopy(documents)), documents

    def accounting(self, corpus, compared, early):
        if compared != {'root'} or early != 1:
            raise ValueError('complete comparable root required')
        return dict(frozen=1, evaluated=1, excluded=0, comparable_early=1)

    def test_exact_float_and_four_six_bit_whole_group_root_metrics(self):
        architecture, parameters = t.retained.initialize('dd8', 2026101134)
        with tempfile.TemporaryDirectory() as directory:
            corpus, _ = self.corpus(directory)
            for bits in (4, 6):
                scales = {key: np.float32(max(float(np.max(np.abs(value))) / ((1 << (bits-1))-1), 1e-8))
                          for key, value in parameters.items()}
                for effective_scales in (None, scales):
                    q = None if effective_scales is None else t.model.quantize(parameters, effective_scales, bits)
                    calls = []
                    def native(root, active):
                        calls.append(len(active))
                        return t.retained.forward(parameters, architecture, active, q)[0]
                    with patch.object(t, 'validation_accounting', side_effect=self.accounting):
                        expected = t.validation(corpus, parameters, architecture, bits, effective_scales, native)
                        calls.clear()
                        actual = validation.validation(corpus, parameters, architecture, bits,
                            effective_scales, native, maximum=127)
                    self.assertEqual(actual, expected)
                    self.assertGreater(len(calls), 1)
                    self.assertLessEqual(max(calls), 127)

    def test_durable_root_reuse_and_changed_recipe_rejection(self):
        architecture, parameters = t.retained.initialize('dd8', 1)
        with tempfile.TemporaryDirectory() as directory:
            corpus, _ = self.corpus(directory)
            output = Path(directory) / 'journal'
            binding = dict(numeric_plan='unchanged', checkpoint='full-state')
            with patch.object(t, 'validation_accounting', side_effect=self.accounting):
                result = validation.validation(corpus, parameters, architecture, 4,
                    output=output, binding=binding, maximum=127)
                with patch.object(validation, 'root_metrics', side_effect=AssertionError('completed root replayed')):
                    self.assertEqual(validation.validation(corpus, parameters, architecture, 4,
                        output=output, binding=binding, maximum=127), result)
                with self.assertRaises(ValueError):
                    validation.validation(corpus, parameters, architecture, 6,
                        output=output, binding=binding, maximum=127)

    def test_native_mismatch_and_unbounded_batch_are_rejected(self):
        architecture, parameters = t.retained.initialize('dd8', 1)
        with tempfile.TemporaryDirectory() as directory:
            corpus, _ = self.corpus(directory)
            with self.assertRaises(ValueError):
                validation.validation(corpus, parameters, architecture, 4,
                    native=lambda root, active: np.ones(len(active), dtype=np.float32), maximum=127)
            with self.assertRaises(ValueError):
                validation.validation(corpus, parameters, architecture, 4, maximum=2049)

    def test_explicit_partition_and_original_train_only_reader(self):
        from tools import rank_two_live_eight_group_stream_v1 as old
        document = dict(edges=8, exhaustive=True, group_id='group', mover=0,
            root_group_id='root', split='validation', successors=[dict(active=[1], teacher_value=.5, value_mover=0)])
        encoded = json.dumps(document, sort_keys=True)
        actual = list(stream.successors(io.StringIO(encoded), 'root', 'group', 'validation'))
        self.assertEqual([child for _, child in actual], document['successors'])
        with self.assertRaises(PermissionError):
            list(stream.successors(io.StringIO(encoded), 'root', 'group', 'train'))
        with self.assertRaises(PermissionError):
            list(old.successors(io.StringIO(encoded), 'root', 'group'))


if __name__ == '__main__':
    unittest.main()
