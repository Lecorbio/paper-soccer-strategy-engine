import io
import json
import unittest

from tools import rank_two_live_eight_group_stream_v1 as stream


class TinyReads(io.StringIO):
    def read(self, size=-1):
        return super().read(min(size, 7) if size >= 0 else 7)


class GroupStreamTests(unittest.TestCase):
    def document(self):
        return dict(group_id='group', root_group_id='root', split='train', mover=0,
            exhaustive=True, metadata='escaped "quote" and unicode π', number=.123456789,
            successors=[dict(active=[1, 316, 6300], teacher_value=.0123456789,
                             value_mover=0, transcript='0/2') for _ in range(3000)])

    def test_chunk_boundaries_preserve_exact_successor_order_and_values(self):
        doc = self.document()
        encoded = json.dumps(doc, sort_keys=True, ensure_ascii=False)
        actual = [child for _, child in stream.successors(TinyReads(encoded), 'root', 'group')]
        self.assertEqual(actual, doc['successors'])

    def test_heldout_or_replaced_root_is_rejected(self):
        for key, value in [('split', 'validation'), ('root_group_id', 'other')]:
            doc = self.document(); doc[key] = value
            with self.assertRaises(PermissionError):
                list(stream.successors(io.StringIO(json.dumps(doc, sort_keys=True)), 'root', 'group'))

    def test_truncated_group_is_rejected(self):
        encoded = json.dumps(self.document(), sort_keys=True)
        with self.assertRaises((ValueError, json.JSONDecodeError)):
            list(stream.successors(TinyReads(encoded[:-10]), 'root', 'group'))


if __name__ == '__main__':
    unittest.main()
