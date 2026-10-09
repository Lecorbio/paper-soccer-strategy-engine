import io
import json
import sqlite3
import unittest

from tools import rank_two_focused_exposure_v4 as exposure

CASES = [['1','0','x'], ['1','03x6'], ['1','0',''], ['1','0',dict(player_id=1,action='2')]]


class InvalidExposureTests(unittest.TestCase):
    def test_malformed_text_is_retained_and_only_its_valid_prefix_is_indexed(self):
        for turns in CASES:
            with self.subTest(turns=turns):
                stream = io.StringIO()
                normal = exposure.Normalizer(stream)
                ref = dict(path='bound-raw-fixture',sha256='fixture')
                normal.emit(turns,ref,'/turns',group='fixture-root')
                normalized = json.loads(stream.getvalue())
                db = sqlite3.connect(':memory:')
                db.execute('CREATE TABLE keys(category TEXT,value TEXT,PRIMARY KEY(category,value)) WITHOUT ROWID')
                try:
                    budget = exposure.census.Budget(1200,4*1024**3)
                    expected = exposure.census.RowCensus('.',db,budget)
                    actual = exposure.census.RowCensus('.',db,budget)
                    expected.trace(turns,ref,'/turns','focused-exclusion-only','fixture-root')
                    actual.trace(normalized['turns'],ref,'/turns',normalized['kind'],normalized['group'])
                    self.assertEqual(expected.states,actual.states)
                    self.assertEqual(expected.features,actual.features)
                    self.assertEqual(expected.ledger,actual.ledger)
                    self.assertEqual(dict(expected.counts),dict(actual.counts))
                    self.assertEqual(actual.counts['truncated_tail_instances'],1)
                    self.assertEqual(normalized['input'],ref)
                    self.assertFalse(normalized['training_eligible'])
                    if any(isinstance(turn,str) and 'x' in turn for turn in turns):
                        self.assertIn('x',json.dumps(normalized['turns']))
                finally:
                    db.close()

    def test_valid_primitive_strings_remain_identical(self):
        stream=io.StringIO();normal=exposure.Normalizer(stream)
        normal.emit('1/0/2',dict(path='fixture',sha256='fixture'),'/turns')
        self.assertEqual(json.loads(stream.getvalue())['turns'],['1','0','2'])


if __name__=='__main__':
    unittest.main()
