import contextlib
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest import mock

from tools import rank_two_focused_census_v4 as delta

campaign, census = delta.campaign, delta.census
CASES = [(['1', '0'], 'focused-teacher-parent'), (['1', '0', '2'], 'focused-teacher-successor'),
         (['1', '0', '3'], 'focused-teacher-successor'), (['2', '0'], 'focused-teacher-parent'),
         (['2', '0', '3'], 'focused-teacher-successor'), (['0'] * 12, 'plain')]


def fixture(root):
    reference = campaign.immutable(root / 'observations.json', dict(cases=[dict(turns=t) for t, _ in CASES]))
    cutoff = campaign.immutable(root / 'CUTOFF.json', dict(cutoff_utc=campaign.now().isoformat()))
    scope = campaign.immutable(root / 'scope.json', dict(schema=census.v2.SCOPE_SCHEMA,
        campaign_root=str(campaign.ROOT / 'results'), cutoff=cutoff, inherit=[], historical_protected_reads=False))
    lines = [json.dumps(dict(mode='trace', turns=turns, input=reference, locator='/cases/' + str(i),
        kind=kind, group='fixture-root-' + str(i)), sort_keys=True) + '\n' for i, (turns, kind) in enumerate(CASES)]
    records = campaign.immutable(root / 'records.jsonl', ''.join(lines).encode())
    closure = campaign.immutable(root / 'closure.json', dict(rows=len(lines), inputs=[reference], scope=scope, all_payload_inputs_hash_verified_before_census=True))
    binding = dict(schema=census.SCHEMA, sources=census.producers(), records=records, scope=scope, closure=closure)
    binding_ref = campaign.immutable(root / 'base-binding.json', binding)
    base_path = root / 'base.sqlite3'
    db = sqlite3.connect(base_path)
    budget = census.Budget(1200, 4 * 1024**3)
    census.initialize(db, binding, campaign.read(scope['path']), campaign.read(closure['path']), None, budget)
    with db:
        db.execute('INSERT INTO keys VALUES (?,?)', ('states', '8' * 64))
        db.execute('INSERT INTO keys VALUES (?,?)', ('features', '9' * 64))
    for number, raw in enumerate(lines[:2], 1):
        row = json.loads(raw)
        item = census.RowCensus(str(campaign.ROOT / 'results'), db, budget)
        item.teacher = True
        item.trace(row['turns'], row['input'], row['locator'], row['kind'], row['group'])
        delta.insert_row(db, item, number, hashlib.sha256(raw.encode()).hexdigest())
    db.close()
    normalized = campaign.immutable(root / 'normalized.json', dict(rows=len(lines), records=records,
        scope=scope, closure=closure, producers=census.producers(), exposures=[reference],
        **{key:campaign.record(delta.__file__) for key in
           ('producer', 'teacher_validator', 'proposal_algorithm', 'prior_validator')}))
    base = campaign.immutable(root / 'base-result.json', dict(passed=True, committed_rows=2,
        database=campaign.record(base_path), binding=binding_ref, normalized_plan=normalized))
    plan = dict(producer=campaign.record(delta.__file__), prefix_producer=campaign.record(delta.prefix.__file__),
        normalized_plan=normalized, base_result=base, committed_rows=2, maximum_new_rows=2,
        delta_results=[], output=str(root / 'stage1'), cpu_seconds=1200, maximum_rss_bytes=4 * 1024**3)
    return plan, lines, base_path, reference


class DeltaCensusTests(unittest.TestCase):
    def test_delta_chain_and_final_merge_match_stock_union_and_ledger(self):
        with tempfile.TemporaryDirectory(prefix='focused-delta-unit-', dir=campaign.ROOT / 'results') as tmp, \
             mock.patch.object(delta.work, 'checked', return_value={}):
            root = Path(tmp)
            plan, lines, base_path, reference = fixture(root)
            base_before = campaign.record(base_path)
            p = campaign.immutable(root / 'PLAN1.json', plan)
            first = delta.stage(p['path'])
            self.assertEqual(first['committed_rows'], 4)
            with contextlib.closing(delta.closed_database(first['database'])) as conn:
                self.assertIsNone(conn.execute("SELECT 1 FROM keys WHERE value=?", ('8' * 64,)).fetchone())
            plan2 = dict(plan, committed_rows=4, maximum_new_rows=10, output=str(root / 'stage2'),
                         delta_results=[campaign.record(root / 'stage1/RESULT.json')])
            p = campaign.immutable(root / 'PLAN2.json', plan2)
            second = delta.stage(p['path'])
            self.assertTrue(second['complete'])
            second_before = second['database']
            merge_plan = dict(plan2, committed_rows=len(lines), output=str(root / 'merged'),
                delta_results=plan2['delta_results'] + [campaign.record(root / 'stage2/RESULT.json')])
            p = campaign.immutable(root / 'MERGE.json', merge_plan)
            ready = delta.merge(p['path'])
            self.assertTrue(ready['passed'])
            prior = campaign.read(ready['prior']['path'])
            with delta.validator.prior_census(prior):
                pass
            expected = sqlite3.connect(':memory:')
            expected.execute('CREATE TABLE keys(category TEXT,value TEXT,PRIMARY KEY(category,value)) WITHOUT ROWID')
            with expected:
                expected.execute('INSERT INTO keys VALUES (?,?)', ('states', '8' * 64))
                expected.execute('INSERT INTO keys VALUES (?,?)', ('features', '9' * 64))
            for raw in lines:
                row = json.loads(raw)
                item = census.RowCensus(str(campaign.ROOT / 'results'), expected, census.Budget(1200, 4 * 1024**3))
                item.trace(row['turns'], row['input'], row['locator'], row['kind'], row['group'])
                with expected:
                    for name in ('states', 'features', 'groups'):
                        census.union(expected, name, getattr(item, name))
                    if row['kind'].startswith('focused-teacher'):
                        census.union(expected, 'teacher:focused_observed_states', item.states)
                        census.union(expected, 'teacher:focused_observed_features', item.features)
            final = delta.closed_database(prior['inputs']['database'])
            try:
                self.assertEqual(list(expected.execute('SELECT * FROM keys ORDER BY category,value')),
                                 list(final.execute('SELECT * FROM keys ORDER BY category,value')))
                self.assertEqual(census.checkpoint(final), len(lines))
                for number, raw in enumerate(lines, 1):
                    stored = final.execute('SELECT row_sha256 FROM completed WHERE number=?', (number,)).fetchone()
                    self.assertEqual(stored[0], hashlib.sha256(raw.encode()).hexdigest())
            finally:
                final.close()
                expected.close()
            self.assertEqual(campaign.record(base_path), base_before)
            self.assertEqual(campaign.record(first['database']['path']), first['database'])
            self.assertEqual(campaign.record(second['database']['path']), second_before)

    def test_duplicate_delta_and_incomplete_merge_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix='focused-delta-unit-', dir=campaign.ROOT / 'results') as tmp, \
             mock.patch.object(delta.work, 'checked', return_value={}):
            root = Path(tmp)
            plan, _, _, _ = fixture(root)
            p = campaign.immutable(root / 'PLAN1.json', plan)
            delta.stage(p['path'])
            ref = campaign.record(root / 'stage1/RESULT.json')
            bad = dict(plan, committed_rows=4, delta_results=[ref, ref], output=str(root / 'bad'))
            p = campaign.immutable(root / 'DUPLICATE.json', bad)
            with self.assertRaisesRegex(ValueError, 'gap, overlap'):
                delta.stage(p['path'])
            p = campaign.immutable(root / 'INCOMPLETE.json', dict(bad, delta_results=[ref]))
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                delta.merge(p['path'])
            self.assertFalse((root / 'bad/CLAIM.json').exists())

    def test_unknown_spent_claim_is_never_repeated(self):
        with tempfile.TemporaryDirectory(prefix='focused-delta-unit-', dir=campaign.ROOT / 'results') as tmp, \
             mock.patch.object(delta.work, 'checked', return_value={}):
            root = Path(tmp)
            plan, _, _, _ = fixture(root)
            campaign.immutable(root / 'stage1/CLAIM.json', dict(unknown=True))
            p = campaign.immutable(root / 'PLAN.json', plan)
            with self.assertRaisesRegex(RuntimeError, 'unknown spent'):
                delta.stage(p['path'])

    def test_changed_normalized_input_is_rejected_before_claim(self):
        with tempfile.TemporaryDirectory(prefix='focused-delta-unit-', dir=campaign.ROOT / 'results') as tmp, \
             mock.patch.object(delta.work, 'checked', return_value={}):
            root = Path(tmp)
            plan, _, _, _ = fixture(root)
            with (root / 'records.jsonl').open('a') as stream:
                stream.write('{}\n')
            p = campaign.immutable(root / 'PLAN.json', plan)
            with self.assertRaisesRegex(ValueError, 'changed bound'):
                delta.stage(p['path'])
            self.assertFalse((root / 'stage1/CLAIM.json').exists())

    def test_unconfirmed_tail_is_retained_without_advancing_frontier(self):
        with tempfile.TemporaryDirectory(prefix='focused-delta-unit-', dir=campaign.ROOT / 'results') as tmp, \
             mock.patch.object(delta.work, 'checked', return_value={}):
            root = Path(tmp)
            plan, _, _, _ = fixture(root)
            p = campaign.immutable(root / 'PLAN.json', plan)
            def interrupted(item, *args):
                item.states.add('a' * 64)
                item.features.add('b' * 64)
                raise TimeoutError('bounded fixture interruption')
            with mock.patch.object(delta.DeltaRows, 'trace', interrupted):
                result = delta.stage(p['path'])
            self.assertEqual(result['committed_rows'], 2)
            self.assertEqual(result['new_rows'], 0)
            self.assertFalse(result['unconfirmed_row']['confirmed'])
            conn = delta.closed_database(result['database'])
            try:
                self.assertIsNotNone(conn.execute("SELECT 1 FROM keys WHERE category='states' AND value=?", ('a' * 64,)).fetchone())
                self.assertIsNotNone(conn.execute("SELECT 1 FROM keys WHERE category='teacher:focused_observed_states' AND value=?", ('a' * 64,)).fetchone())
            finally:
                conn.close()


if __name__ == '__main__':
    unittest.main()
