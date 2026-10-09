import contextlib
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest import mock

from tools import rank_two_focused_census_v3 as incremental

census, campaign = incremental.census, incremental.campaign
REFERENCE = dict(path='retained-fixture', sha256='fixture')
CASES = [[], ['1'], ['1', '0'], ['1', '0', '2'], ['0'] * 12,
         ['1', '0', '6'], ['1', '0', ''], ['1', '0', 'x'], ['1', '0', '22'],
         ['1', '0', dict(player_id=0, action='2')], ['1', '0', dict(player_id=1, action='2')]]


def database():
    db = sqlite3.connect(':memory:')
    db.execute('CREATE TABLE keys(category TEXT, value TEXT, PRIMARY KEY(category,value)) WITHOUT ROWID')
    return db


def commit(db, item, teacher=False):
    with db:
        for name in ('states', 'features', 'groups'):
            census.union(db, name, getattr(item, name))
        if teacher:
            census.union(db, 'teacher:focused_observed_states', item.states)
            census.union(db, 'teacher:focused_observed_features', item.features)


class PrefixCensusTests(unittest.TestCase):
    def test_teacher_prefix_reuse_preserves_union_counts_and_ledger(self):
        old, new = database(), database()
        budget = census.Budget(1200, 4 * 1024**3)
        physical, anchor = {}, None
        kinds = [('plain', ['1', '0']), ('focused-teacher-parent', ['1', '0'])]
        kinds += [('focused-teacher-successor', value) for value in CASES]
        kinds += [('focused-teacher-parent', ['2', '0']), ('focused-teacher-successor', ['2', '0', '3'])]
        hits = 0
        try:
            for index, (kind, turns) in enumerate(kinds):
                teacher = kind.startswith('focused-teacher')
                retained = census.RowCensus('.', old, budget)
                cached = incremental.PrefixRows('.', new, budget, physical, anchor, teacher)
                for item in (retained, cached):
                    item.trace(turns, REFERENCE, '/fixture/' + str(index), kind, 'root-' + str(index))
                self.assertEqual(dict(retained.counts), dict(cached.counts))
                self.assertEqual(retained.ledger, cached.ledger)
                commit(old, retained, teacher)
                commit(new, cached, teacher)
                anchor = cached.publish_committed(physical)
                hits += cached.prefix_reuses
                self.assertEqual(list(old.execute('SELECT * FROM keys ORDER BY category,value')),
                                 list(new.execute('SELECT * FROM keys ORDER BY category,value')))
            self.assertGreater(hits, 5)
        finally:
            old.close()
            new.close()

    def test_uncommitted_row_cannot_publish_a_prefix(self):
        db, physical = database(), {}
        try:
            item = incremental.PrefixRows('.', db, census.Budget(1200, 4 * 1024**3), physical, teacher=True)
            item.trace(['1', '0'], REFERENCE, '/parent', 'focused-teacher-parent')
            self.assertEqual(physical, {})
            successor = incremental.PrefixRows('.', db, item.budget, physical, teacher=True)
            successor.trace(['1', '0', '2'], REFERENCE, '/successor', 'focused-teacher-successor')
            self.assertEqual(successor.prefix_reuses, 0)
            self.assertTrue(successor.states)
        finally:
            db.close()

    def test_inherited_plain_states_are_not_feature_reencoded(self):
        db = database()
        try:
            budget = census.Budget(1200, 4 * 1024**3)
            old = census.RowCensus('.', db, budget)
            old.trace(['1', '0'], REFERENCE, '/input', 'plain')
            commit(db, old)
            with mock.patch.object(census.banks, 'feature_key', side_effect=AssertionError('ancestry replay')):
                item = incremental.PrefixRows('.', db, budget, {})
                item.trace(['1', '0'], REFERENCE, '/input', 'plain')
            self.assertGreater(item.indexed_reuses, 0)
        finally:
            db.close()

    def test_bounded_stages_resume_without_modifying_seed_or_replaying_rows(self):
        with tempfile.TemporaryDirectory(prefix='focused-prefix-unit-', dir=campaign.ROOT / 'results') as tmp:
            root = Path(tmp)
            source = root / 'seed.sqlite3'
            db = sqlite3.connect(source)
            cutoff = campaign.immutable(root / 'CUTOFF.json', dict(cutoff_utc=campaign.now().isoformat()))
            scope = campaign.immutable(root / 'scope.json', dict(campaign_root=str(campaign.ROOT / 'results'),
                cutoff=cutoff, inherit=[]))
            lines = [json.dumps(dict(mode='trace', turns=turns, input=REFERENCE, locator='/fixture',
                     kind=kind, group='root'), sort_keys=True) + '\n' for turns, kind in
                     [(['1', '0'], 'focused-teacher-parent'), (['1', '0', '2'], 'focused-teacher-successor'),
                      (['1', '0', '3'], 'focused-teacher-successor')]]
            records = campaign.immutable(root / 'records.jsonl', ''.join(lines).encode())
            closure = campaign.immutable(root / 'closure.json', dict(rows=3, inputs=[], scope=scope))
            seed_binding = dict(schema=census.SCHEMA, sources=census.producers(), records=records,
                                scope=scope, closure=closure)
            census.initialize(db, seed_binding, campaign.read(scope['path']), campaign.read(closure['path']),
                              None, census.Budget(1200, 4 * 1024**3))
            db.close()
            seed_db = campaign.record(source)
            seed_inputs = dict(database=seed_db, binding=campaign.immutable(root / 'seed-binding.json', seed_binding),
                inventory=campaign.immutable(root / 'seed-inventory.json', {}),
                receipt=campaign.immutable(root / 'seed-receipt.json', {}))
            producer = campaign.record(incremental.__file__)
            original = dict(rows=3, producers=census.producers(), records=records, scope=scope, closure=closure,
                seed_prior=campaign.immutable(root / 'seed-prior.json', dict(inputs=seed_inputs)),
                seed_inputs=seed_inputs, exposures=[], **{key:producer for key in
                    ('producer', 'teacher_validator', 'proposal_algorithm', 'prior_validator')})
            normalized = campaign.immutable(root / 'normalized.json', original)
            plan = dict(producer=producer, normalized_plan=normalized, committed_rows=0, maximum_new_rows=2,
                output=str(root / 'stage1'), cpu_seconds=1200, maximum_rss_bytes=4 * 1024**3)
            p = campaign.immutable(root / 'PLAN1.json', plan)
            with mock.patch.object(incremental.work, 'checked', return_value={}), \
                 mock.patch.object(incremental.validator, 'prior_census', return_value=contextlib.nullcontext()):
                first = incremental.extend(p['path'])
            self.assertFalse(first['complete'])
            self.assertEqual(first['committed_rows'], 2)
            self.assertEqual(first['new_rows'], 2)
            self.assertEqual(campaign.record(source), seed_db)
            second_plan = dict(plan, committed_rows=2, maximum_new_rows=1, output=str(root / 'stage2'),
                               parent_database=first['database'], parent_binding=first['binding'])
            p = campaign.immutable(root / 'PLAN2.json', second_plan)
            with mock.patch.object(incremental.work, 'checked', return_value={}), \
                 mock.patch.object(incremental.validator, 'prior_census', return_value=contextlib.nullcontext()):
                second = incremental.extend(p['path'])
            self.assertTrue(second['complete'])
            self.assertEqual(second['new_rows'], 1)
            self.assertEqual(campaign.record(first['database']['path']), first['database'])
            with sqlite3.connect(second['database']['path']) as done:
                self.assertEqual(done.execute('SELECT row_sha256 FROM completed WHERE number=1').fetchone()[0],
                                 hashlib.sha256(lines[0].encode()).hexdigest())
                self.assertGreater(done.execute("SELECT count(*) FROM keys WHERE category LIKE 'teacher:%'").fetchone()[0], 0)


if __name__ == '__main__':
    unittest.main()
