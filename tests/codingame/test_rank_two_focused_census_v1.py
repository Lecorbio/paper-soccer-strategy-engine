import sqlite3
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

from tools import rank_two_focused_census_v1 as continuation
from tools import rank_two_focused_campaign_v1 as campaign

census = continuation.census


def database():
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE keys(category TEXT, value TEXT, PRIMARY KEY(category,value)) WITHOUT ROWID")
    return db


def insert(db, item):
    with db:
        for category in ("states", "features", "groups"):
            census.union(db, category, getattr(item, category))


class CensusContinuationTests(unittest.TestCase):
    def test_transactional_wal_frontier_continues_without_modifying_parent(self):
        with tempfile.TemporaryDirectory(prefix="focused-census-unit-", dir=campaign.ROOT / "results") as tmp:
            root = Path(tmp)
            cutoff = campaign.immutable(root / "CUTOFF.json", dict(cutoff_utc=campaign.now().isoformat()))
            scope = campaign.immutable(root / "scope.json", dict(campaign_root=str(campaign.ROOT / "results"),
                cutoff=cutoff, inherit=[], historical_protected_reads=False))
            rows = [dict(mode="trace", turns=turns, input=dict(path="fixture", sha256="fixture"),
                locator="/" + str(index), kind="fixture", group="root", training_eligible=False)
                for index, turns in enumerate([["1"], ["1", "0"], ["2", "0"]])]
            lines = [json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows]
            records = campaign.immutable(root / "records.jsonl", "".join(lines).encode())
            closure = campaign.immutable(root / "closure.json", dict(scope=scope, cutoff=cutoff, records=records,
                rows=3, inputs=[], all_payload_inputs_hash_verified_before_census=True))
            binding = dict(schema=census.SCHEMA, scope=scope, closure=closure, records=records,
                           sources=census.producers(), inherited_inputs=[])
            binding_ref = campaign.immutable(root / "binding.json", binding)
            live = root / "live.sqlite3"
            db = sqlite3.connect(live)
            db.execute("PRAGMA journal_mode=WAL")
            budget = census.Budget(1200, 4 * 1024**3)
            census.initialize(db, binding, campaign.read(scope["path"]), campaign.read(closure["path"]), None, budget)
            first = census.RowCensus(str(campaign.ROOT / "results"), db, budget)
            first.trace(rows[0]["turns"], rows[0]["input"], rows[0]["locator"], "fixture", "root")
            with db:
                for category in ("states", "features", "groups"):
                    census.union(db, category, getattr(first, category))
                db.execute("INSERT INTO completed VALUES (?,?,?,?)", (1, hashlib.sha256(lines[0].encode()).hexdigest(),
                    json.dumps(dict(first.counts)), json.dumps(first.ledger)))
            sealed = root / "sealed.sqlite3"
            shutil.copyfile(live, sealed)
            shutil.copyfile(Path(str(live) + "-wal"), Path(str(sealed) + "-wal"))
            db.close()
            parent_db = campaign.record(sealed)
            parent_wal = campaign.record(Path(str(sealed) + "-wal"))
            normalized = campaign.immutable(root / "normalized.json", dict(records=records, scope=scope, closure=closure,
                rows=3, producers=census.producers()))
            activation = dict(path="activation", sha256="fixture")
            current = campaign.immutable(root / "CURRENT.json", dict(owner_thread_id="fixture", ownership_status="active",
                                                                    focused_activation=activation))
            plan = dict(producer=campaign.record(continuation.__file__), output=str(root / "continued"),
                normalized_plan=normalized, parent_binding=binding_ref, parent_database=parent_db, parent_wal=parent_wal,
                campaign_current=current["path"], owner_thread_id="fixture", activation=activation,
                committed_rows=1, maximum_new_rows=1, cpu_seconds=1200, maximum_rss_bytes=4 * 1024**3)
            plan_ref = campaign.immutable(root / "PLAN.json", plan)
            with mock.patch.object(continuation.work, "checked", return_value={}):
                result = continuation.resume(plan_ref["path"])
            self.assertEqual(result["committed_rows"], 2)
            self.assertEqual(result["new_rows"], 1)
            self.assertFalse(result["complete"])
            self.assertEqual(campaign.record(sealed), parent_db)
            self.assertEqual(campaign.record(Path(str(sealed) + "-wal")), parent_wal)
            with sqlite3.connect(result["database"]["path"]) as resumed:
                self.assertEqual(resumed.execute("SELECT row_sha256 FROM completed WHERE number=1").fetchone()[0],
                                 hashlib.sha256(lines[0].encode()).hexdigest())

    def test_cached_and_retained_censuses_produce_identical_unions(self):
        legacy, cached = database(), database()
        physical, traces = {}, {}
        try:
            budget = census.Budget(1200, 4 * 1024**3)
            cases = [["1", "0"], ["1", "0"], ["2", "0"], ["7", "0"], ["1"]]
            for index, turns in enumerate(cases):
                for db, item in [(legacy, census.RowCensus(".", legacy, budget)),
                                 (cached, continuation.CachedRows(".", cached, budget, physical, traces))]:
                    item.trace(turns, dict(path="fixture", sha256="fixture"), "/input", "fixture", "root" + str(index))
                    insert(db, item)
            self.assertEqual(list(legacy.execute("SELECT * FROM keys ORDER BY category,value")),
                             list(cached.execute("SELECT * FROM keys ORDER BY category,value")))
        finally:
            legacy.close(); cached.close()

    def test_already_indexed_states_are_not_feature_reencoded(self):
        db = database()
        try:
            budget = census.Budget(1200, 4 * 1024**3)
            original = census.RowCensus(".", db, budget)
            original.trace(["1", "0"], dict(path="fixture", sha256="fixture"), "/input", "fixture")
            insert(db, original)
            expected = list(db.execute("SELECT * FROM keys ORDER BY category,value"))
            with mock.patch.object(census.banks, "feature_key", side_effect=AssertionError("old ancestry reencoded")):
                item = continuation.CachedRows(".", db, budget, {}, {})
                item.trace(["1", "0"], dict(path="fixture", sha256="fixture"), "/input", "fixture")
                insert(db, item)
            self.assertGreater(item.indexed_reuses, 0)
            self.assertEqual(list(db.execute("SELECT * FROM keys ORDER BY category,value")), expected)
        finally:
            db.close()

    def test_teacher_classification_keeps_observed_states_after_plain_cache(self):
        db = database()
        try:
            budget = census.Budget(1200, 4 * 1024**3)
            physical, traces = {}, {}
            plain = continuation.CachedRows(".", db, budget, physical, traces)
            plain.trace(["1", "0"], dict(path="fixture", sha256="fixture"), "/input", "fixture")
            insert(db, plain)
            teacher = continuation.CachedRows(".", db, budget, physical, traces, teacher_observation=True)
            teacher.trace(["1", "0"], dict(path="fixture", sha256="fixture"), "/input", "focused-teacher-parent")
            self.assertEqual(teacher.states, plain.states)
            self.assertEqual(teacher.features, plain.features)
        finally:
            db.close()


if __name__ == "__main__":
    unittest.main()
