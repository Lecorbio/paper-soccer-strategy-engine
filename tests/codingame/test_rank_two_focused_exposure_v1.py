import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest import mock

from tools import rank_two_focused_campaign_v1 as campaign
from tools import rank_two_focused_exposure_v1 as exposure


class ExposureTests(unittest.TestCase):
    def test_proposal_replay_retains_rejected_primitive_tails(self):
        incomplete = False
        for seed in range(20):
            turns, reason = exposure.proposal_trace(seed, 12)
            self.assertLessEqual(sum(map(len, turns)), 12)
            if reason == "incomplete-turn-boundary":
                incomplete = True
                state = exposure.census.e.rules.ReplayState()
                for action in turns[:-1]:
                    exposure.census.e.rules.apply_complete_turn(state, state.to_move, action)
                mover = state.to_move
                for direction in turns[-1]:
                    exposure.census.e.rules.apply_primitive(state, direction)
                self.assertEqual(state.to_move, mover)
                self.assertIsNone(state.winner)
        self.assertTrue(incomplete)

    def test_live_pointer_and_all_primitive_alternatives_are_retained(self):
        stream = io.StringIO()
        normal = exposure.Normalizer(stream)
        reference = dict(path="fixture", sha256="0" * 64)
        normal.walk(dict(games=[dict(game_id=1, transcript="1/0")],
            proposals=[dict(prefix="1", choices=[dict(partial="", alternatives=[
                dict(direction=0, priority=[]), dict(direction=2, priority=[])])])]), reference, "")
        rows = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertTrue(any(row["locator"] == "/games/0/transcript" for row in rows))
        self.assertTrue(any(row["turns"] == ["1", "0"] and row["kind"] == "focused-primitive-proposal" for row in rows))
        self.assertTrue(any(row["turns"] == ["1", "2"] and row["kind"] == "focused-primitive-proposal" for row in rows))
        self.assertTrue(all(row["training_eligible"] is False for row in rows))

    def test_mover_relative_teacher_successor_is_rotated_once(self):
        stream = io.StringIO()
        normal = exposure.Normalizer(stream)
        value = dict(group=dict(source_binding=dict(prefix=[dict(action="1")], root_group_id="root"),
                                parent_mover=1, successors=[dict(transcript="0")]))
        with mock.patch.object(exposure.teacher, "validate_complete_turn_action_group", return_value=value):
            normal.walk(value, dict(path="fixture", sha256="0" * 64), "")
        rows = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertEqual(rows[-1]["turns"], ["1", "4"])

    def test_seed_database_is_immutable_and_teacher_namespace_is_inherited(self):
        # The fixture is under the explicit worktree so the production input
        # boundary checks remain active. It contains no protected observations.
        temporary_parent = campaign.ROOT / "results"
        with tempfile.TemporaryDirectory(prefix="focused-exposure-unit-", dir=temporary_parent) as tmp:
            root = Path(tmp)
            seed = root / "seed"
            seed.mkdir()
            activation = dict(path="unit-activation", sha256="unit")
            campaign.immutable(root / "CURRENT.json", dict(ownership_status="active", owner_thread_id="fixture",
                                                        focused_activation=activation))
            cutoff = campaign.immutable(seed / "CUTOFF.json", dict(cutoff_utc=campaign.now().isoformat()))
            scope = campaign.immutable(seed / "scope.json", dict(schema=exposure.census.v2.SCOPE_SCHEMA,
                campaign_root=str(campaign.ROOT / "results"), cutoff=cutoff, inherit=[], historical_protected_reads=False))
            records = campaign.immutable(seed / "records.jsonl", b"")
            closure = campaign.immutable(seed / "closure.json", dict(scope=scope, cutoff=cutoff, records=records,
                rows=0, inputs=[], all_payload_inputs_hash_verified_before_census=True, skipped_or_unreturned=[]))
            binding = dict(schema=exposure.census.SCHEMA, scope=scope, closure=closure, records=records,
                           sources=exposure.census.producers(), inherited_inputs=[])
            binding_ref = campaign.immutable(seed / "binding.json", binding)
            database = seed / "checkpoint.sqlite3"
            db = sqlite3.connect(database)
            budget = exposure.census.Budget(1200, 4 * 1024**3)
            exposure.census.initialize(db, binding, campaign.read(scope["path"]), campaign.read(closure["path"]), None, budget)
            with db:
                exposure.census.union(db, "states", ["0" * 64])
                exposure.census.union(db, "features", ["1" * 64])
                exposure.census.union(db, "groups", ["inherited-root"])
                exposure.census.union(db, "teacher:retained_fixture", ["2" * 64])
            receipt = exposure.census.export_inventory(db, seed / "inventory-001.json", binding,
                campaign.read(scope["path"]), campaign.read(closure["path"]), "complete", None, budget)
            receipt.update(checkpoint=str(database), historical_protected_reads=False)
            db.close()
            receipt_ref = campaign.immutable(seed / "receipt-001.json", receipt)
            prior = dict(schema="papersoccer.rank-two.live-prior-exposure.v1", historical_protected_reads=False,
                inputs=dict(inventory=receipt["inventory"], database=campaign.record(database), receipt=receipt_ref,
                            binding=binding_ref, closure=closure, scope=scope))
            prior_ref = campaign.immutable(seed / "bank-prior.json", prior)
            campaign.immutable(seed / "ready.json", dict(passed=True, prior=prior_ref))
            original = campaign.record(database)
            sample = campaign.immutable(root / "new-exposure.json", dict(games=[dict(game_id=1, transcript="1/0")]))
            exposure.prepare(root, seed / "ready.json", [sample], root / "delta")
            ready = exposure.extend(root / "delta/PLAN.json")
            self.assertTrue(ready["passed"])
            self.assertEqual(campaign.record(database), original)
            self.assertFalse(ready["historical_ancestry_rescan"])
            new_prior = campaign.read(ready["prior"]["path"])
            with exposure.prior_validator.prior_census(new_prior) as indexed:
                self.assertIsNotNone(indexed.execute("SELECT 1 FROM keys WHERE category=? AND value=?",
                    ("teacher:retained_fixture", "2" * 64)).fetchone())
                self.assertGreater(indexed.execute("SELECT count(*) FROM keys WHERE category='states'").fetchone()[0], 1)


if __name__ == "__main__":
    unittest.main()
