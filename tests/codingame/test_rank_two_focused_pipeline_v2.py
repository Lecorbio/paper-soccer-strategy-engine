import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from tools import rank_two_focused_campaign_v1 as campaign
from tools import rank_two_focused_campaign_v2 as controller
from tools import rank_two_focused_data_v3 as data
from tools import rank_two_focused_exposure_v3 as exposure
from tools import rank_two_focused_training_v4 as training
from tools import rank_two_focused_work_v1 as work
from tests.codingame.test_rank_two_focused_campaign_v2 import state, usage


class PipelineRegressionTests(unittest.TestCase):
    def test_teacher_exposure_chain_parses_native_output(self):
        with tempfile.TemporaryDirectory(prefix="focused-pipeline-unit-", dir=campaign.ROOT / "results") as tmp:
            root = Path(tmp)
            native = dict(group=dict(source_binding=dict(prefix=[dict(action="1")], root_group_id="root"),
                                    parent_mover=1, successors=[dict(transcript="6")]))
            output = campaign.immutable(root / "native.jsonl", (json.dumps(native) + "\n").encode())
            stderr = campaign.immutable(root / "stderr.txt", b"")
            parent = campaign.immutable(root / "parent.json", dict(output=output, stderr=stderr))
            execution = campaign.immutable(root / "execution.json", dict(parent_receipts=[parent]))
            marker = campaign.immutable(root / "exposure.json", dict(execution=execution))
            stream = io.StringIO()
            normal = exposure.Normalizer(stream)
            with mock.patch.object(exposure.teacher, "validate_complete_turn_action_group", return_value=native):
                normal.bind(marker, parse=True)
            rows = [json.loads(line) for line in stream.getvalue().splitlines()]
            self.assertTrue(any(row["kind"] == "focused-teacher-successor" and row["turns"] == ["1", "2"] for row in rows))
            self.assertIn(output["path"], normal.parsed)

    def test_completed_generation_marker_includes_played_descendants(self):
        with tempfile.TemporaryDirectory(prefix="focused-pipeline-unit-", dir=campaign.ROOT / "results") as tmp:
            root = Path(tmp)
            plan = dict(rows=[dict(root_group_id="root", colors=[0, 1])])
            plan_ref = campaign.immutable(root / "PLAN.json", plan)
            campaign.immutable(root / "RESULT.json", dict(passed=True))
            campaign.immutable(root / "envelopes.jsonl", b"")
            for color in (0, 1):
                campaign.immutable(root / "games" / ("root-p" + str(color)) / "RESULT.json",
                                   dict(game=dict(transcript="1/0")))
            marker = data.completed_exposure(plan_ref["path"], root)
            stream = io.StringIO()
            exposure.Normalizer(stream).bind(marker, parse=True)
            rows = [json.loads(line) for line in stream.getvalue().splitlines()]
            self.assertTrue(any(row["turns"] == ["1", "0"] for row in rows))
            self.assertEqual(campaign.read(marker["path"])["games_retained"], 2)

    def test_input_bound_supports_declared_parent_receipt_triples(self):
        with tempfile.TemporaryDirectory(prefix="focused-pipeline-unit-", dir=campaign.ROOT / "results") as tmp:
            root = Path(tmp)
            reference = campaign.immutable(root / "new.json", dict(metadata_only=True))
            normal = exposure.Normalizer(io.StringIO())
            normal.inputs = {"retained-" + str(i): reference for i in range(3 * 6144 + 512)}
            normal.bind(reference)
            self.assertEqual(len(normal.inputs), 3 * 6144 + 513)

    def test_older_corpus_cannot_be_paired_with_fresh_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fresh = campaign.immutable(root / "fresh.json", dict(seed=2026100710))
            old = campaign.immutable(root / "old.json", dict(seed=2026091604))
            corpus = SimpleNamespace(manifest=dict(generation_plan=old))
            with self.assertRaisesRegex(ValueError, "declared fresh generation"):
                training.validate_corpus_binding(dict(generation_plan=fresh), corpus)

    def test_entry_uses_actual_usage_instead_of_frozen_plan_percentage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            body = state()
            body["control_guard"] = campaign.record(controller.__file__)
            body["closures"] = campaign.immutable(root / "CLOSURES.json", {})
            body["focused_usage_observation"] = campaign.immutable(root / "USAGE.json", usage(70))
            campaign.immutable(root / "CURRENT.json", body)
            plan = dict(campaign_current=str(root / "CURRENT.json"), owner_thread_id="owner",
                        activation=body["focused_activation"], used_percent=1)
            with self.assertRaisesRegex(PermissionError, "weekly reserve"):
                work.checked(plan, "enqueue", launch=True)

    def test_current_deadline_and_incident_apply_at_work_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            body = state()
            body["control_guard"] = campaign.record(controller.__file__)
            body["closures"] = campaign.immutable(root / "CLOSURES.json", {})
            body["focused_usage_observation"] = campaign.immutable(root / "USAGE.json", usage(7))
            body["deadline_utc"] = "2026-10-01T00:00:00+00:00"
            campaign.immutable(root / "CURRENT.json", body)
            plan = dict(campaign_current=str(root / "CURRENT.json"), owner_thread_id="owner", activation=body["focused_activation"])
            with self.assertRaisesRegex(PermissionError, "deadline"):
                work.checked(plan, "enqueue")
            body["deadline_utc"] = "2026-10-21T21:28:13+00:00"
            body["incident"] = dict(unresolved=True)
            campaign.atomic(root / "CURRENT.json", body)
            with self.assertRaisesRegex(PermissionError, "incident"):
                work.checked(plan, "enqueue")

    def test_generation_and_label_activation_change_cannot_continue(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            body = state()
            campaign.immutable(root / "CURRENT.json", body)
            plan = dict(campaign_current=str(root / "CURRENT.json"), owner_thread_id="owner",
                        activation=dict(path="other-activation", sha256="other"))
            for action in ("generate", "label"):
                with self.assertRaisesRegex(PermissionError, "activation changed"):
                    work.checked(plan, action)


if __name__ == "__main__":
    unittest.main()
