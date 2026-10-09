import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools import rank_two_focused_campaign_v1 as campaign
from tools import rank_two_focused_data_v1 as pipeline
from tools import rank_two_focused_labels_v1 as execution


def parent(split="train", identifier="parent", root="root"):
    return dict(position_id=identifier, root_group_id=root, group_id=root,
                game_id="game", split=split, winner=1, mover=0, prefix="",
                source="fresh-fixture", edges=4, parent_active=[1, 2], canonical_state="state")


def native(position):
    source = {key: position[key] for key in ("position_id", "root_group_id", "group_id", "split", "winner", "source")}
    source.update(campaign_id=pipeline.CAMPAIGN_ID, prefix=[])
    return dict(source_bundle_body_sha256="bundle", teacher=dict(artifact_sha256=pipeline.data.TEACHER_SHA256),
        group=dict(source_binding=source, parent_mover=position["mover"], parent_identity=position["position_id"],
            work_budget=dict(max_tree_nodes=64000, max_time_ms=0), successors_exhaustive=False,
            root_value=.3, successors=[dict(active=[2, 3], teacher_value=-1., value_mover=1,
                terminal=True, proof=dict(solved=True, proven_winner=0))]))


class FocusedDataTests(unittest.TestCase):
    def test_native_proof_signs_and_incompleteness_survive_without_outcomes(self):
        position = parent()
        with mock.patch.object(pipeline.labels, "validate_complete_turn_action_group", side_effect=lambda value: value):
            result = pipeline.normalized_label(native(position), position, dict(bundle_sha256="bundle"))
        self.assertEqual(result["successors"][0]["teacher_value"], -1.)
        self.assertEqual(result["successors"][0]["value_mover"], 1)
        self.assertEqual(result["successors"][0]["proof"]["proven_winner"], 0)
        self.assertFalse(result["exhaustive"])
        self.assertNotIn("winner", result)
        self.assertNotIn("outcome", result)

    def test_rejects_wrong_parent_mover_recipe_and_teacher(self):
        position = parent()
        original = native(position)
        mutations = [lambda v: v["group"].update(parent_mover=1),
                     lambda v: v["group"]["source_binding"].update(position_id="other"),
                     lambda v: v.update(source_bundle_body_sha256="other"),
                     lambda v: v["teacher"].update(artifact_sha256="other"),
                     lambda v: v["group"]["work_budget"].update(max_tree_nodes=256000),
                     lambda v: v["group"]["source_binding"].update(winner=0)]
        with mock.patch.object(pipeline.labels, "validate_complete_turn_action_group", side_effect=lambda value: value):
            for mutate in mutations:
                value = copy.deepcopy(original)
                mutate(value)
                with self.assertRaises(ValueError):
                    pipeline.normalized_label(value, position, dict(bundle_sha256="bundle"))

    def test_payload_retains_exact_parent_provenance(self):
        position = parent()
        self.assertEqual(execution.payload(position), execution.payload(copy.deepcopy(position)))
        fields = execution.payload(position).decode().splitlines()[1].split("\t")
        self.assertEqual(fields[-3:], ["1", "0", ""])

    def test_reflected_parent_and_successor_overlap_is_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            positions = [parent("train", "a", "train-root"), parent("validation", "b", "validation-root")]
            plan = dict(positions=positions, bundle_sha256="bundle", generation_plan=dict(path="fixture", sha256="fixture"))
            campaign.immutable(root / "PLAN.json", plan)
            receipts = []
            groups = []
            state = pipeline.rules.ReplayState()
            pipeline.rules.apply_complete_turn(state, 0, "1")
            next_states = [copy.deepcopy(state), copy.deepcopy(state)]
            pipeline.rules.apply_complete_turn(next_states[0], 1, "0")
            pipeline.rules.apply_complete_turn(next_states[1], 1, "2")
            parent_active = list(pipeline.rules.encode_active(state))
            successor_active = [list(pipeline.rules.encode_active(value)) for value in next_states]
            for index, position in enumerate(positions):
                group = dict(group_id=position["position_id"], root_group_id=position["root_group_id"],
                    split=position["split"], edges=4, parent_active=parent_active, exhaustive=True,
                    successors=[dict(active=successor_active[0], teacher_value=.5, value_mover=0),
                                dict(active=successor_active[1], teacher_value=-.5, value_mover=0)])
                if index:
                    group["parent_active"] = list(pipeline.rules.reflect_active(group["parent_active"]))
                groups.append(group)
                campaign.immutable(root / f"native-{index}.jsonl", (json.dumps({"index": index}) + "\n").encode())
                receipt = dict(position_id=position["position_id"], plan_body_sha256="bundle", success=True,
                    unsupported=False, output=pipeline.data.record(root / f"native-{index}.jsonl"))
                receipts.append(campaign.immutable(root / f"receipt-{index}.json", receipt))
            campaign.immutable(root / "EXECUTION.json", dict(passed=True, plan=campaign.record(root / "PLAN.json"), parent_receipts=receipts))
            with mock.patch.object(pipeline, "normalized_label", side_effect=lambda value, *_: groups[value["index"]]):
                execution.import_corpus(root / "PLAN.json", root / "EXECUTION.json", root / "CORPUS.json")
            manifest = json.loads((root / "CORPUS.json").read_text())
            self.assertEqual(manifest["validation_overlap_groups_removed"], 1)
            self.assertEqual(len(manifest["group_artifacts"]), 1)
            self.assertFalse(manifest["production_training_admitted"])
            self.assertFalse(manifest["outcomes_as_targets"])

    def test_unsupported_late_parent_has_no_invented_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            position = parent()
            position["edges"] = 18
            (root / "output.partial.jsonl").write_text("")
            (root / "stderr.txt").write_text("jacek replay search teacher: parent: unsolved search teacher completed no root visits\n")
            result = execution.validate_result(dict(bundle_sha256="bundle"), position, root, 2)
            self.assertTrue(result["unsupported"])
            self.assertFalse(result["success"])
            self.assertEqual((root / "output.partial.jsonl").read_bytes(), b"")
            early_directory = root / "early"
            early_directory.mkdir()
            (early_directory / "output.partial.jsonl").write_text("")
            (early_directory / "stderr.txt").write_text((root / "stderr.txt").read_text())
            position["edges"] = 4
            result = execution.validate_result(dict(bundle_sha256="bundle"), position, early_directory, 2)
            self.assertFalse(result["unsupported"])

    def test_unknown_teacher_claim_is_not_replayed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "bound.txt").write_text("frozen fixture")
            reference = pipeline.data.record(root / "bound.txt")
            plan = dict(schema=pipeline.SCHEMA + ".teacher-plan", nodes=64000,
                outcomes_as_targets=False, teacher=reference, native_teacher=reference,
                games=reference, positions_file=reference, producer=campaign.record(root / "bound.txt"),
                validator=campaign.record(root / "bound.txt"), positions=[parent()], bundle_sha256="bundle")
            campaign.immutable(root / "PLAN.json", plan)
            campaign.immutable(root / "out/parents/00000/CLAIM.json", dict(spent=True))
            with mock.patch.object(pipeline, "check_current"), mock.patch.object(execution.subprocess, "Popen") as process:
                with self.assertRaisesRegex(RuntimeError, "unknown teacher parent claim"):
                    execution.run(root / "PLAN.json", root / "out", workers=1)
                process.assert_not_called()


if __name__ == "__main__":
    unittest.main()
