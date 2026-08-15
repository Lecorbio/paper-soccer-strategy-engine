import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
RECORDER_PATH = ROOT / "tools/record_rank4_jacek_hybrid_sole_legal_edge_clock.py"
sys.path.insert(0, str(ROOT / "tools"))
SPEC = importlib.util.spec_from_file_location("sole_edge_recorder", RECORDER_PATH)
assert SPEC is not None and SPEC.loader is not None
recorder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recorder)

CANDIDATE_GATE_COMMIT = "4c16b7d3a9494f179189000b46dc5724da92ab89"


def identity_from_bytes(path: Path, data: bytes) -> dict[str, object]:
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "ascii": all(byte < 128 for byte in data),
    }


class SoleLegalEdgeClockRecorderTest(unittest.TestCase):
    def test_schema_command_and_plan_are_frozen_development_only(self):
        self.assertEqual(
            recorder.SCHEMA,
            "rank4-jacek-hybrid-sole-legal-edge-clock-v1",
        )
        command = recorder.command_for_gate()
        joined = " ".join(command)
        self.assertIn("development_d20.tsv", joined)
        self.assertNotIn("validation", joined.lower())
        self.assertNotIn("final", joined.lower())
        self.assertEqual(command.count("7"), 2)
        self.assertEqual(
            recorder.expected_configuration()["reference_engine"], "rank4"
        )
        bindings = recorder.validate_plan_and_lineage()
        thresholds = bindings["plan"]["thresholds"]
        self.assertEqual(thresholds["candidate_wins_min"], 38)
        self.assertEqual(thresholds["reference_first_ms_p99_lt"], 900)
        self.assertEqual(thresholds["reference_later_ms_max_lt"], 198)
        self.assertEqual(
            bindings["plan"]["reference"]["function"], "choose_prefastpath"
        )
        self.assertEqual(
            bindings["rollback"]["verification"]["fixed_work_parity"]["status"],
            "passed",
        )

    def test_exact_candidate_reference_lineage_and_prototype_identities(self):
        candidate_paths = {
            recorder.CANDIDATE_BOT,
            recorder.CANDIDATE_SOURCE,
            recorder.CANDIDATE_TEST,
        }
        paths = (
            *candidate_paths,
            recorder.CONTROL_BOT,
            recorder.CONTROL_SOURCE,
            recorder.BANK,
            recorder.PLAN,
            recorder.ROLLBACK,
            recorder.DECISION,
            recorder.CONTROL_MANIFEST,
            *recorder.PROTOTYPE_FILES,
        )
        identities = {}
        for path in paths:
            relative = str(path.relative_to(ROOT))
            data = (
                recorder.base.git_blob(CANDIDATE_GATE_COMMIT, relative)
                if path in candidate_paths else path.read_bytes()
            )
            identities[relative] = identity_from_bytes(path, data)
        recorder.validate_exact_file_identities(identities)
        self.assertEqual(
            identities[str(recorder.CANDIDATE_BOT.relative_to(ROOT))]["sha256"],
            recorder.EXPECTED_CANDIDATE_BOT_SHA256,
        )
        self.assertEqual(
            identities[str(recorder.CANDIDATE_SOURCE.relative_to(ROOT))]["bytes"],
            94_527,
        )
        bindings = recorder.validate_plan_and_lineage()
        selected = {
            item["role"]: item
            for item in bindings["rollback"]["production_artifacts"]
        }
        for path, role in (
            (recorder.CANDIDATE_BOT, "engine-source"),
            (recorder.CANDIDATE_SOURCE, "upload-source"),
            (recorder.CANDIDATE_TEST, "source-contract-test"),
        ):
            self.assertEqual(
                recorder.base.common.file_identity(path)["sha256"],
                selected[role]["sha256"],
            )

    def test_one_content_addressed_attempt_prevents_retry(self):
        head = "a" * 40
        payload = {"schema": recorder.SCHEMA, "attempt_id": recorder.attempt_id(head)}
        raw = recorder.canonical_json(payload)
        digest = hashlib.sha256(raw).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / f"{digest}.json").write_bytes(raw)
            (output / "not-content-addressed.json").write_bytes(raw)
            self.assertEqual(recorder.matching_attempts(head, output), [output / f"{digest}.json"])
            payload["schema"] = "different-ablation"
            other = recorder.canonical_json(payload)
            other_digest = hashlib.sha256(other).hexdigest()
            (output / f"{other_digest}.json").write_bytes(other)
            self.assertEqual(recorder.matching_attempts(head, output), [output / f"{digest}.json"])

    def test_attempt_key_binds_exact_head_and_lineage(self):
        head = "b" * 40
        key = recorder.attempt_key(head)
        self.assertEqual(key["head"], head)
        self.assertEqual(key["plan_sha256"], recorder.EXPECTED_PLAN_SHA256)
        self.assertEqual(key["rollback_sha256"], recorder.EXPECTED_ROLLBACK_SHA256)
        self.assertEqual(key["decision_sha256"], recorder.EXPECTED_DECISION_SHA256)
        self.assertEqual(key["candidate_test_sha256"], recorder.EXPECTED_CANDIDATE_TEST_SHA256)

    def test_process_preflight_allows_lineage_and_rejects_competitor(self):
        clean = recorder.process_preflight_from_table([
            {"pid": 1, "ppid": 0, "command": "/sbin/launchd"},
            {"pid": 10, "ppid": 1, "command": "zsh record_rank4_jacek_hybrid wrapper"},
            {"pid": 11, "ppid": 10, "command": "python sole_legal_edge_clock"},
            {"pid": 12, "ppid": 1, "command": "unrelated editor"},
        ], 11)
        self.assertTrue(clean["clean"])
        conflict = recorder.process_preflight_from_table([
            {"pid": 1, "ppid": 0, "command": "/sbin/launchd"},
            {"pid": 11, "ppid": 1, "command": "python record_rank4_jacek_hybrid_sole"},
            {"pid": 99, "ppid": 1, "command": "papersoccer_codingame_rank_4 worker"},
        ], 11)
        self.assertFalse(conflict["clean"])
        self.assertEqual([item["pid"] for item in conflict["conflicts"]], [99])

    def test_plan_is_canonical_and_content_addressed_bindings_are_exact(self):
        raw = recorder.PLAN.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), recorder.EXPECTED_PLAN_SHA256)
        self.assertEqual(recorder.canonical_json(json.loads(raw)), raw)
        for path in (recorder.ROLLBACK, recorder.DECISION):
            self.assertEqual(path.stem, hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
