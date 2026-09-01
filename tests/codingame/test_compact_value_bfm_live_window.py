from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULE = (
    ROOT / "submissions/codingame/bots/compact_value_bfm/live_window.py"
)
SPEC = importlib.util.spec_from_file_location(
    "compact_value_bfm_live_window", MODULE
)
assert SPEC is not None and SPEC.loader is not None
live = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = live
SPEC.loader.exec_module(live)
qualification = live.qualification_module()


COMMIT = "1" * 40
AGENT_ID = 701
SUBMISSION_ID = 801


def battle(game_id: int, *, submission: int = SUBMISSION_ID, done: bool = True):
    return {
        "gameId": game_id,
        "done": done,
        "players": [
            {
                "playerAgentId": AGENT_ID,
                "submissionId": submission,
                "position": 0,
            },
            {
                "playerAgentId": 10_000 + game_id,
                "submissionId": 20_000 + game_id,
                "position": 1,
            },
        ],
    }


def battles(count: int, *, submission: int = SUBMISSION_ID):
    return [battle(index + 1, submission=submission) for index in range(count)]


def generic_record(game_id: int, *, focus="ok", opponent="ok", result="win"):
    return {
        "schema": live.GENERIC_GAME_SCHEMA,
        "status": "accepted",
        "game_id": game_id,
        "source_sha256": "2" * 64,
        "focus": {
            "agent_id": AGENT_ID,
            "submission_id": SUBMISSION_ID,
            "result": result,
        },
        "opponent": {"agent_id": 10_000 + game_id},
        "operational": {
            "classification": (
                "clean" if focus == opponent == "ok" else "operationally-terminated"
            ),
            "focus_status": focus,
            "opponent_status": opponent,
        },
    }


class Fixture:
    def __init__(self, root: pathlib.Path):
        self.root = root
        self.source = root / "submission.cpp"
        self.source.write_text("int main(){}\n", encoding="ascii")
        source = self.source.read_bytes()
        self.source_sha = hashlib.sha256(source).hexdigest()
        self.authorization = root / "one-upload-authorization.json"
        qualification.write_sealed(self.authorization, {
            "schema": qualification.UPLOAD_AUTH_SCHEMA,
            "namespace": live.NAMESPACE,
            "uploads_authorized": 1,
            "rank4_replacement_authorized": False,
            "candidate_commit": COMMIT,
            "candidate": {
                "path": str(self.source.resolve()),
                "bytes": len(source),
                "sha256": self.source_sha,
                "ascii": True,
            },
            "binding": {"path": "/binding", "sha256": "3" * 64},
            "aggregate": {"path": "/aggregate", "sha256": "4" * 64},
            "ci": {"conclusion": "success"},
            "upload_ledger_root": str(root.resolve()),
        })
        self.attestation = root / "05-submission-attested.json"
        qualification.write_sealed(self.attestation, {
            "schema": qualification.UPLOAD_EVENT_SCHEMA,
            "namespace": live.NAMESPACE,
            "status": "submission-attested",
            "submitted_at_utc": "2026-08-31T12:00:00Z",
            "authorization": qualification.artifact_reference(
                self.authorization, qualification.UPLOAD_AUTH_SCHEMA
            ),
            "candidate_commit": COMMIT,
            "source_sha256": self.source_sha,
            "source_bytes": len(source),
            "agent_id": AGENT_ID,
            "submission_id": SUBMISSION_ID,
            "ambiguity_resolution": None,
            "submit_clicks": 1,
        })
        self.registry = root / "exclusions.json"
        registry = {
            "schema": live.EXCLUSION_SCHEMA,
            "selection": "pre-upload ID-only fixture",
            "sources": [],
            "records": [
                {
                    "game_id": 999,
                    "categories": ["known_local_evidence"],
                    "sources": ["fixture"],
                }
            ],
        }
        self.registry.write_bytes(live.canonical_json_bytes(registry))
        self.exclusion_binding = root / "exclusion-binding.json"
        live.freeze_exclusion_binding(
            self.exclusion_binding,
            registry_path=self.registry,
            frozen_at_utc="2026-08-31T11:00:00Z",
        )

    def identity(self):
        return live.load_live_identity(
            self.attestation, self.exclusion_binding
        )[0]


class BindingTest(unittest.TestCase):
    def test_preupload_id_only_registry_and_one_upload_identity_are_bound(self):
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(pathlib.Path(raw))
            identity, exclusion, registry = live.load_live_identity(
                fixture.attestation, fixture.exclusion_binding
            )
            self.assertEqual(identity.agent_id, AGENT_ID)
            self.assertEqual(identity.submission_id, SUBMISSION_ID)
            self.assertEqual(identity.source_sha256, fixture.source_sha)
            self.assertEqual(identity.repository_commit, COMMIT)
            self.assertEqual(exclusion["registry"]["path"], str(registry))
            self.assertFalse(exclusion["replay_payloads_read"])

    def test_registry_frozen_after_upload_and_replay_fields_are_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            fixture = Fixture(root)
            late = root / "late-binding.json"
            live.freeze_exclusion_binding(
                late,
                registry_path=fixture.registry,
                frozen_at_utc="2026-08-31T13:00:00Z",
            )
            with self.assertRaisesRegex(live.LiveWindowError, "before upload"):
                live.load_live_identity(fixture.attestation, late)
            bad = root / "bad-registry.json"
            bad.write_bytes(live.canonical_json_bytes({
                "schema": live.EXCLUSION_SCHEMA,
                "selection": "bad",
                "sources": [],
                "records": [],
                "frames": [],
            }))
            with self.assertRaisesRegex(live.LiveWindowError, "forbidden"):
                live.freeze_exclusion_binding(
                    root / "bad-binding.json",
                    registry_path=bad,
                    frozen_at_utc="2026-08-31T11:00:00Z",
                )

    def test_tampered_source_or_second_click_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            fixture = Fixture(root)
            attestation = qualification.load_sealed(
                fixture.attestation, qualification.UPLOAD_EVENT_SCHEMA
            )
            attestation.pop("body_sha256")
            attestation["submit_clicks"] = 2
            second = root / "second-click-attestation.json"
            qualification.write_sealed(second, attestation)
            with self.assertRaisesRegex(live.LiveWindowError, "identity"):
                live.load_live_identity(second, fixture.exclusion_binding)
            fixture.source.write_text("changed\n", encoding="ascii")
            with self.assertRaisesRegex(live.LiveWindowError, "source bytes"):
                live.load_live_identity(fixture.attestation, fixture.exclusion_binding)


class MetadataAndOperationalTest(unittest.TestCase):
    def test_89_waits_90_ready_91_rejected_and_other_submission_ignored(self):
        report = live.classify_matching_window(
            battles(89), agent_id=AGENT_ID, submission_id=SUBMISSION_ID
        )
        self.assertEqual(report["status"], "waiting")
        self.assertEqual(report["complete_games"], 89)
        self.assertTrue(live.classify_matching_window(
            battles(90), agent_id=AGENT_ID, submission_id=SUBMISSION_ID
        )["ready"])
        with self.assertRaisesRegex(live.LiveWindowError, "exceeds exactly 90"):
            live.classify_matching_window(
                battles(91), agent_id=AGENT_ID, submission_id=SUBMISSION_ID
            )
        rows = battles(90) + [battle(50_000, submission=999)]
        self.assertTrue(live.classify_matching_window(
            rows, agent_id=AGENT_ID, submission_id=SUBMISSION_ID
        )["ready"])

    def test_focus_and_opponent_failures_are_scoped_separately(self):
        focus = {
            "gameId": 1,
            "agents": [
                {"agentId": AGENT_ID, "index": 0},
                {"agentId": 999, "index": 1},
            ],
            "frames": [{"agentId": 0, "stderr": "timed out"}],
        }
        result = live.classify_operational_detail(
            focus, game_id=1, focus_agent_id=AGENT_ID
        )
        self.assertTrue(result["focus_failure"])
        self.assertFalse(result["opponent_failure"])
        opponent = {
            **focus,
            "gameId": 2,
            "frames": [{"agentId": 1, "stderr": "runtime error"}],
        }
        result = live.classify_operational_detail(
            opponent, game_id=2, focus_agent_id=AGENT_ID
        )
        self.assertFalse(result["focus_failure"])
        self.assertTrue(result["opponent_failure"])

    def test_opponent_failure_is_never_a_strength_win(self):
        records = [generic_record(index) for index in range(1, 91)]
        records[0] = generic_record(1, opponent="timeout", result="win")
        monitor = {
            index: {
                "classification": {
                    "focus_failure": False,
                    "opponent_failure": index == 1,
                    "focus_status": "ok",
                    "opponent_status": "timeout" if index == 1 else "ok",
                }
            }
            for index in range(1, 91)
        }
        summary = live.summarize_window(records, monitor)
        self.assertEqual(summary["status"], "complete-accepted-diagnostic")
        self.assertEqual(summary["opponent_operational_failure_games"], 1)
        self.assertEqual(summary["clean_strength_wins"], 89)
        self.assertEqual(
            summary["opponent_failure_games_counted_as_strength_wins"], 0
        )


class WatchTest(unittest.TestCase):
    def verified_manifest(self, fixture, records):
        return {
            "records": records,
            "manifest": {"fixture": True},
            "manifest_path": None,
            "manifest_sha256": "5" * 64,
        }

    def test_focus_failure_does_not_stop_collection_before_game_90(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            fixture = Fixture(root)
            metadata = iter((battles(89), battles(90)))
            detail_calls = []
            collector_calls = []

            def classify(_detail, *, game_id, focus_agent_id):
                self.assertEqual(focus_agent_id, AGENT_ID)
                return {
                    "game_id": game_id,
                    "focus_status": "timeout" if game_id == 1 else "ok",
                    "opponent_status": "ok",
                    "focus_failure": game_id == 1,
                    "opponent_failure": False,
                }

            def collect(**kwargs):
                collector_calls.append(kwargs)
                return {"opaque": True}

            records = [generic_record(index) for index in range(1, 91)]
            result = live.watch_window(
                submission_attestation_path=fixture.attestation,
                exclusion_binding_path=fixture.exclusion_binding,
                data_root=root / "archive",
                poll_seconds=0,
                timeout_seconds=100,
                fetch_battles=lambda: next(metadata),
                fetch_detail=lambda game_id: (
                    detail_calls.append(game_id) or {"game": game_id}
                ),
                detail_classifier=classify,
                collector=collect,
                collector_verifier=lambda *_args, **_kwargs: self.verified_manifest(
                    fixture, records
                ),
                clock=lambda: "2026-08-31T14:00:00Z",
                monotonic=iter((0.0, 1.0)).__next__,
                sleeper=lambda _seconds: None,
            )
            self.assertEqual(len(detail_calls), 90)
            self.assertEqual(len(collector_calls), 1)
            self.assertEqual(
                result["status"], "complete-rejected-focus-operational-failure"
            )
            self.assertFalse(result["training_eligible"])
            self.assertFalse(result["rollback_authorized"])
            self.assertFalse(result["second_upload_authorized"])
            live.verify_window_reference(
                root / "archive/live-window.reference.json",
                data_root=root / "archive",
            )

    def test_89_times_out_without_collector_and_91_rejects_before_detail(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            fixture = Fixture(root)
            collector_calls = []
            waiting = live.watch_window(
                submission_attestation_path=fixture.attestation,
                exclusion_binding_path=fixture.exclusion_binding,
                data_root=root / "waiting",
                poll_seconds=0,
                timeout_seconds=0,
                fetch_battles=lambda: battles(89),
                fetch_detail=lambda game_id: {"game": game_id},
                detail_classifier=lambda _detail, *, game_id, focus_agent_id: {
                    "game_id": game_id,
                    "focus_status": "ok",
                    "opponent_status": "ok",
                    "focus_failure": False,
                    "opponent_failure": False,
                },
                collector=lambda **kwargs: collector_calls.append(kwargs),
                clock=lambda: "2026-08-31T14:00:00Z",
                monotonic=iter((0.0, 0.0)).__next__,
                sleeper=lambda _seconds: None,
            )
            self.assertEqual(waiting["status"], "waiting")
            self.assertEqual(waiting["complete_games"], 89)
            self.assertFalse(waiting["collector_invoked"])
            self.assertEqual(collector_calls, [])
            detail_calls = []
            with self.assertRaisesRegex(live.LiveWindowError, "exceeds exactly 90"):
                live.watch_window(
                    submission_attestation_path=fixture.attestation,
                    exclusion_binding_path=fixture.exclusion_binding,
                    data_root=root / "overfull",
                    timeout_seconds=0,
                    fetch_battles=lambda: battles(91),
                    fetch_detail=lambda game_id: detail_calls.append(game_id),
                    collector=lambda **kwargs: collector_calls.append(kwargs),
                    clock=lambda: "2026-08-31T14:00:00Z",
                    monotonic=lambda: 0.0,
                )
            self.assertEqual(detail_calls, [])

    def test_append_only_reference_rejects_conflicting_rewrite(self):
        with tempfile.TemporaryDirectory() as raw:
            path = pathlib.Path(raw) / "sealed.json"
            live.write_sealed(path, {
                "schema": live.WAIT_SNAPSHOT_SCHEMA,
                "value": 1,
            })
            with self.assertRaisesRegex(live.LiveWindowError, "collision"):
                live.write_sealed(path, {
                    "schema": live.WAIT_SNAPSHOT_SCHEMA,
                    "value": 2,
                })


class GenericManifestTest(unittest.TestCase):
    def test_collector_identity_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(pathlib.Path(raw))
            identity = fixture.identity()
            records = [generic_record(index) for index in range(1, 91)]
            manifest = {
                "schema": live.GENERIC_BATCH_SCHEMA,
                "coverage": {
                    "expected_games": 90,
                    "battle_window_games": 90,
                    "accepted_games": 90,
                    "full_window_accounted": True,
                },
                "binding": {
                    "schema": live.GENERIC_BINDING_SCHEMA,
                    "agent_id": AGENT_ID,
                    "asserted_submission_id": 999,
                    "repository_commit": COMMIT,
                    "source": {"sha256": fixture.source_sha},
                },
                "exclusion_registry": {
                    "sha256": hashlib.sha256(
                        fixture.registry.read_bytes()
                    ).hexdigest()
                },
                "games": [{"record": record} for record in records],
            }
            with self.assertRaisesRegex(live.LiveWindowError, "identity"):
                live.verify_generic_result(
                    {
                        "manifest": manifest,
                        "manifest_sha256": live.sha256_bytes(
                            live.canonical_json_bytes(manifest)
                        ),
                    },
                    identity=identity,
                    registry_sha256=hashlib.sha256(
                        fixture.registry.read_bytes()
                    ).hexdigest(),
                    expected_game_ids=list(range(1, 91)),
                )


if __name__ == "__main__":
    unittest.main()
