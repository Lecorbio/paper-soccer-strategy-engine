import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULE_PATH = (
    ROOT
    / "submissions"
    / "codingame"
    / "bots"
    / "jacek_arena_bfm"
    / "campaign_provenance.py"
)
SPEC = importlib.util.spec_from_file_location("jacek_arena_campaign_provenance", MODULE_PATH)
provenance = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = provenance
SPEC.loader.exec_module(provenance)


def write_canonical(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(provenance.canonical_json_bytes(value))
    return provenance.sha256_file(path)


def player(agent_id, position, submission_id):
    return {
        "nickname": f"agent-{agent_id}",
        "playerAgentId": agent_id,
        "position": position,
        "publicHandle": f"handle-{agent_id}",
        "submissionId": submission_id,
        "testSessionHandle": f"session-{agent_id}",
        "userId": agent_id,
    }


class CampaignProvenanceTest(unittest.TestCase):
    def make_base_registry(self, root):
        source = {
            "category": "protected_evaluation",
            "game_id_count": 1,
            "path": "sealed-id-only.json",
            "sha256": "1" * 64,
        }
        payload = {
            "records": [
                {
                    "categories": ["protected_evaluation"],
                    "game_id": 11,
                    "sources": ["sealed-id-only.json"],
                }
            ],
            "schema": provenance.EXCLUSION_SCHEMA,
            "selection": "unit-test ID fields only",
            "sources": [source],
        }
        content = provenance.canonical_json_bytes(payload)
        digest = provenance.sha256_bytes(content)
        path = root / f"{digest}.json"
        path.write_bytes(content)
        return path, digest

    def test_exclusion_builder_merges_only_ids_and_is_content_addressed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            base, base_hash = self.make_base_registry(root)
            inventory = root / "protected-ids.txt"
            inventory.write_text("12\n13\n", encoding="ascii")
            battles = root / "battles.json"
            battles.write_text(
                json.dumps(
                    [
                        {
                            "done": True,
                            "gameId": 14,
                            "players": [player(101, 0, 201), player(102, 1, 202)],
                        }
                    ]
                )
            )
            digest, path, payload = provenance.build_exclusion_registry(
                base_registry_path=base,
                base_registry_sha256=base_hash,
                protected_inventory_paths=[inventory],
                battle_snapshot_paths=[battles],
                t0_utc=provenance.DEFAULT_T0_UTC,
                output_directory=root / "out",
                repository=root,
            )
            self.assertEqual(path.name, f"{digest}.json")
            self.assertEqual(path.read_bytes(), provenance.canonical_json_bytes(payload))
            self.assertEqual([record["game_id"] for record in payload["records"]], [11, 12, 13, 14])
            categories = {record["game_id"]: record["categories"] for record in payload["records"]}
            self.assertEqual(categories[14], ["protected_pre_t0_battle_metadata"])
            self.assertNotIn("players", path.read_text())

    def test_battle_snapshot_rejects_replay_bearing_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "bad.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "done": True,
                            "frames": [{"stdout": "0"}],
                            "gameId": 14,
                            "players": [player(101, 0, 201), player(102, 1, 202)],
                        }
                    ]
                )
            )
            with self.assertRaisesRegex(provenance.ProvenanceError, "replay fields"):
                provenance.load_battle_metadata_snapshot(path)

    def make_plan(self, root):
        return provenance.create_window_plan(
            t0_utc="2026-08-13T10:12:52Z",
            planned_at_utc="2026-08-13T10:13:00Z",
            collection_windows=8,
            output_directory=root / "plans",
        )

    def test_window_plan_roles_are_blind_and_immutable(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, _, plan = self.make_plan(pathlib.Path(temporary))
            self.assertEqual(
                [window["role"] for window in plan["windows"]],
                [
                    "training",
                    "training",
                    "training",
                    "arena-validation",
                    "training",
                    "training",
                    "training",
                    "arena-validation",
                    "final-holdout",
                    "rollback-accounting",
                ],
            )
            forged = json.loads(json.dumps(plan))
            forged["windows"][3]["role"] = "training"
            with self.assertRaisesRegex(provenance.ProvenanceError, "role assignment"):
                provenance.validate_window_plan(forged)

    def make_attestation(self, root, plan_hash, plan_path, *, initialize_sources=True):
        source = root / "candidate.cpp"
        copied = root / "editor-copyback.cpp"
        if initialize_sources:
            source.write_bytes(b"int main(){return 0;}\n")
            copied.write_bytes(source.read_bytes())
        return provenance.create_editor_attestation(
            plan_path=plan_path,
            expected_plan_sha256=plan_hash,
            window_id="collection-001",
            source_path=source,
            copied_back_path=copied,
            repository_commit="a" * 40,
            agent_id=101,
            submission_id=201,
            uploaded_at_utc="2026-08-13T10:13:10Z",
            checked_at_utc="2026-08-13T10:13:09Z",
            preflight={key: True for key in provenance._PREFLIGHT_KEYS},
            play_stdout_legal=True,
            play_telemetry_ok=True,
            output_directory=root / "campaign",
            repository=root,
        )

    def test_editor_attestation_requires_exact_ascii_copyback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            plan_hash, plan_path, _ = self.make_plan(root)
            _, _, attestation = self.make_attestation(root, plan_hash, plan_path)
            self.assertEqual(attestation["source"]["sha256"], attestation["editor_copyback"]["sha256"])
            self.assertEqual(attestation["editor_copyback"]["status"], "editor-attested-not-api-readable")
            (root / "editor-copyback.cpp").write_bytes(b"different\n")
            with self.assertRaisesRegex(provenance.ProvenanceError, "not byte-identical"):
                self.make_attestation(root, plan_hash, plan_path, initialize_sources=False)

    def test_derivation_binds_exact_clean_90_game_window(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            plan_hash, plan_path, _ = self.make_plan(root)
            attestation_hash, attestation_path, attestation = self.make_attestation(root, plan_hash, plan_path)
            registry_payload = {
                "records": [],
                "schema": provenance.EXCLUSION_SCHEMA,
                "selection": "unit-test empty sealed ID registry",
                "sources": [],
            }
            registry_hash, registry_path = provenance.write_content_addressed(root / "exclusions", registry_payload)
            records = []
            for offset in range(90):
                game_id = 10_000 + offset
                normalized = {"gameId": game_id, "metadataOnlyTest": True}
                normalized_content = provenance.canonical_json_bytes(normalized)
                normalized_hash = provenance.sha256_bytes(normalized_content)
                raw_content = b'{"gameId":' + str(game_id).encode("ascii") + b'}\n'
                raw_hash = provenance.sha256_bytes(raw_content)
                raw_path = root / "raw" / f"{raw_hash}.json"
                normalized_path = root / "normalized" / f"{normalized_hash}.json"
                replay_path = root / "replays" / str(game_id) / f"{normalized_hash}.json"
                for path, content in ((raw_path, raw_content), (normalized_path, normalized_content), (replay_path, normalized_content)):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(content)
                record = {
                    "acquisition": {
                        "fetched_at_utc": "2026-08-13T10:13:12Z",
                        "normalized_path": normalized_path.relative_to(root).as_posix(),
                        "normalized_sha256": normalized_hash,
                        "raw_path": raw_path.relative_to(root).as_posix(),
                        "raw_sha256": raw_hash,
                        "replay_payload_path": replay_path.relative_to(root).as_posix(),
                    },
                    "focus": {"agent_id": 101, "submission_id": 201},
                    "game_id": game_id,
                    "operational": {
                        "classification": "clean",
                        "focus_status": "ok",
                        "opponent_status": "ok",
                        "signals": [],
                        "unscoped_signals": [],
                    },
                    "opponent": {"frozen_rank": 15},
                    "outcome": {"winner_player_id": offset % 2},
                    "purpose": {},
                    "replay": {
                        "observed_transcript": "0/4",
                        "rules_validation": {"status": "terminal-valid"},
                        "valid_transcript": "0/4",
                    },
                    "schema": provenance.ARENA_GAME_SCHEMA,
                    "source_sha256": attestation["source"]["sha256"],
                    "status": "accepted",
                }
                record_hash, record_path = provenance.write_content_addressed(root / "game-records" / str(game_id), record)
                records.append(
                    {
                        "record": record,
                        "record_path": record_path.relative_to(root).as_posix(),
                        "record_sha256": record_hash,
                    }
                )
            source_archive = root / attestation["source"]["archived_path"]
            manifest = {
                "binding": {
                    "agent_id": 101,
                    "asserted_submission_id": 201,
                    "collector_sha256": "c" * 64,
                    "repository_commit": "a" * 40,
                    "run_id": "unit-test-window",
                    "schema": provenance.ARENA_BINDING_SCHEMA,
                    "source": {
                        "archived_path": source_archive.relative_to(root).as_posix(),
                        "bytes": attestation["source"]["bytes"],
                        "characters": attestation["source"]["characters"],
                        "sha256": attestation["source"]["sha256"],
                    },
                },
                "completed_at_utc": "2026-08-13T10:13:13Z",
                "collector_sha256": "c" * 64,
                "coverage": {
                    "accepted_games": 90,
                    "battle_window_games": 90,
                    "expected_games": 90,
                    "full_window_accounted": True,
                    "status_counts": {"accepted": 90},
                },
                "exclusion_registry": {"path": registry_path.relative_to(root).as_posix(), "sha256": registry_hash},
                "games": records,
                "schema": provenance.ARENA_BATCH_SCHEMA,
                "started_at_utc": "2026-08-13T10:13:11Z",
                "run_id": "unit-test-window",
                "window_snapshot": {
                    "normalized_sha256": "d" * 64,
                    "raw_sha256": "e" * 64,
                },
            }
            manifest_hash, manifest_path = provenance.write_content_addressed(root / "manifests", manifest)
            digest, derived_path, derived = provenance.derive_arena_window(
                plan_path=plan_path,
                expected_plan_sha256=plan_hash,
                attestation_path=attestation_path,
                expected_attestation_sha256=attestation_hash,
                arena_manifest_path=manifest_path,
                exclusion_registry_path=registry_path,
                expected_exclusion_sha256=registry_hash,
                output_directory=root / "derivations",
                repository=root,
            )
            self.assertEqual(derived_path.name, f"{digest}.json")
            self.assertEqual(derived["summary"]["eligible_games"], 90)
            self.assertEqual(derived["summary"]["rejected_games"], 0)
            self.assertEqual(
                derived["games"][0]["uses"],
                ["raw-terminal-value-candidate", "opponent-action-ranking-reanalysis-candidate"],
            )
            self.assertEqual(derived["games"][0]["ranking_candidate_weight"], 1.0)
            self.assertEqual(
                provenance.validate_campaign_sequence(
                    [derived_path],
                    repository=root,
                    plan_path=plan_path,
                    expected_plan_sha256=plan_hash,
                )["status"],
                "sequential-and-complete",
            )

    def test_protected_snapshot_detects_content_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            protected = root / "protected"
            protected.mkdir()
            evidence = protected / "bank.dat"
            evidence.write_bytes(b"sealed bytes\n")
            _, snapshot_path, _ = provenance.create_protected_snapshot(
                protected_paths=[protected],
                label="unit-test protected bank",
                created_at_utc="2026-08-13T10:13:00Z",
                output_directory=root / "snapshots",
                repository=root,
            )
            self.assertEqual(
                provenance.verify_protected_snapshot(snapshot_path, repository=root)["status"],
                "unchanged",
            )
            evidence.write_bytes(b"changed bytes\n")
            with self.assertRaisesRegex(provenance.ProvenanceError, "protected paths changed"):
                provenance.verify_protected_snapshot(snapshot_path, repository=root)


if __name__ == "__main__":
    unittest.main()
