import copy
import datetime as dt
import importlib.util
import json
import pathlib
import sys
import tempfile
import threading
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
COLLECTOR_PATH = (
    ROOT
    / "submissions"
    / "codingame"
    / "tools"
    / "collect_arena_batch.py"
)
FIXTURES = ROOT / "tests" / "fixtures" / "codingame" / "live_replay"
SPEC = importlib.util.spec_from_file_location("arena_batch_collector", COLLECTOR_PATH)
arena = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = arena
SPEC.loader.exec_module(arena)


def fixture(name):
    return json.loads((FIXTURES / name).read_text())


def empty_registry():
    return arena.shared.ExclusionRegistry(
        {
            "schema": arena.shared.EXCLUSION_SCHEMA,
            "selection": "unit-test empty registry",
            "sources": [],
            "records": [],
        }
    )


def battle(game_id, first_agent, second_agent, *, submission_id=777):
    return {
        "done": True,
        "gameId": game_id,
        "players": [
            {
                "nickname": f"agent-{first_agent}",
                "playerAgentId": first_agent,
                "position": 0,
                "publicHandle": f"handle-{first_agent}",
                "submissionId": submission_id if first_agent == 101 else 20_000 + first_agent,
                "testSessionHandle": f"session-{first_agent}",
                "userId": first_agent,
            },
            {
                "nickname": f"agent-{second_agent}",
                "playerAgentId": second_agent,
                "position": 1,
                "publicHandle": f"handle-{second_agent}",
                "submissionId": submission_id if second_agent == 101 else 20_000 + second_agent,
                "testSessionHandle": f"session-{second_agent}",
                "userId": second_agent,
            },
        ],
    }


def clean_game(game_id, first_agent, second_agent):
    payload = fixture("game_1001.json")
    payload["gameId"] = game_id
    payload["agents"][0]["agentId"] = first_agent
    payload["agents"][0]["codingamer"]["pseudo"] = f"agent-{first_agent}"
    payload["agents"][1]["agentId"] = second_agent
    payload["agents"][1]["codingamer"]["pseudo"] = f"agent-{second_agent}"
    return payload


def timeout_game(game_id, focus_agent, opponent_agent):
    return {
        "agents": [
            {
                "agentId": focus_agent,
                "codingamer": {"pseudo": f"agent-{focus_agent}"},
                "index": 0,
                "valid": False,
            },
            {
                "agentId": opponent_agent,
                "codingamer": {"pseudo": f"agent-{opponent_agent}"},
                "index": 1,
                "valid": True,
            },
        ],
        "frames": [
            {"agentId": 0, "stdout": "0"},
            {
                "agentId": 1,
                "stderr": "Timeout: exceeded the time limit",
                "stdout": "",
            },
        ],
        "gameId": game_id,
        "ranks": [0, 1],
    }


class StepClock:
    def __init__(self):
        self.value = 0
        self.lock = threading.Lock()

    def __call__(self):
        with self.lock:
            current = self.value
            self.value += 1
        instant = dt.datetime(2026, 8, 11, tzinfo=dt.timezone.utc) + dt.timedelta(
            seconds=current
        )
        return instant.isoformat().replace("+00:00", "Z")


class FakeApi:
    def __init__(self):
        self.routes = {}
        self.calls = []
        self.lock = threading.Lock()

    def add(self, schema, payload, *responses):
        service = arena.shared.REQUEST_SCHEMAS[schema]["service"]
        key = (service, json.dumps(payload, separators=(",", ":"), sort_keys=True))
        self.routes.setdefault(key, []).extend(responses)

    def post(self, service, payload):
        key = (service, json.dumps(payload, separators=(",", ":"), sort_keys=True))
        with self.lock:
            self.calls.append((service, copy.deepcopy(payload)))
            if key not in self.routes or not self.routes[key]:
                raise AssertionError(f"unexpected API request: {key}")
            response = self.routes[key].pop(0)
        if isinstance(response, BaseException):
            raise response
        return arena.shared.ApiResponse(
            body=arena.shared.canonical_json_bytes(response),
            headers={"content-type": "application/json"},
        )

    def count(self, schema, payload=None):
        service = arena.shared.REQUEST_SCHEMAS[schema]["service"]
        return sum(
            called_service == service and (payload is None or called_payload == payload)
            for called_service, called_payload in self.calls
        )


class ArenaBatchCollectorTest(unittest.TestCase):
    def make_collector(self, temporary, api, registry=None):
        return arena.ArenaBatchCollector(
            repository=ROOT,
            data_root=pathlib.Path(temporary) / "diagnostics",
            api=api,
            clock=StepClock(),
            exclusion_registry=registry or empty_registry(),
            maximum_workers=1,
        )

    def add_common(self, api, battles):
        api.add(
            "leaderboard-v1",
            [
                "paper-soccer",
                None,
                "global",
                {"active": False, "column": "", "filter": ""},
            ],
            fixture("leaderboard.json"),
        )
        api.add("agent-battles-v1", [101, None], battles)

    def bind(self, collector, temporary):
        source = pathlib.Path(temporary) / "submitted.cpp"
        source.write_text("int main() { return 0; }\n")
        return collector.bind_source(
            agent_id=101,
            submission_id=777,
            source_path=source,
            repository_commit="a" * 40,
        )

    def manifest(self, temporary, result):
        path = pathlib.Path(result["manifest_path"])
        if not path.is_absolute():
            path = ROOT / path
        # Temporary roots are outside the repository, so the collector emits
        # an absolute path in unit tests.
        self.assertTrue(path.is_file())
        return json.loads(path.read_text())

    def test_collects_entire_focus_window_without_top_player_filter(self):
        with tempfile.TemporaryDirectory() as temporary:
            api = FakeApi()
            battles = [battle(1001, 202, 101), battle(1002, 9002, 101)]
            self.add_common(api, battles)
            api.add("game-detail-v1", [1001, None], clean_game(1001, 202, 101))
            api.add("game-detail-v1", [1002, None], clean_game(1002, 9002, 101))
            collector = self.make_collector(temporary, api)
            binding = self.bind(collector, temporary)

            result = collector.collect(
                run_id="whole-window", binding=binding, expected_games=2
            )
            manifest = self.manifest(temporary, result)

            self.assertTrue(result["coverage"]["full_window_accounted"])
            self.assertEqual(result["coverage"]["accepted_games"], 2)
            records = [item["record"] for item in manifest["games"]]
            self.assertEqual([item["game_id"] for item in records], [1001, 1002])
            unranked = records[1]
            self.assertIsNone(unranked["opponent"]["frozen_rank"])
            self.assertEqual(unranked["focus"]["color"], "player-1")
            self.assertEqual(unranked["focus"]["result"], "loss")
            self.assertEqual(unranked["replay"]["rules_validation"]["status"], "terminal-valid")
            self.assertEqual(unranked["replay"]["valid_transcript"], "0/0/3/0/61/0/07")
            self.assertFalse(manifest["purpose"]["training_eligible"])
            source_payload = pathlib.Path(binding.archived_path)
            self.assertEqual(source_payload.read_bytes(), b"int main() { return 0; }\n")
            self.assertEqual(api.count("game-detail-v1"), 2)

            export_path = pathlib.Path(temporary) / "auditor.tsv"
            export = arena.export_auditor_tsv(
                pathlib.Path(result["manifest_path"]),
                export_path,
                collector.registry_sha256,
            )
            self.assertEqual(export["exported_games"], 2)
            lines = export_path.read_text().splitlines()
            header = lines.index("game_id\tcandidate_player\twinner\tturns")
            self.assertEqual(
                lines[header:],
                [
                    "game_id\tcandidate_player\twinner\tturns",
                    "1001\t1\t0\t0/0/3/0/61/0/07",
                    "1002\t1\t0\t0/0/3/0/61/0/07",
                ],
            )
            metadata = dict(line[2:].split("=", 1) for line in lines[:header])
            self.assertEqual(metadata["agent_id"], "101")
            self.assertEqual(metadata["asserted_submission_id"], "777")
            self.assertEqual(
                metadata["source_binding_status"],
                "asserted-not-api-verified",
            )
            self.assertEqual(metadata["arena_manifest_sha256"], result["manifest_sha256"])

    def test_operational_forfeit_comes_from_frames_not_agents_valid(self):
        with tempfile.TemporaryDirectory() as temporary:
            api = FakeApi()
            self.add_common(api, [battle(2001, 101, 9002)])
            api.add("game-detail-v1", [2001, None], timeout_game(2001, 101, 9002))
            collector = self.make_collector(temporary, api)
            result = collector.collect(
                run_id="timeout", binding=self.bind(collector, temporary), expected_games=1
            )
            record = self.manifest(temporary, result)["games"][0]["record"]

            self.assertEqual(record["status"], "accepted")
            self.assertEqual(record["operational"]["classification"], "operationally-terminated")
            self.assertEqual(record["operational"]["focus_status"], "ok")
            self.assertEqual(record["operational"]["opponent_status"], "timeout")
            self.assertEqual(record["replay"]["rules_validation"]["status"], "incomplete")
            self.assertEqual(record["replay"]["valid_transcript"], "0")
            self.assertEqual(result["coverage"]["focus_operational_failures"], 0)
            self.assertEqual(result["coverage"]["opponent_operational_failures"], 1)
            with self.assertRaisesRegex(ValueError, "no clean rule-terminal"):
                arena.export_auditor_tsv(
                    pathlib.Path(result["manifest_path"]),
                    pathlib.Path(temporary) / "timeout.tsv",
                    collector.registry_sha256,
                )

    def test_offline_export_validates_approved_archive_cross_references(self):
        with tempfile.TemporaryDirectory() as temporary:
            api = FakeApi()
            self.add_common(api, [battle(1001, 202, 101)])
            api.add("game-detail-v1", [1001, None], clean_game(1001, 202, 101))
            collector = self.make_collector(temporary, api)
            result = collector.collect(
                run_id="offline-export",
                binding=self.bind(collector, temporary),
                expected_games=1,
            )
            manifest_path = pathlib.Path(result["manifest_path"])
            manifest = json.loads(manifest_path.read_text())

            with self.assertRaisesRegex(ValueError, "is not approved"):
                arena.export_auditor_tsv(
                    manifest_path,
                    pathlib.Path(temporary) / "wrong-registry.tsv",
                    "0" * 64,
                )

            forged_source = copy.deepcopy(manifest)
            forged_source["binding"]["source"]["sha256"] = "0" * 64
            _, forged_source_path = arena.shared.write_content_addressed(
                pathlib.Path(temporary) / "forged-manifests", forged_source
            )
            with self.assertRaisesRegex(ValueError, "source payload hash mismatch"):
                arena.export_auditor_tsv(
                    forged_source_path,
                    pathlib.Path(temporary) / "forged-source.tsv",
                    collector.registry_sha256,
                )

            forged_record = copy.deepcopy(manifest)
            forged_record["games"][0]["record"]["status"] = "request-error"
            _, forged_record_path = arena.shared.write_content_addressed(
                pathlib.Path(temporary) / "forged-manifests", forged_record
            )
            with self.assertRaisesRegex(ValueError, "embeds an invalid game record"):
                arena.export_auditor_tsv(
                    forged_record_path,
                    pathlib.Path(temporary) / "forged-record.tsv",
                    collector.registry_sha256,
                )

            protected_registry = arena.shared.ExclusionRegistry(
                {
                    "schema": arena.shared.EXCLUSION_SCHEMA,
                    "selection": "unit-test protected registry",
                    "sources": [],
                    "records": [
                        {
                            "categories": ["protected_evaluation"],
                            "game_id": 1001,
                            "sources": ["sealed.json"],
                        }
                    ],
                }
            )
            protected_hash, protected_path = arena.shared.write_content_addressed(
                pathlib.Path(temporary) / "diagnostics" / "exclusions",
                protected_registry.payload,
            )
            forged_exclusion = copy.deepcopy(manifest)
            forged_exclusion["exclusion_registry"] = {
                "path": protected_path.as_posix(),
                "sha256": protected_hash,
            }
            _, forged_exclusion_path = arena.shared.write_content_addressed(
                pathlib.Path(temporary) / "forged-manifests", forged_exclusion
            )
            with self.assertRaisesRegex(ValueError, "contradicts"):
                arena.export_auditor_tsv(
                    forged_exclusion_path,
                    pathlib.Path(temporary) / "forged-exclusion.tsv",
                    protected_hash,
                )

    def test_protected_game_is_accounted_for_without_detail_request(self):
        registry = arena.shared.ExclusionRegistry(
            {
                "schema": arena.shared.EXCLUSION_SCHEMA,
                "selection": "unit-test protected registry",
                "sources": [],
                "records": [
                    {
                        "categories": ["protected_evaluation"],
                        "game_id": 1001,
                        "sources": ["sealed.json"],
                    }
                ],
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            api = FakeApi()
            self.add_common(api, [battle(1001, 202, 101)])
            collector = self.make_collector(temporary, api, registry)
            result = collector.collect(
                run_id="protected",
                binding=self.bind(collector, temporary),
                expected_games=1,
            )
            record = self.manifest(temporary, result)["games"][0]["record"]

            self.assertEqual(record["status"], "excluded-protected")
            self.assertEqual(api.count("game-detail-v1"), 0)
            self.assertTrue(result["coverage"]["full_window_accounted"])

    def test_initialization_never_scans_protected_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(
                arena.shared,
                "build_exclusion_registry",
                side_effect=AssertionError("must not scan tracked evidence"),
            ) as builder:
                with self.assertRaisesRegex(ValueError, "pre-built frozen"):
                    arena.ArenaBatchCollector(
                        repository=ROOT,
                        data_root=pathlib.Path(temporary) / "diagnostics",
                        api=FakeApi(),
                    )
            builder.assert_not_called()

    def test_registry_and_rank_validation_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            registry_path = pathlib.Path(temporary) / "registry.json"
            registry_path.write_bytes(
                arena.shared.canonical_json_bytes(empty_registry().payload)
            )
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                arena.load_exclusion_registry(registry_path, "0" * 64)

            detail = clean_game(1001, 202, 101)
            detail["ranks"] = [0, 1.9]
            normalized_battle = arena.shared.normalize_battle(
                battle(1001, 202, 101), 101
            )
            with self.assertRaisesRegex(ValueError, "invalid ranks"):
                arena.validate_arena_detail(
                    detail,
                    game_id=1001,
                    battle=normalized_battle,
                    focus_agent_id=101,
                    leaderboard_frozen_at="2026-08-11T00:00:00Z",
                )

    def test_submission_mismatch_fails_closed_before_replay_fetch(self):
        with tempfile.TemporaryDirectory() as temporary:
            api = FakeApi()
            self.add_common(api, [battle(1001, 202, 101, submission_id=778)])
            collector = self.make_collector(temporary, api)
            result = collector.collect(
                run_id="identity",
                binding=self.bind(collector, temporary),
                expected_games=1,
            )
            record = self.manifest(temporary, result)["games"][0]["record"]

            self.assertEqual(record["status"], "identity-mismatch")
            self.assertFalse(result["coverage"]["full_window_accounted"])
            self.assertEqual(api.count("game-detail-v1"), 0)

    def test_rerun_uses_cached_detail_and_store_check_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            first_api = FakeApi()
            self.add_common(first_api, [battle(1001, 202, 101)])
            first_api.add("game-detail-v1", [1001, None], clean_game(1001, 202, 101))
            first = self.make_collector(temporary, first_api)
            binding = self.bind(first, temporary)
            first.collect(run_id="cached-one", binding=binding, expected_games=1)

            second_api = FakeApi()
            self.add_common(second_api, [battle(1001, 202, 101)])
            second = self.make_collector(temporary, second_api)
            second_binding = self.bind(second, temporary)
            second.collect(
                run_id="cached-two", binding=second_binding, expected_games=1
            )

            self.assertEqual(second_api.count("game-detail-v1"), 0)
            report = arena.check_store(pathlib.Path(temporary) / "diagnostics")
            self.assertEqual(report["source_payloads"], 1)
            self.assertEqual(report["replay_payloads"], 1)
            self.assertEqual(report["manifests"], 2)

    def test_same_run_id_resumes_with_immutable_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            first_api = FakeApi()
            self.add_common(first_api, [battle(1001, 202, 101)])
            first_api.add("game-detail-v1", [1001, None], clean_game(1001, 202, 101))
            first = self.make_collector(temporary, first_api)
            binding = self.bind(first, temporary)
            first.collect(run_id="resume", binding=binding, expected_games=1)

            second_api = FakeApi()
            self.add_common(second_api, [battle(1001, 202, 101)])
            second = self.make_collector(temporary, second_api)
            result = second.collect(
                run_id="resume",
                binding=self.bind(second, temporary),
                expected_games=1,
            )

            self.assertEqual(second_api.count("game-detail-v1"), 0)
            self.assertEqual(result["coverage"]["accepted_games"], 1)
            bindings = list(
                (pathlib.Path(temporary) / "diagnostics" / "runs" / "resume").glob(
                    "binding.json"
                )
            )
            self.assertEqual(len(bindings), 1)


if __name__ == "__main__":
    unittest.main()
