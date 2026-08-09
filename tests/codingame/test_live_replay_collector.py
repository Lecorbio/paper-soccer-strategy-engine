import copy
import datetime as dt
import importlib.util
import json
import pathlib
import sys
import tempfile
import threading
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
COLLECTOR_PATH = (
    ROOT
    / "submissions"
    / "codingame"
    / "bots"
    / "neural_puct"
    / "collect_live_replays.py"
)
FIXTURES = ROOT / "tests" / "fixtures" / "codingame" / "live_replay"
SPEC = importlib.util.spec_from_file_location("live_replay_collector", COLLECTOR_PATH)
collector_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = collector_module
SPEC.loader.exec_module(collector_module)


def fixture(name):
    return json.loads((FIXTURES / name).read_text())


def empty_registry():
    return collector_module.ExclusionRegistry(
        {
            "schema": collector_module.EXCLUSION_SCHEMA,
            "selection": "unit-test empty registry",
            "sources": [],
            "records": [],
        }
    )


def battle(game_id, first_agent, second_agent):
    return {
        "done": True,
        "gameId": game_id,
        "players": [
            {
                "nickname": f"agent-{first_agent}",
                "playerAgentId": first_agent,
                "position": 0,
                "publicHandle": f"handle-{first_agent}",
                "submissionId": 10_000 + first_agent,
                "testSessionHandle": f"session-{first_agent}",
                "userId": first_agent,
            },
            {
                "nickname": f"agent-{second_agent}",
                "playerAgentId": second_agent,
                "position": 1,
                "publicHandle": f"handle-{second_agent}",
                "submissionId": 10_000 + second_agent,
                "testSessionHandle": f"session-{second_agent}",
                "userId": second_agent,
            },
        ],
    }


def valid_game(game_id, first_agent, second_agent, *, conflicting=False):
    payload = fixture(
        "game_1001_conflict.json" if conflicting else "game_1001.json"
    )
    payload["gameId"] = game_id
    payload["agents"][0]["agentId"] = first_agent
    payload["agents"][0]["codingamer"]["pseudo"] = f"agent-{first_agent}"
    payload["agents"][1]["agentId"] = second_agent
    payload["agents"][1]["codingamer"]["pseudo"] = f"agent-{second_agent}"
    return payload


class StepClock:
    def __init__(self):
        self.value = 0
        self.lock = threading.Lock()

    def __call__(self):
        with self.lock:
            current = self.value
            self.value += 1
        instant = dt.datetime(2026, 8, 9, tzinfo=dt.timezone.utc) + dt.timedelta(
            seconds=current
        )
        return instant.isoformat().replace("+00:00", "Z")


class FakeApi:
    def __init__(self):
        self.routes = {}
        self.calls = []
        self.lock = threading.Lock()

    def add(self, schema, payload, *responses):
        service = collector_module.REQUEST_SCHEMAS[schema]["service"]
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
        return collector_module.ApiResponse(
            body=collector_module.canonical_json_bytes(response),
            headers={"content-type": "application/json"},
        )

    def count(self, schema, payload=None):
        service = collector_module.REQUEST_SCHEMAS[schema]["service"]
        return sum(
            called_service == service and (payload is None or called_payload == payload)
            for called_service, called_payload in self.calls
        )


class LiveReplayCollectorTest(unittest.TestCase):
    def make_collector(self, temporary, api, **kwargs):
        return collector_module.LiveReplayCollector(
            repository=ROOT,
            data_root=pathlib.Path(temporary) / "live",
            api=api,
            clock=StepClock(),
            sleep=lambda _: None,
            exclusion_registry=kwargs.pop("exclusion_registry", empty_registry()),
            maximum_workers=kwargs.pop("maximum_workers", 1),
            **kwargs,
        )

    def add_leaderboard(self, api, count=1, repeats=1):
        payload = fixture("leaderboard.json")
        for _ in range(repeats):
            api.add(
                "leaderboard-v1",
                [
                    "paper-soccer",
                    None,
                    "global",
                    {"active": False, "column": "", "filter": ""},
                ],
                payload,
            )
        return payload["users"][:count]

    def test_cursor_is_audit_only_and_late_unseen_game_is_collected(self):
        with tempfile.TemporaryDirectory() as temporary:
            api = FakeApi()
            self.add_leaderboard(api, repeats=2)
            first_window = [battle(1002, 9002, 101), battle(1001, 202, 101)]
            second_window = [battle(1002, 9002, 101), battle(1000, 9000, 101)]
            api.add("agent-battles-v1", [101, None], first_window, second_window)
            api.add("game-detail-v1", [1001, None], valid_game(1001, 202, 101))
            api.add("game-detail-v1", [1002, None], valid_game(1002, 9002, 101))
            api.add("game-detail-v1", [1000, None], valid_game(1000, 9000, 101))
            collector = self.make_collector(temporary, api)

            first = collector.collect_poll(
                run_id="cursor",
                poll_index=0,
                minimum_new_games=1,
                initial_top=1,
                expanded_top=1,
            )
            second = collector.collect_poll(
                run_id="cursor",
                poll_index=1,
                minimum_new_games=1,
                initial_top=1,
                expanded_top=1,
            )

            self.assertEqual(first["cursor_after"], {"101": 1002})
            self.assertEqual(second["cursor_before"], {"101": 1002})
            self.assertEqual(second["cursor_after"], {"101": 1002})
            self.assertEqual(api.count("game-detail-v1", [1002, None]), 1)
            self.assertEqual(api.count("game-detail-v1", [1000, None]), 1)
            self.assertEqual(collector._valid_discoveries("cursor"), [1000, 1001, 1002])

    def test_top_player_overlap_fetches_detail_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            api = FakeApi()
            self.add_leaderboard(api)
            api.add("agent-battles-v1", [101, None], fixture("battles_101.json")[:1])
            api.add("agent-battles-v1", [202, None], fixture("battles_202.json"))
            api.add("game-detail-v1", [1001, None], fixture("game_1001.json"))
            collector = self.make_collector(temporary, api)

            poll = collector.collect_poll(
                run_id="overlap",
                poll_index=0,
                minimum_new_games=1,
                initial_top=2,
                expanded_top=2,
            )

            self.assertEqual(api.count("game-detail-v1", [1001, None]), 1)
            accepted = [item for item in poll["decisions"] if item["status"] == "accepted"]
            self.assertEqual([item["game_id"] for item in accepted], [1001])
            record_path = pathlib.Path(temporary) / "live" / "games" / "1001"
            record = json.loads(next(record_path.glob("*.json")).read_text())
            self.assertEqual(record["battle_metadata"]["observation_count"], 2)

    def test_interrupted_run_resumes_without_refetching_completed_game(self):
        with tempfile.TemporaryDirectory() as temporary:
            first_api = FakeApi()
            self.add_leaderboard(first_api)
            first_api.add(
                "agent-battles-v1",
                [101, None],
                [battle(1001, 202, 101), battle(1002, 9002, 101)],
            )
            first_api.add("game-detail-v1", [1001, None], valid_game(1001, 202, 101))
            first_api.add("game-detail-v1", [1002, None], KeyboardInterrupt())
            first = self.make_collector(temporary, first_api, maximum_workers=1)
            with self.assertRaises(KeyboardInterrupt):
                first.run(
                    run_id="interrupted",
                    polls=1,
                    poll_interval_seconds=0,
                    minimum_new_games=2,
                    initial_top=1,
                    expanded_top=1,
                )

            second_api = FakeApi()
            self.add_leaderboard(second_api)
            second_api.add(
                "agent-battles-v1",
                [101, None],
                [battle(1001, 202, 101), battle(1002, 9002, 101)],
            )
            second_api.add("game-detail-v1", [1002, None], valid_game(1002, 9002, 101))
            second = self.make_collector(temporary, second_api, maximum_workers=1)
            result = second.run(
                run_id="interrupted",
                polls=1,
                poll_interval_seconds=0,
                minimum_new_games=2,
                initial_top=1,
                expanded_top=1,
            )

            self.assertEqual(second_api.count("game-detail-v1", [1001, None]), 0)
            self.assertEqual(second_api.count("game-detail-v1", [1002, None]), 1)
            self.assertEqual(result["new_valid_game_ids"], [1001, 1002])
            self.assertEqual(result["decision"], "enough-data")

    def test_conflicting_replay_payloads_are_preserved_and_quarantined(self):
        with tempfile.TemporaryDirectory() as temporary:
            first_api = FakeApi()
            self.add_leaderboard(first_api)
            first_api.add("agent-battles-v1", [101, None], [battle(1001, 202, 101)])
            first_api.add("game-detail-v1", [1001, None], valid_game(1001, 202, 101))
            first = self.make_collector(temporary, first_api)
            first.collect_poll(
                run_id="conflict",
                poll_index=0,
                minimum_new_games=1,
                initial_top=1,
                expanded_top=1,
            )

            second_api = FakeApi()
            self.add_leaderboard(second_api)
            second_api.add("agent-battles-v1", [101, None], [battle(1001, 202, 101)])
            second_api.add(
                "game-detail-v1",
                [1001, None],
                valid_game(1001, 202, 101, conflicting=True),
            )
            second = self.make_collector(temporary, second_api)
            poll = second.collect_poll(
                run_id="conflict",
                poll_index=1,
                minimum_new_games=1,
                initial_top=1,
                expanded_top=1,
                refresh_details=True,
            )

            conflicts = [item for item in poll["decisions"] if item["status"] == "conflict"]
            self.assertEqual(len(conflicts), 1)
            self.assertEqual(len(conflicts[0]["normalized_payload_sha256"]), 2)
            self.assertEqual(second._valid_discoveries("conflict"), [])
            key = collector_module.request_key(
                "game-detail-v1",
                collector_module.REQUEST_SCHEMAS["game-detail-v1"]["service"],
                [1001, None],
            )
            raw_files = list(
                (pathlib.Path(temporary) / "live" / "raw" / "game-detail-v1" / key).glob(
                    "*.json"
                )
            )
            self.assertEqual(len(raw_files), 2)

    def test_protected_game_is_rejected_before_detail_request(self):
        registry = collector_module.ExclusionRegistry(
            {
                "schema": collector_module.EXCLUSION_SCHEMA,
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
            self.add_leaderboard(api)
            api.add("agent-battles-v1", [101, None], [battle(1001, 202, 101)])
            collector = self.make_collector(
                temporary, api, exclusion_registry=registry
            )
            poll = collector.collect_poll(
                run_id="protected",
                poll_index=0,
                minimum_new_games=1,
                initial_top=1,
                expanded_top=1,
            )
            self.assertEqual(api.count("game-detail-v1"), 0)
            self.assertEqual(poll["decisions"][0]["status"], "excluded-protected")

    def test_exclusion_builder_ignores_user_matches_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            repository = pathlib.Path(temporary)
            evidence = repository / "submissions" / "codingame" / "evidence.json"
            evidence.parent.mkdir(parents=True)
            evidence.write_text('{"records":[{"game_id":123}]}\n')
            matches = repository / "matches.json"
            matches.write_text("this is deliberately not JSON")
            registry = collector_module.build_exclusion_registry(
                repository,
                repository / "live",
                evidence_paths=[evidence],
            )
            self.assertEqual(registry.known_ids, {123})
            self.assertEqual(matches.read_text(), "this is deliberately not JSON")

    def test_incomplete_replay_is_structurally_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            api = FakeApi()
            self.add_leaderboard(api, repeats=2)
            api.add(
                "agent-battles-v1",
                [101, None],
                [battle(1001, 202, 101)],
                [battle(1001, 202, 101)],
            )
            incomplete = valid_game(1001, 202, 101)
            incomplete["frames"] = incomplete["frames"][:1]
            api.add("game-detail-v1", [1001, None], incomplete)
            collector = self.make_collector(temporary, api)
            first = collector.collect_poll(
                run_id="incomplete",
                poll_index=0,
                minimum_new_games=1,
                initial_top=1,
                expanded_top=1,
            )
            second = collector.collect_poll(
                run_id="incomplete",
                poll_index=1,
                minimum_new_games=1,
                initial_top=1,
                expanded_top=1,
            )
            rejected = [
                item
                for item in first["decisions"]
                if item["status"] == "structural-rejection"
            ]
            self.assertEqual(len(rejected), 1)
            self.assertIn("incomplete", rejected[0]["reason"])
            self.assertEqual(first["detail_request_count"], 1)
            self.assertEqual(second["detail_request_count"], 0)
            self.assertEqual(second["detail_validation_count"], 1)
            self.assertEqual(api.count("game-detail-v1", [1001, None]), 1)
            rebuilt = collector_module.build_exclusion_registry(
                ROOT,
                pathlib.Path(temporary) / "live",
                evidence_paths=[],
            )
            self.assertEqual(rebuilt.known_ids, {1001})
            self.assertEqual(
                rebuilt.records[1001]["categories"],
                ["live_replay_structural_rejection"],
            )


if __name__ == "__main__":
    unittest.main()
