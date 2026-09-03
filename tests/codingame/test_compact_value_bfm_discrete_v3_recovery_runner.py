import copy
import contextlib
import fcntl
import hashlib
import json
import os
import pathlib
import tempfile
import threading
import time
import unittest
from unittest import mock


from submissions.codingame.bots.compact_value_bfm import (
    discrete_v3_recovery_runner as runner,
)
from submissions.codingame.bots.compact_value_bfm import (
    test_development_runner as base_test,
)
from tests.codingame import (
    test_compact_value_bfm_discrete_v3_development as development_test,
)


q = runner.qualification
development = runner.development
maintained = runner.maintained


def rewrite_journal(directory, entries):
    directory = pathlib.Path(directory)
    for path in directory.iterdir():
        path.unlink()
    previous = "0" * 64
    rewritten = []
    for sequence, value in enumerate(entries, 1):
        body = dict(value)
        body.pop("body_sha256", None)
        body["sequence"] = sequence
        body["previous_sha256"] = previous
        artifact = q.seal(body)
        path = directory / f"{sequence:06d}-{artifact['body_sha256']}.json"
        q.atomic_write_once(path, q.canonical_json_bytes(artifact))
        rewritten.append(artifact)
        previous = artifact["body_sha256"]
    return rewritten


class ConcurrentCampaign:
    def __init__(self, delegate, *, fail=False):
        self.delegate = delegate
        self.fail = fail
        self.calls = 0
        self.active = {}
        self.maximum = {}
        self.stage_calls = {}
        self.lock = threading.Lock()

    def __call__(self, candidate, bank, spec):
        stage = spec["stage"]
        with self.lock:
            self.calls += 1
            self.stage_calls[stage] = self.stage_calls.get(stage, 0) + 1
            self.active[stage] = self.active.get(stage, 0) + 1
            self.maximum[stage] = max(
                self.maximum.get(stage, 0), self.active[stage]
            )
        try:
            time.sleep(0.02)
            if self.fail:
                raise RuntimeError("synthetic launched-stage interruption")
            return self.delegate(candidate, bank, spec)
        finally:
            with self.lock:
                self.active[stage] -= 1


class ThreeTupleCampaign(development_test.V3FakeCampaign):
    def outcome(self, spec):
        if spec["stage"] == "tuple_screen":
            identifier = spec["candidate_id"]
            if identifier.endswith(":c0.80-f0.5-l1"):
                return 160, 80, 80, 9.0
            if identifier.endswith(":c0.65-f0.5-l1"):
                return 159, 79, 80, 10.0
            ordinal = list(maintained.campaign.TUPLE_ROSTER).index(
                tuple(spec["tuple"])
            )
            wins = 130 - ordinal
            return wins, wins // 2, wins - wins // 2, 11.0
        return super().outcome(spec)


class RecoveryFixture:
    def __init__(self, root):
        self.root = root.resolve()
        self.original = development_test.SyntheticContext(self.root)
        self.original.prepare()
        self.original.plan = self.original.plan_loader(
            self.original.plan_path, output_root=self.root
        )
        self.original_fake = ThreeTupleCampaign(self.original)
        self._seal_carried_original_receipts()
        self.recovery_root = self.root / "development-recovery-v1"
        self.routes = self._routes()
        pathlib.Path(self.routes["opening_banks"]).mkdir(parents=True)
        pathlib.Path(self.routes["gate_banks"]).mkdir(parents=True)
        old_tuple = self.original.banks["tuple_confirmation"]
        fresh_tuple = pathlib.Path(self.routes["opening_banks"]) / old_tuple.name
        fresh_tuple.write_bytes(old_tuple.read_bytes())
        old_gate = self.original_runner_banks["tuple_confirmation"].path
        fresh_gate = pathlib.Path(self.routes["gate_banks"]) / old_gate.name
        fresh_gate.write_bytes(old_gate.read_bytes())
        self.materialized_banks = {
            stage: (
                development._regular(fresh_tuple)
                if stage == "tuple_confirmation"
                else dict(self.original.plan["banks"][stage])
            )
            for stage in development.STAGE_ORDER
        }
        self.incident = pathlib.Path(self.routes["incident"])
        q.write_sealed(self.incident, {
            "schema": runner.recovery.INCIDENT_SCHEMA,
            "synthetic": True,
        })
        self.additional = {
            "spent_original_tuple_confirmation": {
                "bank": dict(self.original.plan["banks"]["tuple_confirmation"])
            }
        }
        self.mixed = pathlib.Path(self.routes["mixed_six_exclusion"])
        q.write_sealed(self.mixed, {
            "schema": runner.recovery.MIXED_EXCLUSION_SCHEMA,
            "selected_bank_count": 6,
            "fresh_confirmation_excluded_original_six": True,
            "fresh_confirmation_excluded_historical_seven": True,
            "fresh_confirmation_excluded_protected_fingerprints": True,
            "protected_fingerprint_count": 54_611,
            "cross_source_symmetry_intersection_count": 0,
            "additional_development_exclusions": self.additional,
            "selection_uses_only_selected_six": True,
        })
        self.plan_path = pathlib.Path(self.routes["plan"])
        self.plan = self._plan()
        q.write_sealed(self.plan_path, self.plan)
        self.plan = q.load_sealed(self.plan_path, runner.recovery.PLAN_SCHEMA)

    def _seal_carried_original_receipts(self):
        old = self.original.runner(gate_executor=self.original_fake)
        with self.original_patches():
            banks = old._banks()
            candidates = old._compile_candidates_v3()
            candidate, control = candidates
            default_work = maintained.campaign.PROFILE_ROSTER[
                maintained.campaign.DEFAULT_PROFILE
            ]
            for item in candidates:
                old._run_v3(
                    item,
                    banks["model_screen"],
                    {
                        "stage": "model_screen",
                        "candidate_id": item.candidate_id,
                        "tuple": maintained.campaign.DEFAULT_TUPLE,
                        "work": default_work,
                        "mode": "fixed-work",
                    },
                    metric_extra={
                        "architecture": item.architecture,
                        "target": item.target,
                        "source_bytes": item.source_bytes,
                        "artifact_sha256": item.runtime_sha256,
                        "deployment_eligible": item.deployment_eligible,
                    },
                )
            for values in maintained.campaign.TUPLE_ROSTER:
                identifier = (
                    f"{candidate.candidate_id}:"
                    f"{maintained.campaign.tuple_id(values)}"
                )
                old._run_v3(
                    candidate,
                    banks["tuple_screen"],
                    {
                        "stage": "tuple_screen",
                        "candidate_id": identifier,
                        "tuple": values,
                        "work": default_work,
                        "mode": "fixed-work",
                    },
                    metric_extra={
                        "model_id": candidate.candidate_id,
                        "tuple": list(values),
                    },
                )
        self.candidates = candidates
        self.original_runner_banks = banks
        self.compile_references = dict(old.compile_references)
        by_key = {}
        reference_root = pathlib.Path(self.original.plan["outputs"]["references"])
        for path in reference_root.glob("*.json"):
            reference = q.load_sealed(path, development.RECEIPT_REFERENCE_SCHEMA)
            request = q.load_sealed(
                pathlib.Path(reference["request"]["path"]), development.REQUEST_SCHEMA
            )
            key = (request["spec"]["stage"], request["spec"]["candidate_id"])
            by_key[key] = path
        order = [
            ("model_screen", development.CANDIDATE_ID),
            ("model_screen", development.CONTROL_ID),
            *[
                (
                    "tuple_screen",
                    f"{development.CANDIDATE_ID}:{maintained.campaign.tuple_id(values)}",
                )
                for values in maintained.campaign.TUPLE_ROSTER
            ],
        ]
        self.carried = [
            {
                "order": index,
                "stage": stage,
                "candidate_id": identifier,
                "tuple": (
                    list(maintained.campaign.DEFAULT_TUPLE)
                    if stage == "model_screen"
                    else list(q.load_sealed(
                        pathlib.Path(q.load_sealed(
                            by_key[(stage, identifier)],
                            development.RECEIPT_REFERENCE_SCHEMA,
                        )["request"]["path"]),
                        development.REQUEST_SCHEMA,
                    )["spec"]["tuple"])
                ),
                "reference": development._sealed_record(
                    by_key[(stage, identifier)],
                    development.RECEIPT_REFERENCE_SCHEMA,
                ),
            }
            for index, (stage, identifier) in enumerate(order, 1)
        ]

    def _routes(self):
        root = self.recovery_root
        return {
            "recovery_root": str(root),
            "plan": str(root / "plan.json"),
            "incident": str(root / "incident.json"),
            "mixed_six_exclusion": str(root / "mixed.json"),
            "opening_banks": str(root / "opening-banks"),
            "gate_banks": str(root / "gate-banks"),
            "binaries": str(root / "gate-binaries"),
            "scratch": str(root / "scratch"),
            "requests": str(root / "requests"),
            "base_receipts": str(root / "base-receipts"),
            "receipts": str(root / "receipts"),
            "references": str(root / "references"),
            "claims": str(root / "claims"),
            "journal": str(root / "journal"),
            "result": str(root / "result.json"),
            "finalists": str(root / "finalists"),
            "finalist_reference": str(root / "finalist-reference.json"),
        }

    def _plan(self):
        roster = [
            {
                "candidate_id": f"{development.CANDIDATE_ID}:{maintained.campaign.tuple_id(values)}",
                "tuple": list(values),
            }
            for values in (
                ("0.80", "0.5", "1"),
                ("0.95", "0.5", "1"),
                ("0.65", "0.5", "1"),
            )
        ]
        # The original synthetic screen ranks 0.80, 0.95, then adds default;
        # default is already present, while production's exact screen carries
        # 0.65, 0.80, default. Derive rather than guess below.
        model = [
            self._original_metric("model_screen", development.CANDIDATE_ID),
            self._original_metric("model_screen", development.CONTROL_ID),
        ]
        tuples = [
            self._original_metric(
                "tuple_screen",
                f"{development.CANDIDATE_ID}:{maintained.campaign.tuple_id(values)}",
            )
            for values in maintained.campaign.TUPLE_ROSTER
        ]
        ranked = maintained.campaign._validate_exact_tuple_screen(
            tuples, [model[0]], {row["candidate_id"]: row for row in model}
        )
        default_id = (
            f"{development.CANDIDATE_ID}:"
            f"{maintained.campaign.tuple_id(maintained.campaign.DEFAULT_TUPLE)}"
        )
        carried = []
        for identifier in [row["candidate_id"] for row in ranked[:2]] + [default_id]:
            if identifier not in carried:
                carried.append(identifier)
        descriptor = {row["candidate_id"]: row["tuple"] for row in tuples}
        roster = [
            {"candidate_id": identifier, "tuple": descriptor[identifier]}
            for identifier in carried
        ]
        return {
            "schema": runner.recovery.PLAN_SCHEMA,
            "namespace": development.NAMESPACE,
            "campaign_id": "synthetic-recovery",
            "source_campaign_id": development.CAMPAIGN_ID,
            "recovery_id": "synthetic-recovery",
            "original": {
                "development_plan": development._sealed_record(
                    self.original.plan_path, development.PLAN_SCHEMA
                ),
                "carried_receipt_references": self.carried,
                "terminal_incident": development._sealed_record(
                    self.incident, runner.recovery.INCIDENT_SCHEMA
                ),
            },
            "candidate": dict(self.original.plan["candidate"]),
            "rank4_control": dict(self.original.plan["rank4_control"]),
            "banks": dict(self.materialized_banks),
            "binaries": {
                candidate.candidate_id: development._regular(candidate.binary_path)
                for candidate in self.candidates
            },
            "compile_references": dict(self.compile_references),
            "algorithm": dict(self.original.plan["algorithm"]),
            "recovery_contract": {"tuple_confirmation_roster": roster},
            "compiler": dict(self.original.compiler),
            "tools": {
                "gate_source": development._regular(maintained.GATE_SOURCE),
                "rank4_source": development._regular(maintained.RANK4),
            },
            "concurrency": copy.deepcopy(runner.recovery.CONCURRENCY),
            "outputs": dict(self.routes),
            "additional_development_exclusions": self.additional,
            "mixed_six_exclusion": {
                "path": str(self.mixed),
                "schema": runner.recovery.MIXED_EXCLUSION_SCHEMA,
            },
        }

    def _original_metric(self, stage, identifier):
        for entry in self.carried:
            reference = q.load_sealed(
                pathlib.Path(entry["reference"]["path"]),
                development.RECEIPT_REFERENCE_SCHEMA,
            )
            validated = development.validate_run_receipt(
                reference["receipt"], self.original.plan
            )
            spec = validated["request"]["spec"]
            if spec["stage"] == stage and spec["candidate_id"] == identifier:
                return validated["metric"]
        raise AssertionError((stage, identifier))

    @contextlib.contextmanager
    def original_patches(self):
        with contextlib.ExitStack() as stack:
            for module in {
                maintained.openings,
                development_test.v3runner.maintained.openings,
            }:
                stack.enter_context(mock.patch.object(
                    module,
                    "validate_bank",
                    side_effect=base_test.fake_validate_bank,
                ))
            for module in {
                maintained,
                development.maintained,
                development_test.v3runner.maintained,
            }:
                stack.enter_context(mock.patch.object(
                    module, "paired_bootstrap_lower", return_value=0.01
                ))
            yield

    def plan_loader(self, path, *, output_root):
        if path.resolve() != self.plan_path or output_root.resolve() != self.root:
            raise runner.RecoveryRunnerError("synthetic recovery plan route changed")
        plan = q.load_sealed(path, runner.recovery.PLAN_SCHEMA)
        return {
            "plan": plan,
            "plan_path": path.resolve(),
            "original_plan": self.original.plan,
            "original_plan_path": self.original.plan_path,
            "materialized": True,
            "materialized_banks": self.materialized_banks,
            "mixed_exclusion": q.load_sealed(
                self.mixed, runner.recovery.MIXED_EXCLUSION_SCHEMA
            ),
            "materialized_mixed_six_exclusion": development._sealed_record(
                self.mixed, runner.recovery.MIXED_EXCLUSION_SCHEMA
            ),
        }

    def candidate_builder(self, _plan):
        return self.candidates[0]

    def make_runner(self, gate_executor, *, read_only=False):
        return runner.DiscreteV3RecoveryRunner(
            plan_path=self.plan_path,
            output_root=self.root,
            plan_loader=self.plan_loader,
            compiler_identity=self.original.compiler,
            candidate_builder=self.candidate_builder,
            gate_executor=gate_executor,
            original_plan_loader=self.original.plan_loader,
            read_only=read_only,
        )


class RecoveryRunnerTest(unittest.TestCase):
    def test_full_mocked_recovery_is_concurrent_no_replay_and_seals_finalist(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RecoveryFixture(pathlib.Path(temporary))
            fake = ConcurrentCampaign(fixture.original_fake)
            with fixture.original_patches():
                value = fixture.make_runner(fake).execute()
                calls = fake.calls
                resumed = fixture.make_runner(fake).execute()
                with self.assertRaisesRegex(
                    runner.RecoveryRunnerError, "production chronology contains injected"
                ):
                    runner.validate_recovery_finalist(
                        pathlib.Path(fixture.routes["finalist_reference"]),
                        plan_path=fixture.plan_path,
                        output_root=fixture.root,
                        plan_loader=fixture.plan_loader,
                        compiler_identity=fixture.original.compiler,
                        candidate_builder=fixture.candidate_builder,
                        original_plan_loader=fixture.original.plan_loader,
                    )
            self.assertEqual(value["finalist"], resumed["finalist"])
            self.assertEqual(fake.calls, calls)
            self.assertEqual(fake.maximum["tuple_confirmation"], 3)
            self.assertEqual(fake.maximum["profile_screen"], 3)
            self.assertGreaterEqual(fake.maximum["profile_confirmation"], 2)
            self.assertEqual(fake.maximum["actual_clock"], 1)
            self.assertEqual(fake.calls, value["result"]["request_count"] - 10)
            self.assertEqual(len(list(pathlib.Path(fixture.routes["claims"]).glob("*.json"))), 4)
            self.assertEqual(
                value["finalist"]["mixed_six_exclusion"],
                development._sealed_record(
                    fixture.mixed, runner.recovery.MIXED_EXCLUSION_SCHEMA
                ),
            )

    def test_completed_campaign_rejects_fabricated_missing_launch_chronology(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RecoveryFixture(pathlib.Path(temporary))
            fake = ConcurrentCampaign(fixture.original_fake)
            with fixture.original_patches():
                fixture.make_runner(fake).execute()
                journal = pathlib.Path(fixture.routes["journal"])
                entries = runner._journal_entries(journal)
                stripped = [
                    entry for entry in entries
                    if not (
                        entry.get("stage") == "tuple_confirmation"
                        and entry["event"] in {
                            "batch-launching", "job-started", "batch-launched"
                        }
                    )
                ]
                rewrite_journal(journal, stripped)
                with self.assertRaisesRegex(
                    runner.RecoveryRunnerError,
                    "completed chronology is incomplete or reordered",
                ):
                    fixture.make_runner(fake).execute()

    def test_completed_campaign_rejects_tampered_job_start(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RecoveryFixture(pathlib.Path(temporary))
            fake = ConcurrentCampaign(fixture.original_fake)
            with fixture.original_patches():
                fixture.make_runner(fake).execute()
                journal = pathlib.Path(fixture.routes["journal"])
                entries = runner._journal_entries(journal)
                for entry in entries:
                    if entry["event"] == "job-started":
                        entry["request_sha256"] = "0" * 64
                        break
                rewrite_journal(journal, entries)
                with self.assertRaisesRegex(
                    runner.RecoveryRunnerError, "job-started proof changed"
                ):
                    fixture.make_runner(fake).execute()

    def test_post_completion_error_never_appends_terminal_event(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RecoveryFixture(pathlib.Path(temporary))
            fake = ConcurrentCampaign(fixture.original_fake)
            with fixture.original_patches():
                value = fixture.make_runner(fake).execute()
                journal = pathlib.Path(fixture.routes["journal"])
                before = {
                    path.name: path.read_bytes() for path in journal.iterdir()
                }
                receipt = pathlib.Path(
                    value["result"]["run_receipts"]["fresh_recovery"][0]["path"]
                )
                receipt.write_bytes(b"tampered\n")
                with self.assertRaises(runner.RecoveryRunnerError):
                    fixture.make_runner(fake).execute()
                after = {
                    path.name: path.read_bytes() for path in journal.iterdir()
                }
            self.assertEqual(after, before)
            self.assertEqual(
                runner._journal_entries(journal)[-1]["event"],
                "campaign-complete",
            )

    def test_interrupted_claim_is_terminal_and_never_replayed(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RecoveryFixture(pathlib.Path(temporary))
            fake = ConcurrentCampaign(fixture.original_fake, fail=True)
            with fixture.original_patches():
                with self.assertRaises(RuntimeError):
                    fixture.make_runner(fake).execute()
                calls = fake.calls
                with self.assertRaises(runner.TerminalRecoveryError):
                    fixture.make_runner(fake).execute()
            self.assertEqual(fake.calls, calls)
            entries = runner._journal_entries(pathlib.Path(fixture.routes["journal"]))
            self.assertEqual(entries[-1]["event"], "terminal-failure")
            self.assertTrue(entries[-1]["no_retry"])

    def test_invalid_launched_output_is_terminal_and_never_replayed(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RecoveryFixture(pathlib.Path(temporary))

            class InvalidCampaign(ConcurrentCampaign):
                def __call__(self, candidate, bank, spec):
                    document = super().__call__(candidate, bank, spec)
                    document["result"]["completed_games"] = document["result"].pop(
                        "games"
                    )
                    document["result"]["unfinished_games"] = document["result"].pop(
                        "unfinished"
                    )
                    return document

            fake = InvalidCampaign(fixture.original_fake)
            with fixture.original_patches():
                with self.assertRaises(runner.RecoveryRunnerError):
                    fixture.make_runner(fake).execute()
                calls = fake.calls
                with self.assertRaises(runner.TerminalRecoveryError):
                    fixture.make_runner(fake).execute()
            self.assertEqual(fake.calls, calls)

    def test_crash_after_full_batch_outputs_adopts_without_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RecoveryFixture(pathlib.Path(temporary))
            fake = ConcurrentCampaign(fixture.original_fake)
            first = fixture.make_runner(fake)
            with fixture.original_patches():
                original_rows, _records = first._prepare_contract()
                tuple_rows = original_rows[2:]
                descriptors = {
                    row["candidate_id"]: row for row in tuple_rows
                }
                jobs = [
                    first._build_job(
                        stage="tuple_confirmation",
                        candidate_id=item["candidate_id"],
                        tuple_values=descriptors[item["candidate_id"]]["tuple"],
                        work=maintained.campaign.PROFILE_ROSTER[
                            maintained.campaign.DEFAULT_PROFILE
                        ],
                        metric_extra={
                            "model_id": development.CANDIDATE_ID,
                            "tuple": descriptors[item["candidate_id"]]["tuple"],
                        },
                    )
                    for item in fixture.plan["recovery_contract"][
                        "tuple_confirmation_roster"
                    ]
                ]
                policy = first._stage_concurrency("tuple_confirmation", jobs)
                first._claim_stage("tuple_confirmation", jobs, policy)
                runner._append_event(
                    first.routes["journal"],
                    plan_record=first.plan_record,
                    recovery_id=first.plan["recovery_id"],
                    event="batch-launching",
                    stage="tuple_confirmation",
                    claim=development._sealed_record(
                        first._claim_path("tuple_confirmation"), runner.CLAIM_SCHEMA
                    ),
                    candidate_ids=[job.candidate_id for job in jobs],
                    no_retry=True,
                )
                first._execute_injected_batch("tuple_confirmation", jobs)
                self.assertEqual(fake.stage_calls["tuple_confirmation"], 3)
                value = fixture.make_runner(fake).execute()
            self.assertEqual(fake.stage_calls["tuple_confirmation"], 3)
            self.assertEqual(
                value["finalist"]["status"],
                "development-selected-awaiting-preflight-and-frozen-final",
            )
            completion = [
                event
                for event in runner._journal_entries(
                    pathlib.Path(fixture.routes["journal"])
                )
                if event["event"] == "stage-complete"
                and event["stage"] == "tuple_confirmation"
            ][0]
            self.assertTrue(completion["recovered_without_replay"])

    def test_original_receipt_tamper_fails_before_any_recovery_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RecoveryFixture(pathlib.Path(temporary))
            reference = pathlib.Path(fixture.carried[0]["reference"]["path"])
            reference.write_bytes(b"tampered\n")
            gate = mock.Mock(side_effect=AssertionError("recovery gate ran"))
            with fixture.original_patches():
                with self.assertRaises(runner.RecoveryRunnerError):
                    fixture.make_runner(gate).execute()
            gate.assert_not_called()
            self.assertEqual(
                list(pathlib.Path(fixture.routes["claims"]).iterdir()), []
            )

    def test_exclusive_recovery_lock_prevents_second_orchestrator(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RecoveryFixture(pathlib.Path(temporary))
            lock_path = fixture.recovery_root / "recovery.lock"
            descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                gate = mock.Mock(side_effect=AssertionError("gate ran"))
                with self.assertRaisesRegex(
                    runner.RecoveryRunnerError, "another recovery runner"
                ):
                    fixture.make_runner(gate).execute()
                gate.assert_not_called()
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)

    def test_tiny_real_gate_schema_uses_result_games_and_unfinished(self):
        engine = base_test.engine_summary(1, 1.0)
        games = [
            {
                "opening_id": "tiny-0",
                "pair_index": 0,
                "candidate_player": color,
                "winner": color if color == 0 else 0,
                "turns": 1,
                "failure": None,
                "failure_detail": None,
                "candidate": engine,
                "rank4": engine,
            }
            for color in (0, 1)
        ]
        document = {
            "schema": maintained.gate_support.RESULT_SCHEMA,
            "bindings": {
                "candidate_source_sha256": "a" * 64,
                "candidate_source_bytes": 1,
                "candidate_runtime_body_sha256": "b" * 64,
                "candidate_payload_sha256": "c" * 64,
                "rank4_source_sha256": maintained.RANK4_SHA256,
                "rank4_source_bytes": maintained.RANK4.stat().st_size,
                "opponent_sha256": maintained.RANK4_SHA256,
                "bank_sha256": "d" * 64,
                "bank_bytes": 1,
            },
            "config": {
                "mode": "fixed-work",
                "pair_offset": 0,
                "pair_count": 1,
                "candidate_clocks_ms": [800, 155],
                "rank4_clocks_ms": [800, 165],
                "max_turns": 320,
                "minimum_candidate_wins": -1,
                "minimum_wins_per_color": -1,
            },
            "games": games,
            "result": {
                "games": 2,
                "candidate_wins": 1,
                "candidate_wins_player0": 1,
                "candidate_wins_player1": 0,
                "rank4_wins": 1,
                "failures": 0,
                "unfinished": 0,
                "failure_categories": {},
                "candidate": base_test.engine_summary(2, 1.0),
                "rank4": base_test.engine_summary(2, 1.0),
                "passed": True,
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "tiny.json"
            path.write_bytes(maintained.canonical_json_bytes(document))
            validated = maintained.gate_support.validate_result(
                path,
                expected_bank_sha256="d" * 64,
                expected_candidate_sha256="a" * 64,
            )
        runner.validate_real_gate_counts(validated, pairs=1)
        stale = copy.deepcopy(validated)
        stale["result"]["completed_games"] = stale["result"].pop("games")
        stale["result"]["unfinished_games"] = stale["result"].pop("unfinished")
        with self.assertRaises(runner.RecoveryRunnerError):
            runner.validate_real_gate_counts(stale, pairs=1)


if __name__ == "__main__":
    unittest.main()
