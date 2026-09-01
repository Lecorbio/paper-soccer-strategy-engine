import hashlib
import importlib.util
import fcntl
import os
import pathlib
import tempfile
import unittest
import contextlib
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools/compact_value_bfm_discrete_v3_final.py"
SPEC = importlib.util.spec_from_file_location("compact_v3_final_tested", TOOL)
bridge = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(bridge)
q = bridge.qualification
opening_tools = bridge.opening_tools
RANK4 = ROOT / "submissions/codingame/bots/rank_4/submission.cpp"
COMMIT = "1" * 40


def synthetic_banks(root, fresh_variants):
    result = {}
    seen = set(fresh_variants)
    exclusions = {
        "sources": [], "fingerprints": [],
        "body_sha256": q.sha256_bytes(q.canonical_json_bytes({
            "sources": [], "fingerprints": [],
        })),
    }
    for stage, count in bridge.STAGE_COUNTS.items():
        seed = hashlib.sha256(f"v3-final-test:{stage}".encode()).digest()
        openings = opening_tools.generate_openings(
            stage=stage, count=count, seed=seed, excluded_fingerprints=seen
        )
        for opening in openings:
            seen.update(
                value for name, value in opening["fingerprints"].items()
                if name != "canonical"
            )
        result[stage] = opening_tools.write_bank(
            root / stage,
            opening_tools.bank_payload(
                stage=stage, classification="unprotected-development",
                seed=seed, exclusions=exclusions, openings=openings,
            ),
        )
    return result


class Fixture:
    def __init__(self, root):
        self.root = root.resolve()
        self.repository = self.root / "repository"
        self.repository.mkdir()
        self.generated = self.root / "candidate/generated.cpp"
        self.candidate = self.repository / "candidate.cpp"
        self.generated.parent.mkdir(parents=True)
        source = b"// synthetic discrete-v3 finalist\nint main(){return 0;}\n"
        self.generated.write_bytes(source)
        self.candidate.write_bytes(source)
        self.runtime = self.root / "candidate/runtime.json"
        q.write_sealed(self.runtime, {
            "schema": "papersoccer.synthetic-runtime.v1",
            "quantization": {"payload_sha256": "b" * 64},
        })
        self.gate = self.root / "gate"
        self.gate.write_text("synthetic gate\n", encoding="ascii")
        self.gate.chmod(0o700)
        self.preflight = self.root / "preflight.json"
        q.write_sealed(self.preflight, {
            "schema": bridge.preflight_tools.RECEIPT_SCHEMA,
            "synthetic": True,
        })

        fresh_openings = opening_tools.generate_openings(
            stage="fresh-private", count=20,
            seed=hashlib.sha256(b"v3-final-private").digest(),
            excluded_fingerprints=set(),
        )
        self.fresh = {
            opening["fingerprints"]["canonical"] for opening in fresh_openings
        }
        fresh_variants = {
            value for opening in fresh_openings
            for name, value in opening["fingerprints"].items()
            if name != "canonical"
        }
        self.banks = synthetic_banks(self.root / "development", fresh_variants)

        self.private_payload = self.root / "private-fingerprints.json"
        q.write_sealed(self.private_payload, {
            "schema": bridge.exclusions.FINGERPRINT_SCHEMA,
            "synthetic_count": len(self.fresh),
        })
        self.exclusion_plan = self.root / "exclusion-plan.json"
        q.write_sealed(self.exclusion_plan, {
            "schema": bridge.exclusions.PLAN_SCHEMA, "synthetic": True,
        })
        self.exclusion_receipt = self.root / "exclusion-receipt.json"
        q.write_sealed(self.exclusion_receipt, {
            "schema": bridge.exclusions.RECEIPT_SCHEMA,
            "verdict": {"development_games_authorized": True},
            "intersection": {"unique_canonical_count": 0},
            "counts": {"fresh_unique_canonical": len(self.fresh)},
            "references": {
                "protected_canonical_fingerprints": q.artifact_reference(
                    self.private_payload, bridge.exclusions.FINGERPRINT_SCHEMA
                )
            },
        })
        self.selection = self.root / "candidate/selection.json"
        q.write_sealed(self.selection, {
            "schema": bridge.adapter.v3.SELECTION_SCHEMA,
            "synthetic": True,
        })
        runtime_document = q.load_sealed(self.runtime)
        source_export = {
            "runtime_sha256": q.sha256_file(self.runtime),
            "runtime_body_sha256": runtime_document["body_sha256"],
            "model_header_sha256": "2" * 64,
            "source_sha256": q.sha256_file(self.generated),
            "source_ascii_bytes": self.generated.stat().st_size,
            "source_limit_exclusive": 95_000,
        }
        self.development_candidate = {
            "candidate_id": bridge.development.CANDIDATE_ID,
            "architecture": bridge.development.CAPACITY_ARCHITECTURE,
            "target": "search-target",
            "selection": q.artifact_reference(self.selection),
            "runtime": bridge.development._regular(self.runtime),
            "generated_source": bridge.development._regular(self.generated),
            "source_export": source_export,
            "runtime_identity": {
                "body_sha256": runtime_document["body_sha256"],
                "payload_sha256": "b" * 64,
            },
        }
        self.handoff = self.root / "handoff.json"
        q.write_sealed(self.handoff, {
            "schema": bridge.adapter.HANDOFF_SCHEMA,
            "candidate": {
                key: self.development_candidate[key]
                for key in (
                    "candidate_id", "architecture", "target", "selection",
                    "runtime", "generated_source", "source_export",
                )
            },
        })
        self.evaluation = self.root / "evaluation.json"
        q.write_sealed(self.evaluation, {
            "schema": bridge.adapter.EVALUATION_COMPLETION_SCHEMA,
            "synthetic": True,
        })
        self.adapter_plan = self.root / "adapter-plan.json"
        q.write_sealed(self.adapter_plan, {
            "schema": bridge.adapter.ADAPTER_PLAN_SCHEMA, "synthetic": True,
        })
        self.v3_plan = self.root / "v3-plan.json"
        q.write_sealed(self.v3_plan, {
            "schema": bridge.adapter.v3.PLAN_SCHEMA, "synthetic": True,
        })
        self.development_plan = self.root / "development-v3/plan.json"
        self.development_plan.parent.mkdir(parents=True, exist_ok=True)
        self.development_plan_value = q.write_sealed(self.development_plan, {
            "schema": bridge.development.PLAN_SCHEMA,
            "outputs": {
                "finalist_reference": str(
                    (self.root / "development-v3/finalist-reference.json").resolve()
                )
            },
            "candidate": self.development_candidate,
        })
        self.development_result = self.root / "development-v3/development-result.json"
        q.write_sealed(self.development_result, {
            "schema": bridge.development.RESULT_SCHEMA,
            "synthetic": True,
        })
        binary = self.root / "development-v3/candidate-binary"
        binary.write_bytes(b"synthetic candidate binary")
        finalist_directory = self.root / "development-v3/finalists"
        finalist_directory.mkdir()
        finalist_body = {
            "schema": bridge.development.FINALIST_SCHEMA,
            "namespace": bridge.NAMESPACE,
            "campaign_id": bridge.CAMPAIGN_ID,
            "status": "development-selected-awaiting-preflight-and-frozen-final",
            "created_at_utc": "2026-09-01T09:59:00Z",
            "development_plan": bridge.development._sealed_record(
                self.development_plan, bridge.development.PLAN_SCHEMA
            ),
            "development_result": bridge.development._sealed_record(
                self.development_result, bridge.development.RESULT_SCHEMA
            ),
            "adapter": {
                "handoff": bridge.development._sealed_record(
                    self.handoff, bridge.adapter.HANDOFF_SCHEMA
                ),
                "evaluation_completion": q.artifact_reference(
                    self.evaluation, bridge.adapter.EVALUATION_COMPLETION_SCHEMA
                ),
            },
            "exclusion": {
                "plan": bridge.development._sealed_record(
                    self.exclusion_plan, bridge.exclusions.PLAN_SCHEMA
                ),
                "receipt": bridge.development._sealed_record(
                    self.exclusion_receipt, bridge.exclusions.RECEIPT_SCHEMA
                ),
            },
            "candidate": self.development_candidate,
            "rank4_control": {"synthetic": True},
            "banks": {
                stage: bridge.development._regular(path)
                for stage, path in self.banks.items()
            },
            "binary": bridge.development._regular(binary),
            "tuple": ["0.95", "0.5", "1"],
            "tuple_candidate_id": "synthetic-tuple",
            "profile": "default",
            "profile_work": dict(bridge.campaign.PROFILE_ROSTER["default"]),
            "actual_clock": {
                "pairs": 200, "games": 400, "wins": 211,
                "color_wins": {"0": 104, "1": 107}, "failures": 0,
            },
            "run_receipts": [],
            "fresh_protected_tests_opened": True,
            "fresh_diagnostic_classification": "diagnostic-only-no-pass-fail-verdict",
            "old_protected_tests_accessed": False,
            "model_weights_immutable": True,
            "search_configuration_immutable": True,
            "development_selected": True,
            "preflight_required": True,
            "final_bank_generation_authorized": False,
            "rank4_gate_authorized": False,
            "upload_authorized": False,
        }
        sealed_finalist = q.seal(finalist_body)
        finalist_raw = q.canonical_json_bytes(sealed_finalist)
        self.finalist = finalist_directory / (
            q.sha256_bytes(finalist_raw) + ".finalist.json"
        )
        self.finalist.write_bytes(finalist_raw)
        self.finalist_reference = self.root / "development-v3/finalist-reference.json"
        q.write_sealed(self.finalist_reference, {
            "schema": bridge.development.FINALIST_REFERENCE_SCHEMA,
            "namespace": bridge.NAMESPACE,
            "campaign_id": bridge.CAMPAIGN_ID,
            "development_plan": bridge.development._sealed_record(
                self.development_plan, bridge.development.PLAN_SCHEMA
            ),
            "development_result": bridge.development._sealed_record(
                self.development_result, bridge.development.RESULT_SCHEMA
            ),
            "finalist": bridge.development._sealed_record(
                self.finalist, bridge.development.FINALIST_SCHEMA
            ),
            "complete": True,
            "final_bank_generation_authorized": False,
            "rank4_gate_authorized": False,
            "upload_authorized": False,
        })
        self.historical = []
        for index in range(7):
            path = self.root / f"historical-{index}.tsv"
            path.write_text(f"synthetic-{index}\n", encoding="ascii")
            self.historical.append(path)
        self.historical_loaded = {
            "fingerprints": ["f" * 64],
            "sources": [{"synthetic": True}],
            "body_sha256": "e" * 64,
        }
        self.real_development_validate_finalist = (
            bridge.development.validate_finalist
        )

    @contextlib.contextmanager
    def real_finalist_validation(self):
        banks = {
            stage: bridge._record(bank) for stage, bank in self.banks.items()
        }

        def validate(reference_path, *, plan_path, output_root):
            with mock.patch.object(
                bridge.development, "finalize_result", return_value=self.finalist
            ):
                return self.real_development_validate_finalist(
                    reference_path,
                    plan_path=plan_path,
                    output_root=output_root,
                    plan_validator=lambda *_args, **_kwargs: self.development_plan_value,
                )

        with (
            mock.patch.object(
                bridge.development, "validate_finalist", side_effect=validate
            ),
            mock.patch.object(
                bridge.exclusions, "validate_receipt",
                return_value={"development_ready": True},
            ),
            mock.patch.object(
                bridge.exclusions, "require_development_roster",
                return_value=banks,
            ),
        ):
            yield

    def preflight_validator(self, **kwargs):
        return {
            "commit": COMMIT,
            "candidate": bridge._record(self.candidate, ascii_required=True),
            "runtime": bridge._record(self.runtime, ascii_required=True),
            "preflight": q.artifact_reference(
                self.preflight, bridge.preflight_tools.RECEIPT_SCHEMA
            ),
            "gate": bridge._record(self.gate, executable=True),
            "git": {"commit": COMMIT, "tracked_clean": True},
            "uncontended_timing": {"first_max_ms": 100.0, "later_max_ms": 20.0},
        }

    def historical_validator(self, paths):
        if [path.resolve() for path in paths] != [
            path.resolve() for path in self.historical
        ]:
            raise bridge.BridgeError("synthetic historical roster redirected")
        return {"paths": [path.resolve() for path in paths],
                "loaded": self.historical_loaded}

    def fingerprint_loader(self, **_kwargs):
        return frozenset(self.fresh)

    def prepare(self):
        with self.real_finalist_validation():
            return bridge.prepare(
                campaign_root=self.root,
                development_plan_path=self.development_plan,
                finalist_reference_path=self.finalist_reference,
                preflight_path=self.preflight,
                candidate_source=self.candidate,
                historical_paths=self.historical,
                rank4_source=RANK4,
                gate_path=self.gate,
                repository=self.repository,
                authorized_at_utc="2026-09-01T10:00:00Z",
                planned_at_utc="2026-09-01T10:01:00Z",
                preflight_validator=self.preflight_validator,
                historical_validator=self.historical_validator,
                git_verifier=lambda *_args: {
                    "commit": COMMIT, "tracked_clean": True
                },
            )

    def materialize(self, entropy=lambda count: b"s" * count):
        with self.real_finalist_validation():
            return bridge.materialize_bank(
                plan_path=self.root / bridge.BRIDGE_DIRECTORY / "plan.json",
                campaign_root=self.root,
                development_plan_path=self.development_plan,
                finalist_reference_path=self.finalist_reference,
                preflight_path=self.preflight,
                candidate_source=self.candidate,
                historical_paths=self.historical,
                rank4_source=RANK4,
                gate_path=self.gate,
                repository=self.repository,
                claimed_at_utc="2026-09-01T10:02:00Z",
                entropy=entropy,
                preflight_validator=self.preflight_validator,
                historical_validator=self.historical_validator,
                fingerprint_loader=self.fingerprint_loader,
                git_verifier=lambda *_args: {
                    "commit": COMMIT, "tracked_clean": True
                },
            )

    def run(self, *, wins=1000):
        targets = (
            {0: 260, 1: 267} if wins == 527
            else {0: 263, 1: 263} if wins == 526
            else {0: wins // 2, 1: wins - wins // 2}
        )

        def games(index):
            result = []
            for offset in range(5):
                for color in (0, 1):
                    pair_index = index * 5 + offset
                    result.append({
                        "pair_index": pair_index,
                        "candidate_color": color,
                        "candidate_win": pair_index < targets[color],
                        "turns": 20,
                        "failure": None,
                        "first_ms": 100.0,
                        "later_max_ms": 20.0,
                    })
            return result

        with self.real_finalist_validation():
            return bridge.run_final(
                plan_path=self.root / bridge.BRIDGE_DIRECTORY / "plan.json",
                bank_receipt_path=self.root / bridge.BRIDGE_DIRECTORY
                / "bank-receipt.json",
                campaign_root=self.root,
                development_plan_path=self.development_plan,
                finalist_reference_path=self.finalist_reference,
                preflight_path=self.preflight,
                candidate_source=self.candidate,
                historical_paths=self.historical,
                rank4_source=RANK4,
                gate_path=self.gate,
                repository=self.repository,
                launched_at_utc="2026-09-01T10:03:00Z",
                preflight_validator=self.preflight_validator,
                historical_validator=self.historical_validator,
                fingerprint_loader=self.fingerprint_loader,
                git_verifier=lambda *_args: {
                    "commit": COMMIT, "tracked_clean": True
                },
                runner=lambda spec: spec["index"],
                result_adapter=lambda raw, plan, bank, index: games(index),
                clock=lambda: "2026-09-01T10:04:00Z",
            )


class DiscreteV3FinalBridgeTest(unittest.TestCase):
    def test_maintained_strict_boundary_is_exact_527_260_and_zero_failures(self):
        summary = {
            "games": 1_000,
            "candidate_wins": 527,
            "candidate_color_wins": {"0": 260, "1": 267},
            "failures": {
                name: 0 for name in q.FAILURE_CATEGORIES
            },
            "maximum_turns": 320,
            "timing": {"first_max_ms": 999.0, "later_max_ms": 199.0},
            "uncontended_timing": {
                "first_max_ms": 899.0, "later_max_ms": 179.0,
            },
        }
        self.assertTrue(q.strict_gate_verdict(summary)["passed"])
        for mutation in ("wins", "color", "failure"):
            changed = {
                **summary,
                "candidate_color_wins": dict(summary["candidate_color_wins"]),
                "failures": dict(summary["failures"]),
            }
            if mutation == "wins":
                changed["candidate_wins"] = 526
            elif mutation == "color":
                changed["candidate_color_wins"]["0"] = 259
            else:
                changed["failures"][q.FAILURE_CATEGORIES[0]] = 1
            self.assertFalse(q.strict_gate_verdict(changed)["passed"])

    def test_prepare_is_precommitted_resumable_and_has_no_upload_authority(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary))
            plan = fixture.prepare()
            self.assertEqual(plan, fixture.prepare())
            value = q.load_sealed(plan, bridge.PLAN_SCHEMA)
            auth = q.load_sealed(
                fixture.root / bridge.BRIDGE_DIRECTORY / "authorization.json",
                bridge.AUTHORIZATION_SCHEMA,
            )
            self.assertEqual(auth["uploads_authorized"], 0)
            self.assertFalse(value["policy"]["upload_authorized"])
            self.assertFalse(
                (fixture.root / bridge.BRIDGE_DIRECTORY / "bank-claim.json").exists()
            )
            self.assertNotIn(next(iter(fixture.fresh)), plan.read_text(encoding="ascii"))

    def test_source_byte_mismatch_and_finalist_drift_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary))
            fixture.candidate.write_text("different\n", encoding="ascii")
            with self.assertRaisesRegex(bridge.BridgeError, "differs"):
                fixture.prepare()
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary))
            fixture.prepare()
            fixture.finalist.write_text("{}\n", encoding="ascii")
            with self.assertRaises(bridge.BridgeError):
                fixture.materialize()

    def test_materialize_secret_500_bank_excludes_all_unions_and_resumes(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary))
            fixture.prepare()
            calls = 0

            def entropy(count):
                nonlocal calls
                calls += 1
                return b"z" * count

            receipt_path = fixture.materialize(entropy)
            self.assertEqual(calls, 1)
            self.assertEqual(receipt_path, fixture.materialize(entropy))
            self.assertEqual(calls, 1)
            receipt = q.load_sealed(receipt_path, bridge.BANK_RECEIPT_SCHEMA)
            self.assertEqual(receipt["opening_count"], 500)
            self.assertEqual(receipt["four_way_overlap_count"], 0)
            self.assertFalse(receipt["upload_authorized"])
            bank = opening_tools.validate_bank(
                pathlib.Path(receipt["protected_bank"]["path"])
            )
            self.assertEqual(bank["opening_count"], 500)
            self.assertFalse(
                fixture.fresh.intersection(
                    opening["fingerprints"]["canonical"]
                    for opening in bank["openings"]
                )
            )

    def test_active_bank_lock_and_protected_symlink_reject_before_entropy(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary))
            plan = fixture.prepare()
            plan_value = q.load_sealed(plan, bridge.PLAN_SCHEMA)
            lock = pathlib.Path(plan_value["paths"]["bank_lock"])
            descriptor = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(bridge.BridgeError, "active"):
                    fixture.materialize()
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)

        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary))
            plan = fixture.prepare()
            plan_value = q.load_sealed(plan, bridge.PLAN_SCHEMA)
            external = fixture.root / "external"
            external.mkdir()
            pathlib.Path(plan_value["paths"]["protected"]).symlink_to(
                external, target_is_directory=True
            )
            calls = 0

            def entropy(count):
                nonlocal calls
                calls += 1
                return b"x" * count

            with self.assertRaisesRegex(bridge.BridgeError, "unsafe"):
                fixture.materialize(entropy)
            self.assertEqual(calls, 0)

    def test_every_nested_protected_route_must_be_pristine_before_claim_entropy(self):
        routes = {
            "opening-bank": ("protected/opening-bank", True),
            "gate-bank": ("protected/gate-bank", True),
            "bank-adapter": ("protected-bank-adapter.json", False),
            "gate-binding": ("gate-binding.json", False),
        }
        for route, (relative, directory_route) in routes.items():
            for attack in ("symlink", "file", "nonempty-directory"):
                with (
                    self.subTest(route=route, attack=attack),
                    tempfile.TemporaryDirectory() as temporary,
                ):
                    fixture = Fixture(pathlib.Path(temporary))
                    plan_path = fixture.prepare()
                    plan = q.load_sealed(plan_path, bridge.PLAN_SCHEMA)
                    bridge_root = pathlib.Path(plan["paths"]["bridge_root"])
                    target = bridge_root / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    external = fixture.root / f"external-{route}-{attack}"
                    if attack == "symlink":
                        if directory_route:
                            external.mkdir()
                        else:
                            external.write_text("external\n")
                        target.symlink_to(
                            external, target_is_directory=directory_route
                        )
                    elif attack == "file":
                        target.write_text("foreign\n")
                    else:
                        target.mkdir()
                        (target / "foreign").write_text("foreign\n")
                    calls = 0

                    def entropy(count):
                        nonlocal calls
                        calls += 1
                        return b"x" * count

                    with self.assertRaisesRegex(
                        bridge.BridgeError, "unsafe|pristine"
                    ):
                        fixture.materialize(entropy)
                    self.assertEqual(calls, 0)
                    self.assertFalse(
                        pathlib.Path(plan["paths"]["bank_claim"]).exists()
                    )

    def test_spent_bank_claim_without_receipt_is_terminal(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary))
            plan_path = fixture.prepare()
            plan = q.load_sealed(plan_path, bridge.PLAN_SCHEMA)
            claim = pathlib.Path(plan["paths"]["bank_claim"])
            q.write_sealed(
                claim,
                bridge._bank_claim_body(
                    plan_path, plan, "2026-09-01T10:02:00Z"
                ),
            )
            with self.assertRaisesRegex(bridge.BridgeError, "spent"):
                fixture.materialize()

    def test_strict_rank4_pass_emits_v3_qualified_without_upload_and_resumes(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary))
            fixture.prepare()
            fixture.materialize()
            aggregate = fixture.run(wins=527)
            self.assertTrue(aggregate["verdict"]["passed"])
            self.assertEqual(aggregate["summary"]["candidate_wins"], 527)
            self.assertEqual(aggregate["summary"]["candidate_color_wins"]["0"], 260)
            qualified_path = (
                fixture.root / bridge.BRIDGE_DIRECTORY / "ledger/v3-qualified-inputs.json"
            )
            qualified = q.load_sealed(qualified_path, bridge.QUALIFIED_SCHEMA)
            self.assertEqual(qualified["uploads_authorized"], 0)
            self.assertEqual(qualified["strict_thresholds"]["candidate_wins_min"], 527)
            self.assertNotIn(
                next(iter(fixture.fresh)), qualified_path.read_text(encoding="ascii")
            )
            self.assertEqual(aggregate, fixture.run(wins=527))

            raw = fixture.root / bridge.BRIDGE_DIRECTORY / "ledger/raw/shard-000.json"
            raw.write_text("tampered\n", encoding="ascii")
            with self.assertRaisesRegex(bridge.BridgeError, "changed"):
                fixture.run(wins=527)

    def test_526_wins_fails_and_never_emits_qualified_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary))
            fixture.prepare()
            fixture.materialize()
            aggregate = fixture.run(wins=526)
            self.assertFalse(aggregate["verdict"]["passed"])
            self.assertFalse(
                (fixture.root / bridge.BRIDGE_DIRECTORY
                 / "ledger/v3-qualified-inputs.json").exists()
            )

    def test_bridge_root_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary))
            external = fixture.root / "external"
            external.mkdir()
            (fixture.root / bridge.BRIDGE_DIRECTORY).symlink_to(
                external, target_is_directory=True
            )
            with self.assertRaisesRegex(bridge.BridgeError, "symlink"):
                fixture.prepare()


if __name__ == "__main__":
    unittest.main()
