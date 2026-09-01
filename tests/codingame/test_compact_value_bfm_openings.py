import copy
import importlib.util
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "compact_value_bfm_openings.py"
SPEC = importlib.util.spec_from_file_location("compact_value_bfm_openings", TOOL)
openings = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(openings)
base = openings.base

RANK4 = ROOT / "submissions/codingame/bots/rank_4/submission.cpp"
COMMIT = "c" * 40
EXAMPLE = "5/2/2/0/1/4/1/17/6/0/75"
EXAMPLE_12 = "3/1/6/7/5/3/4/2/2/2/0/0"


def write_tsv(path, transcript=EXAMPLE, opening_id="copied-0"):
    path.write_text(
        "# papersoccer.jacek-replay-bfm-opening-bank.v1\n"
        "# rules=8x10;own-goals-allowed;mover-loses\n"
        "# classification=development\n"
        "# seed=123\n"
        "# minimum-physical-plies=12\n"
        "opening_id\ttranscript\tstate_identity\n"
        f"{opening_id}\t{transcript}\topaque:identity\n",
        encoding="ascii",
    )
    return path


def seven_banks(root):
    paths = []
    for index in range(7):
        path = root / f"bank-{index:03d}.tsv"
        write_tsv(path, EXAMPLE if index % 2 == 0 else EXAMPLE_12,
                  f"copied-{index}")
        paths.append(path)
    return paths


class ProtectedFixture:
    def __init__(self, root):
        self.root = root
        self.candidate = root / "candidate.cpp"
        self.candidate.write_text("int main(){return 0;}\n", encoding="ascii")
        self.source_binding = root / "source-binding.json"
        base.create_source_binding(
            self.source_binding,
            candidate_source=self.candidate,
            candidate_commit=COMMIT,
            rank4_source=RANK4,
            opponent_source=RANK4,
        )
        source = base.load_sealed(
            self.source_binding, base.SOURCE_BINDING_SCHEMA
        )
        inputs = {
            "candidate_commit": source["candidate_commit"],
            "candidate": source["candidate"],
        }
        preflight = base.seal({
            "schema": openings.PREFLIGHT_SCHEMA,
            "namespace": openings.NAMESPACE,
            "status": "passed",
            "inputs_before": inputs,
            "inputs_after": inputs,
            "checks": {"clean-source": "passed"},
            "protected_banks_accessed": [],
            "git_writes": 0,
            "uploads": 0,
        })
        raw = base.canonical_json_bytes(preflight)
        self.clean_binding = root / f"{base.sha256_bytes(raw)}.json"
        self.clean_binding.write_bytes(raw)
        self.exclusions = seven_banks(root)
        generated = openings.generate_development_banks(
            root / "development", exclusion_paths=self.exclusions
        )
        self.development = [generated[stage] for stage in openings.DEVELOPMENT_ORDER]


class ReplayAndFingerprintTest(unittest.TestCase):
    def test_copied_complete_turn_overshoot_replays_nonterminal(self):
        state, primitive_plies = openings.replay_transcript(EXAMPLE)
        self.assertEqual(primitive_plies, 13)
        self.assertIsNone(state.winner)
        self.assertEqual(state.ball, (4, 2))
        with tempfile.TemporaryDirectory() as temporary:
            path = write_tsv(pathlib.Path(temporary) / "bank.tsv")
            bank = openings.load_exclusion_bank(path)
            self.assertEqual(bank["openings"][0]["primitive_plies"], 13)
            self.assertEqual(bank["openings"][0]["completed_turn_overshoot"], 1)

    def test_flat_incomplete_or_shallow_transcripts_fail_closed(self):
        for transcript in (
            EXAMPLE.replace("/", ""),
            "5//2/2/0/1/4/1/17/6/0/75",
            "3/1",
            EXAMPLE[:-1],
        ):
            with self.subTest(transcript=transcript), self.assertRaises(
                openings.OpeningError
            ):
                openings.replay_transcript(transcript)

    def test_four_way_fingerprint_is_invariant_as_a_set(self):
        state, _ = openings.replay_transcript(EXAMPLE)
        original = openings.state_fingerprints(state)
        original_set = {
            value for name, value in original.items() if name != "canonical"
        }
        for rotate, reflect in (
            (False, False), (True, False), (False, True), (True, True)
        ):
            transformed = openings.transform_state(
                state, rotate=rotate, reflect=reflect
            )
            result = openings.state_fingerprints(transformed)
            self.assertEqual(result["canonical"], original["canonical"])
            self.assertEqual(
                {value for name, value in result.items() if name != "canonical"},
                original_set,
            )

    def test_exactly_seven_distinct_paths_are_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = seven_banks(pathlib.Path(temporary))
            loaded = openings.load_all_exclusions(paths)
            self.assertEqual(len(loaded["sources"]), 7)
            self.assertGreaterEqual(len(loaded["fingerprints"]), 4)
            for invalid in (paths[:6], [*paths[:6], paths[0]]):
                with self.assertRaisesRegex(openings.OpeningError, "exactly seven"):
                    openings.load_all_exclusions(invalid)


class DevelopmentGenerationTest(unittest.TestCase):
    def test_stage_roster_and_default_counts_are_exact(self):
        self.assertEqual(
            openings.DEVELOPMENT_COUNTS,
            {
                "model_screen": 100,
                "tuple_screen": 100,
                "tuple_confirmation": 250,
                "profile_screen": 100,
                "profile_confirmation": 250,
                "actual_clock": 200,
            },
        )

    def test_small_injected_run_is_deterministic_and_cross_stage_disjoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            exclusion_paths = seven_banks(root)
            counts = {stage: 3 for stage in openings.DEVELOPMENT_ORDER}
            first = openings._generate_development_banks(
                root / "first", exclusion_paths=exclusion_paths, counts=counts
            )
            second = openings._generate_development_banks(
                root / "second", exclusion_paths=exclusion_paths, counts=counts
            )
            excluded = set(
                openings.load_all_exclusions(exclusion_paths)["fingerprints"]
            )
            seen = set(excluded)
            for stage in openings.DEVELOPMENT_ORDER:
                self.assertEqual(base.sha256_file(first[stage]),
                                 base.sha256_file(second[stage]))
                bank = openings.validate_bank(first[stage])
                self.assertEqual(bank["opening_count"], 3)
                self.assertEqual(bank["classification"],
                                 "unprotected-development")
                for opening in bank["openings"]:
                    self.assertIn("/", opening["transcript"])
                    self.assertGreaterEqual(opening["primitive_plies"], 12)
                    state, count = openings.replay_transcript(
                        opening["transcript"]
                    )
                    self.assertIsNone(state.winner)
                    self.assertEqual(count, opening["primitive_plies"])
                    variants = {
                        value for name, value in opening["fingerprints"].items()
                        if name != "canonical"
                    }
                    self.assertFalse(variants & seen)
                    seen.update(variants)

    def test_content_address_and_body_tampering_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            banks = openings._generate_development_banks(
                root / "out", exclusion_paths=seven_banks(root),
                counts={stage: 1 for stage in openings.DEVELOPMENT_ORDER},
            )
            path = banks["model_screen"]
            openings.validate_bank(path)
            raw = bytearray(path.read_bytes())
            raw[-3] = ord("0") if raw[-3] != ord("0") else ord("1")
            path.write_bytes(raw)
            with self.assertRaises(openings.OpeningError):
                openings.validate_bank(path)


class ProtectedGenerationTest(unittest.TestCase):
    def test_entropy_is_requested_only_after_clean_binding_and_exclusions(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ProtectedFixture(pathlib.Path(temporary))
            fixture.exclusions[0].write_text("invalid\n", encoding="ascii")
            calls = []
            with self.assertRaises(openings.OpeningError):
                openings.create_protected_seed_receipt(
                    fixture.root / "seed.json",
                    source_binding_path=fixture.source_binding,
                    clean_binding_path=fixture.clean_binding,
                    exclusion_paths=fixture.exclusions,
                    development_bank_paths=fixture.development,
                    created_at_utc="2026-08-31T21:00:00Z",
                    entropy=lambda count: calls.append(count) or b"x" * count,
                )
            self.assertEqual(calls, [])

    def test_injected_seed_generates_exact_500_and_binds_launch_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ProtectedFixture(pathlib.Path(temporary))
            seed_path = fixture.root / "protected-seed.json"
            receipt = openings.create_protected_seed_receipt(
                seed_path,
                source_binding_path=fixture.source_binding,
                clean_binding_path=fixture.clean_binding,
                exclusion_paths=fixture.exclusions,
                development_bank_paths=fixture.development,
                created_at_utc="2026-08-31T21:00:00Z",
                entropy=lambda count: b"p" * count,
            )
            self.assertEqual(receipt["entropy_bits"], 256)
            self.assertFalse(receipt["bank_generated"])
            self.assertEqual(len(receipt["exclusion_sources"]), 13)
            path = openings.generate_protected_bank(
                fixture.root / "protected",
                seed_receipt_path=seed_path,
                exclusion_paths=fixture.exclusions,
                development_bank_paths=fixture.development,
            )
            bank = openings.validate_bank(path)
            self.assertEqual(bank["opening_count"], 500)
            self.assertEqual(bank["classification"], "protected-final")
            self.assertTrue(bank["bank_consumed_at_launch_policy"][
                "exclusive_marker_required_before_first_game"
            ])
            self.assertFalse(any(
                candidate.name == "bank-consumed.json"
                for candidate in fixture.root.rglob("*")
            ))
            excluded = set(openings.load_protected_exclusions(
                fixture.exclusions, fixture.development
            )["fingerprints"])
            for opening in bank["openings"]:
                variants = {
                    value for name, value in opening["fingerprints"].items()
                    if name != "canonical"
                }
                self.assertFalse(variants & excluded)

    def test_seed_receipt_is_immutable_and_exclusion_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ProtectedFixture(pathlib.Path(temporary))
            seed_path = fixture.root / "seed.json"
            openings.create_protected_seed_receipt(
                seed_path,
                source_binding_path=fixture.source_binding,
                clean_binding_path=fixture.clean_binding,
                exclusion_paths=fixture.exclusions,
                development_bank_paths=fixture.development,
                created_at_utc="2026-08-31T21:00:00Z",
                entropy=lambda count: b"q" * count,
            )
            with self.assertRaisesRegex(openings.OpeningError, "collision"):
                openings.create_protected_seed_receipt(
                    seed_path,
                    source_binding_path=fixture.source_binding,
                    clean_binding_path=fixture.clean_binding,
                    exclusion_paths=fixture.exclusions,
                    development_bank_paths=fixture.development,
                    created_at_utc="2026-08-31T21:01:00Z",
                    entropy=lambda count: b"r" * count,
                )
            fixture.exclusions[-1].write_text(
                fixture.exclusions[-1].read_text().replace(EXAMPLE, EXAMPLE_12),
                encoding="ascii",
            )
            with self.assertRaisesRegex(openings.OpeningError, "binding changed"):
                openings.generate_protected_bank(
                    fixture.root / "protected",
                    seed_receipt_path=seed_path,
                    exclusion_paths=fixture.exclusions,
                    development_bank_paths=fixture.development,
                )

    def test_dirty_or_mismatched_clean_binding_rejects_before_entropy(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ProtectedFixture(pathlib.Path(temporary))
            clean = base.load_sealed(fixture.clean_binding)
            body = {key: value for key, value in clean.items()
                    if key != "body_sha256"}
            body["status"] = "failed"
            fixture.clean_binding.unlink()
            base.write_sealed(fixture.clean_binding, body)
            calls = []
            with self.assertRaisesRegex(openings.OpeningError, "clean source"):
                openings.create_protected_seed_receipt(
                    fixture.root / "seed.json",
                    source_binding_path=fixture.source_binding,
                    clean_binding_path=fixture.clean_binding,
                    exclusion_paths=fixture.exclusions,
                    development_bank_paths=fixture.development,
                    created_at_utc="2026-08-31T21:00:00Z",
                    entropy=lambda count: calls.append(count) or b"x" * count,
                )
            self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
