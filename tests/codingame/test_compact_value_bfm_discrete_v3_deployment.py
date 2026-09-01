import json
import hashlib
import pathlib
import tempfile
import unittest

from tools import compact_value_bfm_discrete_v3_deployment as deployment


ROOT = pathlib.Path(__file__).resolve().parents[2]


BASE = b"""\
inline constexpr std::size_t kRootPartialPaths = 4'000;
inline constexpr std::size_t kNonrootPartialPaths = 512;
inline constexpr std::size_t kProductionTreeNodes = 80'000;
inline constexpr double kExploration = 0.95;
inline constexpr double kFirstPlayUrgency = 0.5;
inline constexpr double kFinalVisitWeight = 1.0;
std::uint64_t shuffle_seed{0x6a09e667f3bcc909ULL};
MODEL_PAYLOAD_MUST_NOT_CHANGE
"""


class DiscreteV3DeploymentSourceTest(unittest.TestCase):
    def test_fixed_repository_candidate_and_manifest_are_both_absent_or_valid(self):
        candidate = ROOT / deployment.CANDIDATE_RELATIVE
        manifest = ROOT / deployment.MANIFEST_RELATIVE
        self.assertEqual(candidate.exists(), manifest.exists())
        self.assertFalse(candidate.is_symlink())
        self.assertFalse(manifest.is_symlink())
        if candidate.exists():
            value = deployment.verify_manifest_file(manifest, candidate)
            self.assertEqual(
                value["deployed_source"]["sha256"],
                hashlib.sha256(candidate.read_bytes()).hexdigest(),
            )

    def test_nondefault_derivation_changes_only_seven_exact_declarations(self):
        work = deployment.PROFILE_ROSTER["heavy"]
        result = deployment.derive_source(
            BASE, search_tuple=("0.95", "0.75", "1"),
            profile="heavy", work=work,
        )
        self.assertIn(b"kRootPartialPaths = 8'000;", result)
        self.assertIn(b"kNonrootPartialPaths = 512;", result)
        self.assertIn(b"kProductionTreeNodes = 120'000;", result)
        self.assertIn(b"kExploration = 0.95;", result)
        self.assertIn(b"kFirstPlayUrgency = 0.75;", result)
        self.assertIn(b"kFinalVisitWeight = 1.0;", result)
        self.assertIn(b"shuffle_seed{1ULL};", result)
        self.assertIn(b"MODEL_PAYLOAD_MUST_NOT_CHANGE", result)
        record = deployment.attest_derivation(
            BASE, result, search_tuple=("0.95", "0.75", "1"),
            profile="heavy", work=work,
        )
        self.assertEqual(record["replacement_slots"], 7)
        self.assertEqual(record["configuration"]["candidate_shuffle_seed"], 1)
        self.assertNotEqual(
            record["base_source"]["sha256"],
            record["deployed_source"]["sha256"],
        )

    def test_default_tuple_and_profile_still_require_seed_one_derivative(self):
        work = deployment.PROFILE_ROSTER["default"]
        result = deployment.derive_source(
            BASE, search_tuple=("0.95", "0.5", "1"),
            profile="default", work=work,
        )
        self.assertNotEqual(result, BASE)
        self.assertIn(b"shuffle_seed{1ULL};", result)
        with self.assertRaisesRegex(
            deployment.DeploymentSourceError, "exact configured"
        ):
            deployment.attest_derivation(
                BASE, BASE, search_tuple=("0.95", "0.5", "1"),
                profile="default", work=work,
            )

    def test_any_extra_change_missing_slot_or_roster_drift_fails_closed(self):
        work = deployment.PROFILE_ROSTER["light"]
        result = deployment.derive_source(
            BASE, search_tuple=("0.65", "0.5", "1"),
            profile="light", work=work,
        )
        with self.assertRaisesRegex(
            deployment.DeploymentSourceError, "exact configured"
        ):
            deployment.attest_derivation(
                BASE, result + b"// tamper\n",
                search_tuple=("0.65", "0.5", "1"),
                profile="light", work=work,
            )
        with self.assertRaisesRegex(
            deployment.DeploymentSourceError, "exactly one"
        ):
            deployment.derive_source(
                BASE.replace(b"kExploration", b"kChanged"),
                search_tuple=("0.65", "0.5", "1"),
                profile="light", work=work,
            )
        with self.assertRaisesRegex(deployment.DeploymentSourceError, "profile"):
            deployment.derive_source(
                BASE, search_tuple=("0.65", "0.5", "1"),
                profile="light", work=deployment.PROFILE_ROSTER["heavy"],
            )
        with self.assertRaisesRegex(deployment.DeploymentSourceError, "tuple"):
            deployment.derive_source(
                BASE, search_tuple=("0.70", "0.5", "1"),
                profile="light", work=work,
            )

    def test_manifest_recovers_base_and_rejects_every_identity_drift(self):
        selected_tuple = ("0.95", "0.75", "1")
        profile = "heavy"
        work = deployment.PROFILE_ROSTER[profile]
        candidate = deployment.derive_source(
            BASE, search_tuple=selected_tuple, profile=profile, work=work
        )
        manifest = deployment.create_manifest(
            BASE, candidate, search_tuple=selected_tuple,
            profile=profile, work=work,
        )
        self.assertEqual(deployment.validate_manifest(manifest, candidate), manifest)
        self.assertEqual(
            deployment.recover_generated_source(
                candidate, search_tuple=selected_tuple,
                profile=profile, work=work,
            ),
            BASE,
        )
        for mutation in ("candidate", "base", "seed", "tuple"):
            changed = json.loads(json.dumps(manifest))
            source = candidate
            if mutation == "candidate":
                source += b"\n"
            elif mutation == "base":
                changed["base_source"]["sha256"] = "0" * 64
            elif mutation == "seed":
                source = source.replace(b"shuffle_seed{1ULL}", b"shuffle_seed{2ULL}")
            else:
                changed["configuration"]["tuple"][0] = "0.65"
            with self.subTest(mutation=mutation), self.assertRaises(
                deployment.DeploymentSourceError
            ):
                deployment.validate_manifest(changed, source)
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            manifest_path = root / "manifest.json"
            candidate_path = root / "candidate.cpp"
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")),
                encoding="ascii",
            )
            candidate_path.write_bytes(candidate)
            self.assertEqual(
                deployment.verify_manifest_file(manifest_path, candidate_path),
                manifest,
            )

if __name__ == "__main__":
    unittest.main()
