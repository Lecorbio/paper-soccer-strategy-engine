import json
import hashlib
import io
import pathlib
import stat
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

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

COMPACT_MACRO_PREFIXED = b"""\
#define COMPACT_VALUE_BFM_REFERENCE_DESCENDANT_SORT 1
inline constexpr std::size_t kRootPartialPaths=4'000;inline constexpr std::size_t kNonrootPartialPaths=512;inline constexpr std::size_t kProductionTreeNodes=80'000;inline constexpr double kExploration=0.95;inline constexpr double kFirstPlayUrgency=0.5;inline constexpr double kFinalVisitWeight=1.0;std::uint64_t shuffle_seed{0x6a09e667f3bcc909ULL};MODEL_PAYLOAD_MUST_NOT_CHANGE
"""


class DiscreteV3DeploymentSourceTest(unittest.TestCase):
    @staticmethod
    def _expected_pair(selected_tuple=("0.95", "0.75", "1"), profile="heavy"):
        work = deployment.PROFILE_ROSTER[profile]
        candidate = deployment.derive_source(
            BASE, search_tuple=selected_tuple, profile=profile, work=work,
        )
        manifest = deployment.create_manifest(
            BASE, candidate, search_tuple=selected_tuple,
            profile=profile, work=work,
        )
        manifest_payload = json.dumps(
            manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        ).encode("ascii")
        return candidate, manifest, manifest_payload

    @staticmethod
    def _generated_source(root):
        source = root / "generated.cpp"
        source.write_bytes(BASE)
        return source

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

    def test_compacted_macro_prefixed_source_uses_same_strict_seven_slots(self):
        selected_tuple = ("0.95", "0.75", "1")
        profile = "light"
        work = deployment.PROFILE_ROSTER[profile]
        deployed = deployment.derive_source(
            COMPACT_MACRO_PREFIXED, search_tuple=selected_tuple,
            profile=profile, work=work,
        )
        self.assertTrue(deployed.startswith(
            b"#define COMPACT_VALUE_BFM_REFERENCE_DESCENDANT_SORT 1\n"
        ))
        self.assertIn(b"kRootPartialPaths=2'000;", deployed)
        self.assertIn(b"kNonrootPartialPaths=256;", deployed)
        self.assertIn(b"kProductionTreeNodes=60'000;", deployed)
        self.assertIn(b"kExploration=0.95;", deployed)
        self.assertIn(b"kFirstPlayUrgency=0.75;", deployed)
        self.assertIn(b"kFinalVisitWeight=1.0;", deployed)
        self.assertIn(b"shuffle_seed{1ULL};", deployed)
        self.assertEqual(
            deployment.recover_generated_source(
                deployed, search_tuple=selected_tuple,
                profile=profile, work=work,
            ),
            COMPACT_MACRO_PREFIXED,
        )
        manifest = deployment.create_manifest(
            COMPACT_MACRO_PREFIXED, deployed,
            search_tuple=selected_tuple, profile=profile, work=work,
        )
        self.assertEqual(deployment.validate_manifest(manifest, deployed), manifest)

    def test_compacted_source_duplicate_or_wrong_default_still_fails_closed(self):
        work = deployment.PROFILE_ROSTER["default"]
        duplicate = COMPACT_MACRO_PREFIXED + (
            b"inline constexpr double kExploration = 0.95;\n"
        )
        wrong = COMPACT_MACRO_PREFIXED.replace(
            b"kFirstPlayUrgency=0.5", b"kFirstPlayUrgency=0.5001"
        )
        for source in (duplicate, wrong):
            with self.subTest(source=source[-80:]), self.assertRaisesRegex(
                deployment.DeploymentSourceError, "exactly one frozen default"
            ):
                deployment.derive_source(
                    source, search_tuple=("0.95", "0.5", "1"),
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
            root = pathlib.Path(temporary).resolve()
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

    def test_materialize_cli_writes_only_fixed_canonical_pair_and_identities(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary).resolve()
            generated = self._generated_source(root)
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                status = deployment.main([
                    "materialize", "--repository", str(root),
                    "--generated-source", str(generated),
                    "--tuple", "0.95", "0.75", "1", "--profile", "heavy",
                ])
            self.assertEqual(status, 0)
            output = json.loads(stdout.getvalue())
            candidate_path = root / deployment.CANDIDATE_RELATIVE
            manifest_path = root / deployment.MANIFEST_RELATIVE
            candidate, manifest, manifest_payload = self._expected_pair()
            self.assertEqual(candidate_path.read_bytes(), candidate)
            self.assertEqual(manifest_path.read_bytes(), manifest_payload)
            self.assertEqual(
                deployment.verify_manifest_file(manifest_path, candidate_path), manifest
            )
            self.assertEqual(output["candidate"]["path"], str(candidate_path.resolve()))
            self.assertEqual(output["candidate"]["sha256"], hashlib.sha256(candidate).hexdigest())
            self.assertEqual(output["manifest"]["body_sha256"], manifest["body_sha256"])
            self.assertTrue(output["candidate"]["created"])
            self.assertTrue(output["manifest"]["created"])
            self.assertEqual(candidate_path.stat().st_mode & 0o777, 0o444)
            self.assertEqual(manifest_path.stat().st_mode & 0o777, 0o444)

    def test_materialize_idempotently_adopts_exact_existing_pair(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary).resolve()
            generated = self._generated_source(root)
            first = deployment.materialize_deployment(
                root, generated, search_tuple=("0.95", "0.75", "1"), profile="heavy"
            )
            candidate_path = root / deployment.CANDIDATE_RELATIVE
            manifest_path = root / deployment.MANIFEST_RELATIVE
            before = (candidate_path.stat().st_ino, manifest_path.stat().st_ino)
            second = deployment.materialize_deployment(
                root, generated, search_tuple=("0.95", "0.75", "1"), profile="heavy"
            )
            self.assertEqual(
                before, (candidate_path.stat().st_ino, manifest_path.stat().st_ino)
            )
            self.assertTrue(first["candidate"]["created"])
            self.assertTrue(first["manifest"]["created"])
            self.assertFalse(second["candidate"]["created"])
            self.assertFalse(second["manifest"]["created"])
            self.assertEqual(first["candidate"]["sha256"], second["candidate"]["sha256"])
            self.assertEqual(first["manifest"]["sha256"], second["manifest"]["sha256"])

    def test_materialize_rejects_mismatched_preexistence_before_other_write(self):
        for conflicted in ("candidate", "manifest"):
            with self.subTest(conflicted=conflicted), tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary).resolve()
                generated = self._generated_source(root)
                candidate_path = root / deployment.CANDIDATE_RELATIVE
                manifest_path = root / deployment.MANIFEST_RELATIVE
                conflict = candidate_path if conflicted == "candidate" else manifest_path
                conflict.parent.mkdir(parents=True)
                conflict.write_bytes(b"different\n")
                with self.assertRaisesRegex(
                    deployment.DeploymentSourceError, "differs from frozen content"
                ):
                    deployment.materialize_deployment(
                        root, generated, search_tuple=("0.95", "0.75", "1"),
                        profile="heavy",
                    )
                other = manifest_path if conflicted == "candidate" else candidate_path
                self.assertFalse(other.exists())
                self.assertEqual(conflict.read_bytes(), b"different\n")

    def test_materialize_rejects_symlink_or_irregular_preexistence(self):
        for kind in ("candidate-symlink", "manifest-symlink", "candidate-directory"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary).resolve()
                generated = self._generated_source(root)
                candidate_path = root / deployment.CANDIDATE_RELATIVE
                manifest_path = root / deployment.MANIFEST_RELATIVE
                target = candidate_path if kind.startswith("candidate") else manifest_path
                target.parent.mkdir(parents=True)
                if kind.endswith("symlink"):
                    target.symlink_to(generated)
                else:
                    target.mkdir()
                with self.assertRaisesRegex(
                    deployment.DeploymentSourceError, "redirected or irregular"
                ):
                    deployment.materialize_deployment(
                        root, generated, search_tuple=("0.95", "0.75", "1"),
                        profile="heavy",
                    )

    def test_materialize_recovers_either_exact_partial_publication(self):
        candidate, manifest, manifest_payload = self._expected_pair()
        for existing in ("candidate", "manifest"):
            with self.subTest(existing=existing), tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary).resolve()
                generated = self._generated_source(root)
                candidate_path = root / deployment.CANDIDATE_RELATIVE
                manifest_path = root / deployment.MANIFEST_RELATIVE
                candidate_path.parent.mkdir(parents=True)
                if existing == "candidate":
                    candidate_path.write_bytes(candidate)
                else:
                    manifest_path.write_bytes(manifest_payload)
                result = deployment.materialize_deployment(
                    root, generated, search_tuple=("0.95", "0.75", "1"),
                    profile="heavy",
                )
                self.assertEqual(candidate_path.read_bytes(), candidate)
                self.assertEqual(manifest_path.read_bytes(), manifest_payload)
                self.assertFalse(result[existing]["created"])
                missing = "manifest" if existing == "candidate" else "candidate"
                self.assertTrue(result[missing]["created"])
                self.assertEqual(
                    deployment.verify_manifest_file(manifest_path, candidate_path), manifest
                )

    def test_repository_anchor_survives_ancestor_symlink_swap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary).resolve()
            actual_parent = root / "actual"
            actual_repository = actual_parent / "repository"
            attacker_repository = root / "attacker" / "repository"
            actual_repository.mkdir(parents=True)
            attacker_repository.mkdir(parents=True)
            alias = root / "alias"
            alias.symlink_to(actual_parent, target_is_directory=True)
            generated = self._generated_source(root)
            requested_repository = alias / "repository"
            original_open_tree = deployment._open_directory_tree
            swapped = False

            def swap_ancestor_after_anchor(*args, **kwargs):
                nonlocal swapped
                if not swapped:
                    alias.unlink()
                    alias.symlink_to(root / "attacker", target_is_directory=True)
                    swapped = True
                return original_open_tree(*args, **kwargs)

            with mock.patch.object(
                deployment, "_open_directory_tree", side_effect=swap_ancestor_after_anchor
            ):
                result = deployment.materialize_deployment(
                    requested_repository, generated,
                    search_tuple=("0.95", "0.75", "1"), profile="heavy",
                )
            expected = actual_repository / deployment.CANDIDATE_RELATIVE
            self.assertEqual(result["candidate"]["path"], str(expected))
            self.assertTrue(expected.is_file())
            self.assertFalse(
                (attacker_repository / deployment.CANDIDATE_RELATIVE).exists()
            )

    def test_parent_component_swap_cannot_redirect_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary).resolve()
            generated = self._generated_source(root)
            parent = root / deployment.CANDIDATE_RELATIVE.parent
            parent.mkdir(parents=True)
            retained = root / "retained-original-parent"
            outside = root / "outside"
            outside.mkdir()
            original_publish = deployment._publish_once_at
            swapped = False

            def swap_parent_before_publish(*args, **kwargs):
                nonlocal swapped
                if not swapped:
                    parent.rename(retained)
                    parent.symlink_to(outside, target_is_directory=True)
                    swapped = True
                return original_publish(*args, **kwargs)

            with mock.patch.object(
                deployment, "_publish_once_at", side_effect=swap_parent_before_publish
            ), self.assertRaisesRegex(
                deployment.DeploymentSourceError,
                "output parent is redirected or irregular|output parent route changed",
            ):
                deployment.materialize_deployment(
                    root, generated, search_tuple=("0.95", "0.75", "1"),
                    profile="heavy",
                )
            self.assertEqual(list(outside.iterdir()), [])
            self.assertTrue((retained / deployment.CANDIDATE_RELATIVE.name).is_file())

    def test_repeated_invalid_parent_does_not_leak_descriptors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary).resolve()
            generated = self._generated_source(root)
            outside = root / "outside"
            outside.mkdir()
            (root / deployment.CANDIDATE_RELATIVE.parts[0]).symlink_to(
                outside, target_is_directory=True,
            )
            descriptor_directory = pathlib.Path("/dev/fd")
            before = len(list(descriptor_directory.iterdir()))
            for _ in range(20):
                with self.assertRaisesRegex(
                    deployment.DeploymentSourceError,
                    "output parent is redirected or irregular",
                ):
                    deployment.materialize_deployment(
                        root, generated, search_tuple=("0.95", "0.75", "1"),
                        profile="heavy",
                    )
            after = len(list(descriptor_directory.iterdir()))
            self.assertEqual(before, after)

    def test_final_readback_rejects_semantically_valid_noncanonical_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary).resolve()
            generated = self._generated_source(root)
            manifest_path = root / deployment.MANIFEST_RELATIVE
            original_read = deployment._read_regular_at
            manifest_readbacks = 0

            def replace_with_pretty_json(
                directory_descriptor, name, label, *, optional=False,
            ):
                nonlocal manifest_readbacks
                if name == deployment.MANIFEST_RELATIVE.name and not optional:
                    manifest_readbacks += 1
                    if manifest_readbacks == 2:
                        value = json.loads(manifest_path.read_bytes())
                        manifest_path.unlink()
                        manifest_path.write_text(
                            json.dumps(value, indent=2, sort_keys=True), encoding="ascii"
                        )
                return original_read(
                    directory_descriptor, name, label, optional=optional,
                )

            with mock.patch.object(
                deployment, "_read_regular_at", side_effect=replace_with_pretty_json
            ), self.assertRaisesRegex(
                deployment.DeploymentSourceError, "not canonical exact bytes"
            ):
                deployment.materialize_deployment(
                    root, generated, search_tuple=("0.95", "0.75", "1"),
                    profile="heavy",
                )

    def test_failed_file_fsync_closes_descriptor_and_removes_temporary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary).resolve()
            generated = self._generated_source(root)
            original_fsync = deployment.os.fsync
            injected = False

            def fail_first_regular_file(descriptor):
                nonlocal injected
                if not injected and stat.S_ISREG(deployment.os.fstat(descriptor).st_mode):
                    injected = True
                    raise OSError("injected file fsync failure")
                return original_fsync(descriptor)

            with mock.patch.object(
                deployment.os, "fsync", side_effect=fail_first_regular_file
            ), self.assertRaisesRegex(OSError, "injected file fsync failure"):
                deployment.materialize_deployment(
                    root, generated, search_tuple=("0.95", "0.75", "1"),
                    profile="heavy",
                )
            parent = root / deployment.CANDIDATE_RELATIVE.parent
            self.assertEqual(list(parent.iterdir()), [])

    def test_null_tuple_is_fail_closed_in_validator_and_cli(self):
        candidate, manifest, _ = self._expected_pair()
        malformed = json.loads(json.dumps(manifest))
        malformed["configuration"]["tuple"] = None
        body = {key: value for key, value in malformed.items() if key != "body_sha256"}
        malformed["body_sha256"] = hashlib.sha256(
            json.dumps(
                body, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            ).encode("ascii")
        ).hexdigest()
        with self.assertRaisesRegex(deployment.DeploymentSourceError, "tuple"):
            deployment.validate_manifest(malformed, candidate)
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            candidate_path = root / "candidate.cpp"
            manifest_path = root / "manifest.json"
            candidate_path.write_bytes(candidate)
            manifest_path.write_bytes(json.dumps(malformed).encode("ascii"))
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                status = deployment.main([
                    "verify-manifest", "--manifest", str(manifest_path),
                    "--candidate", str(candidate_path),
                ])
            self.assertEqual(status, 2)
            self.assertIn("deployment source failure", stderr.getvalue())

if __name__ == "__main__":
    unittest.main()
