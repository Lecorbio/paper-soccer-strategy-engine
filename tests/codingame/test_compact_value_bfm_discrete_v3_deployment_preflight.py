import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

from tools import compact_value_bfm_discrete_v3_deployment_preflight as preflight


ROOT = pathlib.Path(__file__).resolve().parents[2]
BASE_SOURCE = ROOT / "submissions/codingame/bots/compact_value_bfm/submission.cpp"


def load_maintained_test_module():
    path = ROOT / "tests/codingame/test_compact_value_bfm_preflight.py"
    spec = importlib.util.spec_from_file_location(
        "deployment_preflight_maintained_fixture", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_final_module():
    path = ROOT / "tools/compact_value_bfm_discrete_v3_final.py"
    spec = importlib.util.spec_from_file_location(
        "deployment_preflight_final_bridge", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def regular(path, *, ascii_required=False):
    raw = path.read_bytes()
    if ascii_required:
        raw.decode("ascii")
    return {
        "path": str(path.resolve()), "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        **({"ascii": True} if ascii_required else {}),
    }


def maintained_receipt(root, generated, runtime):
    fixture_module = load_maintained_test_module()
    fixture = fixture_module.ReceiptFixture(root / "maintained-fixture")
    candidate = {
        **regular(generated, ascii_required=True), "bootstrap_zero": False,
    }
    runtime_record = regular(runtime, ascii_required=True)
    fixture.inputs["candidate"] = candidate
    fixture.inputs["runtime"] = runtime_record
    fixture.inputs["sources"][(
        preflight.maintained.BOT_RELATIVE / "submission.cpp"
    ).as_posix()] = candidate
    fixture.inputs["sources"]["tools/compact_value_bfm_preflight.py"] = regular(
        pathlib.Path(preflight.maintained.__file__), ascii_required=True
    )
    fixture.claim = preflight.base.seal({
        "schema": preflight.maintained.CLAIM_SCHEMA,
        "namespace": preflight.maintained.NAMESPACE,
        "one_shot": True,
        "claim_precedes_build_test_or_parity_command": True,
        "claimed_at_utc": "2026-09-02T08:00:00Z",
        "candidate_commit": fixture.inputs["candidate_commit"],
        "candidate_sha256": candidate["sha256"],
        "rank4_sha256": preflight.maintained.RANK4_SHA256,
        "plan_body_sha256": fixture.plan["body_sha256"],
        "inputs_sha256": preflight.base.sha256_bytes(
            preflight.base.canonical_json_bytes(fixture.inputs)
        ),
    })
    receipt = fixture.receipt()
    raw = preflight.base.canonical_json_bytes(receipt)
    directory = root / "maintained-receipts"
    directory.mkdir()
    path = directory / f"{hashlib.sha256(raw).hexdigest()}.json"
    path.write_bytes(raw)
    preflight.maintained.validate_preflight_receipt(
        receipt, claim=receipt["claim"], plan=receipt["plan"],
        inputs=receipt["inputs_before"],
    )
    return path


def timing_receipt(probe):
    samples = []
    for count in preflight.maintained.PROCESS_COUNTS:
        for color in (0, 1):
            for replica in range(count):
                samples.append({
                    "process_count": count, "color": color, "replica": replica,
                    "first_ms": 100.0, "later_max_ms": 20.0,
                })
    return {
        "schema": preflight.maintained.TIMING_SCHEMA,
        "probe_sha256": preflight.base.sha256_file(probe),
        "first_limit_exclusive_ms": preflight.maintained.FIRST_LIMIT_MS,
        "later_limit_exclusive_ms": preflight.maintained.LATER_LIMIT_MS,
        "samples": samples,
    }


def parity_receipt(runtime, probe):
    digest = "1" * 64
    return {
        "schema": preflight.maintained.PARITY_SCHEMA,
        "states": preflight.maintained.PARITY_STATES,
        "feature_states": preflight.maintained.PARITY_STATES,
        "features_sha256": digest, "cpp_sha256": "2" * 64,
        "scalar_sha256": "3" * 64, "maximum_absolute_error": 0.0,
        "all_finite": True,
        "runtime_sha256": preflight.base.sha256_file(runtime),
        "probe_sha256": preflight.base.sha256_file(probe),
    }


class DeploymentPreflightTest(unittest.TestCase):
    def tool(self, *names):
        for name in names:
            path = shutil.which(name)
            if path is not None:
                return pathlib.Path(path)
        if os.environ.get("CI"):
            self.fail(f"required deployment-preflight CI tool unavailable: {names}")
        self.skipTest(f"deployment-preflight integration tool unavailable: {names}")

    def fixture(self, root):
        repository = root / "repository"
        candidate = repository / preflight.CANDIDATE_RELATIVE
        candidate.parent.mkdir(parents=True)
        generated = root / "generated.cpp"
        generated.write_bytes(BASE_SOURCE.read_bytes())
        selected_tuple = ("0.95", "0.75", "1")
        profile = "heavy"
        work = preflight.deployment.PROFILE_ROSTER[profile]
        candidate.write_bytes(preflight.deployment.derive_source(
            generated.read_bytes(), search_tuple=selected_tuple,
            profile=profile, work=work,
        ))
        manifest = preflight.deployment.create_manifest(
            generated.read_bytes(), candidate.read_bytes(),
            search_tuple=selected_tuple, profile=profile, work=work,
        )
        manifest_path = repository / preflight.MANIFEST_RELATIVE
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")),
            encoding="ascii",
        )
        runtime = root / "runtime.json"
        runtime.write_text('{"synthetic":"runtime"}\n', encoding="ascii")
        subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
        subprocess.run(
            ["git", "config", "user.email", "deployment@example.invalid"],
            cwd=repository, check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Deployment Test"],
            cwd=repository, check=True,
        )
        subprocess.run(["git", "add", "."], cwd=repository, check=True)
        subprocess.run(
            ["git", "commit", "-qm", "deployment candidate"],
            cwd=repository, check=True,
        )
        base_receipt = maintained_receipt(root, generated, runtime)
        # Match the common Linux layout where clang++ is a symlink to the
        # clang driver binary.  The invocation spelling must survive sealing:
        # resolving this link selects the C driver and breaks C++ linkage.
        compiler_links = root / "compiler-links"
        compiler_links.mkdir()
        clang_link = compiler_links / "clang++"
        clang_link.symlink_to(self.tool("clang").resolve())
        return {
            "output_root": root / "evidence",
            "base_preflight_path": base_receipt,
            "generated_source": generated, "candidate_source": candidate,
            "runtime_path": runtime, "repository": repository,
            "source_repository": ROOT, "search_tuple": selected_tuple,
            "profile": profile, "work": work,
            "python_path": pathlib.Path(sys.executable),
            "gcc_path": self.tool("g++-15", "g++-14", "g++-13", "g++"),
            "clang_path": clang_link,
            "node_path": self.tool("node"),
        }

    def test_real_nondefault_derivative_compiles_protocol_tests_and_binds_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            context = self.fixture(pathlib.Path(temporary))
            plan = preflight.prepare(
                **context, planned_at_utc="2026-09-02T09:00:00Z"
            )
            run_context = dict(context)
            run_context.pop("output_root")
            reference = preflight.run_preflight(
                **run_context, plan_path=plan,
                claimed_at_utc="2026-09-02T09:01:00Z",
                parity_runner=lambda **kwargs: parity_receipt(
                    kwargs["runtime_path"], kwargs["probe_path"]
                ),
                timing_runner=timing_receipt,
            )
            validated = preflight.validate_reference(
                reference, generated_source=context["generated_source"],
                candidate_source=context["candidate_source"],
                runtime_path=context["runtime_path"],
                repository=context["repository"], source_repository=ROOT,
                search_tuple=context["search_tuple"], profile=context["profile"],
                work=context["work"],
            )
            self.assertEqual(
                validated["plan"]["inputs"]["tools"]["clang"]["path"],
                str(context["clang_path"].absolute()),
            )
            self.assertEqual(
                validated["plan"]["inputs"]["tools"]["clang"]["resolved_path"],
                str(context["clang_path"].resolve()),
            )
            self.assertEqual(
                validated["derivation"]["configuration"]["candidate_fpu"], 0.75
            )
            self.assertEqual(
                validated["derivation"]["configuration"]["candidate_nodes"],
                120_000,
            )
            self.assertTrue(validated["gate_path"].is_file())
            self.assertEqual(
                validated["reference"]["gate"]["sha256"],
                preflight.base.sha256_file(validated["gate_path"]),
            )
            final_bridge = load_final_module()
            final_context = final_bridge._default_preflight_validator(
                preflight_path=reference,
                candidate_source=context["candidate_source"],
                generated_source=context["generated_source"],
                runtime_path=context["runtime_path"],
                search_tuple=context["search_tuple"],
                profile=context["profile"], work=context["work"],
                repository=context["repository"],
                gate_path=validated["gate_path"],
                git_verifier=lambda _repository, _candidate, commit: {
                    "commit": commit, "tracked_clean": True,
                },
            )
            self.assertEqual(
                final_context["preflight"]["sha256"],
                preflight.base.sha256_file(reference),
            )
            self.assertEqual(
                final_context["manifest_body_sha256"],
                validated["plan"]["inputs"]["manifest_body_sha256"],
            )
            reference_raw = reference.read_bytes()
            original_reference = preflight.base.load_sealed(
                reference, preflight.REFERENCE_SCHEMA
            )
            original_receipt_path = pathlib.Path(
                original_reference["receipt"]["path"]
            )
            original_receipt = preflight.base.load_sealed(
                original_receipt_path, preflight.RECEIPT_SCHEMA
            )
            for mutation in ("cwd", "parity_runtime", "timing_probe", "harness"):
                changed = json.loads(json.dumps(original_receipt))
                changed.pop("body_sha256")
                if mutation == "cwd":
                    changed["commands"]["native_test"]["cwd"] = "/wrong"
                elif mutation == "parity_runtime":
                    changed["parity"]["runtime_sha256"] = "0" * 64
                elif mutation == "timing_probe":
                    changed["timing"]["probe_sha256"] = "0" * 64
                else:
                    changed["harness"]["submission.cpp"]["sha256"] = "0" * 64
                changed = preflight.base.seal(changed)
                raw = preflight.base.canonical_json_bytes(changed)
                changed_path = original_receipt_path.parent / (
                    hashlib.sha256(raw).hexdigest() + ".json"
                )
                changed_path.write_bytes(raw)
                changed_reference = {
                    key: value for key, value in original_reference.items()
                    if key != "body_sha256"
                }
                changed_reference["receipt"] = {
                    "path": str(changed_path.resolve()),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
                reference.write_bytes(preflight.base.canonical_json_bytes(
                    preflight.base.seal(changed_reference)
                ))
                with self.subTest(mutation=mutation), self.assertRaises(Exception):
                    preflight.validate_reference(
                        reference,
                        generated_source=context["generated_source"],
                        candidate_source=context["candidate_source"],
                        runtime_path=context["runtime_path"],
                        repository=context["repository"],
                        source_repository=ROOT,
                        search_tuple=context["search_tuple"],
                        profile=context["profile"], work=context["work"],
                    )
                reference.write_bytes(reference_raw)

    def test_base_receipt_candidate_or_uncommitted_derivative_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            context = self.fixture(pathlib.Path(temporary))
            context["candidate_source"].write_bytes(
                context["candidate_source"].read_bytes() + b"\n"
            )
            with self.assertRaises(preflight.DeploymentPreflightError):
                preflight.prepare(
                    **context, planned_at_utc="2026-09-02T09:00:00Z"
                )
        with tempfile.TemporaryDirectory() as temporary:
            context = self.fixture(pathlib.Path(temporary))
            raw = bytearray(context["base_preflight_path"].read_bytes())
            raw[-2] = ord("0") if raw[-2] != ord("0") else ord("1")
            context["base_preflight_path"].write_bytes(bytes(raw))
            with self.assertRaises(preflight.DeploymentPreflightError):
                preflight.prepare(
                    **context, planned_at_utc="2026-09-02T09:00:00Z"
                )


if __name__ == "__main__":
    unittest.main()
