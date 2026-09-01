from __future__ import annotations

import importlib.util
import pathlib
import sys
import tempfile
import threading
import types
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
PATH = ROOT / "tools/compact_value_bfm_iteration.py"
SPEC = importlib.util.spec_from_file_location("compact_value_bfm_iteration", PATH)
assert SPEC is not None and SPEC.loader is not None
iteration = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = iteration
SPEC.loader.exec_module(iteration)
campaign = iteration.campaign_module()
qualification = iteration.qualification_module()


def executable(path: pathlib.Path) -> pathlib.Path:
    path.write_text("#!/bin/sh\nexit 0\n", encoding="ascii")
    path.chmod(0o755)
    return path


def sealed_authorization(root: pathlib.Path) -> tuple[pathlib.Path, dict]:
    safe = root / "safe.operational-safe-actor.json"
    qualification.write_sealed(safe, {
        "schema": campaign.OPERATIONAL_SAFE_ACTOR_SCHEMA,
        "namespace": iteration.NAMESPACE,
    })
    auth = root / "iteration/00-authorization.json"
    document = qualification.write_sealed(auth, {
        "schema": campaign.ITERATION_AUTH_SCHEMA,
        "namespace": iteration.NAMESPACE,
        "status": "one-iteration-authorized",
        "one_shot": True,
        "authorized_at_utc": "2026-08-31T12:00:00Z",
        "offline_family_failure": {"path": str(root / "failure")},
        "operational_safe_actor": {
            "path": str(safe.resolve()),
            "sha256": iteration.sha256_file(safe),
            "body_sha256": qualification.load_sealed(
                safe, campaign.OPERATIONAL_SAFE_ACTOR_SCHEMA
            )["body_sha256"],
        },
        "sample_scaled_learning_rate": 0.00005,
        "specification": campaign.ITERATION_SPEC,
    })
    return auth, document


def write_started(root: pathlib.Path, authorization: dict, environment: dict | None = None):
    return qualification.write_sealed(root / "iteration/01-started.json", {
        "schema": campaign.ITERATION_EVENT_SCHEMA,
        "namespace": iteration.NAMESPACE,
        "status": "iteration-started",
        "started_at_utc": "2026-08-31T13:00:00Z",
        "authorization": authorization,
        "environment": environment or {
            "interactive_launch_agent": True,
            "resume": True,
            "blas_threads": 1,
            "ac_power": True,
            "free_disk_gib": 25.0,
        },
    })


class QuotaAndPlistTest(unittest.TestCase):
    def test_exact_10000_quota_and_ten_disjoint_workers(self):
        rows = iteration.exact_game_rows()
        self.assertEqual(len(rows), 10_000)
        self.assertEqual(dict(iteration.Counter(row["actor_mode"] for row in rows)),
                         iteration.QUOTAS)
        observed = set()
        for worker in range(10):
            selected = iteration.worker_rows(rows, worker)
            self.assertEqual(len(selected), 1_000)
            current = {iteration.game_identity(row) for row in selected}
            self.assertFalse(observed & current)
            observed |= current
        self.assertEqual(len(observed), 10_000)

    def test_interactive_plist_is_plan_bound_persistent_and_resumable(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            plan_path = root / "plan.json"
            plan = iteration.write_document(plan_path, {
                "schema": iteration.PLAN_SCHEMA,
                "namespace": iteration.NAMESPACE,
                "campaign_id": iteration.CAMPAIGN_ID,
            })
            document = iteration.launch_agent_document(
                label="com.papersoccer.compact-value-bfm-iteration",
                plan_path=plan_path, output_root=root,
                python_path=pathlib.Path(sys.executable),
            )
            self.assertEqual(document["ProcessType"], "Interactive")
            self.assertEqual(document["LimitLoadToSessionType"], "Aqua")
            self.assertEqual(document["KeepAlive"], {"SuccessfulExit": False})
            self.assertIn("--resume", document["ProgramArguments"])
            self.assertTrue(document["RunAtLoad"])
            environment = document["EnvironmentVariables"]
            for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                        "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                        "NUMEXPR_NUM_THREADS"):
                self.assertEqual(environment[key], "1")
            self.assertEqual(
                environment["COMPACT_VALUE_BFM_ITERATION_PLAN_BODY_SHA256"],
                plan["body_sha256"],
            )


class PrepareTest(unittest.TestCase):
    def fixture(self, root: pathlib.Path):
        checkpoint = root / "selected.float.npz"
        checkpoint.write_bytes(b"checkpoint")
        runtime = root / "student.runtime.json"
        runtime.write_bytes(b"student")
        previous = root / "previous.runtime.json"
        previous.write_bytes(b"previous")
        source = root / "submission.cpp"
        source.write_text("int main(){}\n", encoding="ascii")
        teacher_runtime = root / "teacher.runtime"
        teacher_runtime.write_bytes(b"teacher")
        selection_path = root / "selection.json"
        selection_path.write_text("selection", encoding="ascii")
        auth_path, authorization = sealed_authorization(root)
        files = {
            "checkpoint": checkpoint, "student": runtime, "previous": previous,
            "source": source, "teacher": teacher_runtime,
            "selection": selection_path,
        }
        for name in ("bundle", "roots.tsv", "roots.json", "audit"):
            files[name] = root / name
            files[name].write_text(name, encoding="ascii")
        files["pack.py"] = iteration.REPLAY_PACK_PATH
        for name in ("continuations", "search-teacher", "rank4-teacher"):
            files[name] = executable(root / name)
        jacek = ROOT / "submissions/codingame/bots/jacek_nn/submission.cpp"
        files["jacek"] = jacek
        architecture = types.SimpleNamespace(name="compact-8x8")
        bundle = types.SimpleNamespace(
            body_sha256="b" * 64,
            routes={
                "teacher_runtime": "teacher-runtime",
                "roots_tsv": "roots-tsv", "roots_manifest": "roots-manifest",
            },
            artifact_path=mock.Mock(side_effect=lambda route: {
                "teacher-runtime": teacher_runtime,
                "roots-tsv": files["roots.tsv"],
                "roots-manifest": files["roots.json"],
            }[route]),
        )
        trainer = types.SimpleNamespace(
            FrozenBundle=types.SimpleNamespace(load=mock.Mock(return_value=bundle)),
            validate_selection=mock.Mock(return_value={
                "architecture": "compact-8x8", "arm": "search-target",
                "seed": 20260907, "deployment_eligible": False,
                "offline_gate": {"passed": False},
                "runtime": {"sha256": iteration.sha256_file(runtime)},
            }),
            ARCHITECTURES={"compact-8x8": architecture},
            load_float_checkpoint=mock.Mock(return_value={}),
            load_runtime=mock.Mock(return_value={}),
        )
        safe = {
            "selection": {"path": str(selection_path.resolve()),
                          "sha256": iteration.sha256_file(selection_path)},
            "float_checkpoint": {"path": str(checkpoint.resolve()),
                                 "sha256": iteration.sha256_file(checkpoint)},
            "runtime": {"path": str(runtime.resolve()),
                        "sha256": iteration.sha256_file(runtime)},
            "generated_source": {"path": str(source.resolve()),
                                 "sha256": iteration.sha256_file(source)},
            "architecture": "6301-8-8-1",
            "operationally_safe": True,
        }
        rejected = [
            {
                "selection": safe["selection"],
                "runtime": {"path": str(runtime.resolve()),
                            "sha256": iteration.sha256_file(runtime)},
            },
            {
                "selection": {"path": str(root / "other-selection")},
                "runtime": {"path": str(previous.resolve()),
                            "sha256": iteration.sha256_file(previous)},
            },
        ]
        rejected.extend({
            "selection": {"path": str(root / f"selection-{index}")},
            "runtime": {"path": str(root / f"runtime-{index}"),
                        "sha256": f"{index + 1:x}" * 64},
        } for index in range(4))
        failure = {
            "bundle_manifest": {
                "path": str(files["bundle"].resolve()),
                "sha256": iteration.sha256_file(files["bundle"]),
            },
            "rejected_deployable_arms": rejected,
        }
        _workflow, identities = iteration.validate_selfsearch_contract(
            iteration.SELFSEARCH_WORKFLOW_PATH
        )
        return files, auth_path, authorization, trainer, safe, failure, identities

    def test_prepare_accepts_authorized_operational_safe_offline_rejection(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            (files, auth_path, authorization, trainer, safe,
             failure, _identities) = self.fixture(root)
            with (mock.patch.object(iteration, "trainer_module", return_value=trainer),
                  mock.patch.object(iteration, "_load_iteration_authorization",
                                    return_value=authorization),
                  mock.patch.object(campaign, "validate_operational_safe_actor",
                                    return_value=safe),
                  mock.patch.object(campaign, "validate_offline_family_failure",
                                    return_value=failure)):
                plan_path = iteration.prepare_plan(
                    root / "output", authorization_path=auth_path,
                    bundle_manifest=files["bundle"],
                    family_selection_path=files["selection"], artifact_root=root,
                    float_checkpoint=files["checkpoint"],
                    student_runtime=files["student"],
                    generated_source=files["source"],
                    previous_compact_runtime=files["previous"],
                    roots_tsv=files["roots.tsv"], roots_manifest=files["roots.json"],
                    input_audit=files["audit"],
                    selfsearch_workflow=iteration.SELFSEARCH_WORKFLOW_PATH,
                    continuation_generator=files["continuations"],
                    jacek_nn_opponent=files["jacek"],
                    search_teacher=files["search-teacher"],
                    rank4_teacher=files["rank4-teacher"], pack_tool=files["pack.py"],
                    python_path=pathlib.Path(sys.executable), learning_rate=0.00005,
                    label="com.papersoccer.compact-value-bfm-iteration",
                )
            plan = iteration.load_document(plan_path, iteration.PLAN_SCHEMA, "plan")
            iteration.validate_plan_contract(
                plan, plan_path=plan_path, output_root=root / "output"
            )
            self.assertEqual(plan["specification"]["total_games"], 10_000)
            self.assertEqual(plan["specification"]["fixed_work_configuration"],
                             iteration.FIXED_WORK_CONFIGURATION)
            self.assertEqual(len(plan["workers"]), 10)
            self.assertEqual(sum(row["expected_positions"] for row in plan["workers"]),
                             200_000)
            self.assertEqual(sum(row["expected_deep_relabel_positions"]
                                 for row in plan["workers"]), 50_000)
            self.assertTrue(
                set(iteration.MAINTAINED_PYTHON_TOOL_PATHS).issubset(plan["tools"])
            )
            self.assertEqual(
                pathlib.Path(plan["tools"]["pack_tool"]["path"]).resolve(),
                iteration.REPLAY_PACK_PATH.resolve(),
            )
            for index, worker in enumerate(plan["workers"]):
                self.assertEqual(worker["worker_index"], index)
                self.assertEqual(worker["expected_games"], 1_000)
                self.assertEqual(worker["command"][2], "worker")
                self.assertIn("--resume", worker["command"])
                self.assertTrue(iteration.valid_sha(worker["game_identities_sha256"]))
            binding = iteration.load_document(
                auth_path.parent / "plan-binding.json",
                iteration.PLAN_BINDING_SCHEMA, "binding",
            )
            self.assertFalse(binding["second_plan_authorized"])

    def test_unproven_pipeline_and_excess_learning_rate_fail_closed(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            other = root / "workflow.py"
            other.write_text("pass\n", encoding="ascii")
            with self.assertRaisesRegex(iteration.IterationError, "maintained"):
                iteration.validate_selfsearch_contract(other)
            auth, authorization = sealed_authorization(root)
            with (mock.patch.object(iteration, "_load_iteration_authorization",
                                    return_value=authorization),
                  self.assertRaisesRegex(iteration.IterationError, "6e-5")):
                iteration.prepare_plan(
                    root / "output", authorization_path=auth,
                    bundle_manifest=root / "missing",
                    family_selection_path=root / "missing", artifact_root=root,
                    float_checkpoint=root / "missing",
                    student_runtime=root / "missing", generated_source=root / "missing",
                    previous_compact_runtime=root / "missing",
                    roots_tsv=root / "missing", roots_manifest=root / "missing",
                    input_audit=root / "missing",
                    selfsearch_workflow=iteration.SELFSEARCH_WORKFLOW_PATH,
                    continuation_generator=root / "missing",
                    jacek_nn_opponent=root / "missing",
                    search_teacher=root / "missing", rank4_teacher=root / "missing",
                    pack_tool=root / "missing", python_path=pathlib.Path(sys.executable),
                    learning_rate=0.000061, label="valid.label",
                )

    def test_roots_and_prior_runtime_must_match_sealed_family_lineage(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            (files, auth_path, authorization, trainer, safe,
             failure, _identities) = self.fixture(root)

            def invoke(output, *, roots_tsv, previous):
                with (mock.patch.object(iteration, "trainer_module", return_value=trainer),
                      mock.patch.object(iteration, "_load_iteration_authorization",
                                        return_value=authorization),
                      mock.patch.object(campaign, "validate_operational_safe_actor",
                                        return_value=safe),
                      mock.patch.object(campaign, "validate_offline_family_failure",
                                        return_value=failure)):
                    return iteration.prepare_plan(
                        output, authorization_path=auth_path,
                        bundle_manifest=files["bundle"],
                        family_selection_path=files["selection"], artifact_root=root,
                        float_checkpoint=files["checkpoint"],
                        student_runtime=files["student"], generated_source=files["source"],
                        previous_compact_runtime=previous, roots_tsv=roots_tsv,
                        roots_manifest=files["roots.json"], input_audit=files["audit"],
                        selfsearch_workflow=iteration.SELFSEARCH_WORKFLOW_PATH,
                        continuation_generator=files["continuations"],
                        jacek_nn_opponent=files["jacek"],
                        search_teacher=files["search-teacher"],
                        rank4_teacher=files["rank4-teacher"], pack_tool=files["pack.py"],
                        python_path=pathlib.Path(sys.executable), learning_rate=.00005,
                        label="com.papersoccer.compact-value-bfm-iteration",
                    )

            rogue_roots = root / "rogue-roots.tsv"
            rogue_roots.write_text("rogue", encoding="ascii")
            with self.assertRaisesRegex(iteration.IterationError, "bundle roots"):
                invoke(root / "bad-roots-output", roots_tsv=rogue_roots,
                       previous=files["previous"])
            with self.assertRaisesRegex(iteration.IterationError, "distinct rejected"):
                invoke(root / "bad-prior-output", roots_tsv=files["roots.tsv"],
                       previous=files["student"])

    def test_python_tool_closure_requires_maintained_exact_paths_and_hashes(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            tools = {
                name: iteration.file_record(path)
                for name, path in iteration.MAINTAINED_PYTHON_TOOL_PATHS.items()
            }
            observed = iteration.validate_maintained_python_tool_closure(tools)
            self.assertEqual(set(observed), set(iteration.MAINTAINED_PYTHON_TOOL_PATHS))

            copied_pack = root / "jacek_replay_pack.py"
            copied_pack.write_bytes(iteration.REPLAY_PACK_PATH.read_bytes())
            wrong_path = dict(tools)
            wrong_path["pack_tool"] = iteration.file_record(copied_pack)
            with self.assertRaisesRegex(iteration.IterationError, "maintained exact path"):
                iteration.validate_maintained_python_tool_closure(wrong_path)

            wrong_hash = {name: dict(record) for name, record in tools.items()}
            wrong_hash["replay_features"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(iteration.IterationError, "changed after plan"):
                iteration.validate_maintained_python_tool_closure(wrong_hash)


def make_launch_fixture(root: pathlib.Path):
    auth_path, _authorization = sealed_authorization(root)
    plan_path = root / "output/iteration-plan.json"
    auth_reference = iteration.artifact_reference(auth_path, campaign.ITERATION_AUTH_SCHEMA)
    plan = iteration.write_document(plan_path, {
        "schema": iteration.PLAN_SCHEMA, "namespace": iteration.NAMESPACE,
        "campaign_id": iteration.CAMPAIGN_ID, "authorization": auth_reference,
        "one_shot_plan_binding": str((auth_path.parent / "plan-binding.json").resolve()),
        "workers": [], "learning_rate": .00005,
    })
    plist = iteration.launch_agent_document(
        label="com.papersoccer.iteration", plan_path=plan_path,
        output_root=root / "output", python_path=pathlib.Path(sys.executable),
    )
    plist_path = root / "output/launchagent/com.papersoccer.iteration.plist"
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    plist_path.write_bytes(iteration.plistlib.dumps(plist, sort_keys=True))
    iteration.write_document(root / "output/launchagent-reference.json", {
        "schema": "papersoccer.compact-value-bfm.launchagent-reference.v1",
        "namespace": iteration.NAMESPACE, "label": "com.papersoccer.iteration",
        "plan_body_sha256": plan["body_sha256"],
        "plist": iteration.file_record(plist_path), "interactive": True,
        "resume": True, "blas_threads": 1,
    })
    return plan_path, auth_path, auth_reference


class LaunchAgentTest(unittest.TestCase):
    def test_start_receipt_precedes_bootstrap_and_resume_is_stable(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            plan, _auth, auth_reference = make_launch_fixture(root)
            calls = []

            def start(root_path, *, environment, started_at_utc):
                return write_started(root_path, auth_reference, dict(environment))

            def run(command, **kwargs):
                if command[1] in {"bootstrap", "kickstart"}:
                    self.assertTrue((root / "iteration/01-started.json").is_file())
                calls.append(command)
                return types.SimpleNamespace(
                    returncode=1 if command[1] == "print" else 0,
                    stdout="", stderr="",
                )

            with mock.patch.object(campaign, "start_iteration", side_effect=start):
                result = iteration.install_launch_agent(
                    plan_path=plan, output_root=root / "output",
                    launch_agents_directory=root / "LaunchAgents", runner=run,
                    power_check=lambda: {"ac_power": True},
                    disk_check=lambda _path: {"free_disk_gib": 25.0},
                )
            self.assertEqual(result["status"], "installed")
            self.assertTrue(any(command[1] == "bootstrap" for command in calls))
            resumed_calls = []

            def resumed_run(command, **kwargs):
                resumed_calls.append(command)
                return types.SimpleNamespace(returncode=0, stdout="loaded", stderr="")

            resumed = iteration.install_launch_agent(
                plan_path=plan, output_root=root / "output",
                launch_agents_directory=root / "LaunchAgents", resume=True,
                runner=resumed_run, power_check=lambda: {"ac_power": True},
                disk_check=lambda _path: {"free_disk_gib": 30.0},
            )
            self.assertEqual(resumed["status"], "resumed")
            self.assertFalse(any(command[1] == "bootstrap" for command in resumed_calls))

    def test_power_and_strict_more_than_20_gib_fail_before_launchctl(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            plan, _auth, _reference = make_launch_fixture(root)
            calls = []
            for power, disk in ((False, 100.0), (True, 20.0)):
                with self.assertRaises(iteration.IterationError):
                    iteration.install_launch_agent(
                        plan_path=plan, output_root=root / "output",
                        launch_agents_directory=root / "LaunchAgents",
                        runner=lambda *args, **kwargs: calls.append(args),
                        power_check=lambda power=power: {"ac_power": power},
                        disk_check=lambda _path, disk=disk: {"free_disk_gib": disk},
                    )
            self.assertEqual(calls, [])


class ExecuteTest(unittest.TestCase):
    def fixture(self, root: pathlib.Path, *, duplicate_manifest: bool = False):
        auth_path, _authorization = sealed_authorization(root)
        auth_reference = iteration.artifact_reference(
            auth_path, campaign.ITERATION_AUTH_SCHEMA
        )
        output = root / "output"
        generated = root / "submission.cpp"
        generated.write_text("int main(){}\n", encoding="ascii")
        rows = iteration.exact_game_rows()
        workers = []
        for index in range(10):
            selected = iteration.worker_rows(rows, index)
            payload = iteration.render_worker_plan(selected)
            workers.append({
                "worker_index": index, "expected_games": 1_000,
                "expected_quotas": iteration.worker_quota(selected),
                "game_plan_sha256": iteration.sha256_bytes(payload),
                "game_identities_sha256": iteration.game_identity_set_sha256(selected),
                "expected_positions": 20_000,
                "expected_deep_relabel_positions": 5_000,
                "result_path": str(output / f"workers/worker-{index:02d}.result.json"),
                "command": ["worker", str(index),
                            str(output / f"workers/worker-{index:02d}.result.json")],
            })
        plan_path = output / "iteration-plan.json"
        plan = iteration.write_document(plan_path, {
            "schema": iteration.PLAN_SCHEMA, "namespace": iteration.NAMESPACE,
            "campaign_id": iteration.CAMPAIGN_ID, "authorization": auth_reference,
            "one_shot_plan_binding": str((auth_path.parent / "plan-binding.json").resolve()),
            "workers": workers, "learning_rate": .00005,
            "inputs": {"generated_source": iteration.file_record(generated)},
            "tools": {
                name: iteration.file_record(path)
                for name, path in iteration.MAINTAINED_PYTHON_TOOL_PATHS.items()
            },
        })
        write_started(root, auth_reference, {
            "interactive_launch_agent": True, "resume": True,
            "blas_threads": 1, "ac_power": True, "free_disk_gib": 25.0,
            "label": "com.papersoccer.iteration",
            "plan_body_sha256": plan["body_sha256"],
        })
        iteration.write_document(auth_path.parent / "plan-binding.json", {
            "schema": iteration.PLAN_BINDING_SCHEMA, "namespace": iteration.NAMESPACE,
            "plan": iteration.artifact_reference(plan_path, iteration.PLAN_SCHEMA),
            "one_shot": True, "second_plan_authorized": False,
        })
        manifests = []
        for index in range(1 if duplicate_manifest else 10):
            path = output / f"workers/manifests/on-policy-train-{index:02d}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"manifest {index}\n", encoding="ascii")
            manifests.append(path)
        runtime = output / "fine-tuned.runtime.json"
        runtime.parent.mkdir(parents=True, exist_ok=True)
        runtime.write_bytes(b"runtime")
        runner_calls = []
        lock = threading.Lock()

        def runner(command, **kwargs):
            index = int(command[1])
            result_path = pathlib.Path(command[2])
            manifest = manifests[0 if duplicate_manifest else index]
            with lock:
                runner_calls.append(index)
            expected = workers[index]
            iteration.write_document(result_path, {
                "schema": iteration.WORKER_RESULT_SCHEMA,
                "namespace": iteration.NAMESPACE,
                "campaign_id": iteration.CAMPAIGN_ID,
                "worker_index": index, "workers": 10, "games": 1_000,
                "quotas": expected["expected_quotas"],
                "game_plan_sha256": expected["game_plan_sha256"],
                "game_plan_rows": 1_000,
                "game_identities_sha256": expected["game_identities_sha256"],
                "positions": 20_000, "deep_relabel_positions": 5_000,
                "train_positions": 12_000 + index,
                "positions_per_game": 20, "fixed_work": True,
                "fixed_work_configuration": iteration.FIXED_WORK_CONFIGURATION,
                "deep_relabel_fraction": .25,
                "target_semantics": iteration.TARGET_SEMANTICS,
                "compact_actor_bindings": {
                    f"compact_{role}_{field}": iteration.sha256_bytes(
                        f"{index}:{role}:{field}".encode()
                    )
                    for role in ("student", "prior")
                    for field in (
                        "runtime_sha256", "runtime_body_sha256", "payload_sha256",
                        "source_bundle_body_sha256", "selection_sha256",
                    )
                },
                "resumed": True, "plan_body_sha256": plan["body_sha256"],
                "train_manifests": [str(manifest.resolve())],
                "train_manifest_sha256": [iteration.sha256_file(manifest)],
            })
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        def fine_tuner(**kwargs):
            selected_checkpoint = output / "fine-tuned.float.npz"
            selected_checkpoint.write_bytes(b"fine-tuned")
            selected_source = output / "fine-tuned.submission.cpp"
            selected_source.write_text("int main(){}\n", encoding="ascii")
            selection = output / "iteration-selection.json"
            iteration.write_document(selection, {
                "schema": iteration.SELECTION_SCHEMA,
                "namespace": iteration.NAMESPACE,
                "plan_body_sha256": plan["body_sha256"],
                "architecture": "compact-8x8",
                "float_checkpoint": iteration.file_record(selected_checkpoint),
                "runtime": iteration.file_record(runtime),
                "generated_source": iteration.file_record(selected_source),
                "source_export": {
                    "runtime_sha256": iteration.sha256_file(runtime),
                    "source_sha256": iteration.sha256_file(selected_source),
                    "source_ascii_bytes": selected_source.stat().st_size,
                    "source_limit_exclusive": 95_000,
                },
                "offline_gate": {"passed": True},
            })
            return selection

        def complete(root_path, *, result, completed_at_utc):
            path = root_path / "iteration/02-completed.json"
            return qualification.write_sealed(path, {
                "schema": campaign.ITERATION_EVENT_SCHEMA,
                "namespace": iteration.NAMESPACE,
                "status": "iteration-completed",
                "result": dict(result),
            })

        return plan_path, output, runner, fine_tuner, complete, runner_calls

    def test_real_campaign_start_reference_reaches_preworker_execute_boundary(self):
        """Exercise the cross-module reference shape hidden by worker mocks."""

        class PreWorkerReached(RuntimeError):
            pass

        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            plan_path, output, _runner, _fine_tuner, _complete, _calls = (
                self.fixture(root)
            )
            (root / "iteration/01-started.json").unlink()
            plan = iteration.load_document(
                plan_path, iteration.PLAN_SCHEMA, "integration plan"
            )
            authorization = qualification.load_sealed(
                root / "iteration/00-authorization.json",
                campaign.ITERATION_AUTH_SCHEMA,
            )
            environment = {
                "interactive_launch_agent": True,
                "resume": True,
                "blas_threads": 1,
                "ac_power": True,
                "free_disk_gib": 25.0,
                "label": "com.papersoccer.iteration",
                "plan_body_sha256": plan["body_sha256"],
            }
            with mock.patch.object(
                campaign, "validate_iteration_authorization",
                return_value=authorization,
            ):
                started = campaign.start_iteration(
                    root, environment=environment,
                    started_at_utc="2026-08-31T13:00:00Z",
                )
            self.assertEqual(started["authorization"], plan["authorization"])

            with (
                mock.patch.object(iteration, "validate_plan_contract"),
                mock.patch.object(
                    iteration, "_run_worker", side_effect=PreWorkerReached
                ),
                self.assertRaises(PreWorkerReached),
            ):
                iteration.execute_plan(
                    plan_path=plan_path, output_root=output, resume=True,
                    power_check=lambda: {"ac_power": True},
                    disk_check=lambda _path: {"free_disk_gib": 25.0},
                    require_launchagent=False,
                )

    def test_exact_workers_resume_handoff_and_no_second_iteration(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            plan, output, runner, fine_tuner, complete, calls = self.fixture(root)
            with (mock.patch.object(campaign, "complete_iteration", side_effect=complete),
                  mock.patch.object(iteration, "validate_plan_contract")):
                result = iteration.execute_plan(
                    plan_path=plan, output_root=output, resume=True, runner=runner,
                    power_check=lambda: {"ac_power": True},
                    disk_check=lambda _path: {"free_disk_gib": 25.0},
                    fine_tuner=fine_tuner, require_launchagent=False,
                )
            self.assertEqual(sorted(calls), list(range(10)))
            self.assertEqual(result["iterations_remaining"], 0)
            self.assertFalse(result["second_iteration_authorized"])
            receipt = iteration.validate_iteration_reference(
                output / "iteration-reference.json", output_root=output,
                plan=iteration.load_document(plan, iteration.PLAN_SCHEMA, "plan"),
            )
            self.assertEqual(receipt, result)
            handoff = qualification.load_sealed(
                output / "post-iteration-development-handoff.json",
                iteration.POST_ITERATION_HANDOFF_SCHEMA,
            )
            self.assertEqual(handoff["status"],
                             "offline-qualified-awaiting-development")
            self.assertFalse(handoff["development_selected"])
            self.assertFalse(handoff["protected_tests_authorized"])
            self.assertEqual(set(handoff["candidate"]), {
                "candidate_id", "architecture", "target", "float_checkpoint",
                "runtime", "generated_source",
            })
            with mock.patch.object(iteration, "validate_plan_contract"):
                second = iteration.execute_plan(
                    plan_path=plan, output_root=output, resume=True,
                    runner=lambda *args, **kwargs: self.fail("completed worker rerun"),
                    power_check=lambda: {"ac_power": True},
                    disk_check=lambda _path: {"free_disk_gib": 25.0},
                    fine_tuner=lambda **kwargs: self.fail("completed fine-tune rerun"),
                    require_launchagent=False,
                )
            self.assertEqual(second["body_sha256"], result["body_sha256"])

    def test_duplicate_worker_manifests_are_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            plan, output, runner, fine_tuner, complete, _calls = self.fixture(
                root, duplicate_manifest=True
            )
            with (mock.patch.object(campaign, "complete_iteration", side_effect=complete),
                  mock.patch.object(iteration, "validate_plan_contract"),
                  self.assertRaisesRegex(iteration.IterationError, "10,000-game")):
                iteration.execute_plan(
                    plan_path=plan, output_root=output, resume=True, runner=runner,
                    power_check=lambda: {"ac_power": True},
                    disk_check=lambda _path: {"free_disk_gib": 25.0},
                    fine_tuner=fine_tuner, require_launchagent=False,
                )

    def test_offline_rejection_is_terminal_and_does_not_authorize_development(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            plan, output, runner, fine_tuner, complete, _calls = self.fixture(root)

            def rejecting(**kwargs):
                selected = fine_tuner(**kwargs)
                document = iteration.load_document(
                    selected, iteration.SELECTION_SCHEMA, "passing selection"
                )
                body = dict(document)
                body.pop("body_sha256")
                body["offline_gate"] = {"passed": False, "errors": ["floor"]}
                body["status"] = "offline-evaluator-rejected"
                rejected = output / "rejected-iteration-selection.json"
                iteration.write_document(rejected, body)
                return rejected

            def record_failure(root_path, *, stage, evidence_path, recorded_at_utc):
                self.assertEqual(stage, "offline-evaluator")
                return qualification.write_sealed(
                    root_path / "iteration/03-post-iteration-failure.json", {
                        "schema": campaign.POST_ITERATION_FAILURE_SCHEMA,
                        "namespace": iteration.NAMESPACE,
                        "status": "post-iteration-family-failure",
                        "failed": True, "stage": stage,
                        "evidence_sha256": iteration.sha256_file(evidence_path),
                        "iterations_remaining": 0, "upload_authorized": False,
                    },
                )

            with (mock.patch.object(campaign, "complete_iteration", side_effect=complete),
                  mock.patch.object(campaign, "record_post_iteration_failure",
                                    side_effect=record_failure),
                  mock.patch.object(iteration, "validate_plan_contract")):
                result = iteration.execute_plan(
                    plan_path=plan, output_root=output, resume=True, runner=runner,
                    power_check=lambda: {"ac_power": True},
                    disk_check=lambda _path: {"free_disk_gib": 25.0},
                    fine_tuner=rejecting, require_launchagent=False,
                )
            receipt_path = pathlib.Path(result["receipt"]["path"])
            receipt = iteration.load_document(
                receipt_path, iteration.RUN_RECEIPT_SCHEMA, "run receipt"
            )
            self.assertFalse(receipt["offline_gate_passed"])
            self.assertTrue(receipt["terminal_offline_rejection"])
            self.assertFalse(receipt["development_screen_required"])
            self.assertFalse((output / "post-iteration-development-handoff.json").exists())
            self.assertTrue((root / "iteration/03-post-iteration-failure.json").is_file())

    def test_resume_is_mandatory(self):
        with tempfile.TemporaryDirectory() as raw:
            plan, output, runner, fine_tuner, _complete, _calls = self.fixture(
                pathlib.Path(raw)
            )
            with (mock.patch.object(iteration, "validate_plan_contract"),
                  self.assertRaisesRegex(iteration.IterationError, "requires --resume")):
                iteration.execute_plan(
                    plan_path=plan, output_root=output, resume=False, runner=runner,
                    power_check=lambda: {"ac_power": True},
                    disk_check=lambda _path: {"free_disk_gib": 25.0},
                    fine_tuner=fine_tuner, require_launchagent=False,
                )


class GateSelectionTest(unittest.TestCase):
    def test_gate_feasibility_precedes_objective_and_violation_distance(self):
        compact = types.SimpleNamespace(
            COMMON_MINIMUM_SIGN=.70, COMMON_MAXIMUM_HUBER=.30,
            CANONICAL_MINIMUM_SIGN=.65, CANONICAL_MAXIMUM_HUBER=.35,
            MAXIMUM_SIGN_LOSS=.005, MAXIMUM_HUBER_RATIO=1.02,
        )

        def gate(base, candidate):
            passed = all([
                candidate["common_adjudicator"]["sign_accuracy"] >= .70,
                candidate["canonical_validation"]["sign_accuracy"] >= .65,
                candidate["common_adjudicator"]["weighted_huber"] <= .30,
                candidate["canonical_validation"]["weighted_huber"] <= .35,
            ])
            return {"passed": passed, "errors": [] if passed else ["failed"]}

        compact.offline_advancement_gate = gate
        base = {
            "common_adjudicator": {"sign_accuracy": .71, "weighted_huber": .29},
            "canonical_validation": {"sign_accuracy": .66, "weighted_huber": .34},
        }

        def metrics(sign, objective):
            return {
                "common_adjudicator": {
                    "sign_accuracy": sign, "weighted_huber": .29,
                    "objective_weighted_huber": objective,
                },
                "canonical_validation": {
                    "sign_accuracy": sign - .05, "weighted_huber": .34,
                    "objective_weighted_huber": objective,
                },
            }

        passing = metrics(.705, 1.0)
        failed_close = metrics(.699, .1)
        failed_far = metrics(.68, .01)
        self.assertLess(iteration.gate_feasibility_key(compact, base, passing),
                        iteration.gate_feasibility_key(compact, base, failed_close))
        self.assertLess(iteration.gate_feasibility_key(compact, base, failed_close),
                        iteration.gate_feasibility_key(compact, base, failed_far))

    def test_iteration_source_export_is_content_addressed_and_strictly_under_cap(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            runtime = root / "runtime.json"
            runtime.write_bytes(b"runtime")
            paths = {
                **iteration.MAINTAINED_PYTHON_TOOL_PATHS,
                "submission_config": iteration.SUBMISSION_CONFIG_PATH,
                "submission_sources": iteration.SUBMISSION_SOURCES_PATH,
            }
            plan = {"tools": {
                name: iteration.file_record(path) for name, path in paths.items()
            }}
            model = types.SimpleNamespace(render_header=lambda path: (
                b"header", {
                    "file_sha256": iteration.sha256_file(path),
                    "body_sha256": "1" * 64,
                    "header_sha256": iteration.sha256_bytes(b"header"),
                },
            ))
            payload = b"x" * 94_999
            submission = types.SimpleNamespace(
                render=lambda **kwargs: (root / "ignored", payload)
            )
            with mock.patch.object(
                iteration, "_load_module", side_effect=[model, submission]
            ):
                source, metadata = iteration.render_iteration_source(
                    runtime=runtime, plan=plan, output_root=root
                )
            self.assertEqual(source.name,
                             iteration.sha256_bytes(payload) + ".submission.cpp")
            self.assertEqual(metadata["source_ascii_bytes"], 94_999)
            too_large = types.SimpleNamespace(
                render=lambda **kwargs: (root / "ignored", b"x" * 95_000)
            )
            with (mock.patch.object(
                    iteration, "_load_module", side_effect=[model, too_large]),
                  self.assertRaisesRegex(iteration.IterationError, "95,000")):
                iteration.render_iteration_source(
                    runtime=runtime, plan=plan, output_root=root / "second"
                )


if __name__ == "__main__":
    unittest.main()
