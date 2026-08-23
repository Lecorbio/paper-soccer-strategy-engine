import hashlib
import json
import os
import pathlib
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import jacek_replay_workflow as workflow  # noqa: E402


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def make_round(directory, round_index, previous=None):
    actual_repository = workflow.repository_identity(ROOT)
    fixture_repository = {
        **actual_repository,
        "branch": "codex/canonical-fixture",
        "clean": True,
        "status_sha256": hashlib.sha256(b"").hexdigest(),
    }
    campaign_directory = directory
    build_directory = campaign_directory / "release-build"
    build_directory.mkdir(exist_ok=True)
    cache = build_directory / "CMakeCache.txt"
    teacher_binary = build_directory / "papersoccer_jacek_replay_teacher"
    continuation_binary = (
        build_directory / "papersoccer_jacek_replay_continuations"
    )
    if not cache.exists():
        cache.write_text(
            "CMAKE_BUILD_TYPE:STRING=Release\n"
            "CMAKE_CXX_COMPILER:FILEPATH=/fixture/c++\n"
        )
        teacher_binary.write_bytes(b"fixture-teacher\n")
        continuation_binary.write_bytes(b"fixture-continuations\n")
    directory = campaign_directory / f"round-{round_index}"
    directory.mkdir()
    prior_entries = previous["validation"]["entries"] if previous else []
    roots_path = directory / f"roots-{round_index}.json"
    roots_path.write_text(f"roots-{round_index}\n")
    teacher_tsv_path = directory / f"teacher-{round_index}.tsv"
    teacher_tsv_path.write_text(
        "group_id\tsource\twinner\ttranscript\nroot\tfixture\t0\t0\n"
    )
    teacher_jsonl_path = directory / f"teacher-{round_index}.jsonl"
    teacher_jsonl_path.write_text(
        json.dumps({"schema": "papersoccer.jacek-replay-teacher.v1"}) + "\n"
    )
    continuation_path = directory / f"continuations-{round_index}.tsv"
    continuation_lines = [
        "# papersoccer.jacek-replay-continuations.v1",
        f"# round={round_index}",
        "group_id\tsource\twinner\ttranscript",
    ]
    continuation_lines.extend(
        f"root\tcontinuation-round-{round_index}\t0\t0" for _ in range(10_000)
    )
    continuation_path.write_text("\n".join(continuation_lines) + "\n")
    quotas = (
        {
            "rank4-vs-rank4": 10_000,
            "candidate-selfplay": 0,
            "candidate-p1-vs-rank4": 0,
            "candidate-p2-vs-rank4": 0,
        }
        if round_index == 0
        else workflow._planned_candidate_quotas(10_000)
    )
    modes = [
        mode
        for mode, count in quotas.items()
        for _ in range(count)
    ]
    colors = {
        "rank4-vs-rank4": "none",
        "candidate-selfplay": "both",
        "candidate-p1-vs-rank4": "player-one",
        "candidate-p2-vs-rank4": "player-two",
    }
    transcript_hash = hashlib.sha256(b"0").hexdigest()
    continuation_manifest_path = directory / f"continuations-{round_index}.json"
    continuation_manifest = {
        "schema": workflow.CONTINUATION_MANIFEST_SCHEMA,
        "tsv_schema": "papersoccer.jacek-replay-continuations.v1",
        "round": round_index,
        "requested_games": 10_000,
        "successful_games": 10_000,
        "seed": 17,
        "actor_nodes": 16_000,
        "candidate_tree_nodes": 2_000,
        "maximum_turns": 400,
        "attempt_cap_per_requested_game": 20,
        "attempts": 10_000,
        "failed_attempts": 0,
        "quota_policy": "fixture",
        "planned_quotas": quotas,
        "successful_quotas": quotas,
        "bindings": {
            "input_sha256": digest(teacher_tsv_path),
            "output_sha256": digest(continuation_path),
            "model_sha256": digest(previous["runtime_path"]) if previous else None,
        },
        "rows": [
            {
                "continuation_id": "continuation:"
                + hashlib.sha256(f"{round_index}:{ordinal}".encode()).hexdigest(),
                "row_ordinal": ordinal,
                "attempt_ordinal": ordinal,
                "game_seed": ordinal,
                "actor_mode": mode,
                "candidate_color": colors[mode],
                "root_lineage": {
                    "root_row_ordinal": 0,
                    "group_id": "root",
                    "root_transcript_sha256": transcript_hash,
                    "prefix_turns": 1,
                },
                "transcript_sha256": transcript_hash,
            }
            for ordinal, mode in enumerate(modes)
        ],
    }
    write_json(continuation_manifest_path, continuation_manifest)
    pack_path = directory / f"pack-{round_index}.json"
    pack_payload = {
        "schema": "papersoccer.jacek-replay-pack-report.v1",
        "packing": "sqlite-streaming-bounded-memory-v1",
        "roots_manifest_sha256": digest(roots_path),
        "shards": {
            split: {
                "sha256": str(round_index) * 63 + suffix,
                "manifest_sha256": suffix * 64,
                "samples": 1,
            }
            for split, suffix in (("train", "a"), ("validation", "b"), ("test", "c"))
        },
        "prior_shards": [
            {
                "manifest_sha256": pack["shards"][split]["manifest_sha256"],
                "npz_sha256": pack["shards"][split]["sha256"],
                "split": split,
            }
            for pack in (previous["packs"] if previous else [])
            for split in ("train", "validation", "test")
        ],
    }
    write_json(pack_path, pack_payload)
    runtime_path = directory / f"round-{round_index}.runtime"
    runtime_path.write_bytes(f"runtime-{round_index}".encode())
    runtime_hash = digest(runtime_path)
    seed_directory = directory / f"seed-checkpoints-{round_index}"
    seed_directory.mkdir()
    seed_publications = []
    for seed in (20260823, 20260824, 20260825):
        checkpoint_name = f"seed-{seed}.runtime"
        seed_receipt_name = f"seed-{seed}.json"
        checkpoint_path = seed_directory / checkpoint_name
        checkpoint_path.write_bytes(runtime_path.read_bytes())
        seed_body = {
            "schema": "papersoccer.jacek-replay-bfm-seed-checkpoint.v1",
            "seed": seed,
            "configuration": {"fixture": True},
            "inputs": {"fixture": True},
            "producer": {"fixture": True},
            "checkpoint": {
                "file": checkpoint_name,
                "artifact_sha256": digest(checkpoint_path),
            },
            "training_report": {"seed": seed},
        }
        seed_receipt = {
            **seed_body,
            "body_sha256": hashlib.sha256(
                workflow.canonical_json_bytes(seed_body)
            ).hexdigest(),
        }
        seed_receipt_path = seed_directory / seed_receipt_name
        workflow.atomic_write(
            seed_receipt_path, workflow.canonical_json_bytes(seed_receipt)
        )
        seed_publications.append(
            {
                "seed": seed,
                "checkpoint": checkpoint_name,
                "checkpoint_sha256": digest(checkpoint_path),
                "receipt": seed_receipt_name,
                "receipt_sha256": digest(seed_receipt_path),
            }
        )
    model_path = directory / f"model-{round_index}.json"
    campaign = {
        "eligible": True,
        "round": round_index,
        "continuation_games": 10_000,
        "bulk_nodes": 32_000,
        "root_and_deep_nodes": 400_000,
        "deep_percent": 10,
        "max_samples_per_game": 100,
        "actor_nodes": 16_000,
        "candidate_tree_nodes": 2_000,
        "campaign_id": workflow.CANONICAL_CAMPAIGN_ID,
        "teacher_workers": 10,
        "teacher_chunk_games": 25,
        "seed_workers": 2,
        "prior_rounds": round_index,
        "test_revealed": round_index == 2,
        "canonical_ancestry": prior_entries,
        "previous_workflow_sha256": (
            previous["validation"]["entry"]["workflow_sha256"]
            if previous else None
        ),
    }
    source_shards = [
        {
            "split": split,
            "npz_sha256": entry["sha256"],
            "samples": entry["samples"],
        }
        for item in ([*previous["packs"], pack_payload] if previous else [pack_payload])
        for split, entry in item["shards"].items()
    ]
    seed_reports = [
        {"seed": seed, "validation": {"weighted_huber": 0.1}}
        for seed in (20260823, 20260824, 20260825)
    ]
    if round_index == 2:
        seed_reports[0]["test"] = {
            "samples": 3,
            "weighted_huber": 0.05,
            "sign_accuracy": 0.75,
            "correlation": 0.5,
            "mae": 0.2,
            "prediction_mean": 0.0,
        }
    model = {
        "schema": workflow.MODEL_SCHEMA,
        "status": "canonical-campaign-candidate-not-game-gated",
        "architecture": {"dimensions": [6301, 192, 32, 1]},
        "runtime": {"path": runtime_path.name, "artifact_sha256": runtime_hash},
        "campaign_contract": campaign,
        "training": {
            "seeds": [20260823, 20260824, 20260825],
            "chosen_seed": 20260823,
            "seed_reports": seed_reports,
            "seed_checkpoints": seed_publications,
            "test_revealed_after_selection": round_index == 2,
        },
        "source_shards": source_shards,
    }
    write_json(model_path, model)
    immediate = prior_entries[-1] if prior_entries else None
    configuration = {
        **workflow.CANONICAL_CONFIGURATION,
        "profile": None,
        "campaign_eligible": True,
        "round": round_index,
        "final_test_revealed": round_index == 2,
        "prior_pack_report_sha256": [
            entry["pack_report_sha256"] for entry in prior_entries
        ],
    }
    common_inputs = {
        "exclusions_sha256": "1" * 64,
        "public_jacek_sha256": "2" * 64,
        "live_snapshot_sha256": "3" * 64,
        "teacher_executable_sha256": "4" * 64,
        "continuation_generator_sha256": "5" * 64,
    }
    environment = workflow.environment_identity()
    teacher_chunk_receipt = directory / f"teacher-chunk-{round_index}.json"
    write_json(
        teacher_chunk_receipt,
        {
            "schema": workflow.CHUNK_RECEIPT_SCHEMA,
            "campaign_id": workflow.CANONICAL_CAMPAIGN_ID,
            "round": round_index,
            "stage": "fixture-labels",
            "chunk_ordinal": 0,
            "game_row_begin": 0,
            "game_rows": 1,
            "configuration": {"fixture": True},
            "environment": environment,
            "producer": workflow.artifact_snapshot(
                ROOT / "tools/jacek_replay_workflow.py"
            ),
            "input": workflow.artifact_snapshot(teacher_tsv_path),
            "output": workflow.artifact_snapshot(teacher_jsonl_path),
            "teacher_rows": 1,
        },
    )
    teacher_chunk_result = {
        "input_games": 1,
        "teacher_rows": 1,
        "chunks": [
            {
                "chunk_ordinal": 0,
                "game_row_begin": 0,
                "game_rows": 1,
                "teacher_rows": 1,
                "input_sha256": digest(teacher_tsv_path),
                "output_sha256": digest(teacher_jsonl_path),
                "input_path": str(teacher_tsv_path),
                "output_path": str(teacher_jsonl_path),
                "receipt_path": str(teacher_chunk_receipt),
                "receipt_sha256": digest(teacher_chunk_receipt),
            }
        ],
    }
    stage_names = [
        (0, "roots"),
        (1, "teacher-tsv"),
        *(([(2, "root-labels")]) if round_index == 0 else []),
        (3, "continuations"),
        (4, "continuation-labels"),
        (5, "concatenation"),
        (6, "packing"),
        (7, "training"),
        (8, "selected-runtime"),
    ]
    stage_bindings = []
    for ordinal, stage_name in stage_names:
        stage_path = directory / f"stage-{round_index}-{ordinal}-{stage_name}.json"
        stage_receipt = {
            "schema": workflow.STAGE_RECEIPT_SCHEMA,
            "campaign_id": workflow.CANONICAL_CAMPAIGN_ID,
            "round": round_index,
            "ordinal": ordinal,
            "stage": stage_name,
            "configuration": {"fixture": True},
            "environment": environment,
            "producers": {
                "workflow": workflow.artifact_snapshot(
                    ROOT / "tools/jacek_replay_workflow.py"
                )
            },
            "inputs": {"roots": workflow.artifact_snapshot(roots_path)},
            "outputs": (
                {
                    "roots": workflow.artifact_snapshot(roots_path),
                    "seed_checkpoints": workflow.artifact_snapshot(seed_directory),
                }
                if stage_name == "training"
                else {"roots": workflow.artifact_snapshot(roots_path)}
            ),
            "result": (
                teacher_chunk_result
                if stage_name in {"root-labels", "continuation-labels"}
                else {"fixture": True}
            ),
        }
        write_json(stage_path, stage_receipt)
        stage_bindings.append(
            {
                "ordinal": ordinal,
                "stage": stage_name,
                "path": str(stage_path),
                "sha256": digest(stage_path),
            }
        )
    receipt = {
        "schema": workflow.WORKFLOW_SCHEMA,
        "producer": {
            "workflow": "6" * 64,
            "corpus": "7" * 64,
            "pack": "8" * 64,
            "trainer": "9" * 64,
        },
        "configuration": configuration,
        "inputs": {
            **common_inputs,
            "previous_roots_sha256": immediate["roots_sha256"] if immediate else None,
            "continuation_model_sha256": immediate["runtime_sha256"] if immediate else None,
        },
        "artifacts": {
            "roots": {"path": str(roots_path), "sha256": digest(roots_path)},
            "teacher_tsv": {
                "path": str(teacher_tsv_path),
                "sha256": digest(teacher_tsv_path),
            },
            "teacher_jsonl": {
                "path": str(teacher_jsonl_path),
                "sha256": digest(teacher_jsonl_path),
                "rows": 1,
            },
            "continuations": {
                "path": str(continuation_path),
                "sha256": digest(continuation_path),
                "manifest_path": str(continuation_manifest_path),
                "manifest_sha256": digest(continuation_manifest_path),
                "successful_games": 10_000,
                "successful_quotas": quotas,
            },
            "pack_report": {**pack_payload, "report": str(pack_path)},
            "training": {"artifact_sha256": runtime_hash},
            "model": {
                "manifest_path": str(model_path),
                "manifest_sha256": digest(model_path),
                "runtime_path": str(runtime_path),
                "runtime_sha256": runtime_hash,
            },
        },
        "lineage": {
            "previous_workflow": (
                {
                    "path": str(previous["receipt_path"]),
                    "sha256": previous["validation"]["entry"]["workflow_sha256"],
                }
                if previous else None
            ),
            "ancestors": prior_entries,
        },
        "execution": {
            "campaign_id": workflow.CANONICAL_CAMPAIGN_ID,
            "resumable_stage_receipt_schema": workflow.STAGE_RECEIPT_SCHEMA,
            "environment": environment,
            "repository_path": str(ROOT),
            "repository": fixture_repository,
            "release_build": workflow.release_build_identity(
                teacher_binary, continuation_binary
            ),
            "feature_encoder": workflow.artifact_snapshot(
                ROOT / "tools/jacek_replay_features.py"
            ),
            "candidate_search_source_closure_sha256": workflow.source_closure_sha256(
                ROOT
            ),
            "stage_receipts": stage_bindings,
        },
    }
    receipt_path = directory / f"workflow-{round_index}.json"
    write_json(receipt_path, receipt)
    final_inputs = {
        "roots": roots_path,
        "teacher_tsv": teacher_tsv_path,
        "teacher_jsonl": teacher_jsonl_path,
        "pack_report": pack_path,
        "model_manifest": model_path,
        "runtime": runtime_path,
    }
    final_inputs.update(
        {
            f"stage_receipt_{binding['ordinal']}": pathlib.Path(binding["path"])
            for binding in stage_bindings
        }
    )
    final_receipt_path = directory / "receipts" / "09-workflow.json"
    final_receipt_path.parent.mkdir()
    write_json(
        final_receipt_path,
        {
            "schema": workflow.STAGE_RECEIPT_SCHEMA,
            "campaign_id": workflow.CANONICAL_CAMPAIGN_ID,
            "round": round_index,
            "ordinal": 9,
            "stage": "workflow",
            "configuration": {
                "campaign_id": workflow.CANONICAL_CAMPAIGN_ID,
                "round": round_index,
                "campaign_eligible": True,
            },
            "environment": environment,
            "producers": workflow._snapshots(
                {
                    "workflow": ROOT / "tools/jacek_replay_workflow.py",
                    "corpus": ROOT / "tools/jacek_replay_corpus.py",
                    "pack": ROOT / "tools/jacek_replay_pack.py",
                    "trainer": ROOT / "tools/jacek_replay_train.py",
                    "features": ROOT / "tools/jacek_replay_features.py",
                }
            ),
            "inputs": workflow._snapshots(final_inputs),
            "outputs": {"workflow": workflow.artifact_snapshot(receipt_path)},
            "result": {"workflow_sha256": digest(receipt_path)},
        },
    )
    with mock.patch.object(
        workflow, "repository_identity", return_value=fixture_repository
    ):
        validation = workflow.validate_canonical_workflow_chain(
            receipt_path, round_index
        )
    return {
        "receipt_path": receipt_path,
        "receipt": receipt,
        "validation": validation,
        "model_path": model_path,
        "runtime_path": runtime_path,
        "roots_path": roots_path,
        "pack_path": pack_path,
        "final_receipt_path": final_receipt_path,
        "packs": [*(previous["packs"] if previous else []), pack_payload],
        "pack_paths": [*(previous["pack_paths"] if previous else []), pack_path],
    }


class JacekReplayWorkflowLineageTests(unittest.TestCase):
    def setUp(self):
        actual = workflow.repository_identity(ROOT)
        self.repository_record = {
            **actual,
            "branch": "codex/canonical-fixture",
            "clean": True,
            "status_sha256": hashlib.sha256(b"").hexdigest(),
        }
        self.repository_patcher = mock.patch.object(
            workflow, "repository_identity", return_value=self.repository_record
        )
        self.repository_patcher.start()

    def tearDown(self):
        self.repository_patcher.stop()

    def test_round_zero_cli_rejects_previous_roots_before_work(self):
        argv = [
            "jacek_replay_workflow.py",
            "--exclusions", "unused-exclusions",
            "--public-jacek", "unused-public",
            "--live-snapshot", "unused-snapshot",
            "--previous-roots", "forbidden-roots",
            "--teacher", sys.executable,
            "--output-directory", "unused-output",
        ]
        with mock.patch.object(sys, "argv", argv), redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit) as raised:
                workflow.main()
        self.assertEqual(raised.exception.code, 2)

    def test_canonical_cli_rejects_dirty_or_nonrelease_freeze_before_work(self):
        base_argv = [
            "jacek_replay_workflow.py",
            "--repository",
            str(ROOT),
            "--exclusions",
            "unused-exclusions",
            "--public-jacek",
            "unused-public",
            "--live-snapshot",
            "unused-snapshot",
            "--teacher",
            sys.executable,
            "--continuation-generator",
            sys.executable,
            "--continuation-games",
            "10000",
            "--output-directory",
            "must-not-exist",
        ]
        release = {
            "cmake_build_type": "Release",
            "cmake_cache": {"fixture": True},
            "binaries": {"teacher": {}, "continuation_generator": {}},
        }
        cases = (
            ({**self.repository_record, "clean": False}, release),
            (self.repository_record, {**release, "cmake_build_type": "Debug"}),
        )
        for repository, build in cases:
            with (
                self.subTest(repository=repository["clean"], build=build["cmake_build_type"]),
                mock.patch.object(sys, "argv", base_argv),
                mock.patch.object(
                    workflow, "repository_identity", return_value=repository
                ),
                mock.patch.object(
                    workflow, "release_build_identity", return_value=build
                ),
                redirect_stderr(StringIO()) as errors,
            ):
                with self.assertRaises(SystemExit) as raised:
                    workflow.main()
                self.assertEqual(raised.exception.code, 2)
                self.assertIn("clean named Git branch", errors.getvalue())

    def test_exact_three_round_chain_validates(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            round0 = make_round(directory, 0)
            round1 = make_round(directory, 1, round0)
            round2 = make_round(directory, 2, round1)
            validation = workflow.validate_canonical_workflow_chain(
                round2["receipt_path"], 2
            )
            predecessor = workflow.validate_canonical_predecessor_inputs(
                round_index=2,
                previous_workflow=round1["receipt_path"],
                previous_roots=round1["roots_path"],
                continuation_model=round1["runtime_path"],
                prior_pack_reports=round1["pack_paths"],
            )
            with self.assertRaisesRegex(ValueError, "selected runtime"):
                workflow.validate_canonical_predecessor_inputs(
                    round_index=2,
                    previous_workflow=round1["receipt_path"],
                    previous_roots=round1["roots_path"],
                    continuation_model=round0["runtime_path"],
                    prior_pack_reports=round1["pack_paths"],
                )
        self.assertEqual([entry["round"] for entry in validation["entries"]], [0, 1, 2])
        self.assertEqual([entry["round"] for entry in predecessor["entries"]], [0, 1])

    def test_round_zero_rejects_previous_roots(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            round0 = make_round(directory, 0)
            receipt = json.loads(round0["receipt_path"].read_bytes())
            receipt["inputs"]["previous_roots_sha256"] = "f" * 64
            write_json(round0["receipt_path"], receipt)
            with self.assertRaisesRegex(ValueError, "round 0"):
                workflow.validate_canonical_workflow_chain(round0["receipt_path"], 0)

    def test_development_or_edited_ancestor_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            round0 = make_round(directory, 0)
            round1 = make_round(directory, 1, round0)
            round2 = make_round(directory, 2, round1)
            receipt0 = json.loads(round0["receipt_path"].read_bytes())
            receipt0["configuration"]["campaign_eligible"] = False
            write_json(round0["receipt_path"], receipt0)
            with self.assertRaisesRegex(ValueError, "previous-workflow hash|campaign_eligible"):
                workflow.validate_canonical_workflow_chain(round2["receipt_path"], 2)

    def test_campaign_artifact_and_stage_corruption_fail_closed(self):
        mutations = {
            "teacher": lambda built: pathlib.Path(
                built["receipt"]["artifacts"]["teacher_jsonl"]["path"]
            ).write_text("{}\n"),
            "continuation": lambda built: pathlib.Path(
                built["receipt"]["artifacts"]["continuations"]["manifest_path"]
            ).write_text("{}\n"),
            "stage": lambda built: pathlib.Path(
                built["receipt"]["execution"]["stage_receipts"][0]["path"]
            ).write_text("{}\n"),
            "final-stage": lambda built: built["final_receipt_path"].write_text(
                "{}\n"
            ),
            "feature-binding": lambda built: self._edit_feature_binding(built),
            "repository-binding": lambda built: self._edit_execution_binding(
                built, "repository", "head", "f" * 40
            ),
            "source-closure": lambda built: self._edit_execution_binding(
                built,
                None,
                "candidate_search_source_closure_sha256",
                "f" * 64,
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                built = make_round(pathlib.Path(temporary), 0)
                mutate(built)
                with self.assertRaises(ValueError):
                    workflow.validate_canonical_workflow_chain(
                        built["receipt_path"], 0
                    )

    def test_test_reveal_is_semantically_bound_to_only_the_finalist(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            round0 = make_round(directory, 0)
            intermediate = json.loads(round0["model_path"].read_bytes())
            intermediate["training"]["seed_reports"][0]["test"] = {
                "samples": 1
            }
            with self.assertRaisesRegex(ValueError, "exposed test metrics"):
                workflow.validate_test_reveal_contract(intermediate, 0)

            round1 = make_round(directory, 1, round0)
            round2 = make_round(directory, 2, round1)
            final = json.loads(round2["model_path"].read_bytes())
            workflow.validate_test_reveal_contract(final, 2)
            final["training"]["seed_reports"][1]["test"] = dict(
                final["training"]["seed_reports"][0]["test"]
            )
            with self.assertRaisesRegex(ValueError, "selected seed"):
                workflow.validate_test_reveal_contract(final, 2)

    def test_expanded_source_closure_and_release_build_are_recursive(self):
        required = {
            "src/bots/bot.cpp",
            "src/bots/mcts_internal.hpp",
            "include/papersoccer/types.hpp",
            "include/papersoccer/geometry.hpp",
            "include/papersoccer/rules.hpp",
            "src/core/geometry.cpp",
            "src/core/rules.cpp",
        }
        self.assertTrue(required.issubset(workflow.CANDIDATE_SOURCE_CLOSURE_PATHS))
        with tempfile.TemporaryDirectory() as temporary:
            built = make_round(pathlib.Path(temporary), 0)
            cache = pathlib.Path(
                built["receipt"]["execution"]["release_build"]["cmake_cache"][
                    "path"
                ]
            )
            cache.write_text("CMAKE_BUILD_TYPE:STRING=Debug\n")
            with self.assertRaisesRegex(ValueError, "provenance is stale"):
                workflow.validate_canonical_workflow_chain(
                    built["receipt_path"], 0
                )

    @staticmethod
    def _edit_feature_binding(built):
        receipt = json.loads(built["receipt_path"].read_bytes())
        receipt["execution"]["feature_encoder"]["sha256"] = "f" * 64
        write_json(built["receipt_path"], receipt)

    @staticmethod
    def _edit_execution_binding(built, section, field, value):
        receipt = json.loads(built["receipt_path"].read_bytes())
        target = receipt["execution"] if section is None else receipt["execution"][section]
        target[field] = value
        write_json(built["receipt_path"], receipt)


class JacekReplayResumableStageTests(unittest.TestCase):
    def make_manager(self, directory, *, resume=False):
        return workflow.StageManager(
            output=directory,
            campaign_id="development-stage-test",
            round_index=0,
            resume=resume,
            environment={"fixture": "stable"},
        )

    def test_completed_stage_is_skipped_only_when_receipt_revalidates(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            source = directory / "source.txt"
            source.write_text("input\n")
            producer = directory / "producer.py"
            producer.write_text("# stable producer\n")
            output = directory / "artifact.txt"
            calls = []

            def action():
                calls.append("called")
                workflow.atomic_write(output, b"result\n")
                return {"rows": 1}

            parameters = dict(
                ordinal=0,
                name="fixture",
                configuration={"version": 1},
                producers={"fixture": producer},
                inputs={"source": source},
                outputs={"artifact": output},
                action=action,
            )
            self.make_manager(directory).execute(**parameters)
            self.make_manager(directory, resume=True).execute(**parameters)
            self.assertEqual(calls, ["called"])
            stale_parameters = {**parameters, "configuration": {"version": 2}}
            with self.assertRaisesRegex(ValueError, "configuration is stale"):
                self.make_manager(directory, resume=True).execute(**stale_parameters)
            output.write_text("corrupt\n")
            with self.assertRaisesRegex(ValueError, "stale or corrupt"):
                self.make_manager(directory, resume=True).execute(**parameters)

    def test_missing_receipt_never_adopts_preexisting_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            source = directory / "source.txt"
            source.write_text("input\n")
            producer = directory / "producer.py"
            producer.write_text("# producer\n")
            output = directory / "artifact.txt"
            output.write_text("unreceipted\n")
            with self.assertRaisesRegex(ValueError, "fresh attempt directory"):
                self.make_manager(directory, resume=True).execute(
                    ordinal=0,
                    name="fixture",
                    configuration={},
                    producers={"fixture": producer},
                    inputs={"source": source},
                    outputs={"artifact": output},
                    action=lambda: {},
                )

    def test_teacher_chunks_are_byte_identical_for_one_and_many_workers(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            teacher = directory / "teacher.py"
            teacher.write_text(
                "#!/usr/bin/env python3\n"
                "import json,sys\n"
                "for line in sys.stdin:\n"
                " line=line.rstrip('\\n')\n"
                " if not line or line.startswith('#') or line.startswith('group_id\\t'):\n"
                "  continue\n"
                " group,source,winner,transcript=line.split('\\t')\n"
                " print(json.dumps({'schema':'papersoccer.jacek-replay-teacher.v1',"
                "'group_id':group,'source':source,'winner':int(winner)},"
                "sort_keys=True,separators=(',',':')))\n"
            )
            os.chmod(teacher, 0o755)
            teacher_input = directory / "teacher.tsv"
            teacher_input.write_text(
                "# fixture\n"
                "group_id\tsource\twinner\ttranscript\n"
                + "".join(
                    f"group-{index}\tfixture\t{index % 2}\t012/345\n"
                    for index in range(11)
                )
            )
            payloads = []
            for worker_count in (1, 2, 10):
                attempt = directory / f"workers-{worker_count}"
                output = attempt / "labels.jsonl"
                result = workflow.run_teacher_chunks(
                    manager=self.make_manager(attempt),
                    stage_ordinal=2,
                    stage_name="root-labels",
                    teacher=teacher,
                    input_path=teacher_input,
                    output_path=output,
                    teacher_arguments=["--nodes", "1"],
                    workers=worker_count,
                    chunk_games=3,
                )
                workflow.validate_teacher_chunks_result(result)
                payloads.append(output.read_bytes())
            self.assertEqual(payloads[0], payloads[1])
            self.assertEqual(payloads[0], payloads[2])

    def test_interrupted_teacher_merge_resumes_from_chunk_receipts(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            teacher = directory / "teacher.py"
            teacher.write_text(
                "#!/usr/bin/env python3\n"
                "import json,sys\n"
                "for line in sys.stdin:\n"
                " if line.startswith('g') and '\\t' in line:\n"
                "  print(json.dumps({'schema':'papersoccer.jacek-replay-teacher.v1'}))\n"
            )
            os.chmod(teacher, 0o755)
            teacher_input = directory / "teacher.tsv"
            teacher_input.write_text(
                "group_id\tsource\twinner\ttranscript\n"
                + "".join(f"g{i}\tfixture\t0\t012\n" for i in range(5))
            )
            attempt = directory / "attempt"
            output = attempt / "labels.jsonl"
            arguments = dict(
                stage_ordinal=2,
                stage_name="root-labels",
                teacher=teacher,
                input_path=teacher_input,
                output_path=output,
                teacher_arguments=[],
                workers=2,
                chunk_games=2,
            )
            first = workflow.run_teacher_chunks(
                manager=self.make_manager(attempt), **arguments
            )
            receipt_stats = {
                item["receipt_path"]: pathlib.Path(item["receipt_path"]).stat().st_mtime_ns
                for item in first["chunks"]
            }
            expected = output.read_bytes()
            output.unlink()
            second = workflow.run_teacher_chunks(
                manager=self.make_manager(attempt, resume=True), **arguments
            )
            self.assertEqual(output.read_bytes(), expected)
            self.assertEqual(first, second)
            self.assertEqual(
                receipt_stats,
                {
                    path: pathlib.Path(path).stat().st_mtime_ns
                    for path in receipt_stats
                },
            )

    def test_teacher_chunk_receipt_corruption_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            teacher = directory / "teacher.py"
            teacher.write_text(
                "#!/usr/bin/env python3\n"
                "import json,sys\n"
                "for line in sys.stdin:\n"
                " if line.startswith('g\\t'):\n"
                "  print(json.dumps({'schema':'papersoccer.jacek-replay-teacher.v1'}))\n"
            )
            os.chmod(teacher, 0o755)
            teacher_input = directory / "teacher.tsv"
            teacher_input.write_text(
                "group_id\tsource\twinner\ttranscript\n"
                "g\tfixture\t0\t012\n"
            )
            attempt = directory / "attempt"
            result = workflow.run_teacher_chunks(
                manager=self.make_manager(attempt),
                stage_ordinal=2,
                stage_name="root-labels",
                teacher=teacher,
                input_path=teacher_input,
                output_path=attempt / "labels.jsonl",
                teacher_arguments=[],
                workers=1,
                chunk_games=1,
            )
            pathlib.Path(result["chunks"][0]["receipt_path"]).write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "chunk binding is stale"):
                workflow.validate_teacher_chunks_result(result)

    def test_training_directory_is_published_atomically(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            final = directory / "model-unbound"
            command = ["trainer", "--output-directory", str(final)]

            def fake_run(staged):
                scratch = pathlib.Path(
                    staged[staged.index("--output-directory") + 1]
                )
                (scratch / "runtime").write_bytes(b"model")
                return json.dumps({"manifest": str(scratch / "manifest.json")})

            with mock.patch.object(workflow, "run", side_effect=fake_run):
                stdout = workflow.run_with_atomic_directory(
                    command, output_flag="--output-directory", output=final
                )
            self.assertEqual((final / "runtime").read_bytes(), b"model")
            self.assertEqual(json.loads(stdout)["manifest"], str(final / "manifest.json"))

    def test_failed_training_never_leaves_unbound_output_collision(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            final = directory / "model-unbound"
            command = ["trainer", "--output-directory", str(final)]

            def fail_after_start(staged):
                scratch = pathlib.Path(
                    staged[staged.index("--output-directory") + 1]
                )
                (scratch / "partial").write_bytes(b"partial")
                raise RuntimeError("interrupted trainer")

            with mock.patch.object(workflow, "run", side_effect=fail_after_start):
                with self.assertRaisesRegex(RuntimeError, "interrupted trainer"):
                    workflow.run_with_atomic_directory(
                        command, output_flag="--output-directory", output=final
                    )
            self.assertFalse(final.exists())
            self.assertEqual(
                list(directory.glob(".model-unbound.inprogress.*")), []
            )


if __name__ == "__main__":
    unittest.main()
