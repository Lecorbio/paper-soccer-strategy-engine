import importlib
import hashlib
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
rebuild = importlib.import_module("jacek_replay_rebuild")
selfsearch = importlib.import_module("jacek_selfsearch_workflow")
retention = importlib.import_module("jacek_replay_retention")


class ReplayRebuildTests(unittest.TestCase):
    def runtime(self, root: pathlib.Path, name: str) -> pathlib.Path:
        path = root / name
        path.write_bytes(name.encode())
        return path

    def bank(
        self, root: pathlib.Path, name: str, classification: str,
        seed: int, state_prefix: str,
    ) -> pathlib.Path:
        path = root / name
        lines = [
            "# papersoccer.jacek-replay-bfm-opening-bank.v1",
            "# rules=8x10;own-goals-allowed;mover-loses",
            f"# classification={classification}",
            f"# seed={seed}",
            "# minimum-physical-plies=12",
            "opening_id\ttranscript\tstate_identity",
        ]
        lines.extend(
            f"opening-{index}\t0/1\t{state_prefix}:{index}"
            for index in range(rebuild.FULL_SCREEN_PAIRS)
        )
        path.write_text("\n".join(lines) + "\n")
        return path

    def comparison(self, *, wins: int, colors: tuple[int, int], p99: float = 20.0):
        results = []
        color_wins = list(colors)
        for index in range(600):
            color = index % 2
            won = color_wins[color] > 0
            if won:
                color_wins[color] -= 1
            results.append(
                {
                    "candidate_player": color,
                    "winner": color if won else 1 - color,
                    "illegal": False,
                    "candidate_ms": [p99],
                }
            )
        self.assertEqual(sum(g["winner"] == g["candidate_player"] for g in results), wins)
        return {"results": results}

    def qualification_fixture(self, root: pathlib.Path) -> dict[str, object]:
        inputs_path = root / "rebuild-inputs.json"
        inputs_path.write_text("{}\n")
        corpus_path = root / "corpus.json"
        corpus_path.write_text("{}\n")
        comparison = self.runtime(root, "comparison")
        incumbent = self.runtime(root, "incumbent.runtime")
        candidate = self.runtime(root, "candidate.runtime")
        matched = self.runtime(root, "matched.runtime")
        final_bank = self.runtime(root, "sealed-final.tsv")
        banks_path = root / "opening-banks.json"
        banks_path.write_text(
            json.dumps({"final": {"artifact": rebuild.artifact_snapshot(final_bank)}})
        )
        selected_path = root / "selected.json"
        selected = {
            "schema": "papersoccer.jacek-replay-rebuild-selected-candidate.v1",
            "protected_test_opened": False,
            "sealed_final_bank_opened": False,
            "blind_holdout_labels_opened": False,
            "selected_runtime": rebuild.artifact_snapshot(candidate),
            "matched_runtime": rebuild.artifact_snapshot(matched),
        }
        selected_path.write_text(json.dumps(selected))
        canonical_metrics = {
            "schema": "papersoccer.jacek-selfsearch-anchor-metrics.v1",
            "candidate_metrics": {
                "sign_accuracy": 0.8,
                "weighted_huber": 0.1,
            },
            "incumbent_metrics": {
                "sign_accuracy": 0.8,
                "weighted_huber": 0.1,
            },
        }
        canonical_path = root / "canonical-test.json"
        canonical_path.write_text(json.dumps(canonical_metrics))
        holdout_path = root / "blind-holdout.json"
        holdout_path.write_text(json.dumps({"pass": True}))
        report_paths = {}
        for name in ("matched", "incumbent", "rank4", "jacek-nn"):
            report_paths[name] = root / f"final-{name}.json"
            report_paths[name].write_text(json.dumps({"panel": name}))
        latency_path = root / "final-latency.json"
        latency_path.write_text(json.dumps({"candidate_max_ms": 999.0}))
        decision = {
            "schema": "papersoccer.jacek-selfsearch-pilot-decision.v1",
            "eligible_for_full": True,
        }
        full = {
            "candidate": rebuild.artifact_snapshot(candidate),
            "matched": rebuild.artifact_snapshot(matched),
            "bank": rebuild.artifact_snapshot(final_bank),
            "classification": "final",
            "reports": {
                name: rebuild.artifact_snapshot(path)
                for name, path in report_paths.items()
            },
            "latency": rebuild.artifact_snapshot(latency_path),
            "decision": decision,
        }
        inputs = {
            "corpus": rebuild.artifact_snapshot(corpus_path),
            "incumbent_runtime": rebuild.artifact_snapshot(incumbent),
            "opening_banks": rebuild.artifact_snapshot(banks_path),
            "comparison": rebuild.artifact_snapshot(comparison),
            "repository": {"path": str(root)},
        }
        body = {
            "schema": rebuild.REBUILD_QUALIFICATION_SCHEMA,
            "rebuild_id": rebuild.REBUILD_ID,
            "inputs": rebuild.artifact_snapshot(inputs_path),
            "selection": rebuild.artifact_snapshot(selected_path),
            "candidate": rebuild.artifact_snapshot(candidate),
            "matched": rebuild.artifact_snapshot(matched),
            "canonical_test": rebuild.artifact_snapshot(canonical_path),
            "blind_holdout": rebuild.artifact_snapshot(holdout_path),
            "final_bank": rebuild.artifact_snapshot(final_bank),
            "final_game_gate": full,
            "pass": True,
            "local_only": True,
            "canonical_rank4_replaced": False,
            "external_upload": False,
        }
        qualification_path = root / "qualification.json"
        qualification_path.write_bytes(
            rebuild.canonical_json_bytes(
                {
                    **body,
                    "body_sha256": rebuild.sha256_bytes(
                        rebuild.canonical_json_bytes(body)
                    ),
                },
                pretty=True,
            )
        )
        return {
            "path": qualification_path,
            "inputs": inputs,
            "selected_path": selected_path,
            "canonical_path": canonical_path,
            "canonical_metrics": canonical_metrics,
            "decision": decision,
        }

    def qualification_patches(self, fixture: dict[str, object]):
        corpus = SimpleNamespace(protected_test_manifest_paths=(pathlib.Path("test.json"),))
        return (
            mock.patch.object(
                rebuild, "validate_rebuild_inputs",
                return_value=fixture["inputs"],
            ),
            mock.patch.object(rebuild, "validate_selected_candidate_lineage"),
            mock.patch.object(
                rebuild, "load_frozen_rebuild_corpus", return_value=corpus
            ),
            mock.patch("jacek_selfsearch_workflow.anchor_metrics", return_value=fixture["canonical_metrics"]),
            mock.patch.object(rebuild, "_validate_qualification_holdout", return_value={"pass": True}),
            mock.patch.object(rebuild, "validate_opening_banks"),
            mock.patch.object(rebuild, "_comparison_report", return_value={}),
            mock.patch("jacek_selfsearch_workflow._source_identities", return_value={}),
            mock.patch(
                "jacek_selfsearch_workflow.run_latency_audit",
                return_value={"candidate_max_ms": 999.0},
            ),
            mock.patch(
                "jacek_selfsearch_workflow.pilot_decision",
                return_value=fixture["decision"],
            ),
        )

    def test_qualification_replays_protected_metrics_and_rejects_rehashed_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.qualification_fixture(pathlib.Path(directory))
            patches = self.qualification_patches(fixture)
            with patches[0], patches[1], patches[2], patches[3], patches[4], \
                    patches[5], patches[6], patches[7], patches[8], patches[9]:
                self.assertTrue(
                    rebuild.validate_qualification_receipt(fixture["path"])["pass"]
                )

            tampered_metrics = dict(fixture["canonical_metrics"])
            tampered_metrics["candidate_metrics"] = {
                "sign_accuracy": 0.0,
                "weighted_huber": 99.0,
            }
            fixture["canonical_path"].write_text(json.dumps(tampered_metrics))
            qualification = json.loads(fixture["path"].read_text())
            qualification["canonical_test"] = rebuild.artifact_snapshot(
                fixture["canonical_path"]
            )
            qualification.pop("body_sha256")
            qualification["body_sha256"] = rebuild.sha256_bytes(
                rebuild.canonical_json_bytes(qualification)
            )
            fixture["path"].write_bytes(
                rebuild.canonical_json_bytes(qualification, pretty=True)
            )
            patches = self.qualification_patches(fixture)
            with patches[0], patches[1], patches[2], patches[3], patches[4], \
                    patches[5], patches[6], patches[7], patches[8], patches[9]:
                with self.assertRaisesRegex(ValueError, "canonical-test evidence"):
                    rebuild.validate_qualification_receipt(fixture["path"])

    def test_qualification_rejects_reveal_before_any_protected_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.qualification_fixture(pathlib.Path(directory))
            selected = json.loads(fixture["selected_path"].read_text())
            selected["blind_holdout_labels_opened"] = True
            fixture["selected_path"].write_text(json.dumps(selected))
            qualification = json.loads(fixture["path"].read_text())
            qualification["selection"] = rebuild.artifact_snapshot(
                fixture["selected_path"]
            )
            qualification.pop("body_sha256")
            qualification["body_sha256"] = rebuild.sha256_bytes(
                rebuild.canonical_json_bytes(qualification)
            )
            fixture["path"].write_bytes(
                rebuild.canonical_json_bytes(qualification, pretty=True)
            )
            patches = self.qualification_patches(fixture)
            with patches[0], patches[1] as lineage, patches[2], \
                    patches[3] as protected_test, patches[4] as holdout, \
                    patches[5], patches[6], patches[7], patches[8], patches[9]:
                with self.assertRaisesRegex(ValueError, "reveal order"):
                    rebuild.validate_qualification_receipt(fixture["path"])
                lineage.assert_not_called()
                protected_test.assert_not_called()
                holdout.assert_not_called()

    def test_qualification_holdout_rechecks_every_fixed_work_label(self):
        import numpy as np
        import jacek_replay_retention as retention

        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            corpus_path = root / "corpus.json"
            corpus_path.write_text("{}\n")
            freeze_path = root / "freeze.json"
            freeze_path.write_text("{}\n")
            positions_path = root / "positions.tsv"
            positions_path.write_text("positions\n")
            labels_path = root / "labels.jsonl"
            labels_path.write_text("label\n")
            shard_path = root / "holdout-shard.json"
            shard_path.write_text("{}\n")
            selection_path = root / "selected.json"
            selection_path.write_text("{}\n")
            candidate = self.runtime(root, "holdout-candidate.runtime")
            incumbent = self.runtime(root, "holdout-incumbent.runtime")
            teacher = self.runtime(root, "rank4-teacher")
            inputs = {
                "corpus": rebuild.artifact_snapshot(corpus_path),
                "blind_holdout": rebuild.artifact_snapshot(freeze_path),
                "blind_holdout_positions": rebuild.artifact_snapshot(positions_path),
                "rank4_teacher": rebuild.artifact_snapshot(teacher),
                "repository": {"path": str(root)},
            }
            selection_snapshot = retention.artifact_snapshot(selection_path)
            freeze = {
                "campaign_id": rebuild.REBUILD_ID,
                "profile": "rebuild",
                "role": "retention-rebuild",
                "training_eligible": False,
                "configuration": {
                    "groups": 600,
                    "rows_per_group": 20,
                    "selection_seed": rebuild.BLIND_HOLDOUT_SEED,
                },
                "timing": {
                    "training_inputs_frozen_by": retention.artifact_snapshot(corpus_path),
                    "teacher_labels_opened": False,
                    "selected_model_opened": False,
                    "required_reveal_order": (
                        "freeze-before-model-selection;labels-after-model-selection;"
                        "metrics-after-selected-runtime-binding"
                    ),
                },
            }
            arrays = {
                "indptr": np.asarray([0, 1], dtype="<i8"),
                "indices": np.asarray([1], dtype="<u2"),
                "targets": np.asarray([0.0], dtype="<f4"),
                "weights": np.asarray([1.0], dtype="<f4"),
                "root_group_ids": np.asarray([b"r" * 32], dtype="V32"),
                "position_ids": np.asarray([b"p" * 32], dtype="V32"),
                "canonical_fingerprints": np.asarray([b"f" * 32], dtype="V32"),
                "orientations": np.asarray([0], dtype="u1"),
            }
            teacher_configuration = {
                "teacher": {"kind": "rank4-fixed-work"},
                "search_config": retention.RANK4_FIXED_CONFIGURATION,
            }
            termination_counts = {"fixed-work-cap": 12_000}
            shard_manifest = {
                "campaign_id": rebuild.REBUILD_ID,
                "profile": "rebuild",
                "role": "retention-rebuild",
                "training_eligible": False,
                "training_loader_compatible": False,
                "base_positions": 12_000,
                "samples": 24_000,
                "root_groups": 600,
                "inputs": {
                    "freeze_manifest": retention.artifact_snapshot(freeze_path),
                    "frozen_positions": retention.artifact_snapshot(positions_path),
                    "labels": retention.artifact_snapshot(labels_path),
                },
                "reveal": {
                    "policy": "labels-opened-only-after-model-selection-receipt",
                    "selection_receipt": selection_snapshot,
                    "training_input_receipt": retention.artifact_snapshot(corpus_path),
                },
                "termination_counts": termination_counts,
                "teacher_configuration": teacher_configuration,
            }
            shard = SimpleNamespace(manifest=shard_manifest, **arrays)
            gates = {
                "point_sign": True,
                "point_huber": True,
                "cluster_sign": True,
                "cluster_huber": True,
                "pass": True,
            }
            evidence = {
                "inputs": {
                    "shard_manifest": retention.artifact_snapshot(shard_path),
                },
                "noninferiority": {
                    "thresholds": retention.FROZEN_THRESHOLDS.record(),
                    "root_groups": 600,
                    "gates": gates,
                },
                "pass": True,
            }
            evidence_path = root / "evidence.json"
            evidence_path.write_text(json.dumps(evidence))
            with mock.patch.object(
                rebuild, "load_frozen_blind_holdout",
                return_value=(freeze, [object()] * 12_000),
            ), mock.patch(
                "jacek_replay_retention.load_holdout_shard", return_value=shard,
            ), mock.patch(
                "jacek_selfsearch_workflow._source_identities",
                return_value={"rank4_teacher_source_sha256": "a" * 64},
            ), mock.patch.object(
                rebuild, "_validate_qualification_label_receipts",
            ), mock.patch(
                "jacek_selfsearch_workflow._validate_label_output",
                return_value=12_000,
            ) as validate_labels, mock.patch(
                "jacek_replay_retention._load_rank4_labels", return_value={},
            ), mock.patch(
                "jacek_replay_retention._holdout_arrays",
                return_value=(arrays, termination_counts, teacher_configuration),
            ), mock.patch(
                "jacek_replay_retention.evaluate_holdout", return_value=evidence,
            ):
                self.assertTrue(
                    rebuild._validate_qualification_holdout(
                        evidence_path=evidence_path,
                        inputs=inputs,
                        selection_path=selection_path,
                        candidate=candidate,
                        incumbent=incumbent,
                    )["pass"]
                )
                self.assertEqual(
                    validate_labels.call_args.kwargs["nodes"],
                    retention.RANK4_FIXED_NODES,
                )
                self.assertEqual(
                    validate_labels.call_args.kwargs["campaign_id"],
                    rebuild.REBUILD_ID,
                )

    def test_qualification_label_receipts_bind_complete_teacher_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            positions = root / "positions.tsv"
            lines = ["header", *(f"row-{index}" for index in range(12_000))]
            positions.write_text("\n".join(lines) + "\n")
            teacher = self.runtime(root, "rank4-teacher")
            label_root = root / "labeling"
            labels = label_root / "labels.jsonl"
            merged = bytearray()
            for index, start in enumerate(range(1, len(lines), 25)):
                rows = lines[start : start + 25]
                source = label_root / "inputs" / f"chunk-{index:04d}.tsv"
                output = label_root / "outputs" / f"chunk-{index:04d}.jsonl"
                receipt = label_root / "receipts" / f"chunk-{index:04d}.json"
                source.parent.mkdir(parents=True, exist_ok=True)
                output.parent.mkdir(parents=True, exist_ok=True)
                receipt.parent.mkdir(parents=True, exist_ok=True)
                source.write_text(lines[0] + "\n" + "\n".join(rows) + "\n")
                output.write_text(f"label-{index}\n")
                merged.extend(output.read_bytes())
                receipt.write_text(json.dumps({
                    "schema": "papersoccer.jacek-rebuild-holdout-label-chunk.v1",
                    "rebuild_id": rebuild.REBUILD_ID,
                    "chunk": index,
                    "configuration": {"nodes": 400_000, "time_ms": 0},
                    "teacher": rebuild.artifact_snapshot(teacher),
                    "teacher_source_sha256": "a" * 64,
                    "positions": rebuild.artifact_snapshot(source),
                    "output": rebuild.artifact_snapshot(output),
                }))
            labels.write_bytes(bytes(merged))
            with mock.patch(
                "jacek_selfsearch_workflow._validate_label_output",
                return_value=25,
            ) as validate_label:
                rebuild._validate_qualification_label_receipts(
                    labels=labels,
                    positions=positions,
                    teacher=teacher,
                    teacher_source_sha256="a" * 64,
                )
                self.assertEqual(validate_label.call_count, 480)

            damaged_path = label_root / "receipts" / "chunk-0000.json"
            damaged = json.loads(damaged_path.read_text())
            damaged["configuration"]["nodes"] = 1
            damaged_path.write_text(json.dumps(damaged))
            with mock.patch(
                "jacek_selfsearch_workflow._validate_label_output",
                return_value=25,
            ):
                with self.assertRaisesRegex(ValueError, "receipt changed"):
                    rebuild._validate_qualification_label_receipts(
                        labels=labels,
                        positions=positions,
                        teacher=teacher,
                        teacher_source_sha256="a" * 64,
                    )

    def test_matrix_freezes_exact_candidate_ladder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            v5s = self.runtime(root, "v5-search.runtime")
            v5r = self.runtime(root, "v5-rank4.runtime")
            bases = [
                (f"r{round_index}-s{seed}", self.runtime(root, f"r{round_index}-s{seed}.runtime"))
                for round_index in range(3)
                for seed in range(3)
            ]
            record = rebuild.matrix_record(
                v5_search=v5s, v5_rank4=v5r, canonical_bases=bases
            )
            rebuild.validate_matrix(record)
            self.assertEqual(len(record["phases"]["v5_recovery"]), 9)
            self.assertEqual(len(record["phases"]["canonical_basins"]), 9)
            self.assertEqual(
                record["phases"]["scratch_pretraining"]["seeds"],
                list(range(20261001, 20261007)),
            )
            tampered = json.loads(json.dumps(record))
            tampered["same_architecture_budget_seconds"] = 1
            with self.assertRaisesRegex(ValueError, "stale or corrupt"):
                rebuild.validate_matrix(tampered)
            rehashed = json.loads(json.dumps(record))
            rehashed["phases"]["v5_recovery"].pop()
            rehashed.pop("body_sha256")
            rehashed["body_sha256"] = rebuild.sha256_bytes(
                rebuild.canonical_json_bytes(rehashed)
            )
            with self.assertRaisesRegex(ValueError, "incomplete|semantics"):
                rebuild.validate_matrix(rehashed)

    def test_canonical_sweep_requires_nine_distinct_bases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            path = self.runtime(root, "base.runtime")
            with self.assertRaisesRegex(ValueError, "exactly nine"):
                rebuild.canonical_basin_specs([("base", path)])

    def test_development_gate_thresholds_remain_strict(self):
        passing = {
            "matched": self.comparison(wins=325, colors=(169, 156)),
            "incumbent": self.comparison(wins=325, colors=(156, 169)),
            "rank4": self.comparison(wins=306, colors=(163, 143)),
            "jacek-nn": self.comparison(wins=306, colors=(143, 163)),
        }
        self.assertTrue(
            rebuild.development_passes(passing, uncontended_max_ms=999.0)
        )
        failing = dict(passing)
        failing["matched"] = self.comparison(wins=324, colors=(168, 156))
        self.assertFalse(
            rebuild.development_passes(failing, uncontended_max_ms=999.0)
        )
        self.assertFalse(
            rebuild.development_passes(passing, uncontended_max_ms=1_000.0)
        )

    def test_short_screen_orders_by_worst_panel(self):
        stronger = rebuild.short_screen_key(
            self.comparison(wins=110, colors=(55, 55)),
            self.comparison(wins=108, colors=(54, 54)),
            "a" * 64,
        )
        weaker = rebuild.short_screen_key(
            self.comparison(wins=120, colors=(60, 60)),
            self.comparison(wins=101, colors=(51, 50)),
            "b" * 64,
        )
        self.assertLess(stronger, weaker)

    def test_budget_is_hard_capped(self):
        self.assertTrue(rebuild.budget_allows_new_work(100.0, now=100.0))
        self.assertFalse(
            rebuild.budget_allows_new_work(
                100.0, now=100.0 + rebuild.SAME_ARCHITECTURE_BUDGET_SECONDS
            )
        )

    def test_git_head_resolution_never_needs_a_git_subprocess(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = pathlib.Path(directory)
            git_directory = repository / ".git"
            git_directory.mkdir()
            first = "1" * 40
            (git_directory / "HEAD").write_text(first + "\n")
            self.assertEqual(rebuild._git_head_commit(repository), first)

            second = "2" * 40
            (git_directory / "refs/heads").mkdir(parents=True)
            (git_directory / "refs/heads/rebuild").write_text(second + "\n")
            (git_directory / "HEAD").write_text("ref: refs/heads/rebuild\n")
            self.assertEqual(rebuild._git_head_commit(repository), second)

    def test_launch_receipt_allows_only_predeadline_work_to_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            corpus = self.runtime(root, "corpus.json")
            incumbent = self.runtime(root, "incumbent.runtime")
            initial = self.runtime(root, "initial.runtime")
            spec = rebuild.CandidateSpec(
                candidate_id="v5-recovery-w3-lr0",
                phase="v5-recovery",
                base_id="v5",
                search_initial_runtime=initial,
                rank4_initial_runtime=initial,
                trainable_layers="w3",
                learning_rate=3e-6,
                selection_policy="v5-recovery-noninferiority",
                training_recipe="recovery",
            )
            with mock.patch.object(rebuild.time, "time", return_value=100.0):
                launched = rebuild._recovery_launch_receipt(
                    spec=spec,
                    corpus_manifest=corpus,
                    incumbent_runtime=incumbent,
                    deadline_unix=110.0,
                    output_directory=root / "phase",
                    allow_create=True,
                )
            self.assertIsNotNone(launched)
            with mock.patch.object(rebuild.time, "time", return_value=120.0):
                resumed = rebuild._recovery_launch_receipt(
                    spec=spec,
                    corpus_manifest=corpus,
                    incumbent_runtime=incumbent,
                    deadline_unix=110.0,
                    output_directory=root / "phase",
                    allow_create=True,
                )
                blocked = rebuild._recovery_launch_receipt(
                    spec=rebuild.dataclasses.replace(
                        spec, candidate_id="v5-recovery-w3-lr1"
                    ),
                    corpus_manifest=corpus,
                    incumbent_runtime=incumbent,
                    deadline_unix=110.0,
                    output_directory=root / "phase",
                    allow_create=True,
                )
            self.assertEqual(resumed, launched)
            self.assertIsNone(blocked)

    def test_phase_order_requires_an_exact_ladder_prefix(self):
        self.assertTrue(
            rebuild.ladder_phase_order_is_valid(
                ["v5-recovery", "canonical-basins"], "canonical-basins"
            )
        )
        self.assertTrue(
            rebuild.ladder_phase_order_is_valid(
                ["v5-recovery", "residual"], "residual"
            )
        )
        self.assertFalse(
            rebuild.ladder_phase_order_is_valid(
                ["canonical-basins"], "canonical-basins"
            )
        )
        self.assertFalse(
            rebuild.ladder_phase_order_is_valid(["residual"], "residual")
        )
        self.assertFalse(
            rebuild.ladder_phase_order_is_valid(
                ["v5-recovery", "scratch-joint", "residual"], "residual"
            )
        )

    def test_official_full_gate_key_cannot_swap_in_second_finalist(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            candidates = []
            for name in ("stronger", "weaker"):
                runtime = self.runtime(root, f"{name}.runtime")
                candidates.append(
                    {
                        "candidate_id": name,
                        "selected_search": {
                            "runtime": rebuild.artifact_snapshot(runtime)
                        },
                    }
                )

            def full(wins: int):
                return {
                    "decision": {
                        "eligible_for_full": True,
                        "counts": {
                            "matched": {"wins": wins, "colors": [wins // 2, wins - wins // 2]},
                            "incumbent": {"wins": wins, "colors": [wins // 2, wins - wins // 2]},
                            "rank4": {"wins": wins, "colors": [wins // 2, wins - wins // 2]},
                            "jacek-nn": {"wins": wins, "colors": [wins // 2, wins - wins // 2]},
                        },
                        "candidate_p99_ms": 20.0,
                        "uncontended_max_ms": 900.0,
                    }
                }

            finalists = [
                (candidates[0], full(350), {"weighted_huber": 0.05}),
                (candidates[1], full(330), {"weighted_huber": 0.04}),
            ]
            ordered = sorted(
                finalists,
                key=lambda item: rebuild._official_qualified_key(*item),
            )
            self.assertEqual(ordered[0][0]["candidate_id"], "stronger")

    def test_short_screen_records_operational_rejection_without_aborting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            comparison = self.runtime(root, "comparison")
            candidate = self.runtime(root, "candidate.runtime")
            matched = self.runtime(root, "matched.runtime")
            incumbent = self.runtime(root, "incumbent.runtime")
            bank = self.runtime(root, "bank.tsv")
            (root / "screen").mkdir()
            self.runtime(root / "screen", "matched.json")
            self.runtime(root / "screen", "incumbent.json")
            bad = self.comparison(wins=325, colors=(169, 156), p99=26.0)
            bad["results"][0]["illegal"] = True
            good = self.comparison(wins=325, colors=(169, 156), p99=20.0)
            with mock.patch.object(
                rebuild, "_comparison_report", side_effect=(bad, good)
            ):
                record = rebuild.run_short_screen(
                    comparison=comparison,
                    candidate=candidate,
                    matched=matched,
                    incumbent=incumbent,
                    bank=bank,
                    output_directory=root / "screen",
                )
            self.assertFalse(record["operational"])
            self.assertIsNone(record["ranking_key"])
            self.assertEqual(
                record["rejection_reasons"],
                ["matched:illegal", "matched:slow"],
            )

    def test_holdout_generation_is_identical_with_1_2_or_10_workers(self):
        samples = [
            ("0/7/1/2/7/43/00/3/56/413", 0),
            ("0/7/1/2/7/43/00/3/56/413/2", 1),
            ("0/7/1/2/7/43/00/3/56/413/2/36/5", 1),
            ("0/7/1/2/7/43/00/3/56/413/2/36/5/7/1336", 1),
            ("0/7/1/2/7/43/00/3/56/413/2/36/5/7/1336/65", 0),
            ("0/7/1/2/7/43/00/3/56/413/2/36/5/7/1336/65/6/3", 0),
            ("0/7/1/2/7/43/00/3/56/413/2/36/5/7/1336/65/6/3/05/4", 0),
            ("0/7/1/2/7/43/00/3/56/413/2/36/5/7/1336/65/6/3/05/4/57", 1),
            ("0/7/1/2/7/43/00/3/56/413/2/36/5/7/1336/65/6/3/05/4/57/41/6572", 1),
            ("0/7/1/2/7/43/00/3/56/413/2/36/5/7/1336/65/6/3/05/4/57/41/6572/72/3601", 1),
            ("0/7/1/2/7/43/00/3/56/413/2/36/5/7/1336/65/6/3/05/4/57/41/6572/72/3601/7", 0),
            ("0/7/1/2/7/43/00/3/56/413/2/36/5/7/1336/65/6/3/05/4/57/41/6572/72/3601/7/2/0", 0),
            ("0/7/1/2/7/43/00/3/56/413/2/36/5/7/1336/65/6/3/05/4/57/41/6572/72/3601/7/2/0/552/146720", 0),
            ("0/7/1/2/7/43/00/3/56/413/2/36/5/7/1336/65/6/3/05/4/57/41/6572/72/3601/7/2/0/552/146720/72", 1),
            ("0/7/1/2/7/43/00/3/56/413/2/36/5/7/1336/65/6/3/05/4/57/41/6572/72/3601/7/2/0/552/146720/72/46171/0", 1),
            ("0/7/1/2/7/43/00/3/56/413/2/36/5/7/1336/65/6/3/05/4/57/41/6572/72/3601/7/2/0/552/146720/72/46171/0/614524/43631", 1),
            ("0/7/1/2/7/43/00/3/56/413/2/36/5/7/1336/65/6/3/05/4/57/41/6572/72/3601/7/2/0/552/146720/72/46171/0/614524/43631/222", 0),
            ("0/7/1/2/7/43/00/3/56/413/2/36/5/7/1336/65/6/3/05/4/57/41/6572/72/3601/7/2/0/552/146720/72/46171/0/614524/43631/222/125634/6", 0),
            ("0/7/1/2/7/43/00/3/56/413/2/36/5/7/1336/65/6/3/05/4/57/41/6572/72/3601/7/2/0/552/146720/72/46171/0/614524/43631/222/125634/6/5567/1", 0),
            ("0/7/1/2/7/43/00/3/56/413/2/36/5/7/1336/65/6/3/05/4/57/41/6572/72/3601/7/2/0/552/146720/72/46171/0/614524/43631/222/125634/6/5567/1/2470", 1),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            generator = root / "fake-generator.py"
            generator.write_text(
                "#!/usr/bin/env python3\n"
                "import json, pathlib, sys\n"
                f"samples = {samples!r}\n"
                "seed = int(sys.argv[5])\n"
                "value = {'schema':'papersoccer.teacher-residual-samples.v1',"
                "'seed':seed,'winner':1,'samples':[{'transcript':p,'player_id':m} "
                "for p,m in samples]}\n"
                "pathlib.Path(sys.argv[1]).write_text(json.dumps(value,"
                "sort_keys=True,separators=(',',':'))+'\\n')\n",
                encoding="utf-8",
            )
            generator.chmod(0o755)
            payloads = []
            with mock.patch.object(rebuild, "HOLDOUT_SOURCE_GAMES", 10), mock.patch.object(
                rebuild, "HOLDOUT_CANDIDATE_GROUPS", 10
            ):
                for workers in (1, 2, 10):
                    output = root / f"workers-{workers}"
                    rebuild.generate_holdout_candidate_positions(
                        generator=generator,
                        output_directory=output,
                        workers=workers,
                    )
                    rebuild.load_frozen_holdout_candidate_pool(
                        output / "candidate-positions.tsv",
                        output / "candidate-positions.json",
                    )
                    payloads.append((output / "candidate-positions.tsv").read_bytes())
                cache = root / "cache"
                cache.mkdir()
                fingerprints = cache / "canonical-fingerprints.bin"
                groups = cache / "root-groups.txt"
                fingerprints.write_bytes(b"x" * 32)
                groups.write_text("excluded-root\n")
                cache_body = {
                    "schema": "papersoccer.jacek-rebuild-holdout-exclusion-cache.v1",
                    "rebuild_id": rebuild.REBUILD_ID,
                    "inputs": {
                        "excluded_shards": [],
                        "excluded_positions": [],
                        "excluded_roots": [],
                    },
                    "producer": retention._producer_identity(),
                    "exclusion_universe": {
                        "root_groups": 1,
                        "canonical_fingerprints": 1,
                        "root_group_ids_sha256": hashlib.sha256(
                            b"excluded-root"
                        ).hexdigest(),
                        "canonical_fingerprints_sha256": hashlib.sha256(
                            b"x" * 32
                        ).hexdigest(),
                    },
                    "artifacts": {
                        "root_groups": retention.artifact_snapshot(groups),
                        "canonical_fingerprints": retention.artifact_snapshot(
                            fingerprints
                        ),
                    },
                }
                cache_receipt = cache / "receipt.json"
                cache_receipt.write_bytes(
                    rebuild.canonical_json_bytes(
                        {
                            **cache_body,
                            "body_sha256": rebuild.sha256_bytes(
                                rebuild.canonical_json_bytes(cache_body)
                            ),
                        },
                        pretty=True,
                    )
                )
                with mock.patch.object(
                    rebuild, "FILTERED_HOLDOUT_CANDIDATE_GROUPS", 1
                ):
                    filtered = root / "filtered"
                    rebuild.filter_holdout_candidate_positions(
                        source_positions=root / "workers-1/candidate-positions.tsv",
                        source_manifest=root / "workers-1/candidate-positions.json",
                        exclusion_cache_receipt=cache_receipt,
                        output_directory=filtered,
                    )
                    rebuild.load_frozen_holdout_candidate_pool(
                        filtered / "candidate-positions.tsv",
                        filtered / "candidate-positions.json",
                    )
            self.assertEqual(payloads[0], payloads[1])
            self.assertEqual(payloads[0], payloads[2])

    def test_final_bank_manifest_is_sealed_and_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            comparison = self.runtime(root, "comparison")
            exclusion = self.bank(
                root, "excluded.tsv", "development", 1, "excluded"
            )
            development = self.bank(
                root, "development.tsv", "development",
                rebuild.DEVELOPMENT_BANK_SEED, "development",
            )
            final = self.bank(
                root, "final.tsv", "final", rebuild.FINAL_BANK_SEED, "final"
            )
            exclusions = [rebuild.artifact_snapshot(exclusion)]
            detailed_exclusions = [selfsearch.artifact_snapshot(exclusion)]
            development_states = {
                f"development:{index}" for index in range(rebuild.FULL_SCREEN_PAIRS)
            }
            final_states = {
                f"final:{index}" for index in range(rebuild.FULL_SCREEN_PAIRS)
            }
            body = {
                "schema": rebuild.REBUILD_BANKS_SCHEMA,
                "rebuild_id": rebuild.REBUILD_ID,
                "comparison": rebuild.artifact_snapshot(comparison),
                "excluded_banks": exclusions,
                "development": {
                    "artifact": rebuild.artifact_snapshot(development),
                    "configuration": {
                        "pairs": rebuild.FULL_SCREEN_PAIRS,
                        "seed": rebuild.DEVELOPMENT_BANK_SEED,
                        "opening_plies": 12,
                        "classification": "development",
                        "states_sha256": hashlib.sha256(
                            "\n".join(sorted(development_states)).encode()
                        ).hexdigest(),
                        "exclusions": detailed_exclusions,
                    },
                    "model_selection_eligible": True,
                },
                "final": {
                    "artifact": rebuild.artifact_snapshot(final),
                    "configuration": {
                        "pairs": rebuild.FULL_SCREEN_PAIRS,
                        "seed": rebuild.FINAL_BANK_SEED,
                        "opening_plies": 12,
                        "classification": "final",
                        "states_sha256": hashlib.sha256(
                            "\n".join(sorted(final_states)).encode()
                        ).hexdigest(),
                        "exclusions": [
                            *detailed_exclusions,
                            selfsearch.artifact_snapshot(development),
                        ],
                    },
                    "model_selection_eligible": False,
                    "sealed_until_selected_runtime_receipt": True,
                },
            }
            record = {
                **body,
                "body_sha256": rebuild.sha256_bytes(
                    rebuild.canonical_json_bytes(body)
                ),
            }
            rebuild.validate_opening_banks(record)
            exposed_body = dict(body)
            exposed_body["final"] = {
                **body["final"], "model_selection_eligible": True
            }
            exposed = {
                **exposed_body,
                "body_sha256": rebuild.sha256_bytes(
                    rebuild.canonical_json_bytes(exposed_body)
                ),
            }
            with self.assertRaisesRegex(ValueError, "exposed"):
                rebuild.validate_opening_banks(exposed)


if __name__ == "__main__":
    unittest.main()
