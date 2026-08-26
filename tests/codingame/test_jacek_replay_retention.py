import hashlib
import json
import pathlib
import random
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import jacek_replay_features as features  # noqa: E402

try:
    import numpy as np
    import jacek_replay_retention as retention
    import jacek_replay_train as training
except ModuleNotFoundError as error:
    if error.name != "numpy":
        raise
    np = None
    retention = None
    training = None


def write_json(path: pathlib.Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(retention.canonical_json_bytes(value))


def random_game(seed: int) -> tuple[int, list[tuple[str, int]]]:
    randomizer = random.Random(seed)
    state = features.ReplayState()
    actions: list[str] = []
    boundaries: list[tuple[str, int]] = []
    while state.winner is None and len(actions) < 320:
        boundaries.append(("/".join(actions), state.to_move))
        mover = state.to_move
        action = ""
        while state.winner is None and state.to_move == mover:
            legal = []
            for direction, (dx, dy) in enumerate(features.DIRECTION_DELTAS):
                destination = state.ball[0] + dx, state.ball[1] + dy
                if features._legal_destination(state, destination):
                    legal.append(direction)
            if not legal:
                raise AssertionError("fixture generator reached a move-less state")
            direction = randomizer.choice(legal)
            action += str(direction)
            features.apply_primitive(state, direction)
        actions.append(action)
    if state.winner not in (0, 1):
        raise AssertionError("fixture generator did not finish")
    return state.winner, boundaries


def candidate_positions(groups: int) -> bytes:
    lines = [retention.POSITION_HEADER]
    accepted = 0
    seed = 1
    while accepted < groups:
        winner, boundaries = random_game(seed)
        seed += 1
        if len(boundaries) < retention.ROWS_PER_GROUP:
            continue
        # Late boundaries avoid the common initial/opening states shared by
        # independently generated games.
        selected = boundaries[-retention.ROWS_PER_GROUP :]
        root = f"retention-root:{accepted}"
        game = f"retention-game:{accepted}"
        for ordinal, (prefix, mover) in enumerate(selected):
            lines.append(
                "\t".join(
                    (
                        f"position:{accepted:03d}:{ordinal:02d}",
                        root,
                        game,
                        "retention-fixture",
                        retention.TEACHER_SPLIT,
                        str(winner),
                        str(mover),
                        prefix,
                    )
                )
            )
        accepted += 1
    return ("\n".join(lines) + "\n").encode()


def rank4_label(
    row: retention.PositionRow,
    campaign_id: str,
    source_sha256: str,
    *,
    proof: bool = False,
) -> dict:
    if proof:
        completed_depth = 1
        nodes = 17
        root_score = 999_999
        proven_winner = 0
        root_solved = True
        attempted_depth = 1
        budget_exhausted = False
        node_cap_reached = False
        termination_reason = "root-solved"
    else:
        completed_depth = 0
        nodes = retention.RANK4_FIXED_NODES
        root_score = 0
        proven_winner = None
        root_solved = False
        attempted_depth = 1
        budget_exhausted = True
        node_cap_reached = True
        termination_reason = "fixed-work-cap"
    return {
        "schema": retention.corpus.RANK4_TEACHER_SCHEMA,
        "campaign_id": campaign_id,
        "position_id": row.position_id,
        "root_group_id": row.root_group_id,
        "group_id": row.group_id,
        "source": row.source,
        "split": row.split,
        "winner": row.winner,
        "prefix": list(row.prefix_records),
        "mover": row.mover,
        "teacher": {
            "kind": "rank4-fixed-work",
            "source_sha256": source_sha256,
        },
        "search_config": {
            "max_nodes": retention.RANK4_FIXED_NODES,
            "max_time_ms": 0,
            "max_turn_depth": 32,
            "replay_value_blend_percent": 15,
            "teacher_residual_weight_percent": 100,
        },
        "completed_depth": completed_depth,
        "nodes": nodes,
        "root_score": root_score,
        "root_solved": root_solved,
        "proven_winner": proven_winner,
        "search_stats": {
            "attempted_depth": attempted_depth,
            "completed_depth": completed_depth,
            "nodes": nodes,
            "leaf_evaluations": max(1, nodes // 2),
            "terminal_nodes": 0,
            "completed_actions": 1,
            "budget_exhausted": budget_exhausted,
            "node_cap_reached": node_cap_reached,
            "depth_cap_reached": False,
            "deadline_reached": False,
            "termination_reason": termination_reason,
        },
        "weight": 1.0,
    }


@unittest.skipIf(np is None, "research tests require requirements-research.txt")
class JacekReplayRetentionTests(unittest.TestCase):
    def test_production_profiles_freeze_exact_full_size_group_counts(self):
        self.assertEqual(retention.PILOT_SPEC.groups, 600)
        self.assertEqual(retention.FULL_SPEC.groups, 1_200)
        self.assertEqual(retention.PILOT_SPEC.rows_per_group, 20)
        self.assertEqual(retention.FULL_SPEC.rows_per_group, 20)
        self.assertEqual(
            sum(count for _source, count in retention.PILOT_SPEC.source_quotas),
            600,
        )
        self.assertEqual(
            dict(retention.FULL_SPEC.source_quotas),
            {
                source: count * 2
                for source, count in retention.PILOT_SPEC.source_quotas
            },
        )

    def freeze_fixture(
        self, root: pathlib.Path, *, groups: int = 2
    ) -> tuple[pathlib.Path, pathlib.Path, dict]:
        candidate = root / "candidate.tsv"
        candidate.write_bytes(candidate_positions(groups + 2))
        training_receipt = root / "training-inputs.json"
        write_json(training_receipt, {"stage": "training-inputs-frozen"})
        payload, manifest = retention.freeze_candidate_groups(
            candidate_positions=candidate,
            training_input_receipt=training_receipt,
            campaign_id="retention-fixture-v1",
            spec=retention.FreezeSpec("fixture", groups, 77),
        )
        positions = root / "frozen.tsv"
        freeze_manifest = root / "freeze.json"
        retention.write_freeze(positions, freeze_manifest, payload, manifest)
        return positions, freeze_manifest, manifest

    def test_freeze_is_deterministic_and_rejects_an_overlapping_whole_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            candidate = root / "candidate.tsv"
            candidate.write_bytes(candidate_positions(5))
            training_receipt = root / "training-inputs.json"
            write_json(training_receipt, {"stage": "training-inputs-frozen"})
            spec = retention.FreezeSpec("fixture", 2, 17)

            candidate_rows = retention.load_position_rows(
                candidate, required_split=retention.TEACHER_SPLIT
            )
            first_group = min(
                {row.root_group_id for row in candidate_rows},
                key=lambda group: retention._selection_key(spec, group),
            )
            overlap = next(
                row for row in candidate_rows if row.root_group_id == first_group
            )
            excluded = root / "excluded.tsv"
            excluded.write_text(
                retention.POSITION_HEADER
                + "\n"
                + "\t".join(
                    (
                        "excluded-position",
                        "different-excluded-root",
                        "different-excluded-game",
                        overlap.source,
                        overlap.split,
                        str(overlap.winner),
                        str(overlap.mover),
                        overlap.prefix,
                    )
                )
                + "\n"
            )
            first = retention.freeze_candidate_groups(
                candidate_positions=candidate,
                training_input_receipt=training_receipt,
                campaign_id="retention-fixture-v1",
                spec=spec,
                excluded_position_tsvs=[excluded],
            )
            second = retention.freeze_candidate_groups(
                candidate_positions=candidate,
                training_input_receipt=training_receipt,
                campaign_id="retention-fixture-v1",
                spec=spec,
                excluded_position_tsvs=[excluded],
            )
            self.assertEqual(first, second)
            frozen_path = root / "frozen.tsv"
            frozen_path.write_bytes(first[0])
            frozen_rows = retention.load_position_rows(
                frozen_path, required_split=retention.TEACHER_SPLIT
            )
            self.assertNotIn(first_group, {row.root_group_id for row in frozen_rows})
            self.assertEqual(len(frozen_rows), 2 * retention.ROWS_PER_GROUP)
            self.assertEqual(
                first[1]["selection"]["rejection_counts"],
                {"excluded-canonical-fingerprint-overlap": 1},
            )

    def test_incomplete_candidate_group_cannot_be_partially_frozen(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            full = candidate_positions(1).decode().splitlines()
            candidate = root / "short.tsv"
            candidate.write_text("\n".join(full[:-1]) + "\n")
            receipt = root / "training-inputs.json"
            write_json(receipt, {"stage": "training-inputs-frozen"})
            with self.assertRaisesRegex(ValueError, "cannot fill its exact group quota"):
                retention.freeze_candidate_groups(
                    candidate_positions=candidate,
                    training_input_receipt=receipt,
                    campaign_id="retention-fixture-v1",
                    spec=retention.FreezeSpec("fixture", 1, 1),
                )

    def test_freeze_satisfies_exact_source_quotas(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            lines = candidate_positions(4).decode().splitlines()
            rewritten = [lines[0]]
            for line in lines[1:]:
                fields = line.split("\t")
                group_number = int(fields[1].rsplit(":", 1)[1])
                fields[3] = "source-a" if group_number < 3 else "source-b"
                rewritten.append("\t".join(fields))
            candidate = root / "candidate.tsv"
            candidate.write_text("\n".join(rewritten) + "\n")
            receipt = root / "training-inputs.json"
            write_json(receipt, {"stage": "training-inputs-frozen"})
            _payload, manifest = retention.freeze_candidate_groups(
                candidate_positions=candidate,
                training_input_receipt=receipt,
                campaign_id="retention-fixture-v1",
                spec=retention.FreezeSpec(
                    "fixture",
                    2,
                    5,
                    source_quotas=(("source-a", 1), ("source-b", 1)),
                ),
            )
            self.assertEqual(
                manifest["selection"]["selected_source_counts"],
                {"source-a": 1, "source-b": 1},
            )

    def test_pack_requires_exact_rank4_fixed_work_and_is_not_train_loadable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            positions, freeze_manifest, _ = self.freeze_fixture(root)
            rows = retention.load_position_rows(
                positions, required_split=retention.TEACHER_SPLIT
            )
            teacher_sha = "a" * 64
            labels = root / "labels.jsonl"
            labels.write_bytes(
                b"".join(
                    retention.canonical_json_bytes(
                        rank4_label(
                            row,
                            "retention-fixture-v1",
                            teacher_sha,
                            proof=index == 0,
                        )
                    )
                    for index, row in enumerate(rows)
                )
            )
            candidate_runtime = root / "candidate.runtime"
            candidate_runtime.write_bytes(b"selected-candidate")
            selection_receipt = root / "selection.json"
            write_json(
                selection_receipt,
                {
                    "stage": "selected-model",
                    "runtime_sha256": hashlib.sha256(
                        candidate_runtime.read_bytes()
                    ).hexdigest(),
                },
            )
            _npz, shard_manifest, manifest = retention.pack_holdout(
                frozen_positions=positions,
                freeze_manifest=freeze_manifest,
                labels=labels,
                selection_receipt=selection_receipt,
                teacher_source_sha256=teacher_sha,
                output_directory=root / "packed",
            )
            shard = retention.load_holdout_shard(shard_manifest)
            self.assertEqual(len(shard), len(rows) * 2)
            self.assertEqual(
                manifest["termination_counts"],
                {"fixed-work-cap": len(rows) - 1, "root-solved": 1},
            )
            self.assertFalse(manifest["training_eligible"])
            with self.assertRaises(ValueError):
                training.load_csr_shard(shard_manifest)

            bad_labels = root / "bad-labels.jsonl"
            bad = [
                rank4_label(row, "retention-fixture-v1", teacher_sha)
                for row in rows
            ]
            bad[0]["nodes"] = retention.RANK4_FIXED_NODES - 1
            bad[0]["search_stats"]["nodes"] = retention.RANK4_FIXED_NODES - 1
            bad_labels.write_bytes(
                b"".join(retention.canonical_json_bytes(row) for row in bad)
            )
            with self.assertRaisesRegex(ValueError, "exact node cap"):
                retention.pack_holdout(
                    frozen_positions=positions,
                    freeze_manifest=freeze_manifest,
                    labels=bad_labels,
                    selection_receipt=selection_receipt,
                    teacher_source_sha256=teacher_sha,
                    output_directory=root / "bad-packed",
                )

    def test_point_and_root_cluster_noninferiority_are_conjunctive(self):
        targets = np.asarray(([0.75, -0.75] * 40), dtype=np.float32)
        weights = np.ones(len(targets), dtype=np.float32)
        groups = [f"root:{index // 20}" for index in range(len(targets))]
        actor = targets * np.float32(0.8)
        passing = retention.noninferiority_evidence(
            actor_predictions=actor,
            candidate_predictions=actor.copy(),
            targets=targets,
            weights=weights,
            root_group_ids=groups,
        )
        self.assertTrue(passing["gates"]["pass"])
        self.assertTrue(all(passing["gates"].values()))

        failing = retention.noninferiority_evidence(
            actor_predictions=actor,
            candidate_predictions=-actor,
            targets=targets,
            weights=weights,
            root_group_ids=groups,
        )
        self.assertFalse(failing["gates"]["pass"])
        self.assertFalse(failing["gates"]["point_sign"])
        self.assertFalse(failing["gates"]["cluster_sign"])

    def test_evaluation_binds_the_same_selection_receipt_used_for_reveal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            positions, freeze_manifest, _ = self.freeze_fixture(root)
            rows = retention.load_position_rows(
                positions, required_split=retention.TEACHER_SPLIT
            )
            teacher_sha = "b" * 64
            labels = root / "labels.jsonl"
            labels.write_bytes(
                b"".join(
                    retention.canonical_json_bytes(
                        rank4_label(row, "retention-fixture-v1", teacher_sha)
                    )
                    for row in rows
                )
            )
            actor_runtime = root / "actor.runtime"
            candidate_runtime = root / "candidate.runtime"
            actor_runtime.write_bytes(b"actor")
            candidate_runtime.write_bytes(b"candidate")
            candidate_sha = hashlib.sha256(candidate_runtime.read_bytes()).hexdigest()
            selection_receipt = root / "selection.json"
            write_json(
                selection_receipt,
                {"stage": "selected-model", "runtime_sha256": candidate_sha},
            )
            _npz, shard_manifest, _ = retention.pack_holdout(
                frozen_positions=positions,
                freeze_manifest=freeze_manifest,
                labels=labels,
                selection_receipt=selection_receipt,
                teacher_source_sha256=teacher_sha,
                output_directory=root / "packed",
            )
            shard = retention.load_holdout_shard(shard_manifest)
            predictions = shard.targets * np.float32(0.8)
            with (
                mock.patch.object(
                    retention.training,
                    "load_runtime",
                    side_effect=[({"runtime": "actor"}, {"kind": "actor"}),
                                 ({"runtime": "candidate"}, {"kind": "candidate"})],
                ),
                mock.patch.object(
                    retention,
                    "_predictions",
                    side_effect=[predictions, predictions.copy()],
                ),
            ):
                report = retention.evaluate_holdout(
                    shard_manifest=shard_manifest,
                    actor_runtime=actor_runtime,
                    candidate_runtime=candidate_runtime,
                    selection_receipt=selection_receipt,
                    output=root / "evidence.json",
                )
            self.assertTrue(report["pass"])
            self.assertEqual(
                report["inputs"]["selection_receipt"],
                shard.manifest["reveal"]["selection_receipt"],
            )


if __name__ == "__main__":
    unittest.main()
