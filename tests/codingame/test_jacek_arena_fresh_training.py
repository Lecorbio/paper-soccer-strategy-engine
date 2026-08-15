import copy
import datetime as dt
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

try:
    import numpy as np
except ImportError:  # The repository's optional research environment owns NumPy.
    np = None


ROOT = pathlib.Path(__file__).resolve().parents[2]
BOT_DIR = ROOT / "submissions" / "codingame" / "bots" / "jacek_arena_bfm"
sys.path.insert(0, str(BOT_DIR))

import fresh_corpus as corpus  # noqa: E402

if np is not None:
    import train_fresh as trainer  # noqa: E402
else:
    trainer = None


PRODUCER = "a" * 64
DERIVATION = "b" * 64
RECORD = "c" * 64
RAW = "d" * 64
NORMALIZED = "e" * 64
SOURCE = "f" * 64
EVIDENCE = "1" * 64


def contract():
    return corpus.CampaignContract(
        campaign_id="jacek_arena_bfm@test",
        t0_utc=dt.datetime(2026, 8, 13, 10, 12, 52, tzinfo=dt.timezone.utc),
        window_roles={"collection-001": "training", "collection-004": "arena-validation"},
        arena_freeze_cutoff_utc=dt.datetime(2026, 8, 13, 12, 47, 52, tzinfo=dt.timezone.utc),
    )


def binding():
    return corpus.ArenaGameBinding(
        game_id="1001",
        derivation_sha256=DERIVATION,
        record_sha256=RECORD,
        raw_sha256=RAW,
        normalized_sha256=NORMALIZED,
        window_id="collection-001",
        window_role="training",
        source_sha256=SOURCE,
        agent_id="101",
        submission_id="201",
        opponent_frozen_rank=10,
        ranking_candidate_weight=1.0,
        uses=frozenset(
            {
                "raw-terminal-value-candidate",
                "opponent-action-ranking-reanalysis-candidate",
            }
        ),
    )


def validator(*, arena=True):
    return corpus.FreshCorpusValidator(
        contract(),
        excluded_game_ids={"42"},
        approved_producer_source_sha256={PRODUCER},
        arena_game_bindings={"1001": binding()} if arena else {},
    )


def base_row():
    return {
        "schema": corpus.CORPUS_SCHEMA,
        "namespace": corpus.NAMESPACE,
        "campaign_id": "jacek_arena_bfm@test",
        "sample_id": "scratch-1:0",
        "game_id": "scratch-1",
        "generated_at_utc": "2026-08-13T10:20:00Z",
        "evidence_at_utc": "2026-08-13T10:19:00Z",
        "producer_source_sha256": PRODUCER,
        "evidence_sha256": EVIDENCE,
        "representation": "mover_relative_316_edges_plus_105x8_distance_v1",
        "weight": 1.0,
        "kind": "value",
        "source_kind": "scratch_selfplay",
        "target": 1.0,
        "label_method": "terminal_outcome",
        "opening_depth": 0,
        "initialization": "random",
        "checkpoint_inputs": [],
        "window_id": None,
        "submission_id": None,
        "features": [0] * corpus.FEATURE_COUNT,
    }


def arena_value_row():
    row = base_row()
    row.update(
        {
            "sample_id": "1001:p0",
            "game_id": "1001",
            "arena_game_id": 1001,
            "source_kind": "arena_terminal",
            "position_id": "p0",
            "weight": 0.5,
            "theoretical_value_claim": False,
            "operational_clean": True,
            "complete_transcript": True,
            "terminal_unambiguous": True,
            "timeout": False,
            "illegal_action": False,
            "malformed_transcript": False,
            "raw_sha256": RAW,
            "normalized_sha256": NORMALIZED,
            "submitted_source_sha256": SOURCE,
            "arena_derivation_sha256": DERIVATION,
            "arena_record_sha256": RECORD,
            "agent_id": "101",
            "submission_id": "201",
            "window_id": "collection-001",
            "window_role": "training",
        }
    )
    row.pop("opening_depth")
    row.pop("initialization")
    row.pop("checkpoint_inputs")
    return row


def pair_row():
    row = arena_value_row()
    row.update(
        {
            "sample_id": "1001:d0:p0",
            "kind": "pairwise",
            "source_kind": corpus.PAIR_SOURCE,
            "actor_origin": "opponent",
            "complete_action_legal": True,
            "counterfactual_replay_verified": True,
            "observed_complete_action": "01",
            "inferior_complete_action": "2",
            "opponent_snapshot_rank": 10,
            "exact": False,
            "preferred_value_30000": 0.5,
            "inferior_value_30000": 0.3,
            "preferred_value_100000": 0.45,
            "inferior_value_100000": 0.25,
            "counterfactual_verdict": "observed-not-proved-losing-vs-winning-alternative",
            "decision_id": "d0",
            "pair_index": 0,
            "preferred_features": [1] + [0] * (corpus.FEATURE_COUNT - 1),
            "inferior_features": [0] * corpus.FEATURE_COUNT,
            "weight": 1.0,
        }
    )
    for key in ("features", "target", "label_method", "position_id", "theoretical_value_claim"):
        row.pop(key)
    return row


class FreshTrainingAuditTest(unittest.TestCase):
    def test_validator_requires_sealed_exclusions_and_producer_identity(self):
        with self.assertRaisesRegex(corpus.CorpusValidationError, "exclusion registry"):
            corpus.FreshCorpusValidator(contract(), approved_producer_source_sha256={PRODUCER})
        with self.assertRaisesRegex(corpus.CorpusValidationError, "producer source"):
            corpus.FreshCorpusValidator(contract(), excluded_game_ids={"42"})

    def test_binary_features_are_not_silently_truncated_to_uint8(self):
        row = base_row()
        row["features"][7] = 1.5
        with self.assertRaisesRegex(corpus.CorpusValidationError, "integer 0 or 1"):
            validator(arena=False).validate_row(row)
        row["features"][7] = -1
        with self.assertRaisesRegex(corpus.CorpusValidationError, "integer 0 or 1"):
            validator(arena=False).validate_row(row)

    def test_arena_rows_must_match_an_eligible_immutable_derivation(self):
        item = validator().validate_row(arena_value_row())
        self.assertEqual(item.split_game_id, "1001")
        forged = arena_value_row()
        forged["raw_sha256"] = "9" * 64
        with self.assertRaisesRegex(corpus.CorpusValidationError, "immutable derivation"):
            validator().validate_row(forged)
        with self.assertRaisesRegex(corpus.CorpusValidationError, "absent from approved"):
            validator(arena=False).validate_row(arena_value_row())

    def test_focus_failure_rejects_entire_arena_derivation_from_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            campaign_root = pathlib.Path(temporary)
            derivation_path = campaign_root / "collection-001.json"
            derivation_path.write_bytes(b"validated-upstream-placeholder\n")
            derivation = {
                "window_plan": {"sha256": corpus.APPROVED_WINDOW_PLAN_SHA256},
                "exclusion_registry": {"sha256": corpus.APPROVED_EXCLUSION_SHA256},
                "window": {"window_id": "collection-001", "role": "training"},
                "summary": {
                    "eligible_games": 74,
                    "focus_operational_failures": 3,
                },
            }
            with mock.patch.object(
                corpus, "validate_arena_derivation", return_value=derivation
            ):
                with self.assertRaisesRegex(
                    corpus.CorpusValidationError,
                    "entire submission window.*forbidden for training",
                ):
                    corpus.load_arena_game_bindings(
                        [derivation_path],
                        campaign_root=campaign_root,
                        repository=ROOT,
                    )

    def test_pairwise_target_is_opponent_bound_stable_and_derivation_weighted(self):
        item = validator().validate_row(pair_row())
        self.assertEqual(item.kind, "pairwise")
        forged = pair_row()
        forged["actor_origin"] = "collection_bot"
        with self.assertRaisesRegex(corpus.CorpusValidationError, "only the opponent"):
            validator().validate_row(forged)
        forged = pair_row()
        forged["counterfactual_verdict"] = "observed_loses_alternative_wins"
        with self.assertRaisesRegex(corpus.CorpusValidationError, "explicitly attest"):
            validator().validate_row(forged)

    def test_exact_arena_value_replaces_raw_trajectory_label(self):
        raw = arena_value_row()
        exact = copy.deepcopy(raw)
        exact.update(
            {
                "sample_id": "1001:p0:exact",
                "source_kind": "arena_reanalysis",
                "label_method": "exact",
                "weight": 1.0,
            }
        )
        exact.pop("theoretical_value_claim")
        with self.assertRaisesRegex(corpus.CorpusValidationError, "must replace"):
            validator().validate_rows([raw, exact])

    @unittest.skipIf(np is None, "fresh-training tests require the campaign NumPy environment")
    def test_combined_exposure_handles_arena_only_pair_objective(self):
        zeros = np.zeros((3, corpus.FEATURE_COUNT), dtype=np.uint8)
        pair = np.zeros((1, corpus.FEATURE_COUNT), dtype=np.uint8)
        data = trainer.Data(
            value_x=zeros,
            value_y=np.asarray([1.0, -1.0, 1.0], dtype=np.float32),
            value_w=np.ones(3, dtype=np.float32),
            value_split=np.asarray([2, 2, 2], dtype=np.uint8),
            value_arena=np.asarray([False, False, True]),
            pair_preferred=pair.copy(),
            pair_inferior=pair.copy(),
            pair_w=np.ones(1, dtype=np.float32),
            pair_split=np.asarray([2], dtype=np.uint8),
            pair_arena=np.asarray([True]),
            games=4,
            scratch_games=2,
            scratch_games_by_opening_depth={0: 1, 4: 1},
            source_counts={},
            corpus_inputs=[],
            window_plan_sha256="2" * 64,
            exclusion_registry_sha256="3" * 64,
            producer_source_sha256=[PRODUCER],
            arena_derivation_sha256=[DERIVATION],
        )
        value_weights, pair_weights, realized = trainer.effective_training_weights(data, 0.40)
        arena_total = float(value_weights[data.value_arena].sum() + pair_weights.sum())
        all_total = float(value_weights.sum() + pair_weights.sum())
        self.assertAlmostEqual(arena_total / all_total, 0.40, places=6)
        self.assertAlmostEqual(realized, 0.40, places=6)
        _, history = trainer.train(data, 32, 7, 1, 2, 0.001, 0.40)
        self.assertAlmostEqual(history[0]["effective_arena_exposure"], 0.40, places=6)

    @unittest.skipIf(np is None, "fresh-training tests require the campaign NumPy environment")
    def test_pairwise_step_pushes_preferred_successor_below_inferior(self):
        network = trainer.Network.random(32, 91)
        preferred = np.zeros((1, trainer.FEATURE_COUNT), dtype=np.uint8)
        inferior = np.zeros_like(preferred)
        preferred[0, 17] = 1
        inferior[0, 29] = 1
        weight = np.ones(1, dtype=np.float32)
        before, gradients = trainer.pair_batch(network, preferred, inferior, weight)
        for parameter, gradient in zip(
            (network.w1, network.w2, network.w3), gradients, strict=True
        ):
            parameter -= 0.05 * gradient
        after, _ = trainer.pair_batch(network, preferred, inferior, weight)
        self.assertLess(after, before)
        preferred_value = network.forward(preferred.astype(np.float32))[2][0]
        inferior_value = network.forward(inferior.astype(np.float32))[2][0]
        self.assertLess(preferred_value, inferior_value)

    @unittest.skipIf(np is None, "fresh-training tests require the campaign NumPy environment")
    def test_empty_pair_split_is_strict_json(self):
        network = trainer.Network.random(32, 3)
        # Deliberately use the original compact Data constructor.  Provenance
        # fields are mandatory in load_data(), not for isolated metrics tests.
        data = trainer.Data(
            value_x=np.zeros((1, trainer.FEATURE_COUNT), dtype=np.uint8),
            value_y=np.ones(1, dtype=np.float32),
            value_w=np.ones(1, dtype=np.float32),
            value_split=np.zeros(1, dtype=np.uint8),
            value_arena=np.zeros(1, dtype=bool),
            pair_preferred=np.empty((0, trainer.FEATURE_COUNT), dtype=np.uint8),
            pair_inferior=np.empty((0, trainer.FEATURE_COUNT), dtype=np.uint8),
            pair_w=np.empty(0, dtype=np.float32),
            pair_split=np.empty(0, dtype=np.uint8),
            pair_arena=np.empty(0, dtype=bool),
            games=1,
            source_counts={"scratch_selfplay": 1},
            corpus_inputs=[],
        )
        measured = trainer.metrics(network, data, 0)
        self.assertIsNone(measured["pair_accuracy"])
        json.dumps(measured, allow_nan=False)

    @unittest.skipIf(np is None, "fresh-training tests require the campaign NumPy environment")
    def test_packed_header_has_runtime_contract(self):
        network = trainer.Network.random(32, 7)
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "model.hpp"
            trainer.emit_header(output, network, 7, "unit-test-model")
            content = output.read_text(encoding="ascii")
        for identifier in (
            "kInputSize", "kHidden1Size", "kHidden2Size",
            "kW1Count", "kW2Count", "kW3Count",
            "kW1Scale", "kW2Scale", "kW3Scale",
            "kW1Packed", "kW2Packed", "kW3Packed",
            "kBootstrapSeed", "kIdentity",
        ):
            self.assertIn(identifier, content)


if __name__ == "__main__":
    unittest.main()
