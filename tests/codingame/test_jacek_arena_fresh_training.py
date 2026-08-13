import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest

try:
    import numpy as np
except ImportError:  # The repository's optional research environment owns NumPy.
    np = None


ROOT = pathlib.Path(__file__).resolve().parents[2]
BOT = ROOT / "submissions" / "codingame" / "bots" / "jacek_arena_bfm"

if np is not None:
    sys.path.insert(0, str(BOT))
    spec = importlib.util.spec_from_file_location("jacek_arena_train_fresh", BOT / "train_fresh.py")
    trainer = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = trainer
    spec.loader.exec_module(trainer)
else:
    trainer = None


@unittest.skipIf(np is None, "fresh-training tests require the campaign NumPy environment")
class FreshTrainingTests(unittest.TestCase):
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

    def test_empty_pair_split_is_strict_json(self):
        network = trainer.Network.random(32, 3)
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
