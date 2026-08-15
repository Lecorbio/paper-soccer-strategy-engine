import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tools" / "train_rank4_jacek_hybrid_ranker.py"
SPEC = importlib.util.spec_from_file_location("hybrid_ranker", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
ranker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ranker
SPEC.loader.exec_module(ranker)


def features(signal: float) -> list[float]:
    values = [0.0] * ranker.FEATURE_COUNT
    values[10] = -signal  # Deliberately make the incumbent anchor backwards.
    values[13] = signal
    return values


def candidate(
    root: str,
    split: str,
    variant: str,
    action: str,
    signal: float,
    depth: int = 4,
):
    rotated = variant in ("rotate", "rotate_mirror")
    return {
        "schema": ranker.CORPUS_SCHEMA,
        "root_id": root,
        "split": split,
        "variant": variant,
        "root_depth": depth,
        "candidate": int(action),
        "action": action,
        "features": features(signal),
        "anchor_score": 0,
        "root_mover_sign": -1 if rotated else 1,
        "successor_mover_sign": 1 if rotated else -1,
        "low_score": int(signal * 10_000),
        "high_score": int(signal * 12_000),
        "low_depth": 3,
        "high_depth": 4,
        "low_nodes_used": 30_000,
        "high_nodes_used": 100_000,
    }


class HybridRankerTest(unittest.TestCase):
    def write_corpus(self, path: pathlib.Path, leaking: bool = False) -> None:
        rows = [
            {
                "schema": ranker.META_SCHEMA,
                "seed": 7,
                "roots": 12,
                "max_actions": 2,
                "low_nodes": 30_000,
                "high_nodes": 100_000,
                "mirror": True,
                "color_swap": True,
                "variants": list(ranker.VARIANTS),
                "rules": "8x10;own-goals-allowed;mover-loses",
                "teacher": "frozen-rank-4-proof-off",
            }
        ]
        index = 0
        for split in ranker.SPLITS:
            for depth in (4, 8, 12, 20):
                root = f"fresh-r{index:04d}-d{depth:02d}-{index:016x}"
                for variant in ranker.VARIANTS:
                    first, second = (
                        ("4", "0") if variant.startswith("rotate") else ("0", "4")
                    )
                    rows.append(candidate(root, split, variant, first, -1.0, depth))
                    rows.append(candidate(root, split, variant, second, 1.0, depth))
                if leaking and index == 0:
                    rows.append(
                        candidate(root, "validation", "base", "2", 0.0, depth)
                    )
                index += 1
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="ascii",
        )

    def test_whole_root_split_leakage_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "corpus.jsonl"
            self.write_corpus(path, leaking=True)
            with self.assertRaisesRegex(ValueError, "whole-root split leakage"):
                ranker.load_corpus(path)

    def test_stable_pair_filter_and_random_init_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "corpus.jsonl"
            self.write_corpus(path)
            _, candidates = ranker.load_corpus(path)
        groups = ranker.group_candidates(candidates)
        pairs = ranker.all_pairs(groups, minimum_margin=1000, minimum_depth=2)
        train = [pair for pair in pairs if pair.split == "train"]
        self.assertEqual(len(pairs), 48)
        weights = ranker.train_logistic(
            train, seed=1701, ridge=0.001, epochs=20, learning_rate=0.03
        )
        metric = ranker.pair_accuracy(train, weights)
        self.assertEqual(metric["root_balanced_accuracy"], 1.0)
        self.assertGreater(weights[13], 0.0)

    def test_int8_projection_preserves_order_and_is_tiny(self):
        weights = [0.0] * ranker.FEATURE_COUNT
        weights[10] = 0.5
        weights[13] = 2.0
        integers, scale, restored = ranker.quantize(weights)
        self.assertEqual(len(integers), ranker.FEATURE_COUNT)
        self.assertGreater(scale, 0.0)
        self.assertGreater(restored[13], restored[10])
        header = ranker.render_header(integers)
        self.assertLess(len(header.encode("ascii")), 512)
        self.assertTrue(header.isascii())


if __name__ == "__main__":
    unittest.main()
