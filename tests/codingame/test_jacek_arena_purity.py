import importlib.util
import json
import pathlib
import shutil
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
BOT = ROOT / "submissions" / "codingame" / "bots" / "jacek_arena_bfm"
SPEC = importlib.util.spec_from_file_location(
    "jacek_arena_check_purity", BOT / "check_purity.py"
)
purity = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(purity)


class PurityGateTests(unittest.TestCase):
    def copied_runtime(self, directory: pathlib.Path) -> pathlib.Path:
        bot = directory / "jacek_arena_bfm"
        bot.mkdir()
        for name in purity.RUNTIME_FILES:
            shutil.copyfile(BOT / name, bot / name)
        return bot

    def test_current_runtime_is_pure(self):
        report = purity.validate_runtime(BOT)
        self.assertTrue(report["valid"])
        self.assertFalse(report["protected_content_read"])
        self.assertEqual(report["model"]["shape"], [1156, 32, 32, 1])

    def test_forbidden_bot_reference_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            bot = self.copied_runtime(pathlib.Path(directory))
            engine = bot / "engine.cpp"
            engine.write_text(
                engine.read_text(encoding="ascii") +
                "\n// submissions/codingame/bots/rank_4/model.hpp\n",
                encoding="ascii",
            )
            with self.assertRaisesRegex(purity.PurityError, "forbidden runtime reference"):
                purity.validate_runtime(bot)

    def test_external_corpus_reference_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            bot = self.copied_runtime(pathlib.Path(directory))
            bot_cpp = bot / "bot.cpp"
            bot_cpp.write_text(
                bot_cpp.read_text(encoding="ascii") +
                '\nconstexpr char forbidden[] = "results/foreign/corpus/training.jsonl";\n',
                encoding="ascii",
            )
            with self.assertRaisesRegex(purity.PurityError, "forbidden runtime reference"):
                purity.validate_runtime(bot)

    def test_source_manifest_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            bot = self.copied_runtime(pathlib.Path(directory))
            (bot / "sources.txt").write_text(
                (bot / "sources.txt").read_text(encoding="ascii") +
                "submissions/codingame/bots/foreign/model.hpp\n",
                encoding="ascii",
            )
            with self.assertRaises(purity.PurityError):
                purity.validate_runtime(bot)

    def test_corrupt_packed_count_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            bot = self.copied_runtime(pathlib.Path(directory))
            model = bot / "model.hpp"
            model.write_text(
                model.read_text(encoding="ascii").replace(
                    "kW1Count = 36992", "kW1Count = 1", 1
                ),
                encoding="ascii",
            )
            with self.assertRaisesRegex(purity.PurityError, "declared count"):
                purity.validate_runtime(bot)

    def test_generated_submission_size_is_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            bot = self.copied_runtime(pathlib.Path(directory))
            submission = bot / "submission.cpp"
            submission.write_text(
                submission.read_text(encoding="ascii") + " " * 12000,
                encoding="ascii",
            )
            with self.assertRaisesRegex(purity.PurityError, "99,999"):
                purity.validate_runtime(bot)

    def test_cli_report_is_strict_json_shape(self):
        report = purity.validate_runtime(BOT)
        json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
