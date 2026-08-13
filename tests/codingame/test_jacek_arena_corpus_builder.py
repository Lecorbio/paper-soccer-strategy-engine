import importlib.util
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
BOT = ROOT / "submissions" / "codingame" / "bots" / "jacek_arena_bfm"
sys.path.insert(0, str(BOT))
SPEC = importlib.util.spec_from_file_location(
    "jacek_arena_build_arena_corpus", BOT / "build_arena_corpus.py"
)
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
assert SPEC.loader is not None
SPEC.loader.exec_module(builder)


def sparse(offset=0):
    return list(range(offset, offset + 105))


def auditor_input(game_id=1001, candidate=1, winner=0, turns=("0",)):
    game = builder.AuditorGame(game_id, candidate, winner, tuple(turns))
    return builder.AuditorInput(
        path=pathlib.Path("auditor.tsv"),
        sha256="a" * 64,
        byte_count=1,
        metadata={},
        games={game_id: game},
    )


def meta(*, values=1, pairs=0):
    return {
        "schema": builder.REANALYSIS_SCHEMA,
        "kind": "meta",
        "games": 1,
        "value_rows": values,
        "pairwise_rows": pairs,
        "decisions_analyzed": 1 if pairs else 0,
        "alternatives_analyzed": pairs,
        "unstable_ordering_rejections": 0,
        "proved_losing_vs_winning_rejections": 0,
        "cap_rejections": 0,
        "exact_pairs": 0,
        "work_checkpoints": [30000, 100000],
        "maximum_analyzed_decisions_per_game": 2,
        "maximum_pairs_per_decision": 4,
        "maximum_pairs_per_game": 16,
        "search_width": 8,
    }


def value_record(game_id=1001, turn=0, candidate=1, winner=0):
    return {
        "schema": builder.REANALYSIS_SCHEMA,
        "kind": "value",
        "game_id": game_id,
        "turn_index": turn,
        "actor": turn % 2,
        "candidate_player": candidate,
        "winner": winner,
        "position_id": f"fnv1a64:{turn + 1:016x}",
        "target": 1 if turn % 2 == winner else -1,
        "active_features": sparse(),
    }


def pair_record(*, actor=0, preferred_30=0.4, inferior_30=0.2,
                preferred_100=0.5, inferior_100=0.3):
    return {
        "schema": builder.REANALYSIS_SCHEMA,
        "kind": "pairwise",
        "game_id": 1001,
        "turn_index": 0,
        "actor": actor,
        "candidate_player": 1,
        "winner": 0,
        "decision_id": "fnv1a64:0000000000000001",
        "pair_index": 0,
        "observed_action": "0",
        "inferior_action": "1",
        "exact": False,
        "preferred_terminal": False,
        "inferior_terminal": False,
        "preferred_value_30000": preferred_30,
        "inferior_value_30000": inferior_30,
        "preferred_value_100000": preferred_100,
        "inferior_value_100000": inferior_100,
        "preferred_work_30000": 30000,
        "inferior_work_30000": 30000,
        "preferred_work_100000": 100000,
        "inferior_work_100000": 100000,
        "preferred_depth_30000": 3,
        "inferior_depth_30000": 3,
        "preferred_depth_100000": 4,
        "inferior_depth_100000": 4,
        "preferred_active_features": sparse(),
        "inferior_active_features": sparse(1),
    }


def reanalysis_payload(rows):
    return b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
        for row in rows
    )


class ArenaCorpusBuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = shutil.which("c++")
        cls.build_directory = tempfile.TemporaryDirectory()
        cls.reanalyzer = pathlib.Path(cls.build_directory.name) / "reanalyzer"
        if cls.compiler:
            completed = subprocess.run(
                [
                    cls.compiler,
                    "-std=c++20",
                    "-O1",
                    str(BOT / "arena_corpus_reanalyzer.cpp"),
                    str(BOT / "engine.cpp"),
                    "-o",
                    str(cls.reanalyzer),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            if completed.returncode:
                raise AssertionError(completed.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.build_directory.cleanup()

    @unittest.skipUnless(shutil.which("c++"), "requires a C++20 compiler")
    def test_clean_room_replayer_accepts_terminal_transcript(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            tsv = root / "auditor.tsv"
            tsv.write_text(
                "game_id\tcandidate_player\twinner\tturns\n"
                "1001\t1\t0\t0/0/3/0/61/0/07\n",
                encoding="ascii",
            )
            ranking = root / "ranking.txt"
            ranking.write_text("", encoding="ascii")
            completed = subprocess.run(
                [
                    str(self.reanalyzer),
                    "--input", str(tsv),
                    "--ranking-game-ids", str(ranking),
                ],
                capture_output=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr.decode())
            audit = auditor_input(
                game_id=1001,
                candidate=1,
                winner=0,
                turns=("0", "0", "3", "0", "61", "0", "07"),
            )
            parsed = builder.parse_reanalysis(completed.stdout, audit)
            self.assertEqual(len(parsed.value_rows), 7)
            self.assertFalse(parsed.pairwise_rows)
            self.assertEqual(parsed.value_rows[0]["target"], 1)
            self.assertEqual(parsed.value_rows[1]["target"], -1)

    def test_reanalysis_rejects_collection_bot_action_as_pair_target(self):
        audit = auditor_input(candidate=0)
        bad_pair = pair_record(actor=0)
        bad_pair["candidate_player"] = 0
        with self.assertRaisesRegex(builder.ArenaCorpusError, "observed opponent"):
            builder.parse_reanalysis(
                reanalysis_payload([meta(values=1, pairs=1), value_record(candidate=0), bad_pair]),
                audit,
            )

    def test_reanalysis_requires_margin_at_both_work_gates(self):
        unstable = pair_record(preferred_30=0.29, inferior_30=0.2)
        with self.assertRaisesRegex(builder.ArenaCorpusError, "not stable"):
            builder.parse_reanalysis(
                reanalysis_payload([meta(values=1, pairs=1), value_record(), unstable]),
                auditor_input(),
            )

    def test_rows_pass_fresh_provenance_and_pairwise_validator(self):
        audit = auditor_input()
        parsed = builder.parse_reanalysis(
            reanalysis_payload([meta(values=1, pairs=1), value_record(), pair_record()]),
            audit,
        )
        derivation_sha = "d" * 64
        record_sha = "e" * 64
        raw_sha = "f" * 64
        normalized_sha = "1" * 64
        source_sha = "2" * 64
        derivation = {
            "campaign": {"namespace": builder.NAMESPACE, "t0_utc": "2026-08-13T10:12:52Z"},
            "timing": {"collection_completed_at_utc": "2026-08-13T11:00:00Z"},
            "source": {
                "agent_id": "12",
                "submission_id": "34",
                "repository_commit": "3" * 40,
                "sha256": source_sha,
            },
            "window": {"window_id": "collection-001", "role": "training"},
            "games": [{
                "game_id": 1001,
                "record_sha256": record_sha,
                "raw_sha256": raw_sha,
                "normalized_sha256": normalized_sha,
                "opponent_frozen_rank": 12,
                "ranking_candidate_weight": 1.0,
                "uses": [
                    "raw-terminal-value-candidate",
                    "opponent-action-ranking-reanalysis-candidate",
                ],
            }],
        }
        rows = builder.build_rows(
            parsed,
            audit,
            derivation,
            derivation_sha256=derivation_sha,
            generated_at_utc="2026-08-13T11:01:00Z",
        )
        self.assertEqual([row["source_kind"] for row in rows], [
            "arena_terminal", "arena_opponent_ranking"
        ])
        binding = builder.ArenaGameBinding(
            game_id="1001",
            derivation_sha256=derivation_sha,
            record_sha256=record_sha,
            raw_sha256=raw_sha,
            normalized_sha256=normalized_sha,
            window_id="collection-001",
            window_role="training",
            source_sha256=source_sha,
            agent_id="12",
            submission_id="34",
            opponent_frozen_rank=12,
            ranking_candidate_weight=1.0,
            uses=frozenset({
                "raw-terminal-value-candidate",
                "opponent-action-ranking-reanalysis-candidate",
            }),
        )
        contract = builder.CampaignContract(
            campaign_id="jacek_arena_bfm@2026-08-13T10:12:52Z",
            t0_utc=builder.parse_utc("2026-08-13T10:12:52Z", "t0"),
            window_roles={"collection-001": "training"},
            arena_freeze_cutoff_utc=builder.parse_utc(
                "2026-08-13T12:47:52Z", "freeze"
            ),
        )
        validator = builder.FreshCorpusValidator(
            contract,
            excluded_game_ids={999},
            approved_producer_source_sha256={source_sha},
            arena_game_bindings={"1001": binding},
        )
        validated, summary = validator.validate_rows(rows)
        self.assertEqual(len(validated), 2)
        self.assertEqual(summary.counts_by_kind, {"pairwise": 1, "value": 1})
        tampered = [dict(row) for row in rows]
        tampered[1]["raw_sha256"] = "9" * 64
        with self.assertRaisesRegex(builder.CorpusValidationError, "binding"):
            validator.validate_rows(tampered)


if __name__ == "__main__":
    unittest.main()
