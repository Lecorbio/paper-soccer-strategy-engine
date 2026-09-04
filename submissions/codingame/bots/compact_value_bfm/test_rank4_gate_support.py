import json
import pathlib
import sys
import tempfile
import unittest


HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rank4_gate_support as support  # noqa: E402


TRANSCRIPT = "4/7/5/2/23/1/7/61/2/7"


def engine(decisions=0):
    return {
        "decisions": decisions,
        "deadline_stops": 0,
        "soft_overruns": 0,
        "headroom_failures": 0,
        "hard_timeouts": 0,
        "work": 0,
        "generated_children": 0,
        "evaluated_children": 0,
        "maximum_first_ms": 0.0,
        "maximum_later_ms": 0.0,
        "times_ms": [0.0] * decisions,
    }


class Rank4GateSupportTests(unittest.TestCase):
    def test_bank_requires_canonical_rows_unique_ids_and_twelve_plies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            bank = root / "bank.tsv"
            bank.write_text(
                "# papersoccer.compact-value-bfm-opening-bank.v1\n"
                "opening_id\ttranscript\n"
                f"first\t{TRANSCRIPT}\nsecond\t{TRANSCRIPT}/0\n"
            )
            value = support.validate_bank(bank)
            self.assertEqual([row["physical_plies"] for row in value["openings"]],
                             [12, 13])
            bank.write_text("opening_id\ttranscript\nshort\t0/1\n")
            with self.assertRaisesRegex(ValueError, "fewer than 12"):
                support.validate_bank(bank)
            bank.write_text(
                "opening_id\ttranscript\n"
                f"same\t{TRANSCRIPT}\nsame\t{TRANSCRIPT}/0\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                support.validate_bank(bank)

    def document(self):
        games = []
        for color in (0, 1):
            games.append({
                "opening_id": "first",
                "pair_index": 3,
                "candidate_player": color,
                "winner": color,
                "turns": 10,
                "failure": None,
                "candidate": engine(1),
                "rank4": engine(1),
            })
        return {
            "schema": support.RESULT_SCHEMA,
            "bindings": {
                "candidate_source_sha256": "1" * 64,
                "candidate_source_bytes": 1,
                "candidate_runtime_body_sha256": "2" * 64,
                "candidate_payload_sha256": "3" * 64,
                "rank4_source_sha256": support.RANK4_SHA256,
                "rank4_source_bytes": 1,
                "opponent_sha256": support.RANK4_SHA256,
                "bank_sha256": "4" * 64,
                "bank_bytes": 1,
            },
            "config": {
                "mode": "fixed-work",
                "pair_offset": 3,
                "pair_count": 1,
                "candidate_c": .95,
                "candidate_fpu": .5,
                "candidate_lambda": 1.0,
                "candidate_actions": 250,
                "candidate_root_partial_paths": 4000,
                "candidate_nonroot_partial_paths": 512,
                "candidate_nodes": 80000,
                "candidate_expansions": 2000000,
                "candidate_shuffle_seed": 1,
                "candidate_clocks_ms": [800, 155],
                "rank4_nodes": 30000,
                "rank4_clocks_ms": [800, 165],
                "max_turns": 320,
                "minimum_candidate_wins": -1,
                "minimum_wins_per_color": -1,
            },
            "games": games,
            "result": {
                "games": 2,
                "candidate_wins": 2,
                "rank4_wins": 0,
                "candidate_wins_player0": 1,
                "candidate_wins_player1": 1,
                "failures": 0,
                "unfinished": 0,
                "failure_categories": {},
                "candidate": engine(2),
                "rank4": engine(2),
                "passed": True,
            },
        }

    def test_result_recomputes_pair_colors_failures_and_bindings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "result.json"
            document = self.document()
            path.write_text(json.dumps(document))
            support.validate_result(
                path,
                expected_bank_sha256="4" * 64,
                expected_candidate_sha256="1" * 64,
            )
            document["games"][1]["candidate_player"] = 0
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "pair/color"):
                support.validate_result(path)
            document = self.document()
            document["bindings"]["opponent_sha256"] = "0" * 64
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "exact maintained Rank-4"):
                support.validate_result(path)


if __name__ == "__main__":
    unittest.main()
