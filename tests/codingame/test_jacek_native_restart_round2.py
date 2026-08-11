#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
import jacek_native_restart_corpus_round2 as corpus  # noqa: E402
import jacek_native_restart_round2 as workflow  # noqa: E402


def collector_bytes(
    rows: tuple[str, ...] = ("1001\t1\t0\t0/0/0/0/0/0",),
    extra_metadata: tuple[str, ...] = (),
) -> bytes:
    lines = [
        "# agent_id=6609056",
        f"# arena_manifest_sha256={'a' * 64}",
        f"# asserted_source_sha256={'b' * 64}",
        "# asserted_submission_id=41123817",
        f"# collector_sha256={'c' * 64}",
        f"# exclusion_registry_sha256={'d' * 64}",
        f"# repository_commit={'e' * 40}",
        "# run_id=restart-test",
        "# source_binding_status=asserted-not-api-verified",
        *extra_metadata,
        "game_id\tcandidate_player\twinner\tturns",
        *rows,
    ]
    return ("\n".join(lines) + "\n").encode()


class RestartContractTests(unittest.TestCase):
    def test_parallel_workers_are_bounded_and_validated(self):
        with mock.patch.object(workflow.os, "cpu_count", return_value=2):
            self.assertEqual(workflow.resolve_parallel_workers(
                argparse.Namespace(shards=3, parallel=None)
            ), 2)
            self.assertEqual(workflow.resolve_parallel_workers(
                argparse.Namespace(shards=1, parallel=None)
            ), 1)
        self.assertEqual(workflow.resolve_parallel_workers(
            argparse.Namespace(shards=3, parallel=3)
        ), 3)
        for shards, parallel in ((0, None), (2, 0), (2, 3), (2, True)):
            with self.subTest(shards=shards, parallel=parallel):
                with self.assertRaisesRegex(
                    ValueError, "shard count|parallel workers"
                ):
                    workflow.resolve_parallel_workers(
                        argparse.Namespace(shards=shards, parallel=parallel)
                    )

    def test_exact_parser_and_deterministic_loss_prefixes(self):
        parsed = corpus.parse_collector_bytes(collector_bytes())
        first = corpus.select_prefixes(parsed, 2)
        second = corpus.select_prefixes(parsed, 2)
        self.assertEqual(first, second)
        self.assertEqual([prefix.prefix_turn for prefix in first], [1, 5])
        self.assertTrue(all(prefix.candidate_player == 1 for prefix in first))
        self.assertEqual(first[1].transcript, "0/0/0/0/0")
        self.assertEqual(
            corpus.select_prefixes(parsed, 2, maximum_selected_prefixes=1),
            (first[1],),
        )

    def test_malformed_duplicate_and_operationally_unclean_inputs_fail(self):
        with self.assertRaisesRegex(ValueError, "metadata fields"):
            corpus.parse_collector_bytes(collector_bytes(
                extra_metadata=("# unexpected=value",)
            ))
        with self.assertRaisesRegex(ValueError, "duplicate game_id"):
            corpus.parse_collector_bytes(collector_bytes(rows=(
                "1001\t1\t0\t0/0/0/0/0/0",
                "1001\t1\t0\t0/0/0/0/0/0",
            )))
        parsed = corpus.parse_collector_bytes(collector_bytes(rows=(
            "1002\t1\t0\t0",
        )))
        with self.assertRaisesRegex(ValueError, "nonterminal"):
            corpus.select_prefixes(parsed, 2)
        parsed = corpus.parse_collector_bytes(collector_bytes(rows=(
            "1003\t1\t0\t000000",
        )))
        with self.assertRaisesRegex(ValueError, "continues after handoff"):
            corpus.select_prefixes(parsed, 2)
        parsed = corpus.parse_collector_bytes(collector_bytes(rows=(
            "1004\t0\t0\t0/0/0/0/0/0",
        )))
        with self.assertRaisesRegex(ValueError, "no candidate losses"):
            corpus.select_prefixes(parsed, 2)

    def test_explicit_path_policy_rejects_forbidden_evidence_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            for relative in (
                pathlib.Path("sealed-clean.tsv"),
                pathlib.Path("final.tsv"),
                pathlib.Path("final") / "clean-auditor.tsv",
            ):
                path = directory / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(collector_bytes())
                with self.subTest(path=relative):
                    with self.assertRaisesRegex(
                        ValueError, "forbidden evidence"
                    ):
                        corpus.read_collector(path)

            final_target = directory / "final.tsv"
            safe_spelling = directory / "clean-link.tsv"
            safe_spelling.symlink_to(final_target)
            with self.assertRaisesRegex(ValueError, "forbidden evidence"):
                corpus.read_collector(safe_spelling)

    @unittest.skipUnless(shutil.which("c++"), "C++ compiler unavailable")
    def test_archived_end_to_end_restart_has_no_observed_move_labels(self):
        checkpoint = ROOT / "models" / "jacek_native_untrained_seed.runtime"
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            source = directory / "clean-auditor.tsv"
            source.write_bytes(collector_bytes())
            output = directory / "restart-run"
            arguments = argparse.Namespace(
                input=source,
                output_dir=output,
                expected_agent_id="6609056",
                expected_submission_id="41123817",
                expected_source_sha256="b" * 64,
                expected_manifest_sha256="a" * 64,
                expected_exclusion_registry_sha256="d" * 64,
                player_one_checkpoint=checkpoint,
                player_two_checkpoint=checkpoint,
                player_one_name="untrained-one",
                player_two_name="untrained-two",
                teacher_checkpoint=None,
                teacher_name="teacher",
                seed=17,
                work=2,
                samples_per_game=4,
                reanalysis_samples_per_game=0,
                prefixes_per_loss=1,
                max_selected_prefixes=1,
                continuations_per_prefix=2,
                shards=2,
                parallel=1,
                temperature=0.0,
                temperature_turns=0,
                max_generated_complete_turns=128,
                reanalysis_work=0,
                verification_work=100_000,
                compiler="c++",
            )
            workflow.run(arguments)
            shards = sorted(output.glob("shard-*.jsonl"))
            self.assertEqual(len(shards), 2)
            games, _, lineage = corpus.load_games(shards)
            self.assertEqual(len(games), 2)
            self.assertEqual(lineage["collector_tsv_sha256"],
                             corpus.sha256_bytes(collector_bytes()))
            records = [
                json.loads(line)
                for shard in shards
                for line in shard.read_text().splitlines()
            ]
            self.assertEqual({record["generator"]["source"]["policy_target"]
                              for record in records}, {None})
            self.assertEqual({record["generator"]["source"]
                              ["observed_moves_usage"] for record in records},
                             {"state-construction-only"})
            rendered = "".join(shard.read_text() for shard in shards)
            self.assertNotIn("teacher_move", rendered)
            self.assertNotIn("actual_action", rendered)
            self.assertEqual(len({record["split_group"] for record in records}), 1)
            manifest = (output / corpus.MANIFEST_NAME).read_bytes()
            self.assertNotIn(b"elapsed", manifest)
            self.assertNotIn(b"throughput", manifest)
            self.assertNotIn(b"parallel", manifest)

            repeated_output = directory / "restart-run-repeat"
            repeated_arguments = argparse.Namespace(
                **{
                    **vars(arguments),
                    "output_dir": repeated_output,
                    "parallel": 2,
                }
            )
            workflow.run(repeated_arguments)
            self.assertEqual(
                manifest,
                (repeated_output / corpus.MANIFEST_NAME).read_bytes(),
            )
            self.assertEqual(
                (output / corpus.BUILD_PROVENANCE_NAME).read_bytes(),
                (repeated_output / corpus.BUILD_PROVENANCE_NAME).read_bytes(),
            )
            repeated_shards = sorted(repeated_output.glob("shard-*.jsonl"))
            self.assertEqual(
                [shard.read_bytes() for shard in shards],
                [shard.read_bytes() for shard in repeated_shards],
            )

            archived = output / corpus.ARCHIVED_INPUT_NAME
            archived.write_bytes(archived.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "blank line|stale"):
                corpus.load_games(shards, verify_local_build=False)


if __name__ == "__main__":
    unittest.main()
