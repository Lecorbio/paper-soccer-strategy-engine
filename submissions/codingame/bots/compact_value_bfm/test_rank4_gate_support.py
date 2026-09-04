import hashlib
import json
import pathlib
import sys
import tempfile
import unittest
from copy import deepcopy


HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rank4_gate_support as support  # noqa: E402


TRANSCRIPT = "4/7/5/2/23/1/7/61/2/7"


def search_intervention(**updates):
    counters = {name: 0 for name in support.SEARCH_INTERVENTION_COUNTERS}
    counters.update(updates)
    return counters


def engine(decisions=0, *, candidate=False, counters=None):
    value = {
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
    if candidate:
        value["search_intervention"] = counters or search_intervention()
    return value


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
                "candidate": engine(1, candidate=True),
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
                "candidate_search_profile": "standard-v1",
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
                "candidate": engine(2, candidate=True),
                "rank4": engine(2),
                "passed": True,
            },
        }

    def profile_document(self, profile, counters):
        document = self.document()
        document["config"]["candidate_search_profile"] = profile
        for game in document["games"]:
            game["candidate"]["search_intervention"] = deepcopy(counters)
        document["result"]["candidate"]["search_intervention"] = {
            name: counters[name] * len(document["games"])
            for name in support.SEARCH_INTERVENTION_COUNTERS
        }
        return document

    def test_result_recomputes_pair_colors_failures_and_bindings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "result.json"
            document = self.document()
            path.write_text(json.dumps(document))
            support.validate_result(
                path,
                expected_bank_sha256="4" * 64,
                expected_candidate_sha256="1" * 64,
                expected_candidate_search_profile="standard-v1",
            )
            evidence = support.require_search_profile_exercised(
                self.document(), expected_profile="standard-v1")
            self.assertTrue(evidence["exercised"])
            self.assertEqual(
                evidence["schema"], support.SEARCH_PROFILE_ACTIVATION_SCHEMA)
            document["games"][1]["candidate_player"] = 0
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "pair/color"):
                support.validate_result(path)
            document = self.document()
            document["bindings"]["opponent_sha256"] = "0" * 64
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "exact maintained Rank-4"):
                support.validate_result(path)

    def test_candidate_counters_are_exact_aggregated_and_candidate_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "result.json"
            counters = search_intervention(
                cache_probes=3, cache_hits=1, cache_misses=2)
            document = self.profile_document(
                "state-evaluation-cache-v1", counters)
            path.write_text(json.dumps(document))
            support.validate_result(
                path,
                expected_candidate_search_profile="state-evaluation-cache-v1",
            )
            evidence = support.require_search_profile_exercised(
                document, expected_profile="state-evaluation-cache-v1")
            self.assertEqual(evidence["search_intervention"]["cache_probes"], 6)
            self.assertEqual(evidence["search_intervention"]["cache_hits"], 2)

            document["result"]["candidate"]["search_intervention"][
                "cache_hits"
            ] += 1
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "cache accounting|aggregate"):
                support.validate_result(path)

            document = self.profile_document(
                "state-evaluation-cache-v1", counters)
            document["games"][0]["rank4"]["search_intervention"] = deepcopy(
                counters)
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "game Rank-4 fields mismatch"):
                support.validate_result(path)

    def test_nonfinite_gate_timings_are_rejected(self):
        for field in ("maximum_first_ms", "maximum_later_ms", "times_ms"):
            for nonfinite in (float("nan"), float("inf"), float("-inf")):
                value = engine(1, candidate=True)
                value[field] = [nonfinite] if field == "times_ms" else nonfinite
                with self.subTest(field=field, nonfinite=nonfinite), \
                        self.assertRaisesRegex(ValueError, "timing"):
                    support._engine(value, "candidate", candidate=True)

    def test_each_intervention_requires_profile_specific_activation(self):
        active = {
            "state-evaluation-cache-v1": search_intervention(
                cache_probes=3, cache_hits=1, cache_misses=2),
            "progressive-widening-v1": search_intervention(
                widening_probes=2, widening_restrictions=1,
                widening_eligible=12, widening_deferred=4),
            "subtree-reuse-v1": search_intervention(
                reuse_probes=3, reuse_hits=1, reuse_misses=1,
                reuse_rejections=1, reused_children=7),
        }
        inactive_effect = {
            "state-evaluation-cache-v1": search_intervention(
                cache_probes=3, cache_misses=3),
            "progressive-widening-v1": search_intervention(
                widening_probes=2, widening_eligible=12),
            "subtree-reuse-v1": search_intervention(
                reuse_probes=3, reuse_misses=2, reuse_rejections=1),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "result.json"
            for profile, counters in active.items():
                with self.subTest(profile=profile, case="active"):
                    document = self.profile_document(profile, counters)
                    path.write_text(json.dumps(document))
                    support.validate_result(path)
                    evidence = support.require_search_profile_exercised(
                        document, expected_profile=profile)
                    self.assertTrue(evidence["exercised"])
            for profile, counters in inactive_effect.items():
                with self.subTest(profile=profile, case="no-effect"):
                    document = self.profile_document(profile, counters)
                    path.write_text(json.dumps(document))
                    support.validate_result(path)
                    with self.assertRaisesRegex(
                            ValueError, "profile was not exercised"):
                        support.require_search_profile_exercised(
                            document, expected_profile=profile)

    def test_aggregate_activation_allows_effect_in_only_one_shard(self):
        cases = {
            "state-evaluation-cache-v1": (
                search_intervention(cache_probes=2, cache_misses=2),
                search_intervention(
                    cache_probes=3, cache_hits=1, cache_misses=2),
            ),
            "progressive-widening-v1": (
                search_intervention(
                    widening_probes=2, widening_eligible=8),
                search_intervention(
                    widening_probes=3, widening_restrictions=1,
                    widening_eligible=12, widening_deferred=4),
            ),
            "subtree-reuse-v1": (
                search_intervention(reuse_probes=2, reuse_misses=2),
                search_intervention(
                    reuse_probes=3, reuse_hits=1, reuse_misses=1,
                    reuse_rejections=1, reused_children=7),
            ),
        }
        for profile, (no_effect, effect) in cases.items():
            with self.subTest(profile=profile):
                first = self.profile_document(profile, no_effect)
                second = self.profile_document(profile, effect)
                with self.assertRaisesRegex(ValueError, "was not exercised"):
                    support.require_search_profile_exercised(
                        first, expected_profile=profile)
                evidence = support.aggregate_search_profile_activation(
                    [first, second], profile)
                self.assertEqual(
                    evidence["schema"],
                    support.SEARCH_PROFILE_ACTIVATION_AGGREGATE_SCHEMA,
                )
                self.assertEqual(evidence["document_count"], 2)
                self.assertEqual(evidence["candidate_decisions"], 4)
                self.assertTrue(evidence["exercised"])
                body = dict(evidence)
                claimed = body.pop("body_sha256")
                self.assertEqual(
                    claimed,
                    hashlib.sha256(
                        support._canonical_json_bytes(body)).hexdigest(),
                )
                self.assertEqual(
                    evidence,
                    support.aggregate_search_profile_activation(
                        [second, first], profile),
                )

    def test_aggregate_activation_rejects_missing_effect_and_profile_mismatch(self):
        no_hit = self.profile_document(
            "state-evaluation-cache-v1",
            search_intervention(cache_probes=2, cache_misses=2),
        )
        with self.assertRaisesRegex(ValueError, "was not exercised"):
            support.aggregate_search_profile_activation(
                [no_hit, deepcopy(no_hit)], "state-evaluation-cache-v1")
        observed = support.aggregate_search_profile_activation(
            [no_hit, deepcopy(no_hit)], "state-evaluation-cache-v1",
            require_exercised=False,
        )
        self.assertFalse(observed["exercised"])
        self.assertFalse(observed["requirements"]["cache_hits_positive"])
        standard = self.document()
        with self.assertRaisesRegex(ValueError, "does not match expectation"):
            support.aggregate_search_profile_activation(
                [no_hit, standard], "state-evaluation-cache-v1")
        with self.assertRaisesRegex(ValueError, "must be nonempty"):
            support.aggregate_search_profile_activation(
                [], "state-evaluation-cache-v1")

    def test_profile_scope_and_counter_accounting_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "result.json"
            counters = search_intervention(
                cache_probes=2, cache_hits=1, cache_misses=1,
                widening_probes=1, widening_restrictions=1,
                widening_eligible=8, widening_deferred=1)
            document = self.profile_document(
                "state-evaluation-cache-v1", counters)
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "inactive search profile"):
                support.validate_result(path)

            counters = search_intervention(
                reuse_probes=2, reuse_hits=1, reuse_misses=1,
                reused_children=0)
            document = self.profile_document("subtree-reuse-v1", counters)
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "subtree-reuse accounting"):
                support.validate_result(path)

    def test_legacy_attempt_zero_requires_explicit_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "attempt-zero.json"
            document = self.document()
            document["schema"] = support.LEGACY_RESULT_SCHEMA
            del document["config"]["candidate_search_profile"]
            for game in document["games"]:
                del game["candidate"]["search_intervention"]
            del document["result"]["candidate"]["search_intervention"]
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "attempt-zero compatibility"):
                support.validate_result(path)
            support.validate_result(path, allow_legacy_attempt_zero=True)
            self.assertNotIn(
                "candidate_search_profile",
                support.legacy_standard_configuration(document),
            )
            with self.assertRaisesRegex(ValueError, "cannot prove"):
                support.validate_result(
                    path,
                    allow_legacy_attempt_zero=True,
                    expected_candidate_search_profile="standard-v1",
                )
            with self.assertRaisesRegex(ValueError, "requires a v2"):
                support.require_search_profile_exercised(document)

    def test_v2_standard_configuration_projects_to_legacy_shape(self):
        document = self.document()
        projected = support.legacy_standard_configuration(document)
        self.assertNotIn("candidate_search_profile", projected)
        self.assertEqual(
            set(document["config"]),
            {*projected, "candidate_search_profile"},
        )
        document["config"]["candidate_search_profile"] = (
            "state-evaluation-cache-v1"
        )
        with self.assertRaisesRegex(ValueError, "cannot consume"):
            support.legacy_standard_configuration(document)

    def test_gate_source_serializes_compile_profile_and_all_counters(self):
        source = (HERE / "rank4_gate.cpp").read_text()
        self.assertIn(support.RESULT_SCHEMA, source)
        for profile in support.SEARCH_PROFILES:
            self.assertIn(f'return "{profile}";', source)
        for counter in support.SEARCH_INTERVENTION_COUNTERS:
            self.assertIn(f"stats.{counter}", source)
            self.assertIn(f'\\"{counter}\\"', source)
        self.assertIn("engine_json(game.candidate, true)", source)
        self.assertIn("engine_json(game.rank4, false)", source)

    def test_candidate_clock_stops_before_harness_state_application(self):
        source = (HERE / "rank4_gate.cpp").read_text()
        candidate = source[
            source.index("Invocation invoke_candidate"):
            source.index("Invocation invoke_rank4")
        ]
        decision = candidate.index(
            "result.search_intervention = search_intervention_counters"
        )
        stopped = candidate.index("stop_timing();", decision)
        applied = candidate.index("apply_both(state, result.action);", decision)
        self.assertLess(stopped, applied)
        for catch in ("catch (const std::exception &error)", "catch (...)"):
            caught = candidate.index(catch)
            stopped = candidate.index("stop_timing();", caught)
            applied = candidate.index("apply_both(state, result.action);", caught)
            self.assertLess(stopped, applied)


if __name__ == "__main__":
    unittest.main()
