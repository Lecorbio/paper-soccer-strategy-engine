import base64
import hashlib
import json
import pathlib
import tempfile
import unittest
from unittest import mock


from submissions.codingame.bots.compact_value_bfm import development_runner as runner


def canonical(value):
    return runner.canonical_json_bytes(value)


def make_runtime(root, architecture, arm):
    shapes = {
        "compact-8x8": (8, 8),
        "source-neutral-8x16": (8, 16),
    }
    h1, h2 = shapes[architecture]
    counts = {"w1": 6301 * h1, "w2": h1 * h2, "w3": h2}
    counts["total"] = sum(counts.values())
    payload = bytes((counts["total"] * 3 + 7) // 8)
    body = {
        "schema": runner.export_model.RUNTIME_SCHEMA,
        "feature_schema": runner.export_model.FEATURE_SCHEMA,
        "architecture": {
            "name": architecture,
            "dimensions": [6301, h1, h2, 1],
            "biases": False,
            "activations": runner.export_model.ACTIVATIONS,
            "payload_layout": runner.export_model.LAYOUT,
        },
        "quantization": {
            **runner.export_model.QUANTIZATION,
            "scales": {"w1": 0.125, "w2": 0.125, "w3": 0.125},
            "weight_counts": counts,
            "packed_byte_count": len(payload),
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "payload_base64": base64.b64encode(payload).decode("ascii"),
        },
        "selection": {
            "arm": arm,
            "seed": 20260907,
            "float_epoch": 1,
            "qat_epoch": 0,
            "source_bundle_body_sha256": "1" * 64,
        },
    }
    document = runner.body_hashed(body)
    raw = canonical(document)
    path = root / "runtimes" / f"{hashlib.sha256(raw).hexdigest()}.runtime.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path


def make_selection(root, architecture, arm, deployment):
    runtime = make_runtime(root, architecture, arm)
    body = {
        "schema": runner.SELECTION_SCHEMA,
        "architecture": architecture,
        "arm": arm,
        "seed": 20260907,
        "float_epoch": 1,
        "qat_epoch": 0,
        "source_bundle_body_sha256": "1" * 64,
        "runtime": {
            "path": str(runtime.relative_to(root)),
            "sha256": runner.sha256_file(runtime),
            "bytes": runtime.stat().st_size,
        },
        "deployment_eligible": deployment,
        "rank4_control_never_deployment_eligible": arm == "rank4-control",
        "protected_tests_opened": False,
        "game_gated": False,
    }
    document = runner.body_hashed(body)
    raw = canonical(document)
    path = root / "selections" / f"{hashlib.sha256(raw).hexdigest()}.selection.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path


def file_record(path):
    return {
        "path": str(path.resolve()), "resolved_path": str(path.resolve()),
        "bytes": path.stat().st_size, "sha256": runner.sha256_file(path),
        "executable": False,
    }


def make_iteration_details(root):
    output = root / "iteration-output"
    runtime = make_runtime(output / "fine-tune", "compact-8x8", "search-target")
    header, metadata = runner.export_model.render_header(runtime)
    _default, source = runner.export_submission.render(model_header=header)
    source_path = output / "fine-tune/generated-sources" / (
        runner.sha256_bytes(source) + ".submission.cpp"
    )
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(source)
    source_export = {
        "runtime_sha256": runner.sha256_file(runtime),
        "runtime_body_sha256": metadata["body_sha256"],
        "model_header_sha256": metadata["header_sha256"],
        "source_sha256": runner.sha256_file(source_path),
        "source_ascii_bytes": len(source),
        "source_limit_exclusive": 95_000,
    }
    selection = runner.body_hashed({
        "schema": runner.campaign.ITERATION_SELECTION_SCHEMA,
        "namespace": runner.NAMESPACE,
        "architecture": "compact-8x8", "seed": 20260907,
        "float_epoch": 1, "qat_epoch": 0,
        "runtime": file_record(runtime),
        "generated_source": file_record(source_path),
        "source_export": source_export,
        "offline_gate": {
            "passed": True,
            "status": "offline-evaluator-qualified-not-game-gated",
            "errors": [],
        },
        "status": "offline-evaluator-qualified-not-game-gated",
        "protected_tests_opened": False,
    })
    selection_raw = canonical(selection)
    selection_path = output / "fine-tune/selections" / (
        runner.sha256_bytes(selection_raw) + ".iteration-selection.json"
    )
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_bytes(selection_raw)
    handoff = runner.body_hashed({
        "schema": runner.campaign.POST_ITERATION_HANDOFF_SCHEMA,
        "namespace": runner.NAMESPACE,
        "status": runner.campaign.POST_ITERATION_HANDOFF_STATUS,
        "plan": {"path": "plan", "sha256": "1" * 64,
                 "body_sha256": "2" * 64},
        "iteration_completion": {
            "path": "completion", "sha256": "3" * 64,
            "body_sha256": "4" * 64,
        },
        "iteration_selection": {
            "path": str(selection_path.resolve()),
            "sha256": runner.sha256_file(selection_path),
            "body_sha256": selection["body_sha256"],
        },
    })
    handoff_path = output / "post-iteration-development-handoff.json"
    handoff_path.write_bytes(canonical(handoff))
    candidate = {
        "candidate_id": runner.campaign.POST_ITERATION_CANDIDATE_ID,
        "architecture": runner.campaign.PRIMARY_ARCHITECTURE,
        "target": "search-target",
        "float_checkpoint": {},
        "runtime": file_record(runtime),
        "generated_source": file_record(source_path),
    }
    return handoff_path, {
        "handoff": handoff, "handoff_path": handoff_path.resolve(),
        "selection": selection, "selection_path": selection_path.resolve(),
        "runtime_path": runtime.resolve(), "source_path": source_path.resolve(),
        "candidate": candidate,
    }


def transcript(stage, index):
    suffix = format(index, "o") or "0"
    stage_digit = str(list(runner.STAGE_PAIRS).index(stage))
    return f"0/1/2/3/4/5/6/7/0/1/{stage_digit}/{suffix}"


def make_banks(root):
    result = {}
    for stage, pairs in runner.STAGE_PAIRS.items():
        rows = [{
            "opening_id": f"{stage}-{index}",
            "transcript": transcript(stage, index),
            "primitive_plies": sum(map(len, transcript(stage, index).split("/"))),
        } for index in range(pairs)]
        artifact = {
            "stage": stage,
            "classification": "unprotected-development",
            "openings": rows,
            "campaign_binding": {
                "bank_id": stage,
                "pairs": pairs,
                "fingerprints": sorted(
                    hashlib.sha256(f"{stage}:{index}".encode()).hexdigest()
                    for index in range(pairs)),
                "transcripts": [row["transcript"] for row in rows],
                "primitive_ply_counts": [row["primitive_plies"] for row in rows],
            },
        }
        raw = canonical(artifact)
        path = root / "banks" / f"{hashlib.sha256(raw).hexdigest()}.opening-bank.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        result[stage] = path
    return result


def fake_validate_bank(path):
    return json.loads(path.read_bytes())


def engine_summary(decisions, latency):
    return {
        "decisions": decisions,
        "deadline_stops": 0,
        "soft_overruns": 0,
        "headroom_failures": 0,
        "hard_timeouts": 0,
        "work": decisions,
        "generated_children": decisions,
        "evaluated_children": decisions,
        "maximum_first_ms": latency if decisions else 0.0,
        "maximum_later_ms": 0.0,
        "times_ms": [latency] * decisions,
    }


class FakeCampaign:
    def __init__(self):
        self.calls = 0

    def outcome(self, spec):
        stage = spec["stage"]
        candidate_id = spec["candidate_id"]
        if stage == "model_screen":
            values = {
                "primary-search": (130, 65, 65, 9.0),
                "primary-teacher": (129, 64, 65, 8.0),
                "neutral-search": (128, 64, 64, 8.0),
                "neutral-teacher": (127, 63, 64, 8.0),
                "rank4-control": (140, 70, 70, 7.0),
            }
            return values[candidate_id]
        if stage == "tuple_screen":
            if candidate_id == "primary-search:c0.80-f0.5-l1":
                return 160, 80, 80, 9.0
            if candidate_id == "primary-search:c0.95-f0.5-l1":
                return 159, 79, 80, 10.0
            ordinal = list(runner.campaign.TUPLE_ROSTER).index(tuple(spec["tuple"]))
            model_penalty = 10 * ["primary-search", "primary-teacher", "neutral-search"].index(
                candidate_id.split(":", 1)[0])
            wins = 120 - ordinal - model_penalty
            return wins, wins // 2, wins - wins // 2, 11.0
        if stage == "tuple_confirmation":
            if candidate_id == "primary-search:c0.80-f0.5-l1":
                return 302, 151, 151, 9.0
            return 300, 150, 150, 10.0
        if stage == "profile_screen":
            values = {"light": (120, 60, 60, 8.0),
                      "default": (159, 79, 80, 10.0),
                      "heavy": (160, 80, 80, 11.0)}
            return values[candidate_id]
        if stage == "profile_confirmation":
            return ((302, 151, 151, 11.0) if candidate_id == "heavy"
                    else (300, 150, 150, 10.0))
        return 211, 104, 107, 155.0

    def __call__(self, candidate, bank, spec):
        self.calls += 1
        wins, color0, color1, latency = self.outcome(spec)
        games = []
        for pair in range(bank.pairs):
            for color, color_wins in ((0, color0), (1, color1)):
                candidate_win = pair < color_wins
                games.append({
                    "opening_id": f"{bank.stage}-{pair}",
                    "pair_index": pair,
                    "candidate_player": color,
                    "winner": color if candidate_win else 1 - color,
                    "turns": 20,
                    "failure": None,
                    "failure_detail": None,
                    "candidate": engine_summary(1, latency),
                    "rank4": engine_summary(1, 1.0),
                })
        candidate_engine = engine_summary(len(games), latency)
        rank4_engine = engine_summary(len(games), 1.0)
        actual = spec["stage"] == "actual_clock"
        return {
            "schema": runner.gate_support.LEGACY_RESULT_SCHEMA,
            "bindings": {
                "candidate_source_sha256": candidate.source_sha256,
                "candidate_source_bytes": candidate.source_bytes,
                "candidate_runtime_body_sha256": candidate.selection_body_sha256,
                "candidate_payload_sha256": "2" * 64,
                "rank4_source_sha256": runner.RANK4_SHA256,
                "rank4_source_bytes": runner.RANK4.stat().st_size,
                "opponent_sha256": runner.RANK4_SHA256,
                "bank_sha256": bank.sha256,
                "bank_bytes": bank.path.stat().st_size,
            },
            "config": {
                "mode": spec["mode"],
                "pair_offset": 0,
                "pair_count": bank.pairs,
                "candidate_c": float(spec["tuple"][0]),
                "candidate_fpu": float(spec["tuple"][1]),
                "candidate_lambda": float(spec["tuple"][2]),
                "candidate_actions": 250,
                "candidate_root_partial_paths": spec["work"]["root_partial_paths"],
                "candidate_nonroot_partial_paths": spec["work"]["nonroot_partial_paths"],
                "candidate_nodes": spec["work"]["nodes"],
                "candidate_expansions": 2_000_000,
                "candidate_shuffle_seed": 1,
                "candidate_clocks_ms": [800, 155],
                "rank4_nodes": 3_000_000,
                "rank4_clocks_ms": [800, 165],
                "max_turns": 320,
                "minimum_candidate_wins": 211 if actual else -1,
                "minimum_wins_per_color": 104 if actual else -1,
            },
            "games": games,
            "result": {
                "games": len(games),
                "candidate_wins": wins,
                "rank4_wins": len(games) - wins,
                "candidate_wins_player0": color0,
                "candidate_wins_player1": color1,
                "failures": 0,
                "unfinished": 0,
                "failure_categories": {},
                "candidate": candidate_engine,
                "rank4": rank4_engine,
                "passed": True,
            },
        }


class PostIterationFakeCampaign(FakeCampaign):
    def outcome(self, spec):
        stage = spec["stage"]
        candidate_id = spec["candidate_id"]
        post = runner.campaign.POST_ITERATION_CANDIDATE_ID
        if stage == "model_screen":
            return ((130, 65, 65, 9.0) if candidate_id == post
                    else (140, 70, 70, 7.0))
        if stage == "tuple_screen":
            if candidate_id == f"{post}:c0.80-f0.5-l1":
                return 160, 80, 80, 9.0
            if candidate_id == f"{post}:c0.95-f0.5-l1":
                return 159, 79, 80, 10.0
            ordinal = list(runner.campaign.TUPLE_ROSTER).index(tuple(spec["tuple"]))
            wins = 120 - ordinal
            return wins, wins // 2, wins - wins // 2, 11.0
        if stage == "tuple_confirmation":
            if candidate_id == f"{post}:c0.80-f0.5-l1":
                return 302, 151, 151, 9.0
            return 300, 150, 150, 10.0
        return super().outcome(spec)


def fake_compiler(_gate, source, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"gate:" + runner.sha256_file(source).encode())


class DevelopmentRunnerTests(unittest.TestCase):
    def fixture(self, root):
        selections = [
            make_selection(root, "compact-8x8", "search-target", True),
            make_selection(root, "compact-8x8", "teacher-assisted", True),
            make_selection(root, "source-neutral-8x16", "search-target", True),
            make_selection(root, "source-neutral-8x16", "teacher-assisted", True),
            make_selection(root, "compact-8x8", "rank4-control", False),
        ]
        return selections, make_banks(root)

    def test_full_mocked_campaign_is_consumable_and_does_not_mutate_active_bot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            selections, banks = self.fixture(root)
            output_root = root / "run"
            development = output_root / "development-input.json"
            active_model = runner.HERE / "model.hpp"
            active_source = runner.HERE / "submission.cpp"
            before = (runner.sha256_file(active_model), runner.sha256_file(active_source))
            fake = FakeCampaign()
            campaign_runner = runner.DevelopmentRunner(
                artifact_root=root, selections=selections, banks=banks,
                output_root=output_root, development_output=development,
                compiler=fake_compiler, gate_executor=fake,
            )
            with (mock.patch.object(runner, "paired_bootstrap_lower", return_value=0.01),
                  mock.patch.object(runner.openings, "validate_bank",
                                    side_effect=fake_validate_bank)):
                payload = campaign_runner.execute()
            self.assertEqual(before, (runner.sha256_file(active_model),
                                      runner.sha256_file(active_source)))
            self.assertEqual(payload["actual_clock"]["wins"], 211)
            runner.campaign.validate_development_input(payload)
            self.assertEqual(json.loads(development.read_bytes()), payload)
            self.assertEqual(fake.calls, 37)
            self.assertTrue(list((output_root / "receipts").glob(
                "*.development-run.json")))

            resumed_fake = mock.Mock(side_effect=AssertionError("gate reran"))
            resumed_compiler = mock.Mock(side_effect=AssertionError("compiler reran"))
            resumed = runner.DevelopmentRunner(
                artifact_root=root, selections=selections, banks=banks,
                output_root=output_root, development_output=development,
                resume=True, compiler=resumed_compiler, gate_executor=resumed_fake,
            )
            with (mock.patch.object(runner, "paired_bootstrap_lower", return_value=0.01),
                  mock.patch.object(runner.openings, "validate_bank",
                                    side_effect=fake_validate_bank)):
                self.assertEqual(resumed.execute(), payload)
            resumed_compiler.assert_not_called()
            resumed_fake.assert_not_called()

    def test_resume_rejects_tampered_reference_and_bank_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            selections, banks = self.fixture(root)
            fake = FakeCampaign()
            output_root = root / "run"
            development = output_root / "development.json"
            first = runner.DevelopmentRunner(
                artifact_root=root, selections=selections, banks=banks,
                output_root=output_root, development_output=development,
                compiler=fake_compiler, gate_executor=fake,
            )
            with (mock.patch.object(runner, "paired_bootstrap_lower", return_value=0.01),
                  mock.patch.object(runner.openings, "validate_bank",
                                    side_effect=fake_validate_bank)):
                first.execute()
            reference = next((output_root / "run-references").glob("*.json"))
            document = json.loads(reference.read_bytes())
            document["receipt_sha256"] = "0" * 64
            reference.write_bytes(canonical(document))
            resumed = runner.DevelopmentRunner(
                artifact_root=root, selections=selections, banks=banks,
                output_root=output_root, development_output=development,
                resume=True, compiler=fake_compiler, gate_executor=fake,
            )
            with (self.assertRaisesRegex(runner.DevelopmentError, "stale"),
                  mock.patch.object(runner, "paired_bootstrap_lower",
                                    return_value=0.01),
                  mock.patch.object(runner.openings, "validate_bank",
                                    side_effect=fake_validate_bank)):
                resumed.execute()

            banks = make_banks(root / "overlap")
            banks["actual_clock"].write_bytes(banks["model_screen"].read_bytes())
            invalid = runner.DevelopmentRunner(
                artifact_root=root, selections=selections, banks=banks,
                output_root=root / "invalid", development_output=root / "invalid.json",
                compiler=fake_compiler, gate_executor=fake,
            )
            with (self.assertRaisesRegex(
                    runner.DevelopmentError,
                    "not content addressed|exactly 200 pairs|mutually disjoint"),
                  mock.patch.object(runner.openings, "validate_bank",
                                    side_effect=fake_validate_bank)):
                invalid.execute()

    def test_family_runtime_is_relative_to_each_campaign_not_shared_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            first = make_selection(
                root / "campaign-a", "compact-8x8", "search-target", True
            )
            second = make_selection(
                root / "campaign-b", "source-neutral-8x16",
                "teacher-assisted", True,
            )
            first_selection, first_runtime = runner._selection_runtime(first)
            second_selection, second_runtime = runner._selection_runtime(second)
            self.assertEqual(first_selection["arm"], "search-target")
            self.assertEqual(second_selection["arm"], "teacher-assisted")
            self.assertEqual(first_runtime.parent.parent, (root / "campaign-a").resolve())
            self.assertEqual(second_runtime.parent.parent, (root / "campaign-b").resolve())

    def test_post_iteration_runs_single_model_path_and_compiles_exact_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            control = make_selection(
                root / "control-campaign", "compact-8x8",
                "rank4-control", False,
            )
            handoff, details = make_iteration_details(root)
            banks = make_banks(root)
            output_root = root / "post-development"
            development = output_root / "development-input.json"
            fake = PostIterationFakeCampaign()
            compiled_sources = []

            def compiler(gate, source, output):
                compiled_sources.append(source.resolve())
                fake_compiler(gate, source, output)

            campaign_runner = runner.DevelopmentRunner(
                artifact_root=root / "unused-shared-root",
                selections=[control], post_iteration_handoff=handoff,
                banks=banks, output_root=output_root,
                development_output=development, compiler=compiler,
                gate_executor=fake,
            )
            with (
                mock.patch.object(
                    runner.campaign, "validate_post_iteration_handoff",
                    return_value=details,
                ),
                mock.patch.object(runner, "paired_bootstrap_lower", return_value=0.01),
                mock.patch.object(
                    runner.openings, "validate_bank", side_effect=fake_validate_bank,
                ),
            ):
                payload = campaign_runner.execute()
            self.assertEqual(
                payload["development_mode"],
                runner.campaign.POST_ITERATION_DEVELOPMENT_MODE,
            )
            self.assertEqual(len(payload["eligible_model_arms"]), 1)
            self.assertEqual(
                payload["eligible_model_arms"][0],
                [runner.campaign.PRIMARY_ARCHITECTURE, "search-target"],
            )
            self.assertEqual(len(payload["tuple_screen"]), 8)
            self.assertEqual(fake.calls, 18)
            self.assertIn(details["source_path"].resolve(), compiled_sources)
            self.assertEqual(payload["actual_clock"]["wins"], 211)

    def test_post_iteration_missing_control_and_runtime_source_drift_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            handoff, details = make_iteration_details(root)
            banks = make_banks(root)
            missing = runner.DevelopmentRunner(
                artifact_root=root, selections=[], post_iteration_handoff=handoff,
                banks=banks, output_root=root / "missing",
                development_output=root / "missing.json", compiler=fake_compiler,
                gate_executor=PostIterationFakeCampaign(),
            )
            with (
                mock.patch.object(
                    runner.campaign, "validate_post_iteration_handoff",
                    return_value=details,
                ),
                mock.patch.object(
                    runner.openings, "validate_bank", side_effect=fake_validate_bank,
                ),
                self.assertRaisesRegex(runner.DevelopmentError, "roster is incomplete"),
            ):
                missing.execute()

            control = make_selection(
                root / "control", "compact-8x8", "rank4-control", False
            )
            original = details["source_path"].read_bytes()
            details["source_path"].write_bytes(original + b"\n")
            drift = runner.DevelopmentRunner(
                artifact_root=root, selections=[control],
                post_iteration_handoff=handoff, banks=banks,
                output_root=root / "drift", development_output=root / "drift.json",
                compiler=fake_compiler, gate_executor=PostIterationFakeCampaign(),
            )
            with (
                mock.patch.object(
                    runner.campaign, "validate_post_iteration_handoff",
                    return_value=details,
                ),
                mock.patch.object(
                    runner.openings, "validate_bank", side_effect=fake_validate_bank,
                ),
                self.assertRaisesRegex(
                    runner.DevelopmentError, "runtime/generated source identity changed"
                ),
            ):
                drift.execute()


if __name__ == "__main__":
    unittest.main()
