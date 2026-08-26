import hashlib
import json
import os
import pathlib
import sys
import tempfile
import textwrap
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import jacek_replay_features as features  # noqa: E402
import jacek_selfsearch_workflow as workflow  # noqa: E402


SHORT_WIN = "0/0/3/0/61/0/07"


def write_json(path: pathlib.Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(workflow.canonical_json_bytes(value, pretty=True))


def search_teacher_row(position_id: str, value: float) -> dict:
    return {
        "schema": workflow.SEARCH_TEACHER_SCHEMA,
        "campaign_id": "selfsearch-workflow-fixture",
        "position_id": position_id,
        "root_group_id": "r",
        "group_id": "g",
        "source": "s",
        "split": "train",
        "winner": 0,
        "prefix": [],
        "mover": 0,
        "teacher": {
            "kind": "jacek_replay_bfm_search",
            "source_sha256": "1" * 64,
            "model_sha256": "2" * 64,
            "feature_schema": features.FEATURE_SCHEMA,
            "feature_schema_sha256": hashlib.sha256(
                features.FEATURE_SCHEMA.encode()
            ).hexdigest(),
        },
        "search_config": {
            "seed": 7,
            "max_time_ms": 0,
            "max_tree_nodes": 64,
            "max_actions": workflow.SEARCH_MAX_ACTIONS,
            "max_partial_paths": workflow.SEARCH_MAX_PARTIAL_PATHS,
            "exploration": 0.5,
            "fpu": 0.5,
        },
        "search_stats": {
            "expansions": 2,
            "generated_actions": 8,
            "retained_actions": 8,
            "neural_evaluations": 4,
            "visits": 63,
            "completed_actions": 8,
            "duplicate_boundaries": 0,
            "partial_paths": 8,
            "fifo_extractions": 0,
            "lifo_extractions": 8,
            "tactical_proofs": 0,
            "tactical_solutions": 0,
            "truncations": 1,
            "generation_action_cap_stops": 1,
            "generation_partial_cap_stops": 0,
            "generation_deadline_stops": 0,
            "materialization_deadline_stops": 0,
            "generation_queue_drops": 0,
            "generation_retention_drops": 0,
            "generation_boundary_replacements": 0,
            "generation_tactical_shortcuts": 0,
            "generation_fallbacks": 0,
            "generation_frontier_resumptions": 0,
            "generation_zero_action_resumptions": 0,
            "generation_max_frontier_depth": 1,
            "progressive_widenings": 0,
            "closed_unsolved_nodes": 0,
            "closed_unsolved_nonexhaustive_nodes": 0,
            "open_unexpanded_nodes": 8,
            "implicit_action_frontiers": 0,
            "max_open_children": 8,
            "tree_nodes": 64,
            "max_complete_turn_depth": 2,
            "deadline_reached": False,
            "tree_cap_reached": True,
            "termination_reason": "fixed-work-cap",
        },
        "teacher_value": value,
        "root_solved": False,
        "proven_winner": None,
        "weight": 1.0,
    }


def rank4_teacher_row(position_id: str, score: int = 100) -> dict:
    return {
        "schema": workflow.RANK4_TEACHER_SCHEMA,
        "campaign_id": "selfsearch-workflow-fixture",
        "position_id": position_id,
        "root_group_id": "r",
        "group_id": "g",
        "source": "s",
        "split": "train",
        "winner": 0,
        "prefix": [],
        "mover": 0,
        "teacher": {
            "kind": "rank4-fixed-work",
            "source_sha256": "3" * 64,
        },
        "search_config": {
            "max_nodes": 32_000,
            "max_time_ms": 0,
            "max_turn_depth": 32,
            "replay_value_blend_percent": 15,
            "teacher_residual_weight_percent": 100,
        },
        "search_stats": {
            "attempted_depth": 1,
            "completed_depth": 0,
            "nodes": 32_000,
            "leaf_evaluations": 1,
            "terminal_nodes": 0,
            "completed_actions": 1,
            "budget_exhausted": True,
            "node_cap_reached": True,
            "depth_cap_reached": False,
            "deadline_reached": False,
            "termination_reason": "fixed-work-cap",
        },
        "root_score": score,
        "completed_depth": 0,
        "nodes": 32_000,
        "root_solved": False,
        "proven_winner": None,
        "weight": 1.0,
    }


class SelfSearchWorkflowTests(unittest.TestCase):
    def test_campaign_output_requires_current_version_basename(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            missing = root / "missing"
            output = root / "selfsearch-auto-20260825-v4"
            executables = workflow.CampaignExecutables(
                continuation_generator=missing,
                search_teacher=missing,
                rank4_teacher=missing,
                comparison=missing,
                pack_tool=missing,
                trainer=missing,
            )
            with self.assertRaisesRegex(ValueError, workflow.AUTO_CAMPAIGN_ID):
                workflow.run_campaign(
                    repository=root,
                    expected_commit="1" * 40,
                    evaluation_directory=root / "evaluation",
                    canonical_campaign=root / "canonical",
                    output=output,
                    executables=executables,
                    build_manifest=missing,
                    resume=False,
                    wait_for_evaluation=False,
                    poll_seconds=1.0,
                    skip_power_check=False,
                )
            self.assertFalse(output.exists())

    def test_evaluation_trigger_recursively_checks_reports_and_latency(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            comparison = root / "comparison"
            comparison.write_bytes(b"comparison")
            comparison_sha = workflow.sha256(comparison)
            write_json(
                root / "run-manifest.json",
                {
                    "schema": "papersoccer.jacek-replay-postcampaign-run.v1",
                    "producer_commit": "1" * 40,
                    "configuration": {
                        "workers": 4,
                        "pair_count_per_panel": 500,
                        "pairs_per_shard": 5,
                        "time_ms": 980,
                        "exploration": 0.5,
                        "fpu": 0.5,
                        "step1_games": 2_000,
                        "step2_games": 3_000,
                    },
                    "inputs": {str(comparison): comparison_sha},
                    "producers": {"comparison_executable_sha256": comparison_sha},
                },
            )
            bindings = []

            def add_job(job_id: str, phase: str, opponent: str, offset: int) -> None:
                expected_games = 20 if phase == "step1" else 10
                report = root / f"shards/{phase}/{job_id}.json"
                report_value = {
                    "schema": "papersoccer.jacek-replay-bfm-comparison.v1",
                    "model_sha256": "a" * 64,
                    "configuration": {
                        "pairs": 5,
                        "pair_offset": offset,
                        "time_ms": 980,
                        "exploration": 0.5,
                        "fpu": 0.5,
                        "opening_plies": 12,
                        "max_turns": 320,
                        "single_thread": True,
                        "opponent": opponent,
                        "opening_bank_sha256": "b" * 64,
                        "comparison_executable_sha256": comparison_sha,
                    },
                    "summary": {
                        "games": expected_games,
                        "wins": expected_games,
                        "losses": 0,
                        "unfinished": 0,
                        "illegal": 0,
                    },
                    "results": [{} for _ in range(expected_games)],
                }
                write_json(report, report_value)
                receipt = root / f"receipts/{job_id}.json"
                write_json(
                    receipt,
                    {
                        "schema": "papersoccer.jacek-replay-shard-receipt.v1",
                        "job": {
                            "job_id": job_id,
                            "phase": phase,
                            "pairs": 5,
                            "offset": offset,
                            "expected_games": expected_games,
                            "time_ms": 980,
                            "opponent": opponent,
                            "model_sha256": "a" * 64,
                            "bank_sha256": "b" * 64,
                        },
                        "report_sha256": workflow.sha256(report),
                        "wins": expected_games,
                        "losses": 0,
                        "unfinished": 0,
                        "illegal": 0,
                    },
                )
                bindings.append({
                    "job_id": job_id,
                    "report_sha256": workflow.sha256(report),
                    "receipt_sha256": workflow.sha256(receipt),
                })

            for offset in range(0, 500, 5):
                add_job(f"step1-r2-controls-{offset:03d}", "step1",
                        "rank4-jacek-nn", offset)
            for matchup in ("r0-r1", "r1-r2", "r0-r2"):
                for offset in range(0, 500, 5):
                    add_job(f"step2-{matchup}-{offset:03d}", "step2",
                            "jacek-replay", offset)

            audit_summary = {
                "games": 20, "wins": 20, "losses": 0,
                "unfinished": 0, "illegal": 0,
            }
            write_json(
                root / "latency-audit.json",
                {
                    "schema": "papersoccer.jacek-replay-bfm-comparison.v1",
                    "configuration": {
                        "pairs": 5, "pair_offset": 0, "time_ms": 980,
                        "exploration": 0.5, "fpu": 0.5,
                        "opening_plies": 12, "max_turns": 320,
                        "single_thread": True, "opponent": "rank4-jacek-nn",
                        "comparison_executable_sha256": comparison_sha,
                    },
                    "summary": audit_summary,
                    "results": [{} for _ in range(20)],
                },
            )
            summary = root / "final-summary.json"
            write_json(
                summary,
                {
                    "schema": "papersoccer.jacek-replay-postcampaign-summary.v1",
                    "games": 5_000,
                    "producer_commit": "1" * 40,
                    "step1": {
                        name: {"games": 1_000, "illegal": 0, "unfinished": 0}
                        for name in ("rank4", "jacek-nn")
                    },
                    "step2": {
                        name: {"lower_round_as_candidate": {
                            "games": 1_000, "illegal": 0, "unfinished": 0,
                        }}
                        for name in ("r0-vs-r1", "r1-vs-r2", "r0-vs-r2")
                    },
                    "sequential_latency_audit": audit_summary,
                    "reports": bindings,
                },
            )
            write_json(
                root / "supervisor-status.json",
                {
                    "phase": "complete",
                    "summary_sha256": workflow.sha256(summary),
                },
            )
            validated = workflow.validate_evaluation_trigger(root)
            self.assertEqual(validated["evaluation_summary"]["sha256"], workflow.sha256(summary))
            report = root / "shards/step1/step1-r2-controls-000.json"
            report.write_text("corrupt\n")
            with self.assertRaisesRegex(ValueError, "report/receipt"):
                workflow.validate_evaluation_trigger(root)

    def test_fake_teacher_chunks_resume_without_rerun_and_fail_closed_on_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            positions = root / "positions.tsv"
            positions.write_text(
                "position_id\troot_group_id\tgroup_id\tsource\tsplit\twinner\tmover\tprefix\n"
                + "".join(
                    f"p{index}\tr{index}\tg{index}\ts\ttrain\t0\t0\t\n"
                    for index in range(26)
                )
            )
            counter = root / "calls.txt"
            teacher = root / "fake-rank4-teacher.py"
            teacher.write_text(
                textwrap.dedent(
                    f"""\
                    #!/usr/bin/env python3
                    import argparse, json, sys
                    parser = argparse.ArgumentParser()
                    parser.add_argument('--campaign-id', required=True)
                    parser.add_argument('--nodes', type=int, required=True)
                    parser.add_argument('--time-ms', type=int, required=True)
                    args = parser.parse_args()
                    with open({str(counter)!r}, 'a', encoding='utf-8') as calls:
                        calls.write('call\\n')
                    lines = sys.stdin.read().splitlines()
                    for line in lines[1:]:
                        fields = line.split('\\t')
                        row = {{
                            'schema': {workflow.RANK4_TEACHER_SCHEMA!r},
                            'campaign_id': args.campaign_id,
                            'position_id': fields[0], 'root_group_id': fields[1],
                            'group_id': fields[2], 'source': fields[3],
                            'split': fields[4], 'winner': int(fields[5]),
                            'mover': int(fields[6]), 'prefix': [],
                            'teacher': {{'kind': 'rank4-fixed-work',
                                        'source_sha256': '1' * 64}},
                            'root_score': 100,
                            'completed_depth': 0, 'nodes': args.nodes,
                            'root_solved': False, 'proven_winner': None, 'weight': 1.0,
                            'search_config': {{'max_nodes': args.nodes,
                                              'max_time_ms': args.time_ms,
                                              'max_turn_depth': 32,
                                              'replay_value_blend_percent': 15,
                                              'teacher_residual_weight_percent': 100}},
                            'search_stats': {{'attempted_depth': 1,
                                             'completed_depth': 0,
                                             'nodes': args.nodes,
                                             'leaf_evaluations': 1,
                                             'terminal_nodes': 0,
                                             'completed_actions': 1,
                                             'budget_exhausted': True,
                                             'node_cap_reached': True,
                                             'depth_cap_reached': False,
                                             'deadline_reached': False,
                                             'termination_reason': 'fixed-work-cap'}},
                        }}
                        print(json.dumps(row, separators=(',', ':')))
                    """
                )
            )
            teacher.chmod(0o755)
            output_root = root / "attempt"
            output_root.mkdir()
            manager = workflow.StageManager(
                output=output_root,
                campaign_id=workflow.PILOT_CAMPAIGN_ID,
                round_index=0,
                resume=False,
                environment={"fixture": True},
            )
            labels = output_root / "labels.jsonl"
            result = workflow.run_label_chunks(
                manager=manager,
                stage_ordinal=3,
                stage_name="rank4-shallow",
                positions=positions,
                output=labels,
                teacher=teacher,
                schema=workflow.RANK4_TEACHER_SCHEMA,
                campaign_id=workflow.PILOT_CAMPAIGN_ID,
                nodes=32_000,
                workers=2,
                source_sha256="1" * 64,
            )
            self.assertEqual(result["teacher_rows"], 26)
            self.assertEqual(len(result["chunks"]), 2)
            self.assertEqual(counter.read_text().splitlines(), ["call", "call"])

            resumed = workflow.StageManager(
                output=output_root,
                campaign_id=workflow.PILOT_CAMPAIGN_ID,
                round_index=0,
                resume=True,
                environment={"fixture": True},
            )
            workflow.run_label_chunks(
                manager=resumed,
                stage_ordinal=3,
                stage_name="rank4-shallow",
                positions=positions,
                output=labels,
                teacher=teacher,
                schema=workflow.RANK4_TEACHER_SCHEMA,
                campaign_id=workflow.PILOT_CAMPAIGN_ID,
                nodes=32_000,
                workers=2,
                source_sha256="1" * 64,
            )
            self.assertEqual(counter.read_text().splitlines(), ["call", "call"])
            chunk = output_root / "label-chunks/rank4-shallow/chunk-000000.jsonl"
            chunk.write_bytes(chunk.read_bytes() + b"{}\n")
            with self.assertRaisesRegex(ValueError, "stale"):
                workflow.run_label_chunks(
                    manager=resumed,
                    stage_ordinal=3,
                    stage_name="rank4-shallow",
                    positions=positions,
                    output=labels,
                    teacher=teacher,
                    schema=workflow.RANK4_TEACHER_SCHEMA,
                    campaign_id=workflow.PILOT_CAMPAIGN_ID,
                    nodes=32_000,
                    workers=2,
                    source_sha256="1" * 64,
                )

    def test_label_idle_watchdog_leaves_no_output_or_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            positions = root / "positions.tsv"
            positions.write_text(
                "position_id\troot_group_id\tgroup_id\tsource\tsplit\t"
                "winner\tmover\tprefix\n"
                "p0\tr0\tg0\ts\ttrain\t0\t0\t\n"
                "p1\tr1\tg1\ts\ttrain\t0\t0\t\n"
            )
            teacher = root / "stream-then-hang.py"
            teacher.write_text(
                textwrap.dedent(
                    f"""\
                    #!{sys.executable}
                    import argparse, json, sys, time
                    parser = argparse.ArgumentParser()
                    parser.add_argument('--campaign-id', required=True)
                    parser.add_argument('--nodes', type=int, required=True)
                    parser.add_argument('--time-ms', type=int, required=True)
                    args = parser.parse_args()
                    fields = sys.stdin.read().splitlines()[1].split('\\t')
                    row = {{
                        'schema': {workflow.RANK4_TEACHER_SCHEMA!r},
                        'campaign_id': args.campaign_id,
                        'position_id': fields[0], 'root_group_id': fields[1],
                        'group_id': fields[2], 'source': fields[3],
                        'split': fields[4], 'winner': int(fields[5]),
                        'mover': int(fields[6]), 'prefix': [],
                        'teacher': {{'kind': 'rank4-fixed-work',
                                    'source_sha256': '1' * 64}},
                        'search_config': {{
                            'max_nodes': args.nodes,
                            'max_time_ms': args.time_ms,
                            'max_turn_depth': 32,
                            'replay_value_blend_percent': 15,
                            'teacher_residual_weight_percent': 100,
                        }},
                        'search_stats': {{
                            'attempted_depth': 1, 'completed_depth': 0,
                            'nodes': args.nodes, 'leaf_evaluations': 1,
                            'terminal_nodes': 0, 'completed_actions': 1,
                            'budget_exhausted': True,
                            'node_cap_reached': True,
                            'depth_cap_reached': False,
                            'deadline_reached': False,
                            'termination_reason': 'fixed-work-cap',
                        }},
                        'root_score': 100, 'completed_depth': 0,
                        'nodes': args.nodes, 'root_solved': False,
                        'proven_winner': None, 'weight': 1.0,
                    }}
                    print(json.dumps(row, separators=(',', ':')), flush=True)
                    time.sleep(60)
                    """
                )
            )
            teacher.chmod(0o755)
            output_root = root / "attempt"
            output_root.mkdir()
            manager = workflow.StageManager(
                output=output_root,
                campaign_id=workflow.PILOT_CAMPAIGN_ID,
                round_index=0,
                resume=False,
                environment={"fixture": True},
            )
            labels = output_root / "labels.jsonl"
            with self.assertRaisesRegex(RuntimeError, "next position_id=p1"):
                workflow.run_label_chunks(
                    manager=manager,
                    stage_ordinal=4,
                    stage_name="rank4-watchdog",
                    positions=positions,
                    output=labels,
                    teacher=teacher,
                    schema=workflow.RANK4_TEACHER_SCHEMA,
                    campaign_id=workflow.PILOT_CAMPAIGN_ID,
                    nodes=64,
                    workers=1,
                    source_sha256="1" * 64,
                    process_idle_timeout_seconds=2.0,
                )
            chunk_root = output_root / "label-chunks/rank4-watchdog"
            receipt = output_root / (
                "receipts/04-rank4-watchdog-chunks/chunk-000000.json"
            )
            self.assertFalse(labels.exists())
            self.assertFalse((chunk_root / "chunk-000000.jsonl").exists())
            self.assertFalse(receipt.exists())
            self.assertEqual(list(chunk_root.glob(".chunk-000000.jsonl.*")), [])

    def test_game_chunks_separate_plan_outputs_resume_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            roots = root / "roots.tsv"
            roots.write_text(
                "group_id\tsource\twinner\ttranscript\n"
                f"root:one\tfixture\t0\t{SHORT_WIN}\n"
            )
            actor = root / "actor.runtime"
            diversity = root / "diversity.runtime"
            actor.write_bytes(b"actor")
            diversity.write_bytes(b"diversity")
            plan = workflow.make_game_plan(
                campaign_id=workflow.PILOT_CAMPAIGN_ID,
                seed=17,
                quotas={"incumbent-selfplay": 26},
            )
            plan_path = root / "plan.json"
            write_json(plan_path, plan)
            counter = root / "generator-calls.txt"
            generator = root / "fake-generator.py"
            generator.write_text(
                textwrap.dedent(
                    f"""\
                    #!/usr/bin/env python3
                    import argparse, hashlib, json, pathlib
                    parser = argparse.ArgumentParser()
                    parser.add_argument('--input', type=pathlib.Path, required=True)
                    parser.add_argument('--output', type=pathlib.Path, required=True)
                    parser.add_argument('--manifest', type=pathlib.Path, required=True)
                    parser.add_argument('--model', type=pathlib.Path, required=True)
                    parser.add_argument('--runner-up-model', type=pathlib.Path, required=True)
                    parser.add_argument('--selfsearch-plan', type=pathlib.Path, required=True)
                    parser.add_argument('--campaign-id', required=True)
                    parser.add_argument('--games', type=int, required=True)
                    parser.add_argument('--actor-nodes', type=int, required=True)
                    parser.add_argument('--candidate-tree-nodes', type=int, required=True)
                    parser.add_argument('--jacek-nn-nodes', type=int, required=True)
                    parser.add_argument('--candidate-exploration', type=float, required=True)
                    parser.add_argument('--candidate-fpu', type=float, required=True)
                    args = parser.parse_args()
                    def digest(path):
                        return hashlib.sha256(path.read_bytes()).hexdigest()
                    with open({str(counter)!r}, 'a', encoding='utf-8') as calls:
                        calls.write('call\\n')
                    plan_rows = [line.split('\\t') for line in
                                 args.selfsearch_plan.read_text().splitlines()[1:]]
                    transcript = {SHORT_WIN!r}
                    payload = 'group_id\\tsource\\twinner\\ttranscript\\n' + ''.join(
                        f'root:one\\t{{args.campaign_id}}\\t0\\t{{transcript}}\\n'
                        for _ in plan_rows
                    )
                    args.output.write_text(payload)
                    sources = {{
                        'producer_source_sha256': '1' * 64,
                        'rank4_actor_source_sha256': '2' * 64,
                        'jacek_nn_actor_source_sha256': '3' * 64,
                    }}
                    rows = []
                    for row_ordinal, (game_ordinal, mode, base_seed) in enumerate(plan_rows):
                        rows.append({{
                            'game_id': f'game:{{game_ordinal}}',
                            'row_ordinal': row_ordinal,
                            'game_ordinal': int(game_ordinal),
                            'attempt_ordinal': 0,
                            'base_seed': int(base_seed),
                            'game_seed': int(base_seed),
                            'actor_mode': mode,
                            'root_group_id': 'root:one',
                            'prefix_turns': 1,
                            'winner': 0,
                            'transcript_sha256': hashlib.sha256(
                                transcript.encode()).hexdigest(),
                        }})
                    manifest = {{
                        'schema': {workflow.GAME_MANIFEST_SCHEMA!r},
                        'campaign_id': args.campaign_id,
                        'requested_games': args.games,
                        'successful_games': args.games,
                        'configuration': {{
                            'bfm_tree_nodes': args.candidate_tree_nodes,
                            'rank4_nodes': args.actor_nodes,
                            'jacek_nn_nodes': args.jacek_nn_nodes,
                            'exploration': args.candidate_exploration,
                            'fpu': args.candidate_fpu,
                            'early_exploration_percent': 15,
                            'early_exploration_turns': 8,
                            'maximum_turns': 320,
                            **sources,
                        }},
                        'bindings': {{
                            'roots_sha256': digest(args.input),
                            'plan_sha256': digest(args.selfsearch_plan),
                            'output_sha256': digest(args.output),
                            'incumbent_model_sha256': digest(args.model),
                            'runner_up_model_sha256': digest(args.runner_up_model),
                            **sources,
                        }},
                        'rows': rows,
                    }}
                    args.manifest.write_text(json.dumps(manifest))
                    """
                )
            )
            generator.chmod(0o755)
            configuration = {
                **workflow.PILOT_CONFIGURATION,
                "games": 26,
                "game_chunk_size": 25,
                "game_workers": 2,
            }
            spec = workflow.PhaseSpec(
                name="pilot",
                campaign_id=workflow.PILOT_CAMPAIGN_ID,
                configuration=configuration,
                quotas={"incumbent-selfplay": 26},
                game_seed=17,
                opening_seed=19,
                pairs=1,
                gate_time_ms=1,
                gate_workers=1,
                bank_classification="development",
            )
            output_root = root / "attempt"
            output_root.mkdir()
            manager = workflow.StageManager(
                output=output_root,
                campaign_id=workflow.PILOT_CAMPAIGN_ID,
                round_index=0,
                resume=False,
                environment={"fixture": True},
            )
            result = workflow.run_game_chunks(
                manager=manager,
                stage_ordinal=1,
                spec=spec,
                plan_path=plan_path,
                roots_tsv=roots,
                actor=actor,
                diversity=diversity,
                generator=generator,
                workers=2,
                source_identities={
                    "continuation_source_sha256": "1" * 64,
                    "rank4_actor_source_sha256": "2" * 64,
                    "jacek_nn_actor_source_sha256": "3" * 64,
                },
            )
            self.assertEqual(result["games"], 26)
            self.assertEqual(counter.read_text().splitlines(), ["call", "call"])
            self.assertTrue((output_root / "game-chunks/plans/chunk-000000.tsv").is_file())
            output_chunk = output_root / "game-chunks/outputs/chunk-000000.tsv"
            self.assertTrue(output_chunk.is_file())

            resumed = workflow.StageManager(
                output=output_root,
                campaign_id=workflow.PILOT_CAMPAIGN_ID,
                round_index=0,
                resume=True,
                environment={"fixture": True},
            )
            workflow.run_game_chunks(
                manager=resumed,
                stage_ordinal=1,
                spec=spec,
                plan_path=plan_path,
                roots_tsv=roots,
                actor=actor,
                diversity=diversity,
                generator=generator,
                workers=2,
                source_identities={
                    "continuation_source_sha256": "1" * 64,
                    "rank4_actor_source_sha256": "2" * 64,
                    "jacek_nn_actor_source_sha256": "3" * 64,
                },
            )
            self.assertEqual(counter.read_text().splitlines(), ["call", "call"])
            output_chunk.write_bytes(output_chunk.read_bytes() + b"corrupt\n")
            with self.assertRaisesRegex(ValueError, "stale"):
                workflow.run_game_chunks(
                    manager=resumed,
                    stage_ordinal=1,
                    spec=spec,
                    plan_path=plan_path,
                    roots_tsv=roots,
                    actor=actor,
                    diversity=diversity,
                    generator=generator,
                    workers=2,
                    source_identities={
                        "continuation_source_sha256": "1" * 64,
                        "rank4_actor_source_sha256": "2" * 64,
                        "jacek_nn_actor_source_sha256": "3" * 64,
                    },
                )

    def test_game_plan_is_exact_and_deterministic(self):
        first = workflow.make_game_plan(
            campaign_id=workflow.PILOT_CAMPAIGN_ID,
            seed=123,
            quotas=workflow.PILOT_QUOTAS,
        )
        second = workflow.make_game_plan(
            campaign_id=workflow.PILOT_CAMPAIGN_ID,
            seed=123,
            quotas=workflow.PILOT_QUOTAS,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["games"], 2_000)
        self.assertEqual(
            {mode: sum(row["actor_mode"] == mode for row in first["rows"])
             for mode in workflow.PILOT_QUOTAS},
            workflow.PILOT_QUOTAS,
        )
        self.assertEqual(
            [row["game_ordinal"] for row in first["rows"]], list(range(2_000))
        )

    def test_incumbent_selection_uses_complete_direct_league(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            campaign = root / "canonical"
            for round_, huber in enumerate((0.05, 0.04, 0.045)):
                model = campaign / f"round-{round_}/model"
                model.mkdir(parents=True)
                runtime = model / "jacek_replay_bfm.runtime"
                runtime.write_bytes(f"round-{round_}".encode())
                write_json(
                    model / "jacek_replay_bfm.runtime.json",
                    {
                        "training": {
                            "chosen_seed": round_,
                            "seed_reports": [
                                {
                                    "seed": round_,
                                    "validation": {
                                        "weighted_huber": huber,
                                        "sign_accuracy": 0.8,
                                        "correlation": 0.7,
                                    },
                                }
                            ],
                        }
                    },
                )

            def matchup(lower_wins):
                return {
                    "lower_round_as_candidate": {
                        "games": 1_000,
                        "wins": lower_wins,
                        "unfinished": 0,
                        "illegal": 0,
                        "opening_sweeps": lower_wins // 2,
                        "opening_losses": (1_000 - lower_wins) // 2,
                        "colors": {
                            "0": {"wins": lower_wins // 2},
                            "1": {"wins": lower_wins - lower_wins // 2},
                        },
                    }
                }

            evaluation = root / "summary.json"
            write_json(
                evaluation,
                {
                    "schema": "papersoccer.jacek-replay-postcampaign-summary.v1",
                    "games": 5_000,
                    "sequential_latency_audit": {"games": 20},
                    "step2": {
                        "r0-vs-r1": matchup(400),
                        "r1-vs-r2": matchup(550),
                        "r0-vs-r2": matchup(450),
                    },
                },
            )
            selected = workflow.select_incumbent(evaluation, campaign)
        self.assertEqual(selected["incumbent"]["round"], 1)
        self.assertEqual(selected["runner_up"]["round"], 2)

    def test_canonical_anchors_cover_all_three_rounds_and_match_model_ancestry(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = pathlib.Path(directory)
            source_shards = []
            expected_train = []
            expected_validation = []
            for round_index in range(3):
                shard_records = {}
                for split in ("train", "validation", "test"):
                    shard_directory = campaign / f"round-{round_index}/shards"
                    npz = shard_directory / f"{split}.npz"
                    npz.parent.mkdir(parents=True, exist_ok=True)
                    npz.write_bytes(f"round-{round_index}-{split}".encode())
                    manifest = shard_directory / f"{split}.json"
                    manifest_value = {
                        "split": split,
                        "npz": npz.name,
                        "npz_sha256": workflow.sha256(npz),
                    }
                    write_json(manifest, manifest_value)
                    source_shards.append(manifest_value)
                    shard_records[split] = {
                        "manifest": str(manifest),
                        "manifest_sha256": workflow.sha256(manifest),
                        "sha256": workflow.sha256(npz),
                    }
                    if split == "train":
                        expected_train.append(manifest.resolve())
                    elif split == "validation":
                        expected_validation.append(manifest.resolve())
                report = campaign / f"round-{round_index}/shards/pack-report.json"
                write_json(report, {"shards": shard_records})
                write_json(
                    campaign / f"round-{round_index}/workflow.json",
                    {"artifacts": {"pack_report": {"report": str(report)}}},
                )
            model = campaign / "round-2/model/jacek_replay_bfm.runtime.json"
            write_json(model, {"source_shards": source_shards})

            train, validation = workflow._canonical_anchor_manifests(campaign)
            self.assertEqual(train, tuple(expected_train))
            self.assertEqual(validation, tuple(expected_validation))

            stale = json.loads(model.read_text())
            stale["source_shards"] = stale["source_shards"][3:]
            write_json(model, stale)
            with self.assertRaisesRegex(ValueError, "ancestry"):
                workflow._canonical_anchor_manifests(campaign)

    def test_incumbent_worst_color_tiebreak_uses_aggregate_color_totals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            campaign = root / "canonical"
            for round_index, huber in enumerate((0.01, 0.02, 0.03)):
                model = campaign / f"round-{round_index}/model"
                model.mkdir(parents=True)
                (model / "jacek_replay_bfm.runtime").write_bytes(
                    f"round-{round_index}".encode()
                )
                write_json(
                    model / "jacek_replay_bfm.runtime.json",
                    {
                        "training": {
                            "chosen_seed": round_index,
                            "seed_reports": [{
                                "seed": round_index,
                                "validation": {"weighted_huber": huber},
                            }],
                        }
                    },
                )

            def matchup(wins: int, color_zero: int) -> dict:
                return {
                    "lower_round_as_candidate": {
                        "games": 1_000,
                        "wins": wins,
                        "unfinished": 0,
                        "illegal": 0,
                        "opening_sweeps": 100,
                        "opening_losses": 100,
                        "colors": {
                            "0": {"wins": color_zero},
                            "1": {"wins": wins - color_zero},
                        },
                    }
                }

            evaluation = root / "summary.json"
            write_json(
                evaluation,
                {
                    "schema": "papersoccer.jacek-replay-postcampaign-summary.v1",
                    "games": 5_000,
                    "sequential_latency_audit": {"games": 20},
                    "step2": {
                        "r0-vs-r1": matchup(600, 100),
                        "r1-vs-r2": matchup(600, 400),
                        "r0-vs-r2": matchup(400, 200),
                    },
                },
            )
            selected = workflow.select_incumbent(evaluation, campaign)

        self.assertEqual(selected["incumbent"]["round"], 1)
        self.assertEqual(selected["incumbent"]["color_wins"], [400, 600])
        self.assertEqual(selected["incumbent"]["worst_color_wins"], 400)

    def test_positions_are_suffix_only_and_inherit_root_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            games = root / "games.tsv"
            games.write_text(
                "group_id\tsource\twinner\ttranscript\n"
                f"root:one\tpilot\t0\t{SHORT_WIN}\n"
            )
            game_manifest = root / "games.json"
            write_json(
                game_manifest,
                {
                    "schema": workflow.GAME_MANIFEST_SCHEMA,
                    "campaign_id": workflow.PILOT_CAMPAIGN_ID,
                    "requested_games": 1,
                    "successful_games": 1,
                    "bindings": {"output_sha256": workflow.sha256(games)},
                    "rows": [
                        {
                            "row_ordinal": 0,
                            "game_id": "game:one",
                            "root_group_id": "root:one",
                            "prefix_turns": 1,
                            "transcript_sha256": hashlib.sha256(
                                SHORT_WIN.encode()
                            ).hexdigest(),
                        }
                    ],
                },
            )
            roots = root / "roots.json"
            write_json(
                roots,
                {"accepted": [{"group_id": "root:one", "split": "validation"}]},
            )
            payload, manifest = workflow.freeze_positions(
                campaign_id=workflow.PILOT_CAMPAIGN_ID,
                games_tsv=games,
                games_manifest=game_manifest,
                roots_manifest=roots,
                maximum_per_game=4,
            )
        lines = payload.decode().splitlines()
        self.assertEqual(len(lines), 5)
        rows = [line.split("\t") for line in lines[1:]]
        self.assertTrue(all(row[4] == "validation" for row in rows))
        self.assertTrue(all(len(row[7].split("/")) >= 1 for row in rows))
        self.assertEqual(manifest["split_counts"], {"validation": 4})

    def test_position_sampler_keeps_exact_quarters_only(self):
        self.assertEqual(workflow._sample_suffix_boundaries(4, 1, 24), [])
        selected = workflow._sample_suffix_boundaries(12, 1, 9)
        self.assertEqual(len(selected), 8)
        self.assertEqual(len(selected) % 4, 0)

    def test_adjudicator_excludes_canonical_overlap_with_current_train(self):
        with tempfile.TemporaryDirectory() as directory:
            positions = pathlib.Path(directory) / "positions.tsv"
            positions.write_text(
                "position_id\troot_group_id\tgroup_id\tsource\tsplit\twinner\tmover\tprefix\n"
                "train-state\tr0\tg0\ts\ttrain\t0\t0\t\n"
                "validation-collision\tr1\tg1\ts\tvalidation\t0\t0\t\n"
                "validation-unique\tr2\tg2\ts\tvalidation\t0\t1\t0\n"
            )
            payload = workflow.common_adjudicator_positions(positions, 1)
        self.assertIn("validation-unique", payload.decode())
        self.assertNotIn("validation-collision", payload.decode())

    def test_hard_selection_and_deep_merge_are_paired(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            positions = root / "positions.tsv"
            positions.write_text(
                "position_id\troot_group_id\tgroup_id\tsource\tsplit\twinner\tmover\tprefix\n"
                + "\n".join(
                    f"p{i}\tr\tg\ts\ttrain\t0\t0\t" for i in range(4)
                )
                + "\n"
            )
            search = root / "search.jsonl"
            rank4 = root / "rank4.jsonl"
            search_rows = []
            rank4_rows = []
            for index, value in enumerate((0.8, 0.1, -0.2, 0.4)):
                search_rows.append(search_teacher_row(f"p{index}", value))
                rank4_rows.append(
                    rank4_teacher_row(
                        f"p{index}", (-1 if index == 0 else 1) * 1_000
                    )
                )
            search.write_text("".join(json.dumps(row) + "\n" for row in search_rows))
            rank4.write_text("".join(json.dumps(row) + "\n" for row in rank4_rows))
            payload, manifest = workflow.select_hard_positions(
                positions_tsv=positions,
                search_labels=search,
                rank4_labels=rank4,
            )
            hard_id = payload.decode().splitlines()[1].split("\t")[0]
            deep = root / "deep.jsonl"
            replacement = next(row for row in search_rows if row["position_id"] == hard_id)
            replacement = {**replacement, "teacher_value": 0.9}
            deep.write_text(json.dumps(replacement) + "\n")
            merged = workflow.merge_deep_labels(
                shallow=search,
                deep=deep,
                expected_schema=workflow.SEARCH_TEACHER_SCHEMA,
            )
        self.assertEqual(manifest["selected"], 1)
        self.assertEqual(len(merged.decode().splitlines()), 4)
        self.assertIn('"teacher_value":0.9', merged.decode())

    def test_hard_selection_rejects_search_rows_without_completion_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            positions = root / "positions.tsv"
            positions.write_text(
                "position_id\troot_group_id\tgroup_id\tsource\tsplit\twinner\tmover\tprefix\n"
                + "".join(f"p{i}\tr\tg\ts\ttrain\t0\t0\t\n" for i in range(4))
            )
            search = root / "search.jsonl"
            invalid = search_teacher_row("p0", 0.2)
            invalid.pop("search_stats")
            rows = [invalid] + [
                search_teacher_row(f"p{i}", 0.2) for i in range(1, 4)
            ]
            search.write_text("".join(json.dumps(row) + "\n" for row in rows))
            rank4 = root / "rank4.jsonl"
            rank4.write_text(
                "".join(
                    json.dumps(rank4_teacher_row(f"p{i}"))
                    + "\n"
                    for i in range(4)
                )
            )
            with self.assertRaisesRegex(ValueError, "teacher contract"):
                workflow.select_hard_positions(
                    positions_tsv=positions,
                    search_labels=search,
                    rank4_labels=rank4,
                )

    def test_rank4_loader_rejects_rows_without_completion_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            labels = pathlib.Path(directory) / "rank4.jsonl"
            invalid = rank4_teacher_row("p0")
            invalid.pop("search_stats")
            labels.write_text(json.dumps(invalid) + "\n")
            with self.assertRaisesRegex(ValueError, "teacher contract"):
                workflow.load_labels(labels, workflow.RANK4_TEACHER_SCHEMA)

    def test_label_validation_binds_source_and_exact_frozen_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            positions = root / "positions.tsv"
            positions.write_text(
                "position_id\troot_group_id\tgroup_id\tsource\tsplit\twinner\tmover\tprefix\n"
                "p0\tr\tg\ts\ttrain\t0\t0\t\n"
            )
            row = search_teacher_row("p0", 0.2)
            row["search_config"]["seed"] = int.from_bytes(
                hashlib.sha256(
                    b"selfsearch-workflow-fixture\0p0\0" + b"64"
                ).digest()[:8],
                "big",
            )
            labels = root / "labels.jsonl"

            def validate(candidate: dict) -> int:
                labels.write_bytes(workflow.canonical_json_bytes(candidate))
                return workflow._validate_label_output(
                    output=labels,
                    positions=positions,
                    schema=workflow.SEARCH_TEACHER_SCHEMA,
                    campaign_id="selfsearch-workflow-fixture",
                    nodes=64,
                    model_sha256="2" * 64,
                    source_sha256="1" * 64,
                )

            self.assertEqual(validate(row), 1)
            with self.assertRaisesRegex(ValueError, "lineage is stale"):
                validate({**row, "source": "different-source"})
            with self.assertRaisesRegex(ValueError, "lineage is stale"):
                validate(
                    {
                        **row,
                        "prefix": [
                            {"player_id": 0, "action": "0"},
                            {"player_id": 1, "action": "0"},
                        ],
                    }
                )

    def test_game_chunk_merge_binds_each_frozen_plan_row_and_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            modes = ("incumbent-selfplay", "incumbent-p1-vs-rank4")
            plan = root / "plan.json"
            write_json(
                plan,
                {
                    "schema": workflow.GAME_PLAN_SCHEMA,
                    "campaign_id": workflow.PILOT_CAMPAIGN_ID,
                    "games": 2,
                    "quotas": {mode: 1 for mode in modes},
                    "rows": [
                        {
                            "game_ordinal": ordinal,
                            "actor_mode": mode,
                            "base_seed": 100 + ordinal,
                        }
                        for ordinal, mode in enumerate(modes)
                    ],
                },
            )
            tsvs, manifests = [], []
            source_fields = {
                "producer_source_sha256": "1" * 64,
                "rank4_actor_source_sha256": "2" * 64,
                "jacek_nn_actor_source_sha256": "3" * 64,
            }
            for ordinal, mode in enumerate(modes):
                tsv = root / f"chunk-{ordinal}.tsv"
                tsv.write_text(
                    "group_id\tsource\twinner\ttranscript\n"
                    f"root:{ordinal}\t{workflow.PILOT_CAMPAIGN_ID}\t0\t{SHORT_WIN}\n"
                )
                manifest = root / f"chunk-{ordinal}.json"
                write_json(
                    manifest,
                    {
                        "schema": workflow.GAME_MANIFEST_SCHEMA,
                        "campaign_id": workflow.PILOT_CAMPAIGN_ID,
                        "requested_games": 1,
                        "successful_games": 1,
                        "configuration": {**source_fields, "bfm_tree_nodes": 8_000},
                        "bindings": {
                            **source_fields,
                            "incumbent_model_sha256": "4" * 64,
                            "runner_up_model_sha256": "5" * 64,
                            "roots_sha256": "6" * 64,
                            "output_sha256": workflow.sha256(tsv),
                        },
                        "rows": [
                            {
                                "row_ordinal": 0,
                                "game_ordinal": ordinal,
                                "actor_mode": mode,
                                "base_seed": 100 + ordinal,
                                "game_id": f"game:{ordinal}",
                                "root_group_id": f"root:{ordinal}",
                                "winner": 0,
                                "transcript_sha256": hashlib.sha256(
                                    SHORT_WIN.encode()
                                ).hexdigest(),
                            }
                        ],
                    },
                )
                tsvs.append(tsv)
                manifests.append(manifest)

            payload, merged = workflow.merge_game_chunks(
                campaign_id=workflow.PILOT_CAMPAIGN_ID,
                plan_path=plan,
                chunk_tsvs=tsvs,
                chunk_manifests=manifests,
            )
            self.assertEqual(merged["successful_games"], 2)
            self.assertEqual(merged["bindings"]["producer_source_sha256"], "1" * 64)
            self.assertEqual(len(payload.decode().splitlines()), 3)

            first = json.loads(manifests[0].read_text())
            first["rows"][0]["actor_mode"] = modes[1]
            write_json(manifests[0], first)
            with self.assertRaisesRegex(ValueError, "frozen plan"):
                workflow.merge_game_chunks(
                    campaign_id=workflow.PILOT_CAMPAIGN_ID,
                    plan_path=plan,
                    chunk_tsvs=tsvs,
                    chunk_manifests=manifests,
                )
            first["rows"][0]["actor_mode"] = modes[0]
            first["bindings"]["producer_source_sha256"] = "7" * 64
            write_json(manifests[0], first)
            with self.assertRaisesRegex(ValueError, "producer_source_sha256"):
                workflow.merge_game_chunks(
                    campaign_id=workflow.PILOT_CAMPAIGN_ID,
                    plan_path=plan,
                    chunk_tsvs=tsvs,
                    chunk_manifests=manifests,
                )

    def test_pilot_gate_requires_both_primary_models_and_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)

            def report(path, opponent, wins):
                rows = []
                for index in range(600):
                    color = index % 2
                    won = index < wins
                    rows.append(
                        {
                            "opening": f"o{index // 2}",
                            "opponent": opponent,
                            "candidate_player": color,
                            "winner": color if won else 1 - color,
                            "illegal": False,
                            "candidate_ms": [20.0],
                        }
                    )
                write_json(
                    path,
                    {
                        "schema": "papersoccer.jacek-replay-bfm-comparison.v1",
                        "configuration": {
                            "pairs": 300,
                            "time_ms": 20,
                            "exploration": 0.5,
                            "fpu": 0.5,
                            "opponent": opponent,
                        },
                        "results": rows,
                    },
                )

            matched = root / "matched.json"
            incumbent = root / "incumbent.json"
            rank4 = root / "rank4.json"
            jacek = root / "jacek.json"
            report(matched, "jacek-replay", 330)
            report(incumbent, "jacek-replay", 330)
            report(rank4, "rank4", 310)
            report(jacek, "jacek-nn", 310)
            with self.assertRaisesRegex(ValueError, "configuration"):
                workflow.validate_gate_report(
                    matched,
                    pairs=300,
                    opponent="jacek-replay",
                    bank_classification="final",
                )
            decision = workflow.pilot_decision(
                matched_report=matched,
                incumbent_report=incumbent,
                rank4_report=rank4,
                jacek_nn_report=jacek,
                anchor_candidate={"sign_accuracy": 0.86, "weighted_huber": 0.05},
                anchor_incumbent={"sign_accuracy": 0.86, "weighted_huber": 0.05},
                uncontended_max_ms=990.0,
            )
        self.assertTrue(decision["eligible_for_full"])

    def test_final_gate_accepts_exact_total_and_color_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)

            def report(path: pathlib.Path, opponent: str,
                       color_wins: tuple[int, int]) -> None:
                rows = []
                for color in (0, 1):
                    for index in range(500):
                        won = index < color_wins[color]
                        rows.append({
                            "opening": f"o{index}",
                            "opponent": opponent,
                            "candidate_player": color,
                            "winner": color if won else 1 - color,
                            "illegal": False,
                            "candidate_ms": [980.0],
                        })
                write_json(path, {
                    "schema": "papersoccer.jacek-replay-bfm-comparison.v1",
                    "configuration": {
                        "pairs": 500, "time_ms": 980,
                        "exploration": 0.5, "fpu": 0.5,
                        "opponent": opponent,
                    },
                    "results": rows,
                })

            pilot = root / "pilot.json"
            matched = root / "matched.json"
            rank4 = root / "rank4.json"
            jacek = root / "jacek.json"
            report(pilot, "jacek-replay", (260, 267))
            report(matched, "jacek-replay", (260, 267))
            report(rank4, "rank4", (238, 263))
            report(jacek, "jacek-nn", (238, 263))
            decision = workflow.final_decision(
                pilot_report=pilot,
                matched_report=matched,
                rank4_report=rank4,
                jacek_nn_report=jacek,
                anchor_candidate={
                    "sign_accuracy": 0.855,
                    "weighted_huber": 0.051,
                },
                anchor_incumbent={
                    "sign_accuracy": 0.86,
                    "weighted_huber": 0.05,
                },
                original_anchor_candidate={
                    "sign_accuracy": 0.855,
                    "weighted_huber": 0.051,
                },
                original_anchor_incumbent={
                    "sign_accuracy": 0.86,
                    "weighted_huber": 0.05,
                },
                uncontended_max_ms=999.999,
            )
            self.assertTrue(decision["eligible_for_local_publication"])
            self.assertEqual(decision["anchor_candidate"]["sign_accuracy"], 0.855)

            report(matched, "jacek-replay", (260, 266))
            rejected = workflow.final_decision(
                pilot_report=pilot,
                matched_report=matched,
                rank4_report=rank4,
                jacek_nn_report=jacek,
                anchor_candidate={
                    "sign_accuracy": 0.855,
                    "weighted_huber": 0.051,
                },
                anchor_incumbent={
                    "sign_accuracy": 0.86,
                    "weighted_huber": 0.05,
                },
                original_anchor_candidate={
                    "sign_accuracy": 0.855,
                    "weighted_huber": 0.051,
                },
                original_anchor_incumbent={
                    "sign_accuracy": 0.86,
                    "weighted_huber": 0.05,
                },
                uncontended_max_ms=999.999,
            )
            self.assertFalse(rejected["eligible_for_local_publication"])

            report(matched, "jacek-replay", (260, 267))
            sign_rejected = workflow.final_decision(
                pilot_report=pilot,
                matched_report=matched,
                rank4_report=rank4,
                jacek_nn_report=jacek,
                anchor_candidate={
                    "sign_accuracy": 0.854999,
                    "weighted_huber": 0.051,
                },
                anchor_incumbent={
                    "sign_accuracy": 0.86,
                    "weighted_huber": 0.05,
                },
                original_anchor_candidate={
                    "sign_accuracy": 0.855,
                    "weighted_huber": 0.051,
                },
                original_anchor_incumbent={
                    "sign_accuracy": 0.86,
                    "weighted_huber": 0.05,
                },
                uncontended_max_ms=999.999,
            )
            self.assertIn(
                "canonical anchor sign noninferiority failed",
                sign_rejected["errors"],
            )

            huber_rejected = workflow.final_decision(
                pilot_report=pilot,
                matched_report=matched,
                rank4_report=rank4,
                jacek_nn_report=jacek,
                anchor_candidate={
                    "sign_accuracy": 0.855,
                    "weighted_huber": 0.051001,
                },
                anchor_incumbent={
                    "sign_accuracy": 0.86,
                    "weighted_huber": 0.05,
                },
                original_anchor_candidate={
                    "sign_accuracy": 0.855,
                    "weighted_huber": 0.051,
                },
                original_anchor_incumbent={
                    "sign_accuracy": 0.86,
                    "weighted_huber": 0.05,
                },
                uncontended_max_ms=999.999,
            )
            self.assertIn(
                "canonical anchor Huber noninferiority failed",
                huber_rejected["errors"],
            )

            original_rejected = workflow.final_decision(
                pilot_report=pilot,
                matched_report=matched,
                rank4_report=rank4,
                jacek_nn_report=jacek,
                anchor_candidate={
                    "sign_accuracy": 0.855,
                    "weighted_huber": 0.051,
                },
                anchor_incumbent={
                    "sign_accuracy": 0.86,
                    "weighted_huber": 0.05,
                },
                original_anchor_candidate={
                    "sign_accuracy": 0.854999,
                    "weighted_huber": 0.051001,
                },
                original_anchor_incumbent={
                    "sign_accuracy": 0.86,
                    "weighted_huber": 0.05,
                },
                uncontended_max_ms=999.999,
            )
            self.assertIn(
                "original incumbent anchor sign noninferiority failed",
                original_rejected["errors"],
            )
            self.assertIn(
                "original incumbent anchor Huber noninferiority failed",
                original_rejected["errors"],
            )

    def test_gate_panels_are_globally_sharded_resumable_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            model = root / "candidate.runtime"
            bank = root / "bank.tsv"
            model.write_bytes(b"candidate")
            bank.write_text("fixture-bank\n")
            counter = root / "calls.txt"
            comparison = root / "fake-comparison.py"
            comparison.write_text(textwrap.dedent(f"""\
                #!/usr/bin/env python3
                import argparse, hashlib, json, pathlib
                parser = argparse.ArgumentParser()
                parser.add_argument('--model', type=pathlib.Path, required=True)
                parser.add_argument('--output', type=pathlib.Path, required=True)
                parser.add_argument('--bank', type=pathlib.Path, required=True)
                parser.add_argument('--bank-classification', required=True)
                parser.add_argument('--opponent', required=True)
                parser.add_argument('--pairs', type=int, required=True)
                parser.add_argument('--pair-offset', type=int, required=True)
                parser.add_argument('--time-ms', type=int, required=True)
                parser.add_argument('--control-model', type=pathlib.Path)
                args, _ = parser.parse_known_args()
                def digest(path):
                    return hashlib.sha256(path.read_bytes()).hexdigest()
                with open({str(counter)!r}, 'a', encoding='utf-8') as calls:
                    calls.write('call\\n')
                identities = {{name: '1' * 64 for name in (
                    'rank4_control_sha256', 'rank4_engine_sha256',
                    'neural_puct_control_sha256', 'neural_puct_engine_sha256',
                    'jacek_nn_control_sha256', 'jacek_nn_engine_sha256',
                    'rank4_adapter_sha256', 'neural_puct_adapter_sha256',
                    'jacek_nn_adapter_sha256', 'jacek_nn_source_sha256',
                    'shared_core_sha256', 'candidate_source_sha256',
                    'comparison_source_sha256')}}
                results = []
                for opening in range(args.pair_offset,
                                     args.pair_offset + args.pairs):
                    for color in (0, 1):
                        results.append({{
                            'opening': f'o{{opening}}',
                            'opponent': args.opponent,
                            'candidate_player': color,
                            'winner': color,
                            'illegal': False,
                            'candidate_ms': [float(args.time_ms)],
                            'control_ms': [float(args.time_ms)],
                        }})
                configuration = {{
                    'pairs': args.pairs, 'pair_offset': args.pair_offset,
                    'opening_bank_sha256': digest(args.bank),
                    'opening_state_identities': [
                        f'state-{{index}}' for index in range(
                            args.pair_offset, args.pair_offset + args.pairs)],
                    'opening_bank_classification': args.bank_classification,
                    'opening_plies': 12,
                    'control_model_sha256': digest(args.control_model)
                        if args.control_model else None,
                    'candidate_tree_nodes': 1000000,
                    'control_tree_nodes': 1000000,
                    'control_work': 3000000,
                    'max_actions': 250, 'max_partial_paths': 50000,
                    'max_turns': 320, 'seed': 20919592877381169,
                    'time_ms': args.time_ms, 'exploration': 0.5, 'fpu': 0.5,
                    'opponent': args.opponent, 'single_thread': True,
                    'comparison_executable_sha256': digest(pathlib.Path(__file__)),
                    **identities,
                }}
                report = {{
                    'schema': 'papersoccer.jacek-replay-bfm-comparison.v1',
                    'model_sha256': digest(args.model),
                    'configuration': configuration,
                    'summary': {{'games': len(results)}},
                    'results': results,
                }}
                args.output.write_text(json.dumps(report))
                """))
            comparison.chmod(0o755)
            source_identities = {name: "1" * 64 for name in (
                "rank4_control_sha256", "rank4_engine_sha256",
                "neural_puct_control_sha256", "neural_puct_engine_sha256",
                "jacek_nn_control_sha256", "jacek_nn_engine_sha256",
                "rank4_adapter_sha256", "neural_puct_adapter_sha256",
                "jacek_nn_adapter_sha256", "jacek_nn_source_sha256",
                "shared_core_sha256", "candidate_source_sha256",
                "comparison_source_sha256",
            )}
            output = root / "attempt"
            output.mkdir()
            manager = workflow.StageManager(
                output=output, campaign_id=workflow.PILOT_CAMPAIGN_ID,
                round_index=0, resume=False, environment={"fixture": True},
            )
            panels = [workflow.Panel("rank4", "rank4"),
                      workflow.Panel("jacek-nn", "jacek-nn")]
            result = workflow.run_comparison_panels(
                manager=manager, stage_ordinal=19, comparison=comparison,
                model=model, bank=bank, panels=panels, pairs=10,
                time_ms=20, workers=4, classification="development",
                source_identities=source_identities, shard_pairs=5,
            )
            self.assertEqual(len(result["panel_receipts"]), 4)
            self.assertEqual(counter.read_text().splitlines(), ["call"] * 4)
            self.assertEqual(
                len(json.loads(pathlib.Path(result["reports"]["rank4"]).read_text())["results"]),
                20,
            )

            resumed = workflow.StageManager(
                output=output, campaign_id=workflow.PILOT_CAMPAIGN_ID,
                round_index=0, resume=True, environment={"fixture": True},
            )
            workflow.run_comparison_panels(
                manager=resumed, stage_ordinal=19, comparison=comparison,
                model=model, bank=bank, panels=panels, pairs=10,
                time_ms=20, workers=4, classification="development",
                source_identities=source_identities, shard_pairs=5,
            )
            self.assertEqual(counter.read_text().splitlines(), ["call"] * 4)
            shard = output / "game-gates/shards/rank4/offset-0000.json"
            shard.write_bytes(shard.read_bytes() + b"corrupt\n")
            with self.assertRaisesRegex(ValueError, "stale"):
                workflow.run_comparison_panels(
                    manager=resumed, stage_ordinal=19, comparison=comparison,
                    model=model, bank=bank, panels=panels, pairs=10,
                    time_ms=20, workers=4, classification="development",
                    source_identities=source_identities, shard_pairs=5,
                )


if __name__ == "__main__":
    unittest.main()
