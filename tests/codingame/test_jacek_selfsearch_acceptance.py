"""Tiny acceptance campaign for the self-search receipt graph.

The expensive C++ actors, fixed-work teachers, trainer, and comparisons are
replaced by deterministic fixture producers.  The production phase runner,
GuardedStageManager, corpus packer, validators, target construction, stage
ordinals, nested evidence, and automatic gate branches remain real.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
import sys
import tempfile
import textwrap
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import jacek_replay_corpus as corpus  # noqa: E402
import jacek_replay_features as features  # noqa: E402
import jacek_selfsearch_workflow as workflow  # noqa: E402


SHORT_WIN = "0/0/3/0/61/0/07"
PILOT_VALIDATION_WIN = (
    "6/7/4/1/2/44/4/7/75/4/3/4/2/42/21/7/0/5/7/25/66/1/"
    "200357/06/4527236436/0530727"
)
FULL_TRAIN_WIN = (
    "2/4/70/2/0/3/657/6/4/7/1/0/3/0/7/46/52/53/22/4/16001/"
    "661/31/3/50/5/256723033/2/0/35/2717/6/674702/27/47574/43646"
)
FULL_VALIDATION_WIN = (
    "0/0/0/2/7/16/3/42/4/1/4/64/4/6/4/77/7/5/3/4/4/5/3/0/7/0/"
    "2553/30611/1/023/1/177714/77"
)

SOURCE_IDENTITIES = {
    "continuation_source_sha256": "1" * 64,
    "rank4_actor_source_sha256": "2" * 64,
    "jacek_nn_actor_source_sha256": "3" * 64,
    "search_teacher_source_sha256": "4" * 64,
    "rank4_teacher_source_sha256": "5" * 64,
}


def write_json(path: pathlib.Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(workflow.canonical_json_bytes(value, pretty=True))


def valid_roots_manifest(path: pathlib.Path) -> None:
    body = {
        "schema": corpus.ROOT_SCHEMA,
        "feature_schema": features.FEATURE_SCHEMA,
        "tool_sha256": {"normalizer": "a" * 64, "features": "b" * 64},
        "exclusion_boundary": {"read_before_candidate_sources": True},
        "accepted": [
            {"group_id": "root:train", "split": "train"},
            {"group_id": "root:validation", "split": "validation"},
        ],
    }
    write_json(
        path,
        {
            **body,
            "body_sha256": hashlib.sha256(
                corpus.canonical_json_bytes(body)
            ).hexdigest(),
        },
    )


class TinyAcceptanceFixture:
    def __init__(self, root: pathlib.Path) -> None:
        self.root = root
        self.build_manifest = root / "release-build.json"
        write_json(self.build_manifest, {"fixture": "frozen-release-build"})
        self.roots_tsv = root / "roots.tsv"
        self.roots_tsv.write_text(
            "group_id\tsource\twinner\ttranscript\n"
            f"root:train\tfixture\t0\t{SHORT_WIN}\n"
            f"root:validation\tfixture\t0\t{PILOT_VALIDATION_WIN}\n"
        )
        self.roots_manifest = root / "roots.json"
        valid_roots_manifest(self.roots_manifest)
        self.bootstrap = ROOT / "models/jacek_replay_bfm_bootstrap.runtime"
        if not self.bootstrap.is_file():
            raise RuntimeError("bootstrap runtime fixture is missing")
        self.original_actor = root / "original-incumbent.runtime"
        self.original_runner = root / "original-runner.runtime"
        shutil.copy2(self.bootstrap, self.original_actor)
        shutil.copy2(self.bootstrap, self.original_runner)
        self.actor = root / "auto/inputs/incumbent.runtime"
        self.runner = root / "auto/inputs/runner-up.runtime"
        self.anchor_train = root / "anchor-train.json"
        self.anchor_validation = root / "anchor-validation.json"
        write_json(self.anchor_train, {"split": "train", "fixture": True})
        write_json(
            self.anchor_validation, {"split": "validation", "fixture": True}
        )
        self.generator = root / "fake-generator.py"
        self.search_teacher = root / "fake-search-teacher.py"
        self.rank4_teacher = root / "fake-rank4-teacher.py"
        self.comparison = root / "fake-comparison"
        self._write_generator()
        self._write_teacher(self.search_teacher, search=True)
        self._write_teacher(self.rank4_teacher, search=False)
        self.comparison.write_text("fixture comparison executable\n")
        self.comparison.chmod(0o755)
        self.executables = workflow.CampaignExecutables(
            continuation_generator=self.generator,
            search_teacher=self.search_teacher,
            rank4_teacher=self.rank4_teacher,
            comparison=self.comparison,
            pack_tool=ROOT / "tools/jacek_replay_pack.py",
            trainer=ROOT / "tools/jacek_replay_train.py",
        )
        self.producer_guard_calls = 0

    def producer_guard(self) -> None:
        self.producer_guard_calls += 1
        for path in (
            self.build_manifest,
            self.generator,
            self.search_teacher,
            self.rank4_teacher,
            self.comparison,
        ):
            if not path.is_file():
                raise ValueError("fixture producer disappeared")

    def _write_generator(self) -> None:
        self.generator.write_text(
            textwrap.dedent(
                f"""\
                #!{sys.executable}
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
                digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
                plans = [line.split('\\t') for line in
                         args.selfsearch_plan.read_text().splitlines()[1:]]
                pilot = 'pilot' in args.campaign_id
                transcripts = (
                    ({SHORT_WIN!r}, {PILOT_VALIDATION_WIN!r}) if pilot else
                    ({FULL_TRAIN_WIN!r}, {FULL_VALIDATION_WIN!r})
                )
                output = ['group_id\\tsource\\twinner\\ttranscript']
                rows = []
                for row_ordinal, (ordinal_raw, mode, seed_raw) in enumerate(plans):
                    ordinal, seed = int(ordinal_raw), int(seed_raw)
                    root = 'root:train' if ordinal % 2 == 0 else 'root:validation'
                    transcript = transcripts[ordinal % 2]
                    output.append(f'{{root}}\\t{{args.campaign_id}}\\t0\\t{{transcript}}')
                    rows.append({{
                        'game_id': f'{{args.campaign_id}}:game:{{ordinal}}',
                        'row_ordinal': row_ordinal, 'game_ordinal': ordinal,
                        'attempt_ordinal': 0, 'base_seed': seed, 'game_seed': seed,
                        'actor_mode': mode, 'root_group_id': root,
                        'prefix_turns': 1, 'winner': 0,
                        'transcript_sha256': hashlib.sha256(
                            transcript.encode()).hexdigest(),
                    }})
                args.output.write_text('\\n'.join(output) + '\\n')
                sources = {{
                    'producer_source_sha256': {SOURCE_IDENTITIES['continuation_source_sha256']!r},
                    'rank4_actor_source_sha256': {SOURCE_IDENTITIES['rank4_actor_source_sha256']!r},
                    'jacek_nn_actor_source_sha256': {SOURCE_IDENTITIES['jacek_nn_actor_source_sha256']!r},
                }}
                configuration = {{
                    'bfm_tree_nodes': args.candidate_tree_nodes,
                    'rank4_nodes': args.actor_nodes,
                    'jacek_nn_nodes': args.jacek_nn_nodes,
                    'exploration': args.candidate_exploration,
                    'fpu': args.candidate_fpu,
                    'early_exploration_percent': 15,
                    'early_exploration_turns': 8,
                    'maximum_turns': 320, **sources,
                }}
                manifest = {{
                    'schema': {workflow.GAME_MANIFEST_SCHEMA!r},
                    'campaign_id': args.campaign_id,
                    'requested_games': args.games, 'successful_games': args.games,
                    'configuration': configuration,
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
                args.manifest.write_text(json.dumps(manifest, sort_keys=True))
                """
            )
        )
        self.generator.chmod(0o755)

    def _write_teacher(self, path: pathlib.Path, *, search: bool) -> None:
        source_hash = SOURCE_IDENTITIES[
            "search_teacher_source_sha256" if search else "rank4_teacher_source_sha256"
        ]
        path.write_text(
            textwrap.dedent(
                f"""\
                #!{sys.executable}
                import argparse, hashlib, json, sys
                parser = argparse.ArgumentParser()
                parser.add_argument('--campaign-id', required=True)
                parser.add_argument('--model')
                parser.add_argument('--model-sha256')
                parser.add_argument('--tree-nodes', type=int)
                parser.add_argument('--nodes', type=int)
                parser.add_argument('--time-ms', type=int, required=True)
                parser.add_argument('--max-actions', type=int)
                parser.add_argument('--max-partial-paths', type=int)
                parser.add_argument('--exploration', type=float)
                parser.add_argument('--fpu', type=float)
                args = parser.parse_args()
                nodes = args.tree_nodes if args.tree_nodes is not None else args.nodes
                lines = sys.stdin.read().splitlines()
                for line in lines[1:]:
                    fields = line.split('\\t')
                    actions = fields[7].split('/') if fields[7] else []
                    prefix = [{{'player_id': i % 2, 'action': action}}
                              for i, action in enumerate(actions)]
                    common = {{
                        'campaign_id': args.campaign_id,
                        'position_id': fields[0], 'root_group_id': fields[1],
                        'group_id': fields[2], 'source': fields[3],
                        'split': fields[4], 'winner': int(fields[5]),
                        'mover': int(fields[6]), 'prefix': prefix,
                        'root_solved': False, 'proven_winner': None, 'weight': 1.0,
                    }}
                    if {search!r}:
                        seed = int.from_bytes(hashlib.sha256(
                            f'{{args.campaign_id}}\\0{{fields[0]}}\\0{{nodes}}'.encode()
                        ).digest()[:8], 'big')
                        row = {{
                            **common, 'schema': {workflow.SEARCH_TEACHER_SCHEMA!r},
                            'teacher': {{
                                'kind': 'jacek_replay_bfm_search',
                                'source_sha256': {source_hash!r},
                                'model_sha256': args.model_sha256,
                                'feature_schema': {features.FEATURE_SCHEMA!r},
                                'feature_schema_sha256': hashlib.sha256(
                                    {features.FEATURE_SCHEMA!r}.encode()).hexdigest(),
                            }},
                            'search_config': {{
                                'seed': seed, 'max_time_ms': args.time_ms,
                                'max_tree_nodes': nodes,
                                'max_actions': args.max_actions,
                                'max_partial_paths': args.max_partial_paths,
                                'exploration': args.exploration, 'fpu': args.fpu,
                            }},
                            'search_stats': {{
                                'expansions': 1, 'generated_actions': 1,
                                'retained_actions': 1, 'neural_evaluations': 1,
                                'visits': nodes, 'completed_actions': 1,
                                'duplicate_boundaries': 0, 'partial_paths': 1,
                                'fifo_extractions': 0, 'lifo_extractions': 1,
                                'tactical_proofs': 0, 'tactical_solutions': 0,
                                'truncations': 0, 'tree_nodes': nodes,
                                'generation_action_cap_stops': 0,
                                'generation_partial_cap_stops': 0,
                                'generation_deadline_stops': 0,
                                'materialization_deadline_stops': 0,
                                'generation_queue_drops': 0,
                                'generation_retention_drops': 0,
                                'generation_boundary_replacements': 0,
                                'generation_tactical_shortcuts': 0,
                                'generation_fallbacks': 0,
                                'generation_frontier_resumptions': 0,
                                'generation_zero_action_resumptions': 0,
                                'generation_max_frontier_depth': 1,
                                'progressive_widenings': 0,
                                'closed_unsolved_nodes': 0,
                                'closed_unsolved_nonexhaustive_nodes': 0,
                                'open_unexpanded_nodes': 1,
                                'implicit_action_frontiers': 0,
                                'max_open_children': 1,
                                'max_complete_turn_depth': 1,
                                'deadline_reached': False, 'tree_cap_reached': True,
                                'termination_reason': 'fixed-work-cap',
                            }},
                            'teacher_value': 0.2,
                        }}
                    else:
                        stats = {{
                            'attempted_depth': 2, 'completed_depth': 1,
                            'nodes': nodes, 'leaf_evaluations': 1,
                            'terminal_nodes': 0, 'completed_actions': 1,
                            'budget_exhausted': True, 'node_cap_reached': True,
                            'deadline_reached': False,
                        }}
                        row = {{
                            **common, 'schema': {workflow.RANK4_TEACHER_SCHEMA!r},
                            'teacher': {{'kind': 'rank4-fixed-work',
                                        'source_sha256': {source_hash!r}}},
                            'root_score': 100, 'completed_depth': 1,
                            'nodes': nodes,
                            'search_config': {{
                                'max_nodes': nodes, 'max_time_ms': args.time_ms,
                                'max_turn_depth': 32,
                                'replay_value_blend_percent': 15,
                                'teacher_residual_weight_percent': 100,
                            }},
                            'search_stats': stats,
                        }}
                    print(json.dumps(row, separators=(',', ':'), sort_keys=True))
                """
            )
        )
        path.chmod(0o755)

    def phase_spec(self, name: str) -> workflow.PhaseSpec:
        pilot = name == "pilot"
        base = workflow.PILOT_CONFIGURATION if pilot else workflow.FULL_CONFIGURATION
        configuration = {
            **base,
            "games": 4,
            "game_chunk_size": 2,
            "game_workers": 2,
            "positions_per_game": 4,
            "adjudicator_positions": 1,
            "bfm_shallow_tree_nodes": 8,
            "rank4_shallow_nodes": 8,
            "bfm_deep_tree_nodes": 16,
            "rank4_deep_nodes": 16,
            "adjudicator_tree_nodes": 32,
            "training_seeds": [101 if pilot else 102],
        }
        quotas = (
            {"incumbent-selfplay": 2, "incumbent-p1-vs-rank4": 2}
            if pilot
            else {"student-selfplay": 2, "student-p1-vs-rank4": 2}
        )
        return workflow.PhaseSpec(
            name=name,
            campaign_id=(workflow.PILOT_CAMPAIGN_ID if pilot else workflow.FULL_CAMPAIGN_ID),
            configuration=configuration,
            quotas=quotas,
            game_seed=701 if pilot else 702,
            opening_seed=703 if pilot else 704,
            pairs=300 if pilot else 500,
            gate_time_ms=20 if pilot else 980,
            gate_workers=1,
            bank_classification="development" if pilot else "final",
        )

    def fake_training_arm(
        self,
        *, new_manifests: list[pathlib.Path] | tuple[pathlib.Path, ...],
        anchor_manifests: list[pathlib.Path] | tuple[pathlib.Path, ...],
        adjudicator_manifest: pathlib.Path,
        output_directory: pathlib.Path,
        seeds: list[int] | tuple[int, ...],
        new_rows: int,
        anchor_rows: int,
        **_arguments: object,
    ) -> dict:
        output_directory.mkdir(parents=True, exist_ok=True)
        runtime = output_directory / "jacek_replay_bfm.runtime"
        shutil.copy2(self.bootstrap, runtime)
        checkpoint_directory = output_directory / "training-seeds"
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        publications = []
        for seed in seeds:
            checkpoint = checkpoint_directory / f"seed-{seed}.runtime"
            receipt = checkpoint_directory / f"seed-{seed}.json"
            shutil.copy2(self.bootstrap, checkpoint)
            write_json(
                receipt,
                {
                    "schema": "papersoccer.jacek-replay-bfm-seed-checkpoint.v1",
                    "seed": seed,
                    "checkpoint": {
                        "file": checkpoint.name,
                        "artifact_sha256": workflow.sha256(checkpoint),
                    },
                },
            )
            publications.append(
                {
                    "seed": seed,
                    "checkpoint": checkpoint.name,
                    "receipt": receipt.name,
                    "checkpoint_sha256": workflow.sha256(checkpoint),
                    "receipt_sha256": workflow.sha256(receipt),
                }
            )
        source_paths = [*new_manifests, *anchor_manifests, adjudicator_manifest]
        manifest = {
            "schema": "papersoccer.jacek-replay-bfm-model.v1",
            "runtime": {"artifact_sha256": workflow.sha256(runtime)},
            "architecture": {"dimensions": [6301, 192, 32, 1], "biases": False},
            "source_shards": [json.loads(path.read_bytes()) for path in source_paths],
            "training": {
                "seed_reports": [{"seed": seed} for seed in seeds],
                "seed_checkpoints": publications,
                "selection_validation": {"kind": "explicit-common-adjudicator"},
                "optimizer": {
                    "name": "adamw", "epochs": 50, "patience": 8,
                    "batch_size": 256, "learning_rate": 0.001,
                    "weight_decay": 1e-5, "gradient_norm_clip": 5.0,
                },
                "loss": {"name": "weighted-huber", "delta": 0.25},
                "batching": {
                    "kind": "deterministic-two-stream-cycling-v1",
                    "new_rows_per_batch": new_rows,
                    "anchor_rows_per_batch": anchor_rows,
                    "epoch_length": "new-stream-covered-once-anchor-sampled",
                    "row_order": "new-then-anchor",
                },
            },
        }
        write_json(output_directory / "jacek_replay_bfm.runtime.json", manifest)
        return manifest

    @staticmethod
    def fake_anchor_metrics(**_arguments: object) -> dict:
        metrics = {"sign_accuracy": 0.85, "weighted_huber": 0.05}
        return {
            "schema": "papersoccer.jacek-selfsearch-anchor-metrics.v1",
            "candidate_metrics": metrics,
            "incumbent_metrics": metrics,
        }

    @staticmethod
    def fake_bank(
        *, output: pathlib.Path, pairs: int, seed: int, classification: str,
        exclusions: list[pathlib.Path] | tuple[pathlib.Path, ...], **_arguments: object,
    ) -> dict:
        lines = [
            "# papersoccer.jacek-replay-bfm-opening-bank.v1",
            "# fixture=true",
            f"# classification={classification}",
            f"# seed={seed}",
            "# minimum-physical-plies=12",
            "opening_id\ttranscript\tstate_identity",
        ]
        lines.extend(
            f"opening:{index}\t0/0\t{classification}:state:{seed}:{index}"
            for index in range(pairs)
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(lines) + "\n")
        return {
            "pairs": pairs,
            "seed": seed,
            "classification": classification,
            "exclusions": [workflow.artifact_snapshot(path) for path in exclusions],
        }

    @staticmethod
    def fake_comparison_panels(
        *, manager: workflow.StageManager, panels: list[workflow.Panel],
        pairs: int, time_ms: int, classification: str,
        model: pathlib.Path, **_arguments: object,
    ) -> dict:
        pass_gate = "reject" not in manager.output.name
        reports: dict[str, str] = {}
        receipts, shards = [], []
        for ordinal, panel in enumerate(panels):
            results = []
            for color in (0, 1):
                for opening in range(pairs):
                    winner = color if pass_gate else 1 - color
                    results.append(
                        {
                            "opening": f"opening:{opening}",
                            "opponent": panel.opponent,
                            "candidate_player": color,
                            "winner": winner,
                            "illegal": False,
                            "candidate_ms": [float(time_ms)],
                        }
                    )
            report = manager.output / "game-gates" / f"{panel.name}.json"
            write_json(
                report,
                {
                    "schema": "papersoccer.jacek-replay-bfm-comparison.v1",
                    "model_sha256": workflow.sha256(model),
                    "configuration": {
                        "pairs": pairs,
                        "time_ms": time_ms,
                        "exploration": 0.5,
                        "fpu": 0.5,
                        "opponent": panel.opponent,
                        "opening_bank_classification": classification,
                    },
                    "summary": {"games": 2 * pairs},
                    "results": results,
                },
            )
            shard = manager.output / "game-gates/shards" / f"{panel.name}.json"
            receipt = manager.receipts / "19-game-gates-panels" / f"{panel.name}.json"
            write_json(shard, {"panel": panel.name, "games": 2 * pairs})
            write_json(
                receipt,
                {
                    "schema": workflow.CAMPAIGN_RECEIPT_SCHEMA,
                    "panel": panel.name,
                    "report": workflow.artifact_snapshot(report),
                    "shard": workflow.artifact_snapshot(shard),
                },
            )
            reports[panel.name] = str(report)
            shards.append(workflow.artifact_snapshot(shard))
            receipts.append(workflow.artifact_snapshot(receipt))
        return {
            "reports": reports,
            "panel_receipts": receipts,
            "panel_shards": shards,
        }

    @staticmethod
    def fake_latency_audit(
        *, output: pathlib.Path, model: pathlib.Path, **_arguments: object,
    ) -> dict:
        results = [
            {
                "candidate_player": index % 2,
                "winner": index % 2,
                "illegal": False,
                "candidate_ms": [999.0],
            }
            for index in range(20)
        ]
        write_json(
            output,
            {
                "schema": "papersoccer.jacek-replay-bfm-comparison.v1",
                "model_sha256": workflow.sha256(model),
                "results": results,
            },
        )
        return {"candidate_samples": 20, "candidate_max_ms": 999.0}

    def initialize_outer_graph(self, *, resume: bool) -> workflow.GuardedStageManager:
        output = self.root / "auto"
        output.mkdir(parents=True, exist_ok=True)
        manager = workflow.GuardedStageManager(
            output=output,
            campaign_id="selfsearch-auto-acceptance",
            round_index=-1,
            resume=resume,
            environment={"fixture": "acceptance-v1"},
            producer_guard=self.producer_guard,
        )
        trigger = output / "evaluation-trigger.json"
        selection = output / "incumbent-selection.json"

        def trigger_action() -> dict:
            write_json(trigger, {"evaluation": "complete", "invalid_games": 0})
            return {"evaluation": "complete"}

        manager.execute(
            ordinal=0,
            name="evaluation-trigger",
            configuration={"fixture": True},
            producers={"workflow": pathlib.Path(workflow.__file__)},
            inputs={"build": self.build_manifest},
            outputs={"trigger": trigger},
            action=trigger_action,
        )

        def selection_action() -> dict:
            self.actor.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.original_actor, self.actor)
            shutil.copy2(self.original_runner, self.runner)
            result = {
                "incumbent_round": 2,
                "runner_up_round": 1,
                "incumbent": workflow.artifact_snapshot(self.actor),
                "runner_up": workflow.artifact_snapshot(self.runner),
            }
            write_json(selection, result)
            return result

        manager.execute(
            ordinal=1,
            name="incumbent-selection",
            configuration={"policy": "tiny-direct-league"},
            producers={"workflow": pathlib.Path(workflow.__file__)},
            inputs={
                "incumbent": self.original_actor,
                "runner_up": self.original_runner,
            },
            outputs={
                "selection": selection,
                "incumbent": self.actor,
                "runner_up": self.runner,
            },
            action=selection_action,
        )
        return manager

    def run_phase(
        self,
        *, name: str,
        output_name: str,
        resume: bool,
        actor: pathlib.Path,
        diversity: pathlib.Path,
        prior_search: list[pathlib.Path] | tuple[pathlib.Path, ...] = (),
        prior_rank4: list[pathlib.Path] | tuple[pathlib.Path, ...] = (),
        exclusions: list[pathlib.Path] | tuple[pathlib.Path, ...] = (),
    ) -> dict:
        return workflow.run_phase(
            spec=self.phase_spec(name),
            output=self.root / "auto" / output_name,
            resume=resume,
            roots_tsv=self.roots_tsv,
            roots_manifest=self.roots_manifest,
            actor=actor,
            diversity=diversity,
            executables=self.executables,
            anchor_train_manifests=[self.anchor_train],
            anchor_validation_manifests=[self.anchor_validation],
            canonical_prior_manifests=[],
            opening_exclusions=exclusions,
            prior_search_manifests=prior_search,
            prior_rank4_manifests=prior_rank4,
            producer_guard=self.producer_guard,
            build_manifest=self.build_manifest,
            source_identities=SOURCE_IDENTITIES,
        )

    def publish(
        self,
        *, outer: workflow.GuardedStageManager,
        full: dict,
    ) -> pathlib.Path:
        destination = self.root / "auto/promoted/selfsearch-full-acceptance"
        runtime = destination / "jacek_replay_bfm.runtime"
        manifest = destination / "jacek_replay_bfm.runtime.json"
        evidence = destination / "publication.json"
        source_runtime = pathlib.Path(full["search_runtime"])
        source_manifest = pathlib.Path(full["search_manifest"])
        decision = pathlib.Path(full["decision_path"])

        def action() -> dict:
            if not full["decision"].get("eligible_for_local_publication"):
                raise ValueError("ineligible model cannot be published")
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_runtime, runtime)
            shutil.copy2(source_manifest, manifest)
            write_json(
                evidence,
                {
                    "local_only": True,
                    "canonical_rank4_replaced": False,
                    "source_runtime": workflow.artifact_snapshot(source_runtime),
                    "decision": workflow.artifact_snapshot(decision),
                },
            )
            return {"eligible": True, "local_only": True}

        outer.execute(
            ordinal=2,
            name="local-publication",
            configuration={"external_upload": False, "replace_rank4": False},
            producers={"workflow": pathlib.Path(workflow.__file__)},
            inputs={
                "runtime": source_runtime,
                "manifest": source_manifest,
                "decision": decision,
            },
            outputs={"runtime": runtime, "manifest": manifest, "evidence": evidence},
            action=action,
        )
        return runtime


class SelfSearchAcceptanceTests(unittest.TestCase):
    def fixture_patches(self, fixture: TinyAcceptanceFixture):
        return (
            mock.patch.object(
                workflow, "run_training_arm", side_effect=fixture.fake_training_arm
            ),
            mock.patch.object(
                workflow, "anchor_metrics", side_effect=fixture.fake_anchor_metrics
            ),
            mock.patch.object(
                workflow, "generate_comparison_bank", side_effect=fixture.fake_bank
            ),
            mock.patch.object(
                workflow,
                "run_comparison_panels",
                side_effect=fixture.fake_comparison_panels,
            ),
            mock.patch.object(
                workflow, "run_latency_audit", side_effect=fixture.fake_latency_audit
            ),
        )

    def test_pilot_rejection_is_terminal_and_does_not_start_full(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = TinyAcceptanceFixture(pathlib.Path(directory))
            outer = fixture.initialize_outer_graph(resume=False)
            patches = self.fixture_patches(fixture)
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                pilot = fixture.run_phase(
                    name="pilot",
                    output_name="pilot-reject",
                    resume=False,
                    actor=fixture.actor,
                    diversity=fixture.runner,
                )
            self.assertFalse(pilot["decision"]["eligible_for_full"])
            self.assertTrue(
                (fixture.root / "auto/pilot-reject/receipts/21-decision.json").is_file()
            )
            self.assertFalse((fixture.root / "auto/full").exists())
            self.assertFalse(outer.receipt_path(2, "local-publication").exists())

    def test_pilot_pass_runs_full_and_publishes_only_local_eligible_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = TinyAcceptanceFixture(pathlib.Path(directory))
            outer = fixture.initialize_outer_graph(resume=False)
            patches = self.fixture_patches(fixture)
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                pilot = fixture.run_phase(
                    name="pilot",
                    output_name="pilot",
                    resume=False,
                    actor=fixture.actor,
                    diversity=fixture.runner,
                )
                self.assertTrue(pilot["decision"]["eligible_for_full"])
                full = fixture.run_phase(
                    name="full",
                    output_name="full",
                    resume=False,
                    actor=pathlib.Path(pilot["search_runtime"]),
                    diversity=fixture.actor,
                    prior_search=[pathlib.Path(path) for path in pilot["search_new_manifests"]],
                    prior_rank4=[pathlib.Path(path) for path in pilot["rank4_new_manifests"]],
                    exclusions=[pathlib.Path(pilot["opening_bank"])],
                )
            self.assertTrue(full["decision"]["eligible_for_local_publication"])
            published = fixture.publish(outer=outer, full=full)
            self.assertEqual(published.read_bytes(), pathlib.Path(full["search_runtime"]).read_bytes())
            evidence = json.loads(
                (published.parent / "publication.json").read_text()
            )
            self.assertTrue(evidence["local_only"])
            self.assertFalse(evidence["canonical_rank4_replaced"])
            for phase in ("pilot", "full"):
                receipts = fixture.root / "auto" / phase / "receipts"
                self.assertEqual(
                    sorted(path.name for path in receipts.glob("[0-9][0-9]-*.json")),
                    [
                        f"{ordinal:02d}-{name}.json"
                        for ordinal, name in enumerate(
                            (
                                "game-plan", "games", "positions", "search-shallow",
                                "rank4-shallow", "hard-selection", "search-deep",
                                "rank4-deep", "search-targets", "rank4-targets",
                                "adjudicator-positions", "adjudicator-labels",
                                "pack-search", "pack-rank4", "pack-adjudicator",
                                "train-search", "train-rank4", "anchor-metrics",
                                "opening-bank", "game-gates", "latency-audit", "decision",
                            )
                        )
                    ],
                )

    def test_interruption_after_every_phase_stage_resumes_once(self):
        stage_names = (
            "game-plan", "games", "positions", "search-shallow", "rank4-shallow",
            "hard-selection", "search-deep", "rank4-deep", "search-targets",
            "rank4-targets", "adjudicator-positions", "adjudicator-labels",
            "pack-search", "pack-rank4", "pack-adjudicator", "train-search",
            "train-rank4", "anchor-metrics", "opening-bank", "game-gates",
            "latency-audit", "decision",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            producer = root / "producer"
            producer.write_text("frozen\n")
            for interrupted in range(len(stage_names)):
                with self.subTest(stage=stage_names[interrupted]):
                    output = root / f"attempt-{interrupted:02d}"
                    counts = [0] * len(stage_names)
                    crashed = False

                    def run(resume: bool) -> None:
                        nonlocal crashed
                        manager = workflow.GuardedStageManager(
                            output=output,
                            campaign_id="interrupt-matrix",
                            round_index=0,
                            resume=resume,
                            environment={"fixture": True},
                            producer_guard=lambda: None,
                        )
                        for ordinal, name in enumerate(stage_names):
                            artifact = output / "artifacts" / f"{ordinal:02d}.bin"
                            nested = output / "nested" / f"{ordinal:02d}.json"

                            def action(
                                ordinal: int = ordinal,
                                artifact: pathlib.Path = artifact,
                                nested: pathlib.Path = nested,
                            ) -> dict:
                                nonlocal crashed
                                counts[ordinal] += 1
                                artifact.parent.mkdir(parents=True, exist_ok=True)
                                artifact.write_bytes(f"stage-{ordinal}\n".encode())
                                write_json(nested, {"stage": ordinal})
                                if ordinal == interrupted and not crashed:
                                    crashed = True
                                    raise RuntimeError("simulated interruption")
                                return {"nested": workflow.artifact_snapshot(nested)}

                            manager.execute(
                                ordinal=ordinal,
                                name=name,
                                configuration={"ordinal": ordinal},
                                producers={"producer": producer},
                                inputs=(
                                    {"previous": output / "artifacts" / f"{ordinal - 1:02d}.bin"}
                                    if ordinal else {}
                                ),
                                outputs={"artifact": artifact},
                                action=action,
                            )

                    with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                        run(False)
                    run(True)
                    self.assertEqual(counts[:interrupted], [1] * interrupted)
                    self.assertEqual(counts[interrupted], 2)
                    self.assertEqual(counts[interrupted + 1 :], [1] * (21 - interrupted))

    def test_pack_model_gate_publication_and_nested_corruption_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = TinyAcceptanceFixture(pathlib.Path(directory))
            outer = fixture.initialize_outer_graph(resume=False)
            patches = self.fixture_patches(fixture)
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                pilot = fixture.run_phase(
                    name="pilot", output_name="pilot", resume=False,
                    actor=fixture.actor, diversity=fixture.runner,
                )
            published_full = {
                "search_runtime": pilot["search_runtime"],
                "search_manifest": pilot["search_manifest"],
                "decision_path": pilot["decision_path"],
                "decision": {"eligible_for_local_publication": True},
            }
            published = fixture.publish(outer=outer, full=published_full)

            search_pack = json.loads(
                (fixture.root / "auto/pilot/shards/search/pack-report.json").read_text()
            )
            corruptions = (
                (pathlib.Path(search_pack["shards"]["train"]["npz"]), "pack"),
                (pathlib.Path(pilot["search_runtime"]), "model"),
                (fixture.root / "auto/pilot/game-gates/matched.json", "gate"),
                (
                    fixture.root
                    / "auto/pilot/receipts/19-game-gates-panels/matched.json",
                    "nested",
                ),
            )
            for target, label in corruptions:
                with self.subTest(artifact=label):
                    original = target.read_bytes()
                    target.write_bytes(original + b"corrupt\n")
                    fresh_patches = self.fixture_patches(fixture)
                    with fresh_patches[0], fresh_patches[1], fresh_patches[2], fresh_patches[3], fresh_patches[4]:
                        with self.assertRaisesRegex(ValueError, "stale|corrupt|changed"):
                            fixture.run_phase(
                                name="pilot", output_name="pilot", resume=True,
                                actor=fixture.actor, diversity=fixture.runner,
                            )
                    target.write_bytes(original)

            original_publication = published.read_bytes()
            published.write_bytes(original_publication + b"corrupt\n")
            resumed_outer = fixture.initialize_outer_graph(resume=True)
            with self.assertRaisesRegex(ValueError, "stale|corrupt|changed"):
                fixture.publish(outer=resumed_outer, full=published_full)


if __name__ == "__main__":
    unittest.main()
