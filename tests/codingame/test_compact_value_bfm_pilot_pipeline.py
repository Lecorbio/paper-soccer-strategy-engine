import base64
import copy
import hashlib
import importlib.util
import json
import pathlib
import threading
import time
import tempfile
import unittest
from collections import Counter
from unittest import mock

from tools import compact_value_bfm_pilot_pipeline as pipeline
from tools import compact_value_bfm_rank4_teacher_challenger as challenger
from tools import jacek_replay_corpus as corpus
from tools import jacek_replay_features as features
from tests.codingame import test_jacek_replay_corpus as corpus_tests


VALID_GAME_A = (
    "6/7/4/1/2/44/4/7/75/4/3/4/2/42/21/7/0/5/7/25/66/1/200357/06/"
    "4527236436/0530727"
)
VALID_GAME_B = (
    "2/4/70/2/0/3/657/6/4/7/1/0/3/0/7/46/52/53/22/4/16001/661/31/3/"
    "50/5/256723033/2/0/35/2717/6/674702/27/47574/43646"
)


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload if isinstance(payload, bytes) else payload.encode())
    return path


def record(path):
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def route_record(path, root):
    value = record(path)
    value["route"] = path.resolve().relative_to(root.resolve()).as_posix()
    value.pop("path")
    return value


def build_manifest(path, producers):
    for producer in producers.values():
        producer.chmod(0o755)
    body = {
        "schema": challenger.BUILD_MANIFEST_SCHEMA,
        "campaign_id": challenger.CAMPAIGN_ID,
        "status": "clean-source-compiler-binaries-frozen",
        "created_at_utc": "2026-09-04T00:00:00Z",
        "repository": challenger._repository_identity(),
        "source_closure": {
            relative: challenger._regular(pipeline.REPOSITORY / relative)
            for relative in pipeline.PIPELINE_REQUIRED_BUILD_SOURCES
        },
        "compiler": challenger._compiler_identity(),
        "binaries": {
            role: {**challenger._regular(producer), "executable": True}
            for role, producer in producers.items()
        },
        "build_contract": {
            "system": "cmake",
            "configuration": "Release",
            "language_standard": "c++20",
            "sources_clean": True,
            "binaries_built_after_source_freeze": True,
        },
    }
    challenger.qualification.write_sealed(path, body)
    return path


def write_game_manifest(command, input_path, output_path, plan, rows):
    compact = "--compact-student-runtime" in command
    configuration = {
        "bfm_tree_nodes": 8_000,
        "rank4_nodes": 16_000,
        "jacek_nn_nodes": 64_000,
        "exploration": 0.5,
        "fpu": 0.5,
    }
    bindings = {
        "roots_sha256": plan["inputs"]["filtered_roots_tsv"]["sha256"],
        "plan_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "incumbent_model_sha256": plan["inputs"]["accepted_teacher_runtime"][
            "sha256"
        ],
        "runner_up_model_sha256": plan["inputs"]["accepted_teacher_runtime"][
            "sha256"
        ],
    }
    if compact:
        configuration.update({
            "actor_backend": "compact-value-bfm-runtime-v1",
            "minimum_post_prefix_turns": 20,
        })
        bindings.update({
            "compact_student_runtime_sha256": plan["inputs"]["student_runtime"][
                "sha256"
            ],
            "compact_prior_runtime_sha256": plan["inputs"]["prior_runtime"][
                "sha256"
            ],
        })
    manifest_path = pathlib.Path(command[command.index("--manifest") + 1])
    manifest_path.write_text(json.dumps({
        "schema": "papersoccer.jacek-selfsearch-games.v1",
        "campaign_id": plan["campaign_id"],
        "requested_games": len(rows),
        "successful_games": len(rows),
        "configuration": configuration,
        "bindings": bindings,
        "rows": rows,
    }))


def make_runtime(root):
    dimensions = (6301, 12, 8, 1)
    counts = {"w1": 6301 * 12, "w2": 12 * 8, "w3": 8}
    counts["total"] = sum(counts.values())
    packed = bytes((counts["total"] * 3 + 7) // 8)
    body = {
        "schema": challenger.export_model.RUNTIME_SCHEMA,
        "feature_schema": challenger.export_model.FEATURE_SCHEMA,
        "architecture": {
            "name": challenger.export_model.ELIGIBLE[(12, 8)],
            "dimensions": list(dimensions),
            "biases": False,
            "activations": challenger.export_model.ACTIVATIONS,
            "payload_layout": challenger.export_model.LAYOUT,
        },
        "quantization": {
            **challenger.export_model.QUANTIZATION,
            "scales": {"w1": 0.125, "w2": 0.125, "w3": 0.125},
            "weight_counts": counts,
            "packed_byte_count": len(packed),
            "payload_sha256": hashlib.sha256(packed).hexdigest(),
            "payload_base64": base64.b64encode(packed).decode("ascii"),
        },
        "selection": {
            "arm": "teacher-assisted",
            "seed": 20260907,
            "float_epoch": 1,
            "qat_epoch": 0,
            "source_bundle_body_sha256": "1" * 64,
        },
    }
    document = challenger.qualification.seal(body)
    payload = challenger.canonical_json_bytes(document)
    return write(root / f"{hashlib.sha256(payload).hexdigest()}.runtime.json", payload)


def fingerprint_file(path, label, fingerprints=None):
    values = sorted(
        set(fingerprints or [hashlib.sha256(label.encode()).hexdigest()])
    )
    body = {
        "schema": pipeline.FINGERPRINT_SET_SCHEMA,
        "classification": label,
        "canonicalization": pipeline.FOUR_WAY_CANONICALIZATION,
        "fingerprint_domain": pipeline.FEATURE_FINGERPRINT_DOMAIN,
        "sources": [],
        "fingerprints": values,
        "fingerprint_count": len(values),
        "source_paths_followed": False,
        "contains_labels": False,
        "contains_metrics": False,
        "contains_transcripts": False,
    }
    body["body_sha256"] = hashlib.sha256(
        challenger.canonical_json_bytes(body)
    ).hexdigest()
    return write(path, challenger.canonical_json_bytes(body))


def protected_fingerprint_file(path, fingerprints):
    values = list(fingerprints)
    body = {
        "schema": "papersoccer.compact-value-bfm.discrete-v3-protected-canonical-fingerprints.v1",
        "canonicalization": "minimum-sha256-over-exact+rotate+reflect+rotate_reflect",
        "contains_labels": False,
        "contains_metrics": False,
        "contains_transcripts": False,
        "position_count": len(values),
        "unique_canonical_count": len(set(values)),
        "rows": [
            {"row_ordinal": ordinal, "canonical_sha256": value}
            for ordinal, value in enumerate(values)
        ],
    }
    body["body_sha256"] = hashlib.sha256(
        challenger.canonical_json_bytes(body)
    ).hexdigest()
    return write(path, challenger.canonical_json_bytes(body))


class Fixture:
    def __init__(
        self, root, *, collision=False, real_prior=False, include_live=False,
        phase="pilot", dynamic=False,
    ):
        self.root = root.resolve()
        self.runtime = make_runtime(self.root)
        self.teacher = write(self.root / "teacher.runtime", b"accepted-teacher")
        self.rank4_source = write(self.root / "rank4-source.cpp", b"maintained-rank4")
        self.game_producer = write(self.root / "game-producer", "mock\n")
        self.action_teacher = write(self.root / "action-teacher", "mock\n")
        self.rank4_teacher = write(self.root / "rank4-teacher", "mock\n")
        self.initial_checkpoint = write(
            self.root / f"{'a' * 64}.float.npz", "checkpoint\n"
        )
        self.build_manifest = build_manifest(
            self.root / "build-manifest.json",
            {
                "continuation_producer": self.game_producer,
                "action_teacher": self.action_teacher,
                "rank4_position_teacher": self.rank4_teacher,
                "rank4_gate": self.game_producer,
            },
        )
        self.campaign_plan = write(self.root / "campaign-plan.json", "campaign\n")
        self.phase_reference = write(self.root / "phase-reference.json", "phase\n")
        self.phase_plan = write(self.root / "phase-plan.json", "phase-plan\n")
        training_bundle_body = {
            "schema": "papersoccer.compact-value-bfm-input-bundle.v1",
            "campaign_id": "compact-value-bfm-20260831-v1",
            "feature_schema": features.FEATURE_SCHEMA,
            "fixture": True,
        }
        training_bundle = challenger.qualification.seal(training_bundle_body)
        self.training_bundle_manifest = write(
            self.root / "training-bundle.json",
            challenger.canonical_json_bytes(training_bundle),
        )
        if real_prior:
            from tools import jacek_replay_train as replay_train

            prior_train_state = features.ReplayState()
            prior_validation_state = copy.deepcopy(prior_train_state)
            features.apply_complete_turn(prior_validation_state, 0, "0")
            _, self.prior_train, _ = replay_train.write_csr_shard(
                self.root / "prior-shards",
                "train",
                [corpus.LabeledSample(
                    features.encode_active(prior_train_state), 0.0, 1.0, "prior-train"
                )],
            )
            _, self.prior_validation, _ = replay_train.write_csr_shard(
                self.root / "prior-shards",
                "validation",
                [corpus.LabeledSample(
                    features.encode_active(prior_validation_state),
                    0.0,
                    1.0,
                    "prior-validation",
                )],
            )
        else:
            self.prior_train = write(
                self.root / "prior-train.json", "prior-train\n"
            )
            self.prior_validation = write(
                self.root / "prior-validation.json", "prior-validation\n"
            )
        self.exclusions = {
            "mixed-development-fingerprints": fingerprint_file(
                self.root / "mixed.json", "mixed-development"
            ),
            "prior-train-fingerprints": fingerprint_file(
                self.root / "prior-train-fingerprints.json", "prior-train"
            ),
            "prior-validation-fingerprints": fingerprint_file(
                self.root / "prior-validation-fingerprints.json", "prior-validation"
            ),
            "protected-fingerprints": protected_fingerprint_file(
                self.root / "protected-fingerprints.json",
                (
                    [
                        challenger.openings.state_fingerprints(
                            features.ReplayState()
                        )["canonical"]
                    ]
                    if collision
                    else [hashlib.sha256(b"protected").hexdigest()]
                ),
            ),
        }
        self.live_exclusion = fingerprint_file(
            self.root / "live-fingerprints.json",
            "live",
            (
                [
                    corpus.canonical_feature_fingerprint(
                        features.encode_active(features.ReplayState())
                    ).hex()
                ]
                if collision
                else None
            ),
        )
        roots = [
            (f"root-{index}", "train" if index < 8 else "validation")
            for index in range(10)
        ] + [("root-test", "test"), ("root-protected", "train")]
        self.roots_tsv = write(
            self.root / "roots.tsv",
            pipeline.GAME_HEADER
            + "\n"
            + "".join(
                f"{group}\tfixture\t0\t{VALID_GAME_A}\n" for group, _ in roots
            ),
        )
        roots_body = {
            "schema": corpus.ROOT_SCHEMA,
            "feature_schema": features.FEATURE_SCHEMA,
            "tool_sha256": {"normalizer": "2" * 64, "features": "3" * 64},
            "exclusion_boundary": {"read_before_candidate_sources": True},
            "accepted": [
                {
                    "group_id": group,
                    "split": split,
                    "source": "fixture",
                    "winner": 0,
                    "turns": [
                        {"player_id": index % 2, "action": action}
                        for index, action in enumerate(VALID_GAME_A.split("/"))
                    ],
                    **(
                        {"classification": "protected-evaluation"}
                        if group == "root-protected"
                        else {}
                    ),
                }
                for group, split in roots
            ],
        }
        roots_body["body_sha256"] = corpus.sha256_bytes(
            corpus.canonical_json_bytes(roots_body)
        )
        self.roots_manifest = write(
            self.root / "roots.json", corpus.canonical_json_bytes(roots_body)
        )
        training_inputs = {
            name: record(path) for name, path in self.exclusions.items()
            if name.startswith("prior-")
        }
        training_inputs.update({
            "prior-train-manifest": record(self.prior_train),
            "prior-validation-manifest": record(self.prior_validation),
        })
        protected = {
            "mixed-development-fingerprints": record(
                self.exclusions["mixed-development-fingerprints"]
            ),
            "protected-fingerprints": record(
                self.exclusions["protected-fingerprints"]
            ),
            "mixed-six": record(write(
                self.root / "mixed-six-evidence.json",
                json.dumps({
                    "schema": "papersoccer.compact-value-bfm.discrete-v3-development-recovery-mixed-six-exclusion.v1",
                    "selected_banks": [{
                        "bank": {"path": "/original-worktree/missing.json", "sha256": "0" * 64}
                    }],
                }),
            )),
        }
        self.campaign_context = {
            "plan": {"outputs": {"input_directory": str(self.root)}},
            "inputs": {
                "body_sha256": "4" * 64,
                "teacher": {"runtime": record(self.teacher)},
                "candidate": {"runtime": record(self.runtime)},
                "rank4_teacher": record(self.rank4_source),
                "training_inputs": training_inputs,
                "training_bundle": {
                    "manifest": {
                        **record(self.training_bundle_manifest),
                        "body_sha256": training_bundle["body_sha256"],
                    }
                },
                "protected_exclusions": protected,
                "live_exclusions": (
                    {"prior-live-fingerprints": record(self.live_exclusion)}
                    if include_live
                    else {}
                ),
            },
        }
        modes = [
            mode
            for mode, count in challenger.PHASE_QUOTAS[phase].items()
            for _ in range(count)
        ]
        phase_rows = [
            {
                "game_ordinal": index,
                "game_id": f"{phase}-{index:05d}",
                "actor_mode": mode,
                "base_seed": index + 10,
                "worker": index % 10,
            }
            for index, mode in enumerate(modes)
        ]
        dynamic_exclusions = []
        if dynamic:
            dynamic_path = self.root / "prior-protected-fingerprints.json"
            challenger.qualification.write_sealed(dynamic_path, {
                "schema": challenger.DYNAMIC_EXCLUSION_SCHEMA,
                "namespace": challenger.NAMESPACE,
                "campaign_id": challenger.CAMPAIGN_ID,
                "attempt": 0,
                "gate_id": "gate-a",
                "classification": "protected-final-canonical-fingerprints",
                "domain": "protected-final-opening-canonical-state",
                "origin": {
                    "candidate_source_sha256": "1" * 64,
                    "candidate_runtime_sha256": "2" * 64,
                    "protected_bank_sha256": "3" * 64,
                    "seed_sha256": "4" * 64,
                },
                "canonicalization": "minimum(exact,rotate180,reflect,rotate180-reflect)",
                "fingerprints": sorted([
                    hashlib.sha256(f"protected:{index}".encode()).hexdigest()
                    for index in range(500)
                ]),
                "fingerprint_count": 500,
                "contains_transcripts": False,
                "contains_metrics": False,
                "contains_labels": False,
                "training_eligible": False,
                "required_for_all_later_development_and_protected_banks": True,
            })
            dynamic_exclusions = [challenger._dynamic_exclusion_record(dynamic_path)]
        self.phase_context = {
            "path": self.phase_plan,
            "phase": {
                "phase": phase,
                "campaign_id": "pilot-pipeline-fixture",
                "attempt": 1,
                "games": len(phase_rows),
                "quotas": challenger.PHASE_QUOTAS[phase],
                "rows": phase_rows,
                "attempt_inputs": {
                    "student_runtime": record(self.runtime),
                    "prior_runtime": record(self.runtime),
                    "initial_float_checkpoint": record(self.initial_checkpoint),
                    "roots_tsv": record(self.roots_tsv),
                    "roots_manifest": record(self.roots_manifest),
                    "build_manifest": record(self.build_manifest),
                    "producer_binaries": {
                        "continuation_producer": record(self.game_producer),
                        "action_teacher": record(self.action_teacher),
                        "rank4_position_teacher": record(self.rank4_teacher),
                        "rank4_gate": record(self.game_producer),
                    },
                },
                "producer_binaries": {
                    "continuation_producer": record(self.game_producer),
                    "action_teacher": record(self.action_teacher),
                    "rank4_position_teacher": record(self.rank4_teacher),
                    "rank4_gate": record(self.game_producer),
                },
                "dynamic_exclusions": dynamic_exclusions,
            },
        }
        self.plan = pipeline.prepare_pipeline(
            campaign_plan=self.campaign_plan,
            phase_reference=self.phase_reference,
            output_root=self.root / "pipeline",
            student_runtime=self.runtime,
            roots_tsv=self.roots_tsv,
            roots_manifest=self.roots_manifest,
            game_producer=self.game_producer,
            action_teacher=self.action_teacher,
            rank4_teacher=self.rank4_teacher,
            created_at_utc="2026-09-04T00:00:00Z",
            campaign_context=self.campaign_context,
            phase_context=self.phase_context,
        )

    def write_two_games(self):
        plan = pipeline.load_pipeline(self.plan)
        rows = []
        games = []
        for planned in plan["game_plan"]["rows"]:
            ordinal = int(planned["game_ordinal"])
            root_group, transcript = (
                ("root-0", VALID_GAME_A)
                if ordinal % 2 == 0
                else ("root-8", VALID_GAME_B)
            )
            rows.append({
                "game_ordinal": ordinal,
                "game_id": planned["game_id"],
                "actor_mode": planned["actor_mode"],
                "base_seed": planned["base_seed"],
                "root_group_id": root_group,
                "source": "fixture",
                "winner": 0,
                "transcript": transcript,
                "prefix_turns": 0 if ordinal < 2 else len(transcript.split("/")) - 1,
            })
            games.append(f"{root_group}\tfixture\t0\t{transcript}")
        games_payload = (pipeline.GAME_HEADER + "\n" + "\n".join(games) + "\n").encode()
        games_path = pathlib.Path(plan["outputs"]["games"])
        pipeline._write_once(games_path, games_payload)
        pipeline._write_sealed(
            pathlib.Path(plan["outputs"]["games_manifest"]),
            {
                "schema": pipeline.GAME_MANIFEST_SCHEMA,
                "pipeline_body_sha256": plan["body_sha256"],
                "phase": plan["phase"],
                "attempt": plan["attempt"],
                "games": len(rows),
                "shards": [],
                "rows": rows,
                "games_sha256": hashlib.sha256(games_payload).hexdigest(),
            },
        )


def complete_actions(state, maximum=2):
    found = []
    mover = state.to_move

    def visit(current, text):
        if len(found) >= maximum:
            return
        for direction in range(8):
            child = copy.deepcopy(current)
            try:
                features.apply_primitive(child, direction)
            except ValueError:
                continue
            action = text + str(direction)
            if child.winner is not None or child.to_move != mover:
                found.append((action, child))
            else:
                visit(child, action)
            if len(found) >= maximum:
                return

    visit(copy.deepcopy(state), "")
    if len(found) < maximum:
        raise AssertionError("fixture has fewer than two complete actions")
    return found


def action_group(plan, fields, gap):
    position_id, root_group, group_id, source, split, winner, mover, prefix = fields
    state = corpus._prefix_state(
        [
            {"player_id": index % 2, "action": action}
            for index, action in enumerate(prefix.split("/") if prefix else [])
        ]
    )
    parent = int(mover)
    successors = []
    student_by_id = {}
    for index, (physical, child) in enumerate(complete_actions(state)):
        canonical = (
            physical
            if parent == 0
            else "".join(str((int(value) + 4) % 8) for value in physical)
        )
        parent_teacher = gap if index == 0 else 0.0
        parent_student = 0.0 if index == 0 else 1.0
        sign = 1.0 if child.to_move == parent else -1.0
        successor_id = corpus._mover_canonical_position_identity(child)
        successors.append({
            "successor_id": successor_id,
            "active": list(features.encode_active(child)),
            "transcript": canonical,
            "teacher_value": sign * parent_teacher,
            "value_mover": child.to_move,
            "proof": {"solved": False, "proven_winner": None},
            "termination": {
                "reason": "fixed-work-cap",
                "value_status": "backed-up-at-root-termination",
            },
            "visits": 1,
            "selection_visits": 0,
        })
        student_by_id[successor_id] = sign * parent_student
    successors.sort(key=lambda item: (item["successor_id"], item["transcript"]))
    campaign = plan["campaign_id"]
    nodes = pipeline.SHALLOW_TREE_NODES
    seed = int(
        hashlib.sha256(f"{campaign}\0{position_id}\0{nodes}".encode()).hexdigest()[:16],
        16,
    )
    row = {
        "schema": corpus.COMPLETE_TURN_ACTION_GROUP_SCHEMA,
        "feature_schema": features.FEATURE_SCHEMA,
        "source_bundle_body_sha256": plan["source_bundle_body_sha256"],
        "teacher": {
            "kind": "jacek_replay_bfm_search",
            "artifact_sha256": plan["inputs"]["accepted_teacher_runtime"]["sha256"],
            "payload_sha256": "5" * 64,
            "feature_schema_sha256": hashlib.sha256(
                features.FEATURE_SCHEMA.encode()
            ).hexdigest(),
            "source_sha256": "6" * 64,
        },
        "ranking": dict(corpus._ACTION_GROUP_RANKING),
        "split": split,
        "group": {
            "group_id": corpus._mover_canonical_position_identity(state),
            "parent_identity": corpus._mover_canonical_position_identity(state),
            "identity_algorithm": "sha256-mover-canonical-boundary-v1",
            "parent_mover": parent,
            "root_value": 0.2 if gap < 0.35 else -0.2,
            "root_solved": False,
            "proven_winner": None,
            "termination_reason": "fixed-work-cap",
            "successors_exhaustive": True,
            "work_budget": {
                "seed": seed,
                "max_time_ms": 0,
                "max_tree_nodes": nodes,
                "max_actions": 250,
                "max_partial_paths": 50_000,
                "exploration": 0.5,
                "fpu": 0.5,
            },
            "source_binding": {
                "campaign_id": campaign,
                "position_id": position_id,
                "root_group_id": root_group,
                "group_id": group_id,
                "source": source,
                "split": split,
                "winner": int(winner),
                "prefix": [
                    {"player_id": index % 2, "action": action}
                    for index, action in enumerate(prefix.split("/") if prefix else [])
                ],
            },
            "successors": successors,
        },
    }
    corpus.validate_complete_turn_action_group(row)
    return row, [student_by_id[item["successor_id"]] for item in successors]


def rank4_row(fields, plan):
    position_id, root_group, group_id, source, split, winner, mover, prefix = fields
    base = corpus_tests.JacekReplayCorpusTests.rank4_teacher_row()
    base.update({
        "campaign_id": "pilot-pipeline-fixture",
        "position_id": position_id,
        "root_group_id": root_group,
        "group_id": group_id,
        "source": source,
        "split": split,
        "winner": int(winner),
        "mover": int(mover),
        "prefix": [
            {"player_id": index % 2, "action": action}
            for index, action in enumerate(prefix.split("/") if prefix else [])
        ],
    })
    base["teacher"]["source_sha256"] = plan["inputs"]["rank4_source"]["sha256"]
    corpus.sample_from_teacher_row(base)
    return base


class PilotPipelineTests(unittest.TestCase):
    def test_pipeline_binds_and_rechecks_exact_build_source_closure(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary))
            plan = pipeline.load_pipeline(fixture.plan)
            closure = plan["build_source_closure"]
            self.assertEqual(
                set(closure["sources"]),
                set(pipeline.PIPELINE_REQUIRED_BUILD_SOURCES),
            )
            self.assertEqual(
                set(closure["producer_binaries"]),
                challenger.BUILD_BINARY_ROLES,
            )
            self.assertEqual(
                closure["closure_sha256"],
                challenger.sha256_bytes(challenger.canonical_json_bytes({
                    key: value
                    for key, value in closure.items()
                    if key != "closure_sha256"
                })),
            )

            original = challenger._regular
            target = pathlib.Path(pipeline.__file__).resolve()

            def drift(path, *, ascii_required=False):
                value = original(path, ascii_required=ascii_required)
                if pathlib.Path(path).resolve() == target:
                    value = {**value, "sha256": "f" * 64}
                return value

            with mock.patch.object(challenger, "_regular", side_effect=drift):
                with self.assertRaisesRegex(
                    pipeline.PilotPipelineError, "frozen build source closure"
                ):
                    pipeline.load_pipeline(fixture.plan)

    def test_prepare_accepts_bundle_route_build_and_producer_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary))
            campaign = copy.deepcopy(fixture.campaign_context)
            phase = copy.deepcopy(fixture.phase_context)
            manifest_copy = write(
                fixture.root / "artifacts/frozen/build-manifest.json",
                fixture.build_manifest.read_bytes(),
            )
            source_records = {}
            for relative in pipeline.PIPELINE_REQUIRED_BUILD_SOURCES:
                copied = write(
                    fixture.root / "artifacts/frozen/source-closure" / relative,
                    (pipeline.REPOSITORY / relative).read_bytes(),
                )
                source_records[relative] = route_record(copied, fixture.root)
            producer_paths = {}
            for role, source in {
                "continuation_producer": fixture.game_producer,
                "action_teacher": fixture.action_teacher,
                "rank4_position_teacher": fixture.rank4_teacher,
                "rank4_gate": fixture.game_producer,
            }.items():
                copied = write(
                    fixture.root / "artifacts/frozen/producers" / role,
                    source.read_bytes(),
                )
                copied.chmod(0o755)
                producer_paths[role] = copied
            producer_records = {
                role: route_record(path, fixture.root)
                for role, path in producer_paths.items()
            }
            manifest_record = route_record(manifest_copy, fixture.root)
            campaign["inputs"]["build_bundle"] = {
                "manifest": manifest_record,
                "sources": source_records,
            }
            phase["phase"]["attempt_inputs"]["build_manifest"] = manifest_record
            phase["phase"]["attempt_inputs"]["producer_binaries"] = producer_records
            phase["phase"]["producer_binaries"] = producer_records
            plan_path = pipeline.prepare_pipeline(
                campaign_plan=fixture.campaign_plan,
                phase_reference=fixture.phase_reference,
                output_root=fixture.root / "route-pipeline",
                created_at_utc="2026-09-04T00:00:01Z",
                campaign_context=campaign,
                phase_context=phase,
            )
            plan = pipeline.load_pipeline(plan_path)
            self.assertEqual(
                pathlib.Path(plan["build_source_closure"]["manifest"]["path"]),
                manifest_copy.resolve(),
            )

    def test_build_closure_rejects_missing_source_head_and_producer_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary))
            with self.assertRaisesRegex(challenger.ChallengerError, "provenance"):
                challenger.verify_phase_build_source_closure(
                    required_sources=(
                        *pipeline.PIPELINE_REQUIRED_BUILD_SOURCES,
                        "tools/not-a-frozen-source.py",
                    ),
                    campaign_context=fixture.campaign_context,
                    phase_context=fixture.phase_context,
                )

            bad_document = challenger.qualification.load_sealed(
                fixture.build_manifest, challenger.BUILD_MANIFEST_SCHEMA
            )
            bad_body = dict(bad_document)
            bad_body.pop("body_sha256")
            bad_body["repository"] = {
                **bad_body["repository"], "commit": "b" * 40
            }
            bad_manifest = fixture.root / "wrong-head-build.json"
            challenger.qualification.write_sealed(bad_manifest, bad_body)
            wrong_head_phase = copy.deepcopy(fixture.phase_context)
            wrong_head_phase["phase"]["attempt_inputs"]["build_manifest"] = record(
                bad_manifest
            )
            with self.assertRaisesRegex(challenger.ChallengerError, "current HEAD"):
                challenger.verify_phase_build_source_closure(
                    required_sources=pipeline.PIPELINE_REQUIRED_BUILD_SOURCES,
                    campaign_context=fixture.campaign_context,
                    phase_context=wrong_head_phase,
                )

            wrong_source_body = dict(
                challenger.qualification.load_sealed(
                    fixture.build_manifest, challenger.BUILD_MANIFEST_SCHEMA
                )
            )
            wrong_source_body.pop("body_sha256")
            wrong_source_body["source_closure"] = copy.deepcopy(
                wrong_source_body["source_closure"]
            )
            relative = pipeline.PIPELINE_REQUIRED_BUILD_SOURCES[0]
            wrong_source_body["source_closure"][relative]["sha256"] = "f" * 64
            wrong_source_manifest = fixture.root / "wrong-source-build.json"
            challenger.qualification.write_sealed(
                wrong_source_manifest, wrong_source_body
            )
            wrong_source_phase = copy.deepcopy(fixture.phase_context)
            wrong_source_phase["phase"]["attempt_inputs"][
                "build_manifest"
            ] = record(wrong_source_manifest)
            with self.assertRaisesRegex(
                challenger.ChallengerError, "current phase source differs"
            ):
                challenger.verify_phase_build_source_closure(
                    required_sources=pipeline.PIPELINE_REQUIRED_BUILD_SOURCES,
                    campaign_context=fixture.campaign_context,
                    phase_context=wrong_source_phase,
                )

            other = write(fixture.root / "other-producer", "other\n")
            other.chmod(0o755)
            wrong_producer_phase = copy.deepcopy(fixture.phase_context)
            wrong_producer_phase["phase"]["producer_binaries"][
                "action_teacher"
            ] = record(other)
            wrong_producer_phase["phase"]["attempt_inputs"][
                "producer_binaries"
            ] = wrong_producer_phase["phase"]["producer_binaries"]
            with self.assertRaisesRegex(challenger.ChallengerError, "producer differs"):
                challenger.verify_phase_build_source_closure(
                    required_sources=pipeline.PIPELINE_REQUIRED_BUILD_SOURCES,
                    campaign_context=fixture.campaign_context,
                    phase_context=wrong_producer_phase,
                )

    def test_dynamic_protected_fingerprints_are_phase_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary), dynamic=True)
            plan = pipeline.load_pipeline(fixture.plan)
            dynamic = {
                role: record
                for role, record in plan["inputs"]["exclusion_sources"].items()
                if role.startswith("dynamic:protected-final-canonical-fingerprints:")
            }
            self.assertEqual(len(dynamic), 1)
            self.assertEqual(
                {record["sha256"] for record in dynamic.values()},
                {
                    record["sha256"]
                    for record in plan["phase_input_binding"]["dynamic_exclusions"]
                },
            )

    def test_prepare_rejects_artifact_override_outside_phase_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary))
            copied_runtime = write(
                fixture.root / "unbound-copy.runtime.json",
                fixture.runtime.read_bytes(),
            )
            with self.assertRaisesRegex(
                pipeline.PilotPipelineError, "override differs"
            ):
                pipeline.prepare_pipeline(
                    campaign_plan=fixture.campaign_plan,
                    phase_reference=fixture.phase_reference,
                    output_root=fixture.root / "unbound-pipeline",
                    student_runtime=copied_runtime,
                    created_at_utc="2026-09-04T00:00:01Z",
                    campaign_context=fixture.campaign_context,
                    phase_context=fixture.phase_context,
                )

    def test_fingerprint_union_is_content_addressed_and_self_contained(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            first = fingerprint_file(
                root / "first.json", "development-a", ["a" * 64, "b" * 64]
            )
            second = fingerprint_file(
                root / "second.json", "development-b", ["b" * 64, "c" * 64]
            )
            output = pipeline.materialize_fingerprint_set(
                output_directory=root / "frozen",
                classification="mixed-development",
                sources=[first, second],
            )
            self.assertEqual(
                pipeline._fingerprint_values(output, "mixed-development"),
                {"a" * 64, "b" * 64, "c" * 64},
            )
            self.assertEqual(
                output.name,
                f"{hashlib.sha256(output.read_bytes()).hexdigest()}.fingerprint-set.json",
            )
            first.unlink()
            second.unlink()
            self.assertEqual(
                pipeline._fingerprint_values(output, "mixed-development"),
                {"a" * 64, "b" * 64, "c" * 64},
            )
            path_following = write(
                root / "mixed-six.json",
                json.dumps({
                    "schema": "papersoccer.compact-value-bfm.discrete-v3-development-recovery-mixed-six-exclusion.v1",
                    "selected_banks": [{"bank": {"path": "/missing", "sha256": "0" * 64}}],
                }),
            )
            with self.assertRaisesRegex(
                pipeline.PilotPipelineError, "fingerprint-only"
            ):
                pipeline._fingerprint_values(path_following, "mixed-development")

            bank_rows = challenger.openings.generate_openings(
                stage="fixture-development",
                count=1,
                seed=hashlib.sha256(b"fixture-development").digest(),
                excluded_fingerprints=set(),
            )
            bank = challenger.openings.write_bank(
                root / "bank-copy",
                challenger.openings.bank_payload(
                    stage="fixture-development",
                    classification="unprotected-development",
                    seed=hashlib.sha256(b"fixture-development").digest(),
                    exclusions={"body_sha256": "d" * 64, "sources": []},
                    openings=bank_rows,
                ),
            )
            state, _ = challenger.openings.replay_transcript(
                bank_rows[0]["transcript"]
            )
            canonical = corpus.canonical_feature_fingerprint(
                features.encode_active(state)
            ).hex()
            bank_union = pipeline.materialize_fingerprint_set(
                output_directory=root / "frozen-bank",
                classification="mixed-development",
                sources=[bank],
            )
            bank.unlink()
            self.assertEqual(
                pipeline._fingerprint_values(bank_union, "mixed-development"),
                {canonical},
            )

            historical = root / "historical.tsv"
            historical.write_text(
                "# papersoccer.jacek-replay-bfm-opening-bank.v1\n"
                f"# rules={challenger.openings.RULES}\n"
                "# classification=historical-development\n"
                "# seed=fixture\n"
                "# minimum-physical-plies=12\n"
                "opening_id\ttranscript\tstate_identity\n"
                f"historical-0\t{bank_rows[0]['transcript']}\t{'e' * 64}\n"
            )
            historical_union = pipeline.materialize_fingerprint_set(
                output_directory=root / "frozen-historical",
                classification="mixed-development",
                sources=[historical],
            )
            historical.unlink()
            self.assertEqual(
                pipeline._fingerprint_values(
                    historical_union, "mixed-development"
                ),
                {canonical},
            )

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy unavailable")
    def test_prior_csr_fingerprint_sets_survive_source_deletion(self):
        from tools import jacek_replay_train as replay_train

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            fixture = Fixture(root, real_prior=True)
            for classification, manifest in (
                ("prior-train", fixture.prior_train),
                ("prior-validation", fixture.prior_validation),
            ):
                shard = replay_train.load_csr_shard(manifest)
                expected = {
                    corpus.canonical_feature_fingerprint(shard.active(row)).hex()
                    for row in range(len(shard))
                }
                output = pipeline.materialize_fingerprint_set(
                    output_directory=root / "frozen-csr",
                    classification=classification,
                    sources=[manifest],
                )
                npz = manifest.parent / json.loads(manifest.read_bytes())["npz"]
                manifest.unlink()
                npz.unlink()
                self.assertEqual(
                    pipeline._fingerprint_values(output, classification), expected
                )

    def test_prepare_and_game_chunks_are_resumable_and_worker_bounded(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary))
            plan = pipeline.load_pipeline(fixture.plan)
            filtered = pathlib.Path(plan["outputs"]["filtered_roots"])
            self.assertNotIn("root-test\t", filtered.read_text())
            self.assertNotIn("root-protected\t", filtered.read_text())
            self.assertEqual(plan["game_plan"]["filtered_root_counts"]["retained"], 10)
            self.assertEqual(
                set(plan["inputs"]["exclusion_sources"]),
                {"mixed-development", "prior-train", "prior-validation", "protected"},
            )
            self.assertEqual(len(filtered.read_text().splitlines()), 11)
            self.assertEqual(
                pathlib.Path(plan["outputs"]["selfsearch_plan"])
                .read_text()
                .splitlines()[0],
                "game_ordinal\tactor_mode\tbase_seed",
            )
            lock = threading.Lock()
            active = maximum = calls = 0
            profiles = Counter()

            def producer(command, input_path, output_path, environment):
                nonlocal active, maximum, calls
                with lock:
                    active += 1
                    calls += 1
                    maximum = max(maximum, active)
                try:
                    self.assertEqual(environment["OMP_NUM_THREADS"], "1")
                    rows = input_path.read_text().splitlines()[1:]
                    modes = {row.split("\t")[1] for row in rows}
                    compact = "--compact-student-runtime" in command
                    self.assertEqual(
                        compact,
                        modes.issubset(pipeline.COMPACT_GAME_MODES),
                    )
                    if not compact:
                        self.assertTrue(modes.issubset(pipeline.INCUMBENT_GAME_MODES))
                    with lock:
                        profiles["compact" if compact else "incumbent"] += 1
                    output = [pipeline.GAME_HEADER]
                    manifest_rows = []
                    for row in rows:
                        game_ordinal, actor_mode, base_seed = row.split("\t")
                        root_group = f"root-{int(game_ordinal) % 10}"
                        output.append(
                            f"{root_group}\t{plan['campaign_id']}\t0\t{VALID_GAME_A}"
                        )
                        manifest_rows.append({
                            "game_ordinal": int(game_ordinal),
                            "base_seed": int(base_seed),
                            "actor_mode": actor_mode,
                            "root_group_id": root_group,
                            "winner": 0,
                            "prefix_turns": 0,
                            "transcript_sha256": hashlib.sha256(
                                VALID_GAME_A.encode()
                            ).hexdigest(),
                        })
                    output_path.write_text("\n".join(output) + "\n")
                    write_game_manifest(
                        command, input_path, output_path, plan, manifest_rows
                    )
                    time.sleep(0.01)
                finally:
                    with lock:
                        active -= 1

            receipt = pipeline.run_game_chunks(
                fixture.plan, workers=8, producer=producer
            )
            self.assertEqual(receipt["details"]["games"], 2_000)
            self.assertEqual(receipt["details"]["execution_chunks"], 20)
            self.assertEqual(calls, 20)
            self.assertEqual(profiles, {"compact": 10, "incumbent": 10})
            self.assertLessEqual(maximum, 8)
            self.assertGreater(maximum, 1)
            pipeline.run_game_chunks(
                fixture.plan, workers=8, resume=True, producer=producer
            )
            self.assertEqual(calls, 20)

    def test_full_phase_uses_same_native_compatible_execution_profiles(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary), phase="full")
            plan = pipeline.load_pipeline(fixture.plan)
            self.assertEqual(plan["phase"], "full")
            self.assertEqual(plan["game_plan"]["games"], 10_000)
            self.assertEqual(
                Counter(row["actor_mode"] for row in plan["game_plan"]["rows"]),
                Counter(challenger.FULL_QUOTAS),
            )
            self.assertEqual(pathlib.Path(plan["outputs"]["root"]).name, "full")
            self.assertEqual(
                pathlib.Path(plan["outputs"]["root"]).parent.name,
                "attempt-001",
            )
            calls = Counter()

            def producer(command, input_path, output_path, environment):
                self.assertEqual(environment["MKL_NUM_THREADS"], "1")
                planned = input_path.read_text().splitlines()[1:]
                modes = {line.split("\t")[1] for line in planned}
                compact = "--compact-student-runtime" in command
                profile = "compact" if compact else "incumbent"
                self.assertTrue(
                    modes.issubset(
                        pipeline.COMPACT_GAME_MODES
                        if compact
                        else pipeline.INCUMBENT_GAME_MODES
                    )
                )
                calls[profile] += 1
                output = [pipeline.GAME_HEADER]
                evidence = []
                for line in planned:
                    game_ordinal, actor_mode, base_seed = line.split("\t")
                    root_group = f"root-{int(game_ordinal) % 10}"
                    output.append(
                        f"{root_group}\t{plan['campaign_id']}\t0\t{VALID_GAME_A}"
                    )
                    evidence.append({
                        "game_ordinal": int(game_ordinal),
                        "base_seed": int(base_seed),
                        "actor_mode": actor_mode,
                        "root_group_id": root_group,
                        "winner": 0,
                        "prefix_turns": 0,
                        "transcript_sha256": hashlib.sha256(
                            VALID_GAME_A.encode()
                        ).hexdigest(),
                    })
                output_path.write_text("\n".join(output) + "\n")
                write_game_manifest(command, input_path, output_path, plan, evidence)

            receipt = pipeline.run_game_chunks(
                fixture.plan, workers=4, producer=producer
            )
            self.assertEqual(receipt["details"]["games"], 10_000)
            self.assertEqual(receipt["details"]["execution_chunks"], 20)
            self.assertEqual(calls, {"compact": 10, "incumbent": 10})

    def test_positions_hard_selection_merge_and_standard_scalar_pack(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(
                pathlib.Path(temporary), collision=True, include_live=True
            )
            fixture.write_two_games()
            position_receipt = pipeline.materialize_positions(fixture.plan)
            self.assertEqual(position_receipt["details"]["positions"], 40)
            plan = pipeline.load_pipeline(fixture.plan)
            manifest = pipeline._load_sealed(
                pathlib.Path(plan["outputs"]["positions_manifest"]),
                pipeline.POSITION_MANIFEST_SCHEMA,
                "positions",
            )
            position_fields = [
                line.split("\t")
                for line in pathlib.Path(plan["outputs"]["positions"])
                .read_text()
                .splitlines()[1:]
            ]
            self.assertEqual(
                {fields[2] for fields in position_fields},
                {row["game_id"] for row in plan["game_plan"]["rows"][:2]},
            )
            self.assertEqual(
                manifest["position_stratum_counts"],
                {"opening": 10, "middle": 10, "late": 10, "decisive": 10},
            )
            self.assertEqual(
                manifest["exclusion_audit"]["train_validation_intersection_count"], 0
            )
            self.assertGreaterEqual(
                manifest["exclusion_audit"]["candidate_external_intersections"][
                    "protected"
                ],
                2,
            )
            self.assertGreaterEqual(
                manifest["exclusion_audit"]["candidate_external_intersections"][
                    "live:prior-live-fingerprints"
                ],
                2,
            )
            self.assertEqual(manifest["split_counts"], {"train": 20, "validation": 20})
            predictions = {}
            expected = set()
            per_game = Counter()

            def action_producer(_command, input_path, output_path, environment):
                self.assertEqual(environment["OPENBLAS_NUM_THREADS"], "1")
                rows = []
                for ordinal, line in enumerate(input_path.read_text().splitlines()[1:]):
                    fields = line.split("\t")
                    gap = (
                        0.9
                        if ordinal % 20 == 0
                        else (0.1, 0.4, 0.2, 0.3)[ordinal % 4]
                    )
                    action, student = action_group(plan, fields, gap)
                    if ordinal % 20 == 0:
                        action["group"]["successors_exhaustive"] = False
                    rows.append(action)
                    predictions[action["group"]["group_id"]] = student
                    if gap == 0.4:
                        expected.add(fields[0])
                        per_game[fields[2]] += 1
                output_path.write_bytes(
                    b"".join(corpus.canonical_json_bytes(row) for row in rows)
                )

            def rank4_producer(_command, input_path, output_path, _environment):
                rows = [
                    rank4_row(line.split("\t"), plan)
                    for line in input_path.read_text().splitlines()[1:]
                ]
                output_path.write_bytes(
                    b"".join(corpus.canonical_json_bytes(row) for row in rows)
                )

            pipeline.run_shallow_action_labels(
                fixture.plan, workers=2, producer=action_producer
            )
            pipeline.run_rank4_labels(
                fixture.plan, workers=2, producer=rank4_producer
            )
            action_rows = corpus.load_complete_turn_action_groups(
                (pathlib.Path(plan["outputs"]["shallow_actions"]),)
            )
            self.assertEqual(len(action_rows), 40)
            self.assertEqual(set(per_game.values()), {5})
            hard = pipeline.select_hard_positions(
                fixture.plan,
                predictor=lambda group: predictions[group["group_id"]],
            )
            self.assertEqual(hard["details"]["selected"], 10)
            hard_ids = {
                line.split("\t")[0]
                for line in pathlib.Path(plan["outputs"]["hard_positions"])
                .read_text()
                .splitlines()[1:]
            }
            self.assertEqual(hard_ids, expected)
            hard_report = pipeline._load_sealed(
                pathlib.Path(plan["outputs"]["hard_report"]),
                pipeline.HARD_SELECTION_SCHEMA,
                "hard report",
            )
            self.assertEqual(hard_report["nonexhaustive_fill"], 0)
            self.assertEqual(hard_report["games"], 2)
            self.assertEqual(
                sum(row["selected_for_deep_label"] for row in hard_report["rows"]),
                10,
            )
            self.assertEqual(hard_report["selected_tactical"]["positive_action_regret"], 10)

            shallow_by_position = {
                row["group"]["source_binding"]["position_id"]: row
                for row in action_rows
            }

            def deep_producer(command, input_path, output_path, environment):
                self.assertEqual(environment["VECLIB_MAXIMUM_THREADS"], "1")
                self.assertEqual(
                    command[command.index("--tree-nodes") + 1], "500000"
                )
                deep = []
                for line in input_path.read_text().splitlines()[1:]:
                    position_id = line.split("\t")[0]
                    value = copy.deepcopy(shallow_by_position[position_id])
                    source = value["group"]["source_binding"]
                    value["group"]["work_budget"]["max_tree_nodes"] = 500_000
                    material = (
                        f"{source['campaign_id']}\0{source['position_id']}\0{500_000}".encode()
                    )
                    value["group"]["work_budget"]["seed"] = int(
                        hashlib.sha256(material).hexdigest()[:16], 16
                    )
                    deep.append(value)
                output_path.write_bytes(
                    b"".join(corpus.canonical_json_bytes(row) for row in deep)
                )

            pipeline.run_deep_action_labels(
                fixture.plan, workers=2, producer=deep_producer
            )

            def packer(_plan, _merged, output_directory):
                result = {}
                for split in ("train", "validation"):
                    npz_payload = f"{split}-npz".encode()
                    npz = write(
                        output_directory
                        / f"{hashlib.sha256(npz_payload).hexdigest()}.npz",
                        npz_payload,
                    )
                    manifest_payload = corpus.canonical_json_bytes({
                        "schema": "papersoccer.jacek-replay-csr-shard.v1",
                        "feature_schema": features.FEATURE_SCHEMA,
                        "split": split,
                        "npz": npz.name,
                        "npz_sha256": hashlib.sha256(npz_payload).hexdigest(),
                        "samples": 40,
                    })
                    manifest = write(
                        output_directory
                        / f"{hashlib.sha256(manifest_payload).hexdigest()}.json",
                        manifest_payload,
                    )
                    result[f"{split}_manifest"] = manifest
                    result[f"{split}_npz"] = npz
                return result

            final = pipeline.finalize_labels(fixture.plan, standard_packer=packer)
            self.assertEqual(final["details"]["groups"], 40)
            self.assertEqual(final["details"]["deep_replacements"], 10)
            self.assertEqual(final["details"]["scalar_samples"], 80)
            aggregate = json.loads(
                pathlib.Path(plan["outputs"]["successor_labels"]).read_bytes()
            )
            corpus.validate_complete_turn_successor_labels(aggregate)
            if importlib.util.find_spec("numpy"):
                from tools import compact_value_bfm_train as compact_trainer

                content_path = pathlib.Path(
                    final["outputs"]["successor_labels"]["path"]
                )
                trainer_labels = compact_trainer.validate_successor_label_document(
                    aggregate,
                    source_bundle_body_sha256=plan[
                        "source_bundle_body_sha256"
                    ],
                    artifact_sha256=hashlib.sha256(
                        content_path.read_bytes()
                    ).hexdigest(),
                )
                self.assertEqual(
                    trainer_labels.source_bundle_body_sha256,
                    plan["source_bundle_body_sha256"],
                )
            for split in ("train", "validation"):
                reference = pipeline._load_sealed(
                    pathlib.Path(plan["outputs"][f"scalar_{split}_reference"]),
                    pipeline.SHARD_REFERENCE_SCHEMA,
                    f"{split} reference",
                )
                self.assertEqual(
                    reference["shard_schema"],
                    "papersoccer.jacek-replay-csr-shard.v1",
                )

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy unavailable")
    def test_default_scalar_packer_writes_standard_train_validation_shards(self):
        import numpy as np
        from tools import compact_value_bfm_train as compact_trainer
        from tools import jacek_replay_train as replay_train

        class LocalBundle:
            def __init__(self, root):
                self.root = root

            def is_protected(self, _relative):
                return False

            def artifact_path(
                self, relative, *, allow_protected=False, protected_context=False
            ):
                self.assert_unprotected = not allow_protected and not protected_context
                return self.root / relative

        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary), real_prior=True)
            plan = pipeline.load_pipeline(fixture.plan)
            rows = []
            expected_targets = {}
            for ordinal, (root_group, split, transcript, turns) in enumerate((
                ("root-0", "train", VALID_GAME_A, 4),
                ("root-8", "validation", VALID_GAME_B, 5),
            )):
                prefix = "/".join(transcript.split("/")[:turns])
                state = corpus._prefix_state([
                    {"player_id": index % 2, "action": action}
                    for index, action in enumerate(prefix.split("/"))
                ])
                fields = [
                    f"position:standard-{ordinal}",
                    root_group,
                    root_group,
                    "fixture",
                    split,
                    "0",
                    str(state.to_move),
                    prefix,
                ]
                row, _student = action_group(plan, fields, 0.25)
                rows.append(row)
                expected_targets[split] = corpus.sample_from_teacher_row(row)[0].target
            merged = write(
                pathlib.Path(plan["outputs"]["merged_actions"]),
                b"".join(corpus.canonical_json_bytes(row) for row in rows),
            )
            packed = pipeline._standard_train_validation_pack(
                plan, merged, pathlib.Path(plan["outputs"]["scalar_shards"])
            )
            self.assertEqual(
                set(packed),
                {
                    "train_manifest", "train_npz",
                    "validation_manifest", "validation_npz",
                },
            )
            for split in ("train", "validation"):
                shard = replay_train.load_csr_shard(packed[f"{split}_manifest"])
                compact_dataset = compact_trainer.load_shard(
                    LocalBundle(packed[f"{split}_manifest"].parent),
                    packed[f"{split}_manifest"].name,
                )
                prior = replay_train.load_csr_shard(
                    pathlib.Path(
                        plan["inputs"]["prior_shard_manifests"][split][0]["path"]
                    )
                )
                self.assertEqual(shard.split, split)
                self.assertEqual(compact_dataset.split, split)
                self.assertEqual(len(compact_dataset), len(shard))
                self.assertTrue(
                    np.allclose(shard.targets, expected_targets[split])
                )
                self.assertTrue(
                    set(shard.group_ids.tolist()).isdisjoint(
                        set(prior.group_ids.tolist())
                    )
                )


if __name__ == "__main__":
    unittest.main()
