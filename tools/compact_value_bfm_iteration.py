#!/usr/bin/env python3
"""Run the single authorized Compact Value-BFM on-policy iteration.

The tool is an orchestration boundary.  Existing operational actor, teacher,
and pack tools do the long data work; this module freezes their exact commands,
installs an interactive LaunchAgent, validates ten resumable worker results,
fine-tunes the selected compact float checkpoint, and hands the result to the
existing fixed-scale QAT/selection implementation.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import plistlib
import re
import shutil
import inspect
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
REPOSITORY = HERE.parent
CAMPAIGN_PATH = HERE / "compact_value_bfm_campaign.py"
QUALIFICATION_PATH = HERE / "compact_value_bfm_qualification.py"
TRAINER_PATH = HERE / "compact_value_bfm_train.py"
COMPACT_WORKFLOW_PATH = HERE / "compact_value_bfm_workflow.py"
PREFLIGHT_PATH = HERE / "compact_value_bfm_preflight.py"
SELFSEARCH_WORKFLOW_PATH = HERE / "jacek_selfsearch_workflow.py"
REPLAY_WORKFLOW_PATH = HERE / "jacek_replay_workflow.py"
REPLAY_PACK_PATH = HERE / "jacek_replay_pack.py"
REPLAY_CORPUS_PATH = HERE / "jacek_replay_corpus.py"
REPLAY_FEATURES_PATH = HERE / "jacek_replay_features.py"
REPLAY_TRAIN_PATH = HERE / "jacek_replay_train.py"
COMPACT_BOT_DIRECTORY = (
    REPOSITORY / "submissions/codingame/bots/compact_value_bfm"
)
MODEL_EXPORTER_PATH = COMPACT_BOT_DIRECTORY / "export_model.py"
SUBMISSION_EXPORTER_PATH = COMPACT_BOT_DIRECTORY / "export_submission.py"
SUBMISSION_CONFIG_PATH = COMPACT_BOT_DIRECTORY / "submission.json"
SUBMISSION_SOURCES_PATH = COMPACT_BOT_DIRECTORY / "sources.txt"

# Every repository-owned Python module that can determine authorization,
# worker scheduling, split assignment, feature encoding, CSR contents, or
# post-worker selection is an immutable member of the iteration tool closure.
# The Python interpreter and non-Python submission manifests are recorded
# separately in the plan.
MAINTAINED_PYTHON_TOOL_PATHS = {
    "campaign": CAMPAIGN_PATH,
    "qualification": QUALIFICATION_PATH,
    "trainer": TRAINER_PATH,
    "compact_workflow": COMPACT_WORKFLOW_PATH,
    "preflight": PREFLIGHT_PATH,
    "iteration_runner": pathlib.Path(__file__).resolve(),
    "selfsearch_workflow": SELFSEARCH_WORKFLOW_PATH,
    "replay_workflow": REPLAY_WORKFLOW_PATH,
    "pack_tool": REPLAY_PACK_PATH,
    "replay_corpus": REPLAY_CORPUS_PATH,
    "replay_features": REPLAY_FEATURES_PATH,
    "replay_train": REPLAY_TRAIN_PATH,
    "model_exporter": MODEL_EXPORTER_PATH,
    "submission_exporter": SUBMISSION_EXPORTER_PATH,
}

NAMESPACE = "compact_value_bfm"
CAMPAIGN_ID = "compact-value-bfm-on-policy-iteration-20260831-v1"
PLAN_SCHEMA = "papersoccer.compact-value-bfm.iteration-plan.v1"
PLAN_BINDING_SCHEMA = "papersoccer.compact-value-bfm.iteration-plan-binding.v1"
CLAIM_SCHEMA = "papersoccer.compact-value-bfm.iteration-execution-claim.v1"
WORKER_RESULT_SCHEMA = "papersoccer.compact-value-bfm.iteration-worker-result.v1"
RUN_RECEIPT_SCHEMA = "papersoccer.compact-value-bfm.iteration-run-receipt.v1"
SELECTION_SCHEMA = "papersoccer.compact-value-bfm.iteration-selection.v1"
POST_ITERATION_HANDOFF_SCHEMA = (
    "papersoccer.compact-value-bfm.post-iteration-development-handoff.v1"
)
REFERENCE_SCHEMA = "papersoccer.compact-value-bfm.iteration-reference.v1"
FINE_TUNE_REFERENCE_SCHEMA = "papersoccer.compact-value-bfm.iteration-fine-tune-reference.v1"

WORKERS = 10
POSITIONS_PER_GAME = 20
DEEP_RELABEL_FRACTION = 0.25
MAXIMUM_LEARNING_RATE = 0.00006
TARGET_SEMANTICS = "75-percent-fixed-work-search-25-percent-terminal-outcome"
WORKFLOW_CONFIGURATION = {
    "games": 10_000,
    "game_chunk_size": 25,
    "game_workers": 10,
    "positions_per_game": 20,
    "bfm_actor_tree_nodes": 8_000,
    "rank4_actor_nodes": 16_000,
    "jacek_nn_actor_nodes": 64_000,
    "exploration": 0.5,
    "fpu": 0.5,
    "early_exploration_percent": 15,
    "early_exploration_turns": 8,
    "bfm_shallow_tree_nodes": 64_000,
    "bfm_deep_tree_nodes": 500_000,
    "rank4_shallow_nodes": 32_000,
    "rank4_deep_nodes": 400_000,
    "hard_fraction_numerator": 1,
    "hard_fraction_denominator": 4,
}
FIXED_WORK_CONFIGURATION = {
    "game_generation": {
        "candidate_tree_nodes": 8_000,
        "rank4_nodes": 16_000,
        "jacek_nn_nodes": 64_000,
        "exploration": 0.5,
        "fpu": 0.5,
        "maximum_turns": 320,
        "minimum_post_prefix_turns": 20,
    },
    "search_shallow": {
        "nodes": 64_000,
        "time_ms": 0,
        "max_actions": 250,
        "max_partial_paths": 50_000,
        "exploration": 0.5,
        "fpu": 0.5,
    },
    "rank4_shallow": {
        "nodes": 32_000,
        "time_ms": 0,
        "max_turn_depth": 32,
        "replay_value_blend_percent": 15,
        "teacher_residual_weight_percent": 100,
    },
    "search_deep": {
        "nodes": 500_000,
        "time_ms": 0,
        "max_actions": 250,
        "max_partial_paths": 50_000,
        "exploration": 0.5,
        "fpu": 0.5,
    },
    "rank4_deep": {
        "nodes": 400_000,
        "time_ms": 0,
        "max_turn_depth": 32,
        "replay_value_blend_percent": 15,
        "teacher_residual_weight_percent": 100,
    },
    "hard_fraction": [1, 4],
    "global_workers": 10,
    "single_thread": True,
}
QUOTAS = {
    "student-selfplay": 5_000,
    "student-p1-vs-rank4": 1_000,
    "student-p2-vs-rank4": 1_000,
    "student-p1-vs-jacek-nn": 1_000,
    "student-p2-vs-jacek-nn": 1_000,
    "student-p1-vs-prior-incumbent": 500,
    "student-p2-vs-prior-incumbent": 500,
}
CAMPAIGN_GAME_KEYS = {
    "student-selfplay": "student_self_play",
    "student-p1-vs-rank4": "rank4_candidate_as_0",
    "student-p2-vs-rank4": "rank4_candidate_as_1",
    "student-p1-vs-jacek-nn": "jacek_nn_candidate_as_0",
    "student-p2-vs-jacek-nn": "jacek_nn_candidate_as_1",
    "student-p1-vs-prior-incumbent": "previous_compact_candidate_as_0",
    "student-p2-vs-prior-incumbent": "previous_compact_candidate_as_1",
}
CAMPAIGN_ARCHITECTURES = {
    "compact-8x8": "6301-8-8-1",
    "source-neutral-8x16": "6301-8-16-1",
    "capacity-12x8": "6301-12-8-1",
}
TOTAL_GAMES = 10_000
SHA256_RE = re.compile(r"[0-9a-f]{64}")
LABEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class IterationError(ValueError):
    """The one-shot iteration plan, environment, or result is invalid."""


def _load_module(path: pathlib.Path, name: str) -> Any:
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise IterationError(f"cannot load iteration dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def campaign_module() -> Any:
    return _load_module(CAMPAIGN_PATH, "compact_iteration_campaign")


def qualification_module() -> Any:
    return _load_module(QUALIFICATION_PATH, "compact_iteration_qualification")


def trainer_module() -> Any:
    return _load_module(TRAINER_PATH, "compact_iteration_trainer")


def selfsearch_module() -> Any:
    tools_path = str(HERE)
    if tools_path not in sys.path:
        sys.path.insert(0, tools_path)
    return _load_module(SELFSEARCH_WORKFLOW_PATH, "compact_iteration_selfsearch")


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, allow_nan=False,
                   sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def valid_sha(value: object) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def body_hashed(body: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(body)
    result["body_sha256"] = sha256_bytes(canonical_json_bytes(body))
    return result


def verify_body_hash(value: Mapping[str, Any], schema: str, label: str) -> None:
    body = dict(value)
    claimed = body.pop("body_sha256", None)
    if (body.get("schema") != schema or not valid_sha(claimed)
            or claimed != sha256_bytes(canonical_json_bytes(body))):
        raise IterationError(f"{label} body SHA-256 is invalid")


def atomic_write_once(path: pathlib.Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise IterationError(f"immutable iteration artifact collision: {path}")
        return
    temporary: pathlib.Path | None = None
    try:
        descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary = pathlib.Path(raw)
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise IterationError(f"immutable iteration artifact raced: {path}")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_document(path: pathlib.Path, body: Mapping[str, Any]) -> dict[str, Any]:
    document = body_hashed(body)
    atomic_write_once(path, canonical_json_bytes(document))
    return document


def write_content_addressed(
    directory: pathlib.Path, body: Mapping[str, Any], suffix: str,
) -> pathlib.Path:
    document = body_hashed(body)
    payload = canonical_json_bytes(document)
    path = directory / f"{sha256_bytes(payload)}{suffix}"
    atomic_write_once(path, payload)
    return path


def load_document(path: pathlib.Path, schema: str, label: str) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IterationError(f"cannot load {label}") from error
    if not isinstance(value, dict) or payload != canonical_json_bytes(value):
        raise IterationError(f"{label} is not canonical JSON")
    verify_body_hash(value, schema, label)
    return value


def file_record(
    path: pathlib.Path, *, executable: bool = False,
    preserve_lexical_path: bool = False,
) -> dict[str, Any]:
    lexical = path.absolute()
    resolved = lexical.resolve()
    if not resolved.is_file() or (executable and not os.access(lexical, os.X_OK)):
        raise IterationError(f"iteration input is not a regular executable/file: {lexical}")
    return {
        "path": str(lexical if preserve_lexical_path else resolved),
        "resolved_path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved), "executable": executable,
    }


def artifact_reference(path: pathlib.Path, schema: str) -> dict[str, Any]:
    document = load_document(path, schema, path.name)
    return {"path": str(path.resolve()), "sha256": sha256_file(path),
            "body_sha256": document["body_sha256"]}


def require_within(path: pathlib.Path, root: pathlib.Path, label: str) -> pathlib.Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise IterationError(f"{label} escaped its iteration output root") from error
    return resolved


def single_thread_environment(base: Mapping[str, str] | None = None) -> dict[str, str]:
    result = dict(os.environ if base is None else base)
    result.update({
        "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "PYTHONHASHSEED": "0",
        "COMPACT_VALUE_BFM_INTERACTIVE_LAUNCHAGENT": "1",
    })
    return result


def exact_game_plan(seed: int = 2026091001) -> dict[str, Any]:
    workflow = selfsearch_module()
    plan = workflow.make_game_plan(
        campaign_id=CAMPAIGN_ID, seed=seed, quotas=QUOTAS
    )
    if (plan.get("schema") != workflow.GAME_PLAN_SCHEMA
            or plan.get("campaign_id") != CAMPAIGN_ID
            or plan.get("seed") != seed
            or plan.get("quotas") != QUOTAS
            or plan.get("games") != TOTAL_GAMES
            or not isinstance(plan.get("rows"), list)
            or len(plan["rows"]) != TOTAL_GAMES):
        raise IterationError("self-search game-plan helper changed")
    rendered = workflow.render_game_plan_tsv(plan)
    if not isinstance(rendered, bytes) or rendered.count(b"\n") != TOTAL_GAMES + 1:
        raise IterationError("self-search game-plan renderer changed")
    return plan


def exact_game_rows(seed: int = 2026091001) -> list[dict[str, Any]]:
    return [dict(row) for row in exact_game_plan(seed)["rows"]]


def worker_rows(rows: Sequence[Mapping[str, Any]], worker: int) -> list[dict[str, Any]]:
    if not 0 <= worker < WORKERS:
        raise IterationError("worker index is outside the exact ten-worker roster")
    result = [dict(row) for index, row in enumerate(rows) if index % WORKERS == worker]
    if len(result) != TOTAL_GAMES // WORKERS:
        raise IterationError("worker plan does not contain exactly 1,000 games")
    return result


def worker_quota(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row["actor_mode"]) for row in rows)
    return {name: counts[name] for name in QUOTAS}


def render_worker_plan(rows: Sequence[Mapping[str, Any]]) -> bytes:
    lines = ["game_ordinal\tactor_mode\tbase_seed"]
    lines.extend(
        f"{row['game_ordinal']}\t{row['actor_mode']}\t{row['base_seed']}"
        for row in rows
    )
    return ("\n".join(lines) + "\n").encode("ascii")


def game_identity(row: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes({
        "game_ordinal": row["game_ordinal"],
        "actor_mode": row["actor_mode"],
        "base_seed": row["base_seed"],
    }))


def game_identity_set_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    identities = sorted(game_identity(row) for row in rows)
    if len(identities) != len(set(identities)):
        raise IterationError("iteration worker plan repeats a game identity")
    return sha256_bytes(canonical_json_bytes(identities))


def validate_selfsearch_contract(path: pathlib.Path) -> tuple[Any, dict[str, str]]:
    if path.resolve() != SELFSEARCH_WORKFLOW_PATH.resolve():
        raise IterationError(
            "iteration pipeline must be the maintained jacek_selfsearch_workflow"
        )
    workflow = selfsearch_module()
    configuration = getattr(workflow, "FULL_CONFIGURATION", None)
    if (dict(getattr(workflow, "FULL_QUOTAS", {})) != QUOTAS
            or not isinstance(configuration, Mapping)
            or any(configuration.get(key) != value
                   for key, value in WORKFLOW_CONFIGURATION.items())
            or getattr(workflow, "SEARCH_MAX_ACTIONS", None) != 250
            or getattr(workflow, "SEARCH_MAX_PARTIAL_PATHS", None) != 50_000
            or getattr(workflow, "FIXED_WORK_TIME_MS", None) != 0
            or getattr(workflow, "POSITION_CHUNK_GAMES", None) != 25):
        raise IterationError("maintained self-search fixed-work contract changed")
    required_helpers = (
        "make_game_plan", "render_game_plan_tsv", "run_game_chunks",
        "merge_game_chunks", "freeze_positions", "run_label_chunks",
        "select_hard_positions", "merge_deep_labels", "run_pack",
        "_validate_chunk_stage_result", "_validate_pack_report",
        "_pack_manifest", "write_pair",
    )
    if any(not callable(getattr(workflow, name, None)) for name in required_helpers):
        raise IterationError("maintained self-search helper contract is incomplete")
    compact_parameters = set(inspect.signature(workflow.run_game_chunks).parameters)
    if not {
        "compact_student_runtime", "compact_prior_runtime",
    }.issubset(compact_parameters):
        raise IterationError(
            "self-search generator has no proven compact student/prior contract"
        )
    try:
        identities = workflow._source_identities(REPOSITORY)
    except Exception as error:
        raise IterationError("cannot freeze self-search producer identities") from error
    required_identities = {
        "continuation_source_sha256", "rank4_actor_source_sha256",
        "jacek_nn_actor_source_sha256", "jacek_nn_control_sha256",
        "search_teacher_source_sha256", "rank4_teacher_source_sha256",
        "compact_actor_source_sha256",
    }
    if (not isinstance(identities, Mapping)
            or any(not valid_sha(identities.get(key)) for key in required_identities)):
        raise IterationError("self-search producer identities are incomplete")
    return workflow, {key: str(value) for key, value in identities.items()}


def _load_iteration_authorization(path: pathlib.Path) -> dict[str, Any]:
    campaign = campaign_module()
    try:
        authorization = campaign.validate_iteration_authorization(path)
    except Exception as error:
        raise IterationError("iteration authorization is invalid") from error
    if (authorization.get("status") != "one-iteration-authorized"
            or authorization.get("one_shot") is not True
            or authorization.get("specification") != campaign.ITERATION_SPEC
            or authorization.get("namespace") != NAMESPACE):
        raise IterationError("iteration authorization contract changed")
    return authorization


def launch_agent_document(
    *, label: str, plan_path: pathlib.Path, output_root: pathlib.Path,
    python_path: pathlib.Path,
) -> dict[str, Any]:
    if LABEL_RE.fullmatch(label) is None:
        raise IterationError("LaunchAgent label is invalid")
    stdout = output_root / "launchagent/stdout.log"
    stderr = output_root / "launchagent/stderr.log"
    plan = load_document(plan_path, PLAN_SCHEMA, "LaunchAgent plan")
    environment = single_thread_environment({})
    environment.update({
        "COMPACT_VALUE_BFM_LAUNCHAGENT_LABEL": label,
        "COMPACT_VALUE_BFM_ITERATION_PLAN_BODY_SHA256": plan["body_sha256"],
    })
    return {
        "Label": label,
        "ProgramArguments": [
            str(python_path.absolute()), str(pathlib.Path(__file__).resolve()),
            "execute", "--plan", str(plan_path.resolve()),
            "--output-root", str(output_root.resolve()), "--resume",
        ],
        "WorkingDirectory": str(REPOSITORY.resolve()),
        "EnvironmentVariables": environment,
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ProcessType": "Interactive",
        "LimitLoadToSessionType": "Aqua",
        "StandardOutPath": str(stdout.resolve()),
        "StandardErrorPath": str(stderr.resolve()),
        "LowPriorityIO": False,
        "AbandonProcessGroup": False,
    }


def prepare_plan(
    output_root: pathlib.Path, *, authorization_path: pathlib.Path,
    bundle_manifest: pathlib.Path, family_selection_path: pathlib.Path,
    artifact_root: pathlib.Path, float_checkpoint: pathlib.Path,
    student_runtime: pathlib.Path, generated_source: pathlib.Path,
    previous_compact_runtime: pathlib.Path, roots_tsv: pathlib.Path,
    roots_manifest: pathlib.Path, input_audit: pathlib.Path,
    selfsearch_workflow: pathlib.Path,
    continuation_generator: pathlib.Path, jacek_nn_opponent: pathlib.Path,
    search_teacher: pathlib.Path, rank4_teacher: pathlib.Path,
    pack_tool: pathlib.Path, python_path: pathlib.Path,
    learning_rate: float, label: str,
) -> pathlib.Path:
    authorization = _load_iteration_authorization(authorization_path)
    if not math.isfinite(learning_rate) or not 0 < learning_rate <= MAXIMUM_LEARNING_RATE:
        raise IterationError("sample-scaled learning rate exceeds 6e-5")
    if float(authorization.get("sample_scaled_learning_rate", -1)) != learning_rate:
        raise IterationError("plan learning rate differs from authorization")
    if pack_tool.resolve() != REPLAY_PACK_PATH.resolve():
        raise IterationError(
            "iteration pack tool must be the maintained jacek_replay_pack.py"
        )
    safe_reference = authorization.get("operational_safe_actor")
    if not isinstance(safe_reference, Mapping):
        raise IterationError("authorization has no operational-safe actor")
    try:
        safe_actor = campaign_module().validate_operational_safe_actor(
            pathlib.Path(str(safe_reference.get("path")))
        )
    except Exception as error:
        raise IterationError("authorization operational-safe actor is invalid") from error
    trainer = trainer_module()
    bundle = trainer.FrozenBundle.load(bundle_manifest)
    try:
        bundle_roots_tsv = bundle.artifact_path(bundle.routes.get("roots_tsv"))
        bundle_roots_manifest = bundle.artifact_path(
            bundle.routes.get("roots_manifest")
        )
    except Exception as error:
        raise IterationError("frozen bundle has no exact replay roots") from error
    if (roots_tsv.resolve() != bundle_roots_tsv.resolve()
            or roots_manifest.resolve() != bundle_roots_manifest.resolve()
            or sha256_file(roots_tsv) != sha256_file(bundle_roots_tsv)
            or sha256_file(roots_manifest) != sha256_file(bundle_roots_manifest)):
        raise IterationError("iteration roots are not the frozen bundle roots")
    selection = trainer.validate_selection(
        family_selection_path, artifact_root, bundle
    )
    architecture = trainer.ARCHITECTURES[str(selection["architecture"])]
    trainer.load_float_checkpoint(float_checkpoint, architecture)
    runtime_record = selection.get("runtime")
    offline_rejected = (
        isinstance(selection.get("offline_gate"), Mapping)
        and selection["offline_gate"].get("passed") is False
    )
    safe_selection = safe_actor.get("selection")
    safe_checkpoint = safe_actor.get("float_checkpoint")
    safe_runtime = safe_actor.get("runtime")
    safe_source = safe_actor.get("generated_source")
    if (family_selection_path.is_symlink() or not offline_rejected
            or not isinstance(runtime_record, Mapping)
            or not all(isinstance(value, Mapping) for value in (
                safe_selection, safe_checkpoint, safe_runtime, safe_source
            ))
            or pathlib.Path(str(safe_selection.get("path"))).resolve()
            != family_selection_path.resolve()
            or safe_selection.get("sha256") != sha256_file(family_selection_path)
            or pathlib.Path(str(safe_checkpoint.get("path"))).resolve()
            != float_checkpoint.resolve()
            or safe_checkpoint.get("sha256") != sha256_file(float_checkpoint)
            or pathlib.Path(str(safe_runtime.get("path"))).resolve()
            != student_runtime.resolve()
            or safe_runtime.get("sha256") != sha256_file(student_runtime)
            or pathlib.Path(str(safe_source.get("path"))).resolve()
            != generated_source.resolve()
            or safe_source.get("sha256") != sha256_file(generated_source)
            or sha256_file(student_runtime) != runtime_record.get("sha256")
            or safe_actor.get("architecture")
            != CAMPAIGN_ARCHITECTURES.get(architecture.name)):
        raise IterationError("iteration inputs do not match the authorized safe actor")
    failure_reference = authorization.get("offline_family_failure")
    if not isinstance(failure_reference, Mapping):
        raise IterationError("authorization has no sealed offline-family failure")
    try:
        failure = campaign_module().validate_offline_family_failure(
            pathlib.Path(str(failure_reference.get("path")))
        )
    except Exception as error:
        raise IterationError("authorization offline-family failure is invalid") from error
    rejected = failure.get("rejected_deployable_arms")
    failure_bundle = failure.get("bundle_manifest")
    if (not isinstance(rejected, list) or len(rejected) != 6
            or not isinstance(failure_bundle, Mapping)
            or pathlib.Path(str(failure_bundle.get("path"))).resolve()
            != bundle_manifest.resolve()
            or failure_bundle.get("sha256") != sha256_file(bundle_manifest)):
        raise IterationError("iteration bundle is not the rejected-family bundle")
    student_sha = sha256_file(student_runtime)
    previous_sha = sha256_file(previous_compact_runtime)
    student_matches = [
        row for row in rejected
        if isinstance(row, Mapping)
        and row.get("selection") == safe_selection
        and isinstance(row.get("runtime"), Mapping)
        and pathlib.Path(str(row["runtime"].get("path"))).resolve()
        == student_runtime.resolve()
        and row["runtime"].get("sha256") == student_sha
    ]
    previous_matches = [
        row for row in rejected
        if isinstance(row, Mapping)
        and isinstance(row.get("runtime"), Mapping)
        and pathlib.Path(str(row["runtime"].get("path"))).resolve()
        == previous_compact_runtime.resolve()
        and row["runtime"].get("sha256") == previous_sha
    ]
    if (len(student_matches) != 1
            or previous_compact_runtime.resolve() == student_runtime.resolve()
            or previous_sha == student_sha or len(previous_matches) != 1
            or previous_matches[0].get("selection") == safe_selection):
        raise IterationError(
            "prior compact runtime is not one distinct rejected family actor"
        )
    try:
        trainer.load_runtime(student_runtime)
        trainer.load_runtime(previous_compact_runtime)
    except Exception as error:
        raise IterationError("compact student/prior runtime is invalid") from error
    workflow, source_identities = validate_selfsearch_contract(selfsearch_workflow)
    if sha256_file(jacek_nn_opponent) != source_identities["jacek_nn_control_sha256"]:
        raise IterationError("Jacek-NN opponent is not the maintained exact source")
    teacher_route = bundle.routes.get("teacher_runtime")
    try:
        teacher_runtime = bundle.artifact_path(teacher_route)
    except Exception as error:
        raise IterationError("source bundle has no accepted Search teacher runtime") from error
    inputs = {
        "authorization": file_record(authorization_path),
        "bundle_manifest": file_record(bundle_manifest),
        "family_selection": file_record(family_selection_path),
        "float_checkpoint": file_record(float_checkpoint),
        "student_runtime": file_record(student_runtime),
        "generated_source": file_record(generated_source),
        "previous_compact_runtime": file_record(previous_compact_runtime),
        "roots_tsv": file_record(roots_tsv),
        "roots_manifest": file_record(roots_manifest),
        "input_audit": file_record(input_audit),
        "search_teacher_runtime": file_record(teacher_runtime),
        "jacek_nn_opponent": file_record(jacek_nn_opponent),
    }
    tools = {
        "selfsearch_workflow": file_record(selfsearch_workflow),
        "continuation_generator": file_record(continuation_generator, executable=True),
        "search_teacher": file_record(search_teacher, executable=True),
        "rank4_teacher": file_record(rank4_teacher, executable=True),
        "pack_tool": file_record(pack_tool),
        "python": file_record(
            python_path, executable=True, preserve_lexical_path=True
        ),
        "trainer": file_record(TRAINER_PATH),
        "campaign": file_record(CAMPAIGN_PATH),
        "qualification": file_record(QUALIFICATION_PATH),
        "compact_workflow": file_record(COMPACT_WORKFLOW_PATH),
        "preflight": file_record(PREFLIGHT_PATH),
        "iteration_runner": file_record(pathlib.Path(__file__)),
        "replay_workflow": file_record(REPLAY_WORKFLOW_PATH),
        "replay_corpus": file_record(REPLAY_CORPUS_PATH),
        "replay_features": file_record(REPLAY_FEATURES_PATH),
        "replay_train": file_record(REPLAY_TRAIN_PATH),
        "model_exporter": file_record(MODEL_EXPORTER_PATH),
        "submission_exporter": file_record(SUBMISSION_EXPORTER_PATH),
        "submission_config": file_record(SUBMISSION_CONFIG_PATH),
        "submission_sources": file_record(SUBMISSION_SOURCES_PATH),
    }
    output_root = output_root.resolve()
    plan_directory = output_root / "worker-plans"
    game_plan = exact_game_plan()
    rows = [dict(row) for row in game_plan["rows"]]
    full_plan_json = plan_directory / "full-game-plan.json"
    full_plan_tsv = plan_directory / "full-game-plan.tsv"
    atomic_write_once(full_plan_json, canonical_json_bytes(game_plan))
    atomic_write_once(full_plan_tsv, workflow.render_game_plan_tsv(game_plan))
    workers = []
    for index in range(WORKERS):
        selected = worker_rows(rows, index)
        payload = render_worker_plan(selected)
        path = plan_directory / f"worker-{index:02d}.tsv"
        atomic_write_once(path, payload)
        result_path = output_root / "workers" / f"worker-{index:02d}.result.json"
        command = [
            tools["python"]["path"], str(pathlib.Path(__file__).resolve()),
            "worker", "--plan", str((output_root / "iteration-plan.json").resolve()),
            "--worker-index", str(index), "--output", str(result_path.resolve()),
            "--resume",
        ]
        workers.append({
            "worker_index": index, "game_plan": file_record(path),
            "expected_quotas": worker_quota(selected),
            "game_plan_sha256": sha256_bytes(payload),
            "game_identities_sha256": game_identity_set_sha256(selected),
            "expected_positions": len(selected) * POSITIONS_PER_GAME,
            "expected_deep_relabel_positions": int(
                len(selected) * POSITIONS_PER_GAME * DEEP_RELABEL_FRACTION
            ),
            "expected_games": len(selected), "result_path": str(result_path.resolve()),
            "command": command,
        })
    body = {
        "schema": PLAN_SCHEMA, "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID,
        "authorization": artifact_reference(
            authorization_path, campaign_module().ITERATION_AUTH_SCHEMA
        ),
        "one_shot_plan_binding": str(
            (authorization_path.parent / "plan-binding.json").resolve()
        ),
        "source_bundle_body_sha256": bundle.body_sha256,
        "full_game_plan": {
            "json": file_record(full_plan_json), "tsv": file_record(full_plan_tsv),
        },
        "source_identities": source_identities,
        "selected_architecture": architecture.name,
        "selected_arm": selection["arm"],
        "selected_seed": selection["seed"],
        "learning_rate": learning_rate,
        "specification": {
            "quotas": QUOTAS, "total_games": TOTAL_GAMES,
            "positions_per_game": POSITIONS_PER_GAME, "workers": WORKERS,
            "fixed_work": True,
            "deep_relabel_fraction": DEEP_RELABEL_FRACTION,
            "target_semantics": TARGET_SEMANTICS,
            "fixed_work_configuration": FIXED_WORK_CONFIGURATION,
            "qat_epochs": 4,
        },
        "inputs": inputs, "tools": tools, "workers": workers,
        "protected_tests_opened": False, "iterations_remaining_after_start": 0,
    }
    plan_path = output_root / "iteration-plan.json"
    plan = write_document(plan_path, body)
    write_document(authorization_path.parent / "plan-binding.json", {
        "schema": PLAN_BINDING_SCHEMA, "namespace": NAMESPACE,
        "authorization": artifact_reference(
            authorization_path, campaign_module().ITERATION_AUTH_SCHEMA
        ),
        "plan": artifact_reference(plan_path, PLAN_SCHEMA),
        "one_shot": True, "second_plan_authorized": False,
    })
    plist = launch_agent_document(
        label=label, plan_path=plan_path, output_root=output_root,
        python_path=python_path,
    )
    plist_path = output_root / "launchagent" / f"{label}.plist"
    plist_payload = plistlib.dumps(plist, fmt=plistlib.FMT_XML, sort_keys=True)
    atomic_write_once(plist_path, plist_payload)
    write_document(output_root / "launchagent-reference.json", {
        "schema": "papersoccer.compact-value-bfm.launchagent-reference.v1",
        "namespace": NAMESPACE, "label": label,
        "plan_body_sha256": plan["body_sha256"],
        "plist": file_record(plist_path), "interactive": True,
        "resume": True, "blas_threads": 1,
    })
    return plan_path


def default_power_check(runner: Callable[..., Any] = subprocess.run) -> dict[str, Any]:
    completed = runner(
        ["pmset", "-g", "batt"], text=True, capture_output=True, check=False
    )
    output = str(completed.stdout)
    if completed.returncode != 0 or "AC Power" not in output:
        raise IterationError("iteration requires AC power")
    return {"ac_power": True, "pmset_sha256": sha256_bytes(output.encode())}


def default_disk_check(path: pathlib.Path) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    free = usage.free / (1024 ** 3)
    if free <= 20.0:
        raise IterationError("iteration requires more than 20 GiB free disk")
    return {"free_disk_gib": free, "minimum_gib": 20.0}


def _launchctl_domain(uid: int | None = None) -> str:
    return f"gui/{os.getuid() if uid is None else uid}"


def _launchctl(
    arguments: Sequence[str], *, runner: Callable[..., Any]
) -> Any:
    return runner(
        ["launchctl", *arguments], text=True, capture_output=True, check=False
    )


def install_launch_agent(
    *, plan_path: pathlib.Path, output_root: pathlib.Path,
    launch_agents_directory: pathlib.Path | None = None,
    resume: bool = False,
    runner: Callable[..., Any] = subprocess.run,
    power_check: Callable[[], Mapping[str, Any]] = default_power_check,
    disk_check: Callable[[pathlib.Path], Mapping[str, Any]] = default_disk_check,
) -> dict[str, Any]:
    plan = load_document(plan_path, PLAN_SCHEMA, "iteration plan")
    reference = load_document(
        output_root / "launchagent-reference.json",
        "papersoccer.compact-value-bfm.launchagent-reference.v1",
        "LaunchAgent reference",
    )
    if reference.get("plan_body_sha256") != plan["body_sha256"]:
        raise IterationError("LaunchAgent reference uses another plan")
    power = dict(power_check())
    disk = dict(disk_check(output_root))
    if power.get("ac_power") is not True or float(disk.get("free_disk_gib", 0)) <= 20.0:
        raise IterationError("LaunchAgent power/disk gate failed")
    source_plist = pathlib.Path(reference["plist"]["path"])
    if sha256_file(source_plist) != reference["plist"]["sha256"]:
        raise IterationError("LaunchAgent plist changed")
    label = str(reference["label"])
    target_root = (
        launch_agents_directory
        if launch_agents_directory is not None
        else pathlib.Path.home() / "Library/LaunchAgents"
    )
    target = target_root / f"{label}.plist"
    payload = source_plist.read_bytes()
    if target.exists() and target.read_bytes() != payload:
        raise IterationError("installed LaunchAgent label has different content")
    if not target.exists():
        atomic_write_once(target, payload)
    iteration_root = pathlib.Path(plan["authorization"]["path"]).parents[1]
    started_path = iteration_root / "iteration/01-started.json"
    if started_path.exists() and not resume:
        raise IterationError("iteration LaunchAgent already started; use --resume")
    environment = {
        "interactive_launch_agent": True, "resume": True,
        "blas_threads": 1, "ac_power": True,
        "free_disk_gib": float(disk["free_disk_gib"]),
        "label": label, "plist_sha256": sha256_bytes(payload),
        "plan_body_sha256": plan["body_sha256"],
    }
    campaign = campaign_module()
    if started_path.exists():
        started = qualification_module().load_sealed(
            started_path, campaign.ITERATION_EVENT_SCHEMA
        )
        prior = started.get("environment")
        if (not isinstance(prior, Mapping)
                or any(prior.get(key) != value for key, value in {
                    "interactive_launch_agent": True, "resume": True,
                    "blas_threads": 1, "ac_power": True,
                    "label": label, "plist_sha256": sha256_bytes(payload),
                    "plan_body_sha256": plan["body_sha256"],
                }.items())
                or float(prior.get("free_disk_gib", 0)) <= 20.0):
            raise IterationError("resumed LaunchAgent environment changed")
    else:
        started = campaign.start_iteration(
            iteration_root, environment=environment,
            started_at_utc=utc_now(),
        )
    # The one-shot start receipt must exist before bootstrap/RunAtLoad can
    # execute the long command.
    domain = _launchctl_domain()
    printed = _launchctl(["print", f"{domain}/{label}"], runner=runner)
    already_loaded = printed.returncode == 0
    if not already_loaded:
        boot = _launchctl(["bootstrap", domain, str(target)], runner=runner)
        if boot.returncode != 0:
            raise IterationError("launchctl bootstrap failed after start receipt")
    kick = _launchctl(["kickstart", "-k", f"{domain}/{label}"], runner=runner)
    if kick.returncode != 0:
        raise IterationError("launchctl kickstart failed")
    return {
        "status": "resumed" if already_loaded else "installed",
        "label": label, "domain": domain, "plist": str(target),
        "power": power, "disk": disk,
        "start_body_sha256": started["body_sha256"],
    }


def utc_now() -> str:
    import datetime as dt
    return dt.datetime.now(dt.timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")


def launch_agent_status(
    *, plan_path: pathlib.Path, output_root: pathlib.Path,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    plan = load_document(plan_path, PLAN_SCHEMA, "iteration plan")
    reference = load_document(
        output_root / "launchagent-reference.json",
        "papersoccer.compact-value-bfm.launchagent-reference.v1",
        "LaunchAgent reference",
    )
    if reference.get("plan_body_sha256") != plan["body_sha256"]:
        raise IterationError("LaunchAgent status plan changed")
    label = reference["label"]
    domain = _launchctl_domain()
    completed = _launchctl(["print", f"{domain}/{label}"], runner=runner)
    text = str(completed.stdout) + str(completed.stderr)
    return {
        "loaded": completed.returncode == 0,
        "label": label, "domain": domain,
        "status_sha256": sha256_bytes(text.encode()),
        "execution_claimed": (output_root / "execution-claim.json").exists(),
        "completed": (output_root / "iteration-reference.json").exists(),
    }


def _validate_worker_result(
    path: pathlib.Path, expected: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, Any]:
    result = load_document(path, WORKER_RESULT_SCHEMA, "iteration worker result")
    if (result.get("namespace") != NAMESPACE
            or result.get("campaign_id") != CAMPAIGN_ID
            or result.get("worker_index") != expected["worker_index"]
            or result.get("workers") != WORKERS
            or result.get("games") != expected["expected_games"]
            or result.get("quotas") != expected["expected_quotas"]
            or result.get("game_plan_sha256") != expected["game_plan_sha256"]
            or result.get("game_plan_rows") != expected["expected_games"]
            or result.get("game_identities_sha256")
            != expected["game_identities_sha256"]
            or result.get("positions") != expected["expected_positions"]
            or result.get("deep_relabel_positions")
            != expected["expected_deep_relabel_positions"]
            or result.get("positions_per_game") != POSITIONS_PER_GAME
            or result.get("fixed_work") is not True
            or result.get("fixed_work_configuration") != FIXED_WORK_CONFIGURATION
            or float(result.get("deep_relabel_fraction", -1)) != DEEP_RELABEL_FRACTION
            or result.get("target_semantics") != TARGET_SEMANTICS
            or not isinstance(result.get("compact_actor_bindings"), Mapping)
            or set(result["compact_actor_bindings"]) != {
                f"compact_{role}_{field}"
                for role in ("student", "prior")
                for field in (
                    "runtime_sha256", "runtime_body_sha256", "payload_sha256",
                    "source_bundle_body_sha256", "selection_sha256",
                )
            }
            or any(not valid_sha(value)
                   for value in result["compact_actor_bindings"].values())
            or isinstance(result.get("train_positions"), bool)
            or not isinstance(result.get("train_positions"), int)
            or result.get("train_positions", 0) <= 0
            or result.get("resumed") is not True
            or result.get("plan_body_sha256") != plan["body_sha256"]
            or not isinstance(result.get("train_manifests"), list)
            or not result["train_manifests"]
            or not isinstance(result.get("train_manifest_sha256"), list)
            or len(result["train_manifests"]) != len(result["train_manifest_sha256"])
            or len(result["train_manifests"]) != len(set(result["train_manifests"]))
            or len(result["train_manifest_sha256"])
            != len(set(result["train_manifest_sha256"]))):
        raise IterationError("iteration worker result contradicts exact plan")
    observed_hashes = []
    for manifest in result["train_manifests"]:
        record = file_record(pathlib.Path(manifest))
        observed_hashes.append(record["sha256"])
    if observed_hashes != result["train_manifest_sha256"]:
        raise IterationError("iteration train manifest order/hash binding changed")
    return result


def _verify_file_record(record: object, label: str) -> pathlib.Path:
    if not isinstance(record, Mapping):
        raise IterationError(f"{label} record is missing")
    raw_path = record.get("path")
    if not isinstance(raw_path, str):
        raise IterationError(f"{label} path is missing")
    path = pathlib.Path(raw_path)
    if (not path.is_file()
            or path.stat().st_size != record.get("bytes")
            or sha256_file(path) != record.get("sha256")):
        raise IterationError(f"{label} changed after plan preparation")
    return path


def validate_maintained_python_tool_closure(
    tools: Mapping[str, Any],
) -> dict[str, pathlib.Path]:
    """Re-hash and exact-path-check every repository-owned Python producer."""

    observed: dict[str, pathlib.Path] = {}
    for name, maintained in MAINTAINED_PYTHON_TOOL_PATHS.items():
        path = _verify_file_record(tools.get(name), f"iteration Python tool {name}")
        if path.resolve() != maintained.resolve():
            raise IterationError(
                f"iteration Python tool {name} is not the maintained exact path"
            )
        observed[name] = path
    return observed


def validate_plan_contract(
    plan: Mapping[str, Any], *, plan_path: pathlib.Path,
    output_root: pathlib.Path,
) -> None:
    expected_specification = {
        "quotas": QUOTAS, "total_games": TOTAL_GAMES,
        "positions_per_game": POSITIONS_PER_GAME, "workers": WORKERS,
        "fixed_work": True,
        "deep_relabel_fraction": DEEP_RELABEL_FRACTION,
        "target_semantics": TARGET_SEMANTICS,
        "fixed_work_configuration": FIXED_WORK_CONFIGURATION,
        "qat_epochs": 4,
    }
    inputs = plan.get("inputs")
    tools = plan.get("tools")
    if (plan.get("namespace") != NAMESPACE
            or plan.get("campaign_id") != CAMPAIGN_ID
            or plan.get("specification") != expected_specification
            or plan.get("protected_tests_opened") is not False
            or plan.get("iterations_remaining_after_start") != 0
            or not isinstance(inputs, Mapping)
            or set(inputs) != {
                "authorization", "bundle_manifest", "family_selection",
                "float_checkpoint", "student_runtime", "generated_source",
                "previous_compact_runtime", "roots_tsv", "roots_manifest",
                "input_audit", "search_teacher_runtime", "jacek_nn_opponent",
            }
            or not isinstance(tools, Mapping)
            or set(tools) != {
                "selfsearch_workflow", "continuation_generator", "search_teacher",
                "rank4_teacher", "pack_tool", "python", "trainer", "campaign",
                "qualification", "compact_workflow", "preflight",
                "iteration_runner", "replay_workflow", "replay_corpus",
                "replay_features", "replay_train",
                "model_exporter", "submission_exporter",
                "submission_config", "submission_sources",
            }):
        raise IterationError("iteration plan contract is incomplete")
    for label, record in inputs.items():
        _verify_file_record(record, f"iteration input {label}")
    for label, record in tools.items():
        _verify_file_record(record, f"iteration tool {label}")
    validate_maintained_python_tool_closure(tools)
    workflow, source_identities = validate_selfsearch_contract(
        pathlib.Path(tools["selfsearch_workflow"]["path"])
    )
    if plan.get("source_identities") != source_identities:
        raise IterationError("iteration plan producer source identities changed")
    if (inputs["jacek_nn_opponent"]["sha256"]
            != source_identities["jacek_nn_control_sha256"]):
        raise IterationError("iteration plan does not bind exact Jacek-NN source")
    full = plan.get("full_game_plan")
    if not isinstance(full, Mapping) or set(full) != {"json", "tsv"}:
        raise IterationError("iteration full game plan binding is missing")
    full_json = _verify_file_record(full["json"], "full game plan JSON")
    full_tsv = _verify_file_record(full["tsv"], "full game plan TSV")
    try:
        game_plan = json.loads(full_json.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise IterationError("iteration full game plan JSON is invalid") from error
    if (full_json.read_bytes() != workflow.canonical_json_bytes(game_plan)
            or game_plan != exact_game_plan(int(game_plan.get("seed", -1)))
            or full_tsv.read_bytes() != workflow.render_game_plan_tsv(game_plan)):
        raise IterationError("iteration full game plan changed")
    workers = plan.get("workers")
    if (not isinstance(workers, list) or len(workers) != WORKERS
            or [worker.get("worker_index") for worker in workers]
            != list(range(WORKERS))):
        raise IterationError("iteration worker roster is not exact")
    for index, worker in enumerate(workers):
        selected = worker_rows(game_plan["rows"], index)
        worker_plan = _verify_file_record(
            worker.get("game_plan"), f"worker {index} game plan"
        )
        result_path = require_within(
            pathlib.Path(str(worker.get("result_path"))), output_root,
            f"worker {index} result",
        )
        expected_command = [
            tools["python"]["path"], str(pathlib.Path(__file__).resolve()),
            "worker", "--plan", str(plan_path.resolve()),
            "--worker-index", str(index), "--output", str(result_path), "--resume",
        ]
        if (worker.get("expected_games") != len(selected)
                or worker.get("expected_quotas") != worker_quota(selected)
                or worker.get("game_plan_sha256") != sha256_file(worker_plan)
                or worker.get("game_identities_sha256")
                != game_identity_set_sha256(selected)
                or worker.get("expected_positions")
                != len(selected) * POSITIONS_PER_GAME
                or worker.get("expected_deep_relabel_positions")
                != len(selected) * POSITIONS_PER_GAME // 4
                or worker.get("command") != expected_command
                or _worker_plan_rows(worker_plan) != selected):
            raise IterationError(f"iteration worker {index} plan changed")


def _worker_plan_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="ascii").splitlines()
    if lines[:1] != ["game_ordinal\tactor_mode\tbase_seed"]:
        raise IterationError("worker game plan header changed")
    rows = []
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) != 3:
            raise IterationError("worker game plan row is malformed")
        try:
            game_ordinal, base_seed = int(fields[0]), int(fields[2])
        except ValueError as error:
            raise IterationError("worker game plan integer is malformed") from error
        if fields[1] not in QUOTAS or game_ordinal < 0 or base_seed < 0:
            raise IterationError("worker game plan identity is invalid")
        rows.append({
            "game_ordinal": game_ordinal,
            "actor_mode": fields[1],
            "base_seed": base_seed,
        })
    return rows


def _bound_train_manifest(
    *, workflow: Any, source: pathlib.Path, plan: Mapping[str, Any],
    expected: Mapping[str, Any], positions: int, deep_positions: int,
) -> pathlib.Path:
    try:
        manifest = json.loads(source.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise IterationError("self-search train manifest is invalid") from error
    if (not isinstance(manifest, dict) or manifest.get("split") != "train"
            or isinstance(manifest.get("samples"), bool)
            or not isinstance(manifest.get("samples"), int)
            or manifest.get("samples", 0) <= 0):
        raise IterationError("self-search train manifest has no train samples")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise IterationError("self-search train manifest has no provenance")
    policies = provenance.get("target_policies")
    if (not isinstance(policies, list) or not policies
            or any(policy.get("mixture") != {
                "teacher_weight": 0.75,
                "outcome_weight": 0.25,
                "outcome_frame": "mover-relative-terminal-winner",
            } for policy in policies if isinstance(policy, Mapping))
            or any(not isinstance(policy, Mapping) for policy in policies)):
        raise IterationError("self-search target mixture is not exact 75/25")
    manifest = dict(manifest)
    manifest["provenance"] = {
        **provenance,
        "compact_value_bfm_iteration": {
            "plan_body_sha256": plan["body_sha256"],
            "worker_index": expected["worker_index"],
            "game_plan_sha256": expected["game_plan_sha256"],
            "game_plan_rows": expected["expected_games"],
            "game_identities_sha256": expected["game_identities_sha256"],
            "fixed_work_configuration": FIXED_WORK_CONFIGURATION,
            "positions_per_game": POSITIONS_PER_GAME,
            "deep_relabel_fraction": DEEP_RELABEL_FRACTION,
            "hard_fraction": [1, 4],
            "target_semantics": TARGET_SEMANTICS,
            "all_positions": positions,
            "all_deep_relabel_positions": deep_positions,
            "train_positions": manifest["samples"],
            "source_manifest_sha256": sha256_file(source),
        },
    }
    payload = workflow.canonical_json_bytes(manifest)
    path = source.parent / f"{sha256_bytes(payload)}.json"
    atomic_write_once(path, payload)
    return path


def run_iteration_worker(
    *, plan_path: pathlib.Path, worker_index: int,
    result_path: pathlib.Path, resume: bool,
) -> dict[str, Any]:
    if not resume:
        raise IterationError("iteration workers always require --resume")
    plan = load_document(plan_path, PLAN_SCHEMA, "iteration plan")
    validate_plan_contract(
        plan, plan_path=plan_path, output_root=result_path.parent.parent
    )
    if (plan.get("namespace") != NAMESPACE
            or plan.get("campaign_id") != CAMPAIGN_ID
            or not isinstance(plan.get("workers"), list)
            or [worker.get("worker_index") for worker in plan["workers"]]
            != list(range(WORKERS))):
        raise IterationError("iteration worker roster changed")
    if not 0 <= worker_index < WORKERS:
        raise IterationError("iteration worker index is invalid")
    expected = plan["workers"][worker_index]
    if pathlib.Path(expected["result_path"]).resolve() != result_path.resolve():
        raise IterationError("iteration worker output differs from plan")
    if result_path.exists():
        return _validate_worker_result(result_path, expected, plan)
    inputs = plan.get("inputs")
    tools = plan.get("tools")
    if not isinstance(inputs, Mapping) or not isinstance(tools, Mapping):
        raise IterationError("iteration worker inputs/tools are missing")
    workflow_path = _verify_file_record(
        tools.get("selfsearch_workflow"), "self-search workflow"
    )
    workflow, source_identities = validate_selfsearch_contract(workflow_path)
    generated_source = _verify_file_record(
        inputs.get("generated_source"), "generated candidate source"
    )
    if source_identities != plan.get("source_identities"):
        raise IterationError("self-search source identities changed after preparation")
    jacek_nn = _verify_file_record(
        inputs.get("jacek_nn_opponent"), "exact Jacek-NN opponent"
    )
    if sha256_file(jacek_nn) != source_identities["jacek_nn_control_sha256"]:
        raise IterationError("exact Jacek-NN opponent changed")
    student_runtime = _verify_file_record(
        inputs.get("student_runtime"), "compact student runtime"
    )
    prior_runtime = _verify_file_record(
        inputs.get("previous_compact_runtime"), "compact prior runtime"
    )
    roots_tsv = _verify_file_record(inputs.get("roots_tsv"), "replay roots TSV")
    roots_manifest = _verify_file_record(
        inputs.get("roots_manifest"), "replay roots manifest"
    )
    teacher_runtime = _verify_file_record(
        inputs.get("search_teacher_runtime"), "accepted Search teacher runtime"
    )
    continuation_generator = _verify_file_record(
        tools.get("continuation_generator"), "continuation generator"
    )
    search_teacher = _verify_file_record(tools.get("search_teacher"), "Search teacher")
    rank4_teacher = _verify_file_record(tools.get("rank4_teacher"), "Rank-4 teacher")
    pack_tool = _verify_file_record(tools.get("pack_tool"), "pack tool")
    python = _verify_file_record(tools.get("python"), "Python interpreter")
    game_plan_tsv = _verify_file_record(expected.get("game_plan"), "worker game plan")
    rows = _worker_plan_rows(game_plan_tsv)
    if (len(rows) != expected["expected_games"]
            or worker_quota(rows) != expected["expected_quotas"]
            or sha256_file(game_plan_tsv) != expected["game_plan_sha256"]
            or game_identity_set_sha256(rows) != expected["game_identities_sha256"]):
        raise IterationError("worker game plan no longer matches its receipt binding")
    worker_root = result_path.parent / f"worker-{worker_index:02d}-pipeline"
    worker_root.mkdir(parents=True, exist_ok=True)
    worker_plan = {
        "schema": workflow.GAME_PLAN_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "seed": json.loads(
            _verify_file_record(
                plan["full_game_plan"]["json"], "full game plan"
            ).read_bytes()
        )["seed"],
        "quotas": expected["expected_quotas"],
        "games": expected["expected_games"],
        "rows": rows,
    }
    worker_plan_json = worker_root / "game-plan.json"
    atomic_write_once(worker_plan_json, workflow.canonical_json_bytes(worker_plan))
    configuration = dict(workflow.FULL_CONFIGURATION)
    configuration.update({"games": expected["expected_games"], "game_workers": 1})
    spec = workflow.PhaseSpec(
        name=f"iteration-worker-{worker_index:02d}", campaign_id=CAMPAIGN_ID,
        configuration=configuration, quotas=expected["expected_quotas"],
        game_seed=int(worker_plan["seed"]), opening_seed=0, pairs=0,
        gate_time_ms=0, gate_workers=0, bank_classification="unprotected-iteration",
    )
    manager = workflow.StageManager(
        output=worker_root, campaign_id=CAMPAIGN_ID,
        round_index=worker_index, resume=True,
        environment={
            "namespace": NAMESPACE,
            "plan_body_sha256": plan["body_sha256"],
            "worker_index": worker_index,
            "single_thread": True,
            "fixed_work_configuration": FIXED_WORK_CONFIGURATION,
        },
    )

    def guarded_python_action(action: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        # Workers can run for days.  Recheck the complete Python producer
        # closure on both sides of every stage so a mid-run source edit cannot
        # silently influence later split, label, or pack work.
        validate_maintained_python_tool_closure(tools)
        value = action()
        validate_maintained_python_tool_closure(tools)
        return value

    games_tsv = worker_root / "games.tsv"
    games_manifest = worker_root / "games.manifest.json"
    game_result = manager.execute(
        ordinal=1, name="games",
        configuration={
            "global_workers": WORKERS, "this_worker_threads": 1,
            "game_chunk_size": 25,
            "fixed_work": FIXED_WORK_CONFIGURATION["game_generation"],
        },
        producers={"workflow": workflow_path,
                   "continuation_generator": continuation_generator},
        inputs={"plan": worker_plan_json, "plan_tsv": game_plan_tsv,
                "roots": roots_tsv, "student_runtime": student_runtime,
                "prior_runtime": prior_runtime, "candidate_source": generated_source,
                "jacek_nn": jacek_nn},
        outputs={"games": games_tsv, "manifest": games_manifest},
        resumable_outputs={"games", "manifest"},
        action=lambda: guarded_python_action(
            lambda: workflow.run_game_chunks(
                manager=manager, stage_ordinal=1, spec=spec,
                plan_path=worker_plan_json, roots_tsv=roots_tsv,
                actor=student_runtime, diversity=prior_runtime,
                generator=continuation_generator, workers=1,
                source_identities=source_identities,
                compact_student_runtime=student_runtime,
                compact_prior_runtime=prior_runtime,
            )
        ),
        validator=lambda value: workflow._validate_chunk_stage_result(
            value, "iteration game stage"
        ),
    )
    if game_result.get("games") != expected["expected_games"]:
        raise IterationError("iteration worker did not finish every planned game")
    compact_bindings = game_result.get("compact_actor_bindings")
    expected_binding_keys = {
        f"compact_{role}_{field}"
        for role in ("student", "prior")
        for field in (
            "runtime_sha256", "runtime_body_sha256", "payload_sha256",
            "source_bundle_body_sha256", "selection_sha256",
        )
    }
    if (not isinstance(compact_bindings, Mapping)
            or set(compact_bindings) != expected_binding_keys
            or compact_bindings.get("compact_student_runtime_sha256")
            != sha256_file(student_runtime)
            or compact_bindings.get("compact_prior_runtime_sha256")
            != sha256_file(prior_runtime)
            or any(not valid_sha(value) for value in compact_bindings.values())):
        raise IterationError("iteration compact actor bindings are incomplete")
    positions_tsv = worker_root / "positions.tsv"
    positions_manifest = worker_root / "positions.manifest.json"

    def freeze() -> dict[str, Any]:
        payload, manifest = workflow.freeze_positions(
            campaign_id=CAMPAIGN_ID, games_tsv=games_tsv,
            games_manifest=games_manifest, roots_manifest=roots_manifest,
            maximum_per_game=POSITIONS_PER_GAME,
        )
        workflow.write_pair(payload, manifest, positions_tsv, positions_manifest)
        return {"positions": manifest["positions"],
                "split_counts": manifest["split_counts"]}

    position_result = manager.execute(
        ordinal=2, name="positions",
        configuration={"maximum_per_game": POSITIONS_PER_GAME},
        producers={"workflow": workflow_path},
        inputs={"games": games_tsv, "game_manifest": games_manifest,
                "roots": roots_manifest},
        outputs={"positions": positions_tsv, "manifest": positions_manifest},
        resumable_outputs={"positions", "manifest"},
        action=lambda: guarded_python_action(freeze),
    )
    expected_positions = expected["expected_positions"]
    if position_result.get("positions") != expected_positions:
        raise IterationError("iteration did not freeze exactly 20 positions per game")
    shallow_search = worker_root / "labels/search-shallow.jsonl"
    shallow_rank4 = worker_root / "labels/rank4-shallow.jsonl"
    deep_search = worker_root / "labels/search-deep.jsonl"
    deep_rank4 = worker_root / "labels/rank4-deep.jsonl"
    hard_positions = worker_root / "hard-positions.tsv"
    hard_manifest = worker_root / "hard-positions.manifest.json"

    def labels(
        ordinal: int, name: str, source: pathlib.Path, output: pathlib.Path,
        teacher: pathlib.Path, schema: str, nodes: int,
        model: pathlib.Path | None = None,
    ) -> dict[str, Any]:
        identity_key = (
            "search_teacher_source_sha256"
            if schema == workflow.SEARCH_TEACHER_SCHEMA
            else "rank4_teacher_source_sha256"
        )
        return manager.execute(
            ordinal=ordinal, name=name,
            configuration={
                "nodes": nodes, "global_workers": WORKERS,
                "this_worker_threads": 1, "fixed_work_time_ms": 0,
                "source_sha256": source_identities[identity_key],
            },
            producers={"workflow": workflow_path, "teacher": teacher},
            inputs={"positions": source, **({"model": model} if model else {})},
            outputs={"labels": output}, resumable_outputs={"labels"},
            action=lambda: guarded_python_action(
                lambda: workflow.run_label_chunks(
                    manager=manager, stage_ordinal=ordinal, stage_name=name,
                    positions=source, output=output, teacher=teacher, schema=schema,
                    campaign_id=CAMPAIGN_ID, nodes=nodes, workers=1,
                    model=model, source_sha256=source_identities[identity_key],
                )
            ),
            validator=lambda value: workflow._validate_chunk_stage_result(
                value, f"iteration {name} stage"
            ),
        )

    labels(3, "search-shallow", positions_tsv, shallow_search,
           search_teacher, workflow.SEARCH_TEACHER_SCHEMA, 64_000,
           teacher_runtime)
    labels(4, "rank4-shallow", positions_tsv, shallow_rank4,
           rank4_teacher, workflow.RANK4_TEACHER_SCHEMA, 32_000)

    def hard_selection() -> dict[str, Any]:
        payload, manifest = workflow.select_hard_positions(
            positions_tsv=positions_tsv, search_labels=shallow_search,
            rank4_labels=shallow_rank4, numerator=1, denominator=4,
        )
        workflow.write_pair(payload, manifest, hard_positions, hard_manifest)
        return {"selected": manifest["selected"], "games": manifest["games"]}

    hard_result = manager.execute(
        ordinal=5, name="hard-selection", configuration={"fraction": [1, 4]},
        producers={"workflow": workflow_path},
        inputs={"positions": positions_tsv, "search": shallow_search,
                "rank4": shallow_rank4},
        outputs={"positions": hard_positions, "manifest": hard_manifest},
        resumable_outputs={"positions", "manifest"},
        action=lambda: guarded_python_action(hard_selection),
    )
    expected_deep = expected["expected_deep_relabel_positions"]
    if hard_result.get("selected") != expected_deep:
        raise IterationError("iteration hard selection is not exactly 25 percent")
    labels(6, "search-deep", hard_positions, deep_search,
           search_teacher, workflow.SEARCH_TEACHER_SCHEMA, 500_000,
           teacher_runtime)
    labels(7, "rank4-deep", hard_positions, deep_rank4,
           rank4_teacher, workflow.RANK4_TEACHER_SCHEMA, 400_000)
    merged_search = worker_root / "labels/search-merged.jsonl"
    merged_rank4 = worker_root / "labels/rank4-merged.jsonl"

    def merge_labels(
        shallow: pathlib.Path, deep: pathlib.Path, output: pathlib.Path,
        schema: str,
    ) -> dict[str, Any]:
        payload = workflow.merge_deep_labels(
            shallow=shallow, deep=deep, expected_schema=schema
        )
        atomic_write_once(output, payload)
        return {"rows": len(payload.splitlines())}

    for ordinal, name, shallow, deep, output, schema in (
        (8, "search-targets", shallow_search, deep_search,
         merged_search, workflow.SEARCH_TEACHER_SCHEMA),
        (9, "rank4-targets", shallow_rank4, deep_rank4,
         merged_rank4, workflow.RANK4_TEACHER_SCHEMA),
    ):
        merged = manager.execute(
            ordinal=ordinal, name=name, configuration={"deep_override": True},
            producers={"workflow": workflow_path},
            inputs={"shallow": shallow, "deep": deep}, outputs={"labels": output},
            resumable_outputs={"labels"},
            action=lambda shallow=shallow, deep=deep, output=output, schema=schema:
                guarded_python_action(
                    lambda: merge_labels(shallow, deep, output, schema)
                ),
        )
        if merged.get("rows") != expected_positions:
            raise IterationError("merged teacher targets do not cover every position")
    pack_directory = worker_root / "shards/search"
    pack_report = pack_directory / "pack-report.json"
    manager.execute(
        ordinal=10, name="pack-search",
        configuration={"streaming": True, "prior_shards": 0},
        producers={"workflow": workflow_path, "pack": pack_tool},
        inputs={"roots": roots_manifest, "labels": merged_search},
        outputs={"report": pack_report}, resumable_outputs={"report"},
        action=lambda: guarded_python_action(
            lambda: workflow.run_pack(
                python=python, pack_tool=pack_tool, roots=roots_manifest,
                labels=merged_search, output_directory=pack_directory,
            )
        ),
        validator=lambda _value: workflow._validate_pack_report(
            pack_report, roots=roots_manifest, labels=merged_search,
        ),
    )
    source_manifest = workflow._pack_manifest(pack_report, "train")
    bound_manifest = _bound_train_manifest(
        workflow=workflow, source=source_manifest, plan=plan,
        expected=expected, positions=expected_positions,
        deep_positions=expected_deep,
    )
    train_manifest = json.loads(bound_manifest.read_bytes())
    body = {
        "schema": WORKER_RESULT_SCHEMA, "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID, "worker_index": worker_index,
        "workers": WORKERS, "game_plan_sha256": expected["game_plan_sha256"],
        "game_plan_rows": expected["expected_games"],
        "game_identities_sha256": expected["game_identities_sha256"],
        "games": expected["expected_games"], "quotas": expected["expected_quotas"],
        "positions": expected_positions,
        "deep_relabel_positions": expected_deep,
        "train_positions": train_manifest["samples"],
        "positions_per_game": POSITIONS_PER_GAME, "fixed_work": True,
        "fixed_work_configuration": FIXED_WORK_CONFIGURATION,
        "deep_relabel_fraction": DEEP_RELABEL_FRACTION,
        "target_semantics": TARGET_SEMANTICS,
        "compact_actor_bindings": dict(compact_bindings),
        "resumed": True, "plan_body_sha256": plan["body_sha256"],
        "train_manifests": [str(bound_manifest.resolve())],
        "train_manifest_sha256": [sha256_file(bound_manifest)],
    }
    write_document(result_path, body)
    return _validate_worker_result(result_path, expected, plan)


def _run_worker(
    worker: Mapping[str, Any], *, plan: Mapping[str, Any],
    resume: bool, runner: Callable[..., Any],
) -> dict[str, Any]:
    result_path = pathlib.Path(worker["result_path"])
    if result_path.exists():
        if not resume:
            raise IterationError("completed iteration worker requires --resume")
        return _validate_worker_result(result_path, worker, plan)
    completed = runner(
        list(worker["command"]), cwd=REPOSITORY,
        env=single_thread_environment(), text=True,
        capture_output=True, check=False,
    )
    if completed.returncode != 0 or not result_path.is_file():
        raise IterationError(
            f"iteration worker {worker['worker_index']} failed "
            f"(stderr_sha256={sha256_bytes(str(completed.stderr).encode())})"
        )
    return _validate_worker_result(result_path, worker, plan)


def _external_dataset(
    manifest_paths: Sequence[pathlib.Path], compact: Any, *,
    plan: Mapping[str, Any], worker_results: Sequence[Mapping[str, Any]],
) -> Any:
    large = _load_module(REPLAY_TRAIN_PATH, "compact_iteration_large_loader")
    datasets = []
    owners = {
        str(pathlib.Path(path).resolve()): result
        for result in worker_results for path in result["train_manifests"]
    }
    positions_by_worker = Counter()
    for path in manifest_paths:
        manifest = json.loads(path.read_bytes())
        policies = manifest.get("provenance", {}).get("target_policies")
        iteration = manifest.get("provenance", {}).get(
            "compact_value_bfm_iteration"
        )
        owner = owners.get(str(path.resolve()))
        if (manifest.get("split") != "train" or not isinstance(policies, list)
                or not policies
                or any(policy.get("mixture") != {
                    "teacher_weight": 0.75, "outcome_weight": 0.25,
                    "outcome_frame": "mover-relative-terminal-winner",
                } for policy in policies)
                or not isinstance(owner, Mapping)
                or not isinstance(iteration, Mapping)
                or iteration.get("plan_body_sha256") != plan["body_sha256"]
                or iteration.get("worker_index") != owner["worker_index"]
                or iteration.get("game_plan_sha256") != owner["game_plan_sha256"]
                or iteration.get("game_plan_rows") != owner["game_plan_rows"]
                or iteration.get("game_identities_sha256")
                != owner["game_identities_sha256"]
                or iteration.get("fixed_work_configuration")
                != FIXED_WORK_CONFIGURATION
                or iteration.get("positions_per_game") != POSITIONS_PER_GAME
                or float(iteration.get("deep_relabel_fraction", -1))
                != DEEP_RELABEL_FRACTION
                or iteration.get("hard_fraction") != [1, 4]
                or iteration.get("all_positions") != owner["positions"]
                or iteration.get("all_deep_relabel_positions")
                != owner["deep_relabel_positions"]
                or iteration.get("target_semantics") != TARGET_SEMANTICS):
            raise IterationError("on-policy train manifest target semantics changed")
        positions = iteration.get("train_positions")
        if (isinstance(positions, bool) or not isinstance(positions, int)
                or positions <= 0 or manifest.get("samples") != positions):
            raise IterationError("on-policy manifest position counts are invalid")
        shard = large.load_csr_shard(path)
        if len(shard.targets) != positions:
            raise IterationError("on-policy manifest position count differs from shard")
        positions_by_worker[owner["worker_index"]] += positions
        datasets.append(compact.Dataset(
            indptr=shard.indptr, indices=shard.indices,
            targets=shard.targets, weights=shard.weights,
            group_ids=shard.group_ids, split="train",
            source_manifest_sha256=sha256_file(path),
            source_npz_sha256=shard.npz_sha256,
            source_route=str(path.resolve()),
        ))
    for result in worker_results:
        worker = result["worker_index"]
        if positions_by_worker[worker] != result["train_positions"]:
            raise IterationError("on-policy manifest totals differ from worker receipt")
    return compact.concatenate_datasets(datasets, split="train")


def gate_feasibility_key(
    compact: Any,
    float_metrics: Mapping[str, Mapping[str, float | int]],
    quantized_metrics: Mapping[str, Mapping[str, float | int]],
) -> tuple[float, ...]:
    gate = compact.offline_advancement_gate(float_metrics, quantized_metrics)
    violations = []
    for name, minimum_sign, maximum_huber in (
        ("common_adjudicator", compact.COMMON_MINIMUM_SIGN,
         compact.COMMON_MAXIMUM_HUBER),
        ("canonical_validation", compact.CANONICAL_MINIMUM_SIGN,
         compact.CANONICAL_MAXIMUM_HUBER),
    ):
        base = float_metrics[name]
        candidate = quantized_metrics[name]
        sign = float(candidate["sign_accuracy"])
        huber = float(candidate["weighted_huber"])
        base_sign = float(base["sign_accuracy"])
        base_huber = float(base["weighted_huber"])
        if not all(math.isfinite(value) for value in (
            sign, huber, base_sign, base_huber,
        )):
            return (1.0, math.inf, math.inf, math.inf,
                    math.inf, math.inf, math.inf, math.inf)
        values = (
            max(0.0, (minimum_sign - sign) / max(minimum_sign, 1e-12)),
            max(0.0, huber / maximum_huber - 1.0),
            max(0.0, (base_sign - sign) / compact.MAXIMUM_SIGN_LOSS - 1.0),
            max(0.0, huber / max(
                base_huber * compact.MAXIMUM_HUBER_RATIO, 1e-12
            ) - 1.0),
        )
        violations.extend(values)
    if any(not math.isfinite(value) for value in violations):
        violations = [math.inf]
    common = quantized_metrics["common_adjudicator"]
    canonical = quantized_metrics["canonical_validation"]
    objective = (
        float(common["objective_weighted_huber"]),
        float(canonical["objective_weighted_huber"]),
        -float(common["sign_accuracy"]),
        -float(canonical["sign_accuracy"]),
    )
    return (
        0.0 if gate["passed"] else 1.0,
        max(violations, default=0.0),
        sum(violations),
        float(len(gate["errors"])),
        *(float(value) for value in objective),
    )


def run_gate_aware_fixed_scale_qat(
    compact: Any, float_result: Any, inputs: Any,
    architecture: Any, arm: Any, seed: int,
) -> Any:
    original_key = compact._validation_key

    def selection_key(metrics: Mapping[str, Mapping[str, float | int]]) -> tuple[float, ...]:
        return gate_feasibility_key(compact, float_result.metrics, metrics)

    try:
        compact._validation_key = selection_key
        result = compact.run_fixed_scale_qat(
            float_result, inputs, architecture, arm, seed,
            qat_epochs=compact.QAT_EPOCHS,
        )
    finally:
        compact._validation_key = original_key
    result.report["iteration_selection_policy"] = {
        "primary": "offline-gate-feasibility",
        "secondary": "maximum-then-sum-normalized-gate-violation",
        "tertiary": "gate-error-count-then-original-validation-key",
        "qat_epochs": compact.QAT_EPOCHS,
        "fixed_scales": True,
        "seed_policy": "authorization-bound-single-seed",
    }
    return result


def render_iteration_source(
    *, runtime: pathlib.Path, plan: Mapping[str, Any], output_root: pathlib.Path,
) -> tuple[pathlib.Path, dict[str, Any]]:
    tools = plan.get("tools")
    if not isinstance(tools, Mapping):
        raise IterationError("iteration plan has no exporter bindings")
    validate_maintained_python_tool_closure(tools)
    model_exporter_path = _verify_file_record(
        tools.get("model_exporter"), "compact model exporter"
    )
    submission_exporter_path = _verify_file_record(
        tools.get("submission_exporter"), "compact submission exporter"
    )
    _verify_file_record(tools.get("submission_config"), "compact submission config")
    _verify_file_record(tools.get("submission_sources"), "compact submission sources")
    if (model_exporter_path.resolve() != MODEL_EXPORTER_PATH.resolve()
            or submission_exporter_path.resolve()
            != SUBMISSION_EXPORTER_PATH.resolve()):
        raise IterationError("iteration source exporter is not the maintained exact tool")
    model_exporter = _load_module(
        model_exporter_path, "compact_iteration_model_exporter"
    )
    submission_exporter = _load_module(
        submission_exporter_path, "compact_iteration_submission_exporter"
    )
    try:
        header, metadata = model_exporter.render_header(runtime)
        _default_output, payload = submission_exporter.render(model_header=header)
        payload.decode("ascii")
    except Exception as error:
        raise IterationError("post-iteration compact source export failed") from error
    validate_maintained_python_tool_closure(tools)
    if (metadata.get("file_sha256") != sha256_file(runtime)
            or metadata.get("body_sha256") is None
            or len(payload) >= 95_000):
        raise IterationError(
            "post-iteration generated source is not strictly below 95,000 ASCII bytes"
        )
    source_path = (
        output_root / "fine-tune/generated-sources"
        / f"{sha256_bytes(payload)}.submission.cpp"
    )
    atomic_write_once(source_path, payload)
    return source_path, {
        "runtime_sha256": sha256_file(runtime),
        "runtime_body_sha256": metadata["body_sha256"],
        "model_header_sha256": metadata["header_sha256"],
        "source_sha256": sha256_file(source_path),
        "source_ascii_bytes": len(payload),
        "source_limit_exclusive": 95_000,
    }


def fine_tune_and_select(
    *, plan: Mapping[str, Any], worker_results: Sequence[Mapping[str, Any]],
    output_root: pathlib.Path,
) -> pathlib.Path:
    tools = plan.get("tools")
    if not isinstance(tools, Mapping):
        raise IterationError("iteration plan has no fine-tune tool closure")
    validate_maintained_python_tool_closure(tools)
    reference_path = output_root / "fine-tune/selection-reference.json"
    if reference_path.exists():
        reference = load_document(
            reference_path, FINE_TUNE_REFERENCE_SCHEMA,
            "iteration fine-tune reference",
        )
        selection_record = reference.get("selection")
        if (reference.get("plan_body_sha256") != plan["body_sha256"]
                or not isinstance(selection_record, Mapping)):
            raise IterationError("fine-tune reference uses another plan")
        selected = pathlib.Path(str(selection_record.get("path")))
        if (not selected.is_file()
                or sha256_file(selected) != selection_record.get("sha256")):
            raise IterationError("fine-tune referenced selection changed")
        load_document(selected, SELECTION_SCHEMA, "iteration selection")
        return selected
    compact = trainer_module()
    bundle = compact.FrozenBundle.load(
        pathlib.Path(plan["inputs"]["bundle_manifest"]["path"])
    )
    base_inputs = compact.load_training_inputs(
        bundle, "search-target",
        input_audit=pathlib.Path(plan["inputs"]["input_audit"]["path"]),
    )
    manifest_paths = [
        pathlib.Path(path)
        for result in worker_results for path in result["train_manifests"]
    ]
    new_dataset = _external_dataset(
        manifest_paths, compact, plan=plan, worker_results=worker_results
    )
    inputs = compact.dataclasses.replace(base_inputs, new=new_dataset)
    split_isolation = compact.validate_unprotected_split_isolation(
        inputs.new, inputs.anchor,
        inputs.common_adjudicator, inputs.canonical_validation,
    )
    inputs = compact.dataclasses.replace(inputs, split_isolation=split_isolation)
    architecture = compact.ARCHITECTURES[plan["selected_architecture"]]
    seed = int(plan["selected_seed"])
    parameters = compact.load_float_checkpoint(
        pathlib.Path(plan["inputs"]["float_checkpoint"]["path"]), architecture
    )
    optimizer = compact.AdamW(
        parameters, learning_rate=float(plan["learning_rate"]),
        weight_decay=compact.WEIGHT_DECAY,
    )
    arm = compact.ARMS["search-target"]
    coverage_epoch = compact.anchor_coverage_complete_epoch(
        len(inputs.new), len(inputs.anchor)
    )
    best = None
    best_metrics = None
    best_key = None
    best_epoch = 0
    last_progress = coverage_epoch
    history = []
    for epoch in range(1, compact.MAX_FLOAT_EPOCHS + 1):
        losses = []
        for new_rows, anchor_rows in compact.mixed_epoch_batches(
            len(inputs.new), len(inputs.anchor), seed=seed, epoch=epoch
        ):
            losses.append(compact._train_mixed_batch(
                parameters, architecture, arm, optimizer, inputs,
                new_rows, anchor_rows,
            ))
        metrics = compact.evaluate_validation_pair(
            parameters, architecture, inputs, arm
        )
        key = compact._validation_key(metrics)
        complete = epoch >= coverage_epoch
        eligible = complete and (best_key is None or key < best_key)
        history.append({"epoch": epoch, "loss": float(sum(losses) / len(losses)),
                        "validation": metrics, "eligible": eligible})
        if eligible:
            best = {name: value.copy() for name, value in parameters.items()}
            best_metrics, best_key, best_epoch = metrics, key, epoch
            last_progress = epoch
        if complete and epoch - last_progress >= compact.PATIENCE:
            break
    if best is None or best_metrics is None:
        raise IterationError("fine-tuning produced no anchor-covered checkpoint")
    float_result = compact.FloatTrainingResult(
        parameters=best, epoch=best_epoch, metrics=best_metrics,
        report={"best_float_epoch": best_epoch, "history": history,
                "initialization": "authorized-selected-float-checkpoint",
                "learning_rate": plan["learning_rate"]},
    )
    quantized = run_gate_aware_fixed_scale_qat(
        compact, float_result, inputs, architecture, arm, seed
    )
    gate = compact.offline_advancement_gate(best_metrics, quantized.metrics)
    checkpoint = compact.write_float_checkpoint(
        output_root / "fine-tune/float-checkpoints", best, architecture
    )
    manifest_records = [file_record(path) for path in manifest_paths]
    iteration_training_body_sha256 = sha256_bytes(canonical_json_bytes({
        "source_bundle_body_sha256": bundle.body_sha256,
        "plan_body_sha256": plan["body_sha256"],
        "manifests": [
            {"sha256": record["sha256"], "bytes": record["bytes"]}
            for record in manifest_records
        ],
    }))
    runtime = compact.write_runtime(
        output_root / "fine-tune/quantized-runtimes",
        architecture, quantized.quantized,
        arm="search-target", seed=seed, float_epoch=best_epoch,
        qat_epoch=quantized.qat_epoch,
        source_bundle_body_sha256=bundle.body_sha256,
    )
    generated_source, source_export = render_iteration_source(
        runtime=runtime, plan=plan, output_root=output_root
    )
    body = {
        "schema": SELECTION_SCHEMA, "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID,
        "plan_body_sha256": plan["body_sha256"],
        "architecture": architecture.name, "seed": seed,
        "float_epoch": best_epoch, "qat_epoch": quantized.qat_epoch,
        "learning_rate": plan["learning_rate"],
        "new_train_manifests": manifest_records,
        "iteration_training_body_sha256": iteration_training_body_sha256,
        "split_isolation": split_isolation,
        "float_checkpoint": file_record(checkpoint),
        "runtime": file_record(runtime),
        "generated_source": file_record(generated_source),
        "source_export": source_export,
        "float_validation": best_metrics,
        "quantized_validation": quantized.metrics,
        "quantized_selection_policy": quantized.report[
            "iteration_selection_policy"
        ],
        "offline_gate": gate,
        "status": gate["status"],
        "protected_tests_opened": False,
        "handoff": "existing-fixed-scale-qAT-and-offline-selection",
    }
    selection_path = write_content_addressed(
        output_root / "fine-tune/selections", body, ".iteration-selection.json"
    )
    write_document(reference_path, {
        "schema": FINE_TUNE_REFERENCE_SCHEMA, "namespace": NAMESPACE,
        "plan_body_sha256": plan["body_sha256"],
        "selection": {
            "path": str(selection_path.resolve()),
            "sha256": sha256_file(selection_path),
            "body_sha256": load_document(
                selection_path, SELECTION_SCHEMA, "iteration selection"
            )["body_sha256"],
        },
    })
    return selection_path


def validate_iteration_reference(
    path: pathlib.Path, *, output_root: pathlib.Path,
    plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    reference = load_document(path, REFERENCE_SCHEMA, "iteration reference")
    receipt_record = reference.get("receipt")
    if (reference.get("namespace") != NAMESPACE
            or reference.get("campaign_id") != CAMPAIGN_ID
            or reference.get("iterations_remaining") != 0
            or reference.get("second_iteration_authorized") is not False
            or not isinstance(receipt_record, Mapping)):
        raise IterationError("iteration reference policy changed")
    receipt = pathlib.Path(receipt_record["path"])
    try:
        receipt.resolve().relative_to(output_root.resolve())
    except ValueError as error:
        raise IterationError("iteration receipt escaped output root") from error
    if sha256_file(receipt) != receipt_record.get("sha256"):
        raise IterationError("iteration receipt hash changed")
    document = load_document(receipt, RUN_RECEIPT_SCHEMA, "iteration receipt")
    outcome = document.get("post_iteration_outcome")
    outcome_schema = document.get("post_iteration_outcome_schema")
    if (document.get("body_sha256") != receipt_record.get("body_sha256")
            or document.get("total_games") != TOTAL_GAMES
            or document.get("workers") != WORKERS
            or document.get("protected_tests_opened") is not False
            or not isinstance(outcome, Mapping)
            or not valid_sha(outcome.get("sha256"))
            or not isinstance(outcome_schema, str)
            or outcome_schema not in {
                POST_ITERATION_HANDOFF_SCHEMA,
                campaign_module().POST_ITERATION_FAILURE_SCHEMA,
            }
            or (plan is not None
                and document.get("plan_body_sha256") != plan.get("body_sha256"))):
        raise IterationError("iteration receipt content changed")
    for field, schema in (
        ("selection", SELECTION_SCHEMA),
        ("post_iteration_outcome", outcome_schema),
        ("campaign_completion", campaign_module().ITERATION_EVENT_SCHEMA),
    ):
        record = document.get(field)
        if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
            raise IterationError(f"iteration receipt {field} binding is missing")
        bound = require_within(
            pathlib.Path(record["path"]), output_root.parent, field
        )
        if (sha256_file(bound) != record.get("sha256")
                or load_document(bound, schema, field).get("body_sha256")
                != record.get("body_sha256")):
            raise IterationError(f"iteration receipt {field} binding changed")
    manifests = document.get("train_manifests")
    if not isinstance(manifests, list) or not manifests:
        raise IterationError("iteration receipt train manifests are missing")
    for record in manifests:
        if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
            raise IterationError("iteration train manifest binding is missing")
        path = require_within(
            pathlib.Path(record["path"]), output_root,
            "iteration train manifest",
        )
        if (path.stat().st_size != record.get("bytes")
                or sha256_file(path) != record.get("sha256")):
            raise IterationError("iteration train manifest binding changed")
    workers = document.get("worker_results")
    if not isinstance(workers, list) or len(workers) != WORKERS:
        raise IterationError("iteration receipt worker bindings are incomplete")
    for index, summary in enumerate(workers):
        result_record = summary.get("result") if isinstance(summary, Mapping) else None
        if not isinstance(result_record, Mapping):
            raise IterationError("iteration worker receipt binding is missing")
        result_path = require_within(
            pathlib.Path(str(result_record.get("path"))), output_root,
            "iteration worker receipt",
        )
        result = load_document(
            result_path, WORKER_RESULT_SCHEMA, "iteration worker receipt"
        )
        if (sha256_file(result_path) != result_record.get("sha256")
                or result.get("body_sha256") != result_record.get("body_sha256")
                or result.get("worker_index") != index
                or result.get("body_sha256") != summary.get("body_sha256")
                or result.get("games") != summary.get("games")
                or result.get("quotas") != summary.get("quotas")):
            raise IterationError("iteration worker receipt binding changed")
    return reference


def execute_plan(
    *, plan_path: pathlib.Path, output_root: pathlib.Path,
    resume: bool,
    runner: Callable[..., Any] = subprocess.run,
    power_check: Callable[[], Mapping[str, Any]] = default_power_check,
    disk_check: Callable[[pathlib.Path], Mapping[str, Any]] = default_disk_check,
    fine_tuner: Callable[..., pathlib.Path] = fine_tune_and_select,
    require_launchagent: bool = True,
) -> dict[str, Any]:
    plan = load_document(plan_path, PLAN_SCHEMA, "iteration plan")
    validate_plan_contract(plan, plan_path=plan_path, output_root=output_root)
    binding = load_document(
        pathlib.Path(plan["one_shot_plan_binding"]),
        PLAN_BINDING_SCHEMA, "iteration plan binding",
    )
    if (binding.get("plan") != artifact_reference(plan_path, PLAN_SCHEMA)
            or binding.get("one_shot") is not True
            or binding.get("second_plan_authorized") is not False):
        raise IterationError("one-shot iteration plan binding changed")
    if not resume:
        raise IterationError("the one-shot iteration always requires --resume")
    iteration_root = pathlib.Path(plan["authorization"]["path"]).parents[1]
    started_path = iteration_root / "iteration/01-started.json"
    started = qualification_module().load_sealed(
        started_path, campaign_module().ITERATION_EVENT_SCHEMA
    )
    start_environment = started.get("environment")
    if (started.get("status") != "iteration-started"
            or started.get("authorization") != plan["authorization"]
            or not isinstance(start_environment, Mapping)
            or start_environment.get("interactive_launch_agent") is not True
            or start_environment.get("resume") is not True
            or start_environment.get("blas_threads") != 1
            or start_environment.get("ac_power") is not True
            or float(start_environment.get("free_disk_gib", 0)) <= 20.0
            or start_environment.get("plan_body_sha256") != plan["body_sha256"]
            or LABEL_RE.fullmatch(str(start_environment.get("label", ""))) is None):
        raise IterationError("iteration start receipt does not bind this plan")
    if require_launchagent and os.environ.get(
        "COMPACT_VALUE_BFM_INTERACTIVE_LAUNCHAGENT"
    ) != "1":
        raise IterationError("iteration execute must run under its Interactive LaunchAgent")
    if require_launchagent and (
        os.environ.get("COMPACT_VALUE_BFM_LAUNCHAGENT_LABEL")
        != start_environment["label"]
        or os.environ.get("COMPACT_VALUE_BFM_ITERATION_PLAN_BODY_SHA256")
        != plan["body_sha256"]
    ):
        raise IterationError("LaunchAgent label/plan environment is not bound")
    power = dict(power_check())
    disk = dict(disk_check(output_root))
    if power.get("ac_power") is not True or float(disk.get("free_disk_gib", 0)) <= 20:
        raise IterationError("iteration lost AC power or minimum free disk")
    reference_path = output_root / "iteration-reference.json"
    if reference_path.exists():
        return validate_iteration_reference(
            reference_path, output_root=output_root, plan=plan
        )
    claim_path = output_root / "execution-claim.json"
    claim_body = {
        "schema": CLAIM_SCHEMA, "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID, "plan": artifact_reference(plan_path, PLAN_SCHEMA),
        "one_iteration_only": True, "resume_required": True,
        "blas_threads": 1, "workers": WORKERS,
    }
    if claim_path.exists():
        claim = load_document(claim_path, CLAIM_SCHEMA, "iteration claim")
        if claim != body_hashed(claim_body) or not resume:
            raise IterationError("iteration claim requires exact --resume binding")
    else:
        claim = write_document(claim_path, claim_body)
    worker_results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {
            int(worker["worker_index"]): executor.submit(
                _run_worker, worker, plan=plan, resume=resume, runner=runner
            )
            for worker in plan["workers"]
        }
        for index in range(WORKERS):
            worker_results.append(futures[index].result())
    aggregate = Counter()
    total_games = 0
    total_positions = 0
    total_deep_positions = 0
    manifests = []
    manifest_hashes = []
    for result in worker_results:
        aggregate.update(result["quotas"])
        total_games += result["games"]
        total_positions += result["positions"]
        total_deep_positions += result["deep_relabel_positions"]
        manifests.extend(result["train_manifests"])
        manifest_hashes.extend(result["train_manifest_sha256"])
    if (dict(aggregate) != QUOTAS or total_games != TOTAL_GAMES
            or total_positions != TOTAL_GAMES * POSITIONS_PER_GAME
            or total_deep_positions != int(
                TOTAL_GAMES * POSITIONS_PER_GAME * DEEP_RELABEL_FRACTION
            )
            or len(manifests) != len(set(manifests))
            or len(manifest_hashes) != len(set(manifest_hashes))):
        raise IterationError("worker results do not aggregate to exact 10,000-game quotas")
    # Worker production may span days; no fine-tune may begin against Python
    # sources that differ from the immutable plan prepared before launch.
    validate_maintained_python_tool_closure(plan["tools"])
    fine_reference_path = output_root / "fine-tune/selection-reference.json"
    if fine_reference_path.exists():
        fine_reference = load_document(
            fine_reference_path, FINE_TUNE_REFERENCE_SCHEMA,
            "iteration fine-tune reference",
        )
        selection_path = pathlib.Path(fine_reference["selection"]["path"])
        if (fine_reference.get("plan_body_sha256") != plan["body_sha256"]
                or sha256_file(selection_path)
                != fine_reference["selection"].get("sha256")):
            raise IterationError("fine-tune resume reference changed")
    else:
        selection_path = fine_tuner(
            plan=plan, worker_results=worker_results, output_root=output_root
        )
        if not fine_reference_path.exists():
            selected_document = load_document(
                selection_path, SELECTION_SCHEMA, "iteration selection"
            )
            write_document(fine_reference_path, {
                "schema": FINE_TUNE_REFERENCE_SCHEMA, "namespace": NAMESPACE,
                "plan_body_sha256": plan["body_sha256"],
                "selection": {
                    "path": str(selection_path.resolve()),
                    "sha256": sha256_file(selection_path),
                    "body_sha256": selected_document["body_sha256"],
                },
            })
    selection = load_document(
        selection_path, SELECTION_SCHEMA, "iteration selection"
    )
    checkpoint = selection.get("float_checkpoint")
    runtime = selection.get("runtime")
    generated_source = selection.get("generated_source")
    source_export = selection.get("source_export")
    gate = selection.get("offline_gate")
    if (selection.get("namespace") != NAMESPACE
            or selection.get("plan_body_sha256") != plan["body_sha256"]
            or not isinstance(checkpoint, Mapping)
            or not valid_sha(checkpoint.get("sha256"))
            or not isinstance(runtime, Mapping)
            or not valid_sha(runtime.get("sha256"))
            or not isinstance(generated_source, Mapping)
            or not valid_sha(generated_source.get("sha256"))
            or not isinstance(source_export, Mapping)
            or not isinstance(gate, Mapping)
            or not isinstance(gate.get("passed"), bool)):
        raise IterationError(
            "fine-tuned selection has an invalid checkpoint/runtime/source verdict"
        )
    checkpoint_path = require_within(
        pathlib.Path(str(checkpoint.get("path"))), output_root,
        "post-iteration float checkpoint",
    )
    runtime_path = require_within(
        pathlib.Path(str(runtime.get("path"))), output_root,
        "post-iteration runtime",
    )
    generated_source_path = require_within(
        pathlib.Path(str(generated_source.get("path"))), output_root,
        "post-iteration generated source",
    )
    try:
        source_bytes = generated_source_path.read_bytes()
        source_bytes.decode("ascii")
    except (OSError, UnicodeDecodeError) as error:
        raise IterationError("post-iteration generated source is not ASCII") from error
    if (sha256_file(checkpoint_path) != checkpoint.get("sha256")
            or sha256_file(runtime_path) != runtime.get("sha256")
            or sha256_file(generated_source_path) != generated_source.get("sha256")
            or len(source_bytes) != generated_source.get("bytes")
            or not 0 < len(source_bytes) < 95_000
            or source_export.get("runtime_sha256") != runtime["sha256"]
            or source_export.get("source_sha256") != generated_source["sha256"]
            or source_export.get("source_ascii_bytes") != len(source_bytes)
            or source_export.get("source_limit_exclusive") != 95_000):
        raise IterationError("post-iteration runtime/generated-source binding changed")
    handoff_path = output_root / "post-iteration-development-handoff.json"
    handoff: dict[str, Any] | None = None
    campaign_result = {
        "games": {
            CAMPAIGN_GAME_KEYS[name]: count for name, count in QUOTAS.items()
        },
        "total_games": TOTAL_GAMES,
        "positions_per_game": POSITIONS_PER_GAME, "workers": WORKERS,
        "fixed_work": True, "deep_relabel_fraction": DEEP_RELABEL_FRACTION,
        "resumed": True, "float_checkpoint_sha256": checkpoint["sha256"],
        "quantized_runtime_sha256": runtime["sha256"],
        "generated_source_sha256": generated_source["sha256"],
        "generated_source_ascii_bytes": len(source_bytes),
        "offline_gate_passed": gate["passed"],
        "iteration_selection_body_sha256": selection["body_sha256"],
        "learning_rate": plan["learning_rate"],
    }
    completed_path = iteration_root / "iteration/02-completed.json"
    campaign = campaign_module()
    if completed_path.exists():
        completed = qualification_module().load_sealed(
            completed_path, campaign.ITERATION_EVENT_SCHEMA
        )
        if completed.get("result") != campaign_result:
            raise IterationError("existing iteration completion changed")
    else:
        completed = campaign.complete_iteration(
            iteration_root, result=campaign_result, completed_at_utc=utc_now()
        )
    if gate["passed"]:
        handoff = write_document(handoff_path, {
            "schema": POST_ITERATION_HANDOFF_SCHEMA,
            "namespace": NAMESPACE,
            "status": "offline-qualified-awaiting-development",
            "plan": artifact_reference(plan_path, PLAN_SCHEMA),
            "iteration_completion": artifact_reference(
                completed_path, campaign.ITERATION_EVENT_SCHEMA
            ),
            "iteration_selection": artifact_reference(
                selection_path, SELECTION_SCHEMA
            ),
            "candidate": {
                "candidate_id": "post-iteration-search-target",
                "architecture": CAMPAIGN_ARCHITECTURES[selection["architecture"]],
                "target": "search-target",
                "float_checkpoint": file_record(checkpoint_path),
                "runtime": file_record(runtime_path),
                "generated_source": file_record(generated_source_path),
            },
            "source_export": dict(source_export),
            "offline_gate": dict(gate),
            "candidate_artifacts_immutable": True,
            "development_screen_required": True,
            "development_selected": False,
            "protected_tests_opened": False,
            "protected_tests_authorized": False,
            "upload_authorized": False,
            "iterations_remaining": 0,
        })
        outcome_path = handoff_path
        outcome_schema = POST_ITERATION_HANDOFF_SCHEMA
    else:
        failure_path = iteration_root / "iteration/03-post-iteration-failure.json"
        if failure_path.exists():
            campaign.validate_post_iteration_failure(iteration_root, failure_path)
        else:
            campaign.record_post_iteration_failure(
                iteration_root, stage="offline-evaluator",
                evidence_path=selection_path, recorded_at_utc=utc_now(),
            )
        outcome_path = failure_path
        outcome_schema = campaign.POST_ITERATION_FAILURE_SCHEMA
    body = {
        "schema": RUN_RECEIPT_SCHEMA, "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID, "plan_body_sha256": plan["body_sha256"],
        "claim_body_sha256": claim["body_sha256"],
        "worker_results": [
            {"worker_index": result["worker_index"],
             "result": artifact_reference(
                 pathlib.Path(plan["workers"][result["worker_index"]]["result_path"]),
                 WORKER_RESULT_SCHEMA,
             ),
             "body_sha256": result["body_sha256"],
             "games": result["games"], "quotas": result["quotas"]}
            for result in worker_results
        ],
        "train_manifests": [file_record(pathlib.Path(path)) for path in manifests],
        "selection": artifact_reference(selection_path, SELECTION_SCHEMA),
        "post_iteration_outcome_schema": outcome_schema,
        "post_iteration_outcome": artifact_reference(
            outcome_path, outcome_schema
        ),
        "campaign_completion": {
            "path": str(completed_path.resolve()),
            "sha256": qualification_module().sha256_file(completed_path),
            "body_sha256": completed["body_sha256"],
        },
        "total_games": TOTAL_GAMES, "quotas": QUOTAS,
        "total_positions": total_positions,
        "deep_relabel_positions": total_deep_positions,
        "positions_per_game": POSITIONS_PER_GAME, "workers": WORKERS,
        "fixed_work": True, "deep_relabel_fraction": DEEP_RELABEL_FRACTION,
        "target_semantics": TARGET_SEMANTICS,
        "single_thread_blas": True, "ac_power_at_launch": True,
        "minimum_free_disk_gib": 20.0, "protected_tests_opened": False,
        "offline_gate_passed": gate["passed"],
        "development_screen_required": gate["passed"],
        "terminal_offline_rejection": not gate["passed"],
        "iterations_remaining": 0, "second_iteration_authorized": False,
    }
    receipt_path = write_content_addressed(
        output_root / "run-receipts", body, ".iteration-receipt.json"
    )
    reference = write_document(reference_path, {
        "schema": REFERENCE_SCHEMA, "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID,
        "receipt": {"path": str(receipt_path.resolve()),
                    "sha256": sha256_file(receipt_path),
                    "body_sha256": load_document(
                        receipt_path, RUN_RECEIPT_SCHEMA, "iteration receipt"
                    )["body_sha256"]},
        "iterations_remaining": 0, "second_iteration_authorized": False,
    })
    validate_iteration_reference(reference_path, output_root=output_root, plan=plan)
    return reference


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--output-root", type=pathlib.Path, required=True)
    prepare.add_argument("--authorization", type=pathlib.Path, required=True)
    prepare.add_argument("--bundle-manifest", type=pathlib.Path, required=True)
    prepare.add_argument("--family-selection", type=pathlib.Path, required=True)
    prepare.add_argument("--artifact-root", type=pathlib.Path, required=True)
    prepare.add_argument("--float-checkpoint", type=pathlib.Path, required=True)
    prepare.add_argument("--student-runtime", type=pathlib.Path, required=True)
    prepare.add_argument("--generated-source", type=pathlib.Path, required=True)
    prepare.add_argument("--previous-compact-runtime", type=pathlib.Path, required=True)
    prepare.add_argument("--roots-tsv", type=pathlib.Path, required=True)
    prepare.add_argument("--roots-manifest", type=pathlib.Path, required=True)
    prepare.add_argument("--input-audit", type=pathlib.Path, required=True)
    prepare.add_argument(
        "--selfsearch-workflow", type=pathlib.Path,
        default=SELFSEARCH_WORKFLOW_PATH,
    )
    prepare.add_argument("--continuation-generator", type=pathlib.Path, required=True)
    prepare.add_argument("--jacek-nn-opponent", type=pathlib.Path, required=True)
    prepare.add_argument("--search-teacher", type=pathlib.Path, required=True)
    prepare.add_argument("--rank4-teacher", type=pathlib.Path, required=True)
    prepare.add_argument("--pack-tool", type=pathlib.Path, required=True)
    prepare.add_argument("--python", type=pathlib.Path, required=True)
    prepare.add_argument("--learning-rate", type=float, required=True)
    prepare.add_argument("--label", default="com.papersoccer.compact-value-bfm-iteration")
    install = commands.add_parser("install")
    install.add_argument("--plan", type=pathlib.Path, required=True)
    install.add_argument("--output-root", type=pathlib.Path, required=True)
    install.add_argument("--resume", action="store_true")
    status = commands.add_parser("status")
    status.add_argument("--plan", type=pathlib.Path, required=True)
    status.add_argument("--output-root", type=pathlib.Path, required=True)
    execute = commands.add_parser("execute")
    execute.add_argument("--plan", type=pathlib.Path, required=True)
    execute.add_argument("--output-root", type=pathlib.Path, required=True)
    execute.add_argument("--resume", action="store_true")
    worker = commands.add_parser("worker")
    worker.add_argument("--plan", type=pathlib.Path, required=True)
    worker.add_argument("--worker-index", type=int, required=True)
    worker.add_argument("--output", type=pathlib.Path, required=True)
    worker.add_argument("--resume", action="store_true")
    verify = commands.add_parser("verify")
    verify.add_argument("--reference", type=pathlib.Path, required=True)
    verify.add_argument("--output-root", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result: Any = {"plan": str(prepare_plan(
                args.output_root, authorization_path=args.authorization,
                bundle_manifest=args.bundle_manifest,
                family_selection_path=args.family_selection,
                artifact_root=args.artifact_root,
                float_checkpoint=args.float_checkpoint,
                student_runtime=args.student_runtime,
                generated_source=args.generated_source,
                previous_compact_runtime=args.previous_compact_runtime,
                roots_tsv=args.roots_tsv, roots_manifest=args.roots_manifest,
                input_audit=args.input_audit,
                selfsearch_workflow=args.selfsearch_workflow,
                continuation_generator=args.continuation_generator,
                jacek_nn_opponent=args.jacek_nn_opponent,
                search_teacher=args.search_teacher,
                rank4_teacher=args.rank4_teacher,
                pack_tool=args.pack_tool,
                python_path=args.python, learning_rate=args.learning_rate,
                label=args.label,
            ))}
        elif args.command == "install":
            result = install_launch_agent(
                plan_path=args.plan, output_root=args.output_root,
                resume=args.resume,
            )
        elif args.command == "status":
            result = launch_agent_status(
                plan_path=args.plan, output_root=args.output_root
            )
        elif args.command == "execute":
            result = execute_plan(
                plan_path=args.plan, output_root=args.output_root,
                resume=args.resume,
            )
        elif args.command == "worker":
            result = run_iteration_worker(
                plan_path=args.plan, worker_index=args.worker_index,
                result_path=args.output, resume=args.resume,
            )
        else:
            result = validate_iteration_reference(
                args.reference, output_root=args.output_root
            )
    except (IterationError, OSError, ValueError, json.JSONDecodeError) as error:
        parser.exit(1, f"compact iteration failure: {error}\n")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
