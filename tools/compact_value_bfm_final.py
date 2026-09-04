#!/usr/bin/env python3
"""Fail-closed protected-final orchestration for compact_value_bfm."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
REPOSITORY = HERE.parent


def _load(path: pathlib.Path, name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load final helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


qualification = _load(
    HERE / "compact_value_bfm_qualification.py", "compact_final_qualification"
)
campaign = _load(HERE / "compact_value_bfm_campaign.py", "compact_final_campaign")
opening_tools = _load(
    HERE / "compact_value_bfm_openings.py", "compact_final_openings"
)
preflight_tools = _load(
    HERE / "compact_value_bfm_preflight.py", "compact_final_preflight"
)
gate_support = _load(
    REPOSITORY /
    "submissions/codingame/bots/compact_value_bfm/rank4_gate_support.py",
    "compact_final_gate_support",
)
base = qualification
FinalError = qualification.QualificationError

NAMESPACE = "compact_value_bfm"
FREEZE_SCHEMA = "papersoccer.compact-value-bfm.final-freeze.v1"
PLAN_SCHEMA = "papersoccer.compact-value-bfm.protected-final-plan.v1"
CONSUMPTION_SCHEMA = "papersoccer.compact-value-bfm.final-bank-consumption.v1"
QUALIFIED_INPUT_SCHEMA = "papersoccer.compact-value-bfm.rank4-qualified-inputs.v1"
RAW_SHARD_EVIDENCE_SCHEMA = (
    "papersoccer.compact-value-bfm.raw-final-shard-evidence.v1"
)
FINAL_BANK_ADAPTER_SCHEMA = qualification.FINAL_BANK_SCHEMA


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _regular(
    path: pathlib.Path, *, ascii_required: bool = False,
    executable: bool = False,
) -> dict[str, Any]:
    if (path.is_symlink() or not path.is_file()
            or (executable and not os.access(path, os.X_OK))):
        raise FinalError(f"required regular file is absent: {path}")
    raw = path.read_bytes()
    if ascii_required:
        try:
            raw.decode("ascii")
        except UnicodeDecodeError as error:
            raise FinalError(f"required source is not ASCII: {path}") from error
    return {
        "path": str(path.resolve()),
        "bytes": len(raw),
        "sha256": qualification.sha256_bytes(raw),
        **({"ascii": True} if ascii_required else {}),
    }


def verify_clean_git(
    repository: pathlib.Path, candidate_source: pathlib.Path, commit: str,
) -> dict[str, Any]:
    def git(*arguments: str) -> bytes:
        completed = subprocess.run(
            ["git", *arguments], cwd=repository,
            capture_output=True, check=False,
        )
        if completed.returncode != 0:
            raise FinalError(f"Git freeze read failed: {' '.join(arguments[:2])}")
        return completed.stdout

    head = git("rev-parse", "HEAD").decode().strip()
    if head != commit:
        raise FinalError("candidate freeze commit is not current HEAD")
    if git("status", "--porcelain=v1", "--untracked-files=no").strip():
        raise FinalError("candidate freeze requires a clean tracked worktree")
    try:
        relative = candidate_source.resolve().relative_to(repository.resolve()).as_posix()
    except ValueError as error:
        raise FinalError("candidate source is outside the repository") from error
    git("ls-files", "--error-unmatch", "--", relative)
    if git("show", f"{commit}:{relative}") != candidate_source.read_bytes():
        raise FinalError("candidate source differs from committed bytes")
    return {"commit": commit, "tracked_clean": True,
            "source_path": relative, "committed_bytes_equal": True}


def _load_content_addressed(path: pathlib.Path, schema: str) -> dict[str, Any]:
    value = qualification.load_sealed(path, schema)
    if not path.name.endswith(".json") or path.name[:-5] != qualification.sha256_file(path):
        raise FinalError(f"artifact is not content-addressed: {path}")
    return value


def _validate_preflight(
    path: pathlib.Path, *, candidate: Mapping[str, Any],
    candidate_commit: str, runtime_sha256: str,
) -> dict[str, Any]:
    receipt = _load_content_addressed(path, preflight_tools.RECEIPT_SCHEMA)
    before = receipt.get("inputs_before")
    embedded_claim = receipt.get("claim")
    embedded_plan = receipt.get("plan")
    if not isinstance(embedded_claim, dict) or not isinstance(embedded_plan, dict):
        raise FinalError("preflight receipt omits its validated claim or plan")
    try:
        preflight_tools.validate_preflight_receipt(
            receipt,
            claim=embedded_claim,
            plan=embedded_plan,
            inputs=before,
        )
    except Exception as error:
        raise FinalError("full preflight receipt validation failed") from error
    checks = receipt.get("checks")
    if (receipt.get("status") != "passed" or not isinstance(before, dict)
            or receipt.get("inputs_after") != before
            or before.get("candidate_commit") != candidate_commit
            or before.get("candidate", {}).get("sha256") != candidate["sha256"]
            or before.get("candidate", {}).get("bytes") != candidate["bytes"]
            or before.get("candidate", {}).get("bootstrap_zero") is not False
            or before.get("runtime", {}).get("sha256") != runtime_sha256
            or before.get("rank4", {}).get("sha256") != qualification.RANK4_SHA256
            or not isinstance(checks, dict) or not checks
            or any(value != "passed" for value in checks.values())
            or receipt.get("protected_banks_accessed") != []
            or receipt.get("git_writes") != 0 or receipt.get("uploads") != 0):
        raise FinalError("preflight receipt does not bind the clean selected source")
    return receipt


def freeze_candidate(
    output_root: pathlib.Path, *, repository: pathlib.Path,
    selection_path: pathlib.Path, protected_test_authorization_path: pathlib.Path,
    preflight_receipt_path: pathlib.Path, candidate_source: pathlib.Path,
    rank4_source: pathlib.Path, runtime_path: pathlib.Path,
    frozen_at_utc: str,
    git_verifier: Callable[[pathlib.Path, pathlib.Path, str], Mapping[str, Any]] =
    verify_clean_git,
) -> dict[str, Any]:
    selection = qualification.load_sealed(
        selection_path, campaign.SELECTION_SCHEMA
    )
    test_auth = qualification.load_sealed(
        protected_test_authorization_path, campaign.TEST_AUTH_SCHEMA
    )
    if (selection.get("selection_immutable") is not True
            or selection.get("status") !=
            "immutable-development-selected-not-tests-opened"
            or test_auth.get("selection") != qualification.artifact_reference(
                selection_path, campaign.SELECTION_SCHEMA
            )
            or test_auth.get("selection_may_change") is not False):
        raise FinalError("selection/protected-test authorization is not immutable")
    candidate = _regular(candidate_source, ascii_required=True)
    rank4 = _regular(rank4_source, ascii_required=True)
    runtime = _regular(runtime_path, ascii_required=True)
    if candidate["bytes"] >= 95_000 or candidate["sha256"] == qualification.RANK4_SHA256:
        raise FinalError("candidate source is oversized or aliases Rank-4")
    if rank4["sha256"] != qualification.RANK4_SHA256 or rank4["bytes"] != qualification.RANK4_BYTES:
        raise FinalError("final freeze Rank-4 source is not exact")
    if (selection.get("model", {}).get("artifact_sha256") != runtime["sha256"]
            or test_auth.get("artifact", {}).get("sha256") != runtime["sha256"]):
        raise FinalError("selected/protected-test runtime identity changed")
    preflight_header = _load_content_addressed(
        preflight_receipt_path, preflight_tools.RECEIPT_SCHEMA
    )
    commit = preflight_header.get("inputs_before", {}).get("candidate_commit")
    if not isinstance(commit, str):
        raise FinalError("preflight receipt omits candidate commit")
    preflight = _validate_preflight(
        preflight_receipt_path, candidate=candidate,
        candidate_commit=commit,
        runtime_sha256=runtime["sha256"],
    )
    git = dict(git_verifier(repository, candidate_source, commit))
    if git.get("commit") != commit or git.get("tracked_clean") is not True:
        raise FinalError("clean Git verifier did not bind the selected commit")
    source_binding_path = output_root / "source-binding.json"
    source_binding = qualification.create_source_binding(
        source_binding_path,
        candidate_source=candidate_source,
        candidate_commit=commit,
        rank4_source=rank4_source,
        opponent_source=rank4_source,
    )
    return qualification.write_sealed(output_root / "freeze.json", {
        "schema": FREEZE_SCHEMA,
        "namespace": NAMESPACE,
        "status": "candidate-source-clean-commit-frozen",
        "frozen_at_utc": frozen_at_utc,
        "selection": qualification.artifact_reference(
            selection_path, campaign.SELECTION_SCHEMA
        ),
        "protected_test_authorization": qualification.artifact_reference(
            protected_test_authorization_path, campaign.TEST_AUTH_SCHEMA
        ),
        "preflight": qualification.artifact_reference(preflight_receipt_path),
        "source_binding": qualification.artifact_reference(
            source_binding_path, qualification.SOURCE_BINDING_SCHEMA
        ),
        "candidate_commit": commit,
        "candidate": candidate,
        "rank4": rank4,
        "runtime": runtime,
        "git": git,
        "source_binding_body_sha256": source_binding["body_sha256"],
    })


def _write_atomic(path: pathlib.Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != raw:
                raise FinalError(f"immutable final artifact collision: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _materialize_gate_bank(
    directory: pathlib.Path, protected_bank_path: pathlib.Path,
) -> pathlib.Path:
    bank = opening_tools.validate_bank(protected_bank_path)
    if bank.get("classification") != "protected-final" or bank.get("opening_count") != 500:
        raise FinalError("gate bank requires the exact protected 500-opening bank")
    raw = (
        "# papersoccer.compact-value-bfm-opening-bank.v1\n"
        "opening_id\ttranscript\n" + "".join(
            f"{opening['opening_id']}\t{opening['transcript']}\n"
            for opening in bank["openings"]
        )
    ).encode("ascii")
    path = directory / f"{qualification.sha256_bytes(raw)}.tsv"
    _write_atomic(path, raw)
    gate_support.validate_bank(path)
    return path


def prepare_protected_final(
    output_root: pathlib.Path, *, freeze_path: pathlib.Path,
    copied_exclusion_paths: Sequence[pathlib.Path],
    development_bank_paths: Sequence[pathlib.Path],
    rank4_gate_executable: pathlib.Path, created_at_utc: str,
    entropy: Any,
) -> dict[str, Any]:
    freeze = qualification.load_sealed(freeze_path, FREEZE_SCHEMA)
    source_binding_path = pathlib.Path(freeze["source_binding"]["path"])
    opening_tools.load_protected_exclusions(
        copied_exclusion_paths, development_bank_paths
    )
    seed_path = output_root / "protected-seed.json"
    opening_tools.create_protected_seed_receipt(
        seed_path,
        source_binding_path=source_binding_path,
        clean_binding_path=pathlib.Path(freeze["preflight"]["path"]),
        exclusion_paths=copied_exclusion_paths,
        development_bank_paths=development_bank_paths,
        created_at_utc=created_at_utc,
        entropy=entropy,
    )
    protected_bank_path = opening_tools.generate_protected_bank(
        output_root / "banks",
        seed_receipt_path=seed_path,
        exclusion_paths=copied_exclusion_paths,
        development_bank_paths=development_bank_paths,
    )
    gate_bank_path = _materialize_gate_bank(
        output_root / "gate-bank", protected_bank_path
    )
    protected_bank = opening_tools.validate_bank(protected_bank_path)
    adapter_path = output_root / "protected-bank-adapter.json"
    qualification.write_sealed(adapter_path, {
        "schema": FINAL_BANK_ADAPTER_SCHEMA,
        "namespace": NAMESPACE,
        "classification": "fresh-protected-final",
        "source_binding": qualification.artifact_reference(
            source_binding_path, qualification.SOURCE_BINDING_SCHEMA
        ),
        "candidate_commit": freeze["candidate_commit"],
        "candidate_sha256": freeze["candidate"]["sha256"],
        "rank4_sha256": qualification.RANK4_SHA256,
        "opening_count": 500,
        "protected_bank": {
            "path": str(protected_bank_path.resolve()),
            "sha256": qualification.sha256_file(protected_bank_path),
        },
        "gate_bank": {
            "path": str(gate_bank_path.resolve()),
            "sha256": qualification.sha256_file(gate_bank_path),
        },
        "seed_receipt": qualification.artifact_reference(
            seed_path, opening_tools.SEED_SCHEMA
        ),
    })
    gate_executable = _regular(rank4_gate_executable, executable=True)
    preflight = qualification.load_sealed(
        pathlib.Path(freeze["preflight"]["path"]), preflight_tools.RECEIPT_SCHEMA
    )
    expected_gate = (
        preflight.get("panels", {})
        .get("clang-release", {})
        .get("binaries", {})
        .get("papersoccer_codingame_compact_value_bfm_rank4_gate")
    )
    if (
        not isinstance(expected_gate, dict)
        or expected_gate.get("sha256") != gate_executable["sha256"]
        or expected_gate.get("bytes") != gate_executable["bytes"]
        or expected_gate.get("executable") is not True
    ):
        raise FinalError(
            "Rank-4 gate executable is not the source-specific preflight binary"
        )
    gate_binding_path = output_root / "gate-binding.json"
    qualification.create_gate_binding(
        gate_binding_path,
        source_binding_path=source_binding_path,
        bank_path=adapter_path,
        harness_path=rank4_gate_executable,
    )
    timing_samples = preflight["timing"]["samples"]
    uncontended = {
        "first_max_ms": max(sample["first_ms"] for sample in timing_samples),
        "later_max_ms": max(sample["later_max_ms"] for sample in timing_samples),
    }
    plan = qualification.write_sealed(output_root / "final-plan.json", {
        "schema": PLAN_SCHEMA,
        "namespace": NAMESPACE,
        "status": "protected-final-ready-unconsumed",
        "created_at_utc": created_at_utc,
        "freeze": qualification.artifact_reference(freeze_path, FREEZE_SCHEMA),
        "source_binding": freeze["source_binding"],
        "candidate_commit": freeze["candidate_commit"],
        "candidate": freeze["candidate"],
        "rank4": freeze["rank4"],
        "runtime": freeze["runtime"],
        "selection": freeze["selection"],
        "preflight": freeze["preflight"],
        "seed_receipt": qualification.artifact_reference(seed_path, opening_tools.SEED_SCHEMA),
        "protected_bank": {"path": str(protected_bank_path.resolve()),
                           "sha256": qualification.sha256_file(protected_bank_path)},
        "gate_bank": {"path": str(gate_bank_path.resolve()),
                      "sha256": qualification.sha256_file(gate_bank_path)},
        "bank_adapter": qualification.artifact_reference(
            adapter_path, qualification.FINAL_BANK_SCHEMA
        ),
        "gate_binding": qualification.artifact_reference(
            gate_binding_path, qualification.GATE_BINDING_SCHEMA
        ),
        "rank4_gate": gate_executable,
        "exclusions": opening_tools.load_protected_exclusions(
            copied_exclusion_paths, development_bank_paths
        )["sources"],
        "uncontended_timing": uncontended,
        "shards": 100,
        "pairs_per_shard": 5,
        "maximum_workers": 4,
        "bank_consumed": False,
    })
    return plan


def consume_bank_at_launch(
    ledger_root: pathlib.Path, *, plan_path: pathlib.Path,
    launched_at_utc: str,
) -> dict[str, Any]:
    plan = qualification.load_sealed(plan_path, PLAN_SCHEMA)
    expected = {
        "schema": CONSUMPTION_SCHEMA,
        "namespace": NAMESPACE,
        "status": "bank-consumed-at-launch",
        "launched_at_utc": launched_at_utc,
        "plan": qualification.artifact_reference(plan_path, PLAN_SCHEMA),
        "protected_bank": plan["protected_bank"],
        "gate_bank": plan["gate_bank"],
        "gate_binding": plan["gate_binding"],
        "one_launch_only": True,
    }
    path = ledger_root / "bank-consumed-at-launch.json"
    if path.exists():
        existing = qualification.load_sealed(path, CONSUMPTION_SCHEMA)
        static = {key: value for key, value in existing.items()
                  if key not in {"body_sha256", "launched_at_utc"}}
        wanted = {key: value for key, value in expected.items()
                  if key != "launched_at_utc"}
        if static != wanted:
            raise FinalError("existing bank consumption marker has another identity")
        return existing
    artifact = qualification.seal(expected)
    _write_atomic(path, qualification.canonical_json_bytes(artifact))
    return qualification.load_sealed(path, CONSUMPTION_SCHEMA)


def gate_command(plan: Mapping[str, Any], index: int, output: pathlib.Path) -> list[str]:
    if _regular(
        pathlib.Path(plan["rank4_gate"]["path"]), executable=True
    ) != plan["rank4_gate"]:
        raise FinalError("source-specific Rank-4 gate binary changed after freeze")
    selection = qualification.load_sealed(
        pathlib.Path(plan["selection"]["path"]), campaign.SELECTION_SCHEMA
    )
    c_value, fpu, visit_weight = selection["tuple"]
    work = selection["profile_work"]
    return [
        plan["rank4_gate"]["path"],
        "--bank", plan["gate_bank"]["path"],
        "--expected-bank-sha256", plan["gate_bank"]["sha256"],
        "--candidate-source", plan["candidate"]["path"],
        "--expected-candidate-sha256", plan["candidate"]["sha256"],
        "--rank4-source", plan["rank4"]["path"],
        "--pair-offset", str(index * 5), "--pair-count", "5",
        "--mode", "actual-clock", "--candidate-c", str(c_value),
        "--candidate-fpu", str(fpu), "--candidate-lambda", str(visit_weight),
        "--candidate-actions", "250",
        "--candidate-root-partial-paths", str(work["root_partial_paths"]),
        "--candidate-nonroot-partial-paths", str(work["nonroot_partial_paths"]),
        "--candidate-nodes", str(work["nodes"]),
        "--candidate-expansions", "2000000", "--candidate-seed", "1",
        "--rank4-nodes", "3000000", "--max-turns", "320",
        "--output", str(output),
    ]


def run_gate_process(spec: Mapping[str, Any]) -> pathlib.Path:
    output = pathlib.Path(spec["raw_output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        spec["command"], cwd=spec["repository"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        check=False, timeout=3_600,
    )
    if completed.returncode not in (0, 2) or not output.is_file():
        raise FinalError(
            "Rank-4 gate shard failed without a complete result "
            f"(stderr_sha256={qualification.sha256_bytes(completed.stderr)})"
        )
    return output


FAILURE_MAP = {
    "candidate_illegal": "illegal", "rank4_illegal": "illegal",
    "lockstep_mismatch": "illegal", "unfinished": "unfinished",
    "candidate_timeout": "timeout", "rank4_timeout": "timeout",
    "candidate_exception": "crash", "rank4_exception": "crash",
    "candidate_malformed": "malformed", "rank4_malformed": "malformed",
}


def adapt_gate_result(
    raw: Any, *, plan: Mapping[str, Any], index: int,
) -> list[dict[str, Any]]:
    path = pathlib.Path(raw)
    document = gate_support.validate_result(
        path,
        expected_bank_sha256=plan["gate_bank"]["sha256"],
        expected_candidate_sha256=plan["candidate"]["sha256"],
        allow_legacy_attempt_zero=True,
    )
    runtime_path = pathlib.Path(plan["runtime"]["path"])
    if _regular(runtime_path, ascii_required=True) != plan["runtime"]:
        raise FinalError("selected runtime changed after final freeze")
    runtime_document = json.loads(runtime_path.read_bytes())
    runtime_body = runtime_document.get("body_sha256")
    runtime_payload = runtime_document.get("quantization", {}).get(
        "payload_sha256"
    )
    if (
        document.get("bindings", {}).get("candidate_runtime_body_sha256")
        != runtime_body
        or document.get("bindings", {}).get("candidate_payload_sha256")
        != runtime_payload
    ):
        raise FinalError("Rank-4 gate executed a different embedded runtime")
    if (document["config"]["pair_offset"] != index * 5
            or document["config"]["pair_count"] != 5
            or document["config"]["mode"] != "actual-clock"):
        raise FinalError("Rank-4 shard result has the wrong range/mode")
    games = []
    for game in document["games"]:
        failure = game["failure"]
        games.append({
            "pair_index": game["pair_index"],
            "candidate_color": game["candidate_player"],
            "candidate_win": failure is None and
                game["winner"] == game["candidate_player"],
            "turns": max(1, game["turns"]),
            "failure": None if failure is None else FAILURE_MAP[failure],
            "first_ms": game["candidate"]["maximum_first_ms"],
            "later_max_ms": game["candidate"]["maximum_later_ms"],
        })
    return games


def _audit_existing_shards(
    ledger_root: pathlib.Path, binding_path: pathlib.Path,
    plan_path: pathlib.Path, plan: Mapping[str, Any],
) -> list[int]:
    missing = []
    for index in range(100):
        claim = ledger_root / "claims" / f"shard-{index:03d}.json"
        receipt = ledger_root / "receipts" / f"shard-{index:03d}.json"
        if claim.exists():
            if not receipt.exists():
                raise qualification.SpentShardError(
                    f"shard {index} started without a valid receipt"
                )
            try:
                validated = qualification.validate_shard_receipt(
                    receipt, binding_path=binding_path, index=index
                )
                evidence_ref = validated.get("evidence", {})
                evidence = qualification.load_sealed(
                    pathlib.Path(str(evidence_ref.get("path"))),
                    RAW_SHARD_EVIDENCE_SCHEMA,
                )
                raw = evidence.get("raw_gate_result", {})
                if (
                    evidence_ref.get("sha256")
                    != qualification.sha256_file(
                        pathlib.Path(str(evidence_ref.get("path")))
                    )
                    or evidence.get("plan")
                    != qualification.artifact_reference(plan_path, PLAN_SCHEMA)
                    or evidence.get("rank4_gate") != plan["rank4_gate"]
                    or evidence.get("shard_index") != index
                    or evidence.get("normalized_games_sha256")
                    != qualification.sha256_bytes(
                        qualification.canonical_json_bytes(validated["games"])
                    )
                    or not isinstance(raw, dict)
                    or not pathlib.Path(str(raw.get("path"))).is_file()
                    or qualification.sha256_file(pathlib.Path(str(raw["path"])))
                    != raw.get("sha256")
                ):
                    raise FinalError("raw final shard evidence binding changed")
            except Exception as error:
                raise qualification.SpentShardError(
                    f"shard {index} has an invalid immutable receipt"
                ) from error
        elif receipt.exists():
            raise FinalError(f"shard {index} receipt exists without start claim")
        else:
            missing.append(index)
    return missing


def run_final_shards(
    ledger_root: pathlib.Path, *, plan_path: pathlib.Path,
    repository: pathlib.Path, maximum_workers: int = 4,
    runner: Callable[[Mapping[str, Any]], Any] = run_gate_process,
    adapter: Callable[..., list[dict[str, Any]]] = adapt_gate_result,
    clock: Callable[[], str] = utc_now,
) -> dict[str, Any]:
    if maximum_workers != 4:
        raise FinalError("protected final requires exactly four calibrated workers")
    plan = qualification.load_sealed(plan_path, PLAN_SCHEMA)
    binding_path = pathlib.Path(plan["gate_binding"]["path"])
    consume_bank_at_launch(
        ledger_root, plan_path=plan_path, launched_at_utc=clock()
    )
    missing = _audit_existing_shards(
        ledger_root, binding_path, plan_path, plan
    )
    aggregate_path = ledger_root / "aggregate.json"
    if aggregate_path.exists():
        if missing:
            raise FinalError(
                "final aggregate exists while one or more shards were never started"
            )
        aggregate = qualification.load_sealed(
            aggregate_path, qualification.FINAL_AGGREGATE_SCHEMA
        )
        if (
            aggregate.get("binding")
            != qualification.artifact_reference(
                binding_path, qualification.GATE_BINDING_SCHEMA
            )
            or aggregate.get("verdict")
            != qualification.strict_gate_verdict(aggregate.get("summary", {}))
        ):
            raise FinalError("existing final aggregate binding or verdict changed")
        qualified_path = ledger_root / "rank4-qualified-inputs.json"
        if aggregate.get("verdict", {}).get("passed") is True:
            qualified = qualification.load_sealed(
                qualified_path, QUALIFIED_INPUT_SCHEMA
            )
            if (
                qualified.get("candidate_commit") != plan["candidate_commit"]
                or qualified.get("aggregate")
                != qualification.artifact_reference(
                    aggregate_path, qualification.FINAL_AGGREGATE_SCHEMA
                )
            ):
                raise FinalError("existing Rank-4 qualification inputs changed")
        elif qualified_path.exists():
            raise FinalError("failed aggregate has Rank-4 qualification inputs")
        return aggregate
    lock = threading.Lock()
    completed = []

    def one(index: int) -> int:
        qualification.start_final_shard(
            ledger_root, binding_path=binding_path, index=index,
            started_at_utc=clock(),
        )
        raw_output = ledger_root / "raw" / f"shard-{index:03d}.json"
        spec = {
            "index": index,
            "repository": str(repository.resolve()),
            "raw_output": str(raw_output),
            "command": gate_command(plan, index, raw_output),
        }
        raw = runner(spec)
        games = adapter(raw, plan=plan, index=index)
        if isinstance(raw, (str, os.PathLike, pathlib.Path)) and pathlib.Path(raw).is_file():
            raw_path = pathlib.Path(raw)
        else:
            raw_output.parent.mkdir(parents=True, exist_ok=True)
            _write_atomic(
                raw_output,
                qualification.canonical_json_bytes({"injected_raw": raw}),
            )
            raw_path = raw_output
        raw_record = _regular(raw_path)
        evidence_path = ledger_root / "raw-evidence" / f"shard-{index:03d}.json"
        qualification.write_sealed(evidence_path, {
            "schema": RAW_SHARD_EVIDENCE_SCHEMA,
            "namespace": NAMESPACE,
            "plan": qualification.artifact_reference(plan_path, PLAN_SCHEMA),
            "rank4_gate": plan["rank4_gate"],
            "shard_index": index,
            "raw_gate_result": raw_record,
            "normalized_games_sha256": qualification.sha256_bytes(
                qualification.canonical_json_bytes(games)
            ),
            "gate_result_validated_before_normalization": True,
        })
        qualification.record_shard_receipt(
            ledger_root, binding_path=binding_path, index=index,
            games=games, completed_at_utc=clock(),
            evidence=qualification.artifact_reference(
                evidence_path, RAW_SHARD_EVIDENCE_SCHEMA
            ),
        )
        with lock:
            completed.append(index)
        return index

    with concurrent.futures.ThreadPoolExecutor(max_workers=maximum_workers) as executor:
        futures = [executor.submit(one, index) for index in missing]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    aggregate = qualification.aggregate_final(
        ledger_root, binding_path=binding_path,
        uncontended_timing=plan["uncontended_timing"],
        completed_at_utc=clock(),
    )
    if aggregate["verdict"]["passed"]:
        qualification.write_sealed(ledger_root / "rank4-qualified-inputs.json", {
            "schema": QUALIFIED_INPUT_SCHEMA,
            "namespace": NAMESPACE,
            "status": "rank4-qualified-awaiting-green-ci",
            "candidate_commit": plan["candidate_commit"],
            "candidate": plan["candidate"],
            "selection": plan["selection"],
            "preflight": plan["preflight"],
            "final_plan": qualification.artifact_reference(plan_path, PLAN_SCHEMA),
            "aggregate": qualification.artifact_reference(
                aggregate_path, qualification.FINAL_AGGREGATE_SCHEMA
            ),
            "one_upload_authorization_requires_green_ci": True,
            "rank4_replacement_authorized": False,
        })
    return aggregate


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--output-root", type=pathlib.Path, required=True)
    freeze.add_argument("--repository", type=pathlib.Path, default=REPOSITORY)
    freeze.add_argument("--selection", type=pathlib.Path, required=True)
    freeze.add_argument("--protected-test-authorization", type=pathlib.Path, required=True)
    freeze.add_argument("--preflight", type=pathlib.Path, required=True)
    freeze.add_argument("--candidate-source", type=pathlib.Path, required=True)
    freeze.add_argument("--rank4-source", type=pathlib.Path, required=True)
    freeze.add_argument("--runtime", type=pathlib.Path, required=True)
    freeze.add_argument("--frozen-at-utc", required=True)
    prepare = commands.add_parser("prepare-protected")
    prepare.add_argument("--output-root", type=pathlib.Path, required=True)
    prepare.add_argument("--freeze", type=pathlib.Path, required=True)
    prepare.add_argument("--copied-exclusion", type=pathlib.Path,
                         action="append", required=True)
    prepare.add_argument("--development-bank", type=pathlib.Path,
                         action="append", required=True)
    prepare.add_argument("--rank4-gate", type=pathlib.Path, required=True)
    prepare.add_argument("--created-at-utc", required=True)
    run = commands.add_parser("run-final")
    run.add_argument("--ledger-root", type=pathlib.Path, required=True)
    run.add_argument("--plan", type=pathlib.Path, required=True)
    run.add_argument("--repository", type=pathlib.Path, default=REPOSITORY)
    run.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        if args.command == "freeze":
            result = freeze_candidate(
                args.output_root, repository=args.repository,
                selection_path=args.selection,
                protected_test_authorization_path=args.protected_test_authorization,
                preflight_receipt_path=args.preflight,
                candidate_source=args.candidate_source,
                rank4_source=args.rank4_source, runtime_path=args.runtime,
                frozen_at_utc=args.frozen_at_utc,
            )
        elif args.command == "prepare-protected":
            result = prepare_protected_final(
                args.output_root, freeze_path=args.freeze,
                copied_exclusion_paths=args.copied_exclusion,
                development_bank_paths=args.development_bank,
                rank4_gate_executable=args.rank4_gate,
                created_at_utc=args.created_at_utc,
                entropy=__import__("secrets").token_bytes,
            )
        else:
            result = run_final_shards(
                args.ledger_root, plan_path=args.plan,
                repository=args.repository, maximum_workers=args.workers,
                clock=utc_now,
            )
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except (FinalError, OSError, UnicodeError, json.JSONDecodeError) as error:
        print(f"compact final failure: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
