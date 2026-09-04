#!/usr/bin/env python3
"""Run and resume the complete Compact Value-BFM development campaign."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import random
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TOOLS = ROOT / "tools"
RANK4 = ROOT / "submissions/codingame/bots/rank_4/submission.cpp"
GATE_SOURCE = HERE / "rank4_gate.cpp"
RANK4_SHA256 = "5c7ebbb38e3b08940eb26ca8cd7585dc5cbce5ad949dfd595bfb0eaab1de53c9"
SELECTION_SCHEMA = "papersoccer.compact-value-bfm-selection.v1"
RUN_SCHEMA = "papersoccer.compact-value-bfm-development-run.v1"
RUN_REFERENCE_SCHEMA = "papersoccer.compact-value-bfm-development-run-reference.v1"
DEVELOPMENT_SCHEMA = "papersoccer.compact-value-bfm.development-input.v1"
NAMESPACE = "compact_value_bfm"
STAGE_PAIRS = {
    "model_screen": 100,
    "tuple_screen": 100,
    "tuple_confirmation": 250,
    "profile_screen": 100,
    "profile_confirmation": 250,
    "actual_clock": 200,
}
ARCHITECTURE = {
    "compact-8x8": ("6301-8-8-1", "primary"),
    "source-neutral-8x16": ("6301-8-16-1", "neutral"),
    "capacity-12x8": ("6301-12-8-1", "capacity"),
}
TARGET = {
    "search-target": "search",
    "teacher-assisted": "teacher",
    "rank4-control": "rank4-control",
}


def _load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


campaign = _load_module("compact_value_bfm_campaign_for_runner", TOOLS / "compact_value_bfm_campaign.py")
openings = _load_module("compact_value_bfm_openings_for_runner", TOOLS / "compact_value_bfm_openings.py")
try:
    from . import export_model, export_submission, rank4_gate_support as gate_support
except ImportError:
    import export_model  # type: ignore[no-redef]
    import export_submission  # type: ignore[no-redef]
    import rank4_gate_support as gate_support  # type: ignore[no-redef]


class DevelopmentError(RuntimeError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(
        value, ensure_ascii=True, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ) + "\n").encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def valid_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value)


def body_hashed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["body_sha256"] = sha256_bytes(canonical_json_bytes(result))
    return result


def verify_body_hash(value: Mapping[str, Any], schema: str, label: str) -> None:
    if value.get("schema") != schema or not valid_sha256(value.get("body_sha256")):
        raise DevelopmentError(f"{label} schema/body hash is invalid")
    body = dict(value)
    expected = body.pop("body_sha256")
    if sha256_bytes(canonical_json_bytes(body)) != expected:
        raise DevelopmentError(f"{label} body hash mismatch")


def atomic_write(path: pathlib.Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_json(path: pathlib.Path, label: str) -> tuple[bytes, dict[str, Any]]:
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DevelopmentError(f"{label} is not JSON") from error
    if not isinstance(value, dict):
        raise DevelopmentError(f"{label} is not an object")
    return payload, value


def resolve_inside(root: pathlib.Path, value: object, label: str) -> pathlib.Path:
    if not isinstance(value, str) or not value or pathlib.PurePath(value).is_absolute():
        raise DevelopmentError(f"{label} must be a relative path")
    path = (root / value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise DevelopmentError(f"{label} escapes the artifact root") from error
    if path.is_symlink() or not path.is_file():
        raise DevelopmentError(f"{label} is missing or a symlink")
    return path


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    architecture: str
    target: str
    selection_path: pathlib.Path
    selection_sha256: str
    selection_body_sha256: str
    runtime_path: pathlib.Path
    runtime_sha256: str
    deployment_eligible: bool
    source_path: pathlib.Path
    source_sha256: str
    source_bytes: int
    binary_path: pathlib.Path
    binary_sha256: str


@dataclass(frozen=True)
class BankInput:
    stage: str
    path: pathlib.Path
    sha256: str
    manifest_path: pathlib.Path
    manifest_sha256: str
    pairs: int
    binding: dict[str, Any]


GateExecutor = Callable[[Candidate, BankInput, Mapping[str, Any]], dict[str, Any]]
Compiler = Callable[[pathlib.Path, pathlib.Path, pathlib.Path], None]


def _candidate_id(architecture: str, arm: str) -> tuple[str, str, str]:
    if architecture not in ARCHITECTURE or arm not in TARGET:
        raise DevelopmentError("selection architecture/arm is outside the family")
    campaign_architecture, prefix = ARCHITECTURE[architecture]
    if arm == "rank4-control":
        if architecture != "compact-8x8":
            raise DevelopmentError("Rank-4 control must use the primary architecture")
        return "rank4-control", campaign_architecture, campaign.CONTROL_TARGET
    return f"{prefix}-{TARGET[arm]}", campaign_architecture, arm


def _selection_runtime(selection_path: pathlib.Path) -> tuple[dict[str, Any], pathlib.Path]:
    payload, selection = load_json(selection_path, "family selection")
    if selection_path.name != f"{sha256_bytes(payload)}.selection.json":
        raise DevelopmentError("family selection is not content addressed")
    verify_body_hash(selection, SELECTION_SCHEMA, "family selection")
    if selection_path.parent.name != "selections":
        raise DevelopmentError("family selection is outside its canonical campaign root")
    campaign_root = selection_path.parent.parent.resolve()
    runtime = selection.get("runtime")
    if not isinstance(runtime, dict) or set(runtime) != {"path", "sha256", "bytes"}:
        raise DevelopmentError("family selection runtime binding is malformed")
    runtime_path = resolve_inside(campaign_root, runtime["path"], "selected runtime")
    if (sha256_file(runtime_path) != runtime.get("sha256")
            or runtime_path.stat().st_size != runtime.get("bytes")):
        raise DevelopmentError("selected runtime bytes changed")
    try:
        if selection.get("arm") == "rank4-control":
            runtime_payload, runtime_document = load_json(
                runtime_path, "control runtime"
            )
            if runtime_path.name != (
                f"{sha256_bytes(runtime_payload)}.runtime.json"
            ):
                raise DevelopmentError("control runtime is not content addressed")
            verify_body_hash(
                runtime_document, export_model.RUNTIME_SCHEMA, "control runtime"
            )
        else:
            runtime_document, _payload, _metadata = export_model.validate_runtime(
                runtime_path
            )
    except Exception as error:
        raise DevelopmentError("selected runtime contract is invalid") from error
    runtime_selection = runtime_document.get("selection", {})
    if (
        selection.get("architecture")
        != runtime_document.get("architecture", {}).get("name")
        or selection.get("arm") != runtime_selection.get("arm")
        or selection.get("seed") != runtime_selection.get("seed")
        or selection.get("float_epoch") != runtime_selection.get("float_epoch")
        or selection.get("qat_epoch") != runtime_selection.get("qat_epoch")
        or selection.get("source_bundle_body_sha256")
        != runtime_selection.get("source_bundle_body_sha256")
    ):
        raise DevelopmentError("selection labels disagree with runtime identity")
    return selection, runtime_path


def _development_header(runtime_path: pathlib.Path, arm: str,
                        scratch: pathlib.Path) -> bytes:
    if arm != "rank4-control":
        return export_model.render_header(runtime_path)[0]
    raw, runtime = load_json(runtime_path, "control runtime")
    if runtime_path.name != f"{sha256_bytes(raw)}.runtime.json":
        raise DevelopmentError("control runtime is not content addressed")
    verify_body_hash(runtime, export_model.RUNTIME_SCHEMA, "control runtime")
    selection = runtime.get("selection")
    if not isinstance(selection, dict) or selection.get("arm") != "rank4-control":
        raise DevelopmentError("control runtime arm mismatch")
    original_body = runtime["body_sha256"]
    patched = dict(runtime)
    patched_selection = dict(selection)
    patched_selection["arm"] = "search-target"
    patched["selection"] = patched_selection
    patched.pop("body_sha256")
    patched = body_hashed(patched)
    payload = canonical_json_bytes(patched)
    path = scratch / f"{sha256_bytes(payload)}.runtime.json"
    atomic_write(path, payload)
    header = export_model.render_header(path)[0]
    patched_body = patched["body_sha256"]
    return header.replace(patched_body.encode(), original_body.encode()).replace(
        patched_body[:12].encode(), original_body[:12].encode())


def _default_compiler(gate_source: pathlib.Path, candidate_source: pathlib.Path,
                      output: pathlib.Path) -> None:
    compiler = os.environ.get("CXX", "c++")
    argument = f'-DCOMPACT_VALUE_BFM_CANDIDATE_SOURCE="{candidate_source.resolve()}"'
    result = subprocess.run(
        [compiler, "-std=c++20", "-O3", argument, str(gate_source), "-o", str(output)],
        cwd=ROOT, text=True, capture_output=True,
    )
    if result.returncode != 0:
        raise DevelopmentError(f"candidate gate compilation failed: {result.stderr}")


def _default_compiler_identity() -> dict[str, str]:
    command = os.environ.get("CXX", "c++")
    executable = shutil.which(command)
    if executable is None:
        raise DevelopmentError(f"C++ compiler is unavailable: {command}")
    completed = subprocess.run(
        [executable, "--version"], text=True, capture_output=True)
    if completed.returncode != 0:
        raise DevelopmentError("cannot identify the C++ compiler")
    return {
        "command": command,
        "executable": str(pathlib.Path(executable).resolve()),
        "version_sha256": sha256_bytes(completed.stdout.encode("utf-8")),
    }


def _metric(document: Mapping[str, Any], candidate_id: str, pairs: int,
            **extra: Any) -> dict[str, Any]:
    result = document["result"]
    times = result["candidate"]["times_ms"]
    latency = sum(times) / len(times) if times else 0.0
    return {
        "candidate_id": candidate_id,
        "pairs": pairs,
        "games": pairs * 2,
        "wins": result["candidate_wins"],
        "color_wins": {
            "0": result["candidate_wins_player0"],
            "1": result["candidate_wins_player1"],
        },
        "failures": result["failures"],
        "latency_ms": latency,
        **extra,
    }


def _pair_scores(document: Mapping[str, Any]) -> dict[tuple[int, str], int]:
    result: dict[tuple[int, str], int] = {}
    for game in document["games"]:
        key = (int(game["pair_index"]), str(game["opening_id"]))
        result.setdefault(key, 0)
        if game["failure"] is None and game["winner"] == game["candidate_player"]:
            result[key] += 1
    return result


def paired_bootstrap_lower(candidate: Mapping[str, Any], default: Mapping[str, Any],
                           seed_material: str, samples: int = 20_000) -> float:
    left = _pair_scores(candidate)
    right = _pair_scores(default)
    if set(left) != set(right) or not left:
        raise DevelopmentError("confirmation receipts do not share exact paired openings")
    deltas = [left[key] - right[key] for key in sorted(left)]
    random_source = random.Random(int(hashlib.sha256(seed_material.encode()).hexdigest(), 16))
    means = []
    for _ in range(samples):
        means.append(sum(deltas[random_source.randrange(len(deltas))]
                         for _ in deltas) / len(deltas))
    means.sort()
    return float(means[max(0, math.floor(0.025 * len(means)) - 1)])


class DevelopmentRunner:
    def __init__(self, *, artifact_root: pathlib.Path,
                 selections: Sequence[pathlib.Path], banks: Mapping[str, pathlib.Path],
                 output_root: pathlib.Path, development_output: pathlib.Path,
                 post_iteration_handoff: pathlib.Path | None = None,
                 resume: bool = False, rank4_nodes: int = 3_000_000,
                 compiler: Compiler | None = None,
                 gate_executor: GateExecutor | None = None,
                 compiler_identity: Mapping[str, str] | None = None) -> None:
        self.artifact_root = artifact_root.resolve()
        self.selection_paths = [path.resolve() for path in selections]
        self.post_iteration_handoff = (
            post_iteration_handoff.resolve()
            if post_iteration_handoff is not None else None
        )
        self.bank_paths = {name: path.resolve() for name, path in banks.items()}
        self.output_root = output_root.resolve()
        self.development_output = development_output.resolve()
        self.resume = resume
        self.rank4_nodes = rank4_nodes
        self.compiler = compiler or _default_compiler
        self.compiler_identity = dict(
            compiler_identity or (
                _default_compiler_identity() if compiler is None
                else {"command": "injected", "executable": "injected",
                      "version_sha256": "0" * 64}
            ))
        self.gate_executor = gate_executor
        self.receipts = self.output_root / "receipts"
        self.references = self.output_root / "run-references"
        self.sources = self.output_root / "candidate-sources"
        self.binaries = self.output_root / "gate-binaries"
        self.gate_banks = self.output_root / "gate-banks"
        self.scratch = self.output_root / "scratch"
        self.run_evidence: dict[str, dict[str, Any]] = {}

    def _record_evidence(self, receipt: Mapping[str, Any]) -> None:
        payload = canonical_json_bytes(dict(receipt))
        digest = sha256_bytes(payload)
        request_sha = str(receipt["request_sha256"])
        receipt_path = self.receipts / f"{digest}.development-run.json"
        self.run_evidence[request_sha] = {
            "request_sha256": request_sha,
            "receipt": receipt_path.name,
            "path": str(receipt_path.resolve()),
            "bytes": len(payload),
            "receipt_sha256": digest,
            "receipt_body_sha256": receipt["body_sha256"],
            "schema": RUN_SCHEMA,
        }

    def _banks(self) -> dict[str, BankInput]:
        if set(self.bank_paths) != set(STAGE_PAIRS):
            raise DevelopmentError("exact six-bank development roster is required")
        result = {}
        all_fingerprints: set[str] = set()
        for stage, pairs in STAGE_PAIRS.items():
            manifest_path = self.bank_paths[stage]
            manifest_sha = sha256_file(manifest_path)
            if manifest_path.name != f"{manifest_sha}.opening-bank.json":
                raise DevelopmentError(f"{stage} opening manifest is not content addressed")
            artifact = openings.validate_bank(manifest_path)
            rows = artifact.get("openings")
            campaign_binding = artifact.get("campaign_binding")
            if (artifact.get("stage") != stage
                    or artifact.get("classification") != "unprotected-development"
                    or not isinstance(rows, list) or len(rows) != pairs
                    or not isinstance(campaign_binding, dict)):
                raise DevelopmentError(f"{stage} bank must contain exactly {pairs} pairs")
            fingerprints = list(campaign_binding.get("fingerprints", []))
            if len(set(fingerprints)) != pairs or all_fingerprints.intersection(fingerprints):
                raise DevelopmentError("development banks are not mutually disjoint")
            all_fingerprints.update(fingerprints)
            if (campaign_binding.get("pairs") != pairs
                    or campaign_binding.get("transcripts") !=
                    [row.get("transcript") for row in rows]
                    or campaign_binding.get("primitive_ply_counts") !=
                    [row.get("primitive_plies") for row in rows]):
                raise DevelopmentError(f"{stage} campaign binding is stale")
            tsv = ("# papersoccer.compact-value-bfm-opening-bank.v1\n"
                   "opening_id\ttranscript\n" + "".join(
                       f"{row['opening_id']}\t{row['transcript']}\n" for row in rows
                   )).encode("ascii")
            gate_sha = sha256_bytes(tsv)
            gate_path = self.gate_banks / f"{gate_sha}.tsv"
            if gate_path.exists() and gate_path.read_bytes() != tsv:
                raise DevelopmentError("content-addressed gate bank collision")
            if not gate_path.exists():
                atomic_write(gate_path, tsv)
            gate_support.validate_bank(gate_path)
            result[stage] = BankInput(
                stage, gate_path, gate_sha, manifest_path, manifest_sha,
                pairs, dict(campaign_binding))
        return result

    def _compile_candidates(self) -> list[Candidate]:
        if sha256_file(RANK4) != RANK4_SHA256:
            raise DevelopmentError("maintained Rank-4 source changed")
        specifications: list[dict[str, Any]] = []
        ids = set()
        self.scratch.mkdir(parents=True, exist_ok=True)
        for selection_path in self.selection_paths:
            selection, runtime_path = _selection_runtime(selection_path)
            arm = str(selection.get("arm"))
            if (
                self.post_iteration_handoff is not None
                and (
                    len(self.selection_paths) != 1
                    or arm != "rank4-control"
                    or selection.get("deployment_eligible") is not False
                )
            ):
                raise DevelopmentError(
                    "post-iteration development accepts only the exact Rank-4 control selection"
                )
            if (
                self.post_iteration_handoff is None
                and arm != "rank4-control"
                and selection.get("deployment_eligible") is not True
            ):
                continue
            candidate_id, architecture, target = _candidate_id(
                str(selection.get("architecture")), arm)
            if candidate_id in ids:
                raise DevelopmentError("family candidate IDs are repeated")
            ids.add(candidate_id)
            header = _development_header(runtime_path, arm, self.scratch)
            _, source = export_submission.render(model_header=header)
            source_sha = sha256_bytes(source)
            source_path = self.sources / f"{source_sha}.cpp"
            if source_path.exists() and source_path.read_bytes() != source:
                raise DevelopmentError("content-addressed candidate source collision")
            if not source_path.exists():
                atomic_write(source_path, source)
            specifications.append({
                "candidate_id": candidate_id,
                "architecture": architecture,
                "target": target,
                "selection_path": selection_path,
                "selection_sha256": sha256_file(selection_path),
                "selection_body_sha256": str(selection["body_sha256"]),
                "runtime_path": runtime_path,
                "runtime_sha256": sha256_file(runtime_path),
                "deployment_eligible": bool(selection.get("deployment_eligible")),
                "source_path": source_path,
                "source_sha256": source_sha,
                "source_bytes": len(source),
            })

        if self.post_iteration_handoff is not None:
            try:
                details = campaign.validate_post_iteration_handoff(
                    self.post_iteration_handoff
                )
            except Exception as error:
                raise DevelopmentError(
                    "post-iteration development handoff did not validate"
                ) from error
            selection = details["selection"]
            selection_path = pathlib.Path(details["selection_path"])
            runtime_path = pathlib.Path(details["runtime_path"])
            source_path = pathlib.Path(details["source_path"])
            candidate = details["candidate"]
            if candidate["candidate_id"] in ids:
                raise DevelopmentError("post-iteration candidate ID is repeated")
            ids.add(candidate["candidate_id"])
            try:
                runtime_document, _runtime_payload, runtime_metadata = (
                    export_model.validate_runtime(runtime_path)
                )
                header, rendered_metadata = export_model.render_header(runtime_path)
                _default_output, rendered_source = export_submission.render(
                    model_header=header
                )
            except Exception as error:
                raise DevelopmentError(
                    "post-iteration runtime/source export did not validate"
                ) from error
            runtime_identity = runtime_document.get("selection")
            source_export = selection.get("source_export")
            source = source_path.read_bytes()
            if (
                not isinstance(runtime_identity, dict)
                or runtime_document.get("architecture", {}).get("name")
                != selection.get("architecture")
                or runtime_identity.get("arm") != "search-target"
                or runtime_identity.get("seed") != selection.get("seed")
                or runtime_identity.get("float_epoch") != selection.get("float_epoch")
                or runtime_identity.get("qat_epoch") != selection.get("qat_epoch")
                or not isinstance(source_export, dict)
                or runtime_metadata.get("file_sha256")
                != candidate["runtime"]["sha256"]
                or rendered_metadata.get("file_sha256")
                != candidate["runtime"]["sha256"]
                or rendered_metadata.get("body_sha256")
                != source_export.get("runtime_body_sha256")
                or rendered_metadata.get("header_sha256")
                != source_export.get("model_header_sha256")
                or sha256_bytes(source) != candidate["generated_source"]["sha256"]
                or len(source) != candidate["generated_source"]["bytes"]
                or rendered_source != source
            ):
                raise DevelopmentError(
                    "post-iteration runtime/generated source identity changed"
                )
            specifications.append({
                "candidate_id": candidate["candidate_id"],
                "architecture": candidate["architecture"],
                "target": candidate["target"],
                "selection_path": selection_path,
                "selection_sha256": sha256_file(selection_path),
                "selection_body_sha256": str(selection["body_sha256"]),
                "runtime_path": runtime_path,
                "runtime_sha256": sha256_file(runtime_path),
                "deployment_eligible": True,
                "source_path": source_path,
                "source_sha256": sha256_bytes(source),
                "source_bytes": len(source),
            })

        candidates = []
        for specification in specifications:
            source_path = specification["source_path"]
            source_sha = specification["source_sha256"]
            compile_key = sha256_bytes(canonical_json_bytes({
                "gate_source_sha256": sha256_file(GATE_SOURCE),
                "candidate_source_sha256": source_sha,
                "rank4_source_sha256": RANK4_SHA256,
                "compiler": self.compiler_identity,
                "flags": ["-std=c++20", "-O3"],
            }))
            staging = self.binaries / f".{compile_key}.tmp"
            reference = self.binaries / f"{compile_key}.binary-reference.json"
            binary_path = None
            binary_sha = None
            if self.resume and reference.exists():
                _, record = load_json(reference, "binary reference")
                candidate_path = self.binaries / str(record.get("path"))
                if (record.get("compile_key") != compile_key or not candidate_path.is_file()
                        or not valid_sha256(record.get("sha256"))
                        or sha256_file(candidate_path) != record.get("sha256")):
                    raise DevelopmentError("resumed gate binary reference is stale")
                binary_path = candidate_path
                binary_sha = record["sha256"]
            if binary_path is None:
                self.binaries.mkdir(parents=True, exist_ok=True)
                self.compiler(GATE_SOURCE, source_path, staging)
                if not staging.is_file():
                    raise DevelopmentError("compiler did not create the gate binary")
                binary_sha = sha256_file(staging)
                binary_path = self.binaries / f"{binary_sha}.rank4-gate"
                if binary_path.exists() and sha256_file(binary_path) != binary_sha:
                    raise DevelopmentError("gate binary hash collision")
                if binary_path.exists():
                    staging.unlink()
                else:
                    os.replace(staging, binary_path)
                atomic_write(reference, canonical_json_bytes({
                    "schema": "papersoccer.compact-value-bfm-gate-binary-reference.v1",
                    "compile_key": compile_key,
                    "path": binary_path.name,
                    "sha256": binary_sha,
                }))
            candidates.append(Candidate(
                candidate_id=specification["candidate_id"],
                architecture=specification["architecture"],
                target=specification["target"],
                selection_path=specification["selection_path"],
                selection_sha256=specification["selection_sha256"],
                selection_body_sha256=specification["selection_body_sha256"],
                runtime_path=specification["runtime_path"],
                runtime_sha256=specification["runtime_sha256"],
                deployment_eligible=specification["deployment_eligible"],
                source_path=source_path,
                source_sha256=source_sha,
                source_bytes=specification["source_bytes"],
                binary_path=binary_path,
                binary_sha256=str(binary_sha),
            ))
        return candidates

    def _gate_command(self, candidate: Candidate, bank: BankInput,
                      spec: Mapping[str, Any], output: pathlib.Path) -> list[str]:
        work = spec["work"]
        search_tuple = spec["tuple"]
        command = [
            str(candidate.binary_path),
            "--bank", str(bank.path),
            "--expected-bank-sha256", bank.sha256,
            "--candidate-source", str(candidate.source_path),
            "--expected-candidate-sha256", candidate.source_sha256,
            "--rank4-source", str(RANK4),
            "--pair-offset", "0", "--pair-count", str(bank.pairs),
            "--mode", str(spec["mode"]),
            "--candidate-c", search_tuple[0],
            "--candidate-fpu", search_tuple[1],
            "--candidate-lambda", search_tuple[2],
            "--candidate-actions", "250",
            "--candidate-root-partial-paths", str(work["root_partial_paths"]),
            "--candidate-nonroot-partial-paths", str(work["nonroot_partial_paths"]),
            "--candidate-nodes", str(work["nodes"]),
            "--candidate-expansions", "2000000",
            "--candidate-seed", "1",
            "--rank4-nodes", str(self.rank4_nodes),
            "--max-turns", "320", "--output", str(output),
        ]
        if spec["stage"] == "actual_clock":
            command.extend([
                "--minimum-candidate-wins", "211",
                "--minimum-wins-per-color", "104",
            ])
        return command

    def _resume_receipt(self, request_sha: str) -> dict[str, Any] | None:
        reference_path = self.references / f"{request_sha}.json"
        if not self.resume or not reference_path.exists():
            return None
        _, reference = load_json(reference_path, "development run reference")
        if (reference.get("schema") != RUN_REFERENCE_SCHEMA
                or reference.get("request_sha256") != request_sha
                or not valid_sha256(reference.get("receipt_sha256"))):
            raise DevelopmentError("resumed development run reference is stale")
        receipt_path = self.receipts / str(reference.get("receipt"))
        if (not receipt_path.is_file()
                or sha256_file(receipt_path) != reference["receipt_sha256"]):
            raise DevelopmentError("resumed development receipt bytes are stale")
        payload, receipt = load_json(receipt_path, "development run receipt")
        if receipt_path.name != f"{sha256_bytes(payload)}.development-run.json":
            raise DevelopmentError("resumed development receipt is not content addressed")
        verify_body_hash(receipt, RUN_SCHEMA, "development run receipt")
        if receipt.get("request_sha256") != request_sha:
            raise DevelopmentError("resumed development receipt request changed")
        return receipt

    def _run(self, candidate: Candidate, bank: BankInput,
             spec: Mapping[str, Any]) -> dict[str, Any]:
        request = {
            "schema": "papersoccer.compact-value-bfm-development-request.v1",
            "candidate_id": spec["candidate_id"],
            "model_candidate_id": candidate.candidate_id,
            "selection_sha256": candidate.selection_sha256,
            "selection_body_sha256": candidate.selection_body_sha256,
            "runtime_sha256": candidate.runtime_sha256,
            "candidate_source_sha256": candidate.source_sha256,
            "binary_sha256": candidate.binary_sha256,
            "rank4_source_sha256": RANK4_SHA256,
            "bank_sha256": bank.sha256,
            "bank_manifest_sha256": bank.manifest_sha256,
            "stage": spec["stage"],
            "mode": spec["mode"],
            "tuple": list(spec["tuple"]),
            "work": dict(spec["work"]),
            "pairs": bank.pairs,
        }
        request_sha = sha256_bytes(canonical_json_bytes(request))
        resumed = self._resume_receipt(request_sha)
        if resumed is not None:
            self._record_evidence(resumed)
            return resumed
        self.scratch.mkdir(parents=True, exist_ok=True)
        gate_output = self.scratch / f"{request_sha}.gate.json"
        if self.gate_executor is None:
            command = self._gate_command(candidate, bank, spec, gate_output)
            completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
            if completed.returncode not in (0, 2) or not gate_output.is_file():
                raise DevelopmentError(
                    f"gate execution failed for {spec['candidate_id']}: {completed.stderr}")
            gate_document = gate_support.validate_result(
                gate_output,
                expected_bank_sha256=bank.sha256,
                expected_candidate_sha256=candidate.source_sha256,
                allow_legacy_attempt_zero=True,
            )
        else:
            gate_document = self.gate_executor(candidate, bank, spec)
            atomic_write(gate_output, canonical_json_bytes(gate_document))
            gate_document = gate_support.validate_result(
                gate_output,
                expected_bank_sha256=bank.sha256,
                expected_candidate_sha256=candidate.source_sha256,
                allow_legacy_attempt_zero=True,
            )
        if (gate_document.get("bindings", {}).get("bank_sha256") != bank.sha256
                or gate_document.get("bindings", {}).get("candidate_source_sha256")
                != candidate.source_sha256
                or gate_document.get("bindings", {}).get("rank4_source_sha256")
                != RANK4_SHA256):
            raise DevelopmentError("mocked/real gate result binding mismatch")
        configuration = gate_document.get("config", {})
        expected_configuration = {
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
            "rank4_nodes": self.rank4_nodes,
            "rank4_clocks_ms": [800, 165],
            "max_turns": 320,
            "minimum_candidate_wins": 211 if spec["stage"] == "actual_clock" else -1,
            "minimum_wins_per_color": 104 if spec["stage"] == "actual_clock" else -1,
        }
        if any(
            configuration.get(name) != value
            for name, value in expected_configuration.items()
        ):
            raise DevelopmentError("gate result used a different search configuration")
        receipt = body_hashed({
            "schema": RUN_SCHEMA,
            "namespace": NAMESPACE,
            "request_sha256": request_sha,
            "request": request,
            "selection_sha256": candidate.selection_sha256,
            "selection_body_sha256": candidate.selection_body_sha256,
            "runtime_sha256": candidate.runtime_sha256,
            "gate_result": gate_document,
        })
        payload = canonical_json_bytes(receipt)
        receipt_path = self.receipts / f"{sha256_bytes(payload)}.development-run.json"
        if receipt_path.exists() and receipt_path.read_bytes() != payload:
            raise DevelopmentError("development receipt hash collision")
        if not receipt_path.exists():
            atomic_write(receipt_path, payload)
        reference = {
            "schema": RUN_REFERENCE_SCHEMA,
            "request_sha256": request_sha,
            "receipt": receipt_path.name,
            "receipt_sha256": sha256_file(receipt_path),
        }
        atomic_write(self.references / f"{request_sha}.json",
                     canonical_json_bytes(reference))
        self._record_evidence(receipt)
        return receipt

    @staticmethod
    def _rank(rows: Sequence[dict[str, Any]], architecture: Mapping[str, str]) -> list[dict[str, Any]]:
        return sorted(
            [row for row in rows if row["failures"] == 0],
            key=lambda row: campaign._rank_key(row, architecture[row["candidate_id"]]),
        )

    def execute(self) -> dict[str, Any]:
        banks = self._banks()
        candidates = self._compile_candidates()
        post_iteration = self.post_iteration_handoff is not None
        by_id = {candidate.candidate_id: candidate for candidate in candidates}
        observed_deployable = {
            (candidate.architecture, candidate.target)
            for candidate in candidates if candidate.deployment_eligible
        }
        controls = [candidate for candidate in candidates
                    if candidate.target == campaign.CONTROL_TARGET]
        if (
            (post_iteration and len(observed_deployable) != 1)
            or (not post_iteration and len(observed_deployable) < 3)
            or len(controls) != 1
        ):
            raise DevelopmentError("family model/control roster is incomplete")
        eligible_architectures = sorted(
            {candidate.architecture for candidate in candidates
             if candidate.deployment_eligible},
            key=lambda value: campaign.ARCHITECTURES[value],
        )
        default_tuple = tuple(campaign.DEFAULT_TUPLE)
        default_work = campaign.PROFILE_ROSTER[campaign.DEFAULT_PROFILE]

        model_rows = []
        for candidate in candidates:
            spec = {
                "stage": "model_screen", "candidate_id": candidate.candidate_id,
                "tuple": default_tuple, "work": default_work, "mode": "fixed-work",
            }
            receipt = self._run(candidate, banks["model_screen"], spec)
            model_rows.append(_metric(
                receipt["gate_result"], candidate.candidate_id, 100,
                architecture=candidate.architecture, target=candidate.target,
                source_bytes=candidate.source_bytes,
                artifact_sha256=candidate.runtime_sha256,
                deployment_eligible=candidate.deployment_eligible,
            ))
        top_models, by_model_metric = campaign._validate_model_screen(
            model_rows, eligible_architectures,
            [
                list(item)
                for item in sorted(
                    observed_deployable,
                    key=lambda item: (
                        campaign.ARCHITECTURES[item[0]], item[1]
                    ),
                )
            ],
            development_mode=(
                campaign.POST_ITERATION_DEVELOPMENT_MODE
                if post_iteration else campaign.FAMILY_DEVELOPMENT_MODE
            ),
        )

        tuple_rows = []
        tuple_receipts: dict[str, dict[str, Any]] = {}
        for model in top_models:
            candidate = by_id[model["candidate_id"]]
            for tuple_value in campaign.TUPLE_ROSTER:
                tuple_candidate_id = (
                    f"{candidate.candidate_id}:{campaign.tuple_id(tuple_value)}")
                spec = {
                    "stage": "tuple_screen", "candidate_id": tuple_candidate_id,
                    "tuple": tuple(tuple_value), "work": default_work,
                    "mode": "fixed-work",
                }
                receipt = self._run(candidate, banks["tuple_screen"], spec)
                tuple_receipts[tuple_candidate_id] = receipt
                tuple_rows.append(_metric(
                    receipt["gate_result"], tuple_candidate_id, 100,
                    model_id=candidate.candidate_id, tuple=list(tuple_value),
                ))
        tuple_ranked = campaign._validate_exact_tuple_screen(
            tuple_rows, top_models, by_model_metric)
        default_model = top_models[0]["candidate_id"]
        default_tuple_id = f"{default_model}:{campaign.tuple_id(default_tuple)}"
        carried_tuples = []
        for candidate_id in [row["candidate_id"] for row in tuple_ranked[:2]] + [default_tuple_id]:
            if candidate_id not in carried_tuples:
                carried_tuples.append(candidate_id)

        tuple_confirmation = []
        tuple_confirmation_receipts = {}
        tuple_descriptors = {row["candidate_id"]: row for row in tuple_rows}
        for tuple_candidate_id in carried_tuples:
            descriptor = tuple_descriptors[tuple_candidate_id]
            candidate = by_id[descriptor["model_id"]]
            tuple_value = tuple(descriptor["tuple"])
            spec = {
                "stage": "tuple_confirmation", "candidate_id": tuple_candidate_id,
                "tuple": tuple_value, "work": default_work, "mode": "fixed-work",
            }
            receipt = self._run(candidate, banks["tuple_confirmation"], spec)
            tuple_confirmation_receipts[tuple_candidate_id] = receipt
        default_receipt = tuple_confirmation_receipts[default_tuple_id]["gate_result"]
        for tuple_candidate_id in carried_tuples:
            descriptor = tuple_descriptors[tuple_candidate_id]
            receipt = tuple_confirmation_receipts[tuple_candidate_id]
            lower = 0.0 if tuple_candidate_id == default_tuple_id else paired_bootstrap_lower(
                receipt["gate_result"], default_receipt,
                f"tuple:{banks['tuple_confirmation'].sha256}:{tuple_candidate_id}")
            tuple_confirmation.append(_metric(
                receipt["gate_result"], tuple_candidate_id, 250,
                model_id=descriptor["model_id"], tuple=descriptor["tuple"],
                paired_bootstrap_lower_95=lower,
            ))
        tuple_architecture = {
            row["candidate_id"]: by_id[row["model_id"]].architecture for row in tuple_rows}
        selected_tuple, _ = campaign._confirmation_choice(
            tuple_confirmation, pairs=250, carried_ids=carried_tuples,
            default_id=default_tuple_id, architecture_by_id=tuple_architecture,
            label="tuple")
        selected_candidate = by_id[selected_tuple["model_id"]]
        selected_tuple_value = tuple(selected_tuple["tuple"])

        profile_screen = []
        profile_receipts = {}
        for profile, work in campaign.PROFILE_ROSTER.items():
            spec = {
                "stage": "profile_screen", "candidate_id": profile,
                "tuple": selected_tuple_value, "work": work, "mode": "fixed-work",
            }
            receipt = self._run(selected_candidate, banks["profile_screen"], spec)
            profile_receipts[profile] = receipt
            profile_screen.append(_metric(
                receipt["gate_result"], profile, 100,
                profile=profile, work=dict(work),
            ))
        profile_rows = campaign._validate_profiles(
            profile_screen, pairs=100, label="profile screen")
        profile_architecture = {
            profile: selected_candidate.architecture for profile in campaign.PROFILE_ROSTER}
        ranked_profiles = self._rank(profile_rows, profile_architecture)
        carried_profiles = []
        for profile in [row["candidate_id"] for row in ranked_profiles[:2]] + [campaign.DEFAULT_PROFILE]:
            if profile not in carried_profiles:
                carried_profiles.append(profile)

        profile_confirmation_receipts = {}
        for profile in carried_profiles:
            spec = {
                "stage": "profile_confirmation", "candidate_id": profile,
                "tuple": selected_tuple_value,
                "work": campaign.PROFILE_ROSTER[profile], "mode": "fixed-work",
            }
            profile_confirmation_receipts[profile] = self._run(
                selected_candidate, banks["profile_confirmation"], spec)
        default_profile_receipt = profile_confirmation_receipts[
            campaign.DEFAULT_PROFILE]["gate_result"]
        profile_confirmation = []
        for profile in carried_profiles:
            receipt = profile_confirmation_receipts[profile]
            lower = 0.0 if profile == campaign.DEFAULT_PROFILE else paired_bootstrap_lower(
                receipt["gate_result"], default_profile_receipt,
                f"profile:{banks['profile_confirmation'].sha256}:{profile}")
            profile_confirmation.append(_metric(
                receipt["gate_result"], profile, 250,
                profile=profile, work=dict(campaign.PROFILE_ROSTER[profile]),
                paired_bootstrap_lower_95=lower,
            ))
        selected_profile, _ = campaign._confirmation_choice(
            profile_confirmation, pairs=250, carried_ids=carried_profiles,
            default_id=campaign.DEFAULT_PROFILE,
            architecture_by_id=profile_architecture, label="profile")

        actual_id = selected_tuple["candidate_id"] + ":" + selected_profile["candidate_id"]
        actual_spec = {
            "stage": "actual_clock", "candidate_id": actual_id,
            "tuple": selected_tuple_value,
            "work": campaign.PROFILE_ROSTER[selected_profile["candidate_id"]],
            "mode": "actual-clock",
        }
        actual_receipt = self._run(
            selected_candidate, banks["actual_clock"], actual_spec)
        actual_clock = _metric(actual_receipt["gate_result"], actual_id, 200)

        payload = {
            "schema": DEVELOPMENT_SCHEMA,
            "namespace": NAMESPACE,
            "development_mode": (
                campaign.POST_ITERATION_DEVELOPMENT_MODE
                if post_iteration else campaign.FAMILY_DEVELOPMENT_MODE
            ),
            "eligible_architectures": eligible_architectures,
            "eligible_model_arms": [
                list(item)
                for item in sorted(
                    observed_deployable,
                    key=lambda item: (
                        campaign.ARCHITECTURES[item[0]], item[1]
                    ),
                )
            ],
            "banks": {name: banks[name].binding for name in STAGE_PAIRS},
            "development_bank_evidence": {
                name: {
                    "manifest_path": str(banks[name].manifest_path),
                    "manifest_sha256": banks[name].manifest_sha256,
                    "gate_path": str(banks[name].path),
                    "gate_sha256": banks[name].sha256,
                }
                for name in STAGE_PAIRS
            },
            "model_screen": model_rows,
            "tuple_screen": tuple_rows,
            "tuple_confirmation": tuple_confirmation,
            "profile_screen": profile_screen,
            "profile_confirmation": profile_confirmation,
            "actual_clock": actual_clock,
            "development_run_receipts": [
                self.run_evidence[key] for key in sorted(self.run_evidence)
            ],
        }
        if post_iteration:
            assert self.post_iteration_handoff is not None
            _raw, handoff = load_json(
                self.post_iteration_handoff, "post-iteration handoff"
            )
            payload["post_iteration_handoff"] = {
                "path": str(self.post_iteration_handoff),
                "sha256": sha256_file(self.post_iteration_handoff),
                "body_sha256": handoff["body_sha256"],
            }
            _control_raw, control_selection = load_json(
                controls[0].selection_path, "Rank-4 control selection"
            )
            payload["rank4_control_selection"] = {
                "path": str(controls[0].selection_path),
                "sha256": controls[0].selection_sha256,
                "body_sha256": control_selection["body_sha256"],
            }
        campaign.validate_development_input(payload)
        # Full selection validation is the final guard that the adaptive runner
        # emitted the exact campaign semantics, including the 211/104 boundary.
        campaign.select_development(
            self.output_root / "development-selection-smoke.json", payload)
        selection_smoke = self.output_root / "development-selection-smoke.json"
        selection_smoke.unlink(missing_ok=True)
        atomic_write(self.development_output, canonical_json_bytes(payload))
        return payload


def parse_bank_argument(values: Sequence[str]) -> dict[str, pathlib.Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise DevelopmentError("--bank must be STAGE=PATH")
        stage, path = value.split("=", 1)
        if stage in result:
            raise DevelopmentError("development bank stage repeated")
        result[stage] = pathlib.Path(path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=pathlib.Path, required=True)
    parser.add_argument("--selection", type=pathlib.Path, action="append", required=True)
    parser.add_argument("--post-iteration-handoff", type=pathlib.Path)
    parser.add_argument("--bank", action="append", required=True)
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    parser.add_argument("--development-output", type=pathlib.Path, required=True)
    parser.add_argument("--rank4-nodes", type=int, default=3_000_000)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args(argv)
    runner = DevelopmentRunner(
        artifact_root=arguments.artifact_root,
        selections=arguments.selection,
        post_iteration_handoff=arguments.post_iteration_handoff,
        banks=parse_bank_argument(arguments.bank),
        output_root=arguments.output_root,
        development_output=arguments.development_output,
        resume=arguments.resume,
        rank4_nodes=arguments.rank4_nodes,
    )
    payload = runner.execute()
    print(json.dumps({
        "development_output": str(arguments.development_output),
        "sha256": sha256_bytes(canonical_json_bytes(payload)),
        "actual_clock": payload["actual_clock"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
