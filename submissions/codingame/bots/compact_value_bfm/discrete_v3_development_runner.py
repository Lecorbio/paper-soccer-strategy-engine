#!/usr/bin/env python3
"""Run the standalone discrete-v3 post-holdout development campaign."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TOOLS = ROOT / "tools"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _load(path: pathlib.Path, name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load discrete-v3 runner dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


maintained = _load(HERE / "development_runner.py", "compact_v3_runner_maintained")
development = _load(
    TOOLS / "compact_value_bfm_discrete_v3_development.py",
    "compact_v3_runner_contract",
)
qualification = development.qualification
campaign = maintained.campaign


RunnerError = development.DevelopmentError
BINARY_REFERENCE_SCHEMA = (
    "papersoccer.compact-value-bfm.discrete-v3-development-binary-reference.v1"
)


PlanLoader = Callable[..., Mapping[str, Any]]
Finalizer = Callable[..., pathlib.Path]
CandidateBuilder = Callable[[Mapping[str, Any]], Sequence[maintained.Candidate]]


def _record(path: pathlib.Path) -> dict[str, Any]:
    return development._regular(path)


@contextlib.contextmanager
def _exclusive_lock(path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise RunnerError("development lock is redirected or irregular")
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RunnerError("another v3 development runner is active") from error
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


class DiscreteV3DevelopmentRunner(maintained.DevelopmentRunner):
    def __init__(
        self, *, plan_path: pathlib.Path, output_root: pathlib.Path,
        resume: bool = False, compiler: maintained.Compiler | None = None,
        gate_executor: maintained.GateExecutor | None = None,
        compiler_identity: Mapping[str, str] | None = None,
        plan_loader: PlanLoader = development.validate_plan,
        finalizer: Finalizer = development.finalize_result,
        candidate_builder: CandidateBuilder | None = None,
    ) -> None:
        self.plan_path = plan_path.resolve()
        self.campaign_root = output_root.resolve()
        self.plan_loader = plan_loader
        self.finalizer = finalizer
        self._candidate_builder = candidate_builder
        plan = dict(plan_loader(self.plan_path, output_root=self.campaign_root))
        development_root = pathlib.Path(plan["outputs"]["development_root"])
        identity = dict(
            compiler_identity
            or (
                maintained._default_compiler_identity()
                if compiler is None else {
                    "command": "injected", "executable": "injected",
                    "version_sha256": "0" * 64,
                }
            )
        )
        if identity != plan["compiler"]:
            raise RunnerError("development compiler identity differs from the plan")
        super().__init__(
            artifact_root=self.campaign_root,
            selections=[],
            banks={
                stage: pathlib.Path(plan["banks"][stage]["path"])
                for stage in development.STAGE_ORDER
            },
            output_root=development_root,
            development_output=pathlib.Path(plan["outputs"]["result"]),
            post_iteration_handoff=None,
            resume=resume,
            rank4_nodes=development.RANK4_NODES,
            compiler=compiler,
            gate_executor=gate_executor,
            compiler_identity=identity,
        )
        self.plan = plan
        self.requests_v3 = pathlib.Path(plan["outputs"]["requests"])
        self.receipts_v3 = pathlib.Path(plan["outputs"]["receipts"])
        self.references_v3 = pathlib.Path(plan["outputs"]["references"])
        self.v3_run_evidence: dict[str, dict[str, Any]] = {}
        self.compile_references: dict[str, dict[str, Any]] = {}

    def _validate_output_routes(self) -> None:
        if self.output_root.is_symlink() or not self.output_root.is_dir():
            raise RunnerError("development output root is redirected or absent")
        result = pathlib.Path(self.plan["outputs"]["result"])
        finalists = pathlib.Path(self.plan["outputs"]["finalists"])
        finalist_reference = pathlib.Path(
            self.plan["outputs"]["finalist_reference"]
        )
        routes = (
            self.requests_v3, self.receipts_v3, self.references_v3,
            self.receipts, self.references, self.binaries, self.sources,
            self.gate_banks, self.scratch, finalists,
            result, finalist_reference,
        )
        for path in routes:
            try:
                path.absolute().relative_to(self.output_root.absolute())
            except ValueError as error:
                raise RunnerError("development output route escaped its root") from error
            current = self.output_root
            relative = path.absolute().relative_to(self.output_root.absolute())
            for part in relative.parts:
                current = current / part
                if current.is_symlink():
                    raise RunnerError("development output route contains a symlink")
            should_be_directory = path in routes[:10]
            if path.exists() and (
                (should_be_directory and not path.is_dir())
                or (not should_be_directory and not path.is_file())
            ):
                raise RunnerError("development output route is redirected or irregular")

    def _preflight_execution_state(self, lock: pathlib.Path) -> None:
        self._validate_output_routes()
        planned = {
            self.requests_v3, self.receipts_v3, self.references_v3,
            self.receipts, self.references, self.binaries, self.sources,
            self.gate_banks, self.scratch,
            pathlib.Path(self.plan["outputs"]["result"]),
            pathlib.Path(self.plan["outputs"]["finalists"]),
            pathlib.Path(self.plan["outputs"]["finalist_reference"]),
        }
        if not self.resume:
            present = [path for path in planned if path.exists() or path.is_symlink()]
            unknown = [
                path for path in self.output_root.iterdir()
                if path not in {self.plan_path, lock}
            ]
            if present or unknown:
                raise RunnerError(
                    "preexisting development state requires --resume"
                )

    def _compile_key(
        self, *, candidate_id: str, selection_sha256: str,
        runtime_sha256: str, source_sha256: str,
    ) -> str:
        return maintained.sha256_bytes(maintained.canonical_json_bytes({
            "development_plan_sha256": qualification.sha256_file(self.plan_path),
            "gate_source_sha256": maintained.sha256_file(maintained.GATE_SOURCE),
            "candidate_id": candidate_id,
            "selection_sha256": selection_sha256,
            "runtime_sha256": runtime_sha256,
            "candidate_source_sha256": source_sha256,
            "rank4_source_sha256": maintained.RANK4_SHA256,
            "compiler": self.compiler_identity,
            "flags": ["-std=c++20", "-O3"],
        }))

    def _compile_binary(
        self, *, candidate_id: str, source_path: pathlib.Path,
        source_sha256: str, runtime_sha256: str, selection_sha256: str,
    ) -> tuple[pathlib.Path, str]:
        if maintained.sha256_file(maintained.RANK4) != maintained.RANK4_SHA256:
            raise RunnerError("maintained Rank-4 source changed")
        compile_key = self._compile_key(
            candidate_id=candidate_id, selection_sha256=selection_sha256,
            runtime_sha256=runtime_sha256, source_sha256=source_sha256,
        )
        reference_path = self.binaries / f"{compile_key}.v3-binary-reference.json"
        if reference_path.exists():
            reference = qualification.load_sealed(
                reference_path, BINARY_REFERENCE_SCHEMA
            )
            binary_record = reference.get("binary")
            if (
                reference.get("compile_key") != compile_key
                or reference.get("development_plan")
                != development._sealed_record(self.plan_path, development.PLAN_SCHEMA)
                or reference.get("candidate_id") != candidate_id
                or reference.get("candidate_source_sha256") != source_sha256
                or reference.get("runtime_sha256") != runtime_sha256
                or reference.get("selection_sha256") != selection_sha256
                or not isinstance(binary_record, Mapping)
            ):
                raise RunnerError("resumed binary reference changed")
            binary_path = development._verify_record(
                binary_record, f"{candidate_id} binary"
            )
            self.compile_references[candidate_id] = development._sealed_record(
                reference_path, BINARY_REFERENCE_SCHEMA
            )
            return binary_path, str(binary_record["sha256"])
        if self.resume and self.binaries.exists() and any(self.binaries.iterdir()):
            # A resume may compile a candidate not yet reached, but an orphaned
            # binary without its exact reference is never trusted.
            for path in self.binaries.iterdir():
                if path.name.startswith(f".{compile_key}"):
                    raise RunnerError("orphaned development binary is not resumable")
        self.binaries.mkdir(parents=True, exist_ok=True)
        staging = self.binaries / f".{compile_key}.tmp"
        self.compiler(maintained.GATE_SOURCE, source_path, staging)
        if staging.is_symlink() or not staging.is_file():
            raise RunnerError("compiler did not create a regular gate binary")
        binary_sha = maintained.sha256_file(staging)
        binary_path = self.binaries / f"{binary_sha}.rank4-gate"
        if binary_path.exists():
            if binary_path.is_symlink() or maintained.sha256_file(binary_path) != binary_sha:
                raise RunnerError("development binary collision")
            staging.unlink()
        else:
            os.replace(staging, binary_path)
        qualification.write_sealed(reference_path, {
            "schema": BINARY_REFERENCE_SCHEMA,
            "namespace": development.NAMESPACE,
            "campaign_id": development.CAMPAIGN_ID,
            "compile_key": compile_key,
            "development_plan": development._sealed_record(
                self.plan_path, development.PLAN_SCHEMA
            ),
            "candidate_id": candidate_id,
            "selection_sha256": selection_sha256,
            "runtime_sha256": runtime_sha256,
            "candidate_source_sha256": source_sha256,
            "compiler": dict(self.compiler_identity),
            "gate_source": _record(maintained.GATE_SOURCE),
            "rank4_source": _record(maintained.RANK4),
            "binary": _record(binary_path),
        })
        self.compile_references[candidate_id] = development._sealed_record(
            reference_path, BINARY_REFERENCE_SCHEMA
        )
        return binary_path, binary_sha

    def _ensure_compile_reference(self, candidate: maintained.Candidate) -> None:
        if candidate.candidate_id not in self.compile_references:
            compile_key = self._compile_key(
                candidate_id=candidate.candidate_id,
                selection_sha256=candidate.selection_sha256,
                runtime_sha256=candidate.runtime_sha256,
                source_sha256=candidate.source_sha256,
            )
            if (
                candidate.binary_path.parent != self.binaries
                or candidate.binary_path.name != f"{candidate.binary_sha256}.rank4-gate"
                or maintained.sha256_file(candidate.binary_path) != candidate.binary_sha256
            ):
                raise RunnerError("injected/compiled binary path or name changed")
            reference_path = self.binaries / f"{compile_key}.v3-binary-reference.json"
            expected = qualification.seal({
                "schema": BINARY_REFERENCE_SCHEMA,
                "namespace": development.NAMESPACE,
                "campaign_id": development.CAMPAIGN_ID,
                "compile_key": compile_key,
                "development_plan": development._sealed_record(
                    self.plan_path, development.PLAN_SCHEMA
                ),
                "candidate_id": candidate.candidate_id,
                "selection_sha256": candidate.selection_sha256,
                "runtime_sha256": candidate.runtime_sha256,
                "candidate_source_sha256": candidate.source_sha256,
                "compiler": dict(self.compiler_identity),
                "gate_source": _record(maintained.GATE_SOURCE),
                "rank4_source": _record(maintained.RANK4),
                "binary": _record(candidate.binary_path),
            })
            if reference_path.exists():
                if qualification.load_sealed(
                    reference_path, BINARY_REFERENCE_SCHEMA
                ) != expected:
                    raise RunnerError("compile reference content changed")
            else:
                qualification.write_sealed(reference_path, {
                    key: value for key, value in expected.items()
                    if key != "body_sha256"
                })
            self.compile_references[candidate.candidate_id] = (
                development._sealed_record(reference_path, BINARY_REFERENCE_SCHEMA)
            )
        development._validate_compile_reference(
            self.compile_references[candidate.candidate_id],
            plan=self.plan,
            candidate=self._candidate_binding(candidate),
            development_plan_record=development._sealed_record(
                self.plan_path, development.PLAN_SCHEMA
            ),
        )

    def _candidate_binding(self, candidate: maintained.Candidate) -> dict[str, Any]:
        planned = (
            self.plan["candidate"]
            if candidate.candidate_id == development.CANDIDATE_ID
            else self.plan["rank4_control"]
        )
        return {
            "candidate_id": candidate.candidate_id,
            "architecture": candidate.architecture,
            "target": candidate.target,
            "selection_sha256": candidate.selection_sha256,
            "selection_body_sha256": candidate.selection_body_sha256,
            "runtime_sha256": candidate.runtime_sha256,
            "runtime_body_sha256": planned["runtime_identity"]["body_sha256"],
            "payload_sha256": planned["runtime_identity"]["payload_sha256"],
            "source_sha256": candidate.source_sha256,
            "source_bytes": candidate.source_bytes,
            "binary_path": str(candidate.binary_path.resolve()),
            "binary_sha256": candidate.binary_sha256,
            "binary_bytes": candidate.binary_path.stat().st_size,
        }

    def _real_candidates(self, plan: Mapping[str, Any]) -> list[maintained.Candidate]:
        self.sources.mkdir(parents=True, exist_ok=True)
        candidate_plan = plan["candidate"]
        selection_path = pathlib.Path(candidate_plan["selection"]["path"])
        selection = qualification.load_sealed(
            selection_path, development.adapter.v3.SELECTION_SCHEMA
        )
        runtime_path = development._verify_record(
            candidate_plan["runtime"], "v3 candidate runtime"
        )
        source_path = development._verify_record(
            candidate_plan["generated_source"], "v3 generated source"
        )
        runtime, _payload, metadata = maintained.export_model.validate_runtime(runtime_path)
        header, rendered = maintained.export_model.render_header(runtime_path)
        _default, source = maintained.export_submission.render(model_header=header)
        if (
            runtime.get("architecture", {}).get("dimensions") != [6301, 12, 8, 1]
            or metadata.get("file_sha256") != candidate_plan["runtime"]["sha256"]
            or rendered.get("file_sha256") != candidate_plan["runtime"]["sha256"]
            or source != source_path.read_bytes()
            or hashlib.sha256(source).hexdigest()
            != candidate_plan["generated_source"]["sha256"]
        ):
            raise RunnerError("v3 candidate runtime/source no longer reproduces")
        binary, binary_sha = self._compile_binary(
            candidate_id=development.CANDIDATE_ID,
            source_path=source_path,
            source_sha256=candidate_plan["generated_source"]["sha256"],
            runtime_sha256=candidate_plan["runtime"]["sha256"],
            selection_sha256=candidate_plan["selection"]["sha256"],
        )
        candidate = maintained.Candidate(
            candidate_id=development.CANDIDATE_ID,
            architecture=development.CAPACITY_ARCHITECTURE,
            target="search-target",
            selection_path=selection_path,
            selection_sha256=candidate_plan["selection"]["sha256"],
            selection_body_sha256=selection["body_sha256"],
            runtime_path=runtime_path,
            runtime_sha256=candidate_plan["runtime"]["sha256"],
            deployment_eligible=True,
            source_path=source_path,
            source_sha256=candidate_plan["generated_source"]["sha256"],
            source_bytes=candidate_plan["generated_source"]["bytes"],
            binary_path=binary,
            binary_sha256=binary_sha,
        )

        control_plan = plan["rank4_control"]
        control_selection_path = pathlib.Path(control_plan["selection"]["path"])
        control_selection, control_runtime_path = maintained._selection_runtime(
            control_selection_path
        )
        header = maintained._development_header(
            control_runtime_path, "rank4-control", self.scratch
        )
        _default, control_source = maintained.export_submission.render(model_header=header)
        control_sha = hashlib.sha256(control_source).hexdigest()
        if (
            control_sha != control_plan["rendered_source"]["sha256"]
            or len(control_source) != control_plan["rendered_source"]["bytes"]
        ):
            raise RunnerError("Rank-4 control rendered source changed")
        control_source_path = self.sources / f"{control_sha}.cpp"
        if control_source_path.exists():
            if control_source_path.read_bytes() != control_source:
                raise RunnerError("Rank-4 control source collision")
        else:
            maintained.atomic_write(control_source_path, control_source)
        control_binary, control_binary_sha = self._compile_binary(
            candidate_id=development.CONTROL_ID,
            source_path=control_source_path,
            source_sha256=control_sha,
            runtime_sha256=control_plan["runtime"]["sha256"],
            selection_sha256=control_plan["selection"]["sha256"],
        )
        control = maintained.Candidate(
            candidate_id=development.CONTROL_ID,
            architecture=development.CONTROL_ARCHITECTURE,
            target=campaign.CONTROL_TARGET,
            selection_path=control_selection_path,
            selection_sha256=control_plan["selection"]["sha256"],
            selection_body_sha256=control_selection["body_sha256"],
            runtime_path=control_runtime_path,
            runtime_sha256=control_plan["runtime"]["sha256"],
            deployment_eligible=False,
            source_path=control_source_path,
            source_sha256=control_sha,
            source_bytes=len(control_source),
            binary_path=control_binary,
            binary_sha256=control_binary_sha,
        )
        return [candidate, control]

    def _compile_candidates_v3(self) -> list[maintained.Candidate]:
        candidates = list(
            self._candidate_builder(self.plan)
            if self._candidate_builder is not None
            else self._real_candidates(self.plan)
        )
        if (
            [item.candidate_id for item in candidates]
            != [development.CANDIDATE_ID, development.CONTROL_ID]
            or candidates[0].architecture != development.CAPACITY_ARCHITECTURE
            or candidates[0].target != "search-target"
            or candidates[0].deployment_eligible is not True
            or candidates[1].architecture != development.CONTROL_ARCHITECTURE
            or candidates[1].target != campaign.CONTROL_TARGET
            or candidates[1].deployment_eligible is not False
        ):
            raise RunnerError("v3 candidate/control compile roster changed")
        for item in candidates:
            development._verify_record(_record(item.binary_path), f"{item.candidate_id} binary")
            self._ensure_compile_reference(item)
        return candidates

    def _expected_configuration(
        self, bank: maintained.BankInput, spec: Mapping[str, Any],
    ) -> dict[str, Any]:
        work = spec["work"]
        values = spec["tuple"]
        return {
            "mode": spec["mode"], "pair_offset": 0, "pair_count": bank.pairs,
            "candidate_c": float(values[0]), "candidate_fpu": float(values[1]),
            "candidate_lambda": float(values[2]), "candidate_actions": 250,
            "candidate_root_partial_paths": work["root_partial_paths"],
            "candidate_nonroot_partial_paths": work["nonroot_partial_paths"],
            "candidate_nodes": work["nodes"],
            "candidate_expansions": 2_000_000, "candidate_shuffle_seed": 1,
            "candidate_clocks_ms": [800, 155], "rank4_nodes": self.rank4_nodes,
            "rank4_clocks_ms": [800, 165], "max_turns": 320,
            "minimum_candidate_wins": 211 if spec["stage"] == "actual_clock" else -1,
            "minimum_wins_per_color": 104 if spec["stage"] == "actual_clock" else -1,
        }

    def _request_body(
        self, candidate: maintained.Candidate, bank: maintained.BankInput,
        spec: Mapping[str, Any], metric_extra: Mapping[str, Any],
    ) -> dict[str, Any]:
        planned = (
            self.plan["candidate"]
            if candidate.candidate_id == development.CANDIDATE_ID
            else self.plan["rank4_control"]
        )
        return {
            "schema": development.REQUEST_SCHEMA,
            "namespace": development.NAMESPACE,
            "campaign_id": development.CAMPAIGN_ID,
            "development_plan": development._sealed_record(
                self.plan_path, development.PLAN_SCHEMA
            ),
            "ancestry": dict(self.plan["request_ancestry"]),
            "candidate": self._candidate_binding(candidate),
            "bank": {
                "stage": bank.stage,
                "manifest_path": str(bank.manifest_path),
                "manifest_bytes": bank.manifest_path.stat().st_size,
                "manifest_sha256": bank.manifest_sha256,
                "gate_path": str(bank.path),
                "gate_bytes": bank.path.stat().st_size,
                "gate_sha256": bank.sha256,
            },
            "spec": {
                "stage": spec["stage"], "candidate_id": spec["candidate_id"],
                "mode": spec["mode"], "tuple": list(spec["tuple"]),
                "work": dict(spec["work"]), "pairs": bank.pairs,
            },
            "metric_extra": dict(metric_extra),
            "compile_reference": dict(
                self.compile_references[candidate.candidate_id]
            ),
            "expected_configuration": self._expected_configuration(bank, spec),
            "compiler": dict(self.compiler_identity),
            "gate_source": _record(maintained.GATE_SOURCE),
            "rank4_source": _record(maintained.RANK4),
        }

    def _audit_base_reference(self, request_sha: str) -> None:
        reference_path = self.references / f"{request_sha}.json"
        if not reference_path.exists():
            return
        if reference_path.is_symlink() or reference_path.parent != self.references:
            raise RunnerError("maintained run reference is redirected")
        _raw, reference = maintained.load_json(reference_path, "maintained reference")
        name = reference.get("receipt")
        if not isinstance(name, str) or pathlib.PurePath(name).name != name:
            raise RunnerError("maintained receipt reference escapes its directory")
        receipt = self.receipts / name
        if receipt.is_symlink() or not receipt.is_file() or receipt.parent != self.receipts:
            raise RunnerError("maintained receipt reference is absent or redirected")

    def _recover_base_state(
        self, *, request_sha: str, request: Mapping[str, Any],
        candidate: maintained.Candidate, bank: maintained.BankInput,
        spec: Mapping[str, Any],
    ) -> None:
        if not self.resume:
            return
        reference_path = self.references / f"{request_sha}.json"
        if reference_path.exists():
            self._audit_base_reference(request_sha)
            return
        matches = []
        if self.receipts.exists():
            for path in self.receipts.glob("*.development-run.json"):
                if path.is_symlink() or not path.is_file():
                    raise RunnerError("maintained receipt directory contains a redirect")
                try:
                    raw, receipt = maintained.load_json(path, "orphan maintained receipt")
                    maintained.verify_body_hash(
                        receipt, maintained.RUN_SCHEMA, "orphan maintained receipt"
                    )
                except Exception as error:
                    raise RunnerError("orphan maintained receipt is invalid") from error
                if receipt.get("request_sha256") == request_sha:
                    if path.name != (
                        f"{maintained.sha256_bytes(raw)}.development-run.json"
                    ):
                        raise RunnerError("orphan maintained receipt is not content addressed")
                    matches.append((path, receipt))
        if len(matches) > 1:
            raise RunnerError("multiple orphan maintained receipts match one request")
        if not matches:
            gate_output = self.scratch / f"{request_sha}.gate.json"
            if not gate_output.exists():
                return
            if gate_output.is_symlink() or not gate_output.is_file():
                raise RunnerError("orphan gate output is redirected")
            gate = maintained.gate_support.validate_result(
                gate_output,
                expected_bank_sha256=bank.sha256,
                expected_candidate_sha256=candidate.source_sha256,
            )
            if gate.get("config") != self._expected_configuration(bank, spec):
                raise RunnerError("orphan gate output used another configuration")
            receipt = maintained.body_hashed({
                "schema": maintained.RUN_SCHEMA,
                "namespace": maintained.NAMESPACE,
                "request_sha256": request_sha,
                "request": dict(request),
                "selection_sha256": candidate.selection_sha256,
                "selection_body_sha256": candidate.selection_body_sha256,
                "runtime_sha256": candidate.runtime_sha256,
                "gate_result": gate,
            })
            raw = maintained.canonical_json_bytes(receipt)
            path = self.receipts / (
                maintained.sha256_bytes(raw) + ".development-run.json"
            )
            maintained.atomic_write(path, raw)
            matches.append((path, receipt))
        path, receipt = matches[0]
        reference = {
            "schema": maintained.RUN_REFERENCE_SCHEMA,
            "request_sha256": request_sha,
            "receipt": path.name,
            "receipt_sha256": maintained.sha256_file(path),
        }
        maintained.atomic_write(
            reference_path, maintained.canonical_json_bytes(reference)
        )

    def _run_v3(
        self, candidate: maintained.Candidate, bank: maintained.BankInput,
        spec: Mapping[str, Any], *, metric_extra: Mapping[str, Any],
    ) -> dict[str, Any]:
        body = self._request_body(candidate, bank, spec, metric_extra)
        request_path, _request = development._write_content_addressed(
            self.requests_v3, body, ".request.json"
        )
        request_sha = qualification.sha256_file(request_path)
        base_probe = {
            "schema": "papersoccer.compact-value-bfm-development-request.v1",
            "candidate_id": spec["candidate_id"],
            "model_candidate_id": candidate.candidate_id,
            "selection_sha256": candidate.selection_sha256,
            "selection_body_sha256": candidate.selection_body_sha256,
            "runtime_sha256": candidate.runtime_sha256,
            "candidate_source_sha256": candidate.source_sha256,
            "binary_sha256": candidate.binary_sha256,
            "rank4_source_sha256": maintained.RANK4_SHA256,
            "bank_sha256": bank.sha256,
            "bank_manifest_sha256": bank.manifest_sha256,
            "stage": spec["stage"], "mode": spec["mode"],
            "tuple": list(spec["tuple"]), "work": dict(spec["work"]),
            "pairs": bank.pairs,
        }
        base_request_sha = maintained.sha256_bytes(
            maintained.canonical_json_bytes(base_probe)
        )
        self._recover_base_state(
            request_sha=base_request_sha, request=base_probe,
            candidate=candidate, bank=bank, spec=spec,
        )
        self._audit_base_reference(base_request_sha)
        base_receipt = super()._run(candidate, bank, spec)
        evidence = self.run_evidence.get(base_receipt["request_sha256"])
        if not isinstance(evidence, Mapping):
            raise RunnerError("maintained run did not expose receipt evidence")
        base_path = pathlib.Path(str(evidence["path"]))
        metric = maintained._metric(
            base_receipt["gate_result"], spec["candidate_id"], bank.pairs,
            **dict(metric_extra),
        )
        gate_output = self.scratch / f"{base_receipt['request_sha256']}.gate.json"
        if gate_output.is_symlink() or not gate_output.is_file():
            raise RunnerError("maintained gate output is absent or redirected")
        receipt_path, _receipt = development._write_content_addressed(
            self.receipts_v3,
            {
                "schema": development.RECEIPT_SCHEMA,
                "namespace": development.NAMESPACE,
                "campaign_id": development.CAMPAIGN_ID,
                "development_plan": development._sealed_record(
                    self.plan_path, development.PLAN_SCHEMA
                ),
                "request": development._sealed_record(
                    request_path, development.REQUEST_SCHEMA
                ),
                "request_sha256": request_sha,
                "maintained_receipt": development._sealed_record(
                    base_path, maintained.RUN_SCHEMA
                ),
                "maintained_request_sha256": base_receipt["request_sha256"],
                "gate_result_sha256": qualification.sha256_bytes(
                    qualification.canonical_json_bytes(base_receipt["gate_result"])
                ),
                "gate_output": _record(gate_output),
                "compile_reference": dict(
                    self.compile_references[candidate.candidate_id]
                ),
                "metric": metric,
                "complete": True,
                "final_bank_generation_authorized": False,
                "rank4_gate_authorized": False,
                "upload_authorized": False,
            },
            ".receipt.json",
        )
        receipt_record = development._sealed_record(
            receipt_path, development.RECEIPT_SCHEMA
        )
        reference_path = self.references_v3 / f"{request_sha}.json"
        expected_reference = qualification.seal({
            "schema": development.RECEIPT_REFERENCE_SCHEMA,
            "namespace": development.NAMESPACE,
            "campaign_id": development.CAMPAIGN_ID,
            "development_plan": development._sealed_record(
                self.plan_path, development.PLAN_SCHEMA
            ),
            "request": development._sealed_record(
                request_path, development.REQUEST_SCHEMA
            ),
            "receipt": receipt_record,
            "complete": True,
        })
        if reference_path.exists():
            if reference_path.is_symlink() or qualification.load_sealed(
                reference_path, development.RECEIPT_REFERENCE_SCHEMA
            ) != expected_reference:
                raise RunnerError("resumed v3 run reference changed")
        else:
            qualification.write_sealed(reference_path, {
                key: value for key, value in expected_reference.items()
                if key != "body_sha256"
            })
        validated = development.validate_run_receipt(receipt_record, self.plan)
        self.v3_run_evidence[request_sha] = receipt_record
        return validated

    def _execute_algorithm(
        self, banks: Mapping[str, maintained.BankInput],
        candidates: Sequence[maintained.Candidate],
    ) -> dict[str, Any]:
        candidate, control = candidates
        default_tuple = tuple(campaign.DEFAULT_TUPLE)
        default_work = campaign.PROFILE_ROSTER[campaign.DEFAULT_PROFILE]
        model_rows = []
        for item in candidates:
            spec = {
                "stage": "model_screen", "candidate_id": item.candidate_id,
                "tuple": default_tuple, "work": default_work, "mode": "fixed-work",
            }
            metric_extra = {
                "architecture": item.architecture, "target": item.target,
                "source_bytes": item.source_bytes,
                "artifact_sha256": item.runtime_sha256,
                "deployment_eligible": item.deployment_eligible,
            }
            row = self._run_v3(
                item, banks["model_screen"], spec, metric_extra=metric_extra
            )["metric"]
            model_rows.append(row)
        if model_rows[0]["failures"] != 0:
            raise RunnerError("v3 model screen candidate is not failure-free")
        tuple_rows = []
        tuple_base = {}
        for value in campaign.TUPLE_ROSTER:
            identifier = f"{candidate.candidate_id}:{campaign.tuple_id(value)}"
            spec = {
                "stage": "tuple_screen", "candidate_id": identifier,
                "tuple": tuple(value), "work": default_work, "mode": "fixed-work",
            }
            extra = {"model_id": candidate.candidate_id, "tuple": list(value)}
            run = self._run_v3(candidate, banks["tuple_screen"], spec, metric_extra=extra)
            tuple_rows.append(run["metric"])
            tuple_base[identifier] = run
        ranked = campaign._validate_exact_tuple_screen(
            tuple_rows, [model_rows[0]], {row["candidate_id"]: row for row in model_rows}
        )
        default_tuple_id = f"{candidate.candidate_id}:{campaign.tuple_id(default_tuple)}"
        carried = []
        for identifier in [row["candidate_id"] for row in ranked[:2]] + [default_tuple_id]:
            if identifier not in carried:
                carried.append(identifier)
        descriptors = {row["candidate_id"]: row for row in tuple_rows}
        confirmation_runs = {}
        for identifier in carried:
            descriptor = descriptors[identifier]
            spec = {
                "stage": "tuple_confirmation", "candidate_id": identifier,
                "tuple": tuple(descriptor["tuple"]), "work": default_work,
                "mode": "fixed-work",
            }
            confirmation_runs[identifier] = self._run_v3(
                candidate, banks["tuple_confirmation"], spec,
                metric_extra={
                    "model_id": candidate.candidate_id,
                    "tuple": list(descriptor["tuple"]),
                },
            )
        default_gate = confirmation_runs[default_tuple_id]["base_receipt"]["gate_result"]
        tuple_confirmation = []
        for identifier in carried:
            row = dict(confirmation_runs[identifier]["metric"])
            row["paired_bootstrap_lower_95"] = (
                0.0 if identifier == default_tuple_id else
                maintained.paired_bootstrap_lower(
                    confirmation_runs[identifier]["base_receipt"]["gate_result"],
                    default_gate,
                    f"tuple:{banks['tuple_confirmation'].sha256}:{identifier}",
                    samples=development.BOOTSTRAP_SAMPLES,
                )
            )
            tuple_confirmation.append(row)
        selected_tuple, _normalized = campaign._confirmation_choice(
            tuple_confirmation, pairs=250, carried_ids=carried,
            default_id=default_tuple_id,
            architecture_by_id={row["candidate_id"]: candidate.architecture for row in tuple_rows},
            label="tuple",
        )
        selected_tuple_value = tuple(selected_tuple["tuple"])
        profile_screen = []
        for profile, work in campaign.PROFILE_ROSTER.items():
            spec = {
                "stage": "profile_screen", "candidate_id": profile,
                "tuple": selected_tuple_value, "work": work, "mode": "fixed-work",
            }
            profile_screen.append(self._run_v3(
                candidate, banks["profile_screen"], spec,
                metric_extra={"profile": profile, "work": dict(work)},
            )["metric"])
        profiles = campaign._validate_profiles(
            profile_screen, pairs=100, label="profile screen"
        )
        ranked_profiles = self._rank(
            profiles, {name: candidate.architecture for name in campaign.PROFILE_ROSTER}
        )
        if len(ranked_profiles) < 2:
            raise RunnerError("fewer than two profiles are failure-free")
        carried_profiles = []
        for profile in [row["candidate_id"] for row in ranked_profiles[:2]] + [campaign.DEFAULT_PROFILE]:
            if profile not in carried_profiles:
                carried_profiles.append(profile)
        profile_runs = {}
        for profile in carried_profiles:
            work = campaign.PROFILE_ROSTER[profile]
            spec = {
                "stage": "profile_confirmation", "candidate_id": profile,
                "tuple": selected_tuple_value, "work": work, "mode": "fixed-work",
            }
            profile_runs[profile] = self._run_v3(
                candidate, banks["profile_confirmation"], spec,
                metric_extra={"profile": profile, "work": dict(work)},
            )
        default_profile_gate = profile_runs[campaign.DEFAULT_PROFILE][
            "base_receipt"
        ]["gate_result"]
        profile_confirmation = []
        for profile in carried_profiles:
            row = dict(profile_runs[profile]["metric"])
            row["paired_bootstrap_lower_95"] = (
                0.0 if profile == campaign.DEFAULT_PROFILE else
                maintained.paired_bootstrap_lower(
                    profile_runs[profile]["base_receipt"]["gate_result"],
                    default_profile_gate,
                    f"profile:{banks['profile_confirmation'].sha256}:{profile}",
                    samples=development.BOOTSTRAP_SAMPLES,
                )
            )
            profile_confirmation.append(row)
        selected_profile, _normalized_profiles = campaign._confirmation_choice(
            profile_confirmation, pairs=250, carried_ids=carried_profiles,
            default_id=campaign.DEFAULT_PROFILE,
            architecture_by_id={name: candidate.architecture for name in carried_profiles},
            label="profile",
        )
        actual_id = selected_tuple["candidate_id"] + ":" + selected_profile["candidate_id"]
        actual_spec = {
            "stage": "actual_clock", "candidate_id": actual_id,
            "tuple": selected_tuple_value,
            "work": campaign.PROFILE_ROSTER[selected_profile["candidate_id"]],
            "mode": "actual-clock",
        }
        actual = self._run_v3(
            candidate, banks["actual_clock"], actual_spec, metric_extra={}
        )["metric"]
        rows = {
            "model_screen": model_rows, "tuple_screen": tuple_rows,
            "tuple_confirmation": tuple_confirmation,
            "profile_screen": profile_screen,
            "profile_confirmation": profile_confirmation,
            "actual_clock": actual,
        }
        selected = development._result_selection(rows)
        return {
            "rows": rows,
            "selected": {
                "tuple_candidate_id": selected["tuple"]["candidate_id"],
                "tuple": selected["tuple"]["tuple"],
                "profile": selected["profile"]["profile"],
                "profile_work": selected["profile"]["work"],
                "actual_clock_candidate_id": selected["actual_clock"]["candidate_id"],
            },
        }

    def execute(self) -> dict[str, Any]:
        lock = self.output_root / "development.lock"
        with _exclusive_lock(lock):
            self.plan = dict(self.plan_loader(
                self.plan_path, output_root=self.campaign_root
            ))
            self._preflight_execution_state(lock)
            finalist_reference = pathlib.Path(
                self.plan["outputs"]["finalist_reference"]
            )
            result_path = pathlib.Path(self.plan["outputs"]["result"])
            if finalist_reference.exists():
                if not self.resume:
                    raise RunnerError("completed development requires --resume")
                validated = development.validate_finalist(
                    finalist_reference, plan_path=self.plan_path,
                    output_root=self.campaign_root,
                    plan_validator=self.plan_loader,
                )
                return {
                    "result": qualification.load_sealed(
                        result_path, development.RESULT_SCHEMA
                    ),
                    "finalist": validated["finalist"],
                    "finalist_path": validated["path"],
                }
            if result_path.exists() and not self.resume:
                raise RunnerError("partial completed result requires --resume")
            banks = self._banks()
            for stage in development.STAGE_ORDER:
                if _record(banks[stage].manifest_path) != self.plan["banks"][stage]:
                    raise RunnerError("development bank differs from the plan")
            candidates = self._compile_candidates_v3()
            payload = self._execute_algorithm(banks, candidates)
            prior_result = (
                qualification.load_sealed(result_path, development.RESULT_SCHEMA)
                if result_path.exists() else None
            )
            result_body = {
                "schema": development.RESULT_SCHEMA,
                "namespace": development.NAMESPACE,
                "campaign_id": development.CAMPAIGN_ID,
                "status": "development-complete-awaiting-finalist-seal",
                "completed_at_utc": (
                    prior_result["completed_at_utc"]
                    if prior_result is not None else development.utc_now()
                ),
                "development_plan": development._sealed_record(
                    self.plan_path, development.PLAN_SCHEMA
                ),
                "binaries": {
                    item.candidate_id: _record(item.binary_path)
                    for item in candidates
                },
                "rows": payload["rows"],
                "selected": payload["selected"],
                "run_receipts": [
                    self.v3_run_evidence[key]
                    for key in sorted(self.v3_run_evidence)
                ],
                "request_count": len(self.v3_run_evidence),
                "policy": {
                    "fresh_protected_tests_opened_diagnostically": True,
                    "development_selected": True,
                    "final_bank_generation_authorized": False,
                    "rank4_gate_authorized": False,
                    "upload_authorized": False,
                },
            }
            sealed = qualification.seal(result_body)
            if result_path.exists():
                if prior_result != sealed:
                    raise RunnerError("resumed development result changed")
            else:
                qualification.write_sealed(result_path, result_body)
            finalist_path = self.finalizer(
                plan_path=self.plan_path, result_path=result_path,
                output_root=self.campaign_root,
                created_at_utc=development.utc_now(),
                plan_validator=self.plan_loader,
            )
            finalist = qualification.load_sealed(
                finalist_path, development.FINALIST_SCHEMA
            )
            return {
                "result": qualification.load_sealed(
                    result_path, development.RESULT_SCHEMA
                ),
                "finalist": finalist,
                "finalist_path": finalist_path,
            }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=pathlib.Path, required=True)
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        value = DiscreteV3DevelopmentRunner(
            plan_path=args.plan, output_root=args.output_root, resume=args.resume
        ).execute()
        print(json.dumps({
            "result": str(pathlib.Path(
                value["finalist"]["development_result"]["path"]
            )),
            "finalist": str(value["finalist_path"]),
            "finalist_sha256": qualification.sha256_file(value["finalist_path"]),
            "actual_clock": value["finalist"]["actual_clock"],
        }, sort_keys=True))
        return 0
    except (RunnerError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"discrete-v3 development runner failure: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
