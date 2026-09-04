#!/usr/bin/env python3
"""Run the fail-closed discrete-v3 development recovery campaign.

The recovery campaign deliberately lives beside, rather than inside, the
interrupted development run.  Ten complete model/tuple receipts are validated
in place and carried by reference.  Every remaining stage uses a fresh bank
and an immutable, full-roster claim.  Once such a claim exists, that stage is
never launched again: an interruption or invalid result is terminal.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import dataclasses
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import re
import signal
import stat
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TOOLS = ROOT / "tools"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _load(path: pathlib.Path, name: str) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load recovery dependency: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


maintained = _load(HERE / "development_runner.py", "compact_v3_recovery_maintained")
development = _load(
    TOOLS / "compact_value_bfm_discrete_v3_development.py",
    "compact_v3_recovery_development",
)
recovery = _load(
    TOOLS / "compact_value_bfm_discrete_v3_recovery.py",
    "compact_v3_recovery_plan",
)
qualification = development.qualification
campaign = maintained.campaign


class RecoveryRunnerError(RuntimeError):
    """Recovery execution cannot safely continue."""


class TerminalRecoveryError(RecoveryRunnerError):
    """A no-retry stage was claimed but did not seal valid completion."""


REQUEST_SCHEMA = (
    "papersoccer.compact-value-bfm.discrete-v3-development-recovery-request.v1"
)
BASE_RECEIPT_SCHEMA = (
    "papersoccer.compact-value-bfm.discrete-v3-development-recovery-gate-receipt.v1"
)
RECEIPT_SCHEMA = (
    "papersoccer.compact-value-bfm.discrete-v3-development-recovery-receipt.v1"
)
RECEIPT_REFERENCE_SCHEMA = (
    "papersoccer.compact-value-bfm.discrete-v3-development-recovery-receipt-reference.v1"
)
CLAIM_SCHEMA = (
    "papersoccer.compact-value-bfm.discrete-v3-development-recovery-stage-claim.v1"
)
JOURNAL_SCHEMA = (
    "papersoccer.compact-value-bfm.discrete-v3-development-recovery-journal-event.v1"
)
RESULT_SCHEMA = (
    "papersoccer.compact-value-bfm.discrete-v3-development-recovery-result.v1"
)
FINALIST_SCHEMA = (
    "papersoccer.compact-value-bfm.discrete-v3-post-holdout-recovery-finalist.v1"
)
FINALIST_REFERENCE_SCHEMA = (
    "papersoccer.compact-value-bfm.discrete-v3-post-holdout-recovery-finalist-reference.v1"
)
FRESH_STAGES = (
    "tuple_confirmation",
    "profile_screen",
    "profile_confirmation",
    "actual_clock",
)
CONCURRENT_STAGES = frozenset(FRESH_STAGES[:-1])
MAX_RESULT_BYTES = 512 * 1024 * 1024
MAX_BATCH_LAUNCH_SECONDS = 5.0
ALLOWED_EVENTS = {
    "stage-claimed",
    "batch-launching",
    "job-started",
    "batch-launched",
    "result-validated",
    "receipt-sealed",
    "stage-complete",
    "terminal-failure",
    "campaign-complete",
}


PlanLoader = Callable[..., Mapping[str, Any]]
GateExecutor = Callable[
    [maintained.Candidate, maintained.BankInput, Mapping[str, Any]],
    Mapping[str, Any],
]
OriginalReceiptValidator = Callable[
    [Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]
]


@dataclasses.dataclass(frozen=True)
class Job:
    stage: str
    candidate_id: str
    tuple_values: tuple[str, str, str]
    work: Mapping[str, int]
    pairs: int
    mode: str
    metric_extra: Mapping[str, Any]
    request: Mapping[str, Any]
    request_path: pathlib.Path
    request_sha256: str
    staging_output: pathlib.Path
    final_output: pathlib.Path
    stdout_path: pathlib.Path
    stderr_path: pathlib.Path
    command: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class FreshEvidence:
    job: Job
    document: Mapping[str, Any]
    metric: Mapping[str, Any]
    receipt_record: Mapping[str, Any]
    reference_record: Mapping[str, Any]


def _canonical(value: Any) -> bytes:
    return qualification.canonical_json_bytes(value)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: pathlib.Path) -> str:
    return qualification.sha256_file(path)


def _utc_now() -> str:
    return development.utc_now()


def _lexists(path: pathlib.Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _record(path: pathlib.Path) -> dict[str, Any]:
    return development._regular(path)


def _sealed_record(path: pathlib.Path, schema: str) -> dict[str, Any]:
    return development._sealed_record(path, schema)


def _verify_record(value: Any, label: str) -> pathlib.Path:
    try:
        return development._verify_record(value, label)
    except Exception as error:
        raise RecoveryRunnerError(str(error)) from error


def _verify_sealed_record(value: Any, schema: str, label: str) -> pathlib.Path:
    try:
        return development._verify_sealed_record(value, schema, label)
    except Exception as error:
        raise RecoveryRunnerError(str(error)) from error


def _write_content_addressed(
    directory: pathlib.Path, body: Mapping[str, Any], suffix: str,
) -> tuple[pathlib.Path, dict[str, Any]]:
    artifact = qualification.seal(body)
    raw = _canonical(artifact)
    path = directory / f"{_sha_bytes(raw)}{suffix}"
    qualification.atomic_write_once(path, raw)
    return path, artifact


def _require_private_directory(path: pathlib.Path, label: str) -> None:
    if path.is_symlink():
        raise RecoveryRunnerError(f"{label} is redirected")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path, 0o700)
    metadata = path.stat()
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o700:
        raise RecoveryRunnerError(f"{label} is not a private directory")


def _atomic_private_json(path: pathlib.Path, value: Mapping[str, Any]) -> None:
    qualification.atomic_write_once(path, _canonical(value))
    os.chmod(path, 0o600)


def _publish_no_replace(staging: pathlib.Path, destination: pathlib.Path) -> None:
    if staging.is_symlink() or not staging.is_file():
        raise RecoveryRunnerError("gate staging output is absent or redirected")
    if _lexists(destination):
        raise RecoveryRunnerError("refusing to replace a recovery gate output")
    os.chmod(staging, 0o600)
    try:
        os.link(staging, destination, follow_symlinks=False)
    except FileExistsError as error:
        raise RecoveryRunnerError("recovery gate output appeared concurrently") from error
    try:
        source_record = _record(staging)
        destination_record = _record(destination)
        if (
            source_record["bytes"] != destination_record["bytes"]
            or source_record["sha256"] != destination_record["sha256"]
        ):
            raise RecoveryRunnerError("published recovery gate output changed")
        staging.unlink()
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            destination.unlink()
        raise


def _workspace_gate_processes() -> list[dict[str, Any]]:
    completed = subprocess.run(
        ["/bin/ps", "-axo", "pid=,ppid=,state=,nice=,command="],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RecoveryRunnerError("cannot inspect competing gate processes")
    result: list[dict[str, Any]] = []
    root = str(ROOT)
    for line in completed.stdout.splitlines():
        fields = line.strip().split(None, 4)
        if len(fields) != 5:
            continue
        pid, ppid, state, nice, command = fields
        if root in command and ".rank4-gate" in command:
            result.append({
                "pid": int(pid),
                "ppid": int(ppid),
                "state": state,
                "nice": int(nice),
                "command": command,
            })
    return result


def validate_real_gate_counts(document: Mapping[str, Any], *, pairs: int) -> None:
    """Validate the real gate's aggregate names (not the obsolete aliases)."""

    games = document.get("games")
    result = document.get("result")
    if (
        not isinstance(games, list)
        or len(games) != 2 * pairs
        or not isinstance(result, Mapping)
        or result.get("games") != 2 * pairs
        or result.get("unfinished") != 0
    ):
        raise RecoveryRunnerError(
            "recovery gate result has incomplete games or unfinished work"
        )


@contextlib.contextmanager
def _exclusive_lock(path: pathlib.Path) -> Iterator[None]:
    _require_private_directory(path.parent, "recovery root")
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise RecoveryRunnerError("recovery lock is redirected or irregular")
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise RecoveryRunnerError("recovery lock is not regular")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RecoveryRunnerError("another recovery runner is active") from error
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _journal_entries(directory: pathlib.Path) -> list[dict[str, Any]]:
    if not directory.exists():
        return []
    if directory.is_symlink() or not directory.is_dir():
        raise RecoveryRunnerError("recovery journal is redirected or irregular")
    entries: list[dict[str, Any]] = []
    previous = "0" * 64
    for path in sorted(directory.iterdir()):
        if path.is_symlink() or not path.is_file():
            raise RecoveryRunnerError("recovery journal contains an irregular entry")
        match = re.fullmatch(r"(\d{6})-([0-9a-f]{64})\.json", path.name)
        if match is None:
            raise RecoveryRunnerError("recovery journal filename is malformed")
        try:
            value = qualification.load_sealed(path, JOURNAL_SCHEMA)
        except Exception as error:
            raise RecoveryRunnerError("recovery journal entry is invalid") from error
        sequence = len(entries) + 1
        if (
            int(match.group(1)) != sequence
            or match.group(2) != value["body_sha256"]
            or value.get("sequence") != sequence
            or value.get("previous_sha256") != previous
            or value.get("event") not in ALLOWED_EVENTS
        ):
            raise RecoveryRunnerError("recovery journal chain changed")
        if entries and entries[-1]["event"] in {"terminal-failure", "campaign-complete"}:
            raise RecoveryRunnerError("recovery journal continues after a terminal event")
        entries.append(value)
        previous = value["body_sha256"]
    return entries


JOURNAL_ENVELOPE_FIELDS = {
    "schema",
    "namespace",
    "recovery_id",
    "recovery_plan",
    "sequence",
    "previous_sha256",
    "event",
    "stage",
    "created_at_utc",
    "body_sha256",
}


def _event_details(event: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in event.items()
        if key not in JOURNAL_ENVELOPE_FIELDS
    }


def _append_event(
    directory: pathlib.Path,
    *,
    plan_record: Mapping[str, Any],
    recovery_id: str,
    event: str,
    stage: str | None,
    **fields: Any,
) -> dict[str, Any]:
    if event not in ALLOWED_EVENTS:
        raise RecoveryRunnerError(f"unknown recovery journal event: {event}")
    _require_private_directory(directory, "recovery journal")
    entries = _journal_entries(directory)
    body = {
        "schema": JOURNAL_SCHEMA,
        "namespace": development.NAMESPACE,
        "recovery_id": recovery_id,
        "recovery_plan": dict(plan_record),
        "sequence": len(entries) + 1,
        "previous_sha256": (
            entries[-1]["body_sha256"] if entries else "0" * 64
        ),
        "event": event,
        "stage": stage,
        "created_at_utc": _utc_now(),
        **fields,
    }
    artifact = qualification.seal(body)
    path = directory / (
        f"{artifact['sequence']:06d}-{artifact['body_sha256']}.json"
    )
    _atomic_private_json(path, artifact)
    return artifact


class DiscreteV3RecoveryRunner:
    """One-process, no-retry orchestrator for the recovery plan."""

    def __init__(
        self,
        *,
        plan_path: pathlib.Path,
        output_root: pathlib.Path,
        plan_loader: PlanLoader = recovery.validate_recovery_plan,
        compiler_identity: Mapping[str, str] | None = None,
        candidate_builder: Callable[[Mapping[str, Any]], maintained.Candidate]
        | None = None,
        gate_executor: GateExecutor | None = None,
        original_plan_loader: PlanLoader = development.validate_plan,
        original_receipt_validator: OriginalReceiptValidator = (
            development.validate_run_receipt
        ),
        read_only: bool = False,
    ) -> None:
        self.plan_path = plan_path.resolve()
        self.output_root = output_root.resolve()
        self.plan_loader = plan_loader
        self.compiler_identity = dict(
            compiler_identity or maintained._default_compiler_identity()
        )
        self.candidate_builder = candidate_builder
        self.gate_executor = gate_executor
        self.original_plan_loader = original_plan_loader
        self.original_receipt_validator = original_receipt_validator
        self.read_only = read_only
        loaded = dict(self.plan_loader(self.plan_path, output_root=self.output_root))
        self.plan_context = loaded
        self.plan = dict(loaded["plan"] if "plan" in loaded else loaded)
        self.routes = {
            key: pathlib.Path(value)
            for key, value in self.plan["outputs"].items()
        }
        self.recovery_root = self.routes["recovery_root"]
        self.plan_record = _sealed_record(self.plan_path, recovery.PLAN_SCHEMA)
        self.candidate: maintained.Candidate | None = None
        self.compile_reference: Mapping[str, Any] | None = None
        self.banks: dict[str, maintained.BankInput] = {}
        self.original_plan: Mapping[str, Any] | None = None

    def _validate_routes(self) -> None:
        if self.compiler_identity != self.plan.get("compiler"):
            raise RecoveryRunnerError("recovery compiler identity differs from the plan")
        if self.recovery_root.is_symlink() or not self.recovery_root.is_dir():
            raise RecoveryRunnerError("recovery root is absent or redirected")
        if self.routes.get("plan") != self.plan_path:
            raise RecoveryRunnerError("recovery plan output route changed")
        file_routes = {
            "plan",
            "incident",
            "mixed_six_exclusion",
            "result",
            "finalist_reference",
        }
        for name, path in self.routes.items():
            try:
                path.absolute().relative_to(self.recovery_root.absolute())
            except ValueError as error:
                raise RecoveryRunnerError(
                    f"recovery output route escaped its root: {name}"
                ) from error
            current = self.recovery_root
            for part in path.absolute().relative_to(self.recovery_root.absolute()).parts:
                current = current / part
                if current.is_symlink():
                    raise RecoveryRunnerError(
                        f"recovery output route contains a symlink: {name}"
                    )
            if path.exists() and (
                (name in file_routes and not path.is_file())
                or (name not in file_routes and not path.is_dir())
            ):
                raise RecoveryRunnerError(
                    f"recovery output route is irregular: {name}"
                )
        for name in (
            "opening_banks",
            "gate_banks",
            "binaries",
            "scratch",
            "requests",
            "base_receipts",
            "receipts",
            "references",
            "claims",
            "journal",
            "finalists",
        ):
            if self.read_only:
                path = self.routes[name]
                if path.is_symlink() or not path.is_dir():
                    raise RecoveryRunnerError(
                        f"recovery {name} is absent or redirected"
                    )
            else:
                _require_private_directory(self.routes[name], f"recovery {name}")
        tools = self.plan.get("tools")
        if not isinstance(tools, Mapping):
            raise RecoveryRunnerError("recovery tool closure is absent")
        for name, record in tools.items():
            if not isinstance(record, Mapping):
                raise RecoveryRunnerError(f"recovery tool record is malformed: {name}")
            _verify_record(record, f"recovery tool {name}")

    def _original_plan_record(self) -> Mapping[str, Any]:
        original = self.plan.get("original")
        if not isinstance(original, Mapping):
            raise RecoveryRunnerError("recovery original evidence is absent")
        value = original.get("development_plan", original.get("plan"))
        if not isinstance(value, Mapping):
            raise RecoveryRunnerError("original development plan record is absent")
        return value

    def _original_reference_records(self) -> list[Mapping[str, Any]]:
        original = self.plan.get("original")
        assert isinstance(original, Mapping)
        values = original.get(
            "carried_receipt_references",
            original.get("carried_receipts"),
        )
        if not isinstance(values, list) or len(values) != 10:
            raise RecoveryRunnerError("exactly ten original receipt references are required")
        if not all(isinstance(value, Mapping) for value in values):
            raise RecoveryRunnerError("original receipt reference roster is malformed")
        return list(values)

    def _validate_original_receipts(self) -> tuple[
        Mapping[str, Any], list[dict[str, Any]], list[Mapping[str, Any]]
    ]:
        original_plan_record = self._original_plan_record()
        original_plan_path = _verify_sealed_record(
            original_plan_record,
            development.PLAN_SCHEMA,
            "original development plan",
        )
        original_root = original_plan_path.parent.parent
        try:
            original_plan = dict(
                self.original_plan_loader(
                    original_plan_path,
                    output_root=original_root,
                )
            )
        except Exception as error:
            raise RecoveryRunnerError(
                "original development plan failed full validation"
            ) from error
        records = self._original_reference_records()
        observed: dict[tuple[str, str], dict[str, Any]] = {}
        normalized_records: list[Mapping[str, Any]] = []
        for entry in records:
            record = entry.get("reference") if "reference" in entry else entry
            if not isinstance(record, Mapping):
                raise RecoveryRunnerError("carried reference record is absent")
            reference_path = _verify_sealed_record(
                record,
                development.RECEIPT_REFERENCE_SCHEMA,
                "original receipt reference",
            )
            if reference_path.parent != pathlib.Path(
                original_plan["outputs"]["references"]
            ):
                raise RecoveryRunnerError("original receipt reference escaped its root")
            reference = qualification.load_sealed(
                reference_path, development.RECEIPT_REFERENCE_SCHEMA
            )
            try:
                value = dict(
                    self.original_receipt_validator(
                        reference["receipt"], original_plan
                    )
                )
            except Exception as error:
                raise RecoveryRunnerError(
                    "carried original receipt failed full validation"
                ) from error
            request = value.get("request")
            metric = value.get("metric")
            if not isinstance(request, Mapping) or not isinstance(metric, Mapping):
                raise RecoveryRunnerError("carried original receipt is malformed")
            spec = request.get("spec")
            if not isinstance(spec, Mapping):
                raise RecoveryRunnerError("carried original request is malformed")
            key = (str(spec.get("stage")), str(spec.get("candidate_id")))
            if key in observed or metric.get("candidate_id") != key[1]:
                raise RecoveryRunnerError("carried original receipt identity is repeated")
            observed[key] = value
            normalized_records.append(dict(entry))

        model_ids = [development.CANDIDATE_ID, development.CONTROL_ID]
        expected = [
            ("model_screen", candidate_id) for candidate_id in model_ids
        ] + [
            (
                "tuple_screen",
                f"{development.CANDIDATE_ID}:{campaign.tuple_id(tuple_values)}",
            )
            for tuple_values in campaign.TUPLE_ROSTER
        ]
        if set(observed) != set(expected):
            raise RecoveryRunnerError("carried original receipt roster changed")
        rows = [dict(observed[key]["metric"]) for key in expected]
        model_rows = rows[:2]
        tuple_rows = rows[2:]
        by_model = {row["candidate_id"]: row for row in model_rows}
        try:
            ranked = campaign._validate_exact_tuple_screen(
                tuple_rows,
                [by_model[development.CANDIDATE_ID]],
                by_model,
            )
        except Exception as error:
            raise RecoveryRunnerError(
                "carried tuple screen failed canonical validation"
            ) from error
        default_id = (
            f"{development.CANDIDATE_ID}:"
            f"{campaign.tuple_id(campaign.DEFAULT_TUPLE)}"
        )
        carried: list[str] = []
        for identifier in [row["candidate_id"] for row in ranked[:2]] + [default_id]:
            if identifier not in carried:
                carried.append(identifier)
        if len(carried) != 3:
            raise RecoveryRunnerError(
                "recovery requires the exact three-arm tuple confirmation roster"
            )
        return original_plan, rows, normalized_records

    def _load_banks(self) -> dict[str, maintained.BankInput]:
        planned = self.plan_context.get("materialized_banks")
        if not isinstance(planned, Mapping) or set(planned) != set(
            development.STAGE_ORDER
        ):
            raise RecoveryRunnerError(
                "recovery mixed six-bank materialization is incomplete"
            )
        mixed_path = self.routes["mixed_six_exclusion"]
        if mixed_path.is_symlink() or not mixed_path.is_file():
            raise RecoveryRunnerError(
                "mixed-six plus spent-bank exclusion receipt is absent"
            )
        mixed_record = self.plan_context.get("materialized_mixed_six_exclusion")
        if isinstance(mixed_record, Mapping):
            _verify_sealed_record(
                mixed_record,
                recovery.MIXED_EXCLUSION_SCHEMA,
                "mixed-six exclusion receipt",
            )
        else:
            try:
                mixed = qualification.load_sealed(
                    mixed_path, recovery.MIXED_EXCLUSION_SCHEMA
                )
            except Exception as error:
                raise RecoveryRunnerError(
                    "mixed-six exclusion receipt failed validation"
                ) from error
            normalized_mixed = self.plan_context.get("mixed_exclusion")
            if normalized_mixed is not None and mixed != normalized_mixed:
                raise RecoveryRunnerError("mixed-six exclusion normalization changed")
        mixed = self.plan_context.get("mixed_exclusion")
        if (
            not isinstance(mixed, Mapping)
            or mixed.get("selected_bank_count") != 6
            or mixed.get("cross_source_symmetry_intersection_count") != 0
            or mixed.get("fresh_confirmation_excluded_original_six") is not True
            or mixed.get("fresh_confirmation_excluded_historical_seven") is not True
            or mixed.get("fresh_confirmation_excluded_protected_fingerprints")
            is not True
            or mixed.get("protected_fingerprint_count") != 54_611
            or mixed.get("additional_development_exclusions")
            != self.plan["additional_development_exclusions"]
            or mixed.get("selection_uses_only_selected_six") is not True
        ):
            raise RecoveryRunnerError(
                "mixed-six receipt does not prove all required exclusions"
            )
        result: dict[str, maintained.BankInput] = {}
        fingerprints: set[str] = set()
        for stage in development.STAGE_ORDER:
            record = planned[stage]
            manifest = _verify_record(record, f"recovery {stage} bank")
            if (
                stage == "tuple_confirmation"
                and manifest.parent != self.routes["opening_banks"]
            ):
                raise RecoveryRunnerError(
                    "fresh recovery tuple-confirmation bank escaped its root"
                )
            try:
                artifact = maintained.openings.validate_bank(manifest)
            except Exception as error:
                raise RecoveryRunnerError(
                    f"recovery {stage} opening bank failed validation"
                ) from error
            rows = artifact.get("openings")
            binding = artifact.get("campaign_binding")
            pairs = development.STAGE_PAIRS[stage]
            if (
                artifact.get("stage") != stage
                or artifact.get("classification") != "unprotected-development"
                or not isinstance(rows, list)
                or len(rows) != pairs
                or not isinstance(binding, Mapping)
                or binding.get("pairs") != pairs
                or binding.get("transcripts")
                != [row.get("transcript") for row in rows]
                or binding.get("primitive_ply_counts")
                != [row.get("primitive_plies") for row in rows]
            ):
                raise RecoveryRunnerError(f"recovery {stage} bank contract changed")
            stage_fingerprints = list(binding.get("fingerprints", []))
            if (
                len(stage_fingerprints) != pairs
                or len(set(stage_fingerprints)) != pairs
                or fingerprints.intersection(stage_fingerprints)
            ):
                raise RecoveryRunnerError("recovery mixed banks are not disjoint")
            fingerprints.update(stage_fingerprints)
            tsv = (
                "# papersoccer.compact-value-bfm-opening-bank.v1\n"
                "opening_id\ttranscript\n"
                + "".join(
                    f"{row['opening_id']}\t{row['transcript']}\n" for row in rows
                )
            ).encode("ascii")
            gate_sha = _sha_bytes(tsv)
            if stage == "tuple_confirmation":
                gate_path = self.routes["gate_banks"] / f"{gate_sha}.tsv"
            else:
                if self.original_plan is None:
                    raise RecoveryRunnerError("original plan was not validated")
                gate_path = (
                    pathlib.Path(self.original_plan["outputs"]["development_root"])
                    / "gate-banks"
                    / f"{gate_sha}.tsv"
                )
            if (
                gate_path.is_symlink()
                or not gate_path.is_file()
                or gate_path.read_bytes() != tsv
            ):
                raise RecoveryRunnerError(
                    f"exact materialized gate bank is absent: {stage}"
                )
            maintained.gate_support.validate_bank(gate_path)
            result[stage] = maintained.BankInput(
                stage=stage,
                path=gate_path,
                sha256=gate_sha,
                manifest_path=manifest,
                manifest_sha256=str(record["sha256"]),
                pairs=pairs,
                binding=dict(binding),
            )
        return result

    def _candidate_contract(self) -> maintained.Candidate:
        candidate = (
            self.candidate_builder(self.plan)
            if self.candidate_builder is not None
            else self._existing_candidate()
        )
        planned = self.plan["candidate"]
        selection = qualification.load_sealed(candidate.selection_path)
        planned_binary = self.plan["binaries"][development.CANDIDATE_ID]
        if (
            candidate.candidate_id != development.CANDIDATE_ID
            or candidate.architecture != development.CAPACITY_ARCHITECTURE
            or candidate.target != "search-target"
            or candidate.deployment_eligible is not True
            or _record(candidate.source_path) != planned["generated_source"]
            or _record(candidate.runtime_path) != planned["runtime"]
            or candidate.selection_sha256 != planned["selection"]["sha256"]
            or candidate.selection_body_sha256 != selection["body_sha256"]
            or candidate.binary_path.is_symlink()
            or not candidate.binary_path.is_file()
            or candidate.binary_sha256 != _sha_file(candidate.binary_path)
            or _record(candidate.binary_path) != planned_binary
        ):
            raise RecoveryRunnerError("recovery candidate/binary binding changed")
        if self.compile_reference is None:
            record = self.plan["compile_references"][development.CANDIDATE_ID]
            path = _verify_sealed_record(
                record,
                "papersoccer.compact-value-bfm.discrete-v3-development-binary-reference.v1",
                "original candidate compile reference",
            )
            self.compile_reference = dict(record)
            if self.original_plan is None:
                raise RecoveryRunnerError("original plan was not validated")
            try:
                development._validate_compile_reference(
                    self.compile_reference,
                    plan=self.original_plan,
                    candidate=self._candidate_binding_for(candidate),
                    development_plan_record=self._original_plan_record(),
                )
            except Exception as error:
                raise RecoveryRunnerError(
                    "original candidate compile reference failed validation"
                ) from error
        return candidate

    def _candidate_binding_for(
        self, candidate: maintained.Candidate
    ) -> dict[str, Any]:
        planned = self.plan["candidate"]
        identity = planned["runtime_identity"]
        return {
            "candidate_id": candidate.candidate_id,
            "architecture": candidate.architecture,
            "target": candidate.target,
            "selection_sha256": candidate.selection_sha256,
            "selection_body_sha256": candidate.selection_body_sha256,
            "runtime_sha256": candidate.runtime_sha256,
            "runtime_body_sha256": identity["body_sha256"],
            "payload_sha256": identity["payload_sha256"],
            "source_sha256": candidate.source_sha256,
            "source_bytes": candidate.source_bytes,
            "binary_path": str(candidate.binary_path.resolve()),
            "binary_sha256": candidate.binary_sha256,
            "binary_bytes": candidate.binary_path.stat().st_size,
        }

    def _existing_candidate(self) -> maintained.Candidate:
        planned = self.plan["candidate"]
        source = _verify_record(planned["generated_source"], "candidate source")
        runtime = _verify_record(planned["runtime"], "candidate runtime")
        selection_path = pathlib.Path(planned["selection"]["path"])
        selection = qualification.load_sealed(selection_path)
        if _sha_file(selection_path) != planned["selection"]["sha256"]:
            raise RecoveryRunnerError("candidate selection bytes changed")
        try:
            runtime_document, _payload, metadata = maintained.export_model.validate_runtime(
                runtime
            )
            header, rendered = maintained.export_model.render_header(runtime)
            _default, generated = maintained.export_submission.render(model_header=header)
        except Exception as error:
            raise RecoveryRunnerError("candidate runtime/source export failed") from error
        identity = planned["runtime_identity"]
        if (
            metadata.get("file_sha256") != planned["runtime"]["sha256"]
            or rendered.get("file_sha256") != planned["runtime"]["sha256"]
            or runtime_document.get("architecture", {}).get("dimensions")
            != [6301, 12, 8, 1]
            or runtime_document.get("body_sha256") != identity["body_sha256"]
            or runtime_document.get("quantization", {}).get("payload_sha256")
            != identity["payload_sha256"]
            or generated != source.read_bytes()
        ):
            raise RecoveryRunnerError("candidate runtime/source no longer reproduces")
        binary = _verify_record(
            self.plan["binaries"][development.CANDIDATE_ID],
            "original candidate gate binary",
        )
        return maintained.Candidate(
            candidate_id=development.CANDIDATE_ID,
            architecture=development.CAPACITY_ARCHITECTURE,
            target="search-target",
            selection_path=selection_path,
            selection_sha256=planned["selection"]["sha256"],
            selection_body_sha256=selection["body_sha256"],
            runtime_path=runtime,
            runtime_sha256=planned["runtime"]["sha256"],
            deployment_eligible=True,
            source_path=source,
            source_sha256=planned["generated_source"]["sha256"],
            source_bytes=planned["generated_source"]["bytes"],
            binary_path=binary,
            binary_sha256=self.plan["binaries"][development.CANDIDATE_ID][
                "sha256"
            ],
        )

    def _validate_control_binary_contract(self) -> None:
        binary_record = self.plan["binaries"][development.CONTROL_ID]
        _verify_record(binary_record, "original control gate binary")
        compile_record = self.plan["compile_references"][development.CONTROL_ID]
        compile_path = _verify_sealed_record(
            compile_record,
            "papersoccer.compact-value-bfm.discrete-v3-development-binary-reference.v1",
            "original control compile reference",
        )
        value = qualification.load_sealed(compile_path)
        if (
            value.get("candidate_id") != development.CONTROL_ID
            or value.get("development_plan") != self._original_plan_record()
            or value.get("compiler") != self.compiler_identity
            or value.get("gate_source") != self.plan["tools"]["gate_source"]
            or value.get("rank4_source") != self.plan["tools"]["rank4_source"]
            or value.get("binary") != binary_record
            or value.get("candidate_source_sha256")
            != self.plan["rank4_control"]["rendered_source"]["sha256"]
        ):
            raise RecoveryRunnerError("original control compile/binary binding changed")

    def _candidate_binding(self) -> dict[str, Any]:
        if self.candidate is None:
            raise RecoveryRunnerError("candidate was not prepared")
        planned = self.plan["candidate"]
        identity = planned["runtime_identity"]
        return {
            "candidate_id": self.candidate.candidate_id,
            "architecture": self.candidate.architecture,
            "target": self.candidate.target,
            "selection_sha256": self.candidate.selection_sha256,
            "selection_body_sha256": self.candidate.selection_body_sha256,
            "runtime_sha256": self.candidate.runtime_sha256,
            "runtime_body_sha256": identity["body_sha256"],
            "payload_sha256": identity["payload_sha256"],
            "source_sha256": self.candidate.source_sha256,
            "source_bytes": self.candidate.source_bytes,
            "binary_path": str(self.candidate.binary_path.resolve()),
            "binary_sha256": self.candidate.binary_sha256,
            "binary_bytes": self.candidate.binary_path.stat().st_size,
        }

    def _bank_binding(self, stage: str) -> dict[str, Any]:
        bank = self.banks[stage]
        return {
            "stage": stage,
            "manifest_path": str(bank.manifest_path.resolve()),
            "manifest_bytes": bank.manifest_path.stat().st_size,
            "manifest_sha256": bank.manifest_sha256,
            "gate_path": str(bank.path.resolve()),
            "gate_bytes": bank.path.stat().st_size,
            "gate_sha256": bank.sha256,
        }

    def _expected_configuration(
        self,
        *,
        stage: str,
        tuple_values: Sequence[str],
        work: Mapping[str, int],
        pairs: int,
    ) -> dict[str, Any]:
        actual = stage == "actual_clock"
        return {
            "mode": "actual-clock" if actual else "fixed-work",
            "pair_offset": 0,
            "pair_count": pairs,
            "candidate_c": float(tuple_values[0]),
            "candidate_fpu": float(tuple_values[1]),
            "candidate_lambda": float(tuple_values[2]),
            "candidate_actions": int(self.plan["algorithm"]["candidate_actions"]),
            "candidate_root_partial_paths": int(work["root_partial_paths"]),
            "candidate_nonroot_partial_paths": int(
                work["nonroot_partial_paths"]
            ),
            "candidate_nodes": int(work["nodes"]),
            "candidate_expansions": int(
                self.plan["algorithm"]["candidate_expansions"]
            ),
            "candidate_shuffle_seed": int(
                self.plan["algorithm"]["candidate_shuffle_seed"]
            ),
            "candidate_clocks_ms": [800, 155],
            "rank4_nodes": int(self.plan["algorithm"]["rank4_nodes"]),
            "rank4_clocks_ms": [800, 165],
            "max_turns": int(self.plan["algorithm"]["max_turns"]),
            "minimum_candidate_wins": 211 if actual else -1,
            "minimum_wins_per_color": 104 if actual else -1,
        }

    def _gate_command(
        self,
        *,
        stage: str,
        tuple_values: Sequence[str],
        work: Mapping[str, int],
        output: pathlib.Path,
    ) -> tuple[str, ...]:
        if self.candidate is None:
            raise RecoveryRunnerError("candidate was not prepared")
        bank = self.banks[stage]
        actual = stage == "actual_clock"
        command = [
            str(self.candidate.binary_path),
            "--bank",
            str(bank.path),
            "--expected-bank-sha256",
            bank.sha256,
            "--candidate-source",
            str(self.candidate.source_path),
            "--expected-candidate-sha256",
            self.candidate.source_sha256,
            "--rank4-source",
            str(pathlib.Path(self.plan["tools"]["rank4_source"]["path"])),
            "--pair-offset",
            "0",
            "--pair-count",
            str(bank.pairs),
            "--mode",
            "actual-clock" if actual else "fixed-work",
            "--candidate-c",
            str(tuple_values[0]),
            "--candidate-fpu",
            str(tuple_values[1]),
            "--candidate-lambda",
            str(tuple_values[2]),
            "--candidate-actions",
            str(self.plan["algorithm"]["candidate_actions"]),
            "--candidate-root-partial-paths",
            str(work["root_partial_paths"]),
            "--candidate-nonroot-partial-paths",
            str(work["nonroot_partial_paths"]),
            "--candidate-nodes",
            str(work["nodes"]),
            "--candidate-expansions",
            str(self.plan["algorithm"]["candidate_expansions"]),
            "--candidate-seed",
            str(self.plan["algorithm"]["candidate_shuffle_seed"]),
            "--rank4-nodes",
            str(self.plan["algorithm"]["rank4_nodes"]),
            "--max-turns",
            str(self.plan["algorithm"]["max_turns"]),
            "--output",
            str(output),
        ]
        if actual:
            command.extend([
                "--minimum-candidate-wins",
                "211",
                "--minimum-wins-per-color",
                "104",
            ])
        return tuple(command)

    def _build_job(
        self,
        *,
        stage: str,
        candidate_id: str,
        tuple_values: Sequence[str],
        work: Mapping[str, int],
        metric_extra: Mapping[str, Any],
    ) -> Job:
        if self.compile_reference is None:
            raise RecoveryRunnerError("candidate compile reference is absent")
        if stage not in FRESH_STAGES:
            raise RecoveryRunnerError("recovery attempted an unplanned fresh stage")
        bank = self.banks[stage]
        values = tuple(str(item) for item in tuple_values)
        if len(values) != 3:
            raise RecoveryRunnerError("recovery search tuple is malformed")
        mode = "actual-clock" if stage == "actual_clock" else "fixed-work"
        spec = {
            "stage": stage,
            "candidate_id": candidate_id,
            "mode": mode,
            "tuple": list(values),
            "work": dict(work),
            "pairs": bank.pairs,
        }
        body = {
            "schema": REQUEST_SCHEMA,
            "namespace": development.NAMESPACE,
            "source_campaign_id": self.plan["source_campaign_id"],
            "recovery_id": self.plan["recovery_id"],
            "recovery_plan": dict(self.plan_record),
            "original_development_plan": dict(self._original_plan_record()),
            "candidate": self._candidate_binding(),
            "bank": self._bank_binding(stage),
            "spec": spec,
            "metric_extra": dict(metric_extra),
            "compile_reference": dict(self.compile_reference),
            "expected_configuration": self._expected_configuration(
                stage=stage,
                tuple_values=values,
                work=work,
                pairs=bank.pairs,
            ),
            "compiler": dict(self.compiler_identity),
            "gate_source": dict(self.plan["tools"]["gate_source"]),
            "rank4_source": dict(self.plan["tools"]["rank4_source"]),
            "policy": {
                "full_roster_preclaim_required": True,
                "no_retry_after_claim": True,
                "protected_data_access_authorized": False,
                "upload_authorized": False,
            },
        }
        request = qualification.seal(body)
        request_raw = _canonical(request)
        request_path = self.routes["requests"] / (
            f"{_sha_bytes(request_raw)}.request.json"
        )
        if request_path.exists():
            if request_path.is_symlink() or request_path.read_bytes() != request_raw:
                raise RecoveryRunnerError("recovery request bytes changed")
        elif self.read_only:
            raise RecoveryRunnerError("recovery request is absent")
        else:
            qualification.atomic_write_once(request_path, request_raw)
        request_sha = _sha_file(request_path)
        staging = self.routes["scratch"] / f".{request_sha}.gate.json.partial"
        final = self.routes["scratch"] / f"{request_sha}.gate.json"
        stdout = self.routes["scratch"] / f"{request_sha}.stdout.log"
        stderr = self.routes["scratch"] / f"{request_sha}.stderr.log"
        return Job(
            stage=stage,
            candidate_id=candidate_id,
            tuple_values=values,
            work=dict(work),
            pairs=bank.pairs,
            mode=mode,
            metric_extra=dict(metric_extra),
            request=request,
            request_path=request_path,
            request_sha256=request_sha,
            staging_output=staging,
            final_output=final,
            stdout_path=stdout,
            stderr_path=stderr,
            command=self._gate_command(
                stage=stage,
                tuple_values=values,
                work=work,
                output=staging,
            ),
        )

    def _validate_request(self, job: Job) -> None:
        value = qualification.load_sealed(job.request_path, REQUEST_SCHEMA)
        if (
            value != job.request
            or _sha_file(job.request_path) != job.request_sha256
            or job.request_path.parent != self.routes["requests"]
        ):
            raise RecoveryRunnerError("recovery request bytes changed")

    def _validate_gate_counts(
        self, document: Mapping[str, Any], *, pairs: int
    ) -> None:
        validate_real_gate_counts(document, pairs=pairs)

    def _validate_gate_output(
        self, job: Job, path: pathlib.Path
    ) -> Mapping[str, Any]:
        if path.is_symlink() or not path.is_file():
            raise RecoveryRunnerError("recovery gate output is absent or redirected")
        size = path.stat().st_size
        if size <= 0 or size > MAX_RESULT_BYTES:
            raise RecoveryRunnerError("recovery gate output size is implausible")
        try:
            document = maintained.gate_support.validate_result(
                path,
                expected_bank_sha256=self.banks[job.stage].sha256,
                expected_candidate_sha256=self.candidate.source_sha256,
                allow_legacy_attempt_zero=True,
            )
        except Exception as error:
            raise RecoveryRunnerError(
                f"recovery gate result failed validation: {job.candidate_id}"
            ) from error
        self._validate_gate_counts(document, pairs=job.pairs)
        expected_bindings = {
            "candidate_source_sha256": self.candidate.source_sha256,
            "candidate_source_bytes": self.candidate.source_bytes,
            "candidate_runtime_body_sha256": self.plan["candidate"][
                "runtime_identity"
            ]["body_sha256"],
            "candidate_payload_sha256": self.plan["candidate"][
                "runtime_identity"
            ]["payload_sha256"],
            "rank4_source_sha256": maintained.RANK4_SHA256,
            "rank4_source_bytes": pathlib.Path(
                self.plan["tools"]["rank4_source"]["path"]
            ).stat().st_size,
            "opponent_sha256": maintained.RANK4_SHA256,
            "bank_sha256": self.banks[job.stage].sha256,
            "bank_bytes": self.banks[job.stage].path.stat().st_size,
        }
        if (
            document.get("bindings") != expected_bindings
            or maintained.gate_support.legacy_standard_configuration(document)
            != job.request["expected_configuration"]
        ):
            raise RecoveryRunnerError("recovery gate binding/configuration changed")
        return document

    def _stage_concurrency(self, stage: str, jobs: Sequence[Job]) -> Mapping[str, Any]:
        concurrency = self.plan.get("concurrency")
        if not isinstance(concurrency, Mapping):
            raise RecoveryRunnerError("recovery concurrency policy is absent")
        stage_policy = concurrency.get(stage)
        if not isinstance(stage_policy, Mapping):
            raise RecoveryRunnerError(f"recovery {stage} concurrency policy is absent")
        expected_parallel = 1 if stage == "actual_clock" else len(jobs)
        maximum = stage_policy.get("maximum_concurrent_jobs")
        maximum_valid = (
            isinstance(maximum, int)
            and not isinstance(maximum, bool)
            and maximum >= expected_parallel
            and (
                (stage == "profile_confirmation" and maximum == 3)
                or (stage != "profile_confirmation" and maximum == expected_parallel)
            )
        )
        if (
            not maximum_valid
            or stage_policy.get("process_nice") != 0
            or stage_policy.get("no_retry_after_claim") is not True
            or (
                stage in CONCURRENT_STAGES
                and (
                    stage_policy.get("full_roster_required") is not True
                    or stage_policy.get("all_jobs_equal_nice") is not True
                    or stage_policy.get("latency_comparison")
                    != "same-stage-concurrent-only"
                )
            )
            or (
                stage == "actual_clock"
                and stage_policy.get("latency_comparison")
                != "serial-actual-clock"
            )
        ):
            raise RecoveryRunnerError(f"recovery {stage} execution policy changed")
        fixed_jobs = stage_policy.get("jobs")
        if fixed_jobs is not None and fixed_jobs != len(jobs):
            raise RecoveryRunnerError(f"recovery {stage} job count changed")
        if os.getpriority(os.PRIO_PROCESS, 0) != stage_policy["process_nice"]:
            raise RecoveryRunnerError(
                "recovery runner nice value differs from the stage policy"
            )
        return stage_policy

    def _claim_path(self, stage: str) -> pathlib.Path:
        return self.routes["claims"] / f"{stage}.claim.json"

    def _claim_roster(self, jobs: Sequence[Job]) -> list[dict[str, Any]]:
        return [
            {
                "candidate_id": job.candidate_id,
                "request": _sealed_record(job.request_path, REQUEST_SCHEMA),
                "tuple": list(job.tuple_values),
                "work": dict(job.work),
                "pairs": job.pairs,
                "mode": job.mode,
                "staging_output": str(job.staging_output),
                "final_output": str(job.final_output),
                "stdout": str(job.stdout_path),
                "stderr": str(job.stderr_path),
                "command_sha256": _sha_bytes(_canonical(list(job.command))),
            }
            for job in jobs
        ]

    def _claim_stage(
        self, stage: str, jobs: Sequence[Job], policy: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        path = self._claim_path(stage)
        if _lexists(path):
            raise RecoveryRunnerError(f"recovery {stage} was already claimed")
        for job in jobs:
            self._validate_request(job)
            for output in (
                job.staging_output,
                job.final_output,
                job.stdout_path,
                job.stderr_path,
                self.routes["references"] / f"{job.request_sha256}.json",
            ):
                if _lexists(output):
                    raise RecoveryRunnerError(
                        f"unclaimed recovery output exists for {job.candidate_id}"
                    )
        roster = self._claim_roster(jobs)
        roster_sha = _sha_bytes(_canonical(roster))
        claim = qualification.write_sealed(path, {
            "schema": CLAIM_SCHEMA,
            "namespace": development.NAMESPACE,
            "source_campaign_id": self.plan["source_campaign_id"],
            "recovery_id": self.plan["recovery_id"],
            "recovery_plan": dict(self.plan_record),
            "stage": stage,
            "created_at_utc": _utc_now(),
            "roster_sha256": roster_sha,
            "jobs": roster,
            "concurrency": dict(policy),
            "full_roster_preclaimed": True,
            "no_retry": True,
            "complete": False,
        })
        _append_event(
            self.routes["journal"],
            plan_record=self.plan_record,
            recovery_id=self.plan["recovery_id"],
            event="stage-claimed",
            stage=stage,
            claim=_sealed_record(path, CLAIM_SCHEMA),
            roster_sha256=roster_sha,
        )
        return claim

    def _validate_claim(self, stage: str, jobs: Sequence[Job]) -> Mapping[str, Any]:
        path = self._claim_path(stage)
        try:
            claim = qualification.load_sealed(path, CLAIM_SCHEMA)
        except Exception as error:
            raise RecoveryRunnerError(f"recovery {stage} claim is invalid") from error
        roster = self._claim_roster(jobs)
        try:
            qualification._utc(claim.get("created_at_utc"), "stage claim timestamp")
        except Exception as error:
            raise RecoveryRunnerError(f"recovery {stage} claim timestamp changed") from error
        if (
            claim.get("namespace") != development.NAMESPACE
            or claim.get("source_campaign_id") != self.plan["source_campaign_id"]
            or claim.get("recovery_plan") != self.plan_record
            or claim.get("recovery_id") != self.plan["recovery_id"]
            or claim.get("stage") != stage
            or claim.get("roster_sha256") != _sha_bytes(_canonical(roster))
            or claim.get("jobs") != roster
            or claim.get("full_roster_preclaimed") is not True
            or claim.get("no_retry") is not True
            or claim.get("complete") is not False
            or claim.get("concurrency") != self.plan["concurrency"][stage]
        ):
            raise RecoveryRunnerError(f"recovery {stage} claim changed")
        return claim

    def _stage_events(self, stage: str) -> list[Mapping[str, Any]]:
        return [
            event
            for event in _journal_entries(self.routes["journal"])
            if event.get("stage") == stage
        ]

    def _validate_journal_binding(self) -> list[dict[str, Any]]:
        entries = _journal_entries(self.routes["journal"])
        for entry in entries:
            try:
                qualification._utc(
                    entry.get("created_at_utc"), "recovery journal timestamp"
                )
            except Exception as error:
                raise RecoveryRunnerError(
                    "recovery journal timestamp changed"
                ) from error
            if (
                entry.get("namespace") != development.NAMESPACE
                or entry.get("recovery_id") != self.plan["recovery_id"]
                or entry.get("recovery_plan") != self.plan_record
                or (
                    entry.get("stage") not in FRESH_STAGES
                    and entry.get("stage") is not None
                )
            ):
                raise RecoveryRunnerError("recovery journal plan binding changed")
        return entries

    def _assert_no_competing_gates(self) -> None:
        processes = _workspace_gate_processes()
        if processes:
            raise RecoveryRunnerError(
                "another Rank-4 gate is active in this workspace"
            )

    def _terminal(self, stage: str | None, reason: str) -> None:
        entries = _journal_entries(self.routes["journal"])
        if entries and entries[-1]["event"] in {
            "terminal-failure", "campaign-complete",
        }:
            return
        _append_event(
            self.routes["journal"],
            plan_record=self.plan_record,
            recovery_id=self.plan["recovery_id"],
            event="terminal-failure",
            stage=stage,
            reason=reason,
            no_retry=True,
        )

    @contextlib.contextmanager
    def _termination_guard(self) -> Iterator[None]:
        previous: dict[int, Any] = {}

        def interrupted(signum: int, _frame: Any) -> None:
            raise KeyboardInterrupt(f"recovery interrupted by signal {signum}")

        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, interrupted)
        try:
            yield
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)

    @contextlib.contextmanager
    def _private_umask(self) -> Iterator[None]:
        previous = os.umask(0o077)
        try:
            yield
        finally:
            os.umask(previous)

    def _open_log(self, path: pathlib.Path) -> Any:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags, 0o600)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise RecoveryRunnerError("recovery gate log is not regular")
        return os.fdopen(descriptor, "wb", buffering=0)

    def _execute_real_batch(self, stage: str, jobs: Sequence[Job]) -> None:
        environment = os.environ.copy()
        thread_environment = self.plan["concurrency"]["thread_environment"]
        environment.update({str(key): str(value) for key, value in thread_environment.items()})
        processes: list[tuple[Job, subprocess.Popen[bytes]]] = []
        launch_started = time.monotonic()
        try:
            for job in jobs:
                stdout = self._open_log(job.stdout_path)
                try:
                    stderr = self._open_log(job.stderr_path)
                except BaseException:
                    stdout.close()
                    raise
                try:
                    process = subprocess.Popen(
                        list(job.command),
                        cwd=ROOT,
                        env=environment,
                        stdin=subprocess.DEVNULL,
                        stdout=stdout,
                        stderr=stderr,
                        close_fds=True,
                    )
                finally:
                    stdout.close()
                    stderr.close()
                processes.append((job, process))
                process_nice = os.getpriority(os.PRIO_PROCESS, process.pid)
                _append_event(
                    self.routes["journal"],
                    plan_record=self.plan_record,
                    recovery_id=self.plan["recovery_id"],
                    event="job-started",
                    stage=stage,
                    candidate_id=job.candidate_id,
                    request_sha256=job.request_sha256,
                    pid=process.pid,
                    command_sha256=_sha_bytes(_canonical(list(job.command))),
                    process_nice=process_nice,
                    injected=False,
                )
            launch_window = time.monotonic() - launch_started
            if launch_window > MAX_BATCH_LAUNCH_SECONDS:
                raise RecoveryRunnerError(
                    "recovery full-batch launch exceeded five seconds"
                )
            priorities = [
                os.getpriority(os.PRIO_PROCESS, process.pid)
                for _job, process in processes
            ]
            expected_nice = self.plan["concurrency"][stage]["process_nice"]
            if priorities != [expected_nice] * len(processes):
                raise RecoveryRunnerError("recovery gate jobs do not have equal nice")
            _append_event(
                self.routes["journal"],
                plan_record=self.plan_record,
                recovery_id=self.plan["recovery_id"],
                event="batch-launched",
                stage=stage,
                candidate_ids=[job.candidate_id for job in jobs],
                process_nice=priorities,
                launch_window_seconds=launch_window,
                maximum_launch_window_seconds=MAX_BATCH_LAUNCH_SECONDS,
                injected=False,
            )
            for job, process in processes:
                returncode = process.wait()
                if returncode not in (0, 2):
                    raise RecoveryRunnerError(
                        f"recovery gate exited {returncode}: {job.candidate_id}"
                    )
            for job, _process in processes:
                if (
                    job.stdout_path.is_symlink()
                    or not job.stdout_path.is_file()
                    or job.stderr_path.is_symlink()
                    or not job.stderr_path.is_file()
                    or job.stderr_path.read_bytes() != b""
                    or not job.staging_output.is_file()
                    or job.stdout_path.read_bytes() != job.staging_output.read_bytes()
                ):
                    raise RecoveryRunnerError(
                        f"recovery gate stdio contract failed: {job.candidate_id}"
                    )
                first = (_record(job.stdout_path), _record(job.staging_output))
                second = (_record(job.stdout_path), _record(job.staging_output))
                if first != second:
                    raise RecoveryRunnerError(
                        f"recovery gate output changed after exit: {job.candidate_id}"
                    )
        except BaseException:
            for _job, process in processes:
                if process.poll() is None:
                    process.terminate()
            for _job, process in processes:
                with contextlib.suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=10)
                if process.poll() is None:
                    process.kill()
                    process.wait()
            raise

    def _execute_injected_batch(self, stage: str, jobs: Sequence[Job]) -> None:
        assert self.gate_executor is not None
        launch_started = time.monotonic()

        def execute(job: Job) -> None:
            document = self.gate_executor(
                self.candidate,
                self.banks[job.stage],
                job.request["spec"],
            )
            if not isinstance(document, Mapping):
                raise RecoveryRunnerError("injected gate did not return an object")
            _atomic_private_json(job.staging_output, document)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(jobs), thread_name_prefix=f"recovery-{stage}"
        ) as pool:
            futures = [pool.submit(execute, job) for job in jobs]
            for job in jobs:
                _append_event(
                    self.routes["journal"],
                    plan_record=self.plan_record,
                    recovery_id=self.plan["recovery_id"],
                    event="job-started",
                    stage=stage,
                    candidate_id=job.candidate_id,
                    request_sha256=job.request_sha256,
                    injected=True,
                    process_nice=0,
                )
            launch_window = time.monotonic() - launch_started
            if launch_window > MAX_BATCH_LAUNCH_SECONDS:
                raise RecoveryRunnerError(
                    "recovery injected full-batch launch exceeded five seconds"
                )
            _append_event(
                self.routes["journal"],
                plan_record=self.plan_record,
                recovery_id=self.plan["recovery_id"],
                event="batch-launched",
                stage=stage,
                candidate_ids=[job.candidate_id for job in jobs],
                process_nice=[0] * len(jobs),
                injected=True,
                launch_window_seconds=launch_window,
                maximum_launch_window_seconds=MAX_BATCH_LAUNCH_SECONDS,
            )
            for future in futures:
                future.result()

    def _seal_fresh_receipt(
        self, job: Job, document: Mapping[str, Any]
    ) -> FreshEvidence:
        metric = maintained._metric(
            document,
            job.candidate_id,
            job.pairs,
            **dict(job.metric_extra),
        )
        gate_record = _record(job.final_output)
        base_path, _base = _write_content_addressed(
            self.routes["base_receipts"],
            {
                "schema": BASE_RECEIPT_SCHEMA,
                "namespace": development.NAMESPACE,
                "recovery_id": self.plan["recovery_id"],
                "recovery_plan": dict(self.plan_record),
                "request": _sealed_record(job.request_path, REQUEST_SCHEMA),
                "request_sha256": job.request_sha256,
                "gate_output": gate_record,
                "gate_result_sha256": _sha_bytes(_canonical(document)),
                "result_games": document["result"]["games"],
                "result_unfinished": document["result"]["unfinished"],
                "complete": True,
            },
            ".gate-receipt.json",
        )
        receipt_path, _receipt = _write_content_addressed(
            self.routes["receipts"],
            {
                "schema": RECEIPT_SCHEMA,
                "namespace": development.NAMESPACE,
                "source_campaign_id": self.plan["source_campaign_id"],
                "recovery_id": self.plan["recovery_id"],
                "recovery_plan": dict(self.plan_record),
                "request": _sealed_record(job.request_path, REQUEST_SCHEMA),
                "request_sha256": job.request_sha256,
                "base_receipt": _sealed_record(base_path, BASE_RECEIPT_SCHEMA),
                "gate_output": gate_record,
                "metric": metric,
                "complete": True,
                "final_bank_generation_authorized": False,
                "rank4_gate_authorized": False,
                "upload_authorized": False,
            },
            ".receipt.json",
        )
        reference_path = self.routes["references"] / f"{job.request_sha256}.json"
        reference = qualification.write_sealed(reference_path, {
            "schema": RECEIPT_REFERENCE_SCHEMA,
            "namespace": development.NAMESPACE,
            "recovery_id": self.plan["recovery_id"],
            "recovery_plan": dict(self.plan_record),
            "request": _sealed_record(job.request_path, REQUEST_SCHEMA),
            "receipt": _sealed_record(receipt_path, RECEIPT_SCHEMA),
            "complete": True,
        })
        return FreshEvidence(
            job=job,
            document=dict(document),
            metric=metric,
            receipt_record=_sealed_record(receipt_path, RECEIPT_SCHEMA),
            reference_record=_sealed_record(
                reference_path, RECEIPT_REFERENCE_SCHEMA
            ),
        )

    def _validate_fresh_receipt(self, job: Job) -> FreshEvidence:
        self._validate_request(job)
        reference_path = self.routes["references"] / f"{job.request_sha256}.json"
        try:
            reference = qualification.load_sealed(
                reference_path, RECEIPT_REFERENCE_SCHEMA
            )
        except Exception as error:
            raise RecoveryRunnerError(
                f"recovery receipt reference is invalid: {job.candidate_id}"
            ) from error
        receipt_path = _verify_sealed_record(
            reference.get("receipt"), RECEIPT_SCHEMA, "recovery receipt"
        )
        receipt = qualification.load_sealed(receipt_path, RECEIPT_SCHEMA)
        base_path = _verify_sealed_record(
            receipt.get("base_receipt"),
            BASE_RECEIPT_SCHEMA,
            "recovery gate receipt",
        )
        base = qualification.load_sealed(base_path, BASE_RECEIPT_SCHEMA)
        output = _verify_record(base.get("gate_output"), "recovery gate output")
        if output != job.final_output or job.staging_output.exists():
            raise RecoveryRunnerError("recovery gate publication path changed")
        document = self._validate_gate_output(job, output)
        metric = maintained._metric(
            document,
            job.candidate_id,
            job.pairs,
            **dict(job.metric_extra),
        )
        request_record = _sealed_record(job.request_path, REQUEST_SCHEMA)
        gate_record = _record(output)
        expected_base = qualification.seal({
            "schema": BASE_RECEIPT_SCHEMA,
            "namespace": development.NAMESPACE,
            "recovery_id": self.plan["recovery_id"],
            "recovery_plan": dict(self.plan_record),
            "request": request_record,
            "request_sha256": job.request_sha256,
            "gate_output": gate_record,
            "gate_result_sha256": _sha_bytes(_canonical(document)),
            "result_games": document["result"]["games"],
            "result_unfinished": document["result"]["unfinished"],
            "complete": True,
        })
        expected_receipt = qualification.seal({
            "schema": RECEIPT_SCHEMA,
            "namespace": development.NAMESPACE,
            "source_campaign_id": self.plan["source_campaign_id"],
            "recovery_id": self.plan["recovery_id"],
            "recovery_plan": dict(self.plan_record),
            "request": request_record,
            "request_sha256": job.request_sha256,
            "base_receipt": _sealed_record(base_path, BASE_RECEIPT_SCHEMA),
            "gate_output": gate_record,
            "metric": metric,
            "complete": True,
            "final_bank_generation_authorized": False,
            "rank4_gate_authorized": False,
            "upload_authorized": False,
        })
        expected_reference = qualification.seal({
            "schema": RECEIPT_REFERENCE_SCHEMA,
            "namespace": development.NAMESPACE,
            "recovery_id": self.plan["recovery_id"],
            "recovery_plan": dict(self.plan_record),
            "request": request_record,
            "receipt": _sealed_record(receipt_path, RECEIPT_SCHEMA),
            "complete": True,
        })
        if base != expected_base or receipt != expected_receipt or reference != expected_reference:
            raise RecoveryRunnerError("recovery receipt evidence changed")
        return FreshEvidence(
            job=job,
            document=dict(document),
            metric=metric,
            receipt_record=_sealed_record(receipt_path, RECEIPT_SCHEMA),
            reference_record=_sealed_record(
                reference_path, RECEIPT_REFERENCE_SCHEMA
            ),
        )

    def _validate_stage_chronology(
        self, stage: str, jobs: Sequence[Job], claim: Mapping[str, Any]
    ) -> list[Mapping[str, Any]]:
        self._validate_journal_binding()
        events = self._stage_events(stage)
        if not events or events[0]["event"] != "stage-claimed":
            raise RecoveryRunnerError(f"recovery {stage} claim event is absent")
        if events[0].get("claim") != _sealed_record(
            self._claim_path(stage), CLAIM_SCHEMA
        ):
            raise RecoveryRunnerError(f"recovery {stage} claim event changed")
        names = [event["event"] for event in events]
        for singleton in (
            "stage-claimed",
            "batch-launching",
            "batch-launched",
            "stage-complete",
        ):
            if names.count(singleton) > 1:
                raise RecoveryRunnerError(
                    f"recovery {stage} chronology repeats {singleton}"
                )
        rank = {
            "stage-claimed": 0,
            "batch-launching": 1,
            "job-started": 2,
            "batch-launched": 3,
            "result-validated": 4,
            "receipt-sealed": 5,
            "stage-complete": 6,
            "terminal-failure": 7,
        }
        observed_ranks = [rank[event["event"]] for event in events]
        if observed_ranks != sorted(observed_ranks):
            raise RecoveryRunnerError(f"recovery {stage} phase order changed")
        identifiers = [job.candidate_id for job in jobs]
        started = [
            event.get("candidate_id")
            for event in events
            if event["event"] == "job-started"
        ]
        if len(started) != len(set(started)) or any(
            identifier not in identifiers for identifier in started
        ):
            raise RecoveryRunnerError(f"recovery {stage} start roster changed")
        launches = [event for event in events if event["event"] == "batch-launched"]
        if launches:
            launch = launches[0]
            window = launch.get("launch_window_seconds")
            nice_values = launch.get("process_nice")
            if (
                launch.get("candidate_ids") != identifiers
                or started != identifiers
                or isinstance(window, bool)
                or not isinstance(window, (int, float))
                or not 0 <= float(window) <= MAX_BATCH_LAUNCH_SECONDS
                or launch.get("maximum_launch_window_seconds")
                != MAX_BATCH_LAUNCH_SECONDS
                or nice_values != [0] * len(jobs)
            ):
                raise RecoveryRunnerError(
                    f"recovery {stage} full-batch launch proof changed"
                )
        for event_name in ("result-validated", "receipt-sealed"):
            values = [
                event.get("candidate_id")
                for event in events
                if event["event"] == event_name
            ]
            if len(values) != len(set(values)) or any(
                identifier not in identifiers for identifier in values
            ):
                raise RecoveryRunnerError(
                    f"recovery {stage} {event_name} roster changed"
                )
        return events

    def _validate_completed_stage_chronology(
        self,
        stage: str,
        jobs: Sequence[Job],
        claim: Mapping[str, Any],
        evidence: Sequence[FreshEvidence],
    ) -> list[Mapping[str, Any]]:
        """Require the exact launch/result/receipt proof for a sealed stage."""

        events = self._validate_stage_chronology(stage, jobs, claim)
        count = len(jobs)
        expected_names = [
            "stage-claimed",
            "batch-launching",
            *(["job-started"] * count),
            "batch-launched",
            *(["result-validated"] * count),
            *(["receipt-sealed"] * count),
            "stage-complete",
        ]
        if [event["event"] for event in events] != expected_names:
            raise RecoveryRunnerError(
                f"recovery {stage} completed chronology is incomplete or reordered"
            )
        claim_record = _sealed_record(self._claim_path(stage), CLAIM_SCHEMA)
        identifiers = [job.candidate_id for job in jobs]
        if _event_details(events[0]) != {
            "claim": claim_record,
            "roster_sha256": claim["roster_sha256"],
        }:
            raise RecoveryRunnerError(f"recovery {stage} claim event fields changed")
        if _event_details(events[1]) != {
            "claim": claim_record,
            "candidate_ids": identifiers,
            "no_retry": True,
        }:
            raise RecoveryRunnerError(
                f"recovery {stage} batch-launching proof changed"
            )

        started = events[2 : 2 + count]
        launch = events[2 + count]
        injected = launch.get("injected") is True
        if injected and self.gate_executor is None:
            raise RecoveryRunnerError(
                f"recovery {stage} production chronology contains injected gates"
            )
        if launch.get("injected") not in {True, False}:
            raise RecoveryRunnerError(f"recovery {stage} launch type is absent")
        expected_nice = self.plan["concurrency"][stage]["process_nice"]
        for job, event in zip(jobs, started, strict=True):
            common = {
                "candidate_id": job.candidate_id,
                "request_sha256": job.request_sha256,
                "process_nice": expected_nice,
                "injected": injected,
            }
            if injected:
                expected = common
            else:
                pid = event.get("pid")
                if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
                    raise RecoveryRunnerError(
                        f"recovery {stage} job start has no positive PID"
                    )
                expected = {
                    **common,
                    "pid": pid,
                    "command_sha256": _sha_bytes(_canonical(list(job.command))),
                }
            if _event_details(event) != expected:
                raise RecoveryRunnerError(
                    f"recovery {stage} job-started proof changed: {job.candidate_id}"
                )
        window = launch.get("launch_window_seconds")
        if (
            isinstance(window, bool)
            or not isinstance(window, (int, float))
            or not math.isfinite(float(window))
            or not 0 <= float(window) <= MAX_BATCH_LAUNCH_SECONDS
        ):
            raise RecoveryRunnerError(
                f"recovery {stage} batch launch window is invalid"
            )
        if _event_details(launch) != {
            "candidate_ids": identifiers,
            "process_nice": [expected_nice] * count,
            "injected": injected,
            "launch_window_seconds": window,
            "maximum_launch_window_seconds": MAX_BATCH_LAUNCH_SECONDS,
        }:
            raise RecoveryRunnerError(
                f"recovery {stage} batch-launched proof changed"
            )

        result_start = 3 + count
        result_events = events[result_start : result_start + count]
        receipt_events = events[result_start + count : result_start + 2 * count]
        if len(evidence) != count:
            raise RecoveryRunnerError(f"recovery {stage} evidence count changed")
        for job, item, result_event, receipt_event in zip(
            jobs, evidence, result_events, receipt_events, strict=True
        ):
            document = item.document
            if _event_details(result_event) != {
                "candidate_id": job.candidate_id,
                "request_sha256": job.request_sha256,
                "output": _record(job.final_output),
                "result_games": document["result"]["games"],
                "result_unfinished": document["result"]["unfinished"],
            }:
                raise RecoveryRunnerError(
                    f"recovery {stage} result chronology changed: {job.candidate_id}"
                )
            if _event_details(receipt_event) != {
                "candidate_id": job.candidate_id,
                "request_sha256": job.request_sha256,
                "receipt_reference": dict(item.reference_record),
            }:
                raise RecoveryRunnerError(
                    f"recovery {stage} receipt chronology changed: {job.candidate_id}"
                )
        completion_details = {
            "claim": claim_record,
            "candidate_ids": identifiers,
            "receipt_references": [dict(item.reference_record) for item in evidence],
            "no_retry": True,
        }
        observed_completion = _event_details(events[-1])
        if observed_completion.get("recovered_without_replay") is True:
            completion_details["recovered_without_replay"] = True
        if observed_completion != completion_details:
            raise RecoveryRunnerError(
                f"recovery {stage} stage-complete proof changed"
            )
        return events

    def _recover_complete_outputs(
        self,
        stage: str,
        jobs: Sequence[Job],
        claim: Mapping[str, Any],
    ) -> list[FreshEvidence]:
        events = self._validate_stage_chronology(stage, jobs, claim)
        launches = [event for event in events if event["event"] == "batch-launched"]
        if len(launches) != 1:
            raise RecoveryRunnerError(
                f"claimed {stage} lacks complete full-batch launch proof"
            )
        self._assert_no_competing_gates()
        injected = launches[0].get("injected") is True
        staged: list[tuple[Job, pathlib.Path, Mapping[str, Any]]] = []
        for job in jobs:
            partial_exists = _lexists(job.staging_output)
            final_exists = _lexists(job.final_output)
            if partial_exists == final_exists:
                raise RecoveryRunnerError(
                    f"claimed {stage} does not have one complete output per job"
                )
            path = job.staging_output if partial_exists else job.final_output
            first_record = _record(path)
            first_document = self._validate_gate_output(job, path)
            second_record = _record(path)
            second_document = self._validate_gate_output(job, path)
            if first_record != second_record or first_document != second_document:
                raise RecoveryRunnerError(
                    f"claimed {stage} result is not stable"
                )
            if not injected and (
                job.stdout_path.is_symlink()
                or not job.stdout_path.is_file()
                or job.stderr_path.is_symlink()
                or not job.stderr_path.is_file()
                or job.stderr_path.read_bytes() != b""
                or job.stdout_path.read_bytes() != path.read_bytes()
            ):
                raise RecoveryRunnerError(
                    f"claimed {stage} completed output lacks exact gate stdio"
                )
            staged.append((job, path, first_document))

        documents: list[Mapping[str, Any]] = []
        for job, path, document in staged:
            if path == job.staging_output:
                _publish_no_replace(job.staging_output, job.final_output)
            published = self._validate_gate_output(job, job.final_output)
            if published != document:
                raise RecoveryRunnerError(
                    f"claimed {stage} output changed during adoption"
                )
            documents.append(published)

        existing_results = {
            event.get("candidate_id"): event
            for event in events
            if event["event"] == "result-validated"
        }
        for job, document in zip(jobs, documents, strict=True):
            expected = {
                "request_sha256": job.request_sha256,
                "output": _record(job.final_output),
                "result_games": document["result"]["games"],
                "result_unfinished": document["result"]["unfinished"],
            }
            prior = existing_results.get(job.candidate_id)
            if prior is not None:
                if any(prior.get(key) != value for key, value in expected.items()):
                    raise RecoveryRunnerError(
                        f"claimed {stage} recorded result changed"
                    )
            else:
                _append_event(
                    self.routes["journal"],
                    plan_record=self.plan_record,
                    recovery_id=self.plan["recovery_id"],
                    event="result-validated",
                    stage=stage,
                    candidate_id=job.candidate_id,
                    **expected,
                )
        evidence = [
            self._seal_fresh_receipt(job, document)
            for job, document in zip(jobs, documents, strict=True)
        ]
        events = self._stage_events(stage)
        existing_receipts = {
            event.get("candidate_id"): event
            for event in events
            if event["event"] == "receipt-sealed"
        }
        for item in evidence:
            expected = {
                "request_sha256": item.job.request_sha256,
                "receipt_reference": dict(item.reference_record),
            }
            prior = existing_receipts.get(item.job.candidate_id)
            if prior is not None:
                if any(prior.get(key) != value for key, value in expected.items()):
                    raise RecoveryRunnerError(
                        f"claimed {stage} recorded receipt changed"
                    )
            else:
                _append_event(
                    self.routes["journal"],
                    plan_record=self.plan_record,
                    recovery_id=self.plan["recovery_id"],
                    event="receipt-sealed",
                    stage=stage,
                    candidate_id=item.job.candidate_id,
                    **expected,
                )
        _append_event(
            self.routes["journal"],
            plan_record=self.plan_record,
            recovery_id=self.plan["recovery_id"],
            event="stage-complete",
            stage=stage,
            claim=_sealed_record(self._claim_path(stage), CLAIM_SCHEMA),
            candidate_ids=[job.candidate_id for job in jobs],
            receipt_references=[dict(item.reference_record) for item in evidence],
            no_retry=True,
            recovered_without_replay=True,
        )
        validated = [self._validate_fresh_receipt(job) for job in jobs]
        self._validate_completed_stage_chronology(
            stage, jobs, claim, validated
        )
        return validated

    def _completed_stage(
        self, stage: str, jobs: Sequence[Job]
    ) -> list[FreshEvidence] | None:
        claim_path = self._claim_path(stage)
        events = self._stage_events(stage)
        completions = [event for event in events if event["event"] == "stage-complete"]
        if not claim_path.exists():
            if events or any(
                (self.routes["references"] / f"{job.request_sha256}.json").exists()
                for job in jobs
            ):
                raise RecoveryRunnerError(
                    f"recovery {stage} evidence exists without a claim"
                )
            return None
        claim = self._validate_claim(stage, jobs)
        events = self._validate_stage_chronology(stage, jobs, claim)
        completions = [event for event in events if event["event"] == "stage-complete"]
        if len(completions) == 1:
            evidence = [self._validate_fresh_receipt(job) for job in jobs]
            expected_references = [dict(item.reference_record) for item in evidence]
            if (
                completions[0].get("claim")
                != _sealed_record(claim_path, CLAIM_SCHEMA)
                or completions[0].get("receipt_references") != expected_references
                or completions[0].get("candidate_ids")
                != [job.candidate_id for job in jobs]
            ):
                raise RecoveryRunnerError(f"recovery {stage} completion changed")
            self._validate_completed_stage_chronology(
                stage, jobs, claim, evidence
            )
            return evidence
        if completions:
            raise RecoveryRunnerError(f"recovery {stage} completed more than once")
        reason = f"claimed {stage} lacks a sealed complete event; replay forbidden"
        if self.read_only:
            raise TerminalRecoveryError(reason)
        try:
            return self._recover_complete_outputs(stage, jobs, claim)
        except Exception as error:
            self._terminal(stage, f"{reason}: {error}")
            raise TerminalRecoveryError(reason) from error

    def _run_stage(self, stage: str, jobs: Sequence[Job]) -> list[FreshEvidence]:
        existing = self._completed_stage(stage, jobs)
        if existing is not None:
            return existing
        policy = self._stage_concurrency(stage, jobs)
        self._assert_no_competing_gates()
        claim = self._claim_stage(stage, jobs, policy)
        claim_record = _sealed_record(self._claim_path(stage), CLAIM_SCHEMA)
        try:
            _append_event(
                self.routes["journal"],
                plan_record=self.plan_record,
                recovery_id=self.plan["recovery_id"],
                event="batch-launching",
                stage=stage,
                claim=claim_record,
                candidate_ids=[job.candidate_id for job in jobs],
                no_retry=True,
            )
            self._assert_no_competing_gates()
            with self._termination_guard():
                with self._private_umask():
                    if self.gate_executor is None:
                        self._execute_real_batch(stage, jobs)
                    else:
                        self._execute_injected_batch(stage, jobs)
            documents: list[Mapping[str, Any]] = []
            for job in jobs:
                document = self._validate_gate_output(job, job.staging_output)
                _publish_no_replace(job.staging_output, job.final_output)
                published = self._validate_gate_output(job, job.final_output)
                if document != published:
                    raise RecoveryRunnerError("recovery gate changed during publication")
                documents.append(published)
                _append_event(
                    self.routes["journal"],
                    plan_record=self.plan_record,
                    recovery_id=self.plan["recovery_id"],
                    event="result-validated",
                    stage=stage,
                    candidate_id=job.candidate_id,
                    request_sha256=job.request_sha256,
                    output=_record(job.final_output),
                    result_games=published["result"]["games"],
                    result_unfinished=published["result"]["unfinished"],
                )
            evidence = [
                self._seal_fresh_receipt(job, document)
                for job, document in zip(jobs, documents, strict=True)
            ]
            for item in evidence:
                _append_event(
                    self.routes["journal"],
                    plan_record=self.plan_record,
                    recovery_id=self.plan["recovery_id"],
                    event="receipt-sealed",
                    stage=stage,
                    candidate_id=item.job.candidate_id,
                    request_sha256=item.job.request_sha256,
                    receipt_reference=dict(item.reference_record),
                )
            _append_event(
                self.routes["journal"],
                plan_record=self.plan_record,
                recovery_id=self.plan["recovery_id"],
                event="stage-complete",
                stage=stage,
                claim=claim_record,
                candidate_ids=[job.candidate_id for job in jobs],
                receipt_references=[dict(item.reference_record) for item in evidence],
                no_retry=True,
            )
            validated = [self._validate_fresh_receipt(job) for job in jobs]
            self._validate_completed_stage_chronology(
                stage, jobs, claim, validated
            )
            return validated
        except BaseException as error:
            with contextlib.suppress(Exception):
                self._terminal(stage, f"{type(error).__name__}: {error}")
            raise

    def _tuple_confirmation_rows(
        self, evidence: Sequence[FreshEvidence]
    ) -> list[dict[str, Any]]:
        by_id = {item.job.candidate_id: item for item in evidence}
        default_id = (
            f"{development.CANDIDATE_ID}:"
            f"{campaign.tuple_id(campaign.DEFAULT_TUPLE)}"
        )
        if default_id not in by_id:
            raise RecoveryRunnerError("fresh tuple confirmation omits the default")
        default = by_id[default_id].document
        bank_sha = self.banks["tuple_confirmation"].sha256
        rows: list[dict[str, Any]] = []
        for item in evidence:
            row = dict(item.metric)
            row["paired_bootstrap_lower_95"] = (
                0.0
                if item.job.candidate_id == default_id
                else maintained.paired_bootstrap_lower(
                    item.document,
                    default,
                    f"tuple:{bank_sha}:{item.job.candidate_id}",
                    samples=development.BOOTSTRAP_SAMPLES,
                )
            )
            rows.append(row)
        return rows

    def _profile_confirmation_rows(
        self, evidence: Sequence[FreshEvidence]
    ) -> list[dict[str, Any]]:
        by_id = {item.job.candidate_id: item for item in evidence}
        default_id = campaign.DEFAULT_PROFILE
        if default_id not in by_id:
            raise RecoveryRunnerError("fresh profile confirmation omits the default")
        default = by_id[default_id].document
        bank_sha = self.banks["profile_confirmation"].sha256
        rows: list[dict[str, Any]] = []
        for item in evidence:
            row = dict(item.metric)
            row["paired_bootstrap_lower_95"] = (
                0.0
                if item.job.candidate_id == default_id
                else maintained.paired_bootstrap_lower(
                    item.document,
                    default,
                    f"profile:{bank_sha}:{item.job.candidate_id}",
                    samples=development.BOOTSTRAP_SAMPLES,
                )
            )
            rows.append(row)
        return rows

    def _execute_algorithm(
        self,
        original_rows: Sequence[Mapping[str, Any]],
        *,
        launch_missing: bool,
    ) -> dict[str, Any]:
        model_rows = [dict(row) for row in original_rows[:2]]
        tuple_rows = [dict(row) for row in original_rows[2:]]
        by_model = {row["candidate_id"]: row for row in model_rows}
        ranked = campaign._validate_exact_tuple_screen(
            tuple_rows,
            [by_model[development.CANDIDATE_ID]],
            by_model,
        )
        default_tuple_id = (
            f"{development.CANDIDATE_ID}:"
            f"{campaign.tuple_id(campaign.DEFAULT_TUPLE)}"
        )
        carried_tuples: list[str] = []
        for identifier in [row["candidate_id"] for row in ranked[:2]] + [
            default_tuple_id
        ]:
            if identifier not in carried_tuples:
                carried_tuples.append(identifier)
        planned_roster = self.plan["recovery_contract"][
            "tuple_confirmation_roster"
        ]
        if carried_tuples != [row["candidate_id"] for row in planned_roster]:
            raise RecoveryRunnerError("fresh tuple confirmation roster changed")
        descriptors = {row["candidate_id"]: row for row in tuple_rows}
        default_work = campaign.PROFILE_ROSTER[campaign.DEFAULT_PROFILE]
        tuple_jobs = [
            self._build_job(
                stage="tuple_confirmation",
                candidate_id=identifier,
                tuple_values=descriptors[identifier]["tuple"],
                work=default_work,
                metric_extra={
                    "model_id": development.CANDIDATE_ID,
                    "tuple": list(descriptors[identifier]["tuple"]),
                },
            )
            for identifier in carried_tuples
        ]
        tuple_evidence = (
            self._run_stage("tuple_confirmation", tuple_jobs)
            if launch_missing
            else self._require_completed_stage("tuple_confirmation", tuple_jobs)
        )
        tuple_confirmation = self._tuple_confirmation_rows(tuple_evidence)
        selected_tuple, normalized_tuples = campaign._confirmation_choice(
            tuple_confirmation,
            pairs=250,
            carried_ids=carried_tuples,
            default_id=default_tuple_id,
            architecture_by_id={
                identifier: development.CAPACITY_ARCHITECTURE
                for identifier in carried_tuples
            },
            label="tuple",
        )
        selected_tuple_values = tuple(str(item) for item in selected_tuple["tuple"])

        profile_jobs = [
            self._build_job(
                stage="profile_screen",
                candidate_id=profile,
                tuple_values=selected_tuple_values,
                work=work,
                metric_extra={"profile": profile, "work": dict(work)},
            )
            for profile, work in campaign.PROFILE_ROSTER.items()
        ]
        profile_evidence = (
            self._run_stage("profile_screen", profile_jobs)
            if launch_missing
            else self._require_completed_stage("profile_screen", profile_jobs)
        )
        profile_screen = [dict(item.metric) for item in profile_evidence]
        profiles = campaign._validate_profiles(
            profile_screen, pairs=100, label="recovery profile screen"
        )
        ranked_profiles = sorted(
            [row for row in profiles if row["failures"] == 0],
            key=lambda row: campaign._rank_key(
                row, development.CAPACITY_ARCHITECTURE
            ),
        )
        if len(ranked_profiles) < 2:
            raise TerminalRecoveryError(
                "profile screen has fewer than two failure-free profiles"
            )
        carried_profiles: list[str] = []
        for profile in [row["candidate_id"] for row in ranked_profiles[:2]] + [
            campaign.DEFAULT_PROFILE
        ]:
            if profile not in carried_profiles:
                carried_profiles.append(profile)
        confirmation_jobs = [
            self._build_job(
                stage="profile_confirmation",
                candidate_id=profile,
                tuple_values=selected_tuple_values,
                work=campaign.PROFILE_ROSTER[profile],
                metric_extra={
                    "profile": profile,
                    "work": dict(campaign.PROFILE_ROSTER[profile]),
                },
            )
            for profile in carried_profiles
        ]
        confirmation_evidence = (
            self._run_stage("profile_confirmation", confirmation_jobs)
            if launch_missing
            else self._require_completed_stage(
                "profile_confirmation", confirmation_jobs
            )
        )
        profile_confirmation = self._profile_confirmation_rows(
            confirmation_evidence
        )
        selected_profile, normalized_profiles = campaign._confirmation_choice(
            profile_confirmation,
            pairs=250,
            carried_ids=carried_profiles,
            default_id=campaign.DEFAULT_PROFILE,
            architecture_by_id={
                profile: development.CAPACITY_ARCHITECTURE
                for profile in carried_profiles
            },
            label="profile",
        )
        actual_id = (
            selected_tuple["candidate_id"]
            + ":"
            + selected_profile["candidate_id"]
        )
        actual_job = self._build_job(
            stage="actual_clock",
            candidate_id=actual_id,
            tuple_values=selected_tuple_values,
            work=campaign.PROFILE_ROSTER[selected_profile["candidate_id"]],
            metric_extra={},
        )
        actual_evidence = (
            self._run_stage("actual_clock", [actual_job])
            if launch_missing
            else self._require_completed_stage("actual_clock", [actual_job])
        )[0]
        actual = dict(actual_evidence.metric)
        if (
            actual["failures"] != 0
            or actual["wins"] < 211
            or actual["color_wins"]["0"] < 104
            or actual["color_wins"]["1"] < 104
            or actual_evidence.document["result"].get("passed") is not True
        ):
            raise TerminalRecoveryError(
                "recovery actual-clock 211/104/zero-failure gate failed"
            )
        all_fresh = [
            *tuple_evidence,
            *profile_evidence,
            *confirmation_evidence,
            actual_evidence,
        ]
        return {
            "rows": {
                "model_screen": model_rows,
                "tuple_screen": tuple_rows,
                "tuple_confirmation": normalized_tuples,
                "profile_screen": profile_screen,
                "profile_confirmation": normalized_profiles,
                "actual_clock": actual,
            },
            "selected": {
                "tuple_candidate_id": selected_tuple["candidate_id"],
                "tuple": list(selected_tuple["tuple"]),
                "profile": selected_profile["candidate_id"],
                "profile_work": dict(
                    campaign.PROFILE_ROSTER[selected_profile["candidate_id"]]
                ),
                "actual_clock_candidate_id": actual_id,
            },
            "fresh_evidence": all_fresh,
        }

    def _require_completed_stage(
        self, stage: str, jobs: Sequence[Job]
    ) -> list[FreshEvidence]:
        evidence = self._completed_stage(stage, jobs)
        if evidence is None:
            raise RecoveryRunnerError(f"recovery {stage} is incomplete")
        return evidence

    def _journal_summary(self) -> dict[str, Any]:
        entries = self._validate_journal_binding()
        stage_entries = [entry for entry in entries if entry["event"] != "campaign-complete"]
        completed_stages = [
            event["stage"]
            for event in stage_entries
            if event["event"] == "stage-complete"
        ]
        stage_ordinals = [
            FRESH_STAGES.index(str(event["stage"])) for event in stage_entries
        ]
        if (
            not stage_entries
            or stage_entries[-1]["event"] != "stage-complete"
            or stage_entries[-1]["stage"] != "actual_clock"
            or completed_stages != list(FRESH_STAGES)
            or stage_ordinals != sorted(stage_ordinals)
            or any(
                event["event"] == "terminal-failure" for event in stage_entries
            )
        ):
            raise RecoveryRunnerError("recovery stage journal is not complete")
        expected_claims = {f"{stage}.claim.json" for stage in FRESH_STAGES}
        if (
            self.routes["claims"].is_symlink()
            or not self.routes["claims"].is_dir()
            or {path.name for path in self.routes["claims"].iterdir()}
            != expected_claims
            or any(
                path.is_symlink() or not path.is_file()
                for path in self.routes["claims"].iterdir()
            )
        ):
            raise RecoveryRunnerError("recovery completed claim roster changed")
        paths = sorted(self.routes["journal"].glob("*.json"))
        head_index = entries.index(stage_entries[-1])
        return {
            "event_count": len(stage_entries),
            "head": _sealed_record(paths[head_index], JOURNAL_SCHEMA),
            "completed_stages": completed_stages,
            "terminal_failures": sum(
                event["event"] == "terminal-failure" for event in stage_entries
            ),
        }

    def _result_body(
        self,
        *,
        payload: Mapping[str, Any],
        original_records: Sequence[Mapping[str, Any]],
        completed_at_utc: str,
    ) -> dict[str, Any]:
        fresh = payload["fresh_evidence"]
        mixed_record = _sealed_record(
            self.routes["mixed_six_exclusion"], recovery.MIXED_EXCLUSION_SCHEMA
        )
        return {
            "schema": RESULT_SCHEMA,
            "namespace": development.NAMESPACE,
            "campaign_id": self.plan["campaign_id"],
            "source_campaign_id": self.plan["source_campaign_id"],
            "recovery_id": self.plan["recovery_id"],
            "status": "recovery-development-complete-awaiting-finalist-seal",
            "completed_at_utc": completed_at_utc,
            "recovery_plan": dict(self.plan_record),
            "original_development_plan": dict(self._original_plan_record()),
            "terminal_incident": dict(self.plan["original"]["terminal_incident"]),
            "mixed_six_exclusion": mixed_record,
            "additional_development_exclusions": dict(
                self.plan["additional_development_exclusions"]
            ),
            "banks": {
                stage: dict(self.plan_context["materialized_banks"][stage])
                for stage in development.STAGE_ORDER
            },
            "binaries": {
                name: dict(record) for name, record in self.plan["binaries"].items()
            },
            "rows": dict(payload["rows"]),
            "selected": dict(payload["selected"]),
            "run_receipts": {
                "carried_original": [dict(record) for record in original_records],
                "fresh_recovery": [
                    dict(item.reference_record) for item in fresh
                ],
            },
            "request_count": len(original_records) + len(fresh),
            "journal": self._journal_summary(),
            "policy": {
                "carried_receipts_immutable": True,
                "original_confirmation_selection_weight": 0,
                "fresh_bank_full_roster_replacement_only": True,
                "development_selected": True,
                "final_bank_generation_authorized": False,
                "rank4_gate_authorized": False,
                "upload_authorized": False,
            },
        }

    def _finalist_body(
        self,
        *,
        result_path: pathlib.Path,
        result: Mapping[str, Any],
    ) -> dict[str, Any]:
        original = self.original_plan
        if original is None:
            raise RecoveryRunnerError("original plan was not validated")
        selected = result["selected"]
        return {
            "schema": FINALIST_SCHEMA,
            "namespace": development.NAMESPACE,
            "campaign_id": self.plan["campaign_id"],
            "source_campaign_id": self.plan["source_campaign_id"],
            "recovery_id": self.plan["recovery_id"],
            "status": "development-selected-awaiting-preflight-and-frozen-final",
            "created_at_utc": result["completed_at_utc"],
            "recovery_plan": dict(self.plan_record),
            "recovery_result": _sealed_record(result_path, RESULT_SCHEMA),
            "original_development_plan": dict(self._original_plan_record()),
            "original_terminal_incident": dict(
                self.plan["original"]["terminal_incident"]
            ),
            "adapter": dict(original["adapter"]),
            "exclusion": dict(original["exclusion"]),
            "candidate": dict(self.plan["candidate"]),
            "rank4_control": dict(self.plan["rank4_control"]),
            "banks": dict(result["banks"]),
            "mixed_six_exclusion": dict(result["mixed_six_exclusion"]),
            "additional_development_exclusions": dict(
                result["additional_development_exclusions"]
            ),
            "binary": dict(self.plan["binaries"][development.CANDIDATE_ID]),
            "tuple": list(selected["tuple"]),
            "tuple_candidate_id": selected["tuple_candidate_id"],
            "profile": selected["profile"],
            "profile_work": dict(selected["profile_work"]),
            "actual_clock": dict(result["rows"]["actual_clock"]),
            "run_receipts": dict(result["run_receipts"]),
            "journal": dict(result["journal"]),
            "fresh_protected_tests_opened": True,
            "fresh_diagnostic_classification": (
                "diagnostic-only-no-pass-fail-verdict"
            ),
            "old_protected_tests_accessed": False,
            "model_weights_immutable": True,
            "search_configuration_immutable": True,
            "development_selected": True,
            "preflight_required": True,
            "final_bank_generation_authorized": False,
            "rank4_gate_authorized": False,
            "upload_authorized": False,
        }

    def _audit_fresh_files(self, evidence: Sequence[FreshEvidence]) -> None:
        expected_requests = {item.job.request_path.name for item in evidence}
        expected_references = {
            pathlib.Path(item.reference_record["path"]).name for item in evidence
        }
        expected_receipts = {
            pathlib.Path(item.receipt_record["path"]).name for item in evidence
        }
        expected_base = set()
        for item in evidence:
            receipt = qualification.load_sealed(
                pathlib.Path(item.receipt_record["path"]), RECEIPT_SCHEMA
            )
            expected_base.add(pathlib.Path(receipt["base_receipt"]["path"]).name)
        for path, expected, pattern, label in (
            (self.routes["requests"], expected_requests, "*.request.json", "requests"),
            (self.routes["references"], expected_references, "*.json", "references"),
            (self.routes["receipts"], expected_receipts, "*.receipt.json", "receipts"),
            (
                self.routes["base_receipts"],
                expected_base,
                "*.gate-receipt.json",
                "gate receipts",
            ),
        ):
            actual = {item.name for item in path.iterdir()}
            if actual != expected or any(
                item.is_symlink() or not item.is_file() for item in path.iterdir()
            ):
                raise RecoveryRunnerError(f"recovery {label} roster changed")
        expected_scratch = {item.job.final_output.name for item in evidence}
        if self.gate_executor is None:
            expected_scratch.update(
                path.name
                for item in evidence
                for path in (item.job.stdout_path, item.job.stderr_path)
            )
        scratch = self.routes["scratch"]
        if (
            scratch.is_symlink()
            or not scratch.is_dir()
            or {path.name for path in scratch.iterdir()} != expected_scratch
            or any(
                path.is_symlink() or not path.is_file()
                for path in scratch.iterdir()
            )
        ):
            raise RecoveryRunnerError("recovery completed scratch roster changed")
        if self.gate_executor is None:
            for item in evidence:
                job = item.job
                first = (
                    _record(job.stdout_path),
                    _record(job.stderr_path),
                    _record(job.final_output),
                )
                if (
                    job.stderr_path.read_bytes() != b""
                    or job.stdout_path.read_bytes()
                    != job.final_output.read_bytes()
                ):
                    raise RecoveryRunnerError(
                        f"recovery completed gate stdio changed: {job.candidate_id}"
                    )
                second = (
                    _record(job.stdout_path),
                    _record(job.stderr_path),
                    _record(job.final_output),
                )
                if first != second:
                    raise RecoveryRunnerError(
                        f"recovery completed gate stdio is unstable: {job.candidate_id}"
                    )

    def _seal_result_and_finalist(
        self,
        *,
        payload: Mapping[str, Any],
        original_records: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        self._audit_fresh_files(payload["fresh_evidence"])
        result_path = self.routes["result"]
        completed_at = (
            qualification.load_sealed(result_path, RESULT_SCHEMA)["completed_at_utc"]
            if result_path.exists()
            else _utc_now()
        )
        body = self._result_body(
            payload=payload,
            original_records=original_records,
            completed_at_utc=completed_at,
        )
        expected_result = qualification.seal(body)
        if result_path.exists():
            if qualification.load_sealed(result_path, RESULT_SCHEMA) != expected_result:
                raise RecoveryRunnerError("existing recovery result changed")
        elif self.read_only:
            raise RecoveryRunnerError("recovery result is absent")
        else:
            qualification.write_sealed(result_path, body)
        result = qualification.load_sealed(result_path, RESULT_SCHEMA)
        finalist_body = self._finalist_body(result_path=result_path, result=result)
        finalist = qualification.seal(finalist_body)
        finalist_raw = _canonical(finalist)
        finalist_path = self.routes["finalists"] / (
            f"{_sha_bytes(finalist_raw)}.finalist.json"
        )
        if finalist_path.exists():
            if finalist_path.read_bytes() != finalist_raw:
                raise RecoveryRunnerError("existing recovery finalist changed")
        elif self.read_only:
            raise RecoveryRunnerError("recovery finalist is absent")
        else:
            qualification.atomic_write_once(finalist_path, finalist_raw)
        reference_body = {
            "schema": FINALIST_REFERENCE_SCHEMA,
            "namespace": development.NAMESPACE,
            "campaign_id": self.plan["campaign_id"],
            "source_campaign_id": self.plan["source_campaign_id"],
            "recovery_id": self.plan["recovery_id"],
            "recovery_plan": dict(self.plan_record),
            "recovery_result": _sealed_record(result_path, RESULT_SCHEMA),
            "finalist": _sealed_record(finalist_path, FINALIST_SCHEMA),
            "complete": True,
            "final_bank_generation_authorized": False,
            "rank4_gate_authorized": False,
            "upload_authorized": False,
        }
        expected_reference = qualification.seal(reference_body)
        reference_path = self.routes["finalist_reference"]
        if reference_path.exists():
            if qualification.load_sealed(
                reference_path, FINALIST_REFERENCE_SCHEMA
            ) != expected_reference:
                raise RecoveryRunnerError("existing recovery finalist reference changed")
        elif self.read_only:
            raise RecoveryRunnerError("recovery finalist reference is absent")
        else:
            qualification.write_sealed(reference_path, reference_body)
        finalists = self.routes["finalists"]
        if (
            finalists.is_symlink()
            or not finalists.is_dir()
            or {path.name for path in finalists.iterdir()} != {finalist_path.name}
            or any(
                path.is_symlink() or not path.is_file()
                for path in finalists.iterdir()
            )
        ):
            raise RecoveryRunnerError("recovery finalist roster changed")
        return {
            "reference": qualification.load_sealed(
                reference_path, FINALIST_REFERENCE_SCHEMA
            ),
            "finalist": qualification.load_sealed(finalist_path, FINALIST_SCHEMA),
            "result": result,
            "path": finalist_path,
        }

    def _prepare_contract(self) -> tuple[list[dict[str, Any]], list[Mapping[str, Any]]]:
        if self.plan_context.get("materialized") is not True:
            raise RecoveryRunnerError(
                "recovery bank must be materialized before execution"
            )
        self._validate_routes()
        original_plan, rows, original_records = self._validate_original_receipts()
        self.original_plan = original_plan
        self.banks = self._load_banks()
        self._validate_control_binary_contract()
        self.candidate = self._candidate_contract()
        return rows, original_records

    def execute(self) -> dict[str, Any]:
        lock = self.recovery_root / "recovery.lock"
        with _exclusive_lock(lock):
            loaded = dict(
                self.plan_loader(self.plan_path, output_root=self.output_root)
            )
            self.plan_context = loaded
            self.plan = dict(loaded["plan"] if "plan" in loaded else loaded)
            self.plan_record = _sealed_record(self.plan_path, recovery.PLAN_SCHEMA)
            entries = _journal_entries(self.routes["journal"])
            if entries and entries[-1]["event"] == "terminal-failure":
                raise TerminalRecoveryError(
                    "recovery is terminal after a claimed-stage failure"
                )
            original_rows, original_records = self._prepare_contract()
            complete = self.routes["finalist_reference"].exists()
            try:
                payload = self._execute_algorithm(
                    original_rows, launch_missing=not complete
                )
                value = self._seal_result_and_finalist(
                    payload=payload,
                    original_records=original_records,
                )
                entries = _journal_entries(self.routes["journal"])
                completions = [
                    entry for entry in entries if entry["event"] == "campaign-complete"
                ]
                if completions:
                    expected_completion = {
                        "finalist_reference": _sealed_record(
                            self.routes["finalist_reference"],
                            FINALIST_REFERENCE_SCHEMA,
                        )
                    }
                    if (
                        len(completions) != 1
                        or completions[0] != entries[-1]
                        or _event_details(completions[0])
                        != expected_completion
                    ):
                        raise RecoveryRunnerError(
                            "recovery campaign completion event changed"
                        )
                elif not self.read_only:
                    _append_event(
                        self.routes["journal"],
                        plan_record=self.plan_record,
                        recovery_id=self.plan["recovery_id"],
                        event="campaign-complete",
                        stage=None,
                        finalist_reference=_sealed_record(
                            self.routes["finalist_reference"],
                            FINALIST_REFERENCE_SCHEMA,
                        ),
                    )
                return value
            except BaseException as error:
                # A stage method terminalizes any error after its immutable
                # claim.  Pre-claim validation failures remain repairable and
                # do not consume the one authorized recovery attempt.
                if any(self.routes["claims"].glob("*.claim.json")):
                    with contextlib.suppress(Exception):
                        self._terminal(None, f"{type(error).__name__}: {error}")
                raise


def validate_recovery_finalist(
    reference_path: pathlib.Path,
    *,
    plan_path: pathlib.Path,
    output_root: pathlib.Path,
    plan_loader: PlanLoader = recovery.validate_recovery_plan,
    compiler_identity: Mapping[str, str] | None = None,
    candidate_builder: Callable[[Mapping[str, Any]], maintained.Candidate]
    | None = None,
    original_plan_loader: PlanLoader = development.validate_plan,
    original_receipt_validator: OriginalReceiptValidator = (
        development.validate_run_receipt
    ),
) -> dict[str, Any]:
    """Fully rederive and validate a sealed recovery finalist by reference."""

    runner = DiscreteV3RecoveryRunner(
        plan_path=plan_path,
        output_root=output_root,
        plan_loader=plan_loader,
        compiler_identity=compiler_identity,
        candidate_builder=candidate_builder,
        original_plan_loader=original_plan_loader,
        original_receipt_validator=original_receipt_validator,
        read_only=True,
    )
    if (
        reference_path.is_symlink()
        or not reference_path.is_file()
        or reference_path.resolve() != runner.routes["finalist_reference"]
    ):
        raise RecoveryRunnerError("recovery finalist reference path changed")
    entries = _journal_entries(runner.routes["journal"])
    if (
        not entries
        or entries[-1]["event"] != "campaign-complete"
        or _event_details(entries[-1])
        != {
            "finalist_reference": _sealed_record(
                reference_path, FINALIST_REFERENCE_SCHEMA
            )
        }
    ):
        raise RecoveryRunnerError("recovery campaign completion is absent")
    original_rows, original_records = runner._prepare_contract()
    payload = runner._execute_algorithm(original_rows, launch_missing=False)
    value = runner._seal_result_and_finalist(
        payload=payload,
        original_records=original_records,
    )
    if value["reference"] != qualification.load_sealed(
        reference_path, FINALIST_REFERENCE_SCHEMA
    ):
        raise RecoveryRunnerError("recovery finalist reference changed")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    execute = commands.add_parser("execute")
    execute.add_argument("--plan", type=pathlib.Path, required=True)
    execute.add_argument("--output-root", type=pathlib.Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--plan", type=pathlib.Path, required=True)
    verify.add_argument("--output-root", type=pathlib.Path, required=True)
    verify.add_argument("--finalist-reference", type=pathlib.Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "execute":
            value = DiscreteV3RecoveryRunner(
                plan_path=arguments.plan,
                output_root=arguments.output_root,
            ).execute()
        else:
            value = validate_recovery_finalist(
                arguments.finalist_reference,
                plan_path=arguments.plan,
                output_root=arguments.output_root,
            )
        print(json.dumps({
            "status": value["finalist"]["status"],
            "result": value["reference"]["recovery_result"]["path"],
            "finalist": str(value["path"]),
            "finalist_sha256": _sha_file(value["path"]),
            "actual_clock": value["finalist"]["actual_clock"],
        }, sort_keys=True))
        return 0
    except (
        RecoveryRunnerError,
        OSError,
        ValueError,
        subprocess.SubprocessError,
    ) as error:
        print(f"discrete-v3 recovery runner failure: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
