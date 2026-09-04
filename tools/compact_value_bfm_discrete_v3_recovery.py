#!/usr/bin/env python3
"""Preregister and validate the one-shot discrete-v3 development recovery.

The original development attempt is terminal and immutable.  This module does
not resume it, reinterpret its partial result, or launch games.  It validates
the original plan, the ten completed pre-confirmation receipts, the complete
terminal journal, and the one completed confirmation output as forensic-only
evidence.  It then creates one sibling recovery plan with a deterministic fresh
tuple-confirmation bank.  The other five development banks remain byte-exact.

The fresh bank excludes every symmetry fingerprint in the seven historical
opening exclusions and all six original development banks.  The plan also
records the spent original confirmation bank as an additional exclusion for
the eventual protected final bank.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import statistics
import subprocess
import sys
from collections.abc import Mapping, Sequence
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
REPOSITORY = HERE.parent
TEST_PATH = REPOSITORY / "tests/codingame/test_compact_value_bfm_discrete_v3_recovery.py"
RUNNER_TEST_PATH = (
    REPOSITORY
    / "tests/codingame/test_compact_value_bfm_discrete_v3_recovery_runner.py"
)
BOT_ROOT = REPOSITORY / "submissions/codingame/bots/compact_value_bfm"
RECOVERY_RUNNER_PATH = BOT_ROOT / "discrete_v3_recovery_runner.py"


def _load(path: pathlib.Path, name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load discrete-v3 recovery dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


qualification = _load(
    HERE / "compact_value_bfm_qualification.py", "compact_v3_recovery_qualification"
)
development = _load(
    HERE / "compact_value_bfm_discrete_v3_development.py",
    "compact_v3_recovery_development",
)
openings = _load(
    HERE / "compact_value_bfm_openings.py", "compact_v3_recovery_openings"
)

RecoveryError = qualification.QualificationError

NAMESPACE = development.NAMESPACE
SOURCE_CAMPAIGN_ID = development.CAMPAIGN_ID
RECOVERY_ID = f"{SOURCE_CAMPAIGN_ID}-development-recovery-v1"
RECOVERY_ROOT_NAME = "development-recovery-v1"

PLAN_SCHEMA = "papersoccer.compact-value-bfm.discrete-v3-development-recovery-plan.v1"
INCIDENT_SCHEMA = (
    "papersoccer.compact-value-bfm.discrete-v3-development-terminal-incident.v1"
)
MIXED_EXCLUSION_SCHEMA = (
    "papersoccer.compact-value-bfm.discrete-v3-development-recovery-"
    "mixed-six-exclusion.v1"
)
RESULT_SCHEMA = (
    "papersoccer.compact-value-bfm.discrete-v3-development-recovery-result.v1"
)
FINALIST_SCHEMA = (
    "papersoccer.compact-value-bfm.discrete-v3-post-holdout-recovery-finalist.v1"
)
FINALIST_REFERENCE_SCHEMA = (
    "papersoccer.compact-value-bfm.discrete-v3-post-holdout-recovery-"
    "finalist-reference.v1"
)

ORIGINAL_PLAN_SHA256 = (
    "650de20809e1d89ffdf0e74d6fe4136023d09c9a951fd1b6222f0948c896f820"
)
ORIGINAL_PLAN_BODY_SHA256 = (
    "7dadbf8fee6f57da56d856056cae29b1b09c6833728b7042767b3adeb807ea7a"
)
HISTORICAL_SUPERVISOR_SHA256 = (
    "61d04f05968c67d17aef26a1387ba5c8dcbb043bef0d64d6ad768cb3311a2a2e"
)
HISTORICAL_SUPERVISOR_BYTES = 168_236
JOURNAL_HEAD_SHA256 = (
    "7da8a3f454f8a0ebb1181b966697ae6a757f6eaa4aa02fb904a7781c7c415600"
)
JOURNAL_ROSTER_SHA256 = (
    "51194806f54ca44e9585a59943da03350d7054f8912d0feb00d5bc7048fbd859"
)
FORENSIC_RESULT_SHA256 = (
    "d275b9a44c218479d43df9d40905a66db76bf5c3f425364c62da7a3b64d6f41c"
)
FORENSIC_RESULT_OBJECT_SHA256 = (
    "d8fe6f03b6448211adb06c7b604cd7b5d0a87bcd10c026214ce84ec8747456db"
)
ORIGINAL_CONFIRMATION_GATE_SHA256 = (
    "7e3143dbad826382b15894a900543e3df63479296405cc2bb6826460ff4893e1"
)
ORIGINAL_CONFIRMATION_MANIFEST_SHA256 = (
    "be2f65786a7f913204ea75947ad75243c43781ee8357a9edb9f65fafb18c96ba"
)
CANDIDATE_BINARY_SHA256 = (
    "d4068d3c28653107bdce3e6f9f1550697aa8783a71f1ac21f73ff764c75d6237"
)
CONTROL_BINARY_SHA256 = (
    "e6d72858696a1501b03c3b60205cf2665c9ab3f8bf03f6449cf13afaacb0085e"
)

RECOVERY_BANK_DOMAIN = (
    "compact-value-bfm-discrete-v3-development-recovery-v1/"
    "tuple-confirmation/fresh-bank-v1"
)
RECOVERY_BANK_SEED = hashlib.sha256(
    (RECOVERY_ID + "\0" + RECOVERY_BANK_DOMAIN + "\0" + ORIGINAL_PLAN_SHA256).encode(
        "ascii"
    )
).digest()

STAGE_ORDER = tuple(development.STAGE_ORDER)
STAGE_PAIRS = dict(development.STAGE_PAIRS)
EXPECTED_JOURNAL_EVENTS = (
    "policy-committed",
    "anchor-freeze-intent",
    "anchor-frozen",
    "fanout-intent",
    "sibling-launch-intent",
    "sibling-started",
    "sibling-launch-intent",
    "sibling-started",
    "fanout-committed",
    "terminal-failure",
)
EXPECTED_CARRIED_REQUESTS = (
    "c881c7b8f999c61a7b5228c7c34220e74dc109648462fbdd903f5d3d42094adf",
    "3cd7911819d157628e2ae3fc10fe154d80ccc05b71999532ed71dbe889c5e084",
    "da9fbe226fc5ed28c6c94f2c967cc2e3c10ab40b5d64e8ad9b85425ef0cce613",
    "8eadc1354c1713ce0c1c5574083ed1029d6ad06b546861ef8cde38aad2c2f840",
    "54228bad8ce5eed3008be90e9ec679566a3c118061a68fdd039286ebb725bf9b",
    "6191fc5de8b9f192b2b7464521786ee337e7699d1db4a45198e1491644f18536",
    "4d3c45782f1373ef3529d8c63a266fad7dcd82a82ea5f580898f174dcf00f033",
    "77f2de0abfbf4cb49d8efc5db709361b3df21ee719e82f6f1dae6dae633cb330",
    "9c897c5dd1e42f01b828d677ab5874030b427a088224a144b4f2291476c5dcef",
    "3739eae2576d6d21d9e1e021d92161ea00141038271327cff55774a34bb3a27c",
)
EXPECTED_CARRIED_KEYS = (
    ("model_screen", development.CANDIDATE_ID, ("0.95", "0.5", "1")),
    ("model_screen", development.CONTROL_ID, ("0.95", "0.5", "1")),
    ("tuple_screen", development.CANDIDATE_ID, ("0.65", "0.5", "1")),
    ("tuple_screen", development.CANDIDATE_ID, ("0.80", "0.5", "1")),
    ("tuple_screen", development.CANDIDATE_ID, ("0.95", "0.5", "1")),
    ("tuple_screen", development.CANDIDATE_ID, ("1.10", "0.5", "1")),
    ("tuple_screen", development.CANDIDATE_ID, ("0.95", "0.25", "1")),
    ("tuple_screen", development.CANDIDATE_ID, ("0.95", "0.75", "1")),
    ("tuple_screen", development.CANDIDATE_ID, ("0.95", "0.5", "0.5")),
    ("tuple_screen", development.CANDIDATE_ID, ("0.95", "0.5", "0")),
)
EXPECTED_CARRIED_WINS = (101, 87, 112, 103, 101, 98, 81, 101, 100, 102)

THREAD_ENVIRONMENT = {
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "VECLIB_MAXIMUM_THREADS": "1",
}
CONCURRENCY = {
    "policy": "fresh-bank-full-roster-concurrent-no-retry",
    "tuple_confirmation": {
        "jobs": 3,
        "maximum_concurrent_jobs": 3,
        "process_nice": 0,
        "all_jobs_equal_nice": True,
        "full_roster_required": True,
        "no_retry_after_claim": True,
        "latency_comparison": "same-stage-concurrent-only",
    },
    "profile_screen": {
        "jobs": 3,
        "maximum_concurrent_jobs": 3,
        "process_nice": 0,
        "all_jobs_equal_nice": True,
        "full_roster_required": True,
        "no_retry_after_claim": True,
        "latency_comparison": "same-stage-concurrent-only",
    },
    "profile_confirmation": {
        "maximum_concurrent_jobs": 3,
        "process_nice": 0,
        "all_jobs_equal_nice": True,
        "full_roster_required": True,
        "no_retry_after_claim": True,
        "latency_comparison": "same-stage-concurrent-only",
    },
    "actual_clock": {
        "jobs": 1,
        "maximum_concurrent_jobs": 1,
        "process_nice": 0,
        "no_retry_after_claim": True,
        "latency_comparison": "serial-actual-clock",
    },
    "thread_environment": THREAD_ENVIRONMENT,
}

_PRIVATE_FINGERPRINT_CACHE: tuple[tuple[str, str, str], frozenset[str]] | None = None

PLAN_FIELDS = {
    "schema",
    "namespace",
    "campaign_id",
    "source_campaign_id",
    "recovery_id",
    "status",
    "created_at_utc",
    "attempt",
    "original",
    "candidate",
    "rank4_control",
    "banks",
    "additional_development_exclusions",
    "mixed_six_exclusion",
    "binaries",
    "compile_references",
    "algorithm",
    "recovery_contract",
    "compiler",
    "tools",
    "concurrency",
    "outputs",
    "policy",
    "body_sha256",
}
OUTPUT_FIELDS = {
    "recovery_root",
    "plan",
    "incident",
    "mixed_six_exclusion",
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
    "result",
    "finalists",
    "finalist_reference",
}


def _utc(value: Any, label: str) -> str:
    qualification._utc(value, label)
    return str(value)


def _regular(path: pathlib.Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RecoveryError(f"required regular artifact is absent or redirected: {path}")
    raw = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _verify_record(value: Any, label: str) -> pathlib.Path:
    if not isinstance(value, Mapping) or set(value) != {"path", "bytes", "sha256"}:
        raise RecoveryError(f"{label} file record is malformed")
    path = pathlib.Path(str(value.get("path", "")))
    if _regular(path) != dict(value):
        raise RecoveryError(f"{label} bytes changed")
    return path.resolve()


def _verify_original_record(value: Any, label: str) -> pathlib.Path:
    if not isinstance(value, Mapping):
        raise RecoveryError(f"{label} record is malformed")
    if set(value) == {"path", "sha256"}:
        path = pathlib.Path(str(value["path"]))
        record = _regular(path)
        if record["sha256"] != value["sha256"]:
            raise RecoveryError(f"{label} bytes changed")
        return path.resolve()
    if "body_sha256" in value and "schema" in value:
        return _verify_sealed_record(value, str(value["schema"]), label)
    return _verify_record(value, label)


def _sealed_record(path: pathlib.Path, schema: str) -> dict[str, Any]:
    artifact = qualification.load_sealed(path, schema)
    return {**_regular(path), "schema": schema, "body_sha256": artifact["body_sha256"]}


def _verify_sealed_record(value: Any, schema: str, label: str) -> pathlib.Path:
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "bytes",
        "sha256",
        "schema",
        "body_sha256",
    } or value.get("schema") != schema:
        raise RecoveryError(f"{label} sealed record is malformed")
    path = pathlib.Path(str(value.get("path", "")))
    if _sealed_record(path, schema) != dict(value):
        raise RecoveryError(f"{label} sealed bytes changed")
    return path.resolve()


def _campaign_root(output_root: pathlib.Path) -> pathlib.Path:
    root = output_root.absolute()
    if root != root.resolve() or root.is_symlink() or not root.is_dir():
        raise RecoveryError("campaign output root is absent or redirected")
    return root.resolve()


def _paths(output_root: pathlib.Path) -> dict[str, str]:
    root = _campaign_root(output_root) / RECOVERY_ROOT_NAME
    return {
        "recovery_root": str(root),
        "plan": str(root / "plan.json"),
        "incident": str(root / "terminal-incident.json"),
        "mixed_six_exclusion": str(root / "mixed-six-exclusion.json"),
        "opening_banks": str(root / "opening-banks"),
        "gate_banks": str(root / "gate-banks"),
        "binaries": str(root / "gate-binaries"),
        "scratch": str(root / "scratch"),
        "requests": str(root / "requests"),
        "base_receipts": str(root / "receipts"),
        "receipts": str(root / "receipts-recovery-v1"),
        "references": str(root / "run-references-recovery-v1"),
        "claims": str(root / "claims"),
        "journal": str(root / "events"),
        "result": str(root / "development-result.json"),
        "finalists": str(root / "finalists"),
        "finalist_reference": str(root / "finalist-reference.json"),
    }


def validate_no_original_campaign_processes(
    output_root: pathlib.Path,
) -> dict[str, Any]:
    """Fail if an old runner, supervisor, or old-bank gate is still live."""

    campaign = _campaign_root(output_root)
    original_plan = campaign / "development-v3/plan.json"
    historical_supervisor = (
        campaign
        / "development-adaptive-stage-barrier/adaptive_stage_barrier_supervisor.py"
    )
    old_gate_bank = (
        campaign
        / "development-v3/gate-banks"
        / f"{ORIGINAL_CONFIRMATION_GATE_SHA256}.tsv"
    )
    rules = [
        {
            "role": "original-development-runner",
            "required_substrings": [
                "discrete_v3_development_runner.py",
                str(original_plan),
            ],
        },
        {
            "role": "historical-adaptive-supervisor",
            "required_substrings": [str(historical_supervisor)],
        },
        {
            "role": "original-confirmation-gate",
            "required_substrings": ["--bank", str(old_gate_bank)],
        },
    ]
    completed = subprocess.run(
        ["/bin/ps", "-axo", "pid=,command="],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RecoveryError("cannot inspect original campaign processes")
    matches = []
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        first, _, command = stripped.partition(" ")
        try:
            pid = int(first)
        except ValueError:
            continue
        if pid == os.getpid():
            continue
        for rule in rules:
            if all(fragment in command for fragment in rule["required_substrings"]):
                matches.append({"pid": pid, "role": rule["role"], "command": command})
    if matches:
        roles = ", ".join(sorted({item["role"] for item in matches}))
        raise RecoveryError(f"original campaign process remains live: {roles}")
    return {
        "checked": True,
        "forbidden_match_count": 0,
        "rules": rules,
        "required_before_recovery_launch": True,
    }


def _original_plan(output_root: pathlib.Path) -> tuple[pathlib.Path, dict[str, Any]]:
    output_root = _campaign_root(output_root)
    path = output_root / "development-v3/plan.json"
    if (
        _regular(path)["sha256"] != ORIGINAL_PLAN_SHA256
        or qualification.load_sealed(path, development.PLAN_SCHEMA).get("body_sha256")
        != ORIGINAL_PLAN_BODY_SHA256
    ):
        raise RecoveryError("original development plan identity changed")
    plan = qualification.load_sealed(path, development.PLAN_SCHEMA)
    if (
        set(plan) != development.PLAN_FIELDS
        or plan.get("namespace") != NAMESPACE
        or plan.get("campaign_id") != SOURCE_CAMPAIGN_ID
        or plan.get("status") != "development-planned-awaiting-games"
        or plan.get("outputs", {}).get("development_root")
        != str((output_root / "development-v3").resolve())
        or plan.get("compiler")
        != dict(development.maintained._default_compiler_identity())
    ):
        raise RecoveryError("original development plan contract changed")
    for section, names in (
        (plan.get("candidate"), ("selection", "runtime", "generated_source")),
        (plan.get("rank4_control"), ("selection", "runtime")),
    ):
        if not isinstance(section, Mapping):
            raise RecoveryError("original candidate/control binding is malformed")
        for name in names:
            record = section.get(name)
            if not isinstance(record, Mapping):
                raise RecoveryError(f"original {name} binding is malformed")
            _verify_original_record(record, f"original {name}")
    tools = plan.get("tools")
    if not isinstance(tools, Mapping):
        raise RecoveryError("original development tool closure is malformed")
    for name, record in tools.items():
        _verify_record(record, f"original {name}")
    deeply_validated = dict(development.validate_plan(path, output_root=output_root))
    if deeply_validated != plan:
        raise RecoveryError("deep validation changed the original development plan")
    return path, dict(plan)


def _carried_receipts(
    output_root: pathlib.Path, original_plan: Mapping[str, Any]
) -> list[dict[str, Any]]:
    root = _campaign_root(output_root) / "development-v3/run-references-v3"
    expected_names = {f"{request}.json" for request in EXPECTED_CARRIED_REQUESTS}
    actual_names = {path.name for path in root.iterdir()}
    if actual_names != expected_names:
        raise RecoveryError("original carried receipt-reference roster changed")
    carried = []
    for order, (request_sha, expected_key, expected_wins) in enumerate(
        zip(EXPECTED_CARRIED_REQUESTS, EXPECTED_CARRIED_KEYS, EXPECTED_CARRIED_WINS),
        start=1,
    ):
        path = root / f"{request_sha}.json"
        reference = qualification.load_sealed(
            path, development.RECEIPT_REFERENCE_SCHEMA
        )
        if set(reference) != {
            "schema",
            "namespace",
            "campaign_id",
            "development_plan",
            "request",
            "receipt",
            "complete",
            "body_sha256",
        } or reference.get("complete") is not True:
            raise RecoveryError("carried receipt reference shape changed")
        if reference.get("development_plan") != _sealed_record(
            pathlib.Path(original_plan["outputs"]["development_root"]) / "plan.json",
            development.PLAN_SCHEMA,
        ):
            raise RecoveryError("carried receipt escaped the original plan")
        request_path = _verify_sealed_record(
            reference.get("request"), development.REQUEST_SCHEMA, "carried request"
        )
        if request_path.name != f"{request_sha}.request.json":
            raise RecoveryError("carried request identity changed")
        validated = development.validate_run_receipt(
            reference.get("receipt"), original_plan
        )
        request = validated["request"]
        metric = validated["metric"]
        key = (
            request["spec"]["stage"],
            request["candidate"]["candidate_id"],
            tuple(request["spec"]["tuple"]),
        )
        if key != expected_key or metric.get("wins") != expected_wins:
            raise RecoveryError("carried receipt ordering/content changed")
        carried.append(
            {
                "order": order,
                "stage": key[0],
                "candidate_id": key[1],
                "tuple": list(key[2]),
                "reference": _sealed_record(
                    path, development.RECEIPT_REFERENCE_SCHEMA
                ),
                "validated": validated,
            }
        )
    if [item["stage"] for item in carried].count("model_screen") != 2 or [
        item["stage"] for item in carried
    ].count("tuple_screen") != 8:
        raise RecoveryError("carried receipts are not exactly two model plus eight tuple screens")
    return carried


def _journal(output_root: pathlib.Path) -> dict[str, Any]:
    campaign = _campaign_root(output_root)
    supervisor = (
        campaign
        / "development-adaptive-stage-barrier/adaptive_stage_barrier_supervisor.py"
    )
    supervisor_record = _regular(supervisor)
    if supervisor_record != {
        "path": str(supervisor.resolve()),
        "bytes": HISTORICAL_SUPERVISOR_BYTES,
        "sha256": HISTORICAL_SUPERVISOR_SHA256,
    }:
        raise RecoveryError("historical terminal supervisor identity changed")
    directory = (
        campaign / "development-adaptive-stage-barrier/tuple_confirmation/events"
    )
    paths = sorted(directory.glob("*.json"))
    if (
        len(paths) != 10
        or {path.name for path in directory.iterdir()} != {path.name for path in paths}
        or any(path.is_symlink() or not path.is_file() for path in paths)
    ):
        raise RecoveryError("terminal incident journal roster changed")
    previous = "0" * 64
    entries = []
    records = []
    for sequence, path in enumerate(paths, start=1):
        value = json.loads(path.read_bytes())
        if qualification.canonical_json_bytes(value) != path.read_bytes():
            raise RecoveryError("terminal incident journal entry is not canonical")
        body = dict(value)
        claimed = body.pop("entry_sha256", None)
        expected = qualification.sha256_bytes(qualification.canonical_json_bytes(body))
        if (
            claimed != expected
            or value.get("sequence") != sequence
            or value.get("previous_sha256") != previous
            or value.get("plan_sha256") != ORIGINAL_PLAN_SHA256
            or value.get("roster_sha256") != JOURNAL_ROSTER_SHA256
            or value.get("mode") != "tuple_confirmation"
            or path.name != f"{sequence:06d}-{claimed}.json"
        ):
            raise RecoveryError("terminal incident journal chain changed")
        previous = str(claimed)
        entries.append(value)
        records.append(_regular(path))
    if (
        tuple(entry.get("event") for entry in entries) != EXPECTED_JOURNAL_EVENTS
        or previous != JOURNAL_HEAD_SHA256
        or entries[-1].get("no_retry") is not True
        or entries[-1].get("reason")
        != "discrete-v3-search-target:c0.80-f0.5-l1 exited without a valid result; replay forbidden"
        or any(entry.get("event") == "result-recorded" for entry in entries)
    ):
        raise RecoveryError("terminal/no-retry incident semantics changed")
    return {
        "supervisor": supervisor_record,
        "directory": str(directory.resolve()),
        "entry_count": 10,
        "ordered_entries": records,
        "head_sha256": previous,
        "events": list(EXPECTED_JOURNAL_EVENTS),
        "terminal": dict(entries[-1]),
    }


def _expected_confirmation_config() -> dict[str, Any]:
    return {
        "mode": "fixed-work",
        "pair_offset": 0,
        "pair_count": 250,
        "candidate_c": 0.8,
        "candidate_fpu": 0.5,
        "candidate_lambda": 1.0,
        "candidate_actions": 250,
        "candidate_root_partial_paths": 4000,
        "candidate_nonroot_partial_paths": 512,
        "candidate_nodes": 80000,
        "candidate_expansions": 2_000_000,
        "candidate_shuffle_seed": 1,
        "candidate_clocks_ms": [800, 155],
        "rank4_nodes": 3_000_000,
        "rank4_clocks_ms": [800, 165],
        "max_turns": 320,
        "minimum_candidate_wins": -1,
        "minimum_wins_per_color": -1,
    }


def _absence_paths(output_root: pathlib.Path) -> list[dict[str, str]]:
    campaign = _campaign_root(output_root)
    development_root = campaign / "development-v3"
    precompute = campaign / "development-adaptive-stage-barrier/tuple_confirmation/precompute"
    return [
        {
            "candidate_id": "discrete-v3-search-target:c0.65-f0.5-l1",
            "role": "anchor-gate-output",
            "path": str(development_root / "scratch/29e9e9897b5cc19c312610c3cfee7e7c36083cb5aa09ea82ee5f72ab99c8101b.gate.json"),
        },
        {
            "candidate_id": "discrete-v3-search-target:c0.65-f0.5-l1",
            "role": "anchor-maintained-reference",
            "path": str(development_root / "run-references/29e9e9897b5cc19c312610c3cfee7e7c36083cb5aa09ea82ee5f72ab99c8101b.json"),
        },
        {
            "candidate_id": "discrete-v3-search-target:c0.65-f0.5-l1",
            "role": "anchor-v3-reference",
            "path": str(development_root / "run-references-v3/1a661728c128ad434f213e4b75256a03ff46dfe21f6e60e76303da726198b2fe.json"),
        },
        {
            "candidate_id": "discrete-v3-search-target:c0.95-f0.5-l1",
            "role": "default-gate-output",
            "path": str(development_root / "scratch/0c400bed0b084c921aaef31bcb5d5db9ae53171b0891526f649dc55310b7d906.gate.json"),
        },
        {
            "candidate_id": "discrete-v3-search-target:c0.95-f0.5-l1",
            "role": "default-precompute-output",
            "path": str(precompute / ".0c400bed0b084c921aaef31bcb5d5db9ae53171b0891526f649dc55310b7d906.fc3f91d4659784896506217e.gate.json.partial"),
        },
        {
            "candidate_id": "discrete-v3-search-target:c0.95-f0.5-l1",
            "role": "default-maintained-reference",
            "path": str(development_root / "run-references/0c400bed0b084c921aaef31bcb5d5db9ae53171b0891526f649dc55310b7d906.json"),
        },
    ]


def _forensic_result(
    output_root: pathlib.Path,
    original_plan: Mapping[str, Any],
    journal: Mapping[str, Any],
) -> dict[str, Any]:
    campaign = _campaign_root(output_root)
    precompute = campaign / "development-adaptive-stage-barrier/tuple_confirmation/precompute"
    raw_path = precompute / ".1abca1c2fc9649de29df9db0965d7a57eefcf5edcc356d304990f7cb37921ccf.d2cc3d28f3eacbb89f64ca7a.gate.json.partial"
    stdout_path = precompute / "1abca1c2fc9649de29df9db0965d7a57eefcf5edcc356d304990f7cb37921ccf.stdout.log"
    raw_record = _regular(raw_path)
    stdout_record = _regular(stdout_path)
    if (
        raw_record["sha256"] != FORENSIC_RESULT_SHA256
        or stdout_record["sha256"] != FORENSIC_RESULT_SHA256
        or raw_path.read_bytes() != stdout_path.read_bytes()
    ):
        raise RecoveryError("c0.80 forensic result bytes changed")
    gate = development.maintained.gate_support.validate_result(
        raw_path,
        expected_bank_sha256=ORIGINAL_CONFIRMATION_GATE_SHA256,
        expected_candidate_sha256=original_plan["candidate"]["generated_source"]["sha256"],
        allow_legacy_attempt_zero=True,
    )
    if qualification.sha256_bytes(qualification.canonical_json_bytes(gate)) != FORENSIC_RESULT_OBJECT_SHA256:
        raise RecoveryError("c0.80 forensic result object identity changed")
    result = gate.get("result")
    games = gate.get("games")
    candidate = result.get("candidate", {}) if isinstance(result, Mapping) else {}
    times = candidate.get("times_ms") if isinstance(candidate, Mapping) else None
    if (
        gate.get("config") != _expected_confirmation_config()
        or gate.get("bindings", {}).get("candidate_source_sha256")
        != original_plan["candidate"]["generated_source"]["sha256"]
        or gate.get("bindings", {}).get("candidate_runtime_body_sha256")
        != original_plan["candidate"]["runtime_identity"]["body_sha256"]
        or gate.get("bindings", {}).get("candidate_payload_sha256")
        != original_plan["candidate"]["runtime_identity"]["payload_sha256"]
        or gate.get("bindings", {}).get("rank4_source_sha256")
        != original_plan["tools"]["rank4_source"]["sha256"]
        or gate.get("bindings", {}).get("bank_sha256")
        != ORIGINAL_CONFIRMATION_GATE_SHA256
        or not isinstance(games, list)
        or len(games) != 500
        or not isinstance(result, Mapping)
        or result.get("games") != 500
        or result.get("candidate_wins") != 258
        or result.get("candidate_wins_player0") != 130
        or result.get("candidate_wins_player1") != 128
        or result.get("rank4_wins") != 242
        or result.get("failures") != 0
        or result.get("unfinished") != 0
        or not isinstance(times, list)
        or not times
        or not math.isclose(statistics.fmean(times), 258.17868222367673, rel_tol=0, abs_tol=1e-12)
    ):
        raise RecoveryError("c0.80 forensic result semantics changed")
    supervisor_source = pathlib.Path(journal["supervisor"]["path"]).read_text(
        encoding="utf-8"
    )
    required_by_bug = ('result.get("completed_games")', 'result.get("unfinished_games")')
    if any(fragment not in supervisor_source for fragment in required_by_bug):
        raise RecoveryError("historical supervisor schema defect changed")
    authoritative_result_fields = sorted(result)
    if (
        "games" not in authoritative_result_fields
        or "unfinished" not in authoritative_result_fields
        or "completed_games" in authoritative_result_fields
        or "unfinished_games" in authoritative_result_fields
    ):
        raise RecoveryError("authoritative gate result schema changed")
    result_completed_unix_ns = raw_path.stat().st_mtime_ns
    terminal_unix_ns = int(journal["terminal"]["time_unix_ns"])
    terminal_delay_seconds = (terminal_unix_ns - result_completed_unix_ns) / 1e9
    if not math.isclose(terminal_delay_seconds, 0.541546158, rel_tol=0, abs_tol=1e-9):
        raise RecoveryError("terminal incident timing changed")
    absences = _absence_paths(output_root)
    for evidence in absences:
        if os.path.lexists(evidence["path"]):
            raise RecoveryError(f"anchor/default output unexpectedly exists: {evidence['path']}")
    return {
        "candidate_id": "discrete-v3-search-target:c0.80-f0.5-l1",
        "stage": "tuple_confirmation",
        "tuple": ["0.80", "0.5", "1"],
        "raw_result": raw_record,
        "identical_stdout_capture": stdout_record,
        "canonical_result_sha256": FORENSIC_RESULT_OBJECT_SHA256,
        "summary": {
            "pairs": 250,
            "games": 500,
            "candidate_wins": 258,
            "candidate_wins_player0": 130,
            "candidate_wins_player1": 128,
            "rank4_wins": 242,
            "failures": 0,
            "unfinished": 0,
            "candidate_mean_latency_ms": statistics.fmean(times),
        },
        "selection_weight": 0,
        "eligible_for_selection": False,
        "reason": "terminal-original-attempt-forensic-only-no-cross-bank-combination",
        "root_cause": {
            "classification": "historical-supervisor-result-schema-defect",
            "validator_required_nonexistent_fields": [
                "result.completed_games",
                "result.unfinished_games",
            ],
            "authoritative_gate_fields": ["result.games", "result.unfinished"],
            "authoritative_result_field_roster": authoritative_result_fields,
            "result_completed_unix_ns": result_completed_unix_ns,
            "terminal_event_unix_ns": terminal_unix_ns,
            "terminal_delay_seconds": terminal_delay_seconds,
            "configured_quiet_seconds": 5,
            "failed_before_quiet_interval": terminal_delay_seconds < 5,
            "gameplay_failure": False,
        },
        "anchor_and_default_absent": absences,
    }


def validate_original_state(output_root: pathlib.Path) -> dict[str, Any]:
    plan_path, plan = _original_plan(output_root)
    carried = _carried_receipts(output_root, plan)
    journal = _journal(output_root)
    forensic = _forensic_result(output_root, plan, journal)
    process_absence = validate_no_original_campaign_processes(output_root)
    return {
        "plan_path": plan_path,
        "plan": plan,
        "carried": carried,
        "journal": journal,
        "forensic": forensic,
        "process_absence": process_absence,
    }


def _copied_exclusion_paths(output_root: pathlib.Path) -> list[pathlib.Path]:
    campaign = _campaign_root(output_root)
    source_root = campaign.parent / "compact-value-bfm-20260831-v1/input-bundle/opening-exclusions"
    result = [source_root / f"bank-{index:03d}.tsv" for index in range(7)]
    # load_all_exclusions validates the historical TSV contracts and exact row states.
    openings.load_all_exclusions(result)
    return result


def _all_variants(bank: Mapping[str, Any]) -> set[str]:
    return {
        str(fingerprint)
        for opening in bank["openings"]
        for name, fingerprint in opening["fingerprints"].items()
        if name != "canonical"
    }


def _exclusion_context(
    output_root: pathlib.Path, original_plan: Mapping[str, Any]
) -> dict[str, Any]:
    global _PRIVATE_FINGERPRINT_CACHE
    copied_paths = _copied_exclusion_paths(output_root)
    copied = openings.load_all_exclusions(copied_paths)
    seen = set(copied["fingerprints"])
    original_records = []
    original_documents = {}
    for stage in STAGE_ORDER:
        path = pathlib.Path(original_plan["banks"][stage]["path"])
        if _regular(path) != original_plan["banks"][stage]:
            raise RecoveryError(f"original {stage} bank bytes changed")
        bank = openings.validate_bank(path)
        if (
            bank.get("stage") != stage
            or bank.get("classification") != "unprotected-development"
            or bank.get("opening_count") != STAGE_PAIRS[stage]
        ):
            raise RecoveryError(f"original {stage} bank contract changed")
        variants = _all_variants(bank)
        if variants & seen:
            raise RecoveryError("original development banks overlap historical exclusions")
        seen.update(variants)
        original_documents[stage] = bank
        original_records.append(
            {
                "stage": stage,
                "bank": _regular(path),
                "opening_count": STAGE_PAIRS[stage],
            }
        )
    expected_copied_sources = original_documents[STAGE_ORDER[0]].get(
        "exclusion_sources"
    )
    if (
        copied.get("sources") != expected_copied_sources
        or any(
            document.get("exclusion_sources") != expected_copied_sources
            for document in original_documents.values()
        )
    ):
        raise RecoveryError("seven copied historical exclusion identities changed")
    protected_path = pathlib.Path(
        original_plan["exclusion"]["protected_fingerprints"]["path"]
    )
    protected_record = _sealed_record(
        protected_path, development.exclusions.FINGERPRINT_SCHEMA
    )
    if (
        protected_record["sha256"]
        != original_plan["exclusion"]["protected_fingerprints"]["sha256"]
    ):
        raise RecoveryError("protected fingerprint payload identity changed")
    exclusion_plan_path = pathlib.Path(original_plan["exclusion"]["plan"]["path"])
    exclusion_receipt_path = pathlib.Path(
        original_plan["exclusion"]["receipt"]["path"]
    )
    cache_key = (
        protected_record["sha256"],
        original_plan["exclusion"]["plan"]["sha256"],
        original_plan["exclusion"]["receipt"]["sha256"],
    )
    if _PRIVATE_FINGERPRINT_CACHE is None or _PRIVATE_FINGERPRINT_CACHE[0] != cache_key:
        private_values = development.exclusions._load_private_canonical_fingerprints(
            exclusion_receipt_path,
            plan_path=exclusion_plan_path,
            output_root=_campaign_root(output_root),
        )
        _PRIVATE_FINGERPRINT_CACHE = (cache_key, frozenset(private_values))
    protected = set(_PRIVATE_FINGERPRINT_CACHE[1])
    if len(protected) != 54_611 or protected & seen:
        raise RecoveryError(
            "protected fingerprints are incomplete or overlap existing development inputs"
        )
    seen.update(protected)
    protected_source = {
        **protected_record,
        "kind": "fresh-protected-canonical-fingerprints",
        "fingerprint_count": len(protected),
        "fingerprints_sha256": qualification.sha256_bytes(
            qualification.canonical_json_bytes(sorted(protected))
        ),
        "private_values_serialized": False,
    }
    sources = [
        {**_regular(path), "kind": "copied-historical-exclusion"}
        for path in copied_paths
    ] + [
        {**entry["bank"], "kind": "original-development-bank", "stage": entry["stage"]}
        for entry in original_records
    ] + [protected_source]
    return {
        "copied_paths": copied_paths,
        "copied_sources": sources[:7],
        "original_records": original_records,
        "original_documents": original_documents,
        "protected_source": protected_source,
        "protected_fingerprints": frozenset(protected),
        "sources": sources,
        "fingerprints": seen,
        "body_sha256": qualification.sha256_bytes(
            qualification.canonical_json_bytes(
                {"sources": sources, "fingerprints": sorted(seen)}
            )
        ),
    }


def _gate_bank_bytes(bank: Mapping[str, Any]) -> bytes:
    return (
        "# papersoccer.compact-value-bfm-opening-bank.v1\n"
        "opening_id\ttranscript\n"
        + "".join(
            f"{row['opening_id']}\t{row['transcript']}\n" for row in bank["openings"]
        )
    ).encode("ascii")


def _fresh_bank_payload(
    *,
    original_plan: Mapping[str, Any],
    context: Mapping[str, Any],
    recovery_plan_path: pathlib.Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    plan_record = _sealed_record(recovery_plan_path, PLAN_SCHEMA)
    fresh = openings.generate_openings(
        stage="tuple_confirmation",
        count=250,
        seed=RECOVERY_BANK_SEED,
        excluded_fingerprints=set(context["fingerprints"]),
    )
    payload = openings.bank_payload(
        stage="tuple_confirmation",
        classification="unprotected-development",
        seed=RECOVERY_BANK_SEED,
        exclusions={
            "sources": context["sources"],
            "body_sha256": context["body_sha256"],
        },
        openings=fresh,
        source_binding={
            "schema": "papersoccer.compact-value-bfm.discrete-v3-development-recovery-bank-binding.v1",
            "namespace": NAMESPACE,
            "source_campaign_id": SOURCE_CAMPAIGN_ID,
            "recovery_id": RECOVERY_ID,
            "recovery_plan": plan_record,
            "original_plan_sha256": ORIGINAL_PLAN_SHA256,
            "domain": RECOVERY_BANK_DOMAIN,
            "attempt": 1,
        },
    )
    return payload, fresh


def create_fresh_confirmation_bank(
    output_root: pathlib.Path,
    original_plan: Mapping[str, Any],
    recovery_plan_path: pathlib.Path,
    bank_directory: pathlib.Path,
    gate_directory: pathlib.Path,
) -> dict[str, Any]:
    context = _exclusion_context(output_root, original_plan)
    if (
        os.path.lexists(bank_directory)
        or os.path.lexists(gate_directory)
    ):
        raise RecoveryError("fresh-bank materialization output already exists")
    payload, fresh = _fresh_bank_payload(
        original_plan=original_plan,
        context=context,
        recovery_plan_path=recovery_plan_path,
    )
    fresh_variants = {
        str(fingerprint)
        for opening in fresh
        for name, fingerprint in opening["fingerprints"].items()
        if name != "canonical"
    }
    if len(fresh) != 250 or fresh_variants & context["fingerprints"]:
        raise RecoveryError("fresh recovery confirmation bank is not symmetry-disjoint")
    path = openings.write_bank(
        bank_directory,
        payload,
    )
    bank = openings.validate_bank(path)
    if (
        bank.get("seed_hex") != RECOVERY_BANK_SEED.hex()
        or bank.get("opening_count") != 250
        or bank.get("exclusions_body_sha256") != context["body_sha256"]
        or _all_variants(bank) & context["fingerprints"]
    ):
        raise RecoveryError("fresh recovery bank validation failed")
    gate_raw = _gate_bank_bytes(bank)
    gate_sha = qualification.sha256_bytes(gate_raw)
    gate_path = gate_directory / f"{gate_sha}.tsv"
    qualification.atomic_write_once(gate_path, gate_raw)
    development.maintained.gate_support.validate_bank(gate_path)
    bank_files = list(bank_directory.glob("*.opening-bank.json"))
    gate_files = list(gate_directory.glob("*.tsv"))
    if bank_files != [path] or gate_files != [gate_path]:
        raise RecoveryError("recovery must contain exactly one fresh confirmation bank")
    return {
        "manifest": _regular(path),
        "gate": _regular(gate_path),
        "document": bank,
        "exclusion_context": context,
    }


def _mixed_bank_context(
    original_plan: Mapping[str, Any], fresh: Mapping[str, Any]
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], set[str]]:
    banks: dict[str, dict[str, Any]] = {}
    rows = []
    seen: set[str] = set()
    for stage in STAGE_ORDER:
        if stage == "tuple_confirmation":
            record = dict(fresh["manifest"])
            document = fresh["document"]
        else:
            record = dict(original_plan["banks"][stage])
            document = openings.validate_bank(pathlib.Path(record["path"]))
        variants = _all_variants(document)
        if variants & seen:
            raise RecoveryError("mixed six-bank recovery roster overlaps by symmetry")
        seen.update(variants)
        banks[stage] = record
        rows.append(
            {"stage": stage, "bank": record, "opening_count": STAGE_PAIRS[stage]}
        )
    return banks, rows, seen


def _additional_exclusions(original_plan: Mapping[str, Any]) -> dict[str, Any]:
    spent = dict(original_plan["banks"]["tuple_confirmation"])
    if spent.get("sha256") != ORIGINAL_CONFIRMATION_MANIFEST_SHA256:
        raise RecoveryError("spent original confirmation bank identity changed")
    return {
        "spent_original_tuple_confirmation": {
            "stage": "tuple_confirmation",
            "bank": spent,
            "opening_count": 250,
            "selection_weight": 0,
            "eligible_for_selection": False,
            "required_for_eventual_protected_final": True,
        },
        "eventual_protected_final_requires_union": True,
    }


def _mixed_exclusion_body(
    *,
    plan_created_at_utc: str,
    materialized_at_utc: str,
    original_plan_path: pathlib.Path,
    recovery_plan_path: pathlib.Path,
    original_plan: Mapping[str, Any],
    banks: Mapping[str, Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    mixed_variants: set[str],
    fresh: Mapping[str, Any],
) -> dict[str, Any]:
    historical = set(fresh["exclusion_context"]["fingerprints"])
    fresh_variants = _all_variants(fresh["document"])
    original_spent = _all_variants(
        fresh["exclusion_context"]["original_documents"]["tuple_confirmation"]
    )
    if fresh_variants & historical or fresh_variants & original_spent:
        raise RecoveryError("fresh confirmation bank overlaps an excluded source")
    selected_six = set(mixed_variants)
    eventual_final_union = set(selected_six)
    eventual_final_union.update(original_spent)
    additional = _additional_exclusions(original_plan)
    return {
        "schema": MIXED_EXCLUSION_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": RECOVERY_ID,
        "source_campaign_id": SOURCE_CAMPAIGN_ID,
        "status": "mixed-six-development-exclusion-sealed",
        "plan_created_at_utc": plan_created_at_utc,
        "materialized_at_utc": materialized_at_utc,
        "original_development_plan": _sealed_record(
            original_plan_path, development.PLAN_SCHEMA
        ),
        "recovery_plan": _sealed_record(recovery_plan_path, PLAN_SCHEMA),
        "original_exclusion_receipt": dict(original_plan["exclusion"]["receipt"]),
        "selected_banks": list(rows),
        "selected_stage_order": list(STAGE_ORDER),
        "selected_bank_count": 6,
        "selected_opening_count": sum(STAGE_PAIRS.values()),
        "historical_exclusions": list(fresh["exclusion_context"]["copied_sources"]),
        "historical_exclusion_count": 7,
        "protected_fingerprint_source": dict(
            fresh["exclusion_context"]["protected_source"]
        ),
        "protected_fingerprint_count": len(
            fresh["exclusion_context"]["protected_fingerprints"]
        ),
        "selected_six_symmetry_fingerprint_count": len(selected_six),
        "selected_six_symmetry_fingerprints_sha256": qualification.sha256_bytes(
            qualification.canonical_json_bytes(sorted(selected_six))
        ),
        "eventual_final_development_symmetry_fingerprint_count": len(
            eventual_final_union
        ),
        "eventual_final_development_symmetry_fingerprints_sha256": qualification.sha256_bytes(
            qualification.canonical_json_bytes(sorted(eventual_final_union))
        ),
        "cross_source_symmetry_intersection_count": 0,
        "fresh_confirmation_excluded_original_six": True,
        "fresh_confirmation_excluded_historical_seven": True,
        "fresh_confirmation_excluded_protected_fingerprints": True,
        "additional_development_exclusions": additional,
        "eventual_protected_final_development_bank_count": 7,
        "selection_uses_only_selected_six": True,
        "protected_final_generation_authorized": False,
        "rank4_gate_authorized": False,
        "upload_authorized": False,
    }


def _write_sealed(path: pathlib.Path, body: Mapping[str, Any]) -> dict[str, Any]:
    return qualification.write_sealed(path, body)


def _incident_body(
    *, created_at_utc: str, original: Mapping[str, Any]
) -> dict[str, Any]:
    carried_records = [
        {
            key: value
            for key, value in item.items()
            if key in {"order", "stage", "candidate_id", "tuple", "reference"}
        }
        for item in original["carried"]
    ]
    return {
        "schema": INCIDENT_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": SOURCE_CAMPAIGN_ID,
        "status": "original-development-attempt-terminal-no-retry",
        "created_at_utc": created_at_utc,
        "original_development_plan": _sealed_record(
            original["plan_path"], development.PLAN_SCHEMA
        ),
        "historical_supervisor": original["journal"]["supervisor"],
        "terminal_journal": {
            key: value
            for key, value in original["journal"].items()
            if key != "supervisor"
        },
        "carried_receipt_references": carried_records,
        "carried_receipt_count": 10,
        "forensic_result": original["forensic"],
        "root_cause": dict(original["forensic"]["root_cause"]),
        "original_process_absence": original["process_absence"],
        "original_attempts_consumed": 1,
        "original_attempts_remaining": 0,
        "replay_under_original_plan_authorized": False,
        "old_tuple_confirmation_results_eligible_for_selection": False,
    }


def _binary_context(carried: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    binaries: dict[str, Any] = {}
    compile_references: dict[str, Any] = {}
    for item in carried:
        validated = item["validated"]
        candidate_id = validated["request"]["candidate"]["candidate_id"]
        candidate = validated["request"]["candidate"]
        binary = _regular(pathlib.Path(candidate["binary_path"]))
        compile_reference = dict(validated["request"]["compile_reference"])
        if candidate_id in binaries and binaries[candidate_id] != binary:
            raise RecoveryError("carried candidate binary identity is inconsistent")
        if candidate_id in compile_references and compile_references[candidate_id] != compile_reference:
            raise RecoveryError("carried compile-reference identity is inconsistent")
        binaries[candidate_id] = binary
        compile_references[candidate_id] = compile_reference
    if (
        set(binaries) != {development.CANDIDATE_ID, development.CONTROL_ID}
        or binaries[development.CANDIDATE_ID]["sha256"] != CANDIDATE_BINARY_SHA256
        or binaries[development.CONTROL_ID]["sha256"] != CONTROL_BINARY_SHA256
    ):
        raise RecoveryError("exact recovery binary roster changed")
    return binaries, compile_references


def _tool_records(*, recovery_runner: pathlib.Path) -> dict[str, Any]:
    if recovery_runner.resolve() != RECOVERY_RUNNER_PATH.resolve():
        raise RecoveryError("recovery runner path changed")
    paths = {
        "recovery_tool": pathlib.Path(__file__).resolve(),
        "recovery_test": TEST_PATH,
        "recovery_runner": recovery_runner,
        "recovery_runner_test": RUNNER_TEST_PATH,
        "original_development_tool": pathlib.Path(development.__file__).resolve(),
        "original_development_runner": development.RUNNER_PATH,
        "maintained_runner": development.MAINTAINED_RUNNER_PATH,
        "gate_source": development.maintained.GATE_SOURCE,
        "gate_support": BOT_ROOT / "rank4_gate_support.py",
        "rank4_source": development.maintained.RANK4,
        "opening_tool": pathlib.Path(openings.__file__).resolve(),
        "qualification_tool": pathlib.Path(qualification.__file__).resolve(),
    }
    return {name: _regular(path.resolve()) for name, path in paths.items()}


def _fresh_descriptor(
    *, outputs: Mapping[str, str], exclusion_context: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "stage": "tuple_confirmation",
        "classification": "unprotected-development",
        "opening_count": 250,
        "seed_hex": RECOVERY_BANK_SEED.hex(),
        "domain": RECOVERY_BANK_DOMAIN,
        "excluded_source_count": 14,
        "protected_fingerprint_source": dict(
            exclusion_context["protected_source"]
        ),
        "excluded_symmetry_fingerprint_count": len(
            exclusion_context["fingerprints"]
        ),
        "excluded_symmetry_fingerprints_sha256": qualification.sha256_bytes(
            qualification.canonical_json_bytes(
                sorted(exclusion_context["fingerprints"])
            )
        ),
        "exclusions_body_sha256": exclusion_context["body_sha256"],
        "manifest_directory": outputs["opening_banks"],
        "gate_directory": outputs["gate_banks"],
        "manifest_filename_policy": "sha256.opening-bank.json",
        "gate_filename_policy": "sha256.tsv",
        "materialized_at_plan_creation": False,
    }


def _recovery_contract(fresh_descriptor: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "resume_from_stage": "tuple_confirmation",
        "carried_stages": ["model_screen", "tuple_screen"],
        "carried_receipt_count": 10,
        "tuple_confirmation_roster": [
            {
                "candidate_id": "discrete-v3-search-target:c0.65-f0.5-l1",
                "tuple": ["0.65", "0.5", "1"],
            },
            {
                "candidate_id": "discrete-v3-search-target:c0.80-f0.5-l1",
                "tuple": ["0.80", "0.5", "1"],
            },
            {
                "candidate_id": "discrete-v3-search-target:c0.95-f0.5-l1",
                "tuple": ["0.95", "0.5", "1"],
            },
        ],
        "tuple_confirmation_bank": dict(fresh_descriptor),
        "games_per_confirmation_candidate": 500,
        "gate_configuration_invariants": {
            "mode": "fixed-work-except-actual-clock",
            "pair_offset": 0,
            "stage_pairs": dict(STAGE_PAIRS),
            "candidate_actions": 250,
            "candidate_expansions": 2_000_000,
            "candidate_shuffle_seed": 1,
            "candidate_clocks_ms": [800, 155],
            "rank4_nodes": 3_000_000,
            "rank4_clocks_ms": [800, 165],
            "max_turns": 320,
            "actual_clock_candidate_wins_min": 211,
            "actual_clock_wins_per_color_min": 104,
            "failures_required": 0,
        },
        "old_tuple_confirmation_selection_weight": 0,
        "fresh_bank_full_roster_replacement_only": True,
        "profile_screen_bank_reused_untouched": True,
        "profile_confirmation_bank_reused_untouched": True,
        "actual_clock_bank_reused_untouched": True,
        "selection_rule": "original-best-two-plus-default-and-bootstrap-unchanged",
        "one_shot_no_replay": True,
    }


def _policy() -> dict[str, Any]:
    return {
        "recovery_games_authorized": True,
        "recovery_attempts_authorized": 1,
        "original_attempt_replay_authorized": False,
        "old_confirmation_results_eligible_for_selection": False,
        "carried_receipts_immutable": True,
        "selection_may_change_model_weights": False,
        "final_bank_generation_authorized": False,
        "rank4_gate_authorized": False,
        "upload_authorized": False,
    }


def prepare_recovery(
    *,
    output_root: pathlib.Path,
    recovery_runner: pathlib.Path,
    created_at_utc: str,
) -> pathlib.Path:
    output_root = _campaign_root(output_root)
    created_at_utc = _utc(created_at_utc, "recovery plan timestamp")
    paths = _paths(output_root)
    plan_path = pathlib.Path(paths["plan"])
    if plan_path.exists():
        validate_recovery_plan(plan_path, output_root=output_root)
        return plan_path
    recovery_root = pathlib.Path(paths["recovery_root"])
    if os.path.lexists(recovery_root):
        if recovery_root.is_symlink() or not recovery_root.is_dir():
            raise RecoveryError("recovery root is redirected or irregular")
        if any(recovery_root.iterdir()):
            raise RecoveryError("recovery root predates its immutable plan")
    original = validate_original_state(output_root)
    incident_path = pathlib.Path(paths["incident"])
    _write_sealed(
        incident_path,
        _incident_body(created_at_utc=created_at_utc, original=original),
    )
    exclusion_context = _exclusion_context(output_root, original["plan"])
    fresh_descriptor = _fresh_descriptor(
        outputs=paths, exclusion_context=exclusion_context
    )
    banks = {
        stage: (
            fresh_descriptor
            if stage == "tuple_confirmation"
            else dict(original["plan"]["banks"][stage])
        )
        for stage in STAGE_ORDER
    }
    binaries, compile_references = _binary_context(original["carried"])
    carried_records = [
        {
            key: value
            for key, value in item.items()
            if key in {"order", "stage", "candidate_id", "tuple", "reference"}
        }
        for item in original["carried"]
    ]
    original_section = {
        "development_plan": _sealed_record(
            original["plan_path"], development.PLAN_SCHEMA
        ),
        "carried_receipt_references": carried_records,
        "carried_receipt_count": 10,
        "terminal_incident": _sealed_record(incident_path, INCIDENT_SCHEMA),
        "attempt_status": "terminal-no-retry",
        "attempts_consumed": 1,
        "attempts_remaining": 0,
    }
    body = {
        "schema": PLAN_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": RECOVERY_ID,
        "source_campaign_id": SOURCE_CAMPAIGN_ID,
        "recovery_id": RECOVERY_ID,
        "status": "recovery-preregistered-awaiting-one-shot-confirmation",
        "created_at_utc": created_at_utc,
        "attempt": {
            "ordinal": 1,
            "maximum": 1,
            "remaining_before_launch": 1,
            "replay_allowed": False,
        },
        "original": original_section,
        "candidate": dict(original["plan"]["candidate"]),
        "rank4_control": dict(original["plan"]["rank4_control"]),
        "banks": banks,
        "additional_development_exclusions": _additional_exclusions(
            original["plan"]
        ),
        "mixed_six_exclusion": {
            "path": paths["mixed_six_exclusion"],
            "schema": MIXED_EXCLUSION_SCHEMA,
            "materialized_at_plan_creation": False,
        },
        "binaries": binaries,
        "compile_references": compile_references,
        "algorithm": dict(original["plan"]["algorithm"]),
        "recovery_contract": _recovery_contract(fresh_descriptor),
        "compiler": dict(original["plan"]["compiler"]),
        "tools": _tool_records(recovery_runner=recovery_runner.resolve()),
        "concurrency": CONCURRENCY,
        "outputs": paths,
        "policy": _policy(),
    }
    _write_sealed(plan_path, body)
    validate_recovery_plan(plan_path, output_root=output_root)
    return plan_path


def materialize_recovery_bank(
    *, plan_path: pathlib.Path, output_root: pathlib.Path,
    materialized_at_utc: str,
) -> dict[str, Any]:
    """Materialize the preregistered bank exactly once, without running games."""

    materialized_at_utc = _utc(
        materialized_at_utc, "recovery-bank materialization timestamp"
    )
    context = validate_recovery_plan(plan_path, output_root=output_root)
    if context["materialized"]:
        return context
    plan = context["plan"]
    plan_time = qualification._utc(
        plan["created_at_utc"], "recovery plan timestamp"
    )
    materialized_time = qualification._utc(
        materialized_at_utc, "recovery-bank materialization timestamp"
    )
    if materialized_time < plan_time:
        raise RecoveryError("recovery bank cannot predate its preregistered plan")
    outputs = plan["outputs"]
    mixed_path = pathlib.Path(outputs["mixed_six_exclusion"])
    if os.path.lexists(mixed_path):
        raise RecoveryError("mixed-six exclusion predates bank materialization")
    original_plan = context["original_plan"]
    fresh = create_fresh_confirmation_bank(
        output_root,
        original_plan,
        plan_path,
        pathlib.Path(outputs["opening_banks"]),
        pathlib.Path(outputs["gate_banks"]),
    )
    banks, rows, mixed_variants = _mixed_bank_context(original_plan, fresh)
    _write_sealed(
        mixed_path,
        _mixed_exclusion_body(
            plan_created_at_utc=plan["created_at_utc"],
            materialized_at_utc=materialized_at_utc,
            original_plan_path=context["original_plan_path"],
            recovery_plan_path=plan_path,
            original_plan=original_plan,
            banks=banks,
            rows=rows,
            mixed_variants=mixed_variants,
            fresh=fresh,
        ),
    )
    result = validate_recovery_plan(plan_path, output_root=output_root)
    if not result["materialized"]:
        raise RecoveryError("recovery bank did not reach a complete materialized state")
    return result


def _validate_mixed_exclusion(
    *,
    plan_path: pathlib.Path,
    plan: Mapping[str, Any],
    original_plan_path: pathlib.Path,
    original_plan: Mapping[str, Any],
    banks: Mapping[str, Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    mixed_variants: set[str],
    fresh: Mapping[str, Any],
) -> dict[str, Any]:
    descriptor = plan.get("mixed_six_exclusion")
    expected_descriptor = {
        "path": plan["outputs"]["mixed_six_exclusion"],
        "schema": MIXED_EXCLUSION_SCHEMA,
        "materialized_at_plan_creation": False,
    }
    if descriptor != expected_descriptor:
        raise RecoveryError("mixed-six output descriptor changed")
    path = pathlib.Path(expected_descriptor["path"])
    mixed = qualification.load_sealed(path, MIXED_EXCLUSION_SCHEMA)
    materialized_at = _utc(
        mixed.get("materialized_at_utc"), "mixed-six materialization timestamp"
    )
    if qualification._utc(
        materialized_at, "mixed-six materialization timestamp"
    ) < qualification._utc(plan["created_at_utc"], "recovery plan timestamp"):
        raise RecoveryError("mixed-six exclusion predates its recovery plan")
    expected = qualification.seal(
        _mixed_exclusion_body(
            plan_created_at_utc=plan["created_at_utc"],
            materialized_at_utc=materialized_at,
            original_plan_path=original_plan_path,
            recovery_plan_path=plan_path,
            original_plan=original_plan,
            banks=banks,
            rows=rows,
            mixed_variants=mixed_variants,
            fresh=fresh,
        )
    )
    if mixed != expected:
        raise RecoveryError("mixed-six exclusion contract changed")
    return mixed


def validate_recovery_plan(
    path: pathlib.Path, *, output_root: pathlib.Path
) -> dict[str, Any]:
    output_root = _campaign_root(output_root)
    expected_paths = _paths(output_root)
    expected_path = pathlib.Path(expected_paths["plan"])
    if path.absolute() != path.resolve() or path.resolve() != expected_path or path.is_symlink():
        raise RecoveryError("recovery plan path changed")
    plan = qualification.load_sealed(path, PLAN_SCHEMA)
    if set(plan) != PLAN_FIELDS:
        raise RecoveryError("recovery plan field roster changed")
    created_at = _utc(plan.get("created_at_utc"), "recovery plan timestamp")
    if (
        plan.get("namespace") != NAMESPACE
        or plan.get("campaign_id") != RECOVERY_ID
        or plan.get("source_campaign_id") != SOURCE_CAMPAIGN_ID
        or plan.get("recovery_id") != RECOVERY_ID
        or plan.get("status") != "recovery-preregistered-awaiting-one-shot-confirmation"
        or plan.get("attempt")
        != {"ordinal": 1, "maximum": 1, "remaining_before_launch": 1, "replay_allowed": False}
        or plan.get("outputs") != expected_paths
        or set(plan.get("outputs", {})) != OUTPUT_FIELDS
        or plan.get("policy") != _policy()
        or plan.get("concurrency") != CONCURRENCY
    ):
        raise RecoveryError("recovery plan identity/policy changed")
    original = validate_original_state(output_root)
    original_section = plan.get("original")
    expected_carried = [
        {
            key: value
            for key, value in item.items()
            if key in {"order", "stage", "candidate_id", "tuple", "reference"}
        }
        for item in original["carried"]
    ]
    incident_path = _verify_sealed_record(
        original_section.get("terminal_incident") if isinstance(original_section, Mapping) else None,
        INCIDENT_SCHEMA,
        "terminal incident",
    )
    incident = qualification.load_sealed(incident_path, INCIDENT_SCHEMA)
    if (
        not isinstance(original_section, Mapping)
        or set(original_section) != {
            "development_plan",
            "carried_receipt_references",
            "carried_receipt_count",
            "terminal_incident",
            "attempt_status",
            "attempts_consumed",
            "attempts_remaining",
        }
        or original_section.get("development_plan")
        != _sealed_record(original["plan_path"], development.PLAN_SCHEMA)
        or original_section.get("carried_receipt_references") != expected_carried
        or original_section.get("carried_receipt_count") != 10
        or original_section.get("attempt_status") != "terminal-no-retry"
        or original_section.get("attempts_consumed") != 1
        or original_section.get("attempts_remaining") != 0
        or incident != qualification.seal(
            _incident_body(created_at_utc=incident["created_at_utc"], original=original)
        )
    ):
        raise RecoveryError("original-attempt recovery ancestry changed")
    if (
        plan.get("candidate") != original["plan"]["candidate"]
        or plan.get("rank4_control") != original["plan"]["rank4_control"]
        or plan.get("algorithm") != original["plan"]["algorithm"]
        or plan.get("compiler") != original["plan"]["compiler"]
        or dict(development.maintained._default_compiler_identity()) != plan["compiler"]
    ):
        raise RecoveryError("candidate/control/compiler/algorithm ancestry changed")
    banks = plan.get("banks")
    if not isinstance(banks, Mapping) or set(banks) != set(STAGE_ORDER):
        raise RecoveryError("recovery six-bank stage roster changed")
    exclusion_context = _exclusion_context(output_root, original["plan"])
    fresh_descriptor = _fresh_descriptor(
        outputs=expected_paths, exclusion_context=exclusion_context
    )
    for stage in STAGE_ORDER:
        if stage == "tuple_confirmation":
            if dict(banks[stage]) != fresh_descriptor:
                raise RecoveryError("fresh confirmation preregistration changed")
        elif dict(banks[stage]) != original["plan"]["banks"][stage]:
            raise RecoveryError(f"recovery changed untouched {stage} bank")
    additional = _additional_exclusions(original["plan"])
    if plan.get("additional_development_exclusions") != additional:
        raise RecoveryError("spent original confirmation exclusion changed")
    expected_mixed_descriptor = {
        "path": expected_paths["mixed_six_exclusion"],
        "schema": MIXED_EXCLUSION_SCHEMA,
        "materialized_at_plan_creation": False,
    }
    if plan.get("mixed_six_exclusion") != expected_mixed_descriptor:
        raise RecoveryError("mixed-six preregistration changed")

    bank_directory = pathlib.Path(expected_paths["opening_banks"])
    gate_directory = pathlib.Path(expected_paths["gate_banks"])
    mixed_path = pathlib.Path(expected_paths["mixed_six_exclusion"])
    bank_files = (
        sorted(bank_directory.glob("*.opening-bank.json"))
        if bank_directory.is_dir() and not bank_directory.is_symlink()
        else []
    )
    gate_files = (
        sorted(gate_directory.glob("*.tsv"))
        if gate_directory.is_dir() and not gate_directory.is_symlink()
        else []
    )
    materialization_markers = (len(bank_files), len(gate_files), os.path.lexists(mixed_path))
    materialized = materialization_markers == (1, 1, True)
    if materialization_markers not in {(0, 0, False), (1, 1, True)}:
        raise RecoveryError("recovery bank materialization is partial or ambiguous")
    if not materialized and (
        os.path.lexists(bank_directory)
        or os.path.lexists(gate_directory)
        or os.path.lexists(mixed_path)
    ):
        raise RecoveryError("recovery materialization outputs must be wholly absent before generation")

    bank_documents: dict[str, dict[str, Any]] = {}
    materialized_banks: dict[str, dict[str, Any]] | None = None
    mixed: dict[str, Any] | None = None
    if materialized:
        fresh_path = bank_files[0]
        fresh_bank = openings.validate_bank(fresh_path)
        expected_payload, expected_openings = _fresh_bank_payload(
            original_plan=original["plan"],
            context=exclusion_context,
            recovery_plan_path=path,
        )
        expected_bank = qualification.seal(expected_payload)
        if (
            fresh_bank != expected_bank
            or fresh_path.name
            != f"{qualification.sha256_file(fresh_path)}.opening-bank.json"
            or fresh_bank.get("opening_count") != 250
            or len(expected_openings) != 250
            or _all_variants(fresh_bank) & exclusion_context["fingerprints"]
        ):
            raise RecoveryError("materialized fresh confirmation bank changed")
        gate_path = gate_files[0]
        expected_gate = _gate_bank_bytes(fresh_bank)
        if (
            gate_path.is_symlink()
            or not gate_path.is_file()
            or gate_path.read_bytes() != expected_gate
            or gate_path.name
            != f"{qualification.sha256_bytes(expected_gate)}.tsv"
        ):
            raise RecoveryError("fresh confirmation gate-bank derivative changed")
        development.maintained.gate_support.validate_bank(gate_path)
        fresh = {
            "manifest": _regular(fresh_path),
            "gate": _regular(gate_path),
            "document": fresh_bank,
            "exclusion_context": exclusion_context,
        }
        materialized_banks, rows, mixed_variants = _mixed_bank_context(
            original["plan"], fresh
        )
        bank_documents = {
            stage: (
                fresh_bank
                if stage == "tuple_confirmation"
                else openings.validate_bank(
                    pathlib.Path(materialized_banks[stage]["path"])
                )
            )
            for stage in STAGE_ORDER
        }
        mixed = _validate_mixed_exclusion(
            plan_path=path,
            plan=plan,
            original_plan_path=original["plan_path"],
            original_plan=original["plan"],
            banks=materialized_banks,
            rows=rows,
            mixed_variants=mixed_variants,
            fresh=fresh,
        )
    binaries, compile_references = _binary_context(original["carried"])
    if plan.get("binaries") != binaries or plan.get("compile_references") != compile_references:
        raise RecoveryError("recovery binary/compile-reference bindings changed")
    tools = plan.get("tools")
    if not isinstance(tools, Mapping) or set(tools) != {
        "recovery_tool",
        "recovery_test",
        "recovery_runner",
        "recovery_runner_test",
        "original_development_tool",
        "original_development_runner",
        "maintained_runner",
        "gate_source",
        "gate_support",
        "rank4_source",
        "opening_tool",
        "qualification_tool",
    }:
        raise RecoveryError("recovery tool closure changed")
    for name, record in tools.items():
        _verify_record(record, name)
    if (
        tools["recovery_tool"] != _regular(pathlib.Path(__file__).resolve())
        or tools["recovery_test"] != _regular(TEST_PATH)
        or tools["recovery_runner"] != _regular(RECOVERY_RUNNER_PATH)
        or tools["recovery_runner_test"] != _regular(RUNNER_TEST_PATH)
        or tools["original_development_tool"]
        != _regular(pathlib.Path(development.__file__).resolve())
        or tools["original_development_runner"] != _regular(development.RUNNER_PATH)
        or tools["maintained_runner"] != _regular(development.MAINTAINED_RUNNER_PATH)
        or tools["gate_source"] != _regular(development.maintained.GATE_SOURCE)
        or tools["gate_support"] != _regular(BOT_ROOT / "rank4_gate_support.py")
        or tools["rank4_source"] != _regular(development.maintained.RANK4)
        or tools["opening_tool"] != _regular(pathlib.Path(openings.__file__).resolve())
        or tools["qualification_tool"] != _regular(pathlib.Path(qualification.__file__).resolve())
    ):
        raise RecoveryError("recovery static tool identities changed")
    contract = plan.get("recovery_contract")
    if not isinstance(contract, Mapping) or contract != _recovery_contract(
        fresh_descriptor
    ):
        raise RecoveryError("recovery execution contract changed")
    return {
        "plan": plan,
        "plan_path": path.resolve(),
        "original_plan": original["plan"],
        "original_plan_path": original["plan_path"],
        "incident": incident,
        "carried": original["carried"],
        "materialized": materialized,
        "materialized_banks": materialized_banks,
        "development_bank_records": materialized_banks,
        "bank_documents": bank_documents,
        "mixed_exclusion": mixed,
        "materialized_mixed_six_exclusion": (
            _sealed_record(
                pathlib.Path(expected_paths["mixed_six_exclusion"]),
                MIXED_EXCLUSION_SCHEMA,
            )
            if materialized
            else None
        ),
        "additional_development_exclusions": additional,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--output-root", type=pathlib.Path, required=True)
    prepare.add_argument("--recovery-runner", type=pathlib.Path, required=True)
    prepare.add_argument("--created-at-utc", required=True)
    materialize = commands.add_parser("materialize-bank")
    materialize.add_argument("--plan", type=pathlib.Path, required=True)
    materialize.add_argument("--output-root", type=pathlib.Path, required=True)
    materialize.add_argument("--materialized-at-utc", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--plan", type=pathlib.Path, required=True)
    validate.add_argument("--output-root", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            plan_path = prepare_recovery(
                output_root=args.output_root,
                recovery_runner=args.recovery_runner,
                created_at_utc=args.created_at_utc,
            )
            result = {
                "status": "recovery-preregistered",
                "plan": str(plan_path),
                "plan_sha256": qualification.sha256_file(plan_path),
            }
        elif args.command == "materialize-bank":
            context = materialize_recovery_bank(
                plan_path=args.plan,
                output_root=args.output_root,
                materialized_at_utc=args.materialized_at_utc,
            )
            result = {
                "status": "recovery-bank-materialized",
                "plan": str(context["plan_path"]),
                "plan_sha256": qualification.sha256_file(context["plan_path"]),
                "mixed_six_exclusion": context["plan"]["outputs"]["mixed_six_exclusion"],
            }
        else:
            context = validate_recovery_plan(
                args.plan, output_root=args.output_root
            )
            result = {
                "status": "recovery-plan-valid",
                "plan": str(context["plan_path"]),
                "plan_sha256": qualification.sha256_file(context["plan_path"]),
                "carried_receipt_count": len(context["carried"]),
                "materialized": context["materialized"],
            }
    except (OSError, RecoveryError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
