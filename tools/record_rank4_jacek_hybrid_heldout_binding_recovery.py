#!/usr/bin/env python3
"""One-shot bind portability recovery for the passed c807 preflight.

This administrative recorder does not repeat or weaken the recovery-v1
preflight.  It carries that exact successful receipt forward, keeps c807 as
the source candidate, binds this script's direct-child administrative commit,
and writes every binding/stage/decision artifact to a new sibling namespace.
"""

from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shlex
import stat
import subprocess
import sys
import time
from typing import Any, Iterable

import record_rank4_jacek_hybrid_final_source_preflight as preflight


ROOT = Path(__file__).resolve().parents[1]
RECORDER = Path(__file__).resolve()
FROZEN_RECORDER = ROOT / (
    "tools/record_rank4_jacek_hybrid_heldout_qualification.py"
)
FROZEN_RECORDER_TEST = ROOT / (
    "tests/codingame/test_rank4_jacek_hybrid_heldout_qualification.py"
)
_FROZEN_SPEC = importlib.util.spec_from_file_location(
    "_rank4_jacek_hybrid_frozen_heldout_qualification", FROZEN_RECORDER
)
if _FROZEN_SPEC is None or _FROZEN_SPEC.loader is None:
    raise RuntimeError("cannot load frozen c807 qualification recorder")
frozen = importlib.util.module_from_spec(_FROZEN_SPEC)
_FROZEN_SPEC.loader.exec_module(frozen)
RECORDER_TEST = ROOT / (
    "tests/codingame/"
    "test_rank4_jacek_hybrid_heldout_binding_recovery.py"
)
QUALIFICATION_ROOT = (
    ROOT / "results/rank_4_jacek_hybrid/gates/heldout_qualification"
)
PARENT_OUTPUT = QUALIFICATION_ROOT / "recovery_v1"
OUTPUT = QUALIFICATION_ROOT / "binding_recovery_v1"
BINDING_RECOVERY_PLAN = OUTPUT / "PLAN.json"
PREBIND_BLOCKER = OUTPUT / (
    "predecessor_failures/"
    "fc2de736ccf5e1e1bc86f54f0effd251dc1715dc74e1a74c8cc6a33e331315c3.json"
)
CANDIDATE_SOURCE_COMMIT = "c8077067cafff2e0fed8b4c85082de0392fc453c"
CARRY_CLAIM = PARENT_OUTPUT / (
    "preflight/claims/"
    f"{CANDIDATE_SOURCE_COMMIT}.json"
)
CARRY_RECEIPT_SHA256 = (
    "de8d2d40076a38d2f973c0e665470f60af69354a7842648267a3ca794e5ce0d3"
)
CARRY_RECEIPT = PARENT_OUTPUT / (
    f"preflight/receipts/{CARRY_RECEIPT_SHA256}.json"
)
CARRY_CLAIM_SHA256 = (
    "c295d626054d856b44cc837088d6db605955610caa2c60014d76ee735b24151d"
)
PARENT_PREFLIGHT_FAILURE_SHA256 = (
    "bb149d5dcf2ae3b33cb0dafd7725e3ad263a60c617224f3d648c9c9a6fdd3bd4"
)
BINDING_RECOVERY_PLAN_SHA256 = (
    "fd34d98b0f3660ec1dc814064452d29fc7f31b990fddfedaed10b79de152d722"
)
PREBIND_BLOCKER_SHA256 = (
    "fc2de736ccf5e1e1bc86f54f0effd251dc1715dc74e1a74c8cc6a33e331315c3"
)
ADMIN_PREREGISTERED_UTC = "2026-08-14T09:07:16Z"
ADMIN_CAMPAIGN_ID = (
    "rank_4_jacek_hybrid-36h-20260813-binding-portability-recovery-v1"
)

PLAN_SCHEMA = "rank4-jacek-hybrid-heldout-binding-portability-plan-v1"
BINDING_SCHEMA = "rank4-jacek-hybrid-heldout-binding-portability-v1"
BIND_CLAIM_SCHEMA = (
    "rank4-jacek-hybrid-heldout-binding-portability-claim-v1"
)
STAGE_CLAIM_SCHEMA = (
    "rank4-jacek-hybrid-heldout-stage-claim-binding-portability-v1"
)
STAGE_REPORT_SCHEMA = (
    "rank4-jacek-hybrid-heldout-stage-report-binding-portability-v1"
)
DECISION_SCHEMA = (
    "rank4-jacek-hybrid-heldout-decision-binding-portability-v1"
)

BIND_LOCK = ROOT / (
    "build/rank4-jacek-hybrid-heldout-binding-portability-v1.lock"
)
PRIVATE_LOCK = ROOT / (
    "build/rank4-jacek-hybrid-heldout-qualification-binding-portability-v1.lock"
)
SHARED_LOCK = Path("/tmp/rank4-hybrid-prototype-benchmark.lock")

SDK_SELECTOR = Path(
    "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk"
)
SDK_SELECTOR_READLINK = "MacOSX26.5.sdk"
SDK_RESOLVED_ROOT = Path(
    "/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk"
)
SDK_ALIAS_SPECS = (
    {
        "lexical_path": str(SDK_SELECTOR / "usr/include/pthread.h"),
        "versioned_link_path": str(
            SDK_RESOLVED_ROOT / "usr/include/pthread.h"
        ),
        "readlink": "pthread/pthread.h",
        "link_mode": "0755",
        "link_bytes": 17,
        "resolved_path": str(
            SDK_RESOLVED_ROOT / "usr/include/pthread/pthread.h"
        ),
        "terminal": {
            "bytes": 28093,
            "mode": "0644",
            "sha256": (
                "9d621c730d1d96b600893b0e3e4c45822a24d565e7b6b166973c41a6a2eb02e7"
            ),
        },
    },
    {
        "lexical_path": str(SDK_SELECTOR / "usr/include/sched.h"),
        "versioned_link_path": str(
            SDK_RESOLVED_ROOT / "usr/include/sched.h"
        ),
        "readlink": "pthread/sched.h",
        "link_mode": "0755",
        "link_bytes": 15,
        "resolved_path": str(
            SDK_RESOLVED_ROOT / "usr/include/pthread/sched.h"
        ),
        "terminal": {
            "bytes": 1410,
            "mode": "0644",
            "sha256": (
                "07d65b8d135be978a33579b5a333bd4576a8c6b36351ef510e0cca2aa0987020"
            ),
        },
    },
)
SDK_ALIAS_PATHS = {Path(item["lexical_path"]) for item in SDK_ALIAS_SPECS}
PS_PATH = Path("/bin/ps")
PS_ARGV = ["/bin/ps", "-axo", "pid=,ppid=,command="]
PS_IDENTITY = {
    "path": "/bin/ps",
    "bytes": 170816,
    "sha256": (
        "472992c470606d28f577590decfecd7f4a20f832fd92c671bebc6d44790b5d02"
    ),
    "ascii": False,
    "mode": "4755",
    "executable": True,
}

ALLOWED_CHANGED_PATHS = tuple(sorted((
    str(BINDING_RECOVERY_PLAN.relative_to(ROOT)),
    str(PREBIND_BLOCKER.relative_to(ROOT)),
    str(CARRY_CLAIM.relative_to(ROOT)),
    str(CARRY_RECEIPT.relative_to(ROOT)),
    str(RECORDER.relative_to(ROOT)),
    str(RECORDER_TEST.relative_to(ROOT)),
)))

PARENT_RUNTIME_REGISTRIES = tuple(
    PARENT_OUTPUT / name
    for name in ("binding_claims", "bindings", "claims", "reports", "decisions")
)
CHILD_RUNTIME_REGISTRIES = tuple(
    OUTPUT / name
    for name in ("binding_claims", "bindings", "claims", "reports", "decisions")
)


def canonical_json(value: Any) -> bytes:
    return frozen.canonical_json(value)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _mode(metadata: os.stat_result) -> str:
    return format(stat.S_IMODE(metadata.st_mode), "04o")


def _reject_tsv_before_metadata(
    lexical: Path, allowed_stage_banks: Iterable[Path],
) -> set[Path]:
    allowed = {_lexical_absolute(item) for item in allowed_stage_banks}
    if lexical.suffix.lower() == ".tsv" and lexical not in allowed:
        raise ValueError(f"TSV path is forbidden before its stage claim: {lexical}")
    if (re.fullmatch(r"(?:validation|final)_d[0-9]+\.tsv", lexical.name.lower())
            and lexical not in allowed):
        raise ValueError(f"sealed bank path is forbidden before claim: {lexical}")
    return allowed


def _ancestor_symlinks(path: Path) -> list[Path]:
    """Return lexical symlink ancestors without resolving the input first."""
    result: list[Path] = []
    parts = path.parts
    cursor = Path(parts[0])
    for part in parts[1:-1]:
        cursor /= part
        metadata = os.lstat(cursor)
        if stat.S_ISLNK(metadata.st_mode):
            result.append(cursor)
    return result


def _regular_identity(path: Path) -> dict[str, Any]:
    metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"expected exact regular file: {path}")
    raw = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(raw),
        "sha256": sha256_bytes(raw),
        "ascii": all(byte < 128 for byte in raw),
        "mode": _mode(metadata),
        "executable": os.access(path, os.X_OK),
    }


def _selector_evidence() -> dict[str, Any]:
    metadata = os.lstat(SDK_SELECTOR)
    if (not stat.S_ISLNK(metadata.st_mode) or
            os.readlink(SDK_SELECTOR) != SDK_SELECTOR_READLINK or
            _mode(metadata) != "0755" or metadata.st_size != 14 or
            SDK_SELECTOR.resolve(strict=True) != SDK_RESOLVED_ROOT or
            SDK_RESOLVED_ROOT.is_symlink() or not SDK_RESOLVED_ROOT.is_dir()):
        raise ValueError("macOS SDK selector contract changed")
    return {
        "lexical_path": str(SDK_SELECTOR),
        "readlink": SDK_SELECTOR_READLINK,
        "link_mode": "0755",
        "link_bytes": 14,
        "resolved_sdk_root": str(SDK_RESOLVED_ROOT),
    }


def _alias_evidence(spec: dict[str, Any]) -> dict[str, Any]:
    lexical = Path(spec["lexical_path"])
    versioned_link = Path(spec["versioned_link_path"])
    expected_resolved = Path(spec["resolved_path"])
    if (Path(spec["readlink"]).is_absolute() or
            not expected_resolved.is_relative_to(SDK_RESOLVED_ROOT)):
        raise ValueError(f"SDK leaf alias escapes its versioned root: {lexical}")
    metadata = os.lstat(versioned_link)
    if (not stat.S_ISLNK(metadata.st_mode) or
            os.readlink(versioned_link) != spec["readlink"] or
            _mode(metadata) != spec["link_mode"] or
            metadata.st_size != spec["link_bytes"] or
            lexical.resolve(strict=True) != expected_resolved or
            versioned_link.resolve(strict=True) != expected_resolved):
        raise ValueError(f"exact SDK leaf alias changed: {lexical}")
    terminal = _regular_identity(expected_resolved)
    terminal.pop("path")
    terminal.pop("ascii")
    terminal.pop("executable")
    if terminal != spec["terminal"]:
        raise ValueError(f"exact SDK leaf target changed: {lexical}")
    return dict(spec)


def _validate_external_alias_policy(path: Path) -> None:
    ancestors = _ancestor_symlinks(path)
    allowed = [SDK_SELECTOR] if path.is_relative_to(SDK_SELECTOR) else []
    if ancestors != allowed:
        raise ValueError(f"unexpected external ancestor symlink: {path}")
    metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode) and path not in SDK_ALIAS_PATHS:
        raise ValueError(f"unexpected external leaf symlink: {path}")


def guard_read_path(
    path: Path, *, allowed_stage_banks: Iterable[Path] = (),
    allow_external: bool = False,
) -> Path:
    """Allow only the two bound SDK aliases and exact receipt-bound /bin/ps."""
    lexical = _lexical_absolute(path)
    allowed = _reject_tsv_before_metadata(lexical, allowed_stage_banks)
    root = _lexical_absolute(ROOT)
    in_root = lexical == root or lexical.is_relative_to(root)
    if in_root:
        if _ancestor_symlinks(lexical):
            raise ValueError(
                f"repository symlink ancestor is forbidden: {lexical}"
            )
        metadata = os.lstat(lexical)
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"repository symlink input is forbidden: {lexical}")
        resolved = lexical.resolve(strict=True)
        if not resolved.is_relative_to(root.resolve(strict=True)):
            raise ValueError(f"repository path escapes repository: {lexical}")
    elif lexical == PS_PATH:
        if not allow_external:
            raise ValueError("exact /bin/ps is not authorized in this context")
        if _regular_identity(PS_PATH) != PS_IDENTITY:
            raise ValueError("receipt-bound /bin/ps identity changed")
        resolved = lexical
    else:
        permitted_roots = tuple(Path(item) for item in (
            "/Applications", "/Library", "/System", "/lib", "/lib64",
            "/opt", "/usr",
        ))
        if (not allow_external or not any(
                lexical == prefix or lexical.is_relative_to(prefix)
                for prefix in permitted_roots)):
            raise ValueError(f"external dependency path is not whitelisted: {lexical}")
        _validate_external_alias_policy(lexical)
        if lexical in SDK_ALIAS_PATHS:
            spec = next(
                item for item in SDK_ALIAS_SPECS
                if Path(item["lexical_path"]) == lexical
            )
            _selector_evidence()
            _alias_evidence(spec)
        elif lexical.is_relative_to(SDK_SELECTOR):
            _selector_evidence()
        resolved = lexical.resolve(strict=True)
        if not any(
                resolved == prefix or resolved.is_relative_to(prefix)
                for prefix in permitted_roots):
            raise ValueError(f"external dependency escapes system roots: {lexical}")
    if resolved.suffix.lower() == ".tsv" and resolved not in allowed:
        raise ValueError(f"resolved TSV is forbidden before claim: {resolved}")
    return resolved


def _raw_depfile_paths(path: Path) -> list[Path]:
    fixed = guard_read_path(path)
    text = fixed.read_text(encoding="utf-8").replace("\\\n", " ")
    if ":" not in text:
        raise ValueError(f"malformed compiler depfile: {path}")
    _, raw = text.split(":", 1)
    result: list[Path] = []
    for token in shlex.split(raw):
        dependency = Path(token)
        if not dependency.is_absolute():
            dependency = ROOT / dependency
        lexical = _lexical_absolute(dependency)
        _reject_tsv_before_metadata(lexical, ())
        result.append(lexical)
    if not result:
        raise ValueError(f"empty compiler depfile: {path}")
    return result


def _require_exact_alias_tokens(tokens: list[Path], label: str) -> None:
    counts = Counter(tokens)
    observed = {path: counts[path] for path in SDK_ALIAS_PATHS}
    expected = {path: 1 for path in SDK_ALIAS_PATHS}
    if observed != expected:
        raise ValueError(f"{label} does not contain each exact SDK alias once")


def _depfile_topology() -> dict[str, Any]:
    whole = sorted(frozen.BUILD_ROOT.rglob("*.o.d"))
    binder = [
        *frozen.DEPFILES,
        *sorted(frozen.CORE_TARGET_DIRECTORY.rglob("*.o.d")),
        *sorted(frozen.OPENING_TARGET_DIRECTORY.rglob("*.o.d")),
    ]
    if len(whole) != 28 or len(binder) != 21 or len(set(binder)) != 21:
        raise ValueError("compiler depfile topology count changed")
    whole_tokens = [_raw_depfile_paths(path) for path in whole]
    binder_tokens = [_raw_depfile_paths(path) for path in binder]
    for index, tokens in enumerate(whole_tokens):
        _require_exact_alias_tokens(tokens, f"whole depfile {index}")
    for index, tokens in enumerate(binder_tokens):
        _require_exact_alias_tokens(tokens, f"binder depfile {index}")
    whole_sets = [set(tokens) for tokens in whole_tokens]
    binder_sets = [set(tokens) for tokens in binder_tokens]
    unique = set().union(*whole_sets)
    missing = 0
    other_leaf: set[Path] = set()
    other_ancestor: set[Path] = set()
    for lexical in sorted(unique):
        try:
            metadata = os.lstat(lexical)
        except FileNotFoundError:
            missing += 1
            continue
        ancestors = _ancestor_symlinks(lexical)
        expected_ancestors = (
            [SDK_SELECTOR] if lexical.is_relative_to(SDK_SELECTOR) else []
        )
        other_ancestor.update(set(ancestors) - set(expected_ancestors))
        if stat.S_ISLNK(metadata.st_mode) and lexical not in SDK_ALIAS_PATHS:
            other_leaf.add(lexical)
        guard_read_path(lexical, allow_external=not lexical.is_relative_to(ROOT))
    whole_counts = [
        {path: Counter(tokens)[path] for path in SDK_ALIAS_PATHS}
        for tokens in whole_tokens
    ]
    binder_counts = [
        {path: Counter(tokens)[path] for path in SDK_ALIAS_PATHS}
        for tokens in binder_tokens
    ]
    exact_alias_count = {path: 1 for path in SDK_ALIAS_PATHS}
    whole_both = sum(counts == exact_alias_count for counts in whole_counts)
    binder_both = sum(counts == exact_alias_count for counts in binder_counts)
    selector_count = sum(
        1 for path in unique if path.is_relative_to(SDK_SELECTOR)
    )
    if (len(unique) != 1089 or selector_count != 1015 or missing != 0 or
            other_leaf or other_ancestor or whole_both != 28 or
            binder_both != 21):
        raise ValueError("compiler dependency portability topology changed")
    return {
        "whole_clang_tree": {
            "depfiles": 28,
            "depfiles_with_both_leaf_aliases": whole_both,
            "unique_lexical_dependencies": len(unique),
            "unique_dependencies_traversing_selector": selector_count,
            "missing_dependencies": missing,
            "other_leaf_symlinks": len(other_leaf),
            "other_ancestor_symlinks": len(other_ancestor),
        },
        "binder_subset": {
            "depfiles": 21,
            "depfiles_with_both_leaf_aliases": binder_both,
            "leaf_alias_references": sum(
                sum(counts.values()) for counts in binder_counts
            ),
        },
    }


def portability_evidence() -> dict[str, Any]:
    ps = _regular_identity(PS_PATH)
    if ps != PS_IDENTITY or preflight.process_table_command() != PS_ARGV:
        raise ValueError("receipt-bound process-table tool changed")
    topology = _depfile_topology()
    return {
        "schema": "rank4-jacek-hybrid-binding-portability-evidence-v1",
        "sdk_selector": _selector_evidence(),
        "sdk_leaf_aliases": [
            _alias_evidence(spec) for spec in SDK_ALIAS_SPECS
        ],
        **topology,
        "process_tool": {"identity": ps, "argv": PS_ARGV},
        "all_other_aliases_forbidden": True,
        "tsv_rejection_precedes_metadata": True,
    }


def portability_sha256(evidence: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(evidence))


def _read_canonical_exact(
    path: Path, expected_sha256: str, expected_bytes: int,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"fixed canonical evidence path is invalid: {path}")
    raw = path.read_bytes()
    if len(raw) != expected_bytes or sha256_bytes(raw) != expected_sha256:
        raise ValueError(f"fixed canonical evidence identity changed: {path}")
    payload = json.loads(raw)
    if not isinstance(payload, dict) or canonical_json(payload) != raw:
        raise ValueError(f"fixed evidence is not canonical JSON: {path}")
    return payload


def _require_exact_registry_entries(
    directory: Path, expected_names: Iterable[str], label: str,
) -> tuple[Path, ...]:
    """Reject missing, aliased, special, or foreign registry entries."""
    metadata = os.lstat(directory)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} registry is not a real directory")
    entries = tuple(sorted(directory.iterdir(), key=lambda item: item.name))
    expected = tuple(sorted(expected_names))
    if tuple(item.name for item in entries) != expected:
        raise ValueError(f"{label} registry cardinality or path mismatch")
    for entry in entries:
        entry_metadata = os.lstat(entry)
        if (stat.S_ISLNK(entry_metadata.st_mode) or
                not stat.S_ISREG(entry_metadata.st_mode)):
            raise ValueError(f"{label} registry entry is not a regular file")
    return entries


def validate_binding_recovery_plan() -> dict[str, Any]:
    _require_exact_registry_entries(
        PREBIND_BLOCKER.parent, (PREBIND_BLOCKER.name,),
        "binding-recovery predecessor failure",
    )
    plan = _read_canonical_exact(
        BINDING_RECOVERY_PLAN, BINDING_RECOVERY_PLAN_SHA256, 12595
    )
    blocker = _read_canonical_exact(
        PREBIND_BLOCKER, PREBIND_BLOCKER_SHA256, 3883
    )
    if (plan.get("schema") != PLAN_SCHEMA or
            plan.get("status") != "preregistered-unbound-one-shot" or
            tuple(sorted(plan["authorized_commit_shape"]["allowed_changed_paths"]))
            != ALLOWED_CHANGED_PATHS or
            plan["authorized_commit_shape"]["candidate_source_commit"] !=
            CANDIDATE_SOURCE_COMMIT or
            plan["prebind_blocker_evidence"]["sha256"] !=
            PREBIND_BLOCKER_SHA256 or
            blocker.get("status") !=
            "read-only-diagnostic-blocked-before-binding-claim" or
            blocker["diagnostic_boundary"]["outer_bind_cli_invoked"] is not False or
            blocker["diagnostic_boundary"]["binding_claim_created"] is not False):
        raise ValueError("binding portability preregistration mismatch")
    return plan


def _require_empty_or_absent(path: Path, label: str) -> None:
    if not os.path.lexists(path):
        return
    if path.is_symlink() or not path.is_dir() or any(path.iterdir()):
        raise ValueError(f"{label} registry is not empty and real: {path}")


def require_parent_runtime_unopened() -> None:
    for path in PARENT_RUNTIME_REGISTRIES:
        _require_empty_or_absent(path, "parent recovery")


def require_child_runtime_unopened_before_bind() -> None:
    for path in CHILD_RUNTIME_REGISTRIES:
        _require_empty_or_absent(path, "binding recovery")


def require_exact_child_binding_registries(
    identifier: str, binding_sha256: str,
) -> None:
    _require_exact_registry_entries(
        OUTPUT / "binding_claims", (f"{identifier}.json",),
        "binding-recovery binding claim",
    )
    _require_exact_registry_entries(
        OUTPUT / "bindings", (f"{binding_sha256}.json",),
        "binding-recovery binding",
    )


def require_admin_after_prereg(value: str, label: str) -> None:
    frozen.require_after_t0(value, label)
    if frozen.parse_utc(value) < frozen.parse_utc(ADMIN_PREREGISTERED_UTC):
        raise ValueError(f"{label} predates binding-recovery preregistration")


def require_clean_admin_tree() -> dict[str, str]:
    git = _ORIGINAL_REQUIRE_CLEAN_TRACKED_TREE()
    head = git["head"]
    parents = frozen.git_text("rev-list", "--parents", "-n", "1", head).split()
    if parents != [head, CANDIDATE_SOURCE_COMMIT]:
        raise ValueError("binding admin commit is not the sole direct child of c807")
    changed = tuple(sorted(filter(None, frozen.git_text(
        "diff", "--name-only", CANDIDATE_SOURCE_COMMIT, head
    ).splitlines())))
    if changed != ALLOWED_CHANGED_PATHS:
        raise ValueError("binding admin commit changed-path closure mismatch")
    require_admin_after_prereg(git["author_utc"], "binding admin author time")
    require_admin_after_prereg(git["committer_utc"], "binding admin commit time")
    for path, expected in (
        (FROZEN_RECORDER, "6d26d3eb76e91abcea0074099a88533ec7a10f0ef5fee92e738e108965e00785"),
        (FROZEN_RECORDER_TEST, "98fc501cff2750468dc00cdf19749b9fa3b8b5852e2b5c8cffe96f47548154ef"),
    ):
        relative = str(path.relative_to(ROOT))
        live = path.read_bytes()
        if (sha256_bytes(live) != expected or
                frozen.git_blob(CANDIDATE_SOURCE_COMMIT, relative) != live or
                frozen.git_blob(head, relative) != live):
            raise ValueError(f"receipt-bound c807 tool drifted: {relative}")
    validate_binding_recovery_plan()
    _require_exact_registry_entries(
        CARRY_CLAIM.parent, (CARRY_CLAIM.name,),
        "carried preflight claim",
    )
    _require_exact_registry_entries(
        CARRY_RECEIPT.parent, (CARRY_RECEIPT.name,),
        "carried preflight receipt",
    )
    _require_exact_registry_entries(
        PARENT_OUTPUT / "preflight/predecessor_failures",
        (f"{PARENT_PREFLIGHT_FAILURE_SHA256}.json",),
        "carried preflight predecessor failure",
    )
    _read_canonical_exact(CARRY_CLAIM, CARRY_CLAIM_SHA256, 585)
    _read_canonical_exact(CARRY_RECEIPT, CARRY_RECEIPT_SHA256, 135815)
    require_parent_runtime_unopened()
    return git


def fixed_carried_preflight_receipt(
    dependency_identities: dict[str, dict[str, Any]],
) -> tuple[Path, dict[str, Any], str]:
    receipt = _read_canonical_exact(
        CARRY_RECEIPT, CARRY_RECEIPT_SHA256, 135815
    )
    claim = _read_canonical_exact(CARRY_CLAIM, CARRY_CLAIM_SHA256, 585)
    embedded = dict(receipt["claim"])
    if (embedded.pop("path", None) != frozen.identity_label(CARRY_CLAIM) or
            embedded != claim):
        raise ValueError("carried preflight claim embedding mismatch")
    preflight.validate_passed_receipt(
        receipt, CARRY_RECEIPT_SHA256, CANDIDATE_SOURCE_COMMIT,
        preflight.RECOVERY_PLAN_SHA256,
    )
    source = dependency_identities.get(frozen.identity_label(frozen.SOURCE_PATH))
    if receipt["source_checks"]["generated_source"] != {
            **source, "source_limit": 99_999,
    }:
        raise ValueError("carried preflight source identity mismatch")
    gate = dependency_identities.get(frozen.identity_label(frozen.GATE))
    if not preflight.exact_json_equal(receipt["comparison_gate"]["binary"], gate):
        raise ValueError("carried preflight gate identity mismatch")
    return CARRY_RECEIPT, receipt, CARRY_RECEIPT_SHA256


def qualification_key(
    plan: dict[str, Any], binding_admin_commit: str,
    dependency_identities: dict[str, dict[str, Any]],
    preflight_sha256: str, portability: dict[str, Any],
    environment: dict[str, Any], host: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": BINDING_SCHEMA,
        "campaign_id": preflight.RECOVERY_CAMPAIGN_ID,
        "binding_recovery_campaign_id": ADMIN_CAMPAIGN_ID,
        "plan_sha256": frozen.PLAN_SHA256,
        "recovery_plan_sha256": preflight.RECOVERY_PLAN_SHA256,
        "binding_recovery_plan_sha256": BINDING_RECOVERY_PLAN_SHA256,
        "prebind_blocker_sha256": PREBIND_BLOCKER_SHA256,
        "candidate_source_commit": CANDIDATE_SOURCE_COMMIT,
        "binding_admin_commit": binding_admin_commit,
        "candidate_engine_sha256": dependency_identities[
            frozen.identity_label(frozen.ENGINE_PATH)
        ]["sha256"],
        "candidate_source_sha256": dependency_identities[
            frozen.identity_label(frozen.SOURCE_PATH)
        ]["sha256"],
        "complete_dependency_sha256": sha256_bytes(
            canonical_json(dependency_identities)
        ),
        "preflight_receipt_sha256": preflight_sha256,
        "portability_evidence_sha256": portability_sha256(portability),
        "environment_sha256": environment["sha256"],
        "host_sha256": host["sha256"],
        "configuration": plan["configuration"],
        "bank_sha256": {
            stage: [item["sha256"] for item in plan["banks"][stage]]
            for stage in ("validation", "final")
        },
    }


def prepare_binding_evidence() -> dict[str, Any]:
    """Pure read-only bind prefix; it must complete before a claim exists."""
    plan = frozen.validate_plan()
    validate_binding_recovery_plan()
    git = require_clean_admin_tree()
    require_child_runtime_unopened_before_bind()
    # Verify the exact receipt-bound executable and argv before invoking ps.
    portability_before = portability_evidence()
    process = frozen.require_clean_processes()
    frozen.validate_process_preflight(process)
    environment = preflight.environment_record()
    host = preflight.host_identity()
    runtime = _binding_runtime()
    binding_recovery_plan_identity = frozen.file_identity(
        BINDING_RECOVERY_PLAN
    )
    prebind_blocker_identity = frozen.file_identity(PREBIND_BLOCKER)
    paths, routing = frozen.collect_binding_paths()
    frozen.require_tracked_head_paths(paths, git["head"])
    before = frozen.identities(paths, allow_external=True)
    receipt_path, receipt, receipt_sha256 = fixed_carried_preflight_receipt(before)
    compiler_before = {
        name: preflight.discover_compiler(name)[1]
        for name in ("clang", "gnu")
    }
    for name in ("clang", "gnu"):
        recorded = receipt["compilers"][name]
        if (not preflight.exact_json_equal(compiler_before[name], recorded["before"])
                or not preflight.exact_json_equal(
                    recorded["before"], recorded["after"]
                ) or recorded["stable"] is not True):
            raise ValueError(f"compiler changed since carried preflight: {name}")
    compiler_identities = {
        name: receipt["compilers"][name]["before"]["executable"]
        for name in ("clang", "gnu")
    }
    tool_identities = {
        name: record["executable"]
        for name, record in receipt["tool_identities_after"].items()
    }
    if (tool_identities.get("ps") != PS_IDENTITY or
            receipt["process_preflight"]["before"]["command"] != PS_ARGV or
            receipt["process_preflight"]["after"]["command"] != PS_ARGV):
        raise ValueError("carried receipt does not bind exact /bin/ps")
    runtime_identities = tuple(
        receipt["comparison_gate"]["runtime_linkage"]
        ["materialized_dependencies"].values()
    )
    bound = dict(before)
    for identity in (
        *compiler_identities.values(), *tool_identities.values(),
        *runtime_identities,
    ):
        bound[identity["path"]] = identity
    after = frozen.identities(paths, allow_external=True)
    after_bound = dict(after)
    compiler_after = {
        name: preflight.discover_compiler(name)[1]
        for name in ("clang", "gnu")
    }
    if not preflight.exact_json_equal(compiler_before, compiler_after):
        raise ValueError("compiler changed while bind evidence was frozen")
    for name in ("clang", "gnu"):
        identity = compiler_after[name]["executable"]
        after_bound[identity["path"]] = identity
    for identity in tool_identities.values():
        after_bound[identity["path"]] = frozen.file_identity(
            Path(identity["path"]), allow_external=True
        )
    for identity in runtime_identities:
        after_bound[identity["path"]] = frozen.file_identity(
            Path(identity["path"]), allow_external=True
        )
    if not preflight.exact_json_equal(bound, after_bound):
        raise ValueError("binding dependency identity changed during prefix")
    portability_after = portability_evidence()
    if not preflight.exact_json_equal(portability_before, portability_after):
        raise ValueError("portability evidence changed during binding prefix")
    routing["complete_dependency_count_with_compilers"] = len(bound)
    routing["portability_evidence_sha256"] = portability_sha256(
        portability_after
    )
    preflight_receipt_identity = frozen.file_identity(receipt_path)
    preflight_summary = _preflight_summary(receipt)
    key = qualification_key(
        plan, git["head"], bound, receipt_sha256, portability_after,
        environment, host,
    )
    identifier = frozen.candidate_qualification_id(key)
    frozen.require_no_prior_binding_attempt()
    return {
        "plan": plan, "git": git, "process": process,
        "dependency_paths": paths, "dependency_identities": bound,
        "dependency_routing": routing, "receipt_path": receipt_path,
        "receipt": receipt, "receipt_sha256": receipt_sha256,
        "compiler_records": compiler_after,
        "compiler_identities": compiler_identities,
        "tool_identities": tool_identities,
        "portability": portability_after,
        "binding_recovery_plan_identity": binding_recovery_plan_identity,
        "prebind_blocker_identity": prebind_blocker_identity,
        "preflight_receipt_identity": preflight_receipt_identity,
        "preflight_summary": preflight_summary,
        "environment": environment, "host": host, "runtime": runtime,
        "qualification_key": key,
        "candidate_qualification_id": identifier,
    }


# Configure only this process's imported frozen state-machine helpers.  The
# receipt validator owns independent path constants, so its c807 validation is
# not weakened by these administrative output/schema bindings.
_ORIGINAL_REQUIRE_CLEAN_TRACKED_TREE = frozen.require_clean_tracked_tree
_ORIGINAL_VALIDATE_STAGE_REPORT = frozen.validate_persisted_stage_report
_ORIGINAL_DECISION_PAYLOAD = frozen.decision_payload
_ORIGINAL_VALIDATE_DECISION = frozen.validate_persisted_decision

frozen.OUTPUT = OUTPUT
frozen.RECORDER = RECORDER
frozen.RECORDER_TEST = RECORDER_TEST
frozen.BIND_LOCK = BIND_LOCK
frozen.PRIVATE_LOCK = PRIVATE_LOCK
frozen.BINDING_SCHEMA = BINDING_SCHEMA
frozen.BIND_CLAIM_SCHEMA = BIND_CLAIM_SCHEMA
frozen.CLAIM_SCHEMA = STAGE_CLAIM_SCHEMA
frozen.REPORT_SCHEMA = STAGE_REPORT_SCHEMA
frozen.DECISION_SCHEMA = DECISION_SCHEMA
frozen.guard_read_path = guard_read_path
frozen.require_clean_tracked_tree = require_clean_admin_tree
frozen.TRACKED_DEPENDENCIES = (
    *frozen.TRACKED_DEPENDENCIES,
    BINDING_RECOVERY_PLAN,
    PREBIND_BLOCKER,
    CARRY_CLAIM,
    CARRY_RECEIPT,
    RECORDER,
    RECORDER_TEST,
)


def _binding_runtime() -> dict[str, Any]:
    return {
        "python_version": sys.version,
        "python_executable": frozen.file_identity(
            preflight.external_executable_path(Path(sys.executable)),
            allow_external=True,
        ),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def _preflight_summary(receipt: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": receipt["status"],
        "checks": receipt["checks"],
        "technical_recovery": receipt["technical_recovery"],
        "compilers": receipt["compilers"],
        "comparison_gate": receipt["comparison_gate"],
        "builds": receipt["builds"],
    }


def _stable_prepared_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    """Return every frozen field except the intentionally fresh process check."""
    return {
        key: value for key, value in evidence.items() if key != "process"
    }


def revalidate_prepared_evidence(
    evidence: dict[str, Any],
) -> dict[str, Any]:
    """Repeat the complete read-only prefix immediately before the claim."""
    frozen.validate_process_preflight(evidence.get("process", {}))
    final = prepare_binding_evidence()
    frozen.validate_process_preflight(final["process"])
    if _stable_prepared_evidence(evidence) != _stable_prepared_evidence(final):
        raise ValueError("binding evidence changed between complete prefixes")
    require_parent_runtime_unopened()
    require_child_runtime_unopened_before_bind()
    if (_regular_identity(PS_PATH) != PS_IDENTITY or
            preflight.process_table_command() != PS_ARGV):
        raise ValueError("exact /bin/ps changed before final process check")
    final_process = frozen.require_clean_processes()
    frozen.validate_process_preflight(final_process)
    final["process"] = final_process
    # All registry/git/dependency/process comparisons above are complete.  The
    # exact SDK and /bin/ps snapshot is deliberately the last read-only action
    # before the O_EXCL binding claim.
    final_portability = portability_evidence()
    if not preflight.exact_json_equal(
            final_portability, final["portability"]):
        raise ValueError("portability changed immediately before binding claim")
    return final


def create_binding_from_evidence(evidence: dict[str, Any]) -> tuple[Path, str]:
    """The only mutating bind suffix: one claim, then one binding receipt."""
    # The second complete prefix and final portability snapshot are followed
    # only by in-memory payload selection before the O_EXCL claim.
    evidence = revalidate_prepared_evidence(evidence)
    identifier = evidence["candidate_qualification_id"]
    claim_path, claim = frozen.create_binding_claim(
        identifier, evidence["qualification_key"],
        evidence["receipt_sha256"],
    )
    require_admin_after_prereg(claim["claimed_utc"], "binding claim")
    created_utc = frozen.utc_now()
    require_admin_after_prereg(created_utc, "binding creation")
    binding = {
        "schema": BINDING_SCHEMA,
        "status": "frozen-unopened-heldout",
        "created_utc": created_utc,
        "campaign_id": preflight.RECOVERY_CAMPAIGN_ID,
        "binding_recovery_campaign_id": ADMIN_CAMPAIGN_ID,
        "parent_campaign_id": preflight.CAMPAIGN_ID,
        "campaign_t0_utc": preflight.CAMPAIGN_T0_UTC,
        "campaign_deadline_utc": preflight.CAMPAIGN_DEADLINE_UTC,
        "plan": {
            "path": frozen.identity_label(frozen.PLAN),
            "sha256": frozen.PLAN_SHA256,
        },
        "recovery_plan": {
            "path": frozen.identity_label(frozen.RECOVERY_PLAN),
            "sha256": preflight.RECOVERY_PLAN_SHA256,
        },
        "binding_recovery_plan": evidence[
            "binding_recovery_plan_identity"
        ],
        "prebind_blocker": evidence["prebind_blocker_identity"],
        "campaign_manifest": {
            "path": frozen.identity_label(frozen.CAMPAIGN),
            "sha256": frozen.CAMPAIGN_SHA256,
        },
        "candidate_source_commit": CANDIDATE_SOURCE_COMMIT,
        "binding_admin_commit": evidence["git"]["head"],
        "candidate_qualification_id": identifier,
        "qualification_key": evidence["qualification_key"],
        "binding_claim": {
            **claim, "path": frozen.identity_label(claim_path),
        },
        "configuration": evidence["plan"]["configuration"],
        "bank_registry_from_campaign_metadata_only": evidence["plan"]["banks"],
        "dependency_identities": evidence["dependency_identities"],
        "dependency_routing": evidence["dependency_routing"],
        "portability_evidence": evidence["portability"],
        "preflight_receipt": evidence["preflight_receipt_identity"],
        "preflight_summary": evidence["preflight_summary"],
        "compiler_identities": evidence["compiler_identities"],
        "tool_identities": evidence["tool_identities"],
        "compiler_records": evidence["compiler_records"],
        "environment": evidence["environment"],
        "host": evidence["host"],
        "runtime": evidence["runtime"],
        "process_preflight": evidence["process"],
        "git_admin": evidence["git"],
        "bank_files_accessed": [],
    }
    return frozen.persist_content_addressed(OUTPUT / "bindings", binding)


def _create_binding_locked() -> tuple[Path, str]:
    return create_binding_from_evidence(prepare_binding_evidence())


def create_binding() -> tuple[Path, str]:
    frozen.ensure_directory_durable(BIND_LOCK.parent)
    with frozen.open_lock(BIND_LOCK) as bind_handle, frozen.open_lock(
            SHARED_LOCK) as shared_handle:
        try:
            fcntl.flock(bind_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(shared_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(
                "another binding/build/benchmark job owns the lock"
            ) from error
        return _create_binding_locked()


def _binding_expected_keys() -> set[str]:
    return {
        "schema", "status", "created_utc", "campaign_id",
        "binding_recovery_campaign_id", "parent_campaign_id",
        "campaign_t0_utc", "campaign_deadline_utc", "plan",
        "recovery_plan", "binding_recovery_plan", "prebind_blocker",
        "campaign_manifest", "candidate_source_commit",
        "binding_admin_commit", "candidate_qualification_id",
        "qualification_key", "binding_claim", "configuration",
        "bank_registry_from_campaign_metadata_only",
        "dependency_identities", "dependency_routing",
        "portability_evidence", "preflight_receipt", "preflight_summary",
        "compiler_identities", "tool_identities", "compiler_records",
        "environment", "host", "runtime", "process_preflight",
        "git_admin", "bank_files_accessed",
    }


def load_and_validate_binding(
    path: Path,
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, str]]:
    fixed_path = frozen.fixed_binding_path()
    if _lexical_absolute(path) != _lexical_absolute(fixed_path):
        raise ValueError("binding path is not the sole child registry entry")
    binding, binding_sha256 = frozen.load_canonical_content_addressed(
        fixed_path, BINDING_SCHEMA, OUTPUT / "bindings"
    )
    claimed_identifier = binding.get("candidate_qualification_id")
    if (not isinstance(claimed_identifier, str) or
            not re.fullmatch(r"[0-9a-f]{64}", claimed_identifier)):
        raise ValueError("binding candidate identity is malformed")
    require_exact_child_binding_registries(
        claimed_identifier, binding_sha256
    )
    plan = frozen.validate_plan()
    validate_binding_recovery_plan()
    if (set(binding) != _binding_expected_keys() or
            binding.get("status") != "frozen-unopened-heldout" or
            binding.get("campaign_id") != preflight.RECOVERY_CAMPAIGN_ID or
            binding.get("binding_recovery_campaign_id") != ADMIN_CAMPAIGN_ID or
            binding.get("parent_campaign_id") != preflight.CAMPAIGN_ID or
            binding.get("campaign_t0_utc") != preflight.CAMPAIGN_T0_UTC or
            binding.get("campaign_deadline_utc") !=
            preflight.CAMPAIGN_DEADLINE_UTC or
            binding.get("candidate_source_commit") !=
            CANDIDATE_SOURCE_COMMIT or
            binding.get("plan") != {
                "path": frozen.identity_label(frozen.PLAN),
                "sha256": frozen.PLAN_SHA256,
            } or binding.get("recovery_plan") != {
                "path": frozen.identity_label(frozen.RECOVERY_PLAN),
                "sha256": preflight.RECOVERY_PLAN_SHA256,
            } or not preflight.exact_json_equal(
                binding.get("binding_recovery_plan"),
                frozen.file_identity(BINDING_RECOVERY_PLAN),
            ) or not preflight.exact_json_equal(
                binding.get("prebind_blocker"),
                frozen.file_identity(PREBIND_BLOCKER),
            ) or binding.get("campaign_manifest") != {
                "path": frozen.identity_label(frozen.CAMPAIGN),
                "sha256": frozen.CAMPAIGN_SHA256,
            } or not preflight.exact_json_equal(
                binding.get("configuration"), plan["configuration"]
            ) or not preflight.exact_json_equal(
                binding.get("bank_registry_from_campaign_metadata_only"),
                plan["banks"],
            ) or binding.get("bank_files_accessed") != []):
        raise ValueError("binding does not match portability preregistration")
    require_admin_after_prereg(binding["created_utc"], "binding creation")
    git = require_clean_admin_tree()
    if git["head"] != binding.get("binding_admin_commit"):
        raise ValueError("binding admin commit is not current HEAD")
    frozen.validate_git_state(binding.get("git_admin", {}), git["head"])
    if not preflight.exact_json_equal(binding["git_admin"], git):
        raise ValueError("bound administrative git provenance changed")
    require_parent_runtime_unopened()
    portability_before = portability_evidence()
    if not preflight.exact_json_equal(
            portability_before, binding.get("portability_evidence")):
        raise ValueError("bound portability evidence changed")
    paths, routing = frozen.collect_binding_paths()
    frozen.require_tracked_head_paths(paths, git["head"])
    live = frozen.identities(paths, allow_external=True)
    receipt_path, receipt, receipt_sha256 = fixed_carried_preflight_receipt(live)
    compiler_records = {
        name: preflight.discover_compiler(name)[1]
        for name in ("clang", "gnu")
    }
    if not preflight.exact_json_equal(
            compiler_records, binding.get("compiler_records")):
        raise ValueError("bound compiler record changed")
    for name in ("clang", "gnu"):
        identity = compiler_records[name]["executable"]
        live[identity["path"]] = identity
    expected_tools = {
        name: record["executable"]
        for name, record in receipt["tool_identities_after"].items()
    }
    if expected_tools.get("ps") != PS_IDENTITY or not preflight.exact_json_equal(
            binding.get("tool_identities"), expected_tools):
        raise ValueError("bound fixed-tool identity changed")
    for identity in expected_tools.values():
        live[identity["path"]] = frozen.file_identity(
            Path(identity["path"]), allow_external=True
        )
    for identity in receipt["comparison_gate"]["runtime_linkage"][
            "materialized_dependencies"].values():
        live[identity["path"]] = frozen.file_identity(
            Path(identity["path"]), allow_external=True
        )
    portability_after = portability_evidence()
    if (not preflight.exact_json_equal(portability_before, portability_after) or
            not preflight.exact_json_equal(
                portability_after, binding["portability_evidence"]
            )):
        raise ValueError("portability changed while binding was loaded")
    routing["complete_dependency_count_with_compilers"] = len(live)
    routing["portability_evidence_sha256"] = portability_sha256(
        portability_after
    )
    if (not preflight.exact_json_equal(
            live, binding.get("dependency_identities")) or
            not preflight.exact_json_equal(
                routing, binding.get("dependency_routing"))):
        raise ValueError("live dependency closure differs from binding")
    environment = preflight.environment_record()
    host = preflight.host_identity()
    runtime = _binding_runtime()
    key = qualification_key(
        plan, git["head"], live, receipt_sha256, portability_after,
        environment, host,
    )
    identifier = frozen.candidate_qualification_id(key)
    if (not preflight.exact_json_equal(
            key, binding.get("qualification_key")) or
            identifier != binding.get("candidate_qualification_id")):
        raise ValueError("candidate/admin qualification identity mismatch")
    if not preflight.exact_json_equal(
            frozen.file_identity(receipt_path), binding.get("preflight_receipt")):
        raise ValueError("carried preflight receipt identity changed")
    if not preflight.exact_json_equal(
            _preflight_summary(receipt), binding.get("preflight_summary")):
        raise ValueError("carried preflight summary changed")
    for name in ("clang", "gnu"):
        if not preflight.exact_json_equal(
                binding.get("compiler_identities", {}).get(name),
                compiler_records[name]["executable"]):
            raise ValueError(f"bound compiler identity changed: {name}")
    if (not preflight.exact_json_equal(
            binding.get("environment"), environment) or
            not preflight.exact_json_equal(binding.get("host"), host) or
            not preflight.exact_json_equal(binding.get("runtime"), runtime)):
        raise ValueError("bound environment, host, or runtime changed")
    frozen.validate_process_preflight(binding.get("process_preflight", {}))
    claim = dict(binding.get("binding_claim", {}))
    claim_label = claim.pop("path", "")
    expected_claim = frozen.binding_claim_path(identifier)
    if claim_label != frozen.identity_label(expected_claim):
        raise ValueError("binding claim label mismatch")
    persisted_claim = frozen.validate_binding_claim(
        expected_claim, identifier, key, receipt_sha256
    )
    require_admin_after_prereg(
        persisted_claim["claimed_utc"], "binding claim"
    )
    if not preflight.exact_json_equal(claim, persisted_claim):
        raise ValueError("embedded binding claim differs from durable claim")
    receipt_created = frozen.parse_utc(receipt["created_utc"])
    process_checked = frozen.parse_utc(
        binding["process_preflight"]["checked_utc"]
    )
    claimed = frozen.parse_utc(persisted_claim["claimed_utc"])
    created = frozen.parse_utc(binding["created_utc"])
    if not receipt_created <= process_checked <= claimed <= created:
        raise ValueError("binding provenance timestamps are out of order")
    return binding, binding_sha256, plan, git


def _legacy_binding(binding: dict[str, Any]) -> dict[str, Any]:
    result = dict(binding)
    result["candidate_commit"] = binding["binding_admin_commit"]
    return result


def run_stage(
    binding: dict[str, Any], binding_sha256: str, plan: dict[str, Any],
    stage: str, process_preflight: dict[str, Any], git_before: dict[str, str],
) -> tuple[Path, str, dict[str, Any]]:
    identifier = binding["candidate_qualification_id"]
    if (git_before["head"] != binding["binding_admin_commit"] or
            binding["candidate_source_commit"] != CANDIDATE_SOURCE_COMMIT):
        raise ValueError("stage dual-commit provenance mismatch")
    current_git = require_clean_admin_tree()
    if not preflight.exact_json_equal(current_git, git_before):
        raise ValueError("administrative git state changed before stage claim")
    require_parent_runtime_unopened()
    require_exact_child_binding_registries(identifier, binding_sha256)
    frozen.validate_stage_claim_registry(identifier)
    if (_regular_identity(PS_PATH) != PS_IDENTITY or
            preflight.process_table_command() != PS_ARGV):
        raise ValueError("exact /bin/ps changed before stage process check")
    process_preflight = frozen.require_clean_processes()
    frozen.validate_process_preflight(process_preflight)
    # This exact alias/process-tool snapshot is the final read-only boundary
    # before the durable stage claim and any stage bank path construction.
    portability_before = portability_evidence()
    if not preflight.exact_json_equal(
            portability_before, binding["portability_evidence"]):
        raise ValueError("portability changed before atomic stage claim")
    claim, claim_payload = frozen.create_stage_claim(
        identifier, stage, binding_sha256
    )
    require_admin_after_prereg(claim_payload["claimed_utc"], f"{stage} claim")
    # Only after the durable claim may this stage's bank paths be constructed.
    bank_paths = [ROOT / item["path"] for item in frozen.stage_records(plan, stage)]
    dependency_paths: tuple[Path, ...] = ()
    command = frozen.command_for_stage(plan, stage)
    started_utc = frozen.utc_now()
    started_ns = time.monotonic_ns()
    before: dict[str, dict[str, Any]] = {}
    after: dict[str, dict[str, Any]] = {}
    parsed: dict[str, Any] = {}
    validation_codes: list[str] = []
    threshold_errors: list[str] = []
    process_result = {
        "returncode": None, "stdout": "", "stderr": "",
        "timed_out": False, "os_error": None,
    }

    def reject(code: str) -> None:
        if code not in validation_codes:
            validation_codes.append(code)

    try:
        bound_paths = [
            frozen._path_from_label(
                label, binding["dependency_identities"], allow_external=True
            )
            for label in binding["dependency_identities"]
        ]
        dependency_paths = (*bound_paths, *bank_paths)
        before = frozen.identities(
            dependency_paths, allowed_stage_banks=bank_paths,
            allow_external=True,
        )
        frozen.validate_stage_bank_identities(plan, stage, before)
    except (OSError, ValueError):
        before = {}
        reject("input_identity_before")
    try:
        compiler_before = {
            name: preflight.discover_compiler(name)[1]
            for name in ("clang", "gnu")
        }
    except (OSError, ValueError, subprocess.SubprocessError):
        compiler_before = {}
        reject("compiler_changed")
    try:
        host_before = preflight.host_identity()
    except (OSError, ValueError, subprocess.SubprocessError):
        host_before = {}
        reject("host_changed")
    if not validation_codes:
        process_result = frozen._run_process(
            command, frozen.STAGE_TIMEOUT_SECONDS[stage]
        )
        if (process_result["returncode"] != 0 or
                process_result["timed_out"] or
                process_result["os_error"] is not None or
                process_result["stderr"] != ""):
            reject("process_execution")
    else:
        reject("process_execution")
    try:
        if before and dependency_paths:
            after = frozen.identities(
                dependency_paths, allowed_stage_banks=bank_paths,
                allow_external=True,
            )
            frozen.validate_stage_bank_identities(plan, stage, after)
        else:
            raise ValueError("before-input identity unavailable")
    except (OSError, ValueError):
        after = {}
        reject("input_identity_after")
    try:
        if process_result["stdout"]:
            parsed = frozen.validate_stage_stdout(
                plan, stage, process_result["stdout"]
            )
            threshold_errors = frozen.stage_threshold_errors(
                stage, parsed["aggregate"]
            )
        else:
            raise ValueError("empty stdout")
    except (ValueError, OverflowError):
        reject("stdout_contract")
        parsed = {}
        threshold_errors = []
    try:
        compiler_after = {
            name: preflight.discover_compiler(name)[1]
            for name in ("clang", "gnu")
        }
    except (OSError, ValueError, subprocess.SubprocessError):
        compiler_after = {}
    if (not preflight.exact_json_equal(compiler_before, compiler_after) or
            not preflight.exact_json_equal(
                compiler_after, binding["compiler_records"]
            )):
        reject("compiler_changed")
        compiler_before = {}
        compiler_after = {}
    try:
        host_after = preflight.host_identity()
    except (OSError, ValueError, subprocess.SubprocessError):
        host_after = {}
    if (not preflight.exact_json_equal(host_before, host_after) or
            not preflight.exact_json_equal(host_after, binding["host"])):
        reject("host_changed")
        host_before = {}
        host_after = {}
    try:
        git_after = require_clean_admin_tree()
    except (OSError, ValueError, subprocess.SubprocessError):
        reject("tracked_state_after")
        git_after = None
    portability_after = portability_evidence()
    stable_portability = (
        preflight.exact_json_equal(
            portability_before, binding["portability_evidence"]
        ) and preflight.exact_json_equal(
            portability_before, portability_after
        ) and preflight.exact_json_equal(
            portability_after, binding["portability_evidence"]
        )
    )
    if not stable_portability:
        # A post-command portability drift leaves the stage claim spent and no
        # reusable report; this is intentionally stronger than a rejected run.
        raise ValueError("portability changed after stage command")
    ended_ns = time.monotonic_ns()
    ended_utc = frozen.utc_now()
    stable_inputs = bool(before) and preflight.exact_json_equal(before, after)
    acceptable = (
        process_result["returncode"] == 0 and
        process_result["timed_out"] is False and
        process_result["os_error"] is None and
        process_result["stderr"] == "" and stable_inputs and
        isinstance(git_after, dict) and
        git_after["head"] == binding["binding_admin_commit"] and
        not validation_codes and not threshold_errors and bool(parsed) and
        preflight.exact_json_equal(compiler_before, compiler_after) and
        preflight.exact_json_equal(
            compiler_after, binding["compiler_records"]
        ) and preflight.exact_json_equal(host_before, host_after) and
        preflight.exact_json_equal(host_after, binding["host"]) and
        stable_portability
    )
    report = {
        "schema": STAGE_REPORT_SCHEMA,
        "campaign_id": preflight.RECOVERY_CAMPAIGN_ID,
        "binding_recovery_campaign_id": ADMIN_CAMPAIGN_ID,
        "campaign_t0_utc": preflight.CAMPAIGN_T0_UTC,
        "classification": f"untouched-{stage}-one-shot-qualification-stage",
        "final_qualification": False,
        "producer": binding["dependency_identities"][
            frozen.identity_label(RECORDER)
        ],
        "candidate_source_commit": CANDIDATE_SOURCE_COMMIT,
        "binding_admin_commit": binding["binding_admin_commit"],
        "candidate_qualification_id": identifier,
        "binding_sha256": binding_sha256,
        "stage": stage,
        "claim": {**claim_payload, "path": frozen.identity_label(claim)},
        "started_utc": started_utc,
        "ended_utc": ended_utc,
        "elapsed_monotonic_ns": ended_ns - started_ns,
        "command_argv": command,
        "command_shell": shlex.join(command),
        "cwd": str(ROOT),
        "timeout_seconds": frozen.STAGE_TIMEOUT_SECONDS[stage],
        "environment": preflight.environment_record(),
        "host_before": host_before,
        "host_after": host_after,
        "runtime": binding["runtime"],
        "returncode": process_result["returncode"],
        "timed_out": process_result["timed_out"],
        "os_error_class": process_result["os_error"],
        "stdout": frozen._stream_evidence(process_result["stdout"]),
        "stderr": frozen._stream_evidence(process_result["stderr"]),
        "process_preflight": process_preflight,
        "git_before": git_before,
        "git_after": git_after,
        "inputs_before": before,
        "inputs_after": after,
        "stable_inputs": stable_inputs,
        "compiler_records_before": compiler_before,
        "compiler_records_after": compiler_after,
        "stable_compilers": compiler_before == compiler_after ==
        binding["compiler_records"],
        "portability_before": portability_before,
        "portability_after": portability_after,
        "stable_portability": stable_portability,
        "accessed_bank_paths": [
            item["path"] for item in frozen.stage_records(plan, stage)
        ],
        "parsed": parsed,
        "validation_codes": validation_codes,
        "threshold_errors": threshold_errors,
        "stage_acceptable": acceptable,
        "replay_corrections": "disabled",
        "transcripts": "not-retained",
    }
    path, digest = frozen.persist_content_addressed(
        OUTPUT / "reports" / stage, report
    )
    persisted, _ = frozen.load_canonical_content_addressed(
        path, STAGE_REPORT_SCHEMA, OUTPUT / "reports" / stage
    )
    if not preflight.exact_json_equal(persisted, report):
        raise OSError("stage report semantic readback failed")
    return path, digest, report


def validate_persisted_stage_report(
    report: dict[str, Any], binding: dict[str, Any], binding_sha256: str,
    plan: dict[str, Any], stage: str,
) -> None:
    added = {
        "candidate_source_commit", "binding_admin_commit",
        "binding_recovery_campaign_id", "portability_before",
        "portability_after", "stable_portability",
    }
    if (not added.issubset(report) or
            report.get("binding_recovery_campaign_id") != ADMIN_CAMPAIGN_ID or
            report.get("candidate_source_commit") != CANDIDATE_SOURCE_COMMIT or
            report.get("binding_admin_commit") !=
            binding["binding_admin_commit"] or
            not preflight.exact_json_equal(
                report.get("portability_before"),
                binding["portability_evidence"],
            ) or not preflight.exact_json_equal(
                report.get("portability_after"),
                binding["portability_evidence"],
            ) or report.get("stable_portability") is not True):
        raise ValueError("persisted stage portability/provenance mismatch")
    live = portability_evidence()
    if not preflight.exact_json_equal(live, binding["portability_evidence"]):
        raise ValueError("live portability differs from persisted stage")
    require_admin_after_prereg(report["started_utc"], f"{stage} start")
    require_admin_after_prereg(report["ended_utc"], f"{stage} end")
    require_admin_after_prereg(
        report["claim"]["claimed_utc"], f"{stage} claim"
    )
    projected = {key: value for key, value in report.items() if key not in added}
    _ORIGINAL_VALIDATE_STAGE_REPORT(
        projected, _legacy_binding(binding), binding_sha256, plan, stage
    )
    if report.get("stage_acceptable") is not True and not report.get(
            "validation_codes") and not report.get("threshold_errors"):
        raise ValueError("stable-portability report has unexplained rejection")


def decision_payload(
    binding: dict[str, Any], binding_sha256: str,
    validation: tuple[Path, str, dict[str, Any]],
    final: tuple[Path, str, dict[str, Any]] | None,
    created_utc: str,
) -> dict[str, Any]:
    payload = _ORIGINAL_DECISION_PAYLOAD(
        _legacy_binding(binding), binding_sha256, validation, final, created_utc
    )
    payload["candidate_source_commit"] = CANDIDATE_SOURCE_COMMIT
    payload["binding_admin_commit"] = binding["binding_admin_commit"]
    payload["binding_recovery_campaign_id"] = ADMIN_CAMPAIGN_ID
    return payload


def validate_persisted_decision(
    decision: dict[str, Any], binding: dict[str, Any],
    binding_sha256: str, plan: dict[str, Any],
) -> None:
    if (decision.get("candidate_source_commit") != CANDIDATE_SOURCE_COMMIT or
            decision.get("binding_admin_commit") !=
            binding["binding_admin_commit"] or
            decision.get("binding_recovery_campaign_id") !=
            ADMIN_CAMPAIGN_ID):
        raise ValueError("persisted decision dual provenance mismatch")
    require_admin_after_prereg(decision["created_utc"], "held-out decision")
    projected = dict(decision)
    projected.pop("candidate_source_commit", None)
    projected.pop("binding_admin_commit", None)
    projected.pop("binding_recovery_campaign_id", None)
    # The frozen validator recomputes the entire decision and recursively calls
    # our report validator.  Temporarily expose only its own pure payload helper
    # inside this private module instance.
    current = frozen.decision_payload
    frozen.decision_payload = _ORIGINAL_DECISION_PAYLOAD
    try:
        _ORIGINAL_VALIDATE_DECISION(
            projected, _legacy_binding(binding), binding_sha256, plan
        )
    finally:
        frozen.decision_payload = current


frozen.load_and_validate_binding = load_and_validate_binding
frozen.run_stage = run_stage
frozen.validate_persisted_stage_report = validate_persisted_stage_report
frozen.decision_payload = decision_payload
frozen.validate_persisted_decision = validate_persisted_decision


def run_qualification() -> tuple[Path, str, dict[str, Any]]:
    return frozen.run_qualification()


def main() -> int:
    parser = argparse.ArgumentParser()
    actions = parser.add_subparsers(dest="action", required=True)
    actions.add_parser("bind")
    actions.add_parser("run")
    arguments = parser.parse_args()
    try:
        if arguments.action == "bind":
            path, digest = create_binding()
            acceptable = True
        else:
            path, digest, decision = run_qualification()
            acceptable = bool(decision["heldout_qualification_acceptable"])
    except (
        KeyError, OSError, TypeError, UnicodeError, ValueError,
        json.JSONDecodeError, subprocess.SubprocessError,
    ) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(path.relative_to(ROOT))
    print(f"sha256={digest}")
    if arguments.action == "run":
        print(f"heldout_qualification_acceptable={str(acceptable).lower()}")
    return 0 if acceptable else 1


if __name__ == "__main__":
    raise SystemExit(main())
