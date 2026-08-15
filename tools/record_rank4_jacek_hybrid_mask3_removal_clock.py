#!/usr/bin/env python3
"""One-shot DEVELOPMENT-only recorder for the preregistered mask-3 removal.

The producer never builds a target and never names or opens a protected bank.
Each DEVELOPMENT gate is claimed with O_EXCL before the gate receives its bank
arguments.  An execution receipt is persisted as soon as the gate returns so
that report/decision finalization can resume without running another game.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
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

# Audit and import must not create or refresh bytecode cache files.
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import record_rank4_jacek_hybrid_full_development_clock as old_full  # noqa: E402


common = old_full.common
OUTPUT = ROOT / "results/rank_4_jacek_hybrid/gates/mask3_removal_clock"
PLAN = OUTPUT / "PLAN.json"
RECORDER = ROOT / "tools/record_rank4_jacek_hybrid_mask3_removal_clock.py"
RECORDER_TEST = (
    ROOT / "tests/codingame/test_rank4_jacek_hybrid_mask3_removal_clock.py"
)
LOCK = OUTPUT / ".recorder.lock"
GLOBAL_BENCHMARK_LOCK = Path("/tmp/rank4-hybrid-prototype-benchmark.lock")
LEGACY_BENCHMARK_LOCKS = (
    ROOT / "build/rank4-jacek-hybrid-full-development-clock.lock",
    ROOT / "build/rank4-jacek-hybrid-null-fastpath-clock.lock",
    ROOT / "build/rank4-jacek-hybrid-proof-scope-clock.lock",
    ROOT / "build/rank4-jacek-hybrid-sole-legal-edge-clock.lock",
)
CAMPAIGN_LOCKS = tuple(sorted(
    (LOCK, GLOBAL_BENCHMARK_LOCK, *LEGACY_BENCHMARK_LOCKS), key=str
))
CLAIMS = OUTPUT / "claims"
EXECUTIONS = OUTPUT / "executions"
REPORTS = OUTPUT / "reports"
DECISIONS = OUTPUT / "decisions"

PLAN_SHA256 = "264437fed831357956d3a4e798957a297924d88d928a42f38b03eb944ae19f36"
CAMPAIGN_ID = "rank_4_jacek_hybrid-mask3-removal-development-20260814-v1"
CAMPAIGN_T0_UTC = "2026-08-14T12:13:43.147626Z"
CAMPAIGN_DEADLINE_UTC = "2026-08-15T07:15:07Z"
CANDIDATE_SOURCE_COMMIT = "c8077067cafff2e0fed8b4c85082de0392fc453c"
ADMIN_PARENT_COMMIT = "da667ae4e3563e0cefdf1fbe1c8246fc60318aec"
ADMIN_CHANGED_PATHS = (
    "results/rank_4_jacek_hybrid/gates/mask3_removal_clock/PLAN.json",
    "tests/codingame/test_rank4_jacek_hybrid_mask3_removal_clock.py",
    "tools/record_rank4_jacek_hybrid_mask3_removal_clock.py",
)
PRODUCER_BOOTSTRAP_IDENTITIES = {
    "tools/record_rank4_jacek_hybrid_full_development_clock.py": {
        "bytes": 34_337,
        "mode": "0644",
        "sha256": "58e72685151f86009b5a682c49363d3dc3ae11a151d15b88906130c4505f251e",
    },
    "tools/record_rank4_jacek_hybrid_proof_scope_clock.py": {
        "bytes": 23_804,
        "mode": "0644",
        "sha256": "1c9a2a505578b02866aa3c2d64231e048ea46a301d910bb04aef345552bf9aca",
    },
}

GATE = (
    ROOT / "build/rank4-jacek-hybrid-heldout-preflight-recovery-v1/"
    "clang-release/papersoccer_codingame_rank_4_jacek_hybrid_comparison_gate"
)
GATE_SHA256 = "733a02c0e24abbb12e17518411abd1084c079a3a93fdc96413c0885cd37f7d1b"
GATE_BYTES = 321_464
BUILD_ROOT = GATE.parent
TARGET_DIRECTORY = (
    BUILD_ROOT / "CMakeFiles/"
    "papersoccer_codingame_rank_4_jacek_hybrid_comparison_gate.dir"
)
DEPFILES = (
    TARGET_DIRECTORY /
    "submissions/codingame/bots/rank_4_jacek_hybrid/comparison_gate.cpp.o.d",
    TARGET_DIRECTORY /
    "submissions/codingame/bots/rank_4_jacek_hybrid/"
    "comparison_gate_hybrid.cpp.o.d",
    TARGET_DIRECTORY /
    "submissions/codingame/bots/rank_4_jacek_hybrid/"
    "comparison_gate_rank4.cpp.o.d",
)
OBJECTS = tuple(Path(str(path)[:-2]) for path in DEPFILES)
DIRECT_BUILD_ARTIFACTS = (
    BUILD_ROOT / "CMakeCache.txt",
    TARGET_DIRECTORY / "flags.make",
    TARGET_DIRECTORY / "link.txt",
    GATE,
    *OBJECTS,
    *DEPFILES,
)
EXPECTED_CLOSURE_ENTRIES = 1_056
EXPECTED_CLOSURE_REPO_ENTRIES = 27
EXPECTED_CLOSURE_EXTERNAL_ENTRIES = 1_029
EXPECTED_CLOSURE_SHA256 = (
    "dddbe79dfebaf118e618bf20f4f5bc1737b89fa6b524800065d35b577cc061f3"
)

PREFLIGHT_RECEIPT = (
    ROOT / "results/rank_4_jacek_hybrid/gates/heldout_qualification/"
    "recovery_v1/preflight/receipts/"
    "de8d2d40076a38d2f973c0e665470f60af69354a7842648267a3ca794e5ce0d3.json"
)
PREFLIGHT_RECEIPT_SHA256 = (
    "de8d2d40076a38d2f973c0e665470f60af69354a7842648267a3ca794e5ce0d3"
)
FROZEN_BINDING = (
    ROOT / "results/rank_4_jacek_hybrid/gates/heldout_qualification/"
    "binding_recovery_v1/bindings/"
    "f537410b66598374619c30f431cf234aed55536ff1b55ed883742b56c759102d.json"
)
FROZEN_BINDING_SHA256 = (
    "f537410b66598374619c30f431cf234aed55536ff1b55ed883742b56c759102d"
)

HISTORICAL_JSON = {
    "proof_scope_matrix": (
        ROOT / "results/rank_4_jacek_hybrid/gates/proof_scope_clock/matrix/"
        "739eaf7d4e2fa9f218e759e309e042271f75a84cf4f5216fbef07eec7a525454.json",
        "739eaf7d4e2fa9f218e759e309e042271f75a84cf4f5216fbef07eec7a525454",
    ),
    "mask3_leaf_report": (
        ROOT / "results/rank_4_jacek_hybrid/gates/proof_scope_clock/"
        "528f94bcfa1294c289e388321c54bd5070711e889cd3f5e9ef107b830f3d8d3b.json",
        "528f94bcfa1294c289e388321c54bd5070711e889cd3f5e9ef107b830f3d8d3b",
    ),
    "mask7_ply1_report": (
        ROOT / "results/rank_4_jacek_hybrid/gates/proof_scope_clock/"
        "338723d65c028d4ea2c46dfd91f7a32a5f9d78bd3fdd1f65a972d8f81ff89c74.json",
        "338723d65c028d4ea2c46dfd91f7a32a5f9d78bd3fdd1f65a972d8f81ff89c74",
    ),
    "full_development_selection": (
        ROOT / "results/rank_4_jacek_hybrid/gates/full_development_clock/"
        "selection/1b6736186006b6820021dc0315faab50dcba97db719ea5bbfe6768a7e2a243d3.json",
        "1b6736186006b6820021dc0315faab50dcba97db719ea5bbfe6768a7e2a243d3",
    ),
    "closed_heldout_decision": (
        ROOT / "results/rank_4_jacek_hybrid/gates/heldout_qualification/"
        "binding_recovery_v1/decisions/"
        "9c12b44cc2ffa475e55e1e166c637f725e8107736677c432b03ea31ef376997f.json",
        "9c12b44cc2ffa475e55e1e166c637f725e8107736677c432b03ea31ef376997f",
    ),
}
FULL_DEVELOPMENT_PLAN = ROOT / "results/rank_4_jacek_hybrid/FULL_DEVELOPMENT_GATE_PLAN.md"
FULL_DEVELOPMENT_PLAN_SHA256 = (
    "50acd3d31df69579e0d6c3d68a71f20c4964f2413523d754be798f607d558438"
)

BANKS = (
    ("development_d04.tsv", 4, "18128950407139886133",
     "984fbb78d85d7f9806c77e675b9b22a9b047bd15311f510ab0cedcd9a63244dc", 78),
    ("development_d08.tsv", 8, "9297997631523997120",
     "6dbec157e7094f07796a9aa1ac97b43919930377ec315c761eaace216630259e", 76),
    ("development_d12.tsv", 12, "11025886481058993262",
     "d30d087020e4946ce77b6d6e578484d583f0cf25f2ffa90a1918f8d9a9a8a11a", 76),
    ("development_d20.tsv", 20, "4624785204876369057",
     "2aa4b635dcaf23b2587b22fdb7558f4c8d6b4dd5a33e3fec2c164931b3fcd8d4", 76),
)
BANK_DIRECTORY = ROOT / "results/rank_4_jacek_hybrid/openings"
BANK_PATHS = tuple(BANK_DIRECTORY / item[0] for item in BANKS)

STAGES = (
    {
        "stage": "remove_ply1_against_mask7",
        "reference_engine": "hybrid-control",
        "candidate_exact_proof_mask": 3,
        "reference_exact_proof_mask": 7,
    },
    {
        "stage": "absolute_against_mask0",
        "reference_engine": "hybrid-control",
        "candidate_exact_proof_mask": 3,
        "reference_exact_proof_mask": 0,
    },
    {
        "stage": "incumbent_against_rank4",
        "reference_engine": "rank4",
        "candidate_exact_proof_mask": 3,
        "reference_exact_proof_mask": 0,
    },
)
STAGE_NAMES = tuple(item["stage"] for item in STAGES)
RUN_TIMEOUT_SECONDS = 3_600
EXPECTED_SUMMARY_FIELDS = frozenset({
    "bank", "candidate_attempted_depth_avg", "candidate_attempted_depth_max",
    "candidate_depth_avg", "candidate_depth_max", "candidate_exceptions",
    "candidate_exhaustions", "candidate_first_ms_max", "candidate_first_ms_p99",
    "candidate_hard_timeouts", "candidate_illegal", "candidate_invocations",
    "candidate_later_ms_max", "candidate_later_ms_p99", "candidate_nodes",
    "candidate_nodes_avg", "candidate_nodes_max", "candidate_nodes_p99",
    "candidate_operational", "candidate_p0", "candidate_p1",
    "candidate_proof_leaf", "candidate_proof_ply1", "candidate_proof_ply2",
    "candidate_proof_rebound", "candidate_proof_root", "candidate_searches",
    "candidate_soft_overruns", "candidate_wins", "failed",
    "games", "reference_attempted_depth_avg", "reference_attempted_depth_max",
    "reference_depth_avg", "reference_depth_max", "reference_exceptions",
    "reference_exhaustions", "reference_first_ms_max", "reference_first_ms_p99",
    "reference_hard_timeouts", "reference_illegal", "reference_invocations",
    "reference_later_ms_max", "reference_later_ms_p99", "reference_nodes",
    "reference_nodes_avg", "reference_nodes_max", "reference_nodes_p99",
    "reference_operational", "reference_proof_leaf", "reference_proof_ply1",
    "reference_proof_ply2", "reference_proof_rebound", "reference_proof_root",
    "reference_searches", "reference_soft_overruns", "reference_wins", "unfinished",
})
EXPECTED_CONFIGURATION_FIELDS = frozenset({
    "bank_count", "bank_validation", "candidate_clock",
    "candidate_exact_proof_mask", "candidate_nodes", "expected_depths",
    "expected_role", "expected_seeds", "expected_sha256", "max_turns",
    "openings", "operational_clock", "profile", "reference_clock",
    "reference_engine", "reference_exact_proof_mask", "reference_nodes",
    "replay_corrections", "transcripts",
})

PLAN_SCHEMA = "rank4-jacek-hybrid-mask3-removal-development-plan-v1"
CLAIM_SCHEMA = "rank4-jacek-hybrid-mask3-removal-claim-v1"
EXECUTION_SCHEMA = "rank4-jacek-hybrid-mask3-removal-execution-v1"
REPORT_SCHEMA = "rank4-jacek-hybrid-mask3-removal-report-v1"
DECISION_SCHEMA = "rank4-jacek-hybrid-mask3-removal-decision-v1"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
ALLOWED_EXTERNAL_ROOTS = tuple(
    Path(value) for value in ("/Applications", "/Library", "/System", "/lib",
                              "/lib64", "/opt", "/usr")
)
PS = Path("/bin/ps")
PS_SHA256 = "472992c470606d28f577590decfecd7f4a20f832fd92c671bebc6d44790b5d02"
PS_BYTES = 170_816
SYSCTL = Path("/usr/sbin/sysctl")
SYSCTL_SHA256 = "7cf2165d121db5e22e0516acc26981a526402b3a8cd682ed5e35b337d8162707"
OTOOL = Path("/usr/bin/otool")
OTOOL_SHA256 = "179301dcb41ea78accc3fa0048a7e6f6710d891945a751a34addd622020c1818"
OTOOL_BYTES = 118_928
LINKAGE_NAMES = ("/usr/lib/libSystem.B.dylib", "/usr/lib/libc++.1.dylib")
LINKAGE_SHA256 = "06ee16472b7f357d9f63b75f8c37a74df1743fddfbe347b4cc5c14b3efb18a9c"


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("ascii") + b"\n"


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp lacks timezone")
    return parsed.astimezone(timezone.utc)


def _object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def decode_json(raw: bytes) -> Any:
    try:
        return json.loads(raw, object_pairs_hook=_object_no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid JSON evidence") from error


def _inode_key(metadata: os.stat_result) -> tuple[int, int]:
    return metadata.st_dev, metadata.st_ino


def _read_regular_nofollow(
    path: Path, *, expected_inode: tuple[int, int] | None = None,
    required_mode: int | None = None, require_single_link: bool = True,
    kind: str = "input",
) -> tuple[bytes, os.stat_result]:
    """Read one stable regular inode without following the pathname."""
    path_before = os.lstat(path)
    if (not stat.S_ISREG(path_before.st_mode) or
            (require_single_link and path_before.st_nlink != 1) or
            (required_mode is not None and
             stat.S_IMODE(path_before.st_mode) != required_mode) or
            (expected_inode is not None and
             _inode_key(path_before) != expected_inode)):
        raise ValueError(f"{kind} is not a stable regular file: {path}")
    pinned_inode = _inode_key(path_before)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)

        def validate(metadata: os.stat_result) -> None:
            if (not stat.S_ISREG(metadata.st_mode) or
                    (require_single_link and metadata.st_nlink != 1) or
                    (required_mode is not None and
                     stat.S_IMODE(metadata.st_mode) != required_mode) or
                    _inode_key(metadata) != pinned_inode):
                raise ValueError(f"{kind} is not a stable regular file: {path}")

        validate(before)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        if (not stat.S_ISREG(after.st_mode) or
                _inode_key(after) != _inode_key(before) or
                after.st_size != before.st_size or
                after.st_mode != before.st_mode or
                after.st_nlink != before.st_nlink or
                len(raw) != after.st_size):
            raise ValueError(f"{kind} changed during descriptor read: {path}")
        validate(after)
        path_after = os.lstat(path)
        if (not stat.S_ISREG(path_after.st_mode) or
                _inode_key(path_after) != _inode_key(before) or
                path_after.st_size != before.st_size or
                path_after.st_mode != before.st_mode or
                path_after.st_nlink != before.st_nlink):
            raise ValueError(f"{kind} path changed during descriptor read: {path}")
        return raw, after
    finally:
        os.close(descriptor)


def read_regular_file(
    path: Path, *, expected_inode: tuple[int, int] | None = None,
) -> bytes:
    return _read_regular_nofollow(path, expected_inode=expected_inode)[0]


def path_label(path: Path) -> str:
    resolved = path.resolve(strict=True)
    try:
        return resolved.relative_to(ROOT.resolve(strict=True)).as_posix()
    except ValueError:
        return str(resolved)


def _file_identity_with_bytes(
    path: Path, *, label: str | None = None,
    allow_sealed_os_multilink: bool = False,
    required_mode: int | None = None,
) -> tuple[bytes, dict[str, Any]]:
    if allow_sealed_os_multilink and path != OTOOL:
        raise ValueError("multi-link exemption is limited to sealed /usr/bin/otool")
    raw, metadata = _read_regular_nofollow(
        path, require_single_link=not allow_sealed_os_multilink,
        required_mode=required_mode,
    )
    try:
        raw.decode("ascii")
        ascii_only = True
    except UnicodeDecodeError:
        ascii_only = False
    identity = {
        "ascii": ascii_only,
        "bytes": len(raw),
        "executable": bool(stat.S_IMODE(metadata.st_mode) & 0o111),
        "mode": f"{stat.S_IMODE(metadata.st_mode):04o}",
        "path": label if label is not None else path_label(path),
        "sha256": sha256_bytes(raw),
    }
    return raw, identity


def file_identity(
    path: Path, *, label: str | None = None,
    allow_sealed_os_multilink: bool = False,
    required_mode: int | None = None,
) -> dict[str, Any]:
    return _file_identity_with_bytes(
        path, label=label,
        allow_sealed_os_multilink=allow_sealed_os_multilink,
        required_mode=required_mode,
    )[1]


def load_exact_bytes(path: Path, digest: str) -> bytes:
    raw = read_regular_file(path)
    if sha256_bytes(raw) != digest:
        raise ValueError(f"evidence SHA-256 mismatch: {path}")
    return raw


def load_canonical_json(path: Path, digest: str) -> dict[str, Any]:
    raw = load_exact_bytes(path, digest)
    payload = decode_json(raw)
    if not isinstance(payload, dict) or canonical_json(payload) != raw:
        raise ValueError(f"evidence is not exact canonical JSON: {path}")
    return payload


def load_plan() -> dict[str, Any]:
    plan = load_canonical_json(PLAN, PLAN_SHA256)
    if (plan.get("schema") != PLAN_SCHEMA or
            plan.get("campaign_id") != CAMPAIGN_ID or
            plan.get("created_utc") != CAMPAIGN_T0_UTC or
            plan.get("campaign_deadline_utc") != CAMPAIGN_DEADLINE_UTC or
            plan.get("classification") !=
            "development-only-mask3-removal-ablation-not-heldout-qualification"):
        raise ValueError("mask3-removal plan provenance mismatch")
    policy = plan.get("admin_commit_policy", {})
    if (policy.get("candidate_source_commit") != CANDIDATE_SOURCE_COMMIT or
            policy.get("direct_parent") != ADMIN_PARENT_COMMIT or
            tuple(policy.get("exact_changed_paths", ())) != ADMIN_CHANGED_PATHS):
        raise ValueError("plan commit policy mismatch")
    plan_stages = plan.get("stages", ())
    if tuple(item.get("stage") for item in plan_stages) != STAGE_NAMES:
        raise ValueError("plan stage order mismatch")
    projected_stages = [
        {key: item.get(key) for key in (
            "stage", "reference_engine", "candidate_exact_proof_mask",
            "reference_exact_proof_mask",
        )}
        for item in plan_stages
    ]
    if projected_stages != list(STAGES):
        raise ValueError("plan stage engine/mask contract mismatch")
    if plan.get("banks") != bank_registry():
        raise ValueError("plan DEVELOPMENT bank metadata mismatch")
    if plan.get("producer_bootstrap_identities") != PRODUCER_BOOTSTRAP_IDENTITIES:
        raise ValueError("plan producer-bootstrap identity registry mismatch")
    environment = gate_environment()
    if plan.get("execution_environment") != {
            "values": environment,
            "sha256": sha256_bytes(canonical_json(environment)),
            }:
        raise ValueError("plan execution environment mismatch")
    thresholds = plan.get("thresholds_each_stage", {})
    if (thresholds.get("exact_games") != 306 or
            thresholds.get("exact_games_by_physical_color") != [153, 153] or
            thresholds.get("candidate_wins_min") != 160 or
            thresholds.get("candidate_wins_by_physical_color_min") != [77, 77]):
        raise ValueError("plan threshold mismatch")
    return plan


def bank_registry() -> list[dict[str, Any]]:
    bytes_by_name = (10_836, 11_185, 11_888, 13_150)
    return [
        {
            "bytes": bytes_by_name[index],
            "depth": item[1],
            "games": item[4],
            "path": f"results/rank_4_jacek_hybrid/openings/{item[0]}",
            "physical_color_games": item[4] // 2,
            "role": "development",
            "seed": item[2],
            "sha256": item[3],
        }
        for index, item in enumerate(BANKS)
    ]


def git_run(*arguments: str, binary: bool = False) -> bytes | str:
    completed = subprocess.run(
        ["/usr/bin/git", *arguments], cwd=ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=not binary, check=False,
        env={
            "PATH": "/usr/bin:/bin", "LC_ALL": "C",
            "GIT_OPTIONAL_LOCKS": "0",
        },
    )
    if completed.returncode != 0 or completed.stderr:
        raise ValueError(f"git {' '.join(arguments)} failed")
    return completed.stdout if binary else completed.stdout.strip()


def require_git_regular_blob(commit: str, label: str) -> None:
    fields = str(git_run("ls-tree", commit, "--", label)).split(None, 3)
    if (len(fields) != 4 or fields[0] != "100644" or fields[1] != "blob" or
            fields[3] != label):
        raise ValueError(f"Git tree mode/type mismatch: {commit}:{label}")


def require_index_regular_blob(label: str) -> None:
    fields = str(git_run("ls-files", "--stage", "--", label)).split(None, 3)
    if (len(fields) != 4 or fields[0] != "100644" or fields[2] != "0" or
            fields[3] != label):
        raise ValueError(f"Git index mode/type mismatch: {label}")


def load_tracked_exact_bytes(
    path: Path, digest: str, *, live_mode: int = 0o644,
) -> bytes:
    label = path.relative_to(ROOT).as_posix()
    raw, identity = _file_identity_with_bytes(
        path, label=label, required_mode=live_mode
    )
    require_git_regular_blob("HEAD", label)
    require_index_regular_blob(label)
    committed = git_run("show", f"HEAD:{label}", binary=True)
    if (identity["mode"] != f"{live_mode:04o}" or
            identity["sha256"] != digest or
            not isinstance(committed, bytes) or
            sha256_bytes(committed) != digest or committed != raw):
        raise ValueError(f"tracked evidence differs from exact HEAD blob: {label}")
    return raw


def load_tracked_canonical_json(
    path: Path, digest: str, *, live_mode: int = 0o644,
) -> dict[str, Any]:
    raw = load_tracked_exact_bytes(path, digest, live_mode=live_mode)
    payload = decode_json(raw)
    if not isinstance(payload, dict) or canonical_json(payload) != raw:
        raise ValueError(f"tracked evidence is not exact canonical JSON: {path}")
    return payload


def bounded_git_paths(plan: dict[str, Any]) -> tuple[str, ...]:
    source_paths = plan.get("source_identities")
    bootstrap = plan.get("producer_bootstrap_identities")
    if not isinstance(source_paths, dict) or not isinstance(bootstrap, dict):
        raise ValueError("bounded Git path registry is malformed")
    paths = {
        *ADMIN_CHANGED_PATHS,
        *source_paths,
        *bootstrap,
        FULL_DEVELOPMENT_PLAN.relative_to(ROOT).as_posix(),
        PREFLIGHT_RECEIPT.relative_to(ROOT).as_posix(),
        FROZEN_BINDING.relative_to(ROOT).as_posix(),
        *(path.relative_to(ROOT).as_posix()
          for path, _ in HISTORICAL_JSON.values()),
    }
    if any(
        Path(path).suffix.lower() == ".tsv" or "openings" in Path(path).parts or
        any(character in path for character in "*?[") or path.endswith("/")
        for path in paths
    ):
        raise ValueError("bounded Git path registry is not an exact non-bank file set")
    return tuple(sorted(paths))


def require_admin_commit(plan: dict[str, Any]) -> dict[str, Any]:
    head = str(git_run("rev-parse", "HEAD"))
    parents = str(git_run("rev-list", "--parents", "-n", "1", "HEAD")).split()
    if len(parents) != 2 or parents[0] != head or parents[1] != ADMIN_PARENT_COMMIT:
        raise ValueError("admin HEAD is not the exact direct child of the frozen parent")
    changed = tuple(sorted(filter(None, str(git_run(
        "diff", "--name-only", f"{ADMIN_PARENT_COMMIT}..HEAD"
    )).splitlines())))
    if changed != tuple(sorted(ADMIN_CHANGED_PATHS)):
        raise ValueError("admin HEAD does not have the exact three-path closure")
    scoped_paths = bounded_git_paths(plan)
    if git_run("diff", "--cached", "--name-only", "--", *scoped_paths):
        raise ValueError("bounded Git index inputs differ from admin HEAD")
    counts = str(git_run(
        "rev-list", "--left-right", "--count", "HEAD...@{upstream}"
    )).split()
    if counts != ["0", "0"]:
        raise ValueError("admin HEAD/upstream is not exact 0/0")
    timestamp_lines = str(git_run(
        "show", "-s", "--format=%aI%n%cI", "HEAD"
    )).splitlines()
    if len(timestamp_lines) != 2:
        raise ValueError("admin commit timestamp record is malformed")
    author_utc = parse_utc(timestamp_lines[0])
    committer_utc = parse_utc(timestamp_lines[1])
    lower = parse_utc(CAMPAIGN_T0_UTC)
    upper = parse_utc(CAMPAIGN_DEADLINE_UTC)
    if not (lower <= author_utc <= upper and lower <= committer_utc <= upper):
        raise ValueError("admin commit timestamp lies outside campaign interval")
    for path in ADMIN_CHANGED_PATHS:
        require_git_regular_blob("HEAD", path)
        require_index_regular_blob(path)
        identity = file_identity(ROOT / path, label=path, required_mode=0o644)
        committed = git_run("show", f"HEAD:{path}", binary=True)
        if (identity["mode"] != "0644" or not isinstance(committed, bytes) or
                sha256_bytes(committed) != identity["sha256"] or
                len(committed) != identity["bytes"]):
            raise ValueError(f"admin input differs from HEAD: {path}")
    return {
        "admin_commit": head,
        "candidate_source_commit": CANDIDATE_SOURCE_COMMIT,
        "direct_parent": ADMIN_PARENT_COMMIT,
        "exact_changed_paths": list(ADMIN_CHANGED_PATHS),
        "bounded_index_inputs_equal_head": True,
        "admin_worktree_inputs_equal_head": True,
        "bounded_git_paths": list(scoped_paths),
        "upstream_ahead_behind": [0, 0],
        "author_timestamp": timestamp_lines[0],
        "committer_timestamp": timestamp_lines[1],
    }


def source_identities(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    expected = plan.get("source_identities")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("plan source identity registry is malformed")
    result: dict[str, dict[str, Any]] = {}
    for label, digest in sorted(expected.items()):
        if not isinstance(label, str) or not SHA256_RE.fullmatch(str(digest)):
            raise ValueError("plan source identity entry is malformed")
        path = ROOT / label
        identity = file_identity(path, label=label, required_mode=0o644)
        require_git_regular_blob(CANDIDATE_SOURCE_COMMIT, label)
        require_git_regular_blob("HEAD", label)
        require_index_regular_blob(label)
        candidate = git_run("show", f"{CANDIDATE_SOURCE_COMMIT}:{label}", binary=True)
        current = git_run("show", f"HEAD:{label}", binary=True)
        if (identity["sha256"] != digest or identity["mode"] != "0644" or
                not isinstance(candidate, bytes) or
                not isinstance(current, bytes) or candidate != current or
                sha256_bytes(candidate) != digest):
            raise ValueError(f"candidate source/config identity drift: {label}")
        result[label] = identity
    return result


def producer_bootstrap_identities(
    plan: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    expected = plan.get("producer_bootstrap_identities")
    if expected != PRODUCER_BOOTSTRAP_IDENTITIES:
        raise ValueError("producer-bootstrap registry differs from frozen plan")
    result: dict[str, dict[str, Any]] = {}
    for label, frozen in sorted(PRODUCER_BOOTSTRAP_IDENTITIES.items()):
        identity = file_identity(
            ROOT / label, label=label, required_mode=0o644
        )
        require_git_regular_blob(CANDIDATE_SOURCE_COMMIT, label)
        require_git_regular_blob("HEAD", label)
        require_index_regular_blob(label)
        candidate = git_run(
            "show", f"{CANDIDATE_SOURCE_COMMIT}:{label}", binary=True
        )
        current = git_run("show", f"HEAD:{label}", binary=True)
        if (identity != {
                "ascii": True, "bytes": frozen["bytes"], "executable": False,
                "mode": frozen["mode"], "path": label,
                "sha256": frozen["sha256"],
                } or not isinstance(candidate, bytes) or
                not isinstance(current, bytes) or candidate != current or
                sha256_bytes(candidate) != frozen["sha256"] or
                len(candidate) != frozen["bytes"]):
            raise ValueError(f"producer-bootstrap identity drift: {label}")
        result[label] = identity
    return result


def validate_historical_evidence() -> dict[str, str]:
    loaded = {
        label: load_tracked_canonical_json(
            path, digest, live_mode=(
                0o444 if label == "closed_heldout_decision" else 0o644
            )
        )
        for label, (path, digest) in HISTORICAL_JSON.items()
    }
    matrix = loaded["proof_scope_matrix"]
    reports = {
        (item.get("candidate_exact_proof_mask"),
         item.get("reference_exact_proof_mask")): item
        for item in matrix.get("reports", ())
    }
    if (reports.get((3, 1), {}).get("candidate_wins") != 40 or
            reports.get((3, 1), {}).get("reference_wins") != 36 or
            reports.get((7, 3), {}).get("candidate_wins") != 39 or
            reports.get((7, 3), {}).get("reference_wins") != 37):
        raise ValueError("pre-heldout mask-removal evidence mismatch")
    selection = loaded["full_development_selection"].get("decision", {})
    if (selection.get("fallback_mask3_triggered") is not False or
            selection.get("selected_exact_proof_mask") != 7):
        raise ValueError("old fallback trigger evidence mismatch")
    closed = loaded["closed_heldout_decision"]
    if (closed.get("heldout_qualification_acceptable") is not False or
            closed.get("arena_authorization") is not False or
            closed.get("final_qualification") is not False):
        raise ValueError("closed predecessor outcome mismatch")
    load_tracked_exact_bytes(FULL_DEVELOPMENT_PLAN, FULL_DEVELOPMENT_PLAN_SHA256)
    return {
        **{label: digest for label, (_, digest) in HISTORICAL_JSON.items()},
        "full_development_plan": FULL_DEVELOPMENT_PLAN_SHA256,
    }


def _resolved_dependency(path: Path) -> Path:
    lexical = Path(os.path.abspath(path))
    if lexical.suffix.lower() == ".tsv":
        raise ValueError("compiler dependency unexpectedly names a TSV")
    resolved = lexical.resolve(strict=True)
    if resolved.suffix.lower() == ".tsv":
        raise ValueError("resolved compiler dependency unexpectedly names a TSV")
    root = ROOT.resolve(strict=True)
    if resolved == root or resolved.is_relative_to(root):
        return resolved
    if not any(
        resolved == allowed or resolved.is_relative_to(allowed)
        for allowed in ALLOWED_EXTERNAL_ROOTS
    ):
        raise ValueError(f"compiler dependency escapes fixed roots: {lexical}")
    return resolved


def parse_depfile(path: Path) -> list[Path]:
    raw = read_regular_file(path)
    try:
        text = raw.decode("ascii").replace("\\\n", " ")
    except UnicodeDecodeError as error:
        raise ValueError(f"compiler depfile is not ASCII: {path}") from error
    if ":" not in text:
        raise ValueError(f"malformed compiler depfile: {path}")
    _, dependencies = text.split(":", 1)
    parsed = [_resolved_dependency(Path(token) if Path(token).is_absolute()
                                   else ROOT / token)
              for token in shlex.split(dependencies)]
    if not parsed:
        raise ValueError(f"empty compiler depfile: {path}")
    return parsed


def host_runtime_identity() -> dict[str, Any]:
    """Reproduce and bind the exact host/runtime used by the frozen binary."""
    binding = load_tracked_canonical_json(
        FROZEN_BINDING, FROZEN_BINDING_SHA256, live_mode=0o444
    )
    python_path = Path(sys.executable).resolve(strict=True)
    python_identity = file_identity(
        python_path, label=str(python_path), required_mode=0o755
    )
    sysctl_identity = file_identity(
        SYSCTL, label=str(SYSCTL), required_mode=0o755
    )
    if (sysctl_identity["sha256"] != SYSCTL_SHA256 or
            sysctl_identity["mode"] != "0755" or
            sysctl_identity["executable"] is not True):
        raise ValueError("host-inspection tool identity changed")
    uname = platform.uname()
    cpu_model = platform.processor().strip()
    completed = subprocess.run(
        [str(SYSCTL), "-n", "machdep.cpu.brand_string"], cwd=ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        check=False, timeout=30,
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
    )
    if completed.returncode != 0 or completed.stderr:
        raise ValueError("host CPU identity inspection failed")
    cpu_model = completed.stdout.strip()
    host_payload = {
        "node": uname.node,
        "system": uname.system,
        "release": uname.release,
        "version": uname.version,
        "machine": uname.machine,
        "processor": uname.processor,
        "cpu_model": cpu_model,
        "logical_cpu_count": os.cpu_count(),
        "python_version": sys.version,
        "python_executable": python_identity,
    }
    host = {
        **host_payload,
        "sha256": sha256_bytes(canonical_json(host_payload)),
    }
    runtime = {
        "python_version": sys.version,
        "python_executable": python_identity,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    if host != binding.get("host") or runtime != binding.get("runtime"):
        raise ValueError("live host/runtime differs from frozen binding")
    return {
        "host": host,
        "runtime": runtime,
        "host_inspector": sysctl_identity,
    }


def runtime_linkage() -> dict[str, Any]:
    tool = file_identity(
        OTOOL, label=str(OTOOL), allow_sealed_os_multilink=True,
        required_mode=0o755,
    )
    if (tool["sha256"] != OTOOL_SHA256 or tool["bytes"] != OTOOL_BYTES or
            tool["mode"] != "0755" or tool["executable"] is not True):
        raise ValueError("runtime linkage tool identity changed")
    completed = subprocess.run(
        [str(OTOOL), "-L", str(GATE)], cwd=ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        check=False, timeout=60,
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
    )
    if completed.returncode != 0 or completed.stderr:
        raise ValueError("runtime linkage inspection failed")
    names = sorted({
        line.strip().split(maxsplit=1)[0]
        for line in completed.stdout.splitlines()[1:] if line.strip()
    })
    normalized = "\n".join(names) + ("\n" if names else "")
    if tuple(names) != LINKAGE_NAMES or sha256_bytes(normalized.encode()) != LINKAGE_SHA256:
        raise ValueError("runtime linkage changed")
    return {
        "dependency_names": names,
        "normalized_sha256": LINKAGE_SHA256,
        "tool": tool,
    }


def preserved_binary_closure() -> dict[str, Any]:
    binding = load_tracked_canonical_json(
        FROZEN_BINDING, FROZEN_BINDING_SHA256, live_mode=0o444
    )
    receipt = load_tracked_canonical_json(PREFLIGHT_RECEIPT, PREFLIGHT_RECEIPT_SHA256)
    if (binding.get("candidate_source_commit") != CANDIDATE_SOURCE_COMMIT or
            receipt.get("status") != "passed"):
        raise ValueError("preserved binary provenance mismatch")
    expected = dict(binding.get("dependency_identities", {}))
    try:
        expected.update(receipt["builds"]["clang_release"]["build_artifacts"])
    except (KeyError, TypeError) as error:
        raise ValueError("preserved build artifact registry is malformed") from error
    paths = set(DIRECT_BUILD_ARTIFACTS)
    for depfile in DEPFILES:
        paths.update(parse_depfile(depfile))
    labels = sorted({path_label(path) for path in paths})
    if (len(labels) != EXPECTED_CLOSURE_ENTRIES or
            sum(label.startswith("/") for label in labels) !=
            EXPECTED_CLOSURE_EXTERNAL_ENTRIES or
            sum(not label.startswith("/") for label in labels) !=
            EXPECTED_CLOSURE_REPO_ENTRIES):
        raise ValueError("preserved normal-gate closure cardinality changed")
    expected_subset: dict[str, dict[str, Any]] = {}
    live_subset: dict[str, dict[str, Any]] = {}
    for label in labels:
        record = expected.get(label)
        if not isinstance(record, dict):
            raise ValueError(f"preserved closure lacks identity: {label}")
        path = Path(label) if label.startswith("/") else ROOT / label
        live = file_identity(path, label=label)
        if live != record:
            raise ValueError(f"preserved closure identity drift: {label}")
        expected_subset[label] = record
        live_subset[label] = live
    expected_digest = sha256_bytes(canonical_json(expected_subset))
    live_digest = sha256_bytes(canonical_json(live_subset))
    if expected_digest != EXPECTED_CLOSURE_SHA256 or live_digest != expected_digest:
        raise ValueError("preserved closure digest mismatch")
    binary = live_subset[path_label(GATE)]
    if (binary["sha256"] != GATE_SHA256 or binary["bytes"] != GATE_BYTES or
            binary["mode"] != "0755" or binary["executable"] is not True):
        raise ValueError("preserved comparison gate identity mismatch")
    return {
        "binary": binary,
        "closure_sha256": live_digest,
        "entries": len(labels),
        "repo_entries": EXPECTED_CLOSURE_REPO_ENTRIES,
        "external_entries": EXPECTED_CLOSURE_EXTERNAL_ENTRIES,
        "frozen_binding_sha256": FROZEN_BINDING_SHA256,
        "preflight_receipt_sha256": PREFLIGHT_RECEIPT_SHA256,
        "runtime_linkage": runtime_linkage(),
    }


def process_snapshot() -> dict[str, Any]:
    identity = file_identity(PS, label=str(PS), required_mode=0o4755)
    if (identity["sha256"] != PS_SHA256 or identity["mode"] != "4755" or
            identity["executable"] is not True):
        raise ValueError("process-table tool identity changed")
    completed = subprocess.run(
        [str(PS), "-axo", "pid=,ppid=,command="], cwd=ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        check=False, timeout=30, env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
    )
    if completed.returncode != 0 or completed.stderr:
        raise ValueError("process-table check failed")
    rows: list[tuple[int, int, str]] = []
    for line in completed.stdout.splitlines():
        pieces = line.strip().split(maxsplit=2)
        if len(pieces) == 3 and pieces[0].isdigit() and pieces[1].isdigit():
            rows.append((int(pieces[0]), int(pieces[1]), pieces[2]))
    parents = {pid: ppid for pid, ppid, _ in rows}
    allowed = {os.getpid()}
    cursor = os.getpid()
    while cursor in parents and parents[cursor] not in allowed:
        cursor = parents[cursor]
        allowed.add(cursor)
    markers = (
        str(GATE), GATE.name, "record_rank4_jacek_hybrid",
        *(item[0] for item in BANKS),
    )
    conflicts = [
        {"pid": pid, "ppid": ppid, "command": command}
        for pid, ppid, command in rows
        if pid not in allowed and any(marker in command for marker in markers)
    ]
    if conflicts:
        raise ValueError("conflicting mask3-removal or DEVELOPMENT process exists")
    return {
        "checked_utc": utc_now(),
        "clean": True,
        "conflicts": [],
        "observed_process_count": len(rows),
        "tool": identity,
    }


def validate_process_snapshot(record: Any) -> None:
    expected_keys = {
        "checked_utc", "clean", "conflicts", "observed_process_count", "tool"
    }
    if (not isinstance(record, dict) or set(record) != expected_keys or
            record.get("clean") is not True or record.get("conflicts") != [] or
            type(record.get("observed_process_count")) is not int or
            record["observed_process_count"] <= 0 or
            not isinstance(record.get("checked_utc"), str)):
        raise ValueError("preclaim process record schema mismatch")
    parse_utc(record["checked_utc"])
    tool = record.get("tool")
    if (not isinstance(tool, dict) or set(tool) != {
            "ascii", "bytes", "executable", "mode", "path", "sha256"
            } or tool != {
                "ascii": False,
                "bytes": PS_BYTES,
                "executable": True,
                "mode": "4755",
                "path": str(PS),
                "sha256": PS_SHA256,
            }):
        raise ValueError("preclaim process tool record mismatch")


def gate_environment() -> dict[str, str]:
    return {
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin",
        "TZ": "UTC",
    }


def stable_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in evidence.items() if key != "process"}


def evidence_digest(evidence: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(stable_evidence(evidence)))


def qualification_key(
    plan: dict[str, Any], admin: dict[str, Any], sources: dict[str, Any],
    bootstrap: dict[str, Any],
    closure: dict[str, Any], historical: dict[str, str],
    host_runtime: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "rank4-jacek-hybrid-mask3-removal-key-v1",
        "campaign_id": CAMPAIGN_ID,
        "plan_sha256": PLAN_SHA256,
        "candidate_source_commit": CANDIDATE_SOURCE_COMMIT,
        "admin_commit": admin["admin_commit"],
        "source_identities_sha256": sha256_bytes(canonical_json(sources)),
        "producer_bootstrap_identities_sha256": sha256_bytes(
            canonical_json(bootstrap)
        ),
        "binary_closure_sha256": closure["closure_sha256"],
        "binary_sha256": GATE_SHA256,
        "environment_sha256": sha256_bytes(canonical_json(gate_environment())),
        "host_sha256": host_runtime["host"]["sha256"],
        "runtime_sha256": sha256_bytes(canonical_json(host_runtime["runtime"])),
        "host_inspector_sha256": host_runtime["host_inspector"]["sha256"],
        "historical_evidence": historical,
        "banks_from_plan_metadata_only": plan["banks"],
        "configuration": plan["configuration"],
        "stages": [
            {key: item[key] for key in (
                "stage", "reference_engine", "candidate_exact_proof_mask",
                "reference_exact_proof_mask",
            )}
            for item in STAGES
        ],
        "thresholds_each_stage": plan["thresholds_each_stage"],
    }


def require_before_deadline() -> None:
    if datetime.now(timezone.utc) >= parse_utc(CAMPAIGN_DEADLINE_UTC):
        raise ValueError("mask3-removal campaign deadline has passed")


def prepare_evidence() -> dict[str, Any]:
    plan = load_plan()
    admin = require_admin_commit(plan)
    historical = validate_historical_evidence()
    sources = source_identities(plan)
    bootstrap = producer_bootstrap_identities(plan)
    closure = preserved_binary_closure()
    host_runtime = host_runtime_identity()
    process = process_snapshot()
    key = qualification_key(
        plan, admin, sources, bootstrap, closure, historical, host_runtime
    )
    identifier = sha256_bytes(canonical_json(key))
    return {
        "plan_sha256": PLAN_SHA256,
        "admin": admin,
        "historical": historical,
        "sources": sources,
        "producer_bootstrap": bootstrap,
        "closure": closure,
        **host_runtime,
        "environment_sha256": sha256_bytes(canonical_json(gate_environment())),
        "qualification_key": key,
        "candidate_identity": identifier,
        "process": process,
    }


def stage_spec(stage: str) -> dict[str, Any]:
    for item in STAGES:
        if item["stage"] == stage:
            return item
    raise ValueError(f"unknown stage: {stage}")


def command_for(stage: str) -> list[str]:
    spec = stage_spec(stage)
    command = [str(GATE), "--profile", "clock", "--reference-engine",
               str(spec["reference_engine"])]
    for path in BANK_PATHS:
        command.extend(("--bank", str(path)))
    command.extend((
        "--expected-role", "development",
        "--expected-depths", ",".join(str(item[1]) for item in BANKS),
        "--expected-seeds", ",".join(item[2] for item in BANKS),
        "--expected-sha256", ",".join(item[3] for item in BANKS),
        "--max-turns", "320",
        "--candidate-nodes", "3000000",
        "--reference-nodes", "3000000",
        "--candidate-first-ms", "800",
        "--candidate-later-ms", "165",
        "--reference-first-ms", "800",
        "--reference-later-ms", "165",
        "--operational-first-ms", "1000",
        "--operational-later-ms", "200",
        "--candidate-exact-proof-mask", "3",
        "--reference-exact-proof-mask", str(spec["reference_exact_proof_mask"]),
    ))
    return command


def configuration_expected(stage: str) -> dict[str, str]:
    spec = stage_spec(stage)
    result = old_full.configuration_expected(
        str(spec["reference_engine"]), candidate_mask=3
    )
    result["reference_exact_proof_mask"] = str(
        spec["reference_exact_proof_mask"]
    )
    return result


def fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_record_metadata(
    metadata: os.stat_result, path: Path,
    *, expected_inode: tuple[int, int] | None = None,
) -> None:
    if (not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or
            stat.S_IMODE(metadata.st_mode) != 0o444 or
            (expected_inode is not None and
             _inode_key(metadata) != expected_inode)):
        raise ValueError(f"record is not an immutable single-link file: {path}")


def read_regular_record(
    path: Path, *, expected_inode: tuple[int, int] | None = None,
) -> bytes:
    """Read one immutable record without ever following a replaced symlink."""
    try:
        return _read_regular_nofollow(
            path, expected_inode=expected_inode, required_mode=0o444,
            require_single_link=True, kind="record",
        )[0]
    except ValueError as error:
        raise ValueError(
            f"record is not an immutable single-link file: {path}"
        ) from error


def fsync_regular_record(
    path: Path, *, expected_inode: tuple[int, int] | None = None,
) -> None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        _validate_record_metadata(
            metadata, path, expected_inode=expected_inode
        )
        os.fsync(descriptor)
        after = os.fstat(descriptor)
        _validate_record_metadata(
            after, path, expected_inode=_inode_key(metadata)
        )
        if after.st_size != metadata.st_size:
            raise ValueError(f"recovery record changed during fsync: {path}")
        path_after = os.lstat(path)
        _validate_record_metadata(
            path_after, path, expected_inode=_inode_key(metadata)
        )
    finally:
        os.close(descriptor)
    fsync_directory(path.parent)


def _ensure_directory(path: Path) -> None:
    try:
        os.mkdir(path, 0o755)
    except FileExistsError:
        pass
    metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"registry is not a regular directory: {path}")
    # This is unconditional so restart repairs a crash between mkdir and the
    # original parent-directory fsync.
    fsync_directory(path.parent)


def prepare_runtime_directories() -> None:
    for path in (CLAIMS, EXECUTIONS, REPORTS, DECISIONS):
        _ensure_directory(path)


def repair_registry_durability(evidence: dict[str, Any]) -> None:
    """Repair visibility-only crash remnants before recovery may trust them."""
    identifier = evidence["candidate_identity"]
    stage_pattern = re.compile(
        rf"{re.escape(identifier)}\.({'|'.join(map(re.escape, STAGE_NAMES))})\.json"
    )
    content_pattern = re.compile(r"[0-9a-f]{64}\.json")
    registries = (
        (CLAIMS, stage_pattern), (EXECUTIONS, stage_pattern),
        (REPORTS, content_pattern), (DECISIONS, content_pattern),
    )
    validated = [
        (directory, _registry_snapshot(directory, pattern))
        for directory, pattern in registries
    ]
    for directory, records in validated:
        for path, inode in records:
            fsync_regular_record(path, expected_inode=inode)
        fsync_directory(directory)


def validate_held_lock(path: Path, descriptor: int) -> None:
    before = os.fstat(descriptor)
    path_metadata = os.lstat(path)
    after = os.fstat(descriptor)
    for metadata in (before, path_metadata, after):
        if (not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or
                stat.S_IMODE(metadata.st_mode) != 0o644):
            raise ValueError(f"held lock identity changed: {path}")
    if (_inode_key(before) != _inode_key(path_metadata) or
            _inode_key(after) != _inode_key(before) or
            after.st_mode != before.st_mode or
            after.st_nlink != before.st_nlink):
        raise ValueError(f"held lock pathname no longer names its inode: {path}")


def open_exclusive_lock(path: Path) -> int:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    created = False
    try:
        descriptor = os.open(
            path, os.O_RDWR | os.O_CREAT | os.O_EXCL | nofollow, 0o644
        )
        created = True
    except FileExistsError:
        metadata = os.lstat(path)
        if (stat.S_ISLNK(metadata.st_mode) or
                not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1):
            raise ValueError(f"lock path is not a regular file: {path}")
        expected_inode = _inode_key(metadata)
        descriptor = os.open(path, os.O_RDWR | nofollow)
    try:
        metadata = os.fstat(descriptor)
        if (not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or
                (not created and _inode_key(metadata) != expected_inode)):
            raise ValueError(f"lock path is not a regular file: {path}")
        if created:
            # Creation permissions are filtered by the caller's unbound umask.
            # Set the preregistered fixed mode only on the newly created inode.
            os.fchmod(descriptor, 0o644)
            metadata = os.fstat(descriptor)
        if stat.S_IMODE(metadata.st_mode) != 0o644:
            raise ValueError(f"lock path has the wrong fixed mode: {path}")
        os.fsync(descriptor)
        fsync_directory(path.parent)
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        validate_held_lock(path, descriptor)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _registry_snapshot(
    path: Path, pattern: re.Pattern[str],
) -> list[tuple[Path, tuple[int, int]]]:
    if not path.exists():
        return []
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"registry is not a regular directory: {path}")
    result: list[tuple[Path, tuple[int, int]]] = []
    with os.scandir(path) as entries:
        for entry in entries:
            metadata = entry.stat(follow_symlinks=False)
            if entry.is_symlink() or pattern.fullmatch(entry.name) is None:
                raise ValueError(f"foreign registry entry: {entry.path}")
            record_path = Path(entry.path)
            _validate_record_metadata(metadata, record_path)
            result.append((record_path, _inode_key(metadata)))
    return sorted(result, key=lambda item: item[0])


def _registry_files(path: Path, pattern: re.Pattern[str]) -> list[Path]:
    return [record[0] for record in _registry_snapshot(path, pattern)]


def validate_output_topology() -> None:
    if OUTPUT.is_symlink() or not OUTPUT.is_dir():
        raise ValueError("mask3-removal output root is not a regular directory")
    allowed = {PLAN.name, LOCK.name, CLAIMS.name, EXECUTIONS.name,
               REPORTS.name, DECISIONS.name}
    with os.scandir(OUTPUT) as entries:
        for entry in entries:
            if entry.name not in allowed or entry.is_symlink():
                raise ValueError(f"foreign mask3-removal output entry: {entry.path}")
            if entry.name in {PLAN.name, LOCK.name}:
                metadata = entry.stat(follow_symlinks=False)
                expected_mode = 0o644
                if (not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or
                        stat.S_IMODE(metadata.st_mode) != expected_mode):
                    raise ValueError("mask3-removal fixed file identity changed")
            if entry.name not in {PLAN.name, LOCK.name} and not entry.is_dir(
                    follow_symlinks=False):
                raise ValueError("mask3-removal registry entry changed type")


def _write_all(descriptor: int, raw: bytes) -> None:
    offset = 0
    while offset < len(raw):
        written = os.write(descriptor, raw[offset:])
        if written <= 0:
            raise OSError("exclusive write made no progress")
        offset += written
    os.fsync(descriptor)


def write_exclusive(path: Path, payload: dict[str, Any]) -> tuple[str, bytes]:
    raw = canonical_json(payload)
    flags = (os.O_WRONLY | os.O_CREAT | os.O_EXCL |
             getattr(os, "O_NOFOLLOW", 0))
    descriptor = os.open(path, flags, 0o444)
    persisted_inode: tuple[int, int] | None = None
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError("new exclusive record is not a single-link regular file")
        os.fchmod(descriptor, 0o444)
        _write_all(descriptor, raw)
        final_metadata = os.fstat(descriptor)
        _validate_record_metadata(final_metadata, path)
        if final_metadata.st_size != len(raw):
            raise OSError("new exclusive record has an unexpected size")
        persisted_inode = _inode_key(final_metadata)
    finally:
        os.close(descriptor)
    fsync_directory(path.parent)
    if persisted_inode is None:
        raise OSError("exclusive record inode was not captured")
    persisted = read_regular_record(path, expected_inode=persisted_inode)
    if persisted != raw or canonical_json(decode_json(persisted)) != persisted:
        raise OSError(f"exclusive canonical write failed readback: {path}")
    return sha256_bytes(raw), raw


def persist_content_addressed(
    directory: Path, payload: dict[str, Any]
) -> tuple[Path, str]:
    metadata = os.lstat(directory)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"content-address registry changed type: {directory}")
    raw = canonical_json(payload)
    digest = sha256_bytes(raw)
    path = directory / f"{digest}.json"
    try:
        persisted_digest, _ = write_exclusive(path, payload)
    except FileExistsError:
        metadata = os.lstat(path)
        _validate_record_metadata(metadata, path)
        inode = _inode_key(metadata)
        fsync_regular_record(path, expected_inode=inode)
        existing = read_regular_record(path, expected_inode=inode)
        if existing != raw:
            raise ValueError("content-address collision or corrupt existing record")
        persisted_digest = sha256_bytes(existing)
    if persisted_digest != digest:
        raise OSError("content-addressed digest mismatch")
    return path, digest


def claim_path(identifier: str, stage: str) -> Path:
    return CLAIMS / f"{identifier}.{stage}.json"


def execution_path(identifier: str, stage: str) -> Path:
    return EXECUTIONS / f"{identifier}.{stage}.json"


def _record_reference(path: Path, digest: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": digest,
        "schema": payload["schema"],
    }


def load_claim(
    path: Path, evidence: dict[str, Any], *, raw: bytes | None = None,
    expected_inode: tuple[int, int] | None = None,
) -> tuple[dict[str, Any], str]:
    if raw is None:
        raw = read_regular_record(path, expected_inode=expected_inode)
    claim = decode_json(raw)
    expected_keys = {
        "admin_commit", "campaign_id", "candidate_identity",
        "candidate_source_commit", "claimed_utc", "evidence_sha256",
        "plan_sha256", "preclaim_process", "prior_reports", "schema", "stage",
    }
    if (not isinstance(claim, dict) or set(claim) != expected_keys or
            canonical_json(claim) != raw or claim.get("schema") != CLAIM_SCHEMA or
            claim.get("campaign_id") != CAMPAIGN_ID or
            claim.get("candidate_identity") != evidence["candidate_identity"] or
            claim.get("candidate_source_commit") != CANDIDATE_SOURCE_COMMIT or
            claim.get("admin_commit") != evidence["admin"]["admin_commit"] or
            claim.get("plan_sha256") != PLAN_SHA256 or
            claim.get("evidence_sha256") != evidence_digest(evidence) or
            claim.get("stage") not in STAGE_NAMES):
        raise ValueError("stage claim provenance mismatch")
    if path != claim_path(evidence["candidate_identity"], claim["stage"]):
        raise ValueError("stage claim filename/stage mismatch")
    validate_process_snapshot(claim["preclaim_process"])
    claimed = parse_utc(str(claim["claimed_utc"]))
    checked = parse_utc(claim["preclaim_process"]["checked_utc"])
    if (not (parse_utc(CAMPAIGN_T0_UTC) <= checked <= claimed <=
             parse_utc(CAMPAIGN_DEADLINE_UTC)) or
            (claimed - checked).total_seconds() > 60.0):
        raise ValueError("stage claim timestamp lies outside campaign interval")
    return claim, sha256_bytes(raw)


def create_claim(
    evidence: dict[str, Any], stage: str,
    prior_reports: list[tuple[Path, str, dict[str, Any]]],
) -> tuple[Path, str, dict[str, Any]]:
    path = claim_path(evidence["candidate_identity"], stage)
    claimed_utc = utc_now()
    claimed = parse_utc(claimed_utc)
    checked = parse_utc(evidence["process"]["checked_utc"])
    predecessor_times = [
        parse_utc(item[2]["created_utc"]) for item in prior_reports
    ]
    if (not (parse_utc(CAMPAIGN_T0_UTC) <= checked <= claimed <=
             parse_utc(CAMPAIGN_DEADLINE_UTC)) or
            (claimed - checked).total_seconds() > 60.0 or
            (predecessor_times and checked < max(predecessor_times))):
        raise ValueError("campaign deadline passed before O_EXCL claim")
    payload = {
        "schema": CLAIM_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "candidate_identity": evidence["candidate_identity"],
        "candidate_source_commit": CANDIDATE_SOURCE_COMMIT,
        "admin_commit": evidence["admin"]["admin_commit"],
        "plan_sha256": PLAN_SHA256,
        "evidence_sha256": evidence_digest(evidence),
        "preclaim_process": evidence["process"],
        "stage": stage,
        "prior_reports": [
            _record_reference(item[0], item[1], item[2])
            for item in prior_reports
        ],
        "claimed_utc": claimed_utc,
    }
    digest, _ = write_exclusive(path, payload)
    loaded, loaded_digest = load_claim(path, evidence)
    if loaded != payload or loaded_digest != digest:
        raise OSError("stage claim readback mismatch")
    return path, digest, payload


def execute_gate(command: list[str]) -> dict[str, Any]:
    started_utc = utc_now()
    started_ns = time.monotonic_ns()
    timed_out = False
    os_error: str | None = None
    try:
        completed = subprocess.run(
            command, cwd=ROOT, env=gate_environment(), text=False,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            timeout=RUN_TIMEOUT_SECONDS,
        )
        returncode: int | None = completed.returncode
        stdout: bytes = completed.stdout
        stderr: bytes = completed.stderr
    except subprocess.TimeoutExpired as error:
        timed_out = True
        returncode = None
        stdout = error.stdout or b""
        stderr = error.stderr or b""
        if isinstance(stdout, str):
            stdout = stdout.encode("utf-8", errors="surrogatepass")
        if isinstance(stderr, str):
            stderr = stderr.encode("utf-8", errors="surrogatepass")
    except OSError as error:
        returncode = None
        stdout = b""
        stderr = b""
        os_error = type(error).__name__
    ended_ns = time.monotonic_ns()
    return {
        "started_utc": started_utc,
        "ended_utc": utc_now(),
        "elapsed_monotonic_ns": ended_ns - started_ns,
        "returncode": returncode,
        "timed_out": timed_out,
        "os_error_class": os_error,
        "stdout": stdout,
        "stderr": stderr,
    }


def _stream_metadata(raw: bytes) -> dict[str, Any]:
    return {
        "bytes": len(raw),
        "empty": not raw,
        "retained": False,
        "sha256": sha256_bytes(raw),
    }


def _stdout_receipt(raw: bytes, stage: str) -> dict[str, Any]:
    record = _stream_metadata(raw)
    try:
        value = raw.decode("ascii")
    except UnicodeDecodeError:
        value = ""
    exact_lines = value[:-1].split("\n") if value.endswith("\n") else []
    valid_shape = (
        bool(value) and "\r" not in value and "\x00" not in value and
        value.count("\n") == 6 and len(exact_lines) == 6 and
        all(exact_lines) and
        all(line.startswith("bank_summary ") for line in exact_lines[:4]) and
        exact_lines[4].startswith("summary ") and
        exact_lines[5].startswith("configuration ")
    )
    if valid_shape:
        try:
            summaries = [common.parse_fields(line) for line in exact_lines[:5]]
            configuration = common.parse_fields(exact_lines[5])
            valid_shape = (
                all(frozenset(item) == EXPECTED_SUMMARY_FIELDS
                    for item in summaries) and
                frozenset(configuration) == EXPECTED_CONFIGURATION_FIELDS and
                configuration == configuration_expected(stage)
            )
        except (KeyError, TypeError, ValueError):
            valid_shape = False
    if valid_shape:
        record["retained"] = True
        record["text"] = value
    return record


def _validate_stream_record(
    record: Any, *, allow_retained: bool, stage: str | None = None,
) -> str | None:
    if not isinstance(record, dict):
        raise ValueError("execution stream record is not an object")
    retained = record.get("retained")
    expected_keys = {"bytes", "empty", "retained", "sha256"}
    if retained is True:
        expected_keys.add("text")
    if (set(record) != expected_keys or type(record.get("bytes")) is not int or
            record["bytes"] < 0 or type(record.get("empty")) is not bool or
            type(retained) is not bool or
            not isinstance(record.get("sha256"), str) or
            SHA256_RE.fullmatch(record["sha256"]) is None or
            (retained and not allow_retained)):
        raise ValueError("execution stream record schema mismatch")
    if retained:
        text = record.get("text")
        if (not isinstance(text, str) or stage is None or
                _stdout_receipt(
                    text.encode("ascii", errors="strict"), stage
                ) != record):
            raise ValueError("retained execution stdout is not exact six-line schema")
        return text
    if record["empty"] != (record["bytes"] == 0):
        raise ValueError("unretained execution stream empty/byte mismatch")
    if record["bytes"] == 0 and record["sha256"] != sha256_bytes(b""):
        raise ValueError("empty execution stream digest mismatch")
    return None


def persist_execution(
    evidence: dict[str, Any], stage: str,
    claim: tuple[Path, str, dict[str, Any]], result: dict[str, Any],
) -> tuple[Path, str, dict[str, Any]]:
    result_fields = {
        key: result[key] for key in (
            "started_utc", "ended_utc", "elapsed_monotonic_ns", "returncode",
            "timed_out", "os_error_class",
        )
    }
    payload = {
        "schema": EXECUTION_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "candidate_identity": evidence["candidate_identity"],
        "stage": stage,
        "claim": _record_reference(claim[0], claim[1], claim[2]),
        "command": command_for(stage),
        "environment_sha256": evidence["environment_sha256"],
        "timeout_seconds": RUN_TIMEOUT_SECONDS,
        **result_fields,
        "stdout": _stdout_receipt(result["stdout"], stage),
        "stderr": _stream_metadata(result["stderr"]),
        "preclaim_process": claim[2]["preclaim_process"],
        "development_bank_paths_accessed_after_claim": [
            path.relative_to(ROOT).as_posix() for path in BANK_PATHS
        ],
        "protected_bank_files_accessed": [],
    }
    path = execution_path(evidence["candidate_identity"], stage)
    digest, _ = write_exclusive(path, payload)
    return path, digest, payload


def load_execution(
    path: Path, evidence: dict[str, Any],
    claim: tuple[Path, str, dict[str, Any]],
    *, raw: bytes | None = None,
    expected_inode: tuple[int, int] | None = None,
) -> tuple[dict[str, Any], str]:
    if raw is None:
        raw = read_regular_record(path, expected_inode=expected_inode)
    execution = decode_json(raw)
    expected_keys = {
        "campaign_id", "candidate_identity", "claim", "command",
        "development_bank_paths_accessed_after_claim", "elapsed_monotonic_ns",
        "ended_utc", "environment_sha256", "os_error_class",
        "preclaim_process", "protected_bank_files_accessed", "returncode",
        "schema", "stage", "started_utc", "stderr", "stdout",
        "timed_out", "timeout_seconds",
    }
    if (not isinstance(execution, dict) or set(execution) != expected_keys or
            canonical_json(execution) != raw or
            execution.get("schema") != EXECUTION_SCHEMA or
            execution.get("campaign_id") != CAMPAIGN_ID or
            execution.get("candidate_identity") != evidence["candidate_identity"] or
            execution.get("stage") != claim[2]["stage"] or
            execution.get("claim") != _record_reference(*claim) or
            execution.get("command") != command_for(claim[2]["stage"]) or
            execution.get("environment_sha256") != evidence["environment_sha256"] or
            type(execution.get("timeout_seconds")) is not int or
            execution.get("timeout_seconds") != RUN_TIMEOUT_SECONDS or
            type(execution.get("elapsed_monotonic_ns")) is not int or
            execution["elapsed_monotonic_ns"] < 0 or
            type(execution.get("timed_out")) is not bool or
            (execution.get("returncode") is not None and
             type(execution.get("returncode")) is not int) or
            (execution.get("os_error_class") is not None and
             not isinstance(execution.get("os_error_class"), str)) or
            execution.get("preclaim_process") != claim[2]["preclaim_process"] or
            execution.get("protected_bank_files_accessed") != [] or
            execution.get("development_bank_paths_accessed_after_claim") != [
                item.relative_to(ROOT).as_posix() for item in BANK_PATHS
            ]):
        raise ValueError("stage execution receipt provenance mismatch")
    if not isinstance(execution.get("started_utc"), str) or not isinstance(
            execution.get("ended_utc"), str):
        raise ValueError("gate execution timestamp type mismatch")
    # Timestamp order, wall/monotonic agreement, and the outcome tuple are
    # semantic observations, not receipt-structure errors.  Keep such a
    # durably produced receipt loadable so it can become a terminal report.
    parse_utc(execution["started_utc"])
    parse_utc(execution["ended_utc"])
    _validate_stream_record(
        execution["stdout"], allow_retained=True, stage=claim[2]["stage"]
    )
    _validate_stream_record(execution["stderr"], allow_retained=False)
    return execution, sha256_bytes(raw)


def validate_full_summaries(
    banks: list[dict[str, str]], aggregate: dict[str, str], stage: str,
) -> None:
    def validate_work(fields: dict[str, str]) -> None:
        for engine in ("candidate", "reference"):
            invocations = common.exact_int(fields, f"{engine}_invocations")
            searches = common.exact_int(fields, f"{engine}_searches")
            nodes = common.exact_int(fields, f"{engine}_nodes")
            nodes_p99 = common.exact_int(fields, f"{engine}_nodes_p99")
            nodes_max = common.exact_int(fields, f"{engine}_nodes_max")
            depth_max = common.exact_int(fields, f"{engine}_depth_max")
            attempted_max = common.exact_int(
                fields, f"{engine}_attempted_depth_max"
            )
            soft_overruns = common.exact_int(
                fields, f"{engine}_soft_overruns"
            )
            exhaustions = common.exact_int(fields, f"{engine}_exhaustions")
            nodes_avg = old_full.finite_nonnegative(
                fields, f"{engine}_nodes_avg"
            )
            depth_avg = old_full.finite_nonnegative(
                fields, f"{engine}_depth_avg"
            )
            attempted_avg = old_full.finite_nonnegative(
                fields, f"{engine}_attempted_depth_avg"
            )
            if nodes_p99 > nodes_max:
                raise ValueError(f"{engine} nodes p99 exceeds maximum")
            if nodes_max > min(nodes, 3_000_000):
                raise ValueError(
                    f"{engine} node maximum exceeds total or configured budget"
                )
            if nodes_avg > nodes_max:
                raise ValueError(f"{engine} average nodes exceeds maximum")
            expected_nodes_avg = nodes / searches
            if not math.isclose(
                    nodes_avg, expected_nodes_avg,
                    rel_tol=0.0, abs_tol=0.000_501):
                raise ValueError(f"{engine} average nodes/accounting mismatch")
            if depth_avg > depth_max:
                raise ValueError(f"{engine} average depth exceeds maximum")
            if (attempted_max < depth_max or attempted_avg < depth_avg or
                    attempted_avg > attempted_max):
                raise ValueError(f"{engine} attempted/completed depth mismatch")
            if soft_overruns > invocations or exhaustions > searches:
                raise ValueError(f"{engine} bounded work counter mismatch")

    spec = stage_spec(stage)
    if len(banks) != len(BANKS):
        raise ValueError("wrong number of DEVELOPMENT bank summaries")
    for index, (fields, bank) in enumerate(zip(banks, BANKS)):
        common.validate_summary(
            fields, str(index), 3, int(spec["reference_exact_proof_mask"]),
            expected_games=bank[4], expected_color_games=bank[4] // 2,
        )
        old_full.validate_timing(fields)
        old_full.validate_rebound_identity(fields)
        validate_work(fields)
    common.validate_summary(
        aggregate, "all", 3, int(spec["reference_exact_proof_mask"]),
        expected_games=306, expected_color_games=153,
    )
    old_full.validate_timing(aggregate)
    old_full.validate_rebound_identity(aggregate)
    validate_work(aggregate)
    old_full.validate_bank_aggregate_consistency(banks, aggregate)


def parse_execution_output(
    stage: str, execution: dict[str, Any]
) -> tuple[dict[str, Any], list[str], list[str]]:
    retained_stdout = _validate_stream_record(
        execution.get("stdout"), allow_retained=True, stage=stage
    )
    lines = retained_stdout[:-1].split("\n") if retained_stdout is not None else []
    bank_lines = [line for line in lines if line.startswith("bank_summary ")]
    summary_lines = [line for line in lines if line.startswith("summary ")]
    configuration_lines = [
        line for line in lines if line.startswith("configuration ")
    ]
    validation_errors: list[str] = []
    threshold_errors: list[str] = []
    parsed: dict[str, Any] = {"banks": [], "aggregate": {}, "configuration": {}}
    try:
        if (len(lines) != 6 or len(bank_lines) != 4 or len(summary_lines) != 1 or
                len(configuration_lines) != 1):
            raise ValueError("gate stdout is not exactly six expected lines")
        parsed["banks"] = [common.parse_fields(line) for line in bank_lines]
        parsed["aggregate"] = common.parse_fields(summary_lines[0])
        parsed["configuration"] = common.parse_fields(configuration_lines[0])
        if (any(frozenset(item) != EXPECTED_SUMMARY_FIELDS
                for item in (*parsed["banks"], parsed["aggregate"])) or
                frozenset(parsed["configuration"]) !=
                EXPECTED_CONFIGURATION_FIELDS):
            raise ValueError("gate output field set differs from frozen schema")
        if parsed["configuration"] != configuration_expected(stage):
            raise ValueError("complete configuration echo mismatch")
        validate_full_summaries(parsed["banks"], parsed["aggregate"], stage)
        threshold_errors = old_full.selection_threshold_errors(parsed["aggregate"])
    except (KeyError, OverflowError, TypeError, ValueError) as error:
        validation_errors.append(str(error))
    return parsed, validation_errors, threshold_errors


def exception_record(error: BaseException) -> dict[str, Any]:
    raw = str(error).encode("utf-8", errors="replace")
    return {
        "class": type(error).__name__,
        "message_bytes": len(raw),
        "message_sha256": sha256_bytes(raw),
        "retained": False,
    }


def validate_exception_record(record: Any) -> None:
    if (not isinstance(record, dict) or set(record) != {
            "class", "message_bytes", "message_sha256", "retained"
            } or not isinstance(record.get("class"), str) or not record["class"] or
            type(record.get("message_bytes")) is not int or
            record["message_bytes"] < 0 or
            not isinstance(record.get("message_sha256"), str) or
            SHA256_RE.fullmatch(record["message_sha256"]) is None or
            record.get("retained") is not False):
        raise ValueError("postflight exception record schema mismatch")


def interrupted_postflight_record() -> dict[str, Any]:
    raw = b"durable execution recovered without a durable postflight report"
    return {
        "class": "InterruptedPostflightRecovery",
        "message_bytes": len(raw),
        "message_sha256": sha256_bytes(raw),
        "retained": False,
    }


def execution_semantics(
    claim: tuple[Path, str, dict[str, Any]], execution: dict[str, Any],
    report_created_utc: str,
) -> tuple[bool, bool, list[str]]:
    claimed = parse_utc(claim[2]["claimed_utc"])
    started = parse_utc(execution["started_utc"])
    ended = parse_utc(execution["ended_utc"])
    reported = parse_utc(report_created_utc)
    lower = parse_utc(CAMPAIGN_T0_UTC)
    upper = parse_utc(CAMPAIGN_DEADLINE_UTC)
    ordered = claimed <= started <= ended <= reported
    within_interval = all(
        lower <= value <= upper
        for value in (claimed, started, ended, reported)
    )
    wall_seconds = (ended - started).total_seconds()
    monotonic_seconds = execution["elapsed_monotonic_ns"] / 1_000_000_000
    wall_monotonic_valid = (
        monotonic_seconds <= RUN_TIMEOUT_SECONDS + 5.0 and
        abs(wall_seconds - monotonic_seconds) <= 5.0
    )
    errors: list[str] = []
    if not ordered:
        errors.append("stage wall-clock chronology is not claim/start/end/report ordered")
    if not wall_monotonic_valid:
        errors.append("gate wall/monotonic elapsed time mismatch")
    if not within_interval:
        errors.append("stage evidence exceeded the campaign interval")
    timed_out = execution["timed_out"]
    returncode = execution["returncode"]
    os_error_class = execution["os_error_class"]
    if ((timed_out and (returncode is not None or os_error_class is not None)) or
            (not timed_out and os_error_class is not None and
             returncode is not None) or
            (not timed_out and os_error_class is None and
             returncode is None)):
        errors.append("gate execution outcome tuple is impossible")
    return ordered and wall_monotonic_valid, within_interval, errors


def report_payload(
    evidence: dict[str, Any], claim: tuple[Path, str, dict[str, Any]],
    execution: tuple[Path, str, dict[str, Any]],
    after_stable: dict[str, Any] | None,
    *, postflight_error: dict[str, Any] | None = None,
    created_utc: str | None = None,
) -> dict[str, Any]:
    stage = claim[2]["stage"]
    parsed, validation_errors, threshold_errors = parse_execution_output(
        stage, execution[2]
    )
    stable = (
        after_stable is not None and
        stable_evidence(evidence) == after_stable and
        postflight_error is None
    )
    process_errors: list[str] = []
    created = created_utc or utc_now()
    chronology_valid, within_interval, semantic_errors = execution_semantics(
        claim, execution[2], created
    )
    process_errors.extend(semantic_errors)
    if execution[2].get("returncode") != 0:
        process_errors.append("gate returncode is not zero")
    if execution[2].get("timed_out") is not False:
        process_errors.append("gate timed out")
    if execution[2].get("os_error_class") is not None:
        process_errors.append("gate raised an OS error")
    stderr_record = execution[2].get("stderr")
    if not isinstance(stderr_record, dict) or stderr_record.get("empty") is not True:
        process_errors.append("gate stderr is not empty")
    if not stable:
        process_errors.append("source/configuration/binary evidence changed")
    if postflight_error is not None:
        validate_exception_record(postflight_error)
        process_errors.append("postflight evidence is unavailable or failed")
    acceptable = not (process_errors or validation_errors or threshold_errors)
    return {
        "schema": REPORT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "classification": "development-only-mask3-removal-stage-report",
        "candidate_identity": evidence["candidate_identity"],
        "candidate_source_commit": CANDIDATE_SOURCE_COMMIT,
        "admin_commit": evidence["admin"]["admin_commit"],
        "plan_sha256": PLAN_SHA256,
        "stage": stage,
        "stage_spec": stage_spec(stage),
        "claim": _record_reference(*claim),
        "execution": _record_reference(*execution),
        "prior_reports": claim[2]["prior_reports"],
        "preclaim_process": claim[2]["preclaim_process"],
        "evidence_before": stable_evidence(evidence),
        "evidence_after": after_stable,
        "evidence_before_sha256": evidence_digest(evidence),
        "evidence_after_sha256": (
            None if after_stable is None else
            sha256_bytes(canonical_json(after_stable))
        ),
        "postflight_error": postflight_error,
        "stage_chronology_valid": chronology_valid,
        "stage_within_campaign_interval": within_interval,
        "stable_evidence": stable,
        "parsed": parsed,
        "process_errors": process_errors,
        "validation_errors": validation_errors,
        "selection_threshold_errors": threshold_errors,
        "development_selection_acceptable": acceptable,
        "final_qualification": False,
        "arena_authorization": False,
        "protected_bank_files_accessed": [],
        "created_utc": created,
    }


def persist_report(
    evidence: dict[str, Any], claim: tuple[Path, str, dict[str, Any]],
    execution: tuple[Path, str, dict[str, Any]],
    *, recovered_execution: bool = False,
) -> tuple[Path, str, dict[str, Any]]:
    if recovered_execution:
        after = None
        postflight_error = interrupted_postflight_record()
    else:
        postflight_error = None
        try:
            after = prepare_evidence()
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            after = None
            postflight_error = exception_record(error)
    payload = report_payload(
        evidence, claim, execution,
        None if after is None else stable_evidence(after),
        postflight_error=postflight_error,
    )
    path, digest = persist_content_addressed(REPORTS, payload)
    return path, digest, payload


def validate_report(
    path: Path, evidence: dict[str, Any],
    claims: dict[str, tuple[Path, str, dict[str, Any]]],
    executions: dict[str, tuple[Path, str, dict[str, Any]]],
    *, raw: bytes | None = None,
    expected_inode: tuple[int, int] | None = None,
) -> tuple[Path, str, dict[str, Any]]:
    if raw is None:
        raw = read_regular_record(path, expected_inode=expected_inode)
    digest = sha256_bytes(raw)
    report = decode_json(raw)
    expected_keys = {
        "admin_commit", "arena_authorization", "campaign_id",
        "candidate_identity", "candidate_source_commit", "claim",
        "classification", "created_utc", "development_selection_acceptable",
        "evidence_after", "evidence_after_sha256", "evidence_before",
        "evidence_before_sha256", "execution", "final_qualification", "parsed",
        "plan_sha256", "postflight_error", "preclaim_process", "prior_reports", "process_errors",
        "protected_bank_files_accessed", "schema", "selection_threshold_errors",
        "stable_evidence", "stage", "stage_chronology_valid", "stage_spec",
        "stage_within_campaign_interval", "validation_errors",
    }
    if (path.name != f"{digest}.json" or not isinstance(report, dict) or
            set(report) != expected_keys or
            canonical_json(report) != raw or report.get("schema") != REPORT_SCHEMA or
            report.get("campaign_id") != CAMPAIGN_ID or
            report.get("candidate_identity") != evidence["candidate_identity"] or
            report.get("candidate_source_commit") != CANDIDATE_SOURCE_COMMIT or
            report.get("admin_commit") != evidence["admin"]["admin_commit"] or
            report.get("plan_sha256") != PLAN_SHA256 or
            report.get("stage") not in STAGE_NAMES or
            report.get("final_qualification") is not False or
            report.get("arena_authorization") is not False or
            report.get("protected_bank_files_accessed") != []):
        raise ValueError("persisted stage report provenance mismatch")
    stage = report["stage"]
    claim = claims.get(stage)
    execution = executions.get(stage)
    if claim is None or execution is None:
        raise ValueError("stage report lacks its claim or execution receipt")
    postflight_error = report.get("postflight_error")
    if postflight_error is not None:
        validate_exception_record(postflight_error)
    persisted_after = report.get("evidence_after")
    persisted_after_sha256 = report.get("evidence_after_sha256")
    if postflight_error is not None:
        if persisted_after is not None or persisted_after_sha256 is not None:
            raise ValueError("postflight failure unexpectedly retains after evidence")
    elif (not isinstance(persisted_after, dict) or
          set(persisted_after) != set(stable_evidence(evidence)) or
          persisted_after_sha256 != sha256_bytes(canonical_json(persisted_after))):
        raise ValueError("persisted after-evidence map/digest mismatch")
    if (report.get("preclaim_process") != claim[2]["preclaim_process"] or
            report.get("evidence_before") != stable_evidence(evidence) or
            report.get("evidence_before_sha256") != evidence_digest(evidence) or
            not isinstance(report.get("created_utc"), str)):
        raise ValueError("stage report evidence/timestamp chain mismatch")
    recomputed = report_payload(
        evidence, claim, execution,
        persisted_after,
        postflight_error=postflight_error,
        created_utc=report["created_utc"],
    )
    if report != recomputed:
        raise ValueError("persisted stage report semantics do not replay exactly")
    return path, digest, report


def _scan_state(evidence: dict[str, Any]) -> dict[str, Any]:
    identifier = evidence["candidate_identity"]
    claim_pattern = re.compile(
        rf"{re.escape(identifier)}\.({'|'.join(map(re.escape, STAGE_NAMES))})\.json"
    )
    execution_pattern = claim_pattern
    content_pattern = re.compile(r"[0-9a-f]{64}\.json")
    # Capture and validate every registry entry before reading any record.  Each
    # subsequent descriptor read is pinned to the captured inode, so a path
    # replacement cannot redirect a semantic read to an unapproved file.
    claim_snapshot = _registry_snapshot(CLAIMS, claim_pattern)
    execution_snapshot = _registry_snapshot(EXECUTIONS, execution_pattern)
    report_snapshot = _registry_snapshot(REPORTS, content_pattern)
    decision_snapshot = _registry_snapshot(DECISIONS, content_pattern)
    claims: dict[str, tuple[Path, str, dict[str, Any]]] = {}
    for path, inode in claim_snapshot:
        raw = read_regular_record(path, expected_inode=inode)
        claim, digest = load_claim(path, evidence, raw=raw)
        stage = claim["stage"]
        if stage in claims:
            raise ValueError("multiple claims exist for a stage")
        claims[stage] = (path, digest, claim)
    executions: dict[str, tuple[Path, str, dict[str, Any]]] = {}
    for path, inode in execution_snapshot:
        match = claim_pattern.fullmatch(path.name)
        if match is None:
            raise ValueError("execution receipt filename is outside stage registry")
        stage = match.group(1)
        claim = claims.get(stage)
        if claim is None:
            raise ValueError("execution receipt exists without a claim")
        raw = read_regular_record(path, expected_inode=inode)
        payload, digest = load_execution(path, evidence, claim, raw=raw)
        executions[stage] = (path, digest, payload)
    reports: dict[str, tuple[Path, str, dict[str, Any]]] = {}
    # First bind report stages, then replay once every claim/execution is known.
    for path, inode in report_snapshot:
        raw = read_regular_record(path, expected_inode=inode)
        payload = decode_json(raw)
        if not isinstance(payload, dict) or payload.get("stage") not in STAGE_NAMES:
            raise ValueError("foreign report exists in single-candidate registry")
        stage = payload["stage"]
        if stage in reports:
            raise ValueError("multiple reports exist for a stage")
        reports[stage] = validate_report(
            path, evidence, claims, executions, raw=raw
        )
    decisions: list[tuple[Path, str, dict[str, Any]]] = []
    for path, inode in decision_snapshot:
        raw = read_regular_record(path, expected_inode=inode)
        digest = sha256_bytes(raw)
        payload = decode_json(raw)
        if (path.name != f"{digest}.json" or not isinstance(payload, dict) or
                canonical_json(payload) != raw or payload.get("schema") != DECISION_SCHEMA or
                payload.get("campaign_id") != CAMPAIGN_ID or
                payload.get("candidate_identity") != identifier):
            raise ValueError("foreign decision exists in single-candidate registry")
        decisions.append((path, digest, payload))
    if len(decisions) > 1:
        raise ValueError("multiple mask3-removal decisions exist")
    for index, stage in enumerate(STAGE_NAMES):
        if stage in executions and stage not in claims:
            raise ValueError("execution exists without stage claim")
        if stage in reports and stage not in executions:
            raise ValueError("report exists without execution receipt")
        if index:
            predecessor = STAGE_NAMES[index - 1]
            if stage in claims and (
                predecessor not in reports or
                reports[predecessor][2]["development_selection_acceptable"] is not True
            ):
                raise ValueError("later stage was claimed without accepted predecessor")
        claim = claims.get(stage)
        if claim is not None:
            expected_prior = [
                _record_reference(*reports[prior])
                for prior in STAGE_NAMES[:index]
            ]
            if claim[2]["prior_reports"] != expected_prior:
                raise ValueError("stage claim predecessor report chain mismatch")
            predecessor_times = [
                parse_utc(reports[prior][2]["created_utc"])
                for prior in STAGE_NAMES[:index]
            ]
            if (predecessor_times and
                    parse_utc(claim[2]["preclaim_process"]["checked_utc"]) <
                    max(predecessor_times)):
                raise ValueError(
                    "stage preclaim process check predates a predecessor report"
                )
        if stage in reports and reports[stage][2]["development_selection_acceptable"] is False:
            for later in STAGE_NAMES[index + 1:]:
                if later in claims or later in executions or later in reports:
                    raise ValueError("later stage exists after a terminal rejection")
    return {
        "claims": claims,
        "executions": executions,
        "reports": reports,
        "decisions": decisions,
    }


def decision_payload(
    evidence: dict[str, Any],
    reports: dict[str, tuple[Path, str, dict[str, Any]]],
    *, spent_claim: tuple[Path, str, dict[str, Any]] | None = None,
    created_utc: str | None = None,
) -> dict[str, Any]:
    ordered_reports = [reports[stage] for stage in STAGE_NAMES if stage in reports]
    rejected = next((item for item in ordered_reports
                     if item[2]["development_selection_acceptable"] is False), None)
    created = created_utc or utc_now()
    created_time = parse_utc(created)
    lower = parse_utc(CAMPAIGN_T0_UTC)
    upper = parse_utc(CAMPAIGN_DEADLINE_UTC)
    predecessor_times = [
        parse_utc(item[2]["created_utc"]) for item in ordered_reports
    ]
    if spent_claim is not None:
        predecessor_times.append(parse_utc(spent_claim[2]["claimed_utc"]))
    report_chronology_valid = all(
        item[2].get("stage_chronology_valid") is True
        for item in ordered_reports
    )
    report_intervals_valid = all(
        item[2].get("stage_within_campaign_interval") is True
        for item in ordered_reports
    )
    decision_chronology_valid = (
        created_time >= lower and
        all(created_time >= predecessor for predecessor in predecessor_times) and
        report_chronology_valid
    )
    decision_within_deadline = (
        lower <= created_time <= upper and report_intervals_valid
    )
    selected = (
        len(ordered_reports) == len(STAGES) and rejected is None and
        spent_claim is None and decision_chronology_valid and
        decision_within_deadline
    )
    if spent_claim is None and not ordered_reports:
        raise ValueError("decision cannot be formed from an incomplete sequence")
    if not decision_chronology_valid:
        status = "terminal-development-clock-rollback-rejection"
        terminal_stage = (
            spent_claim[2]["stage"] if spent_claim is not None else
            ordered_reports[-1][2]["stage"]
        )
    elif spent_claim is not None:
        status = "blocked-spent-stage-without-execution"
        terminal_stage = spent_claim[2]["stage"]
    elif not decision_within_deadline:
        status = "terminal-development-deadline-rejection"
        terminal_stage = ordered_reports[-1][2]["stage"]
    elif rejected is not None:
        status = "terminal-development-rejection"
        terminal_stage = rejected[2]["stage"]
    elif selected:
        status = "selected-for-source-activation-testing-only"
        terminal_stage = STAGE_NAMES[-1]
    else:
        raise ValueError("decision cannot be formed from an incomplete live sequence")
    return {
        "schema": DECISION_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "classification": "development-only-mask3-removal-decision",
        "candidate_identity": evidence["candidate_identity"],
        "candidate_source_commit": CANDIDATE_SOURCE_COMMIT,
        "admin_commit": evidence["admin"]["admin_commit"],
        "plan_sha256": PLAN_SHA256,
        "status": status,
        "terminal_stage": terminal_stage,
        "reports": [_record_reference(*item) for item in ordered_reports],
        "spent_claim": None if spent_claim is None else _record_reference(*spent_claim),
        "selected_for_source_activation_testing": selected,
        "heldout_qualification": False,
        "fresh_bank_campaign_authorization": False,
        "arena_authorization": False,
        "retry_authorized": False,
        "decision_chronology_valid": decision_chronology_valid,
        "decision_within_deadline": decision_within_deadline,
        "protected_bank_files_accessed": [],
        "created_utc": created,
    }


def persist_decision(
    evidence: dict[str, Any],
    reports: dict[str, tuple[Path, str, dict[str, Any]]],
    *, spent_claim: tuple[Path, str, dict[str, Any]] | None = None,
) -> tuple[Path, str, dict[str, Any]]:
    payload = decision_payload(evidence, reports, spent_claim=spent_claim)
    path, digest = persist_content_addressed(DECISIONS, payload)
    decision = (path, digest, payload)
    validate_decision(decision, evidence, {
        "claims": (
            {} if spent_claim is None else
            {spent_claim[2]["stage"]: spent_claim}
        ),
        "executions": {},
        "reports": reports,
        "decisions": [decision],
    })
    return decision


def finalize_decision(
    evidence: dict[str, Any],
    reports: dict[str, tuple[Path, str, dict[str, Any]]],
    *, spent_claim: tuple[Path, str, dict[str, Any]] | None = None,
) -> tuple[Path, str, dict[str, Any]]:
    persisted = persist_decision(evidence, reports, spent_claim=spent_claim)
    state = _scan_state(evidence)
    if state["decisions"] != [persisted]:
        raise ValueError("new decision did not rescan as the sole exact decision")
    validate_decision(persisted, evidence, state)
    return persisted


def validate_decision(
    decision: tuple[Path, str, dict[str, Any]], evidence: dict[str, Any],
    state: dict[str, Any],
) -> None:
    payload = decision[2]
    expected_keys = {
        "admin_commit", "arena_authorization", "campaign_id",
        "candidate_identity", "candidate_source_commit", "classification",
        "created_utc", "decision_within_deadline", "fresh_bank_campaign_authorization",
        "decision_chronology_valid",
        "heldout_qualification", "plan_sha256", "protected_bank_files_accessed",
        "reports", "retry_authorized", "schema",
        "selected_for_source_activation_testing", "spent_claim", "status",
        "terminal_stage",
    }
    if (set(payload) != expected_keys or not isinstance(payload.get("created_utc"), str)):
        raise ValueError("persisted decision schema mismatch")
    spent = None
    if payload.get("spent_claim") is not None:
        stage = payload.get("terminal_stage")
        spent = state["claims"].get(stage)
        if spent is None or payload["spent_claim"] != _record_reference(*spent):
            raise ValueError("decision spent-claim reference mismatch")
    expected = decision_payload(
        evidence, state["reports"], spent_claim=spent,
        created_utc=payload.get("created_utc"),
    )
    if payload != expected:
        raise ValueError("persisted decision semantics do not replay exactly")


def run_stage(
    evidence: dict[str, Any], stage: str,
    prior_reports: list[tuple[Path, str, dict[str, Any]]],
    state: dict[str, Any],
) -> tuple[Path, str, dict[str, Any]]:
    claim = state["claims"].get(stage)
    execution = state["executions"].get(stage)
    report = state["reports"].get(stage)
    if report is not None:
        return report
    if claim is not None and execution is None:
        raise RuntimeError("spent-stage-claim-without-execution")
    execution_created_now = False
    if claim is None:
        require_before_deadline()
        refreshed = prepare_evidence()
        if stable_evidence(refreshed) != stable_evidence(evidence):
            raise ValueError("complete evidence changed immediately before claim")
        # This O_EXCL claim is the last operation before the gate can see banks.
        claim = create_claim(refreshed, stage, prior_reports)
        result = execute_gate(command_for(stage))
        execution = persist_execution(refreshed, stage, claim, result)
        execution_created_now = True
    if execution is None:
        raise RuntimeError("stage execution receipt is unexpectedly absent")
    return persist_report(
        evidence, claim, execution,
        recovered_execution=not execution_created_now,
    )


def run_campaign() -> tuple[Path, str, dict[str, Any]]:
    validate_output_topology()
    held_locks: list[tuple[Path, int]] = []
    try:
        for path in CAMPAIGN_LOCKS:
            try:
                held_locks.append((path, open_exclusive_lock(path)))
            except BlockingIOError as error:
                raise ValueError(f"benchmark/clock lock is busy: {path}") from error
        # Earlier lock pathnames can change while later locks are being opened;
        # bind every pathname back to its held inode before any evidence work.
        for path, descriptor in held_locks:
            validate_held_lock(path, descriptor)
        prepare_runtime_directories()
        validate_output_topology()
        evidence = prepare_evidence()
        repair_registry_durability(evidence)
        state = _scan_state(evidence)
        if state["decisions"]:
            decision = state["decisions"][0]
            validate_decision(decision, evidence, state)
            return decision
        accepted: list[tuple[Path, str, dict[str, Any]]] = []
        for stage in STAGE_NAMES:
            claim = state["claims"].get(stage)
            if claim is not None and stage not in state["executions"]:
                return finalize_decision(
                    evidence, state["reports"], spent_claim=claim
                )
            try:
                report = run_stage(evidence, stage, accepted, state)
            except RuntimeError as error:
                if str(error) != "spent-stage-claim-without-execution":
                    raise
                claim = state["claims"][stage]
                return finalize_decision(
                    evidence, state["reports"], spent_claim=claim
                )
            state = _scan_state(evidence)
            report = state["reports"][stage]
            accepted.append(report)
            if report[2]["development_selection_acceptable"] is not True:
                return finalize_decision(evidence, state["reports"])
        return finalize_decision(evidence, state["reports"])
    finally:
        for _, descriptor in reversed(held_locks):
            os.close(descriptor)


def audit_campaign() -> dict[str, Any]:
    validate_output_topology()
    evidence = prepare_evidence()
    state = _scan_state(evidence)
    if state["decisions"]:
        validate_decision(state["decisions"][0], evidence, state)
    return {
        "candidate_identity": evidence["candidate_identity"],
        "claims": sorted(state["claims"]),
        "executions": sorted(state["executions"]),
        "reports": sorted(state["reports"]),
        "decision_count": len(state["decisions"]),
        "protected_bank_files_accessed": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("audit", "run"))
    args = parser.parse_args()
    try:
        if args.action == "audit":
            print(canonical_json(audit_campaign()).decode("ascii"), end="")
            return 0
        path, digest, decision = run_campaign()
        print(path.relative_to(ROOT).as_posix())
        print(f"sha256={digest}")
        print(f"status={decision['status']}")
        return 0 if decision["selected_for_source_activation_testing"] else 1
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
