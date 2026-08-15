#!/usr/bin/env python3
"""One-shot DEVELOPMENT recorder for the Rank-4 Jacek TT collision ablation.

This module is deliberately standalone and standard-library only.  The command
surface is exactly ``audit`` or ``run``; importing it performs no I/O.  All
filesystem inputs used by the producer are descriptor-read and the campaign
registries are append-only, content-addressed canonical JSON.
"""

import datetime
import errno
import fcntl
import functools
import hashlib
import json
import math
import os
import re
import resource
import selectors
import signal
import stat
import struct
import subprocess
import sys
import time
from decimal import Decimal, InvalidOperation


sys.dont_write_bytecode = True

BASE_PLAN_COMMIT = "253699cda1b403c0437a24fc9de9b2ed2437bd6b"
BASE_PLAN_SHA256 = "c769c4ee646eead1963bbb1b02d1ca4ff5b0d4406050e60896d9d13772b3e195"
BASE_PLAN_PATH = "results/rank_4_jacek_hybrid/gates/tt_exact_collision/PLAN.json"
BASE_PLAN_SCHEMA = "rank4-jacek-hybrid-tt-exact-collision-development-plan-v1"
GENERATED_OVERLAY_PATH = (
    "results/rank_4_jacek_hybrid/gates/"
    "tt_exact_collision_generated_v19/PLAN.json"
)
GENERATED_OVERLAY_SCHEMA = (
    "rank4-jacek-hybrid-tt-exact-collision-generated-v19-development-overlay-v1"
)
# Filled by the root agent immediately after the overlay-only Plan commit.
GENERATED_PLAN_COMMIT = "ee7b01066134ba7c32aeeb9468d72105f4fae4b2"
GENERATED_OVERLAY_SHA256 = (
    "f9474c8bac2d9692928083377d876d22a1722c88a198c2523fad39dd8db76e91"
)

# The invocation contract is v19 from process start, so outer-environment and
# self/HEAD bootstrap checks cannot accidentally use a predecessor runtime.
# The frozen v1 base is loaded later using only the explicit BASE_* constants.
PLAN_COMMIT = GENERATED_PLAN_COMMIT
PLAN_SHA256 = GENERATED_OVERLAY_SHA256
PLAN_BYTES = None
PLAN_PATH = GENERATED_OVERLAY_PATH
RECORDER_PATH = "tools/record_rank4_jacek_hybrid_tt_exact_collision.py"
TEST_PATH = "tests/codingame/test_rank4_jacek_hybrid_tt_exact_collision.py"
PLAN_SCHEMA = GENERATED_OVERLAY_SCHEMA
CAMPAIGN_SCHEMA_PREFIX = (
    "rank4-jacek-hybrid-tt-exact-collision-generated-v19-"
)
CAMPAIGN_T0 = "2026-08-13T19:15:07Z"
CAMPAIGN_DEADLINE = "2026-08-15T07:15:07Z"
GIT = "/Library/Developer/CommandLineTools/usr/bin/git"
PYTHON_RUNTIME_ROOT = (
    "/Library/Developer/CommandLineTools/Library/Frameworks/"
    "Python3.framework/Versions/3.9"
)
PYTHON_RUNTIME_MANIFEST_SHA256 = (
    "2b5e70048566748b07d05ec767caf7797a09597c3a9941196da9072a41de2fb6"
)
PYTHON_RUNTIME_RECORD_COUNT = 2006
PYTHON_RUNTIME_DIRECTORY_COUNT = 186
PYTHON_RUNTIME_REGULAR_COUNT = 1810
PYTHON_RUNTIME_SYMLINK_COUNT = 10
PYTHON_RUNTIME_MANIFEST_BYTES = 392087
PYTHON_RUNTIME_TOTAL_RECORD_BYTES = 48042195
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
CAMPAIGN_ROOT = os.path.join(
    ROOT, "results/rank_4_jacek_hybrid/gates/tt_exact_collision_generated_v19"
)
BUILD_PARENT_REL = "build/rank4-jacek-hybrid-tt-exact-collision-generated-v19"
BUILD_ROOT_REL = BUILD_PARENT_REL + "/clang-release"
TMP_REL = BUILD_ROOT_REL + "/tmp"
GENERATED_OVERLAY_ACTIVE = True
HEX40_RE = re.compile(r"\A[0-9a-f]{40}\Z")
HEX64_RE = re.compile(r"\A[0-9a-f]{64}\Z")
SECOND_UTC_RE = re.compile(
    r"\A[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z"
)
MICRO_UTC_RE = re.compile(
    r"\A[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:"
    r"[0-9]{2}\.[0-9]{6}Z\Z"
)
SAFE_ASCII_RE = re.compile(r"\A[ -~]*\Z")
CANONICAL_UINT_RE = re.compile(r"\A(?:0|[1-9][0-9]*)\Z")
PREEXECUTION_FAILURE_LABEL_RE = re.compile(r"\A[a-z0-9-]{1,64}\Z")
FIXED3_RE = re.compile(r"\A(?:0|[1-9][0-9]*)\.[0-9]{3}\Z")
PROTECTED_COMPONENTS = frozenset(
    ("openings", "validation", "final", "heldout", "arena")
)
RECORD_SCHEMAS = {
    "preexecution_claim": CAMPAIGN_SCHEMA_PREFIX + "preexecution-claim-v1",
    "preexecution_receipt": CAMPAIGN_SCHEMA_PREFIX + "preexecution-receipt-v1",
    "claim": CAMPAIGN_SCHEMA_PREFIX + "claim-v1",
    "execution": CAMPAIGN_SCHEMA_PREFIX + "execution-v1",
    "report": CAMPAIGN_SCHEMA_PREFIX + "report-v1",
    "decision": CAMPAIGN_SCHEMA_PREFIX + "decision-v1",
}


def _record_schemas(prefix):
    return {
        "preexecution_claim": prefix + "preexecution-claim-v1",
        "preexecution_receipt": prefix + "preexecution-receipt-v1",
        "claim": prefix + "claim-v1",
        "execution": prefix + "execution-v1",
        "report": prefix + "report-v1",
        "decision": prefix + "decision-v1",
    }
STAGE_ORDER = (
    "stage0_public_truth_activation_timing",
    "development_d20",
    "development_d04_d08_d12",
)
REGISTRY_ORDER = (
    "preexecution_claims",
    "preexecution_receipts",
    "claims",
    "executions",
    "reports",
    "decisions",
)
_ACTIVE_HELPER_MONOTONIC_DEADLINE = None
_STABLE_TREE_RECORDS_CACHE = {}
REGISTRY_RELATIVE = {
    "preexecution_claims": "preexecution/claims",
    "preexecution_receipts": "preexecution/receipts",
    "claims": "claims",
    "executions": "executions",
    "reports": "reports",
    "decisions": "decisions",
}
ENVIRONMENT_KEYS = (
    "GIT_ATTR_NOSYSTEM",
    "GIT_CONFIG_GLOBAL",
    "GIT_CONFIG_NOSYSTEM",
    "GIT_NO_LAZY_FETCH",
    "GIT_NO_REPLACE_OBJECTS",
    "GIT_OPTIONAL_LOCKS",
    "LANG",
    "LC_ALL",
    "PATH",
    "PYTHONDONTWRITEBYTECODE",
    "TMPDIR",
    "TZ",
)
EXPECTED_ENVIRONMENT = {
    "GIT_ATTR_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_NO_LAZY_FETCH": "1",
    "GIT_NO_REPLACE_OBJECTS": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
    "PYTHONDONTWRITEBYTECODE": "1",
    "TMPDIR": os.path.join(ROOT, TMP_REL),
    "TZ": "UTC",
}
MACOS_PYTHON_BOOTSTRAP_ENVIRONMENT = {
    "__CF_USER_TEXT_ENCODING": "0x1F5:0x0:0x0",
}
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
GIT_METADATA_FILES = (
    (".git", 116, "0644", "efd26b1f6dc157acbe4f3cd8b3f5249cf0a5ed6e5297adeb691d5dd20f5fad2d"),
    ("/Users/lecorbio/Desktop/projects/paper-soccer-strategy-engine/.git/config", 715, "0644", "34817c9b0bc303d41314687801a2a6c7a69836248d70e57439c9deafb335621c"),
    ("/Users/lecorbio/Desktop/projects/paper-soccer-strategy-engine/.git/worktrees/paper-soccer-strategy-engine14/config.worktree", 47, "0644", "da89256d91364984591e686f641271bbbf63fc48a9e6add5690bf9979b795529"),
    ("/Users/lecorbio/Desktop/projects/paper-soccer-strategy-engine/.git/worktrees/paper-soccer-strategy-engine14/commondir", 6, "0644", "340ddcb67a6204f742cd1e28e5b462622dde7daaa8ee36001897196aacdc6d47"),
    ("/Users/lecorbio/Desktop/projects/paper-soccer-strategy-engine/.git/worktrees/paper-soccer-strategy-engine14/gitdir", 72, "0644", "9daf60646619912875e6cfe62ab0581a200da4ace002f472c59c2c98f9e9afc8"),
)
GIT_METADATA_ABSENT = (
    "/Users/lecorbio/Desktop/projects/paper-soccer-strategy-engine/.git/info/attributes",
    "/Users/lecorbio/Desktop/projects/paper-soccer-strategy-engine/.git/info/grafts",
    "/Users/lecorbio/Desktop/projects/paper-soccer-strategy-engine/.git/objects/info/alternates",
    "/Users/lecorbio/Desktop/projects/paper-soccer-strategy-engine/.git/packed-refs",
    "/Users/lecorbio/Desktop/projects/paper-soccer-strategy-engine/.git/refs/replace",
    "/Users/lecorbio/Desktop/projects/paper-soccer-strategy-engine/.git/shallow",
    "/Users/lecorbio/Desktop/projects/paper-soccer-strategy-engine/.git/worktrees/paper-soccer-strategy-engine14/info/attributes",
)


class ContractError(Exception):
    """A sanitized, fail-closed contract violation."""


class ProcessSnapshotError(ContractError):
    """A failed ps observation retaining its exact bounded child outcome."""

    def __init__(self, reason, result=None, checked_utc=None):
        super().__init__(reason)
        self.result = result
        self.checked_utc = checked_utc


def require(condition, reason):
    if not condition:
        raise ContractError(reason)


def canonical_json(value):
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise ContractError("canonical-json") from error
    return text.encode("utf-8") + b"\n"


def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def _reject_constant(_value):
    raise ContractError("json-nonfinite")


def _object_no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ContractError("json-duplicate-key")
        result[key] = value
    return result


def decode_json(raw):
    require(isinstance(raw, bytes), "json-bytes")
    require(b"\x00" not in raw and b"\r" not in raw, "json-control-byte")
    try:
        text = raw.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_object_no_duplicates,
            parse_constant=_reject_constant,
        )
    except ContractError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError("json-decode") from error
    require(canonical_json(value) == raw, "json-noncanonical")
    return value


def exact_keys(value, keys, reason="record-keys"):
    require(isinstance(value, dict), reason)
    require(set(value) == set(keys) and len(value) == len(keys), reason)


def parse_timestamp(value, runtime=True):
    require(isinstance(value, str), "timestamp-type")
    pattern = MICRO_UTC_RE if runtime else SECOND_UTC_RE
    require(pattern.fullmatch(value) is not None, "timestamp-grammar")
    fmt = "%Y-%m-%dT%H:%M:%S.%fZ" if runtime else "%Y-%m-%dT%H:%M:%SZ"
    try:
        parsed = datetime.datetime.strptime(value, fmt).replace(
            tzinfo=datetime.timezone.utc
        )
    except ValueError as error:
        raise ContractError("timestamp-value") from error
    require(parsed.strftime(fmt) == value, "timestamp-roundtrip")
    return parsed


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )


def seconds_until_deadline(now_value=None):
    now = parse_timestamp(now_value, True) if now_value else datetime.datetime.now(
        datetime.timezone.utc
    )
    return (parse_timestamp(CAMPAIGN_DEADLINE, False) - now).total_seconds()


def _timing_value(plan, key, legacy):
    if not GENERATED_OVERLAY_ACTIVE:
        return legacy
    timing = plan["timing"]
    value = timing[key]
    require(type(value) is int and value > 0, "generated-timing-value")
    return value


def _preexecution_aggregate_seconds(plan):
    return _timing_value(
        plan, "preexecution_aggregate_timeout_seconds", 7200,
    )


def _initial_reserve_seconds(plan):
    return _timing_value(plan, "initial_reserve_seconds", 11100)


def _stage_aggregate_seconds(plan):
    return _timing_value(plan, "stage_aggregate_timeout_seconds", 3600)


def _postflight_reserve_seconds(plan):
    return _timing_value(plan, "postflight_reserve_seconds", 300)


def _continuation_reserve_seconds(plan):
    return _timing_value(plan, "continuation_reserve_seconds", 3900)


def canonical_relative_path(value, allow_protected_literal=False):
    require(isinstance(value, str) and value != "", "path-type")
    require("\x00" not in value and len(value.encode("utf-8")) <= 4096, "path-size")
    require(not value.startswith("/") and "\\" not in value, "path-relative")
    components = value.split("/")
    require(all(part not in ("", ".", "..") for part in components), "path-component")
    require(value == "/".join(components), "path-canonical")
    if not allow_protected_literal:
        require(not value.endswith(".tsv"), "protected-path")
        require(not any(part in PROTECTED_COMPONENTS for part in components), "protected-path")
    return value


def canonical_absolute_path(value, allow_runtime=False):
    require(isinstance(value, str) and value.startswith("/"), "absolute-path")
    require("\x00" not in value and len(value.encode("utf-8")) <= 4096, "path-size")
    components = value.split("/")[1:]
    require(all(part not in ("", ".", "..") for part in components), "path-component")
    require(value == "/" + "/".join(components), "path-canonical")
    if not allow_runtime:
        require(not value.endswith(".tsv"), "protected-path")
    return value


def root_path(relative, allow_protected_literal=False):
    relative = canonical_relative_path(relative, allow_protected_literal)
    return os.path.join(ROOT, *relative.split("/"))


def mode_string(metadata):
    return format(stat.S_IMODE(metadata.st_mode), "04o")


def _directory_flags():
    require(hasattr(os, "O_DIRECTORY") and hasattr(os, "O_NOFOLLOW"), "directory-flags")
    return os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW


def _regular_flags(access=os.O_RDONLY):
    require(hasattr(os, "O_NOFOLLOW"), "regular-flags")
    return access | os.O_NONBLOCK | os.O_CLOEXEC | os.O_NOFOLLOW


def _metadata_identity(metadata):
    return (
        metadata.st_dev, metadata.st_ino, metadata.st_mode,
        metadata.st_nlink, metadata.st_size,
    )


def _same_node(left, right):
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _open_directory_component(parent_fd, name, expected_mode=None, missing_ok=False):
    require(
        isinstance(name, str) and name not in ("", ".", "..")
        and "/" not in name and "\x00" not in name,
        "directory-component",
    )
    try:
        before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        if missing_ok:
            return None
        raise ContractError("directory-missing")
    except (OSError, ValueError) as error:
        raise ContractError("directory-stat") from error
    require(stat.S_ISDIR(before.st_mode), "directory-type")
    try:
        descriptor = os.open(name, _directory_flags(), dir_fd=parent_fd)
    except (OSError, ValueError) as error:
        raise ContractError("directory-open") from error
    try:
        opened = os.fstat(descriptor)
        after = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        require(
            stat.S_ISDIR(opened.st_mode) and stat.S_ISDIR(after.st_mode)
            and _same_node(before, opened) and _same_node(opened, after),
            "directory-race",
        )
        require(opened.st_nlink >= 2, "directory-nlink")
        if expected_mode is not None:
            require(mode_string(opened) == expected_mode, "directory-mode")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_absolute_directory(path, expected_mode=None, missing_ok=False):
    if path != "/":
        path = canonical_absolute_path(path, allow_runtime=True)
    descriptor = None
    try:
        before = os.lstat("/")
        descriptor = os.open("/", _directory_flags())
        opened = os.fstat(descriptor)
        after = os.lstat("/")
    except OSError as error:
        if descriptor is not None:
            os.close(descriptor)
        raise ContractError("root-directory-open") from error
    try:
        require(
            stat.S_ISDIR(before.st_mode) and stat.S_ISDIR(opened.st_mode)
            and stat.S_ISDIR(after.st_mode) and _same_node(before, opened)
            and _same_node(opened, after),
            "root-directory-race",
        )
        components = path.split("/")[1:]
        for index, component in enumerate(components):
            child = _open_directory_component(
                descriptor, component,
                expected_mode if index == len(components) - 1 else None,
                missing_ok=missing_ok,
            )
            if child is None:
                os.close(descriptor)
                return None
            os.close(descriptor)
            descriptor = child
        if not components and expected_mode is not None:
            require(mode_string(os.fstat(descriptor)) == expected_mode, "directory-mode")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _verify_absolute_directory_binding(path, descriptor):
    rebound = _open_absolute_directory(path)
    try:
        current = os.fstat(descriptor)
        observed = os.fstat(rebound)
        require(
            stat.S_ISDIR(current.st_mode) and stat.S_ISDIR(observed.st_mode)
            and _same_node(current, observed),
            "directory-path-drift",
        )
    finally:
        os.close(rebound)


def _open_parent_absolute(path):
    if not os.path.isabs(path):
        path = root_path(canonical_relative_path(path))
    path = canonical_absolute_path(path, allow_runtime=True)
    components = path.split("/")[1:]
    require(components, "leaf-path")
    parent_path = "/" if len(components) == 1 else "/" + "/".join(components[:-1])
    return _open_absolute_directory(parent_path), parent_path, components[-1]


def _read_regular_at(
    directory_fd, name, expected_mode=None, expected_nlink=1,
    max_bytes=None, directory_path=None,
):
    require(
        isinstance(name, str) and name not in ("", ".", "..")
        and "/" not in name and "\x00" not in name,
        "regular-name",
    )
    directory_before = os.fstat(directory_fd)
    try:
        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        require(stat.S_ISREG(before.st_mode), "regular-type")
        allowed_nlinks = (
            (expected_nlink,) if isinstance(expected_nlink, int)
            else tuple(expected_nlink)
        )
        require(before.st_nlink in allowed_nlinks, "regular-nlink")
        if expected_mode is not None:
            require(mode_string(before) == expected_mode, "regular-mode")
        if max_bytes is not None:
            require(before.st_size <= max_bytes, "regular-size-cap")
        descriptor = os.open(name, _regular_flags(), dir_fd=directory_fd)
    except (OSError, ValueError) as error:
        raise ContractError("regular-open") from error
    try:
        opened = os.fstat(descriptor)
        opened_identity = _metadata_identity(opened)
        after_open = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        require(
            stat.S_ISREG(before.st_mode) and stat.S_ISREG(opened.st_mode)
            and stat.S_ISREG(after_open.st_mode),
            "regular-type",
        )
        require(
            _same_node(before, opened) and _same_node(opened, after_open),
            "regular-race",
        )
        require(opened.st_nlink in allowed_nlinks, "regular-nlink")
        if expected_mode is not None:
            require(mode_string(opened) == expected_mode, "regular-mode")
        if max_bytes is not None:
            require(opened.st_size <= max_bytes, "regular-size-cap")
        chunks = []
        observed = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            observed += len(chunk)
            if max_bytes is not None:
                require(observed <= max_bytes, "regular-size-cap")
            chunks.append(chunk)
        raw = b"".join(chunks)
        require(os.read(descriptor, 1) == b"", "regular-eof")
        after_fd = os.fstat(descriptor)
        after_path = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        require(opened_identity == _metadata_identity(after_fd), "regular-drift")
        require(
            _metadata_identity(after_path) == opened_identity,
            "regular-path-drift",
        )
        directory_after = os.fstat(directory_fd)
        require(_same_node(directory_before, directory_after), "regular-parent-drift")
        if directory_path is not None:
            _verify_absolute_directory_binding(directory_path, directory_fd)
        require(len(raw) == opened.st_size, "regular-short-read")
        return raw, opened
    finally:
        os.close(descriptor)


def _reread_open_regular(
    descriptor, expected_raw, expected_mode, expected_nlink, expected_node=None,
):
    before = os.fstat(descriptor)
    require(
        stat.S_ISREG(before.st_mode) and mode_string(before) == expected_mode
        and before.st_nlink == expected_nlink
        and before.st_size == len(expected_raw),
        "regular-reread-metadata",
    )
    if expected_node is not None:
        require(_same_node(before, expected_node), "regular-reread-inode")
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks = []
    observed = 0
    while observed < len(expected_raw):
        chunk = os.read(descriptor, len(expected_raw) - observed)
        require(chunk, "regular-reread-short")
        chunks.append(chunk)
        observed += len(chunk)
    require(os.read(descriptor, 1) == b"", "regular-reread-eof")
    after = os.fstat(descriptor)
    require(
        b"".join(chunks) == expected_raw
        and _metadata_identity(before) == _metadata_identity(after),
        "regular-reread-drift",
    )
    return after


def read_regular_nofollow(path, expected_mode=None, expected_nlink=1, max_bytes=None):
    parent_fd, parent_path, leaf = _open_parent_absolute(path)
    try:
        return _read_regular_at(
            parent_fd, leaf, expected_mode, expected_nlink, max_bytes, parent_path,
        )
    finally:
        os.close(parent_fd)


def _path_exists_nofollow(path):
    if not os.path.isabs(path):
        path = root_path(canonical_relative_path(path))
    path = canonical_absolute_path(path, allow_runtime=True)
    components = path.split("/")[1:]
    require(components, "leaf-path")
    parent_path = "/" if len(components) == 1 else "/" + "/".join(components[:-1])
    parent_fd = _open_absolute_directory(parent_path, missing_ok=True)
    if parent_fd is None:
        return False
    try:
        try:
            os.stat(components[-1], dir_fd=parent_fd, follow_symlinks=False)
            present = True
        except FileNotFoundError:
            present = False
        except OSError as error:
            raise ContractError("path-existence") from error
        _verify_absolute_directory_binding(parent_path, parent_fd)
        return present
    finally:
        os.close(parent_fd)


def file_identity(path, label=None, expected_mode=None, expected_sha256=None, max_bytes=None):
    raw, metadata = read_regular_nofollow(path, expected_mode, 1, max_bytes)
    digest = sha256_bytes(raw)
    if expected_sha256 is not None:
        require(digest == expected_sha256, "regular-digest")
    return {
        "path": label if label is not None else path,
        "bytes": len(raw),
        "mode": mode_string(metadata),
        "sha256": digest,
    }, raw


def load_canonical_file(path, expected_mode, expected_sha256=None, max_bytes=None):
    identity, raw = file_identity(
        path, expected_mode=expected_mode, expected_sha256=expected_sha256,
        max_bytes=max_bytes,
    )
    return decode_json(raw), identity, raw


def _runtime_manifest_records():
    root = PYTHON_RUNTIME_ROOT
    root_fd = _open_absolute_directory(root, "0755")
    root_metadata = os.fstat(root_fd)
    require(root_metadata.st_nlink == 10, "python-runtime-root-nlink")
    require(
        root_metadata.st_uid == 0 and root_metadata.st_gid == 0,
        "python-runtime-owner",
    )
    records = []
    stack = [(root_fd, ".", root)]
    while stack:
        directory_fd, relative, absolute = stack.pop()
        metadata = os.fstat(directory_fd)
        base = {
            "path": relative,
            "type": "directory",
            "mode": mode_string(metadata),
            "nlink": metadata.st_nlink,
            "uid": metadata.st_uid,
            "gid": metadata.st_gid,
        }
        try:
            require(
                stat.S_ISDIR(metadata.st_mode)
                and metadata.st_uid == 0 and metadata.st_gid == 0,
                "python-runtime-owner",
            )
            records.append(base)
            names = _list_directory_fd(directory_fd, absolute)
            for name in sorted(names, key=lambda item: item.encode("utf-8"), reverse=True):
                child_relative = name if relative == "." else relative + "/" + name
                child_absolute = os.path.join(absolute, name)
                try:
                    child_metadata = os.stat(
                        name, dir_fd=directory_fd, follow_symlinks=False
                    )
                except OSError as error:
                    raise ContractError("python-runtime-stat") from error
                child_base = {
                    "path": child_relative,
                    "type": None,
                    "mode": mode_string(child_metadata),
                    "nlink": child_metadata.st_nlink,
                    "uid": child_metadata.st_uid,
                    "gid": child_metadata.st_gid,
                }
                require(
                    child_metadata.st_uid == 0 and child_metadata.st_gid == 0,
                    "python-runtime-owner",
                )
                if stat.S_ISDIR(child_metadata.st_mode):
                    child_fd = _open_directory_component(directory_fd, name)
                    stack.append((child_fd, child_relative, child_absolute))
                elif stat.S_ISREG(child_metadata.st_mode):
                    raw, opened = _read_regular_at(
                        directory_fd, name, expected_nlink=child_metadata.st_nlink,
                        directory_path=absolute,
                    )
                    require(
                        opened.st_uid == 0 and opened.st_gid == 0,
                        "python-runtime-owner",
                    )
                    child_base.update(
                        type="regular", bytes=len(raw), sha256=sha256_bytes(raw)
                    )
                    records.append(child_base)
                elif stat.S_ISLNK(child_metadata.st_mode):
                    try:
                        link_text = os.readlink(name, dir_fd=directory_fd)
                        after = os.stat(
                            name, dir_fd=directory_fd, follow_symlinks=False
                        )
                    except OSError as error:
                        raise ContractError("python-runtime-link") from error
                    require(
                        _metadata_identity(child_metadata)
                        == _metadata_identity(after),
                        "python-runtime-link-drift",
                    )
                    encoded = link_text.encode("utf-8")
                    child_base.update(
                        type="symlink", bytes=len(encoded), link_text=link_text
                    )
                    records.append(child_base)
                else:
                    raise ContractError("python-runtime-node")
            _verify_absolute_directory_binding(absolute, directory_fd)
        finally:
            os.close(directory_fd)
    records.sort(key=lambda item: item["path"].encode("utf-8"))
    return records


def validate_python_runtime_manifest():
    records = _runtime_manifest_records()
    raw = canonical_json(records)
    require(len(records) == PYTHON_RUNTIME_RECORD_COUNT, "python-runtime-count")
    require(
        sum(item["type"] == "directory" for item in records)
        == PYTHON_RUNTIME_DIRECTORY_COUNT,
        "python-runtime-directory-count",
    )
    require(
        sum(item["type"] == "regular" for item in records)
        == PYTHON_RUNTIME_REGULAR_COUNT,
        "python-runtime-regular-count",
    )
    require(
        sum(item["type"] == "symlink" for item in records)
        == PYTHON_RUNTIME_SYMLINK_COUNT,
        "python-runtime-symlink-count",
    )
    require(len(raw) == PYTHON_RUNTIME_MANIFEST_BYTES, "python-runtime-manifest-bytes")
    require(sha256_bytes(raw) == PYTHON_RUNTIME_MANIFEST_SHA256, "python-runtime-manifest")
    require(
        sum(item.get("bytes", 0) for item in records)
        == PYTHON_RUNTIME_TOTAL_RECORD_BYTES,
        "python-runtime-record-bytes",
    )
    return {
        "schema": (
            "rank4-jacek-hybrid-tt-exact-collision-"
            "python-runtime-manifest-v1"
        ),
        "root": PYTHON_RUNTIME_ROOT,
        "record_count": len(records),
        "directory_count": PYTHON_RUNTIME_DIRECTORY_COUNT,
        "regular_count": PYTHON_RUNTIME_REGULAR_COUNT,
        "symlink_count": PYTHON_RUNTIME_SYMLINK_COUNT,
        "manifest_bytes": len(raw),
        "manifest_sha256": sha256_bytes(raw),
        "total_record_bytes": PYTHON_RUNTIME_TOTAL_RECORD_BYTES,
        "uid": 0,
        "gid": 0,
        "root_mode": "0755",
        "root_nlink": 10,
        "canonicalization": (
            "walk the exact root component-by-component without following symlinks; "
            "include root as path '.' and every descendant in UTF-8 path-byte order. "
            "Directory record exact keys path,type='directory',mode,nlink,uid,gid. "
            "Regular record adds bytes,sha256; symlink record adds bytes,link_text. "
            "Encode the sorted array as compact sorted-key ensure_ascii JSON plus one LF"
        ),
        "validation": (
            "the physical root-owned Python3.9 terminal is launched directly with "
            "-I,-S,-B and no developer-tool dispatch"
        ),
    }


def stream_receipt(raw, cap, truncated=False, observed_bytes_at_least=None):
    require(isinstance(raw, bytes), "stream-bytes")
    if not truncated:
        require(len(raw) <= cap and observed_bytes_at_least is None, "stream-cap")
        return {"bytes": len(raw), "sha256": sha256_bytes(raw), "truncated": False}
    require(len(raw) == cap, "stream-prefix")
    require(
        isinstance(observed_bytes_at_least, int)
        and not isinstance(observed_bytes_at_least, bool)
        and observed_bytes_at_least >= cap + 1,
        "stream-observed",
    )
    return {
        "captured_prefix_bytes": len(raw),
        "captured_prefix_sha256": sha256_bytes(raw),
        "truncated": True,
        "observed_bytes_at_least": observed_bytes_at_least,
    }


def validate_stream_receipt(value, cap):
    exact_keys(value, ("bytes", "sha256", "truncated") if not value.get("truncated") else (
        "captured_prefix_bytes", "captured_prefix_sha256", "truncated", "observed_bytes_at_least"
    ), "stream-receipt-keys")
    if value["truncated"] is False:
        require(
            isinstance(value["bytes"], int) and not isinstance(value["bytes"], bool)
            and 0 <= value["bytes"] <= cap,
            "stream-receipt-bytes",
        )
        require(HEX64_RE.fullmatch(value["sha256"]) is not None, "stream-receipt-sha")
    else:
        require(value["truncated"] is True, "stream-receipt-truncated")
        require(value["captured_prefix_bytes"] == cap, "stream-receipt-prefix")
        require(HEX64_RE.fullmatch(value["captured_prefix_sha256"]) is not None, "stream-receipt-sha")
        require(value["observed_bytes_at_least"] >= cap + 1, "stream-receipt-observed")


def sanitized_reason(error):
    if isinstance(error, ContractError) and error.args:
        reason = str(error.args[0])
    else:
        reason = "internal-contract-error"
    reason = "".join(character if 32 <= ord(character) <= 126 else "?" for character in reason)
    reason = reason[:160]
    return reason if reason else "contract-error"


def sanitized_error_record(error):
    reason = sanitized_reason(error)
    return {"type": type(error).__name__, "message_sha256": sha256_bytes(reason.encode("ascii"))}


def _kill_process_group(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError as error:
        if error.errno != errno.ESRCH:
            raise


def run_bounded(argv, timeout_seconds, stdout_cap, stderr_cap, environment=None, cwd=None):
    require(isinstance(argv, list) and argv and all(isinstance(x, str) for x in argv), "child-argv")
    require(timeout_seconds > 0, "child-timeout")
    require(stdout_cap >= 0 and stderr_cap >= 0, "child-stream-cap")
    started_wall = utc_now()
    started_monotonic = time.monotonic()
    process = None
    cwd_fd = None
    saved_cwd_fd = None
    try:
        if cwd is not None:
            cwd_fd = _open_absolute_directory(cwd)
            saved_cwd_fd = os.open(".", _directory_flags())
            _verify_absolute_directory_binding(cwd, cwd_fd)
            os.fchdir(cwd_fd)
        process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # The child inherits the descriptor-bound directory selected
            # above.  Passing its pathname back to Popen would reopen it and
            # reintroduce an intermediate-component swap window.
            cwd=None,
            env=environment,
            shell=False,
            close_fds=True,
            start_new_session=True,
        )
    except OSError as error:
        ended_wall = utc_now()
        return {
            "started_utc": started_wall,
            "ended_utc": ended_wall,
            "elapsed_monotonic_seconds": max(0.0, time.monotonic() - started_monotonic),
            "returncode": None,
            "timed_out": False,
            "os_error": sanitized_error_record(error),
            "stdout": b"",
            "stderr": b"",
            "stdout_receipt": stream_receipt(b"", stdout_cap),
            "stderr_receipt": stream_receipt(b"", stderr_cap),
        }
    finally:
        cwd_error = None
        try:
            if saved_cwd_fd is not None:
                os.fchdir(saved_cwd_fd)
            if cwd_fd is not None:
                _verify_absolute_directory_binding(cwd, cwd_fd)
        except BaseException as error:
            cwd_error = error
        finally:
            if saved_cwd_fd is not None:
                os.close(saved_cwd_fd)
            if cwd_fd is not None:
                os.close(cwd_fd)
        if cwd_error is not None:
            if process is not None:
                _kill_process_group(process)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired as error:
                    raise ContractError("child-cwd-reap-timeout") from error
            raise cwd_error
    selector = selectors.DefaultSelector()
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    caps = {"stdout": stdout_cap, "stderr": stderr_cap}
    observed = {"stdout": 0, "stderr": 0}
    overflow = {"stdout": False, "stderr": False}
    timed_out = False
    killed_for_cap = False
    termination_started = None
    completed = False
    try:
        require(process.stdout is not None and process.stderr is not None, "child-pipes")
        for name, pipe in (("stdout", process.stdout), ("stderr", process.stderr)):
            os.set_blocking(pipe.fileno(), False)
            selector.register(pipe, selectors.EVENT_READ, name)
        deadline = started_monotonic + timeout_seconds
        while selector.get_map() or process.poll() is None:
            now_monotonic = time.monotonic()
            remaining = deadline - now_monotonic
            if remaining <= 0 and not timed_out:
                # The process-group lifetime, rather than the direct child's
                # poll state, owns the deadline.  A reaped leader can leave a
                # descendant holding either pipe indefinitely.
                timed_out = True
                _kill_process_group(process)
                termination_started = now_monotonic
            if (
                termination_started is not None
                and now_monotonic - termination_started >= 5.0
                and selector.get_map()
            ):
                # SIGKILL cannot be ignored, but a descendant that escaped the
                # group may retain inherited pipes.  The recorder owns its read
                # descriptors and closes them after a bounded drain interval.
                for key in list(selector.get_map().values()):
                    selector.unregister(key.fileobj)
                    key.fileobj.close()
            if (
                termination_started is not None
                and now_monotonic - termination_started >= 10.0
                and process.poll() is None
            ):
                raise ContractError("child-reap-timeout")
            wait_seconds = (
                0.1 if termination_started is not None
                else max(0.0, min(0.1, remaining))
            )
            if selector.get_map():
                events = selector.select(wait_seconds)
            else:
                time.sleep(max(0.001, wait_seconds))
                events = []
            for key, _mask in events:
                name = key.data
                try:
                    chunk = os.read(key.fileobj.fileno(), 65536)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(key.fileobj)
                    key.fileobj.close()
                    continue
                observed[name] += len(chunk)
                available = max(0, caps[name] - len(buffers[name]))
                if available:
                    buffers[name].extend(chunk[:available])
                if observed[name] > caps[name]:
                    overflow[name] = True
                    if not killed_for_cap:
                        killed_for_cap = True
                        _kill_process_group(process)
                        termination_started = time.monotonic()
        try:
            returncode = process.wait(timeout=1.0)
        except subprocess.TimeoutExpired as error:
            raise ContractError("child-reap-timeout") from error
        completed = True
        elapsed = max(0.0, time.monotonic() - started_monotonic)
        ended_wall = utc_now()
        receipts = {}
        for name in ("stdout", "stderr"):
            raw = bytes(buffers[name])
            receipts[name] = stream_receipt(
                raw,
                caps[name],
                truncated=overflow[name],
                observed_bytes_at_least=observed[name] if overflow[name] else None,
            )
        return {
            "started_utc": started_wall,
            "ended_utc": ended_wall,
            "elapsed_monotonic_seconds": elapsed,
            "returncode": returncode,
            "timed_out": timed_out,
            "os_error": None,
            "stdout": bytes(buffers["stdout"]),
            "stderr": bytes(buffers["stderr"]),
            "stdout_receipt": receipts["stdout"],
            "stderr_receipt": receipts["stderr"],
        }
    finally:
        selector.close()
        if not completed:
            try:
                _kill_process_group(process)
            finally:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    _kill_process_group(process)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired as error:
                        raise ContractError("child-final-reap-timeout") from error
            for pipe in (process.stdout, process.stderr):
                if pipe is not None and not pipe.closed:
                    pipe.close()


def exact_environment(plan=None):
    expected = dict(EXPECTED_ENVIRONMENT)
    if plan is not None:
        require(plan["execution_policy"]["environment"] == expected, "plan-environment")
    return expected


def validate_outer_environment():
    observed = dict(os.environ)
    for key, value in MACOS_PYTHON_BOOTSTRAP_ENVIRONMENT.items():
        require(observed.pop(key, None) == value, "outer-bootstrap-environment")
    require(observed == EXPECTED_ENVIRONMENT, "outer-environment")
    for key in MACOS_PYTHON_BOOTSTRAP_ENVIRONMENT:
        del os.environ[key]
    require(dict(os.environ) == EXPECTED_ENVIRONMENT, "outer-environment-normalized")
    require(os.getcwd() == ROOT, "outer-cwd")


def validate_git_metadata_static():
    for path, expected_bytes, expected_mode, expected_sha in GIT_METADATA_FILES:
        absolute = root_path(path) if not path.startswith("/") else path
        identity, _raw = file_identity(
            absolute, path, expected_mode, expected_sha, expected_bytes
        )
        require(identity["bytes"] == expected_bytes, "git-metadata-bytes")
    for path in GIT_METADATA_ABSENT:
        require(not _path_exists_nofollow(path), "git-metadata-absent")


def _helper_timeout(requested):
    require(requested > 0, "helper-timeout-request")
    effective = float(requested)
    campaign_remaining = seconds_until_deadline()
    if campaign_remaining > 0:
        effective = min(effective, campaign_remaining)
    if _ACTIVE_HELPER_MONOTONIC_DEADLINE is not None:
        effective = min(
            effective,
            _ACTIVE_HELPER_MONOTONIC_DEADLINE - time.monotonic(),
            campaign_remaining,
        )
    require(effective > 0, "helper-deadline-expired")
    return effective


def git_query(arguments, expected_returncode=0, stdout_cap=16 * 1024 * 1024):
    validate_git_metadata_static()
    argv = [
        GIT,
        "--no-pager",
        "--no-replace-objects",
        "-c", "core.hooksPath=/dev/null",
        "-c", "core.fsmonitor=false",
        "-c", "maintenance.auto=false",
        "-c", "gc.auto=0",
        "-c", "commit.gpgSign=false",
    ] + list(arguments)
    result = run_bounded(
        argv, _helper_timeout(60), stdout_cap, 1024 * 1024,
        exact_environment(), ROOT,
    )
    require(result["os_error"] is None and not result["timed_out"], "git-process")
    require(not result["stdout_receipt"]["truncated"], "git-stdout-cap")
    require(not result["stderr_receipt"]["truncated"], "git-stderr-cap")
    require(result["returncode"] == expected_returncode, "git-returncode")
    require(result["stderr"] == b"", "git-stderr")
    validate_git_metadata_static()
    return result["stdout"]


def bootstrap_self_validation():
    self_absolute = root_path(RECORDER_PATH)
    identity, raw = file_identity(self_absolute, RECORDER_PATH, "0644", max_bytes=4 * 1024 * 1024)
    head_raw = git_query(["show", "HEAD:" + RECORDER_PATH], stdout_cap=4 * 1024 * 1024)
    require(head_raw == raw, "recorder-head-mismatch")
    head = git_query(["rev-parse", "--verify", "HEAD^{commit}"]).decode("ascii").strip()
    require(HEX40_RE.fullmatch(head) is not None, "head-format")
    return head, identity


def _load_base_plan():
    plan, identity, raw = load_canonical_file(
        root_path(BASE_PLAN_PATH), "0644", BASE_PLAN_SHA256,
        4 * 1024 * 1024,
    )
    require(identity["bytes"] == len(raw), "base-plan-bytes")
    require(plan.get("schema") == BASE_PLAN_SCHEMA, "base-plan-schema")
    require(plan.get("campaign_t0_utc") == CAMPAIGN_T0, "base-plan-t0")
    require(
        plan.get("campaign_deadline_utc") == CAMPAIGN_DEADLINE,
        "base-plan-deadline",
    )
    require(
        tuple(plan["execution_policy"]["record_schema"]["stage_order"])
        == STAGE_ORDER,
        "base-plan-stages",
    )
    require(plan["execution_policy"]["registry_paths"] == {
        key: "results/rank_4_jacek_hybrid/gates/tt_exact_collision/" + value
        for key, value in REGISTRY_RELATIVE.items()
    }, "base-plan-registries")
    base_environment = dict(EXPECTED_ENVIRONMENT)
    base_environment["TMPDIR"] = os.path.join(
        ROOT,
        "build/rank4-jacek-hybrid-tt-exact-collision/clang-release/tmp",
    )
    require(
        plan["execution_policy"]["environment"] == base_environment,
        "base-plan-environment",
    )
    return plan, identity


def _replace_strings(value, old, new):
    if isinstance(value, dict):
        return {
            key: _replace_strings(child, old, new)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_replace_strings(child, old, new) for child in value]
    if isinstance(value, str):
        return value.replace(old, new)
    return value


def _materialize_generated_plan(base, overlay):
    overrides = overlay["effective_plan_overrides"]
    exact_keys(
        overrides,
        (
            "array_element_replacements", "array_index_deletions",
            "canonical_encoding", "ordering",
            "recursive_exact_string_substitutions", "root_additions",
            "root_replacements", "scalar_replacements", "schema",
            "shallow_object_merges",
        ),
        "generated-effective-overrides-keys",
    )
    require(
        overrides["schema"]
        == "rank4-jacek-hybrid-tt-exact-collision-generated-v19-fixed-materializer-v1",
        "generated-materializer-schema",
    )

    # First discard every v1 file-bank metadata object without inspecting it.
    # No later materialization step can visit that old list.
    effective = dict(base)
    effective["development_banks"] = overlay["generated_banks"]

    allowed_substitution_sections = frozenset((
        "commit_governance", "execution_policy", "frozen_inputs",
        "gate_architecture", "stage0",
    ))
    substitutions = overrides["recursive_exact_string_substitutions"]
    require(isinstance(substitutions, list), "generated-substitution-list")
    for record in substitutions:
        exact_keys(
            record, ("from", "sections", "to"),
            "generated-substitution-keys",
        )
        old = record["from"]
        new = record["to"]
        sections = record["sections"]
        require(
            isinstance(old, str) and old != ""
            and isinstance(new, str) and new != "" and old != new,
            "generated-substitution-values",
        )
        require(
            isinstance(sections, list) and sections
            and len(sections) == len(set(sections))
            and set(sections).issubset(allowed_substitution_sections),
            "generated-substitution-sections",
        )
        for section in sections:
            effective[section] = _replace_strings(
                effective[section], old, new,
            )

    require(overrides["root_replacements"] == {
        "schema_from": "/schema",
        "campaign_id_from": "/campaign_id",
        "created_utc_from": "/created_utc",
        "campaign_t0_utc_from": "/campaign_t0_utc",
        "campaign_deadline_utc_from": "/deadline_utc",
        "classification_from": "/classification",
        "development_banks_from": "/generated_banks",
        "stages_from": "/stages",
        "protected_boundary_from": "/protected_boundary",
        "terminal_policy_from": "/terminal_policy",
    }, "generated-root-replacements")
    effective.update({
        "schema": overlay["schema"],
        "campaign_id": overlay["campaign_id"],
        "created_utc": overlay["created_utc"],
        "campaign_t0_utc": overlay["campaign_t0_utc"],
        "campaign_deadline_utc": overlay["deadline_utc"],
        "classification": overlay["classification"],
        "development_banks": overlay["generated_banks"],
        "stages": overlay["stages"],
        "protected_boundary": overlay["protected_boundary"],
        "terminal_policy": overlay["terminal_policy"],
    })
    require(overrides["root_additions"] == {
        "incident_from": "/incident",
        "runtime_from": "/runtime",
        "timing_from": "/timeout_policy",
        "generation_protocol_from": "/generation_protocol",
        "implementation_closure_from": "/implementation_closure",
        "governance_binding_from": "/governance",
        "trace_free_binary_record_from": "/trace_free_binary_record",
    }, "generated-root-additions")
    effective.update({
        "incident": overlay["incident"],
        "runtime": overlay["runtime"],
        "timing": overlay["timeout_policy"],
        "generation_protocol": overlay["generation_protocol"],
        "implementation_closure": overlay["implementation_closure"],
        "governance_binding": overlay["governance"],
        "trace_free_binary_record": overlay["trace_free_binary_record"],
    })
    require(overlay["trace_free_binary_record"] == {
        "link_trace": [],
        "link_trace_count": 0,
        "link_trace_sha256": sha256_bytes(b""),
    }, "generated-trace-free-binary-record")

    require(overrides["shallow_object_merges"] == {
        "/summary_validation": {
            "configuration_values_from":
            "/configuration_overrides/configuration_values",
        },
        "/execution_policy": {
            "campaign_locks_from": "/runtime/local_lock+/runtime/shared_locks",
            "registry_paths_from": "/runtime/registries",
            "timeout_seconds_per_stage_from":
            "/timeout_policy/stage_aggregate_timeout_seconds",
            "no_retry_from": "/execution_policy/no_retry",
            "latest_claim_utc_from": "/timeout_policy/latest_claim_utc",
        },
        "/execution_policy/bounded_evidence": {
            "external_dependency_aliases_from":
            "/execution_policy/external_dependency_aliases",
            "git_object_queries_from":
            "/execution_policy/git_object_queries",
        },
        "/commit_governance/implementation_commit": {
            "path_closure_from": "/implementation_closure/path_closure",
        },
    }, "generated-shallow-merges")
    summary = dict(effective["summary_validation"])
    summary["configuration_values"] = overlay[
        "configuration_overrides"
    ]["configuration_values"]
    effective["summary_validation"] = summary
    execution = dict(effective["execution_policy"])
    execution["campaign_locks"] = [
        overlay["runtime"]["local_lock"],
        *overlay["runtime"]["shared_locks"],
    ]
    execution["registry_paths"] = overlay["runtime"]["registries"]
    execution["timeout_seconds_per_stage"] = overlay[
        "timeout_policy"
    ]["stage_aggregate_timeout_seconds"]
    execution["no_retry"] = overlay["execution_policy"]["no_retry"]
    execution["latest_claim_utc"] = overlay["timeout_policy"]["latest_claim_utc"]
    bounded = dict(execution["bounded_evidence"])
    bounded["external_dependency_aliases"] = overlay[
        "execution_policy"
    ]["external_dependency_aliases"]
    bounded["git_object_queries"] = overlay[
        "execution_policy"
    ]["git_object_queries"]
    execution["bounded_evidence"] = bounded
    effective["execution_policy"] = execution
    governance = dict(effective["commit_governance"])
    implementation = dict(governance["implementation_commit"])
    implementation["path_closure"] = overlay[
        "implementation_closure"
    ]["path_closure"]
    compiler = dict(
        implementation["preexecution_no_protected_bank_verification"]
    )
    require(overrides["array_element_replacements"] == {
        (
            "/commit_governance/implementation_commit/"
            "preexecution_no_protected_bank_verification/commands/2"
        ): "/trace_free_compile_commands/command2",
        (
            "/commit_governance/implementation_commit/"
            "preexecution_no_protected_bank_verification/commands/4"
        ): "/warning_clean_compile_commands/command4",
        (
            "/commit_governance/implementation_commit/"
            "preexecution_no_protected_bank_verification/commands/6"
        ): "/warning_clean_compile_commands/command6",
        (
            "/commit_governance/implementation_commit/"
            "preexecution_no_protected_bank_verification/commands/8"
        ): "/preexecution_generated_self_test/argv",
    }, "generated-array-replacements")
    commands = [list(command) for command in compiler["commands"]]
    require(len(commands) == 9, "generated-preexecution-command-count")
    replacements = overlay["trace_free_compile_commands"]
    exact_keys(
        replacements, ("command2", "command4", "command6"),
        "generated-trace-free-command-keys",
    )
    for index in (2, 4, 6):
        prior = commands[index]
        require(prior.count("-Wl,-t") == 1, "generated-trace-token-count")
        expected = [token for token in prior if token != "-Wl,-t"]
        replacement = list(replacements["command" + str(index)])
        require(replacement == expected, "generated-trace-free-command")
        commands[index] = replacement
    warning = overlay["warning_clean_compile_commands"]
    exact_keys(
        warning,
        (
            "changed_indices", "command4", "command6", "insert_after",
            "insert_token",
        ),
        "generated-warning-clean-command-keys",
    )
    require(
        warning["changed_indices"] == [4, 6]
        and warning["insert_after"] == "-Wpedantic"
        and warning["insert_token"] == "-Wno-keyword-macro"
        and "-Wno-unused-function" not in commands[2],
        "generated-warning-clean-policy",
    )
    for index in (4, 6):
        prior = commands[index]
        require(
            prior.count("-Wpedantic") == 1
            and "-Wno-keyword-macro" not in prior
            and "-Wno-unused-function" not in prior,
            "generated-warning-clean-baseline",
        )
        insertion = prior.index("-Wpedantic") + 1
        expected = (
            prior[:insertion]
            + ["-Wno-keyword-macro"]
            + prior[insertion:]
        )
        replacement = warning["command" + str(index)]
        require(
            isinstance(replacement, list) and replacement == expected,
            "generated-warning-clean-command",
        )
        commands[index] = list(replacement)
    commands[8] = list(overlay["preexecution_generated_self_test"]["argv"])
    compiler["commands"] = commands
    require(overrides["scalar_replacements"] == {
        (
            "/commit_governance/implementation_commit/"
            "preexecution_no_protected_bank_verification/"
            "safe_bank_reader_self_test"
        ): "/preexecution_generated_self_test/contract",
        (
            "/commit_governance/implementation_commit/"
            "preexecution_no_protected_bank_verification/"
            "link_trace_contract"
        ): "/trace_free_prose_correction/link_trace_contract",
        (
            "/commit_governance/implementation_commit/"
            "preexecution_no_protected_bank_verification/"
            "direct_compiler_architecture"
        ): "/trace_free_prose_correction/direct_compiler_architecture",
        (
            "/commit_governance/implementation_commit/"
            "preexecution_no_protected_bank_verification/"
            "filesystem_threat_boundary"
        ): "/trace_free_prose_correction/filesystem_threat_boundary",
        (
            "/commit_governance/implementation_commit/"
            "preexecution_no_protected_bank_verification/required"
        ): "/trace_free_prose_correction/required",
        (
            "/commit_governance/implementation_commit/"
            "preexecution_no_protected_bank_verification/"
            "translation_unit_configurations"
        ): "/trace_free_prose_correction/translation_unit_configurations",
        (
            "/frozen_inputs/runtime_tools/macho_linkage_parser/"
            "expected_linkage_names"
        ): "/linkage_correction/expected_linkage_names",
        (
            "/frozen_inputs/runtime_tools/macho_linkage_parser/"
            "expected_normalized_bytes"
        ): "/linkage_correction/expected_normalized_bytes",
        (
            "/frozen_inputs/runtime_tools/macho_linkage_parser/"
            "expected_normalized_sha256"
        ): "/linkage_correction/expected_normalized_sha256",
        (
            "/execution_policy/preexecution/receipt_evidence_shapes"
        ): "/host_projection_correction/receipt_evidence_shapes",
        (
            "/execution_policy/preexecution/receipt_requirements"
        ): "/host_projection_correction/receipt_requirements",
        (
            "/execution_policy/bounded_evidence/external_dependency_aliases"
        ): "/trace_free_prose_correction/external_dependency_aliases",
        (
            "/execution_policy/candidate_identity/derivation"
        ): "/trace_free_prose_correction/candidate_identity_derivation",
        (
            "/execution_policy/candidate_identity/payload_schema/host_runtime"
        ): "/host_projection_correction/host_runtime",
        (
            "/execution_policy/candidate_identity/stability"
        ): "/host_projection_correction/stability",
        (
            "/execution_policy/stdout_stderr"
        ): "/trace_free_prose_correction/stdout_stderr",
        (
            "/execution_policy/record_schema/stable_evidence_schema"
        ): "/governance_projection_correction/stable_evidence_schema",
    }, "generated-scalar-replacements")
    compiler["safe_bank_reader_self_test"] = overlay[
        "preexecution_generated_self_test"
    ]["contract"]
    prose = overlay["trace_free_prose_correction"]
    exact_keys(
        prose,
        (
            "candidate_identity_derivation", "direct_compiler_architecture",
            "external_dependency_aliases", "filesystem_threat_boundary",
            "link_trace_contract", "receipt_evidence_shapes",
            "receipt_requirements", "required", "stdout_stderr",
            "translation_unit_configurations",
        ),
        "generated-trace-free-prose-keys",
    )
    compiler["link_trace_contract"] = prose["link_trace_contract"]
    compiler["direct_compiler_architecture"] = prose[
        "direct_compiler_architecture"
    ]
    compiler["filesystem_threat_boundary"] = prose[
        "filesystem_threat_boundary"
    ]
    compiler["required"] = prose["required"]
    compiler["translation_unit_configurations"] = prose[
        "translation_unit_configurations"
    ]
    implementation["preexecution_no_protected_bank_verification"] = compiler
    governance["implementation_commit"] = implementation
    effective["commit_governance"] = governance

    correction = overlay["linkage_correction"]
    require(correction == {
        "expected_linkage_names": [
            "/usr/lib/libc++.1.dylib",
            "/usr/lib/libSystem.B.dylib",
        ],
        "expected_normalized_bytes": 51,
        "expected_normalized_sha256":
        "675f19d7a700adb463e931766b30c0f7ed19f0b112f2a67cc98a7d45b1e64d62",
    }, "generated-linkage-correction")
    frozen = dict(effective["frozen_inputs"])
    runtime_tools = dict(frozen["runtime_tools"])
    linkage_parser = dict(runtime_tools["macho_linkage_parser"])
    for key in (
        "expected_linkage_names", "expected_normalized_bytes",
        "expected_normalized_sha256",
    ):
        linkage_parser[key] = correction[key]
    runtime_tools["macho_linkage_parser"] = linkage_parser
    frozen["runtime_tools"] = runtime_tools
    effective["frozen_inputs"] = frozen
    diagnostic = overlay["diagnostic_receipt_correction"]
    exact_keys(
        diagnostic,
        (
            "errors_shape", "label_contract", "receipt_evidence_shapes",
            "receipt_requirements",
        ),
        "generated-diagnostic-correction-keys",
    )
    require(
        diagnostic["errors_shape"]
        == ["preexecution-observed-failure", "detail:<label>"],
        "generated-diagnostic-errors-shape",
    )
    execution = dict(effective["execution_policy"])
    preexecution = dict(execution["preexecution"])
    host_correction = overlay["host_projection_correction"]
    exact_keys(
        host_correction,
        (
            "current_host_keys_exact", "host_runtime",
            "prior_host_projection_algorithm", "receipt_evidence_shapes",
            "receipt_requirements", "stability",
        ),
        "generated-host-projection-correction-keys",
    )
    require(
        host_correction["current_host_keys_exact"] == [
            "cpu_model", "logical_cpu_count", "machine", "node",
            "release", "system", "version",
        ],
        "generated-host-projection-keys",
    )
    preexecution["receipt_evidence_shapes"] = host_correction[
        "receipt_evidence_shapes"
    ]
    preexecution["receipt_requirements"] = host_correction[
        "receipt_requirements"
    ]
    bounded = dict(execution["bounded_evidence"])
    bounded["external_dependency_aliases"] = prose[
        "external_dependency_aliases"
    ]
    execution["bounded_evidence"] = bounded
    candidate_identity = dict(execution["candidate_identity"])
    candidate_identity["derivation"] = prose["candidate_identity_derivation"]
    candidate_payload = dict(candidate_identity["payload_schema"])
    candidate_payload["host_runtime"] = host_correction["host_runtime"]
    candidate_identity["payload_schema"] = candidate_payload
    candidate_identity["stability"] = host_correction["stability"]
    execution["candidate_identity"] = candidate_identity
    execution["stdout_stderr"] = prose["stdout_stderr"]
    execution["preexecution"] = preexecution
    governance_correction = overlay["governance_projection_correction"]
    exact_keys(
        governance_correction,
        (
            "algorithm", "implementation_closure_paths",
            "stable_evidence_schema", "validation_retained",
        ),
        "generated-governance-projection-correction-keys",
    )
    implementation_paths = sorted(
        set(
            overlay["implementation_closure"]["path_closure"]["modified"]
            + overlay["implementation_closure"]["path_closure"]["new"]
        ),
        key=lambda item: item.encode("utf-8"),
    )
    require(
        len(implementation_paths) == 8
        and governance_correction["implementation_closure_paths"]
        == implementation_paths,
        "generated-governance-projection-paths",
    )
    record_schema = dict(execution["record_schema"])
    record_schema["stable_evidence_schema"] = governance_correction[
        "stable_evidence_schema"
    ]
    execution["record_schema"] = record_schema
    effective["execution_policy"] = execution

    require(overrides["array_index_deletions"] == {
        "/execution_policy/process_exclusion/markers_exact_substrings": [
            44, 43, 42, 41,
        ],
    }, "generated-array-deletions")
    execution = dict(effective["execution_policy"])
    process_exclusion = dict(execution["process_exclusion"])
    markers = list(process_exclusion["markers_exact_substrings"])
    for index in (44, 43, 42, 41):
        require(index < len(markers), "generated-marker-delete-index")
        del markers[index]
    process_exclusion["markers_exact_substrings"] = markers
    execution["process_exclusion"] = process_exclusion
    effective["execution_policy"] = execution

    generated_argvs = [record["argv"] for record in overlay["stages"]]
    generated_argvs.append(overlay["preexecution_generated_self_test"]["argv"])
    forbidden = frozenset(("--bank", "--expected-bytes", "--expected-sha256"))
    require(
        all(
            isinstance(argv, list) and argv
            and not forbidden.intersection(argv)
            for argv in generated_argvs
        ),
        "generated-bank-argv-boundary",
    )
    effective_raw = canonical_json(effective)
    require(b".tsv" not in effective_raw, "generated-effective-protected-literal")
    require(
        len(effective_raw) == overlay["effective_plan_bytes"],
        "generated-effective-plan-bytes",
    )
    require(
        sha256_bytes(effective_raw)
        == overlay["effective_plan_sha256"],
        "generated-effective-plan-digest",
    )
    return effective


def _activate_generated_runtime(overlay, identity, raw):
    global BUILD_PARENT_REL, BUILD_ROOT_REL, CAMPAIGN_DEADLINE
    global CAMPAIGN_ROOT, CAMPAIGN_SCHEMA_PREFIX, CAMPAIGN_T0
    global EXPECTED_ENVIRONMENT, GENERATED_OVERLAY_ACTIVE, PLAN_BYTES
    global PLAN_COMMIT, PLAN_PATH, PLAN_SCHEMA, PLAN_SHA256
    global RECORD_SCHEMAS, TMP_REL

    runtime = overlay["runtime"]
    exact_keys(
        runtime,
        (
            "build_parent_rel", "build_root_rel", "campaign_root",
            "local_lock", "plan_path", "registries", "schema_prefix",
            "shared_locks", "tmp_rel",
        ),
        "generated-runtime-keys",
    )
    plan_path = canonical_relative_path(runtime["plan_path"])
    campaign_root_rel = canonical_relative_path(runtime["campaign_root"])
    build_parent = canonical_relative_path(runtime["build_parent_rel"])
    build_root = canonical_relative_path(runtime["build_root_rel"])
    tmp = canonical_relative_path(runtime["tmp_rel"])
    require(plan_path == GENERATED_OVERLAY_PATH, "generated-plan-path")
    require(
        plan_path == campaign_root_rel + "/PLAN.json",
        "generated-plan-root-binding",
    )
    require(build_root == build_parent + "/clang-release", "generated-build-root")
    require(tmp == build_root + "/tmp", "generated-tmp-root")
    require(
        runtime["local_lock"] == campaign_root_rel + "/.recorder.lock",
        "generated-local-lock",
    )
    require(
        isinstance(runtime["schema_prefix"], str)
        and runtime["schema_prefix"]
        == "rank4-jacek-hybrid-tt-exact-collision-generated-v19-",
        "generated-schema-prefix",
    )
    expected_registries = {
        key: campaign_root_rel + "/" + value
        for key, value in REGISTRY_RELATIVE.items()
    }
    require(runtime["registries"] == expected_registries, "generated-registries")
    require(
        isinstance(runtime["shared_locks"], list)
        and len(runtime["shared_locks"]) == 5,
        "generated-shared-locks",
    )

    PLAN_COMMIT = GENERATED_PLAN_COMMIT
    PLAN_PATH = plan_path
    PLAN_SHA256 = identity["sha256"]
    PLAN_BYTES = len(raw)
    PLAN_SCHEMA = overlay["schema"]
    CAMPAIGN_T0 = overlay["campaign_t0_utc"]
    CAMPAIGN_DEADLINE = overlay["deadline_utc"]
    CAMPAIGN_SCHEMA_PREFIX = runtime["schema_prefix"]
    CAMPAIGN_ROOT = root_path(campaign_root_rel)
    BUILD_PARENT_REL = build_parent
    BUILD_ROOT_REL = build_root
    TMP_REL = tmp
    RECORD_SCHEMAS = _record_schemas(CAMPAIGN_SCHEMA_PREFIX)
    EXPECTED_ENVIRONMENT = dict(EXPECTED_ENVIRONMENT)
    EXPECTED_ENVIRONMENT["TMPDIR"] = root_path(TMP_REL)
    GENERATED_OVERLAY_ACTIVE = True


def load_plan():
    require(
        HEX40_RE.fullmatch(GENERATED_PLAN_COMMIT) is not None,
        "generated-plan-commit-placeholder",
    )
    require(
        HEX64_RE.fullmatch(GENERATED_OVERLAY_SHA256) is not None,
        "generated-plan-sha-placeholder",
    )
    base, _base_identity = _load_base_plan()
    overlay, identity, raw = load_canonical_file(
        root_path(GENERATED_OVERLAY_PATH), "0644",
        GENERATED_OVERLAY_SHA256, 1024 * 1024,
    )
    require(identity["bytes"] == len(raw), "generated-plan-bytes")
    exact_keys(
        overlay,
        (
            "base_plan", "campaign_id", "campaign_t0_utc",
            "classification", "configuration_overrides", "created_utc",
            "deadline_utc", "effective_plan_overrides",
            "effective_plan_bytes", "effective_plan_sha256",
            "diagnostic_receipt_correction", "execution_policy", "generated_banks",
            "generation_protocol", "governance", "implementation_closure",
            "governance_projection_correction", "host_projection_correction",
            "incident", "inherited_exact_paths",
            "link_trace_correction",
            "linkage_correction", "protected_boundary",
            "preexecution_generated_self_test", "runtime", "schema",
            "stages", "terminal_policy", "trace_free_compile_commands",
            "trace_free_binary_record", "trace_free_contracts",
            "trace_free_prose_correction",
            "thresholds", "timeout_policy", "warning_clean_compile_commands",
        ),
        "generated-overlay-keys",
    )
    require(overlay["schema"] == GENERATED_OVERLAY_SCHEMA, "generated-plan-schema")
    require(overlay["base_plan"] == {
        "path": BASE_PLAN_PATH,
        "commit": BASE_PLAN_COMMIT,
        "sha256": BASE_PLAN_SHA256,
    }, "generated-base-plan-binding")
    require(
        overlay["campaign_t0_utc"] == CAMPAIGN_T0
        and overlay["deadline_utc"] == CAMPAIGN_DEADLINE,
        "generated-campaign-window",
    )
    parse_timestamp(overlay["created_utc"], False)
    require(
        isinstance(overlay["campaign_id"], str)
        and overlay["campaign_id"] != "",
        "generated-campaign-id",
    )
    banks = overlay["generated_banks"]
    require(isinstance(banks, list) and len(banks) == 4, "generated-bank-count")
    expected_bank_shape = (
        ("d04", 0, 4, 39, 78, "13381769204529788392", []),
        ("d08", 1, 8, 38, 76, "8380659145160962254", ["d04"]),
        (
            "d12", 2, 12, 38, 76, "9226364621029683959",
            ["d04", "d08"],
        ),
        (
            "d20", 3, 20, 38, 76, "7561814393490952798",
            ["d04", "d08", "d12"],
        ),
    )
    for bank, expected in zip(banks, expected_bank_shape):
        exact_keys(
            bank,
            (
                "depth", "excluded_prior_ids", "expected_games",
                "generation_order", "generator", "id", "pairs", "role",
                "seed", "seed_domain",
            ),
            "generated-bank-keys",
        )
        identifier, order, depth, pairs, games, seed, excluded = expected
        require(
            bank["id"] == identifier
            and bank["generation_order"] == order
            and bank["depth"] == depth
            and bank["pairs"] == pairs
            and bank["expected_games"] == games
            and bank["seed"] == seed
            and bank["role"] == "development"
            and bank["generator"] == "opening_bank::generate_bank"
            and bank["excluded_prior_ids"] == excluded
            and bank["seed_domain"]
            == (
                "rank4-jacek-hybrid-tt-exact-collision-generated-"
                "development-20260815-v1:" + identifier
            ),
            "generated-bank-contract",
        )
    timing = overlay["timeout_policy"]
    exact_keys(
        timing,
        (
            "continuation_reserve_seconds", "initial_reserve_seconds",
            "latest_claim_utc", "postflight_reserve_seconds",
            "preexecution_aggregate_timeout_seconds",
            "stage_aggregate_timeout_seconds", "total_reserve_seconds",
        ),
        "generated-timing-keys",
    )
    require(
        timing["preexecution_aggregate_timeout_seconds"] == 180
        and timing["stage_aggregate_timeout_seconds"] == 1700
        and timing["postflight_reserve_seconds"] == 180
        and timing["continuation_reserve_seconds"] == 1880
        and timing["initial_reserve_seconds"] == 5460
        and timing["total_reserve_seconds"] == 5460,
        "generated-timing-contract",
    )
    parse_timestamp(timing["latest_claim_utc"], False)
    plan = _materialize_generated_plan(base, overlay)
    _activate_generated_runtime(overlay, identity, raw)
    require(plan.get("schema") == PLAN_SCHEMA, "plan-schema")
    require(plan.get("campaign_t0_utc") == CAMPAIGN_T0, "plan-t0")
    require(plan.get("campaign_deadline_utc") == CAMPAIGN_DEADLINE, "plan-deadline")
    require(
        tuple(plan["execution_policy"]["record_schema"]["stage_order"])
        == STAGE_ORDER,
        "plan-stages",
    )
    require(
        plan["execution_policy"]["registry_paths"]
        == overlay["runtime"]["registries"],
        "plan-registries",
    )
    exact_environment(plan)
    return plan, identity


def validate_implementation_head(plan, head):
    require(head != PLAN_COMMIT, "implementation-head-missing")
    parents = git_query(["rev-list", "--parents", "-n", "1", head]).decode("ascii").strip().split()
    require(parents == [head, PLAN_COMMIT], "implementation-parent")
    closure = plan["commit_governance"]["implementation_commit"]["path_closure"]
    paths = sorted(closure["modified"] + closure["new"], key=lambda item: item.encode("utf-8"))
    require(len(paths) == 8 and len(set(paths)) == 8, "implementation-path-count")
    for path in paths:
        canonical_relative_path(path)
    diff = git_query(
        ["diff-tree", "--no-commit-id", "--name-status", "-r", "-z", PLAN_COMMIT, head]
    )
    fields = diff.split(b"\x00")
    require(fields and fields[-1] == b"", "implementation-diff-format")
    fields = fields[:-1]
    require(len(fields) == 2 * len(paths), "implementation-diff-count")
    observed = {}
    for index in range(0, len(fields), 2):
        status_text = fields[index].decode("ascii")
        path_text = fields[index + 1].decode("utf-8")
        require(status_text in ("A", "M"), "implementation-diff-status")
        require(path_text not in observed, "implementation-diff-duplicate")
        observed[path_text] = status_text
    require(set(observed) == set(paths), "implementation-diff-paths")
    require(all(observed[path] == "M" for path in closure["modified"]), "implementation-modified")
    require(all(observed[path] == "A" for path in closure["new"]), "implementation-new")
    identities = []
    for path in paths:
        tree_entry = git_query(["ls-tree", "-z", head, "--", path])
        require(tree_entry.endswith(b"\x00") and tree_entry.count(b"\x00") == 1, "implementation-tree-entry")
        metadata, tree_path = tree_entry[:-1].split(b"\t", 1)
        metadata_fields = metadata.split(b" ")
        require(
            metadata_fields[0] == b"100644"
            and metadata_fields[1] == b"blob"
            and len(metadata_fields) == 3
            and HEX40_RE.fullmatch(metadata_fields[2].decode("ascii")) is not None
            and tree_path.decode("utf-8") == path,
            "implementation-git-mode",
        )
        work_identity, work_raw = file_identity(root_path(path), path, "0644", max_bytes=16 * 1024 * 1024)
        blob = git_query(["show", head + ":" + path], stdout_cap=16 * 1024 * 1024)
        require(blob == work_raw, "implementation-worktree")
        identities.append(work_identity)
    return paths, identities


def create_lock_epoch():
    started = utc_now()
    try:
        nonce = os.urandom(32).hex()
    except OSError as error:
        raise ContractError("lock-epoch-random") from error
    payload = {"nonce": nonce, "pid": os.getpid(), "started_utc": started}
    identifier = sha256_bytes(canonical_json(payload))
    return {
        "schema": CAMPAIGN_SCHEMA_PREFIX + "lock-epoch-v1",
        "id": identifier,
        "nonce": nonce,
        "pid": os.getpid(),
        "started_utc": started,
    }


def validate_lock_epoch(value):
    exact_keys(value, ("schema", "id", "nonce", "pid", "started_utc"), "lock-epoch-keys")
    require(value["schema"] == CAMPAIGN_SCHEMA_PREFIX + "lock-epoch-v1", "lock-epoch-schema")
    require(HEX64_RE.fullmatch(value["id"]) is not None, "lock-epoch-id")
    require(HEX64_RE.fullmatch(value["nonce"]) is not None, "lock-epoch-nonce")
    require(isinstance(value["pid"], int) and not isinstance(value["pid"], bool) and value["pid"] > 0, "lock-epoch-pid")
    parse_timestamp(value["started_utc"], True)
    expected = sha256_bytes(canonical_json({
        "nonce": value["nonce"], "pid": value["pid"], "started_utc": value["started_utc"]
    }))
    require(value["id"] == expected, "lock-epoch-derived")


def parse_make_dependencies(raw, expected_target):
    require(isinstance(raw, bytes), "dependency-bytes")
    require(len(raw) > 0 and raw.endswith(b"\n"), "dependency-ending")
    require(b"\x00" not in raw and b"\r" not in raw, "dependency-control")
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise ContractError("dependency-ascii") from error
    unfolded = text.replace("\\\n", "")
    require(not unfolded.endswith("\\"), "dependency-continuation")
    dependencies = []
    saw_rule = False
    for line in unfolded.splitlines():
        require(line != "", "dependency-empty-rule")
        colon = None
        escaped = False
        for index, character in enumerate(line):
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == ":":
                colon = index
                break
        require(colon is not None and not escaped, "dependency-colon")
        target = line[:colon]
        require(target == expected_target, "dependency-target")
        body = line[colon + 1 :]
        tokens = []
        current = []
        index = 0
        while index < len(body):
            character = body[index]
            if character in " \t":
                if current:
                    tokens.append("".join(current))
                    current = []
                index += 1
                continue
            if character == "\\":
                require(index + 1 < len(body), "dependency-escape")
                escaped_character = body[index + 1]
                require(escaped_character in " #:\\", "dependency-escape")
                current.append(escaped_character)
                index += 2
                continue
            current.append(character)
            index += 1
        if current:
            tokens.append("".join(current))
        require(tokens, "dependency-no-input")
        saw_rule = True
        dependencies.extend(tokens)
    require(saw_rule, "dependency-no-rule")
    result = sorted(set(dependencies), key=lambda item: item.encode("utf-8"))
    require(result and all(item != "" for item in result), "dependency-token")
    return result


def normalized_dependency_path(token):
    require(isinstance(token, str) and token != "", "dependency-token")
    if token.startswith("/"):
        return canonical_absolute_path(token, allow_runtime=True)
    normalized = os.path.normpath(token)
    require(
        normalized not in (".", "..")
        and not normalized.startswith("../"),
        "dependency-relative-escape",
    )
    return canonical_relative_path(normalized)


def parse_macho_linkage(raw, expected_names):
    require(isinstance(raw, bytes) and len(raw) >= 32, "macho-header")
    magic, cputype, _cpusubtype, filetype, ncmds, sizeofcmds, _flags, _reserved = struct.unpack_from("<8I", raw, 0)
    require(magic == 0xFEEDFACF and cputype == 0x0100000C and filetype == 2, "macho-identity")
    require(32 + sizeofcmds <= len(raw), "macho-command-size")
    recognized = {0x0000000C, 0x80000018, 0x8000001F, 0x00000020, 0x80000023}
    names = []
    offset = 32
    for _index in range(ncmds):
        require(offset + 8 <= 32 + sizeofcmds, "macho-command-header")
        command, command_size = struct.unpack_from("<II", raw, offset)
        require(command_size >= 8 and command_size % 8 == 0, "macho-command-alignment")
        end = offset + command_size
        require(end <= 32 + sizeofcmds and end > offset, "macho-command-bounds")
        if command in recognized:
            require(command == 0x0000000C and command_size >= 24, "macho-dylib-command")
            name_offset = struct.unpack_from("<I", raw, offset + 8)[0]
            require(24 <= name_offset < command_size, "macho-name-offset")
            field = raw[offset + name_offset : end]
            nul = field.find(b"\x00")
            require(nul > 0 and all(byte == 0 for byte in field[nul:]), "macho-name-padding")
            try:
                name = field[:nul].decode("ascii")
            except UnicodeDecodeError as error:
                raise ContractError("macho-name-ascii") from error
            require(all(32 <= ord(character) <= 126 for character in name), "macho-name-character")
            names.append(name)
        offset = end
    require(offset == 32 + sizeofcmds, "macho-command-final")
    require(names == list(expected_names), "macho-linkage-names")
    normalized = "".join(name + "\n" for name in names).encode("ascii")
    return {
        "parser_schema": "macho64-little-endian-load-dylib-v1",
        "normalized_linkage_names": names,
        "normalized_sha256": sha256_bytes(normalized),
    }


def parse_process_table(raw):
    require(raw and b"\x00" not in raw and b"\r" not in raw, "process-table-shape")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ContractError("process-table-decode") from error
    rows = []
    seen = set()
    line_re = re.compile(r"\A *([1-9][0-9]*) +([0-9]+) +(.*)\Z")
    for line in text.splitlines():
        if line == "":
            continue
        match = line_re.fullmatch(line)
        require(match is not None, "process-table-line")
        pid = int(match.group(1))
        ppid = int(match.group(2))
        command = match.group(3)
        require(
            pid not in seen and command != ""
            and len(command.encode("utf-8")) <= 65536,
            "process-table-duplicate",
        )
        seen.add(pid)
        rows.append({"pid": pid, "ppid": ppid, "command": command})
    require(rows and len(rows) <= 4096, "process-table-cardinality")
    rows.sort(key=lambda item: item["pid"])
    by_pid = {row["pid"]: row for row in rows}
    for origin in by_pid:
        cursor = origin
        visited = set()
        while cursor in by_pid:
            require(cursor not in visited, "process-cycle")
            visited.add(cursor)
            cursor = by_pid[cursor]["ppid"]
    return rows


def derive_process_record(raw, plan, checked_utc=None, own_pid=None):
    rows = parse_process_table(raw)
    own_pid = os.getpid() if own_pid is None else own_pid
    by_pid = {row["pid"]: row for row in rows}
    require(own_pid in by_pid, "process-self-missing")
    excluded = []
    cursor = own_pid
    visited = set()
    while cursor != 0:
        require(cursor not in visited, "process-cycle")
        visited.add(cursor)
        excluded.append(cursor)
        row = by_pid.get(cursor)
        require(row is not None, "process-ancestor-missing")
        cursor = row["ppid"]
    excluded.sort()
    excluded_set = set(excluded)
    markers = plan["execution_policy"]["process_exclusion"]["markers_exact_substrings"]
    digest_rows = []
    conflicts = []
    for row in rows:
        command_digest = sha256_bytes(row["command"].encode("utf-8"))
        digest_rows.append({"pid": row["pid"], "ppid": row["ppid"], "command_sha256": command_digest})
        matched = [marker for marker in markers if marker in row["command"]]
        if row["pid"] not in excluded_set and matched:
            conflicts.append({
                "pid": row["pid"], "ppid": row["ppid"],
                "command_sha256": command_digest, "matched_markers": matched,
            })
    return {
        "schema": CAMPAIGN_SCHEMA_PREFIX + "process-v1",
        "checked_utc": checked_utc or utc_now(),
        "ps_tool": plan["frozen_inputs"]["runtime_tools"]["ps"],
        "ps_argv": plan["frozen_inputs"]["runtime_tools"]["ps"]["argv"],
        "observed_count": len(rows),
        "table_sha256": sha256_bytes(canonical_json(digest_rows)),
        "excluded_pids": excluded,
        "conflicts": conflicts,
        "clean": conflicts == [],
    }


def process_snapshot(plan, timeout=60):
    tool = plan["frozen_inputs"]["runtime_tools"]["ps"]
    identity, _raw = file_identity(tool["path"], tool["path"], tool["mode"], tool["sha256"], tool["bytes"])
    require(identity["bytes"] == tool["bytes"], "ps-bytes")
    result = run_bounded(
        tool["argv"], _helper_timeout(timeout), 4 * 1024 * 1024,
        1024 * 1024, exact_environment(plan), ROOT,
    )
    checked_utc = result["ended_utc"]
    try:
        require(result["os_error"] is None and not result["timed_out"], "process-scan-process")
        require(result["returncode"] == 0 and result["stderr"] == b"", "process-scan-result")
        require(not result["stdout_receipt"]["truncated"], "process-scan-cap")
        # A successfully parsed conflict is evidence, not a parser failure.
        # Callers persist the non-clean record and stop their schedule.
        return derive_process_record(result["stdout"], plan, checked_utc)
    except BaseException as error:
        raise ProcessSnapshotError(
            sanitized_reason(error), result=result, checked_utc=checked_utc
        ) from error


def build_stage0_schedule(plan):
    exact = plan["stage0"]["exact_commands"]
    schedule = [
        list(exact["table_control"]), list(exact["table_candidate"]),
        list(exact["public_control"]), list(exact["public_candidate"]),
    ]
    timeouts = [300, 300, 900, 900]
    templates = {
        "control": exact["timing_control_template"],
        "candidate": exact["timing_candidate_template"],
    }
    for panel, panel_size in (("forced-prod", 8), ("mixed-prod", 64)):
        for phase, pair_count in (("warmup", 30), ("measured", 300)):
            for pair_index in range(pair_count):
                order = (("control", 0), ("candidate", 1)) if pair_index % 2 == 0 else (("candidate", 0), ("control", 1))
                for engine, position in order:
                    replacements = {
                        "{forced-prod|mixed-prod}": panel,
                        "{warmup|measured}": phase,
                        "{decimal-index}": str(pair_index),
                        "{0|1}": str(position),
                    }
                    command = []
                    decimal_seen = 0
                    for token in templates[engine]:
                        if token == "{decimal-index}":
                            value = str(pair_index) if decimal_seen == 0 else str(pair_index % panel_size)
                            decimal_seen += 1
                            command.append(value)
                        else:
                            command.append(replacements.get(token, token))
                    schedule.append(command)
                    timeouts.append(30)
    require(len(schedule) == 1324 and len(timeouts) == 1324, "stage0-schedule-count")
    return schedule, timeouts


def fsync_directory(path):
    descriptor = _open_absolute_directory(path)
    try:
        metadata = os.fstat(descriptor)
        require(stat.S_ISDIR(metadata.st_mode), "directory-type")
        os.fsync(descriptor)
        _verify_absolute_directory_binding(path, descriptor)
    except OSError as error:
        raise ContractError("directory-fsync") from error
    finally:
        os.close(descriptor)


def validate_directory(path, expected_mode="0755"):
    return _open_absolute_directory(path, expected_mode)


def mkdir_exact(path, mode):
    parent_fd, parent_path, leaf = _open_parent_absolute(path)
    try:
        created = False
        try:
            os.mkdir(leaf, int(mode, 8), dir_fd=parent_fd)
            created = True
        except FileExistsError:
            pass
        except OSError as error:
            raise ContractError("directory-create") from error
        descriptor = _open_directory_component(parent_fd, leaf, mode)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if created:
            os.fsync(parent_fd)
        _verify_absolute_directory_binding(parent_path, parent_fd)
    finally:
        os.close(parent_fd)


def mkdir_new_exact(path, mode):
    parent_fd, parent_path, leaf = _open_parent_absolute(path)
    try:
        try:
            os.mkdir(leaf, int(mode, 8), dir_fd=parent_fd)
        except OSError as error:
            raise ContractError("fresh-directory-create") from error
        descriptor = _open_directory_component(parent_fd, leaf, mode)
        try:
            require(_list_directory_fd(descriptor, path) == [], "fresh-directory-not-empty")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.fsync(parent_fd)
        _verify_absolute_directory_binding(parent_path, parent_fd)
    finally:
        os.close(parent_fd)


def acquire_one_lock(path):
    parent_fd, parent_path, leaf = _open_parent_absolute(path)
    flags = _regular_flags(os.O_RDWR)
    try:
        while True:
            created = False
            try:
                before = os.stat(
                    leaf, dir_fd=parent_fd, follow_symlinks=False
                )
                require(stat.S_ISREG(before.st_mode), "lock-type")
                require(
                    mode_string(before) == "0644" and before.st_nlink == 1
                    and before.st_size == 0,
                    "lock-metadata",
                )
                descriptor = os.open(leaf, flags, dir_fd=parent_fd)
                break
            except FileNotFoundError:
                try:
                    descriptor = os.open(
                        leaf, flags | os.O_CREAT | os.O_EXCL, 0o644,
                        dir_fd=parent_fd,
                    )
                    created = True
                    try:
                        before = os.stat(
                            leaf, dir_fd=parent_fd, follow_symlinks=False
                        )
                    except BaseException:
                        os.close(descriptor)
                        raise
                    break
                except FileExistsError:
                    # The absent-path observation raced an exact-name create.
                    # Restart the existing-path branch before opening it.
                    continue
                except OSError as error:
                    raise ContractError("lock-create") from error
            except OSError as error:
                raise ContractError("lock-open") from error
        try:
            metadata = os.fstat(descriptor)
            path_metadata = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
            require(
                stat.S_ISREG(before.st_mode) and stat.S_ISREG(metadata.st_mode)
                and stat.S_ISREG(path_metadata.st_mode),
                "lock-type",
            )
            require(
                mode_string(metadata) == "0644" and metadata.st_nlink == 1
                and metadata.st_size == 0,
                "lock-metadata",
            )
            require(
                _same_node(before, metadata) and _same_node(metadata, path_metadata),
                "lock-race",
            )
            if created:
                os.fsync(descriptor)
                os.fsync(parent_fd)
            _verify_absolute_directory_binding(parent_path, parent_fd)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ContractError("lock-busy") from error
            path_metadata = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
            require(_same_node(metadata, path_metadata), "lock-race")
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise
    finally:
        os.close(parent_fd)


def campaign_lock_paths(plan):
    paths = []
    for value in plan["execution_policy"]["campaign_locks"]:
        if value.startswith("/"):
            absolute = canonical_absolute_path(value, allow_runtime=True)
        else:
            absolute = root_path(value)
        paths.append(absolute)
    require(len(paths) == 6 and len(set(paths)) == 6, "lock-paths")
    require(
        os.path.join(CAMPAIGN_ROOT, ".recorder.lock") in paths,
        "local-lock-path",
    )
    return sorted(paths, key=lambda item: item.encode("utf-8"))


def acquire_campaign_locks(plan):
    held = []
    try:
        for path in campaign_lock_paths(plan):
            held.append((path, acquire_one_lock(path)))
        validate_campaign_locks(held)
        return held
    except BaseException:
        release_campaign_locks(held)
        raise


def validate_campaign_locks(held):
    require(len(held) == 6, "lock-count")
    for path, descriptor in held:
        parent_fd, parent_path, leaf = _open_parent_absolute(path)
        try:
            metadata = os.fstat(descriptor)
            path_metadata = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
            require(stat.S_ISREG(metadata.st_mode), "lock-type")
            require(
                mode_string(metadata) == "0644" and metadata.st_nlink == 1
                and metadata.st_size == 0,
                "lock-metadata",
            )
            require(_same_node(metadata, path_metadata), "lock-race")
            _verify_absolute_directory_binding(parent_path, parent_fd)
        finally:
            os.close(parent_fd)


def release_campaign_locks(held):
    for _path, descriptor in reversed(held):
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def registry_absolute(name):
    require(name in REGISTRY_RELATIVE, "registry-name")
    return os.path.join(CAMPAIGN_ROOT, *REGISTRY_RELATIVE[name].split("/"))


def _list_directory_fd(descriptor, absolute_path=None):
    try:
        entries = os.listdir(descriptor)
    except OSError as error:
        raise ContractError("directory-enumeration") from error
    require(
        all(
            isinstance(name, str) and name not in ("", ".", "..")
            and "/" not in name and "\x00" not in name
            for name in entries
        ),
        "directory-entry-name",
    )
    require(len(entries) == len(set(entries)), "directory-entry-duplicate")
    if absolute_path is not None:
        _verify_absolute_directory_binding(absolute_path, descriptor)
    return entries


def _list_absolute_directory(path, expected_mode):
    descriptor = _open_absolute_directory(path, expected_mode)
    try:
        return _list_directory_fd(descriptor, path)
    finally:
        os.close(descriptor)


def _open_relative_directory(base_fd, relative, expected_mode=None, missing_ok=False):
    if relative == "":
        descriptor = os.dup(base_fd)
        if expected_mode is not None:
            require(mode_string(os.fstat(descriptor)) == expected_mode, "directory-mode")
        return descriptor
    relative = canonical_relative_path(relative)
    descriptor = os.dup(base_fd)
    try:
        components = relative.split("/")
        for index, component in enumerate(components):
            child = _open_directory_component(
                descriptor, component,
                expected_mode if index == len(components) - 1 else None,
                missing_ok=missing_ok,
            )
            if child is None:
                os.close(descriptor)
                return None
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_registry_directory(registry_name, pending=False, missing_ok=False):
    path = registry_absolute(registry_name)
    if pending:
        path = os.path.join(path, ".pending")
    return _open_absolute_directory(path, "0755", missing_ok=missing_ok), path


def _campaign_static_root_entries():
    entries = {"PLAN.json", ".recorder.lock"}
    if not GENERATED_OVERLAY_ACTIVE:
        entries.add("prototype")
    return entries


def _validate_complete_registry_topology(root_fd):
    expected_root = _campaign_static_root_entries() | {
        "preexecution", "claims", "executions", "reports", "decisions",
    }
    require(
        set(_list_directory_fd(root_fd, CAMPAIGN_ROOT)) == expected_root,
        "campaign-root-topology",
    )
    preexecution_fd = _open_directory_component(root_fd, "preexecution", "0755")
    try:
        require(
            set(_list_directory_fd(
                preexecution_fd, os.path.join(CAMPAIGN_ROOT, "preexecution")
            )) == {"claims", "receipts"},
            "preexecution-topology",
        )
    finally:
        os.close(preexecution_fd)
    for name in REGISTRY_ORDER:
        registry_fd, registry_path = _open_registry_directory(name)
        try:
            entries = set(_list_directory_fd(registry_fd, registry_path))
            require(".pending" in entries, "registry-pending-missing")
            pending_fd = _open_directory_component(
                registry_fd, ".pending", "0755"
            )
            try:
                _list_directory_fd(
                    pending_fd, os.path.join(registry_path, ".pending")
                )
            finally:
                os.close(pending_fd)
        finally:
            os.close(registry_fd)


def prepare_registry_topology():
    creation = (
        "preexecution", "preexecution/claims", "preexecution/claims/.pending",
        "preexecution/receipts", "preexecution/receipts/.pending",
        "claims", "claims/.pending", "executions", "executions/.pending",
        "reports", "reports/.pending", "decisions", "decisions/.pending",
    )
    root_fd = _open_absolute_directory(CAMPAIGN_ROOT, "0755")
    try:
        present = []
        for relative in creation:
            parent_relative, leaf = os.path.split(relative)
            parent_fd = _open_relative_directory(
                root_fd, parent_relative, missing_ok=True
            )
            if parent_fd is None:
                present.append(False)
                continue
            try:
                child_fd = _open_directory_component(
                    parent_fd, leaf, "0755", missing_ok=True
                )
                present.append(child_fd is not None)
                if child_fd is not None:
                    os.close(child_fd)
            finally:
                os.close(parent_fd)
        prefix_length = 0
        while prefix_length < len(present) and present[prefix_length]:
            prefix_length += 1
        require(
            present == [True] * prefix_length + [False] * (len(present) - prefix_length),
            "registry-topology-nonprefix",
        )

        created_suffix = prefix_length < len(creation)
        if created_suffix:
            existing = set(creation[:prefix_length])
            direct_children = {"": set()}
            for relative in creation:
                parent_relative, leaf = os.path.split(relative)
                direct_children.setdefault(parent_relative, set())
                if relative in existing:
                    direct_children[parent_relative].add(leaf)
            root_expected = _campaign_static_root_entries()
            root_expected.update(direct_children[""])
            require(
                set(_list_directory_fd(root_fd, CAMPAIGN_ROOT)) == root_expected,
                "registry-prefix-root",
            )
            for relative in creation[:prefix_length]:
                descriptor = _open_relative_directory(root_fd, relative, "0755")
                try:
                    require(
                        set(_list_directory_fd(
                            descriptor,
                            os.path.join(CAMPAIGN_ROOT, *relative.split("/")),
                        )) == direct_children.get(relative, set()),
                        "registry-prefix-not-empty",
                    )
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)

            for relative in creation[prefix_length:]:
                parent_relative, leaf = os.path.split(relative)
                parent_fd = _open_relative_directory(root_fd, parent_relative, "0755")
                try:
                    try:
                        os.mkdir(leaf, 0o755, dir_fd=parent_fd)
                    except OSError as error:
                        raise ContractError("registry-directory-create") from error
                    child_fd = _open_directory_component(parent_fd, leaf, "0755")
                    try:
                        require(
                            _list_directory_fd(child_fd) == [],
                            "registry-created-not-empty",
                        )
                        os.fsync(child_fd)
                    finally:
                        os.close(child_fd)
                    os.fsync(parent_fd)
                finally:
                    os.close(parent_fd)
                _verify_absolute_directory_binding(CAMPAIGN_ROOT, root_fd)

        _validate_complete_registry_topology(root_fd)
        if created_suffix:
            os.fsync(root_fd)
        _verify_absolute_directory_binding(CAMPAIGN_ROOT, root_fd)
    finally:
        os.close(root_fd)


def record_cap(plan, registry_name):
    mapping = {
        "preexecution_claims": "preexecution_claim",
        "preexecution_receipts": "preexecution_receipt",
        "claims": "claim",
        "executions": "execution",
        "reports": "report",
        "decisions": "decision",
    }
    return plan["execution_policy"]["limits"]["canonical_record_bytes_max"][mapping[registry_name]]


def expected_record_keys(plan, registry_name):
    if registry_name == "preexecution_claims":
        return plan["execution_policy"]["preexecution"]["claim_keys_exact"]
    if registry_name == "preexecution_receipts":
        return plan["execution_policy"]["preexecution"]["receipt_keys_exact"]
    if registry_name == "claims":
        return plan["execution_policy"]["claim_payload_schema"]["keys_exact"]
    key = {"executions": "execution_keys_exact", "reports": "report_keys_exact", "decisions": "decision_keys_exact"}[registry_name]
    return plan["execution_policy"]["record_schema"][key]


def _expected_schema_for_registry(registry_name):
    return RECORD_SCHEMAS[{
        "preexecution_claims": "preexecution_claim",
        "preexecution_receipts": "preexecution_receipt",
        "claims": "claim", "executions": "execution",
        "reports": "report", "decisions": "decision",
    }[registry_name]]


def validate_record_name(registry_name, name, payload, raw):
    require(name.endswith(".json") and len(name.encode("utf-8")) <= 192, "record-name")
    stem = name[:-5]
    digest = sha256_bytes(raw)
    if registry_name == "preexecution_claims":
        require(HEX40_RE.fullmatch(stem) is not None and payload["implementation_commit"] == stem, "preclaim-name")
    elif registry_name in ("preexecution_receipts", "reports", "decisions"):
        require(stem == digest and HEX64_RE.fullmatch(stem) is not None, "content-name")
    else:
        parts = stem.split(".", 1)
        require(len(parts) == 2 and HEX64_RE.fullmatch(parts[0]) is not None, "stage-name")
        require(parts[1] in STAGE_ORDER, "stage-name")
        require(payload["candidate_identity"] == parts[0] and payload["stage"] == parts[1], "stage-name-binding")
    return digest


def validate_record_basename_syntax(registry_name, name):
    require(
        isinstance(name, str) and name.endswith(".json")
        and len(name.encode("utf-8")) <= 192,
        "record-name",
    )
    stem = name[:-5]
    if registry_name == "preexecution_claims":
        require(HEX40_RE.fullmatch(stem) is not None, "preclaim-name")
    elif registry_name in ("preexecution_receipts", "reports", "decisions"):
        require(HEX64_RE.fullmatch(stem) is not None, "content-name")
    else:
        parts = stem.split(".", 1)
        require(
            len(parts) == 2 and HEX64_RE.fullmatch(parts[0]) is not None
            and parts[1] in STAGE_ORDER,
            "stage-name",
        )


def _load_record_at(
    plan, registry_name, name, directory_fd, directory_path,
    pending=False, allow_linked=False,
):
    path = os.path.join(directory_path, name)
    expected_mode = None if pending else "0444"
    expected_nlink = (1, 2) if allow_linked else 1
    raw, metadata = _read_regular_at(
        directory_fd, name, expected_mode, expected_nlink,
        record_cap(plan, registry_name), directory_path,
    )
    if pending:
        require(mode_string(metadata) in ("0600", "0444"), "pending-mode")
    payload = decode_json(raw)
    exact_keys(payload, expected_record_keys(plan, registry_name), "record-keys")
    require(payload["schema"] == _expected_schema_for_registry(registry_name), "record-schema")
    digest = validate_record_name(registry_name, name, payload, raw)
    return {"name": name, "path": path, "payload": payload, "raw": raw, "sha256": digest, "mode": mode_string(metadata)}


def load_record(plan, registry_name, name, pending=False, allow_linked=False):
    directory_fd, directory_path = _open_registry_directory(
        registry_name, pending=pending
    )
    try:
        return _load_record_at(
            plan, registry_name, name, directory_fd, directory_path,
            pending, allow_linked,
        )
    finally:
        os.close(directory_fd)


def _load_pending_tolerant_at(
    plan, registry_name, name, directory_fd, directory_path,
):
    path = os.path.join(directory_path, name)
    raw, metadata = _read_regular_at(
        directory_fd, name, None, (1, 2), record_cap(plan, registry_name),
        directory_path,
    )
    mode = mode_string(metadata)
    require(mode in ("0600", "0444"), "pending-mode")
    result = {
        "name": name, "path": path, "raw": raw, "mode": mode,
        "nlink": metadata.st_nlink, "complete": False, "payload": None,
        "sha256": sha256_bytes(raw),
    }
    try:
        payload = decode_json(raw)
        exact_keys(payload, expected_record_keys(plan, registry_name), "record-keys")
        require(payload["schema"] == _expected_schema_for_registry(registry_name), "record-schema")
        digest = validate_record_name(registry_name, name, payload, raw)
        result.update(complete=True, payload=payload, sha256=digest)
    except ContractError:
        # Exact-name bounded bytes that do not form a complete record are an
        # attributable interrupted publication; recovery never interprets them.
        stem = name[:-5] if name.endswith(".json") else ""
        if registry_name == "preexecution_claims":
            require(HEX40_RE.fullmatch(stem) is not None, "pending-name")
        elif registry_name in ("preexecution_receipts", "reports", "decisions"):
            require(HEX64_RE.fullmatch(stem) is not None, "pending-name")
        else:
            pieces = stem.split(".", 1)
            require(
                len(pieces) == 2 and HEX64_RE.fullmatch(pieces[0]) is not None
                and pieces[1] in STAGE_ORDER,
                "pending-name",
            )
    return result


def load_pending_tolerant(plan, registry_name, name):
    directory_fd, directory_path = _open_registry_directory(
        registry_name, pending=True
    )
    try:
        return _load_pending_tolerant_at(
            plan, registry_name, name, directory_fd, directory_path,
        )
    finally:
        os.close(directory_fd)


def scan_registry(plan, registry_name, permit_absent=False):
    descriptor, directory = _open_registry_directory(
        registry_name, missing_ok=True
    )
    if descriptor is None:
        require(permit_absent, "registry-absent")
        return [], []
    try:
        entries = _list_directory_fd(descriptor, directory)
        require(".pending" in entries, "registry-topology")
        pending_path = os.path.join(directory, ".pending")
        pending_fd = _open_directory_component(descriptor, ".pending", "0755")
        try:
            pending_names = sorted(
                _list_directory_fd(pending_fd, pending_path),
                key=lambda item: item.encode("utf-8"),
            )
            for name in pending_names:
                validate_record_basename_syntax(registry_name, name)
            finals = []
            final_names_ordered = sorted(
                (item for item in entries if item != ".pending"),
                key=lambda item: item.encode("utf-8"),
            )
            for name in final_names_ordered:
                validate_record_basename_syntax(registry_name, name)
                finals.append(_load_record_at(
                    plan, registry_name, name, descriptor, directory,
                    False, name in pending_names,
                ))
            pendings = [
                _load_pending_tolerant_at(
                    plan, registry_name, name, pending_fd, pending_path,
                )
                for name in pending_names
            ]
            final_names = set(entries) - {".pending"}
            require(
                all(
                    pending["nlink"] == (2 if pending["name"] in final_names else 1)
                    for pending in pendings
                ),
                "pending-link-topology",
            )
        finally:
            os.close(pending_fd)
        maxima = plan["execution_policy"]["limits"]["registry_final_cardinality_max"][registry_name]
        require(len(finals) <= maxima and len(pendings) <= 1, "registry-cardinality")
        _verify_absolute_directory_binding(directory, descriptor)
        return finals, pendings
    finally:
        os.close(descriptor)


def publish_record(plan, registry_name, basename, payload, held_locks):
    validate_campaign_locks(held_locks)
    raw = canonical_json(payload)
    require(len(raw) <= record_cap(plan, registry_name), "record-cap")
    validate_record_name(registry_name, basename, payload, raw)
    registry_fd, registry = _open_registry_directory(registry_name)
    pending = os.path.join(registry, ".pending")
    try:
        pending_fd = _open_directory_component(registry_fd, ".pending", "0755")
    except BaseException:
        os.close(registry_fd)
        raise
    flags = _regular_flags(os.O_RDWR) | os.O_CREAT | os.O_EXCL
    try:
        try:
            descriptor = os.open(basename, flags, 0o600, dir_fd=pending_fd)
        except FileExistsError as error:
            raise ContractError("publication-pending-exists") from error
        except OSError as error:
            raise ContractError("publication-open") from error
        try:
            created = os.fstat(descriptor)
            created_path = os.stat(
                basename, dir_fd=pending_fd, follow_symlinks=False
            )
            require(
                stat.S_ISREG(created.st_mode) and _same_node(created, created_path)
                and mode_string(created) == "0600" and created.st_nlink == 1
                and created.st_size == 0,
                "publication-created",
            )
            offset = 0
            while offset < len(raw):
                written = os.write(descriptor, raw[offset:])
                require(written > 0, "publication-write")
                offset += written
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
            os.lseek(descriptor, 0, os.SEEK_SET)
            reread = b""
            while len(reread) < len(raw):
                chunk = os.read(descriptor, len(raw) - len(reread))
                require(chunk, "publication-short-read")
                reread += chunk
            require(os.read(descriptor, 1) == b"", "publication-eof")
            metadata = os.fstat(descriptor)
            pending_metadata = os.stat(
                basename, dir_fd=pending_fd, follow_symlinks=False
            )
            require(
                reread == raw and stat.S_ISREG(metadata.st_mode)
                and mode_string(metadata) == "0444" and metadata.st_nlink == 1
                and metadata.st_size == len(raw) and _same_node(created, metadata)
                and _metadata_identity(pending_metadata) == _metadata_identity(metadata),
                "publication-reread",
            )
            _verify_absolute_directory_binding(pending, pending_fd)
            validate_campaign_locks(held_locks)
            try:
                os.link(
                    basename, basename, src_dir_fd=pending_fd,
                    dst_dir_fd=registry_fd, follow_symlinks=False,
                )
            except FileExistsError:
                # The only acceptable EEXIST is the durable post-link crash
                # state: both names are the same inode and both report nlink2.
                pass
            except OSError as error:
                raise ContractError("publication-link") from error

            linked_fd_metadata = os.fstat(descriptor)
            pending_link_metadata = os.stat(
                basename, dir_fd=pending_fd, follow_symlinks=False
            )
            final_raw, final_link_metadata = _read_regular_at(
                registry_fd, basename, "0444", 2, len(raw), registry,
            )
            require(
                final_raw == raw and linked_fd_metadata.st_nlink == 2
                and pending_link_metadata.st_nlink == 2
                and final_link_metadata.st_nlink == 2
                and _same_node(metadata, linked_fd_metadata)
                and _same_node(linked_fd_metadata, pending_link_metadata)
                and _same_node(pending_link_metadata, final_link_metadata)
                and linked_fd_metadata.st_size == len(raw),
                "publication-link-identity",
            )
            validate_campaign_locks(held_locks)
            os.fsync(descriptor)
            os.fsync(registry_fd)
            validate_campaign_locks(held_locks)
            os.unlink(basename, dir_fd=pending_fd)
            os.fsync(pending_fd)
            validate_campaign_locks(held_locks)

            after_unlink_fd = _reread_open_regular(
                descriptor, raw, "0444", 1, metadata,
            )
            final_raw, final_metadata = _read_regular_at(
                registry_fd, basename, "0444", 1, len(raw), registry,
            )
            require(
                final_raw == raw and final_metadata.st_nlink == 1
                and final_metadata.st_size == len(raw)
                and after_unlink_fd.st_nlink == 1
                and after_unlink_fd.st_size == len(raw)
                and _same_node(metadata, after_unlink_fd)
                and _same_node(after_unlink_fd, final_metadata),
                "publication-final",
            )
            _verify_absolute_directory_binding(registry, registry_fd)
            _verify_absolute_directory_binding(pending, pending_fd)
            validate_campaign_locks(held_locks)
            return _load_record_at(
                plan, registry_name, basename, registry_fd, registry, False, False,
            )
        except BaseException:
            # Recovery owns any incomplete pending inode.  Never repair here.
            raise
        finally:
            os.close(descriptor)
    finally:
        os.close(pending_fd)
        os.close(registry_fd)


def record_reference(plan, registry_name, record):
    if registry_name in ("preexecution_claims", "preexecution_receipts"):
        return {
            "path": plan["execution_policy"]["registry_paths"][registry_name] + "/" + record["name"],
            "bytes": len(record["raw"]), "sha256": record["sha256"],
            "schema": record["payload"]["schema"],
            "role": "preexecution-claim" if registry_name == "preexecution_claims" else "preexecution-receipt",
        }
    return {
        "path": plan["execution_policy"]["registry_paths"][registry_name] + "/" + record["name"],
        "bytes": len(record["raw"]), "sha256": record["sha256"],
        "schema": record["payload"]["schema"], "stage": record["payload"]["stage"],
    }


def scan_all_registries(plan, permit_absent=False):
    result = {}
    for name in REGISTRY_ORDER:
        finals, pendings = scan_registry(plan, name, permit_absent)
        result[name] = finals
        result[name + "_pending"] = pendings
    require(
        sum(len(result[name + "_pending"]) for name in REGISTRY_ORDER) <= 1,
        "campaign-pending-cardinality",
    )
    return result


def _replay_json_int(value, reason, minimum=None):
    require(isinstance(value, int) and not isinstance(value, bool), reason)
    if minimum is not None:
        require(value >= minimum, reason)
    return value


def _replay_finite_nonnegative(value, reason):
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(value) and value >= 0,
        reason,
    )
    return float(value)


def _replay_record(plan, registry_name, record):
    exact_keys(
        record, ("name", "path", "payload", "raw", "sha256", "mode"),
        "replay-record-wrapper",
    )
    require(isinstance(record["payload"], dict), "replay-record-payload")
    require(record["raw"] == canonical_json(record["payload"]), "replay-record-canonical")
    require(record["sha256"] == sha256_bytes(record["raw"]), "replay-record-digest")
    require(record["mode"] == "0444", "replay-record-mode")
    require(
        record["path"] == os.path.join(
            ROOT, plan["execution_policy"]["registry_paths"][registry_name],
            record["name"],
        ),
        "replay-record-path",
    )
    exact_keys(
        record["payload"], expected_record_keys(plan, registry_name),
        "replay-record-keys",
    )
    require(
        record["payload"]["schema"] == _expected_schema_for_registry(registry_name),
        "replay-record-schema",
    )
    validate_record_name(registry_name, record["name"], record["payload"], record["raw"])
    return record


def _replay_reference(plan, reference, registry_name, record):
    expected = record_reference(plan, registry_name, record)
    exact_keys(reference, expected.keys(), "replay-reference-keys")
    require(reference == expected, "replay-reference-binding")
    return reference


def _replay_simple_error(value, reason):
    exact_keys(value, ("type", "message_sha256"), reason)
    require(isinstance(value["type"], str) and value["type"] != "", reason)
    require(HEX64_RE.fullmatch(value["message_sha256"]) is not None, reason)


def _replay_process_error(value, plan):
    exact_keys(
        value,
        (
            "checked_utc", "type", "message_sha256", "returncode",
            "timed_out", "stdout_receipt", "stderr_receipt",
        ),
        "replay-process-error-keys",
    )
    parse_timestamp(value["checked_utc"], True)
    require(
        value["type"] in (
            "spawn-error", "timeout", "nonzero", "stream-cap",
            "decode-error", "parse-error",
        ),
        "replay-process-error-type",
    )
    require(HEX64_RE.fullmatch(value["message_sha256"]) is not None, "replay-process-error-sha")
    require(
        value["returncode"] is None
        or (isinstance(value["returncode"], int) and not isinstance(value["returncode"], bool)),
        "replay-process-error-returncode",
    )
    require(isinstance(value["timed_out"], bool), "replay-process-error-timeout")
    validate_stream_receipt(
        value["stdout_receipt"],
        plan["execution_policy"]["limits"]["process_snapshot_bytes_max"],
    )
    validate_stream_receipt(
        value["stderr_receipt"], plan["execution_policy"]["limits"]["child_stderr_bytes_max"]
    )
    return parse_timestamp(value["checked_utc"], True)


def _replay_process(value, plan, require_clean=False):
    exact_keys(
        value,
        (
            "schema", "checked_utc", "ps_tool", "ps_argv", "observed_count",
            "table_sha256", "excluded_pids", "conflicts", "clean",
        ),
        "replay-process-keys",
    )
    require(value["schema"] == CAMPAIGN_SCHEMA_PREFIX + "process-v1", "replay-process-schema")
    checked = parse_timestamp(value["checked_utc"], True)
    ps_tool = plan["frozen_inputs"]["runtime_tools"]["ps"]
    require(value["ps_tool"] == ps_tool and value["ps_argv"] == ps_tool["argv"], "replay-process-tool")
    _replay_json_int(value["observed_count"], "replay-process-count", 1)
    require(
        value["observed_count"]
        <= plan["execution_policy"]["limits"]["process_records_max"],
        "replay-process-count-cap",
    )
    require(HEX64_RE.fullmatch(value["table_sha256"]) is not None, "replay-process-table")
    require(
        isinstance(value["excluded_pids"], list)
        and value["excluded_pids"] == sorted(set(value["excluded_pids"])),
        "replay-process-excluded",
    )
    for pid in value["excluded_pids"]:
        _replay_json_int(pid, "replay-process-pid", 1)
    require(isinstance(value["conflicts"], list), "replay-process-conflicts")
    previous_pid = 0
    markers = plan["execution_policy"]["process_exclusion"]["markers_exact_substrings"]
    for conflict in value["conflicts"]:
        exact_keys(
            conflict, ("pid", "ppid", "command_sha256", "matched_markers"),
            "replay-process-conflict-keys",
        )
        pid = _replay_json_int(conflict["pid"], "replay-process-conflict-pid", 1)
        _replay_json_int(conflict["ppid"], "replay-process-conflict-ppid", 0)
        require(pid > previous_pid and pid not in value["excluded_pids"], "replay-process-conflict-order")
        previous_pid = pid
        require(HEX64_RE.fullmatch(conflict["command_sha256"]) is not None, "replay-process-conflict-sha")
        require(
            isinstance(conflict["matched_markers"], list)
            and conflict["matched_markers"]
            and conflict["matched_markers"]
            == [marker for marker in markers if marker in conflict["matched_markers"]],
            "replay-process-markers",
        )
    require(
        value["observed_count"]
        >= len(value["excluded_pids"]) + len(value["conflicts"]),
        "replay-process-count-derived",
    )
    require(value["clean"] is (value["conflicts"] == []), "replay-process-clean-derived")
    if require_clean:
        require(value["clean"] is True, "replay-process-not-clean")
    return checked


def _replay_clock_order(*values):
    return all(left <= right for left, right in zip(values, values[1:]))


def _replay_wall_matches(started, ended, elapsed):
    return abs((ended - started).total_seconds() - elapsed) <= 5.0


def _replay_preexecution_claim(plan, record):
    claim = _replay_record(plan, "preexecution_claims", record)["payload"]
    validate_lock_epoch(claim["lock_epoch"])
    require(claim["schema"] == RECORD_SCHEMAS["preexecution_claim"], "replay-preclaim-schema")
    require(claim["plan"] == plan_reference(), "replay-preclaim-plan")
    require(
        HEX40_RE.fullmatch(claim["implementation_commit"]) is not None
        and record["name"] == claim["implementation_commit"] + ".json",
        "replay-preclaim-implementation",
    )
    require(claim["one_shot"] is True and claim["core_dump_disabled"] is True, "replay-preclaim-flags")
    require(
        claim["aggregate_timeout_seconds"]
        == _preexecution_aggregate_seconds(plan),
        "replay-preclaim-timeout",
    )
    require(claim["build_roots"] == [BUILD_ROOT_REL], "replay-preclaim-build-root")
    require(claim["environment"] == exact_environment(plan), "replay-preclaim-environment")
    contract = plan["execution_policy"]["preexecution"]
    schedule = plan["commit_governance"]["implementation_commit"]["preexecution_no_protected_bank_verification"]["commands"]
    timeouts = contract["child_timeout_seconds_exact"]
    overrides = contract["command_environment_overrides_exact"]
    bindings = [{
        "argv_index": 8,
        "producer_command_index": 2,
        "binary_role": "game_gate",
        "path": schedule[2][-1],
    }]
    require(claim["command_schedule"] == schedule, "replay-preclaim-schedule")
    require(claim["child_timeout_seconds"] == timeouts, "replay-preclaim-child-timeouts")
    require(claim["command_environment_overrides"] == overrides, "replay-preclaim-overrides")
    require(claim["produced_executable_bindings"] == bindings, "replay-preclaim-bindings")
    schedule_object = {
        "commands": schedule,
        "child_timeout_seconds": timeouts,
        "command_environment_overrides": overrides,
        "produced_executable_bindings": bindings,
    }
    require(
        claim["command_schedule_sha256"] == sha256_bytes(canonical_json(schedule_object)),
        "replay-preclaim-schedule-digest",
    )
    prechecked = _replay_process(claim["preclaim_process"], plan, True)
    lock_started = parse_timestamp(claim["lock_epoch"]["started_utc"], True)
    claimed = parse_timestamp(claim["claimed_utc"], True)
    campaign_t0 = parse_timestamp(CAMPAIGN_T0, False)
    deadline = parse_timestamp(CAMPAIGN_DEADLINE, False)
    clock_valid = (
        _replay_clock_order(campaign_t0, lock_started, prechecked, claimed, deadline)
        and (claimed - prechecked).total_seconds() <= 60.0
        and (deadline - claimed).total_seconds()
        >= _initial_reserve_seconds(plan)
    )
    return {
        "payload": claim, "claimed": claimed, "prechecked": prechecked,
        "clock_valid": clock_valid,
    }


def _replay_child_result(plan, child, claimed_argv, override, index, preexecution):
    keys = (
        (
            "index", "argv", "environment_override", "resolved_executable_path",
            "started_utc", "ended_utc", "elapsed_monotonic_seconds", "returncode",
            "timed_out", "os_error", "stdout_receipt", "stderr_receipt",
            "postchild_process", "postchild_process_error",
        )
        if preexecution else
        (
            "argv_index", "argv_sha256", "resolved_executable_path", "started_utc",
            "ended_utc", "elapsed_monotonic_seconds", "returncode", "timed_out",
            "os_error", "stdout_receipt", "stderr_receipt", "postchild_process",
            "postchild_process_error",
        )
    )
    exact_keys(child, keys, "replay-child-keys")
    if preexecution:
        require(
            child["index"] == index and child["argv"] == claimed_argv
            and child["environment_override"] == override,
            "replay-child-claim-binding",
        )
    else:
        require(child["argv_index"] == index, "replay-child-index")
        require(
            child["argv_sha256"] == sha256_bytes(canonical_json(claimed_argv)),
            "replay-child-argv-digest",
        )
    executable = claimed_argv[0]
    expected_resolved = executable if executable.startswith("/") else os.path.join(
        ROOT, *canonical_relative_path(executable, True).split("/")
    )
    require(child["resolved_executable_path"] == expected_resolved, "replay-child-resolved")
    canonical_absolute_path(child["resolved_executable_path"], allow_runtime=True)
    started = parse_timestamp(child["started_utc"], True)
    ended = parse_timestamp(child["ended_utc"], True)
    elapsed = _replay_finite_nonnegative(child["elapsed_monotonic_seconds"], "replay-child-elapsed")
    require(
        child["returncode"] is None
        or (isinstance(child["returncode"], int) and not isinstance(child["returncode"], bool)),
        "replay-child-returncode",
    )
    require(isinstance(child["timed_out"], bool), "replay-child-timeout")
    if child["os_error"] is not None:
        _replay_simple_error(child["os_error"], "replay-child-os-error")
    stdout_cap = plan["execution_policy"]["limits"]["child_stdout_bytes_max"]
    stderr_cap = plan["execution_policy"]["limits"]["child_stderr_bytes_max"]
    validate_stream_receipt(child["stdout_receipt"], stdout_cap)
    validate_stream_receipt(child["stderr_receipt"], stderr_cap)
    require(
        (child["postchild_process"] is None)
        != (child["postchild_process_error"] is None),
        "replay-child-post-outcome",
    )
    if child["postchild_process"] is not None:
        postchecked = _replay_process(child["postchild_process"], plan)
    else:
        postchecked = _replay_process_error(child["postchild_process_error"], plan)
    successful = (
        child["returncode"] == 0 and child["os_error"] is None
        and child["timed_out"] is False
        and child["stdout_receipt"].get("truncated") is False
        and child["stderr_receipt"].get("truncated") is False
        and child["postchild_process"] is not None
        and child["postchild_process"]["clean"] is True
        and child["postchild_process_error"] is None
    )
    clock_valid = (
        _replay_clock_order(started, ended, postchecked)
        and _replay_wall_matches(started, ended, elapsed)
        and child["timed_out"] is False
    )
    return {
        "started": started, "ended": ended, "postchecked": postchecked,
        "elapsed": elapsed, "clock_valid": clock_valid, "successful": successful,
        "resolved": expected_resolved,
    }


def _replay_preexecution_receipt(plan, record, preclaim_record, preclaim_replay):
    receipt = _replay_record(plan, "preexecution_receipts", record)["payload"]
    claim = preclaim_replay["payload"]
    _replay_reference(plan, receipt["claim"], "preexecution_claims", preclaim_record)
    require(receipt["claim_sha256"] == preclaim_record["sha256"], "replay-receipt-claim-sha")
    validate_lock_epoch(receipt["lock_epoch"])
    require(receipt["aggregate_timeout_seconds"] == claim["aggregate_timeout_seconds"], "replay-receipt-timeout")
    require(receipt["child_timeout_seconds"] == claim["child_timeout_seconds"], "replay-receipt-child-timeouts")
    require(isinstance(receipt["passed"], bool), "replay-receipt-passed-type")
    synthetic = receipt["errors"] == ["interrupted-preexecution-after-durable-claim"]
    if receipt["passed"]:
        require(receipt["errors"] == [], "replay-receipt-passed-errors")
    elif synthetic:
        require(receipt["commands"] == [], "replay-receipt-synthetic-commands")
    else:
        require(
            isinstance(receipt["errors"], list)
            and len(receipt["errors"]) == 2
            and receipt["errors"][0] == "preexecution-observed-failure"
            and type(receipt["errors"][1]) is str
            and receipt["errors"][1].startswith("detail:")
            and PREEXECUTION_FAILURE_LABEL_RE.fullmatch(
                receipt["errors"][1][len("detail:") :]
            ) is not None,
            "replay-receipt-failed-errors",
        )
    same_epoch = receipt["lock_epoch"] == claim["lock_epoch"]
    require(same_epoch is (not synthetic), "replay-receipt-epoch-transition")
    created = parse_timestamp(receipt["created_utc"], True)
    command_results = []
    if synthetic:
        require(
            receipt["started_utc"] is None and receipt["ended_utc"] is None
            and receipt["elapsed_monotonic_seconds"] is None,
            "replay-receipt-synthetic-clock-shape",
        )
        clock_valid = _replay_clock_order(
            preclaim_replay["claimed"],
            parse_timestamp(receipt["lock_epoch"]["started_utc"], True),
            created,
        )
    else:
        require(
            isinstance(receipt["commands"], list)
            and len(receipt["commands"]) <= len(claim["command_schedule"]),
            "replay-receipt-command-prefix",
        )
        started = parse_timestamp(receipt["started_utc"], True)
        ended = parse_timestamp(receipt["ended_utc"], True)
        elapsed = _replay_finite_nonnegative(
            receipt["elapsed_monotonic_seconds"], "replay-receipt-elapsed"
        )
        previous = preclaim_replay["prechecked"]
        clock_valid = _replay_clock_order(preclaim_replay["claimed"], started)
        for index, child in enumerate(receipt["commands"]):
            replayed = _replay_child_result(
                plan, child, claim["command_schedule"][index],
                claim["command_environment_overrides"][index], index, True,
            )
            command_results.append(replayed)
            effective_cap = min(
                float(claim["child_timeout_seconds"][index]),
                max(
                    0.0,
                    _preexecution_aggregate_seconds(plan)
                    - (replayed["started"] - started).total_seconds(),
                ),
                max(
                    0.0,
                    (
                        parse_timestamp(CAMPAIGN_DEADLINE, False)
                        - replayed["started"]
                    ).total_seconds(),
                ),
            )
            clock_valid = (
                clock_valid and replayed["clock_valid"]
                and started <= replayed["started"]
                and previous <= replayed["started"]
                and (replayed["started"] - previous).total_seconds() <= 60.0
                and replayed["elapsed"] <= effective_cap
            )
            previous = replayed["postchecked"]
            if child["postchild_process_error"] is not None:
                require(index + 1 == len(receipt["commands"]), "replay-receipt-command-after-process-error")
        clock_valid = (
            clock_valid and _replay_clock_order(previous, ended, created)
            and _replay_wall_matches(started, ended, elapsed)
            and elapsed <= _preexecution_aggregate_seconds(plan)
        )
        if receipt["passed"]:
            clock_valid = clock_valid and (
                parse_timestamp(CAMPAIGN_DEADLINE, False) - created
            ).total_seconds() >= _continuation_reserve_seconds(plan)
            require(
                len(command_results) == 9 and all(item["successful"] for item in command_results),
                "replay-receipt-passed-children",
            )
    null_evidence = (
        receipt["toolchain"] is None and receipt["binary_records"] is None
        and receipt["compiler_records"] is None and receipt["dependency_closure"] is None
        and receipt["host_observation"] is None
    )
    if receipt["passed"]:
        require(not null_evidence, "replay-receipt-passed-evidence")
    else:
        require(null_evidence, "replay-receipt-failed-evidence")
    if synthetic:
        require(
            receipt["build_roots"] == [{
                "path": BUILD_ROOT_REL, "mode": None, "fresh": None,
                "tmp_mode": None, "tmp_empty": None, "entries_count": None,
                "entries_sha256": None,
            }],
            "replay-receipt-synthetic-root",
        )
    elif not receipt["passed"]:
        require(
            isinstance(receipt["build_roots"], list)
            and len(receipt["build_roots"]) == 1,
            "replay-receipt-failed-root-count",
        )
        failed_root = receipt["build_roots"][0]
        exact_keys(
            failed_root,
            (
                "path", "mode", "fresh", "tmp_mode", "tmp_empty",
                "entries_count", "entries_sha256",
            ),
            "replay-receipt-failed-root-keys",
        )
        require(
            failed_root["path"] == BUILD_ROOT_REL
            and failed_root["mode"] in (None, "0755")
            and failed_root["fresh"] in (None, True)
            and failed_root["tmp_mode"] in (None, "0700")
            and failed_root["tmp_empty"] in (None, True, False)
            and failed_root["entries_sha256"] is None,
            "replay-receipt-failed-root-shape",
        )
        if failed_root["entries_count"] is not None:
            _replay_json_int(
                failed_root["entries_count"],
                "replay-receipt-failed-root-entries", 0,
            )
    return {
        "payload": receipt, "created": created, "clock_valid": clock_valid,
        "synthetic": synthetic, "same_epoch": same_epoch,
        "commands": command_results,
    }


def _replay_identity(value, reason):
    exact_keys(value, ("path", "bytes", "mode", "sha256"), reason)
    require(isinstance(value["path"], str) and value["path"] != "", reason)
    _replay_json_int(value["bytes"], reason, 1)
    require(re.fullmatch(r"[0-7]{4}", value["mode"]) is not None, reason)
    require(HEX64_RE.fullmatch(value["sha256"]) is not None, reason)


def _replay_stable_evidence(
    plan, evidence, candidate_identity, receipt_record, implementation_commit,
):
    exact_keys(
        evidence,
        (
            "schema", "candidate_identity", "candidate_identity_payload", "admin",
            "frozen_governance", "portability", "preexecution_receipt",
        ),
        "replay-stable-evidence-keys",
    )
    require(
        evidence["schema"] == CAMPAIGN_SCHEMA_PREFIX + "stable-evidence-v1",
        "replay-stable-evidence-schema",
    )
    require(evidence["candidate_identity"] == candidate_identity, "replay-stable-candidate")
    payload = evidence["candidate_identity_payload"]
    require(isinstance(payload, dict), "replay-stable-candidate-payload")
    require(
        sha256_bytes(canonical_json(payload)) == candidate_identity,
        "replay-stable-candidate-digest",
    )
    require(
        payload.get("schema") == CAMPAIGN_SCHEMA_PREFIX + "candidate-identity-v1"
        and payload.get("implementation_commit") == implementation_commit
        and payload.get("plan") == plan_reference(),
        "replay-stable-candidate-binding",
    )
    receipt_reference = record_reference(plan, "preexecution_receipts", receipt_record)
    expected_candidate_receipt = {
        "path": receipt_reference["path"], "bytes": receipt_reference["bytes"],
        "sha256": receipt_reference["sha256"], "schema": receipt_reference["schema"],
        "passed": True,
    }
    require(
        payload.get("preexecution_receipt") == expected_candidate_receipt,
        "replay-stable-candidate-receipt",
    )
    _replay_reference(
        plan, evidence["preexecution_receipt"], "preexecution_receipts", receipt_record
    )
    admin = evidence["admin"]
    exact_keys(
        admin,
        (
            "head", "parent", "tree", "plan_commit", "implementation_commit",
            "bounded_tree_sha256", "bounded_worktree_sha256",
        ),
        "replay-stable-admin-keys",
    )
    require(
        admin["head"] == implementation_commit
        and admin["implementation_commit"] == implementation_commit
        and admin["parent"] == PLAN_COMMIT and admin["plan_commit"] == PLAN_COMMIT,
        "replay-stable-admin-commits",
    )
    require(HEX40_RE.fullmatch(admin["tree"]) is not None, "replay-stable-admin-tree")
    require(
        HEX64_RE.fullmatch(admin["bounded_tree_sha256"]) is not None
        and HEX64_RE.fullmatch(admin["bounded_worktree_sha256"]) is not None,
        "replay-stable-admin-digests",
    )
    governance = evidence["frozen_governance"]
    require(isinstance(governance, list) and governance, "replay-stable-governance")
    governance_paths = []
    for identity in governance:
        _replay_identity(identity, "replay-stable-governance-identity")
        governance_paths.append(identity["path"])
    require(
        governance_paths == sorted(set(governance_paths), key=lambda item: item.encode("utf-8")),
        "replay-stable-governance-order",
    )
    require(
        governance == frozen_governance_projection(plan),
        "replay-stable-governance-derived",
    )
    portability = evidence["portability"]
    exact_keys(
        portability,
        (
            "prior_sha256", "new_sha256", "dependency_count",
            "dependency_records_sha256", "sdk_alias_records",
        ),
        "replay-stable-portability-keys",
    )
    require(
        portability["prior_sha256"]
        == "b058be2c2fccac9c4f907ba9e81ba0a372a024307a9ed62541419411afd2a5eb"
        and HEX64_RE.fullmatch(portability["new_sha256"]) is not None
        and HEX64_RE.fullmatch(portability["dependency_records_sha256"]) is not None,
        "replay-stable-portability-digest",
    )
    _replay_json_int(portability["dependency_count"], "replay-stable-dependency-count", 1)
    require(
        portability["sdk_alias_records"] == sdk_alias_records_projection(),
        "replay-stable-aliases",
    )
    dependency_closure = payload.get("dependency_closure")
    require(
        isinstance(dependency_closure, dict)
        and portability["dependency_count"] == dependency_closure.get("count")
        and portability["dependency_records_sha256"]
        == dependency_closure.get("records_sha256"),
        "replay-stable-portability-closure",
    )
    new_object = {
        "dependency_count": portability["dependency_count"],
        "dependency_records_sha256": portability["dependency_records_sha256"],
        "sdk_alias_records": portability["sdk_alias_records"],
    }
    require(
        portability["new_sha256"] == sha256_bytes(canonical_json(new_object)),
        "replay-stable-portability-derived",
    )
    return payload


def _replay_stage_claim(
    plan, record, stage, receipt_record, receipt_replay, candidate_identity,
    predecessor_report_records,
):
    claim = _replay_record(plan, "claims", record)["payload"]
    require(claim["schema"] == RECORD_SCHEMAS["claim"], "replay-claim-schema")
    require(claim["stage"] == stage and claim["candidate_identity"] == candidate_identity, "replay-claim-routing")
    validate_lock_epoch(claim["lock_epoch"])
    require(claim["plan"] == plan_reference(), "replay-claim-plan")
    implementation_commit = receipt_replay["payload"]["claim"]["path"].rsplit("/", 1)[-1][:-5]
    require(claim["implementation_commit"] == implementation_commit, "replay-claim-implementation")
    require(claim["one_shot"] is True and claim["core_dump_disabled"] is True, "replay-claim-flags")
    require(claim["environment"] == exact_environment(plan), "replay-claim-environment")
    require(claim["outer_invocation"] == plan["execution_policy"]["outer_cli"]["run"], "replay-claim-outer")
    _replay_reference(
        plan, claim["preexecution_receipt"], "preexecution_receipts", receipt_record
    )
    schedule, timeouts, bank_metadata, executable_roles = stage_specification(plan, stage)
    require(claim["argv_schedule"] == schedule, "replay-claim-schedule")
    require(
        claim["argv_schedule_sha256"] == sha256_bytes(canonical_json(schedule)),
        "replay-claim-schedule-digest",
    )
    require(claim["child_timeout_seconds"] == timeouts, "replay-claim-timeouts")
    require(
        claim["timeout_seconds"] == _stage_aggregate_seconds(plan),
        "replay-claim-aggregate-timeout",
    )
    require(claim["authorized_bank_plan_metadata"] == bank_metadata, "replay-claim-bank-metadata")
    _base_configuration, stage_digests = configuration_digests(plan)
    require(claim["configuration_sha256"] == stage_digests[stage], "replay-claim-configuration")
    binary_records = receipt_replay["payload"]["binary_records"]
    expected_executables = [{
        "path": binary_records[role]["path"], "bytes": binary_records[role]["bytes"],
        "mode": binary_records[role]["mode"], "sha256": binary_records[role]["sha256"],
    } for role in executable_roles]
    require(claim["executables"] == expected_executables, "replay-claim-executables")
    executable_paths = [item["path"] for item in claim["executables"]]
    require(len(executable_paths) == len(set(executable_paths)), "replay-claim-executable-unique")
    require(
        all(command and command[0] in executable_paths for command in schedule)
        and all(any(command[0] == path for command in schedule) for path in executable_paths),
        "replay-claim-executable-use",
    )
    expected_predecessors = [
        record_reference(plan, "reports", item) for item in predecessor_report_records
    ]
    require(claim["predecessor_reports"] == expected_predecessors, "replay-claim-predecessors")
    require(
        len(predecessor_report_records) == STAGE_ORDER.index(stage)
        and all(item["payload"]["acceptable"] is True for item in predecessor_report_records),
        "replay-claim-predecessor-prefix",
    )
    evidence = claim["evidence_before"]
    require(
        claim["evidence_before_sha256"] == sha256_bytes(canonical_json(evidence)),
        "replay-claim-evidence-digest",
    )
    _replay_stable_evidence(
        plan, evidence, candidate_identity, receipt_record, implementation_commit
    )
    prechecked = _replay_process(claim["preclaim_process"], plan, True)
    claimed = parse_timestamp(claim["claimed_utc"], True)
    predecessor_created = (
        receipt_replay["created"] if not predecessor_report_records
        else parse_timestamp(predecessor_report_records[-1]["payload"]["created_utc"], True)
    )
    expected_epoch = (
        receipt_replay["payload"]["lock_epoch"] if not predecessor_report_records
        else predecessor_report_records[-1]["payload"]["lock_epoch"]
    )
    require(claim["lock_epoch"] == expected_epoch, "replay-claim-epoch-transition")
    deadline = parse_timestamp(CAMPAIGN_DEADLINE, False)
    clock_valid = (
        _replay_clock_order(predecessor_created, prechecked, claimed, deadline)
        and (claimed - prechecked).total_seconds() <= 60.0
        and (deadline - claimed).total_seconds()
        >= _continuation_reserve_seconds(plan)
    )
    return {
        "payload": claim, "claimed": claimed, "clock_valid": clock_valid,
        "prechecked": prechecked,
    }


def _replay_stage_execution(plan, record, claim_record, claim_replay):
    execution = _replay_record(plan, "executions", record)["payload"]
    claim = claim_replay["payload"]
    _replay_reference(plan, execution["claim"], "claims", claim_record)
    validate_lock_epoch(execution["lock_epoch"])
    require(execution["lock_epoch"] == claim["lock_epoch"], "replay-execution-epoch")
    require(
        execution["candidate_identity"] == claim["candidate_identity"]
        and execution["stage"] == claim["stage"],
        "replay-execution-routing",
    )
    echo_keys = (
        "candidate_identity", "stage", "lock_epoch", "core_dump_disabled",
        "executables", "argv_schedule", "argv_schedule_sha256",
        "configuration_sha256", "environment", "outer_invocation",
        "timeout_seconds", "child_timeout_seconds", "authorized_bank_plan_metadata",
    )
    require(
        execution["claim_payload_echo"] == {key: claim[key] for key in echo_keys},
        "replay-execution-claim-echo",
    )
    schedule = claim["argv_schedule"]
    expected_children = len(schedule)
    require(execution["expected_children"] == expected_children, "replay-execution-expected-count")
    require(
        isinstance(execution["children"], list)
        and len(execution["children"]) <= expected_children,
        "replay-execution-child-prefix",
    )
    require(
        isinstance(execution["resolved_executable_paths"], list)
        and len(execution["resolved_executable_paths"]) == len(execution["children"]),
        "replay-execution-resolved-prefix",
    )
    started = parse_timestamp(execution["started_utc"], True)
    ended = parse_timestamp(execution["ended_utc"], True)
    created = parse_timestamp(execution["created_utc"], True)
    elapsed = _replay_finite_nonnegative(
        execution["elapsed_monotonic_seconds"], "replay-execution-elapsed"
    )
    children = []
    previous = claim_replay["prechecked"]
    clock_valid = _replay_clock_order(claim_replay["claimed"], started)
    elapsed_sum = 0.0
    for index, child in enumerate(execution["children"]):
        replayed = _replay_child_result(
            plan, child, schedule[index], {}, index, False
        )
        children.append(replayed)
        require(
            execution["resolved_executable_paths"][index] == replayed["resolved"],
            "replay-execution-resolved-binding",
        )
        matches = [
            item for item in claim["executables"] if item["path"] == schedule[index][0]
        ]
        require(len(matches) == 1, "replay-execution-executable-binding")
        effective_cap = min(
            float(claim["child_timeout_seconds"][index]),
            max(
                0.0,
                _stage_aggregate_seconds(plan)
                - (replayed["started"] - started).total_seconds(),
            ),
            max(
                0.0,
                (
                    parse_timestamp(CAMPAIGN_DEADLINE, False)
                    - replayed["started"]
                ).total_seconds(),
            ),
        )
        clock_valid = (
            clock_valid and replayed["clock_valid"]
            and started <= replayed["started"]
            and previous <= replayed["started"]
            and (replayed["started"] - previous).total_seconds() <= 60.0
        )
        previous = replayed["postchecked"]
        elapsed_sum += replayed["elapsed"]
        if not replayed["successful"]:
            require(index + 1 == len(execution["children"]), "replay-execution-after-failure")
        clock_valid = (
            clock_valid
            and replayed["elapsed"] <= effective_cap
        )
    require(
        execution["all_children_completed"] is (len(children) == expected_children),
        "replay-execution-completion-derived",
    )
    clock_valid = (
        clock_valid and _replay_clock_order(previous, ended, created)
        and _replay_wall_matches(started, ended, elapsed)
        and elapsed <= _stage_aggregate_seconds(plan)
        and elapsed_sum <= _stage_aggregate_seconds(plan)
        and created <= parse_timestamp(CAMPAIGN_DEADLINE, False)
    )
    return {
        "payload": execution, "children": children, "started": started,
        "ended": ended, "created": created, "elapsed": elapsed,
        "clock_valid": clock_valid,
    }


def _replay_parsed_report(plan, stage, parsed, execution_replay):
    if parsed is None:
        return False
    try:
        if stage == STAGE_ORDER[0]:
            exact_keys(
                parsed,
                ("schema", "child_outputs", "natural_activation_count", "timing_panels"),
                "replay-stage0-parsed-keys",
            )
            require(
                parsed["schema"] == CAMPAIGN_SCHEMA_PREFIX + "stage0-parsed-v1",
                "replay-stage0-parsed-schema",
            )
            outputs = parsed["child_outputs"]
            require(
                isinstance(outputs, list)
                and len(outputs) == len(execution_replay["payload"]["children"]),
                "replay-stage0-output-count",
            )
            raw_outputs = []
            for value, child in zip(outputs, execution_replay["payload"]["children"]):
                raw = canonical_json(value)
                receipt = child["stdout_receipt"]
                require(
                    receipt["truncated"] is False
                    and receipt["bytes"] == len(raw)
                    and receipt["sha256"] == sha256_bytes(raw),
                    "replay-stage0-output-binding",
                )
                require(
                    child["stderr_receipt"] == {
                        "bytes": 0, "sha256": EMPTY_SHA256, "truncated": False,
                    },
                    "replay-stage0-stderr-binding",
                )
                raw_outputs.append(raw)
            require(parse_stage0_outputs(raw_outputs, plan) == parsed, "replay-stage0-derived")
        else:
            require(isinstance(parsed, dict), "replay-game-parsed-object")
            stdout_ascii = parsed.get("stdout_ascii")
            require(isinstance(stdout_ascii, str) and stdout_ascii.isascii(), "replay-game-stdout-ascii")
            raw = stdout_ascii.encode("ascii")
            child = execution_replay["payload"]["children"][0]
            require(
                child["stdout_receipt"]["truncated"] is False
                and child["stdout_receipt"]["bytes"] == len(raw)
                and child["stdout_receipt"]["sha256"] == sha256_bytes(raw),
                "replay-game-output-binding",
            )
            require(
                child["stderr_receipt"] == {
                    "bytes": 0, "sha256": EMPTY_SHA256, "truncated": False,
                },
                "replay-game-stderr-binding",
            )
            require(parse_game_stdout(raw, plan, stage) == parsed, "replay-game-derived")
        return True
    except BaseException:
        return False


def _replay_stage_report(
    plan, record, claim_record, claim_replay, execution_record, execution_replay,
    receipt_record, predecessor_report_records,
):
    report = _replay_record(plan, "reports", record)["payload"]
    claim = claim_replay["payload"]
    execution = execution_replay["payload"]
    stage = claim["stage"]
    _replay_reference(plan, report["claim"], "claims", claim_record)
    _replay_reference(plan, report["execution"], "executions", execution_record)
    validate_lock_epoch(report["lock_epoch"])
    require(
        report["lock_epoch"] == claim["lock_epoch"] == execution["lock_epoch"],
        "replay-report-epoch",
    )
    require(
        report["candidate_identity"] == claim["candidate_identity"]
        and report["stage"] == stage,
        "replay-report-routing",
    )
    require(report["predecessor_reports"] == claim["predecessor_reports"], "replay-report-predecessors")
    require(
        report["evidence_before"] == claim["evidence_before"]
        and report["evidence_before_sha256"] == claim["evidence_before_sha256"],
        "replay-report-evidence-before",
    )
    implementation_commit = claim["implementation_commit"]
    postflight_error = report["postflight_error"]
    if postflight_error is None:
        require(
            report["evidence_after"] is not None
            and report["evidence_after_sha256"]
            == sha256_bytes(canonical_json(report["evidence_after"])),
            "replay-report-evidence-after",
        )
        _replay_stable_evidence(
            plan, report["evidence_after"], claim["candidate_identity"],
            receipt_record, implementation_commit,
        )
    else:
        _replay_simple_error(postflight_error, "replay-report-postflight-error")
        require(
            report["evidence_after"] is None and report["evidence_after_sha256"] is None,
            "replay-report-postflight-null-shape",
        )
    stable_expected = (
        postflight_error is None
        and report["evidence_after"] == report["evidence_before"]
        and report["evidence_after_sha256"] == report["evidence_before_sha256"]
    )
    require(report["stable"] is stable_expected, "replay-report-stable-derived")
    postflight_utc = parse_timestamp(report["postflight_utc"], True)
    created = parse_timestamp(report["created_utc"], True)
    postflight_elapsed = _replay_finite_nonnegative(
        report["postflight_elapsed_monotonic_seconds"], "replay-report-postflight-elapsed"
    )
    chronology_valid = (
        execution_replay["clock_valid"]
        and _replay_clock_order(execution_replay["created"], postflight_utc, created)
        and execution_replay["elapsed"] <= postflight_elapsed <= min(
            _continuation_reserve_seconds(plan),
            execution_replay["elapsed"] + _postflight_reserve_seconds(plan),
        )
        and _replay_wall_matches(execution_replay["started"], postflight_utc, postflight_elapsed)
        and created <= parse_timestamp(CAMPAIGN_DEADLINE, False)
    )
    children_ok = (
        execution["all_children_completed"] is True
        and len(execution_replay["children"]) == execution["expected_children"]
        and all(item["successful"] for item in execution_replay["children"])
        and all(
            child["stderr_receipt"] == {
                "bytes": 0, "sha256": EMPTY_SHA256, "truncated": False,
            }
            for child in execution["children"]
        )
    )
    process_ok = chronology_valid and stable_expected and children_ok
    process_errors = [] if process_ok else ["report-process-or-chronology-failure"]
    validation_ok = _replay_parsed_report(plan, stage, report["parsed"], execution_replay)
    validation_errors = [] if validation_ok else ["report-validation-failure"]
    threshold_ok = False
    if validation_ok:
        try:
            threshold_ok = bool(_stage_threshold_passed(
                stage, report["parsed"], predecessor_report_records, plan
            ))
        except BaseException:
            threshold_ok = False
    threshold_errors = [] if threshold_ok else ["report-threshold-failure"]
    require(report["process_errors"] == process_errors, "replay-report-process-errors")
    require(report["validation_errors"] == validation_errors, "replay-report-validation-errors")
    require(report["threshold_errors"] == threshold_errors, "replay-report-threshold-errors")
    acceptable = not process_errors and not validation_errors and not threshold_errors
    require(report["acceptable"] is acceptable, "replay-report-acceptable-derived")
    return {
        "payload": report, "created": created, "clock_valid": chronology_valid,
        "acceptable": acceptable,
    }


def _replay_terminal_observation(
    plan, observation, replay, decision_created, latest_predecessor_time,
    continuation_reserve_required,
):
    keys = (
        "schema", "evidence_checked_utc", "stable_evidence",
        "stable_evidence_sha256", "evidence_error", "process",
        "process_error", "errors",
    )
    exact_keys(observation, keys, "replay-terminal-observation-keys")
    synthetic_expected = synthetic_interrupted_decision_observation()
    if observation["schema"] == CAMPAIGN_SCHEMA_PREFIX + "interrupted-decision-observation-v1":
        require(observation == synthetic_expected, "replay-terminal-synthetic-exact")
        return {
            "synthetic": True, "valid": False, "clock_valid": True,
            "errors": ["interrupted-decision-publication"],
        }
    require(
        observation["schema"] == CAMPAIGN_SCHEMA_PREFIX + "terminal-observation-v1",
        "replay-terminal-observation-schema",
    )
    evidence_checked = parse_timestamp(observation["evidence_checked_utc"], True)
    evidence_valid = False
    if observation["evidence_error"] is None:
        require(
            observation["stable_evidence"] is not None
            and observation["stable_evidence_sha256"]
            == sha256_bytes(canonical_json(observation["stable_evidence"])),
            "replay-terminal-evidence-shape",
        )
        _replay_stable_evidence(
            plan, observation["stable_evidence"], replay["candidate_identity"],
            replay["receipt_record"], replay["preclaim"]["payload"]["implementation_commit"],
        )
        expected_evidence = None
        if replay["reports"]:
            expected_evidence = replay["reports"][
                max(replay["reports"], key=STAGE_ORDER.index)
            ]["payload"]["evidence_after"]
        elif replay["claims"]:
            expected_evidence = replay["claims"][
                max(replay["claims"], key=STAGE_ORDER.index)
            ]["payload"]["evidence_before"]
        evidence_valid = expected_evidence is None or observation["stable_evidence"] == expected_evidence
    else:
        _replay_simple_error(observation["evidence_error"], "replay-terminal-evidence-error")
        require(
            observation["stable_evidence"] is None
            and observation["stable_evidence_sha256"] is None,
            "replay-terminal-evidence-null-shape",
        )
    if observation["process_error"] is None:
        require(observation["process"] is not None, "replay-terminal-process-shape")
        process_checked = _replay_process(observation["process"], plan)
        process_valid = observation["process"]["clean"] is True
    else:
        require(observation["process"] is None, "replay-terminal-process-null-shape")
        process_checked = _replay_process_error(observation["process_error"], plan)
        process_valid = False
    deadline = parse_timestamp(CAMPAIGN_DEADLINE, False)
    clock_valid = _replay_clock_order(
        latest_predecessor_time, evidence_checked, process_checked, decision_created
    ) and decision_created <= deadline
    reserve_valid = (
        not continuation_reserve_required
        or (deadline - process_checked).total_seconds()
        >= _continuation_reserve_seconds(plan)
    )
    valid = evidence_valid and process_valid and clock_valid and reserve_valid
    expected_errors = [] if valid else ["terminal-observation-failure"]
    require(observation["errors"] == expected_errors, "replay-terminal-errors-derived")
    return {
        "synthetic": False, "valid": valid, "clock_valid": clock_valid and reserve_valid,
        "errors": expected_errors,
    }


def _replay_decision(
    plan, record, preclaim_record, receipt_record, registries, replay,
):
    decision = _replay_record(plan, "decisions", record)["payload"]
    preclaim = replay["preclaim"]["payload"]
    receipt = replay["receipt"]["payload"]
    validate_lock_epoch(decision["lock_epoch"])
    _replay_reference(plan, decision["preexecution_claim"], "preexecution_claims", preclaim_record)
    _replay_reference(plan, decision["preexecution_receipt"], "preexecution_receipts", receipt_record)
    require(
        decision["plan"] == preclaim["plan"] == plan_reference()
        and decision["implementation_commit"] == preclaim["implementation_commit"],
        "replay-decision-governance",
    )
    expected_claim_refs = [
        record_reference(plan, "claims", replay["claim_records"][stage])
        for stage in STAGE_ORDER if stage in replay["claim_records"]
    ]
    expected_execution_refs = [
        record_reference(plan, "executions", replay["execution_records"][stage])
        for stage in STAGE_ORDER if stage in replay["execution_records"]
    ]
    expected_report_refs = [
        record_reference(plan, "reports", replay["report_records"][stage])
        for stage in STAGE_ORDER if stage in replay["report_records"]
    ]
    require(decision["reached_claims"] == expected_claim_refs, "replay-decision-claim-refs")
    require(decision["reached_executions"] == expected_execution_refs, "replay-decision-execution-refs")
    require(decision["reached_reports"] == expected_report_refs, "replay-decision-report-refs")
    expected_cardinalities = {
        "preexecution_claims": 1, "preexecution_receipts": 1,
        "claims": len(expected_claim_refs),
        "executions": len(expected_execution_refs),
        "reports": len(expected_report_refs), "decisions": 1,
    }
    require(decision["registry_cardinalities"] == expected_cardinalities, "replay-decision-cardinalities")
    for key in (
        "arena_authorization", "fresh_bank_campaign_authorization",
        "heldout_qualification", "retry_authorized",
        "source_activation_authorization", "upload_authorization",
    ):
        require(decision[key] is False, "replay-decision-authorization")
    require(
        isinstance(decision["decision_chronology_valid"], bool)
        and isinstance(decision["development_selection_acceptable"], bool),
        "replay-decision-boolean",
    )
    statuses = (
        "terminal-development-preexecution-rejection",
        "terminal-development-interrupted-execution",
        "terminal-development-interrupted-postflight",
        "terminal-development-clock-or-deadline-rejection",
        "terminal-development-rejection",
        "development-selection-acceptable-pending-separate-source-activation-review",
    )
    require(decision["status"] in statuses, "replay-decision-status")
    require(
        decision["terminal_stage"] == "preexecution"
        or decision["terminal_stage"] in STAGE_ORDER,
        "replay-decision-terminal-stage",
    )
    require(
        decision["candidate_identity"] == replay["candidate_identity"]
        and (decision["candidate_identity"] is None) is (receipt["passed"] is False),
        "replay-decision-candidate",
    )
    created = parse_timestamp(decision["created_utc"], True)
    latest_time = replay["receipt"]["created"]
    latest_epoch = receipt["lock_epoch"]
    latest_stage = "preexecution"
    reached_stages = set(replay["claims"]) | set(replay["executions"]) | set(replay["reports"])
    if reached_stages:
        latest_stage = max(reached_stages, key=STAGE_ORDER.index)
        if latest_stage in replay["reports"]:
            latest_time = replay["reports"][latest_stage]["created"]
            latest_epoch = replay["reports"][latest_stage]["payload"]["lock_epoch"]
        elif latest_stage in replay["executions"]:
            latest_time = replay["executions"][latest_stage]["created"]
            latest_epoch = replay["executions"][latest_stage]["payload"]["lock_epoch"]
        else:
            latest_time = replay["claims"][latest_stage]["claimed"]
            latest_epoch = replay["claims"][latest_stage]["payload"]["lock_epoch"]
    require(decision["terminal_stage"] == latest_stage, "replay-decision-terminal-stage-derived")

    claim_stages = [stage for stage in STAGE_ORDER if stage in replay["claims"]]
    execution_stages = [stage for stage in STAGE_ORDER if stage in replay["executions"]]
    report_stages = [stage for stage in STAGE_ORDER if stage in replay["reports"]]
    interruption = None
    branch = "failed-receipt"
    continuation_reserve_required = False
    if receipt["passed"]:
        branch = "passed-prefix"
        if not claim_stages:
            continuation_reserve_required = True
            if decision["terminal_observation"] is None:
                interruption = "interrupted-after-passed-preexecution-before-stage0"
        else:
            stage = claim_stages[-1]
            if stage not in execution_stages:
                branch = "interrupted-execution"
                interruption = "interrupted-execution-after-durable-claim:" + stage
            elif stage not in report_stages:
                branch = "interrupted-postflight"
                interruption = "interrupted-postflight-after-durable-execution:" + stage
            else:
                report = replay["reports"][stage]
                if report["acceptable"] is False:
                    branch = "rejected-report"
                elif stage != STAGE_ORDER[-1]:
                    continuation_reserve_required = True
                    next_stage = STAGE_ORDER[STAGE_ORDER.index(stage) + 1]
                    if next_stage not in claim_stages and decision["terminal_observation"] is None:
                        interruption = (
                            "interrupted-after-accepted-stage0-before-d20"
                            if stage == STAGE_ORDER[0]
                            else "interrupted-after-accepted-d20-before-remainder"
                        )
                elif decision["terminal_observation"] is None:
                    interruption = "interrupted-after-accepted-final-before-decision"

    observation_replay = None
    if decision["terminal_observation"] is not None:
        observation_replay = _replay_terminal_observation(
            plan, decision["terminal_observation"], replay, created, latest_time,
            continuation_reserve_required,
        )
        if continuation_reserve_required and not observation_replay["synthetic"]:
            require(
                observation_replay["valid"] is False,
                "replay-valid-observation-requires-successor",
            )
        if observation_replay["synthetic"]:
            require(
                decision["lock_epoch"] != latest_epoch and receipt["passed"] is True,
                "replay-decision-synthetic-epoch",
            )
            interruption = None
        else:
            require(decision["lock_epoch"] == latest_epoch, "replay-decision-normal-epoch")
    elif interruption is not None:
        require(decision["lock_epoch"] != latest_epoch, "replay-decision-recovery-epoch")
    elif branch == "passed-prefix" and report_stages and replay["reports"][report_stages[-1]]["acceptable"]:
        raise ContractError("replay-decision-missing-observation")
    else:
        require(decision["lock_epoch"] == latest_epoch, "replay-decision-fresh-epoch")

    clock_valid = (
        replay["preclaim"]["clock_valid"] and replay["receipt"]["clock_valid"]
        and all(item["clock_valid"] for item in replay["claims"].values())
        and all(item["clock_valid"] for item in replay["executions"].values())
        and all(item["clock_valid"] for item in replay["reports"].values())
        and latest_time <= created <= parse_timestamp(CAMPAIGN_DEADLINE, False)
        and (observation_replay is None or observation_replay["clock_valid"])
    )
    require(
        decision["decision_chronology_valid"] is clock_valid,
        "replay-decision-chronology-derived",
    )
    if not clock_valid:
        expected_status = "terminal-development-clock-or-deadline-rejection"
    elif observation_replay is not None and observation_replay["synthetic"]:
        expected_status = "terminal-development-rejection"
    elif receipt["passed"] is False:
        expected_status = "terminal-development-preexecution-rejection"
    elif branch == "interrupted-execution":
        expected_status = "terminal-development-interrupted-execution"
    elif branch == "interrupted-postflight":
        expected_status = "terminal-development-interrupted-postflight"
    elif (
        report_stages and report_stages[-1] == STAGE_ORDER[-1]
        and replay["reports"][STAGE_ORDER[-1]]["acceptable"] is True
        and observation_replay is not None and observation_replay["valid"]
    ):
        expected_status = "development-selection-acceptable-pending-separate-source-activation-review"
    else:
        expected_status = "terminal-development-rejection"
    require(decision["status"] == expected_status, "replay-decision-status-derived")
    selected = expected_status == "development-selection-acceptable-pending-separate-source-activation-review"
    require(
        decision["development_selection_acceptable"] is selected,
        "replay-decision-selection-derived",
    )
    expected_errors = list(receipt["errors"])
    if interruption is not None:
        expected_errors.append(interruption)
    for stage in STAGE_ORDER:
        if stage in replay["reports"]:
            report = replay["reports"][stage]["payload"]
            expected_errors.extend(report["process_errors"])
            expected_errors.extend(report["validation_errors"])
            expected_errors.extend(report["threshold_errors"])
    if observation_replay is not None:
        expected_errors.extend(observation_replay["errors"])
    require(decision["errors"] == expected_errors, "replay-decision-errors-derived")


def validate_registry_chain(
    plan, registries, allow_pending=False, expected_candidate_identity=None,
):
    require(
        isinstance(registries, dict)
        and set(registries) == set(REGISTRY_ORDER).union(
            name + "_pending" for name in REGISTRY_ORDER
        ),
        "registry-map-keys",
    )
    if not allow_pending:
        require(all(not registries[name + "_pending"] for name in REGISTRY_ORDER), "registry-pending")
    for registry_name in REGISTRY_ORDER:
        require(isinstance(registries[registry_name], list), "registry-record-array")
        for record in registries[registry_name]:
            _replay_record(plan, registry_name, record)
    preclaims = registries["preexecution_claims"]
    receipts = registries["preexecution_receipts"]
    require(len(preclaims) <= 1 and len(receipts) <= 1, "preexecution-cardinality")
    require(not receipts or len(preclaims) == 1, "receipt-orphan")
    decisions = registries["decisions"]
    require(len(decisions) <= 1, "decision-cardinality")
    if not preclaims:
        require(
            not receipts and not decisions
            and not any(registries[name] for name in ("claims", "executions", "reports")),
            "record-without-preclaim",
        )
        require(expected_candidate_identity is None, "virgin-candidate")
        return {
            "candidate_identity": None, "claims": [], "executions": [],
            "reports": [], "clock_valid": True,
        }

    preclaim_replay = _replay_preexecution_claim(plan, preclaims[0])
    if not receipts:
        require(
            not decisions and not any(registries[name] for name in ("claims", "executions", "reports")),
            "stage-without-receipt",
        )
        require(expected_candidate_identity is None, "preexecution-candidate")
        return {
            "candidate_identity": None, "claims": [], "executions": [],
            "reports": [], "clock_valid": preclaim_replay["clock_valid"],
        }

    receipt_replay = _replay_preexecution_receipt(
        plan, receipts[0], preclaims[0], preclaim_replay
    )
    receipt = receipt_replay["payload"]
    if receipt["passed"]:
        validate_passed_receipt(plan, receipts[0])
    stage_maps = {}
    stage_lists = {}
    for registry_name in ("claims", "executions", "reports"):
        by_stage = {}
        for record in registries[registry_name]:
            stage = record["payload"]["stage"]
            require(stage in STAGE_ORDER and stage not in by_stage, "stage-duplicate")
            by_stage[stage] = record
        ordered = [stage for stage in STAGE_ORDER if stage in by_stage]
        require(ordered == list(STAGE_ORDER[: len(ordered)]), "stage-prefix")
        stage_maps[registry_name] = by_stage
        stage_lists[registry_name] = ordered
    require(
        len(stage_lists["reports"]) <= len(stage_lists["executions"])
        <= len(stage_lists["claims"]),
        "stage-prefix-cardinality",
    )
    require(
        len(stage_lists["claims"]) <= len(stage_lists["reports"]) + 1,
        "stage-successor-without-report",
    )

    candidate_values = []
    for registry_name in ("claims", "executions", "reports"):
        candidate_values.extend(
            record["payload"]["candidate_identity"]
            for record in registries[registry_name]
        )
    if decisions:
        candidate_values.append(decisions[0]["payload"]["candidate_identity"])
    if receipt["passed"]:
        nonnull_candidates = [value for value in candidate_values if value is not None]
        require(
            all(isinstance(value, str) and HEX64_RE.fullmatch(value) is not None for value in nonnull_candidates),
            "candidate-format",
        )
        candidate = expected_candidate_identity
        if candidate is None and nonnull_candidates:
            candidate = nonnull_candidates[0]
        if candidate is not None:
            require(HEX64_RE.fullmatch(candidate) is not None, "derived-candidate-format")
            require(all(value == candidate for value in nonnull_candidates), "candidate-mismatch")
        require(
            all(value is not None for value in candidate_values),
            "passed-receipt-null-candidate",
        )
    else:
        require(expected_candidate_identity is None, "failed-receipt-derived-candidate")
        require(
            not any(registries[name] for name in ("claims", "executions", "reports"))
            and all(value is None for value in candidate_values),
            "stage-without-passed-receipt",
        )
        candidate = None

    claim_replays = {}
    execution_replays = {}
    report_replays = {}
    for index, stage in enumerate(stage_lists["claims"]):
        predecessor_report_records = [
            stage_maps["reports"][previous] for previous in STAGE_ORDER[:index]
        ]
        require(
            len(predecessor_report_records) == index,
            "claim-missing-accepted-predecessor",
        )
        claim_record = stage_maps["claims"][stage]
        claim_replays[stage] = _replay_stage_claim(
            plan, claim_record, stage, receipts[0], receipt_replay, candidate,
            predecessor_report_records,
        )
        if stage not in stage_maps["executions"]:
            continue
        execution_record = stage_maps["executions"][stage]
        execution_replays[stage] = _replay_stage_execution(
            plan, execution_record, claim_record, claim_replays[stage]
        )
        if stage not in stage_maps["reports"]:
            continue
        report_record = stage_maps["reports"][stage]
        report_replays[stage] = _replay_stage_report(
            plan, report_record, claim_record, claim_replays[stage],
            execution_record, execution_replays[stage], receipts[0],
            predecessor_report_records,
        )
    for index, stage in enumerate(stage_lists["claims"][1:], 1):
        previous = STAGE_ORDER[index - 1]
        require(
            previous in report_replays and report_replays[previous]["acceptable"] is True,
            "claim-after-unaccepted-report",
        )

    replay = {
        "preclaim": preclaim_replay,
        "receipt": receipt_replay,
        "claims": claim_replays,
        "executions": execution_replays,
        "reports": report_replays,
        "claim_records": stage_maps["claims"],
        "execution_records": stage_maps["executions"],
        "report_records": stage_maps["reports"],
        "receipt_record": receipts[0],
        "candidate_identity": candidate,
    }
    if decisions:
        _replay_decision(plan, decisions[0], preclaims[0], receipts[0], registries, replay)
    return {
        "candidate_identity": candidate,
        "claims": stage_lists["claims"],
        "executions": stage_lists["executions"],
        "reports": stage_lists["reports"],
        "clock_valid": (
            preclaim_replay["clock_valid"] and receipt_replay["clock_valid"]
            and all(item["clock_valid"] for item in claim_replays.values())
            and all(item["clock_valid"] for item in execution_replays.values())
            and all(item["clock_valid"] for item in report_replays.values())
        ),
    }


def _logical_entry(key_class=None, depth=0, bound="Exact", way=0, incoming=False):
    keys = {
        "incoming": (0x1111111111111111, 0x2222222222222222),
        "collision-a": (0x3333333333333333, 0x4444444444444444),
        "collision-b": (0x5555555555555555, 0x6666666666666666),
    }
    bounds = {"Exact": 0, "Lower": 1, "Upper": 2}
    if key_class is None:
        return {
            "occupied": False, "key_high": "0000000000000000",
            "key_low": "0000000000000000", "depth": 0, "bound": "Exact",
            "score": 0, "move_x": 0, "move_y": 0,
        }
    high, low = keys[key_class]
    bound_index = bounds[bound]
    if incoming:
        score = -200000 - 100 * depth - 10 * bound_index
        move_x = -10 - depth
        move_y = 10 + bound_index
    else:
        key_index = ("incoming", "collision-a", "collision-b").index(key_class)
        score = 100000 + 10000 * way + 1000 * key_index + 100 * depth + 10 * bound_index
        move_x = 10 * way + 3 * key_index + bound_index
        move_y = 10 * depth + bound_index
    return {
        "occupied": True, "key_high": format(high, "016x"), "key_low": format(low, "016x"),
        "depth": depth, "bound": bound, "score": score,
        "move_x": move_x, "move_y": move_y,
    }


def _incoming_without_occupied(depth, bound):
    value = _logical_entry("incoming", depth, bound, incoming=True)
    del value["occupied"]
    return value


def _find_result(entries, key_class):
    key = _logical_entry(key_class)["key_high"], _logical_entry(key_class)["key_low"]
    for way, entry in enumerate(entries):
        if entry["occupied"] and (entry["key_high"], entry["key_low"]) == key:
            return {"way": way, "entry": entry}
    return None


def _store_oracle(before, incoming, candidate):
    entries = [dict(before[0]), dict(before[1])]
    victim = None
    incoming_key = incoming["key_high"], incoming["key_low"]
    for way, entry in enumerate(entries):
        entry_key = entry["key_high"], entry["key_low"]
        if entry["occupied"] and entry_key == incoming_key:
            if entry["depth"] > incoming["depth"]:
                return False, None, entries
            victim = way
            break
        choose = not entry["occupied"] or victim is None or entry["depth"] < entries[victim]["depth"]
        if candidate and victim is not None and entry["occupied"]:
            choose = choose or (
                entry["depth"] == entries[victim]["depth"]
                and entry["bound"] != "Exact" and entries[victim]["bound"] == "Exact"
            )
        if choose:
            victim = way
    if victim is None:
        return False, None, entries
    resident = entries[victim]
    reject = resident["occupied"] and resident["depth"] > incoming["depth"]
    if candidate and resident["occupied"]:
        resident_key = resident["key_high"], resident["key_low"]
        reject = reject or (
            resident_key != incoming_key and resident["depth"] == incoming["depth"]
            and resident["bound"] == "Exact" and incoming["bound"] != "Exact"
        )
    if reject:
        return False, None, entries
    entries[victim] = dict(incoming, occupied=True)
    return True, victim, entries


_TABLE_DIGEST_CACHE = {}


def table_truth_digests(candidate):
    cache_key = bool(candidate)
    if cache_key in _TABLE_DIGEST_CACHE:
        return _TABLE_DIGEST_CACHE[cache_key]
    resident_states = [None]
    for key_class in ("incoming", "collision-a", "collision-b"):
        for depth in range(4):
            for bound in ("Exact", "Lower", "Upper"):
                resident_states.append((key_class, depth, bound))
    cases = []
    case_index = 0
    for first in resident_states:
        for second in resident_states:
            before = [
                _logical_entry() if first is None else _logical_entry(*first, way=0),
                _logical_entry() if second is None else _logical_entry(*second, way=1),
            ]
            for incoming_depth in range(4):
                for incoming_bound in ("Exact", "Lower", "Upper"):
                    incoming = _incoming_without_occupied(incoming_depth, incoming_bound)
                    stored, victim, after = _store_oracle(before, incoming, candidate)
                    nonvictim = True
                    if stored:
                        nonvictim = after[1 - victim] == before[1 - victim]
                    find_results = {
                        name.replace("-", "_"): _find_result(after, name)
                        for name in ("collision-a", "collision-b", "incoming")
                    }
                    cases.append({
                        "case_index": case_index, "before": before, "incoming": incoming,
                        "expected_return": stored, "expected_victim": victim,
                        "expected_after": after, "actual_return": stored,
                        "actual_victim": victim, "actual_after": after,
                        "find_results": find_results, "nonvictim_unchanged": nonvictim,
                    })
                    case_index += 1
    require(case_index == 16428, "table-case-count")
    routing = []
    routing_index = 0
    keys = ((0, 0), (1, 0), (0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF))
    for capacity in (0, 1, 2, 4, 6, 4096, 262144):
        for high, low in keys:
            for depth in range(4):
                for bound in ("Exact", "Lower", "Upper"):
                    if capacity < 2:
                        bucket = None
                        stores = False
                    else:
                        combined = high ^ (((low << 23) & 0xFFFFFFFFFFFFFFFF) | (low >> 41))
                        bucket = 2 * (combined % (capacity // 2))
                        stores = True
                    routing.append({
                        "capacity": capacity, "key_high": format(high, "016x"),
                        "key_low": format(low, "016x"), "depth": depth, "bound": bound,
                        "expected_bucket": bucket, "actual_bucket": bucket,
                        "expected_store": stores, "actual_store": stores,
                        "expected_find": stores, "actual_find": stores,
                    })
                    routing_index += 1
    require(routing_index == 252, "routing-case-count")
    answer = (sha256_bytes(canonical_json(cases)), sha256_bytes(canonical_json(routing)))
    _TABLE_DIGEST_CACHE[cache_key] = answer
    return answer


_STAGE0_MASK64 = (1 << 64) - 1
_STAGE0_DIRECTIONS = (
    (0, -1), (1, -1), (1, 0), (1, 1),
    (0, 1), (-1, 1), (-1, 0), (-1, -1),
)
_STAGE0_TACTICAL_TRANSCRIPTS = (
    "6/1",
    "4/3/6/4/3/0",
    "0/6/5/4/5/53/61/0633",
    "1/1/7/6/0/75/74/3/00523/135/01/13/27435/35",
)


class _ContestState:
    """Independent logical state for the Stage0 CodinGame-rules corpus."""

    __slots__ = (
        "ball", "to_move", "status", "path", "used_segments", "visit_count",
    )

    def __init__(
        self, ball, to_move, status, path, used_segments, visit_count,
    ):
        self.ball = ball
        self.to_move = to_move
        self.status = status
        self.path = path
        self.used_segments = used_segments
        self.visit_count = visit_count


def _strict_bool(value, reason):
    require(type(value) is bool, reason)
    return value


def _strict_int(value, reason, minimum=None, maximum=None):
    require(type(value) is int, reason)
    if minimum is not None:
        require(value >= minimum, reason)
    if maximum is not None:
        require(value <= maximum, reason)
    return value


def _strict_string(value, reason):
    require(type(value) is str, reason)
    return value


def _point_key(point):
    return point[1], point[0]


def _segment(first, second):
    if _point_key(second) < _point_key(first):
        first, second = second, first
    return first, second


def _copy_contest_state(state):
    return _ContestState(
        state.ball,
        state.to_move,
        state.status,
        list(state.path),
        set(state.used_segments),
        dict(state.visit_count),
    )


def _initial_contest_state():
    ball = (4, 6)
    return _ContestState(ball, "one", "in_progress", [ball], set(), {ball: 1})


def _contest_opponent(player):
    require(player in ("one", "two"), "stage0-player")
    return "two" if player == "one" else "one"


def _contest_is_regular(point):
    x, y = point
    return 0 <= x <= 8 and 1 <= y <= 11


def _contest_is_goal(point):
    x, y = point
    return 3 <= x <= 5 and y in (0, 12)


def _contest_is_boundary(point):
    if not _contest_is_regular(point):
        return False
    x, y = point
    if x in (0, 8):
        return True
    return y in (1, 11) and x != 4


def _contest_is_forbidden_boundary_segment(segment):
    first, second = segment
    touches_north = (first[1] == 0 and 3 <= first[0] <= 5) or (
        second[1] == 0 and 3 <= second[0] <= 5
    )
    if touches_north and first[0] == second[0] and first[0] in (3, 5):
        if {first[1], second[1]} == {0, 1}:
            return True
    touches_south = (first[1] == 12 and 3 <= first[0] <= 5) or (
        second[1] == 12 and 3 <= second[0] <= 5
    )
    if touches_south and first[0] == second[0] and first[0] in (3, 5):
        if {first[1], second[1]} == {11, 12}:
            return True
    if not (_contest_is_regular(first) and _contest_is_regular(second)):
        return False
    if not (_contest_is_boundary(first) and _contest_is_boundary(second)):
        return False
    dx = abs(first[0] - second[0])
    dy = abs(first[1] - second[1])
    if first[1] == second[1] and first[1] in (1, 11):
        return dx == 1 and dy == 0
    if first[0] == second[0] and first[0] in (0, 8):
        return dx == 0 and dy == 1
    return False


def _contest_neighbors(state):
    if not _contest_is_regular(state.ball):
        return []
    x, y = state.ball
    result = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            candidate = (x + dx, y + dy)
            if _contest_is_regular(candidate):
                result.append(candidate)
    if 3 <= x <= 5 and y in (1, 11):
        goal_y = 0 if y == 1 else 12
        for goal_x in range(3, 6):
            candidate = (goal_x, goal_y)
            if abs(x - goal_x) <= 1:
                result.append(candidate)
    return result


def _contest_legal_moves(state):
    if state.status != "in_progress":
        return []
    result = []
    for destination in _contest_neighbors(state):
        edge = _segment(state.ball, destination)
        if edge in state.used_segments:
            continue
        if _contest_is_forbidden_boundary_segment(edge):
            continue
        result.append(destination)
    return result


def _contest_apply_move(state, destination, reason):
    require(state.status == "in_progress", reason)
    require(destination in _contest_legal_moves(state), reason)
    result = _copy_contest_state(state)
    result.used_segments.add(_segment(state.ball, destination))
    result.ball = destination
    result.path.append(destination)
    result.visit_count[destination] = result.visit_count.get(destination, 0) + 1
    if _contest_is_goal(destination):
        result.status = "won_by_one" if destination[1] == 0 else "won_by_two"
        return result
    extra_turn = _contest_is_boundary(destination) or (
        state.visit_count.get(destination, 0) > 0
    )
    result.to_move = state.to_move if extra_turn else _contest_opponent(state.to_move)
    result.status = "in_progress"
    if not _contest_legal_moves(result):
        # The Stage0 contest configuration is MoverLoses.
        result.status = "won_by_two" if state.to_move == "one" else "won_by_one"
    return result


def _replay_contest_action(initial, action, reason):
    _strict_string(action, reason)
    require(action != "" and all("0" <= char <= "7" for char in action), reason)
    state = _copy_contest_state(initial)
    mover = state.to_move
    for character in action:
        require(state.status == "in_progress" and state.to_move == mover, reason)
        dx, dy = _STAGE0_DIRECTIONS[ord(character) - ord("0")]
        destination = (state.ball[0] + dx, state.ball[1] + dy)
        state = _contest_apply_move(state, destination, reason)
    require(state.status != "in_progress" or state.to_move != mover, reason)
    return state


def _rotate_contest_state(source):
    def rotate(point):
        return 8 - point[0], 12 - point[1]

    statuses = {
        "in_progress": "in_progress",
        "won_by_one": "won_by_two",
        "won_by_two": "won_by_one",
    }
    require(source.status in statuses, "stage0-rotation-status")
    return _ContestState(
        rotate(source.ball),
        _contest_opponent(source.to_move),
        statuses[source.status],
        [rotate(point) for point in source.path],
        {_segment(rotate(edge[0]), rotate(edge[1])) for edge in source.used_segments},
        {rotate(point): count for point, count in source.visit_count.items()},
    )


def _rotate_action(action):
    _strict_string(action, "stage0-rotation-action")
    require(action != "" and all("0" <= char <= "7" for char in action), "stage0-rotation-action")
    return "".join(chr(ord("0") + (ord(char) - ord("0") + 4) % 8) for char in action)


def _contest_state_hash(state):
    """Match opening_bank::state_hash after normalizing only RulesConfig."""

    require(state.to_move in ("one", "two"), "stage0-state-player")
    require(state.status in ("in_progress", "won_by_one", "won_by_two"), "stage0-state-status")
    segments = sorted(
        state.used_segments,
        key=lambda edge: (
            edge[0][1], edge[0][0], edge[1][1], edge[1][0],
        ),
    )
    visits = sorted(state.visit_count.items(), key=lambda item: _point_key(item[0]))
    require(all(type(count) is int and count > 0 for _point, count in visits), "stage0-state-visits")
    lines = [
        "papersoccer.logical-game-state.v1",
        "rules=8x10;opponent_goal_only;player_to_move_loses",
        "ball=" + str(state.ball[0]) + "," + str(state.ball[1]),
        "to_move=" + state.to_move,
        "status=" + state.status,
        "segments=" + str(len(segments)),
    ]
    for first, second in segments:
        lines.append(
            str(first[0]) + "," + str(first[1]) + "-"
            + str(second[0]) + "," + str(second[1])
        )
    lines.append("visits=" + str(len(visits)))
    for point, count in visits:
        lines.append(str(point[0]) + "," + str(point[1]) + ":" + str(count))
    return sha256_bytes(("\n".join(lines) + "\n").encode("ascii"))


def _next_xorshift64(state):
    state = (state ^ ((state << 13) & _STAGE0_MASK64)) & _STAGE0_MASK64
    state = (state ^ (state >> 7)) & _STAGE0_MASK64
    state = (state ^ ((state << 17) & _STAGE0_MASK64)) & _STAGE0_MASK64
    return state


def _reconstruct_tactical_state(transcript):
    _strict_string(transcript, "stage0-tactical-transcript")
    turns = transcript.split("/")
    require(turns and all(turn != "" for turn in turns), "stage0-tactical-transcript")
    state = _initial_contest_state()
    for turn in turns:
        state = _replay_contest_action(state, turn, "stage0-tactical-action")
    return state


def _build_stage0_corpus(plan):
    corpus_plan = plan["stage0"]["corpus"]
    require(corpus_plan["seed_hex"] == "0x4f1bbcdc676f2b31", "stage0-corpus-seed")
    require(_strict_int(corpus_plan["procedural_live_states"], "stage0-corpus-count") == 1000, "stage0-corpus-count")
    require(_strict_int(corpus_plan["total_base_states"], "stage0-corpus-total") == 1004, "stage0-corpus-total")
    require(
        type(corpus_plan["tactical_transcripts"]) is list
        and tuple(corpus_plan["tactical_transcripts"]) == _STAGE0_TACTICAL_TRANSCRIPTS,
        "stage0-corpus-transcripts",
    )
    random_state = int(corpus_plan["seed_hex"], 16)
    procedural = []
    state = _initial_contest_state()
    while len(procedural) < 1000:
        if state.status != "in_progress":
            state = _initial_contest_state()
        procedural.append(_copy_contest_state(state))
        legal = _contest_legal_moves(state)
        require(legal, "stage0-corpus-live-state")
        random_state = _next_xorshift64(random_state)
        state = _contest_apply_move(
            state, legal[random_state % len(legal)], "stage0-corpus-selected-move",
        )
    base_states = procedural + [
        _reconstruct_tactical_state(transcript)
        for transcript in _STAGE0_TACTICAL_TRANSCRIPTS
    ]
    require(len(base_states) == 1004, "stage0-corpus-total")
    pairs = []
    identities = []
    for index, base in enumerate(base_states):
        rotated = _rotate_contest_state(base)
        base_hash = _contest_state_hash(base)
        rotated_hash = _contest_state_hash(rotated)
        pairs.append((base, rotated, base_hash, rotated_hash))
        identities.append({
            "state_index": index,
            "base_state_sha256": base_hash,
            "rotated_state_sha256": rotated_hash,
        })
    return {
        "pairs": pairs,
        "identities": identities,
        "corpus_sha256": sha256_bytes(canonical_json(identities)),
    }


def _timing_panel_state(corpus, panel, state_index):
    _strict_string(panel, "timing-panel")
    _strict_int(state_index, "timing-state-index", 0)
    if panel == "forced-prod":
        require(state_index < 8, "timing-state-index")
        base_index = 1000 + state_index // 2
    elif panel == "mixed-prod":
        require(state_index < 64, "timing-state-index")
        base_index = state_index // 2
    else:
        raise ContractError("timing-panel")
    pair = corpus["pairs"][base_index]
    if state_index % 2 == 0:
        return pair[0], pair[2]
    return pair[1], pair[3]


def _parse_child_json(raw):
    value = decode_json(raw)
    require(isinstance(value, dict), "child-json-object")
    return value


def _require_uint(value, reason="unsigned-token"):
    require(isinstance(value, str) and CANONICAL_UINT_RE.fullmatch(value) is not None, reason)
    return int(value)


def _require_fixed3(value, reason="fixed3-token"):
    require(isinstance(value, str) and FIXED3_RE.fullmatch(value) is not None, reason)
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise ContractError(reason) from error
    require(parsed.is_finite() and parsed >= 0, reason)
    return parsed


def round3(total, count):
    require(isinstance(total, int) and total >= 0, "round3-total")
    require(isinstance(count, int) and count > 0, "round3-count")
    scaled = 1000 * total
    base, remainder = divmod(scaled, count)
    require(2 * remainder != count, "round3-half-tie")
    rounded = base if 2 * remainder < count else base + 1
    return str(rounded // 1000) + "." + format(rounded % 1000, "03d")


def allowed_round3_totals(token, count, maximum):
    _require_fixed3(token)
    require(maximum >= 0, "round3-maximum")
    result = []
    for total in range(count * maximum + 1):
        try:
            if round3(total, count) == token:
                result.append(total)
        except ContractError:
            continue
    require(result, "round3-unrealizable")
    return result


def parse_token_line(line, kind, order):
    prefix = kind + " "
    require(line.startswith(prefix) and line != prefix, "output-kind")
    tokens = line[len(prefix):].split(" ")
    require(tokens and all(token != "" and token.count("=") == 1 for token in tokens), "output-token")
    pairs = [token.split("=", 1) for token in tokens]
    require([pair[0] for pair in pairs] == list(order), "output-order")
    require(all(pair[1] != "" for pair in pairs), "output-value")
    return {key: value for key, value in pairs}


def parse_game_stdout(raw, plan, stage):
    require(raw and raw.endswith(b"\n") and b"\r" not in raw and b"\x00" not in raw, "game-output-shape")
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise ContractError("game-output-ascii") from error
    lines = text[:-1].split("\n")
    bank_count = 1 if stage == "development_d20" else 3
    require(len(lines) == bank_count + 2 and all(lines), "game-output-lines")
    summary_order = plan["summary_validation"]["summary_output_order_exact"]
    configuration_order = plan["summary_validation"]["configuration_output_order_exact"]
    banks = [parse_token_line(lines[index], "bank_summary", summary_order) for index in range(bank_count)]
    aggregate = parse_token_line(lines[-2], "summary", summary_order)
    configuration = parse_token_line(lines[-1], "configuration", configuration_order)
    require([bank["bank"] for bank in banks] == [str(index) for index in range(bank_count)], "game-bank-index")
    require(aggregate["bank"] == "all", "game-aggregate-bank")
    parsed = {
        "schema": CAMPAIGN_SCHEMA_PREFIX + "game-parsed-v1", "stage": stage,
        "stdout_ascii": text, "stdout_sha256": sha256_bytes(raw),
        "banks": banks, "aggregate": aggregate, "configuration": configuration,
    }
    validate_game_parsed(parsed, plan)
    return parsed


def _parse_tuple(token, length):
    pieces = token.split("/")
    require(len(pieces) == length, "tuple-arity")
    return [_require_uint(piece, "tuple-token") for piece in pieces]


def _validate_proof(summary, engine):
    values = {}
    for name, length in (("rebound", 3), ("root", 3), ("leaf", 3), ("ply1", 4), ("ply2", 4)):
        values[name] = _parse_tuple(summary[engine + "_proof_" + name], length)
        probes, wins, losses = values[name][:3]
        require(wins <= probes and losses <= probes and wins + losses <= probes, "proof-counts")
        if length == 4:
            require(values[name][3] <= probes, "proof-cutoff")
    for index in range(3):
        require(values["rebound"][index] == sum(values[name][index] for name in ("root", "leaf", "ply1", "ply2")), "proof-additive")
    require(values["ply1"][3] == values["ply1"][1] + values["ply1"][2], "proof-ply1")
    require(values["root"][0] > 0 and values["leaf"][0] > 0 and values["ply1"][0] > 0, "proof-mask7")
    require(values["ply2"] == [0, 0, 0, 0], "proof-mask7-ply2")
    return values


def _validate_game_summary(summary, expected_games, expected_color_games):
    integer_fields = ("games", "candidate_wins", "reference_wins", "unfinished", "failed")
    integers = {name: _require_uint(summary[name]) for name in integer_fields}
    require(integers["games"] == expected_games, "summary-games")
    require(integers["candidate_wins"] + integers["reference_wins"] + integers["unfinished"] == expected_games, "summary-accounting")
    require(integers["failed"] <= integers["unfinished"], "summary-failed")
    require(integers["unfinished"] == 0 and integers["failed"] == 0, "summary-terminal-games")
    p0 = _parse_tuple(summary["candidate_p0"], 5)
    p1 = _parse_tuple(summary["candidate_p1"], 5)
    require(p0[4] == expected_color_games and p1[4] == expected_color_games, "summary-color-games")
    require([p0[index] + p1[index] for index in range(5)] == [
        integers["candidate_wins"], integers["reference_wins"],
        integers["unfinished"], integers["failed"], integers["games"],
    ], "summary-color-addition")
    for engine in ("candidate", "reference"):
        scalar_names = (
            "invocations", "searches", "illegal", "operational", "exceptions",
            "hard_timeouts", "soft_overruns", "nodes", "nodes_p99", "nodes_max",
            "depth_max", "attempted_depth_max", "exhaustions",
        )
        scalar = {name: _require_uint(summary[engine + "_" + name]) for name in scalar_names}
        require(scalar["searches"] == scalar["invocations"] > 0, "summary-searches")
        require(all(scalar[name] == 0 for name in ("illegal", "operational", "exceptions", "hard_timeouts")), "summary-engine-failures")
        require(scalar["nodes_max"] <= scalar["nodes"] <= scalar["searches"] * scalar["nodes_max"], "summary-nodes")
        require(scalar["nodes_max"] <= 3000000 and scalar["nodes_p99"] <= scalar["nodes_max"], "summary-node-cap")
        require(summary[engine + "_nodes_avg"] == round3(scalar["nodes"], scalar["searches"]), "summary-node-average")
        allowed_round3_totals(summary[engine + "_depth_avg"], scalar["searches"], scalar["depth_max"])
        allowed_round3_totals(summary[engine + "_attempted_depth_avg"], scalar["searches"], scalar["attempted_depth_max"])
        first_p99 = _require_fixed3(summary[engine + "_first_ms_p99"])
        first_max = _require_fixed3(summary[engine + "_first_ms_max"])
        later_p99 = _require_fixed3(summary[engine + "_later_ms_p99"])
        later_max = _require_fixed3(summary[engine + "_later_ms_max"])
        require(first_p99 <= first_max < Decimal("990.000") and first_p99 < Decimal("900.000"), "summary-first-clock")
        require(later_p99 <= later_max < Decimal("198.000") and later_p99 < Decimal("180.000"), "summary-later-clock")
        _validate_proof(summary, engine)
    return integers, p0, p1


def validate_game_parsed(parsed, plan):
    exact_keys(parsed, ("schema", "stage", "stdout_ascii", "stdout_sha256", "banks", "aggregate", "configuration"), "game-parsed-keys")
    stage = parsed["stage"]
    require(stage in STAGE_ORDER[1:], "game-stage")
    raw = parsed["stdout_ascii"].encode("ascii")
    require(sha256_bytes(raw) == parsed["stdout_sha256"], "game-stdout-digest")
    summary_keys = set(plan["summary_validation"]["summary_fields_exact"])
    require(all(set(value) == summary_keys for value in parsed["banks"] + [parsed["aggregate"]]), "game-summary-fields")
    require(set(parsed["configuration"]) == set(plan["summary_validation"]["configuration_fields_exact"]), "game-configuration-fields")
    banks_meta = [plan["development_banks"][3]] if stage == "development_d20" else plan["development_banks"][:3]
    for bank, metadata in zip(parsed["banks"], banks_meta):
        _validate_game_summary(bank, metadata["expected_games"], metadata["pairs"])
    stage_spec = next(item for item in plan["stages"] if item["stage"] == stage)
    aggregate_values = _validate_game_summary(parsed["aggregate"], stage_spec["expected_games"], stage_spec["expected_color_games"])
    additive = ("games", "candidate_wins", "reference_wins", "unfinished", "failed")
    for name in additive:
        require(_require_uint(parsed["aggregate"][name]) == sum(_require_uint(bank[name]) for bank in parsed["banks"]), "game-bank-additive")
    for engine in ("candidate", "reference"):
        for suffix in (
            "invocations", "searches", "illegal", "operational", "exceptions",
            "hard_timeouts", "soft_overruns", "nodes", "exhaustions",
        ):
            key = engine + "_" + suffix
            require(
                _require_uint(parsed["aggregate"][key])
                == sum(_require_uint(bank[key]) for bank in parsed["banks"]),
                "game-work-additive",
            )
        for suffix in ("nodes_max", "depth_max", "attempted_depth_max"):
            key = engine + "_" + suffix
            require(
                _require_uint(parsed["aggregate"][key])
                == max(_require_uint(bank[key]) for bank in parsed["banks"]),
                "game-work-maximum",
            )
        for suffix in ("first_ms_max", "later_ms_max"):
            key = engine + "_" + suffix
            require(
                _require_fixed3(parsed["aggregate"][key])
                == max(_require_fixed3(bank[key]) for bank in parsed["banks"]),
                "game-clock-maximum",
            )
        for proof_name, length in (("rebound", 3), ("root", 3), ("leaf", 3), ("ply1", 4), ("ply2", 4)):
            key = engine + "_proof_" + proof_name
            aggregate_tuple = _parse_tuple(parsed["aggregate"][key], length)
            bank_tuples = [_parse_tuple(bank[key], length) for bank in parsed["banks"]]
            require(
                aggregate_tuple
                == [sum(value[index] for value in bank_tuples) for index in range(length)],
                "game-proof-additive",
            )
        for average_name, maximum_name in (("depth_avg", "depth_max"), ("attempted_depth_avg", "attempted_depth_max")):
            bank_sets = []
            for bank in parsed["banks"]:
                bank_sets.append(set(allowed_round3_totals(
                    bank[engine + "_" + average_name],
                    _require_uint(bank[engine + "_searches"]),
                    _require_uint(bank[engine + "_" + maximum_name]),
                )))
            possible = {0}
            for values in bank_sets:
                possible = {left + right for left in possible for right in values}
            aggregate_set = set(allowed_round3_totals(
                parsed["aggregate"][engine + "_" + average_name],
                _require_uint(parsed["aggregate"][engine + "_searches"]),
                _require_uint(parsed["aggregate"][engine + "_" + maximum_name]),
            ))
            require(not possible.isdisjoint(aggregate_set), "game-average-additive")
    for tuple_key in ("candidate_p0", "candidate_p1"):
        aggregate_tuple = _parse_tuple(parsed["aggregate"][tuple_key], 5)
        bank_tuples = [_parse_tuple(bank[tuple_key], 5) for bank in parsed["banks"]]
        require(
            aggregate_tuple
            == [sum(value[index] for value in bank_tuples) for index in range(5)],
            "game-color-additive",
        )
    configuration = parsed["configuration"]
    expected_fixed = {
        "profile": "clock", "reference_engine": "hybrid-control",
        "expected_role": "development", "max_turns": "320",
        "candidate_nodes": "3000000", "reference_nodes": "3000000",
        "candidate_clock": "800/165", "reference_clock": "800/165",
        "operational_clock": "1000/200", "candidate_exact_proof_mask": "7",
        "reference_exact_proof_mask": "7",
        "openings": (
            "preregistered-generated-public-rules"
            if GENERATED_OVERLAY_ACTIVE
            else "preregistered-public-rules"
        ),
        "replay_corrections": "disabled", "transcripts": "not-retained",
        "bank_validation": "schema,header,role,depth,seed,replay,state-sha256,canonical-sha256,disjoint",
    }
    for key, value in expected_fixed.items():
        require(configuration[key] == value, "game-configuration")
    require(configuration["bank_count"] == str(len(banks_meta)), "game-bank-count")
    require(configuration["expected_depths"] == ",".join(str(item["depth"]) for item in banks_meta), "game-depths")
    require(
        configuration["expected_seeds"]
        == ",".join(str(item["seed"]) for item in banks_meta),
        "game-seeds",
    )
    expected_hashes = (
        "none" if GENERATED_OVERLAY_ACTIVE
        else ",".join(item["sha256"] for item in banks_meta)
    )
    require(configuration["expected_sha256"] == expected_hashes, "game-hashes")
    return aggregate_values


def _validate_table_child(value, engine):
    required = ("capacities", "case_digest_sha256", "cases", "engine", "failure_count", "mode", "passed", "policy", "routing_cases", "routing_digest_sha256", "schema")
    exact_keys(value, required, "table-child-keys")
    candidate = engine == "candidate"
    require(
        _strict_string(value["schema"], "stage0-schema")
        == "rank4-jacek-hybrid-tt-exact-collision-stage0-v1",
        "stage0-schema",
    )
    require(_strict_string(value["mode"], "table-child-mode") == "table-truth", "table-child-mode")
    require(_strict_string(value["engine"], "table-child-mode") == engine, "table-child-mode")
    expected_policy = "depth-primary-exact-secondary" if candidate else "legacy-depth-only"
    require(_strict_string(value["policy"], "table-policy") == expected_policy, "table-policy")
    require(type(value["capacities"]) is list, "table-capacities")
    require(all(type(item) is int for item in value["capacities"]), "table-capacities")
    require(value["capacities"] == [0, 1, 2, 4, 6, 4096, 262144], "table-capacities")
    require(_strict_int(value["cases"], "table-counts", 0) == 16428, "table-counts")
    require(_strict_int(value["routing_cases"], "table-counts", 0) == 252, "table-counts")
    require(_strict_int(value["failure_count"], "table-pass", 0) == 0, "table-pass")
    require(_strict_bool(value["passed"], "table-pass") is True, "table-pass")
    case_digest, routing_digest = table_truth_digests(candidate)
    require(_strict_string(value["case_digest_sha256"], "table-digest") == case_digest, "table-digest")
    require(_strict_string(value["routing_digest_sha256"], "table-digest") == routing_digest, "table-digest")


def _validate_search_result(result, cap, state):
    keys = (
        "legal", "exception", "budget_exhausted", "nodes", "root_proof_shortcut",
        "root_proof_probes", "root_proof_hits", "completed_depth", "attempted_depth",
        "root_score", "action_ascii", "post_state_sha256", "logical_table_sha256",
    )
    exact_keys(result, keys, "activation-result-keys")
    require(_strict_bool(result["legal"], "activation-result") is True, "activation-result")
    require(_strict_bool(result["exception"], "activation-result") is False, "activation-result")
    _strict_bool(result["budget_exhausted"], "activation-budget")
    nodes = _strict_int(result["nodes"], "activation-nodes", 0, cap)
    probes = _strict_int(result["root_proof_probes"], "activation-proof", 0)
    hits = _strict_int(result["root_proof_hits"], "activation-proof", 0)
    require(hits <= probes, "activation-proof")
    completed = _strict_int(result["completed_depth"], "activation-depth", 0)
    attempted = _strict_int(result["attempted_depth"], "activation-depth", 0)
    _strict_int(result["root_score"], "activation-score")
    shortcut = _strict_bool(result["root_proof_shortcut"], "activation-shortcut")
    post_digest = _strict_string(result["post_state_sha256"], "activation-digest")
    table_digest = _strict_string(result["logical_table_sha256"], "activation-digest")
    require(
        HEX64_RE.fullmatch(post_digest) is not None
        and HEX64_RE.fullmatch(table_digest) is not None,
        "activation-digest",
    )
    post = _replay_contest_action(state, result["action_ascii"], "activation-action")
    require(post_digest == _contest_state_hash(post), "activation-post-state")
    if shortcut:
        require(nodes == completed == attempted == 0, "activation-shortcut")
        require(probes > 0 and hits > 0, "activation-shortcut-proof")
    else:
        require(nodes > 0 and attempted > 0, "activation-search")
    return post


def _validate_fixed_depth_result(result, state):
    required = (
        "completed", "exception", "legal", "nodes", "root_score",
        "action_ascii", "action_value", "post_state_sha256", "state_unchanged",
    )
    exact_keys(result, required, "oracle-result-keys")
    require(_strict_bool(result["completed"], "oracle-result") is True, "oracle-result")
    require(_strict_bool(result["exception"], "oracle-result") is False, "oracle-result")
    require(_strict_bool(result["legal"], "oracle-result") is True, "oracle-result")
    require(_strict_bool(result["state_unchanged"], "oracle-result") is True, "oracle-result")
    _strict_int(result["nodes"], "oracle-nodes", 1, 2000000)
    score = _strict_int(result["root_score"], "oracle-score")
    require(_strict_int(result["action_value"], "oracle-value") == score, "oracle-value")
    post_digest = _strict_string(result["post_state_sha256"], "oracle-post-state")
    require(HEX64_RE.fullmatch(post_digest) is not None, "oracle-post-state")
    post = _replay_contest_action(state, result["action_ascii"], "oracle-action")
    require(post_digest == _contest_state_hash(post), "oracle-post-state")
    return post


def _validate_public_child(value, engine, plan, corpus):
    keys = plan["stage0"]["machine_contract"]["public_screen"]["keys_exact"]
    exact_keys(value, keys, "public-child-keys")
    require(
        _strict_string(value["schema"], "stage0-schema")
        == "rank4-jacek-hybrid-tt-exact-collision-stage0-v1",
        "stage0-schema",
    )
    require(_strict_string(value["mode"], "public-child-mode") == "public-screen", "public-child-mode")
    require(_strict_string(value["engine"], "public-child-mode") == engine, "public-child-mode")
    expected_policy = "depth-primary-exact-secondary" if engine == "candidate" else "legacy-depth-only"
    require(_strict_string(value["policy"], "public-policy") == expected_policy, "public-policy")
    require(_strict_string(value["seed_hex"], "public-seed") == plan["stage0"]["corpus"]["seed_hex"], "public-seed")
    for name, expected in (
        ("procedural_states", 1000), ("activation_states", 64),
        ("activation_nodes", 50000), ("depth1_states", 64),
        ("depth2_states", 24), ("oracle_max_nodes", 2000000),
        ("exact_proof_mask", 7),
    ):
        require(_strict_int(value[name], "public-config", 0) == expected, "public-config")
    emitted_corpus = _strict_string(value["corpus_sha256"], "public-corpus-digest")
    require(HEX64_RE.fullmatch(emitted_corpus) is not None, "public-corpus-digest")
    require(emitted_corpus == corpus["corpus_sha256"], "public-corpus-digest")
    require(type(value["activation_records"]) is list, "public-record-count")
    require(type(value["oracle_records"]) is list, "public-record-count")
    require(len(value["activation_records"]) == 64 and len(value["oracle_records"]) == 88, "public-record-count")
    for index, record in enumerate(value["activation_records"]):
        exact_keys(record, ("state_index", "base_state_sha256", "rotated_state_sha256", "base", "rotated"), "activation-record-keys")
        require(_strict_int(record["state_index"], "activation-index", 0) == index, "activation-index")
        base, rotated, base_hash, rotated_hash = corpus["pairs"][index]
        require(_strict_string(record["base_state_sha256"], "activation-state-digest") == base_hash, "activation-state-digest")
        require(_strict_string(record["rotated_state_sha256"], "activation-state-digest") == rotated_hash, "activation-state-digest")
        _validate_search_result(record["base"], 50000, base)
        _validate_search_result(record["rotated"], 50000, rotated)
    rotation_mismatches = set()
    for index, record in enumerate(value["oracle_records"]):
        exact_keys(record, ("depth", "state_index", "base_state_sha256", "rotated_state_sha256", "base", "rotated"), "oracle-record-keys")
        expected_depth = 1 if index < 64 else 2
        expected_index = index if index < 64 else index - 64
        require(_strict_int(record["depth"], "oracle-index", 0) == expected_depth, "oracle-index")
        require(_strict_int(record["state_index"], "oracle-index", 0) == expected_index, "oracle-index")
        base_state, rotated_state, base_hash, rotated_hash = corpus["pairs"][expected_index]
        require(_strict_string(record["base_state_sha256"], "oracle-state-digest") == base_hash, "oracle-state-digest")
        require(_strict_string(record["rotated_state_sha256"], "oracle-state-digest") == rotated_hash, "oracle-state-digest")
        for side in ("base", "rotated"):
            exact_keys(record[side], ("tt", "no_tt"), "oracle-side-keys")
            tt = record[side]["tt"]
            no_tt = record[side]["no_tt"]
            side_state = base_state if side == "base" else rotated_state
            _validate_fixed_depth_result(tt, side_state)
            _validate_fixed_depth_result(no_tt, side_state)
            require(tt["root_score"] == no_tt["root_score"], "oracle-score")
        base_no_tt = record["base"]["no_tt"]
        rotated_no_tt = record["rotated"]["no_tt"]
        require(base_no_tt["root_score"] == rotated_no_tt["root_score"], "oracle-no-tt-rotation-score")
        require(rotated_no_tt["action_ascii"] == _rotate_action(base_no_tt["action_ascii"]), "oracle-no-tt-rotation-action")
        base_tt = record["base"]["tt"]
        rotated_tt = record["rotated"]["tt"]
        require(base_tt["root_score"] == rotated_tt["root_score"], "oracle-tt-rotation-score")
        if rotated_tt["action_ascii"] != _rotate_action(base_tt["action_ascii"]):
            rotation_mismatches.add((expected_depth, expected_index))
    return rotation_mismatches


def _validate_timing_child(value, command, plan, corpus):
    keys = plan["stage0"]["machine_contract"]["timing"]["keys_exact"]
    exact_keys(value, keys, "timing-child-keys")
    require(
        _strict_string(value["schema"], "timing-mode")
        == "rank4-jacek-hybrid-tt-exact-collision-stage0-v1",
        "timing-mode",
    )
    require(_strict_string(value["mode"], "timing-mode") == "timing", "timing-mode")
    def argument(name):
        index = command.index(name)
        return command[index + 1]
    expected_engine = "candidate" if "candidate" in command[0] else "control"
    require(_strict_string(value["engine"], "timing-engine") == expected_engine, "timing-engine")
    expected_policy = "depth-primary-exact-secondary" if expected_engine == "candidate" else "legacy-depth-only"
    require(_strict_string(value["policy"], "timing-policy") == expected_policy, "timing-policy")
    panel = _strict_string(value["panel"], "timing-panel")
    phase = _strict_string(value["phase"], "timing-panel")
    require(panel == argument("--panel") and phase == argument("--phase"), "timing-panel")
    pair_index = _strict_int(value["pair_index"], "timing-pair", 0)
    order_position = _strict_int(value["order_position"], "timing-pair", 0, 1)
    state_index = _strict_int(value["state_index"], "timing-config", 0)
    require(pair_index == int(argument("--pair-index")) and order_position == int(argument("--order-position")), "timing-pair")
    require(state_index == int(argument("--state-index")), "timing-config")
    require(_strict_int(value["proof_mask"], "timing-config", 0) == 7, "timing-config")
    require(_strict_int(value["repetitions"], "timing-config", 0) == 8, "timing-config")
    _strict_int(value["elapsed_ns"], "timing-elapsed", 1)
    require(type(value["results"]) is list and len(value["results"]) == 8, "timing-results")
    state, state_hash = _timing_panel_state(corpus, panel, state_index)
    result_keys = (
        "action_ascii", "attempted_depth", "budget_exhausted", "completed_depth",
        "exception", "illegal", "nodes", "post_state_sha256", "root_proof_hits",
        "root_proof_probes", "root_proof_shortcut", "root_score",
    )
    for result in value["results"]:
        exact_keys(result, result_keys, "timing-result-keys")
        require(_strict_bool(result["exception"], "timing-result-failure") is False, "timing-result-failure")
        require(_strict_bool(result["illegal"], "timing-result-failure") is False, "timing-result-failure")
        _strict_bool(result["budget_exhausted"], "timing-result-budget")
        nodes = _strict_int(result["nodes"], "timing-result-nodes", 0, 50000)
        completed = _strict_int(result["completed_depth"], "timing-result-depth", 0)
        attempted = _strict_int(result["attempted_depth"], "timing-result-depth", 0)
        probes = _strict_int(result["root_proof_probes"], "timing-result-proof", 0)
        hits = _strict_int(result["root_proof_hits"], "timing-result-proof", 0)
        require(hits <= probes, "timing-result-proof")
        _strict_int(result["root_score"], "timing-result-score")
        shortcut = _strict_bool(result["root_proof_shortcut"], "timing-result-shortcut")
        post_digest = _strict_string(result["post_state_sha256"], "timing-result-state")
        require(HEX64_RE.fullmatch(post_digest) is not None, "timing-result-state")
        post = _replay_contest_action(state, result["action_ascii"], "timing-result-action")
        require(post_digest == _contest_state_hash(post), "timing-result-state")
        if shortcut:
            require(nodes == completed == attempted == 0, "timing-result-shortcut")
            require(probes > 0 and hits > 0, "timing-result-shortcut-proof")
        else:
            require(nodes > 0 and attempted > 0, "timing-result-search")
    require(all(item == value["results"][0] for item in value["results"]), "timing-results")
    input_object = {
        "evaluation_entries": 131072,
        "max_nodes": 50000,
        "pair_index": pair_index,
        "panel": panel,
        "phase": phase,
        "proof_mask": 7,
        "repetitions": 8,
        "state_index": state_index,
        "state_sha256": state_hash,
        "transposition_entries": 262144,
    }
    input_digest = _strict_string(value["input_signature_sha256"], "timing-input-digest")
    result_digest = _strict_string(value["result_signature_sha256"], "timing-result-digest")
    require(HEX64_RE.fullmatch(input_digest) is not None, "timing-input-digest")
    require(HEX64_RE.fullmatch(result_digest) is not None, "timing-result-digest")
    require(input_digest == sha256_bytes(canonical_json(input_object)), "timing-input-preimage")
    require(result_digest == sha256_bytes(canonical_json(value["results"])), "timing-result-digest")
    return value


def parse_stage0_outputs(raw_outputs, plan):
    schedule, _timeouts = build_stage0_schedule(plan)
    require(type(raw_outputs) is list, "stage0-output-count")
    require(len(raw_outputs) == len(schedule), "stage0-output-count")
    values = [_parse_child_json(raw) for raw in raw_outputs]
    corpus = _build_stage0_corpus(plan)
    _validate_table_child(values[0], "control")
    _validate_table_child(values[1], "candidate")
    control_rotation_mismatches = _validate_public_child(
        values[2], "control", plan, corpus,
    )
    candidate_rotation_mismatches = _validate_public_child(
        values[3], "candidate", plan, corpus,
    )
    require(values[2]["corpus_sha256"] == values[3]["corpus_sha256"], "public-corpus")
    require(values[2]["corpus_sha256"] == corpus["corpus_sha256"], "public-corpus-digest")
    require(
        candidate_rotation_mismatches.issubset(control_rotation_mismatches),
        "oracle-candidate-rotation-mismatch",
    )
    require(
        len(candidate_rotation_mismatches) <= len(control_rotation_mismatches),
        "oracle-candidate-rotation-count",
    )
    natural = 0
    for control, candidate in zip(values[2]["activation_records"], values[3]["activation_records"]):
        require(control["state_index"] == candidate["state_index"] and control["base_state_sha256"] == candidate["base_state_sha256"] and control["rotated_state_sha256"] == candidate["rotated_state_sha256"], "activation-alignment")
        for side in ("base", "rotated"):
            if control[side]["logical_table_sha256"] != candidate[side]["logical_table_sha256"]:
                natural += 1
    for control, candidate in zip(values[2]["oracle_records"], values[3]["oracle_records"]):
        require(
            control["depth"] == candidate["depth"]
            and control["state_index"] == candidate["state_index"]
            and control["base_state_sha256"] == candidate["base_state_sha256"]
            and control["rotated_state_sha256"] == candidate["rotated_state_sha256"],
            "oracle-alignment",
        )
        for side in ("base", "rotated"):
            require(
                control[side]["no_tt"] == candidate[side]["no_tt"],
                "oracle-no-tt-alignment",
            )
    timing_values = [
        _validate_timing_child(value, command, plan, corpus)
        for value, command in zip(values[4:], schedule[4:])
    ]
    require(len(timing_values) % 2 == 0, "timing-pair-count")
    for index in range(0, len(timing_values), 2):
        first = timing_values[index]
        second = timing_values[index + 1]
        require(
            first["panel"] == second["panel"]
            and first["phase"] == second["phase"]
            and first["pair_index"] == second["pair_index"]
            and first["state_index"] == second["state_index"]
            and first["input_signature_sha256"] == second["input_signature_sha256"],
            "timing-pair-input",
        )
    timing_panels = []
    cursor = 0
    for panel in plan["stage0"]["timing"]["panels"]:
        cursor += 30 * 2
        measured = timing_values[cursor:cursor + 300 * 2]
        cursor += 300 * 2
        candidates = [None] * 300
        controls = [None] * 300
        for value in measured:
            target = candidates if value["engine"] == "candidate" else controls
            require(target[value["pair_index"]] is None, "timing-duplicate")
            target[value["pair_index"]] = value["elapsed_ns"]
        require(all(type(item) is int and item > 0 for item in candidates + controls), "timing-complete")
        ratios = list(range(300))
        def compare(left, right):
            cross_left = candidates[left] * controls[right]
            cross_right = candidates[right] * controls[left]
            if cross_left != cross_right:
                return -1 if cross_left < cross_right else 1
            return left - right
        ratios.sort(key=functools.cmp_to_key(compare))
        low = ratios[149]
        high = ratios[150]
        p99 = ratios[296]
        median_numerator = candidates[low] * controls[high] + candidates[high] * controls[low]
        median_denominator = 2 * controls[low] * controls[high]
        timing_panels.append({
            "panel": panel, "candidate_elapsed_ns": candidates, "control_elapsed_ns": controls,
            "candidate_array_sha256": sha256_bytes(canonical_json(candidates)),
            "control_array_sha256": sha256_bytes(canonical_json(controls)),
            "median_source_pair_indices": [low, high],
            "median_ratio_numerator": median_numerator,
            "median_ratio_denominator": median_denominator,
            "p99_source_pair_index": p99,
            "p99_ratio_numerator": candidates[p99],
            "p99_ratio_denominator": controls[p99],
        })
    require(cursor == len(timing_values), "timing-cursor")
    return {
        "schema": CAMPAIGN_SCHEMA_PREFIX + "stage0-parsed-v1",
        "child_outputs": values,
        "natural_activation_count": natural,
        "timing_panels": timing_panels,
    }


def stage0_threshold_passed(parsed, plan):
    require(parsed["natural_activation_count"] >= plan["stage0"]["activation"]["minimum_natural_activations"], "natural-activation")
    for panel in parsed["timing_panels"]:
        require(
            panel["median_ratio_numerator"] * 1000
            <= panel["median_ratio_denominator"] * 1005,
            "timing-median",
        )
        require(
            panel["p99_ratio_numerator"] * 100
            <= panel["p99_ratio_denominator"] * 101,
            "timing-p99",
        )
    return True


def load_prior_binding(plan):
    record = plan["prior_evidence"]["heldout_binding_host_portability_baseline"]
    require(
        not record["path"].endswith(".tsv")
        and "openings" not in record["path"].split("/"),
        "binding-bank-path",
    )
    path = root_path(record["path"], allow_protected_literal=True)
    value, identity, _raw = load_canonical_file(path, record["mode"], record["sha256"], record["bytes"])
    require(identity["bytes"] == record["bytes"], "binding-bytes")
    require(value.get("bank_files_accessed") == [], "binding-bank-access")
    return value


def _identity_projection(record):
    return {
        "path": record["path"], "bytes": record["bytes"],
        "mode": record["mode"], "sha256": record["sha256"],
    }


def _external_provenance_only_paths(plan, external):
    require(isinstance(external, dict), "external-nlink-baseline-map")
    compiler_plan = plan["commit_governance"]["implementation_commit"][
        "preexecution_no_protected_bank_verification"
    ]
    policy = compiler_plan["external_dependency_nlink_policy"]
    exact_keys(
        policy,
        (
            "default_nlink", "exception_count", "exceptions",
            "live_validation", "schema", "scope",
        ),
        "external-nlink-policy-keys",
    )
    require(
        policy["schema"]
        == CAMPAIGN_SCHEMA_PREFIX + "external-nlink-policy-v1",
        "external-nlink-policy-schema",
    )
    require(
        type(policy["default_nlink"]) is int
        and policy["default_nlink"] == 1
        and type(policy["exception_count"]) is int
        and policy["exception_count"] == 4
        and policy["live_validation"] is False
        and isinstance(policy["scope"], str)
        and policy["scope"] != "",
        "external-nlink-policy-shape",
    )
    exceptions = policy["exceptions"]
    require(
        isinstance(exceptions, list)
        and len(exceptions) == policy["exception_count"],
        "external-nlink-policy-count",
    )
    paths = []
    for record in exceptions:
        exact_keys(
            record, ("path", "bytes", "mode", "nlink", "sha256"),
            "external-nlink-exception-keys",
        )
        path = canonical_absolute_path(record["path"], allow_runtime=True)
        require(
            type(record["bytes"]) is int and record["bytes"] > 0
            and isinstance(record["mode"], str)
            and re.fullmatch(r"[0-7]{4}", record["mode"]) is not None
            and type(record["nlink"]) is int and record["nlink"] > 1
            and isinstance(record["sha256"], str)
            and HEX64_RE.fullmatch(record["sha256"]) is not None,
            "external-nlink-exception-shape",
        )
        require(path not in paths, "external-nlink-exception-duplicate")
        baseline = external.get(path)
        require(
            baseline is not None
            and baseline == _identity_projection(record),
            "external-nlink-exception-baseline",
        )
        paths.append(path)
    require(
        paths == sorted(paths, key=lambda item: item.encode("utf-8")),
        "external-nlink-exception-order",
    )
    return frozenset(paths)


def _require_external_provenance_disjoint(plan, provenance_only):
    compiler_plan = plan["commit_governance"]["implementation_commit"][
        "preexecution_no_protected_bank_verification"
    ]
    actual_paths = []
    commands = compiler_plan["commands"]
    require(isinstance(commands, list), "external-nlink-command-list")
    for command in commands:
        require(
            isinstance(command, list) and command
            and isinstance(command[0], str) and command[0] != "",
            "external-nlink-command",
        )
        actual_paths.append(command[0])
    for key in (
        "implicit_driver_input_identities",
        "nested_program_identities",
        "nested_runtime_identities",
        "link_input_identities",
    ):
        records = compiler_plan[key]
        require(isinstance(records, list), "external-nlink-input-list")
        for record in records:
            require(
                isinstance(record, dict) and isinstance(record.get("path"), str),
                "external-nlink-input-record",
            )
            actual_paths.append(record["path"])
            resolved = record.get("resolved")
            if resolved is not None:
                require(
                    isinstance(resolved, dict)
                    and isinstance(resolved.get("path"), str),
                    "external-nlink-resolved-record",
                )
                actual_paths.append(resolved["path"])
    identity_records = []
    _walk_identity_records(plan["frozen_inputs"]["toolchain_executables"], identity_records)
    _walk_identity_records(plan["frozen_inputs"]["runtime_tools"], identity_records)
    _walk_identity_records(
        plan["execution_policy"]["outer_cli"]["bootstrap_env_identity"],
        identity_records,
    )
    actual_paths.extend(record["path"] for record in identity_records)
    require(
        provenance_only.isdisjoint(actual_paths),
        "external-nlink-provenance-used",
    )


def source_and_external_closures(plan, head, binding, validate_bytes=True):
    dependencies = binding["dependency_identities"]
    repository = {}
    external = {}
    for path, record in dependencies.items():
        require(record["path"] == path, "binding-dependency-key")
        if path.startswith("/"):
            require(
                not path.endswith(".tsv")
                and "openings" not in path.split("/"),
                "binding-protected-dependency",
            )
            external[path] = _identity_projection(record)
        elif not path.startswith("build/") and not path.startswith("results/"):
            require(
                not path.endswith(".tsv")
                and "openings" not in path.split("/"),
                "binding-protected-dependency",
            )
            canonical_relative_path(path, allow_protected_literal=True)
            repository[path] = _identity_projection(record)
    require(len(repository) == 72, "repository-baseline-count")
    repository_raw = sorted(repository.values(), key=lambda item: item["path"].encode("utf-8"))
    require(sha256_bytes(canonical_json(repository_raw)) == "eb26426fd615bf55d5ed1ba737cce22005657a042d20a1fecfe6f71feddaa217", "repository-baseline-digest")
    require(len(external) == 1037, "external-baseline-count")
    external_raw = sorted(external.values(), key=lambda item: item["path"].encode("utf-8"))
    require(sha256_bytes(canonical_json(external_raw)) == "20728feb3caaeb695be1a65fd98cab6d6768d6cf614eca2946440bbbb5f16cc4", "external-baseline-digest")
    provenance_only = _external_provenance_only_paths(plan, external)
    _require_external_provenance_disjoint(plan, provenance_only)
    closure = plan["commit_governance"]["implementation_commit"]["path_closure"]
    for path in closure["modified"] + closure["new"]:
        identity, raw = file_identity(root_path(path), path, "0644", max_bytes=16 * 1024 * 1024)
        require(git_query(["show", head + ":" + path], stdout_cap=16 * 1024 * 1024) == raw, "source-head")
        repository[path] = identity
    require(len(repository) == 78, "repository-closure-count")
    for record in plan["commit_governance"]["implementation_commit"]["preexecution_no_protected_bank_verification"]["external_dependency_supplement"]["records"]:
        external[record["path"]] = dict(record)
    require(len(external) == 1044, "external-closure-count")
    require(
        len(external) - len(provenance_only) == 1040,
        "external-live-closure-count",
    )
    combined = sorted(external.values(), key=lambda item: item["path"].encode("utf-8"))
    require(sha256_bytes(canonical_json(combined)) == "e187d430edcedd90a48624d1e99dc683f3e4ecbf201af3bcb1fba13d4ef9afd0", "external-closure-digest")
    if validate_bytes:
        for path, expected in repository.items():
            observed, _raw = file_identity(root_path(path), path, expected["mode"], expected["sha256"], expected["bytes"])
            require(observed == expected, "repository-input-drift")
        for path, expected in external.items():
            if path in provenance_only:
                continue
            canonical_absolute_path(path, allow_runtime=True)
            observed, _raw = file_identity(path, path, expected["mode"], expected["sha256"], expected["bytes"])
            require(observed == expected, "external-input-drift")
    return repository, external


def _revalidate_regular_identity(record, reason):
    path = record["path"]
    absolute = path if path.startswith("/") else root_path(path)
    observed, _raw = file_identity(
        absolute, path, record["mode"], record["sha256"], record["bytes"]
    )
    require(observed == _identity_projection(record), reason)


def _revalidate_link_input(record):
    if record.get("type") == "regular":
        _revalidate_regular_identity(record, "compiler-link-input-drift")
        return
    require(record.get("type") == "symlink", "compiler-link-input-type")
    parent_fd, parent_path, leaf = _open_parent_absolute(record["path"])
    try:
        before = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        require(
            stat.S_ISLNK(before.st_mode)
            and mode_string(before) == record["mode"]
            and before.st_nlink == record["nlink"]
            and before.st_size == record["bytes"],
            "compiler-link-alias-metadata",
        )
        require(
            os.readlink(leaf, dir_fd=parent_fd) == record["link_text"],
            "compiler-link-alias-text",
        )
        after = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        require(
            _metadata_identity(before) == _metadata_identity(after),
            "compiler-link-alias-drift",
        )
        _verify_absolute_directory_binding(parent_path, parent_fd)
    finally:
        os.close(parent_fd)
    resolved = record["resolved"]
    _revalidate_regular_identity(resolved, "compiler-link-terminal-drift")


def revalidate_preexecution_inputs(plan, repository, external):
    # These maps are the preregistered, already-Git-bound compiler input
    # closure.  Reopening every member around each scan/link closes drift
    # without rerunning an unbounded repository query.
    provenance_only = _external_provenance_only_paths(plan, external)
    _require_external_provenance_disjoint(plan, provenance_only)
    for record in repository.values():
        _revalidate_regular_identity(record, "repository-input-drift")
    for record in external.values():
        if record["path"] in provenance_only:
            continue
        _revalidate_regular_identity(record, "external-input-drift")

    compiler_plan = plan["commit_governance"]["implementation_commit"][
        "preexecution_no_protected_bank_verification"
    ]
    for group in (
        compiler_plan["implicit_driver_input_identities"],
        compiler_plan["nested_program_identities"],
        compiler_plan["nested_runtime_identities"],
    ):
        for record in group:
            _revalidate_regular_identity(record, "compiler-bootstrap-drift")
    for record in compiler_plan["link_input_identities"]:
        _revalidate_link_input(record)

    tool_records = []
    _walk_identity_records(plan["frozen_inputs"]["toolchain_executables"], tool_records)
    for record in tool_records:
        _revalidate_regular_identity(record, "toolchain-executable-drift")


def _dependency_aliases():
    sdk = "/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk/usr/include/"
    return {
        sdk + "pthread.h": sdk + "pthread/pthread.h",
        sdk + "sched.h": sdk + "pthread/sched.h",
    }


def normalized_dependency_paths(raw, target):
    aliases = _dependency_aliases()
    normalized = set()
    for token in parse_make_dependencies(raw, target):
        value = normalized_dependency_path(token)
        normalized.add(aliases.get(value, value))
    result = sorted(normalized, key=lambda item: item.encode("utf-8"))
    require(result, "dependency-empty")
    return result


def _captured_external_dependency(path):
    require(
        path.startswith(
            "/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk/"
        )
        or path.startswith(
            "/Library/Developer/CommandLineTools/usr/lib/clang/21/"
        ),
        "dependency-unregistered",
    )
    require(
        not path.endswith(".tsv")
        and "openings" not in path.split("/"),
        "dependency-protected-path",
    )
    observed, _raw = file_identity(
        path, path, max_bytes=16 * 1024 * 1024
    )
    require(observed["mode"] in ("0444", "0644"), "dependency-mode")
    return observed


def _revalidate_captured_dependencies(records, repository, external):
    for record in records:
        if record["path"] not in repository and record["path"] not in external:
            require(
                record == _captured_external_dependency(record["path"]),
                "dependency-captured-drift",
            )


def dependency_records(plan, raw, target, literal_sources, repository, external):
    provenance_only = _external_provenance_only_paths(plan, external)
    normalized = normalized_dependency_paths(raw, target)
    for source in literal_sources:
        require(source in normalized, "dependency-source-missing")
    records = []
    for path in sorted(normalized, key=lambda item: item.encode("utf-8")):
        require(path not in provenance_only, "dependency-provenance-only")
        expected = external.get(path) if path.startswith("/") else repository.get(path)
        if expected is None:
            require(path.startswith("/"), "dependency-unregistered")
            records.append(_captured_external_dependency(path))
            continue
        absolute = path if path.startswith("/") else root_path(path)
        observed, _raw = file_identity(absolute, path, expected["mode"], expected["sha256"], expected["bytes"])
        require(observed == expected, "dependency-drift")
        records.append(expected)
    require(records, "dependency-empty")
    return records


def translation_unit_configurations(plan):
    commands = plan["commit_governance"]["implementation_commit"]["preexecution_no_protected_bank_verification"]["commands"]
    roles = (("game", 1, 2), ("stage0-control", 3, 4), ("stage0-candidate", 5, 6))
    configurations = []
    for role, scan_index, compile_index in roles:
        scan = commands[scan_index]
        compile_command = commands[compile_index]
        compiler = scan[0]
        require(compiler == compile_command[0], "translation-compiler")
        source_start = next(index for index, token in enumerate(scan) if token == "-M")
        mt_index = scan.index("-MT")
        sources = scan[mt_index + 2 :]
        common_and_macros = scan[1:source_start]
        role_macro_start = next((index for index, token in enumerate(common_and_macros) if token.startswith("-DPAPERSOCCER_TT_EXACT_COLLISION") or token.startswith("-DPAPER_SOCCER_TURN_ACTION")), len(common_and_macros))
        include_start = common_and_macros.index("-Iinclude")
        common_prefix = common_and_macros[:role_macro_start]
        role_macros = common_and_macros[role_macro_start:include_start]
        common_suffix = common_and_macros[include_start:]
        output_index = compile_command.index("-o")
        compile_sources_start = compile_command.index(sources[0])
        require(compile_command[compile_sources_start:compile_sources_start + len(sources)] == sources, "translation-sources")
        configurations.append({
            "role": role, "compiler": compiler, "common_prefix": common_prefix,
            "role_macros": role_macros, "common_suffix": common_suffix,
            "sources": sources, "dependency_scan_only": ["-M", "-MT", scan[mt_index + 1]],
            "compile_link_only": compile_command[compile_sources_start + len(sources):],
            "output": compile_command[output_index + 1],
        })
    return configurations


def configuration_digests(plan):
    stage0_object = {
        "schema": CAMPAIGN_SCHEMA_PREFIX + "stage0-configuration-v1",
        "stage0": plan["stage0"],
        "aggregate_timeout_seconds": _stage_aggregate_seconds(plan),
    }
    game_object = {
        "schema": CAMPAIGN_SCHEMA_PREFIX + "game-configuration-v1",
        "stages": plan["stages"], "summary_validation": plan["summary_validation"],
    }
    if GENERATED_OVERLAY_ACTIVE:
        bank_records = [dict(record) for record in plan["development_banks"]]
    else:
        bank_records = [
            {
                key: record[key]
                for key in (
                    "id", "path", "depth", "role", "seed", "bytes",
                    "mode", "sha256", "pairs", "expected_games",
                )
            }
            for record in plan["development_banks"]
        ]
    bank_object = {
        "schema": CAMPAIGN_SCHEMA_PREFIX + "bank-plan-metadata-v1",
        "banks": bank_records,
    }
    base = {
        "stage0_sha256": sha256_bytes(canonical_json(stage0_object)),
        "game_stages_sha256": sha256_bytes(canonical_json(game_object)),
        "development_bank_plan_metadata_sha256": sha256_bytes(canonical_json(bank_object)),
    }
    stages = {STAGE_ORDER[0]: base["stage0_sha256"]}
    for stage, bank_ids in ((STAGE_ORDER[1], ["d20"]), (STAGE_ORDER[2], ["d04", "d08", "d12"])):
        stages[stage] = sha256_bytes(canonical_json({
            "schema": CAMPAIGN_SCHEMA_PREFIX + "stage-configuration-v1",
            "stage": stage, "game_stages_sha256": base["game_stages_sha256"],
            "development_bank_plan_metadata_sha256": base["development_bank_plan_metadata_sha256"],
            "bank_ids": bank_ids,
        }))
    return base, stages


def validate_source_projection(plan):
    projection = plan["candidate_projection"]
    base_source = projection["base"]["maintained_source"]
    base_upload = projection["base"]["production_upload"]
    source_identity, source_raw = file_identity(root_path(base_source["path"]), base_source["path"], base_source["mode"], base_source["sha256"], base_source["bytes"])
    upload_identity, upload_raw = file_identity(root_path(base_upload["path"]), base_upload["path"], base_upload["mode"], base_upload["sha256"], base_upload["bytes"])
    require(source_identity["bytes"] == base_source["bytes"] and upload_identity["bytes"] == base_upload["bytes"], "projection-base")
    transform = projection["exact_transform"]
    source_rule = transform["maintained_source"]
    upload_rule = transform["generated_upload"]
    old_source = source_rule["old_text"].encode("ascii")
    new_source = source_rule["new_text"].encode("ascii")
    old_upload = upload_rule["old_text"].encode("ascii")
    new_upload = upload_rule["new_text"].encode("ascii")
    require(source_raw.count(old_source) == 1 and source_raw.count(new_source) == 0, "projection-source-occurrence")
    require(upload_raw.count(old_upload) == 1 and upload_raw.count(new_upload) == 0, "projection-upload-occurrence")
    candidate_source = source_raw.replace(old_source, new_source, 1)
    candidate_upload = upload_raw.replace(old_upload, new_upload, 1)
    expected_source = projection["candidate"]["maintained_source"]
    expected_upload = projection["candidate"]["production_upload"]
    require(len(candidate_source) == expected_source["bytes"] and sha256_bytes(candidate_source) == expected_source["sha256"], "projection-source")
    require(len(candidate_upload) == expected_upload["bytes"] and sha256_bytes(candidate_upload) == expected_upload["sha256"], "projection-upload")
    require(candidate_source.isascii() and candidate_upload.isascii() and len(candidate_upload) <= expected_upload["maximum_bytes"], "projection-ascii")
    implementation_path = expected_source["implementation_path"]
    implementation_identity, implementation_raw = file_identity(root_path(implementation_path), implementation_path, expected_source["expected_mode"], expected_source["sha256"], expected_source["bytes"])
    require(implementation_raw == candidate_source, "projection-implementation")
    return {
        "base_bot": {"bytes": len(source_raw), "sha256": sha256_bytes(source_raw)},
        "base_upload": {"bytes": len(upload_raw), "sha256": sha256_bytes(upload_raw)},
        "candidate_bot": {"bytes": len(candidate_source), "sha256": sha256_bytes(candidate_source)},
        "candidate_upload": {"bytes": len(candidate_upload), "sha256": sha256_bytes(candidate_upload)},
        "transform_bundle_sha256": transform["literal_pair_bundle_sha256"],
    }


def sources_projection_from_plan(plan):
    projection = plan["candidate_projection"]
    base_source = projection["base"]["maintained_source"]
    base_upload = projection["base"]["production_upload"]
    candidate_source = projection["candidate"]["maintained_source"]
    candidate_upload = projection["candidate"]["production_upload"]
    return {
        "base_bot": {"bytes": base_source["bytes"], "sha256": base_source["sha256"]},
        "base_upload": {"bytes": base_upload["bytes"], "sha256": base_upload["sha256"]},
        "candidate_bot": {"bytes": candidate_source["bytes"], "sha256": candidate_source["sha256"]},
        "candidate_upload": {"bytes": candidate_upload["bytes"], "sha256": candidate_upload["sha256"]},
        "transform_bundle_sha256": projection["exact_transform"]["literal_pair_bundle_sha256"],
    }


def create_fresh_build_root():
    parent = root_path(BUILD_PARENT_REL)
    root = root_path(BUILD_ROOT_REL)
    require(
        not _path_exists_nofollow(parent) and not _path_exists_nofollow(root),
        "build-root-not-fresh",
    )
    mkdir_new_exact(parent, "0755")
    mkdir_new_exact(root, "0755")
    mkdir_new_exact(root_path(TMP_REL), "0700")
    tmp_fd = _open_absolute_directory(root_path(TMP_REL), "0700")
    try:
        require(_list_directory_fd(tmp_fd, root_path(TMP_REL)) == [], "tmp-not-empty")
    finally:
        os.close(tmp_fd)


def validate_build_root(binary_records=None):
    root = root_path(BUILD_ROOT_REL)
    root_fd = _open_absolute_directory(root, "0755")
    try:
        tmp = root_path(TMP_REL)
        tmp_fd = _open_directory_component(root_fd, "tmp", "0700")
        try:
            require(_list_directory_fd(tmp_fd, tmp) == [], "tmp-not-empty")
        finally:
            os.close(tmp_fd)
        expected = {"tmp"}
        if binary_records:
            expected.update(
                os.path.basename(record["path"])
                for record in binary_records.values()
            )
        require(
            set(_list_directory_fd(root_fd, root)) == expected,
            "build-root-entries",
        )
    finally:
        os.close(root_fd)


def _process_error(error, checked_utc=None):
    reason = sanitized_reason(error)
    result = error.result if isinstance(error, ProcessSnapshotError) else None
    if result is not None:
        if result["os_error"] is not None:
            error_type = "spawn-error"
        elif result["timed_out"]:
            error_type = "timeout"
        elif (
            result["stdout_receipt"].get("truncated") is True
            or result["stderr_receipt"].get("truncated") is True
        ):
            error_type = "stream-cap"
        elif result["returncode"] != 0 or result["stderr"] != b"":
            error_type = "nonzero"
        elif "decode" in reason:
            error_type = "decode-error"
        else:
            error_type = "parse-error"
        return {
            "checked_utc": checked_utc or error.checked_utc or result["ended_utc"],
            "type": error_type,
            "message_sha256": sha256_bytes(reason.encode("ascii")),
            "returncode": result["returncode"],
            "timed_out": result["timed_out"],
            "stdout_receipt": result["stdout_receipt"],
            "stderr_receipt": result["stderr_receipt"],
        }
    return {
        "checked_utc": checked_utc or utc_now(), "type": "parse-error",
        "message_sha256": sha256_bytes(reason.encode("ascii")), "returncode": None,
        "timed_out": False, "stdout_receipt": stream_receipt(b"", 4 * 1024 * 1024),
        "stderr_receipt": stream_receipt(b"", 1024 * 1024),
    }


def _child_record(index, claimed_argv, resolved_path, result, post_process, post_error):
    return {
        "index": index, "argv": claimed_argv, "environment_override": {},
        "resolved_executable_path": resolved_path,
        "started_utc": result["started_utc"], "ended_utc": result["ended_utc"],
        "elapsed_monotonic_seconds": result["elapsed_monotonic_seconds"],
        "returncode": result["returncode"], "timed_out": result["timed_out"],
        "os_error": result["os_error"], "stdout_receipt": result["stdout_receipt"],
        "stderr_receipt": result["stderr_receipt"], "postchild_process": post_process,
        "postchild_process_error": post_error,
    }


def _stage_child_record(index, claimed_argv, resolved_path, result, post_process, post_error):
    return {
        "argv_index": index, "argv_sha256": sha256_bytes(canonical_json(claimed_argv)),
        "resolved_executable_path": resolved_path,
        "started_utc": result["started_utc"], "ended_utc": result["ended_utc"],
        "elapsed_monotonic_seconds": result["elapsed_monotonic_seconds"],
        "returncode": result["returncode"], "timed_out": result["timed_out"],
        "os_error": result["os_error"], "stdout_receipt": result["stdout_receipt"],
        "stderr_receipt": result["stderr_receipt"], "postchild_process": post_process,
        "postchild_process_error": post_error,
    }


def _child_succeeded(result):
    return (
        result["returncode"] == 0 and result["os_error"] is None and not result["timed_out"]
        and not result["stdout_receipt"]["truncated"] and not result["stderr_receipt"]["truncated"]
    )


def plan_reference():
    return {"path": PLAN_PATH, "blob_sha256": PLAN_SHA256, "commit": PLAN_COMMIT}


def preexecution_claim_payload(plan, head, lock_epoch, preclaim_process):
    commands = plan["commit_governance"]["implementation_commit"]["preexecution_no_protected_bank_verification"]["commands"]
    child_timeouts = plan["execution_policy"]["preexecution"]["child_timeout_seconds_exact"]
    overrides = plan["execution_policy"]["preexecution"]["command_environment_overrides_exact"]
    bindings = [{
        "argv_index": 8, "producer_command_index": 2, "binary_role": "game_gate",
        "path": commands[2][-1],
    }]
    schedule_object = {
        "commands": commands, "child_timeout_seconds": child_timeouts,
        "command_environment_overrides": overrides,
        "produced_executable_bindings": bindings,
    }
    return {
        "aggregate_timeout_seconds": _preexecution_aggregate_seconds(plan),
        "build_roots": [BUILD_ROOT_REL],
        "child_timeout_seconds": child_timeouts,
        "claimed_utc": utc_now(),
        "command_environment_overrides": overrides,
        "command_schedule": commands,
        "command_schedule_sha256": sha256_bytes(canonical_json(schedule_object)),
        "core_dump_disabled": True,
        "environment": exact_environment(plan),
        "implementation_commit": head,
        "lock_epoch": lock_epoch,
        "one_shot": True,
        "plan": plan_reference(),
        "preclaim_process": preclaim_process,
        "produced_executable_bindings": bindings,
        "schema": RECORD_SCHEMAS["preexecution_claim"],
    }


def publish_preexecution_claim(plan, head, lock_epoch, held_locks):
    process = process_snapshot(plan)
    require(process["clean"] is True, "preexecution-process-conflict")
    payload = preexecution_claim_payload(plan, head, lock_epoch, process)
    require(
        parse_timestamp(payload["claimed_utc"], True)
        >= parse_timestamp(lock_epoch["started_utc"], True),
        "preexecution-claim-clock",
    )
    require(
        (
            parse_timestamp(CAMPAIGN_DEADLINE, False)
            - parse_timestamp(payload["claimed_utc"], True)
        ).total_seconds() >= _initial_reserve_seconds(plan),
        "preexecution-reserve",
    )
    return publish_record(plan, "preexecution_claims", head + ".json", payload, held_locks)


def _host_observation(binding):
    uname = os.uname()
    observed = {
        "system": uname.sysname, "node": uname.nodename, "release": uname.release,
        "version": uname.version, "machine": uname.machine,
        "logical_cpu_count": os.cpu_count(),
    }
    prior = binding["host"]
    for key, value in observed.items():
        require(value == prior[key], "host-drift")
    return {
        "schema": CAMPAIGN_SCHEMA_PREFIX + "host-observation-v1",
        "checked_utc": utc_now(), **observed, "cpu_model": prior["cpu_model"],
        "sysctl_command_index": 7,
    }


def _canonical_array_digest(values):
    return sha256_bytes(canonical_json(values))


def _build_root_receipt(binary_records, passed):
    if passed:
        ordered = [{"path": "tmp", "type": "directory", "mode": "0700", "empty": True}]
        for role in ("game_gate", "stage0_control", "stage0_candidate"):
            record = binary_records[role]
            ordered.append({
                "path": os.path.basename(record["path"]), "type": "regular",
                "bytes": record["bytes"], "mode": "0755", "nlink": 1,
                "sha256": record["sha256"],
            })
        return [{
            "path": BUILD_ROOT_REL, "mode": "0755", "fresh": True,
            "tmp_mode": "0700", "tmp_empty": True,
            "entries_count": 4, "entries_sha256": _canonical_array_digest(ordered),
        }]
    mode = None
    fresh = None
    tmp_mode = None
    tmp_empty = None
    entries_count = None
    entries_digest = None
    root_absolute = root_path(BUILD_ROOT_REL)
    root_fd = _open_absolute_directory(root_absolute, missing_ok=True)
    if root_fd is not None:
        try:
            metadata = os.fstat(root_fd)
            mode = mode_string(metadata)
            require(mode == "0755", "failed-root-mode")
            fresh = True
            names = _list_directory_fd(root_fd, root_absolute)
            entries_count = len(names)
            if "tmp" in names:
                tmp_fd = _open_directory_component(root_fd, "tmp", "0700")
                try:
                    tmp_mode = mode_string(os.fstat(tmp_fd))
                    tmp_empty = _list_directory_fd(
                        tmp_fd, root_path(TMP_REL)
                    ) == []
                finally:
                    os.close(tmp_fd)
            _verify_absolute_directory_binding(root_absolute, root_fd)
        finally:
            os.close(root_fd)
    return [{
        "path": BUILD_ROOT_REL, "mode": mode, "fresh": fresh,
        "tmp_mode": tmp_mode, "tmp_empty": tmp_empty,
        "entries_count": entries_count, "entries_sha256": entries_digest,
    }]


def _resolve_preexecution_executable(index, claimed_argv, binary_records):
    if claimed_argv[0].startswith("/"):
        return (
            canonical_absolute_path(claimed_argv[0], allow_runtime=True),
            None, None, None, None, None, None,
        )
    require(index == 8, "preexecution-relative-executable")
    resolved = root_path(claimed_argv[0])
    game_record = binary_records.get("game_gate")
    require(game_record is not None, "produced-executable-missing")
    observed, retained_raw = file_identity(
        resolved, claimed_argv[0], "0755",
        game_record["sha256"], game_record["bytes"],
    )
    require(observed["sha256"] == game_record["sha256"], "produced-executable-drift")
    parent_fd, parent_path, leaf = _open_parent_absolute(resolved)
    descriptor = None
    try:
        descriptor = os.open(leaf, _regular_flags(), dir_fd=parent_fd)
        metadata = os.fstat(descriptor)
        path_metadata = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        require(
            stat.S_ISREG(metadata.st_mode)
            and mode_string(metadata) == "0755"
            and metadata.st_nlink == 1
            and metadata.st_size == game_record["bytes"]
            and _same_node(metadata, path_metadata),
            "produced-executable-binding",
        )
        _reread_open_regular(
            descriptor, retained_raw, "0755", 1, metadata,
        )
        return (
            resolved, descriptor, parent_fd, metadata, retained_raw, leaf,
            parent_path,
        )
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)
        raise


def _preexecution_failure_errors(failure):
    if failure is None:
        label = "clock-invalid"
    elif (
        isinstance(failure, ContractError) and len(failure.args) == 1
        and type(failure.args[0]) is str
        and PREEXECUTION_FAILURE_LABEL_RE.fullmatch(failure.args[0]) is not None
    ):
        label = failure.args[0]
    else:
        label = "internal-error"
    return ["preexecution-observed-failure", "detail:" + label]


def execute_preexecution(plan, head, lock_epoch, claim_record, held_locks, binding, repository, external):
    global _ACTIVE_HELPER_MONOTONIC_DEADLINE
    claim = claim_record["payload"]
    commands = claim["command_schedule"]
    timeouts = claim["child_timeout_seconds"]
    configs = translation_unit_configurations(plan)
    require(len(configs) == 3, "translation-config-count")
    config_by_scan = {1: configs[0], 3: configs[1], 5: configs[2]}
    config_by_compile = {2: configs[0], 4: configs[1], 6: configs[2]}
    role_names = {2: "game_gate", 4: "stage0_control", 6: "stage0_candidate"}
    dependency_by_role = {}
    binary_records = {}
    children = []
    started_utc = utc_now()
    monotonic_start = time.monotonic()
    _ACTIVE_HELPER_MONOTONIC_DEADLINE = (
        monotonic_start + _preexecution_aggregate_seconds(plan)
    )
    failure = None
    try:
        create_fresh_build_root()
    except BaseException as error:
        failure = error
    for index, claimed_argv in enumerate(commands):
        if failure is not None:
            break
        try:
            require(resource.getrlimit(resource.RLIMIT_CORE) == (0, 0), "core-limit")
            if index == 0:
                validate_python_runtime_manifest()
                _revalidate_regular_identity(
                    plan["frozen_inputs"]["toolchain_executables"]["python"],
                    "focused-python-drift",
                )
            elif 1 <= index <= 6:
                revalidate_preexecution_inputs(plan, repository, external)
                if index in config_by_compile:
                    config = config_by_compile[index]
                    require(config["role"] in dependency_by_role, "compile-without-scan")
                    _revalidate_captured_dependencies(
                        dependency_by_role[config["role"]]["records"],
                        repository, external,
                    )
            elif index == 7:
                _revalidate_regular_identity(
                    plan["frozen_inputs"]["runtime_tools"]["sysctl"],
                    "sysctl-tool-drift",
                )
        except BaseException as error:
            failure = error
            break
        remaining_aggregate = (
            _preexecution_aggregate_seconds(plan)
            - (time.monotonic() - monotonic_start)
        )
        remaining_campaign = seconds_until_deadline()
        cap = min(float(timeouts[index]), remaining_aggregate, remaining_campaign)
        if cap <= 0:
            failure = ContractError("preexecution-time-cap")
            break
        try:
            (
                resolved, retained_exec_fd, retained_parent_fd,
                retained_exec_metadata, retained_exec_raw,
                retained_exec_leaf, retained_parent_path,
            ) = _resolve_preexecution_executable(
                index, claimed_argv, binary_records,
            )
        except BaseException as error:
            failure = error
            break
        actual_argv = [resolved] + claimed_argv[1:]
        try:
            validate_campaign_locks(held_locks)
            result = run_bounded(
                actual_argv, cap,
                plan["execution_policy"]["limits"]["child_stdout_bytes_max"],
                plan["execution_policy"]["limits"]["child_stderr_bytes_max"],
                exact_environment(plan), ROOT,
            )
            validate_campaign_locks(held_locks)
            if 1 <= index <= 6:
                revalidate_preexecution_inputs(plan, repository, external)
                if index in config_by_compile:
                    config = config_by_compile[index]
                    _revalidate_captured_dependencies(
                        dependency_by_role[config["role"]]["records"],
                        repository, external,
                    )
            if retained_exec_fd is not None:
                _reread_open_regular(
                    retained_exec_fd, retained_exec_raw, "0755", 1,
                    retained_exec_metadata,
                )
                rebound = os.stat(
                    retained_exec_leaf, dir_fd=retained_parent_fd,
                    follow_symlinks=False,
                )
                require(
                    _same_node(retained_exec_metadata, rebound),
                    "produced-executable-post-binding",
                )
                _verify_absolute_directory_binding(
                    retained_parent_path, retained_parent_fd
                )
        except BaseException as error:
            failure = error
        finally:
            if retained_exec_fd is not None:
                os.close(retained_exec_fd)
            if retained_parent_fd is not None:
                os.close(retained_parent_fd)
        if failure is not None:
            break
        post_process = None
        post_error = None
        post_validation_error = None
        try:
            validate_python_runtime_manifest()
        except BaseException as error:
            post_validation_error = error
        try:
            post_process = process_snapshot(plan)
        except BaseException as error:
            post_error = _process_error(error)
        child = _child_record(index, claimed_argv, resolved, result, post_process, post_error)
        children.append(child)
        try:
            require(post_validation_error is None, "preexecution-post-evidence")
            require(_child_succeeded(result), "preexecution-child-failure")
            require(post_process is not None and post_process["clean"], "preexecution-post-process")
            if index == 0:
                require(result["stdout"] == b"tt_exact_collision_tests=pass tests=48\n" and result["stderr"] == b"", "focused-test-output")
                require(
                    _list_absolute_directory(root_path(TMP_REL), "0700") == [],
                    "focused-test-cleanup",
                )
            elif index in config_by_scan:
                require(result["stderr"] == b"", "dependency-stderr")
                config = config_by_scan[index]
                records = dependency_records(
                    plan, result["stdout"], config["dependency_scan_only"][2],
                    config["sources"], repository, external,
                )
                dependency_by_role[config["role"]] = {
                    "stdout_ascii": result["stdout"].decode("ascii"),
                    "stdout_sha256": sha256_bytes(result["stdout"]),
                    "records": records,
                }
            elif index in config_by_compile:
                config = config_by_compile[index]
                role = role_names[index]
                require(config["role"] in dependency_by_role, "compile-without-scan")
                require(
                    result["stdout"] == b"" and result["stderr"] == b"",
                    "trace-free-compile-streams",
                )
                require(
                    _list_absolute_directory(root_path(TMP_REL), "0700") == [],
                    "trace-free-compile-cleanup",
                )
                trace = []
                output = root_path(config["output"])
                output_parent_fd, output_parent, output_leaf = _open_parent_absolute(output)
                try:
                    before = os.stat(
                        output_leaf, dir_fd=output_parent_fd,
                        follow_symlinks=False,
                    )
                    require(
                        stat.S_ISREG(before.st_mode) and before.st_nlink == 1,
                        "binary-output-type",
                    )
                    descriptor = os.open(
                        output_leaf, _regular_flags(), dir_fd=output_parent_fd
                    )
                    try:
                        opened = os.fstat(descriptor)
                        require(_same_node(before, opened), "binary-output-race")
                        os.fchmod(descriptor, 0o755)
                        os.fsync(descriptor)
                        after_fd = os.fstat(descriptor)
                        after_path = os.stat(
                            output_leaf, dir_fd=output_parent_fd,
                            follow_symlinks=False,
                        )
                        require(
                            mode_string(after_fd) == "0755"
                            and after_fd.st_nlink == 1
                            and _same_node(opened, after_fd)
                            and _metadata_identity(after_fd)
                            == _metadata_identity(after_path),
                            "binary-output-drift",
                        )
                    finally:
                        os.close(descriptor)
                    os.fsync(output_parent_fd)
                    _verify_absolute_directory_binding(
                        output_parent, output_parent_fd
                    )
                finally:
                    os.close(output_parent_fd)
                identity, binary_raw = file_identity(output, config["output"], "0755", max_bytes=64 * 1024 * 1024)
                linkage = parse_macho_linkage(
                    binary_raw,
                    plan["frozen_inputs"]["runtime_tools"]["macho_linkage_parser"]["expected_linkage_names"],
                )
                dependency = dependency_by_role[config["role"]]
                binary_records[role] = {
                    "role": role, "path": config["output"], "bytes": identity["bytes"],
                    "mode": "0755", "nlink": 1, "sha256": identity["sha256"],
                    "dependency_scan_argv_sha256": sha256_bytes(canonical_json(commands[index - 1])),
                    "dependency_stdout_ascii": dependency["stdout_ascii"],
                    "dependency_stdout_sha256": dependency["stdout_sha256"],
                    "dependency_records": dependency["records"],
                    "dependency_count": len(dependency["records"]),
                    "dependency_records_sha256": _canonical_array_digest(dependency["records"]),
                    "compile_link_argv_sha256": sha256_bytes(canonical_json(claimed_argv)),
                    "link_trace": trace, "link_trace_count": 0,
                    "link_trace_sha256": sha256_bytes(b""),
                    "runtime_linkage_sha256": linkage["normalized_sha256"],
                }
            elif index == 7:
                sysctl = plan["frozen_inputs"]["runtime_tools"]["sysctl"]
                require(result["stdout"] == sysctl["expected_stdout_ascii"].encode("ascii") and result["stderr"] == b"", "sysctl-output")
            elif index == 8:
                expected_self_test = (
                    b"heldout_pair_self_test candidate_sweeps=1 reference_sweeps=1 "
                    b"split_pairs=1 unresolved_pairs=1 exact_accounting=pass\n"
                    b"self_test deterministic_bank_generation=pass "
                    b"exact_four_bank_count=pass render_parse_roundtrip=pass "
                    b"all_four_disjoint=pass paired_state=pass "
                    b"public_rules_only=pass transcripts=not-retained\n"
                    if GENERATED_OVERLAY_ACTIVE
                    else b"safe_bank_reader_self_test=pass\n"
                )
                require(
                    result["stdout"] == expected_self_test
                    and result["stderr"] == b"",
                    "safe-reader-output",
                )
                require(
                    _list_absolute_directory(root_path(TMP_REL), "0700") == [],
                    "safe-reader-cleanup",
                )
        except BaseException as error:
            failure = error
    ended_utc = utc_now()
    elapsed = max(0.0, time.monotonic() - monotonic_start)
    passed = (
        failure is None and len(children) == 9
        and elapsed <= _preexecution_aggregate_seconds(plan)
        and _replay_wall_matches(
            parse_timestamp(started_utc, True),
            parse_timestamp(ended_utc, True), elapsed,
        )
    )
    passed_host = None
    if passed:
        try:
            require(
                set(binary_records)
                == {"game_gate", "stage0_control", "stage0_candidate"},
                "preexecution-binary-record-set",
            )
            validate_build_root(binary_records)
            passed_host = _host_observation(binding)
        except BaseException as error:
            failure = error
            passed = False
    if passed:
        union = {}
        for role in ("game_gate", "stage0_control", "stage0_candidate"):
            for record in binary_records[role]["dependency_records"]:
                existing = union.get(record["path"])
                require(existing is None or existing == record, "dependency-union-conflict")
                union[record["path"]] = record
        dependency_closure = {
            "algorithm": "sorted-path-file-identity-v1",
            "count": len(union),
            "records_sha256": _canonical_array_digest(sorted(union.values(), key=lambda item: item["path"].encode("utf-8"))),
            "records": sorted(union.values(), key=lambda item: item["path"].encode("utf-8")),
        }
        host = passed_host
        compiler_plan = plan["commit_governance"]["implementation_commit"]["preexecution_no_protected_bank_verification"]
        translation_digest = _canonical_array_digest(configs)
        compiler_records = {"clang-release": {
            "build_root": BUILD_ROOT_REL,
            "compiler": commands[1][0], "driver_mode": "g++",
            "sdk_path": "/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk",
            "resource_dir": "/Library/Developer/CommandLineTools/usr/lib/clang/21",
            "minimum_macos_version": "26.0",
            "translation_unit_configurations_sha256": translation_digest,
            "dependency_records_sha256": dependency_closure["records_sha256"],
            "implicit_driver_input_records_sha256": _canonical_array_digest(compiler_plan["implicit_driver_input_identities"]),
            "nested_program_records_sha256": _canonical_array_digest(compiler_plan["nested_program_identities"]),
            "nested_runtime_records_sha256": _canonical_array_digest(compiler_plan["nested_runtime_identities"]),
            "link_input_records_sha256": _canonical_array_digest(compiler_plan["link_input_identities"]),
        }}
        toolchain = plan["frozen_inputs"]["toolchain_executables"]
        errors = []
        build_roots = _build_root_receipt(binary_records, True)
    else:
        binary_records = None
        compiler_records = None
        dependency_closure = None
        host = None
        toolchain = None
        errors = _preexecution_failure_errors(failure)
        build_roots = _build_root_receipt({}, False)
    receipt_created_utc = utc_now()
    if passed and (
        parse_timestamp(CAMPAIGN_DEADLINE, False)
        - parse_timestamp(receipt_created_utc, True)
    ).total_seconds() < _continuation_reserve_seconds(plan):
        passed = False
        binary_records = None
        compiler_records = None
        dependency_closure = None
        host = None
        toolchain = None
        errors = _preexecution_failure_errors(None)
        build_roots = _build_root_receipt({}, False)
    payload = {
        "aggregate_timeout_seconds": _preexecution_aggregate_seconds(plan),
        "binary_records": binary_records,
        "build_roots": build_roots,
        "child_timeout_seconds": claim["child_timeout_seconds"],
        "claim": record_reference(plan, "preexecution_claims", claim_record),
        "claim_sha256": claim_record["sha256"],
        "commands": children,
        "compiler_records": compiler_records,
        "created_utc": receipt_created_utc,
        "dependency_closure": dependency_closure,
        "elapsed_monotonic_seconds": elapsed,
        "ended_utc": ended_utc,
        "errors": errors,
        "host_observation": host,
        "lock_epoch": lock_epoch,
        "passed": passed,
        "schema": RECORD_SCHEMAS["preexecution_receipt"],
        "started_utc": started_utc,
        "toolchain": toolchain,
    }
    raw = canonical_json(payload)
    name = sha256_bytes(raw) + ".json"
    receipt = publish_record(plan, "preexecution_receipts", name, payload, held_locks)
    _ACTIVE_HELPER_MONOTONIC_DEADLINE = None
    return receipt


def candidate_identity_from_receipt(plan, head, receipt_record, binding, sources):
    receipt = receipt_record["payload"]
    require(receipt["passed"] is True and receipt["errors"] == [], "candidate-receipt")
    validate_passed_receipt(plan, receipt_record)
    configuration, _stages = configuration_digests(plan)
    binary_records = receipt["binary_records"]
    binaries = {
        role: {
            "path": binary_records[role]["path"], "bytes": binary_records[role]["bytes"],
            "mode": binary_records[role]["mode"], "sha256": binary_records[role]["sha256"],
        }
        for role in ("game_gate", "stage0_candidate", "stage0_control")
    }
    host_observation = receipt["host_observation"]
    current_host = {
        key: host_observation[key]
        for key in ("cpu_model", "logical_cpu_count", "machine", "node", "release", "system", "version")
    }
    linkage_parser = plan["frozen_inputs"]["runtime_tools"]["macho_linkage_parser"]
    runtime_linkage = {}
    for role in ("game_gate", "stage0_candidate", "stage0_control"):
        runtime_linkage[role] = {
            "binary_path": binaries[role]["path"],
            "parser_schema": linkage_parser["algorithm"],
            "normalized_linkage_names": linkage_parser["expected_linkage_names"],
            "normalized_sha256": binary_records[role]["runtime_linkage_sha256"],
        }
    prior_host = dict(binding["host"])
    prior_host_sha = prior_host.pop("sha256")
    require(prior_host_sha == "1a7f59560af8acc4bc4533679ffc1fe83a835bf979928bb47909c7cbffbed30c", "prior-host-sha")
    prior_host_projection = {
        key: prior_host[key] for key in current_host
    }
    require(current_host == prior_host_projection, "candidate-host-binding")
    receipt_reference = record_reference(plan, "preexecution_receipts", receipt_record)
    payload = {
        "schema": CAMPAIGN_SCHEMA_PREFIX + "candidate-identity-v1",
        "implementation_commit": head,
        "plan": plan_reference(),
        "preexecution_receipt": {
            "path": receipt_reference["path"], "bytes": receipt_reference["bytes"],
            "sha256": receipt_reference["sha256"], "schema": receipt_reference["schema"],
            "passed": True,
        },
        "binaries": binaries,
        "dependency_closure": receipt["dependency_closure"],
        "configuration": configuration,
        "environment": exact_environment(plan),
        "host_runtime": {
            "prior_host": prior_host, "prior_host_sha256": prior_host_sha,
            "prior_runtime": binding["runtime"],
            "bootstrap_runtime": plan["frozen_inputs"]["python_runtime_manifest"],
            "current_host": current_host,
            "process_tools": {
                "ps": plan["frozen_inputs"]["runtime_tools"]["ps"],
                "sysctl": plan["frozen_inputs"]["runtime_tools"]["sysctl"],
            },
            "runtime_linkage": runtime_linkage,
        },
        "sources": sources,
        "toolchain": plan["frozen_inputs"]["toolchain_executables"],
    }
    identifier = sha256_bytes(canonical_json(payload))
    require(HEX64_RE.fullmatch(identifier) is not None, "candidate-identity")
    return identifier, payload


def validate_passed_receipt(plan, receipt_record):
    receipt = receipt_record["payload"]
    require(receipt["passed"] is True and receipt["errors"] == [], "receipt-passed")
    require(len(receipt["commands"]) == 9, "receipt-command-count")
    empty_stream = {"bytes": 0, "sha256": EMPTY_SHA256, "truncated": False}

    def exact_stream(raw):
        return {
            "bytes": len(raw), "sha256": sha256_bytes(raw), "truncated": False,
        }

    command_keys = (
        "index", "argv", "environment_override", "resolved_executable_path",
        "started_utc", "ended_utc", "elapsed_monotonic_seconds", "returncode",
        "timed_out", "os_error", "stdout_receipt", "stderr_receipt",
        "postchild_process", "postchild_process_error",
    )
    schedule = plan["commit_governance"]["implementation_commit"]["preexecution_no_protected_bank_verification"]["commands"]
    for index, child in enumerate(receipt["commands"]):
        exact_keys(child, command_keys, "receipt-command-keys")
        require(child["index"] == index and child["argv"] == schedule[index], "receipt-command-binding")
        require(child["environment_override"] == {} and child["returncode"] == 0, "receipt-command-result")
        require(child["timed_out"] is False and child["os_error"] is None, "receipt-command-result")
        require(child["postchild_process"] is not None and child["postchild_process_error"] is None, "receipt-command-process")
        validate_stream_receipt(child["stdout_receipt"], plan["execution_policy"]["limits"]["child_stdout_bytes_max"])
        validate_stream_receipt(child["stderr_receipt"], plan["execution_policy"]["limits"]["child_stderr_bytes_max"])
        require(child["stdout_receipt"]["truncated"] is False and child["stderr_receipt"]["truncated"] is False, "receipt-command-cap")
    require(
        receipt["commands"][0]["stdout_receipt"]
        == exact_stream(b"tt_exact_collision_tests=pass tests=48\n")
        and receipt["commands"][0]["stderr_receipt"] == empty_stream,
        "receipt-focused-output",
    )
    sysctl_stdout = plan["frozen_inputs"]["runtime_tools"]["sysctl"][
        "expected_stdout_ascii"
    ].encode("ascii")
    require(
        receipt["commands"][7]["stdout_receipt"] == exact_stream(sysctl_stdout)
        and receipt["commands"][7]["stderr_receipt"] == empty_stream,
        "receipt-sysctl-output",
    )
    require(
        receipt["commands"][8]["stdout_receipt"]
        == exact_stream(
            (
                b"heldout_pair_self_test candidate_sweeps=1 "
                b"reference_sweeps=1 split_pairs=1 unresolved_pairs=1 "
                b"exact_accounting=pass\n"
                b"self_test deterministic_bank_generation=pass "
                b"exact_four_bank_count=pass render_parse_roundtrip=pass "
                b"all_four_disjoint=pass paired_state=pass "
                b"public_rules_only=pass transcripts=not-retained\n"
            )
            if GENERATED_OVERLAY_ACTIVE
            else b"safe_bank_reader_self_test=pass\n"
        )
        and receipt["commands"][8]["stderr_receipt"] == empty_stream,
        "receipt-safe-reader-output",
    )
    binary_keys = (
        "role", "path", "bytes", "mode", "nlink", "sha256",
        "dependency_scan_argv_sha256", "dependency_stdout_ascii",
        "dependency_stdout_sha256", "dependency_records", "dependency_count",
        "dependency_records_sha256", "compile_link_argv_sha256", "link_trace",
        "link_trace_count", "link_trace_sha256", "runtime_linkage_sha256",
    )
    require(
        isinstance(receipt["binary_records"], dict)
        and set(receipt["binary_records"])
        == {"game_gate", "stage0_candidate", "stage0_control"},
        "receipt-binary-roles",
    )
    role_indices = {"game_gate": (1, 2), "stage0_control": (3, 4), "stage0_candidate": (5, 6)}
    configurations = translation_unit_configurations(plan)
    configuration_by_role = {
        "game_gate": configurations[0],
        "stage0_control": configurations[1],
        "stage0_candidate": configurations[2],
    }
    implementation_commit = os.path.basename(receipt["claim"]["path"])[:-5]
    require(
        HEX40_RE.fullmatch(implementation_commit) is not None,
        "receipt-implementation-commit",
    )
    binding = load_prior_binding(plan)
    repository_allowed, external_allowed = source_and_external_closures(
        plan, implementation_commit, binding, validate_bytes=False,
    )
    provenance_only = _external_provenance_only_paths(plan, external_allowed)
    allowed_dependencies = dict(repository_allowed)
    allowed_dependencies.update(external_allowed)
    union = {}
    for role, indices in role_indices.items():
        record = receipt["binary_records"][role]
        exact_keys(record, binary_keys, "receipt-binary-keys")
        scan_index, compile_index = indices
        configuration = configuration_by_role[role]
        require(
            record["role"] == role and record["path"] == configuration["output"]
            and record["mode"] == "0755" and record["nlink"] == 1,
            "receipt-binary-identity",
        )
        _replay_json_int(record["bytes"], "receipt-binary-bytes", 1)
        require(HEX64_RE.fullmatch(record["sha256"]) is not None, "receipt-binary-sha")
        require(record["dependency_scan_argv_sha256"] == sha256_bytes(canonical_json(schedule[scan_index])), "receipt-scan-argv")
        require(record["compile_link_argv_sha256"] == sha256_bytes(canonical_json(schedule[compile_index])), "receipt-compile-argv")
        dependency_raw = record["dependency_stdout_ascii"].encode("ascii")
        require(record["dependency_stdout_sha256"] == sha256_bytes(dependency_raw), "receipt-dependency-stream")
        require(receipt["commands"][scan_index]["stdout_receipt"]["bytes"] == len(dependency_raw), "receipt-dependency-bytes")
        require(receipt["commands"][scan_index]["stdout_receipt"]["sha256"] == record["dependency_stdout_sha256"], "receipt-dependency-receipt")
        require(
            receipt["commands"][scan_index]["stdout_receipt"]
            == exact_stream(dependency_raw)
            and receipt["commands"][scan_index]["stderr_receipt"] == empty_stream,
            "receipt-dependency-streams",
        )
        require(record["dependency_count"] == len(record["dependency_records"]) > 0, "receipt-dependency-count")
        require(record["dependency_records_sha256"] == _canonical_array_digest(record["dependency_records"]), "receipt-dependency-digest")
        dependency_paths = normalized_dependency_paths(
            dependency_raw, configuration["dependency_scan_only"][2]
        )
        require(
            all(source in dependency_paths for source in configuration["sources"]),
            "receipt-dependency-source",
        )
        require(
            [item["path"] for item in record["dependency_records"]]
            == dependency_paths,
            "receipt-dependency-paths",
        )
        require(
            record["link_trace"] == []
            and record["link_trace_count"] == 0
            and record["link_trace_sha256"] == sha256_bytes(b""),
            "receipt-trace-free",
        )
        compile_child = receipt["commands"][compile_index]
        require(
            compile_child["stdout_receipt"] == empty_stream
            and compile_child["stderr_receipt"] == empty_stream,
            "receipt-trace-receipt",
        )
        require(record["runtime_linkage_sha256"] == plan["frozen_inputs"]["runtime_tools"]["macho_linkage_parser"]["expected_normalized_sha256"], "receipt-linkage")
        for dependency in record["dependency_records"]:
            exact_keys(dependency, ("path", "bytes", "mode", "sha256"), "receipt-dependency-record")
            require(
                dependency["path"] not in provenance_only,
                "receipt-dependency-provenance-only",
            )
            _replay_identity(dependency, "receipt-dependency-record")
            expected_dependency = allowed_dependencies.get(dependency["path"])
            if expected_dependency is None:
                require(
                    dependency == _captured_external_dependency(
                        dependency["path"]
                    ),
                    "receipt-dependency-captured-drift",
                )
            else:
                require(
                    dependency == expected_dependency,
                    "receipt-dependency-not-preregistered",
                )
            existing = union.get(dependency["path"])
            require(existing is None or existing == dependency, "receipt-dependency-conflict")
            union[dependency["path"]] = dependency
    ordered_union = sorted(union.values(), key=lambda item: item["path"].encode("utf-8"))
    closure = receipt["dependency_closure"]
    exact_keys(closure, ("algorithm", "count", "records_sha256", "records"), "receipt-closure-keys")
    require(closure["algorithm"] == "sorted-path-file-identity-v1" and closure["records"] == ordered_union, "receipt-closure")
    require(closure["count"] == len(ordered_union) and closure["records_sha256"] == _canonical_array_digest(ordered_union), "receipt-closure-digest")
    require(receipt["toolchain"] == plan["frozen_inputs"]["toolchain_executables"], "receipt-toolchain")
    compiler_plan = plan["commit_governance"]["implementation_commit"][
        "preexecution_no_protected_bank_verification"
    ]
    expected_compiler_records = {"clang-release": {
        "build_root": BUILD_ROOT_REL,
        "compiler": schedule[1][0],
        "driver_mode": "g++",
        "sdk_path": "/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk",
        "resource_dir": "/Library/Developer/CommandLineTools/usr/lib/clang/21",
        "minimum_macos_version": "26.0",
        "translation_unit_configurations_sha256": _canonical_array_digest(configurations),
        "dependency_records_sha256": closure["records_sha256"],
        "implicit_driver_input_records_sha256": _canonical_array_digest(
            compiler_plan["implicit_driver_input_identities"]
        ),
        "nested_program_records_sha256": _canonical_array_digest(
            compiler_plan["nested_program_identities"]
        ),
        "nested_runtime_records_sha256": _canonical_array_digest(
            compiler_plan["nested_runtime_identities"]
        ),
        "link_input_records_sha256": _canonical_array_digest(
            compiler_plan["link_input_identities"]
        ),
    }}
    require(
        receipt["compiler_records"] == expected_compiler_records,
        "receipt-compiler-records",
    )
    host = receipt["host_observation"]
    exact_keys(
        host,
        (
            "schema", "checked_utc", "system", "node", "release", "version",
            "machine", "logical_cpu_count", "cpu_model", "sysctl_command_index",
        ),
        "receipt-host-keys",
    )
    require(
        host["schema"] == CAMPAIGN_SCHEMA_PREFIX + "host-observation-v1"
        and host["sysctl_command_index"] == 7,
        "receipt-host-schema",
    )
    for key in ("system", "node", "release", "version", "machine", "cpu_model"):
        require(isinstance(host[key], str) and host[key] != "", "receipt-host-value")
    _replay_json_int(host["logical_cpu_count"], "receipt-host-cpu-count", 1)
    require(
        sysctl_stdout.endswith(b"\n")
        and host["cpu_model"] == sysctl_stdout[:-1].decode("ascii"),
        "receipt-host-sysctl-binding",
    )
    require(
        parse_timestamp(receipt["ended_utc"], True)
        <= parse_timestamp(host["checked_utc"], True)
        <= parse_timestamp(receipt["created_utc"], True),
        "receipt-host-clock",
    )
    require(
        receipt["build_roots"] == _build_root_receipt(receipt["binary_records"], True),
        "receipt-build-root",
    )
    return True


def _walk_identity_records(value, destination):
    if isinstance(value, dict):
        if set(("path", "bytes", "mode", "sha256")).issubset(value):
            path = value["path"]
            if isinstance(path, str):
                destination.append(_identity_projection(value))
        for child in value.values():
            _walk_identity_records(child, destination)
    elif isinstance(value, list):
        for child in value:
            _walk_identity_records(child, destination)


def frozen_governance_projection(plan):
    records = []
    frozen = plan["frozen_inputs"]
    for key in ("build_and_gate", "generator_and_models", "mask7_source_closure", "recorder_reference_helpers"):
        _walk_identity_records(frozen[key], records)
    _walk_identity_records(frozen["git_metadata"]["files"], records)
    _walk_identity_records(frozen["runtime_tools"]["ps"], records)
    _walk_identity_records(frozen["runtime_tools"]["sysctl"], records)
    _walk_identity_records(frozen["toolchain_executables"], records)
    _walk_identity_records(plan["prior_evidence"], records)
    _walk_identity_records(plan["prior_terminal_decisions"], records)
    records.append(_identity_projection(plan["execution_policy"]["outer_cli"]["bootstrap_env_identity"]))
    by_path = {}
    for record in records:
        existing = by_path.get(record["path"])
        require(existing is None or existing == record, "governance-identity-conflict")
        by_path[record["path"]] = record
    closure = plan["commit_governance"]["implementation_commit"][
        "path_closure"
    ]
    implementation_paths = set(closure["modified"] + closure["new"])
    require(len(implementation_paths) == 8, "governance-implementation-paths")
    for path in implementation_paths:
        by_path.pop(path, None)
    return sorted(by_path.values(), key=lambda item: item["path"].encode("utf-8"))


def frozen_governance_records(plan):
    ordered = frozen_governance_projection(plan)
    for record in ordered:
        path = record["path"]
        if path.startswith("/"):
            absolute = canonical_absolute_path(path, allow_runtime=True)
        else:
            require(
                not path.endswith(".tsv")
                and "openings" not in path.split("/"),
                "governance-bank-path",
            )
            absolute = root_path(path, allow_protected_literal=True)
        observed, _raw = file_identity(absolute, path, record["mode"], record["sha256"], record["bytes"])
        require(observed == record, "governance-drift")
    return ordered


def sdk_alias_records_projection():
    return [
        {
            "type": "selector", "link_path": "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk",
            "link_mode": "0755", "link_nlink": 1, "link_text": "MacOSX26.5.sdk",
            "resolved_path": "/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk",
            "resolved_type": "directory",
        },
        {
            "type": "leaf", "link_path": "/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk/usr/include/pthread.h",
            "link_mode": "0755", "link_nlink": 1, "link_text": "pthread/pthread.h",
            "resolved_path": "/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk/usr/include/pthread/pthread.h",
            "resolved_type": "regular", "resolved_mode": "0644", "resolved_nlink": 1,
            "resolved_bytes": 28093,
            "resolved_sha256": "9d621c730d1d96b600893b0e3e4c45822a24d565e7b6b166973c41a6a2eb02e7",
        },
        {
            "type": "leaf", "link_path": "/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk/usr/include/sched.h",
            "link_mode": "0755", "link_nlink": 1, "link_text": "pthread/sched.h",
            "resolved_path": "/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk/usr/include/pthread/sched.h",
            "resolved_type": "regular", "resolved_mode": "0644", "resolved_nlink": 1,
            "resolved_bytes": 1410,
            "resolved_sha256": "07d65b8d135be978a33579b5a333bd4576a8c6b36351ef510e0cca2aa0987020",
        },
    ]


def portability_projection(plan, candidate_payload):
    alias_records = sdk_alias_records_projection()
    for record in alias_records:
        parent_fd, parent_path, leaf = _open_parent_absolute(record["link_path"])
        try:
            link_metadata = os.stat(
                leaf, dir_fd=parent_fd, follow_symlinks=False
            )
            require(
                stat.S_ISLNK(link_metadata.st_mode)
                and mode_string(link_metadata) == record["link_mode"]
                and link_metadata.st_nlink == record["link_nlink"],
                "sdk-alias-link",
            )
            link_text = os.readlink(leaf, dir_fd=parent_fd)
            after_link = os.stat(
                leaf, dir_fd=parent_fd, follow_symlinks=False
            )
            require(
                link_text == record["link_text"]
                and _metadata_identity(link_metadata)
                == _metadata_identity(after_link),
                "sdk-alias-text",
            )
            _verify_absolute_directory_binding(parent_path, parent_fd)
        finally:
            os.close(parent_fd)
        if record["type"] == "selector":
            resolved = os.path.normpath(
                os.path.join(os.path.dirname(record["link_path"]), link_text)
            )
            require(resolved == record["resolved_path"], "sdk-selector")
            resolved_fd = _open_absolute_directory(resolved)
            try:
                require(
                    stat.S_ISDIR(os.fstat(resolved_fd).st_mode),
                    "sdk-selector",
                )
            finally:
                os.close(resolved_fd)
        else:
            observed, _raw = file_identity(
                record["resolved_path"], record["resolved_path"], record["resolved_mode"],
                record["resolved_sha256"], record["resolved_bytes"],
            )
            require(observed["bytes"] == record["resolved_bytes"], "sdk-leaf")
    closure = candidate_payload["dependency_closure"]
    new_object = {
        "dependency_count": closure["count"],
        "dependency_records_sha256": closure["records_sha256"],
        "sdk_alias_records": alias_records,
    }
    return {
        "prior_sha256": "b058be2c2fccac9c4f907ba9e81ba0a372a024307a9ed62541419411afd2a5eb",
        "new_sha256": sha256_bytes(canonical_json(new_object)),
        **new_object,
    }


def stable_evidence(plan, head, candidate_identity, candidate_payload, receipt_record, repository):
    validate_python_runtime_manifest()
    live_head = git_query(
        ["rev-parse", "--verify", "HEAD^{commit}"]
    ).decode("ascii").strip()
    require(live_head == head, "stable-head-drift")
    require(
        sha256_bytes(canonical_json(candidate_payload)) == candidate_identity,
        "stable-candidate-derivation",
    )
    uname = os.uname()
    current = candidate_payload["host_runtime"]["current_host"]
    observed_host = {
        "system": uname.sysname, "node": uname.nodename, "release": uname.release,
        "version": uname.version, "machine": uname.machine,
        "logical_cpu_count": os.cpu_count(),
    }
    for key, value in observed_host.items():
        require(current[key] == value, "stable-host-drift")
    binary_records = candidate_payload["binaries"]
    validate_build_root({role: value for role, value in binary_records.items()})
    for record in binary_records.values():
        _revalidate_regular_identity(record, "stable-binary-drift")
    for record in candidate_payload["dependency_closure"]["records"]:
        _revalidate_regular_identity(record, "stable-dependency-drift")
    governance = frozen_governance_records(plan)
    governance_by_path = {
        record["path"]: _identity_projection(record)
        for record in governance
        if isinstance(record, dict)
        and all(key in record for key in ("path", "bytes", "mode", "sha256"))
    }
    bounded_paths = set(repository)
    bounded_paths.add(PLAN_PATH)
    for record in governance:
        path = record["path"]
        if (
            not path.startswith("/") and path != ".git"
            and not path.startswith(".git/") and not path.endswith(".tsv")
            and "openings" not in path.split("/")
        ):
            bounded_paths.add(path)
    bounded_paths.update(
        plan["commit_governance"]["implementation_commit"]["production_paths_must_remain_byte_identical"]
    )
    ordered_bounded_paths = sorted(
        bounded_paths, key=lambda item: item.encode("utf-8")
    )
    tree_cache_key = (head, tuple(ordered_bounded_paths))
    cached_tree_records = _STABLE_TREE_RECORDS_CACHE.get(tree_cache_key)
    tree_records = []
    work_records = []
    for path_index, path in enumerate(ordered_bounded_paths):
        canonical_relative_path(path, allow_protected_literal=True)
        observed, work_raw = file_identity(root_path(path, allow_protected_literal=True), path, max_bytes=32 * 1024 * 1024)
        expected_identity = repository.get(path, governance_by_path.get(path))
        if path == PLAN_PATH:
            expected_identity = {
                "path": PLAN_PATH, "bytes": PLAN_BYTES,
                "mode": "0644", "sha256": PLAN_SHA256,
            }
        require(expected_identity is not None, "stable-worktree-identity-missing")
        require(observed == expected_identity, "stable-worktree-identity-drift")
        if cached_tree_records is None:
            tree_entry = git_query(
                ["ls-tree", "-z", head, "--", path], stdout_cap=4096
            )
            require(tree_entry.endswith(b"\x00"), "stable-tree-entry-terminator")
            metadata_raw, tree_path_raw = tree_entry[:-1].split(b"\t", 1)
            mode_raw, object_type_raw, object_id_raw = metadata_raw.split(b" ", 2)
            require(
                mode_raw == b"100644" and object_type_raw == b"blob"
                and HEX40_RE.fullmatch(object_id_raw.decode("ascii")) is not None
                and tree_path_raw.decode("utf-8") == path,
                "stable-tree-entry",
            )
            raw = git_query(
                ["show", head + ":" + path],
                stdout_cap=32 * 1024 * 1024,
            )
            require(raw == work_raw, "stable-worktree")
            tree_records.append({
                "path": path, "git_mode": mode_raw.decode("ascii"),
                "blob_sha256": sha256_bytes(raw),
            })
        else:
            cached_record = cached_tree_records[path_index]
            require(
                cached_record["path"] == path
                and cached_record["git_mode"] == "100644"
                and cached_record["blob_sha256"] == expected_identity["sha256"],
                "stable-tree-cache",
            )
            tree_records.append(dict(cached_record))
        work_records.append(observed)
    tree = git_query(
        ["rev-parse", "--verify", "HEAD^{tree}"]
    ).decode("ascii").strip()
    require(HEX40_RE.fullmatch(tree) is not None, "stable-tree")
    live_head_after = git_query(
        ["rev-parse", "--verify", "HEAD^{commit}"]
    ).decode("ascii").strip()
    require(live_head_after == head, "stable-head-drift")
    if cached_tree_records is None:
        _STABLE_TREE_RECORDS_CACHE[tree_cache_key] = tuple(
            dict(record) for record in tree_records
        )
    admin = {
        "head": head, "parent": PLAN_COMMIT, "tree": tree,
        "plan_commit": PLAN_COMMIT, "implementation_commit": head,
        "bounded_tree_sha256": sha256_bytes(canonical_json(tree_records)),
        "bounded_worktree_sha256": sha256_bytes(canonical_json(work_records)),
    }
    receipt_reference = record_reference(plan, "preexecution_receipts", receipt_record)
    return {
        "schema": CAMPAIGN_SCHEMA_PREFIX + "stable-evidence-v1",
        "candidate_identity": candidate_identity,
        "candidate_identity_payload": candidate_payload,
        "admin": admin,
        "frozen_governance": governance,
        "portability": portability_projection(plan, candidate_payload),
        "preexecution_receipt": receipt_reference,
    }


def revalidate_stage_child_evidence(
    plan, claim, receipt_record, repository,
):
    expected = claim["evidence_before"]
    candidate_payload = expected["candidate_identity_payload"]
    observed = stable_evidence(
        plan, claim["implementation_commit"], claim["candidate_identity"],
        candidate_payload, receipt_record, repository,
    )
    require(observed == expected, "stage-stable-evidence-drift")


def stage_specification(plan, stage):
    if stage == STAGE_ORDER[0]:
        schedule, timeouts = build_stage0_schedule(plan)
        bank_metadata = []
        executable_roles = ("stage0_control", "stage0_candidate")
    elif stage == STAGE_ORDER[1]:
        stage_record = next(
            item for item in plan["stages"] if item["stage"] == stage
        )
        schedule = [[*stage_record[
            "argv" if GENERATED_OVERLAY_ACTIVE else "command"
        ]]]
        timeouts = [_stage_aggregate_seconds(plan)]
        bank_metadata = (
            [dict(item) for item in plan["development_banks"]]
            if GENERATED_OVERLAY_ACTIVE
            else [dict(plan["development_banks"][3])]
        )
        executable_roles = ("game_gate",)
    elif stage == STAGE_ORDER[2]:
        stage_record = next(
            item for item in plan["stages"] if item["stage"] == stage
        )
        schedule = [[*stage_record[
            "argv" if GENERATED_OVERLAY_ACTIVE else "command"
        ]]]
        timeouts = [_stage_aggregate_seconds(plan)]
        bank_metadata = [
            dict(item) for item in (
                plan["development_banks"]
                if GENERATED_OVERLAY_ACTIVE
                else plan["development_banks"][:3]
            )
        ]
        executable_roles = ("game_gate",)
    else:
        raise ContractError("stage-name")
    return schedule, timeouts, bank_metadata, executable_roles


def stage_claim_payload(
    plan, head, lock_epoch, stage, candidate_identity, candidate_payload,
    receipt_record, evidence, predecessor_report_records, preclaim_process,
    claimed_utc,
):
    schedule, timeouts, bank_metadata, executable_roles = stage_specification(plan, stage)
    binaries = candidate_payload["binaries"]
    executables = [binaries[role] for role in executable_roles]
    _base_configuration, stage_digests = configuration_digests(plan)
    return {
        "argv_schedule": schedule,
        "argv_schedule_sha256": sha256_bytes(canonical_json(schedule)),
        "authorized_bank_plan_metadata": bank_metadata,
        "candidate_identity": candidate_identity,
        "child_timeout_seconds": timeouts,
        "claimed_utc": claimed_utc,
        "configuration_sha256": stage_digests[stage],
        "core_dump_disabled": True,
        "environment": exact_environment(plan),
        "evidence_before": evidence,
        "evidence_before_sha256": sha256_bytes(canonical_json(evidence)),
        "executables": executables,
        "implementation_commit": head,
        "lock_epoch": lock_epoch,
        "one_shot": True,
        "outer_invocation": plan["execution_policy"]["outer_cli"]["run"],
        "plan": plan_reference(),
        "preclaim_process": preclaim_process,
        "preexecution_receipt": record_reference(plan, "preexecution_receipts", receipt_record),
        "predecessor_reports": [record_reference(plan, "reports", record) for record in predecessor_report_records],
        "schema": RECORD_SCHEMAS["claim"],
        "stage": stage,
        "timeout_seconds": _stage_aggregate_seconds(plan),
    }


def publish_stage_claim(
    plan, head, lock_epoch, held_locks, stage, candidate_identity,
    candidate_payload, receipt_record, predecessor_reports, evidence,
    preclaim_process,
):
    require(preclaim_process["clean"] is True, "stage-process-conflict")
    claimed_utc = preclaim_process["checked_utc"]
    payload = stage_claim_payload(
        plan, head, lock_epoch, stage, candidate_identity, candidate_payload,
        receipt_record, evidence, predecessor_reports, preclaim_process,
        claimed_utc,
    )
    name = candidate_identity + "." + stage + ".json"
    return publish_record(plan, "claims", name, payload, held_locks)


def _execution_echo(claim):
    keys = (
        "candidate_identity", "stage", "lock_epoch", "core_dump_disabled",
        "executables", "argv_schedule", "argv_schedule_sha256",
        "configuration_sha256", "environment", "outer_invocation",
        "timeout_seconds", "child_timeout_seconds", "authorized_bank_plan_metadata",
    )
    return {key: claim[key] for key in keys}


def execute_stage(
    plan, lock_epoch, held_locks, claim_record, receipt_record, repository,
):
    global _ACTIVE_HELPER_MONOTONIC_DEADLINE
    claim = claim_record["payload"]
    schedule = claim["argv_schedule"]
    timeouts = claim["child_timeout_seconds"]
    children = []
    raw_outputs = []
    resolved_paths = []
    started_utc = utc_now()
    monotonic_start = time.monotonic()
    _ACTIVE_HELPER_MONOTONIC_DEADLINE = (
        monotonic_start + _continuation_reserve_seconds(plan)
    )
    stopped = False
    for index, claimed_argv in enumerate(schedule):
        if stopped:
            break
        try:
            require(resource.getrlimit(resource.RLIMIT_CORE) == (0, 0), "core-limit")
            executable = next((item for item in claim["executables"] if item["path"] == claimed_argv[0]), None)
            require(executable is not None, "stage-executable")
            resolved = root_path(claimed_argv[0])
            observed, _binary_raw = file_identity(resolved, claimed_argv[0], executable["mode"], executable["sha256"], executable["bytes"])
            require(observed == executable, "stage-executable-drift")
            remaining = (
                _stage_aggregate_seconds(plan)
                - (time.monotonic() - monotonic_start)
            )
            cap = min(float(timeouts[index]), remaining, seconds_until_deadline())
            require(cap > 0, "stage-time-cap")
            validate_campaign_locks(held_locks)
            result = run_bounded(
                [resolved] + claimed_argv[1:], cap,
                plan["execution_policy"]["limits"]["child_stdout_bytes_max"],
                plan["execution_policy"]["limits"]["child_stderr_bytes_max"],
                exact_environment(plan), ROOT,
            )
            validate_campaign_locks(held_locks)
        except BaseException:
            stopped = True
            break
        post_process = None
        post_error = None
        post_validation_error = None
        try:
            revalidate_stage_child_evidence(
                plan, claim, receipt_record, repository,
            )
        except BaseException as error:
            post_validation_error = error
        try:
            post_process = process_snapshot(plan)
        except BaseException as error:
            post_error = _process_error(error)
        children.append(_stage_child_record(index, claimed_argv, resolved, result, post_process, post_error))
        resolved_paths.append(resolved)
        raw_outputs.append(result["stdout"])
        stopped = not (
            _child_succeeded(result) and result["stderr"] == b""
            and post_process is not None and post_process["clean"]
            and post_validation_error is None
        )
    ended_utc = utc_now()
    payload = {
        "all_children_completed": len(children) == len(schedule),
        "candidate_identity": claim["candidate_identity"],
        "children": children,
        "claim": record_reference(plan, "claims", claim_record),
        "claim_payload_echo": _execution_echo(claim),
        "created_utc": utc_now(),
        "elapsed_monotonic_seconds": max(0.0, time.monotonic() - monotonic_start),
        "ended_utc": ended_utc,
        "expected_children": len(schedule),
        "lock_epoch": lock_epoch,
        "resolved_executable_paths": resolved_paths,
        "schema": RECORD_SCHEMAS["execution"],
        "stage": claim["stage"],
        "started_utc": started_utc,
    }
    name = claim["candidate_identity"] + "." + claim["stage"] + ".json"
    record = publish_record(plan, "executions", name, payload, held_locks)
    return record, raw_outputs, monotonic_start


def _stage_threshold_passed(stage, parsed, predecessor_reports, plan):
    if stage == STAGE_ORDER[0]:
        return parsed is not None and stage0_threshold_passed(parsed, plan)
    if parsed is None:
        return False
    aggregate = parsed["aggregate"]
    candidate_wins = _require_uint(aggregate["candidate_wins"])
    p0 = _parse_tuple(aggregate["candidate_p0"], 5)[0]
    p1 = _parse_tuple(aggregate["candidate_p1"], 5)[0]
    if stage == STAGE_ORDER[1]:
        return candidate_wins >= 38 and p0 >= 19 and p1 >= 19
    require(len(predecessor_reports) == 2, "final-predecessors")
    d20 = predecessor_reports[1]["payload"]["parsed"]["aggregate"]
    return (
        candidate_wins + _require_uint(d20["candidate_wins"]) >= 160
        and p0 + _parse_tuple(d20["candidate_p0"], 5)[0] >= 77
        and p1 + _parse_tuple(d20["candidate_p1"], 5)[0] >= 77
    )


def create_report(
    plan, head, lock_epoch, held_locks, claim_record, execution_record,
    raw_outputs, receipt_record, candidate_identity, candidate_payload,
    repository, predecessor_reports, stage_monotonic_start,
    execution_chain_clock_valid,
):
    claim = claim_record["payload"]
    execution = execution_record["payload"]
    stage = claim["stage"]
    postflight_start = time.monotonic()
    evidence_after = None
    evidence_after_sha = None
    postflight_error = None
    try:
        evidence_after = stable_evidence(
            plan, head, candidate_identity, candidate_payload, receipt_record, repository
        )
        evidence_after_sha = sha256_bytes(canonical_json(evidence_after))
    except BaseException as error:
        postflight_error = sanitized_error_record(error)
    parsed = None
    validation_errors = []
    threshold_errors = []
    try:
        require(execution["all_children_completed"], "report-incomplete-execution")
        require(len(raw_outputs) == execution["expected_children"], "report-output-prefix")
        if stage == STAGE_ORDER[0]:
            parsed = parse_stage0_outputs(raw_outputs, plan)
        else:
            require(len(raw_outputs) == 1, "game-output-count")
            parsed = parse_game_stdout(raw_outputs[0], plan, stage)
    except BaseException:
        validation_errors = ["report-validation-failure"]
    try:
        if not _stage_threshold_passed(stage, parsed, predecessor_reports, plan):
            threshold_errors = ["report-threshold-failure"]
    except BaseException:
        threshold_errors = ["report-threshold-failure"]
    child_process_ok = (
        execution["all_children_completed"]
        and len(execution["children"]) == execution["expected_children"]
        and all(
            child["returncode"] == 0 and child["os_error"] is None
            and child["timed_out"] is False and child["stdout_receipt"]["truncated"] is False
            and child["stderr_receipt"]["truncated"] is False
            and child["stderr_receipt"]["bytes"] == 0
            and child["postchild_process"] is not None
            and child["postchild_process"]["clean"] is True
            and child["postchild_process_error"] is None
            for child in execution["children"]
        )
    )
    stable = (
        evidence_after is not None and evidence_after == claim["evidence_before"]
        and evidence_after_sha == claim["evidence_before_sha256"]
    )
    postflight_utc = utc_now()
    postflight_elapsed = max(0.0, time.monotonic() - stage_monotonic_start)
    report_created_utc = utc_now()
    chronology_ok = (
        parse_timestamp(claim["claimed_utc"], True)
        <= parse_timestamp(execution["started_utc"], True)
        <= parse_timestamp(execution["ended_utc"], True)
        <= parse_timestamp(execution["created_utc"], True)
        <= parse_timestamp(postflight_utc, True)
        <= parse_timestamp(report_created_utc, True)
        <= parse_timestamp(CAMPAIGN_DEADLINE, False)
        and _replay_wall_matches(
            parse_timestamp(execution["started_utc"], True),
            parse_timestamp(postflight_utc, True), postflight_elapsed,
        )
        and postflight_elapsed
        <= min(
            _continuation_reserve_seconds(plan),
            execution["elapsed_monotonic_seconds"]
            + _postflight_reserve_seconds(plan),
        )
        and execution_chain_clock_valid
    )
    process_errors = [] if (
        child_process_ok and stable and postflight_error is None and chronology_ok
        and execution["elapsed_monotonic_seconds"]
        <= _stage_aggregate_seconds(plan)
        and execution["elapsed_monotonic_seconds"]
        <= postflight_elapsed <= _continuation_reserve_seconds(plan)
    ) else ["report-process-or-chronology-failure"]
    payload = {
        "acceptable": not process_errors and not validation_errors and not threshold_errors,
        "candidate_identity": candidate_identity,
        "claim": record_reference(plan, "claims", claim_record),
        "created_utc": report_created_utc,
        "evidence_after": evidence_after,
        "evidence_after_sha256": evidence_after_sha,
        "evidence_before": claim["evidence_before"],
        "evidence_before_sha256": claim["evidence_before_sha256"],
        "execution": record_reference(plan, "executions", execution_record),
        "lock_epoch": lock_epoch,
        "parsed": parsed,
        "postflight_elapsed_monotonic_seconds": postflight_elapsed,
        "postflight_error": postflight_error,
        "postflight_utc": postflight_utc,
        "predecessor_reports": claim["predecessor_reports"],
        "process_errors": process_errors,
        "schema": RECORD_SCHEMAS["report"],
        "stable": stable,
        "stage": stage,
        "threshold_errors": threshold_errors,
        "validation_errors": validation_errors,
    }
    raw = canonical_json(payload)
    return publish_record(plan, "reports", sha256_bytes(raw) + ".json", payload, held_locks)


def terminal_observation_from_samples(
    evidence, evidence_error, evidence_checked_utc, process, process_error,
    expected_evidence=None, require_continuation_reserve=False,
    latest_predecessor_utc=None,
):
    evidence_sha = (
        sha256_bytes(canonical_json(evidence)) if evidence is not None else None
    )
    process_outcome = process if process is not None else process_error
    process_checked = (
        parse_timestamp(process_outcome["checked_utc"], True)
        if process_outcome is not None else None
    )
    remaining_at_process = (
        (
            parse_timestamp(CAMPAIGN_DEADLINE, False) - process_checked
        ).total_seconds()
        if process_checked is not None else -1.0
    )
    observation_order_valid = (
        process_checked is not None
        and parse_timestamp(evidence_checked_utc, True) <= process_checked
        and (
            latest_predecessor_utc is None
            or parse_timestamp(latest_predecessor_utc, True)
            <= parse_timestamp(evidence_checked_utc, True)
        )
    )
    errors = [] if (
        evidence is not None and evidence_error is None and process is not None
        and process_error is None and process["clean"] and remaining_at_process >= 0
        and (expected_evidence is None or evidence == expected_evidence)
        and (
            not require_continuation_reserve
            or remaining_at_process >= _continuation_reserve_seconds(plan)
        )
        and observation_order_valid
    ) else ["terminal-observation-failure"]
    return {
        "schema": CAMPAIGN_SCHEMA_PREFIX + "terminal-observation-v1",
        "evidence_checked_utc": evidence_checked_utc,
        "stable_evidence": evidence,
        "stable_evidence_sha256": evidence_sha,
        "evidence_error": evidence_error,
        "process": process,
        "process_error": process_error,
        "errors": errors,
    }


def normal_terminal_observation(
    plan, head, candidate_identity, candidate_payload, receipt_record,
    repository, expected_evidence=None, require_continuation_reserve=False,
    latest_predecessor_utc=None,
):
    evidence = None
    evidence_error = None
    try:
        evidence = stable_evidence(
            plan, head, candidate_identity, candidate_payload,
            receipt_record, repository,
        )
    except BaseException as error:
        evidence_error = sanitized_error_record(error)
    checked = utc_now()
    process = None
    process_error = None
    try:
        process = process_snapshot(plan)
    except BaseException as error:
        process_error = _process_error(error)
    return terminal_observation_from_samples(
        evidence, evidence_error, checked, process, process_error,
        expected_evidence, require_continuation_reserve,
        latest_predecessor_utc,
    )


def synthetic_interrupted_decision_observation():
    return {
        "schema": CAMPAIGN_SCHEMA_PREFIX + "interrupted-decision-observation-v1",
        "evidence_checked_utc": None,
        "stable_evidence": None,
        "stable_evidence_sha256": None,
        "evidence_error": {
            "type": "InterruptedDecisionPublication",
            "message_sha256": "42de9e50cf3dfeb61edcd3164b74bd4cc9d666a841edab547a0e14c017e28422",
        },
        "process": None,
        "process_error": None,
        "errors": ["interrupted-decision-publication"],
    }


def decision_payload(
    plan, head, lock_epoch, registries, status, terminal_stage,
    candidate_identity, terminal_observation=None, interruption_error=None,
):
    require(len(registries["preexecution_claims"]) == 1 and len(registries["preexecution_receipts"]) == 1, "decision-preexecution-chain")
    preclaim = registries["preexecution_claims"][0]
    receipt = registries["preexecution_receipts"][0]
    claims_by_stage = {record["payload"]["stage"]: record for record in registries["claims"]}
    executions_by_stage = {record["payload"]["stage"]: record for record in registries["executions"]}
    reports_by_stage = {record["payload"]["stage"]: record for record in registries["reports"]}
    reached_claims = [record_reference(plan, "claims", claims_by_stage[stage]) for stage in STAGE_ORDER if stage in claims_by_stage]
    reached_executions = [record_reference(plan, "executions", executions_by_stage[stage]) for stage in STAGE_ORDER if stage in executions_by_stage]
    reached_reports = [record_reference(plan, "reports", reports_by_stage[stage]) for stage in STAGE_ORDER if stage in reports_by_stage]
    errors = list(receipt["payload"]["errors"])
    if interruption_error is not None:
        errors.append(interruption_error)
    for stage in STAGE_ORDER:
        if stage in reports_by_stage:
            report = reports_by_stage[stage]["payload"]
            errors.extend(report["process_errors"])
            errors.extend(report["validation_errors"])
            errors.extend(report["threshold_errors"])
    created = utc_now()
    chain_projection = validate_registry_chain(plan, registries)
    decision_within_deadline = (
        parse_timestamp(created, True)
        <= parse_timestamp(CAMPAIGN_DEADLINE, False)
    )
    chronology_valid = chain_projection["clock_valid"] and decision_within_deadline
    if (
        terminal_observation is not None
        and terminal_observation["schema"]
        == CAMPAIGN_SCHEMA_PREFIX + "terminal-observation-v1"
    ):
        latest_created = receipt["payload"]["created_utc"]
        if reports_by_stage:
            latest_stage = max(reports_by_stage, key=STAGE_ORDER.index)
            latest_created = reports_by_stage[latest_stage]["payload"]["created_utc"]
        process_outcome = (
            terminal_observation["process"]
            if terminal_observation["process"] is not None
            else terminal_observation["process_error"]
        )
        observation_clock_valid = _replay_clock_order(
            parse_timestamp(latest_created, True),
            parse_timestamp(terminal_observation["evidence_checked_utc"], True),
            parse_timestamp(process_outcome["checked_utc"], True),
            parse_timestamp(created, True),
        ) and decision_within_deadline
        continuation_reserve_required = False
        if receipt["payload"]["passed"] is True:
            if not claims_by_stage:
                continuation_reserve_required = True
            elif reports_by_stage:
                latest_report_stage = max(reports_by_stage, key=STAGE_ORDER.index)
                continuation_reserve_required = (
                    reports_by_stage[latest_report_stage]["payload"]["acceptable"] is True
                    and latest_report_stage != STAGE_ORDER[-1]
                    and len(claims_by_stage) == len(reports_by_stage)
                )
        if continuation_reserve_required:
            observation_clock_valid = observation_clock_valid and (
                parse_timestamp(CAMPAIGN_DEADLINE, False)
                - parse_timestamp(process_outcome["checked_utc"], True)
            ).total_seconds() >= _continuation_reserve_seconds(plan)
        terminal_observation = dict(terminal_observation)
        observation_valid = (
            terminal_observation["errors"] == []
            and observation_clock_valid
        )
        terminal_observation["errors"] = (
            [] if observation_valid else ["terminal-observation-failure"]
        )
        chronology_valid = chronology_valid and observation_clock_valid
    if terminal_observation is not None:
        errors.extend(terminal_observation["errors"])
    if not chronology_valid:
        status = "terminal-development-clock-or-deadline-rejection"
    selected = status == "development-selection-acceptable-pending-separate-source-activation-review"
    return {
        "arena_authorization": False,
        "candidate_identity": candidate_identity,
        "created_utc": created,
        "decision_chronology_valid": chronology_valid,
        "development_selection_acceptable": selected,
        "errors": errors,
        "fresh_bank_campaign_authorization": False,
        "heldout_qualification": False,
        "implementation_commit": head,
        "lock_epoch": lock_epoch,
        "plan": preclaim["payload"]["plan"],
        "preexecution_claim": record_reference(plan, "preexecution_claims", preclaim),
        "preexecution_receipt": record_reference(plan, "preexecution_receipts", receipt),
        "reached_claims": reached_claims,
        "reached_executions": reached_executions,
        "reached_reports": reached_reports,
        "registry_cardinalities": {
            "preexecution_claims": len(registries["preexecution_claims"]),
            "preexecution_receipts": len(registries["preexecution_receipts"]),
            "claims": len(registries["claims"]),
            "executions": len(registries["executions"]),
            "reports": len(registries["reports"]),
            "decisions": 1,
        },
        "retry_authorized": False,
        "schema": RECORD_SCHEMAS["decision"],
        "source_activation_authorization": False,
        "status": status,
        "terminal_stage": terminal_stage,
        "terminal_observation": terminal_observation,
        "upload_authorization": False,
    }


def publish_decision(
    plan, head, lock_epoch, held_locks, registries, status, terminal_stage,
    candidate_identity, terminal_observation=None, interruption_error=None,
):
    if terminal_observation is None:
        decision_process = process_snapshot(plan)
        require(decision_process["clean"] is True, "decision-process-conflict")
    else:
        if (
            terminal_observation["schema"]
            == CAMPAIGN_SCHEMA_PREFIX + "interrupted-decision-observation-v1"
        ):
            require(
                terminal_observation == synthetic_interrupted_decision_observation(),
                "decision-synthetic-observation",
            )
        else:
            require(
                (terminal_observation["process"] is None)
                != (terminal_observation["process_error"] is None),
                "decision-observation-process-outcome",
            )
    current = scan_all_registries(plan)
    for registry_name in REGISTRY_ORDER:
        require(
            [record["sha256"] for record in current[registry_name]]
            == [record["sha256"] for record in registries[registry_name]],
            "decision-registry-drift",
        )
        require(
            not current[registry_name + "_pending"],
            "decision-pending-registry",
        )
    registries = current
    require(not registries["decisions"], "decision-already-present")
    payload = decision_payload(
        plan, head, lock_epoch, registries, status, terminal_stage,
        candidate_identity, terminal_observation, interruption_error,
    )
    raw = canonical_json(payload)
    record = publish_record(plan, "decisions", sha256_bytes(raw) + ".json", payload, held_locks)
    return record


def synthetic_interrupted_receipt(plan, lock_epoch, claim_record):
    claim = claim_record["payload"]
    payload = {
        "aggregate_timeout_seconds": claim["aggregate_timeout_seconds"],
        "binary_records": None,
        "build_roots": [{
            "path": BUILD_ROOT_REL, "mode": None, "fresh": None, "tmp_mode": None,
            "tmp_empty": None, "entries_count": None, "entries_sha256": None,
        }],
        "child_timeout_seconds": claim["child_timeout_seconds"],
        "claim": record_reference(plan, "preexecution_claims", claim_record),
        "claim_sha256": claim_record["sha256"],
        "commands": [], "compiler_records": None,
        "created_utc": utc_now(), "dependency_closure": None,
        "elapsed_monotonic_seconds": None, "ended_utc": None,
        "errors": ["interrupted-preexecution-after-durable-claim"],
        "host_observation": None, "lock_epoch": lock_epoch, "passed": False,
        "schema": RECORD_SCHEMAS["preexecution_receipt"], "started_utc": None,
        "toolchain": None,
    }
    return payload


def _derive_persisted_candidate_identity(plan, registries, expected_head=None):
    preclaims = registries["preexecution_claims"]
    if expected_head is not None and preclaims:
        require(
            len(preclaims) == 1
            and preclaims[0]["payload"]["implementation_commit"] == expected_head,
            "candidate-implementation-head",
        )
    receipts = registries["preexecution_receipts"]
    if not receipts or receipts[0]["payload"]["passed"] is not True:
        return None
    require(len(preclaims) == 1 and len(receipts) == 1, "candidate-preexecution-cardinality")
    implementation_commit = preclaims[0]["payload"]["implementation_commit"]
    binding = load_prior_binding(plan)
    sources = sources_projection_from_plan(plan)
    candidate_identity, _payload = candidate_identity_from_receipt(
        plan, implementation_commit, receipts[0], binding, sources
    )
    return candidate_identity


def terminal_result(plan, registries):
    candidate_identity = _derive_persisted_candidate_identity(plan, registries)
    validate_registry_chain(
        plan, registries, expected_candidate_identity=candidate_identity
    )
    require(len(registries["decisions"]) == 1, "terminal-decision-count")
    record = registries["decisions"][0]
    decision = record["payload"]
    selected = decision["development_selection_acceptable"]
    require(isinstance(selected, bool), "terminal-selection")
    path = plan["execution_policy"]["registry_paths"]["decisions"] + "/" + record["name"]
    output = (
        path + "\n" + "sha256=" + record["sha256"] + "\n"
        + "development_selection_acceptable=" + ("true" if selected else "false") + "\n"
    ).encode("ascii")
    return output, 0 if selected else 1


def _pending_record_as_final(plan, registry_name, pending):
    require(pending["complete"] is True, "pending-record-incomplete")
    return {
        "name": pending["name"],
        "path": os.path.join(
            ROOT, plan["execution_policy"]["registry_paths"][registry_name],
            pending["name"],
        ),
        "payload": pending["payload"],
        "raw": pending["raw"],
        "sha256": pending["sha256"],
        "mode": "0444",
    }


def _pending_decision_promotable(pending):
    require(
        pending["complete"] is True and isinstance(pending["payload"], dict),
        "pending-decision-complete",
    )
    created = parse_timestamp(pending["payload"]["created_utc"], True)
    return (
        created <= parse_timestamp(CAMPAIGN_DEADLINE, False)
        or pending["payload"]["status"]
        == "terminal-development-clock-or-deadline-rejection"
    )


def _prevalidate_pending_recovery(plan, head, registries):
    projected = {
        key: list(value) for key, value in registries.items()
    }
    for registry_name in REGISTRY_ORDER:
        final_names = {record["name"] for record in projected[registry_name]}
        for pending in projected[registry_name + "_pending"]:
            if pending["name"] in final_names or not pending["complete"]:
                continue
            # A decision that never acquired its final link cannot establish
            # acceptance after the campaign deadline.  Recovery will replace
            # it with the deterministic interruption/clock decision.
            if (
                registry_name == "decisions"
                and not _pending_decision_promotable(pending)
            ):
                continue
            projected[registry_name].append(
                _pending_record_as_final(plan, registry_name, pending)
            )
    candidate_identity = _derive_persisted_candidate_identity(
        plan, projected, expected_head=head
    )
    validate_registry_chain(
        plan, projected, allow_pending=True,
        expected_candidate_identity=candidate_identity,
    )
    for pending in projected["decisions_pending"]:
        if not pending["complete"] or _pending_decision_promotable(pending):
            continue
        decision = pending["payload"]
        require(
            not projected["decisions"]
            and len(projected["preexecution_claims"]) == 1
            and len(projected["preexecution_receipts"]) == 1,
            "discarded-decision-prefix",
        )
        require(
            decision["preexecution_claim"] == record_reference(
                plan, "preexecution_claims", projected["preexecution_claims"][0]
            )
            and decision["preexecution_receipt"] == record_reference(
                plan, "preexecution_receipts", projected["preexecution_receipts"][0]
            ),
            "discarded-decision-preexecution-refs",
        )
        require(
            decision["reached_claims"] == [
                record_reference(
                    plan, "claims",
                    next(record for record in projected["claims"] if record["payload"]["stage"] == stage),
                )
                for stage in STAGE_ORDER
                if any(record["payload"]["stage"] == stage for record in projected["claims"])
            ]
            and decision["reached_executions"] == [
                record_reference(
                    plan, "executions",
                    next(record for record in projected["executions"] if record["payload"]["stage"] == stage),
                )
                for stage in STAGE_ORDER
                if any(record["payload"]["stage"] == stage for record in projected["executions"])
            ]
            and decision["reached_reports"] == [
                record_reference(
                    plan, "reports",
                    next(record for record in projected["reports"] if record["payload"]["stage"] == stage),
                )
                for stage in STAGE_ORDER
                if any(record["payload"]["stage"] == stage for record in projected["reports"])
            ],
            "discarded-decision-stage-refs",
        )
        latest_stage = (
            max(
                (record["payload"]["stage"] for record in projected["claims"]),
                key=STAGE_ORDER.index,
            )
            if projected["claims"] else "preexecution"
        )
        require(
            decision["candidate_identity"] == candidate_identity
            and decision["terminal_stage"] == latest_stage
            and decision["implementation_commit"] == head
            and decision["plan"] == plan_reference(),
            "discarded-decision-routing",
        )
        success_status = (
            "development-selection-acceptable-pending-separate-source-activation-review"
        )
        require(
            decision["status"] in (
                "terminal-development-preexecution-rejection",
                "terminal-development-interrupted-execution",
                "terminal-development-interrupted-postflight",
                "terminal-development-clock-or-deadline-rejection",
                "terminal-development-rejection",
                success_status,
            )
            and decision["development_selection_acceptable"]
            is (decision["status"] == success_status),
            "discarded-decision-original-selection",
        )
        normalized_payload = dict(decision)
        normalized_payload.update({
            "status": "terminal-development-clock-or-deadline-rejection",
            "decision_chronology_valid": False,
            "development_selection_acceptable": False,
        })
        normalized_raw = canonical_json(normalized_payload)
        normalized_sha = sha256_bytes(normalized_raw)
        normalized_projection = {
            key: list(value) for key, value in projected.items()
        }
        normalized_projection["decisions"] = [{
            "name": normalized_sha + ".json",
            "path": os.path.join(
                ROOT, plan["execution_policy"]["registry_paths"]["decisions"],
                normalized_sha + ".json",
            ),
            "payload": normalized_payload,
            "raw": normalized_raw,
            "sha256": normalized_sha,
            "mode": "0444",
        }]
        validate_registry_chain(
            plan, normalized_projection, allow_pending=True,
            expected_candidate_identity=candidate_identity,
        )
    incomplete = [
        (registry_name, pending)
        for registry_name in REGISTRY_ORDER
        for pending in projected[registry_name + "_pending"]
        if not pending["complete"]
    ]
    require(len(incomplete) <= 1, "incomplete-pending-cardinality")
    if not incomplete:
        return
    registry_name, pending = incomplete[0]
    finals_present = any(projected[name] for name in REGISTRY_ORDER)
    if registry_name == "preexecution_claims":
        require(
            not finals_present and pending["name"] == head + ".json",
            "incomplete-preclaim-attribution",
        )
    elif registry_name == "preexecution_receipts":
        require(
            len(projected["preexecution_claims"]) == 1
            and projected["preexecution_claims"][0]["payload"]["implementation_commit"] == head
            and not projected["preexecution_receipts"]
            and not any(projected[name] for name in ("claims", "executions", "reports", "decisions")),
            "incomplete-receipt-attribution",
        )
    elif registry_name == "claims":
        require(
            candidate_identity is not None and not projected["decisions"]
            and len(projected["claims"]) == len(projected["reports"])
            and len(projected["reports"]) < len(STAGE_ORDER)
            and all(
                record["payload"]["acceptable"] is True
                for record in projected["reports"]
            ),
            "incomplete-claim-prefix",
        )
        next_stage = STAGE_ORDER[len(projected["reports"])]
        require(
            pending["name"] == candidate_identity + "." + next_stage + ".json",
            "incomplete-claim-attribution",
        )
    elif registry_name == "executions":
        require(
            candidate_identity is not None and not projected["decisions"]
            and len(projected["claims"]) == len(projected["executions"]) + 1,
            "incomplete-execution-prefix",
        )
        latest_claim = max(
            projected["claims"],
            key=lambda record: STAGE_ORDER.index(record["payload"]["stage"]),
        )
        require(
            pending["name"] == candidate_identity + "." + latest_claim["payload"]["stage"] + ".json",
            "incomplete-execution-attribution",
        )
    elif registry_name == "reports":
        require(
            candidate_identity is not None and not projected["decisions"]
            and len(projected["executions"]) == len(projected["reports"]) + 1,
            "incomplete-report-attribution",
        )
    else:
        require(
            registry_name == "decisions" and finals_present
            and len(projected["preexecution_claims"]) == 1
            and len(projected["preexecution_receipts"]) == 1
            and not projected["decisions"],
            "incomplete-decision-attribution",
        )


def _unlink_bound_pending(pending_fd, name, pending, held_locks):
    try:
        descriptor = os.open(name, _regular_flags(), dir_fd=pending_fd)
    except OSError as error:
        raise ContractError("pending-discard-open") from error
    try:
        opened = os.fstat(descriptor)
        rebound = os.stat(name, dir_fd=pending_fd, follow_symlinks=False)
        require(
            stat.S_ISREG(opened.st_mode)
            and mode_string(opened) == pending["mode"]
            and opened.st_nlink == pending["nlink"]
            and opened.st_size == len(pending["raw"])
            and _same_node(opened, rebound),
            "pending-discard-binding",
        )
        os.lseek(descriptor, 0, os.SEEK_SET)
        reread = b""
        while len(reread) < len(pending["raw"]):
            chunk = os.read(descriptor, len(pending["raw"]) - len(reread))
            require(chunk, "pending-discard-short-read")
            reread += chunk
        require(
            reread == pending["raw"] and os.read(descriptor, 1) == b"",
            "pending-discard-reread",
        )
        validate_campaign_locks(held_locks)
        os.unlink(name, dir_fd=pending_fd)
        os.fsync(pending_fd)
        require(os.fstat(descriptor).st_nlink == 0, "pending-discard-unlinked")
    finally:
        os.close(descriptor)


def recover_pending_publications(plan, head, registries, held_locks):
    validate_campaign_locks(held_locks)
    _prevalidate_pending_recovery(plan, head, registries)
    incomplete_decision = False
    for registry_name in REGISTRY_ORDER:
        registry_fd, registry_path = _open_registry_directory(registry_name)
        pending_path = os.path.join(registry_path, ".pending")
        try:
            pending_fd = _open_directory_component(
                registry_fd, ".pending", "0755"
            )
        except BaseException:
            os.close(registry_fd)
            raise
        try:
            finals = {record["name"]: record for record in registries[registry_name]}
            for pending in registries[registry_name + "_pending"]:
                validate_campaign_locks(held_locks)
                name = pending["name"]
                final = finals.get(name)
                if final is not None:
                    require(
                        pending["complete"] and pending["raw"] == final["raw"],
                        "pending-final-bytes",
                    )
                    pending_meta = os.stat(
                        name, dir_fd=pending_fd, follow_symlinks=False
                    )
                    final_meta = os.stat(
                        name, dir_fd=registry_fd, follow_symlinks=False
                    )
                    require(
                        pending_meta.st_nlink == 2 and final_meta.st_nlink == 2
                        and _same_node(pending_meta, final_meta),
                        "pending-final-inode",
                    )
                    try:
                        record_fd = os.open(
                            name, _regular_flags(), dir_fd=registry_fd
                        )
                    except OSError as error:
                        raise ContractError("pending-final-open") from error
                    try:
                        opened = os.fstat(record_fd)
                        require(
                            stat.S_ISREG(opened.st_mode)
                            and mode_string(opened) == "0444"
                            and opened.st_nlink == 2
                            and _same_node(opened, final_meta),
                            "pending-final-open",
                        )
                        os.fsync(record_fd)
                        os.fsync(registry_fd)
                        validate_campaign_locks(held_locks)
                        os.unlink(name, dir_fd=pending_fd)
                        os.fsync(pending_fd)
                        cleaned_fd = _reread_open_regular(
                            record_fd, final["raw"], "0444", 1, opened,
                        )
                        final_raw, final_metadata = _read_regular_at(
                            registry_fd, name, "0444", 1,
                            record_cap(plan, registry_name), registry_path,
                        )
                        require(
                            final_raw == final["raw"]
                            and final_metadata.st_nlink == 1
                            and _same_node(cleaned_fd, final_metadata),
                            "pending-final-cleanup",
                        )
                    finally:
                        os.close(record_fd)
                    validate_campaign_locks(held_locks)
                    continue
                if not pending["complete"]:
                    _unlink_bound_pending(
                        pending_fd, name, pending, held_locks,
                    )
                    if registry_name == "decisions":
                        incomplete_decision = True
                    validate_campaign_locks(held_locks)
                    continue
                if (
                    registry_name == "decisions"
                    and not _pending_decision_promotable(pending)
                ):
                    _unlink_bound_pending(
                        pending_fd, name, pending, held_locks,
                    )
                    validate_campaign_locks(held_locks)
                    continue
                if registry_name in ("preexecution_claims", "claims"):
                    _unlink_bound_pending(
                        pending_fd, name, pending, held_locks,
                    )
                    validate_campaign_locks(held_locks)
                    continue
                try:
                    descriptor = os.open(
                        name,
                        _regular_flags(
                            os.O_RDWR if pending["mode"] == "0600" else os.O_RDONLY
                        ),
                        dir_fd=pending_fd,
                    )
                except OSError as error:
                    raise ContractError("pending-promote-open") from error
                try:
                    opened = os.fstat(descriptor)
                    pending_meta = os.stat(
                        name, dir_fd=pending_fd, follow_symlinks=False
                    )
                    require(
                        stat.S_ISREG(opened.st_mode) and _same_node(opened, pending_meta)
                        and opened.st_nlink == 1 and opened.st_size == len(pending["raw"])
                        and mode_string(opened) == pending["mode"],
                        "pending-promote-open",
                    )
                    if pending["mode"] == "0600":
                        os.fchmod(descriptor, 0o444)
                    os.fsync(descriptor)
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    reread = b""
                    while len(reread) < len(pending["raw"]):
                        chunk = os.read(
                            descriptor, len(pending["raw"]) - len(reread)
                        )
                        require(chunk, "pending-promote-short-read")
                        reread += chunk
                    require(
                        reread == pending["raw"] and os.read(descriptor, 1) == b"",
                        "pending-promote-reread",
                    )
                    reread_meta = os.fstat(descriptor)
                    pending_meta = os.stat(
                        name, dir_fd=pending_fd, follow_symlinks=False
                    )
                    require(
                        mode_string(reread_meta) == "0444"
                        and reread_meta.st_nlink == 1
                        and reread_meta.st_size == len(reread)
                        and _same_node(opened, reread_meta)
                        and _metadata_identity(reread_meta)
                        == _metadata_identity(pending_meta),
                        "pending-promote-mode",
                    )
                    validate_campaign_locks(held_locks)
                    try:
                        os.link(
                            name, name, src_dir_fd=pending_fd,
                            dst_dir_fd=registry_fd, follow_symlinks=False,
                        )
                    except OSError as error:
                        raise ContractError("pending-promote-link") from error
                    linked_fd = os.fstat(descriptor)
                    linked_pending = os.stat(
                        name, dir_fd=pending_fd, follow_symlinks=False
                    )
                    linked_final = os.stat(
                        name, dir_fd=registry_fd, follow_symlinks=False
                    )
                    require(
                        linked_fd.st_nlink == 2 and linked_pending.st_nlink == 2
                        and linked_final.st_nlink == 2
                        and _same_node(linked_fd, linked_pending)
                        and _same_node(linked_pending, linked_final),
                        "pending-promote-link-identity",
                    )
                    validate_campaign_locks(held_locks)
                    os.fsync(descriptor)
                    os.fsync(registry_fd)
                    validate_campaign_locks(held_locks)
                    os.unlink(name, dir_fd=pending_fd)
                    os.fsync(pending_fd)
                    final_raw, final_meta = _read_regular_at(
                        registry_fd, name, "0444", 1,
                        record_cap(plan, registry_name), registry_path,
                    )
                    final_fd = _reread_open_regular(
                        descriptor, pending["raw"], "0444", 1, linked_fd,
                    )
                    require(
                        final_raw == pending["raw"] and final_meta.st_nlink == 1
                        and final_fd.st_nlink == 1
                        and final_meta.st_size == len(pending["raw"])
                        and _same_node(linked_fd, final_fd)
                        and _same_node(final_fd, final_meta),
                        "pending-promote-final",
                    )
                finally:
                    os.close(descriptor)
                validate_campaign_locks(held_locks)
            _verify_absolute_directory_binding(registry_path, registry_fd)
            _verify_absolute_directory_binding(pending_path, pending_fd)
        finally:
            os.close(pending_fd)
            os.close(registry_fd)
    validate_campaign_locks(held_locks)
    return incomplete_decision


def _publish_recovery_decision(
    plan, head, lock_epoch, held_locks, registries, candidate_identity,
    status, stage, error=None, observation=None,
):
    publish_decision(
        plan, head, lock_epoch, held_locks, registries, status, stage,
        candidate_identity, observation, error,
    )
    final = scan_all_registries(plan)
    validate_registry_chain(plan, final)
    return terminal_result(plan, final)


def recover_or_continue(
    plan, head, lock_epoch, held_locks, registries, binding,
    repository, sources, incomplete_decision=False,
):
    validate_registry_chain(plan, registries)
    if registries["decisions"]:
        return terminal_result(plan, registries)
    preclaims = registries["preexecution_claims"]
    receipts = registries["preexecution_receipts"]
    if preclaims and not receipts:
        payload = synthetic_interrupted_receipt(plan, lock_epoch, preclaims[0])
        raw = canonical_json(payload)
        publish_record(plan, "preexecution_receipts", sha256_bytes(raw) + ".json", payload, held_locks)
        registries = scan_all_registries(plan)
        return _publish_recovery_decision(
            plan, head, lock_epoch, held_locks, registries, None,
            "terminal-development-preexecution-rejection", "preexecution",
        )
    if not receipts:
        raise ContractError("recovery-without-preexecution")
    receipt_record = receipts[0]
    receipt = receipt_record["payload"]
    if receipt["passed"] is False:
        return _publish_recovery_decision(
            plan, head, lock_epoch, held_locks, registries, None,
            "terminal-development-preexecution-rejection", "preexecution",
        )
    candidate_identity, _candidate_payload = candidate_identity_from_receipt(
        plan, head, receipt_record, binding, sources
    )
    if incomplete_decision:
        reached = (
            [record["payload"]["stage"] for record in registries["reports"]]
            or [record["payload"]["stage"] for record in registries["executions"]]
            or [record["payload"]["stage"] for record in registries["claims"]]
        )
        terminal = reached[-1] if reached else "preexecution"
        return _publish_recovery_decision(
            plan, head, lock_epoch, held_locks, registries, candidate_identity,
            "terminal-development-rejection", terminal,
            observation=synthetic_interrupted_decision_observation(),
        )
    claims = {record["payload"]["stage"]: record for record in registries["claims"]}
    executions = {record["payload"]["stage"]: record for record in registries["executions"]}
    reports = {record["payload"]["stage"]: record for record in registries["reports"]}
    if not claims:
        return _publish_recovery_decision(
            plan, head, lock_epoch, held_locks, registries, candidate_identity,
            "terminal-development-rejection", "preexecution",
            "interrupted-after-passed-preexecution-before-stage0",
        )
    for index, stage in enumerate(STAGE_ORDER):
        if stage not in claims:
            previous = STAGE_ORDER[index - 1]
            error = {
                STAGE_ORDER[0]: "interrupted-after-passed-preexecution-before-stage0",
                STAGE_ORDER[1]: "interrupted-after-accepted-stage0-before-d20",
                STAGE_ORDER[2]: "interrupted-after-accepted-d20-before-remainder",
            }[stage]
            terminal = "preexecution" if index == 0 else previous
            return _publish_recovery_decision(
                plan, head, lock_epoch, held_locks, registries, candidate_identity,
                "terminal-development-rejection", terminal, error,
            )
        if stage not in executions:
            return _publish_recovery_decision(
                plan, head, lock_epoch, held_locks, registries, candidate_identity,
                "terminal-development-interrupted-execution", stage,
                "interrupted-execution-after-durable-claim:" + stage,
            )
        if stage not in reports:
            return _publish_recovery_decision(
                plan, head, lock_epoch, held_locks, registries, candidate_identity,
                "terminal-development-interrupted-postflight", stage,
                "interrupted-postflight-after-durable-execution:" + stage,
            )
        if reports[stage]["payload"]["acceptable"] is False:
            return _publish_recovery_decision(
                plan, head, lock_epoch, held_locks, registries, candidate_identity,
                "terminal-development-rejection", stage,
            )
    return _publish_recovery_decision(
        plan, head, lock_epoch, held_locks, registries, candidate_identity,
        "terminal-development-rejection", STAGE_ORDER[-1],
        "interrupted-after-accepted-final-before-decision",
    )


def run_fresh_campaign(
    plan, head, lock_epoch, held_locks, binding, repository, external, sources,
):
    global _ACTIVE_HELPER_MONOTONIC_DEADLINE
    claim_record = publish_preexecution_claim(plan, head, lock_epoch, held_locks)
    receipt_record = execute_preexecution(
        plan, head, lock_epoch, claim_record, held_locks, binding, repository, external
    )
    registries = scan_all_registries(plan)
    validate_registry_chain(plan, registries)
    preclaim_replay = _replay_preexecution_claim(
        plan, registries["preexecution_claims"][0]
    )
    receipt_replay = _replay_preexecution_receipt(
        plan, registries["preexecution_receipts"][0],
        registries["preexecution_claims"][0], preclaim_replay,
    )
    if receipt_record["payload"]["passed"] is False:
        publish_decision(
            plan, head, lock_epoch, held_locks, registries,
            (
                "terminal-development-preexecution-rejection"
                if preclaim_replay["clock_valid"] and receipt_replay["clock_valid"]
                else "terminal-development-clock-or-deadline-rejection"
            ),
            "preexecution", None,
        )
        return terminal_result(plan, scan_all_registries(plan))
    candidate_identity, candidate_payload = candidate_identity_from_receipt(
        plan, head, receipt_record, binding, sources
    )
    if not (preclaim_replay["clock_valid"] and receipt_replay["clock_valid"]):
        observation = normal_terminal_observation(
            plan, head, candidate_identity, candidate_payload,
            receipt_record, repository, None, True,
            receipt_record["payload"]["created_utc"],
        )
        publish_decision(
            plan, head, lock_epoch, held_locks, registries,
            "terminal-development-clock-or-deadline-rejection",
            "preexecution", candidate_identity, observation,
        )
        return terminal_result(plan, scan_all_registries(plan))
    evidence = None
    predecessor_reports = []
    for stage in STAGE_ORDER:
        # The complete durable prefix is reloaded and replayed before the one
        # live observation that either authorizes this claim or is retained in
        # the terminal rejection.  A restored second sample cannot launder a
        # failure from the first.
        registries = scan_all_registries(plan)
        execution_projection = validate_registry_chain(
            plan, registries,
            expected_candidate_identity=candidate_identity,
        )
        predecessor_reports = [
            next(
                record for record in registries["reports"]
                if record["payload"]["stage"] == reached_stage
            )
            for reached_stage in STAGE_ORDER[: STAGE_ORDER.index(stage)]
        ]
        current_evidence = None
        evidence_error = None
        try:
            current_evidence = stable_evidence(
                plan, head, candidate_identity, candidate_payload,
                receipt_record, repository,
            )
        except BaseException as error:
            evidence_error = sanitized_error_record(error)
        evidence_checked = utc_now()
        preclaim_process = None
        preclaim_process_error = None
        try:
            preclaim_process = process_snapshot(plan)
        except BaseException as error:
            preclaim_process_error = _process_error(error)
        continuation_failed = (
            current_evidence is None or evidence_error is not None
            or (evidence is not None and current_evidence != evidence)
            or preclaim_process is None or preclaim_process_error is not None
            or (preclaim_process is not None and not preclaim_process["clean"])
            or (
                preclaim_process is not None
                and (
                    parse_timestamp(CAMPAIGN_DEADLINE, False)
                    - parse_timestamp(preclaim_process["checked_utc"], True)
                ).total_seconds() < _continuation_reserve_seconds(plan)
            )
        )
        if continuation_failed:
            observation = terminal_observation_from_samples(
                current_evidence, evidence_error, evidence_checked,
                preclaim_process, preclaim_process_error, evidence, True,
                (
                    predecessor_reports[-1]["payload"]["created_utc"]
                    if predecessor_reports
                    else receipt_record["payload"]["created_utc"]
                ),
            )
            terminal_stage = (
                predecessor_reports[-1]["payload"]["stage"]
                if predecessor_reports else "preexecution"
            )
            publish_decision(
                plan, head, lock_epoch, held_locks, registries,
                "terminal-development-rejection", terminal_stage,
                candidate_identity, observation,
            )
            return terminal_result(plan, scan_all_registries(plan))
        evidence = current_evidence
        claim = publish_stage_claim(
            plan, head, lock_epoch, held_locks, stage, candidate_identity,
            candidate_payload, receipt_record, predecessor_reports, evidence,
            preclaim_process,
        )
        registries = scan_all_registries(plan)
        claim_projection = validate_registry_chain(
            plan, registries,
            expected_candidate_identity=candidate_identity,
        )
        claim = next(
            record for record in registries["claims"]
            if record["payload"]["stage"] == stage
        )
        if not claim_projection["clock_valid"]:
            publish_decision(
                plan, head, lock_epoch, held_locks, registries,
                "terminal-development-clock-or-deadline-rejection",
                stage, candidate_identity,
            )
            return terminal_result(plan, scan_all_registries(plan))
        execution, raw_outputs, stage_monotonic_start = execute_stage(
            plan, lock_epoch, held_locks, claim, receipt_record, repository,
        )
        registries = scan_all_registries(plan)
        validate_registry_chain(
            plan, registries,
            expected_candidate_identity=candidate_identity,
        )
        claim = next(
            record for record in registries["claims"]
            if record["payload"]["stage"] == stage
        )
        execution = next(
            record for record in registries["executions"]
            if record["payload"]["stage"] == stage
        )
        report = create_report(
            plan, head, lock_epoch, held_locks, claim, execution, raw_outputs,
            receipt_record, candidate_identity, candidate_payload, repository,
            predecessor_reports, stage_monotonic_start,
            execution_projection["clock_valid"],
        )
        registries = scan_all_registries(plan)
        report_projection = validate_registry_chain(
            plan, registries,
            expected_candidate_identity=candidate_identity,
        )
        report = next(
            record for record in registries["reports"]
            if record["payload"]["stage"] == stage
        )
        _ACTIVE_HELPER_MONOTONIC_DEADLINE = None
        predecessor_reports = [
            next(
                record for record in registries["reports"]
                if record["payload"]["stage"] == reached_stage
            )
            for reached_stage in STAGE_ORDER[: STAGE_ORDER.index(stage) + 1]
        ]
        if report["payload"]["acceptable"] is False:
            publish_decision(
                plan, head, lock_epoch, held_locks, registries,
                "terminal-development-clock-or-deadline-rejection"
                if not report_projection["clock_valid"]
                else "terminal-development-rejection",
                stage, candidate_identity,
            )
            return terminal_result(plan, scan_all_registries(plan))
        evidence = report["payload"]["evidence_after"]
    observation = normal_terminal_observation(
        plan, head, candidate_identity, candidate_payload, receipt_record,
        repository, evidence, False,
        predecessor_reports[-1]["payload"]["created_utc"],
    )
    status = (
        "development-selection-acceptable-pending-separate-source-activation-review"
        if observation["errors"] == []
        else "terminal-development-rejection"
    )
    registries = scan_all_registries(plan)
    publish_decision(
        plan, head, lock_epoch, held_locks, registries, status,
        STAGE_ORDER[-1], candidate_identity, observation,
    )
    return terminal_result(plan, scan_all_registries(plan))


def run_command(plan, head):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    require(resource.getrlimit(resource.RLIMIT_CORE) == (0, 0), "core-limit")
    lock_epoch = create_lock_epoch()
    previous_umask = os.umask(0o022)
    held_locks = []
    try:
        held_locks = acquire_campaign_locks(plan)
        entry_process = process_snapshot(plan)
        require(entry_process["clean"] is True, "run-entry-process-conflict")
        prepare_registry_topology()
        registries = scan_all_registries(plan)
        incomplete_decision = recover_pending_publications(
            plan, head, registries, held_locks
        )
        registries = scan_all_registries(plan)
        require(all(not registries[name + "_pending"] for name in REGISTRY_ORDER), "pending-recovery-required")
        if any(registries[name] for name in REGISTRY_ORDER):
            binding = load_prior_binding(plan) if (
                registries["preexecution_receipts"]
                and registries["preexecution_receipts"][0]["payload"]["passed"] is True
            ) else None
            repository = None
            sources = sources_projection_from_plan(plan)
            output, returncode = recover_or_continue(
                plan, head, lock_epoch, held_locks, registries,
                binding, repository, sources, incomplete_decision,
            )
        else:
            binding = load_prior_binding(plan)
            repository, external = source_and_external_closures(plan, head, binding, True)
            sources = validate_source_projection(plan)
            output, returncode = run_fresh_campaign(
                plan, head, lock_epoch, held_locks, binding,
                repository, external, sources,
            )
        validate_python_runtime_manifest()
        return output, returncode
    finally:
        if held_locks:
            release_campaign_locks(held_locks)
        os.umask(previous_umask)


def _registry_topology_presence():
    presence = []
    for name in REGISTRY_ORDER:
        descriptor, _path = _open_registry_directory(name, missing_ok=True)
        presence.append(descriptor is not None)
        if descriptor is not None:
            os.close(descriptor)
    return presence


def _audit_topology_exists():
    return all(_registry_topology_presence())


def audit_command(plan, head):
    # Audit is deliberately read-only: no lock acquisition, mkdir, fsync or repair.
    audit_process = process_snapshot(plan)
    require(audit_process["clean"] is True, "audit-process-conflict")
    topology_presence = _registry_topology_presence()
    if all(topology_presence):
        registries = scan_all_registries(plan)
    else:
        require(not any(topology_presence), "audit-partial-topology")
        registries = {name: [] for name in REGISTRY_ORDER}
        registries.update({name + "_pending": [] for name in REGISTRY_ORDER})
    candidate_identity = _derive_persisted_candidate_identity(
        plan, registries, expected_head=head
    )
    projection = validate_registry_chain(
        plan, registries, expected_candidate_identity=candidate_identity
    )
    decision_reference = None
    selected = None
    if registries["decisions"]:
        decision = registries["decisions"][0]
        decision_reference = {
            "path": plan["execution_policy"]["registry_paths"]["decisions"] + "/" + decision["name"],
            "bytes": len(decision["raw"]), "sha256": decision["sha256"],
            "schema": decision["payload"]["schema"],
        }
        selected = decision["payload"]["development_selection_acceptable"]
    represented = projection["claims"] + projection["executions"] + projection["reports"]
    if registries["decisions"]:
        state = "terminal"
    elif represented:
        state = max(set(represented), key=STAGE_ORDER.index)
    elif registries["preexecution_claims"] or registries["preexecution_receipts"]:
        state = "preexecution"
    else:
        state = "virgin"
    if state == STAGE_ORDER[0]:
        state = "stage0"
    payload = {
        "candidate_identity": candidate_identity,
        "claims": projection["claims"],
        "decision": decision_reference,
        "decision_count": len(registries["decisions"]),
        "development_selection_acceptable": selected,
        "errors": [],
        "executions": projection["executions"],
        "preexecution_claim_count": len(registries["preexecution_claims"]),
        "preexecution_receipt_count": len(registries["preexecution_receipts"]),
        "protected_bank_files_accessed": [],
        "reports": projection["reports"],
        "schema": CAMPAIGN_SCHEMA_PREFIX + "audit-v1",
        "state": state,
        "valid": True,
    }
    return canonical_json(payload), 0 if state == "terminal" else 1


def main(argv=None):
    try:
        validate_outer_environment()
        arguments = sys.argv[1:] if argv is None else list(argv)
        if len(arguments) != 1 or arguments[0] not in ("audit", "run"):
            os.write(2, b"usage-error\n")
            return 2
        validate_python_runtime_manifest()
        head, _self_identity = bootstrap_self_validation()
        plan, _plan_identity = load_plan()
        validate_implementation_head(plan, head)
        if arguments[0] == "audit":
            output, returncode = audit_command(plan, head)
        else:
            output, returncode = run_command(plan, head)
        os.write(1, output)
        return returncode
    except BaseException as error:
        os.write(2, (sanitized_reason(error) + "\n").encode("ascii"))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
