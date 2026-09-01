#!/usr/bin/env python3
"""Freeze and run the standalone compact value-BFM campaign.

``freeze-inputs`` is the only command in this workflow that may inspect the
accepted large-teacher campaign.  It follows fixed metadata routes, rejects
protected path markers before touching a path, and atomically creates a
self-contained bundle.  Every other command accepts only that copied bundle
and therefore has no Git or old-worktree dependency.
"""

from __future__ import annotations

import argparse
import concurrent.futures
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
import tempfile
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


CAMPAIGN_ID = "compact-value-bfm-20260831-v1"
SOURCE_CAMPAIGN_ID = "large-teacher-campaign-20260828-v1"
BUNDLE_SCHEMA = "papersoccer.compact-value-bfm-input-bundle.v1"
RUN_START_SCHEMA = "papersoccer.compact-value-bfm-run-start.v1"
RUN_RECEIPT_SCHEMA = "papersoccer.compact-value-bfm-run-receipt.v1"
RUN_REFERENCE_SCHEMA = "papersoccer.compact-value-bfm-run-reference.v1"
PREREQUISITE_REFERENCE_SCHEMA = (
    "papersoccer.compact-value-bfm-prerequisite-reference.v1"
)
FEATURE_SCHEMA = (
    "papersoccer.jacek-replay-bfm.features.v1:edge316+vertex105x57:"
    "mover-relative-rotate180:true-turn-distance+free-degree"
)

HANDOFF_FILE_SHA256 = (
    "1c4e026b013c705ac4ed238a8aa234e035fea028426e7e24e0aae1bb6a8d5459"
)
HANDOFF_BODY_SHA256 = (
    "3abd23e0849828b339b400d78b35374558f18244ab87d31141c23b1e08da957c"
)
ACCEPTANCE_SHA256 = (
    "5b17a4dbd72578d57baf146a3732759eaaa34eca1f61fc3586390740984fbafb"
)
TEACHER_RUNTIME_SHA256 = (
    "f7bdb201a377c04531f1ba98fd73457f7f77961aa0f0f9b1ac32c59b6e85ee75"
)
TEACHER_MANIFEST_SHA256 = (
    "9222b43d46d4e8ae3e7211f429fa306688a2917c7795f26e7514d7a41314ac95"
)
SOURCE_BUNDLE_MANIFEST_SHA256 = (
    "de0e6a8d3060de2b6aac86a32fa3d240e26be6f029cf44b6c0cd73ad4f4fa225"
)
COMMON_ADJUDICATOR_MANIFEST_SHA256 = (
    "41478c5836f61e1d81fdfda8226c3921da16e18164e575424364ff17394bffcc"
)
CANONICAL_MANIFEST_SHA256 = (
    "583f6fec52a0b1b6ff986c963225c68ad4d98fe8e9c9cc659c3a479b0224a89c",
    "c42d571b6d15aa3eb78b73c384cb861a91d936484b2a658fb41fa0955d8cf10f",
    "c79c63e3e63225e3d992ac505d770871a5b9eff9cf6f3891004cb78cc39a9125",
    "279df8861674ed691b6b34c665ebfd4aefe88b853037b96e42344ca7ae4c5d65",
    "26757f60ff1b68c452b32d039b14222e621623ef3f042b591f78a8ce885bbb3e",
    "66f74cea8f29070e7a93f17f5626cac929a7596f8f2139def83cccf121757d85",
    "3c908f7d9295206a4bb2050538104523f24329285802a25365d1d8b1d19ae22c",
    "e6091ac04195662f6acb12333d700be9bdaebbd5f02083ceb9307cdb871f1a53",
    "f5fe53d36145975a142ae2ac6a504340df85e6cd11fc834ca64b7aaeff69e080",
)
CANONICAL_SPLITS = (
    "train", "validation", "test",
    "train", "validation", "test",
    "train", "validation", "test",
)
EXPECTED_NEW_ROWS = {"train": 241_365, "validation": 29_418, "test": 30_706}
EXPECTED_CANONICAL_ROWS = {
    "train": 997_914,
    "validation": 110_004,
    "test": 121_052,
}
EXPECTED_ADJUDICATOR_ROWS = 8_000
ROOT_ARTIFACTS = {
    "roots_tsv": (
        "canonical-roots-tsv", "canonical/roots/teacher-input.tsv",
        "deab1da276fcf6eb3b837eadcf88794d60a22ebc8eb09443a56179e183eb7631",
        17_262,
    ),
    "roots_manifest": (
        "canonical-roots-manifest", "canonical/roots/replay-roots.json",
        "27e3029445c4bff3df0a97f1b159b7433e4ae252e99da3ebad0c12a35aa90926",
        138_227,
    ),
}
OPENING_EXCLUSIONS = (
    ("opening-exclusions/bank-000.tsv", "fde89ddd2dfde2fea62804f17f304c8ef8f54bb2a3353f4d7820242fc604de6b", 43_841),
    ("opening-exclusions/bank-001.tsv", "98af9ff685391d93e6b0d18d2cc06fd98bc33900f4cbfee915e34d23ab8ba245", 43_877),
    ("opening-exclusions/bank-002.tsv", "d8aa66b887fd152c1682c5986e3d6fc868df6bf4db874e5a30b27ad8733b04cc", 17_595),
    ("opening-exclusions/bank-003.tsv", "593da0a7676fd12f37ee4a59460c4e9b7ed6a44c692eddbba7787ef7ece3a597", 26_380),
    ("opening-exclusions/bank-004.tsv", "593da0a7676fd12f37ee4a59460c4e9b7ed6a44c692eddbba7787ef7ece3a597", 26_380),
    ("opening-exclusions/bank-005.tsv", "ab81f04bf43bf5de4c3f57897b8cdc886438c1c1f51c86dc15d2bbf92e8bda4d", 26_343),
    ("opening-exclusions/bank-006.tsv", "dbc36b91ab2b7e937523a5bf59bd9e6225de6518546a3d6b82729fd6a6d5ca90", 26_389),
)
FORBIDDEN_PATH_MARKERS = (
    "sealed-final", "sealed_final", "blind-label", "blind_label",
)

REPOSITORY = pathlib.Path(__file__).resolve().parents[1]
TRAINER_PATH = REPOSITORY / "tools/compact_value_bfm_train.py"
BOT_DIRECTORY = REPOSITORY / "submissions/codingame/bots/compact_value_bfm"
MODEL_EXPORTER_PATH = BOT_DIRECTORY / "export_model.py"
SUBMISSION_EXPORTER_PATH = BOT_DIRECTORY / "export_submission.py"
ARCHITECTURE_ORDER = (
    "compact-8x8",
    "source-neutral-8x16",
    "capacity-12x8",
)
DEPLOYMENT_ARMS = ("search-target", "teacher-assisted")


class WorkflowError(ValueError):
    """A campaign input or receipt violates the frozen contract."""


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=False,
                   separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _reject_path_markers(raw: os.PathLike[str] | str, label: str) -> None:
    # This check intentionally precedes Path construction, resolve, stat, open,
    # hashing, or copy.  A protected decoy is therefore never probed.
    try:
        text = os.fspath(raw)
    except TypeError as error:
        raise WorkflowError(f"{label} is not a filesystem path") from error
    if not isinstance(text, str) or not text:
        raise WorkflowError(f"{label} is not a nonempty text path")
    lowered = text.lower()
    if any(marker in lowered for marker in FORBIDDEN_PATH_MARKERS):
        raise WorkflowError(f"{label} contains a protected path marker")


def checked_root(raw: os.PathLike[str] | str) -> pathlib.Path:
    _reject_path_markers(raw, "source root")
    root = pathlib.Path(raw).resolve()
    _reject_path_markers(root, "resolved source root")
    if root.name != SOURCE_CAMPAIGN_ID or not root.is_dir():
        raise WorkflowError("source root is not the accepted campaign")
    return root


def checked_source_file(
    raw: os.PathLike[str] | str, root: pathlib.Path, label: str,
) -> pathlib.Path:
    _reject_path_markers(raw, label)
    path = pathlib.Path(raw).resolve()
    _reject_path_markers(path, f"resolved {label}")
    try:
        path.relative_to(root)
    except ValueError as error:
        raise WorkflowError(f"{label} escapes the accepted campaign") from error
    if not path.is_file() or path.is_symlink():
        raise WorkflowError(f"{label} is not a regular campaign file")
    return path


def checked_relative_source(
    root: pathlib.Path, raw: object, label: str,
) -> pathlib.Path:
    if not isinstance(raw, str):
        raise WorkflowError(f"{label} route is not text")
    _reject_path_markers(raw, label)
    relative = pathlib.PurePosixPath(raw)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise WorkflowError(f"{label} route is unsafe")
    return checked_source_file(root / relative.as_posix(), root, label)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_file(
    path: pathlib.Path, *, expected_sha256: str | None = None,
    label: str,
) -> tuple[bytes, dict[str, Any]]:
    payload = path.read_bytes()
    if expected_sha256 is not None and sha256_bytes(payload) != expected_sha256:
        raise WorkflowError(f"{label} SHA-256 changed")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise WorkflowError(f"{label} is not JSON") from error
    if not isinstance(value, dict):
        raise WorkflowError(f"{label} must be an object")
    return payload, value


def verify_body_hash(
    value: Mapping[str, object], *, expected: str, label: str,
) -> None:
    body = dict(value)
    claimed = body.pop("body_sha256", None)
    actual = sha256_bytes(canonical_json_bytes(body))
    if claimed != expected or actual != expected:
        raise WorkflowError(f"{label} body SHA-256 changed")


def safe_bundle_path(root: pathlib.Path, raw: object, label: str) -> pathlib.Path:
    if not isinstance(raw, str):
        raise WorkflowError(f"{label} is not a relative path")
    _reject_path_markers(raw, label)
    relative = pathlib.PurePosixPath(raw)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise WorkflowError(f"{label} is unsafe")
    path = (root / relative.as_posix()).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise WorkflowError(f"{label} escapes the input bundle") from error
    return path


@dataclass(frozen=True)
class ImportArtifact:
    role: str
    source: pathlib.Path
    relative_path: str
    sha256: str
    bytes: int


def snapshot_source(
    *, role: str, source: pathlib.Path, relative_path: str,
    expected_sha256: str | None = None, expected_bytes: int | None = None,
) -> ImportArtifact:
    _reject_path_markers(relative_path, role)
    relative = pathlib.PurePosixPath(relative_path)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise WorkflowError(f"invalid destination for {role}")
    size = source.stat().st_size
    digest = sha256_file(source)
    if expected_sha256 is not None and digest != expected_sha256:
        raise WorkflowError(f"{role} SHA-256 changed")
    if expected_bytes is not None and size != expected_bytes:
        raise WorkflowError(f"{role} byte count changed")
    return ImportArtifact(role, source, relative.as_posix(), digest, size)


def _manifest_pair(
    *, role: str, manifest_path: pathlib.Path, destination: str,
    expected_manifest_sha256: str,
) -> tuple[ImportArtifact, ImportArtifact, dict[str, Any]]:
    payload, manifest = load_json_file(
        manifest_path, expected_sha256=expected_manifest_sha256,
        label=f"{role} manifest",
    )
    if (
        manifest.get("schema") != "papersoccer.jacek-replay-csr-shard.v1"
        or manifest.get("feature_schema") != FEATURE_SCHEMA
        or manifest_path.stem != expected_manifest_sha256
        or sha256_bytes(payload) != expected_manifest_sha256
        or not isinstance(manifest.get("npz"), str)
        or not isinstance(manifest.get("npz_sha256"), str)
        or manifest["npz"] != f"{manifest['npz_sha256']}.npz"
    ):
        raise WorkflowError(f"{role} shard manifest is incompatible")
    npz_path = checked_source_file(
        manifest_path.parent / manifest["npz"],
        # All callers already checked that the manifest is below this root;
        # using its nearest accepted-campaign ancestor is handled by caller.
        _accepted_root_for(manifest_path),
        f"{role} adjacent NPZ",
    )
    destination_path = pathlib.PurePosixPath(destination)
    manifest_artifact = snapshot_source(
        role=f"{role}-manifest", source=manifest_path,
        relative_path=(destination_path / manifest_path.name).as_posix(),
        expected_sha256=expected_manifest_sha256,
    )
    npz_artifact = snapshot_source(
        role=f"{role}-npz", source=npz_path,
        relative_path=(destination_path / npz_path.name).as_posix(),
        expected_sha256=str(manifest["npz_sha256"]),
    )
    return manifest_artifact, npz_artifact, manifest


def _accepted_root_for(path: pathlib.Path) -> pathlib.Path:
    for candidate in (path, *path.parents):
        if candidate.name == SOURCE_CAMPAIGN_ID:
            return candidate
    raise WorkflowError("artifact is outside the accepted campaign")


def _artifact_record(artifact: ImportArtifact) -> dict[str, object]:
    return {
        "role": artifact.role,
        "relative_path": artifact.relative_path,
        "sha256": artifact.sha256,
        "bytes": artifact.bytes,
    }


def _append_unique(
    artifacts: list[ImportArtifact], artifact: ImportArtifact,
) -> None:
    if any(existing.role == artifact.role for existing in artifacts):
        raise WorkflowError(f"duplicate input role: {artifact.role}")
    if any(existing.relative_path == artifact.relative_path for existing in artifacts):
        raise WorkflowError(f"duplicate bundle path: {artifact.relative_path}")
    artifacts.append(artifact)


def _handoff_records(
    handoff: Mapping[str, Any], root: pathlib.Path,
    artifacts: list[ImportArtifact], routes: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    manifests: dict[str, list[dict[str, Any]]] = {"search": [], "rank4": []}
    for phase in ("pilot", "full"):
        for arm in ("search", "rank4"):
            key = f"{phase}_{arm}_manifests"
            records = handoff.get(key)
            if not isinstance(records, list) or len(records) != 3:
                raise WorkflowError(f"handoff {key} roster changed")
            destination_routes: list[str] = []
            splits: list[str] = []
            for index, record in enumerate(records):
                if not isinstance(record, dict):
                    raise WorkflowError(f"handoff {key} entry is malformed")
                source = checked_source_file(record.get("path", ""), root, key)
                expected = str(record.get("sha256", ""))
                destination = f"new/{arm}/{phase}"
                prefix = f"new-{arm}-{phase}-{index}"
                manifest_artifact, npz_artifact, manifest = _manifest_pair(
                    role=prefix, manifest_path=source, destination=destination,
                    expected_manifest_sha256=expected,
                )
                if (
                    record.get("bytes") != manifest_artifact.bytes
                    or manifest.get("split") not in {"train", "validation", "test"}
                ):
                    raise WorkflowError(f"handoff {key} snapshot changed")
                _append_unique(artifacts, manifest_artifact)
                _append_unique(artifacts, npz_artifact)
                destination_routes.append(manifest_artifact.relative_path)
                splits.append(str(manifest["split"]))
                manifests[arm].append(manifest)
            if splits != ["train", "validation", "test"]:
                raise WorkflowError(f"handoff {key} split order changed")
            routes[f"{phase}_{arm}_manifests"] = destination_routes
    return manifests


def collect_imports(source_root: pathlib.Path) -> tuple[
    list[ImportArtifact], dict[str, Any], dict[str, Any]
]:
    artifacts: list[ImportArtifact] = []
    routes: dict[str, Any] = {}

    handoff_path = checked_source_file(
        source_root / "compact-student-handoff.json", source_root, "handoff"
    )
    handoff_payload, handoff = load_json_file(
        handoff_path, expected_sha256=HANDOFF_FILE_SHA256, label="handoff"
    )
    verify_body_hash(handoff, expected=HANDOFF_BODY_SHA256, label="handoff")
    if (
        handoff.get("schema") != "papersoccer.compact-student-handoff.v1"
        or handoff.get("campaign_id") != SOURCE_CAMPAIGN_ID
        or handoff.get("student_training_eligible") is not True
        or handoff.get("student_training_started") is not False
        or handoff.get("external_upload") is not False
        or handoff.get("replace_rank4") is not False
        or handoff.get("leaderboard_claim") is not False
    ):
        raise WorkflowError("handoff policy changed")
    _append_unique(artifacts, snapshot_source(
        role="handoff", source=handoff_path,
        relative_path="provenance/compact-student-handoff.json",
        expected_sha256=HANDOFF_FILE_SHA256,
        expected_bytes=len(handoff_payload),
    ))
    routes["handoff"] = "provenance/compact-student-handoff.json"

    fixed = (
        ("acceptance", "teacher_candidate_accepted", ACCEPTANCE_SHA256,
         "provenance/teacher-candidate-accepted.json"),
        ("teacher-runtime", "teacher_runtime", TEACHER_RUNTIME_SHA256,
         "teacher/jacek_replay_bfm.runtime"),
        ("teacher-manifest", "teacher_manifest", TEACHER_MANIFEST_SHA256,
         "teacher/jacek_replay_bfm.runtime.json"),
    )
    for role, key, digest, destination in fixed:
        record = handoff.get(key)
        if not isinstance(record, dict):
            raise WorkflowError(f"handoff omits {key}")
        source = checked_source_file(record.get("path", ""), source_root, role)
        artifact = snapshot_source(
            role=role, source=source, relative_path=destination,
            expected_sha256=digest,
            expected_bytes=int(record.get("bytes", -1)),
        )
        _append_unique(artifacts, artifact)
        routes[key] = destination

    acceptance_path = next(item.source for item in artifacts if item.role == "acceptance")
    _, acceptance = load_json_file(
        acceptance_path, expected_sha256=ACCEPTANCE_SHA256, label="acceptance"
    )
    if (
        acceptance.get("schema") != "papersoccer.teacher-candidate-accepted.v1"
        or acceptance.get("pilot_20_ms_passed") is not False
        or acceptance.get("pilot_passed") is not False
        or acceptance.get("external_upload") is not False
        or acceptance.get("replace_rank4") is not False
        or acceptance.get("leaderboard_claim") is not False
    ):
        raise WorkflowError("teacher acceptance truth changed")

    arm_manifests = _handoff_records(handoff, source_root, artifacts, routes)
    for arm, manifests in arm_manifests.items():
        totals = {
            split: sum(int(item["samples"]) for item in manifests
                       if item.get("split") == split)
            for split in ("train", "validation", "test")
        }
        if totals != EXPECTED_NEW_ROWS:
            raise WorkflowError(f"{arm} frozen row totals changed: {totals}")

    source_bundle_path = checked_source_file(
        source_root / "input-bundle/bundle-manifest.json",
        source_root, "accepted input-bundle manifest",
    )
    _, source_bundle = load_json_file(
        source_bundle_path, expected_sha256=SOURCE_BUNDLE_MANIFEST_SHA256,
        label="accepted input-bundle manifest"
    )
    source_routes = source_bundle.get("routes")
    source_records = source_bundle.get("artifacts")
    if not isinstance(source_routes, dict) or not isinstance(source_records, list):
        raise WorkflowError("accepted input-bundle routing is malformed")
    canonical_routes = source_routes.get("canonical_prior_manifests")
    if (
        not isinstance(canonical_routes, list)
        or [pathlib.PurePosixPath(str(item)).stem for item in canonical_routes]
        != list(CANONICAL_MANIFEST_SHA256)
    ):
        raise WorkflowError("canonical R0/R1/R2 ordering changed")
    canonical_by_split: dict[str, list[str]] = {
        "train": [], "validation": [], "test": [],
    }
    canonical_manifests: list[dict[str, Any]] = []
    source_bundle_root = source_bundle_path.parent
    for index, (relative, digest, split) in enumerate(zip(
        canonical_routes, CANONICAL_MANIFEST_SHA256, CANONICAL_SPLITS,
        strict=True,
    )):
        source = checked_relative_source(
            source_bundle_root, relative, f"canonical-r{index // 3}-{split}"
        )
        destination = f"canonical/r{index // 3}"
        manifest_artifact, npz_artifact, manifest = _manifest_pair(
            role=f"canonical-r{index // 3}-{split}", manifest_path=source,
            destination=destination, expected_manifest_sha256=digest,
        )
        if manifest.get("split") != split:
            raise WorkflowError("canonical split routing changed")
        _append_unique(artifacts, manifest_artifact)
        _append_unique(artifacts, npz_artifact)
        canonical_by_split[split].append(manifest_artifact.relative_path)
        canonical_manifests.append(manifest)
    canonical_totals = {
        split: sum(int(item["samples"]) for item in canonical_manifests
                   if item.get("split") == split)
        for split in ("train", "validation", "test")
    }
    if canonical_totals != EXPECTED_CANONICAL_ROWS:
        raise WorkflowError(f"canonical frozen row totals changed: {canonical_totals}")
    routes["canonical_splits"] = canonical_by_split
    routes["canonical_prior_manifests"] = [
        f"canonical/r{index // 3}/{digest}.json"
        for index, digest in enumerate(CANONICAL_MANIFEST_SHA256)
    ]

    for route_key, fixed in ROOT_ARTIFACTS.items():
        role, fixed_relative, fixed_sha256, fixed_bytes = fixed
        relative = source_routes.get(route_key)
        record = next(
            (item for item in source_records
             if isinstance(item, dict) and item.get("role") == role),
            None,
        )
        if (
            not isinstance(record, dict)
            or relative != fixed_relative
            or record.get("relative_path") != fixed_relative
            or record.get("sha256") != fixed_sha256
            or record.get("bytes") != fixed_bytes
        ):
            raise WorkflowError(f"accepted bundle omits {role}")
        source = checked_relative_source(source_bundle_root, relative, role)
        artifact = snapshot_source(
            role=role, source=source, relative_path=fixed_relative,
            expected_sha256=fixed_sha256, expected_bytes=fixed_bytes,
        )
        _append_unique(artifacts, artifact)
        routes[route_key] = fixed_relative

    exclusions = source_routes.get("opening_exclusions")
    if exclusions != [item[0] for item in OPENING_EXCLUSIONS]:
        raise WorkflowError("opening-exclusion roster changed")
    exclusion_routes: list[str] = []
    for index, (relative, fixed_sha256, fixed_bytes) in enumerate(OPENING_EXCLUSIONS):
        expected_roles = {f"opening-exclusion-{index:03d}"}
        if index == 6:
            expected_roles.add("opening-exclusion-pilot")
        record = next(
            (item for item in source_records
             if isinstance(item, dict) and item.get("role") in expected_roles
             and item.get("relative_path") == relative),
            None,
        )
        if (
            not isinstance(record, dict)
            or record.get("sha256") != fixed_sha256
            or record.get("bytes") != fixed_bytes
        ):
            raise WorkflowError(f"opening exclusion {index} is unbound")
        source = checked_relative_source(
            source_bundle_root, relative, f"opening-exclusion-{index:03d}"
        )
        destination = f"opening-exclusions/bank-{index:03d}.tsv"
        artifact = snapshot_source(
            role=f"opening-exclusion-{index:03d}", source=source,
            relative_path=destination,
            expected_sha256=fixed_sha256, expected_bytes=fixed_bytes,
        )
        _append_unique(artifacts, artifact)
        exclusion_routes.append(destination)
    routes["opening_exclusions"] = exclusion_routes

    adjudicator_path = checked_source_file(
        source_root / "full/shards/adjudicator" /
        f"{COMMON_ADJUDICATOR_MANIFEST_SHA256}.json",
        source_root, "common adjudicator manifest",
    )
    adjudicator_manifest, adjudicator_npz, adjudicator = _manifest_pair(
        role="common-adjudicator", manifest_path=adjudicator_path,
        destination="adjudicator",
        expected_manifest_sha256=COMMON_ADJUDICATOR_MANIFEST_SHA256,
    )
    if (
        adjudicator.get("split") != "validation"
        or adjudicator.get("samples") != EXPECTED_ADJUDICATOR_ROWS
    ):
        raise WorkflowError("common adjudicator is not the frozen 8,000-row shard")
    _append_unique(artifacts, adjudicator_manifest)
    _append_unique(artifacts, adjudicator_npz)
    routes["common_adjudicator_manifest"] = adjudicator_manifest.relative_path

    policy = {
        "source_campaign_id": SOURCE_CAMPAIGN_ID,
        "explicit_allowlist": True,
        "source_campaign_scanned": False,
        "protected_path_markers_rejected_before_access": True,
        "sealed_final_accessed": False,
        "blind_labels_accessed": False,
        "runtime_uses_source_paths": False,
        "git_required_after_freeze": False,
        "protected_tests_locked": True,
        "external_upload": False,
        "replace_rank4": False,
        "rank1_claim": False,
    }
    return artifacts, routes, policy


def _copy_artifact(artifact: ImportArtifact, staging: pathlib.Path) -> None:
    target = safe_bundle_path(staging, artifact.relative_path, artifact.role)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.copy")
    with artifact.source.open("rb") as source, temporary.open("xb") as output:
        shutil.copyfileobj(source, output, length=1024 * 1024)
        output.flush()
        os.fsync(output.fileno())
    os.chmod(temporary, 0o444)
    os.replace(temporary, target)
    if target.stat().st_size != artifact.bytes or sha256_file(target) != artifact.sha256:
        raise WorkflowError(f"{artifact.role} changed while copying")


def _bundle_manifest(
    artifacts: Sequence[ImportArtifact], routes: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema": BUNDLE_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "feature_schema": FEATURE_SCHEMA,
        "routes": dict(routes),
        "artifacts": sorted(
            (_artifact_record(item) for item in artifacts),
            key=lambda item: str(item["role"]),
        ),
        "row_counts": {
            "search": dict(EXPECTED_NEW_ROWS),
            "rank4_control": dict(EXPECTED_NEW_ROWS),
            "canonical": dict(EXPECTED_CANONICAL_ROWS),
            "common_adjudicator": EXPECTED_ADJUDICATOR_ROWS,
        },
        "protected_splits": ["search:test", "rank4:test", "canonical:test"],
        "policy": dict(policy),
    }
    result = dict(body)
    result["body_sha256"] = sha256_bytes(canonical_json_bytes(body))
    return result


def freeze_inputs(source: pathlib.Path, output_directory: pathlib.Path) -> dict[str, Any]:
    source = checked_root(source)
    output_directory = output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    bundle_root = output_directory / "input-bundle"
    manifest_path = bundle_root / "bundle-manifest.json"
    artifacts, routes, policy = collect_imports(source)
    expected = _bundle_manifest(artifacts, routes, policy)
    if bundle_root.exists():
        actual = verify_bundle(manifest_path)
        if actual != expected:
            raise WorkflowError("existing frozen bundle differs from the allowlist")
        return actual

    staging = pathlib.Path(tempfile.mkdtemp(
        dir=output_directory, prefix=".compact-value-inputs-"
    )).resolve()
    try:
        for artifact in artifacts:
            _copy_artifact(artifact, staging)
        manifest = _bundle_manifest(artifacts, routes, policy)
        manifest_payload = canonical_json_bytes(manifest)
        manifest_target = staging / "bundle-manifest.json"
        with manifest_target.open("xb") as output:
            output.write(manifest_payload)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(manifest_target, 0o444)
        os.replace(staging, bundle_root)
        return verify_bundle(manifest_path)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _route_strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            result.extend(_route_strings(item))
        return result
    if isinstance(value, dict):
        result = []
        for item in value.values():
            result.extend(_route_strings(item))
        return result
    raise WorkflowError("bundle route map is malformed")


def verify_bundle(manifest_path: pathlib.Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    bundle_root = manifest_path.parent
    payload, manifest = load_json_file(manifest_path, label="compact input bundle")
    if payload != canonical_json_bytes(manifest):
        raise WorkflowError("bundle manifest is not canonical JSON")
    body = dict(manifest)
    claimed = body.pop("body_sha256", None)
    if (
        body.get("schema") != BUNDLE_SCHEMA
        or body.get("campaign_id") != CAMPAIGN_ID
        or body.get("feature_schema") != FEATURE_SCHEMA
        or claimed != sha256_bytes(canonical_json_bytes(body))
        or body.get("row_counts") != {
            "search": EXPECTED_NEW_ROWS,
            "rank4_control": EXPECTED_NEW_ROWS,
            "canonical": EXPECTED_CANONICAL_ROWS,
            "common_adjudicator": EXPECTED_ADJUDICATOR_ROWS,
        }
    ):
        raise WorkflowError("bundle manifest body binding changed")
    policy = body.get("policy")
    if (
        not isinstance(policy, dict)
        or policy.get("explicit_allowlist") is not True
        or policy.get("source_campaign_scanned") is not False
        or policy.get("sealed_final_accessed") is not False
        or policy.get("blind_labels_accessed") is not False
        or policy.get("runtime_uses_source_paths") is not False
        or policy.get("git_required_after_freeze") is not False
        or policy.get("protected_tests_locked") is not True
    ):
        raise WorkflowError("bundle safety policy changed")
    records = body.get("artifacts")
    routes = body.get("routes")
    if not isinstance(records, list) or not isinstance(routes, dict):
        raise WorkflowError("bundle registry is malformed")
    by_path: dict[str, dict[str, Any]] = {}
    roles: set[str] = set()
    for record in records:
        if (
            not isinstance(record, dict)
            or set(record) != {"role", "relative_path", "sha256", "bytes"}
            or not isinstance(record.get("role"), str)
            or not isinstance(record.get("relative_path"), str)
            or not isinstance(record.get("sha256"), str)
            or len(record["sha256"]) != 64
            or type(record.get("bytes")) is not int
            or record["bytes"] < 0
            or record["role"] in roles
            or record["relative_path"] in by_path
        ):
            raise WorkflowError("bundle artifact registry is malformed")
        path = safe_bundle_path(bundle_root, record["relative_path"], record["role"])
        if (
            not path.is_file() or path.is_symlink()
            or path.stat().st_size != record["bytes"]
            or sha256_file(path) != record["sha256"]
        ):
            raise WorkflowError(f"copied artifact changed: {record['role']}")
        roles.add(record["role"])
        by_path[record["relative_path"]] = record
    actual_files = {
        path.relative_to(bundle_root).as_posix()
        for path in bundle_root.rglob("*") if path.is_file()
    }
    if actual_files != {*by_path, manifest_path.name}:
        raise WorkflowError("bundle contains an unregistered file")
    route_values = _route_strings(routes)
    if any(relative not in by_path for relative in route_values):
        raise WorkflowError("bundle route is not backed by an artifact")
    if len(routes.get("canonical_prior_manifests", [])) != 9:
        raise WorkflowError("bundle canonical ordering is incomplete")
    if len(routes.get("opening_exclusions", [])) != 7:
        raise WorkflowError("bundle opening exclusions are incomplete")
    for key in (
        "pilot_search_manifests", "full_search_manifests",
        "pilot_rank4_manifests", "full_rank4_manifests",
    ):
        if len(routes.get(key, [])) != 3:
            raise WorkflowError(f"bundle {key} is incomplete")
    return manifest


def _body_hashed(body: Mapping[str, object]) -> dict[str, object]:
    result = dict(body)
    result["body_sha256"] = sha256_bytes(canonical_json_bytes(body))
    return result


def _verify_hashed_document(
    value: Mapping[str, object], *, schema: str, label: str,
) -> None:
    body = dict(value)
    claimed = body.pop("body_sha256", None)
    if (
        body.get("schema") != schema
        or not isinstance(claimed, str)
        or claimed != sha256_bytes(canonical_json_bytes(body))
    ):
        raise WorkflowError(f"{label} body SHA-256 is invalid")


def _atomic_bytes(path: pathlib.Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: pathlib.Path | None = None
    try:
        descriptor, raw = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}."
        )
        temporary = pathlib.Path(raw)
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, 0o444)
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_stable_document(
    path: pathlib.Path, value: Mapping[str, object], label: str,
) -> None:
    payload = canonical_json_bytes(value)
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise WorkflowError(f"existing {label} differs from frozen content")
        return
    _atomic_bytes(path, payload)


def _write_content_addressed(
    directory: pathlib.Path, value: Mapping[str, object], suffix: str,
) -> pathlib.Path:
    payload = canonical_json_bytes(value)
    path = directory / f"{sha256_bytes(payload)}{suffix}"
    if path.exists():
        if path.read_bytes() != payload:
            raise WorkflowError("content-addressed run artifact conflicts")
    else:
        _atomic_bytes(path, payload)
    return path


def _load_canonical_document(path: pathlib.Path, label: str) -> tuple[bytes, dict[str, Any]]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise WorkflowError(f"could not read {label}") from error
    if not isinstance(value, dict) or payload != canonical_json_bytes(value):
        raise WorkflowError(f"{label} is not canonical JSON")
    return payload, value


def _relative_to(path: pathlib.Path, root: pathlib.Path, label: str) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise WorkflowError(f"{label} escaped the run output") from error


def _run_output_path(
    root: pathlib.Path, raw: object, label: str,
) -> pathlib.Path:
    relative = safe_bundle_path(root, raw, label)
    return relative


@dataclass(frozen=True)
class RuntimeComponents:
    trainer: Any
    model_exporter: Any
    submission_exporter: Any


def _load_python_module(name: str, path: pathlib.Path) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise WorkflowError(f"could not load {path.name}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    try:
        specification.loader.exec_module(module)
    except Exception as error:
        raise WorkflowError(f"could not import {path.name}") from error
    return module


def _lazy_runtime_components() -> RuntimeComponents:
    """Import NumPy/trainer/exporters only for run or run verification."""

    trainer = _load_python_module(
        "compact_value_bfm_train_for_workflow", TRAINER_PATH
    )
    model_exporter = _load_python_module(
        "compact_value_bfm_model_exporter_for_workflow", MODEL_EXPORTER_PATH
    )
    submission_exporter = _load_python_module(
        "compact_value_bfm_submission_exporter_for_workflow",
        SUBMISSION_EXPORTER_PATH,
    )
    return RuntimeComponents(trainer, model_exporter, submission_exporter)


def _file_record(path: pathlib.Path) -> dict[str, object]:
    return {
        "path": path.resolve().relative_to(REPOSITORY.resolve()).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _tool_bindings() -> dict[str, dict[str, object]]:
    paths: dict[str, pathlib.Path] = {
        "workflow": pathlib.Path(__file__).resolve(),
        "trainer": TRAINER_PATH,
        "model_exporter": MODEL_EXPORTER_PATH,
        "submission_exporter": SUBMISSION_EXPORTER_PATH,
        "submission_configuration": BOT_DIRECTORY / "submission.json",
        "source_manifest": BOT_DIRECTORY / "sources.txt",
    }
    source_manifest = paths["source_manifest"]
    try:
        source_lines = source_manifest.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise WorkflowError("could not read compact source manifest") from error
    source_index = 0
    for raw in source_lines:
        relative = raw.strip()
        if not relative or relative.startswith("#"):
            continue
        candidate = (REPOSITORY / relative).resolve()
        try:
            candidate.relative_to(REPOSITORY.resolve())
        except ValueError as error:
            raise WorkflowError("compact source manifest escapes repository") from error
        # The model header is deliberately replaced in memory during every
        # size probe.  Binding it would make an unrelated active model affect
        # a source-neutral campaign measurement.
        if candidate == (BOT_DIRECTORY / "model.hpp").resolve():
            continue
        paths[f"source_{source_index:02d}"] = candidate
        source_index += 1
    records = {}
    for name, path in sorted(paths.items()):
        if not path.is_file():
            raise WorkflowError(f"run tool is missing: {path}")
        records[name] = _file_record(path)
    return records


def _active_output_snapshots() -> dict[str, dict[str, object] | None]:
    result: dict[str, dict[str, object] | None] = {}
    for name in ("model.hpp", "submission.cpp"):
        path = BOT_DIRECTORY / name
        result[name] = (
            {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            if path.is_file()
            else None
        )
    return result


def _measure_one_runtime_source(
    components: RuntimeComponents, runtime_path: pathlib.Path,
) -> dict[str, object]:
    before = _active_output_snapshots()
    try:
        header, metadata = components.model_exporter.render_header(runtime_path)
        _unused_output, source = components.submission_exporter.render(
            model_header=header
        )
        source.decode("ascii")
    except Exception as error:
        raise WorkflowError("compact source exporter measurement failed") from error
    if _active_output_snapshots() != before:
        raise WorkflowError("source size measurement overwrote active model or source")
    return {
        "architecture": metadata["architecture"]["name"],
        "runtime_file_sha256": metadata["file_sha256"],
        "runtime_body_sha256": metadata["body_sha256"],
        "runtime_payload_sha256": metadata["payload_sha256"],
        "header_ascii_characters": len(header),
        "header_sha256": sha256_bytes(header),
        "complete_source_ascii_characters": len(source),
        "complete_source_sha256": sha256_bytes(source),
        "limit": 95_000,
        "eligible": len(source) <= 95_000,
        "active_model_and_source_overwritten": False,
    }


def measure_architecture_source_sizes(
    components: RuntimeComponents,
    bundle_body_sha256: str,
    scratch_parent: pathlib.Path,
) -> dict[str, dict[str, object]]:
    """Measure every architecture through both exporters without publication."""

    scratch_parent.mkdir(parents=True, exist_ok=True)
    reports: dict[str, dict[str, object]] = {}
    with tempfile.TemporaryDirectory(
        dir=scratch_parent, prefix=".compact-source-size-probe."
    ) as raw:
        scratch = pathlib.Path(raw)
        for architecture_name in ARCHITECTURE_ORDER:
            architecture = components.trainer.ARCHITECTURES[architecture_name]
            quantized = components.trainer.QuantizedWeights(
                {
                    name: components.trainer.np.zeros(shape, dtype=components.trainer.np.int8)
                    for name, shape in architecture.shapes.items()
                },
                {
                    "w1": components.trainer.np.float32(0.01),
                    "w2": components.trainer.np.float32(0.02),
                    "w3": components.trainer.np.float32(0.03),
                },
            )
            runtime = components.trainer.write_runtime(
                scratch,
                architecture,
                quantized,
                arm="search-target",
                seed=20260907,
                float_epoch=1,
                qat_epoch=0,
                source_bundle_body_sha256=bundle_body_sha256,
            )
            report = _measure_one_runtime_source(components, runtime)
            if report["architecture"] != architecture_name:
                raise WorkflowError("source size probe architecture changed")
            reports[architecture_name] = report
    return reports


def _bundle_run_binding(
    bundle_manifest: pathlib.Path, bundle: Any,
) -> dict[str, object]:
    return {
        "manifest_sha256": sha256_file(bundle_manifest),
        "manifest_bytes": bundle_manifest.stat().st_size,
        "body_sha256": bundle.body_sha256,
        "campaign_id": bundle.manifest.get("campaign_id"),
    }


def _validate_prerequisite_reference(
    components: RuntimeComponents,
    bundle: Any,
    output_directory: pathlib.Path,
    reference_path: pathlib.Path,
    kind: str,
) -> tuple[pathlib.Path, dict[str, Any]]:
    _payload, reference = _load_canonical_document(
        reference_path, f"{kind} prerequisite reference"
    )
    _verify_hashed_document(
        reference,
        schema=PREREQUISITE_REFERENCE_SCHEMA,
        label=f"{kind} prerequisite reference",
    )
    artifact = reference.get("artifact")
    expected_trainer = _file_record(TRAINER_PATH)
    if (
        set(reference) != {
            "schema", "campaign_id", "kind", "source_bundle_body_sha256",
            "trainer", "artifact", "protected_tests_opened", "body_sha256",
        }
        or reference.get("campaign_id") != CAMPAIGN_ID
        or reference.get("kind") != kind
        or reference.get("source_bundle_body_sha256") != bundle.body_sha256
        or reference.get("trainer") != expected_trainer
        or reference.get("protected_tests_opened") is not False
        or not isinstance(artifact, dict)
        or set(artifact) != {"path", "sha256", "body_sha256", "bytes"}
    ):
        raise WorkflowError(f"{kind} prerequisite reference changed")
    artifact_path = _run_output_path(
        output_directory, artifact["path"], f"{kind} artifact"
    )
    if (
        not artifact_path.is_file()
        or artifact_path.stat().st_size != artifact["bytes"]
        or sha256_file(artifact_path) != artifact["sha256"]
    ):
        raise WorkflowError(f"{kind} prerequisite artifact changed")
    if kind == "input-audit":
        document = components.trainer.validate_input_audit(bundle, artifact_path)
    elif kind == "teacher-sidecars":
        components.trainer.load_sidecar_index(bundle, artifact_path)
        document = _load_canonical_document(
            artifact_path, "teacher sidecar index"
        )[1]
    else:
        raise WorkflowError("unknown run prerequisite")
    if document.get("body_sha256") != artifact["body_sha256"]:
        raise WorkflowError(f"{kind} prerequisite body binding changed")
    return artifact_path, reference


def _publish_prerequisite_reference(
    bundle: Any,
    output_directory: pathlib.Path,
    reference_path: pathlib.Path,
    kind: str,
    artifact_path: pathlib.Path,
) -> dict[str, Any]:
    document = _load_canonical_document(
        artifact_path, f"{kind} prerequisite artifact"
    )[1]
    body: dict[str, object] = {
        "schema": PREREQUISITE_REFERENCE_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "kind": kind,
        "source_bundle_body_sha256": bundle.body_sha256,
        "trainer": _file_record(TRAINER_PATH),
        "artifact": {
            "path": _relative_to(artifact_path, output_directory, kind),
            "sha256": sha256_file(artifact_path),
            "body_sha256": document.get("body_sha256"),
            "bytes": artifact_path.stat().st_size,
        },
        "protected_tests_opened": False,
    }
    reference = _body_hashed(body)
    _write_stable_document(reference_path, reference, f"{kind} reference")
    return reference


def _ensure_run_prerequisites(
    components: RuntimeComponents,
    bundle: Any,
    output_directory: pathlib.Path,
) -> dict[str, object]:
    state = output_directory / "run-state"
    audit_reference_path = state / "input-audit.reference.json"
    if audit_reference_path.exists():
        audit_path, audit_reference = _validate_prerequisite_reference(
            components, bundle, output_directory, audit_reference_path, "input-audit"
        )
    else:
        audit_path = components.trainer.generate_input_audit(
            bundle, output_directory / "prerequisites/input-audit"
        )
        audit_reference = _publish_prerequisite_reference(
            bundle,
            output_directory,
            audit_reference_path,
            "input-audit",
            audit_path,
        )

    sidecar_reference_path = state / "teacher-sidecars.reference.json"
    if sidecar_reference_path.exists():
        sidecar_path, sidecar_reference = _validate_prerequisite_reference(
            components,
            bundle,
            output_directory,
            sidecar_reference_path,
            "teacher-sidecars",
        )
    else:
        sidecar_path = components.trainer.generate_teacher_sidecars(
            bundle, output_directory / "prerequisites/teacher-sidecars"
        )
        sidecar_reference = _publish_prerequisite_reference(
            bundle,
            output_directory,
            sidecar_reference_path,
            "teacher-sidecars",
            sidecar_path,
        )
    return {
        "input_audit_path": audit_path,
        "input_audit_reference": audit_reference,
        "sidecar_index_path": sidecar_path,
        "sidecar_reference": sidecar_reference,
    }


@dataclass(frozen=True)
class CampaignSpec:
    architecture: str
    arm: str
    measured_source_ascii_characters: int

    @property
    def name(self) -> str:
        return f"{self.architecture}--{self.arm}"


def family_campaign_specs(
    source_sizes: Mapping[str, Mapping[str, object]],
) -> tuple[CampaignSpec, ...]:
    result: list[CampaignSpec] = []
    for architecture in ARCHITECTURE_ORDER:
        measurement = source_sizes.get(architecture)
        if (
            not isinstance(measurement, Mapping)
            or measurement.get("eligible") is not True
            or type(measurement.get("complete_source_ascii_characters")) is not int
            or int(measurement["complete_source_ascii_characters"]) > 95_000
        ):
            continue
        for arm in DEPLOYMENT_ARMS:
            result.append(CampaignSpec(
                architecture,
                arm,
                int(measurement["complete_source_ascii_characters"]),
            ))
    primary = source_sizes.get("compact-8x8")
    if isinstance(primary, Mapping) and primary.get("eligible") is True:
        result.append(CampaignSpec(
            "compact-8x8",
            "rank4-control",
            int(primary["complete_source_ascii_characters"]),
        ))
    return tuple(result)


def _single_thread_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.update({
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONHASHSEED": "0",
    })
    return environment


def _run_training_process(
    spec: CampaignSpec,
    *,
    bundle_manifest: pathlib.Path,
    output_directory: pathlib.Path,
    input_audit: pathlib.Path,
    sidecar_index: pathlib.Path,
    resume: bool,
) -> dict[str, object]:
    campaign_output = output_directory / "campaigns" / spec.name
    command = [
        sys.executable,
        str(TRAINER_PATH),
        "train",
        "--bundle-manifest", str(bundle_manifest),
        "--output-directory", str(campaign_output),
        "--architecture", spec.architecture,
        "--arm", spec.arm,
        "--input-audit", str(input_audit),
        "--generated-source-ascii-bytes",
        str(spec.measured_source_ascii_characters),
    ]
    if spec.arm == "teacher-assisted":
        command.extend(("--sidecar-index", str(sidecar_index)))
    if resume:
        command.append("--resume")
    completed = subprocess.run(
        command,
        cwd=REPOSITORY,
        env=_single_thread_environment(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip()[-4_000:]
        raise WorkflowError(
            f"training campaign {spec.name} failed: {detail}"
        )
    try:
        report = json.loads(completed.stdout)
        raw_selection = report["selection"]
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise WorkflowError(
            f"training campaign {spec.name} returned malformed output"
        ) from error
    selection_path = pathlib.Path(raw_selection)
    if not selection_path.is_absolute():
        selection_path = REPOSITORY / selection_path
    selection_path = selection_path.resolve()
    try:
        selection_path.relative_to(campaign_output.resolve())
    except ValueError as error:
        raise WorkflowError("training selection escaped its campaign output") from error
    return {
        "spec": spec,
        "campaign_output": campaign_output,
        "selection_path": selection_path,
        "stdout_sha256": sha256_bytes(completed.stdout.encode("utf-8")),
        "single_thread_blas": True,
    }


def _execute_campaigns(
    specs: Sequence[CampaignSpec],
    *,
    workers: int,
    runner: Any,
) -> list[dict[str, object]]:
    if workers not in {1, 2}:
        raise WorkflowError("run workers must be one or two")
    if not specs:
        raise WorkflowError("source measurement left no eligible campaign")
    if workers == 1:
        return [runner(spec) for spec in specs]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {spec.name: executor.submit(runner, spec) for spec in specs}
        # Resolve in the frozen specification order so worker count cannot
        # affect receipt bytes or selection ordering.
        return [futures[spec.name].result() for spec in specs]


def _campaign_result(
    components: RuntimeComponents,
    bundle: Any,
    output_directory: pathlib.Path,
    process_result: Mapping[str, object],
) -> dict[str, object]:
    spec = process_result.get("spec")
    selection_path = process_result.get("selection_path")
    campaign_output = process_result.get("campaign_output")
    if (
        not isinstance(spec, CampaignSpec)
        or not isinstance(selection_path, pathlib.Path)
        or not isinstance(campaign_output, pathlib.Path)
    ):
        raise WorkflowError("training process result is malformed")
    try:
        selection = components.trainer.validate_selection(
            selection_path, campaign_output, bundle
        )
    except Exception as error:
        raise WorkflowError(
            f"training selection {spec.name} failed validation"
        ) from error
    if (
        selection.get("architecture") != spec.architecture
        or selection.get("arm") != spec.arm
        or selection.get("protected_tests_opened") is not False
        or selection.get("game_gated") is not False
    ):
        raise WorkflowError("training selection identity or test lock changed")
    runtime_record = selection.get("runtime")
    if not isinstance(runtime_record, dict):
        raise WorkflowError("training selection has no runtime")
    runtime_path = _run_output_path(
        campaign_output, runtime_record.get("path"), "selected runtime"
    )
    if (
        not runtime_path.is_file()
        or sha256_file(runtime_path) != runtime_record.get("sha256")
        or runtime_path.stat().st_size != runtime_record.get("bytes")
    ):
        raise WorkflowError("selected runtime content binding changed")
    source_measurement: dict[str, object] | None = None
    if spec.arm in DEPLOYMENT_ARMS:
        source_measurement = _measure_one_runtime_source(
            components, runtime_path
        )
        if (
            source_measurement.get("architecture") != spec.architecture
            or source_measurement.get("eligible") is not True
        ):
            raise WorkflowError("selected runtime exceeds the compact source cap")
    return {
        "name": spec.name,
        "architecture": spec.architecture,
        "arm": spec.arm,
        "preflight_source_ascii_characters": (
            spec.measured_source_ascii_characters
        ),
        "campaign_output": _relative_to(
            campaign_output, output_directory, "campaign output"
        ),
        "selection": {
            "path": _relative_to(
                selection_path, output_directory, "campaign selection"
            ),
            "sha256": sha256_file(selection_path),
            "bytes": selection_path.stat().st_size,
            "body_sha256": selection["body_sha256"],
            "status": selection["status"],
            "deployment_eligible": selection["deployment_eligible"],
            "seed": selection["seed"],
            "float_epoch": selection["float_epoch"],
            "qat_epoch": selection["qat_epoch"],
        },
        "runtime": {
            "path": _relative_to(runtime_path, output_directory, "runtime"),
            "sha256": sha256_file(runtime_path),
            "bytes": runtime_path.stat().st_size,
            "body_sha256": _load_canonical_document(
                runtime_path, "selected runtime"
            )[1]["body_sha256"],
        },
        "exact_complete_source": source_measurement,
        "rank4_control_never_deployment_eligible": spec.arm == "rank4-control",
        "single_thread_blas": process_result.get("single_thread_blas") is True,
        "protected_tests_opened": False,
    }


@contextlib.contextmanager
def _exclusive_run_lock(output_directory: pathlib.Path) -> Any:
    state = output_directory / "run-state"
    state.mkdir(parents=True, exist_ok=True)
    path = state / "run.lock"
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise WorkflowError("compact family run is already active") from error
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _run_start_body(
    *,
    bundle_binding: Mapping[str, object],
    tools: Mapping[str, object],
    prerequisites: Mapping[str, object],
    source_sizes: Mapping[str, object],
    specs: Sequence[CampaignSpec],
) -> dict[str, object]:
    return {
        "schema": RUN_START_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "bundle": dict(bundle_binding),
        "tools": dict(tools),
        "input_audit_reference_body_sha256": prerequisites[
            "input_audit_reference"
        ]["body_sha256"],
        "teacher_sidecars_reference_body_sha256": prerequisites[
            "sidecar_reference"
        ]["body_sha256"],
        "source_size_preflight": dict(source_sizes),
        "campaigns": [dataclasses_asdict(spec) for spec in specs],
        "worker_policy": (
            "one-or-two-independent-subprocesses;receipt-order-specification;"
            "single-thread-blas"
        ),
        "protected_tests_opened": False,
        "git_required": False,
        "old_worktree_required": False,
    }


def dataclasses_asdict(value: Any) -> dict[str, object]:
    # Keep this helper local instead of importing the dataclasses module solely
    # for one frozen three-field record.
    if not isinstance(value, CampaignSpec):
        raise WorkflowError("run start campaign specification is invalid")
    return {
        "architecture": value.architecture,
        "arm": value.arm,
        "measured_source_ascii_characters": value.measured_source_ascii_characters,
    }


def _load_run_reference(
    output_directory: pathlib.Path, reference_path: pathlib.Path,
) -> tuple[pathlib.Path, dict[str, Any]]:
    _payload, reference = _load_canonical_document(
        reference_path, "compact run reference"
    )
    _verify_hashed_document(
        reference, schema=RUN_REFERENCE_SCHEMA, label="compact run reference"
    )
    receipt = reference.get("receipt")
    if (
        set(reference) != {
            "schema", "campaign_id", "bundle_body_sha256",
            "run_start_body_sha256", "receipt", "protected_tests_opened",
            "body_sha256",
        }
        or reference.get("campaign_id") != CAMPAIGN_ID
        or reference.get("protected_tests_opened") is not False
        or not isinstance(receipt, dict)
        or set(receipt) != {"path", "sha256", "body_sha256", "bytes"}
    ):
        raise WorkflowError("compact run reference contract changed")
    receipt_path = _run_output_path(
        output_directory, receipt["path"], "run receipt"
    )
    if (
        not receipt_path.is_file()
        or receipt_path.stat().st_size != receipt["bytes"]
        or sha256_file(receipt_path) != receipt["sha256"]
    ):
        raise WorkflowError("compact run receipt reference changed")
    return receipt_path, reference


def run_family(
    bundle_manifest: pathlib.Path,
    output_directory: pathlib.Path,
    *,
    resume: bool = False,
    workers: int = 1,
) -> dict[str, Any]:
    if workers not in {1, 2}:
        raise WorkflowError("run workers must be one or two")
    _reject_path_markers(bundle_manifest, "bundle manifest")
    bundle_manifest = bundle_manifest.resolve()
    output_directory = output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    with _exclusive_run_lock(output_directory):
        components = _lazy_runtime_components()
        try:
            # Unlike ``verify_bundle``, this validates only the manifest and
            # subsequently opened unprotected routes.  Protected tests remain
            # completely unopened throughout the family run.
            bundle = components.trainer.FrozenBundle.load(bundle_manifest)
        except Exception as error:
            raise WorkflowError("could not open compact frozen bundle") from error
        reference_path = output_directory / "run-state/run-reference.json"
        if reference_path.exists():
            if not resume:
                raise WorkflowError("completed family run requires --resume")
            return verify_family_run(
                bundle_manifest,
                output_directory,
                reference_path=reference_path,
                components=components,
                bundle=bundle,
            )

        tools = _tool_bindings()
        bundle_binding = _bundle_run_binding(bundle_manifest, bundle)
        prerequisites = _ensure_run_prerequisites(
            components, bundle, output_directory
        )
        source_sizes = measure_architecture_source_sizes(
            components,
            bundle.body_sha256,
            output_directory / "source-size-probes",
        )
        specs = family_campaign_specs(source_sizes)
        expected_names = {
            *(f"{architecture}--{arm}"
              for architecture in ARCHITECTURE_ORDER
              for arm in DEPLOYMENT_ARMS),
            "compact-8x8--rank4-control",
        }
        if {spec.name for spec in specs} != expected_names:
            raise WorkflowError("not all compact family architectures fit source cap")
        start = _body_hashed(_run_start_body(
            bundle_binding=bundle_binding,
            tools=tools,
            prerequisites=prerequisites,
            source_sizes=source_sizes,
            specs=specs,
        ))
        start_path = output_directory / "run-state/run-start.json"
        if start_path.exists():
            if not resume:
                raise WorkflowError("interrupted family run requires --resume")
            _existing_payload, existing = _load_canonical_document(
                start_path, "compact run start"
            )
            if existing != start:
                raise WorkflowError("compact run resume binding changed")
        else:
            _write_stable_document(start_path, start, "compact run start")

        def invoke(spec: CampaignSpec) -> dict[str, object]:
            return _run_training_process(
                spec,
                bundle_manifest=bundle_manifest,
                output_directory=output_directory,
                input_audit=prerequisites["input_audit_path"],
                sidecar_index=prerequisites["sidecar_index_path"],
                resume=resume,
            )

        process_results = _execute_campaigns(
            specs, workers=workers, runner=invoke
        )
        campaign_results = [
            _campaign_result(
                components, bundle, output_directory, process_result
            )
            for process_result in process_results
        ]
        if _tool_bindings() != tools:
            raise WorkflowError("run producer changed while training")
        start_payload = start_path.read_bytes()
        body: dict[str, object] = {
            "schema": RUN_RECEIPT_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "bundle": bundle_binding,
            "run_start": {
                "path": _relative_to(start_path, output_directory, "run start"),
                "sha256": sha256_bytes(start_payload),
                "body_sha256": start["body_sha256"],
                "bytes": len(start_payload),
            },
            "tools": tools,
            "input_audit_reference": prerequisites["input_audit_reference"],
            "teacher_sidecars_reference": prerequisites["sidecar_reference"],
            "source_size_preflight": source_sizes,
            "campaigns": campaign_results,
            "campaign_order": [spec.name for spec in specs],
            "worker_policy": start["worker_policy"],
            "all_seven_campaigns_complete": len(campaign_results) == 7,
            "protected_tests_opened": False,
            "protected_tests_locked": True,
            "status": "offline-family-trained-selection-artifacts-not-game-gated",
            "game_gated": False,
            "external_upload": False,
            "replace_rank4": False,
            "rank1_claim": False,
        }
        receipt = _body_hashed(body)
        receipt_path = _write_content_addressed(
            output_directory / "run-receipts",
            receipt,
            ".run-receipt.json",
        )
        reference = _body_hashed({
            "schema": RUN_REFERENCE_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "bundle_body_sha256": bundle.body_sha256,
            "run_start_body_sha256": start["body_sha256"],
            "receipt": {
                "path": _relative_to(receipt_path, output_directory, "run receipt"),
                "sha256": sha256_file(receipt_path),
                "body_sha256": receipt["body_sha256"],
                "bytes": receipt_path.stat().st_size,
            },
            "protected_tests_opened": False,
        })
        _write_stable_document(
            reference_path, reference, "compact run reference"
        )
        return verify_family_run(
            bundle_manifest,
            output_directory,
            reference_path=reference_path,
            components=components,
            bundle=bundle,
        )


def verify_family_run(
    bundle_manifest: pathlib.Path,
    output_directory: pathlib.Path,
    *,
    reference_path: pathlib.Path | None = None,
    receipt_path: pathlib.Path | None = None,
    components: RuntimeComponents | None = None,
    bundle: Any | None = None,
) -> dict[str, Any]:
    if reference_path is None and receipt_path is None:
        reference_path = output_directory / "run-state/run-reference.json"
    bundle_manifest = bundle_manifest.resolve()
    output_directory = output_directory.resolve()
    components = components or _lazy_runtime_components()
    if bundle is None:
        try:
            bundle = components.trainer.FrozenBundle.load(bundle_manifest)
        except Exception as error:
            raise WorkflowError("could not validate run bundle manifest") from error
    reference: dict[str, Any] | None = None
    referenced_receipt: pathlib.Path | None = None
    if reference_path is not None:
        reference_path = reference_path.resolve()
        referenced_receipt, reference = _load_run_reference(
            output_directory, reference_path
        )
    if receipt_path is None:
        receipt_path = referenced_receipt
    elif referenced_receipt is not None and receipt_path.resolve() != referenced_receipt:
        raise WorkflowError("run receipt and reference disagree")
    if receipt_path is None:
        raise WorkflowError("run verification has no receipt")
    receipt_path = receipt_path.resolve()
    payload, receipt = _load_canonical_document(
        receipt_path, "compact family run receipt"
    )
    if receipt_path.name != f"{sha256_bytes(payload)}.run-receipt.json":
        raise WorkflowError("family run receipt is not content addressed")
    _verify_hashed_document(
        receipt, schema=RUN_RECEIPT_SCHEMA, label="family run receipt"
    )
    expected_fields = {
        "schema", "campaign_id", "bundle", "run_start", "tools",
        "input_audit_reference", "teacher_sidecars_reference",
        "source_size_preflight", "campaigns", "campaign_order",
        "worker_policy", "all_seven_campaigns_complete",
        "protected_tests_opened", "protected_tests_locked", "status",
        "game_gated", "external_upload", "replace_rank4", "rank1_claim",
        "body_sha256",
    }
    campaigns = receipt.get("campaigns")
    if (
        set(receipt) != expected_fields
        or receipt.get("campaign_id") != CAMPAIGN_ID
        or receipt.get("bundle") != _bundle_run_binding(bundle_manifest, bundle)
        or receipt.get("tools") != _tool_bindings()
        or receipt.get("all_seven_campaigns_complete") is not True
        or receipt.get("protected_tests_opened") is not False
        or receipt.get("protected_tests_locked") is not True
        or receipt.get("game_gated") is not False
        or receipt.get("external_upload") is not False
        or receipt.get("replace_rank4") is not False
        or receipt.get("rank1_claim") is not False
        or receipt.get("status")
        != "offline-family-trained-selection-artifacts-not-game-gated"
        or not isinstance(campaigns, list)
        or len(campaigns) != 7
    ):
        raise WorkflowError("family run receipt policy or binding changed")
    start_record = receipt.get("run_start")
    if not isinstance(start_record, dict) or set(start_record) != {
        "path", "sha256", "body_sha256", "bytes"
    }:
        raise WorkflowError("family run start binding is malformed")
    start_path = _run_output_path(
        output_directory, start_record["path"], "run start"
    )
    start_payload, start = _load_canonical_document(
        start_path, "compact run start"
    )
    _verify_hashed_document(start, schema=RUN_START_SCHEMA, label="compact run start")
    if (
        len(start_payload) != start_record["bytes"]
        or sha256_bytes(start_payload) != start_record["sha256"]
        or start.get("body_sha256") != start_record["body_sha256"]
        or start.get("protected_tests_opened") is not False
        or start.get("tools") != receipt["tools"]
        or start.get("bundle") != receipt["bundle"]
    ):
        raise WorkflowError("family run start content changed")
    audit_path, audit_reference = _validate_prerequisite_reference(
        components,
        bundle,
        output_directory,
        output_directory / "run-state/input-audit.reference.json",
        "input-audit",
    )
    sidecar_path, sidecar_reference = _validate_prerequisite_reference(
        components,
        bundle,
        output_directory,
        output_directory / "run-state/teacher-sidecars.reference.json",
        "teacher-sidecars",
    )
    del audit_path, sidecar_path
    if (
        receipt.get("input_audit_reference") != audit_reference
        or receipt.get("teacher_sidecars_reference") != sidecar_reference
        or start.get("input_audit_reference_body_sha256")
        != audit_reference["body_sha256"]
        or start.get("teacher_sidecars_reference_body_sha256")
        != sidecar_reference["body_sha256"]
    ):
        raise WorkflowError("family run prerequisite binding changed")
    measured = measure_architecture_source_sizes(
        components,
        bundle.body_sha256,
        output_directory / "source-size-probes",
    )
    if (
        receipt.get("source_size_preflight") != measured
        or start.get("source_size_preflight") != measured
    ):
        raise WorkflowError("family source size preflight changed")
    specs = family_campaign_specs(measured)
    order = [spec.name for spec in specs]
    if (
        receipt.get("campaign_order") != order
        or start.get("campaigns") != [dataclasses_asdict(spec) for spec in specs]
        or len(order) != 7
    ):
        raise WorkflowError("family campaign roster changed")
    expected_campaigns = []
    for spec, record in zip(specs, campaigns, strict=True):
        if not isinstance(record, dict):
            raise WorkflowError("family campaign receipt is malformed")
        campaign_output = _run_output_path(
            output_directory, record.get("campaign_output"), "campaign output"
        )
        selection_record = record.get("selection")
        if not isinstance(selection_record, dict):
            raise WorkflowError("family selection record is malformed")
        selection_path = _run_output_path(
            output_directory, selection_record.get("path"), "campaign selection"
        )
        expected_campaigns.append(_campaign_result(
            components,
            bundle,
            output_directory,
            {
                "spec": spec,
                "campaign_output": campaign_output,
                "selection_path": selection_path,
                "single_thread_blas": True,
            },
        ))
    if campaigns != expected_campaigns:
        raise WorkflowError("family campaign selection or source binding changed")
    if reference is not None and (
        reference.get("bundle_body_sha256") != bundle.body_sha256
        or reference.get("run_start_body_sha256") != start["body_sha256"]
        or reference.get("receipt", {}).get("body_sha256")
        != receipt["body_sha256"]
    ):
        raise WorkflowError("family run reference disagrees with receipt")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze-inputs")
    freeze.add_argument("--source-campaign", type=pathlib.Path, required=True)
    freeze.add_argument("--output-directory", type=pathlib.Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--bundle-manifest", type=pathlib.Path, required=True)
    run.add_argument("--output-directory", type=pathlib.Path, required=True)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--workers", type=int, choices=(1, 2), default=1)
    verify = commands.add_parser("verify")
    verify.add_argument("--bundle-manifest", type=pathlib.Path, required=True)
    verify.add_argument("--run-output-directory", type=pathlib.Path)
    verify.add_argument("--run-reference", type=pathlib.Path)
    verify.add_argument("--run-receipt", type=pathlib.Path)
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "freeze-inputs":
            result = freeze_inputs(arguments.source_campaign,
                                   arguments.output_directory)
        elif arguments.command == "run":
            result = run_family(
                arguments.bundle_manifest,
                arguments.output_directory,
                resume=arguments.resume,
                workers=arguments.workers,
            )
        else:
            bundle_result = verify_bundle(arguments.bundle_manifest)
            run_requested = any((
                arguments.run_output_directory,
                arguments.run_reference,
                arguments.run_receipt,
            ))
            if run_requested:
                if arguments.run_output_directory is None:
                    raise WorkflowError(
                        "run verification requires --run-output-directory"
                    )
                run_result = verify_family_run(
                    arguments.bundle_manifest,
                    arguments.run_output_directory,
                    reference_path=arguments.run_reference,
                    receipt_path=arguments.run_receipt,
                )
                result = {"bundle": bundle_result, "run": run_result}
            else:
                result = bundle_result
    except (OSError, WorkflowError) as error:
        parser.exit(1, f"compact value-BFM workflow failure: {error}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
