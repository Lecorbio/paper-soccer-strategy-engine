#!/usr/bin/env python3
"""Freeze and verify the compact post-campaign publication bundle.

The freezer reads an explicit allowlist of terminal campaign artifacts only:
four merged panel reports, the stage-20 latency report, terminal decisions,
receipts, and retention metrics.  It never walks the campaign tree and never
opens models, labels, shards, NPZs, TSVs, or old worktrees.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import pathlib
import re
import stat
import tempfile
from collections.abc import Mapping, Sequence
from typing import Any


IMPLEMENTATION_COMMIT = "e91e09f5a6d5f1278f7e4a919a8819e1ceeab2cb"
BASE_COMMIT = "c2b06168676bfa0ea7e600da042710b29f3089c5"
CAMPAIGN_ID = "large-teacher-campaign-20260828-v1"
FULL_CAMPAIGN_ID = "large-teacher-full-20260828-v1"
ORIGINAL_RETENTION_REFERENCE_SHA256 = (
    "bfcc1755ab9b71261bedc9b9c9b59e38e3d440d7c80e7056a9f0bc812ffc9c80"
)
ACCEPTED_RUNTIME_SHA256 = (
    "f7bdb201a377c04531f1ba98fd73457f7f77961aa0f0f9b1ac32c59b6e85ee75"
)
ACCEPTED_RUNTIME_MANIFEST_SHA256 = (
    "9222b43d46d4e8ae3e7211f429fa306688a2917c7795f26e7514d7a41314ac95"
)
PILOT_TEACHER_RUNTIME_SHA256 = (
    "6cafef972aef2b6495ce486b3fb55b9b6b5da8e2593ba0966c2e454e8bfbca86"
)
OPENING_BANK_SHA256 = (
    "8348b44eb765013f3cdad495a0307f6ae0f5103761012ed0f3c690fda1a0fc01"
)
TRANSITION_AMENDMENT_SHA256 = (
    "6bc1636fb8ceff3c72dbd2a895599dc255ee3a929b0c1be3141d2476e3f4e681"
)
TRANSITION_ARCHIVE_SHA256 = (
    "fbd3ce8e4fa5b7f8a02aee92a0e6a7a38d48d13dd95a9073697e7df57f62528f"
)
COMPATIBILITY_QUALIFICATION_SHA256 = (
    "2c16ed3d3421b283b061c5d282ff39ab4bdca08536140e0ecda4ddd6481e8279"
)
RUNNER_REPAIR_AMENDMENT_SHA256 = (
    "763e150b1b78a7d886b9aa26ad7310befb6a0b01fc547e6555fbf99c5490cfab"
)
RUNNER_FAILURE_STATUS_SHA256 = (
    "625903e076c5d93f19e8df671324d3646ce002b441c1375bf6ddd8f8dc389270"
)
BUNDLE_DIRECTORY = pathlib.Path(__file__).resolve().parent
REPOSITORY = BUNDLE_DIRECTORY.parents[1]
DEFAULT_CAMPAIGN = (
    REPOSITORY / "results/large_teacher" / CAMPAIGN_ID
)

RESULTS_SCHEMA = "papersoccer.large-teacher-public-compact-results.v1"
OUTCOMES_SCHEMA = "papersoccer.large-teacher-public-paired-outcomes.v1"
MANIFEST_SCHEMA = "papersoccer.large-teacher-public-manifest.v1"

PANELS = ("matched", "pilot-teacher", "rank4", "jacek-nn")
PANEL_POLICY = {
    "matched": {"minimum_wins": 527, "minimum_per_color": 260},
    "pilot-teacher": {"minimum_wins": 527, "minimum_per_color": 260},
    "rank4": {"minimum_wins": 501, "minimum_per_color": 238},
    "jacek-nn": {"minimum_wins": 501, "minimum_per_color": 238},
}

SOURCE_FILES = {
    "final_summary": "final-summary.json",
    "acceptance": "teacher-candidate-accepted.json",
    "student_handoff": "compact-student-handoff.json",
    "decision": "full/decision.json",
    "stage19_receipt": "full/receipts/19-game-gates.json",
    "stage20_receipt": "full/receipts/20-latency-audit.json",
    "stage21_receipt": "full/receipts/21-decision.json",
    "latency_report": "full/latency-audit.json",
    "anchor_phase_actor": "full/anchor-metrics.json",
    "anchor_original_incumbent": "full/anchor-metrics-original-incumbent.json",
    **{
        f"panel_{name.replace('-', '_')}": f"full/game-gates/{name}.json"
        for name in PANELS
    },
}

EXPECTED_SOURCE_ARTIFACTS = {
    "acceptance": (2951, "5b17a4dbd72578d57baf146a3732759eaaa34eca1f61fc3586390740984fbafb"),
    "anchor_original_incumbent": (2497, "5ef3359dadc6136e48c35e0df6c9bba42d59af86ad07af8459c07feaf5ebe820"),
    "anchor_phase_actor": (2491, "ece4de9884efea45b1e0bd60a174326a31934748c9e708d52e996a71e68deb63"),
    "decision": (3280, "bf655f979afd391687a3aa4d1c6e09ce306e5c327fd760d388f4301737221ad5"),
    "final_summary": (14189, "5766424a7a84ade4481b9fa609525fc56a46d72ba8c83923c1d73c0a94180ef0"),
    "latency_report": (30041, "115a80f4ca25ecd204452285b7603c91939259802dabf2b6b934565938d5e37c"),
    "panel_jacek_nn": (1824006, "a8348e54c67e816940db00cfcf19bcd25b7c67ed324bb8f73d9780c8daab5ec6"),
    "panel_matched": (1804288, "f9c2f75769a14c9400cb7919a1dd8d49fd7d2791d5df5d30ceb2d6858bcf5914"),
    "panel_pilot_teacher": (1806145, "895f649e9391fbbf8ed700a1b0006d5dc40d10e3f8df4b5de117ba20738bcef4"),
    "panel_rank4": (1800070, "ee1f80250a7fcfa62a23271bc157cd45b0f75a4fcb86bc880c1f8cee5677b1ef"),
    "stage19_receipt": (310031, "428b71ac40d9da457c954f3afb0df4de6de84a2ffa7d738e4af52f7e3bb5fa6c"),
    "stage20_receipt": (7368, "3fee752f15e4ee663327bdf22af46d8b6deb78cca45a2878fbe0c3c4d1377bc0"),
    "stage21_receipt": (12333, "47a95aac198a1e40ca59bedced08dd5e5286443bbd1213133b5130d409aa1381"),
    "student_handoff": (6389, "1c4e026b013c705ac4ed238a8aa234e035fea028426e7e24e0aae1bb6a8d5459"),
}

BUNDLE_FILE_NAMES = {
    "REPORT.md",
    "publish.py",
    "compact_results.json",
    "paired_outcomes.json",
    "manifest.json",
}
STATIC_BUNDLE_FILE_NAMES = {"REPORT.md", "publish.py"}
GENERATED_FILE_NAMES = (
    "compact_results.json",
    "paired_outcomes.json",
    "manifest.json",
)
DERIVED_PATHS = {
    "report": "REPORT.md",
    "publisher": "publish.py",
    "compact_results": "compact_results.json",
    "paired_outcomes": "paired_outcomes.json",
}
EXPECTED_DERIVED_CONTENT = {
    "report": (
        6_028,
        "4e3b08bb45a1a6c42007b4e92de03903d11534dbec8ecf49088aecb06490770a",
    ),
    "compact_results": (
        8_854,
        "9b99ebfd977b26461adc32e6f87a7aed2022f45b2acab3adf22f7a2364c53602",
    ),
    "paired_outcomes": (
        173_698,
        "384ca42adef4178f1be68691cecf26e210d1285a94e827d0c62c78e684df109d",
    ),
}

EXPECTED_FROZEN_CONFIGURATION = {
    "candidate_tree_nodes": 1_000_000,
    "control_tree_nodes": 1_000_000,
    "exploration": 0.5,
    "fpu": 0.5,
    "max_actions": 250,
    "max_partial_paths": 50_000,
    "max_turns": 320,
    "opening_bank_seed": 2026082507,
    "opening_bank_sha256": OPENING_BANK_SHA256,
    "opening_plies": 12,
    "pairs": 500,
    "panels": list(PANELS),
    "single_thread": True,
    "time_ms": 980,
}
EXPECTED_PILOT_TRUTH = {
    "pilot_passed": False,
    "pilot_20_ms_passed": False,
    "teacher_only_override": True,
    "bypassed_errors": [
        "matched primary strength gate failed",
        "incumbent primary strength gate failed",
    ],
}
EXPECTED_CLAIMS = {
    "local_teacher_candidate_accepted": True,
    "canonical_promotion_eligible": False,
    "canonical_promotion_performed": False,
    "deployment_performed": False,
    "rank4_replaced": False,
    "leaderboard_claim": False,
}
EXPECTED_PUBLICATION_CONTEXT = {
    "source_campaign_publication": None,
    "source_campaign_external_upload": False,
    "authorization": "separate explicit post-campaign publication request",
    "derived_evidence_only": True,
    "source_receipts_modified": False,
    "models_included": False,
    "datasets_included": False,
    "raw_transcripts_included": False,
}
EXPECTED_SOURCE_CLOSURE = {
    "candidate_source_sha256": "01c3fb4503cd984f63ca34e0b4442d5ae634d9d05e3522267991482a401e6e68",
    "comparison_executable_sha256": "a0d150589dbf7b69f35e2a34da359d79ab8c3782f0eea22b6264af1fc3a1e40d",
    "comparison_source_sha256": "0c22cb6276a4fbbec048cec3f897ade8ae0e8d506f6c2e0090ff1585d34d57dd",
    "jacek_nn_adapter_sha256": "871c1821531ee031032b98403c8f0a6fd6ce003cfcb85ba766c717133025b669",
    "jacek_nn_control_sha256": "fb570f7d60157ad1681569011b4249a5db415c1aeca6f665936b26ba5cc52102",
    "jacek_nn_engine_sha256": "d16bc7028952ab3de1842d58b0933dd440a16e8e39ff650ce29502cfff959a8a",
    "jacek_nn_source_sha256": "212fd66669807f9782e183f2240b28d62b68741cefa76fe149234b96250c2983",
    "neural_puct_adapter_sha256": "4d4dbd37d01b29551384a2aed75a5189c9fa6c392e762125c0f0fd259c4ec158",
    "neural_puct_control_sha256": "11e33be069b6e01d8435ec81e93d95663170466ffde50c4ab2fcdccc60e3325c",
    "neural_puct_engine_sha256": "c18904c2aacd3bc0d33c83a1aff55b8f3df22e1c61b2d377d14b456b3324edc9",
    "rank4_adapter_sha256": "e910e5a97c6c08022546b4f806cc8f85da413328228ca0480129dec75e64fe54",
    "rank4_control_sha256": "5c7ebbb38e3b08940eb26ca8cd7585dc5cbce5ad949dfd595bfb0eaab1de53c9",
    "rank4_engine_sha256": "9276c258cd613b6b78948aeb8aa2649851d226947d419f84a351968a9035c0ad",
    "shared_core_sha256": "10c09564657c36fbff991608f0d7710a7deb31d1b4116e10b6fba7c8f7cd42d4",
}
EXPECTED_SCOPE = {
    "self_contained_threshold_verification": True,
    "campaign_rerun_possible_without_ignored_inputs": False,
    "contains_models": False,
    "contains_datasets": False,
    "contains_raw_transcripts": False,
    "contains_raw_shards": False,
}


class PublicationError(RuntimeError):
    pass


_PRE_OPEN_HOOK: Any = None
_READ_OBSERVER: Any = None


@dataclasses.dataclass(frozen=True)
class RegularFile:
    path: pathlib.Path
    payload: bytes
    sha256: str
    size: int
    mtime_ns: int


def read_regular_file(
    root: pathlib.Path, relative: str, label: str
) -> RegularFile:
    """Open one contained regular file without following any path-component link."""

    root = pathlib.Path(root)
    pure = pathlib.PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or any(
        part in {"", ".", ".."} for part in pure.parts
    ):
        raise PublicationError(f"{label} relative path is unsafe")

    required_flags = ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW")
    if any(not hasattr(os, name) for name in required_flags):
        raise PublicationError("secure no-follow file access is unavailable")
    directory_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    file_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptors: list[int] = []
    try:
        try:
            root_before = root.lstat()
            if stat.S_ISLNK(root_before.st_mode) or not stat.S_ISDIR(
                root_before.st_mode
            ):
                raise PublicationError(f"{label} root is not a regular directory")
            current_fd = os.open(root, directory_flags)
        except OSError as error:
            raise PublicationError(f"{label} root is unavailable: {root}") from error
        descriptors.append(current_fd)
        root_opened = os.fstat(current_fd)
        if (
            not stat.S_ISDIR(root_opened.st_mode)
            or (root_before.st_dev, root_before.st_ino)
            != (root_opened.st_dev, root_opened.st_ino)
        ):
            raise PublicationError(f"{label} root changed before open")

        for part in pure.parts[:-1]:
            try:
                before = os.stat(part, dir_fd=current_fd, follow_symlinks=False)
                if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                    raise PublicationError(
                        f"{label} parent is not a regular directory: {relative}"
                    )
                child_fd = os.open(part, directory_flags, dir_fd=current_fd)
            except OSError as error:
                raise PublicationError(
                    f"{label} parent is unavailable: {relative}"
                ) from error
            descriptors.append(child_fd)
            opened = os.fstat(child_fd)
            if (
                not stat.S_ISDIR(opened.st_mode)
                or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
            ):
                raise PublicationError(f"{label} parent changed before open")
            current_fd = child_fd

        final = pure.parts[-1]
        try:
            before = os.stat(final, dir_fd=current_fd, follow_symlinks=False)
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
                raise PublicationError(f"{label} is not a regular file: {relative}")
            if _PRE_OPEN_HOOK is not None:
                _PRE_OPEN_HOOK(root, pure.as_posix())
            file_fd = os.open(final, file_flags, dir_fd=current_fd)
        except OSError as error:
            raise PublicationError(f"{label} is unavailable: {relative}") from error
        descriptors.append(file_fd)
        opened = os.fstat(file_fd)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
        ):
            raise PublicationError(f"{label} changed before open")
        if _READ_OBSERVER is not None:
            _READ_OBSERVER(pure.as_posix(), opened.st_dev, opened.st_ino)
        chunks = []
        while chunk := os.read(file_fd, 1024 * 1024):
            chunks.append(chunk)
        payload = b"".join(chunks)
        after = os.fstat(file_fd)
        stable_identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        if stable_identity != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) or len(payload) != opened.st_size:
            raise PublicationError(f"{label} changed while being read")
        return RegularFile(
            path=root.joinpath(*pure.parts),
            payload=payload,
            sha256=sha256_bytes(payload),
            size=opened.st_size,
            mtime_ns=opened.st_mtime_ns,
        )
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def validate_bundle_roster(
    directory: pathlib.Path, *, allow_generated_missing: bool = False
) -> dict[str, RegularFile]:
    """Read a stable exact bundle snapshot through one held directory fd."""

    directory = pathlib.Path(directory)
    required_flags = ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW")
    if any(not hasattr(os, name) for name in required_flags):
        raise PublicationError("secure no-follow bundle access is unavailable")
    directory_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    file_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptors: list[int] = []
    try:
        root_info = directory.lstat()
        if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
            raise PublicationError("bundle directory is not a regular directory")
        root_fd = os.open(directory, directory_flags)
    except OSError as error:
        raise PublicationError("bundle directory is unavailable") from error
    descriptors.append(root_fd)
    try:
        root_opened = os.fstat(root_fd)
        if (
            not stat.S_ISDIR(root_opened.st_mode)
            or (root_info.st_dev, root_info.st_ino)
            != (root_opened.st_dev, root_opened.st_ino)
        ):
            raise PublicationError("bundle directory changed before open")
        names_before = set(os.listdir(root_fd))
        if allow_generated_missing:
            if (
                not STATIC_BUNDLE_FILE_NAMES <= names_before
                or not names_before <= BUNDLE_FILE_NAMES
            ):
                raise PublicationError(
                    "bundle staging roster contains missing or extra entries"
                )
        elif names_before != BUNDLE_FILE_NAMES:
            raise PublicationError("bundle file roster is not exact")
        directory_identity = (
            root_opened.st_dev,
            root_opened.st_ino,
            root_opened.st_mtime_ns,
            root_opened.st_ctime_ns,
        )
        blobs: dict[str, RegularFile] = {}
        for name in sorted(names_before):
            label = f"bundle artifact {name}"
            try:
                before = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
                if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
                    raise PublicationError(f"{label} is not a regular file")
                if _PRE_OPEN_HOOK is not None:
                    _PRE_OPEN_HOOK(directory, name)
                file_fd = os.open(name, file_flags, dir_fd=root_fd)
            except OSError as error:
                raise PublicationError(f"{label} is unavailable") from error
            descriptors.append(file_fd)
            opened = os.fstat(file_fd)
            if (
                not stat.S_ISREG(opened.st_mode)
                or (before.st_dev, before.st_ino)
                != (opened.st_dev, opened.st_ino)
            ):
                raise PublicationError(f"{label} changed before open")
            if _READ_OBSERVER is not None:
                _READ_OBSERVER(name, opened.st_dev, opened.st_ino)
            chunks = []
            while chunk := os.read(file_fd, 1024 * 1024):
                chunks.append(chunk)
            payload = b"".join(chunks)
            after = os.fstat(file_fd)
            stable_identity = (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_ctime_ns,
            )
            if stable_identity != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ) or len(payload) != opened.st_size:
                raise PublicationError(f"{label} changed while being read")
            blobs[name] = RegularFile(
                path=directory / name,
                payload=payload,
                sha256=sha256_bytes(payload),
                size=opened.st_size,
                mtime_ns=opened.st_mtime_ns,
            )
        names_after = set(os.listdir(root_fd))
        root_after = os.fstat(root_fd)
        if names_after != names_before:
            raise PublicationError("bundle file roster changed during verification")
        if directory_identity != (
            root_after.st_dev,
            root_after.st_ino,
            root_after.st_mtime_ns,
            root_after.st_ctime_ns,
        ):
            raise PublicationError("bundle directory changed during verification")
        return blobs
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    return read_regular_file(path.parent, path.name, f"hash source {path.name}").sha256


def parse_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise PublicationError(f"could not parse {label}") from error
    if not isinstance(value, dict):
        raise PublicationError(f"{label} must be an object")
    return value


def load_json(path: pathlib.Path, label: str) -> dict[str, Any]:
    opened = read_regular_file(path.parent, path.name, label)
    return parse_json_bytes(opened.payload, label)


def body_hashed(body: Mapping[str, object]) -> dict[str, object]:
    result = dict(body)
    result["body_sha256"] = sha256_bytes(canonical_json_bytes(body))
    return result


def verify_body_hash(value: Mapping[str, object], schema: str, label: str) -> None:
    body = dict(value)
    claimed = body.pop("body_sha256", None)
    if body.get("schema") != schema or claimed != sha256_bytes(
        canonical_json_bytes(body)
    ):
        raise PublicationError(f"{label} body hash is invalid")


def atomic_write(path: pathlib.Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = pathlib.Path(raw)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_exact(path: pathlib.Path, value: Mapping[str, object], label: str) -> None:
    payload = canonical_json_bytes(value)
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise PublicationError(f"existing {label} is not a regular file")
        if read_regular_file(path.parent, path.name, label).payload == payload:
            return
    atomic_write(path, payload)


def artifact_from_opened(opened: RegularFile, relative: str) -> dict[str, object]:
    return {"path": relative, "sha256": opened.sha256, "bytes": opened.size}


def _snapshot_identity(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise PublicationError(f"{label} identity is missing")
    if (
        value.get("kind") != "file"
        or not isinstance(value.get("sha256"), str)
        or type(value.get("bytes")) is not int
    ):
        raise PublicationError(f"{label} identity is malformed")
    return {"sha256": value["sha256"], "bytes": value["bytes"]}


def _p99(samples: Sequence[float]) -> float:
    if not samples:
        raise PublicationError("latency sample list is empty")
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, math.ceil(0.99 * len(ordered)) - 1)]


def _panel_counts(pairs: Sequence[object]) -> dict[str, object]:
    wins = illegal = unfinished = games = 0
    colors = [0, 0]
    state_identities: set[str] = set()
    for expected_index, record in enumerate(pairs):
        if not isinstance(record, list) or len(record) != 3:
            raise PublicationError("compact pair record is malformed")
        pair_index, state_identity, pair_games = record
        if pair_index != expected_index or not isinstance(state_identity, str):
            raise PublicationError("compact pair identity is malformed")
        if state_identity in state_identities:
            raise PublicationError("compact opening state identity is duplicate")
        state_identities.add(state_identity)
        if not isinstance(pair_games, list) or len(pair_games) != 2:
            raise PublicationError("compact pair does not contain two games")
        seen_colors = []
        for game in pair_games:
            if not isinstance(game, list) or len(game) != 4:
                raise PublicationError("compact game record is malformed")
            candidate_player, winner, is_illegal, is_unfinished = game
            if candidate_player not in (0, 1):
                raise PublicationError("candidate color is invalid")
            if winner not in (0, 1, None):
                raise PublicationError("winner is invalid")
            if type(is_illegal) is not bool or type(is_unfinished) is not bool:
                raise PublicationError("game safety flags are invalid")
            if is_unfinished != (winner not in (0, 1)):
                raise PublicationError("unfinished flag contradicts winner")
            seen_colors.append(candidate_player)
            games += 1
            illegal += int(is_illegal)
            unfinished += int(is_unfinished)
            if winner == candidate_player:
                wins += 1
                colors[candidate_player] += 1
        if sorted(seen_colors) != [0, 1]:
            raise PublicationError("pair does not swap candidate colors")
    return {
        "games": games,
        "wins": wins,
        "losses": games - wins - unfinished,
        "illegal": illegal,
        "unfinished": unfinished,
        "colors": colors,
    }


def _normalized_transition(summary: Mapping[str, object]) -> dict[str, object]:
    full = summary.get("full")
    transition = full.get("environment_transition") if isinstance(full, dict) else None
    if not isinstance(transition, dict):
        raise PublicationError("terminal summary has no environment transition")
    repair = transition.get("runner_repair")
    if not isinstance(repair, dict):
        raise PublicationError("terminal summary has no runner repair chain")
    execution = transition.get("execution_environment")
    if not isinstance(execution, dict):
        raise PublicationError("terminal execution environment is missing")
    return {
        "classification": transition.get("classification"),
        "previous_platform": transition.get("previous_platform"),
        "resumed_platform": transition.get("resumed_platform"),
        "execution_environment": {
            key: execution.get(key)
            for key in (
                "platform",
                "machine",
                "python_implementation",
                "python_version",
                "numpy_version",
                "python_executable_sha256",
            )
        },
        "pretransition_stage19_reused": transition.get(
            "pretransition_stage19_reused"
        ),
        "pretransition_stage19_classification": transition.get(
            "pretransition_stage19_classification"
        ),
        "authoritative_stage19_shards": transition.get(
            "authoritative_stage19_shards"
        ),
        "amendment_sha256": _snapshot_identity(
            transition.get("amendment"), "transition amendment"
        )["sha256"],
        "archive_manifest_sha256": _snapshot_identity(
            transition.get("archive_manifest"), "transition archive"
        )["sha256"],
        "compatibility_qualification_sha256": _snapshot_identity(
            transition.get("compatibility_qualification"),
            "compatibility qualification",
        )["sha256"],
        "runner_repair": {
            "classification": repair.get("classification"),
            "reason": "delayed jacek_replay_train import path was repaired",
            "replacement_stage19_outputs_before_repair": repair.get(
                "replacement_stage19_outputs_before_repair"
            ),
            "amendment_sha256": _snapshot_identity(
                repair.get("amendment"), "runner repair amendment"
            )["sha256"],
            "first_attempt_failure_status_sha256": _snapshot_identity(
                repair.get("first_attempt_failure_status"),
                "runner repair failure status",
            )["sha256"],
        },
    }


def _snapshot_matches_source(
    value: object, observed_artifact: Mapping[str, object], label: str
) -> None:
    expected = _snapshot_identity(value, label)
    observed = {
        "sha256": observed_artifact.get("sha256"),
        "bytes": observed_artifact.get("bytes"),
    }
    if expected != observed:
        raise PublicationError(f"{label} source binding changed")


def _transition_core_matches(
    stage_transition: object, terminal_transition: Mapping[str, object], label: str
) -> None:
    if not isinstance(stage_transition, dict):
        raise PublicationError(f"{label} transition binding is missing")
    projected = {
        key: terminal_transition.get(key) for key in stage_transition
    }
    if projected != stage_transition:
        raise PublicationError(f"{label} transition differs from terminal chain")


def validate_semantic_sources(
    *,
    source: Mapping[str, Mapping[str, object]],
    source_artifacts: Mapping[str, Mapping[str, object]],
    summary: Mapping[str, object],
    acceptance: Mapping[str, object],
    handoff: Mapping[str, object],
    decision: Mapping[str, object],
    latency: Mapping[str, object],
) -> None:
    runtime = _snapshot_identity(acceptance.get("runtime"), "accepted runtime")
    runtime_manifest = _snapshot_identity(
        acceptance.get("manifest"), "accepted runtime manifest"
    )
    if (
        runtime != {"sha256": ACCEPTED_RUNTIME_SHA256, "bytes": 4_864_000}
        or runtime_manifest
        != {"sha256": ACCEPTED_RUNTIME_MANIFEST_SHA256, "bytes": 73_824}
        or acceptance.get("pilot_passed") is not False
        or acceptance.get("pilot_20_ms_passed") is not False
        or acceptance.get("canonical_promotion_eligible") is not False
        or acceptance.get("publication") is not False
        or acceptance.get("external_upload") is not False
        or acceptance.get("replace_rank4") is not False
        or acceptance.get("leaderboard_claim") is not False
        or acceptance.get("final_acceptance") != {
            "pairs_per_panel": 500,
            "time_ms": 980,
            "pilot_teacher_and_matched": {
                "minimum_wins": 527,
                "minimum_per_color": 260,
            },
            "rank4_and_external_neural": {
                "minimum_wins": 501,
                "minimum_per_color": 238,
            },
            "illegal": 0,
            "unfinished": 0,
            "maximum_ms_exclusive": 1000,
            "canonical_retention_references": [
                "pilot-teacher",
                "original-v6-reference",
            ],
        }
    ):
        raise PublicationError("accepted teacher policy or identity changed")
    _snapshot_matches_source(
        acceptance.get("decision"), source_artifacts["decision"], "accepted decision"
    )
    if _snapshot_identity(acceptance.get("launch"), "accepted launch") != {
        "sha256": "13f32e9feef97dd3911f3c71466b2ad1163f3855101eedd9ceedc94cb18b7ddd",
        "bytes": 162_071,
    } or _snapshot_identity(acceptance.get("run_start"), "accepted run start") != {
        "sha256": "f97a3407564e2fef3b02ab412510ff973977521e272e7cf9139a2c97d61e2bfb",
        "bytes": 3_168,
    }:
        raise PublicationError("launch/run-start provenance changed")

    if (
        _snapshot_identity(handoff.get("teacher_runtime"), "handoff runtime")
        != runtime
        or _snapshot_identity(handoff.get("teacher_manifest"), "handoff manifest")
        != runtime_manifest
        or handoff.get("student_training_eligible") is not True
        or handoff.get("student_training_started") is not False
        or handoff.get("canonical_promotion_eligible") is not False
        or handoff.get("publication") is not False
        or handoff.get("external_upload") is not False
        or handoff.get("replace_rank4") is not False
        or handoff.get("leaderboard_claim") is not False
    ):
        raise PublicationError("student handoff policy or identity changed")
    _snapshot_matches_source(
        handoff.get("teacher_candidate_accepted"),
        source_artifacts["acceptance"],
        "handoff acceptance",
    )
    _snapshot_matches_source(
        summary.get("teacher_candidate_accepted"),
        source_artifacts["acceptance"],
        "terminal acceptance",
    )
    _snapshot_matches_source(
        summary.get("compact_student_handoff"),
        source_artifacts["student_handoff"],
        "terminal student handoff",
    )

    phase_anchor = source["anchor_phase_actor"]
    original_anchor = source["anchor_original_incumbent"]
    if (
        phase_anchor.get("schema")
        != "papersoccer.jacek-selfsearch-anchor-metrics.v1"
        or original_anchor.get("schema")
        != "papersoccer.jacek-selfsearch-anchor-metrics.v1"
        or _snapshot_identity(phase_anchor.get("candidate"), "phase candidate")
        != runtime
        or _snapshot_identity(original_anchor.get("candidate"), "original candidate")
        != runtime
        or _snapshot_identity(phase_anchor.get("incumbent"), "phase incumbent").get(
            "sha256"
        )
        != PILOT_TEACHER_RUNTIME_SHA256
        or _snapshot_identity(
            original_anchor.get("incumbent"), "original incumbent"
        ).get("sha256")
        != ORIGINAL_RETENTION_REFERENCE_SHA256
        or phase_anchor.get("candidate_metrics") != decision.get("anchor_candidate")
        or phase_anchor.get("incumbent_metrics") != decision.get("anchor_incumbent")
        or original_anchor.get("candidate_metrics")
        != decision.get("original_anchor_candidate")
        or original_anchor.get("incumbent_metrics")
        != decision.get("original_anchor_incumbent")
        or phase_anchor.get("anchor_validation")
        != original_anchor.get("anchor_validation")
        or not isinstance(phase_anchor.get("anchor_validation"), list)
        or len(phase_anchor["anchor_validation"]) != 3
    ):
        raise PublicationError("anchor metric provenance changed")

    stage19 = source["stage19_receipt"]
    stage20 = source["stage20_receipt"]
    stage21 = source["stage21_receipt"]
    for receipt, stage, ordinal in (
        (stage19, "game-gates", 19),
        (stage20, "latency-audit", 20),
        (stage21, "decision", 21),
    ):
        if (
            receipt.get("schema")
            != "papersoccer.jacek-replay-bfm-stage-receipt.v1"
            or receipt.get("campaign_id") != FULL_CAMPAIGN_ID
            or receipt.get("stage") != stage
            or receipt.get("ordinal") != ordinal
        ):
            raise PublicationError(f"stage-{ordinal} receipt header changed")
    terminal_transition = summary.get("full", {}).get("environment_transition")
    if not isinstance(terminal_transition, dict):
        raise PublicationError("terminal transition provenance is missing")
    for receipt, label in ((stage19, "stage-19"), (stage20, "stage-20"), (stage21, "stage-21")):
        environment = receipt.get("environment")
        if not isinstance(environment, dict):
            raise PublicationError(f"{label} environment is missing")
        _transition_core_matches(
            environment.get("environment_transition"), terminal_transition, label
        )
        if environment.get("platform") != "macOS-26.6.2-arm64-arm-64bit-Mach-O":
            raise PublicationError(f"{label} platform changed")

    stage19_configuration = stage19.get("configuration")
    if stage19_configuration != {
        "pairs": 500,
        "time_ms": 980,
        "workers": 4,
        "shard_pairs": 5,
        "panels": list(PANELS),
        "bank_classification": "final",
    }:
        raise PublicationError("stage-19 configuration changed")
    stage19_inputs = stage19.get("inputs")
    stage19_outputs = stage19.get("outputs")
    stage19_result = stage19.get("result")
    if not all(isinstance(value, dict) for value in (
        stage19_inputs, stage19_outputs, stage19_result
    )):
        raise PublicationError("stage-19 receipt topology changed")
    assert isinstance(stage19_inputs, dict)
    assert isinstance(stage19_outputs, dict)
    assert isinstance(stage19_result, dict)
    if (
        _snapshot_identity(stage19_inputs.get("model"), "stage-19 model")
        != runtime
        or _snapshot_identity(stage19_inputs.get("bank"), "stage-19 bank").get(
            "sha256"
        )
        != OPENING_BANK_SHA256
        or _snapshot_identity(stage19_inputs.get("actor"), "stage-19 actor").get(
            "sha256"
        )
        != PILOT_TEACHER_RUNTIME_SHA256
        or _snapshot_identity(stage19_inputs.get("matched"), "stage-19 matched").get(
            "sha256"
        )
        != PILOT_TEACHER_RUNTIME_SHA256
        or stage19_result.get("replacement_stage19_shards") != 400
        or stage19_result.get("pretransition_stage19_reused") is not False
    ):
        raise PublicationError("stage-19 input/result binding changed")
    _transition_core_matches(
        stage19_result.get("environment_transition"), terminal_transition, "stage-19 result"
    )
    for name in PANELS:
        _snapshot_matches_source(
            stage19_outputs.get(name),
            source_artifacts[f"panel_{name.replace('-', '_')}"],
            f"stage-19 output {name}",
        )

    if stage20.get("configuration") != {
        "pairs": 10,
        "time_ms": 980,
        "workers": 1,
        "bank_classification": "final",
    }:
        raise PublicationError("stage-20 configuration changed")
    stage20_inputs = stage20.get("inputs")
    stage20_outputs = stage20.get("outputs")
    stage20_result = stage20.get("result")
    if not all(isinstance(value, dict) for value in (
        stage20_inputs, stage20_outputs, stage20_result
    )):
        raise PublicationError("stage-20 receipt topology changed")
    assert isinstance(stage20_inputs, dict)
    assert isinstance(stage20_outputs, dict)
    assert isinstance(stage20_result, dict)
    if (
        _snapshot_identity(stage20_inputs.get("model"), "stage-20 model")
        != runtime
        or _snapshot_identity(stage20_inputs.get("bank"), "stage-20 bank").get(
            "sha256"
        )
        != OPENING_BANK_SHA256
        or stage20_result.get("candidate_samples")
        != latency.get("summary", {}).get("candidate", {}).get("decisions")
        or stage20_result.get("candidate_max_ms")
        != latency.get("summary", {}).get("candidate", {}).get("max_ms")
    ):
        raise PublicationError("stage-20 input/result binding changed")
    _snapshot_matches_source(
        stage20_outputs.get("report"), source_artifacts["latency_report"], "stage-20 report"
    )

    if stage21.get("configuration") != {
        "profile": "full",
        "anchor_noninferiority": {
            "sign_accuracy_margin": 0.005,
            "weighted_huber_multiplier": 1.02,
            "references": ["phase-actor", "original-incumbent"],
        },
    }:
        raise PublicationError("stage-21 configuration changed")
    stage21_inputs = stage21.get("inputs")
    stage21_outputs = stage21.get("outputs")
    if not isinstance(stage21_inputs, dict) or not isinstance(stage21_outputs, dict):
        raise PublicationError("stage-21 receipt topology changed")
    expected_stage21_inputs = {
        "matched": "panel_matched",
        "pilot-teacher": "panel_pilot_teacher",
        "rank4": "panel_rank4",
        "jacek-nn": "panel_jacek_nn",
        "latency": "latency_report",
        "anchor_metrics": "anchor_phase_actor",
        "original_incumbent_anchor_metrics": "anchor_original_incumbent",
    }
    for receipt_label, source_label in expected_stage21_inputs.items():
        _snapshot_matches_source(
            stage21_inputs.get(receipt_label),
            source_artifacts[source_label],
            f"stage-21 input {receipt_label}",
        )
    _snapshot_matches_source(
        stage21_outputs.get("decision"), source_artifacts["decision"], "stage-21 decision"
    )
    if stage21.get("result") != decision:
        raise PublicationError("stage-21 result differs from terminal decision")


def build_publication(campaign: pathlib.Path) -> tuple[dict, dict, dict]:
    campaign = pathlib.Path(campaign)
    try:
        campaign_info = campaign.lstat()
    except OSError as error:
        raise PublicationError("campaign root is unavailable") from error
    if stat.S_ISLNK(campaign_info.st_mode) or not stat.S_ISDIR(
        campaign_info.st_mode
    ):
        raise PublicationError("campaign root is not a regular directory")
    campaign = campaign.resolve(strict=True)
    if campaign.name != CAMPAIGN_ID:
        raise PublicationError("campaign root has the wrong identity")
    lowered = str(campaign).lower()
    if any(marker in lowered for marker in ("sealed-final", "blind-label")):
        raise PublicationError("campaign path violates the publication allowlist")
    for old_identifier in ("a21d", "12e8"):
        marker = (
            os.sep + "worktrees" + os.sep + old_identifier + os.sep
        ).lower()
        if marker in lowered:
            raise PublicationError("old worktree access is forbidden")

    source_opened = {
        label: read_regular_file(campaign, relative, f"source artifact {label}")
        for label, relative in SOURCE_FILES.items()
    }
    source = {
        label: parse_json_bytes(opened.payload, label.replace("_", " "))
        for label, opened in source_opened.items()
    }
    source_artifacts = {
        label: artifact_from_opened(source_opened[label], relative)
        for label, relative in SOURCE_FILES.items()
    }
    summary = source["final_summary"]
    acceptance = source["acceptance"]
    handoff = source["student_handoff"]
    decision = source["decision"]
    latency = source["latency_report"]

    if (
        summary.get("terminal") != "teacher-candidate-accepted"
        or summary.get("student_training_eligible") is not True
        or summary.get("student_training_started") is not False
        or summary.get("pilot_passed") is not False
        or summary.get("pilot_20_ms_passed") is not False
        or summary.get("canonical_promotion_eligible") is not False
        or summary.get("external_upload") is not False
        or summary.get("rank4_replaced") is not False
        or summary.get("leaderboard_claim") is not False
        or acceptance.get("classification") != "local-teacher-candidate"
        or handoff.get("classification")
        != "local-compact-student-training-input"
        or handoff.get("student_training_eligible") is not True
        or handoff.get("student_training_started") is not False
        or decision.get("eligible_for_local_publication") is not True
        or decision.get("errors") != []
    ):
        raise PublicationError("terminal campaign policy is not publishable evidence")
    full = summary.get("full")
    if not isinstance(full, dict) or full.get("decision") != decision:
        raise PublicationError("terminal summary/decision binding changed")
    validate_semantic_sources(
        source=source,
        source_artifacts=source_artifacts,
        summary=summary,
        acceptance=acceptance,
        handoff=handoff,
        decision=decision,
        latency=latency,
    )

    panel_payload: dict[str, object] = {
        "schema": OUTCOMES_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "field_order": [
            "candidate_player",
            "winner",
            "illegal",
            "unfinished",
        ],
        "pair_record_order": ["pair_index", "opening_state_identity", "games"],
        "panels": {},
    }
    compact_panels: dict[str, object] = {}
    common_configuration: dict[str, object] | None = None
    source_closure: dict[str, object] | None = None
    opening_bank_sha256: str | None = None
    common_state_identities: list[str] | None = None

    for name in PANELS:
        report = source[f"panel_{name.replace('-', '_')}"]
        configuration = report.get("configuration")
        results = report.get("results")
        report_summary = report.get("summary")
        if (
            report.get("schema") != "papersoccer.jacek-replay-bfm-comparison.v1"
            or not isinstance(configuration, dict)
            or not isinstance(results, list)
            or len(results) != 1000
            or not isinstance(report_summary, dict)
            or configuration.get("pairs") != 500
            or configuration.get("pair_offset") != 0
            or configuration.get("time_ms") != 980
            or configuration.get("opening_bank_classification") != "final"
            or configuration.get("single_thread") is not True
            or report.get("model_sha256") != ACCEPTED_RUNTIME_SHA256
        ):
            raise PublicationError(f"{name} merged report is malformed")
        expected_opponents = {
            "matched": "jacek-replay",
            "pilot-teacher": "jacek-replay",
            "rank4": "rank4",
            "jacek-nn": "jacek-nn",
        }
        expected_control = (
            PILOT_TEACHER_RUNTIME_SHA256
            if name in {"matched", "pilot-teacher"}
            else None
        )
        if (
            configuration.get("opponent") != expected_opponents[name]
            or configuration.get("control_model_sha256") != expected_control
        ):
            raise PublicationError(f"{name} opponent/control identity changed")
        state_ids = configuration.get("opening_state_identities")
        if not isinstance(state_ids, list) or len(state_ids) != 500:
            raise PublicationError(f"{name} opening identities are incomplete")
        if common_state_identities is None:
            common_state_identities = list(state_ids)
        elif common_state_identities != state_ids:
            raise PublicationError("panel opening-state rosters differ")
        pairs = []
        for index in range(500):
            pair_games = results[index * 2 : index * 2 + 2]
            opening_names = set()
            compact_games = []
            for game in pair_games:
                if not isinstance(game, dict):
                    raise PublicationError(f"{name} game result is malformed")
                opening_names.add(game.get("opening"))
                winner = game.get("winner")
                compact_games.append(
                    [
                        game.get("candidate_player"),
                        winner,
                        game.get("illegal"),
                        winner not in (0, 1),
                    ]
                )
            if len(opening_names) != 1:
                raise PublicationError(f"{name} pair opening mismatch")
            pairs.append([index, state_ids[index], compact_games])
        counts = _panel_counts(pairs)
        expected = decision.get("counts", {}).get(name)
        expected_normalized = (
            {
                "games": expected.get("games"),
                "wins": expected.get("wins"),
                "losses": expected.get("games")
                - expected.get("wins")
                - expected.get("unfinished"),
                "illegal": expected.get("illegal"),
                "unfinished": expected.get("unfinished"),
                "colors": expected.get("colors"),
            }
            if isinstance(expected, dict)
            else None
        )
        if counts != expected_normalized:
            raise PublicationError(f"{name} compact outcome recount differs")
        report_colors = report_summary.get("colors")
        if (
            report_summary.get("games") != counts["games"]
            or report_summary.get("wins") != counts["wins"]
            or report_summary.get("losses") != counts["losses"]
            or report_summary.get("illegal") != counts["illegal"]
            or report_summary.get("unfinished") != counts["unfinished"]
            or not isinstance(report_colors, list)
            or [record.get("wins") for record in report_colors] != counts["colors"]
            or [record.get("games") for record in report_colors] != [500, 500]
        ):
            raise PublicationError(f"{name} report summary differs from outcomes")
        raw_report = source_artifacts[f"panel_{name.replace('-', '_')}"]
        decision_report = decision.get("reports", {}).get(name)
        if (
            not isinstance(decision_report, dict)
            or decision_report.get("sha256") != raw_report["sha256"]
            or decision_report.get("bytes") != raw_report["bytes"]
        ):
            raise PublicationError(f"{name} decision/report identity changed")
        policy = PANEL_POLICY[name]
        passed = (
            counts["wins"] >= policy["minimum_wins"]
            and all(
                value >= policy["minimum_per_color"] for value in counts["colors"]
            )
            and counts["illegal"] == 0
            and counts["unfinished"] == 0
        )
        if not passed:
            raise PublicationError(f"{name} no longer satisfies its frozen gate")
        panel_payload["panels"][name] = {
            "source_report": raw_report,
            "model_sha256": report.get("model_sha256"),
            "opponent": configuration.get("opponent"),
            "control_model_sha256": configuration.get("control_model_sha256"),
            "pairs": pairs,
        }
        compact_panels[name] = {
            **counts,
            "threshold": policy,
            "passed": passed,
            "source_report_sha256": raw_report["sha256"],
        }
        selected_configuration = {
            key: configuration.get(key)
            for key in (
                "pairs",
                "time_ms",
                "opening_plies",
                "opening_bank_seed",
                "candidate_tree_nodes",
                "control_tree_nodes",
                "max_actions",
                "max_partial_paths",
                "exploration",
                "fpu",
                "max_turns",
                "single_thread",
            )
        }
        if common_configuration is None:
            common_configuration = selected_configuration
        elif common_configuration != selected_configuration:
            raise PublicationError("panel configurations are not matched")
        current_opening_hash = configuration.get("opening_bank_sha256")
        if opening_bank_sha256 is None:
            opening_bank_sha256 = current_opening_hash
        elif opening_bank_sha256 != current_opening_hash:
            raise PublicationError("panel opening bank identities differ")
        closure = {
            key: configuration.get(key)
            for key in (
                "candidate_source_sha256",
                "comparison_source_sha256",
                "shared_core_sha256",
                "rank4_control_sha256",
                "rank4_engine_sha256",
                "neural_puct_control_sha256",
                "neural_puct_engine_sha256",
                "jacek_nn_control_sha256",
                "jacek_nn_engine_sha256",
                "jacek_nn_source_sha256",
                "rank4_adapter_sha256",
                "neural_puct_adapter_sha256",
                "jacek_nn_adapter_sha256",
                "comparison_executable_sha256",
            )
        }
        if source_closure is None:
            source_closure = closure
        elif source_closure != closure:
            raise PublicationError("panel source closures differ")

    candidate_samples = [
        float(sample)
        for game in latency.get("results", [])
        if isinstance(game, dict)
        for sample in game.get("candidate_ms", [])
    ]
    latency_summary = latency.get("summary", {}).get("candidate")
    if not isinstance(latency_summary, dict):
        raise PublicationError("stage-20 candidate latency summary is missing")
    latency_compact = {
        "sample_count": len(candidate_samples),
        "samples_ms": candidate_samples,
        "p99_ms": _p99(candidate_samples),
        "max_ms": max(candidate_samples),
        "threshold_max_ms_exclusive": 1000,
        "passed": max(candidate_samples) < 1000,
        "source_report_sha256": source_artifacts["latency_report"]["sha256"],
    }
    if (
        latency_compact["sample_count"] != latency_summary.get("decisions")
        or latency_compact["p99_ms"] != latency_summary.get("p99_ms")
        or latency_compact["max_ms"] != latency_summary.get("max_ms")
        or latency_compact["max_ms"] != decision.get("uncontended_max_ms")
        or not latency_compact["passed"]
    ):
        raise PublicationError("stage-20 compact latency recount differs")

    runtime = _snapshot_identity(acceptance.get("runtime"), "accepted runtime")
    runtime_manifest = _snapshot_identity(
        acceptance.get("manifest"), "accepted runtime manifest"
    )
    transition = _normalized_transition(summary)
    compact_body = {
        "schema": RESULTS_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "full_campaign_id": FULL_CAMPAIGN_ID,
        "classification": "post-campaign-derived-public-evidence",
        "terminal": "teacher-candidate-accepted",
        "candidate": {
            "runtime": runtime,
            "manifest": runtime_manifest,
            "original_retention_reference_sha256": (
                ORIGINAL_RETENTION_REFERENCE_SHA256
            ),
        },
        "frozen_configuration": {
            **(common_configuration or {}),
            "opening_bank_sha256": opening_bank_sha256,
            "panels": list(PANELS),
        },
        "panels": compact_panels,
        "latency": latency_compact,
        "retention": {
            "phase_actor": {
                "candidate": decision.get("anchor_candidate"),
                "incumbent": decision.get("anchor_incumbent"),
                "sign_tolerance": 0.005,
                "huber_ratio": 1.02,
                "passed": True,
            },
            "original_incumbent": {
                "candidate": decision.get("original_anchor_candidate"),
                "incumbent": decision.get("original_anchor_incumbent"),
                "sign_tolerance": 0.005,
                "huber_ratio": 1.02,
                "passed": True,
            },
        },
        "pilot_truth": {
            "pilot_passed": False,
            "pilot_20_ms_passed": False,
            "teacher_only_override": True,
            "bypassed_errors": [
                "matched primary strength gate failed",
                "incumbent primary strength gate failed",
            ],
        },
        "transition": transition,
        "student": {"eligible": True, "training_started": False},
        "claims": {
            "local_teacher_candidate_accepted": True,
            "canonical_promotion_eligible": False,
            "canonical_promotion_performed": False,
            "deployment_performed": False,
            "rank4_replaced": False,
            "leaderboard_claim": False,
        },
        "publication_context": {
            "source_campaign_publication": summary.get("publication"),
            "source_campaign_external_upload": summary.get("external_upload"),
            "authorization": "separate explicit post-campaign publication request",
            "derived_evidence_only": True,
            "source_receipts_modified": False,
            "models_included": False,
            "datasets_included": False,
            "raw_transcripts_included": False,
        },
    }
    outcomes = body_hashed(panel_payload)
    compact = body_hashed(compact_body)

    if set(source_artifacts) != set(EXPECTED_SOURCE_ARTIFACTS):
        raise PublicationError("source artifact roster changed")
    for label, (expected_bytes, expected_sha256) in EXPECTED_SOURCE_ARTIFACTS.items():
        record = source_artifacts[label]
        if (
            record.get("path") != SOURCE_FILES[label]
            or record.get("bytes") != expected_bytes
            or record.get("sha256") != expected_sha256
        ):
            raise PublicationError(f"source artifact identity changed: {label}")
    stage21_environment = source["stage21_receipt"].get("environment")
    release_build = (
        stage21_environment.get("release_build")
        if isinstance(stage21_environment, dict)
        else None
    )
    provenance = {
        "frozen_launch": _snapshot_identity(
            acceptance.get("launch"), "accepted launch"
        ),
        "run_start": _snapshot_identity(
            acceptance.get("run_start"), "accepted run start"
        ),
        "release_build": _snapshot_identity(
            release_build, "terminal Release build"
        ),
        "terminal_decision": source_artifacts["decision"],
        "terminal_summary": source_artifacts["final_summary"],
        "acceptance": source_artifacts["acceptance"],
        "student_handoff": source_artifacts["student_handoff"],
        "stage_receipts": {
            "19-game-gates": source_artifacts["stage19_receipt"],
            "20-latency-audit": source_artifacts["stage20_receipt"],
            "21-decision": source_artifacts["stage21_receipt"],
        },
        "candidate_runtime": runtime,
        "candidate_runtime_manifest": runtime_manifest,
        "pilot_teacher_runtime_sha256": source[
            "panel_pilot_teacher"
        ]["configuration"]["control_model_sha256"],
        "original_retention_reference_sha256": (
            ORIGINAL_RETENTION_REFERENCE_SHA256
        ),
        "environment_transition_amendment_sha256": transition[
            "amendment_sha256"
        ],
        "runner_repair_amendment_sha256": transition["runner_repair"][
            "amendment_sha256"
        ],
    }
    manifest_core = {
        "schema": MANIFEST_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "full_campaign_id": FULL_CAMPAIGN_ID,
        "classification": "post-campaign-derived-public-evidence-manifest",
        "campaign_implementation_commit": IMPLEMENTATION_COMMIT,
        "campaign_base_commit": BASE_COMMIT,
        "candidate_runtime_sha256": runtime["sha256"],
        "opening_bank_sha256": opening_bank_sha256,
        "source_closure": source_closure,
        "provenance": provenance,
        "source_artifacts": source_artifacts,
        "derived_artifacts": {},
        "scope": {
            "self_contained_threshold_verification": True,
            "campaign_rerun_possible_without_ignored_inputs": False,
            "contains_models": False,
            "contains_datasets": False,
            "contains_raw_transcripts": False,
            "contains_raw_shards": False,
        },
    }
    return compact, outcomes, manifest_core


def _bundle_artifact(
    path: pathlib.Path, directory: pathlib.Path = BUNDLE_DIRECTORY
) -> dict[str, object]:
    if path.parent != directory:
        raise PublicationError("bundle artifact path is outside bundle")
    opened = read_regular_file(directory, path.name, f"bundle artifact {path.name}")
    return artifact_from_opened(opened, path.name)


def _render_staged_bundle(
    campaign: pathlib.Path,
    staging: pathlib.Path,
    bundle_directory: pathlib.Path,
) -> None:
    compact, outcomes, manifest_core = build_publication(campaign)
    staging.mkdir(parents=True, exist_ok=False)
    for name in sorted(STATIC_BUNDLE_FILE_NAMES):
        source = read_regular_file(bundle_directory, name, f"static bundle {name}")
        atomic_write(staging / name, source.payload)
    compact_path = staging / "compact_results.json"
    outcomes_path = staging / "paired_outcomes.json"
    manifest_path = staging / "manifest.json"
    write_exact(compact_path, compact, "compact results")
    write_exact(outcomes_path, outcomes, "paired outcomes")
    derived = {
        "report": _bundle_artifact(staging / "REPORT.md", staging),
        "publisher": _bundle_artifact(staging / "publish.py", staging),
        "compact_results": _bundle_artifact(compact_path, staging),
        "paired_outcomes": _bundle_artifact(outcomes_path, staging),
    }
    manifest = body_hashed({**manifest_core, "derived_artifacts": derived})
    write_exact(manifest_path, manifest, "publication manifest")
    verify_bundle(campaign=campaign, directory=staging)


def _commit_staged_bundle(
    staging: pathlib.Path,
    *,
    campaign: pathlib.Path | None,
    bundle_directory: pathlib.Path,
    fail_after_commit: int | None = None,
) -> None:
    originals: dict[str, bytes | None] = {}
    for name in GENERATED_FILE_NAMES:
        target = bundle_directory / name
        originals[name] = (
            read_regular_file(
                bundle_directory, name, f"existing generated {name}"
            ).payload
            if target.exists()
            else None
        )
    writes = 0
    try:
        for name in GENERATED_FILE_NAMES:
            staged = read_regular_file(staging, name, f"staged generated {name}")
            payload = staged.payload
            target = bundle_directory / name
            if originals[name] == payload:
                continue
            atomic_write(target, payload)
            writes += 1
            if fail_after_commit is not None and writes >= fail_after_commit:
                raise PublicationError("injected late publication failure")
        verify_bundle(campaign=campaign, directory=bundle_directory)
    except Exception:
        for name in GENERATED_FILE_NAMES:
            target = bundle_directory / name
            original = originals[name]
            if original is None:
                if target.exists():
                    if target.is_symlink() or not target.is_file():
                        raise PublicationError(
                            "cannot roll back non-regular generated artifact"
                        )
                    target.unlink()
            elif not target.exists():
                atomic_write(target, original)
            else:
                current = read_regular_file(
                    bundle_directory, name, f"rollback generated {name}"
                )
                if current.payload != original:
                    atomic_write(target, original)
        raise


def freeze(
    campaign: pathlib.Path,
    *,
    bundle_directory: pathlib.Path = BUNDLE_DIRECTORY,
    fail_after_commit: int | None = None,
) -> None:
    bundle_directory = pathlib.Path(bundle_directory)
    validate_bundle_roster(bundle_directory, allow_generated_missing=True)
    with tempfile.TemporaryDirectory(
        dir=bundle_directory.parent, prefix=".large_teacher_campaign."
    ) as directory:
        staging = pathlib.Path(directory) / "bundle"
        _render_staged_bundle(campaign, staging, bundle_directory)
        _commit_staged_bundle(
            staging,
            campaign=campaign,
            bundle_directory=bundle_directory,
            fail_after_commit=fail_after_commit,
        )


def _reject_absolute_paths(value: object) -> None:
    if isinstance(value, dict):
        for child in value.values():
            _reject_absolute_paths(child)
    elif isinstance(value, list):
        for child in value:
            _reject_absolute_paths(child)
    elif isinstance(value, str):
        user_marker = "/" + "Users" + "/"
        codex_marker = "." + "codex" + "/"
        if value.startswith(("/", "~")) or user_marker in value or codex_marker in value:
            raise PublicationError("bundle contains an absolute local path")


def _text_contains_local_path(text: str, *, allow_shebang: bool = False) -> bool:
    user_marker = "/" + "Users" + "/"
    codex_marker = "." + "codex" + "/"
    local_markers = (
        user_marker,
        codex_marker,
        "/" + "private/var/",
        "/" + "var/folders/",
        "file" + "://",
    )
    segment = r"[A-Za-z0-9._~\-]+"
    absolute_pattern = re.compile(
        r"(?<![A-Za-z0-9:/])"
        + "/"
        + r"(?!/)"
        + segment
        + r"(?:/"
        + segment
        + r")*/?"
    )
    windows_pattern = re.compile(
        r"(?<![A-Za-z0-9])[A-Za-z]:[\\/][^\s\"']+"
    )
    scan_text = text
    if allow_shebang and text.startswith("#!"):
        first_line, separator, remainder = text.partition("\n")
        expected_shebang = "#!" + "/" + "usr/bin/env python3"
        if first_line != expected_shebang or not separator:
            return True
        scan_text = remainder
    scan_text = re.sub(r"(?<=\]\()/[^\s)]+(?=\))", "", scan_text)
    return (
        any(marker in scan_text for marker in local_markers)
        or absolute_pattern.search(scan_text) is not None
        or windows_pattern.search(scan_text) is not None
        or ("~" + "/") in scan_text
    )


def _scan_bundle_files(blobs: Mapping[str, RegularFile]) -> None:
    for name in sorted(BUNDLE_FILE_NAMES):
        try:
            text = blobs[name].payload.decode("utf-8", errors="strict")
        except UnicodeError as error:
            raise PublicationError(f"bundle artifact is not UTF-8 text: {name}") from error
        if _text_contains_local_path(text, allow_shebang=name == "publish.py"):
            raise PublicationError(f"bundle artifact contains a local path: {name}")


def verify_bundle(
    campaign: pathlib.Path | None = None,
    *,
    directory: pathlib.Path = BUNDLE_DIRECTORY,
) -> dict[str, object]:
    directory = pathlib.Path(directory)
    blobs = validate_bundle_roster(directory)
    _scan_bundle_files(blobs)
    manifest = parse_json_bytes(blobs["manifest.json"].payload, "publication manifest")
    compact = parse_json_bytes(blobs["compact_results.json"].payload, "compact results")
    outcomes = parse_json_bytes(blobs["paired_outcomes.json"].payload, "paired outcomes")
    verify_body_hash(manifest, MANIFEST_SCHEMA, "publication manifest")
    verify_body_hash(compact, RESULTS_SCHEMA, "compact results")
    verify_body_hash(outcomes, OUTCOMES_SCHEMA, "paired outcomes")
    for value in (manifest, compact, outcomes):
        _reject_absolute_paths(value)

    manifest_body_keys = {
        "schema",
        "campaign_id",
        "full_campaign_id",
        "classification",
        "campaign_implementation_commit",
        "campaign_base_commit",
        "candidate_runtime_sha256",
        "opening_bank_sha256",
        "source_closure",
        "provenance",
        "source_artifacts",
        "derived_artifacts",
        "scope",
        "body_sha256",
    }
    if (
        set(manifest) != manifest_body_keys
        or manifest.get("campaign_id") != CAMPAIGN_ID
        or manifest.get("full_campaign_id") != FULL_CAMPAIGN_ID
        or manifest.get("classification")
        != "post-campaign-derived-public-evidence-manifest"
        or manifest.get("campaign_implementation_commit") != IMPLEMENTATION_COMMIT
        or manifest.get("campaign_base_commit") != BASE_COMMIT
        or manifest.get("candidate_runtime_sha256") != ACCEPTED_RUNTIME_SHA256
        or manifest.get("opening_bank_sha256") != OPENING_BANK_SHA256
        or manifest.get("source_closure") != EXPECTED_SOURCE_CLOSURE
        or manifest.get("scope") != EXPECTED_SCOPE
    ):
        raise PublicationError("publication manifest policy or provenance changed")
    source_artifacts = manifest.get("source_artifacts")
    if not isinstance(source_artifacts, dict) or set(source_artifacts) != set(
        EXPECTED_SOURCE_ARTIFACTS
    ):
        raise PublicationError("manifest source artifact roster is not exact")
    for label, (expected_bytes, expected_sha256) in EXPECTED_SOURCE_ARTIFACTS.items():
        expected = {
            "path": SOURCE_FILES[label],
            "bytes": expected_bytes,
            "sha256": expected_sha256,
        }
        if source_artifacts.get(label) != expected:
            raise PublicationError(f"manifest source identity changed: {label}")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict) or set(provenance) != {
        "frozen_launch",
        "run_start",
        "release_build",
        "terminal_decision",
        "terminal_summary",
        "acceptance",
        "student_handoff",
        "stage_receipts",
        "candidate_runtime",
        "candidate_runtime_manifest",
        "pilot_teacher_runtime_sha256",
        "original_retention_reference_sha256",
        "environment_transition_amendment_sha256",
        "runner_repair_amendment_sha256",
    }:
        raise PublicationError("manifest provenance roster is not exact")
    if (
        provenance.get("frozen_launch")
        != {
            "sha256": "13f32e9feef97dd3911f3c71466b2ad1163f3855101eedd9ceedc94cb18b7ddd",
            "bytes": 162_071,
        }
        or provenance.get("run_start")
        != {
            "sha256": "f97a3407564e2fef3b02ab412510ff973977521e272e7cf9139a2c97d61e2bfb",
            "bytes": 3_168,
        }
        or provenance.get("release_build")
        != {
            "sha256": "2379a5212e1816e8d7a7db48325e3a56f3dc01f70efc835868fcd1572079e679",
            "bytes": 6_787,
        }
        or provenance.get("terminal_decision") != source_artifacts["decision"]
        or provenance.get("terminal_summary") != source_artifacts["final_summary"]
        or provenance.get("acceptance") != source_artifacts["acceptance"]
        or provenance.get("student_handoff") != source_artifacts["student_handoff"]
        or provenance.get("stage_receipts")
        != {
            "19-game-gates": source_artifacts["stage19_receipt"],
            "20-latency-audit": source_artifacts["stage20_receipt"],
            "21-decision": source_artifacts["stage21_receipt"],
        }
        or provenance.get("candidate_runtime")
        != {"sha256": ACCEPTED_RUNTIME_SHA256, "bytes": 4_864_000}
        or provenance.get("candidate_runtime_manifest")
        != {"sha256": ACCEPTED_RUNTIME_MANIFEST_SHA256, "bytes": 73_824}
        or provenance.get("pilot_teacher_runtime_sha256")
        != PILOT_TEACHER_RUNTIME_SHA256
        or provenance.get("original_retention_reference_sha256")
        != ORIGINAL_RETENTION_REFERENCE_SHA256
        or provenance.get("environment_transition_amendment_sha256")
        != TRANSITION_AMENDMENT_SHA256
        or provenance.get("runner_repair_amendment_sha256")
        != RUNNER_REPAIR_AMENDMENT_SHA256
    ):
        raise PublicationError("manifest provenance relationship changed")

    derived = manifest.get("derived_artifacts")
    if not isinstance(derived, dict) or set(derived) != set(DERIVED_PATHS):
        raise PublicationError("derived artifact roster is not exact")
    for label, expected_path in DERIVED_PATHS.items():
        record = derived[label]
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise PublicationError(f"derived artifact record is malformed: {label}")
        if record.get("path") != expected_path:
            raise PublicationError(f"derived artifact path changed: {label}")
        observed_derived = artifact_from_opened(blobs[expected_path], expected_path)
        if observed_derived != record:
            raise PublicationError(f"derived artifact changed: {label}")
        if label in EXPECTED_DERIVED_CONTENT:
            expected_bytes, expected_sha256 = EXPECTED_DERIVED_CONTENT[label]
            if (
                record.get("bytes") != expected_bytes
                or record.get("sha256") != expected_sha256
            ):
                raise PublicationError(f"derived evidence identity changed: {label}")

    panels = outcomes.get("panels")
    if set(outcomes) != {
        "schema",
        "campaign_id",
        "field_order",
        "pair_record_order",
        "panels",
        "body_sha256",
    } or outcomes.get("campaign_id") != CAMPAIGN_ID or outcomes.get(
        "field_order"
    ) != [
        "candidate_player",
        "winner",
        "illegal",
        "unfinished",
    ] or outcomes.get("pair_record_order") != [
        "pair_index",
        "opening_state_identity",
        "games",
    ]:
        raise PublicationError("paired outcome topology changed")
    if (
        set(compact)
        != {
            "schema",
            "campaign_id",
            "full_campaign_id",
            "classification",
            "terminal",
            "candidate",
            "frozen_configuration",
            "panels",
            "latency",
            "retention",
            "pilot_truth",
            "transition",
            "student",
            "claims",
            "publication_context",
            "body_sha256",
        }
        or compact.get("campaign_id") != CAMPAIGN_ID
        or compact.get("full_campaign_id") != FULL_CAMPAIGN_ID
        or compact.get("classification")
        != "post-campaign-derived-public-evidence"
        or compact.get("terminal") != "teacher-candidate-accepted"
        or compact.get("candidate")
        != {
            "runtime": {"sha256": ACCEPTED_RUNTIME_SHA256, "bytes": 4_864_000},
            "manifest": {
                "sha256": ACCEPTED_RUNTIME_MANIFEST_SHA256,
                "bytes": 73_824,
            },
            "original_retention_reference_sha256": (
                ORIGINAL_RETENTION_REFERENCE_SHA256
            ),
        }
        or compact.get("frozen_configuration") != EXPECTED_FROZEN_CONFIGURATION
        or compact.get("pilot_truth") != EXPECTED_PILOT_TRUTH
        or compact.get("student") != {"eligible": True, "training_started": False}
        or compact.get("claims") != EXPECTED_CLAIMS
        or compact.get("publication_context") != EXPECTED_PUBLICATION_CONTEXT
    ):
        raise PublicationError("compact result policy or topology changed")
    if not isinstance(panels, dict) or set(panels) != set(PANELS):
        raise PublicationError("paired outcome panel roster changed")
    for name in PANELS:
        record = panels[name]
        source_label = f"panel_{name.replace('-', '_')}"
        expected_control = (
            PILOT_TEACHER_RUNTIME_SHA256
            if name in {"matched", "pilot-teacher"}
            else None
        )
        expected_opponent = {
            "matched": "jacek-replay",
            "pilot-teacher": "jacek-replay",
            "rank4": "rank4",
            "jacek-nn": "jacek-nn",
        }[name]
        if (
            not isinstance(record, dict)
            or set(record)
            != {
                "source_report",
                "model_sha256",
                "opponent",
                "control_model_sha256",
                "pairs",
            }
            or not isinstance(record.get("pairs"), list)
            or record.get("source_report") != source_artifacts[source_label]
            or record.get("model_sha256") != ACCEPTED_RUNTIME_SHA256
            or record.get("opponent") != expected_opponent
            or record.get("control_model_sha256") != expected_control
        ):
            raise PublicationError(f"paired outcomes are malformed: {name}")
        if len(record["pairs"]) != 500:
            raise PublicationError(f"paired outcome cardinality changed: {name}")
        counts = _panel_counts(record["pairs"])
        expected = compact.get("panels", {}).get(name)
        if (
            not isinstance(expected, dict)
            or set(expected)
            != {
                "games",
                "wins",
                "losses",
                "illegal",
                "unfinished",
                "colors",
                "threshold",
                "passed",
                "source_report_sha256",
            }
            or expected.get("threshold") != PANEL_POLICY[name]
            or any(
            counts[key] != expected.get(key)
            for key in ("games", "wins", "losses", "illegal", "unfinished", "colors")
            )
        ):
            raise PublicationError(f"paired outcome recount changed: {name}")
        policy = PANEL_POLICY[name]
        if (
            counts["wins"] < policy["minimum_wins"]
            or min(counts["colors"]) < policy["minimum_per_color"]
            or counts["illegal"] != 0
            or counts["unfinished"] != 0
            or expected.get("passed") is not True
        ):
            raise PublicationError(f"panel gate no longer passes: {name}")

    state_rosters = [
        [record[1] for record in panels[name]["pairs"]] for name in PANELS
    ]
    if any(roster != state_rosters[0] for roster in state_rosters[1:]):
        raise PublicationError("compact panel opening-state rosters differ")

    latency = compact.get("latency")
    if (
        not isinstance(latency, dict)
        or set(latency)
        != {
            "sample_count",
            "samples_ms",
            "p99_ms",
            "max_ms",
            "threshold_max_ms_exclusive",
            "passed",
            "source_report_sha256",
        }
        or not isinstance(latency.get("samples_ms"), list)
        or latency.get("threshold_max_ms_exclusive") != 1000
        or latency.get("source_report_sha256")
        != EXPECTED_SOURCE_ARTIFACTS["latency_report"][1]
    ):
        raise PublicationError("compact latency evidence is malformed")
    samples = [float(value) for value in latency["samples_ms"]]
    if (
        any(not math.isfinite(value) or value < 0 for value in samples)
        or len(samples) != latency.get("sample_count")
        or _p99(samples) != latency.get("p99_ms")
        or max(samples) != latency.get("max_ms")
        or max(samples) >= 1000
        or latency.get("passed") is not True
    ):
        raise PublicationError("compact latency recount changed")

    retention = compact.get("retention")
    if not isinstance(retention, dict):
        raise PublicationError("compact retention evidence is malformed")
    for name in ("phase_actor", "original_incumbent"):
        record = retention.get(name)
        if not isinstance(record, dict):
            raise PublicationError(f"retention record is missing: {name}")
        candidate = record.get("candidate")
        incumbent = record.get("incumbent")
        if not isinstance(candidate, dict) or not isinstance(incumbent, dict):
            raise PublicationError(f"retention metrics are malformed: {name}")
        if record.get("sign_tolerance") != 0.005 or record.get("huber_ratio") != 1.02:
            raise PublicationError(f"retention policy changed: {name}")
        passed = (
            float(candidate["sign_accuracy"])
            >= float(incumbent["sign_accuracy"]) - float(record["sign_tolerance"])
            and float(candidate["weighted_huber"])
            <= float(incumbent["weighted_huber"]) * float(record["huber_ratio"])
        )
        if passed is not True or record.get("passed") is not True:
            raise PublicationError(f"retention gate no longer passes: {name}")

    transition = compact.get("transition")
    if (
        not isinstance(transition, dict)
        or transition.get("classification")
        != "campaign-local-environment-transition-recovery"
        or transition.get("previous_platform")
        != "macOS-26.5.2-arm64-arm-64bit-Mach-O"
        or transition.get("resumed_platform")
        != "macOS-26.6.2-arm64-arm-64bit-Mach-O"
        or transition.get("pretransition_stage19_reused") is not False
        or transition.get("pretransition_stage19_classification")
        != "diagnostic-only-superseded"
        or transition.get("authoritative_stage19_shards") != 400
        or transition.get("amendment_sha256") != TRANSITION_AMENDMENT_SHA256
        or transition.get("archive_manifest_sha256") != TRANSITION_ARCHIVE_SHA256
        or transition.get("compatibility_qualification_sha256")
        != COMPATIBILITY_QUALIFICATION_SHA256
        or not isinstance(transition.get("runner_repair"), dict)
        or transition["runner_repair"].get("classification")
        != "campaign-local-transition-runner-import-path-repair"
        or transition["runner_repair"].get(
            "replacement_stage19_outputs_before_repair"
        )
        != 0
        or transition["runner_repair"].get("amendment_sha256")
        != RUNNER_REPAIR_AMENDMENT_SHA256
        or transition["runner_repair"].get("first_attempt_failure_status_sha256")
        != RUNNER_FAILURE_STATUS_SHA256
    ):
        raise PublicationError("environment transition chain changed")

    claims = compact.get("claims")
    publication = compact.get("publication_context")
    student = compact.get("student")
    pilot = compact.get("pilot_truth")
    if (
        claims != EXPECTED_CLAIMS
        or publication != EXPECTED_PUBLICATION_CONTEXT
        or student != {"eligible": True, "training_started": False}
        or pilot != EXPECTED_PILOT_TRUTH
    ):
        raise PublicationError("publication scope or claim policy changed")

    bundle_bytes = sum(blob.size for blob in blobs.values())
    if bundle_bytes >= 1024 * 1024:
        raise PublicationError("compact publication bundle exceeds 1 MiB")

    if campaign is not None:
        expected_compact, expected_outcomes, expected_manifest_core = build_publication(
            campaign
        )
        if compact != expected_compact or outcomes != expected_outcomes:
            raise PublicationError("bundle differs from terminal campaign derivation")
        observed_core = dict(manifest)
        observed_core.pop("body_sha256", None)
        observed_core.pop("derived_artifacts", None)
        expected_core = dict(expected_manifest_core)
        expected_core.pop("derived_artifacts", None)
        if observed_core != expected_core:
            raise PublicationError("manifest source provenance changed")

    return {
        "schema": "papersoccer.large-teacher-public-verification.v1",
        "bundle_bytes": bundle_bytes,
        "panels": {name: compact["panels"][name]["wins"] for name in PANELS},
        "latency_samples": len(samples),
        "latency_max_ms": max(samples),
        "self_contained": True,
        "source_verified": campaign is not None,
    }


def _copy_bundle_for_test(destination: pathlib.Path) -> None:
    source_snapshot = validate_bundle_roster(BUNDLE_DIRECTORY)
    destination.mkdir(parents=True, exist_ok=False)
    for name in sorted(BUNDLE_FILE_NAMES):
        atomic_write(destination / name, source_snapshot[name].payload)
    validate_bundle_roster(destination)


def _rewrite_body_hashed_json(
    path: pathlib.Path, mutator: Any
) -> None:
    value = load_json(path, f"test mutation {path.name}")
    body = dict(value)
    body.pop("body_sha256", None)
    mutator(body)
    atomic_write(path, canonical_json_bytes(body_hashed(body)))


def _refresh_derived_binding(
    directory: pathlib.Path, label: str
) -> None:
    path_name = DERIVED_PATHS[label]

    def mutate(body: dict[str, Any]) -> None:
        body["derived_artifacts"][label] = _bundle_artifact(
            directory / path_name, directory
        )

    _rewrite_body_hashed_json(directory / "manifest.json", mutate)


def _expect_publication_failure(
    label: str, action: Any, expected_fragment: str
) -> None:
    try:
        action()
    except PublicationError as error:
        if expected_fragment not in str(error):
            raise PublicationError(
                f"{label} failed for the wrong reason: {error}"
            ) from error
        return
    raise PublicationError(f"adversarial test unexpectedly passed: {label}")


def _expect_semantic_failure_after_rebinding(
    *,
    label: str,
    directory: pathlib.Path,
    derived_label: str,
    action: Any,
    expected_fragment: str,
) -> None:
    original = EXPECTED_DERIVED_CONTENT[derived_label]
    record = _bundle_artifact(
        directory / DERIVED_PATHS[derived_label], directory
    )
    EXPECTED_DERIVED_CONTENT[derived_label] = (
        int(record["bytes"]),
        str(record["sha256"]),
    )
    try:
        _expect_publication_failure(label, action, expected_fragment)
    finally:
        EXPECTED_DERIVED_CONTENT[derived_label] = original


def _copy_source_allowlist(
    campaign: pathlib.Path, destination_parent: pathlib.Path
) -> pathlib.Path:
    campaign = pathlib.Path(campaign)
    destination = destination_parent / CAMPAIGN_ID
    destination.mkdir(parents=True, exist_ok=False)
    for label, relative in SOURCE_FILES.items():
        source = read_regular_file(campaign, relative, f"test source {label}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(target, source.payload)
    return destination


def _payload(directory: pathlib.Path, name: str, label: str) -> bytes:
    return read_regular_file(directory, name, label).payload


def _assert_preopen_swap_is_not_read(
    *,
    label: str,
    secure_root: pathlib.Path,
    relative: str,
    target: pathlib.Path,
    external: pathlib.Path,
    operation: Any,
) -> None:
    """Replace a validated entry just before open and prove no sentinel read."""

    global _PRE_OPEN_HOOK, _READ_OBSERVER
    sentinel = os.stat(external, follow_symlinks=False)
    expected_root = secure_root.resolve(strict=True)
    hook_ran = False
    sentinel_read = False

    def swap(observed_root: pathlib.Path, observed_relative: str) -> None:
        nonlocal hook_ran
        if (
            hook_ran
            or pathlib.Path(observed_root).resolve(strict=True) != expected_root
            or observed_relative != relative
        ):
            return
        replacement = target.parent / f".{target.name}.preopen-swap"
        os.link(external, replacement)
        os.replace(replacement, target)
        hook_ran = True

    def observe(_relative: str, device: int, inode: int) -> None:
        nonlocal sentinel_read
        if (device, inode) == (sentinel.st_dev, sentinel.st_ino):
            sentinel_read = True

    previous_hook = _PRE_OPEN_HOOK
    previous_observer = _READ_OBSERVER
    _PRE_OPEN_HOOK = swap
    _READ_OBSERVER = observe
    try:
        _expect_publication_failure(label, operation, "changed before open")
    finally:
        _PRE_OPEN_HOOK = previous_hook
        _READ_OBSERVER = previous_observer
    if not hook_ran:
        raise PublicationError(f"pre-open attack hook did not run: {label}")
    if sentinel_read:
        raise PublicationError(f"forbidden external sentinel was read: {label}")


def _assert_midread_extra_file_is_rejected(bundle: pathlib.Path) -> None:
    global _PRE_OPEN_HOOK
    hook_ran = False
    expected_root = bundle.resolve(strict=True)

    def inject(observed_root: pathlib.Path, _observed_relative: str) -> None:
        nonlocal hook_ran
        if hook_ran or pathlib.Path(observed_root).resolve(strict=True) != expected_root:
            return
        atomic_write(bundle / "candidate.runtime", b"forbidden model bytes\n")
        hook_ran = True

    previous_hook = _PRE_OPEN_HOOK
    _PRE_OPEN_HOOK = inject
    try:
        _expect_publication_failure(
            "midread-extra-file-roster",
            lambda: verify_bundle(directory=bundle),
            "roster changed during verification",
        )
    finally:
        _PRE_OPEN_HOOK = previous_hook
    if not hook_ran:
        raise PublicationError("mid-read extra-file attack hook did not run")


def run_adversarial_tests(
    campaign: pathlib.Path | None = None,
) -> dict[str, object]:
    campaign = pathlib.Path(campaign) if campaign is not None else None
    verify_bundle()
    if campaign is not None:
        verify_bundle(campaign=campaign)
    tests: list[str] = []

    def fresh_bundle(root: pathlib.Path) -> pathlib.Path:
        bundle = root / "bundle"
        _copy_bundle_for_test(bundle)
        return bundle

    with tempfile.TemporaryDirectory(prefix="large-teacher-public-tests.") as raw:
        root = pathlib.Path(raw)

        bundle = fresh_bundle(root / "derived-roster")
        _rewrite_body_hashed_json(
            bundle / "manifest.json",
            lambda body: body.__setitem__("derived_artifacts", {}),
        )
        _expect_publication_failure(
            "missing-derived-roster",
            lambda: verify_bundle(directory=bundle),
            "derived artifact roster",
        )
        tests.append("missing-derived-roster")

        bundle = fresh_bundle(root / "extra-file")
        atomic_write(bundle / "candidate.runtime", b"x" * (2 * 1024 * 1024))
        _expect_publication_failure(
            "extra-bundle-file",
            lambda: verify_bundle(directory=bundle),
            "bundle file roster",
        )
        tests.append("extra-bundle-file")

        bundle = fresh_bundle(root / "midread-extra-file")
        _assert_midread_extra_file_is_rejected(bundle)
        tests.append("midread-extra-file-roster")

        bundle = fresh_bundle(root / "bundle-symlink")
        external = root / "external-report.md"
        atomic_write(external, _payload(bundle, "REPORT.md", "symlink test report"))
        (bundle / "REPORT.md").unlink()
        (bundle / "REPORT.md").symlink_to(external)
        _expect_publication_failure(
            "external-report-symlink",
            lambda: verify_bundle(directory=bundle),
            "not a regular file",
        )
        tests.append("external-report-symlink")

        for case_name, artifact_name, derived_label, local_path in (
            ("report", "REPORT.md", "report", b"/" + b"etc/passwd"),
            ("publisher-posix", "publish.py", "publisher", b"/" + b"tmp"),
            (
                "publisher-windows",
                "publish.py",
                "publisher",
                b"D:" + b"/" + b"tmp/file",
            ),
        ):
            bundle = fresh_bundle(root / f"generic-local-path-{case_name}")
            original = _payload(bundle, artifact_name, f"{derived_label} path test")
            atomic_write(
                bundle / artifact_name,
                original + b"\nLocal path: " + local_path + b"\n",
            )
            _refresh_derived_binding(bundle, derived_label)
            _expect_publication_failure(
                f"generic-local-path-{case_name}",
                lambda bundle=bundle: verify_bundle(directory=bundle),
                "local path",
            )
            tests.append(f"generic-local-path-{case_name}")

        bundle = fresh_bundle(root / "generic-local-path-json")

        def add_generic_json_path(body: dict[str, Any]) -> None:
            body["publication_context"]["authorization"] = (
                "inspect " + "/" + "etc/passwd"
            )

        _rewrite_body_hashed_json(
            bundle / "compact_results.json", add_generic_json_path
        )
        _refresh_derived_binding(bundle, "compact_results")
        _expect_publication_failure(
            "generic-local-path-json",
            lambda: verify_bundle(directory=bundle),
            "local path",
        )
        tests.append("generic-local-path-json")

        safe_syntax = (
            "https://example.org/reference/a/b\n"
            "[relative](docs/reference.md)\n"
            "[root-relative](/reference/path)\n"
            "ratio = numerator / denominator\n"
            "regex = r'^/+$'\n"
        )
        if _text_contains_local_path(safe_syntax):
            raise PublicationError("local-path scanner rejected safe URL or syntax")
        shebang = "#!" + "/" + "usr/bin/env python3\nprint('ok')\n"
        if _text_contains_local_path(shebang, allow_shebang=True):
            raise PublicationError("local-path scanner rejected the publisher shebang")
        unsafe_shebang = shebang.split("\n", 1)[0] + " " + "/" + "etc/passwd\n"
        if not _text_contains_local_path(unsafe_shebang, allow_shebang=True):
            raise PublicationError("local-path scanner exempted a modified shebang")
        tests.append("local-path-safe-syntax")

        bundle = fresh_bundle(root / "root-relative-markdown")
        report_payload = _payload(bundle, "REPORT.md", "root-relative report")
        atomic_write(
            bundle / "REPORT.md",
            report_payload + b"\n[docs](" + b"/" + b"reference/path)\n",
        )
        _refresh_derived_binding(bundle, "report")
        original_report_identity = EXPECTED_DERIVED_CONTENT["report"]
        modified_report = read_regular_file(
            bundle, "REPORT.md", "root-relative modified report"
        )
        EXPECTED_DERIVED_CONTENT["report"] = (
            modified_report.size,
            modified_report.sha256,
        )
        try:
            verify_bundle(directory=bundle)
        finally:
            EXPECTED_DERIVED_CONTENT["report"] = original_report_identity
        tests.append("root-relative-markdown-allowed")

        for unsafe_path in (
            "/" + "tmp",
            "/" + "tmp/",
            "D:" + "/" + "tmp/file",
        ):
            if not _text_contains_local_path(f"local: {unsafe_path}\n"):
                raise PublicationError(
                    f"local-path scanner missed an absolute path: {unsafe_path}"
                )
        tests.append("local-path-single-component-and-windows")

        bundle = fresh_bundle(root / "manifest-runtime-extra")

        def add_runtime_manifest_field(body: dict[str, Any]) -> None:
            body["provenance"]["candidate_runtime_manifest"]["forged"] = True

        _rewrite_body_hashed_json(
            bundle / "manifest.json", add_runtime_manifest_field
        )
        _expect_publication_failure(
            "runtime-manifest-extra-field",
            lambda: verify_bundle(directory=bundle),
            "provenance relationship",
        )
        tests.append("runtime-manifest-extra-field")

        bundle = fresh_bundle(root / "manifest-runtime-bytes")

        def alter_runtime_manifest_bytes(body: dict[str, Any]) -> None:
            body["provenance"]["candidate_runtime_manifest"]["bytes"] = 1

        _rewrite_body_hashed_json(
            bundle / "manifest.json", alter_runtime_manifest_bytes
        )
        _expect_publication_failure(
            "runtime-manifest-byte-count",
            lambda: verify_bundle(directory=bundle),
            "provenance relationship",
        )
        tests.append("runtime-manifest-byte-count")

        bundle = fresh_bundle(root / "pilot-policy")

        def falsify_pilot(body: dict[str, Any]) -> None:
            body["pilot_truth"]["bypassed_errors"].append(
                "game gate has illegal or unfinished games"
            )
            body["publication_context"]["source_campaign_external_upload"] = True

        _rewrite_body_hashed_json(bundle / "compact_results.json", falsify_pilot)
        _refresh_derived_binding(bundle, "compact_results")
        _expect_semantic_failure_after_rebinding(
            label="pilot-publication-policy-tamper",
            directory=bundle,
            derived_label="compact_results",
            action=lambda: verify_bundle(directory=bundle),
            expected_fragment="compact result policy",
        )
        tests.append("pilot-publication-policy-tamper")

        bundle = fresh_bundle(root / "pair-cardinality")

        def remove_pair(body: dict[str, Any]) -> None:
            body["panels"]["matched"]["pairs"].pop()

        _rewrite_body_hashed_json(bundle / "paired_outcomes.json", remove_pair)
        _refresh_derived_binding(bundle, "paired_outcomes")
        _expect_semantic_failure_after_rebinding(
            label="pair-cardinality-tamper",
            directory=bundle,
            derived_label="paired_outcomes",
            action=lambda: verify_bundle(directory=bundle),
            expected_fragment="cardinality",
        )
        tests.append("pair-cardinality-tamper")

        bundle = fresh_bundle(root / "frozen-configuration")

        def alter_configuration(body: dict[str, Any]) -> None:
            body["frozen_configuration"]["time_ms"] = 999

        _rewrite_body_hashed_json(
            bundle / "compact_results.json", alter_configuration
        )
        _refresh_derived_binding(bundle, "compact_results")
        _expect_semantic_failure_after_rebinding(
            label="frozen-configuration-tamper",
            directory=bundle,
            derived_label="compact_results",
            action=lambda: verify_bundle(directory=bundle),
            expected_fragment="compact result policy",
        )
        tests.append("frozen-configuration-tamper")

        bundle = fresh_bundle(root / "opening-roster")

        def alter_opening_roster(body: dict[str, Any]) -> None:
            for panel in PANELS:
                body["panels"][panel]["pairs"][0][1] = "1:1"

        _rewrite_body_hashed_json(
            bundle / "paired_outcomes.json", alter_opening_roster
        )
        _refresh_derived_binding(bundle, "paired_outcomes")
        _expect_publication_failure(
            "opening-roster-content-tamper",
            lambda: verify_bundle(directory=bundle),
            "derived evidence identity",
        )
        tests.append("opening-roster-content-tamper")

        bundle = fresh_bundle(root / "latency-content")

        def alter_latency_sample(body: dict[str, Any]) -> None:
            body["latency"]["samples_ms"][0] += 0.001

        _rewrite_body_hashed_json(
            bundle / "compact_results.json", alter_latency_sample
        )
        _refresh_derived_binding(bundle, "compact_results")
        _expect_publication_failure(
            "latency-content-tamper",
            lambda: verify_bundle(directory=bundle),
            "derived evidence identity",
        )
        tests.append("latency-content-tamper")

        bundle = fresh_bundle(root / "retention-policy")

        def loosen_retention(body: dict[str, Any]) -> None:
            body["retention"]["phase_actor"]["sign_tolerance"] = 1.0
            body["retention"]["phase_actor"]["huber_ratio"] = 10.0

        _rewrite_body_hashed_json(bundle / "compact_results.json", loosen_retention)
        _refresh_derived_binding(bundle, "compact_results")
        _expect_semantic_failure_after_rebinding(
            label="retention-policy-tamper",
            directory=bundle,
            derived_label="compact_results",
            action=lambda: verify_bundle(directory=bundle),
            expected_fragment="retention policy",
        )
        tests.append("retention-policy-tamper")

        source_parent = root / "synthetic-source-symlink"
        source = source_parent / CAMPAIGN_ID
        source.mkdir(parents=True)
        external = root / "external-final-summary.json"
        atomic_write(external, b"{}\n")
        (source / "final-summary.json").symlink_to(external)
        _expect_publication_failure(
            "source-symlink-before-open",
            lambda: build_publication(source),
            "not a regular file",
        )
        tests.append("source-symlink-before-open")

        source = root / "synthetic-source-swap" / CAMPAIGN_ID
        source.mkdir(parents=True)
        source_target = source / "final-summary.json"
        atomic_write(source_target, b"{}\n")
        source_sentinel = root / "forbidden-source-sentinel.json"
        atomic_write(
            source_sentinel,
            canonical_json_bytes({"forbidden": "/" + "etc/passwd"}),
        )
        _assert_preopen_swap_is_not_read(
            label="source-preopen-swap-no-read",
            secure_root=source,
            relative="final-summary.json",
            target=source_target,
            external=source_sentinel,
            operation=lambda: build_publication(source),
        )
        tests.append("source-preopen-swap-no-read")

        bundle = fresh_bundle(root / "bundle-preopen-swap")
        bundle_sentinel = root / "forbidden-bundle-sentinel.md"
        atomic_write(
            bundle_sentinel,
            b"forbidden sentinel " + b"/" + b"etc/passwd\n",
        )
        _assert_preopen_swap_is_not_read(
            label="bundle-preopen-swap-no-read",
            secure_root=bundle,
            relative="REPORT.md",
            target=bundle / "REPORT.md",
            external=bundle_sentinel,
            operation=lambda: verify_bundle(directory=bundle),
        )
        tests.append("bundle-preopen-swap-no-read")

        if campaign is not None:
            source = _copy_source_allowlist(campaign, root / "semantic-source")
            anchor_path = source / SOURCE_FILES["anchor_phase_actor"]
            anchor = load_json(anchor_path, "semantic attack anchor")
            anchor["candidate_metrics"]["sign_accuracy"] = 0.0
            atomic_write(anchor_path, canonical_json_bytes(anchor))
            original_identity = EXPECTED_SOURCE_ARTIFACTS["anchor_phase_actor"]
            opened_anchor = read_regular_file(
                anchor_path.parent, anchor_path.name, "altered semantic anchor"
            )
            EXPECTED_SOURCE_ARTIFACTS["anchor_phase_actor"] = (
                opened_anchor.size,
                opened_anchor.sha256,
            )
            try:
                _expect_publication_failure(
                    "semantic-anchor-receipt-mismatch",
                    lambda: build_publication(source),
                    "anchor metric provenance",
                )
            finally:
                EXPECTED_SOURCE_ARTIFACTS["anchor_phase_actor"] = original_identity
            tests.append("semantic-anchor-receipt-mismatch")

        for failure_after in range(1, len(GENERATED_FILE_NAMES) + 1):
            bundle = fresh_bundle(root / f"late-failure-{failure_after}")
            staging = root / f"late-failure-staging-{failure_after}"
            _copy_bundle_for_test(staging)
            for name in GENERATED_FILE_NAMES:
                atomic_write(
                    bundle / name,
                    _payload(bundle, name, f"late mutation {name}") + b" ",
                )
            originals = {
                name: _payload(bundle, name, f"late original {name}")
                for name in GENERATED_FILE_NAMES
            }
            _expect_publication_failure(
                f"late-failure-rollback-{failure_after}",
                lambda failure_after=failure_after, bundle=bundle, staging=staging: (
                    _commit_staged_bundle(
                        staging,
                        campaign=None,
                        bundle_directory=bundle,
                        fail_after_commit=failure_after,
                    )
                ),
                "injected late publication failure",
            )
            if any(
                _payload(bundle, name, f"late restored {name}") != payload
                for name, payload in originals.items()
            ):
                raise PublicationError(
                    f"late failure {failure_after} did not roll back generated files"
                )
            tests.append(f"late-failure-rollback-{failure_after}")

        bundle = fresh_bundle(root / "idempotence")
        compact_path = bundle / "compact_results.json"
        atomic_write(
            compact_path,
            _payload(bundle, compact_path.name, "idempotence compact") + b" ",
        )
        staging = root / "idempotence-staging"
        _copy_bundle_for_test(staging)
        _commit_staged_bundle(
            staging, campaign=None, bundle_directory=bundle
        )
        first = {
            name: (
                opened.payload,
                opened.mtime_ns,
            )
            for name in BUNDLE_FILE_NAMES
            for opened in (
                read_regular_file(bundle, name, f"first idempotence {name}"),
            )
        }
        _commit_staged_bundle(
            staging, campaign=None, bundle_directory=bundle
        )
        second = {
            name: (
                opened.payload,
                opened.mtime_ns,
            )
            for name in BUNDLE_FILE_NAMES
            for opened in (
                read_regular_file(bundle, name, f"second idempotence {name}"),
            )
        }
        if first != second:
            raise PublicationError("two-run commit is not byte/mtime idempotent")
        tests.append("two-run-commit-idempotence")

        if campaign is not None:
            bundle = fresh_bundle(root / "source-bound-idempotence")
            freeze(campaign, bundle_directory=bundle)
            first = {
                name: (
                    opened.payload,
                    opened.mtime_ns,
                )
                for name in BUNDLE_FILE_NAMES
                for opened in (
                    read_regular_file(bundle, name, f"source first {name}"),
                )
            }
            freeze(campaign, bundle_directory=bundle)
            second = {
                name: (
                    opened.payload,
                    opened.mtime_ns,
                )
                for name in BUNDLE_FILE_NAMES
                for opened in (
                    read_regular_file(bundle, name, f"source second {name}"),
                )
            }
            if first != second:
                raise PublicationError(
                    "two-run source-bound freeze is not byte/mtime idempotent"
                )
            tests.append("source-bound-freeze-idempotence")

    return {
        "schema": "papersoccer.large-teacher-public-adversarial-tests.v1",
        "passed": len(tests),
        "tests": tests,
        "source_bound": campaign is not None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze_parser = commands.add_parser("freeze")
    freeze_parser.add_argument(
        "--campaign-root", type=pathlib.Path, default=DEFAULT_CAMPAIGN
    )
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--campaign-root", type=pathlib.Path)
    test_parser = commands.add_parser("test")
    test_parser.add_argument("--campaign-root", type=pathlib.Path)
    arguments = parser.parse_args()
    if arguments.command == "freeze":
        freeze(arguments.campaign_root)
        result = verify_bundle(campaign=arguments.campaign_root)
    elif arguments.command == "verify":
        result = verify_bundle(campaign=arguments.campaign_root)
    else:
        result = run_adversarial_tests(arguments.campaign_root)
    print(canonical_json_bytes(result).decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
