#!/usr/bin/env python3
"""Build and verify deterministic release assets for the flagship study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

try:
    from benchmarks.flagship_study import release_summary
except ModuleNotFoundError:  # Direct execution from outside the repository root.
    import release_summary  # type: ignore[no-redef]


RELEASE_ID = "flagship-study-v4"
SOURCE_TAG = "flagship-study-v4-record"
CORE_ARCHIVE_NAME = f"{RELEASE_ID}-core.zip"
DECISION_ARCHIVE_NAME = f"{RELEASE_ID}-decision-data.zip"
CHECKSUMS_NAME = "SHA256SUMS"
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
ZIP_MODE = stat.S_IFREG | 0o644
COMPRESSION_LEVEL = 9

STUDY_ROOT = pathlib.PurePosixPath("benchmarks/flagship_study")
MANIFEST_PATH = STUDY_ROOT / "manifest.json"
SELECTION_PATH = STUDY_ROOT / "selection_lock.json"
REPORT_PATH = STUDY_ROOT / "REPORT.md"
RELEASE_NOTES_PATH = STUDY_ROOT / "RELEASE_NOTES.md"
CHART_PATHS = (
    STUDY_ROOT / "charts/test_bradley_terry.svg",
    STUDY_ROOT / "charts/test_calibration.svg",
    STUDY_ROOT / "charts/validation_pareto.svg",
)
DECISION_PATHS = tuple(
    STUDY_ROOT / f"data/{phase}.json"
    for phase in ("development", "validation", "test")
)
LINEAGE_ATTACHMENT_PATHS = (
    STUDY_ROOT / "V3_VALIDATION_FAILURE.md",
    STUDY_ROOT / "superseded/manifest-b7553a24.json",
)
SUMMARY_JSON_PATH = STUDY_ROOT / "summary/summary.json"
PAIRWISE_CSV_PATH = STUDY_ROOT / "summary/pairwise.csv"
CONFIGURATIONS_CSV_PATH = STUDY_ROOT / "summary/configurations.csv"
CORE_SOURCE_PATHS = (
    MANIFEST_PATH,
    SELECTION_PATH,
    REPORT_PATH,
    RELEASE_NOTES_PATH,
    *CHART_PATHS,
)
SUMMARY_PATHS = (
    SUMMARY_JSON_PATH,
    PAIRWISE_CSV_PATH,
    CONFIGURATIONS_CSV_PATH,
)
CORE_ARCHIVE_PATHS = (*CORE_SOURCE_PATHS, *SUMMARY_PATHS)
TAG_IMMUTABLE_PATHS = (
    MANIFEST_PATH,
    SELECTION_PATH,
    *DECISION_PATHS,
    *LINEAGE_ATTACHMENT_PATHS,
)


class PackagingError(RuntimeError):
    """Raised when release inputs or outputs fail an integrity check."""


@dataclass(frozen=True)
class ArchiveEntry:
    """One deterministic archive member backed by a repository file."""

    archive_path: str
    source_path: pathlib.Path

    def open(self):
        return self.source_path.open("rb")


@dataclass(frozen=True)
class ReleaseInputs:
    repository: pathlib.Path
    manifest: dict[str, Any]
    selection: dict[str, Any]
    phases: dict[str, dict[str, Any]]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PackagingError(message)


def _repository_path(repository: pathlib.Path, relative: pathlib.PurePosixPath) -> pathlib.Path:
    return repository.joinpath(*relative.parts)


def _validate_source_paths(
    repository: pathlib.Path,
    relative_paths: Sequence[pathlib.PurePosixPath],
) -> None:
    missing = []
    for relative in relative_paths:
        path = _repository_path(repository, relative)
        component = repository
        missing_component = False
        for part in relative.parts:
            component /= part
            try:
                mode = component.lstat().st_mode
            except FileNotFoundError:
                missing.append(str(relative))
                missing_component = True
                break
            except OSError as exc:
                raise PackagingError(f"cannot inspect release input {relative}: {exc}") from exc
            if stat.S_ISLNK(mode):
                raise PackagingError(
                    f"release input must not use a symlink: {relative} ({component})"
                )
        if missing_component:
            continue
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(repository)
        except ValueError as exc:
            raise PackagingError(
                f"release input resolves outside the repository: {relative}"
            ) from exc
        except OSError as exc:
            raise PackagingError(f"cannot resolve release input {relative}: {exc}") from exc
        if not stat.S_ISREG(path.stat().st_mode):
            raise PackagingError(f"release input is not a regular file: {relative}")
    if missing:
        raise PackagingError("missing release input(s): " + ", ".join(missing))


def _sha256_stream(stream) -> str:
    digest = hashlib.sha256()
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def _sha256_file(path: pathlib.Path) -> str:
    with path.open("rb") as stream:
        return _sha256_stream(stream)


def _load_json(path: pathlib.Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PackagingError(f"cannot read {description} at {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise PackagingError(f"invalid JSON in {description} at {path}: {exc}") from exc
    _require(isinstance(payload, dict), f"{description} must contain a JSON object")
    return payload


def _git(repository: pathlib.Path, *arguments: str, binary: bool = False):
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=not binary,
        )
    except OSError as exc:
        raise PackagingError(f"cannot execute git: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode() if binary else completed.stderr
        raise PackagingError(
            f"git {' '.join(arguments)} failed: {detail.strip() or 'unknown error'}"
        )
    return completed.stdout


def _git_blob_sha256(
    repository: pathlib.Path,
    source_tag: str,
    relative_path: pathlib.PurePosixPath,
) -> str:
    object_name = str(
        _git(repository, "rev-parse", f"{source_tag}:{relative_path}")
    ).strip()
    try:
        process = subprocess.Popen(
            ["git", "-C", str(repository), "cat-file", "blob", object_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise PackagingError(f"cannot execute git cat-file: {exc}") from exc
    assert process.stdout is not None
    digest = _sha256_stream(process.stdout)
    stderr = process.stderr.read() if process.stderr is not None else b""
    returncode = process.wait()
    if returncode != 0:
        raise PackagingError(
            f"cannot read {relative_path} from {source_tag}: "
            f"{stderr.decode(errors='replace').strip() or 'unknown git error'}"
        )
    return digest


def _verify_source_tag(
    repository: pathlib.Path,
    source_hashes: Mapping[str, str],
) -> None:
    tag_target = str(_git(repository, "rev-parse", f"{SOURCE_TAG}^{{}}")).strip()
    _require(len(tag_target) == 40, f"{SOURCE_TAG} did not resolve to a commit")
    for relative in TAG_IMMUTABLE_PATHS:
        tagged_hash = _git_blob_sha256(repository, SOURCE_TAG, relative)
        working_hash = source_hashes[str(relative)]
        _require(
            tagged_hash == working_hash,
            f"{relative} differs from frozen tag {SOURCE_TAG}: "
            f"tag {tagged_hash}, working tree {working_hash}",
        )


def _validate_inputs(
    manifest: Mapping[str, Any],
    selection: Mapping[str, Any],
    phases: Mapping[str, Mapping[str, Any]],
    manifest_hash: str,
) -> None:
    study = manifest.get("study")
    _require(isinstance(study, dict), "manifest.study must be an object")
    _require(study.get("frozen") is True, "manifest does not identify a frozen study")
    _require(
        selection.get("manifest_sha256") == manifest_hash,
        "selection lock does not match the packaged manifest",
    )
    _require(
        selection.get("test_authorized") is True,
        "selection lock does not authorize the frozen test phase",
    )

    for phase, payload in phases.items():
        _require(payload.get("phase") == phase, f"{phase}.json has the wrong phase")
        _require(
            payload.get("manifest_sha256") == manifest_hash,
            f"{phase}.json does not match the packaged manifest",
        )
        completeness = payload.get("completeness")
        _require(
            isinstance(completeness, dict),
            f"{phase}.json completeness must be an object",
        )
        _require(
            completeness.get("operationally_valid") is True,
            f"{phase}.json is not operationally valid",
        )

    test = phases["test"]
    _require(test.get("analysis_complete") is True, "test analysis is not complete")
    _require(
        test["completeness"].get("truncations") == 0,
        "test data contains truncations",
    )
    _require(isinstance(test.get("matchups"), dict), "test matchups must be an object")
    _require(
        isinstance(selection.get("validation_pareto"), list),
        "selection lock validation_pareto must be an array",
    )

    configurations = manifest.get("configurations")
    _require(isinstance(configurations, list), "manifest configurations must be an array")
    config_ids = {
        config.get("id")
        for config in configurations
        if isinstance(config, dict) and isinstance(config.get("id"), str)
    }
    selected = selection.get("selected_configurations")
    _require(isinstance(selected, dict), "selected_configurations must be an object")
    selected_ids = set(selected.values())
    rank5_id = selection.get("fixed_rank5_configuration")
    _require(isinstance(rank5_id, str), "fixed Rank5 configuration is missing")
    _require(
        selected_ids | {rank5_id} <= config_ids,
        "selection lock references an unknown configuration",
    )


def _validate_release_summaries(repository: pathlib.Path) -> None:
    try:
        expected = release_summary.generate_release_files(
            _repository_path(repository, MANIFEST_PATH),
            _repository_path(repository, SELECTION_PATH),
            _repository_path(repository, DECISION_PATHS[2]),
        )
    except release_summary.ReleaseSummaryError as exc:
        raise PackagingError(f"cannot validate release summaries: {exc}") from exc

    stale = []
    for relative in SUMMARY_PATHS:
        name = relative.name
        path = _repository_path(repository, relative)
        try:
            actual = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise PackagingError(f"cannot read release summary at {path}: {exc}") from exc
        if actual != expected[name]:
            stale.append(str(relative))
    if stale:
        raise PackagingError(
            "stale release summary file(s): "
            + ", ".join(stale)
            + "; run python3 benchmarks/flagship_study/release_summary.py --write"
        )


def _read_inputs(
    repository: pathlib.Path,
    *,
    verify_source_tag: bool,
) -> ReleaseInputs:
    repository = repository.resolve()
    all_sources = (
        *CORE_ARCHIVE_PATHS,
        *DECISION_PATHS,
        *LINEAGE_ATTACHMENT_PATHS,
    )
    _validate_source_paths(repository, all_sources)

    source_hashes = {
        str(relative): _sha256_file(_repository_path(repository, relative))
        for relative in all_sources
    }
    manifest = _load_json(_repository_path(repository, MANIFEST_PATH), "manifest")
    selection = _load_json(_repository_path(repository, SELECTION_PATH), "selection lock")
    phases = {
        phase: _load_json(
            _repository_path(repository, STUDY_ROOT / f"data/{phase}.json"),
            f"{phase} decision-level data",
        )
        for phase in ("development", "validation", "test")
    }
    _validate_inputs(
        manifest,
        selection,
        phases,
        source_hashes[str(MANIFEST_PATH)],
    )
    _validate_release_summaries(repository)
    if verify_source_tag:
        _verify_source_tag(repository, source_hashes)
    return ReleaseInputs(
        repository=repository,
        manifest=manifest,
        selection=selection,
        phases=phases,
    )


def _expected_entries(inputs: ReleaseInputs) -> tuple[list[ArchiveEntry], list[ArchiveEntry]]:
    core = [
        ArchiveEntry(str(relative), _repository_path(inputs.repository, relative))
        for relative in CORE_ARCHIVE_PATHS
    ]
    decisions = [
        ArchiveEntry(str(relative), _repository_path(inputs.repository, relative))
        for relative in DECISION_PATHS
    ]
    return core, decisions


def _zip_info(archive_path: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(archive_path, date_time=FIXED_ZIP_TIMESTAMP)
    info.create_system = 3
    info.external_attr = ZIP_MODE << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    info._compresslevel = COMPRESSION_LEVEL
    return info


def _write_archive(path: pathlib.Path, entries: Sequence[ArchiveEntry]) -> None:
    with zipfile.ZipFile(
        path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=COMPRESSION_LEVEL,
        strict_timestamps=True,
    ) as archive:
        for entry in entries:
            with entry.open() as source, archive.open(
                _zip_info(entry.archive_path), mode="w"
            ) as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)


def _entry_sha256(entry: ArchiveEntry) -> str:
    with entry.open() as stream:
        return _sha256_stream(stream)


def _verify_archive(path: pathlib.Path, entries: Sequence[ArchiveEntry]) -> None:
    if not path.is_file():
        raise PackagingError(f"missing release asset: {path}")
    expected_names = [entry.archive_path for entry in entries]
    expected_hashes = {
        entry.archive_path: _entry_sha256(entry)
        for entry in entries
    }
    try:
        with zipfile.ZipFile(path, mode="r") as archive:
            actual_names = archive.namelist()
            _require(
                actual_names == expected_names,
                f"{path.name} member order/content differs: "
                f"expected {expected_names}, got {actual_names}",
            )
            for info in archive.infolist():
                _require(
                    info.date_time == FIXED_ZIP_TIMESTAMP,
                    f"{path.name}:{info.filename} has a non-deterministic timestamp",
                )
                _require(
                    info.create_system == 3 and (info.external_attr >> 16) == ZIP_MODE,
                    f"{path.name}:{info.filename} has unexpected file metadata",
                )
                _require(
                    info.compress_type == zipfile.ZIP_DEFLATED,
                    f"{path.name}:{info.filename} is not deflate-compressed",
                )
                with archive.open(info, mode="r") as stream:
                    actual_hash = _sha256_stream(stream)
                _require(
                    actual_hash == expected_hashes[info.filename],
                    f"{path.name}:{info.filename} content does not match its source",
                )
            bad_member = archive.testzip()
            _require(bad_member is None, f"{path.name} has a corrupt member: {bad_member}")
    except (OSError, zipfile.BadZipFile) as exc:
        raise PackagingError(f"cannot verify {path}: {exc}") from exc


def _checksum_text(directory: pathlib.Path) -> str:
    rows = []
    for name in sorted((CORE_ARCHIVE_NAME, DECISION_ARCHIVE_NAME)):
        path = directory / name
        _require(path.is_file(), f"missing release asset: {path}")
        rows.append(f"{_sha256_file(path)}  {name}\n")
    return "".join(rows)


def _verify_checksums(directory: pathlib.Path) -> None:
    checksums = directory / CHECKSUMS_NAME
    if not checksums.is_file():
        raise PackagingError(f"missing release asset: {checksums}")
    try:
        actual = checksums.read_text(encoding="ascii")
    except OSError as exc:
        raise PackagingError(f"cannot read {checksums}: {exc}") from exc
    expected = _checksum_text(directory)
    _require(actual == expected, f"{CHECKSUMS_NAME} does not match the release archives")


def verify_release(
    repository: pathlib.Path,
    output_directory: pathlib.Path,
    *,
    verify_source_tag: bool = True,
) -> dict[str, str]:
    inputs = _read_inputs(repository, verify_source_tag=verify_source_tag)
    core_entries, decision_entries = _expected_entries(inputs)
    _verify_archive(output_directory / CORE_ARCHIVE_NAME, core_entries)
    _verify_archive(output_directory / DECISION_ARCHIVE_NAME, decision_entries)
    _verify_checksums(output_directory)
    return {
        name: _sha256_file(output_directory / name)
        for name in (CORE_ARCHIVE_NAME, DECISION_ARCHIVE_NAME, CHECKSUMS_NAME)
    }


def package_release(
    repository: pathlib.Path,
    output_directory: pathlib.Path,
    *,
    verify_source_tag: bool = True,
) -> dict[str, str]:
    inputs = _read_inputs(repository, verify_source_tag=verify_source_tag)
    core_entries, decision_entries = _expected_entries(inputs)
    output_directory = output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{RELEASE_ID}-", dir=output_directory
    ) as temporary:
        staging = pathlib.Path(temporary)
        _write_archive(staging / CORE_ARCHIVE_NAME, core_entries)
        _write_archive(staging / DECISION_ARCHIVE_NAME, decision_entries)
        (staging / CHECKSUMS_NAME).write_text(
            _checksum_text(staging), encoding="ascii", newline="\n"
        )
        _verify_archive(staging / CORE_ARCHIVE_NAME, core_entries)
        _verify_archive(staging / DECISION_ARCHIVE_NAME, decision_entries)
        _verify_checksums(staging)
        for name in (CORE_ARCHIVE_NAME, DECISION_ARCHIVE_NAME, CHECKSUMS_NAME):
            os.replace(staging / name, output_directory / name)

    return verify_release(
        repository,
        output_directory,
        verify_source_tag=verify_source_tag,
    )


def _default_repository() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        choices=("build", "check"),
        default="build",
        help="build assets (default) or verify existing assets",
    )
    parser.add_argument(
        "--repository",
        type=pathlib.Path,
        default=_default_repository(),
        help="repository root (defaults to the package script's checkout)",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        help=(
            "asset directory (defaults to "
            "results/releases/flagship-study-v4 under the repository)"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    repository = args.repository.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else repository / "results/releases" / RELEASE_ID
    )
    try:
        if args.command == "build":
            hashes = package_release(repository, output)
            verb = "Built and verified"
        else:
            hashes = verify_release(repository, output)
            verb = "Verified"
    except PackagingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"{verb} release assets in {output}")
    for name in (CORE_ARCHIVE_NAME, DECISION_ARCHIVE_NAME, CHECKSUMS_NAME):
        print(f"{hashes[name]}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
