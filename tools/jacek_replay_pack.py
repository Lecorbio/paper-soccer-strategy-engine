#!/usr/bin/env python3
"""Bridge normalized replay roots, the Rank-4 teacher, and CSR shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import sqlite3
import struct
import sys
import tempfile
import zipfile


TOOL_DIR = pathlib.Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))
import jacek_replay_corpus as corpus  # noqa: E402
import jacek_replay_features as features  # noqa: E402


SPLITS = ("train", "validation", "test")


def load_roots(path: pathlib.Path) -> dict:
    try:
        manifest = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid roots manifest: {path}") from error
    if not isinstance(manifest, dict) or manifest.get("schema") != corpus.ROOT_SCHEMA:
        raise ValueError("unexpected replay-roots schema")
    body_sha256 = manifest.get("body_sha256")
    if (
        not isinstance(body_sha256, str)
        or len(body_sha256) != 64
        or any(character not in "0123456789abcdef" for character in body_sha256)
    ):
        raise ValueError("roots manifest has no valid body SHA-256")
    body = dict(manifest)
    del body["body_sha256"]
    if corpus.sha256_bytes(corpus.canonical_json_bytes(body)) != body_sha256:
        raise ValueError("roots manifest body SHA-256 mismatch")
    if manifest.get("feature_schema") != features.FEATURE_SCHEMA:
        raise ValueError("roots manifest feature schema mismatch")
    tool_hashes = manifest.get("tool_sha256")
    if (
        not isinstance(tool_hashes, dict)
        or set(tool_hashes) != {"normalizer", "features"}
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in tool_hashes.values()
        )
    ):
        raise ValueError("roots manifest tool provenance is invalid")
    boundary = manifest.get("exclusion_boundary")
    if not isinstance(boundary, dict) or boundary.get(
        "read_before_candidate_sources"
    ) is not True:
        raise ValueError("roots manifest does not bind an exclusion-first boundary")
    accepted = manifest.get("accepted")
    if not isinstance(accepted, list) or not accepted:
        raise ValueError("roots manifest has no accepted records")
    return manifest


def teacher_tsv_bytes(manifest: dict) -> bytes:
    lines = ["group_id\tsource\twinner\ttranscript"]
    for record in manifest["accepted"]:
        if not isinstance(record, dict):
            raise ValueError("accepted root must be an object")
        group_id, source, winner, turns = (
            record.get("group_id"),
            record.get("source"),
            record.get("winner"),
            record.get("turns"),
        )
        if (
            not isinstance(group_id, str)
            or not group_id
            or "\t" in group_id
            or not isinstance(source, str)
            or not source
            or "\t" in source
            or winner not in (0, 1)
            or not isinstance(turns, list)
            or not turns
        ):
            raise ValueError("accepted root cannot be represented in teacher TSV")
        actions = []
        for expected_player, turn in enumerate(turns):
            if (
                not isinstance(turn, dict)
                or turn.get("player_id") != expected_player % 2
                or not isinstance(turn.get("action"), str)
                or not turn["action"]
            ):
                raise ValueError("accepted root turn is not a canonical complete turn")
            actions.append(turn["action"])
        lines.append(f"{group_id}\t{source}\t{winner}\t{'/'.join(actions)}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def frozen_assignments(manifest: dict) -> dict[str, str]:
    result = {}
    for record in manifest["accepted"]:
        group_id, split = record.get("group_id"), record.get("split")
        if not isinstance(group_id, str) or split not in {"train", "validation", "test"}:
            raise ValueError("roots manifest contains an invalid split assignment")
        if group_id in result:
            raise ValueError("roots manifest repeats a group id")
        result[group_id] = split
    return result


def _iter_teacher_samples(paths: list[pathlib.Path]):
    for path in paths:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    yield from corpus.sample_from_teacher_row(json.loads(line))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error


def _stream_file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_npz_from_npy_files(
    path: pathlib.Path, arrays: dict[str, pathlib.Path]
) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(
                f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0)
            )
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            with arrays[name].open("rb") as source, archive.open(
                info, "w", force_zip64=True
            ) as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)


def _write_streaming_shard(
    *,
    training,
    connection: sqlite3.Connection,
    directory: pathlib.Path,
    split: str,
    split_index: int,
    provenance: dict,
) -> tuple[pathlib.Path, pathlib.Path, dict]:
    import numpy as np

    count, active_bytes = connection.execute(
        "SELECT COUNT(*), COALESCE(SUM(LENGTH(active)), 0) "
        "FROM observations WHERE split = ?",
        (split_index,),
    ).fetchone()
    count, active_count = int(count), int(active_bytes) // 2
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=directory, prefix=f".jacek-replay-{split}."
    ) as temporary_name:
        temporary = pathlib.Path(temporary_name)
        array_paths = {
            name: temporary / f"{name}.npy"
            for name in ("group_ids", "indices", "indptr", "targets", "weights")
        }
        indptr = np.lib.format.open_memmap(
            array_paths["indptr"], mode="w+", dtype="<i8", shape=(count + 1,)
        )
        indices = np.lib.format.open_memmap(
            array_paths["indices"], mode="w+", dtype="<u2", shape=(active_count,)
        )
        targets = np.lib.format.open_memmap(
            array_paths["targets"], mode="w+", dtype="<f4", shape=(count,)
        )
        weights = np.lib.format.open_memmap(
            array_paths["weights"], mode="w+", dtype="<f4", shape=(count,)
        )
        group_ids = np.lib.format.open_memmap(
            array_paths["group_ids"],
            mode="w+",
            dtype="V32",
            shape=(count,),
        )
        indptr[0] = 0
        offset = 0
        cursor = connection.execute(
            "SELECT active, weighted_target, weight FROM observations "
            "WHERE split = ? ORDER BY active",
            (split_index,),
        )
        for row, (active_blob, weighted_target, weight) in enumerate(cursor):
            active = np.frombuffer(active_blob, dtype="<u2")
            next_offset = offset + len(active)
            indices[offset:next_offset] = active
            indptr[row + 1] = next_offset
            targets[row] = float(weighted_target) / float(weight)
            weights[row] = float(weight)
            group_ids[row] = np.void(
                hashlib.sha256(split.encode() + active_blob).digest()
            )
            offset = next_offset
        for array in (indptr, indices, targets, weights, group_ids):
            array.flush()
        del indptr, indices, targets, weights, group_ids
        temporary_npz = temporary / "shard.npz"
        _write_npz_from_npy_files(temporary_npz, array_paths)
        npz_digest = _stream_file_sha256(temporary_npz)
        npz_path = directory / f"{npz_digest}.npz"
        if npz_path.exists():
            if _stream_file_sha256(npz_path) != npz_digest:
                raise RuntimeError("content-addressed streaming shard conflicts")
        else:
            try:
                os.link(temporary_npz, npz_path)
                os.chmod(npz_path, 0o644)
            except FileExistsError:
                if _stream_file_sha256(npz_path) != npz_digest:
                    raise RuntimeError(
                        "content-addressed streaming shard raced with conflict"
                    )
    manifest = {
        "schema": training.SHARD_SCHEMA,
        "feature_schema": features.FEATURE_SCHEMA,
        "split": split,
        "npz": npz_path.name,
        "npz_sha256": npz_digest,
        "samples": count,
        "active_features": active_count,
        "array_contract": {
            "indptr": "little-endian-int64[n+1]",
            "indices": "little-endian-uint16[nnz]",
            "targets": "little-endian-float32[n]",
            "weights": "little-endian-float32[n]",
            "group_ids": "raw-sha256-32bytes[n]",
        },
        "provenance": provenance,
    }
    manifest_bytes = corpus.canonical_json_bytes(manifest)
    manifest_path = directory / f"{hashlib.sha256(manifest_bytes).hexdigest()}.json"
    training._write_once(manifest_path, manifest_bytes)
    return npz_path, manifest_path, manifest


def pack_teacher_rows(
    *,
    roots_path: pathlib.Path,
    teacher_paths: list[pathlib.Path],
    output_directory: pathlib.Path,
) -> dict:
    # NumPy is needed only for this command, keeping teacher TSV emission
    # available in minimal collector environments.
    try:
        import jacek_replay_train as training
    except ModuleNotFoundError as error:
        if error.name == "numpy":
            raise RuntimeError(
                "packing requires NumPy; install requirements-research.txt"
            ) from error
        raise
    roots = load_roots(roots_path)
    assignments = frozen_assignments(roots)
    samples = corpus.load_teacher_rows(teacher_paths)
    retained, removed, aggregated = corpus.split_and_purge_samples(
        samples, assignments
    )
    empty = [split for split, rows in retained.items() if not rows]
    if empty:
        raise ValueError("overlap purge left empty splits: " + ", ".join(empty))
    root_sha = corpus.sha256_file(roots_path)
    tool_hashes = {
        "pack": corpus.sha256_file(pathlib.Path(__file__)),
        "corpus": corpus.sha256_file(pathlib.Path(corpus.__file__)),
        "features": corpus.sha256_file(pathlib.Path(features.__file__)),
    }
    teacher_hashes = [
        {"name": path.name, "sha256": _stream_file_sha256(path)}
        for path in sorted(teacher_paths)
    ]
    if len({record["name"] for record in teacher_hashes}) != len(teacher_hashes):
        raise ValueError("teacher JSONL inputs must have distinct basenames")
    shards = {}
    for split in ("train", "validation", "test"):
        npz_path, manifest_path, shard_manifest = training.write_csr_shard(
            output_directory,
            split,
            retained[split],
            provenance={
                "roots_manifest_sha256": root_sha,
                "teacher_jsonl_sha256": teacher_hashes,
                "tool_sha256": tool_hashes,
                "reflection_augmentation": True,
                "deduplication_policy": (
                    "exact-orientation rows aggregate by summed weight and weighted "
                    "target within split; canonical rotate/reflection overlap is "
                    "purged only across train, validation, and test"
                ),
            },
        )
        shards[split] = {
            "manifest": str(manifest_path),
            "manifest_sha256": corpus.sha256_file(manifest_path),
            "npz": str(npz_path),
            "samples": shard_manifest["samples"],
            "sha256": shard_manifest["npz_sha256"],
        }
    report = {
        "schema": "papersoccer.jacek-replay-pack-report.v1",
        "roots_manifest": str(roots_path),
        "roots_manifest_sha256": root_sha,
        "teacher_jsonl_sha256": teacher_hashes,
        "tool_sha256": tool_hashes,
        "input_samples_after_reflection": len(samples),
        "cross_split_canonical_rows_removed": removed,
        "same_orientation_rows_aggregated": aggregated,
        "shards": shards,
    }
    report_path = output_directory / "pack-report.json"
    report_path.write_bytes(corpus.canonical_json_bytes(report))
    return {**report, "report": str(report_path)}


def pack_teacher_rows_streaming(
    *,
    roots_path: pathlib.Path,
    teacher_paths: list[pathlib.Path],
    output_directory: pathlib.Path,
    prior_shard_manifests: list[pathlib.Path] | None = None,
) -> dict:
    """Disk-backed equivalent of :func:`pack_teacher_rows` for large rounds."""

    try:
        import jacek_replay_train as training
    except ModuleNotFoundError as error:
        if error.name == "numpy":
            raise RuntimeError(
                "packing requires NumPy; install requirements-research.txt"
            ) from error
        raise
    roots = load_roots(roots_path)
    assignments = frozen_assignments(roots)
    output_directory.mkdir(parents=True, exist_ok=True)
    root_sha = corpus.sha256_file(roots_path)
    tool_hashes = {
        "pack": corpus.sha256_file(pathlib.Path(__file__)),
        "corpus": corpus.sha256_file(pathlib.Path(corpus.__file__)),
        "features": corpus.sha256_file(pathlib.Path(features.__file__)),
    }
    teacher_hashes = [
        {"name": path.name, "sha256": _stream_file_sha256(path)}
        for path in sorted(teacher_paths)
    ]
    if len({record["name"] for record in teacher_hashes}) != len(teacher_hashes):
        raise ValueError("teacher JSONL inputs must have distinct basenames")
    removed = {split: 0 for split in SPLITS}
    prior_removed = {split: 0 for split in SPLITS}
    accepted = {split: 0 for split in SPLITS}
    aggregated = {split: 0 for split in SPLITS}
    with tempfile.TemporaryDirectory(
        dir=output_directory, prefix=".jacek-replay-sqlite."
    ) as temporary_name:
        database_path = pathlib.Path(temporary_name) / "pack.sqlite3"
        connection = sqlite3.connect(database_path)
        try:
            connection.execute("PRAGMA journal_mode = OFF")
            connection.execute("PRAGMA synchronous = OFF")
            connection.execute("PRAGMA temp_store = FILE")
            connection.execute(
                "CREATE TABLE canonical (fingerprint BLOB PRIMARY KEY, priority INTEGER)"
            )
            connection.execute(
                "CREATE TABLE prior_canonical ("
                "fingerprint BLOB PRIMARY KEY, split INTEGER)"
            )
            connection.execute(
                "CREATE TABLE observations ("
                "split INTEGER, active BLOB, fingerprint BLOB, weighted_target REAL, "
                "weight REAL, observations INTEGER, "
                "PRIMARY KEY (split, active)) WITHOUT ROWID"
            )
            prior_shards = []
            for manifest_path in prior_shard_manifests or []:
                shard = training.load_csr_shard(manifest_path)
                split_index = SPLITS.index(shard.split)
                with connection:
                    for row in range(len(shard)):
                        fingerprint = corpus.canonical_feature_fingerprint(
                            shard.active(row).tolist()
                        )
                        existing = connection.execute(
                            "SELECT split FROM prior_canonical WHERE fingerprint = ?",
                            (fingerprint,),
                        ).fetchone()
                        if existing is not None and int(existing[0]) != split_index:
                            raise ValueError(
                                "prior shards contain a cross-split canonical overlap"
                            )
                        connection.execute(
                            "INSERT OR IGNORE INTO prior_canonical VALUES (?, ?)",
                            (fingerprint, split_index),
                        )
                prior_shards.append(
                    {
                        "manifest_sha256": _stream_file_sha256(manifest_path),
                        "npz_sha256": shard.npz_sha256,
                        "split": shard.split,
                    }
                )
                del shard
            with connection:
                for sample in _iter_teacher_samples(teacher_paths):
                    assigned = assignments.get(sample.group_id)
                    if assigned not in SPLITS:
                        raise ValueError(
                            f"no frozen split for teacher group {sample.group_id}"
                        )
                    split_index = SPLITS.index(assigned)
                    fingerprint = corpus.canonical_feature_fingerprint(sample.active)
                    connection.execute(
                        "INSERT INTO canonical VALUES (?, ?) "
                        "ON CONFLICT(fingerprint) DO UPDATE SET "
                        "priority = MIN(priority, excluded.priority)",
                        (fingerprint, split_index),
                    )
                    active_blob = struct.pack(
                        f"<{len(sample.active)}H", *sample.active
                    )
                    connection.execute(
                        "INSERT INTO observations VALUES (?, ?, ?, ?, ?, 1) "
                        "ON CONFLICT(split, active) DO UPDATE SET "
                        "weighted_target = weighted_target + excluded.weighted_target, "
                        "weight = weight + excluded.weight, "
                        "observations = observations + 1",
                        (
                            split_index,
                            active_blob,
                            fingerprint,
                            sample.target * sample.weight,
                            sample.weight,
                        ),
                    )
                    accepted[assigned] += 1
            for split_index, split in enumerate(SPLITS):
                prior_removed[split] = int(
                    connection.execute(
                        "SELECT COALESCE(SUM(observations), 0) FROM observations o "
                        "JOIN prior_canonical p ON p.fingerprint = o.fingerprint "
                        "WHERE o.split = ? AND p.split != o.split",
                        (split_index,),
                    ).fetchone()[0]
                )
                with connection:
                    connection.execute(
                        "DELETE FROM observations WHERE split = ? AND EXISTS ("
                        "SELECT 1 FROM prior_canonical p WHERE "
                        "p.fingerprint = observations.fingerprint "
                        "AND p.split != observations.split)",
                        (split_index,),
                    )
            with connection:
                connection.execute("DELETE FROM canonical")
                connection.execute(
                    "INSERT INTO canonical "
                    "SELECT fingerprint, MIN(split) FROM observations "
                    "GROUP BY fingerprint"
                )
            for split_index, split in enumerate(SPLITS):
                current_removed = int(
                    connection.execute(
                        "SELECT COALESCE(SUM(observations), 0) FROM observations o "
                        "JOIN canonical c ON c.fingerprint = o.fingerprint "
                        "WHERE o.split = ? AND c.priority < o.split",
                        (split_index,),
                    ).fetchone()[0]
                )
                removed[split] = prior_removed[split] + current_removed
                with connection:
                    connection.execute(
                        "DELETE FROM observations WHERE split = ? AND EXISTS ("
                        "SELECT 1 FROM canonical c WHERE "
                        "c.fingerprint = observations.fingerprint "
                        "AND c.priority < observations.split)",
                        (split_index,),
                    )
                unique = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM observations WHERE split = ?",
                        (split_index,),
                    ).fetchone()[0]
                )
                aggregated[split] = accepted[split] - removed[split] - unique

            provenance = {
                "roots_manifest_sha256": root_sha,
                "teacher_jsonl_sha256": teacher_hashes,
                "tool_sha256": tool_hashes,
                "reflection_augmentation": True,
                "packing": "sqlite-streaming-bounded-memory-v1",
                "prior_shards": prior_shards,
                "deduplication_policy": (
                    "exact-orientation rows aggregate by summed weight and weighted "
                    "target within split; canonical rotate/reflection overlap is "
                    "purged only across train, validation, and test"
                ),
            }
            shards = {}
            for split_index, split in enumerate(SPLITS):
                npz_path, manifest_path, shard_manifest = _write_streaming_shard(
                    training=training,
                    connection=connection,
                    directory=output_directory,
                    split=split,
                    split_index=split_index,
                    provenance=provenance,
                )
                shards[split] = {
                    "manifest": str(manifest_path),
                    "manifest_sha256": corpus.sha256_file(manifest_path),
                    "npz": str(npz_path),
                    "samples": shard_manifest["samples"],
                    "sha256": shard_manifest["npz_sha256"],
                }
        finally:
            connection.close()
    report = {
        "schema": "papersoccer.jacek-replay-pack-report.v1",
        "packing": "sqlite-streaming-bounded-memory-v1",
        "roots_manifest": str(roots_path),
        "roots_manifest_sha256": root_sha,
        "teacher_jsonl_sha256": teacher_hashes,
        "tool_sha256": tool_hashes,
        "input_samples_after_reflection": sum(accepted.values()),
        "cross_split_canonical_rows_removed": removed,
        "same_orientation_rows_aggregated": aggregated,
        "prior_cross_split_rows_removed": prior_removed,
        "prior_shards": prior_shards,
        "shards": shards,
    }
    report_path = output_directory / "pack-report.json"
    report_path.write_bytes(corpus.canonical_json_bytes(report))
    return {**report, "report": str(report_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    teacher = commands.add_parser("teacher-tsv", help="emit roots for the C++ teacher")
    teacher.add_argument("--roots", type=pathlib.Path, required=True)
    teacher.add_argument("--output", type=pathlib.Path, required=True)

    pack = commands.add_parser("pack", help="convert teacher JSONL to CSR shards")
    pack.add_argument("--roots", type=pathlib.Path, required=True)
    pack.add_argument("--teacher", type=pathlib.Path, action="append", required=True)
    pack.add_argument("--output-directory", type=pathlib.Path, required=True)
    pack.add_argument(
        "--streaming",
        action="store_true",
        help="use bounded-memory SQLite aggregation for article-scale corpora",
    )
    pack.add_argument(
        "--prior-shard-manifest",
        type=pathlib.Path,
        action="append",
        default=[],
        help="bind a prior-round shard when purging cross-split overlap",
    )

    arguments = parser.parse_args()
    if arguments.command == "teacher-tsv":
        payload = teacher_tsv_bytes(load_roots(arguments.roots))
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_bytes(payload)
        print(json.dumps({"output": str(arguments.output), "roots": len(payload.splitlines()) - 1}))
        return 0
    pack_function = (
        pack_teacher_rows_streaming if arguments.streaming else pack_teacher_rows
    )
    if arguments.prior_shard_manifest and not arguments.streaming:
        parser.error("--prior-shard-manifest requires --streaming")
    pack_arguments = {
        "roots_path": arguments.roots,
        "teacher_paths": arguments.teacher,
        "output_directory": arguments.output_directory,
    }
    if arguments.streaming:
        pack_arguments["prior_shard_manifests"] = arguments.prior_shard_manifest
    report = pack_function(
        **pack_arguments,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
