#!/usr/bin/env python3
"""Derive the exact deployable source for a discrete-v3 finalist.

Development evaluates a content-addressed generated source with its search
configuration supplied by the Rank-4 gate.  CodinGame, however, executes that
source without gate arguments.  This module performs the only permitted
deployment transformation: seven exact default declarations in the generated
source are replaced by the selected search tuple/profile and the campaign's
fixed shuffle seed.  Every other byte, including the quantized model payload,
must remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import secrets
import stat
from collections.abc import Mapping, Sequence
from typing import Any


DERIVATION_SCHEMA = (
    "papersoccer.compact-value-bfm.discrete-v3-deployment-source-derivation.v1"
)
MANIFEST_SCHEMA = (
    "papersoccer.compact-value-bfm.discrete-v3-deployment-source-manifest.v1"
)
ALGORITHM = "exact-seven-default-declaration-replacements-v1"
SOURCE_LIMIT_EXCLUSIVE = 95_000
REPOSITORY = pathlib.Path(__file__).resolve().parents[1]
CANDIDATE_RELATIVE = pathlib.Path(
    "submissions/codingame/bots/compact_value_bfm/discrete_v3_deployment.cpp"
)
MANIFEST_RELATIVE = pathlib.Path(
    "submissions/codingame/bots/compact_value_bfm/discrete_v3_deployment.json"
)

TUPLE_ROSTER = (
    ("0.65", "0.5", "1"),
    ("0.80", "0.5", "1"),
    ("0.95", "0.5", "1"),
    ("1.10", "0.5", "1"),
    ("0.95", "0.25", "1"),
    ("0.95", "0.75", "1"),
    ("0.95", "0.5", "0.5"),
    ("0.95", "0.5", "0"),
)
PROFILE_ROSTER = {
    "light": {
        "root_partial_paths": 2_000,
        "nonroot_partial_paths": 256,
        "nodes": 60_000,
    },
    "default": {
        "root_partial_paths": 4_000,
        "nonroot_partial_paths": 512,
        "nodes": 80_000,
    },
    "heavy": {
        "root_partial_paths": 8_000,
        "nonroot_partial_paths": 512,
        "nodes": 120_000,
    },
}

FIXED_ACTIONS = 250
FIXED_EXPANSIONS = 2_000_000
FIXED_SHUFFLE_SEED = 1


class DeploymentSourceError(ValueError):
    """The source cannot be proven to be the exact deployment derivative."""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _source_record(value: bytes) -> dict[str, Any]:
    try:
        value.decode("ascii")
    except UnicodeDecodeError as error:
        raise DeploymentSourceError("deployment source is not ASCII") from error
    return {"bytes": len(value), "sha256": _sha256(value), "ascii": True}


def deployment_configuration(
    search_tuple: Sequence[Any], profile: Any, work: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the one canonical deployment configuration for a finalist."""

    if (
        not isinstance(search_tuple, Sequence)
        or isinstance(search_tuple, (str, bytes))
        or len(search_tuple) != 3
        or any(not isinstance(value, str) for value in search_tuple)
    ):
        raise DeploymentSourceError("deployment tuple is not three exact strings")
    selected_tuple = tuple(search_tuple)
    if selected_tuple not in TUPLE_ROSTER:
        raise DeploymentSourceError("deployment tuple is outside the frozen roster")
    if not isinstance(profile, str) or profile not in PROFILE_ROSTER:
        raise DeploymentSourceError("deployment profile is outside the frozen roster")
    if not isinstance(work, Mapping) or dict(work) != PROFILE_ROSTER[profile]:
        raise DeploymentSourceError("deployment work differs from its frozen profile")
    return {
        "tuple": list(selected_tuple),
        "profile": profile,
        "candidate_c": float(selected_tuple[0]),
        "candidate_fpu": float(selected_tuple[1]),
        "candidate_lambda": float(selected_tuple[2]),
        "candidate_actions": FIXED_ACTIONS,
        "candidate_root_partial_paths": work["root_partial_paths"],
        "candidate_nonroot_partial_paths": work["nonroot_partial_paths"],
        "candidate_nodes": work["nodes"],
        "candidate_expansions": FIXED_EXPANSIONS,
        "candidate_shuffle_seed": FIXED_SHUFFLE_SEED,
    }


def _size_literal(value: int) -> str:
    return f"{value:,}".replace(",", "'")


def _double_literal(value: str) -> str:
    return {"0": "0.0", "1": "1.0"}.get(value, value)


def _replacement_pairs(configuration: Mapping[str, Any]) -> tuple[tuple[bytes, bytes], ...]:
    selected_tuple = configuration["tuple"]
    return (
        (
            b"inline constexpr std::size_t kRootPartialPaths = 4'000;",
            (
                "inline constexpr std::size_t kRootPartialPaths = "
                f"{_size_literal(configuration['candidate_root_partial_paths'])};"
            ).encode("ascii"),
        ),
        (
            b"inline constexpr std::size_t kNonrootPartialPaths = 512;",
            (
                "inline constexpr std::size_t kNonrootPartialPaths = "
                f"{_size_literal(configuration['candidate_nonroot_partial_paths'])};"
            ).encode("ascii"),
        ),
        (
            b"inline constexpr std::size_t kProductionTreeNodes = 80'000;",
            (
                "inline constexpr std::size_t kProductionTreeNodes = "
                f"{_size_literal(configuration['candidate_nodes'])};"
            ).encode("ascii"),
        ),
        (
            b"inline constexpr double kExploration = 0.95;",
            (
                "inline constexpr double kExploration = "
                f"{_double_literal(selected_tuple[0])};"
            ).encode("ascii"),
        ),
        (
            b"inline constexpr double kFirstPlayUrgency = 0.5;",
            (
                "inline constexpr double kFirstPlayUrgency = "
                f"{_double_literal(selected_tuple[1])};"
            ).encode("ascii"),
        ),
        (
            b"inline constexpr double kFinalVisitWeight = 1.0;",
            (
                "inline constexpr double kFinalVisitWeight = "
                f"{_double_literal(selected_tuple[2])};"
            ).encode("ascii"),
        ),
        (
            b"std::uint64_t shuffle_seed{0x6a09e667f3bcc909ULL};",
            b"std::uint64_t shuffle_seed{1ULL};",
        ),
    )


def derive_source(
    generated_source: bytes, *, search_tuple: Sequence[Any], profile: Any,
    work: Mapping[str, Any],
) -> bytes:
    """Return the exact source that must be committed, gated, and uploaded."""

    if not isinstance(generated_source, bytes) or not generated_source:
        raise DeploymentSourceError("generated deployment source is empty or not bytes")
    _source_record(generated_source)
    configuration = deployment_configuration(search_tuple, profile, work)
    result = generated_source
    for original, replacement in _replacement_pairs(configuration):
        if generated_source.count(original) != 1:
            raise DeploymentSourceError(
                "generated source does not contain exactly one frozen default declaration"
            )
        result = result.replace(original, replacement, 1)
    if not 0 < len(result) < SOURCE_LIMIT_EXCLUSIVE:
        raise DeploymentSourceError(
            "derived deployment source is not strictly below 95,000 bytes"
        )
    _source_record(result)
    return result


def attest_derivation(
    generated_source: bytes, candidate_source: bytes, *,
    search_tuple: Sequence[Any], profile: Any, work: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed unless ``candidate_source`` is the exact derivative."""

    expected = derive_source(
        generated_source, search_tuple=search_tuple, profile=profile, work=work
    )
    if not isinstance(candidate_source, bytes) or candidate_source != expected:
        raise DeploymentSourceError(
            "committed candidate is not the exact configured deployment source"
        )
    configuration = deployment_configuration(search_tuple, profile, work)
    return {
        "schema": DERIVATION_SCHEMA,
        "algorithm": ALGORITHM,
        "base_source": _source_record(generated_source),
        "deployed_source": _source_record(candidate_source),
        "configuration": configuration,
        "replacement_slots": 7,
        "only_declared_configuration_changed": True,
    }


def recover_generated_source(
    candidate_source: bytes, *, search_tuple: Sequence[Any], profile: Any,
    work: Mapping[str, Any],
) -> bytes:
    """Reverse the seven permitted slots and recover the frozen base bytes."""

    if not isinstance(candidate_source, bytes) or not candidate_source:
        raise DeploymentSourceError("candidate deployment source is empty or not bytes")
    _source_record(candidate_source)
    configuration = deployment_configuration(search_tuple, profile, work)
    result = candidate_source
    for original, replacement in _replacement_pairs(configuration):
        if candidate_source.count(replacement) != 1:
            raise DeploymentSourceError(
                "candidate source does not contain exactly one configured declaration"
            )
        result = result.replace(replacement, original, 1)
    return result


def _manifest_body(derivation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": MANIFEST_SCHEMA,
        "algorithm": ALGORITHM,
        "base_source": dict(derivation["base_source"]),
        "deployed_source": dict(derivation["deployed_source"]),
        "configuration": dict(derivation["configuration"]),
        "replacement_slots": 7,
        "only_declared_configuration_changed": True,
    }


def create_manifest(
    generated_source: bytes, candidate_source: bytes, *,
    search_tuple: Sequence[Any], profile: Any, work: Mapping[str, Any],
) -> dict[str, Any]:
    derivation = attest_derivation(
        generated_source, candidate_source, search_tuple=search_tuple,
        profile=profile, work=work,
    )
    body = _manifest_body(derivation)
    body["body_sha256"] = hashlib.sha256(
        json.dumps(
            body, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()
    return body


def validate_manifest(value: Any, candidate_source: bytes) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DeploymentSourceError("deployment manifest is not an object")
    body = {key: item for key, item in value.items() if key != "body_sha256"}
    if (
        set(value) != {
            "schema", "algorithm", "base_source", "deployed_source",
            "configuration", "replacement_slots",
            "only_declared_configuration_changed", "body_sha256",
        }
        or value.get("schema") != MANIFEST_SCHEMA
        or value.get("algorithm") != ALGORITHM
        or value.get("replacement_slots") != 7
        or value.get("only_declared_configuration_changed") is not True
        or value.get("body_sha256") != hashlib.sha256(
            json.dumps(
                body, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            ).encode("ascii")
        ).hexdigest()
    ):
        raise DeploymentSourceError("deployment manifest seal/contract is invalid")
    configuration = value.get("configuration")
    if not isinstance(configuration, Mapping):
        raise DeploymentSourceError("deployment manifest configuration is absent")
    search_tuple = configuration.get("tuple")
    profile = configuration.get("profile")
    work = PROFILE_ROSTER.get(profile) if isinstance(profile, str) else None
    expected_configuration = deployment_configuration(search_tuple, profile, work)
    if dict(configuration) != expected_configuration:
        raise DeploymentSourceError("deployment manifest configuration changed")
    recovered = recover_generated_source(
        candidate_source, search_tuple=search_tuple, profile=profile, work=work
    )
    expected = create_manifest(
        recovered, candidate_source, search_tuple=search_tuple,
        profile=profile, work=work,
    )
    if dict(value) != expected:
        raise DeploymentSourceError("deployment manifest/source binding changed")
    return dict(value)


def verify_manifest_file(path: pathlib.Path, candidate: pathlib.Path) -> dict[str, Any]:
    if (
        path.is_symlink() or not path.is_file()
        or candidate.is_symlink() or not candidate.is_file()
    ):
        raise DeploymentSourceError("deployment manifest/candidate is absent or redirected")
    try:
        value = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DeploymentSourceError("deployment manifest JSON is malformed") from error
    return validate_manifest(value, candidate.read_bytes())


def _canonical_manifest_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("ascii")


def _read_regular_file(path: pathlib.Path, label: str) -> bytes:
    """Read one exact regular file without following a final symlink."""

    try:
        before = path.lstat()
    except FileNotFoundError as error:
        raise DeploymentSourceError(f"{label} is absent") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise DeploymentSourceError(f"{label} is redirected or irregular")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise DeploymentSourceError(f"cannot safely open {label}") from error
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise DeploymentSourceError(f"{label} changed while opening")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            return stream.read()
    finally:
        os.close(descriptor)


def _safe_relative_output(relative: pathlib.Path, label: str) -> pathlib.Path:
    if (
        relative.is_absolute() or not relative.parts
        or any(part in ("", ".", "..") for part in relative.parts)
        or relative.name in ("", ".", "..")
    ):
        raise DeploymentSourceError(f"fixed {label} route escaped the repository")
    return relative


def _directory_flags() -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise DeploymentSourceError("platform lacks no-follow directory operations")
    return os.O_RDONLY | nofollow | directory


def _open_repository(repository: pathlib.Path) -> tuple[pathlib.Path, int]:
    requested = repository.absolute()
    try:
        requested_mode = requested.lstat().st_mode
    except FileNotFoundError as error:
        raise DeploymentSourceError("deployment repository is absent") from error
    if stat.S_ISLNK(requested_mode) or not stat.S_ISDIR(requested_mode):
        raise DeploymentSourceError("deployment repository is redirected or irregular")
    try:
        canonical = requested.resolve(strict=True)
        descriptor = os.open(canonical, _directory_flags())
    except OSError as error:
        raise DeploymentSourceError("cannot safely anchor deployment repository") from error
    try:
        opened = os.fstat(descriptor)
        canonical_stat = canonical.lstat()
        if (
            not stat.S_ISDIR(opened.st_mode)
            or stat.S_ISLNK(canonical_stat.st_mode)
            or not stat.S_ISDIR(canonical_stat.st_mode)
            or (opened.st_dev, opened.st_ino)
            != (canonical_stat.st_dev, canonical_stat.st_ino)
        ):
            raise DeploymentSourceError(
                "deployment repository changed while anchoring"
            )
    except BaseException:
        os.close(descriptor)
        raise
    return canonical, descriptor


def _verify_repository_route(repository: pathlib.Path, expected: int) -> None:
    try:
        mode = repository.lstat().st_mode
        current = os.open(repository, _directory_flags())
    except OSError as error:
        raise DeploymentSourceError("deployment repository route changed") from error
    try:
        if (
            stat.S_ISLNK(mode) or not stat.S_ISDIR(mode)
            or not _same_open_directory(current, expected)
        ):
            raise DeploymentSourceError("deployment repository route changed")
    finally:
        os.close(current)


def _open_directory_tree(
    repository_descriptor: int, relative: pathlib.Path, *, create: bool,
) -> int:
    """Open one fixed directory route without following any component symlink."""

    current = os.dup(repository_descriptor)
    try:
        for part in relative.parts:
            try:
                following = os.open(
                    part, _directory_flags(), dir_fd=current,
                )
            except FileNotFoundError:
                if not create:
                    raise DeploymentSourceError("deployment output parent is absent")
                try:
                    os.mkdir(part, mode=0o755, dir_fd=current)
                    os.fsync(current)
                except FileExistsError:
                    pass
                try:
                    following = os.open(
                        part, _directory_flags(), dir_fd=current,
                    )
                except OSError as error:
                    raise DeploymentSourceError(
                        "deployment output parent is redirected or irregular"
                    ) from error
            except OSError as error:
                raise DeploymentSourceError(
                    "deployment output parent is redirected or irregular"
                ) from error
            os.close(current)
            current = following
        return current
    except BaseException:
        os.close(current)
        raise


def _same_open_directory(left: int, right: int) -> bool:
    left_stat = os.fstat(left)
    right_stat = os.fstat(right)
    return (left_stat.st_dev, left_stat.st_ino) == (right_stat.st_dev, right_stat.st_ino)


def _verify_directory_route(
    repository_descriptor: int, relative: pathlib.Path, expected: int,
) -> None:
    current = _open_directory_tree(repository_descriptor, relative, create=False)
    try:
        if not _same_open_directory(current, expected):
            raise DeploymentSourceError("deployment output parent route changed")
    finally:
        os.close(current)


def _read_regular_at(
    directory_descriptor: int, name: str, label: str, *, optional: bool = False,
) -> bytes | None:
    try:
        before = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        if optional:
            return None
        raise DeploymentSourceError(f"{label} is absent")
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise DeploymentSourceError(f"{label} is redirected or irregular")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise DeploymentSourceError("platform lacks no-follow file operations")
    try:
        descriptor = os.open(
            name, os.O_RDONLY | nofollow, dir_fd=directory_descriptor,
        )
    except OSError as error:
        raise DeploymentSourceError(f"cannot safely open {label}") from error
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise DeploymentSourceError(f"{label} changed while opening")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            return stream.read()
    finally:
        os.close(descriptor)


def _check_destination_at(
    directory_descriptor: int, name: str, payload: bytes, label: str,
) -> bool:
    """Return whether an exact destination exists, rejecting every conflict."""

    existing = _read_regular_at(
        directory_descriptor, name, label, optional=True,
    )
    if existing is None:
        return False
    if existing != payload:
        raise DeploymentSourceError(f"existing {label} differs from frozen content")
    return True


def _publish_once_at(
    directory_descriptor: int, name: str, payload: bytes, label: str,
) -> bool:
    """Durably publish immutable bytes, or adopt an exact racing artifact."""

    if _check_destination_at(directory_descriptor, name, payload, label):
        return False
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise DeploymentSourceError("platform lacks no-follow file operations")
    temporary = f".{name}.{os.getpid()}.{secrets.token_hex(12)}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow
    descriptor = os.open(
        temporary, flags, 0o444, dir_fd=directory_descriptor,
    )
    try:
        output = os.fdopen(descriptor, "wb")
        descriptor = -1
        with output:
            output.write(payload)
            output.flush()
            os.fchmod(output.fileno(), 0o444)
            # Persist both content and the deliberate immutable publication mode.
            os.fsync(output.fileno())
        if _read_regular_at(
            directory_descriptor, temporary, f"temporary {label}"
        ) != payload:
            raise DeploymentSourceError(f"temporary {label} readback changed")
        try:
            os.link(
                temporary, name,
                src_dir_fd=directory_descriptor,
                dst_dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            created = True
        except FileExistsError:
            if not _check_destination_at(
                directory_descriptor, name, payload, label,
            ):
                raise DeploymentSourceError(f"{label} publication race was not durable")
            created = False
        os.fsync(directory_descriptor)
        if _read_regular_at(directory_descriptor, name, label) != payload:
            raise DeploymentSourceError(f"published {label} readback changed")
        return created
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=directory_descriptor)
        except FileNotFoundError:
            pass
        os.fsync(directory_descriptor)


def materialize_deployment(
    repository: pathlib.Path, generated_source_path: pathlib.Path, *,
    search_tuple: Sequence[Any], profile: Any,
) -> dict[str, Any]:
    """Materialize the canonical deployment pair at its fixed repository routes.

    Exact pre-existing artifacts are adopted.  Any different, redirected, or
    irregular artifact fails closed.  Consequently a crash after publishing
    only one member of the pair is recovered by the same invocation.
    """

    repository = pathlib.Path(repository)
    generated_source_path = pathlib.Path(generated_source_path).absolute()
    if not isinstance(profile, str) or profile not in PROFILE_ROSTER:
        raise DeploymentSourceError("deployment profile is outside the frozen roster")
    work = PROFILE_ROSTER[profile]
    generated_source = _read_regular_file(
        generated_source_path, "generated deployment source"
    )
    candidate_source = derive_source(
        generated_source, search_tuple=search_tuple, profile=profile, work=work
    )
    manifest = create_manifest(
        generated_source, candidate_source, search_tuple=search_tuple,
        profile=profile, work=work,
    )
    manifest_payload = _canonical_manifest_bytes(manifest)
    if json.loads(manifest_payload) != manifest:
        raise DeploymentSourceError("canonical deployment manifest changed")
    validate_manifest(manifest, candidate_source)

    candidate_relative = _safe_relative_output(
        CANDIDATE_RELATIVE, "candidate"
    )
    manifest_relative = _safe_relative_output(MANIFEST_RELATIVE, "manifest")
    repository, repository_descriptor = _open_repository(repository)
    candidate_path = repository / candidate_relative
    manifest_path = repository / manifest_relative
    try:
        candidate_parent = _open_directory_tree(
            repository_descriptor, candidate_relative.parent, create=True,
        )
        try:
            manifest_parent = _open_directory_tree(
                repository_descriptor, manifest_relative.parent, create=True,
            )
            try:
                # Inspect both members before creating either one so a known
                # conflict does not create a partial publication.
                candidate_existed = _check_destination_at(
                    candidate_parent, candidate_relative.name, candidate_source,
                    "deployment candidate",
                )
                manifest_existed = _check_destination_at(
                    manifest_parent, manifest_relative.name, manifest_payload,
                    "deployment manifest",
                )
                candidate_created = (
                    False if candidate_existed else _publish_once_at(
                        candidate_parent, candidate_relative.name, candidate_source,
                        "deployment candidate",
                    )
                )
                manifest_created = (
                    False if manifest_existed else _publish_once_at(
                        manifest_parent, manifest_relative.name, manifest_payload,
                        "deployment manifest",
                    )
                )

                _verify_directory_route(
                    repository_descriptor, candidate_relative.parent,
                    candidate_parent,
                )
                _verify_directory_route(
                    repository_descriptor, manifest_relative.parent,
                    manifest_parent,
                )
                # Both validators consume descriptor-anchored exact-byte
                # readbacks; no path traversal remains in the success decision.
                candidate_readback = _read_regular_at(
                    candidate_parent, candidate_relative.name,
                    "deployment candidate",
                )
                manifest_readback = _read_regular_at(
                    manifest_parent, manifest_relative.name,
                    "deployment manifest",
                )
                if manifest_readback != manifest_payload:
                    raise DeploymentSourceError(
                        "published deployment manifest is not canonical exact bytes"
                    )
                try:
                    readback_value = json.loads(manifest_readback)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise DeploymentSourceError(
                        "published deployment manifest JSON changed"
                    ) from error
                verified_manifest = validate_manifest(
                    readback_value, candidate_readback,
                )
                attest_derivation(
                    generated_source, candidate_readback,
                    search_tuple=search_tuple, profile=profile, work=work,
                )
                if verified_manifest != manifest:
                    raise DeploymentSourceError(
                        "published deployment manifest identity changed"
                    )
                _verify_directory_route(
                    repository_descriptor, candidate_relative.parent,
                    candidate_parent,
                )
                _verify_directory_route(
                    repository_descriptor, manifest_relative.parent,
                    manifest_parent,
                )
                _verify_repository_route(repository, repository_descriptor)
                return {
                    "status": "deployment-source-materialized",
                    "candidate": {
                        "path": str(candidate_path),
                        **_source_record(candidate_readback),
                        "created": candidate_created,
                    },
                    "manifest": {
                        "path": str(manifest_path),
                        "bytes": len(manifest_readback),
                        "sha256": _sha256(manifest_readback),
                        "body_sha256": verified_manifest["body_sha256"],
                        "created": manifest_created,
                    },
                    "base_source": dict(verified_manifest["base_source"]),
                    "configuration": dict(verified_manifest["configuration"]),
                }
            finally:
                os.close(manifest_parent)
        finally:
            os.close(candidate_parent)
    finally:
        os.close(repository_descriptor)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify-manifest")
    verify.add_argument("--manifest", type=pathlib.Path, required=True)
    verify.add_argument("--candidate", type=pathlib.Path, required=True)
    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--repository", type=pathlib.Path, default=REPOSITORY)
    materialize.add_argument("--generated-source", type=pathlib.Path, required=True)
    materialize.add_argument("--tuple", nargs=3, required=True)
    materialize.add_argument("--profile", choices=tuple(PROFILE_ROSTER), required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "verify-manifest":
            value = verify_manifest_file(args.manifest, args.candidate)
            result = {
                "status": "deployment-manifest-valid",
                "body_sha256": value["body_sha256"],
                "candidate_sha256": value["deployed_source"]["sha256"],
            }
        else:
            result = materialize_deployment(
                args.repository, args.generated_source,
                search_tuple=args.tuple, profile=args.profile,
            )
        print(json.dumps(result, sort_keys=True))
        return 0
    except (DeploymentSourceError, OSError) as error:
        print(f"deployment source failure: {error}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
