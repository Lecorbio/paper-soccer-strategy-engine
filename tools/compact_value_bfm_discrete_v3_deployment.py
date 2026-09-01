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

import hashlib
import json
import pathlib
import argparse
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
        isinstance(search_tuple, (str, bytes))
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify-manifest")
    verify.add_argument("--manifest", type=pathlib.Path, required=True)
    verify.add_argument("--candidate", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    try:
        value = verify_manifest_file(args.manifest, args.candidate)
        print(json.dumps({
            "status": "deployment-manifest-valid",
            "body_sha256": value["body_sha256"],
            "candidate_sha256": value["deployed_source"]["sha256"],
        }, sort_keys=True))
        return 0
    except (DeploymentSourceError, OSError) as error:
        print(f"deployment source failure: {error}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
