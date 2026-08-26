#!/usr/bin/env python3
"""Evidence-gated replay-BFM rebuild ladder.

The rebuild reuses frozen canonical and self-search artifacts.  It never
generates teacher labels, opens a protected final bank during model selection,
or changes the established promotion thresholds.  Expensive training and game
work is delegated to the rebuild-only recovery trainer and the existing
comparison executable; this module owns the immutable experiment matrix,
budget, deterministic screening, and final qualification lineage.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import math
import os
import pathlib
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence


REBUILD_ID = "replay-rebuild-20260826-v1"
REBUILD_INPUT_SCHEMA = "papersoccer.jacek-replay-rebuild-inputs.v1"
REBUILD_MATRIX_SCHEMA = "papersoccer.jacek-replay-rebuild-matrix.v1"
REBUILD_STATUS_SCHEMA = "papersoccer.jacek-replay-rebuild-status.v1"
REBUILD_CANDIDATE_SCHEMA = "papersoccer.jacek-replay-rebuild-candidate.v1"
REBUILD_CANDIDATE_SPEC_SCHEMA = (
    "papersoccer.jacek-replay-rebuild-candidate-spec.v1"
)
REBUILD_DECISION_SCHEMA = "papersoccer.jacek-replay-rebuild-decision.v1"
REBUILD_BANKS_SCHEMA = "papersoccer.jacek-replay-rebuild-banks.v1"
REBUILD_BUILD_SCHEMA = "papersoccer.jacek-replay-rebuild-build.v1"
REBUILD_QUALIFICATION_SCHEMA = (
    "papersoccer.jacek-replay-rebuild-qualification.v1"
)

SAME_ARCHITECTURE_BUDGET_SECONDS = 24 * 60 * 60
ORDER_SEEDS = (20261011, 20261012, 20261013)
SCRATCH_SEEDS = tuple(range(20261001, 20261007))
RECOVERY_LEARNING_RATES = (3e-6, 1e-5, 3e-5)
RECOVERY_LAYER_SCOPES = ("w3", "w2-w3", "all")
JOINT_LEARNING_RATE = 6e-5
NEW_ROWS_PER_BATCH = 64
ANCHOR_ROWS_PER_BATCH = 192
NEW_LOSS_COEFFICIENT = 0.25
ANCHOR_LOSS_COEFFICIENT = 0.75
CHECKPOINT_UPDATES = 782
MAX_ANCHOR_PASSES = 2

SHORT_SCREEN_PAIRS = 100
FULL_SCREEN_PAIRS = 300
SHORTLIST_LIMIT = 6
FULL_SCREEN_LIMIT = 2

PRIMARY_WIN_THRESHOLD = 325
PRIMARY_COLOR_THRESHOLD = 156
EXTERNAL_WIN_THRESHOLD = 306
EXTERNAL_COLOR_THRESHOLD = 143
SIGN_MARGIN = 0.005
HUBER_MULTIPLIER = 1.02
P99_LIMIT_MS = 25.0
UNCONTENDED_LIMIT_MS = 1_000.0
DEVELOPMENT_BANK_SEED = 2026082701
FINAL_BANK_SEED = 2026082705
BLIND_HOLDOUT_SEED = 2026082703
HOLDOUT_GAME_SEED = 2026082704
HOLDOUT_SOURCE_GAMES = 40_000
HOLDOUT_CANDIDATE_GROUPS = 35_000
HOLDOUT_GENERATOR_WORKERS = 10
HOLDOUT_GENERATOR_BASE_BUDGET = 1_000


def canonical_json_bytes(value: object, *, pretty: bool = False) -> bytes:
    options: dict[str, object] = {
        "sort_keys": True,
        "separators": (",", ":"),
        "ensure_ascii": False,
    }
    if pretty:
        options.update(indent=2, separators=None)
    return (json.dumps(value, **options) + "\n").encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_snapshot(path: pathlib.Path) -> dict[str, object]:
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"rebuild artifact is missing: {path}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def load_json(path: pathlib.Path, label: str) -> dict:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def verify_body_hash(
    record: Mapping[str, object], *, schema: str, label: str
) -> None:
    body = dict(record)
    claimed = body.pop("body_sha256", None)
    if (
        body.get("schema") != schema
        or claimed != sha256_bytes(canonical_json_bytes(body))
    ):
        raise ValueError(f"{label} integrity failed")


def atomic_write(path: pathlib.Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: pathlib.Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
            temporary = pathlib.Path(output.name)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def freeze_opening_banks(
    *, comparison: pathlib.Path, output_directory: pathlib.Path,
    excluded_banks: Sequence[pathlib.Path],
) -> dict[str, object]:
    """Freeze disjoint development/final banks before candidate selection."""

    import jacek_selfsearch_workflow as selfsearch

    comparison = comparison.resolve()
    exclusions = tuple(path.resolve() for path in excluded_banks)
    if not comparison.is_file() or not exclusions:
        raise ValueError("rebuild bank inputs are incomplete")
    output_directory = output_directory.resolve()
    development = output_directory / "development-openings.tsv"
    final = output_directory / "sealed-final-openings.tsv"
    manifest_path = output_directory / "opening-banks.json"
    development_record = selfsearch.generate_comparison_bank(
        comparison=comparison,
        output=development,
        pairs=FULL_SCREEN_PAIRS,
        seed=DEVELOPMENT_BANK_SEED,
        exclusions=exclusions,
        classification="development",
    )
    final_record = selfsearch.generate_comparison_bank(
        comparison=comparison,
        output=final,
        pairs=FULL_SCREEN_PAIRS,
        seed=FINAL_BANK_SEED,
        exclusions=(*exclusions, development),
        classification="final",
    )
    body: dict[str, object] = {
        "schema": REBUILD_BANKS_SCHEMA,
        "rebuild_id": REBUILD_ID,
        "comparison": artifact_snapshot(comparison),
        "excluded_banks": [artifact_snapshot(path) for path in exclusions],
        "development": {
            "artifact": artifact_snapshot(development),
            "configuration": development_record,
            "model_selection_eligible": True,
        },
        "final": {
            "artifact": artifact_snapshot(final),
            "configuration": final_record,
            "model_selection_eligible": False,
            "sealed_until_selected_runtime_receipt": True,
        },
    }
    record = {**body, "body_sha256": sha256_bytes(canonical_json_bytes(body))}
    if manifest_path.exists():
        if load_json(manifest_path, "rebuild opening banks") != record:
            raise ValueError("existing rebuild opening banks are stale")
    else:
        atomic_write(manifest_path, canonical_json_bytes(record, pretty=True))
    return record


def freeze_banks_from_campaigns(
    *, comparison: pathlib.Path, output_directory: pathlib.Path,
    evaluation_directory: pathlib.Path, v5_campaign: pathlib.Path,
    v6_campaign: pathlib.Path,
) -> dict[str, object]:
    import jacek_selfsearch_workflow as selfsearch

    exclusions = list(
        selfsearch._evaluation_opening_banks(evaluation_directory.resolve())
    )
    exclusions.extend(
        (
            (v5_campaign / "pilot/gate-openings.tsv").resolve(),
            (v6_campaign / "pilot/gate-openings.tsv").resolve(),
        )
    )
    unique = []
    seen = set()
    for path in exclusions:
        if not path.is_file():
            raise ValueError(f"protected opening bank is missing: {path}")
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return freeze_opening_banks(
        comparison=comparison,
        output_directory=output_directory,
        excluded_banks=unique,
    )


def validate_opening_banks(record: Mapping[str, object]) -> None:
    import jacek_selfsearch_workflow as selfsearch

    body = dict(record)
    claimed = body.pop("body_sha256", None)
    if (
        body.get("schema") != REBUILD_BANKS_SCHEMA
        or body.get("rebuild_id") != REBUILD_ID
        or claimed != sha256_bytes(canonical_json_bytes(body))
    ):
        raise ValueError("rebuild opening-bank manifest is stale or corrupt")
    development_value = body.get("development")
    final_value = body.get("final")
    if (
        not isinstance(development_value, dict)
        or development_value.get("model_selection_eligible") is not True
        or not isinstance(final_value, dict)
        or final_value.get("model_selection_eligible") is not False
        or final_value.get("sealed_until_selected_runtime_receipt") is not True
    ):
        raise ValueError("sealed final bank is exposed to model selection")
    for label in ("comparison",):
        record_value = body.get(label)
        if (
            not isinstance(record_value, dict)
            or artifact_snapshot(pathlib.Path(str(record_value.get("path", ""))))
            != record_value
        ):
            raise ValueError(f"rebuild bank {label} binding is stale")
    for label in ("development", "final"):
        value = body.get(label)
        artifact = value.get("artifact") if isinstance(value, dict) else None
        if (
            not isinstance(artifact, dict)
            or artifact_snapshot(pathlib.Path(str(artifact.get("path", ""))))
            != artifact
        ):
            raise ValueError(f"rebuild {label} bank binding is stale")
    exclusions = body.get("excluded_banks")
    if not isinstance(exclusions, list) or not exclusions:
        raise ValueError("rebuild bank exclusions are incomplete")
    for exclusion in exclusions:
        if (
            not isinstance(exclusion, dict)
            or artifact_snapshot(pathlib.Path(str(exclusion.get("path", ""))))
            != exclusion
        ):
            raise ValueError("rebuild bank exclusion binding is stale")
    development_path = pathlib.Path(development_value["artifact"]["path"])
    final_path = pathlib.Path(final_value["artifact"]["path"])
    development_states = selfsearch._comparison_bank_states(
        development_path, "development"
    )
    final_states = selfsearch._comparison_bank_states(final_path, "final")
    excluded_states = set()
    for exclusion in exclusions:
        excluded_states.update(
            selfsearch._comparison_bank_states(pathlib.Path(exclusion["path"]))
        )
    detailed_exclusions = [
        selfsearch.artifact_snapshot(pathlib.Path(exclusion["path"]))
        for exclusion in exclusions
    ]
    if (
        len(development_states) != FULL_SCREEN_PAIRS
        or len(final_states) != FULL_SCREEN_PAIRS
        or development_states & final_states
        or development_states & excluded_states
        or final_states & excluded_states
    ):
        raise ValueError("rebuild opening banks overlap or have wrong counts")

    def expected_configuration(
        *, classification: str, seed: int,
        states: set[str], configuration_exclusions: Sequence[dict],
    ) -> dict[str, object]:
        return {
            "pairs": FULL_SCREEN_PAIRS,
            "seed": seed,
            "opening_plies": 12,
            "classification": classification,
            "states_sha256": hashlib.sha256(
                "\n".join(sorted(states)).encode()
            ).hexdigest(),
            "exclusions": list(configuration_exclusions),
        }

    if development_value.get("configuration") != expected_configuration(
        classification="development",
        seed=DEVELOPMENT_BANK_SEED,
        states=development_states,
        configuration_exclusions=detailed_exclusions,
    ) or final_value.get("configuration") != expected_configuration(
        classification="final",
        seed=FINAL_BANK_SEED,
        states=final_states,
        configuration_exclusions=(
            *detailed_exclusions,
            selfsearch.artifact_snapshot(development_path),
        ),
    ):
        raise ValueError("rebuild opening-bank configuration changed")


def _sealed_feature_fingerprints(
    manifest_paths: Sequence[pathlib.Path],
) -> set[bytes]:
    """Read only sparse feature identities; protected targets stay unopened."""

    import numpy as np
    import jacek_replay_corpus as replay_corpus
    import jacek_replay_features as replay_features

    fingerprints: set[bytes] = set()
    for raw_path in manifest_paths:
        path = raw_path.resolve()
        manifest = load_json(path, "sealed feature-identity shard")
        npz_name = manifest.get("npz")
        npz_sha256 = manifest.get("npz_sha256")
        if (
            path.suffix != ".json"
            or sha256_file(path) != path.stem
            or manifest.get("schema")
            != "papersoccer.jacek-replay-csr-shard.v1"
            or manifest.get("feature_schema") != replay_features.FEATURE_SCHEMA
            or not isinstance(npz_name, str)
            or pathlib.PurePath(npz_name).name != npz_name
            or not isinstance(npz_sha256, str)
        ):
            raise ValueError("sealed feature-identity manifest changed")
        npz_path = path.parent / npz_name
        if sha256_file(npz_path) != npz_sha256 or npz_path.stem != npz_sha256:
            raise ValueError("sealed feature-identity NPZ changed")
        with np.load(npz_path, allow_pickle=False) as archive:
            if not {"indptr", "indices"}.issubset(archive.files):
                raise ValueError("sealed sparse feature identity is incomplete")
            indptr = np.asarray(archive["indptr"])
            indices = np.asarray(archive["indices"])
        rows = int(manifest.get("samples", -1))
        if (
            indptr.dtype != np.dtype("<i8")
            or indices.dtype != np.dtype("<u2")
            or indptr.shape != (rows + 1,)
            or int(indptr[0]) != 0
            or int(indptr[-1]) != len(indices)
            or len(indices) != int(manifest.get("active_features", -1))
        ):
            raise ValueError("sealed sparse feature-identity contract changed")
        for row in range(rows):
            start, end = int(indptr[row]), int(indptr[row + 1])
            fingerprints.add(
                replay_corpus.canonical_feature_fingerprint(
                    indices[start:end].tolist()
                )
            )
    return fingerprints


def load_frozen_rebuild_corpus(
    manifest_path: pathlib.Path,
    *, expected_canonical_counts: Mapping[str, int] | None = None,
) -> object:
    """Load a once-deep-validated corpus through immutable hash contracts."""

    import jacek_rebuild_corpus as rebuild_corpus
    import jacek_replay_features as replay_features

    path = manifest_path.resolve()
    manifest, raw = rebuild_corpus._read_canonical_json(
        path, "frozen rebuild corpus manifest"
    )
    expected_counts = rebuild_corpus._normalized_expected_counts(
        expected_canonical_counts
    )
    if (
        manifest.get("schema") != rebuild_corpus.MANIFEST_SCHEMA
        or manifest.get("feature_schema") != replay_features.FEATURE_SCHEMA
        or path.suffix != ".json"
        or path.stem != sha256_bytes(raw)
        or manifest.get("canonical_count_contract") != expected_counts
        or manifest.get("producer") != rebuild_corpus._producer_identity()
    ):
        raise ValueError("frozen rebuild corpus identity changed")
    rebuild_corpus._verify_body_sha256(manifest)
    base = path.parent
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != set(
        rebuild_corpus._INPUT_SPECS
    ):
        raise ValueError("frozen rebuild corpus input roles changed")
    for key, (campaign, channel, role) in rebuild_corpus._INPUT_SPECS.items():
        identities = inputs.get(key)
        if not isinstance(identities, list) or not identities:
            raise ValueError("frozen rebuild corpus input role is empty")
        for identity in identities:
            if not isinstance(identity, dict):
                raise ValueError("frozen rebuild corpus input identity is malformed")
            observed = rebuild_corpus._sealed_source_identity(
                rebuild_corpus._resolve_bound_path(
                    base, identity.get("manifest_path"), "frozen source manifest"
                ),
                base,
                campaign=campaign,
                channel=channel,
                role=role,
            )
            if observed != identity:
                raise ValueError("frozen rebuild corpus source identity changed")
    for role in ("train", "validation", "test"):
        rows = sum(
            int(identity["rows"])
            for identity in inputs[f"canonical_{role}"]
        )
        if rows != expected_counts[role]:
            raise ValueError("frozen canonical corpus count changed")

    deduplicated = manifest.get("deduplicated")
    if not isinstance(deduplicated, dict) or set(deduplicated) != {
        "search", "rank4", "adjudicator"
    }:
        raise ValueError("frozen deduplicated corpus roles changed")
    generated: dict[str, dict[str, object]] = {}
    for channel in ("search", "rank4", "adjudicator"):
        record = deduplicated[channel]
        if not isinstance(record, dict) or set(record) != {
            "selection", "selection_sha256", "shard"
        }:
            raise ValueError("frozen deduplicated corpus record shape changed")
        selection = record["selection"]
        selection_sha256 = record["selection_sha256"]
        shard_identity = record["shard"]
        expected_role = "validation" if channel == "adjudicator" else "train"
        input_keys = [f"v6_{channel}_validation", f"v5_{channel}_validation"] if (
            channel == "adjudicator"
        ) else [f"v6_{channel}_train", f"v5_{channel}_train"]
        if (
            not isinstance(selection, dict)
            or not isinstance(shard_identity, dict)
            or selection_sha256
            != sha256_bytes(canonical_json_bytes(selection))
            or selection.get("schema") != rebuild_corpus.DEDUPLICATION_SCHEMA
            or selection.get("channel") != channel
            or selection.get("role") != expected_role
            or selection.get("campaign_precedence") != ["v6", "v5"]
            or selection.get("equivalence")
            != "canonical-rotate-reflection-feature-fingerprint"
            or selection.get("tie_break")
            != "manifest-sha256-then-manifest-path-then-row-index"
            or selection.get("input_rows")
            != sum(
                int(identity["rows"])
                for input_key in input_keys for identity in inputs[input_key]
            )
            or selection.get("discarded_rows")
            != selection.get("input_rows") - selection.get("retained_rows")
            or shard_identity.get("rows") != selection.get("retained_rows")
        ):
            raise ValueError("frozen deduplication selection semantics changed")
        observed_shard = rebuild_corpus._sealed_source_identity(
            rebuild_corpus._resolve_bound_path(
                base,
                shard_identity.get("manifest_path"),
                "frozen generated manifest",
            ),
            base,
            campaign="rebuild",
            channel=channel,
            role=expected_role,
        )
        if observed_shard != shard_identity:
            raise ValueError("frozen generated shard identity changed")
        generated_manifest = load_json(
            rebuild_corpus._resolve_bound_path(
                base, shard_identity["manifest_path"], "generated manifest"
            ),
            "frozen generated shard manifest",
        )
        expected_provenance = {
            "schema": rebuild_corpus.DEDUPLICATION_SCHEMA,
            "channel": channel,
            "role": expected_role,
            "campaign_precedence": ["v6", "v5"],
            "selection_sha256": selection_sha256,
            "canonical_fingerprints_sha256": selection.get(
                "canonical_fingerprints_sha256"
            ),
            "producer": manifest["producer"],
        }
        if generated_manifest.get("provenance") != expected_provenance:
            raise ValueError("frozen generated shard provenance changed")
        generated[channel] = dict(shard_identity)
    if manifest.get("interfaces") != rebuild_corpus._interfaces(inputs, generated):
        raise ValueError("frozen rebuild training interfaces changed")
    protected = manifest.get("protected_test")
    if protected != {
        "manifest_paths": rebuild_corpus._manifest_paths(
            inputs["canonical_test"]
        ),
        "rows": expected_counts["test"],
        "selection_eligible": False,
        "training_eligible": False,
    }:
        raise ValueError("frozen protected-test interface changed")
    leakage = manifest.get("leakage")
    roles = leakage.get("roles") if isinstance(leakage, dict) else None
    expected_train_rows = expected_counts["train"] + int(
        generated["search"]["rows"]
    ) + int(generated["rank4"]["rows"])
    expected_validation_rows = expected_counts["validation"] + int(
        generated["adjudicator"]["rows"]
    )
    if (
        not isinstance(roles, dict)
        or leakage.get("policy")
        != "train-validation-fingerprint-and-root-isolation;"
        "canonical-test-sealed-by-upstream-workflow-provenance"
        or leakage.get("verified") is not True
        or roles.get("test")
        != {
            "rows_bound": expected_counts["test"],
            "sealed": True,
            "isolation_evidence": "canonical-workflow-provenance",
            "arrays_decoded": False,
        }
        or roles.get("train", {}).get("rows_scanned") != expected_train_rows
        or roles.get("validation", {}).get("rows_scanned")
        != expected_validation_rows
        or any(
            not isinstance(roles.get(role, {}).get(field), int)
            or roles[role][field] <= 0
            for role in ("train", "validation")
            for field in (
                "unique_canonical_fingerprints", "unique_root_groups"
            )
        )
    ):
        raise ValueError("frozen rebuild leakage proof changed")
    return rebuild_corpus.RebuildCorpus(path, manifest)


def _freeze_holdout_exclusion_cache(
    *, shard_manifests: Sequence[pathlib.Path],
    position_tsvs: Sequence[pathlib.Path],
    root_manifests: Sequence[pathlib.Path],
    output_directory: pathlib.Path,
) -> tuple[set[str], set[bytes]]:
    import jacek_replay_retention as retention

    root = output_directory.resolve() / "exclusion-cache"
    fingerprint_path = root / "canonical-fingerprints.bin"
    groups_path = root / "root-groups.txt"
    receipt_path = root / "receipt.json"
    shard_paths = tuple(path.resolve() for path in shard_manifests)
    position_paths = tuple(path.resolve() for path in position_tsvs)
    root_paths = tuple(path.resolve() for path in root_manifests)
    common: dict[str, object] = {
        "schema": "papersoccer.jacek-rebuild-holdout-exclusion-cache.v1",
        "rebuild_id": REBUILD_ID,
        "inputs": {
            "excluded_shards": [
                retention.artifact_snapshot(path) for path in shard_paths
            ],
            "excluded_positions": [
                retention.artifact_snapshot(path) for path in position_paths
            ],
            "excluded_roots": [
                retention.artifact_snapshot(path) for path in root_paths
            ],
        },
        "producer": retention._producer_identity(),
    }

    def load_cached() -> tuple[set[str], set[bytes]]:
        receipt = load_json(receipt_path, "holdout exclusion cache receipt")
        verify_body_hash(
            receipt,
            schema="papersoccer.jacek-rebuild-holdout-exclusion-cache.v1",
            label="holdout exclusion cache receipt",
        )
        fingerprint_payload = fingerprint_path.read_bytes()
        group_payload = groups_path.read_bytes()
        if len(fingerprint_payload) % 32:
            raise ValueError("holdout exclusion fingerprint cache is malformed")
        fingerprints = {
            fingerprint_payload[offset : offset + 32]
            for offset in range(0, len(fingerprint_payload), 32)
        }
        try:
            group_lines = group_payload.decode("utf-8").splitlines()
        except UnicodeDecodeError as error:
            raise ValueError("holdout exclusion root cache is malformed") from error
        groups = set(group_lines)
        expected = {
            **common,
            "exclusion_universe": {
                "root_groups": len(groups),
                "canonical_fingerprints": len(fingerprints),
                "root_group_ids_sha256": hashlib.sha256(
                    "\n".join(sorted(groups)).encode()
                ).hexdigest(),
                "canonical_fingerprints_sha256": hashlib.sha256(
                    b"".join(sorted(fingerprints))
                ).hexdigest(),
            },
            "artifacts": {
                "root_groups": retention.artifact_snapshot(groups_path),
                "canonical_fingerprints": retention.artifact_snapshot(
                    fingerprint_path
                ),
            },
        }
        expected = {
            **expected,
            "body_sha256": sha256_bytes(canonical_json_bytes(expected)),
        }
        if (
            receipt != expected
            or not groups
            or not fingerprints
            or len(group_lines) != len(groups)
            or group_payload
            != ("\n".join(sorted(groups)) + "\n").encode()
            or fingerprint_payload != b"".join(sorted(fingerprints))
        ):
            raise ValueError("holdout exclusion cache changed")
        return groups, fingerprints

    if receipt_path.exists() or fingerprint_path.exists() or groups_path.exists():
        if not all(
            path.is_file()
            for path in (receipt_path, fingerprint_path, groups_path)
        ):
            raise ValueError("holdout exclusion cache is partial")
        return load_cached()

    groups, fingerprints = retention._exclusion_sets(
        shard_manifests=shard_paths,
        position_tsvs=position_paths,
        root_manifests=root_paths,
    )
    if not groups or not fingerprints:
        raise ValueError("holdout exclusion universe is empty")
    group_payload = ("\n".join(sorted(groups)) + "\n").encode()
    fingerprint_payload = b"".join(sorted(fingerprints))
    atomic_write(groups_path, group_payload)
    atomic_write(fingerprint_path, fingerprint_payload)
    body: dict[str, object] = {
        **common,
        "exclusion_universe": {
            "root_groups": len(groups),
            "canonical_fingerprints": len(fingerprints),
            "root_group_ids_sha256": hashlib.sha256(
                "\n".join(sorted(groups)).encode()
            ).hexdigest(),
            "canonical_fingerprints_sha256": hashlib.sha256(
                fingerprint_payload
            ).hexdigest(),
        },
        "artifacts": {
            "root_groups": retention.artifact_snapshot(groups_path),
            "canonical_fingerprints": retention.artifact_snapshot(
                fingerprint_path
            ),
        },
    }
    receipt = {**body, "body_sha256": sha256_bytes(canonical_json_bytes(body))}
    atomic_write(receipt_path, canonical_json_bytes(receipt, pretty=True))
    return load_cached()


def freeze_blind_holdout(
    *, candidate_positions: pathlib.Path, training_input_receipt: pathlib.Path,
    excluded_shards: Sequence[pathlib.Path],
    excluded_sealed_shards: Sequence[pathlib.Path] = (),
    excluded_positions: Sequence[pathlib.Path],
    excluded_roots: Sequence[pathlib.Path], output_directory: pathlib.Path,
) -> dict[str, object]:
    """Freeze 600 whole procedural groups before any model is selected."""

    import jacek_replay_retention as retention

    output_directory = output_directory.resolve()
    output_positions = output_directory / "blind-retention-positions.tsv"
    output_manifest = output_directory / "blind-retention-freeze.json"
    spec = retention.FreezeSpec(
        profile="rebuild",
        groups=600,
        selection_seed=BLIND_HOLDOUT_SEED,
        source_quotas=(),
    )
    resolved_shards = tuple(path.resolve() for path in excluded_shards)
    resolved_positions = tuple(path.resolve() for path in excluded_positions)
    resolved_roots = tuple(path.resolve() for path in excluded_roots)
    excluded_groups, excluded_fingerprints = _freeze_holdout_exclusion_cache(
        shard_manifests=resolved_shards,
        position_tsvs=resolved_positions,
        root_manifests=resolved_roots,
        output_directory=output_directory,
    )
    payload, manifest = retention.freeze_candidate_groups(
        candidate_positions=candidate_positions.resolve(),
        training_input_receipt=training_input_receipt.resolve(),
        campaign_id=REBUILD_ID,
        spec=spec,
        excluded_shard_manifests=resolved_shards,
        excluded_position_tsvs=resolved_positions,
        excluded_root_manifests=resolved_roots,
        precomputed_excluded_groups=excluded_groups,
        precomputed_excluded_fingerprints=excluded_fingerprints,
    )
    sealed_paths = tuple(path.resolve() for path in excluded_sealed_shards)
    if sealed_paths:
        sealed_fingerprints = _sealed_feature_fingerprints(sealed_paths)
        selected_roots = {
            str(record["root_group_id"])
            for record in manifest["selection"]["groups"]
        }
        candidate_rows = retention.load_position_rows(
            candidate_positions.resolve(), required_split=retention.TEACHER_SPLIT
        )
        selected_fingerprints = {
            row.canonical_fingerprint
            for row in candidate_rows
            if row.root_group_id in selected_roots
        }
        overlap = selected_fingerprints & sealed_fingerprints
        if overlap or len(selected_fingerprints) != 12_000:
            raise ValueError(
                "blind holdout overlaps sealed canonical-test feature identities"
            )
        body = dict(manifest)
        body.pop("body_sha256", None)
        body["inputs"] = {
            **body["inputs"],
            "excluded_sealed_identity_shards": [
                retention.artifact_snapshot(path) for path in sealed_paths
            ],
        }
        body["sealed_identity_exclusion"] = {
            "policy": "feature-indices-only-targets-and-weights-unopened",
            "canonical_fingerprints": len(sealed_fingerprints),
            "canonical_fingerprints_sha256": hashlib.sha256(
                b"".join(sorted(sealed_fingerprints))
            ).hexdigest(),
            "selected_overlap": 0,
        }
        manifest = {
            **body,
            "body_sha256": sha256_bytes(canonical_json_bytes(body)),
        }
    if output_positions.exists() or output_manifest.exists():
        if (
            not output_positions.is_file()
            or not output_manifest.is_file()
            or output_positions.read_bytes() != payload
            or load_json(output_manifest, "blind retention freeze") != manifest
        ):
            raise ValueError("existing blind retention freeze is stale")
    else:
        retention.write_freeze(
            output_positions, output_manifest, payload, manifest
        )
    return {
        "positions": artifact_snapshot(output_positions),
        "manifest": artifact_snapshot(output_manifest),
        "groups": 600,
        "rows": 12_000,
        "labels_opened": False,
        "selected_model_opened": False,
    }


def load_frozen_blind_holdout(
    positions_path: pathlib.Path, manifest_path: pathlib.Path,
) -> tuple[dict, list[object]]:
    """Replay frozen selection identities without rescanning million-row anchors."""

    from collections import Counter, defaultdict
    import jacek_replay_retention as retention
    import jacek_replay_features as replay_features

    positions_path = positions_path.resolve()
    manifest = load_json(manifest_path.resolve(), "frozen rebuild holdout")
    body = dict(manifest)
    claimed = body.pop("body_sha256", None)
    configuration = manifest.get("configuration")
    timing = manifest.get("timing")
    inputs = manifest.get("inputs")
    if (
        claimed != sha256_bytes(retention.canonical_json_bytes(body))
        or manifest.get("schema") != retention.FREEZE_SCHEMA
        or manifest.get("campaign_id") != REBUILD_ID
        or manifest.get("profile") != "rebuild"
        or manifest.get("feature_schema") != replay_features.FEATURE_SCHEMA
        or manifest.get("role") != "retention-rebuild"
        or manifest.get("teacher_split") != retention.TEACHER_SPLIT
        or manifest.get("training_eligible") is not False
        or not isinstance(configuration, dict)
        or configuration
        != {
            "groups": 600,
            "rows_per_group": retention.ROWS_PER_GROUP,
            "selection_seed": BLIND_HOLDOUT_SEED,
            "selection": "sha256-seeded-whole-root-groups-v1",
            "source_quotas": {},
            "overlap_identity": (
                "rotate-and-reflect-canonical-feature-fingerprint"
            ),
            "partial_group_policy": (
                "reject-whole-candidate-group-before-freeze"
            ),
            "post_freeze_drop_policy": "forbidden",
        }
        or not isinstance(timing, dict)
        or timing.get("teacher_labels_opened") is not False
        or timing.get("selected_model_opened") is not False
        or timing.get("required_reveal_order")
        != "freeze-before-model-selection;labels-after-model-selection;"
        "metrics-after-selected-runtime-binding"
        or not isinstance(inputs, dict)
    ):
        raise ValueError("frozen rebuild holdout contract changed")
    retention._validate_snapshot(
        timing.get("training_inputs_frozen_by"),
        "rebuild holdout training receipt",
    )
    retention._validate_snapshot(
        inputs.get("candidate_positions"), "rebuild holdout candidate positions"
    )
    for key in ("excluded_shards", "excluded_positions", "excluded_roots"):
        retention._validate_snapshot_list(
            inputs.get(key), f"rebuild holdout {key}"
        )
    sealed_records = inputs.get("excluded_sealed_identity_shards")
    if not isinstance(sealed_records, list) or not sealed_records:
        raise ValueError("rebuild holdout sealed exclusions are missing")
    for record in sealed_records:
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("path"), str)
            or retention.artifact_snapshot(pathlib.Path(record["path"]))
            != record
        ):
            raise ValueError("rebuild holdout sealed exclusion binding changed")
    retention._validate_producer_identity(manifest.get("producer"))
    exclusion = manifest.get("exclusion_universe")
    if (
        not isinstance(exclusion, dict)
        or set(exclusion) != {
            "root_groups", "canonical_fingerprints",
            "root_group_ids_sha256", "canonical_fingerprints_sha256",
        }
        or any(
            isinstance(exclusion.get(key), bool)
            or not isinstance(exclusion.get(key), int)
            or exclusion[key] <= 0
            for key in ("root_groups", "canonical_fingerprints")
        )
        or any(
            not isinstance(exclusion.get(key), str)
            or len(exclusion[key]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in exclusion[key]
            )
            for key in (
                "root_group_ids_sha256", "canonical_fingerprints_sha256"
            )
        )
    ):
        raise ValueError("rebuild holdout deep exclusion receipt changed")
    output = manifest.get("output")
    payload = positions_path.read_bytes()
    if (
        not isinstance(output, dict)
        or output
        != {
            "sha256": sha256_bytes(payload),
            "bytes": len(payload),
            "lines": len(payload.splitlines()),
        }
    ):
        raise ValueError("rebuild holdout output binding changed")
    rows = retention.load_position_rows(
        positions_path, required_split=retention.TEACHER_SPLIT
    )
    selection = manifest.get("selection")
    groups = selection.get("groups") if isinstance(selection, dict) else None
    if (
        not isinstance(selection, dict)
        or not isinstance(groups, list)
        or len(groups) != 600
        or len(rows) != 12_000
        or len({row.position_id for row in rows}) != len(rows)
        or len({row.root_group_id for row in rows}) != 600
        or any(
            count != retention.ROWS_PER_GROUP
            for count in Counter(row.root_group_id for row in rows).values()
        )
        or len({row.canonical_fingerprint for row in rows}) != len(rows)
        or selection.get("selected_root_groups") != 600
        or selection.get("selected_positions") != 12_000
        or selection.get("position_ids_sha256")
        != hashlib.sha256(
            "\n".join(row.position_id for row in rows).encode()
        ).hexdigest()
        or selection.get("canonical_fingerprints_sha256")
        != hashlib.sha256(
            b"".join(sorted(row.canonical_fingerprint for row in rows))
        ).hexdigest()
    ):
        raise ValueError("rebuild holdout selected coverage changed")
    rows_by_root: dict[str, list[object]] = defaultdict(list)
    for row in rows:
        rows_by_root[row.root_group_id].append(row)
    selected_root_ids = []
    for record in groups:
        bound = (
            rows_by_root.get(str(record.get("root_group_id")), [])
            if isinstance(record, dict) else []
        )
        if (
            not isinstance(record, dict)
            or record.get("rows") != retention.ROWS_PER_GROUP
            or not bound
            or record.get("generated_group_id") != bound[0].group_id
            or record.get("source") != bound[0].source
            or record.get("position_ids_sha256")
            != hashlib.sha256(
                "\n".join(row.position_id for row in bound).encode()
            ).hexdigest()
            or record.get("canonical_fingerprints_sha256")
            != hashlib.sha256(
                b"".join(sorted(row.canonical_fingerprint for row in bound))
            ).hexdigest()
        ):
            raise ValueError("rebuild holdout group receipt changed")
        selected_root_ids.append(record["root_group_id"])
    row_root_ids = list(dict.fromkeys(row.root_group_id for row in rows))
    if (
        selected_root_ids != row_root_ids
        or selection.get("root_group_ids_sha256")
        != hashlib.sha256("\n".join(row_root_ids).encode()).hexdigest()
        or selection.get("selected_source_counts")
        != dict(
            sorted(
                Counter(
                    group_rows[0].source
                    for group_rows in rows_by_root.values()
                ).items()
            )
        )
    ):
        raise ValueError("rebuild holdout root/source selection changed")
    sealed_fingerprints = _sealed_feature_fingerprints(
        tuple(pathlib.Path(record["path"]) for record in sealed_records)
    )
    sealed_exclusion = manifest.get("sealed_identity_exclusion")
    if (
        sealed_exclusion
        != {
            "policy": "feature-indices-only-targets-and-weights-unopened",
            "canonical_fingerprints": len(sealed_fingerprints),
            "canonical_fingerprints_sha256": hashlib.sha256(
                b"".join(sorted(sealed_fingerprints))
            ).hexdigest(),
            "selected_overlap": 0,
        }
        or any(
            row.canonical_fingerprint in sealed_fingerprints for row in rows
        )
    ):
        raise ValueError("rebuild holdout sealed exclusion changed")
    return manifest, rows


def freeze_holdout_from_corpus(
    *, candidate_positions: pathlib.Path, candidate_manifest: pathlib.Path,
    corpus_manifest: pathlib.Path,
    canonical_campaign: pathlib.Path, v5_campaign: pathlib.Path,
    v6_campaign: pathlib.Path, output_directory: pathlib.Path,
) -> dict[str, object]:
    import jacek_rebuild_corpus as rebuild_corpus

    corpus = load_frozen_rebuild_corpus(corpus_manifest)
    validate_holdout_candidate_pool(candidate_positions, candidate_manifest)
    shards: set[pathlib.Path] = set()
    for arm in ("search", "rank4"):
        shards.update(corpus.training_manifest_paths(arm))
        shards.update(corpus.anchor_manifest_paths(arm))
        shards.update(corpus.validation_manifest_paths(arm))
        shards.update(corpus.retention_validation_manifest_paths(arm))
    return freeze_blind_holdout(
        candidate_positions=candidate_positions,
        training_input_receipt=corpus_manifest,
        excluded_shards=tuple(sorted(shards)),
        excluded_sealed_shards=tuple(
            sorted(corpus.protected_test_manifest_paths)
        ),
        excluded_positions=(
            (v5_campaign / "pilot/positions.tsv").resolve(),
            (v6_campaign / "pilot/positions.tsv").resolve(),
        ),
        excluded_roots=(
            *(
                (canonical_campaign / f"round-{round_index}/replay-roots.json").resolve()
                for round_index in range(3)
            ),
        ),
        output_directory=output_directory,
    )


def generate_holdout_candidate_positions(
    *, generator: pathlib.Path, output_directory: pathlib.Path,
    workers: int = HOLDOUT_GENERATOR_WORKERS,
) -> dict[str, object]:
    """Generate independent whole-game position groups for blind freezing."""

    import jacek_replay_retention as retention

    generator = generator.resolve()
    output_directory = output_directory.resolve()
    if not generator.is_file() or workers <= 0 or HOLDOUT_SOURCE_GAMES % workers:
        raise ValueError("holdout candidate generator configuration is invalid")
    raw_directory = output_directory / "procedural-games"
    raw_directory.mkdir(parents=True, exist_ok=True)
    plan_body: dict[str, object] = {
        "schema": "papersoccer.jacek-rebuild-holdout-game-plan.v1",
        "rebuild_id": REBUILD_ID,
        "generator": artifact_snapshot(generator),
        "games": HOLDOUT_SOURCE_GAMES,
        "base_budget": HOLDOUT_GENERATOR_BASE_BUDGET,
        "maximum_turns": 200,
        "root_seed": HOLDOUT_GAME_SEED,
        "rows": [
            {
                "game_ordinal": ordinal,
                "base_seed": (
                    HOLDOUT_GAME_SEED
                    + ordinal * 32 * 0x9E3779B97F4A7C15
                ) & ((1 << 64) - 1),
            }
            for ordinal in range(HOLDOUT_SOURCE_GAMES)
        ],
    }
    plan = {
        **plan_body, "body_sha256": sha256_bytes(canonical_json_bytes(plan_body))
    }
    plan_path = output_directory / "procedural-game-plan.json"
    if plan_path.exists():
        if load_json(plan_path, "procedural holdout game plan") != plan:
            raise ValueError("procedural holdout game plan is stale")
    else:
        atomic_write(plan_path, canonical_json_bytes(plan, pretty=True))

    def run_game(row: Mapping[str, int]) -> pathlib.Path:
        ordinal = int(row["game_ordinal"])
        base_seed = int(row["base_seed"])
        path = raw_directory / f"game-{ordinal:04d}.jsonl"
        receipt_path = raw_directory / f"game-{ordinal:04d}.receipt.json"
        expected_common = {
            "schema": "papersoccer.jacek-rebuild-holdout-game-receipt.v1",
            "rebuild_id": REBUILD_ID,
            "game_ordinal": ordinal,
            "base_seed": base_seed,
            "generator": artifact_snapshot(generator),
            "configuration": {
                "games": 1,
                "base_budget": HOLDOUT_GENERATOR_BASE_BUDGET,
                "maximum_turns": 200,
            },
            "plan": artifact_snapshot(plan_path),
        }
        if receipt_path.exists():
            saved = load_json(receipt_path, "procedural holdout game receipt")
            lines = path.read_text(encoding="utf-8").splitlines()
            game = json.loads(lines[0]) if len(lines) == 1 else None
            successful_seed = game.get("seed") if isinstance(game, dict) else None
            if (
                not isinstance(game, dict)
                or game.get("schema")
                != "papersoccer.teacher-residual-samples.v1"
                or successful_seed
                not in {
                    (
                        base_seed + attempt * 0x9E3779B97F4A7C15
                    ) & ((1 << 64) - 1)
                    for attempt in range(30)
                }
            ):
                raise ValueError("procedural holdout saved game identity is invalid")
            expected = {**expected_common, "successful_seed": successful_seed}
            if (
                set(saved) != {*expected, "output"}
                or any(saved.get(key) != value for key, value in expected.items())
                or saved.get("output") != artifact_snapshot(path)
            ):
                raise ValueError("procedural holdout game receipt is stale")
            return path
        if path.exists():
            raise ValueError("procedural holdout game has no receipt")
        with tempfile.NamedTemporaryFile(
            dir=raw_directory, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = pathlib.Path(handle.name)
        temporary.unlink()
        try:
            _run(
                (
                    str(generator), str(temporary), "1",
                    str(HOLDOUT_GENERATOR_BASE_BUDGET), "200", str(base_seed),
                )
            )
            lines = temporary.read_text(encoding="utf-8").splitlines()
            if len(lines) != 1:
                raise ValueError("procedural holdout game output count is invalid")
            game = json.loads(lines[0])
            if (
                not isinstance(game, dict)
                or game.get("schema")
                != "papersoccer.teacher-residual-samples.v1"
                or not isinstance(game.get("seed"), int)
                or game.get("seed")
                not in {
                    (
                        base_seed + attempt * 0x9E3779B97F4A7C15
                    ) & ((1 << 64) - 1)
                    for attempt in range(30)
                }
            ):
                raise ValueError("procedural holdout game identity is invalid")
            os.replace(temporary, path)
            atomic_write(
                receipt_path,
                canonical_json_bytes(
                    {
                        **expected_common,
                        "successful_seed": game["seed"],
                        "output": artifact_snapshot(path),
                    },
                    pretty=True,
                ),
            )
            return path
        finally:
            temporary.unlink(missing_ok=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        raw_paths = list(executor.map(run_game, plan["rows"]))
    groups: list[tuple[bytes, str, list[str]]] = []
    rejected = 0
    for game_ordinal, path in enumerate(raw_paths):
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                try:
                    game = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError("procedural holdout game is invalid JSON") from error
                samples = game.get("samples") if isinstance(game, dict) else None
                winner = game.get("winner") if isinstance(game, dict) else None
                seed = game.get("seed") if isinstance(game, dict) else None
                if (
                    game.get("schema") != "papersoccer.teacher-residual-samples.v1"
                    or winner not in (0, 1)
                    or not isinstance(seed, int)
                    or not isinstance(samples, list)
                ):
                    raise ValueError("procedural holdout game schema is invalid")
                candidates = []
                seen_prefixes = set()
                for sample in samples:
                    if not isinstance(sample, dict):
                        continue
                    prefix = sample.get("transcript")
                    mover = sample.get("player_id")
                    if (
                        not isinstance(prefix, str)
                        or not prefix
                        or mover not in (0, 1)
                        or prefix in seen_prefixes
                    ):
                        continue
                    seen_prefixes.add(prefix)
                    candidates.append((prefix, mover))
                if len(candidates) < retention.ROWS_PER_GROUP:
                    rejected += 1
                    continue
                identity = hashlib.sha256(
                    f"{game_ordinal}\0{line_number}\0{seed}".encode()
                ).hexdigest()
                root_group_id = f"rebuild-holdout-root:{identity}"
                group_id = f"rebuild-holdout-game:{identity}"
                ordered = sorted(
                    candidates,
                    key=lambda row: hashlib.sha256(
                        f"{identity}\0{row[0]}".encode()
                    ).digest(),
                )[: retention.ROWS_PER_GROUP]
                rows = []
                for ordinal, (prefix, mover) in enumerate(ordered):
                    position_id = "position:" + hashlib.sha256(
                        f"{REBUILD_ID}\0{group_id}\0{ordinal}\0{prefix}".encode()
                    ).hexdigest()
                    rows.append(
                        "\t".join(
                            (
                                position_id, root_group_id, group_id,
                                "rank4-vs-rank4", "validation", str(winner),
                                str(mover), prefix,
                            )
                        )
                    )
                key = hashlib.sha256(
                    b"rebuild-holdout-candidate-v1\0"
                    + BLIND_HOLDOUT_SEED.to_bytes(8, "little")
                    + root_group_id.encode()
                ).digest()
                groups.append((key, root_group_id, rows))
    groups.sort(key=lambda item: (item[0], item[1]))
    if len(groups) < HOLDOUT_CANDIDATE_GROUPS:
        raise ValueError(
            "procedural holdout source cannot fill its candidate-group quota"
        )
    selected = groups[:HOLDOUT_CANDIDATE_GROUPS]
    payload = (
        retention.POSITION_HEADER
        + "\n"
        + "\n".join(row for _key, _group, rows in selected for row in rows)
        + "\n"
    ).encode("utf-8")
    positions_path = output_directory / "candidate-positions.tsv"
    manifest_path = output_directory / "candidate-positions.json"
    body: dict[str, object] = {
        "schema": "papersoccer.jacek-replay-rebuild-holdout-candidates.v1",
        "rebuild_id": REBUILD_ID,
        "generator": artifact_snapshot(generator),
        "configuration": {
            "source_games": HOLDOUT_SOURCE_GAMES,
            "worker_count_affects_output": False,
            "base_budget": HOLDOUT_GENERATOR_BASE_BUDGET,
            "maximum_turns": 200,
            "seed": HOLDOUT_GAME_SEED,
            "candidate_groups": HOLDOUT_CANDIDATE_GROUPS,
            "rows_per_group": retention.ROWS_PER_GROUP,
            "source": "rank4-vs-rank4",
        },
        "raw_outputs": [artifact_snapshot(path) for path in raw_paths],
        "raw_receipts": [
            artifact_snapshot(
                raw_directory / f"game-{ordinal:04d}.receipt.json"
            )
            for ordinal in range(HOLDOUT_SOURCE_GAMES)
        ],
        "game_plan": artifact_snapshot(plan_path),
        "accepted_source_groups": len(groups),
        "rejected_short_games": rejected,
        "positions_sha256": sha256_bytes(payload),
        "rows": HOLDOUT_CANDIDATE_GROUPS * retention.ROWS_PER_GROUP,
        "teacher_labels_opened": False,
        "selected_model_opened": False,
    }
    manifest = {**body, "body_sha256": sha256_bytes(canonical_json_bytes(body))}
    if positions_path.exists() or manifest_path.exists():
        if (
            not positions_path.is_file()
            or positions_path.read_bytes() != payload
            or not manifest_path.is_file()
            or load_json(manifest_path, "holdout candidates") != manifest
        ):
            raise ValueError("existing holdout candidate pool is stale")
    else:
        atomic_write(positions_path, payload)
        atomic_write(manifest_path, canonical_json_bytes(manifest, pretty=True))
    # Reparse every row through the same strict position loader used by freeze.
    parsed = retention.load_position_rows(
        positions_path, required_split=retention.TEACHER_SPLIT
    )
    if len(parsed) != body["rows"]:
        raise RuntimeError("procedural holdout candidate coverage changed")
    return {
        "positions": artifact_snapshot(positions_path),
        "manifest": artifact_snapshot(manifest_path),
        "groups": HOLDOUT_CANDIDATE_GROUPS,
        "rows": body["rows"],
    }


def validate_holdout_candidate_pool(
    positions_path: pathlib.Path, manifest_path: pathlib.Path
) -> dict[str, object]:
    import jacek_replay_retention as retention

    manifest = load_json(manifest_path, "holdout candidate manifest")
    verify_body_hash(
        manifest,
        schema="papersoccer.jacek-replay-rebuild-holdout-candidates.v1",
        label="holdout candidate manifest",
    )
    configuration = manifest.get("configuration")
    if (
        not isinstance(configuration, dict)
        or configuration
        != {
            "source_games": HOLDOUT_SOURCE_GAMES,
            "worker_count_affects_output": False,
            "base_budget": HOLDOUT_GENERATOR_BASE_BUDGET,
            "maximum_turns": 200,
            "seed": HOLDOUT_GAME_SEED,
            "candidate_groups": HOLDOUT_CANDIDATE_GROUPS,
            "rows_per_group": retention.ROWS_PER_GROUP,
            "source": "rank4-vs-rank4",
        }
        or manifest.get("teacher_labels_opened") is not False
        or manifest.get("selected_model_opened") is not False
        or manifest.get("rows")
        != HOLDOUT_CANDIDATE_GROUPS * retention.ROWS_PER_GROUP
        or sha256_file(positions_path) != manifest.get("positions_sha256")
    ):
        raise ValueError("holdout candidate policy changed")
    generator = manifest.get("generator")
    plan_snapshot = manifest.get("game_plan")
    raw_outputs = manifest.get("raw_outputs")
    raw_receipts = manifest.get("raw_receipts")
    if (
        not isinstance(generator, dict)
        or not isinstance(plan_snapshot, dict)
        or not isinstance(raw_outputs, list)
        or len(raw_outputs) != HOLDOUT_SOURCE_GAMES
        or not isinstance(raw_receipts, list)
        or len(raw_receipts) != HOLDOUT_SOURCE_GAMES
    ):
        raise ValueError("holdout candidate lineage is incomplete")
    for snapshot in (generator, plan_snapshot, *raw_outputs, *raw_receipts):
        if (
            not isinstance(snapshot.get("path"), str)
            or artifact_snapshot(pathlib.Path(snapshot["path"])) != snapshot
        ):
            raise ValueError("holdout candidate lineage binding is stale")
    plan = load_json(pathlib.Path(plan_snapshot["path"]), "holdout game plan")
    verify_body_hash(
        plan,
        schema="papersoccer.jacek-rebuild-holdout-game-plan.v1",
        label="holdout game plan",
    )
    expected_rows = [
        {
            "game_ordinal": ordinal,
            "base_seed": (
                HOLDOUT_GAME_SEED + ordinal * 32 * 0x9E3779B97F4A7C15
            ) & ((1 << 64) - 1),
        }
        for ordinal in range(HOLDOUT_SOURCE_GAMES)
    ]
    if (
        plan.get("rebuild_id") != REBUILD_ID
        or plan.get("generator") != generator
        or plan.get("games") != HOLDOUT_SOURCE_GAMES
        or plan.get("base_budget") != HOLDOUT_GENERATOR_BASE_BUDGET
        or plan.get("maximum_turns") != 200
        or plan.get("root_seed") != HOLDOUT_GAME_SEED
        or plan.get("rows") != expected_rows
    ):
        raise ValueError("holdout game plan semantics changed")
    for ordinal, (output_snapshot, receipt_snapshot, row) in enumerate(
        zip(raw_outputs, raw_receipts, expected_rows, strict=True)
    ):
        receipt = load_json(
            pathlib.Path(receipt_snapshot["path"]), "holdout game receipt"
        )
        output_lines = pathlib.Path(output_snapshot["path"]).read_text(
            encoding="utf-8"
        ).splitlines()
        game = json.loads(output_lines[0]) if len(output_lines) == 1 else None
        successful_seed = game.get("seed") if isinstance(game, dict) else None
        if successful_seed not in {
            (
                row["base_seed"] + attempt * 0x9E3779B97F4A7C15
            ) & ((1 << 64) - 1)
            for attempt in range(30)
        }:
            raise ValueError("holdout game successful seed is outside its range")
        expected_receipt = {
            "schema": "papersoccer.jacek-rebuild-holdout-game-receipt.v1",
            "rebuild_id": REBUILD_ID,
            "game_ordinal": ordinal,
            "base_seed": row["base_seed"],
            "successful_seed": successful_seed,
            "generator": generator,
            "configuration": {
                "games": 1,
                "base_budget": HOLDOUT_GENERATOR_BASE_BUDGET,
                "maximum_turns": 200,
            },
            "plan": plan_snapshot,
            "output": output_snapshot,
        }
        if receipt != expected_receipt:
            raise ValueError("holdout game receipt semantics changed")
    groups: list[tuple[bytes, str, list[str]]] = []
    rejected = 0
    for game_ordinal, output_snapshot in enumerate(raw_outputs):
        lines = pathlib.Path(output_snapshot["path"]).read_text(
            encoding="utf-8"
        ).splitlines()
        game = json.loads(lines[0]) if len(lines) == 1 else None
        if (
            not isinstance(game, dict)
            or game.get("schema") != "papersoccer.teacher-residual-samples.v1"
            or game.get("winner") not in (0, 1)
            or not isinstance(game.get("seed"), int)
            or not isinstance(game.get("samples"), list)
        ):
            raise ValueError("holdout raw game semantics changed")
        candidates = []
        seen_prefixes = set()
        for sample in game["samples"]:
            if not isinstance(sample, dict):
                continue
            prefix, mover = sample.get("transcript"), sample.get("player_id")
            if (
                not isinstance(prefix, str)
                or not prefix
                or mover not in (0, 1)
                or prefix in seen_prefixes
            ):
                continue
            seen_prefixes.add(prefix)
            candidates.append((prefix, mover))
        if len(candidates) < retention.ROWS_PER_GROUP:
            rejected += 1
            continue
        identity = hashlib.sha256(
            f"{game_ordinal}\0{1}\0{game['seed']}".encode()
        ).hexdigest()
        root_group_id = f"rebuild-holdout-root:{identity}"
        group_id = f"rebuild-holdout-game:{identity}"
        ordered = sorted(
            candidates,
            key=lambda value: hashlib.sha256(
                f"{identity}\0{value[0]}".encode()
            ).digest(),
        )[: retention.ROWS_PER_GROUP]
        rendered = []
        for row_ordinal, (prefix, mover) in enumerate(ordered):
            position_id = "position:" + hashlib.sha256(
                f"{REBUILD_ID}\0{group_id}\0{row_ordinal}\0{prefix}".encode()
            ).hexdigest()
            rendered.append(
                "\t".join(
                    (
                        position_id, root_group_id, group_id,
                        "rank4-vs-rank4", "validation", str(game["winner"]),
                        str(mover), prefix,
                    )
                )
            )
        key = hashlib.sha256(
            b"rebuild-holdout-candidate-v1\0"
            + BLIND_HOLDOUT_SEED.to_bytes(8, "little")
            + root_group_id.encode()
        ).digest()
        groups.append((key, root_group_id, rendered))
    groups.sort(key=lambda item: (item[0], item[1]))
    if len(groups) < HOLDOUT_CANDIDATE_GROUPS:
        raise ValueError("holdout raw games no longer fill candidate quota")
    expected_payload = (
        retention.POSITION_HEADER
        + "\n"
        + "\n".join(
            row for _key, _group, rendered in groups[:HOLDOUT_CANDIDATE_GROUPS]
            for row in rendered
        )
        + "\n"
    ).encode("utf-8")
    if (
        positions_path.read_bytes() != expected_payload
        or manifest.get("accepted_source_groups") != len(groups)
        or manifest.get("rejected_short_games") != rejected
    ):
        raise ValueError("holdout candidate selection was substituted")
    rows = retention.load_position_rows(
        positions_path, required_split=retention.TEACHER_SPLIT
    )
    if (
        len(rows) != manifest["rows"]
        or len({row.root_group_id for row in rows}) != HOLDOUT_CANDIDATE_GROUPS
    ):
        raise ValueError("holdout candidate position coverage changed")
    return manifest


def freeze_rebuild_inputs(
    *, repository: pathlib.Path, corpus_manifest: pathlib.Path,
    build_manifest: pathlib.Path,
    matrix_manifest: pathlib.Path, banks_manifest: pathlib.Path,
    holdout_manifest: pathlib.Path, holdout_positions: pathlib.Path,
    holdout_candidate_manifest: pathlib.Path,
    holdout_candidate_positions: pathlib.Path,
    comparison: pathlib.Path, rank4_teacher: pathlib.Path,
    incumbent_runtime: pathlib.Path, output: pathlib.Path,
    started_at_unix: float | None = None,
) -> dict[str, object]:
    import jacek_rebuild_corpus as rebuild_corpus
    import jacek_replay_retention as retention

    repository = repository.resolve()
    if not (repository / ".git").exists() and not (
        repository / ".git"
    ).is_file():
        raise ValueError("rebuild repository is not a Git checkout")
    corpus = load_json(corpus_manifest, "rebuild corpus manifest")
    load_frozen_rebuild_corpus(corpus_manifest)
    validate_rebuild_build_manifest(build_manifest)
    matrix = load_json(matrix_manifest, "rebuild matrix")
    validate_matrix(matrix)
    banks = load_json(banks_manifest, "rebuild opening banks")
    validate_opening_banks(banks)
    holdout, holdout_rows = load_frozen_blind_holdout(
        holdout_positions, holdout_manifest
    )
    if (
        holdout.get("timing", {}).get("teacher_labels_opened") is not False
        or holdout.get("timing", {}).get("selected_model_opened") is not False
        or holdout.get("timing", {}).get("training_inputs_frozen_by")
        != retention.artifact_snapshot(corpus_manifest)
        or len(holdout_rows) != 12_000
    ):
        raise ValueError("blind holdout was not frozen before model selection")
    started = time.time() if started_at_unix is None else started_at_unix
    if not math.isfinite(started) or started <= 0.0:
        raise ValueError("rebuild start time is invalid")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repository, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )
    if head.returncode or status.returncode or status.stdout:
        raise ValueError("rebuild input freeze requires a clean repository")
    body: dict[str, object] = {
        "schema": REBUILD_INPUT_SCHEMA,
        "rebuild_id": REBUILD_ID,
        "repository": {
            "path": str(repository), "commit": head.stdout.strip(), "clean": True,
        },
        "started_at_unix": started,
        "same_architecture_deadline_unix": (
            started + SAME_ARCHITECTURE_BUDGET_SECONDS
        ),
        "corpus": artifact_snapshot(corpus_manifest),
        "build_manifest": artifact_snapshot(build_manifest),
        "matrix": artifact_snapshot(matrix_manifest),
        "opening_banks": artifact_snapshot(banks_manifest),
        "blind_holdout": artifact_snapshot(holdout_manifest),
        "blind_holdout_positions": artifact_snapshot(holdout_positions),
        "blind_holdout_candidate_manifest": artifact_snapshot(
            holdout_candidate_manifest
        ),
        "blind_holdout_candidate_positions": artifact_snapshot(
            holdout_candidate_positions
        ),
        "comparison": artifact_snapshot(comparison),
        "rank4_teacher": artifact_snapshot(rank4_teacher),
        "incumbent_runtime": artifact_snapshot(incumbent_runtime),
        "policies": {
            "regenerate_teacher_labels": False,
            "external_upload": False,
            "replace_rank4": False,
            "canonical_test_model_selection_eligible": False,
            "sealed_final_bank_model_selection_eligible": False,
        },
    }
    record = {**body, "body_sha256": sha256_bytes(canonical_json_bytes(body))}
    if output.exists():
        if load_json(output, "frozen rebuild inputs") != record:
            raise ValueError("existing frozen rebuild inputs are stale")
    else:
        atomic_write(output, canonical_json_bytes(record, pretty=True))
    validate_rebuild_inputs(output)
    return record


def validate_rebuild_inputs(path: pathlib.Path) -> dict[str, object]:
    import jacek_rebuild_corpus as rebuild_corpus
    import jacek_replay_retention as retention

    record = load_json(path, "frozen rebuild inputs")
    body = dict(record)
    claimed = body.pop("body_sha256", None)
    if (
        body.get("schema") != REBUILD_INPUT_SCHEMA
        or body.get("rebuild_id") != REBUILD_ID
        or claimed != sha256_bytes(canonical_json_bytes(body))
    ):
        raise ValueError("frozen rebuild inputs are stale or corrupt")
    policies = body.get("policies")
    if not isinstance(policies, dict) or any(
        policies.get(key) is not False
        for key in (
            "regenerate_teacher_labels", "external_upload", "replace_rank4",
            "canonical_test_model_selection_eligible",
            "sealed_final_bank_model_selection_eligible",
        )
    ):
        raise ValueError("frozen rebuild input policy was weakened")
    for label in (
        "corpus", "build_manifest", "matrix", "opening_banks", "blind_holdout", "comparison",
        "blind_holdout_positions", "blind_holdout_candidate_manifest",
        "blind_holdout_candidate_positions",
        "rank4_teacher", "incumbent_runtime",
    ):
        snapshot = body.get(label)
        if (
            not isinstance(snapshot, dict)
            or artifact_snapshot(pathlib.Path(str(snapshot.get("path", ""))))
            != snapshot
        ):
            raise ValueError(f"frozen rebuild {label} binding is stale")
    corpus_path = pathlib.Path(body["corpus"]["path"])
    corpus = load_frozen_rebuild_corpus(corpus_path)
    validate_matrix(load_json(pathlib.Path(body["matrix"]["path"]), "rebuild matrix"))
    validate_opening_banks(
        load_json(pathlib.Path(body["opening_banks"]["path"]), "rebuild banks")
    )
    build = validate_rebuild_build_manifest(
        pathlib.Path(body["build_manifest"]["path"])
    )
    if (
        build["binaries"]["comparison"] != body["comparison"]
        or build["binaries"]["rank4_teacher"] != body["rank4_teacher"]
    ):
        raise ValueError("rebuild inputs do not match the frozen build")
    repository = pathlib.Path(body["repository"]["path"])
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repository, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )
    started = body.get("started_at_unix")
    deadline = body.get("same_architecture_deadline_unix")
    if (
        head.returncode
        or head.stdout.strip() != body["repository"].get("commit")
        or status.returncode
        or status.stdout
        or body["repository"].get("clean") is not True
        or not isinstance(started, (int, float))
        or not isinstance(deadline, (int, float))
        or float(deadline) != float(started) + SAME_ARCHITECTURE_BUDGET_SECONDS
    ):
        raise ValueError("frozen rebuild repository/deadline semantics changed")
    freeze_path = pathlib.Path(body["blind_holdout"]["path"])
    positions_path = pathlib.Path(body["blind_holdout_positions"]["path"])
    candidate_manifest_path = pathlib.Path(
        body["blind_holdout_candidate_manifest"]["path"]
    )
    candidate_positions_path = pathlib.Path(
        body["blind_holdout_candidate_positions"]["path"]
    )
    validate_holdout_candidate_pool(
        candidate_positions_path, candidate_manifest_path
    )
    candidate_manifest = load_json(
        candidate_manifest_path, "holdout candidate manifest"
    )
    freeze, rows = load_frozen_blind_holdout(positions_path, freeze_path)
    expected_shards = {
        *corpus.training_manifest_paths("search"),
        *corpus.training_manifest_paths("rank4"),
        *corpus.anchor_manifest_paths("search"),
        *corpus.validation_manifest_paths("search"),
        *corpus.retention_validation_manifest_paths("search"),
    }
    observed_shards = {
        pathlib.Path(record["path"]).resolve()
        for record in freeze.get("inputs", {}).get("excluded_shards", [])
        if isinstance(record, dict) and isinstance(record.get("path"), str)
    }
    sealed_shard_records = freeze.get("inputs", {}).get(
        "excluded_sealed_identity_shards", []
    )
    if not isinstance(sealed_shard_records, list):
        raise ValueError("sealed holdout exclusion shard list is malformed")
    observed_sealed_shards = {
        pathlib.Path(record["path"]).resolve()
        for record in sealed_shard_records
        if isinstance(record, dict) and isinstance(record.get("path"), str)
    }
    expected_sealed_shards = {
        path.resolve() for path in corpus.protected_test_manifest_paths
    }
    sealed_fingerprints = _sealed_feature_fingerprints(
        tuple(sorted(expected_sealed_shards))
    )
    sealed_exclusion = freeze.get("sealed_identity_exclusion")
    excluded_positions = freeze.get("inputs", {}).get("excluded_positions", [])
    excluded_roots = freeze.get("inputs", {}).get("excluded_roots", [])
    if (
        freeze.get("timing", {}).get("training_inputs_frozen_by")
        != retention.artifact_snapshot(corpus_path)
        or freeze.get("inputs", {}).get("candidate_positions")
        != retention.artifact_snapshot(candidate_positions_path)
        or candidate_manifest.get("generator")
        != build["binaries"]["holdout_generator"]
        or len(rows) != 12_000
        or observed_shards != {path.resolve() for path in expected_shards}
        or any(
            not isinstance(record, dict)
            or not isinstance(record.get("path"), str)
            or retention.artifact_snapshot(pathlib.Path(record["path"]))
            != record
            for record in sealed_shard_records
        )
        or observed_sealed_shards != expected_sealed_shards
        or not isinstance(sealed_exclusion, dict)
        or sealed_exclusion
        != {
            "policy": "feature-indices-only-targets-and-weights-unopened",
            "canonical_fingerprints": len(sealed_fingerprints),
            "canonical_fingerprints_sha256": hashlib.sha256(
                b"".join(sorted(sealed_fingerprints))
            ).hexdigest(),
            "selected_overlap": 0,
        }
        or any(
            row.canonical_fingerprint in sealed_fingerprints for row in rows
        )
        or not isinstance(excluded_positions, list)
        or len(excluded_positions) != 2
        or {
            pathlib.Path(record["path"]).name
            for record in excluded_positions if isinstance(record, dict)
        } != {"positions.tsv"}
        or {
            pathlib.Path(record["path"]).parents[1].name
            for record in excluded_positions if isinstance(record, dict)
        }
        != {
            "selfsearch-auto-20260825-v5",
            "selfsearch-auto-20260825-v6",
        }
        or not isinstance(excluded_roots, list)
        or len(excluded_roots) != 3
        or {
            pathlib.Path(record["path"]).parent.name
            for record in excluded_roots if isinstance(record, dict)
        } != {"round-0", "round-1", "round-2"}
    ):
        raise ValueError("frozen blind holdout exclusion universe changed")
    return record


def write_rebuild_build_manifest(
    *, repository: pathlib.Path, expected_commit: str,
    comparison: pathlib.Path, rank4_teacher: pathlib.Path,
    holdout_generator: pathlib.Path, output: pathlib.Path,
) -> dict[str, object]:
    repository = repository.resolve()
    binaries = tuple(
        path.resolve() for path in (comparison, rank4_teacher, holdout_generator)
    )
    if any(not path.is_file() for path in binaries):
        raise ValueError("rebuild Release binary is missing")
    build_directories = {path.parent for path in binaries}
    if len(build_directories) != 1:
        raise ValueError("rebuild binaries must come from one build")
    build_directory = next(iter(build_directories))
    cache = build_directory / "CMakeCache.txt"
    if not cache.is_file():
        raise ValueError("rebuild Release build cache is missing")
    cache_text = cache.read_text(encoding="utf-8", errors="replace")
    if (
        "CMAKE_BUILD_TYPE:STRING=Release" not in cache_text
        or "PAPERSOCCER_ENABLE_SANITIZERS:BOOL=OFF" not in cache_text
        or f"CMAKE_HOME_DIRECTORY:INTERNAL={repository}" not in cache_text
    ):
        raise ValueError("rebuild binaries are not an unsanitized Release build")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repository, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )
    if (
        head.returncode
        or head.stdout.strip() != expected_commit
        or status.returncode
        or status.stdout
    ):
        raise ValueError("rebuild build manifest requires the exact clean commit")
    tools = {
        name: repository / path
        for name, path in {
            "workflow": "tools/jacek_replay_rebuild.py",
            "corpus": "tools/jacek_rebuild_corpus.py",
            "recovery": "tools/jacek_replay_recovery.py",
            "trainer": "tools/jacek_replay_train.py",
            "retention": "tools/jacek_replay_retention.py",
            "selfsearch_workflow": "tools/jacek_selfsearch_workflow.py",
            "features": "tools/jacek_replay_features.py",
            "cmake": "CMakeLists.txt",
        }.items()
    }
    body: dict[str, object] = {
        "schema": REBUILD_BUILD_SCHEMA,
        "rebuild_id": REBUILD_ID,
        "repository": {
            "path": str(repository), "commit": expected_commit, "clean": True,
        },
        "build": {
            "directory": str(build_directory),
            "cmake_cache": artifact_snapshot(cache),
            "type": "Release",
            "sanitizers": False,
        },
        "binaries": {
            "comparison": artifact_snapshot(binaries[0]),
            "rank4_teacher": artifact_snapshot(binaries[1]),
            "holdout_generator": artifact_snapshot(binaries[2]),
        },
        "tools": {name: artifact_snapshot(path) for name, path in tools.items()},
    }
    manifest = {**body, "body_sha256": sha256_bytes(canonical_json_bytes(body))}
    if output.exists():
        if load_json(output, "rebuild build manifest") != manifest:
            raise ValueError("existing rebuild build manifest is stale")
    else:
        atomic_write(output, canonical_json_bytes(manifest, pretty=True))
    return manifest


def validate_rebuild_build_manifest(path: pathlib.Path) -> dict[str, object]:
    manifest = load_json(path, "rebuild build manifest")
    body = dict(manifest)
    claimed = body.pop("body_sha256", None)
    if (
        body.get("schema") != REBUILD_BUILD_SCHEMA
        or body.get("rebuild_id") != REBUILD_ID
        or claimed != sha256_bytes(canonical_json_bytes(body))
    ):
        raise ValueError("rebuild build manifest integrity failed")
    expected_tool_paths = {
        "workflow": "tools/jacek_replay_rebuild.py",
        "corpus": "tools/jacek_rebuild_corpus.py",
        "recovery": "tools/jacek_replay_recovery.py",
        "trainer": "tools/jacek_replay_train.py",
        "retention": "tools/jacek_replay_retention.py",
        "selfsearch_workflow": "tools/jacek_selfsearch_workflow.py",
        "features": "tools/jacek_replay_features.py",
        "cmake": "CMakeLists.txt",
    }
    expected_binary_names = {
        "comparison": "papersoccer_jacek_replay_comparison",
        "rank4_teacher": "papersoccer_jacek_replay_rank4_position_teacher",
        "holdout_generator": "papersoccer_codingame_rank_4_teacher_sample_generator",
    }
    if (
        set(body.get("tools", {})) != set(expected_tool_paths)
        or set(body.get("binaries", {})) != set(expected_binary_names)
    ):
        raise ValueError("rebuild build role set changed")
    for section in ("binaries", "tools"):
        records = body.get(section)
        if not isinstance(records, dict) or not records:
            raise ValueError(f"rebuild build {section} are missing")
        for record in records.values():
            if (
                not isinstance(record, dict)
                or artifact_snapshot(pathlib.Path(str(record.get("path", ""))))
                != record
            ):
                raise ValueError(f"rebuild build {section} binding is stale")
    repository = pathlib.Path(body["repository"]["path"])
    commit = str(body["repository"]["commit"])
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repository, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )
    cache = pathlib.Path(body["build"]["cmake_cache"]["path"])
    cache_text = cache.read_text(encoding="utf-8", errors="replace")
    build_directory = pathlib.Path(body["build"]["directory"])
    if (
        head.returncode
        or head.stdout.strip() != commit
        or status.returncode
        or status.stdout
        or body["repository"].get("clean") is not True
        or body["build"].get("type") != "Release"
        or body["build"].get("sanitizers") is not False
        or cache.parent != build_directory
        or f"CMAKE_HOME_DIRECTORY:INTERNAL={repository}" not in cache_text
        or "CMAKE_BUILD_TYPE:STRING=Release" not in cache_text
        or "PAPERSOCCER_ENABLE_SANITIZERS:BOOL=OFF" not in cache_text
        or any(
            pathlib.Path(record["path"]).parent != build_directory
            for record in body["binaries"].values()
        )
        or any(
            pathlib.Path(body["tools"][name]["path"]).resolve()
            != (repository / relative).resolve()
            for name, relative in expected_tool_paths.items()
        )
        or any(
            pathlib.Path(body["binaries"][name]["path"]).name != basename
            for name, basename in expected_binary_names.items()
        )
    ):
        raise ValueError("rebuild build repository/build semantics changed")
    return manifest


@dataclasses.dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    phase: str
    base_id: str
    search_initial_runtime: pathlib.Path
    rank4_initial_runtime: pathlib.Path
    trainable_layers: str
    learning_rate: float
    selection_policy: str
    training_recipe: str
    order_seeds: tuple[int, ...] = ORDER_SEEDS

    def validate(self) -> None:
        if (
            not self.candidate_id
            or self.phase not in {"v5-recovery", "canonical-basin", "scratch", "residual"}
            or not self.base_id
            or self.trainable_layers not in {"w3", "w2-w3", "all", "adapter"}
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0.0
            or self.selection_policy
            not in {"v5-recovery-noninferiority", "epoch-zero-improvement"}
            or len(self.order_seeds) != 3
            or self.training_recipe not in {"recovery", "v6-joint"}
            or len(set(self.order_seeds)) != 3
        ):
            raise ValueError("rebuild candidate specification is invalid")
        for path in (self.search_initial_runtime, self.rank4_initial_runtime):
            if not path.is_file():
                raise ValueError(f"rebuild base runtime is missing: {path}")

    def record(self) -> dict[str, object]:
        self.validate()
        record = {
            "schema": REBUILD_CANDIDATE_SPEC_SCHEMA,
            "candidate_id": self.candidate_id,
            "phase": self.phase,
            "base_id": self.base_id,
            "search_initial_runtime": artifact_snapshot(
                self.search_initial_runtime
            ),
            "rank4_initial_runtime": artifact_snapshot(
                self.rank4_initial_runtime
            ),
            "trainable_layers": self.trainable_layers,
            "learning_rate": self.learning_rate,
            "selection_policy": self.selection_policy,
            "training_recipe": self.training_recipe,
            "order_seeds": list(self.order_seeds),
            "batching": {
                "new_rows": NEW_ROWS_PER_BATCH,
                "anchor_rows": ANCHOR_ROWS_PER_BATCH,
            },
            "loss": {
                "new_coefficient": NEW_LOSS_COEFFICIENT,
                "anchor_coefficient": ANCHOR_LOSS_COEFFICIENT,
            },
        }
        record["training_schedule"] = (
            {
                "kind": "fixed-update-recovery-v1",
                "checkpoint_interval_updates": CHECKPOINT_UPDATES,
                "maximum_anchor_passes": MAX_ANCHOR_PASSES,
            }
            if self.training_recipe == "recovery"
            else {
                "kind": "v6-epoch-retention-safe-v1",
                "epochs": 50,
                "patience": 8,
                "patience_starts_after_complete_anchor_coverage": True,
                "new_stream": "fresh-complete-permutation-each-epoch",
                "anchor_stream": "continuous-no-repeat",
                "base_learning_rate": JOINT_LEARNING_RATE,
                "reference_new_rows": 50_000,
                "reference_optimizer_steps": 782,
            }
        )
        return record


def v5_recovery_specs(
    *, search_runtime: pathlib.Path, rank4_runtime: pathlib.Path
) -> tuple[CandidateSpec, ...]:
    result = []
    for layers in RECOVERY_LAYER_SCOPES:
        for ordinal, learning_rate in enumerate(RECOVERY_LEARNING_RATES):
            result.append(
                CandidateSpec(
                    candidate_id=f"v5-recovery-{layers}-lr{ordinal}",
                    phase="v5-recovery",
                    base_id="v5-selected-pair",
                    search_initial_runtime=search_runtime,
                    rank4_initial_runtime=rank4_runtime,
                    trainable_layers=layers,
                    learning_rate=learning_rate,
                    selection_policy="v5-recovery-noninferiority",
                    training_recipe="recovery",
                )
            )
    return tuple(result)


def canonical_basin_specs(
    runtimes: Sequence[tuple[str, pathlib.Path]],
) -> tuple[CandidateSpec, ...]:
    if len(runtimes) != 9 or len({name for name, _path in runtimes}) != 9:
        raise ValueError("canonical basin sweep requires exactly nine bases")
    result = tuple(
        CandidateSpec(
            candidate_id=f"canonical-{name}",
            phase="canonical-basin",
            base_id=name,
            search_initial_runtime=path,
            rank4_initial_runtime=path,
            trainable_layers="all",
            learning_rate=JOINT_LEARNING_RATE,
            selection_policy="epoch-zero-improvement",
            training_recipe="v6-joint",
        )
        for name, path in sorted(runtimes)
    )
    for spec in result:
        spec.validate()
    return result


def scratch_seed_groups() -> tuple[tuple[int, int, int], ...]:
    return tuple(
        tuple(SCRATCH_SEEDS[index : index + 3])
        for index in range(0, len(SCRATCH_SEEDS), 3)
    )


def _load_historical_seed_checkpoint(
    *, directory: pathlib.Path, seed: int,
    expected_configuration: Mapping[str, object],
    expected_inputs: Mapping[str, object],
    expected_producer: Mapping[str, object],
    validation_dataset: object,
) -> tuple[dict[str, object], dict, dict]:
    """Validate an immutable v1 seed receipt and recompute selection metrics."""

    import jacek_replay_train as training

    checkpoint_path, receipt_path = training._seed_checkpoint_paths(directory, seed)
    try:
        receipt_payload = receipt_path.read_bytes()
        receipt = json.loads(receipt_payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("historical seed checkpoint receipt is invalid") from error
    if (
        not isinstance(receipt, dict)
        or set(receipt) != {
            "schema", "seed", "configuration", "inputs", "producer",
            "checkpoint", "training_report", "body_sha256",
        }
        or receipt.get("schema")
        != "papersoccer.jacek-replay-bfm-seed-checkpoint.v1"
        or receipt_payload != training.canonical_json_bytes(receipt)
    ):
        raise ValueError("historical seed checkpoint receipt shape changed")
    body = dict(receipt)
    claimed = body.pop("body_sha256", None)
    if (
        claimed != sha256_bytes(training.canonical_json_bytes(body))
        or receipt.get("seed") != seed
        or receipt.get("configuration") != expected_configuration
        or receipt.get("inputs") != expected_inputs
        or receipt.get("producer") != expected_producer
    ):
        raise ValueError("historical seed checkpoint provenance changed")
    parameters, runtime_report = training.load_runtime(checkpoint_path)
    if receipt.get("checkpoint") != {
        "file": checkpoint_path.name, **runtime_report
    }:
        raise ValueError("historical seed checkpoint runtime changed")
    report = training._validate_seed_report(
        receipt.get("training_report"),
        seed,
        expected_configuration,
        expected_inputs,
        runtime_report,
    )
    if report.get("validation") != training.metrics(
        parameters, validation_dataset
    ):
        raise ValueError("historical seed validation metrics changed")
    publication = training._seed_checkpoint_publication(
        seed, checkpoint_path, receipt_path, runtime_report, receipt_payload
    )
    return parameters, report, publication


def _validate_opaque_dataset_identity(
    identity: object, manifests: Sequence[Mapping[str, object]], label: str
) -> dict[str, object]:
    """Validate counts and digest shape without decoding a non-selection split."""

    if (
        not isinstance(identity, dict)
        or set(identity) != {"samples", "active_features", "sha256"}
        or identity.get("samples")
        != sum(int(manifest["samples"]) for manifest in manifests)
        or identity.get("active_features")
        != sum(int(manifest["active_features"]) for manifest in manifests)
        or not isinstance(identity.get("sha256"), str)
        or len(identity["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in identity["sha256"])
    ):
        raise ValueError(f"{label} dataset identity is malformed")
    return dict(identity)


def canonical_runtime_bindings(
    canonical_campaign: pathlib.Path,
) -> tuple[tuple[str, pathlib.Path], ...]:
    import jacek_replay_train as training
    import jacek_replay_workflow as canonical_workflow
    import jacek_selfsearch_workflow as selfsearch

    canonical_campaign = canonical_campaign.resolve()
    canonical_workflow.validate_canonical_workflow_chain(
        canonical_campaign / "round-2/workflow.json",
        expected_round=2,
        offline=True,
    )
    manifests = selfsearch._canonical_split_manifests(canonical_campaign)
    bindings = []
    for round_index in range(3):
        round_manifests = {
            split: tuple(paths[: round_index + 1])
            for split, paths in manifests.items()
        }
        loaded_validation = [
            training.load_csr_shard(path)
            for path in round_manifests["validation"]
        ]
        validation_dataset = training.combine_shards(loaded_validation)
        manifest_path = (
            canonical_campaign
            / f"round-{round_index}/model/jacek_replay_bfm.runtime.json"
        )
        model_manifest = load_json(manifest_path, "canonical round model")
        model_training = model_manifest.get("training")
        tool_sha = model_manifest.get("tool_sha256")
        if not isinstance(model_training, dict) or not isinstance(tool_sha, dict):
            raise ValueError("canonical round model provenance is incomplete")
        expected_configuration = {
            "architecture": {
                "dimensions": [training.features.INPUT_COUNT, 192, 32, 1],
                "biases": False,
                "activations": [
                    "square-leaky-0.01", "leaky-relu-0.01", "tanh"
                ],
            },
            "optimizer": {
                "name": "adamw", "epochs": 50, "patience": 8,
                "batch_size": 256, "learning_rate": 0.001,
                "weight_decay": 1e-5, "gradient_norm_clip": 5.0,
            },
            "loss": {"name": "weighted-huber", "delta": 0.25},
            "metrics_batch_size": 4096,
            "augmentation": "reflection rows inherit root game split",
        }
        expected_shards = training._shard_identities(
            tuple(
                round_manifests[split][ordinal]
                for ordinal in range(round_index + 1)
                for split in ("train", "validation", "test")
            )
        )
        first_receipt = load_json(
            canonical_campaign
            / f"round-{round_index}/training-seeds/seed-20260823.json",
            "canonical seed checkpoint",
        )
        receipt_datasets = first_receipt.get("inputs", {}).get("datasets", {})
        train_manifests = [
            load_json(path, "canonical training shard")
            for path in round_manifests["train"]
        ]
        test_manifests = [
            load_json(path, "sealed canonical test shard")
            for path in round_manifests["test"]
        ]
        train_identity = _validate_opaque_dataset_identity(
            receipt_datasets.get("train"), train_manifests,
            "canonical training",
        )
        sealed_test = _validate_opaque_dataset_identity(
            receipt_datasets.get("test"), test_manifests,
            "canonical protected-test",
        )
        expected_inputs = {
            "datasets": {
                "train": train_identity,
                "validation": training._dataset_identity(validation_dataset),
                "test": sealed_test,
            },
            "feature_schema": training.features.FEATURE_SCHEMA,
            "shards": expected_shards,
        }
        expected_producer = {
            "trainer_sha256": tool_sha.get("trainer"),
            "features_sha256": tool_sha.get("features"),
            "corpus_sha256": tool_sha.get("corpus"),
        }
        seeds = (20260823, 20260824, 20260825)
        reports = model_training.get("seed_reports")
        publications = model_training.get("seed_checkpoints")
        if (
            model_training.get("seeds") != list(seeds)
            or not isinstance(reports, list)
            or not isinstance(publications, list)
            or len(reports) != 3
            or len(publications) != 3
        ):
            raise ValueError("canonical round seed roster is incomplete")
        seed_directory = canonical_campaign / f"round-{round_index}/training-seeds"
        for seed, expected_report, expected_publication in zip(
            seeds, reports, publications, strict=True
        ):
            runtime = seed_directory / f"seed-{seed}.runtime"
            receipt = seed_directory / f"seed-{seed}.json"
            if not runtime.is_file() or not receipt.is_file():
                raise ValueError("canonical rebuild base is missing")
            _parameters, report, publication = _load_historical_seed_checkpoint(
                directory=seed_directory,
                seed=seed,
                expected_configuration=expected_configuration,
                expected_inputs=expected_inputs,
                expected_producer=expected_producer,
                validation_dataset=validation_dataset,
            )
            receipt_expected_report = dict(expected_report)
            revealed_test = receipt_expected_report.pop("test", None)
            if revealed_test is not None and (
                round_index != 2
                or seed != model_training.get("chosen_seed")
                or not isinstance(revealed_test, dict)
                or not training._metric_report_is_valid(
                    revealed_test, int(sealed_test["samples"])
                )
            ):
                raise ValueError("canonical protected-test reveal binding changed")
            if (
                report != receipt_expected_report
                or publication != expected_publication
            ):
                raise ValueError(
                    "canonical rebuild base selection evidence changed for "
                    f"round {round_index} seed {seed}: "
                    f"report={report == receipt_expected_report}, "
                    f"publication={publication == expected_publication}"
                )
            bindings.append((f"r{round_index}-s{seed}", runtime.resolve()))
    return tuple(bindings)


def v5_runtime_pair(
    v5_campaign: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path]:
    import jacek_replay_train as training
    import jacek_selfsearch_workflow as selfsearch

    v5_campaign = v5_campaign.resolve()
    release_path = v5_campaign / "release-build.json"
    release = load_json(release_path, "v5 release build")
    verify_body_hash(
        release,
        schema="papersoccer.jacek-selfsearch-release-build.v1",
        label="v5 release build",
    )
    commit = release.get("repository", {}).get("head")
    tool_sources = release.get("tool_sources")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or not isinstance(tool_sources, dict)
    ):
        raise ValueError("v5 release source provenance is incomplete")
    source_paths = {
        "trainer": "tools/jacek_replay_train.py",
        "features": "tools/jacek_replay_features.py",
        "corpus": "tools/jacek_replay_corpus.py",
        "workflow": "tools/jacek_selfsearch_workflow.py",
    }
    for name, relative in source_paths.items():
        source = tool_sources.get(name)
        completed = subprocess.run(
            ("git", "show", f"{commit}:{relative}"),
            cwd=pathlib.Path(__file__).resolve().parents[1],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if (
            not isinstance(source, dict)
            or completed.returncode
            or sha256_bytes(completed.stdout) != source.get("sha256")
        ):
            raise ValueError("v5 release source does not match its Git commit")
    trigger = load_json(v5_campaign / "evaluation-trigger.json", "v5 trigger")
    if (
        v5_campaign.name != "selfsearch-auto-20260825-v5"
        or trigger.get("schema")
        != "papersoccer.jacek-selfsearch-evaluation-trigger.v1"
        or trigger.get("repository", {}).get("head") != commit
        or trigger.get("release_build")
        != selfsearch.artifact_snapshot(release_path)
    ):
        raise ValueError("v5 release commit is not bound to the campaign trigger")

    root = v5_campaign.resolve() / "pilot/models"
    canonical_train = tuple(
        pathlib.Path(trigger["anchor_train"][index]["path"])
        for index in range(3)
    )
    expected_stage_configuration = {
        "anchor_rows_per_batch": 128,
        "batch_size": 256,
        "epochs": 50,
        "learning_rate": 0.001,
        "new_rows_per_batch": 128,
        "patience": 8,
        "seed_workers": 2,
        "seeds": [20260901, 20260902, 20260903],
        "weight_decay": 1e-5,
    }
    expected_seed_configuration = {
        "architecture": {
            "dimensions": [training.features.INPUT_COUNT, 192, 32, 1],
            "biases": False,
            "activations": ["square-leaky-0.01", "leaky-relu-0.01", "tanh"],
        },
        "optimizer": {
            "name": "adamw", "epochs": 50, "patience": 8,
            "batch_size": 256, "learning_rate": 0.001,
            "weight_decay": 1e-5, "gradient_norm_clip": 5.0,
        },
        "loss": {"name": "weighted-huber", "delta": 0.25},
        "metrics_batch_size": 4096,
        "augmentation": "reflection rows inherit root game split",
        "batching": {
            "kind": "deterministic-two-stream-cycling-v1",
            "new_rows_per_batch": 128,
            "anchor_rows_per_batch": 128,
            "epoch_length": "new-stream-covered-once-anchor-sampled",
            "row_order": "new-then-anchor",
        },
        "selection_validation": "explicit-common-adjudicator",
    }
    expected_producer = {
        "trainer_sha256": tool_sources["trainer"]["sha256"],
        "features_sha256": tool_sources["features"]["sha256"],
        "corpus_sha256": tool_sources["corpus"]["sha256"],
    }
    selected_paths = []
    for arm, ordinal in (("search", 15), ("rank4", 16)):
        path = root / arm / "jacek_replay_bfm.runtime"
        manifest_path = path.with_suffix(path.suffix + ".json")
        stage_path = (
            v5_campaign / "pilot/receipts"
            / f"{ordinal:02d}-train-{arm}.json"
        )
        stage = load_json(stage_path, f"v5 {arm} training stage")
        manifest = load_json(manifest_path, f"v5 {arm} selected model")
        new_manifest = pathlib.Path(stage.get("inputs", {}).get("new_0", {}).get("path", ""))
        adjudicator = pathlib.Path(
            stage.get("inputs", {}).get("adjudicator", {}).get("path", "")
        )
        expected_inputs_snapshots = {
            "new_0": selfsearch.artifact_snapshot(new_manifest),
            **{
                f"anchor_{index}": selfsearch.artifact_snapshot(anchor)
                for index, anchor in enumerate(canonical_train)
            },
            "adjudicator": selfsearch.artifact_snapshot(adjudicator),
        }
        if (
            stage.get("schema")
            != "papersoccer.jacek-replay-bfm-stage-receipt.v1"
            or stage.get("campaign_id") != "selfsearch-pilot-20260825-v5"
            or stage.get("round") != 0
            or stage.get("ordinal") != ordinal
            or stage.get("stage") != f"train-{arm}"
            or stage.get("configuration") != expected_stage_configuration
            or stage.get("inputs") != expected_inputs_snapshots
            or stage.get("environment", {}).get("release_build")
            != selfsearch.artifact_snapshot(release_path)
            or stage.get("producers", {}).get("trainer", {}).get("sha256")
            != expected_producer["trainer_sha256"]
            or stage.get("producers", {}).get("workflow", {}).get("sha256")
            != tool_sources["workflow"]["sha256"]
            or stage.get("outputs")
            != {
                "runtime": selfsearch.artifact_snapshot(path),
                "manifest": selfsearch.artifact_snapshot(manifest_path),
            }
            or stage.get("result") != manifest
        ):
            raise ValueError("v5 training stage provenance is stale")

        manifest_paths = (new_manifest, *canonical_train, adjudicator)
        source_manifests = [
            load_json(source, f"v5 {arm} source shard")
            for source in manifest_paths
        ]
        validation_dataset = training.combine_shards(
            [training.load_csr_shard(adjudicator)]
        )
        checkpoint_directory = root / arm / "training-seeds"
        first_seed_receipt = load_json(
            checkpoint_directory / "seed-20260901.json",
            f"v5 {arm} first seed receipt",
        )
        receipt_datasets = first_seed_receipt.get("inputs", {}).get(
            "datasets", {}
        )
        receipt_streams = first_seed_receipt.get("inputs", {}).get(
            "training_streams", {}
        )
        train_identity = _validate_opaque_dataset_identity(
            receipt_datasets.get("train"), source_manifests[:4],
            f"v5 {arm} training",
        )
        new_identity = _validate_opaque_dataset_identity(
            receipt_streams.get("new"), source_manifests[:1],
            f"v5 {arm} new stream",
        )
        anchor_identity = _validate_opaque_dataset_identity(
            receipt_streams.get("anchor"), source_manifests[1:4],
            f"v5 {arm} anchor stream",
        )
        expected_inputs = {
            "datasets": {
                "train": train_identity,
                "validation": training._dataset_identity(validation_dataset),
            },
            "feature_schema": training.features.FEATURE_SCHEMA,
            "shards": training._shard_identities(manifest_paths),
            "training_streams": {
                "new": new_identity,
                "anchor": anchor_identity,
            },
        }
        model_training = manifest.get("training")
        runtime_parameters, runtime_report = training.load_runtime(path)
        del runtime_parameters
        expected_architecture = {
            **expected_seed_configuration["architecture"],
            "payload_layout": training.RUNTIME_V1_PAYLOAD_LAYOUT,
        }
        seeds = tuple(expected_stage_configuration["seeds"])
        if (
            manifest.get("schema") != "papersoccer.jacek-replay-bfm-model.v1"
            or manifest.get("status") != "research-candidate-not-game-gated"
            or manifest.get("architecture") != expected_architecture
            or manifest.get("feature_schema") != training.features.FEATURE_SCHEMA
            or manifest.get("source_shards") != source_manifests
            or manifest.get("tool_sha256")
            != {
                "trainer": expected_producer["trainer_sha256"],
                "features": expected_producer["features_sha256"],
                "corpus": expected_producer["corpus_sha256"],
            }
            or manifest.get("runtime")
            != {"path": path.name, **runtime_report}
            or not isinstance(model_training, dict)
            or model_training.get("seeds") != list(seeds)
            or model_training.get("optimizer")
            != expected_seed_configuration["optimizer"]
            or model_training.get("loss") != expected_seed_configuration["loss"]
            or model_training.get("batching")
            != expected_seed_configuration["batching"]
            or model_training.get("selection_validation")
            != {
                "kind": "explicit-common-adjudicator",
                "dataset": expected_inputs["datasets"]["validation"],
            }
            or model_training.get("test_revealed_after_selection") is not False
        ):
            raise ValueError("v5 selected model semantics changed")
        reports = model_training.get("seed_reports")
        publications = model_training.get("seed_checkpoints")
        if (
            not isinstance(reports, list)
            or not isinstance(publications, list)
            or len(reports) != 3
            or len(publications) != 3
        ):
            raise ValueError("v5 seed evidence is incomplete")
        validated_reports = []
        for seed, expected_report, expected_publication in zip(
            seeds, reports, publications, strict=True
        ):
            _parameters, report, publication = _load_historical_seed_checkpoint(
                directory=checkpoint_directory,
                seed=seed,
                expected_configuration=expected_seed_configuration,
                expected_inputs=expected_inputs,
                expected_producer=expected_producer,
                validation_dataset=validation_dataset,
            )
            if report != expected_report or publication != expected_publication:
                raise ValueError("v5 seed evidence changed after recomputation")
            validated_reports.append(report)
        chosen = min(
            validated_reports,
            key=lambda report: (
                *training._selection_key(report["validation"]),
                int(report["seed"]),
            ),
        )
        chosen_seed = int(chosen["seed"])
        chosen_checkpoint = checkpoint_directory / f"seed-{chosen_seed}.runtime"
        if (
            model_training.get("chosen_seed") != chosen_seed
            or path.read_bytes() != chosen_checkpoint.read_bytes()
        ):
            raise ValueError("v5 selected seed or runtime was substituted")
        selected_paths.append(path.resolve())
    search, rank4 = selected_paths
    return search.resolve(), rank4.resolve()


def _campaign_pilot_training_inputs(
    campaign: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    campaign = campaign.resolve()
    summary = load_json(campaign / "final-summary.json", "self-search summary")
    if summary.get("terminal") != "pilot-rejected" or not isinstance(
        summary.get("pilot"), dict
    ):
        raise ValueError("rebuild source campaign is not a completed pilot")
    pilot = summary["pilot"]
    search = pathlib.Path(str(pilot.get("search_new_train_manifest", ""))).resolve()
    rank4 = pathlib.Path(str(pilot.get("rank4_new_train_manifest", ""))).resolve()
    adjudicator_candidates = []
    for path in (campaign / "pilot/shards/adjudicator").glob("*.json"):
        manifest = load_json(path, "campaign adjudicator shard")
        if manifest.get("split") == "validation":
            adjudicator_candidates.append(path.resolve())
    if (
        not search.is_file()
        or not rank4.is_file()
        or len(adjudicator_candidates) != 1
    ):
        raise ValueError("rebuild source campaign shards are incomplete")
    return search, rank4, adjudicator_candidates[0]


def freeze_corpus_from_campaigns(
    *, output_directory: pathlib.Path, canonical_campaign: pathlib.Path,
    v5_campaign: pathlib.Path, v6_campaign: pathlib.Path,
) -> pathlib.Path:
    import jacek_rebuild_corpus as rebuild_corpus
    import jacek_selfsearch_workflow as selfsearch

    canonical = selfsearch._canonical_split_manifests(
        canonical_campaign.resolve()
    )
    v5_search, v5_rank4, v5_adjudicator = _campaign_pilot_training_inputs(
        v5_campaign
    )
    v6_search, v6_rank4, v6_adjudicator = _campaign_pilot_training_inputs(
        v6_campaign
    )
    manifest_path, _manifest = rebuild_corpus.freeze_rebuild_corpus(
        output_directory,
        canonical_train=canonical["train"],
        canonical_validation=canonical["validation"],
        canonical_test=canonical["test"],
        v5_search_train=(v5_search,),
        v6_search_train=(v6_search,),
        v5_rank4_train=(v5_rank4,),
        v6_rank4_train=(v6_rank4,),
        v5_adjudicator_validation=(v5_adjudicator,),
        v6_adjudicator_validation=(v6_adjudicator,),
    )
    rebuild_corpus.validate_rebuild_manifest(manifest_path)
    return manifest_path


def matrix_record(
    *, v5_search: pathlib.Path, v5_rank4: pathlib.Path,
    canonical_bases: Sequence[tuple[str, pathlib.Path]],
) -> dict[str, object]:
    recovery = v5_recovery_specs(
        search_runtime=v5_search, rank4_runtime=v5_rank4
    )
    canonical = canonical_basin_specs(canonical_bases)
    body: dict[str, object] = {
        "schema": REBUILD_MATRIX_SCHEMA,
        "rebuild_id": REBUILD_ID,
        "same_architecture_budget_seconds": SAME_ARCHITECTURE_BUDGET_SECONDS,
        "phases": {
            "v5_recovery": [spec.record() for spec in recovery],
            "canonical_basins": [spec.record() for spec in canonical],
            "scratch_pretraining": {
                "seeds": list(SCRATCH_SEEDS),
                "seed_groups": [list(group) for group in scratch_seed_groups()],
                "optimizer": {
                    "name": "adamw",
                    "batch_size": 256,
                    "learning_rate": 0.001,
                    "weight_decay": 1e-5,
                    "epochs": 50,
                    "patience": 8,
                    "gradient_norm_clip": 5.0,
                },
                "loss": {"name": "weighted-huber", "delta": 0.25},
                "maximum_joint_bases": 3,
            },
            "residual_fallback": {
                "base": "v5-selected-pair",
                "rank": 16,
                "learning_rates": [1e-4, 3e-4, 1e-3],
                "order_seeds": list(ORDER_SEEDS),
            },
        },
        "screening": {
            "offline_shortlist": SHORTLIST_LIMIT,
            "short_pairs": SHORT_SCREEN_PAIRS,
            "full_screen_limit": FULL_SCREEN_LIMIT,
            "full_pairs": FULL_SCREEN_PAIRS,
        },
        "thresholds": {
            "primary_wins": PRIMARY_WIN_THRESHOLD,
            "primary_color_wins": PRIMARY_COLOR_THRESHOLD,
            "external_wins": EXTERNAL_WIN_THRESHOLD,
            "external_color_wins": EXTERNAL_COLOR_THRESHOLD,
            "sign_margin": SIGN_MARGIN,
            "huber_multiplier": HUBER_MULTIPLIER,
            "p99_ms": P99_LIMIT_MS,
            "uncontended_ms": UNCONTENDED_LIMIT_MS,
        },
    }
    return {**body, "body_sha256": sha256_bytes(canonical_json_bytes(body))}


def candidate_spec_from_record(record: Mapping[str, object]) -> CandidateSpec:
    if record.get("schema") != REBUILD_CANDIDATE_SPEC_SCHEMA:
        raise ValueError("candidate specification schema is invalid")
    search = record.get("search_initial_runtime")
    rank4 = record.get("rank4_initial_runtime")
    if not isinstance(search, dict) or not isinstance(rank4, dict):
        raise ValueError("candidate base runtime bindings are missing")
    for snapshot in (search, rank4):
        if (
            not isinstance(snapshot.get("path"), str)
            or artifact_snapshot(pathlib.Path(snapshot["path"])) != snapshot
        ):
            raise ValueError("candidate base runtime binding is stale")
    spec = CandidateSpec(
        candidate_id=str(record.get("candidate_id", "")),
        phase=str(record.get("phase", "")),
        base_id=str(record.get("base_id", "")),
        search_initial_runtime=pathlib.Path(search["path"]),
        rank4_initial_runtime=pathlib.Path(rank4["path"]),
        trainable_layers=str(record.get("trainable_layers", "")),
        learning_rate=float(record.get("learning_rate", float("nan"))),
        selection_policy=str(record.get("selection_policy", "")),
        training_recipe=str(record.get("training_recipe", "")),
        order_seeds=tuple(record.get("order_seeds", ())),
    )
    if spec.record() != dict(record):
        raise ValueError("candidate specification semantics changed")
    return spec


def validate_matrix(record: Mapping[str, object]) -> None:
    body = dict(record)
    claimed = body.pop("body_sha256", None)
    if (
        body.get("schema") != REBUILD_MATRIX_SCHEMA
        or body.get("rebuild_id") != REBUILD_ID
        or claimed != sha256_bytes(canonical_json_bytes(body))
    ):
        raise ValueError("rebuild matrix is stale or corrupt")
    phases = body.get("phases")
    recovery_records = phases.get("v5_recovery") if isinstance(phases, dict) else None
    canonical_records = phases.get("canonical_basins") if isinstance(phases, dict) else None
    if (
        not isinstance(recovery_records, list)
        or len(recovery_records) != 9
        or not isinstance(canonical_records, list)
        or len(canonical_records) != 9
    ):
        raise ValueError("rebuild matrix candidate phases are incomplete")
    first = candidate_spec_from_record(recovery_records[0])
    canonical_bases = []
    for candidate_record in canonical_records:
        spec = candidate_spec_from_record(candidate_record)
        if spec.search_initial_runtime != spec.rank4_initial_runtime:
            raise ValueError("canonical matrix base is not paired")
        canonical_bases.append((spec.base_id, spec.search_initial_runtime))
    expected = matrix_record(
        v5_search=first.search_initial_runtime,
        v5_rank4=first.rank4_initial_runtime,
        canonical_bases=canonical_bases,
    )
    if dict(record) != expected:
        raise ValueError("rebuild matrix semantics changed")


def gate_counts(report: Mapping[str, object]) -> dict[str, object]:
    results = report.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("comparison report has no results")
    wins = sum(
        game.get("winner") == game.get("candidate_player") for game in results
    )
    colors = [
        sum(
            game.get("candidate_player") == color
            and game.get("winner") == color
            for game in results
        )
        for color in (0, 1)
    ]
    samples = [
        float(sample)
        for game in results
        for sample in game.get("candidate_ms", [])
    ]
    if not samples or any(not math.isfinite(sample) or sample < 0 for sample in samples):
        raise ValueError("comparison latency samples are invalid")
    samples.sort()
    p99 = samples[min(len(samples) - 1, math.ceil(0.99 * len(samples)) - 1)]
    return {
        "games": len(results),
        "wins": wins,
        "colors": colors,
        "illegal": sum(bool(game.get("illegal")) for game in results),
        "unfinished": sum(game.get("winner") not in (0, 1) for game in results),
        "p99_ms": p99,
    }


def short_screen_key(
    incumbent: Mapping[str, object], matched: Mapping[str, object],
    runtime_sha256: str,
) -> tuple[float, float, int, str]:
    values = [gate_counts(incumbent), gate_counts(matched)]
    if any(value["illegal"] or value["unfinished"] for value in values):
        raise ValueError("short screen has illegal or unfinished games")
    if any(float(value["p99_ms"]) > P99_LIMIT_MS for value in values):
        raise ValueError("short screen exceeds latency limit")
    worst_rate = min(float(value["wins"]) / int(value["games"]) for value in values)
    worst_color_rate = min(
        min(map(int, value["colors"])) / (int(value["games"]) / 2)
        for value in values
    )
    return (-worst_rate, -worst_color_rate, -sum(int(value["wins"]) for value in values), runtime_sha256)


def development_gate_margins(
    reports: Mapping[str, Mapping[str, object]],
) -> dict[str, float]:
    if set(reports) != {"matched", "incumbent", "rank4", "jacek-nn"}:
        raise ValueError("development report set is incomplete")
    margins: dict[str, float] = {}
    for name, report in reports.items():
        counts = gate_counts(report)
        primary = name in {"matched", "incumbent"}
        wins = PRIMARY_WIN_THRESHOLD if primary else EXTERNAL_WIN_THRESHOLD
        colors = PRIMARY_COLOR_THRESHOLD if primary else EXTERNAL_COLOR_THRESHOLD
        margins[f"{name}_wins"] = float(counts["wins"]) / wins - 1.0
        margins[f"{name}_colors"] = min(map(int, counts["colors"])) / colors - 1.0
        margins[f"{name}_p99"] = P99_LIMIT_MS / float(counts["p99_ms"]) - 1.0
        if counts["illegal"] or counts["unfinished"]:
            margins[f"{name}_legality"] = -1.0
    return margins


def development_passes(
    reports: Mapping[str, Mapping[str, object]],
    *, uncontended_max_ms: float,
) -> bool:
    margins = development_gate_margins(reports)
    return (
        all(value >= 0.0 for value in margins.values())
        and math.isfinite(uncontended_max_ms)
        and uncontended_max_ms < UNCONTENDED_LIMIT_MS
    )


def qualified_candidate_key(
    *, reports: Mapping[str, Mapping[str, object]],
    uncontended_max_ms: float, canonical_huber: float,
    runtime_sha256: str,
) -> tuple[float, float, str]:
    if not development_passes(reports, uncontended_max_ms=uncontended_max_ms):
        raise ValueError("candidate does not pass development gates")
    margins = development_gate_margins(reports)
    margins["uncontended"] = UNCONTENDED_LIMIT_MS / uncontended_max_ms - 1.0
    if not math.isfinite(canonical_huber) or canonical_huber < 0.0:
        raise ValueError("candidate canonical Huber is invalid")
    return (-min(margins.values()), canonical_huber, runtime_sha256)


def _official_qualified_key(
    candidate: Mapping[str, object], full: Mapping[str, object],
    canonical: Mapping[str, object],
) -> tuple[float, float, str]:
    decision = full.get("decision")
    if not isinstance(decision, dict) or decision.get("eligible_for_full") is not True:
        raise ValueError("candidate does not pass the full development gate")
    counts = decision.get("counts")
    if not isinstance(counts, dict) or set(counts) != {
        "matched", "incumbent", "rank4", "jacek-nn"
    }:
        raise ValueError("candidate development counts are incomplete")
    margins = []
    for name, value in counts.items():
        if not isinstance(value, dict):
            raise ValueError("candidate development count is malformed")
        primary = name in {"matched", "incumbent"}
        win_threshold = (
            PRIMARY_WIN_THRESHOLD if primary else EXTERNAL_WIN_THRESHOLD
        )
        color_threshold = (
            PRIMARY_COLOR_THRESHOLD if primary else EXTERNAL_COLOR_THRESHOLD
        )
        margins.extend(
            (
                int(value["wins"]) / win_threshold - 1.0,
                min(map(int, value["colors"])) / color_threshold - 1.0,
            )
        )
    candidate_p99 = float(decision["candidate_p99_ms"])
    uncontended = float(decision["uncontended_max_ms"])
    margins.extend(
        (
            math.inf if candidate_p99 == 0.0 else P99_LIMIT_MS / candidate_p99 - 1.0,
            math.inf if uncontended == 0.0 else UNCONTENDED_LIMIT_MS / uncontended - 1.0,
        )
    )
    huber = float(canonical["weighted_huber"])
    if any(math.isnan(value) for value in margins) or not math.isfinite(huber):
        raise ValueError("candidate development ranking metrics are invalid")
    return (
        -min(margins),
        huber,
        sha256_file(_candidate_runtime_path(candidate, "search")),
    )


def budget_allows_new_work(started_at_unix: float, *, now: float | None = None) -> bool:
    now = time.time() if now is None else now
    if (
        not math.isfinite(started_at_unix)
        or not math.isfinite(now)
        or now < started_at_unix
    ):
        raise ValueError("rebuild budget timestamps are invalid")
    return now - started_at_unix < SAME_ARCHITECTURE_BUDGET_SECONDS


def ladder_phase_order_is_valid(
    observed_order: Sequence[str], winning_phase: str
) -> bool:
    allowed = ("v5-recovery", "canonical-basins", "scratch-joint", "residual")
    if winning_phase not in allowed or not observed_order:
        return False
    winning_index = allowed.index(winning_phase)
    if winning_phase == "residual":
        prior = tuple(observed_order[:-1])
        return bool(prior) and prior == allowed[: len(prior)] and tuple(
            observed_order[-1:]
        ) == ("residual",)
    return tuple(observed_order) == allowed[: winning_index + 1]


def _run(arguments: Sequence[str]) -> None:
    completed = subprocess.run(
        list(arguments), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.decode("utf-8", "replace"))


def _comparison_report(
    *, comparison: pathlib.Path, model: pathlib.Path,
    control_model: pathlib.Path | None, bank: pathlib.Path,
    classification: str, opponent: str, pairs: int, output: pathlib.Path,
) -> dict:
    import jacek_selfsearch_workflow as selfsearch

    if classification not in {"development", "final"}:
        raise ValueError("rebuild comparison classification is invalid")
    panel = selfsearch.Panel("rebuild", opponent, control_model)
    command = selfsearch._comparison_command(
        comparison=comparison,
        model=model,
        bank=bank,
        output=output,
        panel=panel,
        pairs=pairs,
        time_ms=20,
        classification=classification,
    )
    source_identities = selfsearch._source_identities(
        pathlib.Path(__file__).resolve().parents[1]
    )
    def validate(path: pathlib.Path) -> dict:
        return selfsearch._validate_panel_report(
            path=path,
            comparison=comparison,
            model=model,
            bank=bank,
            panel=panel,
            pairs=pairs,
            pair_offset=0,
            time_ms=20,
            classification=classification,
            source_identities=source_identities,
        )

    if not output.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=output.parent, prefix=f".{output.name}.", delete=False
        ) as handle:
            temporary = pathlib.Path(handle.name)
        temporary.unlink()
        try:
            command[command.index(str(output))] = str(temporary)
            _run(command)
            validate(temporary)
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
    return validate(output)


def run_short_screen(
    *, comparison: pathlib.Path, candidate: pathlib.Path,
    matched: pathlib.Path, incumbent: pathlib.Path, bank: pathlib.Path,
    output_directory: pathlib.Path,
) -> dict[str, object]:
    reports = {
        "matched": _comparison_report(
            comparison=comparison, model=candidate, control_model=matched,
            bank=bank, classification="development", opponent="jacek-replay",
            pairs=SHORT_SCREEN_PAIRS, output=output_directory / "matched.json",
        ),
        "incumbent": _comparison_report(
            comparison=comparison, model=candidate, control_model=incumbent,
            bank=bank, classification="development", opponent="jacek-replay",
            pairs=SHORT_SCREEN_PAIRS, output=output_directory / "incumbent.json",
        ),
    }
    counts = {name: gate_counts(report) for name, report in reports.items()}
    rejection_reasons = []
    for name in ("matched", "incumbent"):
        if counts[name]["illegal"]:
            rejection_reasons.append(f"{name}:illegal")
        if counts[name]["unfinished"]:
            rejection_reasons.append(f"{name}:unfinished")
        if float(counts[name]["p99_ms"]) > P99_LIMIT_MS:
            rejection_reasons.append(f"{name}:slow")
    operational = not rejection_reasons
    key = (
        short_screen_key(
            reports["incumbent"], reports["matched"], sha256_file(candidate)
        )
        if operational
        else None
    )
    return {
        "candidate": artifact_snapshot(candidate),
        "matched": artifact_snapshot(matched),
        "bank": artifact_snapshot(bank),
        "reports": {
            name: artifact_snapshot(output_directory / f"{name}.json")
            for name in reports
        },
        "counts": counts,
        "operational": operational,
        "rejection_reasons": rejection_reasons,
        "ranking_key": list(key) if key is not None else None,
    }


def run_full_screen(
    *, comparison: pathlib.Path, candidate: pathlib.Path,
    matched: pathlib.Path, incumbent: pathlib.Path, bank: pathlib.Path,
    output_directory: pathlib.Path, classification: str,
    canonical_candidate: Mapping[str, float],
    canonical_incumbent: Mapping[str, float],
) -> dict[str, object]:
    import jacek_selfsearch_workflow as selfsearch

    panels = {
        "matched": ("jacek-replay", matched),
        "incumbent": ("jacek-replay", incumbent),
        "rank4": ("rank4", None),
        "jacek-nn": ("jacek-nn", None),
    }
    reports = {
        name: _comparison_report(
            comparison=comparison, model=candidate, control_model=control,
            bank=bank, classification=classification, opponent=opponent,
            pairs=FULL_SCREEN_PAIRS, output=output_directory / f"{name}.json",
        )
        for name, (opponent, control) in panels.items()
    }
    latency_path = output_directory / "latency.json"
    source_identities = selfsearch._source_identities(
        pathlib.Path(__file__).resolve().parents[1]
    )
    latency = selfsearch.run_latency_audit(
        comparison=comparison, model=candidate, bank=bank,
        output=latency_path, classification=classification,
        source_identities=source_identities,
    )
    decision = selfsearch.pilot_decision(
        matched_report=output_directory / "matched.json",
        incumbent_report=output_directory / "incumbent.json",
        rank4_report=output_directory / "rank4.json",
        jacek_nn_report=output_directory / "jacek-nn.json",
        anchor_candidate=canonical_candidate,
        anchor_incumbent=canonical_incumbent,
        uncontended_max_ms=float(latency["candidate_max_ms"]),
    )
    return {
        "candidate": artifact_snapshot(candidate),
        "matched": artifact_snapshot(matched),
        "bank": artifact_snapshot(bank),
        "classification": classification,
        "reports": {
            name: artifact_snapshot(output_directory / f"{name}.json")
            for name in reports
        },
        "latency": artifact_snapshot(latency_path),
        "decision": decision,
    }


def _recovery_seed_task(
    *, prepared: object, trainable_layers: str, learning_rate: float,
    selection_policy: str, seed: int, output_directory: pathlib.Path,
    resume: bool, training_recipe: str,
) -> dict[str, object]:
    import jacek_replay_recovery as recovery

    if training_recipe == "v6-joint":
        _selected, report = recovery.run_v6_joint(
            prepared,
            recovery.V6JointConfiguration(seed),
            output_directory,
            resume=resume,
        )
        runtime_name = recovery.V6_JOINT_RUNTIME_NAME
        report_name = recovery.V6_JOINT_REPORT_NAME
        receipt_name = recovery.V6_JOINT_RECEIPT_NAME
    elif training_recipe == "recovery":
        _selected, report = recovery.run_recovery(
            prepared,
            recovery.RecoveryConfiguration(
                trainable_layers=trainable_layers,
                learning_rate=learning_rate,
                seed=seed,
                selection_policy=selection_policy,
            ),
            output_directory,
            resume=resume,
        )
        runtime_name = recovery.RUNTIME_NAME
        report_name = recovery.REPORT_NAME
        receipt_name = recovery.RECEIPT_NAME
    else:
        raise ValueError("unknown rebuild training recipe")
    runtime = output_directory / runtime_name
    return {
        "seed": seed,
        "runtime": artifact_snapshot(runtime),
        "report": artifact_snapshot(output_directory / report_name),
        "receipt": artifact_snapshot(output_directory / receipt_name),
        "result": report["result"],
    }


def _best_recovery_seed(records: Sequence[Mapping[str, object]]) -> dict[str, object] | None:
    import jacek_replay_recovery as recovery

    eligible = [
        record for record in records
        if isinstance(record.get("result"), dict)
        and record["result"].get("eligible") is True
    ]
    if not eligible:
        return None
    return dict(
        min(
            eligible,
            key=lambda record: recovery.selection_key(
                record["result"]["selection"],
                record["runtime"]["sha256"],
            ),
        )
    )


def _recovery_launch_receipt(
    *, spec: CandidateSpec, corpus_manifest: pathlib.Path,
    incumbent_runtime: pathlib.Path, deadline_unix: float,
    output_directory: pathlib.Path, allow_create: bool,
) -> dict[str, object] | None:
    path = output_directory / "launches" / f"{spec.candidate_id}.json"
    if path.exists():
        receipt = load_json(path, "recovery configuration launch receipt")
        verify_body_hash(
            receipt,
            schema="papersoccer.jacek-replay-rebuild-config-launch.v1",
            label="recovery configuration launch receipt",
        )
        launched = receipt.get("launched_at_unix")
        if (
            set(receipt) != {
                "schema", "rebuild_id", "candidate_id", "specification",
                "corpus", "incumbent", "deadline_unix",
                "launched_at_unix", "body_sha256",
            }
            or receipt.get("rebuild_id") != REBUILD_ID
            or receipt.get("candidate_id") != spec.candidate_id
            or receipt.get("specification") != spec.record()
            or receipt.get("corpus") != artifact_snapshot(corpus_manifest)
            or receipt.get("incumbent") != artifact_snapshot(incumbent_runtime)
            or receipt.get("deadline_unix") != deadline_unix
            or isinstance(launched, bool)
            or not isinstance(launched, (int, float))
            or not math.isfinite(float(launched))
            or float(launched)
            < deadline_unix - SAME_ARCHITECTURE_BUDGET_SECONDS
            or float(launched) > deadline_unix
        ):
            raise ValueError("recovery configuration launch semantics changed")
        return receipt
    if not allow_create or time.time() >= deadline_unix:
        return None
    launched_at = time.time()
    if launched_at >= deadline_unix:
        return None
    body: dict[str, object] = {
        "schema": "papersoccer.jacek-replay-rebuild-config-launch.v1",
        "rebuild_id": REBUILD_ID,
        "candidate_id": spec.candidate_id,
        "specification": spec.record(),
        "corpus": artifact_snapshot(corpus_manifest),
        "incumbent": artifact_snapshot(incumbent_runtime),
        "deadline_unix": deadline_unix,
        "launched_at_unix": launched_at,
    }
    receipt = {**body, "body_sha256": sha256_bytes(canonical_json_bytes(body))}
    atomic_write(path, canonical_json_bytes(receipt, pretty=True))
    return receipt


def run_recovery_phase(
    *, specs: Sequence[CandidateSpec], corpus_manifest: pathlib.Path,
    incumbent_runtime: pathlib.Path, output_directory: pathlib.Path,
    resume: bool, workers: int = 10, deadline_unix: float | None = None,
) -> list[dict[str, object]]:
    """Run paired, three-seed recovery configs and publish offline survivors."""

    import jacek_rebuild_corpus as rebuild_corpus
    import jacek_replay_recovery as recovery

    corpus = load_frozen_rebuild_corpus(corpus_manifest)
    if not specs:
        raise ValueError("recovery phase has no candidate specifications")
    if workers <= 0:
        raise ValueError("recovery phase workers must be positive")
    if (
        deadline_unix is None
        or not math.isfinite(deadline_unix)
        or deadline_unix <= 0.0
    ):
        raise ValueError("recovery phase requires a frozen deadline")
    output_directory = output_directory.resolve()
    datasets_by_arm = {
        arm: recovery.prepare_recovery_datasets(
            new_manifests=corpus.training_manifest_paths(arm),
            anchor_manifests=corpus.anchor_manifest_paths(arm),
            selection_manifests=corpus.validation_manifest_paths(arm),
            retention_manifests=corpus.retention_validation_manifest_paths(arm),
        )
        for arm in ("search", "rank4")
    }
    prepared_by_spec_arm: dict[tuple[str, str], object] = {}
    for spec in specs:
        spec.validate()
        for arm, initial in (
            ("search", spec.search_initial_runtime),
            ("rank4", spec.rank4_initial_runtime),
        ):
            prepared_by_spec_arm[(spec.candidate_id, arm)] = (
                recovery.bind_recovery_runtimes(
                    datasets_by_arm[arm],
                    initial_runtime=initial,
                    retention_reference_runtime=incumbent_runtime,
                    new_reference_runtime=initial,
                )
            )
    def execute_spec(spec: CandidateSpec) -> tuple[str, dict[str, list[dict]]]:
        completed = {"search": [], "rank4": []}
        for arm in ("search", "rank4"):
            for seed in spec.order_seeds:
                path = (
                    output_directory / "runs" / spec.candidate_id / arm
                    / f"seed-{seed}"
                )
                completed[arm].append(
                    _recovery_seed_task(
                        prepared=prepared_by_spec_arm[(spec.candidate_id, arm)],
                        trainable_layers=spec.trainable_layers,
                        learning_rate=spec.learning_rate,
                        selection_policy=spec.selection_policy,
                        seed=seed,
                        output_directory=path,
                        resume=resume,
                        training_recipe=spec.training_recipe,
                    )
                )
        return spec.candidate_id, completed

    records: dict[tuple[str, str], list[dict]] = {
        (spec.candidate_id, arm): []
        for spec in specs for arm in ("search", "rank4")
    }
    launch_snapshots: dict[str, dict[str, object]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        remaining = iter(specs)
        futures: dict[concurrent.futures.Future, CandidateSpec] = {}

        def submit_next() -> bool:
            for spec in remaining:
                receipt = _recovery_launch_receipt(
                    spec=spec,
                    corpus_manifest=corpus_manifest,
                    incumbent_runtime=incumbent_runtime,
                    deadline_unix=deadline_unix,
                    output_directory=output_directory,
                    allow_create=True,
                )
                if receipt is None:
                    continue
                launch_path = (
                    output_directory / "launches" / f"{spec.candidate_id}.json"
                )
                launch_snapshots[spec.candidate_id] = artifact_snapshot(
                    launch_path
                )
                futures[executor.submit(execute_spec, spec)] = spec
                return True
            return False

        while len(futures) < workers and submit_next():
            pass
        while futures:
            done, _pending = concurrent.futures.wait(
                futures, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                futures.pop(future)
                candidate_id, completed = future.result()
                for arm in ("search", "rank4"):
                    records[(candidate_id, arm)].extend(completed[arm])
                submit_next()
    candidates = []
    for spec in specs:
        search_records = sorted(
            records[(spec.candidate_id, "search")], key=lambda row: row["seed"]
        )
        rank4_records = sorted(
            records[(spec.candidate_id, "rank4")], key=lambda row: row["seed"]
        )
        complete = (
            [record["seed"] for record in search_records]
            == sorted(spec.order_seeds)
            and [record["seed"] for record in rank4_records]
            == sorted(spec.order_seeds)
        )
        selected_search = _best_recovery_seed(search_records)
        selected_rank4 = _best_recovery_seed(rank4_records)
        body: dict[str, object] = {
            "schema": REBUILD_CANDIDATE_SCHEMA,
            "candidate_id": spec.candidate_id,
            "specification": spec.record(),
            "corpus": artifact_snapshot(corpus_manifest),
            "incumbent": artifact_snapshot(incumbent_runtime),
            "launch": launch_snapshots.get(spec.candidate_id),
            "search_seed_runs": search_records,
            "rank4_seed_runs": rank4_records,
            "offline_eligible": (
                complete and selected_search is not None and selected_rank4 is not None
            ),
            "selected_search": selected_search,
            "selected_rank4": selected_rank4,
        }
        candidate = {
            **body, "body_sha256": sha256_bytes(canonical_json_bytes(body))
        }
        candidate_path = output_directory / "candidates" / f"{spec.candidate_id}.json"
        if candidate_path.exists():
            if load_json(candidate_path, "rebuild candidate") != candidate:
                raise ValueError("existing rebuild candidate record is stale")
        else:
            atomic_write(candidate_path, canonical_json_bytes(candidate, pretty=True))
        candidates.append(candidate)
    return candidates


def run_residual_phase(
    *, corpus_manifest: pathlib.Path, incumbent_runtime: pathlib.Path,
    v5_search_runtime: pathlib.Path, v5_rank4_runtime: pathlib.Path,
    output_directory: pathlib.Path, resume: bool, workers: int = 10,
) -> list[dict[str, object]]:
    """Run the sole fixed rank-16 adapter fallback matrix."""

    import jacek_rebuild_corpus as rebuild_corpus
    import jacek_replay_recovery as recovery

    corpus = load_frozen_rebuild_corpus(corpus_manifest)
    datasets = {
        arm: recovery.prepare_recovery_datasets(
            new_manifests=corpus.training_manifest_paths(arm),
            anchor_manifests=corpus.anchor_manifest_paths(arm),
            selection_manifests=corpus.validation_manifest_paths(arm),
            retention_manifests=corpus.retention_validation_manifest_paths(arm),
        )
        for arm in ("search", "rank4")
    }
    prepared = {
        "search": recovery.bind_recovery_runtimes(
            datasets["search"], initial_runtime=v5_search_runtime,
            retention_reference_runtime=incumbent_runtime,
            new_reference_runtime=v5_search_runtime,
        ),
        "rank4": recovery.bind_recovery_runtimes(
            datasets["rank4"], initial_runtime=v5_rank4_runtime,
            retention_reference_runtime=incumbent_runtime,
            new_reference_runtime=v5_rank4_runtime,
        ),
    }
    tasks = [
        (learning_rate, arm, seed)
        for learning_rate in (1e-4, 3e-4, 1e-3)
        for arm in ("search", "rank4")
        for seed in ORDER_SEEDS
    ]

    def execute(item: tuple[float, str, int]) -> tuple[float, str, dict]:
        learning_rate, arm, seed = item
        rate_id = {1e-4: "1e4", 3e-4: "3e4", 1e-3: "1e3"}[learning_rate]
        path = (
            output_directory / "runs" / f"residual-lr{rate_id}" / arm
            / f"seed-{seed}"
        )
        _selected, report = recovery.run_residual_recovery(
            prepared[arm],
            recovery.ResidualRecoveryConfiguration(learning_rate, seed),
            path,
            resume=resume,
        )
        return learning_rate, arm, {
            "seed": seed,
            "runtime": artifact_snapshot(path / recovery.RESIDUAL_RUNTIME_NAME),
            "report": artifact_snapshot(path / recovery.RESIDUAL_REPORT_NAME),
            "receipt": artifact_snapshot(path / recovery.RESIDUAL_RECEIPT_NAME),
            "result": report["result"],
        }

    records: dict[tuple[float, str], list[dict]] = {
        (learning_rate, arm): []
        for learning_rate in (1e-4, 3e-4, 1e-3)
        for arm in ("search", "rank4")
    }
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(execute, item) for item in tasks]
        for future in concurrent.futures.as_completed(futures):
            learning_rate, arm, result = future.result()
            records[(learning_rate, arm)].append(result)
    candidates = []
    for learning_rate in (1e-4, 3e-4, 1e-3):
        search_records = sorted(
            records[(learning_rate, "search")], key=lambda row: row["seed"]
        )
        rank4_records = sorted(
            records[(learning_rate, "rank4")], key=lambda row: row["seed"]
        )
        selected_search = _best_recovery_seed(search_records)
        selected_rank4 = _best_recovery_seed(rank4_records)
        candidate_id = {
            1e-4: "residual-lr1e4",
            3e-4: "residual-lr3e4",
            1e-3: "residual-lr1e3",
        }[learning_rate]
        body: dict[str, object] = {
            "schema": REBUILD_CANDIDATE_SCHEMA,
            "candidate_id": candidate_id,
            "phase": "residual",
            "learning_rate": learning_rate,
            "rank": 16,
            "corpus": artifact_snapshot(corpus_manifest),
            "incumbent": artifact_snapshot(incumbent_runtime),
            "search_seed_runs": search_records,
            "rank4_seed_runs": rank4_records,
            "offline_eligible": (
                selected_search is not None and selected_rank4 is not None
            ),
            "selected_search": selected_search,
            "selected_rank4": selected_rank4,
        }
        candidate = {
            **body, "body_sha256": sha256_bytes(canonical_json_bytes(body))
        }
        path = output_directory / "candidates" / f"{candidate_id}.json"
        if path.exists():
            if load_json(path, "residual candidate") != candidate:
                raise ValueError("existing residual candidate record is stale")
        else:
            atomic_write(path, canonical_json_bytes(candidate, pretty=True))
        candidates.append(candidate)
    return candidates


def _canonical_pretrain_seed(
    *, datasets: Mapping[str, object], seed: int,
    corpus_manifest: pathlib.Path, output_directory: pathlib.Path,
    resume: bool,
) -> dict[str, object]:
    import jacek_replay_train as training

    runtime_path = output_directory / f"seed-{seed}.runtime"
    report_path = output_directory / f"seed-{seed}.json"
    config = {
        "seed": seed,
        "architecture": [training.features.INPUT_COUNT, 192, 32, 1],
        "optimizer": {
            "name": "adamw", "batch_size": 256, "learning_rate": 0.001,
            "weight_decay": 1e-5, "epochs": 50, "patience": 8,
            "gradient_norm_clip": 5.0,
        },
        "loss": {"name": "weighted-huber", "delta": 0.25},
        "protected_test_supplied": False,
    }
    if runtime_path.exists() or report_path.exists():
        if not resume or not runtime_path.is_file() or not report_path.is_file():
            raise ValueError("canonical pretraining output is partial or unresumable")
        report = load_json(report_path, "canonical pretraining report")
        receipt_body = dict(report)
        receipt_hash = receipt_body.pop("body_sha256", None)
        parameters, runtime = training.load_runtime(runtime_path)
        if (
            receipt_hash != sha256_bytes(canonical_json_bytes(receipt_body))
            or
            report.get("schema")
            != "papersoccer.jacek-replay-rebuild-pretraining.v1"
            or report.get("configuration") != config
            or report.get("corpus") != artifact_snapshot(corpus_manifest)
            or report.get("producer")
            != artifact_snapshot(pathlib.Path(training.__file__))
            or report.get("runtime") != runtime
            or report.get("training_report", {}).get("validation")
            != training.metrics(parameters, datasets["validation"])
        ):
            raise ValueError("canonical pretraining receipt is stale or corrupt")
        replayed_parameters, replayed_training_report = training.train_seed(
            datasets,
            seed,
            epochs=50,
            patience=8,
            batch_size=256,
            learning_rate=0.001,
            weight_decay=1e-5,
        )
        persisted_payload, persisted_runtime = training.runtime_bytes(parameters)
        replayed_payload, replayed_runtime = training.runtime_bytes(
            replayed_parameters
        )
        if (
            persisted_payload != replayed_payload
            or persisted_runtime != replayed_runtime
            or report.get("training_report") != replayed_training_report
        ):
            raise ValueError(
                "canonical pretraining full replay disagrees with persisted evidence"
            )
        return report
    parameters, training_report = training.train_seed(
        datasets,
        seed,
        epochs=50,
        patience=8,
        batch_size=256,
        learning_rate=0.001,
        weight_decay=1e-5,
    )
    runtime = training.export_runtime(runtime_path, parameters)
    body: dict[str, object] = {
        "schema": "papersoccer.jacek-replay-rebuild-pretraining.v1",
        "configuration": config,
        "corpus": artifact_snapshot(corpus_manifest),
        "producer": artifact_snapshot(pathlib.Path(training.__file__)),
        "runtime": runtime,
        "training_report": training_report,
    }
    report = {**body, "body_sha256": sha256_bytes(canonical_json_bytes(body))}
    atomic_write(report_path, canonical_json_bytes(report, pretty=True))
    return report


def _scratch_launch_receipt(
    *, corpus_manifest: pathlib.Path, comparison: pathlib.Path,
    incumbent_runtime: pathlib.Path, development_bank: pathlib.Path,
    deadline_unix: float, output_directory: pathlib.Path,
    allow_create: bool,
) -> dict[str, object] | None:
    path = output_directory / "pretraining-launch.json"
    common: dict[str, object] = {
        "schema": "papersoccer.jacek-replay-rebuild-scratch-launch.v1",
        "rebuild_id": REBUILD_ID,
        "seeds": list(SCRATCH_SEEDS),
        "corpus": artifact_snapshot(corpus_manifest),
        "comparison": artifact_snapshot(comparison),
        "incumbent": artifact_snapshot(incumbent_runtime),
        "development_bank": artifact_snapshot(development_bank),
        "deadline_unix": deadline_unix,
    }
    if path.exists():
        receipt = load_json(path, "scratch pretraining launch receipt")
        verify_body_hash(
            receipt,
            schema="papersoccer.jacek-replay-rebuild-scratch-launch.v1",
            label="scratch pretraining launch receipt",
        )
        launched = receipt.get("launched_at_unix")
        if (
            set(receipt) != {*common, "launched_at_unix", "body_sha256"}
            or any(receipt.get(key) != value for key, value in common.items())
            or isinstance(launched, bool)
            or not isinstance(launched, (int, float))
            or not math.isfinite(float(launched))
            or float(launched)
            < deadline_unix - SAME_ARCHITECTURE_BUDGET_SECONDS
            or float(launched) > deadline_unix
        ):
            raise ValueError("scratch pretraining launch semantics changed")
        return receipt
    if not allow_create or time.time() >= deadline_unix:
        return None
    launched_at = time.time()
    if launched_at >= deadline_unix:
        return None
    body = {**common, "launched_at_unix": launched_at}
    receipt = {**body, "body_sha256": sha256_bytes(canonical_json_bytes(body))}
    atomic_write(path, canonical_json_bytes(receipt, pretty=True))
    return receipt


def run_scratch_pretraining(
    *, corpus_manifest: pathlib.Path, comparison: pathlib.Path,
    incumbent_runtime: pathlib.Path, development_bank: pathlib.Path,
    output_directory: pathlib.Path, resume: bool,
    deadline_unix: float, workers: int = 6,
) -> tuple[list[CandidateSpec], dict[str, object]]:
    """Pretrain six bases without ever loading the protected test split."""

    import jacek_rebuild_corpus as rebuild_corpus
    import jacek_replay_train as training

    corpus = load_frozen_rebuild_corpus(corpus_manifest)
    launch = _scratch_launch_receipt(
        corpus_manifest=corpus_manifest,
        comparison=comparison,
        incumbent_runtime=incumbent_runtime,
        development_bank=development_bank,
        deadline_unix=deadline_unix,
        output_directory=output_directory,
        allow_create=True,
    )
    if launch is None:
        raise ValueError("scratch pretraining was not launched before the deadline")
    train_paths = corpus.anchor_manifest_paths("search")
    validation_paths = corpus.retention_validation_manifest_paths("search")
    loaded = [
        training.load_csr_shard(path)
        for path in (*train_paths, *validation_paths)
    ]
    training.validate_shard_collection(loaded)
    train_shards = loaded[: len(train_paths)]
    validation_shards = loaded[len(train_paths) :]
    if (
        any(shard.split != "train" for shard in train_shards)
        or any(shard.split != "validation" for shard in validation_shards)
    ):
        raise ValueError("canonical pretraining split roles are invalid")
    datasets = {
        "train": training.combine_shards(train_shards),
        "validation": training.combine_shards(validation_shards),
    }
    del loaded
    seed_root = output_directory / "pretraining"
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _canonical_pretrain_seed,
                datasets=datasets,
                seed=seed,
                corpus_manifest=corpus_manifest,
                output_directory=seed_root,
                resume=resume,
            ): seed
            for seed in SCRATCH_SEEDS
        }
        reports = []
        for future in concurrent.futures.as_completed(futures):
            reports.append(future.result())
    reports.sort(key=lambda report: int(report["configuration"]["seed"]))
    base_records = []
    for report in reports:
        seed = int(report["configuration"]["seed"])
        runtime = seed_root / f"seed-{seed}.runtime"
        game_path = output_directory / "base-screen" / f"seed-{seed}.json"
        game = _comparison_report(
            comparison=comparison,
            model=runtime,
            control_model=incumbent_runtime,
            bank=development_bank,
            classification="development",
            opponent="jacek-replay",
            pairs=SHORT_SCREEN_PAIRS,
            output=game_path,
        )
        counts = gate_counts(game)
        rejection_reasons = []
        if counts["illegal"]:
            rejection_reasons.append("illegal")
        if counts["unfinished"]:
            rejection_reasons.append("unfinished")
        if float(counts["p99_ms"]) > P99_LIMIT_MS:
            rejection_reasons.append("slow")
        base_records.append(
            {
                "seed": seed,
                "runtime": artifact_snapshot(runtime),
                "pretraining_report": artifact_snapshot(
                    seed_root / f"seed-{seed}.json"
                ),
                "game_report": artifact_snapshot(game_path),
                "game_counts": counts,
                "validation": report["training_report"]["validation"],
                "operational": not rejection_reasons,
                "rejection_reasons": rejection_reasons,
            }
        )
    ranked = sorted(
        (record for record in base_records if record["operational"] is True),
        key=lambda record: (
            -int(record["game_counts"]["wins"]),
            -min(map(int, record["game_counts"]["colors"])),
            float(record["validation"]["weighted_huber"]),
            record["runtime"]["sha256"],
        )
    )
    selected = ranked[:3]
    if len(selected) != 3:
        raise ValueError("fewer than three operational scratch bases survived")
    specs = [
        CandidateSpec(
            candidate_id=f"scratch-s{record['seed']}",
            phase="scratch",
            base_id=f"scratch-s{record['seed']}",
            search_initial_runtime=pathlib.Path(record["runtime"]["path"]),
            rank4_initial_runtime=pathlib.Path(record["runtime"]["path"]),
            trainable_layers="all",
            learning_rate=JOINT_LEARNING_RATE,
            selection_policy="epoch-zero-improvement",
            training_recipe="v6-joint",
        )
        for record in selected
    ]
    lineage_body: dict[str, object] = {
        "schema": "papersoccer.jacek-replay-rebuild-scratch-bases.v1",
        "launch": artifact_snapshot(output_directory / "pretraining-launch.json"),
        "corpus": artifact_snapshot(corpus_manifest),
        "development_bank": artifact_snapshot(development_bank),
        "incumbent": artifact_snapshot(incumbent_runtime),
        "pretraining_reports": [
            artifact_snapshot(seed_root / f"seed-{seed}.json")
            for seed in SCRATCH_SEEDS
        ],
        "base_records": base_records,
        "ranked_eligible_seeds": [record["seed"] for record in ranked],
        "selected_specs": [spec.record() for spec in specs],
        "selection": (
            "wins-desc,worst-color-desc,canonical-huber-asc,runtime-hash"
        ),
        "protected_test_supplied": False,
    }
    lineage = {
        **lineage_body,
        "body_sha256": sha256_bytes(canonical_json_bytes(lineage_body)),
    }
    return specs, lineage


def _candidate_runtime_path(candidate: Mapping[str, object], arm: str) -> pathlib.Path:
    selected = candidate.get(f"selected_{arm}")
    runtime = selected.get("runtime") if isinstance(selected, dict) else None
    if not isinstance(runtime, dict) or not isinstance(runtime.get("path"), str):
        raise ValueError(f"candidate has no selected {arm} runtime")
    path = pathlib.Path(runtime["path"])
    if artifact_snapshot(path) != runtime:
        raise ValueError(f"candidate selected {arm} runtime is stale")
    return path


def _candidate_recovery_report(
    candidate: Mapping[str, object], arm: str,
) -> dict:
    selected = candidate.get(f"selected_{arm}")
    report = selected.get("report") if isinstance(selected, dict) else None
    if not isinstance(report, dict) or not isinstance(report.get("path"), str):
        raise ValueError(f"candidate has no selected {arm} report")
    path = pathlib.Path(report["path"])
    if artifact_snapshot(path) != report:
        raise ValueError(f"candidate selected {arm} report is stale")
    return load_json(path, f"candidate {arm} recovery report")


def screen_candidate_phase(
    *, phase: str, candidates: Sequence[Mapping[str, object]],
    comparison: pathlib.Path, incumbent_runtime: pathlib.Path,
    development_bank: pathlib.Path, output_directory: pathlib.Path,
    workers: int = 4, phase_inputs: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Apply six-candidate short screening and two-candidate full gates."""

    candidate_records = [dict(candidate) for candidate in candidates]
    candidate_ids = [str(candidate.get("candidate_id", "")) for candidate in candidates]
    if (
        not candidate_records
        or any(not candidate_id for candidate_id in candidate_ids)
        or len(candidate_ids) != len(set(candidate_ids))
    ):
        raise ValueError("phase candidate roster is empty or duplicated")
    offline = []
    for candidate in candidate_records:
        if candidate.get("offline_eligible") is not True:
            continue
        report = _candidate_recovery_report(candidate, "search")
        result = report.get("result", {})
        runtime = _candidate_runtime_path(candidate, "search")
        offline.append(
            (
                (
                    float(result["selection"]["weighted_huber"]),
                    -float(result["selection"]["sign_accuracy"]),
                    -float(result["selection"]["correlation"]),
                    sha256_file(runtime),
                ),
                candidate,
            )
        )
    offline.sort(key=lambda item: item[0])
    shortlist = [candidate for _key, candidate in offline[:SHORTLIST_LIMIT]]

    def short(candidate: Mapping[str, object]) -> tuple[Mapping, dict]:
        candidate_id = str(candidate["candidate_id"])
        screen = run_short_screen(
            comparison=comparison,
            candidate=_candidate_runtime_path(candidate, "search"),
            matched=_candidate_runtime_path(candidate, "rank4"),
            incumbent=incumbent_runtime,
            bank=development_bank,
            output_directory=output_directory / "short" / candidate_id,
        )
        return candidate, screen

    short_results: dict[str, tuple[Mapping, dict]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(short, candidate): str(candidate["candidate_id"])
            for candidate in shortlist
        }
        for future in concurrent.futures.as_completed(futures):
            candidate, record = future.result()
            short_results[futures[future]] = (candidate, record)
    short_records = [
        {
            "candidate_id": str(candidate["candidate_id"]),
            "screen": short_results[str(candidate["candidate_id"])][1],
        }
        for candidate in shortlist
    ]
    short_rejections = [
        {
            "candidate_id": record["candidate_id"],
            "reasons": record["screen"]["rejection_reasons"],
        }
        for record in short_records
        if record["screen"]["operational"] is not True
    ]
    ranked_short = sorted(
        (
            (
                tuple(record["screen"]["ranking_key"]),
                short_results[record["candidate_id"]][0],
                record["screen"],
            )
            for record in short_records
            if record["screen"]["operational"] is True
        ),
        key=lambda item: item[0],
    )
    finalists = ranked_short[:FULL_SCREEN_LIMIT]
    full_records = []
    qualified = []
    for _short_key, candidate, short_record in finalists:
        candidate_id = str(candidate["candidate_id"])
        recovery_report = _candidate_recovery_report(candidate, "search")
        canonical_candidate = recovery_report["result"]["retention"]
        canonical_incumbent = recovery_report["references"][
            "retention_reference"
        ]["retention"]
        full = run_full_screen(
            comparison=comparison,
            candidate=_candidate_runtime_path(candidate, "search"),
            matched=_candidate_runtime_path(candidate, "rank4"),
            incumbent=incumbent_runtime,
            bank=development_bank,
            output_directory=output_directory / "full" / candidate_id,
            classification="development",
            canonical_candidate=canonical_candidate,
            canonical_incumbent=canonical_incumbent,
        )
        record = {
            "candidate_id": candidate_id,
            "short": short_record,
            "full": full,
        }
        full_records.append(record)
        if full["decision"]["eligible_for_full"] is True:
            qualified.append((candidate, full, canonical_candidate))
    qualified.sort(
        key=lambda item: _official_qualified_key(item[0], item[1], item[2])
    )
    selected = qualified[0][0] if qualified else None
    body: dict[str, object] = {
        "schema": "papersoccer.jacek-replay-rebuild-phase-screen.v1",
        "phase": phase,
        "phase_inputs": dict(phase_inputs or {}),
        "candidate_records": candidate_records,
        "offline_eligible": len(offline),
        "offline_candidate_records": [candidate for _key, candidate in offline],
        "shortlisted": [str(candidate["candidate_id"]) for candidate in shortlist],
        "short_records": short_records,
        "short_ranked": [
            str(candidate["candidate_id"])
            for _key, candidate, _record in ranked_short
        ],
        "short_rejections": short_rejections,
        "full_candidate_records": [candidate for _key, candidate, _short in finalists],
        "full_records": full_records,
        "qualified": [str(item[0]["candidate_id"]) for item in qualified],
        "selected_candidate_id": (
            str(selected["candidate_id"]) if selected is not None else None
        ),
    }
    result = {**body, "body_sha256": sha256_bytes(canonical_json_bytes(body))}
    output = output_directory / "phase-screen.json"
    if output.exists():
        if load_json(output, "phase screen") != result:
            raise ValueError("existing phase screen is stale")
    else:
        atomic_write(output, canonical_json_bytes(result, pretty=True))
    return {"record": result, "selected": selected}


def _status(path: pathlib.Path, phase: str, **fields: object) -> None:
    atomic_write(
        path,
        canonical_json_bytes(
            {
                "schema": REBUILD_STATUS_SCHEMA,
                "rebuild_id": REBUILD_ID,
                "phase": phase,
                "updated_at_unix": time.time(),
                **fields,
            },
            pretty=True,
        ),
    )


def _freeze_selected_candidate(
    *, candidate: Mapping[str, object], phase_screen: Mapping[str, object],
    ladder_phase_screens: Sequence[Mapping[str, object]],
    inputs_manifest: pathlib.Path, output: pathlib.Path,
) -> dict[str, object]:
    search = _candidate_runtime_path(candidate, "search")
    matched = _candidate_runtime_path(candidate, "rank4")
    body: dict[str, object] = {
        "schema": "papersoccer.jacek-replay-rebuild-selected-candidate.v1",
        "rebuild_id": REBUILD_ID,
        "candidate_id": candidate["candidate_id"],
        "inputs": artifact_snapshot(inputs_manifest),
        "candidate": candidate,
        "phase_screen": phase_screen,
        "ladder_phase_screens": list(ladder_phase_screens),
        "selected_runtime": artifact_snapshot(search),
        "matched_runtime": artifact_snapshot(matched),
        "selection_policy": (
            "development-gates-then-worst-normalized-margin-canonical-huber-hash"
        ),
        "protected_test_opened": False,
        "sealed_final_bank_opened": False,
        "blind_holdout_labels_opened": False,
    }
    record = {**body, "body_sha256": sha256_bytes(canonical_json_bytes(body))}
    if output.exists():
        if load_json(output, "selected rebuild candidate") != record:
            raise ValueError("selected rebuild candidate receipt is stale")
    else:
        atomic_write(output, canonical_json_bytes(record, pretty=True))
    return record


def run_ladder(
    *, inputs_manifest: pathlib.Path, output_directory: pathlib.Path,
    resume: bool, training_workers: int = 10,
) -> dict[str, object]:
    """Run the evidence-gated ladder through the sole residual fallback."""

    inputs = validate_rebuild_inputs(inputs_manifest)
    output_directory = output_directory.resolve()
    status_path = output_directory / "status.json"
    corpus_path = pathlib.Path(inputs["corpus"]["path"])
    matrix_path = pathlib.Path(inputs["matrix"]["path"])
    banks_path = pathlib.Path(inputs["opening_banks"]["path"])
    comparison = pathlib.Path(inputs["comparison"]["path"])
    incumbent = pathlib.Path(inputs["incumbent_runtime"]["path"])
    matrix = load_json(matrix_path, "rebuild matrix")
    validate_matrix(matrix)
    banks = load_json(banks_path, "rebuild opening banks")
    validate_opening_banks(banks)
    development_bank = pathlib.Path(banks["development"]["artifact"]["path"])
    same_architecture_deadline = float(
        inputs["same_architecture_deadline_unix"]
    )

    def same_architecture_work_is_open() -> bool:
        return time.time() < same_architecture_deadline

    def recovery_phase_was_started(name: str) -> bool:
        root = output_directory / name
        return any((root / "launches").glob("*.json")) or (
            root / "screening/phase-screen.json"
        ).exists()

    def scratch_pretraining_was_started() -> bool:
        return (
            output_directory / "scratch/pretraining-launch.json"
        ).exists() or (output_directory / "scratch/base-screen.json").exists()

    phase_results = []

    def train_and_screen(
        name: str, specs: Sequence[CandidateSpec],
        phase_inputs: Mapping[str, object] | None = None,
    ) -> dict:
        _status(status_path, f"training-{name}")
        candidates = run_recovery_phase(
            specs=specs,
            corpus_manifest=corpus_path,
            incumbent_runtime=incumbent,
            output_directory=output_directory / name,
            resume=resume,
            workers=training_workers,
            deadline_unix=same_architecture_deadline,
        )
        _status(
            status_path,
            f"screening-{name}",
            offline_eligible=sum(
                candidate.get("offline_eligible") is True
                for candidate in candidates
            ),
        )
        screen = screen_candidate_phase(
            phase=name,
            candidates=candidates,
            comparison=comparison,
            incumbent_runtime=incumbent,
            development_bank=development_bank,
            output_directory=output_directory / name / "screening",
            phase_inputs=phase_inputs,
        )
        phase_results.append(screen["record"])
        return screen

    recovery_specs = tuple(
        candidate_spec_from_record(record)
        for record in matrix["phases"]["v5_recovery"]
    )
    screen: dict[str, object] = {"selected": None, "record": {}}
    if same_architecture_work_is_open() or (
        resume and recovery_phase_was_started("v5-recovery")
    ):
        screen = train_and_screen("v5-recovery", recovery_specs)
    if screen["selected"] is None and (
        same_architecture_work_is_open()
        or (resume and recovery_phase_was_started("canonical-basins"))
    ):
        canonical_specs = tuple(
            candidate_spec_from_record(record)
            for record in matrix["phases"]["canonical_basins"]
        )
        screen = train_and_screen("canonical-basins", canonical_specs)
    if screen["selected"] is None and (
        same_architecture_work_is_open()
        or (resume and scratch_pretraining_was_started())
    ):
        _status(status_path, "scratch-pretraining")
        scratch_specs, scratch_lineage = run_scratch_pretraining(
            corpus_manifest=corpus_path,
            comparison=comparison,
            incumbent_runtime=incumbent,
            development_bank=development_bank,
            output_directory=output_directory / "scratch",
            resume=resume,
            deadline_unix=same_architecture_deadline,
        )
        scratch_lineage_path = output_directory / "scratch/base-screen.json"
        scratch_payload = canonical_json_bytes(scratch_lineage, pretty=True)
        if scratch_lineage_path.exists():
            if scratch_lineage_path.read_bytes() != scratch_payload:
                raise ValueError("existing scratch base lineage is stale")
        else:
            atomic_write(scratch_lineage_path, scratch_payload)
        if same_architecture_work_is_open() or (
            resume and recovery_phase_was_started("scratch-joint")
        ):
            screen = train_and_screen(
                "scratch-joint", scratch_specs,
                phase_inputs={
                    "scratch_base_lineage": artifact_snapshot(
                        output_directory / "scratch/base-screen.json"
                    )
                },
            )
    if screen["selected"] is None:
        if not any(
            candidate.get("launch") is not None
            for phase_record in phase_results
            for candidate in phase_record.get("candidate_records", [])
            if isinstance(candidate, dict)
        ):
            raise ValueError(
                "same-architecture ladder never launched before its deadline"
            )
        _status(status_path, "training-residual-fallback")
        v5_spec = recovery_specs[0]
        residual_candidates = run_residual_phase(
            corpus_manifest=corpus_path,
            incumbent_runtime=incumbent,
            v5_search_runtime=v5_spec.search_initial_runtime,
            v5_rank4_runtime=v5_spec.rank4_initial_runtime,
            output_directory=output_directory / "residual",
            resume=resume,
            workers=training_workers,
        )
        screen = screen_candidate_phase(
            phase="residual",
            candidates=residual_candidates,
            comparison=comparison,
            incumbent_runtime=incumbent,
            development_bank=development_bank,
            output_directory=output_directory / "residual/screening",
        )
        phase_results.append(screen["record"])
    if screen["selected"] is None:
        summary: dict[str, object] = {
            "schema": REBUILD_DECISION_SCHEMA,
            "terminal": "no-development-qualified-candidate",
            "same_architecture_budget_seconds": SAME_ARCHITECTURE_BUDGET_SECONDS,
            "phases": phase_results,
            "residual_fallback_exhausted": True,
            "next_scope": "action-ranking-or-wider-network-requires-new-plan",
        }
        atomic_write(
            output_directory / "final-summary.json",
            canonical_json_bytes(summary, pretty=True),
        )
        _status(status_path, "complete-no-candidate")
        return summary
    selected = _freeze_selected_candidate(
        candidate=screen["selected"],
        phase_screen=screen["record"],
        ladder_phase_screens=phase_results,
        inputs_manifest=inputs_manifest,
        output=output_directory / "selected-candidate.json",
    )
    _status(status_path, "running-final-qualification")
    qualification = qualify_selected_candidate(
        inputs_manifest=inputs_manifest,
        selected_receipt=output_directory / "selected-candidate.json",
        output_directory=output_directory / "qualification",
        workers=training_workers,
    )
    summary = {
        "schema": REBUILD_DECISION_SCHEMA,
        "terminal": (
            "qualified-local-research-incumbent"
            if qualification["pass"] is True
            else "final-qualification-rejected"
        ),
        "phases": phase_results,
        "selected_candidate": selected,
        "qualification": qualification,
    }
    atomic_write(
        output_directory / "final-summary.json",
        canonical_json_bytes(summary, pretty=True),
    )
    _status(
        status_path,
        "complete-qualified" if qualification["pass"] else "complete-rejected",
    )
    return summary


def label_blind_holdout(
    *, teacher: pathlib.Path, positions: pathlib.Path,
    teacher_source_sha256: str, output_directory: pathlib.Path,
    workers: int = 10, chunk_rows: int = 25,
) -> pathlib.Path:
    """Produce strict 400k fixed-work labels with per-chunk receipts."""

    import jacek_replay_corpus as corpus
    import jacek_selfsearch_workflow as selfsearch

    lines = positions.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != selfsearch._position_rows(positions)[0]:
        raise ValueError("blind holdout position TSV is invalid")
    rows = lines[1:]
    if not rows or workers <= 0 or chunk_rows <= 0:
        raise ValueError("blind holdout labeling configuration is invalid")
    chunks = [rows[start : start + chunk_rows] for start in range(0, len(rows), chunk_rows)]
    input_root = output_directory / "inputs"
    output_root = output_directory / "outputs"
    receipt_root = output_directory / "receipts"
    for index, chunk in enumerate(chunks):
        path = input_root / f"chunk-{index:04d}.tsv"
        payload = (lines[0] + "\n" + "\n".join(chunk) + "\n").encode()
        if path.exists():
            if path.read_bytes() != payload:
                raise ValueError("blind holdout label chunk input is stale")
        else:
            atomic_write(path, payload)

    def execute(index: int) -> pathlib.Path:
        source = input_root / f"chunk-{index:04d}.tsv"
        target = output_root / f"chunk-{index:04d}.jsonl"
        receipt = receipt_root / f"chunk-{index:04d}.json"
        expected = {
            "schema": "papersoccer.jacek-rebuild-holdout-label-chunk.v1",
            "rebuild_id": REBUILD_ID,
            "chunk": index,
            "configuration": {"nodes": 400_000, "time_ms": 0},
            "teacher": artifact_snapshot(teacher),
            "teacher_source_sha256": teacher_source_sha256,
            "positions": artifact_snapshot(source),
        }
        if receipt.exists():
            saved = load_json(receipt, "blind holdout label receipt")
            if (
                set(saved) != {*expected, "output"}
                or any(saved.get(key) != value for key, value in expected.items())
                or saved.get("output") != artifact_snapshot(target)
            ):
                raise ValueError("blind holdout label receipt is stale")
            selfsearch._validate_label_output(
                output=target, positions=source,
                schema=corpus.RANK4_TEACHER_SCHEMA,
                campaign_id=REBUILD_ID, nodes=400_000,
                model_sha256=None, source_sha256=teacher_source_sha256,
            )
            return target
        if target.exists():
            raise ValueError("blind holdout label output has no receipt")
        target.parent.mkdir(parents=True, exist_ok=True)
        receipt.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=f".{target.name}.", delete=False
        ) as handle:
            temporary = pathlib.Path(handle.name)
        try:
            with source.open("rb") as input_file, temporary.open("wb") as output_file:
                completed = subprocess.run(
                    (
                        str(teacher), "--campaign-id", REBUILD_ID,
                        "--nodes", "400000", "--time-ms", "0",
                    ),
                    stdin=input_file,
                    stdout=output_file,
                    stderr=subprocess.PIPE,
                    check=False,
                )
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr.decode("utf-8", "replace"))
            selfsearch._validate_label_output(
                output=temporary, positions=source,
                schema=corpus.RANK4_TEACHER_SCHEMA,
                campaign_id=REBUILD_ID, nodes=400_000,
                model_sha256=None, source_sha256=teacher_source_sha256,
            )
            os.replace(temporary, target)
            saved = {**expected, "output": artifact_snapshot(target)}
            atomic_write(receipt, canonical_json_bytes(saved, pretty=True))
            return target
        finally:
            temporary.unlink(missing_ok=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        outputs = list(executor.map(execute, range(len(chunks))))
    labels_path = output_directory / "labels.jsonl"
    payload = b"".join(path.read_bytes() for path in outputs)
    if labels_path.exists():
        if labels_path.read_bytes() != payload:
            raise ValueError("blind holdout merged labels are stale")
    else:
        atomic_write(labels_path, payload)
    expected_ids = [line.split("\t", 1)[0] for line in rows]
    labels = selfsearch.load_labels(labels_path, corpus.RANK4_TEACHER_SCHEMA)
    if list(labels) != expected_ids:
        raise ValueError("blind holdout labels do not preserve exact coverage")
    return labels_path


def _scratch_pretraining_datasets(corpus: object) -> dict[str, object]:
    import jacek_replay_train as training

    train_paths = corpus.anchor_manifest_paths("search")
    validation_paths = corpus.retention_validation_manifest_paths("search")
    loaded = [
        training.load_csr_shard(path)
        for path in (*train_paths, *validation_paths)
    ]
    training.validate_shard_collection(loaded)
    train_shards = loaded[: len(train_paths)]
    validation_shards = loaded[len(train_paths) :]
    if (
        any(shard.split != "train" for shard in train_shards)
        or any(shard.split != "validation" for shard in validation_shards)
    ):
        raise ValueError("scratch pretraining split roles are invalid")
    return {
        "train": training.combine_shards(train_shards),
        "validation": training.combine_shards(validation_shards),
    }


def validate_scratch_base_lineage(
    *, snapshot: Mapping[str, object], inputs: Mapping[str, object],
    corpus: object, comparison: pathlib.Path, incumbent: pathlib.Path,
    development_bank: pathlib.Path,
) -> list[dict[str, object]]:
    """Recompute all six scratch-base metrics, games, and the top-three cut."""

    if (
        not isinstance(snapshot.get("path"), str)
        or artifact_snapshot(pathlib.Path(snapshot["path"])) != snapshot
    ):
        raise ValueError("scratch base lineage binding is stale")
    lineage = load_json(pathlib.Path(snapshot["path"]), "scratch base lineage")
    verify_body_hash(
        lineage,
        schema="papersoccer.jacek-replay-rebuild-scratch-bases.v1",
        label="scratch base lineage",
    )
    expected_fields = {
        "schema", "launch", "corpus", "development_bank", "incumbent",
        "pretraining_reports", "base_records", "ranked_eligible_seeds",
        "selected_specs", "selection", "protected_test_supplied",
        "body_sha256",
    }
    if set(lineage) != expected_fields:
        raise ValueError("scratch base lineage shape changed")
    corpus_path = pathlib.Path(inputs["corpus"]["path"])
    launch_snapshot = lineage.get("launch")
    if (
        not isinstance(launch_snapshot, dict)
        or not isinstance(launch_snapshot.get("path"), str)
        or artifact_snapshot(pathlib.Path(launch_snapshot["path"]))
        != launch_snapshot
    ):
        raise ValueError("scratch pretraining launch binding is stale")
    launch_path = pathlib.Path(launch_snapshot["path"])
    if launch_path.name != "pretraining-launch.json":
        raise ValueError("scratch pretraining launch layout changed")
    if _scratch_launch_receipt(
        corpus_manifest=corpus_path,
        comparison=comparison,
        incumbent_runtime=incumbent,
        development_bank=development_bank,
        deadline_unix=float(inputs["same_architecture_deadline_unix"]),
        output_directory=launch_path.parent,
        allow_create=False,
    ) is None:
        raise ValueError("scratch pretraining has no on-time launch receipt")
    if (
        lineage.get("corpus") != artifact_snapshot(corpus_path)
        or lineage.get("development_bank") != artifact_snapshot(development_bank)
        or lineage.get("incumbent") != artifact_snapshot(incumbent)
        or lineage.get("selection")
        != "wins-desc,worst-color-desc,canonical-huber-asc,runtime-hash"
        or lineage.get("protected_test_supplied") is not False
    ):
        raise ValueError("scratch base lineage policy changed")
    report_snapshots = lineage.get("pretraining_reports")
    base_records = lineage.get("base_records")
    if (
        not isinstance(report_snapshots, list)
        or len(report_snapshots) != len(SCRATCH_SEEDS)
        or not isinstance(base_records, list)
        or len(base_records) != len(SCRATCH_SEEDS)
    ):
        raise ValueError("scratch base evidence is incomplete")
    datasets = _scratch_pretraining_datasets(corpus)
    expected_base_records = []
    for index, seed in enumerate(SCRATCH_SEEDS):
        report_snapshot = report_snapshots[index]
        if (
            not isinstance(report_snapshot, dict)
            or not isinstance(report_snapshot.get("path"), str)
            or artifact_snapshot(pathlib.Path(report_snapshot["path"]))
            != report_snapshot
        ):
            raise ValueError("scratch pretraining report binding is stale")
        report_path = pathlib.Path(report_snapshot["path"])
        if report_path.name != f"seed-{seed}.json":
            raise ValueError("scratch pretraining seed order changed")
        report = _canonical_pretrain_seed(
            datasets=datasets,
            seed=seed,
            corpus_manifest=corpus_path,
            output_directory=report_path.parent,
            resume=True,
        )
        runtime = report_path.parent / f"seed-{seed}.runtime"
        game_snapshot = base_records[index].get("game_report") if isinstance(
            base_records[index], dict
        ) else None
        if (
            not isinstance(game_snapshot, dict)
            or not isinstance(game_snapshot.get("path"), str)
            or pathlib.Path(game_snapshot["path"]).name != f"seed-{seed}.json"
        ):
            raise ValueError("scratch development-game binding is missing")
        game = _comparison_report(
            comparison=comparison,
            model=runtime,
            control_model=incumbent,
            bank=development_bank,
            classification="development",
            opponent="jacek-replay",
            pairs=SHORT_SCREEN_PAIRS,
            output=pathlib.Path(game_snapshot["path"]),
        )
        if artifact_snapshot(pathlib.Path(game_snapshot["path"])) != game_snapshot:
            raise ValueError("scratch development-game binding is stale")
        counts = gate_counts(game)
        reasons = []
        if counts["illegal"]:
            reasons.append("illegal")
        if counts["unfinished"]:
            reasons.append("unfinished")
        if float(counts["p99_ms"]) > P99_LIMIT_MS:
            reasons.append("slow")
        expected_base_records.append(
            {
                "seed": seed,
                "runtime": artifact_snapshot(runtime),
                "pretraining_report": report_snapshot,
                "game_report": game_snapshot,
                "game_counts": counts,
                "validation": report["training_report"]["validation"],
                "operational": not reasons,
                "rejection_reasons": reasons,
            }
        )
    if base_records != expected_base_records:
        raise ValueError("scratch base evidence changed after recomputation")
    ranked = sorted(
        (record for record in expected_base_records if record["operational"]),
        key=lambda record: (
            -int(record["game_counts"]["wins"]),
            -min(map(int, record["game_counts"]["colors"])),
            float(record["validation"]["weighted_huber"]),
            record["runtime"]["sha256"],
        ),
    )
    if len(ranked) < 3:
        raise ValueError("fewer than three operational scratch bases survived")
    expected_specs = [
        CandidateSpec(
            candidate_id=f"scratch-s{record['seed']}",
            phase="scratch",
            base_id=f"scratch-s{record['seed']}",
            search_initial_runtime=pathlib.Path(record["runtime"]["path"]),
            rank4_initial_runtime=pathlib.Path(record["runtime"]["path"]),
            trainable_layers="all",
            learning_rate=JOINT_LEARNING_RATE,
            selection_policy="epoch-zero-improvement",
            training_recipe="v6-joint",
        ).record()
        for record in ranked[:3]
    ]
    if (
        lineage.get("ranked_eligible_seeds")
        != [record["seed"] for record in ranked]
        or lineage.get("selected_specs") != expected_specs
    ):
        raise ValueError("scratch top-three selection changed")
    return expected_specs


def _validate_phase_candidate_record(
    *, candidate: Mapping[str, object], expected_spec: Mapping[str, object] | None,
    residual_learning_rate: float | None, inputs: Mapping[str, object],
    corpus: object, incumbent: pathlib.Path,
    prepared_cache: dict[tuple[str, str], object],
) -> None:
    """Replay one complete paired candidate, including every shared seed."""

    import jacek_replay_recovery as recovery

    verify_body_hash(
        candidate, schema=REBUILD_CANDIDATE_SCHEMA, label="rebuild candidate"
    )
    common_fields = {
        "schema", "candidate_id", "corpus", "incumbent",
        "search_seed_runs", "rank4_seed_runs", "offline_eligible",
        "selected_search", "selected_rank4", "body_sha256",
    }
    residual = expected_spec is None
    expected_fields = common_fields | (
        {"phase", "learning_rate", "rank"}
        if residual else {"specification", "launch"}
    )
    if set(candidate) != expected_fields:
        raise ValueError("rebuild candidate record shape changed")
    if (
        candidate.get("corpus") != inputs.get("corpus")
        or candidate.get("incumbent") != inputs.get("incumbent_runtime")
    ):
        raise ValueError("rebuild candidate frozen input binding changed")
    if residual:
        rate_ids = {1e-4: "1e4", 3e-4: "3e4", 1e-3: "1e3"}
        if (
            residual_learning_rate not in rate_ids
            or candidate.get("phase") != "residual"
            or candidate.get("rank") != 16
            or candidate.get("learning_rate") != residual_learning_rate
            or candidate.get("candidate_id")
            != f"residual-lr{rate_ids[residual_learning_rate]}"
        ):
            raise ValueError("residual candidate differs from the frozen matrix")
        matrix = load_json(pathlib.Path(inputs["matrix"]["path"]), "rebuild matrix")
        base_spec = candidate_spec_from_record(matrix["phases"]["v5_recovery"][0])
        initial = {
            "search": base_spec.search_initial_runtime,
            "rank4": base_spec.rank4_initial_runtime,
        }
        expected_seeds = ORDER_SEEDS
        training_recipe = "residual"
    else:
        if candidate.get("specification") != expected_spec:
            raise ValueError("candidate specification differs from the phase roster")
        spec = candidate_spec_from_record(expected_spec)
        if candidate.get("candidate_id") != spec.candidate_id:
            raise ValueError("candidate identifier differs from its specification")
        initial = {
            "search": spec.search_initial_runtime,
            "rank4": spec.rank4_initial_runtime,
        }
        expected_seeds = spec.order_seeds
        training_recipe = spec.training_recipe
        launch_snapshot = candidate.get("launch")
        if launch_snapshot is None:
            launched = False
        else:
            if (
                not isinstance(launch_snapshot, dict)
                or not isinstance(launch_snapshot.get("path"), str)
                or artifact_snapshot(pathlib.Path(launch_snapshot["path"]))
                != launch_snapshot
            ):
                raise ValueError("candidate launch receipt binding is stale")
            launch_path = pathlib.Path(launch_snapshot["path"])
            if (
                launch_path.parent.name != "launches"
                or launch_path.name != f"{spec.candidate_id}.json"
            ):
                raise ValueError("candidate launch receipt layout changed")
            _recovery_launch_receipt(
                spec=spec,
                corpus_manifest=pathlib.Path(inputs["corpus"]["path"]),
                incumbent_runtime=incumbent,
                deadline_unix=float(inputs["same_architecture_deadline_unix"]),
                output_directory=launch_path.parent.parent,
                allow_create=False,
            )
            launched = True

    complete_by_arm = []
    for arm in ("search", "rank4"):
        records = candidate.get(f"{arm}_seed_runs")
        if not isinstance(records, list):
            raise ValueError("candidate seed-run transcript is malformed")
        observed_seeds = [
            int(record.get("seed", -1)) if isinstance(record, dict) else -1
            for record in records
        ]
        if observed_seeds not in ([], sorted(expected_seeds)):
            raise ValueError("candidate seed-run set differs from the frozen matrix")
        complete_by_arm.append(bool(records))
        if not records:
            continue
        cache_key = (arm, str(initial[arm].resolve()))
        prepared = prepared_cache.get(cache_key)
        if prepared is None:
            datasets = recovery.prepare_recovery_datasets(
                new_manifests=corpus.training_manifest_paths(arm),
                anchor_manifests=corpus.anchor_manifest_paths(arm),
                selection_manifests=corpus.validation_manifest_paths(arm),
                retention_manifests=corpus.retention_validation_manifest_paths(arm),
            )
            prepared = recovery.bind_recovery_runtimes(
                datasets,
                initial_runtime=initial[arm],
                retention_reference_runtime=incumbent,
                new_reference_runtime=initial[arm],
            )
            prepared_cache[cache_key] = prepared
        for record in records:
            if not isinstance(record, dict) or set(record) != {
                "seed", "runtime", "report", "receipt", "result"
            }:
                raise ValueError("candidate seed-run record shape changed")
            for artifact_name in ("runtime", "report", "receipt"):
                snapshot = record.get(artifact_name)
                if (
                    not isinstance(snapshot, dict)
                    or not isinstance(snapshot.get("path"), str)
                    or artifact_snapshot(pathlib.Path(snapshot["path"])) != snapshot
                ):
                    raise ValueError("candidate seed artifact binding is stale")
            directory = pathlib.Path(record["receipt"]["path"]).parent
            seed = int(record["seed"])
            if training_recipe == "residual":
                _parameters, report = recovery.load_residual_recovery_result(
                    prepared,
                    recovery.ResidualRecoveryConfiguration(
                        float(residual_learning_rate), seed
                    ),
                    directory,
                )
            elif training_recipe == "v6-joint":
                _parameters, report = recovery.load_v6_joint_result(
                    prepared, recovery.V6JointConfiguration(seed), directory
                )
            else:
                _parameters, report = recovery.load_recovery_result(
                    prepared,
                    recovery.RecoveryConfiguration(
                        spec.trainable_layers,
                        spec.learning_rate,
                        seed,
                        spec.selection_policy,
                    ),
                    directory,
                )
            if report.get("result") != record.get("result"):
                raise ValueError("candidate seed result changed after full replay")
        if candidate.get(f"selected_{arm}") != _best_recovery_seed(records):
            raise ValueError("candidate selected seed is not deterministic")
    if complete_by_arm not in ([False, False], [True, True]):
        raise ValueError("candidate paired arms have different coverage")
    if not residual and complete_by_arm[0] is not launched:
        raise ValueError("candidate work is not backed by an on-time launch receipt")
    expected_eligible = bool(
        complete_by_arm[0]
        and candidate.get("selected_search") is not None
        and candidate.get("selected_rank4") is not None
    )
    if candidate.get("offline_eligible") is not expected_eligible:
        raise ValueError("candidate offline eligibility changed")


def validate_phase_screen(
    *, phase_screen: Mapping[str, object], inputs: Mapping[str, object]
) -> Mapping[str, object] | None:
    """Recompute the complete phase roster, shortlist, gates, and winner."""

    import jacek_rebuild_corpus as rebuild_corpus
    import jacek_replay_recovery as recovery

    verify_body_hash(
        phase_screen,
        schema="papersoccer.jacek-replay-rebuild-phase-screen.v1",
        label="rebuild phase screen",
    )
    expected_fields = {
        "schema", "phase", "phase_inputs", "candidate_records",
        "offline_eligible", "offline_candidate_records", "shortlisted",
        "short_records", "short_ranked", "short_rejections",
        "full_candidate_records", "full_records", "qualified",
        "selected_candidate_id", "body_sha256",
    }
    if set(phase_screen) != expected_fields:
        raise ValueError("rebuild phase-screen shape changed")
    phase = phase_screen.get("phase")
    matrix = load_json(pathlib.Path(inputs["matrix"]["path"]), "rebuild matrix")
    validate_matrix(matrix)
    corpus = load_frozen_rebuild_corpus(pathlib.Path(inputs["corpus"]["path"]))
    incumbent = pathlib.Path(inputs["incumbent_runtime"]["path"])
    comparison = pathlib.Path(inputs["comparison"]["path"])
    banks = load_json(pathlib.Path(inputs["opening_banks"]["path"]), "rebuild banks")
    validate_opening_banks(banks)
    development_bank = pathlib.Path(banks["development"]["artifact"]["path"])
    phase_inputs = phase_screen.get("phase_inputs")
    if not isinstance(phase_inputs, dict):
        raise ValueError("rebuild phase inputs are malformed")
    residual_rates: list[float | None]
    if phase == "v5-recovery":
        if phase_inputs:
            raise ValueError("v5 recovery phase has unexpected inputs")
        expected_specs = list(matrix["phases"]["v5_recovery"])
        residual_rates = [None] * len(expected_specs)
    elif phase == "canonical-basins":
        if phase_inputs:
            raise ValueError("canonical basin phase has unexpected inputs")
        expected_specs = list(matrix["phases"]["canonical_basins"])
        residual_rates = [None] * len(expected_specs)
    elif phase == "scratch-joint":
        if set(phase_inputs) != {"scratch_base_lineage"} or not isinstance(
            phase_inputs["scratch_base_lineage"], dict
        ):
            raise ValueError("scratch phase lineage input is missing")
        expected_specs = validate_scratch_base_lineage(
            snapshot=phase_inputs["scratch_base_lineage"],
            inputs=inputs,
            corpus=corpus,
            comparison=comparison,
            incumbent=incumbent,
            development_bank=development_bank,
        )
        residual_rates = [None] * len(expected_specs)
    elif phase == "residual":
        if phase_inputs:
            raise ValueError("residual phase has unexpected inputs")
        residual_rates = [
            float(value)
            for value in matrix["phases"]["residual_fallback"]["learning_rates"]
        ]
        expected_specs = [None] * len(residual_rates)
    else:
        raise ValueError("rebuild phase name is outside the frozen ladder")

    candidates = phase_screen.get("candidate_records")
    if not isinstance(candidates, list) or len(candidates) != len(expected_specs):
        raise ValueError("rebuild phase candidate roster is incomplete")
    expected_ids = [
        (
            str(spec["candidate_id"])
            if isinstance(spec, dict)
            else {
                1e-4: "residual-lr1e4",
                3e-4: "residual-lr3e4",
                1e-3: "residual-lr1e3",
            }[float(rate)]
        )
        for spec, rate in zip(expected_specs, residual_rates, strict=True)
    ]
    if [
        str(candidate.get("candidate_id", ""))
        if isinstance(candidate, dict) else ""
        for candidate in candidates
    ] != expected_ids:
        raise ValueError("rebuild phase candidate roster order changed")
    prepared_cache: dict[tuple[str, str], object] = {}
    for candidate, expected_spec, residual_rate in zip(
        candidates, expected_specs, residual_rates, strict=True
    ):
        if not isinstance(candidate, dict):
            raise ValueError("rebuild phase candidate is malformed")
        _validate_phase_candidate_record(
            candidate=candidate,
            expected_spec=expected_spec,
            residual_learning_rate=residual_rate,
            inputs=inputs,
            corpus=corpus,
            incumbent=incumbent,
            prepared_cache=prepared_cache,
        )
    complete_flags = [
        bool(candidate["search_seed_runs"]) for candidate in candidates
    ]
    if complete_flags != sorted(complete_flags, reverse=True):
        raise ValueError("deadline-limited candidate launch order changed")

    offline = []
    for candidate in candidates:
        if candidate["offline_eligible"] is not True:
            continue
        report = _candidate_recovery_report(candidate, "search")
        result = report["result"]
        offline.append(
            (
                recovery.selection_key(
                    result["selection"],
                    sha256_file(_candidate_runtime_path(candidate, "search")),
                ),
                candidate,
            )
        )
    offline.sort(key=lambda item: item[0])
    expected_offline_records = [candidate for _key, candidate in offline]
    if (
        phase_screen.get("offline_eligible") != len(offline)
        or phase_screen.get("offline_candidate_records")
        != expected_offline_records
    ):
        raise ValueError("rebuild offline shortlist ordering changed")
    shortlist = expected_offline_records[:SHORTLIST_LIMIT]
    shortlist_ids = [str(candidate["candidate_id"]) for candidate in shortlist]
    if phase_screen.get("shortlisted") != shortlist_ids:
        raise ValueError("rebuild short-screen roster changed")

    short_records = phase_screen.get("short_records")
    if not isinstance(short_records, list) or len(short_records) != len(shortlist):
        raise ValueError("rebuild short-screen evidence is incomplete")
    expected_short_records = []
    ranked_short = []
    expected_rejections = []
    for candidate, saved in zip(shortlist, short_records, strict=True):
        candidate_id = str(candidate["candidate_id"])
        if not isinstance(saved, dict) or set(saved) != {"candidate_id", "screen"}:
            raise ValueError("rebuild short-screen record shape changed")
        screen = saved.get("screen")
        reports = screen.get("reports") if isinstance(screen, dict) else None
        if (
            saved.get("candidate_id") != candidate_id
            or not isinstance(reports, dict)
            or set(reports) != {"matched", "incumbent"}
        ):
            raise ValueError("rebuild short-screen candidate binding changed")
        report_paths = [pathlib.Path(reports[name]["path"]) for name in ("matched", "incumbent")]
        if (
            report_paths[0].parent != report_paths[1].parent
            or report_paths[0].name != "matched.json"
            or report_paths[1].name != "incumbent.json"
        ):
            raise ValueError("rebuild short-screen output layout changed")
        recomputed = run_short_screen(
            comparison=comparison,
            candidate=_candidate_runtime_path(candidate, "search"),
            matched=_candidate_runtime_path(candidate, "rank4"),
            incumbent=incumbent,
            bank=development_bank,
            output_directory=report_paths[0].parent,
        )
        expected = {"candidate_id": candidate_id, "screen": recomputed}
        expected_short_records.append(expected)
        if recomputed["operational"] is True:
            ranked_short.append(
                (tuple(recomputed["ranking_key"]), candidate, recomputed)
            )
        else:
            expected_rejections.append(
                {
                    "candidate_id": candidate_id,
                    "reasons": recomputed["rejection_reasons"],
                }
            )
    ranked_short.sort(key=lambda item: item[0])
    if (
        short_records != expected_short_records
        or phase_screen.get("short_ranked")
        != [str(candidate["candidate_id"]) for _key, candidate, _screen in ranked_short]
        or phase_screen.get("short_rejections") != expected_rejections
    ):
        raise ValueError("rebuild short-screen result changed")

    finalists = ranked_short[:FULL_SCREEN_LIMIT]
    expected_finalist_records = [candidate for _key, candidate, _screen in finalists]
    if phase_screen.get("full_candidate_records") != expected_finalist_records:
        raise ValueError("rebuild full-screen finalist cut changed")
    saved_full_records = phase_screen.get("full_records")
    if not isinstance(saved_full_records, list) or len(saved_full_records) != len(finalists):
        raise ValueError("rebuild full-screen evidence is incomplete")
    expected_full_records = []
    qualified = []
    for (_key, candidate, short_record), saved in zip(
        finalists, saved_full_records, strict=True
    ):
        candidate_id = str(candidate["candidate_id"])
        if not isinstance(saved, dict) or set(saved) != {"candidate_id", "short", "full"}:
            raise ValueError("rebuild full-screen record shape changed")
        full = saved.get("full")
        reports = full.get("reports") if isinstance(full, dict) else None
        latency = full.get("latency") if isinstance(full, dict) else None
        if (
            saved.get("candidate_id") != candidate_id
            or saved.get("short") != short_record
            or not isinstance(reports, dict)
            or set(reports) != {"matched", "incumbent", "rank4", "jacek-nn"}
            or not isinstance(latency, dict)
            or not isinstance(latency.get("path"), str)
        ):
            raise ValueError("rebuild full-screen candidate binding changed")
        directories = {
            pathlib.Path(snapshot["path"]).parent
            for snapshot in reports.values()
            if isinstance(snapshot, dict) and isinstance(snapshot.get("path"), str)
        }
        directories.add(pathlib.Path(latency["path"]).parent)
        if len(directories) != 1:
            raise ValueError("rebuild full-screen output layout changed")
        recovery_report = _candidate_recovery_report(candidate, "search")
        canonical_candidate = recovery_report["result"]["retention"]
        canonical_incumbent = recovery_report["references"][
            "retention_reference"
        ]["retention"]
        recomputed = run_full_screen(
            comparison=comparison,
            candidate=_candidate_runtime_path(candidate, "search"),
            matched=_candidate_runtime_path(candidate, "rank4"),
            incumbent=incumbent,
            bank=development_bank,
            output_directory=next(iter(directories)),
            classification="development",
            canonical_candidate=canonical_candidate,
            canonical_incumbent=canonical_incumbent,
        )
        expected = {
            "candidate_id": candidate_id,
            "short": short_record,
            "full": recomputed,
        }
        expected_full_records.append(expected)
        if recomputed["decision"]["eligible_for_full"] is True:
            qualified.append((candidate, recomputed, canonical_candidate))
    if saved_full_records != expected_full_records:
        raise ValueError("rebuild full-screen result changed")
    qualified.sort(
        key=lambda item: _official_qualified_key(item[0], item[1], item[2])
    )
    qualified_ids = [str(item[0]["candidate_id"]) for item in qualified]
    selected = qualified[0][0] if qualified else None
    if (
        phase_screen.get("qualified") != qualified_ids
        or phase_screen.get("selected_candidate_id")
        != (str(selected["candidate_id"]) if selected is not None else None)
    ):
        raise ValueError("rebuild official qualified-candidate selection changed")
    return selected


def validate_selected_candidate_lineage(
    *, selected: Mapping[str, object], inputs: Mapping[str, object]
) -> None:
    """Replay training and development gates before protected evidence opens."""

    import jacek_rebuild_corpus as rebuild_corpus
    import jacek_replay_recovery as recovery
    import jacek_selfsearch_workflow as selfsearch

    verify_body_hash(
        selected,
        schema="papersoccer.jacek-replay-rebuild-selected-candidate.v1",
        label="selected rebuild candidate",
    )
    if (
        selected.get("rebuild_id") != REBUILD_ID
        or selected.get("selection_policy")
        != "development-gates-then-worst-normalized-margin-canonical-huber-hash"
        or selected.get("protected_test_opened") is not False
        or selected.get("sealed_final_bank_opened") is not False
        or selected.get("blind_holdout_labels_opened") is not False
    ):
        raise ValueError("selected rebuild policy was weakened")
    inputs_path = pathlib.Path(inputs["corpus"]["path"])
    if selected.get("inputs") != inputs.get("_self_snapshot"):
        raise ValueError("selected candidate uses different frozen inputs")
    candidate = selected.get("candidate")
    if not isinstance(candidate, dict):
        raise ValueError("selected candidate record is missing")
    verify_body_hash(
        candidate,
        schema=REBUILD_CANDIDATE_SCHEMA,
        label="rebuild candidate",
    )
    if candidate.get("candidate_id") != selected.get("candidate_id"):
        raise ValueError("selected candidate identity changed")
    if (
        selected.get("selected_runtime")
        != artifact_snapshot(_candidate_runtime_path(candidate, "search"))
        or selected.get("matched_runtime")
        != artifact_snapshot(_candidate_runtime_path(candidate, "rank4"))
    ):
        raise ValueError("selected candidate runtime was substituted")
    corpus = load_frozen_rebuild_corpus(inputs_path)
    incumbent = pathlib.Path(inputs["incumbent_runtime"]["path"])
    phase = str(candidate.get("phase", candidate.get("specification", {}).get("phase", "")))
    matrix = load_json(pathlib.Path(inputs["matrix"]["path"]), "rebuild matrix")
    if phase == "residual":
        base_spec = candidate_spec_from_record(matrix["phases"]["v5_recovery"][0])
        initial = {
            "search": base_spec.search_initial_runtime,
            "rank4": base_spec.rank4_initial_runtime,
        }
        learning_rate = float(candidate["learning_rate"])
        residual = True
        spec = None
        expected_residual_ids = {
            1e-4: "residual-lr1e4",
            3e-4: "residual-lr3e4",
            1e-3: "residual-lr1e3",
        }
        if (
            candidate.get("rank") != 16
            or candidate.get("candidate_id")
            != expected_residual_ids.get(learning_rate)
            or learning_rate
            not in matrix["phases"]["residual_fallback"]["learning_rates"]
        ):
            raise ValueError("residual candidate is outside the frozen matrix")
    else:
        spec = candidate_spec_from_record(candidate["specification"])
        initial = {
            "search": spec.search_initial_runtime,
            "rank4": spec.rank4_initial_runtime,
        }
        learning_rate = spec.learning_rate
        residual = False
        matrix_key = {
            "v5-recovery": "v5_recovery",
            "canonical-basin": "canonical_basins",
            "scratch": "scratch_pretraining",
        }.get(spec.phase)
        if matrix_key in {"v5_recovery", "canonical_basins"} and candidate[
            "specification"
        ] not in matrix["phases"][matrix_key]:
            raise ValueError("candidate specification is outside the frozen matrix")
        if spec.phase == "scratch" and (
            not spec.base_id.startswith("scratch-s")
            or int(spec.base_id.removeprefix("scratch-s"))
            not in matrix["phases"]["scratch_pretraining"]["seeds"]
            or spec.trainable_layers != "all"
            or spec.learning_rate != JOINT_LEARNING_RATE
            or spec.selection_policy != "epoch-zero-improvement"
            or spec.training_recipe != "v6-joint"
        ):
            raise ValueError("scratch candidate is outside the frozen matrix")
    for arm in ("search", "rank4"):
        datasets = recovery.prepare_recovery_datasets(
            new_manifests=corpus.training_manifest_paths(arm),
            anchor_manifests=corpus.anchor_manifest_paths(arm),
            selection_manifests=corpus.validation_manifest_paths(arm),
            retention_manifests=corpus.retention_validation_manifest_paths(arm),
        )
        prepared = recovery.bind_recovery_runtimes(
            datasets,
            initial_runtime=initial[arm],
            retention_reference_runtime=incumbent,
            new_reference_runtime=initial[arm],
        )
        records = candidate.get(f"{arm}_seed_runs")
        if not isinstance(records, list) or len(records) != 3:
            raise ValueError("candidate seed-run set is incomplete")
        expected_seeds = ORDER_SEEDS if residual else spec.order_seeds
        if sorted(int(record.get("seed", -1)) for record in records) != sorted(
            expected_seeds
        ):
            raise ValueError("candidate seed-run set differs from frozen matrix")
        for record in records:
            receipt = record.get("receipt") if isinstance(record, dict) else None
            if (
                not isinstance(receipt, dict)
                or not isinstance(receipt.get("path"), str)
                or artifact_snapshot(pathlib.Path(receipt["path"])) != receipt
            ):
                raise ValueError("candidate seed receipt binding is stale")
            directory = pathlib.Path(receipt["path"]).parent
            seed = int(record["seed"])
            if residual:
                _parameters, report = recovery.load_residual_recovery_result(
                    prepared,
                    recovery.ResidualRecoveryConfiguration(learning_rate, seed),
                    directory,
                )
            elif spec.training_recipe == "v6-joint":
                _parameters, report = recovery.load_v6_joint_result(
                    prepared,
                    recovery.V6JointConfiguration(seed),
                    directory,
                )
            else:
                _parameters, report = recovery.load_recovery_result(
                    prepared,
                    recovery.RecoveryConfiguration(
                        spec.trainable_layers,
                        learning_rate,
                        seed,
                        spec.selection_policy,
                    ),
                    directory,
                )
            if report["result"] != record.get("result"):
                raise ValueError("candidate seed result changed after replay")
            for artifact_name in ("runtime", "report"):
                artifact = record.get(artifact_name)
                if (
                    not isinstance(artifact, dict)
                    or not isinstance(artifact.get("path"), str)
                    or artifact_snapshot(pathlib.Path(artifact["path"])) != artifact
                ):
                    raise ValueError("candidate seed artifact binding is stale")
        if candidate.get(f"selected_{arm}") != _best_recovery_seed(records):
            raise ValueError("candidate selected seed is not deterministic")
    if candidate.get("offline_eligible") is not True:
        raise ValueError("selected candidate was not offline eligible")

    phase_screen = selected.get("phase_screen")
    if not isinstance(phase_screen, dict):
        raise ValueError("selected candidate phase screen is missing")
    verify_body_hash(
        phase_screen,
        schema="papersoccer.jacek-replay-rebuild-phase-screen.v1",
        label="rebuild phase screen",
    )
    if phase == "scratch":
        lineage_snapshot = phase_screen.get("phase_inputs", {}).get(
            "scratch_base_lineage"
        )
        if (
            not isinstance(lineage_snapshot, dict)
            or not isinstance(lineage_snapshot.get("path"), str)
            or artifact_snapshot(pathlib.Path(lineage_snapshot["path"]))
            != lineage_snapshot
        ):
            raise ValueError("scratch base lineage binding is missing")
        scratch_lineage = load_json(
            pathlib.Path(lineage_snapshot["path"]), "scratch base lineage"
        )
        verify_body_hash(
            scratch_lineage,
            schema="papersoccer.jacek-replay-rebuild-scratch-bases.v1",
            label="scratch base lineage",
        )
        if (
            scratch_lineage.get("protected_test_supplied") is not False
            or len(scratch_lineage.get("pretraining_reports", [])) != 6
            or len(scratch_lineage.get("selected_specs", [])) != 3
            or candidate.get("specification")
            not in scratch_lineage.get("selected_specs", [])
        ):
            raise ValueError("scratch base selection semantics changed")
        for report_snapshot in scratch_lineage["pretraining_reports"]:
            if (
                not isinstance(report_snapshot, dict)
                or not isinstance(report_snapshot.get("path"), str)
                or artifact_snapshot(pathlib.Path(report_snapshot["path"]))
                != report_snapshot
            ):
                raise ValueError("scratch pretraining report binding is stale")
    ladder_screens = selected.get("ladder_phase_screens")
    if not isinstance(ladder_screens, list) or not ladder_screens:
        raise ValueError("selected candidate ladder transcript is missing")
    allowed_order = ["v5-recovery", "canonical-basins", "scratch-joint", "residual"]
    observed_order = []
    winners = []
    for ladder_screen in ladder_screens:
        if not isinstance(ladder_screen, dict):
            raise ValueError("selected ladder phase screen is malformed")
        replayed_winner = validate_phase_screen(
            phase_screen=ladder_screen, inputs=inputs
        )
        observed_order.append(str(ladder_screen.get("phase")))
        if ladder_screen.get("selected_candidate_id") is not None:
            if replayed_winner != candidate:
                raise ValueError("selected ladder winner was substituted")
            winners.append(ladder_screen)
    winning_phase = str(phase_screen.get("phase"))
    if winning_phase not in allowed_order:
        raise ValueError("selected candidate phase is outside the frozen ladder")
    if (
        not ladder_phase_order_is_valid(observed_order, winning_phase)
        or winners != [phase_screen]
        or ladder_screens[-1] != phase_screen
    ):
        raise ValueError("selected candidate is not the first passing ladder phase")
    selected_seed_receipt = candidate.get("selected_search", {}).get("receipt")
    if (
        not isinstance(selected_seed_receipt, dict)
        or not isinstance(selected_seed_receipt.get("path"), str)
    ):
        raise ValueError("selected candidate seed receipt is missing")
    phase_directory = next(
        (
            parent
            for parent in pathlib.Path(selected_seed_receipt["path"]).parents
            if parent.name == winning_phase
        ),
        None,
    )
    if phase_directory is None:
        raise ValueError("selected candidate output phase layout changed")
    ladder_root = phase_directory.parent
    included = set(observed_order)
    for prior_phase in allowed_order[:3]:
        prior_root = ladder_root / prior_phase
        started = any((prior_root / "launches").glob("*.json")) or (
            prior_root / "screening/phase-screen.json"
        ).exists()
        if started and prior_phase not in included:
            raise ValueError("selected ladder transcript omitted a started phase")
    if phase_screen.get("selected_candidate_id") != candidate.get("candidate_id"):
        raise ValueError("phase screen selected a different candidate")
    full_record = next(
        (
            row for row in phase_screen.get("full_records", [])
            if isinstance(row, dict)
            and row.get("candidate_id") == candidate.get("candidate_id")
        ),
        None,
    )
    if not isinstance(full_record, dict):
        raise ValueError("selected candidate has no full development gate")
    full = full_record.get("full")
    if not isinstance(full, dict):
        raise ValueError("selected candidate full gate is malformed")
    banks = load_json(pathlib.Path(inputs["opening_banks"]["path"]), "rebuild banks")
    bank = pathlib.Path(banks["development"]["artifact"]["path"])
    comparison = pathlib.Path(inputs["comparison"]["path"])
    candidate_runtime = _candidate_runtime_path(candidate, "search")
    matched_runtime = _candidate_runtime_path(candidate, "rank4")
    panels = {
        "matched": ("jacek-replay", matched_runtime),
        "incumbent": ("jacek-replay", incumbent),
        "rank4": ("rank4", None),
        "jacek-nn": ("jacek-nn", None),
    }
    for name, (opponent, control) in panels.items():
        snapshot = full.get("reports", {}).get(name)
        if not isinstance(snapshot, dict) or not isinstance(snapshot.get("path"), str):
            raise ValueError("selected development report binding is missing")
        report = _comparison_report(
            comparison=comparison,
            model=candidate_runtime,
            control_model=control,
            bank=bank,
            classification="development",
            opponent=opponent,
            pairs=FULL_SCREEN_PAIRS,
            output=pathlib.Path(snapshot["path"]),
        )
        if artifact_snapshot(pathlib.Path(snapshot["path"])) != snapshot:
            raise ValueError("selected development report snapshot is stale")
        del report
    source_identities = selfsearch._source_identities(
        pathlib.Path(inputs["repository"]["path"])
    )
    latency_snapshot = full.get("latency")
    if not isinstance(latency_snapshot, dict) or not isinstance(
        latency_snapshot.get("path"), str
    ):
        raise ValueError("selected development latency binding is missing")
    latency = selfsearch.run_latency_audit(
        comparison=comparison,
        model=candidate_runtime,
        bank=bank,
        output=pathlib.Path(latency_snapshot["path"]),
        classification="development",
        source_identities=source_identities,
    )
    if artifact_snapshot(pathlib.Path(latency_snapshot["path"])) != latency_snapshot:
        raise ValueError("selected development latency snapshot is stale")
    recovery_report = _candidate_recovery_report(candidate, "search")
    recomputed = selfsearch.pilot_decision(
        matched_report=pathlib.Path(full["reports"]["matched"]["path"]),
        incumbent_report=pathlib.Path(full["reports"]["incumbent"]["path"]),
        rank4_report=pathlib.Path(full["reports"]["rank4"]["path"]),
        jacek_nn_report=pathlib.Path(full["reports"]["jacek-nn"]["path"]),
        anchor_candidate=recovery_report["result"]["retention"],
        anchor_incumbent=recovery_report["references"]["retention_reference"]["retention"],
        uncontended_max_ms=float(latency["candidate_max_ms"]),
    )
    if recomputed != full.get("decision") or not recomputed["eligible_for_full"]:
        raise ValueError("selected candidate development qualification changed")


def _qualification_artifact(record: object, label: str) -> pathlib.Path:
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        raise ValueError(f"rebuild qualification {label} binding is malformed")
    path = pathlib.Path(record["path"])
    if artifact_snapshot(path) != record:
        raise ValueError(f"rebuild qualification {label} binding is stale")
    return path


def _validate_qualification_label_receipts(
    *, labels: pathlib.Path, positions: pathlib.Path,
    teacher: pathlib.Path, teacher_source_sha256: str,
) -> None:
    """Bind the merged blind labels to every fixed-work teacher invocation."""

    import jacek_replay_corpus as replay_corpus
    import jacek_selfsearch_workflow as selfsearch

    lines = positions.read_text(encoding="utf-8").splitlines()
    if len(lines) != 12_001:
        raise ValueError("rebuild blind-holdout position coverage changed")
    chunks = [lines[start : start + 25] for start in range(1, len(lines), 25)]
    input_root = labels.parent / "inputs"
    output_root = labels.parent / "outputs"
    receipt_root = labels.parent / "receipts"
    expected_names = {f"chunk-{index:04d}" for index in range(len(chunks))}
    if (
        not input_root.is_dir()
        or not output_root.is_dir()
        or not receipt_root.is_dir()
        or {path.stem for path in input_root.glob("chunk-*.tsv")}
        != expected_names
        or {
            path.name.removesuffix(".jsonl")
            for path in output_root.glob("chunk-*.jsonl")
        }
        != expected_names
        or {
            path.name.removesuffix(".json")
            for path in receipt_root.glob("chunk-*.json")
        }
        != expected_names
    ):
        raise ValueError("rebuild blind-holdout label receipt set is incomplete")

    merged = bytearray()
    for index, rows in enumerate(chunks):
        source = input_root / f"chunk-{index:04d}.tsv"
        output = output_root / f"chunk-{index:04d}.jsonl"
        receipt_path = receipt_root / f"chunk-{index:04d}.json"
        expected_source = (lines[0] + "\n" + "\n".join(rows) + "\n").encode()
        if source.read_bytes() != expected_source:
            raise ValueError("rebuild blind-holdout label chunk input changed")
        expected = {
            "schema": "papersoccer.jacek-rebuild-holdout-label-chunk.v1",
            "rebuild_id": REBUILD_ID,
            "chunk": index,
            "configuration": {"nodes": 400_000, "time_ms": 0},
            "teacher": artifact_snapshot(teacher),
            "teacher_source_sha256": teacher_source_sha256,
            "positions": artifact_snapshot(source),
            "output": artifact_snapshot(output),
        }
        if load_json(receipt_path, "blind-holdout label receipt") != expected:
            raise ValueError("rebuild blind-holdout label receipt changed")
        if selfsearch._validate_label_output(
            output=output,
            positions=source,
            schema=replay_corpus.RANK4_TEACHER_SCHEMA,
            campaign_id=REBUILD_ID,
            nodes=400_000,
            model_sha256=None,
            source_sha256=teacher_source_sha256,
        ) != len(rows):
            raise ValueError("rebuild blind-holdout label chunk lost coverage")
        merged.extend(output.read_bytes())
    if bytes(merged) != labels.read_bytes():
        raise ValueError("rebuild blind-holdout merged labels changed")


def _validate_qualification_holdout(
    *, evidence_path: pathlib.Path, inputs: Mapping[str, object],
    selection_path: pathlib.Path, candidate: pathlib.Path,
    incumbent: pathlib.Path,
) -> dict[str, object]:
    """Rebuild the sealed retention evidence from its fixed-work labels."""

    import numpy as np

    import jacek_replay_corpus as replay_corpus
    import jacek_replay_retention as retention
    import jacek_selfsearch_workflow as selfsearch

    evidence = load_json(evidence_path, "rebuild blind-holdout evidence")
    evidence_inputs = evidence.get("inputs")
    if not isinstance(evidence_inputs, dict):
        raise ValueError("rebuild blind-holdout evidence inputs are missing")
    shard_record = evidence_inputs.get("shard_manifest")
    if (
        not isinstance(shard_record, dict)
        or not isinstance(shard_record.get("path"), str)
    ):
        raise ValueError("rebuild blind-holdout shard binding is malformed")
    shard_manifest = pathlib.Path(shard_record["path"])
    if retention.artifact_snapshot(shard_manifest) != shard_record:
        raise ValueError("rebuild blind-holdout shard binding is stale")

    freeze_manifest = pathlib.Path(inputs["blind_holdout"]["path"])
    frozen_positions = pathlib.Path(inputs["blind_holdout_positions"]["path"])
    corpus_manifest = pathlib.Path(inputs["corpus"]["path"])
    selection_snapshot = retention.artifact_snapshot(selection_path)
    freeze, frozen_rows = load_frozen_blind_holdout(
        frozen_positions, freeze_manifest
    )
    if (
        freeze.get("campaign_id") != REBUILD_ID
        or freeze.get("profile") != "rebuild"
        or freeze.get("role") != "retention-rebuild"
        or freeze.get("training_eligible") is not False
        or freeze.get("configuration", {}).get("groups") != 600
        or freeze.get("configuration", {}).get("rows_per_group") != 20
        or freeze.get("configuration", {}).get("selection_seed")
        != BLIND_HOLDOUT_SEED
        or freeze.get("timing", {}).get("training_inputs_frozen_by")
        != retention.artifact_snapshot(corpus_manifest)
        or freeze.get("timing", {}).get("teacher_labels_opened") is not False
        or freeze.get("timing", {}).get("selected_model_opened") is not False
        or freeze.get("timing", {}).get("required_reveal_order")
        != (
            "freeze-before-model-selection;labels-after-model-selection;"
            "metrics-after-selected-runtime-binding"
        )
        or len(frozen_rows) != 12_000
    ):
        raise ValueError("rebuild blind-holdout freeze semantics changed")

    shard = retention.load_holdout_shard(shard_manifest)
    shard_inputs = shard.manifest.get("inputs")
    shard_reveal = shard.manifest.get("reveal")
    if (
        shard.manifest.get("campaign_id") != REBUILD_ID
        or shard.manifest.get("profile") != "rebuild"
        or shard.manifest.get("role") != "retention-rebuild"
        or shard.manifest.get("training_eligible") is not False
        or shard.manifest.get("training_loader_compatible") is not False
        or shard.manifest.get("base_positions") != 12_000
        or shard.manifest.get("samples") != 24_000
        or shard.manifest.get("root_groups") != 600
        or not isinstance(shard_inputs, dict)
        or shard_inputs.get("freeze_manifest")
        != retention.artifact_snapshot(freeze_manifest)
        or shard_inputs.get("frozen_positions")
        != retention.artifact_snapshot(frozen_positions)
        or not isinstance(shard_reveal, dict)
        or shard_reveal.get("policy")
        != "labels-opened-only-after-model-selection-receipt"
        or shard_reveal.get("selection_receipt") != selection_snapshot
        or shard_reveal.get("training_input_receipt")
        != retention.artifact_snapshot(corpus_manifest)
    ):
        raise ValueError("rebuild blind-holdout shard semantics changed")
    labels_record = shard_inputs.get("labels")
    if (
        not isinstance(labels_record, dict)
        or not isinstance(labels_record.get("path"), str)
    ):
        raise ValueError("rebuild blind-holdout labels binding is malformed")
    labels_path = pathlib.Path(labels_record["path"])
    if retention.artifact_snapshot(labels_path) != labels_record:
        raise ValueError("rebuild blind-holdout labels binding is stale")

    repository = pathlib.Path(inputs["repository"]["path"])
    teacher_source_sha256 = selfsearch._source_identities(repository)[
        "rank4_teacher_source_sha256"
    ]
    _validate_qualification_label_receipts(
        labels=labels_path,
        positions=frozen_positions,
        teacher=pathlib.Path(inputs["rank4_teacher"]["path"]),
        teacher_source_sha256=teacher_source_sha256,
    )
    if selfsearch._validate_label_output(
        output=labels_path,
        positions=frozen_positions,
        schema=replay_corpus.RANK4_TEACHER_SCHEMA,
        campaign_id=REBUILD_ID,
        nodes=retention.RANK4_FIXED_NODES,
        model_sha256=None,
        source_sha256=teacher_source_sha256,
    ) != len(frozen_rows):
        raise ValueError("rebuild blind-holdout labels lost exact coverage")
    label_rows = retention._load_rank4_labels(labels_path)
    arrays, termination_counts, teacher_configuration = (
        retention._holdout_arrays(frozen_rows, label_rows)
    )
    observed_arrays = {
        "indptr": shard.indptr,
        "indices": shard.indices,
        "targets": shard.targets,
        "weights": shard.weights,
        "root_group_ids": shard.root_group_ids,
        "position_ids": shard.position_ids,
        "canonical_fingerprints": shard.canonical_fingerprints,
        "orientations": shard.orientations,
    }
    if (
        set(arrays) != set(observed_arrays)
        or any(
            not np.array_equal(arrays[name], observed_arrays[name])
            for name in arrays
        )
        or shard.manifest.get("termination_counts") != termination_counts
        or shard.manifest.get("teacher_configuration")
        != teacher_configuration
    ):
        raise ValueError("rebuild blind-holdout packed labels changed")

    recomputed = retention.evaluate_holdout(
        shard_manifest=shard_manifest,
        actor_runtime=incumbent,
        candidate_runtime=candidate,
        selection_receipt=selection_path,
    )
    noninferiority = recomputed.get("noninferiority")
    gates = (
        noninferiority.get("gates")
        if isinstance(noninferiority, dict) else None
    )
    if (
        recomputed != evidence
        or not isinstance(noninferiority, dict)
        or noninferiority.get("thresholds")
        != retention.FROZEN_THRESHOLDS.record()
        or noninferiority.get("root_groups") != 600
        or not isinstance(gates, dict)
        or set(gates)
        != {
            "point_sign", "point_huber", "cluster_sign", "cluster_huber",
            "pass",
        }
        or recomputed.get("pass")
        is not gates.get("pass")
    ):
        raise ValueError("rebuild blind-holdout evidence changed after replay")
    return recomputed


def validate_qualification_receipt(path: pathlib.Path) -> dict[str, object]:
    """Strictly replay every protected gate in a rebuild qualification."""

    import jacek_rebuild_corpus as rebuild_corpus
    import jacek_selfsearch_workflow as selfsearch

    path = path.resolve()
    qualification = load_json(path, "rebuild qualification")
    verify_body_hash(
        qualification,
        schema=REBUILD_QUALIFICATION_SCHEMA,
        label="rebuild qualification",
    )
    if (
        qualification.get("rebuild_id") != REBUILD_ID
        or not isinstance(qualification.get("pass"), bool)
        or qualification.get("local_only") is not True
        or qualification.get("canonical_rank4_replaced") is not False
        or qualification.get("external_upload") is not False
    ):
        raise ValueError("rebuild qualification policy was weakened")

    inputs_path = _qualification_artifact(
        qualification.get("inputs"), "inputs"
    )
    selection_path = _qualification_artifact(
        qualification.get("selection"), "selection"
    )
    candidate = _qualification_artifact(
        qualification.get("candidate"), "candidate"
    )
    matched = _qualification_artifact(
        qualification.get("matched"), "matched"
    )

    # Replay selection before opening any protected test, final-bank, or
    # blind-label evidence.  The receipt must attest that all three were still
    # sealed when this candidate became immutable.
    inputs = validate_rebuild_inputs(inputs_path)
    inputs = {**inputs, "_self_snapshot": artifact_snapshot(inputs_path)}
    selected = load_json(selection_path, "selected rebuild candidate")
    if (
        selected.get("schema")
        != "papersoccer.jacek-replay-rebuild-selected-candidate.v1"
        or selected.get("protected_test_opened") is not False
        or selected.get("sealed_final_bank_opened") is not False
        or selected.get("blind_holdout_labels_opened") is not False
    ):
        raise ValueError("rebuild qualification reveal order is invalid")
    validate_selected_candidate_lineage(selected=selected, inputs=inputs)
    if (
        selected.get("selected_runtime") != qualification.get("candidate")
        or selected.get("matched_runtime") != qualification.get("matched")
    ):
        raise ValueError("rebuild qualification substituted selected runtimes")

    corpus = load_frozen_rebuild_corpus(pathlib.Path(inputs["corpus"]["path"]))
    incumbent = pathlib.Path(inputs["incumbent_runtime"]["path"])
    canonical_path = _qualification_artifact(
        qualification.get("canonical_test"), "canonical test"
    )
    canonical_metrics = load_json(canonical_path, "canonical test metrics")
    recomputed_canonical = selfsearch.anchor_metrics(
        candidate_runtime=candidate,
        incumbent_runtime=incumbent,
        anchor_validation_manifests=corpus.protected_test_manifest_paths,
        expected_split="test",
    )
    if canonical_metrics != recomputed_canonical:
        raise ValueError("rebuild canonical-test evidence changed after replay")

    holdout_path = _qualification_artifact(
        qualification.get("blind_holdout"), "blind holdout"
    )
    holdout = _validate_qualification_holdout(
        evidence_path=holdout_path,
        inputs=inputs,
        selection_path=selection_path,
        candidate=candidate,
        incumbent=incumbent,
    )

    banks_path = pathlib.Path(inputs["opening_banks"]["path"])
    banks = load_json(banks_path, "rebuild opening banks")
    validate_opening_banks(banks)
    final_bank = _qualification_artifact(
        qualification.get("final_bank"), "final bank"
    )
    if banks.get("final", {}).get("artifact") != qualification.get("final_bank"):
        raise ValueError("rebuild qualification uses a different sealed final bank")

    full = qualification.get("final_game_gate")
    if not isinstance(full, dict):
        raise ValueError("rebuild qualification final game gate is missing")
    comparison = pathlib.Path(inputs["comparison"]["path"])
    panels = {
        "matched": ("jacek-replay", matched),
        "incumbent": ("jacek-replay", incumbent),
        "rank4": ("rank4", None),
        "jacek-nn": ("jacek-nn", None),
    }
    report_snapshots = full.get("reports")
    if not isinstance(report_snapshots, dict) or set(report_snapshots) != set(panels):
        raise ValueError("rebuild qualification final report set is incomplete")
    for name, (opponent, control_model) in panels.items():
        report_path = _qualification_artifact(
            report_snapshots.get(name), f"final {name} report"
        )
        _comparison_report(
            comparison=comparison,
            model=candidate,
            control_model=control_model,
            bank=final_bank,
            classification="final",
            opponent=opponent,
            pairs=FULL_SCREEN_PAIRS,
            output=report_path,
        )

    latency_path = _qualification_artifact(
        full.get("latency"), "final latency"
    )
    latency = selfsearch.run_latency_audit(
        comparison=comparison,
        model=candidate,
        bank=final_bank,
        output=latency_path,
        classification="final",
        source_identities=selfsearch._source_identities(
            pathlib.Path(inputs["repository"]["path"])
        ),
    )
    decision = selfsearch.pilot_decision(
        matched_report=pathlib.Path(report_snapshots["matched"]["path"]),
        incumbent_report=pathlib.Path(report_snapshots["incumbent"]["path"]),
        rank4_report=pathlib.Path(report_snapshots["rank4"]["path"]),
        jacek_nn_report=pathlib.Path(report_snapshots["jacek-nn"]["path"]),
        anchor_candidate=canonical_metrics["candidate_metrics"],
        anchor_incumbent=canonical_metrics["incumbent_metrics"],
        uncontended_max_ms=float(latency["candidate_max_ms"]),
    )
    expected_full = {
        "candidate": artifact_snapshot(candidate),
        "matched": artifact_snapshot(matched),
        "bank": artifact_snapshot(final_bank),
        "classification": "final",
        "reports": {
            name: artifact_snapshot(
                pathlib.Path(report_snapshots[name]["path"])
            )
            for name in panels
        },
        "latency": artifact_snapshot(latency_path),
        "decision": decision,
    }
    if full != expected_full:
        raise ValueError("rebuild qualification final game gate changed after replay")

    passed = bool(
        holdout.get("pass") is True
        and decision.get("eligible_for_full") is True
    )
    expected_body: dict[str, object] = {
        "schema": REBUILD_QUALIFICATION_SCHEMA,
        "rebuild_id": REBUILD_ID,
        "inputs": artifact_snapshot(inputs_path),
        "selection": artifact_snapshot(selection_path),
        "candidate": artifact_snapshot(candidate),
        "matched": artifact_snapshot(matched),
        "canonical_test": artifact_snapshot(canonical_path),
        "blind_holdout": artifact_snapshot(holdout_path),
        "final_bank": artifact_snapshot(final_bank),
        "final_game_gate": expected_full,
        "pass": passed,
        "local_only": True,
        "canonical_rank4_replaced": False,
        "external_upload": False,
    }
    observed_body = dict(qualification)
    observed_body.pop("body_sha256", None)
    if observed_body != expected_body:
        raise ValueError("rebuild qualification receipt is not exact")
    return qualification


def qualify_selected_candidate(
    *, inputs_manifest: pathlib.Path, selected_receipt: pathlib.Path,
    output_directory: pathlib.Path, workers: int = 10,
) -> dict[str, object]:
    """Open protected evidence only after one development-qualified runtime."""

    import jacek_rebuild_corpus as rebuild_corpus
    import jacek_replay_retention as retention
    import jacek_selfsearch_workflow as selfsearch

    inputs = validate_rebuild_inputs(inputs_manifest)
    inputs = {**inputs, "_self_snapshot": artifact_snapshot(inputs_manifest)}
    selected = load_json(selected_receipt, "selected rebuild candidate")
    if (
        selected.get("schema")
        != "papersoccer.jacek-replay-rebuild-selected-candidate.v1"
        or selected.get("protected_test_opened") is not False
        or selected.get("sealed_final_bank_opened") is not False
        or selected.get("blind_holdout_labels_opened") is not False
    ):
        raise ValueError("selected candidate receipt reveal state is invalid")
    validate_selected_candidate_lineage(selected=selected, inputs=inputs)
    candidate = pathlib.Path(selected["selected_runtime"]["path"])
    matched = pathlib.Path(selected["matched_runtime"]["path"])
    incumbent = pathlib.Path(inputs["incumbent_runtime"]["path"])
    comparison = pathlib.Path(inputs["comparison"]["path"])
    teacher = pathlib.Path(inputs["rank4_teacher"]["path"])
    frozen_positions = pathlib.Path(inputs["blind_holdout_positions"]["path"])
    freeze_manifest = pathlib.Path(inputs["blind_holdout"]["path"])
    corpus = load_frozen_rebuild_corpus(pathlib.Path(inputs["corpus"]["path"]))
    banks = load_json(
        pathlib.Path(inputs["opening_banks"]["path"]), "rebuild opening banks"
    )
    validate_opening_banks(banks)
    final_bank = pathlib.Path(banks["final"]["artifact"]["path"])
    repository = pathlib.Path(inputs["repository"]["path"])
    source_sha256 = selfsearch._source_identities(repository)[
        "rank4_teacher_source_sha256"
    ]
    freeze, frozen_rows = load_frozen_blind_holdout(
        frozen_positions, freeze_manifest
    )
    if (
        freeze.get("timing", {}).get("training_inputs_frozen_by")
        != retention.artifact_snapshot(pathlib.Path(inputs["corpus"]["path"]))
        or len(frozen_rows) != 12_000
        or freeze.get("configuration", {}).get("groups") != 600
        or freeze.get("configuration", {}).get("rows_per_group") != 20
    ):
        raise ValueError("blind holdout freeze is not bound to rebuild inputs")
    labels = label_blind_holdout(
        teacher=teacher,
        positions=frozen_positions,
        teacher_source_sha256=source_sha256,
        output_directory=output_directory / "blind-holdout-labels",
        workers=workers,
    )
    _npz, holdout_manifest, _packed = retention.pack_holdout(
        frozen_positions=frozen_positions,
        freeze_manifest=freeze_manifest,
        labels=labels,
        selection_receipt=selected_receipt,
        teacher_source_sha256=source_sha256,
        output_directory=output_directory / "blind-holdout-shard",
    )
    holdout_evidence_path = output_directory / "blind-holdout-evidence.json"
    holdout = retention.evaluate_holdout(
        shard_manifest=holdout_manifest,
        actor_runtime=incumbent,
        candidate_runtime=candidate,
        selection_receipt=selected_receipt,
        output=holdout_evidence_path,
    )
    canonical_metrics_path = output_directory / "canonical-test-metrics.json"
    canonical_metrics = selfsearch.anchor_metrics(
        candidate_runtime=candidate,
        incumbent_runtime=incumbent,
        anchor_validation_manifests=corpus.protected_test_manifest_paths,
        expected_split="test",
    )
    if canonical_metrics_path.exists():
        if load_json(canonical_metrics_path, "canonical test metrics") != canonical_metrics:
            raise ValueError("existing canonical test metrics are stale")
    else:
        atomic_write(
            canonical_metrics_path,
            canonical_json_bytes(canonical_metrics, pretty=True),
        )
    full = run_full_screen(
        comparison=comparison,
        candidate=candidate,
        matched=matched,
        incumbent=incumbent,
        bank=final_bank,
        output_directory=output_directory / "final-game-gates",
        classification="final",
        canonical_candidate=canonical_metrics["candidate_metrics"],
        canonical_incumbent=canonical_metrics["incumbent_metrics"],
    )
    passed = bool(
        holdout.get("pass") is True
        and full["decision"].get("eligible_for_full") is True
    )
    body: dict[str, object] = {
        "schema": REBUILD_QUALIFICATION_SCHEMA,
        "rebuild_id": REBUILD_ID,
        "inputs": artifact_snapshot(inputs_manifest),
        "selection": artifact_snapshot(selected_receipt),
        "candidate": artifact_snapshot(candidate),
        "matched": artifact_snapshot(matched),
        "canonical_test": artifact_snapshot(canonical_metrics_path),
        "blind_holdout": artifact_snapshot(holdout_evidence_path),
        "final_bank": artifact_snapshot(final_bank),
        "final_game_gate": full,
        "pass": passed,
        "local_only": True,
        "canonical_rank4_replaced": False,
        "external_upload": False,
    }
    qualification = {
        **body, "body_sha256": sha256_bytes(canonical_json_bytes(body))
    }
    qualification_path = output_directory / "qualification.json"
    if qualification_path.exists():
        if load_json(qualification_path, "rebuild qualification") != qualification:
            raise ValueError("existing rebuild qualification is stale")
    else:
        atomic_write(
            qualification_path, canonical_json_bytes(qualification, pretty=True)
        )
    if validate_qualification_receipt(qualification_path) != qualification:
        raise ValueError("published rebuild qualification changed during replay")
    if passed:
        promotion = output_directory / "promoted"
        promoted_runtime = promotion / "jacek_replay_bfm.runtime"
        if promoted_runtime.exists():
            if promoted_runtime.read_bytes() != candidate.read_bytes():
                raise ValueError("promoted rebuild runtime is stale")
        else:
            atomic_write(promoted_runtime, candidate.read_bytes())
        launch_body: dict[str, object] = {
            "schema": "papersoccer.jacek-selfsearch-v7-starting-actor.v1",
            "campaign_id": "selfsearch-auto-20260826-v7",
            "pilot_campaign_id": "selfsearch-pilot-20260826-v7",
            "full_campaign_id": "selfsearch-full-20260826-v7",
            "starting_actor": artifact_snapshot(promoted_runtime),
            "qualification": artifact_snapshot(qualification_path),
            "pilot_games": 2_000,
            "conditional_full_games": 10_000,
            "external_upload": False,
            "replace_rank4": False,
        }
        launch = {
            **launch_body,
            "body_sha256": sha256_bytes(canonical_json_bytes(launch_body)),
        }
        atomic_write(
            promotion / "v7-starting-actor.json",
            canonical_json_bytes(launch, pretty=True),
        )
    return qualification


def _runtime_argument(value: str) -> tuple[str, pathlib.Path]:
    try:
        name, raw_path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("runtime must be NAME=PATH") from error
    path = pathlib.Path(raw_path).resolve()
    if not name or not path.is_file():
        raise argparse.ArgumentTypeError("runtime binding is invalid")
    return name, path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    matrix = subparsers.add_parser("matrix")
    matrix.add_argument("--v5-search-runtime", type=pathlib.Path, required=True)
    matrix.add_argument("--v5-rank4-runtime", type=pathlib.Path, required=True)
    matrix.add_argument(
        "--canonical-runtime", action="append", type=_runtime_argument,
        required=True,
    )
    matrix.add_argument("--output", type=pathlib.Path, required=True)
    discovered_matrix = subparsers.add_parser("matrix-from-campaigns")
    discovered_matrix.add_argument(
        "--canonical-campaign", type=pathlib.Path, required=True
    )
    discovered_matrix.add_argument("--v5-campaign", type=pathlib.Path, required=True)
    discovered_matrix.add_argument("--output", type=pathlib.Path, required=True)
    corpus = subparsers.add_parser("freeze-corpus")
    corpus.add_argument("--canonical-campaign", type=pathlib.Path, required=True)
    corpus.add_argument("--v5-campaign", type=pathlib.Path, required=True)
    corpus.add_argument("--v6-campaign", type=pathlib.Path, required=True)
    corpus.add_argument("--output-directory", type=pathlib.Path, required=True)
    validate = subparsers.add_parser("validate-matrix")
    validate.add_argument("matrix", type=pathlib.Path)
    banks = subparsers.add_parser("freeze-banks")
    banks.add_argument("--comparison", type=pathlib.Path, required=True)
    banks.add_argument(
        "--exclude-bank", action="append", type=pathlib.Path, required=True
    )
    banks.add_argument("--output-directory", type=pathlib.Path, required=True)
    campaign_banks = subparsers.add_parser("freeze-banks-from-campaigns")
    campaign_banks.add_argument("--comparison", type=pathlib.Path, required=True)
    campaign_banks.add_argument(
        "--evaluation-directory", type=pathlib.Path, required=True
    )
    campaign_banks.add_argument("--v5-campaign", type=pathlib.Path, required=True)
    campaign_banks.add_argument("--v6-campaign", type=pathlib.Path, required=True)
    campaign_banks.add_argument(
        "--output-directory", type=pathlib.Path, required=True
    )
    validate_banks = subparsers.add_parser("validate-banks")
    validate_banks.add_argument("manifest", type=pathlib.Path)
    holdout = subparsers.add_parser("freeze-holdout")
    holdout.add_argument("--candidate-positions", type=pathlib.Path, required=True)
    holdout.add_argument("--training-input-receipt", type=pathlib.Path, required=True)
    holdout.add_argument(
        "--exclude-shard", action="append", type=pathlib.Path, default=[]
    )
    holdout.add_argument(
        "--exclude-positions", action="append", type=pathlib.Path, default=[]
    )
    holdout.add_argument(
        "--exclude-roots", action="append", type=pathlib.Path, default=[]
    )
    holdout.add_argument("--output-directory", type=pathlib.Path, required=True)
    corpus_holdout = subparsers.add_parser("freeze-holdout-from-corpus")
    corpus_holdout.add_argument(
        "--candidate-positions", type=pathlib.Path, required=True
    )
    corpus_holdout.add_argument(
        "--candidate-manifest", type=pathlib.Path, required=True
    )
    corpus_holdout.add_argument("--corpus-manifest", type=pathlib.Path, required=True)
    corpus_holdout.add_argument(
        "--canonical-campaign", type=pathlib.Path, required=True
    )
    corpus_holdout.add_argument("--v5-campaign", type=pathlib.Path, required=True)
    corpus_holdout.add_argument("--v6-campaign", type=pathlib.Path, required=True)
    corpus_holdout.add_argument(
        "--output-directory", type=pathlib.Path, required=True
    )
    holdout_candidates = subparsers.add_parser("generate-holdout-candidates")
    holdout_candidates.add_argument("--generator", type=pathlib.Path, required=True)
    holdout_candidates.add_argument(
        "--output-directory", type=pathlib.Path, required=True
    )
    holdout_candidates.add_argument("--workers", type=int, default=10)
    inputs = subparsers.add_parser("freeze-inputs")
    inputs.add_argument("--repository", type=pathlib.Path, required=True)
    inputs.add_argument("--corpus-manifest", type=pathlib.Path, required=True)
    inputs.add_argument("--build-manifest", type=pathlib.Path, required=True)
    inputs.add_argument("--matrix-manifest", type=pathlib.Path, required=True)
    inputs.add_argument("--banks-manifest", type=pathlib.Path, required=True)
    inputs.add_argument("--holdout-manifest", type=pathlib.Path, required=True)
    inputs.add_argument("--holdout-positions", type=pathlib.Path, required=True)
    inputs.add_argument(
        "--holdout-candidate-manifest", type=pathlib.Path, required=True
    )
    inputs.add_argument(
        "--holdout-candidate-positions", type=pathlib.Path, required=True
    )
    inputs.add_argument("--comparison", type=pathlib.Path, required=True)
    inputs.add_argument("--rank4-teacher", type=pathlib.Path, required=True)
    inputs.add_argument("--incumbent-runtime", type=pathlib.Path, required=True)
    inputs.add_argument("--output", type=pathlib.Path, required=True)
    validate_inputs = subparsers.add_parser("validate-inputs")
    validate_inputs.add_argument("manifest", type=pathlib.Path)
    build = subparsers.add_parser("write-build-manifest")
    build.add_argument("--repository", type=pathlib.Path, required=True)
    build.add_argument("--expected-commit", required=True)
    build.add_argument("--comparison", type=pathlib.Path, required=True)
    build.add_argument("--rank4-teacher", type=pathlib.Path, required=True)
    build.add_argument("--holdout-generator", type=pathlib.Path, required=True)
    build.add_argument("--output", type=pathlib.Path, required=True)
    validate_build = subparsers.add_parser("validate-build-manifest")
    validate_build.add_argument("manifest", type=pathlib.Path)
    run = subparsers.add_parser("run")
    run.add_argument("--inputs", type=pathlib.Path, required=True)
    run.add_argument("--output-directory", type=pathlib.Path, required=True)
    run.add_argument("--training-workers", type=int, default=10)
    run.add_argument("--resume", action="store_true")
    qualify = subparsers.add_parser("qualify")
    qualify.add_argument("--inputs", type=pathlib.Path, required=True)
    qualify.add_argument("--selected-receipt", type=pathlib.Path, required=True)
    qualify.add_argument("--output-directory", type=pathlib.Path, required=True)
    qualify.add_argument("--workers", type=int, default=10)
    arguments = parser.parse_args()
    if arguments.command in {"matrix", "matrix-from-campaigns"}:
        if arguments.command == "matrix":
            v5_search = arguments.v5_search_runtime.resolve()
            v5_rank4 = arguments.v5_rank4_runtime.resolve()
            canonical_bases = arguments.canonical_runtime
        else:
            v5_search, v5_rank4 = v5_runtime_pair(arguments.v5_campaign)
            canonical_bases = canonical_runtime_bindings(
                arguments.canonical_campaign
            )
        record = matrix_record(
            v5_search=v5_search,
            v5_rank4=v5_rank4,
            canonical_bases=canonical_bases,
        )
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        if arguments.output.exists():
            existing = load_json(arguments.output, "rebuild matrix")
            if existing != record:
                raise ValueError("existing rebuild matrix is stale")
        else:
            arguments.output.write_bytes(canonical_json_bytes(record, pretty=True))
        print(json.dumps({"matrix": str(arguments.output), "sha256": sha256_file(arguments.output)}, indent=2))
    elif arguments.command == "freeze-corpus":
        path = freeze_corpus_from_campaigns(
            output_directory=arguments.output_directory,
            canonical_campaign=arguments.canonical_campaign,
            v5_campaign=arguments.v5_campaign,
            v6_campaign=arguments.v6_campaign,
        )
        print(json.dumps({"manifest": str(path), "sha256": sha256_file(path)}, indent=2))
    elif arguments.command == "validate-matrix":
        validate_matrix(load_json(arguments.matrix, "rebuild matrix"))
    elif arguments.command in {"freeze-banks", "freeze-banks-from-campaigns"}:
        if arguments.command == "freeze-banks":
            result = freeze_opening_banks(
                comparison=arguments.comparison,
                output_directory=arguments.output_directory,
                excluded_banks=arguments.exclude_bank,
            )
        else:
            result = freeze_banks_from_campaigns(
                comparison=arguments.comparison,
                output_directory=arguments.output_directory,
                evaluation_directory=arguments.evaluation_directory,
                v5_campaign=arguments.v5_campaign,
                v6_campaign=arguments.v6_campaign,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif arguments.command == "validate-banks":
        validate_opening_banks(
            load_json(arguments.manifest, "rebuild opening banks")
        )
    elif arguments.command == "generate-holdout-candidates":
        print(
            json.dumps(
                generate_holdout_candidate_positions(
                    generator=arguments.generator,
                    output_directory=arguments.output_directory,
                    workers=arguments.workers,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif arguments.command == "freeze-holdout-from-corpus":
        print(
            json.dumps(
                freeze_holdout_from_corpus(
                    candidate_positions=arguments.candidate_positions,
                    candidate_manifest=arguments.candidate_manifest,
                    corpus_manifest=arguments.corpus_manifest,
                    canonical_campaign=arguments.canonical_campaign,
                    v5_campaign=arguments.v5_campaign,
                    v6_campaign=arguments.v6_campaign,
                    output_directory=arguments.output_directory,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif arguments.command == "freeze-holdout":
        print(
            json.dumps(
                freeze_blind_holdout(
                    candidate_positions=arguments.candidate_positions,
                    training_input_receipt=arguments.training_input_receipt,
                    excluded_shards=arguments.exclude_shard,
                    excluded_positions=arguments.exclude_positions,
                    excluded_roots=arguments.exclude_roots,
                    output_directory=arguments.output_directory,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif arguments.command == "freeze-inputs":
        print(
            json.dumps(
                freeze_rebuild_inputs(
                    repository=arguments.repository,
                    corpus_manifest=arguments.corpus_manifest,
                    build_manifest=arguments.build_manifest,
                    matrix_manifest=arguments.matrix_manifest,
                    banks_manifest=arguments.banks_manifest,
                    holdout_manifest=arguments.holdout_manifest,
                    holdout_positions=arguments.holdout_positions,
                    holdout_candidate_manifest=arguments.holdout_candidate_manifest,
                    holdout_candidate_positions=arguments.holdout_candidate_positions,
                    comparison=arguments.comparison,
                    rank4_teacher=arguments.rank4_teacher,
                    incumbent_runtime=arguments.incumbent_runtime,
                    output=arguments.output,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif arguments.command == "write-build-manifest":
        print(
            json.dumps(
                write_rebuild_build_manifest(
                    repository=arguments.repository,
                    expected_commit=arguments.expected_commit,
                    comparison=arguments.comparison,
                    rank4_teacher=arguments.rank4_teacher,
                    holdout_generator=arguments.holdout_generator,
                    output=arguments.output,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif arguments.command == "validate-build-manifest":
        validate_rebuild_build_manifest(arguments.manifest)
    elif arguments.command == "validate-inputs":
        validate_rebuild_inputs(arguments.manifest)
    elif arguments.command == "run":
        print(
            json.dumps(
                run_ladder(
                    inputs_manifest=arguments.inputs,
                    output_directory=arguments.output_directory,
                    resume=arguments.resume,
                    training_workers=arguments.training_workers,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(
            json.dumps(
                qualify_selected_candidate(
                    inputs_manifest=arguments.inputs,
                    selected_receipt=arguments.selected_receipt,
                    output_directory=arguments.output_directory,
                    workers=arguments.workers,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
