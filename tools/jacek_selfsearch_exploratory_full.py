#!/usr/bin/env python3
"""Freeze and run one provenance-bound, local large-teacher campaign.

The interactive ``freeze-full-launch`` command is the only code path that may
inspect Git or the preserved source campaigns.  It imports an explicit,
self-contained input bundle.  The persistent ``run-full`` command validates
only that bundle and local producer hashes before calling the unchanged strict
self-search full phase.
"""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import hashlib
import json
import math
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Mapping, Sequence

import jacek_selfsearch_workflow as selfsearch
from jacek_replay_workflow import artifact_snapshot, canonical_json_bytes


BASE_COMMIT = "c2b06168676bfa0ea7e600da042710b29f3089c5"
TOPIC_BRANCH = "large-teacher-campaign"
CAMPAIGN_ID = "large-teacher-campaign-20260828-v1"
FULL_CAMPAIGN_ID = "large-teacher-full-20260828-v1"
SOURCE_CAMPAIGN_ID = "selfsearch-exploratory-20260827-v1"
SOURCE_PILOT_CAMPAIGN_ID = "selfsearch-exploratory-pilot-20260827-v1"

SOURCE_SUMMARY_SHA256 = (
    "f979a4ad131a26e3b363adf01a8ec3e14bbd474bb832e2565602042bf60b81ef"
)
SOURCE_DECISION_SHA256 = (
    "b6ac366d39b98b7a7d1455708b1e001d1acf5bbbd8053fc267a92040cc26b1b2"
)
TEACHER_ACTOR_SHA256 = (
    "6cafef972aef2b6495ce486b3fb55b9b6b5da8e2593ba0966c2e454e8bfbca86"
)
RETENTION_REFERENCE_SHA256 = (
    "bfcc1755ab9b71261bedc9b9c9b59e38e3d440d7c80e7056a9f0bc812ffc9c80"
)
EXACT_BYPASSED_ERRORS = [
    "matched primary strength gate failed",
    "incumbent primary strength gate failed",
]

BUNDLE_SCHEMA = "papersoccer.large-teacher-input-bundle.v1"
OVERRIDE_SCHEMA = "papersoccer.large-teacher-pilot-override.v1"
LAUNCH_SCHEMA = "papersoccer.large-teacher-full-launch.v1"
RUN_START_SCHEMA = "papersoccer.large-teacher-run-start.v1"
STATUS_SCHEMA = "papersoccer.large-teacher-status.v1"
SUMMARY_SCHEMA = "papersoccer.large-teacher-summary.v1"
ACCEPTANCE_SCHEMA = "papersoccer.teacher-candidate-accepted.v1"
HANDOFF_SCHEMA = "papersoccer.compact-student-handoff.v1"

FULL_SPEC = dataclasses.replace(
    selfsearch.FULL_SPEC,
    campaign_id=FULL_CAMPAIGN_ID,
    configuration={
        **selfsearch.FULL_SPEC.configuration,
        "campaign_id": FULL_CAMPAIGN_ID,
    },
)

RELEVANT_STAGE_RECEIPTS = (
    "12-pack-search",
    "13-pack-rank4",
    "15-train-search",
    "16-train-rank4",
    "17-anchor-metrics",
    "18-opening-bank",
    "19-game-gates",
    "20-latency-audit",
    "21-decision",
)
PANEL_FILE_NAMES = {
    "matched": "primary-matched.json",
    "incumbent": "primary-incumbent.json",
    "rank4": "external-rank4.json",
    "jacek-nn": "external-neural.json",
}
FORBIDDEN_PATH_MARKERS = (
    "sealed-final", "sealed_final", "blind-label", "blind_label",
)


@dataclasses.dataclass(frozen=True)
class ImportArtifact:
    role: str
    source: pathlib.Path
    relative_path: str
    sha256: str
    bytes: int


def _load_json(path: pathlib.Path, label: str) -> dict:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _atomic_json(path: pathlib.Path, value: Mapping[str, object]) -> None:
    selfsearch._atomic_json(path, dict(value))


def _body_hashed(body: Mapping[str, object]) -> dict[str, object]:
    result = dict(body)
    result["body_sha256"] = hashlib.sha256(
        canonical_json_bytes(body)
    ).hexdigest()
    return result


def _verify_body_hash(
    value: Mapping[str, object], *, schema: str, label: str,
) -> None:
    body = dict(value)
    claimed = body.pop("body_sha256", None)
    if (
        body.get("schema") != schema
        or claimed != hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    ):
        raise ValueError(f"{label} body hash is invalid")


def _write_exact_json(
    path: pathlib.Path, value: Mapping[str, object], label: str,
) -> None:
    if path.exists():
        if _load_json(path, label) != dict(value):
            raise ValueError(f"existing {label} differs from frozen content")
        return
    _atomic_json(path, value)


def _is_snapshot(value: object) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("path"), str)
        and isinstance(value.get("sha256"), str)
        and type(value.get("bytes")) is int
    )


def _snapshot_matches(record: object, *, expected: pathlib.Path | None = None) -> bool:
    if not _is_snapshot(record):
        return False
    assert isinstance(record, dict)
    path = pathlib.Path(record["path"]).resolve()
    if expected is not None and path != expected.resolve():
        return False
    try:
        current = artifact_snapshot(path)
    except (OSError, ValueError):
        return False
    return all(current.get(key) == value for key, value in record.items())


def _assert_allowed_source(path: pathlib.Path) -> pathlib.Path:
    resolved = path.resolve()
    lowered = str(resolved).lower()
    if (
        any(marker in lowered for marker in FORBIDDEN_PATH_MARKERS)
        or ("blind" in lowered and "label" in lowered)
    ):
        raise ValueError(f"protected artifact is outside the import allowlist: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"frozen input is missing: {resolved}")
    return resolved


def _under(path: pathlib.Path, root: pathlib.Path, label: str) -> pathlib.Path:
    path = path.resolve()
    root = root.resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes its frozen source root") from error
    return path


def _record_artifact(
    *, role: str, source: pathlib.Path, relative_path: str,
    expected: Mapping[str, object] | None = None,
) -> ImportArtifact:
    source = _assert_allowed_source(source)
    if expected is not None and not _snapshot_matches(expected, expected=source):
        raise ValueError(f"{role} differs from its frozen snapshot")
    snapshot = artifact_snapshot(source)
    if expected is not None and (
        snapshot["sha256"] != expected.get("sha256")
        or snapshot["bytes"] != expected.get("bytes")
    ):
        raise ValueError(f"{role} content binding changed")
    relative = pathlib.PurePosixPath(relative_path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"invalid bundle path for {role}")
    return ImportArtifact(
        role=role,
        source=source,
        relative_path=relative.as_posix(),
        sha256=str(snapshot["sha256"]),
        bytes=int(snapshot["bytes"]),
    )


def _full_spec_record() -> dict[str, object]:
    expected = dataclasses.asdict(selfsearch.FULL_SPEC)
    actual = dataclasses.asdict(FULL_SPEC)
    expected["campaign_id"] = FULL_CAMPAIGN_ID
    expected_configuration = dict(expected["configuration"])
    expected_configuration["campaign_id"] = FULL_CAMPAIGN_ID
    expected["configuration"] = expected_configuration
    if actual != expected:
        raise ValueError("large-teacher FULL_SPEC differs beyond campaign identity")
    configuration = actual["configuration"]
    if (
        sum(actual["quotas"].values()) != 10_000
        or configuration.get("games") != 10_000
        or configuration.get("positions_per_game") != 20
        or configuration.get("bfm_shallow_tree_nodes") != 64_000
        or configuration.get("bfm_deep_tree_nodes") != 500_000
        or configuration.get("rank4_shallow_nodes") != 32_000
        or configuration.get("rank4_deep_nodes") != 400_000
        or configuration.get("adjudicator_positions") != 4_000
        or configuration.get("adjudicator_tree_nodes") != 1_000_000
        or configuration.get("training_seeds")
        != [20260904, 20260905, 20260906]
        or actual["game_seed"] != 2026082503
        or actual["opening_seed"] != 2026082507
        or actual["pairs"] != 500
        or actual["gate_time_ms"] != 980
        or actual["bank_classification"] != "final"
    ):
        raise ValueError("large-teacher FULL_SPEC constants changed")
    return actual


def validate_teacher_only_override(
    decision: Mapping[str, object],
) -> dict[str, object]:
    """Allow exactly the two recorded pilot primary-strength failures."""

    if (
        decision.get("schema") != selfsearch.PILOT_DECISION_SCHEMA
        or decision.get("eligible_for_full") is not False
        or decision.get("errors") != EXACT_BYPASSED_ERRORS
    ):
        raise ValueError("pilot override requires the exact two recorded errors")
    counts = decision.get("counts")
    if not isinstance(counts, dict) or set(counts) != {
        "matched", "incumbent", "rank4", "jacek-nn",
    }:
        raise ValueError("pilot override panel coverage is incomplete")
    for name, record in counts.items():
        if (
            not isinstance(record, dict)
            or record.get("games") != 600
            or record.get("illegal") != 0
            or record.get("unfinished") != 0
            or not isinstance(record.get("colors"), list)
            or len(record["colors"]) != 2
        ):
            raise ValueError("pilot override rejects legality or completion failures")
        threshold = 325 if name in {"matched", "incumbent"} else 306
        color_threshold = 156 if name in {"matched", "incumbent"} else 143
        failed = (
            int(record.get("wins", -1)) < threshold
            or min(map(int, record["colors"])) < color_threshold
        )
        if name in {"matched", "incumbent"} and not failed:
            raise ValueError("recorded primary-strength failure no longer exists")
        if name in {"rank4", "jacek-nn"} and failed:
            raise ValueError("pilot override rejects external-strength failures")

    p99 = decision.get("candidate_p99_ms")
    uncontended = decision.get("uncontended_max_ms")
    if (
        isinstance(p99, bool)
        or not isinstance(p99, (int, float))
        or not math.isfinite(float(p99))
        or float(p99) < 0.0
        or float(p99) > 25.0
        or isinstance(uncontended, bool)
        or not isinstance(uncontended, (int, float))
        or not math.isfinite(float(uncontended))
        or float(uncontended) < 0.0
        or float(uncontended) >= 1_000.0
    ):
        raise ValueError("pilot override rejects p99 or uncontended-latency failures")

    candidate = decision.get("anchor_candidate")
    incumbent = decision.get("anchor_incumbent")
    if not isinstance(candidate, dict) or not isinstance(incumbent, dict):
        raise ValueError("pilot override retention evidence is incomplete")
    try:
        candidate_sign = float(candidate["sign_accuracy"])
        incumbent_sign = float(incumbent["sign_accuracy"])
        candidate_huber = float(candidate["weighted_huber"])
        incumbent_huber = float(incumbent["weighted_huber"])
        retention_values = (
            candidate_sign, incumbent_sign, candidate_huber, incumbent_huber,
        )
        retention_failed = (
            not all(math.isfinite(value) for value in retention_values)
            or candidate_sign < incumbent_sign - 0.005
            or candidate_huber > incumbent_huber * 1.02
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("pilot override retention evidence is malformed") from error
    if retention_failed:
        raise ValueError("pilot override rejects canonical retention failures")
    return {
        "bypassed_errors": list(EXACT_BYPASSED_ERRORS),
        "pilot_passed": False,
        "pilot_20_ms_passed": False,
        "pilot_gate_time_ms": 20,
        "candidate_p99_ms": float(p99),
        "uncontended_max_ms": float(uncontended),
        "legality_bypassed": False,
        "completion_bypassed": False,
        "external_strength_bypassed": False,
        "retention_bypassed": False,
        "p99_bypassed": False,
        "uncontended_latency_bypassed": False,
        "scope": "large-teacher-data-generation-only",
    }


def _validate_stage_receipt(
    path: pathlib.Path, *, stage: str,
) -> dict[str, object]:
    receipt = _load_json(path, f"pilot {stage} receipt")
    if (
        receipt.get("schema")
        != "papersoccer.jacek-replay-bfm-stage-receipt.v1"
        or receipt.get("campaign_id") != SOURCE_PILOT_CAMPAIGN_ID
        or receipt.get("round") != 0
        or receipt.get("stage") != stage.split("-", 1)[1]
    ):
        raise ValueError(f"pilot {stage} receipt is malformed")
    return receipt


def _append_unique(
    artifacts: list[ImportArtifact], artifact: ImportArtifact,
) -> None:
    if any(item.role == artifact.role for item in artifacts):
        raise ValueError(f"duplicate input role: {artifact.role}")
    if any(item.relative_path == artifact.relative_path for item in artifacts):
        raise ValueError(f"duplicate bundle path: {artifact.relative_path}")
    artifacts.append(artifact)


def _manifest_artifacts(
    *, role_prefix: str, manifest_path: pathlib.Path,
    manifest_relative: str, npz_relative: str,
    expected: Mapping[str, object],
) -> tuple[ImportArtifact, ImportArtifact, dict[str, object]]:
    manifest_path = _assert_allowed_source(manifest_path)
    manifest_artifact = _record_artifact(
        role=f"{role_prefix}-manifest",
        source=manifest_path,
        relative_path=manifest_relative,
        expected=expected,
    )
    if manifest_path.stem != manifest_artifact.sha256:
        raise ValueError(f"{role_prefix} manifest is not content-addressed")
    manifest = _load_json(manifest_path, f"{role_prefix} shard manifest")
    npz_name = manifest.get("npz")
    npz_sha256 = manifest.get("npz_sha256")
    if (
        manifest.get("schema") != "papersoccer.jacek-replay-csr-shard.v1"
        or manifest.get("split") not in {"train", "validation", "test"}
        or not isinstance(npz_name, str)
        or pathlib.PurePath(npz_name).name != npz_name
        or not npz_name.endswith(".npz")
        or not isinstance(npz_sha256, str)
        or pathlib.Path(npz_name).stem != npz_sha256
    ):
        raise ValueError(f"{role_prefix} shard manifest is malformed")
    npz_path = _assert_allowed_source(manifest_path.parent / npz_name)
    npz_artifact = _record_artifact(
        role=f"{role_prefix}-npz",
        source=npz_path,
        relative_path=npz_relative,
    )
    if npz_artifact.sha256 != npz_sha256:
        raise ValueError(f"{role_prefix} adjacent NPZ binding changed")
    return manifest_artifact, npz_artifact, manifest


def build_import_plan(
    *, source_pilot: pathlib.Path, canonical_campaign: pathlib.Path,
) -> tuple[
    list[ImportArtifact], dict[str, object], dict[str, object], dict[str, object]
]:
    """Validate the exact source evidence and return one explicit copy plan."""

    source_pilot = source_pilot.resolve()
    canonical_campaign = canonical_campaign.resolve()
    if source_pilot.name != SOURCE_CAMPAIGN_ID:
        raise ValueError(f"source pilot must be named {SOURCE_CAMPAIGN_ID}")
    if canonical_campaign.name != "canonical-20260823-v1":
        raise ValueError("canonical campaign identity changed")
    if (source_pilot / "full").exists():
        raise ValueError("source pilot unexpectedly contains a full campaign")

    summary_path = source_pilot / "final-summary.json"
    decision_path = source_pilot / "pilot/decision.json"
    if selfsearch.sha256(summary_path) != SOURCE_SUMMARY_SHA256:
        raise ValueError("source pilot summary differs from the recorded SHA-256")
    if selfsearch.sha256(decision_path) != SOURCE_DECISION_SHA256:
        raise ValueError("source pilot decision differs from the recorded SHA-256")
    summary = _load_json(summary_path, "source pilot summary")
    decision = _load_json(decision_path, "source pilot decision")
    pilot = summary.get("pilot")
    if (
        summary.get("schema")
        != "papersoccer.jacek-selfsearch-exploratory-summary.v1"
        or summary.get("campaign_id") != SOURCE_CAMPAIGN_ID
        or summary.get("terminal") != "pilot-rejected"
        or summary.get("full_continuation_eligible") is not False
        or summary.get("full_started") is not False
        or summary.get("full_continuation_receipt") is not None
        or summary.get("sealed_final_bank_opened") is not False
        or summary.get("blind_holdout_labels_opened") is not False
        or summary.get("canonical_incumbent_replaced") is not False
        or summary.get("canonical_promotion_eligible") is not False
        or summary.get("external_upload") is not False
        or summary.get("rank4_replaced") is not False
        or not isinstance(pilot, dict)
        or pilot.get("profile") != "pilot"
        or pilot.get("campaign_id") != SOURCE_PILOT_CAMPAIGN_ID
        or pilot.get("decision") != decision
        or pathlib.Path(str(pilot.get("decision_path", ""))).resolve()
        != decision_path.resolve()
    ):
        raise ValueError("source pilot is not the exact rejected pilot")
    override_truth = validate_teacher_only_override(decision)

    launch_path = source_pilot / "exploratory-launch.json"
    lineage_path = source_pilot / "exploratory-lineage.json"
    if not _snapshot_matches(summary.get("launch"), expected=launch_path):
        raise ValueError("source pilot launch binding changed")
    if not _snapshot_matches(summary.get("lineage"), expected=lineage_path):
        raise ValueError("source pilot lineage binding changed")
    launch = _load_json(launch_path, "source pilot launch")
    lineage = _load_json(lineage_path, "source pilot lineage")
    _verify_body_hash(
        launch,
        schema="papersoccer.jacek-selfsearch-exploratory-launch.v1",
        label="source pilot launch",
    )
    _verify_body_hash(
        lineage,
        schema="papersoccer.jacek-selfsearch-exploratory-lineage.v1",
        label="source pilot lineage",
    )
    if (
        launch.get("campaign_id") != SOURCE_CAMPAIGN_ID
        or launch.get("lineage_record") != lineage
        or launch.get("pilot_games") != 2_000
        or launch.get("automatic_full_launch") is not False
        or launch.get("full_directory_creation") is not False
        or launch.get("sealed_final_untouched", {}).get("opened") is not False
        or launch.get("sealed_final_untouched", {}).get("used_as_exclusion")
        is not False
    ):
        raise ValueError("source pilot launch policy changed")

    artifacts: list[ImportArtifact] = []
    routes: dict[str, object] = {}
    for role, source, relative in (
        ("pilot-launch", launch_path, "pilot/source/launch.json"),
        ("pilot-lineage", lineage_path, "pilot/source/lineage.json"),
        ("pilot-summary", summary_path, "pilot/source/summary.json"),
        ("pilot-decision", decision_path, "pilot/evidence/decision.json"),
    ):
        _append_unique(
            artifacts,
            _record_artifact(role=role, source=source, relative_path=relative),
        )

    actor_path = pathlib.Path(str(pilot.get("search_runtime", ""))).resolve()
    actor_manifest_path = pathlib.Path(
        str(pilot.get("search_manifest", ""))
    ).resolve()
    expected_actor_path = source_pilot / "pilot/models/search/jacek_replay_bfm.runtime"
    expected_actor_manifest = pathlib.Path(str(expected_actor_path) + ".json")
    if actor_path != expected_actor_path.resolve() or actor_manifest_path != expected_actor_manifest.resolve():
        raise ValueError("source pilot selected actor path changed")
    actor_artifact = _record_artifact(
        role="pilot-teacher-actor",
        source=actor_path,
        relative_path="pilot/actor/jacek_replay_bfm.runtime",
    )
    if actor_artifact.sha256 != TEACHER_ACTOR_SHA256:
        raise ValueError("source pilot selected Search runtime changed")
    actor_manifest_artifact = _record_artifact(
        role="pilot-teacher-manifest",
        source=actor_manifest_path,
        relative_path="pilot/actor/jacek_replay_bfm.runtime.json",
    )
    actor_manifest = _load_json(actor_manifest_path, "source pilot actor manifest")
    runtime_record = actor_manifest.get("runtime")
    if (
        actor_manifest.get("schema") != "papersoccer.jacek-replay-bfm-model.v2"
        or actor_manifest.get("status") != "research-candidate-not-game-gated"
        or not isinstance(runtime_record, dict)
        or runtime_record.get("path") != "jacek_replay_bfm.runtime"
        or runtime_record.get("artifact_sha256") != TEACHER_ACTOR_SHA256
        or runtime_record.get("bytes") != actor_artifact.bytes
    ):
        raise ValueError("source pilot actor manifest is malformed")
    _append_unique(artifacts, actor_artifact)
    _append_unique(artifacts, actor_manifest_artifact)
    routes["actor"] = actor_artifact.relative_path

    reference_path = source_pilot / "inputs/v6-experimental-reference.runtime"
    reference_artifact = _record_artifact(
        role="original-retention-reference",
        source=reference_path,
        relative_path="pilot/reference/original-retention.runtime",
    )
    if reference_artifact.sha256 != RETENTION_REFERENCE_SHA256:
        raise ValueError("v6 original-retention reference changed")
    if (
        summary.get("pilot_actor_snapshot", {}).get("sha256")
        != RETENTION_REFERENCE_SHA256
        or summary.get("experimental_reference_actor", {}).get("sha256")
        != RETENTION_REFERENCE_SHA256
    ):
        raise ValueError("source summary does not bind the v6 reference")
    _append_unique(artifacts, reference_artifact)
    routes["diversity_reference"] = reference_artifact.relative_path
    routes["original_retention_reference"] = reference_artifact.relative_path

    anchor_path = source_pilot / "pilot/anchor-metrics.json"
    latency_path = source_pilot / "pilot/latency-audit.json"
    anchor = _load_json(anchor_path, "source pilot anchor metrics")
    latency = _load_json(latency_path, "source pilot latency audit")
    if (
        anchor.get("candidate_metrics") != decision.get("anchor_candidate")
        or anchor.get("incumbent_metrics") != decision.get("anchor_incumbent")
        or latency.get("summary", {}).get("candidate", {}).get("max_ms")
        != decision.get("uncontended_max_ms")
    ):
        raise ValueError("source pilot retention or latency evidence changed")
    _append_unique(
        artifacts,
        _record_artifact(
            role="pilot-anchor-metrics", source=anchor_path,
            relative_path="pilot/evidence/anchor-metrics.json",
        ),
    )
    _append_unique(
        artifacts,
        _record_artifact(
            role="pilot-latency-audit", source=latency_path,
            relative_path="pilot/evidence/latency-audit.json",
        ),
    )

    report_paths: dict[str, pathlib.Path] = {}
    for name, file_name in PANEL_FILE_NAMES.items():
        report_path = source_pilot / f"pilot/game-gates/{name}.json"
        record = decision.get("reports", {}).get(name)
        if not _snapshot_matches(record, expected=report_path):
            raise ValueError(f"source pilot {name} gate report changed")
        report = _load_json(report_path, f"source pilot {name} gate report")
        if report.get("configuration", {}).get("opening_bank_classification") != "development":
            raise ValueError("source pilot gate did not use its development bank")
        report_paths[name] = report_path
        _append_unique(
            artifacts,
            _record_artifact(
                role=f"pilot-gate-report-{name}", source=report_path,
                relative_path=f"pilot/evidence/reports/{file_name}", expected=record,
            ),
        )

    recomputed = selfsearch.pilot_decision(
        matched_report=report_paths["matched"],
        incumbent_report=report_paths["incumbent"],
        rank4_report=report_paths["rank4"],
        jacek_nn_report=report_paths["jacek-nn"],
        anchor_candidate=anchor["candidate_metrics"],
        anchor_incumbent=anchor["incumbent_metrics"],
        uncontended_max_ms=float(decision["uncontended_max_ms"]),
    )
    if recomputed != decision:
        raise ValueError("source pilot decision does not replay from bound evidence")

    receipts: dict[str, dict[str, object]] = {}
    for stage in RELEVANT_STAGE_RECEIPTS:
        receipt_path = source_pilot / f"pilot/receipts/{stage}.json"
        receipts[stage] = _validate_stage_receipt(receipt_path, stage=stage)
        _append_unique(
            artifacts,
            _record_artifact(
                role=f"pilot-stage-receipt-{stage}", source=receipt_path,
                relative_path=f"pilot/evidence/receipts/{stage}.json",
            ),
        )
    if (
        receipts["21-decision"].get("result") != decision
        or not _snapshot_matches(
            receipts["21-decision"].get("outputs", {}).get("decision"),
            expected=decision_path,
        )
        or receipts["20-latency-audit"].get("result", {}).get("candidate_max_ms")
        != decision.get("uncontended_max_ms")
        or not _snapshot_matches(
            receipts["15-train-search"].get("outputs", {}).get("runtime"),
            expected=actor_path,
        )
        or not _snapshot_matches(
            receipts["15-train-search"].get("outputs", {}).get("manifest"),
            expected=actor_manifest_path,
        )
        or receipts["15-train-search"].get("result") != actor_manifest
    ):
        raise ValueError("source pilot decision or actor receipt binding changed")

    stage19_result = receipts["19-game-gates"].get("result")
    if not isinstance(stage19_result, dict):
        raise ValueError("source pilot game-gate receipt is malformed")
    for kind, source_subdir, target_subdir in (
        ("panel_shards", "pilot/game-gates/shards", "pilot/evidence/gate-shards"),
        (
            "panel_receipts", "pilot/receipts/19-game-gates-panels",
            "pilot/evidence/gate-receipts",
        ),
    ):
        records = stage19_result.get(kind)
        if not isinstance(records, list) or len(records) != 240:
            raise ValueError(f"source pilot {kind} coverage is incomplete")
        seen_paths: set[pathlib.Path] = set()
        for index, record in enumerate(records):
            if not _is_snapshot(record):
                raise ValueError(f"source pilot {kind} entry is malformed")
            assert isinstance(record, dict)
            source = _under(
                pathlib.Path(record["path"]), source_pilot / source_subdir,
                f"pilot {kind}",
            )
            if source in seen_paths or not _snapshot_matches(record, expected=source):
                raise ValueError(f"source pilot {kind} binding changed")
            seen_paths.add(source)
            suffix = source.relative_to(source_pilot / source_subdir).as_posix()
            _append_unique(
                artifacts,
                _record_artifact(
                    role=f"pilot-{kind}-{index:03d}", source=source,
                    relative_path=f"{target_subdir}/{suffix}", expected=record,
                ),
            )

    pilot_manifest_routes: dict[str, list[str]] = {"search": [], "rank4": []}
    pilot_inline_manifests: dict[str, list[dict[str, object]]] = {
        "search": [], "rank4": [],
    }
    for arm, summary_key in (
        ("search", "search_new_manifests"),
        ("rank4", "rank4_new_manifests"),
    ):
        raw_paths = pilot.get(summary_key)
        if not isinstance(raw_paths, list) or len(raw_paths) != 3:
            raise ValueError(f"source pilot {arm} shard roster is incomplete")
        splits: list[str] = []
        for index, raw_path in enumerate(raw_paths):
            manifest_path = _under(
                pathlib.Path(str(raw_path)), source_pilot / f"pilot/shards/{arm}",
                f"pilot {arm} shard",
            )
            manifest_expected = artifact_snapshot(manifest_path)
            manifest_relative = f"pilot/shards/{arm}/{manifest_path.name}"
            manifest_artifact, npz_artifact, manifest = _manifest_artifacts(
                role_prefix=f"pilot-{arm}-{index}",
                manifest_path=manifest_path,
                manifest_relative=manifest_relative,
                npz_relative="pending-adjacent-npz",
                expected=manifest_expected,
            )
            # The adjacent NPZ name is known only after reading the manifest.
            npz_artifact = dataclasses.replace(
                npz_artifact,
                relative_path=f"pilot/shards/{arm}/{pathlib.Path(npz_artifact.source).name}",
            )
            _append_unique(artifacts, manifest_artifact)
            _append_unique(artifacts, npz_artifact)
            pilot_manifest_routes[arm].append(manifest_artifact.relative_path)
            pilot_inline_manifests[arm].append(manifest)
            splits.append(str(manifest.get("split")))
        if splits != ["train", "validation", "test"]:
            raise ValueError(f"source pilot {arm} shard routing changed")
        pack_report_path = source_pilot / f"pilot/shards/{arm}/pack-report.json"
        pack_report = _load_json(pack_report_path, f"source pilot {arm} pack report")
        pack_stage = f"{'12' if arm == 'search' else '13'}-pack-{arm}"
        pack_receipt = receipts[pack_stage]
        if (
            [pack_report.get("shards", {}).get(split) for split in splits]
            != [
                pack_receipt.get("result", {}).get("shards", {}).get(split)
                for split in splits
            ]
            or not _snapshot_matches(
                pack_receipt.get("outputs", {}).get("report"),
                expected=pack_report_path,
            )
        ):
            raise ValueError(f"source pilot {arm} pack receipt changed")
        for index, split in enumerate(splits):
            record = pack_report.get("shards", {}).get(split)
            manifest_artifact = next(
                item for item in artifacts
                if item.role == f"pilot-{arm}-{index}-manifest"
            )
            npz_artifact = next(
                item for item in artifacts
                if item.role == f"pilot-{arm}-{index}-npz"
            )
            if (
                not isinstance(record, dict)
                or pathlib.Path(str(record.get("manifest", ""))).resolve()
                != manifest_artifact.source
                or record.get("manifest_sha256") != manifest_artifact.sha256
                or pathlib.Path(str(record.get("npz", ""))).resolve()
                != npz_artifact.source
                or record.get("sha256") != npz_artifact.sha256
                or record.get("samples")
                != pilot_inline_manifests[arm][index].get("samples")
            ):
                raise ValueError(f"source pilot {arm} {split} shard binding changed")
        _append_unique(
            artifacts,
            _record_artifact(
                role=f"pilot-{arm}-pack-report", source=pack_report_path,
                relative_path=f"pilot/evidence/pack/{arm}.json",
            ),
        )
    routes["pilot_search_manifests"] = pilot_manifest_routes["search"]
    routes["pilot_rank4_manifests"] = pilot_manifest_routes["rank4"]
    if actor_manifest.get("source_shards", [None])[0] != pilot_inline_manifests["search"][0]:
        raise ValueError("source pilot actor omits its selected Search training shard")

    roots = launch.get("roots")
    if not isinstance(roots, dict):
        raise ValueError("source launch canonical roots are missing")
    roots_tsv = canonical_campaign / "round-2/teacher-input.tsv"
    roots_manifest = canonical_campaign / "round-2/replay-roots.json"
    for role, key, source, relative in (
        ("canonical-roots-tsv", "tsv", roots_tsv, "canonical/roots/teacher-input.tsv"),
        (
            "canonical-roots-manifest", "manifest", roots_manifest,
            "canonical/roots/replay-roots.json",
        ),
    ):
        record = roots.get(key)
        artifact = _record_artifact(
            role=role, source=source, relative_path=relative, expected=record,
        )
        _append_unique(artifacts, artifact)
        routes["roots_tsv" if key == "tsv" else "roots_manifest"] = relative

    split_records = launch.get("splits")
    if not isinstance(split_records, dict) or set(split_records) != {
        "train", "validation", "test",
    }:
        raise ValueError("source launch canonical split roster is malformed")
    canonical_routes: dict[str, list[str]] = {
        "train": [], "validation": [], "test": [],
    }
    canonical_records_by_round: dict[tuple[int, str], object] = {}
    for split in ("train", "validation", "test"):
        records = split_records.get(split)
        if not isinstance(records, list) or len(records) != 3:
            raise ValueError(f"canonical {split} shard roster is incomplete")
        for round_index, record in enumerate(records):
            if not _is_snapshot(record):
                raise ValueError(f"canonical {split} shard binding is malformed")
            assert isinstance(record, dict)
            manifest_path = _under(
                pathlib.Path(record["path"]),
                canonical_campaign / f"round-{round_index}/shards",
                f"canonical Round-{round_index} {split} shard",
            )
            manifest_relative = f"canonical/shards/{manifest_path.name}"
            manifest_artifact, npz_artifact, manifest = _manifest_artifacts(
                role_prefix=f"canonical-r{round_index}-{split}",
                manifest_path=manifest_path,
                manifest_relative=manifest_relative,
                npz_relative="pending-adjacent-npz",
                expected=record,
            )
            npz_artifact = dataclasses.replace(
                npz_artifact,
                relative_path=f"canonical/shards/{pathlib.Path(npz_artifact.source).name}",
            )
            if manifest.get("split") != split:
                raise ValueError("canonical shard split routing changed")
            _append_unique(artifacts, manifest_artifact)
            _append_unique(artifacts, npz_artifact)
            canonical_routes[split].append(manifest_relative)
            canonical_records_by_round[(round_index, split)] = record
    expected_prior = [
        canonical_records_by_round[(round_index, split)]
        for round_index in range(3)
        for split in ("train", "validation", "test")
    ]
    if launch.get("canonical_prior_manifests") != expected_prior:
        raise ValueError("canonical cumulative manifest order changed")
    routes["canonical_splits"] = canonical_routes
    routes["canonical_prior_manifests"] = [
        canonical_routes[split][round_index]
        for round_index in range(3)
        for split in ("train", "validation", "test")
    ]

    exclusions = launch.get("opening_exclusions")
    if not isinstance(exclusions, list) or len(exclusions) != 6:
        raise ValueError("source launch already-open exclusion roster changed")
    exclusion_routes: list[str] = []
    for index, record in enumerate(exclusions):
        if not _is_snapshot(record):
            raise ValueError("source launch opening exclusion is malformed")
        assert isinstance(record, dict)
        source = pathlib.Path(record["path"])
        relative = f"opening-exclusions/bank-{index:03d}.tsv"
        artifact = _record_artifact(
            role=f"opening-exclusion-{index:03d}", source=source,
            relative_path=relative, expected=record,
        )
        _append_unique(artifacts, artifact)
        exclusion_routes.append(relative)
    pilot_bank_path = source_pilot / "pilot/gate-openings.tsv"
    pilot_bank_record = receipts["18-opening-bank"].get("outputs", {}).get("bank")
    pilot_bank_relative = "opening-exclusions/bank-006.tsv"
    pilot_bank_artifact = _record_artifact(
        role="opening-exclusion-pilot", source=pilot_bank_path,
        relative_path=pilot_bank_relative, expected=pilot_bank_record,
    )
    gate_inputs = receipts["19-game-gates"].get("inputs", {})
    if (
        not _snapshot_matches(gate_inputs.get("bank"), expected=pilot_bank_path)
        or not _snapshot_matches(gate_inputs.get("model"), expected=actor_path)
        or receipts["18-opening-bank"].get("result", {}).get("exclusions")
        != exclusions
    ):
        raise ValueError("source pilot gate bank ancestry changed")
    for report_path in report_paths.values():
        report = _load_json(report_path, "source pilot gate report")
        if (
            report.get("model_sha256") != TEACHER_ACTOR_SHA256
            or report.get("configuration", {}).get("opening_bank_sha256")
            != pilot_bank_artifact.sha256
        ):
            raise ValueError("source pilot gate report actor/bank binding changed")
    _append_unique(artifacts, pilot_bank_artifact)
    exclusion_routes.append(pilot_bank_relative)
    routes["opening_exclusions"] = exclusion_routes

    source_fingerprints: dict[str, object] = {
        "source_summary_sha256": SOURCE_SUMMARY_SHA256,
        "source_decision_sha256": SOURCE_DECISION_SHA256,
        "teacher_actor_sha256": TEACHER_ACTOR_SHA256,
        "original_retention_reference_sha256": RETENTION_REFERENCE_SHA256,
        "source_launch_sha256": selfsearch.sha256(launch_path),
        "source_lineage_sha256": selfsearch.sha256(lineage_path),
        "source_pilot_artifacts": len(
            [item for item in artifacts if item.role.startswith("pilot-")]
        ),
        "canonical_manifest_count": 9,
        "pilot_manifest_count": 6,
        "opening_exclusion_count": 7,
    }
    return artifacts, routes, override_truth, source_fingerprints


def _copy_import_artifact(artifact: ImportArtifact, staging: pathlib.Path) -> None:
    target = staging / artifact.relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    selfsearch._copy_atomic(artifact.source, target)
    if (
        target.stat().st_size != artifact.bytes
        or selfsearch.sha256(target) != artifact.sha256
    ):
        raise ValueError(f"input changed while importing {artifact.role}")


def _local_artifact_record(
    *, role: str, path: pathlib.Path, root: pathlib.Path,
) -> dict[str, object]:
    path = path.resolve()
    root = root.resolve()
    relative = path.relative_to(root).as_posix()
    return {
        "role": role,
        "relative_path": relative,
        "sha256": selfsearch.sha256(path),
        "bytes": path.stat().st_size,
    }


def _safe_local_path(root: pathlib.Path, relative_path: object) -> pathlib.Path:
    if not isinstance(relative_path, str):
        raise ValueError("bundle route is not a string")
    relative = pathlib.PurePosixPath(relative_path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("bundle route is unsafe")
    root = root.resolve()
    path = (root / relative.as_posix()).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("bundle route escapes the input bundle") from error
    return path


def _route_strings(value: object) -> list[str]:
    result: list[str] = []
    if isinstance(value, str):
        result.append(value)
    elif isinstance(value, list):
        for item in value:
            result.extend(_route_strings(item))
    elif isinstance(value, dict):
        for item in value.values():
            result.extend(_route_strings(item))
    else:
        raise ValueError("bundle route map is malformed")
    return result


def _override_body(override_truth: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema": OVERRIDE_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "source_campaign_id": SOURCE_CAMPAIGN_ID,
        "source_pilot_campaign_id": SOURCE_PILOT_CAMPAIGN_ID,
        "source_summary_sha256": SOURCE_SUMMARY_SHA256,
        "source_decision_sha256": SOURCE_DECISION_SHA256,
        "teacher_actor_sha256": TEACHER_ACTOR_SHA256,
        "original_retention_reference_sha256": RETENTION_REFERENCE_SHA256,
        "pilot_decision_eligible_for_full": False,
        "pilot_truth": dict(override_truth),
        "teacher_only_launch_authorized": True,
        "student_training_authorized_before_full_acceptance": False,
        "full_specification": _full_spec_record(),
        "sealed_final_opening_bank_accessed": False,
        "blind_labels_accessed": False,
        "canonical_promotion_eligible": False,
        "publication": False,
        "external_upload": False,
        "replace_rank4": False,
        "leaderboard_claim": False,
    }


def validate_input_bundle(manifest_path: pathlib.Path) -> dict[str, object]:
    """Validate only copied files; archival JSON payloads remain opaque bytes."""

    manifest_path = manifest_path.resolve()
    bundle_root = manifest_path.parent
    manifest = _load_json(manifest_path, "large-teacher input bundle")
    _verify_body_hash(manifest, schema=BUNDLE_SCHEMA, label="input bundle")
    body = dict(manifest)
    body.pop("body_sha256", None)
    artifacts = body.get("artifacts")
    routes = body.get("routes")
    override_record = body.get("teacher_only_override")
    if (
        set(body) != {
            "schema", "campaign_id", "source_fingerprints", "routes",
            "teacher_only_override", "full_specification", "artifacts",
            "atomic_import", "explicit_allowlist", "runtime_uses_source_paths",
            "sealed_final_opening_bank_accessed", "blind_labels_accessed",
        }
        or body.get("campaign_id") != CAMPAIGN_ID
        or body.get("full_specification") != _full_spec_record()
        or body.get("source_fingerprints", {}).get("source_summary_sha256")
        != SOURCE_SUMMARY_SHA256
        or body.get("source_fingerprints", {}).get("source_decision_sha256")
        != SOURCE_DECISION_SHA256
        or body.get("source_fingerprints", {}).get("teacher_actor_sha256")
        != TEACHER_ACTOR_SHA256
        or body.get("source_fingerprints", {}).get(
            "original_retention_reference_sha256"
        ) != RETENTION_REFERENCE_SHA256
        or body.get("sealed_final_opening_bank_accessed") is not False
        or body.get("blind_labels_accessed") is not False
        or body.get("runtime_uses_source_paths") is not False
        or body.get("atomic_import") is not True
        or body.get("explicit_allowlist") is not True
        or not isinstance(artifacts, list)
        or not artifacts
        or not isinstance(routes, dict)
        or not isinstance(override_record, dict)
    ):
        raise ValueError("large-teacher input bundle policy is invalid")

    by_relative: dict[str, dict[str, object]] = {}
    roles: set[str] = set()
    for record in artifacts:
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
            or record["relative_path"] in by_relative
        ):
            raise ValueError("large-teacher input artifact registry is malformed")
        lowered = record["relative_path"].lower()
        if (
            any(marker in lowered for marker in FORBIDDEN_PATH_MARKERS)
            or ("blind" in lowered and "label" in lowered)
        ):
            raise ValueError("protected path entered the copied bundle")
        path = _safe_local_path(bundle_root, record["relative_path"])
        if (
            not path.is_file()
            or path.stat().st_size != record["bytes"]
            or selfsearch.sha256(path) != record["sha256"]
        ):
            raise ValueError(f"copied input changed: {record['relative_path']}")
        roles.add(record["role"])
        by_relative[record["relative_path"]] = record

    actual_files = {
        path.relative_to(bundle_root).as_posix()
        for path in bundle_root.rglob("*")
        if path.is_file()
    }
    if actual_files != {*by_relative, manifest_path.name}:
        raise ValueError("large-teacher input bundle inventory is not exact")
    for relative in _route_strings(routes):
        if relative not in by_relative:
            raise ValueError("bundle route is not backed by a copied artifact")

    required_roles = {
        "pilot-launch", "pilot-lineage", "pilot-summary", "pilot-decision",
        "pilot-teacher-actor", "pilot-teacher-manifest",
        "original-retention-reference", "canonical-roots-tsv",
        "canonical-roots-manifest", "opening-exclusion-pilot",
        "teacher-only-override",
    }
    if not required_roles <= roles:
        raise ValueError("large-teacher input bundle omits a required role")
    if sum(role.startswith("pilot-search-") and role.endswith("-manifest")
           for role in roles) != 3:
        raise ValueError("large-teacher input bundle omits pilot Search shards")
    if sum(role.startswith("pilot-rank4-") and role.endswith("-manifest")
           for role in roles) != 3:
        raise ValueError("large-teacher input bundle omits pilot Rank-4 shards")
    if sum(
        role.startswith(("canonical-r0-", "canonical-r1-", "canonical-r2-"))
        and role.endswith("-manifest")
        for role in roles
    ) != 9:
        raise ValueError("large-teacher input bundle omits canonical shards")
    if sum(role.startswith("opening-exclusion-") for role in roles) != 7:
        raise ValueError("large-teacher input bundle opening exclusions changed")

    actor_record = by_relative.get(str(routes.get("actor")))
    reference_record = by_relative.get(str(routes.get("original_retention_reference")))
    if (
        actor_record is None
        or actor_record.get("sha256") != TEACHER_ACTOR_SHA256
        or reference_record is None
        or reference_record.get("sha256") != RETENTION_REFERENCE_SHA256
        or routes.get("diversity_reference")
        != routes.get("original_retention_reference")
    ):
        raise ValueError("large-teacher actor/reference routing changed")

    manifest_routes = [
        *routes.get("pilot_search_manifests", []),
        *routes.get("pilot_rank4_manifests", []),
        *routes.get("canonical_prior_manifests", []),
    ]
    if len(manifest_routes) != 15 or len(set(manifest_routes)) != 15:
        raise ValueError("large-teacher cumulative shard routing is malformed")
    for relative in manifest_routes:
        shard_path = _safe_local_path(bundle_root, relative)
        shard = _load_json(shard_path, "copied shard manifest")
        npz_path = _safe_local_path(bundle_root, str(
            pathlib.PurePosixPath(relative).parent / str(shard.get("npz", ""))
        ))
        if (
            shard_path.stem != selfsearch.sha256(shard_path)
            or not npz_path.is_file()
            or selfsearch.sha256(npz_path) != shard.get("npz_sha256")
            or npz_path.name != shard.get("npz")
        ):
            raise ValueError("copied shard adjacency or content address changed")

    override_path = _safe_local_path(
        bundle_root, override_record.get("relative_path")
    )
    if (
        by_relative.get(str(override_record.get("relative_path")))
        != override_record
        or selfsearch.sha256(override_path) != override_record.get("sha256")
    ):
        raise ValueError("teacher-only override artifact changed")
    override = _load_json(override_path, "teacher-only override")
    _verify_body_hash(override, schema=OVERRIDE_SCHEMA, label="teacher-only override")
    if (
        override != _body_hashed(_override_body(override.get("pilot_truth", {})))
        or override.get("teacher_only_launch_authorized") is not True
        or override.get("pilot_decision_eligible_for_full") is not False
        or override.get("pilot_truth", {}).get("pilot_passed") is not False
        or override.get("pilot_truth", {}).get("pilot_20_ms_passed") is not False
        or override.get("pilot_truth", {}).get("bypassed_errors")
        != EXACT_BYPASSED_ERRORS
        or override.get("sealed_final_opening_bank_accessed") is not False
        or override.get("blind_labels_accessed") is not False
    ):
        raise ValueError("teacher-only override truth was weakened")
    return manifest


def import_input_bundle(
    *, output: pathlib.Path, artifacts: Sequence[ImportArtifact],
    routes: Mapping[str, object], override_truth: Mapping[str, object],
    source_fingerprints: Mapping[str, object],
) -> dict[str, object]:
    """Copy the entire explicit allowlist through one atomic directory rename."""

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    bundle_root = output / "input-bundle"
    bundle_manifest = bundle_root / "bundle-manifest.json"
    if bundle_root.exists():
        manifest = validate_input_bundle(bundle_manifest)
        expected_records = sorted(
            (
                {
                    "role": artifact.role,
                    "relative_path": artifact.relative_path,
                    "sha256": artifact.sha256,
                    "bytes": artifact.bytes,
                }
                for artifact in artifacts
            ),
            key=lambda item: str(item["role"]),
        )
        actual_records = sorted(
            (
                record for record in manifest.get("artifacts", [])
                if record.get("role") != "teacher-only-override"
            ),
            key=lambda item: str(item["role"]),
        )
        override_path = _safe_local_path(
            bundle_root, manifest["teacher_only_override"]["relative_path"]
        )
        override = _load_json(override_path, "existing teacher-only override")
        if (
            manifest.get("routes") != dict(routes)
            or manifest.get("source_fingerprints") != dict(source_fingerprints)
            or actual_records != expected_records
            or override.get("pilot_truth") != dict(override_truth)
        ):
            raise ValueError("existing input bundle differs from the frozen import")
        return manifest

    staging = pathlib.Path(
        tempfile.mkdtemp(prefix=".input-bundle-", dir=output)
    ).resolve()
    try:
        for artifact in artifacts:
            _copy_import_artifact(artifact, staging)
        copied_records = [
            _local_artifact_record(
                role=artifact.role,
                path=staging / artifact.relative_path,
                root=staging,
            )
            for artifact in artifacts
        ]
        override_relative = "pilot/teacher-only-override.json"
        override_path = staging / override_relative
        override_body = _override_body(override_truth)
        override = _body_hashed(override_body)
        _atomic_json(override_path, override)
        override_record = _local_artifact_record(
            role="teacher-only-override", path=override_path, root=staging,
        )
        copied_records.append(override_record)
        body: dict[str, object] = {
            "schema": BUNDLE_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "source_fingerprints": dict(source_fingerprints),
            "routes": dict(routes),
            "teacher_only_override": override_record,
            "full_specification": _full_spec_record(),
            "artifacts": sorted(copied_records, key=lambda item: str(item["role"])),
            "atomic_import": True,
            "explicit_allowlist": True,
            "runtime_uses_source_paths": False,
            "sealed_final_opening_bank_accessed": False,
            "blind_labels_accessed": False,
        }
        manifest = _body_hashed(body)
        _atomic_json(staging / "bundle-manifest.json", manifest)
        os.replace(staging, bundle_root)
        return validate_input_bundle(bundle_manifest)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _git(repository: pathlib.Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=repository, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if completed.returncode != 0:
        raise ValueError(f"Git inspection failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def _branch_repository_record(
    *, repository: pathlib.Path, expected_commit: str, expected_branch: str,
) -> dict[str, object]:
    repository = repository.resolve()
    record = selfsearch._repository_record(repository, expected_commit)
    if record.get("branch") != expected_branch or not expected_branch:
        raise ValueError("large-teacher launch requires its bound topic branch")
    if _git(repository, "rev-parse", f"{BASE_COMMIT}^{{commit}}") != BASE_COMMIT:
        raise ValueError("large-teacher base commit is unavailable")
    main_commit = _git(repository, "rev-parse", "refs/heads/main")
    origin_main_commit = _git(repository, "rev-parse", "refs/remotes/origin/main")
    if main_commit != BASE_COMMIT or origin_main_commit != BASE_COMMIT:
        raise ValueError("large-teacher base is not the frozen pushed main commit")
    if _git(repository, "merge-base", expected_commit, BASE_COMMIT) != BASE_COMMIT:
        raise ValueError("large-teacher topic branch does not descend from the exact base")
    topic_commit_count = int(
        _git(repository, "rev-list", "--count", f"{BASE_COMMIT}..{expected_commit}")
    )
    if topic_commit_count < 1:
        raise ValueError("large-teacher implementation has no topic-branch commit")
    return {
        **record,
        "base_commit": BASE_COMMIT,
        "base_local_main": main_commit,
        "base_origin_main": origin_main_commit,
        "topic_branch": expected_branch,
        "topic_commit_count": topic_commit_count,
        "branch_bound": True,
    }


def _artifact_matches(record: object) -> bool:
    if not _is_snapshot(record):
        return False
    assert isinstance(record, dict)
    try:
        current = artifact_snapshot(pathlib.Path(record["path"]))
    except (OSError, ValueError):
        return False
    return current == record


def _build_record_without_git(
    path: pathlib.Path, *, launch_repository: Mapping[str, object],
    executable_records: Mapping[str, object],
) -> dict[str, object]:
    record = _load_json(path, "large-teacher Release build")
    _verify_body_hash(
        record, schema=selfsearch.BUILD_MANIFEST_SCHEMA,
        label="large-teacher Release build",
    )
    repository = record.get("repository")
    build = record.get("build")
    if (
        not isinstance(repository, dict)
        or repository.get("path") != launch_repository.get("path")
        or repository.get("head") != launch_repository.get("head")
        or repository.get("branch") != launch_repository.get("branch")
        or repository.get("tree") != launch_repository.get("tree")
        or repository.get("clean") is not True
        or not isinstance(build, dict)
        or build.get("type") != "Release"
        or build.get("sanitizers") is not False
        or record.get("executables") != dict(executable_records)
        or not isinstance(record.get("source_identities"), dict)
        or not record["source_identities"]
        or not isinstance(record.get("tool_sources"), dict)
        or not record["tool_sources"]
    ):
        raise ValueError("large-teacher Release build record is malformed")
    return record


def _launch_guard_records(
    *, executables: selfsearch.CampaignExecutables,
    build_manifest: pathlib.Path, build_record: Mapping[str, object],
    bundle_manifest: pathlib.Path, wrapper: pathlib.Path,
    python_runtime: pathlib.Path,
) -> list[dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    candidates: list[object] = [
        *executables.snapshots().values(),
        artifact_snapshot(build_manifest),
        artifact_snapshot(bundle_manifest),
        artifact_snapshot(wrapper),
        artifact_snapshot(python_runtime),
        *build_record.get("tool_sources", {}).values(),
        build_record.get("build", {}).get("cmake_cache"),
    ]
    for candidate in candidates:
        if not _is_snapshot(candidate):
            raise ValueError("large-teacher launch guard input is malformed")
        assert isinstance(candidate, dict)
        records[str(pathlib.Path(candidate["path"]).resolve())] = dict(candidate)
    return [records[path] for path in sorted(records)]


def freeze_full_launch(
    *, repository: pathlib.Path, expected_commit: str, expected_branch: str,
    source_pilot: pathlib.Path, canonical_campaign: pathlib.Path,
    output: pathlib.Path, executables: selfsearch.CampaignExecutables,
    build_manifest: pathlib.Path,
) -> dict[str, object]:
    """Perform every Git/source check and freeze one Git-free launch receipt."""

    repository = repository.resolve()
    output = output.resolve()
    build_manifest = build_manifest.resolve()
    if output.name != CAMPAIGN_ID:
        raise ValueError(f"output directory must be named {CAMPAIGN_ID}")
    output.mkdir(parents=True, exist_ok=True)
    executables = executables.resolved()
    executables.validate()
    repository_record = _branch_repository_record(
        repository=repository, expected_commit=expected_commit,
        expected_branch=expected_branch,
    )
    build_record = selfsearch.validate_build_manifest(
        build_manifest, repository=repository, expected_commit=expected_commit,
        executables=executables,
    )
    if build_record.get("repository", {}).get("branch") != expected_branch:
        raise ValueError("Release build is not bound to the topic branch")

    artifacts, routes, override_truth, source_fingerprints = build_import_plan(
        source_pilot=source_pilot,
        canonical_campaign=canonical_campaign,
    )
    bundle = import_input_bundle(
        output=output, artifacts=artifacts, routes=routes,
        override_truth=override_truth,
        source_fingerprints=source_fingerprints,
    )
    if _branch_repository_record(
        repository=repository, expected_commit=expected_commit,
        expected_branch=expected_branch,
    ) != repository_record:
        raise ValueError("repository identity changed during the input import")
    if selfsearch.validate_build_manifest(
        build_manifest, repository=repository, expected_commit=expected_commit,
        executables=executables,
    ) != build_record:
        raise ValueError("Release build identity changed during the input import")
    executables.validate()
    bundle_manifest = output / "input-bundle/bundle-manifest.json"
    override_path = _safe_local_path(
        bundle_manifest.parent,
        bundle["teacher_only_override"]["relative_path"],
    )
    environment = selfsearch.environment_identity()
    # Preserve a venv launcher path when present.  artifact_snapshot resolves
    # its target, while the environment identity binds the active interpreter.
    python_runtime = pathlib.Path(os.path.abspath(sys.executable))
    wrapper = pathlib.Path(__file__).resolve()
    executable_records = executables.snapshots()
    guard = _launch_guard_records(
        executables=executables,
        build_manifest=build_manifest,
        build_record=build_record,
        bundle_manifest=bundle_manifest,
        wrapper=wrapper,
        python_runtime=python_runtime,
    )
    body: dict[str, object] = {
        "schema": LAUNCH_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "full_campaign_id": FULL_CAMPAIGN_ID,
        "output_directory": str(output),
        "base_commit": BASE_COMMIT,
        "expected_commit": expected_commit,
        "expected_branch": expected_branch,
        "repository": repository_record,
        "release_build": artifact_snapshot(build_manifest),
        "release_build_record": build_record,
        "executables": executable_records,
        "wrapper": artifact_snapshot(wrapper),
        "python_runtime": artifact_snapshot(python_runtime),
        "environment": environment,
        "input_bundle": artifact_snapshot(bundle_manifest),
        "input_bundle_record": bundle,
        "full_specification": _full_spec_record(),
        "artifact_guard": guard,
        "runtime_git_access": False,
        "runtime_old_worktree_access": False,
        "persistent_power_check_required": True,
        "pilot_passed": False,
        "pilot_20_ms_passed": False,
        "teacher_only_override": artifact_snapshot(override_path),
        "canonical_promotion_eligible": False,
        "publication": False,
        "external_upload": False,
        "replace_rank4": False,
        "leaderboard_claim": False,
    }
    launch = _body_hashed(body)
    launch_path = output / "full-launch.json"
    if launch_path.exists():
        _write_exact_json(launch_path, launch, "large-teacher full launch")
        candidate_launch = launch_path
    else:
        candidate_launch = output / f".full-launch-{os.getpid()}.json"
        if candidate_launch.exists():
            raise ValueError("stale large-teacher launch staging file exists")
        _atomic_json(candidate_launch, launch)
    try:
        if _branch_repository_record(
            repository=repository, expected_commit=expected_commit,
            expected_branch=expected_branch,
        ) != repository_record:
            raise ValueError(
                "repository identity changed while writing the launch receipt"
            )
        if selfsearch.validate_build_manifest(
            build_manifest, repository=repository, expected_commit=expected_commit,
            executables=executables,
        ) != build_record:
            raise ValueError(
                "Release build identity changed while writing the launch receipt"
            )
        validate_full_launch(candidate_launch)
        if candidate_launch != launch_path:
            os.replace(candidate_launch, launch_path)
        return validate_full_launch(launch_path)
    finally:
        if candidate_launch != launch_path and candidate_launch.exists():
            candidate_launch.unlink()


def validate_full_launch(path: pathlib.Path) -> dict[str, object]:
    """Validate the launch and local hashes without inspecting Git."""

    path = path.resolve()
    launch = _load_json(path, "large-teacher full launch")
    _verify_body_hash(launch, schema=LAUNCH_SCHEMA, label="large-teacher launch")
    body = dict(launch)
    body.pop("body_sha256", None)
    repository = body.get("repository")
    executable_records = body.get("executables")
    if (
        body.get("campaign_id") != CAMPAIGN_ID
        or body.get("full_campaign_id") != FULL_CAMPAIGN_ID
        or body.get("output_directory") != str(path.parent.resolve())
        or body.get("base_commit") != BASE_COMMIT
        or body.get("expected_branch") != TOPIC_BRANCH
        or body.get("full_specification") != _full_spec_record()
        or body.get("runtime_git_access") is not False
        or body.get("runtime_old_worktree_access") is not False
        or body.get("persistent_power_check_required") is not True
        or body.get("pilot_passed") is not False
        or body.get("pilot_20_ms_passed") is not False
        or body.get("canonical_promotion_eligible") is not False
        or body.get("publication") is not False
        or body.get("external_upload") is not False
        or body.get("replace_rank4") is not False
        or body.get("leaderboard_claim") is not False
        or not isinstance(repository, dict)
        or repository.get("head") != body.get("expected_commit")
        or repository.get("branch") != TOPIC_BRANCH
        or repository.get("topic_branch") != TOPIC_BRANCH
        or repository.get("base_commit") != BASE_COMMIT
        or repository.get("clean") is not True
        or repository.get("branch_bound") is not True
        or not isinstance(executable_records, dict)
    ):
        raise ValueError("large-teacher launch policy is invalid")
    if body.get("environment") != selfsearch.environment_identity():
        raise ValueError("large-teacher runtime environment changed")

    guard = body.get("artifact_guard")
    if not isinstance(guard, list) or not guard:
        raise ValueError("large-teacher launch has no artifact guard")
    for record in guard:
        if not _artifact_matches(record):
            raise ValueError("large-teacher frozen producer changed")
    if not _artifact_matches(body.get("input_bundle")):
        raise ValueError("large-teacher bundle manifest changed")
    bundle_path = pathlib.Path(body["input_bundle"]["path"]).resolve()
    if bundle_path != path.parent / "input-bundle/bundle-manifest.json":
        raise ValueError("large-teacher bundle is outside its output directory")
    bundle = validate_input_bundle(bundle_path)
    if bundle != body.get("input_bundle_record"):
        raise ValueError("large-teacher copied bundle record changed")
    override = body.get("teacher_only_override")
    if (
        not _artifact_matches(override)
        or pathlib.Path(str(override.get("path", ""))).resolve().parent
        != bundle_path.parent / "pilot"
    ):
        raise ValueError("large-teacher teacher-only override changed")

    build_snapshot = body.get("release_build")
    if not _artifact_matches(build_snapshot):
        raise ValueError("large-teacher Release build manifest changed")
    build_path = pathlib.Path(build_snapshot["path"])
    build_record = _build_record_without_git(
        build_path, launch_repository=repository,
        executable_records=executable_records,
    )
    if build_record != body.get("release_build_record"):
        raise ValueError("large-teacher Release build record changed")
    executables = selfsearch.CampaignExecutables(
        **{
            name: pathlib.Path(record["path"])
            for name, record in executable_records.items()
        }
    ).resolved()
    executables.validate()
    if executables.snapshots() != executable_records:
        raise ValueError("large-teacher executable roster changed")
    return launch


def _status(path: pathlib.Path, phase: str, **details: object) -> None:
    _atomic_json(
        path,
        {
            "schema": STATUS_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "phase": phase,
            "updated_at_unix": time.time(),
            **details,
        },
    )


def _resolve_runtime_routes(
    launch: Mapping[str, object],
) -> dict[str, object]:
    bundle_path = pathlib.Path(launch["input_bundle"]["path"]).resolve()
    bundle_root = bundle_path.parent
    bundle = launch["input_bundle_record"]
    routes = bundle.get("routes")
    if not isinstance(routes, dict):
        raise ValueError("large-teacher runtime has no copied route map")
    canonical = routes.get("canonical_splits")
    if not isinstance(canonical, dict):
        raise ValueError("large-teacher canonical split routes are malformed")
    resolved = {
        "bundle_manifest": bundle_path,
        "actor": _safe_local_path(bundle_root, routes.get("actor")),
        "diversity_reference": _safe_local_path(
            bundle_root, routes.get("diversity_reference")
        ),
        "original_retention_reference": _safe_local_path(
            bundle_root, routes.get("original_retention_reference")
        ),
        "roots_tsv": _safe_local_path(bundle_root, routes.get("roots_tsv")),
        "roots_manifest": _safe_local_path(
            bundle_root, routes.get("roots_manifest")
        ),
        "canonical_splits": {
            split: tuple(
                _safe_local_path(bundle_root, relative)
                for relative in canonical.get(split, [])
            )
            for split in ("train", "validation", "test")
        },
        "canonical_prior_manifests": tuple(
            _safe_local_path(bundle_root, relative)
            for relative in routes.get("canonical_prior_manifests", [])
        ),
        "pilot_search_manifests": tuple(
            _safe_local_path(bundle_root, relative)
            for relative in routes.get("pilot_search_manifests", [])
        ),
        "pilot_rank4_manifests": tuple(
            _safe_local_path(bundle_root, relative)
            for relative in routes.get("pilot_rank4_manifests", [])
        ),
        "opening_exclusions": tuple(
            _safe_local_path(bundle_root, relative)
            for relative in routes.get("opening_exclusions", [])
        ),
    }
    if (
        resolved["actor"] == resolved["diversity_reference"]
        or resolved["diversity_reference"]
        != resolved["original_retention_reference"]
        or selfsearch.sha256(resolved["actor"]) != TEACHER_ACTOR_SHA256
        or selfsearch.sha256(resolved["diversity_reference"])
        != RETENTION_REFERENCE_SHA256
        or any(len(resolved["canonical_splits"][split]) != 3
               for split in ("train", "validation", "test"))
        or len(resolved["canonical_prior_manifests"]) != 9
        or len(resolved["pilot_search_manifests"]) != 3
        or len(resolved["pilot_rank4_manifests"]) != 3
        or len(resolved["opening_exclusions"]) != 7
    ):
        raise ValueError("large-teacher runtime routing changed")
    for value in resolved.values():
        paths: list[pathlib.Path] = []
        if isinstance(value, pathlib.Path):
            paths = [value]
        elif isinstance(value, tuple):
            paths = list(value)
        elif isinstance(value, dict):
            paths = [path for group in value.values() for path in group]
        for path in paths:
            try:
                path.resolve().relative_to(bundle_root)
            except ValueError as error:
                raise ValueError("runtime route escaped the copied bundle") from error
    return resolved


class _FrozenStatGuard:
    """Use startup content hashes plus cheap per-stage mutation detection."""

    def __init__(self, paths: Sequence[pathlib.Path]) -> None:
        unique = sorted({path.resolve() for path in paths})
        self._records = {path: self._signature(path) for path in unique}

    @staticmethod
    def _signature(path: pathlib.Path) -> tuple[int, int, int, int]:
        if not path.is_file():
            raise ValueError(f"frozen runtime input is missing: {path}")
        stat = path.stat()
        return (stat.st_size, stat.st_mtime_ns, stat.st_dev, stat.st_ino)

    def __call__(self) -> None:
        for path, expected in self._records.items():
            if self._signature(path) != expected:
                raise ValueError(f"frozen runtime input changed: {path}")


def _prepare_run_start(
    *, output: pathlib.Path, launch_path: pathlib.Path,
    launch: Mapping[str, object], resume: bool,
    skip_power_check: bool = False,
) -> dict[str, object]:
    run_start_path = output / "run-start.json"
    full_directory = output / "full"
    common: dict[str, object] = {
        "schema": RUN_START_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "full_campaign_id": FULL_CAMPAIGN_ID,
        "launch": artifact_snapshot(launch_path),
        "input_bundle": launch["input_bundle"],
        "teacher_only_override": launch["teacher_only_override"],
        "full_specification": _full_spec_record(),
        "game_workers": 10,
        "label_workers": 10,
        "resume_contract": "exact-stage-receipts-only",
        "runtime_git_access": False,
        "runtime_old_worktree_access": False,
        "power_check_skipped": skip_power_check,
    }
    if run_start_path.exists():
        if not resume:
            raise ValueError("existing large-teacher run requires --resume")
        existing = _load_json(run_start_path, "large-teacher run start")
        _verify_body_hash(existing, schema=RUN_START_SCHEMA, label="run start")
        comparable = dict(existing)
        comparable.pop("body_sha256", None)
        comparable.pop("started_at_unix", None)
        if comparable != common:
            raise ValueError("large-teacher resume binding changed")
        return existing
    if full_directory.exists() and any(full_directory.iterdir()):
        raise ValueError("large-teacher output has unreceipted full-campaign state")
    record = _body_hashed({**common, "started_at_unix": time.time()})
    _atomic_json(run_start_path, record)
    return record


def _validate_existing_summary(
    path: pathlib.Path, *, launch_path: pathlib.Path | None = None,
    run_start_path: pathlib.Path | None = None,
) -> dict[str, object]:
    path = path.resolve()
    summary = _load_json(path, "large-teacher final summary")
    _verify_body_hash(summary, schema=SUMMARY_SCHEMA, label="final summary")
    terminal = summary.get("terminal")
    if terminal not in {"teacher-candidate-accepted", "full-rejected"}:
        raise ValueError("large-teacher final summary is not terminal")
    if (
        summary.get("campaign_id") != CAMPAIGN_ID
        or summary.get("pilot_passed") is not False
        or summary.get("pilot_20_ms_passed") is not False
        or summary.get("student_training_started") is not False
        or not isinstance(summary.get("full"), dict)
        or summary.get("canonical_promotion_eligible") is not False
        or summary.get("publication") is not None
        or summary.get("external_upload") is not False
        or summary.get("rank4_replaced") is not False
        or summary.get("leaderboard_claim") is not False
    ):
        raise ValueError("large-teacher final summary policy changed")
    if launch_path is not None and (
        not _snapshot_matches(summary.get("launch"), expected=launch_path)
    ):
        raise ValueError("large-teacher final summary launch binding changed")
    if run_start_path is not None and (
        not _snapshot_matches(summary.get("run_start"), expected=run_start_path)
    ):
        raise ValueError("large-teacher final summary run-start binding changed")
    if launch_path is not None or run_start_path is not None:
        decision = _validate_full_result(summary["full"])
        expected_terminal = (
            "teacher-candidate-accepted"
            if decision.get("eligible_for_local_publication") is True
            else "full-rejected"
        )
        if terminal != expected_terminal:
            raise ValueError("large-teacher final terminal contradicts its decision")
    if terminal == "full-rejected" and (
        summary.get("student_training_eligible") is not False
        or summary.get("teacher_candidate_accepted") is not None
        or summary.get("compact_student_handoff") is not None
        or (path.parent / "teacher-candidate-accepted.json").exists()
        or (path.parent / "compact-student-handoff.json").exists()
    ):
        raise ValueError("rejected full campaign cannot train a student")
    if terminal == "teacher-candidate-accepted" and (
        summary.get("student_training_eligible") is not True
        or not _snapshot_matches(summary.get("teacher_candidate_accepted"))
        or not _snapshot_matches(summary.get("compact_student_handoff"))
        or pathlib.Path(summary["teacher_candidate_accepted"]["path"]).resolve()
        != path.parent / "teacher-candidate-accepted.json"
        or pathlib.Path(summary["compact_student_handoff"]["path"]).resolve()
        != path.parent / "compact-student-handoff.json"
    ):
        raise ValueError("accepted full campaign receipts are stale")
    if terminal == "teacher-candidate-accepted":
        _validate_accepted_receipts(summary=summary, summary_path=path)
    return summary


def _validate_full_result(full: Mapping[str, object]) -> dict[str, object]:
    decision = full.get("decision")
    decision_path = pathlib.Path(str(full.get("decision_path", ""))).resolve()
    if (
        full.get("profile") != "full"
        or full.get("campaign_id") != FULL_CAMPAIGN_ID
        or not isinstance(decision, dict)
        or decision.get("schema") != selfsearch.FINAL_DECISION_SCHEMA
        or decision.get("canonical_promotion_eligible") is not False
        or _load_json(decision_path, "large-teacher final decision") != decision
    ):
        raise ValueError("large-teacher full result is malformed")
    reports = decision.get("reports")
    if not isinstance(reports, dict) or set(reports) != {
        "pilot-teacher", "matched", "rank4", "jacek-nn",
    }:
        raise ValueError("large-teacher final panel evidence is incomplete")
    for record in reports.values():
        if not _artifact_matches(record):
            raise ValueError("large-teacher final panel report changed")
    recomputed = selfsearch.final_decision(
        pilot_report=pathlib.Path(reports["pilot-teacher"]["path"]),
        matched_report=pathlib.Path(reports["matched"]["path"]),
        rank4_report=pathlib.Path(reports["rank4"]["path"]),
        jacek_nn_report=pathlib.Path(reports["jacek-nn"]["path"]),
        anchor_candidate=decision["anchor_candidate"],
        anchor_incumbent=decision["anchor_incumbent"],
        original_anchor_candidate=decision["original_anchor_candidate"],
        original_anchor_incumbent=decision["original_anchor_incumbent"],
        uncontended_max_ms=decision["uncontended_max_ms"],
    )
    if recomputed != decision:
        raise ValueError("large-teacher final decision changed after replay")
    return decision


def _final_acceptance_policy() -> dict[str, object]:
    return {
        "pairs_per_panel": 500,
        "time_ms": 980,
        "pilot_teacher_and_matched": {
            "minimum_wins": 527, "minimum_per_color": 260,
        },
        "rank4_and_external_neural": {
            "minimum_wins": 501, "minimum_per_color": 238,
        },
        "illegal": 0,
        "unfinished": 0,
        "maximum_ms_exclusive": 1_000,
        "canonical_retention_references": [
            "pilot-teacher", "original-v6-reference",
        ],
    }


def _validate_snapshot_list(
    value: object, expected_paths: Sequence[pathlib.Path], label: str,
) -> None:
    if not isinstance(value, list) or len(value) != len(expected_paths):
        raise ValueError(f"{label} snapshot roster is incomplete")
    for record, expected in zip(value, expected_paths, strict=True):
        if not _snapshot_matches(record, expected=expected):
            raise ValueError(f"{label} snapshot changed")


def _validate_accepted_receipts(
    *, summary: Mapping[str, object], summary_path: pathlib.Path,
) -> None:
    output = summary_path.resolve().parent
    acceptance_path = output / "teacher-candidate-accepted.json"
    handoff_path = output / "compact-student-handoff.json"
    acceptance = _load_json(acceptance_path, "teacher-candidate acceptance")
    _verify_body_hash(
        acceptance, schema=ACCEPTANCE_SCHEMA,
        label="teacher-candidate acceptance",
    )
    acceptance_body = dict(acceptance)
    acceptance_body.pop("body_sha256", None)
    full = summary.get("full")
    if not isinstance(full, dict):
        raise ValueError("accepted summary has no full result")
    if (
        set(acceptance_body) != {
            "schema", "campaign_id", "full_campaign_id", "classification",
            "runtime", "manifest", "decision", "launch", "run_start",
            "pilot_passed", "pilot_20_ms_passed", "teacher_only_override",
            "final_acceptance", "canonical_promotion_eligible", "publication",
            "external_upload", "replace_rank4", "leaderboard_claim",
        }
        or acceptance.get("campaign_id") != CAMPAIGN_ID
        or acceptance.get("full_campaign_id") != FULL_CAMPAIGN_ID
        or acceptance.get("classification") != "local-teacher-candidate"
        or acceptance.get("pilot_passed") is not False
        or acceptance.get("pilot_20_ms_passed") is not False
        or acceptance.get("final_acceptance") != _final_acceptance_policy()
        or acceptance.get("canonical_promotion_eligible") is not False
        or acceptance.get("publication") is not False
        or acceptance.get("external_upload") is not False
        or acceptance.get("replace_rank4") is not False
        or acceptance.get("leaderboard_claim") is not False
        or not _snapshot_matches(
            acceptance.get("runtime"),
            expected=pathlib.Path(str(full.get("search_runtime", ""))),
        )
        or not _snapshot_matches(
            acceptance.get("manifest"),
            expected=pathlib.Path(str(full.get("search_manifest", ""))),
        )
        or not _snapshot_matches(
            acceptance.get("decision"),
            expected=pathlib.Path(str(full.get("decision_path", ""))),
        )
        or not _snapshot_matches(
            acceptance.get("launch"), expected=output / "full-launch.json"
        )
        or not _snapshot_matches(
            acceptance.get("run_start"), expected=output / "run-start.json"
        )
        or not _snapshot_matches(
            acceptance.get("teacher_only_override"),
            expected=output / "input-bundle/pilot/teacher-only-override.json",
        )
    ):
        raise ValueError("teacher-candidate acceptance receipt is stale")

    handoff = _load_json(handoff_path, "compact-student handoff")
    _verify_body_hash(handoff, schema=HANDOFF_SCHEMA, label="compact-student handoff")
    handoff_body = dict(handoff)
    handoff_body.pop("body_sha256", None)
    if (
        set(handoff_body) != {
            "schema", "campaign_id", "classification",
            "teacher_candidate_accepted", "teacher_runtime", "teacher_manifest",
            "pilot_search_manifests", "full_search_manifests",
            "pilot_rank4_manifests", "full_rank4_manifests",
            "student_training_eligible", "student_training_started",
            "canonical_promotion_eligible", "publication", "external_upload",
            "replace_rank4", "leaderboard_claim",
        }
        or handoff.get("campaign_id") != CAMPAIGN_ID
        or handoff.get("classification")
        != "local-compact-student-training-input"
        or handoff.get("teacher_candidate_accepted")
        != artifact_snapshot(acceptance_path)
        or handoff.get("teacher_runtime") != acceptance.get("runtime")
        or handoff.get("teacher_manifest") != acceptance.get("manifest")
        or handoff.get("student_training_eligible") is not True
        or handoff.get("student_training_started") is not False
        or handoff.get("canonical_promotion_eligible") is not False
        or handoff.get("publication") is not False
        or handoff.get("external_upload") is not False
        or handoff.get("replace_rank4") is not False
        or handoff.get("leaderboard_claim") is not False
    ):
        raise ValueError("compact-student handoff policy changed")
    bundle_path = output / "input-bundle/bundle-manifest.json"
    bundle = _load_json(bundle_path, "large-teacher input bundle")
    routes = bundle.get("routes", {})
    bundle_root = bundle_path.parent
    _validate_snapshot_list(
        handoff.get("pilot_search_manifests"),
        [
            _safe_local_path(bundle_root, relative)
            for relative in routes.get("pilot_search_manifests", [])
        ],
        "pilot Search handoff",
    )
    _validate_snapshot_list(
        handoff.get("pilot_rank4_manifests"),
        [
            _safe_local_path(bundle_root, relative)
            for relative in routes.get("pilot_rank4_manifests", [])
        ],
        "pilot Rank-4 handoff",
    )
    _validate_snapshot_list(
        handoff.get("full_search_manifests"),
        [pathlib.Path(path) for path in full.get("search_new_manifests", [])],
        "full Search handoff",
    )
    _validate_snapshot_list(
        handoff.get("full_rank4_manifests"),
        [pathlib.Path(path) for path in full.get("rank4_new_manifests", [])],
        "full Rank-4 handoff",
    )


def _write_terminal_summary(
    *, output: pathlib.Path, launch_path: pathlib.Path,
    run_start_path: pathlib.Path, full: Mapping[str, object],
    decision: Mapping[str, object],
) -> dict[str, object]:
    eligible = decision.get("eligible_for_local_publication") is True
    acceptance_path = output / "teacher-candidate-accepted.json"
    handoff_path = output / "compact-student-handoff.json"
    if eligible:
        if decision.get("errors") != []:
            raise ValueError("accepted teacher has final gate errors")
        runtime = pathlib.Path(str(full.get("search_runtime", ""))).resolve()
        manifest = pathlib.Path(str(full.get("search_manifest", ""))).resolve()
        acceptance = _body_hashed(
            {
                "schema": ACCEPTANCE_SCHEMA,
                "campaign_id": CAMPAIGN_ID,
                "full_campaign_id": FULL_CAMPAIGN_ID,
                "classification": "local-teacher-candidate",
                "runtime": artifact_snapshot(runtime),
                "manifest": artifact_snapshot(manifest),
                "decision": artifact_snapshot(
                    pathlib.Path(str(full["decision_path"]))
                ),
                "launch": artifact_snapshot(launch_path),
                "run_start": artifact_snapshot(run_start_path),
                "pilot_passed": False,
                "pilot_20_ms_passed": False,
                "teacher_only_override": artifact_snapshot(
                    output / "input-bundle/pilot/teacher-only-override.json"
                ),
                "final_acceptance": _final_acceptance_policy(),
                "canonical_promotion_eligible": False,
                "publication": False,
                "external_upload": False,
                "replace_rank4": False,
                "leaderboard_claim": False,
            }
        )
        _write_exact_json(
            acceptance_path, acceptance, "teacher-candidate acceptance receipt"
        )
        bundle_path = output / "input-bundle/bundle-manifest.json"
        bundle = _load_json(bundle_path, "large-teacher input bundle")
        bundle_routes = bundle.get("routes", {})
        bundle_root = bundle_path.parent
        handoff = _body_hashed(
            {
                "schema": HANDOFF_SCHEMA,
                "campaign_id": CAMPAIGN_ID,
                "classification": "local-compact-student-training-input",
                "teacher_candidate_accepted": artifact_snapshot(acceptance_path),
                "teacher_runtime": artifact_snapshot(runtime),
                "teacher_manifest": artifact_snapshot(manifest),
                "pilot_search_manifests": [
                    artifact_snapshot(_safe_local_path(bundle_root, relative))
                    for relative in bundle_routes.get("pilot_search_manifests", [])
                ],
                "full_search_manifests": [
                    artifact_snapshot(pathlib.Path(path))
                    for path in full.get("search_new_manifests", [])
                ],
                "pilot_rank4_manifests": [
                    artifact_snapshot(_safe_local_path(bundle_root, relative))
                    for relative in bundle_routes.get("pilot_rank4_manifests", [])
                ],
                "full_rank4_manifests": [
                    artifact_snapshot(pathlib.Path(path))
                    for path in full.get("rank4_new_manifests", [])
                ],
                "student_training_eligible": True,
                "student_training_started": False,
                "canonical_promotion_eligible": False,
                "publication": False,
                "external_upload": False,
                "replace_rank4": False,
                "leaderboard_claim": False,
            }
        )
        _write_exact_json(handoff_path, handoff, "compact-student handoff receipt")
        terminal = "teacher-candidate-accepted"
        accepted_snapshot: dict[str, object] | None = artifact_snapshot(acceptance_path)
        handoff_snapshot: dict[str, object] | None = artifact_snapshot(handoff_path)
    else:
        if acceptance_path.exists() or handoff_path.exists():
            raise ValueError("rejected full campaign found stale acceptance artifacts")
        terminal = "full-rejected"
        accepted_snapshot = None
        handoff_snapshot = None
    summary = _body_hashed(
        {
            "schema": SUMMARY_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "terminal": terminal,
            "launch": artifact_snapshot(launch_path),
            "run_start": artifact_snapshot(run_start_path),
            "full": dict(full),
            "teacher_candidate_accepted": accepted_snapshot,
            "compact_student_handoff": handoff_snapshot,
            "student_training_eligible": eligible,
            "student_training_started": False,
            "pilot_passed": False,
            "pilot_20_ms_passed": False,
            "canonical_promotion_eligible": False,
            "publication": None,
            "external_upload": False,
            "rank4_replaced": False,
            "leaderboard_claim": False,
        }
    )
    _write_exact_json(output / "final-summary.json", summary, "final summary")
    return summary


def run_full(
    *, launch_receipt: pathlib.Path, output: pathlib.Path,
    resume: bool, skip_power_check: bool,
) -> dict[str, object]:
    """Run the exact full phase with copied inputs and no Git/source access."""

    launch_receipt = launch_receipt.resolve()
    output = output.resolve()
    if (
        output.name != CAMPAIGN_ID
        or launch_receipt != output / "full-launch.json"
    ):
        raise ValueError("large-teacher run does not match its frozen output")
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "supervisor-status.json"
    final_summary_path = output / "final-summary.json"
    lock_path = output / "supervisor.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("large-teacher supervisor is already running") from error
        selfsearch._CAMPAIGN_LOCK_FD = lock.fileno()
        try:
            launch = validate_full_launch(launch_receipt)
            if final_summary_path.exists():
                if not resume:
                    raise ValueError("completed large-teacher run requires --resume")
                summary = _validate_existing_summary(
                    final_summary_path,
                    launch_path=launch_receipt,
                    run_start_path=output / "run-start.json",
                )
                _status(
                    status_path,
                    str(summary["terminal"]),
                    summary=artifact_snapshot(final_summary_path),
                    student_training_eligible=summary["student_training_eligible"],
                    canonical_promotion_eligible=False,
                    resumed_terminal=True,
                )
                return summary
            if (
                skip_power_check
                and launch.get("persistent_power_check_required") is True
            ):
                raise ValueError(
                    "frozen persistent launch cannot skip the AC power check"
                )
            health = selfsearch.validate_host_health(
                output, skip_power=skip_power_check
            )
            run_start = _prepare_run_start(
                output=output, launch_path=launch_receipt,
                launch=launch, resume=resume,
                skip_power_check=skip_power_check,
            )
            routes = _resolve_runtime_routes(launch)
            executable_records = launch["executables"]
            executables = selfsearch.CampaignExecutables(
                **{
                    name: pathlib.Path(record["path"])
                    for name, record in executable_records.items()
                }
            ).resolved()
            build_manifest = pathlib.Path(launch["release_build"]["path"])
            build_record = launch["release_build_record"]
            watched_paths = [
                launch_receipt,
                routes["bundle_manifest"],
                pathlib.Path(launch["teacher_only_override"]["path"]),
                *(
                    pathlib.Path(record["path"])
                    for record in launch["artifact_guard"]
                ),
            ]
            for record in launch["input_bundle_record"]["artifacts"]:
                watched_paths.append(
                    routes["bundle_manifest"].parent / record["relative_path"]
                )
            producer_guard = _FrozenStatGuard(watched_paths)
            producer_guard()
            _status(
                status_path,
                "running-full",
                run_start=artifact_snapshot(output / "run-start.json"),
                launch=artifact_snapshot(launch_receipt),
                games=10_000,
                game_workers=10,
                label_workers=10,
                actor_sha256=TEACHER_ACTOR_SHA256,
                original_retention_reference_sha256=RETENTION_REFERENCE_SHA256,
                launch_health=health,
                pilot_passed=False,
                pilot_20_ms_passed=False,
            )
            full = selfsearch.run_phase(
                spec=FULL_SPEC,
                output=output / "full",
                resume=resume,
                roots_tsv=routes["roots_tsv"],
                roots_manifest=routes["roots_manifest"],
                actor=routes["actor"],
                diversity=routes["diversity_reference"],
                original_incumbent=routes["original_retention_reference"],
                executables=executables,
                anchor_train_manifests=routes["canonical_splits"]["train"],
                retention_validation_manifests=(
                    routes["canonical_splits"]["validation"]
                ),
                anchor_validation_manifests=routes["canonical_splits"]["test"],
                canonical_prior_manifests=routes["canonical_prior_manifests"],
                opening_exclusions=routes["opening_exclusions"],
                prior_search_manifests=routes["pilot_search_manifests"],
                prior_rank4_manifests=routes["pilot_rank4_manifests"],
                producer_guard=producer_guard,
                build_manifest=build_manifest,
                source_identities=build_record["source_identities"],
            )
            producer_guard()
            full["launch_health"] = health
            decision = _validate_full_result(full)
            if (output / "promoted").exists():
                raise ValueError("large-teacher workflow must not publish or promote")
            summary = _write_terminal_summary(
                output=output,
                launch_path=launch_receipt,
                run_start_path=output / "run-start.json",
                full=full,
                decision=decision,
            )
            _status(
                status_path,
                str(summary["terminal"]),
                summary=artifact_snapshot(output / "final-summary.json"),
                student_training_eligible=summary["student_training_eligible"],
                canonical_promotion_eligible=False,
            )
            return summary
        except Exception as error:
            preserved_summary: dict[str, object] | None = None
            if final_summary_path.is_file():
                try:
                    preserved_summary = _validate_existing_summary(
                        final_summary_path,
                        launch_path=launch_receipt,
                        run_start_path=output / "run-start.json",
                    )
                except Exception:
                    preserved_summary = None
            if preserved_summary is not None:
                _status(
                    status_path,
                    str(preserved_summary["terminal"]),
                    summary=artifact_snapshot(final_summary_path),
                    student_training_eligible=preserved_summary[
                        "student_training_eligible"
                    ],
                    canonical_promotion_eligible=False,
                    last_invocation_error=str(error),
                    terminal_preserved=True,
                )
                raise
            _status(
                status_path,
                "failed",
                error=str(error),
                traceback=traceback.format_exc(),
                student_training_eligible=False,
                canonical_promotion_eligible=False,
            )
            raise
        finally:
            selfsearch._CAMPAIGN_LOCK_FD = None


def _add_executable_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--continuation-generator", type=pathlib.Path, required=True)
    parser.add_argument("--search-teacher", type=pathlib.Path, required=True)
    parser.add_argument("--rank4-teacher", type=pathlib.Path, required=True)
    parser.add_argument("--comparison", type=pathlib.Path, required=True)
    parser.add_argument("--pack-tool", type=pathlib.Path, required=True)
    parser.add_argument("--trainer", type=pathlib.Path, required=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser(
        "freeze-full-launch",
        help="import exact inputs and write a branch-bound Git-free launch receipt",
    )
    freeze.add_argument("--repository", type=pathlib.Path, required=True)
    freeze.add_argument("--expected-commit", required=True)
    freeze.add_argument("--expected-branch", default=TOPIC_BRANCH)
    freeze.add_argument("--source-pilot", type=pathlib.Path, required=True)
    freeze.add_argument("--canonical-campaign", type=pathlib.Path, required=True)
    freeze.add_argument("--output-directory", type=pathlib.Path, required=True)
    freeze.add_argument("--build-manifest", type=pathlib.Path, required=True)
    _add_executable_arguments(freeze)

    run = commands.add_parser(
        "run-full", help="run or resume the frozen full campaign without Git"
    )
    run.add_argument("--launch-receipt", type=pathlib.Path, required=True)
    run.add_argument("--output-directory", type=pathlib.Path, required=True)
    run.add_argument("--resume", action="store_true")
    run.add_argument(
        "--skip-power-check", action="store_true",
        help="tests/development only; the persistent launch must never set this",
    )
    arguments = parser.parse_args()
    if arguments.command == "freeze-full-launch":
        if (
            len(arguments.expected_commit) != 40
            or any(character not in "0123456789abcdef"
                   for character in arguments.expected_commit)
            or arguments.expected_branch != TOPIC_BRANCH
        ):
            parser.error(
                "freeze requires the lowercase implementation commit and bound topic branch"
            )
        freeze_full_launch(
            repository=arguments.repository,
            expected_commit=arguments.expected_commit,
            expected_branch=arguments.expected_branch,
            source_pilot=arguments.source_pilot,
            canonical_campaign=arguments.canonical_campaign,
            output=arguments.output_directory,
            executables=selfsearch.CampaignExecutables(
                continuation_generator=arguments.continuation_generator,
                search_teacher=arguments.search_teacher,
                rank4_teacher=arguments.rank4_teacher,
                comparison=arguments.comparison,
                pack_tool=arguments.pack_tool,
                trainer=arguments.trainer,
            ),
            build_manifest=arguments.build_manifest,
        )
    else:
        run_full(
            launch_receipt=arguments.launch_receipt,
            output=arguments.output_directory,
            resume=arguments.resume,
            skip_power_check=arguments.skip_power_check,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
