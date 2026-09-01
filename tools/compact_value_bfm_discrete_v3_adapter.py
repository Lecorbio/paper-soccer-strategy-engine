#!/usr/bin/env python3
"""Seal the discrete-v3 candidate's powerless development handoff.

This adapter is deliberately narrower than the legacy post-iteration handoff.
It validates the exact canonical discrete-v3 selection/runtime/source closure
and a completed fresh-holdout *diagnostic* report, then emits the candidate
descriptor needed by a discrete-v3-aware development runner.  Its one-shot
``evaluate`` phase reads the protected shards only after an adapter-owned
claim is sealed under an exclusive lock.  It never interprets the diagnostic
metrics as a pass/fail gate, selects a development candidate, authorizes the
strict Rank-4 final, or authorizes an upload.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
REPOSITORY = HERE.parent


def _load(path: pathlib.Path, name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load discrete-v3 adapter dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v3 = _load(
    HERE / "compact_value_bfm_discrete_v3.py",
    "compact_discrete_v3_development_adapter_campaign",
)
qualification = v3.qualification
AdapterError = v3.V3Error

NAMESPACE = v3.NAMESPACE
HANDOFF_SCHEMA = (
    "papersoccer.compact-value-bfm.discrete-v3-development-handoff.v2"
)
HANDOFF_STATUS = "fresh-diagnostic-complete-awaiting-development"
ADAPTER_PLAN_SCHEMA = (
    "papersoccer.compact-value-bfm.discrete-v3-development-adapter-plan.v2"
)
ADAPTER_PLAN_STATUS = "adapter-planned-awaiting-fresh-diagnostic"
V1_ADAPTER_PLAN_SCHEMA = (
    "papersoccer.compact-value-bfm.discrete-v3-development-adapter-plan.v1"
)
V1_RETIREMENT_SCHEMA = (
    "papersoccer.compact-value-bfm.discrete-v3-adapter-v1-retirement.v1"
)
V1_RETIREMENT_STATUS = "adapter-v1-retired-before-evaluation"
EVALUATION_CLAIM_SCHEMA = (
    "papersoccer.compact-value-bfm.discrete-v3-adapter-evaluation-claim.v1"
)
ADAPTER_REPORT_SCHEMA = (
    "papersoccer.compact-value-bfm.discrete-v3-adapter-diagnostic-report.v1"
)
ADAPTER_REPORT_REFERENCE_SCHEMA = (
    "papersoccer.compact-value-bfm.discrete-v3-adapter-report-reference.v1"
)
EVALUATION_COMPLETION_SCHEMA = (
    "papersoccer.compact-value-bfm.discrete-v3-adapter-evaluation-completion.v1"
)
EVALUATION_CLAIM_STATUS = "adapter-evaluation-claimed-once"
EVALUATION_COMPLETION_STATUS = "adapter-owned-fresh-diagnostic-complete"
FRESH_MATERIALIZATION_SCHEMA = (
    "papersoccer.compact-value-bfm.fresh-holdout-materialization.v1"
)
FRESH_MATERIALIZATION_CLAIM_SCHEMA = (
    "papersoccer.compact-value-bfm.fresh-holdout-claim.v1"
)

# The adapter exists only for this already-selected, immutable campaign.  These
# values intentionally make a new v3 search result require a reviewed adapter,
# rather than silently inheriting this handoff route.
V3_PLAN_SHA256 = "2d6f302a8d98869ccad6609e0489b12c2cabdf6ce326672f47ff2b85b65ce84e"
V3_PLAN_BODY_SHA256 = "1a484e42f678c35f76f1bb2f25167be75c5640c3d79a222601e80a3307573e98"
V3_SELECTION_SHA256 = "6d0bb64f4e24a7c5181c85b53747ef0eabafbd680659142f99f9474678999ddf"
V3_SELECTION_BODY_SHA256 = "9be439ac56bdbb3da035d4d0ea28a573509781f23e7829b6fee86cae9872ffe9"
V3_RUNTIME_SHA256 = "130c6ef1d2311a76c7a94fd144a805aa22477a32bced59a8079021e4293ea336"
V3_RUNTIME_BYTES = 38_960
V3_SOURCE_SHA256 = "f5e67d699be19c3d495673c04ee2453570391c59e5f7be2a779198ce98b2d621"
V3_SOURCE_BYTES = 94_834

V1_PLAN_SHA256 = "e66471b65a752bf140ca09b9feaf2779f889f989edf164cc869e9b1b86f17713"
V1_PLAN_BODY_SHA256 = "e354155128bf4e00a5091e784b9d3540ba88d83e00bab8a4ce03ec50221fb8c0"
V1_PLAN_BYTES = 17_869
V1_ADAPTER_SHA256 = "0bc79a880713e6c7b016812d70e1abbdec08001bd20127fd8704c1a59af8bd63"
V1_ADAPTER_BYTES = 57_801
V1_TEST_SHA256 = "5226ee8965b34e8959c59f13bc09e7afb1d38737a90c44088e4b652b16566805"
V1_TEST_BYTES = 27_914
EXPECTED_RUNTIME_ARCHITECTURE = "capacity-12x8"
EXPECTED_DIMENSIONS = (6_301, 12, 8, 1)
EXPECTED_CAMPAIGN_ARCHITECTURE = "6301-12-8-1"

CANDIDATE_ID = "discrete-v3-search-target"
DEVELOPMENT_STAGES = {
    "model_screen": {"pairs": 100, "mode": "fixed-work"},
    "tuple_screen": {"pairs": 100, "mode": "fixed-work"},
    "tuple_confirmation": {"pairs": 250, "mode": "fixed-work"},
    "profile_screen": {"pairs": 100, "mode": "fixed-work"},
    "profile_confirmation": {"pairs": 250, "mode": "fixed-work"},
    "actual_clock": {"pairs": 200, "mode": "actual-clock"},
}
TOOL_CLOSURE_KEYS = (
    "campaign", "canonical_teacher", "compact_workflow",
    "continuation_generator", "discrete_v3", "discrete_v3_holdout",
    "fresh_holdout", "iteration_runner", "model_exporter",
    "opening_generator", "pack_tool", "preflight", "python",
    "qualification", "quantization_v2", "quantization_v2_holdout",
    "rank4_teacher", "replay_corpus", "replay_features", "replay_train",
    "replay_workflow", "search_teacher", "selfsearch_workflow",
    "submission_config", "submission_exporter", "submission_sources",
    "successor", "trainer",
)
HANDOFF_FIELDS = {
    "schema", "namespace", "campaign_id", "status", "created_at_utc",
    "adapter_plan", "v1_retirement", "evaluation_claim", "evaluation_completion",
    "v3_plan", "v3_selection_reference", "v3_selection", "v3_outcome",
    "fresh_report_reference", "fresh_report", "materialization", "candidate",
    "diagnostic_evidence", "development_contract", "tool_closure", "policy",
    "body_sha256",
}
ADAPTER_PLAN_FIELDS = {
    "schema", "namespace", "campaign_id", "status", "planned_at_utc",
    "v1_retirement", "v3_plan", "v3_selection_reference", "v3_selection", "v3_outcome",
    "candidate", "expected_fresh_evidence", "fixed_handoff",
    "development_contract", "tool_closure", "policy", "body_sha256",
}
V1_RETIREMENT_FIELDS = {
    "schema", "namespace", "campaign_id", "status", "retired_at_utc",
    "v1_plan", "v1_plan_body_sha256", "v1_tool_closure", "defect",
    "absent_routes", "adapter_evaluation_started", "protected_metrics_opened",
    "legacy_evaluation_observed", "replacement", "retirement_tool_closure",
    "rank4_gate_authorized", "upload_authorized", "body_sha256",
}
ADAPTER_TEST_PATH = (
    REPOSITORY / "tests/codingame/test_compact_value_bfm_discrete_v3_adapter.py"
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_record(
    path: pathlib.Path, *, ascii_required: bool = False,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise AdapterError(f"adapter input is not a regular file: {path}")
    raw = path.read_bytes()
    if ascii_required:
        try:
            raw.decode("ascii")
        except UnicodeDecodeError as error:
            raise AdapterError(f"adapter source is not ASCII: {path}") from error
    return {
        "path": str(path.resolve()),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _sealed_reference(path: pathlib.Path, schema: str) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        raise AdapterError(f"sealed adapter input is absent or redirected: {path}")
    qualification.load_sealed(path, schema)
    return {"path": str(path.resolve()), "sha256": _sha256_file(path)}


def _valid_sha(value: object) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _syntactic_file_record(value: object, *, label: str) -> dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"path", "bytes", "sha256"}
        or not isinstance(value.get("path"), str)
        or isinstance(value.get("bytes"), bool)
        or not isinstance(value.get("bytes"), int)
        or value.get("bytes", 0) <= 0
        or not _valid_sha(value.get("sha256"))
    ):
        raise AdapterError(f"{label} file binding is malformed")
    return dict(value)


def _rerender_source(
    *, runtime_path: pathlib.Path, source_path: pathlib.Path,
    plan: Mapping[str, Any], expected_export: object,
) -> None:
    """Re-render the submission in memory and require exact source identity."""

    tools = plan.get("tools")
    if not isinstance(tools, Mapping):
        raise AdapterError("discrete-v3 plan omits its exporter closure")
    try:
        v3.iteration.validate_maintained_python_tool_closure(tools)
        model_path = v3.iteration._verify_file_record(
            tools.get("model_exporter"), "compact model exporter"
        )
        submission_path = v3.iteration._verify_file_record(
            tools.get("submission_exporter"), "compact submission exporter"
        )
        model_exporter = v3.iteration._load_module(
            model_path, "compact_discrete_v3_adapter_model_exporter"
        )
        submission_exporter = v3.iteration._load_module(
            submission_path, "compact_discrete_v3_adapter_submission_exporter"
        )
        header, metadata = model_exporter.render_header(runtime_path)
        _default, payload = submission_exporter.render(model_header=header)
        payload.decode("ascii")
    except Exception as error:
        raise AdapterError("discrete-v3 source could not be re-rendered") from error
    actual_source = source_path.read_bytes()
    export = {
        "runtime_sha256": _sha256_file(runtime_path),
        "runtime_body_sha256": metadata.get("body_sha256"),
        "model_header_sha256": metadata.get("header_sha256"),
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "source_ascii_bytes": len(payload),
        "source_limit_exclusive": 95_000,
    }
    if payload != actual_source or expected_export != export:
        raise AdapterError("discrete-v3 source export no longer reproduces exactly")


def _canonical_candidate(
    plan_path: pathlib.Path, output_root: pathlib.Path,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    plan_path = plan_path.resolve()
    if output_root != v3.canonical_v3_root() or plan_path != (
        output_root / "discrete-v3-plan.json"
    ):
        raise AdapterError("adapter requires the canonical discrete-v3 root and plan")
    if _sha256_file(plan_path) != V3_PLAN_SHA256:
        raise AdapterError("discrete-v3 plan file identity changed")
    plan = v3.load_plan(plan_path, output_root=output_root)
    if plan.get("body_sha256") != V3_PLAN_BODY_SHA256:
        raise AdapterError("discrete-v3 plan body identity changed")

    selection_reference_path = output_root / "selection-reference.json"
    selection_path = v3._selection_reference(
        selection_reference_path, plan=plan, output_root=output_root
    )
    if selection_path is None:
        raise AdapterError("canonical discrete-v3 selection is absent")
    selection = v3._validate_selection_closure(
        selection_path, plan=plan, output_root=output_root
    )
    if (
        _sha256_file(selection_path) != V3_SELECTION_SHA256
        or selection.get("body_sha256") != V3_SELECTION_BODY_SHA256
    ):
        raise AdapterError("discrete-v3 selection identity changed")

    runtime_path = pathlib.Path(str(selection.get("runtime", {}).get("path", "")))
    source_path = pathlib.Path(
        str(selection.get("generated_source", {}).get("path", ""))
    )
    runtime = _regular_record(runtime_path)
    source = _regular_record(source_path, ascii_required=True)
    if runtime != selection.get("runtime") or source != selection.get(
        "generated_source"
    ):
        raise AdapterError("discrete-v3 selection file records changed")
    if (
        runtime["sha256"] != V3_RUNTIME_SHA256
        or runtime["bytes"] != V3_RUNTIME_BYTES
        or source["sha256"] != V3_SOURCE_SHA256
        or source["bytes"] != V3_SOURCE_BYTES
    ):
        raise AdapterError("discrete-v3 runtime/source identity changed")
    if (
        selection.get("architecture") != v3.ARCHITECTURE
        or selection.get("arm") != "search-target"
        or selection.get("offline_gate", {}).get("passed") is not True
        or selection.get("selection_immutable") is not True
        or selection.get("selection_may_change_after_fresh_protected_tests")
        is not False
        or selection.get("game_gated") is not False
        or selection.get("upload_authorized") is not False
    ):
        raise AdapterError("discrete-v3 selection is not an immutable offline pass")
    _rerender_source(
        runtime_path=runtime_path, source_path=source_path, plan=plan,
        expected_export=selection.get("source_export"),
    )
    runtime_architecture, _weights, runtime_selection, runtime_document = (
        v3.compact.load_runtime(runtime_path)
    )
    dimensions = tuple(runtime_architecture.dimensions)
    derived_campaign_architecture = "-".join(str(value) for value in dimensions)
    if (
        selection.get("architecture") != runtime_architecture.name
        or runtime_architecture.name != EXPECTED_RUNTIME_ARCHITECTURE
        or dimensions != EXPECTED_DIMENSIONS
        or derived_campaign_architecture != EXPECTED_CAMPAIGN_ARCHITECTURE
        or runtime_document.get("architecture", {}).get("dimensions")
        != list(EXPECTED_DIMENSIONS)
        or runtime_selection.get("arm") != "search-target"
    ):
        raise AdapterError("discrete-v3 runtime architecture binding changed")

    outcome_path = output_root / "governance" / "02-outcome.json"
    return {
        "output_root": output_root,
        "plan_path": plan_path,
        "plan": plan,
        "plan_reference": _sealed_reference(plan_path, v3.PLAN_SCHEMA),
        "selection_reference_path": selection_reference_path,
        "selection_reference": _sealed_reference(
            selection_reference_path, v3.SELECTION_REFERENCE_SCHEMA
        ),
        "selection_path": selection_path,
        "selection": selection,
        "selection_artifact": _sealed_reference(selection_path, v3.SELECTION_SCHEMA),
        "outcome_reference": _sealed_reference(outcome_path, v3.OUTCOME_SCHEMA),
        "runtime_path": runtime_path,
        "runtime": runtime,
        "architecture": {
            "runtime_name": runtime_architecture.name,
            "dimensions": list(dimensions),
            "campaign_name": derived_campaign_architecture,
        },
        "source_path": source_path,
        "source": source,
    }


def _finite_metric_report(value: object, *, samples: int, label: str) -> None:
    expected = {
        "samples", "weighted_huber", "objective_weighted_huber",
        "sign_accuracy", "correlation", "mae", "prediction_mean",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise AdapterError(f"{label} fresh metric report has an unexpected shape")
    if value.get("samples") != samples:
        raise AdapterError(f"{label} fresh metric sample binding changed")
    for name in expected - {"samples"}:
        item = value.get(name)
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise AdapterError(f"{label} fresh metric {name} is not numeric")
        if not math.isfinite(float(item)):
            raise AdapterError(f"{label} fresh metric {name} is nonfinite")


def _fresh_report(
    report_reference_path: pathlib.Path, candidate: Mapping[str, Any],
) -> dict[str, Any]:
    del report_reference_path, candidate
    raise AdapterError(
        "legacy fresh reports are never acceptable adapter ancestry"
    )


def _path_present(path: pathlib.Path) -> bool:
    return path.is_symlink() or path.exists()


def _evaluation_paths(output_root: pathlib.Path) -> dict[str, pathlib.Path]:
    root = output_root / "development-adapter" / "evaluation-v2"
    return {
        "root": root,
        "lock": output_root / "development-adapter" / "evaluation-v2.lock",
        "claim": root / "00-claim.json",
        "report_reference": root / "report-reference.json",
        "reports": root / "reports",
        "completion": root / "01-completed.json",
    }


def _standard_evaluation_outputs(output_root: pathlib.Path) -> list[pathlib.Path]:
    holdout = output_root / "fresh-holdout"
    observed = []
    for path in (
        holdout / "report-reference.json",
        holdout / "evaluation-receipt.json",
        holdout / "evaluation-completion.json",
    ):
        if _path_present(path):
            observed.append(path)
    reports = holdout / "reports"
    # Any pre-existing node at the legacy reports route is prior/foreign
    # evaluation state.  Treat even an empty directory as poisoned so a
    # regular file, special node, or nested irregular entry cannot hide behind
    # directory-only traversal logic.
    if _path_present(reports):
        observed.append(reports)
    return observed


def _require_no_standard_evaluation(output_root: pathlib.Path) -> None:
    observed = _standard_evaluation_outputs(output_root)
    if observed:
        raise AdapterError(
            "direct/legacy fresh evaluation output is forbidden: "
            + ", ".join(str(path) for path in observed)
        )


def _v1_routes(output_root: pathlib.Path) -> dict[str, pathlib.Path]:
    root = output_root / "development-adapter"
    return {
        "evaluation_lock": root / "evaluation.lock",
        "evaluation_root": root / "evaluation",
        "handoff": root / "handoff.json",
    }


def _v1_evaluation_outputs(output_root: pathlib.Path) -> list[pathlib.Path]:
    observed = []
    for path in _v1_routes(output_root).values():
        if _path_present(path):
            observed.append(path)
    return observed


def _require_no_v1_evaluation(output_root: pathlib.Path) -> None:
    observed = _v1_evaluation_outputs(output_root)
    if observed:
        raise AdapterError(
            "retired v1 evaluation/handoff output is forbidden: "
            + ", ".join(str(path) for path in observed)
        )


def _adapter_evaluation_outputs(output_root: pathlib.Path) -> list[pathlib.Path]:
    paths = _evaluation_paths(output_root)
    observed = []
    root = paths["root"]
    if root.is_symlink():
        observed.append(root)
    elif root.exists():
        observed.extend(
            path for path in sorted(root.rglob("*"))
            if path.is_symlink() or path.is_file()
        )
        if not observed:
            observed.append(root)
    if _path_present(paths["lock"]):
        observed.append(paths["lock"])
    return observed


def _require_pristine_evaluation_routes(output_root: pathlib.Path) -> None:
    _require_no_standard_evaluation(output_root)
    _require_no_v1_evaluation(output_root)
    v2_handoff = output_root / "development-adapter" / "handoff-v2.json"
    if _path_present(v2_handoff):
        raise AdapterError("adapter v2 handoff exists before v2 planning")
    observed = _adapter_evaluation_outputs(output_root)
    if observed:
        raise AdapterError(
            "adapter evaluation route was not pristine before planning: "
            + ", ".join(str(path) for path in observed)
        )


@contextlib.contextmanager
def _exclusive_adapter_lock(path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise AdapterError("adapter evaluation lock is redirected or irregular")
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise AdapterError("another adapter evaluation is active") from error
        yield path
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _materialization_state(candidate: Mapping[str, Any]) -> dict[str, Any]:
    output_root = pathlib.Path(candidate["output_root"])
    path = output_root / "fresh-holdout" / "materialization-receipt.json"
    if path.is_symlink() or not path.is_file():
        raise AdapterError("fresh materialization receipt is absent or redirected")
    materialization = qualification.load_sealed(
        path, FRESH_MATERIALIZATION_SCHEMA
    )
    expected_selection = candidate["selection_artifact"]
    fields = {
        "schema", "namespace", "campaign_id", "status", "claim",
        "immutable_selection", "game_plan", "game_plan_tsv", "game_plan_rows",
        "fresh_roots", "fresh_roots_tsv", "fresh_opening_bank", "games",
        "games_manifest", "positions", "positions_manifest", "hard_positions",
        "search_labels", "rank4_labels", "canonical_labels",
        "canonical_label_rows", "packing_priors", "test_shards", "test_samples",
        "group_isolation", "split_isolation", "stage_receipts",
        "selection_changed", "old_protected_tests_accessed",
        "fresh_protected_tests_opened", "body_sha256",
    }
    test_shards = materialization.get("test_shards")
    samples = materialization.get("test_samples")
    if (
        set(materialization) != fields
        or materialization.get("namespace") != NAMESPACE
        or materialization.get("campaign_id")
        != f"{v3.SUCCESSOR_CAMPAIGN_ID}-holdout"
        or materialization.get("status")
        != "fresh-protected-holdout-materialized-once"
        or materialization.get("immutable_selection") != expected_selection
        or materialization.get("group_isolation", {}).get("passed") is not True
        or materialization.get("split_isolation", {}).get("passed") is not True
        or materialization.get("selection_changed") is not False
        or materialization.get("old_protected_tests_accessed") is not False
        or materialization.get("fresh_protected_tests_opened") is not True
        or not isinstance(test_shards, Mapping)
        or set(test_shards) != {"search", "rank4", "canonical"}
        or not isinstance(samples, Mapping)
        or set(samples) != set(test_shards)
    ):
        raise AdapterError("fresh materialization receipt ancestry/policy changed")
    for label, record in test_shards.items():
        _syntactic_file_record(record, label=f"fresh {label} test-shard")
    _syntactic_file_record(
        materialization.get("positions_manifest"),
        label="fresh protected position manifest",
    )
    for label, count in samples.items():
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise AdapterError(f"fresh {label} materialization sample count changed")

    claim_path = output_root / "fresh-holdout" / "00-materialization-claim.json"
    if claim_path.is_symlink() or not claim_path.is_file():
        raise AdapterError("fresh materialization claim is absent or redirected")
    claim = qualification.load_sealed(
        claim_path, FRESH_MATERIALIZATION_CLAIM_SCHEMA
    )
    claim_fields = {
        "schema", "namespace", "campaign_id", "status", "successor_plan",
        "immutable_selection", "selected_runtime", "prior_runtime",
        "configuration", "selection_may_change",
        "old_protected_tests_permitted", "materialization_attempts_authorized",
        "exclusive_process_lock", "claimed_at_utc", "body_sha256",
    }
    prior_record = candidate["plan"]["training"]["prior_compact_runtime"]
    prior_path = pathlib.Path(str(prior_record.get("path", "")))
    if (
        set(claim) != claim_fields
        or claim.get("namespace") != NAMESPACE
        or claim.get("campaign_id")
        != f"{v3.SUCCESSOR_CAMPAIGN_ID}-holdout"
        or claim.get("status")
        != "fresh-protected-holdout-materialization-claimed-once"
        or claim.get("successor_plan") != candidate["plan_reference"]
        or claim.get("immutable_selection") != expected_selection
        or claim.get("selected_runtime") != candidate["runtime"]
        or claim.get("prior_runtime") != _regular_record(prior_path)
        or claim.get("configuration")
        != candidate["plan"]["fresh_protected_holdout"]
        or claim.get("selection_may_change") is not False
        or claim.get("old_protected_tests_permitted") is not False
        or claim.get("materialization_attempts_authorized") != 1
        or claim.get("exclusive_process_lock")
        != str((output_root / "fresh-holdout/materialization.lock").resolve())
        or materialization.get("claim")
        != _sealed_reference(claim_path, FRESH_MATERIALIZATION_CLAIM_SCHEMA)
    ):
        raise AdapterError("fresh materialization claim ancestry changed")
    qualification._utc(claim.get("claimed_at_utc"), "materialization claim timestamp")
    return {
        "path": path,
        "artifact": _sealed_reference(path, FRESH_MATERIALIZATION_SCHEMA),
        "document": materialization,
        "claim_path": claim_path,
        "claim": claim,
        "claim_artifact": _sealed_reference(
            claim_path, FRESH_MATERIALIZATION_CLAIM_SCHEMA
        ),
    }


def _validate_diagnostic_values(
    *, samples: object, metrics: object, minimum: int,
    materialization_samples: Mapping[str, Any],
) -> tuple[dict[str, int], dict[str, Any]]:
    if (
        isinstance(minimum, bool) or not isinstance(minimum, int) or minimum <= 0
        or not isinstance(samples, Mapping)
        or set(samples) != {"search", "rank4", "canonical"}
        or dict(samples) != dict(materialization_samples)
        or not isinstance(metrics, Mapping)
        or set(metrics) != set(samples)
    ):
        raise AdapterError("adapter diagnostic sample/metric roster changed")
    normalized_samples = {}
    normalized_metrics = {}
    for label, count in samples.items():
        if (
            isinstance(count, bool) or not isinstance(count, int)
            or count < minimum
        ):
            raise AdapterError("adapter diagnostic sample floor changed")
        _finite_metric_report(metrics[label], samples=count, label=label)
        normalized_samples[label] = count
        normalized_metrics[label] = dict(metrics[label])
    return normalized_samples, normalized_metrics


def _protected_dataset(record: Mapping[str, Any], label: str) -> Any:
    manifest = pathlib.Path(str(record.get("path", "")))
    if _regular_record(manifest) != dict(record):
        raise AdapterError(f"fresh {label} protected manifest changed")
    document = v3.v1._canonical_json(manifest, f"fresh {label} protected manifest")
    shard = v3.large_training.load_csr_shard(manifest)
    if document.get("split") != "test" or shard.split != "test" or len(shard) <= 0:
        raise AdapterError(f"fresh {label} shard is not a nonempty test set")
    return v3.compact.Dataset(
        indptr=shard.indptr,
        indices=shard.indices,
        targets=shard.targets,
        weights=shard.weights,
        group_ids=shard.group_ids,
        split="test",
        source_manifest_sha256=_sha256_file(manifest),
        source_npz_sha256=shard.npz_sha256,
        source_route=str(manifest.resolve()),
    )


def _evaluate_protected_diagnostic(
    candidate: Mapping[str, Any], materialization_state: Mapping[str, Any],
) -> dict[str, Any]:
    architecture, quantized, _selection, _runtime = v3.compact.load_runtime(
        pathlib.Path(candidate["runtime_path"])
    )
    materialization = materialization_state["document"]
    datasets = {
        label: _protected_dataset(record, label)
        for label, record in materialization["test_shards"].items()
    }
    effective = quantized.effective()
    arm = v3.compact.ARMS["search-target"]
    metrics = {}
    for label, dataset in datasets.items():
        predictions = v3.compact.predict_dataset(
            effective, architecture, dataset, quantized=quantized
        )
        metrics[label] = v3.compact.metrics_from_predictions(
            predictions, dataset, arm
        )
    return {
        "samples": {label: len(dataset) for label, dataset in datasets.items()},
        "metrics": metrics,
    }


def _plan_tool_closure(candidate: Mapping[str, Any]) -> dict[str, Any]:
    tools = candidate["plan"].get("tools")
    if (
        not isinstance(tools, Mapping)
        or any(name not in tools for name in TOOL_CLOSURE_KEYS)
    ):
        raise AdapterError("discrete-v3 adapter tool closure is incomplete")
    return {
        "adapter": _regular_record(pathlib.Path(__file__).resolve()),
        "adapter_tests": _regular_record(ADAPTER_TEST_PATH),
        **{name: dict(tools[name]) for name in TOOL_CLOSURE_KEYS},
    }


def _validate_exact_v1_plan(output_root: pathlib.Path) -> dict[str, Any]:
    output_root = output_root.resolve()
    path = output_root / "development-adapter" / "adapter-plan.json"
    if path.is_symlink() or not path.is_file():
        raise AdapterError("sealed adapter v1 plan is absent or redirected")
    raw = path.read_bytes()
    plan = qualification.load_sealed(path, V1_ADAPTER_PLAN_SCHEMA)
    if (
        len(raw) != V1_PLAN_BYTES
        or hashlib.sha256(raw).hexdigest() != V1_PLAN_SHA256
        or plan.get("body_sha256") != V1_PLAN_BODY_SHA256
        or plan.get("status") != ADAPTER_PLAN_STATUS
        or plan.get("campaign_id") != v3.SUCCESSOR_CAMPAIGN_ID
    ):
        raise AdapterError("sealed adapter v1 plan identity changed")
    tool = plan.get("tool_closure", {}).get("adapter")
    tests = plan.get("tool_closure", {}).get("adapter_tests")
    expected_tool = {
        "path": str((HERE / "compact_value_bfm_discrete_v3_adapter.py").resolve()),
        "bytes": V1_ADAPTER_BYTES,
        "sha256": V1_ADAPTER_SHA256,
    }
    expected_tests = {
        "path": str(ADAPTER_TEST_PATH.resolve()),
        "bytes": V1_TEST_BYTES,
        "sha256": V1_TEST_SHA256,
    }
    candidate = plan.get("candidate")
    if (
        tool != expected_tool
        or tests != expected_tests
        or not isinstance(candidate, Mapping)
        or candidate.get("architecture") != "6301-8-8-1"
        or candidate.get("runtime_architecture") != EXPECTED_RUNTIME_ARCHITECTURE
        or candidate.get("selection", {}).get("sha256") != V3_SELECTION_SHA256
        or candidate.get("runtime", {}).get("sha256") != V3_RUNTIME_SHA256
        or candidate.get("generated_source", {}).get("sha256") != V3_SOURCE_SHA256
        or plan.get("policy", {}).get("fresh_protected_tests_opened") is not False
        or plan.get("expected_fresh_evidence", {}).get(
            "adapter_owned_metrics_opened_at_prepare"
        ) is not False
    ):
        raise AdapterError("sealed adapter v1 defect/tool closure changed")
    runtime_path = pathlib.Path(str(candidate["runtime"]["path"]))
    runtime_architecture, _weights, _selection, runtime = v3.compact.load_runtime(
        runtime_path
    )
    dimensions = tuple(runtime_architecture.dimensions)
    if (
        runtime_architecture.name != EXPECTED_RUNTIME_ARCHITECTURE
        or dimensions != EXPECTED_DIMENSIONS
        or runtime.get("architecture", {}).get("dimensions")
        != list(EXPECTED_DIMENSIONS)
    ):
        raise AdapterError("sealed adapter v1 actual runtime architecture changed")
    return {
        "path": path,
        "plan": plan,
        "artifact": _sealed_reference(path, V1_ADAPTER_PLAN_SCHEMA),
        "tool": expected_tool,
        "tests": expected_tests,
        "declared_architecture": candidate["architecture"],
        "runtime_architecture": runtime_architecture.name,
        "dimensions": list(dimensions),
        "derived_architecture": "-".join(str(value) for value in dimensions),
    }


def _retirement_tool_closure() -> dict[str, Any]:
    return {
        "adapter": _regular_record(pathlib.Path(__file__).resolve()),
        "adapter_tests": _regular_record(ADAPTER_TEST_PATH),
    }


def _retirement_absent_routes(output_root: pathlib.Path) -> list[str]:
    holdout = output_root / "fresh-holdout"
    paths = [
        *_v1_routes(output_root).values(),
        holdout / "report-reference.json",
        holdout / "reports",
        holdout / "evaluation-receipt.json",
        holdout / "evaluation-completion.json",
    ]
    return [str(path.resolve()) for path in paths]


def _v1_retirement_body(
    output_root: pathlib.Path, v1: Mapping[str, Any], *, retired_at_utc: str,
) -> dict[str, Any]:
    return {
        "schema": V1_RETIREMENT_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": v3.SUCCESSOR_CAMPAIGN_ID,
        "status": V1_RETIREMENT_STATUS,
        "retired_at_utc": retired_at_utc,
        "v1_plan": v1["artifact"],
        "v1_plan_body_sha256": V1_PLAN_BODY_SHA256,
        "v1_tool_closure": {
            "adapter": dict(v1["tool"]),
            "adapter_tests": dict(v1["tests"]),
        },
        "defect": {
            "field": "candidate.architecture",
            "incorrect_declared_architecture": v1["declared_architecture"],
            "runtime_architecture": v1["runtime_architecture"],
            "runtime_dimensions": list(v1["dimensions"]),
            "correct_derived_architecture": v1["derived_architecture"],
            "reason": "v1-hardcoded-primary-instead-of-runtime-derived-capacity",
        },
        "absent_routes": _retirement_absent_routes(output_root),
        "adapter_evaluation_started": False,
        "protected_metrics_opened": False,
        "legacy_evaluation_observed": False,
        "replacement": {
            "adapter_plan_schema": ADAPTER_PLAN_SCHEMA,
            "adapter_plan_path": str(
                (output_root / "development-adapter/adapter-plan-v2.json").resolve()
            ),
            "evaluation_root": str(
                (output_root / "development-adapter/evaluation-v2").resolve()
            ),
            "handoff_schema": HANDOFF_SCHEMA,
            "handoff_path": str(
                (output_root / "development-adapter/handoff-v2.json").resolve()
            ),
        },
        "retirement_tool_closure": _retirement_tool_closure(),
        "rank4_gate_authorized": False,
        "upload_authorized": False,
    }


V1PlanLoader = Callable[[pathlib.Path], Mapping[str, Any]]


def retire_v1(
    output: pathlib.Path, *, output_root: pathlib.Path, retired_at_utc: str,
    v1_plan_loader: V1PlanLoader = _validate_exact_v1_plan,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    expected = output_root / "development-adapter" / "v1-retirement.json"
    if output.resolve() != expected:
        raise AdapterError("adapter v1 retirement path is not canonical")
    if output.exists():
        return validate_v1_retirement(
            output, output_root=output_root, v1_plan_loader=v1_plan_loader
        )
    _require_no_v1_evaluation(output_root)
    _require_no_standard_evaluation(output_root)
    v2_paths = _evaluation_paths(output_root)
    for path in (
        output_root / "development-adapter/adapter-plan-v2.json",
        output_root / "development-adapter/handoff-v2.json",
        v2_paths["lock"], v2_paths["root"],
    ):
        if _path_present(path):
            raise AdapterError("adapter v2 state exists before v1 retirement")
    qualification._utc(retired_at_utc, "adapter v1 retirement timestamp")
    v1 = dict(v1_plan_loader(output_root))
    qualification.write_sealed(
        output,
        _v1_retirement_body(output_root, v1, retired_at_utc=retired_at_utc),
    )
    return validate_v1_retirement(
        output, output_root=output_root, v1_plan_loader=v1_plan_loader
    )


def validate_v1_retirement(
    path: pathlib.Path, *, output_root: pathlib.Path,
    v1_plan_loader: V1PlanLoader = _validate_exact_v1_plan,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    expected_path = output_root / "development-adapter" / "v1-retirement.json"
    if path.is_symlink() or not path.is_file() or path.resolve() != expected_path:
        raise AdapterError("adapter v1 retirement path changed")
    _require_no_v1_evaluation(output_root)
    _require_no_standard_evaluation(output_root)
    receipt = qualification.load_sealed(path, V1_RETIREMENT_SCHEMA)
    if set(receipt) != V1_RETIREMENT_FIELDS:
        raise AdapterError("adapter v1 retirement field roster changed")
    qualification._utc(receipt.get("retired_at_utc"), "adapter v1 retirement timestamp")
    v1 = dict(v1_plan_loader(output_root))
    expected = qualification.seal(_v1_retirement_body(
        output_root, v1, retired_at_utc=str(receipt["retired_at_utc"])
    ))
    if receipt != expected:
        raise AdapterError("adapter v1 retirement content changed")
    return receipt


def _planned_development_contract() -> dict[str, Any]:
    return {
        "mode": "discrete-v3-post-holdout",
        "candidate_id": CANDIDATE_ID,
        "rank4_control_required": True,
        "stages": dict(DEVELOPMENT_STAGES),
        "actual_clock": {
            "candidate_wins_min": 211,
            "candidate_color_wins_min": 104,
            "failures_required": 0,
        },
        "required_output_schema": (
            "papersoccer.compact-value-bfm."
            "discrete-v3-post-holdout-finalist.v1"
        ),
        "fresh_position_symmetry_exclusion_audit": {
            "required_before_development": True,
            "completed": False,
            "evidence_schema": (
                "papersoccer.compact-value-bfm."
                "discrete-v3-fresh-position-exclusion-audit.v1"
            ),
            "source": "validated-materialization.positions_manifest",
            "equivalences": [
                "exact", "rotate", "reflect", "rotate_reflect",
            ],
            "must_exclude_from": [
                "all-six-development-banks", "protected-final-bank",
            ],
        },
        "strict_rank4_after_development_selection": {
            "rank4_sha256": qualification.RANK4_SHA256,
            "pairs": qualification.FINAL_OPENINGS,
            "games": qualification.FINAL_GAMES,
            "candidate_wins_min": qualification.MINIMUM_WINS,
            "candidate_color_wins_min": qualification.MINIMUM_COLOR_WINS,
            "zero_failures_required": True,
        },
    }


def _validated_candidate_architecture(candidate: Mapping[str, Any]) -> dict[str, Any]:
    value = candidate.get("architecture")
    selection = candidate.get("selection")
    if not isinstance(value, Mapping) or not isinstance(selection, Mapping):
        raise AdapterError("adapter candidate architecture binding is missing")
    dimensions = value.get("dimensions")
    if (
        value.get("runtime_name") != EXPECTED_RUNTIME_ARCHITECTURE
        or dimensions != list(EXPECTED_DIMENSIONS)
        or value.get("campaign_name") != EXPECTED_CAMPAIGN_ARCHITECTURE
        or "-".join(str(item) for item in dimensions or [])
        != value.get("campaign_name")
        or selection.get("architecture") != value.get("runtime_name")
    ):
        raise AdapterError("adapter candidate architecture is not runtime-derived")
    return dict(value)


def _adapter_plan_body(
    candidate: Mapping[str, Any], *, retirement_reference: Mapping[str, Any],
    planned_at_utc: str,
) -> dict[str, Any]:
    selection = candidate["selection"]
    architecture = _validated_candidate_architecture(candidate)
    output_root = pathlib.Path(candidate["output_root"])
    evaluation = _evaluation_paths(output_root)
    return {
        "schema": ADAPTER_PLAN_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": v3.SUCCESSOR_CAMPAIGN_ID,
        "status": ADAPTER_PLAN_STATUS,
        "planned_at_utc": planned_at_utc,
        "v1_retirement": dict(retirement_reference),
        "v3_plan": candidate["plan_reference"],
        "v3_selection_reference": candidate["selection_reference"],
        "v3_selection": candidate["selection_artifact"],
        "v3_outcome": candidate["outcome_reference"],
        "candidate": {
            "candidate_id": CANDIDATE_ID,
            "architecture": architecture["campaign_name"],
            "runtime_architecture": architecture["runtime_name"],
            "dimensions": list(architecture["dimensions"]),
            "target": "search-target",
            "selection": candidate["selection_artifact"],
            "runtime": dict(candidate["runtime"]),
            "generated_source": dict(candidate["source"]),
            "source_export": dict(selection["source_export"]),
            "offline_gate": dict(selection["offline_gate"]),
        },
        "expected_fresh_evidence": {
            "evaluation_lock_path": str(evaluation["lock"].resolve()),
            "evaluation_claim_path": str(evaluation["claim"].resolve()),
            "evaluation_claim_schema": EVALUATION_CLAIM_SCHEMA,
            "adapter_report_reference_path": str(
                evaluation["report_reference"].resolve()
            ),
            "adapter_report_reference_schema": ADAPTER_REPORT_REFERENCE_SCHEMA,
            "adapter_report_schema": ADAPTER_REPORT_SCHEMA,
            "evaluation_completion_path": str(evaluation["completion"].resolve()),
            "evaluation_completion_schema": EVALUATION_COMPLETION_SCHEMA,
            "materialization_path": str(
                (output_root / "fresh-holdout/materialization-receipt.json").resolve()
            ),
            "materialization_schema": FRESH_MATERIALIZATION_SCHEMA,
            "legacy_report_reference_path": str(
                (output_root / "fresh-holdout/report-reference.json").resolve()
            ),
            "legacy_reports_directory": str(
                (output_root / "fresh-holdout/reports").resolve()
            ),
            "legacy_evaluation_outputs_accepted": False,
            "classification": "diagnostic-only-no-pass-fail-verdict",
            "metric_dependent_branch_authorized": False,
            "no_evaluation_outputs_observed_at_prepare": True,
            "adapter_owned_metrics_opened_at_prepare": False,
        },
        "fixed_handoff": {
            "path": str(
                (output_root / "development-adapter/handoff-v2.json").resolve()
            ),
            "schema": HANDOFF_SCHEMA,
            "status": HANDOFF_STATUS,
        },
        "development_contract": _planned_development_contract(),
        "tool_closure": _plan_tool_closure(candidate),
        "policy": {
            "candidate_artifacts_immutable": True,
            "model_selection_may_change": False,
            "fresh_protected_tests_opened": False,
            "old_protected_tests_accessed": False,
            "development_screen_required": True,
            "development_selected": False,
            "rank4_final_bank_generation_authorized": False,
            "rank4_gate_authorized": False,
            "upload_authorized": False,
        },
    }


CandidateLoader = Callable[[pathlib.Path, pathlib.Path], Mapping[str, Any]]
RetirementValidator = Callable[..., Mapping[str, Any]]


def prepare_adapter(
    output: pathlib.Path, *, plan_path: pathlib.Path,
    output_root: pathlib.Path, retirement_path: pathlib.Path,
    planned_at_utc: str,
    candidate_loader: CandidateLoader = _canonical_candidate,
    retirement_validator: RetirementValidator = validate_v1_retirement,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    expected_output = output_root / "development-adapter" / "adapter-plan-v2.json"
    if output.resolve() != expected_output:
        raise AdapterError("discrete-v3 adapter plan path is not canonical")
    if output.exists():
        return validate_adapter_plan(
            output, plan_path=plan_path, output_root=output_root,
            candidate_loader=candidate_loader,
            retirement_validator=retirement_validator,
        )
    _require_pristine_evaluation_routes(output_root)
    qualification._utc(planned_at_utc, "discrete-v3 adapter-plan timestamp")
    retirement_validator(retirement_path, output_root=output_root)
    retirement_reference = _sealed_reference(
        retirement_path, V1_RETIREMENT_SCHEMA
    )
    candidate = dict(candidate_loader(plan_path, output_root))
    qualification.write_sealed(
        output, _adapter_plan_body(
            candidate, retirement_reference=retirement_reference,
            planned_at_utc=planned_at_utc,
        )
    )
    return validate_adapter_plan(
        output, plan_path=plan_path, output_root=output_root,
        candidate_loader=candidate_loader,
        retirement_validator=retirement_validator,
    )


def validate_adapter_plan(
    path: pathlib.Path, *, plan_path: pathlib.Path, output_root: pathlib.Path,
    candidate_loader: CandidateLoader = _canonical_candidate,
    retirement_validator: RetirementValidator = validate_v1_retirement,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    expected_path = output_root / "development-adapter" / "adapter-plan-v2.json"
    if path.resolve() != expected_path or path.is_symlink() or not path.is_file():
        raise AdapterError("discrete-v3 adapter plan path changed")
    plan = qualification.load_sealed(path, ADAPTER_PLAN_SCHEMA)
    if set(plan) != ADAPTER_PLAN_FIELDS:
        raise AdapterError("discrete-v3 adapter plan field roster changed")
    qualification._utc(plan.get("planned_at_utc"), "discrete-v3 adapter-plan timestamp")
    retirement_record = plan.get("v1_retirement")
    if (
        not isinstance(retirement_record, Mapping)
        or set(retirement_record) != {"path", "sha256"}
    ):
        raise AdapterError("adapter v2 plan retirement binding is malformed")
    retirement_path = pathlib.Path(str(retirement_record["path"]))
    retirement_validator(retirement_path, output_root=output_root)
    retirement_reference = _sealed_reference(
        retirement_path, V1_RETIREMENT_SCHEMA
    )
    if retirement_record != retirement_reference:
        raise AdapterError("adapter v2 plan retirement binding changed")
    candidate = dict(candidate_loader(plan_path, output_root))
    expected = qualification.seal(_adapter_plan_body(
        candidate, retirement_reference=retirement_reference,
        planned_at_utc=str(plan["planned_at_utc"])
    ))
    if plan != expected:
        raise AdapterError("discrete-v3 adapter plan content changed")
    return plan


def _evaluation_claim_body(
    candidate: Mapping[str, Any], materialization: Mapping[str, Any],
    *, adapter_plan_reference: Mapping[str, Any], lock_path: pathlib.Path,
    claimed_at_utc: str,
) -> dict[str, Any]:
    return {
        "schema": EVALUATION_CLAIM_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": v3.SUCCESSOR_CAMPAIGN_ID,
        "status": EVALUATION_CLAIM_STATUS,
        "claimed_at_utc": claimed_at_utc,
        "adapter_plan": dict(adapter_plan_reference),
        "v3_plan": candidate["plan_reference"],
        "v3_selection": candidate["selection_artifact"],
        "runtime": dict(candidate["runtime"]),
        "materialization": materialization["artifact"],
        "materialization_claim": materialization["claim_artifact"],
        "tool_closure": _plan_tool_closure(candidate),
        "exclusive_adapter_lock": str(lock_path.resolve()),
        "evaluation_attempts_authorized": 1,
        "legacy_evaluator_outputs_accepted": False,
        "metric_dependent_branch_authorized": False,
        "diagnostic_only": True,
        "selection_may_change": False,
        "rank4_gate_authorized": False,
        "upload_authorized": False,
    }


def _diagnostic_report_body(
    candidate: Mapping[str, Any], materialization: Mapping[str, Any],
    *, adapter_plan_reference: Mapping[str, Any],
    claim_reference: Mapping[str, Any], samples: Mapping[str, int],
    metrics: Mapping[str, Any], evaluated_at_utc: str,
) -> dict[str, Any]:
    minimum = candidate["plan"]["fresh_protected_holdout"][
        "minimum_samples_per_report"
    ]
    return {
        "schema": ADAPTER_REPORT_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": v3.SUCCESSOR_CAMPAIGN_ID,
        "status": "adapter-owned-fresh-diagnostic-complete",
        "evaluated_at_utc": evaluated_at_utc,
        "adapter_plan": dict(adapter_plan_reference),
        "evaluation_claim": dict(claim_reference),
        "v3_selection": candidate["selection_artifact"],
        "runtime": dict(candidate["runtime"]),
        "materialization": materialization["artifact"],
        "materialization_claim": materialization["claim_artifact"],
        "samples": dict(samples),
        "metrics": {name: dict(value) for name, value in metrics.items()},
        "minimum_samples_per_report": minimum,
        "sample_floor_passed": True,
        "classification": "diagnostic-only-no-pass-fail-verdict",
        "selection_changed": False,
        "deployment_decision_changed": False,
        "old_protected_tests_accessed": False,
        "fresh_protected_tests_opened": True,
        "complete": True,
    }


def _evaluation_completion_body(
    candidate: Mapping[str, Any], materialization: Mapping[str, Any],
    *, adapter_plan_reference: Mapping[str, Any],
    claim_reference: Mapping[str, Any], report_reference: Mapping[str, Any],
    report_artifact: Mapping[str, Any], completed_at_utc: str,
) -> dict[str, Any]:
    return {
        "schema": EVALUATION_COMPLETION_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": v3.SUCCESSOR_CAMPAIGN_ID,
        "status": EVALUATION_COMPLETION_STATUS,
        "completed_at_utc": completed_at_utc,
        "adapter_plan": dict(adapter_plan_reference),
        "evaluation_claim": dict(claim_reference),
        "adapter_report_reference": dict(report_reference),
        "adapter_report": dict(report_artifact),
        "v3_selection": candidate["selection_artifact"],
        "runtime": dict(candidate["runtime"]),
        "materialization": materialization["artifact"],
        "materialization_claim": materialization["claim_artifact"],
        "tool_closure": _plan_tool_closure(candidate),
        "evaluation_attempts_consumed": 1,
        "legacy_evaluator_outputs_observed": [],
        "classification": "diagnostic-only-no-pass-fail-verdict",
        "selection_changed": False,
        "development_selected": False,
        "rank4_final_bank_generation_authorized": False,
        "rank4_gate_authorized": False,
        "upload_authorized": False,
    }


DiagnosticEvaluator = Callable[
    [Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]
]


def _adapter_evaluation_payloads(output_root: pathlib.Path) -> list[pathlib.Path]:
    root = _evaluation_paths(output_root)["root"]
    if root.is_symlink():
        return [root]
    if not root.exists():
        return []
    observed = [
        path for path in sorted(root.rglob("*"))
        if path.is_symlink() or path.is_file()
    ]
    return observed or [root]


def evaluate_adapter(
    *, adapter_plan_path: pathlib.Path, plan_path: pathlib.Path,
    output_root: pathlib.Path,
    evaluator: DiagnosticEvaluator = _evaluate_protected_diagnostic,
    candidate_loader: CandidateLoader = _canonical_candidate,
    retirement_validator: RetirementValidator = validate_v1_retirement,
    clock: Callable[[], str] = utc_now,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    paths = _evaluation_paths(output_root)
    lock_preexisting = _path_present(paths["lock"])
    with _exclusive_adapter_lock(paths["lock"]):
        if _path_present(paths["completion"]):
            return validate_evaluation_completion(
                paths["completion"], adapter_plan_path=adapter_plan_path,
                plan_path=plan_path, output_root=output_root,
                candidate_loader=candidate_loader,
                retirement_validator=retirement_validator,
            )
        if lock_preexisting:
            raise AdapterError(
                "partial or foreign adapter evaluation lock is not resumable"
            )
        existing = _adapter_evaluation_payloads(output_root)
        if existing:
            raise AdapterError(
                "partial or foreign adapter evaluation state is not resumable: "
                + ", ".join(str(path) for path in existing)
            )
        _require_no_standard_evaluation(output_root)
        validate_adapter_plan(
            adapter_plan_path, plan_path=plan_path, output_root=output_root,
            candidate_loader=candidate_loader,
            retirement_validator=retirement_validator,
        )
        candidate = dict(candidate_loader(plan_path, output_root))
        materialization = _materialization_state(candidate)
        claimed_at = clock()
        qualification._utc(claimed_at, "adapter evaluation claim timestamp")
        adapter_plan_reference = _sealed_reference(
            adapter_plan_path, ADAPTER_PLAN_SCHEMA
        )
        qualification.write_sealed(paths["claim"], _evaluation_claim_body(
            candidate, materialization,
            adapter_plan_reference=adapter_plan_reference,
            lock_path=paths["lock"], claimed_at_utc=claimed_at,
        ))
        claim_reference = _sealed_reference(
            paths["claim"], EVALUATION_CLAIM_SCHEMA
        )
        # This is the first operation permitted to open protected shard bytes.
        result = evaluator(candidate, materialization)
        if not isinstance(result, Mapping):
            raise AdapterError("adapter diagnostic evaluator returned no report")
        minimum = candidate["plan"]["fresh_protected_holdout"][
            "minimum_samples_per_report"
        ]
        samples, metrics = _validate_diagnostic_values(
            samples=result.get("samples"), metrics=result.get("metrics"),
            minimum=minimum,
            materialization_samples=materialization["document"]["test_samples"],
        )
        # A direct legacy evaluator that published while this one ran poisons
        # the attempt.  The sealed claim remains, making the partial state
        # deliberately non-resumable.
        _require_no_standard_evaluation(output_root)
        evaluated_at = clock()
        qualification._utc(evaluated_at, "adapter diagnostic timestamp")
        report_path, _report = v3._write_content_addressed(
            paths["reports"],
            _diagnostic_report_body(
                candidate, materialization,
                adapter_plan_reference=adapter_plan_reference,
                claim_reference=claim_reference,
                samples=samples, metrics=metrics,
                evaluated_at_utc=evaluated_at,
            ),
            ".adapter-diagnostic-report.json",
        )
        report_artifact = _sealed_reference(report_path, ADAPTER_REPORT_SCHEMA)
        qualification.write_sealed(paths["report_reference"], {
            "schema": ADAPTER_REPORT_REFERENCE_SCHEMA,
            "namespace": NAMESPACE,
            "campaign_id": v3.SUCCESSOR_CAMPAIGN_ID,
            "adapter_plan": adapter_plan_reference,
            "evaluation_claim": claim_reference,
            "report": {
                **_regular_record(report_path),
                "body_sha256": _report["body_sha256"],
            },
            "complete": True,
            "diagnostic_only": True,
            "legacy_evaluator_output": False,
        })
        _require_no_standard_evaluation(output_root)
        completed_at = clock()
        qualification._utc(completed_at, "adapter evaluation completion timestamp")
        qualification.write_sealed(
            paths["completion"],
            _evaluation_completion_body(
                candidate, materialization,
                adapter_plan_reference=adapter_plan_reference,
                claim_reference=claim_reference,
                report_reference=_sealed_reference(
                    paths["report_reference"], ADAPTER_REPORT_REFERENCE_SCHEMA
                ),
                report_artifact=report_artifact,
                completed_at_utc=completed_at,
            ),
        )
        # Do not trust the just-written chain without a complete independent
        # re-read.  A repeated evaluate call follows this same validator path.
        return validate_evaluation_completion(
            paths["completion"], adapter_plan_path=adapter_plan_path,
            plan_path=plan_path, output_root=output_root,
            candidate_loader=candidate_loader,
            retirement_validator=retirement_validator,
        )


def validate_evaluation_completion(
    path: pathlib.Path, *, adapter_plan_path: pathlib.Path,
    plan_path: pathlib.Path, output_root: pathlib.Path,
    candidate_loader: CandidateLoader = _canonical_candidate,
    retirement_validator: RetirementValidator = validate_v1_retirement,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    paths = _evaluation_paths(output_root)
    if path.is_symlink() or not path.is_file() or path.resolve() != paths[
        "completion"
    ]:
        raise AdapterError("adapter evaluation completion path changed")
    _require_no_standard_evaluation(output_root)
    validate_adapter_plan(
        adapter_plan_path, plan_path=plan_path, output_root=output_root,
        candidate_loader=candidate_loader,
        retirement_validator=retirement_validator,
    )
    candidate = dict(candidate_loader(plan_path, output_root))
    materialization = _materialization_state(candidate)
    adapter_plan_reference = _sealed_reference(
        adapter_plan_path, ADAPTER_PLAN_SCHEMA
    )

    if paths["claim"].is_symlink() or not paths["claim"].is_file():
        raise AdapterError("adapter evaluation claim is absent or redirected")
    claim = qualification.load_sealed(paths["claim"], EVALUATION_CLAIM_SCHEMA)
    qualification._utc(claim.get("claimed_at_utc"), "adapter evaluation claim timestamp")
    expected_claim = qualification.seal(_evaluation_claim_body(
        candidate, materialization,
        adapter_plan_reference=adapter_plan_reference,
        lock_path=paths["lock"],
        claimed_at_utc=str(claim["claimed_at_utc"]),
    ))
    if claim != expected_claim:
        raise AdapterError("adapter evaluation claim content changed")
    claim_reference = _sealed_reference(paths["claim"], EVALUATION_CLAIM_SCHEMA)

    if (
        paths["report_reference"].is_symlink()
        or not paths["report_reference"].is_file()
    ):
        raise AdapterError("adapter report reference is absent or redirected")
    reference = qualification.load_sealed(
        paths["report_reference"], ADAPTER_REPORT_REFERENCE_SCHEMA
    )
    reference_fields = {
        "schema", "namespace", "campaign_id", "adapter_plan",
        "evaluation_claim", "report", "complete", "diagnostic_only",
        "legacy_evaluator_output", "body_sha256",
    }
    record = reference.get("report")
    if (
        set(reference) != reference_fields
        or reference.get("namespace") != NAMESPACE
        or reference.get("campaign_id") != v3.SUCCESSOR_CAMPAIGN_ID
        or reference.get("adapter_plan") != adapter_plan_reference
        or reference.get("evaluation_claim") != claim_reference
        or reference.get("complete") is not True
        or reference.get("diagnostic_only") is not True
        or reference.get("legacy_evaluator_output") is not False
        or not isinstance(record, Mapping)
        or set(record) != {"path", "bytes", "sha256", "body_sha256"}
    ):
        raise AdapterError("adapter report reference content changed")
    report_path = pathlib.Path(str(record.get("path", "")))
    if (
        report_path.is_symlink() or not report_path.is_file()
        or report_path.parent != paths["reports"]
        or not report_path.name.endswith(".adapter-diagnostic-report.json")
        or report_path.name.removesuffix(".adapter-diagnostic-report.json")
        != record.get("sha256")
        or _regular_record(report_path) != {
            key: record[key] for key in ("path", "bytes", "sha256")
        }
    ):
        raise AdapterError("adapter diagnostic report path/content changed")
    report = qualification.load_sealed(report_path, ADAPTER_REPORT_SCHEMA)
    if report.get("body_sha256") != record.get("body_sha256"):
        raise AdapterError("adapter diagnostic report body binding changed")
    minimum = candidate["plan"]["fresh_protected_holdout"][
        "minimum_samples_per_report"
    ]
    samples, metrics = _validate_diagnostic_values(
        samples=report.get("samples"), metrics=report.get("metrics"),
        minimum=minimum,
        materialization_samples=materialization["document"]["test_samples"],
    )
    qualification._utc(report.get("evaluated_at_utc"), "adapter diagnostic timestamp")
    expected_report = qualification.seal(_diagnostic_report_body(
        candidate, materialization,
        adapter_plan_reference=adapter_plan_reference,
        claim_reference=claim_reference,
        samples=samples, metrics=metrics,
        evaluated_at_utc=str(report["evaluated_at_utc"]),
    ))
    if report != expected_report:
        raise AdapterError("adapter diagnostic report content changed")
    report_artifact = _sealed_reference(report_path, ADAPTER_REPORT_SCHEMA)

    completion = qualification.load_sealed(path, EVALUATION_COMPLETION_SCHEMA)
    qualification._utc(
        completion.get("completed_at_utc"), "adapter evaluation completion timestamp"
    )
    expected_completion = qualification.seal(_evaluation_completion_body(
        candidate, materialization,
        adapter_plan_reference=adapter_plan_reference,
        claim_reference=claim_reference,
        report_reference=_sealed_reference(
            paths["report_reference"], ADAPTER_REPORT_REFERENCE_SCHEMA
        ),
        report_artifact=report_artifact,
        completed_at_utc=str(completion["completed_at_utc"]),
    ))
    if completion != expected_completion:
        raise AdapterError("adapter evaluation completion content changed")
    _require_no_standard_evaluation(output_root)
    return {
        "completion": completion,
        "completion_artifact": _sealed_reference(
            path, EVALUATION_COMPLETION_SCHEMA
        ),
        "claim": claim,
        "claim_artifact": claim_reference,
        "reference": reference,
        "reference_artifact": _sealed_reference(
            paths["report_reference"], ADAPTER_REPORT_REFERENCE_SCHEMA
        ),
        "report": report,
        "report_artifact": report_artifact,
        "materialization": materialization["document"],
        "materialization_artifact": materialization["artifact"],
    }


def _handoff_body(
    candidate: Mapping[str, Any], report_state: Mapping[str, Any],
    *, adapter_plan_reference: Mapping[str, Any], created_at_utc: str,
) -> dict[str, Any]:
    selection = candidate["selection"]
    architecture = _validated_candidate_architecture(candidate)
    adapter_plan = qualification.load_sealed(
        pathlib.Path(str(adapter_plan_reference["path"])), ADAPTER_PLAN_SCHEMA
    )
    report = report_state["report"]
    materialization = report_state["materialization"]
    planned_contract = _planned_development_contract()
    exclusion = dict(planned_contract["fresh_position_symmetry_exclusion_audit"])
    exclusion["fresh_position_manifest"] = dict(
        materialization["positions_manifest"]
    )
    del exclusion["source"]
    planned_contract["fresh_position_symmetry_exclusion_audit"] = exclusion
    return {
        "schema": HANDOFF_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": v3.SUCCESSOR_CAMPAIGN_ID,
        "status": HANDOFF_STATUS,
        "created_at_utc": created_at_utc,
        "adapter_plan": dict(adapter_plan_reference),
        "v1_retirement": dict(adapter_plan["v1_retirement"]),
        "evaluation_claim": report_state["claim_artifact"],
        "evaluation_completion": report_state["completion_artifact"],
        "v3_plan": candidate["plan_reference"],
        "v3_selection_reference": candidate["selection_reference"],
        "v3_selection": candidate["selection_artifact"],
        "v3_outcome": candidate["outcome_reference"],
        "fresh_report_reference": report_state["reference_artifact"],
        "fresh_report": report_state["report_artifact"],
        "materialization": report_state["materialization_artifact"],
        "candidate": {
            "candidate_id": CANDIDATE_ID,
            "architecture": architecture["campaign_name"],
            "runtime_architecture": architecture["runtime_name"],
            "dimensions": list(architecture["dimensions"]),
            "target": "search-target",
            "selection": candidate["selection_artifact"],
            "runtime": dict(candidate["runtime"]),
            "generated_source": dict(candidate["source"]),
            "source_export": dict(selection["source_export"]),
        },
        "diagnostic_evidence": {
            "classification": "diagnostic-only-no-pass-fail-verdict",
            "report": report_state["report_artifact"],
            "sample_floor_passed": True,
            "samples": dict(report["samples"]),
            "metrics": dict(report["metrics"]),
        },
        "development_contract": planned_contract,
        "tool_closure": _plan_tool_closure(candidate),
        "policy": {
            "candidate_artifacts_immutable": True,
            "model_selection_may_change": False,
            "fresh_protected_tests_opened": True,
            "old_protected_tests_accessed": False,
            "development_screen_required": True,
            "development_selected": False,
            "rank4_final_bank_generation_authorized": False,
            "rank4_gate_authorized": False,
            "upload_authorized": False,
        },
    }


def create_handoff(
    output: pathlib.Path, *, adapter_plan_path: pathlib.Path,
    plan_path: pathlib.Path,
    output_root: pathlib.Path, evaluation_completion_path: pathlib.Path,
    created_at_utc: str, candidate_loader: CandidateLoader = _canonical_candidate,
    retirement_validator: RetirementValidator = validate_v1_retirement,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    expected_output = output_root / "development-adapter" / "handoff-v2.json"
    if output.resolve() != expected_output:
        raise AdapterError("discrete-v3 development handoff path is not canonical")
    qualification._utc(created_at_utc, "discrete-v3 adapter timestamp")
    validate_adapter_plan(
        adapter_plan_path, plan_path=plan_path, output_root=output_root,
        candidate_loader=candidate_loader,
        retirement_validator=retirement_validator,
    )
    candidate = dict(candidate_loader(plan_path, output_root))
    report_state = validate_evaluation_completion(
        evaluation_completion_path,
        adapter_plan_path=adapter_plan_path,
        plan_path=plan_path,
        output_root=output_root,
        candidate_loader=candidate_loader,
        retirement_validator=retirement_validator,
    )
    body = _handoff_body(
        candidate, report_state,
        adapter_plan_reference=_sealed_reference(
            adapter_plan_path, ADAPTER_PLAN_SCHEMA
        ),
        created_at_utc=created_at_utc,
    )
    handoff = qualification.write_sealed(output, body)
    return validate_handoff(
        output, adapter_plan_path=adapter_plan_path,
        plan_path=plan_path, output_root=output_root,
        evaluation_completion_path=evaluation_completion_path,
        candidate_loader=candidate_loader,
        retirement_validator=retirement_validator,
    )


def validate_handoff(
    path: pathlib.Path, *, adapter_plan_path: pathlib.Path,
    plan_path: pathlib.Path,
    output_root: pathlib.Path, evaluation_completion_path: pathlib.Path,
    candidate_loader: CandidateLoader = _canonical_candidate,
    retirement_validator: RetirementValidator = validate_v1_retirement,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    expected_path = output_root / "development-adapter" / "handoff-v2.json"
    if path.resolve() != expected_path or path.is_symlink() or not path.is_file():
        raise AdapterError("discrete-v3 development handoff path changed")
    handoff = qualification.load_sealed(path, HANDOFF_SCHEMA)
    if set(handoff) != HANDOFF_FIELDS:
        raise AdapterError("discrete-v3 development handoff field roster changed")
    qualification._utc(handoff.get("created_at_utc"), "discrete-v3 adapter timestamp")
    validate_adapter_plan(
        adapter_plan_path, plan_path=plan_path, output_root=output_root,
        candidate_loader=candidate_loader,
        retirement_validator=retirement_validator,
    )
    candidate = dict(candidate_loader(plan_path, output_root))
    report_state = validate_evaluation_completion(
        evaluation_completion_path,
        adapter_plan_path=adapter_plan_path,
        plan_path=plan_path,
        output_root=output_root,
        candidate_loader=candidate_loader,
        retirement_validator=retirement_validator,
    )
    expected = qualification.seal(_handoff_body(
        candidate, report_state,
        adapter_plan_reference=_sealed_reference(
            adapter_plan_path, ADAPTER_PLAN_SCHEMA
        ),
        created_at_utc=str(handoff["created_at_utc"]),
    ))
    if handoff != expected:
        raise AdapterError("discrete-v3 development handoff content changed")
    return handoff


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    retire = commands.add_parser("retire-v1")
    retire.add_argument("--output-root", type=pathlib.Path, required=True)
    retire.add_argument("--output", type=pathlib.Path, required=True)
    retire.add_argument("--retired-at-utc", default=utc_now())
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--plan", type=pathlib.Path, required=True)
    prepare.add_argument("--output-root", type=pathlib.Path, required=True)
    prepare.add_argument("--v1-retirement", type=pathlib.Path, required=True)
    prepare.add_argument("--output", type=pathlib.Path, required=True)
    prepare.add_argument("--planned-at-utc", default=utc_now())
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--plan", type=pathlib.Path, required=True)
    evaluate.add_argument("--output-root", type=pathlib.Path, required=True)
    evaluate.add_argument("--adapter-plan", type=pathlib.Path, required=True)
    for name in ("create", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--plan", type=pathlib.Path, required=True)
        command.add_argument("--output-root", type=pathlib.Path, required=True)
        command.add_argument("--adapter-plan", type=pathlib.Path, required=True)
        command.add_argument(
            "--evaluation-completion", type=pathlib.Path, required=True
        )
        command.add_argument("--output", type=pathlib.Path, required=True)
        if name == "create":
            command.add_argument("--created-at-utc", default=utc_now())
    args = parser.parse_args(argv)
    try:
        if args.command == "retire-v1":
            result = retire_v1(
                args.output,
                output_root=args.output_root,
                retired_at_utc=args.retired_at_utc,
            )
        elif args.command == "prepare":
            result = prepare_adapter(
                args.output,
                plan_path=args.plan,
                output_root=args.output_root,
                retirement_path=args.v1_retirement,
                planned_at_utc=args.planned_at_utc,
            )
        elif args.command == "evaluate":
            result = evaluate_adapter(
                adapter_plan_path=args.adapter_plan,
                plan_path=args.plan,
                output_root=args.output_root,
            )
        elif args.command == "create":
            result = create_handoff(
                args.output,
                adapter_plan_path=args.adapter_plan,
                plan_path=args.plan,
                output_root=args.output_root,
                evaluation_completion_path=args.evaluation_completion,
                created_at_utc=args.created_at_utc,
            )
        else:
            result = validate_handoff(
                args.output,
                adapter_plan_path=args.adapter_plan,
                plan_path=args.plan,
                output_root=args.output_root,
                evaluation_completion_path=args.evaluation_completion,
            )
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (AdapterError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"compact discrete-v3 adapter failure: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
