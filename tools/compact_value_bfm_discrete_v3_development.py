#!/usr/bin/env python3
"""Plan, validate, and finalize discrete-v3 post-holdout development.

This module is a standalone bridge.  It does not alter the frozen family
campaign, opening generator, maintained development runner, v3 campaign,
adapter, holdout, or exclusion tools.  It accepts only the exact v2 adapter
handoff and a development-ready four-way exclusion receipt, then preserves the
existing six-stage adaptive tuple/profile/actual-clock algorithm.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import pathlib
import tempfile
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
REPOSITORY = HERE.parent
BOT_ROOT = REPOSITORY / "submissions/codingame/bots/compact_value_bfm"
RUNNER_PATH = BOT_ROOT / "discrete_v3_development_runner.py"
MAINTAINED_RUNNER_PATH = BOT_ROOT / "development_runner.py"
TEST_PATH = (
    REPOSITORY / "tests/codingame/test_compact_value_bfm_discrete_v3_development.py"
)
if str(BOT_ROOT) not in sys.path:
    sys.path.insert(0, str(BOT_ROOT))


def _load(path: pathlib.Path, name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load discrete-v3 development dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


qualification = _load(
    HERE / "compact_value_bfm_qualification.py",
    "compact_v3_development_qualification",
)
adapter = _load(
    HERE / "compact_value_bfm_discrete_v3_adapter.py",
    "compact_v3_development_adapter",
)
exclusions = _load(
    HERE / "compact_value_bfm_discrete_v3_exclusions.py",
    "compact_v3_development_exclusions",
)
maintained = _load(
    MAINTAINED_RUNNER_PATH, "compact_v3_development_maintained_runner"
)
campaign = maintained.campaign


DevelopmentError = qualification.QualificationError
NAMESPACE = adapter.NAMESPACE
CAMPAIGN_ID = adapter.v3.SUCCESSOR_CAMPAIGN_ID
PLAN_SCHEMA = "papersoccer.compact-value-bfm.discrete-v3-development-plan.v1"
REQUEST_SCHEMA = "papersoccer.compact-value-bfm.discrete-v3-development-request.v1"
RECEIPT_SCHEMA = "papersoccer.compact-value-bfm.discrete-v3-development-receipt.v1"
RECEIPT_REFERENCE_SCHEMA = (
    "papersoccer.compact-value-bfm.discrete-v3-development-receipt-reference.v1"
)
RESULT_SCHEMA = "papersoccer.compact-value-bfm.discrete-v3-development-result.v1"
FINALIST_SCHEMA = (
    "papersoccer.compact-value-bfm.discrete-v3-post-holdout-finalist.v1"
)
FINALIST_REFERENCE_SCHEMA = (
    "papersoccer.compact-value-bfm.discrete-v3-post-holdout-finalist-reference.v1"
)

CANDIDATE_ID = adapter.CANDIDATE_ID
CONTROL_ID = "rank4-control"
CAPACITY_ARCHITECTURE = adapter.EXPECTED_CAMPAIGN_ARCHITECTURE
CONTROL_ARCHITECTURE = campaign.PRIMARY_ARCHITECTURE
STAGE_PAIRS = dict(maintained.STAGE_PAIRS)
STAGE_ORDER = tuple(STAGE_PAIRS)
RANK4_NODES = 3_000_000
BOOTSTRAP_SAMPLES = 20_000
RANK4_CONTROL_SELECTION_SHA256 = (
    "b27854708f205578689f9a4726f9dbae264e33a8c9e4c310ee44acd97fc8b04f"
)
RANK4_CONTROL_SELECTION_BODY_SHA256 = (
    "77ce4bcb6de4f9a0e9f8bff6be32c5baaddc56445cb968d02707d4d4590749b9"
)
RANK4_CONTROL_RUNTIME_SHA256 = (
    "41661543c6314c378368298ebe15ef0008c465d6c2ed157b993552df81455d84"
)
RANK4_CONTROL_SELECTION_PATH = (
    REPOSITORY / "results/compact_value_bfm/compact-value-bfm-20260831-v1"
    "/family-run-v4-source-bound/campaigns/compact-8x8--rank4-control"
    f"/selections/{RANK4_CONTROL_SELECTION_SHA256}.selection.json"
).resolve()

PLAN_FIELDS = {
    "schema", "namespace", "campaign_id", "status", "created_at_utc",
    "adapter", "exclusion", "candidate", "rank4_control", "banks",
    "algorithm", "compiler", "tools", "request_ancestry", "outputs",
    "policy", "body_sha256",
}
RESULT_FIELDS = {
    "schema", "namespace", "campaign_id", "status", "completed_at_utc",
    "development_plan", "binaries", "rows", "selected", "run_receipts",
    "request_count", "policy", "body_sha256",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _utc(value: Any, label: str) -> str:
    qualification._utc(value, label)
    return str(value)


def _regular(path: pathlib.Path, *, ascii_required: bool = False) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise DevelopmentError(f"{path} is absent, irregular, or redirected")
    raw = path.read_bytes()
    if ascii_required:
        try:
            raw.decode("ascii")
        except UnicodeDecodeError as error:
            raise DevelopmentError(f"{path} is not ASCII") from error
    return {
        "path": str(path.resolve()),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _verify_record(value: Any, label: str) -> pathlib.Path:
    if not isinstance(value, Mapping) or set(value) != {"path", "bytes", "sha256"}:
        raise DevelopmentError(f"{label} file record is malformed")
    path = pathlib.Path(str(value.get("path", "")))
    if _regular(path) != dict(value):
        raise DevelopmentError(f"{label} bytes changed")
    return path.resolve()


def _runtime_identity(value: Mapping[str, Any], label: str) -> dict[str, str]:
    path = _verify_record(value, label)
    runtime = qualification.load_sealed(path)
    payload = runtime.get("quantization", {}).get("payload_sha256")
    if not isinstance(payload, str) or len(payload) != 64:
        raise DevelopmentError(f"{label} omits its payload identity")
    return {
        "body_sha256": runtime["body_sha256"],
        "payload_sha256": payload,
    }


def _sealed_record(path: pathlib.Path, schema: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise DevelopmentError(f"sealed artifact is absent or redirected: {path}")
    value = qualification.load_sealed(path, schema)
    raw = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "body_sha256": value["body_sha256"],
        "schema": schema,
    }


def _verify_sealed_record(value: Any, schema: str, label: str) -> pathlib.Path:
    if not isinstance(value, Mapping) or set(value) != {
        "path", "bytes", "sha256", "body_sha256", "schema",
    } or value.get("schema") != schema:
        raise DevelopmentError(f"{label} sealed record is malformed")
    path = pathlib.Path(str(value.get("path", "")))
    if _sealed_record(path, schema) != dict(value):
        raise DevelopmentError(f"{label} sealed artifact changed")
    return path.resolve()


def _write_content_addressed(
    directory: pathlib.Path, body: Mapping[str, Any], suffix: str,
) -> tuple[pathlib.Path, dict[str, Any]]:
    artifact = qualification.seal(body)
    raw = qualification.canonical_json_bytes(artifact)
    path = directory / f"{qualification.sha256_bytes(raw)}{suffix}"
    qualification.atomic_write_once(path, raw)
    return path, artifact


def _tool_closure() -> dict[str, Any]:
    paths = {
        "development_tool": pathlib.Path(__file__).resolve(),
        "development_runner": RUNNER_PATH,
        "development_test": TEST_PATH,
        "maintained_runner": MAINTAINED_RUNNER_PATH,
        "gate_source": maintained.GATE_SOURCE,
        "rank4_source": maintained.RANK4,
        "rank4_gate_support": BOT_ROOT / "rank4_gate_support.py",
        "model_exporter": BOT_ROOT / "export_model.py",
        "submission_exporter": BOT_ROOT / "export_submission.py",
        "adapter_tool": pathlib.Path(adapter.__file__).resolve(),
        "adapter_test": adapter.ADAPTER_TEST_PATH,
        "exclusion_tool": pathlib.Path(exclusions.__file__).resolve(),
        "exclusion_test": exclusions.TEST_PATH,
        "qualification_tool": pathlib.Path(qualification.__file__).resolve(),
        "campaign_tool": pathlib.Path(campaign.__file__).resolve(),
        "openings_tool": pathlib.Path(maintained.openings.__file__).resolve(),
    }
    return {name: _regular(path) for name, path in paths.items()}


AdapterValidator = Callable[[pathlib.Path, pathlib.Path], Mapping[str, Any]]
ExclusionValidator = Callable[
    [pathlib.Path, pathlib.Path, pathlib.Path], Mapping[str, Any]
]
ControlValidator = Callable[[pathlib.Path], Mapping[str, Any]]
CompilerIdentity = Callable[[], Mapping[str, Any]]


def _default_adapter_validator(
    handoff_path: pathlib.Path, output_root: pathlib.Path,
) -> dict[str, Any]:
    expected = output_root / "development-adapter/handoff-v2.json"
    if handoff_path.is_symlink() or not handoff_path.is_file() or handoff_path.resolve() != expected:
        raise DevelopmentError("v2 adapter handoff path changed")
    handoff = qualification.load_sealed(handoff_path, adapter.HANDOFF_SCHEMA)
    adapter_plan_path = pathlib.Path(str(handoff.get("adapter_plan", {}).get("path", "")))
    v3_plan_path = pathlib.Path(str(handoff.get("v3_plan", {}).get("path", "")))
    completion_path = pathlib.Path(
        str(handoff.get("evaluation_completion", {}).get("path", ""))
    )
    validated = adapter.validate_handoff(
        handoff_path,
        adapter_plan_path=adapter_plan_path,
        plan_path=v3_plan_path,
        output_root=output_root,
        evaluation_completion_path=completion_path,
    )
    candidate = validated.get("candidate")
    contract = validated.get("development_contract")
    policy = validated.get("policy")
    if (
        not isinstance(candidate, Mapping)
        or candidate.get("candidate_id") != CANDIDATE_ID
        or candidate.get("architecture") != CAPACITY_ARCHITECTURE
        or candidate.get("runtime_architecture") != adapter.EXPECTED_RUNTIME_ARCHITECTURE
        or candidate.get("dimensions") != list(adapter.EXPECTED_DIMENSIONS)
        or candidate.get("target") != "search-target"
        or not isinstance(contract, Mapping)
        or contract.get("mode") != "discrete-v3-post-holdout"
        or contract.get("required_output_schema") != FINALIST_SCHEMA
        or contract.get("fresh_position_symmetry_exclusion_audit", {}).get(
            "evidence_schema"
        ) != exclusions.RECEIPT_SCHEMA
        or not isinstance(policy, Mapping)
        or policy.get("fresh_protected_tests_opened") is not True
        or policy.get("old_protected_tests_accessed") is not False
        or policy.get("development_screen_required") is not True
        or policy.get("development_selected") is not False
        or policy.get("rank4_final_bank_generation_authorized") is not False
        or policy.get("rank4_gate_authorized") is not False
        or policy.get("upload_authorized") is not False
    ):
        raise DevelopmentError("v2 adapter handoff is not development-only capacity")
    return {
        "handoff": validated,
        "handoff_path": handoff_path.resolve(),
        "adapter_plan_path": adapter_plan_path.resolve(),
        "adapter_plan": qualification.load_sealed(
            adapter_plan_path, adapter.ADAPTER_PLAN_SCHEMA
        ),
        "candidate": dict(candidate),
    }


def _default_exclusion_validator(
    plan_path: pathlib.Path, receipt_path: pathlib.Path,
    output_root: pathlib.Path,
) -> dict[str, Any]:
    validated = exclusions.validate_receipt(
        receipt_path,
        plan_path=plan_path,
        output_root=output_root,
        recompute_positions=True,
    )
    if validated.get("development_ready") is not True:
        raise DevelopmentError("fresh-position exclusion did not authorize development")
    return dict(validated)


def _default_control_validator(selection_path: pathlib.Path) -> dict[str, Any]:
    selection = qualification.load_sealed(
        selection_path, maintained.SELECTION_SCHEMA
    )
    reference = {
        "path": str(selection_path.resolve()),
        "sha256": qualification.sha256_file(selection_path),
        "body_sha256": selection["body_sha256"],
    }
    details = campaign.validate_rank4_control_reference(reference)
    runtime_path = pathlib.Path(details["runtime_path"])
    if (
        selection_path.resolve() != RANK4_CONTROL_SELECTION_PATH
        or qualification.sha256_file(selection_path)
        != RANK4_CONTROL_SELECTION_SHA256
        or selection.get("body_sha256")
        != RANK4_CONTROL_SELECTION_BODY_SHA256
        or details["runtime"].get("sha256") != RANK4_CONTROL_RUNTIME_SHA256
    ):
        raise DevelopmentError("Rank-4 control is not the exact frozen selection")
    declared_runtime = details.get("runtime")
    if (
        not isinstance(declared_runtime, Mapping)
        or set(declared_runtime) != {"path", "bytes", "sha256"}
        or pathlib.PurePath(str(declared_runtime["path"])).is_absolute()
    ):
        raise DevelopmentError("Rank-4 control runtime declaration is not relative")
    campaign_root = selection_path.resolve().parent.parent
    declared_resolved = (campaign_root / str(declared_runtime["path"])).resolve()
    absolute_runtime = _regular(runtime_path)
    if (
        declared_resolved != runtime_path.resolve()
        or declared_runtime.get("bytes") != absolute_runtime["bytes"]
        or declared_runtime.get("sha256") != absolute_runtime["sha256"]
    ):
        raise DevelopmentError("Rank-4 control relative runtime ancestry changed")
    with tempfile.TemporaryDirectory() as temporary:
        header = maintained._development_header(
            runtime_path, "rank4-control", pathlib.Path(temporary)
        )
        _default, source = maintained.export_submission.render(model_header=header)
    return {
        **details,
        "runtime_declaration": dict(declared_runtime),
        "runtime": absolute_runtime,
        "selection_record": _sealed_record(
            selection_path, maintained.SELECTION_SCHEMA
        ),
        "rendered_source": {
            "bytes": len(source),
            "sha256": hashlib.sha256(source).hexdigest(),
        },
    }


def _algorithm_contract() -> dict[str, Any]:
    return {
        "stage_pairs": dict(STAGE_PAIRS),
        "stage_order": list(STAGE_ORDER),
        "model_roster": [CANDIDATE_ID, CONTROL_ID],
        "deployable_model_id": CANDIDATE_ID,
        "rank4_control_id": CONTROL_ID,
        "tuple_roster": [list(value) for value in campaign.TUPLE_ROSTER],
        "default_tuple": list(campaign.DEFAULT_TUPLE),
        "tuple_retention": "best-two-plus-default",
        "profile_roster": {
            name: dict(work) for name, work in campaign.PROFILE_ROSTER.items()
        },
        "default_profile": campaign.DEFAULT_PROFILE,
        "profile_retention": "best-two-plus-default",
        "paired_bootstrap_samples": BOOTSTRAP_SAMPLES,
        "actual_clock_thresholds": {
            "candidate_wins_min": 211,
            "candidate_color_wins_min": 104,
            "failures": 0,
        },
        "rank4_nodes": RANK4_NODES,
        "candidate_expansions": 2_000_000,
        "candidate_actions": 250,
        "candidate_shuffle_seed": 1,
        "max_turns": 320,
    }


def _request_ancestry(
    *, adapter_context: Mapping[str, Any],
    exclusion_context: Mapping[str, Any], control_context: Mapping[str, Any],
    tools: Mapping[str, Any], banks: Mapping[str, Any],
) -> dict[str, Any]:
    handoff = adapter_context["handoff"]
    exclusion = exclusion_context["receipt"]
    return {
        "adapter_plan_sha256": handoff["adapter_plan"]["sha256"],
        "adapter_handoff_sha256": qualification.sha256_file(
            adapter_context["handoff_path"]
        ),
        "adapter_report_sha256": handoff["fresh_report"]["sha256"],
        "adapter_completion_sha256": handoff["evaluation_completion"]["sha256"],
        "v3_selection_sha256": handoff["v3_selection"]["sha256"],
        "candidate_runtime_sha256": handoff["candidate"]["runtime"]["sha256"],
        "candidate_source_sha256": handoff["candidate"]["generated_source"]["sha256"],
        "adapter_tool_sha256": tools["adapter_tool"]["sha256"],
        "adapter_test_sha256": tools["adapter_test"]["sha256"],
        "exclusion_plan_sha256": qualification.sha256_file(
            exclusion_context["plan_path"]
        ),
        "exclusion_receipt_sha256": qualification.sha256_file(
            exclusion_context["receipt_path"]
        ),
        "exclusion_payload_sha256": exclusion[
            "references"
        ]["protected_canonical_fingerprints"]["sha256"],
        "exclusion_tool_sha256": tools["exclusion_tool"]["sha256"],
        "exclusion_test_sha256": tools["exclusion_test"]["sha256"],
        "development_tool_sha256": tools["development_tool"]["sha256"],
        "development_runner_sha256": tools["development_runner"]["sha256"],
        "development_test_sha256": tools["development_test"]["sha256"],
        "maintained_runner_sha256": tools["maintained_runner"]["sha256"],
        "qualification_tool_sha256": tools["qualification_tool"]["sha256"],
        "campaign_tool_sha256": tools["campaign_tool"]["sha256"],
        "openings_tool_sha256": tools["openings_tool"]["sha256"],
        "gate_source_sha256": tools["gate_source"]["sha256"],
        "rank4_source_sha256": tools["rank4_source"]["sha256"],
        "rank4_control_selection_sha256": control_context[
            "selection_record"
        ]["sha256"],
        "rank4_control_runtime_sha256": control_context["runtime"]["sha256"],
        "rank4_control_source_sha256": control_context["rendered_source"]["sha256"],
        "bank_manifest_sha256": {
            stage: banks[stage]["sha256"] for stage in STAGE_ORDER
        },
    }


def _plan_body(
    *, output_root: pathlib.Path, adapter_context: Mapping[str, Any],
    exclusion_context: Mapping[str, Any], control_context: Mapping[str, Any],
    compiler: Mapping[str, Any], created_at_utc: str,
) -> dict[str, Any]:
    handoff_value = adapter_context.get("handoff")
    exclusion_value = exclusion_context.get("receipt")
    bank_values = exclusion_context.get("development_bank_records")
    if (
        not isinstance(handoff_value, Mapping)
        or handoff_value.get("schema") != adapter.HANDOFF_SCHEMA
        or handoff_value.get("candidate", {}).get("candidate_id") != CANDIDATE_ID
        or handoff_value.get("candidate", {}).get("architecture")
        != CAPACITY_ARCHITECTURE
        or handoff_value.get("policy", {}).get("development_selected") is not False
        or handoff_value.get("policy", {}).get("rank4_gate_authorized") is not False
        or not isinstance(exclusion_value, Mapping)
        or exclusion_value.get("schema") != exclusions.RECEIPT_SCHEMA
        or exclusion_context.get("development_ready") is not True
        or not isinstance(bank_values, Mapping)
        or set(bank_values) != set(STAGE_ORDER)
        or control_context.get("selection", {}).get("arm") != "rank4-control"
        or control_context.get("selection", {}).get("deployment_eligible") is not False
    ):
        raise DevelopmentError("development prerequisite context is incomplete")
    tools = _tool_closure()
    banks = exclusion_context["development_bank_records"]
    handoff = adapter_context["handoff"]
    exclusion_plan_path = exclusion_context["plan_path"]
    exclusion_receipt_path = exclusion_context["receipt_path"]
    development_root = output_root / "development-v3"
    return {
        "schema": PLAN_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID,
        "status": "development-planned-awaiting-games",
        "created_at_utc": _utc(created_at_utc, "development plan timestamp"),
        "adapter": {
            "plan": _sealed_record(
                adapter_context["adapter_plan_path"], adapter.ADAPTER_PLAN_SCHEMA
            ),
            "handoff": _sealed_record(
                adapter_context["handoff_path"], adapter.HANDOFF_SCHEMA
            ),
            "evaluation_claim": dict(handoff["evaluation_claim"]),
            "evaluation_completion": dict(handoff["evaluation_completion"]),
            "diagnostic_report_reference": dict(handoff["fresh_report_reference"]),
            "diagnostic_report": dict(handoff["fresh_report"]),
            "selection": dict(handoff["v3_selection"]),
        },
        "exclusion": {
            "plan": _sealed_record(exclusion_plan_path, exclusions.PLAN_SCHEMA),
            "receipt": _sealed_record(
                exclusion_receipt_path, exclusions.RECEIPT_SCHEMA
            ),
            "protected_fingerprints": dict(
                exclusion_context["receipt"]["references"][
                    "protected_canonical_fingerprints"
                ]
            ),
            "development_ready": True,
        },
        "candidate": {
            "candidate_id": CANDIDATE_ID,
            "architecture": CAPACITY_ARCHITECTURE,
            "target": "search-target",
            "selection": dict(handoff["candidate"]["selection"]),
            "runtime": dict(handoff["candidate"]["runtime"]),
            "generated_source": dict(handoff["candidate"]["generated_source"]),
            "source_export": dict(handoff["candidate"]["source_export"]),
            "runtime_identity": _runtime_identity(
                handoff["candidate"]["runtime"], "v3 candidate runtime"
            ),
        },
        "rank4_control": {
            "candidate_id": CONTROL_ID,
            "architecture": CONTROL_ARCHITECTURE,
            "target": campaign.CONTROL_TARGET,
            "selection": dict(control_context["selection_record"]),
            "runtime_declaration": dict(control_context["runtime_declaration"]),
            "runtime": dict(control_context["runtime"]),
            "rendered_source": dict(control_context["rendered_source"]),
            "runtime_identity": _runtime_identity(
                control_context["runtime"], "Rank-4 control runtime"
            ),
            "deployment_eligible": False,
        },
        "banks": {stage: dict(banks[stage]) for stage in STAGE_ORDER},
        "algorithm": _algorithm_contract(),
        "compiler": dict(compiler),
        "tools": tools,
        "request_ancestry": _request_ancestry(
            adapter_context=adapter_context,
            exclusion_context=exclusion_context,
            control_context=control_context,
            tools=tools,
            banks=banks,
        ),
        "outputs": {
            "development_root": str(development_root.resolve()),
            "requests": str((development_root / "requests").resolve()),
            "receipts": str((development_root / "receipts-v3").resolve()),
            "references": str((development_root / "run-references-v3").resolve()),
            "base_receipts": str((development_root / "receipts").resolve()),
            "base_references": str((development_root / "run-references").resolve()),
            "binaries": str((development_root / "gate-binaries").resolve()),
            "result": str((development_root / "development-result.json").resolve()),
            "finalists": str((development_root / "finalists").resolve()),
            "finalist_reference": str(
                (development_root / "finalist-reference.json").resolve()
            ),
        },
        "policy": {
            "fresh_protected_tests_opened_diagnostically": True,
            "diagnostic_metrics_are_not_development_acceptance": True,
            "exclusion_development_ready": True,
            "development_games_authorized": True,
            "selection_may_change_model_weights": False,
            "search_configuration_selection_pending": True,
            "final_bank_generation_authorized": False,
            "rank4_gate_authorized": False,
            "upload_authorized": False,
        },
    }


def prepare_plan(
    output_root: pathlib.Path, *, adapter_handoff_path: pathlib.Path,
    exclusion_plan_path: pathlib.Path, exclusion_receipt_path: pathlib.Path,
    rank4_control_selection_path: pathlib.Path, created_at_utc: str,
    adapter_validator: AdapterValidator = _default_adapter_validator,
    exclusion_validator: ExclusionValidator = _default_exclusion_validator,
    control_validator: ControlValidator = _default_control_validator,
    compiler_identity: CompilerIdentity = maintained._default_compiler_identity,
) -> pathlib.Path:
    output_root = output_root.resolve()
    if output_root != adapter.v3.canonical_v3_root():
        raise DevelopmentError("development plan requires the canonical v3 root")
    plan_path = output_root / "development-v3/plan.json"
    adapter_context = dict(adapter_validator(adapter_handoff_path, output_root))
    exclusion_context = dict(exclusion_validator(
        exclusion_plan_path, exclusion_receipt_path, output_root
    ))
    exclusion_context["plan_path"] = exclusion_plan_path.resolve()
    exclusion_context["receipt_path"] = exclusion_receipt_path.resolve()
    control_context = dict(control_validator(rank4_control_selection_path))
    compiler = dict(compiler_identity())
    body = _plan_body(
        output_root=output_root,
        adapter_context=adapter_context,
        exclusion_context=exclusion_context,
        control_context=control_context,
        compiler=compiler,
        created_at_utc=created_at_utc,
    )
    if plan_path.exists():
        if qualification.load_sealed(plan_path, PLAN_SCHEMA) != qualification.seal(body):
            raise DevelopmentError("existing v3 development plan changed")
    else:
        development_root = output_root / "development-v3"
        if development_root.is_symlink() or (
            development_root.exists()
            and any(development_root.iterdir())
        ):
            raise DevelopmentError("development output predates its sealed plan")
        qualification.write_sealed(plan_path, body)
    validate_plan(
        plan_path,
        output_root=output_root,
        adapter_validator=adapter_validator,
        exclusion_validator=exclusion_validator,
        control_validator=control_validator,
        compiler_identity=lambda: compiler,
    )
    return plan_path


def validate_plan(
    path: pathlib.Path, *, output_root: pathlib.Path,
    adapter_validator: AdapterValidator = _default_adapter_validator,
    exclusion_validator: ExclusionValidator = _default_exclusion_validator,
    control_validator: ControlValidator = _default_control_validator,
    compiler_identity: CompilerIdentity = maintained._default_compiler_identity,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    expected = output_root / "development-v3/plan.json"
    if path.is_symlink() or not path.is_file() or path.resolve() != expected:
        raise DevelopmentError("v3 development plan path changed")
    plan = qualification.load_sealed(path, PLAN_SCHEMA)
    if set(plan) != PLAN_FIELDS:
        raise DevelopmentError("v3 development plan field roster changed")
    adapter_handoff = pathlib.Path(str(plan["adapter"]["handoff"]["path"]))
    exclusion_plan = pathlib.Path(str(plan["exclusion"]["plan"]["path"]))
    exclusion_receipt = pathlib.Path(str(plan["exclusion"]["receipt"]["path"]))
    control_selection = pathlib.Path(str(plan["rank4_control"]["selection"]["path"]))
    adapter_context = dict(adapter_validator(adapter_handoff, output_root))
    exclusion_context = dict(exclusion_validator(
        exclusion_plan, exclusion_receipt, output_root
    ))
    exclusion_context["plan_path"] = exclusion_plan.resolve()
    exclusion_context["receipt_path"] = exclusion_receipt.resolve()
    control_context = dict(control_validator(control_selection))
    created_at = _utc(plan.get("created_at_utc"), "development plan timestamp")
    expected_plan = qualification.seal(_plan_body(
        output_root=output_root,
        adapter_context=adapter_context,
        exclusion_context=exclusion_context,
        control_context=control_context,
        compiler=dict(compiler_identity()),
        created_at_utc=created_at,
    ))
    if plan != expected_plan:
        raise DevelopmentError("v3 development plan content changed")
    return plan


def _metric(value: Any, *, pairs: int, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DevelopmentError(f"{label} metric is absent")
    required = {"candidate_id", "pairs", "games", "wins", "color_wins", "failures", "latency_ms"}
    if not required.issubset(value) or value.get("pairs") != pairs or value.get("games") != 2 * pairs:
        raise DevelopmentError(f"{label} metric shape/count changed")
    for name in ("wins", "failures"):
        if isinstance(value.get(name), bool) or not isinstance(value.get(name), int):
            raise DevelopmentError(f"{label} {name} is invalid")
    colors = value.get("color_wins")
    if not isinstance(colors, Mapping) or set(colors) != {"0", "1"}:
        raise DevelopmentError(f"{label} color wins changed")
    latency = value.get("latency_ms")
    if isinstance(latency, bool) or not isinstance(latency, (int, float)) or not math.isfinite(float(latency)):
        raise DevelopmentError(f"{label} latency is invalid")
    return dict(value)


def _result_selection(rows: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(rows, Mapping) or set(rows) != set(STAGE_ORDER):
        raise DevelopmentError("development result stage roster changed")
    model_rows = rows.get("model_screen")
    if not isinstance(model_rows, list) or len(model_rows) != 2:
        raise DevelopmentError("model screen must contain candidate and Rank-4 control")
    if [row.get("candidate_id") for row in model_rows] != [CANDIDATE_ID, CONTROL_ID]:
        raise DevelopmentError("model screen order changed")
    by_model = {row.get("candidate_id"): _metric(row, pairs=100, label="model screen") for row in model_rows if isinstance(row, Mapping)}
    if set(by_model) != {CANDIDATE_ID, CONTROL_ID}:
        raise DevelopmentError("model screen roster changed")
    candidate_model = by_model[CANDIDATE_ID]
    if (
        candidate_model.get("architecture") != CAPACITY_ARCHITECTURE
        or candidate_model.get("target") != "search-target"
        or candidate_model.get("deployment_eligible") is not True
        or candidate_model["failures"] != 0
        or by_model[CONTROL_ID].get("architecture") != CONTROL_ARCHITECTURE
        or by_model[CONTROL_ID].get("target") != campaign.CONTROL_TARGET
        or by_model[CONTROL_ID].get("deployment_eligible") is not False
    ):
        raise DevelopmentError("model screen candidate/control provenance changed")
    retained = [candidate_model]
    tuple_rows = rows.get("tuple_screen")
    if (
        not isinstance(tuple_rows, list)
        or [row.get("tuple") for row in tuple_rows]
        != [list(value) for value in campaign.TUPLE_ROSTER]
    ):
        raise DevelopmentError("tuple screen order/roster changed")
    tuple_ranked = campaign._validate_exact_tuple_screen(
        tuple_rows, retained, by_model
    )
    default_tuple_id = f"{CANDIDATE_ID}:{campaign.tuple_id(campaign.DEFAULT_TUPLE)}"
    carried_tuples = []
    for value in [row["candidate_id"] for row in tuple_ranked[:2]] + [default_tuple_id]:
        if value not in carried_tuples:
            carried_tuples.append(value)
    if [row.get("candidate_id") for row in rows.get("tuple_confirmation", [])] != carried_tuples:
        raise DevelopmentError("tuple confirmation order changed")
    tuple_architecture = {
        row["candidate_id"]: CAPACITY_ARCHITECTURE for row in tuple_rows
    }
    selected_tuple, normalized_tuple_confirmation = campaign._confirmation_choice(
        rows.get("tuple_confirmation"), pairs=250,
        carried_ids=carried_tuples, default_id=default_tuple_id,
        architecture_by_id=tuple_architecture, label="tuple",
    )
    profiles = campaign._validate_profiles(
        rows.get("profile_screen"), pairs=100, label="profile screen"
    )
    if [row.get("profile") for row in rows["profile_screen"]] != list(
        campaign.PROFILE_ROSTER
    ):
        raise DevelopmentError("profile screen order changed")
    ranked_profiles = sorted(
        [row for row in profiles if row["failures"] == 0],
        key=lambda row: campaign._rank_key(row, CAPACITY_ARCHITECTURE),
    )
    if len(ranked_profiles) < 2:
        raise DevelopmentError("profile screen has fewer than two failure-free profiles")
    carried_profiles = []
    for value in [row["candidate_id"] for row in ranked_profiles[:2]] + [campaign.DEFAULT_PROFILE]:
        if value not in carried_profiles:
            carried_profiles.append(value)
    if [row.get("profile") for row in rows.get("profile_confirmation", [])] != carried_profiles:
        raise DevelopmentError("profile confirmation order changed")
    profile_rows = campaign._validate_profiles(
        rows.get("profile_confirmation"), pairs=250,
        label="profile confirmation", expected_profiles=carried_profiles,
    )
    selected_profile, normalized_profile_confirmation = campaign._confirmation_choice(
        profile_rows, pairs=250, carried_ids=carried_profiles,
        default_id=campaign.DEFAULT_PROFILE,
        architecture_by_id={name: CAPACITY_ARCHITECTURE for name in carried_profiles},
        label="profile",
    )
    actual = _metric(rows.get("actual_clock"), pairs=200, label="actual clock")
    expected_actual = selected_tuple["candidate_id"] + ":" + selected_profile["candidate_id"]
    if (
        actual.get("candidate_id") != expected_actual
        or actual["failures"] != 0 or actual["wins"] < 211
        or actual["color_wins"]["0"] < 104
        or actual["color_wins"]["1"] < 104
    ):
        raise DevelopmentError("actual-clock 211/104/zero-failure gate failed")
    return {
        "model": candidate_model,
        "tuple": selected_tuple,
        "profile": selected_profile,
        "actual_clock": actual,
        "tuple_confirmation": normalized_tuple_confirmation,
        "profile_confirmation": normalized_profile_confirmation,
    }


def _finalist_body(
    *, plan_path: pathlib.Path, plan: Mapping[str, Any],
    result_path: pathlib.Path, result: Mapping[str, Any],
    selected: Mapping[str, Any], created_at_utc: str,
) -> dict[str, Any]:
    return {
        "schema": FINALIST_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID,
        "status": "development-selected-awaiting-preflight-and-frozen-final",
        "created_at_utc": created_at_utc,
        "development_plan": _sealed_record(plan_path, PLAN_SCHEMA),
        "development_result": _sealed_record(result_path, RESULT_SCHEMA),
        "adapter": dict(plan["adapter"]),
        "exclusion": dict(plan["exclusion"]),
        "candidate": dict(plan["candidate"]),
        "rank4_control": dict(plan["rank4_control"]),
        "banks": dict(plan["banks"]),
        "binary": dict(result["binaries"][CANDIDATE_ID]),
        "tuple": list(selected["tuple"]["tuple"]),
        "tuple_candidate_id": selected["tuple"]["candidate_id"],
        "profile": selected["profile"]["profile"],
        "profile_work": dict(selected["profile"]["work"]),
        "actual_clock": dict(selected["actual_clock"]),
        "run_receipts": list(result["run_receipts"]),
        "fresh_protected_tests_opened": True,
        "fresh_diagnostic_classification": "diagnostic-only-no-pass-fail-verdict",
        "old_protected_tests_accessed": False,
        "model_weights_immutable": True,
        "search_configuration_immutable": True,
        "development_selected": True,
        "preflight_required": True,
        "final_bank_generation_authorized": False,
        "rank4_gate_authorized": False,
        "upload_authorized": False,
    }


PlanValidator = Callable[..., Mapping[str, Any]]
ReceiptValidator = Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]


def _compile_key(plan: Mapping[str, Any], candidate: Mapping[str, Any]) -> str:
    return qualification.sha256_bytes(qualification.canonical_json_bytes({
        "development_plan_sha256": qualification.sha256_file(
            pathlib.Path(plan["outputs"]["development_root"]) / "plan.json"
        ),
        "gate_source_sha256": plan["tools"]["gate_source"]["sha256"],
        "candidate_id": candidate["candidate_id"],
        "selection_sha256": candidate["selection_sha256"],
        "runtime_sha256": candidate["runtime_sha256"],
        "candidate_source_sha256": candidate["source_sha256"],
        "rank4_source_sha256": plan["tools"]["rank4_source"]["sha256"],
        "compiler": plan["compiler"],
        "flags": ["-std=c++20", "-O3"],
    }))


def _validate_compile_reference(
    value: Any, *, plan: Mapping[str, Any], candidate: Mapping[str, Any],
    development_plan_record: Mapping[str, Any],
) -> dict[str, Any]:
    path = _verify_sealed_record(
        value, "papersoccer.compact-value-bfm.discrete-v3-development-"
        "binary-reference.v1", "compile reference",
    )
    compile_key = _compile_key(plan, candidate)
    binaries = pathlib.Path(plan["outputs"]["binaries"])
    if (
        path.parent != binaries
        or path.name != f"{compile_key}.v3-binary-reference.json"
    ):
        raise DevelopmentError("compile reference path/key changed")
    reference = qualification.load_sealed(path)
    expected_fields = {
        "schema", "namespace", "campaign_id", "compile_key",
        "development_plan", "candidate_id", "selection_sha256",
        "runtime_sha256", "candidate_source_sha256", "compiler",
        "gate_source", "rank4_source", "binary", "body_sha256",
    }
    binary = reference.get("binary")
    if (
        set(reference) != expected_fields
        or reference.get("namespace") != NAMESPACE
        or reference.get("campaign_id") != CAMPAIGN_ID
        or reference.get("compile_key") != compile_key
        or reference.get("development_plan") != development_plan_record
        or reference.get("candidate_id") != candidate["candidate_id"]
        or reference.get("selection_sha256") != candidate["selection_sha256"]
        or reference.get("runtime_sha256") != candidate["runtime_sha256"]
        or reference.get("candidate_source_sha256") != candidate["source_sha256"]
        or reference.get("compiler") != plan["compiler"]
        or reference.get("gate_source") != plan["tools"]["gate_source"]
        or reference.get("rank4_source") != plan["tools"]["rank4_source"]
        or not isinstance(binary, Mapping)
    ):
        raise DevelopmentError("compile reference content changed")
    binary_path = _verify_record(binary, "compiled gate binary")
    if (
        binary_path.parent != binaries
        or binary_path.name != f"{binary['sha256']}.rank4-gate"
        or candidate.get("binary_path") != str(binary_path)
        or candidate.get("binary_sha256") != binary["sha256"]
        or candidate.get("binary_bytes") != binary["bytes"]
    ):
        raise DevelopmentError("compiled gate binary escaped or changed name")
    return reference


def validate_run_receipt(
    reference: Mapping[str, Any], plan: Mapping[str, Any],
) -> dict[str, Any]:
    path = _verify_sealed_record(reference, RECEIPT_SCHEMA, "development run receipt")
    if (
        path.parent != pathlib.Path(plan["outputs"]["receipts"])
        or not path.name.endswith(".receipt.json")
        or path.name.removesuffix(".receipt.json") != qualification.sha256_file(path)
    ):
        raise DevelopmentError("development run receipt path is not content addressed")
    receipt = qualification.load_sealed(path, RECEIPT_SCHEMA)
    request_record = receipt.get("request")
    base_record = receipt.get("maintained_receipt")
    receipt_fields = {
        "schema", "namespace", "campaign_id", "development_plan", "request",
        "request_sha256", "maintained_receipt", "maintained_request_sha256",
        "gate_result_sha256", "gate_output", "compile_reference", "metric", "complete",
        "final_bank_generation_authorized", "rank4_gate_authorized",
        "upload_authorized", "body_sha256",
    }
    if (
        set(receipt) != receipt_fields
        or receipt.get("namespace") != NAMESPACE
        or receipt.get("campaign_id") != CAMPAIGN_ID
        or receipt.get("complete") is not True
        or receipt.get("final_bank_generation_authorized") is not False
        or receipt.get("rank4_gate_authorized") is not False
        or receipt.get("upload_authorized") is not False
        or not isinstance(request_record, Mapping)
        or not isinstance(base_record, Mapping)
    ):
        raise DevelopmentError("development run receipt ancestry changed")
    request_path = _verify_sealed_record(
        request_record, REQUEST_SCHEMA, "development request"
    )
    if (
        request_path.parent != pathlib.Path(plan["outputs"]["requests"])
        or not request_path.name.endswith(".request.json")
        or request_path.name.removesuffix(".request.json")
        != qualification.sha256_file(request_path)
    ):
        raise DevelopmentError("development request path is not content addressed")
    request = qualification.load_sealed(request_path, REQUEST_SCHEMA)
    request_fields = {
        "schema", "namespace", "campaign_id", "development_plan", "ancestry",
        "candidate", "bank", "spec", "metric_extra", "expected_configuration",
        "compile_reference", "compiler", "gate_source", "rank4_source", "body_sha256",
    }
    if (
        set(request) != request_fields
        or request.get("namespace") != NAMESPACE
        or request.get("campaign_id") != CAMPAIGN_ID
        or request.get("development_plan") != receipt.get("development_plan")
        or request.get("development_plan") != _sealed_record(
            pathlib.Path(plan["outputs"]["development_root"]) / "plan.json",
            PLAN_SCHEMA,
        )
        or request.get("ancestry") != plan["request_ancestry"]
        or request.get("compiler") != plan["compiler"]
        or request.get("gate_source") != plan["tools"]["gate_source"]
        or request.get("rank4_source") != plan["tools"]["rank4_source"]
        or receipt.get("request_sha256") != qualification.sha256_file(request_path)
    ):
        raise DevelopmentError("development request/receipt binding changed")
    candidate = request["candidate"]
    base_path = pathlib.Path(str(base_record.get("path", "")))
    if (
        base_path.is_symlink() or not base_path.is_file()
        or base_path.parent != pathlib.Path(plan["outputs"]["base_receipts"])
        or not base_path.name.endswith(".development-run.json")
        or base_path.name.removesuffix(".development-run.json")
        != qualification.sha256_file(base_path)
    ):
        raise DevelopmentError("maintained receipt path is not content addressed")
    base_raw, base_receipt = maintained.load_json(base_path, "maintained receipt")
    maintained.verify_body_hash(
        base_receipt, maintained.RUN_SCHEMA, "maintained receipt"
    )
    if (
        set(base_receipt) != {
            "schema", "namespace", "request_sha256", "request",
            "selection_sha256", "selection_body_sha256", "runtime_sha256",
            "gate_result", "body_sha256",
        }
        or set(base_record) != {"path", "bytes", "sha256", "body_sha256", "schema"}
        or base_record.get("schema") != maintained.RUN_SCHEMA
        or base_record.get("bytes") != len(base_raw)
        or base_record.get("sha256") != hashlib.sha256(base_raw).hexdigest()
        or base_record.get("body_sha256") != base_receipt["body_sha256"]
        or receipt.get("maintained_request_sha256") != base_receipt["request_sha256"]
        or base_receipt.get("selection_sha256") != candidate["selection_sha256"]
        or base_receipt.get("selection_body_sha256")
        != candidate["selection_body_sha256"]
        or base_receipt.get("runtime_sha256") != candidate["runtime_sha256"]
        or receipt.get("gate_result_sha256") != qualification.sha256_bytes(
            qualification.canonical_json_bytes(base_receipt["gate_result"])
        )
    ):
        raise DevelopmentError("maintained receipt binding changed")
    base_request = base_receipt["request"]
    spec = request["spec"]
    bank = request["bank"]
    values = spec.get("tuple")
    work = spec.get("work")
    if (
        not isinstance(values, list) or len(values) != 3
        or not isinstance(work, Mapping)
    ):
        raise DevelopmentError("development request search configuration is malformed")
    expected_configuration = {
        "mode": spec["mode"], "pair_offset": 0, "pair_count": spec["pairs"],
        "candidate_c": float(values[0]), "candidate_fpu": float(values[1]),
        "candidate_lambda": float(values[2]), "candidate_actions": 250,
        "candidate_root_partial_paths": work["root_partial_paths"],
        "candidate_nonroot_partial_paths": work["nonroot_partial_paths"],
        "candidate_nodes": work["nodes"], "candidate_expansions": 2_000_000,
        "candidate_shuffle_seed": 1, "candidate_clocks_ms": [800, 155],
        "rank4_nodes": RANK4_NODES, "rank4_clocks_ms": [800, 165],
        "max_turns": 320,
        "minimum_candidate_wins": 211 if spec["stage"] == "actual_clock" else -1,
        "minimum_wins_per_color": 104 if spec["stage"] == "actual_clock" else -1,
    }
    if request.get("expected_configuration") != expected_configuration:
        raise DevelopmentError("development request effective configuration changed")
    model_id = candidate.get("candidate_id")
    planned_candidate = (
        plan["candidate"] if model_id == CANDIDATE_ID
        else plan["rank4_control"] if model_id == CONTROL_ID else None
    )
    if not isinstance(planned_candidate, Mapping):
        raise DevelopmentError("development request names a foreign candidate")
    binary_path = pathlib.Path(str(candidate.get("binary_path", "")))
    selection_path = pathlib.Path(str(planned_candidate["selection"]["path"]))
    planned_selection = qualification.load_sealed(selection_path)
    if (
        set(candidate) != {
            "candidate_id", "architecture", "target", "selection_sha256",
            "selection_body_sha256", "runtime_sha256", "runtime_body_sha256",
            "payload_sha256", "source_sha256", "source_bytes", "binary_path",
            "binary_sha256", "binary_bytes",
        }
        or candidate.get("architecture") != planned_candidate["architecture"]
        or candidate.get("target") != planned_candidate["target"]
        or binary_path.is_symlink() or not binary_path.is_file()
        or candidate.get("binary_bytes") != binary_path.stat().st_size
        or candidate.get("binary_sha256") != qualification.sha256_file(binary_path)
        or candidate.get("selection_sha256")
        != planned_candidate["selection"]["sha256"]
        or candidate.get("selection_body_sha256")
        != planned_selection.get("body_sha256")
        or candidate.get("runtime_sha256") != planned_candidate["runtime"]["sha256"]
        or candidate.get("runtime_body_sha256")
        != planned_candidate["runtime_identity"]["body_sha256"]
        or candidate.get("payload_sha256")
        != planned_candidate["runtime_identity"]["payload_sha256"]
        or candidate.get("source_sha256")
        != (
            planned_candidate["generated_source"]["sha256"]
            if model_id == CANDIDATE_ID
            else planned_candidate["rendered_source"]["sha256"]
        )
        or candidate.get("source_bytes")
        != (
            planned_candidate["generated_source"]["bytes"]
            if model_id == CANDIDATE_ID
            else planned_candidate["rendered_source"]["bytes"]
        )
    ):
        raise DevelopmentError("development request candidate/binary binding changed")
    stage = spec.get("stage")
    expected_extra_keys = {
        "model_screen": {
            "architecture", "target", "source_bytes", "artifact_sha256",
            "deployment_eligible",
        },
        "tuple_screen": {"model_id", "tuple"},
        "tuple_confirmation": {"model_id", "tuple"},
        "profile_screen": {"profile", "work"},
        "profile_confirmation": {"profile", "work"},
        "actual_clock": set(),
    }
    metric_extra = request.get("metric_extra")
    if (
        stage not in STAGE_PAIRS
        or set(bank) != {
            "stage", "manifest_path", "manifest_bytes", "manifest_sha256",
            "gate_path", "gate_bytes", "gate_sha256",
        }
        or set(spec) != {"stage", "candidate_id", "mode", "tuple", "work", "pairs"}
        or not isinstance(metric_extra, Mapping)
        or set(metric_extra) != expected_extra_keys.get(stage, set())
        or spec.get("pairs") != STAGE_PAIRS[stage]
        or bank.get("stage") != stage
        or bank.get("manifest_path") != plan["banks"][stage]["path"]
        or bank.get("manifest_bytes") != plan["banks"][stage]["bytes"]
        or bank.get("manifest_sha256") != plan["banks"][stage]["sha256"]
        or bank.get("gate_sha256") != base_receipt["request"]["bank_sha256"]
        or pathlib.Path(str(bank.get("gate_path", ""))).is_symlink()
        or qualification.sha256_file(pathlib.Path(bank["gate_path"]))
        != bank.get("gate_sha256")
        or pathlib.Path(bank["gate_path"]).stat().st_size != bank.get("gate_bytes")
    ):
        raise DevelopmentError("development request bank/stage binding changed")
    if (
        (stage == "model_screen" and spec.get("candidate_id") != model_id)
        or (stage != "model_screen" and model_id != CANDIDATE_ID)
    ):
        raise DevelopmentError("development request model/stage roster changed")
    if stage == "model_screen":
        expected_metric_extra = {
            "architecture": candidate["architecture"],
            "target": candidate["target"],
            "source_bytes": candidate["source_bytes"],
            "artifact_sha256": candidate["runtime_sha256"],
            "deployment_eligible": model_id == CANDIDATE_ID,
        }
    elif stage in {"tuple_screen", "tuple_confirmation"}:
        expected_metric_extra = {
            "model_id": CANDIDATE_ID,
            "tuple": list(spec["tuple"]),
        }
    elif stage in {"profile_screen", "profile_confirmation"}:
        expected_metric_extra = {
            "profile": spec["candidate_id"],
            "work": dict(spec["work"]),
        }
    else:
        expected_metric_extra = {}
    if dict(metric_extra) != expected_metric_extra:
        raise DevelopmentError("development request metric descriptor changed")
    if receipt.get("compile_reference") != request.get("compile_reference"):
        raise DevelopmentError("development receipt compile reference changed")
    _validate_compile_reference(
        request.get("compile_reference"), plan=plan, candidate=candidate,
        development_plan_record=request["development_plan"],
    )
    expected_base = {
        "candidate_id": spec["candidate_id"],
        "model_candidate_id": candidate["candidate_id"],
        "selection_sha256": candidate["selection_sha256"],
        "selection_body_sha256": candidate["selection_body_sha256"],
        "runtime_sha256": candidate["runtime_sha256"],
        "candidate_source_sha256": candidate["source_sha256"],
        "binary_sha256": candidate["binary_sha256"],
        "rank4_source_sha256": maintained.RANK4_SHA256,
        "bank_sha256": bank["gate_sha256"],
        "bank_manifest_sha256": bank["manifest_sha256"],
        "stage": spec["stage"], "mode": spec["mode"],
        "tuple": spec["tuple"], "work": spec["work"], "pairs": spec["pairs"],
    }
    if any(base_request.get(key) != value for key, value in expected_base.items()):
        raise DevelopmentError("maintained request differs from v3 request")
    gate_output_path = _verify_record(
        receipt.get("gate_output"), "development gate output"
    )
    if gate_output_path.parent != pathlib.Path(plan["outputs"]["development_root"]) / "scratch":
        raise DevelopmentError("development gate output escaped its scratch root")
    validated_gate = maintained.gate_support.validate_result(
        gate_output_path,
        expected_bank_sha256=bank["gate_sha256"],
        expected_candidate_sha256=candidate["source_sha256"],
        allow_legacy_attempt_zero=True,
    )
    if validated_gate != base_receipt.get("gate_result"):
        raise DevelopmentError("maintained receipt differs from validated gate output")
    gate = base_receipt.get("gate_result")
    if (
        not isinstance(gate, Mapping)
        or gate.get("bindings", {}).get("candidate_source_sha256")
        != candidate["source_sha256"]
        or gate.get("bindings", {}).get("candidate_source_bytes")
        != candidate["source_bytes"]
        or gate.get("bindings", {}).get("candidate_runtime_body_sha256")
        != candidate["runtime_body_sha256"]
        or gate.get("bindings", {}).get("candidate_payload_sha256")
        != candidate["payload_sha256"]
        or gate.get("bindings", {}).get("bank_sha256") != bank["gate_sha256"]
        or gate.get("bindings", {}).get("bank_bytes") != bank["gate_bytes"]
        or gate.get("bindings", {}).get("rank4_source_sha256")
        != maintained.RANK4_SHA256
        or gate.get("bindings", {}).get("rank4_source_bytes")
        != maintained.RANK4.stat().st_size
        or maintained.gate_support.legacy_standard_configuration(gate)
        != request.get("expected_configuration")
    ):
        raise DevelopmentError("development gate binding/configuration changed")
    metric = maintained._metric(
        base_receipt["gate_result"], spec["candidate_id"], spec["pairs"],
        **dict(request["metric_extra"]),
    )
    if receipt.get("metric") != metric:
        raise DevelopmentError("development receipt metric changed")
    return {
        "receipt": receipt,
        "request": request,
        "base_receipt": base_receipt,
        "metric": metric,
    }


def _validate_rows_against_receipts(
    rows: Mapping[str, Any], validated: Sequence[Mapping[str, Any]],
) -> None:
    by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for item in validated:
        request = item["request"]
        spec = request["spec"]
        key = (str(spec["stage"]), str(spec["candidate_id"]))
        if key in by_key:
            raise DevelopmentError("development receipt stage/candidate is repeated")
        by_key[key] = item
    expected_keys: set[tuple[str, str]] = set()
    for stage in STAGE_ORDER:
        value = rows.get(stage)
        stage_rows = [value] if stage == "actual_clock" else value
        if not isinstance(stage_rows, list):
            raise DevelopmentError(f"{stage} result rows are malformed")
        for row in stage_rows:
            if not isinstance(row, Mapping):
                raise DevelopmentError(f"{stage} result row is malformed")
            key = (stage, str(row.get("candidate_id")))
            item = by_key.get(key)
            if item is None or key in expected_keys:
                raise DevelopmentError("result row has no unique development receipt")
            expected_keys.add(key)
            normalized = dict(row)
            normalized.pop("paired_bootstrap_lower_95", None)
            if normalized != item["metric"]:
                raise DevelopmentError("result row differs from its development receipt")
    if expected_keys != set(by_key):
        raise DevelopmentError("development result omits or adds receipt evidence")

    tuple_rows = rows["tuple_confirmation"]
    default_tuple_id = f"{CANDIDATE_ID}:{campaign.tuple_id(campaign.DEFAULT_TUPLE)}"
    default_tuple_gate = by_key[("tuple_confirmation", default_tuple_id)][
        "base_receipt"
    ]["gate_result"]
    tuple_bank = by_key[("tuple_confirmation", default_tuple_id)]["request"]["bank"]
    for row in tuple_rows:
        identifier = row["candidate_id"]
        expected = (
            0.0 if identifier == default_tuple_id else
            maintained.paired_bootstrap_lower(
                by_key[("tuple_confirmation", identifier)]["base_receipt"]["gate_result"],
                default_tuple_gate,
                f"tuple:{tuple_bank['gate_sha256']}:{identifier}",
                samples=BOOTSTRAP_SAMPLES,
            )
        )
        if row.get("paired_bootstrap_lower_95") != expected:
            raise DevelopmentError("tuple bootstrap evidence changed")

    default_profile_gate = by_key[(
        "profile_confirmation", campaign.DEFAULT_PROFILE
    )]["base_receipt"]["gate_result"]
    profile_bank = by_key[(
        "profile_confirmation", campaign.DEFAULT_PROFILE
    )]["request"]["bank"]
    for row in rows["profile_confirmation"]:
        profile = row["candidate_id"]
        expected = (
            0.0 if profile == campaign.DEFAULT_PROFILE else
            maintained.paired_bootstrap_lower(
                by_key[("profile_confirmation", profile)]["base_receipt"]["gate_result"],
                default_profile_gate,
                f"profile:{profile_bank['gate_sha256']}:{profile}",
                samples=BOOTSTRAP_SAMPLES,
            )
        )
        if row.get("paired_bootstrap_lower_95") != expected:
            raise DevelopmentError("profile bootstrap evidence changed")


def finalize_result(
    *, plan_path: pathlib.Path, result_path: pathlib.Path,
    output_root: pathlib.Path, created_at_utc: str,
    plan_validator: PlanValidator = validate_plan,
    receipt_validator: ReceiptValidator = validate_run_receipt,
) -> pathlib.Path:
    output_root = output_root.resolve()
    plan = dict(plan_validator(plan_path, output_root=output_root))
    expected_result = pathlib.Path(plan["outputs"]["result"])
    if result_path.is_symlink() or not result_path.is_file() or result_path.resolve() != expected_result:
        raise DevelopmentError("development result path changed")
    result = qualification.load_sealed(result_path, RESULT_SCHEMA)
    if set(result) != RESULT_FIELDS:
        raise DevelopmentError("development result field roster changed")
    policy = result.get("policy")
    if (
        result.get("namespace") != NAMESPACE
        or result.get("campaign_id") != CAMPAIGN_ID
        or result.get("status") != "development-complete-awaiting-finalist-seal"
        or not isinstance(result.get("request_count"), int)
        or not 18 <= result["request_count"] <= 20
        or result.get("development_plan") != _sealed_record(plan_path, PLAN_SCHEMA)
        or policy != {
            "fresh_protected_tests_opened_diagnostically": True,
            "development_selected": True,
            "final_bank_generation_authorized": False,
            "rank4_gate_authorized": False,
            "upload_authorized": False,
        }
    ):
        raise DevelopmentError("development result uses another plan")
    references = result.get("run_receipts")
    if not isinstance(references, list) or len(references) != result.get("request_count"):
        raise DevelopmentError("development receipt roster changed")
    validated = [receipt_validator(reference, plan) for reference in references]
    if len({row["receipt"]["request_sha256"] for row in validated}) != len(validated):
        raise DevelopmentError("development requests are repeated")
    _validate_rows_against_receipts(result.get("rows", {}), validated)
    selected = _result_selection(result.get("rows", {}))
    if result.get("selected") != {
        "tuple_candidate_id": selected["tuple"]["candidate_id"],
        "tuple": selected["tuple"]["tuple"],
        "profile": selected["profile"]["profile"],
        "profile_work": selected["profile"]["work"],
        "actual_clock_candidate_id": selected["actual_clock"]["candidate_id"],
    }:
        raise DevelopmentError("development result selection changed")
    binaries = result.get("binaries")
    if not isinstance(binaries, Mapping) or set(binaries) != {CANDIDATE_ID, CONTROL_ID}:
        raise DevelopmentError("development binary roster changed")
    for name, record in binaries.items():
        _verify_record(record, f"{name} binary")
        observed = {
            row["request"]["candidate"]["binary_sha256"]
            for row in validated
            if row["request"]["candidate"]["candidate_id"] == name
        }
        if observed != {record["sha256"]}:
            raise DevelopmentError("finalist binary differs from run receipts")
    created_at = _utc(created_at_utc, "finalist timestamp")
    finalist_path, _finalist = _write_content_addressed(
        pathlib.Path(plan["outputs"]["finalists"]),
        _finalist_body(
            plan_path=plan_path, plan=plan, result_path=result_path,
            result=result, selected=selected, created_at_utc=created_at,
        ),
        ".finalist.json",
    )
    reference_path = pathlib.Path(plan["outputs"]["finalist_reference"])
    expected_reference = qualification.seal({
        "schema": FINALIST_REFERENCE_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID,
        "development_plan": _sealed_record(plan_path, PLAN_SCHEMA),
        "development_result": _sealed_record(result_path, RESULT_SCHEMA),
        "finalist": _sealed_record(finalist_path, FINALIST_SCHEMA),
        "complete": True,
        "final_bank_generation_authorized": False,
        "rank4_gate_authorized": False,
        "upload_authorized": False,
    })
    if reference_path.exists():
        if qualification.load_sealed(
            reference_path, FINALIST_REFERENCE_SCHEMA
        ) != expected_reference:
            raise DevelopmentError("existing finalist reference changed")
    else:
        qualification.write_sealed(reference_path, {
            key: value for key, value in expected_reference.items()
            if key != "body_sha256"
        })
    return finalist_path


def validate_finalist(
    reference_path: pathlib.Path, *, plan_path: pathlib.Path,
    output_root: pathlib.Path, plan_validator: PlanValidator = validate_plan,
    receipt_validator: ReceiptValidator = validate_run_receipt,
) -> dict[str, Any]:
    plan = dict(plan_validator(plan_path, output_root=output_root.resolve()))
    if (
        reference_path.is_symlink() or not reference_path.is_file()
        or reference_path.resolve()
        != pathlib.Path(plan["outputs"]["finalist_reference"])
    ):
        raise DevelopmentError("finalist reference path changed")
    reference = qualification.load_sealed(
        reference_path, FINALIST_REFERENCE_SCHEMA
    )
    finalist_path = _verify_sealed_record(
        reference.get("finalist"), FINALIST_SCHEMA, "finalist"
    )
    result_path = _verify_sealed_record(
        reference.get("development_result"), RESULT_SCHEMA, "development result"
    )
    finalist = qualification.load_sealed(finalist_path, FINALIST_SCHEMA)
    recreated = finalize_result(
        plan_path=plan_path, result_path=result_path,
        output_root=output_root, created_at_utc=finalist["created_at_utc"],
        plan_validator=plan_validator, receipt_validator=receipt_validator,
    )
    if recreated != finalist_path:
        raise DevelopmentError("finalist is not the deterministic development choice")
    if (
        finalist.get("fresh_protected_tests_opened") is not True
        or finalist.get("fresh_diagnostic_classification")
        != "diagnostic-only-no-pass-fail-verdict"
        or finalist.get("development_selected") is not True
        or finalist.get("final_bank_generation_authorized") is not False
        or finalist.get("rank4_gate_authorized") is not False
        or finalist.get("upload_authorized") is not False
    ):
        raise DevelopmentError("finalist authority boundary changed")
    return {"reference": reference, "finalist": finalist, "path": finalist_path}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--output-root", type=pathlib.Path, required=True)
    prepare.add_argument("--adapter-handoff", type=pathlib.Path, required=True)
    prepare.add_argument("--exclusion-plan", type=pathlib.Path, required=True)
    prepare.add_argument("--exclusion-receipt", type=pathlib.Path, required=True)
    prepare.add_argument("--rank4-control-selection", type=pathlib.Path, required=True)
    prepare.add_argument("--created-at-utc", default=utc_now())
    verify = commands.add_parser("verify")
    verify.add_argument("--output-root", type=pathlib.Path, required=True)
    verify.add_argument("--plan", type=pathlib.Path, required=True)
    verify.add_argument("--finalist-reference", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            path = prepare_plan(
                args.output_root,
                adapter_handoff_path=args.adapter_handoff,
                exclusion_plan_path=args.exclusion_plan,
                exclusion_receipt_path=args.exclusion_receipt,
                rank4_control_selection_path=args.rank4_control_selection,
                created_at_utc=args.created_at_utc,
            )
            output = {"plan": str(path.resolve()), "sha256": qualification.sha256_file(path)}
        else:
            value = validate_finalist(
                args.finalist_reference,
                plan_path=args.plan,
                output_root=args.output_root,
            )
            output = {
                "finalist": str(value["path"]),
                "sha256": qualification.sha256_file(value["path"]),
                "status": value["finalist"]["status"],
            }
        print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (DevelopmentError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"discrete-v3 development failure: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
