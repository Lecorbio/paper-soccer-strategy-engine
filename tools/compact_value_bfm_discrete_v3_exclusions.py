#!/usr/bin/env python3
"""Seal the fresh-v3/development symmetry-exclusion boundary.

``prepare`` is intentionally safe before the protected holdout is materialized:
it binds the maintained tool, immutable v3 selection, and the complete existing
six-bank development roster, including a deterministic full-roster fallback
policy.  ``audit`` is the only command that reads protected position prefixes.
It writes a content-addressed protected payload containing exactly 64,000
position/canonical-fingerprint rows and a public receipt containing only
artifact references, counts, intersection counts, and the resulting verdict.

This tool never reads evaluator labels or metrics and never grants final-bank,
Rank-4, or upload authority.  If even one fresh canonical state overlaps the
existing development roster, development remains forbidden until a separately
reviewed tool regenerates and binds all six banks; partial replacement is not
accepted here.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
REPOSITORY = HERE.parent
TEST_PATH = (
    REPOSITORY / "tests/codingame/test_compact_value_bfm_discrete_v3_exclusions.py"
)


def _load(path: pathlib.Path, name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load v3 exclusion dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


qualification = _load(
    HERE / "compact_value_bfm_qualification.py", "compact_v3_exclusion_qualification"
)
opening_tools = _load(
    HERE / "compact_value_bfm_openings.py", "compact_v3_exclusion_openings"
)
v3 = _load(
    HERE / "compact_value_bfm_discrete_v3.py", "compact_v3_exclusion_campaign"
)
holdout = _load(
    HERE / "compact_value_bfm_discrete_v3_holdout.py",
    "compact_v3_exclusion_holdout",
).fresh
selfsearch = holdout.selfsearch


ExclusionError = qualification.QualificationError
NAMESPACE = v3.NAMESPACE
CAMPAIGN_ID = v3.SUCCESSOR_CAMPAIGN_ID
HOLDOUT_CAMPAIGN_ID = f"{CAMPAIGN_ID}-holdout"

V1_PLAN_SCHEMA = "papersoccer.compact-value-bfm.discrete-v3-symmetry-exclusion-plan.v1"
V2_PLAN_SCHEMA = "papersoccer.compact-value-bfm.discrete-v3-symmetry-exclusion-plan.v2"
PLAN_SCHEMA = "papersoccer.compact-value-bfm.discrete-v3-symmetry-exclusion-plan.v3"
RETIREMENT_SCHEMA = (
    "papersoccer.compact-value-bfm.discrete-v3-symmetry-exclusion-v1-retirement.v1"
)
V2_RETIREMENT_SCHEMA = (
    "papersoccer.compact-value-bfm.discrete-v3-symmetry-exclusion-v2-retirement.v1"
)
FINGERPRINT_SCHEMA = (
    "papersoccer.compact-value-bfm.discrete-v3-protected-canonical-fingerprints.v1"
)
RECEIPT_SCHEMA = (
    "papersoccer.compact-value-bfm.discrete-v3-fresh-position-exclusion-audit.v1"
)

POSITION_COUNT = 64_000
POSITIONS_PER_GAME = 20
STAGE_COUNTS = dict(opening_tools.DEVELOPMENT_COUNTS)
STAGE_ORDER = tuple(opening_tools.DEVELOPMENT_ORDER)
DEVELOPMENT_OPENINGS = sum(STAGE_COUNTS.values())
SYMMETRY_NAMES = ("exact", "rotate", "reflect", "rotate_reflect")
CANONICALIZATION = (
    "minimum-sha256-over-exact+rotate+reflect+rotate_reflect"
)
SOURCE_CAMPAIGN_ID = "compact-value-bfm-20260831-v1"
EXPECTED_DEVELOPMENT_BANK_SHA256 = {
    "model_screen": "e678ec2efe90d39a3f28fd1a6a7235f8ee14a55634936aca04d25705847e1900",
    "tuple_screen": "7d365dcd2cb195c7147c1e810e5eddc51efb85db61495cd00bf4f00f44a76463",
    "tuple_confirmation": "be2f65786a7f913204ea75947ad75243c43781ee8357a9edb9f65fafb18c96ba",
    "profile_screen": "10fc4dbd5776ea3f27bf8e23deb09402a85ff886b7bf579bab6d11f35ec9e930",
    "profile_confirmation": "1935edf0afa818fac085a7000d8d682329e46f6b7facc60d5ebd82cb7d42baf2",
    "actual_clock": "6726e151667d1a93fe5737994d9bc30ffce8ebd44410a68c5bcf4a6abcc98d32",
}
FALLBACK_DOMAIN = "fresh-symmetry-exclusion-full-development-roster-v1"
FALLBACK_MASTER_SEED_HEX = hashlib.sha256(
    f"{CAMPAIGN_ID}\0{FALLBACK_DOMAIN}".encode("ascii")
).hexdigest()
SHA256_RE = re.compile(r"[0-9a-f]{64}")
V1_PLAN_SHA256 = "6f9d0f4017dc5496e6b0e6f9fdabc817ed82220c8cf665362ccaa871054b51ed"
V1_PLAN_BODY_SHA256 = "b30d1a597eb6962ceb15804fe570134388b3c4922e93a5f923225cfc4cf92dab"
V1_TOOL_SHA256 = "c6cadf3d2400f7361a5062a421587c1672a7ea411f7049d3389e48dbc461741b"
V1_TEST_SHA256 = "86a9a5ab135741ec7247e1ca638da3f34622fd6cfd126575e841465279efa53d"
V1_TOOL_BYTES = 55_701
V1_TEST_BYTES = 29_639
V1_FAILURE_REASON = (
    "fresh-position-prefix-replay-used-opening-only-depth-and-slash-contract"
)
V1_RETIREMENT_SHA256 = (
    "d803eb8615951953c537887d4dfafdb42a38b39ac4d62395616e0f0be90eb637"
)
V1_RETIREMENT_BODY_SHA256 = (
    "b5e5d0e0d47bca675214a0544f4870fdfbff45ca417da634ad27699ccc517150"
)
V1_RETIREMENT_BYTES = 1_979
V1_RETIREMENT_TOOL_SHA256 = (
    "976eae9357de3a785a7f4fd6b10bce6de4dd9bb1d8ac769cbde36ae44553d2d8"
)
V1_RETIREMENT_TEST_SHA256 = (
    "681141fb2714053c69b32c30a344e5a539e34ec9f1f17bc93c29960d543f0af4"
)
V2_PLAN_SHA256 = "f08f1b2226e8e766ab3259c685c03acefa4e2d845c4bc4e0e2a4f05c227f7ad5"
V2_PLAN_BODY_SHA256 = (
    "b873a37ae687bae1d273b629bc7f38b5483381a27fe2b429d9b23300ebe836b8"
)
V2_PLAN_BYTES = 6_637
V2_TOOL_SHA256 = V1_RETIREMENT_TOOL_SHA256
V2_TOOL_BYTES = 65_038
V2_TEST_SHA256 = V1_RETIREMENT_TEST_SHA256
V2_TEST_BYTES = 37_659
V2_OPENING_TOOL_SHA256 = (
    "a768cb14222a642a89423766a68418621084973026ada217510cef0ab6b44e4c"
)
V2_OPENING_TOOL_BYTES = 29_375
V2_FAILURE_REASON = "empty-initial-prefix-forbidden"


CampaignValidator = Callable[
    [pathlib.Path, pathlib.Path], Mapping[str, Any]
]
BankIdentityValidator = Callable[
    [Mapping[str, pathlib.Path], Mapping[str, Mapping[str, Any]]], None
]
RetirementValidator = Callable[[pathlib.Path, pathlib.Path], Mapping[str, Any]]
V1PlanValidator = Callable[[pathlib.Path], Mapping[str, Any]]
V2PlanValidator = Callable[[pathlib.Path], Mapping[str, Any]]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _utc(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ExclusionError(f"{label} must be an ISO-8601 UTC timestamp")
    try:
        parsed = dt.datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ExclusionError(f"{label} is invalid") from error
    if parsed.tzinfo != dt.timezone.utc:
        raise ExclusionError(f"{label} must be UTC")
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ExclusionError(f"{label} must be a lowercase SHA-256")
    return value


def _reject_symlink_or_nondirectory_components(
    path: pathlib.Path, *, final_may_be_file: bool,
) -> pathlib.Path:
    absolute = path.absolute()
    parts = absolute.parts
    current = pathlib.Path(parts[0])
    for index, part in enumerate(parts[1:], start=1):
        current = current / part
        final = index == len(parts) - 1
        if not os.path.lexists(current):
            continue
        if current.is_symlink():
            raise ExclusionError(f"path component is a symlink: {current}")
        if (not final or not final_may_be_file) and not current.is_dir():
            raise ExclusionError(f"path component is not a directory: {current}")
    return absolute


def _safe_directory(path: pathlib.Path, *, create: bool) -> pathlib.Path:
    absolute = _reject_symlink_or_nondirectory_components(
        path, final_may_be_file=False
    )
    if create:
        parts = absolute.parts
        current = pathlib.Path(parts[0])
        for part in parts[1:]:
            current = current / part
            if not os.path.lexists(current):
                os.mkdir(current, 0o700)
            if current.is_symlink() or not current.is_dir():
                raise ExclusionError(
                    f"output directory component is unsafe: {current}"
                )
    elif not absolute.is_dir():
        raise ExclusionError(f"required output root is not a directory: {absolute}")
    _reject_symlink_or_nondirectory_components(
        absolute, final_may_be_file=False
    )
    return absolute


def _safe_output_root(path: pathlib.Path) -> pathlib.Path:
    lexical = path.absolute()
    resolved = lexical.resolve()
    if lexical != resolved or lexical.is_symlink() or not lexical.is_dir():
        raise ExclusionError(f"output root is absent, redirected, or not a directory: {lexical}")
    return resolved


def _safe_output_file(path: pathlib.Path) -> pathlib.Path:
    absolute = _reject_symlink_or_nondirectory_components(
        path, final_may_be_file=True
    )
    _safe_directory(absolute.parent, create=True)
    if os.path.lexists(absolute) and (
        absolute.is_symlink() or not absolute.is_file()
    ):
        raise ExclusionError(f"output file target is unsafe: {absolute}")
    return absolute


def _regular_record(path: pathlib.Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ExclusionError(f"required artifact is not a regular file: {path}")
    path = path.resolve()
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": qualification.sha256_file(path),
    }


def _verify_record(value: Any, label: str) -> pathlib.Path:
    if not isinstance(value, Mapping) or set(value) != {"path", "bytes", "sha256"}:
        raise ExclusionError(f"{label} record is malformed")
    path = pathlib.Path(str(value.get("path", "")))
    if _regular_record(path) != dict(value):
        raise ExclusionError(f"{label} artifact changed")
    return path.resolve()


def _reference(path: pathlib.Path, schema: str | None = None) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ExclusionError(f"referenced artifact is not a regular file: {path}")
    return qualification.artifact_reference(path, schema)


def _verify_reference(value: Any, schema: str, label: str) -> pathlib.Path:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ExclusionError(f"{label} reference is malformed")
    path = pathlib.Path(str(value.get("path", "")))
    if path.is_symlink() or not path.is_file():
        raise ExclusionError(f"{label} reference is absent or redirected")
    if dict(value) != _reference(path, schema):
        raise ExclusionError(f"{label} reference changed")
    return path.resolve()


def _load_canonical_json(path: pathlib.Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ExclusionError(f"{label} is not a regular file")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExclusionError(f"{label} is not JSON") from error
    if (
        not isinstance(value, dict)
        or selfsearch.canonical_json_bytes(value, pretty=True) != raw
    ):
        raise ExclusionError(f"{label} is not exact pretty canonical JSON")
    return value


def _write_content_addressed(
    directory: pathlib.Path, body: Mapping[str, Any], suffix: str,
) -> pathlib.Path:
    directory = _safe_directory(directory, create=True)
    artifact = qualification.seal(body)
    raw = qualification.canonical_json_bytes(artifact)
    path = _safe_output_file(
        directory / f"{qualification.sha256_bytes(raw)}{suffix}"
    )
    qualification.atomic_write_once(path, raw)
    if qualification.sha256_file(path) != path.name.removesuffix(suffix):
        raise ExclusionError("content-addressed artifact did not publish exactly")
    return path


def _default_campaign_validator(
    plan_path: pathlib.Path, output_root: pathlib.Path,
) -> Mapping[str, Any]:
    output_root = output_root.resolve()
    plan = v3.load_plan(plan_path, output_root=output_root)
    selection_path = v3._selection_reference(
        output_root / "selection-reference.json", plan=plan, output_root=output_root
    )
    if selection_path is None:
        raise ExclusionError("v3 immutable selection is absent")
    selection = v3._validate_selection_closure(
        selection_path, plan=plan, output_root=output_root
    )
    return {
        "plan": plan,
        "selection_path": selection_path.resolve(),
        "selection": selection,
    }


def _campaign_context(
    plan_path: pathlib.Path, output_root: pathlib.Path,
    campaign_validator: CampaignValidator,
) -> dict[str, Any]:
    value = campaign_validator(plan_path, output_root)
    if not isinstance(value, Mapping):
        raise ExclusionError("v3 campaign validator returned no context")
    plan = value.get("plan")
    selection = value.get("selection")
    selection_path = pathlib.Path(str(value.get("selection_path", "")))
    if (
        not isinstance(plan, Mapping)
        or not isinstance(selection, Mapping)
        or selection_path.is_symlink()
        or not selection_path.is_file()
        or plan.get("schema") != v3.PLAN_SCHEMA
        or plan.get("namespace") != NAMESPACE
        or plan.get("campaign_id") != CAMPAIGN_ID
        or selection.get("schema") != v3.SELECTION_SCHEMA
        or selection.get("namespace") != NAMESPACE
        or selection.get("campaign_id") != CAMPAIGN_ID
        or selection.get("selection_immutable") is not True
        or selection.get("offline_gate", {}).get("passed") is not True
    ):
        raise ExclusionError("v3 plan/selection context is not immutable and qualified")
    fresh = plan.get("fresh_protected_holdout")
    if (
        not isinstance(fresh, Mapping)
        or fresh.get("campaign_id") != HOLDOUT_CAMPAIGN_ID
        or fresh.get("positions") != POSITION_COUNT
        or fresh.get("positions_per_game") != POSITIONS_PER_GAME
        or fresh.get("fresh_root_split") != "test"
        or fresh.get("diagnostic_only") is not True
        or fresh.get("selection_may_change_after_results") is not False
    ):
        raise ExclusionError("v3 fresh-holdout contract changed")
    runtime = selection.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ExclusionError("v3 selection has no runtime record")
    _verify_record(runtime, "v3 selected runtime")
    return {
        "plan": dict(plan),
        "selection": dict(selection),
        "selection_path": selection_path.resolve(),
        "plan_reference": _reference(plan_path, v3.PLAN_SCHEMA),
        "selection_reference": _reference(selection_path, v3.SELECTION_SCHEMA),
    }


def _bank_roster(
    paths: Mapping[str, pathlib.Path],
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    if set(paths) != set(STAGE_ORDER):
        raise ExclusionError("exact six-stage development bank roster is required")
    records: dict[str, dict[str, Any]] = {}
    all_canonical: set[str] = set()
    seen_paths: set[pathlib.Path] = set()
    seen_hashes: set[str] = set()
    for stage in STAGE_ORDER:
        lexical = paths[stage].absolute()
        if (
            lexical != lexical.resolve()
            or lexical.is_symlink()
            or not lexical.is_file()
        ):
            raise ExclusionError("development bank is absent or redirected")
        path = lexical.resolve()
        if path in seen_paths:
            raise ExclusionError("development bank paths are repeated or redirected")
        bank = opening_tools.validate_bank(path)
        if (
            bank.get("stage") != stage
            or bank.get("classification") != "unprotected-development"
            or bank.get("opening_count") != STAGE_COUNTS[stage]
        ):
            raise ExclusionError(f"{stage} development bank contract changed")
        digest = qualification.sha256_file(path)
        if digest in seen_hashes:
            raise ExclusionError("development bank byte identities are repeated")
        canonical = {
            str(opening["fingerprints"]["canonical"])
            for opening in bank["openings"]
        }
        if len(canonical) != STAGE_COUNTS[stage] or canonical & all_canonical:
            raise ExclusionError("development bank roster overlaps by symmetry")
        all_canonical.update(canonical)
        seen_paths.add(path)
        seen_hashes.add(digest)
        records[stage] = _regular_record(path)
    if len(all_canonical) != DEVELOPMENT_OPENINGS:
        raise ExclusionError("development bank canonical roster is incomplete")
    return records, all_canonical


def _default_bank_identity_validator(
    paths: Mapping[str, pathlib.Path], records: Mapping[str, Mapping[str, Any]],
) -> None:
    base = (
        REPOSITORY / "results/compact_value_bfm" / SOURCE_CAMPAIGN_ID
        / "openings/development-v1"
    ).resolve()
    for stage in STAGE_ORDER:
        digest = EXPECTED_DEVELOPMENT_BANK_SHA256[stage]
        expected = base / stage / f"{digest}.opening-bank.json"
        if (
            paths[stage].resolve() != expected
            or records[stage].get("sha256") != digest
        ):
            raise ExclusionError(
                f"{stage} is not the exact frozen v1 development bank"
            )


def _validate_v1_plan(output_root: pathlib.Path) -> dict[str, Any]:
    output_root = _safe_output_root(output_root)
    artifact_root = _safe_directory(
        output_root / "fresh-symmetry-exclusion", create=False
    )
    plan_path = artifact_root / "plan.json"
    protected = _safe_directory(artifact_root / "protected", create=False)
    if (
        plan_path.is_symlink() or not plan_path.is_file()
        or qualification.sha256_file(plan_path) != V1_PLAN_SHA256
        or protected.is_symlink() or not protected.is_dir()
    ):
        raise ExclusionError("retired v1 exclusion plan/output state changed")
    plan = qualification.load_sealed(plan_path, V1_PLAN_SCHEMA)
    tools = plan.get("tools")
    if (
        plan.get("body_sha256") != V1_PLAN_BODY_SHA256
        or plan.get("status")
        != "fresh-symmetry-exclusion-policy-precommitted-before-audit-read"
        or not isinstance(tools, Mapping)
        or tools.get("exclusion_tool", {}).get("sha256") != V1_TOOL_SHA256
        or tools.get("exclusion_test", {}).get("sha256") != V1_TEST_SHA256
        or pathlib.Path(str(plan.get("outputs", {}).get(
            "protected_directory", ""
        ))) != protected
        or pathlib.Path(str(plan.get("outputs", {}).get("public_receipt", "")))
        != artifact_root / "receipt.json"
    ):
        raise ExclusionError("retired v1 exclusion plan contract changed")
    allowed = {
        "plan.json", "protected", "retirement-v1.json", "plan-v2.json",
        "retirement-v2.json", "plan-v3.json", "receipt.json",
    }
    if any(child.name not in allowed for child in artifact_root.iterdir()):
        raise ExclusionError("foreign output exists beside v1/v2 exclusion state")
    return {
        "artifact_root": artifact_root,
        "plan_path": plan_path,
        "protected": protected,
        "plan": plan,
    }


def _require_pristine_v1_retirement_state(v1: Mapping[str, Any]) -> None:
    artifact_root = _safe_directory(
        pathlib.Path(v1["artifact_root"]), create=False
    )
    protected = _safe_directory(pathlib.Path(v1["protected"]), create=False)
    if any(protected.iterdir()):
        raise ExclusionError("v1 protected directory was not empty before retirement")
    if {child.name for child in artifact_root.iterdir()} != {
        "plan.json", "protected",
    }:
        raise ExclusionError("v1 exclusion output exists before retirement")


def _retirement_body(v1: Mapping[str, Any], retired_at_utc: str) -> dict[str, Any]:
    plan = v1["plan"]
    return {
        "schema": RETIREMENT_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID,
        "status": "symmetry-exclusion-v1-retired-before-audit-output",
        "retired_at_utc": _utc(retired_at_utc, "v1 exclusion retirement time"),
        "v1_plan": _reference(v1["plan_path"], V1_PLAN_SCHEMA),
        "v1_plan_body_sha256": V1_PLAN_BODY_SHA256,
        "v1_tool": dict(plan["tools"]["exclusion_tool"]),
        "v1_test": dict(plan["tools"]["exclusion_test"]),
        "failure_reason": V1_FAILURE_REASON,
        "receipt_created": False,
        "private_payload_created": False,
        "protected_directory_empty": True,
        "metrics_opened": False,
        "development_games_authorized": False,
        "rank4_gate_authorized": False,
        "upload_authorized": False,
        "retirement_tool_closure": {
            "tool": _regular_record(pathlib.Path(__file__).resolve()),
            "test": _regular_record(TEST_PATH),
        },
    }


def retire_v1(
    *, output_root: pathlib.Path, retired_at_utc: str,
    v1_validator: V1PlanValidator = _validate_v1_plan,
) -> pathlib.Path:
    v1 = dict(v1_validator(output_root))
    path = _safe_output_file(v1["artifact_root"] / "retirement-v1.json")
    if v1_validator is _validate_v1_plan:
        if not path.exists():
            raise ExclusionError(
                "frozen v1 retirement is absent and its retired implementation "
                "cannot be replayed"
            )
        validate_retirement(path, output_root=output_root)
        return path
    body = _retirement_body(v1, retired_at_utc)
    if path.exists():
        if qualification.load_sealed(path, RETIREMENT_SCHEMA) != qualification.seal(body):
            raise ExclusionError("existing v1 exclusion retirement changed")
    else:
        _require_pristine_v1_retirement_state(v1)
        qualification.write_sealed(path, body)
    validate_retirement(
        path, output_root=output_root, v1_validator=v1_validator
    )
    return path


def validate_retirement(
    path: pathlib.Path, *, output_root: pathlib.Path,
    v1_validator: V1PlanValidator = _validate_v1_plan,
) -> dict[str, Any]:
    v1 = dict(v1_validator(output_root))
    expected_path = v1["artifact_root"] / "retirement-v1.json"
    if path.absolute() != expected_path or path.is_symlink() or not path.is_file():
        raise ExclusionError("v1 exclusion retirement path changed")
    if v1_validator is _validate_v1_plan:
        record = _regular_record(path)
        value = qualification.load_sealed(path, RETIREMENT_SCHEMA)
        expected_fields = {
            "schema", "namespace", "campaign_id", "status", "retired_at_utc",
            "v1_plan", "v1_plan_body_sha256", "v1_tool", "v1_test",
            "failure_reason", "receipt_created", "private_payload_created",
            "protected_directory_empty", "metrics_opened",
            "development_games_authorized", "rank4_gate_authorized",
            "upload_authorized", "retirement_tool_closure", "body_sha256",
        }
        historical_tool = {
            "path": str(pathlib.Path(__file__).resolve()),
            "bytes": V1_TOOL_BYTES,
            "sha256": V1_TOOL_SHA256,
        }
        historical_test = {
            "path": str(TEST_PATH.resolve()),
            "bytes": V1_TEST_BYTES,
            "sha256": V1_TEST_SHA256,
        }
        retirement_tool = {
            "path": str(pathlib.Path(__file__).resolve()),
            "bytes": V2_TOOL_BYTES,
            "sha256": V1_RETIREMENT_TOOL_SHA256,
        }
        retirement_test = {
            "path": str(TEST_PATH.resolve()),
            "bytes": V2_TEST_BYTES,
            "sha256": V1_RETIREMENT_TEST_SHA256,
        }
        if (
            set(value) != expected_fields
            or record["bytes"] != V1_RETIREMENT_BYTES
            or record["sha256"] != V1_RETIREMENT_SHA256
            or value.get("body_sha256") != V1_RETIREMENT_BODY_SHA256
            or value.get("status")
            != "symmetry-exclusion-v1-retired-before-audit-output"
            or value.get("v1_plan") != _reference(v1["plan_path"], V1_PLAN_SCHEMA)
            or value.get("v1_plan_body_sha256") != V1_PLAN_BODY_SHA256
            or value.get("v1_tool") != historical_tool
            or value.get("v1_test") != historical_test
            or value.get("failure_reason") != V1_FAILURE_REASON
            or value.get("receipt_created") is not False
            or value.get("private_payload_created") is not False
            or value.get("protected_directory_empty") is not True
            or value.get("metrics_opened") is not False
            or value.get("development_games_authorized") is not False
            or value.get("rank4_gate_authorized") is not False
            or value.get("upload_authorized") is not False
            or value.get("retirement_tool_closure") != {
                "tool": retirement_tool, "test": retirement_test,
            }
        ):
            raise ExclusionError("frozen v1 exclusion retirement changed")
        _utc(value.get("retired_at_utc"), "v1 exclusion retirement time")
        return value
    value = qualification.load_sealed(path, RETIREMENT_SCHEMA)
    expected = qualification.seal(_retirement_body(
        v1, str(value.get("retired_at_utc"))
    ))
    if value != expected:
        raise ExclusionError("v1 exclusion retirement content changed")
    return value


def _validate_v2_plan(output_root: pathlib.Path) -> dict[str, Any]:
    output_root = _safe_output_root(output_root)
    artifact_root = _safe_directory(
        output_root / "fresh-symmetry-exclusion", create=False
    )
    protected = _safe_directory(artifact_root / "protected", create=False)
    plan_path = artifact_root / "plan-v2.json"
    retirement_v1_path = artifact_root / "retirement-v1.json"
    if (
        plan_path.is_symlink() or not plan_path.is_file()
        or plan_path.stat().st_size != V2_PLAN_BYTES
        or qualification.sha256_file(plan_path) != V2_PLAN_SHA256
    ):
        raise ExclusionError("retired v2 exclusion plan changed")
    retirement_v1 = validate_retirement(
        retirement_v1_path, output_root=output_root
    )
    plan = qualification.load_sealed(plan_path, V2_PLAN_SCHEMA)
    tools = plan.get("tools")
    expected_tool = {
        "path": str(pathlib.Path(__file__).resolve()),
        "bytes": V2_TOOL_BYTES,
        "sha256": V2_TOOL_SHA256,
    }
    expected_test = {
        "path": str(TEST_PATH.resolve()),
        "bytes": V2_TEST_BYTES,
        "sha256": V2_TEST_SHA256,
    }
    expected_opening_tool = {
        "path": str(pathlib.Path(opening_tools.__file__).resolve()),
        "bytes": V2_OPENING_TOOL_BYTES,
        "sha256": V2_OPENING_TOOL_SHA256,
    }
    outputs = plan.get("outputs")
    if (
        plan.get("body_sha256") != V2_PLAN_BODY_SHA256
        or plan.get("status")
        != "fresh-symmetry-exclusion-v2-policy-precommitted-before-audit-read"
        or plan.get("v1_retirement")
        != _reference(retirement_v1_path, RETIREMENT_SCHEMA)
        or not isinstance(tools, Mapping)
        or tools.get("exclusion_tool") != expected_tool
        or tools.get("exclusion_test") != expected_test
        or tools.get("opening_tool") != expected_opening_tool
        or not isinstance(outputs, Mapping)
        or pathlib.Path(str(outputs.get("artifact_root", ""))) != artifact_root
        or pathlib.Path(str(outputs.get("protected_directory", ""))) != protected
        or pathlib.Path(str(outputs.get("public_receipt", "")))
        != artifact_root / "receipt.json"
    ):
        raise ExclusionError("retired v2 exclusion plan contract changed")
    allowed = {
        "plan.json", "protected", "retirement-v1.json", "plan-v2.json",
        "retirement-v2.json", "plan-v3.json", "receipt.json",
    }
    if any(child.name not in allowed for child in artifact_root.iterdir()):
        raise ExclusionError("foreign output exists beside v1/v2/v3 exclusion state")
    return {
        "artifact_root": artifact_root,
        "protected": protected,
        "plan_path": plan_path,
        "plan": plan,
        "v1_retirement_path": retirement_v1_path,
        "v1_retirement": retirement_v1,
    }


def _require_pristine_v2_retirement_state(v2: Mapping[str, Any]) -> None:
    artifact_root = _safe_directory(
        pathlib.Path(v2["artifact_root"]), create=False
    )
    protected = _safe_directory(pathlib.Path(v2["protected"]), create=False)
    if any(protected.iterdir()):
        raise ExclusionError("v2 protected directory was not empty before retirement")
    if {child.name for child in artifact_root.iterdir()} != {
        "plan.json", "protected", "retirement-v1.json", "plan-v2.json",
    }:
        raise ExclusionError("v2 exclusion output exists before retirement")


def _v2_retirement_body(
    v2: Mapping[str, Any], retired_at_utc: str,
) -> dict[str, Any]:
    plan = v2["plan"]
    return {
        "schema": V2_RETIREMENT_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID,
        "status": "symmetry-exclusion-v2-retired-before-audit-output",
        "retired_at_utc": _utc(retired_at_utc, "v2 exclusion retirement time"),
        "v2_plan": _reference(v2["plan_path"], V2_PLAN_SCHEMA),
        "v2_plan_body_sha256": V2_PLAN_BODY_SHA256,
        "v1_retirement": _reference(
            v2["v1_retirement_path"], RETIREMENT_SCHEMA
        ),
        "v2_tool": dict(plan["tools"]["exclusion_tool"]),
        "v2_test": dict(plan["tools"]["exclusion_test"]),
        "v2_opening_tool": dict(plan["tools"]["opening_tool"]),
        "failure_reason": V2_FAILURE_REASON,
        "receipt_created": False,
        "private_payload_created": False,
        "protected_directory_empty": True,
        "metrics_opened": False,
        "development_games_authorized": False,
        "final_bank_generation_authorized": False,
        "rank4_gate_authorized": False,
        "upload_authorized": False,
        "retirement_tool_closure": {
            "tool": _regular_record(pathlib.Path(__file__).resolve()),
            "test": _regular_record(TEST_PATH),
        },
    }


def retire_v2(
    *, output_root: pathlib.Path, retired_at_utc: str,
    v2_validator: V2PlanValidator = _validate_v2_plan,
) -> pathlib.Path:
    v2 = dict(v2_validator(output_root))
    path = _safe_output_file(v2["artifact_root"] / "retirement-v2.json")
    body = _v2_retirement_body(v2, retired_at_utc)
    if path.exists():
        if (
            qualification.load_sealed(path, V2_RETIREMENT_SCHEMA)
            != qualification.seal(body)
        ):
            raise ExclusionError("existing v2 exclusion retirement changed")
    else:
        _require_pristine_v2_retirement_state(v2)
        qualification.write_sealed(path, body)
    validate_v2_retirement(
        path, output_root=output_root, v2_validator=v2_validator
    )
    return path


def validate_v2_retirement(
    path: pathlib.Path, *, output_root: pathlib.Path,
    v2_validator: V2PlanValidator = _validate_v2_plan,
) -> dict[str, Any]:
    v2 = dict(v2_validator(output_root))
    expected_path = v2["artifact_root"] / "retirement-v2.json"
    if path.absolute() != expected_path or path.is_symlink() or not path.is_file():
        raise ExclusionError("v2 exclusion retirement path changed")
    value = qualification.load_sealed(path, V2_RETIREMENT_SCHEMA)
    expected_fields = {
        "schema", "namespace", "campaign_id", "status", "retired_at_utc",
        "v2_plan", "v2_plan_body_sha256", "v1_retirement", "v2_tool",
        "v2_test", "v2_opening_tool", "failure_reason", "receipt_created",
        "private_payload_created", "protected_directory_empty", "metrics_opened",
        "development_games_authorized", "final_bank_generation_authorized",
        "rank4_gate_authorized", "upload_authorized", "retirement_tool_closure",
        "body_sha256",
    }
    plan = v2["plan"]
    closure = value.get("retirement_tool_closure")
    if (
        set(value) != expected_fields
        or value.get("namespace") != NAMESPACE
        or value.get("campaign_id") != CAMPAIGN_ID
        or value.get("status")
        != "symmetry-exclusion-v2-retired-before-audit-output"
        or value.get("v2_plan") != _reference(v2["plan_path"], V2_PLAN_SCHEMA)
        or value.get("v2_plan_body_sha256") != V2_PLAN_BODY_SHA256
        or value.get("v1_retirement")
        != _reference(v2["v1_retirement_path"], RETIREMENT_SCHEMA)
        or value.get("v2_tool") != plan["tools"]["exclusion_tool"]
        or value.get("v2_test") != plan["tools"]["exclusion_test"]
        or value.get("v2_opening_tool") != plan["tools"]["opening_tool"]
        or value.get("failure_reason") != V2_FAILURE_REASON
        or value.get("receipt_created") is not False
        or value.get("private_payload_created") is not False
        or value.get("protected_directory_empty") is not True
        or value.get("metrics_opened") is not False
        or value.get("development_games_authorized") is not False
        or value.get("final_bank_generation_authorized") is not False
        or value.get("rank4_gate_authorized") is not False
        or value.get("upload_authorized") is not False
        or not isinstance(closure, Mapping)
        or set(closure) != {"tool", "test"}
    ):
        raise ExclusionError("v2 exclusion retirement content changed")
    for name, expected_path_value in (
        ("tool", pathlib.Path(__file__).resolve()),
        ("test", TEST_PATH.resolve()),
    ):
        record = closure[name]
        if (
            not isinstance(record, Mapping)
            or set(record) != {"path", "bytes", "sha256"}
            or record.get("path") != str(expected_path_value)
            or isinstance(record.get("bytes"), bool)
            or not isinstance(record.get("bytes"), int)
            or record["bytes"] <= 0
            or SHA256_RE.fullmatch(str(record.get("sha256", ""))) is None
        ):
            raise ExclusionError("v2 exclusion retirement closure is malformed")
    _utc(value.get("retired_at_utc"), "v2 exclusion retirement time")
    return value


def _require_current_v2_retirement_closure(value: Mapping[str, Any]) -> None:
    if value.get("retirement_tool_closure") != {
        "tool": _regular_record(pathlib.Path(__file__).resolve()),
        "test": _regular_record(TEST_PATH),
    }:
        raise ExclusionError("v2 retirement was not sealed by the current v3 closure")


def _plan_body(
    *, context: Mapping[str, Any], development_banks: Mapping[str, dict[str, Any]],
    output_root: pathlib.Path, v2_retirement_reference: Mapping[str, Any],
    created_at_utc: str,
) -> dict[str, Any]:
    artifact_root = output_root.resolve() / "fresh-symmetry-exclusion"
    return {
        "schema": PLAN_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID,
        "status": "fresh-symmetry-exclusion-v3-policy-precommitted-before-audit-read",
        "created_at_utc": _utc(created_at_utc, "exclusion plan creation time"),
        "v2_retirement": dict(v2_retirement_reference),
        "v3_plan": dict(context["plan_reference"]),
        "immutable_selection": dict(context["selection_reference"]),
        "materialization_path": str(
            (output_root.resolve() / "fresh-holdout/materialization-receipt.json")
        ),
        "positions_path": str(
            (output_root.resolve() / "fresh-holdout/materialized/positions.tsv")
        ),
        "positions_manifest_path": str(
            (output_root.resolve() / "fresh-holdout/materialized/positions.manifest.json")
        ),
        "development_banks": {
            stage: dict(development_banks[stage]) for stage in STAGE_ORDER
        },
        "contract": {
            "protected_position_count": POSITION_COUNT,
            "positions_per_game": POSITIONS_PER_GAME,
            "symmetries": list(SYMMETRY_NAMES),
            "canonicalization": CANONICALIZATION,
            "development_stage_counts": dict(STAGE_COUNTS),
            "development_opening_count": DEVELOPMENT_OPENINGS,
            "empty_initial_prefix_allowed": True,
            "prefix_replay": "maintained-ReplayState+apply_complete_turn",
            "protected_labels_or_metrics_read": False,
        },
        "regeneration_policy": {
            "trigger": "positive-fresh-development-canonical-intersection",
            "action": "regenerate-entire-six-bank-development-roster",
            "partial_roster_replacement_authorized": False,
            "existing_roster_reuse_requires_zero_intersection": True,
            "development_before_full-roster-resolution_authorized": False,
            "fresh_canonical_payload_must_be_excluded": True,
            "separately_reviewed_full_roster_generator_required": True,
            "regenerated_roster_receipt_required_before_development": True,
            "master_seed_domain": FALLBACK_DOMAIN,
            "master_seed_hex": FALLBACK_MASTER_SEED_HEX,
        },
        "outputs": {
            "artifact_root": str(artifact_root),
            "protected_directory": str(artifact_root / "protected"),
            "public_receipt": str(artifact_root / "receipt.json"),
        },
        "tools": {
            "exclusion_tool": _regular_record(pathlib.Path(__file__).resolve()),
            "exclusion_test": _regular_record(TEST_PATH),
            "opening_tool": _regular_record(pathlib.Path(opening_tools.__file__).resolve()),
        },
        "policy": {
            "diagnostic_metrics_are_not_an_acceptance_gate": True,
            "development_games_authorized": False,
            "final_bank_generation_authorized": False,
            "rank4_gate_authorized": False,
            "upload_authorized": False,
        },
    }


def prepare(
    *, output_root: pathlib.Path, v3_plan_path: pathlib.Path,
    v2_retirement_path: pathlib.Path,
    development_bank_paths: Mapping[str, pathlib.Path], created_at_utc: str,
    campaign_validator: CampaignValidator = _default_campaign_validator,
    bank_identity_validator: BankIdentityValidator = _default_bank_identity_validator,
    retirement_validator: RetirementValidator = validate_v2_retirement,
) -> pathlib.Path:
    output_root = _safe_output_root(output_root)
    artifact_root = _safe_directory(
        output_root / "fresh-symmetry-exclusion", create=False
    )
    protected = _safe_directory(artifact_root / "protected", create=False)
    try:
        retirement = dict(retirement_validator(
            v2_retirement_path, output_root=output_root
        ))
    except Exception as error:
        raise ExclusionError("valid v2 exclusion retirement is required") from error
    _require_current_v2_retirement_closure(retirement)
    context = _campaign_context(v3_plan_path, output_root, campaign_validator)
    records, _canonical = _bank_roster(development_bank_paths)
    bank_identity_validator(development_bank_paths, records)
    path = _safe_output_file(artifact_root / "plan-v3.json")
    body = _plan_body(
        context=context, development_banks=records, output_root=output_root,
        v2_retirement_reference=_reference(
            v2_retirement_path, V2_RETIREMENT_SCHEMA
        ),
        created_at_utc=created_at_utc,
    )
    if path.exists():
        existing = qualification.load_sealed(path, PLAN_SCHEMA)
        expected = qualification.seal(body)
        if existing != expected:
            raise ExclusionError("existing symmetry-exclusion plan changed")
    else:
        receipt = _safe_output_file(artifact_root / "receipt.json")
        if receipt.exists() or receipt.is_symlink() or (
            any(protected.iterdir())
        ):
            raise ExclusionError(
                "symmetry audit output exists without its precommitted plan"
            )
        qualification.write_sealed(path, body)
    validate_plan(
        path, output_root=output_root, campaign_validator=campaign_validator,
        bank_identity_validator=bank_identity_validator,
        retirement_validator=retirement_validator,
    )
    return path


def validate_plan(
    path: pathlib.Path, *, output_root: pathlib.Path,
    campaign_validator: CampaignValidator = _default_campaign_validator,
    bank_identity_validator: BankIdentityValidator = _default_bank_identity_validator,
    retirement_validator: RetirementValidator = validate_v2_retirement,
) -> dict[str, Any]:
    output_root = _safe_output_root(output_root)
    _safe_directory(output_root / "fresh-symmetry-exclusion", create=False)
    expected_path = output_root / "fresh-symmetry-exclusion/plan-v3.json"
    lexical_path = _reject_symlink_or_nondirectory_components(
        path, final_may_be_file=True
    )
    if lexical_path != expected_path or path.is_symlink() or not path.is_file():
        raise ExclusionError("symmetry-exclusion plan path is not canonical")
    plan = qualification.load_sealed(path, PLAN_SCHEMA)
    retirement_path = _verify_reference(
        plan.get("v2_retirement"), V2_RETIREMENT_SCHEMA,
        "v2 exclusion retirement",
    )
    try:
        retirement = dict(retirement_validator(
            retirement_path, output_root=output_root
        ))
    except Exception as error:
        raise ExclusionError("v2 exclusion retirement changed") from error
    _require_current_v2_retirement_closure(retirement)
    context = _campaign_context(
        pathlib.Path(str(plan.get("v3_plan", {}).get("path", ""))),
        output_root,
        campaign_validator,
    )
    bank_values = plan.get("development_banks")
    if not isinstance(bank_values, Mapping) or set(bank_values) != set(STAGE_ORDER):
        raise ExclusionError("symmetry-exclusion plan bank roster changed")
    bank_paths = {
        stage: _verify_record(bank_values[stage], f"planned {stage} bank")
        for stage in STAGE_ORDER
    }
    records, _canonical = _bank_roster(bank_paths)
    bank_identity_validator(bank_paths, records)
    created_at = _utc(plan.get("created_at_utc"), "exclusion plan creation time")
    expected = qualification.seal(_plan_body(
        context=context, development_banks=records, output_root=output_root,
        v2_retirement_reference=_reference(
            retirement_path, V2_RETIREMENT_SCHEMA
        ),
        created_at_utc=created_at,
    ))
    if plan != expected:
        raise ExclusionError("symmetry-exclusion plan contract changed")
    return plan


def _verify_snapshot_against_record(
    snapshot: Any, record: Mapping[str, Any], label: str,
) -> None:
    if not isinstance(snapshot, Mapping):
        raise ExclusionError(f"{label} snapshot is missing")
    if (
        snapshot.get("kind") != "file"
        or snapshot.get("path") != record.get("path")
        or snapshot.get("sha256") != record.get("sha256")
        or snapshot.get("bytes") != record.get("bytes")
        or isinstance(snapshot.get("lines"), bool)
        or not isinstance(snapshot.get("lines"), int)
        or snapshot["lines"] <= 0
    ):
        raise ExclusionError(f"{label} snapshot differs from materialization")


def _validate_materialization(
    *, plan: Mapping[str, Any], plan_path: pathlib.Path,
    output_root: pathlib.Path, context: Mapping[str, Any],
) -> dict[str, Any]:
    materialization_path = pathlib.Path(str(plan["materialization_path"]))
    expected = output_root / "fresh-holdout/materialization-receipt.json"
    if (
        materialization_path != expected
        or materialization_path.is_symlink()
        or not materialization_path.is_file()
    ):
        raise ExclusionError("fresh materialization is absent or redirected")
    materialization = qualification.load_sealed(
        materialization_path, holdout.MATERIALIZATION_SCHEMA
    )
    claim_path = output_root / "fresh-holdout/00-materialization-claim.json"
    claim = qualification.load_sealed(claim_path, holdout.CLAIM_SCHEMA)
    expected_materialization_fields = {
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
    expected_claim_fields = {
        "schema", "namespace", "campaign_id", "status", "successor_plan",
        "immutable_selection", "selected_runtime", "prior_runtime",
        "configuration", "selection_may_change", "old_protected_tests_permitted",
        "materialization_attempts_authorized", "exclusive_process_lock",
        "claimed_at_utc", "body_sha256",
    }
    expected_selection = context["selection_reference"]
    if (
        set(materialization) != expected_materialization_fields
        or materialization.get("namespace") != NAMESPACE
        or materialization.get("campaign_id") != HOLDOUT_CAMPAIGN_ID
        or materialization.get("status")
        != "fresh-protected-holdout-materialized-once"
        or materialization.get("claim")
        != _reference(claim_path, holdout.CLAIM_SCHEMA)
        or materialization.get("immutable_selection") != expected_selection
        or materialization.get("selection_changed") is not False
        or materialization.get("old_protected_tests_accessed") is not False
        or materialization.get("fresh_protected_tests_opened") is not True
        or materialization.get("group_isolation", {}).get("passed") is not True
        or materialization.get("split_isolation", {}).get("passed") is not True
    ):
        raise ExclusionError("fresh materialization ancestry or isolation changed")
    configuration = context["plan"]["fresh_protected_holdout"]
    if (
        set(claim) != expected_claim_fields
        or claim.get("namespace") != NAMESPACE
        or claim.get("campaign_id") != HOLDOUT_CAMPAIGN_ID
        or claim.get("status")
        != "fresh-protected-holdout-materialization-claimed-once"
        or claim.get("successor_plan") != context["plan_reference"]
        or claim.get("immutable_selection") != expected_selection
        or claim.get("selected_runtime") != context["selection"]["runtime"]
        or claim.get("configuration") != configuration
        or claim.get("selection_may_change") is not False
        or claim.get("old_protected_tests_permitted") is not False
        or claim.get("materialization_attempts_authorized") != 1
        or claim.get("exclusive_process_lock")
        != str((output_root / "fresh-holdout/materialization.lock").resolve())
    ):
        raise ExclusionError("fresh materialization claim changed")
    _utc(claim.get("claimed_at_utc"), "materialization claim time")
    _verify_record(claim.get("selected_runtime"), "claimed selected runtime")
    _verify_record(claim.get("prior_runtime"), "claimed prior runtime")
    samples = materialization.get("test_samples")
    shards = materialization.get("test_shards")
    if (
        not isinstance(samples, Mapping)
        or set(samples) != {"search", "rank4", "canonical"}
        or any(
            isinstance(samples.get(name), bool)
            or not isinstance(samples.get(name), int)
            or samples[name] < int(configuration["minimum_samples_per_report"])
            for name in ("search", "rank4", "canonical")
        )
        or not isinstance(shards, Mapping)
        or set(shards) != {"search", "rank4", "canonical"}
    ):
        raise ExclusionError("fresh materialization test-shard roster changed")
    for name in ("search", "rank4", "canonical"):
        _verify_record(shards[name], f"fresh {name} test shard")
    stage_receipts = materialization.get("stage_receipts")
    if not isinstance(stage_receipts, list) or len(stage_receipts) != 13:
        raise ExclusionError("fresh materialization stage receipt roster changed")
    for index, record in enumerate(stage_receipts):
        _verify_record(record, f"fresh stage receipt {index + 1}")
    positions_record = materialization.get("positions")
    manifest_record = materialization.get("positions_manifest")
    positions_path = _verify_record(positions_record, "fresh positions")
    manifest_path = _verify_record(manifest_record, "fresh position manifest")
    if (
        positions_path != pathlib.Path(str(plan["positions_path"]))
        or manifest_path != pathlib.Path(str(plan["positions_manifest_path"]))
    ):
        raise ExclusionError("fresh position paths changed")
    position_manifest = _load_canonical_json(manifest_path, "fresh position manifest")
    if (
        position_manifest.get("schema") != selfsearch.POSITION_MANIFEST_SCHEMA
        or position_manifest.get("campaign_id") != HOLDOUT_CAMPAIGN_ID
        or position_manifest.get("maximum_positions_per_game") != POSITIONS_PER_GAME
        or position_manifest.get("positions") != POSITION_COUNT
        or position_manifest.get("split_counts") != {"test": POSITION_COUNT}
        or position_manifest.get("output_sha256") != positions_record["sha256"]
    ):
        raise ExclusionError("fresh position manifest contract changed")
    rows = position_manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != POSITION_COUNT:
        raise ExclusionError("fresh position manifest does not contain 64,000 rows")
    source_paths: dict[str, pathlib.Path] = {}
    for snapshot_name, materialization_name in (
        ("games", "games"), ("game_manifest", "games_manifest"),
        ("roots", "fresh_roots"),
    ):
        record = materialization.get(materialization_name)
        if not isinstance(record, Mapping):
            raise ExclusionError(f"materialization omits {materialization_name}")
        source_paths[snapshot_name] = _verify_record(
            record, f"fresh {materialization_name}"
        )
        _verify_snapshot_against_record(
            position_manifest.get(snapshot_name), record, snapshot_name
        )
    try:
        regenerated_payload, regenerated_manifest = selfsearch.freeze_positions(
            campaign_id=HOLDOUT_CAMPAIGN_ID,
            games_tsv=source_paths["games"],
            games_manifest=source_paths["game_manifest"],
            roots_manifest=source_paths["roots"],
            maximum_per_game=POSITIONS_PER_GAME,
        )
    except Exception as error:
        raise ExclusionError(
            "maintained freeze_positions could not reproduce fresh ancestry"
        ) from error
    regenerated_manifest_bytes = selfsearch.canonical_json_bytes(
        regenerated_manifest, pretty=True
    )
    if (
        regenerated_payload != positions_path.read_bytes()
        or regenerated_manifest != position_manifest
        or regenerated_manifest_bytes != manifest_path.read_bytes()
    ):
        raise ExclusionError(
            "fresh positions are not the exact maintained freeze_positions output"
        )
    return {
        "path": materialization_path,
        "document": materialization,
        "positions_path": positions_path,
        "positions_record": dict(positions_record),
        "manifest_path": manifest_path,
        "manifest_record": dict(manifest_record),
        "manifest": position_manifest,
        "producer_replay_validated": True,
    }


def _replay_fresh_position_prefix(prefix: Any) -> Any:
    if not isinstance(prefix, str):
        raise ExclusionError("fresh position prefix must be text")
    state = opening_tools.reference.ReplayState()
    if prefix == "":
        return state
    actions = prefix.split("/")
    if any(not action or re.fullmatch(r"[0-7]+", action) is None for action in actions):
        raise ExclusionError("fresh position prefix is not complete-turn text")
    for action in actions:
        if state.winner is not None:
            raise ExclusionError("fresh position prefix continues after terminal state")
        try:
            opening_tools.reference.apply_complete_turn(
                state, state.to_move, action
            )
        except ValueError as error:
            raise ExclusionError(
                "fresh position prefix is not a legal complete-turn prefix"
            ) from error
    if state.winner is not None:
        raise ExclusionError("fresh position prefix is terminal")
    return state


def _fresh_position_prefix_identity(
    prefix: Any, expected_mover: Any,
) -> tuple[str, int]:
    state = _replay_fresh_position_prefix(prefix)
    if expected_mover not in (0, 1) or state.to_move != expected_mover:
        raise ExclusionError("fresh position mover differs from replayed state")
    fingerprints = opening_tools.state_fingerprints(state)
    if set(fingerprints) != {*SYMMETRY_NAMES, "canonical"}:
        raise ExclusionError("fresh position symmetry roster changed")
    canonical = min(fingerprints[name] for name in SYMMETRY_NAMES)
    if fingerprints.get("canonical") != canonical:
        raise ExclusionError("fresh position canonical fingerprint changed")
    return canonical, state.to_move


def _fingerprint_positions(
    positions_path: pathlib.Path, manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], set[str]]:
    try:
        lines = positions_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ExclusionError("cannot read fresh protected positions") from error
    expected_header = (
        "position_id\troot_group_id\tgroup_id\tsource\tsplit\twinner\tmover\tprefix"
    )
    if not lines or lines[0] != expected_header or len(lines) != POSITION_COUNT + 1:
        raise ExclusionError("fresh positions TSV does not contain exactly 64,000 rows")
    manifest_rows = manifest["rows"]
    output: list[dict[str, Any]] = []
    unique: set[str] = set()
    position_ids: set[str] = set()
    group_roots: dict[str, str] = {}
    group_counts: dict[str, int] = {}
    cached: dict[str, tuple[str, int]] = {}
    for ordinal, (line, manifest_row) in enumerate(
        zip(lines[1:], manifest_rows, strict=True)
    ):
        fields = line.split("\t")
        if len(fields) != 8 or not isinstance(manifest_row, Mapping):
            raise ExclusionError("fresh position row is malformed")
        position_id, root_group_id, group_id, source, split, winner_raw, mover_raw, prefix = fields
        if (
            not position_id or position_id in position_ids
            or not root_group_id.startswith("fresh-protected-root:")
            or not group_id or not source
        ):
            raise ExclusionError("fresh position identities are empty or repeated")
        try:
            winner = int(winner_raw)
            mover = int(mover_raw)
        except ValueError as error:
            raise ExclusionError("fresh position winner/mover is invalid") from error
        if (
            split != "test" or winner_raw not in {"0", "1"}
            or mover_raw not in {"0", "1"}
            or winner not in (0, 1) or mover not in (0, 1)
        ):
            raise ExclusionError("fresh position split/winner/mover changed")
        expected_manifest_fields = {
            "position_id", "row_ordinal", "game_id", "game_row_ordinal",
            "turn", "root_group_id", "split", "mover", "winner",
        }
        if (
            set(manifest_row) != expected_manifest_fields
            or manifest_row.get("row_ordinal") != ordinal
            or manifest_row.get("position_id") != position_id
            or manifest_row.get("game_id") != group_id
            or manifest_row.get("root_group_id") != root_group_id
            or manifest_row.get("split") != "test"
            or manifest_row.get("winner") != winner
            or manifest_row.get("mover") != mover
            or type(manifest_row.get("game_row_ordinal")) is not int
            or not 0 <= manifest_row["game_row_ordinal"] < 3_200
            or type(manifest_row.get("turn")) is not int
            or manifest_row["turn"] < 0
        ):
            raise ExclusionError("fresh position TSV/manifest row ancestry changed")
        if prefix in cached:
            canonical, replayed_mover = cached[prefix]
        else:
            try:
                canonical, replayed_mover = _fresh_position_prefix_identity(
                    prefix, mover
                )
            except Exception as error:
                if isinstance(error, ExclusionError):
                    raise
                raise ExclusionError(
                    "fresh position prefix fingerprinting failed"
                ) from error
            cached[prefix] = (canonical, replayed_mover)
        if replayed_mover != mover:
            raise ExclusionError("fresh position mover differs from replayed state")
        previous_root = group_roots.setdefault(group_id, root_group_id)
        if previous_root != root_group_id:
            raise ExclusionError("fresh game spans multiple root groups")
        group_counts[group_id] = group_counts.get(group_id, 0) + 1
        position_ids.add(position_id)
        unique.add(canonical)
        output.append({
            "row_ordinal": ordinal,
            "position_id": position_id,
            "canonical_sha256": canonical,
        })
    if (
        len(output) != POSITION_COUNT or len(position_ids) != POSITION_COUNT
        or len(group_counts) != 3_200
        or any(count != POSITIONS_PER_GAME for count in group_counts.values())
    ):
        raise ExclusionError("fresh canonical fingerprint payload is incomplete")
    return output, unique


def _fingerprint_body(
    *, context: Mapping[str, Any], materialization: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]], unique_count: int,
) -> dict[str, Any]:
    ordered = [str(row["canonical_sha256"]) for row in rows]
    return {
        "schema": FINGERPRINT_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID,
        "classification": "protected-derived-fresh-holdout-canonical-fingerprints",
        "v3_plan": dict(context["plan_reference"]),
        "immutable_selection": dict(context["selection_reference"]),
        "materialization": _reference(
            pathlib.Path(materialization["path"]), holdout.MATERIALIZATION_SCHEMA
        ),
        "positions": dict(materialization["positions_record"]),
        "positions_manifest": dict(materialization["manifest_record"]),
        "position_count": POSITION_COUNT,
        "unique_canonical_count": unique_count,
        "duplicate_position_state_count": POSITION_COUNT - unique_count,
        "symmetries": list(SYMMETRY_NAMES),
        "canonicalization": CANONICALIZATION,
        "ordered_canonical_sha256": qualification.sha256_bytes(
            qualification.canonical_json_bytes(ordered)
        ),
        "rows": [dict(row) for row in rows],
        "contains_transcripts": False,
        "contains_labels": False,
        "contains_metrics": False,
    }


def _receipt_body(
    *, plan_path: pathlib.Path, materialization_path: pathlib.Path,
    fingerprint_path: pathlib.Path,
    bank_records: Mapping[str, Mapping[str, Any]],
    fresh_rows: Sequence[Mapping[str, Any]], fresh_unique: set[str],
    development_canonical: set[str],
) -> dict[str, Any]:
    overlap = fresh_unique & development_canonical
    overlapping_rows = sum(
        str(row["canonical_sha256"]) in development_canonical for row in fresh_rows
    )
    reusable = not overlap
    return {
        "schema": RECEIPT_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID,
        "status": (
            "zero-overlap-existing-development-roster-ready"
            if reusable
            else "overlap-detected-full-development-roster-regeneration-required"
        ),
        "references": {
            "plan": _reference(plan_path, PLAN_SCHEMA),
            "materialization": _reference(
                materialization_path, holdout.MATERIALIZATION_SCHEMA
            ),
            "protected_canonical_fingerprints": _reference(
                fingerprint_path, FINGERPRINT_SCHEMA
            ),
            "development_banks": {
                stage: {
                    "path": str(bank_records[stage]["path"]),
                    "sha256": str(bank_records[stage]["sha256"]),
                }
                for stage in STAGE_ORDER
            },
        },
        "counts": {
            "fresh_position_rows": POSITION_COUNT,
            "fresh_unique_canonical": len(fresh_unique),
            "fresh_duplicate_position_states": POSITION_COUNT - len(fresh_unique),
            "development_openings": DEVELOPMENT_OPENINGS,
            "development_unique_canonical": len(development_canonical),
            "development_stage_openings": dict(STAGE_COUNTS),
        },
        "intersection": {
            "unique_canonical_count": len(overlap),
            "fresh_position_row_count": overlapping_rows,
        },
        "verdict": {
            "existing_full_roster_reusable": reusable,
            "full_roster_regeneration_required": not reusable,
            "partial_roster_replacement_authorized": False,
            "development_games_authorized": reusable,
            "final_bank_generation_authorized": False,
            "rank4_gate_authorized": False,
            "upload_authorized": False,
        },
    }


def audit(
    *, output_root: pathlib.Path, plan_path: pathlib.Path,
    campaign_validator: CampaignValidator = _default_campaign_validator,
    bank_identity_validator: BankIdentityValidator = _default_bank_identity_validator,
    retirement_validator: RetirementValidator = validate_v2_retirement,
) -> pathlib.Path:
    output_root = _safe_output_root(output_root)
    plan = validate_plan(
        plan_path, output_root=output_root, campaign_validator=campaign_validator,
        bank_identity_validator=bank_identity_validator,
        retirement_validator=retirement_validator,
    )
    _safe_directory(
        pathlib.Path(plan["outputs"]["protected_directory"]), create=True
    )
    _safe_output_file(pathlib.Path(plan["outputs"]["public_receipt"]))
    context = _campaign_context(
        pathlib.Path(plan["v3_plan"]["path"]), output_root, campaign_validator
    )
    selection_sha_before = qualification.sha256_file(context["selection_path"])
    materialization = _validate_materialization(
        plan=plan, plan_path=plan_path, output_root=output_root, context=context
    )
    rows, fresh_unique = _fingerprint_positions(
        materialization["positions_path"], materialization["manifest"]
    )
    bank_paths = {
        stage: _verify_record(plan["development_banks"][stage], f"planned {stage} bank")
        for stage in STAGE_ORDER
    }
    bank_records, development_canonical = _bank_roster(bank_paths)
    fingerprint_path = _write_content_addressed(
        pathlib.Path(plan["outputs"]["protected_directory"]),
        _fingerprint_body(
            context=context,
            materialization={
                **materialization,
                "path": str(materialization["path"]),
            },
            rows=rows,
            unique_count=len(fresh_unique),
        ),
        ".fresh-canonical-fingerprints.json",
    )
    receipt_path = _safe_output_file(
        pathlib.Path(plan["outputs"]["public_receipt"])
    )
    body = _receipt_body(
        plan_path=plan_path,
        materialization_path=materialization["path"],
        fingerprint_path=fingerprint_path,
        bank_records=bank_records,
        fresh_rows=rows,
        fresh_unique=fresh_unique,
        development_canonical=development_canonical,
    )
    if receipt_path.exists():
        existing = qualification.load_sealed(receipt_path, RECEIPT_SCHEMA)
        if existing != qualification.seal(body):
            raise ExclusionError("existing symmetry-exclusion receipt changed")
    else:
        qualification.write_sealed(receipt_path, body)
    if qualification.sha256_file(context["selection_path"]) != selection_sha_before:
        raise ExclusionError("v3 immutable selection changed during exclusion audit")
    validate_receipt(
        receipt_path, plan_path=plan_path, output_root=output_root,
        campaign_validator=campaign_validator, recompute_positions=False,
        bank_identity_validator=bank_identity_validator,
        retirement_validator=retirement_validator,
    )
    return receipt_path


def _load_fingerprint_payload(
    path: pathlib.Path, *, context: Mapping[str, Any],
    materialization: Mapping[str, Any], recomputed_rows: Sequence[Mapping[str, Any]] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], set[str]]:
    if (
        path.parent.name != "protected"
        or not path.name.endswith(".fresh-canonical-fingerprints.json")
        or path.name.removesuffix(".fresh-canonical-fingerprints.json")
        != qualification.sha256_file(path)
    ):
        raise ExclusionError("protected canonical fingerprint path is not content addressed")
    payload = qualification.load_sealed(path, FINGERPRINT_SCHEMA)
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != POSITION_COUNT:
        raise ExclusionError("protected canonical fingerprint payload is not 64,000 rows")
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    unique: set[str] = set()
    for ordinal, row in enumerate(rows):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"row_ordinal", "position_id", "canonical_sha256"}
            or row.get("row_ordinal") != ordinal
            or not isinstance(row.get("position_id"), str)
            or not row["position_id"]
            or row["position_id"] in seen_ids
        ):
            raise ExclusionError("protected canonical fingerprint row is malformed")
        canonical = _sha(row.get("canonical_sha256"), "canonical fingerprint")
        seen_ids.add(row["position_id"])
        unique.add(canonical)
        normalized.append(dict(row))
    ordered = [row["canonical_sha256"] for row in normalized]
    if (
        payload.get("namespace") != NAMESPACE
        or payload.get("campaign_id") != CAMPAIGN_ID
        or payload.get("classification")
        != "protected-derived-fresh-holdout-canonical-fingerprints"
        or payload.get("v3_plan") != context["plan_reference"]
        or payload.get("immutable_selection") != context["selection_reference"]
        or payload.get("materialization")
        != _reference(materialization["path"], holdout.MATERIALIZATION_SCHEMA)
        or payload.get("positions") != materialization["positions_record"]
        or payload.get("positions_manifest") != materialization["manifest_record"]
        or payload.get("position_count") != POSITION_COUNT
        or payload.get("unique_canonical_count") != len(unique)
        or payload.get("duplicate_position_state_count") != POSITION_COUNT - len(unique)
        or payload.get("symmetries") != list(SYMMETRY_NAMES)
        or payload.get("canonicalization") != CANONICALIZATION
        or payload.get("ordered_canonical_sha256")
        != qualification.sha256_bytes(qualification.canonical_json_bytes(ordered))
        or payload.get("contains_transcripts") is not False
        or payload.get("contains_labels") is not False
        or payload.get("contains_metrics") is not False
    ):
        raise ExclusionError("protected canonical fingerprint ancestry changed")
    if recomputed_rows is not None and normalized != list(recomputed_rows):
        raise ExclusionError("protected canonical fingerprints differ from positions")
    return payload, normalized, unique


def _validate_receipt_details(
    receipt_path: pathlib.Path, *, plan_path: pathlib.Path,
    output_root: pathlib.Path,
    campaign_validator: CampaignValidator = _default_campaign_validator,
    bank_identity_validator: BankIdentityValidator = _default_bank_identity_validator,
    retirement_validator: RetirementValidator = validate_v2_retirement,
    recompute_positions: bool = True,
) -> dict[str, Any]:
    output_root = _safe_output_root(output_root)
    plan = validate_plan(
        plan_path, output_root=output_root, campaign_validator=campaign_validator,
        bank_identity_validator=bank_identity_validator,
        retirement_validator=retirement_validator,
    )
    expected_receipt = _safe_output_file(
        pathlib.Path(plan["outputs"]["public_receipt"])
    )
    if (
        receipt_path.resolve() != expected_receipt
        or receipt_path.is_symlink()
        or not receipt_path.is_file()
    ):
        raise ExclusionError("symmetry-exclusion receipt path is not canonical")
    receipt = qualification.load_sealed(receipt_path, RECEIPT_SCHEMA)
    context = _campaign_context(
        pathlib.Path(plan["v3_plan"]["path"]), output_root, campaign_validator
    )
    materialization = _validate_materialization(
        plan=plan, plan_path=plan_path, output_root=output_root, context=context
    )
    references = receipt.get("references")
    if not isinstance(references, Mapping) or set(references) != {
        "plan", "materialization", "protected_canonical_fingerprints",
        "development_banks",
    }:
        raise ExclusionError("public symmetry receipt references are malformed")
    if (
        references.get("plan") != _reference(plan_path, PLAN_SCHEMA)
        or references.get("materialization")
        != _reference(materialization["path"], holdout.MATERIALIZATION_SCHEMA)
    ):
        raise ExclusionError("public symmetry receipt ancestry changed")
    fingerprint_path = _verify_reference(
        references.get("protected_canonical_fingerprints"),
        FINGERPRINT_SCHEMA,
        "protected canonical fingerprints",
    )
    if fingerprint_path.parent != pathlib.Path(plan["outputs"]["protected_directory"]):
        raise ExclusionError("protected canonical fingerprints escaped fixed directory")
    recomputed = None
    if recompute_positions:
        recomputed, _fresh = _fingerprint_positions(
            materialization["positions_path"], materialization["manifest"]
        )
    _payload, rows, fresh_unique = _load_fingerprint_payload(
        fingerprint_path, context=context, materialization=materialization,
        recomputed_rows=recomputed,
    )
    planned_paths = {
        stage: _verify_record(plan["development_banks"][stage], f"planned {stage} bank")
        for stage in STAGE_ORDER
    }
    bank_records, development_canonical = _bank_roster(planned_paths)
    public_banks = references.get("development_banks")
    expected_public_banks = {
        stage: {
            "path": bank_records[stage]["path"],
            "sha256": bank_records[stage]["sha256"],
        }
        for stage in STAGE_ORDER
    }
    if public_banks != expected_public_banks:
        raise ExclusionError("public symmetry receipt development banks changed")
    expected_body = qualification.seal(_receipt_body(
        plan_path=plan_path,
        materialization_path=materialization["path"],
        fingerprint_path=fingerprint_path,
        bank_records=bank_records,
        fresh_rows=rows,
        fresh_unique=fresh_unique,
        development_canonical=development_canonical,
    ))
    if receipt != expected_body:
        raise ExclusionError("public symmetry-exclusion receipt changed")
    return {
        "receipt": receipt,
        "plan": plan,
        "protected_fingerprint_path": fingerprint_path,
        "fresh_canonical_fingerprints": fresh_unique,
        "development_bank_records": bank_records,
        "development_ready": receipt["verdict"]["development_games_authorized"],
    }


def validate_receipt(
    receipt_path: pathlib.Path, *, plan_path: pathlib.Path,
    output_root: pathlib.Path,
    campaign_validator: CampaignValidator = _default_campaign_validator,
    bank_identity_validator: BankIdentityValidator = _default_bank_identity_validator,
    retirement_validator: RetirementValidator = validate_v2_retirement,
    recompute_positions: bool = True,
) -> dict[str, Any]:
    details = _validate_receipt_details(
        receipt_path,
        plan_path=plan_path,
        output_root=output_root,
        campaign_validator=campaign_validator,
        bank_identity_validator=bank_identity_validator,
        retirement_validator=retirement_validator,
        recompute_positions=recompute_positions,
    )
    return {
        "receipt": details["receipt"],
        "plan": details["plan"],
        "protected_fingerprint_path": details["protected_fingerprint_path"],
        "development_bank_records": details["development_bank_records"],
        "development_ready": details["development_ready"],
    }


def _load_private_canonical_fingerprints(
    receipt_path: pathlib.Path, *, plan_path: pathlib.Path,
    output_root: pathlib.Path,
    campaign_validator: CampaignValidator = _default_campaign_validator,
    bank_identity_validator: BankIdentityValidator = _default_bank_identity_validator,
    retirement_validator: RetirementValidator = validate_v2_retirement,
) -> frozenset[str]:
    """Return protected fingerprints only to in-module final-bank integration."""

    details = _validate_receipt_details(
        receipt_path,
        plan_path=plan_path,
        output_root=output_root,
        campaign_validator=campaign_validator,
        bank_identity_validator=bank_identity_validator,
        retirement_validator=retirement_validator,
        recompute_positions=True,
    )
    return frozenset(details["fresh_canonical_fingerprints"])


def require_development_roster(
    receipt_path: pathlib.Path, *, plan_path: pathlib.Path,
    output_root: pathlib.Path,
    campaign_validator: CampaignValidator = _default_campaign_validator,
    bank_identity_validator: BankIdentityValidator = _default_bank_identity_validator,
    retirement_validator: RetirementValidator = validate_v2_retirement,
) -> dict[str, dict[str, Any]]:
    validated = validate_receipt(
        receipt_path, plan_path=plan_path, output_root=output_root,
        campaign_validator=campaign_validator,
        bank_identity_validator=bank_identity_validator,
        retirement_validator=retirement_validator,
    )
    if validated["development_ready"] is not True:
        policy = validated["plan"]["regeneration_policy"]
        if (
            policy.get("action") != "regenerate-entire-six-bank-development-roster"
            or policy.get("partial_roster_replacement_authorized") is not False
        ):
            raise ExclusionError("full-roster regeneration policy changed")
        raise ExclusionError(
            "fresh/development symmetry overlap requires full six-bank regeneration; "
            "partial reuse is forbidden and development remains unauthorized"
        )
    return validated["development_bank_records"]


def _parse_banks(values: Sequence[str]) -> dict[str, pathlib.Path]:
    result: dict[str, pathlib.Path] = {}
    for raw in values:
        if "=" not in raw:
            raise ExclusionError("--development-bank must be STAGE=PATH")
        stage, value = raw.split("=", 1)
        if stage in result:
            raise ExclusionError("development bank stage is repeated")
        result[stage] = pathlib.Path(value)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    retire_v1_command = commands.add_parser("retire-v1")
    retire_v1_command.add_argument("--output-root", type=pathlib.Path, required=True)
    retire_v1_command.add_argument("--retired-at-utc", default=utc_now())
    retire_v2_command = commands.add_parser("retire-v2")
    retire_v2_command.add_argument("--output-root", type=pathlib.Path, required=True)
    retire_v2_command.add_argument("--retired-at-utc", default=utc_now())
    prepare_command = commands.add_parser("prepare")
    prepare_command.add_argument("--output-root", type=pathlib.Path, required=True)
    prepare_command.add_argument("--v3-plan", type=pathlib.Path, required=True)
    prepare_command.add_argument(
        "--v2-retirement", type=pathlib.Path, required=True
    )
    prepare_command.add_argument(
        "--development-bank", action="append", required=True,
    )
    prepare_command.add_argument("--created-at-utc", default=utc_now())
    audit_command = commands.add_parser("audit")
    audit_command.add_argument("--output-root", type=pathlib.Path, required=True)
    audit_command.add_argument("--plan", type=pathlib.Path, required=True)
    validate_command = commands.add_parser("validate")
    validate_command.add_argument("--output-root", type=pathlib.Path, required=True)
    validate_command.add_argument("--plan", type=pathlib.Path, required=True)
    validate_command.add_argument("--receipt", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "retire-v1":
            result = retire_v1(
                output_root=args.output_root,
                retired_at_utc=args.retired_at_utc,
            )
            output: Any = {
                "retirement": str(result),
                "sha256": qualification.sha256_file(result),
            }
        elif args.command == "retire-v2":
            result = retire_v2(
                output_root=args.output_root,
                retired_at_utc=args.retired_at_utc,
            )
            output = {
                "retirement": str(result),
                "sha256": qualification.sha256_file(result),
            }
        elif args.command == "prepare":
            result = prepare(
                output_root=args.output_root,
                v3_plan_path=args.v3_plan,
                v2_retirement_path=args.v2_retirement,
                development_bank_paths=_parse_banks(args.development_bank),
                created_at_utc=args.created_at_utc,
            )
            output = {"plan": str(result), "sha256": qualification.sha256_file(result)}
        elif args.command == "audit":
            result = audit(
                output_root=args.output_root, plan_path=args.plan,
            )
            value = qualification.load_sealed(result, RECEIPT_SCHEMA)
            output = {
                "receipt": str(result),
                "sha256": qualification.sha256_file(result),
                "status": value["status"],
                "counts": value["counts"],
                "intersection": value["intersection"],
                "verdict": value["verdict"],
            }
        else:
            validated = validate_receipt(
                args.receipt, plan_path=args.plan, output_root=args.output_root,
            )
            output = {
                "receipt": str(args.receipt.resolve()),
                "sha256": qualification.sha256_file(args.receipt),
                "status": validated["receipt"]["status"],
                "counts": validated["receipt"]["counts"],
                "intersection": validated["receipt"]["intersection"],
                "verdict": validated["receipt"]["verdict"],
            }
        print(json.dumps(output, sort_keys=True, allow_nan=False))
        return 0
    except (ExclusionError, OSError, UnicodeError, json.JSONDecodeError) as error:
        print(f"compact v3 symmetry-exclusion failure: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
