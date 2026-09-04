#!/usr/bin/env python3
"""Release, preflight, CI, and one-upload bridge for the Rank-4 challenger.

The challenger selects either the immutable attempt-zero generated source or a
teacher-training source with a frozen search-variant macro prefix.  Neither is
uploaded directly.  This bridge proves the selected source was exported from
the selected 6301-12-8-1 runtime, applies only the frozen seven-slot deployment
configuration, promotes the exact release artifacts, and performs a
source-specific preflight on the clean committed release branch.

The sealed release evidence is deliberately produced before challenger dual
authorization.  After two protected gates pass, this module recursively
validates their deep evidence and emits the existing one-upload authorization
schema.  No command pushes Git, starts CI, opens protected banks, accesses a
browser, or uploads to CodinGame.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import functools
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
REPOSITORY = HERE.parent
TEST_PATH = (
    REPOSITORY
    / "tests/codingame/test_compact_value_bfm_rank4_teacher_release.py"
)


def _load(path: pathlib.Path, name: str) -> Any:
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Rank-4 release dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


qualification = _load(
    HERE / "compact_value_bfm_qualification.py",
    "rank4_teacher_release_qualification",
)
maintained = _load(
    HERE / "compact_value_bfm_preflight.py",
    "rank4_teacher_release_maintained_preflight",
)
deployment = _load(
    HERE / "compact_value_bfm_discrete_v3_deployment.py",
    "rank4_teacher_release_deployment",
)
deployment_preflight = _load(
    HERE / "compact_value_bfm_discrete_v3_deployment_preflight.py",
    "rank4_teacher_release_deployment_preflight",
)
upload = _load(
    HERE / "compact_value_bfm_upload.py",
    "rank4_teacher_release_upload",
)
challenger = _load(
    HERE / "compact_value_bfm_rank4_teacher_challenger.py",
    "rank4_teacher_release_challenger",
)
source_exporter = _load(
    REPOSITORY
    / "submissions/codingame/bots/compact_value_bfm/export_submission.py",
    "rank4_teacher_release_source_exporter",
)


class ReleaseBridgeError(ValueError):
    """The challenger release chain is incomplete or changed."""


NAMESPACE = challenger.NAMESPACE
CAMPAIGN_ID = challenger.CAMPAIGN_ID
RELEASE_BRANCH = upload.BRANCH
BOT_RELATIVE = maintained.BOT_RELATIVE
RANK4_RELATIVE = maintained.RANK4_RELATIVE
SOURCE_LIMIT_EXCLUSIVE = 95_000
SOURCE_RESERVE_TARGET = 2_000
SOURCE_MAXIMUM_FOR_TARGET = SOURCE_LIMIT_EXCLUSIVE - SOURCE_RESERVE_TARGET

PROMOTION_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-challenger-promotion.v1"
)
SOURCE_DERIVATION_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-challenger-"
    "release-source-derivation.v1"
)
PREFLIGHT_PLAN_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-challenger-"
    "release-preflight-plan.v1"
)
PREFLIGHT_CLAIM_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-challenger-"
    "release-preflight-claim.v1"
)
PREFLIGHT_RECEIPT_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-challenger-"
    "release-preflight-receipt.v1"
)
PREFLIGHT_REFERENCE_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-challenger-"
    "release-preflight-reference.v1"
)
RELEASE_EVIDENCE_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-challenger-release-evidence.v1"
)

ATTEMPT_ZERO_LEGACY_COMMIT = (
    "c380ae74b999eb6fd16d7bbfd49e16cc24f95ded"
)
ATTEMPT_ZERO_LEGACY_CLOSURE = {
    "submissions/codingame/bots/compact_value_bfm/export_submission.py":
        "6af655d91d86ad972835611c22fea8cd90998cb6f3d96a122d5e0002d8251d26",
    "submissions/codingame/bots/compact_value_bfm/export_model.py":
        "a9a30ec9df1636dcbc9aaf83c86eafcd506ab09e7c1f28506d78466fe72042a5",
    "submissions/codingame/bots/compact_value_bfm/submission.json":
        "3eba94d384c54a3812aa5f346f38298fc8d3fac60ff24b745d904c9cbba84de7",
    "submissions/codingame/bots/compact_value_bfm/sources.txt":
        "fa3f32f814c49b2fe03c1f09a41ded58d97ce9334398053737e593d06e38561a",
    "submissions/codingame/bots/compact_value_bfm/engine.hpp":
        "d0f48446b11cc1b4fad3dc8a169f3185d0bcdc891882a451fdebf65291eb774b",
    "submissions/codingame/bots/compact_value_bfm/engine.cpp":
        "d3621d5e62dc0b2359aeb9d155217a358c9b6e486cca85ef3f2e1acbf86ba05f",
    "submissions/codingame/bots/compact_value_bfm/bot.cpp":
        "8460ba488afa3d42bbab17e93678bb4fbf7dcfcc970f961119fe6c81a5f63d17",
}
UPLOAD_INPUTS_SCHEMA = (
    "papersoccer.compact-value-bfm.rank4-teacher-challenger-"
    "one-upload-authorization-inputs.v1"
)

MODEL_RELATIVE = BOT_RELATIVE / "model.hpp"
GENERATED_RELATIVE = BOT_RELATIVE / "submission.cpp"
DEPLOYED_RELATIVE = deployment.CANDIDATE_RELATIVE
MANIFEST_RELATIVE = deployment.MANIFEST_RELATIVE
PROMOTED_RELATIVES = (
    MODEL_RELATIVE,
    GENERATED_RELATIVE,
    DEPLOYED_RELATIVE,
    MANIFEST_RELATIVE,
)
PREFLIGHT_DIRECTORY = "release-preflight"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")


def _utc(value: Any, label: str) -> dt.datetime:
    try:
        return qualification._utc(value, label)
    except Exception as error:
        raise ReleaseBridgeError(f"{label} is not an ISO-8601 UTC timestamp") from error


def _record(
    path: pathlib.Path, *, ascii_required: bool = False,
    executable: bool = False, allow_symlink: bool = False,
) -> dict[str, Any]:
    path = pathlib.Path(path)
    if (
        (path.is_symlink() and not allow_symlink)
        or not path.is_file()
        or (executable and not os.access(path, os.X_OK))
    ):
        raise ReleaseBridgeError(f"required regular artifact is absent: {path}")
    raw = path.read_bytes()
    if ascii_required:
        try:
            raw.decode("ascii")
        except UnicodeDecodeError as error:
            raise ReleaseBridgeError(f"required artifact is not ASCII: {path}") from error
    return {
        "path": str(path.resolve()),
        "bytes": len(raw),
        "sha256": qualification.sha256_bytes(raw),
        **({"ascii": True} if ascii_required else {}),
        **({"executable": True} if executable else {}),
    }


def _verify_record(
    value: Any, label: str, *, ascii_required: bool = False,
    executable: bool = False, allow_symlink: bool = False,
) -> pathlib.Path:
    if not isinstance(value, Mapping):
        raise ReleaseBridgeError(f"{label} record is absent")
    path = pathlib.Path(str(value.get("path", "")))
    expected = _record(
        path, ascii_required=ascii_required, executable=executable,
        allow_symlink=allow_symlink,
    )
    if dict(value) != expected:
        raise ReleaseBridgeError(f"{label} changed")
    return path.resolve()


def _reference(path: pathlib.Path, schema: str) -> dict[str, str]:
    try:
        return qualification.artifact_reference(path, schema)
    except Exception as error:
        raise ReleaseBridgeError(f"sealed artifact failed validation: {path}") from error


def _verify_reference(value: Any, schema: str, label: str) -> pathlib.Path:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ReleaseBridgeError(f"{label} reference is malformed")
    path = pathlib.Path(str(value.get("path", "")))
    if path.is_symlink() or not path.is_file() or dict(value) != _reference(path, schema):
        raise ReleaseBridgeError(f"{label} reference changed")
    return path.resolve()


def _safe_directory(path: pathlib.Path, *, create: bool) -> pathlib.Path:
    absolute = pathlib.Path(path).absolute()
    resolved = absolute.resolve()
    if (
        absolute == pathlib.Path(absolute.anchor)
        or resolved == pathlib.Path(resolved.anchor)
        or absolute.is_symlink()
        or (absolute.exists() and not absolute.is_dir())
    ):
        raise ReleaseBridgeError(f"unsafe release directory: {path}")
    if create:
        absolute.mkdir(parents=True, exist_ok=True)
    if not absolute.is_dir():
        raise ReleaseBridgeError(f"release directory is absent: {path}")
    return absolute.resolve()


def _safe_output(path: pathlib.Path) -> pathlib.Path:
    parent = _safe_directory(path.parent, create=True)
    result = parent / path.name
    if result.is_symlink() or (result.exists() and not result.is_file()):
        raise ReleaseBridgeError(f"unsafe release output: {result}")
    return result


def _atomic_replace(path: pathlib.Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ReleaseBridgeError(f"promotion target is redirected or irregular: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _git(repository: pathlib.Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", *arguments], cwd=repository, capture_output=True, check=False,
    )
    if completed.returncode != 0:
        raise ReleaseBridgeError(
            f"Git verification failed: {' '.join(arguments[:3])}"
        )
    return completed.stdout


def _git_status_paths(repository: pathlib.Path) -> set[str]:
    raw = _git(
        repository, "status", "--porcelain=v1", "--untracked-files=all", "-z"
    )
    paths: set[str] = set()
    entries = [entry for entry in raw.split(b"\0") if entry]
    index = 0
    while index < len(entries):
        entry = entries[index]
        if len(entry) < 4:
            raise ReleaseBridgeError("Git status returned a malformed entry")
        status = entry[:2]
        try:
            path = entry[3:].decode("utf-8")
        except UnicodeDecodeError as error:
            raise ReleaseBridgeError("Git status contains a non-UTF-8 path") from error
        paths.add(path)
        if status[:1] in {b"R", b"C"} or status[1:2] in {b"R", b"C"}:
            index += 1
            if index >= len(entries):
                raise ReleaseBridgeError("Git rename/copy status is incomplete")
            paths.add(entries[index].decode("utf-8"))
        index += 1
    return paths


def _git_identity(
    repository: pathlib.Path, *, require_clean: bool,
    branch: str = RELEASE_BRANCH,
) -> dict[str, Any]:
    repository = pathlib.Path(repository).resolve()
    head = _git(repository, "rev-parse", "HEAD").decode("ascii").strip()
    current_branch = _git(
        repository, "branch", "--show-current"
    ).decode("ascii").strip()
    if qualification.COMMIT_RE.fullmatch(head) is None:
        raise ReleaseBridgeError("release HEAD is not a full lowercase commit")
    if current_branch != branch:
        raise ReleaseBridgeError(
            f"release checkout must be on exact branch {branch}"
        )
    dirty = _git_status_paths(repository)
    if require_clean and dirty:
        raise ReleaseBridgeError("release checkout is not completely clean")
    return {
        "repository": str(repository),
        "commit": head,
        "branch": current_branch,
        "head_ref": f"refs/heads/{current_branch}",
        "clean": not dirty,
        "dirty_paths": sorted(dirty),
    }


def _architecture(runtime_path: pathlib.Path) -> dict[str, Any]:
    try:
        runtime, _payload, metadata = challenger.export_model.validate_runtime(
            runtime_path
        )
    except Exception as error:
        raise ReleaseBridgeError("selected runtime failed compact validation") from error
    quantization = runtime.get("quantization")
    dimensions = runtime.get("architecture", {}).get("dimensions")
    if dimensions != challenger.DIMENSIONS or not isinstance(quantization, Mapping):
        raise ReleaseBridgeError("selected runtime is not fixed 6301-12-8-1")
    return {
        "id": challenger.ARCHITECTURE,
        "dimensions": list(challenger.DIMENSIONS),
        "biases": False,
        "outputs": 1,
        "head": "scalar-value-only",
        "policy_head": False,
        "runtime_body_sha256": metadata["body_sha256"],
        "payload_sha256": quantization["payload_sha256"],
    }


def _selected_candidate(
    campaign_plan_path: pathlib.Path, *, attempt: int,
    candidate_runtime: pathlib.Path, candidate_source: pathlib.Path,
) -> dict[str, Any]:
    """Resolve the canonical selected source and its frozen search macros."""

    try:
        context = challenger.validate_campaign(campaign_plan_path)
        entries = challenger.load_ledger(context["plan"])
    except Exception as error:
        raise ReleaseBridgeError("challenger campaign failed validation") from error
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0:
        raise ReleaseBridgeError("release attempt is invalid")
    source_kind: str
    macros: list[str]
    configuration: dict[str, Any]
    search_throughput_profile = "standard-v1"
    candidate_search_profile = "standard-v1"
    search_variant: str | None = None
    if attempt == 0:
        matches = [
            entry for entry in entries
            if entry.get("event") == "attempt-zero-result-recorded"
            and entry.get("attempt") == 0 and entry.get("passed") is True
            and entry.get("adaptation_route") == "prepare-dual-final"
        ]
        if len(matches) != 1:
            raise ReleaseBridgeError("attempt zero is not selected for final release")
        event = matches[0]
        runtime_path = challenger._verify_bundle_record(
            context["inputs"]["candidate"]["runtime"],
            input_directory=pathlib.Path(
                context["plan"]["outputs"]["input_directory"]
            ),
            label="attempt-zero release runtime",
        )
        source_path = challenger._verify_bundle_record(
            context["inputs"]["candidate"]["source"],
            input_directory=pathlib.Path(
                context["plan"]["outputs"]["input_directory"]
            ),
            label="attempt-zero generated source",
        )
        expected_identity = event.get("candidate_identity")
        finalist_path = challenger._verify_sealed_record(
            event.get("referenced_finalist"), challenger.RECOVERY_FINALIST_SCHEMA,
            "attempt-zero selected finalist",
        )
        finalist = qualification.load_sealed(
            finalist_path, challenger.RECOVERY_FINALIST_SCHEMA
        )
        search_tuple = finalist.get("tuple")
        profile = finalist.get("profile")
        work = finalist.get("profile_work")
        configuration = deployment.deployment_configuration(
            search_tuple, profile, work
        )
        if expected_identity != {
            "runtime_sha256": qualification.sha256_file(runtime_path),
            "source_sha256": qualification.sha256_file(source_path),
        }:
            raise ReleaseBridgeError("attempt-zero selected identity changed")
        macros = []
        source_kind = "attempt-zero-exported-source"
        selection_evidence = challenger._sealed_record(
            finalist_path, challenger.RECOVERY_FINALIST_SCHEMA
        )
    else:
        matches = [
            entry for entry in entries
            if entry.get("event") == "attempt-outcome-recorded"
            and entry.get("attempt") == attempt
            and entry.get("phase") == "full"
            and entry.get("admitted") is True
            and entry.get("adaptation_route") == "prepare-dual-final"
        ]
        if len(matches) != 1:
            raise ReleaseBridgeError("trained attempt is not selected for final release")
        event = matches[0]
        candidate = event.get("candidate")
        if not isinstance(candidate, Mapping):
            raise ReleaseBridgeError("trained selected candidate is absent")
        runtime_path = challenger._verify_record(
            candidate.get("runtime"), "trained release runtime"
        )
        source_path = challenger._verify_record(
            candidate.get("source"), "trained generated source"
        )
        admission_path = challenger._verify_sealed_record(
            event.get("admission_receipt"),
            "papersoccer.compact-value-bfm.rank4-teacher-phase-admission.v1",
            "trained full admission",
        )
        teacher_training = _load(
            HERE / "compact_value_bfm_teacher_training.py",
            "rank4_teacher_release_training",
        )
        try:
            admission = teacher_training.load_phase_admission(admission_path)
        except Exception as error:
            raise ReleaseBridgeError("trained full admission failed validation") from error
        selected = admission.get("selected_candidate")
        if (
            admission.get("phase") != "full"
            or admission.get("attempt") != attempt
            or admission.get("admitted") is not True
            or not isinstance(selected, Mapping)
            or selected.get("runtime", {}).get("sha256")
            != qualification.sha256_file(runtime_path)
            or selected.get("source", {}).get("sha256")
            != qualification.sha256_file(source_path)
        ):
            raise ReleaseBridgeError("trained full admission selected another candidate")
        variant = selected.get("search_variant")
        search_throughput_profile = selected.get("search_throughput_profile")
        try:
            variants = teacher_training.active_search_variants(
                search_throughput_profile
            )
            metadata = teacher_training._search_variant_metadata(
                search_throughput_profile, variant
            )
        except Exception as error:
            raise ReleaseBridgeError(
                "trained search profile/variant is not frozen"
            ) from error
        if variant not in variants:
            raise ReleaseBridgeError("trained search variant is not frozen")
        macros = list(variants[variant])
        search_variant = str(variant)
        candidate_search_profile = str(metadata["candidate_search_profile"])
        if (
            selected.get("compile_time_macros") != macros
            or selected.get("candidate_search_profile")
            != candidate_search_profile
            or selected.get("standard_base_variant")
            != metadata["standard_base_variant"]
            or selected.get("search_treatment") is not metadata["is_treatment"]
        ):
            raise ReleaseBridgeError("trained search macro roster changed")
        configuration = deployment.deployment_configuration(
            ("0.95", "0.5", "1"), "default",
            deployment.PROFILE_ROSTER["default"],
        )
        source_kind = "teacher-training-search-variant-source"
        selection_evidence = challenger._sealed_record(
            admission_path,
            "papersoccer.compact-value-bfm.rank4-teacher-phase-admission.v1",
        )
    supplied_runtime = _record(candidate_runtime)
    supplied_source = _record(candidate_source, ascii_required=True)
    canonical_runtime = _record(runtime_path)
    canonical_source = _record(source_path, ascii_required=True)
    if any(
        supplied_runtime[key] != canonical_runtime[key]
        for key in ("bytes", "sha256")
    ) or any(
        supplied_source[key] != canonical_source[key]
        for key in ("bytes", "sha256", "ascii")
    ):
        raise ReleaseBridgeError("supplied release candidate is not the selected candidate")
    architecture = _architecture(runtime_path)
    frozen_execution_sources = None
    if context.get("inputs", {}).get("production_allowlist_enforced") is True:
        try:
            build_manifest_record = challenger._attempt_build_manifest_record(
                entries, attempt
            )
            frozen_execution_sources = (
                challenger._frozen_execution_source_evidence(
                    context,
                    tool_roles=challenger.POST_PROMOTION_RELEASE_TOOL_ROLES,
                    build_manifest_record=build_manifest_record,
                    revalidate_current=False,
                    allowed_current_drift_routes=tuple(
                        path.as_posix() for path in PROMOTED_RELATIVES
                    ),
                )
            )
        except Exception as error:
            raise ReleaseBridgeError(
                "release tools differ from the attempt's frozen source closure"
            ) from error
    return {
        "attempt": attempt,
        "origin": source_kind,
        "runtime": canonical_runtime,
        "generated_source": canonical_source,
        "architecture": architecture,
        "search_throughput_profile": search_throughput_profile,
        "candidate_search_profile": candidate_search_profile,
        "search_variant": search_variant,
        "compile_time_macros": macros,
        "configuration": configuration,
        "selection_evidence": selection_evidence,
        "frozen_execution_sources": frozen_execution_sources,
    }


SelectedValidator = Callable[..., Mapping[str, Any]]


def _guard_release_hooks(
    campaign_plan_path: pathlib.Path, *, hooks_used: bool,
    allow_injected_test_evidence: bool,
) -> Mapping[str, Any] | None:
    try:
        context = challenger.validate_campaign(campaign_plan_path.resolve())
    except Exception as error:
        if hooks_used and allow_injected_test_evidence is True:
            return None
        raise ReleaseBridgeError(
            "release campaign failed validation before hook authorization"
        ) from error
    production = (
        context.get("inputs", {}).get("production_allowlist_enforced") is True
    )
    if hooks_used and (production or allow_injected_test_evidence is not True):
        raise ReleaseBridgeError(
            "injected release hooks are explicit nonproduction test evidence only"
        )
    return context


def _production_release_root(
    context: Mapping[str, Any], *, attempt: int,
) -> pathlib.Path:
    return (
        pathlib.Path(context["plan"]["outputs"]["root"]).resolve()
        / "release" / f"attempt-{attempt:03d}"
    )


@contextlib.contextmanager
def _release_heavy_stage_lease(
    campaign_plan_path: pathlib.Path, *, allow_injected_test_evidence: bool,
):
    try:
        context = challenger.validate_campaign(campaign_plan_path.resolve())
    except Exception:
        if allow_injected_test_evidence:
            yield
            return
        raise
    if context.get("inputs", {}).get("production_allowlist_enforced") is not True:
        yield
        return
    lock_path = (
        pathlib.Path(context["plan"]["outputs"]["root"]).resolve()
        / ".rank4-teacher-heavy-stage.lock"
    )
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise ReleaseBridgeError(
                "another Rank-4 campaign heavy stage is active"
            ) from error
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _release_heavy_stage(function):
    @functools.wraps(function)
    def guarded(plan_path: pathlib.Path, *args, **kwargs):
        campaign_plan_path = kwargs.get("campaign_plan_path")
        if campaign_plan_path is None:
            raise ReleaseBridgeError("release heavy stage lacks campaign plan")
        with _release_heavy_stage_lease(
            campaign_plan_path,
            allow_injected_test_evidence=bool(
                kwargs.get("allow_injected_test_evidence", False)
            ),
        ):
            return function(plan_path, *args, **kwargs)
    return guarded


def _variant_source(base: bytes, macros: Sequence[str]) -> bytes:
    try:
        text = base.decode("ascii")
    except UnicodeDecodeError as error:
        raise ReleaseBridgeError("exported base source is not ASCII") from error
    if not macros:
        if not 0 < len(base) < challenger.SOURCE_LIMIT:
            raise ReleaseBridgeError("exported source violates 95KB")
        # Attempt zero was produced by the pre-compaction exporter.  Keeping
        # its bytes exact here makes the bridge usable from the c380ae7 release
        # lineage without silently reformatting the selected candidate.
        return base
    prefix = "".join(f"#define {macro} 1\n" for macro in macros)
    try:
        compactor = getattr(source_exporter, "compact_cpp_code", None)
        if compactor is None:
            raise ReleaseBridgeError(
                "search-macro candidate requires the audited compacting exporter"
            )
        result = compactor(
            prefix + text
        ).encode("ascii")
    except Exception as error:
        raise ReleaseBridgeError("search macro source derivation failed") from error
    if not 0 < len(result) < challenger.SOURCE_LIMIT:
        raise ReleaseBridgeError("search macro source violates 95KB")
    return result


def _legacy_attempt_zero_export(
    repository: pathlib.Path, runtime_path: pathlib.Path,
) -> tuple[bytes, dict[str, Any]]:
    """Rebuild the selected source with the exact pre-minifier c380 closure."""

    repository = pathlib.Path(repository).resolve()
    closure: dict[str, dict[str, Any]] = {}
    blobs: dict[str, bytes] = {}
    for relative, expected_sha256 in ATTEMPT_ZERO_LEGACY_CLOSURE.items():
        try:
            raw = _git(
                repository, "show", f"{ATTEMPT_ZERO_LEGACY_COMMIT}:{relative}"
            )
        except Exception as error:
            raise ReleaseBridgeError("legacy c380 exporter closure is unavailable") from error
        digest = qualification.sha256_bytes(raw)
        if digest != expected_sha256:
            raise ReleaseBridgeError("legacy c380 exporter closure changed")
        blobs[relative] = raw
        closure[relative] = {"bytes": len(raw), "sha256": digest}
    with tempfile.TemporaryDirectory(prefix="compact-rank4-legacy-export-") as temporary:
        root = pathlib.Path(temporary) / "repository"
        for relative, raw in blobs.items():
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw)
        exporter = root / (
            "submissions/codingame/bots/compact_value_bfm/export_submission.py"
        )
        output = pathlib.Path(temporary) / "submission.cpp"
        completed = subprocess.run(
            [
                sys.executable, str(exporter), "--runtime", str(runtime_path),
                "--render-output", str(output),
            ],
            cwd=exporter.parent,
            capture_output=True,
            check=False,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            timeout=120,
        )
        if completed.returncode != 0 or not output.is_file():
            raise ReleaseBridgeError(
                "legacy c380 exporter reconstruction failed "
                f"(stderr_sha256={qualification.sha256_bytes(completed.stderr)})"
            )
        result = output.read_bytes()
    return result, {
        "commit": ATTEMPT_ZERO_LEGACY_COMMIT,
        "closure": closure,
        "reconstructed_source": {
            "bytes": len(result),
            "sha256": qualification.sha256_bytes(result),
            "ascii": True,
        },
    }


def _validate_legacy_attempt_zero_source(
    repository: pathlib.Path, selected: Mapping[str, Any],
) -> dict[str, Any]:
    if selected.get("attempt") != 0 or selected.get("compile_time_macros") != []:
        raise ReleaseBridgeError("legacy exporter route is restricted to attempt zero")
    source = selected.get("generated_source")
    runtime = selected.get("runtime")
    expected_source = challenger.ALLOWLIST["attempt_zero_source"]
    expected_runtime = challenger.ALLOWLIST["attempt_zero_runtime"]
    if (
        not isinstance(source, Mapping) or not isinstance(runtime, Mapping)
        or any(source.get(key) != expected_source[key] for key in ("bytes", "sha256"))
        or any(runtime.get(key) != expected_runtime[key] for key in ("bytes", "sha256"))
    ):
        raise ReleaseBridgeError("attempt-zero source/runtime left the frozen allowlist")
    source_path = _verify_record(source, "legacy attempt-zero source", ascii_required=True)
    runtime_path = _verify_record(runtime, "legacy attempt-zero runtime")
    reconstructed, closure = _legacy_attempt_zero_export(repository, runtime_path)
    if reconstructed != source_path.read_bytes():
        raise ReleaseBridgeError("legacy c380 exporter does not reproduce the finalist")
    runtime_document = qualification.load_sealed(
        runtime_path, challenger.export_model.RUNTIME_SCHEMA
    )
    body_sha256 = runtime_document.get("body_sha256")
    payload_sha256 = runtime_document.get("quantization", {}).get("payload_sha256")
    text = reconstructed.decode("ascii")
    required = (
        "kInputs = 6301;", "kHiddenOne = 12;", "kHiddenTwo = 8;",
        "kOutputs = 1;", "kBootstrapZero = false;",
        f'kRuntimeBodySha256 = "{body_sha256}";',
        f'kPayloadSha256 = "{payload_sha256}";',
    )
    if any(text.count(marker) != 1 for marker in required):
        raise ReleaseBridgeError("legacy finalist does not embed the selected runtime")
    return {
        **closure,
        "frozen_allowlist": {
            "runtime": {key: runtime[key] for key in ("bytes", "sha256")},
            "source": {key: source[key] for key in ("bytes", "sha256")},
        },
        "finalist": dict(selected["selection_evidence"]),
        "embedded_runtime_body_sha256": body_sha256,
        "embedded_payload_sha256": payload_sha256,
        "source_hash_preserved": True,
        "release_preflight_parity_required": True,
    }


def _release_payloads(
    repository: pathlib.Path, selected: Mapping[str, Any]
) -> dict[str, Any]:
    runtime_path = pathlib.Path(selected["runtime"]["path"])
    try:
        model_header, metadata = challenger.export_model.render_header(runtime_path)
        _output, base_source = source_exporter.render(
            model_header=model_header
        )
    except Exception as error:
        raise ReleaseBridgeError("runtime-to-source exporter failed") from error
    if metadata.get("architecture", {}).get("dimensions") != challenger.DIMENSIONS:
        raise ReleaseBridgeError("exporter changed the fixed architecture")
    selected_path = pathlib.Path(selected["generated_source"]["path"])
    legacy = None
    if selected.get("attempt") == 0:
        legacy = _validate_legacy_attempt_zero_source(repository, selected)
        variant = selected_path.read_bytes()
    else:
        variant = _variant_source(base_source, selected["compile_time_macros"])
        if variant != selected_path.read_bytes():
            raise ReleaseBridgeError(
                "selected generated source is not runtime export plus frozen search macros"
            )
    configuration = selected["configuration"]
    profile = configuration["profile"]
    candidate = deployment.derive_source(
        variant,
        search_tuple=configuration["tuple"],
        profile=profile,
        work=deployment.PROFILE_ROSTER[profile],
    )
    maximum = (
        SOURCE_LIMIT_EXCLUSIVE - 1
        if selected.get("attempt") == 0 else SOURCE_MAXIMUM_FOR_TARGET
    )
    if not 0 < len(candidate) <= maximum:
        raise ReleaseBridgeError(
            "deployed source violates its hard/2KB-reserve size contract"
        )
    manifest = deployment.create_manifest(
        variant, candidate,
        search_tuple=configuration["tuple"],
        profile=profile,
        work=deployment.PROFILE_ROSTER[profile],
    )
    manifest_bytes = deployment._canonical_manifest_bytes(manifest)
    tracked_submission = variant if legacy is not None else base_source
    derivation = {
        "schema": SOURCE_DERIVATION_SCHEMA,
        "algorithm": (
            (
                "frozen-c380-legacy-export+"
                if legacy is not None else
                "runtime-export+frozen-search-macro-prefix+"
            )
            + deployment.ALGORITHM
        ),
        "runtime": dict(selected["runtime"]),
        "base_exported_source": {
            "bytes": len(base_source),
            "sha256": qualification.sha256_bytes(base_source),
            "ascii": True,
        },
        "selected_generated_source": {
            key: selected["generated_source"][key]
            for key in ("bytes", "sha256", "ascii")
        },
        "search_throughput_profile": selected[
            "search_throughput_profile"
        ],
        "candidate_search_profile": selected["candidate_search_profile"],
        "search_variant": selected["search_variant"],
        "compile_time_macros": list(selected["compile_time_macros"]),
        "macros_embedded_at_source_start": True,
        "legacy_attempt_zero": legacy,
        "deployment": deployment.attest_derivation(
            variant, candidate,
            search_tuple=configuration["tuple"], profile=profile,
            work=deployment.PROFILE_ROSTER[profile],
        ),
        "configuration": dict(configuration),
        "current_exporter_equality_required": legacy is None,
        "selected_finalist_sha256_preserved": selected["generated_source"][
            "sha256"
        ],
        "release_preflight_embedded_runtime_parity_required": True,
        "only_runtime_payload_search_macros_and_deployment_slots_changed": True,
        "source_limit_exclusive": SOURCE_LIMIT_EXCLUSIVE,
        "source_reserve_target": SOURCE_RESERVE_TARGET,
        "deployed_source_reserve": SOURCE_LIMIT_EXCLUSIVE - len(candidate),
        "reserve_target_required": selected.get("attempt") != 0,
    }
    return {
        "model.hpp": model_header,
        # The legacy attempt-zero source is the output of the release
        # checkout's shared generator.  Replacing it with the newer compacting
        # exporter output makes the maintained submission-current CTest fail
        # and breaks exact finalist promotion.
        "submission.cpp": tracked_submission,
        "discrete_v3_deployment.cpp": candidate,
        "discrete_v3_deployment.json": manifest_bytes,
        "manifest": manifest,
        "derivation": derivation,
    }


def _artifact_records(
    repository: pathlib.Path, payloads: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = {}
    for relative in PROMOTED_RELATIVES:
        path = repository / relative
        record = _record(path, ascii_required=True)
        if payloads is not None:
            expected = payloads[relative.name]
            if path.read_bytes() != expected:
                raise ReleaseBridgeError(
                    f"promoted artifact differs from exact payload: {relative}"
                )
        result[relative.name] = {
            "relative_path": relative.as_posix(),
            "bytes": record["bytes"],
            "sha256": record["sha256"],
            "ascii": True,
        }
    return result


def promote_candidate(
    *, campaign_plan_path: pathlib.Path, attempt: int,
    candidate_runtime: pathlib.Path, candidate_source: pathlib.Path,
    repository: pathlib.Path, output_path: pathlib.Path,
    promoted_at_utc: str,
    selected_validator: SelectedValidator = _selected_candidate,
) -> dict[str, Any]:
    """Write only the four exact release artifacts; never commits or pushes."""

    promoted_at = _utc(promoted_at_utc, "promotion time")
    del promoted_at
    repository = pathlib.Path(repository).resolve()
    selected = dict(selected_validator(
        campaign_plan_path, attempt=attempt,
        candidate_runtime=candidate_runtime,
        candidate_source=candidate_source,
    ))
    payloads = _release_payloads(repository, selected)
    git_before = _git_identity(repository, require_clean=False)
    allowed = {path.as_posix() for path in PROMOTED_RELATIVES}
    dirty_before = set(git_before["dirty_paths"])
    if dirty_before and not dirty_before.issubset(allowed):
        raise ReleaseBridgeError("unrelated worktree changes predate promotion")
    if dirty_before:
        # Resume is allowed only if every already-dirty target is already exact.
        for relative in PROMOTED_RELATIVES:
            if relative.as_posix() in dirty_before:
                target = repository / relative
                if not target.is_file() or target.read_bytes() != payloads[relative.name]:
                    raise ReleaseBridgeError("conflicting release change predates promotion")
    for relative in PROMOTED_RELATIVES:
        _atomic_replace(repository / relative, payloads[relative.name])
    artifacts = _artifact_records(repository, payloads)
    dirty_after = _git_status_paths(repository)
    if not dirty_after:
        # A freshly frozen governance campaign may resume after the exact four
        # artifacts were already committed by an earlier superseded receipt.
        # Accept only byte-identical tracked HEAD content; this is not a route
        # for silently broadening the promotion diff.
        for relative in PROMOTED_RELATIVES:
            route = relative.as_posix()
            _git(repository, "ls-files", "--error-unmatch", "--", route)
            if _git(repository, "show", f"HEAD:{route}") != payloads[relative.name]:
                raise ReleaseBridgeError(
                    "clean resumed promotion is not already committed exactly"
                )
    elif not dirty_after.issubset(allowed):
        raise ReleaseBridgeError("promotion changed files outside the fixed release roster")
    output_path = _safe_output(output_path)
    body = {
        "schema": PROMOTION_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID,
        "attempt": attempt,
        "status": "exact-release-artifacts-promoted-awaiting-commit",
        "promoted_at_utc": promoted_at_utc,
        "campaign_plan": _reference(campaign_plan_path, challenger.PLAN_SCHEMA),
        "selected_candidate": selected,
        "derivation": payloads["derivation"],
        "frozen_execution_sources": selected["frozen_execution_sources"],
        "configuration": dict(selected["configuration"]),
        "repository": str(repository),
        "branch": RELEASE_BRANCH,
        "pre_promotion_head": git_before["commit"],
        "tracked_artifacts": artifacts,
        "dirty_paths_after": sorted(dirty_after),
        "allowed_changed_paths": sorted(allowed),
        "commit_performed": False,
        "push_performed": False,
        "ci_started": False,
        "uploads": 0,
    }
    expected = qualification.seal(body)
    if output_path.exists():
        existing = qualification.load_sealed(output_path, PROMOTION_SCHEMA)
        if existing != expected:
            raise ReleaseBridgeError("existing promotion receipt changed")
        return existing
    return qualification.write_sealed(output_path, body)


def _verify_promoted_commit(
    repository: pathlib.Path, *, payloads: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    git = _git_identity(repository, require_clean=True)
    for relative in PROMOTED_RELATIVES:
        route = relative.as_posix()
        _git(repository, "ls-files", "--error-unmatch", "--", route)
        committed = _git(repository, "show", f"{git['commit']}:{route}")
        if committed != payloads[relative.name]:
            raise ReleaseBridgeError(f"committed release artifact changed: {route}")
    return git, _artifact_records(repository, payloads)


def validate_promotion(
    path: pathlib.Path, *, campaign_plan_path: pathlib.Path, attempt: int,
    candidate_runtime: pathlib.Path, candidate_source: pathlib.Path,
    repository: pathlib.Path,
    selected_validator: SelectedValidator = _selected_candidate,
) -> dict[str, Any]:
    value = qualification.load_sealed(path, PROMOTION_SCHEMA)
    selected = dict(selected_validator(
        campaign_plan_path, attempt=attempt,
        candidate_runtime=candidate_runtime, candidate_source=candidate_source,
    ))
    payloads = _release_payloads(repository, selected)
    artifacts = _artifact_records(repository, payloads)
    if (
        path.is_symlink()
        or value.get("namespace") != NAMESPACE
        or value.get("campaign_id") != CAMPAIGN_ID
        or value.get("attempt") != attempt
        or value.get("status")
        != "exact-release-artifacts-promoted-awaiting-commit"
        or value.get("campaign_plan")
        != _reference(campaign_plan_path, challenger.PLAN_SCHEMA)
        or value.get("selected_candidate") != selected
        or value.get("derivation") != payloads["derivation"]
        or value.get("frozen_execution_sources")
        != selected.get("frozen_execution_sources")
        or value.get("configuration") != selected["configuration"]
        or value.get("repository") != str(pathlib.Path(repository).resolve())
        or value.get("branch") != RELEASE_BRANCH
        or qualification.COMMIT_RE.fullmatch(
            str(value.get("pre_promotion_head"))
        ) is None
        or value.get("tracked_artifacts") != artifacts
        or not set(value.get("dirty_paths_after", [])).issubset(
            {item.as_posix() for item in PROMOTED_RELATIVES}
        )
        or value.get("allowed_changed_paths")
        != sorted(item.as_posix() for item in PROMOTED_RELATIVES)
        or value.get("commit_performed") is not False
        or value.get("push_performed") is not False
        or value.get("ci_started") is not False
        or value.get("uploads") != 0
    ):
        raise ReleaseBridgeError("promotion receipt changed")
    _utc(value.get("promoted_at_utc"), "promotion time")
    return value


def _validate_base_preflight(
    path: pathlib.Path, *, repository: pathlib.Path,
    runtime_path: pathlib.Path, base_source: pathlib.Path,
) -> dict[str, Any]:
    if (
        path.is_symlink() or not path.is_file()
        or path.stem != qualification.sha256_file(path)
    ):
        raise ReleaseBridgeError("maintained preflight is not content-addressed")
    receipt = qualification.load_sealed(path, maintained.RECEIPT_SCHEMA)
    try:
        maintained.validate_preflight_receipt(
            receipt, claim=receipt["claim"], plan=receipt["plan"],
            inputs=receipt["inputs_before"],
        )
    except Exception as error:
        raise ReleaseBridgeError("maintained compact preflight failed deep validation") from error
    inputs = receipt["inputs_before"]
    if (
        inputs.get("candidate_commit")
        != _git_identity(repository, require_clean=True)["commit"]
        or inputs.get("candidate", {}).get("sha256")
        != qualification.sha256_file(base_source)
        or inputs.get("candidate", {}).get("bytes") != base_source.stat().st_size
        or inputs.get("runtime", {}).get("sha256")
        != qualification.sha256_file(runtime_path)
        or inputs.get("runtime", {}).get("bytes") != runtime_path.stat().st_size
        or receipt.get("protected_banks_accessed") != []
        or receipt.get("git_writes") != 0
        or receipt.get("uploads") != 0
    ):
        raise ReleaseBridgeError("maintained preflight is not release source/runtime bound")
    _uncontended_timing(receipt.get("timing", {}))
    return receipt


def _uncontended_timing(timing: Mapping[str, Any]) -> dict[str, Any]:
    try:
        maintained.validate_timing_receipt(timing)
    except Exception as error:
        raise ReleaseBridgeError("preflight timing receipt failed validation") from error
    samples = [
        row for row in timing.get("samples", [])
        if isinstance(row, Mapping) and row.get("process_count") == 1
    ]
    if (
        len(samples) != 2
        or {row.get("color") for row in samples} != {0, 1}
        or {row.get("replica") for row in samples} != {0}
    ):
        raise ReleaseBridgeError(
            "preflight lacks exactly one uncontended sample for both colors"
        )
    return {
        "workers": 1,
        "colors": [0, 1],
        "first_max_ms": max(float(row["first_ms"]) for row in samples),
        "later_max_ms": max(float(row["later_max_ms"]) for row in samples),
        "first_limit_exclusive_ms": maintained.FIRST_LIMIT_MS,
        "later_limit_exclusive_ms": maintained.LATER_LIMIT_MS,
    }


def _source_inputs(repository: pathlib.Path) -> dict[str, pathlib.Path]:
    bot = repository / BOT_RELATIVE
    return {
        "rank4": repository / RANK4_RELATIVE,
        "gate_source": bot / "rank4_gate.cpp",
        "protocol_smoke": repository
        / "submissions/codingame/tools/protocol_smoke_test.mjs",
        "export_model": bot / "export_model.py",
        "feature_parity": bot / "feature_parity.py",
        "submission_test": bot / "submission_test.cpp",
        "timing_probe": bot / "timing_probe.cpp",
        "inference_probe": bot / "inference_probe.cpp",
    }


def _preflight_tool_closure() -> dict[str, Any]:
    return {
        "release_bridge": _record(pathlib.Path(__file__).resolve()),
        "release_tests": _record(TEST_PATH),
        "qualification": _record(pathlib.Path(qualification.__file__).resolve()),
        "maintained_preflight": _record(pathlib.Path(maintained.__file__).resolve()),
        "deployment": _record(pathlib.Path(deployment.__file__).resolve()),
        "deployment_preflight_support": _record(
            pathlib.Path(deployment_preflight.__file__).resolve()
        ),
        "upload": _record(pathlib.Path(upload.__file__).resolve()),
        "challenger": _record(pathlib.Path(challenger.__file__).resolve()),
    }


def _preflight_snapshot(
    *, campaign_plan_path: pathlib.Path, attempt: int,
    candidate_runtime: pathlib.Path, candidate_source: pathlib.Path,
    repository: pathlib.Path, promotion_path: pathlib.Path,
    base_preflight_path: pathlib.Path, python_path: pathlib.Path,
    gcc_path: pathlib.Path, clang_path: pathlib.Path, node_path: pathlib.Path,
    selected_validator: SelectedValidator = _selected_candidate,
) -> dict[str, Any]:
    repository = pathlib.Path(repository).resolve()
    selected = dict(selected_validator(
        campaign_plan_path, attempt=attempt,
        candidate_runtime=candidate_runtime, candidate_source=candidate_source,
    ))
    heavy_stage_lock = None
    if selected.get("frozen_execution_sources") is not None:
        campaign_context = challenger.validate_campaign(
            campaign_plan_path.resolve()
        )
        heavy_stage_lock = str(
            pathlib.Path(campaign_context["plan"]["outputs"]["root"]).resolve()
            / ".rank4-teacher-heavy-stage.lock"
        )
    payloads = _release_payloads(repository, selected)
    validate_promotion(
        promotion_path, campaign_plan_path=campaign_plan_path, attempt=attempt,
        candidate_runtime=candidate_runtime, candidate_source=candidate_source,
        repository=repository, selected_validator=selected_validator,
    )
    git, artifacts = _verify_promoted_commit(repository, payloads=payloads)
    base_source = repository / GENERATED_RELATIVE
    runtime_path = pathlib.Path(selected["runtime"]["path"])
    base_receipt = _validate_base_preflight(
        base_preflight_path, repository=repository, runtime_path=runtime_path,
        base_source=base_source,
    )
    candidate_path = repository / DEPLOYED_RELATIVE
    manifest_path = repository / MANIFEST_RELATIVE
    manifest = deployment.verify_manifest_file(manifest_path, candidate_path)
    if (
        manifest.get("configuration") != selected["configuration"]
        or manifest.get("base_source")
        != payloads["derivation"]["selected_generated_source"]
        or manifest.get("deployed_source", {}).get("sha256")
        != qualification.sha256_file(candidate_path)
    ):
        raise ReleaseBridgeError("tracked deployment manifest changed")
    sources = {
        name: _record(path, ascii_required=True)
        for name, path in _source_inputs(repository).items()
    }
    if (
        sources["rank4"]["sha256"] != qualification.RANK4_SHA256
        or sources["rank4"]["bytes"] != qualification.RANK4_BYTES
    ):
        raise ReleaseBridgeError("maintained Rank-4 source identity changed")
    tools = {
        "python": deployment_preflight._record(
            python_path, executable=True, allow_symlink=True
        ),
        "gcc": deployment_preflight._compiler_record(gcc_path, "GNU"),
        "clang": deployment_preflight._compiler_record(clang_path, "Clang"),
        "node": deployment_preflight._record(
            node_path, executable=True, allow_symlink=True
        ),
    }
    return {
        "campaign_plan": _reference(campaign_plan_path, challenger.PLAN_SCHEMA),
        "attempt": attempt,
        "selected_candidate": selected,
        "production_allowlist_enforced": (
            selected.get("frozen_execution_sources") is not None
        ),
        "frozen_execution_sources": selected.get("frozen_execution_sources"),
        "heavy_stage_lock": heavy_stage_lock,
        "promotion": _reference(promotion_path, PROMOTION_SCHEMA),
        "base_preflight": _reference(
            base_preflight_path, maintained.RECEIPT_SCHEMA
        ),
        "base_preflight_body_sha256": base_receipt["body_sha256"],
        "generated_source": _record(base_source, ascii_required=True),
        "candidate": _record(candidate_path, ascii_required=True),
        "runtime": dict(selected["runtime"]),
        "repository": str(repository),
        "git": git,
        "tracked_artifacts": artifacts,
        "manifest": _record(manifest_path, ascii_required=True),
        "manifest_body_sha256": manifest["body_sha256"],
        "derivation": payloads["derivation"],
        "configuration": dict(selected["configuration"]),
        "sources": sources,
        "tools": tools,
        "tool_closure": _preflight_tool_closure(),
    }


def _preflight_plan_body(
    inputs: Mapping[str, Any], *, root: pathlib.Path, planned_at_utc: str,
) -> dict[str, Any]:
    build = root / "build"
    return {
        "schema": PREFLIGHT_PLAN_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID,
        "status": "release-preflight-planned-unclaimed",
        "planned_at_utc": planned_at_utc,
        "inputs": dict(inputs),
        "paths": {
            "root": str(root),
            "build": str(build),
            "claim": str(root / "claim.json"),
            "receipts": str(root / "receipts"),
            "reference": str(root / "reference.json"),
        },
        "commands": deployment_preflight._commands(inputs, build),
        "configuration_probe_source": {
            "bytes": len(deployment_preflight._configuration_probe_source()),
            "sha256": qualification.sha256_bytes(
                deployment_preflight._configuration_probe_source()
            ),
        },
        "thresholds": {
            "source_bytes_exclusive": challenger.SOURCE_LIMIT,
            "parity_states_minimum": maintained.PARITY_STATES,
            "inference_error_maximum": maintained.PARITY_MAX_ERROR,
            "process_counts": list(maintained.PROCESS_COUNTS),
            "first_ms_exclusive": maintained.FIRST_LIMIT_MS,
            "later_ms_exclusive": maintained.LATER_LIMIT_MS,
            "uncontended_workers": 1,
            "uncontended_colors": [0, 1],
        },
        "policy": {
            "protected_banks_accessed": False,
            "git_writes": 0,
            "uploads": 0,
            "release_branch": RELEASE_BRANCH,
        },
    }


def prepare_preflight(
    *, campaign_plan_path: pathlib.Path, attempt: int,
    candidate_runtime: pathlib.Path, candidate_source: pathlib.Path,
    repository: pathlib.Path, promotion_path: pathlib.Path,
    base_preflight_path: pathlib.Path, output_root: pathlib.Path,
    python_path: pathlib.Path, gcc_path: pathlib.Path,
    clang_path: pathlib.Path, node_path: pathlib.Path,
    planned_at_utc: str,
    selected_validator: SelectedValidator = _selected_candidate,
    allow_injected_test_evidence: bool = False,
) -> pathlib.Path:
    _utc(planned_at_utc, "release preflight plan time")
    context = _guard_release_hooks(
        campaign_plan_path,
        hooks_used=selected_validator is not _selected_candidate,
        allow_injected_test_evidence=allow_injected_test_evidence,
    )
    if (
        context is not None
        and context.get("inputs", {}).get("production_allowlist_enforced") is True
        and output_root.resolve()
        != _production_release_root(context, attempt=attempt)
    ):
        raise ReleaseBridgeError(
            "production release-preflight root is not campaign-derived"
        )
    root = _safe_directory(output_root / PREFLIGHT_DIRECTORY, create=True)
    plan_path = _safe_output(root / "plan.json")
    inputs = _preflight_snapshot(
        campaign_plan_path=campaign_plan_path, attempt=attempt,
        candidate_runtime=candidate_runtime, candidate_source=candidate_source,
        repository=repository, promotion_path=promotion_path,
        base_preflight_path=base_preflight_path, python_path=python_path,
        gcc_path=gcc_path, clang_path=clang_path, node_path=node_path,
        selected_validator=selected_validator,
    )
    body = _preflight_plan_body(
        inputs, root=root, planned_at_utc=planned_at_utc
    )
    expected = qualification.seal(body)
    if plan_path.exists():
        if qualification.load_sealed(plan_path, PREFLIGHT_PLAN_SCHEMA) != expected:
            raise ReleaseBridgeError("existing release-preflight plan changed")
    else:
        if any((root / name).exists() for name in (
            "claim.json", "receipts", "reference.json", "build"
        )):
            raise ReleaseBridgeError("preflight output predates its plan")
        qualification.write_sealed(plan_path, body)
    validate_preflight_plan(
        plan_path, campaign_plan_path=campaign_plan_path, attempt=attempt,
        candidate_runtime=candidate_runtime, candidate_source=candidate_source,
        repository=repository, promotion_path=promotion_path,
        base_preflight_path=base_preflight_path,
        selected_validator=selected_validator,
    )
    return plan_path


def validate_preflight_plan(
    path: pathlib.Path, *, campaign_plan_path: pathlib.Path, attempt: int,
    candidate_runtime: pathlib.Path, candidate_source: pathlib.Path,
    repository: pathlib.Path, promotion_path: pathlib.Path,
    base_preflight_path: pathlib.Path,
    selected_validator: SelectedValidator = _selected_candidate,
) -> dict[str, Any]:
    plan = qualification.load_sealed(path, PREFLIGHT_PLAN_SCHEMA)
    root = path.parent.resolve()
    if path.is_symlink() or path.resolve() != root / "plan.json":
        raise ReleaseBridgeError("release-preflight plan route changed")
    if plan.get("inputs", {}).get("production_allowlist_enforced") is True:
        campaign = challenger.validate_campaign(campaign_plan_path.resolve())
        expected_root = (
            _production_release_root(campaign, attempt=attempt)
            / PREFLIGHT_DIRECTORY
        )
        if root != expected_root:
            raise ReleaseBridgeError(
                "production release-preflight plan root changed"
            )
    tools = plan.get("inputs", {}).get("tools", {})
    if not isinstance(tools, Mapping):
        raise ReleaseBridgeError("release-preflight tool bindings are absent")
    inputs = _preflight_snapshot(
        campaign_plan_path=campaign_plan_path, attempt=attempt,
        candidate_runtime=candidate_runtime, candidate_source=candidate_source,
        repository=repository, promotion_path=promotion_path,
        base_preflight_path=base_preflight_path,
        python_path=pathlib.Path(str(tools.get("python", {}).get("path", ""))),
        gcc_path=pathlib.Path(str(tools.get("gcc", {}).get("path", ""))),
        clang_path=pathlib.Path(str(tools.get("clang", {}).get("path", ""))),
        node_path=pathlib.Path(str(tools.get("node", {}).get("path", ""))),
        selected_validator=selected_validator,
    )
    expected = qualification.seal(_preflight_plan_body(
        inputs, root=root, planned_at_utc=str(plan.get("planned_at_utc", ""))
    ))
    if plan != expected:
        raise ReleaseBridgeError("release-preflight plan changed")
    _utc(plan.get("planned_at_utc"), "release preflight plan time")
    return plan


CommandRunner = Callable[..., tuple[Mapping[str, Any], bytes, bytes]]
ParityRunner = Callable[..., Mapping[str, Any]]
TimingRunner = Callable[[pathlib.Path], Mapping[str, Any]]


PREFLIGHT_CHECKS = {
    "maintained_base_preflight_deep",
    "runtime_export_and_search_macros_exact",
    "exact_seven_slot_deployment",
    "clean_committed_release_branch",
    "four_tracked_release_artifacts",
    "clang_candidate",
    "gcc_clang_native_and_sanitized",
    "source_specific_rank4_gate",
    "protocol_both_colors",
    "compiled_configuration_exact",
    "feature_inference_parity_4096",
    "timing_1_2_10_both_colors",
    "uncontended_one_worker_both_colors",
    "inputs_unchanged",
}


def _copy_release_harness(plan: Mapping[str, Any]) -> dict[str, Any]:
    build = pathlib.Path(plan["paths"]["build"])
    harness = _safe_directory(build / "harness", create=True)
    _safe_directory(build / "binaries", create=True)
    inputs = plan["inputs"]
    copies = {
        "submission.cpp": pathlib.Path(inputs["candidate"]["path"]),
        "submission_test.cpp": pathlib.Path(
            inputs["sources"]["submission_test"]["path"]
        ),
        "timing_probe.cpp": pathlib.Path(
            inputs["sources"]["timing_probe"]["path"]
        ),
        "inference_probe.cpp": pathlib.Path(
            inputs["sources"]["inference_probe"]["path"]
        ),
    }
    for name, source in copies.items():
        destination = harness / name
        if destination.exists() or destination.is_symlink():
            raise ReleaseBridgeError("release-preflight harness was not pristine")
        shutil.copyfile(source, destination)
    probe = harness / "configuration_probe.cpp"
    probe.write_bytes(deployment_preflight._configuration_probe_source())
    records = {
        path.name: _record(path, ascii_required=True)
        for path in sorted(harness.iterdir())
    }
    expected = {
        "submission.cpp": inputs["candidate"],
        "submission_test.cpp": inputs["sources"]["submission_test"],
        "timing_probe.cpp": inputs["sources"]["timing_probe"],
        "inference_probe.cpp": inputs["sources"]["inference_probe"],
    }
    for name, source in expected.items():
        if any(records[name][key] != source[key] for key in ("bytes", "sha256", "ascii")):
            raise ReleaseBridgeError(f"release-preflight harness copy changed: {name}")
    return records


@_release_heavy_stage
def run_preflight(
    plan_path: pathlib.Path, *, campaign_plan_path: pathlib.Path,
    attempt: int, candidate_runtime: pathlib.Path,
    candidate_source: pathlib.Path, repository: pathlib.Path,
    promotion_path: pathlib.Path, base_preflight_path: pathlib.Path,
    claimed_at_utc: str,
    selected_validator: SelectedValidator = _selected_candidate,
    command_runner: CommandRunner = maintained.run_command,
    parity_runner: ParityRunner = maintained.run_inference_parity,
    timing_runner: TimingRunner = maintained.run_timing_suite,
    allow_injected_test_evidence: bool = False,
) -> pathlib.Path:
    _guard_release_hooks(
        campaign_plan_path,
        hooks_used=(
            selected_validator is not _selected_candidate
            or command_runner is not maintained.run_command
            or parity_runner is not maintained.run_inference_parity
            or timing_runner is not maintained.run_timing_suite
        ),
        allow_injected_test_evidence=allow_injected_test_evidence,
    )
    plan = validate_preflight_plan(
        plan_path, campaign_plan_path=campaign_plan_path, attempt=attempt,
        candidate_runtime=candidate_runtime, candidate_source=candidate_source,
        repository=repository, promotion_path=promotion_path,
        base_preflight_path=base_preflight_path,
        selected_validator=selected_validator,
    )
    root = plan_path.parent.resolve()
    reference_path = _safe_output(root / "reference.json")
    if reference_path.exists():
        validate_preflight_reference(
            reference_path, campaign_plan_path=campaign_plan_path,
            attempt=attempt, candidate_runtime=candidate_runtime,
            candidate_source=candidate_source, repository=repository,
            promotion_path=promotion_path,
            base_preflight_path=base_preflight_path,
            selected_validator=selected_validator,
        )
        return reference_path
    claim_path = _safe_output(root / "claim.json")
    receipts = root / "receipts"
    if claim_path.exists() or (receipts.exists() and any(receipts.iterdir())):
        raise ReleaseBridgeError("release-preflight claim is spent")
    if pathlib.Path(plan["paths"]["build"]).exists():
        raise ReleaseBridgeError("release-preflight build root is not fresh")
    _utc(claimed_at_utc, "release preflight claim time")
    qualification.write_sealed(claim_path, {
        "schema": PREFLIGHT_CLAIM_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID,
        "attempt": attempt,
        "status": "release-preflight-claimed-before-execution",
        "claimed_at_utc": claimed_at_utc,
        "plan": _reference(plan_path, PREFLIGHT_PLAN_SCHEMA),
        "candidate_commit": plan["inputs"]["git"]["commit"],
        "candidate_sha256": plan["inputs"]["candidate"]["sha256"],
        "frozen_execution_sources": plan["inputs"][
            "frozen_execution_sources"
        ],
        "heavy_stage_lock": plan["inputs"]["heavy_stage_lock"],
        "one_shot": True,
    })
    harness = _copy_release_harness(plan)
    commands, outputs = deployment_preflight._run_commands(
        plan, command_runner=command_runner
    )
    binaries = deployment_preflight._binary_records(plan)
    configuration_probe = deployment_preflight._parse_configuration(
        outputs["configuration_probe"], plan["inputs"]["configuration"]
    )
    parity = dict(parity_runner(
        repository=pathlib.Path(plan["inputs"]["repository"]),
        runtime_path=pathlib.Path(plan["inputs"]["runtime"]["path"]),
        probe_path=pathlib.Path(binaries["inference_probe"]["path"]),
        states=maintained.PARITY_STATES,
    ))
    timing = dict(timing_runner(pathlib.Path(binaries["timing_probe"]["path"])))
    maintained.validate_parity_receipt(parity)
    maintained.validate_timing_receipt(timing)
    uncontended = _uncontended_timing(timing)
    tools = plan["inputs"]["tools"]
    after = _preflight_snapshot(
        campaign_plan_path=campaign_plan_path, attempt=attempt,
        candidate_runtime=candidate_runtime, candidate_source=candidate_source,
        repository=repository, promotion_path=promotion_path,
        base_preflight_path=base_preflight_path,
        python_path=pathlib.Path(tools["python"]["path"]),
        gcc_path=pathlib.Path(tools["gcc"]["path"]),
        clang_path=pathlib.Path(tools["clang"]["path"]),
        node_path=pathlib.Path(tools["node"]["path"]),
        selected_validator=selected_validator,
    )
    if after != plan["inputs"]:
        raise ReleaseBridgeError("release-preflight inputs changed during execution")
    body = {
        "schema": PREFLIGHT_RECEIPT_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID,
        "attempt": attempt,
        "status": "release-preflight-passed",
        "completed_at_utc": utc_now(),
        "plan": _reference(plan_path, PREFLIGHT_PLAN_SCHEMA),
        "claim": _reference(claim_path, PREFLIGHT_CLAIM_SCHEMA),
        "inputs": plan["inputs"],
        "commands": commands,
        "harness": harness,
        "binaries": binaries,
        "configuration_probe": configuration_probe,
        "parity": parity,
        "timing": timing,
        "uncontended_timing": uncontended,
        "frozen_execution_sources": plan["inputs"][
            "frozen_execution_sources"
        ],
        "heavy_stage_lock": plan["inputs"]["heavy_stage_lock"],
        "checks": {name: "passed" for name in sorted(PREFLIGHT_CHECKS)},
        "protected_banks_accessed": [],
        "git_writes": 0,
        "uploads": 0,
    }
    sealed = qualification.seal(body)
    raw = qualification.canonical_json_bytes(sealed)
    receipt_path = _safe_output(
        receipts / f"{qualification.sha256_bytes(raw)}.json"
    )
    qualification.atomic_write_once(receipt_path, raw)
    qualification.write_sealed(reference_path, {
        "schema": PREFLIGHT_REFERENCE_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID,
        "attempt": attempt,
        "status": "release-preflight-passed-awaiting-ci",
        "plan": _reference(plan_path, PREFLIGHT_PLAN_SCHEMA),
        "receipt": _reference(receipt_path, PREFLIGHT_RECEIPT_SCHEMA),
        "candidate_commit": plan["inputs"]["git"]["commit"],
        "candidate": plan["inputs"]["candidate"],
        "runtime": plan["inputs"]["runtime"],
        "derivation": plan["inputs"]["derivation"],
        "configuration": plan["inputs"]["configuration"],
        "gate": binaries["rank4_gate"],
        "upload_authorized": False,
    })
    validate_preflight_reference(
        reference_path, campaign_plan_path=campaign_plan_path, attempt=attempt,
        candidate_runtime=candidate_runtime, candidate_source=candidate_source,
        repository=repository, promotion_path=promotion_path,
        base_preflight_path=base_preflight_path,
        selected_validator=selected_validator,
    )
    return reference_path


def validate_preflight_reference(
    path: pathlib.Path, *, campaign_plan_path: pathlib.Path, attempt: int,
    candidate_runtime: pathlib.Path, candidate_source: pathlib.Path,
    repository: pathlib.Path, promotion_path: pathlib.Path,
    base_preflight_path: pathlib.Path,
    selected_validator: SelectedValidator = _selected_candidate,
) -> dict[str, Any]:
    reference = qualification.load_sealed(path, PREFLIGHT_REFERENCE_SCHEMA)
    plan_path = _verify_reference(
        reference.get("plan"), PREFLIGHT_PLAN_SCHEMA, "release-preflight plan"
    )
    plan = validate_preflight_plan(
        plan_path, campaign_plan_path=campaign_plan_path, attempt=attempt,
        candidate_runtime=candidate_runtime, candidate_source=candidate_source,
        repository=repository, promotion_path=promotion_path,
        base_preflight_path=base_preflight_path,
        selected_validator=selected_validator,
    )
    if path.is_symlink() or path.resolve() != pathlib.Path(
        plan["paths"]["reference"]
    ).resolve():
        raise ReleaseBridgeError("release-preflight reference route changed")
    receipt_path = _verify_reference(
        reference.get("receipt"), PREFLIGHT_RECEIPT_SCHEMA,
        "release-preflight receipt",
    )
    if receipt_path.stem != qualification.sha256_file(receipt_path):
        raise ReleaseBridgeError("release-preflight receipt is not content-addressed")
    receipt = qualification.load_sealed(
        receipt_path, PREFLIGHT_RECEIPT_SCHEMA
    )
    claim_path = _verify_reference(
        receipt.get("claim"), PREFLIGHT_CLAIM_SCHEMA,
        "release-preflight claim",
    )
    claim = qualification.load_sealed(claim_path, PREFLIGHT_CLAIM_SCHEMA)
    if (
        set(claim) != {
            "schema", "namespace", "campaign_id", "attempt", "status",
            "claimed_at_utc", "plan", "candidate_commit",
            "candidate_sha256", "frozen_execution_sources", "heavy_stage_lock",
            "one_shot",
            "body_sha256",
        }
        or set(receipt) != {
            "schema", "namespace", "campaign_id", "attempt", "status",
            "completed_at_utc", "plan", "claim", "inputs", "commands",
            "harness", "binaries", "configuration_probe", "parity",
            "timing", "uncontended_timing", "checks",
            "frozen_execution_sources", "heavy_stage_lock",
            "protected_banks_accessed", "git_writes", "uploads",
            "body_sha256",
        }
        or claim.get("namespace") != NAMESPACE
        or claim.get("campaign_id") != CAMPAIGN_ID
        or claim.get("status") != "release-preflight-claimed-before-execution"
        or claim.get("attempt") != attempt
        or claim.get("plan") != _reference(plan_path, PREFLIGHT_PLAN_SCHEMA)
        or claim.get("candidate_commit") != plan["inputs"]["git"]["commit"]
        or claim.get("candidate_sha256") != plan["inputs"]["candidate"]["sha256"]
        or claim.get("frozen_execution_sources")
        != plan["inputs"].get("frozen_execution_sources")
        or claim.get("heavy_stage_lock") != plan["inputs"].get("heavy_stage_lock")
        or claim.get("one_shot") is not True
    ):
        raise ReleaseBridgeError("release-preflight claim changed")
    _utc(claim.get("claimed_at_utc"), "release preflight claim time")
    if (
        receipt.get("status") != "release-preflight-passed"
        or receipt.get("namespace") != NAMESPACE
        or receipt.get("campaign_id") != CAMPAIGN_ID
        or receipt.get("attempt") != attempt
        or receipt.get("plan") != _reference(plan_path, PREFLIGHT_PLAN_SCHEMA)
        or receipt.get("inputs") != plan["inputs"]
        or receipt.get("protected_banks_accessed") != []
        or receipt.get("git_writes") != 0
        or receipt.get("uploads") != 0
        or receipt.get("frozen_execution_sources")
        != plan["inputs"].get("frozen_execution_sources")
        or receipt.get("heavy_stage_lock")
        != plan["inputs"].get("heavy_stage_lock")
        or receipt.get("checks")
        != {name: "passed" for name in sorted(PREFLIGHT_CHECKS)}
    ):
        raise ReleaseBridgeError("release-preflight receipt changed")
    _utc(receipt.get("completed_at_utc"), "release preflight completion time")
    commands = receipt.get("commands")
    if not isinstance(commands, Mapping) or set(commands) != set(
        deployment_preflight.COMMAND_NAMES
    ):
        raise ReleaseBridgeError("release-preflight command roster changed")
    markers = {
        "native_test": ("compact_value_bfm submission tests passed",),
        "gcc_native_test": ("compact_value_bfm submission tests passed",),
        "sanitized_test": ("compact_value_bfm submission tests passed",),
        "protocol": ("Player 0 and Player 1 protocol smoke tests passed",),
        "rank4_gate_self_test": ("compact_value_bfm Rank-4 gate self-test passed",),
    }
    for name in deployment_preflight.COMMAND_NAMES:
        maintained.validate_command_receipt(
            commands[name], label=name, argv=plan["commands"][name],
            required_markers=markers.get(name, ()),
        )
        if commands[name].get("cwd") != plan["inputs"]["repository"]:
            raise ReleaseBridgeError(
                f"release-preflight command changed directory: {name}"
            )
    harness = receipt.get("harness")
    expected_harness = {
        "submission.cpp": plan["inputs"]["candidate"],
        "submission_test.cpp": plan["inputs"]["sources"]["submission_test"],
        "timing_probe.cpp": plan["inputs"]["sources"]["timing_probe"],
        "inference_probe.cpp": plan["inputs"]["sources"]["inference_probe"],
    }
    if not isinstance(harness, Mapping) or set(harness) != {
        *expected_harness, "configuration_probe.cpp"
    }:
        raise ReleaseBridgeError("release-preflight harness roster changed")
    for name, source in expected_harness.items():
        _verify_record(harness[name], f"release-preflight harness {name}", ascii_required=True)
        if any(harness[name].get(key) != source.get(key) for key in ("bytes", "sha256", "ascii")):
            raise ReleaseBridgeError(f"release-preflight harness binding changed: {name}")
    _verify_record(
        harness["configuration_probe.cpp"],
        "release-preflight configuration harness", ascii_required=True,
    )
    if any(
        harness["configuration_probe.cpp"].get(key)
        != plan["configuration_probe_source"].get(key)
        for key in ("bytes", "sha256")
    ):
        raise ReleaseBridgeError("release-preflight configuration harness changed")
    binaries = receipt.get("binaries")
    if not isinstance(binaries, Mapping) or set(binaries) != set(
        deployment_preflight.BINARY_NAMES
    ):
        raise ReleaseBridgeError("release-preflight binary roster changed")
    for name in deployment_preflight.BINARY_NAMES:
        _verify_record(
            binaries[name], f"release-preflight binary {name}", executable=True
        )
    configuration_check = subprocess.run(
        [str(binaries["configuration_probe"]["path"])],
        capture_output=True, check=False, timeout=30,
    )
    if configuration_check.returncode != 0:
        raise ReleaseBridgeError("release-preflight configuration recheck failed")
    expected_configuration = deployment_preflight._parse_configuration(
        configuration_check.stdout, plan["inputs"]["configuration"]
    )
    parity = receipt.get("parity", {})
    timing = receipt.get("timing", {})
    maintained.validate_parity_receipt(parity)
    maintained.validate_timing_receipt(timing)
    uncontended = _uncontended_timing(timing)
    macro = (
        f'-DCOMPACT_VALUE_BFM_CANDIDATE_SOURCE="'
        f'{plan["inputs"]["candidate"]["path"]}"'
    )
    compile_gate = commands.get("compile_rank4_gate", {})
    if (
        expected_configuration != receipt.get("configuration_probe")
        or receipt.get("uncontended_timing") != uncontended
        or parity.get("runtime_sha256") != plan["inputs"]["runtime"]["sha256"]
        or parity.get("probe_sha256") != binaries["inference_probe"]["sha256"]
        or timing.get("probe_sha256") != binaries["timing_probe"]["sha256"]
        or macro not in compile_gate.get("argv", [])
    ):
        raise ReleaseBridgeError("release-preflight source/parity/timing binding changed")
    expected_reference = qualification.seal({
        "schema": PREFLIGHT_REFERENCE_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID,
        "attempt": attempt,
        "status": "release-preflight-passed-awaiting-ci",
        "plan": _reference(plan_path, PREFLIGHT_PLAN_SCHEMA),
        "receipt": _reference(receipt_path, PREFLIGHT_RECEIPT_SCHEMA),
        "candidate_commit": plan["inputs"]["git"]["commit"],
        "candidate": plan["inputs"]["candidate"],
        "runtime": plan["inputs"]["runtime"],
        "derivation": plan["inputs"]["derivation"],
        "configuration": plan["inputs"]["configuration"],
        "gate": binaries["rank4_gate"],
        "upload_authorized": False,
    })
    if reference != expected_reference:
        raise ReleaseBridgeError("release-preflight reference changed")
    return {
        "reference": reference,
        "reference_path": path.resolve(),
        "plan": plan,
        "plan_path": plan_path,
        "receipt": receipt,
        "receipt_path": receipt_path,
        "base_preflight_path": pathlib.Path(
            plan["inputs"]["base_preflight"]["path"]
        ).resolve(),
        "candidate_commit": plan["inputs"]["git"]["commit"],
        "candidate": plan["inputs"]["candidate"],
        "runtime": plan["inputs"]["runtime"],
        "derivation": plan["inputs"]["derivation"],
        "configuration": plan["inputs"]["configuration"],
        "gate": binaries["rank4_gate"],
        "compile_command": commands["compile_rank4_gate"],
        "compiler": plan["inputs"]["tools"]["clang"],
        "timing": timing,
        "uncontended_timing": uncontended,
        "repository": plan["inputs"]["repository"],
    }


def seal_ci_evidence(
    output_path: pathlib.Path, *, gh_payload: Mapping[str, Any],
    expected_head: str, fetched_at_utc: str,
) -> dict[str, Any]:
    return upload.seal_ci_evidence(
        _safe_output(output_path), gh_payload=gh_payload,
        expected_head=expected_head, fetched_at_utc=fetched_at_utc,
    )


def _normalized_ci(ci: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "run_id": ci["run_id"],
        "repository": ci["repository"],
        "workflow_database_id": ci["workflow_database_id"],
        "attempt": ci["attempt"],
        "head_sha": ci["head_sha"],
        "conclusion": "success",
        "jobs": {job_id: "success" for job_id in upload.REQUIRED_JOB_IDS},
        "workflow": upload.WORKFLOW_NAME,
        "workflow_file": upload.WORKFLOW_FILE,
        "event": "workflow_dispatch",
        "head_branch": RELEASE_BRANCH,
        "head_ref": f"refs/heads/{RELEASE_BRANCH}",
        "url": ci["url"],
    }


def seal_release_evidence(
    output_path: pathlib.Path, *, campaign_plan_path: pathlib.Path,
    attempt: int, candidate_runtime: pathlib.Path,
    candidate_source: pathlib.Path, repository: pathlib.Path,
    promotion_path: pathlib.Path, preflight_path: pathlib.Path,
    ci_path: pathlib.Path, created_at_utc: str,
    selected_validator: SelectedValidator = _selected_candidate,
) -> dict[str, Any]:
    selected = dict(selected_validator(
        campaign_plan_path, attempt=attempt,
        candidate_runtime=candidate_runtime, candidate_source=candidate_source,
    ))
    payloads = _release_payloads(repository, selected)
    git, artifacts = _verify_promoted_commit(repository, payloads=payloads)
    promotion = validate_promotion(
        promotion_path, campaign_plan_path=campaign_plan_path, attempt=attempt,
        candidate_runtime=candidate_runtime, candidate_source=candidate_source,
        repository=repository, selected_validator=selected_validator,
    )
    preflight = validate_preflight_reference(
        preflight_path, campaign_plan_path=campaign_plan_path, attempt=attempt,
        candidate_runtime=candidate_runtime, candidate_source=candidate_source,
        repository=repository, promotion_path=promotion_path,
        base_preflight_path=pathlib.Path(
            qualification.load_sealed(
                pathlib.Path(
                    qualification.load_sealed(
                        preflight_path, PREFLIGHT_REFERENCE_SCHEMA
                    )["plan"]["path"]
                ), PREFLIGHT_PLAN_SCHEMA,
            )["inputs"]["base_preflight"]["path"]
        ),
        selected_validator=selected_validator,
    )
    if preflight["candidate_commit"] != git["commit"]:
        raise ReleaseBridgeError("release preflight belongs to another commit")
    ci = upload.validate_ci_evidence(ci_path, expected_head=git["commit"])
    created = _utc(created_at_utc, "release evidence time")
    chronology = (
        (_utc(promotion["promoted_at_utc"], "promotion time"), "promotion"),
        (_utc(preflight["receipt"]["completed_at_utc"], "preflight completion"), "preflight"),
        (_utc(ci["fetched_at_utc"], "CI fetch time"), "CI"),
    )
    if any(created < timestamp for timestamp, _label in chronology):
        raise ReleaseBridgeError("release evidence predates one of its inputs")
    source_binding_path = _safe_output(output_path.parent / "source-binding.json")
    qualification.create_source_binding(
        source_binding_path,
        candidate_source=pathlib.Path(preflight["candidate"]["path"]),
        candidate_commit=git["commit"],
        rank4_source=pathlib.Path(repository) / RANK4_RELATIVE,
        opponent_source=pathlib.Path(repository) / RANK4_RELATIVE,
    )
    candidate = {
        "runtime": dict(selected["runtime"]),
        "source": dict(preflight["candidate"]),
        "generated_source": dict(selected["generated_source"]),
        "architecture": dict(selected["architecture"]),
    }
    body = {
        "schema": RELEASE_EVIDENCE_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID,
        "attempt": attempt,
        "status": "release-committed-preflighted-green-ci-before-dual-final",
        "created_at_utc": created_at_utc,
        "campaign_plan": _reference(campaign_plan_path, challenger.PLAN_SCHEMA),
        "selection_evidence": selected["selection_evidence"],
        "candidate": candidate,
        "candidate_commit": git["commit"],
        "repository": str(pathlib.Path(repository).resolve()),
        "branch": RELEASE_BRANCH,
        "git": git,
        "tracked_artifacts": artifacts,
        "promotion": _reference(promotion_path, PROMOTION_SCHEMA),
        "derivation": payloads["derivation"],
        "frozen_execution_sources": selected["frozen_execution_sources"],
        "configuration": dict(selected["configuration"]),
        "source_binding": _reference(
            source_binding_path, qualification.SOURCE_BINDING_SCHEMA
        ),
        "maintained_preflight": _reference(
            preflight["base_preflight_path"], maintained.RECEIPT_SCHEMA
        ),
        "release_preflight": _reference(
            preflight_path, PREFLIGHT_REFERENCE_SCHEMA
        ),
        "gate": dict(preflight["gate"]),
        "compile_binding": {
            "compiler": dict(preflight["compiler"]),
            "command_sha256": qualification.sha256_bytes(
                qualification.canonical_json_bytes(preflight["compile_command"])
            ),
            "candidate_sha256": candidate["source"]["sha256"],
            "gate": dict(preflight["gate"]),
            "candidate_embedded": True,
        },
        "timing": dict(preflight["timing"]),
        "uncontended_timing": dict(preflight["uncontended_timing"]),
        "ci": _reference(ci_path, upload.CI_SCHEMA),
        "policy": {
            "created_before_dual_authorization": True,
            "protected_banks_accessed": False,
            "candidate_change_authorized": False,
            "uploads_authorized": 0,
            "rank4_replacement_authorized": False,
        },
    }
    output_path = _safe_output(output_path)
    expected = qualification.seal(body)
    if output_path.exists():
        if qualification.load_sealed(
            output_path, RELEASE_EVIDENCE_SCHEMA
        ) != expected:
            raise ReleaseBridgeError("existing release evidence changed")
    else:
        qualification.write_sealed(output_path, body)
    validate_release_evidence(
        output_path, campaign_plan_path=campaign_plan_path, attempt=attempt,
        candidate_runtime=candidate_runtime, candidate_source=candidate_source,
        selected_validator=selected_validator,
    )
    return qualification.load_sealed(output_path, RELEASE_EVIDENCE_SCHEMA)


def validate_release_evidence(
    path: pathlib.Path, *, campaign_plan_path: pathlib.Path, attempt: int,
    candidate_runtime: pathlib.Path, candidate_source: pathlib.Path,
    selected_validator: SelectedValidator = _selected_candidate,
) -> dict[str, Any]:
    """Deep validator consumed by challenger governance before authorization."""

    value = qualification.load_sealed(path, RELEASE_EVIDENCE_SCHEMA)
    if path.is_symlink() or path.name != "release-evidence.json":
        raise ReleaseBridgeError("release evidence route changed")
    selected = dict(selected_validator(
        campaign_plan_path, attempt=attempt,
        candidate_runtime=candidate_runtime, candidate_source=candidate_source,
    ))
    repository = pathlib.Path(str(value.get("repository", "")))
    payloads = _release_payloads(repository, selected)
    git, artifacts = _verify_promoted_commit(repository, payloads=payloads)
    promotion_path = _verify_reference(
        value.get("promotion"), PROMOTION_SCHEMA, "release promotion"
    )
    promotion = validate_promotion(
        promotion_path, campaign_plan_path=campaign_plan_path, attempt=attempt,
        candidate_runtime=candidate_runtime, candidate_source=candidate_source,
        repository=repository, selected_validator=selected_validator,
    )
    preflight_path = _verify_reference(
        value.get("release_preflight"), PREFLIGHT_REFERENCE_SCHEMA,
        "release preflight",
    )
    preflight_header = qualification.load_sealed(
        preflight_path, PREFLIGHT_REFERENCE_SCHEMA
    )
    preflight_plan_path = _verify_reference(
        preflight_header.get("plan"), PREFLIGHT_PLAN_SCHEMA,
        "release preflight plan",
    )
    preflight_plan = qualification.load_sealed(
        preflight_plan_path, PREFLIGHT_PLAN_SCHEMA
    )
    base_preflight_path = _verify_reference(
        value.get("maintained_preflight"), maintained.RECEIPT_SCHEMA,
        "maintained preflight",
    )
    if preflight_plan.get("inputs", {}).get("base_preflight") != _reference(
        base_preflight_path, maintained.RECEIPT_SCHEMA
    ):
        raise ReleaseBridgeError("release and maintained preflight references differ")
    preflight = validate_preflight_reference(
        preflight_path, campaign_plan_path=campaign_plan_path, attempt=attempt,
        candidate_runtime=candidate_runtime, candidate_source=candidate_source,
        repository=repository, promotion_path=promotion_path,
        base_preflight_path=base_preflight_path,
        selected_validator=selected_validator,
    )
    ci_path = _verify_reference(value.get("ci"), upload.CI_SCHEMA, "release CI")
    ci = upload.validate_ci_evidence(ci_path, expected_head=git["commit"])
    source_binding_path = _verify_reference(
        value.get("source_binding"), qualification.SOURCE_BINDING_SCHEMA,
        "release source binding",
    )
    binding = qualification.load_sealed(
        source_binding_path, qualification.SOURCE_BINDING_SCHEMA
    )
    qualification.validate_source_binding(binding)
    candidate = {
        "runtime": dict(selected["runtime"]),
        "source": dict(preflight["candidate"]),
        "generated_source": dict(selected["generated_source"]),
        "architecture": dict(selected["architecture"]),
    }
    compile_binding = {
        "compiler": dict(preflight["compiler"]),
        "command_sha256": qualification.sha256_bytes(
            qualification.canonical_json_bytes(preflight["compile_command"])
        ),
        "candidate_sha256": candidate["source"]["sha256"],
        "gate": dict(preflight["gate"]),
        "candidate_embedded": True,
    }
    expected = qualification.seal({
        "schema": RELEASE_EVIDENCE_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID,
        "attempt": attempt,
        "status": "release-committed-preflighted-green-ci-before-dual-final",
        "created_at_utc": str(value.get("created_at_utc")),
        "campaign_plan": _reference(campaign_plan_path, challenger.PLAN_SCHEMA),
        "selection_evidence": selected["selection_evidence"],
        "candidate": candidate,
        "candidate_commit": git["commit"],
        "repository": str(repository.resolve()),
        "branch": RELEASE_BRANCH,
        "git": git,
        "tracked_artifacts": artifacts,
        "promotion": _reference(promotion_path, PROMOTION_SCHEMA),
        "derivation": payloads["derivation"],
        "frozen_execution_sources": selected["frozen_execution_sources"],
        "configuration": dict(selected["configuration"]),
        "source_binding": _reference(
            source_binding_path, qualification.SOURCE_BINDING_SCHEMA
        ),
        "maintained_preflight": _reference(
            base_preflight_path, maintained.RECEIPT_SCHEMA
        ),
        "release_preflight": _reference(
            preflight_path, PREFLIGHT_REFERENCE_SCHEMA
        ),
        "gate": dict(preflight["gate"]),
        "compile_binding": compile_binding,
        "timing": dict(preflight["timing"]),
        "uncontended_timing": dict(preflight["uncontended_timing"]),
        "ci": _reference(ci_path, upload.CI_SCHEMA),
        "policy": {
            "created_before_dual_authorization": True,
            "protected_banks_accessed": False,
            "candidate_change_authorized": False,
            "uploads_authorized": 0,
            "rank4_replacement_authorized": False,
        },
    })
    created = _utc(value.get("created_at_utc"), "release evidence time")
    if created < max(
        _utc(promotion.get("promoted_at_utc"), "promotion time"),
        _utc(
            preflight["receipt"].get("completed_at_utc"),
            "release preflight completion time",
        ),
        _utc(ci.get("fetched_at_utc"), "CI fetch time"),
    ):
        raise ReleaseBridgeError("release evidence chronology changed")
    if (
        value != expected
        or preflight["candidate_commit"] != git["commit"]
        or preflight["candidate"] != candidate["source"]
        or preflight["runtime"] != candidate["runtime"]
        or preflight["derivation"] != payloads["derivation"]
        or preflight["configuration"] != selected["configuration"]
        or binding.get("candidate_commit") != git["commit"]
        or binding.get("candidate", {}).get("sha256")
        != candidate["source"]["sha256"]
        or ci.get("head_branch") != RELEASE_BRANCH
    ):
        raise ReleaseBridgeError("release evidence chain changed")
    return value


def dual_final_preflight_state(
    release_evidence_path: pathlib.Path, *, campaign_plan_path: pathlib.Path,
    attempt: int, candidate_runtime: pathlib.Path,
    candidate_source: pathlib.Path,
) -> dict[str, Any]:
    """Adapt validated release evidence to the dual runner's preflight shape.

    ``candidate_source`` is the governor's selected generated source.  The
    returned ``candidate`` is the exact committed deployment derivative that
    the dual authorization and both protected gates must use.
    """

    evidence = validate_release_evidence(
        release_evidence_path, campaign_plan_path=campaign_plan_path,
        attempt=attempt, candidate_runtime=candidate_runtime,
        candidate_source=candidate_source,
    )
    preflight_path = pathlib.Path(evidence["release_preflight"]["path"])
    reference = qualification.load_sealed(
        preflight_path, PREFLIGHT_REFERENCE_SCHEMA
    )
    plan_path = pathlib.Path(reference["plan"]["path"])
    plan = qualification.load_sealed(plan_path, PREFLIGHT_PLAN_SCHEMA)
    receipt_path = pathlib.Path(reference["receipt"]["path"])
    receipt = qualification.load_sealed(
        receipt_path, PREFLIGHT_RECEIPT_SCHEMA
    )
    return {
        "reference": reference,
        "receipt": receipt,
        "plan": plan,
        "candidate_commit": evidence["candidate_commit"],
        "candidate": evidence["candidate"]["source"],
        "runtime": evidence["candidate"]["runtime"],
        "derivation": {
            "schema": SOURCE_DERIVATION_SCHEMA,
            "configuration": evidence["configuration"],
            "source": evidence["derivation"],
        },
        "timing": evidence["timing"],
        "uncontended_timing": evidence["uncontended_timing"],
        "ci": evidence["ci"],
        "release_evidence": _reference(
            release_evidence_path, RELEASE_EVIDENCE_SCHEMA
        ),
    }


def _dual_chain(
    dual_qualified_path: pathlib.Path, *, campaign_plan_path: pathlib.Path,
    release_path: pathlib.Path, release: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        context = challenger.validate_campaign(campaign_plan_path)
        qualified = qualification.load_sealed(
            dual_qualified_path, challenger.DUAL_QUALIFICATION_SCHEMA
        )
    except Exception as error:
        raise ReleaseBridgeError("dual qualification failed campaign validation") from error
    attempt = release["attempt"]
    expected_path = (
        pathlib.Path(context["plan"]["outputs"]["dual_final"])
        / f"attempt-{attempt:03d}/dual-qualified.json"
    )
    if dual_qualified_path.is_symlink() or dual_qualified_path.resolve() != expected_path.resolve():
        raise ReleaseBridgeError("dual qualification route changed")
    try:
        dual_path = challenger._verify_sealed_record(
            qualified.get("dual_final_plan"), challenger.DUAL_FINAL_SCHEMA,
            "dual-final plan",
        )
    except Exception as error:
        raise ReleaseBridgeError("dual-final plan reference changed") from error
    dual_reference = dual_path.parent / "dual-final-reference.json"
    try:
        dual_state = challenger.validate_dual_final(
            dual_reference, plan_path=campaign_plan_path
        )
    except Exception as error:
        raise ReleaseBridgeError("dual-final plan failed validation") from error
    try:
        authorization_path = challenger._verify_sealed_record(
            dual_state["plan"].get("authorization"),
            challenger.DUAL_FINAL_AUTHORIZATION_SCHEMA,
            "dual-final authorization",
        )
    except Exception as error:
        raise ReleaseBridgeError("dual-final authorization reference changed") from error
    authorization = qualification.load_sealed(
        authorization_path, challenger.DUAL_FINAL_AUTHORIZATION_SCHEMA
    )
    release_reference = challenger._sealed_record(
        release_path, RELEASE_EVIDENCE_SCHEMA
    )
    candidate = {
        "runtime": {
            key: release["candidate"]["runtime"][key]
            for key in ("path", "bytes", "sha256")
        },
        "source": {
            key: release["candidate"]["source"][key]
            for key in ("path", "bytes", "sha256")
        },
        "architecture": release["candidate"]["architecture"],
    }
    if (
        qualified.get("attempt") != attempt
        or qualified.get("status")
        != "two-independent-strict-final-gates-passed"
        or qualified.get("candidate") != candidate
        or qualified.get("candidate_unchanged") is not True
        or qualified.get("independent_banks") is not True
        or qualified.get("rank4_replacement_authorized") is not False
        or qualified.get("upload_authorized") is not False
        or dual_state["plan"].get("candidate") != candidate
        or authorization.get("candidate") != candidate
        or authorization.get("release_evidence") != release_reference
    ):
        raise ReleaseBridgeError("dual qualification is not release-evidence bound")
    results = qualified.get("gate_results")
    if not isinstance(results, list) or len(results) != 2:
        raise ReleaseBridgeError("dual qualification lacks two results")
    dual_runner = _load(
        HERE / "compact_value_bfm_rank4_teacher_dual_final.py",
        "rank4_teacher_release_dual_runner",
    )
    deep = []
    for gate_id, result_reference in zip(("gate-a", "gate-b"), results, strict=True):
        try:
            result_path = challenger._verify_sealed_record(
                result_reference, challenger.FINAL_RESULT_SCHEMA,
                f"{gate_id} final result",
            )
        except Exception as error:
            raise ReleaseBridgeError(
                f"{gate_id} final result reference changed"
            ) from error
        try:
            result = challenger.validate_final_result(
                result_path, dual=dual_state["plan"], gate_id=gate_id
            )
            evidence_path = _verify_record(
                result["evidence"], f"{gate_id} deep evidence"
            )
            evidence = dual_runner.validate_governance_evidence(
                evidence_path, campaign_plan_path=campaign_plan_path,
                dual_reference=dual_reference,
            )
        except Exception as error:
            raise ReleaseBridgeError(f"{gate_id} deep evidence failed validation") from error
        maintained_path = _verify_reference(
            evidence.get("maintained_aggregate"),
            qualification.FINAL_AGGREGATE_SCHEMA,
            f"{gate_id} maintained aggregate",
        )
        binding_path = _verify_reference(
            evidence.get("gate_binding"), qualification.GATE_BINDING_SCHEMA,
            f"{gate_id} gate binding",
        )
        aggregate = qualification.load_sealed(
            maintained_path, qualification.FINAL_AGGREGATE_SCHEMA
        )
        if (
            evidence.get("candidate_commit") != release["candidate_commit"]
            or evidence.get("candidate", {}).get("source_sha256")
            != candidate["source"]["sha256"]
            or evidence.get("candidate", {}).get("runtime_sha256")
            != candidate["runtime"]["sha256"]
            or aggregate.get("verdict", {}).get("passed") is not True
        ):
            raise ReleaseBridgeError(f"{gate_id} belongs to another release candidate")
        deep.append({
            "gate_id": gate_id,
            "result_path": result_path,
            "evidence_path": evidence_path,
            "evidence": evidence,
            "aggregate_path": maintained_path,
            "aggregate": aggregate,
            "binding_path": binding_path,
        })
    if deep[0]["aggregate_path"].resolve() == deep[1]["aggregate_path"].resolve():
        raise ReleaseBridgeError("dual qualification reused one maintained aggregate")
    return {
        "qualified": qualified,
        "dual_state": dual_state,
        "authorization": authorization,
        "gates": deep,
    }


def _campaign_upload_binding(
    entries: Sequence[Mapping[str, Any]], *, attempt: int,
    upload_ordinal: int | None = None, require_unused: bool,
) -> dict[str, Any]:
    """Bind one release upload to the exact challenger continuation capability."""

    uploads = [
        (index, entry) for index, entry in enumerate(entries)
        if entry.get("event") == "upload-attested"
    ]
    if upload_ordinal is None:
        upload_ordinal = len(uploads) + 1
    if (
        isinstance(upload_ordinal, bool) or not isinstance(upload_ordinal, int)
        or upload_ordinal <= 0
    ):
        raise ReleaseBridgeError("campaign upload ordinal is invalid")
    used = [
        entry for _index, entry in uploads
        if entry.get("upload_ordinal") == upload_ordinal
    ]
    if len(used) > 1 or (used and used[0].get("attempt") != attempt):
        raise ReleaseBridgeError("campaign upload ordinal was reused by another attempt")
    if require_unused and used:
        raise ReleaseBridgeError("campaign upload authorization is already consumed")
    if upload_ordinal == 1:
        if require_unused and uploads:
            raise ReleaseBridgeError("first upload authorization follows an existing upload")
        return {
            "upload_ordinal": 1,
            "additional_upload_authorization": None,
            "authorization_event_body_sha256": None,
            "rejected_live_reference": None,
            "rejected_live_dynamic_exclusion": None,
        }

    prior_ordinals = {
        entry.get("upload_ordinal") for _index, entry in uploads
        if isinstance(entry.get("upload_ordinal"), int)
        and not isinstance(entry.get("upload_ordinal"), bool)
        and entry["upload_ordinal"] < upload_ordinal
    }
    if prior_ordinals != set(range(1, upload_ordinal)):
        raise ReleaseBridgeError("additional upload lacks its complete prior upload chain")
    capabilities = [
        (index, entry) for index, entry in enumerate(entries)
        if entry.get("event") == "additional-upload-authorized"
        and entry.get("next_attempt") == attempt
        and entry.get("next_upload_ordinal") == upload_ordinal
    ]
    if len(capabilities) != 1:
        raise ReleaseBridgeError(
            "additional upload lacks one exact challenger authorization"
        )
    index, event = capabilities[0]
    if index == 0:
        raise ReleaseBridgeError("additional upload authorization has no rejected window")
    rejected = entries[index - 1]
    authorization_record = event.get("authorization")
    try:
        authorization_path = challenger._verify_sealed_record(
            authorization_record,
            challenger.ADDITIONAL_UPLOAD_AUTHORIZATION_SCHEMA,
            "release additional-upload authorization",
        )
        authorization = qualification.load_sealed(
            authorization_path,
            challenger.ADDITIONAL_UPLOAD_AUTHORIZATION_SCHEMA,
        )
        live_reference = rejected.get("source_live_reference")
        if not isinstance(live_reference, Mapping):
            raise ReleaseBridgeError("rejected live reference is absent")
        challenger._verify_sealed_record(
            live_reference, str(live_reference.get("schema", "")),
            "release rejected live reference",
        )
        challenger._verify_dynamic_exclusion_record(
            rejected.get("dynamic_exclusion"),
            "release rejected live dynamic exclusion",
        )
    except Exception as error:
        if isinstance(error, ReleaseBridgeError):
            raise
        raise ReleaseBridgeError(
            "additional-upload authorization closure failed validation"
        ) from error
    expected_authorization_fields = {
        "schema", "campaign_id", "previous_attempt", "next_attempt",
        "next_upload_ordinal", "rejected_live_reference",
        "rejected_live_dynamic_exclusion", "explicit_user_authorization",
        "attempt_openings_authorized", "additional_uploads_authorized",
        "protected_or_live_data_training_allowed", "automatic_action",
        "body_sha256",
    }
    if (
        set(authorization) != expected_authorization_fields
        or rejected.get("event") != "live-window-recorded"
        or rejected.get("passed") is not False
        or rejected.get("adaptation_route")
        != "await-explicit-additional-upload-authorization"
        or rejected.get("upload_ordinal") != upload_ordinal - 1
        or event.get("attempt") != rejected.get("attempt")
        or event.get("consumed") is not False
        or event.get("adaptation_route")
        != "open-next-attempt-explicit-after-live-failure"
        or authorization.get("campaign_id") != CAMPAIGN_ID
        or authorization.get("previous_attempt") != rejected.get("attempt")
        or authorization.get("next_attempt") != attempt
        or authorization.get("next_upload_ordinal") != upload_ordinal
        or authorization.get("rejected_live_reference")
        != rejected.get("source_live_reference")
        or authorization.get("rejected_live_dynamic_exclusion")
        != rejected.get("dynamic_exclusion")
        or authorization.get("explicit_user_authorization") is not True
        or authorization.get("attempt_openings_authorized") != 1
        or authorization.get("additional_uploads_authorized") != 1
        or authorization.get("protected_or_live_data_training_allowed") is not False
        or authorization.get("automatic_action") is not False
    ):
        raise ReleaseBridgeError("additional-upload authorization binding changed")
    return {
        "upload_ordinal": upload_ordinal,
        "additional_upload_authorization": dict(authorization_record),
        "authorization_event_body_sha256": event.get("body_sha256"),
        "rejected_live_reference": dict(rejected["source_live_reference"]),
        "rejected_live_dynamic_exclusion": dict(rejected["dynamic_exclusion"]),
    }


def authorize_upload(
    output_root: pathlib.Path, *, release_evidence_path: pathlib.Path,
    campaign_plan_path: pathlib.Path, attempt: int,
    candidate_runtime: pathlib.Path, candidate_source: pathlib.Path,
    dual_qualified_path: pathlib.Path, authorized_at_utc: str,
) -> dict[str, Any]:
    release = validate_release_evidence(
        release_evidence_path, campaign_plan_path=campaign_plan_path,
        attempt=attempt, candidate_runtime=candidate_runtime,
        candidate_source=candidate_source,
    )
    chain = _dual_chain(
        dual_qualified_path, campaign_plan_path=campaign_plan_path,
        release_path=release_evidence_path, release=release,
    )
    campaign_context = challenger.validate_campaign(campaign_plan_path)
    entries = challenger.load_ledger(campaign_context["plan"])
    campaign_upload = _campaign_upload_binding(
        entries, attempt=attempt, require_unused=True,
    )
    if campaign_context["inputs"].get("production_allowlist_enforced") is True:
        expected_root = (
            pathlib.Path(campaign_context["plan"]["outputs"]["root"]).resolve()
            / "release" / f"attempt-{attempt:03d}" / "upload"
        )
        if output_root.resolve() != expected_root:
            raise ReleaseBridgeError(
                "production upload authorization root is not campaign-derived"
            )
    root = _safe_directory(output_root, create=True)
    authorization_path = _safe_output(root / "one-upload-authorization.json")
    inputs_path = _safe_output(root / "authorization-inputs.json")
    authorization_exists = authorization_path.exists()
    inputs_exist = inputs_path.exists()
    if inputs_exist and not authorization_exists:
        raise ReleaseBridgeError(
            "upload authorization inputs exist without their capability"
        )
    existing_authorization = (
        qualification.load_sealed(
            authorization_path, qualification.UPLOAD_AUTH_SCHEMA
        )
        if authorization_exists else None
    )
    existing_ledger_claim = next((
        entry for entry in reversed(entries)
        if entry.get("event") == "upload-authorization-claimed"
        and entry.get("attempt") == attempt
        and entry.get("upload_ordinal") == campaign_upload["upload_ordinal"]
    ), None)
    authorized_text = str(
        existing_ledger_claim.get("authorized_at_utc")
        if existing_ledger_claim is not None
        else existing_authorization.get("authorized_at_utc")
        if existing_authorization is not None
        else authorized_at_utc
    )
    authorized = _utc(authorized_text, "one-upload authorization time")
    gate_b = chain["gates"][1]
    ci_path = pathlib.Path(release["ci"]["path"])
    ci = upload.validate_ci_evidence(
        ci_path, expected_head=release["candidate_commit"]
    )
    chronological_inputs = [
        _utc(release["created_at_utc"], "release evidence time"),
        _utc(chain["qualified"]["completed_at_utc"], "dual completion time"),
        _utc(ci["fetched_at_utc"], "CI fetch time"),
    ]
    if campaign_upload["additional_upload_authorization"] is not None:
        capability_event = next(
            entry for entry in entries
            if entry.get("body_sha256")
            == campaign_upload["authorization_event_body_sha256"]
        )
        chronological_inputs.append(_utc(
            capability_event.get("created_at_utc"),
            "additional-upload authorization time",
        ))
    if authorized < max(chronological_inputs):
        raise ReleaseBridgeError("upload authorization predates qualification")
    binding = qualification.load_sealed(
        gate_b["binding_path"], qualification.GATE_BINDING_SCHEMA
    )
    expected_candidate = release["candidate"]["source"]
    if (
        binding.get("candidate_commit") != release["candidate_commit"]
        or binding.get("candidate", {}).get("sha256")
        != expected_candidate["sha256"]
        or binding.get("candidate", {}).get("bytes")
        != expected_candidate["bytes"]
    ):
        raise ReleaseBridgeError("Gate B binding differs from release source")
    claim = challenger.claim_upload_authorization(
        campaign_plan_path,
        attempt=attempt,
        output_root=root,
        release_evidence_path=release_evidence_path,
        candidate_commit=release["candidate_commit"],
        campaign_upload_binding=campaign_upload,
        authorized_at_utc=authorized_text,
    )
    claim_binding = {
        "sequence": claim["sequence"],
        "body_sha256": claim["body_sha256"],
        "attempt": claim["attempt"],
        "upload_ordinal": claim["upload_ordinal"],
    }
    authorization_body = {
        "schema": qualification.UPLOAD_AUTH_SCHEMA,
        "namespace": NAMESPACE,
        "uploads_authorized": 1,
        "rank4_replacement_authorized": False,
        "authorized_at_utc": authorized_text,
        "candidate_commit": release["candidate_commit"],
        "candidate": binding["candidate"],
        "binding": _reference(
            gate_b["binding_path"], qualification.GATE_BINDING_SCHEMA
        ),
        "aggregate": _reference(
            gate_b["aggregate_path"], qualification.FINAL_AGGREGATE_SCHEMA
        ),
        "ci": _normalized_ci(ci),
        "upload_ledger_root": str(root),
        "release_evidence": _reference(
            release_evidence_path, RELEASE_EVIDENCE_SCHEMA
        ),
        "dual_qualification": _reference(
            dual_qualified_path, challenger.DUAL_QUALIFICATION_SCHEMA
        ),
        "two_independent_rank4_gates_passed": True,
        "campaign_upload_binding": campaign_upload,
        "campaign_upload_claim": claim_binding,
    }
    expected_authorization = qualification.seal(authorization_body)
    if existing_authorization is None:
        qualification.write_sealed(authorization_path, authorization_body)
    elif existing_authorization != expected_authorization:
        raise ReleaseBridgeError(
            "orphaned upload authorization changed before input publication"
        )
    qualification.write_sealed(inputs_path, {
        "schema": UPLOAD_INPUTS_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID,
        "attempt": attempt,
        "status": "exactly-one-upload-authorized-after-dual-qualification",
        "authorized_at_utc": authorized_text,
        "release_evidence": _reference(
            release_evidence_path, RELEASE_EVIDENCE_SCHEMA
        ),
        "dual_qualification": _reference(
            dual_qualified_path, challenger.DUAL_QUALIFICATION_SCHEMA
        ),
        "gate_results": [
            _reference(gate["result_path"], challenger.FINAL_RESULT_SCHEMA)
            for gate in chain["gates"]
        ],
        "authorization": _reference(
            authorization_path, qualification.UPLOAD_AUTH_SCHEMA
        ),
        "candidate_commit": release["candidate_commit"],
        "candidate": release["candidate"],
        "ci": release["ci"],
        "campaign_upload_binding": campaign_upload,
        "campaign_upload_claim": claim_binding,
        "uploads_authorized": 1,
        "submit_clicks_authorized": 1,
        "second_upload_authorized": False,
        "rank4_replacement_authorized": False,
    })
    validated = validate_upload_authorization(
        output_root, release_evidence_path=release_evidence_path,
        campaign_plan_path=campaign_plan_path, attempt=attempt,
        candidate_runtime=candidate_runtime, candidate_source=candidate_source,
        dual_qualified_path=dual_qualified_path,
    )
    challenger.record_upload_authorization(
        campaign_plan_path,
        attempt=attempt,
        authorization_path=authorization_path,
        authorization_inputs_path=inputs_path,
        created_at_utc=str(validated["inputs"]["authorized_at_utc"]),
    )
    return validated["authorization"]


def validate_upload_authorization(
    output_root: pathlib.Path, *, release_evidence_path: pathlib.Path,
    campaign_plan_path: pathlib.Path, attempt: int,
    candidate_runtime: pathlib.Path, candidate_source: pathlib.Path,
    dual_qualified_path: pathlib.Path,
) -> dict[str, Any]:
    root = _safe_directory(output_root, create=False)
    authorization_path = root / "one-upload-authorization.json"
    inputs_path = root / "authorization-inputs.json"
    authorization = qualification.load_sealed(
        authorization_path, qualification.UPLOAD_AUTH_SCHEMA
    )
    inputs = qualification.load_sealed(inputs_path, UPLOAD_INPUTS_SCHEMA)
    release = validate_release_evidence(
        release_evidence_path, campaign_plan_path=campaign_plan_path,
        attempt=attempt, candidate_runtime=candidate_runtime,
        candidate_source=candidate_source,
    )
    chain = _dual_chain(
        dual_qualified_path, campaign_plan_path=campaign_plan_path,
        release_path=release_evidence_path, release=release,
    )
    gate_b = chain["gates"][1]
    binding = qualification.load_sealed(
        gate_b["binding_path"], qualification.GATE_BINDING_SCHEMA
    )
    ci_path = pathlib.Path(release["ci"]["path"])
    ci = upload.validate_ci_evidence(
        ci_path, expected_head=release["candidate_commit"]
    )
    stored_campaign_upload = inputs.get("campaign_upload_binding")
    if not isinstance(stored_campaign_upload, Mapping):
        raise ReleaseBridgeError("campaign upload binding is absent")
    campaign_context = challenger.validate_campaign(campaign_plan_path)
    if campaign_context["inputs"].get("production_allowlist_enforced") is True:
        expected_root = (
            pathlib.Path(campaign_context["plan"]["outputs"]["root"]).resolve()
            / "release" / f"attempt-{attempt:03d}" / "upload"
        )
        if root.resolve() != expected_root:
            raise ReleaseBridgeError(
                "production upload authorization root is not campaign-derived"
            )
    entries = challenger.load_ledger(campaign_context["plan"])
    campaign_upload = _campaign_upload_binding(
        entries, attempt=attempt,
        upload_ordinal=stored_campaign_upload.get("upload_ordinal"),
        require_unused=False,
    )
    claims = [
        entry for entry in entries
        if entry.get("event") == "upload-authorization-claimed"
        and entry.get("attempt") == attempt
        and entry.get("upload_ordinal") == campaign_upload["upload_ordinal"]
    ]
    if len(claims) != 1 or claims[0].get("output_root") != str(root.resolve()):
        raise ReleaseBridgeError("upload authorization lacks its unique ledger claim")
    claim_binding = {
        "sequence": claims[0]["sequence"],
        "body_sha256": claims[0]["body_sha256"],
        "attempt": claims[0]["attempt"],
        "upload_ordinal": claims[0]["upload_ordinal"],
    }
    expected_authorization = qualification.seal({
        "schema": qualification.UPLOAD_AUTH_SCHEMA,
        "namespace": NAMESPACE,
        "uploads_authorized": 1,
        "rank4_replacement_authorized": False,
        "authorized_at_utc": str(inputs.get("authorized_at_utc")),
        "candidate_commit": release["candidate_commit"],
        "candidate": binding["candidate"],
        "binding": _reference(
            gate_b["binding_path"], qualification.GATE_BINDING_SCHEMA
        ),
        "aggregate": _reference(
            gate_b["aggregate_path"], qualification.FINAL_AGGREGATE_SCHEMA
        ),
        "ci": _normalized_ci(ci),
        "upload_ledger_root": str(root),
        "release_evidence": _reference(
            release_evidence_path, RELEASE_EVIDENCE_SCHEMA
        ),
        "dual_qualification": _reference(
            dual_qualified_path, challenger.DUAL_QUALIFICATION_SCHEMA
        ),
        "two_independent_rank4_gates_passed": True,
        "campaign_upload_binding": campaign_upload,
        "campaign_upload_claim": claim_binding,
    })
    expected_inputs = qualification.seal({
        "schema": UPLOAD_INPUTS_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID,
        "attempt": attempt,
        "status": "exactly-one-upload-authorized-after-dual-qualification",
        "authorized_at_utc": str(inputs.get("authorized_at_utc")),
        "release_evidence": _reference(
            release_evidence_path, RELEASE_EVIDENCE_SCHEMA
        ),
        "dual_qualification": _reference(
            dual_qualified_path, challenger.DUAL_QUALIFICATION_SCHEMA
        ),
        "gate_results": [
            _reference(gate["result_path"], challenger.FINAL_RESULT_SCHEMA)
            for gate in chain["gates"]
        ],
        "authorization": _reference(
            authorization_path, qualification.UPLOAD_AUTH_SCHEMA
        ),
        "candidate_commit": release["candidate_commit"],
        "candidate": release["candidate"],
        "ci": release["ci"],
        "campaign_upload_binding": campaign_upload,
        "campaign_upload_claim": claim_binding,
        "uploads_authorized": 1,
        "submit_clicks_authorized": 1,
        "second_upload_authorized": False,
        "rank4_replacement_authorized": False,
    })
    authorized = _utc(
        inputs.get("authorized_at_utc"), "one-upload authorization time"
    )
    chronological_inputs = [
        _utc(release["created_at_utc"], "release evidence time"),
        _utc(chain["qualified"]["completed_at_utc"], "dual completion time"),
        _utc(ci["fetched_at_utc"], "CI fetch time"),
    ]
    if campaign_upload["additional_upload_authorization"] is not None:
        capability_event = next(
            entry for entry in entries
            if entry.get("body_sha256")
            == campaign_upload["authorization_event_body_sha256"]
        )
        chronological_inputs.append(_utc(
            capability_event.get("created_at_utc"),
            "additional-upload authorization time",
        ))
    if (
        authorization != expected_authorization
        or inputs != expected_inputs
        or authorized < max(chronological_inputs)
        or authorization.get("uploads_authorized") != 1
        or authorization.get("rank4_replacement_authorized") is not False
    ):
        raise ReleaseBridgeError("one-upload authorization chain changed")
    return {
        "authorization": authorization,
        "authorization_path": authorization_path,
        "inputs": inputs,
        "inputs_path": inputs_path,
        "release": release,
        "dual": chain,
    }


def _json(path: pathlib.Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReleaseBridgeError(f"JSON input is unreadable: {path}") from error
    if not isinstance(value, dict):
        raise ReleaseBridgeError(f"JSON input is not an object: {path}")
    return value


def _selected_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--campaign-plan", type=pathlib.Path, required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--candidate-runtime", type=pathlib.Path, required=True)
    parser.add_argument("--candidate-source", type=pathlib.Path, required=True)


def _preflight_arguments(parser: argparse.ArgumentParser) -> None:
    _selected_arguments(parser)
    parser.add_argument("--repository", type=pathlib.Path, required=True)
    parser.add_argument("--promotion", type=pathlib.Path, required=True)
    parser.add_argument("--base-preflight", type=pathlib.Path, required=True)


def _discover(name: str) -> pathlib.Path:
    value = shutil.which(name)
    if value is None:
        raise ReleaseBridgeError(f"required executable is absent: {name}")
    return pathlib.Path(value)


def _discover_gnu_compiler() -> pathlib.Path:
    """Find a real GNU C++ driver, never Apple's ``/usr/bin/g++`` alias."""

    candidates = (
        "g++-15", "g++-14", "g++-13", "g++-12", "g++-11", "g++",
    )
    for name in candidates:
        value = shutil.which(name)
        if value is None:
            continue
        path = pathlib.Path(value)
        try:
            deployment_preflight._compiler_record(path, "GNU")
        except Exception:
            continue
        return path
    raise ReleaseBridgeError("no exact GNU C++ compiler is available")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    promote = commands.add_parser("promote")
    _selected_arguments(promote)
    promote.add_argument("--repository", type=pathlib.Path, required=True)
    promote.add_argument("--output", type=pathlib.Path, required=True)
    promote.add_argument("--promoted-at-utc", default=utc_now())

    prepare = commands.add_parser("prepare-preflight")
    _preflight_arguments(prepare)
    prepare.add_argument("--output-root", type=pathlib.Path, required=True)
    prepare.add_argument("--python", type=pathlib.Path)
    prepare.add_argument("--gcc", type=pathlib.Path)
    prepare.add_argument("--clang", type=pathlib.Path)
    prepare.add_argument("--node", type=pathlib.Path)
    prepare.add_argument("--planned-at-utc", default=utc_now())

    run = commands.add_parser("run-preflight")
    _preflight_arguments(run)
    run.add_argument("--plan", type=pathlib.Path, required=True)
    run.add_argument("--claimed-at-utc", default=utc_now())

    ci = commands.add_parser("seal-ci")
    ci.add_argument("--output", type=pathlib.Path, required=True)
    ci.add_argument("--head", required=True)
    ci.add_argument("--run-id", type=int)
    ci.add_argument("--gh-json", type=pathlib.Path)
    ci.add_argument("--gh", type=pathlib.Path, default=pathlib.Path("gh"))
    ci.add_argument("--fetched-at-utc", default=utc_now())

    evidence = commands.add_parser("seal-release-evidence")
    _selected_arguments(evidence)
    evidence.add_argument("--repository", type=pathlib.Path, required=True)
    evidence.add_argument("--promotion", type=pathlib.Path, required=True)
    evidence.add_argument("--preflight", type=pathlib.Path, required=True)
    evidence.add_argument("--ci", type=pathlib.Path, required=True)
    evidence.add_argument("--output", type=pathlib.Path, required=True)
    evidence.add_argument("--created-at-utc", default=utc_now())

    validate = commands.add_parser("validate-release-evidence")
    _selected_arguments(validate)
    validate.add_argument("--release-evidence", type=pathlib.Path, required=True)

    authorize = commands.add_parser("authorize-upload")
    _selected_arguments(authorize)
    authorize.add_argument("--release-evidence", type=pathlib.Path, required=True)
    authorize.add_argument("--dual-qualified", type=pathlib.Path, required=True)
    authorize.add_argument("--output-root", type=pathlib.Path, required=True)
    authorize.add_argument("--authorized-at-utc", default=utc_now())

    verify = commands.add_parser("validate-upload")
    _selected_arguments(verify)
    verify.add_argument("--release-evidence", type=pathlib.Path, required=True)
    verify.add_argument("--dual-qualified", type=pathlib.Path, required=True)
    verify.add_argument("--output-root", type=pathlib.Path, required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "promote":
            result: Any = promote_candidate(
                campaign_plan_path=args.campaign_plan, attempt=args.attempt,
                candidate_runtime=args.candidate_runtime,
                candidate_source=args.candidate_source,
                repository=args.repository, output_path=args.output,
                promoted_at_utc=args.promoted_at_utc,
            )
        elif args.command == "prepare-preflight":
            result = prepare_preflight(
                campaign_plan_path=args.campaign_plan, attempt=args.attempt,
                candidate_runtime=args.candidate_runtime,
                candidate_source=args.candidate_source,
                repository=args.repository, promotion_path=args.promotion,
                base_preflight_path=args.base_preflight,
                output_root=args.output_root,
                python_path=args.python or args.repository / ".venv/bin/python",
                gcc_path=args.gcc or _discover_gnu_compiler(),
                clang_path=args.clang or _discover("clang++"),
                node_path=args.node or _discover("node"),
                planned_at_utc=args.planned_at_utc,
            )
            result = {"plan": str(result), "sha256": qualification.sha256_file(result)}
        elif args.command == "run-preflight":
            result = run_preflight(
                args.plan, campaign_plan_path=args.campaign_plan,
                attempt=args.attempt, candidate_runtime=args.candidate_runtime,
                candidate_source=args.candidate_source,
                repository=args.repository, promotion_path=args.promotion,
                base_preflight_path=args.base_preflight,
                claimed_at_utc=args.claimed_at_utc,
            )
            result = {"reference": str(result), "sha256": qualification.sha256_file(result)}
        elif args.command == "seal-ci":
            if (args.run_id is None) == (args.gh_json is None):
                raise ReleaseBridgeError(
                    "seal-ci requires exactly one of --run-id/--gh-json"
                )
            payload = _json(args.gh_json) if args.gh_json else upload.fetch_gh_run(
                args.run_id, gh_executable=args.gh
            )
            result = seal_ci_evidence(
                args.output, gh_payload=payload, expected_head=args.head,
                fetched_at_utc=args.fetched_at_utc,
            )
        elif args.command == "seal-release-evidence":
            result = seal_release_evidence(
                args.output, campaign_plan_path=args.campaign_plan,
                attempt=args.attempt, candidate_runtime=args.candidate_runtime,
                candidate_source=args.candidate_source,
                repository=args.repository, promotion_path=args.promotion,
                preflight_path=args.preflight, ci_path=args.ci,
                created_at_utc=args.created_at_utc,
            )
        elif args.command == "validate-release-evidence":
            result = validate_release_evidence(
                args.release_evidence, campaign_plan_path=args.campaign_plan,
                attempt=args.attempt, candidate_runtime=args.candidate_runtime,
                candidate_source=args.candidate_source,
            )
        elif args.command == "authorize-upload":
            result = authorize_upload(
                args.output_root, release_evidence_path=args.release_evidence,
                campaign_plan_path=args.campaign_plan, attempt=args.attempt,
                candidate_runtime=args.candidate_runtime,
                candidate_source=args.candidate_source,
                dual_qualified_path=args.dual_qualified,
                authorized_at_utc=args.authorized_at_utc,
            )
        else:
            state = validate_upload_authorization(
                args.output_root, release_evidence_path=args.release_evidence,
                campaign_plan_path=args.campaign_plan, attempt=args.attempt,
                candidate_runtime=args.candidate_runtime,
                candidate_source=args.candidate_source,
                dual_qualified_path=args.dual_qualified,
            )
            result = state["authorization"]
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except (
        ReleaseBridgeError, qualification.QualificationError,
        OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError,
        ValueError,
    ) as error:
        print(f"Rank-4 release bridge failure: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
