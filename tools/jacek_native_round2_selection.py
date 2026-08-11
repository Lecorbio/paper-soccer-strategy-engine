#!/usr/bin/env python3
"""Record and finalize provenance-safe Jacek-native round-two seed gates.

The training model is immutable and deliberately has no chosen seed.  This
tool records each actual-clock match as content-addressed evidence, then emits
one separate immutable selection sidecar only after every retained seed has a
valid screen and decisive report.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import fcntl
import hashlib
import json
import math
import os
import pathlib
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
EXPORTER_DIRECTORY = ROOT / "submissions" / "codingame" / "tools"
if str(EXPORTER_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(EXPORTER_DIRECTORY))
import generate_jacek_native_model as round1_exporter  # noqa: E402
import generate_jacek_native_model_round2 as round2_exporter  # noqa: E402


REPORT_SCHEMA = "papersoccer.jacek-native-round2-gate-report/v1"
SELECTION_SCHEMA = "papersoccer.jacek-native-round2-selection/v1"
OPENING_TURNS = (0, 4, 8, 12)
OPENING_SEED = 6_517_766_227_279_252_335
MAXIMUM_TURNS = 384


@dataclasses.dataclass(frozen=True)
class GateProfile:
    name: str
    pairs: int
    first_ms: int
    later_ms: int
    minimum_candidate_wins: int
    minimum_wins_per_color: int

    def payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "pairs": self.pairs,
            "games": self.pairs * 2,
            "first_ms": self.first_ms,
            "later_ms": self.later_ms,
            "maximum_turns": MAXIMUM_TURNS,
            "opening_turns": list(OPENING_TURNS),
            "opening_seed": str(OPENING_SEED),
            "shuffle_seed_policy": "deployment-constant",
            "minimum_candidate_wins": self.minimum_candidate_wins,
            "minimum_wins_per_color": self.minimum_wins_per_color,
            "require_zero_unfinished": True,
            "require_zero_headroom_failures": True,
            "require_zero_operational_timeouts": True,
        }


PROFILES = {
    "screen": GateProfile("screen", 500, 50, 10, 530, 0),
    "decisive": GateProfile("decisive", 106, 800, 155, 112, 50),
}

GATE_SOURCE_PATHS = (
    "tools/jacek_native_model_gate.cpp",
    "submissions/codingame/bots/jacek_native_bfm/bot.cpp",
    "submissions/codingame/bots/jacek_native_bfm/jacek_native_model.hpp",
    "src/core/rules.cpp",
    "src/core/geometry.cpp",
    "src/bots/mcts_internal.hpp",
    "include/papersoccer/types.hpp",
    "include/papersoccer/geometry.hpp",
    "include/papersoccer/rules.hpp",
)

SUMMARY_INTEGER_FIELDS = {
    "candidate",
    "baseline",
    "unfinished",
    "candidate_player_one",
    "candidate_player_two",
    "games",
    "candidate_decisions",
    "candidate_expansions",
    "candidate_child_evaluations",
    "candidate_max_tree",
    "candidate_deadline_searches",
    "candidate_headroom_failures",
    "candidate_operational_timeouts",
    "baseline_decisions",
    "baseline_expansions",
    "baseline_headroom_failures",
    "baseline_operational_timeouts",
    "required_total",
    "required_per_color",
}
SUMMARY_FLOAT_FIELDS = {
    "candidate_ms",
    "candidate_max_first_ms",
    "candidate_max_later_ms",
    "baseline_max_first_ms",
    "baseline_max_later_ms",
}
SUMMARY_TEXT_FIELDS = {"profile", "shuffle_seed_policy", "passed"}
PAIR_PATTERN = re.compile(
    r"pair=(0|[1-9][0-9]*) opening_turns=(0|[1-9][0-9]*) "
    r"seed=(0|[1-9][0-9]*) c0=(-1|0|1) c1=(-1|0|1)"
)


class SelectionError(ValueError):
    """The gate evidence or immutable selection contract is invalid."""


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()


def _duplicate_rejecting_object(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SelectionError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise SelectionError(f"non-finite JSON value: {value}")


def _strict_json(raw: bytes, label: str, canonical: bool = True) -> Any:
    try:
        text = raw.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise SelectionError(f"{label} is not strict JSON") from error
    if canonical and canonical_json_bytes(value) != raw:
        raise SelectionError(f"{label} is not canonical JSON")
    return value


def _load_canonical(path: pathlib.Path, label: str) -> tuple[bytes, Any]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise SelectionError(f"cannot read {label}: {path}") from error
    return raw, _strict_json(raw, label)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _integer(value: object, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SelectionError(f"{label} is not a valid integer")
    return value


def _model_seed_metadata(model: Mapping[str, Any], seed: int) -> dict[str, Any]:
    training = model.get("training")
    checkpoints = model.get("checkpoints")
    if not isinstance(training, Mapping) or not isinstance(checkpoints, list):
        raise SelectionError("round-two model has no retained checkpoints")
    if training.get("chosen_seed") is not None:
        raise SelectionError("round-two model was mutated after training")
    matches = [
        checkpoint
        for checkpoint in checkpoints
        if isinstance(checkpoint, Mapping) and checkpoint.get("seed") == seed
    ]
    if len(matches) != 1:
        raise SelectionError(f"round-two seed {seed} is not retained exactly once")
    checkpoint_sha = matches[0].get("checkpoint_sha256")
    if not _valid_sha256(checkpoint_sha):
        raise SelectionError(f"round-two seed {seed} has no checkpoint identity")
    return {"seed": seed, "checkpoint_sha256": checkpoint_sha}


def _runtime_lines(raw: bytes, label: str) -> dict[str, str]:
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeError as error:
        raise SelectionError(f"{label} is not UTF-8") from error
    if len(lines) != 7 or not all(_valid_sha256(lines[index]) for index in (3, 4)):
        raise SelectionError(f"{label} runtime metadata is malformed")
    return {"model_sha256": lines[3], "packed_sha256": lines[4]}


def _round2_identity(
    model_path: pathlib.Path, seed: int, runtime_path: pathlib.Path
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    model_raw, model = _load_canonical(model_path, "round-two model")
    if not isinstance(model, Mapping):
        raise SelectionError("round-two model root is not an object")
    model_sha = _sha256(model_raw)
    seed_metadata = _model_seed_metadata(model, seed)
    try:
        expected = round2_exporter.render_runtime(model, model_sha, seed).encode()
    except (KeyError, TypeError, ValueError) as error:
        raise SelectionError("round-two model failed frozen export validation") from error
    runtime_raw = runtime_path.read_bytes()
    if runtime_raw != expected:
        raise SelectionError(
            "candidate runtime is not the exact explicit-seed model export"
        )
    runtime = _runtime_lines(runtime_raw, "candidate")
    return ({
        **seed_metadata,
        "model_sha256": model_sha,
        "runtime_sha256": _sha256(runtime_raw),
        "packed_sha256": runtime["packed_sha256"],
        "runtime_bytes": len(runtime_raw),
    }, model)


def _round1_identity(
    model_path: pathlib.Path, seed: int, runtime_path: pathlib.Path
) -> dict[str, Any]:
    model_raw, model = _load_canonical(model_path, "baseline model")
    if not isinstance(model, Mapping):
        raise SelectionError("baseline model root is not an object")
    model_sha = _sha256(model_raw)
    try:
        expected = round1_exporter.render_runtime(model, model_sha, seed).encode()
    except (KeyError, TypeError, ValueError) as error:
        raise SelectionError("baseline model failed frozen export validation") from error
    runtime_raw = runtime_path.read_bytes()
    if runtime_raw != expected:
        raise SelectionError("baseline runtime is not the exact seed model export")
    runtime = _runtime_lines(runtime_raw, "baseline")
    return {
        "seed": seed,
        "model_sha256": model_sha,
        "runtime_sha256": _sha256(runtime_raw),
        "packed_sha256": runtime["packed_sha256"],
        "runtime_bytes": len(runtime_raw),
    }


def _source_identities() -> list[dict[str, str]]:
    result = []
    for relative in GATE_SOURCE_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise SelectionError(f"gate source is missing: {relative}")
        result.append({"path": relative, "sha256": _sha256(path.read_bytes())})
    return result


def _key_values(line: str, prefix: str) -> dict[str, str]:
    if not line.startswith(prefix):
        raise SelectionError(f"gate output omits {prefix.strip()} line")
    result: dict[str, str] = {}
    for token in line[len(prefix):].split():
        if "=" not in token:
            raise SelectionError("gate output contains a malformed field")
        key, value = token.split("=", 1)
        if not key or not value or key in result:
            raise SelectionError("gate output contains a duplicate/empty field")
        result[key] = value
    return result


def _identity_fields(line: str, label: str) -> dict[str, str]:
    fields = _key_values(line, "")
    prefix = f"{label}_"
    if not fields or any(not key.startswith(prefix) for key in fields):
        raise SelectionError(f"gate {label} identity prefix is malformed")
    return {key[len(prefix):]: value for key, value in fields.items()}


def _parse_unsigned(text: str, label: str) -> int:
    if not re.fullmatch(r"0|[1-9][0-9]*", text):
        raise SelectionError(f"gate {label} is not an unsigned integer")
    return int(text)


def _parse_float(text: str, label: str) -> float:
    try:
        value = float(text)
    except ValueError as error:
        raise SelectionError(f"gate {label} is not numeric") from error
    if not math.isfinite(value) or value < 0.0:
        raise SelectionError(f"gate {label} is not finite/nonnegative")
    return value


def parse_gate_stdout(
    raw: bytes,
    profile: GateProfile,
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    exit_code: int,
) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeError as error:
        raise SelectionError("gate stdout is not UTF-8") from error
    if not text.endswith("\n") or "\r" in text or "\x00" in text:
        raise SelectionError("gate stdout is not canonical line text")
    lines = text.splitlines()
    if len(lines) != profile.pairs + 3:
        raise SelectionError("gate stdout has an incomplete/extra pair transcript")

    candidate_fields = _identity_fields(lines[0], "candidate")
    baseline_fields = _identity_fields(lines[1], "baseline")
    expected_identity_keys = {
        "runtime_sha256", "model_sha256", "packed_sha256"
    }
    if set(candidate_fields) != expected_identity_keys or set(
            baseline_fields) != expected_identity_keys:
        raise SelectionError("gate identity fields are not frozen")
    for label, observed, expected in (
        ("candidate", candidate_fields, candidate),
        ("baseline", baseline_fields, baseline),
    ):
        for field in expected_identity_keys:
            if observed[field] != expected[field]:
                raise SelectionError(f"gate {label} {field} is stale")

    candidate_wins = 0
    baseline_wins = 0
    unfinished = 0
    candidate_player_one = 0
    candidate_player_two = 0
    opening_seeds: set[int] = set()
    for index, line in enumerate(lines[2:-1]):
        match = PAIR_PATTERN.fullmatch(line)
        if match is None:
            raise SelectionError(f"gate pair {index} line is malformed")
        pair, depth, opening_seed, player_zero, player_one = map(
            int, match.groups()
        )
        if pair != index or depth != OPENING_TURNS[index % len(OPENING_TURNS)]:
            raise SelectionError("gate pair ordering/opening schedule is stale")
        if opening_seed in opening_seeds:
            raise SelectionError("gate opening seeds are not unique")
        opening_seeds.add(opening_seed)
        for candidate_player, winner in enumerate((player_zero, player_one)):
            if winner < 0:
                unfinished += 1
            elif winner == candidate_player:
                candidate_wins += 1
                if candidate_player == 0:
                    candidate_player_one += 1
                else:
                    candidate_player_two += 1
            else:
                baseline_wins += 1

    raw_summary = _key_values(lines[-1], "summary ")
    expected_summary_fields = (
        SUMMARY_INTEGER_FIELDS | SUMMARY_FLOAT_FIELDS | SUMMARY_TEXT_FIELDS
    )
    if set(raw_summary) != expected_summary_fields:
        raise SelectionError("gate summary fields are not frozen")
    summary: dict[str, Any] = {
        key: _parse_unsigned(raw_summary[key], key)
        for key in SUMMARY_INTEGER_FIELDS
    }
    summary.update({
        key: _parse_float(raw_summary[key], key)
        for key in SUMMARY_FLOAT_FIELDS
    })
    if raw_summary["passed"] not in {"true", "false"}:
        raise SelectionError("gate passed flag is malformed")
    summary.update({
        "profile": raw_summary["profile"],
        "shuffle_seed_policy": raw_summary["shuffle_seed_policy"],
        "passed": raw_summary["passed"] == "true",
    })
    recomputed = {
        "candidate": candidate_wins,
        "baseline": baseline_wins,
        "unfinished": unfinished,
        "candidate_player_one": candidate_player_one,
        "candidate_player_two": candidate_player_two,
        "games": profile.pairs * 2,
    }
    if any(summary[key] != value for key, value in recomputed.items()):
        raise SelectionError("gate summary disagrees with its full transcript")
    if summary["candidate"] + summary["baseline"] + summary[
            "unfinished"] != summary["games"]:
        raise SelectionError("gate outcome totals are inconsistent")
    if (
        summary["profile"] != f"{profile.first_ms}/{profile.later_ms}"
        or summary["shuffle_seed_policy"] != "deployment-constant"
        or summary["required_total"] != profile.minimum_candidate_wins
        or summary["required_per_color"] != profile.minimum_wins_per_color
        or summary["candidate_decisions"] <= 0
        or summary["baseline_decisions"] <= 0
    ):
        raise SelectionError("gate execution profile is stale")
    passed = (
        summary["unfinished"] == 0
        and summary["candidate_headroom_failures"] == 0
        and summary["baseline_headroom_failures"] == 0
        and summary["candidate_operational_timeouts"] == 0
        and summary["baseline_operational_timeouts"] == 0
        and summary["candidate"] >= profile.minimum_candidate_wins
        and summary["candidate_player_one"]
        >= profile.minimum_wins_per_color
        and summary["candidate_player_two"]
        >= profile.minimum_wins_per_color
    )
    if passed != summary["passed"]:
        raise SelectionError("gate passed flag disagrees with frozen thresholds")
    if exit_code not in (0, 1) or (exit_code == 0) != passed:
        raise SelectionError("gate exit status disagrees with its verified result")
    return summary


def _gate_command(
    gate_binary: pathlib.Path,
    candidate_runtime: pathlib.Path,
    baseline_runtime: pathlib.Path,
    profile: GateProfile,
) -> list[str]:
    return [
        str(gate_binary),
        "--candidate-checkpoint", str(candidate_runtime),
        "--baseline-checkpoint", str(baseline_runtime),
        "--pairs", str(profile.pairs),
        "--first-ms", str(profile.first_ms),
        "--later-ms", str(profile.later_ms),
        "--maximum-turns", str(MAXIMUM_TURNS),
        "--opening-turns", ",".join(map(str, OPENING_TURNS)),
        "--seed", str(OPENING_SEED),
        "--minimum-candidate-wins", str(profile.minimum_candidate_wins),
        "--minimum-wins-per-color", str(profile.minimum_wins_per_color),
    ]


@contextlib.contextmanager
def _serial_gate_lock() -> Iterable[None]:
    lock_path = ROOT / "build" / ".jacek-native-round2-actual-clock.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SelectionError(
                "another actual-clock native gate is running; parallel timing "
                "evidence is invalid"
            ) from error
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _write_exclusive(path: pathlib.Path, raw: bytes) -> None:
    try:
        with path.open("xb") as output:
            output.write(raw)
    except FileExistsError as error:
        raise SelectionError(f"refusing to overwrite immutable evidence: {path}") from error


def record_gate(
    *,
    profile_name: str,
    model_path: pathlib.Path,
    seed: int,
    candidate_runtime: pathlib.Path,
    baseline_model: pathlib.Path,
    baseline_seed: int,
    baseline_runtime: pathlib.Path,
    gate_binary: pathlib.Path,
    output_dir: pathlib.Path,
) -> pathlib.Path:
    profile = PROFILES[profile_name]
    candidate, _ = _round2_identity(model_path, seed, candidate_runtime)
    baseline = _round1_identity(
        baseline_model, baseline_seed, baseline_runtime
    )
    if candidate["packed_sha256"] == baseline["packed_sha256"]:
        raise SelectionError("candidate and baseline packed models are identical")
    if not gate_binary.is_file() or not os.access(gate_binary, os.X_OK):
        raise SelectionError("gate binary does not exist or is not executable")
    gate_binary_raw = gate_binary.read_bytes()
    command = _gate_command(
        gate_binary.resolve(), candidate_runtime.resolve(),
        baseline_runtime.resolve(), profile,
    )
    with _serial_gate_lock():
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            check=False,
        )
    if completed.stderr:
        raise SelectionError("gate wrote stderr; evidence is not a clean match run")
    summary = parse_gate_stdout(
        completed.stdout, profile, candidate, baseline, completed.returncode
    )
    tool_sha = _sha256(pathlib.Path(__file__).read_bytes())
    round1_exporter_sha = _sha256(pathlib.Path(round1_exporter.__file__).read_bytes())
    round2_exporter_sha = _sha256(pathlib.Path(round2_exporter.__file__).read_bytes())
    stdout_sha = _sha256(completed.stdout)
    stdout_name = f"{stdout_sha}.stdout.txt"
    report = {
        "schema": REPORT_SCHEMA,
        "profile": profile.payload(),
        "candidate": candidate,
        "baseline": baseline,
        "execution": {
            "gate_binary_sha256": _sha256(gate_binary_raw),
            "gate_binary_bytes": len(gate_binary_raw),
            "gate_sources": _source_identities(),
            "selection_tool_sha256": tool_sha,
            "round1_exporter_sha256": round1_exporter_sha,
            "round2_exporter_sha256": round2_exporter_sha,
            "serial_actual_clock_lock": True,
            "command": [
                "$GATE_BINARY",
                "--candidate-checkpoint", "$CANDIDATE_RUNTIME",
                "--baseline-checkpoint", "$BASELINE_RUNTIME",
                *command[5:],
            ],
            "exit_code": completed.returncode,
        },
        "stdout": {
            "path": stdout_name,
            "sha256": stdout_sha,
            "bytes": len(completed.stdout),
        },
        "result": summary,
    }
    report_raw = canonical_json_bytes(report)
    report_sha = _sha256(report_raw)
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / stdout_name
    report_path = output_dir / f"{report_sha}.json"
    if stdout_path.exists() or report_path.exists():
        raise SelectionError("refusing to overwrite immutable gate evidence")
    _write_exclusive(stdout_path, completed.stdout)
    try:
        _write_exclusive(report_path, report_raw)
    except Exception:
        stdout_path.unlink(missing_ok=True)
        raise
    return report_path


def _report_paths(paths: Sequence[pathlib.Path]) -> list[pathlib.Path]:
    result: list[pathlib.Path] = []
    for path in paths:
        if path.is_dir():
            result.extend(sorted(path.glob("*.json")))
        else:
            result.append(path)
    resolved = [path.resolve() for path in result]
    if not resolved or len(set(resolved)) != len(resolved):
        raise SelectionError("gate report set is empty or duplicated")
    return resolved


def _validate_report_file(
    path: pathlib.Path,
    model_sha: str,
    expected_candidates: Mapping[int, Mapping[str, Any]],
    baseline: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    raw, report = _load_canonical(path, "gate report")
    report_sha = _sha256(raw)
    if path.name != f"{report_sha}.json":
        raise SelectionError("gate report filename is not content-addressed")
    if not isinstance(report, dict) or set(report) != {
        "schema", "profile", "candidate", "baseline", "execution",
        "stdout", "result",
    } or report.get("schema") != REPORT_SCHEMA:
        raise SelectionError("gate report schema is not frozen")
    profile_payload = report.get("profile")
    if not isinstance(profile_payload, dict):
        raise SelectionError("gate report profile is missing")
    profile_name = profile_payload.get("name")
    if profile_name not in PROFILES or profile_payload != PROFILES[
            profile_name].payload():
        raise SelectionError("gate report profile is not frozen")
    candidate = report.get("candidate")
    if not isinstance(candidate, dict):
        raise SelectionError("gate candidate identity is missing")
    seed = _integer(candidate.get("seed"), "gate candidate seed")
    expected_candidate = expected_candidates.get(seed)
    if expected_candidate is None or candidate != expected_candidate:
        raise SelectionError("gate candidate identity is stale")
    if candidate.get("model_sha256") != model_sha:
        raise SelectionError("gate report names a different training model")
    if report.get("baseline") != baseline:
        raise SelectionError("gate baseline identity is stale")

    execution = report.get("execution")
    if not isinstance(execution, dict) or set(execution) != {
        "gate_binary_sha256", "gate_binary_bytes", "gate_sources",
        "selection_tool_sha256", "round1_exporter_sha256",
        "round2_exporter_sha256", "serial_actual_clock_lock", "command",
        "exit_code",
    }:
        raise SelectionError("gate execution identity is malformed")
    if (
        not _valid_sha256(execution.get("gate_binary_sha256"))
        or _integer(execution.get("gate_binary_bytes"), "gate binary bytes", 1) <= 0
        or execution.get("gate_sources") != _source_identities()
        or execution.get("selection_tool_sha256")
        != _sha256(pathlib.Path(__file__).read_bytes())
        or execution.get("round1_exporter_sha256")
        != _sha256(pathlib.Path(round1_exporter.__file__).read_bytes())
        or execution.get("round2_exporter_sha256")
        != _sha256(pathlib.Path(round2_exporter.__file__).read_bytes())
        or execution.get("serial_actual_clock_lock") is not True
    ):
        raise SelectionError("gate execution provenance is stale")
    expected_command_tail = _gate_command(
        pathlib.Path("$GATE_BINARY"), pathlib.Path("$CANDIDATE_RUNTIME"),
        pathlib.Path("$BASELINE_RUNTIME"), PROFILES[profile_name],
    )
    if execution.get("command") != expected_command_tail:
        raise SelectionError("gate command is not the frozen profile")
    exit_code = _integer(execution.get("exit_code"), "gate exit code")

    stdout = report.get("stdout")
    if not isinstance(stdout, dict) or set(stdout) != {"path", "sha256", "bytes"}:
        raise SelectionError("gate stdout identity is malformed")
    stdout_sha = stdout.get("sha256")
    if not _valid_sha256(stdout_sha) or stdout.get("path") != (
            f"{stdout_sha}.stdout.txt"):
        raise SelectionError("gate stdout is not content-addressed")
    stdout_path = path.parent / stdout["path"]
    stdout_raw = stdout_path.read_bytes()
    if _sha256(stdout_raw) != stdout_sha or len(stdout_raw) != stdout.get("bytes"):
        raise SelectionError("gate stdout bytes do not match their identity")
    parsed = parse_gate_stdout(
        stdout_raw, PROFILES[profile_name], candidate, baseline, exit_code
    )
    if report.get("result") != parsed:
        raise SelectionError("gate result does not match its full stdout")
    return report, report_sha


def _validation_mse(model: Mapping[str, Any], seed: int) -> float:
    reports = model.get("seed_reports")
    if not isinstance(reports, list):
        raise SelectionError("round-two model seed reports are missing")
    matches = [
        report for report in reports
        if isinstance(report, Mapping) and report.get("seed") == seed
    ]
    if len(matches) != 1:
        raise SelectionError(f"seed {seed} has no unique training report")
    try:
        value = float(
            matches[0]["quantized_metrics"]["validation"]["outcome_mse"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SelectionError(f"seed {seed} validation MSE is missing") from error
    if not math.isfinite(value) or value < 0.0:
        raise SelectionError(f"seed {seed} validation MSE is invalid")
    return value


def _selection_payload_hash(sidecar: Mapping[str, Any]) -> str:
    payload = dict(sidecar)
    payload.pop("selection_payload_sha256", None)
    return _sha256(canonical_json_bytes(payload))


def finalize_selection(
    *,
    model_path: pathlib.Path,
    baseline_model: pathlib.Path,
    baseline_seed: int,
    baseline_runtime: pathlib.Path,
    report_paths: Sequence[pathlib.Path],
    output: pathlib.Path,
) -> dict[str, Any]:
    if output.exists():
        raise SelectionError("refusing to overwrite immutable selection sidecar")
    model_raw, model = _load_canonical(model_path, "round-two model")
    if not isinstance(model, Mapping):
        raise SelectionError("round-two model root is not an object")
    training = model.get("training")
    if not isinstance(training, Mapping) or training.get("chosen_seed") is not None:
        raise SelectionError("round-two model is not immutable pending evidence")
    seeds_value = training.get("seeds")
    if (
        not isinstance(seeds_value, list)
        or not seeds_value
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
               for seed in seeds_value)
        or len(set(seeds_value)) != len(seeds_value)
    ):
        raise SelectionError("round-two retained seed set is malformed")
    seeds = list(seeds_value)
    model_sha = _sha256(model_raw)
    expected_candidates: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        seed_metadata = _model_seed_metadata(model, seed)
        try:
            runtime_raw = round2_exporter.render_runtime(
                model, model_sha, seed
            ).encode()
        except (KeyError, TypeError, ValueError) as error:
            raise SelectionError(f"seed {seed} cannot be exported exactly") from error
        runtime = _runtime_lines(runtime_raw, f"candidate seed {seed}")
        expected_candidates[seed] = {
            **seed_metadata,
            "model_sha256": model_sha,
            "runtime_sha256": _sha256(runtime_raw),
            "packed_sha256": runtime["packed_sha256"],
            "runtime_bytes": len(runtime_raw),
        }
    baseline = _round1_identity(
        baseline_model, baseline_seed, baseline_runtime
    )

    reports: dict[tuple[int, str], tuple[dict[str, Any], str]] = {}
    for path in _report_paths(report_paths):
        report, report_sha = _validate_report_file(
            path, model_sha, expected_candidates, baseline
        )
        key = (report["candidate"]["seed"], report["profile"]["name"])
        if key in reports:
            raise SelectionError("duplicate gate report for one seed/profile")
        reports[key] = (report, report_sha)
    expected_keys = {
        (seed, profile) for seed in seeds for profile in PROFILES
    }
    if set(reports) != expected_keys:
        missing = sorted(expected_keys - set(reports))
        extra = sorted(set(reports) - expected_keys)
        raise SelectionError(
            f"gate report coverage is incomplete (missing={missing}, extra={extra})"
        )

    ranking_rows = []
    for seed in seeds:
        screen = reports[(seed, "screen")][0]["result"]
        decisive = reports[(seed, "decisive")][0]["result"]
        passed = bool(screen["passed"] and decisive["passed"])
        ranking_rows.append({
            "seed": seed,
            "passed": passed,
            "decisive_wins": decisive["candidate"],
            "decisive_worst_color_wins": min(
                decisive["candidate_player_one"],
                decisive["candidate_player_two"],
            ),
            "screen_wins": screen["candidate"],
            "screen_worst_color_wins": min(
                screen["candidate_player_one"],
                screen["candidate_player_two"],
            ),
            "quantized_validation_outcome_mse": _validation_mse(model, seed),
        })
    passing = [row for row in ranking_rows if row["passed"]]
    if not passing:
        raise SelectionError("no retained seed passed both frozen actual-clock gates")
    order_key = lambda row: (
        -row["decisive_wins"],
        -row["decisive_worst_color_wins"],
        -row["screen_wins"],
        -row["screen_worst_color_wins"],
        row["quantized_validation_outcome_mse"],
        row["seed"],
    )
    ranked = sorted(passing, key=order_key)
    selected_seed = ranked[0]["seed"]
    selected = expected_candidates[selected_seed]
    report_index = [{
        "seed": seed,
        "profile": profile,
        "report_sha256": reports[(seed, profile)][1],
        "stdout_sha256": reports[(seed, profile)][0]["stdout"]["sha256"],
        "runtime_sha256": expected_candidates[seed]["runtime_sha256"],
        "passed": reports[(seed, profile)][0]["result"]["passed"],
    } for seed in sorted(seeds) for profile in sorted(PROFILES)]
    sidecar: dict[str, Any] = {
        "schema": SELECTION_SCHEMA,
        "model": {
            "sha256": model_sha,
            "bytes": len(model_raw),
            "chosen_seed_in_model": None,
        },
        "baseline": baseline,
        "profiles": {
            name: PROFILES[name].payload() for name in sorted(PROFILES)
        },
        "reports": report_index,
        "ranking": {
            "order": [
                "decisive-wins-descending",
                "decisive-worst-color-wins-descending",
                "screen-wins-descending",
                "screen-worst-color-wins-descending",
                "quantized-validation-outcome-mse-ascending",
                "seed-ascending",
            ],
            "passing_seeds": ranked,
        },
        "selected": {
            **selected,
            "tested_runtime_sha256": selected["runtime_sha256"],
            "deployment_runtime_sha256": selected["runtime_sha256"],
            "exact_tested_deployed_runtime": True,
        },
        "selection_tool_sha256": _sha256(pathlib.Path(__file__).read_bytes()),
        "round2_exporter_sha256": _sha256(
            pathlib.Path(round2_exporter.__file__).read_bytes()
        ),
    }
    sidecar["selection_payload_sha256"] = _selection_payload_hash(sidecar)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_exclusive(output, canonical_json_bytes(sidecar))
    return sidecar


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record/finalize immutable Jacek-native round-two gates."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    record = commands.add_parser("record", help="run one frozen actual-clock gate")
    record.add_argument("--profile", choices=sorted(PROFILES), required=True)
    record.add_argument("--model", type=pathlib.Path, required=True)
    record.add_argument("--seed", type=int, required=True)
    record.add_argument("--candidate-runtime", type=pathlib.Path, required=True)
    record.add_argument("--baseline-model", type=pathlib.Path, required=True)
    record.add_argument("--baseline-seed", type=int, required=True)
    record.add_argument("--baseline-runtime", type=pathlib.Path, required=True)
    record.add_argument("--gate-binary", type=pathlib.Path, required=True)
    record.add_argument("--output-dir", type=pathlib.Path, required=True)

    finalize = commands.add_parser(
        "finalize", help="create the immutable selected-seed sidecar"
    )
    finalize.add_argument("--model", type=pathlib.Path, required=True)
    finalize.add_argument("--baseline-model", type=pathlib.Path, required=True)
    finalize.add_argument("--baseline-seed", type=int, required=True)
    finalize.add_argument("--baseline-runtime", type=pathlib.Path, required=True)
    finalize.add_argument("--reports", nargs="+", type=pathlib.Path, required=True)
    finalize.add_argument("--output", type=pathlib.Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "record":
            path = record_gate(
                profile_name=arguments.profile,
                model_path=arguments.model,
                seed=arguments.seed,
                candidate_runtime=arguments.candidate_runtime,
                baseline_model=arguments.baseline_model,
                baseline_seed=arguments.baseline_seed,
                baseline_runtime=arguments.baseline_runtime,
                gate_binary=arguments.gate_binary,
                output_dir=arguments.output_dir,
            )
            print(json.dumps({
                "report": str(path),
                "report_sha256": path.stem,
            }, indent=2, sort_keys=True))
        else:
            sidecar = finalize_selection(
                model_path=arguments.model,
                baseline_model=arguments.baseline_model,
                baseline_seed=arguments.baseline_seed,
                baseline_runtime=arguments.baseline_runtime,
                report_paths=arguments.reports,
                output=arguments.output,
            )
            print(json.dumps({
                "output": str(arguments.output),
                "selected_seed": sidecar["selected"]["seed"],
                "selection_payload_sha256": sidecar[
                    "selection_payload_sha256"
                ],
                "runtime_sha256": sidecar["selected"]["runtime_sha256"],
            }, indent=2, sort_keys=True))
    except (OSError, SelectionError, subprocess.SubprocessError) as error:
        print(f"round-two selection failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
