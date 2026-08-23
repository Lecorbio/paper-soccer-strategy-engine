#!/usr/bin/env python3
"""Apply the preregistered offline strength gate to a comparison report."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

import jacek_replay_provenance as provenance
import jacek_replay_tuning_gate as tuning_gate


SCHEMA = "papersoccer.jacek-replay-bfm-comparison.v1"
TUNING_SCHEMA = "papersoccer.jacek-replay-bfm-tuning-receipt.v1"
BASELINE_SCHEMA = "papersoccer.jacek-replay-bfm-baseline-gate.v1"
Z_ONE_SIDED_95 = 1.6448536269514722
FINAL_GAMES_PER_OPPONENT = 1_000
FINAL_MAXIMUM_MS = 1_000.0
FINAL_OPENING_PLIES = 12
FINAL_BANK_SEED = 987_654_321
FINAL_MAX_TURNS = 320
DEVELOPMENT_PAIRS = 200
DEVELOPMENT_TIME_MS = 20
DEVELOPMENT_BANK_SEED = 123_456_789
ATTEMPT_SCHEMA = "papersoccer.jacek-replay-bfm-final-attempt.v1"
DECISION_SCHEMA = "papersoccer.jacek-replay-bfm-promotion-decision.v1"
PUBLISHED_SCHEMA = "papersoccer.jacek-replay-bfm-published-model.v1"
FINAL_PAIRS = FINAL_GAMES_PER_OPPONENT // 2


def valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def file_sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json_bytes(value: object, *, pretty: bool = False) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
            allow_nan=False,
        ).encode()
        + b"\n"
    )


def atomic_json(path: pathlib.Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: pathlib.Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as output:
            output.write(canonical_json_bytes(value, pretty=True))
            output.flush()
            os.fsync(output.fileno())
            temporary = pathlib.Path(output.name)
        os.replace(temporary, path)
        os.chmod(path, 0o644)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def load_canonical_json(path: pathlib.Path, label: str) -> dict:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or raw != canonical_json_bytes(value, pretty=True):
        raise ValueError(f"{label} is not canonical JSON")
    return value


def bank_opening_ids(path: pathlib.Path) -> set[str]:
    result = set()
    transcripts = set()
    for line in path.read_text().splitlines():
        if (
            not line
            or line.startswith("#")
            or line == "opening_id\ttranscript\tstate_identity"
        ):
            continue
        fields = line.split("\t")
        if len(fields) != 3:
            raise ValueError("opening bank row has the wrong field count")
        opening, transcript, _ = fields
        if (
            not opening
            or opening in result
            or transcript in transcripts
        ):
            raise ValueError("opening bank contains an invalid or duplicate row")
        result.add(opening)
        transcripts.add(transcript)
    if not result:
        raise ValueError("opening bank is empty")
    return result


def bank_transcript_hashes(path: pathlib.Path) -> set[str]:
    hashes = set()
    for line in path.read_text().splitlines():
        if (
            not line
            or line.startswith("#")
            or line == "opening_id\ttranscript\tstate_identity"
        ):
            continue
        fields = line.split("\t")
        if len(fields) != 3:
            raise ValueError("opening bank row is malformed")
        transcript = fields[1]
        digest = hashlib.sha256(transcript.encode()).hexdigest()
        if digest in hashes:
            raise ValueError("opening bank repeats a transcript")
        hashes.add(digest)
    return hashes


def bank_state_identities(path: pathlib.Path) -> set[str]:
    identities = set()
    for line in path.read_text().splitlines():
        if (
            not line
            or line.startswith("#")
            or line == "opening_id\ttranscript\tstate_identity"
        ):
            continue
        fields = line.split("\t")
        if len(fields) != 3 or not fields[2] or fields[2] in identities:
            raise ValueError("opening bank repeats or omits a state identity")
        identities.add(fields[2])
    return identities


def bank_classification(path: pathlib.Path) -> str | None:
    prefix = "# classification="
    values = [
        line[len(prefix) :]
        for line in path.read_text().splitlines()
        if line.startswith(prefix)
    ]
    return values[0] if len(values) == 1 else None


def validate_bank_metadata(path: pathlib.Path, expected: str) -> tuple[int, int]:
    lines = path.read_text().splitlines()
    if (
        len(lines) < 7
        or lines[0] != "# papersoccer.jacek-replay-bfm-opening-bank.v1"
        or lines[1] != "# rules=8x10;own-goals-allowed;mover-loses"
        or lines[2] != f"# classification={expected}"
        or not lines[3].startswith("# seed=")
        or not lines[4].startswith("# minimum-physical-plies=")
        or lines[5] != "opening_id\ttranscript\tstate_identity"
        or any(not value.isdecimal() for value in (lines[3][7:], lines[4][25:]))
        or any(not line or line.startswith("#") for line in lines[6:])
    ):
        raise ValueError("opening bank metadata contract is invalid")
    return int(lines[3][7:]), int(lines[4][25:])


def shared_core_sha256() -> str:
    return provenance.shared_core_sha256()


def game_timing_samples(game: dict, index: int) -> list[float]:
    raw = game.get("candidate_ms")
    if not isinstance(raw, list):
        raise ValueError(f"game {index} omits candidate timings")
    samples: list[float] = []
    for value in raw:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"game {index} has an invalid candidate timing")
        samples.append(float(value))
    return samples


def verified_tuning_receipt(
    tuning_path: pathlib.Path, baseline_path: pathlib.Path
) -> tuple[dict, dict]:
    tuning_path = tuning_path.resolve()
    baseline_path = baseline_path.resolve()
    tuning = json.loads(tuning_path.read_bytes())
    baseline = json.loads(baseline_path.read_bytes())
    if (
        not isinstance(tuning, dict)
        or tuning.get("schema") != TUNING_SCHEMA
        or not isinstance(tuning.get("reports"), dict)
    ):
        raise ValueError("final attempt tuning receipt is invalid")
    reports = tuning["reports"]
    rebuilt = tuning_gate.select(
        [pathlib.Path(reports[str(value)]["path"]) for value in tuning_gate.GRID],
        baseline_path,
    )
    if rebuilt != tuning:
        raise ValueError("final attempt tuning receipt does not recompute")
    tuning_gate.validate_baseline_receipt(
        baseline, tuning.get("binding", {}).get("model_sha256"), verify_files=True
    )
    return tuning, baseline


def attempt_binding(
    *,
    tuning_path: pathlib.Path,
    baseline_path: pathlib.Path,
    model_path: pathlib.Path,
    executable_path: pathlib.Path,
    bank_path: pathlib.Path,
    report_path: pathlib.Path,
) -> dict:
    tuning, _ = verified_tuning_receipt(tuning_path, baseline_path)
    binding = tuning["binding"]
    paths = {
        "tuning_receipt_path": tuning_path.resolve(),
        "baseline_receipt_path": baseline_path.resolve(),
        "model_path": model_path.resolve(),
        "comparison_executable_path": executable_path.resolve(),
        "opening_bank_path": bank_path.resolve(),
        "comparison_report_path": report_path.resolve(),
    }
    for label in (
        "tuning_receipt_path",
        "baseline_receipt_path",
        "model_path",
        "comparison_executable_path",
    ):
        if not paths[label].is_file():
            raise ValueError(f"final attempt {label} is unavailable")
    expected = {
        "model_sha256": binding.get("model_sha256"),
        "baseline_receipt_sha256": binding.get("baseline_receipt_sha256"),
        "comparison_executable_sha256": binding.get(
            "comparison_executable_sha256"
        ),
    }
    actual = {
        "model_sha256": file_sha256(paths["model_path"]),
        "baseline_receipt_sha256": file_sha256(paths["baseline_receipt_path"]),
        "comparison_executable_sha256": file_sha256(
            paths["comparison_executable_path"]
        ),
    }
    if actual != expected:
        raise ValueError("final attempt inputs differ from the tuning receipt")
    if pathlib.Path(binding.get("comparison_executable_path", "")).resolve() != paths[
        "comparison_executable_path"
    ]:
        raise ValueError("final attempt comparison executable path changed")
    if file_sha256(paths["tuning_receipt_path"]) != hashlib.sha256(
        canonical_json_bytes(tuning, pretty=True)
    ).hexdigest():
        # Tuning receipts are emitted as pretty, canonical JSON. Refuse an
        # equivalent reserialization because the comparison binds raw bytes.
        raise ValueError("final attempt tuning receipt is not canonical")
    return {
        **{name: str(path) for name, path in paths.items()},
        "promotion_gate_sha256": file_sha256(pathlib.Path(__file__)),
        "tuning_receipt_sha256": file_sha256(paths["tuning_receipt_path"]),
        **actual,
        "chosen_exploration": tuning["chosen_exploration"],
        "search_seed": binding["seed"],
        "pairs": FINAL_PAIRS,
        "opening_plies": FINAL_OPENING_PLIES,
        "opening_bank_seed": FINAL_BANK_SEED,
        "time_ms": 980,
        "max_turns": FINAL_MAX_TURNS,
        "candidate_tree_nodes": 1_000_000,
        "control_tree_nodes": 100_000,
        "control_work": 3_000_000,
        "max_actions": 250,
        "max_partial_paths": 50_000,
        "fpu": 0.5,
    }


def prepare_final_attempt(
    ledger_path: pathlib.Path,
    *,
    tuning_path: pathlib.Path,
    baseline_path: pathlib.Path,
    model_path: pathlib.Path,
    executable_path: pathlib.Path,
    bank_path: pathlib.Path,
    report_path: pathlib.Path,
) -> dict:
    ledger_path = ledger_path.resolve()
    bank_path = bank_path.resolve()
    report_path = report_path.resolve()
    if len({ledger_path, bank_path, report_path}) != 3:
        raise ValueError("ledger, final bank, and report paths must be distinct")
    if ledger_path.exists():
        raise ValueError("final-attempt ledger already exists")
    if bank_path.exists() or report_path.exists():
        raise ValueError("final bank and report must not exist before preparation")
    binding = attempt_binding(
        tuning_path=tuning_path,
        baseline_path=baseline_path,
        model_path=model_path,
        executable_path=executable_path,
        bank_path=bank_path,
        report_path=report_path,
    )
    attempt_id = hashlib.sha256(canonical_json_bytes(binding)).hexdigest()
    ledger = {
        "schema": ATTEMPT_SCHEMA,
        "attempt_id": attempt_id,
        "state": "prepared",
        "binding": binding,
        "prepared": {"at_utc": utc_now()},
        "bank_registration": None,
        "report_consumption": None,
    }
    atomic_json(ledger_path, ledger)
    return ledger


def load_attempt(ledger_path: pathlib.Path) -> tuple[dict, str]:
    ledger_path = ledger_path.resolve()
    ledger = load_canonical_json(ledger_path, "final-attempt ledger")
    if (
        ledger.get("schema") != ATTEMPT_SCHEMA
        or not valid_sha256(ledger.get("attempt_id"))
        or not isinstance(ledger.get("binding"), dict)
    ):
        raise ValueError("final-attempt ledger schema is invalid")
    return ledger, file_sha256(ledger_path)


def verify_live_attempt_binding(ledger: dict) -> tuple[dict, dict]:
    raw = ledger["binding"]
    rebuilt = attempt_binding(
        tuning_path=pathlib.Path(raw["tuning_receipt_path"]),
        baseline_path=pathlib.Path(raw["baseline_receipt_path"]),
        model_path=pathlib.Path(raw["model_path"]),
        executable_path=pathlib.Path(raw["comparison_executable_path"]),
        bank_path=pathlib.Path(raw["opening_bank_path"]),
        report_path=pathlib.Path(raw["comparison_report_path"]),
    )
    if rebuilt != raw or hashlib.sha256(canonical_json_bytes(raw)).hexdigest() != ledger[
        "attempt_id"
    ]:
        raise ValueError("final-attempt live binding changed")
    return verified_tuning_receipt(
        pathlib.Path(raw["tuning_receipt_path"]),
        pathlib.Path(raw["baseline_receipt_path"]),
    )


def register_final_bank(ledger_path: pathlib.Path) -> dict:
    ledger, prepared_hash = load_attempt(ledger_path)
    if ledger.get("state") != "prepared":
        raise ValueError("final bank can be registered only once after preparation")
    tuning, _ = verify_live_attempt_binding(ledger)
    binding = ledger["binding"]
    bank_path = pathlib.Path(binding["opening_bank_path"])
    if pathlib.Path(binding["comparison_report_path"]).exists():
        raise ValueError("final report exists before bank registration")
    seed, plies = validate_bank_metadata(bank_path, "final")
    ids = bank_opening_ids(bank_path)
    states = bank_state_identities(bank_path)
    transcripts = bank_transcript_hashes(bank_path)
    if (
        seed != FINAL_BANK_SEED
        or plies != FINAL_OPENING_PLIES
        or len(ids) != FINAL_PAIRS
        or len(states) != FINAL_PAIRS
        or len(transcripts) != FINAL_PAIRS
    ):
        raise ValueError("final bank does not match the frozen 500-pair panel")
    development_states = set(tuning.get("binding", {}).get(
        "opening_state_identities", []
    ))
    development_transcripts = set(tuning.get("opening_transcript_sha256", []))
    if states.intersection(development_states) or transcripts.intersection(
        development_transcripts
    ):
        raise ValueError("development and final banks are not disjoint")
    ledger["state"] = "bank-registered"
    ledger["bank_registration"] = {
        "at_utc": utc_now(),
        "prepared_ledger_sha256": prepared_hash,
        "opening_bank_sha256": file_sha256(bank_path),
        "opening_state_identities_sha256": hashlib.sha256(
            "\n".join(sorted(states)).encode()
        ).hexdigest(),
        "opening_transcript_identities_sha256": hashlib.sha256(
            "\n".join(sorted(transcripts)).encode()
        ).hexdigest(),
    }
    atomic_json(ledger_path.resolve(), ledger)
    return ledger


def consume_final_report(ledger_path: pathlib.Path) -> tuple[dict, dict, str]:
    ledger, registered_hash = load_attempt(ledger_path)
    binding = ledger["binding"]
    report_path = pathlib.Path(binding["comparison_report_path"])
    state = ledger.get("state")
    if state not in ("running-consumed", "consumed"):
        raise ValueError(
            "final report can be consumed only after the bound comparison launch"
        )
    if not report_path.is_file():
        raise ValueError("bound comparison did not publish a final report")
    report_hash = file_sha256(report_path)
    if state == "consumed":
        consumption = ledger.get("report_consumption", {})
        if (
            consumption.get("comparison_report_sha256") != report_hash
            or consumption.get("comparison_report_path") != str(report_path)
        ):
            raise ValueError("consumed final report was replaced")
        return ledger, json.loads(report_path.read_bytes()), registered_hash
    verify_live_attempt_binding(ledger)
    registration = ledger.get("bank_registration")
    if (
        not isinstance(registration, dict)
        or registration.get("opening_bank_sha256")
        != file_sha256(pathlib.Path(binding["opening_bank_path"]))
    ):
        raise ValueError("registered final bank changed")
    payload = json.loads(report_path.read_bytes())
    previous_consumption = ledger.get("report_consumption")
    ledger["state"] = "consumed"
    ledger["report_consumption"] = {
        "at_utc": utc_now(),
        "started_at_utc": (
            previous_consumption.get("started_at_utc")
            if isinstance(previous_consumption, dict)
            else None
        ),
        "bank_registered_ledger_sha256": (
            previous_consumption.get("bank_registered_ledger_sha256")
            if isinstance(previous_consumption, dict)
            else registered_hash
        ),
        "comparison_started_ledger_sha256": registered_hash,
        "comparison_report_path": str(report_path),
        "comparison_report_sha256": report_hash,
    }
    atomic_json(ledger_path.resolve(), ledger)
    return ledger, payload, file_sha256(ledger_path.resolve())


def run_final_attempt(ledger_path: pathlib.Path) -> tuple[dict, dict]:
    """Run the one supported final comparison and immediately consume its bank."""

    ledger, _ = load_attempt(ledger_path)
    if ledger.get("state") == "consumed":
        consumed, payload, _ = consume_final_report(ledger_path)
        return consumed, payload
    if ledger.get("state") == "running-consumed":
        report_path = pathlib.Path(
            ledger["binding"]["comparison_report_path"]
        )
        if not report_path.is_file():
            raise ValueError(
                "interrupted protected comparison has no complete report; rerun is forbidden"
            )
        consumed, payload, _ = consume_final_report(ledger_path)
        return consumed, payload
    if ledger.get("state") != "bank-registered":
        raise ValueError("final comparison requires a registered unused bank")
    verify_live_attempt_binding(ledger)
    binding = ledger["binding"]
    report_path = pathlib.Path(binding["comparison_report_path"])
    if report_path.exists():
        ledger["state"] = "failed-consumed"
        ledger["report_consumption"] = {
            "at_utc": utc_now(),
            "reason": "report-existed-before-bound-comparison-launch",
            "preexisting_report_path": str(report_path),
            "preexisting_report_sha256": (
                file_sha256(report_path) if report_path.is_file() else None
            ),
        }
        atomic_json(ledger_path.resolve(), ledger)
        raise ValueError(
            "final report existed before the bound comparison launch; "
            "protected attempt is consumed"
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
            binding["comparison_executable_path"],
            "--model",
            binding["model_path"],
            "--opponent",
            "both",
            "--bank",
            binding["opening_bank_path"],
            "--bank-classification",
            "final",
            "--baseline-receipt",
            binding["baseline_receipt_path"],
            "--tuning-receipt",
            binding["tuning_receipt_path"],
            "--pairs",
            str(binding["pairs"]),
            "--opening-plies",
            str(binding["opening_plies"]),
            "--max-turns",
            str(binding["max_turns"]),
            "--seed",
            str(binding["search_seed"]),
            "--time-ms",
            str(binding["time_ms"]),
            "--control-work",
            str(binding["control_work"]),
            "--tree-nodes",
            str(binding["candidate_tree_nodes"]),
            "--control-tree-nodes",
            str(binding["control_tree_nodes"]),
            "--max-actions",
            str(binding["max_actions"]),
            "--max-partial-paths",
            str(binding["max_partial_paths"]),
            "--exploration",
            str(binding["chosen_exploration"]),
            "--fpu",
            str(binding["fpu"]),
            "--output",
            str(report_path),
    ]
    registered_hash = file_sha256(ledger_path.resolve())
    ledger["state"] = "running-consumed"
    ledger["report_consumption"] = {
        "started_at_utc": utc_now(),
        "bank_registered_ledger_sha256": registered_hash,
    }
    atomic_json(ledger_path.resolve(), ledger)
    try:
        process = subprocess.run(
            command, capture_output=True, text=True, check=False
        )
    except OSError as error:
        ledger["state"] = "failed-consumed"
        ledger["report_consumption"] = {
            "at_utc": utc_now(),
            "launch_error": type(error).__name__,
            "message_sha256": hashlib.sha256(str(error).encode()).hexdigest(),
        }
        atomic_json(ledger_path.resolve(), ledger)
        raise ValueError(
            "final comparison could not launch; protected attempt is consumed"
        ) from error
    if process.returncode != 0 or not report_path.is_file():
        ledger["state"] = "failed-consumed"
        ledger["report_consumption"] = {
            "at_utc": utc_now(),
            "returncode": process.returncode,
            "stdout_sha256": hashlib.sha256(process.stdout.encode()).hexdigest(),
            "stderr_sha256": hashlib.sha256(process.stderr.encode()).hexdigest(),
        }
        atomic_json(ledger_path.resolve(), ledger)
        raise ValueError(
            "final comparison failed; protected attempt is consumed and cannot rerun"
        )
    consumed, payload, _ = consume_final_report(ledger_path)
    return consumed, payload


def wilson_lower(wins: int, games: int) -> float:
    if games <= 0 or wins < 0 or wins > games:
        raise ValueError("invalid win count")
    p = wins / games
    z2 = Z_ONE_SIDED_95 * Z_ONE_SIDED_95
    denominator = 1.0 + z2 / games
    center = p + z2 / (2.0 * games)
    radius = Z_ONE_SIDED_95 * math.sqrt(
        (p * (1.0 - p) + z2 / (4.0 * games)) / games
    )
    return (center - radius) / denominator


def evaluate(
    payload: dict,
    minimum_games: int,
    maximum_ms: float,
    *,
    verify_files: bool = True,
    tuning_receipt: dict | None = None,
    baseline_receipt: dict | None = None,
    report_path: pathlib.Path | None = None,
    attempt_ledger: dict | None = None,
    attempt_ledger_path: pathlib.Path | None = None,
) -> dict:
    if minimum_games != FINAL_GAMES_PER_OPPONENT or not math.isclose(
        maximum_ms, FINAL_MAXIMUM_MS, rel_tol=0.0, abs_tol=0.0
    ):
        raise ValueError("promotion thresholds are frozen at 1000 games and 1000 ms")
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported comparison schema")
    if report_path is None:
        if verify_files:
            raise ValueError("promotion decision requires the raw comparison report")
        report_identity = {
            "path": "<in-memory>",
            "sha256": hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
        }
    else:
        report_path = report_path.resolve()
        raw_report = report_path.read_bytes()
        if json.loads(raw_report) != payload:
            raise ValueError("loaded comparison report differs from supplied payload")
        report_identity = {
            "path": str(report_path),
            "sha256": hashlib.sha256(raw_report).hexdigest(),
        }
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("comparison report has no games")

    errors: list[str] = []
    grouped: dict[str, list[dict]] = {"rank4": [], "neural-puct": []}
    identities: set[tuple[str, str, int]] = set()
    timing_samples: list[float] = []
    for index, game in enumerate(results):
        if not isinstance(game, dict):
            raise ValueError(f"game {index} is not an object")
        opponent = game.get("opponent")
        if opponent not in grouped:
            raise ValueError(f"game {index} has an unknown opponent")
        opening = game.get("opening")
        color = game.get("candidate_player")
        if (
            not isinstance(opening, str)
            or not opening
            or isinstance(color, bool)
            or type(color) is not int
            or color not in (0, 1)
        ):
            raise ValueError(f"game {index} has an invalid opening or color")
        identity = (opening, opponent, int(color))
        if identity in identities:
            errors.append(f"duplicate game identity: {identity}")
        identities.add(identity)
        grouped[opponent].append(game)
        if game.get("illegal") is not False:
            errors.append(f"{opponent}: game {index} is illegal")
        winner = game.get("winner")
        if isinstance(winner, bool) or type(winner) is not int or winner not in (0, 1):
            errors.append(f"{opponent}: game {index} is unfinished")
        timing_samples.extend(game_timing_samples(game, index))

    configuration = payload.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("comparison configuration is missing")
    expected_pairs = minimum_games // 2
    final_state_identities = configuration.get("opening_state_identities")
    if minimum_games % 2 or configuration.get("pairs") != expected_pairs:
        errors.append(f"configuration must bind exactly {expected_pairs} pairs")
    if (
        not isinstance(final_state_identities, list)
        or len(final_state_identities) != expected_pairs
        or len(set(final_state_identities)) != expected_pairs
        or not all(
            isinstance(value, str) and value for value in final_state_identities
        )
    ):
        errors.append("final comparison does not bind unique canonical states")
    if configuration.get("time_ms") != 980:
        errors.append("final comparison must use the frozen 980 ms budget")
    if configuration.get("max_turns") != FINAL_MAX_TURNS:
        errors.append("final comparison must use the frozen 320-turn limit")
    if configuration.get("opening_plies") != FINAL_OPENING_PLIES:
        errors.append("final comparison must use 12-ply openings")
    if configuration.get("opening_bank_seed") != FINAL_BANK_SEED:
        errors.append("final comparison must bind the frozen final-bank seed")
    if (
        configuration.get("opening_bank_minimum_physical_plies")
        != FINAL_OPENING_PLIES
    ):
        errors.append("final opening bank must require 12 physical plies")
    if configuration.get("single_thread") is not True:
        errors.append("final comparison must assert single-thread search")
    opening_source = configuration.get("opening_source")
    opening_hash = configuration.get("opening_bank_sha256")
    if (
        not isinstance(opening_source, str)
        or not opening_source
        or opening_source == "generated"
    ):
        errors.append("final comparison requires a frozen opening bank")
    if configuration.get("opening_bank_classification") != "final":
        errors.append("promotion comparison must use a final-classified bank")
    if not valid_sha256(opening_hash):
        errors.append("opening bank SHA-256 is missing or invalid")
    model_hash = payload.get("model_sha256")
    model_path = payload.get("model")
    if not valid_sha256(model_hash):
        errors.append("model SHA-256 is missing or invalid")
    if not isinstance(model_path, str) or not model_path:
        errors.append("model path is missing")
    rank4_hash = configuration.get("rank4_control_sha256")
    neural_hash = configuration.get("neural_puct_control_sha256")
    rank4_engine_hash = configuration.get("rank4_engine_sha256")
    neural_engine_hash = configuration.get("neural_puct_engine_sha256")
    rank4_adapter_hash = configuration.get("rank4_adapter_sha256")
    neural_adapter_hash = configuration.get("neural_puct_adapter_sha256")
    shared_core_hash = configuration.get("shared_core_sha256")
    candidate_source_hash = configuration.get("candidate_source_sha256")
    comparison_source_hash = configuration.get("comparison_source_sha256")
    comparison_executable_path = configuration.get(
        "comparison_executable_path"
    )
    comparison_executable_hash = configuration.get(
        "comparison_executable_sha256"
    )
    if not all(
        valid_sha256(value)
        for value in (
            rank4_hash,
            neural_hash,
            rank4_engine_hash,
            neural_engine_hash,
            rank4_adapter_hash,
            neural_adapter_hash,
            shared_core_hash,
            candidate_source_hash,
            comparison_source_hash,
            comparison_executable_hash,
        )
    ):
        errors.append("control source hashes are missing or invalid")
    if not isinstance(comparison_executable_path, str) or not comparison_executable_path:
        errors.append("comparison executable path is missing")
    if configuration.get("control_tree_nodes") != 100_000:
        errors.append("neural PUCT control must retain its 100000-node cap")
    if configuration.get("control_work") != 3_000_000:
        errors.append("controls must retain the frozen 3000000 work cap")
    if configuration.get("candidate_tree_nodes") != 1_000_000:
        errors.append("candidate must retain the frozen 1000000-node cap")
    if configuration.get("max_actions") != 250:
        errors.append("candidate must retain 250 complete-turn actions")
    if configuration.get("max_partial_paths") != 50_000:
        errors.append("candidate must retain 50000 partial paths")
    if configuration.get("fpu") != 0.5:
        errors.append("candidate must retain FPU 0.5")
    exploration = configuration.get("exploration")
    if not isinstance(exploration, (int, float)) or not any(
        math.isclose(float(exploration), value, rel_tol=0.0, abs_tol=1e-12)
        for value in (0.25, 0.5, 0.95)
    ):
        errors.append("candidate exploration was not selected from the frozen grid")
    tuning_path = configuration.get("tuning_receipt_path")
    tuning_hash = configuration.get("tuning_receipt_sha256")
    if not isinstance(tuning_path, str) or not valid_sha256(tuning_hash):
        errors.append("final comparison does not bind a tuning receipt")
    if (
        not isinstance(tuning_receipt, dict)
        or tuning_receipt.get("schema") != TUNING_SCHEMA
        or tuning_receipt.get("classification")
        != "development-only-exploration-selection"
    ):
        errors.append("development tuning receipt is missing or invalid")
    else:
        if tuning_receipt.get("chosen_exploration") != exploration:
            errors.append("final exploration differs from development selection")
        tuning_binding = tuning_receipt.get("binding")
        if not isinstance(tuning_binding, dict):
            errors.append("tuning receipt binding is missing")
        else:
            if tuning_binding.get("opening_bank_classification") != "development":
                errors.append("tuning receipt is not bound to a development bank")
            if (
                tuning_binding.get("pairs") != DEVELOPMENT_PAIRS
                or tuning_binding.get("time_ms") != DEVELOPMENT_TIME_MS
                or tuning_binding.get("opening_plies") != FINAL_OPENING_PLIES
                or tuning_binding.get("opening_bank_seed")
                != DEVELOPMENT_BANK_SEED
                or tuning_binding.get("opening_bank_minimum_physical_plies")
                != FINAL_OPENING_PLIES
                or tuning_binding.get("max_turns") != FINAL_MAX_TURNS
            ):
                errors.append("tuning receipt development panel is not frozen")
            if tuning_binding.get("opening_bank_sha256") == opening_hash:
                errors.append("development and final opening banks must be disjoint")
            development_states = tuning_binding.get("opening_state_identities")
            if (
                not isinstance(development_states, list)
                or len(development_states) != DEVELOPMENT_PAIRS
                or len(set(development_states)) != DEVELOPMENT_PAIRS
                or not all(
                    isinstance(value, str) and value
                    for value in development_states
                )
                or set(development_states).intersection(final_state_identities or [])
            ):
                errors.append(
                    "development and final banks share or omit canonical states"
                )
            development_transcripts = tuning_receipt.get(
                "opening_transcript_sha256"
            )
            if (
                not isinstance(development_transcripts, list)
                or len(development_transcripts) != DEVELOPMENT_PAIRS
                or len(set(development_transcripts)) != DEVELOPMENT_PAIRS
                or not all(valid_sha256(value) for value in development_transcripts)
            ):
                errors.append("tuning receipt omits development opening identities")
            tuning_reports = tuning_receipt.get("reports")
            if (
                tuning_receipt.get("grid") != list(tuning_gate.GRID)
                or not isinstance(tuning_reports, dict)
                or set(tuning_reports) != {str(value) for value in tuning_gate.GRID}
                or any(not isinstance(item, dict) for item in tuning_reports.values())
            ):
                errors.append("tuning receipt does not bind the frozen grid reports")
            else:
                try:
                    selected = max(
                        tuning_gate.GRID,
                        key=lambda value: (
                            int(tuning_reports[str(value)]["wins"]),
                            int(tuning_reports[str(value)]["minimum_color_wins"]),
                            -float(tuning_reports[str(value)]["p99_ms"]),
                            -value,
                        ),
                    )
                    if selected != tuning_receipt.get("chosen_exploration"):
                        errors.append(
                            "tuning receipt selection differs from its reports"
                        )
                    if any(
                        not valid_sha256(item.get("sha256"))
                        or not isinstance(item.get("path"), str)
                        or not item["path"]
                        for item in tuning_reports.values()
                    ):
                        errors.append("tuning receipt report bindings are invalid")
                except (KeyError, TypeError, ValueError, OverflowError):
                    errors.append("tuning receipt report scores are invalid")
            for field in (
                "model_sha256",
                "rank4_control_sha256",
                "neural_puct_control_sha256",
                "rank4_engine_sha256",
                "neural_puct_engine_sha256",
                "rank4_adapter_sha256",
                "neural_puct_adapter_sha256",
                "baseline_receipt_sha256",
                "seed",
                "candidate_tree_nodes",
                "control_tree_nodes",
                "control_work",
                "max_actions",
                "max_partial_paths",
                "fpu",
                "single_thread",
                "opponent",
                "shared_core_sha256",
                "candidate_source_sha256",
                "comparison_source_sha256",
                "comparison_executable_path",
                "comparison_executable_sha256",
                "opening_plies",
                "max_turns",
            ):
                expected = (
                    model_hash if field == "model_sha256" else configuration.get(field)
                )
                if tuning_binding.get(field) != expected:
                    errors.append(f"tuning receipt {field} differs from final report")
    baseline_path = configuration.get("baseline_receipt_path")
    baseline_hash = configuration.get("baseline_receipt_sha256")
    if not isinstance(baseline_path, str) or not valid_sha256(baseline_hash):
        errors.append("final comparison does not bind a matched-baseline receipt")
    try:
        tuning_gate.validate_baseline_receipt(baseline_receipt, model_hash)
    except ValueError as error:
        errors.append(str(error))
    attempt_identity: dict | None = None
    if attempt_ledger is None:
        if verify_files:
            errors.append("promotion decision requires a consumed final-attempt ledger")
    else:
        attempt_binding_value = attempt_ledger.get("binding")
        registration = attempt_ledger.get("bank_registration")
        consumption = attempt_ledger.get("report_consumption")
        if (
            attempt_ledger.get("schema") != ATTEMPT_SCHEMA
            or attempt_ledger.get("state") != "consumed"
            or not isinstance(attempt_binding_value, dict)
            or not isinstance(registration, dict)
            or not isinstance(consumption, dict)
            or hashlib.sha256(
                canonical_json_bytes(attempt_binding_value)
            ).hexdigest()
            != attempt_ledger.get("attempt_id")
            or attempt_binding_value.get("model_sha256") != model_hash
            or attempt_binding_value.get("tuning_receipt_sha256") != tuning_hash
            or attempt_binding_value.get("baseline_receipt_sha256") != baseline_hash
            or attempt_binding_value.get("comparison_executable_sha256")
            != comparison_executable_hash
            or (
                verify_files
                and attempt_binding_value.get("promotion_gate_sha256")
                != file_sha256(pathlib.Path(__file__))
            )
            or attempt_binding_value.get("opening_bank_path") != str(
                pathlib.Path(str(opening_source)).resolve()
            )
            or attempt_binding_value.get("comparison_report_path")
            != report_identity["path"]
            or attempt_binding_value.get("chosen_exploration") != exploration
            or registration.get("opening_bank_sha256") != opening_hash
            or consumption.get("comparison_report_path") != report_identity["path"]
            or consumption.get("comparison_report_sha256")
            != report_identity["sha256"]
        ):
            errors.append("final-attempt ledger does not bind this exact gate")
        attempt_identity = {
            "attempt_id": attempt_ledger.get("attempt_id"),
            "path": (
                str(attempt_ledger_path.resolve())
                if attempt_ledger_path is not None
                else "<in-memory>"
            ),
            "sha256": hashlib.sha256(
                canonical_json_bytes(attempt_ledger, pretty=True)
            ).hexdigest(),
        }
        if verify_files:
            try:
                if attempt_ledger_path is None:
                    raise ValueError("final-attempt ledger path is missing")
                raw_ledger = attempt_ledger_path.resolve().read_bytes()
                if (
                    hashlib.sha256(raw_ledger).hexdigest()
                    != attempt_identity["sha256"]
                    or json.loads(raw_ledger) != attempt_ledger
                ):
                    raise ValueError("final-attempt ledger bytes are stale")
            except (OSError, json.JSONDecodeError, ValueError) as error:
                errors.append(str(error))
    if verify_files:
        try:
            opening_path = pathlib.Path(str(opening_source))
            if file_sha256(opening_path) != opening_hash:
                errors.append("opening bank bytes do not match the report hash")
            if bank_opening_ids(opening_path) != {
                game["opening"] for game in grouped["rank4"]
            }:
                errors.append("comparison opening IDs differ from the frozen bank")
            if bank_classification(opening_path) != "final":
                errors.append("opening bank bytes are not final-classified")
            bank_seed, minimum_plies = validate_bank_metadata(opening_path, "final")
            if bank_seed != FINAL_BANK_SEED or minimum_plies != FINAL_OPENING_PLIES:
                errors.append("final opening bank parameters are not frozen")
            if bank_state_identities(opening_path) != set(
                final_state_identities or []
            ):
                errors.append("final report state identities differ from its bank")
            if isinstance(tuning_receipt, dict):
                overlap = bank_transcript_hashes(opening_path).intersection(
                    tuning_receipt.get("opening_transcript_sha256", [])
                )
                if overlap:
                    errors.append(
                        "development and final banks share opening transcripts"
                    )
        except OSError:
            errors.append("opening bank is unavailable for hash verification")
        try:
            if file_sha256(pathlib.Path(str(model_path))) != model_hash:
                errors.append("model bytes do not match the report hash")
        except OSError:
            errors.append("model is unavailable for hash verification")
        try:
            receipt_path = pathlib.Path(str(tuning_path))
            if file_sha256(receipt_path) != tuning_hash:
                errors.append("tuning receipt bytes do not match the report hash")
            if json.loads(receipt_path.read_bytes()) != tuning_receipt:
                errors.append("loaded tuning receipt differs from supplied receipt")
            if isinstance(tuning_receipt, dict):
                reports = tuning_receipt.get("reports")
                if isinstance(reports, dict) and all(
                    isinstance(reports.get(str(value)), dict)
                    for value in tuning_gate.GRID
                ):
                    rebuilt = tuning_gate.select(
                        [
                            pathlib.Path(reports[str(value)]["path"])
                            for value in tuning_gate.GRID
                        ],
                        pathlib.Path(str(baseline_path)),
                    )
                    if rebuilt != tuning_receipt:
                        errors.append(
                            "tuning receipt differs from its verified reports"
                        )
        except (OSError, json.JSONDecodeError):
            errors.append("tuning receipt is unavailable for verification")
        except (KeyError, TypeError, ValueError):
            errors.append("tuning receipt reports are unavailable for verification")
        try:
            receipt_path = pathlib.Path(str(baseline_path))
            if file_sha256(receipt_path) != baseline_hash:
                errors.append("baseline receipt bytes do not match the report hash")
            if json.loads(receipt_path.read_bytes()) != baseline_receipt:
                errors.append("loaded baseline receipt differs from supplied receipt")
            tuning_gate.validate_baseline_receipt(
                baseline_receipt, model_hash, verify_files=True
            )
        except (OSError, json.JSONDecodeError, ValueError):
            errors.append("baseline receipt is unavailable for verification")
        maintained_controls = provenance.control_source_sha256()
        reported_controls = {
            "rank4_control_sha256": rank4_hash,
            "rank4_engine_sha256": rank4_engine_hash,
            "neural_puct_control_sha256": neural_hash,
            "neural_puct_engine_sha256": neural_engine_hash,
            "rank4_adapter_sha256": rank4_adapter_hash,
            "neural_puct_adapter_sha256": neural_adapter_hash,
        }
        for field, expected in maintained_controls.items():
            if reported_controls[field] != expected:
                errors.append(f"{field} differs from maintained source")
        if shared_core_hash != shared_core_sha256():
            errors.append("shared control core hash differs from maintained sources")
        if candidate_source_hash != provenance.candidate_source_sha256():
            errors.append("candidate source closure differs from maintained sources")
        if comparison_source_hash != provenance.comparison_source_sha256():
            errors.append("comparison source differs from maintained source")
        try:
            if file_sha256(pathlib.Path(str(comparison_executable_path))) != (
                comparison_executable_hash
            ):
                errors.append("comparison executable differs from bound binary")
        except OSError:
            errors.append("comparison executable is unavailable for verification")

    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("comparison summary is missing")
    if summary.get("games") != len(results):
        errors.append("summary game count differs from results")
    if summary.get("illegal") != 0 or summary.get("unfinished") != 0:
        errors.append("summary reports illegal or unfinished games")
    candidate_summary = summary.get("candidate")
    if not isinstance(candidate_summary, dict):
        raise ValueError("comparison summary omits candidate work")
    maximum_observed = candidate_summary.get("max_ms")
    if not timing_samples:
        errors.append("candidate decision timings are missing or invalid")
        recomputed_maximum = float("inf")
    else:
        recomputed_maximum = max(timing_samples)
    if (
        not isinstance(maximum_observed, (int, float))
        or not math.isfinite(float(maximum_observed))
        or not math.isclose(
            float(maximum_observed),
            recomputed_maximum,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        errors.append("candidate maximum latency is missing or non-finite")
    elif recomputed_maximum >= maximum_ms:
        errors.append(
            f"candidate maximum latency {recomputed_maximum} is not below {maximum_ms}"
        )

    decisions: dict[str, dict] = {}
    thresholds = {
        "neural-puct": {"lower": 0.50, "color": 0.52},
        "rank4": {"lower": 0.475, "color": 0.475},
    }
    rank4_openings = {game["opening"] for game in grouped["rank4"]}
    neural_openings = {game["opening"] for game in grouped["neural-puct"]}
    if rank4_openings != neural_openings:
        errors.append("Rank-4 and neural PUCT panels use different openings")
    for opponent, games in grouped.items():
        if len(games) != minimum_games:
            errors.append(
                f"{opponent}: {len(games)} games is not exactly {minimum_games}"
            )
        opening_colors: dict[str, set[int]] = {}
        for game in games:
            opening_colors.setdefault(game["opening"], set()).add(
                int(game["candidate_player"])
            )
        if len(opening_colors) != expected_pairs or any(
            colors != {0, 1} for colors in opening_colors.values()
        ):
            errors.append(
                f"{opponent}: openings are not {expected_pairs} exact color-swapped pairs"
            )
        wins = sum(
            game.get("winner") == game.get("candidate_player") for game in games
        )
        colors = []
        for color in (0, 1):
            color_games = [
                game for game in games if game.get("candidate_player") == color
            ]
            color_wins = sum(game.get("winner") == color for game in color_games)
            rate = color_wins / len(color_games) if color_games else 0.0
            colors.append(
                {"color": color, "games": len(color_games), "wins": color_wins, "rate": rate}
            )
            if not color_games or rate < thresholds[opponent]["color"]:
                errors.append(
                    f"{opponent}: color {color} rate {rate:.6f} is below "
                    f"{thresholds[opponent]['color']:.6f}"
                )
        lower = wilson_lower(wins, len(games))
        if lower <= thresholds[opponent]["lower"]:
            errors.append(
                f"{opponent}: Wilson lower {lower:.6f} does not exceed "
                f"{thresholds[opponent]['lower']:.6f}"
            )
        decisions[opponent] = {
            "games": len(games),
            "wins": wins,
            "rate": wins / len(games) if games else 0.0,
            "wilson_lower_one_sided_95": lower,
            "colors": colors,
            "thresholds": thresholds[opponent],
        }

    recomputed_wins = sum(
        game.get("winner") == game.get("candidate_player") for game in results
    )
    recomputed_unfinished = sum(game.get("winner") not in (0, 1) for game in results)
    recomputed_illegal = sum(game.get("illegal") is not False for game in results)
    summary_colors = summary.get("colors")
    if (
        summary.get("wins") != recomputed_wins
        or summary.get("losses")
        != len(results) - recomputed_wins - recomputed_unfinished
        or summary.get("unfinished") != recomputed_unfinished
        or summary.get("illegal") != recomputed_illegal
        or not isinstance(summary_colors, list)
        or len(summary_colors) != 2
        or any(not isinstance(item, dict) for item in summary_colors)
    ):
        errors.append("comparison summary differs from game results")
    else:
        for color in (0, 1):
            color_games = [
                game for game in results if game.get("candidate_player") == color
            ]
            color_wins = sum(game.get("winner") == color for game in color_games)
            if (
                summary_colors[color].get("games") != len(color_games)
                or summary_colors[color].get("wins") != color_wins
            ):
                errors.append("comparison color summary differs from game results")
                break

    return {
        "schema": DECISION_SCHEMA,
        "eligible": not errors,
        "comparison_report": report_identity,
        "final_attempt": attempt_identity,
        "model_sha256": model_hash,
        "opening_bank_sha256": opening_hash,
        "tuning_receipt_sha256": tuning_hash,
        "baseline_receipt_sha256": baseline_hash,
        "control_sha256": {
            "rank4": rank4_hash,
            "neural_puct": neural_hash,
            "rank4_engine": rank4_engine_hash,
            "neural_puct_engine": neural_engine_hash,
            "rank4_adapter": rank4_adapter_hash,
            "neural_puct_adapter": neural_adapter_hash,
            "shared_core": shared_core_hash,
            "candidate_source": candidate_source_hash,
            "comparison_source": comparison_source_hash,
            "comparison_executable": comparison_executable_hash,
        },
        "minimum_games_per_opponent": minimum_games,
        "maximum_decision_ms_lt": maximum_ms,
        "opponents": decisions,
        "errors": errors,
    }


def prepare_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="prepare one protected final attempt")
    parser.add_argument("--ledger", type=pathlib.Path, required=True)
    parser.add_argument("--tuning-receipt", type=pathlib.Path, required=True)
    parser.add_argument("--baseline-receipt", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--comparison-executable", type=pathlib.Path, required=True)
    parser.add_argument("--bank", type=pathlib.Path, required=True)
    parser.add_argument("--report", type=pathlib.Path, required=True)
    arguments = parser.parse_args(argv)
    ledger = prepare_final_attempt(
        arguments.ledger,
        tuning_path=arguments.tuning_receipt,
        baseline_path=arguments.baseline_receipt,
        model_path=arguments.model,
        executable_path=arguments.comparison_executable,
        bank_path=arguments.bank,
        report_path=arguments.report,
    )
    print(json.dumps({"ledger": str(arguments.ledger), "attempt_id": ledger["attempt_id"]}))
    return 0


def register_bank_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="register the one protected final bank")
    parser.add_argument("--ledger", type=pathlib.Path, required=True)
    arguments = parser.parse_args(argv)
    ledger = register_final_bank(arguments.ledger)
    print(json.dumps({"ledger": str(arguments.ledger), "state": ledger["state"]}))
    return 0


def run_attempt_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="run the one protected final comparison")
    parser.add_argument("--ledger", type=pathlib.Path, required=True)
    arguments = parser.parse_args(argv)
    ledger, _ = run_final_attempt(arguments.ledger)
    print(json.dumps({"ledger": str(arguments.ledger), "state": ledger["state"]}))
    return 0


def decision_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=pathlib.Path)
    parser.add_argument("--tuning-receipt", type=pathlib.Path, required=True)
    parser.add_argument("--baseline-receipt", type=pathlib.Path, required=True)
    parser.add_argument("--attempt-ledger", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    arguments = parser.parse_args(argv)
    try:
        attempt, payload, _ = consume_final_report(arguments.attempt_ledger)
        if pathlib.Path(attempt["binding"]["comparison_report_path"]) != (
            arguments.report.resolve()
        ):
            raise ValueError("decision report path differs from final-attempt ledger")
        tuning_receipt = json.loads(arguments.tuning_receipt.read_text())
        baseline_receipt = json.loads(arguments.baseline_receipt.read_text())
        decision = evaluate(
            payload,
            FINAL_GAMES_PER_OPPONENT,
            FINAL_MAXIMUM_MS,
            tuning_receipt=tuning_receipt,
            baseline_receipt=baseline_receipt,
            report_path=arguments.report,
            attempt_ledger=attempt,
            attempt_ledger_path=arguments.attempt_ledger,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"promotion gate: {error}", file=sys.stderr)
        return 2
    rendered = json.dumps(decision, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        atomic_json(arguments.output.resolve(), decision)
    else:
        sys.stdout.write(rendered)
    return 0 if decision["eligible"] else 1


def publish_promoted(
    *,
    decision_path: pathlib.Path,
    candidate_manifest_path: pathlib.Path,
    baseline_receipt_path: pathlib.Path,
    tuning_receipt_path: pathlib.Path,
    attempt_ledger_path: pathlib.Path,
    report_path: pathlib.Path,
    output_directory: pathlib.Path,
) -> dict:
    """Atomically publish one fully revalidated, eligible Round-2 model."""

    paths = {
        "decision": decision_path.resolve(),
        "candidate_manifest": candidate_manifest_path.resolve(),
        "baseline_receipt": baseline_receipt_path.resolve(),
        "tuning_receipt": tuning_receipt_path.resolve(),
        "attempt_ledger": attempt_ledger_path.resolve(),
        "comparison_report": report_path.resolve(),
    }
    if output_directory.resolve().exists():
        raise ValueError("promoted model output directory already exists")
    loaded = {
        name: load_canonical_json(paths[name], name.replace("_", " "))
        for name in (
            "decision",
            "baseline_receipt",
            "tuning_receipt",
            "attempt_ledger",
        )
    }
    for name in ("candidate_manifest", "comparison_report"):
        value = json.loads(paths[name].read_bytes())
        if not isinstance(value, dict):
            raise ValueError(f"{name.replace('_', ' ')} must be a JSON object")
        loaded[name] = value
    decision = loaded["decision"]
    if (
        decision.get("schema") != DECISION_SCHEMA
        or decision.get("eligible") is not True
        or decision.get("errors") != []
    ):
        raise ValueError("only an eligible error-free decision can be published")
    report = loaded["comparison_report"]
    tuning = loaded["tuning_receipt"]
    baseline = loaded["baseline_receipt"]
    attempt = loaded["attempt_ledger"]
    rebuilt = evaluate(
        report,
        FINAL_GAMES_PER_OPPONENT,
        FINAL_MAXIMUM_MS,
        tuning_receipt=tuning,
        baseline_receipt=baseline,
        report_path=paths["comparison_report"],
        attempt_ledger=attempt,
        attempt_ledger_path=paths["attempt_ledger"],
    )
    if rebuilt != decision:
        raise ValueError("promotion decision does not recompute from bound evidence")
    manifest = loaded["candidate_manifest"]
    runtime = manifest.get("runtime")
    contract = manifest.get("campaign_contract")
    baseline_bindings = baseline.get("bindings")
    canonical_entries = (
        baseline_bindings.get("canonical_workflow_entries")
        if isinstance(baseline_bindings, dict)
        else None
    )
    if (
        manifest.get("schema") != "papersoccer.jacek-replay-bfm-model.v1"
        or manifest.get("status")
        != "canonical-campaign-candidate-not-game-gated"
        or not isinstance(runtime, dict)
        or not isinstance(runtime.get("path"), str)
        or not isinstance(contract, dict)
        or contract.get("eligible") is not True
        or contract.get("round") != 2
    ):
        raise ValueError("candidate manifest is not an eligible canonical Round-2 model")
    if (
        not isinstance(baseline_bindings, dict)
        or baseline_bindings.get("candidate_manifest_sha256")
        != file_sha256(paths["candidate_manifest"])
        or not isinstance(canonical_entries, list)
        or len(canonical_entries) != 3
        or pathlib.Path(
            str(canonical_entries[-1].get("model_manifest_path", ""))
        ).resolve()
        != paths["candidate_manifest"]
        or canonical_entries[-1].get("model_manifest_sha256")
        != file_sha256(paths["candidate_manifest"])
    ):
        raise ValueError(
            "candidate manifest is not the exact baseline-bound Round-2 manifest"
        )
    runtime_path = (paths["candidate_manifest"].parent / runtime["path"]).resolve()
    if (
        not runtime_path.is_file()
        or file_sha256(runtime_path) != runtime.get("artifact_sha256")
        or runtime.get("artifact_sha256") != decision.get("model_sha256")
        or pathlib.Path(str(report.get("model"))).resolve() != runtime_path
    ):
        raise ValueError("candidate runtime differs from the eligible decision")
    baseline_artifact = pathlib.Path(
        str(baseline.get("baseline_artifact", {}).get("path", ""))
    ).resolve()
    if not baseline_artifact.is_file():
        raise ValueError("matched-baseline diagnostic artifact is unavailable")

    output_directory = output_directory.resolve()
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    scratch = pathlib.Path(
        tempfile.mkdtemp(
            dir=output_directory.parent,
            prefix=f".{output_directory.name}.inprogress.",
        )
    )
    published = False
    try:
        copies = {
            "jacek_replay_bfm.runtime": runtime_path,
            "jacek_replay_bfm.runtime.json": paths["candidate_manifest"],
            "baseline-gate.json": paths["baseline_receipt"],
            "baseline-gate.baseline.runtime": baseline_artifact,
            "tuning-receipt.json": paths["tuning_receipt"],
            "final-attempt.json": paths["attempt_ledger"],
            "final-comparison.json": paths["comparison_report"],
            "final-decision.json": paths["decision"],
        }
        for name, source in copies.items():
            shutil.copy2(source, scratch / name)
        publication = {
            "schema": PUBLISHED_SCHEMA,
            "status": "promoted",
            "model_sha256": decision["model_sha256"],
            "source": {
                name: {"path": str(source), "sha256": file_sha256(source)}
                for name, source in sorted(paths.items())
            },
            "baseline_artifact_sha256": file_sha256(baseline_artifact),
            "published_files": {
                path.name: file_sha256(path)
                for path in sorted(scratch.iterdir())
                if path.is_file()
            },
        }
        atomic_json(scratch / "promotion-manifest.json", publication)
        os.replace(scratch, output_directory)
        published = True
        return publication
    finally:
        if not published:
            shutil.rmtree(scratch, ignore_errors=True)


def publish_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="publish one eligible promoted model")
    parser.add_argument("--decision", type=pathlib.Path, required=True)
    parser.add_argument("--candidate-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--baseline-receipt", type=pathlib.Path, required=True)
    parser.add_argument("--tuning-receipt", type=pathlib.Path, required=True)
    parser.add_argument("--attempt-ledger", type=pathlib.Path, required=True)
    parser.add_argument("--report", type=pathlib.Path, required=True)
    parser.add_argument("--output-directory", type=pathlib.Path, required=True)
    arguments = parser.parse_args(argv)
    publication = publish_promoted(
        decision_path=arguments.decision,
        candidate_manifest_path=arguments.candidate_manifest,
        baseline_receipt_path=arguments.baseline_receipt,
        tuning_receipt_path=arguments.tuning_receipt,
        attempt_ledger_path=arguments.attempt_ledger,
        report_path=arguments.report,
        output_directory=arguments.output_directory,
    )
    print(
        json.dumps(
            {
                "output": str(arguments.output_directory.resolve()),
                "model_sha256": publication["model_sha256"],
            }
        )
    )
    return 0


def main() -> int:
    commands = {
        "prepare-final-attempt": prepare_main,
        "register-final-bank": register_bank_main,
        "run-final-attempt": run_attempt_main,
        "decide": decision_main,
        "publish": publish_main,
    }
    if len(sys.argv) > 1 and sys.argv[1] in commands:
        try:
            return commands[sys.argv[1]](sys.argv[2:])
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            print(f"promotion gate: {error}", file=sys.stderr)
            return 2
    # Keep the old report-first shape parseable, but require the new ledger.
    return decision_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
