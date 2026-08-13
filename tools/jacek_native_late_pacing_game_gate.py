#!/usr/bin/env python3
"""Record the frozen 128-game root/source-game-held-out checkpoint gate.

The exact evaluation panel contributes 32 trap and 32 matched-control roots,
all from distinct source games and disjoint from every focused continuation
starting root and source game.  Each root is continued twice, swapping the
candidate and baseline checkpoints between board orientations.  The panel is
from the same frozen public arena source family, so this remains a diagnostic
gate rather than a claim of wholly independent evaluation data.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) in sys.path:
    sys.path.remove(str(TOOLS))
sys.path.insert(0, str(TOOLS))
import jacek_native_restart_corpus_round2 as restart_contract  # noqa: E402
import jacek_native_workflow_round2 as round2_workflow  # noqa: E402


PANEL_SCHEMA = "papersoccer.jacek-native-late-pacing-eval-panel/v2"
EXPECTED_PANEL_SHA256 = (
    "64283c5a4e7c5ac360969120a79e966a10cff9eb39d9cb0380cadccc14198246"
)
PLAN_SCHEMA = "papersoccer.jacek-native-late-pacing-game-gate-plan/v2"
REPORT_SCHEMA = "papersoccer.jacek-native-late-pacing-game-gate-report/v2"
SELECTION = "all-root/source-game-heldout-panel-roots/v1"
INDEPENDENCE = "canonical-root-and-source-game-disjoint-from-focused-starts/v1"
RUNS = 4
ROOTS = 64
GAMES = 128
REQUIRED_TOTAL = 70
REQUIRED_PER_ORIENTATION = 31
WORK = 4096
SEED = 2026090108
PLAN_HEADER = (
    "run_id\tpopulation\trole\tcollector_sha256\tarena_manifest_sha256\t"
    "game_id\tcandidate_player\tprefix_turn\tstate_id\tcanonical_key\t"
    "transcript\n"
)
SOURCE_PATHS = (
    "tools/jacek_native_late_pacing_eval_panel.py",
    "tools/jacek_native_late_pacing_game_gate.cpp",
    "tools/jacek_native_restart_round2.cpp",
    "tools/jacek_native_selfplay_round2.cpp",
    "tools/jacek_native_selfplay.cpp",
    "submissions/codingame/bots/jacek_native_bfm/bot.cpp",
    "submissions/codingame/bots/jacek_native_bfm/jacek_native_model.hpp",
    "src/core/rules.cpp",
    "src/core/geometry.cpp",
    "src/bots/mcts_internal.hpp",
    "include/papersoccer/types.hpp",
    "include/papersoccer/geometry.hpp",
    "include/papersoccer/rules.hpp",
)
GAME_PATTERN = re.compile(
    r"game=(0|[1-9][0-9]*) root=(0|[1-9][0-9]*) "
    r"run=([A-Za-z0-9_.:-]+) population=(trap|control) "
    r"source_color=(0|1) candidate_orientation=(0|1) winner=(-1|0|1)"
)


class GateError(ValueError):
    """The panel, continuation run, or immutable evidence is invalid."""


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256(path: pathlib.Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value, allow_nan=False, ensure_ascii=False,
            separators=(",", ":"), sort_keys=True,
        )
        + "\n"
    ).encode()


def safe_file(path: pathlib.Path, label: str) -> pathlib.Path:
    try:
        return restart_contract._safe_explicit_path(path, label)
    except ValueError as error:
        raise GateError(str(error)) from error


def load_panel(path: pathlib.Path) -> tuple[dict[str, Any], str]:
    path = safe_file(path, "late-pacing evaluation panel")
    raw = path.read_bytes()
    try:
        panel = json.loads(raw)
    except json.JSONDecodeError as error:
        raise GateError("late-pacing evaluation panel is not JSON") from error
    if canonical_json_bytes(panel) != raw:
        raise GateError("late-pacing evaluation panel is not canonical JSON")
    if (
        not isinstance(panel, dict)
        or panel.get("schema") != PANEL_SCHEMA
        or panel.get("purpose") != {
            "diagnostic_only": True,
            "observed_moves_usage": "state-construction-only",
            "training_eligible": False,
        }
        or not isinstance(panel.get("sources"), list)
        or not isinstance(panel.get("trap_states"), list)
        or not isinstance(panel.get("matched_winning_controls"), list)
    ):
        raise GateError("late-pacing evaluation panel contract is malformed")
    panel_sha = sha256_bytes(raw)
    counts = panel.get("counts")
    independence = panel.get("independence")
    traps = panel["trap_states"]
    controls = panel["matched_winning_controls"]
    entries = traps + controls
    trap_state_ids = {entry.get("state_id") for entry in traps}
    canonical_keys = [entry.get("canonical_key") for entry in entries]
    source_games = [(entry.get("run_id"), entry.get("game_id"))
                    for entry in entries]
    if (
        panel_sha != EXPECTED_PANEL_SHA256
        or len(panel["sources"]) != RUNS
        or not isinstance(counts, dict)
        or counts != {
            "clean_games": 312,
            "excluded_focused_source_games": 53,
            "excluded_focused_start_roots": 96,
            "matched_controls": 32,
            "trap_states": 32,
        }
        or len(traps) != 32 or len(controls) != 32
        or counts.get("clean_games") != sum(
            int(source.get("clean_games", -1)) for source in panel["sources"]
        )
        or independence != {
            "contract": INDEPENDENCE,
            "control_source_games_distinct": True,
            "focused_start_canonical_overlap": 0,
            "focused_start_source_game_overlap": 0,
            "one_selected_root_per_source_game": True,
            "scope": (
                "root/source-game held out from focused continuation starts "
                "only; same frozen public arena source family"
            ),
        }
        or len(set(canonical_keys)) != ROOTS
        or len(set(source_games)) != ROOTS
        or Counter(entry.get("candidate_player") for entry in traps)
        != {0: 16, 1: 16}
        or Counter(entry.get("candidate_player") for entry in controls)
        != {0: 16, 1: 16}
        or len(trap_state_ids) != 32
        or {entry.get("matched_trap_state_id") for entry in controls}
        != trap_state_ids
        or any(entry.get("match_exact") != {
            "color": True,
            "run_id": True,
            "turn_band": True,
            "used_edge_band": True,
            "zone": True,
        } for entry in controls)
        or restart_contract.LOWER_SHA.fullmatch(
            str(panel.get("source_sha256"))
        ) is None
    ):
        raise GateError(
            "late-pacing game gate requires the exact held-out 64-root panel"
        )
    traps_by_state = {entry["state_id"]: entry for entry in traps}
    for control in controls:
        trap = traps_by_state[control["matched_trap_state_id"]]
        if (
            control.get("run_id") != trap.get("run_id")
            or control.get("candidate_player") != trap.get("candidate_player")
            or control.get("turn_band") != trap.get("turn_band")
            or control.get("used_edge_band") != trap.get("used_edge_band")
            or control.get("zone") != trap.get("zone")
        ):
            raise GateError("matched control changes a required exact stratum")
    return panel, panel_sha


def load_collectors(
    panel: Mapping[str, Any], values: Sequence[Sequence[str]],
) -> tuple[dict[str, restart_contract.CollectorInput], list[dict[str, Any]]]:
    supplied: dict[str, pathlib.Path] = {}
    for pair in values:
        if len(pair) != 2 or pair[0] in supplied:
            raise GateError("collector run declarations are malformed or duplicated")
        supplied[pair[0]] = pathlib.Path(pair[1])
    sources = panel["sources"]
    source_by_run = {str(source.get("run_id")): source for source in sources}
    if len(source_by_run) != RUNS or set(supplied) != set(source_by_run):
        raise GateError("all and only panel source collectors must be supplied")
    result = {}
    reports = []
    for run_id in sorted(source_by_run):
        source = source_by_run[run_id]
        path = safe_file(supplied[run_id], f"collector for {run_id}")
        raw = path.read_bytes()
        try:
            collector = restart_contract.parse_collector_bytes(raw)
        except ValueError as error:
            raise GateError(f"collector {run_id} is invalid") from error
        expected = {
            "run_id": run_id,
            "arena_manifest_sha256": source.get("manifest_sha256"),
            "asserted_source_sha256": panel.get("source_sha256"),
        }
        if (
            sha256_bytes(raw) != source.get("collector_tsv_sha256")
            or any(collector.metadata.get(key) != value
                   for key, value in expected.items())
        ):
            raise GateError(f"collector {run_id} does not bind the panel")
        result[run_id] = collector
        reports.append({
            "arena_manifest_sha256": source["manifest_sha256"],
            "bytes": len(raw),
            "run_id": run_id,
            "sha256": collector.sha256,
        })
    return result, reports


def entry_state(entry: Mapping[str, Any], population: str,
                collector: restart_contract.CollectorInput) -> dict[str, Any]:
    game_id = str(entry.get("game_id"))
    matches = [game for game in collector.games if game.game_id == game_id]
    if len(matches) != 1:
        raise GateError("panel state has no unique collector game")
    game = matches[0]
    prefix_turn = entry.get("prefix_turn")
    candidate_player = entry.get("candidate_player")
    expected_result = "loss" if population == "trap" else "win"
    valid_roles = (
        {"last-enemy-shell", "one-own-before-mate", "two-own-before-mate"}
        if population == "trap" else {"matched-winning-control"}
    )
    if (
        isinstance(prefix_turn, bool) or not isinstance(prefix_turn, int)
        or not 0 < prefix_turn < len(game.actions)
        or candidate_player not in (0, 1)
        or entry.get("role") not in valid_roles
        or entry.get("observed_result") != expected_result
        or entry.get("observed_winner") != game.winner
        or ((game.winner == candidate_player) != (population == "control"))
        or candidate_player != game.candidate_player
        or entry.get("arena_manifest_sha256")
        != collector.metadata["arena_manifest_sha256"]
        or entry.get("training_eligible") is not False
        or entry.get("observed_moves_usage") != "state-construction-only"
        or entry.get("policy_target") is not None
        or entry.get("value_target") is not None
    ):
        raise GateError("panel state role/outcome contract is malformed")
    state = restart_contract.round1._initial_replay_state()
    own_decision = 0
    observed = None
    for turn, action in enumerate(game.actions):
        if turn == prefix_turn:
            active = restart_contract.round1._encode_replay_features(state)
            reflected = restart_contract.round1._encode_replay_features(
                state, reflected=True
            )
            observed = {
                "candidate_own_decision": own_decision,
                "canonical_key": restart_contract.round1.canonical_state_id(
                    min(active, reflected)
                ),
                "state_id": restart_contract.round1.canonical_state_id(active),
                "to_move": state.to_move,
                "transcript": "/".join(game.actions[:turn]),
            }
        if state.to_move == game.candidate_player:
            own_decision += 1
        restart_contract.round1._apply_complete_turn(
            state, action, turn, 1, opening=False
        )
    if state.winner != game.winner or observed is None:
        raise GateError("collector transcript is nonterminal or misses the prefix")
    if (
        observed["to_move"] != candidate_player
        or observed["candidate_own_decision"]
        != entry.get("candidate_own_decision")
        or observed["state_id"] != entry.get("state_id")
        or observed["canonical_key"] != entry.get("canonical_key")
        or observed["transcript"] != entry.get("transcript")
    ):
        raise GateError("panel state does not replay to its frozen identity")
    checked = dict(entry)
    checked["collector_tsv_sha256"] = collector.sha256
    return checked


def select_roots(panel: Mapping[str, Any], collectors: Mapping[
        str, restart_contract.CollectorInput]) -> tuple[list[dict[str, Any]], dict]:
    populations = {
        "trap": panel["trap_states"],
        "control": panel["matched_winning_controls"],
    }
    validated: dict[str, list[dict[str, Any]]] = {"trap": [], "control": []}
    seen_keys: set[str] = set()
    seen_games: set[tuple[str, str]] = set()
    for population, entries in populations.items():
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise GateError("panel state entry is not an object")
            run_id = entry.get("run_id")
            if run_id not in collectors:
                raise GateError("panel state names an unknown source run")
            checked = entry_state(entry, population, collectors[run_id])
            canonical = checked["canonical_key"]
            source_game = checked["run_id"], str(checked["game_id"])
            if canonical in seen_keys or source_game in seen_games:
                raise GateError("panel repeats a canonical root or source game")
            seen_keys.add(canonical)
            seen_games.add(source_game)
            validated[population].append(checked)

    runs = sorted(collectors)
    selected = validated["control"] + validated["trap"]
    if len(selected) != ROOTS or len({entry["canonical_key"]
                                      for entry in selected}) != ROOTS:
        raise GateError("held-out panel coverage is incomplete or duplicated")
    selected.sort(key=lambda entry: (
        entry["run_id"],
        "control" if entry["role"] == "matched-winning-control" else "trap",
        entry["candidate_player"], entry["canonical_key"],
        int(entry["game_id"]), entry["prefix_turn"],
    ))
    counts = Counter(
        (entry["run_id"],
         "control" if entry["role"] == "matched-winning-control" else "trap",
         entry["candidate_player"])
        for entry in selected
    )
    if (
        len(counts) != RUNS * 4
        or Counter(entry["candidate_player"] for entry in selected)
        != {0: 32, 1: 32}
        or Counter("control" if entry["role"] == "matched-winning-control"
                   else "trap" for entry in selected)
        != {"control": 32, "trap": 32}
    ):
        raise GateError("held-out panel is not balanced by population and color")
    group_report = [{
        "candidate_player": color,
        "population": population,
        "run_id": run_id,
        "selected": counts[(run_id, population, color)],
    } for run_id in runs for population in ("control", "trap") for color in (0, 1)]
    return selected, {
        "contract": SELECTION,
        "groups": group_report,
        "roots": len(selected),
        "source_games": len(seen_games),
    }


def render_plan(panel: Mapping[str, Any], panel_sha: str,
                roots: Sequence[Mapping[str, Any]]) -> bytes:
    metadata = {
        "panel_sha256": panel_sha,
        "schema": PLAN_SCHEMA,
        "selection": SELECTION,
        "source_sha256": panel["source_sha256"],
    }
    text = "".join(f"# {key}={value}\n" for key, value in sorted(metadata.items()))
    text += PLAN_HEADER
    for entry in roots:
        population = (
            "control" if entry["role"] == "matched-winning-control" else "trap"
        )
        fields = (
            entry["run_id"], population, entry["role"],
            entry["collector_tsv_sha256"], entry["arena_manifest_sha256"],
            str(entry["game_id"]), str(entry["candidate_player"]),
            str(entry["prefix_turn"]), entry["state_id"],
            entry["canonical_key"], entry["transcript"],
        )
        if any("\t" in value or "\n" in value or "\r" in value
               for value in fields):
            raise GateError("selected root cannot be represented canonically")
        text += "\t".join(fields) + "\n"
    return text.encode("ascii")


def source_identities() -> list[dict[str, str]]:
    result = []
    for relative in SOURCE_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise GateError(f"late-pacing gate source is missing: {relative}")
        result.append({"path": relative, "sha256": sha256(path)})
    return result


def runtime_identity(path: pathlib.Path, label: str) -> tuple[pathlib.Path, dict]:
    path = safe_file(path, label)
    try:
        identity = round2_workflow.runtime_identity(path)
    except ValueError as error:
        raise GateError(f"{label} runtime is invalid") from error
    return path, {**identity, "bytes": path.stat().st_size}


def key_values(line: str, prefix: str) -> dict[str, str]:
    if not line.startswith(prefix):
        raise GateError(f"gate stdout omits {prefix.strip()} line")
    result = {}
    for token in line[len(prefix):].split():
        if token.count("=") != 1:
            raise GateError("gate stdout field is malformed")
        key, value = token.split("=", 1)
        if not key or not value or key in result:
            raise GateError("gate stdout field is empty or duplicated")
        result[key] = value
    return result


def parse_stdout(raw: bytes, exit_code: int, plan_sha: str, panel_sha: str,
                 source_sha: str, candidate: Mapping[str, Any],
                 baseline: Mapping[str, Any], roots: Sequence[
                     Mapping[str, Any]]) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeError as error:
        raise GateError("gate stdout is not UTF-8") from error
    if not text.endswith("\n") or "\r" in text or "\x00" in text:
        raise GateError("gate stdout is not canonical line text")
    lines = text.splitlines()
    if len(lines) != GAMES + 4:
        raise GateError("gate stdout game coverage is incomplete")
    candidate_line = key_values(lines[0], "")
    baseline_line = key_values(lines[1], "")
    plan_line = key_values(lines[2], "")
    if candidate_line != {
        "candidate_runtime_sha256": candidate["artifact_sha256"],
        "candidate_model_sha256": candidate["model_sha256"],
        "candidate_packed_sha256": candidate["packed_sha256"],
    } or baseline_line != {
        "baseline_runtime_sha256": baseline["artifact_sha256"],
        "baseline_model_sha256": baseline["model_sha256"],
        "baseline_packed_sha256": baseline["packed_sha256"],
    } or plan_line != {
        "plan_sha256": plan_sha,
        "panel_sha256": panel_sha,
        "source_sha256": source_sha,
    }:
        raise GateError("gate stdout identity binding is stale")

    candidate_wins = 0
    baseline_wins = 0
    unfinished = 0
    orientation_wins = [0, 0]
    population_wins = {"trap": 0, "control": 0}
    for game, line in enumerate(lines[3:-1]):
        match = GAME_PATTERN.fullmatch(line)
        if match is None:
            raise GateError(f"gate game {game} row is malformed")
        observed_game, root_index, run_id, population, source_color, orientation, winner = (
            int(match.group(1)), int(match.group(2)), match.group(3),
            match.group(4), int(match.group(5)), int(match.group(6)),
            int(match.group(7)),
        )
        root = roots[game // 2]
        expected_population = (
            "control" if root["role"] == "matched-winning-control" else "trap"
        )
        if (
            observed_game != game or root_index != game // 2
            or orientation != game % 2 or run_id != root["run_id"]
            or population != expected_population
            or source_color != root["candidate_player"]
        ):
            raise GateError("gate game schedule is stale")
        if winner < 0:
            unfinished += 1
        elif winner == orientation:
            candidate_wins += 1
            orientation_wins[orientation] += 1
            population_wins[population] += 1
        else:
            baseline_wins += 1
    summary = key_values(lines[-1], "summary ")
    integer_fields = {
        "candidate", "baseline", "unfinished", "operational_failures",
        "candidate_player_one", "candidate_player_two", "trap_candidate",
        "control_candidate", "games", "searches", "expansions",
        "maximum_tree", "work", "seed", "temperature_turns",
        "maximum_generated_turns", "required_total",
        "required_per_orientation",
    }
    text_fields = {"temperature", "passed"}
    if set(summary) != integer_fields | text_fields:
        raise GateError("gate summary fields are not frozen")
    try:
        parsed = {field: int(summary[field]) for field in integer_fields}
    except ValueError as error:
        raise GateError("gate summary integer is malformed") from error
    if summary["passed"] not in {"true", "false"}:
        raise GateError("gate passed value is malformed")
    parsed.update({
        "passed": summary["passed"] == "true",
        "temperature": summary["temperature"],
    })
    recomputed = {
        "candidate": candidate_wins,
        "baseline": baseline_wins,
        "unfinished": unfinished,
        "candidate_player_one": orientation_wins[0],
        "candidate_player_two": orientation_wins[1],
        "trap_candidate": population_wins["trap"],
        "control_candidate": population_wins["control"],
        "games": GAMES,
    }
    if any(parsed[field] != value for field, value in recomputed.items()):
        raise GateError("gate summary disagrees with its full game transcript")
    passed = (
        candidate_wins >= REQUIRED_TOTAL
        and min(orientation_wins) >= REQUIRED_PER_ORIENTATION
        and unfinished == 0
        and parsed["operational_failures"] == 0
        and candidate_wins + baseline_wins == GAMES
    )
    if (
        parsed["work"] != WORK or parsed["seed"] != SEED
        or parsed["temperature"] != "3"
        or parsed["temperature_turns"] != 12
        or parsed["maximum_generated_turns"] != 384
        or parsed["required_total"] != REQUIRED_TOTAL
        or parsed["required_per_orientation"] != REQUIRED_PER_ORIENTATION
        or parsed["searches"] <= 0 or parsed["maximum_tree"] <= 0
        or parsed["passed"] != passed or exit_code not in (0, 1)
        or (exit_code == 0) != passed
    ):
        raise GateError("gate fixed profile or exit status is inconsistent")
    return parsed


@contextlib.contextmanager
def serial_lock():
    path = ROOT / "build" / ".jacek-native-late-pacing-game-gate.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise GateError("another late-pacing game gate is running") from error
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def write_exclusive(path: pathlib.Path, raw: bytes) -> None:
    try:
        with path.open("xb") as output:
            output.write(raw)
    except FileExistsError as error:
        raise GateError(f"refusing to overwrite immutable evidence: {path}") from error


def record(*, panel_path: pathlib.Path, collector_values: Sequence[Sequence[str]],
           candidate_path: pathlib.Path, baseline_path: pathlib.Path,
           gate_binary: pathlib.Path, output_dir: pathlib.Path) -> pathlib.Path:
    panel, panel_sha = load_panel(panel_path)
    collectors, collector_reports = load_collectors(panel, collector_values)
    roots, selection_report = select_roots(panel, collectors)
    plan_raw = render_plan(panel, panel_sha, roots)
    plan_sha = sha256_bytes(plan_raw)
    candidate_path, candidate = runtime_identity(candidate_path, "candidate")
    baseline_path, baseline = runtime_identity(baseline_path, "baseline")
    if candidate["packed_sha256"] == baseline["packed_sha256"]:
        raise GateError("candidate and baseline checkpoints are identical")
    gate_binary = safe_file(gate_binary, "late-pacing game gate binary")
    if not os.access(gate_binary, os.X_OK):
        raise GateError("late-pacing game gate binary is not executable")
    command = [
        str(gate_binary.resolve()),
        "--plan", "$PLAN",
        "--plan-sha256", plan_sha,
        "--candidate-checkpoint", str(candidate_path),
        "--candidate-artifact-sha256", candidate["artifact_sha256"],
        "--baseline-checkpoint", str(baseline_path),
        "--baseline-artifact-sha256", baseline["artifact_sha256"],
    ]
    with tempfile.TemporaryDirectory(prefix="jacek-late-gate-plan-") as temporary:
        temporary_plan = pathlib.Path(temporary) / "plan.tsv"
        temporary_plan.write_bytes(plan_raw)
        actual_command = list(command)
        actual_command[2] = str(temporary_plan)
        with serial_lock():
            completed = subprocess.run(
                actual_command, cwd=ROOT, capture_output=True, check=False
            )
    if completed.stderr:
        raise GateError("late-pacing game gate wrote stderr")
    result = parse_stdout(
        completed.stdout, completed.returncode, plan_sha, panel_sha,
        panel["source_sha256"], candidate, baseline, roots,
    )
    stdout_sha = sha256_bytes(completed.stdout)
    report = {
        "baseline": baseline,
        "candidate": candidate,
        "execution": {
            "command": [
                "$GATE_BINARY", "--plan", "$PLAN", "--plan-sha256", plan_sha,
                "--candidate-checkpoint", "$CANDIDATE_RUNTIME",
                "--candidate-artifact-sha256", candidate["artifact_sha256"],
                "--baseline-checkpoint", "$BASELINE_RUNTIME",
                "--baseline-artifact-sha256", baseline["artifact_sha256"],
            ],
            "exit_code": completed.returncode,
            "gate_binary_bytes": gate_binary.stat().st_size,
            "gate_binary_sha256": sha256(gate_binary),
            "gate_sources": source_identities(),
            "recorder_sha256": sha256(pathlib.Path(__file__)),
            "serial_fixed_work_lock": True,
        },
        "panel": {
            "bytes": safe_file(
                panel_path, "late-pacing evaluation panel"
            ).stat().st_size,
            "collectors": collector_reports,
            "sha256": panel_sha,
            "source_sha256": panel["source_sha256"],
        },
        "plan": {
            "bytes": len(plan_raw),
            "path": f"{plan_sha}.plan.tsv",
            "selection": selection_report,
            "sha256": plan_sha,
        },
        "profile": {
            "games": GAMES,
            "required_per_orientation": REQUIRED_PER_ORIENTATION,
            "required_total": REQUIRED_TOTAL,
            "roots": ROOTS,
            "seed": str(SEED),
            "work": WORK,
        },
        "result": result,
        "schema": REPORT_SCHEMA,
        "stdout": {
            "bytes": len(completed.stdout),
            "path": f"{stdout_sha}.stdout.txt",
            "sha256": stdout_sha,
        },
    }
    report_raw = canonical_json_bytes(report)
    report_sha = sha256_bytes(report_raw)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_output = output_dir / report["plan"]["path"]
    if plan_output.exists():
        if plan_output.read_bytes() != plan_raw:
            raise GateError("content-addressed plan path contains different bytes")
    else:
        write_exclusive(plan_output, plan_raw)
    stdout_output = output_dir / report["stdout"]["path"]
    report_output = output_dir / f"{report_sha}.json"
    write_exclusive(stdout_output, completed.stdout)
    try:
        write_exclusive(report_output, report_raw)
    except Exception:
        stdout_output.unlink(missing_ok=True)
        raise
    return report_output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True, type=pathlib.Path)
    parser.add_argument(
        "--collector", required=True, action="append", nargs=2,
        metavar=("RUN_ID", "PATH"),
    )
    parser.add_argument("--candidate-runtime", required=True, type=pathlib.Path)
    parser.add_argument("--baseline-runtime", required=True, type=pathlib.Path)
    parser.add_argument("--gate-binary", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    arguments = parser.parse_args(argv)
    try:
        output = record(
            panel_path=arguments.panel,
            collector_values=arguments.collector,
            candidate_path=arguments.candidate_runtime,
            baseline_path=arguments.baseline_runtime,
            gate_binary=arguments.gate_binary,
            output_dir=arguments.output_dir,
        )
    except (GateError, OSError, subprocess.SubprocessError, ValueError) as error:
        print(f"late-pacing 128-game gate failed: {error}", file=sys.stderr)
        return 1
    report = json.loads(output.read_bytes())
    print(json.dumps({
        "output": str(output),
        "passed": report["result"]["passed"],
        "sha256": output.stem,
    }, sort_keys=True))
    return 0 if report["result"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
