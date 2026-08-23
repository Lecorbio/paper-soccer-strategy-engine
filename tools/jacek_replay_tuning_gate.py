#!/usr/bin/env python3
"""Freeze the development-only exploration sweep before final evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys

import jacek_replay_provenance as provenance

try:
    import jacek_replay_baseline as baseline_gate
except ModuleNotFoundError as error:
    if error.name != "numpy":
        raise
    baseline_gate = None


REPORT_SCHEMA = "papersoccer.jacek-replay-bfm-comparison.v1"
RECEIPT_SCHEMA = "papersoccer.jacek-replay-bfm-tuning-receipt.v1"
BASELINE_SCHEMA = "papersoccer.jacek-replay-bfm-baseline-gate.v1"
GRID = (0.25, 0.5, 0.95)
DEVELOPMENT_PAIRS = 200
DEVELOPMENT_TIME_MS = 20
DEVELOPMENT_OPENING_PLIES = 12
DEVELOPMENT_BANK_SEED = 123_456_789
DEVELOPMENT_MAX_TURNS = 320


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def identity_sha256(identities: set[tuple[str, str, int]]) -> str:
    rendered = "".join(
        f"{opening}\t{opponent}\t{color}\n"
        for opening, opponent, color in sorted(identities)
    )
    return hashlib.sha256(rendered.encode()).hexdigest()


def game_timing_samples(game: dict, index: int) -> list[float]:
    raw = game.get("candidate_ms")
    if not isinstance(raw, list):
        raise ValueError(f"development game {index} omits candidate timings")
    samples: list[float] = []
    for value in raw:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(
                f"development game {index} has an invalid candidate timing"
            )
        samples.append(float(value))
    return samples


def opening_transcript_hashes(path: pathlib.Path) -> list[str]:
    hashes = []
    for line in path.read_text().splitlines():
        if (
            not line
            or line.startswith("#")
            or line == "opening_id\ttranscript\tstate_identity"
        ):
            continue
        fields = line.split("\t")
        if len(fields) != 3:
            raise ValueError("development opening bank row is malformed")
        transcript = fields[1]
        hashes.append(hashlib.sha256(transcript.encode()).hexdigest())
    if not hashes or len(hashes) != len(set(hashes)):
        raise ValueError("development opening bank repeats a transcript")
    return sorted(hashes)


def opening_ids(path: pathlib.Path) -> set[str]:
    result = set()
    for line in path.read_text().splitlines():
        if (
            not line
            or line.startswith("#")
            or line == "opening_id\ttranscript\tstate_identity"
        ):
            continue
        fields = line.split("\t")
        if len(fields) != 3:
            raise ValueError("development opening bank row is malformed")
        opening = fields[0]
        if not opening or opening in result:
            raise ValueError("development opening bank repeats an id")
        result.add(opening)
    return result


def opening_state_identities(path: pathlib.Path) -> set[str]:
    result = set()
    for line in path.read_text().splitlines():
        if (
            not line
            or line.startswith("#")
            or line == "opening_id\ttranscript\tstate_identity"
        ):
            continue
        fields = line.split("\t")
        if len(fields) != 3 or not fields[2] or fields[2] in result:
            raise ValueError("development opening bank repeats a state")
        result.add(fields[2])
    return result


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
        raise ValueError("development opening bank metadata is invalid")
    return int(lines[3][7:]), int(lines[4][25:])


def load_report(path: pathlib.Path) -> dict:
    payload = json.loads(path.read_bytes())
    if payload.get("schema") != REPORT_SCHEMA:
        raise ValueError(f"unsupported comparison report: {path}")
    return payload


def validate_baseline_receipt(
    receipt: object,
    model_sha256: object,
    *,
    verify_files: bool = False,
) -> None:
    if baseline_gate is None:
        raise ValueError(
            "matched-baseline verification requires requirements-research.txt"
        )
    baseline_gate.validate_receipt(
        receipt, model_sha256, verify_files=verify_files
    )


def select(
    paths: list[pathlib.Path],
    baseline_receipt_path: pathlib.Path,
    *,
    verify_files: bool = True,
) -> dict:
    if len(paths) != len(GRID):
        raise ValueError("tuning requires exactly three development reports")
    reports = [(path, load_report(path)) for path in paths]
    by_exploration = {}
    binding = None
    for path, report in reports:
        config = report.get("configuration")
        summary = report.get("summary")
        if not isinstance(config, dict) or not isinstance(summary, dict):
            raise ValueError("development report is incomplete")
        exploration = config.get("exploration")
        if exploration not in GRID or exploration in by_exploration:
            raise ValueError("development reports do not cover the frozen grid")
        if summary.get("illegal") != 0 or summary.get("unfinished") != 0:
            raise ValueError("development report has an operational failure")
        results = report.get("results")
        if not isinstance(results, list) or not results:
            raise ValueError("development report has no games")
        identities: set[tuple[str, str, int]] = set()
        samples: list[float] = []
        for index, game in enumerate(results):
            if not isinstance(game, dict):
                raise ValueError(f"development game {index} is not an object")
            opening = game.get("opening")
            opponent = game.get("opponent")
            color = game.get("candidate_player")
            winner = game.get("winner")
            if not isinstance(opening, str) or not opening:
                raise ValueError(f"development game {index} has no opening id")
            if opponent not in ("rank4", "neural-puct"):
                raise ValueError(f"development game {index} has an unknown opponent")
            if isinstance(color, bool) or type(color) is not int or color not in (0, 1):
                raise ValueError(f"development game {index} has an invalid color")
            if (
                isinstance(winner, bool)
                or type(winner) is not int
                or winner not in (0, 1)
                or game.get("illegal") is not False
            ):
                raise ValueError(
                    f"development game {index} is illegal or unfinished"
                )
            identity = (opening, opponent, color)
            if identity in identities:
                raise ValueError("development report repeats a game identity")
            identities.add(identity)
            samples.extend(game_timing_samples(game, index))
        if len(identities) != len(results) or {
            identity[1] for identity in identities
        } != {"rank4", "neural-puct"}:
            raise ValueError("development report game identities are incomplete")
        pairs = config.get("pairs")
        openings = {identity[0] for identity in identities}
        state_identities = config.get("opening_state_identities")
        if (
            not isinstance(pairs, int)
            or pairs <= 0
            or config.get("opponent") != "both"
            or len(openings) != pairs
            or len(identities) != pairs * 4
            or summary.get("games") != len(results)
            or not isinstance(state_identities, list)
            or len(state_identities) != pairs
            or len(set(state_identities)) != pairs
            or not all(
                isinstance(value, str) and value for value in state_identities
            )
            or not isinstance(summary.get("wins"), int)
            or not 0 <= summary["wins"] <= len(results)
        ):
            raise ValueError("development report is not a complete paired panel")
        colors = summary.get("colors")
        if (
            not isinstance(colors, list)
            or len(colors) != 2
            or any(not isinstance(item, dict) for item in colors)
        ):
            raise ValueError("development report omits color results")
        recomputed_wins = sum(
            game.get("winner") == game.get("candidate_player")
            for game in results
        )
        recomputed_colors = []
        for color in (0, 1):
            color_games = [
                game for game in results if game.get("candidate_player") == color
            ]
            recomputed_colors.append(
                {
                    "games": len(color_games),
                    "wins": sum(game.get("winner") == color for game in color_games),
                }
            )
        if summary.get("wins") != recomputed_wins or any(
            colors[index].get("games") != recomputed_colors[index]["games"]
            or colors[index].get("wins") != recomputed_colors[index]["wins"]
            for index in (0, 1)
        ):
            raise ValueError("development summary differs from game results")
        if summary.get("losses") != len(results) - recomputed_wins:
            raise ValueError("development summary differs from game results")
        if not samples:
            raise ValueError("development decision timings are missing or invalid")
        samples.sort()
        p99_index = min(
            len(samples) - 1, math.ceil(0.99 * len(samples)) - 1
        )
        recomputed_p99 = samples[p99_index]
        current_binding = {
            "model_sha256": report.get("model_sha256"),
            "model_path": report.get("model"),
            "opening_bank_sha256": config.get("opening_bank_sha256"),
            "opening_source": config.get("opening_source"),
            "opening_state_identities": config.get(
                "opening_state_identities"
            ),
            "baseline_receipt_sha256": config.get("baseline_receipt_sha256"),
            "rank4_control_sha256": config.get("rank4_control_sha256"),
            "rank4_engine_sha256": config.get("rank4_engine_sha256"),
            "neural_puct_control_sha256": config.get(
                "neural_puct_control_sha256"
            ),
            "neural_puct_engine_sha256": config.get(
                "neural_puct_engine_sha256"
            ),
            "rank4_adapter_sha256": config.get("rank4_adapter_sha256"),
            "neural_puct_adapter_sha256": config.get(
                "neural_puct_adapter_sha256"
            ),
            "shared_core_sha256": config.get("shared_core_sha256"),
            "candidate_source_sha256": config.get("candidate_source_sha256"),
            "comparison_source_sha256": config.get("comparison_source_sha256"),
            "comparison_executable_path": config.get(
                "comparison_executable_path"
            ),
            "comparison_executable_sha256": config.get(
                "comparison_executable_sha256"
            ),
            "pairs": config.get("pairs"),
            "time_ms": config.get("time_ms"),
            "max_turns": config.get("max_turns"),
            "opening_plies": config.get("opening_plies"),
            "opening_bank_seed": config.get("opening_bank_seed"),
            "opening_bank_minimum_physical_plies": config.get(
                "opening_bank_minimum_physical_plies"
            ),
            "opponent": config.get("opponent"),
            "seed": config.get("seed"),
            "candidate_tree_nodes": config.get("candidate_tree_nodes"),
            "control_tree_nodes": config.get("control_tree_nodes"),
            "control_work": config.get("control_work"),
            "max_actions": config.get("max_actions"),
            "max_partial_paths": config.get("max_partial_paths"),
            "fpu": config.get("fpu"),
            "single_thread": config.get("single_thread"),
            "opening_bank_classification": config.get(
                "opening_bank_classification"
            ),
            "game_identity_sha256": identity_sha256(identities),
        }
        if binding is None:
            binding = current_binding
        elif current_binding != binding:
            raise ValueError("development reports do not share exact inputs")
        if (
            config.get("opponent") != "both"
            or config.get("pairs") != DEVELOPMENT_PAIRS
            or config.get("time_ms") != DEVELOPMENT_TIME_MS
            or config.get("max_turns") != DEVELOPMENT_MAX_TURNS
            or config.get("opening_plies") != DEVELOPMENT_OPENING_PLIES
            or config.get("opening_bank_seed") != DEVELOPMENT_BANK_SEED
            or config.get("opening_bank_minimum_physical_plies")
            != DEVELOPMENT_OPENING_PLIES
            or config.get("candidate_tree_nodes") != 1_000_000
            or config.get("control_tree_nodes") != 100_000
            or config.get("control_work") != 3_000_000
            or config.get("max_actions") != 250
            or config.get("max_partial_paths") != 50_000
            or config.get("fpu") != 0.5
            or config.get("single_thread") is not True
            or config.get("opening_bank_classification") != "development"
            or not isinstance(report.get("model"), str)
            or not report["model"]
            or not valid_sha256(report.get("model_sha256"))
            or not isinstance(config.get("opening_source"), str)
            or not config["opening_source"]
            or config["opening_source"] == "generated"
            or not all(
                valid_sha256(config.get(field))
                for field in (
                    "rank4_control_sha256",
                    "rank4_engine_sha256",
                    "neural_puct_control_sha256",
                    "neural_puct_engine_sha256",
                    "rank4_adapter_sha256",
                    "neural_puct_adapter_sha256",
                    "shared_core_sha256",
                    "candidate_source_sha256",
                    "comparison_source_sha256",
                )
            )
            or not isinstance(config.get("comparison_executable_path"), str)
            or not config["comparison_executable_path"]
            or not valid_sha256(config.get("comparison_executable_sha256"))
            or not valid_sha256(config.get("baseline_receipt_sha256"))
        ):
            raise ValueError("development report changed a frozen search input")
        candidate = summary.get("candidate")
        if not isinstance(candidate, dict):
            raise ValueError("development report omits candidate work")
        p99 = candidate.get("p99_ms")
        if (
            not isinstance(p99, (int, float))
            or not math.isfinite(float(p99))
            or float(p99) < 0.0
            or not math.isclose(
                float(p99), recomputed_p99, rel_tol=0.0, abs_tol=1e-9
            )
            or any(
                not isinstance(item.get("games"), int)
                or not isinstance(item.get("wins"), int)
                or item["games"] <= 0
                or not 0 <= item["wins"] <= item["games"]
                for item in colors
            )
        ):
            raise ValueError("development timing or color counts are invalid")
        by_exploration[exploration] = {
            "path": str(path),
            "sha256": sha256(path),
            "wins": int(summary.get("wins", -1)),
            "minimum_color_wins": min(int(item.get("wins", -1)) for item in colors),
            "p99_ms": float(p99),
        }
    if set(by_exploration) != set(GRID):
        raise ValueError("development reports do not cover the frozen grid")
    development_transcripts: list[str]
    if verify_files:
        opening_path = pathlib.Path(str(binding["opening_source"]))
        if sha256(opening_path) != binding["opening_bank_sha256"]:
            raise ValueError("development opening bank SHA-256 mismatch")
        bank_seed, minimum_plies = validate_bank_metadata(
            opening_path, "development"
        )
        if (
            bank_seed != DEVELOPMENT_BANK_SEED
            or minimum_plies != DEVELOPMENT_OPENING_PLIES
        ):
            raise ValueError("development opening bank parameters are not frozen")
        if opening_ids(opening_path) != {identity[0] for identity in identities}:
            raise ValueError("development report openings differ from its bank")
        if opening_state_identities(opening_path) != set(
            binding["opening_state_identities"]
        ):
            raise ValueError("development report states differ from its bank")
        development_transcripts = opening_transcript_hashes(opening_path)
        model_path = pathlib.Path(str(binding["model_path"]))
        if sha256(model_path) != binding["model_sha256"]:
            raise ValueError("candidate model changed after comparison")
        for field, expected in provenance.control_source_sha256().items():
            if binding[field] != expected:
                raise ValueError(f"{field} changed after comparison")
        if binding["shared_core_sha256"] != provenance.shared_core_sha256():
            raise ValueError("shared control core changed after comparison")
        if binding["candidate_source_sha256"] != provenance.candidate_source_sha256():
            raise ValueError("candidate source closure changed after comparison")
        if binding["comparison_source_sha256"] != provenance.comparison_source_sha256():
            raise ValueError("comparison source changed after comparison")
        executable_path = pathlib.Path(str(binding["comparison_executable_path"]))
        if sha256(executable_path) != binding["comparison_executable_sha256"]:
            raise ValueError("comparison executable changed after comparison")
    else:
        development_transcripts = [binding["game_identity_sha256"]]
    baseline = json.loads(baseline_receipt_path.read_bytes())
    validate_baseline_receipt(
        baseline, binding["model_sha256"], verify_files=verify_files
    )
    if sha256(baseline_receipt_path) != binding["baseline_receipt_sha256"]:
        raise ValueError("development reports do not bind a passing baseline gate")
    chosen = max(
        GRID,
        key=lambda value: (
            by_exploration[value]["wins"],
            by_exploration[value]["minimum_color_wins"],
            -by_exploration[value]["p99_ms"],
            -value,
        ),
    )
    return {
        "schema": RECEIPT_SCHEMA,
        "classification": "development-only-exploration-selection",
        "binding": binding,
        "grid": list(GRID),
        "reports": {str(key): value for key, value in sorted(by_exploration.items())},
        "chosen_exploration": chosen,
        "opening_transcript_sha256": development_transcripts,
        "selection": (
            "maximum wins; then maximum minimum-color wins; "
            "then minimum p99 latency; then lower exploration"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs=3, type=pathlib.Path)
    parser.add_argument("--baseline-receipt", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    arguments = parser.parse_args()
    try:
        receipt = select(arguments.reports, arguments.baseline_receipt)
        if baseline_gate is None:  # pragma: no cover - select already rejects it
            raise ValueError("matched-baseline verification dependencies are missing")
        baseline_gate.atomic_json(arguments.output.resolve(), receipt)
        print(
            json.dumps(
                {
                    "output": str(arguments.output),
                    "chosen_exploration": receipt["chosen_exploration"],
                }
            )
        )
        return 0
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        print(f"jacek replay tuning gate: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
