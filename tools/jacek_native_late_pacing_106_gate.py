#!/usr/bin/env python3
"""Audit the user-required 106-game prefix of a decisive model gate.

The canonical deployment selector deliberately remains stronger: it runs 106
pairs (212 games).  This tool validates that immutable selector report with the
selector itself, then scores the first 53 pairs as a separate, content-
addressed 106-game qualification.  It cannot create deployment evidence or
weaken the canonical selector thresholds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from collections.abc import Mapping, Sequence
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) in sys.path:
    sys.path.remove(str(TOOLS))
sys.path.insert(0, str(TOOLS))
import jacek_native_round2_selection as selector  # noqa: E402


SCHEMA = "papersoccer.jacek-native-late-pacing-106-gate/v1"
PAIRS = 53
GAMES = 106
MINIMUM_WINS = 58
MINIMUM_PER_COLOR = 25


class GateError(ValueError):
    """The decisive evidence cannot support the 106-game qualification."""


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value, allow_nan=False, ensure_ascii=False,
            separators=(",", ":"), sort_keys=True,
        )
        + "\n"
    ).encode()


def _content_addressed_report(path: pathlib.Path) -> tuple[dict[str, Any], str]:
    path = path.resolve()
    try:
        raw = path.read_bytes()
        parsed = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise GateError(f"cannot read decisive report: {path}") from error
    if canonical_json_bytes(parsed) != raw:
        raise GateError("decisive report is not canonical JSON")
    digest = sha256_bytes(raw)
    if path.name != f"{digest}.json":
        raise GateError("decisive report filename is not content-addressed")
    if not isinstance(parsed, dict):
        raise GateError("decisive report root is not an object")
    return parsed, digest


def validate_decisive_report(path: pathlib.Path) -> tuple[dict[str, Any], str, bytes]:
    report, report_sha = _content_addressed_report(path)
    if report.get("schema") != selector.REPORT_SCHEMA:
        raise GateError("source is not a canonical selector report")
    if report.get("profile") != selector.PROFILES["decisive"].payload():
        raise GateError("source report is not the exact decisive profile")
    candidate = report.get("candidate")
    baseline = report.get("baseline")
    if not isinstance(candidate, dict) or not isinstance(baseline, dict):
        raise GateError("source report model identities are missing")
    seed = candidate.get("seed")
    model_sha = candidate.get("model_sha256")
    if (
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        or not isinstance(model_sha, str)
    ):
        raise GateError("source report candidate identity is malformed")
    try:
        validated, validated_sha = selector._validate_report_file(
            path.resolve(), model_sha, {seed: candidate}, baseline
        )
    except (OSError, selector.SelectionError) as error:
        raise GateError("decisive report fails the canonical selector audit") from error
    if validated != report or validated_sha != report_sha:
        raise GateError("selector validation changed the source report")
    stdout = report["stdout"]
    stdout_path = path.resolve().parent / stdout["path"]
    stdout_raw = stdout_path.read_bytes()
    return report, report_sha, stdout_raw


def score_prefix(report: Mapping[str, Any], stdout_raw: bytes) -> dict[str, Any]:
    try:
        text = stdout_raw.decode("utf-8")
    except UnicodeError as error:
        raise GateError("decisive stdout is not UTF-8") from error
    lines = text.splitlines()
    decisive = selector.PROFILES["decisive"]
    if len(lines) != decisive.pairs + 3:
        raise GateError("decisive stdout does not contain all 106 pairs")

    candidate_wins = 0
    baseline_wins = 0
    unfinished = 0
    color_wins = [0, 0]
    pair_sha = hashlib.sha256()
    for expected_pair, line in enumerate(lines[2:2 + PAIRS]):
        match = selector.PAIR_PATTERN.fullmatch(line)
        if match is None:
            raise GateError(f"decisive pair {expected_pair} is malformed")
        pair, depth, _opening_seed, player_zero, player_one = map(
            int, match.groups()
        )
        if (
            pair != expected_pair
            or depth != selector.OPENING_TURNS[
                expected_pair % len(selector.OPENING_TURNS)
            ]
        ):
            raise GateError("decisive prefix pair schedule is stale")
        pair_sha.update(line.encode())
        pair_sha.update(b"\n")
        for color, winner in enumerate((player_zero, player_one)):
            if winner < 0:
                unfinished += 1
            elif winner == color:
                candidate_wins += 1
                color_wins[color] += 1
            else:
                baseline_wins += 1

    result = report.get("result")
    if not isinstance(result, Mapping):
        raise GateError("decisive report result is missing")
    operational_fields = (
        "candidate_headroom_failures", "candidate_operational_timeouts",
        "baseline_headroom_failures", "baseline_operational_timeouts",
    )
    operational_failures = sum(int(result.get(field, -1))
                               for field in operational_fields)
    passed = (
        unfinished == 0
        and operational_failures == 0
        and candidate_wins >= MINIMUM_WINS
        and color_wins[0] >= MINIMUM_PER_COLOR
        and color_wins[1] >= MINIMUM_PER_COLOR
    )
    return {
        "baseline_wins": baseline_wins,
        "candidate_player_one_wins": color_wins[0],
        "candidate_player_two_wins": color_wins[1],
        "candidate_wins": candidate_wins,
        "games": GAMES,
        "operational_failures_in_full_decisive_window": operational_failures,
        "pair_transcript_sha256": pair_sha.hexdigest(),
        "passed": passed,
        "required_per_color": MINIMUM_PER_COLOR,
        "required_total": MINIMUM_WINS,
        "unfinished": unfinished,
    }


def build_report(source_path: pathlib.Path) -> dict[str, Any]:
    report, report_sha, stdout_raw = validate_decisive_report(source_path)
    stdout = report["stdout"]
    return {
        "candidate": {
            key: report["candidate"][key]
            for key in (
                "seed", "checkpoint_sha256", "runtime_sha256",
                "model_sha256", "packed_sha256",
            )
        },
        "profile": {
            "derivation": "first-53-pairs-of-canonical-decisive/v1",
            "first_ms": 800,
            "games": GAMES,
            "later_ms": 155,
            "pairs": PAIRS,
        },
        "result": score_prefix(report, stdout_raw),
        "schema": SCHEMA,
        "source": {
            "canonical_decisive_games": selector.PROFILES["decisive"].pairs * 2,
            "report_sha256": report_sha,
            "selector_tool_sha256": sha256_bytes(
                pathlib.Path(selector.__file__).read_bytes()
            ),
            "stdout_bytes": len(stdout_raw),
            "stdout_sha256": stdout["sha256"],
        },
    }


def write_content_addressed(output_dir: pathlib.Path,
                            report: Mapping[str, Any]) -> pathlib.Path:
    raw = canonical_json_bytes(report)
    digest = sha256_bytes(raw)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{digest}.json"
    try:
        with output.open("xb") as stream:
            stream.write(raw)
    except FileExistsError as error:
        raise GateError(f"refusing to overwrite qualification: {output}") from error
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisive-report", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    arguments = parser.parse_args(argv)
    try:
        output = write_content_addressed(
            arguments.output_dir, build_report(arguments.decisive_report)
        )
    except (GateError, OSError, ValueError) as error:
        print(f"late-pacing 106-game gate failed: {error}", file=sys.stderr)
        return 1
    value = json.loads(output.read_bytes())
    print(json.dumps({
        "output": str(output),
        "passed": value["result"]["passed"],
        "sha256": output.stem,
    }, sort_keys=True))
    return 0 if value["result"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
