#!/usr/bin/env python3
"""Build the root/source-game-held-out late-pacing evaluation panel.

The focused late-pacing corpus started from 96 replay-located trap roots.  This
additive diagnostic derives new roots from the same frozen 312 clean games but
excludes every focused-start canonical key and every loss game that supplied a
focused start.  It selects one state from each of 32 untouched loss games and
matches 32 controls from distinct winning games.  Replay outcomes and actions
locate states only; generated continuations provide the evaluation outcomes.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
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
import jacek_native_late_trap_panel as trap_builder  # noqa: E402
import jacek_native_restart_corpus_round2 as restart  # noqa: E402


SCHEMA = "papersoccer.jacek-native-late-pacing-eval-panel/v2"
INDEPENDENCE = (
    "canonical-root-and-source-game-disjoint-from-focused-starts/v1"
)
SOURCE_SHA256 = trap_builder.SOURCE_SHA256
TARGET_TRAPS_PER_COLOR = 16
TARGET_TRAPS = 32
TARGET_CONTROLS = 32
EXPECTED_EXCLUSION_PANELS = {
    "43a64b3d3cb5363ec69488d5fa5654686c9f0abdd895334acd14b562d3f280e8":
        "papersoccer.jacek-native-late-trap-panel/v1",
    "e21285b2582c162bf784dfd90c4e6ba33f15d6728f71d6c430b6ce6f430cee4a":
        "papersoccer.jacek-native-late-trap-panel/v2",
}
EXPECTED_FOCUSED_MANIFESTS = frozenset({
    "a0d963b5abb350af68513e79224fa36be5c0ea92f7d1a9796e90c9af323159f5",
    "cc6760aa517a84dd67fc3d46cc6025985f7a505761dcddf134ed2622d1248d67",
    "b558d65ea5fa4c2555c437a0eefbd3bc143c2d55dcaa0aa7dd7d8679b22c2de5",
    "b2571b5cd14d005900354f5468d082a5a67138df23f564b012c73fb78c8d3676",
})


class PanelError(ValueError):
    """The frozen sources cannot support an independent evaluation panel."""


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


def display_path(path: pathlib.Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def safe_file(path: pathlib.Path, label: str) -> pathlib.Path:
    try:
        return restart._safe_explicit_path(path, label)
    except ValueError as error:
        raise PanelError(str(error)) from error


def load_canonical(path: pathlib.Path, label: str) -> tuple[
        pathlib.Path, bytes, dict[str, Any], str]:
    path = safe_file(path, label)
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise PanelError(f"{label} is not JSON") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
        raise PanelError(f"{label} is not canonical JSON")
    return path, raw, value, sha256_bytes(raw)


def load_exclusion_panels(paths: Sequence[pathlib.Path]) -> tuple[
        set[str], set[tuple[str, str]], list[dict[str, Any]]]:
    observed = {}
    keys: set[str] = set()
    games: set[tuple[str, str]] = set()
    reports = []
    for supplied in paths:
        path, raw, panel, digest = load_canonical(
            supplied, "focused-start exclusion panel"
        )
        schema = EXPECTED_EXCLUSION_PANELS.get(digest)
        if schema is None or panel.get("schema") != schema or digest in observed:
            raise PanelError("exclusion panel identity/schema is not frozen")
        traps = panel.get("trap_states")
        if (
            panel.get("source_sha256") != SOURCE_SHA256
            or not isinstance(traps, list) or len(traps) != 96
        ):
            raise PanelError("exclusion panel trap coverage is stale")
        panel_keys = {str(entry.get("canonical_key")) for entry in traps}
        panel_games = {
            (str(entry.get("run_id")), str(entry.get("game_id")))
            for entry in traps
        }
        if len(panel_keys) != 96:
            raise PanelError("exclusion panel canonical roots are duplicated")
        observed[digest] = panel_keys
        keys.update(panel_keys)
        games.update(panel_games)
        reports.append({
            "bytes": len(raw), "path": display_path(path),
            "schema": schema, "sha256": digest,
            "source_games": len(panel_games), "trap_roots": len(panel_keys),
        })
    if set(observed) != set(EXPECTED_EXCLUSION_PANELS):
        raise PanelError("both exact v1/v2 exclusion panels are required")
    if len({frozenset(value) for value in observed.values()}) != 1:
        raise PanelError("v1/v2 training-start trap keys disagree")
    return keys, games, sorted(reports, key=lambda item: item["sha256"])


def load_focused_runs(
    paths: Sequence[pathlib.Path],
    collectors: Mapping[str, restart.CollectorInput],
) -> tuple[set[str], set[tuple[str, str]], list[dict[str, Any]]]:
    reports = []
    observed_manifests = set()
    focused_keys: set[str] = set()
    focused_games: set[tuple[str, str]] = set()
    observed_runs = set()
    for supplied in paths:
        path, raw, manifest, digest = load_canonical(
            supplied, "focused continuation manifest"
        )
        if digest not in EXPECTED_FOCUSED_MANIFESTS or digest in observed_manifests:
            raise PanelError("focused continuation manifest identity is not frozen")
        input_meta = manifest.get("input")
        config = manifest.get("config")
        selected = manifest.get("selected_prefixes")
        metadata = input_meta.get("metadata") if isinstance(input_meta, dict) else None
        run_id = metadata.get("run_id") if isinstance(metadata, dict) else None
        collector = collectors.get(run_id)
        if (
            manifest.get("schema") != restart.RUN_SCHEMA
            or collector is None or run_id in observed_runs
            or not isinstance(config, dict)
            or config.get("continuations_per_prefix") != 4
            or config.get("records") != 96
            or not isinstance(selected, list) or len(selected) != 24
            or input_meta.get("sha256") != collector.sha256
            or metadata.get("asserted_source_sha256") != SOURCE_SHA256
            or input_meta.get("selected_prefixes_path")
            != restart.ARCHIVED_SELECTED_PREFIXES_NAME
        ):
            raise PanelError("focused continuation manifest contract is stale")
        selected_path = safe_file(
            path.parent / input_meta["selected_prefixes_path"],
            "focused selected-prefix manifest",
        )
        selected_raw = selected_path.read_bytes()
        selected_sha = sha256_bytes(selected_raw)
        if selected_sha != input_meta.get("selected_prefixes_sha256"):
            raise PanelError("focused selected-prefix bytes are stale")
        try:
            requests, parsed_sha = restart.parse_selected_prefix_bytes(
                selected_raw, collector
            )
            replayed = restart.select_manifest_prefixes(collector, selected_raw)
        except ValueError as error:
            raise PanelError("focused selected-prefix replay failed") from error
        if (
            parsed_sha != selected_sha or len(requests) != 24
            or [dataclasses.asdict(item) for item in replayed] != selected
        ):
            raise PanelError("focused selected-prefix manifest disagrees with run")
        run_keys = {request.canonical_key for request in requests}
        run_games = {(run_id, request.game_id) for request in requests}
        if len(run_keys) != 24:
            raise PanelError("focused run repeats a canonical starting root")
        focused_keys.update(run_keys)
        focused_games.update(run_games)
        observed_runs.add(run_id)
        observed_manifests.add(digest)
        reports.append({
            "bytes": len(raw), "manifest_path": display_path(path),
            "manifest_sha256": digest, "run_id": run_id,
            "selected_prefix_bytes": len(selected_raw),
            "selected_prefix_path": display_path(selected_path),
            "selected_prefix_sha256": selected_sha,
            "starting_games": len(run_games), "starting_roots": len(run_keys),
        })
    if (
        observed_manifests != EXPECTED_FOCUSED_MANIFESTS
        or observed_runs != set(collectors)
        or len(focused_keys) != 96
    ):
        raise PanelError("focused continuation coverage is not exactly 96 roots")
    return focused_keys, focused_games, sorted(
        reports, key=lambda item: item["run_id"]
    )


def one_per_game(entries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    role_priority = {
        "two-own-before-mate": 0,
        "last-enemy-shell": 1,
        "one-own-before-mate": 2,
    }
    chosen: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in entries:
        key = entry["run_id"], entry["game_id"]
        marker = (
            role_priority[entry["role"]], entry["prefix_turn"],
            entry["canonical_key"],
        )
        current = chosen.get(key)
        if current is None or marker < (
            role_priority[current["role"]], current["prefix_turn"],
            current["canonical_key"],
        ):
            chosen[key] = entry
    return sorted(chosen.values(), key=lambda entry: (
        entry["run_id"], entry["candidate_player"],
        int(entry["game_id"]), entry["prefix_turn"], entry["canonical_key"],
    ))


def allocate_quotas(capacities: Mapping[str, int], target: int) -> dict[str, int]:
    runs = sorted(capacities)
    if not runs or any(capacities[run] < 1 for run in runs) or sum(
            capacities.values()) < target or target < len(runs):
        raise PanelError("independent trap capacity cannot satisfy color balance")
    quotas = {run: 1 for run in runs}
    remaining = target - len(runs)
    while remaining:
        progressed = False
        for run in runs:
            if quotas[run] < capacities[run]:
                quotas[run] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise PanelError("independent trap quota allocation exhausted")
    return quotas


def select_traps(
    all_traps: Sequence[dict[str, Any]], excluded_keys: set[str],
    excluded_games: set[tuple[str, str]], runs: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eligible = [entry for entry in all_traps if (
        entry["canonical_key"] not in excluded_keys
        and (entry["run_id"], entry["game_id"]) not in excluded_games
    )]
    representatives = one_per_game(eligible)
    selected = []
    quota_report = []
    for color in (0, 1):
        groups = {
            run: [entry for entry in representatives if (
                entry["run_id"] == run and entry["candidate_player"] == color
            )]
            for run in sorted(runs)
        }
        capacities = {run: len(entries) for run, entries in groups.items()}
        quotas = allocate_quotas(capacities, TARGET_TRAPS_PER_COLOR)
        for run in sorted(runs):
            chosen = trap_builder.evenly_spaced(groups[run], quotas[run])
            selected.extend(chosen)
            quota_report.append({
                "available_source_games": capacities[run],
                "candidate_player": color,
                "run_id": run,
                "selected_source_games": len(chosen),
            })
    selected.sort(key=lambda entry: (
        entry["run_id"], entry["candidate_player"],
        int(entry["game_id"]), entry["prefix_turn"],
    ))
    selected_keys = {entry["canonical_key"] for entry in selected}
    selected_games = {(entry["run_id"], entry["game_id"]) for entry in selected}
    if (
        len(selected) != TARGET_TRAPS or len(selected_keys) != TARGET_TRAPS
        or len(selected_games) != TARGET_TRAPS
        or selected_keys & excluded_keys or selected_games & excluded_games
        or collections.Counter(entry["candidate_player"] for entry in selected)
        != {0: TARGET_TRAPS_PER_COLOR, 1: TARGET_TRAPS_PER_COLOR}
    ):
        raise PanelError("independent trap selection is not exactly balanced")
    return selected, {
        "deduplicated_trap_roots": len(all_traps),
        "eligible_roots_after_exclusion": len(eligible),
        "eligible_source_games_after_exclusion": len(representatives),
        "quotas": quota_report,
    }


def select_controls(
    games: Sequence[dict[str, Any]], traps: Sequence[dict[str, Any]],
    forbidden_keys: set[str], all_trap_keys: set[str],
) -> list[dict[str, Any]]:
    candidates_by_group: dict[tuple[str, int], list[dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    seen_keys = set()
    for game in games:
        if game["winner"] != game["candidate_player"]:
            continue
        for item in game["candidate_geometry"]:
            entry = trap_builder.make_entry(
                game, item, "matched-winning-control", None
            )
            key = entry["canonical_key"]
            if key in forbidden_keys or key in all_trap_keys or key in seen_keys:
                continue
            seen_keys.add(key)
            candidates_by_group[(entry["run_id"], entry["candidate_player"])].append(
                entry
            )
    controls = []
    used_games = set()
    used_keys = set()
    for trap in sorted(traps, key=lambda entry: (
        entry["run_id"], entry["candidate_player"], int(entry["game_id"]),
        entry["prefix_turn"],
    )):
        pool = [entry for entry in candidates_by_group[
            (trap["run_id"], trap["candidate_player"])
        ] if (
            (entry["run_id"], entry["game_id"]) not in used_games
            and entry["canonical_key"] not in used_keys
        )]
        if not pool:
            raise PanelError("matched control capacity is exhausted")
        chosen = min(pool, key=lambda control: (
            control["turn_band"] != trap["turn_band"],
            control["zone"] != trap["zone"],
            control["used_edge_band"] != trap["used_edge_band"],
            abs(control["prefix_turn"] - trap["prefix_turn"]),
            abs(control["used_edges"] - trap["used_edges"]),
            int(control["game_id"]), control["prefix_turn"],
            control["canonical_key"],
        ))
        chosen["matched_trap_state_id"] = trap["state_id"]
        chosen["match_exact"] = {
            "color": True,
            "run_id": True,
            "turn_band": chosen["turn_band"] == trap["turn_band"],
            "used_edge_band": (
                chosen["used_edge_band"] == trap["used_edge_band"]
            ),
            "zone": chosen["zone"] == trap["zone"],
        }
        controls.append(chosen)
        used_games.add((chosen["run_id"], chosen["game_id"]))
        used_keys.add(chosen["canonical_key"])
    trap_keys = {entry["canonical_key"] for entry in traps}
    if (
        len(controls) != TARGET_CONTROLS or len(used_games) != TARGET_CONTROLS
        or len(used_keys) != TARGET_CONTROLS or used_keys & trap_keys
        or used_keys & forbidden_keys
        or collections.Counter(entry["candidate_player"] for entry in controls)
        != {0: TARGET_TRAPS_PER_COLOR, 1: TARGET_TRAPS_PER_COLOR}
    ):
        raise PanelError("independent control selection is not exactly balanced")
    return controls


def build_panel(
    sources: Sequence[trap_builder.Source], exclusion_paths: Sequence[pathlib.Path],
    focused_paths: Sequence[pathlib.Path],
) -> dict[str, Any]:
    games, source_reports = trap_builder.load_all(list(sources))
    collectors = {
        report["run_id"]: restart.parse_collector_bytes(
            (ROOT / report["collector_tsv_path"]).read_bytes()
        )
        for report in source_reports
    }
    excluded_keys, excluded_games, exclusion_reports = load_exclusion_panels(
        exclusion_paths
    )
    focused_keys, focused_games, focused_reports = load_focused_runs(
        focused_paths, collectors
    )
    if focused_keys != excluded_keys or not focused_games.issubset(excluded_games):
        raise PanelError("focused continuations disagree with exclusion panels")
    all_traps = trap_builder.deduplicate(trap_builder.trap_entries(games))
    selected_traps, selection_report = select_traps(
        all_traps, excluded_keys | focused_keys,
        excluded_games | focused_games, sorted(collectors),
    )
    controls = select_controls(
        games, selected_traps, excluded_keys | focused_keys,
        {entry["canonical_key"] for entry in all_traps},
    )
    trap_builder.require_disjoint_populations(selected_traps, controls)
    selected_games = {
        (entry["run_id"], entry["game_id"])
        for entry in selected_traps + controls
    }
    if (
        len(selected_games) != TARGET_TRAPS + TARGET_CONTROLS
        or selected_games & (excluded_games | focused_games)
    ):
        raise PanelError(
            "evaluation panel duplicates or reuses a focused source game"
        )
    return {
        "counts": {
            "clean_games": len(games),
            "excluded_focused_source_games": len(focused_games),
            "excluded_focused_start_roots": len(focused_keys),
            "matched_controls": len(controls),
            "trap_states": len(selected_traps),
        },
        "dependencies": {
            "exclusion_panels": exclusion_reports,
            "focused_continuations": focused_reports,
        },
        "independence": {
            "contract": INDEPENDENCE,
            "control_source_games_distinct": True,
            "focused_start_canonical_overlap": 0,
            "focused_start_source_game_overlap": 0,
            "one_selected_root_per_source_game": True,
            "scope": (
                "root/source-game held out from focused continuation starts "
                "only; same frozen public arena source family"
            ),
        },
        "matched_winning_controls": controls,
        "mate_onset": (
            "first actual fixed30k backed value below -100; fallback first "
            "chosen value below -100"
        ),
        "purpose": {
            "diagnostic_only": True,
            "observed_moves_usage": trap_builder.OBSERVED_USAGE,
            "training_eligible": False,
        },
        "schema": SCHEMA,
        "selection": {
            "control_matching": (
                "same-run/same-color nearest bands then distances/v1"
            ),
            "trap_role_priority": [
                "two-own-before-mate", "last-enemy-shell",
                "one-own-before-mate",
            ],
            **selection_report,
        },
        "source_sha256": SOURCE_SHA256,
        "sources": source_reports,
        "trap_states": selected_traps,
    }


def write_once(path: pathlib.Path, panel: Mapping[str, Any]) -> str:
    raw = canonical_json_bytes(panel)
    digest = sha256_bytes(raw)
    path = path.resolve()
    if path.exists():
        if path.read_bytes() != raw:
            raise PanelError(f"refusing to replace different panel: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as output:
            output.write(raw)
    return digest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", action="append", nargs=3, required=True,
        metavar=("MANIFEST", "FIXED30K_AUDIT", "COLLECTOR_TSV"),
        type=pathlib.Path,
    )
    parser.add_argument(
        "--exclude-panel", action="append", required=True, type=pathlib.Path
    )
    parser.add_argument(
        "--focused-manifest", action="append", required=True,
        type=pathlib.Path,
    )
    parser.add_argument("--output", required=True, type=pathlib.Path)
    arguments = parser.parse_args(argv)
    try:
        panel = build_panel(
            trap_builder.parse_sources(arguments.source),
            arguments.exclude_panel,
            arguments.focused_manifest,
        )
        digest = write_once(arguments.output, panel)
    except (OSError, PanelError, ValueError) as error:
        print(f"late-pacing evaluation panel failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({
        "counts": panel["counts"], "output": str(arguments.output),
        "sha256": digest,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
