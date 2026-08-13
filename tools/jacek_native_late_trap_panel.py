#!/usr/bin/env python3
"""Build a provenance-bound late-trap diagnostic panel from explicit archives.

The panel is diagnostic and restart-state construction only. Observed complete
turns locate replay states; they are never action or value labels.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import json
import pathlib
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOL_DIR = pathlib.Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))
import jacek_native_corpus as native  # noqa: E402
import jacek_native_restart_corpus_round2 as restart  # noqa: E402


SCHEMA = "papersoccer.jacek-native-late-trap-panel/v2"
SELECTION_SCHEMA = "papersoccer.jacek-native-selected-prefixes.v1"
REVIEW_SCHEMA = "papersoccer.jacek-native-visual-review-queue/v1"
SOURCE_SHA256 = (
    "653eba7d4b5f9b3e8737a6fb50bf16945e416bcdbc53e72520a6ee68acbbef90"
)
OBSERVED_USAGE = "state-construction-only"
ELITE_NAMES = {
    "jacek", "Deltaspace", "Marchete", "Snekkers", "Laars",
    "Waffle3z", "EricSMSO", "derjack",
}
FORBIDDEN = (
    "matches.json", "protected", "sealed", "prospective", "final-bank",
    "final_bank",
)


@dataclasses.dataclass(frozen=True)
class Source:
    manifest_path: pathlib.Path
    audit_path: pathlib.Path
    collector_path: pathlib.Path


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(
        value, sort_keys=True, allow_nan=False, ensure_ascii=False,
        separators=(",", ":")
    ) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: pathlib.Path) -> str:
    return sha256_bytes(path.read_bytes())


def display_path(path: pathlib.Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def safe_explicit(path: pathlib.Path, label: str) -> pathlib.Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{label} is not an explicit file: {path}") from error
    for candidate in (path, resolved):
        rendered = str(candidate).lower()
        parts = tuple(part.lower() for part in candidate.parts)
        if (
            any(token in rendered for token in FORBIDDEN)
            or any(part == "final" or part.startswith("final.") for part in parts)
        ):
            raise ValueError(f"{label} path contains forbidden evidence")
    if not resolved.is_file():
        raise ValueError(f"{label} is not an explicit file: {path}")
    return resolved


def parse_sources(values: list[list[pathlib.Path]]) -> list[Source]:
    result = []
    for manifest, audit, collector in values:
        result.append(Source(
            safe_explicit(manifest, "arena manifest"),
            safe_explicit(audit, "decision audit"),
            safe_explicit(collector, "collector TSV"),
        ))
    if not result:
        raise ValueError("at least one explicit source pair is required")
    return result


def clean_records(manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
    coverage = manifest.get("coverage") or {}
    if not coverage.get("full_window_accounted"):
        raise ValueError("arena manifest is not fully accounted")
    source = ((manifest.get("binding") or {}).get("source") or {})
    if source.get("sha256") != SOURCE_SHA256:
        raise ValueError("arena manifest is not the exact retained source")
    result = {}
    for stored in manifest.get("games", []):
        record = stored.get("record") or {}
        if (
            record.get("status") == "accepted"
            and (record.get("operational") or {}).get("classification") == "clean"
            and (((record.get("replay") or {}).get("rules_validation") or {}).get(
                "status") == "terminal-valid")
        ):
            result[int(record["game_id"])] = record
    if len(result) != int(coverage.get("clean_rule_terminal_games", -1)):
        raise ValueError("arena clean-game coverage is inconsistent")
    return result


def load_audit(path: pathlib.Path) -> tuple[dict[int, list[dict[str, Any]]], str]:
    raw = path.read_bytes()
    rows: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    provenance = None
    for number, line in enumerate(raw.splitlines(), 1):
        if not line:
            raise ValueError(f"blank audit row at {path}:{number}")
        row = json.loads(line)
        if row.get("schema_version") not in {
            "jacek-native-decision-audit-v1",
            "jacek-native-decision-audit-v2",
        }:
            raise ValueError("decision audit schema is unsupported")
        if row.get("audit_mode") != "fixed-work" or row.get("fixed_work_limit") != 30000:
            raise ValueError("decision audit must be the exact fixed-30k profile")
        if provenance is None:
            provenance = row.get("input_provenance")
        elif row.get("input_provenance") != provenance:
            raise ValueError("decision audit provenance changes between rows")
        rows[int(row["game_id"])].append(row)
    if not rows or not isinstance(provenance, dict):
        raise ValueError("decision audit is empty")
    for game_rows in rows.values():
        game_rows.sort(key=lambda row: int(row["own_decision_index"]))
        if [int(row["own_decision_index"]) for row in game_rows] != list(
            range(len(game_rows))
        ):
            raise ValueError("decision audit does not cover each own decision")
    return dict(rows), sha256_bytes(raw)


def prefix_geometry(actions: tuple[str, ...]) -> list[dict[str, Any]]:
    state = native._initial_replay_state()
    result = []
    for turn, action in enumerate(actions):
        active = native._encode_replay_features(state)
        reflected = native._encode_replay_features(state, reflected=True)
        reflected_is_canonical = reflected < active
        canonical_active = reflected if reflected_is_canonical else active
        mover_ball = native._transform_feature_point(
            state.ball, state.to_move, reflected_is_canonical
        )
        result.append({
            "turn": turn,
            "player": state.to_move,
            "ball_x": mover_ball[0],
            "ball_y": mover_ball[1],
            "used_edges": len(state.used_segments),
            "state_id": native.canonical_state_id(active),
            "canonical_key": native.canonical_state_id(canonical_active),
        })
        native._apply_complete_turn(state, action, turn, 1, opening=False)
    if state.winner is None:
        raise ValueError("clean transcript is not terminal")
    return result


def zone(ball_y: int) -> str:
    if ball_y <= 4:
        return "enemy-shell"
    if ball_y <= 7:
        return "middle"
    return "own-shell"


def turn_band(turn: int) -> str:
    if turn < 24:
        return "early"
    if turn < 48:
        return "middle"
    return "late"


def edge_band(count: int) -> str:
    if count < 64:
        return "sparse"
    if count < 128:
        return "building"
    return "closed"


def load_all(sources: list[Source]):
    games = []
    source_reports = []
    seen_game_ids = set()
    for source in sources:
        manifest_raw = source.manifest_path.read_bytes()
        manifest = json.loads(manifest_raw)
        if canonical_json_bytes(manifest) != manifest_raw:
            raise ValueError("arena manifest is not canonical JSON")
        manifest_hash = sha256_bytes(manifest_raw)
        if source.manifest_path.stem != manifest_hash:
            raise ValueError("arena manifest path is not content-addressed")
        records = clean_records(manifest)
        audits, audit_hash = load_audit(source.audit_path)
        collector_raw = source.collector_path.read_bytes()
        collector = restart.parse_collector_bytes(collector_raw)
        binding = manifest["binding"]
        provenance = next(iter(audits.values()))[0]["input_provenance"]
        if (
            provenance.get("arena_manifest_sha256") != manifest_hash
            or provenance.get("asserted_source_sha256") != SOURCE_SHA256
            or str(provenance.get("agent_id")) != str(binding["agent_id"])
            or str(provenance.get("asserted_submission_id"))
            != str(binding["asserted_submission_id"])
            or set(audits) != set(records)
            or collector.metadata["arena_manifest_sha256"] != manifest_hash
            or collector.metadata["asserted_source_sha256"] != SOURCE_SHA256
            or {int(game.game_id) for game in collector.games} != set(records)
        ):
            raise ValueError("arena manifest and decision audit do not bind exactly")
        source_reports.append({
            "run_id": manifest["run_id"],
            "agent_id": str(binding["agent_id"]),
            "submission_id": str(binding["asserted_submission_id"]),
            "manifest_path": source.manifest_path.relative_to(ROOT).as_posix(),
            "manifest_sha256": manifest_hash,
            "decision_audit_path": source.audit_path.relative_to(ROOT).as_posix(),
            "decision_audit_sha256": audit_hash,
            "collector_tsv_path": source.collector_path.relative_to(ROOT).as_posix(),
            "collector_tsv_sha256": sha256_bytes(collector_raw),
            "clean_games": len(records),
        })
        for game_id, record in records.items():
            if game_id in seen_game_ids:
                raise ValueError(f"game {game_id} repeats across source histories")
            seen_game_ids.add(game_id)
            actions = tuple(record["replay"]["valid_transcript"].split("/"))
            # Replay twice by independent maintained validators: the panel state
            # builder and the already-frozen fixed-30k decision audit.
            geometry = prefix_geometry(actions)
            rows = audits[game_id]
            candidate = int(record["focus"]["player_id"])
            winner = int(record["outcome"]["winner_player_id"])
            candidate_geometry = [item for item in geometry if item["player"] == candidate]
            if len(candidate_geometry) != len(rows):
                raise ValueError("candidate decision geometry/audit coverage differs")
            for item, row in zip(candidate_geometry, rows):
                if (
                    item["turn"] != int(row["turn_index"])
                    or "/".join(actions[:item["turn"]])
                    != str(row["transcript_prefix"])
                    or candidate != int(row["candidate_player"])
                ):
                    raise ValueError("candidate decision states do not align")
                item["auditor_state_id"] = str(row["state_id"])
            games.append({
                "run_id": manifest["run_id"], "manifest_sha256": manifest_hash,
                "collector_tsv_sha256": sha256_bytes(collector_raw),
                "game_id": str(game_id), "candidate_player": candidate,
                "winner": winner, "actions": actions, "geometry": geometry,
                "candidate_geometry": candidate_geometry, "audit": rows,
                "opponent": record["opponent"],
            })
    return games, source_reports


def make_entry(game: dict[str, Any], item: dict[str, Any], role: str,
               mate_turn: int | None) -> dict[str, Any]:
    actions = game["actions"]
    prefix_turn = int(item["turn"])
    return {
        "role": role,
        "run_id": game["run_id"],
        "arena_manifest_sha256": game["manifest_sha256"],
        "game_id": game["game_id"],
        "candidate_player": game["candidate_player"],
        "observed_winner": game["winner"],
        "observed_result": (
            "win" if game["winner"] == game["candidate_player"] else "loss"
        ),
        "opponent": {
            "agent_id": int(game["opponent"]["agent_id"]),
            "name": str(game["opponent"]["name"]),
            "frozen_rank": game["opponent"].get("frozen_rank"),
        },
        "prefix_turn": prefix_turn,
        "candidate_own_decision": next(
            index for index, value in enumerate(game["candidate_geometry"])
            if value["turn"] == prefix_turn
        ),
        "observed_turn_count": len(actions),
        "transcript": "/".join(actions[:prefix_turn]),
        "state_id": item["state_id"],
        "auditor_state_id": item["auditor_state_id"],
        "canonical_key": item["canonical_key"],
        "ball_x": item["ball_x"],
        "ball_y": item["ball_y"],
        "zone": zone(item["ball_y"]),
        "turn_band": turn_band(prefix_turn),
        "used_edges": item["used_edges"],
        "used_edge_band": edge_band(item["used_edges"]),
        "mate_onset_turn": mate_turn,
        "observed_moves_usage": OBSERVED_USAGE,
        "policy_target": None,
        "value_target": None,
        "training_eligible": False,
    }


def trap_entries(games: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for game in games:
        if game["winner"] == game["candidate_player"]:
            continue
        rows = game["audit"]
        geometry = game["candidate_geometry"]
        mate_index = next((
            index for index, row in enumerate(rows)
            if row.get("actual_final_backed_value") is not None
            and float(row["actual_final_backed_value"]) < -100.0
        ), None)
        if mate_index is None:
            mate_index = next((
                index for index, row in enumerate(rows)
                if float(row["chosen_value"]) < -100.0
            ), None)
        if mate_index is None:
            continue
        mate_turn = int(geometry[mate_index]["turn"])
        shell_before = [
            index for index, item in enumerate(geometry[:mate_index + 1])
            if item["ball_y"] <= 4
        ]
        roles = []
        if shell_before:
            roles.append((shell_before[-1], "last-enemy-shell"))
        for offset, role in ((2, "two-own-before-mate"), (1, "one-own-before-mate")):
            if mate_index >= offset:
                roles.append((mate_index - offset, role))
        for index, role in roles:
            result.append(make_entry(game, geometry[index], role, mate_turn))
    return result


def deduplicate(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chosen: dict[str, dict[str, Any]] = {}
    priority = {"one-own-before-mate": 0, "two-own-before-mate": 1,
                "last-enemy-shell": 2, "matched-winning-control": 3}
    for entry in entries:
        key = entry["canonical_key"]
        current = chosen.get(key)
        marker = (priority[entry["role"]], entry["run_id"], int(entry["game_id"]),
                  entry["prefix_turn"])
        if current is None:
            chosen[key] = entry
        else:
            old = (priority[current["role"]], current["run_id"],
                   int(current["game_id"]), current["prefix_turn"])
            if marker < old:
                chosen[key] = entry
    return sorted(chosen.values(), key=lambda item: (
        item["role"], item["run_id"], int(item["game_id"]), item["prefix_turn"]
    ))


def evenly_spaced(values: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count >= len(values):
        return values
    if count == 1:
        return [values[len(values) // 2]]
    return [values[index * (len(values) - 1) // (count - 1)]
            for index in range(count)]


def balance_traps(entries: list[dict[str, Any]], maximum_per_run: int = 24):
    selected = []
    for run_id in sorted({entry["run_id"] for entry in entries}):
        run = [entry for entry in entries if entry["run_id"] == run_id]
        chosen = []
        for role in ("last-enemy-shell", "two-own-before-mate",
                     "one-own-before-mate"):
            for color in (0, 1):
                group = sorted((entry for entry in run if (
                    entry["role"] == role and entry["candidate_player"] == color
                )), key=lambda item: (int(item["game_id"]), item["prefix_turn"]))
                chosen.extend(evenly_spaced(group, min(4, len(group))))
        chosen_keys = {entry["canonical_key"] for entry in chosen}
        remaining = [entry for entry in run if entry["canonical_key"] not in chosen_keys]
        chosen.extend(evenly_spaced(remaining, min(
            maximum_per_run - len(chosen), len(remaining)
        )))
        selected.extend(chosen[:maximum_per_run])
    return sorted(selected, key=lambda item: (
        item["run_id"], item["role"], item["candidate_player"],
        int(item["game_id"]), item["prefix_turn"]
    ))


def control_entries(games: list[dict[str, Any]], traps: list[dict[str, Any]]):
    trap_keys = {entry["canonical_key"] for entry in traps}
    candidates = []
    for game in games:
        if game["winner"] != game["candidate_player"]:
            continue
        for item in game["candidate_geometry"]:
            entry = make_entry(game, item, "matched-winning-control", None)
            if entry["canonical_key"] not in trap_keys:
                candidates.append(entry)
    unused = {entry["canonical_key"]: entry for entry in candidates}
    controls = []
    for trap in sorted(traps, key=lambda item: (
        item["run_id"], int(item["game_id"]), item["prefix_turn"], item["role"]
    )):
        pool = list(unused.values())
        def metric(control):
            return (
                control["candidate_player"] != trap["candidate_player"],
                control["turn_band"] != trap["turn_band"],
                control["zone"] != trap["zone"],
                control["used_edge_band"] != trap["used_edge_band"],
                abs(control["prefix_turn"] - trap["prefix_turn"]),
                abs(control["used_edges"] - trap["used_edges"]),
                control["run_id"], int(control["game_id"]), control["prefix_turn"],
            )
        if not pool:
            break
        chosen = min(pool, key=metric)
        chosen["matched_trap_state_id"] = trap["state_id"]
        chosen["match_exact"] = {
            "color": chosen["candidate_player"] == trap["candidate_player"],
            "turn_band": chosen["turn_band"] == trap["turn_band"],
            "zone": chosen["zone"] == trap["zone"],
            "used_edge_band": chosen["used_edge_band"] == trap["used_edge_band"],
        }
        controls.append(chosen)
        unused.pop(chosen["canonical_key"], None)
    return controls


def require_disjoint_populations(
    traps: list[dict[str, Any]], controls: list[dict[str, Any]],
) -> None:
    for field in ("canonical_key", "state_id", "auditor_state_id"):
        trap_ids = [str(entry[field]) for entry in traps]
        control_ids = [str(entry[field]) for entry in controls]
        if (
            len(set(trap_ids)) != len(trap_ids)
            or len(set(control_ids)) != len(control_ids)
            or set(trap_ids) & set(control_ids)
        ):
            raise ValueError(
                f"late-trap v2 populations are not disjoint by {field}"
            )


def visual_queue(games: list[dict[str, Any]]) -> list[dict[str, Any]]:
    losses = [game for game in games if game["winner"] != game["candidate_player"]]
    required = [game for game in losses if (
        game["opponent"]["name"] == "jacek"
        or (game["opponent"].get("frozen_rank") is not None
            and int(game["opponent"]["frozen_rank"]) <= 5)
    )]
    remaining = [game for game in losses if game not in required]
    selected = list(required)
    for color in (0, 1):
        pool = sorted(
            (game for game in remaining if game["candidate_player"] == color),
            key=lambda game: (game["run_id"], int(game["game_id"])),
        )
        selected.extend(pool[:6])
    unique = {(game["run_id"], game["game_id"]): game for game in selected}
    result = []
    for game in sorted(unique.values(), key=lambda game: (
        game["opponent"].get("frozen_rank") or 10**9,
        game["opponent"]["name"], game["run_id"], int(game["game_id"]),
    )):
        result.append({
            "run_id": game["run_id"], "game_id": game["game_id"],
            "candidate_player": game["candidate_player"],
            "opponent_name": game["opponent"]["name"],
            "opponent_agent_id": int(game["opponent"]["agent_id"]),
            "opponent_frozen_rank": game["opponent"].get("frozen_rank"),
            "selection": (
                "jacek" if game["opponent"]["name"] == "jacek"
                else "top5" if int(game["opponent"].get("frozen_rank") or 999) <= 5
                else "balanced-other"
            ),
            "observed_turn_count": len(game["actions"]),
            "observed_moves_usage": OBSERVED_USAGE,
            "training_eligible": False,
        })
    return result


def write_once(path: pathlib.Path, value: Any) -> str:
    raw = canonical_json_bytes(value)
    if path.exists():
        if path.read_bytes() != raw:
            raise RuntimeError(f"refusing to replace different artifact: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    return sha256_bytes(raw)


def write_once_bytes(path: pathlib.Path, raw: bytes) -> str:
    if path.exists():
        if path.read_bytes() != raw:
            raise RuntimeError(f"refusing to replace different artifact: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as output:
            output.write(raw)
    return sha256_bytes(raw)


def selected_prefix_bytes(source: dict[str, Any], panel_sha: str,
                          entries: list[dict[str, Any]]) -> bytes:
    metadata = {
        "arena_manifest_sha256": source["manifest_sha256"],
        "collector_tsv_sha256": source["collector_tsv_sha256"],
        "observed_moves_usage": OBSERVED_USAGE,
        "panel_sha256": panel_sha,
        "policy_target": "null",
        "schema": SELECTION_SCHEMA,
        "source_sha256": SOURCE_SHA256,
        "value_target": "null",
    }
    text = "".join(f"# {key}={value}\n" for key, value in sorted(metadata.items()))
    text += ("game_id\tcandidate_own_decision\tprefix_turn\tstate_id\t"
             "canonical_key\trole\n")
    for entry in sorted(entries, key=lambda item: (
        int(item["game_id"]), item["prefix_turn"], item["role"]
    )):
        text += (
            f"{entry['game_id']}\t{entry['candidate_own_decision']}\t"
            f"{entry['prefix_turn']}\t{entry['state_id']}\t"
            f"{entry['canonical_key']}\t{entry['role']}\n"
        )
    return text.encode("ascii")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", nargs=3, required=True,
                        metavar=("MANIFEST", "FIXED30K_AUDIT", "COLLECTOR_TSV"),
                        type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--selected-prefix-dir", required=True, type=pathlib.Path)
    parser.add_argument("--review-output", required=True, type=pathlib.Path)
    arguments = parser.parse_args()
    games, source_reports = load_all(parse_sources(arguments.source))
    traps = balance_traps(deduplicate(trap_entries(games)))
    controls = deduplicate(control_entries(games, traps))
    require_disjoint_populations(traps, controls)
    panel = {
        "schema": SCHEMA, "source_sha256": SOURCE_SHA256,
        "purpose": {"diagnostic_only": True, "training_eligible": False,
                    "observed_moves_usage": OBSERVED_USAGE},
        "mate_onset": "first actual fixed30k backed value below -100; fallback first chosen value below -100",
        "matching": ["candidate_player", "turn_band", "zone", "used_edge_band",
                     "absolute turn distance", "absolute used-edge distance"],
        "sources": source_reports,
        "counts": {"clean_games": len(games), "trap_states": len(traps),
                   "matched_controls": len(controls)},
        "trap_states": traps, "matched_winning_controls": controls,
    }
    panel_sha = write_once(arguments.output.resolve(), panel)
    by_run: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for entry in traps:
        by_run[entry["run_id"]].append(entry)
    selected_reports = []
    source_by_run = {source["run_id"]: source for source in source_reports}
    for run_id, entries in sorted(by_run.items()):
        source = source_by_run[run_id]
        path = arguments.selected_prefix_dir.resolve() / f"{run_id}.tsv"
        digest = write_once_bytes(
            path, selected_prefix_bytes(source, panel_sha, entries)
        )
        selected_reports.append({"run_id": run_id, "path": display_path(path),
                                 "sha256": digest, "prefixes": len(entries)})
    queue = {"schema": REVIEW_SCHEMA, "source_sha256": SOURCE_SHA256,
             "observed_moves_usage": OBSERVED_USAGE,
             "selection_rule": "all clean Jacek/top5 losses plus six other losses per candidate color",
             "games": visual_queue(games)}
    review_sha = write_once(arguments.review_output.resolve(), queue)
    print(json.dumps({"panel_sha256": panel_sha, "counts": panel["counts"],
                      "selected_prefixes": selected_reports,
                      "review_sha256": review_sha,
                      "review_games": len(queue["games"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
