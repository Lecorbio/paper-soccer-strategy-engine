"""Read completed rebuild evidence without replaying training or game gates.

This checks the saved artifact graph and its internal bindings. It does not
recompute predictions, losses, games, or qualification decisions. The existing
strict replay validators remain the authority for an explicit computational
audit. Keep this module standard-library-only so completion checks cannot load
training datasets or create numerical workers as an import side effect.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


TERMINALS = {
    "no-development-qualified-candidate",
    "qualified-local-research-incumbent",
    "final-qualification-rejected",
}
PHASES = ("v5-recovery", "canonical-basins", "scratch-joint", "residual")
PREFIX = "papersoccer.jacek-replay-rebuild-"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("completed rebuild: " + message)


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode()


def _object(pairs: list[tuple[str, object]]) -> dict:
    value = dict(pairs)
    _require(len(value) == len(pairs), "duplicate JSON keys")
    return value


def _constant(value: str) -> None:
    raise ValueError("completed rebuild: nonfinite JSON number " + value)


def _body(value: dict, schema: str | None = None) -> None:
    _require(isinstance(value, dict), "receipt must be an object")
    body = dict(value)
    claimed = body.pop("body_sha256", None)
    _require(schema is None or body.get("schema") == schema, "receipt schema changed")
    _require(claimed == hashlib.sha256(_canonical(body)).hexdigest(),
             "receipt body hash changed")


class _Files:
    """Stream file hashes once per observed file version; never decode arrays."""

    def __init__(self) -> None:
        self.cache: dict[tuple, dict] = {}

    def snapshot(self, path: Path) -> dict:
        path = path.resolve()
        try:
            stat = path.stat()
            _require(path.is_file(), f"not a regular file: {path}")
            key = (str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
            if key not in self.cache:
                digest = hashlib.sha256()
                with path.open("rb") as source:
                    while chunk := source.read(1024 * 1024):
                        digest.update(chunk)
                after = path.stat()
                _require((after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                         == key[1:], f"artifact changed while reading: {path}")
                self.cache[key] = {"path": str(path), "bytes": stat.st_size,
                                   "sha256": digest.hexdigest()}
            return self.cache[key]
        except OSError as error:
            raise ValueError(f"completed rebuild: missing artifact: {path}") from error

    def verify(self, record: dict) -> Path:
        _require(isinstance(record, dict) and isinstance(record.get("path"), str)
                 and type(record.get("bytes")) is int
                 and isinstance(record.get("sha256"), str), "invalid artifact binding")
        path = Path(record["path"])
        _require(path.is_absolute() and path.resolve() == path,
                 "artifact path is relative or redirected")
        actual = self.snapshot(path)
        _require(all(record[k] == actual[k] for k in actual), f"artifact changed: {path}")
        return path

    def read(self, path: Path) -> dict:
        try:
            value = json.loads(path.read_bytes(), object_pairs_hook=_object,
                               parse_constant=_constant)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"completed rebuild: cannot read metadata: {path}") from error
        _require(isinstance(value, dict), f"metadata is not an object: {path}")
        return value

    def bound_json(self, record: dict) -> dict:
        path = self.verify(record)
        value = self.read(path)
        # A second stat also catches replacement between verification and read.
        self.verify(record)
        return value

    def walk(self, value: object) -> None:
        if isinstance(value, list):
            for item in value:
                self.walk(item)
        elif isinstance(value, dict):
            if "body_sha256" in value:
                _body(value)
            if {"path", "bytes", "sha256"} <= value.keys():
                self.verify(value)
            for item in value.values():
                self.walk(item)


def _seed(files: _Files, row: dict, initial: dict, incumbent: dict) -> None:
    _require(set(row) == {"seed", "runtime", "report", "receipt", "result"},
             "seed record shape changed")
    receipt = files.bound_json(row["receipt"])
    _body(receipt)
    _require(receipt.get("schema") in {
        "papersoccer.jacek-replay-recovery-receipt.v1",
        "papersoccer.jacek-replay-v6-joint-receipt.v1",
        "papersoccer.jacek-replay-residual-recovery-receipt.v1",
    }, "unknown seed receipt")
    report = files.bound_json(row["report"])
    for key in ("configuration", "inputs", "producer"):
        _require(receipt.get(key) == report.get(key) and isinstance(receipt.get(key), dict),
                 f"seed {key} binding changed")
    _require(receipt["configuration"].get("optimizer", {}).get("seed") == row["seed"],
             "seed configuration changed")
    _require(report.get("result") == row["result"], "seed result differs from report")
    for key, digest_key in (("report", "sha256"), ("runtime", "artifact_sha256")):
        bound = receipt.get(key, {})
        _require(bound.get("file") == Path(row[key]["path"]).name
                 and bound.get(digest_key) == row[key]["sha256"]
                 and bound.get("bytes") == row[key]["bytes"], f"seed {key} changed")
    runtimes = receipt["inputs"].get("runtimes", {})
    for key, expected in (("initial", initial), ("new_reference", initial),
                          ("retention_reference", incumbent)):
        bound = runtimes.get(key, {})
        _require(bound.get("artifact_sha256") == expected["sha256"]
                 and bound.get("bytes") == expected["bytes"], "seed reference changed")
    files.walk(receipt)


def _selection_key(row: dict) -> tuple:
    metrics = row["result"]["selection"]
    return (float(metrics["weighted_huber"]), -float(metrics["sign_accuracy"]),
            -float(metrics["correlation"]), row["runtime"]["sha256"])


def _candidate(files: _Files, candidate: dict, spec: dict | None,
               inputs: dict, residual_base: dict, seeds: tuple[int, ...]) -> None:
    _body(candidate, PREFIX + "candidate.v1")
    _require(candidate.get("corpus") == inputs["corpus"]
             and candidate.get("incumbent") == inputs["incumbent_runtime"],
             "candidate belongs to different inputs")
    if spec is not None:
        _require(candidate.get("specification") == spec
                 and candidate.get("candidate_id") == spec["candidate_id"],
                 "candidate specification changed")
        initial = spec
    else:
        initial = residual_base
        _require(candidate.get("phase") == "residual" and candidate.get("rank") == 16,
                 "residual candidate changed")
    complete = []
    for arm in ("search", "rank4"):
        rows = candidate.get(arm + "_seed_runs")
        _require(isinstance(rows, list), "seed roster missing")
        _require([r.get("seed") for r in rows] in ([], list(seeds)), "seed roster changed")
        complete.append(bool(rows))
        for row in rows:
            _seed(files, row, initial[arm + "_initial_runtime"], inputs["incumbent_runtime"])
        selected = candidate.get("selected_" + arm)
        eligible = [row for row in rows if row["result"].get("eligible") is True]
        _require(selected == (min(eligible, key=_selection_key) if eligible else None),
                 "selected seed is not the recorded deterministic best")
    _require(complete[0] == complete[1], "candidate has an incomplete evaluator pair")
    if spec is not None:
        launch = candidate.get("launch")
        _require(bool(launch) == complete[0], "candidate launch/completion mismatch")
        if launch:
            record = files.bound_json(launch)
            _body(record, PREFIX + "config-launch.v1")
            _require(record.get("candidate_id") == spec["candidate_id"]
                     and record.get("rebuild_id") == inputs["rebuild_id"]
                     and record.get("specification") == spec
                     and record.get("corpus") == inputs["corpus"]
                     and record.get("incumbent") == inputs["incumbent_runtime"]
                     and record.get("deadline_unix") == inputs["same_architecture_deadline_unix"],
                     "candidate launch input binding changed")
            launched = record.get("launched_at_unix")
            _require(type(launched) in (int, float)
                     and inputs["started_at_unix"] <= launched <= inputs["same_architecture_deadline_unix"],
                     "candidate launch lies outside the frozen budget")
    _require(candidate.get("offline_eligible") is bool(
        candidate.get("selected_search") and candidate.get("selected_rank4")),
        "candidate eligibility changed")


def _phase(files: _Files, phase: dict, inputs: dict, matrix: dict,
           output: Path, seeds: tuple[int, ...]) -> None:
    _body(phase, PREFIX + "phase-screen.v1")
    name = phase["phase"]
    _require(files.read(output / name / "screening/phase-screen.json") == phase,
             "summary differs from saved phase screen")
    matrix_phases = matrix["phases"]
    if name in ("v5-recovery", "canonical-basins"):
        specs = matrix_phases["v5_recovery" if name == "v5-recovery" else "canonical_basins"]
        ids = [spec["candidate_id"] for spec in specs]
    elif name == "scratch-joint":
        lineage = files.bound_json(phase["phase_inputs"]["scratch_base_lineage"])
        _body(lineage, PREFIX + "scratch-bases.v1")
        files.walk(lineage)
        specs = lineage["selected_specs"]
        ids = [spec["candidate_id"] for spec in specs]
        _require(len(specs) == 3, "scratch roster changed")
    else:
        ids = ["residual-lr1e4", "residual-lr3e4", "residual-lr1e3"]
        specs = [None] * len(ids)
    candidates = phase.get("candidate_records")
    _require(isinstance(candidates, list)
             and [c.get("candidate_id") for c in candidates] == ids, "phase roster changed")
    for candidate, spec in zip(candidates, specs, strict=True):
        if name == "residual":
            rates = dict(zip(ids, (1e-4, 3e-4, 1e-3), strict=True))
            _require(candidate.get("learning_rate") == rates[candidate["candidate_id"]],
                     "residual rate changed")
        _candidate(files, candidate, spec, inputs, matrix_phases["v5_recovery"][0], seeds)
    by_id = {c["candidate_id"]: c for c in candidates}
    offline = phase.get("offline_candidate_records")
    expected_offline = sorted((c for c in candidates if c["offline_eligible"]),
                              key=lambda c: _selection_key(c["selected_search"]))
    _require(isinstance(offline, list)
             and phase.get("offline_eligible") == len(offline)
             and offline == expected_offline,
             "offline roster changed")
    _require(phase.get("shortlisted") == [c["candidate_id"] for c in offline[:6]],
             "shortlist changed")
    short = phase.get("short_records")
    _require(isinstance(short, list)
             and [r["candidate_id"] for r in short] == phase["shortlisted"],
             "short screen evidence missing")
    full = phase.get("full_records")
    finalists = phase.get("full_candidate_records")
    _require(isinstance(full, list) and isinstance(finalists, list)
             and len(full) <= 2
             and [r["candidate_id"] for r in full] == [c["candidate_id"] for c in finalists]
             and all(c == by_id[c["candidate_id"]] for c in finalists),
             "full screen evidence missing")
    qualified = phase.get("qualified")
    expected = {r["candidate_id"] for r in full
                if r["full"]["decision"].get("eligible_for_full") is True}
    _require(isinstance(qualified, list) and len(qualified) == len(set(qualified))
             and set(qualified) == expected
             and phase.get("selected_candidate_id") == (qualified[0] if qualified else None),
             "phase terminal decision changed")


def _completed_result(inputs_manifest: Path, output_directory: Path) -> dict | None:
    """Return a bound historical outcome, or None for an unfinished run.

    A missing/corrupt terminal record never falls through to numerical work.
    File hashing verifies integrity, not fresh scientific qualification.
    """
    import jacek_replay_rebuild as rebuild

    files = _Files()
    output = output_directory.resolve()
    summary_path = output / "final-summary.json"
    if not summary_path.exists() and not summary_path.is_symlink():
        status_path = output / "status.json"
        if status_path.exists():
            status = files.read(status_path)
            _require(not str(status.get("phase", "")).startswith("complete-"),
                     "terminal status exists but final-summary.json is missing")
        return None
    summary = files.read(summary_path)
    terminal = summary.get("terminal")
    _require(summary.get("schema") == PREFIX + "decision.v1" and terminal in TERMINALS,
             "unknown terminal outcome")
    inputs = files.read(inputs_manifest)
    _body(inputs, rebuild.REBUILD_INPUT_SCHEMA)
    _require(inputs.get("rebuild_id") == rebuild.REBUILD_ID, "input campaign changed")
    for key in (
        "corpus", "build_manifest", "matrix", "opening_banks", "blind_holdout",
        "blind_holdout_positions", "blind_holdout_candidate_manifest",
        "blind_holdout_candidate_positions", "comparison", "rank4_teacher", "incumbent_runtime",
    ):
        files.verify(inputs.get(key))
    repository = inputs.get("repository", {})
    commit = repository.get("commit", "")
    _require(repository.get("clean") is True and isinstance(commit, str)
             and len(commit) == 40 and all(c in "0123456789abcdef" for c in commit)
             and isinstance(repository.get("path"), str)
             and Path(repository["path"]).is_absolute(), "frozen repository identity changed")
    policies = inputs.get("policies", {})
    _require(all(policies.get(k) is False for k in (
        "regenerate_teacher_labels", "external_upload", "replace_rank4",
        "canonical_test_model_selection_eligible", "sealed_final_bank_model_selection_eligible",
    )), "input policy changed")
    start, deadline = inputs.get("started_at_unix"), inputs.get("same_architecture_deadline_unix")
    _require(type(start) in (int, float) and type(deadline) in (int, float)
             and math.isfinite(start) and deadline == start + rebuild.SAME_ARCHITECTURE_BUDGET_SECONDS,
             "input budget changed")
    files.walk(inputs)
    matrix = files.bound_json(inputs["matrix"])
    rebuild.validate_matrix(matrix)
    files.walk(matrix)
    phases = summary.get("phases")
    _require(isinstance(phases, list) and bool(phases), "terminal phase evidence missing")
    names = [p.get("phase") for p in phases]
    _require(rebuild.ladder_phase_order_is_valid(names, names[-1]), "phase order changed")
    files.walk(summary)
    for phase in phases:
        _phase(files, phase, inputs, matrix, output, rebuild.ORDER_SEEDS)
    _require(all(p["selected_candidate_id"] is None for p in phases[:-1]),
             "ladder continued after selection")
    if terminal == "no-development-qualified-candidate":
        _require(set(summary) == {"schema", "terminal", "same_architecture_budget_seconds",
                                 "phases", "residual_fallback_exhausted", "next_scope"}
                 and names[-1] == "residual" and phases[-1]["selected_candidate_id"] is None
                 and summary["residual_fallback_exhausted"] is True
                 and summary["same_architecture_budget_seconds"] == rebuild.SAME_ARCHITECTURE_BUDGET_SECONDS
                 and summary["next_scope"] == "action-ranking-or-wider-network-requires-new-plan",
                 "no-candidate outcome is inconsistent")
    else:
        _require(set(summary) == {"schema", "terminal", "phases", "selected_candidate", "qualification"},
                 "qualification outcome shape changed")
        selected = files.read(output / "selected-candidate.json")
        _body(selected, PREFIX + "selected-candidate.v1")
        qualification = files.read(output / "qualification/qualification.json")
        _body(qualification, rebuild.REBUILD_QUALIFICATION_SCHEMA)
        _require(summary["selected_candidate"] == selected
                 and summary["qualification"] == qualification, "terminal receipt changed")
        _require(selected.get("inputs") == files.snapshot(inputs_manifest)
                 and selected.get("ladder_phase_screens") == phases
                 and selected.get("phase_screen") == phases[-1]
                 and selected.get("candidate_id") == phases[-1]["selected_candidate_id"]
                 and selected.get("candidate") in phases[-1]["candidate_records"]
                 and selected["candidate"]["candidate_id"] == selected["candidate_id"]
                 and selected.get("selected_runtime") == selected["candidate"]["selected_search"]["runtime"]
                 and selected.get("matched_runtime") == selected["candidate"]["selected_rank4"]["runtime"],
                 "selection input or phase binding changed")
        _require(all(selected.get(k) is False for k in (
            "protected_test_opened", "sealed_final_bank_opened", "blind_holdout_labels_opened",
        )), "selection reveal state changed")
        _require(qualification.get("inputs") == files.snapshot(inputs_manifest)
                 and qualification.get("selection") == files.snapshot(output / "selected-candidate.json")
                 and qualification.get("candidate") == selected["selected_runtime"]
                 and qualification.get("matched") == selected["matched_runtime"]
                 and qualification.get("rebuild_id") == rebuild.REBUILD_ID
                 and qualification.get("local_only") is True
                 and qualification.get("canonical_rank4_replaced") is False
                 and qualification.get("external_upload") is False,
                 "qualification binding or scope changed")
        holdout = files.bound_json(qualification["blind_holdout"])
        expected_pass = bool(holdout.get("pass") is True
                             and qualification["final_game_gate"]["decision"].get("eligible_for_full") is True)
        _require(qualification.get("pass") is expected_pass
                 and expected_pass == (terminal == "qualified-local-research-incumbent"),
                 "qualification outcome contradicts saved gates")
    return summary


def completed_result(inputs_manifest: Path, output_directory: Path) -> dict | None:
    """Validate saved completion metadata; never fall back after damaged evidence."""
    try:
        return _completed_result(inputs_manifest, output_directory)
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError("completed rebuild: malformed terminal evidence") from error
