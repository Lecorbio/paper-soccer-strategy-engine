#!/usr/bin/env python3
"""Lightweight black-box contracts for the self-search actor producers."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import pathlib
import subprocess
import tempfile


HEADER = (
    "position_id\troot_group_id\tgroup_id\tsource\tsplit\t"
    "winner\tmover\tprefix\n"
)
SHORT_WIN = "0/0/3/0/61/0/07"
RANK4_FIXED_CAP_POSITION = (
    "4/4/7/23/1/2/4/163635/27/457/41/24765/67/2/5050/21/47/23/21/"
    "145075/46167/14/17/2505/47/23224/630750547/0/721/3606/1/3606/71/4/"
    "72714/43/0/5443522301/77/0/632/3/1/3/5741/2255/461274606160505711/"
    "432/3607"
)
ACTOR_MODES = (
    "incumbent-selfplay",
    "incumbent-p1-vs-rank4",
    "incumbent-p2-vs-rank4",
    "incumbent-p1-vs-jacek-nn",
    "incumbent-p2-vs-jacek-nn",
    "incumbent-p1-vs-runner-up",
    "incumbent-p2-vs-runner-up",
    "student-selfplay",
    "student-p1-vs-rank4",
    "student-p2-vs-rank4",
    "student-p1-vs-jacek-nn",
    "student-p2-vs-jacek-nn",
    "student-p1-vs-prior-incumbent",
    "student-p2-vs-prior-incumbent",
)
MASK64 = (1 << 64) - 1


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True,
                   separators=(",", ":"))
        + "\n"
    ).encode("ascii")


def compact_runtime(directory: pathlib.Path, name: str, seed: int) -> pathlib.Path:
    hidden_one, hidden_two = 8, 8
    counts = {
        "w1": 6301 * hidden_one,
        "w2": hidden_one * hidden_two,
        "w3": hidden_two,
    }
    counts["total"] = sum(counts.values())
    payload = bytes((counts["total"] * 3 + 7) // 8)
    body = {
        "schema": "papersoccer.compact-value-bfm-runtime.v1",
        "feature_schema": (
            "papersoccer.jacek-replay-bfm.features.v1:edge316+vertex105x57:"
            "mover-relative-rotate180:true-turn-distance+free-degree"
        ),
        "architecture": {
            "name": "compact-8x8",
            "dimensions": [6301, hidden_one, hidden_two, 1],
            "biases": False,
            "activations": [
                "square-leaky-0.01", "leaky-relu-0.01",
                "fast-tanh-rational-v1",
            ],
            "payload_layout": "w1-input-major,w2-input-major,w3",
        },
        "quantization": {
            "bits": 3,
            "minimum": -3,
            "maximum": 3,
            "scheme": "symmetric-signed-three-bit-per-layer-fixed-scale",
            "packing": "signed-three-bit-twos-complement-lsb-first",
            "scales": {"w1": 0.03125, "w2": 0.0625, "w3": 0.125},
            "weight_counts": counts,
            "packed_byte_count": len(payload),
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "payload_base64": base64.b64encode(payload).decode("ascii"),
        },
        "selection": {
            "arm": "search-target",
            "seed": seed,
            "float_epoch": 1,
            "qat_epoch": 4,
            "source_bundle_body_sha256": "a" * 64,
        },
    }
    document = {
        **body,
        "body_sha256": hashlib.sha256(canonical_json(body)).hexdigest(),
    }
    path = directory / f"{name}.runtime.json"
    path.write_bytes(canonical_json(document))
    return path


def splitmix_next(state: int) -> tuple[int, int]:
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    value = state
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, (value ^ (value >> 31)) & MASK64


def near_goal_seeds(count: int) -> list[int]:
    """Choose attempt-zero seeds with prefix 6 and no random first turn."""
    result = []
    seed = 0
    while len(result) < count:
        state, prefix_draw = splitmix_next(seed)
        _, exploration_draw = splitmix_next(state)
        if prefix_draw % 7 == 6 and exploration_draw % 100 >= 15:
            result.append(seed)
        seed += 1
    return result


def run_continuations(
    executable: pathlib.Path,
    model: pathlib.Path,
    directory: pathlib.Path,
    suffix: str,
    compact_student: pathlib.Path | None = None,
    compact_prior: pathlib.Path | None = None,
) -> tuple[bytes, dict]:
    if (compact_student is None) != (compact_prior is None):
        raise RuntimeError("compact smoke runtimes must be paired")
    modes = ACTOR_MODES if compact_student is None else ACTOR_MODES[7:]
    roots = directory / "roots.tsv"
    root_transcript = SHORT_WIN if compact_student is None else "0"
    roots.write_text(
        "group_id\tsource\twinner\ttranscript\n"
        f"root:fixture\tfixture\t0\t{root_transcript}\n",
        encoding="utf-8",
    )
    plan = directory / "plan.tsv"
    seeds = (
        near_goal_seeds(len(modes))
        if compact_student is None
        else list(range(101, 101 + len(modes)))
    )
    plan.write_text(
        "game_ordinal\tactor_mode\tbase_seed\n"
        + "".join(
            f"{ordinal}\t{mode}\t{seed}\n"
            for ordinal, (mode, seed) in enumerate(zip(modes, seeds, strict=True))
        ),
        encoding="utf-8",
    )
    output = directory / f"games-{suffix}.tsv"
    manifest = directory / f"games-{suffix}.json"
    command = (
        str(executable),
        "--input", str(roots),
        "--output", str(output),
        "--manifest", str(manifest),
        "--model", str(model),
        "--runner-up-model", str(model),
        "--selfsearch-plan", str(plan),
        "--campaign-id", "selfsearch-actor-smoke",
        "--games", str(len(modes)),
        "--candidate-tree-nodes", "16",
        "--actor-nodes", "16",
        "--jacek-nn-nodes", "16",
        "--candidate-exploration", "0.5",
        "--candidate-fpu", "0.5",
        "--max-turns", "64" if compact_student is None else "160",
    )
    if compact_student is not None and compact_prior is not None:
        command += (
            "--compact-student-runtime", str(compact_student),
            "--compact-prior-runtime", str(compact_prior),
        )
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"continuation smoke failed: {completed.stderr}")
    payload = output.read_bytes()
    report = json.loads(manifest.read_bytes())
    if (
        report.get("schema") != "papersoccer.jacek-selfsearch-games.v1"
        or report.get("requested_games") != len(modes)
        or report.get("successful_games") != len(modes)
        or report.get("bindings", {}).get("roots_sha256") != sha256(roots)
        or report.get("bindings", {}).get("plan_sha256") != sha256(plan)
        or report.get("bindings", {}).get("output_sha256")
        != hashlib.sha256(payload).hexdigest()
        or report.get("bindings", {}).get("incumbent_model_sha256") != sha256(model)
        or report.get("bindings", {}).get("runner_up_model_sha256") != sha256(model)
    ):
        raise RuntimeError("continuation smoke manifest bindings are invalid")
    if compact_student is not None and compact_prior is not None:
        configuration = report.get("configuration", {})
        bindings = report.get("bindings", {})
        student = json.loads(compact_student.read_bytes())
        prior = json.loads(compact_prior.read_bytes())
        expected = {
            "compact_student_runtime_sha256": sha256(compact_student),
            "compact_student_runtime_body_sha256": student["body_sha256"],
            "compact_student_payload_sha256":
                student["quantization"]["payload_sha256"],
            "compact_student_source_bundle_body_sha256":
                student["selection"]["source_bundle_body_sha256"],
            "compact_student_selection_sha256": hashlib.sha256(
                canonical_json(student["selection"])
            ).hexdigest(),
            "compact_prior_runtime_sha256": sha256(compact_prior),
            "compact_prior_runtime_body_sha256": prior["body_sha256"],
            "compact_prior_payload_sha256":
                prior["quantization"]["payload_sha256"],
            "compact_prior_source_bundle_body_sha256":
                prior["selection"]["source_bundle_body_sha256"],
            "compact_prior_selection_sha256": hashlib.sha256(
                canonical_json(prior["selection"])
            ).hexdigest(),
        }
        if (
            configuration.get("actor_backend")
            != "compact-value-bfm-runtime-v1"
            or configuration.get("compact_runtime_schema")
            != "papersoccer.compact-value-bfm-runtime.v1"
            or configuration.get("minimum_post_prefix_turns") != 20
            or not isinstance(configuration.get("compact_actor_source_sha256"), str)
            or len(configuration["compact_actor_source_sha256"]) != 64
            or bindings.get("compact_actor_source_sha256")
            != configuration["compact_actor_source_sha256"]
            or any(bindings.get(key) != value for key, value in expected.items())
        ):
            raise RuntimeError("compact continuation bindings are invalid")
    rows = report.get("rows")
    if (
        not isinstance(rows, list)
        or [row.get("row_ordinal") for row in rows] != list(range(len(modes)))
        or [row.get("game_ordinal") for row in rows] != list(range(len(modes)))
        or [row.get("actor_mode") for row in rows] != list(modes)
        or [row.get("base_seed") for row in rows] != seeds
        or (
            compact_student is None
            and (any(row.get("prefix_turns") != 6 for row in rows)
                 or any(row.get("attempt_ordinal") != 0 for row in rows))
        )
        or (
            compact_student is not None
            and any(row.get("prefix_turns") != 0 for row in rows)
        )
    ):
        raise RuntimeError("continuation smoke lineage is invalid")
    for field in (
        "producer_source_sha256",
        "rank4_actor_source_sha256",
        "jacek_nn_actor_source_sha256",
    ):
        value = report.get("configuration", {}).get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise RuntimeError(f"continuation smoke lacks {field}")
    lines = payload.decode("utf-8").splitlines()
    if (lines[0] != "group_id\tsource\twinner\ttranscript"
            or len(lines) != len(modes) + 1):
        raise RuntimeError("continuation smoke TSV is invalid")
    if compact_student is not None:
        for line, row in zip(lines[1:], rows, strict=True):
            transcript = line.split("\t", 3)[3]
            if len(transcript.split("/")) - row["prefix_turns"] < 20:
                raise RuntimeError("compact continuation suffix is shorter than 20 turns")
    return payload, report


def run_rank4_teacher(executable: pathlib.Path) -> None:
    row = "p0\troot:near-goal\tgame:0\tpilot\tvalidation\t0\t0\t0/0/3/0/61/0\n"
    command = (
        str(executable), "--campaign-id", "selfsearch-actor-smoke",
        "--nodes", "64", "--time-ms", "0",
    )
    first = subprocess.run(
        command, input=HEADER + row, text=True, capture_output=True, check=False
    )
    second = subprocess.run(
        command, input=HEADER + row, text=True, capture_output=True, check=False
    )
    if first.returncode != 0 or first.stdout != second.stdout:
        raise RuntimeError(f"Rank-4 teacher is not deterministic: {first.stderr}")
    label = json.loads(first.stdout)
    source_hash = label.get("teacher", {}).get("source_sha256")
    if (
        label.get("schema") != "papersoccer.jacek-replay-teacher.v3"
        or label.get("position_id") != "p0"
        or label.get("mover") != 0
        or not label.get("root_solved")
        or label.get("proven_winner") != 0
        or label.get("search_config", {}).get("max_nodes") != 64
        or label.get("search_config", {}).get("max_time_ms") != 0
        or label.get("search_stats", {}).get("deadline_reached") is not False
        or not isinstance(source_hash, str)
        or len(source_hash) != 64
    ):
        raise RuntimeError("Rank-4 teacher label contract is invalid")

    timed = subprocess.run(
        (*command[:-1], "1"), input=HEADER + row, text=True,
        capture_output=True, check=False,
    )
    if timed.returncode == 0 or timed.stdout:
        raise RuntimeError("Rank-4 fixed-work labels accepted a wall-clock limit")

    duplicate = subprocess.run(
        command, input=HEADER + row + row, text=True, capture_output=True, check=False
    )
    if duplicate.returncode == 0 or duplicate.stdout:
        raise RuntimeError("Rank-4 teacher did not fail closed on duplicate IDs")

    capped_row = (
        "position:c1eb882b3646bfb79053eb6379dacd5100567eed0459b06dc181d417c0d92748"
        "\town-live:898437522"
        "\tselfsearch-game:cef40925f005f13ffa5e0c40e69f910ee025a6a6f7a1ecbcf5d44c096a7e0015"
        "\tselfsearch-pilot-20260825-v3\ttrain\t0\t1\t"
        f"{RANK4_FIXED_CAP_POSITION}\n"
    )
    capped_command = (
        str(executable), "--campaign-id", "selfsearch-pilot-20260825-v3",
        "--nodes", "32000", "--time-ms", "0",
    )
    capped_first = subprocess.run(
        capped_command, input=HEADER + capped_row, text=True,
        capture_output=True, check=False,
    )
    capped_second = subprocess.run(
        capped_command, input=HEADER + capped_row, text=True,
        capture_output=True, check=False,
    )
    if capped_first.returncode != 0 or capped_first.stdout != capped_second.stdout:
        raise RuntimeError(
            "Rank-4 fixed-cap regression is not deterministic: "
            f"{capped_first.stderr}"
        )
    capped = json.loads(capped_first.stdout)
    capped_stats = capped.get("search_stats", {})
    if (
        capped.get("schema") != "papersoccer.jacek-replay-teacher.v3"
        or capped.get("completed_depth") != 0
        or capped.get("nodes") != 32_000
        or capped.get("root_score") != 26_407
        or capped.get("root_solved") is not False
        or capped.get("proven_winner") is not None
        or capped.get("search_config", {}).get("max_time_ms") != 0
        or capped_stats.get("attempted_depth") != 1
        or capped_stats.get("completed_actions") != 9_996
        or capped_stats.get("budget_exhausted") is not True
        or capped_stats.get("node_cap_reached") is not True
        or capped_stats.get("depth_cap_reached") is not False
        or capped_stats.get("deadline_reached") is not False
        or capped_stats.get("termination_reason") != "fixed-work-cap"
    ):
        raise RuntimeError("Rank-4 fixed-cap regression label is invalid")

    no_action = subprocess.run(
        (*capped_command[:-4], "--nodes", "1", "--time-ms", "0"),
        input=HEADER + capped_row, text=True, capture_output=True, check=False,
    )
    if no_action.returncode == 0 or no_action.stdout:
        raise RuntimeError("Rank-4 teacher accepted a cap without a searched action")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--continuations", type=pathlib.Path, required=True)
    parser.add_argument("--rank4-teacher", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    arguments = parser.parse_args()
    with tempfile.TemporaryDirectory() as raw_directory:
        directory = pathlib.Path(raw_directory)
        first_payload, first_report = run_continuations(
            arguments.continuations, arguments.model, directory, "one"
        )
        second_payload, second_report = run_continuations(
            arguments.continuations, arguments.model, directory, "two"
        )
        if first_payload != second_payload or first_report != second_report:
            raise RuntimeError("self-search continuation output is not deterministic")
        student = compact_runtime(directory, "student", 20260907)
        prior = compact_runtime(directory, "prior", 20260908)
        compact_first_payload, compact_first_report = run_continuations(
            arguments.continuations, arguments.model, directory, "compact-one",
            student, prior,
        )
        compact_second_payload, compact_second_report = run_continuations(
            arguments.continuations, arguments.model, directory, "compact-two",
            student, prior,
        )
        if (compact_first_payload != compact_second_payload
                or compact_first_report != compact_second_report):
            raise RuntimeError("compact self-search output is not deterministic")
        damaged = json.loads(student.read_bytes())
        damaged["body_sha256"] = "0" * 64
        damaged_path = directory / "damaged.runtime.json"
        damaged_path.write_bytes(canonical_json(damaged))
        try:
            run_continuations(
                arguments.continuations, arguments.model, directory,
                "compact-damaged", damaged_path, prior,
            )
        except RuntimeError as error:
            if "continuation smoke failed" not in str(error):
                raise
        else:
            raise RuntimeError("compact self-search accepted a damaged body hash")
        noncanonical = json.loads(student.read_bytes())
        noncanonical_body = dict(noncanonical)
        noncanonical_body.pop("body_sha256")
        noncanonical_body_bytes = canonical_json(noncanonical_body).replace(
            b'"w1":0.03125', b'"w1":3.125e-2'
        )
        noncanonical["body_sha256"] = hashlib.sha256(
            noncanonical_body_bytes
        ).hexdigest()
        noncanonical_bytes = canonical_json(noncanonical).replace(
            b'"w1":0.03125', b'"w1":3.125e-2'
        )
        noncanonical_path = directory / "noncanonical.runtime.json"
        noncanonical_path.write_bytes(noncanonical_bytes)
        try:
            run_continuations(
                arguments.continuations, arguments.model, directory,
                "compact-noncanonical", noncanonical_path, prior,
            )
        except RuntimeError as error:
            if "continuation smoke failed" not in str(error):
                raise
        else:
            raise RuntimeError("compact self-search accepted noncanonical numbers")
        run_rank4_teacher(arguments.rank4_teacher)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
