#!/usr/bin/env python3
"""Generate and evaluate the clean successor's one-shot protected holdout.

The plan is sealed before model selection, but this tool creates no protected
bytes until an immutable, offline-qualified successor selection exists.  It
never accepts or resolves a retired bundle/test route.
"""

from __future__ import annotations

import argparse
import atexit
import concurrent.futures
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _load(path: pathlib.Path, name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load fresh-holdout dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


successor = _load(
    HERE / "compact_value_bfm_successor.py", "compact_fresh_successor"
)
qualification = successor.qualification
compact = successor.compact
selfsearch = successor.selfsearch
large_training = successor.large_training
corpus = selfsearch.corpus
pack_tool = _load(HERE / "jacek_replay_pack.py", "compact_fresh_pack")
opening_tools = _load(
    HERE / "compact_value_bfm_openings.py", "compact_fresh_openings"
)


HoldoutError = successor.SuccessorError
NAMESPACE = successor.NAMESPACE
CAMPAIGN_ID = f"{successor.SUCCESSOR_CAMPAIGN_ID}-holdout"
CLAIM_SCHEMA = "papersoccer.compact-value-bfm.fresh-holdout-claim.v1"
MATERIALIZATION_SCHEMA = (
    "papersoccer.compact-value-bfm.fresh-holdout-materialization.v1"
)
REPORT_SCHEMA = "papersoccer.compact-value-bfm.fresh-holdout-report.v1"
REPORT_REFERENCE_SCHEMA = (
    "papersoccer.compact-value-bfm.fresh-holdout-report-reference.v1"
)
_HOLDOUT_LOCK_FD: int | None = None


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _acquire_holdout_lock(holdout_root: pathlib.Path) -> pathlib.Path:
    global _HOLDOUT_LOCK_FD
    lock_path = holdout_root / "materialization.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if _HOLDOUT_LOCK_FD is not None:
        return lock_path
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        os.close(descriptor)
        raise HoldoutError("another fresh holdout materialization is active") from error
    _HOLDOUT_LOCK_FD = descriptor
    selfsearch._CAMPAIGN_LOCK_FD = descriptor
    atexit.register(_release_holdout_lock)
    return lock_path


def _release_holdout_lock() -> None:
    global _HOLDOUT_LOCK_FD
    descriptor = _HOLDOUT_LOCK_FD
    if descriptor is None:
        return
    _HOLDOUT_LOCK_FD = None
    selfsearch._CAMPAIGN_LOCK_FD = None
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: pathlib.Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise HoldoutError(f"fresh-holdout artifact is not a regular file: {path}")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _verify(record: object, label: str) -> pathlib.Path:
    return successor._verify_record(record, label)


def _canonical(path: pathlib.Path, label: str) -> dict[str, Any]:
    return successor._canonical_json(path, label)


def _atomic_bytes(path: pathlib.Path, payload: bytes) -> None:
    qualification.atomic_write_once(path, payload)


def _selection(
    plan: Mapping[str, Any], output_root: pathlib.Path,
) -> tuple[pathlib.Path, dict[str, Any], pathlib.Path]:
    reference_path = output_root / "selection-reference.json"
    selection_path = successor._selection_reference(
        reference_path,
        plan=plan,
        output_root=output_root,
    )
    if selection_path is None:
        raise HoldoutError("successor immutable selection is absent")
    selection = successor._validate_selection_closure(
        selection_path, plan=plan, output_root=output_root
    )
    runtime_record = selection.get("runtime")
    if not isinstance(runtime_record, Mapping) or not isinstance(
        runtime_record.get("path"), str
    ):
        raise HoldoutError("selected successor runtime binding is missing")
    runtime = pathlib.Path(runtime_record["path"])
    if runtime.parent != output_root.resolve() / "training" / "quantized-runtimes":
        raise HoldoutError("selected successor runtime escaped its fixed directory")
    _verify(runtime_record, "selected successor runtime")
    if (
        selection.get("plan_body_sha256") != plan["body_sha256"]
        or selection.get("campaign_id") != successor.SUCCESSOR_CAMPAIGN_ID
        or selection.get("offline_gate", {}).get("passed") is not True
        or selection.get("selection_immutable") is not True
        or selection.get("selection_may_change_after_fresh_protected_tests")
        is not False
        or selection.get("old_protected_tests_accessed") is not False
        or selection.get("old_protected_tests_permanently_excluded") is not True
        or selection.get("fresh_protected_tests_opened") is not False
        or selection.get("fresh_protected_tests_authorized") is not True
    ):
        raise HoldoutError("successor selection is not eligible for fresh tests")
    return selection_path, selection, runtime


def _source_opening_exclusions(source_tsv: pathlib.Path) -> set[str]:
    excluded_fingerprints: set[str] = set()
    lines = source_tsv.read_text(encoding="utf-8").splitlines()
    if lines[:1] != ["group_id\tsource\twinner\ttranscript"]:
        raise HoldoutError("safe root TSV header changed")
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) != 4 or not fields[3]:
            raise HoldoutError("safe root TSV row changed")
        actions = fields[3].split("/")
        # Source roots are complete historical games and therefore terminal.
        # Exclude only their nonterminal opening-prefix states; attempting to
        # fingerprint the final state would violate the opening-bank contract.
        state = opening_tools.reference.ReplayState()
        physical_plies = 0
        for action in actions:
            mover = state.to_move
            opening_tools.reference.apply_complete_turn(state, mover, action)
            physical_plies += len(action)
            if state.winner is not None:
                break
            if physical_plies < opening_tools.MINIMUM_PHYSICAL_PLIES:
                continue
            excluded_fingerprints.update(
                value
                for key, value in opening_tools.state_fingerprints(state).items()
                if key != "canonical"
            )
    return excluded_fingerprints


def _fresh_test_roots(
    plan: Mapping[str, Any], holdout_root: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    source_path = _verify(plan["training"]["roots_manifest"], "safe roots manifest")
    opening_generator = _verify(
        plan["tools"]["opening_generator"], "fresh opening generator"
    )
    if opening_generator != pathlib.Path(opening_tools.__file__).resolve():
        raise HoldoutError("fresh opening generator path changed")
    source = _canonical(source_path, "safe roots manifest")
    source_tsv = _verify(plan["training"]["roots_tsv"], "safe root TSV")
    excluded_fingerprints = _source_opening_exclusions(source_tsv)
    seed_material = (
        f"{CAMPAIGN_ID}\0{plan['fresh_protected_holdout']['game_plan_seed']}"
        "\0fresh-protected-root-openings-v1"
    ).encode("ascii")
    seed = hashlib.sha256(seed_material).digest()
    openings = opening_tools.generate_openings(
        stage="fresh_holdout_root",
        count=int(plan["fresh_protected_holdout"]["fresh_root_openings"]),
        seed=seed,
        excluded_fingerprints=excluded_fingerprints,
    )
    rows = []
    tsv = ["group_id\tsource\twinner\ttranscript"]
    for index, opening in enumerate(openings):
        group_id = f"fresh-protected-root:{index:04d}"
        transcript = str(opening["transcript"])
        turns = [
            {"player_id": turn % 2, "action": action}
            for turn, action in enumerate(transcript.split("/"))
        ]
        rows.append({
            "group_id": group_id,
            "root_group_id": group_id,
            "source": "fresh-protected-opening",
            "winner": 0,
            "turns": turns,
            "split": "test",
            "opening_id": opening["opening_id"],
            "opening_fingerprints": opening["fingerprints"],
            "source_record_sha256": hashlib.sha256(
                qualification.canonical_json_bytes(opening)
            ).hexdigest(),
        })
        # The continuation generator deliberately ignores the historical winner
        # field and consumes only group_id + transcript.
        tsv.append(f"{group_id}\tfresh-protected-opening\t0\t{transcript}")
    body = {
        "schema": source["schema"],
        "feature_schema": source["feature_schema"],
        "tool_sha256": source["tool_sha256"],
        "exclusion_boundary": source["exclusion_boundary"],
        "accepted": rows,
        "counts": {
            "accepted": len(rows),
            "excluded_records": 0,
            "source_preexcluded_aggregate": 0,
            "split_games": {"train": 0, "validation": 0, "test": len(rows)},
            "structurally_rejected": 0,
        },
        "excluded": [],
        "structurally_rejected": [],
        "sources": [{
            "kind": "fresh-protected-opening-generator",
            "seed_sha256": hashlib.sha256(seed).hexdigest(),
            "count": len(rows),
        }],
        "split_parent": {
            "schema": "papersoccer.compact-value-bfm.fresh-holdout-parent.v1",
            "source_roots_sha256": _sha256_file(source_path),
            "successor_plan_body_sha256": plan["body_sha256"],
        },
        "split_policy": (
            "one-shot-post-selection-protected-holdout-all-accepted-roots-test"
        ),
        "successor_provenance": {
            "campaign_id": CAMPAIGN_ID,
            "old_protected_tests_permitted": False,
            "selection_may_change": False,
            "opening_generator": dict(plan["tools"]["opening_generator"]),
        },
    }
    path = holdout_root / "roots" / "fresh-test-roots.json"
    tsv_path = holdout_root / "roots" / "fresh-test-roots.tsv"
    bank_path = holdout_root / "roots" / "fresh-opening-bank.json"
    qualification.write_sealed(path, body)
    _atomic_bytes(tsv_path, ("\n".join(tsv) + "\n").encode("utf-8"))
    qualification.write_sealed(bank_path, {
        "schema": "papersoccer.compact-value-bfm.fresh-holdout-opening-bank.v1",
        "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID,
        "seed_sha256": hashlib.sha256(seed).hexdigest(),
        "opening_generator": dict(plan["tools"]["opening_generator"]),
        "excluded_source_root_fingerprints": len(excluded_fingerprints),
        "opening_count": len(openings),
        "openings": openings,
        "all_group_ids_new": True,
        "old_protected_tests_accessed": False,
    })
    # The maintained packer validates the generic replay-root body contract.
    pack_tool.load_roots(path)
    return path, tsv_path, bank_path


def _game_plan(
    plan: Mapping[str, Any], holdout_root: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path, dict[str, Any]]:
    configuration = plan["fresh_protected_holdout"]
    game_plan = selfsearch.make_game_plan(
        campaign_id=configuration["campaign_id"],
        seed=int(configuration["game_plan_seed"]),
        quotas=configuration["quotas"],
    )
    if game_plan.get("games") != 3_200:
        raise HoldoutError("fresh holdout game plan does not contain 3,200 games")
    json_path = holdout_root / "game-plan.json"
    tsv_path = holdout_root / "game-plan.tsv"
    _atomic_bytes(json_path, selfsearch.canonical_json_bytes(game_plan))
    _atomic_bytes(tsv_path, selfsearch.render_game_plan_tsv(game_plan))
    return json_path, tsv_path, game_plan


def _phase_spec(configuration: Mapping[str, Any]) -> Any:
    phase_configuration = {
        **selfsearch.FULL_CONFIGURATION,
        "campaign_id": configuration["campaign_id"],
        "games": configuration["games"],
        "game_chunk_size": configuration["game_chunk_size"],
        "game_workers": configuration["game_workers"],
        "positions_per_game": configuration["positions_per_game"],
        "bfm_actor_tree_nodes": configuration["compact_tree_nodes"],
        "rank4_actor_nodes": configuration["rank4_actor_nodes"],
        "jacek_nn_actor_nodes": configuration["jacek_nn_actor_nodes"],
        "exploration": configuration["exploration"],
        "fpu": configuration["fpu"],
        "bfm_shallow_tree_nodes": configuration["search_shallow_nodes"],
        "bfm_deep_tree_nodes": configuration["search_deep_nodes"],
        "rank4_shallow_nodes": configuration["rank4_shallow_nodes"],
        "rank4_deep_nodes": configuration["rank4_deep_nodes"],
        "hard_fraction_numerator": configuration["hard_fraction"][0],
        "hard_fraction_denominator": configuration["hard_fraction"][1],
    }
    return selfsearch.PhaseSpec(
        name="fresh-holdout",
        campaign_id=configuration["campaign_id"],
        configuration=phase_configuration,
        quotas=configuration["quotas"],
        game_seed=configuration["game_plan_seed"],
        opening_seed=0,
        pairs=0,
        gate_time_ms=0,
        gate_workers=configuration["game_workers"],
        bank_classification="fresh-protected-holdout",
    )


def _manager(
    output: pathlib.Path, *, resume: bool, plan: Mapping[str, Any],
) -> Any:
    return selfsearch.StageManager(
        output=output,
        campaign_id=CAMPAIGN_ID,
        round_index=0,
        resume=resume,
        environment={
            "namespace": NAMESPACE,
            "successor_plan_body_sha256": plan["body_sha256"],
            "protected_holdout": True,
            "single_thread": True,
            "old_protected_tests_accessed": False,
        },
    )


def _write_pair(
    payload: bytes, manifest: Mapping[str, Any],
    data_path: pathlib.Path, manifest_path: pathlib.Path,
) -> None:
    selfsearch.write_pair(payload, dict(manifest), data_path, manifest_path)


def _label(
    *, manager: Any, ordinal: int, name: str, positions: pathlib.Path,
    output: pathlib.Path, teacher: pathlib.Path, schema: str,
    campaign_id: str, nodes: int, source_sha256: str,
    workers: int, chunk_games: int, model: pathlib.Path | None = None,
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    return manager.execute(
        ordinal=ordinal,
        name=name,
        configuration={
            "nodes": nodes,
            "workers": workers,
            "chunk_games": chunk_games,
            "schema": schema,
            "source_sha256": source_sha256,
        },
        producers={"teacher": teacher, "workflow": pathlib.Path(selfsearch.__file__)},
        inputs={"positions": positions, **({"model": model} if model else {})},
        outputs={"labels": output},
        resumable_outputs={"labels"},
        action=lambda: selfsearch.run_label_chunks(
            manager=manager,
            stage_ordinal=ordinal,
            stage_name=name,
            positions=positions,
            output=output,
            teacher=teacher,
            schema=schema,
            campaign_id=campaign_id,
            nodes=nodes,
            workers=workers,
            source_sha256=source_sha256,
            model=model,
            chunk_games=chunk_games,
        ),
    )


def _merge_labels(
    shallow: pathlib.Path, deep: pathlib.Path, output: pathlib.Path, schema: str,
) -> pathlib.Path:
    payload = selfsearch.merge_deep_labels(
        shallow=shallow, deep=deep, expected_schema=schema
    )
    _atomic_bytes(output, payload)
    return output


def _canonical_label_chunks(
    *, games: pathlib.Path, teacher: pathlib.Path, output: pathlib.Path,
    receipt_root: pathlib.Path, nodes: int, deep_nodes: int,
    deep_percent: int, max_samples: int, workers: int, chunk_games: int,
) -> dict[str, Any]:
    lines = games.read_bytes().splitlines(keepends=True)
    if not lines or lines[0] != b"group_id\tsource\twinner\ttranscript\n":
        raise HoldoutError("fresh game TSV is malformed for canonical teacher")
    rows = lines[1:]
    if len(rows) != 3_200:
        raise HoldoutError("canonical teacher requires exactly 3,200 fresh games")
    chunk_root = output.parent / "canonical-chunks"
    chunk_root.mkdir(parents=True, exist_ok=True)
    receipt_root.mkdir(parents=True, exist_ok=True)
    command_prefix = [
        str(teacher),
        "--nodes", str(nodes),
        "--deep-nodes", str(deep_nodes),
        "--deep-percent", str(deep_percent),
        "--max-samples", str(max_samples),
    ]
    specs = []
    for ordinal, begin in enumerate(range(0, len(rows), chunk_games)):
        input_path = chunk_root / f"chunk-{ordinal:06d}.tsv"
        output_path = chunk_root / f"chunk-{ordinal:06d}.jsonl"
        receipt_path = receipt_root / f"chunk-{ordinal:06d}.json"
        payload = lines[0] + b"".join(rows[begin : begin + chunk_games])
        _atomic_bytes(input_path, payload)
        specs.append((ordinal, input_path, output_path, receipt_path))

    def output_row_count(output_path: pathlib.Path) -> int:
        count = 0
        with output_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("schema") != corpus.TEACHER_SCHEMA:
                    raise HoldoutError("fresh canonical teacher schema changed")
                corpus.sample_from_teacher_row(row)
                count += 1
        if count <= 0:
            raise HoldoutError("fresh canonical chunk is empty")
        return count

    def validate(spec: tuple[int, pathlib.Path, pathlib.Path, pathlib.Path]) -> dict[str, Any]:
        ordinal, input_path, output_path, receipt_path = spec
        receipt = qualification.load_sealed(receipt_path)
        if (
            receipt.get("schema")
            != "papersoccer.compact-value-bfm.fresh-canonical-chunk.v1"
            or receipt.get("chunk_ordinal") != ordinal
            or receipt.get("input") != _record(input_path)
            or receipt.get("output") != _record(output_path)
            or receipt.get("teacher") != _record(teacher)
            or receipt.get("configuration") != {
                "nodes": nodes,
                "deep_nodes": deep_nodes,
                "deep_percent": deep_percent,
                "max_samples": max_samples,
            }
        ):
            raise HoldoutError("fresh canonical chunk receipt changed")
        count = output_row_count(output_path)
        if count != receipt.get("rows") or count <= 0:
            raise HoldoutError("fresh canonical chunk row count changed")
        return receipt

    def execute(spec: tuple[int, pathlib.Path, pathlib.Path, pathlib.Path]) -> None:
        ordinal, input_path, output_path, receipt_path = spec
        if receipt_path.exists():
            validate(spec)
            return
        if output_path.exists():
            rows_count = output_row_count(output_path)
            qualification.write_sealed(receipt_path, {
                "schema": "papersoccer.compact-value-bfm.fresh-canonical-chunk.v1",
                "namespace": NAMESPACE,
                "campaign_id": CAMPAIGN_ID,
                "chunk_ordinal": ordinal,
                "input": _record(input_path),
                "output": _record(output_path),
                "teacher": _record(teacher),
                "configuration": {
                    "nodes": nodes,
                    "deep_nodes": deep_nodes,
                    "deep_percent": deep_percent,
                    "max_samples": max_samples,
                },
                "rows": rows_count,
                "old_protected_tests_accessed": False,
                "recovered_after_output_before_receipt": True,
            })
            validate(spec)
            return
        temporary: pathlib.Path | None = None
        try:
            with input_path.open("rb") as source, tempfile.NamedTemporaryFile(
                dir=chunk_root,
                prefix=f".{output_path.name}.",
                delete=False,
            ) as destination:
                temporary = pathlib.Path(destination.name)
                completed = subprocess.run(
                    command_prefix,
                    stdin=source,
                    stdout=destination,
                    stderr=subprocess.PIPE,
                    check=False,
                    timeout=7_200,
                    pass_fds=(
                        () if _HOLDOUT_LOCK_FD is None else (_HOLDOUT_LOCK_FD,)
                    ),
                )
                destination.flush()
                os.fsync(destination.fileno())
            if completed.returncode != 0:
                raise HoldoutError(
                    "fresh canonical teacher failed: "
                    + completed.stderr.decode("utf-8", "replace")
                )
            os.replace(temporary, output_path)
            rows_count = sum(1 for _ in output_path.open("rb"))
            qualification.write_sealed(receipt_path, {
                "schema": "papersoccer.compact-value-bfm.fresh-canonical-chunk.v1",
                "namespace": NAMESPACE,
                "campaign_id": CAMPAIGN_ID,
                "chunk_ordinal": ordinal,
                "input": _record(input_path),
                "output": _record(output_path),
                "teacher": _record(teacher),
                "configuration": {
                    "nodes": nodes,
                    "deep_nodes": deep_nodes,
                    "deep_percent": deep_percent,
                    "max_samples": max_samples,
                },
                "rows": rows_count,
                "old_protected_tests_accessed": False,
            })
            validate(spec)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    missing = [spec for spec in specs if not spec[3].exists()]
    if missing:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(execute, spec) for spec in missing]
            for future in concurrent.futures.as_completed(futures):
                future.result()
    receipts = [validate(spec) for spec in specs]
    if output.exists():
        expected = hashlib.sha256()
        for spec in specs:
            with spec[2].open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    expected.update(chunk)
        if _sha256_file(output) != expected.hexdigest():
            raise HoldoutError("fresh canonical merged output changed")
    else:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=output.parent, prefix=".canonical-merged.", suffix=".jsonl"
        )
        temporary = pathlib.Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as destination:
                for spec in specs:
                    with spec[2].open("rb") as source:
                        shutil.copyfileobj(source, destination, 8 * 1024 * 1024)
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
    return {
        "chunks": len(specs),
        "rows": sum(int(receipt["rows"]) for receipt in receipts),
        "labels": _record(output),
        "receipts": [_record(spec[3]) for spec in specs],
    }


def _packing_priors(
    plan: Mapping[str, Any], output_root: pathlib.Path,
) -> list[pathlib.Path]:
    training_path = output_root / "training-input.json"
    if training_path.is_symlink() or not training_path.is_file():
        raise HoldoutError("successor training input is not a regular file")
    training = qualification.load_sealed(training_path, successor.TRAINING_INPUT_SCHEMA)
    artifacts = plan["training"]["safe_input_artifacts"]
    declarations = [training["new_train_manifest"]]
    declarations.extend(record["manifest"] for record in artifacts["anchor"])
    declarations.extend(
        record["manifest"] for record in artifacts["canonical_validation"]
    )
    declarations.extend(
        record["manifest"] for record in artifacts["common_adjudicator"]
    )
    if len(declarations) != 8:
        raise HoldoutError("fresh holdout requires exactly eight clean priors")
    bundle_path = pathlib.Path(
        plan["training"]["source_bundle_manifest"]["path"]
    )
    retired = successor.retired_protected_paths(bundle_path)
    priors = []
    for index, declaration in enumerate(declarations):
        path = successor._declared_record(declaration, "fresh holdout prior")
        if path in retired:
            raise HoldoutError("retired protected test prior is forbidden before access")
        if index == 0:
            expected_parent = (output_root / "global-repack" / "search").resolve()
            if path.parent != expected_parent:
                raise HoldoutError("successor global train prior path changed")
        priors.append(_verify(declaration, "fresh holdout prior"))
    for path in priors:
        if _canonical(path, "fresh holdout prior").get("split") == "test":
            raise HoldoutError("fresh holdout cannot use a test shard as a prior")
    return priors


def _pack_labels(
    *, plan: Mapping[str, Any], roots: pathlib.Path, labels: pathlib.Path,
    output: pathlib.Path, priors: Sequence[pathlib.Path],
) -> dict[str, Any]:
    report = selfsearch.run_pack(
        python=_verify(plan["tools"]["python"], "successor Python"),
        pack_tool=_verify(plan["tools"]["pack_tool"], "successor pack tool"),
        roots=roots,
        labels=labels,
        output_directory=output,
        prior_manifests=priors,
    )
    shards = report.get("shards")
    if (
        not isinstance(shards, Mapping)
        or set(shards) != {"train", "validation", "test"}
        or shards["train"].get("samples") != 0
        or shards["validation"].get("samples") != 0
        or not isinstance(shards["test"].get("samples"), int)
        or shards["test"]["samples"] <= 0
    ):
        raise HoldoutError("fresh holdout pack did not remain all-test")
    return report


def _group_isolation(
    priors: Sequence[pathlib.Path], reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    prior_groups: set[bytes] = set()
    for path in priors:
        shard = large_training.load_csr_shard(path)
        prior_groups.update(bytes(group) for group in shard.group_ids)
    test_counts = {}
    for name, report in reports.items():
        test_manifest = pathlib.Path(report["shards"]["test"]["manifest"])
        shard = large_training.load_csr_shard(test_manifest)
        test_groups = {bytes(group) for group in shard.group_ids}
        if prior_groups.intersection(test_groups):
            raise HoldoutError(
                f"fresh {name} holdout has train/validation root-group overlap"
            )
        test_counts[name] = {
            "rows": len(shard),
            "unique_group_ids": len(test_groups),
        }
    return {
        "policy": "raw-sha256-group-id-disjoint-from-all-training-and-validation",
        "prior_unique_group_ids": len(prior_groups),
        "tests": test_counts,
        "passed": True,
    }


def _as_compact_dataset(path: pathlib.Path) -> Any:
    shard = large_training.load_csr_shard(path)
    return compact.Dataset(
        indptr=shard.indptr,
        indices=shard.indices,
        targets=shard.targets,
        weights=shard.weights,
        group_ids=shard.group_ids,
        split=shard.split,
        source_manifest_sha256=_sha256_file(path),
        source_npz_sha256=shard.npz_sha256,
        source_route=str(path.resolve()),
    )


def _fresh_split_isolation(
    priors: Sequence[pathlib.Path], reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    prior_datasets = [_as_compact_dataset(path) for path in priors]
    test_datasets = {
        name: _as_compact_dataset(
            pathlib.Path(report["shards"]["test"]["manifest"])
        )
        for name, report in reports.items()
    }
    prior_train = [
        compact.dataclasses.replace(dataset, split="train")
        for dataset in prior_datasets
    ]
    protected_validation = {
        name: compact.dataclasses.replace(dataset, split="validation")
        for name, dataset in test_datasets.items()
    }
    result = compact.validate_unprotected_split_isolation(
        prior_train[0],
        compact.concatenate_datasets(prior_train[1:], split="train"),
        protected_validation["search"],
        compact.concatenate_datasets(
            [
                protected_validation["rank4"],
                protected_validation["canonical"],
            ],
            split="validation",
        ),
    )
    result = dict(result)
    result.pop("protected_tests_opened", None)
    result.update({
        "policy_scope": "all-clean-training-validation-vs-all-fresh-protected-tests",
        "fresh_protected_tests_opened": True,
        "passed": True,
    })
    return result


def materialize(
    *, plan_path: pathlib.Path, output_root: pathlib.Path, resume: bool,
) -> pathlib.Path:
    if not resume:
        raise HoldoutError("fresh holdout materialization always requires --resume")
    output_root = output_root.resolve()
    plan = successor.load_plan(plan_path, output_root=output_root)
    selection_path, selection, selected_runtime = _selection(plan, output_root)
    configuration = plan["fresh_protected_holdout"]
    if configuration.get("materialized") is not False:
        raise HoldoutError("fresh holdout plan was already materialized")
    holdout_root = output_root / "fresh-holdout"
    lock_path = _acquire_holdout_lock(holdout_root)
    claim_path = holdout_root / "00-materialization-claim.json"
    prior_runtime = _verify(
        plan["training"]["prior_compact_runtime"], "prior compact runtime"
    )
    expected_claim = {
        "schema": CLAIM_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID,
        "status": "fresh-protected-holdout-materialization-claimed-once",
        "successor_plan": qualification.artifact_reference(
            plan_path, successor.PLAN_SCHEMA
        ),
        "immutable_selection": qualification.artifact_reference(
            selection_path, successor.SELECTION_SCHEMA
        ),
        "selected_runtime": _record(selected_runtime),
        "prior_runtime": _record(prior_runtime),
        "configuration": dict(configuration),
        "selection_may_change": False,
        "old_protected_tests_permitted": False,
        "materialization_attempts_authorized": 1,
        "exclusive_process_lock": str(lock_path.resolve()),
    }
    if claim_path.exists():
        claim = qualification.load_sealed(claim_path, CLAIM_SCHEMA)
        if any(
            claim.get(field) != value for field, value in expected_claim.items()
        ):
            raise HoldoutError("fresh holdout materialization claim changed")
    else:
        claim = qualification.write_sealed(claim_path, {
            **expected_claim,
            "claimed_at_utc": utc_now(),
        })
    roots, fresh_roots_tsv, fresh_opening_bank = _fresh_test_roots(
        plan, holdout_root
    )
    game_plan_json, game_plan_tsv, game_plan = _game_plan(plan, holdout_root)
    materialized = holdout_root / "materialized"
    manager = _manager(materialized, resume=resume, plan=plan)
    source_plan = qualification.load_sealed(
        pathlib.Path(plan["source"]["plan"]["path"]), successor.SOURCE_PLAN_SCHEMA
    )
    if source_plan.get("source_identities") != configuration["source_identities"]:
        raise HoldoutError("fresh holdout producer source identities changed")
    generator = _verify(plan["tools"]["continuation_generator"], "continuation generator")
    spec = _phase_spec(configuration)
    games = materialized / "games.tsv"
    games_manifest = materialized / "games.manifest.json"
    game_result = manager.execute(
        ordinal=1,
        name="games",
        configuration={
            "games": 3_200,
            "quotas": dict(configuration["quotas"]),
            "workers": configuration["game_workers"],
            "chunk_games": configuration["game_chunk_size"],
        },
        producers={
            "generator": generator,
            "workflow": pathlib.Path(selfsearch.__file__),
        },
        inputs={
            "game_plan": game_plan_json,
            "fresh_roots": roots,
            "fresh_roots_tsv": fresh_roots_tsv,
            "selected_runtime": selected_runtime,
            "prior_runtime": prior_runtime,
        },
        outputs={"games": games, "manifest": games_manifest},
        resumable_outputs={"games", "manifest"},
        action=lambda: selfsearch.run_game_chunks(
            manager=manager,
            stage_ordinal=1,
            spec=spec,
            plan_path=game_plan_json,
            roots_tsv=fresh_roots_tsv,
            actor=selected_runtime,
            diversity=prior_runtime,
            generator=generator,
            workers=configuration["game_workers"],
            source_identities=configuration["source_identities"],
            compact_student_runtime=selected_runtime,
            compact_prior_runtime=prior_runtime,
        ),
    )
    if game_result.get("games") != 3_200:
        raise HoldoutError("fresh holdout game generation is incomplete")
    positions = materialized / "positions.tsv"
    positions_manifest = materialized / "positions.manifest.json"
    def freeze_stage() -> dict[str, Any]:
        positions_payload, positions_document = selfsearch.freeze_positions(
            campaign_id=CAMPAIGN_ID,
            games_tsv=games,
            games_manifest=games_manifest,
            roots_manifest=roots,
            maximum_per_game=configuration["positions_per_game"],
        )
        if (
            positions_document.get("positions") != 64_000
            or positions_document.get("split_counts") != {"test": 64_000}
        ):
            raise HoldoutError(
                "fresh holdout did not freeze exactly 64,000 test positions"
            )
        _write_pair(positions_payload, positions_document, positions, positions_manifest)
        return {
            "positions": positions_document["positions"],
            "split_counts": positions_document["split_counts"],
        }

    position_result = manager.execute(
        ordinal=2,
        name="positions",
        configuration={"maximum_per_game": configuration["positions_per_game"]},
        producers={"workflow": pathlib.Path(selfsearch.__file__)},
        inputs={"games": games, "games_manifest": games_manifest, "roots": roots},
        outputs={"positions": positions, "manifest": positions_manifest},
        resumable_outputs={"positions", "manifest"},
        action=freeze_stage,
    )
    if position_result != {"positions": 64_000, "split_counts": {"test": 64_000}}:
        raise HoldoutError("fresh position stage receipt changed")
    labels = materialized / "labels"
    labels.mkdir(parents=True, exist_ok=True)
    search_teacher = _verify(plan["tools"]["search_teacher"], "Search teacher")
    rank4_teacher = _verify(plan["tools"]["rank4_teacher"], "Rank-4 teacher")
    search_model = _verify(
        plan["training"]["search_teacher_runtime"], "Search teacher runtime"
    )
    source_ids = configuration["source_identities"]
    search_shallow = labels / "search-shallow.jsonl"
    rank4_shallow = labels / "rank4-shallow.jsonl"
    _label(
        manager=manager, ordinal=3, name="search-shallow",
        positions=positions, output=search_shallow,
        teacher=search_teacher, schema=selfsearch.SEARCH_TEACHER_SCHEMA,
        campaign_id=CAMPAIGN_ID, nodes=configuration["search_shallow_nodes"],
        source_sha256=source_ids["search_teacher_source_sha256"],
        workers=configuration["label_workers"],
        chunk_games=configuration["label_chunk_games"], model=search_model,
    )
    _label(
        manager=manager, ordinal=4, name="rank4-shallow",
        positions=positions, output=rank4_shallow,
        teacher=rank4_teacher, schema=selfsearch.RANK4_TEACHER_SCHEMA,
        campaign_id=CAMPAIGN_ID, nodes=configuration["rank4_shallow_nodes"],
        source_sha256=source_ids["rank4_teacher_source_sha256"],
        workers=configuration["label_workers"],
        chunk_games=configuration["label_chunk_games"],
    )
    hard = materialized / "hard-positions.tsv"
    hard_manifest = materialized / "hard-positions.manifest.json"
    def hard_stage() -> dict[str, Any]:
        hard_payload, hard_document = selfsearch.select_hard_positions(
            positions_tsv=positions,
            search_labels=search_shallow,
            rank4_labels=rank4_shallow,
            numerator=1,
            denominator=4,
        )
        if hard_document.get("selected") != 16_000:
            raise HoldoutError("fresh holdout hard subset is not exactly one quarter")
        _write_pair(hard_payload, hard_document, hard, hard_manifest)
        return {"selected": hard_document["selected"], "games": hard_document["games"]}

    hard_result = manager.execute(
        ordinal=5,
        name="hard-selection",
        configuration={"fraction": [1, 4]},
        producers={"workflow": pathlib.Path(selfsearch.__file__)},
        inputs={
            "positions": positions,
            "search_labels": search_shallow,
            "rank4_labels": rank4_shallow,
        },
        outputs={"positions": hard, "manifest": hard_manifest},
        resumable_outputs={"positions", "manifest"},
        action=hard_stage,
    )
    if hard_result.get("selected") != 16_000:
        raise HoldoutError("fresh hard-selection receipt changed")
    search_deep = labels / "search-deep.jsonl"
    rank4_deep = labels / "rank4-deep.jsonl"
    _label(
        manager=manager, ordinal=6, name="search-deep",
        positions=hard, output=search_deep,
        teacher=search_teacher, schema=selfsearch.SEARCH_TEACHER_SCHEMA,
        campaign_id=CAMPAIGN_ID, nodes=configuration["search_deep_nodes"],
        source_sha256=source_ids["search_teacher_source_sha256"],
        workers=configuration["label_workers"],
        chunk_games=configuration["label_chunk_games"], model=search_model,
    )
    _label(
        manager=manager, ordinal=7, name="rank4-deep",
        positions=hard, output=rank4_deep,
        teacher=rank4_teacher, schema=selfsearch.RANK4_TEACHER_SCHEMA,
        campaign_id=CAMPAIGN_ID, nodes=configuration["rank4_deep_nodes"],
        source_sha256=source_ids["rank4_teacher_source_sha256"],
        workers=configuration["label_workers"],
        chunk_games=configuration["label_chunk_games"],
    )
    search_merged = labels / "search-merged.jsonl"
    rank4_merged = labels / "rank4-merged.jsonl"

    def merge_stage(
        shallow: pathlib.Path, deep: pathlib.Path, output: pathlib.Path, schema: str
    ) -> dict[str, Any]:
        _merge_labels(shallow, deep, output, schema)
        return {"rows": sum(1 for _ in output.open("rb"))}

    search_merge_result = manager.execute(
        ordinal=8,
        name="search-targets",
        configuration={"deep_override": True},
        producers={"workflow": pathlib.Path(selfsearch.__file__)},
        inputs={"shallow": search_shallow, "deep": search_deep},
        outputs={"labels": search_merged},
        resumable_outputs={"labels"},
        action=lambda: merge_stage(
            search_shallow, search_deep, search_merged,
            selfsearch.SEARCH_TEACHER_SCHEMA,
        ),
    )
    rank4_merge_result = manager.execute(
        ordinal=9,
        name="rank4-targets",
        configuration={"deep_override": True},
        producers={"workflow": pathlib.Path(selfsearch.__file__)},
        inputs={"shallow": rank4_shallow, "deep": rank4_deep},
        outputs={"labels": rank4_merged},
        resumable_outputs={"labels"},
        action=lambda: merge_stage(
            rank4_shallow, rank4_deep, rank4_merged,
            selfsearch.RANK4_TEACHER_SCHEMA,
        ),
    )
    if (
        search_merge_result.get("rows") != 64_000
        or rank4_merge_result.get("rows") != 64_000
    ):
        raise HoldoutError("fresh merged-label stage count changed")
    canonical_teacher = _verify(
        plan["tools"]["canonical_teacher"], "fresh canonical teacher"
    )
    canonical_labels = labels / "canonical.jsonl"
    canonical_result = manager.execute(
        ordinal=10,
        name="canonical-targets",
        configuration={
            "nodes": configuration["canonical_nodes"],
            "deep_nodes": configuration["canonical_deep_nodes"],
            "deep_percent": configuration["canonical_deep_percent"],
            "max_samples": configuration["canonical_max_samples_per_game"],
            "workers": configuration["label_workers"],
            "chunk_games": configuration["label_chunk_games"],
        },
        producers={
            "teacher": canonical_teacher,
            "workflow": pathlib.Path(__file__),
        },
        inputs={"games": games},
        outputs={"labels": canonical_labels},
        resumable_outputs={"labels"},
        action=lambda: _canonical_label_chunks(
            games=games,
            teacher=canonical_teacher,
            output=canonical_labels,
            receipt_root=manager.receipts / "10-canonical-chunks",
            nodes=configuration["canonical_nodes"],
            deep_nodes=configuration["canonical_deep_nodes"],
            deep_percent=configuration["canonical_deep_percent"],
            max_samples=configuration["canonical_max_samples_per_game"],
            workers=configuration["label_workers"],
            chunk_games=configuration["label_chunk_games"],
        ),
    )
    priors = _packing_priors(plan, output_root)
    pack_root = materialized / "shards"
    def pack_stage(
        ordinal: int, name: str, label_path: pathlib.Path,
    ) -> dict[str, Any]:
        output = pack_root / name
        report_path = output / "pack-report.json"
        return manager.execute(
            ordinal=ordinal,
            name=f"pack-{name}",
            configuration={
                "all_roots_split": "test",
                "prior_manifests": len(priors),
            },
            producers={
                "pack": _verify(plan["tools"]["pack_tool"], "successor pack tool"),
                "workflow": pathlib.Path(selfsearch.__file__),
            },
            inputs={
                "roots": roots,
                "labels": label_path,
                **{f"prior_{index}": path for index, path in enumerate(priors)},
            },
            outputs={"report": report_path},
            resumable_outputs={"report"},
            action=lambda: _pack_labels(
                plan=plan,
                roots=roots,
                labels=label_path,
                output=output,
                priors=priors,
            ),
        )

    search_report = pack_stage(11, "search", search_merged)
    rank4_report = pack_stage(12, "rank4", rank4_merged)
    canonical_report = pack_stage(
        13, "canonical", pathlib.Path(canonical_result["labels"]["path"])
    )
    reports = {
        "search": search_report,
        "rank4": rank4_report,
        "canonical": canonical_report,
    }
    group_isolation = _group_isolation(priors, reports)
    split_isolation = _fresh_split_isolation(priors, reports)
    stage_names = {
        1: "games",
        2: "positions",
        3: "search-shallow",
        4: "rank4-shallow",
        5: "hard-selection",
        6: "search-deep",
        7: "rank4-deep",
        8: "search-targets",
        9: "rank4-targets",
        10: "canonical-targets",
        11: "pack-search",
        12: "pack-rank4",
        13: "pack-canonical",
    }
    stage_receipts = [
        _record(manager.receipt_path(ordinal, name))
        for ordinal, name in stage_names.items()
    ]
    receipt_path = holdout_root / "materialization-receipt.json"
    qualification.write_sealed(receipt_path, {
        "schema": MATERIALIZATION_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID,
        "status": "fresh-protected-holdout-materialized-once",
        "claim": qualification.artifact_reference(claim_path, CLAIM_SCHEMA),
        "immutable_selection": qualification.artifact_reference(
            selection_path, successor.SELECTION_SCHEMA
        ),
        "game_plan": _record(game_plan_json),
        "game_plan_tsv": _record(game_plan_tsv),
        "game_plan_rows": game_plan["games"],
        "fresh_roots": _record(roots),
        "fresh_roots_tsv": _record(fresh_roots_tsv),
        "fresh_opening_bank": qualification.artifact_reference(fresh_opening_bank),
        "games": _record(games),
        "games_manifest": _record(games_manifest),
        "positions": _record(positions),
        "positions_manifest": _record(positions_manifest),
        "hard_positions": _record(hard),
        "search_labels": _record(search_merged),
        "rank4_labels": _record(rank4_merged),
        "canonical_labels": canonical_result["labels"],
        "canonical_label_rows": canonical_result["rows"],
        "packing_priors": [_record(path) for path in priors],
        "test_shards": {
            "search": _record(pathlib.Path(search_report["shards"]["test"]["manifest"])),
            "rank4": _record(pathlib.Path(rank4_report["shards"]["test"]["manifest"])),
            "canonical": _record(
                pathlib.Path(canonical_report["shards"]["test"]["manifest"])
            ),
        },
        "test_samples": {
            "search": search_report["shards"]["test"]["samples"],
            "rank4": rank4_report["shards"]["test"]["samples"],
            "canonical": canonical_report["shards"]["test"]["samples"],
        },
        "group_isolation": group_isolation,
        "split_isolation": split_isolation,
        "stage_receipts": stage_receipts,
        "selection_changed": False,
        "old_protected_tests_accessed": False,
        "fresh_protected_tests_opened": True,
    })
    if _sha256_file(selection_path) != claim["immutable_selection"]["sha256"]:
        raise HoldoutError("immutable selection changed during holdout materialization")
    return receipt_path


def _dataset(manifest: pathlib.Path) -> Any:
    document = _canonical(manifest, "fresh protected test manifest")
    shard = large_training.load_csr_shard(manifest)
    if document.get("split") != "test" or shard.split != "test" or len(shard) <= 0:
        raise HoldoutError("fresh protected test shard is not a nonempty test set")
    return compact.Dataset(
        indptr=shard.indptr,
        indices=shard.indices,
        targets=shard.targets,
        weights=shard.weights,
        group_ids=shard.group_ids,
        split="test",
        source_manifest_sha256=_sha256_file(manifest),
        source_npz_sha256=shard.npz_sha256,
        source_route=str(manifest.resolve()),
    )


def evaluate(
    *, plan_path: pathlib.Path, output_root: pathlib.Path,
) -> pathlib.Path:
    output_root = output_root.resolve()
    plan = successor.load_plan(plan_path, output_root=output_root)
    selection_path, selection, runtime_path = _selection(plan, output_root)
    selection_file_sha256 = _sha256_file(selection_path)
    holdout_root = output_root / "fresh-holdout"
    materialization = qualification.load_sealed(
        holdout_root / "materialization-receipt.json", MATERIALIZATION_SCHEMA
    )
    expected_selection = qualification.artifact_reference(
        selection_path, successor.SELECTION_SCHEMA
    )
    expected_claim = qualification.artifact_reference(
        holdout_root / "00-materialization-claim.json", CLAIM_SCHEMA
    )
    expected_materialization = qualification.artifact_reference(
        holdout_root / "materialization-receipt.json", MATERIALIZATION_SCHEMA
    )
    if (
        materialization.get("campaign_id") != CAMPAIGN_ID
        or materialization.get("status")
        != "fresh-protected-holdout-materialized-once"
        or materialization.get("claim") != expected_claim
        or materialization.get("immutable_selection") != expected_selection
        or materialization.get("group_isolation", {}).get("passed") is not True
        or materialization.get("split_isolation", {}).get("passed") is not True
        or materialization.get("selection_changed") is not False
        or materialization.get("old_protected_tests_accessed") is not False
        or materialization.get("fresh_protected_tests_opened") is not True
    ):
        raise HoldoutError("fresh holdout materialization ancestry changed")
    reference_path = holdout_root / "report-reference.json"
    if reference_path.exists():
        reference = qualification.load_sealed(
            reference_path, REPORT_REFERENCE_SCHEMA
        )
        report_record = reference.get("report")
        if not isinstance(report_record, Mapping) or not isinstance(
            report_record.get("path"), str
        ):
            raise HoldoutError("fresh holdout report reference is malformed")
        report_path = pathlib.Path(report_record["path"])
        if report_path.parent != holdout_root / "reports" or report_path.is_symlink():
            raise HoldoutError("fresh holdout report path changed")
        _verify(report_record, "fresh holdout report")
        report = qualification.load_sealed(report_path, REPORT_SCHEMA)
        if (
            reference.get("campaign_id") != CAMPAIGN_ID
            or reference.get("selection") != expected_selection
            or reference.get("complete") is not True
            or reference.get("selection_changed") is not False
            or report.get("successor_plan")
            != qualification.artifact_reference(plan_path, successor.PLAN_SCHEMA)
            or report.get("immutable_selection") != expected_selection
            or report.get("materialization") != expected_materialization
            or report.get("complete") is not True
            or report.get("selection_changed") is not False
            or report.get("old_protected_tests_accessed") is not False
        ):
            raise HoldoutError("fresh holdout report ancestry changed")
        return report_path
    architecture, quantized, _runtime_selection, _runtime = compact.load_runtime(
        runtime_path
    )
    effective = quantized.effective()
    datasets = {
        name: _dataset(_verify(record, f"fresh {name} test manifest"))
        for name, record in materialization["test_shards"].items()
    }
    arm = compact.ARMS["search-target"]
    metrics = {}
    for name, dataset in datasets.items():
        predictions = compact.predict_dataset(
            effective, architecture, dataset, quantized=quantized
        )
        metrics[name] = compact.metrics_from_predictions(
            predictions, dataset, arm
        )
        if any(
            isinstance(value, float) and not math.isfinite(value)
            for value in metrics[name].values()
        ):
            raise HoldoutError("fresh protected metrics are nonfinite")
    minimum = int(plan["fresh_protected_holdout"]["minimum_samples_per_report"])
    enough = all(len(dataset) >= minimum for dataset in datasets.values())
    if not enough:
        raise HoldoutError("fresh protected holdout is smaller than precommitted")
    body = {
        "schema": REPORT_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID,
        "status": "fresh-protected-holdout-complete",
        "successor_plan": qualification.artifact_reference(
            plan_path, successor.PLAN_SCHEMA
        ),
        "immutable_selection": qualification.artifact_reference(
            selection_path, successor.SELECTION_SCHEMA
        ),
        "runtime": _record(runtime_path),
        "materialization": qualification.artifact_reference(
            holdout_root / "materialization-receipt.json", MATERIALIZATION_SCHEMA,
        ),
        "samples": {name: len(dataset) for name, dataset in datasets.items()},
        "metrics": metrics,
        "minimum_samples_per_report": minimum,
        "sample_floor_passed": enough,
        "diagnostic_only": True,
        "selection_changed": False,
        "selection_may_change": False,
        "deployment_decision_changed": False,
        "old_protected_tests_accessed": False,
        "fresh_protected_tests_opened": True,
        "complete": True,
    }
    report_path, _report = successor._write_content_addressed(
        holdout_root / "reports", body, ".fresh-holdout-report.json"
    )
    qualification.write_sealed(reference_path, {
        "schema": REPORT_REFERENCE_SCHEMA,
        "namespace": NAMESPACE,
        "campaign_id": CAMPAIGN_ID,
        "report": _record(report_path),
        "selection": qualification.artifact_reference(
            selection_path, successor.SELECTION_SCHEMA
        ),
        "complete": True,
        "selection_changed": False,
    })
    if _sha256_file(selection_path) != selection_file_sha256:
        raise HoldoutError("successor selection changed during fresh evaluation")
    return report_path


def verify(
    *, plan_path: pathlib.Path, output_root: pathlib.Path,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    plan = successor.load_plan(plan_path, output_root=output_root)
    result = {
        "schema": "papersoccer.compact-value-bfm.fresh-holdout-status.v1",
        "campaign_id": CAMPAIGN_ID,
        "plan_body_sha256": plan["body_sha256"],
        "materialized": False,
        "evaluated": False,
        "old_protected_tests_accessed": False,
    }
    materialized = output_root / "fresh-holdout" / "materialization-receipt.json"
    if materialized.exists():
        value = qualification.load_sealed(materialized, MATERIALIZATION_SCHEMA)
        result["materialized"] = True
        result["samples"] = value["test_samples"]
    report_reference = output_root / "fresh-holdout" / "report-reference.json"
    if report_reference.exists():
        reference = qualification.load_sealed(
            report_reference, REPORT_REFERENCE_SCHEMA
        )
        report_path = _verify(reference["report"], "fresh holdout report")
        report = qualification.load_sealed(report_path, REPORT_SCHEMA)
        result["evaluated"] = True
        result["report"] = _record(report_path)
        result["metrics"] = report["metrics"]
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("materialize", "evaluate", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--plan", type=pathlib.Path, required=True)
        command.add_argument("--output-root", type=pathlib.Path, required=True)
        if name == "materialize":
            command.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "materialize":
            result: Any = materialize(
                plan_path=args.plan,
                output_root=args.output_root,
                resume=args.resume,
            )
        elif args.command == "evaluate":
            result = evaluate(plan_path=args.plan, output_root=args.output_root)
        else:
            result = verify(plan_path=args.plan, output_root=args.output_root)
        if isinstance(result, pathlib.Path):
            result = {"path": str(result.resolve()), "sha256": _sha256_file(result)}
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (
        HoldoutError, compact.TrainingError, OSError, ValueError,
        json.JSONDecodeError, subprocess.SubprocessError,
    ) as error:
        print(f"compact fresh holdout failure: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
