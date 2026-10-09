"""Ten fixed teacher workers with immutable per-parent claims and recovery."""
from __future__ import annotations

import os
from pathlib import Path
import resource
import sys

if __name__ == "__main__" and sys.argv[1:2] == ["--cpu-child"]:
    bound_seconds = int(sys.argv[2])
    resource.setrlimit(resource.RLIMIT_CPU, (bound_seconds, bound_seconds))
    os.execv(sys.argv[3], sys.argv[3:])

import argparse
import gzip
import json
import subprocess
import time

from tools import rank_two_focused_campaign_v1 as campaign
from tools import rank_two_focused_teacher_budget_v1 as data_pipeline

data = data_pipeline.data


def payload(position):
    return (data.HEADER + "\t".join(str(position[key]) for key in
            ("position_id", "root_group_id", "group_id", "source", "split", "winner", "mover", "prefix"))
            + "\n").encode("ascii")


def validate_result(plan, position, directory, returncode):
    raw, error = directory / "output.partial.jsonl", directory / "stderr.txt"
    unsupported = (returncode == 2 and position["edges"] > 12 and
        error.read_text().splitlines()[:1] == ["jacek replay search teacher: " + position["position_id"]
        + ": unsolved search teacher completed no root visits"])
    success, message = False, None
    try:
        if returncode != 0:
            raise ValueError("teacher returned " + str(returncode))
        with raw.open() as stream:
            line = stream.readline()
            if not line or stream.readline():
                raise ValueError("one source-bound parent result required")
        data_pipeline.normalized_label(json.loads(line), position, plan)
        success = True
    except Exception as error_value:
        message = type(error_value).__name__ + ": " + str(error_value)
    result = dict(plan_body_sha256=plan["bundle_sha256"], position_id=position["position_id"],
                  success=success, unsupported=unsupported, returncode=returncode, failure=message,
                  output=data.record(raw), stderr=data.record(error),
                  classification="valid-teacher-label" if success else
                  "unlabeled-late-parent-no-root-visits-at-declared-budget" if unsupported else "unexpected-label-failure")
    campaign.immutable(directory / "RESULT.json", result)
    campaign.immutable(directory / "EXPOSURE.json", dict(
        schema=data_pipeline.SCHEMA + ".teacher-parent-exposure", input=campaign.record(directory / "CLAIM.json")
        if (directory / "CLAIM.json").exists() else None, output=result["output"],
        stderr=result["stderr"], result=campaign.record(directory / "RESULT.json"),
        training_eligible=False, before_next_fresh_bank=True))
    return result


def run(plan_path, output, workers=10):
    plan = campaign.read(plan_path)
    data_pipeline.check_current(plan, "label", launch=True)
    data_pipeline.verify_teacher_plan(plan)
    for reference in plan["inputs"].values():
        campaign.verify(reference)
    if (plan["schema"] != data_pipeline.SCHEMA + ".teacher-plan" or plan["nodes"] != 256000
            or plan["outcomes_as_targets"] is not False or workers != plan["maximum_workers"]
            or plan["inputs"]["label_runner"] != campaign.record(__file__)):
        raise ValueError("matched teacher recipe or fixed worker count differs")
    for name in ("teacher", "native_teacher", "games", "positions_file"):
        data.bound(plan[name])
    for name in ("producer", "validator", "lifecycle"):
        campaign.verify(plan[name])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    binding = campaign.record(plan_path)
    campaign.immutable(output / "EXECUTION.json", dict(plan=binding, producer=campaign.record(__file__),
        workers=workers, numerical_threads=1, maximum_position_cpu_seconds=plan["maximum_position_cpu_seconds"],
        unknown_claim_policy="retain-and-stop; never automatically replay"))
    campaign.immutable(output / "RESERVED_EXPOSURE.json", dict(
        schema=data_pipeline.SCHEMA + ".teacher-exposure", plan=binding,
        execution=campaign.record(output / "EXECUTION.json"), training_eligible=False,
        before_next_fresh_bank=True, all_parent_inputs_and_native_recipes_retained=True))
    pending, completed, active = [], {}, {}
    for index, position in enumerate(plan["positions"]):
        directory = output / "parents" / f"{index:05d}"
        if (directory / "RESULT.json").exists():
            result = campaign.read(directory / "RESULT.json")
            if (result["plan_body_sha256"] != plan["bundle_sha256"]
                    or result["position_id"] != position["position_id"]
                    or not (result["success"] or result["unsupported"])):
                raise RuntimeError("failed/changed parent retained; no automatic replay")
            data.bound(result["output"])
            data.bound(result["stderr"])
            completed[index] = result
        elif (directory / "CLAIM.json").exists():
            raise RuntimeError("unknown teacher parent claim retained; explicit recovery required")
        else:
            pending.append(index)
    started = time.monotonic()
    try:
        while pending or active:
            data_pipeline.check_current(plan, "label")
            while pending and len(active) < workers:
                data_pipeline.check_current(plan, "label")
                index = pending.pop(0)
                position = plan["positions"][index]
                directory = output / "parents" / f"{index:05d}"
                campaign.immutable(directory / "positions.tsv", payload(position))
                campaign.immutable(directory / "CLAIM.json", dict(plan=binding, position=position,
                    command=plan["command"], maximum_cpu_seconds=plan["maximum_position_cpu_seconds"], maximum_wall_seconds=plan["maximum_position_wall_seconds"]))
                stdin = (directory / "positions.tsv").open("rb")
                stdout = (directory / "output.partial.jsonl").open("xb")
                stderr = (directory / "stderr.txt").open("xb")
                command = [sys.executable, "-m", "tools.rank_two_focused_labels_v5", "--cpu-child", str(plan["maximum_position_cpu_seconds"]), *plan["command"]]
                try:
                    process = subprocess.Popen(command, stdin=stdin, stdout=stdout, stderr=stderr)
                finally:
                    stdin.close(); stdout.close(); stderr.close()
                active[index] = (process, time.monotonic())
            progressed = False
            for index, (process, begin) in list(active.items()):
                if time.monotonic() - begin > plan["maximum_position_wall_seconds"] and process.poll() is None:
                    process.kill()
                code = process.poll()
                if code is None:
                    continue
                result = validate_result(plan, plan["positions"][index],
                                         output / "parents" / f"{index:05d}", code)
                completed[index] = result
                del active[index]
                progressed = True
                if not (result["success"] or result["unsupported"]):
                    raise RuntimeError("teacher failure retained; cancel only unclaimed parents")
            if progressed:
                campaign.atomic(output / "PROGRESS.json", dict(completed_parents=len(completed),
                    requested_parents=len(plan["positions"]), valid_labels=sum(r["success"] for r in completed.values()),
                    unsupported_labels=sum(r["unsupported"] for r in completed.values()),
                    wall_seconds=time.monotonic() - started, fixed_workers=workers))
            else:
                time.sleep(.05)
    finally:
        # Any interrupted child has its claim, input, stdout and stderr retained.
        # It receives no success receipt, so subsequent runs refuse to repeat it.
        for process, _ in active.values():
            if process.poll() is None:
                process.kill()
            process.wait()
    result = dict(passed=True, plan=binding, requested_parents=len(plan["positions"]),
                  valid_labels=sum(r["success"] for r in completed.values()),
                  unsupported_labels=sum(r["unsupported"] for r in completed.values()),
                  parent_receipts=[campaign.record(output / "parents" / f"{i:05d}" / "RESULT.json")
                                   for i in range(len(plan["positions"]))],
                  outcomes_as_targets=False, wall_seconds=time.monotonic() - started)
    campaign.immutable(output / "RESULT.json", result)
    campaign.immutable(output / "EXPOSURE_PENDING.json", dict(
        schema=data_pipeline.SCHEMA + ".teacher-exposure", plan=binding,
        execution=campaign.record(output / "RESULT.json"), training_eligible=False,
        before_next_fresh_bank=True, all_partial_outputs_retained=True))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    print(json.dumps(run(args.plan, args.output, args.workers)))


if __name__ == "__main__":
    main()
