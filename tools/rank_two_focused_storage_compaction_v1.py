"""Recover disk headroom while preserving generated artifacts' exact bytes."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from tools import rank_two_focused_campaign_v1 as campaign
from tools import rank_two_focused_work_v1 as work

SCHEMA = campaign.SCHEMA + ".storage-compaction.v1"


def run(plan_path):
    plan = campaign.read(plan_path)
    work.checked(plan, "archive", launch=True)
    if plan["schema"] != SCHEMA + ".plan" or plan["producer"] != campaign.record(__file__):
        raise ValueError("storage compaction producer differs")
    campaign.verify(plan["ditto"])
    out = Path(plan["output"])
    out.mkdir(parents=True, exist_ok=True)
    receipts = []
    for index, reference in enumerate(plan["files"]):
        work.checked(plan, "archive")
        source = campaign.verify(reference)
        if not source.is_relative_to(campaign.ROOT / "results"):
            raise ValueError("only generated campaign data may be compacted")
        directory = out / "files" / f"{index:05d}"
        result_path = directory / "RESULT.json"
        if result_path.exists():
            previous = campaign.read(result_path)
            if previous["source"] != reference or previous["logical_bytes_preserved"] is not True:
                raise ValueError("retained compaction receipt differs")
            receipts.append(campaign.record(result_path))
            continue
        if (directory / "CLAIM.json").exists():
            raise RuntimeError("unknown file compaction claim retained; inspect exact bytes before recovery")
        before = source.stat()
        temporary = source.with_name(source.name + ".hfs-compaction-temporary")
        if temporary.exists():
            raise RuntimeError("unclassified compaction temporary retained")
        # Start with small/generated plaintext files, then larger indexes once
        # the prior savings provide space for their compressed temporary.
        if shutil.disk_usage(source).free < plan["minimum_working_free_bytes"]:
            raise RuntimeError("insufficient working headroom for lossless compaction")
        campaign.immutable(directory / "CLAIM.json", dict(source=reference,
            plan=campaign.record(plan_path), temporary=str(temporary), original_logical_bytes=before.st_size,
            original_allocated_bytes=before.st_blocks * 512))
        command = [plan["ditto"]["path"], "--noclone", "--hfsCompression", str(source), str(temporary)]
        result = subprocess.run(command, capture_output=True, timeout=plan["maximum_file_wall_seconds"])
        campaign.immutable(directory / "stderr.txt", result.stderr)
        if result.returncode:
            raise RuntimeError("transparent compaction failed; original and temporary retained")
        compressed = campaign.record(temporary)
        if compressed["sha256"] != reference["sha256"] or temporary.stat().st_size != before.st_size:
            raise ValueError("lossless storage copy changed logical bytes; original retained")
        saved = before.st_blocks * 512 - temporary.stat().st_blocks * 512
        if saved > 0:
            os.replace(temporary, source)
        else:
            temporary.unlink()
        campaign.verify(reference)
        after = source.stat()
        record = dict(source=reference, logical_bytes_preserved=True, logical_bytes=before.st_size,
            before_allocated_bytes=before.st_blocks * 512, after_allocated_bytes=after.st_blocks * 512,
            saved_allocated_bytes=max(0, saved), transparent_compression=after.st_blocks * 512 < before.st_blocks * 512,
            command=command, free_disk_bytes=shutil.disk_usage(source).free)
        campaign.immutable(result_path, record)
        receipts.append(campaign.record(result_path))
        campaign.atomic(out / "PROGRESS.json", dict(files_completed=len(receipts), total_files=len(plan["files"]),
            free_disk_bytes=shutil.disk_usage(source).free))
        if index == 0 and not record["transparent_compression"]:
            raise RuntimeError("filesystem transparent compression unavailable; original bytes preserved")
        if shutil.disk_usage(source).free >= plan["target_free_bytes"]:
            break
    result = dict(passed=True, plan=campaign.record(plan_path), files_completed=len(receipts),
        total_declared_files=len(plan["files"]), receipts=receipts, all_logical_bytes_preserved=True,
        free_disk_bytes=shutil.disk_usage(out).free, target_free_bytes=plan["target_free_bytes"],
        target_reached=shutil.disk_usage(out).free >= plan["target_free_bytes"],
        unrelated_files_modified=False, data_or_checkpoints_discarded=False, new_games=0)
    campaign.immutable(out / "RESULT.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.plan)
    print(json.dumps({key: result[key] for key in ("passed", "files_completed", "free_disk_bytes", "target_reached")}))


if __name__ == "__main__":
    main()
