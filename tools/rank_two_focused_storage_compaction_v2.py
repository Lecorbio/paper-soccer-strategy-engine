"""Single-thread streaming filesystem compression with independent byte checks."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import struct
import zlib

from tools import rank_two_focused_campaign_v1 as campaign
from tools import rank_two_focused_work_v1 as work

SCHEMA = campaign.SCHEMA + ".storage-compaction.v2"
BLOCK = 65536


def compressed_copy(source, temporary):
    """Write the documented zlib resource-fork representation in64KiB chunks."""
    size = source.stat().st_size
    blocks = (size + BLOCK - 1) // BLOCK
    if not 0 < size < 2**32 - blocks * 8 - 400:
        raise ValueError("bounded resource-fork offsets required")
    temporary.open("xb").close()
    cursor = 0x108 + blocks * 8
    with source.open("rb") as stream, open(str(temporary) + "/..namedfork/rsrc", "w+b") as fork:
        fork.truncate(cursor)
        fork.seek(0x104); fork.write(struct.pack("<I", blocks))
        for index in range(blocks):
            raw = stream.read(BLOCK)
            encoded = zlib.compress(raw, 6)
            if len(encoded) > len(raw):
                encoded = b"\xff" + raw
            fork.seek(cursor); fork.write(encoded)
            fork.seek(0x108 + index * 8)
            fork.write(struct.pack("<II", cursor - 0x104, len(encoded)))
            cursor += len(encoded)
        if stream.read(1):
            raise ValueError("source grew during compression")
        fork.seek(0)
        fork.write(struct.pack(">IIII", 0x100, cursor, cursor - 0x100, 0x32))
        fork.seek(0x100); fork.write(struct.pack(">I", cursor - 0x104))
        footer = bytearray(50)
        struct.pack_into(">HHHI I", footer, 24, 0x1C, 0x32, 0, 0x636D7066, 0xA)
        struct.pack_into("<Q", footer, 38, 0xFFFF0100)
        fork.seek(cursor); fork.write(footer); fork.truncate(cursor + 50)
        fork.flush(); os.fsync(fork.fileno())
    os.setxattr(temporary, "com.apple.decmpfs", struct.pack("<IIQ", 0x636D7066, 4, size))
    os.chflags(temporary, temporary.stat().st_flags | 0x20)


def run(plan_path):
    plan = campaign.read(plan_path)
    work.checked(plan, "archive", launch=True)
    if plan["schema"] != SCHEMA + ".plan" or plan["producer"] != campaign.record(__file__):
        raise ValueError("streaming storage producer differs")
    for reference in plan["format_sources"]:
        campaign.verify(reference)
    out = Path(plan["output"]); out.mkdir(parents=True, exist_ok=True)
    receipts = []
    for index, reference in enumerate(plan["files"]):
        work.checked(plan, "archive")
        source = campaign.verify(reference)
        if not source.is_relative_to(campaign.ROOT / "results"):
            raise ValueError("only declared generated data may be compacted")
        directory = out / "files" / f"{index:05d}"
        if (directory / "CLAIM.json").exists():
            raise RuntimeError("spent streaming file claim retained")
        before = source.stat()
        temporary = source.with_name(source.name + ".stream-compaction-temporary")
        campaign.immutable(directory / "CLAIM.json", dict(plan=campaign.record(plan_path),
            source=reference, temporary=str(temporary), original_allocated_bytes=before.st_blocks * 512))
        compressed_copy(source, temporary)
        actual = campaign.record(temporary)
        if actual["sha256"] != reference["sha256"] or temporary.stat().st_size != before.st_size:
            raise ValueError("kernel-decoded copy differs; original retained")
        saved = before.st_blocks * 512 - temporary.stat().st_blocks * 512
        if saved > 0:
            os.chmod(temporary, before.st_mode & 0o7777)
            os.utime(temporary, ns=(before.st_atime_ns, before.st_mtime_ns))
            os.replace(temporary, source)
        else:
            temporary.unlink()
        campaign.verify(reference)
        result = dict(source=reference, logical_bytes_preserved=True,
            before_allocated_bytes=before.st_blocks * 512, after_allocated_bytes=source.stat().st_blocks * 512,
            saved_allocated_bytes=max(0, saved), free_disk_bytes=shutil.disk_usage(source).free,
            numerical_threads=1, compression_threads=1)
        campaign.immutable(directory / "RESULT.json", result)
        receipts.append(campaign.record(directory / "RESULT.json"))
        campaign.atomic(out / "PROGRESS.json", dict(files_completed=len(receipts),
            total_files=len(plan["files"]), free_disk_bytes=shutil.disk_usage(source).free))
        if index == 0 and saved <= 0:
            raise RuntimeError("transparent streaming compression ineffective; original preserved")
        if shutil.disk_usage(source).free >= plan["target_free_bytes"]:
            break
    result = dict(passed=True, plan=campaign.record(plan_path), receipts=receipts,
        files_completed=len(receipts), all_logical_bytes_preserved=True,
        free_disk_bytes=shutil.disk_usage(out).free, target_reached=shutil.disk_usage(out).free >= plan["target_free_bytes"],
        compression_threads=1, unrelated_files_modified=False, data_or_checkpoints_discarded=False)
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
