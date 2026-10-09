"""Independently decode and retain every completed focused root checkpoint."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

from tools import rank_two_focused_checkpoint_v1 as storage
from tools import rank_two_focused_campaign_v1 as campaign
from tools import rank_two_focused_work_v1 as work

SCHEMA = campaign.SCHEMA + ".training-archive.v1"


def run(plan_path):
    plan = campaign.read(plan_path)
    work.checked(plan, "archive", launch=True)
    if plan["producer"] != campaign.record(__file__):
        raise ValueError("archive producer changed")
    for ref in plan["inputs"].values():
        campaign.verify(ref)
    training_plan = campaign.read(campaign.verify(plan["training_plan"]))
    execution = Path(training_plan["output"])
    result = campaign.read(campaign.verify(plan["training_result"]))
    if result["status"] != "complete" or result["plan"] != plan["training_plan"]:
        raise ValueError("completed training result differs")
    out = Path(plan["output"])
    campaign.immutable(out / "CLAIM.json", dict(plan=campaign.record(plan_path),
        new_games=0, training_updates=0))
    index_path = out / "ROOT_CHECKPOINT_INDEX.jsonl"
    count, epochs = 0, []
    with index_path.open("x") as stream:
        for phase, total in (("float", result["float_epochs"]), ("qat", result["qat_epochs"])):
            if len(list((execution/phase).glob("epoch-*/receipt.json"))) != total:
                raise ValueError("declared completed epoch count differs")
            for epoch in range(total):
                directory = execution/phase/f"epoch-{epoch:02d}"
                receipt = campaign.read(directory/"receipt.json")
                order = campaign.read(directory/"ORDER.json")["roots"]
                if len(order) != 512 or len(set(order)) != 512:
                    raise ValueError("complete512root update order required")
                if (receipt["plan"] != plan["training_plan"] or receipt["phase"] != phase
                        or receipt["epoch"] != epoch):
                    raise ValueError("epoch receipt binding differs")
                if len(list(directory.glob("root-*/CLAIM.json"))) != 512 or len(list(directory.glob("root-*/checkpoint.json"))) != 512:
                    raise ValueError("claimed root checkpoint archive incomplete")
                for index, root in enumerate(order):
                    if count % 32 == 0:
                        work.checked(plan, "archive")
                        gc.collect()
                    step = directory/f"root-{index:03d}"
                    claim = campaign.read(step/"CLAIM.json")
                    saved = campaign.read(step/"checkpoint.json")
                    if (claim["plan"] != plan["training_plan"] or claim["root"] != root
                            or claim["index"] != index or saved["cursor"]["phase"] != phase
                            or saved["cursor"]["epoch"] != epoch
                            or saved["cursor"]["next_root"] != index+1
                            or saved["cursor"]["root_order"] != order):
                        raise ValueError("root claim/checkpoint/cursor differs")
                    layout, raw = storage.decode(saved, plan["training_plan"])
                    if len(raw) != 605760:
                        raise ValueError("dd8 float/optimizer tensor length differs")
                    stream.write(json.dumps(dict(phase=phase, epoch=epoch, index=index,
                        root_group_id=root, receipt=campaign.record(step/"checkpoint.json"),
                        checkpoint=saved["checkpoint"], raw_sha256=saved["storage"]["raw_sha256"],
                        parameter_sha256=saved["parameter_sha256"], optimizer_step=saved["optimizer_step"]),
                        sort_keys=True, separators=(",", ":"))+"\n")
                    count += 1
                    del raw, saved
                if receipt["checkpoint"] != campaign.read(directory/"root-511/checkpoint.json"):
                    raise ValueError("final epoch checkpoint differs")
                for key in ("source", "runtime"):
                    if receipt.get(key):
                        campaign.verify(receipt[key])
                if phase == "qat":
                    if not receipt["native"]["passed"]:
                        raise ValueError("retained native diagnostic failed")
                    campaign.verify(receipt["native"]["binary"])
                epochs.append(campaign.record(directory/"receipt.json"))
                campaign.atomic(out/"PROGRESS.json", dict(root_checkpoints_decoded=count,
                    expected=(result["float_epochs"]+result["qat_epochs"])*512))
    expected = (result["float_epochs"]+result["qat_epochs"])*512
    if count != expected:
        raise ValueError("complete root checkpoint count differs")
    proofs = dict(calibration=campaign.record(execution/"CALIBRATION.json"),
        warm_start=campaign.record(execution/"WARM_START.json"),
        float_stop=campaign.record(execution/"FLOAT_STOP.json"),
        qat_stop=campaign.record(execution/"QAT_STOP.json"))
    archive = dict(schema=SCHEMA, passed=True, plan=campaign.record(plan_path),
        training_result=plan["training_result"], training_plan=plan["training_plan"],
        root_checkpoint_count=count, every_tensor_and_optimizer_digest_verified=True,
        checkpoint_index=campaign.record(index_path), epochs=epochs, retained_proofs=proofs,
        new_games=0, training_updates=0, all_checkpoints_retained=True)
    campaign.immutable(out/"RESULT.json", archive)
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    result = run(parser.parse_args().plan)
    print(json.dumps(dict(passed=result["passed"], root_checkpoints=result["root_checkpoint_count"])))


if __name__ == "__main__":
    main()
