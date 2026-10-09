"""Check exact streaming coordinate calibration before retained QAT recovery."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools import rank_two_focused_training_v6 as retained
from tools import rank_two_focused_training_v7 as streaming
from tools import compact_representation_train as reader
from tools import rank_two_focused_campaign_v1 as campaign
from tools import rank_two_focused_work_v1 as work

SCHEMA = campaign.SCHEMA + ".calibration-stream-checks.v1"


def run(plan_path):
    plan = campaign.read(plan_path)
    work.checked(plan, "archive", launch=True)
    if plan["producer"] != campaign.record(__file__):
        raise ValueError("calibration check producer differs")
    for reference in plan["inputs"].values():
        campaign.verify(reference)
    training = campaign.read(campaign.verify(plan["training_plan"]))
    out = Path(plan["output"])
    campaign.immutable(out / "CLAIM.json", dict(plan=campaign.record(plan_path),
        calibration_only=True, validation_consumed=False, training_updates=0))
    corpus = reader.Corpus(campaign.verify(training["corpus"]))
    roots = sorted(corpus.roots["train"])
    selected = [roots[0], roots[len(roots)//2], roots[-1]]
    groups = {root: corpus.root("train", root) for root in selected}
    architecture = retained.Architecture("dd8", 8, 8)
    parameters = {name: retained.np.zeros(shape, dtype=retained.np.float32)
                  for name, shape in architecture.shapes.items()}
    optimizer = retained.core.AdamW(parameters, learning_rate=.001, weight_decay=1e-5)
    best = campaign.read(campaign.verify(plan["best_float_epoch"]))
    retained.restore(best["checkpoint"], parameters, optimizer, plan["training_plan"])
    class Fixture:
        roots = {"train": {root: True for root in selected}}
        def root(self, split, root):
            if split != "train":
                raise AssertionError("validation must never calibrate scales")
            return groups[root]
    old_scales, old_report = retained.calibrate(parameters, architecture, [groups[root] for root in selected])
    new_scales, new_report = streaming.calibrate_stream(parameters, architecture, Fixture(), selected)
    if old_report != new_report or any(old_scales[name].tobytes() != new_scales[name].tobytes() for name in old_scales):
        raise AssertionError("streaming changes float32 scales or ordered calibration losses")
    rejected = False
    try:
        streaming.calibrate_stream(parameters, architecture, Fixture(), selected[:2])
    except ValueError:
        rejected = True
    if not rejected:
        raise AssertionError("incomplete TRAIN root list accepted")
    result = dict(passed=True, plan=campaign.record(plan_path),
        retained_producer=campaign.record(retained.__file__), streaming_producer=campaign.record(streaming.__file__),
        calibration_reports_bit_equal=True, scale_float32_bytes_equal=True,
        coordinate_trials=len(new_report["trials"]), root_order=selected,
        selected_scales=new_report["selected_scales"], exact_training_root_set_required=True,
        validation_consumed=False, training_updates=0, new_games=0, best_float_epoch=plan["best_float_epoch"])
    campaign.immutable(out / "RESULT.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.plan)))


if __name__ == "__main__":
    main()
