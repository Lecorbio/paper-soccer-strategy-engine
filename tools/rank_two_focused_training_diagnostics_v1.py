"""Read-only, complete-split diagnostics on retained focused checkpoints."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

from tools import rank_two_focused_training_v8 as training
from tools import compact_representation_train as corpus_reader
from tools import rank_two_focused_native_v3 as native_probe
from tools import rank_two_focused_campaign_v1 as campaign
from tools import rank_two_focused_work_v1 as work

SCHEMA = campaign.SCHEMA + ".retained-training-diagnostics.v1"


class SplitView:
    def __init__(self, corpus, split, plan, output, phase):
        self.corpus, self.split, self.plan = corpus, split, plan
        self.output, self.phase, self.completed = output, phase, 0
        self.roots = {"validation": corpus.roots[split]}

    def root(self, requested_split, root):
        if requested_split != "validation":
            raise ValueError("diagnostic evaluation split differs")
        work.checked(self.plan, "experiment")
        if self.completed % 32 == 0:
            gc.collect()
        self.completed += 1
        campaign.atomic(self.output / "PROGRESS.json", dict(
            phase=self.phase, split=self.split, roots_claimed=self.completed,
            total_roots=len(self.roots["validation"]), training_updates=0, new_games=0))
        return self.corpus.root(self.split, root)


def compare(actual, expected):
    for name in ("huber", "ranking", "objective", "regret", "early_regret",
                 "quantized_action_flip_fraction"):
        if abs(actual[name] - expected[name]) > 1e-9:
            raise ValueError("retained validation replay differs: " + name)
    for name in ("validation_roots", "comparable_roots", "early_roots"):
        if actual[name] != expected[name]:
            raise ValueError("retained validation coverage differs: " + name)


def run(plan_path):
    plan = campaign.read(plan_path)
    work.checked(plan, "experiment", launch=True)
    if plan["producer"] != campaign.record(__file__):
        raise ValueError("diagnostic producer binding differs")
    for reference in plan["inputs"].values():
        campaign.verify(reference)
    corpus = corpus_reader.Corpus(campaign.verify(plan["corpus"]))
    if len(corpus.roots["train"]) != 512 or len(corpus.roots["validation"]) != 128:
        raise ValueError("complete retained512TRAIN/128VALIDATION scope required")
    output = Path(plan["output"])
    if (output / "CLAIM.json").exists():
        raise RuntimeError("spent diagnostic claim retained")
    campaign.immutable(output / "CLAIM.json", dict(plan=campaign.record(plan_path),
        training_updates=0, new_games=0, fresh_teacher_queries=0))
    selected = campaign.read(campaign.verify(plan["selected_epoch"]))
    best_float = campaign.read(campaign.verify(plan["best_float_epoch"]))
    runtime = campaign.read(campaign.verify(selected["runtime"]))
    scales = dict(zip(("w1", "w2", "w3"), runtime["scales"], strict=True))
    binary = campaign.verify(selected["native"]["binary"])
    reports = {}
    for phase, receipt, quantized in (
            ("best_float", best_float, False),
            ("selected_qat_float", selected, False),
            ("selected_qat_native", selected, True)):
        architecture, parameters = training.initialize("dd8", 0)
        optimizer = training.core.AdamW(parameters, learning_rate=.001, weight_decay=1e-5)
        training.restore(receipt["checkpoint"], parameters, optimizer, receipt["plan"])
        def native(root, active):
            requests = ["eval " + str(len(row)) + " " + " ".join(map(str, row))
                        for row in active]
            answers = native_probe.query(binary, requests)
            if any(answer.startswith("ERROR") for answer in answers):
                raise ValueError("retained native diagnostic prediction failed")
            return [float(answer) for answer in answers]
        reports[phase] = {}
        for split in ("train", "validation"):
            view = SplitView(corpus, split, plan, output, phase)
            metrics = training.validation(view, parameters, architecture,
                scales if quantized else None, native if quantized else None)
            reports[phase][split] = metrics
            campaign.immutable(output / (phase + "-" + split + ".json"), dict(
                passed=True, metrics=metrics, phase=phase, split=split,
                epoch=plan["selected_epoch"] if receipt is selected else plan["best_float_epoch"],
                corpus=plan["corpus"], all_declared_roots_used=True, training_updates=0))
            if split == "validation":
                expected = receipt["validation"] if quantized or phase == "best_float" else receipt["float_validation"]
                compare(metrics, expected)
    result = dict(schema=SCHEMA, passed=True, plan=campaign.record(plan_path),
        reports=reports, training_roots=512, validation_roots=128,
        prior_validation_metrics_reproduced=True, source_modified=False,
        training_updates=0, new_checkpoints=0, new_games=0, fresh_teacher_queries=0,
        gameplay_or_protected_data_read=False, all_declared_roots_used=True,
        playing_strength_qualified=False)
    campaign.immutable(output / "RESULT.json", result)
    campaign.immutable(output / "EXPOSURE_PENDING.json", dict(
        schema=SCHEMA + ".exposure", retained_artifacts=[campaign.record(output / "RESULT.json")],
        retained_plans=[campaign.record(plan_path)], before_next_fresh_bank=True,
        training_eligible=False, new_games=0, states_previously_retained=True))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.plan)
    print(json.dumps(dict(passed=result["passed"], training_roots=512,
                          validation_roots=128, training_updates=0, new_games=0)))


if __name__ == "__main__":
    main()
