"""Freeze a root-balanced hard-position mixture using TRAIN labels only."""
from __future__ import annotations

import argparse
import ast
import copy
import gc
import hashlib
import json
import math
from pathlib import Path
import shutil
import zlib

from tools import rank_two_focused_training_v9 as retained
from tools import compact_representation_train as reader
from tools import rank_two_focused_campaign_v1 as campaign
from tools import rank_two_focused_work_v1 as work

np = retained.np
SCHEMA = campaign.SCHEMA + ".hard-position-mixture.v2"


def weights_for_root(groups, parameters, architecture, recipe):
    if not groups or any(group["split"] != "train" for group in groups):
        raise ValueError("hard-position calibration accepts TRAIN only")
    active, targets, signs, spans = retained.root_rows(groups)
    predictions, _ = retained.forward(parameters, architecture, active)
    scores = []
    for group, span in zip(groups, spans, strict=True):
        teacher = targets[span] * signs[span]
        comparable = (group["exhaustive"] and len(teacher) >= 2
            and float(np.max(teacher) - np.min(teacher)) > recipe["comparable_tolerance"])
        best = float(np.max(teacher))
        picked = int(np.argmax(predictions[span] * signs[span]))
        regret = best - float(teacher[picked])
        hard = (comparable and regret >= recipe["teacher_regret_threshold"]
            and best > recipe["minimum_best_teacher_value"])
        scores.append(dict(group_id=group["group_id"], hard=bool(hard),
            teacher_regret=regret, best_teacher_value=best, exhaustive=group["exhaustive"]))
    count, hard_count = len(groups), sum(row["hard"] for row in scores)
    mass = recipe["hard_mass"]
    for row in scores:
        row["weight"] = ((1-mass)/count + (mass/hard_count if row["hard"] else 0.)
            if hard_count else 1/count)
    if not math.isclose(sum(row["weight"] for row in scores), 1., abs_tol=1e-12):
        raise ValueError("root training mass differs")
    return scores


class MixtureCorpus:
    def __init__(self, corpus, reference):
        self.corpus = corpus
        self.manifest, self.roots = corpus.manifest, corpus.roots
        self.mixture = campaign.read(campaign.verify(reference))
        if (self.mixture["corpus"] != corpus.identity
                or set(self.mixture["roots"]) != set(corpus.roots["train"])):
            raise ValueError("exact TRAIN mixture/corpus binding differs")

    def root(self, split, root):
        groups = self.corpus.root(split, root)
        if split != "train":
            return groups
        rows = self.mixture["roots"][root]
        if [row["group_id"] for row in rows] != [g["group_id"] for g in groups]:
            raise ValueError("fixed TRAIN group order differs")
        for group, row in zip(groups, rows, strict=True):
            group["training_group_weight"] = row["weight"]
        return groups


def objective(groups, predictions, targets, signs, spans, ranking_weight=.25):
    if all("training_group_weight" not in g for g in groups):
        return retained.objective(groups, predictions, targets, signs, spans, ranking_weight)
    if (any(g["split"] != "train" or "training_group_weight" not in g for g in groups)
            or not math.isclose(sum(g["training_group_weight"] for g in groups), 1., abs_tol=1e-12)):
        raise ValueError("weighted training root membership/mass differs")
    if all(g["training_group_weight"] == 1/len(groups) for g in groups):
        return retained.objective(groups, predictions, targets, signs, spans, ranking_weight)
    gradient = np.zeros_like(predictions)
    huber = ranking = 0.
    for group, span in zip(groups, spans, strict=True):
        weight = group["training_group_weight"]
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("finite positive TRAIN group mass required")
        h, r, local = retained.objective([group], predictions[span], targets[span],
            signs[span], [slice(0, span.stop-span.start)], ranking_weight)
        huber += h * weight
        ranking += r * weight
        gradient[span] = local * np.float32(weight)
    return huber, ranking, gradient


def function_trees(path):
    tree = ast.parse(Path(path).read_text())
    return {node.name: ast.dump(node, include_attributes=False)
        for node in tree.body if isinstance(node, ast.FunctionDef)}


def prepare(plan_path):
    plan = campaign.read(plan_path)
    current = work.checked(plan, "experiment", launch=True)
    if plan["producer"] != campaign.record(__file__) or current["focused_round_04_declaration"] != plan["declaration"]:
        raise ValueError("mixture preparation declaration/producer differs")
    for ref in plan["inputs"].values():
        campaign.verify(ref)
    declaration = campaign.read(campaign.verify(plan["declaration"]))
    recipe = declaration["mixture_recipe"]
    if recipe != dict(hard_mass=.5, teacher_regret_threshold=.05,
            minimum_best_teacher_value=-.95, comparable_tolerance=1e-4,
            split="train", score_checkpoint=plan["score_epoch"], fixed_during_training=True):
        raise ValueError("human-approved fixed mixture recipe differs")
    corpus = reader.Corpus(campaign.verify(plan["corpus"]))
    if len(corpus.roots["train"]) != 512 or len(corpus.roots["validation"]) != 128:
        raise ValueError("matched retained512TRAIN/128VALIDATION required")
    out = Path(plan["output"])
    campaign.immutable(out / "CLAIM.json", dict(plan=campaign.record(plan_path),
        new_games=0, training_updates=0, validation_position_reads=0))
    epoch = campaign.read(campaign.verify(plan["score_epoch"]))
    architecture, parameters = retained.initialize("dd8", 0)
    optimizer = retained.core.AdamW(parameters, learning_rate=.001, weight_decay=1e-5)
    retained.restore(epoch["checkpoint"], parameters, optimizer, epoch["plan"])
    roots, gradient_checks, hard_roots = {}, [], 0
    for index, root in enumerate(sorted(corpus.roots["train"])):
        work.checked(plan, "experiment")
        groups = corpus.root("train", root)
        rows = weights_for_root(groups, parameters, architecture, recipe)
        roots[root] = rows
        hard_roots += any(row["hard"] for row in rows)
        # Verify the unweighted branch exactly and the weighted prediction
        # derivatives against an independent central finite difference.
        if index < 3:
            active, target, signs, spans = retained.root_rows(groups)
            prediction, _ = retained.forward(parameters, architecture, active)
            a = retained.objective(groups, prediction, target, signs, spans)
            b = objective(groups, prediction, target, signs, spans)
            if a[:2] != b[:2] or not np.array_equal(a[2], b[2]):
                raise ValueError("uniform objective ceased to be identical")
            weighted = copy.deepcopy(groups)
            for group, row in zip(weighted, rows, strict=True):
                group["training_group_weight"] = row["weight"]
            h, r, gradient = objective(weighted, prediction, target, signs, spans)
            samples = np.linspace(0, len(prediction)-1, min(8, len(prediction)), dtype=int)
            maximum_error = 0.
            for coordinate in samples:
                plus, minus = prediction.copy(), prediction.copy()
                delta = .001
                plus[coordinate] += np.float32(delta)
                minus[coordinate] -= np.float32(delta)
                hp, rp, _ = objective(weighted, plus, target, signs, spans)
                hm, rm, _ = objective(weighted, minus, target, signs, spans)
                numerical = ((hp+.25*rp)-(hm+.25*rm))/(2*delta)
                maximum_error = max(maximum_error, abs(numerical-float(gradient[coordinate])))
            if maximum_error > 1e-4:
                raise ValueError("weighted prediction gradient check failed")
            gradient_checks.append(dict(root_group_id=root, samples=len(samples),
                maximum_absolute_error=maximum_error, uniform_exact=True))
        del groups
        if index % 32 == 0:
            gc.collect()
        campaign.atomic(out / "PROGRESS.json", dict(roots_completed=index+1, total_roots=512))
    mixture = dict(schema=SCHEMA, passed=True, plan=campaign.record(plan_path),
        corpus=plan["corpus"], score_epoch=plan["score_epoch"], recipe=recipe,
        roots=roots, root_count=512, hard_roots=hard_roots,
        validation_position_reads=0, outcomes_as_targets=False, frozen_before_training=True)
    mref = campaign.immutable(out / "MIXTURE.json", mixture)
    old = function_trees(plan["retained_training"]["path"])
    new = function_trees(plan["training_adapter"]["path"])
    preserved = [name for name in old if name not in ("objective", "validate_corpus_binding", "fit")]
    if any(old[name] != new.get(name) for name in preserved):
        raise ValueError("nonintervention numerical/storage functions changed")
    proof = campaign.immutable(out / "CHECKS.json", dict(passed=True, mixture=mref,
        all512TRAIN_roots_scored=True, fixed_mixture=True, gradient_checks=gradient_checks,
        preserved_functions=preserved, original_producer=plan["retained_training"],
        new_producer=plan["training_adapter"], all_optimizer_moments_preserved=True,
        validation_weighting_unchanged=True, exact_proof_and_terminal_targets_unchanged=True,
        new_games=0, training_updates=0, fresh_teacher_queries=0))
    result = dict(passed=True, mixture=mref, checks=proof, roots=512,
        hard_roots=hard_roots, new_games=0, training_updates=0, validation_position_reads=0)
    campaign.immutable(out / "RESULT.json", result)
    campaign.immutable(out / "EXPOSURE_PENDING.json", dict(schema=SCHEMA+".exposure",
        retained_plans=[campaign.record(plan_path)], retained_artifacts=[mref, proof],
        training_eligible=False, before_next_fresh_bank=True, new_games=0))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    result = prepare(parser.parse_args().plan)
    print(json.dumps({key: result[key] for key in ("passed", "roots", "hard_roots")}))


if __name__ == "__main__":
    main()
