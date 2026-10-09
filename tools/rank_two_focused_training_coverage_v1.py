"""Audit all retained TRAIN roots without updates, games or teacher queries."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import json
from pathlib import Path

from tools import rank_two_focused_training_v8 as training
from tools import compact_representation_train as reader
from tools import rank_two_focused_campaign_v1 as campaign
from tools import rank_two_focused_work_v1 as work

np = training.np
SCHEMA = campaign.SCHEMA + ".training-coverage.v1"


def mean(rows, name):
    values = [row[name] for row in rows if row[name] is not None]
    return float(np.mean(values)) if values else None


def run(plan_path):
    plan = campaign.read(plan_path)
    current = work.checked(plan, "experiment", launch=True)
    if (plan["producer"] != campaign.record(__file__)
            or current["focused_opponent_training_review"] != plan["review"]
            or current["focused_family"] != "dd8"
            or plan["split"] != "train"
            or plan["roots"] != 512):
        raise ValueError("complete training-only diagnostic binding differs")
    for ref in plan["inputs"].values():
        campaign.verify(ref)
    corpus = reader.Corpus(campaign.verify(plan["corpus"]))
    if len(corpus.roots["train"]) != plan["roots"]:
        raise ValueError("retained complete training scope differs")
    root_names = {}
    for ref, key in ((plan["original_generation_plan"], "rows"),
                     (plan["expansion_bank"], "rows")):
        for row in campaign.read(campaign.verify(ref))[key]:
            if row["split"] != "train":
                continue
            root = row["root_group_id"]
            if root in root_names:
                raise ValueError("training root metadata duplicated")
            root_names[root] = row["opponent"]
    if set(root_names) != set(corpus.roots["train"]):
        raise ValueError("opponent metadata does not cover every TRAIN root")
    roster = campaign.read(campaign.verify(plan["roster"]))
    if set(root_names.values()) != set(roster["opponents"]):
        raise ValueError("all frozen opponents required")
    out = Path(plan["output"])
    campaign.immutable(out / "CLAIM.json", dict(plan=campaign.record(plan_path),
        split="train", roots=512, training_updates=0, new_games=0, teacher_queries=0))
    epoch = campaign.read(campaign.verify(plan["float_epoch"]))
    if epoch["phase"] != "float":
        raise ValueError("pre-QAT float checkpoint required")
    architecture, parameters = training.initialize("dd8", 0)
    optimizer = training.core.AdamW(parameters, learning_rate=.001, weight_decay=1e-5)
    training.restore(epoch["checkpoint"], parameters, optimizer, epoch["plan"])
    roots, label_status, per_opponent = [], Counter(), defaultdict(list)
    for index, root in enumerate(sorted(corpus.roots["train"])):
        work.checked(plan, "experiment")
        groups = corpus.root("train", root)
        if any(group["split"] != "train" for group in groups):
            raise ValueError("nontraining group entered diagnostics")
        active, target, signs, spans = training.root_rows(groups)
        predictions, _ = training.forward(parameters, architecture, active)
        huber, rank, _ = training.objective(groups, predictions, target, signs, spans)
        regrets, early, margins = [], [], []
        informative, exhaustive, alternatives = 0, 0, 0
        for group, span in zip(groups, spans, strict=True):
            teacher = target[span] * signs[span]
            for child in group["successors"]:
                proof = child.get("proof") or {}
                proven = proof.get("solved") is True and proof.get("proven_winner") in (0, 1)
                label_status["solved_proof" if proven else "not_proven_by_search"] += 1
            if not group["exhaustive"] or len(teacher) < 2:
                continue
            exhaustive += 1
            if float(np.max(teacher) - np.min(teacher)) <= plan["comparable_tolerance"]:
                continue
            informative += 1
            picked = int(np.argmax(predictions[span] * signs[span]))
            regret = float(np.max(teacher) - teacher[picked])
            regrets.append(regret)
            if group["edges"] <= 12:
                early.append(regret)
            sorted_values = np.sort(teacher)
            margins.append(float(sorted_values[-1] - sorted_values[-2]))
            alternatives += regret >= plan["hard_regret_threshold"]
        row = dict(root_group_id=root, opponent=root_names[root], groups=len(groups),
            exhaustive_multi_action_groups=exhaustive, informative_groups=informative,
            huber=huber, ranking=rank, objective=huber+.25*rank,
            regret=float(np.mean(regrets)) if regrets else None,
            early_regret=float(np.mean(early)) if early else None,
            mean_best_second_teacher_margin=float(np.mean(margins)) if margins else None,
            hard_disagreements=alternatives,
            hard_disagreement_fraction=alternatives/informative if informative else None)
        roots.append(row)
        per_opponent[row["opponent"]].append(row)
        campaign.atomic(out / "PROGRESS.json", dict(roots_completed=index+1,
            total_roots=512, split="train", training_updates=0, new_games=0))
        del groups, active, target, signs, spans, predictions
        if index % 32 == 0:
            gc.collect()
    names = ("huber", "ranking", "objective", "regret", "early_regret")
    aggregate = {name: mean(roots, name) for name in names}
    previous = campaign.read(campaign.verify(plan["previous_diagnostics"]))["reports"]["best_float"]["train"]
    if any(abs(aggregate[name] - previous[name]) > 1e-9 for name in names):
        raise ValueError("complete retained TRAIN metrics did not reproduce")
    root_ref = campaign.immutable(out / "ROOT_DIAGNOSTICS.json", dict(split="train", roots=roots))
    result = dict(schema=SCHEMA, passed=True, plan=campaign.record(plan_path),
        root_diagnostics=root_ref, roots=512, all_declared_training_roots_used=True,
        aggregate=aggregate, prior_train_metrics_reproduced=True,
        per_opponent={name: dict(roots=len(rows),
            **{metric: mean(rows, metric) for metric in (*names,
                "mean_best_second_teacher_margin", "hard_disagreement_fraction")},
            informative_groups=sum(row["informative_groups"] for row in rows),
            hard_disagreements=sum(row["hard_disagreements"] for row in rows))
            for name, rows in sorted(per_opponent.items())},
        teacher_label_counts=dict(label_status), teacher_quality_independently_established=False,
        actual_live_jacek_or_marchete_data=False, playing_strength_qualified=False,
        new_games=0, fresh_teacher_queries=0, training_updates=0, validation_position_reads=0,
        protected_or_per_game_development_reads=False, original_sources_modified=False)
    campaign.immutable(out / "RESULT.json", result)
    campaign.immutable(out / "EXPOSURE_PENDING.json", dict(schema=SCHEMA+".exposure",
        retained_artifacts=[campaign.record(out / "RESULT.json"), root_ref],
        retained_plans=[campaign.record(plan_path)], new_games=0,
        training_eligible=False, before_next_fresh_bank=True, states_previously_retained=True))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    result = run(parser.parse_args().plan)
    print(json.dumps(dict(passed=result["passed"], roots=result["roots"],
        prior_train_metrics_reproduced=result["prior_train_metrics_reproduced"])))


if __name__ == "__main__":
    main()
