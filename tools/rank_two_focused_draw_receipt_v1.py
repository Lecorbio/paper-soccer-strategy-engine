"""Replay one failed preparation recipe, retaining every primitive proposal."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tools import rank_two_focused_campaign_v1 as campaign
from tools import rank_two_focused_data_v2 as original
from tools import rank_two_focused_work_v1 as work


def replay(plan_path, output):
    plan = campaign.read(plan_path)
    work.checked(plan, "archive", launch=True)
    if plan["producer"] != campaign.record(__file__) or plan["original_producer"] != campaign.record(original.__file__):
        raise ValueError("failed draw recipe changed")
    ready = campaign.read(campaign.verify(plan["ancestry_ready"]))
    prior = campaign.read(campaign.verify(ready["prior"]))
    database = campaign.verify(prior["inputs"]["database"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    campaign.immutable(output / "CLAIM.json", dict(plan=campaign.record(plan_path), seed=2026100710, depth_schedule=[4, 6, 8]))
    rules = original.generation.rules
    apply = rules.apply_primitive
    rows, identifiers, calls = [], set(), 0

    def observed(state, direction):
        nonlocal calls
        calls += 1
        if calls > 2000000:
            raise RuntimeError("bounded failed-recipe proposal ceiling reached")
        turns = getattr(state, "_receipt_turns", ())
        partial = getattr(state, "_receipt_partial", "") + str(direction)
        path = "/".join((*turns, partial))
        identity = hashlib.sha256(path.encode()).hexdigest()
        if identity not in identifiers:
            identifiers.add(identity)
            rows.append(dict(prefix=path, kind="failed-preparation-primitive-proposal", training_eligible=False))
        mover = state.to_move
        result = apply(state, direction)
        if state.winner is not None or state.to_move != mover:
            state._receipt_turns = (*turns, partial)
            state._receipt_partial = ""
        else:
            state._receipt_turns = turns
            state._receipt_partial = partial
        return result

    rng, seen, accepted, failure = random.Random(2026100710), set(), [], None
    rules.apply_primitive = observed
    try:
        with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as db:
            for index in range(256):
                for _ in range(10000):
                    row = original.generation.fresh_root(rng, (4, 6, 8)[index % 3], seen)
                    feature = original.data.feature_key(original.rules.encode_active(original.games.state(row["transcript"])))
                    if feature in seen:
                        continue
                    blocked = any(db.execute("SELECT 1 FROM keys WHERE category=? AND value=?", (category, key)).fetchone()
                                  for category, key in (("states", row["state_sha256"]), ("features", feature)))
                    if blocked:
                        continue
                    seen.add(feature)
                    accepted.append(row)
                    break
                else:
                    raise ValueError("fresh training root draw exhausted; no seed replacement")
    except RuntimeError as error:
        failure = str(error)
    finally:
        rules.apply_primitive = apply
    campaign.immutable(output / "OBSERVED.json", dict(schema=campaign.SCHEMA + ".failed-root-observations",
        seed=2026100710, depth_schedule=[4, 6, 8], primitive_proposals=rows, accepted_roots=accepted,
        original_producer=plan["original_producer"], ancestry_ready=plan["ancestry_ready"], training_eligible=False))
    result = dict(passed=failure == "could not produce a fresh complete-turn root at the frozen depth",
                  reproduced_failure=failure, primitive_calls=calls, unique_proposal_paths=len(rows),
                  accepted_before_failure=len(accepted), new_games=0, observations=campaign.record(output / "OBSERVED.json"))
    campaign.immutable(output / "RESULT.json", result)
    campaign.immutable(output / "EXPOSURE_PENDING.json", dict(schema=campaign.SCHEMA + ".failed-root-mechanism-exposure",
        input=result["observations"], result=campaign.record(output / "RESULT.json"), recipe=campaign.record(__file__),
        training_eligible=False, before_next_fresh_bank=True))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(replay(args.plan, args.output)))


if __name__ == "__main__":
    main()
