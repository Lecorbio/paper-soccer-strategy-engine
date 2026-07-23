#!/usr/bin/env python3

import argparse
import hashlib
import json
import math
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "jacek_article_value_model.json"
DEFAULT_HEADER = ROOT / "src" / "bots" / "jacek_neural_model.hpp"
TRAINER = ROOT / "tools" / "train_jacek_neural.py"


def float_literal(value):
    rendered = f"{float(value):.9g}"
    if "." not in rendered and "e" not in rendered.lower():
        rendered += ".0"
    return rendered + "F"


def array_lines(values, width=8):
    rows = []
    for start in range(0, len(values), width):
        row = ", ".join(float_literal(value)
                        for value in values[start:start + width])
        rows.append(f"    {row},")
    return "\n".join(rows)


def tensor(model, name, expected_shape):
    value = model["model"][name]
    if value.get("shape") != list(expected_shape):
        raise ValueError(f"{name} does not have the expected shape")
    values = value.get("values")
    count = 1
    for dimension in expected_shape:
        count *= dimension
    if not isinstance(values, list) or len(values) != count:
        raise ValueError(f"{name} does not contain every parameter")
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError(f"{name} contains a non-finite parameter")
    return values


def render(model, model_sha256):
    if model.get("schema") != "papersoccer.jacek-inspired-model.v1":
        raise ValueError("unexpected Jacek model schema")
    expected = {
        "feature_schema":
            "canonical-edges316-onehot-true-turn-distance105x8-v1",
        "input_count": 1156,
        "edge_count": 316,
        "vertex_count": 105,
        "distance_buckets": 8,
        "hidden_one": 32,
        "hidden_two": 32,
        "rules": {
            "width": 8,
            "height": 10,
            "goal_rule": "opponent-goal-only",
            "blocked_rule": "player-to-move-loses",
        },
    }
    for name, value in expected.items():
        if model.get(name) != value:
            raise ValueError(f"unexpected model field {name}")
    target = model.get("target", {})
    if target.get("kind") != "mover-relative-soft-alpha-beta-root-score":
        raise ValueError("unexpected Jacek model target")
    if target.get("temperature") != 12_000.0:
        raise ValueError("unexpected Jacek model target temperature")
    training = model.get("training", {})
    trainer_sha = hashlib.sha256(TRAINER.read_bytes()).hexdigest()
    if training.get("trainer_sha256") != trainer_sha:
        raise ValueError("model was not produced by the current trainer")
    expected_contract = {
        "feature_schema":
            "canonical-edges316-onehot-true-turn-distance105x8-v1",
        "rules": expected["rules"],
        "teacher": {
            "kind": "alpha-beta",
            "max_turn_depth": 5,
            "max_nodes": 4000,
            "transposition_table_entries": 16_384,
            "max_search_plies": 12,
        },
    }
    if training.get("corpus_contract") != expected_contract:
        raise ValueError("unexpected Jacek training corpus contract")

    w1 = tensor(model, "w1", (1156, 32))
    b1 = tensor(model, "b1", (32,))
    w2 = tensor(model, "w2", (32, 32))
    b2 = tensor(model, "b2", (32,))
    w3 = tensor(model, "w3", (32, 1))
    b3 = tensor(model, "b3", (1,))

    return f"""#pragma once

#include <array>
#include <cstddef>
#include <string_view>

namespace papersoccer::detail::jacek_neural_model {{

inline constexpr std::size_t kInputCount = 1156;
inline constexpr std::size_t kHiddenOne = 32;
inline constexpr std::size_t kHiddenTwo = 32;
inline constexpr float kTargetTemperature =
    {float_literal(target["temperature"])};
inline constexpr std::string_view kModelSha256 =
    "{model_sha256}";

inline constexpr std::array<float, {len(w1)}> kW1{{{{
{array_lines(w1)}
}}}};
inline constexpr std::array<float, 32> kB1{{{{
{array_lines(b1)}
}}}};
inline constexpr std::array<float, {len(w2)}> kW2{{{{
{array_lines(w2)}
}}}};
inline constexpr std::array<float, 32> kB2{{{{
{array_lines(b2)}
}}}};
inline constexpr std::array<float, 32> kW3{{{{
{array_lines(w3)}
}}}};
inline constexpr float kB3 = {float_literal(b3[0])};

}}  // namespace papersoccer::detail::jacek_neural_model
"""


def main():
    parser = argparse.ArgumentParser(
        description="Generate the checked-in Jacek neural model header.")
    parser.add_argument("--model", type=pathlib.Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_HEADER)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()

    raw = arguments.model.read_bytes()
    rendered = render(
        json.loads(raw), hashlib.sha256(raw).hexdigest())
    if arguments.check:
        if not arguments.output.exists() or arguments.output.read_text() != rendered:
            print(f"{arguments.output} is stale", file=sys.stderr)
            return 1
        return 0

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(rendered)
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
