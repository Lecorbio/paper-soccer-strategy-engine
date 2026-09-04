#!/usr/bin/env python3
"""Campaign-bound fresh-holdout entrypoint for discrete successor v3."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from collections.abc import Mapping


HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load v3 holdout dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v3 = _load(
    HERE / "compact_value_bfm_discrete_v3.py",
    "compact_discrete_v3_holdout_campaign",
)
fresh = _load(
    HERE / "compact_value_bfm_fresh_holdout.py",
    "compact_discrete_v3_fresh_primitives",
)

fresh.successor = v3
fresh.qualification = v3.qualification
fresh.HoldoutError = v3.V3Error
fresh.NAMESPACE = v3.NAMESPACE
fresh.CAMPAIGN_ID = f"{v3.SUCCESSOR_CAMPAIGN_ID}-holdout"


def _v3_packing_priors(plan: Mapping, output_root: pathlib.Path) -> list[pathlib.Path]:
    training_path = output_root / "training-input.json"
    if training_path.is_symlink() or not training_path.is_file():
        raise v3.V3Error("v3 training input is not a regular file")
    training = v3.qualification.load_sealed(
        training_path, v3.TRAINING_INPUT_SCHEMA
    )
    artifacts = plan["training"]["safe_input_artifacts"]
    declarations = [training["new_train_manifest"]]
    declarations.extend(record["manifest"] for record in artifacts["anchor"])
    declarations.extend(
        record["manifest"] for record in artifacts["canonical_validation"]
    )
    declarations.extend(
        record["manifest"] for record in artifacts["common_adjudicator"]
    )
    if len(declarations) != 8:
        raise v3.V3Error("v3 fresh holdout requires exactly eight clean priors")
    retired = v3.retired_protected_paths(
        pathlib.Path(plan["training"]["source_bundle_manifest"]["path"])
    )
    priors = []
    for index, declaration in enumerate(declarations):
        path = v3._declared_record(declaration, "v3 fresh holdout prior")
        if path in retired:
            raise v3.V3Error("retired protected test prior is forbidden before access")
        if index == 0 and path.parent != (
            v3.canonical_v1_root() / "global-repack" / "search"
        ):
            raise v3.V3Error("v3 predecessor global train prior path changed")
        path = v3._verify_record(declaration, "v3 fresh holdout prior")
        if fresh._canonical(path, "v3 fresh holdout prior").get("split") == "test":
            raise v3.V3Error("v3 fresh holdout cannot use a test shard as prior")
        priors.append(path)
    return priors


fresh._packing_priors = _v3_packing_priors


if __name__ == "__main__":
    raise SystemExit(fresh.main())
