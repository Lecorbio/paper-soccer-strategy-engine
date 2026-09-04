#!/usr/bin/env python3
"""Campaign-bound fresh-holdout entrypoint for quantization successor v2."""

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
        raise RuntimeError(f"cannot load v2 holdout dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v2 = _load(
    HERE / "compact_value_bfm_quantization_v2.py",
    "compact_quantization_v2_holdout_campaign",
)
fresh = _load(
    HERE / "compact_value_bfm_fresh_holdout.py",
    "compact_quantization_v2_fresh_primitives",
)

# The reviewed primitive implementation reads these globals dynamically.  This
# wrapper changes only the campaign-specific plan/selection validator and IDs;
# all generation, teacher, packing, isolation, and report logic remains bound
# byte-for-byte to the maintained fresh-holdout implementation.
fresh.successor = v2
fresh.qualification = v2.qualification
fresh.HoldoutError = v2.V2Error
fresh.NAMESPACE = v2.NAMESPACE
fresh.CAMPAIGN_ID = f"{v2.SUCCESSOR_CAMPAIGN_ID}-holdout"


def _v2_packing_priors(plan: Mapping, output_root: pathlib.Path) -> list[pathlib.Path]:
    training_path = output_root / "training-input.json"
    if training_path.is_symlink() or not training_path.is_file():
        raise v2.V2Error("v2 training input is not a regular file")
    training = v2.qualification.load_sealed(
        training_path, v2.TRAINING_INPUT_SCHEMA
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
        raise v2.V2Error("v2 fresh holdout requires exactly eight clean priors")
    bundle = pathlib.Path(plan["training"]["source_bundle_manifest"]["path"])
    retired = v2.retired_protected_paths(bundle)
    priors = []
    for index, declaration in enumerate(declarations):
        path = v2._declared_record(declaration, "v2 fresh holdout prior")
        if path in retired:
            raise v2.V2Error("retired protected test prior is forbidden before access")
        if index == 0 and path.parent != (
            v2.canonical_v1_root() / "global-repack" / "search"
        ):
            raise v2.V2Error("v2 predecessor global train prior path changed")
        path = v2._verify_record(declaration, "v2 fresh holdout prior")
        if fresh._canonical(path, "v2 fresh holdout prior").get("split") == "test":
            raise v2.V2Error("v2 fresh holdout cannot use a test shard as prior")
        priors.append(path)
    return priors


fresh._packing_priors = _v2_packing_priors


if __name__ == "__main__":
    raise SystemExit(fresh.main())
