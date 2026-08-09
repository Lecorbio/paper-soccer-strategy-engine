#!/usr/bin/env python3

"""Freeze the still-unseen T8 banks as the candidate-independent T9 ladder."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pathlib


HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REFERENCE = HERE / "reference"

T8_EVIDENCE_MANIFEST = REFERENCE / "t8_evidence_manifest.json"
T8_VALIDATION = REFERENCE / "t8_prospective_validation.tsv"
T8_FINAL = REFERENCE / "t8_sealed_final.tsv"
T9_VALIDATION = REFERENCE / "t9_prospective_validation.tsv"
T9_FINAL = REFERENCE / "t9_sealed_final.tsv"
T9_EVIDENCE_MANIFEST = REFERENCE / "t9_evidence_manifest.json"

T8_EVIDENCE_MANIFEST_SHA256 = (
    "0723c580f0e4f01f433bfae83aa71a1ede83c9e7e04cce969dd1818019121c08"
)
T8_VALIDATION_SHA256 = (
    "e670fc39902308b66debd8deb3dc82fb9e0ce0f61562b3078328e99350c54f3b"
)
T8_FINAL_SHA256 = (
    "e6c8efaa094576ad4ac3dc22a69ea595f224aaa64d7c3ecdc39b7e98c7dfb204"
)
T8_PROTOCOL_SHA256 = (
    "8d73a1c92d43d73a8ebe48a63084f5f5c578ead9516058920c0700165ec3851c"
)

T8_FRONTIER_CANDIDATE = "frontier_proof"
T8_FRONTIER_SUBMISSION_SHA256 = (
    "35ffa4c9b30327750c1ca5fa50f6d41a282252f98d187c372a74131b148cafe1"
)
T8_BOUND_MANIFEST_SHA256 = (
    "fb52a513eb29e074814c1cec8dc0cecbf619b4aea2bfb7797d13a6070ab0b810"
)
T8_DEVELOPMENT_BANK_SHA256 = (
    "048b86ab0ba781a7cb1289b1ff4712070af6e384f91fd157895ac9e92772f319"
)
T8_DECISION_SHA256 = (
    "de4ccaf5c52497b93bced41b6ba4ba1bd77c5e5cd735ea3de44a9be2333ca6f3"
)
T8_DEVELOPMENT_REPORT_SHA256 = (
    "e6ae982d50bcefda59eb12a2481c5815dc4090fb4714f942981207ee2b4f75b2"
)

T8_RESULT_DIRECTORY = (
    ROOT
    / "results/codingame/promotion/frontier_proof/"
      "35ffa4c9b3032775-fb52a513eb29"
)
HOLDOUT_LEDGER = ROOT / ".git/papersoccer-promotion"
T8_FINAL_LEDGER = (
    HOLDOUT_LEDGER
    / f"locked-test-consumption-{T8_FINAL_SHA256[:16]}.json"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_json(value) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def verified_bytes(path: pathlib.Path, expected_sha256: str) -> bytes:
    data = path.read_bytes()
    actual_sha256 = sha256_bytes(data)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"hash mismatch for {path.relative_to(ROOT)}: expected "
            f"{expected_sha256}, found {actual_sha256}"
        )
    return data


def verify_bound_manifest_if_still_active() -> None:
    """Verify the historical binding while it is still the active manifest."""
    path = HERE / "manifest.json"
    data = path.read_bytes()
    if sha256_bytes(data) != T8_BOUND_MANIFEST_SHA256:
        # Once T9 is bound, the immutable decision below remains the durable
        # identity for this retired T8 binding.
        return
    manifest = json.loads(data)
    if (
        manifest.get("candidate") != T8_FRONTIER_CANDIDATE
        or manifest.get("candidate_submission_sha256")
        != T8_FRONTIER_SUBMISSION_SHA256
        or manifest.get("evidence", {}).get("status")
        != "candidate_bound_before_t8_outcomes"
        or manifest.get("evidence_manifest_sha256")
        != T8_EVIDENCE_MANIFEST_SHA256
    ):
        raise RuntimeError("T8 bound-manifest identity changed")


def audit_nonconsumption() -> dict:
    """Inspect artifact identities and names, never prospective game outcomes."""
    verify_bound_manifest_if_still_active()
    decision_path = T8_RESULT_DIRECTORY / "decision.json"
    development_path = T8_RESULT_DIRECTORY / "development.json"
    decision_data = verified_bytes(decision_path, T8_DECISION_SHA256)
    verified_bytes(development_path, T8_DEVELOPMENT_REPORT_SHA256)
    decision = json.loads(decision_data)
    expected_stage_status = {
        "development": "reject",
        "initial": "pass",
        "test": "not_run_due_to_rejection",
        "validation": "not_run_due_to_rejection",
    }
    expected_artifacts = {
        str((T8_RESULT_DIRECTORY / name).relative_to(ROOT))
        for name in (
            "preflight.json",
            "initial.json",
            "development.json",
        )
    }
    if (
        decision.get("schema")
        != "papersoccer.codingame-promotion-decision.v1"
        or decision.get("bot") != T8_FRONTIER_CANDIDATE
        or decision.get("candidate_submission_sha256")
        != T8_FRONTIER_SUBMISSION_SHA256
        or decision.get("manifest_sha256") != T8_BOUND_MANIFEST_SHA256
        or decision.get("failed_stage") != "development"
        or decision.get("stage_status") != expected_stage_status
        or set(decision.get("artifacts", [])) != expected_artifacts
        or decision.get("verdict") != "REJECT"
        or decision.get("submission_worthy") is not False
    ):
        raise RuntimeError("T8 retirement decision identity changed")

    forbidden_paths = [
        T8_RESULT_DIRECTORY / "validation.json",
        T8_RESULT_DIRECTORY / "test.json",
        T8_RESULT_DIRECTORY / "shards/validation",
        T8_RESULT_DIRECTORY / "shards/test",
    ]
    present = [path for path in forbidden_paths if path.exists()]
    if present:
        raise RuntimeError(
            "T8 prospective result artifacts exist: "
            + ", ".join(str(path.relative_to(ROOT)) for path in present)
        )

    prospective_hashes = (T8_VALIDATION_SHA256, T8_FINAL_SHA256)
    snapshot_hits = []
    if T8_RESULT_DIRECTORY.exists():
        for path in T8_RESULT_DIRECTORY.rglob("*"):
            if path.is_file() and any(
                digest in path.name for digest in prospective_hashes
            ):
                snapshot_hits.append(path)
    if snapshot_hits:
        raise RuntimeError(
            "T8 immutable prospective bank snapshots exist: "
            + ", ".join(
                str(path.relative_to(ROOT)) for path in snapshot_hits
            )
        )

    if T8_FINAL_LEDGER.exists():
        raise RuntimeError(
            f"T8 final ledger marker exists: {T8_FINAL_LEDGER.relative_to(ROOT)}"
        )
    ledger_bank_hits = []
    if HOLDOUT_LEDGER.exists():
        for path in HOLDOUT_LEDGER.glob("locked-test-consumption-*.json"):
            try:
                payload = json.loads(path.read_text())
            except json.JSONDecodeError as error:
                raise RuntimeError(f"invalid holdout ledger JSON: {path}") from error
            if payload.get("bank_sha256") == T8_FINAL_SHA256:
                ledger_bank_hits.append(path)
    if ledger_bank_hits:
        raise RuntimeError("a holdout ledger consumed the T8 final bank")

    return {
        "audit_scope": {
            "official_result_identity": str(T8_RESULT_DIRECTORY.relative_to(ROOT)),
            "holdout_ledger": str(HOLDOUT_LEDGER.relative_to(ROOT)),
            "method": (
                "verify deterministic decision identity and artifact paths; "
                "inspect no prospective report or shard contents"
            ),
        },
        "retired_binding": {
            "candidate": T8_FRONTIER_CANDIDATE,
            "candidate_submission_sha256": T8_FRONTIER_SUBMISSION_SHA256,
            "bound_manifest_sha256": T8_BOUND_MANIFEST_SHA256,
            "decision_sha256": T8_DECISION_SHA256,
            "development_bank_sha256": T8_DEVELOPMENT_BANK_SHA256,
            "development_report_sha256": T8_DEVELOPMENT_REPORT_SHA256,
            "failed_stage": "development",
            "retirement_reason": "exposed_adaptive_development_failure_only",
            "stage_status": expected_stage_status,
        },
        "prospective_artifacts": {
            "validation_report_exists": False,
            "test_report_exists": False,
            "validation_shard_directory_exists": False,
            "test_shard_directory_exists": False,
            "validation_immutable_bank_snapshot_exists": False,
            "test_immutable_bank_snapshot_exists": False,
            "final_ledger_marker_exists": False,
        },
        "conclusion": (
            "neither T8 prospective bank was run or consumed; the byte-identical "
            "banks remain outcome-unseen and may be carried forward to T9"
        ),
    }


def build() -> tuple[bytes, bytes, bytes]:
    t8_manifest_data = verified_bytes(
        T8_EVIDENCE_MANIFEST, T8_EVIDENCE_MANIFEST_SHA256
    )
    validation_data = verified_bytes(T8_VALIDATION, T8_VALIDATION_SHA256)
    final_data = verified_bytes(T8_FINAL, T8_FINAL_SHA256)
    t8 = json.loads(t8_manifest_data)
    if (
        t8.get("schema")
        != "papersoccer.candidate-independent-t8-evidence.v1"
        or t8.get("status") != "frozen_before_candidate_binding"
        or t8.get("candidate") is not None
        or t8.get("candidate_submission_sha256") is not None
        or t8.get("prospective_strength_protocol_sha256")
        != T8_PROTOCOL_SHA256
    ):
        raise RuntimeError("T8 evidence-manifest identity changed")
    protocol = copy.deepcopy(t8["prospective_strength_protocol"])
    if sha256_bytes(stable_json(protocol)) != T8_PROTOCOL_SHA256:
        raise RuntimeError("T8 prospective protocol bytes changed")

    source_bank_keys = {
        "validation": "reference/t8_prospective_validation.tsv",
        "test": "reference/t8_sealed_final.tsv",
    }
    alias_bank_keys = {
        "validation": "reference/t9_prospective_validation.tsv",
        "test": "reference/t9_sealed_final.tsv",
    }
    banks = {}
    for stage in ("validation", "test"):
        entry = copy.deepcopy(t8["banks"][source_bank_keys[stage]])
        entry["carried_forward_from"] = {
            "evidence_manifest_sha256": T8_EVIDENCE_MANIFEST_SHA256,
            "reference": source_bank_keys[stage],
            "sha256": entry["sha256"],
        }
        banks[alias_bank_keys[stage]] = entry

    audit = audit_nonconsumption()
    sources = copy.deepcopy(t8["sources"])
    sources.update({
        str(T8_EVIDENCE_MANIFEST.relative_to(ROOT)): T8_EVIDENCE_MANIFEST_SHA256,
        str(T8_VALIDATION.relative_to(ROOT)): T8_VALIDATION_SHA256,
        str(T8_FINAL.relative_to(ROOT)): T8_FINAL_SHA256,
        str(pathlib.Path(__file__).resolve().relative_to(ROOT)): sha256_bytes(
            pathlib.Path(__file__).read_bytes()
        ),
    })
    manifest = {
        "schema": "papersoccer.candidate-independent-t9-evidence.v1",
        "status": "frozen_before_candidate_binding",
        "candidate": None,
        "candidate_submission_sha256": None,
        "prospective_strength_protocol": protocol,
        "prospective_strength_protocol_sha256": T8_PROTOCOL_SHA256,
        "protocol_carry_forward": {
            "source_ladder": "T8",
            "source_evidence_manifest_sha256": T8_EVIDENCE_MANIFEST_SHA256,
            "source_protocol_sha256": T8_PROTOCOL_SHA256,
            "semantic_changes": [],
            "version_and_provenance_changes_only": True,
            "stage_bank_aliases": {
                source_bank_keys[stage]: alias_bank_keys[stage]
                for stage in ("validation", "test")
            },
        },
        "nonconsumption_audit": audit,
        "prior_decisions": {
            **copy.deepcopy(t8["prior_decisions"]),
            "t8_frontier_binding": {
                "status": (
                    "retired_on_exposed_development_failure_before_"
                    "prospective_stages"
                ),
                "candidate_submission_sha256": T8_FRONTIER_SUBMISSION_SHA256,
                "bound_manifest_sha256": T8_BOUND_MANIFEST_SHA256,
                "decision_sha256": T8_DECISION_SHA256,
            },
            "t8_validation": {
                "status": "unconsumed_carried_forward_to_t9",
                "sha256": T8_VALIDATION_SHA256,
            },
            "t8_final": {
                "status": "unconsumed_carried_forward_to_t9",
                "sha256": T8_FINAL_SHA256,
            },
        },
        "immutability": {
            "before_candidate_binding": (
                "T9 candidate and harness identities remain null; aliases, this "
                "manifest, and the inherited protocol are frozen"
            ),
            "at_candidate_binding": (
                "the future active manifest must pin this T9 evidence-manifest "
                "hash, inherited protocol hash, candidate submission, and runner "
                "hashes before either prospective stage runs"
            ),
            "after_candidate_binding": (
                "this evidence manifest and both T9 aliases are immutable; any "
                "candidate change requires a new versioned evidence ladder"
            ),
        },
        "selection": {
            **copy.deepcopy(t8["selection"]),
            "carry_forward": {
                "candidate_independent": True,
                "bank_content_changes": False,
                "bank_selection_changes": False,
                "prospective_outcomes_inspected": False,
                "reason": (
                    "the sole T8-bound candidate failed the exposed adaptive "
                    "development prerequisite before validation or final began"
                ),
            },
        },
        "source_semantics": (
            "T8 source hashes remain immutable acquisition provenance; T9 adds "
            "only the carry-forward freezer, source evidence manifest, and "
            "byte-identical reference aliases"
        ),
        "sources": dict(sorted(sources.items())),
        "banks": banks,
    }
    return validation_data, final_data, stable_json(manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    validation, final, manifest = build()
    artifacts = {
        T9_VALIDATION: validation,
        T9_FINAL: final,
        T9_EVIDENCE_MANIFEST: manifest,
    }
    stale = []
    for path, content in artifacts.items():
        if arguments.check:
            if not path.exists() or path.read_bytes() != content:
                stale.append(str(path.relative_to(ROOT)))
            continue
        if path.exists():
            if path.read_bytes() != content:
                raise FileExistsError(f"refusing to replace frozen {path}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        print(
            f"froze {path.relative_to(ROOT)} "
            f"sha256={sha256_bytes(content)}"
        )
    if stale:
        raise SystemExit("stale T9 evidence artifacts: " + ", ".join(stale))


if __name__ == "__main__":
    main()
