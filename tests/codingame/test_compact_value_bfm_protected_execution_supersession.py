from __future__ import annotations

import hashlib
import pathlib
import tempfile
import unittest

from tools import compact_value_bfm_protected_execution_supersession as supersession


q = supersession.qualification
c = supersession.challenger
d = supersession.dual


def sealed(path: pathlib.Path, schema: str, **fields) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    q.write_sealed(path, {"schema": schema, **fields})
    return path


def rich(path: pathlib.Path, schema: str) -> dict:
    value = q.load_sealed(path, schema)
    raw = path.read_bytes()
    return {
        "path": str(path.resolve()), "bytes": len(raw),
        "sha256": q.sha256_bytes(raw), "schema": schema,
        "body_sha256": value["body_sha256"],
    }


def regular(path: pathlib.Path) -> dict:
    raw = path.read_bytes()
    return {
        "path": str(path.resolve()), "bytes": len(raw),
        "sha256": q.sha256_bytes(raw),
    }


class Fixture:
    def __init__(self, base: pathlib.Path, *, rich_bridge: bool = False):
        self.base = base
        self.campaign_root = base / "superseded-campaign"
        self.execution_root = self.campaign_root / "dual-final/attempt-000/execution"
        self.candidate_source = base / "candidate.cpp"
        self.candidate_source.write_bytes(b"int main(){}\n")
        self.candidate_runtime = base / "candidate.runtime.json"
        self.candidate_runtime.write_bytes(b"{}\n")
        self.candidate = {
            "runtime_sha256": q.sha256_file(self.candidate_runtime),
            "source_sha256": q.sha256_file(self.candidate_source),
        }
        candidate_record = {
            "runtime": regular(self.candidate_runtime),
            "source": regular(self.candidate_source),
            "architecture": {"id": "6301-12-8-1"},
        }
        self.campaign = sealed(
            self.campaign_root / "campaign-plan.json", c.PLAN_SCHEMA,
            outputs={"root": str(self.campaign_root)},
        )
        self.authorization = sealed(
            self.campaign_root / "dual-final/attempt-000/dual-final-authorization.json",
            c.DUAL_FINAL_AUTHORIZATION_SCHEMA,
            candidate=candidate_record,
        )
        self.execution = sealed(
            self.execution_root / "execution-plan.json", d.PLAN_SCHEMA,
            root=str(self.execution_root), production=True,
            evidence_mode="production-default-callables",
            status="dual-final-execution-planned-banks-unclaimed",
            campaign_plan=q.artifact_reference(self.campaign, c.PLAN_SCHEMA),
            authorization=q.artifact_reference(
                self.authorization, c.DUAL_FINAL_AUTHORIZATION_SCHEMA
            ),
            candidate=candidate_record,
        )
        self.bank_paths = {}
        bank_records = {}
        for gate_id in ("gate-a", "gate-b"):
            bank_path = self.execution_root / f"gates/{gate_id}/bank.json"
            bank_path.parent.mkdir(parents=True, exist_ok=True)
            bank_path.write_bytes(gate_id.encode("ascii"))
            self.bank_paths[gate_id] = bank_path
            bank_records[gate_id] = regular(bank_path)
        self.dual_plan = sealed(
            self.campaign_root / "dual-final/attempt-000/dual-final-plan.json",
            c.DUAL_FINAL_SCHEMA,
            candidate=candidate_record,
            authorization=rich(self.authorization, c.DUAL_FINAL_AUTHORIZATION_SCHEMA),
            gates=[
                {
                    "gate_id": gate_id, "bank": bank_records[gate_id],
                    "pairs": 500, "games": 1_000,
                    "result_path": str(
                        self.campaign_root
                        / f"dual-final/attempt-000/{gate_id}.result.json"
                    ),
                }
                for gate_id in ("gate-a", "gate-b")
            ],
        )
        self.dual_reference = sealed(
            self.campaign_root / "dual-final/attempt-000/dual-final-reference.json",
            c.DUAL_FINAL_REFERENCE_SCHEMA,
            dual_final_plan=rich(self.dual_plan, c.DUAL_FINAL_SCHEMA),
        )
        exclusion_records = {}
        for gate_index, gate_id in enumerate(("gate-a", "gate-b")):
            fingerprints = [
                hashlib.sha256(f"{gate_id}-{index}".encode()).hexdigest()
                for index in range(500)
            ]
            path = sealed(
                self.execution_root / f"gates/{gate_id}/fingerprint-exclusion.json",
                d.FINGERPRINT_EXCLUSION_SCHEMA,
                gate_id=gate_id,
                classification="protected-final-canonical-fingerprints",
                fingerprint_count=500, fingerprints=fingerprints,
                origin={
                    "candidate_source_sha256": self.candidate["source_sha256"],
                    "candidate_runtime_sha256": self.candidate["runtime_sha256"],
                    "protected_bank_sha256": bank_records[gate_id]["sha256"],
                    "seed_sha256": hashlib.sha256(
                        f"seed-{gate_index}".encode()
                    ).hexdigest(),
                },
                contains_transcripts=False, contains_metrics=False,
                contains_labels=False, training_eligible=False,
                required_for_all_later_development_and_protected_banks=True,
            )
            exclusion_records[gate_id] = q.artifact_reference(
                path, d.FINGERPRINT_EXCLUSION_SCHEMA
            )
        self.prepared = sealed(
            self.execution_root / "prepared.json", d.PREPARED_SCHEMA,
            execution_plan=q.artifact_reference(self.execution, d.PLAN_SCHEMA),
            dual_final_reference=q.artifact_reference(
                self.dual_reference, c.DUAL_FINAL_REFERENCE_SCHEMA
            ),
            fingerprint_exclusions=exclusion_records,
            candidate_unchanged=True, independent_banks=True,
            gate_b_excludes_gate_a=True, games_launched=0,
        )
        ledger = self.execution_root / "gates/gate-a/ledger"
        rosters = {}
        specifications = {
            "claims": ("claims", q.SHARD_CLAIM_SCHEMA, False),
            "receipts": ("receipts", q.SHARD_RECEIPT_SCHEMA, False),
            "raw_evidence": ("raw-evidence", d.RAW_EVIDENCE_SCHEMA, False),
            "raw_gate_results": ("raw", None, True),
        }
        for field, (dirname, schema, is_regular) in specifications.items():
            records = []
            for index in range(100):
                path = ledger / dirname / f"shard-{index:03d}.json"
                if is_regular:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(f"raw-{index}\n".encode())
                    records.append(regular(path))
                else:
                    sealed(path, str(schema), index=index)
                    records.append(q.artifact_reference(path, str(schema)))
            rosters[field] = records
        summary = {"fixture": "must-not-enter-supersession-receipt"}
        aggregate = sealed(
            ledger / "governance-aggregate.json", d.NORMALIZED_AGGREGATE_SCHEMA,
            candidate_source_sha256=self.candidate["source_sha256"],
            candidate_runtime_sha256=self.candidate["runtime_sha256"],
            bank_sha256=bank_records["gate-a"]["sha256"],
            workers=4, threads_per_worker=1, summary=summary,
        )
        bridge_reference = (
            rich(self.dual_plan, c.DUAL_FINAL_SCHEMA)
            if rich_bridge
            else q.artifact_reference(self.dual_plan, c.DUAL_FINAL_SCHEMA)
        )
        self.evidence = sealed(
            ledger / "governance-gate-evidence.json", c.FINAL_GATE_EVIDENCE_SCHEMA,
            bridge_schema=d.DEEP_GATE_EVIDENCE_SCHEMA,
            gate_id="gate-a", status="complete", candidate=self.candidate,
            bank={
                "sha256": bank_records["gate-a"]["sha256"],
                "bytes": bank_records["gate-a"]["bytes"],
            },
            pairs=500, games=1_000, workers=4, threads_per_worker=1,
            shards=100, all_shards_complete=True,
            dual_final_plan=bridge_reference,
            aggregate=regular(aggregate), summary=summary,
            **rosters,
        )


class ProtectedExecutionSupersessionTests(unittest.TestCase):
    def test_create_and_validate_exports_only_hash_exclusions(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary))
            output = fixture.base / "successor-input/supersession.json"
            created = supersession.create_receipt(
                fixture.execution, output=output,
                created_at_utc="2026-09-04T14:25:53Z",
            )
            value = supersession.validate_receipt(
                created, candidate_source=fixture.candidate_source,
                candidate_runtime=fixture.candidate_runtime,
            )
            self.assertEqual(value["fingerprint_count"], 1_000)
            self.assertEqual(len(set(value["fingerprints"])), 1_000)
            self.assertFalse(value["contains_transcripts"])
            self.assertFalse(value["contains_metrics"])
            self.assertFalse(value["contains_labels"])
            self.assertFalse(value["training_eligible"])
            self.assertFalse(value["policy"]["source_result_reuse_authorized"])
            self.assertFalse(value["policy"]["source_qualification_reuse_authorized"])
            self.assertNotIn("summary", value)

    def test_rejects_rich_bridge_or_started_gate_b(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary), rich_bridge=True)
            with self.assertRaisesRegex(
                supersession.SupersessionError, "exact thin-reference"
            ):
                supersession.create_receipt(
                    fixture.execution,
                    output=fixture.base / "successor-input/supersession.json",
                    created_at_utc="2026-09-04T14:25:53Z",
                )

        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary))
            sealed(
                fixture.execution_root / "gates/gate-b/ledger/consumption.json",
                "fixture.consumption.v1",
            )
            with self.assertRaisesRegex(
                supersession.SupersessionError, "Gate B was consumed"
            ):
                supersession.create_receipt(
                    fixture.execution,
                    output=fixture.base / "successor-input/supersession.json",
                    created_at_utc="2026-09-04T14:25:53Z",
                )


if __name__ == "__main__":
    unittest.main()
