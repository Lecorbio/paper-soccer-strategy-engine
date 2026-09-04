import copy
import importlib.util
import json
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULE = ROOT / "benchmarks/compact_value_bfm/publish.py"
SPEC = importlib.util.spec_from_file_location("compact_value_bfm_publish", MODULE)
publish = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(publish)


COMMIT = "a" * 40


def write_json(path, value, *, sealed=True):
    if sealed:
        value = publish.body_hashed(value)
    path.write_bytes(publish.canonical_json_bytes(value))
    return path


def runtime(path):
    return write_json(path, {
        "schema": "papersoccer.compact-value-bfm-runtime.v1",
        "feature_schema": "feature",
        "architecture": {"name": "compact-8x8", "dimensions": [6301, 8, 8, 1]},
        "selection": {"arm": "search-target", "seed": 20260907},
        "quantization": {"payload_sha256": "b" * 64, "payload_base64": "AAAA"},
    })


def family(path):
    return write_json(path, {
        "schema": "papersoccer.compact-value-bfm-selection.v1",
        "namespace": publish.NAMESPACE,
        "architecture": "compact-8x8",
        "arm": "search-target",
        "seed": 20260907,
        "status": "offline-evaluator-qualified-not-game-gated",
        "deployment_eligible": True,
        "offline_gate": {
            "passed": True,
            "common_sign_accuracy": .85,
            "common_weighted_huber": .05,
            "canonical_sign_accuracy": .87,
            "canonical_weighted_huber": .05,
        },
    })


def development(path):
    # Raw opening transcripts may exist in the private receipt, but the publisher
    # must select only compact aggregates and never copy them to public evidence.
    return write_json(path, {
        "schema": "papersoccer.compact-value-bfm.development-input.v1",
        "namespace": publish.NAMESPACE,
        "eligible_architectures": ["6301-8-8-1", "6301-8-16-1"],
        "banks": {"private": {"transcripts": ["0/1/2/3/4/5/6/7/0/1/2/3"]}},
        "actual_clock": {
            "candidate_id": "primary-search:c0.95-f0.5-l1:default",
            "pairs": 200, "games": 400, "wins": 211,
            "color_wins": {"0": 104, "1": 107},
            "failures": 0, "latency_ms": 155.0,
        },
    })


def protected(path):
    failures = {
        name: 0 for name in (
            "illegal", "unfinished", "timeout", "crash", "malformed", "over320"
        )
    }
    return write_json(path, {
        "schema": "papersoccer.compact-value-bfm.final-aggregate.v1",
        "namespace": publish.NAMESPACE,
        "status": "rank4-qualified",
        "summary": {
            "games": 1000,
            "candidate_wins": 527,
            "candidate_color_wins": {"0": 260, "1": 267},
            "failures": failures,
            "maximum_turns": 320,
            "timing": {"first_max_ms": 999.0, "later_max_ms": 199.0},
            "uncontended_timing": {"first_max_ms": 899.0, "later_max_ms": 179.0},
        },
        "verdict": {"passed": True},
        "rank4_replacement_authorized": False,
    })


def preflight(path):
    return write_json(path, {
        "schema": "papersoccer.compact-value-bfm.preflight-receipt.v1",
        "namespace": publish.NAMESPACE,
        "status": "passed",
        "checks": {"gcc": "passed", "clang": "passed", "sanitizers": "passed"},
    })


def ci(path):
    return write_json(path, {
        "run_id": 123,
        "head_sha": COMMIT,
        "head_branch": "compact-value-bfm",
        "conclusion": "success",
        "jobs": {"test-gcc": "success", "test-clang": "success"},
    }, sealed=False)


def upload(path, source_sha, source_bytes):
    return write_json(path, {
        "schema": "papersoccer.compact-value-bfm.upload-event.v1",
        "namespace": publish.NAMESPACE,
        "status": "submission-attested",
        "submitted_at_utc": "2026-08-31T15:04:00Z",
        "candidate_commit": COMMIT,
        "source_sha256": source_sha,
        "source_bytes": source_bytes,
        "agent_id": 701,
        "submission_id": 801,
        "submit_clicks": 1,
    })


def live(path, source_sha, own_failures=0, games=90):
    return write_json(path, {
        "schema": "papersoccer.compact-value-bfm.live-window.v1",
        "namespace": publish.NAMESPACE,
        "status": "complete-accepted" if own_failures == 0 else
                  "complete-rejected-own-failure",
        "exact_games": games,
        "identity": {
            "agent_id": 701, "submission_id": 801,
            "source_sha256": source_sha, "repository_commit": COMMIT,
            "game_ids": list(range(games)),
        },
        "focus_operational_failures": own_failures,
        "opponent_operational_failures": 7,
        "opponent_failures_count_as_strength_wins": False,
        "training_eligible": False,
    })


class Fixture:
    def __init__(self, root):
        self.root = root
        self.source = root / "selected.cpp"
        self.source.write_text("int main(){return 0;}\n", encoding="ascii")
        source_bytes = self.source.read_bytes()
        source_sha = publish.sha256_bytes(source_bytes)
        self.paths = {
            "runtime": runtime(root / "selected.runtime.json"),
            "source": self.source,
            "family": family(root / "family.json"),
            "development": development(root / "development.json"),
            "protected": protected(root / "protected.json"),
            "preflight": preflight(root / "preflight.json"),
            "ci": ci(root / "ci.json"),
            "upload": upload(root / "upload.json", source_sha, len(source_bytes)),
            "live": live(root / "live.json", source_sha),
        }


class CompactPublisherTests(unittest.TestCase):
    def test_incomplete_publication_is_explicit_and_contains_no_runtime_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source = root / "selected.cpp"
            source.write_text("int main(){}\n", encoding="ascii")
            paths = {role: None for role in publish.REQUIRED_COMPLETE}
            paths.update({
                "runtime": runtime(root / "selected.runtime.json"),
                "source": source,
            })
            evidence = publish.build_evidence(paths)
            self.assertEqual(evidence["status"], "incomplete")
            self.assertFalse(evidence["claims"]["publication_complete"])
            rendered = json.dumps(evidence, sort_keys=True)
            self.assertNotIn("payload_base64", rendered)
            self.assertNotIn("AAAA", rendered)
            self.assertNotIn(str(root), rendered)
            report = publish.render_report(evidence)
            self.assertIn("Status: **incomplete**", report)
            publish.verify_report(report, evidence)

    def test_complete_publication_verifies_exact_one_upload_and_exact_90(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            fixture = Fixture(root)
            evidence = publish.build_evidence(fixture.paths)
            self.assertEqual(evidence["status"], "complete")
            self.assertEqual(evidence["claims"]["uploads"], 1)
            self.assertEqual(evidence["claims"]["live_games"], 90)
            self.assertTrue(evidence["claims"]["rank4_qualified"])
            self.assertFalse(evidence["claims"]["rank4_replaced"])
            output = root / "public" / "compact.json"
            report = root / "public" / "REPORT.md"
            args = type("Args", (), {
                **fixture.paths, "output": output, "report": report,
            })()
            published = publish.publish(args)
            self.assertEqual(published, evidence)
            verify_args = type("Args", (), {"evidence": output, "report": report})()
            publish.verify(verify_args)
            public_text = output.read_text()
            for forbidden in ('"transcripts":', '"game_ids":',
                              "payload_base64", "0/1/2/3"):
                self.assertNotIn(forbidden, public_text)

    def test_raw_inputs_and_forbidden_promotion_claims_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            raw = root / "openings.tsv"
            raw.write_text("opening_id\ttranscript\n")
            paths = {role: None for role in publish.REQUIRED_COMPLETE}
            paths["runtime"] = raw
            with self.assertRaisesRegex(publish.PublicationError, "forbidden raw"):
                publish.build_evidence(paths)

            fixture = Fixture(root)
            document = json.loads(fixture.paths["protected"].read_bytes())
            document["rank4_replacement_authorized"] = True
            document.pop("body_sha256")
            fixture.paths["protected"] = write_json(
                root / "protected-claim.json", document)
            with self.assertRaisesRegex(publish.PublicationError, "forbidden promotion"):
                publish.build_evidence(fixture.paths)

    def test_89_or_91_games_and_multiple_uploads_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            fixture = Fixture(root)
            source_sha = publish.sha256_bytes(fixture.source.read_bytes())
            for games in (89, 91):
                with self.subTest(games=games):
                    changed = dict(fixture.paths)
                    changed["live"] = live(root / f"live-{games}.json", source_sha,
                                           games=games)
                    with self.assertRaisesRegex(publish.PublicationError, "exactly 90"):
                        publish.build_evidence(changed)
            changed = dict(fixture.paths)
            upload_document = json.loads(changed["upload"].read_bytes())
            upload_document.pop("body_sha256")
            upload_document["submit_clicks"] = 2
            changed["upload"] = write_json(root / "upload-twice.json", upload_document)
            with self.assertRaisesRegex(publish.PublicationError, "more than one upload"):
                publish.build_evidence(changed)

    def test_verifier_rejects_tampering_and_report_claim_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(pathlib.Path(directory))
            evidence = publish.build_evidence(fixture.paths)
            tampered = copy.deepcopy(evidence)
            tampered["claims"]["rank1_claimed"] = True
            tampered.pop("body_sha256")
            tampered = publish.body_hashed(tampered)
            with self.assertRaisesRegex(publish.PublicationError, "forbidden Rank-1"):
                publish.verify_evidence(tampered)
            leaked = copy.deepcopy(evidence)
            leaked["metrics"]["private"] = "0/1/2/3"
            leaked.pop("body_sha256")
            leaked = publish.body_hashed(leaked)
            with self.assertRaisesRegex(publish.PublicationError, "raw turn transcript"):
                publish.verify_evidence(leaked)
            with self.assertRaisesRegex(publish.PublicationError, "omits"):
                publish.verify_report("Status: **complete**.\n", evidence)


if __name__ == "__main__":
    unittest.main()
