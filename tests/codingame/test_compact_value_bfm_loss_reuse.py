import copy
import hashlib
import pathlib
import tempfile
import unittest
from unittest import mock

from tools import compact_value_bfm_loss_reuse as reuse
from tools import compact_value_bfm_qualification as qualification
from tools import compact_value_bfm_rank4_teacher_challenger as challenger
from tools import jacek_replay_corpus as corpus
from tools import jacek_replay_features as features
from tools import jacek_replay_pack as replay_pack


BASE_GAME = "0/0/3/0/61/0/07"
LOSS_GAME_A = (
    "6/7/4/1/2/44/4/7/75/4/3/4/2/42/21/7/0/5/7/25/66/1/200357/06/"
    "4527236436/0530727"
)
LOSS_GAME_B = (
    "2/4/70/2/0/3/657/6/4/7/1/0/3/0/7/46/52/53/22/4/16001/661/31/3/"
    "50/5/256723033/2/0/35/2717/6/674702/27/47574/43646"
)


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload if isinstance(payload, bytes) else payload.encode())
    return path


def record(path):
    path = path.resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def sealed(path, body):
    document = qualification.write_sealed(path, body)
    return {
        **record(path),
        "schema": document["schema"],
        "body_sha256": document["body_sha256"],
    }


def replay_boundaries(transcript):
    state = features.ReplayState()
    result = []
    for action in transcript.split("/"):
        result.append({
            "state": reuse.openings.state_fingerprints(state)["canonical"],
            "feature": corpus.canonical_feature_fingerprint(
                features.encode_active(state)
            ).hex(),
        })
        features.apply_complete_turn(state, state.to_move, action)
    if state.winner != 0:
        raise AssertionError("fixture transcript does not end in a player-zero win")
    return result


def feature_exclusion(path, classification, fingerprints):
    values = sorted(set(fingerprints))
    body = {
        "schema": reuse.FINGERPRINT_SET_SCHEMA,
        "classification": classification,
        "canonicalization": reuse.FOUR_WAY_CANONICALIZATION,
        "fingerprint_domain": reuse.FEATURE_FINGERPRINT_DOMAIN,
        "sources": [],
        "fingerprints": values,
        "fingerprint_count": len(values),
        "source_paths_followed": False,
        "contains_labels": False,
        "contains_metrics": False,
        "contains_transcripts": False,
    }
    qualification.write_sealed(path, body)
    return path


def protected_exclusion(path, fingerprints, *, contains_transcripts=False):
    values = sorted(set(fingerprints))
    qualification.write_sealed(path, {
        "schema": reuse.PROTECTED_FINGERPRINT_SCHEMA,
        "classification": "protected-derived-fixture-canonical-fingerprints",
        "canonicalization": (
            "minimum-sha256-over-exact+rotate+reflect+rotate_reflect"
        ),
        "position_count": len(values),
        "unique_canonical_count": len(values),
        "rows": [
            {"position_id": f"protected:{index}", "canonical_sha256": value}
            for index, value in enumerate(values)
        ],
        "contains_labels": False,
        "contains_metrics": False,
        "contains_transcripts": contains_transcripts,
    })
    return path


def dynamic_protected_exclusion(path):
    values = sorted(
        hashlib.sha256(f"dynamic:{index}".encode()).hexdigest()
        for index in range(500)
    )
    document = qualification.write_sealed(path, {
        "schema": challenger.DYNAMIC_EXCLUSION_SCHEMA,
        "namespace": challenger.NAMESPACE,
        "campaign_id": challenger.CAMPAIGN_ID,
        "attempt": 0,
        "gate_id": "gate-a",
        "classification": "protected-final-canonical-fingerprints",
        "domain": "protected-final-opening-canonical-state",
        "origin": {
            "candidate_source_sha256": "1" * 64,
            "candidate_runtime_sha256": "2" * 64,
            "protected_bank_sha256": "3" * 64,
            "seed_sha256": "4" * 64,
        },
        "canonicalization": (
            "minimum(exact,rotate180,reflect,rotate180-reflect)"
        ),
        "fingerprints": values,
        "fingerprint_count": len(values),
        "contains_transcripts": False,
        "contains_metrics": False,
        "contains_labels": False,
        "training_eligible": False,
        "required_for_all_later_development_and_protected_banks": True,
    })
    return {
        **record(path),
        "schema": challenger.DYNAMIC_EXCLUSION_SCHEMA,
        "body_sha256": document["body_sha256"],
        "classification": "protected-final-canonical-fingerprints",
        "fingerprint_count": len(values),
    }


class Fixture:
    def __init__(self, root, *, phase="pilot"):
        self.root = root
        self.phase = phase
        self.input_directory = root / "campaign/inputs"
        self.ledger = root / "campaign/attempt-ledger/events"
        self.pipeline_root = root / "pipeline"

        self.campaign_plan = root / "campaign/campaign-plan.json"
        qualification.write_sealed(self.campaign_plan, {
            "schema": challenger.PLAN_SCHEMA,
            "campaign_id": challenger.CAMPAIGN_ID,
        })

        self.base_manifest, self.base_tsv = self.make_base_roots()
        self.rows = self.make_game_rows()
        self.games_manifest, self.games = self.make_games()
        self.ab_development = self.make_development_exclusion(
            "search-ab", union=False
        )
        self.development = (
            self.make_development_exclusion("qualification-union", union=True)
            if phase == "full" else self.ab_development
        )
        self.pipeline_plan = self.make_pipeline_plan()
        self.gate_plan, self.gate_result, self.gate_execution = self.make_gate()
        self.full_search_selection = None
        self.full_qualification_plan = None
        self.full_qualification_execution = None
        self.qualification_result = None
        if phase == "full":
            (
                self.full_search_selection,
                self.full_qualification_plan,
                self.full_qualification_execution,
                self.qualification_result,
            ) = self.make_full_qualification()
        self.outcome_receipt = self.make_outcome_receipt()
        self.dynamic_exclusion = dynamic_protected_exclusion(
            root / "exclusions/dynamic-protected.json"
        )
        self.entries = self.make_entries()

        dummy = hashlib.sha256(b"does-not-intersect").hexdigest()
        prior_train = feature_exclusion(
            root / "exclusions/prior-train.json", "prior-train", [dummy]
        )
        prior_validation = feature_exclusion(
            root / "exclusions/prior-validation.json", "prior-validation",
            [hashlib.sha256(b"validation").hexdigest()],
        )
        mixed = feature_exclusion(
            root / "exclusions/mixed.json", "mixed-development",
            [hashlib.sha256(b"mixed").hexdigest()],
        )
        live = feature_exclusion(
            root / "exclusions/live-fingerprints.json",
            "live-diagnostic-canonical-fingerprints",
            [hashlib.sha256(b"live").hexdigest()],
        )
        # LOSS_GAME_B is rejected because one post-branch state occurs in this
        # protected fingerprint-only set.  LOSS_GAME_A remains eligible.
        protected = protected_exclusion(
            root / "exclusions/protected.json",
            [replay_boundaries(LOSS_GAME_B)[7]["state"]],
        )
        self.context = {
            "root": str((root / "campaign").resolve()),
            "plan": {
                "outputs": {
                    "input_directory": str(self.input_directory.resolve()),
                    "ledger": str(self.ledger.resolve()),
                }
            },
            "inputs": {
                "training_inputs": {
                    "prior-train-fingerprints": record(prior_train),
                    "prior-validation-fingerprints": record(prior_validation),
                },
                "protected_exclusions": {
                    "mixed-development-fingerprints": record(mixed),
                    "protected-fingerprints": record(protected),
                },
                "live_exclusions": {
                    "live-fingerprints": record(live),
                },
            },
        }

    def make_base_roots(self):
        accepted = []
        actions = BASE_GAME.split("/")
        for index in range(18):
            group = f"base:{index:02d}"
            accepted.append({
                "game_id": index + 1,
                "group_id": group,
                "root_group_id": group,
                "source": "fixture-base",
                "focus_player": index % 2,
                "winner": 0,
                "opponent_tier": "fixture",
                "turns": [
                    {"player_id": turn % 2, "action": action}
                    for turn, action in enumerate(actions)
                ],
                "source_record_sha256": hashlib.sha256(
                    f"base:{index}".encode()
                ).hexdigest(),
                "split": "train" if index < 14 else (
                    "validation" if index < 16 else "test"
                ),
            })
        manifest = {
            "schema": corpus.ROOT_SCHEMA,
            "feature_schema": features.FEATURE_SCHEMA,
            "tool_sha256": {"normalizer": "1" * 64, "features": "2" * 64},
            "exclusion_boundary": {"read_before_candidate_sources": True},
            "accepted": accepted,
        }
        tsv_payload = replay_pack.teacher_tsv_bytes(manifest)
        tsv = write(self.root / "base/roots.tsv", tsv_payload)
        manifest.update({
            "source_roots": record(tsv),
            "output_sha256": hashlib.sha256(tsv_payload).hexdigest(),
        })
        manifest["body_sha256"] = hashlib.sha256(
            qualification.canonical_json_bytes(manifest)
        ).hexdigest()
        manifest_path = write(
            self.root / "base/roots.json",
            qualification.canonical_json_bytes(manifest),
        )
        return manifest_path, tsv

    def make_game_rows(self):
        reflected_a = LOSS_GAME_A.translate(reuse.REFLECT)
        return [
            {
                "game_ordinal": 0,
                "game_id": "loss-a",
                "actor_mode": "student-p2-vs-rank4",
                "base_seed": 1,
                "root_group_id": "base:00",
                "source": challenger.CAMPAIGN_ID,
                "winner": 0,
                "transcript": LOSS_GAME_A,
                "prefix_turns": 5,
            },
            {
                "game_ordinal": 1,
                "game_id": "loss-a-reflected",
                "actor_mode": "student-p2-vs-rank4",
                "base_seed": 2,
                "root_group_id": "base:01",
                "source": challenger.CAMPAIGN_ID,
                "winner": 0,
                "transcript": reflected_a,
                "prefix_turns": 5,
            },
            {
                "game_ordinal": 2,
                "game_id": "loss-b-protected-overlap",
                "actor_mode": "student-p2-vs-rank4",
                "base_seed": 3,
                "root_group_id": "base:02",
                "source": challenger.CAMPAIGN_ID,
                "winner": 0,
                "transcript": LOSS_GAME_B,
                "prefix_turns": 5,
            },
            {
                "game_ordinal": 3,
                "game_id": "student-win",
                "actor_mode": "student-p1-vs-rank4",
                "base_seed": 4,
                "root_group_id": "base:03",
                "source": challenger.CAMPAIGN_ID,
                "winner": 0,
                "transcript": LOSS_GAME_A,
                "prefix_turns": 5,
            },
            {
                "game_ordinal": 4,
                "game_id": "selfplay",
                "actor_mode": "student-selfplay",
                "base_seed": 5,
                "root_group_id": "base:04",
                "source": challenger.CAMPAIGN_ID,
                "winner": 0,
                "transcript": LOSS_GAME_A,
                "prefix_turns": 5,
            },
        ]

    def make_games(self):
        lines = ["group_id\tsource\twinner\ttranscript"]
        lines.extend(
            f"{row['root_group_id']}\t{row['source']}\t{row['winner']}\t"
            f"{row['transcript']}" for row in self.rows
        )
        payload = ("\n".join(lines) + "\n").encode()
        games = write(self.pipeline_root / "games/games.tsv", payload)
        manifest = self.pipeline_root / "games/games.manifest.json"
        self.games_payload_sha = hashlib.sha256(payload).hexdigest()
        return manifest, games

    def make_development_exclusion(self, name, *, union):
        values = sorted({
            boundary["state"]
            for row in self.rows
            for boundary in replay_boundaries(row["transcript"])
        })
        if union:
            values.append(hashlib.sha256(b"qualification-union").hexdigest())
            values.sort()
        path = self.root / f"phase/{name}-development.json"
        qualification.write_sealed(path, {
            "schema": challenger.DEVELOPMENT_EXCLUSION_SCHEMA,
            "campaign_id": challenger.CAMPAIGN_ID,
            "attempt": 1,
            "phase": self.phase,
            "classification": "unprotected-development-fingerprints",
            "canonicalization": (
                "minimum-sha256-over-exact+rotate+reflect+rotate-reflect"
            ),
            "fingerprints": values,
            "fingerprint_count": len(values),
            "includes_search_ab_bank": self.phase == "full" and union,
            "includes_post_selection_qualification_bank": (
                self.phase == "full" and union
            ),
            "protected_or_live_data_included": False,
        })
        return path

    def make_pipeline_plan(self):
        path = self.pipeline_root / "pipeline-plan.json"
        document = qualification.write_sealed(path, {
            "schema": reuse.PIPELINE_SCHEMA,
            "campaign_id": challenger.CAMPAIGN_ID,
            "attempt": 1,
            "phase": self.phase,
            "outputs": {
                "games": str(self.games.resolve()),
                "games_manifest": str(self.games_manifest.resolve()),
            },
        })
        qualification.write_sealed(self.games_manifest, {
            "schema": reuse.GAME_MANIFEST_SCHEMA,
            "pipeline_body_sha256": document["body_sha256"],
            "attempt": 1,
            "phase": self.phase,
            "games": len(self.rows),
            "shards": [],
            "rows": self.rows,
            "games_sha256": self.games_payload_sha,
        })
        return path

    def make_gate(self):
        source = write(self.root / "gate/candidate.cpp", "int main(){}\n")
        bank = write(self.root / "gate/bank.json", "unprotected fixture\n")
        bank_tsv = write(self.root / "gate/bank.tsv", "opening fixture\n")
        result = write(self.root / "gate/result.json", "{}\n")
        request = self.root / "gate/request.json"
        bank_record = {
            "manifest": record(bank),
            "gate_tsv": record(bank_tsv),
            "classification": "fresh-unprotected",
        }
        request_record = sealed(request, {
            "schema": reuse.SCREEN_REQUEST_SCHEMA,
            "gate_purpose": (
                "pilot-screen" if self.phase == "pilot" else "full-search-ab"
            ),
            "search_variant": "baseline",
            "search_variant_metadata": {
                "candidate_search_profile": "standard-v1",
            },
            "candidate_source": record(source),
            "bank": bank_record,
            "protected_tests_opened": False,
        })
        gate = self.root / "gate/gate-plan.json"
        sealed(gate, {
            "schema": reuse.GATE_PLAN_SCHEMA,
            "campaign_id": challenger.CAMPAIGN_ID,
            "attempt": 1,
            "phase": self.phase,
            "bank": bank_record,
            "development_exclusion": {
                **record(self.ab_development),
                "body_sha256": qualification.load_sealed(
                    self.ab_development
                )["body_sha256"],
            },
            "active_search_variant_roster": ["baseline"],
            "requests": [{
                "variant": "baseline",
                "request": request_record,
                "request_body_sha256": request_record["body_sha256"],
            }],
            "protected_tests_opened": False,
        })
        self.selected = {
            "search_variant": "baseline",
            "source": record(source),
        }
        execution = self.make_gate_execution(
            gate, {"baseline": request_record}, {"baseline": result},
            name="search-ab",
        )
        return gate, result, execution

    def make_gate_execution(self, gate, requests, results, *, name):
        gate_document = qualification.load_sealed(gate)
        receipt_records = {}
        clock_base = 0 if name == "search-ab" else 5
        for variant, request_record in requests.items():
            request = qualification.load_sealed(
                pathlib.Path(request_record["path"])
            )
            claim = self.root / f"gate/{name}-{variant}-claim.json"
            claim_record = sealed(claim, {
                "schema": reuse.GATE_EXECUTION_CLAIM_SCHEMA,
                "campaign_id": challenger.CAMPAIGN_ID,
                "attempt": 1,
                "phase": self.phase,
                "gate_plan": record(gate),
                "gate_plan_body_sha256": gate_document["body_sha256"],
                "variant": variant,
                "request": record(pathlib.Path(request_record["path"])),
                "request_body_sha256": request["body_sha256"],
                "claimed_at_utc": f"2026-09-04T00:00:{clock_base + 1:02d}Z",
                "worker": {
                    "workers": 1,
                    "threads_per_worker": 1,
                    "whole_bank_process": True,
                    "process_nice": 0,
                    "thread_environment": {"OMP_NUM_THREADS": "1"},
                },
                "prelaunch_audit": {
                    "process_nice": 0,
                    "competing_rank4_gate_processes": [],
                },
                "no_retry": True,
            })
            receipt = self.root / f"gate/{name}-{variant}-receipt.json"
            receipt_records[variant] = sealed(receipt, {
                "schema": reuse.GATE_VARIANT_EXECUTION_SCHEMA,
                "campaign_id": challenger.CAMPAIGN_ID,
                "attempt": 1,
                "phase": self.phase,
                "gate_plan_body_sha256": gate_document["body_sha256"],
                "variant": variant,
                "claim": claim_record,
                "request": record(pathlib.Path(request_record["path"])),
                "raw_result": record(results[variant]),
                "profile_activation": {
                    "schema": (
                        reuse.rank4_gate_support.SEARCH_PROFILE_ACTIVATION_SCHEMA
                    ),
                    "candidate_search_profile": request[
                        "search_variant_metadata"
                    ]["candidate_search_profile"],
                    "exercised": True,
                },
                "execution": {
                    "launched_at_utc": (
                        f"2026-09-04T00:00:{clock_base + 2:02d}Z"
                    ),
                    "finished_at_utc": (
                        f"2026-09-04T00:00:{clock_base + 3:02d}Z"
                    ),
                    "workers": 1,
                    "threads_per_worker": 1,
                    "whole_bank_process": True,
                    "variants_serial": True,
                    "process_nice": 0,
                },
                "status": "complete-no-retry",
                "retry_authorized": False,
            })
        execution = self.root / f"gate/{name}-execution.json"
        sealed(execution, {
            "schema": reuse.GATE_EXECUTION_SCHEMA,
            "campaign_id": challenger.CAMPAIGN_ID,
            "attempt": 1,
            "phase": self.phase,
            "gate_plan": record(gate),
            "gate_plan_body_sha256": gate_document["body_sha256"],
            "variant_order": list(requests),
            "variant_receipts": receipt_records,
            "status": "complete-serial-one-worker-no-retry",
            "retry_authorized": False,
        })
        return execution

    def make_full_qualification(self):
        selection = self.root / "full/full-search-selection.json"
        sealed(selection, {
            "schema": reuse.FULL_SEARCH_SELECTION_SCHEMA,
            "campaign_id": challenger.CAMPAIGN_ID,
            "attempt": 1,
            "phase": "full",
            "selected_at_utc": "2026-09-04T00:00:04Z",
            "search_ab_gate_plan": sealed_record(self.gate_plan),
            "search_ab_execution": sealed_record(self.gate_execution),
            "search_ab_results": {"baseline": record(self.gate_result)},
            "selected_candidate": self.selected,
            "selected_before_qualification_bank_read": True,
            "qualification_bank_read": False,
            "protected_tests_opened": False,
        })

        bank = write(self.root / "full/qualification-bank.json", "qualified\n")
        bank_tsv = write(self.root / "full/qualification-bank.tsv", "qualified\n")
        result = write(
            self.root / "full/qualification-result.json",
            '{"qualification":true}\n',
        )
        bank_record = {
            "manifest": record(bank),
            "gate_tsv": record(bank_tsv),
            "classification": "fresh-unprotected",
        }
        request = self.root / "full/qualification-request.json"
        request_record = sealed(request, {
            "schema": reuse.SCREEN_REQUEST_SCHEMA,
            "gate_purpose": "full-qualification",
            "full_search_selection": sealed_record(selection),
            "search_variant": "baseline",
            "search_variant_metadata": {
                "candidate_search_profile": "standard-v1",
            },
            "candidate_source": self.selected["source"],
            "bank": bank_record,
            "protected_tests_opened": False,
        })
        plan = self.root / "full/full-qualification-plan.json"
        sealed(plan, {
            "schema": reuse.FULL_QUALIFICATION_PLAN_SCHEMA,
            "campaign_id": challenger.CAMPAIGN_ID,
            "attempt": 1,
            "phase": "full",
            "prepared_at_utc": "2026-09-04T00:00:05Z",
            "full_search_selection": sealed_record(selection),
            "selected_candidate": self.selected,
            "qualification_bank_opened_after_selection": True,
            "search_ab_bank": qualification.load_sealed(self.gate_plan)["bank"],
            "bank": bank_record,
            "bank_disjointness": {"passed": True},
            "development_exclusion": sealed_record(self.development),
            "active_search_variant_roster": ["baseline"],
            "requests": [{
                "variant": "baseline",
                "request": request_record,
                "request_body_sha256": request_record["body_sha256"],
            }],
            "protected_tests_opened": False,
        })
        execution = self.make_gate_execution(
            plan, {"baseline": request_record}, {"baseline": result},
            name="qualification",
        )
        return selection, plan, execution, result

    def make_outcome_receipt(self):
        path = self.root / "phase/outcome.json"
        full_search_selection = (
            None if self.full_search_selection is None
            else sealed_record(self.full_search_selection)
        )
        full_qualification_plan = (
            None if self.full_qualification_plan is None
            else sealed_record(self.full_qualification_plan)
        )
        full_qualification_execution = (
            None if self.full_qualification_execution is None
            else sealed_record(self.full_qualification_execution)
        )
        qualification_result = (
            None if self.qualification_result is None
            else record(self.qualification_result)
        )
        closure = {
            "pipeline_plan": sealed_record(self.pipeline_plan),
            "gate_plan": sealed_record(self.gate_plan),
            "gate_results": {"baseline": record(self.gate_result)},
            "gate_execution": sealed_record(self.gate_execution),
            "full_search_selection": full_search_selection,
            "full_qualification_plan": full_qualification_plan,
            "full_qualification_execution": full_qualification_execution,
            "qualification_result": qualification_result,
            "selected_candidate": self.selected,
            "protected_tests_opened": False,
        }
        sealed(path, {
            "schema": challenger.PHASE_OUTCOME_EVIDENCE_SCHEMA,
            "campaign_id": challenger.CAMPAIGN_ID,
            "attempt": 1,
            "phase": self.phase,
            "status": "complete",
            "protected_or_live_metrics_read": False,
            "all_games_finished": True,
            "development_exclusion": sealed_record(self.development),
            "gate_execution": closure["gate_execution"],
            "full_search_selection": full_search_selection,
            "full_qualification_plan": full_qualification_plan,
            "full_qualification_execution": full_qualification_execution,
            "qualification_result": qualification_result,
            "evidence_closure": closure,
        })
        return path

    def make_entries(self):
        opened = {
            "event": "attempt-opened",
            "attempt": 1,
            "attempt_inputs": {
                "roots_tsv": record(self.base_tsv),
                "roots_manifest": record(self.base_manifest),
            },
            "dynamic_exclusions": [self.dynamic_exclusion],
        }
        body = {
            "schema": challenger.LEDGER_SCHEMA,
            "sequence": 1,
            "event": "attempt-outcome-recorded",
            "attempt": 1,
            "phase": self.phase,
            "admitted": False,
            "adaptation_route": "open-next-attempt-same-contract",
            "outcome_receipt": sealed_record(self.outcome_receipt),
            "development_exclusion": sealed_record(self.development),
        }
        event_path = self.ledger / "placeholder.json"
        document = qualification.seal(body)
        event_path = self.ledger / f"000001-{document['body_sha256']}.json"
        write(event_path, qualification.canonical_json_bytes(document))
        return [opened, document]


def sealed_record(path):
    value = qualification.load_sealed(path)
    return {
        **record(path),
        "schema": value["schema"],
        "body_sha256": value["body_sha256"],
    }


class LossReuseTest(unittest.TestCase):
    def run_materialize(self, fixture, output=None):
        with (
            mock.patch.object(
                reuse.challenger, "validate_campaign",
                return_value=fixture.context,
            ),
            mock.patch.object(
                reuse.challenger, "load_ledger",
                return_value=fixture.entries,
            ),
            mock.patch.object(
                reuse.openings, "validate_bank",
                return_value={"classification": "unprotected-development"},
            ),
            mock.patch.object(
                reuse.rank4_gate_support, "validate_result",
                return_value={
                    "config": {
                        "mode": "actual-clock",
                        "candidate_search_profile": "standard-v1",
                    }
                },
            ),
            mock.patch.object(
                reuse.rank4_gate_support,
                "require_search_profile_exercised",
                return_value={
                    "schema": (
                        reuse.rank4_gate_support.SEARCH_PROFILE_ACTIVATION_SCHEMA
                    ),
                    "candidate_search_profile": "standard-v1",
                    "exercised": True,
                },
            ),
        ):
            return reuse.materialize_loss_reuse(
                campaign_plan=fixture.campaign_plan,
                attempt=1,
                phase=fixture.phase,
                output_directory=output or fixture.root / "reuse",
            )

    def test_materializes_only_canonical_nonintersecting_student_losses(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary))
            outputs = self.run_materialize(fixture)
            manifest = reuse.validate_loss_reuse_manifest(
                outputs["roots_manifest"]
            )
            self.assertEqual(manifest["reuse_schema"], reuse.REUSE_SCHEMA)
            self.assertEqual(manifest["counts"]["new_loss_roots"], 1)
            self.assertEqual(
                manifest["counts"]["game_classification"],
                {
                    "accepted_loss_trajectories": 1,
                    "excluded_intersecting_trajectories": 1,
                    "non_student_rank4": 1,
                    "student_losses": 3,
                    "student_wins": 1,
                    "symmetry_duplicate_trajectories": 1,
                },
            )
            new = [
                row for row in manifest["accepted"]
                if row["source"] == reuse.SOURCE
            ]
            self.assertEqual(new[0]["loss_reuse"]["game_id"], "loss-a")
            self.assertEqual(
                new[0]["loss_reuse"]["protected_or_live_position_intersections"],
                0,
            )
            self.assertFalse(
                manifest["exclusion_boundary"]["protected_or_live_metrics_used"]
            )
            exclusion_roles = set(
                manifest["exclusion_boundary"]["sources"]
            )
            self.assertIn("dynamic:0000", exclusion_roles)
            self.assertIn("frozen-live:live-fingerprints", exclusion_roles)
            self.assertIn(
                "frozen-protected:protected-fingerprints", exclusion_roles
            )
            self.assertEqual(
                set(replay_pack.frozen_assignments(manifest).values()),
                {"train", "validation", "test"},
            )
            self.assertEqual(
                replay_pack.teacher_tsv_bytes(manifest),
                outputs["roots_tsv"].read_bytes(),
            )
            self.assertEqual(
                self.run_materialize(fixture), outputs,
                "a resume must resolve to the same content-addressed artifacts",
            )

    def test_full_reuse_binds_qualifier_and_preserves_search_ab_ancestry(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary), phase="full")
            outputs = self.run_materialize(fixture)
            manifest = reuse.validate_loss_reuse_manifest(
                outputs["roots_manifest"]
            )
            source = next(
                item for item in manifest["sources"]
                if item["kind"] == "rejected-unprotected-attempt"
            )
            self.assertEqual(source["phase"], "full")
            self.assertEqual(
                source["search_ab_gate_plan"]["schema"],
                reuse.GATE_PLAN_SCHEMA,
            )
            self.assertEqual(
                source["selected_gate_plan"]["schema"],
                reuse.FULL_QUALIFICATION_PLAN_SCHEMA,
            )
            self.assertEqual(
                source["selected_gate_plan"],
                source["full_qualification_plan"],
            )
            self.assertEqual(
                source["selected_gate_result"], source["qualification_result"]
            )
            self.assertNotEqual(
                source["selected_gate_result"]["sha256"],
                source["search_ab_results"]["baseline"]["sha256"],
            )
            self.assertEqual(
                source["search_ab_results"]["baseline"]["sha256"],
                record(fixture.gate_result)["sha256"],
            )
            development = manifest["exclusion_boundary"][
                "source_attempt_development_exclusion"
            ]
            self.assertEqual(development["sha256"], record(fixture.development)["sha256"])
            self.assertNotEqual(
                development["sha256"], record(fixture.ab_development)["sha256"]
            )
            self.assertEqual(
                self.run_materialize(fixture), outputs,
                "full qualifier ancestry must resume byte-for-byte",
            )

    def test_campaign_validation_rederives_and_rejects_resealed_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary))
            outputs = self.run_materialize(fixture)
            activation = {
                "schema": (
                    reuse.rank4_gate_support.SEARCH_PROFILE_ACTIVATION_SCHEMA
                ),
                "candidate_search_profile": "standard-v1",
                "exercised": True,
            }
            def validation_patches():
                return (
                    mock.patch.object(
                    reuse.challenger, "validate_campaign",
                    return_value=fixture.context,
                    ),
                    mock.patch.object(
                    reuse.challenger, "load_ledger",
                    return_value=fixture.entries,
                    ),
                    mock.patch.object(
                    reuse.openings, "validate_bank",
                    return_value={"classification": "unprotected-development"},
                    ),
                    mock.patch.object(
                    reuse.rank4_gate_support, "validate_result",
                    return_value={
                        "config": {
                            "mode": "actual-clock",
                            "candidate_search_profile": "standard-v1",
                        }
                    },
                    ),
                    mock.patch.object(
                    reuse.rank4_gate_support,
                    "require_search_profile_exercised",
                    return_value=activation,
                    ),
                )
            patches = validation_patches()
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                validated = reuse.validate_loss_reuse_for_campaign(
                    outputs["roots_manifest"],
                    campaign_plan=fixture.campaign_plan,
                    expected_source_attempt=1,
                    expected_source_phase="pilot",
                )
            self.assertEqual(validated["reuse_schema"], reuse.REUSE_SCHEMA)
            forged = copy.deepcopy(validated)
            forged.pop("body_sha256")
            source = next(
                item for item in forged["sources"]
                if item["kind"] == "rejected-unprotected-attempt"
            )
            source["loss_selection"] = "forged-loss-selection-policy"
            forged["body_sha256"] = hashlib.sha256(
                qualification.canonical_json_bytes(forged)
            ).hexdigest()
            payload = qualification.canonical_json_bytes(forged)
            forged_path = outputs["roots_manifest"].parent / (
                hashlib.sha256(payload).hexdigest()
                + ".loss-reuse-roots.json"
            )
            forged_path.write_bytes(payload)
            reuse.validate_loss_reuse_manifest(forged_path)
            patches = validation_patches()
            with patches[0], patches[1], patches[2], patches[3], patches[4], \
                    self.assertRaisesRegex(
                        reuse.LossReuseError, "does not rederive"
                    ):
                reuse.validate_loss_reuse_for_campaign(
                    forged_path, campaign_plan=fixture.campaign_plan,
                    expected_source_attempt=1,
                    expected_source_phase="pilot",
                )

    def test_archived_validation_survives_removed_staging_outputs(self):
        for phase in ("pilot", "full"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as temporary:
                fixture = Fixture(pathlib.Path(temporary), phase=phase)
                outputs = self.run_materialize(fixture)
                self.assertEqual(self.run_materialize(fixture), outputs)
                archived = fixture.root / "campaign-local-reuse"
                archived_manifest = write(
                    archived / outputs["roots_manifest"].name,
                    outputs["roots_manifest"].read_bytes(),
                )
                archived_tsv = write(
                    archived / outputs["roots_tsv"].name,
                    outputs["roots_tsv"].read_bytes(),
                )

                outputs["roots_manifest"].unlink()
                outputs["roots_tsv"].unlink()

                manifest = reuse.validate_archived_loss_reuse_manifest(
                    archived_manifest, roots_tsv=archived_tsv,
                )
                self.assertEqual(manifest["reuse_schema"], reuse.REUSE_SCHEMA)
                self.assertEqual(manifest["sources"][1]["phase"], phase)
                self.assertEqual(
                    replay_pack.teacher_tsv_bytes(manifest),
                    archived_tsv.read_bytes(),
                )
                archived_tsv.write_bytes(
                    archived_tsv.read_bytes() + b"tampered\n"
                )
                with self.assertRaisesRegex(
                    reuse.LossReuseError,
                    "archived loss reuse roots TSV binding",
                ):
                    reuse.validate_archived_loss_reuse_manifest(
                        archived_manifest, roots_tsv=archived_tsv,
                    )

    def test_rejects_non_fingerprint_live_input_before_opening_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary))
            forbidden = fixture.root / "must-not-open-live.json"
            fixture.context["inputs"]["live_exclusions"] = {
                "raw-live-replay": {
                    "path": str(forbidden), "bytes": 10, "sha256": "f" * 64,
                }
            }
            with self.assertRaisesRegex(
                reuse.LossReuseError, "no fingerprint-only projection"
            ):
                self.run_materialize(fixture)
            self.assertFalse(forbidden.exists())

    def test_rejects_protected_fingerprint_file_with_transcripts(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary))
            bad = protected_exclusion(
                fixture.root / "exclusions/bad-protected.json",
                [hashlib.sha256(b"protected").hexdigest()],
                contains_transcripts=True,
            )
            fixture.context["inputs"]["protected_exclusions"][
                "protected-fingerprints"
            ] = record(bad)
            with self.assertRaisesRegex(
                reuse.LossReuseError, "protected fingerprint contract"
            ):
                self.run_materialize(fixture)

    def test_requires_rejected_current_terminal_outcome(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary))
            admitted = copy.deepcopy(fixture.entries[-1])
            admitted["admitted"] = True
            fixture.entries[-1] = admitted
            with self.assertRaisesRegex(
                reuse.LossReuseError, "rejected terminal phase"
            ):
                self.run_materialize(fixture)

    def test_horizontal_reflections_share_canonical_trajectory(self):
        exact = reuse._canonical_trajectory(
            LOSS_GAME_A, winner=0, focus_player=1
        )
        reflected = reuse._canonical_trajectory(
            LOSS_GAME_A.translate(reuse.REFLECT), winner=0, focus_player=1
        )
        self.assertEqual(exact["identity"], reflected["identity"])
        self.assertEqual(exact["transcript"], reflected["transcript"])


if __name__ == "__main__":
    unittest.main()
