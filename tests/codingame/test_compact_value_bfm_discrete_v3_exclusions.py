import copy
import hashlib
import importlib.util
import json
import pathlib
import shutil
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools/compact_value_bfm_discrete_v3_exclusions.py"
SPEC = importlib.util.spec_from_file_location(
    "compact_value_bfm_discrete_v3_exclusions_tested", TOOL
)
exclusions = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(exclusions)
q = exclusions.qualification
opening_tools = exclusions.opening_tools
selfsearch = exclusions.selfsearch


FULL_GAME = (
    "2/4/70/2/0/3/657/6/4/7/1/0/3/0/7/46/52/53/22/4/16001/"
    "661/31/3/50/5/256723033/2/0/35/2717/6/674702/27/47574/43646"
)
PREFIX_TURNS = 9


def accept_synthetic_bank_identity(_paths, _records):
    return None


def sampled_position_openings():
    actions = FULL_GAME.split("/")
    boundaries = selfsearch._sample_suffix_boundaries(
        len(actions), PREFIX_TURNS, exclusions.POSITIONS_PER_GAME
    )
    if len(boundaries) != exclusions.POSITIONS_PER_GAME:
        raise AssertionError("synthetic game does not provide exactly 20 positions")
    result = []
    for boundary in boundaries:
        transcript = "/".join(actions[:boundary])
        state, primitive_plies = opening_tools.replay_transcript(transcript)
        result.append({
            "transcript": transcript,
            "primitive_plies": primitive_plies,
            "completed_turn_overshoot": (
                primitive_plies - opening_tools.MINIMUM_PHYSICAL_PLIES
            ),
            "ball": list(state.ball),
            "to_move": state.to_move,
            "fingerprints": opening_tools.state_fingerprints(state),
        })
    return result


def synthetic_development_banks(root, *, overlap):
    protected_openings = sampled_position_openings()
    protected_variants = {
        value
        for opening in protected_openings
        for name, value in opening["fingerprints"].items()
        if name != "canonical"
    }
    forced = []
    if overlap:
        for transform in exclusions.SYMMETRY_NAMES:
            match = next(
                opening for opening in protected_openings
                if opening["fingerprints"]["canonical"]
                == opening["fingerprints"][transform]
            )
            forced.append(copy.deepcopy(match))
        if len({item["fingerprints"]["canonical"] for item in forced}) != 4:
            raise AssertionError("four-transform fixture is not canonical-distinct")

    result = {}
    seen = set(protected_variants)
    exclusion_header = {
        "sources": [],
        "fingerprints": [],
        "body_sha256": q.sha256_bytes(q.canonical_json_bytes({
            "sources": [], "fingerprints": [],
        })),
    }
    for stage, count in exclusions.STAGE_COUNTS.items():
        stage_forced = forced if stage == "model_screen" else []
        seed = hashlib.sha256(
            f"synthetic-development:{overlap}:{stage}".encode()
        ).digest()
        generated = opening_tools.generate_openings(
            stage=stage,
            count=count - len(stage_forced),
            seed=seed,
            excluded_fingerprints=seen,
        )
        openings = [copy.deepcopy(item) for item in stage_forced] + generated
        for index, opening in enumerate(openings):
            opening["opening_id"] = f"{stage}-{index:03d}"
            seen.update(
                value for name, value in opening["fingerprints"].items()
                if name != "canonical"
            )
        result[stage] = opening_tools.write_bank(
            root / stage,
            opening_tools.bank_payload(
                stage=stage,
                classification="unprotected-development",
                seed=seed,
                exclusions=exclusion_header,
                openings=openings,
            ),
        )
        opening_tools.validate_bank(result[stage])
    return result


class SyntheticCampaign:
    def __init__(self, root):
        self.root = root.resolve()
        runtime = self.root / "training/quantized-runtimes/synthetic.runtime.json"
        runtime.parent.mkdir(parents=True, exist_ok=True)
        runtime.write_text('{"synthetic":true}\n', encoding="ascii")
        self.runtime = exclusions._regular_record(runtime)
        self.plan_path = self.root / "discrete-v3-plan.json"
        self.plan = q.write_sealed(self.plan_path, {
            "schema": exclusions.v3.PLAN_SCHEMA,
            "namespace": exclusions.NAMESPACE,
            "campaign_id": exclusions.CAMPAIGN_ID,
            "fresh_protected_holdout": {
                "campaign_id": exclusions.HOLDOUT_CAMPAIGN_ID,
                "positions": exclusions.POSITION_COUNT,
                "positions_per_game": exclusions.POSITIONS_PER_GAME,
                "fresh_root_split": "test",
                "diagnostic_only": True,
                "selection_may_change_after_results": False,
                "minimum_samples_per_report": 20_000,
            },
        })
        selection_dir = self.root / "selections"
        selection_dir.mkdir(parents=True, exist_ok=True)
        self.selection_path = selection_dir / "synthetic.selection.json"
        self.selection = q.write_sealed(self.selection_path, {
            "schema": exclusions.v3.SELECTION_SCHEMA,
            "namespace": exclusions.NAMESPACE,
            "campaign_id": exclusions.CAMPAIGN_ID,
            "selection_immutable": True,
            "offline_gate": {"passed": True},
            "runtime": self.runtime,
        })
        retirement_root = self.root / "fresh-symmetry-exclusion"
        retirement_root.mkdir(parents=True, exist_ok=True)
        (retirement_root / "protected").mkdir()
        self.v1_retirement_path = retirement_root / "retirement-v1.json"
        self.v1_retirement = q.write_sealed(self.v1_retirement_path, {
            "schema": exclusions.RETIREMENT_SCHEMA,
            "namespace": exclusions.NAMESPACE,
            "campaign_id": exclusions.CAMPAIGN_ID,
            "status": "synthetic-v1-retired",
        })
        self.v2_plan_path = retirement_root / "plan-v2.json"
        self.v2_plan = q.write_sealed(self.v2_plan_path, {
            "schema": exclusions.V2_PLAN_SCHEMA,
            "namespace": exclusions.NAMESPACE,
            "campaign_id": exclusions.CAMPAIGN_ID,
            "status": "synthetic-v2-retired",
        })
        self.retirement_path = retirement_root / "retirement-v2.json"
        self.retirement = q.write_sealed(self.retirement_path, {
            "schema": exclusions.V2_RETIREMENT_SCHEMA,
            "namespace": exclusions.NAMESPACE,
            "campaign_id": exclusions.CAMPAIGN_ID,
            "status": "synthetic-v2-retired",
            "retirement_tool_closure": {
                "tool": exclusions._regular_record(TOOL),
                "test": exclusions._regular_record(exclusions.TEST_PATH),
            },
        })

    def validate(self, plan_path, output_root):
        if plan_path.absolute() != self.plan_path:
            raise exclusions.ExclusionError("synthetic plan redirected")
        if output_root.absolute() != self.root:
            raise exclusions.ExclusionError("synthetic output root redirected")
        return {
            "plan": self.plan,
            "selection": self.selection,
            "selection_path": self.selection_path,
        }

    def validate_retirement(self, path, *, output_root):
        if (
            path.resolve() != self.retirement_path.resolve()
            or output_root.resolve() != self.root
        ):
            raise exclusions.ExclusionError("synthetic retirement redirected")
        return q.load_sealed(path, exclusions.V2_RETIREMENT_SCHEMA)


class SyntheticMaterialization:
    def __init__(self, campaign):
        self.campaign = campaign
        root = campaign.root
        self.holdout_root = root / "fresh-holdout"
        self.materialized = self.holdout_root / "materialized"
        self.materialized.mkdir(parents=True, exist_ok=True)

        self.roots = self.holdout_root / "roots/fresh-test-roots.json"
        self.roots.parent.mkdir(parents=True, exist_ok=True)
        q.write_sealed(self.roots, {
            "schema": "papersoccer.synthetic-fresh-roots.v1",
            "accepted": [
                {
                    "group_id": f"fresh-protected-root:{index:04d}",
                    "split": "test",
                    "source": "synthetic-valid-terminal-game",
                }
                for index in range(3_200)
            ],
        })

        transcript_sha = hashlib.sha256(FULL_GAME.encode()).hexdigest()
        self.games = self.materialized / "games.tsv"
        game_lines = ["group_id\tsource\twinner\ttranscript"]
        game_rows = []
        for index in range(3_200):
            root_group_id = f"fresh-protected-root:{index:04d}"
            game_id = f"synthetic-game:{index:04d}"
            game_lines.append(
                f"{root_group_id}\tsynthetic-valid\t0\t{FULL_GAME}"
            )
            game_rows.append({
                "row_ordinal": index,
                "game_id": game_id,
                "root_group_id": root_group_id,
                "prefix_turns": PREFIX_TURNS,
                "transcript_sha256": transcript_sha,
            })
        self.games.write_text("\n".join(game_lines) + "\n", encoding="utf-8")
        self.games_manifest = self.materialized / "games.manifest.json"
        games_document = {
            "schema": selfsearch.GAME_MANIFEST_SCHEMA,
            "campaign_id": exclusions.HOLDOUT_CAMPAIGN_ID,
            "requested_games": 3_200,
            "successful_games": 3_200,
            "bindings": {"output_sha256": q.sha256_file(self.games)},
            "rows": game_rows,
        }
        self.games_manifest.write_bytes(
            selfsearch.canonical_json_bytes(games_document, pretty=True)
        )

        positions_payload, position_document = selfsearch.freeze_positions(
            campaign_id=exclusions.HOLDOUT_CAMPAIGN_ID,
            games_tsv=self.games,
            games_manifest=self.games_manifest,
            roots_manifest=self.roots,
            maximum_per_game=exclusions.POSITIONS_PER_GAME,
        )
        if (
            position_document["positions"] != exclusions.POSITION_COUNT
            or position_document["split_counts"]
            != {"test": exclusions.POSITION_COUNT}
        ):
            raise AssertionError("maintained producer did not create exact synthetic 64k")
        self.positions = self.materialized / "positions.tsv"
        self.positions.write_bytes(positions_payload)
        self.positions_manifest = self.materialized / "positions.manifest.json"
        self.positions_manifest.write_bytes(
            selfsearch.canonical_json_bytes(position_document, pretty=True)
        )

        def regular(name, content="synthetic\n"):
            path = self.materialized / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="ascii")
            return exclusions._regular_record(path)

        prior = root / "training/prior.runtime.json"
        prior.parent.mkdir(parents=True, exist_ok=True)
        prior.write_text('{"prior":true}\n', encoding="ascii")
        self.claim_path = self.holdout_root / "00-materialization-claim.json"
        q.write_sealed(self.claim_path, {
            "schema": exclusions.holdout.CLAIM_SCHEMA,
            "namespace": exclusions.NAMESPACE,
            "campaign_id": exclusions.HOLDOUT_CAMPAIGN_ID,
            "status": "fresh-protected-holdout-materialization-claimed-once",
            "successor_plan": q.artifact_reference(
                campaign.plan_path, exclusions.v3.PLAN_SCHEMA
            ),
            "immutable_selection": q.artifact_reference(
                campaign.selection_path, exclusions.v3.SELECTION_SCHEMA
            ),
            "selected_runtime": campaign.runtime,
            "prior_runtime": exclusions._regular_record(prior),
            "configuration": dict(campaign.plan["fresh_protected_holdout"]),
            "selection_may_change": False,
            "old_protected_tests_permitted": False,
            "materialization_attempts_authorized": 1,
            "exclusive_process_lock": str(
                (self.holdout_root / "materialization.lock").absolute()
            ),
            "claimed_at_utc": "2026-09-01T08:00:00Z",
        })

        fresh_bank = self.holdout_root / "roots/fresh-opening-bank.json"
        q.write_sealed(fresh_bank, {
            "schema": "papersoccer.synthetic-fresh-bank.v1",
            "count": 3_200,
        })
        test_shards = {
            name: regular(f"packed/{name}.test.manifest.json")
            for name in ("search", "rank4", "canonical")
        }
        stage_receipts = [
            regular(f"receipts/stage-{index:02d}.json") for index in range(1, 14)
        ]
        packing_priors = [
            regular(f"priors/prior-{index:02d}.json") for index in range(8)
        ]
        self.path = self.holdout_root / "materialization-receipt.json"
        self.body = {
            "schema": exclusions.holdout.MATERIALIZATION_SCHEMA,
            "namespace": exclusions.NAMESPACE,
            "campaign_id": exclusions.HOLDOUT_CAMPAIGN_ID,
            "status": "fresh-protected-holdout-materialized-once",
            "claim": q.artifact_reference(
                self.claim_path, exclusions.holdout.CLAIM_SCHEMA
            ),
            "immutable_selection": q.artifact_reference(
                campaign.selection_path, exclusions.v3.SELECTION_SCHEMA
            ),
            "game_plan": regular("game-plan.json"),
            "game_plan_tsv": regular("game-plan.tsv"),
            "game_plan_rows": 3_200,
            "fresh_roots": exclusions._regular_record(self.roots),
            "fresh_roots_tsv": regular("fresh-test-roots.tsv"),
            "fresh_opening_bank": q.artifact_reference(fresh_bank),
            "games": exclusions._regular_record(self.games),
            "games_manifest": exclusions._regular_record(self.games_manifest),
            "positions": exclusions._regular_record(self.positions),
            "positions_manifest": exclusions._regular_record(self.positions_manifest),
            "hard_positions": regular("hard-positions.tsv"),
            "search_labels": regular("labels/search-merged.jsonl"),
            "rank4_labels": regular("labels/rank4-merged.jsonl"),
            "canonical_labels": regular("labels/canonical-merged.jsonl"),
            "canonical_label_rows": 20_000,
            "packing_priors": packing_priors,
            "test_shards": test_shards,
            "test_samples": {
                "search": exclusions.POSITION_COUNT,
                "rank4": exclusions.POSITION_COUNT,
                "canonical": 20_000,
            },
            "group_isolation": {"passed": True},
            "split_isolation": {"passed": True},
            "stage_receipts": stage_receipts,
            "selection_changed": False,
            "old_protected_tests_accessed": False,
            "fresh_protected_tests_opened": True,
        }
        self._write_materialization()

    def _write_materialization(self):
        self.body["positions"] = exclusions._regular_record(self.positions)
        self.body["positions_manifest"] = exclusions._regular_record(
            self.positions_manifest
        )
        self.path.write_bytes(q.canonical_json_bytes(q.seal(self.body)))

    def rewrite_positions(self, lines, manifest):
        self.positions.write_text("\n".join(lines) + "\n", encoding="utf-8")
        manifest["output_sha256"] = q.sha256_file(self.positions)
        self.positions_manifest.write_bytes(
            selfsearch.canonical_json_bytes(manifest, pretty=True)
        )
        self._write_materialization()


class DiscreteV3SymmetryExclusionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._bank_temporary = tempfile.TemporaryDirectory()
        bank_root = pathlib.Path(cls._bank_temporary.name).resolve()
        cls.safe_banks = synthetic_development_banks(
            bank_root / "safe", overlap=False
        )
        cls.overlap_banks = synthetic_development_banks(
            bank_root / "overlap", overlap=True
        )

    @classmethod
    def tearDownClass(cls):
        cls._bank_temporary.cleanup()

    def test_fresh_prefix_replay_accepts_single_turn_and_shallow_complete_turns(self):
        for prefix in ("0", "0/0"):
            state = exclusions._replay_fresh_position_prefix(prefix)
            self.assertIsNone(state.winner)
            canonical, mover = exclusions._fresh_position_prefix_identity(
                prefix, state.to_move
            )
            self.assertRegex(canonical, r"^[0-9a-f]{64}$")
            self.assertEqual(mover, state.to_move)
            with self.assertRaisesRegex(
                exclusions.ExclusionError, "mover differs"
            ):
                exclusions._fresh_position_prefix_identity(
                    prefix, 1 - state.to_move
                )

    def test_360_empty_root_prefixes_are_exact_initial_state_with_mover_zero(self):
        expected = opening_tools.reference.ReplayState()
        state = exclusions._replay_fresh_position_prefix("")
        self.assertEqual(state, expected)
        canonical = opening_tools.state_fingerprints(expected)["canonical"]
        identities = [
            exclusions._fresh_position_prefix_identity("", 0)
            for _index in range(360)
        ]
        self.assertEqual(identities, [(canonical, 0)] * 360)
        with self.assertRaisesRegex(exclusions.ExclusionError, "mover differs"):
            exclusions._fresh_position_prefix_identity("", 1)

    def test_fresh_prefix_replay_rejects_invalid_incomplete_and_terminal(self):
        for prefix in (None, "8", "0//0", "00", "0/0/3/0/61/0/07"):
            with self.subTest(prefix=prefix), self.assertRaises(
                exclusions.ExclusionError
            ):
                exclusions._replay_fresh_position_prefix(prefix)

    def test_v1_retirement_is_write_once_and_records_exact_failure_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary).resolve()
            artifact_root = root / "fresh-symmetry-exclusion"
            protected = artifact_root / "protected"
            protected.mkdir(parents=True)
            old_tool = root / "old-tool.py"
            old_test = root / "old-test.py"
            old_tool.write_text("old tool\n")
            old_test.write_text("old test\n")
            plan_path = artifact_root / "plan.json"
            plan = q.write_sealed(plan_path, {
                "schema": exclusions.V1_PLAN_SCHEMA,
                "tools": {
                    "exclusion_tool": {
                        "path": str(old_tool), "bytes": old_tool.stat().st_size,
                        "sha256": exclusions.V1_TOOL_SHA256,
                    },
                    "exclusion_test": {
                        "path": str(old_test), "bytes": old_test.stat().st_size,
                        "sha256": exclusions.V1_TEST_SHA256,
                    },
                },
            })

            def validate_v1(output_root):
                if output_root.resolve() != root:
                    raise exclusions.ExclusionError("synthetic v1 state changed")
                return {
                    "artifact_root": artifact_root,
                    "plan_path": plan_path,
                    "protected": protected,
                    "plan": plan,
                }

            retirement = exclusions.retire_v1(
                output_root=root, retired_at_utc="2026-09-01T14:00:00Z",
                v1_validator=validate_v1,
            )
            self.assertEqual(retirement, exclusions.retire_v1(
                output_root=root, retired_at_utc="2026-09-01T14:00:00Z",
                v1_validator=validate_v1,
            ))
            value = q.load_sealed(retirement, exclusions.RETIREMENT_SCHEMA)
            self.assertEqual(value["failure_reason"], exclusions.V1_FAILURE_REASON)
            self.assertFalse(value["receipt_created"])
            self.assertFalse(value["private_payload_created"])
            self.assertFalse(value["metrics_opened"])
            self.assertFalse(value["upload_authorized"])

            (artifact_root / "plan-v2.json").write_text("v2 plan\n")
            (artifact_root / "receipt.json").write_text("v2 receipt\n")
            (protected / "v2-private-payload.json").write_text("v2 payload\n")
            self.assertEqual(
                q.load_sealed(
                    exclusions.retire_v1(
                        output_root=root,
                        retired_at_utc="2026-09-01T14:00:00Z",
                        v1_validator=validate_v1,
                    ),
                    exclusions.RETIREMENT_SCHEMA,
                ),
                value,
            )
            self.assertEqual(
                exclusions.validate_retirement(
                    retirement,
                    output_root=root,
                    v1_validator=validate_v1,
                ),
                value,
            )

    def test_v2_retirement_is_write_once_pristine_and_valid_after_v3_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary).resolve()
            artifact_root = root / "fresh-symmetry-exclusion"
            protected = artifact_root / "protected"
            protected.mkdir(parents=True)
            q.write_sealed(artifact_root / "plan.json", {
                "schema": exclusions.V1_PLAN_SCHEMA,
            })
            v1_retirement_path = artifact_root / "retirement-v1.json"
            q.write_sealed(v1_retirement_path, {
                "schema": exclusions.RETIREMENT_SCHEMA,
            })
            old_tool = root / "v2-tool.py"
            old_test = root / "v2-test.py"
            old_opening_tool = root / "v2-opening-tool.py"
            old_tool.write_text("v2 tool\n")
            old_test.write_text("v2 test\n")
            old_opening_tool.write_text("v2 opening tool\n")
            plan_path = artifact_root / "plan-v2.json"
            plan = q.write_sealed(plan_path, {
                "schema": exclusions.V2_PLAN_SCHEMA,
                "v1_retirement": q.artifact_reference(
                    v1_retirement_path, exclusions.RETIREMENT_SCHEMA
                ),
                "tools": {
                    "exclusion_tool": exclusions._regular_record(old_tool),
                    "exclusion_test": exclusions._regular_record(old_test),
                    "opening_tool": exclusions._regular_record(old_opening_tool),
                },
            })

            def validate_v2(output_root):
                if output_root.resolve() != root:
                    raise exclusions.ExclusionError("synthetic v2 state changed")
                return {
                    "artifact_root": artifact_root,
                    "protected": protected,
                    "plan_path": plan_path,
                    "plan": plan,
                    "v1_retirement_path": v1_retirement_path,
                }

            foreign = protected / "foreign-before-v2-retirement"
            foreign.write_text("foreign\n")
            with self.assertRaisesRegex(
                exclusions.ExclusionError, "not empty"
            ):
                exclusions.retire_v2(
                    output_root=root,
                    retired_at_utc="2026-09-01T18:30:00Z",
                    v2_validator=validate_v2,
                )
            self.assertFalse((artifact_root / "retirement-v2.json").exists())
            foreign.unlink()

            retirement = exclusions.retire_v2(
                output_root=root,
                retired_at_utc="2026-09-01T18:30:00Z",
                v2_validator=validate_v2,
            )
            self.assertEqual(retirement, exclusions.retire_v2(
                output_root=root,
                retired_at_utc="2026-09-01T18:30:00Z",
                v2_validator=validate_v2,
            ))
            value = q.load_sealed(
                retirement, exclusions.V2_RETIREMENT_SCHEMA
            )
            self.assertEqual(value["failure_reason"], exclusions.V2_FAILURE_REASON)
            self.assertEqual(
                value["v2_plan"],
                q.artifact_reference(plan_path, exclusions.V2_PLAN_SCHEMA),
            )
            self.assertEqual(
                value["v1_retirement"],
                q.artifact_reference(
                    v1_retirement_path, exclusions.RETIREMENT_SCHEMA
                ),
            )
            self.assertFalse(value["receipt_created"])
            self.assertFalse(value["private_payload_created"])
            self.assertFalse(value["metrics_opened"])

            (artifact_root / "plan-v3.json").write_text("v3 plan\n")
            (artifact_root / "receipt.json").write_text("v3 receipt\n")
            (protected / "v3-private-payload.json").write_text("v3 payload\n")
            self.assertEqual(
                exclusions.validate_v2_retirement(
                    retirement,
                    output_root=root,
                    v2_validator=validate_v2,
                ),
                value,
            )
            self.assertEqual(
                exclusions.retire_v2(
                    output_root=root,
                    retired_at_utc="2026-09-01T18:30:00Z",
                    v2_validator=validate_v2,
                ),
                retirement,
            )

    def test_v3_prepare_requires_v2_retirement_empty_protected_and_disjoint_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            campaign = SyntheticCampaign(pathlib.Path(temporary))
            with self.assertRaisesRegex(
                exclusions.ExclusionError, "retirement is required"
            ):
                exclusions.prepare(
                    output_root=campaign.root,
                    v3_plan_path=campaign.plan_path,
                    v2_retirement_path=campaign.retirement_path,
                    development_bank_paths=self.safe_banks,
                    created_at_utc="2026-09-01T14:01:00Z",
                    campaign_validator=campaign.validate,
                    bank_identity_validator=accept_synthetic_bank_identity,
                    retirement_validator=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                        exclusions.ExclusionError("missing")
                    ),
                )
            plan = self.prepare(campaign, self.safe_banks)
            self.assertEqual(plan.name, "plan-v3.json")
            self.assertNotEqual(
                plan, campaign.root / "fresh-symmetry-exclusion/plan-v2.json"
            )
            value = q.load_sealed(plan, exclusions.PLAN_SCHEMA)
            self.assertEqual(
                value["v2_retirement"], q.artifact_reference(
                    campaign.retirement_path, exclusions.V2_RETIREMENT_SCHEMA
                )
            )
            self.assertTrue(value["contract"]["empty_initial_prefix_allowed"])
            self.assertEqual(
                value["contract"]["prefix_replay"],
                "maintained-ReplayState+apply_complete_turn",
            )

        with tempfile.TemporaryDirectory() as temporary:
            campaign = SyntheticCampaign(pathlib.Path(temporary))
            (campaign.root / "fresh-symmetry-exclusion/protected/foreign").write_text(
                "foreign\n"
            )
            with self.assertRaisesRegex(
                exclusions.ExclusionError, "output exists"
            ):
                self.prepare(campaign, self.safe_banks)

    def prepare(self, campaign, banks, *, created="2026-09-01T07:00:00Z"):
        return exclusions.prepare(
            output_root=campaign.root,
            v3_plan_path=campaign.plan_path,
            v2_retirement_path=campaign.retirement_path,
            development_bank_paths=banks,
            created_at_utc=created,
            campaign_validator=campaign.validate,
            bank_identity_validator=accept_synthetic_bank_identity,
            retirement_validator=campaign.validate_retirement,
        )

    def audit(self, campaign, plan):
        return exclusions.audit(
            output_root=campaign.root,
            plan_path=plan,
            campaign_validator=campaign.validate,
            bank_identity_validator=accept_synthetic_bank_identity,
            retirement_validator=campaign.validate_retirement,
        )

    def test_zero_overlap_exact_resume_private_payload_and_public_surface(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            campaign = SyntheticCampaign(root)
            plan = self.prepare(campaign, self.safe_banks)
            self.assertEqual(plan, self.prepare(campaign, self.safe_banks))
            SyntheticMaterialization(campaign)
            receipt_path = self.audit(campaign, plan)
            self.assertEqual(receipt_path, self.audit(campaign, plan))
            receipt = q.load_sealed(receipt_path, exclusions.RECEIPT_SCHEMA)
            self.assertEqual(
                receipt["status"],
                "zero-overlap-existing-development-roster-ready",
            )
            self.assertEqual(receipt["intersection"], {
                "unique_canonical_count": 0,
                "fresh_position_row_count": 0,
            })
            self.assertTrue(receipt["verdict"]["development_games_authorized"])
            self.assertFalse(receipt["verdict"]["rank4_gate_authorized"])
            self.assertFalse(receipt["verdict"]["upload_authorized"])
            protected_ref = receipt["references"]["protected_canonical_fingerprints"]
            protected_path = pathlib.Path(protected_ref["path"])
            self.assertEqual(
                protected_path.name.removesuffix(
                    ".fresh-canonical-fingerprints.json"
                ),
                q.sha256_file(protected_path),
            )
            payload = q.load_sealed(protected_path, exclusions.FINGERPRINT_SCHEMA)
            self.assertEqual(payload["position_count"], 64_000)
            self.assertEqual(len(payload["rows"]), 64_000)
            self.assertFalse(payload["contains_transcripts"])
            self.assertEqual(protected_path.stat().st_mode & 0o777, 0o600)
            public_text = receipt_path.read_text(encoding="ascii")
            self.assertNotIn("canonical_sha256", public_text)
            self.assertNotIn("transcript", public_text)
            self.assertNotIn("position_id", public_text)
            validated = exclusions.validate_receipt(
                receipt_path,
                plan_path=plan,
                output_root=campaign.root,
                campaign_validator=campaign.validate,
                bank_identity_validator=accept_synthetic_bank_identity,
                retirement_validator=campaign.validate_retirement,
            )
            self.assertEqual(set(validated), {
                "receipt", "plan", "protected_fingerprint_path",
                "development_bank_records", "development_ready",
            })
            self.assertNotIn("fresh_canonical_fingerprints", validated)
            self.assertTrue(validated["development_ready"])
            self.assertEqual(
                len(exclusions._load_private_canonical_fingerprints(
                    receipt_path,
                    plan_path=plan,
                    output_root=campaign.root,
                    campaign_validator=campaign.validate,
                    bank_identity_validator=accept_synthetic_bank_identity,
                    retirement_validator=campaign.validate_retirement,
                )),
                exclusions.POSITIONS_PER_GAME,
            )

    def test_interrupted_payload_only_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            campaign = SyntheticCampaign(root)
            plan = self.prepare(campaign, self.safe_banks)
            SyntheticMaterialization(campaign)
            receipt = self.audit(campaign, plan)
            value = q.load_sealed(receipt, exclusions.RECEIPT_SCHEMA)
            protected = pathlib.Path(
                value["references"]["protected_canonical_fingerprints"]["path"]
            )
            protected_sha = q.sha256_file(protected)
            receipt.unlink()
            resumed = self.audit(campaign, plan)
            self.assertTrue(resumed.is_file())
            self.assertEqual(q.sha256_file(protected), protected_sha)

    def test_all_four_transforms_overlap_and_force_full_roster_regeneration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            campaign = SyntheticCampaign(root)
            plan = self.prepare(campaign, self.overlap_banks)
            SyntheticMaterialization(campaign)
            receipt_path = self.audit(campaign, plan)
            receipt = q.load_sealed(receipt_path, exclusions.RECEIPT_SCHEMA)
            self.assertEqual(
                receipt["status"],
                "overlap-detected-full-development-roster-regeneration-required",
            )
            self.assertEqual(receipt["intersection"]["unique_canonical_count"], 4)
            self.assertEqual(
                receipt["intersection"]["fresh_position_row_count"], 12_800
            )
            self.assertFalse(receipt["verdict"]["development_games_authorized"])
            plan_value = q.load_sealed(plan, exclusions.PLAN_SCHEMA)
            self.assertEqual(
                plan_value["contract"]["symmetries"],
                ["exact", "rotate", "reflect", "rotate_reflect"],
            )
            self.assertEqual(
                plan_value["regeneration_policy"]["master_seed_hex"],
                exclusions.FALLBACK_MASTER_SEED_HEX,
            )
            with self.assertRaisesRegex(
                exclusions.ExclusionError, "full six-bank regeneration"
            ):
                exclusions.require_development_roster(
                    receipt_path,
                    plan_path=plan,
                    output_root=campaign.root,
                    campaign_validator=campaign.validate,
                    bank_identity_validator=accept_synthetic_bank_identity,
                    retirement_validator=campaign.validate_retirement,
                )

    def test_exact_freeze_positions_replay_rejects_foreign_legal_row(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            campaign = SyntheticCampaign(root)
            plan = self.prepare(campaign, self.safe_banks)
            materialization = SyntheticMaterialization(campaign)
            lines = materialization.positions.read_text(encoding="utf-8").splitlines()
            manifest = json.loads(
                materialization.positions_manifest.read_text(encoding="utf-8")
            )
            first = lines[1].split("\t")
            alternate = lines[2].split("\t")
            first[0] = "foreign-but-well-formed-position-id"
            first[6] = alternate[6]
            first[7] = alternate[7]
            lines[1] = "\t".join(first)
            manifest["rows"][0]["position_id"] = first[0]
            manifest["rows"][0]["mover"] = int(first[6])
            manifest["rows"][0]["turn"] = manifest["rows"][1]["turn"]
            manifest["rows"][0]["game_row_ordinal"] = 1
            materialization.rewrite_positions(lines, manifest)
            with self.assertRaisesRegex(
                exclusions.ExclusionError, "exact maintained freeze_positions output"
            ):
                self.audit(campaign, plan)

    def test_incomplete_position_manifest_is_rejected_before_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            campaign = SyntheticCampaign(root)
            plan = self.prepare(campaign, self.safe_banks)
            materialization = SyntheticMaterialization(campaign)
            lines = materialization.positions.read_text(encoding="utf-8").splitlines()
            manifest = json.loads(
                materialization.positions_manifest.read_text(encoding="utf-8")
            )
            manifest["positions"] = 63_999
            manifest["split_counts"] = {"test": 63_999}
            materialization.rewrite_positions(lines, manifest)
            with self.assertRaisesRegex(
                exclusions.ExclusionError, "position manifest contract"
            ):
                self.audit(campaign, plan)
            self.assertFalse(
                (root / "fresh-symmetry-exclusion/receipt.json").exists()
            )

    def test_output_and_bank_symlinks_and_nondirectory_components_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = pathlib.Path(temporary)
            real_root = base / "real"
            real_root.mkdir()
            campaign = SyntheticCampaign(real_root)
            alias = base / "alias"
            alias.symlink_to(real_root, target_is_directory=True)
            with self.assertRaisesRegex(
                exclusions.ExclusionError, "redirected|symlink"
            ):
                exclusions.prepare(
                    output_root=alias,
                    v3_plan_path=campaign.plan_path,
                    v2_retirement_path=campaign.retirement_path,
                    development_bank_paths=self.safe_banks,
                    created_at_utc="2026-09-01T07:00:00Z",
                    campaign_validator=campaign.validate,
                    bank_identity_validator=accept_synthetic_bank_identity,
                    retirement_validator=campaign.validate_retirement,
                )

            nested_real = real_root / "nested"
            nested_real.mkdir()
            nested_campaign = SyntheticCampaign(nested_real)
            ancestor_alias = base / "ancestor-alias"
            ancestor_alias.symlink_to(real_root, target_is_directory=True)
            with self.assertRaisesRegex(
                exclusions.ExclusionError, "redirected|symlink"
            ):
                exclusions.prepare(
                    output_root=ancestor_alias / "nested",
                    v3_plan_path=nested_campaign.plan_path,
                    v2_retirement_path=nested_campaign.retirement_path,
                    development_bank_paths=self.safe_banks,
                    created_at_utc="2026-09-01T07:00:00Z",
                    campaign_validator=nested_campaign.validate,
                    bank_identity_validator=accept_synthetic_bank_identity,
                    retirement_validator=nested_campaign.validate_retirement,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            campaign = SyntheticCampaign(root)
            external = root / "external"
            external.mkdir()
            shutil.rmtree(root / "fresh-symmetry-exclusion")
            (root / "fresh-symmetry-exclusion").symlink_to(
                external, target_is_directory=True
            )
            with self.assertRaisesRegex(exclusions.ExclusionError, "symlink"):
                self.prepare(campaign, self.safe_banks)

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            campaign = SyntheticCampaign(root)
            shutil.rmtree(root / "fresh-symmetry-exclusion")
            (root / "fresh-symmetry-exclusion").write_text("not-directory\n")
            with self.assertRaisesRegex(exclusions.ExclusionError, "not a directory"):
                self.prepare(campaign, self.safe_banks)

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            campaign = SyntheticCampaign(root)
            linked = root / "linked-bank.json"
            linked.symlink_to(self.safe_banks["model_screen"])
            banks = dict(self.safe_banks)
            banks["model_screen"] = linked
            with self.assertRaisesRegex(
                exclusions.ExclusionError, "redirected|symlink"
            ):
                self.prepare(campaign, banks)

    def test_protected_directory_and_public_receipt_symlinks_reject_before_audit(self):
        for target in ("protected", "receipt.json"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary)
                campaign = SyntheticCampaign(root)
                plan = self.prepare(campaign, self.safe_banks)
                external = root / "external"
                external.mkdir()
                path = root / "fresh-symmetry-exclusion" / target
                if target == "protected":
                    path.rmdir()
                path.symlink_to(
                    external,
                    target_is_directory=(target == "protected"),
                )
                with self.assertRaisesRegex(exclusions.ExclusionError, "symlink"):
                    self.audit(campaign, plan)

    def test_prepare_rejects_partial_repeated_and_foreign_rosters(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            campaign = SyntheticCampaign(root)
            partial = dict(self.safe_banks)
            partial.pop("actual_clock")
            with self.assertRaisesRegex(exclusions.ExclusionError, "exact six-stage"):
                self.prepare(campaign, partial)
            repeated = dict(self.safe_banks)
            repeated["actual_clock"] = repeated["model_screen"]
            with self.assertRaises(
                (exclusions.ExclusionError, opening_tools.OpeningError)
            ):
                self.prepare(campaign, repeated)
            with self.assertRaisesRegex(
                exclusions.ExclusionError, "exact frozen v1 development bank"
            ):
                exclusions.prepare(
                    output_root=campaign.root,
                    v3_plan_path=campaign.plan_path,
                    v2_retirement_path=campaign.retirement_path,
                    development_bank_paths=self.safe_banks,
                    created_at_utc="2026-09-01T07:00:00Z",
                    campaign_validator=campaign.validate,
                    retirement_validator=campaign.validate_retirement,
                )

    def test_private_payload_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            campaign = SyntheticCampaign(root)
            plan = self.prepare(campaign, self.safe_banks)
            SyntheticMaterialization(campaign)
            receipt = self.audit(campaign, plan)
            value = q.load_sealed(receipt, exclusions.RECEIPT_SCHEMA)
            protected = pathlib.Path(
                value["references"]["protected_canonical_fingerprints"]["path"]
            )
            protected.write_text("{}\n", encoding="ascii")
            with self.assertRaisesRegex(
                exclusions.ExclusionError, "reference changed|body SHA-256"
            ):
                exclusions.validate_receipt(
                    receipt,
                    plan_path=plan,
                    output_root=campaign.root,
                    campaign_validator=campaign.validate,
                    bank_identity_validator=accept_synthetic_bank_identity,
                    retirement_validator=campaign.validate_retirement,
                    recompute_positions=False,
                )


if __name__ == "__main__":
    unittest.main()
