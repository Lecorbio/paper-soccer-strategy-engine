import io
import unittest
from unittest.mock import patch

from tools import rank_two_live_eight_ancestry_v7 as ancestry


class FixedValidationScopeTests(unittest.TestCase):
    def fixture(self):
        base = ancestry.c.ROOT / 'results/scope-test-only'
        root = base / 'live-eight-v1/live-data-continuation-04/fresh-validation-v1'
        teacher = root / 'teacher256-v1'
        def ref(path):
            return dict(path=str(path), sha256=str(path))
        exposures = [ref(path) for path in [root / 'EXPOSURE_PENDING.json',
            root / 'COMPLETE_EXPOSURE.json', teacher / 'RESERVED_EXPOSURE.json',
            teacher / 'execution/RESERVED_EXPOSURE.json', teacher / 'execution/EXPOSURE_PENDING.json']]
        generation = ref(root / 'QUEUE_COMPLETION.json')
        labels = ref(teacher / 'QUEUE_COMPLETION.json')
        prior = ref(base / 'prior.json')
        declaration = ref(root.parent / 'DECLARATION.json')
        generation_plan = ref(root / 'PLAN.json')
        teacher_plan = ref(teacher / 'TEACHER_PLAN.json')
        plan = dict(base=str(base), validation_root=str(root),
            scope_kind='completed-current-fixed-validation',
            payload_roots=[str(root / 'games'), str(teacher)],
            payload_files=[ref(root / 'PLAN.json'), *exposures[:2]],
            exposures=exposures, generation_completion=generation,
            teacher_completion=labels, declaration=declaration,
            generation_plan=generation_plan, teacher_plan=teacher_plan, prior_validated_union=prior,
            maximum_normalized_rows=100000)
        state = dict(pending_exposures=exposures, live_eight_ancestry_validated=prior,
            live_eight_continuation_04_validation_complete=generation,
            live_eight_continuation_04_validation_teacher_complete=labels,
            live_eight_continuation_04_declaration=declaration,
            live_eight_continuation_04_validation_plan=generation_plan,
            live_eight_continuation_04_validation_teacher_plan=teacher_plan)
        docs = {generation['path']: dict(passed=True, games=256, roots=128,
                plan=generation_plan, seed=2026101235),
            labels['path']: dict(passed=True, validation_roots=128, validation_only=True,
                teacher_plan=teacher_plan, outcomes_as_targets=False),
            declaration['path']: dict(architecture=[6301, 8, 8, 1], outcomes_as_targets=False,
                fresh_validation_recipe=dict(colors=[0, 1], maximum_early_parents_per_root=2,
                    minimum_comparable_early_roots=100, no_seed_or_root_replacement=True,
                    outcomes_as_targets=False, root_depths=[8, 10], roots=128, seed=2026101235))}
        return plan, state, docs

    def check(self, plan, state, docs):
        with patch.object(ancestry.c, 'verify', side_effect=lambda ref: ref['path']), \
             patch.object(ancestry.c, 'read', side_effect=lambda path: docs[str(path)]):
            ancestry.validate_scope(plan, state)

    def test_complete_fixed_bank_passes(self):
        self.check(*self.fixture())

    def test_current_count_schema_passes_and_conflicting_aliases_fail(self):
        plan, state, docs = self.fixture()
        generation = docs[plan['generation_completion']['path']]
        generation['complete_games'] = generation.pop('games')
        generation['frozen_roots'] = generation.pop('roots')
        self.check(plan, state, docs)
        generation['games'] = 255
        with self.assertRaises(PermissionError):
            self.check(plan, state, docs)

    def test_cross_continuation_or_replaced_declaration_is_rejected(self):
        for field in ('declaration', 'generation_plan', 'generation_completion', 'teacher_plan', 'teacher_completion'):
            with self.subTest(field=field):
                plan, state, docs = self.fixture()
                plan[field] = {}
                with self.assertRaises(PermissionError):
                    self.check(plan, state, docs)

    def test_relaxed_early_root_recipe_is_rejected(self):
        plan, state, docs = self.fixture()
        docs[plan['declaration']['path']]['fresh_validation_recipe']['minimum_comparable_early_roots'] = 99
        with self.assertRaises(PermissionError):
            self.check(plan, state, docs)

    def test_old_payload_is_rejected_before_file_access(self):
        normal = ancestry.ScopedNormalizer(io.StringIO(), [ancestry.c.ROOT / 'results/current-only'], [], 100000)
        with self.assertRaises(PermissionError):
            normal.bind(dict(path=str(ancestry.c.ROOT / 'results/old-missing.json'), sha256='unused'), True)

    def test_broad_or_previously_closed_scope_is_rejected(self):
        for change in ['broad', 'closed', 'missing']:
            with self.subTest(change=change):
                plan, state, docs = self.fixture()
                if change == 'broad':
                    plan['payload_roots'] = [str(ancestry.c.ROOT)]
                elif change == 'closed':
                    state['pending_exposures'] = []
                else:
                    plan['exposures'] = plan['exposures'][:-1]
                with self.assertRaises(PermissionError):
                    self.check(plan, state, docs)

    def test_replaced_seed_or_incomplete_bank_is_rejected(self):
        for field, value in [('seed', 2026101136), ('roots', 127), ('games', 255)]:
            with self.subTest(field=field):
                plan, state, docs = self.fixture()
                docs[plan['generation_completion']['path']][field] = value
                with self.assertRaises(PermissionError):
                    self.check(plan, state, docs)


if __name__ == '__main__':
    unittest.main()
