import io
from pathlib import Path
import unittest
from unittest.mock import patch

from tools import rank_two_live_eight_ancestry_v6 as ancestry


class TeacherScopeTests(unittest.TestCase):
    def fixture(self):
        base = ancestry.c.ROOT / 'results/scope-test-only'
        root = base / 'live-eight-v1/live-data-continuation-04'
        teacher = root / 'teacher256-v1'
        def ref(path):
            return dict(path=str(path), sha256=str(path))
        plan_ref = ref(teacher / 'TEACHER_PLAN.json')
        completion_ref = ref(teacher / 'QUEUE_COMPLETION.json')
        result_ref = ref(teacher / 'execution/RESULT.json')
        count_ref = ref(teacher / 'NEW_PAYLOAD_COUNTS.json')
        prior = ref(base / 'prior.json')
        declaration = ref(root / 'DECLARATION.json')
        exposures = [ref(teacher / name) for name in ('RESERVED_EXPOSURE.json',
            'execution/RESERVED_EXPOSURE.json', 'execution/EXPOSURE_PENDING.json')]
        plan = dict(base=str(base), continuation_root=str(root),
            scope_kind='completed-current-TRAIN-teacher',
            payload_roots=[str(teacher / 'execution')],
            payload_files=[plan_ref, exposures[0]], exposures=exposures,
            prior_validated_union=prior, teacher_completion=completion_ref,
            teacher_plan=plan_ref, declaration=declaration,
            payload_counts=count_ref, maximum_normalized_rows=5)
        state = dict(pending_exposures=exposures, live_eight_ancestry_validated=prior,
            live_eight_continuation_04_teacher_complete=completion_ref,
            live_eight_continuation_04_teacher_plan=plan_ref,
            live_eight_continuation_04_declaration=declaration)
        docs = {
            completion_ref['path']: dict(passed=True, result=result_ref, payload_counts=count_ref),
            plan_ref['path']: dict(nodes=256000, outcomes_as_targets=False,
                inputs=dict(declaration=declaration), positions=[dict(split='train')]),
            result_ref['path']: dict(passed=True, plan=plan_ref, requested_parents=1,
                valid_labels=1, unsupported_labels=0, parent_receipts=[{}]),
            count_ref['path']: dict(execution=result_ref, teacher_plan=plan_ref,
                parent_receipts=1, successor_traces=3, plan_parent_traces=1,
                expected_normalized_rows=5)}
        return plan, state, docs

    def check(self, plan, state, docs):
        with patch.object(ancestry.c, 'verify', side_effect=lambda ref: ref['path']), \
             patch.object(ancestry.c, 'read', side_effect=lambda path: docs[str(path)]):
            return ancestry.validate_scope(plan, state)

    def test_complete_batch_uses_exact_trace_count(self):
        self.assertEqual(self.check(*self.fixture()), 5)

    def test_other_continuation_receipts_cannot_satisfy_current_batch(self):
        plan, state, docs = self.fixture()
        for field in ('declaration', 'teacher_plan', 'teacher_complete'):
            state['live_eight_continuation_03_' + field] = state.pop(
                'live_eight_continuation_04_' + field)
        with self.assertRaises(PermissionError):
            self.check(plan, state, docs)

    def test_changed_declaration_and_teacher_declaration_are_rejected(self):
        for change_teacher in (False, True):
            with self.subTest(change_teacher=change_teacher):
                plan, state, docs = self.fixture()
                if change_teacher:
                    docs[plan['teacher_plan']['path']]['inputs']['declaration'] = {}
                else:
                    state['live_eight_continuation_04_declaration'] = {}
                with self.assertRaises(PermissionError):
                    self.check(plan, state, docs)

    def test_ambiguous_continuation_name_is_rejected(self):
        for name in ('live-data-continuation-4', 'live-data-continuation-04-old',
                     'live-data-continuation-00'):
            with self.subTest(name=name):
                plan, state, docs = self.fixture()
                plan['continuation_root'] = str(Path(plan['continuation_root']).with_name(name))
                with self.assertRaises(PermissionError):
                    self.check(plan, state, docs)

    def test_prior_payload_is_rejected_before_file_access(self):
        normal = ancestry.ScopedNormalizer(io.StringIO(), [ancestry.c.ROOT / 'results/new-only'], [], 90000)
        with self.assertRaises(PermissionError):
            normal.bind(dict(path=str(ancestry.c.ROOT / 'results/old-missing.json'),
                             sha256='unused'), parse=True)

    def test_unclosed_or_changed_batch_is_rejected(self):
        for field in ('pending_exposures', 'live_eight_ancestry_validated',
                      'live_eight_continuation_04_teacher_complete'):
            with self.subTest(field=field):
                plan, state, docs = self.fixture()
                state[field] = [] if field == 'pending_exposures' else {}
                with self.assertRaises(PermissionError):
                    self.check(plan, state, docs)

    def test_broad_scope_and_inflated_cap_are_rejected(self):
        for field, value in [('payload_roots', [str(ancestry.c.ROOT)]),
                             ('payload_files', []), ('maximum_normalized_rows', 6)]:
            with self.subTest(field=field):
                plan, state, docs = self.fixture()
                plan[field] = value
                with self.assertRaises(PermissionError):
                    self.check(plan, state, docs)

    def test_row_cap_stops_before_extra_trace_is_written(self):
        stream = io.StringIO()
        normal = ancestry.ScopedNormalizer(stream, [], [], 1)
        normal.emit([], dict(path='new', sha256='bound'), '/parent')
        with self.assertRaises(PermissionError):
            normal.emit(['0'], dict(path='new', sha256='bound'), '/successor')
        self.assertEqual(len(stream.getvalue().splitlines()), 1)


if __name__ == '__main__':
    unittest.main()
