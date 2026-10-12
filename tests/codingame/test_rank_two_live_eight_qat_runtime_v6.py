import copy
import unittest
from unittest.mock import patch

from tools import rank_two_live_eight_qat_runtime_v6 as runtime
from tools import rank_two_live_eight_precision_v6 as precision


class MatchedRuntimeBindingTests(unittest.TestCase):
    def fixture(self):
        def ref(name): return dict(path=name, sha256=name)
        plan = dict(path='runtime-plan', job_id='qat4')
        numeric = dict(producer=ref('training'), phase_mode='qat', weight_bits=4,
            profile='dd8', corpus=ref('corpus'), fresh_validation=ref('validation'),
            float_parent=ref('float-parent'))
        state = dict(active_jobs=['qat4'], live_eight_continuation_04_qat4_runtime_plan=ref('runtime-plan'),
            live_eight_next_corpus=ref('corpus'), live_eight_next_fresh_validation=ref('validation'))
        verified = dict(passed=True, selected_parent=ref('float-parent'),
            all_epochs_native128parity_retained=True,
            all_journals_root_orders_full_moments_verified=True, atomic_head_exact_last_epoch=True)
        return plan, numeric, state, verified

    def check(self, *args):
        def record(path):
            name = 'training' if str(path).endswith('rank_two_live_eight_training_v5.py') else str(path)
            return dict(path=name, sha256=name)
        with patch.object(runtime.c, 'record', side_effect=record):
            runtime.validate_binding(*args)

    def test_current_exact_parent_passes(self):
        self.check(*self.fixture())

    def test_replaced_parent_or_unverified_moments_are_rejected(self):
        for changed in ('parent', 'moments'):
            with self.subTest(changed=changed):
                plan, numeric, state, verified = self.fixture()
                if changed == 'parent': verified['selected_parent'] = {}
                else: verified['all_journals_root_orders_full_moments_verified'] = False
                with self.assertRaises(PermissionError): self.check(plan, numeric, state, verified)

    def test_foreign_execution_bank_or_unsupported_precision_is_rejected(self):
        for changed in ('job', 'bank', 'bits'):
            with self.subTest(changed=changed):
                plan, numeric, state, verified = self.fixture()
                if changed == 'job': state['active_jobs'] = ['another-job']
                elif changed == 'bank': numeric['fresh_validation'] = {}
                else: numeric['weight_bits'] = 7
                with self.assertRaises(PermissionError): self.check(plan, numeric, state, verified)

    def test_precision_freeze_requires_actual_audited_float_runtime(self):
        ref = dict(path='numeric', sha256='numeric')
        job = dict(path='job', sha256='job')
        producer = dict(path='producer', sha256='producer')
        runtime_plan = dict(numeric_plan=ref, resource_budget=dict(workers=10), producer=producer)
        verified = dict(passed=True, numeric_plan=ref, all_epochs_native128parity_retained=True,
            all_journals_root_orders_full_moments_verified=True, atomic_head_exact_last_epoch=True)
        completion = dict(passed=True, job=job)
        with patch.object(precision.c, 'record', return_value=producer):
            precision.validate_runtime_binding(runtime_plan, ref, verified, completion, job)
            for changed in ('plan', 'moments', 'job'):
                rp, audit, completed = map(copy.deepcopy, (runtime_plan, verified, completion))
                if changed == 'plan': rp['numeric_plan'] = {}
                elif changed == 'moments': audit['all_journals_root_orders_full_moments_verified'] = False
                else: completed['job'] = {}
                with self.subTest(changed=changed), self.assertRaises(PermissionError):
                    precision.validate_runtime_binding(rp, ref, audit, completed, job)


if __name__ == '__main__': unittest.main()
