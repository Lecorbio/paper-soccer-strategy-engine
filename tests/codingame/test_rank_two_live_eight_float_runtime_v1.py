import copy
import unittest
from unittest.mock import patch

from tools import rank_two_live_eight_float_runtime_v1 as runtime
from tools import rank_two_live_eight_float_preflight_v1 as preflight


class FloatRuntimeBindingTests(unittest.TestCase):
    def fixture(self):
        ref = dict(path='numeric', sha256='numeric')
        numeric = dict(corpus=dict(path='corpus', sha256='corpus'),
                       float_parent=dict(path='parent', sha256='parent'))
        report = dict(passed=True, exact_metrics_equal=True, exact_native_predictions=True,
            optimizer_and_parameters_unchanged=True, numeric_plan=ref, **numeric,
            compared_roots=128, training_updates=0,
            validation_root_accounting=dict(frozen=128, evaluated=128, excluded=0, comparable_early=128))
        return report, ref, numeric

    def test_complete_exact_preflight_passes(self):
        runtime.validate_preflight(*self.fixture())

    def test_replaced_parent_bank_or_numeric_plan_is_rejected(self):
        for key in ('numeric_plan', 'corpus', 'float_parent'):
            with self.subTest(key=key):
                report, ref, numeric = self.fixture()
                report[key] = {}
                with self.assertRaises(PermissionError):
                    runtime.validate_preflight(report, ref, numeric)

    def test_changed_metrics_native_parity_or_optimizer_is_rejected(self):
        for key in ('exact_metrics_equal', 'exact_native_predictions', 'optimizer_and_parameters_unchanged'):
            with self.subTest(key=key):
                report, ref, numeric = self.fixture()
                report[key] = False
                with self.assertRaises(PermissionError):
                    runtime.validate_preflight(report, ref, numeric)

    def test_prior_quarantine_or_insufficient_early_coverage_is_rejected(self):
        for changed in ('compared_roots', 'training_updates', 'accounting', 'early'):
            with self.subTest(changed=changed):
                report, ref, numeric = self.fixture()
                if changed == 'compared_roots': report[changed] = 127
                elif changed == 'training_updates': report[changed] = 1
                elif changed == 'accounting': report['validation_root_accounting'].update(evaluated=127, excluded=1)
                else: report['validation_root_accounting']['comparable_early'] = 99
                with self.assertRaises(PermissionError):
                    runtime.validate_preflight(report, ref, numeric)

    def test_numeric_recipe_binds_current_parent_seed_and_closed_inputs(self):
        report, ref, numeric = self.fixture()
        declaration_ref = dict(path='declaration', sha256='declaration')
        numeric.update(producer={}, profile='dd8', phase_mode='float', float_maximum=80,
            fresh_validation={}, seed=2026101234, inputs=dict(declaration=declaration_ref))
        plan = dict(declaration=declaration_ref)
        state = dict(live_eight_continuation_04_declaration=declaration_ref,
            live_eight_next_corpus=numeric['corpus'], live_eight_next_fresh_validation={},
            pending_exposures=[], active_claims={})
        declaration = dict(float_parent=numeric['float_parent'], train_seed=2026101234,
                           architecture=[6301, 8, 8, 1])
        with patch.object(preflight.c, 'record', return_value={}), \
             patch.object(preflight.c, 'verify', return_value='declaration'), \
             patch.object(preflight.c, 'read', return_value=declaration):
            preflight.check_numeric(plan, numeric, state)
            for changed in ('parent', 'seed', 'pending'):
                altered, current = copy.deepcopy(numeric), copy.deepcopy(state)
                if changed == 'parent': altered['float_parent'] = {}
                elif changed == 'seed': altered['seed'] += 1
                else: current['pending_exposures'] = ['unclosed']
                with self.subTest(changed=changed), self.assertRaises(PermissionError):
                    preflight.check_numeric(plan, altered, current)


if __name__ == '__main__':
    unittest.main()
