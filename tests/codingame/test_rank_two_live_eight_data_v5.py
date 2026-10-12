import copy
import unittest

from tools import rank_two_live_eight_data_v5 as data
from tests.codingame import test_rank_two_live_eight_data_v4 as previous


class BoundedLiveDataTests(previous.BoundedLiveDataTest):
    def setUp(self):
        self.original = previous.data
        previous.data = data

    def tearDown(self):
        previous.data = self.original


class DeclarationBindingTests(unittest.TestCase):
    def fixture(self):
        plan = dict(output='/campaign/live-data-continuation-05/live-parents-v1',
                    declaration='declaration', owner_thread_id='owner', ancestry='ancestry',
                    windows=['window1', 'window2'], seed=42, sampling={'root_cap': 64},
                    teacher_nodes=256000)
        state = dict(live_eight_continuation_05_declaration='declaration',
                     live_eight_activation='activation', live_eight_ancestry_validated='ancestry')
        declaration = dict(owner_thread_id='owner', activation='activation',
                           all_prior_census_layers_validated='ancestry', windows=plan['windows'],
                           train_seed=42, sampling=plan['sampling'], teacher_nodes=256000,
                           architecture=[6301, 8, 8, 1], outcomes_as_targets=False,
                           train_root_cap=2048, retained_live_train_mass=[.5, .5])
        return plan, state, declaration

    def test_current_frozen_continuation(self):
        data.validate_declaration(*self.fixture())

    def test_rejects_changed_ownership_windows_seed_architecture_and_mass(self):
        for key, value in [('owner_thread_id', 'other'), ('windows', ['other']),
                           ('train_seed', 43), ('architecture', [6301, 12, 8, 1]),
                           ('retained_live_train_mass', [.4, .6]),
                           ('all_prior_census_layers_validated', 'old'),
                           ('outcomes_as_targets', True)]:
            values = copy.deepcopy(self.fixture())
            values[2][key] = value
            with self.subTest(key=key), self.assertRaises(PermissionError):
                data.validate_declaration(*values)

    def test_prior_continuation_cannot_supply_current_approval(self):
        values = self.fixture()
        values[1].pop('live_eight_continuation_05_declaration')
        values[1]['live_eight_continuation_04_declaration'] = 'declaration'
        with self.assertRaises(PermissionError):
            data.validate_declaration(*values)

    def test_unknown_namespace_is_rejected(self):
        with self.assertRaises(PermissionError):
            data.declaration_key('/campaign/arbitrary/live-parents-v1')


if __name__ == '__main__':
    unittest.main()
