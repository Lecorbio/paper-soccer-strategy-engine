import io
import unittest

from tools import rank_two_live_eight_ancestry_v3 as ancestry


class PlayedWindowScopeTests(unittest.TestCase):
    def fixture(self):
        base = ancestry.c.ROOT / 'results/scope-test-only'
        ids = ['live-eight-008-slot-01', 'live-eight-008-slot-02']
        roots = [base / 'live-eight-v1/attempts' / aid for aid in ids]
        refs = [dict(path=str(root / 'EXPOSURE.json'), sha256=str(index))
                for index, root in enumerate(roots)]
        prior = dict(path='current-prior', sha256='bound-prior')
        plan = dict(base=str(base), scope_kind='declared-played-windows', attempt_ids=ids,
            payload_roots=list(map(str, roots)), payload_files=[], exposures=refs,
            prior_validated_union=prior)
        state = dict(live_eight_declared_windows={aid: {} for aid in ids},
            live_eight_known_windows={aid: dict(all90_archived=True) for aid in ids},
            pending_exposures=refs, live_eight_ancestry_validated=prior)
        return plan, state

    def test_exact_closed_pending_windows_pass(self):
        ancestry.validate_scope(*self.fixture())

    def test_unarchived_or_previously_closed_window_cannot_be_rewalked(self):
        plan, state = self.fixture()
        state['live_eight_known_windows'][plan['attempt_ids'][0]]['all90_archived'] = False
        with self.assertRaises(PermissionError):
            ancestry.validate_scope(plan, state)
        plan, state = self.fixture()
        state['pending_exposures'] = []
        with self.assertRaises(PermissionError):
            ancestry.validate_scope(plan, state)

    def test_arbitrary_payload_root_or_prior_is_rejected(self):
        for field, value in [('payload_roots', [str(ancestry.c.ROOT)]),
                             ('payload_files', [dict(path='old-payload', sha256='unused')]),
                             ('prior_validated_union', dict(path='older', sha256='old'))]:
            with self.subTest(field=field):
                plan, state = self.fixture()
                plan[field] = value
                with self.assertRaises(PermissionError):
                    ancestry.validate_scope(plan, state)

    def test_prior_payload_is_rejected_before_file_access(self):
        normal = ancestry.ScopedNormalizer(io.StringIO(), [ancestry.c.ROOT / 'results/new-only'], [], 10)
        with self.assertRaises(PermissionError):
            normal.bind(dict(path=str(ancestry.c.ROOT / 'results/old-missing.json'),
                             sha256='unused'), parse=True)


if __name__ == '__main__':
    unittest.main()
