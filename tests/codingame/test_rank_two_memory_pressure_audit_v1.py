import unittest
from tools.rank_two_memory_pressure_audit_v1 import classify


class PressureGateTests(unittest.TestCase):
    def test_only_normal_pressure_is_admissible(self):
        self.assertEqual(classify(0, '1\n'), 'normal')
        for value in ['2', '4', '', 'unrecognized']:
            self.assertEqual(classify(0, value), 'pressure')
        self.assertEqual(classify(1, '1'), 'unknown')


if __name__ == '__main__':
    unittest.main()
