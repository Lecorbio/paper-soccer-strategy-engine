"""Verify the public release identity and standalone opening response."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / 'submissions/codingame/releases/20261009-rank2'


class RankTwoRelease(unittest.TestCase):
    def test_published_source_and_screenshot_match_manifest(self):
        manifest = json.loads((RELEASE / 'manifest.json').read_text())
        source = (RELEASE / 'submission.cpp').read_bytes()
        source.decode('ascii')
        self.assertEqual(len(source), manifest['source']['ascii_characters'])
        self.assertLess(len(source), 100000)
        self.assertEqual(hashlib.sha256(source).hexdigest(), manifest['source']['sha256'])
        self.assertEqual(source, (RELEASE.parent / '20260927-rank3/submission.cpp').read_bytes())
        image = (RELEASE / 'leaderboard.png').read_bytes()
        self.assertTrue(image.startswith(b'\x89PNG\r\n\x1a\n'))
        self.assertEqual(hashlib.sha256(image).hexdigest(), manifest['screenshot']['sha256'])
        self.assertEqual(image, (ROOT / 'web/assets/codingame-rank2-20261009.png').read_bytes())
        result = manifest['best_recorded_result']
        self.assertEqual((result['rank'], result['players']), (2, 212))
        self.assertTrue(result['historical_snapshot'])
        self.assertFalse(result['new_formal_qualification'])

    def test_standalone_release_compiles_and_answers_opening(self):
        compiler = shutil.which(os.environ.get('CXX', 'clang++'))
        self.assertIsNotNone(compiler, 'A C++20 compiler is required')
        with tempfile.TemporaryDirectory(prefix='papersoccer-rank2-') as directory:
            binary = str(Path(directory) / 'rank2')
            subprocess.run([compiler, '-std=c++20', '-O3', '-DNDEBUG',
                            str(RELEASE / 'submission.cpp'), '-o', binary],
                           check=True, capture_output=True, text=True, timeout=120)
            result = subprocess.run([binary], input='0\n0\n-\n',
                                    capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        actions = result.stdout.splitlines()
        self.assertEqual(len(actions), 1)
        self.assertRegex(actions[0], r'^[0-7]$')


if __name__ == '__main__':
    unittest.main()
