"""Verify the quantized alpha-beta artifact and standalone opening response."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
BOT = ROOT / 'submissions/codingame/bots/quantized_alpha_beta'


class RankTwoRelease(unittest.TestCase):
    def test_published_source_and_screenshot_match_manifest(self):
        manifest = json.loads((BOT / 'manifest.json').read_text())
        source = (BOT / 'submission.cpp').read_bytes()
        source.decode('ascii')
        self.assertEqual(len(source), manifest['source']['ascii_characters'])
        self.assertLess(len(source), 100000)
        self.assertEqual(hashlib.sha256(source).hexdigest(), manifest['source']['sha256'])
        for filename in ('confirmation.json', 'live-calibration.json', 'current-status.json'):
            evidence = json.loads((BOT / 'history/2026-09-27' / filename).read_text())
            self.assertEqual(evidence['source_sha256'], manifest['source']['sha256'])
        image = (BOT / 'leaderboard.png').read_bytes()
        self.assertTrue(image.startswith(b'\x89PNG\r\n\x1a\n'))
        self.assertEqual(hashlib.sha256(image).hexdigest(), manifest['screenshot']['sha256'])
        self.assertEqual(image, (ROOT / 'web/assets/codingame-rank2-20261009.png').read_bytes())
        result = manifest['best_recorded_result']
        self.assertEqual((result['rank'], result['players']), (2, 212))
        self.assertTrue(result['historical_snapshot'])
        self.assertFalse(result['new_formal_qualification'])

    def test_standalone_release_compiles_and_answers_opening(self):
        supplied_binary = os.environ.get('PAPERSOCCER_QUANTIZED_ALPHA_BETA')
        if supplied_binary:
            self.check_opening(supplied_binary)
            return
        compiler = shutil.which(os.environ.get('CXX', 'clang++'))
        self.assertIsNotNone(compiler, 'A C++20 compiler is required')
        with tempfile.TemporaryDirectory(prefix='papersoccer-rank2-') as directory:
            binary = str(Path(directory) / 'rank2')
            flags = ['-std=c++20', '-O3', '-DNDEBUG']
            if sys.platform == 'darwin' and 'clang' in Path(compiler).name:
                flags += ['-fintegrated-cc1', '-Wl,-ld_classic']
            subprocess.run([compiler, *flags,
                            str(BOT / 'submission.cpp'), '-o', binary],
                           check=True, capture_output=True, text=True, timeout=120)
            self.check_opening(binary)

    def check_opening(self, binary):
        result = subprocess.run([binary], input='0\n0\n-\n',
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        actions = result.stdout.splitlines()
        self.assertEqual(len(actions), 1)
        self.assertRegex(actions[0], r'^[0-7]$')


if __name__ == '__main__':
    unittest.main()
