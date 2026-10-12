"""Record exact macOS pressure observations without weakening the queue gate."""
from pathlib import Path
import subprocess

from tools import rank_two_live_eight_v1 as c


def classify(returncode, stdout):
    if returncode != 0:
        return 'unknown'
    return 'normal' if stdout.strip() == '1' else 'pressure'


class Audit:
    def __init__(self, output):
        self.output, self.number = Path(output), 0

    def __call__(self):
        self.number += 1
        try:
            result = subprocess.run(['/usr/sbin/sysctl', '-n', 'kern.memorystatus_vm_pressure_level'],
                capture_output=True, text=True, timeout=5)
            observation = dict(returncode=result.returncode, stdout=result.stdout,
                               stderr=result.stderr, status=classify(result.returncode, result.stdout))
        except (OSError, subprocess.SubprocessError) as error:
            observation = dict(status='unknown', error=type(error).__name__ + ': ' + str(error))
        ref = c.immutable(self.output / ('%06d.json' % self.number),
            dict(observation, observed_at=c.now().isoformat(), producer=c.record(__file__)))
        c.atomic(self.output / 'HEAD.json', dict(observation, receipt=ref))
        return observation['status']
