#!/usr/bin/env python3
"""Run the shared campaign queue with bounded-memory input identity checks."""
import hashlib
from pathlib import Path
try:
    from . import top_three_queue_v4 as queue
except ImportError:
    import top_three_queue_v4 as queue


def record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path), 'sha256': digest.hexdigest()}


# Retain v4's schema, ownership, resource locks, interruption and admission rules.
queue.record = record
if __name__ == '__main__':
    queue.main()
