#!/usr/bin/env bash
set -euo pipefail

cat >&2 <<'EOF'
The replay-blend Rank5Derived gate is an archived historical experiment.
Rank5DerivedBot is now an immutable 50k, zero-blend profile, so this command
cannot rerun a configurable blend under that identity. Use the source revision
recorded with the archived artifact to reproduce it.
EOF
exit 2
