#!/usr/bin/env bash
set -euo pipefail

repository_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${RANK5_GATE_BUILD_DIR:-${repository_dir}/build}"
executable="${build_dir}/papersoccer_rank5_derived_gate"

if [[ ! -f "${build_dir}/CMakeCache.txt" ]]; then
  cmake -S "${repository_dir}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release
else
  cmake -S "${repository_dir}" -B "${build_dir}"
fi
cmake --build "${build_dir}" --target papersoccer_rank5_derived_gate

exec "${executable}" "$@"
