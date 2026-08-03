#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_dir="$(cd -- "${script_dir}/.." && pwd)"

build_dir="${PAPERSOCCER_BUILD_DIR:-${repository_dir}/build/native}"
build_type="${PAPERSOCCER_BUILD_TYPE:-Release}"
sanitizers="${PAPERSOCCER_ENABLE_SANITIZERS:-OFF}"

cmake_args=(
  -S "${repository_dir}"
  -B "${build_dir}"
  "-DCMAKE_BUILD_TYPE=${build_type}"
  "-DPAPERSOCCER_ENABLE_SANITIZERS=${sanitizers}"
)

printf 'Configuring %s build in %s\n' "${build_type}" "${build_dir}"
cmake "${cmake_args[@]}"

build_args=(--build "${build_dir}" --parallel)
if [[ -n "${PAPERSOCCER_BUILD_JOBS:-}" ]]; then
  build_args+=("${PAPERSOCCER_BUILD_JOBS}")
fi

cmake "${build_args[@]}"
ctest --test-dir "${build_dir}" --output-on-failure
