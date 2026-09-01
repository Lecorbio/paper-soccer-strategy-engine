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

python_executable="${PAPERSOCCER_PYTHON:-}"
if [[ -z "${python_executable}" && -x "${repository_dir}/.venv/bin/python" ]]; then
  python_executable="${repository_dir}/.venv/bin/python"
fi
if [[ -n "${python_executable}" ]]; then
  if [[ ! -x "${python_executable}" ]]; then
    printf 'Configured Python is not executable: %s\n' "${python_executable}" >&2
    exit 1
  fi
  cmake_args+=("-DPython3_EXECUTABLE:FILEPATH=${python_executable}")
fi

printf 'Configuring %s build in %s\n' "${build_type}" "${build_dir}"
cmake "${cmake_args[@]}"

build_args=(--build "${build_dir}" --parallel)
if [[ -n "${PAPERSOCCER_BUILD_JOBS:-}" ]]; then
  build_args+=("${PAPERSOCCER_BUILD_JOBS}")
fi

cmake "${build_args[@]}"

ctest_args=(--test-dir "${build_dir}" --output-on-failure)
if [[ -n "${PAPERSOCCER_CTEST_EXCLUDE_LABELS:-}" ]]; then
  ctest_args+=(--label-exclude "${PAPERSOCCER_CTEST_EXCLUDE_LABELS}")
fi

ctest "${ctest_args[@]}"
