#!/usr/bin/env bash
set -euo pipefail

# A clean conda environment can still see ~/.local packages unless user-site
# discovery is disabled explicitly.  Keep both installation and later
# activation isolated from machine-global Python packages.
export PYTHONNOUSERSITE=1
# CANN exposes compiler helper packages through PYTHONPATH.  Their wheel
# metadata contains pseudo-dependencies on Python standard-library modules,
# which makes an otherwise clean `pip check` fail.  Keep package installation
# isolated, then source CANN again immediately before NPU validation.
unset PYTHONPATH

# Usage: ./scripts/install_ascend.sh [CONDA_ENV_NAME]
# Set TORCH_NPU_WHEEL=/absolute/path/to/wheel for an offline/private wheel;
# otherwise torch-npu==2.4.0.post2 is resolved from the configured pip index.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ENV_NAME="${1:-dllm-cache-ascend}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
# Despite the historical filename, this contains pinned top-level
# requirements rather than a hash-locked transitive dependency closure.
LOCK_FILE="${REPO_ROOT}/requirements-ascend.lock"
TORCH_NPU_WHEEL="${TORCH_NPU_WHEEL:-}"
PREFLIGHT_MIN_DEVICES="${PREFLIGHT_MIN_DEVICES:-8}"
PREFLIGHT_RUN_HCCL="${PREFLIGHT_RUN_HCCL:-1}"

if ! command -v conda >/dev/null 2>&1; then
    echo "error: conda is required to create the isolated Ascend environment" >&2
    exit 1
fi

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
    echo "error: conda environment '${ENV_NAME}' already exists" >&2
    echo "remove it explicitly or choose another name; this script will not mutate it" >&2
    exit 1
fi

echo "Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}"
conda create --yes --name "${ENV_NAME}" "python=${PYTHON_VERSION}" pip
conda env config vars set --name "${ENV_NAME}" PYTHONNOUSERSITE=1

echo "Installing the pinned Ascend inference environment"
mapfile -t LOCKED_REQUIREMENTS < <(
    awk 'NF && $0 !~ /^#/ && $0 !~ /^torch-npu==/' "${LOCK_FILE}"
)
conda run --no-capture-output --name "${ENV_NAME}" \
    python -m pip install "${LOCKED_REQUIREMENTS[@]}"

if [[ -n "${TORCH_NPU_WHEEL}" ]]; then
    if [[ ! -f "${TORCH_NPU_WHEEL}" ]]; then
        echo "error: TORCH_NPU_WHEEL does not exist: ${TORCH_NPU_WHEEL}" >&2
        exit 1
    fi
    echo "Installing torch_npu from the supplied local wheel"
    conda run --no-capture-output --name "${ENV_NAME}" \
        python -m pip install --no-deps "${TORCH_NPU_WHEEL}"
else
    echo "Installing the pinned torch_npu wheel from the configured pip index"
    conda run --no-capture-output --name "${ENV_NAME}" \
        python -m pip install --no-deps "torch-npu==2.4.0.post2"
fi
conda run --no-capture-output --name "${ENV_NAME}" \
    python -c 'import importlib.metadata as m; actual = m.version("torch-npu"); expected = "2.4.0.post2"; assert actual == expected, f"torch-npu: expected {expected}, got {actual}"'
conda run --no-capture-output --name "${ENV_NAME}" python -m pip check

echo "Running the Ascend preflight checks"
if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
PREFLIGHT_ARGS=(--min-devices "${PREFLIGHT_MIN_DEVICES}")
if [[ "${PREFLIGHT_RUN_HCCL}" == "1" ]]; then
    PREFLIGHT_ARGS+=(--run-hccl)
fi
conda run --no-capture-output --name "${ENV_NAME}" \
    python "${SCRIPT_DIR}/preflight_ascend.py" "${PREFLIGHT_ARGS[@]}"

echo "Environment '${ENV_NAME}' is ready. Model weights were not downloaded."
echo "Run: conda run -n ${ENV_NAME} ${SCRIPT_DIR}/download_models_ascend.sh"
