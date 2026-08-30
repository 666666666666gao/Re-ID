#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "tri_reid" ]]; then
    echo "Activate the tri_reid conda environment before running this script." >&2
    exit 2
fi
if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "CONDA_PREFIX is unavailable; the conda environment is not active." >&2
    exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
workspace_root="${TRIFUSION_WORKSPACE_ROOT:-$(cd "${repo_root}/.." && pwd)}"
vendor_root="${workspace_root}/vendor"
causal_root="${vendor_root}/causal-conv1d"
mamba_root="${vendor_root}/mamba"
causal_commit="da6dbaa9fd5a919967f14d3fd031da1288ad5025"
mamba_commit="10b5d6358f27966f6a40e4bf0baa17a460688128"

mkdir -p "${vendor_root}"

clone_if_missing() {
    local url="$1"
    local tag="$2"
    local destination="$3"
    if [[ ! -e "${destination}" ]]; then
        git clone --branch "${tag}" --depth 1 "${url}" "${destination}"
    elif [[ ! -d "${destination}/.git" ]]; then
        echo "Existing vendor path is not a git checkout: ${destination}" >&2
        exit 3
    fi
}

require_commit() {
    local checkout="$1"
    local expected="$2"
    local actual
    actual="$(git -C "${checkout}" rev-parse HEAD)"
    if [[ "${actual}" != "${expected}" ]]; then
        echo "Unexpected commit in ${checkout}: ${actual}; expected ${expected}" >&2
        exit 4
    fi
}

apply_once() {
    local checkout="$1"
    local patch_file="$2"
    if git -C "${checkout}" apply --check "${patch_file}"; then
        git -C "${checkout}" apply "${patch_file}"
    elif ! git -C "${checkout}" apply --reverse --check "${patch_file}"; then
        echo "Patch is neither applicable nor already applied: ${patch_file}" >&2
        exit 5
    fi
}

clone_if_missing https://github.com/Dao-AILab/causal-conv1d.git v1.6.0 "${causal_root}"
clone_if_missing https://github.com/state-spaces/mamba.git v2.2.6.post3 "${mamba_root}"
require_commit "${causal_root}" "${causal_commit}"
require_commit "${mamba_root}" "${mamba_commit}"
apply_once "${causal_root}" "${repo_root}/environment/patches/causal-conv1d-sm89.patch"
apply_once "${mamba_root}" "${repo_root}/environment/patches/mamba-sm89.patch"

export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CONDA_PREFIX}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
build_jobs="${TRIFUSION_BUILD_JOBS:-4}"

python -c "import transformers; assert transformers.__version__ == '4.45.2', transformers.__version__"
CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS="${build_jobs}" \
    python -m pip install "${causal_root}" --no-deps --force-reinstall \
    --no-build-isolation --no-cache-dir
MAMBA_FORCE_BUILD=TRUE MAX_JOBS="${build_jobs}" \
    python -m pip install "${mamba_root}" --no-deps --force-reinstall \
    --no-build-isolation --no-cache-dir

python "${repo_root}/tools/smoke_mamba.py" \
    --json-out "${workspace_root}/artifacts/mamba_cuda_smoke_20260831.json"
