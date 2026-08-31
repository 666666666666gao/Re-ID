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
vendor_root="${workspace_root}/vendor-sm86"
causal_root="${vendor_root}/causal-conv1d"
mamba_root="${vendor_root}/mamba"
causal_commit="da6dbaa9fd5a919967f14d3fd031da1288ad5025"
mamba_commit="10b5d6358f27966f6a40e4bf0baa17a460688128"
causal_patch_sha256="cfb14993e5ca1f7a087066cb6a59d1bd38a96e4504c5a82f4b9472dd75c084a6"
mamba_patch_sha256="9e376c9d917ce57cf00014052effd98bd1460720f73ca5e5704e5dd12de95102"
causal_setup_sha256="dff4b2a65ce18c31a05d9924ca517733f7388d60364015d4ba1a34d398b6d73a"
mamba_setup_sha256="75e15894ce3fd8514682be4b005174174e32972bfcf6d7d3435ac5909b3601bd"

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

require_sha256() {
    local path="$1"
    local expected="$2"
    local actual
    actual="$(sha256sum "${path}" | cut -d ' ' -f 1)"
    if [[ "${actual}" != "${expected}" ]]; then
        echo "Unexpected SHA-256 for ${path}: ${actual}; expected ${expected}" >&2
        exit 5
    fi
}

require_exact_patch_state() {
    local checkout="$1"
    local patched_file_sha256="$2"
    local status
    status="$(git -C "${checkout}" status --porcelain=v1 --untracked-files=all)"
    if [[ "${status}" != " M setup.py" ]]; then
        echo "Vendor checkout has changes beyond the registered setup.py patch:" >&2
        printf '%s\n' "${status}" >&2
        exit 6
    fi
    if git -C "${checkout}" diff --quiet -- setup.py; then
        echo "Registered setup.py patch is missing in ${checkout}" >&2
        exit 6
    fi
    require_sha256 "${checkout}/setup.py" "${patched_file_sha256}"
    if ! git -C "${checkout}" diff --cached --quiet; then
        echo "Vendor checkout contains staged changes: ${checkout}" >&2
        exit 6
    fi
}


apply_once() {
    local checkout="$1"
    local patch_file="$2"
    if git -C "${checkout}" apply --unidiff-zero --check "${patch_file}" >/dev/null 2>&1; then
        git -C "${checkout}" apply --unidiff-zero "${patch_file}"
    elif ! git -C "${checkout}" apply --unidiff-zero --reverse --check "${patch_file}" >/dev/null 2>&1; then
        echo "Patch is neither applicable nor already applied: ${patch_file}" >&2
        exit 7
    fi
}

clone_if_missing https://github.com/Dao-AILab/causal-conv1d.git v1.6.0 "${causal_root}"
clone_if_missing https://github.com/state-spaces/mamba.git v2.2.6.post3 "${mamba_root}"
require_commit "${causal_root}" "${causal_commit}"
require_commit "${mamba_root}" "${mamba_commit}"
causal_patch="${repo_root}/environment/patches/causal-conv1d-sm86.patch"
mamba_patch="${repo_root}/environment/patches/mamba-sm86.patch"
require_sha256 "${causal_patch}" "${causal_patch_sha256}"
require_sha256 "${mamba_patch}" "${mamba_patch_sha256}"
apply_once "${causal_root}" "${causal_patch}"
apply_once "${mamba_root}" "${mamba_patch}"
require_exact_patch_state "${causal_root}" "${causal_setup_sha256}"
require_exact_patch_state "${mamba_root}" "${mamba_setup_sha256}"

export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CONDA_PREFIX}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
build_jobs="${TRIFUSION_BUILD_JOBS:-6}"

python -c "import torch; assert torch.cuda.get_device_capability() == (8, 6)"
python -c "import transformers; assert transformers.__version__ == '4.45.2', transformers.__version__"
CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS="${build_jobs}" \
    python -m pip install "${causal_root}" --no-deps --force-reinstall \
    --no-build-isolation --no-cache-dir
MAMBA_FORCE_BUILD=TRUE MAX_JOBS="${build_jobs}" \
    python -m pip install "${mamba_root}" --no-deps --force-reinstall \
    --no-build-isolation --no-cache-dir

mkdir -p "${workspace_root}/artifacts"
python "${repo_root}/tools/smoke_mamba.py" \
    --json-out "${workspace_root}/artifacts/mamba_cuda_smoke_sm86.json" \
    --causal-conv1d-commit "${causal_commit}" \
    --mamba-commit "${mamba_commit}" \
    --causal-conv1d-patch-sha256 "${causal_patch_sha256}" \
    --mamba-patch-sha256 "${mamba_patch_sha256}"
