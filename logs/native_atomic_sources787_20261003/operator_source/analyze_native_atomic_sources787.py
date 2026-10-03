from datetime import datetime
from pathlib import Path
import hashlib
import json
import urllib.request

base = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
root = base / 'native_atomic_sources787'
record = json.loads((root / 'INTAKE.json').read_bytes())
assert all(row['installed_python_byte_equal'] for name, row in record['files'].items()
           if name.startswith('mamba_ssm/'))
assert len(record['files']) == 6
for name, row in record['files'].items():
    data = (root / 'upstream' / name).read_bytes()
    assert len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256']

url = 'https://raw.githubusercontent.com/pytorch/pytorch/v2.5.1/docs/source/notes/randomness.rst'
with urllib.request.urlopen(url, timeout=30) as response:
    assert response.status == 200
    data = response.read()
path = root / 'references' / 'pytorch-v2.5.1-randomness.rst'
path.parent.mkdir()
path.write_bytes(data)
reference = {'url': url, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}

kernel = (root / 'upstream/csrc/selective_scan/selective_scan_bwd_kernel.cuh').read_text()
cpp = (root / 'upstream/csrc/selective_scan/selective_scan.cpp').read_text()
simple = (root / 'upstream/mamba_ssm/modules/mamba_simple.py').read_text()
interface = (root / 'upstream/mamba_ssm/ops/selective_scan_interface.py').read_text()
captured = base / 'native_original_control785_terminal_intake/_source'
roles = (captured / 'modeling/trifusion/independent_native_roles.py').read_text()
factory = (captured / 'modeling/trifusion/experts/mamba.py').read_text()
assert 'Mamba(d_model=width, d_state=16, d_conv=4, expand=2)' in factory
assert 'assert self.width == 128 and self.anchor_count == 16' in roles
assert 'self.mamba(sequence)' in roles and 'self.mamba(sequence.flip(1))' in roles
assert 'self.d_inner = int(self.expand * self.d_model)' in simple
assert 'B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L)' in interface
assert 'C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L)' in interface
assert 'if B.dim() == 3:' in interface and 'if C.dim() == 3:' in interface
assert 'params.dim_ngroups_ratio = dim / n_groups;' in cpp
assert 'const int n_groups = is_variable_B ? B.size(1) : 1;' in cpp
assert 'dim3 grid(params.batch, params.dim);' in kernel
assert 'const int group_id = dim_id / (params.dim_ngroups_ratio);' in kernel
assert 'gpuAtomicAdd(dB_cur + i * kNThreads, dB_vals[i])' in kernel
assert 'gpuAtomicAdd(dC_cur + i * kNThreads, dC_vals[i])' in kernel
assert 'dB.to(B.dtype()), dC.to(C.dtype())' in cpp

measured = json.loads((base / 'native_original_control785_terminal_intake/primary/MEASURED.json').read_bytes())
assert measured['witness_samples'] == 32 and measured['gates']['gradients'] is False
assert sum(not row['fixed_gate_pass'] for row in measured['gradients'].values()) == 83
assert measured['optimizer_updates'] == measured['weights_generated'] == 0

result = {
    'at': datetime.now().astimezone().isoformat(),
    'status': 'SOURCE_RISK_IDENTIFIED_NOT_RUNTIME_CAUSE',
    'installed_python_byte_equal': True,
    'project_source': 'native_original_control785_terminal_intake/_source',
    'upstream_version': 'v2.2.6.post3',
    'conditional_kernel_geometry': {'witness_batch': 32, 'sequence_length': 48, 'd_model': 128,
        'expand': 2, 'd_inner': 256, 'd_state': 16, 'B_C_groups': 1,
        'blocks_per_selective_scan_if_this_upstream_kernel_is_used': 8192,
        'channel_blocks_addressing_each_variable_B_C_gradient_element': 256},
    'original_repeat_failed_gradients': 83,
    'localized_source_repair': None,
    'compiled_binary_provenance_verified': False,
    'runtime_branch_or_intermediate_backward_captured': False,
    'authorized_action': 'Continue source/primary analysis only; retain registered stop. No GPU retry, '
        'new diagnostic arm, precision/gate change, dependency upgrade, M0 or formal training.',
    'reference': reference,
    'new_gpu_runs': 0, 'new_weights': 0,
}
(root / 'ANALYSIS.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
(root / 'SOURCE_ANALYSIS.md').write_text('''# Native backward: versioned dependency source analysis

**SOURCE_RISK_IDENTIFIED_NOT_RUNTIME_CAUSE**. This supplement narrows the source-level possibilities after the sealed original/native repeat failed 83/295 gradient comparisons. It does not invalidate the failure, identify a unique runtime cause, or authorize a repair or another GPU trial. The three-dataset scientific goal remains ACTIVE_UNMET.

## What was newly verified

Six official Mamba `v2.2.6.post3` source texts were downloaded on 2026-10-03. The installed `mamba_simple.py` and `selective_scan_interface.py`, acquired separately from the actual Python 3.10 environment in section786, match those official files byte for byte. The intake retains exact sizes, existing hashes, URLs and timestamps. This verifies the Python text version; it does **not** establish which sources or compiler flags produced the installed CUDA extension.

The as-launched `IndependentNativeRoles` has 16 anchors, width128, and forward/reverse calls to the same production Mamba. The factory passes d_model128, d_state16, d_conv4, expand2. Thus the source-derived internal width is256 and the regional sequence length48. This is a spatial/modality sequence, not video time.

Both ordinary Mamba forward branches lead to `selective_scan_cuda`. The fast branch passes input-dependent B/C and expands them to `batch × 1 × 16 × length`; the non-fast branch also generates input-dependent B/C, and `SelectiveScanFn` inserts the same single group. No inference cache is supplied by the project. The fast-path condition additionally depends on causal-conv availability; its actual value was not captured. The conclusion that both candidate branches use single-group selective scan does not require claiming which branch ran. See [versioned Mamba forward](https://github.com/state-spaces/mamba/blob/v2.2.6.post3/mamba_ssm/modules/mamba_simple.py#L143-L204) and [interface](https://github.com/state-spaces/mamba/blob/v2.2.6.post3/mamba_ssm/ops/selective_scan_interface.py#L219-L264).

## Concrete source-level reduction risk

The corresponding official C++ backward sets `n_groups = B.size(1)`, then `dim_ngroups_ratio = dim / n_groups`. Its kernel launches one block for each batch/channel pair. Variable dB/dC pointers depend on batch and **group**, whereas block identity also includes channel. With one group and internal width256, all256 channel blocks for a given sample address the same variable-B/C gradient elements. The CUDA source accumulates them with floating-point `gpuAtomicAdd`. It also accumulates dA/dD/delta-bias across batch blocks. For the sealed 32-sample witness, the conditional launch geometry is32×256 blocks. These are source-derived counts, not a measured kernel trace. See [C++ parameter setup](https://github.com/state-spaces/mamba/blob/v2.2.6.post3/csrc/selective_scan/selective_scan.cpp#L87-L98), [kernel addressing](https://github.com/state-spaces/mamba/blob/v2.2.6.post3/csrc/selective_scan/selective_scan_bwd_kernel.cuh#L117-L139), and [atomic accumulation](https://github.com/state-spaces/mamba/blob/v2.2.6.post3/csrc/selective_scan/selective_scan_bwd_kernel.cuh#L307-L321).

Floating-point addition is order-dependent, and this kernel does not impose a fixed inter-block summation order. Therefore this is a concrete potential source of backward variation, rather than an invented general CUDA concern. It is still **not evidence that these atomics caused the observed 83 failures**. Reference: [NVIDIA floating-point arithmetic, Operations and Accuracy](https://docs.nvidia.com/cuda/floating-point/index.html#operations-and-accuracy).

The versioned C++ source casts accumulated dB/dC back to their input dtype before returning them. The Python fast backward places these gradients in `dx_dbl`, applies the x-projection backward and returns a gradient to the Mamba input. A difference could therefore propagate toward the shared CLIP path. No intermediate values were saved to show that this occurred, or quantify FP16 rounding amplification. It must remain a hypothesis; it cannot explain the Signal/adapters failure counts by assertion.

## Limits and excluded shortcuts

The measured control had identical raw/fused/global/head outputs and loss, matching buffers/RNG, and no parameter updates. Roles/detail gradients passed their fixed tolerances while Signal/adapters contained the83 failures. Small downstream differences do not by themselves establish where the first difference began. Existing summaries contain maxima and allclose results, not full historical gradients or pre-forward RNG states. Neither 90−83 nor the category maxima isolate a partition-specific effect.

Actual native sampling uses the FP32 all-patch override, so the old grid-sampler explanation remains excluded. A cuDNN deterministic flag concerns convolution; matching random states and that flag do not establish global deterministic execution. The [PyTorch v2.5.1 reproducibility note](https://github.com/pytorch/pytorch/blob/v2.5.1/docs/source/notes/randomness.rst#L63-L124) distinguishes RNG, algorithm selection and deterministic operations. No backend flag was changed. Disabling Mamba fast mode alone would still leave the selective-scan extension in the candidate path. Changing checkpoint_lvl would not remove its atomic reductions. An upgrade or a reference scan replacement would change the dependency/execution route and is not a localized repair already justified by the saved failure.

## Decision and missing evidence

Retain **STOP_GPU_PARITY** and the failed V5/original-control artifacts. Production code, fixed gates, precision, batch, scaler, seed and dependencies are unchanged. There is no new GPU run, scorer call, optimizer update or weight. No ordinary M0 or formal50 run is permitted by this supplement.

The saved artifacts cannot answer which backward boundary first diverged. A future diagnosis would need actual extension/build provenance and separately captured input/output/upstream/downstream gradients at the two Mamba calls, followed by the relevant shared-backbone boundary. This describes the missing evidence, **not a registered or authorized GPU experiment**. Before any runtime work, the original stop would need a concrete, reviewable amendment; a dependency risk alone is not permission to retry or weaken the gate. Source analysis is now bounded: no uniquely supported repair has been found.
''', encoding='utf-8')
(root / 'MANIFEST.md').write_text('# Source-only supplement manifest\n\n- INTAKE.json: six official versioned Mamba text files and installed-Python equality.\n- upstream/: those exact source texts.\n- references/pytorch-v2.5.1-randomness.rst: official versioned reproducibility note.\n- ANALYSIS.json and SOURCE_ANALYSIS.md: source-derived risk and unchanged stop boundary.\n- collect_native_atomic_sources787.py / analyze_native_atomic_sources787.py: operator source copied at publication.\n\nNo runtime/kernel measurement or repair is represented by this manifest.\n', encoding='utf-8')
print(json.dumps({key: value for key, value in result.items()
                  if key in ('at', 'status', 'installed_python_byte_equal', 'localized_source_repair',
                             'compiled_binary_provenance_verified', 'new_gpu_runs', 'new_weights')}))
