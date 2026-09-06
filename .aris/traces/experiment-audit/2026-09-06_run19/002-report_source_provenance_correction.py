#!/usr/bin/env python3
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
RUN19 = ROOT / '.aris/traces/experiment-audit/2026-09-06_run19'
MD_PATH = ROOT / 'EXPERIMENT_AUDIT_MSVR310_TRIFUSION_TERMINAL.md'
JSON_PATH = ROOT / 'EXPERIMENT_AUDIT_MSVR310_TRIFUSION_TERMINAL.json'
MANIFEST_PATH = RUN19 / 'input_manifest.json'
DISPATCH_PATH = RUN19 / 'dispatch_observation.json'


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def file_info(path: Path) -> dict[str, object]:
    return {
        'path': str(path.relative_to(ROOT)),
        'bytes': path.stat().st_size,
        'sha256': sha256_file(path),
    }


def recheck_manifest() -> dict[str, object]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
    mismatches = []
    for item in manifest['files']:
        path = ROOT / item['path']
        actual = sha256_file(path)
        actual_bytes = path.stat().st_size
        if actual != item['sha256'] or actual_bytes != item['bytes']:
            mismatches.append({
                'path': item['path'],
                'expected_sha256': item['sha256'],
                'actual_sha256': actual,
                'expected_bytes': item['bytes'],
                'actual_bytes': actual_bytes,
            })
    return {
        'manifest_path': str(MANIFEST_PATH.relative_to(ROOT)),
        'manifest_sha256': sha256_file(MANIFEST_PATH),
        'expected_manifest_sha256': 'af1df38e530edd0a7702d7fdd6544b11efa1e317a8697d0a3e02edd9ce5a9059',
        'matches_expected_manifest_sha256': sha256_file(MANIFEST_PATH) == 'af1df38e530edd0a7702d7fdd6544b11efa1e317a8697d0a3e02edd9ce5a9059',
        'file_count': len(manifest['files']),
        'bytes_total': sum(int(item['bytes']) for item in manifest['files']),
        'mismatch_count': len(mismatches),
        'mismatches': mismatches,
    }


def replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        raise AssertionError(f'missing old text: {old[:120]}')
    return text.replace(old, new, 1)


def main() -> int:
    generated_at = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
    before = {'md': file_info(MD_PATH), 'json': file_info(JSON_PATH)}
    dispatch = json.loads(DISPATCH_PATH.read_text(encoding='utf-8'))
    manifest_recheck = recheck_manifest()
    if manifest_recheck['mismatch_count'] != 0 or not manifest_recheck['matches_expected_manifest_sha256']:
        raise AssertionError(manifest_recheck)

    md = MD_PATH.read_text(encoding='utf-8')
    md = replace_once(
        md,
        '- `.aris/traces/experiment-audit/2026-09-06_run19/dispatch_observation.json` was not present. The request metadata shows requested model/reasoning, but that is not independent backend attestation.',
        '- Initial audit observation: `.aris/traces/experiment-audit/2026-09-06_run19/dispatch_observation.json` was not present when I first checked. It now exists and records `recorded_at=2026-09-06T09:21:14.338716+08:00`, after dispatch, with `resolved_backend_independently_attested=false` and `review_family_scope=GPT-family Type-A; no cross-family attestation`. This corrects the provenance wording only: requested/accepted metadata and delayed `list_agents` observation still do not attest backend identity or cross-family independence.'
    )
    md = replace_once(
        md,
        'The SIM dispatch diagnosis supports the mechanism within its fixed-input profiler scope: original/repeat/restored requires-grad states keep bitwise parity, while `freeze_sim_only`, build-before-load, load-final-roles, and original-after-load states change the SIM result by max 1.9073486328125e-06 (`evidence/trifusion_msvr310_trifusion_v1_sim_operation_parity_diagnosis_20260906.json:76-87,89-143`). The captured PyTorch source shows matmul folding decisions depend on `requires_grad` for the smaller operand and alternative bmm paths exist (`evidence/msvr310_pytorch251_LinearAlgebra.cpp.txt:1925-1958,2111-2118,2157-2166`; `evidence/msvr310_pytorch251_Linear.cpp.txt:180-231`; MHA also gates fast path on `requires_grad`, `evidence/msvr310_sim_operation_pytorch_mha_source_20260906.py.txt:162-175,621-639,667-678`).',
        'The SIM dispatch diagnosis supports the mechanism within its fixed-input profiler scope: original/repeat/restored requires-grad states keep bitwise parity, while `freeze_sim_only`, build-before-load, load-final-roles, and original-after-load states change the SIM result by max 1.9073486328125e-06 (`evidence/trifusion_msvr310_trifusion_v1_sim_operation_parity_diagnosis_20260906.json:76-87,89-143`). The applicable source evidence is the noncontiguous projection `linear()` path: `Tensor linear` begins at `evidence/msvr310_pytorch251_Linear.cpp.txt:73`, reaches `at::matmul(input, weight.t())` at line 111, and then the `matmul` implementation enters `should_fold` / expanded `bmm` dispatch depending on shape, contiguity, and the smaller operand `requires_grad` state (`evidence/msvr310_pytorch251_Linear.cpp.txt:73-120`; `evidence/msvr310_pytorch251_LinearAlgebra.cpp.txt:1925-1958,2030-2075,2108-2120,2157-2166`). The earlier `Linear.cpp:180-231` and `Linear.cpp:820-824` excerpts are einsum/tensordot background and are not evidence for this observed projection path. Captured native MHA self-attention fastpath is blocked earlier by `query is not key` at source line 107; the later MHA `requires_grad` fastpath gate is therefore not the observed cause here (`evidence/msvr310_sim_operation_pytorch_mha_source_20260906.py.txt:100-110,160-175`).'
    )
    md = replace_once(
        md,
        '3. **Remote binary evidence is limited.** The local immutable package has no raw tensor/checkpoint binaries. The 15 stored-array path claims and checkpoint hashes are supported by terminal remote receipts and local text/JSON consistency, not by local binary reload. A stronger future audit package should include local raw binaries or an independently generated text digest sufficient to replay binary-array equality without torch.',
        '3. **Remote binary evidence is limited.** The local immutable package has no raw tensor/checkpoint binaries. The 15 stored-array path claims and checkpoint hashes are supported by terminal remote receipts and local text/JSON consistency, not by local binary reload. This remains a qualification only; I did not request local tensor/image copies and did not require any new tensor runtime.'
    )
    md = replace_once(
        md,
        '4. **Backend reviewer attestation is absent.** `001-terminal.request.json` records requested model `gpt-5.5` and reasoning `xhigh`, but `dispatch_observation.json` was absent; that request metadata is not independent backend attestation (`.aris/traces/experiment-audit/2026-09-06_run19/001-terminal.request.json:2,7-8`).',
        '4. **Backend reviewer attestation remains unproven.** `001-terminal.request.json` records requested model `gpt-5.5` and reasoning `xhigh`; `dispatch_observation.json` now exists, but it was recorded after dispatch and marks `resolved_backend_independently_attested=false`. This supports only delayed root-side dispatch/list-agents provenance, not backend identity or cross-family independence (`.aris/traces/experiment-audit/2026-09-06_run19/001-terminal.request.json:2,7-8`; `.aris/traces/experiment-audit/2026-09-06_run19/dispatch_observation.json:1-10`).'
    )
    md = replace_once(
        md,
        '- `pytorch_linear`: `evidence/msvr310_pytorch251_Linear.cpp.txt:180-231,820-824`',
        '- `pytorch_linear`: `evidence/msvr310_pytorch251_Linear.cpp.txt:73-120`'
    )
    md = replace_once(
        md,
        '- `pytorch_linear_algebra`: `evidence/msvr310_pytorch251_LinearAlgebra.cpp.txt:1925-1958,2111-2118,2157-2166`',
        '- `pytorch_linear_algebra`: `evidence/msvr310_pytorch251_LinearAlgebra.cpp.txt:1925-1958,2030-2075,2108-2120,2157-2166`'
    )
    md = replace_once(
        md,
        '- `pytorch_mha`: `evidence/msvr310_sim_operation_pytorch_mha_source_20260906.py.txt:162-175,621-639,667-678`',
        '- `pytorch_mha`: `evidence/msvr310_sim_operation_pytorch_mha_source_20260906.py.txt:100-110,160-175,621-639,667-678`'
    )
    md += '\n## Report-only correction round 2\n\n'
    md += f'- Correction generated: {generated_at} (Asia/Shanghai).\n'
    md += '- Preserved verdicts and all numerical replay values. No 3000-query replay, 780-step replay, training, torch/model runtime, tensor/image library, network, or remote command was run.\n'
    md += f'- Rechecked immutable manifest inputs without replay: {manifest_recheck["file_count"]} files, {manifest_recheck["bytes_total"]:,} bytes, mismatch_count={manifest_recheck["mismatch_count"]}, manifest_sha256=`{manifest_recheck["manifest_sha256"]}`.\n'
    md += '- Corrected dispatch provenance from current absence to initial absence plus delayed observation.\n'
    md += '- Corrected PyTorch source references for the observed noncontiguous projection `linear -> matmul -> should_fold/mm/bmm` path and made the MHA fastpath non-cause explicit.\n'
    md += '- Removed the prior future-audit wording that could be read as asking for local raw binaries; the remote-binary receipt limit remains only a qualification.\n'

    report = json.loads(JSON_PATH.read_text(encoding='utf-8'))
    report['actual_issues'] = [
        'Scientific support fails all fixed gates; original three-role method should not be claimed as supported on this MSVR310 comparison.',
        'Unrepaired frozen full-model/SIM paths are not bitwise exact to B0; exact-helper boundary must remain explicit.',
        'Local package lacks raw tensor/checkpoint binaries; binary array claims are limited to terminal remote receipts. This is retained as a qualification only, without requesting local tensor/image copies or new tensor runtime.',
        'dispatch_observation.json now exists but was recorded after dispatch and does not attest backend identity or cross-family independence.',
    ]
    report['limits'] = [
        'GPT-family Type-A local text/JSON audit; dispatch_observation.json now exists but is delayed provenance and resolved_backend_independently_attested=false.',
        'No local raw .pt/.pth tensor/checkpoint binaries in immutable package; binary claims are remote receipt limited.',
        'No torch/model/tensor/image runtime used in this audit.',
        'B0/M0 prerequisite numerical audits not repeated; only bindings inspected.',
        'Error census verified descriptively only; no exclusive causal mechanism inferred.',
        'Scope is internal MSVR310 train-split OOF 600/1032, not official 591/1055 test evaluation.',
    ]
    report['dispatch_observation'] = dispatch
    report['line_references']['dispatch_observation'] = '.aris/traces/experiment-audit/2026-09-06_run19/dispatch_observation.json:1-10'
    report['line_references']['pytorch_linear'] = 'evidence/msvr310_pytorch251_Linear.cpp.txt:73-120'
    report['line_references']['pytorch_linear_algebra'] = 'evidence/msvr310_pytorch251_LinearAlgebra.cpp.txt:1925-1958,2030-2075,2108-2120,2157-2166'
    report['line_references']['pytorch_linear_background_not_observed_path'] = 'evidence/msvr310_pytorch251_Linear.cpp.txt:180-231,820-824'
    report['line_references']['pytorch_mha'] = 'evidence/msvr310_sim_operation_pytorch_mha_source_20260906.py.txt:100-110,160-175,621-639,667-678'
    report['correction_round_2'] = {
        'generated_at': generated_at,
        'scope': 'report-only source/provenance correction',
        'changed': [
            'qualified dispatch_observation as initially absent but now present and delayed',
            'corrected Linear.cpp source references to Tensor linear line 73 and at::matmul line 111',
            'made native MHA self-attention fastpath query-is-not-key block explicit and requires_grad fastpath gate non-causal for this observation',
            'removed wording that suggested adding local raw binaries; retained remote-binary receipt limit without requesting copies',
        ],
        'unchanged': [
            'independent verdicts',
            'all numerical replay values',
            'terminal_audit_replay_result.json',
            'terminal_audit_query_outputs.jsonl',
            'terminal_audit_training_steps.jsonl',
            'terminal_audit_epoch_means.jsonl',
            'terminal_audit_hashcheck_pre.jsonl',
            'terminal_audit_hashcheck_post.jsonl',
            '001-prefixed archived round1 files',
        ],
        'manifest_recheck': manifest_recheck,
        'source_excerpt_checks': {
            'linear_function_begins_line': 'evidence/msvr310_pytorch251_Linear.cpp.txt:73',
            'linear_calls_at_matmul_line': 'evidence/msvr310_pytorch251_Linear.cpp.txt:111',
            'matmul_should_fold_requires_grad_line': 'evidence/msvr310_pytorch251_LinearAlgebra.cpp.txt:1953-1958',
            'folded_mm_path': 'evidence/msvr310_pytorch251_LinearAlgebra.cpp.txt:2030-2075',
            'expanded_bmm_path': 'evidence/msvr310_pytorch251_LinearAlgebra.cpp.txt:2157-2166',
            'mha_query_not_key_fastpath_block': 'evidence/msvr310_sim_operation_pytorch_mha_source_20260906.py.txt:107',
            'mha_requires_grad_fastpath_gate_not_observed_cause': 'evidence/msvr310_sim_operation_pytorch_mha_source_20260906.py.txt:162-168',
        },
    }

    MD_PATH.write_text(md, encoding='utf-8', newline='\n')
    JSON_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8', newline='\n')

    after = {'md': file_info(MD_PATH), 'json': file_info(JSON_PATH), 'script': file_info(Path(__file__).resolve())}
    summary = {
        'status': 'PASS_REPORT_ONLY_CORRECTION',
        'generated_at': generated_at,
        'before': before,
        'after': after,
        'manifest_recheck': manifest_recheck,
        'dispatch_observation_recorded_at': dispatch.get('recorded_at'),
        'resolved_backend_independently_attested': dispatch.get('resolved_backend_independently_attested'),
        'review_family_scope': dispatch.get('review_family_scope'),
        'replay_files_recomputed': False,
        'numerical_replay_values_changed': False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
