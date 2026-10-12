$ErrorActionPreference='Stop'
$work='C:/Users/gb/.codex_tmp/independent_evidence_draft/text_semantic_prior901'
if (!(Test-Path -LiteralPath "$work/full_result_review901/.aris/traces/experiment-audit/2026-10-10_full901/001-full901.meta.json")) { throw 'Audit closure missing' }
Get-Content -LiteralPath "$work/retire_failed_msvr_probes902.py" -Encoding utf8 -Raw | & ssh -o ProxyCommand=none -o ProxyJump=none -o BatchMode=yes -i 'C:/Users/gb/.ssh/id_ed25519' -p 2026 gaob@172.19.12.138 /usr/bin/python3 -
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
New-Item -ItemType Directory -Path "$work/runtime/FAILED_MSVR_PROBE_RETIREMENT_902" -ErrorAction Stop | Out-Null
& scp -o ProxyCommand=none -o ProxyJump=none -o BatchMode=yes -i 'C:/Users/gb/.ssh/id_ed25519' -P 2026 gaob@172.19.12.138:/data/gaob/Re-ID/Trifusion/logs/text_semantic_prior_resume901/FAILED_MSVR_PROBE_RETIREMENT_902/QUALIFIED.json gaob@172.19.12.138:/data/gaob/Re-ID/Trifusion/logs/text_semantic_prior_resume901/FAILED_MSVR_PROBE_RETIREMENT_902/DELETIONS.jsonl gaob@172.19.12.138:/data/gaob/Re-ID/Trifusion/logs/text_semantic_prior_resume901/FAILED_MSVR_PROBE_RETIREMENT_902/RESULT.json "$work/runtime/FAILED_MSVR_PROBE_RETIREMENT_902/"
exit $LASTEXITCODE
