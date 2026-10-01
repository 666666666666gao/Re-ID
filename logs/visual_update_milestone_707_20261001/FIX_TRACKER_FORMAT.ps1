$ErrorActionPreference='Stop'
$repo='C:/Users/gb/.trifusion_github_publish_22c3bee'
$prefix=Join-Path $repo 'logs/visual_update_milestone_707_20261001'
$utf8=[Text.UTF8Encoding]::new($false)
$record=Get-Content -LiteralPath 'C:/Users/gb/.codex_tmp/visual_update_milestone_707_document_preparation_20261001.json' -Raw|ConvertFrom-Json -DateKind String
$doc=Join-Path $repo 'docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md'
if((Get-FileHash -LiteralPath $doc).Hash.ToLower() -ne $record.doc_sha256){throw 'Document changed'}
$failure=[ordered]@{recorded_at=(Get-Date).ToString('o');stage='PUBLICATION_STAGING_ONLY';actual_error='Whitespace check failed: both tracker line43 trailing whitespace; PowerShell array addition split one concatenated sentence across lines';actual_staged_count_before_failure=176;training_and_results_unchanged=$true;document_reappend=$false;correction='Join only the malformed tracker sentence; retain original preparation source and raw results'}
[IO.File]::WriteAllText((Join-Path $prefix 'STAGING_FAILURE.json'),($failure|ConvertTo-Json)+"`n",$utf8)
foreach($name in @('EXPERIMENT_TRACKER.md','EXPERIMENT_TRACKER_20261001_707.md')){
    $path=Join-Path $repo ('refine-logs/visual_update_control_v1/'+$name)
    $text=[IO.File]::ReadAllText($path)
    $start=$text.IndexOf('Visual-update roles delta mAP/R1: ')
    $end=$text.IndexOf('Full12 report/audit still pending.')
    if($start -lt 0 -or $end -le $start){throw 'Tracker format mismatch'}
    $line=('Visual-update roles delta mAP/R1: {0:R}/{1:R}. Original all-pair R1>=0 requirement is not met by this pair; no rule retuning.' -f $record.rgbnt100_roles_visual_delta.mAP,$record.rgbnt100_roles_visual_delta.'Rank-1')
    [IO.File]::WriteAllText($path,$text.Substring(0,$start)+$line+"`n"+$text.Substring($end),$utf8)
}
Copy-Item -LiteralPath $PSCommandPath -Destination (Join-Path $prefix 'FIX_TRACKER_FORMAT.ps1')
Set-Location -LiteralPath $repo
git restore --staged -- 'logs/visual_update_milestone_707_20261001' 'docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md' 'MANIFEST.md' 'refine-logs/visual_update_control_v1/EXPERIMENT_TRACKER.md' 'refine-logs/visual_update_control_v1/EXPERIMENT_TRACKER_20261001_707.md'
if($LASTEXITCODE -ne 0){throw 'Owned index reset failed'}
if(@(git diff --cached --name-only).Count){throw 'Unexpected remaining staged paths'}
[ordered]@{corrected_trackers=2;document_sha256=$record.doc_sha256;scope='Publication format only, no reappend or scientific modification'}|ConvertTo-Json
