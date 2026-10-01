$ErrorActionPreference='Stop'
$repo='C:/Users/gb/.trifusion_github_publish_22c3bee'
$archive='C:/Users/gb/.codex_tmp/visual_update_milestone_intake_707_20261001.tar.gz'
$receipt='C:/Users/gb/.codex_tmp/visual_update_milestone_intake_707_20261001.json'
$dest=Join-Path $repo 'logs/visual_update_milestone_707_20261001'
if(Test-Path -LiteralPath $dest){throw 'Milestone intake already exists'}
$r=Get-Content -LiteralPath $receipt -Raw -Encoding UTF8|ConvertFrom-Json -DateKind String
if((Get-FileHash -LiteralPath $archive).Hash.ToLower() -ne $r.archive_sha256 -or (Get-Item -LiteralPath $archive).Length -ne $r.archive_bytes){throw 'Archive SHA/size mismatch'}
New-Item -ItemType Directory -Path $dest|Out-Null
tar -xzf $archive -C $dest
if($LASTEXITCODE -ne 0){throw 'Archive extraction failed'}
foreach($entry in $r.files){
    $p=Join-Path (Join-Path $dest 'raw') $entry.path
    if((Get-Item -LiteralPath $p).Length -ne $entry.bytes -or (Get-FileHash -LiteralPath $p).Hash.ToLower() -ne $entry.sha256){throw "Raw file mismatch $($entry.path)"}
}
if((Get-FileHash -LiteralPath (Join-Path $dest 'SNAPSHOT.json')).Hash.ToLower() -ne $r.snapshot_sha256){throw 'Snapshot mismatch'}
Copy-Item -LiteralPath $receipt -Destination (Join-Path $dest 'INTAKE.json')
Copy-Item -LiteralPath $PSCommandPath -Destination (Join-Path $dest 'RAW_INTAKE.ps1')
$s=Get-Content -LiteralPath (Join-Path $dest 'SNAPSHOT.json') -Raw -Encoding UTF8|ConvertFrom-Json -DateKind String
$f=@($s.endpoints|Where-Object{$_.dataset -eq 'MSVR310' -and $_.variant -eq 'frozen_global_only'})[0]
if($f.child_status -ne 'COMPLETE' -or $f.parent_status -ne 'COMPLETE'){throw 'FrozenMSVRglobal not complete'}
[ordered]@{snapshot_at=$s.observed_at;verified_raw_files=$r.files.Count;parent_complete=$s.formal_parent_complete;new_frozen_globalMSVR=$f.verification;free_disk_bytes=$s.free_disk_bytes}|ConvertTo-Json -Depth 8
