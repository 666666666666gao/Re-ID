$ErrorActionPreference='Stop'
$repo='C:/Users/gb/.trifusion_github_publish_22c3bee'
Set-Location -LiteralPath $repo
if((git rev-parse HEAD).Trim() -ne 'd72b62deb7ef87aef71f690c6a55d4bd707b92ad'){throw 'Unexpected HEAD'}
if(@(git diff --cached --name-only).Count){throw 'Index not empty'}
$prefix='logs/visual_update_complete_708_20261001'
Copy-Item -LiteralPath $PSCommandPath -Destination "$prefix/RAW_STAGING.ps1"
Copy-Item -LiteralPath 'C:/Users/gb/.codex_tmp/sync_visual_update_complete_708_20261001.py' -Destination "$prefix/SYNC_HELPER.py"
$owned=@('docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md','MANIFEST.md','refine-logs/visual_update_control_v1/EXPERIMENT_TRACKER_20261001_708.md','refine-logs/visual_update_control_v1/EXPERIMENT_TRACKER.md')
$owned+=@(Get-ChildItem -LiteralPath 'results/visual_update_control_complete_20261001' -File -Recurse|ForEach-Object{[IO.Path]::GetRelativePath($repo,$_.FullName).Replace('\','/')}|Sort-Object)
$owned+=@(Get-ChildItem -LiteralPath $prefix -File -Recurse|ForEach-Object{[IO.Path]::GetRelativePath($repo,$_.FullName).Replace('\','/')}|Sort-Object)
$objects=@($owned|git hash-object -w --no-filters --stdin-paths)
if($LASTEXITCODE -ne 0 -or $objects.Count -ne $owned.Count){throw 'Raw hash failure'}
$lines=@();$entries=@()
for($i=0;$i -lt $owned.Count;$i++){
    $mode=if($i -eq 0){'100755'}else{'100644'}
    $lines+=$mode+' '+$objects[$i]+"`t"+$owned[$i]
    $entries+=[ordered]@{path=$owned[$i];mode=$mode;git_blob=$objects[$i];sha256=(Get-FileHash -LiteralPath $owned[$i]).Hash.ToLower()}
}
$psi=[Diagnostics.ProcessStartInfo]::new();$psi.FileName='git';$psi.WorkingDirectory=$repo;$psi.UseShellExecute=$false
$psi.RedirectStandardInput=$true;$psi.RedirectStandardOutput=$true;$psi.RedirectStandardError=$true
$psi.StandardInputEncoding=[Text.UTF8Encoding]::new($false)
foreach($a in @('update-index','--add','--index-info')){$psi.ArgumentList.Add($a)}
$p=[Diagnostics.Process]::Start($psi);$p.StandardInput.Write(($lines -join "`n")+"`n");$p.StandardInput.Close()
$out=$p.StandardOutput.ReadToEnd();$err=$p.StandardError.ReadToEnd();$p.WaitForExit()
if($p.ExitCode -ne 0 -or $err){throw "Index failure $err"}
$staged=@(git diff --cached --name-only)
if($staged.Count -ne $owned.Count -or @(Compare-Object ($staged|Sort-Object) ($owned|Sort-Object)).Count){throw 'Unexpected staged paths'}
$indexed=@(git ls-files --stage -- $prefix 'results/visual_update_control_complete_20261001' 'docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md' 'MANIFEST.md' 'refine-logs/visual_update_control_v1/EXPERIMENT_TRACKER_20261001_708.md' 'refine-logs/visual_update_control_v1/EXPERIMENT_TRACKER.md')
if($LASTEXITCODE -ne 0){throw 'Index inventory failure'}
$byPath=@{}
foreach($line in $indexed){$parts=$line.Split("`t",2);$byPath.Add($parts[1],$parts[0])}
foreach($e in $entries){if($byPath[$e.path] -ne ($e.mode+' '+$e.git_blob+' 0')){throw "Blob/mode mismatch $($e.path)"}}
git -c core.whitespace=cr-at-eol diff --cached --check
if($LASTEXITCODE -ne 0){throw 'Whitespace check failed'}
$record=[ordered]@{recorded_at=(Get-Date).ToString('o');changed_blobs=$entries.Count;entries=$entries;scope='Only708owned text/raw blobs, unrelated dirt excluded; index inventory batched'}
[IO.File]::WriteAllText('C:/Users/gb/.codex_tmp/visual_update_complete_708_staged_raw_blobs_20261001.json',(($record|ConvertTo-Json -Depth 5).Replace("`r`n","`n"))+"`n",[Text.UTF8Encoding]::new($false))
[ordered]@{staged_blobs=$entries.Count;all_raw_modes_verified=$true}|ConvertTo-Json
