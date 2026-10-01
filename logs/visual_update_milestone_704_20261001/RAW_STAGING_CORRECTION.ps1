$ErrorActionPreference='Stop'
$repo='C:/Users/gb/.trifusion_github_publish_22c3bee'
Set-Location -LiteralPath $repo
if((git rev-parse HEAD).Trim() -ne '02b22ba43fbc1ae274fc942fe12b59b03f5f3b99'){throw 'Unexpected HEAD'}
$prefix='logs/visual_update_milestone_704_20261001'
if(@(git diff --cached --name-only).Count -ne 116){throw 'Unexpected partial staging'}
$record=[ordered]@{recorded_at=(Get-Date).ToString('o');scope='Publication-only correction before commit; no experimental source, runtime, metric or document-byte change';original_failure='Original704staging retained dated703tracker path, so116changed paths instead of117';action='Stage actual dated704tracker and correction provenance; verify every final owned raw blob/mode'}
[IO.File]::WriteAllText("$repo/$prefix/PUBLICATION_CORRECTION.json",(($record|ConvertTo-Json).Replace("`r`n","`n"))+"`n",[Text.UTF8Encoding]::new($false))
Copy-Item -LiteralPath $PSCommandPath -Destination "$prefix/RAW_STAGING_CORRECTION.ps1"
$owned=@('docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md','MANIFEST.md','refine-logs/visual_update_control_v1/EXPERIMENT_TRACKER_20261001_704.md','refine-logs/visual_update_control_v1/EXPERIMENT_TRACKER.md')
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
foreach($e in $entries){if((git ls-files --stage -- $e.path).Trim() -ne ($e.mode+' '+$e.git_blob+" 0`t"+$e.path)){throw "Blob/mode mismatch $($e.path)"}}
git -c core.whitespace=cr-at-eol diff --cached --check
if($LASTEXITCODE -ne 0){throw 'Whitespace check failed'}
[IO.File]::WriteAllText('C:/Users/gb/.codex_tmp/visual_update_milestone_704_staged_raw_blobs_20261001.json',(([ordered]@{recorded_at=(Get-Date).ToString('o');changed_blobs=$entries.Count;entries=$entries}|ConvertTo-Json -Depth 5).Replace("`r`n","`n"))+"`n",[Text.UTF8Encoding]::new($false))
[ordered]@{staged_blobs=$entries.Count;scope='Owned704text only; all raw blob/modes verified; unrelated dirty paths excluded'}|ConvertTo-Json
