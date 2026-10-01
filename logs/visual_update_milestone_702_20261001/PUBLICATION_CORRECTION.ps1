$ErrorActionPreference='Stop'
$repo='C:\Users\gb\.trifusion_github_publish_22c3bee';Set-Location -LiteralPath $repo
$utf8=[Text.UTF8Encoding]::new($false)
$doc='docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md'
$prefix='logs/visual_update_milestone_702_20261001'
$raw=[IO.File]::ReadAllBytes((Join-Path $repo $doc))
$oldLength=1934098
$hash=[Security.Cryptography.SHA256]::Create()
if([Convert]::ToHexString($hash.ComputeHash($raw,0,$oldLength)).ToLower() -ne '6aece58171d40990c67881d2cc826bde0ecb98d61086bca658effbd643140578'){throw 'Prior document prefix changed'}
$append=$utf8.GetString($raw,$oldLength,$raw.Length-$oldLength)
$snapshotRaw=[IO.File]::ReadAllText((Join-Path $repo "$prefix/SNAPSHOT.json"),$utf8)
$snapshotISO=[regex]::Match($snapshotRaw,'"observed_at": "([^"]+)"').Groups[1].Value
$oldTime=[string](ConvertFrom-Json $snapshotRaw).observed_at
if(-not $append.Contains($oldTime)){throw 'Expected timestamp conversion not present'}
$append=$append.Replace($oldTime,$snapshotISO).TrimEnd([char[]]@("`r","`n"))+"`n"
$extra=$utf8.GetBytes($append);$all=[byte[]]::new($oldLength+$extra.Length)
[Array]::Copy($raw,$all,$oldLength);[Array]::Copy($extra,0,$all,$oldLength,$extra.Length)
[IO.File]::WriteAllBytes((Join-Path $repo $doc),$all)
[IO.File]::WriteAllBytes('C:\Users\gb\Desktop\document\TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md',$all)
foreach($tracker in @('refine-logs/visual_update_control_v1/EXPERIMENT_TRACKER_20261001_702.md','refine-logs/visual_update_control_v1/EXPERIMENT_TRACKER.md')){
    $tp=Join-Path $repo $tracker;$text=[IO.File]::ReadAllText($tp,$utf8).Replace($oldTime,$snapshotISO)
    [IO.File]::WriteAllText($tp,$text,$utf8)
}
$receipt=[ordered]@{corrected_at=(Get-Date).ToString('o');prior_prefix_bytes_preserved=$oldLength;observed_at_exact_ISO_restored=$snapshotISO;issue='PowerShell DateTime default string formatting removed ISO precision/offset in new document/tracker; initial staged diff check also rejected one added EOF blank line. First correction attempt rejected ToString-versus-cast mismatch before any write; same PowerShell cast as original preparation then used.';correction='Only new document/tracker timestamp and document trailing blank line corrected; original1934098bytes and all raw results unchanged.';training_or_reporting_replay=$false;doc_bytes=$all.Length;doc_sha256=(Get-FileHash -LiteralPath $doc).Hash.ToLower()}
[IO.File]::WriteAllText((Join-Path $repo "$prefix/PUBLICATION_CORRECTION.json"),($receipt|ConvertTo-Json)+"`n",$utf8)
Copy-Item -LiteralPath $PSCommandPath -Destination (Join-Path $repo "$prefix/PUBLICATION_CORRECTION.ps1")
$owned=@($doc,'MANIFEST.md','refine-logs/visual_update_control_v1/EXPERIMENT_TRACKER_20261001_702.md','refine-logs/visual_update_control_v1/EXPERIMENT_TRACKER.md')
$owned+=@(Get-ChildItem -LiteralPath $prefix -File -Recurse|ForEach-Object{[IO.Path]::GetRelativePath($repo,$_.FullName).Replace('\','/')}|Sort-Object)
$current=@(git diff --cached --name-only)
if(@($current|Where-Object{$_ -notin $owned}).Count){throw 'Unrelated staged paths'}
$objects=@($owned|git hash-object -w --no-filters --stdin-paths)
if($LASTEXITCODE -ne 0 -or $objects.Count -ne $owned.Count){throw 'Raw hash failure'}
$lines=@();$entries=@()
for($i=0;$i -lt $owned.Count;$i++){
    $mode=if($i -eq 0){'100755'}else{'100644'}
    $lines+=$mode+' '+$objects[$i]+"`t"+$owned[$i]
    $entries+=[ordered]@{path=$owned[$i];mode=$mode;git_blob=$objects[$i];sha256=(Get-FileHash -LiteralPath $owned[$i]).Hash.ToLower()}
}
$psi=[Diagnostics.ProcessStartInfo]::new();$psi.FileName='git';$psi.WorkingDirectory=$repo;$psi.UseShellExecute=$false
$psi.RedirectStandardInput=$true;$psi.RedirectStandardOutput=$true;$psi.RedirectStandardError=$true;$psi.StandardInputEncoding=$utf8
foreach($a in @('update-index','--add','--index-info')){$psi.ArgumentList.Add($a)}
$p=[Diagnostics.Process]::Start($psi);$p.StandardInput.Write(($lines -join "`n")+"`n");$p.StandardInput.Close()
$out=$p.StandardOutput.ReadToEnd();$err=$p.StandardError.ReadToEnd();$p.WaitForExit()
if($p.ExitCode -ne 0 -or $err){throw "Index failure $err"}
$staged=@(git diff --cached --name-only)
if($staged.Count -ne $owned.Count -or @(Compare-Object ($staged|Sort-Object) ($owned|Sort-Object)).Count){throw 'Unexpected staged paths'}
foreach($e in $entries){if((git ls-files --stage -- $e.path).Trim() -ne ($e.mode+' '+$e.git_blob+" 0`t"+$e.path)){throw "Blob/mode mismatch $($e.path)"}}
git -c core.whitespace=cr-at-eol,-blank-at-eol diff --cached --check
if($LASTEXITCODE -ne 0){throw 'Whitespace check failed'}
[IO.File]::WriteAllText('C:\Users\gb\.codex_tmp\visual_update_milestone_702_staged_raw_blobs_20261001.json',([ordered]@{recorded_at=(Get-Date).ToString('o');changed_blobs=$entries.Count;entries=$entries}|ConvertTo-Json -Depth 5)+"`n",$utf8)
[ordered]@{staged_blobs=$entries.Count;doc_bytes=$all.Length;doc_sha256=$receipt.doc_sha256;scope='Document publication correction only; original prefix and actual experimental evidence retained'}|ConvertTo-Json
