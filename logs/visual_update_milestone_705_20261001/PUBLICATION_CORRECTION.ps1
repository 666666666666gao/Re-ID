$ErrorActionPreference='Stop'
$repo='C:/Users/gb/.trifusion_github_publish_22c3bee'
$prefix=Join-Path $repo 'logs/visual_update_milestone_705_20261001'
$doc=Join-Path $repo 'docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md'
if((Get-FileHash -LiteralPath $doc).Hash.ToLower() -ne '9b7f046286605c741bd4d0dadf32b0ca26f1b0bd289eb00d0a5bf3aca13070e6'){throw 'Unexpected already-appended document'}
$old=[IO.File]::ReadAllBytes('C:/Users/gb/.codex_tmp/github_handoff_704_20261001.md')
$all=[IO.File]::ReadAllBytes($doc)
if($old.Length -ne 1945284 -or [Convert]::ToBase64String($all,0,$old.Length) -cne [Convert]::ToBase64String($old)){throw 'Historical prefix changed'}
if((Get-FileHash -LiteralPath 'C:/Users/gb/Desktop/document/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md').Hash.ToLower() -ne '9b7f046286605c741bd4d0dadf32b0ca26f1b0bd289eb00d0a5bf3aca13070e6'){throw 'Desktop mismatch'}
if(Test-Path -LiteralPath (Join-Path $prefix 'PARTIAL_MILESTONE_DERIVATION.json')){throw 'Remaining metadata already exists'}
$s=Get-Content -LiteralPath (Join-Path $prefix 'SNAPSHOT.json') -Raw -Encoding UTF8|ConvertFrom-Json -DateKind String
$prior=Get-Content -LiteralPath "$repo/logs/visual_update_milestone_704_20261001/SNAPSHOT.json" -Raw -Encoding UTF8|ConvertFrom-Json -DateKind String
$accepted=@($s.endpoints|Where-Object{$_.parent_status -eq 'COMPLETE' -and $_.child_status -eq 'COMPLETE'})
$priorKeys=@($prior.endpoints|Where-Object{$_.parent_status -eq 'COMPLETE' -and $_.child_status -eq 'COMPLETE'}|ForEach-Object{$_.dataset+'/'+$_.variant})
$new=@($accepted|Where-Object{($_.dataset+'/'+$_.variant) -notin $priorKeys})
$pairs=@()
foreach($d in @('RGBNT201','RGBNT100','MSVR310')){
    foreach($readout in @('roles','global_only')){
        $left=@($accepted|Where-Object{$_.dataset -eq $d -and $_.variant -eq ('low_lr_'+$readout)})
        $right=@($accepted|Where-Object{$_.dataset -eq $d -and $_.variant -eq ('frozen_'+$readout)})
        if($left.Count -and $right.Count){
            $lv=$left[0].verification;$rv=$right[0].verification
            if(($lv.common_initializer_sha256|ConvertTo-Json -Depth 4 -Compress) -ne ($rv.common_initializer_sha256|ConvertTo-Json -Depth 4 -Compress)){throw 'Common initializer mismatch'}
            $pairs+=[ordered]@{dataset=$d;readout=$readout;delta_map=$lv.metrics.mAP-$rv.metrics.mAP;delta_r1=$lv.metrics.'Rank-1'-$rv.metrics.'Rank-1';status='COMPLETE_PAIR_FULL12_AUDIT_PENDING'}
        }
    }
}
$utf8=[Text.UTF8Encoding]::new($false)
$record=[ordered]@{recorded_at=(Get-Date).ToString('o');snapshot_at=$s.observed_at;new_complete_endpoints=@($new|ForEach-Object{$_.dataset+'/'+$_.variant});C1_partial_pairs=$pairs;scope='Existing full50 completed endpoint text only, no model/array/report replay';doc_bytes=$all.Length;doc_sha256=(Get-FileHash -LiteralPath $doc).Hash.ToLower();prefix_bytes_preserved=$old.Length;full_gates='PENDING_FULL12_AUDIT'}
$json=($record|ConvertTo-Json -Depth 8).Replace("`r`n","`n")+"`n"
[IO.File]::WriteAllText((Join-Path $prefix 'PARTIAL_MILESTONE_DERIVATION.json'),$json,$utf8)
[IO.File]::WriteAllText('C:/Users/gb/.codex_tmp/visual_update_milestone_705_document_preparation_20261001.json',$json,$utf8)
$failure=[ordered]@{recorded_at=$record.recorded_at;scope='Publication-only correction; no experiment/model/metric action';original_exit_code=1;original_location='prepare705line77';error='IO.File.WriteAllText relative path resolved to process cwd C:/Users/gb rather than PowerShell provider cwd';partial_state='Doc/Desktop/tracker/manifest and prep source already written; metadata absent';action='Verify unchanged historical prefix and exact appended document; write remaining metadata with absolute paths only';no_document_reappend=$true;no_neural_replay=$true}
[IO.File]::WriteAllText((Join-Path $prefix 'PUBLICATION_CORRECTION.json'),(($failure|ConvertTo-Json -Depth 5).Replace("`r`n","`n"))+"`n",$utf8)
Copy-Item -LiteralPath $PSCommandPath -Destination (Join-Path $prefix 'PUBLICATION_CORRECTION.ps1')
$record|ConvertTo-Json -Depth 8
