$ErrorActionPreference='Stop'
$target=[DateTimeOffset]::Parse('2026-10-01T17:40:10+08:00')
$record=[ordered]@{started_at=(Get-Date).ToString('o');pid=$PID;target=$target.ToString('o');scope='One scheduled text archive, existing240sec observer/report/queue unchanged';status='WAITING_FOR_ESTIMATED_MILESTONE'}
[IO.File]::WriteAllText('C:/Users/gb/.codex_tmp/visual_update_complete_708_timer_20261001.json',(($record|ConvertTo-Json).Replace("`r`n","`n"))+"`n",[Text.UTF8Encoding]::new($false))
while([DateTimeOffset]::Now -lt $target){
    $remaining=($target-[DateTimeOffset]::Now).TotalSeconds
    Start-Sleep -Seconds ([Math]::Min(60,[Math]::Ceiling($remaining)))
}
& ssh -i C:/Users/gb/.ssh/id_ed25519 -o ProxyCommand=none -o BatchMode=yes -o ConnectTimeout=10 -p 2026 gaob@172.19.12.138 /data/gaob/Re-ID/conda-envs/tri_reid/bin/python -B /data/gaob/Re-ID/Trifusion/.codex_tmp/preserve_visual_update_complete_708_20261001.py
$code=$LASTEXITCODE
$record.status='ONE_ARCHIVE_INVOCATION_EXIT_'+$code
$record.completed_at=(Get-Date).ToString('o')
[IO.File]::WriteAllText('C:/Users/gb/.codex_tmp/visual_update_complete_708_timer_20261001.json',(($record|ConvertTo-Json).Replace("`r`n","`n"))+"`n",[Text.UTF8Encoding]::new($false))
exit $code
