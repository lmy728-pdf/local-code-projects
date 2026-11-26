param([switch]$Once)
$ErrorActionPreference = 'Stop'
$g5Code = Split-Path -Parent $PSScriptRoot
$g5Folder = Join-Path $g5Code '_work\g5_rf'
$Host.UI.RawUI.WindowTitle = 'G5 运行状态 - 每 10 秒刷新'
do {
    try {
        $g5Files = @(Get-ChildItem -LiteralPath (Join-Path $g5Folder 'records') -Filter '*.json' -ErrorAction SilentlyContinue)
        $g5Latest = $g5Files | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        $g5Processes = @(Get-CimInstance Win32_Process -Filter "name = 'python.exe'")
        $g5Runners = @($g5Processes | Where-Object { $_.CommandLine -match 'runner.py' -and $_.CommandLine -match '--stage\s+G5(?:\s|$)' })
        $g5ParentIds = @($g5Runners.ProcessId)
        $g5Workers = @($g5Processes | Where-Object { $_.ParentProcessId -in $g5ParentIds -and $_.CommandLine -match 'spawn_main' })
        $g5State = if ($g5Workers.Count -gt 0) {'运行中：计算重复实验'} elseif ($g5Runners.Count -gt 0) {'运行中：启动检查 / 验收 / 汇总 / 出图'} elseif (Test-Path -LiteralPath (Join-Path $g5Folder 'delivery_audit.json')) {'已完成：可查看 CSV、MD 和 PDF 图'} else {'进程已停止：检查下方日志'}
        $g5Snapshot = [pscustomobject]@{time=(Get-Date).ToString('o');status=$g5State;records=$g5Files.Count;target=8000;workers=$g5Workers.Count;runner_pids=$g5ParentIds;latest_record=$g5Latest.Name}
        $g5Snapshot | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'G5_monitor_status.json') -Encoding utf8
        if (-not $Once) { Clear-Host }
        Write-Host '=================== G5 运行状态 ===================' -ForegroundColor Cyan
        Write-Host ('更新时间：' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))
        Write-Host ('状态：' + $g5State) -ForegroundColor Green
        Write-Host ('已落盘：{0} / 8000 ({1:N2}%)    剩余：{2}' -f $g5Files.Count,($g5Files.Count/80),(8000-$g5Files.Count))
        Write-Host ('计算 worker：{0} / 12    主进程：{1}' -f $g5Workers.Count,($g5ParentIds -join ', '))
        if ($g5Latest) { Write-Host ('最近结果：{0}  {1}' -f $g5Latest.LastWriteTime.ToString('HH:mm:ss'),$g5Latest.Name) }
        Write-Host ''
        Write-Host '--- 最近运行日志（程序每完成 100 次更新一条）---'
        $g5LaunchFile = Join-Path $g5Folder 'last_launch.json'
        if (Test-Path -LiteralPath $g5LaunchFile) {
            $g5Launch = Get-Content -LiteralPath $g5LaunchFile -Raw -Encoding utf8 | ConvertFrom-Json
            if (Test-Path -LiteralPath $g5Launch.stdout) { Get-Content -LiteralPath $g5Launch.stdout -Tail 5 -Encoding utf8 }
            if ((Test-Path -LiteralPath $g5Launch.stderr) -and (Get-Item -LiteralPath $g5Launch.stderr).Length -gt 0) {
                Write-Host '--- 最近错误 / 警告 ---' -ForegroundColor Yellow
                Get-Content -LiteralPath $g5Launch.stderr -Tail 6 -Encoding utf8
            }
        }
        Write-Host ''
        Write-Host '每 10 秒刷新；关闭这个窗口只会关闭监测，不会停止 G5。' -ForegroundColor Cyan
        Write-Host ('结果目录：' + (Split-Path -Parent $g5Code))
    } catch { Write-Host ('监测读取失败，下轮重试：' + $_.Exception.Message) -ForegroundColor Yellow }
    if ($Once) { break }
    Start-Sleep -Seconds 10
} while ($true)
