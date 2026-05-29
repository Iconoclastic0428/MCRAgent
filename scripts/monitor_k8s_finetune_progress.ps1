param(
    [string]$Namespace = "nourish-sdsc",
    [string]$Pod = "mcr-transformer-l40-finetune-medhard-20260528b-d99jk",
    [string]$Job = "mcr-transformer-l40-finetune-medhard-20260528b",
    [string]$LogPath = "runs\mcr-transformer-l40-finetune-medhard-20260528b-monitor.jsonl",
    [int]$IntervalSeconds = 900,
    [int]$TotalBatchesPerEpoch = 530,
    [int]$TotalEpochs = 2,
    [int]$PythonPid = 162,
    [switch]$Once
)

$ErrorActionPreference = "Continue"
$logFullPath = if ([System.IO.Path]::IsPathRooted($LogPath)) {
    $LogPath
} else {
    Join-Path (Get-Location) $LogPath
}
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $logFullPath) | Out-Null

function Invoke-KubectlText {
    param([string[]]$KubectlArgs)
    $output = & kubectl @KubectlArgs 2>&1
    return ($output | Out-String).Trim()
}

function Try-ParseJson {
    param([string]$Text)
    try {
        return $Text | ConvertFrom-Json -ErrorAction Stop
    } catch {
        foreach ($line in (($Text | Out-String) -split "`r?`n")) {
            $trimmed = $line.Trim()
            if (-not $trimmed.StartsWith("{")) {
                continue
            }
            try {
                return $trimmed | ConvertFrom-Json -ErrorAction Stop
            } catch {
            }
        }
        return $null
    }
}

function Parse-JsonLogEvents {
    param([string]$LogText)
    $events = @()
    foreach ($line in ($LogText -split "`r?`n")) {
        $trimmed = $line.Trim()
        if (-not ($trimmed.StartsWith("{") -and $trimmed.Contains('"event"'))) {
            continue
        }
        $parsed = Try-ParseJson $trimmed
        if ($null -ne $parsed -and $parsed.event) {
            $events += $parsed
        }
    }
    return $events
}

function Get-ProcessProbe {
    $stat = Invoke-KubectlText @(
        "exec", "-n", $Namespace, $Pod, "--", "cat", "/proc/$PythonPid/stat"
    )
    if ($stat -match "^\s*(\d+)\s+\((.*)\)\s+(\S+)\s+(.*)$") {
        $fieldsAfterState = $Matches[4] -split "\s+"
        if ($fieldsAfterState.Count -ge 21) {
            return @{
                pid = $Matches[1]
                command = $Matches[2]
                exists = $true
                state = $Matches[3]
                utime = [int64]$fieldsAfterState[10]
                stime = [int64]$fieldsAfterState[11]
                rss_pages = [int64]$fieldsAfterState[20]
            }
        }
    }
    return @{ pid = "$PythonPid"; exists = $false; raw = $stat }
}

function Get-GpuProbe {
    $gpu = Invoke-KubectlText @(
        "exec", "-n", $Namespace, $Pod, "--",
        "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"
    )
    $parts = $gpu -split "," | ForEach-Object { $_.Trim() }
    if ($parts.Count -ge 3) {
        $utilization = 0
        $memoryUsed = 0
        $memoryTotal = 0
        if (-not [int]::TryParse($parts[0], [ref]$utilization) -or
            -not [int]::TryParse($parts[1], [ref]$memoryUsed) -or
            -not [int]::TryParse($parts[2], [ref]$memoryTotal)) {
            return @{ raw = $gpu }
        }
        return @{
            utilization_percent = $utilization
            memory_used_mib = $memoryUsed
            memory_total_mib = $memoryTotal
        }
    }
    return @{ raw = $gpu }
}

$lastCompleted = -1
$lastCpuTicks = $null
$lastProgressAt = Get-Date

while ($true) {
    $now = Get-Date
    $podJson = Try-ParseJson (Invoke-KubectlText @("get", "pod", "-n", $Namespace, $Pod, "-o", "json"))
    $jobJson = Try-ParseJson (Invoke-KubectlText @("get", "job", "-n", $Namespace, $Job, "-o", "json"))
    $logs = Invoke-KubectlText @("logs", "-n", $Namespace, $Pod, "--tail=5000")
    $events = Parse-JsonLogEvents $logs
    $progressEvents = @($events | Where-Object { $_.event -eq "train_progress" -or $_.event -eq "validation_snapshot" })
    $latestProgress = $progressEvents | Select-Object -Last 1
    $completedBatches = 0
    if ($latestProgress -and $latestProgress.batches) {
        $completedBatches = (([int]$latestProgress.epoch - 1) * $TotalBatchesPerEpoch) + [int]$latestProgress.batches
    }
    if ($completedBatches -gt $lastCompleted) {
        $lastCompleted = $completedBatches
        $lastProgressAt = $now
    }

    $proc = Get-ProcessProbe
    $gpu = Get-GpuProbe
    $cpuAdvancing = $null
    if ($proc -and $proc.exists -and $proc.utime -ne $null) {
        $ticks = [int64]$proc.utime + [int64]$proc.stime
        if ($lastCpuTicks -ne $null) {
            $cpuAdvancing = $ticks -gt $lastCpuTicks
        }
        $lastCpuTicks = $ticks
    }

    $podPhase = if ($podJson) { $podJson.status.phase } else { "Unknown" }
    $jobFailed = if ($jobJson -and $jobJson.status.failed) { [int]$jobJson.status.failed } else { 0 }
    $jobSucceeded = if ($jobJson -and $jobJson.status.succeeded) { [int]$jobJson.status.succeeded } else { 0 }
    $issue = "none"
    if ($jobFailed -gt 0 -or $podPhase -eq "Failed") {
        $issue = "job_or_pod_failed_requires_intervention"
    } elseif ($podPhase -eq "Pending") {
        $issue = "pod_pending_waiting_for_resources"
    } elseif ($completedBatches -eq 0 -and $proc -and $proc.exists -and $gpu.memory_used_mib -eq 0 -and $cpuAdvancing -eq $true) {
        $issue = "cpu_preprocessing_before_first_batch"
    } elseif ($completedBatches -eq 0 -and $proc -and $proc.exists -and $gpu.memory_used_mib -eq 0 -and $cpuAdvancing -eq $false) {
        $issue = "possible_stall_no_gpu_no_cpu_progress"
    } elseif ($jobSucceeded -gt 0) {
        $issue = "job_completed"
    }

    $record = [ordered]@{
        timestamp = $now.ToString("o")
        namespace = $Namespace
        pod = $Pod
        job = $Job
        pod_phase = $podPhase
        job_failed = $jobFailed
        job_succeeded = $jobSucceeded
        completed_batches = $completedBatches
        total_batches = $TotalBatchesPerEpoch * $TotalEpochs
        latest_progress = $latestProgress
        progress_event_count = $progressEvents.Count
        seconds_since_progress = [int](New-TimeSpan -Start $lastProgressAt -End $now).TotalSeconds
        process = $proc
        gpu = $gpu
        cpu_advancing_since_last_check = $cpuAdvancing
        issue = $issue
    }
    ($record | ConvertTo-Json -Depth 12 -Compress) | Add-Content -Path $logFullPath -Encoding UTF8
    if ($Once) {
        break
    }
    Start-Sleep -Seconds $IntervalSeconds
}
