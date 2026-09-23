[CmdletBinding()]
param([ValidateSet('cpu', 'gpu', 'e2e', 'all')][string]$Stage = 'all')

# Host orchestration only. Python, quality checks and all workloads run in Docker.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $workspace
$runId = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ') + '-' + [guid]::NewGuid().ToString('N').Substring(0, 8)
$project = 'zero-ttt-s01-' + $runId.ToLowerInvariant()
$reportRoot = Join-Path $workspace "tmp/acceptance/s01/$runId"
New-Item -ItemType Directory -Path $reportRoot -Force | Out-Null
$containerReportRoot = "/workspace/tmp/acceptance/s01/$runId"
$utf8 = New-Object System.Text.UTF8Encoding($false)
$baseCompose = @('compose', '-f', 'compose.yaml', '--profile', 'dev')
$e2eCompose = @('compose', '-f', 'compose.yaml', '-f', 'compose.e2e.yaml', '--project-name', $project, '--profile', 'dev')
$report = [ordered]@{
    schema_version = 1; run_id = $runId; selected_stage = $Stage; project = $project
    started_utc = [DateTime]::UtcNow.ToString('o'); finished_utc = $null
    stages = [ordered]@{ preflight = 'not_run'; cpu = 'not_run'; gpu = 'not_run'; e2e = 'not_run' }
    s01_passed = $false; failure = $null; cleanup_command = $null
    commands = [System.Collections.Generic.List[object]]::new()
}
$currentStage = 'preflight'
$environmentStep = $true
$e2eStarted = $false
$probeNames = [System.Collections.Generic.List[string]]::new()

function Write-JsonFile($Value, [string]$Name) {
    [IO.File]::WriteAllText((Join-Path $reportRoot $Name), ($Value | ConvertTo-Json -Depth 50), $utf8)
}

function Save-Report { Write-JsonFile $report 'report.json' }

function Quote-NativeArgument([string]$Value) {
    if ($Value.Length -gt 0 -and $Value -notmatch '[\s"]') { return $Value }
    $escaped = [regex]::Replace($Value, '(\\*)"', '$1$1\"')
    return '"' + [regex]::Replace($escaped, '(\\+)$', '$1$1') + '"'
}

function Invoke-Recorded([string]$Name, [string]$File, [string[]]$CommandArgs) {
    $number = $report.commands.Count + 1
    $prefix = '{0:D3}-{1}' -f $number, $Name
    $entry = [ordered]@{
        name = $Name; stage = $currentStage; executable = $File; arguments = $CommandArgs
        started_utc = [DateTime]::UtcNow.ToString('o'); elapsed_seconds = 0; exit_code = $null
        stdout = "$prefix.stdout.log"; stderr = "$prefix.stderr.log"; pytest_summary = $null
    }
    $report.commands.Add($entry)
    Save-Report
    Write-Host "[$currentStage] $Name"
    $info = New-Object Diagnostics.ProcessStartInfo
    $info.FileName = $File
    $info.Arguments = ($CommandArgs | ForEach-Object { Quote-NativeArgument $_ }) -join ' '
    $info.WorkingDirectory = $workspace
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    $process = New-Object Diagnostics.Process
    $process.StartInfo = $info
    $timer = [Diagnostics.Stopwatch]::StartNew()
    $stdout = [IO.File]::Open((Join-Path $reportRoot $entry.stdout), [IO.FileMode]::Create, [IO.FileAccess]::Write, [IO.FileShare]::Read)
    $stderr = [IO.File]::Open((Join-Path $reportRoot $entry.stderr), [IO.FileMode]::Create, [IO.FileAccess]::Write, [IO.FileShare]::Read)
    try {
        [void]$process.Start()
        $outTask = $process.StandardOutput.BaseStream.CopyToAsync($stdout)
        $errTask = $process.StandardError.BaseStream.CopyToAsync($stderr)
        $process.WaitForExit()
        [void]$outTask.GetAwaiter().GetResult()
        [void]$errTask.GetAwaiter().GetResult()
        $entry.exit_code = $process.ExitCode
    } finally {
        $entry.elapsed_seconds = [Math]::Round($timer.Elapsed.TotalSeconds, 3)
        $stdout.Dispose(); $stderr.Dispose(); $process.Dispose()
        Save-Report
    }
    $output = [IO.File]::ReadAllText((Join-Path $reportRoot $entry.stdout))
    $errors = [IO.File]::ReadAllText((Join-Path $reportRoot $entry.stderr))
    if ($Name -eq 'pytest') {
        $entry.pytest_summary = @($output -split "`n" | Where-Object { $_ -match '\d+ (passed|failed|error|skipped)' }) -join "`n"
    }
    Save-Report
    if ($entry.exit_code -ne 0) {
        $tail = @(($errors + "`n" + $output) -split "`n" | Select-Object -Last 15) -join "`n"
        throw "$Name exited $($entry.exit_code). See $reportRoot/$prefix.*.log`n$tail"
    }
    return $output.Trim()
}

function Invoke-Docker([string]$Name, [string[]]$CommandArgs) {
    return Invoke-Recorded $Name 'docker' $CommandArgs
}

function Invoke-Test([string]$Name, [string[]]$CommandArgs) {
    return Invoke-Docker $Name ($baseCompose + @('run', '--rm', '--no-deps', '-T', 'test') + $CommandArgs)
}

function Invoke-Driver([string]$Phase) {
    $driverArgs = @('python', 'scripts/compose_e2e_test.py', 'run', $Phase, '--profile-id', 's01', '--config', 'configs/acceptance/s01.toml')
    [void](Invoke-Docker "e2e-$Phase" (@('exec', "$project-driver") + $driverArgs))
    [void](Invoke-Docker "evidence-$Phase" @('cp', "${project}-driver:/datasets/work/e2e-$Phase-result.json", (Join-Path $reportRoot "e2e-$Phase.json")))
}

function Assert-GpuIdle {
    $ids = Invoke-Docker 'running-containers' @('ps', '-q')
    if ($ids) {
        $containers = ConvertFrom-Json -InputObject (Invoke-Docker 'running-container-devices' (@('inspect') + @($ids -split '\s+')))
        foreach ($container in $containers) {
            $owner = if ($container.Config.Labels.PSObject.Properties['com.docker.compose.project']) { $container.Config.Labels.'com.docker.compose.project' } else { '' }
            if ($container.HostConfig.DeviceRequests -and $owner -ne $project) {
                throw "GPU is reserved by $($container.Name); no other container was stopped."
            }
        }
    }
    $gpuXml = [xml](Invoke-Recorded 'gpu-processes' 'nvidia-smi' @('-q', '-x'))
    $gpus = $gpuXml.SelectNodes('/nvidia_smi_log/gpu')
    if ($gpus.Count -eq 0) { throw 'nvidia-smi did not return any GPU devices.' }
    foreach ($gpu in $gpus) {
        # WDDM also reports idle C+G desktop contexts as compute applications.
        # Reject compute-only contexts, unknown process types, and measurable load.
        $compute = @($gpu.SelectNodes('processes/process_info') | Where-Object { $_.type -notin @('G', 'C+G') })
        $utilMatch = [regex]::Match([string]$gpu.utilization.gpu_util, '^\d+')
        $memoryMatch = [regex]::Match([string]$gpu.fb_memory_usage.used, '^\d+')
        if (-not $utilMatch.Success -or -not $memoryMatch.Success) { throw 'Cannot determine current GPU load.' }
        $utilization = [int]$utilMatch.Value
        $memoryMiB = [int]$memoryMatch.Value
        if ($compute.Count -gt 0 -or $utilization -gt 5 -or $memoryMiB -gt 512) {
            throw "GPU has compute contexts or active load (utilization=$utilization%, memory=$memoryMiB MiB)."
        }
    }
    [void](Invoke-Recorded 'gpu-device' 'nvidia-smi' @('--query-gpu=name,uuid,memory.total,memory.used,utilization.gpu,driver_version', '--format=csv,noheader'))
}

function Assert-Runtime([string[]]$Ids, [string]$Name) {
    $inspection = ConvertFrom-Json -InputObject (Invoke-Docker "inspect-$Name" (@('inspect') + $Ids))
    $payload = @{ workspace = $workspace; project = $project; e2e = $resolvedE2e; containers = @($inspection | ForEach-Object { $_ }) }
    Write-JsonFile $payload "runtime-$Name.json"
    [void](Invoke-Test "runtime-$Name" @('python', 'scripts/check_s01.py', 'runtime', '--input', "$containerReportRoot/runtime-$Name.json"))
}

function Assert-Probe([string]$Service) {
    $name = "$project-$Service-probe"
    $probeNames.Add($name)
    [void](Invoke-Docker "probe-$Service" ($baseCompose + @('run', '-d', '--no-deps', '--name', $name, $Service, 'python', 'scripts/check_s01.py', 'hold')))
    Assert-Runtime @($name) $Service
    [void](Invoke-Docker "remove-$Service-probe" @('rm', '-f', $name))
    [void]$probeNames.Remove($name)
}

function Run-Cpu {
    $script:currentStage = 'cpu'; $script:environmentStep = $false
    $report.stages.cpu = 'running'
    [void](Invoke-Test 'dependency-check' @('python', '-m', 'pip', 'check'))
    [void](Invoke-Test 'ruff' @('python', '-m', 'ruff', 'check', '.'))
    [void](Invoke-Test 'format' @('python', '-m', 'ruff', 'format', '--check', '.'))
    [void](Invoke-Test 'pyright' @('pyright'))
    [void](Invoke-Test 'pytest' @('python', '-m', 'pytest', '-q', '-W', 'error::starlette.exceptions.StarletteDeprecationWarning'))
    [void](Invoke-Test 'docs' @('python', 'scripts/check_docs.py'))
    [void](Invoke-Test 'contracts' @('python', 'scripts/generate_contracts.py', '--check'))
    [void](Invoke-Test 'diff-check' @('git', '-c', 'safe.directory=/workspace', '-c', 'core.autocrlf=true', '-c', 'core.safecrlf=false', 'diff', '--check'))
    $report.stages.cpu = 'passed'; Save-Report
}

function Run-Gpu {
    $script:currentStage = 'gpu'; $script:environmentStep = $true
    $report.stages.gpu = 'running'
    Assert-GpuIdle
    Assert-Probe 'gpu-smoke'
    $script:environmentStep = $false
    $prefix = $baseCompose + @('run', '--rm', '--no-deps', '-T', 'gpu-smoke', 'python')
    [void](Invoke-Docker 'gpu-driver-fp32' ($prefix + @('scripts/docker_smoke_test.py')))
    [void](Invoke-Docker 'gpu-model' ($prefix + @('scripts/model_smoke_test.py', '--configs', 'configs/acceptance/s01.toml', '--default-optimizer-steps', '1', '--baseline-optimizer-steps', '1')))
    [void](Invoke-Docker 'gpu-selfplay' ($prefix + @('scripts/selfplay_gpu_smoke_test.py', '--config', 'configs/acceptance/s01.toml')))
    $report.stages.gpu = 'passed'; Save-Report
}

function Run-E2e {
    $script:currentStage = 'e2e'; $script:environmentStep = $true
    $report.stages.e2e = 'running'
    [void](Invoke-Docker 'build-services' ($e2eCompose + @('build', 'control', 'data-worker', 'trainer-worker', 'selfplay-worker', 'ui')))
    foreach ($service in @('control', 'data', 'trainer', 'selfplay', 'ui')) {
        [void](Invoke-Docker "dependencies-$service" @('run', '--rm', '--network', 'none', '--entrypoint', 'python', "zero-ttt/${service}:1.0.0", '-m', 'pip', 'check'))
    }
    [void](Invoke-Docker 'service-images' @('image', 'inspect', 'zero-ttt/control:1.0.0', 'zero-ttt/data:1.0.0', 'zero-ttt/trainer:1.0.0', 'zero-ttt/selfplay:1.0.0', 'zero-ttt/ui:1.0.0'))
    Assert-GpuIdle
    $script:e2eStarted = $true
    # Keep one CPU-only driver alive so fixture state and evidence are accessible across restarts.
    [void](Invoke-Docker 'start-driver' ($e2eCompose + @('run', '-d', '--no-deps', '--name', "$project-driver", '--use-aliases', 'e2e-driver', 'python', 'scripts/check_s01.py', 'hold')))
    Assert-Runtime @("$project-driver") 'driver'
    $script:environmentStep = $false
    [void](Invoke-Docker 'prepare' @('exec', "$project-driver", 'python', 'scripts/compose_e2e_test.py', 'prepare', '/datasets'))
    [void](Invoke-Docker 'start-control' ($e2eCompose + @('up', '-d', '--wait', '--wait-timeout', '90', 'control', 'ui')))
    Invoke-Driver 'recovery-start'
    [void](Invoke-Docker 'restart-with-lease' ($e2eCompose + @('restart', 'control', 'ui')))
    [void](Invoke-Docker 'wait-restart' ($e2eCompose + @('up', '-d', '--wait', '--wait-timeout', '40', 'control', 'ui')))
    Invoke-Driver 'recovery-check'
    [void](Invoke-Docker 'start-workers' ($e2eCompose + @('up', '-d', '--wait', '--wait-timeout', '90', 'data-worker', 'trainer-worker', 'selfplay-worker')))
    $ids = Invoke-Docker 'e2e-containers' ($e2eCompose + @('ps', '-q', 'control', 'ui', 'data-worker', 'trainer-worker', 'selfplay-worker'))
    Assert-Runtime @($ids -split '\s+') 'services'
    Invoke-Driver 'bootstrap'
    [void](Invoke-Docker 'restart-after-cold' ($e2eCompose + @('restart', 'control', 'ui')))
    [void](Invoke-Docker 'wait-after-cold' ($e2eCompose + @('up', '-d', '--wait', '--wait-timeout', '90', 'control', 'ui')))
    Invoke-Driver 'alpha'
    [void](Invoke-Docker 'e2e-service-logs' ($e2eCompose + @('logs', '--no-color')))
    [void](Invoke-Docker 'remove-driver' @('rm', '-f', "$project-driver"))
    [void](Invoke-Docker 'e2e-cleanup' ($e2eCompose + @('down', '-v', '--remove-orphans')))
    $script:e2eStarted = $false
    $report.stages.e2e = 'passed'; Save-Report
}

try {
    $report.stages.preflight = 'running'; Save-Report
    [void](Invoke-Recorded 'git-head' 'git' @('rev-parse', 'HEAD'))
    [void](Invoke-Recorded 'git-status' 'git' @('status', '--short', '--untracked-files=all'))
    [void](Invoke-Recorded 'git-diff' 'git' @('diff', 'HEAD', '--binary'))
    [void](Invoke-Docker 'engine' @('version', '--format', '{{json .}}'))
    [void](Invoke-Docker 'compose-version' @('compose', 'version'))
    $resolvedBase = Invoke-Docker 'compose-base' ($baseCompose + @('config', '--format', 'json')) | ConvertFrom-Json
    $resolvedE2e = Invoke-Docker 'compose-e2e' ($e2eCompose + @('config', '--format', 'json')) | ConvertFrom-Json
    Write-JsonFile @{ workspace = $workspace; project = $project; base = $resolvedBase; e2e = $resolvedE2e } 'compose.json'
    [void](Invoke-Docker 'build-dev' ($baseCompose + @('build', 'test')))
    [void](Invoke-Docker 'dev-image' @('image', 'inspect', 'zero-ttt/dev:1.0.0'))
    $environmentStep = $false
    [void](Invoke-Test 'compose-isolation' @('python', 'scripts/check_s01.py', 'compose', '--input', "$containerReportRoot/compose.json"))
    Assert-Probe 'test'
    $evidence = Invoke-Test 'environment-and-origins' @('python', 'scripts/check_s01.py', 'environment')
    [IO.File]::WriteAllText((Join-Path $reportRoot 'environment.json'), $evidence, $utf8)
    $report.stages.preflight = 'passed'
    if ($Stage -in @('cpu', 'all')) { Run-Cpu }
    if ($Stage -in @('gpu', 'all')) { Run-Gpu }
    if ($Stage -in @('e2e', 'all')) { Run-E2e }
    $report.s01_passed = $report.stages.cpu -eq 'passed' -and $report.stages.gpu -eq 'passed' -and $report.stages.e2e -eq 'passed'
    Write-JsonFile $report 'report-check.json'
    [void](Invoke-Test 'report-consistency' @('python', 'scripts/check_s01.py', 'report', '--input', "$containerReportRoot/report-check.json"))
} catch {
    $report.stages[$currentStage] = $(if ($environmentStep) { 'blocked' } else { 'failed' })
    $report.s01_passed = $false
    $report.failure = $_.Exception.Message
    Write-Host $report.failure -ForegroundColor Red
} finally {
    foreach ($name in @($probeNames.ToArray())) {
        try { [void](Invoke-Docker 'cleanup-probe' @('rm', '-f', $name)) } catch { Write-Warning $_ }
    }
    if ($e2eStarted) {
        $report.cleanup_command = 'docker ' + (($e2eCompose + @('down', '-v', '--remove-orphans')) -join ' ')
        try { [void](Invoke-Docker 'failure-service-logs' ($e2eCompose + @('logs', '--no-color'))) } catch { Write-Warning $_ }
        try { [void](Invoke-Docker 'failure-remove-driver' @('rm', '-f', "$project-driver")) } catch { Write-Warning $_ }
        try { [void](Invoke-Docker 'failure-stop-services' ($e2eCompose + @('down', '--remove-orphans'))) } catch { Write-Warning $_ }
    }
    $report.finished_utc = [DateTime]::UtcNow.ToString('o')
    Save-Report
    Write-Host "S01 evidence: $reportRoot"
}
if ($report.failure) { exit 1 }
exit 0
