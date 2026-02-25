param(
    [ValidateRange(1, 1000)]
    [int]$Runs = 5
)

$ErrorActionPreference = "Stop"

$cpuExe = Join-Path $PSScriptRoot "0bench_GPU2CPU_CPUTime\gpu_d2h_overhead.exe"
$gpuExe = Join-Path $PSScriptRoot "1bench_GPU2CPU_GPUTime\gpu_d2h_overhead_events.exe"

function Require-File {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        throw "Executable not found: $Path`nBuild it first with nvcc."
    }
}

function Get-Stat {
    param([double[]]$Values)
    $avg = ($Values | Measure-Object -Average).Average
    $min = ($Values | Measure-Object -Minimum).Minimum
    $max = ($Values | Measure-Object -Maximum).Maximum
    return [PSCustomObject]@{
        Avg = [double]$avg
        Min = [double]$min
        Max = [double]$max
    }
}

function Parse-PerIterMs {
    param(
        [string[]]$Lines,
        [string]$Prefix
    )
    $line = $Lines | Where-Object { $_ -like "$Prefix*" } | Select-Object -First 1
    if (-not $line) {
        throw "Could not find line with prefix: $Prefix"
    }
    $m = [regex]::Match($line, "per-iter ([0-9.]+) ms")
    if (-not $m.Success) {
        throw "Could not parse per-iter ms from line: $line"
    }
    return [double]$m.Groups[1].Value
}

function Measure-Benchmark {
    param(
        [string]$Name,
        [string]$ExePath,
        [string]$BlockingPrefix,
        [string]$AsyncPrefix,
        [int]$Count
    )

    $blocking = @()
    $async = @()

    1..$Count | ForEach-Object {
        $i = $_
        $output = & $ExePath
        if ($LASTEXITCODE -ne 0) {
            throw "$Name failed on run $i with exit code $LASTEXITCODE"
        }

        $b = Parse-PerIterMs -Lines $output -Prefix $BlockingPrefix
        $a = Parse-PerIterMs -Lines $output -Prefix $AsyncPrefix
        $blocking += $b
        $async += $a

        Write-Host ("[{0}] run {1}: blocking={2:N6} ms, async={3:N6} ms" -f $Name, $i, $b, $a)
    }

    $bs = Get-Stat -Values $blocking
    $as = Get-Stat -Values $async

    return [PSCustomObject]@{
        Name = $Name
        BlockingAvg = $bs.Avg
        BlockingMin = $bs.Min
        BlockingMax = $bs.Max
        AsyncAvg = $as.Avg
        AsyncMin = $as.Min
        AsyncMax = $as.Max
        AsyncOverBlocking = ($as.Avg / $bs.Avg)
    }
}

Require-File $cpuExe
Require-File $gpuExe

Write-Host "Running benchmarks with Runs=$Runs"
$cpuResult = Measure-Benchmark `
    -Name "CPU wall time (std::chrono)" `
    -ExePath $cpuExe `
    -BlockingPrefix "Blocking D2H" `
    -AsyncPrefix "Non-blocking D2H" `
    -Count $Runs

$gpuResult = Measure-Benchmark `
    -Name "GPU time (cudaEvent)" `
    -ExePath $gpuExe `
    -BlockingPrefix "GPU time (Blocking" `
    -AsyncPrefix "GPU time (Non-blocking" `
    -Count $Runs

Write-Host ""
Write-Host "Summary"
Write-Host "-------"
@($cpuResult, $gpuResult) | ForEach-Object {
    Write-Host ("[{0}]" -f $_.Name)
    Write-Host ("  Blocking : avg={0:N6} ms, min={1:N6}, max={2:N6}" -f $_.BlockingAvg, $_.BlockingMin, $_.BlockingMax)
    Write-Host ("  Async    : avg={0:N6} ms, min={1:N6}, max={2:N6}" -f $_.AsyncAvg, $_.AsyncMin, $_.AsyncMax)
    Write-Host ("  Ratio    : async/blocking = {0:N2}x" -f $_.AsyncOverBlocking)
}
