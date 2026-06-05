<#
  One-command evaluation pipeline:
    1) run deblur prediction
    2) compute metrics CSV
    3) compare against optional baseline CSVs

  Example:
    powershell -NoProfile -ExecutionPolicy Bypass -File .\run_eval_and_compare.ps1 `
      -ModelName NAFNetBlurDM-light `
      -ModelPath ".\experiments\NAFNet\GoPro\stage3\best_deblur_NAFNetBlurDM-light.pth" `
      -DmPath ".\experiments\NAFNet\GoPro\stage2\best_dm_BlurDM.pth" `
      -RunName "NAFNet\GoPro_new" `
      -WithTTA
#>
param(
  [Parameter(Mandatory = $true)]
  [string]$ModelName,
  [Parameter(Mandatory = $true)]
  [string]$ModelPath,
  [Parameter(Mandatory = $true)]
  [string]$DmPath,
  [string]$Model = '',
  [string]$DatasetRoot = "D:/dataset/test",
  [string]$DatasetName = "GoPro",
  [string]$RunName = "NAFNet/GoPro_auto",
  [switch]$WithTTA,
  [int]$Tile = 0,
  [int]$Overlap = 32,
  [string[]]$CompareCsv = @(
    "results/MIMO_UNet/GoPro/metrics_gopro.csv",
    "results/NAFNet/GoPro/metrics_gopro.csv",
    "results/NAFNet/GoPro_tta/metrics_gopro_tta.csv"
  )
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONPATH = "src"

function Assert-LastExit([string]$Step) {
  if ($LASTEXITCODE -ne 0) {
    throw "$Step failed with exit code $LASTEXITCODE."
  }
}

$resultDir = Join-Path "results" $RunName
$predDir = Join-Path $resultDir $DatasetName
$csvPath = Join-Path $resultDir ("metrics_{0}.csv" -f $DatasetName.ToLower())

New-Item -ItemType Directory -Force -Path $resultDir | Out-Null

Write-Host "=== 1/3 Prediction ===" -ForegroundColor Cyan
$predictScript = "src/NAFNet/deblur_predict.py"
$predictArgs   = @(
  "--data_path", $DatasetRoot,
  "--dataset", $DatasetName,
  "--dir_path", $resultDir,
  "--model_path", $ModelPath,
  "--dm_path", $DmPath
)
if ($ModelName -like "RestormerBlurDM*") {
  $predictScript = "src/Restormer/deblur_predict.py"
  $predictArgs += @("--model_name", $ModelName)
} elseif ($ModelName -like "*MIMOUNet*") {
  $predictScript = "src/MIMO_UNet/deblur_predict.py"
  $predictArgs += @("--model", $(if ($Model -ne '') { $Model } else { 'MIMO-UNet' }))
} else {
  $predictArgs += @("--model_name", $ModelName)
}
$predCmd = @($predictScript) + $predictArgs
if ($WithTTA) { $predCmd += "--tta" }
if ($Tile -gt 0) {
  $predCmd += @("--tile", "$Tile", "--overlap", "$Overlap")
}
python @predCmd
Assert-LastExit "Prediction"

Write-Host "=== 2/3 Metrics ===" -ForegroundColor Cyan
$gtDir = Join-Path (Join-Path $DatasetRoot $DatasetName) "target"
python eval_metrics.py --gt $gtDir --pred $predDir --output-csv $csvPath
Assert-LastExit "Metrics"

Write-Host "=== 3/3 Compare ===" -ForegroundColor Cyan
$existing = @()
foreach ($c in $CompareCsv) {
  if (Test-Path $c) { $existing += $c }
}
$existing += $csvPath

python compare_metrics.py --csv $existing
Assert-LastExit "Comparison"

Write-Host ""
Write-Host "Done." -ForegroundColor Green
Write-Host "Prediction dir: $predDir"
Write-Host "Metrics CSV   : $csvPath"
