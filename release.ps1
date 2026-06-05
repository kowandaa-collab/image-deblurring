<#
  BlurDM release / evaluation utilities.

  Actions:
    eval    Run prediction → compute metrics → compare against baselines.
    zip     Package results/ PNGs into release_assets/*.zip.
    upload  Upload zips to a GitHub Release (requires: gh auth login).

  Usage:
    .\release.ps1 -Action eval `
        -ModelName NAFNetBlurDM-light `
        -ModelPath ".\experiments\NAFNet\GoPro\stage3\best_deblur_NAFNetBlurDM-light.pth" `
        -DmPath    ".\experiments\NAFNet\GoPro\stage2\best_dm_BlurDM.pth"

    .\release.ps1 -Action zip

    .\release.ps1 -Action upload [-Tag gopro-results-v1]
#>
param(
  [Parameter(Mandatory = $true)]
  [ValidateSet('eval', 'zip', 'upload')]
  [string]$Action,

  # --- eval ---
  [string]$ModelName    = '',
  [string]$ModelPath    = '',
  [string]$DmPath       = '',
  [string]$Backbone     = '',          # auto-detected from ModelName if blank
  [string]$DatasetRoot  = 'D:/dataset/test',
  [string]$DatasetName  = 'GoPro',
  [string]$RunName      = 'NAFNet/GoPro_auto',
  [switch]$WithTTA,
  [int]$Tile            = 0,
  [int]$Overlap         = 32,
  [string[]]$CompareCsv = @(
    'results/MIMO_UNet/GoPro/metrics_gopro.csv',
    'results/NAFNet/GoPro/metrics_gopro.csv',
    'results/NAFNet/GoPro_tta/metrics_gopro_tta.csv'
  ),

  # --- upload ---
  [string]$Tag  = 'gopro-results-v1',
  [string]$Repo = 'kowandaa-collab/image-deblurring'
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

function Assert-LastExit([string]$Step) {
  if ($LASTEXITCODE -ne 0) { throw "$Step failed with exit code $LASTEXITCODE." }
}


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------
function Invoke-Eval {
  if (-not $ModelName) { throw '-ModelName is required for -Action eval.' }
  if (-not $ModelPath) { throw '-ModelPath is required for -Action eval.' }
  if (-not $DmPath)    { throw '-DmPath is required for -Action eval.' }

  # Auto-detect backbone from model name
  $backbone = $Backbone
  if (-not $backbone) {
    if ($ModelName -like 'RestormerBlurDM*')  { $backbone = 'Restormer'   }
    elseif ($ModelName -like '*MIMOUNet*')    { $backbone = 'MIMO_UNet'   }
    elseif ($ModelName -like 'StripformerPrior*') { $backbone = 'Stripformer' }
    else                                      { $backbone = 'NAFNet'      }
  }

  $resultDir = Join-Path 'results' $RunName
  $predDir   = Join-Path $resultDir $DatasetName
  $csvPath   = Join-Path $resultDir ("metrics_{0}.csv" -f $DatasetName.ToLower())
  New-Item -ItemType Directory -Force -Path $resultDir | Out-Null

  Write-Host '=== 1/3  Prediction ===' -ForegroundColor Cyan
  $predictArgs = @(
    'predict.py',
    '--backbone',    $backbone,
    '--model_name',  $ModelName,
    '--model_path',  $ModelPath,
    '--dm_path',     $DmPath,
    '--data_path',   $DatasetRoot,
    '--dir_path',    $resultDir,
    '--dataset',     $DatasetName
  )
  if ($WithTTA)    { $predictArgs += '--tta' }
  if ($Tile -gt 0) { $predictArgs += @('--tile', "$Tile", '--overlap', "$Overlap") }
  python @predictArgs
  Assert-LastExit 'Prediction'

  Write-Host '=== 2/3  Metrics ===' -ForegroundColor Cyan
  $gtDir = Join-Path (Join-Path $DatasetRoot $DatasetName) 'target'
  python tools.py eval --gt $gtDir --pred $predDir --output-csv $csvPath
  Assert-LastExit 'Metrics'

  Write-Host '=== 3/3  Compare ===' -ForegroundColor Cyan
  $existing = @($CompareCsv | Where-Object { Test-Path $_ }) + $csvPath
  python tools.py compare --csv $existing
  Assert-LastExit 'Comparison'

  Write-Host "`nDone." -ForegroundColor Green
  Write-Host "Prediction dir : $predDir"
  Write-Host "Metrics CSV    : $csvPath"
}


# ---------------------------------------------------------------------------
# zip
# ---------------------------------------------------------------------------
function Invoke-Zip {
  $out = Join-Path $PSScriptRoot 'release_assets'
  New-Item -ItemType Directory -Force -Path $out | Out-Null

  $runs = @(
    @{ Name = 'MIMO_UNet_GoPro';   Src = 'results\MIMO_UNet\GoPro\GoPro' },
    @{ Name = 'NAFNet_GoPro';      Src = 'results\NAFNet\GoPro\GoPro' },
    @{ Name = 'NAFNet_GoPro_tta';  Src = 'results\NAFNet\GoPro_tta\GoPro' },
    @{ Name = 'NAFNet_GoPro_full'; Src = 'results\NAFNet\GoPro_full\GoPro' },
    @{ Name = 'Restormer_GoPro';   Src = 'results\Restormer\GoPro\GoPro' }
  )

  foreach ($r in $runs) {
    $src = Join-Path $PSScriptRoot $r.Src
    $zip = Join-Path $out ($r.Name + '.zip')
    if (-not (Test-Path $src))  { Write-Warning "Skip $($r.Name): $src missing"; continue }
    if (Test-Path $zip)         { Write-Host "Exists: $($r.Name).zip"; continue }
    Write-Host "Zipping $($r.Name) ..."
    Compress-Archive -Path (Join-Path $src '*') -DestinationPath $zip -CompressionLevel Fastest
    $mb = [math]::Round((Get-Item $zip).Length / 1MB, 1)
    Write-Host "  -> $mb MB"
  }
  Write-Host "Zips in $out"
}


# ---------------------------------------------------------------------------
# upload
# ---------------------------------------------------------------------------
function Invoke-Upload {
  $assets = Join-Path $PSScriptRoot 'release_assets'
  if (-not (Test-Path $assets)) {
    throw "Run '.\release.ps1 -Action zip' first (release_assets/ missing)."
  }

  $zips = Get-ChildItem $assets -Filter '*.zip' | Sort-Object Name
  if ($zips.Count -eq 0) { throw 'No zip files found in release_assets/.' }

  $exists = $false
  try { gh release view $Tag -R $Repo 2>$null | Out-Null; $exists = ($LASTEXITCODE -eq 0) }
  catch {}

  if (-not $exists) {
    gh release create $Tag -R $Repo `
      --title 'GoPro deblurred images' `
      --notes '1111 PNGs per run (1280x720). Extract into results/<model>/GoPro/GoPro/.'
    if ($LASTEXITCODE -ne 0) { throw "Failed to create release $Tag." }
  }

  foreach ($z in $zips) {
    Write-Host "Uploading $($z.Name) ..."
    gh release upload $Tag $z.FullName -R $Repo --clobber
  }
  Write-Host "Done: https://github.com/$Repo/releases/tag/$Tag"
}


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
switch ($Action) {
  'eval'   { Invoke-Eval }
  'zip'    { Invoke-Zip }
  'upload' { Invoke-Upload }
}
