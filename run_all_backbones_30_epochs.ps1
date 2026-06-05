param(
  [string]$DataPath = 'D:/GOPRO_Large',
  [string]$DatasetRoot = 'D:/dataset/test',
  [string]$DatasetName = 'GoPro',
  [int]$EndEpoch = 30,
  [int]$BatchSize = 4,
  [int]$CropSize = 128,
  [int]$NumWorkers = 0,
  [int]$ValidationEpoch = 1,
  [int]$CheckpointEpoch = 1,
  [int]$ValSaveEpochs = 1
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$backbones = @('MIMO_UNet', 'NAFNet', 'Restormer')

foreach ($backbone in $backbones) {
    Write-Host "=== Running backbone: $backbone ($EndEpoch epochs per stage) ===" -ForegroundColor Cyan
    $outRoot = Join-Path $PSScriptRoot "experiments\$backbone\GoPro"

    & .\run_all_stages_backbone.ps1 `
      -Backbone $backbone `
      -DataPath $DataPath `
      -OutRoot $outRoot `
      -EndEpoch $EndEpoch `
      -BatchSize $BatchSize `
      -CropSize $CropSize `
      -NumWorkers $NumWorkers `
      -ValidationEpoch $ValidationEpoch `
      -CheckpointEpoch $CheckpointEpoch `
      -ValSaveEpochs $ValSaveEpochs
    if ($LASTEXITCODE -ne 0) {
        throw "Backbone $backbone failed with exit code $LASTEXITCODE."
    }

    switch ($backbone) {
        'MIMO_UNet' {
            $modelName = 'MIMOUNetBlurDM'
            $modelArg  = 'MIMO-UNet'
            $modelPath = Join-Path $outRoot 'stage3\final_deblur_MIMOUNetBlurDM.pth'
            $dmPath    = Join-Path $outRoot 'stage2\final_dm_BlurDM.pth'
        }
        'NAFNet' {
            $modelName = 'NAFNetBlurDM-light'
            $modelArg  = ''
            $modelPath = Join-Path $outRoot 'stage3\best_deblur_NAFNetBlurDM-light.pth'
            $dmPath    = Join-Path $outRoot 'stage2\best_dm_BlurDM.pth'
        }
        'Restormer' {
            $modelName = 'RestormerBlurDM-light'
            $modelArg  = ''
            $modelPath = Join-Path $outRoot 'stage3\best_deblur_RestormerBlurDM-light.pth'
            $dmPath    = Join-Path $outRoot 'stage2\best_dm_BlurDM.pth'
        }
    }

    if (!(Test-Path $modelPath)) {
        throw "Expected model path not found: $modelPath"
    }
    if (!(Test-Path $dmPath)) {
        throw "Expected DM path not found: $dmPath"
    }

    Write-Host "=== Evaluating backbone: $backbone ===" -ForegroundColor Cyan
    $scriptPath = '.\run_eval_and_compare.ps1'
    $scriptArgs = @(
      '-ModelName', $modelName,
      '-ModelPath', $modelPath,
      '-DmPath', $dmPath,
      '-DatasetRoot', $DatasetRoot,
      '-DatasetName', $DatasetName,
      '-RunName', "$backbone/GoPro_30"
    )
    if ($modelArg) { $scriptArgs += @('-Model', $modelArg) }

    & $scriptPath @scriptArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Evaluation failed for $backbone with exit code $LASTEXITCODE."
    }
}

Write-Host "All backbones completed with 30 epochs per stage and evaluation." -ForegroundColor Green
