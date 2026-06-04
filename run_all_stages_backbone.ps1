<#
  Run BlurDM stages 1–3 for a chosen backbone (same dataset/hyperparams as run_all_stages_full.ps1).

  Usage:
    .\run_all_stages_backbone.ps1 -Backbone MIMO_UNet
    .\run_all_stages_backbone.ps1 -Backbone NAFNet -CropSize 256 -BatchSize 8
    .\run_all_stages_backbone.ps1 -Backbone Stripformer -CropSize 256 -BatchSize 4

  Default OutRoot when -OutRoot is omitted:
    NAFNet       -> <repo>\experiments\NAFNet\GoPro
    MIMO_UNet    -> D:\BlurDM_experiments\MIMO_UNet\GoPro
    Stripformer  -> <repo>\experiments\Stripformer\GoPro
#>
param(
  [Parameter(Mandatory = $false)]
  [ValidateSet('MIMO_UNet', 'NAFNet', 'Stripformer')]
  [string]$Backbone = 'MIMO_UNet',

  [string]$DataPath = 'D:/GOPRO_Large',
  [string]$OutRoot = '',
  [int]$EndEpoch = 3,
  [int]$BatchSize = 4,
  [int]$CropSize = 128,
  [int]$NumWorkers = 0,
  [int]$ValidationEpoch = 1,
  [int]$CheckpointEpoch = 1,
  [int]$ValSaveEpochs = 1,

  # NAFNet stage1/3 default from train_stage*.py (filenames use this suffix)
  [string]$NAFNetModelName = 'NAFNetBlurDM-light',
  # MIMO stage2 diffusion checkpoint name (default BlurDM in train_stage2.py)
  [string]$MIMO_Stage2DmName = 'BlurDM',
  # NAFNet stage2 saves best_dm_<name>.pth (default BlurDM in NAFNet train_stage2.py)
  [string]$NAFNet_Stage2DmName = 'BlurDM',
  # Stripformer stage2 dm checkpoint name (default Stripformer in train_stage2.py)
  [string]$Stripformer_Stage2DmName = 'Stripformer'
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

if ([string]::IsNullOrWhiteSpace($OutRoot)) {
  if ($Backbone -eq 'NAFNet') {
    $OutRoot = Join-Path $PSScriptRoot 'experiments\NAFNet\GoPro'
  } elseif ($Backbone -eq 'Stripformer') {
    $OutRoot = Join-Path $PSScriptRoot 'experiments\Stripformer\GoPro'
  } else {
    $OutRoot = "D:/BlurDM_experiments/$Backbone/GoPro"
  }
}

$stage1Dir = Join-Path $OutRoot 'stage1'
$stage2Dir = Join-Path $OutRoot 'stage2'
$stage3Dir = Join-Path $OutRoot 'stage3'

function Assert-LastCommandSucceeded {
  param(
    [Parameter(Mandatory = $true)]
    [string]$StepName
  )
  if ($LASTEXITCODE -ne 0) {
    throw "$StepName failed with exit code $LASTEXITCODE."
  }
}

function Invoke-MIMO {
  Write-Host '=== MIMO-UNet: Stage 1 (backbone + latent encoder) ===' -ForegroundColor Cyan
  $env:MASTER_PORT = '29611'
  python src/MIMO_UNet/train_stage1.py `
    --data_path $DataPath `
    --dir_path $stage1Dir `
    --num_workers $NumWorkers `
    --batch_size $BatchSize `
    --crop_size $CropSize `
    --end_epoch $EndEpoch `
    --validation_epoch $ValidationEpoch `
    --check_point_epoch $CheckpointEpoch `
    --val_save_epochs $ValSaveEpochs
  Assert-LastCommandSucceeded 'MIMO stage 1'

  $le = Join-Path $stage1Dir 'final_le_MIMOUNetBlurDM.pth'
  $db = Join-Path $stage1Dir 'final_deblur_MIMOUNetBlurDM.pth'
  if (!(Test-Path $le)) { throw "Stage 1 failed: missing $le" }
  if (!(Test-Path $db)) { throw "Stage 1 failed: missing $db" }

  Write-Host '=== MIMO-UNet: Stage 2 (BlurDM prior) ===' -ForegroundColor Cyan
  $env:MASTER_PORT = '29612'
  python src/MIMO_UNet/train_stage2.py `
    --data_path $DataPath `
    --dir_path $stage2Dir `
    --model_le_path $le `
    --num_workers $NumWorkers `
    --batch_size $BatchSize `
    --crop_size $CropSize `
    --end_epoch $EndEpoch `
    --validation_epoch $ValidationEpoch `
    --check_point_epoch $CheckpointEpoch `
    --val_save_epochs $ValSaveEpochs
  Assert-LastCommandSucceeded 'MIMO stage 2'

  $dm = Join-Path $stage2Dir "final_dm_$MIMO_Stage2DmName.pth"
  if (!(Test-Path $dm)) { throw "Stage 2 failed: missing $dm" }

  Write-Host '=== MIMO-UNet: Stage 3 (joint fine-tuning) ===' -ForegroundColor Cyan
  $env:MASTER_PORT = '29613'
  python src/MIMO_UNet/train_stage3.py `
    --data_path $DataPath `
    --dir_path $stage3Dir `
    --model_path $db `
    --model_dm_path $dm `
    --num_workers $NumWorkers `
    --batch_size $BatchSize `
    --crop_size $CropSize `
    --end_epoch $EndEpoch `
    --validation_epoch $ValidationEpoch `
    --check_point_epoch $CheckpointEpoch `
    --val_save_epochs $ValSaveEpochs
  Assert-LastCommandSucceeded 'MIMO stage 3'
}

function Invoke-NAFNet {
  Write-Host '=== NAFNet: Stage 1 (backbone + latent encoder) ===' -ForegroundColor Cyan
  $env:MASTER_PORT = '29621'
  python src/NAFNet/train_stage1.py `
    --data_path $DataPath `
    --dir_path $stage1Dir `
    --model_name $NAFNetModelName `
    --model $NAFNetModelName `
    --num_workers $NumWorkers `
    --batch_size $BatchSize `
    --crop_size $CropSize `
    --end_epoch $EndEpoch `
    --validation_epoch $ValidationEpoch `
    --check_point_epoch $CheckpointEpoch `
    --val_save_epochs $ValSaveEpochs
  Assert-LastCommandSucceeded 'NAFNet stage 1'

  $le = Join-Path $stage1Dir "best_le_$NAFNetModelName.pth"
  $db = Join-Path $stage1Dir "best_deblur_$NAFNetModelName.pth"
  if (!(Test-Path $le)) { throw "Stage 1 failed: missing $le (NAFNet saves best_*, not final_*)" }
  if (!(Test-Path $db)) { throw "Stage 1 failed: missing $db" }

  Write-Host '=== NAFNet: Stage 2 (BlurDM prior) ===' -ForegroundColor Cyan
  $env:MASTER_PORT = '29622'
  python src/NAFNet/train_stage2.py `
    --data_path $DataPath `
    --dir_path $stage2Dir `
    --model_le_path $le `
    --model_name $NAFNet_Stage2DmName `
    --num_workers $NumWorkers `
    --batch_size $BatchSize `
    --crop_size $CropSize `
    --end_epoch $EndEpoch `
    --validation_epoch $ValidationEpoch `
    --check_point_epoch $CheckpointEpoch
  Assert-LastCommandSucceeded 'NAFNet stage 2'

  $dm = Join-Path $stage2Dir "best_dm_$NAFNet_Stage2DmName.pth"
  if (!(Test-Path $dm)) { throw "Stage 2 failed: missing $dm" }

  Write-Host '=== NAFNet: Stage 3 (joint fine-tuning) ===' -ForegroundColor Cyan
  $env:MASTER_PORT = '29623'
  python src/NAFNet/train_stage3.py `
    --data_path $DataPath `
    --dir_path $stage3Dir `
    --model_name $NAFNetModelName `
    --model $NAFNetModelName `
    --deblur_path $db `
    --dm_path $dm `
    --num_workers $NumWorkers `
    --batch_size $BatchSize `
    --crop_size $CropSize `
    --end_epoch $EndEpoch `
    --validation_epoch $ValidationEpoch `
    --check_point_epoch $CheckpointEpoch `
    --val_save_epochs $ValSaveEpochs
  Assert-LastCommandSucceeded 'NAFNet stage 3'
}

function Invoke-Stripformer {
  Write-Host '=== Stripformer: Stage 1 (backbone + latent encoder) ===' -ForegroundColor Cyan
  $env:MASTER_PORT = '29631'
  python src/Stripformer/train_stage1.py `
    --data_path $DataPath `
    --dir_path $stage1Dir `
    --num_workers $NumWorkers `
    --batch_size $BatchSize `
    --crop_size $CropSize `
    --end_epoch $EndEpoch `
    --validation_epoch $ValidationEpoch `
    --check_point_epoch $CheckpointEpoch `
    --val_save_epochs $ValSaveEpochs
  Assert-LastCommandSucceeded 'Stripformer stage 1'

  $le = Join-Path $stage1Dir 'best_le_StripformerPrior.pth'
  $db = Join-Path $stage1Dir 'best_deblur_StripformerPrior.pth'
  if (!(Test-Path $le)) { throw "Stage 1 failed: missing $le" }
  if (!(Test-Path $db)) { throw "Stage 1 failed: missing $db" }

  Write-Host '=== Stripformer: Stage 2 (BlurDM prior) ===' -ForegroundColor Cyan
  $env:MASTER_PORT = '29632'
  python src/Stripformer/train_stage2.py `
    --data_path $DataPath `
    --dir_path $stage2Dir `
    --model_le_path $le `
    --model_name $Stripformer_Stage2DmName `
    --num_workers $NumWorkers `
    --batch_size $BatchSize `
    --crop_size $CropSize `
    --end_epoch $EndEpoch `
    --validation_epoch $ValidationEpoch `
    --check_point_epoch $CheckpointEpoch
  Assert-LastCommandSucceeded 'Stripformer stage 2'

  $dm = Join-Path $stage2Dir "best_dm_$Stripformer_Stage2DmName.pth"
  if (!(Test-Path $dm)) { throw "Stage 2 failed: missing $dm" }

  Write-Host '=== Stripformer: Stage 3 (joint fine-tuning) ===' -ForegroundColor Cyan
  $env:MASTER_PORT = '29633'
  python src/Stripformer/train_stage3.py `
    --data_path $DataPath `
    --dir_path $stage3Dir `
    --deblur_path $db `
    --dm_path $dm `
    --num_workers $NumWorkers `
    --batch_size $BatchSize `
    --crop_size $CropSize `
    --end_epoch $EndEpoch `
    --validation_epoch $ValidationEpoch `
    --check_point_epoch $CheckpointEpoch `
    --val_save_epochs $ValSaveEpochs
  Assert-LastCommandSucceeded 'Stripformer stage 3'
}

Write-Host "Backbone: $Backbone | OutRoot: $OutRoot" -ForegroundColor Green

switch ($Backbone) {
  'MIMO_UNet'   { Invoke-MIMO }
  'NAFNet'      { Invoke-NAFNet }
  'Stripformer' { Invoke-Stripformer }
}

Write-Host "All 3 stages completed successfully. Checkpoints under: $OutRoot" -ForegroundColor Green
