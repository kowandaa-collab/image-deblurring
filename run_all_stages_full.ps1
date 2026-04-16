param(
  [string]$DataPath = "D:/GOPRO_Large",
  [string]$OutRoot = "./experiments/MIMO_UNet/GoPro",
  [int]$EndEpoch = 3,
  [int]$BatchSize = 4,
  [int]$CropSize = 128,
  [int]$NumWorkers = 0,
  [int]$ValidationEpoch = 1,
  [int]$CheckpointEpoch = 1,
  [int]$ValSaveEpochs = 1
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$stage1Dir = Join-Path $OutRoot "stage1"
$stage2Dir = Join-Path $OutRoot "stage2"
$stage3Dir = Join-Path $OutRoot "stage3"

Write-Host "Stage 1: training backbone + latent encoder"
$env:MASTER_PORT = "29611"
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

$stage1Le = Join-Path $stage1Dir "final_le_MIMOUNetBlurDM.pth"
$stage1Deblur = Join-Path $stage1Dir "final_deblur_MIMOUNetBlurDM.pth"
if (!(Test-Path $stage1Le)) { throw "Stage 1 failed: missing $stage1Le" }
if (!(Test-Path $stage1Deblur)) { throw "Stage 1 failed: missing $stage1Deblur" }

Write-Host "Stage 2: training BlurDM"
$env:MASTER_PORT = "29612"
python src/MIMO_UNet/train_stage2.py `
  --data_path $DataPath `
  --dir_path $stage2Dir `
  --model_le_path $stage1Le `
  --num_workers $NumWorkers `
  --batch_size $BatchSize `
  --crop_size $CropSize `
  --end_epoch $EndEpoch `
  --validation_epoch $ValidationEpoch `
  --check_point_epoch $CheckpointEpoch `
  --val_save_epochs $ValSaveEpochs

$stage2Dm = Join-Path $stage2Dir "final_dm_BlurDM.pth"
if (!(Test-Path $stage2Dm)) { throw "Stage 2 failed: missing $stage2Dm" }

Write-Host "Stage 3: joint fine-tuning"
$env:MASTER_PORT = "29613"
python src/MIMO_UNet/train_stage3.py `
  --data_path $DataPath `
  --dir_path $stage3Dir `
  --model_path $stage1Deblur `
  --model_dm_path $stage2Dm `
  --num_workers $NumWorkers `
  --batch_size $BatchSize `
  --crop_size $CropSize `
  --end_epoch $EndEpoch `
  --validation_epoch $ValidationEpoch `
  --check_point_epoch $CheckpointEpoch `
  --val_save_epochs $ValSaveEpochs

Write-Host "All 3 stages completed successfully."
