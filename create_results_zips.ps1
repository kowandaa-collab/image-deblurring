# Build release_assets/*.zip from results/**/GoPro/GoPro PNG folders.
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$out = Join-Path $root "release_assets"
New-Item -ItemType Directory -Force -Path $out | Out-Null

$runs = @(
    @{ Name = "MIMO_UNet_GoPro";           Src = "results\MIMO_UNet\GoPro\GoPro" },
    @{ Name = "NAFNet_GoPro";              Src = "results\NAFNet\GoPro\GoPro" },
    @{ Name = "NAFNet_GoPro_tta";          Src = "results\NAFNet\GoPro_tta\GoPro" },
    @{ Name = "NAFNet_GoPro_full";         Src = "results\NAFNet\GoPro_full\GoPro" },
    @{ Name = "Restormer_GoPro";           Src = "results\Restormer\GoPro\GoPro" }
)

foreach ($r in $runs) {
    $src = Join-Path $root $r.Src
    $zip = Join-Path $out ($r.Name + ".zip")
    if (-not (Test-Path $src)) { Write-Warning "Skip $($r.Name): $src missing"; continue }
    if (Test-Path $zip) { Write-Host "Exists: $($r.Name).zip"; continue }
    Write-Host "Zipping $($r.Name) ..."
    Compress-Archive -Path (Join-Path $src "*") -DestinationPath $zip -CompressionLevel Fastest
    $mb = [math]::Round((Get-Item $zip).Length / 1MB, 1)
    Write-Host "  -> $mb MB"
}
Write-Host "Zips in $out"
