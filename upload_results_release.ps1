# Upload GoPro deblurred PNG archives to GitHub Releases (each zip < 2 GB).
# Prerequisite: gh auth login
param(
    [string]$Tag = "gopro-results-v1",
    [string]$Repo = "kowandaa-collab/image-deblurring"
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$assets = Join-Path $root "release_assets"
if (-not (Test-Path $assets)) {
    Write-Error "Run create_results_zips.ps1 first (missing release_assets/)."
}

$zips = Get-ChildItem $assets -Filter "*.zip" | Sort-Object Name
if ($zips.Count -eq 0) { Write-Error "No zip files in release_assets/" }

gh release view $Tag -R $Repo 2>$null
if ($LASTEXITCODE -ne 0) {
    gh release create $Tag -R $Repo `
        --title "GoPro deblurred images" `
        --notes "1111 PNGs per run (1280x720). Extract into results/<model>/GoPro/GoPro/."
}

foreach ($z in $zips) {
    Write-Host "Uploading $($z.Name) ..."
    gh release upload $Tag $z.FullName -R $Repo --clobber
}
Write-Host "Done: https://github.com/$Repo/releases/tag/$Tag"
