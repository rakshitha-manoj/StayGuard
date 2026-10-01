# Deletes the StayGuard DVC bucket and ALL its contents. Run manually.
$Bucket = "gs://stayg-510316-stayguard-dvc"
$gcloudBin = "C:\Users\raksh\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin"
if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) { $env:PATH = "$gcloudBin;$env:PATH" }

Write-Host "This will permanently delete $Bucket and everything in it (project stayg-510316)."
$answer = Read-Host "Type the bucket name to confirm"
if ($answer -ne $Bucket) { Write-Host "Aborted."; exit 1 }
gcloud storage rm --recursive $Bucket --project stayg-510316
