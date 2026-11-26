$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
& 'C:\Users\Lenovo\Documents\ATLAS\runtime\Scripts\python.exe' -u '.\runner.py' --workers 12
if ($LASTEXITCODE -ne 0) { throw 'Experiment failed; inspect the error above.' }
