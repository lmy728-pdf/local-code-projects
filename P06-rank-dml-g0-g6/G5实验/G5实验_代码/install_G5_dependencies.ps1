$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
& 'C:\Users\Lenovo\Documents\ATLAS\runtime\Scripts\python.exe' '.\prepare_environment.py' --install
if ($LASTEXITCODE -ne 0) { throw 'Experiment failed; inspect the error above.' }
