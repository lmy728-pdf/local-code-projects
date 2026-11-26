$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
& 'C:\Users\Lenovo\Documents\ATLAS\runtime\Scripts\python.exe' -m pip install --only-binary=:all: --no-deps --target '.\dependencies' scikit-learn joblib threadpoolctl
if ($LASTEXITCODE -ne 0) { throw 'Experiment failed; inspect the error above.' }
