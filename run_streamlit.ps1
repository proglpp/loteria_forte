Set-Location -LiteralPath $PSScriptRoot

$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

python -m streamlit run dashboard.py --server.address=localhost --server.port=8501 --server.headless=true *> "$PSScriptRoot\latest_logs\streamlit.log"
