Set-Location -LiteralPath $PSScriptRoot
python -m streamlit run dashboard.py --server.address=localhost --server.port=8501 --server.headless=true *> "$PSScriptRoot\latest_logs\streamlit.log"
