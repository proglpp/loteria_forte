import sqlite3
from pathlib import Path

DB = Path('database/lotomania.db')
if not DB.exists():
    print(f"ERROR: Database not found: {DB}")
    raise SystemExit(1)

conn = sqlite3.connect(DB)
cur = conn.cursor()
cur.execute("SELECT concurso, draw_date FROM draws ORDER BY concurso DESC LIMIT 1")
row = cur.fetchone()
if not row:
    print("No draws found in database")
else:
    concurso, draw_date = row
    print(f"LAST_DRAW_CONCURSO:{concurso}")
    print(f"LAST_DRAW_DATE:{draw_date}")
conn.close()
