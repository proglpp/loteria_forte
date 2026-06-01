import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent
DATA_FILE = ROOT_DIR / os.getenv("DATA_FILE", "data/Lotomania.csv")
DATABASE_FILE = ROOT_DIR / os.getenv("DATABASE_FILE", "database/lotomania.db")
CACHE_DIR = ROOT_DIR / os.getenv("CACHE_DIR", "database/cache")
LOG_DIR = ROOT_DIR / os.getenv("LOG_DIR", "logs")
N_JOBS = int(os.getenv("N_JOBS", "4"))
HMM_STATES = int(os.getenv("HMM_STATES", "3"))
SEED = int(os.getenv("SEED", "42"))

# Ensure persistent directories exist
for path in (CACHE_DIR, LOG_DIR, ROOT_DIR / "exports", ROOT_DIR / "reports", ROOT_DIR / "database"):
    path.mkdir(parents=True, exist_ok=True)

NUMBER_RANGE = list(range(100))
DRAW_COLUMNS = [f"Bola{i}" for i in range(1, 21)]
