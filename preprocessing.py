import logging
import pickle
from pathlib import Path
from typing import Optional

import pandas as pd

from config import CACHE_DIR, DRAW_COLUMNS

logger = logging.getLogger(__name__)
CACHE_FILE = Path(CACHE_DIR) / "lotomania_draws.pkl"


def preprocess_draws(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Concurso"] = pd.to_numeric(df["Concurso"], errors="coerce").astype(int)
    df = df.sort_values(["Concurso", "Data Sorteio"]).reset_index(drop=True)
    df["draw_id"] = df["Concurso"].rank(method="dense").astype(int)
    df["draw_index"] = df.index + 1
    logger.info("Preprocessing draw dataset: %d rows", len(df))
    if df[DRAW_COLUMNS].isnull().values.any():
        logger.warning("Missing numbers found in draw data. Filling with 00.")
        df[DRAW_COLUMNS] = df[DRAW_COLUMNS].fillna("00")
    return df


def cache_draws(df: pd.DataFrame) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "wb") as handle:
        pickle.dump(df, handle)
    logger.info("Cached preprocessed draws to %s", CACHE_FILE)


def load_cached_draws(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if not CACHE_FILE.exists():
        return None
    try:
        with open(CACHE_FILE, "rb") as handle:
            cached = pickle.load(handle)
        if not cached.empty and len(cached) == len(df):
            return cached
    except Exception as exc:
        logger.warning("Failed to load cache: %s", exc)
    return None
