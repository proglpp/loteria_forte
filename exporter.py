import json
import logging
import sqlite3
from pathlib import Path
from typing import Dict

import pandas as pd

from config import ROOT_DIR

logger = logging.getLogger(__name__)
EXPORT_DIR = ROOT_DIR / "exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def _save_dataframe(df: pd.DataFrame, name: str) -> None:
    csv_path = EXPORT_DIR / f"{name}.csv"
    json_path = EXPORT_DIR / f"{name}.json"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    df.to_json(json_path, orient="records", force_ascii=False, indent=2)
    logger.info("Exported %s to %s and %s", name, csv_path, json_path)


class Exporter:
    def __init__(self, database_file: Path):
        self.database_file = Path(database_file)

    def save_statistics(self, df: pd.DataFrame) -> None:
        _save_dataframe(df, "statistics")

    def save_predictions(self, df: pd.DataFrame) -> None:
        _save_dataframe(df, "predictions")

    def save_games(self, df: pd.DataFrame) -> None:
        _save_dataframe(df, "generated_games")

    def save_backtests(self, df: pd.DataFrame) -> None:
        _save_dataframe(df, "backtests")

    def save_last_draw_evaluation(self, payload: dict) -> None:
        json_path = EXPORT_DIR / "last_draw_evaluation.json"
        import json as json_module

        with open(json_path, "w", encoding="utf-8") as handle:
            json_module.dump(payload, handle, ensure_ascii=False, indent=2)
        logger.info("Exported last draw evaluation to %s", json_path)

    def export_all(
        self,
        draws: pd.DataFrame,
        features: pd.DataFrame,
        statistics: pd.DataFrame,
        predictions: pd.DataFrame | None = None,
        backtests: pd.DataFrame | None = None,
        generated_games: pd.DataFrame | None = None,
    ) -> None:
        _save_dataframe(draws, "draws")
        _save_dataframe(features, "features")
        _save_dataframe(statistics, "statistics")
        if predictions is not None:
            _save_dataframe(predictions, "predictions")
        if backtests is not None:
            _save_dataframe(backtests, "backtests")
        if generated_games is not None:
            _save_dataframe(generated_games, "generated_games")
        logger.info("Exported all major artifacts")

    def save_to_sqlite(self, table_name: str, df: pd.DataFrame) -> None:
        logger.info("Saving %s to SQLite table %s", table_name, table_name)
        with sqlite3.connect(self.database_file) as connection:
            df.to_sql(table_name, connection, if_exists="replace", index=False)

    def save_database_exports(self, exports: Dict[str, pd.DataFrame]) -> None:
        for table_name, df in exports.items():
            self.save_to_sqlite(table_name, df)
