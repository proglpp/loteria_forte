import csv
import logging
import sqlite3
from pathlib import Path
from typing import List

import pandas as pd

from config import DATA_FILE, DATABASE_FILE, DRAW_COLUMNS

logger = logging.getLogger(__name__)


class LotomaniaDataLoader:
    def __init__(self, data_file: Path = DATA_FILE, database_file: Path = DATABASE_FILE):
        self.data_file = Path(data_file)
        self.database_file = Path(database_file)
        self.database_file.parent.mkdir(parents=True, exist_ok=True)

    def load_data(self, validate: bool = True) -> pd.DataFrame:
        logger.info("Loading Lotomania CSV data from %s", self.data_file)
        if not self.data_file.exists():
            raise FileNotFoundError(f"Lotomania data file not found: {self.data_file}")

        df = self._read_csv_with_detected_separator()
        self._ensure_columns(df)
        df["Concurso"] = pd.to_numeric(df["Concurso"], errors="coerce").astype("Int64")
        if df["Concurso"].isna().any():
            raise ValueError("Invalid or missing Concurso values found in source data")

        df["Data Sorteio"] = pd.to_datetime(
            df["Data Sorteio"].astype(str).str.strip(), dayfirst=True, errors="coerce"
        )
        if df["Data Sorteio"].isna().any():
            raise ValueError("Invalid or missing Data Sorteio values found in source data")

        df = self._normalize_draws(df)

        if validate:
            self._validate_draws(df)
        return df

    def load_draws_from_db(self) -> pd.DataFrame:
        if not self.database_file.exists():
            raise FileNotFoundError(f"SQLite database not found: {self.database_file}")
        logger.info("Loading draws from database %s", self.database_file)
        with sqlite3.connect(self.database_file) as connection:
            query = f"""
                SELECT
                    d.concurso AS Concurso,
                    d.draw_date AS "Data Sorteio",
                    {', '.join(
                        f'MAX(CASE WHEN n.position = {i} THEN n.number END) AS Bola{i}'
                        for i in range(1, 21)
                    )}
                FROM draws d
                JOIN numbers n ON n.draw_id = d.id
                GROUP BY d.id
                ORDER BY d.concurso
            """
            df = pd.read_sql_query(query, connection, parse_dates=["Data Sorteio"])
        return self._normalize_draws(df)

    def _read_csv_with_detected_separator(self) -> pd.DataFrame:
        common_kwargs = {
            "parse_dates": ["Data Sorteio"],
            "dtype": {col: str for col in DRAW_COLUMNS},
            "na_values": ["", "nan", None],
            "encoding": "utf-8-sig",
            "engine": "python",
        }

        with open(self.data_file, "r", encoding="utf-8-sig", errors="replace") as file_handle:
            sample = file_handle.read(4096)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
                separator = dialect.delimiter
            except csv.Error:
                separator = ";"

        try:
            df = pd.read_csv(self.data_file, sep=separator, **common_kwargs)
            self._ensure_columns(df)
            return df
        except Exception as exc:
            logger.warning("Failed to read CSV with separator '%s': %s", separator, exc)
            df = pd.read_csv(self.data_file, sep=";", **common_kwargs)
            self._ensure_columns(df)
            return df

    def _ensure_columns(self, df: pd.DataFrame) -> None:
        expected = {"Concurso", "Data Sorteio", *DRAW_COLUMNS}
        missing = expected.difference(df.columns)
        if missing:
            raise ValueError(f"Missing expected columns in source data: {sorted(missing)}")

    def _normalize_draws(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[DRAW_COLUMNS] = df[DRAW_COLUMNS].apply(lambda column: column.map(self._normalize_number))
        df = df.sort_values(["Concurso", "Data Sorteio"]).reset_index(drop=True)
        return df

    @staticmethod
    def _normalize_number(value: str) -> str:
        if pd.isna(value):
            return "00"
        value = str(value).strip().zfill(2)
        if not value.isdigit():
            raise ValueError(f"Invalid drawn number: {value}")
        number = int(value)
        if number < 0 or number > 99:
            raise ValueError(f"Invalid drawn number: {value}")
        return f"{number:02d}"

    def _validate_draws(self, df: pd.DataFrame) -> None:
        if df.duplicated(subset=["Concurso"]).any():
            duplicates = df[df.duplicated(subset=["Concurso"], keep=False)]["Concurso"].tolist()
            raise ValueError(f"Duplicate contest numbers found: {duplicates}")

        missing_contests = self._find_missing_contests(df)
        if missing_contests:
            logger.warning("Missing contests detected: %s", missing_contests)

        if df[DRAW_COLUMNS].isnull().values.any():
            missing_rows = df[df[DRAW_COLUMNS].isnull().any(axis=1)]["Concurso"].tolist()
            raise ValueError(f"Missing draw numbers found in rows: {missing_rows}")

        duplicate_rows = df[df[DRAW_COLUMNS].apply(lambda row: len(set(row)) != len(row), axis=1)]
        if not duplicate_rows.empty:
            bad_rows = duplicate_rows["Concurso"].tolist()
            raise ValueError(f"Duplicate numbers inside individual draw rows: {bad_rows}")

        invalid_values = df[DRAW_COLUMNS].apply(lambda column: column.map(
            lambda x: not str(x).isdigit() or int(str(x)) < 0 or int(str(x)) > 99
        ))
        if invalid_values.any().any():
            invalid_rows = df[invalid_values.any(axis=1)]["Concurso"].tolist()
            raise ValueError(f"Invalid number values detected in rows: {invalid_rows}")

    @staticmethod
    def _find_missing_contests(df: pd.DataFrame) -> List[int]:
        concursos = sorted(pd.to_numeric(df["Concurso"], errors="coerce").dropna().astype(int).tolist())
        if not concursos:
            return []
        full_range = set(range(concursos[0], concursos[-1] + 1))
        return sorted(list(full_range.difference(concursos)))

    def create_database(self) -> None:
        logger.info("Creating SQLite database at %s", self.database_file)
        with sqlite3.connect(self.database_file) as connection:
            cursor = connection.cursor()
            cursor.execute("PRAGMA foreign_keys = ON")
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS draws (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    concurso INTEGER UNIQUE,
                    draw_date TEXT
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS numbers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    draw_id INTEGER,
                    number TEXT,
                    position INTEGER,
                    UNIQUE(draw_id, number),
                    FOREIGN KEY(draw_id) REFERENCES draws(id) ON DELETE CASCADE
                )
                """
            )
            connection.commit()

    def persist_draws(self, df: pd.DataFrame) -> None:
        self.create_database()
        logger.info("Persisting draws to SQLite database")
        with sqlite3.connect(self.database_file) as connection:
            cursor = connection.cursor()
            cursor.execute("PRAGMA foreign_keys = ON")
            cursor.execute("DELETE FROM numbers")
            cursor.execute("DELETE FROM draws")

            draw_rows = [
                (int(row["Concurso"]), row["Data Sorteio"].strftime("%Y-%m-%d"))
                for _, row in df.iterrows()
            ]
            cursor.executemany(
                "INSERT OR IGNORE INTO draws (concurso, draw_date) VALUES (?, ?)",
                draw_rows,
            )

            cursor.execute("SELECT id, concurso FROM draws")
            draw_id_map = {concurso: id_ for id_, concurso in cursor.fetchall()}

            number_rows = []
            for _, row in df.iterrows():
                draw_id = draw_id_map[int(row["Concurso"])]
                for pos, col in enumerate(DRAW_COLUMNS, start=1):
                    number_rows.append((draw_id, row[col], pos))
            cursor.executemany(
                "INSERT INTO numbers (draw_id, number, position) VALUES (?, ?, ?)",
                number_rows,
            )
            connection.commit()
        logger.info("Persisted %d draws", len(df))
