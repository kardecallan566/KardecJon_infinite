import sqlite3
import pandas as pd
from datetime import datetime
import os

class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                color TEXT,
                number INTEGER,
                date TEXT,
                time TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                predicted_color TEXT,
                actual_color TEXT,
                is_correct INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()

    def save_result(self, color: str, number: int, date: str, time: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO history (color, number, date, time)
            VALUES (?, ?, ?, ?)
        ''', (color, number, date, time))
        conn.commit()
        conn.close()

    def save_prediction(self, predicted: str, actual: str, is_correct: bool):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions (predicted_color, actual_color, is_correct)
            VALUES (?, ?, ?)
        ''', (predicted, actual, int(is_correct)))
        conn.commit()
        conn.close()

    def get_history(self, limit: int = 1000):
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(f"SELECT * FROM history ORDER BY id DESC LIMIT {limit}", conn)
        conn.close()
        return df

    def get_stats(self):
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", conn)
        conn.close()
        return df
