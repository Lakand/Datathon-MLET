# app/monitor.py
import sqlite3
import json
import datetime
import os

DB_PATH = "monitoring.db"

def init_db():
    """Cria a tabela de logs se não existir."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            ra TEXT,
            input_data TEXT,
            predicted_pedra TEXT,
            model_version TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_prediction(ra: str, input_data: dict, prediction: str, version: str = "v1"):
    """Salva a predição no banco para cálculo futuro de Drift."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions (ra, input_data, predicted_pedra, model_version)
            VALUES (?, ?, ?, ?)
        ''', (ra, json.dumps(input_data), prediction, version))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Erro ao salvar log: {e}")