# app/monitor.py
import sqlite3
import json
import datetime
import os

DB_PATH = "monitoring.db"

def init_db():
    """Cria a tabela de logs se não existir."""
    # O uso do 'with' garante o fechamento da conexão automaticamente
    with sqlite3.connect(DB_PATH) as conn:
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

def log_prediction(ra: str, input_data: dict, prediction: str, version: str = "v1"):
    """Salva a predição no banco para cálculo futuro de Drift."""
    try:
        # Abre, executa e fecha a conexão de forma segura
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO predictions (ra, input_data, predicted_pedra, model_version)
                VALUES (?, ?, ?, ?)
            ''', (ra, json.dumps(input_data), prediction, version))
            conn.commit()
            
    except Exception as e:
        print(f"Erro ao salvar log: {e}")