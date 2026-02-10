# app/monitor.py
"""Módulo de Monitoramento e Logging.

Gerencia a persistência de logs de predição em um banco de dados SQLite local.
Esses dados são essenciais para o monitoramento contínuo do modelo, permitindo
a detecção de Data Drift e auditoria das requisições.
"""

import sqlite3
import json

DB_PATH = "monitoring.db"

def init_db() -> None:
    """Inicializa o banco de dados de monitoramento.

    Cria a tabela 'predictions' caso ela ainda não exista. A tabela armazena
    o timestamp, identificador do aluno (RA), dados de entrada brutos e o
    resultado da predição.
    """
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

def log_prediction(ra: str, input_data: dict, prediction: str, version: str = "v1") -> None:
    """Registra uma predição no banco de dados.

    Salva os detalhes da inferência para permitir cálculos futuros de drift
    entre os dados de treino e os dados de produção.

    Args:
        ra (str): O Registro Acadêmico do aluno (identificador).
        input_data (dict): Dicionário contendo as features enviadas para o modelo.
            Será convertido para JSON antes de salvar.
        prediction (str): A classe (Pedra) prevista pelo modelo.
        version (str, optional): Versão do modelo utilizado. Padrão é "v1".
    """
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