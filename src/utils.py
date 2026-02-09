# src/utils.py
import pandas as pd
import joblib
import os
from typing import Any

def load_data(file_path: str) -> dict:
    """
    Carrega o arquivo Excel e retorna um dicionário de DataFrames (um por aba).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
    
    # Retorna um dicionário onde a chave é o nome da aba
    return pd.read_excel(file_path, sheet_name=None)

def save_artifact(obj: Any, file_path: str) -> None:
    """
    Salva um objeto (modelo, pipeline, etc.) usando joblib.
    Cria o diretório se não existir.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(obj, file_path)

def load_artifact(file_path: str) -> Any:
    """
    Carrega um artefato salvo com joblib.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Artefato não encontrado: {file_path}")
    return joblib.load(file_path)

def calculate_risk_level(pedra_nome: str) -> str:
    """
    Calcula o nível de risco de defasagem com base na Pedra prevista.
    
    Regra de Negócio:
    - Topázio e Ametista: Risco Baixo (Alunos com bom desempenho ou em evolução)
    - Ágata e Quartzo: Risco Alto (Alunos que precisam de atenção)
    """
    # Normaliza para garantir comparação segura (embora o modelo retorne capitalizado)
    pedra = pedra_nome.capitalize()
    
    if pedra in ['Topázio', 'Ametista']:
        return "Baixo"
    else:
        return "Alto"