# src/utils.py
import pandas as pd
import joblib
import os

def load_data(file_path, sheet_name=None):
    """
    Carrega o arquivo Excel bruto.
    
    Args:
        file_path (str): Caminho do arquivo .xlsx
        sheet_name (str, list, None): Nome da aba específica ou None para todas. 
                                      Padrão é None (carrega todas).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
    
    # Passamos o argumento sheet_name para permitir carregamento sob demanda (performance)
    return pd.read_excel(file_path, sheet_name=sheet_name)

def save_artifact(obj, filepath):
    """Salva modelos, scalers ou outros artefatos com joblib."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(obj, filepath)
    print(f"Artefato salvo em: {filepath}")

def load_artifact(filepath):
    """Carrega artefatos salvos."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Artefato não encontrado: {filepath}")
    return joblib.load(filepath)