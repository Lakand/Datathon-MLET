import pandas as pd
import joblib
import os

def load_data(file_path):
    """Carrega o arquivo Excel bruto."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
    return pd.read_excel(file_path, sheet_name=None)

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