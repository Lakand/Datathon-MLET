# src/config.py
import os
from pathlib import Path

# --- Caminhos ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
IMG_DIR = BASE_DIR / "images"  # <--- Novo: Pasta para salvar gráficos

# Criar diretórios se não existirem
MODELS_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True) # <--- Novo

RAW_DATA_PATH = DATA_DIR / "BASE DE DADOS PEDE 2024 - DATATHON.xlsx"
TEST_DATA_PATH = DATA_DIR / "test_dataset.csv"
MODEL_PATH = MODELS_DIR / "mlp_model.joblib"
PIPELINE_PATH = MODELS_DIR / "pipeline_features.joblib"

# --- MLflow ---
MLFLOW_EXPERIMENT_NAME = "Passos_Magicos_Classification" # <--- Novo

# --- Hiperparâmetros ---
MODEL_PARAMS = {
    'hidden_layer_sizes': (50,),
    'activation': 'relu',
    'alpha': 0.01,
    'learning_rate_init': 0.001,
    'max_iter': 3000,
    'solver': 'adam',
    'random_state': 42
}

SPLIT_PARAMS = {
    'test_size': 0.2,
    'random_state': 42,
    'n_splits_cv': 10
}

MAPA_PEDRA = {'Quartzo': 0, 'Ágata': 1, 'Ametista': 2, 'Topázio': 3}