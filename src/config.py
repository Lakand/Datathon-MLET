# src/config.py
"""Módulo de Configuração Central.

Este módulo define as constantes globais, caminhos de diretórios,
hiperparâmetros do modelo e configurações do MLflow utilizados em todo o projeto.
Centraliza as definições para facilitar a manutenção e garantir consistência
entre os ambientes de treino e produção.
"""

from pathlib import Path

# --- Definições de Estrutura de Diretórios ---

# Define a raiz do projeto baseada na localização deste arquivo
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Garante a existência do diretório de modelos para evitar erros de salvamento
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# --- Caminhos de Arquivos e Artefatos ---

# Arquivo Excel original contendo os dados brutos (2022-2024)
RAW_DATA_PATH = DATA_DIR / "BASE DE DADOS PEDE 2024 - DATATHON.xlsx"

# Dataset de teste (holdout) gerado após o split
TEST_DATA_PATH = DATA_DIR / "test_dataset.csv"

# Caminhos para persistência do modelo treinado e do pipeline de features
MODEL_PATH = MODELS_DIR / "mlp_model.joblib"
PIPELINE_PATH = MODELS_DIR / "pipeline_features.joblib"

# --- Configurações de Rastreamento (MLflow) ---
MLFLOW_EXPERIMENT_NAME = "Passos_Magicos_Classification"

# --- Hiperparâmetros do Modelo e Treinamento ---

# Dicionário de configuração para o MLPClassifier (Scikit-Learn)
MODEL_PARAMS = {
    'hidden_layer_sizes': (50,),
    'activation': 'relu',
    'alpha': 0.01,
    'learning_rate_init': 0.001,
    'max_iter': 3000,
    'solver': 'adam',
    'random_state': 42
}

# Parâmetros para a estratégia de validação
SPLIT_PARAMS = {
    'test_size': 0.2,
    'random_state': 42,
    'n_splits_cv': 10
}

# Mapeamento ordinal da variável alvo 'PEDRA'
MAPA_PEDRA = {'Quartzo': 0, 'Ágata': 1, 'Ametista': 2, 'Topázio': 3}