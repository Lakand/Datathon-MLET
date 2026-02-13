# src/config.py
"""Módulo de Configuração Central do Projeto Passos Mágicos.

Este módulo define constantes globais, caminhos de diretórios, hiperparâmetros 
do modelo e configurações do MLflow. Centraliza as definições para garantir 
consistência entre os ambientes de treino e produção.
"""

from pathlib import Path

# ==============================================================================
# ESTRUTURA DE DIRETÓRIOS
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Criação preventiva do diretório de modelos para operações de escrita.
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# ARTEFATOS E DADOS
# ==============================================================================

# Localização da fonte de dados original (formato Excel).
RAW_DATA_PATH = DATA_DIR / "BASE DE DADOS PEDE 2024 - DATATHON.xlsx"

# Caminho para o conjunto de dados de teste (holdout).
TEST_DATA_PATH = DATA_DIR / "test_dataset.csv"

# Caminhos para serialização do modelo e do pipeline de transformação.
MODEL_PATH = MODELS_DIR / "mlp_model.joblib"
PIPELINE_PATH = MODELS_DIR / "pipeline_features.joblib"

# ==============================================================================
# EXPERIMENTAÇÃO (MLFLOW)
# ==============================================================================

MLFLOW_EXPERIMENT_NAME = "Passos_Magicos_Classification"

# ==============================================================================
# PARÂMETROS DE MACHINE LEARNING
# ==============================================================================

# Configuração técnica do estimador MLPClassifier (Rede Neural).
MODEL_PARAMS = {
'hidden_layer_sizes': (100,),
    'activation': 'relu',
    'alpha': 0.01,
    'learning_rate_init': 0.001,
    'max_iter': 2000,
    'random_state': 42
}

# Parâmetros para divisão de dados e validação cruzada.
SPLIT_PARAMS = {
    'test_size': 0.2,
    'random_state': 42,
    'n_splits_cv': 5
}

# Mapeamento ordinal das classes da variável alvo.
MAPA_PEDRA = {'Quartzo': 0, 'Ágata': 1, 'Ametista': 2, 'Topázio': 3}

# Configurações de Monitoramento e Drift
DRIFT_WINDOW_SIZE = 5000