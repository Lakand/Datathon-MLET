# src/train.py
"""Módulo de Treinamento do Modelo.

Este script orquestra todo o pipeline de treinamento para o modelo de previsão
de risco (Passos Mágicos). Ele abrange desde a carga e limpeza dos dados,
passando pela validação cruzada com estratégia de grupos, até o treinamento
final e registro de experimentos no MLflow.
"""

import logging
import sys
import os
import tempfile
import warnings
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

from src.utils import load_data, save_artifact
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src import config

# Configuração de filtros de aviso
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Configuração de logging temporário para captura durante a execução do pipeline
temp_log_file = tempfile.NamedTemporaryFile(delete=False, suffix=".log")
temp_log_path = temp_log_file.name
temp_log_file.close()

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(temp_log_path, mode='w')
    ]
)
logger = logging.getLogger(__name__)

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Confusion Matrix") -> str:
    """Gera e salva a matriz de confusão em um arquivo temporário.

    Cria um heatmap utilizando Seaborn para visualizar o desempenho do classificador
    em relação às classes reais versus preditas.

    Args:
        y_true (np.ndarray): Array com os rótulos verdadeiros.
        y_pred (np.ndarray): Array com os rótulos preditos pelo modelo.
        title (str, optional): Título do gráfico. Padrão é "Confusion Matrix".

    Returns:
        str: O caminho absoluto do arquivo de imagem (.png) gerado temporariamente.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=config.MAPA_PEDRA.keys(), 
                yticklabels=config.MAPA_PEDRA.keys())
    plt.title(title)
    plt.ylabel('Real')
    plt.xlabel('Previsto')
    
    temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(temp_img.name)
    plt.close()
    temp_img.close()
    
    return temp_img.name

def train_pipeline() -> dict:
    """Executa o pipeline completo de treinamento e avaliação do modelo.

    O pipeline consiste nas seguintes etapas:
    1. Carregamento e pré-processamento dos dados brutos.
    2. Divisão dos dados em treino e teste (respeitando grupos de alunos - RA).
    3. Validação Cruzada (Stratified Group K-Fold) para estimativa robusta de métricas.
    4. Treinamento do modelo final (MLPClassifier) com todo o conjunto de treino e SMOTE.
    5. Avaliação no conjunto de teste reservado.
    6. Registro de artefatos, métricas e logs no MLflow.

    Returns:
        dict: Um dicionário contendo o status da execução, métricas principais
        (F1, Acurácia) e caminhos dos artefatos salvos.

    Raises:
        Exception: Propaga qualquer exceção crítica que ocorra durante o pipeline
        para ser tratada pelo chamador (ex: API).
    """
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    
    temp_img_path = None
    metrics_summary = {}

    with mlflow.start_run():
        logger.info(f"=== Pipeline Iniciado ===")
        
        mlflow.log_params(config.MODEL_PARAMS)
        mlflow.log_params(config.SPLIT_PARAMS)
        mlflow.log_param("strategy", "Meio Termo (No Leakage)")

        try:
            # Carregamento e Limpeza Inicial
            logger.info(f"Carregando dados...")
            dict_abas = load_data(str(config.RAW_DATA_PATH))
            preprocessor = DataPreprocessor()
            df_clean = preprocessor.run(dict_abas)
            df_clean = df_clean.dropna(subset=['RA']).reset_index(drop=True)
            
            # Divisão Treino/Teste (GroupShuffleSplit para evitar data leakage por RA)
            splitter = GroupShuffleSplit(
                test_size=config.SPLIT_PARAMS['test_size'], 
                n_splits=1, 
                random_state=config.SPLIT_PARAMS['random_state']
            )
            train_idx, test_idx = next(splitter.split(df_clean, groups=df_clean['RA']))
            df_train = df_clean.iloc[train_idx].copy().reset_index(drop=True)
            df_test = df_clean.iloc[test_idx].copy().reset_index(drop=True)

            logger.info(f"Salvando dataset de teste em: {config.TEST_DATA_PATH}")
            df_test.to_csv(config.TEST_DATA_PATH, index=False)

            # Validação Cruzada
            logger.info("Iniciando Cross-Validation...")
            sgkf = StratifiedGroupKFold(n_splits=config.SPLIT_PARAMS['n_splits_cv'])
            f1_scores = []

            for fold, (t_idx, v_idx) in enumerate(sgkf.split(df_train, df_train['PEDRA'], groups=df_train['RA']), 1):
                fe_cv = FeatureEngineer()
                fe_cv.fit(df_train.iloc[t_idx])
                X_t, y_t = fe_cv.transform(df_train.iloc[t_idx])
                X_v, y_v = fe_cv.transform(df_train.iloc[v_idx])
                
                smote = SMOTE(random_state=config.MODEL_PARAMS['random_state'])
                X_t_res, y_t_res = smote.fit_resample(X_t, y_t)
                
                params_cv = config.MODEL_PARAMS.copy()
                params_cv['max_iter'] = 2000 
                
                mlp_cv = MLPClassifier(**params_cv)
                mlp_cv.fit(X_t_res, y_t_res)
                
                score = f1_score(y_v, mlp_cv.predict(X_v), average='macro')
                f1_scores.append(score)

            mean_cv_f1 = np.mean(f1_scores)
            mlflow.log_metric("cv_mean_f1", mean_cv_f1)
            logger.info(f"CV F1-Macro: {mean_cv_f1:.4f}")

            # Treinamento Final
            logger.info("Treinando modelo final...")
            fe_final = FeatureEngineer()
            fe_final.fit(df_train)
            X_final, y_final = fe_final.transform(df_train)
            
            smote_final = SMOTE(random_state=config.MODEL_PARAMS['random_state'])
            X_res, y_res = smote_final.fit_resample(X_final, y_final)
            
            model_final = MLPClassifier(**config.MODEL_PARAMS)
            model_final.fit(X_res, y_res)
            
            # Avaliação no conjunto de Teste
            X_test_scaled, y_test_real = fe_final.transform(df_test)
            y_pred_test = model_final.predict(X_test_scaled)
            
            test_f1 = f1_score(y_test_real, y_pred_test, average='macro')
            test_acc = accuracy_score(y_test_real, y_pred_test)
            
            mlflow.log_metric("test_f1_macro", test_f1)
            mlflow.log_metric("test_accuracy", test_acc)
            logger.info(f"Test F1: {test_f1:.4f} | Acc: {test_acc:.4f}")

            # Persistência dos Artefatos
            save_artifact(model_final, config.MODEL_PATH)
            save_artifact(fe_final, config.PIPELINE_PATH)
            
            # Registro de Gráficos e Logs
            temp_img_path = plot_confusion_matrix(y_test_real, y_pred_test)
            mlflow.log_artifact(temp_img_path)
            
            mlflow.sklearn.log_model(model_final, "mlp_model")
            
            mlflow.log_artifact(temp_log_path, artifact_path="logs")
            logger.info("Pipeline concluído e registrado no MLflow!")
            
            metrics_summary = {
                "status": "success",
                "cv_mean_f1": float(mean_cv_f1),
                "test_f1_macro": float(test_f1),
                "test_accuracy": float(test_acc),
                "artifacts_saved": [str(config.MODEL_PATH), str(config.PIPELINE_PATH)]
            }

        except Exception as e:
            logger.critical(f"Erro no pipeline: {e}")
            raise e
        finally:
            # Limpeza de arquivos temporários
            try:
                for handler in logging.root.handlers:
                    handler.close()
                if os.path.exists(temp_log_path):
                    os.remove(temp_log_path)
            except Exception as e:
                print(f"Erro ao limpar log: {e}")
            
            try:
                if temp_img_path and os.path.exists(temp_img_path):
                    os.remove(temp_img_path)
            except Exception as e:
                print(f"Erro ao limpar imagem: {e}")
                
    return metrics_summary

if __name__ == "__main__":
    print(train_pipeline())