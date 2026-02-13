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
import time
import tempfile
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.metrics import classification_report, f1_score, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

from src.utils import load_data, save_artifact
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src import config

# Configuração de filtros de aviso para manter a saída do console limpa
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Configuração de persistência de logs em arquivo temporário
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
    """Gera e salva a matriz de confusão como um artefato visual.

    Args:
        y_true: Valores reais do target.
        y_pred: Valores preditos pelo modelo.
        title: Título do gráfico.

    Returns:
        str: Caminho do arquivo de imagem temporário.
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
    """Executa o pipeline completo de treinamento e registro de experimentos.

    Realiza o processamento de dados, validação cruzada por grupos (RA),
    treinamento do modelo MLP e log de métricas e artefatos no MLflow.

    Returns:
        dict: Resumo das métricas calculadas durante o treinamento.
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
            logger.info(f"Carregando dados...")
            dict_abas = load_data(str(config.RAW_DATA_PATH))
            preprocessor = DataPreprocessor()
            df_clean = preprocessor.run(dict_abas)

            # Limpeza de alvos para garantir qualidade no conjunto de treinamento
            logger.info(f"Shape antes da limpeza de targets: {df_clean.shape}")
            df_clean = df_clean.dropna(subset=['RA', 'PEDRA'])
            df_clean = df_clean[df_clean['PEDRA'].isin(config.MAPA_PEDRA.keys())]
            df_clean = df_clean.reset_index(drop=True)
            logger.info(f"Shape pós limpeza de targets: {df_clean.shape}")
            
            # Divisão dos conjuntos de dados mantendo integridade por Registro Acadêmico
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

            # Validação Cruzada Estratificada por Grupo
            logger.info("Iniciando Cross-Validation...")
            sgkf = StratifiedGroupKFold(n_splits=config.SPLIT_PARAMS['n_splits_cv'])
            f1_scores = []

            for fold, (t_idx, v_idx) in enumerate(sgkf.split(df_train, df_train['PEDRA'], groups=df_train['RA']), 1):
                fe_cv = FeatureEngineer()
                fe_cv.fit(df_train.iloc[t_idx])
                X_t, y_t = fe_cv.transform(df_train.iloc[t_idx])
                X_v, y_v = fe_cv.transform(df_train.iloc[v_idx])
                
                # Balanceamento de classes para treinamento da rede neural
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

            # Ajuste do modelo final
            logger.info("Treinando modelo final...")
            fe_final = FeatureEngineer()
            fe_final.fit(df_train)
            X_final, y_final = fe_final.transform(df_train)
            
            smote_final = SMOTE(random_state=config.MODEL_PARAMS['random_state'])
            X_res, y_res = smote_final.fit_resample(X_final, y_final)
            
            model_final = MLPClassifier(**config.MODEL_PARAMS)
            start_time = time.time()
            model_final.fit(X_res, y_res)
            training_duration = time.time() - start_time
            
            # Validação no conjunto de holdout (Teste)
            X_test_scaled, y_test_real = fe_final.transform(df_test)
            y_pred_test = model_final.predict(X_test_scaled)
            
            test_f1 = f1_score(y_test_real, y_pred_test, average='macro')
            test_acc = accuracy_score(y_test_real, y_pred_test)
            
            mlflow.log_metric("test_f1_macro", test_f1)
            mlflow.log_metric("test_accuracy", test_acc)
            mlflow.log_metric("training_time_seconds", training_duration)

            report = classification_report(
                y_test_real, 
                y_pred_test, 
                target_names=list(config.MAPA_PEDRA.keys()), # Garante nomes legíveis
                output_dict=True
            )
            
            for pedra, metrics in report.items():
                if isinstance(metrics, dict): # Ignora a chave 'accuracy' que é float
                    # Ex: f1_Ametista, recall_Topazio, etc.
                    mlflow.log_metric(f"precision_{pedra}", metrics['precision'])
                    mlflow.log_metric(f"recall_{pedra}", metrics['recall'])
                    mlflow.log_metric(f"f1_{pedra}", metrics['f1-score'])
            
            logger.info(f"Test F1: {test_f1:.4f} | Acc: {test_acc:.4f} | | Tempo: {training_duration:.2f}s")

            # Salvamento de artefatos e registro no MLflow
            save_artifact(model_final, config.MODEL_PATH)
            save_artifact(fe_final, config.PIPELINE_PATH)
            
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
            # Limpeza de recursos e fechamento de handlers
            try:
                for handler in logging.root.handlers:
                    handler.close()
                if os.path.exists(temp_log_path):
                    os.remove(temp_log_path)
            except Exception:
                pass
            
            try:
                if temp_img_path and os.path.exists(temp_img_path):
                    os.remove(temp_img_path)
            except Exception:
                pass
                
    return metrics_summary


if __name__ == "__main__":
    train_pipeline()