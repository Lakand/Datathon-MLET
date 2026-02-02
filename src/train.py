# src/train.py
import logging
import sys
import os
import tempfile  # <--- Biblioteca nativa para lidar com arq temporários
import warnings
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

# Importações locais
from src.utils import load_data, save_artifact
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src import config

# --- 1. FILTRO DE WARNINGS ---
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- 2. CONFIGURAÇÃO DE LOGGING COM ARQUIVO TEMPORÁRIO ---
# Cria um arquivo temporário no SO. O delete=False é necessário no Windows
# para permitir que o logging e o mlflow acessem o arquivo simultaneamente.
temp_log_file = tempfile.NamedTemporaryFile(delete=False, suffix=".log")
temp_log_path = temp_log_file.name
temp_log_file.close() # Fechamos aqui para o Logging poder abrir em seguida

# Reseta handlers anteriores
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),       # Console
        logging.FileHandler(temp_log_path, mode='w') # Arquivo Temp Oculto
    ]
)
logger = logging.getLogger(__name__)

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Gera e salva a matriz de confusão como imagem"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=config.MAPA_PEDRA.keys(), 
                yticklabels=config.MAPA_PEDRA.keys())
    plt.title(title)
    plt.ylabel('Real')
    plt.xlabel('Previsto')
    
    save_path = config.IMG_DIR / "confusion_matrix.png"
    plt.savefig(save_path)
    plt.close()
    return str(save_path)

def train_pipeline():
    # --- Configura o MLflow ---
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    
    with mlflow.start_run():
        logger.info(f"=== Pipeline Iniciado (Log Temporário: {temp_log_path}) ===")
        
        mlflow.log_params(config.MODEL_PARAMS)
        mlflow.log_params(config.SPLIT_PARAMS)
        mlflow.log_param("strategy", "Meio Termo (No Leakage)")

        try:
            # --- Carga e Prep ---
            logger.info(f"Carregando dados...")
            dict_abas = load_data(str(config.RAW_DATA_PATH))
            preprocessor = DataPreprocessor()
            df_clean = preprocessor.run(dict_abas)
            df_clean = df_clean.dropna(subset=['RA']).reset_index(drop=True)
            
            # --- Split ---
            splitter = GroupShuffleSplit(
                test_size=config.SPLIT_PARAMS['test_size'], 
                n_splits=1, 
                random_state=config.SPLIT_PARAMS['random_state']
            )
            train_idx, test_idx = next(splitter.split(df_clean, groups=df_clean['RA']))
            df_train = df_clean.iloc[train_idx].copy().reset_index(drop=True)
            df_test = df_clean.iloc[test_idx].copy().reset_index(drop=True)

            # --- Cross Validation ---
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
                
                # Aumentamos iterações para evitar warning de convergência
                params_cv = config.MODEL_PARAMS.copy()
                params_cv['max_iter'] = 2000 
                
                mlp_cv = MLPClassifier(**params_cv)
                mlp_cv.fit(X_t_res, y_t_res)
                
                score = f1_score(y_v, mlp_cv.predict(X_v), average='macro')
                f1_scores.append(score)

            mean_cv_f1 = np.mean(f1_scores)
            mlflow.log_metric("cv_mean_f1", mean_cv_f1)
            logger.info(f"CV F1-Macro: {mean_cv_f1:.4f}")

            # --- Treino Final ---
            logger.info("Treinando modelo final...")
            fe_final = FeatureEngineer()
            fe_final.fit(df_train)
            X_final, y_final = fe_final.transform(df_train)
            
            smote_final = SMOTE(random_state=config.MODEL_PARAMS['random_state'])
            X_res, y_res = smote_final.fit_resample(X_final, y_final)
            
            model_final = MLPClassifier(**config.MODEL_PARAMS)
            model_final.fit(X_res, y_res)
            
            # --- Avaliação Final ---
            X_test_scaled, y_test_real = fe_final.transform(df_test)
            y_pred_test = model_final.predict(X_test_scaled)
            
            test_f1 = f1_score(y_test_real, y_pred_test, average='macro')
            test_acc = accuracy_score(y_test_real, y_pred_test)
            
            mlflow.log_metric("test_f1_macro", test_f1)
            mlflow.log_metric("test_accuracy", test_acc)
            logger.info(f"Test F1: {test_f1:.4f} | Acc: {test_acc:.4f}")

            # --- Salvando Artefatos ---
            save_artifact(model_final, config.MODEL_PATH)
            save_artifact(fe_final, config.PIPELINE_PATH)
            
            cm_path = plot_confusion_matrix(y_test_real, y_pred_test)
            mlflow.log_artifact(cm_path)
            mlflow.sklearn.log_model(model_final, "mlp_model")
            
            # --- UPLOAD DO LOG TEMPORÁRIO ---
            # Aqui está a mágica: enviamos o arquivo temp para o MLflow
            # mas renomeamos ele para "training.log" dentro do MLflow para ficar bonito
            mlflow.log_artifact(temp_log_path, artifact_path="logs")
            logger.info("Pipeline concluído e registrado no MLflow!")

        except Exception as e:
            logger.critical(f"Erro no pipeline: {e}")
            raise e
        finally:
            # Limpeza final do arquivo temporário do sistema
            try:
                # Fecha handlers para liberar o arquivo
                for handler in logging.root.handlers:
                    handler.close()
                if os.path.exists(temp_log_path):
                    os.remove(temp_log_path)
            except Exception as e:
                print(f"Erro ao limpar log temporário: {e}")

if __name__ == "__main__":
    train_pipeline()