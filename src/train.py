# src/train.py
import logging
import sys
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

# Configuração de Logging (Texto)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training_pipeline.log', mode='w')
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
    
    # Salva no diretório de imagens definido no config
    save_path = config.IMG_DIR / "confusion_matrix.png"
    plt.savefig(save_path)
    plt.close()
    return str(save_path)

def train_pipeline():
    # --- Configura o MLflow ---
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    
    # Inicia uma "Run" (Execução) do MLflow
    with mlflow.start_run():
        logger.info("=== Iniciando Pipeline com Rastreamento MLflow ===")
        
        # 1. Logar Hiperparâmetros (O que usamos para configurar)
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
                # Pipeline Local (Feature Eng -> SMOTE -> Modelo)
                fe_cv = FeatureEngineer()
                fe_cv.fit(df_train.iloc[t_idx])
                X_t, y_t = fe_cv.transform(df_train.iloc[t_idx])
                X_v, y_v = fe_cv.transform(df_train.iloc[v_idx])
                
                smote = SMOTE(random_state=config.MODEL_PARAMS['random_state'])
                X_t_res, y_t_res = smote.fit_resample(X_t, y_t)
                
                # Ajuste de iter para ser rápido no CV
                params_cv = config.MODEL_PARAMS.copy()
                params_cv['max_iter'] = 1000
                mlp_cv = MLPClassifier(**params_cv)
                mlp_cv.fit(X_t_res, y_t_res)
                
                score = f1_score(y_v, mlp_cv.predict(X_v), average='macro')
                f1_scores.append(score)

            # Logar métricas do CV no MLflow
            mean_cv_f1 = np.mean(f1_scores)
            mlflow.log_metric("cv_mean_f1", mean_cv_f1)
            mlflow.log_metric("cv_std_f1", np.std(f1_scores))
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
            
            # --- Avaliação Final (Teste) ---
            X_test_scaled, y_test_real = fe_final.transform(df_test)
            y_pred_test = model_final.predict(X_test_scaled)
            
            # Métricas Finais
            test_f1 = f1_score(y_test_real, y_pred_test, average='macro')
            test_acc = accuracy_score(y_test_real, y_pred_test)
            
            # Logar no MLflow
            mlflow.log_metric("test_f1_macro", test_f1)
            mlflow.log_metric("test_accuracy", test_acc)
            
            logger.info(f"Test F1: {test_f1:.4f} | Acc: {test_acc:.4f}")

            # --- Salvando Artefatos ---
            # 1. Salvar Modelos Locais
            save_artifact(model_final, config.MODEL_PATH)
            save_artifact(fe_final, config.PIPELINE_PATH)
            
            # 2. Gerar e Logar Matriz de Confusão (Imagem)
            cm_path = plot_confusion_matrix(y_test_real, y_pred_test)
            mlflow.log_artifact(cm_path) # Envia a imagem para o MLflow

            # 3. Logar o modelo (formato sklearn nativo)
            mlflow.sklearn.log_model(model_final, "mlp_model")
            
            # 4. Logar o arquivo de texto de logs também
            mlflow.log_artifact("training_pipeline.log")

            logger.info("Pipeline concluído e registrado no MLflow!")

        except Exception as e:
            logger.critical(f"Erro no pipeline: {e}")
            raise e

if __name__ == "__main__":
    train_pipeline()