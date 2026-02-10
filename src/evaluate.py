# src/evaluate.py
"""Módulo de Avaliação do Modelo.

Este script carrega o modelo treinado e o pipeline de features persistidos,
aplica-os ao conjunto de dados de teste (holdout) e calcula métricas de
desempenho como Relatório de Classificação e Matriz de Confusão.
"""

import json
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from src.utils import load_artifact
from src.feature_engineering import FeatureEngineer
from src import config 

def evaluate_model() -> dict:
    """Avalia o modelo atual contra o dataset de teste.

    Realiza as seguintes etapas:
    1. Carrega o modelo (MLPClassifier) e o FeatureEngineer salvos.
    2. Carrega e transforma os dados de teste (test.csv).
    3. Gera predições e compara com os valores reais.
    4. Compila métricas de performance.

    Returns:
        dict: Um dicionário contendo o status da operação e, em caso de sucesso,
        as métricas 'classification_report' e 'confusion_matrix'. Em caso de
        falha, retorna uma chave 'error' com a descrição do problema.
    """
    print("1. Carregando artefatos e dados de teste...")
    try:
        mlp = load_artifact(config.MODEL_PATH)
        fe = load_artifact(config.PIPELINE_PATH)
        df_test = pd.read_csv(config.TEST_DATA_PATH)
    except FileNotFoundError as e:
        return {"error": f"Arquivos não encontrados. Treine o modelo primeiro. Detalhe: {str(e)}"}
    except Exception as e:
        return {"error": f"Erro inesperado ao carregar modelo: {str(e)}"}

    print("2. Transformando dados de teste...")
    try:
        X_test, y_test = fe.transform(df_test)
    except Exception as e:
        return {"error": f"Erro na transformação dos dados: {str(e)}"}
    
    print("3. Realizando previsões...")
    y_pred = mlp.predict(X_test)
    
    target_names = ['Quartzo', 'Ágata', 'Ametista', 'Topázio']
    
    report_dict = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        "status": "success",
        "metrics": {
            "classification_report": report_dict,
            "confusion_matrix": cm.tolist() 
        }
    }

if __name__ == "__main__":
    resultado = evaluate_model()
    print(json.dumps(resultado, indent=4))