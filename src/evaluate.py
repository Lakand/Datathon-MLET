# src/evaluate.py
"""Módulo de Avaliação de Desempenho.

Este script automatiza o carregamento de modelos MLP e pipelines de features
para validação contra o dataset de holdout, gerando métricas de classificação
necessárias para a auditoria do modelo.
"""

import json
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from src.utils import load_artifact
from src import config 


def evaluate_model() -> dict:
    """Realiza a avaliação completa do modelo contra o conjunto de teste.

    A função orquestra o carregamento de artefatos serializados, a aplicação
    da engenharia de features nos dados de teste e a extração de métricas
    estatísticas (Classification Report e Matriz de Confusão).

    Returns:
        dict: Resultado da avaliação. Em caso de sucesso, retorna o status e 
            as métricas. Em caso de falha (IO ou Transformação), retorna o 
            erro encontrado.
    """
    print("1. Carregando artefatos e dados de teste...")
    try:
        mlp = load_artifact(config.MODEL_PATH)
        fe = load_artifact(config.PIPELINE_PATH)
        df_test = pd.read_csv(config.TEST_DATA_PATH)
    except FileNotFoundError as e:
        return {
            "error": f"Arquivos não encontrados. Treine o modelo primeiro. Detalhe: {str(e)}"
        }
    except Exception as e:
        return {"error": f"Erro inesperado ao carregar modelo: {str(e)}"}

    print("2. Transformando dados de teste...")
    try:
        X_test, y_test = fe.transform(df_test)
    except Exception as e:
        return {"error": f"Erro na transformação dos dados: {str(e)}"}
    
    print("3. Realizando previsões...")
    y_pred = mlp.predict(X_test)
    
    # Mapeamento qualitativo das classes para o relatório final
    target_names = ['Quartzo', 'Ágata', 'Ametista', 'Topázio']
    
    # Compilação das métricas de performance
    report_dict = classification_report(
        y_test, y_pred, target_names=target_names, output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        "status": "success",
        "metrics": {
            "classification_report": report_dict,
            "confusion_matrix": cm.tolist() 
        }
    }


if __name__ == "__main__":
    # Ponto de entrada para execução manual e debug via CLI
    resultado = evaluate_model()
    print(json.dumps(resultado, indent=4))