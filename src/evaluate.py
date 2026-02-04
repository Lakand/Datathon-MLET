# src/evaluate.py
import json
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from src.utils import load_artifact
from src.feature_engineering import FeatureEngineer # Necessário para o joblib reconhecer a classe
from src import config 

def evaluate_model():
    """
    Carrega o modelo atual e avalia contra o dataset de teste.
    Retorna um dicionário com métricas.
    """
    print("1. Carregando artefatos e dados de teste...")
    try:
        # Carrega usando os caminhos do config
        mlp = load_artifact(config.MODEL_PATH)
        fe = load_artifact(config.PIPELINE_PATH)
        df_test = pd.read_csv(config.TEST_DATA_PATH)
    except FileNotFoundError as e:
        return {"error": f"Arquivos não encontrados. Treine o modelo primeiro. Detalhe: {str(e)}"}
    except Exception as e:
        return {"error": f"Erro inesperado ao carregar modelo: {str(e)}"}

    print("2. Transformando dados de teste...")
    # Usa o Feature Engineer JÁ TREINADO (não faz fit aqui)
    try:
        X_test, y_test = fe.transform(df_test)
    except Exception as e:
        return {"error": f"Erro na transformação dos dados: {str(e)}"}
    
    print("3. Realizando previsões...")
    y_pred = mlp.predict(X_test)
    
    # Nomes das classes na ordem correta do mapa
    target_names = ['Quartzo', 'Ágata', 'Ametista', 'Topázio']
    
    # Gera o report como dicionário para ser JSON serializable
    report_dict = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    
    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        "status": "success",
        "metrics": {
            "classification_report": report_dict,
            "confusion_matrix": cm.tolist() # Converte array numpy para lista (JSON não aceita numpy)
        }
    }

if __name__ == "__main__":
    # Se rodar via terminal, imprime bonito
    resultado = evaluate_model()
    print(json.dumps(resultado, indent=4))