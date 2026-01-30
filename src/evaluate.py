import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from src.utils import load_artifact
from src.feature_engineering import FeatureEngineer # Necessário para o pickle carregar a classe

def evaluate_model():
    print("1. Carregando artefatos e dados de teste...")
    try:
        mlp = load_artifact('models/mlp_model.joblib')
        fe = load_artifact('models/pipeline_features.joblib')
        df_test = pd.read_csv('data/test_dataset.csv')
    except FileNotFoundError as e:
        print(e)
        return

    print("2. Transformando dados de teste...")
    # Usa o Feature Engineer JÁ TREINADO (não faz fit aqui)
    X_test, y_test = fe.transform(df_test)
    
    print("3. Realizando previsões...")
    y_pred = mlp.predict(X_test)
    
    print("\n" + "="*40)
    print("RELATÓRIO DE CLASSIFICAÇÃO - MLP")
    print("="*40)
    
    target_names = ['Quartzo', 'Ágata', 'Ametista', 'Topázio'] # Ordem do mapa no FeatureEngineer
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    print("\nMATRIZ DE CONFUSÃO:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    evaluate_model()