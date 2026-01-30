from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupShuffleSplit # <--- O segredo anti-leakage
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
from src.utils import load_data, save_artifact
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer

def train_pipeline():
    print("1. Carregando dados...")
    dict_abas = load_data('data/BASE DE DADOS PEDE 2024 - DATATHON.xlsx')
    
    print("2. Pré-processamento inicial...")
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.run(dict_abas)
    
    print("3. Divisão Treino/Teste por GRUPO DE ALUNOS (RA)...")
    
    # --- CORREÇÃO DO PONTO 3: SEPARAÇÃO POR RA ---
    # Isso garante que o mesmo aluno NÃO apareça em treino e teste ao mesmo tempo
    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    
    # Obtém os índices de treino e teste baseados no RA
    train_idx, test_idx = next(splitter.split(df_clean, groups=df_clean['RA']))
    
    df_train = df_clean.iloc[train_idx].copy()
    df_test = df_clean.iloc[test_idx].copy()
    
    print(f"   Total de Alunos Únicos: {df_clean['RA'].nunique()}")
    print(f"   Alunos no Treino: {df_train['RA'].nunique()}")
    print(f"   Alunos no Teste: {df_test['RA'].nunique()}")
    
    # Feature Engineering
    fe = FeatureEngineer()
    fe.fit(df_train)
    
    X_train_scaled, y_train = fe.transform(df_train)
    
    # SMOTE (Balanceamento)
    print("4. Aplicando SMOTE...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    print(f"5. Treinando MLP (Completo)...")
    
    # Configuração MLP (Otimizada)
    mlp = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='tanh',
        alpha=0.0001,
        learning_rate_init=0.001,
        solver='adam',
        max_iter=3000, 
        random_state=42
    )
    
    mlp.fit(X_resampled, y_resampled)
    print("Modelo treinado com sucesso!")
    
    # Salvando
    save_artifact(mlp, 'models/mlp_model.joblib')
    save_artifact(fe, 'models/pipeline_features.joblib')
    
    df_test.to_csv('data/test_dataset.csv', index=False)
    print("Pipeline concluído.")

if __name__ == "__main__":
    train_pipeline()