from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import joblib
from src.utils import load_data, save_artifact
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer

def train_pipeline():
    print("1. Carregando dados...")
    dict_abas = load_data('data/BASE DE DADOS PEDE 2024 - DATATHON.xlsx')
    
    print("2. Pré-processamento inicial...")
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.run(dict_abas)
    
    # Validação rápida de RAs nulos
    df_clean = df_clean.dropna(subset=['RA']).reset_index(drop=True)
    
    print("3. Divisão Treino/Teste (Holdout) por RA...")
    # Mantemos o GroupShuffleSplit para separar um teste final intocado
    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, test_idx = next(splitter.split(df_clean, groups=df_clean['RA']))
    
    df_train = df_clean.iloc[train_idx].copy().reset_index(drop=True)
    df_test = df_clean.iloc[test_idx].copy().reset_index(drop=True)
    
    print(f"   Alunos no Treino: {df_train['RA'].nunique()}")
    print(f"   Alunos no Teste (Final): {df_test['RA'].nunique()}")

    # --- INÍCIO DA VALIDAÇÃO CRUZADA (CROSS VALIDATION) ---
    print("\n4. Iniciando Cross-Validation (K-Fold) no Treino...")
    
    # Usamos StratifiedGroupKFold para garantir que:
    # 1. O mesmo RA não apareça em treino e validação (Group)
    # 2. A proporção das classes (Pedras) seja mantida (Stratified)
    sgkf = StratifiedGroupKFold(n_splits=10)
    
    f1_scores = []
    fold = 1
    
    # Precisamos preparar os arrays para o Split
    X_cv = df_train # O FeatureEngineer vai selecionar as colunas internamente
    y_cv = df_train['PEDRA'] # Apenas para o split, será processado depois
    groups_cv = df_train['RA']
    
    # Loop de Validação
    for t_idx, v_idx in sgkf.split(X_cv, y_cv, groups=groups_cv):
        # Separa os dados da dobra
        cv_train = df_train.iloc[t_idx].copy()
        cv_val = df_train.iloc[v_idx].copy()
        
        # Pipeline Local (Feature Engineering -> SMOTE -> Modelo)
        fe_cv = FeatureEngineer()
        fe_cv.fit(cv_train)
        
        X_t_scaled, y_t = fe_cv.transform(cv_train)
        X_v_scaled, y_v = fe_cv.transform(cv_val)
        
        # SMOTE apenas no treino da dobra
        smote = SMOTE(random_state=42)
        X_t_res, y_t_res = smote.fit_resample(X_t_scaled, y_t)
        
        # Modelo MLP
        mlp_cv = MLPClassifier(
            hidden_layer_sizes=(50,),
            activation='relu',
            alpha=0.01,
            learning_rate_init=0.001,
            max_iter=1000, # Menos iteracoes para ser mais rapido no CV
            random_state=42
        )
        mlp_cv.fit(X_t_res, y_t_res)
        
        # Avaliação
        y_pred_val = mlp_cv.predict(X_v_scaled)
        score = f1_score(y_v, y_pred_val, average='macro')
        f1_scores.append(score)
        print(f"   Fold {fold}: F1-Macro = {score:.4f}")
        fold += 1
        
    print(f"   >>> Média F1-Score (CV): {np.mean(f1_scores):.4f} (+/- {np.std(f1_scores):.4f})")
    
    # --- TREINO FINAL (FULL DATASET) ---
    print("\n5. Treinando Modelo Final (Todo o Dataset de Treino)...")
    
    # 1. Feature Engineering Global
    fe_final = FeatureEngineer()
    fe_final.fit(df_train)
    X_train_final, y_train_final = fe_final.transform(df_train)
    
    # 2. SMOTE Global
    smote_final = SMOTE(random_state=42)
    X_resampled, y_resampled = smote_final.fit_resample(X_train_final, y_train_final)
    
    # 3. Modelo Final (Melhores Hiperparâmetros)
    mlp_final = MLPClassifier(
        hidden_layer_sizes=(50,),
        activation='relu',
        alpha=0.01,
        learning_rate_init=0.001,
        solver='adam',
        max_iter=3000, 
        random_state=42
    )
    
    mlp_final.fit(X_resampled, y_resampled)
    print("Modelo Final treinado com sucesso!")
    
    # --- AVALIAÇÃO FINAL (TEST SET) ---
    print("\n6. Avaliação no Conjunto de Teste (Holdout)...")
    X_test_scaled, y_test_real = fe_final.transform(df_test)
    y_pred_test = mlp_final.predict(X_test_scaled)
    
    print(classification_report(y_test_real, y_pred_test))
    
    # Salvando artefatos
    save_artifact(mlp_final, 'models/mlp_model.joblib')
    save_artifact(fe_final, 'models/pipeline_features.joblib')
    
    # Salvando datasets processados para auditoria
    df_test.to_csv('data/test_dataset.csv', index=False)
    print("Pipeline concluído e artefatos salvos.")

if __name__ == "__main__":
    train_pipeline()