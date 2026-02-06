import pandas as pd
import numpy as np
import pytest
from src.feature_engineering import FeatureEngineer
from src.utils import save_artifact, load_artifact
import os

def test_feature_engineering_full():
    # 1. Dados simulados já pré-processados
    df_input = pd.DataFrame({
        'RA': ['1', '2', '3'],
        'GENERO': ['Masculino', 'Feminino', 'Masculino'],
        'PEDRA': ['Ametista', 'Topázio', 'Quartzo'],
        'NOTA_MAT': [5.0, np.nan, 8.0], # Um nulo para testar inputação
        'NOTA_PORT': [6.0, 7.0, 8.0],
        'NOTA_ING': [6.0, 7.0, 8.0],
        'IPP': [5.0, 6.0, 7.0],
        'IAA': [5.0, 6.0, 7.0],
        'IEG': [5.0, 6.0, 7.0],
        'IPS': [5.0, 6.0, 7.0],
        'IDA': [5.0, 6.0, 7.0],
        'IPV': [5.0, 6.0, 7.0],
        'IAN': [5.0, 6.0, 7.0],
        'IDADE': [15, 16, 17],
        'ANO_INGRESSO': [2020, 2021, 2022],
        'FASE': [1, 2, 3],
        'DEFASAGEM': [0, 1, 0]
    })

    # 2. Testar FIT e TRANSFORM
    engineer = FeatureEngineer()
    
    # Treina (fit) e transforma
    X_processed, y_processed = engineer.fit_transform(df_input)
    
    # Validações
    assert X_processed.shape[0] == 3
    # Verifica se preencheu o Nulo de matemática (não deve ter NaN)
    assert not np.isnan(X_processed).any()
    
    # Verifica se a Pedra foi transformada em número (Target)
    assert y_processed is not None
    assert len(y_processed) == 3
    
    # CORREÇÃO: Verifica se é um tipo inteiro de forma robusta (compatível com Pandas/Numpy)
    assert pd.api.types.is_integer_dtype(y_processed)

def test_utils_save_load(tmp_path):
    # Testa as funções do utils.py usando uma pasta temporária
    arquivo = tmp_path / "teste.joblib"
    dados = {"chave": "valor"}
    
    # Salva
    save_artifact(dados, str(arquivo))
    assert os.path.exists(arquivo)
    
    # Carrega
    carregado = load_artifact(str(arquivo))
    assert carregado["chave"] == "valor"