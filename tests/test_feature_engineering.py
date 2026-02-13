# tests/test_feature_engineering.py
"""Testes Unitários de Engenharia de Features e Utilitários.

Este módulo valida a lógica de transformação de dados da classe FeatureEngineer
e a integridade das funções de persistência de artefatos. Garante que o pipeline
de Machine Learning receba dados normalizados e que o armazenamento em disco
seja resiliente.
"""

import os
import pandas as pd
import numpy as np
from src.feature_engineering import FeatureEngineer
from src.utils import save_artifact, load_artifact


def test_feature_engineering_full():
    """Valida o ciclo completo de fit e transform do FeatureEngineer.

    Simula a entrada de dados após o processamento bruto, garantindo que:
    1. Variáveis categóricas (Gênero) já estejam em formato binário.
    2. Valores numéricos ausentes sejam imputados corretamente.
    3. O mapeamento do target (Pedra) seja convertido para inteiros.

    Returns:
        None: O teste falha caso as asserções de integridade de shape, 
            imputação de nulos ou tipagem do target não sejam atendidas.
    """
    # Preparação de dataset simulado com GENERO numérico e nulos em NOTA_MAT
    df_input = pd.DataFrame({
        'RA': ['1', '2', '3'],
        'GENERO': [0, 1, 0], 
        'PEDRA': ['Ametista', 'Topázio', 'Quartzo'],
        'NOTA_MAT': [5.0, np.nan, 8.0],
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

    engineer = FeatureEngineer()
    X_processed, y_processed = engineer.fit_transform(df_input)
    
    # Asserções de conformidade estrutural e de dados
    assert X_processed.shape[0] == 3
    assert not np.isnan(X_processed).any()
    assert y_processed is not None
    assert len(y_processed) == 3
    assert pd.api.types.is_integer_dtype(y_processed)


def test_utils_save_load(tmp_path):
    """Testa a persistência e integridade de artefatos via joblib.

    Utiliza um diretório temporário para validar se dicionários e modelos
    são gravados e lidos corretamente sem perda de informação.

    Args:
        tmp_path: Fixture do Pytest que fornece um diretório temporário isolado.
    """
    arquivo = tmp_path / "teste.joblib"
    dados = {"chave": "valor"}
    
    # Validação do fluxo de I/O
    save_artifact(dados, str(arquivo))
    assert os.path.exists(arquivo)
    
    carregado = load_artifact(str(arquivo))
    assert carregado["chave"] == "valor"