# tests/test_feature_engineering.py
"""Testes Unitários de Engenharia de Features e Utilitários.

Este módulo valida a lógica de transformação de dados da classe FeatureEngineer,
garantindo que o pré-processamento para o modelo (tratamento de nulos,
scaling e encoding de variáveis) ocorra sem erros. Também inclui testes
para as funções utilitárias de persistência de objetos (salvar/carregar).
"""

import pandas as pd
import numpy as np
import os
from src.feature_engineering import FeatureEngineer
from src.utils import save_artifact, load_artifact

def test_feature_engineering_full():
    """Valida o ciclo completo de engenharia de features (Fit e Transform).

    Simula um DataFrame de entrada contendo:
    - Variáveis categóricas (Gênero, Pedra).
    - Variáveis numéricas com valores ausentes (NaN).
    
    Verifica se:
    1. O pipeline processa os dados sem erros.
    2. Valores nulos são imputados corretamente (não restam NaNs).
    3. A variável alvo 'PEDRA' é convertida para formato numérico (inteiro).
    4. As dimensões do output correspondem ao input.
    """
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
    
    # Verifica se é um tipo inteiro de forma robusta (compatível com Pandas/Numpy)
    assert pd.api.types.is_integer_dtype(y_processed)

def test_utils_save_load(tmp_path: os.PathLike):
    """Testa as funções de persistência de artefatos (I/O).

    Utiliza a fixture `tmp_path` do pytest para criar um diretório temporário
    isolado, garantindo que o teste não deixe lixo no sistema de arquivos real.

    Verifica:
    1. Se o arquivo é salvo corretamente no disco.
    2. Se o objeto carregado é idêntico ao original.

    Args:
        tmp_path (os.PathLike): Fixture do pytest que fornece um caminho de diretório temporário.
    """
    # Testa as funções do utils.py usando uma pasta temporária
    arquivo = tmp_path / "teste.joblib"
    dados = {"chave": "valor"}
    
    # Salva
    save_artifact(dados, str(arquivo))
    assert os.path.exists(arquivo)
    
    # Carrega
    carregado = load_artifact(str(arquivo))
    assert carregado["chave"] == "valor"