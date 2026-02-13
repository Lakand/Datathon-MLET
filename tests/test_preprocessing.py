# tests/test_preprocessing.py
"""Módulo de Testes Unitários para o Pré-processamento.

Este módulo valida a capacidade da classe DataPreprocessor de lidar com
inconsistências estruturais e de dados provenientes de fontes externas (Excel),
garantindo a integridade do dataset antes da etapa de modelagem.
"""

import pandas as pd
from src.preprocessing import DataPreprocessor


def test_preprocessing_flow():
    """Valida o pipeline de limpeza e normalização de dados brutos.

    O teste simula a ingestão de múltiplas abas com formatos de colunas distintos
    (safras 2022 e 2024), contendo ruídos propositais como RAs alfabéticos,
    datas incorretas na idade e erros ortográficos na variável alvo.

    Verificações realizadas:
        1. Descarte de abas fora do escopo de análise.
        2. Unificação de nomenclaturas de colunas entre diferentes anos.
        3. Sanitização do Registro Acadêmico (RA) para formato numérico.
        4. Tratamento de anomalias de tipo (datas interpretadas como idade).
        5. Correção de strings na coluna 'PEDRA'.
    """
    # Preparação de dados brutos simulando a safra 2022 com inconsistências
    df_2022 = pd.DataFrame({
        'RA': ['123-4', '5678', 'ABC'], 
        'Idade 22': ['15', '1900-01-14', '20'], 
        'Gênero': ['Menino', 'Menina', 'Masculino'],
        'Fase': ['Fase 1', '2', '3'],
        'Matem': [5.0, 6.0, 7.0],
        'Portug': [5.0, 6.0, 7.0],
        'Inglês': [5.0, 6.0, 7.0],
        'Pedra 22': ['Ametista', 'Agata', 'Quartzo'] 
    })

    # Preparação de dados simulando a safra 2024 em formato padrão
    df_2024 = pd.DataFrame({
        'RA': ['9999'],
        'Idade': [16],
        'Gênero': ['Feminino'],
        'Fase': [3],
        'Mat': [8.0],
        'Por': [8.0],
        'Ing': [8.0],
        'Pedra 2024': ['Topázio']
    })

    # Mapeamento simulado de múltiplas abas para processamento em lote
    mock_dict = {
        "Dados 2022": df_2022,
        "Dados 2024": df_2024,
        "Lixo": pd.DataFrame() 
    }

    processor = DataPreprocessor()
    df_final = processor.run(mock_dict)

    # Asserções de integridade e limpeza de dados
    
    # Valida a filtragem de registros (descarte de RA inválido)
    assert len(df_final) == 3
    
    # Valida a normalização de caracteres no RA
    assert '1234' in df_final['RA'].values
    assert 'ABC' not in df_final['RA'].values
    
    # Valida o mapeamento de colunas para o esquema interno
    assert 'NOTA_MAT' in df_final.columns
    assert 'ANO_DATATHON' in df_final.columns
    
    # Valida a correção ortográfica do target qualitativo
    assert 'Ágata' in df_final['PEDRA'].values
    
    # Valida a conformidade de tipos de dados para processamento numérico
    assert pd.api.types.is_numeric_dtype(df_final['IDADE'])