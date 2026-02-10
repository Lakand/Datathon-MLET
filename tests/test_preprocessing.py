# tests/test_preprocessing.py
"""Testes Unitários de Pré-processamento.

Este módulo valida a lógica de limpeza e padronização de dados da classe
DataPreprocessor. Assegura que o sistema lide corretamente com inconsistências
comuns encontradas nos arquivos Excel de entrada (ex: datas formatadas incorretamente,
nomes de colunas variados entre anos e RAs inválidos).
"""

import pandas as pd
import pytest
from src.preprocessing import DataPreprocessor

def test_preprocessing_flow():
    """Valida o fluxo completo de pré-processamento de dados brutos.

    Cria um cenário simulado com dados de 2022 e 2024 contendo intencionalmente
    erros de formatação e sujeira (dirty data). O teste verifica se:
    1. Abas irrelevantes são ignoradas.
    2. Colunas de diferentes anos são mapeadas para nomes padronizados.
    3. Registros com RA inválido (não numérico) são removidos.
    4. Formatações de data na coluna de idade são corrigidas.
    5. Erros de digitação na coluna 'PEDRA' (ex: 'Agata') são ajustados.

    O teste garante que o DataFrame resultante esteja limpo e pronto para
    a etapa de engenharia de features.
    """
    # Simulação de dados brutos (Mock) representando a aba de 2022
    # Inclui casos de borda: RA com pontuação, data tipo Excel em Idade e erro de digitação na Pedra
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

    # Simulação de dados brutos para 2024 (formato limpo)
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

    # Dicionário que simula o retorno do pd.read_excel com múltiplas abas
    mock_dict = {
        "Dados 2022": df_2022,
        "Dados 2024": df_2024,
        "Lixo": pd.DataFrame() # Aba propositalmente irrelevante para testar robustez
    }

    # Execução do pipeline de pré-processamento
    processor = DataPreprocessor()
    df_final = processor.run(mock_dict)

    # Validação dos Resultados
    
    # Verifica a remoção de linhas inválidas (RA 'ABC' deve ser descartado)
    # Espera-se 3 registros válidos: '1234', '5678', '9999'
    assert len(df_final) == 3
    
    # Valida a limpeza de caracteres não numéricos no RA ('123-4' -> '1234')
    assert '1234' in df_final['RA'].values
    
    # Confirma que o RA puramente alfabético foi removido
    assert 'ABC' not in df_final['RA'].values
    
    # Verifica se as colunas foram renomeadas para o padrão interno (ex: 'Matem' -> 'NOTA_MAT')
    assert 'NOTA_MAT' in df_final.columns
    assert 'ANO_DATATHON' in df_final.columns
    
    # Verifica a correção ortográfica da variável alvo ('Agata' -> 'Ágata')
    assert 'Ágata' in df_final['PEDRA'].values
    
    # Verifica se a conversão de tipos funcionou para a coluna de idade (deve ser numérica)
    assert pd.api.types.is_numeric_dtype(df_final['IDADE'])