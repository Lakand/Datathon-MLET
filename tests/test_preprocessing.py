import pandas as pd
import pytest
from src.preprocessing import DataPreprocessor

def test_preprocessing_flow():
    # 1. Criar dados falsos (Mock) simulando uma aba de 2022 e uma de 2024
    df_2022 = pd.DataFrame({
        'RA': ['123-4', '5678', 'ABC'], # 'ABC' será removido pela limpeza (Correto!)
        'Idade 22': ['15', '1900-01-14', '20'], 
        'Gênero': ['Menino', 'Menina', 'Masculino'],
        'Fase': ['Fase 1', '2', '3'],
        'Matem': [5.0, 6.0, 7.0],
        'Portug': [5.0, 6.0, 7.0],
        'Inglês': [5.0, 6.0, 7.0],
        'Pedra 22': ['Ametista', 'Agata', 'Quartzo'] 
    })

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

    mock_dict = {
        "Dados 2022": df_2022,
        "Dados 2024": df_2024,
        "Lixo": pd.DataFrame() # Aba que deve ser ignorada com warning
    }

    # 2. Instanciar e rodar
    processor = DataPreprocessor()
    df_final = processor.run(mock_dict)

    # 3. Asserções (Validações)
    
    # CORREÇÃO: Esperamos 3 alunos, pois o RA 'ABC' foi removido corretamente.
    assert len(df_final) == 3
    
    # Verifica limpeza de RA (removeu o traço de '123-4')
    assert '1234' in df_final['RA'].values
    
    # Verifica se o RA inválido 'ABC' realmente sumiu
    assert 'ABC' not in df_final['RA'].values
    
    # Verifica padronização de colunas
    assert 'NOTA_MAT' in df_final.columns
    assert 'ANO_DATATHON' in df_final.columns
    
    # Verifica correção de Pedra (Agata -> Ágata)
    assert 'Ágata' in df_final['PEDRA'].values
    
    # Verifica se a idade "bugada" (1900-01-14) virou numérica
    assert pd.api.types.is_numeric_dtype(df_final['IDADE'])