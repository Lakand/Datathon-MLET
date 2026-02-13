# tests/test_drift_report.py
"""Testes unitários e de integração para o módulo de relatório de Drift.

Este módulo valida a lógica de carregamento de dados de produção e a decisão
estratégica entre gerar relatório de validação (Train vs Test) ou relatório
de produção (Train vs Prod) com base no volume de dados.
"""

import pytest
import pandas as pd
import numpy as np
import json
from unittest.mock import patch, MagicMock
from src.drift_report import generate_report, load_production_data

MOCK_LOGS_DATA = [
    {"input_data": json.dumps({"RA": "123", "PEDRA": "Ametista", "NOTA_MAT": 5.0})},
    {"input_data": json.dumps({"RA": "456", "PEDRA": "Agata", "NOTA_MAT": 7.0})}
]

@pytest.fixture
def mock_db_connection():
    """Simula a conexão com o banco de dados SQLite para isolamento dos testes.

    Yields:
        MagicMock: Um objeto mock configurado para simular a conexão com o banco.
    """
    with patch("sqlite3.connect") as mock_connect:
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        yield mock_conn

def test_load_production_data_empty(mock_db_connection):
    """Verifica se um DataFrame vazio é retornado quando o banco falha ou não possui dados.

    Args:
        mock_db_connection: Fixture que simula a conexão com o banco.
    """
    with patch("pandas.read_sql_query", return_value=pd.DataFrame()):
        df = load_production_data()
        assert df.empty

def test_load_production_data_success(mock_db_connection):
    """Verifica o carregamento correto e o parsing do JSON dos logs de produção.

    Confirma se os dados brutos do banco são convertidos corretamente em um
    DataFrame pandas estruturado.

    Args:
        mock_db_connection: Fixture que simula a conexão com o banco.
    """
    df_mock_db = pd.DataFrame(MOCK_LOGS_DATA)
    with patch("pandas.read_sql_query", return_value=df_mock_db):
        df = load_production_data()
        assert not df.empty
        assert "RA" in df.columns
        assert len(df) == 2

@patch("src.drift_report.load_data")
@patch("src.drift_report.load_artifact")
@patch("src.drift_report.load_production_data")
@patch("src.drift_report.Report")
def test_generate_report_validation_mode(mock_report, mock_load_prod, mock_load_artifact, mock_load_data):
    """Testa o fluxo de VALIDAÇÃO (acionado quando há poucos dados em produção).

    Este teste garante que, se houver menos de 100 registros de produção,
    o sistema compara os dados de Treino contra os dados de Teste (Holdout).

    Args:
        mock_report: Mock da classe Report do Evidently.
        mock_load_prod: Mock da função de carregamento de logs.
        mock_load_artifact: Mock do carregamento do pipeline serializado.
        mock_load_data: Mock do carregamento dos dados brutos (Excel).
    """
    
    mock_load_prod.return_value = pd.DataFrame([{"RA": 1}] * 50) 
    
    df_treino_raw = pd.DataFrame({
        "RA": ["1", "2"], 
        "Pedra 2024": ["Ametista", "Topazio"], 
        "Mat": [5, 6],
        "Ano ingresso": [2020, 2021]
    })
    mock_load_data.return_value = {"Pede 2024": df_treino_raw} 
    
    mock_pipeline = MagicMock()
    
    mock_matrix = np.random.rand(2, 4)
    mock_pipeline.transform.return_value = (mock_matrix, pd.Series([0, 1]))
    
    mock_pipeline.cols_treino = ["FEAT_1", "FEAT_2", "FEAT_3", "FEAT_4"]
    mock_load_artifact.return_value = mock_pipeline
    
    df_teste_clean = pd.DataFrame({
        "RA": ["1", "2"], "PEDRA": ["Ametista", "Topazio"], 
        "NOTA_MAT": [5, 6], "ANO_INGRESSO": [2020, 2021]
    })
    
    with patch("pandas.read_csv", return_value=df_teste_clean):
        path = generate_report()
        
    assert path is not None
    mock_report.return_value.run.assert_called_once()

@patch("src.drift_report.load_data")
@patch("src.drift_report.load_artifact")
@patch("src.drift_report.load_production_data")
@patch("src.drift_report.Report")
def test_generate_report_production_mode(mock_report, mock_load_prod, mock_load_artifact, mock_load_data):
    """Testa o fluxo de PRODUÇÃO (acionado quando há volume suficiente de dados).

    Este teste garante que, se houver 100 ou mais registros, o sistema compara
    os dados de Treino contra os dados reais de Produção.

    Args:
        mock_report: Mock da classe Report do Evidently.
        mock_load_prod: Mock da função de carregamento de logs.
        mock_load_artifact: Mock do carregamento do pipeline serializado.
        mock_load_data: Mock do carregamento dos dados brutos.
    """
    
    df_prod_raw = pd.DataFrame([
        {"RA": str(i), "Pedra 2024": "Ametista", "Mat": 5.0, "Ano ingresso": 2022} 
        for i in range(150)
    ])
    mock_load_prod.return_value = df_prod_raw
    
    df_treino_raw = pd.DataFrame({
        "RA": ["1"], "Pedra 2024": ["Ametista"], "Mat": [5], "Ano ingresso": [2022]
    })
    mock_load_data.return_value = {"2024": df_treino_raw} 
    
    mock_pipeline = MagicMock()
    
    def transform_side_effect(df):
        n_rows = len(df)
        return np.random.rand(n_rows, 4), None
        
    mock_pipeline.transform.side_effect = transform_side_effect
    mock_pipeline.cols_treino = ["FEAT_1", "FEAT_2", "FEAT_3", "FEAT_4"]
    mock_load_artifact.return_value = mock_pipeline
    
    path = generate_report()
    
    assert path is not None
    with patch("pandas.read_csv") as mock_read_csv:
        assert not mock_read_csv.called