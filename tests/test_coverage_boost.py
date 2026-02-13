# tests/test_coverage_boost.py
"""Módulo de Testes de Cobertura e Integração.

Este módulo contém suítes de testes unitários para validar os componentes de 
avaliação de modelo, monitoramento de drift e o ciclo de vida da API FastAPI. 
Utiliza mocks extensivos para isolar a lógica de negócios de dependências 
de I/O e modelos serializados.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from src import evaluate, drift_report
from app.main import app

# ==============================================================================
# TESTES: src/evaluate.py
# ==============================================================================

def test_evaluate_success():
    """Valida o fluxo completo de sucesso da avaliação do modelo.

    Simula o carregamento bem-sucedido de artefatos, a transformação de dados
    de teste e a geração do relatório de classificação para as quatro 
    classes definidas.
    """
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0, 1, 2, 3])
    
    mock_fe = MagicMock()
    mock_fe.transform.return_value = (
        np.array([[1, 2], [3, 4], [5, 6], [7, 8]]), 
        np.array([0, 1, 2, 3])
    )
    
    df_mock = pd.DataFrame({'col1': [1, 2, 3, 4]})
    
    with patch('src.evaluate.load_artifact', side_effect=[mock_model, mock_fe]):
        with patch('pandas.read_csv', return_value=df_mock):
            result = evaluate.evaluate_model()
            
    assert "metrics" in result
    assert "classification_report" in result["metrics"]


def test_evaluate_file_not_found():
    """Valida o tratamento de exceção para arquivos de artefatos ausentes.

    Returns:
        Garante que o dicionário de retorno contenha a chave 'error' e uma 
        mensagem explicativa apropriada.
    """
    with patch('src.evaluate.load_artifact', side_effect=FileNotFoundError("Modelo sumiu")):
        result = evaluate.evaluate_model()
    
    assert "error" in result
    assert "Arquivos não encontrados" in result["error"]


def test_evaluate_transform_error():
    """Valida a captura de erros genéricos durante a fase de engenharia de features."""
    mock_model = MagicMock()
    mock_fe = MagicMock()
    mock_fe.transform.side_effect = Exception("Erro de calculo")
    
    with patch('src.evaluate.load_artifact', side_effect=[mock_model, mock_fe]):
        with patch('pandas.read_csv', return_value=pd.DataFrame()):
             result = evaluate.evaluate_model()
             
    assert "error" in result
    assert "Erro na transformação" in result["error"]


# ==============================================================================
# TESTES: src/drift_report.py
# ==============================================================================

def test_drift_load_prod_no_db():
    """Valida o comportamento do carregamento de dados quando o SQLite está ausente."""
    with patch('os.path.exists', return_value=False):
        df = drift_report.load_production_data()
        assert df.empty


def test_drift_load_prod_success():
    """Valida a extração e o parsing de logs de produção armazenados como JSON."""
    mock_conn = MagicMock()
    df_sql = pd.DataFrame({
        'input_data': ['{"NOTA_MAT": 5.5}', '{"NOTA_MAT": 8.0}']
    })
    
    with patch('os.path.exists', return_value=True):
        with patch('sqlite3.connect', return_value=mock_conn):
            with patch('pandas.read_sql_query', return_value=df_sql):
                df = drift_report.load_production_data()
                
    assert not df.empty
    assert 'NOTA_MAT' in df.columns


def test_drift_generate_ref_fail():
    """Valida a interrupção da geração do relatório caso os dados de referência falhem."""
    with patch('src.drift_report.load_data', side_effect=Exception("Erro Excel")):
        result = drift_report.generate_report()
        assert result is None


def test_drift_generate_success(tmp_path):
    """Valida o workflow end-to-end de geração do Data Drift Report via Evidently.

    Verifica se o componente Report é instanciado, executado e se o método 
    de salvamento do arquivo HTML é invocado no modo de validação.
    """
    df_ref_mock = pd.DataFrame({
        'RA': ['1', '2'], 
        'PEDRA': ['Ametista', 'Topázio'],
        'NOTA_MAT': [5.0, 6.0]
    })
    
    df_prod_mock = pd.DataFrame() 
    df_test_mock = pd.DataFrame({'NOTA_MAT': [5.0, 6.0]})
    
    mock_fe = MagicMock()
    mock_fe.transform.return_value = (np.array([[0.5], [0.6]]), None)
    mock_fe.cols_treino = ['NOTA_MAT']

    with patch('src.drift_report.load_data', return_value={}):
        with patch('src.drift_report.DataPreprocessor') as MockPrep:
            MockPrep.return_value.run.return_value = df_ref_mock
            MockPrep.return_value.clean_dataframe.return_value = df_test_mock
            
            with patch('src.drift_report.load_artifact', return_value=mock_fe):
                with patch('src.drift_report.load_production_data', return_value=df_prod_mock):
                    with patch('pandas.read_csv', return_value=df_test_mock):
                        with patch('src.drift_report.Report') as MockReport:
                            drift_report.generate_report()
                            MockReport.return_value.run.assert_called()
                            MockReport.return_value.save_html.assert_called()


# ==============================================================================
# TESTES: app/main.py (Ciclo de Vida da API)
# ==============================================================================

def test_lifespan_load_failure():
    """Valida a resiliência do startup da API caso o modelo não possa ser carregado."""
    with patch('joblib.load', side_effect=FileNotFoundError):
        with TestClient(app) as client:
            assert client.app.state.model is None


def test_lifespan_generic_error():
    """Valida a estabilidade do startup diante de exceções genéricas inesperadas."""
    with patch('joblib.load', side_effect=Exception("Erro bizarro")):
        with TestClient(app) as client:
            assert client.app.state.model is None