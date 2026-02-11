# tests/test_coverage_boost.py
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# Imports dos módulos que vamos testar
from src import evaluate, drift_report
from app.main import app

# ==========================================
# 1. Testes para src/evaluate.py
# ==========================================

def test_evaluate_success():
    """Testa o caminho feliz do evaluate (tudo funciona)."""
    # Mock do modelo
    mock_model = MagicMock()
    # PRECISÃO: Retorna as 4 classes (0,1,2,3) para bater com os target_names
    mock_model.predict.return_value = np.array([0, 1, 2, 3])
    
    mock_fe = MagicMock()
    # Retorna X (array) e y (array) simulados com 4 registros
    mock_fe.transform.return_value = (
        np.array([[1, 2], [3, 4], [5, 6], [7, 8]]), 
        np.array([0, 1, 2, 3])
    )
    
    # Mock do CSV de teste
    df_mock = pd.DataFrame({'col1': [1, 2, 3, 4]})
    
    with patch('src.evaluate.load_artifact', side_effect=[mock_model, mock_fe]):
        with patch('pandas.read_csv', return_value=df_mock):
            result = evaluate.evaluate_model()
            
    assert "metrics" in result
    assert "classification_report" in result["metrics"]

def test_evaluate_file_not_found():
    """Testa se o evaluate captura corretamente o erro de arquivo inexistente."""
    with patch('src.evaluate.load_artifact', side_effect=FileNotFoundError("Modelo sumiu")):
        result = evaluate.evaluate_model()
    
    assert "error" in result
    assert "Arquivos não encontrados" in result["error"]

def test_evaluate_transform_error():
    """Testa se o evaluate captura erros durante a transformação dos dados."""
    mock_model = MagicMock()
    mock_fe = MagicMock()
    mock_fe.transform.side_effect = Exception("Erro de calculo")
    
    with patch('src.evaluate.load_artifact', side_effect=[mock_model, mock_fe]):
        with patch('pandas.read_csv', return_value=pd.DataFrame()):
             result = evaluate.evaluate_model()
             
    assert "error" in result
    assert "Erro na transformação" in result["error"]

# ==========================================
# 2. Testes para src/drift_report.py
# ==========================================

def test_drift_load_prod_no_db():
    """Testa load_production_data quando o banco não existe."""
    with patch('os.path.exists', return_value=False):
        df = drift_report.load_production_data()
        assert df.empty

def test_drift_load_prod_success():
    """Testa load_production_data lendo e parseando JSON corretamente."""
    mock_conn = MagicMock()
    
    # Simula retorno do SQL: JSON string na coluna input_data
    # NOTA: O código atual do drift_report.py lê APENAS 'input_data', ignorando 'predicted_pedra'
    df_sql = pd.DataFrame({
        'input_data': ['{"NOTA_MAT": 5.5}', '{"NOTA_MAT": 8.0}']
    })
    
    with patch('os.path.exists', return_value=True):
        with patch('sqlite3.connect', return_value=mock_conn):
            with patch('pandas.read_sql_query', return_value=df_sql):
                df = drift_report.load_production_data()
                
    assert not df.empty
    assert 'NOTA_MAT' in df.columns # Verifica se expandiu o JSON

def test_drift_generate_ref_fail():
    """Testa falha ao carregar dados de treino (referência)."""
    # Simula erro no load_data
    with patch('src.drift_report.load_data', side_effect=Exception("Erro Excel")):
        result = drift_report.generate_report()
        assert result is None

def test_drift_generate_success(tmp_path):
    """Testa o fluxo completo de geração do HTML (Modo Validação)."""
    # 1. Mock dos dados de Treino (Referência)
    df_ref_mock = pd.DataFrame({
        'RA': ['1', '2'], 
        'PEDRA': ['Ametista', 'Topázio'],
        'NOTA_MAT': [5.0, 6.0]
    })
    
    # 2. Mock dos dados de Produção (Vazio ou < 100 para cair no modo validação)
    df_prod_mock = pd.DataFrame() 
    
    # 3. Mock do Dataset de Teste
    df_test_mock = pd.DataFrame({'NOTA_MAT': [5.0, 6.0]})
    
    # 4. Mock do Feature Engineer
    mock_fe = MagicMock()
    # O transform deve retornar (X_scaled, y) ou apenas X_scaled dependendo de como é chamado.
    # No drift_report.py, ele chama fe.transform(df) e espera retornar X_scaled, y (mas usa _ para o y)
    mock_fe.transform.return_value = (np.array([[0.5], [0.6]]), None)
    mock_fe.cols_treino = ['NOTA_MAT']

    # Patching de tudo necessário
    with patch('src.drift_report.load_data', return_value={}): # Mock do Excel bruto
        with patch('src.drift_report.DataPreprocessor') as MockPrep:
            # Configura o mock do preprocessor para retornar o df_ref_mock
            MockPrep.return_value.run.return_value = df_ref_mock
            MockPrep.return_value.clean_dataframe.return_value = df_test_mock
            
            with patch('src.drift_report.load_artifact', return_value=mock_fe):
                with patch('src.drift_report.load_production_data', return_value=df_prod_mock):
                    with patch('pandas.read_csv', return_value=df_test_mock):
                        with patch('src.drift_report.Report') as MockReport:
                            
                            # Executa
                            drift_report.generate_report()
                            
                            # Verifica se o Report do Evidently foi chamado e salvo
                            MockReport.return_value.run.assert_called()
                            MockReport.return_value.save_html.assert_called()

# ==========================================
# 3. Testes para app/main.py
# ==========================================

def test_lifespan_load_failure():
    """Testa se a API inicia mesmo se não encontrar os modelos."""
    with patch('joblib.load', side_effect=FileNotFoundError):
        with TestClient(app) as client:
            assert client.app.state.model is None

def test_lifespan_generic_error():
    """Testa erro genérico no startup."""
    with patch('joblib.load', side_effect=Exception("Erro bizarro")):
        with TestClient(app) as client:
            assert client.app.state.model is None