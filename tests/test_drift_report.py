#tests/test_drift_report.py
"""Testes unitários e de integração para o módulo de relatório de Drift.

Valida o carregamento de dados, a robustez contra falhas (JSON inválido,
erros de IO) e a lógica de decisão entre modo Produção vs Validação.
"""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.drift_report import (
    apply_feature_engineering_transform,
    generate_report,
    load_production_data,
)


@pytest.fixture
def mock_db_connection():
    """Cria um mock para a conexão com o banco de dados SQLite.

    Yields:
        MagicMock: Um objeto mock simulando a conexão retornada por sqlite3.connect.
    """
    with patch("sqlite3.connect") as mock_connect:
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        yield mock_conn


def test_load_production_data_empty(mock_db_connection):
    """Verifica o comportamento ao carregar dados de produção vazios.

    Args:
        mock_db_connection: Fixture do mock de conexão.

    Returns:
        None: Verifica se o DataFrame retornado está vazio.
    """
    with patch("pandas.read_sql_query", return_value=pd.DataFrame()):
        df = load_production_data()
        assert df.empty


@patch("os.path.exists", return_value=False)
def test_load_production_data_no_file(mock_exists):
    """Testa o cenário onde o arquivo do banco de dados não existe.

    Args:
        mock_exists: Mock para os.path.exists retornando False.

    Returns:
        None: Verifica se a função retorna um DataFrame vazio preventivamente.
    """
    df = load_production_data()
    assert df.empty


def test_load_production_data_db_error():
    """Testa a resiliência contra erros de conexão com o banco de dados.

    Simula uma exceção ao tentar conectar ao SQLite, mesmo com o arquivo existindo.

    Returns:
        None: Verifica se a função captura a exceção e retorna um DataFrame vazio.
    """
    with patch("os.path.exists", return_value=True):
        with patch("sqlite3.connect", side_effect=Exception("Banco corrompido")):
            df = load_production_data()
            assert df.empty


def test_load_production_data_json_error(mock_db_connection):
    """Testa a robustez do parser JSON contra entradas inválidas nos logs.

    Cria um cenário misto com um JSON válido e um inválido (corrompido).

    Args:
        mock_db_connection: Fixture do mock de conexão.

    Returns:
        None: Verifica se apenas o registro válido é processado e incluído no DataFrame.
    """
    data = [
        {"input_data": '{"RA": "123", "PEDRA": "Ametista"}'},
        {"input_data": '{JSON_QUEBRADO...'},
    ]
    df_mock = pd.DataFrame(data)

    with patch("os.path.exists", return_value=True), patch(
        "pandas.read_sql_query", return_value=df_mock
    ):
        df = load_production_data()
        assert len(df) == 1
        assert df.iloc[0]["RA"] == "123"


def test_apply_feature_engineering_transform_error():
    """Verifica se a função auxiliar propaga exceções corretamente.

    Simula um erro interno no método `transform` do pipeline de feature engineering.

    Raises:
        Exception: Espera-se que a exceção original seja relançada.
    """
    mock_fe = MagicMock()
    mock_fe.transform.side_effect = Exception("Erro interno no pipeline")

    with pytest.raises(Exception):
        apply_feature_engineering_transform(pd.DataFrame(), mock_fe)


@patch("src.drift_report.load_data")
def test_generate_report_fail_load_data(mock_load_data):
    """Testa a falha na primeira etapa: carregamento dos dados de treino.

    Args:
        mock_load_data: Mock da função load_data.

    Returns:
        None: Verifica se a função retorna None ao falhar no carregamento.
    """
    mock_load_data.side_effect = Exception("Arquivo Excel corrompido")
    path = generate_report()
    assert path is None


@patch("src.drift_report.load_data")
@patch("src.drift_report.load_artifact")
def test_generate_report_fail_load_pipeline(mock_load_artifact, mock_load_data):
    """Testa a falha na segunda etapa: carregamento do pipeline (.joblib).

    Args:
        mock_load_artifact: Mock da função load_artifact.
        mock_load_data: Mock da função load_data (sucesso).

    Returns:
        None: Verifica se a função retorna None ao não encontrar o pipeline.
    """
    mock_load_data.return_value = {"2024": pd.DataFrame()}
    mock_load_artifact.side_effect = FileNotFoundError("Pipeline não encontrado")

    path = generate_report()
    assert path is None


@patch("src.drift_report.DataPreprocessor")
@patch("src.drift_report.load_data")
@patch("src.drift_report.load_artifact")
def test_generate_report_fail_transform_train(
    mock_load_artifact, mock_load_data, mock_preprocessor
):
    """Testa a falha durante a transformação dos dados de treino.

    Simula um erro durante a execução do método `transform` no pipeline carregado.

    Args:
        mock_load_artifact: Mock do artefato (pipeline).
        mock_load_data: Mock dos dados iniciais.
        mock_preprocessor: Mock da classe DataPreprocessor.

    Returns:
        None: Verifica se a função retorna None em caso de erro na transformação.
    """
    mock_load_data.return_value = {"dummy": "data"}

    instance_preprocessor = mock_preprocessor.return_value
    instance_preprocessor.run.return_value = pd.DataFrame({"RA": [1], "PEDRA": ["A"]})

    mock_pipeline = MagicMock()
    mock_pipeline.transform.side_effect = Exception("Erro na transformação do treino")
    mock_load_artifact.return_value = mock_pipeline

    path = generate_report()
    assert path is None


@patch("src.drift_report.DataPreprocessor")
@patch("src.drift_report.load_data")
@patch("src.drift_report.load_artifact")
@patch("src.drift_report.load_production_data")
def test_generate_report_fail_validation_dataset(
    mock_load_prod, mock_load_artifact, mock_load_data, mock_preprocessor
):
    """Testa a falha ao carregar o dataset de validação (Modo Validação).

    Configura o cenário onde há poucos dados de produção (dataframe vazio),
    forçando o sistema a buscar o CSV de validação local, que falha.

    Args:
        mock_load_prod: Mock que retorna dados insuficientes (vazio).
        mock_load_artifact: Mock do pipeline.
        mock_load_data: Mock dos dados de treino.
        mock_preprocessor: Mock do pré-processador.

    Returns:
        None: Verifica se a função retorna None quando o CSV de validação não é achado.
    """
    instance_preprocessor = mock_preprocessor.return_value
    instance_preprocessor.run.return_value = pd.DataFrame({"RA": [1], "PEDRA": ["A"]})

    mock_pipeline = MagicMock()
    mock_pipeline.transform.return_value = (np.array([[1]]), None)
    mock_pipeline.cols_treino = ["A"]
    mock_load_artifact.return_value = mock_pipeline

    mock_load_prod.return_value = pd.DataFrame()

    with patch("pandas.read_csv", side_effect=FileNotFoundError("CSV sumiu")):
        path = generate_report()
        assert path is None


@patch("src.drift_report.DataPreprocessor")
@patch("src.drift_report.load_data")
@patch("src.drift_report.load_artifact")
@patch("src.drift_report.load_production_data")
@patch("src.drift_report.Report")
def test_generate_report_success_validation_mode(
    mock_report,
    mock_load_prod,
    mock_load_artifact,
    mock_load_data,
    mock_preprocessor,
):
    """Testa o fluxo completo de sucesso no Modo Validação.

    Cenário onde os dados de produção são insuficientes (< 100 logs),
    levando o sistema a usar o dataset de validação estático (`test_dataset.csv`).

    Args:
        mock_report: Mock da classe Report (Evidently).
        mock_load_prod: Mock retornando DataFrame vazio.
        mock_load_artifact: Mock do pipeline.
        mock_load_data: Mock dos dados de treino.
        mock_preprocessor: Mock do pré-processador.

    Returns:
        None: Verifica se um caminho de arquivo (não None) é retornado e o report é gerado.
    """
    df_pronto = pd.DataFrame({"RA": ["1"], "PEDRA": ["Ametista"], "MAT": [10]})

    mock_load_data.return_value = {"dummy": "data"}
    mock_load_prod.return_value = pd.DataFrame()

    mock_preprocessor.return_value.run.return_value = df_pronto

    pipeline = MagicMock()
    pipeline.cols_treino = ["MAT"]
    pipeline.transform.return_value = (pd.DataFrame({"MAT": [10]}), None)
    mock_load_artifact.return_value = pipeline

    with patch("pandas.read_csv", return_value=df_pronto):
        path = generate_report()

    assert path is not None
    mock_report.return_value.run.assert_called_once()


@patch("src.drift_report.DataPreprocessor")
@patch("src.drift_report.load_data")
@patch("src.drift_report.load_artifact")
@patch("src.drift_report.load_production_data")
@patch("src.drift_report.Report")
def test_generate_report_success_production_mode(
    mock_report,
    mock_load_prod,
    mock_load_artifact,
    mock_load_data,
    mock_preprocessor,
):
    """Testa o fluxo completo de sucesso no Modo Produção.

    Cenário onde existem dados de produção suficientes (> 100 logs).
    Verifica se o pipeline usa os dados de produção e aplica a limpeza (`clean_dataframe`).

    Args:
        mock_report: Mock da classe Report.
        mock_load_prod: Mock retornando DataFrame com 150 registros.
        mock_load_artifact: Mock do pipeline.
        mock_load_data: Mock dos dados de treino.
        mock_preprocessor: Mock do pré-processador.

    Returns:
        None: Verifica se retorna um caminho válido e se `clean_dataframe` foi invocado.
    """
    df_treino = pd.DataFrame({"RA": ["1"], "PEDRA": ["A"], "MAT": [10]})
    mock_preprocessor.return_value.run.return_value = df_treino

    df_prod = pd.DataFrame(
        [{"RA": str(i), "PEDRA": "A", "MAT": 10} for i in range(150)]
    )
    mock_load_prod.return_value = df_prod

    pipeline = MagicMock()
    pipeline.cols_treino = ["MAT"]
    pipeline.transform.return_value = (pd.DataFrame({"MAT": [10] * 150}), None)
    mock_load_artifact.return_value = pipeline

    # Mock identity function para clean_dataframe
    mock_preprocessor.return_value.clean_dataframe.side_effect = lambda x: x

    path = generate_report()

    assert path is not None
    mock_preprocessor.return_value.clean_dataframe.assert_called()