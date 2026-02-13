#tests/test_main.py
"""Testes para o ciclo de vida (lifespan) da aplicação FastAPI.

Este módulo valida o comportamento do gerenciador de contexto `lifespan`,
garantindo que os recursos (banco de dados e modelos de ML) sejam
inicializados corretamente ou tratados adequadamente em caso de falha.
"""

from unittest.mock import MagicMock, patch

from fastapi import FastAPI
import pytest

from app.main import lifespan

app_mock = FastAPI()


@pytest.mark.asyncio
async def test_lifespan_startup_success():
    """Testa o cenário de sucesso na inicialização da aplicação.

    Este teste simula o carregamento bem-sucedido do banco de dados e
    dos modelos de Machine Learning (via joblib). Verifica se o estado
    da aplicação (app.state) é preenchido corretamente com os objetos carregados.

    Verificações:
        - app.state.model deve conter o objeto retornado pelo mock.
        - app.state.pipeline deve conter o objeto retornado pelo mock.
        - init_db deve ser chamado exatamente uma vez.
        - joblib.load deve ser chamado duas vezes (modelo e pipeline).
    """
    with patch("app.main.init_db") as mock_init, patch(
        "joblib.load", return_value="MODELO_CARREGADO"
    ) as mock_load:
        async with lifespan(app_mock):
            assert app_mock.state.model == "MODELO_CARREGADO"
            assert app_mock.state.pipeline == "MODELO_CARREGADO"

            mock_init.assert_called_once()
            assert mock_load.call_count == 2


@pytest.mark.asyncio
async def test_lifespan_startup_file_not_found():
    """Testa o comportamento quando os arquivos de modelo não são encontrados.

    Simula uma exceção `FileNotFoundError` ao tentar carregar os modelos via joblib.
    O teste garante que a aplicação inicializa sem travar, mas mantém os atributos
    de estado como `None`.

    Verificações:
        - app.state.model deve ser None.
        - app.state.pipeline deve ser None.
    """
    with patch("app.main.init_db"), patch(
        "joblib.load", side_effect=FileNotFoundError("Arquivo não achado")
    ):
        async with lifespan(app_mock):
            assert app_mock.state.model is None
            assert app_mock.state.pipeline is None


@pytest.mark.asyncio
async def test_lifespan_startup_generic_error():
    """Testa a robustez contra erros genéricos durante a inicialização.

    Simula uma exceção genérica (`Exception`) durante o carregamento.
    O objetivo é garantir que o bloco `try/except` capture o erro e permita
    que a aplicação continue rodando (pass), evitando um crash fatal no startup.
    """
    with patch("app.main.init_db"), patch(
        "joblib.load", side_effect=Exception("Erro fatal")
    ):
        async with lifespan(app_mock):
            pass