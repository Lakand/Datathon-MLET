# tests/conftest.py
"""Configurações Compartilhadas de Testes (Fixtures).

Este módulo centraliza as fixtures do Pytest reutilizáveis por todo o conjunto
de testes do projeto. Sua principal responsabilidade é fornecer uma instância
configurada do TestClient, permitindo a execução de testes de integração
nos endpoints da API sem a necessidade de subir um servidor HTTP real.
"""

import pytest
from fastapi.testclient import TestClient
from typing import Generator
from app.main import app

@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Cria e gerencia o ciclo de vida do cliente de testes da API.

    Esta fixture instancia a aplicação FastAPI envolvida pelo TestClient.
    Ela utiliza o padrão de generator (yield) para garantir que recursos
    sejam inicializados antes dos testes e limpos adequadamente após a
    execução (teardown), se necessário.

    O cliente gerado permite disparar requisições HTTP (GET, POST, etc.)
    diretamente contra a aplicação em memória.

    Yields:
        TestClient: Uma instância do cliente de testes pronta para uso.
    """
    with TestClient(app) as c:
        yield c