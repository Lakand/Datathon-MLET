import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture
def client():
    """Gera um cliente de teste que simula requisições à API."""
    with TestClient(app) as c:
        yield c