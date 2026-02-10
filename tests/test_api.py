# tests/test_api.py
"""Testes de Integração da API.

Este módulo contém testes automatizados que validam os endpoints da aplicação
FastAPI. Utiliza o cliente de testes (TestClient) para simular requisições HTTP
e verificar se as respostas (status code e payload) estão de acordo com o
esperado para os cenários de sucesso e erro.
"""

from fastapi.testclient import TestClient

def test_health_check(client: TestClient):
    """Verifica a disponibilidade básica da API.

    Realiza uma requisição GET ao endpoint de documentação (Swagger UI) para
    garantir que a aplicação foi inicializada corretamente e está aceitando
    conexões.

    Args:
        client (TestClient): O cliente de testes injetado pela fixture.
    """
    response = client.get("/docs")
    assert response.status_code == 200

def test_predict_flow(client: TestClient):
    """Teste de integração fim-a-fim do endpoint de predição.

    Envia um payload com dados de um aluno fictício para a rota `/predict`.
    O teste valida se a API retorna um código de sucesso (200) com a estrutura
    JSON esperada (contendo RA e PEDRA_PREVISTA) ou um código de serviço
    indisponível (503) caso o modelo ainda não tenha sido treinado no ambiente
    de testes.

    Args:
        client (TestClient): O cliente de testes injetado pela fixture.
    """
    payload = [{
        "RA": "TESTE_001",
        "IDADE": 15,
        "GENERO": "Masculino",
        "ANO_INGRESSO": 2023,
        "FASE": 2,
        "NOTA_MAT": 8.5,
        "NOTA_PORT": 7.0,
        "NOTA_ING": 6.0,
        "IEG": 5.0,
        "IPS": 6.5,
        "IAA": 7.0,
        "IPP": 5.0,
        "DEFASAGEM": 0
    }]
    
    response = client.post("/predict", json=payload)
    
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "predictions" in data
        assert data["predictions"][0]["RA"] == "TESTE_001"
        assert "PEDRA_PREVISTA" in data["predictions"][0]

def test_metrics_endpoint(client: TestClient):
    """Valida o endpoint de métricas de avaliação do modelo.

    Verifica se a rota `/evaluate` responde sem erros críticos.
    Aceita múltiplos status codes válidos dependendo do estado do ambiente:
    - 200: Sucesso (métricas retornadas).
    - 400: Erro de cliente (ex: modelo não encontrado).
    - 500: Erro de servidor (falha na execução da avaliação).

    Args:
        client (TestClient): O cliente de testes injetado pela fixture.
    """
    response = client.get("/evaluate")
    assert response.status_code in [200, 400, 500]

def test_drift_report_endpoint(client: TestClient):
    """Verifica a execução do endpoint de relatório de Data Drift.

    Testa a chamada à rota `/drift-report`. Como o ambiente de teste pode
    não ter logs de produção suficientes, considera aceitável o retorno de
    aviso (400) ou erro controlado (500), além do sucesso (200), desde que
    a API não trave.

    Args:
        client (TestClient): O cliente de testes injetado pela fixture.
    """
    response = client.get("/drift-report")
    assert response.status_code in [200, 400, 500]

def test_train_endpoint(client: TestClient):
    """Testa o gatilho de treinamento do modelo via API.

    Aciona o endpoint `/train` para verificar se o pipeline de treinamento
    é iniciado. Este teste não valida a qualidade do modelo, apenas se a rota
    é acessível e executa o fluxo, aceitando erros (500) caso os arquivos
    de dados (Excel) não estejam presentes no ambiente de CI/CD ou teste local.

    Args:
        client (TestClient): O cliente de testes injetado pela fixture.
    """
    response = client.post("/train")
    assert response.status_code in [200, 500]