# tests/test_main.py
import pytest
from unittest.mock import patch, MagicMock
from fastapi import FastAPI
from app.main import lifespan

# Cria uma app fake para passar para o lifespan
app_mock = FastAPI()

@pytest.mark.asyncio
async def test_lifespan_startup_success():
    """Cenário 1: Sucesso - Carrega modelos e inicia DB."""
    
    # Mockamos o init_db para não criar arquivo real
    # Mockamos o joblib para retornar um objeto dummy
    with patch("app.main.init_db") as mock_init, \
         patch("joblib.load", return_value="MODELO_CARREGADO") as mock_load:
        
        # Entra no contexto (simula o startup)
        async with lifespan(app_mock):
            # Verifica se definiu o estado corretamente
            assert app_mock.state.model == "MODELO_CARREGADO"
            assert app_mock.state.pipeline == "MODELO_CARREGADO"
            
            # Verifica se chamou as funções esperadas
            mock_init.assert_called_once()
            assert mock_load.call_count == 2 # 1 pro model, 1 pro pipeline

@pytest.mark.asyncio
async def test_lifespan_startup_file_not_found():
    """Cenário 2: Erro - Arquivos de modelo não existem."""
    
    with patch("app.main.init_db"), \
         patch("joblib.load", side_effect=FileNotFoundError("Arquivo não achado")):
        
        async with lifespan(app_mock):
            # O app deve iniciar, mas com estado None
            assert app_mock.state.model is None
            assert app_mock.state.pipeline is None

@pytest.mark.asyncio
async def test_lifespan_startup_generic_error():
    """Cenário 3: Erro Genérico - Falha desconhecida no joblib."""
    
    with patch("app.main.init_db"), \
         patch("joblib.load", side_effect=Exception("Erro fatal")):
        
        async with lifespan(app_mock):
            # Deve capturar a exceção e não quebrar o teste
            # O estado pode não ter sido definido ou ser o anterior, 
            # mas o importante é passar pelo bloco 'except Exception'
            pass