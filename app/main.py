# app/main.py
"""Ponto de entrada da API FastAPI.

Este módulo configura a aplicação FastAPI, define o gerenciamento do ciclo de vida
(inicialização e desligamento) e carrega os modelos de Machine Learning necessários
para as rotas de inferência.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
import joblib
import os

from app.routes import router
from app.monitor import init_db
from src import config
from src.feature_engineering import FeatureEngineer 

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia o ciclo de vida da aplicação FastAPI.

    Executa ações de inicialização (startup) e finalização (shutdown):
    1. Startup:
       - Inicializa o banco de dados de monitoramento/logs.
       - Tenta carregar o modelo treinado e o pipeline de engenharia de features
         do disco. Se os arquivos não existirem, define o estado como None e
         emite um aviso sugerindo o treinamento.
    
    2. Shutdown:
       - Executa procedimentos de limpeza (se houver).

    Args:
        app (FastAPI): A instância da aplicação.
    """
    # --- STARTUP ---
    print("Iniciando API...")
    
    # 1. Inicializa banco de logs
    init_db()
    print("Banco de monitoramento iniciado.")
    
    # 2. Carrega Modelos
    try:
        app.state.model = joblib.load(config.MODEL_PATH)
        app.state.pipeline = joblib.load(config.PIPELINE_PATH)
        print("Modelos carregados com sucesso!")
    except FileNotFoundError:
        print(f"AVISO: Modelos não encontrados em '{config.MODEL_PATH}'.")
        print("A API iniciou, mas você deve chamar POST /train antes de fazer predições.")
        app.state.model = None
        app.state.pipeline = None
    except Exception as e:
        print(f"ERRO DESCONHECIDO ao carregar modelos: {e}")
        
    yield
    # --- SHUTDOWN ---
    print("Desligando API...")

app = FastAPI(
    title="API Passos Mágicos - Previsão de Risco",
    description="API para estimar a 'Pedra' e o risco de defasagem escolar.",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    # Roda servidor de desenvolvimento
    uvicorn.run(app, host="0.0.0.0", port=8000)