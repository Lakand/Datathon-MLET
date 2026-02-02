# app/main.py
import sys
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
import joblib

# Adiciona a raiz ao path para conseguir importar 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.routes import router
from app.monitor import init_db
from src import config
# Importante: Importar a classe FeatureEngineer para o joblib não falhar
from src.feature_engineering import FeatureEngineer 

@asynccontextmanager
async def lifespan(app: FastAPI):
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
    except Exception as e:
        print(f"ERRO CRÍTICO ao carregar modelos: {e}")
        # Em produção, talvez fosse melhor falhar o startup aqui
        
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