# app/routes.py
"""Módulo de Rotas da API (Endpoints).

Define os endpoints da aplicação FastAPI, orquestrando as chamadas para
os serviços de predição, treinamento, avaliação e monitoramento de drift.
Gerencia a interação HTTP e delega a lógica de negócios para os módulos
especializados.
"""

import os
import joblib
import pandas as pd
import time
import numpy as np
from typing import List
from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse

from app.schemas import AlunoInput
from app.monitor import log_prediction
from src.config import MAPA_PEDRA, MODEL_PATH, PIPELINE_PATH
from src.preprocessing import DataPreprocessor
from src.train import train_pipeline
from src.evaluate import evaluate_model
from src.drift_report import generate_report
from src.utils import calculate_risk_level 

router = APIRouter()

REVERSE_MAPA_PEDRA = {v: k for k, v in MAPA_PEDRA.items()}

# --- ROTA 1: PREDIÇÃO (COM BACKGROUND TASKS) ---
@router.post("/predict", summary="Predição de Risco (Pedra)")
def predict(request: Request, alunos: List[AlunoInput], background_tasks: BackgroundTasks) -> dict:
    """Realiza a inferência de risco para uma lista de alunos.

    Recebe os dados cadastrais e acadêmicos, processa através do pipeline
    de machine learning e retorna a 'Pedra' prevista e o nível de risco.
    
    Adicionalmente, agenda uma tarefa em segundo plano (Background Task)
    para salvar os logs da requisição no banco de dados, permitindo
    monitoramento assíncrono sem impactar a latência da resposta.

    Args:
        request (Request): Objeto da requisição (usado para acessar o estado da app).
        alunos (List[AlunoInput]): Lista de objetos com os dados dos alunos (validado pelo Pydantic).
        background_tasks (BackgroundTasks): Gerenciador de tarefas assíncronas do FastAPI.

    Returns:
        dict: Dicionário contendo a lista de resultados com RA, Pedra Prevista e Risco.

    Raises:
        HTTPException(503): Se o modelo ainda não tiver sido treinado/carregado.
        HTTPException(500): Erro interno durante o processamento.
    """
    start_time = time.time()
    try:
        model = request.app.state.model
        pipeline = request.app.state.pipeline
        
        if not model or not pipeline:
            raise HTTPException(status_code=503, detail="Modelo não carregado. Execute /train primeiro ou verifique os logs.")

        input_data = [aluno.dict() for aluno in alunos]
        df_input = pd.DataFrame(input_data)

        preprocessor = DataPreprocessor()
        df_input = preprocessor.clean_dataframe(df_input)

        X_scaled, _ = pipeline.transform(df_input)
        
        y_probs = model.predict_proba(X_scaled)
        
        y_pred_idx = np.argmax(y_probs, axis=1)
        
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        results = []
        for i, pred_idx in enumerate(y_pred_idx):
            pedra_nome = REVERSE_MAPA_PEDRA.get(pred_idx, "Desconhecido")
            aluno_raw = input_data[i]
            
            confidence = float(np.max(y_probs[i]))

            risco = calculate_risk_level(pedra_nome)
            
            result = {
                "RA": aluno_raw.get("RA"),
                "PEDRA_PREVISTA": pedra_nome,
                "RISCO_DEFASAGEM": risco,
                "CONFIANCA": confidence
            }
            results.append(result)
            
            background_tasks.add_task(
                log_prediction,
                ra=str(aluno_raw.get("RA")), 
                input_data=aluno_raw, 
                prediction=pedra_nome,
                confidence=confidence,
                execution_time=latency_ms
            )
            
        return {"predictions": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- ROTA 2: TREINAMENTO ---
@router.post("/train", summary="Executar Treinamento do Modelo")
def run_training(request: Request) -> dict:
    """Dispara o pipeline de retreinamento do modelo completo.

    Executa o processo de carga de dados, engenharia de features, treinamento
    e validação. Após o sucesso, recarrega o modelo na memória da aplicação
    para que as próximas predições usem a nova versão imediatamente.

    Args:
        request (Request): Objeto da requisição para atualizar o estado global (app.state).

    Returns:
        dict: Mensagem de sucesso e resumo das métricas de treinamento.

    Raises:
        HTTPException(500): Se ocorrer erro durante o treinamento.
    """
    try:
        print("Iniciando treinamento via API...")
        train_results = train_pipeline()
        
        print("Recarregando modelos em memória...")
        request.app.state.model = joblib.load(MODEL_PATH)
        request.app.state.pipeline = joblib.load(PIPELINE_PATH)
        
        return {
            "message": "Treinamento concluído e modelo atualizado!",
            "details": train_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no treinamento: {str(e)}")

# --- ROTA 3: AVALIAÇÃO ---
@router.get("/evaluate", summary="Avaliar Performance do Modelo")
def run_evaluation() -> dict:
    """Executa a avaliação do modelo atual contra a base de teste (holdout).

    Returns:
        dict: Métricas de classificação (precision, recall, f1-score) e matriz de confusão.

    Raises:
        HTTPException(400): Se houver erro de validação (ex: arquivos não encontrados).
        HTTPException(500): Erro interno inesperado.
    """
    try:
        metrics = evaluate_model()
        if "error" in metrics:
            raise HTTPException(status_code=400, detail=metrics["error"])
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- ROTA 4: MONITORAMENTO DE DRIFT ---
@router.get("/drift-report", summary="Gerar Relatório de Drift (HTML)")
def get_drift_report():
    """Gera e retorna um relatório visual de Data Drift.

    Compara a distribuição estatística dos dados de treinamento com os dados
    recebidos em produção (armazenados nos logs).

    Returns:
        FileResponse: O arquivo HTML do relatório renderizado.

    Raises:
        HTTPException(400): Se não houver dados suficientes em produção.
        HTTPException(500): Falha na geração do arquivo.
    """
    try:
        report_path = generate_report()
        
        if report_path == "SEM_DADOS":
            raise HTTPException(
                status_code=400, 
                detail="Não há dados de predição suficientes (Logs) para gerar o relatório. Use a rota /predict primeiro."
            )
            
        if not report_path or not os.path.exists(report_path):
            raise HTTPException(status_code=500, detail="Falha interna ao gerar o arquivo de relatório.")
            
        return FileResponse(path=report_path, media_type='text/html', filename="drift_report.html")
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")