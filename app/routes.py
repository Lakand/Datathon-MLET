# app/routes.py
import os
import joblib
import pandas as pd
from typing import List
from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse

# Importações do projeto
from app.schemas import AlunoInput
from app.monitor import log_prediction
from src.config import MAPA_PEDRA, MODEL_PATH, PIPELINE_PATH
from src.train import train_pipeline
from src.evaluate import evaluate_model
from src.drift_report import generate_report

router = APIRouter()

# Cria o mapa reverso para traduzir números (0, 1...) em nomes (Quartzo, etc.)
REVERSE_MAPA_PEDRA = {v: k for k, v in MAPA_PEDRA.items()}

# --- ROTA 1: PREDIÇÃO (COM BACKGROUND TASKS) ---
@router.post("/predict", summary="Predição de Risco (Pedra)")
def predict(request: Request, alunos: List[AlunoInput], background_tasks: BackgroundTasks):
    """
    Recebe dados de alunos, retorna a previsão do modelo e
    grava os logs no banco de dados em segundo plano (assíncrono).
    """
    try:
        # Recupera os modelos carregados na inicialização (main.py)
        model = request.app.state.model
        pipeline = request.app.state.pipeline
        
        if not model or not pipeline:
            raise HTTPException(status_code=503, detail="Modelo não carregado. Execute /train primeiro ou verifique os logs.")

        # Converte a entrada (Pydantic) para DataFrame
        input_data = [aluno.dict() for aluno in alunos]
        df_input = pd.DataFrame(input_data)
        
        # Passa pelo Pipeline de Features (Scaler, Encoders, etc)
        X_scaled, _ = pipeline.transform(df_input)
        
        # Realiza a predição
        y_pred_idx = model.predict(X_scaled)
        
        results = []
        for i, pred_idx in enumerate(y_pred_idx):
            pedra_nome = REVERSE_MAPA_PEDRA.get(pred_idx, "Desconhecido")
            aluno_raw = input_data[i]
            
            # Regra de Negócio simples para Risco
            risco = "Baixo" if pedra_nome in ['Topázio', 'Ametista'] else "Alto"
            
            result = {
                "RA": aluno_raw.get("RA"),
                "PEDRA_PREVISTA": pedra_nome,
                "RISCO_DEFASAGEM": risco
            }
            results.append(result)
            
            # [MELHORIA PRO] Gravação de Log em Segundo Plano
            # O usuário recebe a resposta imediatamente, e o servidor grava o log depois.
            background_tasks.add_task(
                log_prediction,
                ra=str(aluno_raw.get("RA")), 
                input_data=aluno_raw, 
                prediction=pedra_nome
            )
            
        return {"predictions": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- ROTA 2: TREINAMENTO ---
@router.post("/train", summary="Executar Treinamento do Modelo")
def run_training(request: Request):
    try:
        print("Iniciando treinamento via API...")
        train_results = train_pipeline()
        
        print("Recarregando modelos em memória...")
        # Atualiza o estado da aplicação com o novo modelo treinado
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
def run_evaluation():
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
    """
    Gera um relatório HTML comparando os dados de Treino vs Produção.
    Retorna o arquivo HTML diretamente para visualização no navegador.
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
            
        # Retorna o arquivo HTML diretamente
        return FileResponse(path=report_path, media_type='text/html', filename="drift_report.html")
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")