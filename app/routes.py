# app/routes.py
import os
from fastapi import APIRouter, Request, HTTPException
# Removido: from fastapi.responses import FileResponse (Não precisamos mais enviar o arquivo)
from typing import List
import pandas as pd
import joblib

from app.schemas import AlunoInput
from app.monitor import log_prediction
from src.config import MAPA_PEDRA, MODEL_PATH, PIPELINE_PATH
from src.train import train_pipeline
from src.evaluate import evaluate_model
from src.drift_report import generate_report

router = APIRouter()

REVERSE_MAPA_PEDRA = {v: k for k, v in MAPA_PEDRA.items()}

# --- ROTA 1: PREDIÇÃO ---
@router.post("/predict", summary="Predição de Risco (Pedra)")
def predict(request: Request, alunos: List[AlunoInput]):
    try:
        model = request.app.state.model
        pipeline = request.app.state.pipeline
        
        if not model or not pipeline:
            raise HTTPException(status_code=503, detail="Modelo não carregado. Execute /train primeiro.")

        input_data = [aluno.dict() for aluno in alunos]
        df_input = pd.DataFrame(input_data)
        
        X_scaled, _ = pipeline.transform(df_input)
        y_pred_idx = model.predict(X_scaled)
        
        results = []
        for i, pred_idx in enumerate(y_pred_idx):
            pedra_nome = REVERSE_MAPA_PEDRA.get(pred_idx, "Desconhecido")
            aluno_raw = input_data[i]
            risco = "Baixo" if pedra_nome in ['Topázio', 'Ametista'] else "Alto"
            
            result = {
                "RA": aluno_raw.get("RA"),
                "PEDRA_PREVISTA": pedra_nome,
                "RISCO_DEFASAGEM": risco
            }
            results.append(result)
            
            log_prediction(
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
        
        print("Recarregando modelos...")
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

# --- ROTA 4: MONITORAMENTO DE DRIFT (Modificada) ---
@router.get("/drift-report", summary="Gerar Relatório de Drift (Salvar Localmente)")
def get_drift_report():
    """
    Gera um relatório HTML comparando os dados de Treino vs Produção.
    O arquivo será salvo na pasta 'docs/' do projeto.
    """
    try:
        # A função generate_report JÁ SALVA o arquivo no disco.
        # Nós só precisamos pegar o caminho que ela retorna.
        report_path = generate_report()
        
        if report_path == "SEM_DADOS":
            raise HTTPException(
                status_code=400, 
                detail="Não há dados de predição suficientes (Logs) para gerar o relatório. Use a rota /predict primeiro."
            )
            
        if not report_path or not os.path.exists(report_path):
            raise HTTPException(status_code=500, detail="Falha interna ao gerar o arquivo.")
            
        # Retorna apenas a mensagem de sucesso e o caminho
        return {
            "message": "Relatório gerado com sucesso!",
            "file_location": report_path,
            "instruction": "Vá até a pasta do projeto e abra este arquivo html manualmente no navegador."
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")