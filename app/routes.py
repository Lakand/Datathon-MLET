# app/routes.py
from fastapi import APIRouter, Request, HTTPException
from typing import List
import pandas as pd
from app.schemas import AlunoInput
from app.monitor import log_prediction
from src.config import MAPA_PEDRA

router = APIRouter()

# Inverte o mapa para retornar o nome da Pedra (0 -> 'Quartzo')
REVERSE_MAPA_PEDRA = {v: k for k, v in MAPA_PEDRA.items()}

@router.post("/predict", summary="Predição de Risco (Pedra)")
def predict(request: Request, alunos: List[AlunoInput]):
    """
    Recebe uma lista de dados de alunos e retorna a classificação da Pedra (Risco).
    """
    try:
        model = request.app.state.model
        pipeline = request.app.state.pipeline
        
        # 1. Converter entrada (Pydantic) para DataFrame
        input_data = [aluno.dict() for aluno in alunos]
        df_input = pd.DataFrame(input_data)
        
        # 2. Aplicar o Pipeline de Features (já treinado)
        # O FeatureEngineer trata nulos e faz o scaling
        X_scaled, _ = pipeline.transform(df_input)
        
        # 3. Predição
        y_pred_idx = model.predict(X_scaled)
        
        # 4. Formatar Resposta e Logar
        results = []
        for i, pred_idx in enumerate(y_pred_idx):
            pedra_nome = REVERSE_MAPA_PEDRA.get(pred_idx, "Desconhecido")
            aluno_raw = input_data[i]
            
            # Interpretação de Risco (Opcional, mas bom para negócio)
            risco = "Baixo" if pedra_nome in ['Topázio', 'Ametista'] else "Alto"
            
            result = {
                "RA": aluno_raw.get("RA"),
                "PEDRA_PREVISTA": pedra_nome,
                "RISCO_DEFASAGEM": risco
            }
            results.append(result)
            
            # LOG (Monitoramento)
            log_prediction(
                ra=str(aluno_raw.get("RA")), 
                input_data=aluno_raw, 
                prediction=pedra_nome
            )
            
        return {"predictions": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))