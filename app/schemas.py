# app/schemas.py
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional

class AlunoInput(BaseModel):
    RA: str
    IDADE: float
    GENERO: str  # "Masculino" ou "Feminino"
    ANO_INGRESSO: int
    FASE: int
    NOTA_MAT: float
    NOTA_PORT: float
    NOTA_ING: float
    IEG: float
    IPS: float
    IAA: float
    IPP: float
    DEFASAGEM: int # 0 ou 1 (se tiver essa info prévia, senão trate como opcional)

    # --- CORREÇÃO AQUI (Pydantic V2) ---
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "RA": "123456",
                "IDADE": 14,
                "GENERO": "Masculino",
                "ANO_INGRESSO": 2022,
                "FASE": 1,
                "NOTA_MAT": 8.5,
                "NOTA_PORT": 7.0,
                "NOTA_ING": 6.5,
                "IEG": 7.0,
                "IPS": 6.5,
                "IAA": 8.0,
                "IPP": 7.5,
                "DEFASAGEM": 0
            }
        }
    )