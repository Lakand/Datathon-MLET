# app/schemas.py
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
from enum import Enum

# CORREÇÃO: Adicionamos as variações de 2022 (Menino/Menina)
# Agora a API aceita qualquer uma dessas 4 opções sem dar erro.
class GeneroEnum(str, Enum):
    MASCULINO = "Masculino"
    FEMININO = "Feminino"
    MENINO = "Menino"
    MENINA = "Menina"

class AlunoInput(BaseModel):
    RA: str
    IDADE: float
    GENERO: GeneroEnum 
    ANO_INGRESSO: int
    FASE: int
    NOTA_MAT: float
    NOTA_PORT: float
    NOTA_ING: float
    IEG: float
    IPS: float
    IAA: float
    IPP: float
    DEFASAGEM: int 

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "RA": "123456",
                "IDADE": 14,
                "GENERO": "Menino",  # Exemplo testando a variação
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