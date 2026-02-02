# app/schemas.py
from pydantic import BaseModel, Field
from typing import Optional

class AlunoInput(BaseModel):
    # Campos obrigatórios baseados no seu FeatureEngineer
    RA: Optional[str] = Field(None, description="Registro do Aluno (usado para log)")
    IDADE: float = Field(..., description="Idade do aluno")
    GENERO: str = Field(..., description="Gênero (Masculino, Feminino, etc)")
    ANO_INGRESSO: int = Field(..., description="Ano de ingresso na associação")
    FASE: int = Field(..., description="Fase atual do aluno")
    DEFASAGEM: int = Field(..., description="Nível de defasagem (0, 1, etc)")
    
    # Notas (Podem vir nulas, o FeatureEngineer trata com a mediana, mas é bom pedir)
    NOTA_MAT: Optional[float] = 0.0
    NOTA_PORT: Optional[float] = 0.0
    NOTA_ING: Optional[float] = 0.0
    
    # Indicadores Psicossociais
    IEG: Optional[float] = 0.0
    IPS: Optional[float] = 0.0
    IAA: Optional[float] = 0.0
    IPP: Optional[float] = 0.0

    class Config:
        schema_extra = {
            "example": {
                "RA": "12345",
                "IDADE": 12,
                "GENERO": "Feminino",
                "ANO_INGRESSO": 2022,
                "FASE": 2,
                "DEFASAGEM": 0,
                "NOTA_MAT": 7.5,
                "NOTA_PORT": 8.0,
                "NOTA_ING": 6.5,
                "IEG": 6.8,
                "IPS": 7.5,
                "IAA": 8.2,
                "IPP": 7.0
            }
        }