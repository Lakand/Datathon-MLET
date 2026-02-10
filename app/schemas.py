# app/schemas.py
"""Esquemas de Dados Pydantic (Data Transfer Objects).

Este módulo define os modelos de dados utilizados para validação e serialização
das entradas e saídas da API. Garante que os dados recebidos nos endpoints
estejam no formato correto antes de serem processados pelo modelo de ML.
"""

from pydantic import BaseModel, ConfigDict
from enum import Enum

class GeneroEnum(str, Enum):
    """Enumeração para padronização dos valores de Gênero.

    Aceita variações encontradas nos diferentes anos da base de dados (2022-2024),
    como 'Masculino'/'Feminino' e 'Menino'/'Menina', garantindo que a API não
    falhe por inconsistências de nomenclatura.
    """
    MASCULINO = "Masculino"
    FEMININO = "Feminino"
    MENINO = "Menino"
    MENINA = "Menina"

class AlunoInput(BaseModel):
    """Modelo de entrada para os dados de um aluno.

    Define os campos obrigatórios e seus tipos para a predição do risco.
    Utiliza o Pydantic para validação automática dos tipos de dados.

    Attributes:
        RA (str): Registro Acadêmico do aluno (identificador único).
        IDADE (float): Idade do aluno.
        GENERO (GeneroEnum): Gênero (Masculino/Feminino/Menino/Menina).
        ANO_INGRESSO (int): Ano em que o aluno ingressou na instituição.
        FASE (int): Fase atual do aluno no programa.
        NOTA_MAT (float): Nota em Matemática.
        NOTA_PORT (float): Nota em Português.
        NOTA_ING (float): Nota em Inglês.
        IEG (float): Índice de Engajamento Global.
        IPS (float): Índice de Psicossocial.
        IAA (float): Índice de Autoavaliação.
        IPP (float): Índice Psicopedagógico.
        DEFASAGEM (int): Nível de defasagem escolar (0, 1, 2...).
    """
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
                "GENERO": "Menino",
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