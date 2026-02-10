# src/utils.py
"""Módulo de Funções Utilitárias.

Fornece funções auxiliares genéricas para operações de entrada/saída (I/O),
como leitura de arquivos Excel, persistência de objetos Python (serialização)
e implementação de regras de negócio simples utilizadas em vários pontos do projeto.
"""

import pandas as pd
import joblib
import os
from typing import Any, Dict

def load_data(file_path: str) -> Dict[str, pd.DataFrame]:
    """Carrega um arquivo Excel e retorna um dicionário de DataFrames.

    Lê todas as abas (sheets) de um arquivo Excel especificado.

    Args:
        file_path (str): O caminho absoluto ou relativo para o arquivo .xlsx.

    Returns:
        Dict[str, pd.DataFrame]: Um dicionário onde as chaves são os nomes das abas
        e os valores são os DataFrames correspondentes.

    Raises:
        FileNotFoundError: Se o arquivo especificado não existir no caminho.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
    
    return pd.read_excel(file_path, sheet_name=None)

def save_artifact(obj: Any, file_path: str) -> None:
    """Serializa e salva um objeto Python no disco utilizando joblib.

    Garante que o diretório pai do arquivo de destino exista antes de salvar.

    Args:
        obj (Any): O objeto a ser salvo (ex: modelo scikit-learn, pipeline, dict).
        file_path (str): O caminho completo onde o arquivo será salvo.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(obj, file_path)

def load_artifact(file_path: str) -> Any:
    """Carrega um objeto Python serializado do disco.

    Args:
        file_path (str): O caminho do arquivo a ser carregado.

    Returns:
        Any: O objeto Python desserializado.

    Raises:
        FileNotFoundError: Se o artefato não for encontrado no caminho especificado.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Artefato não encontrado: {file_path}")
    return joblib.load(file_path)

def calculate_risk_level(pedra_nome: str) -> str:
    """Calcula o nível de risco de defasagem com base na classificação da Pedra.

    Aplica a regra de negócio para categorizar o risco do aluno:
    - Risco Baixo: Pedras 'Topázio' e 'Ametista' (indicam bom desempenho).
    - Risco Alto: Pedras 'Ágata' e 'Quartzo' (indicam necessidade de atenção).

    Args:
        pedra_nome (str): O nome da pedra prevista pelo modelo (ex: 'Topázio').

    Returns:
        str: "Baixo" ou "Alto", representando a categoria de risco.
    """
    pedra = pedra_nome.capitalize()
    
    if pedra in ['Topázio', 'Ametista']:
        return "Baixo"
    else:
        return "Alto"