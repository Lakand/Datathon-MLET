# src/drift_report.py
"""Módulo de Monitoramento e Relatório de Data Drift.

Este módulo é responsável por comparar os dados utilizados no treinamento
(referência) com os dados recebidos em produção (armazenados no banco de dados),
gerando um relatório HTML detalhado sobre o desvio (drift) das distribuições
utilizando a biblioteca Evidently.
"""

import os
import logging
import pandas as pd
import sqlite3
import json
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently import ColumnMapping

from src import config
from src.utils import load_data
from src.preprocessing import DataPreprocessor

logger = logging.getLogger(__name__)

DB_PATH = "monitoring.db"
REPORT_PATH = os.path.join(config.BASE_DIR, "docs", "drift_report.html")

def load_production_data() -> pd.DataFrame:
    """Extrai e estrutura os logs de produção do banco de dados SQLite.

    Lê a tabela de predições, decodifica o payload JSON de entrada e
    consolida as informações em um DataFrame para análise.

    Returns:
        pd.DataFrame: Um DataFrame contendo as features de entrada e a
        classe predita para cada requisição feita à API. Retorna um
        DataFrame vazio em caso de erro ou base vazia.
    """
    if not os.path.exists(DB_PATH):
        logger.warning("Banco de dados de monitoramento não encontrado.")
        return pd.DataFrame()

    try:
        conn = sqlite3.connect(DB_PATH)
        query = "SELECT input_data, predicted_pedra FROM predictions"
        df_logs = pd.read_sql_query(query, conn)
        conn.close()
    except Exception as e:
        logger.error(f"Erro ao ler banco de dados: {e}")
        return pd.DataFrame()

    if df_logs.empty:
        return pd.DataFrame()

    dados_expandidos = []
    for _, row in df_logs.iterrows():
        try:
            data_dict = json.loads(row['input_data'])
            # Adiciona a predição como se fosse a coluna alvo para monitoramento
            data_dict['PEDRA_PREVISTA'] = row['predicted_pedra'] 
            dados_expandidos.append(data_dict)
        except json.JSONDecodeError:
            continue

    return pd.DataFrame(dados_expandidos)

def generate_report() -> str | None:
    """Gera o relatório de Data Drift (Treino vs Produção)."""
    
    # 1. Carregando Referência
    logger.info("1. Carregando dados de REFERÊNCIA (Treino)...")
    try:
        dict_abas = load_data(str(config.RAW_DATA_PATH))
        preprocessor = DataPreprocessor()
        
        # Limpa os dados de treino
        df_ref = preprocessor.run(dict_abas)
        
        # === FIX: Criar coluna de predição na Referência ===
        # O Evidently precisa que a coluna de 'prediction' exista em AMBOS os datasets.
        # No treino, a nossa "predição perfeita" (ground truth) é a coluna PEDRA.
        # Então duplicamos ela com o nome esperado.
        if 'PEDRA' in df_ref.columns:
            df_ref['PEDRA_PREVISTA'] = df_ref['PEDRA']
        else:
            logger.error("Coluna PEDRA não encontrada nos dados de referência.")
            return None
        # ===================================================
        
    except Exception as e:
        logger.error(f"Erro ao carregar dados de treino: {e}")
        return None

    # 2. Carregando Produção
    logger.info("2. Carregando dados de PRODUÇÃO (Logs)...")
    df_prod_raw = load_production_data()
    
    if df_prod_raw.empty:
        logger.warning("Sem dados de produção suficientes para gerar relatório.")
        return "SEM_DADOS"

    # Pré-processamento dos logs
    logger.info("Pré-processando dados de produção para compatibilidade...")
    try:
        df_prod = preprocessor.clean_dataframe(df_prod_raw)
    except Exception as e:
        logger.error(f"Erro ao limpar dados de produção: {e}")
        return None

    # Define colunas comuns para análise
    # (Excluímos as colunas de target/predição da lista de features comuns para não duplicar no mapping)
    colunas_comuns = [c for c in df_ref.columns if c in df_prod.columns]
    colunas_ignorar = ['RA', 'PEDRA', 'PEDRA_PREVISTA', 'ANO_DATATHON']
    
    # Remove as colunas ignoradas da lista de colunas comuns
    colunas_comuns = [c for c in colunas_comuns if c not in colunas_ignorar]
        
    logger.info(f"Colunas analisadas para Drift: {colunas_comuns}")

    # Configuração do Evidently
    col_mapping = ColumnMapping()
    
    # Define numéricas automaticamente
    col_mapping.numerical_features = [
        c for c in colunas_comuns 
        if pd.api.types.is_numeric_dtype(df_ref[c])
    ]
    
    # Define categóricas
    if 'GENERO' in colunas_comuns:
        col_mapping.categorical_features = ['GENERO']

    # Mapeamento da Predição
    # Agora ambos os DataFrames possuem 'PEDRA_PREVISTA'
    if 'PEDRA_PREVISTA' in df_ref.columns and 'PEDRA_PREVISTA' in df_prod.columns:
        col_mapping.prediction = 'PEDRA_PREVISTA'

    logger.info("Calculando Drift...")
    report = Report(metrics=[DataDriftPreset()])
    
    try:
        # Seleciona apenas colunas relevantes para passar ao Evidently
        cols_ref = colunas_comuns + ['PEDRA_PREVISTA']
        cols_prod = colunas_comuns + ['PEDRA_PREVISTA']
        
        report.run(
            reference_data=df_ref[cols_ref], 
            current_data=df_prod[cols_prod],
            column_mapping=col_mapping
        )
        
        os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
        report.save_html(REPORT_PATH)
        logger.info(f"Relatório salvo com sucesso em: {REPORT_PATH}")
        
        return REPORT_PATH
        
    except Exception as e:
        logger.error(f"Erro ao executar Evidently: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    generate_report()