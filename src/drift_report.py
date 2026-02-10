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
            data_dict['PEDRA_PREVISTA'] = row['predicted_pedra'] 
            dados_expandidos.append(data_dict)
        except json.JSONDecodeError:
            continue

    return pd.DataFrame(dados_expandidos)

def generate_report() -> str | None:
    """Gera o relatório de Data Drift (Treino vs Produção).

    Executa o pipeline de comparação utilizando o Evidently:
    1. Carrega os dados de referência (dataset original de treino).
    2. Carrega os dados atuais (logs de produção).
    3. Mapeia as colunas numéricas e categóricas.
    4. Calcula métricas de drift estatístico.
    5. Exporta o resultado para um arquivo HTML.

    Returns:
        str | None: O caminho absoluto do arquivo HTML gerado em caso de sucesso,
        ou None (ou mensagem de erro) caso falhe.
    """
    logger.info("1. Carregando dados de REFERÊNCIA (Treino)...")
    try:
        dict_abas = load_data(str(config.RAW_DATA_PATH))
        preprocessor = DataPreprocessor()
        df_ref = preprocessor.run(dict_abas)
    except Exception as e:
        logger.error(f"Erro ao carregar dados de treino: {e}")
        return None

    logger.info("2. Carregando dados de PRODUÇÃO (Logs)...")
    df_prod = load_production_data()
    
    if df_prod.empty:
        logger.warning("Sem dados de produção suficientes para gerar relatório.")
        return "SEM_DADOS"

    colunas_comuns = [c for c in df_ref.columns if c in df_prod.columns]
    
    if 'PEDRA' in colunas_comuns:
        colunas_comuns.remove('PEDRA')
        
    logger.info(f"Colunas analisadas: {colunas_comuns}")

    col_mapping = ColumnMapping()
    
    colunas_ignorar = ['RA', 'PEDRA', 'PEDRA_PREVISTA']
    col_mapping.numerical_features = [
        c for c in colunas_comuns 
        if c not in colunas_ignorar and pd.api.types.is_numeric_dtype(df_ref[c])
    ]
    
    if 'GENERO' in colunas_comuns:
        col_mapping.categorical_features = ['GENERO']

    logger.info("Calculando Drift...")
    report = Report(metrics=[DataDriftPreset()])
    
    try:
        report.run(
            reference_data=df_ref[colunas_comuns], 
            current_data=df_prod[colunas_comuns],
            column_mapping=col_mapping
        )
        
        os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
        report.save_html(REPORT_PATH)
        logger.info(f"Relatório salvo em: {REPORT_PATH}")
        
        return REPORT_PATH
        
    except Exception as e:
        logger.error(f"Erro ao executar Evidently: {e}")
        return None

if __name__ == "__main__":
    generate_report()