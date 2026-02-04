# src/drift_report.py
import sys
import os
import logging
import pandas as pd
import sqlite3
import json
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently import ColumnMapping

# Adiciona raiz ao path para imports funcionarem
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config
from src.utils import load_data
from src.preprocessing import DataPreprocessor

# Configura logger local
logger = logging.getLogger(__name__)

# Configurações de Caminhos
DB_PATH = "monitoring.db"
# Salva dentro da pasta docs para organização
REPORT_PATH = os.path.join(config.BASE_DIR, "docs", "drift_report.html")

def load_production_data():
    """Lê os logs do SQLite da API e transforma em DataFrame."""
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

    # Expande o JSON armazenado
    dados_expandidos = []
    for _, row in df_logs.iterrows():
        try:
            data_dict = json.loads(row['input_data'])
            # Adiciona a predição como coluna para análise
            data_dict['PEDRA_PREVISTA'] = row['predicted_pedra'] 
            dados_expandidos.append(data_dict)
        except json.JSONDecodeError:
            continue

    return pd.DataFrame(dados_expandidos)

def generate_report():
    """
    Gera o relatório de Data Drift comparando Treino vs Produção.
    Retorna: Caminho do arquivo HTML gerado ou None se falhar.
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
        # Retorna erro amigável para a API
        return "SEM_DADOS"

    # --- Interseção de Colunas ---
    colunas_comuns = [c for c in df_ref.columns if c in df_prod.columns]
    
    # Remove targets da análise de input drift se necessário
    if 'PEDRA' in colunas_comuns:
        colunas_comuns.remove('PEDRA')
        
    logger.info(f"Colunas analisadas: {colunas_comuns}")

    # --- Configuração do Evidently ---
    col_mapping = ColumnMapping()
    
    colunas_ignorar = ['RA', 'PEDRA', 'PEDRA_PREVISTA']
    col_mapping.numerical_features = [
        c for c in colunas_comuns 
        if c not in colunas_ignorar and pd.api.types.is_numeric_dtype(df_ref[c])
    ]
    
    if 'GENERO' in colunas_comuns:
        col_mapping.categorical_features = ['GENERO']

    # --- Geração ---
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