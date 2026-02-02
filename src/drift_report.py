# src/drift_report.py
import sys
import os

# --- 1. CORREÇÃO DE CAMINHO (Path Fix) ---
# Adiciona a raiz do projeto ao Python Path para conseguir importar 'src'
# isso permite rodar "python src/drift_report.py" sem erro de ModuleNotFoundError
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import sqlite3
import json
import logging
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently import ColumnMapping

# Importações do próprio projeto
from src import config
from src.utils import load_data
from src.preprocessing import DataPreprocessor

# Configuração de Logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configurações de Arquivos
DB_PATH = "monitoring.db"
REPORT_PATH = "docs/drift_report.html"

def load_production_data():
    """Lê os logs do SQLite da API e transforma em DataFrame."""
    if not os.path.exists(DB_PATH):
        logger.warning("Banco de dados de monitoramento não encontrado. Rode a API e faça predições primeiro!")
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

    # O input_data está como string JSON, precisamos expandir para colunas
    dados_expandidos = []
    for _, row in df_logs.iterrows():
        try:
            data_dict = json.loads(row['input_data'])
            data_dict['PEDRA_PREVISTA'] = row['predicted_pedra'] # Adiciona a predição como coluna
            dados_expandidos.append(data_dict)
        except json.JSONDecodeError:
            continue

    return pd.DataFrame(dados_expandidos)

def generate_report():
    logger.info("1. Carregando dados de REFERÊNCIA (Dataset de Treino)...")
    try:
        dict_abas = load_data(str(config.RAW_DATA_PATH))
        preprocessor = DataPreprocessor()
        df_ref = preprocessor.run(dict_abas)
    except Exception as e:
        logger.error(f"Erro ao carregar dados de treino: {e}")
        return

    logger.info("2. Carregando dados de PRODUÇÃO (Logs da API)...")
    df_prod = load_production_data()
    
    if df_prod.empty:
        logger.error("Sem dados de produção (Logs) para gerar relatório. Faça chamadas na API primeiro!")
        return

    logger.info(f"Linhas carregadas -> Referência: {len(df_ref)} | Produção: {len(df_prod)}")

    # --- 3. FILTRAGEM DE COLUNAS COMUNS (A Correção Principal) ---
    # O Evidently quebraria se tentasse comparar colunas que só existem no Excel (como IAN, IPV, etc)
    # e não existem na API. Por isso, pegamos a interseção.
    colunas_comuns = [c for c in df_ref.columns if c in df_prod.columns]
    
    # Removemos a coluna alvo do treino ('PEDRA') da lista de features comuns,
    # pois na produção ela tem outro nome ('PEDRA_PREVISTA') ou não existe.
    if 'PEDRA' in colunas_comuns:
        colunas_comuns.remove('PEDRA')
        
    logger.info(f"Colunas comuns utilizadas para análise de Drift: {colunas_comuns}")

    # --- 4. CONFIGURAÇÃO DO EVIDENTLY ---
    col_mapping = ColumnMapping()
    
    # Define quais colunas são numéricas automaticamente, excluindo IDs e Targets
    colunas_ignorar = ['RA', 'PEDRA', 'PEDRA_PREVISTA']
    
    col_mapping.numerical_features = [
        c for c in colunas_comuns 
        if c not in colunas_ignorar and pd.api.types.is_numeric_dtype(df_ref[c])
    ]
    
    # Define categóricas (se existirem nas comuns)
    if 'GENERO' in colunas_comuns:
        col_mapping.categorical_features = ['GENERO']

    # --- 5. GERAÇÃO DO RELATÓRIO ---
    logger.info("Calculando Drift e gerando HTML...")
    
    report = Report(metrics=[
        DataDriftPreset(), # Analisa distribuição das features
    ])
    
    # Passamos apenas as colunas comuns para ambos os dataframes
    report.run(
        reference_data=df_ref[colunas_comuns], 
        current_data=df_prod[colunas_comuns],
        column_mapping=col_mapping
    )
    
    # Salva o arquivo
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    report.save_html(REPORT_PATH)
    
    logger.info(f"SUCESSO! Relatório salvo em: {os.path.abspath(REPORT_PATH)}")

if __name__ == "__main__":
    generate_report()