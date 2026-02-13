# src/drift_report.py
"""Módulo de Drift Report - VERSÃO VALIDAÇÃO (Train vs Test).

Este módulo automatiza a detecção de desvios estatísticos (drift) entre os dados
usados no treinamento e os dados que o modelo está recebendo em produção ou teste.
"""

import os
import logging
import pandas as pd
import sqlite3
import json
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently import ColumnMapping

from src import config
from src.utils import load_data, load_artifact
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer

# Inicialização do Logger para monitoramento de execução
logger = logging.getLogger(__name__)

# Configurações de banco de dados e saída de relatórios
DB_PATH = "monitoring.db"
REPORT_PATH = os.path.join(config.BASE_DIR, "docs", "drift_report.html")

def load_production_data() -> pd.DataFrame:
    """Carrega dados de produção do banco.

    Recupera os logs de inferência armazenados no SQLite e reconstrói o 
    DataFrame a partir das strings JSON enviadas para a API.

    Returns:
        pd.DataFrame: Dados brutos coletados em produção.
    """
    if not os.path.exists(DB_PATH):
        logger.warning("Banco de dados não encontrado.")
        return pd.DataFrame()

    try:
        conn = sqlite3.connect(DB_PATH)
        query = "SELECT input_data FROM predictions"
        df_logs = pd.read_sql_query(query, conn)
        conn.close()
    except Exception as e:
        logger.error(f"Erro ao ler banco: {e}")
        return pd.DataFrame()

    if df_logs.empty:
        return pd.DataFrame()

    dados_expandidos = []
    for _, row in df_logs.iterrows():
        try:
            data_dict = json.loads(row['input_data'])
            dados_expandidos.append(data_dict)
        except json.JSONDecodeError:
            continue

    return pd.DataFrame(dados_expandidos)

def apply_feature_engineering_transform(df: pd.DataFrame, fe: FeatureEngineer) -> pd.DataFrame:
    """Aplica o FeatureEngineer e retorna DataFrame transformado.

    Args:
        df: DataFrame original a ser transformado.
        fe: Instância do FeatureEngineer carregada via artefato.

    Returns:
        pd.DataFrame: Dados processados com os nomes das colunas de treino.
    """
    try:
        X_scaled, _ = fe.transform(df)
        df_transformed = pd.DataFrame(X_scaled, columns=fe.cols_treino)
        return df_transformed
    except Exception as e:
        logger.error(f"Erro ao aplicar feature engineering: {e}")
        raise

def generate_report() -> str | None:
    """Gera relatório de drift.
    
    ESTRATÉGIA:
    1. Se houver >= 100 registros de produção → Compara Train vs Produção
    2. Se houver < 100 registros → Compara Train vs Test (validação)

    Returns:
        str | None: Caminho do relatório gerado ou None em caso de falha.
    """
    
    logger.info("=" * 80)
    logger.info("DRIFT REPORT - VERSÃO INTELIGENTE")
    logger.info("=" * 80)
    
    # 1. CARREGAR DADOS DE REFERÊNCIA (TREINO)
    logger.info("1. Carregando dados de TREINO...")
    try:
        dict_abas = load_data(str(config.RAW_DATA_PATH))
        preprocessor = DataPreprocessor()
        df_ref_raw = preprocessor.run(dict_abas)
        
        df_ref_raw = df_ref_raw.dropna(subset=['RA', 'PEDRA'])
        df_ref_raw = df_ref_raw[df_ref_raw['PEDRA'].isin(config.MAPA_PEDRA.keys())]
        logger.info(f"   Registros de treino: {len(df_ref_raw)}")
        
    except Exception as e:
        logger.error(f"Erro ao carregar treino: {e}")
        return None

    # 2. CARREGAR FEATURE ENGINEER
    logger.info("2. Carregando FeatureEngineer...")
    try:
        fe = load_artifact(config.PIPELINE_PATH)
        logger.info(f"   Pipeline carregado com sucesso")
    except FileNotFoundError:
        logger.error("   Pipeline não encontrado! Execute /train primeiro.")
        return None

    # 3. TRANSFORMAR DADOS DE TREINO
    logger.info("3. Transformando dados de TREINO...")
    try:
        df_ref_transformed = apply_feature_engineering_transform(df_ref_raw, fe)
        logger.info(f"   Shape: {df_ref_transformed.shape}")
    except Exception as e:
        logger.error(f"Erro ao transformar treino: {e}")
        return None

    # 4. DECIDIR ESTRATÉGIA: PRODUÇÃO OU TESTE?
    df_prod_raw = load_production_data()
    
    if len(df_prod_raw) >= 100:
        mode = "PRODUCTION"
        logger.info(f"4. Modo: PRODUÇÃO ({len(df_prod_raw)} registros)")
        df_current_raw = df_prod_raw
    else:
        mode = "VALIDATION"
        logger.info(f"4. Modo: VALIDAÇÃO (usando dataset de teste)")
        logger.info(f"   Produção tem apenas {len(df_prod_raw)} registros (< 100)")
        
        try:
            df_test = pd.read_csv(config.TEST_DATA_PATH)
            df_current_raw = df_test
            logger.info(f"   Dataset de teste carregado: {len(df_current_raw)} registros")
        except FileNotFoundError:
            logger.error("   Dataset de teste não encontrado! Execute /train primeiro.")
            return None

    # 5. LIMPAR E TRANSFORMAR DADOS ATUAIS
    logger.info(f"5. Processando dados {'de PRODUÇÃO' if mode == 'PRODUCTION' else 'de TESTE'}...")
    
    try:
        if mode == "PRODUCTION":
            df_current_clean = preprocessor.clean_dataframe(df_current_raw)
        else:
            df_current_clean = df_current_raw  # Teste já está limpo
        
        df_current_transformed = apply_feature_engineering_transform(df_current_clean, fe)
        logger.info(f"   Shape: {df_current_transformed.shape}")
        
        logger.info("   Comparação de médias:")
        for col in df_ref_transformed.columns[:5]:
            ref_mean = df_ref_transformed[col].mean()
            curr_mean = df_current_transformed[col].mean()
            diff = abs(ref_mean - curr_mean)
            status = "OK" if diff < 0.3 else "DRIFT"
            logger.info(f"      {col}: ref={ref_mean:.3f}, curr={curr_mean:.3f} [{status}]")
            
    except Exception as e:
        logger.error(f"Erro ao processar dados atuais: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

    # 6. GERAR RELATÓRIO
    logger.info("6. Gerando relatório Evidently...")
    
    col_mapping = ColumnMapping()
    col_mapping.numerical_features = df_ref_transformed.columns.tolist()
    col_mapping.categorical_features = []
    
    report = Report(metrics=[DataDriftPreset()])
    
    try:
        df_ref_final = df_ref_transformed.fillna(0)
        df_current_final = df_current_transformed.fillna(0)
        
        report.run(
            reference_data=df_ref_final,
            current_data=df_current_final,
            column_mapping=col_mapping
        )
        
        os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
        report.save_html(REPORT_PATH)
        
        logger.info("=" * 80)
        if mode == "PRODUCTION":
            logger.info("RELATÓRIO GERADO: Train vs Produção")
        else:
            logger.info("RELATÓRIO DE VALIDAÇÃO GERADO: Train vs Test")
            logger.info("NOTA: Para análise real, aguarde 100+ registros de produção")
        logger.info(f"Local: {REPORT_PATH}")
        logger.info("=" * 80)
        
        return REPORT_PATH
        
    except Exception as e:
        logger.error(f"Erro ao gerar relatório: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    result = generate_report()
    
    if result:
        print(f"\nRelatorio: {result}\n")
    else:
        print("\nErro ao gerar relatorio\n")