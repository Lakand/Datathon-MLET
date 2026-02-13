# src/preprocessing.py
"""Módulo de Pré-processamento de Dados Brutos.

Este módulo é responsável pela padronização de esquemas de dados provenientes 
de diferentes anos letivos, realizando a limpeza de tipos, tratamento de 
strings e unificação de nomenclaturas para consumo do pipeline de ML.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Orquestrador de limpeza e padronização de datasets históricos.

    Gerencia mapas de tradução para as colunas das safras 2022, 2023 e 2024,
    garantindo que inputs heterogêneos resultem em um DataFrame consolidado.
    """

    def __init__(self):
        """Inicializa os mapeamentos de colunas específicos por ano."""
        self.mapa_2022 = {
            'RA': 'RA', 'Fase': 'FASE', 'Gênero': 'GENERO', 'Idade 22': 'IDADE', 
            'Ano ingresso': 'ANO_INGRESSO', 'IAA': 'IAA', 'IEG': 'IEG', 
            'IPS': 'IPS', 'IDA': 'IDA', 'IPV': 'IPV', 'IAN': 'IAN',
            'Matem': 'NOTA_MAT', 'Portug': 'NOTA_PORT', 'Inglês': 'NOTA_ING',
            'Fase ideal': 'FASE_IDEAL', 'Defas': 'DEFASAGEM', 
            'Pedra 22': 'PEDRA', 'IPP': 'IPP'
        }
        self.mapa_2023 = {
            'RA': 'RA', 'Fase': 'FASE', 'Gênero': 'GENERO', 'Idade': 'IDADE', 
            'Ano ingresso': 'ANO_INGRESSO', 'IAA': 'IAA', 'IEG': 'IEG', 
            'IPS': 'IPS', 'IDA': 'IDA', 'IPV': 'IPV', 'IAN': 'IAN', 
            'Mat': 'NOTA_MAT', 'Por': 'NOTA_PORT', 'Ing': 'NOTA_ING',
            'Fase Ideal': 'FASE_IDEAL', 'Defasagem': 'DEFASAGEM', 
            'IPP': 'IPP', 'Pedra 2023': 'PEDRA'
        }
        self.mapa_2024 = {
            'RA': 'RA', 'Fase': 'FASE', 'Gênero': 'GENERO', 'Idade': 'IDADE', 
            'Ano ingresso': 'ANO_INGRESSO', 'IAA': 'IAA', 'IEG': 'IEG', 
            'IPS': 'IPS', 'IDA': 'IDA', 'IPV': 'IPV', 'IAN': 'IAN', 
            'Mat': 'NOTA_MAT', 'Por': 'NOTA_PORT', 'Ing': 'NOTA_ING',
            'Fase Ideal': 'FASE_IDEAL', 'Defasagem': 'DEFASAGEM', 
            'IPP': 'IPP', 'Pedra 2024': 'PEDRA'
        }

    def clean_dataframe(self, df: pd.DataFrame, ano_default: int = 2024) -> pd.DataFrame:
        """Processa um DataFrame avulso, comum em fluxos de produção ou API.

        Args:
            df: DataFrame bruto recebido via request ou carga externa.
            ano_default: Ano de referência para preenchimento de metadados.

        Returns:
            pd.DataFrame: Dados limpos e normalizados.
        """
        df = df.copy()

        # Aplicação iterativa de mapas para identificação automática do esquema
        for mapa in [self.mapa_2024, self.mapa_2023, self.mapa_2022]:
            cols_to_rename = {k: v for k, v in mapa.items() if k in df.columns}
            if cols_to_rename:
                df = df.rename(columns=cols_to_rename)

        if 'ANO_DATATHON' not in df.columns:
            df['ANO_DATATHON'] = ano_default

        if 'IPP' not in df.columns:
            df['IPP'] = np.nan

        return self._clean_data(df)

    def run(self, dict_abas: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Consolida múltiplas abas de um arquivo Excel em um único dataset.

        Args:
            dict_abas: Dicionário onde chaves são nomes das abas e valores são DataFrames.

        Returns:
            pd.DataFrame: União de todas as safras processadas.

        Raises:
            ValueError: Se nenhuma aba válida (2022-2024) for detectada no input.
        """
        logger.info("Iniciando pré-processamento das abas...")
        dfs = []
        for nome_aba, df in dict_abas.items():
            df_temp = df.copy()
            ano = None
            mapa = None
            
            if "2022" in nome_aba: 
                ano = 2022
                mapa = self.mapa_2022
            elif "2023" in nome_aba: 
                ano = 2023
                mapa = self.mapa_2023
            elif "2024" in nome_aba: 
                ano = 2024
                mapa = self.mapa_2024
            else:
                logger.warning(f"Aba '{nome_aba}' ignorada.")
                continue 

            if ano:
                # Remoção de duplicidade estrutural e filtragem de colunas mapeadas
                df_temp = df_temp.loc[:, ~df_temp.columns.duplicated()]
                cols = [c for c in mapa.keys() if c in df_temp.columns]
                df_clean = df_temp[cols].rename(columns=mapa)
                df_clean['ANO_DATATHON'] = ano
                
                if 'IPP' not in df_clean.columns:
                    df_clean['IPP'] = np.nan
                    
                dfs.append(df_clean)
        
        if not dfs:
            raise ValueError("Nenhuma aba válida encontrada!")
            
        df_final = pd.concat(dfs, ignore_index=True)
        return self._clean_data(df_final)

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Executa a sanitização fina e conversão de tipos de dados.

        Args:
            df: DataFrame com colunas já renomeadas.

        Returns:
            pd.DataFrame: DataFrame com tipos numéricos e categóricos validados.
        """
        # Normalização ortográfica de variáveis categóricas
        if 'PEDRA' in df.columns:
            correcoes = {'Agata': 'Ágata'}
            df['PEDRA'] = df['PEDRA'].replace(correcoes)

        # Conversão de Idade: trata casos onde datas de 1900 foram importadas indevidamente
        if 'IDADE' in df.columns:
            df['IDADE'] = df['IDADE'].astype(str)
            mask_datas = df['IDADE'].str.startswith('1900-')
            if mask_datas.any():
                df.loc[mask_datas, 'IDADE'] = pd.to_datetime(
                    df.loc[mask_datas, 'IDADE'], errors='coerce'
                ).dt.day
            df['IDADE'] = pd.to_numeric(df['IDADE'], errors='coerce')

        # Extração numérica de identificadores de fase
        for col in ['FASE', 'FASE_IDEAL']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.extract(r'(\d+)')
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Codificação binária de gênero para compatibilidade com modelos numéricos
        if 'GENERO' in df.columns:
            mapa_genero = {
                'Feminino': 1, 'Menina': 1, 
                'Masculino': 0, 'Menino': 0
            }
            
            if df['GENERO'].dtype == 'object':
                df['GENERO'] = df['GENERO'].map(mapa_genero)
            
            df['GENERO'] = pd.to_numeric(df['GENERO'], errors='coerce')
            df['GENERO'] = df['GENERO'].fillna(0)

        # Coerção numérica para notas e indicadores de performance
        cols_notas = ['NOTA_MAT', 'NOTA_PORT', 'NOTA_ING', 'IEG', 'IPS', 'IAA', 'IPV', 'IAN', 'IDA']
        for col in cols_notas:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Sanitização do Registro Acadêmico (RA)
        if 'RA' in df.columns:
            df['RA'] = df['RA'].astype(str).str.replace(r'\D', '', regex=True)
            df = df[df['RA'] != '']
            
        return df