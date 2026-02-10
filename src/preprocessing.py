# src/preprocessing.py
"""Módulo de Pré-processamento de Dados Brutos."""

import pandas as pd
import numpy as np
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Gerencia o pré-processamento dos datasets do Passos Mágicos."""

    def __init__(self):
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
        """Limpa um DataFrame avulso (ex: dados de produção/API)."""
        df = df.copy()

        # Tenta aplicar renomeação baseada nos mapas conhecidos
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
        """Executa o pipeline de pré-processamento nas abas do Excel."""
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
        """Aplica regras específicas de limpeza de dados."""
        # 1. Limpeza da Pedra
        if 'PEDRA' in df.columns:
            correcoes = {'Agata': 'Ágata'}
            df['PEDRA'] = df['PEDRA'].replace(correcoes)
            # NOTA: O filtro de pedras válidas foi movido para o src/train.py
            # para não deletar dados durante a inferência ou monitoramento.

        # 2. Limpeza de Idade
        if 'IDADE' in df.columns:
            df['IDADE'] = df['IDADE'].astype(str)
            mask_datas = df['IDADE'].str.startswith('1900-')
            if mask_datas.any():
                df.loc[mask_datas, 'IDADE'] = pd.to_datetime(
                    df.loc[mask_datas, 'IDADE'], errors='coerce'
                ).dt.day
            df['IDADE'] = pd.to_numeric(df['IDADE'], errors='coerce')

        # 3. Limpeza de Fases (extrair números)
        for col in ['FASE', 'FASE_IDEAL']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.extract(r'(\d+)')
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 4. Limpeza de Notas e Indicadores (Forçar Numérico)
        # Isso evita que strings como "10,5" ou "N/A" quebrem o StandardScaler mais à frente
        cols_notas = ['NOTA_MAT', 'NOTA_PORT', 'NOTA_ING', 'IEG', 'IPS', 'IAA', 'IPV', 'IAN', 'IDA']
        for col in cols_notas:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 5. Limpeza de RA
        if 'RA' in df.columns:
            df['RA'] = df['RA'].astype(str).str.replace(r'\D', '', regex=True)
            df = df[df['RA'] != '']
            
        return df