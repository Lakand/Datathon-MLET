# src/preprocessing.py
"""Módulo de Pré-processamento de Dados Brutos.

Este módulo é responsável pela consolidação e limpeza inicial dos dados
provenientes de arquivos Excel com múltiplas abas (anos). Ele padroniza
os nomes das colunas e unifica os dados em um único DataFrame.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Gerencia o pré-processamento dos datasets do Passos Mágicos.

    Responsável por identificar o esquema de colunas de cada ano (2022-2024),
    renomear para um padrão interno único e realizar limpezas básicas de tipos
    de dados.
    """

    def __init__(self):
        """Inicializa o DataPreprocessor com os mapas de colunas.

        Define dicionários que mapeiam os nomes originais das colunas em cada
        aba do Excel para os nomes padronizados do sistema.
        """
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

    def run(self, dict_abas: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Executa o pipeline de pré-processamento nas abas fornecidas.

        Itera sobre o dicionário de DataFrames, identifica o ano correspondente
        pelo nome da aba, seleciona/renomeia colunas e concatena os resultados.

        Args:
            dict_abas (Dict[str, pd.DataFrame]): Dicionário onde as chaves são os
                nomes das abas e os valores são DataFrames brutos.

        Returns:
            pd.DataFrame: DataFrame único consolidado e limpo.

        Raises:
            ValueError: Se nenhuma aba válida (contendo 2022, 2023 ou 2024) for encontrada.
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
                logger.warning(f"Aba '{nome_aba}' ignorada: Ano não identificado no nome ou fora do escopo (2022-2024).")
                continue 

            if ano:
                logger.debug(f"Processando aba: {nome_aba} (Ano: {ano})")
                df_temp = df_temp.loc[:, ~df_temp.columns.duplicated()]
                
                cols = [c for c in mapa.keys() if c in df_temp.columns]
                df_clean = df_temp[cols].rename(columns=mapa)
                df_clean['ANO_DATATHON'] = ano
                
                if 'IPP' not in df_clean.columns:
                    df_clean['IPP'] = np.nan
                    
                dfs.append(df_clean)
        
        if not dfs:
            logger.error("Nenhuma aba válida encontrada no arquivo Excel.")
            raise ValueError("Nenhuma aba válida encontrada!")
            
        df_final = pd.concat(dfs, ignore_index=True)
        
        df_final = self._clean_data(df_final)
        
        logger.info(f"Pré-processamento concluído. Shape final: {df_final.shape}")
        return df_final

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica regras específicas de limpeza de dados.

        Trata inconsistências conhecidas nas colunas 'PEDRA', 'IDADE',
        'FASE' e 'RA'.

        Args:
            df (pd.DataFrame): DataFrame consolidado bruto.

        Returns:
            pd.DataFrame: DataFrame limpo pronto para engenharia de features.
        """
        if 'PEDRA' in df.columns:
            correcoes = {'Agata': 'Ágata'}
            df['PEDRA'] = df['PEDRA'].replace(correcoes)
            pedras_validas = ['Quartzo', 'Ágata', 'Ametista', 'Topázio']
            df = df[df['PEDRA'].isin(pedras_validas)].copy()

        if 'IDADE' in df.columns:
            df['IDADE'] = df['IDADE'].astype(str)
            mask_datas = df['IDADE'].str.startswith('1900-')
            df.loc[mask_datas, 'IDADE'] = pd.to_datetime(df.loc[mask_datas, 'IDADE'], errors='coerce').dt.day
            df['IDADE'] = pd.to_numeric(df['IDADE'], errors='coerce')

        for col in ['FASE', 'FASE_IDEAL']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.extract(r'(\d+)')
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if 'RA' in df.columns:
            df['RA'] = df['RA'].astype(str).str.replace(r'\D', '', regex=True)
            df = df[df['RA'] != '']
            
        return df