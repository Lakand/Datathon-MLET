# src/preprocessing.py
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional

# Configuração de Log local
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        # Mapas de colunas originais
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
        logger.info("Iniciando pré-processamento das abas...")
        dfs = []
        for nome_aba, df in dict_abas.items():
            df_temp = df.copy()
            
            # Identifica o ano e mapa correto
            ano = None
            mapa = None
            if "2022" in nome_aba: ano = 2022; mapa = self.mapa_2022
            elif "2023" in nome_aba: ano = 2023; mapa = self.mapa_2023
            elif "2024" in nome_aba: ano = 2024; mapa = self.mapa_2024
            
            if ano:
                logger.debug(f"Processando aba: {nome_aba} (Ano: {ano})")
                # Remove colunas duplicadas
                df_temp = df_temp.loc[:, ~df_temp.columns.duplicated()]
                
                # Seleciona e renomeia
                cols = [c for c in mapa.keys() if c in df_temp.columns]
                df_clean = df_temp[cols].rename(columns=mapa)
                df_clean['ANO_DATATHON'] = ano
                
                # Cria IPP nulo se não existir (caso 2022)
                if 'IPP' not in df_clean.columns:
                    df_clean['IPP'] = np.nan
                    
                dfs.append(df_clean)
        
        if not dfs:
            logger.error("Nenhuma aba válida encontrada no arquivo Excel.")
            raise ValueError("Nenhuma aba válida encontrada!")
            
        df_final = pd.concat(dfs, ignore_index=True)
        
        # --- APLICAÇÃO DAS LIMPEZAS ---
        df_final = self._clean_data(df_final)
        
        logger.info(f"Pré-processamento concluído. Shape final: {df_final.shape}")
        return df_final

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # 1. Padronizar Pedra
        correcoes = {'Agata': 'Ágata'}
        if 'PEDRA' in df.columns:
            df['PEDRA'] = df['PEDRA'].replace(correcoes)
            pedras_validas = ['Quartzo', 'Ágata', 'Ametista', 'Topázio']
            df = df[df['PEDRA'].isin(pedras_validas)].copy()

        # 2. Limpeza de IDADE
        if 'IDADE' in df.columns:
            df['IDADE'] = df['IDADE'].astype(str)
            mask_datas = df['IDADE'].str.startswith('1900-')
            df.loc[mask_datas, 'IDADE'] = pd.to_datetime(df.loc[mask_datas, 'IDADE'], errors='coerce').dt.day
            df['IDADE'] = pd.to_numeric(df['IDADE'], errors='coerce')

        # 3. Limpeza de FASE e FASE_IDEAL
        for col in ['FASE', 'FASE_IDEAL']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.extract(r'(\d+)')
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 4. Limpeza de RA
        if 'RA' in df.columns:
            df['RA'] = df['RA'].astype(str).str.replace(r'\D', '', regex=True)
            df = df[df['RA'] != '']
            
        return df