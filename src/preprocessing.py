import pandas as pd
import numpy as np

class DataPreprocessor:
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
            'Pedra 2023': 'PEDRA', 'IPP': 'IPP'
        }
        self.mapa_2024 = {
            'RA': 'RA', 'Fase': 'FASE', 'Gênero': 'GENERO', 'Idade': 'IDADE', 
            'Ano ingresso': 'ANO_INGRESSO', 'IAA': 'IAA', 'IEG': 'IEG', 
            'IPS': 'IPS', 'IDA': 'IDA', 'IPV': 'IPV', 'IAN': 'IAN',
            'Mat': 'NOTA_MAT', 'Por': 'NOTA_PORT', 'Ing': 'NOTA_ING',
            'Fase Ideal': 'FASE_IDEAL', 'Defasagem': 'DEFASAGEM',
            'Pedra 2024': 'PEDRA', 'IPP': 'IPP'
        }

    def _clean_numeric(self, series):
        """Remove caracteres não numéricos."""
        return pd.to_numeric(series.astype(str).str.replace(r'\D', '', regex=True), errors='coerce')

    def run(self, dict_abas):
        dfs = []
        for nome_aba, df in dict_abas.items():
            df_temp = df.copy()
            mapa = None
            
            if "2022" in nome_aba: mapa = self.mapa_2022
            elif "2023" in nome_aba: mapa = self.mapa_2023
            elif "2024" in nome_aba: mapa = self.mapa_2024
            
            if mapa:
                # Garante coluna IPP (vazia em 2022)
                if 'IPP' not in df_temp.columns and "2022" in nome_aba:
                    df_temp['IPP'] = np.nan
                
                cols = [c for c in mapa.keys() if c in df_temp.columns]
                df_temp = df_temp[cols].rename(columns=mapa)
                # (Removido: Criação de ANO_DATATHON)
                dfs.append(df_temp)
        
        df_final = pd.concat(dfs, ignore_index=True)

        # --- LÓGICA DO NOTEBOOK (DATA/IDADE) ---
        df_final['IDADE'] = df_final['IDADE'].astype(str)
        mask_datas = df_final['IDADE'].str.startswith('1900-')
        if mask_datas.any():
            df_final.loc[mask_datas, 'IDADE'] = pd.to_datetime(
                df_final.loc[mask_datas, 'IDADE'], errors='coerce'
            ).dt.day
        df_final['IDADE'] = pd.to_numeric(df_final['IDADE'], errors='coerce')

        # --- CONVERSÃO DE TIPOS ---
        # Removido ANO_DATATHON da lista de conversão
        cols_numericas = [
            'RA', 'FASE', 'ANO_INGRESSO', 'IAA', 'IEG', 'IPS', 'IPV', 'IAN', 
            'IPP', 'NOTA_MAT', 'NOTA_PORT', 'NOTA_ING', 'DEFASAGEM'
        ]
        
        for col in cols_numericas:
            if col in df_final.columns:
                 if col in ['RA', 'FASE', 'ANO_INGRESSO']:
                     df_final[col] = self._clean_numeric(df_final[col])
                 else:
                     df_final[col] = pd.to_numeric(df_final[col], errors='coerce')

        # Filtra apenas pedras válidas
        pedras_validas = ['Ametista', 'Topázio', 'Quartzo', 'Ágata']
        df_final = df_final[df_final['PEDRA'].isin(pedras_validas)].copy()
        
        return df_final