import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class FeatureEngineer:
    def __init__(self):
        self.medianas = {}
        self.scaler = StandardScaler()
        
        # --- VOLTANDO PARA O MODELO COMPLETO ---
        # Inclui indicadores + Flags criadas
        self.cols_treino = [
            'IDADE', 'GENERO', 'ANO_INGRESSO', 'FASE', 'DEFASAGEM', 
            'IAA', 'IEG', 'IPS', 'IPV', 'IAN', 'IPP', 
            'NOTA_MAT', 'NOTA_PORT', 'NOTA_ING', 
            'IPP_COLETADO', 'TEM_INGLES'
        ]
        self.is_fitted = False

    def fit(self, df):
        # Calcula medianas para TODAS as colunas numéricas
        cols_para_mediana = [
            'IDADE', 'ANO_INGRESSO', 'FASE', 'DEFASAGEM',
            'IAA', 'IEG', 'IPS', 'IPV', 'IAN', 'IPP',
            'NOTA_MAT', 'NOTA_PORT', 'NOTA_ING'
        ]
        
        for col in cols_para_mediana:
            if col in df.columns:
                self.medianas[col] = df[col].median()
            else:
                self.medianas[col] = 0
            
        df_temp = self._transform_logic(df, fit_mode=True)
        self.scaler.fit(df_temp[self.cols_treino])
        
        self.is_fitted = True
        return self

    def transform(self, df):
        if not self.is_fitted:
            raise Exception("FeatureEngineer não treinado!")
            
        df_proc = self._transform_logic(df, fit_mode=False)
        X_scaled = self.scaler.transform(df_proc[self.cols_treino])
        
        y = None
        if 'PEDRA' in df_proc.columns:
            mapa_pedra = {'Quartzo': 0, 'Ágata': 1, 'Ametista': 2, 'Topázio': 3}
            y = df_proc['PEDRA'].map(mapa_pedra)
            
        return X_scaled, y

    def _transform_logic(self, df, fit_mode=False):
        df = df.copy()
        
        # Criação de Flags
        df['IPP_COLETADO'] = df['IPP'].notnull().astype(int)
        df['TEM_INGLES'] = df['NOTA_ING'].notnull().astype(int)
        
        # Lista completa de imputação
        cols_fillna = [
            'IDADE', 'ANO_INGRESSO', 'FASE', 'DEFASAGEM',
            'IAA', 'IEG', 'IPS', 'IPV', 'IAN', 'IPP',
            'NOTA_MAT', 'NOTA_PORT', 'NOTA_ING'
        ]

        for col in cols_fillna:
            if col not in df.columns: df[col] = np.nan
            
            if fit_mode:
                df[col] = df[col].fillna(df[col].median())
            else:
                val = self.medianas.get(col, 0)
                df[col] = df[col].fillna(val)

        mapa_genero = {'Feminino': 1, 'Menina': 1, 'Masculino': 0, 'Menino': 0}
        df['GENERO'] = df['GENERO'].map(mapa_genero).fillna(0)

        return df