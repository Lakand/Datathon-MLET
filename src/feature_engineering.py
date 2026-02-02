# src/feature_engineering.py
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional, Tuple
from src import config # Importa as configurações

logger = logging.getLogger(__name__)

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.medianas = {}
        self.scaler = StandardScaler()
        
        # Estratégia "Meio Termo"
        self.cols_treino = [
            'IDADE', 'GENERO', 'ANO_INGRESSO', 'FASE', 'DEFASAGEM',
            'NOTA_MAT', 'NOTA_PORT', 'NOTA_ING',
            'IEG', 'IPS', 'IAA', 'IPP'
        ]
        self.is_fitted = False

    def fit(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureEngineer':
        logger.info("Ajustando (Fitting) FeatureEngineer...")
        cols_numericas = df.select_dtypes(include=[np.number]).columns
        
        for col in cols_numericas:
            self.medianas[col] = df[col].median()
            
        df_temp = self._transform_logic(df, fit_mode=True)
        self.scaler.fit(df_temp[self.cols_treino])
        
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[pd.Series]]:
        if not self.is_fitted:
            logger.error("Tentativa de transformação sem fit prévio.")
            raise RuntimeError("FeatureEngineer não treinado!")
            
        df_proc = self._transform_logic(df, fit_mode=False)
        X_scaled = self.scaler.transform(df_proc[self.cols_treino])
        
        y = None
        if 'PEDRA' in df_proc.columns:
            # Usa o mapa centralizado no config
            y = df_proc['PEDRA'].map(config.MAPA_PEDRA)
            
        return X_scaled, y

    def _transform_logic(self, df: pd.DataFrame, fit_mode: bool = False) -> pd.DataFrame:
        df = df.copy()
        
        # Preenchimento de Nulos
        for col in self.cols_treino:
            if col == 'GENERO': continue
            
            if col not in df.columns:
                df[col] = np.nan
            
            val = self.medianas.get(col, 0)
            df[col] = df[col].fillna(val)

        # Tratamento de Gênero
        if 'GENERO' in df.columns:
            mapa_genero = {'Feminino': 1, 'Menina': 1, 'Masculino': 0, 'Menino': 0}
            if df['GENERO'].dtype == 'object':
                 df['GENERO'] = df['GENERO'].map(mapa_genero)
            df['GENERO'] = df['GENERO'].fillna(0)

        return df