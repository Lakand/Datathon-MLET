# src/feature_engineering.py
"""Módulo de Engenharia de Features.

Define classes transformadoras compatíveis com o Scikit-Learn para preparar
os dados para o modelo de Machine Learning, incluindo imputação de valores
nulos, codificação de variáveis categóricas e normalização (StandardScaler).
"""

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional, Tuple
from src import config 

logger = logging.getLogger(__name__)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Transformer customizado para o pipeline de dados Passos Mágicos.

    Realiza o pré-processamento focado em redes neurais (MLP), tratando 
    valores ausentes via imputação por mediana e normalização de escala.
    Assume que os dados já passaram pelo DataPreprocessor (limpeza bruta).
    """

    def __init__(self):
        """Inicializa o estado interno do transformador."""
        self.medianas = {}
        self.scaler = StandardScaler()
        
        # Atributos determinísticos para seleção de features
        self.cols_treino = [
            'IDADE', 'GENERO', 'ANO_INGRESSO', 'FASE', 'DEFASAGEM',
            'NOTA_MAT', 'NOTA_PORT', 'NOTA_ING',
            'IEG', 'IPS', 'IAA', 'IPP'
        ]
        self.is_fitted = False

    def fit(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureEngineer':
        """Ajusta o transformador calculando estatísticas de treino.

        Args:
            df: DataFrame de treino limpo.
            y: Variável alvo (mantido para compatibilidade com sklearn API).

        Returns:
            A própria instância do FeatureEngineer ajustada.
        """
        logger.info("Ajustando (Fitting) FeatureEngineer...")
        
        cols_numericas = df.select_dtypes(include=[np.number]).columns
        
        for col in cols_numericas:
            self.medianas[col] = df[col].median()
        
        df_temp = self._transform_logic(df, fit_mode=True)
        self.scaler.fit(df_temp[self.cols_treino])
        
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[pd.Series]]:
        """Aplica as transformações aprendidas nos dados de entrada.

        Args:
            df: DataFrame de entrada a ser transformado.

        Returns:
            Uma tupla contendo a matriz de features escalonadas (np.ndarray) 
            e a variável alvo mapeada (pd.Series), caso 'PEDRA' esteja presente.

        Raises:
            RuntimeError: Caso a transformação seja chamada antes do ajuste (fit).
        """
        if not self.is_fitted:
            logger.error("Tentativa de transformação sem fit prévio.")
            raise RuntimeError("FeatureEngineer não treinado!")
        
        df_proc = self._transform_logic(df, fit_mode=False)
        X_scaled = self.scaler.transform(df_proc[self.cols_treino])
        
        y = None
        if 'PEDRA' in df_proc.columns:
            y = df_proc['PEDRA'].map(config.MAPA_PEDRA)
            
        return X_scaled, y

    def _transform_logic(self, df: pd.DataFrame, fit_mode: bool = False) -> pd.DataFrame:
        """Executa a lógica interna de imputação e tratamento de colunas.

        Args:
            df: DataFrame original.
            fit_mode: Flag para controle de comportamento entre fit/transform.

        Returns:
            DataFrame processado com nulos imputados e colunas alinhadas.
        """
        df = df.copy()
        
        for col in self.cols_treino:
            # Garantia de integridade estrutural para colunas ausentes
            if col not in df.columns:
                df[col] = np.nan
            
            mediana = self.medianas.get(col, 0)
            df[col] = df[col].fillna(mediana)

        return df