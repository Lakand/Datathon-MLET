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
    """Transformer customizado para engenharia de features no pipeline Passos Mágicos.

    Realiza o pré-processamento focado em preparação para redes neurais,
    lidando com valores ausentes pela mediana e normalizando as variáveis.
    
    IMPORTANTE: Assume que os dados já foram limpos pelo DataPreprocessor,
    incluindo a conversão de GENERO de string para numérico (0/1).
    """

    def __init__(self):
        """Inicializa o FeatureEngineer.
        
        Define os atributos para armazenamento de estatísticas (medianas),
        o scaler interno e a lista de colunas selecionadas para treinamento.
        """
        self.medianas = {}
        self.scaler = StandardScaler()
        
        self.cols_treino = [
            'IDADE', 'GENERO', 'ANO_INGRESSO', 'FASE', 'DEFASAGEM',
            'NOTA_MAT', 'NOTA_PORT', 'NOTA_ING',
            'IEG', 'IPS', 'IAA', 'IPP'
        ]
        self.is_fitted = False

    def fit(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureEngineer':
        """Ajusta o transformador aos dados de treino.

        Calcula as medianas das colunas numéricas para imputação futura e
        ajusta o StandardScaler aos dados processados.

        Args:
            df (pd.DataFrame): DataFrame de treino (já limpo pelo DataPreprocessor).
            y (pd.Series, optional): Variável alvo (não utilizada no fit, mantida
                para compatibilidade com pipelines do sklearn).

        Returns:
            FeatureEngineer: A própria instância ajustada.
        """
        logger.info("Ajustando (Fitting) FeatureEngineer...")
        
        # Calcula medianas para todas as colunas numéricas
        # (GENERO já deve estar como numérico após preprocessing)
        cols_numericas = df.select_dtypes(include=[np.number]).columns
        
        for col in cols_numericas:
            self.medianas[col] = df[col].median()
        
        # Prepara os dados e ajusta o scaler
        df_temp = self._transform_logic(df, fit_mode=True)
        self.scaler.fit(df_temp[self.cols_treino])
        
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[pd.Series]]:
        """Transforma os dados aplicando as estatísticas aprendidas.

        Aplica imputação de nulos com as medianas calculadas no fit
        e normalização via StandardScaler.
        
        Também processa a variável alvo 'PEDRA' se estiver presente.

        Args:
            df (pd.DataFrame): DataFrame de entrada (já limpo pelo DataPreprocessor).

        Returns:
            Tuple[np.ndarray, Optional[pd.Series]]: Uma tupla contendo:
                - X_scaled (np.ndarray): Matriz de features normalizadas.
                - y (pd.Series ou None): Vetor alvo mapeado (se existir no input).

        Raises:
            RuntimeError: Se o método for chamado antes do fit().
        """
        if not self.is_fitted:
            logger.error("Tentativa de transformação sem fit prévio.")
            raise RuntimeError("FeatureEngineer não treinado!")
        
        # Aplica imputação e preparação
        df_proc = self._transform_logic(df, fit_mode=False)
        
        # Normaliza via StandardScaler
        X_scaled = self.scaler.transform(df_proc[self.cols_treino])
        
        # Processa target se existir
        y = None
        if 'PEDRA' in df_proc.columns:
            y = df_proc['PEDRA'].map(config.MAPA_PEDRA)
            
        return X_scaled, y

    def _transform_logic(self, df: pd.DataFrame, fit_mode: bool = False) -> pd.DataFrame:
        """Aplica a lógica de imputação de valores ausentes (método interno).
        
        NOTA: Este método NÃO faz conversão de tipos. Assume que:
        - GENERO já está como numérico (0/1) após DataPreprocessor
        - Todas as outras colunas já foram limpas e convertidas
        
        Responsabilidade deste método:
        - Criar colunas faltantes (se necessário)
        - Imputar valores nulos com medianas

        Args:
            df (pd.DataFrame): DataFrame a ser processado.
            fit_mode (bool): Flag indicativo (para futuras extensões).

        Returns:
            pd.DataFrame: DataFrame com valores nulos imputados.
        """
        df = df.copy()
        
        # Para cada coluna de treino, garantir que existe e imputar nulos
        for col in self.cols_treino:
            # Se a coluna não existir, criar com NaN
            if col not in df.columns:
                df[col] = np.nan
            
            # Buscar mediana (ou 0 se não tiver sido calculada)
            mediana = self.medianas.get(col, 0)
            
            # Imputar valores nulos
            df[col] = df[col].fillna(mediana)

        return df