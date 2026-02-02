import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.medianas = {}
        self.scaler = StandardScaler()
        
        # --- ESTRATÉGIA MEIO TERMO ---
        # Apenas variáveis de contexto, notas brutas e soft skills.
        # Removemos: INDE, PEDRA, IPV (Ponto de Virada) e IAN.
        self.cols_treino = [
            'IDADE', 'GENERO', 'ANO_INGRESSO', 'FASE', 'DEFASAGEM',  # Contexto
            'NOTA_MAT', 'NOTA_PORT', 'NOTA_ING',                     # Acadêmico (Hard Skills)
            'IEG', 'IPS', 'IAA', 'IPP'                               # Comportamental (Soft Skills)
        ]
        self.is_fitted = False

    def fit(self, df, y=None):
        # Calcula medianas para TODAS as colunas numéricas possíveis
        # (Isso evita erros se mudar a lista de features no futuro)
        cols_numericas = df.select_dtypes(include=[np.number]).columns
        
        for col in cols_numericas:
            self.medianas[col] = df[col].median()
            
        # Ajusta o Scaler nos dados processados
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
            # Remove linhas onde PEDRA é NaN se necessário, ou mantém alinhado
            y = df_proc['PEDRA'].map(mapa_pedra)
            
        return X_scaled, y

    def _transform_logic(self, df, fit_mode=False):
        df = df.copy()
        
        # Preenchimento de Nulos (Imputação)
        for col in self.cols_treino:
            if col == 'GENERO': continue # Tratado separadamente
            
            if col not in df.columns:
                df[col] = np.nan
            
            # Se for fit_mode, usa a mediana calculada agora (já feito no fit principal),
            # mas aqui aplicamos o fillna com o valor guardado.
            val = self.medianas.get(col, 0)
            df[col] = df[col].fillna(val)

        # Tratamento de Gênero
        if 'GENERO' in df.columns:
            mapa_genero = {'Feminino': 1, 'Menina': 1, 'Masculino': 0, 'Menino': 0}
            # Se já for numérico, mantém, se for texto, mapeia
            if df['GENERO'].dtype == 'object':
                 df['GENERO'] = df['GENERO'].map(mapa_genero)
            df['GENERO'] = df['GENERO'].fillna(0)

        return df