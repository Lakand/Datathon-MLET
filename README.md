# API Passos Mágicos - Previsão de Risco de Defasagem Escolar

## 📋 Visão Geral do Projeto

### Objetivo

Este projeto foi desenvolvido para o programa **Passos Mágicos**, uma organização que trabalha com alunos em situação de vulnerabilidade social. O objetivo principal é **prever o risco de defasagem escolar** dos alunos, classificando-os em quatro categorias (representadas por nomes de pedras):

- **Quartzo** - Sem risco aparente
- **Ágata** - Risco baixo
- **Ametista** - Risco médio  
- **Topázio** - Risco alto

A defasagem escolar é um indicador crítico do desempenho acadêmico e do engajamento dos alunos no programa, permitindo intervenções preventivas direcionadas.

### Solução Proposta

A solução implementa uma **pipeline completa de Machine Learning** em produção:

1. **Coleta e Limpeza de Dados**: Unificação de dados de múltiplas safras (2022-2024) com esquemas heterogêneos
2. **Engenharia de Features**: Transformação e normalização automática de variáveis
3. **Treinamento do Modelo**: Classificador neural (MLPClassifier) com validação cruzada estratificada
4. **Deploy em API**: FastAPI com endpoints RESTful para predição, treinamento e monitoramento
5. **Monitoramento de Drift**: Detecção automática de mudanças na distribuição dos dados em produção
6. **Registro de Experimentos**: Rastreamento de modelos e métricas com MLflow

### Stack Tecnológica

| Componente | Tecnologias |
|-----------|------------|
| **API & Deploy** | FastAPI, Uvicorn, Python 3.10 |
| **Machine Learning** | Scikit-Learn, XGBoost, Imbalanced-Learn (SMOTE) |
| **Dados** | Pandas, NumPy, Openpyxl |
| **MLOps** | MLflow (rastreamento de experimentos), Evidently (drift detection) |
| **Containerização** | Docker, Docker Compose |
| **Testes** | Pytest, Pytest-Cov |
| **Visualização** | Matplotlib, Seaborn |

---

## 📁 Estrutura do Projeto

```
.
├── app/                           # Aplicação FastAPI
│   ├── __init__.py
│   ├── main.py                    # Ponto de entrada da API
│   ├── routes.py                  # Definição dos endpoints
│   ├── schemas.py                 # Modelos Pydantic (validação)
│   └── monitor.py                 # Logging e monitoramento
│
├── src/                           # Lógica de ML e processamento
│   ├── config.py                  # Configurações centralizadas
│   ├── preprocessing.py           # Limpeza e padronização de dados
│   ├── feature_engineering.py     # Transformação de features
│   ├── train.py                   # Pipeline de treinamento
│   ├── evaluate.py                # Avaliação do modelo
│   ├── drift_report.py            # Geração de relatórios de drift
│   └── utils.py                   # Funções auxiliares
│
├── tests/                         # Suite de testes
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_drift_report.py
│   ├── test_feature_engineering.py
│   ├── test_preprocessing.py
│   └── test_utils.py
│
├── data/                          # Dados e base de dados
│   ├── BASE DE DADOS PEDE 2024 - DATATHON.xlsx
│   └── test_dataset.csv
│
├── models/                        # Artefatos de ML
│   ├── mlp_model.joblib           # Modelo treinado
│   └── pipeline_features.joblib   # Pipeline de transformação
│
├── docs/                          # Documentação e relatórios
│   └── drift_report.html
│
├── notebooks/                     # Análise exploratória
│   └── exploracao_dados.ipynb
│
├── mlruns/                        # Artefatos e histórico do MLflow
│
├── requirements.txt               # Dependências Python
├── Dockerfile                     # Imagem Docker da aplicação
├── docker-compose.yml             # Orquestração de containers
├── pytest.ini                     # Configuração de testes
└── README.md                      # Este arquivo
```

---

## 🚀 Instruções de Deploy

### Pré-requisitos

- **Docker** 20.10+
- **Docker Compose** 1.29+ (recomendado)
- **Python** 3.10+ (para desenvolvimento local)
- **Pip** ou **Conda**

### Instalação de Dependências (Desenvolvimento Local)

#### Via Pip

```bash
# Clonar o repositório ou navegar até a pasta do projeto
cd "Python/Fase 5"

# Criar um ambiente virtual (opcional, mas recomendado)
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt
```

#### Via Conda (Alternativa)

```bash
conda create -n passos-magicos python=3.10
conda activate passos-magicos
pip install -r requirements.txt
```

---

### Deploy com Docker Compose (Recomendado)

Este é o método mais simples para executar a aplicação em produção.

#### 1. Build e Execução

```bash
# Navegar até a pasta do projeto
cd "Python/Fase 5"

# Build da imagem Docker
docker-compose build

# Iniciar a aplicação
docker-compose up -d

# Verificar logs em tempo real
docker-compose logs -f api-passos-magicos
```

#### 2. Parar a Aplicação

```bash
docker-compose down
```

#### 3. Estrutura do Docker Compose

A aplicação será disponibilizada em **http://127.0.0.1:8000** com os seguintes volumes montados:

- `./data:/app/data` - Base de dados e arquivos de entrada
- `./models:/app/models` - Artefatos de ML (modelos treinados)
- `./docs:/app/docs` - Relatórios gerados
- `./mlruns:/app/mlruns` - Histórico de experimentos (MLflow)

---

### Deploy com Docker (Alternativa Manual)

#### 1. Build da Imagem

```bash
docker build -t passos-magicos-api:latest .
```

#### 2. Execução do Container

```bash
docker run -d \
  --name passos-magicos-api \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/docs:/app/docs \
  -v $(pwd)/mlruns:/app/mlruns \
  -e PYTHONUNBUFFERED=1 \
  -e DB_PATH=/app/data/monitoring.db \
  passos-magicos-api:latest
```

#### 3. Acessar a API

- **URL Base**: http://127.0.0.1:8000
- **Documentação Interativa (Swagger UI)**: http://127.0.0.1:8000/docs
- **Documentação Alternativa (ReDoc)**: http://127.0.0.1:8000/redoc

---

## 🔌 Exemplos de Chamadas à API

### 1. Predição de Risco (POST /predict)

Realiza predição de risco de defasagem para um ou mais alunos.

#### Usando cURL

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '[
    {
      "RA": "123456",
      "IDADE": 14,
      "GENERO": "Menino",
      "ANO_INGRESSO": 2022,
      "FASE": 1,
      "NOTA_MAT": 8.5,
      "NOTA_PORT": 7.0,
      "NOTA_ING": 6.5,
      "IEG": 7.0,
      "IPS": 6.5,
      "IAA": 8.0,
      "IPP": 7.5,
      "DEFASAGEM": 0
    }
  ]'
```

#### Usando Python + Requests

```python
import requests
import json

url = "http://127.0.0.1:8000/predict"

payload = [
    {
        "RA": "123456",
        "IDADE": 14,
        "GENERO": "Menino",
        "ANO_INGRESSO": 2022,
        "FASE": 1,
        "NOTA_MAT": 8.5,
        "NOTA_PORT": 7.0,
        "NOTA_ING": 6.5,
        "IEG": 7.0,
        "IPS": 6.5,
        "IAA": 8.0,
        "IPP": 7.5,
        "DEFASAGEM": 0
    }
]

response = requests.post(url, json=payload)
print(json.dumps(response.json(), indent=2, ensure_ascii=False))
```

**Resposta Esperada:**

```json
{
  "predictions": [
    {
      "RA": "123456",
      "PEDRA_PREVISTA": "Quartzo",
      "RISCO_DEFASAGEM": "Baixo",
      "CONFIANCA": 0.92
    }
  ]
}
```

#### Usando Postman

1. Abrir Postman
2. Criar uma nova requisição **POST**
3. **URL**: `http://127.0.0.1:8000/predict`
4. **Headers**: `Content-Type: application/json`
5. **Body** (raw, JSON):
```json
[
  {
    "RA": "123456",
    "IDADE": 14,
    "GENERO": "Menino",
    "ANO_INGRESSO": 2022,
    "FASE": 1,
    "NOTA_MAT": 8.5,
    "NOTA_PORT": 7.0,
    "NOTA_ING": 6.5,
    "IEG": 7.0,
    "IPS": 6.5,
    "IAA": 8.0,
    "IPP": 7.5,
    "DEFASAGEM": 0
  }
]
```
6. Clicar **Send**

---

### 2. Treinamento do Modelo (POST /train)

Treina um novo modelo usando todos os dados disponíveis.

#### Usando cURL

```bash
curl -X POST "http://127.0.0.1:8000/train"
```

#### Usando Python

```python
import requests

url = "http://127.0.0.1:8000/train"
response = requests.post(url)
print(response.json())
```

**Resposta Esperada:**

```json
{
  "message": "Treinamento concluído e modelo atualizado!",
  "details": {
    "accuracy": 0.85,
    "f1_weighted": 0.84,
    "training_time_seconds": 45.3,
    "data_size": 500
  }
}
```

---

### 3. Avaliação do Modelo (GET /evaluate)

Avalia a performance do modelo contra a base de teste (holdout).

#### Usando cURL

```bash
curl -X GET "http://127.0.0.1:8000/evaluate"
```

#### Usando Python

```python
import requests
import json

url = "http://127.0.0.1:8000/evaluate"
response = requests.get(url)
print(json.dumps(response.json(), indent=2, ensure_ascii=False))
```

**Resposta Esperada:**

```json
{
  "accuracy": 0.82,
  "f1_weighted": 0.81,
  "classification_report": {
    "Quartzo": {
      "precision": 0.88,
      "recall": 0.80,
      "f1-score": 0.84
    },
    "Ágata": {
      "precision": 0.78,
      "recall": 0.75,
      "f1-score": 0.77
    }
  }
}
```

---

### 4. Relatório de Data Drift (GET /drift-report)

Gera um relatório HTML comparando a distribuição dos dados de treinamento com dados de produção.

#### Usando cURL

```bash
curl -X GET "http://127.0.0.1:8000/drift-report" -o drift_report.html
```

#### Usando Python

```python
import requests

url = "http://127.0.0.1:8000/drift-report"
response = requests.get(url)

with open("drift_report.html", "wb") as f:
    f.write(response.content)

print("Relatório salvo em: drift_report.html")
```

---

## 🔄 Etapas do Pipeline de Machine Learning

### Diagrama do Pipeline

```
┌─────────────────────────────────┐
│   Dados Brutos (2022-2024)      │
│     diferentes schemas          │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│   1. PRÉ-PROCESSAMENTO          │
│   ├─ Carregamento de dados      │
│   ├─ Identificação de schema    │
│   ├─ Renomeação de colunas      │
│   ├─ Limpeza de tipos           │
│   └─ Tratamento de valores nulos│
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│   2. ENGENHARIA DE FEATURES     │
│   ├─ Seleção de features        │
│   ├─ Imputação (mediana)        │
│   ├─ Codificação (categorical)  │
│   └─ Normalização (StandardSca) │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│   3. BALANCEAMENTO DE DADOS     │
│   ├─ Detecção de desbalanceio   │
│   └─ SMOTE (oversampling)       │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│   4. DIVISÃO DE DADOS           │
│   ├─ Train: 80%                 │
│   ├─ Test: 20%                  │
│   └─ Validação: K-Fold (5)      │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│   5. TREINAMENTO DO MODELO      │
│   ├─ MLPClassifier (NN)         │
│   ├─ Hidden layers: (100,)      │
│   ├─ Max iterations: 2000       │
│   └─ Rastreamento (MLflow)      │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│   6. AVALIAÇÃO DO MODELO        │
│   ├─ Accuracy                   │
│   ├─ F1-Score (weighted)        │
│   ├─ Precision & Recall         │
│   └─ Matriz de Confusão         │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│   7. PERSISTÊNCIA               │
│   ├─ Salvar modelo              │
│   ├─ Salvar pipeline            │
│   └─ Registrar artefatos        │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│   Modelo em Produção (API)      │
│     Predições em tempo real     │
│     Monitoramento de Drift      │
└─────────────────────────────────┘
```

### Descrição Detalhada das Etapas

#### **1. Pré-Processamento** (`src/preprocessing.py`)

Unifica e limpa dados de múltiplas safras com esquemas heterogêneos:

- **Identificação automática de schema**: Detecta se os dados são de 2022, 2023 ou 2024
- **Renomeação de colunas**: Padroniza nomenclatura variável entre anos
- **Limpeza de tipos**:
  - Converte idades incorretas (ex: "1900-01-15" → 15)
  - Normaliza gênero ("Menino" → "Masculino")
  - Corrige ortografia ("Agata" → "Ágata")
- **Tratamento de valores ausentes**: Flag de detecção para imputação posterior

**Exemplos de mapeamento (2022 → 2024):**

```python
{
  "Matem" → "NOTA_MAT",
  "Portug" → "NOTA_PORT",
  "Inglês" → "NOTA_ING",
  "Pedra 22" → "PEDRA",
  "Idade 22" → "IDADE"
}
```

#### **2. Engenharia de Features** (`src/feature_engineering.py`)

Transforma dados brutos em features otimizadas para o modelo neural:

- **Seleção de features**:
  ```python
  ['IDADE', 'GENERO', 'ANO_INGRESSO', 'FASE', 'DEFASAGEM',
   'NOTA_MAT', 'NOTA_PORT', 'NOTA_ING', 'IEG', 'IPS', 'IAA', 'IPP']
  ```

- **Imputação**: Valores nulos preenchidos com mediana das variáveis numéricas
- **Codificação categórica**: Gênero é mapeado para valores numéricos
- **Normalização (StandardScaler)**: Centrado em média 0, desvio padrão 1
  - **Fórmula**: z = (x - μ) / σ

#### **3. Balanceamento de Dados**

Classes desbalanceadas são tratadas com **SMOTE** (Synthetic Minority Oversampling):

- Gera amostras sintéticas para classes minoritárias
- Mantém distribuição realística dos dados

#### **4. Divisão de Dados**

- **Train**: 80% dados (para ajuste do modelo)
- **Test**: 20% dados (reserved para avaliação final)
- **Validação Cruzada**: 5-fold estratificada por grupo (RA do aluno)

#### **5. Treinamento do Modelo** (`src/train.py`)

Treina um **MLPClassifier** (Rede Neural Multicamadas):

```python
MODEL_PARAMS = {
    'hidden_layer_sizes': (100,),  # Uma camada oculta com 100 neurônios
    'activation': 'relu',          # ReLU como função de ativação
    'alpha': 0.01,                 # Regularização L2
    'learning_rate_init': 0.001,   # Taxa de aprendizado inicial
    'max_iter': 2000,              # Máximo de iterações
    'random_state': 42             # Reprodutibilidade
}
```

- **Validação Cruzada**: Avalia desempenho em 5 folds
- **Rastreamento MLflow**: Registra métricas, parâmetros e artefatos

#### **6. Avaliação do Modelo** (`src/evaluate.py`)

Calcula métricas de desempenho no conjunto de teste:

- **Accuracy**: Proporção de previsões corretas
- **Precision**: Proporção de positivos corretamente identificados
- **Recall**: Proporção real de positivos identificados
- **F1-Score**: Média harmônica entre Precision e Recall
- **Matriz de Confusão**: Visualização de erros por classe

#### **7. Monitoramento de Drift** (`src/drift_report.py`)

Detecta mudanças na distribuição dos dados entre treino e produção:

- **Uso da biblioteca Evidently**: Compara distribuições estatísticas
- **Relatório HTML**: Visualizações interativas dos desvios
- **Acionamento automático**: Alertas quando drift é detectado

---

## 📝 Comandos Úteis

### Desenvolvimento & Testes

#### Executar Testes Locais

```bash
# Rodar todos os testes
pytest

# Rodar com cobertura de código
pytest --cov=src --cov=app

# Rodar teste específico
pytest tests/test_api.py -v

# Rodar teste com output detalhado
pytest -vv -s
```

#### Treinar Modelo Localmente

```bash
# Ativar ambiente virtual (se necessário)
source venv/bin/activate  # No Windows: venv\Scripts\activate

# Executar pipeline de treinamento
python -m src.train
```

#### Avaliar Modelo

```bash
python -m src.evaluate
```

#### Explorar Dados

```bash
# Abrir notebook Jupyter
jupyter notebook notebooks/exploracao_dados.ipynb
```

### Desenvolvimento da API

#### Rodando a API Localmente

```bash
# Ambiente virtual ativado
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Acesse: http://127.0.0.1:8000/docs

#### Acessar MLflow UI

```bash
# Na raiz do projeto
mlflow ui

# Acesse em http://127.0.0.1:5000
```

### Gerenciamento de Docker

```bash
# Verificar containers em execução
docker ps -a

# Ver logs de um container
docker logs -f api-passos-magicos

# Acessar shell do container
docker exec -it api-passos-magicos bash

# Parar um container
docker stop api-passos-magicos

# Remover um container
docker rm api-passos-magicos

# Remover imagem Docker
docker rmi passos-magicos-api:latest

# Limpar recursos não utilizados
docker system prune -a
```

### Troubleshooting

#### API não inicia após container estar rodando

```bash
# Verificar logs
docker-compose logs api-passos-magicos

# Reiniciar serviço
docker-compose restart api-passos-magicos
```

#### Modelo não está carregado

```bash
# Treinar novo modelo via API
curl -X POST "http://127.0.0.1:8000/train"

# Ou via linha de comando
python -m src.train
```

#### Erro de permissão em volumes Linux

```bash
# Ajustar permissões
sudo chown -R $USER:$USER data/ models/ docs/ mlruns/
```

---

## 📊 Features e Descrição de Entrada

### Variáveis de Entrada

| Variável | Tipo | Descrição | Exemplo |
|----------|------|-----------|---------|
| RA | string | Registro Acadêmico (ID único do aluno) | "123456" |
| IDADE | float | Idade do aluno em anos | 14.5 |
| GENERO | enum | Gênero (Masculino/Feminino/Menino/Menina) | "Menino" |
| ANO_INGRESSO | int | Ano de entrada no programa | 2022 |
| FASE | int | Fase atual do aluno | 1-8 |
| NOTA_MAT | float | Nota em Matemática (0-10) | 8.5 |
| NOTA_PORT | float | Nota em Português (0-10) | 7.0 |
| NOTA_ING | float | Nota em Inglês (0-10) | 6.5 |
| IEG | float | Índice de Engajamento Global (0-10) | 7.0 |
| IPS | float | Índice Psicossocial (0-10) | 6.5 |
| IAA | float | Índice de Autoavaliação (0-10) | 8.0 |
| IPP | float | Índice Psicopedagógico (0-10) | 7.5 |
| DEFASAGEM | int | Nível de defasagem escolar (0, 1, 2...) | 0 |

### Variável Alvo (Output)

| Pedra | Classificação |
|-------|---------------|
| **Quartzo** | Sem risco aparente |
| **Ágata** | Risco baixo |
| **Ametista** | Risco médio |
| **Topázio** | Risco alto |

---

## 🛠️ Contribuições e Desenvolvimento

### Adicionar Novas Features

1. Atualizar dados de entrada em `app/schemas.py`
2. Adicionar lógica de transformação em `src/feature_engineering.py`
3. Atualizar testes em `tests/test_feature_engineering.py`
4. Retreinar modelo via `/train`

### Melhorar o Modelo

1. Ajustar hiperparâmetros em `src/config.py`
2. Testar diferentes algoritmos em um notebook
3. Registrar experimentos no MLflow
4. Comparar métricas e selecionar o melhor

---

## 📞 Suporte e Documentação

- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc
- **MLflow Experiments**: http://127.0.0.1:5000 (após `mlflow ui`)

---

## 📄 Licença

Este projeto foi desenvolvido para o programa **Passos Mágicos**.

---

**Última atualização**: 13 de fevereiro de 2026
