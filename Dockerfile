# Usa uma imagem leve do Python
FROM python:3.10-slim

# Define o diretório de trabalho dentro do container
WORKDIR /app

# [NOVO] Adiciona o diretório atual ao caminho de busca do Python
# Isso substitui a necessidade de usar sys.path.append nos códigos
ENV PYTHONPATH=/app

# Instala dependências do sistema operacional (necessário para algumas libs numéricas)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copia o arquivo de requisitos
COPY requirements.txt .

# Instala as dependências do Python
# O --no-cache-dir deixa a imagem mais leve
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o código do projeto para dentro do container
COPY . .

# Cria as pastas necessárias para logs e modelos, caso não existam
RUN mkdir -p docs models data

# Expõe a porta 8000 (padrão do FastAPI)
EXPOSE 8000

# Comando para iniciar a API quando o container rodar
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]